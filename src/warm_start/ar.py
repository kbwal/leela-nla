#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast

os.environ["WANDB_PROJECT"] = "lc0-nla"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "data/pretrain/shard-combined_teacher_tp4_encoder14_ln2_betas.jsonl"
DEFAULT_MODEL = "Qwen/Qwen3.5-2B"
DEFAULT_CACHE = "/scratch/hub"
DEFAULT_ACTIVATION_DIM = 512
DEFAULT_PREFIX_LEN = 64
DEFAULT_RECONSTRUCTOR_HIDDEN_SIZE = 8192


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm-start the activation reconstructor (AR): teacher summary -> activation."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/warm_start/qwen3.5-2b-nla-ar",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--activation-dim", type=int, default=DEFAULT_ACTIVATION_DIM)
    parser.add_argument("--prefix-len", type=int, default=DEFAULT_PREFIX_LEN)
    parser.add_argument(
        "--reconstructor-hidden-size",
        type=int,
        default=DEFAULT_RECONSTRUCTOR_HIDDEN_SIZE,
        help="Hidden size for the AR MLP head.",
    )
    parser.add_argument("--max-summary-length", type=int, default=512)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--ar-cosine-weight", type=float, default=0.25)
    parser.add_argument(
        "--reconstruction-loss",
        choices=["mse", "cosine", "mse_cosine"],
        default="mse_cosine",
    )
    parser.add_argument("--per-device-train-batch-size", type=int, default=1)
    parser.add_argument("--per-device-eval-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--logging-steps", type=int, default=10)
    parser.add_argument("--eval-steps", type=int, default=500)
    parser.add_argument("--save-steps", type=int, default=500)
    parser.add_argument("--save-total-limit", type=int, default=3)
    parser.add_argument("--bf16", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument(
        "--fsdp",
        default=None,
        help='Optional Trainer FSDP setting, e.g. "full_shard auto_wrap".',
    )
    parser.add_argument(
        "--fsdp-transformer-layer-cls-to-wrap",
        default="Qwen3_5DecoderLayer",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


class ActivationReconstructor(nn.Module):
    """Map a summary final-token hidden state back to a Leela activation prefix."""

    def __init__(
        self,
        hidden_size: int,
        activation_dim: int,
        prefix_len: int,
        reconstructor_hidden_size: int | None = None,
    ) -> None:
        super().__init__()
        self.prefix_len = prefix_len
        self.activation_dim = activation_dim
        self.hidden_size = hidden_size
        self.reconstructor_hidden_size = reconstructor_hidden_size or hidden_size
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, self.reconstructor_hidden_size, bias=True),
            nn.GELU(),
            nn.Linear(
                self.reconstructor_hidden_size,
                prefix_len * activation_dim,
                bias=True,
            ),
        )

    def forward(self, final_summary_hidden: torch.Tensor) -> torch.Tensor:
        flat = self.mlp(final_summary_hidden)
        return flat.view(-1, self.prefix_len, self.activation_dim)


class ActivationReconstructorWarmStart(nn.Module):
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        *,
        activation_dim: int,
        prefix_len: int,
        reconstructor_hidden_size: int | None,
        ar_cosine_weight: float,
        reconstruction_loss: Literal["mse", "cosine", "mse_cosine"],
    ) -> None:
        super().__init__()
        self.model = base_model
        self.prefix_len = prefix_len
        self.ar_cosine_weight = ar_cosine_weight
        self.reconstruction_loss = reconstruction_loss
        self.config = base_model.config

        hidden_size = base_model.config.hidden_size
        self.ar_reconstructor = ActivationReconstructor(
            hidden_size, activation_dim, prefix_len, reconstructor_hidden_size
        )
        self.ar_reconstructor.to(next(base_model.parameters()).dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def _summary_mask(self, input_ids: torch.Tensor) -> torch.Tensor:
        return (input_ids != self.config.pad_token_id).long()

    def reconstruction_loss_for(
        self,
        activation: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if attention_mask is None:
            attention_mask = self._summary_mask(input_ids)

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = outputs.hidden_states[-1]
        last_token_index = attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(input_ids.shape[0], device=input_ids.device)
        final_summary_hidden = final_hidden[batch_index, last_token_index]
        reconstructed = self.ar_reconstructor(final_summary_hidden)

        target = activation.to(dtype=reconstructed.dtype)
        flat_reconstructed = reconstructed.flatten(start_dim=1)
        flat_target = target.flatten(start_dim=1)
        mse = F.mse_loss(reconstructed, target)
        cosine = (
            1.0 - F.cosine_similarity(flat_reconstructed, flat_target, dim=1).mean()
        )

        if self.reconstruction_loss == "mse":
            loss = mse
        elif self.reconstruction_loss == "cosine":
            loss = cosine
        else:
            loss = mse + self.ar_cosine_weight * cosine
        return loss, reconstructed

    def forward(
        self,
        activation: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        loss, _ = self.reconstruction_loss_for(
            activation=activation,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        return CausalLMOutputWithPast(loss=loss)


@dataclass(frozen=True)
class DistillRow:
    activation_path: str
    teacher_summary: str
    record_id: str


class ActivationReconstructorDataset(Dataset):
    def __init__(
        self, rows: list[DistillRow], tokenizer: Any, max_summary_length: int
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.max_summary_length = max_summary_length

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        activation = torch.load(
            row.activation_path, map_location="cpu", weights_only=False
        ).float()
        if activation.ndim != 2:
            raise ValueError(
                f"{row.record_id}: expected 2D activation, got {tuple(activation.shape)}"
            )

        encoded = self.tokenizer(
            row.teacher_summary,
            truncation=True,
            max_length=self.max_summary_length,
            padding=False,
            add_special_tokens=True,
        )
        input_ids = encoded["input_ids"]
        if len(input_ids) < 1:
            raise ValueError(f"{row.record_id}: empty teacher_summary")

        return {
            "activation": activation,
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(encoded["attention_mask"], dtype=torch.long),
            "record_id": row.record_id,
        }


@dataclass
class ActivationReconstructorCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        activations = torch.stack([feature["activation"] for feature in features])
        max_len = max(feature["input_ids"].shape[0] for feature in features)

        input_ids: list[torch.Tensor] = []
        attention_mask: list[torch.Tensor] = []
        for feature in features:
            pad_size = max_len - feature["input_ids"].shape[0]
            input_ids.append(
                nn.functional.pad(
                    feature["input_ids"], (0, pad_size), value=self.pad_token_id
                )
            )
            attention_mask.append(
                nn.functional.pad(feature["attention_mask"], (0, pad_size), value=0)
            )

        return {
            "activation": activations,
            "input_ids": torch.stack(input_ids),
            "attention_mask": torch.stack(attention_mask),
        }


class ARTrainer(Trainer):
    def __init__(self, *args: Any, head_learning_rate: float, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.head_learning_rate = head_learning_rate

    def create_optimizer(self, optimizer_cls_and_kwargs=None):
        if self.optimizer is None:
            base_params = []
            head_params = []
            for name, param in self.model.named_parameters():
                if not param.requires_grad:
                    continue
                if "ar_reconstructor" in name:
                    head_params.append(param)
                else:
                    base_params.append(param)
            self.optimizer = torch.optim.AdamW(
                [
                    {"params": base_params, "lr": self.args.learning_rate},
                    {"params": head_params, "lr": self.head_learning_rate},
                ],
                weight_decay=self.args.weight_decay,
            )
        return self.optimizer

    def save_model(self, output_dir: str | None = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        state_dict = self.accelerator.get_state_dict(self.model)
        if self.args.should_save:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            unwrapped = self.accelerator.unwrap_model(self.model)

            base_model_sd = {
                key[6:]: value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
            unwrapped.model.save_pretrained(output_path, state_dict=base_model_sd)

            ar_sd = {
                key[17:]: value
                for key, value in state_dict.items()
                if key.startswith("ar_reconstructor.")
            }
            torch.save(
                {
                    "ar_reconstructor_state_dict": ar_sd,
                    "activation_dim": unwrapped.ar_reconstructor.activation_dim,
                    "hidden_size": unwrapped.ar_reconstructor.hidden_size,
                    "reconstructor_hidden_size": unwrapped.ar_reconstructor.reconstructor_hidden_size,
                    "prefix_len": unwrapped.prefix_len,
                    "ar_cosine_weight": unwrapped.ar_cosine_weight,
                    "reconstruction_loss": unwrapped.reconstruction_loss,
                },
                output_path / "ar_head.pt",
            )


def has_training_fields(record: dict[str, Any]) -> bool:
    if not str(record.get("teacher_summary", "")).strip():
        return False
    activation = record.get("activation") or {}
    return bool(activation.get("path"))


def build_rows_by_split(
    args: argparse.Namespace,
) -> tuple[list[DistillRow], list[DistillRow]]:
    train_rows: list[DistillRow] = []
    eval_rows: list[DistillRow] = []
    with args.data.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if args.limit is not None and line_no >= args.limit:
                break
            if not line.strip():
                continue
            record = json.loads(line)
            if not has_training_fields(record):
                continue
            row = DistillRow(
                activation_path=record["activation"]["path"],
                teacher_summary=record["teacher_summary"],
                record_id=record.get("id", f"row-{line_no}"),
            )
            split = str(record.get("split", "train"))
            if split == args.train_split:
                train_rows.append(row)
            elif split == args.eval_split:
                eval_rows.append(row)
    return train_rows, eval_rows


def print_trainable_parameters(model: nn.Module) -> None:
    trainable = sum(param.numel() for param in model.parameters() if param.requires_grad)
    total = sum(param.numel() for param in model.parameters())
    pct = 100 * trainable / total
    print(f"Trainable parameters: {trainable:,} / {total:,} ({pct:.2f}%)", flush=True)


def build_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[ActivationReconstructorWarmStart, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        device_map={"": local_rank},
        trust_remote_code=True,
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing and not args.fsdp:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.config.use_cache = False

    model = ActivationReconstructorWarmStart(
        base_model,
        activation_dim=args.activation_dim,
        prefix_len=args.prefix_len,
        reconstructor_hidden_size=args.reconstructor_hidden_size,
        ar_cosine_weight=args.ar_cosine_weight,
        reconstruction_loss=args.reconstruction_loss,
    )
    print_trainable_parameters(model)
    return model, tokenizer


def build_training_args(
    args: argparse.Namespace, eval_dataset: Dataset | None
) -> TrainingArguments:
    fsdp_config = None
    if args.fsdp:
        fsdp_config = {
            "transformer_layer_cls_to_wrap": args.fsdp_transformer_layer_cls_to_wrap,
            "activation_checkpointing": args.gradient_checkpointing,
        }

    return TrainingArguments(
        output_dir=str(args.output_dir),
        seed=args.seed,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        logging_steps=args.logging_steps,
        eval_strategy="steps" if eval_dataset is not None else "no",
        eval_steps=args.eval_steps if eval_dataset is not None else None,
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        bf16=args.bf16,
        remove_unused_columns=False,
        label_names=["activation"],
        gradient_checkpointing=False if args.fsdp else args.gradient_checkpointing,
        optim="adamw_torch",
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name="arinitialization1",
    )


def main() -> None:
    if "LOCAL_RANK" in os.environ:
        torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")

    torch.manual_seed(args.seed)
    model, tokenizer = build_model_and_tokenizer(args)
    train_rows, eval_rows = build_rows_by_split(args)
    if not train_rows:
        raise SystemExit("No training rows found.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_rows = train_rows.copy()
    train_indices = torch.randperm(len(train_rows), generator=generator).tolist()
    train_rows = [train_rows[index] for index in train_indices]

    train_dataset = ActivationReconstructorDataset(
        train_rows, tokenizer, args.max_summary_length
    )
    eval_dataset = (
        ActivationReconstructorDataset(eval_rows, tokenizer, args.max_summary_length)
        if eval_rows
        else None
    )

    print(
        f"Loaded AR train={len(train_dataset)} eval={0 if eval_dataset is None else len(eval_dataset)} "
        f"activation_prefix=({args.prefix_len}, {args.activation_dim})",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = ARTrainer(
        model=model,
        args=build_training_args(args, eval_dataset),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=ActivationReconstructorCollator(
            pad_token_id=tokenizer.pad_token_id
        ),
        head_learning_rate=args.head_learning_rate,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.args.should_save:
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
