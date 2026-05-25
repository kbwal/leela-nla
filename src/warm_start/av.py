#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, Trainer, TrainingArguments
from transformers.modeling_outputs import CausalLMOutputWithPast

os.environ["WANDB_PROJECT"] = "lc0-nla"
ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = (
    ROOT / "data/pretrain/shard-combined_teacher_tp4_encoder14_ln2_betas.jsonl"
)
DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_CACHE = "/scratch/hub"
DEFAULT_ACTIVATION_DIM = 512
DEFAULT_PREFIX_LEN = 64
ACTIVATION_TOKEN = "<activation/>"
DEFAULT_PROMPT = f"Explain the ideas in this chess position: {ACTIVATION_TOKEN}\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Warm-start the activation verbalizer (AV): activation -> teacher summary."
        )
    )
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--cache-dir", default=DEFAULT_CACHE)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "outputs/warm_start/qwen3-4b-nla-av",
    )
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--activation-dim", type=int, default=DEFAULT_ACTIVATION_DIM)
    parser.add_argument("--prefix-len", type=int, default=DEFAULT_PREFIX_LEN)
    parser.add_argument("--max-summary-length", type=int, default=512)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--learning-rate", type=float, default=1e-5)
    parser.add_argument("--head-learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
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
        default="Qwen3DecoderLayer",
    )
    parser.add_argument("--resume-from-checkpoint", default=None)
    return parser.parse_args()


class ActivationVerbalizerProjector(nn.Module):
    """Map Leela activation vectors into the student LM hidden size."""

    def __init__(self, activation_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(activation_dim, hidden_size)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.proj(activation)


class ActivationVerbalizerWarmStart(nn.Module):
    def __init__(
        self,
        base_model: AutoModelForCausalLM,
        *,
        activation_dim: int,
        prefix_len: int,
        activation_token_id: int,
    ) -> None:
        super().__init__()
        self.model = base_model
        self.prefix_len = prefix_len
        self.activation_token_id = activation_token_id
        self.config = base_model.config

        hidden_size = base_model.config.hidden_size
        self.av_projector = ActivationVerbalizerProjector(activation_dim, hidden_size)
        self.av_projector.to(next(base_model.parameters()).dtype)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    def activation_embeds(self, activation: torch.Tensor) -> torch.Tensor:
        if activation.ndim != 3:
            raise ValueError(
                f"Expected activation [B, T, D], got {tuple(activation.shape)}"
            )
        if activation.shape[1] != self.prefix_len:
            raise ValueError(
                f"Expected activation prefix length {self.prefix_len}, got {activation.shape[1]}"
            )
        dtype = self.av_projector.proj.weight.dtype
        return self.av_projector(activation.to(dtype=dtype))

    def _expand_activation_token(
        self,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        activation: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        token_embedding = self.model.get_input_embeddings()
        activation_embeds = self.activation_embeds(activation)

        expanded_embeds: list[torch.Tensor] = []
        expanded_masks: list[torch.Tensor] = []
        for batch_idx in range(prompt_input_ids.shape[0]):
            token_positions = (
                prompt_input_ids[batch_idx] == self.activation_token_id
            ).nonzero(as_tuple=False)
            if token_positions.shape[0] != 1:
                raise ValueError(
                    f"Expected exactly one {ACTIVATION_TOKEN} in every prompt, "
                    f"got {token_positions.shape[0]}"
                )
            activation_pos = token_positions.item()
            before_ids = prompt_input_ids[batch_idx, :activation_pos]
            after_ids = prompt_input_ids[batch_idx, activation_pos + 1 :]

            before_embeds = token_embedding(before_ids)
            after_embeds = token_embedding(after_ids)
            expanded_embeds.append(
                torch.cat(
                    [before_embeds, activation_embeds[batch_idx], after_embeds], dim=0
                )
            )

            before_mask = prompt_attention_mask[batch_idx, :activation_pos]
            after_mask = prompt_attention_mask[batch_idx, activation_pos + 1 :]
            activation_mask = torch.ones(
                self.prefix_len,
                dtype=prompt_attention_mask.dtype,
                device=prompt_attention_mask.device,
            )
            expanded_masks.append(torch.cat([before_mask, activation_mask, after_mask]))

        return torch.stack(expanded_embeds, dim=0), torch.stack(expanded_masks, dim=0)

    def forward(
        self,
        activation: torch.Tensor,
        prompt_input_ids: torch.Tensor,
        prompt_attention_mask: torch.Tensor,
        summary_input_ids: torch.Tensor,
        summary_attention_mask: torch.Tensor,
        labels: torch.Tensor | None = None,
        **kwargs: Any,
    ) -> CausalLMOutputWithPast:
        prompt_embeds, prompt_mask = self._expand_activation_token(
            prompt_input_ids=prompt_input_ids,
            prompt_attention_mask=prompt_attention_mask,
            activation=activation,
        )
        summary_embeds = self.model.get_input_embeddings()(summary_input_ids)
        inputs_embeds = torch.cat([prompt_embeds, summary_embeds], dim=1)
        attention_mask = torch.cat([prompt_mask, summary_attention_mask], dim=1)

        full_labels = None
        if labels is not None:
            full_labels = torch.full(
                (labels.shape[0], prompt_embeds.shape[1] + labels.shape[1]),
                -100,
                dtype=labels.dtype,
                device=labels.device,
            )
            full_labels[:, prompt_embeds.shape[1] :] = labels

        return self.model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=full_labels,
            use_cache=False,
            **kwargs,
        )


@dataclass(frozen=True)
class DistillRow:
    activation_path: str
    teacher_summary: str
    record_id: str


class ActivationVerbalizerDataset(Dataset):
    def __init__(
        self,
        rows: list[DistillRow],
        tokenizer: Any,
        *,
        prompt: str,
        max_summary_length: int,
    ) -> None:
        self.rows = rows
        self.tokenizer = tokenizer
        self.prompt = prompt
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

        prompt = self.tokenizer(
            self.prompt,
            padding=False,
            add_special_tokens=True,
        )
        summary_ids = self.tokenizer(
            row.teacher_summary,
            truncation=True,
            max_length=self.max_summary_length,
            padding=False,
            add_special_tokens=False,
        )["input_ids"]
        if self.tokenizer.eos_token_id is not None:
            summary_ids = summary_ids[: self.max_summary_length - 1]
            summary_ids.append(self.tokenizer.eos_token_id)
        if not summary_ids:
            raise ValueError(f"{row.record_id}: empty teacher_summary")

        return {
            "activation": activation,
            "prompt_input_ids": torch.tensor(prompt["input_ids"], dtype=torch.long),
            "prompt_attention_mask": torch.tensor(
                prompt["attention_mask"], dtype=torch.long
            ),
            "summary_input_ids": torch.tensor(summary_ids, dtype=torch.long),
            "summary_attention_mask": torch.ones(len(summary_ids), dtype=torch.long),
            "labels": torch.tensor(summary_ids, dtype=torch.long),
            "record_id": row.record_id,
        }


@dataclass
class ActivationVerbalizerCollator:
    pad_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        activations = torch.stack([feature["activation"] for feature in features])
        max_prompt_len = max(
            feature["prompt_input_ids"].shape[0] for feature in features
        )
        max_summary_len = max(
            feature["summary_input_ids"].shape[0] for feature in features
        )

        prompt_input_ids: list[torch.Tensor] = []
        prompt_attention_mask: list[torch.Tensor] = []
        summary_input_ids: list[torch.Tensor] = []
        summary_attention_mask: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []

        for feature in features:
            prompt_pad = max_prompt_len - feature["prompt_input_ids"].shape[0]
            summary_pad = max_summary_len - feature["summary_input_ids"].shape[0]

            prompt_input_ids.append(
                nn.functional.pad(
                    feature["prompt_input_ids"],
                    (0, prompt_pad),
                    value=self.pad_token_id,
                )
            )
            prompt_attention_mask.append(
                nn.functional.pad(
                    feature["prompt_attention_mask"], (0, prompt_pad), value=0
                )
            )
            summary_input_ids.append(
                nn.functional.pad(
                    feature["summary_input_ids"],
                    (0, summary_pad),
                    value=self.pad_token_id,
                )
            )
            summary_attention_mask.append(
                nn.functional.pad(
                    feature["summary_attention_mask"], (0, summary_pad), value=0
                )
            )
            labels.append(
                nn.functional.pad(feature["labels"], (0, summary_pad), value=-100)
            )

        return {
            "activation": activations,
            "prompt_input_ids": torch.stack(prompt_input_ids),
            "prompt_attention_mask": torch.stack(prompt_attention_mask),
            "summary_input_ids": torch.stack(summary_input_ids),
            "summary_attention_mask": torch.stack(summary_attention_mask),
            "labels": torch.stack(labels),
        }


class AVTrainer(Trainer):
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
                if "av_projector" in name:
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
            state_dict = {key: value.clone().cpu() for key, value in state_dict.items()}

            base_model_sd = {
                key[6:]: value
                for key, value in state_dict.items()
                if key.startswith("model.")
            }
            unwrapped.model.save_pretrained(output_path, state_dict=base_model_sd)

            av_sd = {
                key[13:]: value
                for key, value in state_dict.items()
                if key.startswith("av_projector.")
            }
            torch.save(
                {
                    "av_projector_state_dict": av_sd,
                    "activation_dim": unwrapped.av_projector.proj.in_features,
                    "hidden_size": unwrapped.av_projector.proj.out_features,
                    "prefix_len": unwrapped.prefix_len,
                    "activation_token": ACTIVATION_TOKEN,
                    "activation_token_id": unwrapped.activation_token_id,
                },
                output_path / "av_projector.pt",
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


def build_model_and_tokenizer(
    args: argparse.Namespace,
) -> tuple[ActivationVerbalizerWarmStart, Any]:
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, cache_dir=args.cache_dir, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.add_special_tokens({"additional_special_tokens": [ACTIVATION_TOKEN]})

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    base_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        cache_dir=args.cache_dir,
        dtype=torch.bfloat16 if args.bf16 else torch.float32,
        trust_remote_code=True,
        device_map={"": local_rank},
    )
    base_model.resize_token_embeddings(len(tokenizer))
    base_model.config.pad_token_id = tokenizer.pad_token_id
    if args.gradient_checkpointing and not args.fsdp:
        base_model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        base_model.config.use_cache = False

    activation_token_id = tokenizer.convert_tokens_to_ids(ACTIVATION_TOKEN)
    model = ActivationVerbalizerWarmStart(
        base_model,
        activation_dim=args.activation_dim,
        prefix_len=args.prefix_len,
        activation_token_id=activation_token_id,
    )
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
        # report_to=[],
        remove_unused_columns=False,
        label_names=["labels"],
        gradient_checkpointing=False if args.fsdp else args.gradient_checkpointing,
        optim="adamw_torch",
        fsdp=args.fsdp,
        fsdp_config=fsdp_config,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="wandb",
        run_name="avinitialization1",
    )


def main() -> None:
    if "LOCAL_RANK" in os.environ:
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
    args = parse_args()
    if not args.data.exists():
        raise SystemExit(f"Data file not found: {args.data}")
    if ACTIVATION_TOKEN not in args.prompt:
        raise SystemExit(f"--prompt must contain {ACTIVATION_TOKEN}")

    torch.manual_seed(args.seed)
    model, tokenizer = build_model_and_tokenizer(args)
    model = model.to(f"cuda:{local_rank}")
    train_rows, eval_rows = build_rows_by_split(args)
    if not train_rows:
        raise SystemExit("No training rows found.")

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_rows = train_rows.copy()
    train_indices = torch.randperm(len(train_rows), generator=generator).tolist()
    train_rows = [train_rows[index] for index in train_indices]

    train_dataset = ActivationVerbalizerDataset(
        train_rows,
        tokenizer,
        prompt=args.prompt,
        max_summary_length=args.max_summary_length,
    )
    eval_dataset = (
        ActivationVerbalizerDataset(
            eval_rows,
            tokenizer,
            prompt=args.prompt,
            max_summary_length=args.max_summary_length,
        )
        if eval_rows
        else None
    )

    print(
        f"Loaded AV train={len(train_dataset)} eval={0 if eval_dataset is None else len(eval_dataset)} "
        f"activation_prefix=({args.prefix_len}, {args.activation_dim})",
        flush=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = AVTrainer(
        model=model,
        args=build_training_args(args, eval_dataset),
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        data_collator=ActivationVerbalizerCollator(pad_token_id=tokenizer.pad_token_id),
        head_learning_rate=args.head_learning_rate,
    )
    trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)

    final_dir = args.output_dir / "final"
    trainer.save_model(str(final_dir))
    if trainer.args.should_save:
        tokenizer.save_pretrained(final_dir)


if __name__ == "__main__":
    main()
