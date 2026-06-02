#!/usr/bin/env python
from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from warm_start.ar import ActivationReconstructor
from warm_start.av import ACTIVATION_TOKEN, ActivationVerbalizerProjector

DATA = ROOT / "data/pretrain/shard-combined_teacher_tp4_encoder14_ln2_betas.jsonl"
AV_DIR = ROOT / "outputs/warm_start/qwen3.5-2b-nla-av/final"
AR_DIR = ROOT / "outputs/warm_start/qwen3.5-2b-nla-ar/final"
CACHE_DIR = "/scratch/hub"
PROMPT = f"Explain the ideas in this chess position: {ACTIVATION_TOKEN}\n"
SPLIT = "val"
N_SAMPLES = 1000
SEED = 42
MAX_NEW_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE != "cpu" else torch.float32


@dataclass(frozen=True)
class Row:
    record_id: str
    activation_path: str


def load_validation_sample() -> list[Row]:
    rows: list[Row] = []
    with DATA.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("split") != SPLIT:
                continue
            activation = record.get("activation") or {}
            activation_path = activation.get("path")
            if activation_path:
                rows.append(
                    Row(
                        record_id=record.get("id", f"row-{line_no}"),
                        activation_path=activation_path,
                    )
                )

    if len(rows) < N_SAMPLES:
        raise SystemExit(f"Only found {len(rows)} validation rows, need {N_SAMPLES}.")

    rng = random.Random(SEED)
    return rng.sample(rows, N_SAMPLES)


def load_av() -> tuple[
    AutoModelForCausalLM,
    AutoTokenizer,
    ActivationVerbalizerProjector,
    dict[str, Any],
]:
    tokenizer = AutoTokenizer.from_pretrained(
        AV_DIR, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        AV_DIR,
        cache_dir=CACHE_DIR,
        dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    payload = torch.load(
        AV_DIR / "av_projector.pt", map_location="cpu", weights_only=False
    )
    projector = ActivationVerbalizerProjector(
        activation_dim=payload["activation_dim"],
        hidden_size=payload["hidden_size"],
    )
    projector.load_state_dict(payload["av_projector_state_dict"])
    projector.to(device=DEVICE, dtype=DTYPE)
    projector.eval()
    return model, tokenizer, projector, payload


def load_ar() -> (
    tuple[AutoModelForCausalLM, AutoTokenizer, ActivationReconstructor, dict[str, Any]]
):
    tokenizer = AutoTokenizer.from_pretrained(
        AR_DIR, cache_dir=CACHE_DIR, trust_remote_code=True
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        AR_DIR,
        cache_dir=CACHE_DIR,
        dtype=DTYPE,
        trust_remote_code=True,
    ).to(DEVICE)
    model.config.pad_token_id = tokenizer.pad_token_id
    model.eval()

    payload = torch.load(AR_DIR / "ar_head.pt", map_location="cpu", weights_only=False)
    reconstructor = ActivationReconstructor(
        hidden_size=payload["hidden_size"],
        activation_dim=payload["activation_dim"],
        prefix_len=payload["prefix_len"],
        reconstructor_hidden_size=payload["reconstructor_hidden_size"],
    )
    reconstructor.load_state_dict(payload["ar_reconstructor_state_dict"])
    reconstructor.to(device=DEVICE, dtype=DTYPE)
    reconstructor.eval()
    return model, tokenizer, reconstructor, payload


def expand_activation_prompt(
    *,
    av_model: AutoModelForCausalLM,
    av_tokenizer: AutoTokenizer,
    projector: ActivationVerbalizerProjector,
    activation: torch.Tensor,
    activation_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    encoded = av_tokenizer(PROMPT, add_special_tokens=True, return_tensors="pt").to(
        DEVICE
    )
    token_positions = (encoded.input_ids[0] == activation_token_id).nonzero(
        as_tuple=False
    )
    if token_positions.shape[0] != 1:
        raise ValueError(f"Expected exactly one {ACTIVATION_TOKEN} in PROMPT.")

    activation_pos = token_positions.item()
    token_embedding = av_model.get_input_embeddings()
    activation_embeds = projector(activation.to(dtype=projector.proj.weight.dtype))

    before_ids = encoded.input_ids[:, :activation_pos]
    after_ids = encoded.input_ids[:, activation_pos + 1 :]
    before_mask = encoded.attention_mask[:, :activation_pos]
    after_mask = encoded.attention_mask[:, activation_pos + 1 :]
    activation_mask = torch.ones(
        (1, activation_embeds.shape[1]), dtype=before_mask.dtype, device=DEVICE
    )

    inputs_embeds = torch.cat(
        [
            token_embedding(before_ids),
            activation_embeds,
            token_embedding(after_ids),
        ],
        dim=1,
    )
    attention_mask = torch.cat([before_mask, activation_mask, after_mask], dim=1)
    return inputs_embeds, attention_mask


@torch.no_grad()
def verbalize_activation(
    *,
    av_model: AutoModelForCausalLM,
    av_tokenizer: AutoTokenizer,
    projector: ActivationVerbalizerProjector,
    activation: torch.Tensor,
    activation_token_id: int,
) -> str:
    inputs_embeds, attention_mask = expand_activation_prompt(
        av_model=av_model,
        av_tokenizer=av_tokenizer,
        projector=projector,
        activation=activation,
        activation_token_id=activation_token_id,
    )
    generated = av_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=av_tokenizer.pad_token_id,
        eos_token_id=av_tokenizer.eos_token_id,
    )
    return av_tokenizer.decode(generated[0], skip_special_tokens=True)


@torch.no_grad()
def reconstruct_activation(
    *,
    ar_model: AutoModelForCausalLM,
    ar_tokenizer: AutoTokenizer,
    reconstructor: ActivationReconstructor,
    text: str,
) -> torch.Tensor:
    encoded = ar_tokenizer(
        text,
        padding=False,
        truncation=True,
        max_length=MAX_NEW_TOKENS,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(DEVICE)
    outputs = ar_model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )
    final_hidden = outputs.hidden_states[-1]
    last_token_index = encoded.attention_mask.sum(dim=1).clamp_min(1) - 1
    final_summary_hidden = final_hidden[
        torch.arange(encoded.input_ids.shape[0], device=DEVICE), last_token_index
    ]
    return reconstructor(final_summary_hidden).float().cpu()[0]


def expand_activation_prompt_batched(
    *,
    av_model: AutoModelForCausalLM,
    av_tokenizer: AutoTokenizer,
    projector: ActivationVerbalizerProjector,
    activations: torch.Tensor,
    activation_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Batched version of expand_activation_prompt.
    Replaces the single activation token in the prompt with the batched projected continuous embeddings.
    """
    encoded = av_tokenizer(PROMPT, add_special_tokens=True, return_tensors="pt").to(
        DEVICE
    )

    token_positions = (encoded.input_ids[0] == activation_token_id).nonzero(
        as_tuple=False
    )
    if token_positions.shape[0] != 1:
        raise ValueError(f"Expected exactly one {ACTIVATION_TOKEN} in PROMPT.")

    activation_pos = token_positions.item()
    token_embedding = av_model.get_input_embeddings()

    # activations is assumed to be batched: (batch_size, prefix_len, activation_dim)
    activation_embeds = projector(activations.to(dtype=projector.proj.weight.dtype))
    batch_size = activation_embeds.shape[0]

    # Slice the prompt around the activation token and expand to batch size
    before_ids = encoded.input_ids[:, :activation_pos].expand(batch_size, -1)
    after_ids = encoded.input_ids[:, activation_pos + 1 :].expand(batch_size, -1)

    before_mask = encoded.attention_mask[:, :activation_pos].expand(batch_size, -1)
    after_mask = encoded.attention_mask[:, activation_pos + 1 :].expand(batch_size, -1)

    # The mask for the projected embeddings needs to cover their entire sequence length
    activation_mask = torch.ones(
        (batch_size, activation_embeds.shape[1]), dtype=before_mask.dtype, device=DEVICE
    )

    # Embed the text tokens and concatenate everything together
    inputs_embeds = torch.cat(
        [
            token_embedding(before_ids),
            activation_embeds,
            token_embedding(after_ids),
        ],
        dim=1,
    )

    attention_mask = torch.cat([before_mask, activation_mask, after_mask], dim=1)

    return inputs_embeds, attention_mask


@torch.no_grad()
def verbalize_activation_batched(
    *,
    av_model: AutoModelForCausalLM,
    av_tokenizer: AutoTokenizer,
    projector: ActivationVerbalizerProjector,
    activations: torch.Tensor,
    activation_token_id: int,
) -> list[str]:
    # Ensure left-padding is used for batched generation if your tokenizer isn't already set up for it
    av_tokenizer.padding_side = "left"

    inputs_embeds, attention_mask = expand_activation_prompt_batched(
        av_model=av_model,
        av_tokenizer=av_tokenizer,
        projector=projector,
        activations=activations,
        activation_token_id=activation_token_id,
    )

    generated = av_model.generate(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=False,
        pad_token_id=av_tokenizer.pad_token_id,
        eos_token_id=av_tokenizer.eos_token_id,
    )
    return av_tokenizer.batch_decode(generated, skip_special_tokens=True)


@torch.no_grad()
def reconstruct_activation_batched(
    *,
    ar_model: AutoModelForCausalLM,
    ar_tokenizer: AutoTokenizer,
    reconstructor: ActivationReconstructor,
    texts: list[str],
) -> torch.Tensor:
    # Ensure standard right-padding for the forward pass
    ar_tokenizer.padding_side = "right"

    encoded = ar_tokenizer(
        texts,
        padding=True,  # Changed from False to support variable length batched strings
        truncation=True,
        max_length=MAX_NEW_TOKENS,
        add_special_tokens=True,
        return_tensors="pt",
    ).to(DEVICE)

    outputs = ar_model(
        input_ids=encoded.input_ids,
        attention_mask=encoded.attention_mask,
        output_hidden_states=True,
        return_dict=True,
        use_cache=False,
    )

    final_hidden = outputs.hidden_states[-1]

    # Grab the hidden state of the last non-padded token for each sequence in the batch
    last_token_indices = encoded.attention_mask.sum(dim=1).clamp_min(1) - 1
    final_summary_hidden = final_hidden[
        torch.arange(encoded.input_ids.shape[0], device=DEVICE), last_token_indices
    ]

    return reconstructor(final_summary_hidden).float().cpu()


def main() -> None:
    BATCH_SIZE = 32

    rows = load_validation_sample()
    print(f"Loaded {N_SAMPLES} random validation samples with seed {SEED}.", flush=True)

    av_model, av_tokenizer, projector, av_payload = load_av()
    ar_model, ar_tokenizer, reconstructor, ar_payload = load_ar()

    av_shape = (av_payload["prefix_len"], av_payload["activation_dim"])
    ar_shape = (ar_payload["prefix_len"], ar_payload["activation_dim"])
    if av_shape != ar_shape:
        raise SystemExit(
            f"AV activation shape {av_shape} != AR activation shape {ar_shape}"
        )

    activation_token = av_payload.get("activation_token", ACTIVATION_TOKEN)
    activation_token_id = av_tokenizer.convert_tokens_to_ids(activation_token)

    all_activations: list[torch.Tensor] = []
    all_reconstructions: list[torch.Tensor] = []

    for i in range(0, len(rows), BATCH_SIZE):
        batch_rows = rows[i : i + BATCH_SIZE]
        batch_activations = []
        for row in batch_rows:
            act = torch.load(
                row.activation_path, map_location="cpu", weights_only=False
            ).float()
            if act.shape != torch.Size(av_shape):
                raise ValueError(
                    f"{row.record_id}: expected activation {av_shape}, got {tuple(act.shape)}"
                )
            batch_activations.append(act)

        activations_tensor = torch.stack(batch_activations).to(DEVICE)
        texts = verbalize_activation_batched(
            av_model=av_model,
            av_tokenizer=av_tokenizer,
            projector=projector,
            activations=activations_tensor,
            activation_token_id=activation_token_id,
        )
        reconstructions_tensor = reconstruct_activation_batched(
            ar_model=ar_model,
            ar_tokenizer=ar_tokenizer,
            reconstructor=reconstructor,
            texts=texts,
        )

        if i == 0:
            print("\nFirst sample record_id:", batch_rows[0].record_id, flush=True)
            print("\nFirst sample AV verbalization:\n", texts[0], flush=True)
            print(
                "\nFirst sample reconstruction:\n",
                reconstructions_tensor[0],
                flush=True,
            )

        all_activations.extend(batch_activations)
        all_reconstructions.extend(reconstructions_tensor.unbind(0))

        print(f"Processed {min(i + BATCH_SIZE, N_SAMPLES)}/{N_SAMPLES}", flush=True)

    target = torch.stack(all_activations)
    reconstructed = torch.stack(all_reconstructions)
    mean_activation = target.mean(dim=0, keepdim=True)

    reconstruction_sse = (target - reconstructed).square().sum().item()
    variance_sse = (target - mean_activation).square().sum().item()
    fve = 1.0 - reconstruction_sse / variance_sse

    print("\nFVE over random validation sample", flush=True)
    print(f"n = {N_SAMPLES}", flush=True)
    print(f"fve = {fve:.6f}", flush=True)
    print(f"reconstruction_sse = {reconstruction_sse:.6f}", flush=True)
    print(f"variance_sse = {variance_sse:.6f}", flush=True)


if __name__ == "__main__":
    main()
