import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoModel,
    get_cosine_schedule_with_warmup,
)
import bitsandbytes as bnb
import wandb
import os
import json
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import concurrent.futures
import time
from collections import defaultdict
from contextlib import contextmanager

torch._dynamo.config.capture_scalar_outputs = True

av_checkpoint_dir = "../../outputs/warm_start/qwen3.5-2b-nla-av/final"
ar_checkpoint_dir = "../../outputs/warm_start/qwen3.5-2b-nla-ar/final"
av_device = "cuda:0"
ar_device = "cuda:1"
frozen_device = "cuda:2"
av_device_2 = "cuda:3"
G = 8
B = 32
COSINE_WEIGHT = 0.25
KL_WEIGHT = 0.25
LR = 3e-6
EPS = 0.2
PPO_STEPS = 1
MAX_CHECKPOINTS = 3
EVAL_EVERY = 1000
CHECKPOINT_EVERY = 1000

config = {
    "learning_rate": LR,
    "num_rollouts": G,
    "batch_size": B,
    "cosine_weight": COSINE_WEIGHT,
    "kl_weight": KL_WEIGHT,
    "eval_every": EVAL_EVERY,
    "checkpoint_every": CHECKPOINT_EVERY,
    "ppo_steps": PPO_STEPS,
    "eps": EPS,
}

PROJECT_NAME = "nla-train-2-ppo"
PROD = False

if PROD:
    wandb.login()
    wandb.init(project="lc0-nla", name=PROJECT_NAME, config=config)


class LeelaActivationDataset(Dataset):
    def __init__(
        self,
        activation_dir: (
            str | Path
        ) = "../../data/nla_training/activations_encoder14_ln2_betas",
        split="train",
        device=av_device,
    ):
        self.device = device
        self.activation_dir = Path(activation_dir)
        jsonl_path = self.activation_dir.parent / "shard-1M.jsonl"
        self.activation_paths = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != split:
                    continue
                activation = record.get("activation")
                self.activation_paths.append("../../" + activation["path"])

    def __len__(self):
        return len(self.activation_paths)

    def __getitem__(self, index):
        return torch.load(
            self.activation_paths[index], map_location="cpu", weights_only=False
        ).to(torch.bfloat16)


class AVProjector(nn.Module):
    def __init__(self, activation_dim: int, hidden_size: int) -> None:
        super().__init__()
        self.proj = nn.Linear(activation_dim, hidden_size)

    def forward(self, activation: torch.Tensor) -> torch.Tensor:
        return self.proj(activation)


class ARReconstructor(nn.Module):
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


av_tokenizer = AutoTokenizer.from_pretrained(av_checkpoint_dir, trust_remote_code=True)
ar_tokenizer = AutoTokenizer.from_pretrained(ar_checkpoint_dir, trust_remote_code=True)
av_model = AutoModelForCausalLM.from_pretrained(
    av_checkpoint_dir,
    dtype=torch.bfloat16,
    device_map=av_device,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
av_model_2 = AutoModelForCausalLM.from_pretrained(
    av_checkpoint_dir,
    dtype=torch.bfloat16,
    device_map=av_device_2,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
av_model.config.pad_token_id = av_tokenizer.pad_token_id
av_model_2.config.pad_token_id = av_tokenizer.pad_token_id
# av_model.gradient_checkpointing_enable()
av_checkpoint = torch.load(
    f"{av_checkpoint_dir}/av_projector.pt", map_location=av_device
)
av_projectors = AVProjector(
    av_checkpoint["activation_dim"], av_checkpoint["hidden_size"]
).to(av_device, dtype=torch.bfloat16)
av_projectors.load_state_dict(av_checkpoint["av_projector_state_dict"])
av_projectors_2 = AVProjector(
    av_checkpoint["activation_dim"], av_checkpoint["hidden_size"]
).to(av_device_2, dtype=torch.bfloat16)
av_projectors_2.load_state_dict(av_checkpoint["av_projector_state_dict"])
av_optimizer = bnb.optim.adamw.PagedAdamW8bit(
    params=list(av_projectors.parameters()) + list(av_model.parameters()), lr=LR
)
activation_token = "<activation/>"
prompt_inputs = av_tokenizer(
    f"Explain the ideas in this chess position: {activation_token}\n",
    return_tensors="pt",
).to(av_device)
prompt_input_ids = prompt_inputs.input_ids
prompt_attention_mask = prompt_inputs.attention_mask
activation_token_id = av_tokenizer.convert_tokens_to_ids(activation_token)
activation_pos = (
    (prompt_input_ids[0] == activation_token_id).nonzero(as_tuple=False)[0].item()
)

# this will warning because it's AutoModel and no lm_head
ar_model = AutoModel.from_pretrained(
    ar_checkpoint_dir,
    dtype=torch.bfloat16,
    device_map=ar_device,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
ar_model.config.pad_token_id = ar_tokenizer.pad_token_id
# ar_model.gradient_checkpointing_enable()
ar_checkpoint = torch.load(f"{ar_checkpoint_dir}/ar_head.pt", map_location=ar_device)
ar_projectors = ARReconstructor(
    ar_checkpoint["hidden_size"],
    ar_checkpoint["activation_dim"],
    ar_checkpoint["prefix_len"],
    ar_checkpoint.get("reconstructor_hidden_size"),
).to(ar_device, dtype=torch.bfloat16)
ar_projectors.load_state_dict(ar_checkpoint["ar_reconstructor_state_dict"])
ar_optimizer = bnb.optim.adamw.PagedAdamW8bit(
    params=list(ar_projectors.parameters()) + list(ar_model.parameters()), lr=LR
)

frozen_model = AutoModelForCausalLM.from_pretrained(
    av_checkpoint_dir,
    dtype=torch.bfloat16,
    device_map=frozen_device,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
frozen_model.config.pad_token_id = av_tokenizer.pad_token_id
frozen_projectors = AVProjector(
    av_checkpoint["activation_dim"], av_checkpoint["hidden_size"]
).to(frozen_device, dtype=torch.bfloat16)
frozen_projectors.load_state_dict(av_checkpoint["av_projector_state_dict"])
frozen_model.eval()
frozen_model.requires_grad_(False)
frozen_projectors.eval()
frozen_projectors.requires_grad_(False)

saved_checkpoints = []

leela_activation_dataset = LeelaActivationDataset()
train_dataloader = DataLoader(
    dataset=leela_activation_dataset,
    batch_size=B,
    shuffle=True,
    drop_last=True,
    num_workers=8,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=4,
)

av_scheduler = get_cosine_schedule_with_warmup(
    optimizer=av_optimizer,
    num_warmup_steps=int(len(train_dataloader) * 0.05),
    num_training_steps=len(train_dataloader),
)
ar_scheduler = get_cosine_schedule_with_warmup(
    optimizer=ar_optimizer,
    num_warmup_steps=int(len(train_dataloader) * 0.05),
    num_training_steps=len(train_dataloader),
)

av_model = torch.compile(av_model, dynamic=True)
av_model_2 = torch.compile(av_model_2, dynamic=True)
ar_model = torch.compile(ar_model, dynamic=True)
frozen_model = torch.compile(frozen_model, dynamic=True)

av_model.train()
av_model_2.train()
ar_model.train()
av_projectors.train()
av_projectors_2.train()
ar_projectors.train()


def run_av_generate(av_model, inputs_embeds, attention_mask, device):
    with torch.cuda.device(device):
        with torch.no_grad():
            rollouts = av_model.generate(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=200,
                pad_token_id=av_tokenizer.pad_token_id,
                eos_token_id=av_tokenizer.eos_token_id,
                do_sample=True,
                temperature=1.0,
                top_p=0.9,
                use_cache=True,
            )
            return rollouts


def run_av_fwd(av_model, inputs_embeds, attention_mask, device):
    with torch.cuda.device(device):
        with torch.no_grad():
            outputs = av_model.model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return outputs


def run_frozen_log_probs(
    frozen_model,
    full_attention_mask,
    frozen_full_inputs_embeds,
    rollouts,
    not_pad_mask,
    chunk_size_frozen,
    prompt_len,
    frozen_device,
):
    with torch.no_grad():
        frozen_token_log_probs = []
        for i in range(0, B * G, chunk_size_frozen):
            frozen_chunked_mask = full_attention_mask.to(frozen_device)[
                i : i + chunk_size_frozen
            ]
            frozen_chunked_full_inputs_embeds = frozen_full_inputs_embeds[
                i : i + chunk_size_frozen
            ]
            frozen_outputs = frozen_model(
                inputs_embeds=frozen_chunked_full_inputs_embeds,
                attention_mask=frozen_chunked_mask,
                use_cache=False,
            )
            frozen_logits = frozen_outputs.logits[:, prompt_len - 1 : -1, :]
            frozen_chunked_log_probs = -F.cross_entropy(
                frozen_logits.reshape(-1, frozen_logits.size(-1)),
                rollouts[i : i + chunk_size_frozen].to(frozen_device).view(-1),
                reduction="none",
            ).detach()
            frozen_chunked_log_probs = frozen_chunked_log_probs * not_pad_mask[
                i : i + chunk_size_frozen
            ].to(frozen_device).view(-1)
            frozen_token_log_probs.append(frozen_chunked_log_probs)
        return torch.cat(frozen_token_log_probs, dim=0).view(B * G, -1)


def run_av_old_log_probs(
    av_model, rollouts, generated_old_outputs, not_pad_mask, chunk_size_av
):
    with torch.no_grad():
        old_token_log_probs_list = []
        for i in range(0, B * G, chunk_size_av):
            chunk_outputs = generated_old_outputs[i : i + chunk_size_av]
            chunk_rollouts = rollouts[i : i + chunk_size_av]
            chunk_logits = av_model.lm_head(chunk_outputs)
            chunk_log_probs = -F.cross_entropy(
                chunk_logits.reshape(-1, chunk_logits.size(-1)),
                chunk_rollouts.view(-1),
                reduction="none",
            )
            old_token_log_probs_list.append(chunk_log_probs.detach())
        old_token_log_probs = torch.cat(old_token_log_probs_list, dim=0).view(B * G, -1)
        old_token_log_probs = old_token_log_probs * not_pad_mask
        return old_token_log_probs


def run_ar_update(
    ar_model, ar_projectors, rollouts, leela_activations, ar_device, mini_batch_size_ar
):
    rewards = []
    ar_loss = torch.tensor(0.0, device=ar_device)
    train_mse = torch.tensor(0.0, device=ar_device)
    train_cosine = torch.tensor(0.0, device=ar_device)
    residual_sse = torch.tensor(0.0, device=ar_device, dtype=torch.float32)
    variance_sse = torch.tensor(0.0, device=ar_device, dtype=torch.float32)
    batch_activation_mean = leela_activations.to(ar_device, dtype=torch.float32).mean(
        dim=0, keepdim=True
    )
    for k in range(0, B, mini_batch_size_ar):
        sliced_rollouts = (
            rollouts[k * G : k * G + G * mini_batch_size_ar].to(ar_device).contiguous()
        )
        generated_attention_mask = (
            (sliced_rollouts != av_tokenizer.pad_token_id)
            .long()
            .to(ar_device)
            .contiguous()
        )
        sliced_leela_activations = (
            leela_activations[k : k + mini_batch_size_ar]
            .repeat_interleave(G, dim=0)
            .to(ar_device)
            .contiguous()
        )
        reconstructed = ar_model(
            input_ids=sliced_rollouts,
            attention_mask=generated_attention_mask,
            return_dict=True,
            use_cache=False,
        )
        final_hidden = reconstructed.last_hidden_state
        last_token_index = generated_attention_mask.sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(sliced_rollouts.shape[0], device=ar_device)
        final_summary_hidden = final_hidden[batch_index, last_token_index].to(
            dtype=torch.bfloat16
        )
        reconstructed_vector = ar_projectors(final_summary_hidden)
        mse_tensor = ((reconstructed_vector - sliced_leela_activations) ** 2).mean(
            dim=(1, 2)
        )
        cos_loss = 1 - F.cosine_similarity(
            reconstructed_vector.flatten(1),
            sliced_leela_activations.flatten(1),
        )
        num_steps = B // mini_batch_size_ar
        ar_loss_this_mini_batch = (mse_tensor + COSINE_WEIGHT * cos_loss).mean()
        (ar_loss_this_mini_batch / num_steps).backward()
        ar_loss += ar_loss_this_mini_batch.detach() / num_steps
        train_mse += mse_tensor.mean().detach() / num_steps
        train_cosine += cos_loss.mean().detach() / num_steps
        residual_sse += (
            (reconstructed_vector.detach().float() - sliced_leela_activations.float())
            ** 2
        ).sum()
        variance_sse += (
            (sliced_leela_activations.float() - batch_activation_mean) ** 2
        ).sum()
        mini_batch_rewards = -(mse_tensor + COSINE_WEIGHT * cos_loss).detach()
        rewards.append(mini_batch_rewards)
    train_fve = 1 - residual_sse / variance_sse
    return rewards, ar_loss, train_mse, train_cosine, train_fve


def run_av_ppo(
    k_start,
    k_end,
    mini_batch_size_av,
    num_steps,
    device,
    av_model,
    av_projectors,
    token_embeddings,
    rollouts,
    av_tokenizer,
    leela_activations,
    prompt_len,
    advantages,
    old_token_log_probs,
    frozen_token_log_probs,
):
    local_av_loss = torch.tensor(0.0, device=device)
    local_kl_div = torch.tensor(0.0, device=device)

    for k in range(k_start, k_end, mini_batch_size_av):
        sliced_rollouts = (
            rollouts[k * G : k * G + G * mini_batch_size_av].to(device).contiguous()
        )
        generated_attention_mask = (
            (sliced_rollouts != av_tokenizer.pad_token_id)
            .long()
            .to(device)
            .contiguous()
        )
        sliced_leela_activations = (
            leela_activations[k : k + mini_batch_size_av].to(device).contiguous()
        )

        before_mask = (
            prompt_attention_mask[:, :activation_pos]
            .to(device)
            .expand(mini_batch_size_av, -1)
        )
        after_mask = (
            prompt_attention_mask[:, activation_pos + 1 :]
            .to(device)
            .expand(mini_batch_size_av, -1)
        )
        av_mask = torch.ones(
            mini_batch_size_av,
            av_checkpoint["prefix_len"],
            dtype=prompt_attention_mask.dtype,
            device=device,
        )

        before_embeds = (
            token_embeddings(prompt_input_ids[:, :activation_pos].to(device))
            .to(device)
            .expand(mini_batch_size_av, -1, -1)
        )
        after_embeds = (
            token_embeddings(prompt_input_ids[:, activation_pos + 1 :].to(device))
            .to(device)
            .expand(mini_batch_size_av, -1, -1)
        )
        av_embeds = av_projectors(sliced_leela_activations)

        attention_mask_this_batch = torch.cat(
            [before_mask, av_mask, after_mask], dim=1
        ).repeat_interleave(G, dim=0)
        input_embeds_this_batch = torch.cat(
            [before_embeds, av_embeds, after_embeds], dim=1
        ).repeat_interleave(G, dim=0)
        generated_embeds = token_embeddings(sliced_rollouts)

        full_mask = torch.cat(
            [attention_mask_this_batch, generated_attention_mask], dim=1
        )
        full_embeds = torch.cat([input_embeds_this_batch, generated_embeds], dim=1)

        av_outputs = av_model.model(
            inputs_embeds=full_embeds,
            attention_mask=full_mask,
            use_cache=False,
        ).last_hidden_state

        generated_av_outputs = av_outputs[:, prompt_len - 1 : -1, :].contiguous()
        logits = av_model.lm_head(generated_av_outputs)
        token_ids = sliced_rollouts.unsqueeze(-1)
        chosen_token_log_probs = -F.cross_entropy(
            logits.reshape(-1, logits.size(-1)),
            token_ids.squeeze(-1).view(-1),
            reduction="none",
        ).view(mini_batch_size_av * G, -1)

        sliced_advantages = advantages[k * G : k * G + G * mini_batch_size_av].to(
            device
        )
        chosen_token_log_probs = chosen_token_log_probs * generated_attention_mask

        sliced_old_token_log_probs = old_token_log_probs[
            k * G : k * G + G * mini_batch_size_av
        ].to(device)
        token_ratio = torch.exp(chosen_token_log_probs - sliced_old_token_log_probs)
        sliced_advantages = sliced_advantages.unsqueeze(-1)
        surr1 = token_ratio * sliced_advantages
        surr2 = torch.clamp(token_ratio, 1 - EPS, 1 + EPS) * sliced_advantages
        clipped_obj = torch.min(surr1, surr2) * generated_attention_mask
        valid_tokens = generated_attention_mask.sum(dim=-1).clamp_min(1)
        policy_loss = -(clipped_obj.sum(dim=-1) / valid_tokens).mean()

        sliced_frozen_token_log_probs = frozen_token_log_probs[
            k * G : k * G + G * mini_batch_size_av
        ].to(device)
        log_r = torch.clamp(
            sliced_frozen_token_log_probs - chosen_token_log_probs,
            min=-10,
            max=10,
        )
        kl_approx = torch.exp(log_r) - 1 - log_r
        kl_approx = kl_approx.to(device)

        masked_kl_div = kl_approx * generated_attention_mask
        valid_tokens_per_rollout = generated_attention_mask.sum(dim=-1).clamp_min(1)
        kl_div_per_rollout = masked_kl_div.sum(dim=-1) / valid_tokens_per_rollout

        av_loss_this_mini_batch = policy_loss + KL_WEIGHT * kl_div_per_rollout.mean()
        (av_loss_this_mini_batch / num_steps).backward()

        local_kl_div += kl_div_per_rollout.mean().detach() / num_steps
        local_av_loss += av_loss_this_mini_batch.detach() / num_steps
    return local_av_loss, local_kl_div


for batch_idx, leela_activations in enumerate(tqdm(train_dataloader, smoothing=1)):
    leela_activations = leela_activations.to(av_device, non_blocking=True)
    av_model.eval()
    av_model_2.eval()
    with torch.no_grad():
        token_embeddings = av_model.get_input_embeddings()
        token_embeddings_2 = av_model_2.get_input_embeddings()
        before_embeds = token_embeddings(prompt_input_ids[:, :activation_pos])
        after_embeds = token_embeddings(prompt_input_ids[:, activation_pos + 1 :])
        before_mask = prompt_attention_mask[:, :activation_pos]
        after_mask = prompt_attention_mask[:, activation_pos + 1 :]

        av_embeds = av_projectors(leela_activations)
        batch_before_embeds = before_embeds.expand(B, -1, -1)
        batch_after_embeds = after_embeds.expand(B, -1, -1)
        batch_before_mask = before_mask.expand(B, -1)
        batch_after_mask = after_mask.expand(B, -1)
        inputs_embeds = torch.cat(
            [batch_before_embeds, av_embeds, batch_after_embeds], dim=1
        )
        av_mask = torch.ones(
            B,
            av_checkpoint["prefix_len"],
            dtype=prompt_attention_mask.dtype,
            device=av_device,
        )
        attention_mask = torch.cat(
            [batch_before_mask, av_mask, batch_after_mask], dim=1
        )
        inputs_embeds = inputs_embeds.repeat_interleave(
            G, dim=0
        )  # [B * G, input_len, d_model]
        inputs_embeds, inputs_embeds_2 = torch.chunk(inputs_embeds, 2, dim=0)
        attention_mask = attention_mask.repeat_interleave(
            G, dim=0
        )  # [B * G, input_len]
        attention_mask, attention_mask_2 = torch.chunk(attention_mask, 2, dim=0)
        inputs_embeds_2 = inputs_embeds_2.to(av_device_2)
        attention_mask_2 = attention_mask_2.to(av_device_2)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            rollouts = executor.submit(
                run_av_generate, av_model, inputs_embeds, attention_mask, 0
            )
            rollouts_2 = executor.submit(
                run_av_generate, av_model_2, inputs_embeds_2, attention_mask_2, 3
            )
            rollouts = rollouts.result()
            rollouts_2 = rollouts_2.result()

            diff = rollouts.shape[1] - rollouts_2.shape[1]
            if diff > 0:
                rollouts_2 = F.pad(
                    rollouts_2, (0, diff), value=av_tokenizer.pad_token_id
                )
            elif diff < 0:
                rollouts = F.pad(rollouts, (0, -diff), value=av_tokenizer.pad_token_id)

        not_pad_mask = (rollouts != av_tokenizer.pad_token_id).long()
        not_pad_mask_2 = (rollouts_2 != av_tokenizer.pad_token_id).long()
        rollout_embeds = token_embeddings(rollouts)
        rollout_embeds_2 = token_embeddings_2(rollouts_2)

        full_inputs_embeds = torch.cat([inputs_embeds, rollout_embeds], dim=1)
        full_inputs_embeds_2 = torch.cat([inputs_embeds_2, rollout_embeds_2], dim=1)
        full_attention_mask = torch.cat([attention_mask, not_pad_mask], dim=1)
        full_attention_mask_2 = torch.cat([attention_mask_2, not_pad_mask_2], dim=1)

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            av_outputs = executor.submit(
                run_av_fwd, av_model, full_inputs_embeds, full_attention_mask, 0
            )
            av_outputs_2 = executor.submit(
                run_av_fwd, av_model_2, full_inputs_embeds_2, full_attention_mask_2, 3
            )
            av_outputs = av_outputs.result().last_hidden_state
            av_outputs_2 = av_outputs_2.result().last_hidden_state

        rollouts = torch.cat((rollouts, rollouts_2.to(av_device)), dim=0)
        av_outputs = torch.cat((av_outputs, av_outputs_2.to(av_device)), dim=0)
        full_attention_mask = torch.cat(
            (full_attention_mask, full_attention_mask_2.to(av_device)), dim=0
        )
        not_pad_mask = torch.cat((not_pad_mask, not_pad_mask_2.to(av_device)), dim=0)

        prompt_len = inputs_embeds.shape[1]
        generated_old_outputs = (
            av_outputs[:, prompt_len - 1 : -1, :].detach().contiguous()
        )

        frozen_embeds = frozen_projectors(leela_activations.to(frozen_device))
        frozen_inputs_embeds = torch.cat(
            [
                batch_before_embeds.to(frozen_device),
                frozen_embeds,
                batch_after_embeds.to(frozen_device),
            ],
            dim=1,
        ).repeat_interleave(G, dim=0)
        frozen_full_inputs_embeds = torch.cat(
            [
                frozen_inputs_embeds,
                frozen_model.get_input_embeddings()(rollouts.to(frozen_device)),
            ],
            dim=1,
        )

    chunk_size_frozen = 32
    chunk_size_av = 16
    mini_batch_size_ar = 2
    ar_optimizer.zero_grad()

    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        frozen_token_log_probs = executor.submit(
            run_frozen_log_probs,
            frozen_model,
            full_attention_mask,
            frozen_full_inputs_embeds,
            rollouts,
            not_pad_mask,
            chunk_size_frozen,
            prompt_len,
            frozen_device,
        )
        old_token_log_probs = executor.submit(
            run_av_old_log_probs,
            av_model,
            rollouts,
            generated_old_outputs,
            not_pad_mask,
            chunk_size_av,
        )
        ar_update_call = executor.submit(
            run_ar_update,
            ar_model,
            ar_projectors,
            rollouts,
            leela_activations,
            ar_device,
            mini_batch_size_ar,
        )
        frozen_token_log_probs = frozen_token_log_probs.result()
        old_token_log_probs = old_token_log_probs.result()
        rewards, ar_loss, train_mse, train_cosine, train_fve = ar_update_call.result()
    ar_grad_norm = torch.nn.utils.clip_grad_norm_(
        list(ar_projectors.parameters()) + list(ar_model.parameters()),
        max_norm=1.0,
    )
    ar_optimizer.step()
    ar_scheduler.step()

    r = torch.cat(rewards)
    r = r.view(B, G)
    r_std = r.std(dim=1).mean()
    r = (r - r.mean(dim=1, keepdim=True)) / (r.std(dim=1, keepdim=True) + 1e-8)
    r = r.view(B * G)
    advantages = r.view(-1).detach().to(av_device)  # [B * G]

    av_optimizer.zero_grad()
    av_model.train()
    av_model_2.train()
    av_loss = torch.tensor(0.0, device=av_device)
    train_kl_div = torch.tensor(0.0, device=av_device)
    av_grad_norm_tensor = torch.tensor(0.0, device=av_device)
    mini_batch_size_av = 1

    # at PPO_STEPS = 1, this is basically just reinforce, but at least it's now easily tunable :-)
    for ppo_step in range(PPO_STEPS):
        av_optimizer.zero_grad()

        num_steps = B // 2 // mini_batch_size_av
        common_kwargs = dict(
            mini_batch_size_av=mini_batch_size_av,
            num_steps=num_steps,
            rollouts=rollouts,
            av_tokenizer=av_tokenizer,
            leela_activations=leela_activations,
            prompt_len=prompt_len,
            advantages=advantages,
            old_token_log_probs=old_token_log_probs,
            frozen_token_log_probs=frozen_token_log_probs,
        )
        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
            f1 = executor.submit(
                run_av_ppo,
                0,
                B // 2,
                device=av_device,
                av_model=av_model,
                av_projectors=av_projectors,
                token_embeddings=token_embeddings,
                **common_kwargs,
            )
            f2 = executor.submit(
                run_av_ppo,
                B // 2,
                B,
                device=av_device_2,
                av_model=av_model_2,
                av_projectors=av_projectors_2,
                token_embeddings=token_embeddings_2,
                **common_kwargs,
            )
            loss0, kl0 = f1.result()
            loss1, kl1 = f2.result()
        av_loss += 0.5 * (loss0.to(av_device) + loss1.to(av_device))
        train_kl_div += 0.5 * (kl0.to(av_device) + kl1.to(av_device))
        params0 = list(av_projectors.parameters()) + list(av_model.parameters())
        params1 = list(av_projectors_2.parameters()) + list(av_model_2.parameters())
        for p0, p1 in zip(params0, params1):
            if p0.grad is None or p1.grad is None:
                continue
            # add gradients, and divide by 2 for mean
            p0.grad.add_(p1.grad.to(av_device)).mul_(0.5)
        av_grad_norm = torch.nn.utils.clip_grad_norm_(
            params0,
            max_norm=1.0,
        )
        av_grad_norm_tensor += av_grad_norm
        av_optimizer.step()
        for p in params1:
            p.grad = None
        with torch.no_grad():
            av_model_2.load_state_dict(av_model.state_dict())
            av_projectors_2.load_state_dict(av_projectors.state_dict())
    av_scheduler.step()

    if batch_idx % 5 == 0 and PROD:
        is_eval_step = batch_idx > 0 and batch_idx % EVAL_EVERY == 0
        wandb.log(
            {
                "train/av_loss": av_loss.item(),
                "train/ar_loss": ar_loss.item(),
                "train/kl_div_mean": train_kl_div.item(),
                "train/mse_term": train_mse.item(),
                "train/cosine_term": train_cosine.item(),
                "train/fve": train_fve.item(),
                "train/av_grad_norm": av_grad_norm_tensor.item(),
                "train/ar_grad_norm": ar_grad_norm.item(),
                "train/av_lr": av_optimizer.param_groups[0]["lr"],
                "train/ar_lr": ar_optimizer.param_groups[0]["lr"],
                "train/r_std": r_std.item(),
                "iteration": batch_idx,
            },
            step=batch_idx,
            commit=not is_eval_step,
        )
    elif batch_idx % 5 == 0 and not PROD:
        print(
            "train/av_loss: ",
            av_loss.item(),
            "train/ar_loss: ",
            ar_loss.item(),
            "train/kl_div_mean: ",
            train_kl_div.item(),
            "train/mse_term: ",
            train_mse.item(),
            "train/cosine_term: ",
            train_cosine.item(),
            "train/fve: ",
            train_fve.item(),
            "train/av_grad_norm: ",
            av_grad_norm_tensor.item(),
            "train/ar_grad_norm: ",
            ar_grad_norm.item(),
            "train/av_lr: ",
            av_optimizer.param_groups[0]["lr"],
            "train/ar_lr: ",
            ar_optimizer.param_groups[0]["lr"],
            "train/r_std: ",
            r_std.item(),
            "iteration: ",
            batch_idx,
        )

    ar_test_loss = torch.tensor(0.0, device=ar_device)
    av_test_loss = torch.tensor(0.0, device=av_device)
    max_test_batches = 30
    if batch_idx > 0 and batch_idx % EVAL_EVERY == 0:
        print("starting eval at step: ", batch_idx)
        av_model.eval()
        ar_model.eval()
        test_dataset = LeelaActivationDataset(split="val")
        test_dl = DataLoader(
            dataset=test_dataset, batch_size=B, shuffle=True, drop_last=True
        )
        for test_batch_idx, leela_test_activations in enumerate(tqdm(test_dl)):
            leela_test_activations = leela_test_activations.to(av_device)
            if test_batch_idx >= max_test_batches:
                ar_model.train()
                av_model.train()
                break
            with torch.no_grad():
                token_embeddings = av_model.get_input_embeddings()
                before_embeds = token_embeddings(prompt_input_ids[:, :activation_pos])
                after_embeds = token_embeddings(
                    prompt_input_ids[:, activation_pos + 1 :]
                )
                before_mask = prompt_attention_mask[:, :activation_pos]
                after_mask = prompt_attention_mask[:, activation_pos + 1 :]

                av_embeds = av_projectors(leela_test_activations)
                batch_before_embeds = before_embeds.expand(B, -1, -1)
                batch_after_embeds = after_embeds.expand(B, -1, -1)
                batch_before_mask = before_mask.expand(B, -1)
                batch_after_mask = after_mask.expand(B, -1)
                inputs_embeds = torch.cat(
                    [batch_before_embeds, av_embeds, batch_after_embeds], dim=1
                )
                av_mask = torch.ones(
                    B,
                    av_checkpoint["prefix_len"],
                    dtype=prompt_attention_mask.dtype,
                    device=av_device,
                )
                attention_mask = torch.cat(
                    [batch_before_mask, av_mask, batch_after_mask], dim=1
                )
                inputs_embeds = inputs_embeds.repeat_interleave(G, dim=0)
                attention_mask = attention_mask.repeat_interleave(G, dim=0)
                rollouts = av_model.generate(  # type: ignore
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    max_new_tokens=200,
                    pad_token_id=av_tokenizer.pad_token_id,
                    eos_token_id=av_tokenizer.eos_token_id,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    use_cache=True,
                )
                for k in range(B):
                    input_embeds_this_batch = inputs_embeds[k * G : k * G + G]
                    attention_mask_this_batch = attention_mask[k * G : k * G + G]
                    sliced_rollouts = rollouts[k * G : k * G + G]
                    generated_attention_mask = (
                        sliced_rollouts != av_tokenizer.pad_token_id
                    ).long()
                    ar_generated_attention_mask = generated_attention_mask.to(ar_device)
                    sliced_rollouts_ar = sliced_rollouts.to(ar_device)
                    token_embeddings = av_model.get_input_embeddings()
                    sliced_leela_activations = leela_test_activations[k].unsqueeze(0)
                    generated_embeds = token_embeddings(sliced_rollouts)

                    full_embeds = torch.cat(
                        [input_embeds_this_batch, generated_embeds], dim=1
                    )
                    full_mask = torch.cat(
                        [
                            attention_mask_this_batch,
                            generated_attention_mask,
                        ],
                        dim=1,
                    )
                    reconstructed = ar_model(
                        input_ids=sliced_rollouts_ar,
                        attention_mask=ar_generated_attention_mask,
                        return_dict=True,
                        use_cache=False,
                    )
                    av_outputs = av_model.model(
                        inputs_embeds=full_embeds,
                        attention_mask=full_mask,
                        use_cache=False,
                    ).last_hidden_state
                    frozen_model_outputs = frozen_model(
                        inputs_embeds=full_embeds.detach().to(frozen_device),
                        attention_mask=full_mask.to(frozen_device),
                        use_cache=False,
                    )
                    final_hidden = reconstructed.last_hidden_state
                    last_token_index = (
                        ar_generated_attention_mask.sum(dim=1).clamp_min(1) - 1
                    )
                    batch_index = torch.arange(
                        sliced_rollouts.shape[0], device=ar_device
                    )
                    final_summary_hidden = final_hidden[
                        batch_index, last_token_index
                    ].to(dtype=torch.bfloat16)
                    reconstructed_vector = ar_projectors(final_summary_hidden)
                    sliced_leela_activations_ar = sliced_leela_activations.to(ar_device)
                    mse_tensor = (
                        (reconstructed_vector - sliced_leela_activations_ar) ** 2
                    ).mean(dim=(1, 2))
                    cos_loss = 1 - F.cosine_similarity(
                        reconstructed_vector.flatten(1),
                        sliced_leela_activations_ar.flatten(1),
                    )
                    ar_loss_this_mini_batch = (
                        mse_tensor.mean() + COSINE_WEIGHT * cos_loss.mean()
                    )
                    ar_test_loss += ar_loss_this_mini_batch.detach() / B
                    raw_reward = -mse_tensor - COSINE_WEIGHT * cos_loss
                    r = (raw_reward - raw_reward.mean()) / (raw_reward.std() + 1e-8)
                    r = r.detach()
                    generated_av_outputs = av_outputs[
                        :, prompt_len - 1 : -1, :
                    ].contiguous()
                    logits = av_model.lm_head(generated_av_outputs)
                    token_ids = sliced_rollouts.unsqueeze(-1)
                    chosen_token_log_probs = -F.cross_entropy(
                        logits.reshape(-1, logits.size(-1)),
                        token_ids.squeeze(-1).view(-1),
                        reduction="none",
                    ).view(G, -1)
                    chosen_token_log_probs = (
                        chosen_token_log_probs * generated_attention_mask
                    )

                    # per-token PPO ratio is 1
                    token_ratio = torch.ones_like(chosen_token_log_probs)
                    r_expanded = r.unsqueeze(-1).to(av_device)
                    surr1 = token_ratio * r_expanded
                    surr2 = torch.clamp(token_ratio, 1 - EPS, 1 + EPS) * r_expanded
                    clipped_obj = torch.min(surr1, surr2) * generated_attention_mask
                    valid_tokens_per_rollout = generated_attention_mask.sum(
                        dim=-1
                    ).clamp_min(1)
                    policy_loss = -(
                        clipped_obj.sum(dim=-1) / valid_tokens_per_rollout
                    ).mean()

                    frozen_logits = frozen_model_outputs.logits[
                        :, prompt_len - 1 : -1, :
                    ]
                    sliced_frozen_token_log_probs = -F.cross_entropy(
                        frozen_logits.reshape(-1, frozen_logits.size(-1)),
                        token_ids.to(frozen_device).squeeze(-1).view(-1),
                        reduction="none",
                    ).view(G, -1)
                    log_r = torch.clamp(
                        sliced_frozen_token_log_probs
                        - chosen_token_log_probs.to(frozen_device),
                        min=-5,
                        max=5,
                    )
                    kl_approx = torch.exp(log_r) - 1 - log_r
                    kl_approx = kl_approx.to(av_device)

                    masked_kl_div = kl_approx * generated_attention_mask
                    kl_div_per_rollout = (
                        masked_kl_div.sum(dim=-1) / valid_tokens_per_rollout
                    )
                    av_loss_this_mini_batch = (
                        policy_loss + KL_WEIGHT * kl_div_per_rollout.mean()
                    )
                    av_test_loss += av_loss_this_mini_batch.detach() / B
        print("val/av_loss: ", av_test_loss.item() / max_test_batches)
        print("val/ar_loss: ", ar_test_loss.item() / max_test_batches)
        wandb.log(
            {
                "val/av_loss": av_test_loss.item() / max_test_batches,
                "val/ar_loss": ar_test_loss.item() / max_test_batches,
            },
            step=batch_idx,
            commit=True,
        )

    if batch_idx > 0 and batch_idx % CHECKPOINT_EVERY == 0:
        checkpoint_dir = Path(
            f"../../outputs/{PROJECT_NAME}/checkpoint-{batch_idx:05d}"
        )
        av_save_dir = checkpoint_dir / "av"
        ar_save_dir = checkpoint_dir / "ar"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        av_save_dir.mkdir(parents=True, exist_ok=True)
        ar_save_dir.mkdir(parents=True, exist_ok=True)
        print("saving checkpoint at step: ", batch_idx)
        av_model.save_pretrained(av_save_dir)
        ar_model.save_pretrained(ar_save_dir)
        av_tokenizer.save_pretrained(av_save_dir)
        ar_tokenizer.save_pretrained(ar_save_dir)

        torch.save(
            {
                "batch_idx": batch_idx,
                "av_projector_state_dict": av_projectors.state_dict(),
                "activation_dim": av_checkpoint["activation_dim"],
                "hidden_size": av_checkpoint["hidden_size"],
                "prefix_len": av_checkpoint["prefix_len"],
                "activation_token": activation_token,
                "activation_token_id": activation_token_id,
            },
            av_save_dir / "av_projector.pt",
        )
        torch.save(av_optimizer.state_dict(), av_save_dir / "optimizer.pt")
        torch.save(av_scheduler.state_dict(), av_save_dir / "scheduler.pt")

        torch.save(
            {
                "batch_idx": batch_idx,
                "ar_reconstructor_state_dict": ar_projectors.state_dict(),
                "hidden_size": ar_checkpoint["hidden_size"],
                "activation_dim": ar_checkpoint["activation_dim"],
                "prefix_len": ar_checkpoint["prefix_len"],
                "reconstructor_hidden_size": ar_checkpoint.get(
                    "reconstructor_hidden_size"
                ),
            },
            ar_save_dir / "ar_head.pt",
        )
        torch.save(ar_optimizer.state_dict(), ar_save_dir / "optimizer.pt")
        torch.save(ar_scheduler.state_dict(), ar_save_dir / "scheduler.pt")

        torch.save(
            {
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state": torch.cuda.get_rng_state_all(),
            },
            checkpoint_dir / "rng_state.pt",
        )

        with (checkpoint_dir / "train_state.json").open("w") as f:
            json.dump(
                {
                    "batch_idx": batch_idx,
                },
                f,
                indent=4,
            )

        with (checkpoint_dir / "config.json").open("w") as f:
            json.dump(config, f, indent=4)

        print("saved checkpoint at iteration: ", batch_idx)
        saved_checkpoints.append(checkpoint_dir)

        if len(saved_checkpoints) > MAX_CHECKPOINTS:
            oldest_checkpoint = saved_checkpoints.pop(0)
            if os.path.exists(oldest_checkpoint):
                shutil.rmtree(oldest_checkpoint, ignore_errors=True)

wandb.finish()
