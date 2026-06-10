# see anthropic_method.png
# get the data -> we have this logic already
# pass through AV -> AR
# loss is just msecosine b/w reconstructed and input tensor + KL penalty b/w AV & frozen warm-started model
# backprop through AR and RL through AV

# for each activation (in practice, batch this)
# generate G rollouts of the AV
# for each of the G explanations, reconstruct them w/ AR
# create some rewards from this reconstruction
# update AV based on these rewards, applying a KL divergence penalty
# update AR based on mse/cosine loss between reconstructed and actual activation
import torch.nn as nn
import torch.nn.functional as F
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel
import bitsandbytes as bnb
import wandb
import os
import json
import shutil
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

av_checkpoint_dir = "../../outputs/warm_start/qwen3.5-2b-nla-av/final"
ar_checkpoint_dir = "../../outputs/warm_start/qwen3.5-2b-nla-ar/final"
av_device = "cuda:0"
ar_device = "cuda:1"
frozen_device = "cuda:2"
G = 8
B = 1
COSINE_WEIGHT = 0.25
KL_WEIGHT = 0.15
LR = 1e-5
EPOCHS = 1
MAX_CHECKPOINTS = 3

config = {
    "learning_rate": LR,
    "num_rollouts": G,
    "batch_size": B,
    "epochs": EPOCHS,
    "cosine_weight": COSINE_WEIGHT,
    "kl_weight": KL_WEIGHT,
}

# wandb.login()
# wandb.init(project="lc0-nla", name="nla-train-1", config=config)


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
        jsonl_path = self.activation_dir.parent / "shard-500k.jsonl"
        self.activation_paths = []
        with jsonl_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                if record.get("split") != split:
                    continue
                activation = record.get("activation")
                # reason for this double ../ is the jsonl file assumes root, but i'm gonna run this from /src/nla
                # and i'm too lazy to change all the other ../../'s
                self.activation_paths.append("../../" + activation["path"])

    def __len__(self):
        return len(self.activation_paths)

    def __getitem__(self, index):
        return torch.load(
            self.activation_paths[index], map_location=self.device, weights_only=False
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
av_model = AutoModelForCausalLM.from_pretrained(
    av_checkpoint_dir,
    dtype=torch.bfloat16,
    device_map=av_device,
    trust_remote_code=True,
    attn_implementation="flash_attention_2",
)
av_model.gradient_checkpointing_enable()
av_checkpoint = torch.load(
    f"{av_checkpoint_dir}/av_projector.pt", map_location=av_device
)
av_projectors = AVProjector(
    av_checkpoint["activation_dim"], av_checkpoint["hidden_size"]
).to(av_device, dtype=torch.bfloat16)
av_projectors.load_state_dict(av_checkpoint["av_projector_state_dict"])
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
)
ar_model.gradient_checkpointing_enable()
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
frozen_model.eval()
frozen_model.requires_grad_(False)

saved_checkpoints = []

leela_activation_dataset = LeelaActivationDataset()
train_dataloader = DataLoader(
    dataset=leela_activation_dataset,
    batch_size=B,
    shuffle=True,
    drop_last=True,
)

av_model = torch.compile(av_model, dynamic=True)
ar_model = torch.compile(ar_model, dynamic=True)
frozen_model = torch.compile(frozen_model, dynamic=True)

start_event = torch.cuda.Event(enable_timing=True)
end_event = torch.cuda.Event(enable_timing=True)

for i in range(EPOCHS):
    for batch_idx, leela_activations in enumerate(train_dataloader):
        token_embeddings = av_model.get_input_embeddings()
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
        inputs_embeds = inputs_embeds.repeat_interleave(G, dim=0)
        attention_mask = attention_mask.repeat_interleave(G, dim=0)

        start_event.record()
        with torch.no_grad():
            rollouts = av_model.generate(  # type: ignore
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                max_new_tokens=200,
                pad_token_id=av_tokenizer.eos_token_id,
                do_sample=True,
                temperature=0.6,
                top_p=0.9,
                use_cache=True,
            )
        end_event.record()
        torch.cuda.synchronize()
        print("time taken for generate: ", start_event.elapsed_time(end_event) / 1000)

        generated_attention_mask = (rollouts != av_tokenizer.pad_token_id).long()

        reconstructed = ar_model(
            input_ids=rollouts.to(ar_device),
            attention_mask=generated_attention_mask.to(ar_device),
            return_dict=True,
            use_cache=False,
        )

        final_hidden = reconstructed.last_hidden_state
        ar_mask = generated_attention_mask.to(ar_device)
        last_token_index = ar_mask.sum(dim=1).clamp_min(1) - 1
        batch_index = torch.arange(rollouts.shape[0], device=ar_device)
        final_summary_hidden = final_hidden[batch_index, last_token_index]

        final_summary_hidden = final_summary_hidden.to(dtype=torch.bfloat16)
        activation_prefix = ar_projectors(final_summary_hidden)
        mse_tensor = ((activation_prefix - leela_activations.to(ar_device)) ** 2).mean(
            dim=(1, 2)
        )
        cos_loss = 1 - F.cosine_similarity(
            activation_prefix.view(G, -1),
            leela_activations.to(ar_device).expand(G, -1, -1).view(G, -1),
        )
        ar_loss = mse_tensor.mean() + COSINE_WEIGHT * cos_loss.mean()
        ar_optimizer.zero_grad()
        ar_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(ar_projectors.parameters()) + list(ar_model.parameters()),
            max_norm=1.0,
        )
        ar_optimizer.step()

        r = -mse_tensor - COSINE_WEIGHT * cos_loss
        r = (r - r.mean()) / (r.std() + 1e-8)
        r = r.detach()

        generated_embeds = av_model.get_input_embeddings()(rollouts.to(av_device))
        full_embeds = torch.cat([inputs_embeds, generated_embeds], dim=1)
        full_mask = torch.cat(
            [attention_mask.to(av_device), generated_attention_mask.to(av_device)],
            dim=1,
        )

        av_outputs = av_model(inputs_embeds=full_embeds, attention_mask=full_mask)
        prompt_len = inputs_embeds.shape[1]
        logits = av_outputs.logits[
            :, prompt_len - 1 : -1, :
        ]  # [G, seq_len, vocab_size]
        log_probs = F.log_softmax(logits, dim=-1)  # log softmaxes over vocab_size
        token_ids = rollouts.unsqueeze(-1).to(av_device)  # [G, seq_len, 1]
        chosen_token_log_probs = log_probs.gather(dim=-1, index=token_ids).squeeze(
            -1
        )  # index by G & token id, remove trailing dim to create [G, seq_len] probabilities tensor
        # mask out padding tokens
        chosen_token_log_probs = chosen_token_log_probs * generated_attention_mask.to(
            av_device
        )
        # log probs for each of the G rollouts!
        seq_log_probs = chosen_token_log_probs.sum(dim=-1)

        with torch.no_grad():
            frozen_model_outputs = frozen_model(
                inputs_embeds=full_embeds.detach().to(frozen_device),
                attention_mask=full_mask.to(frozen_device),
                use_cache=False,  # it's one forward pass, so we don't really need a kv cache
            )
            frozen_logits = frozen_model_outputs.logits[:, prompt_len - 1 : -1, :]
            frozen_log_probs = F.log_softmax(frozen_logits, dim=-1)

        frozen_log_probs = frozen_log_probs.to(av_device)
        kl_div_no_reduce = F.kl_div(
            frozen_log_probs, log_probs, reduction="none", log_target=True
        )
        kl_div_per_token = kl_div_no_reduce.sum(dim=-1)  # [G, seq_len]
        response_mask_frozen = generated_attention_mask.to(av_device)
        masked_kl_div = kl_div_per_token * response_mask_frozen
        valid_tokens_per_rollout = response_mask_frozen.sum(dim=-1).clamp_min(1)  # [G]
        kl_div_per_rollout = masked_kl_div.sum(dim=-1) / valid_tokens_per_rollout

        av_loss = (
            -(seq_log_probs * r.to(av_device)).mean()
            + KL_WEIGHT * kl_div_per_rollout.mean()
        )
        av_optimizer.zero_grad()
        av_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(av_projectors.parameters()) + list(av_model.parameters()),
            max_norm=1.0,
        )
        av_optimizer.step()

        absolute_step_index = i * len(train_dataloader) + batch_idx
        # if absolute_step_index % 100 == 0:
        #     wandb.log(
        #         {
        #             "train/av_loss": av_loss.item(),
        #             "train/ar_loss": ar_loss.item(),
        #             "train/kl_div_mean": kl_div_per_rollout.mean().item(),
        #             "train/mse_term": mse_tensor.mean().item(),
        #             "train/cosine_term": cos_loss.mean().item(),
        #             "iteration": absolute_step_index,
        #         }
        #     )
        print(absolute_step_index)
        if absolute_step_index == 11:
            break

        if absolute_step_index > 0 and absolute_step_index % 1000 == 0:
            checkpoint_dir = (
                f"../../outputs/nla_training/checkpoint{absolute_step_index}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            print("saving checkpoint at step: ", absolute_step_index)
            av_model.save_pretrained(f"{checkpoint_dir}/av_model")
            ar_model.save_pretrained(f"{checkpoint_dir}/ar_model")
            av_tokenizer.save_pretrained(f"{checkpoint_dir}/av_tokenizer")

            torch.save(
                {
                    "iteration": i,
                    "av_projector_state_dict": av_projectors.state_dict(),
                    "av_optimizer_state_dict": av_optimizer.state_dict(),
                    "activation_dim": av_checkpoint["activation_dim"],
                    "hidden_size": av_checkpoint["hidden_size"],
                },
                f"{checkpoint_dir}/av_projector.pt",
            )

            torch.save(
                {
                    "iteration": i,
                    "ar_reconstructor_state_dict": ar_projectors.state_dict(),
                    "ar_optimizer_state_dict": ar_optimizer.state_dict(),
                    "hidden_size": ar_checkpoint["hidden_size"],
                    "activation_dim": ar_checkpoint["activation_dim"],
                    "prefix_len": ar_checkpoint["prefix_len"],
                    "reconstructor_hidden_size": ar_checkpoint.get(
                        "reconstructor_hidden_size"
                    ),
                },
                f"{checkpoint_dir}/ar_head.pt",
            )

            torch.save(
                {
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": torch.cuda.get_rng_state_all(),
                },
                f"{checkpoint_dir}/rng_state.pt",
            )

            with open(f"{checkpoint_dir}/config.json", "w") as f:
                json.dump(config, f, indent=4)

            print("saved checkpoint at iteration: ", i)
            saved_checkpoints.append(checkpoint_dir)

            if len(saved_checkpoints) > MAX_CHECKPOINTS:
                oldest_checkpoint = saved_checkpoints.pop(0)
                if os.path.exists(oldest_checkpoint):
                    shutil.rmtree(oldest_checkpoint, ignore_errors=True)

# wandb.finish()
