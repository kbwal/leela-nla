#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = (
    "/scratch/hub/models--Intel--gemma-4-31B-it-int4-AutoRound/"
    "snapshots/a428c96a57976947b0f12735f0cf5fcae69019ad"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fill teacher_summary in a pretrain JSONL shard using vLLM offline batching."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=ROOT / "data/pretrain/shard-00000_reformatted.jsonl",
        help="Input JSONL with teacher_prompt fields.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "data/pretrain/shard-00000_teacher.jsonl",
        help="Output JSONL with teacher_summary filled. Existing files are resumed by line count.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Local HF/vLLM model path.")
    parser.add_argument("--batch-size", type=int, default=64, help="Prompts submitted per vLLM call.")
    parser.add_argument("--max-new-tokens", type=int, default=240)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallel degree. Gemma4 has 32 attention heads, so 5 is not valid.",
    )
    parser.add_argument(
        "--pipeline-parallel-size",
        type=int,
        default=5,
        help="Pipeline parallel degree. Default uses all five 3090s without splitting heads.",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.92)
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=4096,
        help="Keep this near prompt+generation length to maximize KV-cache capacity.",
    )
    parser.add_argument("--max-num-seqs", type=int, default=128)
    parser.add_argument("--max-num-batched-tokens", type=int, default=65536)
    parser.add_argument("--dtype", default="bfloat16")
    parser.add_argument(
        "--quantization",
        default=None,
        help="Optional vLLM quantization override, e.g. auto-round. By default vLLM reads config.json.",
    )
    parser.add_argument("--trust-remote-code", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enable-prefix-caching", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--enforce-eager", action="store_true")
    parser.add_argument(
        "--raw-prompt",
        action="store_true",
        help="Do not wrap teacher_prompt in the model chat template.",
    )
    parser.add_argument(
        "--overwrite-summaries",
        action="store_true",
        help="Regenerate records that already have a non-empty teacher_summary.",
    )
    parser.add_argument("--limit", type=int, help="Process at most this many additional input rows.")
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args()


def count_jsonl(path: Path) -> int:
    if not path.exists():
        return 0

    count = 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(
                    f"{path} contains invalid JSONL at output line {count + 1}: {exc}. "
                    "Move or repair the output file before resuming."
                ) from exc
            count += 1
    return count


def clean_summary(text: str) -> str:
    text = text.strip()
    for token in (
        "<eos>",
        "<turn|>",
        "<|turn>",
        "<channel|>",
        "<|channel>final",
        "<|channel>analysis",
        "<|channel>thought",
    ):
        text = text.replace(token, "")
    return text.strip()


def make_render_prompt(tokenizer: Any, raw_prompt: bool):
    warned = False

    def render(prompt: str) -> str:
        nonlocal warned
        if raw_prompt:
            return prompt
        try:
            return tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception as exc:  # noqa: BLE001 - fallback keeps long jobs from dying on templates.
            if not warned:
                print(
                    f"Warning: chat template failed ({exc!r}); falling back to raw prompts.",
                    file=sys.stderr,
                    flush=True,
                )
                warned = True
            return prompt

    return render


def build_llm(args: argparse.Namespace):
    try:
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise SystemExit(
            "Could not import vLLM in this Python environment. This can mean vLLM is missing, "
            f"or that its CUDA/Torch binary dependencies are mismatched. Import error: {exc}"
        ) from exc

    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "tensor_parallel_size": args.tensor_parallel_size,
        "pipeline_parallel_size": args.pipeline_parallel_size,
        "dtype": args.dtype,
        "trust_remote_code": args.trust_remote_code,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "max_model_len": args.max_model_len,
        "max_num_seqs": args.max_num_seqs,
        "max_num_batched_tokens": args.max_num_batched_tokens,
        "enable_prefix_caching": args.enable_prefix_caching,
        "enforce_eager": args.enforce_eager,
        "seed": args.seed,
    }
    if args.quantization:
        llm_kwargs["quantization"] = args.quantization

    sampling = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        skip_special_tokens=True,
    )
    return LLM(**llm_kwargs), sampling


def flush_batch(
    *,
    llm: Any,
    sampling: Any,
    output_handle: Any,
    records: list[dict[str, Any]],
    prompts: list[str],
) -> int:
    if not records:
        return 0

    outputs = llm.generate(prompts, sampling)
    if len(outputs) != len(records):
        raise RuntimeError(f"vLLM returned {len(outputs)} outputs for {len(records)} prompts")

    for record, generated in zip(records, outputs, strict=True):
        record["teacher_summary"] = clean_summary(generated.outputs[0].text)
        output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    output_handle.flush()
    return len(records)


def main() -> None:
    args = parse_args()

    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Refusing to write in-place. Use a separate --output path.")
    if args.tensor_parallel_size == 5:
        raise SystemExit("Gemma4 has 32 attention heads, so --tensor-parallel-size 5 is invalid.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    resume_lines = count_jsonl(args.output)
    if resume_lines:
        print(f"Resuming: {args.output} already has {resume_lines} JSONL rows.", flush=True)

    llm, sampling = build_llm(args)
    render_prompt = make_render_prompt(llm.get_tokenizer(), args.raw_prompt)

    processed = 0
    generated = 0
    skipped_existing = 0
    started = time.time()
    batch_records: list[dict[str, Any]] = []
    batch_prompts: list[str] = []

    with args.input.open("r", encoding="utf-8") as input_handle, args.output.open(
        "a", encoding="utf-8"
    ) as output_handle:
        for input_index, line in enumerate(input_handle):
            if input_index < resume_lines:
                continue
            if args.limit is not None and processed >= args.limit:
                break
            if not line.strip():
                continue

            record = json.loads(line)
            processed += 1

            if record.get("teacher_summary") and not args.overwrite_summaries:
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                skipped_existing += 1
            else:
                prompt = record.get("teacher_prompt")
                if not prompt:
                    raise ValueError(f"Missing teacher_prompt on input row {input_index + 1}")
                batch_records.append(record)
                batch_prompts.append(render_prompt(prompt))

            if len(batch_records) >= args.batch_size:
                generated += flush_batch(
                    llm=llm,
                    sampling=sampling,
                    output_handle=output_handle,
                    records=batch_records,
                    prompts=batch_prompts,
                )
                batch_records.clear()
                batch_prompts.clear()

            if processed % args.log_every == 0:
                elapsed = max(time.time() - started, 1e-6)
                print(
                    f"rows={processed} generated={generated} existing={skipped_existing} "
                    f"rate={processed / elapsed:.2f} rows/s",
                    flush=True,
                )

        generated += flush_batch(
            llm=llm,
            sampling=sampling,
            output_handle=output_handle,
            records=batch_records,
            prompts=batch_prompts,
        )

    elapsed = max(time.time() - started, 1e-6)
    print(
        f"Done. rows={processed} generated={generated} existing={skipped_existing} "
        f"elapsed={elapsed / 60:.1f} min output={args.output}",
        flush=True,
    )


if __name__ == "__main__":
    main()
