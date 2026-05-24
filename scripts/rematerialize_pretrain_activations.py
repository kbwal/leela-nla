#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from start_data.activations import ActivationConfig, LeelaActivationExtractor
from start_data.schema import position_id, split_for_id
from start_data.validate import validate_records, write_stats


DEFAULT_LAYER = "encoder14/ln2/betas"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rewrite existing pretrain JSONL shards with activations extracted from a new ONNX layer."
        )
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Input JSONL shard to read. Records are streamed line-by-line.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output JSONL shard with updated ids, splits, and activation refs.",
    )
    parser.add_argument("--onnx", type=Path, default=ROOT / "lc0.onnx")
    parser.add_argument("--layer", default=DEFAULT_LAYER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--activation-dir",
        type=Path,
        default=ROOT / "data/pretrain/activations_encoder14_ln2_betas",
        help="Directory for the rematerialized .pt activation files.",
    )
    parser.add_argument(
        "--stats-out",
        type=Path,
        help="Optional path for validation stats. Validation loads every activation file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Process at most this many additional input rows. Useful for smoke tests.",
    )
    parser.add_argument("--log-every", type=int, default=1000)
    parser.add_argument(
        "--overwrite-activations",
        action="store_true",
        help="Regenerate activation files even when the target .pt file already exists.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Refuse to append to an existing output instead of resuming by row count.",
    )
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


def load_activation_ref(path: Path, layer: str) -> dict[str, Any]:
    tensor = torch.load(path, map_location="cpu", weights_only=False)
    return {
        "layer": layer,
        "path": str(path),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype).replace("torch.", ""),
    }


def extract_activation_ref(
    *,
    extractor: LeelaActivationExtractor,
    fen: str,
    output_path: Path,
    layer: str,
    overwrite: bool,
) -> dict[str, Any]:
    if output_path.exists() and not overwrite:
        return load_activation_ref(output_path, layer)

    activation = extractor.extract_to_file(fen, output_path)
    return {
        "layer": activation.layer,
        "path": activation.path,
        "shape": activation.shape,
        "dtype": activation.dtype,
    }


def rematerialize(args: argparse.Namespace) -> int:
    if args.input.resolve() == args.output.resolve():
        raise SystemExit("Refusing to write in-place. Use a separate --output path.")
    if args.no_resume and args.output.exists():
        raise SystemExit(f"{args.output} already exists. Remove it or omit --no-resume to append.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.activation_dir.mkdir(parents=True, exist_ok=True)

    resume_lines = 0 if args.no_resume else count_jsonl(args.output)
    if resume_lines:
        print(f"Resuming: {args.output} already has {resume_lines} JSONL rows.", flush=True)

    extractor = LeelaActivationExtractor(
        ActivationConfig(onnx_path=args.onnx, layer=args.layer, device=args.device)
    )

    processed = 0
    seen = 0
    mode = "a" if resume_lines else "w"
    with args.input.open("r", encoding="utf-8") as input_handle, args.output.open(
        mode, encoding="utf-8"
    ) as output_handle:
        for line_no, line in enumerate(input_handle, start=1):
            if not line.strip():
                continue
            if seen < resume_lines:
                seen += 1
                continue

            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise SystemExit(f"{args.input}:{line_no}: invalid JSON: {exc}") from exc

            fen = record.get("fen")
            if not isinstance(fen, str):
                raise SystemExit(f"{args.input}:{line_no}: record is missing string field 'fen'")

            record_id = position_id(fen, args.layer)
            activation_path = (args.activation_dir / f"{record_id}.pt").resolve()
            record["id"] = record_id
            record["split"] = split_for_id(record_id)
            record["activation"] = extract_activation_ref(
                extractor=extractor,
                fen=fen,
                output_path=activation_path,
                layer=args.layer,
                overwrite=args.overwrite_activations,
            )

            output_handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
            processed += 1
            seen += 1

            if args.log_every > 0 and processed % args.log_every == 0:
                print(f"Processed {processed} new records ({seen} total rows).", flush=True)
            if args.limit is not None and processed >= args.limit:
                break

    return processed


def main() -> None:
    args = parse_args()
    processed = rematerialize(args)
    print(f"Done. Wrote {processed} new records to {args.output}.", flush=True)

    if args.stats_out is not None:
        stats = validate_records(args.output)
        write_stats(args.stats_out, stats)
        print(json.dumps(stats, indent=2, sort_keys=True), flush=True)
        if stats["errors"]:
            raise SystemExit("Validation failed")


if __name__ == "__main__":
    main()
