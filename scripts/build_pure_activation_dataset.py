#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from start_data.activations import DEFAULT_PRETRAIN_LAYER, ActivationConfig, LeelaActivationExtractor
from start_data.positions import DEFAULT_LICHESS_URL, download_lichess_pgn, sample_positions
from start_data.schema import (
    PureActivationRecord,
    position_id,
    split_for_id,
    write_pure_activation_jsonl,
)
from start_data.validate import validate_pure_activation_records, write_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a pure Leela activation dataset (no lc0 analysis or teacher fields)."
    )
    parser.add_argument("--pgn", type=Path, help="Existing .pgn or .pgn.bz2 file.")
    parser.add_argument("--download-url", default=DEFAULT_LICHESS_URL)
    parser.add_argument("--download-to", type=Path, default=ROOT / "data/raw/lichess_sample.pgn.zst")
    parser.add_argument("--download", action="store_true", help="Download the Lichess PGN before sampling.")
    parser.add_argument("--limit", type=int, default=50, help="Number of activations to extract.")
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Skip the first N deterministically sampled positions (e.g. 200000 to avoid reusing prior data).",
    )
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "data/pure_activations/shard-00000.jsonl")
    parser.add_argument(
        "--activation-dir",
        type=Path,
        default=ROOT / "data/pure_activations/activations_encoder14_ln2_betas",
    )
    parser.add_argument("--onnx", type=Path, default=ROOT / "lc0.onnx")
    parser.add_argument("--layer", default=DEFAULT_PRETRAIN_LAYER)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument("--stats-out", type=Path, default=ROOT / "data/pure_activations/stats-00000.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pgn_path = resolve_pgn(args)
    total_needed = args.skip + args.limit
    positions = sample_positions(pgn_path, limit=total_needed, max_games=args.max_games)
    if len(positions) < total_needed:
        raise SystemExit(
            f"Only sampled {len(positions)} positions, need {total_needed} "
            f"(skip={args.skip} + limit={args.limit}). Increase --max-games."
        )
    positions = positions[args.skip :]

    activation_extractor = None
    if not args.skip_activations:
        activation_extractor = LeelaActivationExtractor(
            ActivationConfig(onnx_path=args.onnx, layer=args.layer, device=args.device)
        )

    records: list[PureActivationRecord] = []
    for index, position in enumerate(positions, start=1):
        record_id = position_id(position.fen, args.layer)
        activation_ref = None
        if activation_extractor is not None:
            activation_path = args.activation_dir / f"{record_id}.pt"
            activation_ref = activation_extractor.extract_to_file(position.fen, activation_path)
        records.append(
            PureActivationRecord(
                id=record_id,
                fen=position.fen,
                side_to_move="white" if position.board.turn else "black",
                split=split_for_id(record_id),
                metadata=position.metadata,
                activation=activation_ref,
            )
        )
        print(f"[{index}/{len(positions)}] wrote record {record_id}", flush=True)

    write_pure_activation_jsonl(args.out, records)
    stats = validate_pure_activation_records(args.out)
    write_stats(args.stats_out, stats)
    print(json.dumps(stats, indent=2, sort_keys=True))
    if stats["errors"]:
        raise SystemExit("Validation failed")


def resolve_pgn(args: argparse.Namespace) -> Path:
    if args.download:
        return download_lichess_pgn(args.download_url, args.download_to)
    if args.pgn is not None:
        return args.pgn
    if args.download_to.exists():
        return args.download_to
    raise SystemExit("Provide --pgn or use --download to fetch a Lichess PGN.")


if __name__ == "__main__":
    main()
