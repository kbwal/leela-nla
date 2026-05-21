#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from start_data.activations import ActivationConfig, LeelaActivationExtractor
from start_data.lc0_uci import Lc0Config, Lc0Engine
from start_data.positions import DEFAULT_LICHESS_URL, download_lichess_pgn, sample_positions
from start_data.schema import PretrainRecord, position_id, split_for_id, write_jsonl
from start_data.teacher import build_teacher_prompt, empty_teacher_summary
from start_data.validate import validate_records, write_stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Leela NLA pretraining data.")
    parser.add_argument("--pgn", type=Path, help="Existing .pgn or .pgn.bz2 file.")
    parser.add_argument("--download-url", default=DEFAULT_LICHESS_URL)
    parser.add_argument("--download-to", type=Path, default=ROOT / "data/raw/lichess_sample.pgn.zst")
    parser.add_argument("--download", action="store_true", help="Download the Lichess PGN before sampling.")
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--max-games", type=int, default=200)
    parser.add_argument("--out", type=Path, default=ROOT / "data/pretrain/shard-00000.jsonl")
    parser.add_argument("--activation-dir", type=Path, default=ROOT / "data/pretrain/activations")
    parser.add_argument("--onnx", type=Path, default=ROOT / "lc0.onnx")
    parser.add_argument("--layer", default="encoder0/mha/out/skip")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--skip-activations", action="store_true")
    parser.add_argument("--lc0", type=Path, default=Path("/home/kushalb/lc0/build/release/lc0"))
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--multipv", type=int, default=3)
    parser.add_argument("--nodes", type=int, default=800)
    parser.add_argument("--movetime-ms", type=int)
    parser.add_argument("--skip-lc0", action="store_true")
    parser.add_argument("--stats-out", type=Path, default=ROOT / "data/pretrain/stats-00000.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pgn_path = resolve_pgn(args)
    positions = sample_positions(pgn_path, limit=args.limit, max_games=args.max_games)
    if not positions:
        raise SystemExit(f"No positions sampled from {pgn_path}")

    activation_extractor = None
    if not args.skip_activations:
        activation_extractor = LeelaActivationExtractor(
            ActivationConfig(onnx_path=args.onnx, layer=args.layer, device=args.device)
        )

    lc0_engine = None
    if not args.skip_lc0:
        lc0_engine = Lc0Engine(
            Lc0Config(
                engine_path=args.lc0,
                weights_path=args.weights,
                multipv=args.multipv,
                nodes=args.nodes if args.movetime_ms is None else None,
                movetime_ms=args.movetime_ms,
            )
        )

    records: list[PretrainRecord] = []
    try:
        for index, position in enumerate(positions, start=1):
            record_id = position_id(position.fen, args.layer)
            analysis = analyze_position(lc0_engine, position.fen)
            activation_ref = None
            if activation_extractor is not None:
                activation_path = args.activation_dir / f"{record_id}.pt"
                activation_ref = activation_extractor.extract_to_file(position.fen, activation_path)
            prompt = build_teacher_prompt(
                fen=position.fen,
                eval_info=analysis.get("eval"),
                top_moves=analysis.get("top_moves", []),
                principal_variations=analysis.get("principal_variations", []),
            )
            records.append(
                PretrainRecord(
                    id=record_id,
                    fen=position.fen,
                    side_to_move="white" if position.board.turn else "black",
                    split=split_for_id(record_id),
                    metadata=position.metadata,
                    activation=activation_ref,
                    eval=analysis.get("eval"),
                    top_moves=analysis.get("top_moves", []),
                    principal_variations=analysis.get("principal_variations", []),
                    teacher_prompt=prompt,
                    teacher_summary=empty_teacher_summary(),
                    raw_uci=analysis.get("raw_uci", []),
                )
            )
            print(f"[{index}/{len(positions)}] wrote record {record_id}", flush=True)
    finally:
        if lc0_engine is not None:
            lc0_engine.close()

    write_jsonl(args.out, records)
    stats = validate_records(args.out)
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


def analyze_position(engine: Lc0Engine | None, fen: str) -> dict:
    if engine is None:
        return {
            "eval": None,
            "top_moves": [],
            "principal_variations": [],
            "raw_uci": [],
        }
    return engine.analyze_fen(fen)


if __name__ == "__main__":
    main()

