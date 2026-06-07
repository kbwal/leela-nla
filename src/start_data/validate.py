from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import chess
import torch


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
    return records


def validate_records(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    eval_buckets: Counter[str] = Counter()
    prompt_lengths: list[int] = []
    summary_lengths: list[int] = []
    plies: list[int] = []

    for idx, record in enumerate(records):
        prefix = f"record {idx} ({record.get('id', 'missing-id')})"
        fen = record.get("fen")
        if not isinstance(fen, str):
            errors.append(f"{prefix}: missing fen")
            continue
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                errors.append(f"{prefix}: invalid board")
        except ValueError as exc:
            errors.append(f"{prefix}: invalid fen: {exc}")
            continue

        split_counts[str(record.get("split"))] += 1
        side_counts[str(record.get("side_to_move"))] += 1
        prompt_lengths.append(len(record.get("teacher_prompt", "")))
        summary_lengths.append(len(record.get("teacher_summary", "")))
        metadata = record.get("metadata") or {}
        if isinstance(metadata.get("ply"), int):
            plies.append(metadata["ply"])

        activation = record.get("activation")
        if activation:
            activation_path = Path(activation.get("path", ""))
            if not activation_path.exists():
                errors.append(f"{prefix}: activation path does not exist: {activation_path}")
            else:
                tensor = torch.load(activation_path, map_location="cpu", weights_only=False)
                shape = list(tensor.shape)
                expected = activation.get("shape")
                if expected is not None and shape != expected:
                    errors.append(f"{prefix}: activation shape {shape} != recorded {expected}")
                shape_counts[str(shape)] += 1

        eval_info = record.get("eval")
        if isinstance(eval_info, dict) and eval_info.get("type") == "cp":
            value = int(eval_info.get("value", 0))
            bucket = f"{round(value / 100) * 100:+d}cp"
            eval_buckets[bucket] += 1
        elif isinstance(eval_info, dict) and eval_info.get("type") == "mate":
            eval_buckets["mate"] += 1
        else:
            eval_buckets["missing"] += 1

        for pv in record.get("principal_variations", []):
            san = pv.get("pv_san")
            uci = pv.get("pv_uci")
            if not isinstance(san, list) or not isinstance(uci, list):
                errors.append(f"{prefix}: malformed PV entry")

    return {
        "records": len(records),
        "errors": errors,
        "split_counts": dict(split_counts),
        "side_counts": dict(side_counts),
        "activation_shapes": dict(shape_counts),
        "eval_buckets": dict(eval_buckets),
        "ply_min": min(plies) if plies else None,
        "ply_max": max(plies) if plies else None,
        "prompt_length_avg": sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0,
        "summary_length_avg": sum(summary_lengths) / len(summary_lengths) if summary_lengths else 0,
    }


def validate_pure_activation_records(path: Path) -> dict[str, Any]:
    records = load_jsonl(path)
    errors: list[str] = []
    split_counts: Counter[str] = Counter()
    side_counts: Counter[str] = Counter()
    shape_counts: Counter[str] = Counter()
    plies: list[int] = []

    for idx, record in enumerate(records):
        prefix = f"record {idx} ({record.get('id', 'missing-id')})"
        fen = record.get("fen")
        if not isinstance(fen, str):
            errors.append(f"{prefix}: missing fen")
            continue
        try:
            board = chess.Board(fen)
            if not board.is_valid():
                errors.append(f"{prefix}: invalid board")
        except ValueError as exc:
            errors.append(f"{prefix}: invalid fen: {exc}")
            continue

        split_counts[str(record.get("split"))] += 1
        side_counts[str(record.get("side_to_move"))] += 1
        metadata = record.get("metadata") or {}
        if isinstance(metadata.get("ply"), int):
            plies.append(metadata["ply"])

        activation = record.get("activation")
        if activation:
            activation_path = Path(activation.get("path", ""))
            if not activation_path.exists():
                errors.append(f"{prefix}: activation path does not exist: {activation_path}")
            else:
                tensor = torch.load(activation_path, map_location="cpu", weights_only=False)
                shape = list(tensor.shape)
                expected = activation.get("shape")
                if expected is not None and shape != expected:
                    errors.append(f"{prefix}: activation shape {shape} != recorded {expected}")
                shape_counts[str(shape)] += 1
        else:
            errors.append(f"{prefix}: missing activation")

        for forbidden in (
            "eval",
            "top_moves",
            "principal_variations",
            "teacher_prompt",
            "teacher_summary",
            "raw_uci",
        ):
            if forbidden in record:
                errors.append(f"{prefix}: unexpected field {forbidden}")

    return {
        "records": len(records),
        "errors": errors,
        "split_counts": dict(split_counts),
        "side_counts": dict(side_counts),
        "activation_shapes": dict(shape_counts),
        "ply_min": min(plies) if plies else None,
        "ply_max": max(plies) if plies else None,
    }


def write_stats(stats_path: Path, stats: dict[str, Any]) -> None:
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    stats_path.write_text(json.dumps(stats, indent=2, sort_keys=True), encoding="utf-8")

