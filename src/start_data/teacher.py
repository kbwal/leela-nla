from __future__ import annotations

import json
from typing import Any

import chess


SYSTEM_PROMPT = """You are writing supervised warm-start targets for a natural-language autoencoder over Leela Chess Zero activations.

Your job is to produce concise, coherent chess-language summaries from the evidence provided. Use normal English and standard chess terms. Do not claim Leela saw anything that is not in the evidence. If the evidence is thin, say what can be inferred cautiously.
"""


def build_teacher_prompt(
    fen: str,
    eval_info: dict[str, Any] | None,
    top_moves: list[dict[str, Any]],
    principal_variations: list[dict[str, Any]],
) -> str:
    board = chess.Board(fen)
    evidence = {
        "fen": fen,
        "side_to_move": "white" if board.turn == chess.WHITE else "black",
        "legal_position": board.is_valid(),
        "eval": eval_info,
        "top_moves": top_moves,
        "principal_variations": [
            {
                "rank": pv.get("multipv"),
                "score": pv.get("score"),
                "line_san": pv.get("pv_san"),
                "line_uci": pv.get("pv_uci"),
            }
            for pv in principal_variations
        ],
    }
    return (
        f"{SYSTEM_PROMPT}\n\n"
        "Write one short training target with these sections:\n"
        "1. Position gist: one or two sentences about the strategic/tactical situation.\n"
        "2. Leela preference: explain the top candidate move(s) and PV theme.\n"
        "3. Key features: mention material, king safety, pawn structure, threats, or endgame factors only when supported.\n"
        "4. Uncertainty: briefly flag if the eval/PVs are close or if the evidence is insufficient.\n\n"
        "Keep it under 180 words. Avoid bullet lists unless they are natural. Do not include raw JSON in the answer.\n\n"
        "Evidence JSON:\n"
        f"{json.dumps(evidence, ensure_ascii=False, indent=2, sort_keys=True)}"
    )


def empty_teacher_summary() -> str:
    return ""

