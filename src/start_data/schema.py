from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class PositionMetadata:
    source: str
    game_id: str | None = None
    event: str | None = None
    site: str | None = None
    white: str | None = None
    black: str | None = None
    result: str | None = None
    ply: int | None = None


@dataclass(frozen=True)
class ActivationRef:
    layer: str
    path: str
    shape: list[int]
    dtype: str


@dataclass(frozen=True)
class PureActivationRecord:
    id: str
    fen: str
    side_to_move: str
    split: SplitName
    metadata: PositionMetadata
    activation: ActivationRef | None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


@dataclass(frozen=True)
class PretrainRecord:
    id: str
    fen: str
    side_to_move: str
    split: SplitName
    metadata: PositionMetadata
    activation: ActivationRef | None
    eval: dict[str, Any] | None
    top_moves: list[dict[str, Any]] = field(default_factory=list)
    principal_variations: list[dict[str, Any]] = field(default_factory=list)
    teacher_prompt: str = ""
    teacher_summary: str = ""
    raw_uci: list[str] = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False, sort_keys=True)


def position_id(fen: str, layer: str) -> str:
    digest = hashlib.sha256(f"{fen}|{layer}".encode("utf-8")).hexdigest()
    return digest[:16]


def split_for_id(record_id: str, val_pct: int = 5, test_pct: int = 5) -> SplitName:
    bucket = int(record_id[:8], 16) % 100
    if bucket < test_pct:
        return "test"
    if bucket < test_pct + val_pct:
        return "val"
    return "train"


def write_jsonl(path: Path, records: list[PretrainRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json())
            f.write("\n")


def write_pure_activation_jsonl(path: Path, records: list[PureActivationRecord]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(record.to_json())
            f.write("\n")

