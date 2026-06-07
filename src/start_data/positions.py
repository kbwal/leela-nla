from __future__ import annotations

import bz2
import io
import itertools
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chess
import chess.pgn
import requests
import zstandard as zstd

from .schema import PositionMetadata

DEFAULT_LICHESS_URL = (
    "https://database.lichess.org/standard/lichess_db_standard_rated_2013-01.pgn.zst"
)


@dataclass(frozen=True)
class SampledPosition:
    fen: str
    board: chess.Board
    metadata: PositionMetadata


def download_lichess_pgn(
    url: str, output_path: Path, max_bytes: int | None = None
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        written = 0
        with output_path.open("wb") as f:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if not chunk:
                    continue
                if max_bytes is not None and written + len(chunk) > max_bytes:
                    chunk = chunk[: max_bytes - written]
                f.write(chunk)
                written += len(chunk)
                if max_bytes is not None and written >= max_bytes:
                    break
    return output_path


def open_pgn_text(path: Path) -> io.TextIOBase:
    if path.suffix == ".bz2":
        return io.TextIOWrapper(
            bz2.open(path, "rb"), encoding="utf-8", errors="replace"
        )
    if path.suffix == ".zst":
        compressed = path.open("rb")
        stream = zstd.ZstdDecompressor().stream_reader(compressed)
        return io.TextIOWrapper(stream, encoding="utf-8", errors="replace")
    return path.open("r", encoding="utf-8", errors="replace")


def iter_games(path: Path, max_games: int | None = None) -> Iterable[chess.pgn.Game]:
    with open_pgn_text(path) as pgn:
        for _ in itertools.count():
            if max_games is not None and _ >= max_games:
                return
            game = chess.pgn.read_game(pgn)
            if game is None:
                return
            yield game


def game_id_from_headers(headers: chess.pgn.Headers) -> str | None:
    site = headers.get("Site", "")
    if site:
        return site.rstrip("/").split("/")[-1]
    return headers.get("LichessId") or None


def iter_positions_from_game(
    game: chess.pgn.Game,
    plies: set[int] | None = None,
    min_ply: int = 8,
    max_ply: int | None = None,
) -> Iterable[SampledPosition]:
    board = game.board()
    headers = game.headers
    game_id = game_id_from_headers(headers)
    for ply, move in enumerate(game.mainline_moves(), start=1):
        board.push(move)
        if ply < min_ply:
            continue
        if max_ply is not None and ply > max_ply:
            break
        if plies is not None and ply not in plies:
            continue
        yield SampledPosition(
            fen=board.fen(),
            board=board.copy(stack=False),
            metadata=PositionMetadata(
                source="lichess",
                game_id=game_id,
                event=headers.get("Event"),
                site=headers.get("Site"),
                white=headers.get("White"),
                black=headers.get("Black"),
                result=headers.get("Result"),
                ply=ply,
            ),
        )


def sample_positions(
    pgn_path: Path,
    limit: int,
    max_games: int | None = None,
    plies: Iterable[int] = (
        12,
        13,
        20,
        21,
        32,
        33,
        40,
        41,
        56,
        57,
        64,
        65,
        72,
        73,
        84,
        85,
        96,
        97,
        108,
        109,
        116,
        117,
        128,
        129,
        136,
        137,
        150,
        151,
    ),
    min_ply: int = 8,
    max_ply: int | None = 100,
) -> list[SampledPosition]:
    wanted_plies = set(plies)
    sampled: list[SampledPosition] = []
    seen_fens: set[str] = set()
    for game in iter_games(pgn_path, max_games=max_games):
        for position in iter_positions_from_game(
            game,
            plies=wanted_plies,
            min_ply=min_ply,
            max_ply=max_ply,
        ):
            key = " ".join(position.fen.split(" ")[:4])
            if key in seen_fens:
                continue
            seen_fens.add(key)
            sampled.append(position)
            if len(sampled) >= limit:
                return sampled
    return sampled
