from __future__ import annotations

import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess


INFO_SCORE_RE = re.compile(r"score\s+(cp|mate)\s+(-?\d+)")


@dataclass(frozen=True)
class Lc0Config:
    engine_path: Path = Path("/home/kushalb/lc0/build/release/lc0")
    weights_path: Path | None = None
    multipv: int = 3
    nodes: int | None = 800
    movetime_ms: int | None = None


class Lc0Engine:
    def __init__(self, config: Lc0Config) -> None:
        self.config = config
        cmd = [str(config.engine_path)]
        if config.weights_path is not None:
            cmd.extend(["--weights", str(config.weights_path)])
        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._read_until("uciok", timeout=30)
        self._set_option("MultiPV", str(config.multipv))
        self._send("isready")
        self._read_until("readyok", timeout=30)

    def close(self) -> None:
        if self.process.poll() is None:
            try:
                self._send("quit")
                self.process.wait(timeout=5)
            except Exception:
                self.process.kill()

    def __enter__(self) -> "Lc0Engine":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def _send(self, command: str) -> None:
        if self.process.stdin is None:
            raise RuntimeError("lc0 stdin is closed")
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def _read_line(self, timeout: float) -> str:
        if self.process.stdout is None:
            raise RuntimeError("lc0 stdout is closed")
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self.process.stdout.readline()
            if line:
                return line.rstrip("\n")
            if self.process.poll() is not None:
                raise RuntimeError(f"lc0 exited with code {self.process.returncode}")
        raise TimeoutError("Timed out waiting for lc0 output")

    def _read_until(self, marker: str, timeout: float) -> list[str]:
        lines: list[str] = []
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            line = self._read_line(timeout=max(0.1, deadline - time.monotonic()))
            lines.append(line)
            if marker in line:
                return lines
        raise TimeoutError(f"Timed out waiting for {marker!r}")

    def _set_option(self, name: str, value: str) -> None:
        self._send(f"setoption name {name} value {value}")

    def analyze_fen(self, fen: str) -> dict[str, Any]:
        self._send(f"position fen {fen}")
        if self.config.nodes is not None:
            self._send(f"go nodes {self.config.nodes}")
        elif self.config.movetime_ms is not None:
            self._send(f"go movetime {self.config.movetime_ms}")
        else:
            self._send("go nodes 800")

        raw: list[str] = []
        while True:
            line = self._read_line(timeout=120)
            raw.append(line)
            if line.startswith("bestmove "):
                break
        return parse_uci_analysis(fen, raw)


def parse_uci_analysis(fen: str, raw_lines: list[str]) -> dict[str, Any]:
    board = chess.Board(fen)
    latest_by_multipv: dict[int, dict[str, Any]] = {}
    bestmove: str | None = None
    for line in raw_lines:
        if line.startswith("bestmove "):
            parts = line.split()
            bestmove = parts[1] if len(parts) > 1 else None
            continue
        if not line.startswith("info "):
            continue
        parts = line.split()
        multipv = _read_int_after(parts, "multipv") or 1
        pv_index = parts.index("pv") if "pv" in parts else -1
        pv = parts[pv_index + 1 :] if pv_index >= 0 else []
        if not pv:
            continue
        score = _parse_score(line)
        visits = _read_int_after(parts, "nodes") or _read_int_after(parts, "nps")
        entry = {
            "multipv": multipv,
            "score": score,
            "nodes_or_nps": visits,
            "pv_uci": pv,
            "pv_san": pv_to_san(board, pv),
            "move_uci": pv[0],
            "move_san": move_to_san(board, pv[0]),
            "raw": line,
        }
        latest_by_multipv[multipv] = entry

    pvs = [latest_by_multipv[k] for k in sorted(latest_by_multipv)]
    top_moves = [
        {
            "rank": pv["multipv"],
            "uci": pv["move_uci"],
            "san": pv["move_san"],
            "score": pv["score"],
            "nodes_or_nps": pv["nodes_or_nps"],
        }
        for pv in pvs
    ]
    return {
        "eval": pvs[0]["score"] if pvs else None,
        "bestmove": bestmove,
        "top_moves": top_moves,
        "principal_variations": pvs,
        "raw_uci": raw_lines,
    }


def _read_int_after(parts: list[str], key: str) -> int | None:
    if key not in parts:
        return None
    idx = parts.index(key)
    if idx + 1 >= len(parts):
        return None
    try:
        return int(parts[idx + 1])
    except ValueError:
        return None


def _parse_score(line: str) -> dict[str, int | str] | None:
    match = INFO_SCORE_RE.search(line)
    if match is None:
        return None
    kind, value = match.groups()
    return {"type": kind, "value": int(value)}


def move_to_san(board: chess.Board, move_uci: str) -> str | None:
    try:
        move = chess.Move.from_uci(move_uci)
        if move not in board.legal_moves:
            return None
        return board.san(move)
    except ValueError:
        return None


def pv_to_san(board: chess.Board, pv_uci: list[str]) -> list[str]:
    tmp = board.copy(stack=False)
    san: list[str] = []
    for move_uci in pv_uci:
        try:
            move = chess.Move.from_uci(move_uci)
        except ValueError:
            break
        if move not in tmp.legal_moves:
            break
        san.append(tmp.san(move))
        tmp.push(move)
    return san

