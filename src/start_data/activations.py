from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chess
import torch
from onnx2torch import convert

from .schema import ActivationRef

DEFAULT_PRETRAIN_LAYER = "encoder14/ln2/betas"


PIECE_ORDER = [
    chess.PAWN,
    chess.KNIGHT,
    chess.BISHOP,
    chess.ROOK,
    chess.QUEEN,
    chess.KING,
]


@dataclass(frozen=True)
class ActivationConfig:
    onnx_path: Path = Path("/home/kushalb/leela-nla/lc0.onnx")
    layer: str = DEFAULT_PRETRAIN_LAYER
    device: str = "cpu"


class LeelaActivationExtractor:
    def __init__(self, config: ActivationConfig) -> None:
        self.config = config
        self.model = convert(str(config.onnx_path)).to(config.device)
        self.model.eval()
        self.activations: dict[str, torch.Tensor] = {}
        self._register_hook(config.layer)

    def _register_hook(self, layer: str) -> None:
        for name, module in self.model.named_modules():
            if name == layer:
                module.register_forward_hook(self._capture(layer))
                return
        available = [name for name, _ in self.model.named_modules()]
        preview = ", ".join(available[:20])
        raise ValueError(f"Layer {layer!r} not found. First available modules: {preview}")

    def _capture(self, name: str) -> Any:
        def hook(_model: Any, _input: Any, output: torch.Tensor) -> None:
            self.activations[name] = output.detach().cpu()

        return hook

    @torch.no_grad()
    def extract(self, fen: str) -> torch.Tensor:
        self.activations.clear()
        planes = encode_fen_planes(fen).to(self.config.device)
        _ = self.model(planes)
        if self.config.layer not in self.activations:
            raise RuntimeError(f"No activation captured for {self.config.layer}")
        return self.activations[self.config.layer]

    def extract_to_file(self, fen: str, output_path: Path) -> ActivationRef:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        tensor = self.extract(fen)
        torch.save(tensor, output_path)
        return ActivationRef(
            layer=self.config.layer,
            path=str(output_path),
            shape=list(tensor.shape),
            dtype=str(tensor.dtype).replace("torch.", ""),
        )


def encode_fen_planes(fen: str) -> torch.Tensor:
    """Best-effort LCZero-style 112-plane encoding from the current position.

    Planes 0-11 encode current side pieces then opponent pieces from the side-to-move
    perspective. Historical planes are left empty until we add PGN-history features.
    """
    board = chess.Board(fen)
    planes = torch.zeros((112, 8, 8), dtype=torch.float32)
    perspective = board.turn

    for square, piece in board.piece_map().items():
        plane = piece_plane(piece, perspective)
        row, col = square_to_plane_coords(square, perspective)
        planes[plane, row, col] = 1.0

    aux_start = 104
    planes[aux_start + 0].fill_(1.0 if board.turn == chess.WHITE else 0.0)
    planes[aux_start + 1].fill_(1.0 if board.has_kingside_castling_rights(chess.WHITE) else 0.0)
    planes[aux_start + 2].fill_(1.0 if board.has_queenside_castling_rights(chess.WHITE) else 0.0)
    planes[aux_start + 3].fill_(1.0 if board.has_kingside_castling_rights(chess.BLACK) else 0.0)
    planes[aux_start + 4].fill_(1.0 if board.has_queenside_castling_rights(chess.BLACK) else 0.0)
    planes[aux_start + 5].fill_(min(board.halfmove_clock / 100.0, 1.0))
    planes[aux_start + 6].fill_(min(board.fullmove_number / 200.0, 1.0))
    planes[aux_start + 7].fill_(1.0)
    return planes.unsqueeze(0)


def piece_plane(piece: chess.Piece, perspective: chess.Color) -> int:
    type_offset = PIECE_ORDER.index(piece.piece_type)
    color_offset = 0 if piece.color == perspective else 6
    return color_offset + type_offset


def square_to_plane_coords(square: chess.Square, perspective: chess.Color) -> tuple[int, int]:
    if perspective == chess.BLACK:
        square = chess.square_mirror(square)
    rank = chess.square_rank(square)
    file = chess.square_file(square)
    return 7 - rank, file

