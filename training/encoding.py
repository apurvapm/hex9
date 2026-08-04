"""Position encoding for hex9, mirroring include/hex/encoding.hpp.

The C++ self-play driver and this trainer must produce byte-identical tensors
from the same position. A mismatch here does not crash: training loss falls
smoothly while the network learns from misaligned inputs. The golden fixture
written by tests/test_encoding.cpp is the contract, and test_encoding.py checks
this file against it.

Canonicalisation: Hex is not symmetric between players, since Red connects top
to bottom and Blue connects left to right. But transposing (r, c) -> (c, r) maps
Red's goal onto Blue's, so the position is always presented from the perspective
of the player to move, oriented onto the top-bottom axis. The network learns one
goal direction rather than two.
"""

from __future__ import annotations

import numpy as np

PLANES = 3

EMPTY, RED, BLUE = 0, 1, 2

NEIGHBOUR_OFFSETS = ((-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0))


class Position:
    """Minimal board used to replay move sequences. Not a search board.

    The engine owns game logic; this exists only so the trainer can rebuild a
    position from the move list stored in a self-play record.
    """

    def __init__(self, size: int):
        self.size = size
        self.cells = np.zeros(size * size, dtype=np.int8)
        self.to_play = RED
        self.ply = 0
        self._swap_used = False

    def index(self, row: int, col: int) -> int:
        return row * self.size + col

    @property
    def swap_move(self) -> int:
        return self.size * self.size

    def can_swap(self) -> bool:
        return self.ply == 1

    def place(self, move: int, player: int) -> None:
        """Place a stone for an explicit player, ignoring whose turn it is.

        Mirrors Board::PlaceStone. Used for building test positions; play() is
        the normal entry point.
        """
        assert self.cells[move] == EMPTY, "cell already occupied"
        self.cells[move] = player
        self.to_play = BLUE if player == RED else RED
        self.ply += 1

    def play(self, move: int) -> None:
        if move == self.swap_move:
            self.play_swap()
            return
        assert self.cells[move] == EMPTY, "cell already occupied"
        self.cells[move] = self.to_play
        self.to_play = BLUE if self.to_play == RED else RED
        self.ply += 1

    def play_swap(self) -> None:
        """Blue takes Red's opening stone.

        Implemented as transpose-plus-recolour, matching Board::PlaySwap: the
        stone leaves (r, c) and reappears as Blue's at (c, r), with Red to move.
        """
        assert self.can_swap(), "swap is only legal as the reply to the opening"
        (original,) = np.nonzero(self.cells)
        original = int(original[0])
        row, col = divmod(original, self.size)
        self.cells[original] = EMPTY
        self.cells[self.index(col, row)] = BLUE
        self.to_play = RED
        self.ply = 2
        self._swap_used = True


def needs_transpose(position: Position) -> bool:
    return position.to_play == BLUE


def canonicalise(transpose: bool, move: int, size: int) -> int:
    """Map a board action to canonical space. Transposing is its own inverse."""
    if not transpose or move == size * size:
        return move
    row, col = divmod(move, size)
    return col * size + row


def encode(position: Position) -> np.ndarray:
    """Return a (PLANES, size, size) float32 tensor.

    Plane 0 holds the stones of the player to move, plane 1 the opponent's, and
    plane 2 is filled when the swap action is legal. Plane 2 is not redundant:
    after a swap the board holds one stone at ply 2, which is indistinguishable
    in planes 0 and 1 from the one-stone position at ply 1 where swap is still
    available.
    """
    size = position.size
    transpose = needs_transpose(position)
    mine = position.to_play

    planes = np.zeros((PLANES, size, size), dtype=np.float32)
    for cell in range(size * size):
        value = position.cells[cell]
        if value == EMPTY:
            continue
        row, col = divmod(cell, size)
        if transpose:
            row, col = col, row
        planes[0 if value == mine else 1, row, col] = 1.0

    if position.can_swap():
        planes[2, :, :] = 1.0
    return planes


def encode_policy_target(
    position: Position, visits: dict[int, int]
) -> np.ndarray:
    """Normalised visit counts in canonical action space."""
    size = position.size
    transpose = needs_transpose(position)
    target = np.zeros(size * size + 1, dtype=np.float32)
    total = float(sum(visits.values()))
    if total <= 0.0:
        return target
    for move, count in visits.items():
        target[canonicalise(transpose, move, size)] = count / total
    return target


def encode_legal_mask(position: Position) -> np.ndarray:
    size = position.size
    transpose = needs_transpose(position)
    mask = np.zeros(size * size + 1, dtype=np.float32)
    for cell in range(size * size):
        if position.cells[cell] == EMPTY:
            mask[canonicalise(transpose, cell, size)] = 1.0
    if position.can_swap():
        mask[size * size] = 1.0
    return mask


def replay(size: int, moves: list[int]) -> Position:
    position = Position(size)
    for move in moves:
        position.play(move)
    return position
