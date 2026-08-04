"""Reader for the self-play shards written by tools/selfplay.cpp.

Games are stored as move sequences plus visit counts, so the trainer replays
each game and encodes positions through encoding.py — the same path the golden
fixture already pins. There is no second encoding implementation to drift.

The reader also recomputes each game's winner from scratch with a flood fill.
The engine determines the winner incrementally with a union-find; agreeing with
a separately written check means a corrupt or misaligned shard is caught here
rather than showing up later as a value head that will not converge.
"""

from __future__ import annotations

import pathlib
import struct
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from encoding import BLUE, EMPTY, NEIGHBOUR_OFFSETS, RED, Position, encode

MAGIC = b"HEX9"
VERSION = 1


@dataclass
class Game:
    moves: list[int]
    visits: list[list[tuple[int, int]]]
    winner: int  # +1 red, -1 blue


def read_shard(path: str | pathlib.Path) -> tuple[int, list[Game]]:
    """Returns (board_size, games)."""
    data = pathlib.Path(path).read_bytes()
    if data[:4] != MAGIC:
        raise ValueError(f"{path} is not a hex9 shard")

    version, size, count = struct.unpack_from("<HHI", data, 4)
    if version != VERSION:
        raise ValueError(f"unsupported shard version {version}")

    offset = 12
    games: list[Game] = []
    for _ in range(count):
        plies, winner_byte = struct.unpack_from("<Hb", data, offset)
        offset += 3
        winner = 1 if winner_byte == 1 else -1

        moves: list[int] = []
        visits: list[list[tuple[int, int]]] = []
        for _ in range(plies):
            move, entries = struct.unpack_from("<BB", data, offset)
            offset += 2
            row: list[tuple[int, int]] = []
            for _ in range(entries):
                action, count_ = struct.unpack_from("<BH", data, offset)
                offset += 3
                row.append((action, count_))
            moves.append(move)
            visits.append(row)
        games.append(Game(moves=moves, visits=visits, winner=winner))

    return size, games


def winner_by_flood_fill(position: Position) -> int:
    """Recompute the winner independently of the engine's union-find.

    Red joins row 0 to row N-1, Blue column 0 to column N-1. Hex admits no
    draws, so a finished game has exactly one winner.
    """
    size = position.size
    for player in (RED, BLUE):
        seen = set()
        stack = []
        for i in range(size):
            cell = i if player == RED else i * size
            if position.cells[cell] == player:
                seen.add(cell)
                stack.append(cell)

        while stack:
            current = stack.pop()
            row, col = divmod(current, size)
            if player == RED and row == size - 1:
                return 1
            if player == BLUE and col == size - 1:
                return -1
            for dr, dc in NEIGHBOUR_OFFSETS:
                nr, nc = row + dr, col + dc
                if not (0 <= nr < size and 0 <= nc < size):
                    continue
                neighbour = nr * size + nc
                if neighbour not in seen and position.cells[neighbour] == player:
                    seen.add(neighbour)
                    stack.append(neighbour)
    return 0


def samples(size: int, game: Game, verify: bool = True) -> Iterator[
    tuple[np.ndarray, np.ndarray, float]
]:
    """Yield (planes, policy_target, value_target) for every position.

    The value target is +1 when the player to move at that position went on to
    win. Because the encoding is canonical — always from the mover's
    perspective — the sign is a property of the position, not of the colours.
    """
    position = Position(size)
    policy_width = size * size + 1

    for ply, move in enumerate(game.moves):
        planes = encode(position)

        target = np.zeros(policy_width, dtype=np.float32)
        total = float(sum(count for _, count in game.visits[ply]))
        if total > 0.0:
            transpose = position.to_play == BLUE
            for action, count in game.visits[ply]:
                index = action
                if transpose and action != size * size:
                    row, col = divmod(action, size)
                    index = col * size + row
                target[index] = count / total

        mover_is_red = position.to_play == RED
        value = 1.0 if (mover_is_red == (game.winner > 0)) else -1.0

        yield planes, target, value
        position.play(move)

    if verify:
        recomputed = winner_by_flood_fill(position)
        if recomputed != game.winner:
            raise ValueError(
                f"shard winner {game.winner} disagrees with flood fill "
                f"{recomputed}; the records are corrupt or misaligned"
            )


def load_dataset(
    paths: list[str | pathlib.Path], verify: bool = True
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Flatten shards into arrays ready for training."""
    all_planes: list[np.ndarray] = []
    all_policy: list[np.ndarray] = []
    all_value: list[float] = []
    size = None

    for path in paths:
        shard_size, games = read_shard(path)
        if size is None:
            size = shard_size
        elif size != shard_size:
            raise ValueError("shards mix board sizes")
        for game in games:
            for planes, policy, value in samples(shard_size, game, verify):
                all_planes.append(planes)
                all_policy.append(policy)
                all_value.append(value)

    if not all_planes:
        raise ValueError("no positions found")

    return (
        np.stack(all_planes),
        np.stack(all_policy),
        np.asarray(all_value, dtype=np.float32),
    )
