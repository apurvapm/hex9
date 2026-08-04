"""Tests for the self-play record reader.

A shard is generated on the fly by tools/selfplay.cpp in heuristic mode, so this
runs in CI without an ONNX Runtime dependency.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess

import numpy as np
import pytest

from encoding import BLUE, RED, Position
from records import (
    load_dataset,
    read_shard,
    samples,
    winner_by_flood_fill,
)

BOARD_SIZE = 5
GAMES = 40


def find_selfplay() -> pathlib.Path | None:
    root = pathlib.Path(__file__).resolve().parent.parent
    for candidate in (root / "build" / "selfplay", root / "selfplay"):
        if candidate.exists():
            return candidate
    found = shutil.which("selfplay")
    return pathlib.Path(found) if found else None


@pytest.fixture(scope="module")
def shard(tmp_path_factory) -> pathlib.Path:
    binary = find_selfplay()
    if binary is None:
        pytest.skip("selfplay binary not built; run cmake --build build first")

    path = tmp_path_factory.mktemp("shards") / "test.bin"
    subprocess.run(
        [
            str(binary),
            f"--size={BOARD_SIZE}",
            "--heuristic",
            f"--games={GAMES}",
            "--sims=60",
            "--seed=17",
            f"--out={path}",
        ],
        check=True,
        capture_output=True,
    )
    return path


def test_header_and_game_count(shard):
    size, games = read_shard(shard)
    assert size == BOARD_SIZE
    assert len(games) == GAMES


def test_games_are_well_formed(shard):
    size, games = read_shard(shard)
    for game in games:
        assert game.winner in (-1, 1), "Hex admits no draws"
        assert len(game.moves) == len(game.visits)
        assert len(game.moves) > 0
        for move in game.moves:
            assert 0 <= move <= size * size, "move outside the action space"
        for row in game.visits:
            assert row, "every search should record at least one visited action"
            assert all(count > 0 for _, count in row)


def test_recorded_winner_matches_independent_flood_fill(shard):
    """The engine derives the winner incrementally with a union-find. Replaying
    and flood-filling is a separate implementation, so agreement is real
    evidence the shard is intact."""
    size, games = read_shard(shard)
    for game in games:
        position = Position(size)
        for move in game.moves:
            position.play(move)
        assert winner_by_flood_fill(position) == game.winner


def test_samples_have_valid_targets(shard):
    size, games = read_shard(shard)
    planes, policy, value = load_dataset([shard], verify=True)

    positions = sum(len(game.moves) for game in games)
    assert planes.shape == (positions, 3, size, size)
    assert policy.shape == (positions, size * size + 1)
    assert value.shape == (positions,)

    assert np.allclose(policy.sum(axis=1), 1.0), "policy targets must normalise"
    assert set(np.unique(value)) <= {-1.0, 1.0}, "Hex outcomes are win or loss"
    assert np.all(planes[:, :2].sum(axis=(1, 2, 3)) >= 0)


def test_policy_mass_lands_only_on_legal_actions(shard):
    size, games = read_shard(shard)
    for game in games[:10]:
        position = Position(size)
        for ply, (planes, policy, _) in enumerate(
            samples(size, game, verify=False)
        ):
            occupied = planes[0] + planes[1]
            board_mass = policy[: size * size].reshape(size, size)
            assert not np.any(board_mass[occupied > 0]), (
                "policy target put mass on an occupied cell"
            )
            position.play(game.moves[ply])


def test_value_target_flips_with_the_mover(shard):
    """Consecutive positions have opposite movers, so their value targets must
    be opposite too — except across a swap, which hands the move back."""
    size, games = read_shard(shard)
    for game in games[:10]:
        values = [value for _, _, value in samples(size, game, verify=False)]
        position = Position(size)
        for ply in range(len(game.moves) - 1):
            was_swap = game.moves[ply] == size * size
            if not was_swap:
                assert values[ply] == -values[ply + 1], (
                    "value target should alternate between plies"
                )
            position.play(game.moves[ply])


def test_swap_positions_keep_the_mover(shard):
    """After a swap it is Red's move again, so the sign does not flip."""
    size, games = read_shard(shard)
    swap_action = size * size
    seen_swap = False
    for game in games:
        if swap_action not in game.moves:
            continue
        seen_swap = True
        index = game.moves.index(swap_action)
        position = Position(size)
        for move in game.moves[:index]:
            position.play(move)
        assert position.to_play == BLUE, "only Blue may swap"
        position.play(swap_action)
        assert position.to_play == RED, "Red moves again after a swap"
        assert position.ply == 2
        break
    if not seen_swap:
        pytest.skip("no swap appeared in this shard")
