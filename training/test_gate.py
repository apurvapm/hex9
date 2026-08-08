"""Checks that the Phase 2e gate actually catches the bugs it exists to catch.

A gate is only worth having if a broken network fails it, so these tests feed
gate.score() synthetic oracles rather than a real checkpoint. That separates two
questions which are easy to confuse: whether the network is good, and whether the
measurement can tell. Only the second is testable without training.

Run the fixture generator first:

    ./build/gate --out=training/gate_fixture.txt
    python -m pytest training/test_gate.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from encoding import canonicalise, needs_transpose, replay
from gate import load_fixture, score

FIXTURE = pathlib.Path(__file__).with_name("gate_fixture.txt")


@pytest.fixture(scope="module")
def fixture():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing; run ./build/gate --out={FIXTURE}")
    return load_fixture(FIXTURE)


def oracle(
    size: int,
    cases: list,
    *,
    invert_value: bool = False,
    misalign_policy: bool = False,
    constant_value: float | None = None,
):
    """A forward pass that knows the answers, optionally with one bug injected.

    The policy places all its mass on a genuinely winning move. Canonicalising
    that move is the step under test: skipping it is exactly the misalignment
    bug, and it only corrupts positions where Blue is to move.
    """
    policy_width = size * size + 1

    def forward(planes: np.ndarray):
        logits = np.zeros((len(cases), policy_width), dtype=np.float32)
        values = np.zeros(len(cases), dtype=np.float32)
        for index, case in enumerate(cases):
            if constant_value is not None:
                values[index] = constant_value
            else:
                values[index] = (
                    -case.mover_value if invert_value else case.mover_value
                )
            if not case.winning:
                continue
            position = replay(size, case.moves)
            move = case.winning[0]
            target = (
                move
                if misalign_policy
                else canonicalise(needs_transpose(position), move, size)
            )
            logits[index, target] = 10.0
        return logits, values

    return forward


def test_fixture_has_both_value_classes(fixture):
    size, cases = fixture
    won = sum(1 for case in cases if case.mover_value == 1)
    assert won > 20, "too few won positions to score a policy"
    assert len(cases) - won > 20, "too few lost positions to score a value"


def test_fixture_positions_are_non_terminal_and_mid_game(fixture):
    """Every case must still be a real decision, not an already-decided board."""
    size, cases = fixture
    for case in cases:
        position = replay(size, case.moves)
        empty = int(np.count_nonzero(position.cells == 0))
        assert empty >= size * size - 18, "position has too many stones"
        assert empty > 0, "position has no legal moves"


def test_oracle_scores_perfectly(fixture):
    size, cases = fixture
    result = score(size, cases, oracle(size, cases))

    assert result["value_balanced"] == pytest.approx(1.0)
    assert result["policy_hit_rate"] == pytest.approx(1.0)
    # Both halves perfect means the canonical mapping round-trips in the gate.
    assert result["policy_hit_direct"] == pytest.approx(1.0)
    assert result["policy_hit_transposed"] == pytest.approx(1.0)


def test_inverted_value_head_scores_zero_not_fifty(fixture):
    """The load-bearing test.

    A value target with the player-to-move sign flipped is the single most
    expensive bug available in this project: it trains smoothly and plays like
    garbage. It must not look like a network that merely guesses. Guessing scores
    50% balanced accuracy; being confidently backwards scores 0%.
    """
    size, cases = fixture
    result = score(size, cases, oracle(size, cases, invert_value=True))

    assert result["value_balanced"] == pytest.approx(0.0)
    assert result["recall_won"] == pytest.approx(0.0)
    assert result["recall_lost"] == pytest.approx(0.0)


def test_constant_value_head_scores_fifty_and_matches_the_null(fixture):
    """Always predicting the majority class must not look like skill."""
    size, cases = fixture
    result = score(size, cases, oracle(size, cases, constant_value=1.0))

    assert result["value_balanced"] == pytest.approx(0.5)
    assert result["recall_won"] == pytest.approx(1.0)
    assert result["recall_lost"] == pytest.approx(0.0)
    # Raw agreement is exactly the majority baseline, which is why the gate
    # refuses to lead with it.
    assert result["value_raw"] == pytest.approx(result["majority_baseline"])


def test_misaligned_policy_shows_up_as_a_transpose_gap(fixture):
    """A canonicalisation bug corrupts only the transposed half.

    This is why the gate splits the policy metric. The aggregate rate stays
    respectable — roughly half the positions are unaffected — so an aggregate
    threshold alone would pass a genuinely broken encoder.
    """
    size, cases = fixture
    result = score(size, cases, oracle(size, cases, misalign_policy=True))

    assert result["policy_hit_direct"] == pytest.approx(1.0), (
        "red-to-move positions need no transpose and must be unaffected"
    )
    assert result["policy_hit_transposed"] < 0.5, (
        "blue-to-move positions should collapse when canonicalisation is skipped"
    )
    gap = result["policy_hit_direct"] - result["policy_hit_transposed"]
    assert gap > 0.20, "the gap the gate thresholds on should be obvious"


def test_aggregate_alone_would_have_missed_the_misalignment(fixture):
    """Justifies the split metric by showing the aggregate is not enough."""
    size, cases = fixture
    result = score(size, cases, oracle(size, cases, misalign_policy=True))

    floor = result["policy_chance"] + 0.25
    assert result["policy_hit_rate"] > floor, (
        "if this ever fails the aggregate threshold would have caught the bug "
        "on its own, and the transpose split could be simplified away"
    )
