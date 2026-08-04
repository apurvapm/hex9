"""Checks training/encoding.py against the fixture written by the C++ tests.

Run the C++ suite first to produce the fixture:

    ./build/test_encoding training/encoding_fixture.txt
    python -m pytest training/test_encoding.py

The fixture stores, per case, the move sequence plus two summaries of the
resulting tensor: the number of set entries and an FNV-1a checksum over their
flat indices. Matching both means the two encoders agree on every position, not
merely on how many stones are on the board.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest

from encoding import BLUE, PLANES, RED, encode, encode_legal_mask, replay

FIXTURE = pathlib.Path(__file__).with_name("encoding_fixture.txt")

FNV_OFFSET = 1469598103934665603
FNV_PRIME = 1099511628211
MASK64 = (1 << 64) - 1


def checksum(planes: np.ndarray) -> tuple[float, int]:
    """Reproduce the C++ summary: total set entries, and a hash of their indices."""
    flat = planes.reshape(-1)
    digest = FNV_OFFSET
    for index in np.nonzero(flat)[0]:
        digest ^= int(index)
        digest = (digest * FNV_PRIME) & MASK64
    return float(flat.sum()), digest


def load_cases():
    if not FIXTURE.exists():
        pytest.skip(f"fixture missing; run ./build/test_encoding {FIXTURE}")

    lines = [
        line.strip()
        for line in FIXTURE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    size, planes, policy_size = (int(x) for x in lines[0].split())

    cases = []
    for line in lines[1:]:
        fields = line.split()
        sentinel = fields.index("-1")
        moves = [int(x) for x in fields[:sentinel]]
        expected_sum = float(fields[sentinel + 1])
        expected_digest = int(fields[sentinel + 2])
        cases.append((moves, expected_sum, expected_digest))
    return size, planes, policy_size, cases


def test_header_matches_this_module():
    size, planes, policy_size, _ = load_cases()
    assert planes == PLANES, "plane count disagrees with the C++ encoder"
    assert policy_size == size * size + 1, "policy width should include swap"


def test_every_fixture_case_matches():
    size, _, _, cases = load_cases()
    assert cases, "fixture contained no cases"

    for moves, expected_sum, expected_digest in cases:
        position = replay(size, moves)
        planes = encode(position)
        actual_sum, actual_digest = checksum(planes)

        assert actual_sum == expected_sum, (
            f"set-entry count differs for moves {moves}: "
            f"python {actual_sum} vs c++ {expected_sum}"
        )
        assert actual_digest == expected_digest, (
            f"tensor contents differ for moves {moves}"
        )


def test_transpose_is_an_involution():
    """A position and its mirror-with-colours-swapped must encode identically.

    Recolouring is essential: transposing alone maps Red's goal onto Blue's, so
    the mirrored position is only the same game state if the colours also swap.
    """
    size = 7
    direct = replay(size, [])
    mirror = replay(size, [])
    for ply, move in enumerate([10, 24, 33, 8, 41]):
        row, col = divmod(move, size)
        mover = RED if ply % 2 == 0 else BLUE
        direct.place(move, mover)
        mirror.place(col * size + row, BLUE if mover == RED else RED)
    assert np.array_equal(encode(direct), encode(mirror))


def test_swap_plane_separates_ply_one_from_post_swap():
    size = 7
    before = replay(size, [2 * size + 4])
    after = replay(size, [2 * size + 4, size * size])

    assert before.can_swap()
    assert not after.can_swap()
    assert int(np.count_nonzero(before.cells)) == 1
    assert int(np.count_nonzero(after.cells)) == 1

    assert encode(before)[2].all(), "swap plane should be set at ply 1"
    assert not encode(after)[2].any(), "swap plane should be clear after swap"
    assert encode_legal_mask(before)[size * size] == 1.0
    assert encode_legal_mask(after)[size * size] == 0.0
