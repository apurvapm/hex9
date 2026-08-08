"""Writes a golden fixture pinning what the network outputs for given positions.

    python write_inference_fixture.py --board-size 9 --checkpoint runs/best.pt \\
        --onnx web/hex9.onnx --out web/inference_fixture.json

`export_onnx.py` already proves PyTorch matches ONNX Runtime *native*. It says
nothing about ONNX Runtime *Web*, which is a separate build — and, as it turns out, a
separate version: onnxruntime-web's latest is 1.27.0 while the native runtime here is
1.28.0, because the two are published independently. A reduced operator set or a
differently fused kernel in the web build would produce a model that loads fine in a
browser and plays badly, with nothing in the console to explain it.

So the reference is PyTorch, not the native runtime. Both runtimes are compared
against the weights as trained, rather than against each other, and a disagreement
then points at the runtime that drifted rather than leaving two suspects.

Positions come from real play rather than random tensors. Random planes can contain
things the board cannot — stones of both colours on one cell, a swap plane set at ply
40 — and a kernel that mishandles only reachable inputs would pass on noise.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np
import torch

from encoding import encode, replay
from model import build


def sample_positions(size: int, count: int, seed: int) -> list[list[int]]:
    """Legal move prefixes of varying length, avoiding finished games.

    Moves are chosen against a live position rather than from a pre-shuffled list of
    cells. That is not fussiness: a swap *relocates* the opening stone to its
    transposed cell, so a pre-shuffled list would still offer the cell the stone moved
    onto and produce an illegal line.
    """
    from records import winner_by_flood_fill

    rng = np.random.default_rng(seed)
    cells = size * size
    positions: list[list[int]] = []
    attempts = 0

    while len(positions) < count and attempts < count * 50:
        attempts += 1
        target = int(rng.integers(1, max(2, min(cells - 4, 40))))
        # Include the swap action on some lines: plane 2 is only ever set at ply 1,
        # and a post-swap board is the case the third plane exists to distinguish.
        use_swap = len(positions) % 4 == 0

        position = replay(size, [])
        moves: list[int] = []
        ok = True
        for ply in range(target):
            if ply == 1 and use_swap:
                moves.append(cells)
                position.play(cells)
                continue
            empty = [c for c in range(cells) if position.cells[c] == 0]
            if not empty:
                ok = False
                break
            move = int(empty[int(rng.integers(0, len(empty)))])
            moves.append(move)
            position.play(move)
            # Reject as soon as the game ends: the demo never asks the network about a
            # finished position, and its value is known without inference.
            if winner_by_flood_fill(position) != 0:
                ok = False
                break
        if not ok:
            continue
        positions.append(moves)

    if len(positions) < count:
        raise SystemExit(f"only sampled {len(positions)} of {count} positions")
    return positions


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--board-size", type=int, default=9)
    parser.add_argument("--preset", default="full", choices=["tiny", "light", "full"])
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--cases", type=int, default=24)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", type=pathlib.Path, required=True)
    args = parser.parse_args()

    model = build(args.board_size, args.preset)
    if args.checkpoint is not None:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
    model.eval()

    moves_per_case = sample_positions(args.board_size, args.cases, args.seed)
    planes = np.stack(
        [encode(replay(args.board_size, moves)) for moves in moves_per_case]
    )

    with torch.no_grad():
        logits, values = model(torch.from_numpy(planes))
    logits = logits.numpy()
    values = np.asarray(values).reshape(-1)

    fixture = {
        "boardSize": args.board_size,
        "preset": args.preset,
        "planeCount": int(planes.shape[1]),
        "policySize": int(logits.shape[1]),
        # Tolerance is generous next to the 1e-7 that export_onnx.py sees against the
        # native runtime. A different build may fuse or reorder differently; what
        # matters is that no operator is missing or wrong, and 1e-3 on a logit cannot
        # change a move while still catching a broken kernel.
        "tolerance": 1e-3,
        # Planes travel with the fixture rather than being rebuilt by the consumer.
        # The question this fixture answers is whether a *runtime* computes the same
        # thing, and shipping the input keeps the encoder out of the comparison --
        # it is already cross-verified between C++ and Python by
        # tests/test_encoding.cpp and training/test_encoding.py. Rebuilding the planes
        # in JavaScript would put a third encoder in the path and confuse a runtime
        # failure with an encoding one.
        "cases": [
            {
                "moves": moves,
                "planes": [int(x) for x in planes[i].reshape(-1)],
                "logits": [round(float(x), 6) for x in logits[i]],
                "value": round(float(values[i]), 6),
            }
            for i, moves in enumerate(moves_per_case)
        ],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(fixture, indent=1))
    print(f"wrote {len(fixture['cases'])} cases to {args.out}")
    print(f"  board {args.board_size}x{args.board_size}, preset {args.preset}")
    print(f"  logit range {logits.min():.3f} to {logits.max():.3f}")
    print(f"  value range {values.min():.3f} to {values.max():.3f}")
    swaps = sum(1 for m in moves_per_case if args.board_size**2 in m)
    print(f"  {swaps} of {len(moves_per_case)} cases include a swap")
    return 0


if __name__ == "__main__":
    sys.exit(main())
