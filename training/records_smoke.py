"""Smoke check: read a shard and confirm every derived target is well formed.

Run as `python -m records_smoke <shard>`. Used by CI so the self-play driver
and the reader are exercised together on every push.
"""

from __future__ import annotations

import sys

import numpy as np

from records import load_dataset


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: python -m records_smoke <shard.bin>")
        return 1

    planes, policy, value = load_dataset([sys.argv[1]], verify=True)

    assert np.allclose(policy.sum(axis=1), 1.0), "policy targets must normalise"
    assert set(np.unique(value)) <= {-1.0, 1.0}, "Hex outcomes are win or loss"
    assert planes.ndim == 4 and planes.shape[1] == 3, "unexpected plane layout"

    print(
        f"{len(planes)} positions, "
        f"{planes.shape[2]}x{planes.shape[3]} board, "
        f"policy width {policy.shape[1]}, winners verified by flood fill"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
