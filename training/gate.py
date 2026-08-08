"""Scores a checkpoint against exhaustively solved 5x5 positions.

This is the Phase 2e gate. It exists to catch two bugs that a loss curve cannot
show: a value target with the player-to-move sign inverted, and a policy target
misaligned with the canonical encoding. Both train smoothly to a falling loss
and then play like garbage.

Build the fixture first:

    ./build/gate --out=training/gate_fixture.txt
    python gate.py --checkpoint runs/best.pt

Two reported numbers are deliberately paired with their null hypothesis, because
neither is interpretable alone:

*Balanced value accuracy*, not raw agreement. Random play leaves the mover
winning in about 73% of sampled positions, so a network that always outputs +1
scores 73% raw. Averaging recall over the two classes makes that strategy score
50% and cannot be gamed by the class skew.

*Policy hit rate against chance*. On a position with 15 empty cells and 4
winning moves, guessing scores 27%. The gate reports the mean chance rate over
exactly the positions it scored, so a hit rate is never read without it.

A sign-inverted value head scores near 0% balanced accuracy rather than near
50%: it is not guessing, it is confidently wrong. That gap is the whole point.
"""

from __future__ import annotations

import argparse
import dataclasses
import pathlib
import sys

import numpy as np

from encoding import (
    canonicalise,
    encode,
    encode_legal_mask,
    needs_transpose,
    replay,
)

FIXTURE = pathlib.Path(__file__).with_name("gate_fixture.txt")


class Case:
    __slots__ = ("moves", "mover_value", "winning")

    def __init__(self, moves: list[int], mover_value: int, winning: list[int]):
        self.moves = moves
        self.mover_value = mover_value
        self.winning = winning


def load_fixture(path: str | pathlib.Path) -> tuple[int, list[Case]]:
    """Parse the fixture written by tools/gate.cpp."""
    path = pathlib.Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing; run ./build/gate --out={path}"
        )

    lines = [
        line.strip()
        for line in path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    size, policy_size = (int(x) for x in lines[0].split())
    if policy_size != size * size + 1:
        raise ValueError("fixture policy width does not include the swap action")

    cases: list[Case] = []
    for line in lines[1:]:
        fields = [int(x) for x in line.split()]
        sentinel = fields.index(-1)
        moves = fields[:sentinel]
        mover_value = fields[sentinel + 1]
        count = fields[sentinel + 2]
        winning = fields[sentinel + 3 : sentinel + 3 + count]
        if len(winning) != count:
            raise ValueError(f"truncated winning-move list: {line}")
        if (mover_value == 1) != bool(winning):
            raise ValueError(f"fixture value disagrees with its move list: {line}")
        cases.append(Case(moves, mover_value, winning))

    if not cases:
        raise ValueError(f"{path} contained no cases")
    return size, cases


def evaluate_batch(model, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Run the network over stacked planes. Returns (policy_logits, values)."""
    import torch

    model.eval()
    with torch.no_grad():
        logits, values = model(torch.from_numpy(planes))
    return logits.numpy(), values.numpy()


def evaluate_batch_onnx(session, planes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    logits, values = session.run(["policy", "value"], {"board": planes})
    return logits, np.asarray(values).reshape(-1)


def score(size: int, cases: list[Case], forward) -> dict:
    """Encode every case, evaluate once, and score policy and value."""
    positions = [replay(size, case.moves) for case in cases]
    planes = np.stack([encode(position) for position in positions])
    logits, values = forward(planes)

    value_correct = {1: 0, -1: 0}
    value_total = {1: 0, -1: 0}

    # Policy statistics are split by whether the position had to be transposed
    # onto the mover's axis. An aggregate hit rate hides a canonicalisation bug,
    # because such a bug corrupts only the transposed half: the untouched half
    # still scores well and pulls the mean above any single threshold. A gap
    # between these two rates is the signature, and nothing else produces it.
    policy_hits = {False: 0, True: 0}
    policy_total = {False: 0, True: 0}
    policy_chance = {False: 0.0, True: 0.0}

    for index, (case, position) in enumerate(zip(cases, positions)):
        predicted = 1 if values[index] >= 0.0 else -1
        value_total[case.mover_value] += 1
        value_correct[case.mover_value] += predicted == case.mover_value

        # A lost position has no winning move, so the policy is unscoreable
        # there: every choice is equally doomed and a hit rate would be zero by
        # construction rather than by weakness.
        if case.mover_value != 1:
            continue

        transpose = needs_transpose(position)
        mask = encode_legal_mask(position)
        masked = np.where(mask > 0.0, logits[index], -np.inf)
        # The argmax lands in canonical space. Transposing is an involution, so
        # the same call that built the target maps the choice back to a board
        # index -- which is why the fixture stores winning moves unmapped.
        choice = canonicalise(transpose, int(np.argmax(masked)), size)

        policy_total[transpose] += 1
        policy_hits[transpose] += choice in case.winning
        legal = int(mask.sum())
        policy_chance[transpose] += len(case.winning) / legal if legal else 0.0

    def rate(hits: float, total: int) -> float:
        return hits / total if total else 0.0

    recall_won = rate(value_correct[1], value_total[1])
    recall_lost = rate(value_correct[-1], value_total[-1])
    scored = policy_total[False] + policy_total[True]

    return {
        "cases": len(cases),
        "won_cases": value_total[1],
        "lost_cases": value_total[-1],
        "value_balanced": 0.5 * (recall_won + recall_lost),
        "value_raw": rate(value_correct[1] + value_correct[-1], len(cases)),
        "recall_won": recall_won,
        "recall_lost": recall_lost,
        "majority_baseline": rate(max(value_total.values()), len(cases)),
        "policy_scored": scored,
        "policy_hit_rate": rate(policy_hits[False] + policy_hits[True], scored),
        "policy_chance": rate(
            policy_chance[False] + policy_chance[True], scored
        ),
        "policy_hit_direct": rate(policy_hits[False], policy_total[False]),
        "policy_hit_transposed": rate(policy_hits[True], policy_total[True]),
        "direct_cases": policy_total[False],
        "transposed_cases": policy_total[True],
    }


def report(result: dict) -> None:
    print(f"gate: {result['cases']} solved 5x5 positions")
    print(
        f"  value balanced accuracy : {result['value_balanced']:6.1%}"
        f"   (recall: won {result['recall_won']:.1%}, "
        f"lost {result['recall_lost']:.1%})"
    )
    print(
        f"  value raw agreement     : {result['value_raw']:6.1%}"
        f"   (always-majority scores {result['majority_baseline']:.1%})"
    )
    print(
        f"  policy picks a winner   : {result['policy_hit_rate']:6.1%}"
        f"   (chance {result['policy_chance']:.1%}"
        f" over {result['policy_scored']} won positions)"
    )
    print(
        f"    red to move, as-is    : {result['policy_hit_direct']:6.1%}"
        f"   ({result['direct_cases']} positions)"
    )
    print(
        f"    blue to move, mirrored: {result['policy_hit_transposed']:6.1%}"
        f"   ({result['transposed_cases']} positions)"
    )


@dataclasses.dataclass
class Thresholds:
    """What counts as passing. Defaults are the ones train.py --gate applies."""

    min_value_balanced: float = 0.80
    min_policy_margin: float = 0.25
    min_per_class: int = 20
    max_transpose_gap: float = 0.20


UNSCOREABLE = "unscoreable"


def verdict(result: dict, thresholds: Thresholds) -> list[str]:
    """Reasons the result fails, empty when it passes.

    A single element equal to UNSCOREABLE means the fixture itself is too thin to
    judge, which is neither a pass nor a network failure.
    """
    # Balanced accuracy over three lost positions is noise, and reporting that as
    # a pass would be worse than refusing to answer.
    thin = min(result["won_cases"], result["lost_cases"])
    if thin < thresholds.min_per_class:
        return [UNSCOREABLE]

    failures = []
    if result["value_balanced"] < thresholds.min_value_balanced:
        failures.append(
            f"balanced value accuracy {result['value_balanced']:.1%} is below "
            f"{thresholds.min_value_balanced:.1%}"
        )

    # A margin over chance rather than a multiple of it. A multiple is unstable
    # exactly where it matters: when a fixture happens to hold easy positions the
    # chance rate rises, and "2x chance" can exceed 100% and become unreachable
    # no matter how strong the network is.
    floor = result["policy_chance"] + thresholds.min_policy_margin
    if result["policy_hit_rate"] < floor:
        failures.append(
            f"policy hit rate {result['policy_hit_rate']:.1%} is below chance "
            f"plus {thresholds.min_policy_margin:.0%} ({floor:.1%})"
        )

    # The network sees both halves through the same canonical encoding, so it has
    # no way to be genuinely better at one than the other. A gap means the mapping
    # between board and canonical action space is wrong somewhere, and the
    # aggregate rate above would not have caught it.
    gap = abs(result["policy_hit_direct"] - result["policy_hit_transposed"])
    if gap > thresholds.max_transpose_gap:
        failures.append(
            f"red and blue policy hit rates differ by {gap:.1%} "
            f"({result['policy_hit_direct']:.1%} vs "
            f"{result['policy_hit_transposed']:.1%}); the canonical action "
            "mapping is suspect"
        )
    return failures


def announce(failures: list[str], min_per_class: int) -> int:
    """Print a verdict and return the process exit code it implies."""
    if failures == [UNSCOREABLE]:
        print(
            f"\nGATE UNSCOREABLE\n  fewer than {min_per_class} positions in one "
            "value class; regenerate the fixture with more --cases"
        )
        return 2
    if failures:
        print("\nGATE FAILED")
        for failure in failures:
            print(f"  {failure}")
        return 1
    print("\ngate passed")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture", type=pathlib.Path, default=FIXTURE)
    parser.add_argument("--checkpoint", type=pathlib.Path)
    parser.add_argument("--onnx", type=pathlib.Path)
    parser.add_argument("--preset", default="tiny", choices=["tiny", "light", "full"])
    parser.add_argument(
        "--min-value-balanced",
        type=float,
        default=0.80,
        help="minimum balanced value accuracy to pass",
    )
    parser.add_argument(
        "--min-policy-margin",
        type=float,
        default=0.25,
        help="policy hit rate must exceed the chance rate by this margin",
    )
    parser.add_argument(
        "--min-per-class",
        type=int,
        default=20,
        help="minimum solved positions in each value class to score at all",
    )
    parser.add_argument(
        "--max-transpose-gap",
        type=float,
        default=0.20,
        help="largest tolerated gap between the red and blue policy hit rates",
    )
    args = parser.parse_args()

    if (args.checkpoint is None) == (args.onnx is None):
        parser.error("pass exactly one of --checkpoint or --onnx")

    size, cases = load_fixture(args.fixture)

    if args.onnx is not None:
        import onnxruntime as ort

        session = ort.InferenceSession(
            str(args.onnx), providers=["CPUExecutionProvider"]
        )
        forward = lambda planes: evaluate_batch_onnx(session, planes)
    else:
        import torch

        from model import build

        model = build(size, args.preset)
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state["model"] if "model" in state else state)
        forward = lambda planes: evaluate_batch(model, planes)

    result = score(size, cases, forward)
    report(result)

    thresholds = Thresholds(
        min_value_balanced=args.min_value_balanced,
        min_policy_margin=args.min_policy_margin,
        min_per_class=args.min_per_class,
        max_transpose_gap=args.max_transpose_gap,
    )
    return announce(verdict(result, thresholds), thresholds.min_per_class)


if __name__ == "__main__":
    sys.exit(main())
