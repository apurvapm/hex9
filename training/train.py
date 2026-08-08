"""Trains HexNet on self-play shards and gates the result against perfect play.

    ./build/selfplay --size=5 --heuristic --games=400 --sims=200 --out=s0.bin
    python train.py --shards s0.bin --preset tiny --epochs 30 --gate

The loss is the AlphaZero one: cross-entropy against the normalised visit counts
plus mean squared error against the game outcome. Both targets come out of
records.py, which replays each game through the same encoder the golden fixture
pins, so there is no second encoding to drift.

Two decisions here are not the obvious ones.

*The train/validation split is by game, not by position.* Positions inside one
game share almost their whole board, so splitting at position level leaks the
validation set into training and produces a validation curve that tracks the
training curve no matter how badly the model generalises. Splitting by game is
the only version of the number that means anything.

*Passing --gate is the point of the exercise.* A falling loss is not evidence of
a working trainer: a value target with the player-to-move sign inverted trains
just as smoothly. The gate compares the network against exhaustively solved 5x5
positions, and it is what turns "training ran" into "training worked".

Shuffling uses torch's generator with a fixed seed. That is deliberately not the
hand-rolled sampler discipline the engine follows: the engine's samplers exist so
self-play data is bit-identical across toolchains, whereas trained weights carry
no such requirement. One machine reproducing its own run is the useful property
here.
"""

from __future__ import annotations

import argparse
import pathlib
import sys

import numpy as np
import torch
import torch.nn.functional as F

from model import build
from records import read_shard, samples


def load_grouped(
    paths: list[pathlib.Path], verify: bool = True
) -> tuple[int, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load shards, keeping a game index per position so splits stay honest.

    Returns (size, planes, policy, value, game_id).
    """
    all_planes: list[np.ndarray] = []
    all_policy: list[np.ndarray] = []
    all_value: list[float] = []
    all_group: list[int] = []
    size: int | None = None
    game_id = 0

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
                all_group.append(game_id)
            game_id += 1

    if not all_planes:
        raise ValueError("no positions found in the given shards")

    return (
        int(size),
        np.stack(all_planes),
        np.stack(all_policy),
        np.asarray(all_value, dtype=np.float32),
        np.asarray(all_group, dtype=np.int64),
    )


def split_by_game(
    group: np.ndarray, holdout: float, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """Partition position indices so no game appears on both sides."""
    games = np.unique(group)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(games)
    cut = max(1, int(round(len(shuffled) * holdout)))
    validation_games = set(shuffled[:cut].tolist())

    is_validation = np.array([g in validation_games for g in group])
    return np.nonzero(~is_validation)[0], np.nonzero(is_validation)[0]


def losses(
    model, planes: torch.Tensor, policy: torch.Tensor, value: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Policy cross-entropy against soft targets, and value MSE.

    The targets are already zero on illegal actions, so the softmax runs over the
    full action space and the cross-entropy pushes illegal mass down on its own.
    PUCT renormalises over legal moves at inference regardless, so a small
    residual there costs nothing.
    """
    logits, predicted = model(planes)
    policy_loss = -(policy * F.log_softmax(logits, dim=1)).sum(dim=1).mean()
    value_loss = F.mse_loss(predicted, value)
    return policy_loss, value_loss


def run_epoch(
    model, optimiser, tensors, indices: np.ndarray, batch: int, generator
) -> tuple[float, float]:
    """One pass. Training when an optimiser is given, evaluation otherwise."""
    planes, policy, value = tensors
    training = optimiser is not None
    model.train(training)

    order = (
        indices[torch.randperm(len(indices), generator=generator).numpy()]
        if training
        else indices
    )

    policy_total = value_total = 0.0
    seen = 0
    for start in range(0, len(order), batch):
        chunk = order[start : start + batch]
        if len(chunk) == 0:
            continue
        with torch.set_grad_enabled(training):
            policy_loss, value_loss = losses(
                model, planes[chunk], policy[chunk], value[chunk]
            )
            if training:
                optimiser.zero_grad(set_to_none=True)
                (policy_loss + value_loss).backward()
                optimiser.step()
        # Detach before accumulating: the tensors still carry grad during the
        # training pass, and converting them straight to floats keeps the graph
        # alive for no reason.
        policy_total += policy_loss.detach().item() * len(chunk)
        value_total += value_loss.detach().item() * len(chunk)
        seen += len(chunk)

    return policy_total / seen, value_total / seen


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--shards", type=pathlib.Path, nargs="+", required=True)
    parser.add_argument("--preset", default="tiny", choices=["tiny", "light", "full"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--holdout", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=20260804)
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("runs"))
    parser.add_argument(
        "--gate",
        action="store_true",
        help="score the best checkpoint against solved 5x5 positions and fail "
        "the run if it does not pass",
    )
    parser.add_argument("--no-verify", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    size, planes, policy, value, group = load_grouped(
        args.shards, verify=not args.no_verify
    )
    train_idx, validation_idx = split_by_game(group, args.holdout, args.seed)

    tensors = (
        torch.from_numpy(planes),
        torch.from_numpy(policy),
        torch.from_numpy(value),
    )

    model = build(size, args.preset)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print(model.describe())
    print(
        f"{len(planes)} positions from {len(np.unique(group))} games: "
        f"{len(train_idx)} train, {len(validation_idx)} validation "
        f"(split by game)"
    )

    args.out.mkdir(parents=True, exist_ok=True)
    best_path = args.out / "best.pt"
    best_loss = float("inf")

    for epoch in range(1, args.epochs + 1):
        train_policy, train_value = run_epoch(
            model, optimiser, tensors, train_idx, args.batch, generator
        )
        validation_policy, validation_value = run_epoch(
            model, None, tensors, validation_idx, args.batch, generator
        )
        total = validation_policy + validation_value

        marker = ""
        if total < best_loss:
            best_loss = total
            torch.save(
                {
                    "model": model.state_dict(),
                    "board_size": size,
                    "preset": args.preset,
                    "epoch": epoch,
                    "validation_loss": total,
                },
                best_path,
            )
            marker = "  <- best"

        # Flushed per epoch: a real run is long and redirected to a log, and
        # block buffering would show nothing until it finished.
        print(
            f"epoch {epoch:3d}  train p {train_policy:.4f} v {train_value:.4f}"
            f"   val p {validation_policy:.4f} v {validation_value:.4f}{marker}",
            flush=True,
        )

    print(f"\nbest validation loss {best_loss:.4f} saved to {best_path}")

    if not args.gate:
        print("run gate.py against it before trusting this checkpoint")
        return 0

    from gate import Thresholds, announce, load_fixture, report, score, verdict

    fixture_size, cases = load_fixture(
        pathlib.Path(__file__).with_name("gate_fixture.txt")
    )
    if fixture_size != size:
        print(
            f"\ngate fixture is {fixture_size}x{fixture_size} but the model is "
            f"{size}x{size}; there is no ground truth to gate against"
        )
        return 1

    # Gate the checkpoint that will actually be shipped, not the final epoch's
    # weights, which may be past the point where validation loss turned.
    state = torch.load(best_path, map_location="cpu")
    model.load_state_dict(state["model"])
    model.eval()

    def forward(batch_planes: np.ndarray):
        with torch.no_grad():
            logits, values = model(torch.from_numpy(batch_planes))
        return logits.numpy(), values.numpy()

    print()
    result = score(size, cases, forward)
    report(result)
    thresholds = Thresholds()
    return announce(verdict(result, thresholds), thresholds.min_per_class)


if __name__ == "__main__":
    sys.exit(main())
