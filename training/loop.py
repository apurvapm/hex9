"""Closes the AlphaZero loop: self-play, train, export, gate, repeat.

    python loop.py --generations 4 --games 1200 --epochs 15

Generation 0 bootstraps from the connection-distance heuristic, because there is
no network yet. Every later generation plays its games with the previous
generation's exported ONNX model, which is what makes this a loop rather than a
single supervised fit.

Each generation is gated twice, and the pair is the interesting part:

    gate.py         scores the raw network against solved positions
    gate_search     scores PUCT *driving* that network against the same positions

Search normally scores far higher than the bare policy. If it ever does not, the
network is not the thing that is broken.

Training restarts from scratch each generation over a sliding window of recent
shards, rather than warm-starting. Warm-starting is faster and standard, but it
compounds whatever the previous generation got wrong, and on a 5x5 board with a
tiny network the retrain costs minutes. Reproducibility is worth more than the
minutes here; revisit that trade at 9x9.

Shards accumulate rather than being replaced so that a generation trains on a mix
of its own and its predecessors' games. Training only on the newest generation
makes the network chase its own most recent quirks.
"""

from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys

REPO = pathlib.Path(__file__).resolve().parent.parent


def run(command: list[str], label: str) -> None:
    """Run a step, streaming nothing but failing loudly."""
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n{label} failed (exit {result.returncode}):")
        print(result.stdout[-4000:])
        print(result.stderr[-2000:])
        raise SystemExit(result.returncode)


def selfplay(
    binary: pathlib.Path,
    size: int,
    games: int,
    sims: int,
    seed: int,
    out: pathlib.Path,
    model: pathlib.Path | None,
    threads: int,
) -> None:
    command = [
        str(binary),
        f"--size={size}",
        f"--games={games}",
        f"--sims={sims}",
        f"--seed={seed}",
        f"--threads={threads}",
        f"--out={out}",
    ]
    command.append("--heuristic" if model is None else f"--model={model}")
    run(command, "self-play")


def score_network(onnx_path: pathlib.Path, size: int) -> dict | None:
    """Gate the exported graph, which is the artifact later stages consume.

    Returns None when the fixture does not match the board size. That is the normal
    case at 9x9, not an error: the gate needs exhaustively solved positions, and the
    solver only reaches mid-game positions on 5x5. A 9x9 fixture could only hold
    nearly-decided endgames, which would pass trivially and prove nothing.

    Absolute correctness therefore comes from the 5x5 gate, run once before scaling,
    and generation-over-generation progress at 9x9 comes from the arena.
    """
    import onnxruntime as ort

    from gate import evaluate_batch_onnx, load_fixture, score

    fixture_size, cases = load_fixture(
        pathlib.Path(__file__).with_name("gate_fixture.txt")
    )
    if fixture_size != size:
        return None

    session = ort.InferenceSession(
        str(onnx_path), providers=["CPUExecutionProvider"]
    )
    return score(size, cases, lambda planes: evaluate_batch_onnx(session, planes))


def score_elo(
    binary: pathlib.Path,
    challenger: pathlib.Path,
    incumbent: pathlib.Path,
    size: int,
    pairs: int,
    sims: int,
    threads: int,
) -> str:
    """Colour-balanced match against the previous generation, reported as Elo.

    This is the progress signal at sizes the gate cannot reach. Pairing is not
    optional: red wins about 53% under random play, so an unpaired match would report
    the colour assignment rather than the improvement.
    """
    result = subprocess.run(
        [
            str(binary),
            f"--size={size}",
            f"--a={challenger}",
            f"--b={incumbent}",
            f"--pairs={pairs}",
            f"--sims={sims}",
            f"--threads={threads}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    if result.returncode != 0:
        return "arena failed"

    # The arena reports both the estimate and whether it separates the two agents, and
    # the second half is not optional. A summary row showing a bare "-191" reads as a
    # regression when the interval spans zero and the match in fact said nothing. Any
    # number carried out of here without its caveat is worse than no number.
    elo = "n/a"
    significant = True
    for line in result.stdout.splitlines():
        stripped = line.strip()
        # Anchored rather than a substring test: a model path containing "elo" would
        # otherwise be mistaken for the result line.
        if stripped.startswith("elo") and elo == "n/a":
            elo = stripped.split(":", 1)[1].strip()
        if "interval spans zero" in stripped:
            significant = False
    return elo if significant else f"{elo} (not significant)"


def score_agent(
    binary: pathlib.Path, onnx_path: pathlib.Path, sims: int
) -> tuple[bool, str]:
    """Run the C++ search gate. Returns (passed, its report)."""
    result = subprocess.run(
        [
            str(binary),
            f"--model={onnx_path}",
            f"--sims={sims}",
            f"--fixture={REPO / 'training' / 'gate_fixture.txt'}",
        ],
        capture_output=True,
        text=True,
        cwd=REPO,
    )
    return result.returncode == 0, result.stdout.strip()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--size", type=int, default=5)
    parser.add_argument("--games", type=int, default=1200)
    parser.add_argument("--sims", type=int, default=200)
    parser.add_argument("--gate-sims", type=int, default=200)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--preset", default="tiny", choices=["tiny", "light", "full"])
    parser.add_argument(
        "--window",
        type=int,
        default=3,
        help="how many recent generations of shards to train on",
    )
    parser.add_argument("--seed", type=int, default=20260805)
    parser.add_argument(
        "--threads",
        type=int,
        default=1,
        help="self-play workers; shards stay bit-identical whatever this is",
    )
    parser.add_argument(
        "--pairs",
        type=int,
        default=60,
        help="colour-balanced pairs per arena match against the previous generation",
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("loop"))
    args = parser.parse_args()

    # Resolved immediately, because two kinds of subprocess run here with different
    # working directories: train.py and export_onnx.py inherit this one, while the
    # arena and search gate run with cwd=REPO so they can find the fixture. A relative
    # --out silently means different directories to each, and the failure surfaces as
    # an arena that cannot open a model rather than as a path error.
    args.out = args.out.resolve()

    selfplay_binary = REPO / "build" / "selfplay"
    search_binary = REPO / "build" / "gate_search"
    arena_binary = REPO / "build" / "arena"
    for binary in (selfplay_binary, search_binary, arena_binary):
        if not binary.exists():
            print(
                f"{binary.relative_to(REPO)} missing; reconfigure with "
                "-DONNXRUNTIME_ROOT=<path> and rebuild"
            )
            return 1

    shard_dir = args.out / "shards"
    model_dir = args.out / "models"
    for directory in (shard_dir, model_dir):
        directory.mkdir(parents=True, exist_ok=True)

    shards: list[pathlib.Path] = []
    history: list[dict] = []
    previous_model: pathlib.Path | None = None

    for generation in range(args.generations):
        source = "heuristic" if previous_model is None else f"gen{generation - 1}"
        print(f"\n=== generation {generation} (games from {source}) ===", flush=True)

        shard = shard_dir / f"gen{generation}.bin"
        selfplay(
            selfplay_binary,
            args.size,
            args.games,
            args.sims,
            args.seed + generation,
            shard,
            previous_model,
            args.threads,
        )
        shards.append(shard)
        window = shards[-args.window :]
        print(f"  self-play done; training on {len(window)} shard(s)", flush=True)

        run_dir = args.out / f"run{generation}"
        run(
            [
                sys.executable,
                str(pathlib.Path(__file__).with_name("train.py")),
                "--shards",
                *[str(path) for path in window],
                "--preset",
                args.preset,
                "--epochs",
                str(args.epochs),
                "--out",
                str(run_dir),
            ],
            "training",
        )

        onnx_path = model_dir / f"gen{generation}.onnx"
        run(
            [
                sys.executable,
                str(pathlib.Path(__file__).with_name("export_onnx.py")),
                "--board-size",
                str(args.size),
                "--preset",
                args.preset,
                "--checkpoint",
                str(run_dir / "best.pt"),
                "--output",
                str(onnx_path),
            ],
            "onnx export",
        )

        # The gate is absolute but 5x5-only; the arena is relative but works at any
        # size. Whichever is available gets reported, and at 9x9 that is the arena.
        network = score_network(onnx_path, args.size)
        entry: dict = {"generation": generation, "elo": "-"}

        if network is not None:
            passed, report = score_agent(search_binary, onnx_path, args.gate_sims)
            agent_hit = next(
                (
                    line.split(":")[1].split("%")[0].strip()
                    for line in report.splitlines()
                    if "agent plays a winner" in line
                ),
                "n/a",
            )
            entry.update(
                value=f"{network['value_balanced']:.1%}",
                policy=f"{network['policy_hit_rate']:.1%}",
                agent=f"{agent_hit}%",
                gate="pass" if passed else "FAIL",
            )
            print(
                f"  network: value {network['value_balanced']:.1%}, "
                f"policy {network['policy_hit_rate']:.1%} "
                f"(chance {network['policy_chance']:.1%})",
                flush=True,
            )
            print(
                f"  agent  : plays a winner {agent_hit}%  "
                f"{'pass' if passed else 'FAIL'}",
                flush=True,
            )
        else:
            entry.update(value="-", policy="-", agent="-", gate="n/a")
            print(
                f"  no {args.size}x{args.size} gate fixture: the solver cannot reach "
                "mid-game positions at this size, so progress comes from the arena",
                flush=True,
            )

        if previous_model is not None:
            elo = score_elo(arena_binary, onnx_path, previous_model, args.size,
                            args.pairs, args.gate_sims, args.threads)
            entry["elo"] = elo
            print(f"  arena  : vs gen{generation - 1}  elo {elo}", flush=True)

        history.append(entry)
        previous_model = onnx_path

    print("\n=== summary ===")
    print(f"{'gen':>3}  {'value':>7}  {'policy':>7}  {'agent':>7}  {'gate':>5}  elo vs previous")
    for row in history:
        print(
            f"{row['generation']:3d}  {row['value']:>7}  {row['policy']:>7}  "
            f"{row['agent']:>7}  {row['gate']:>5}  {row['elo']}"
        )

    if any(row["gate"] == "FAIL" for row in history):
        print("\na generation did not pass the search gate")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
