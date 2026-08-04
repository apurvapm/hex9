# Design notes

Decisions that constrain future work, and the reasoning behind them. The README
covers what the code does; this covers why it is shaped the way it is.

## Invariants

**The board is templated on size.** `Board<N>` exists so that tiny boards can be
solved exhaustively and used as ground truth. Hex is a first-player win on every
N×N board by strategy stealing, so a solver that disagrees has a rules bug rather
than a weak evaluation. This is the only source of absolute correctness in the
project and it is worth the template.

**Win detection is incremental union-find**, with four virtual nodes standing in
for the board edges, rather than a flood fill per move.

**The disjoint-set structure has no path compression.** MCTS needs make and
unmake, and compression is not undoable. Union-by-size plus a merge stack gives
rollback at a worst case of O(log n). Compression could in principle be undone by
logging every parent-pointer rewrite, but that is self-defeating: the log would
grow with reads rather than writes, and search performs far more finds than
unions. The amortisation that makes compression worthwhile is exactly what the
logging destroys. Measured depth is under 5 in practice.

**Move generation uses a dense empty-cell array** with a swap-removal that parks
the removed cell just past the live region, so undo is a single swap back.

**Zobrist keys derive from a fixed SplitMix64 seed.** Hashes must be identical
across runs, machines, and the native and WebAssembly builds, so nothing is ever
seeded from the clock or from `random_device`.

**Randomness never goes through `std::shuffle` or the `<random>` distribution
classes** where results must be comparable across builds. Those are
implementation-defined and consume random bits differently under libstdc++ and
libc++ — and the native build uses libstdc++ while Emscripten uses libc++. The
project uses `mt19937` directly with hand-rolled helpers (`NextBelow`,
`NextBool`, `Shuffle` in the test suite). This applies to Dirichlet noise and
temperature sampling in self-play, not only to tests.

**Geometry.** Red connects row 0 to row N-1; Blue connects column 0 to column
N-1. Neighbours are the six axial offsets in `detail::kNeighbourOffsets`.

**Swap is a board transformation, not a turn-logic special case.** Transposing
`(r, c) → (c, r)` maps Red's goal onto Blue's, so the swap rule is implemented as
transpose-plus-recolour and encoded as a sentinel action one past the last cell.
The policy head is therefore `N² + 1` wide and swap is an ordinary action to the
network.

## Verification

Correctness comes from properties rather than from example positions.

The load-bearing test is the **Hex no-draw theorem**: any complete two-colouring
of the board has exactly one winner, never zero and never two. Almost every
possible neighbourhood or connectivity bug violates it. Each random colouring is
additionally cross-checked against an independently written flood fill.

Search is verified against the exhaustive solver rather than against itself. On
4×4, only 4 of 16 openings win, so an agent choosing a provably winning move
across eight independent seeds is a discriminating result rather than a
formality.

The board suite pins its expected check count and exits non-zero on a mismatch.
The guard exists to catch random-stream drift between toolchains, which otherwise
presents exactly like an engine bug. When tests are added the constant is
recomputed deliberately and the change noted in the commit message.

Both suites run clean under AddressSanitizer and UndefinedBehaviorSanitizer in
CI.

Anything new is validated on 5×5 or smaller, where the game is exhaustively
solvable, before being scaled to 9×9. This matters most for self-play: a wrong
player-to-move sign in the value target trains smoothly, shows falling loss, and
plays like garbage. Small-board ground truth is the only reliable defence.

## Parallel self-play

Training runs on CPU. There is no GPU anywhere in the project, which keeps every
number in the README reproducible by anyone who clones the repository.

Measured single-core ONNX Runtime throughput for the 9×9 network:

```
full  (464k params)  batch=1: 1,697   batch=32: 1,940   batch=128: 2,040 evals/sec
light (187k params)  batch=1: 4,226   batch=32: 4,497   batch=128: 4,810 evals/sec
```

Batching buys about 20% across a 128× range. That is the opposite of the GPU
case, where a single 9×9 position leaves the device almost idle and batching is
worth an order of magnitude or more. A CPU core is already saturated by one
convolution, so batching only amortises call overhead.

This determines the architecture. Rather than a bounded queue feeding one
batching evaluator — which exists to keep a hungry accelerator fed — self-play
runs as independent workers, each owning its own single-threaded ORT session.
That parallelises close to linearly and removes the queue entirely.

Requirements:

- A thread pool over independent self-play games, one ORT session per worker
  with `intra_op_threads = 1`. Per-worker sessions avoid the contention that
  arises when several threads share one session's internal thread pool.
- A comparison against the alternative: fewer workers, each with a
  multi-threaded session. Report which wins and why. The expectation is that
  per-worker sessions win, because ORT's intra-op parallelism scales worse
  across small convolutions than embarrassingly parallel games do.
- Batch-size scaling curves, measured rather than assumed. The flat CPU curve
  above is a result worth reporting alongside the reason for it.
- Virtual loss where workers share a tree, applied on selection and reverted on
  backup. Relaxed atomics for visit counts and value sums, with the
  memory-ordering choice documented.
- Lock-free or fine-grained statistics on shared nodes. Compare tree-parallel
  against root-parallel and leaf-parallel on the same hardware.

Deliverables are a scaling curve to the machine's core count, a contention
analysis, and a note on the bugs found along the way. ThreadSanitizer runs in CI
over a reduced self-play workload, and a clean run is part of the definition of
done.

Threaded self-play is not reproducible, so a `--threads=1` path stays bit-for-bit
deterministic given a seed, with each game seeded from `(run_seed, game_index)`
rather than from a shared generator. The single-threaded driver already does
this.

Budget on a modern laptop, at roughly 50k evaluations per second across all
cores: the 5×5 validation gate is about twenty minutes, and a 9×9 agent strong
enough to beat a casual player is an overnight run. Superhuman play is not the
target; the demo needs an opponent that is fun to lose to.

## Browser demo

The demo is embedded in a Hugo site through a cross-origin iframe, so the page is
not cross-origin isolated and the WebAssembly build is single-threaded. Enabling
cross-origin isolation to unlock `SharedArrayBuffer` would apply to the whole
page and require CORP headers on every embedded resource, which is a poor trade
for threads the demo does not need. A threaded Emscripten build exists locally
only to produce benchmark numbers.

The demo has roughly a forty-second budget on a phone, so legibility outranks
feature count. The default view is the board alone, with the agent ready to play.
A single toggle reveals the live search tree, the policy heatmap, the value
readout, and a top-k move table showing prior, visits, value, and PUCT score.
Simulation count, `c_puct`, and temperature sit behind a further toggle;
simulation count doubles as the difficulty control. A deterministic seed and a
replay button are always available.

Explanations of MCTS, PUCT, and self-play belong in the accompanying writeup, not
inside the demo. At 380px the tree is unreadable, so mobile renders the board
alone with the analysis panels collapsed beneath it, and work per frame is capped
so low-end devices degrade rather than stall.

Deliberately out of scope: side-by-side games between early and late agents, and
self-play replay. Both cost real work and add little for a visitor who does not
already know Hex, and the checkpoint scrubber already answers whether training
worked, interactively rather than as a spectacle.

## Style

C++20 throughout, standard headers only. Google-ish naming: `PascalCase` methods,
`snake_case_` private members, `kConstant`, two-space indent. The core is
header-only and depends on nothing beyond the standard library. Comments explain
why rather than what.

## Open items

Hex positions have a 180° rotational symmetry that preserves colours.
Canonicalising via `min(hash, rotated_hash)` should roughly halve the search in
the alpha-beta baseline; the 4×4 exhaustive solve currently costs 11.2M nodes.

First player wins about 53% under uniform random play, so arena evaluation plays
paired games with colours alternated. An unpaired match measures colour
assignment rather than strength.
