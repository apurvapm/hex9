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
libc++, so one seed diverges across a Linux host, a macOS host, and Emscripten.
The project uses `mt19937` directly with hand-rolled helpers (`NextBelow`,
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

Anything new is validated on 5×5 or smaller before being scaled to 9×9. This
matters most for self-play: a wrong player-to-move sign in the value target
trains smoothly, shows falling loss, and plays like garbage. Small-board ground
truth is the only reliable defence.

**Ground truth comes from sampled positions, not from whole small games.** The
empty 4×4 board solves in 11.2M nodes; cost rises about 12× per two stones
removed, so an empty 5×5 board is out of reach by several orders of magnitude. A
5×5 position with 10 stones placed solves in 0.18s and still has 15 empty cells,
which is ample room for a search to be wrong in. `tools/gate.cpp` samples such
positions, solves each for both its true value and its full set of winning moves,
and writes a fixture that `training/gate.py` scores a checkpoint against.

Two properties of that gate are load-bearing, and both exist because the obvious
version of the metric is misleading:

*Value accuracy is balanced across the two outcome classes.* Random play leaves
the mover winning in about 73% of sampled positions, so a network that always
answers "winning" scores 73% raw. Averaging recall over the classes scores that
strategy at 50%, and a sign-inverted value head at 0% — which is the number that
distinguishes a broken trainer from a merely weak one.

*Policy accuracy is reported separately for the two board orientations.* A
canonicalisation bug corrupts only the positions that had to be transposed, so
the aggregate hit rate stays respectable and clears any single threshold. In
measurement, skipping canonicalisation left the aggregate at 65% while the
transposed half collapsed from 100% to 20%. The split is what makes that
visible; nothing else produces a gap between two halves the network cannot
distinguish.

Won positions are also filtered for sharpness. About a third of randomly sampled
won positions have 90% or more of their legal moves winning, which drags the
policy metric's chance rate to 50% and makes a hit rate unreadable. Keeping only
positions at or below 0.30 puts chance near 14%.

The gate has two layers over one fixture. `training/gate.py` scores the raw
network; `tools/gate_search.cpp` scores PUCT driving that network, which is the
claim that matters, since search can repair a mediocre policy and a mediocre
policy can waste a good search. Measured on the same checkpoint, the agent reaches
97.1% against the network's 89.6%.

Keeping the fixture generator free of ONNX Runtime is deliberate. Ground truth has
to be buildable in CI with no extra dependency, so `tools/gate.cpp` runs only the
solver and `gate_search` is added only when `ONNXRUNTIME_ROOT` is set. The fixture
is also byte-reproducible from a seeded generator, and CI regenerates and diffs it
— the cheap drift diagnostic that the cross-toolchain RNG incident argues for.

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
  multi-threaded session. **Measured, and per-worker sessions win by 5.3×.** At a
  fixed budget of 15 cores on the 9×9 network:

  ```
  workers  intra-op  moves/sec  relative
       15         1      255      1.00x
        7         2      189      0.74x
        3         5       92      0.36x
        1        15       48      0.19x
  ```

  Degradation is monotonic as cores move from workers to intra-op threads. The
  sharpest way to put it: ORT's intra-op parallelism turns 15 threads into 1.45×
  on a 9×9 board (48 against 33 moves/sec single-threaded), while 15 independent
  games turn them into 7.08×. A 9×9 convolution is too small to parallelise
  usefully, which is the same fact the flat batching curve above reports from
  another angle.

  Note the bias runs against the winner: 15 workers pay fifteen ORT session
  constructions and one worker pays one, so this understates the gap.
- Batch-size scaling curves, measured rather than assumed. The flat CPU curve
  above is a result worth reporting alongside the reason for it.
- Virtual loss where workers share a tree, applied on selection and reverted on
  backup, with the memory-ordering choice documented. Note this applies only to
  tree-parallel search: self-play across *independent* games shares no tree, so
  there are no shared node statistics and nothing for virtual loss to do. The two
  are separate axes and the independent-games driver deliberately has neither.

  The ordering choice already made, for the game-index counter in
  `include/hex/parallel_selfplay.hpp`, is `acq_rel` rather than the relaxed default
  this list assumed. Relaxed is correct under the standard, since the counter
  carries no data and `std::barrier` orders a worker's writes before the completion
  step that reads them — but ThreadSanitizer does not derive that edge from
  libc++'s barrier and reported races on data that a per-slot ownership check
  proved was never shared. `acq_rel` supplies an edge the tool does track, costs
  one read-modify-write per game against a game measured in milliseconds, and left
  throughput unchanged at 1, 5, and 15 threads. Prefer the verifiable ordering when
  the stronger one is free.
- Lock-free or fine-grained statistics on shared nodes. **Done** in
  `include/hex/parallel_puct.hpp`: node statistics are atomics, and expansion is
  claimed with a compare-exchange on a three-state field rather than a lock. A
  thread that loses the claim does not wait — it has already evaluated the
  position, so it backs that value up and moves on, costing one duplicated
  evaluation and no blocking. The winner publishes `first_child`, `num_children`
  and every initialised child with one release store, which is what lets selectors
  read them as plain values after an acquire.

  Value sums are fixed point at 1e6 rather than float, because `atomic<float>` has
  no `fetch_add` and a compare-exchange loop on the hottest counter in the search
  is worse than integer scaling.

- **Compared on the same hardware.** 9×9, 4000 simulations, heuristic evaluator,
  medians of three:

  ```
  mode    1 thread   15 threads   at 15 threads
  root      456k       4548k         1.00x
  tree      659k       2884k         0.63x
  leaf      722k        817k         0.18x
  ```

  Root-parallel wins on raw throughput, which is the expected answer for a CPU: it
  shares nothing, so it synchronises nothing. Tree-parallel pays for a shared tree
  and gets breadth in return. Leaf-parallel barely scales at all — 1.13× across
  fifteen threads — because only the evaluation is parallel and a 9×9 convolution
  is the one thing that does not need parallelising. That is the same fact the flat
  batching curve reports.

  Two caveats, both of which would otherwise flatter one mode:

  Compare the absolute columns, not the speedups. Root-parallel's one-thread number
  is depressed by an evaluation-counting wrapper on its hot path that the other
  modes do not pay, which inflates its speedup while leaving the 15-thread figure
  sound.

  The tree-node counts these modes report are not comparable. Root-parallel sums
  node counts across independent trees and counts duplicates, so its ~305k is an
  upper bound on coverage; tree-parallel's is one tree and exact. Tree-parallel's
  count *falls* with thread count (302k at one thread to 172k at fifteen) because
  threads increasingly collide on the same leaf and lose the expansion claim. That
  is a real cost of sharing a tree and worth reporting as one.

  Throughput is not strength. All three choose a provably winning 4×4 opening on
  4 of 4 seeds in `tests/test_parallel.cpp`, but ranking them by playing strength
  would need arena matches between modes, which is not done.

Deliverables are a scaling curve, a contention analysis, and a note on the bugs
found along the way.

**The scaling curve must name its denominator.** The development machine is an M5
Pro: 15 logical cores, but 5 performance and 10 efficiency. For the 9×9 network,
single-thread throughput is 30 moves/sec pinned high against 9 pinned to an
efficiency core, a ratio near 3.3, so the achievable ceiling is about
`5 + 10/3.3 = 8×` and not 15×. Measured scaling reaches 7.08× at 15 threads:
roughly 88% of what the hardware can give, which reads as 47% against core count.
Reporting the latter would attribute to contention what is really core asymmetry.

That ceiling is approximate and workload-dependent — the 5×5 heuristic sweep gives
a 3.8 ratio and a 7.6× ceiling, then measures 7.87×, slightly above it, because
`taskpolicy` does not pin strictly. Measure the ratio next to whatever curve is
being published rather than treating one number as a property of the machine.

Curves also need repeated samples. The scheduler's placement of small thread counts
varies between runs, so single samples came out non-monotonic; `bench_parallel`
reports medians with a spread column. ThreadSanitizer runs in CI
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

Upstream ONNX Runtime Web ships no single-threaded wasm binary: there is one
`-threaded` build that runs single-threaded when SharedArrayBuffer is absent. The demo
pins `numThreads = 1` so that is chosen rather than inherited. Parity with PyTorch is
verified under node (`tests/test_ortweb_parity.mjs`), but node *has*
SharedArrayBuffer, so the no-SAB path the deployed page relies on remains unverified
and needs a hosted browser check.

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
the alpha-beta baseline, which currently reaches full depth on 4×4 in 147,873
nodes. The raw solver's 11.2M nodes for the same position is a different
baseline, and not the one symmetry would improve.

First player wins about 53% under uniform random play, so arena evaluation plays
paired games with colours alternated. An unpaired match measures colour
assignment rather than strength.
