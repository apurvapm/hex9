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

Self-play generation is the throughput bottleneck and is where the project
demonstrates concurrent systems work, so the following are requirements rather
than later optimisation.

- A thread pool over hundreds of concurrent self-play games. Workers descend
  their own trees and emit unevaluated leaf positions.
- A bounded multi-producer, multi-consumer queue feeding a single batching
  evaluator, with backpressure when the GPU falls behind. Workers block rather
  than let the queue grow without limit.
- Batch formation on a deadline: the evaluator waits a bounded time to fill a
  batch, then fires with whatever it has. Batch size and wait window are exposed
  as knobs, and the throughput-versus-latency curve across them is the
  interesting result rather than any single number.
- Virtual loss in tree-parallel search, so concurrent workers do not all descend
  the same path. Applied on selection, reverted on backup.
- Relaxed atomics for visit counts and value sums, with the memory-ordering
  choice documented.
- A comparison of tree-parallel against root-parallel and leaf-parallel on the
  same hardware.

Deliverables are a scaling curve to at least 32 threads, a contention analysis,
and a note on the bugs found along the way. ThreadSanitizer runs in CI over a
reduced self-play workload, and a clean run is part of the definition of done.

Threaded self-play is not reproducible, so a `--threads=1` path stays bit-for-bit
deterministic given a seed, with each game seeded from `(run_seed, game_index)`
rather than from a shared generator.

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
