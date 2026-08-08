# hex9

[![ci](https://github.com/apurvapm/hex9/actions/workflows/ci.yml/badge.svg)](https://github.com/apurvapm/hex9/actions/workflows/ci.yml)

An AlphaZero-style agent for 9×9 Hex, trained offline and served entirely in the
browser. C++ engine compiled to WebAssembly, policy/value network exported to
ONNX, MCTS search tree visualised live as the agent thinks.

Status: **phase 3 complete** — game core, search, encoding, network, a training
loop gated against exhaustively solved positions, parallel self-play scaling to 15
cores, and Elo tracking between checkpoints. Phase 4 is the browser build.

## Build

```
cmake -B build && cmake --build build -j
./build/test_board
./build/test_mcts && ./build/test_alphabeta && ./build/test_puct
./build/test_parallel
./build/test_encoding training/encoding_fixture.txt
./build/gate --out=training/gate_fixture.txt
./build/bench_board
cd training && python -m pytest -q
```

Sanitiser build: `cmake -B build-asan -DHEX_SANITIZE=ON && cmake --build build-asan -j`

WebAssembly build, which runs the same suites under a different standard library:

```
source ~/opt/emsdk/emsdk_env.sh
emcmake cmake -B build-wasm -DCMAKE_BUILD_TYPE=Release && cmake --build build-wasm -j
node build-wasm/test_board.js
node tests/test_wasm_engine.mjs build-wasm/hex_engine.mjs
```

## Phase 0 — game core

Header-only, templated on board size so the same code serves the 9×9 target and
the tiny boards used for exhaustive verification.

**Board representation.** Axial coordinates on a rhombus; the six neighbours of
`(r, c)` are `(r±1, c)`, `(r, c±1)`, `(r-1, c+1)`, `(r+1, c-1)`. Red connects
row 0 to row N-1, Blue connects column 0 to column N-1.

**Win detection** is incremental union-find with four virtual nodes — one per
board edge. Placing a stone unions it with same-coloured neighbours and, if it
sits on a relevant edge, with that edge's virtual node. A player has won exactly
when their two edge nodes share a root. Each move costs a constant number of
`Unite` calls instead of a full flood fill.

Because the DSU forgoes path compression (see below), `Find()` is O(log n)
rather than the O(α) of the fully optimised structure. That distinction is
theoretical here: across 20k random games the deepest walk observed was **4
steps**, with a mean of **0.97**, against a log₂(85) ≈ 6.4 bound. Hex groups are
geographically local, so the trees stay flat.

**Rollback.** MCTS needs make/unmake, and path compression is not undoable, so
the DSU uses union-by-size with no compression and a merge stack. `Mark()` and
`RollbackTo()` bracket a move.

Compression could in principle be rolled back by logging every parent-pointer
rewrite, but that is self-defeating: the log would then grow with *reads* rather
than writes, and MCTS performs far more `Find` calls than `Unite` calls. The
amortisation that makes compression worthwhile is exactly what the logging
destroys. Union-by-size alone gives a worst case of O(log n) and, as measured
above, an observed depth under 5.

**Move generation** keeps a dense array of empty cells with a swap-removal that
parks the removed cell just past the live region, making undo a single swap
back. Legal-move enumeration is a contiguous read; no scanning.

**The swap (pie) rule** is supported. Hex is a first-player win on every N×N
board, so every competitive ruleset lets Blue answer Red's opening by taking
that stone. Transposing `(r, c) -> (c, r)` maps Red's top-bottom goal onto
Blue's left-right goal, so swap is implemented as transpose-plus-recolour rather
than as a special case in the turn logic. It is encoded as a sentinel action one
past the last cell, making the policy head `N² + 1` wide.

**Zobrist hashing** from a fixed SplitMix64 seed, so hashes are identical across
runs, machines, and the native and WASM builds. This matters for the
transposition table and for reproducible replays in the demo.

## Verification

The load-bearing test is the **Hex no-draw theorem**: any complete two-colouring
of the board has exactly one winner — never zero, never two. This is a strong
property, and almost every possible neighbourhood or connectivity bug violates
it. Each random colouring is also cross-checked against an independently written
flood-fill winner check.

Current results (all passing, clean under ASan and UBSan):

```
neighbour symmetry
  edges: 208 (expected 208)                    # 3N^2 - 4N + 1
no-draw theorem (random full colourings)
  20000 trials, red won 10018 (50.1%), no draws, no disagreements
incremental win detection during legal play
  3000 games, mean game length 71.1 plies
undo restores full state
  500 random games unwound to the empty position
zobrist hash distinguishes positions
  50000 distinct positions hashed, 0 collisions
swap rule
  2000 swapped games, mirror and undo verified
exhaustive solve of small boards
  2x2: first player wins (14 nodes)
  3x3: first player wins (3405 nodes)
  4x4: first player wins (11203672 nodes)

79796 checks, 0 failures
```

The small-board solves are ground truth, not a smoke test: Hex is a first-player
win on every N×N board by a strategy-stealing argument, so a solver that
disagrees has a bug in the rules, not a weak evaluation.

## Search

Two engines share the board core.

**MCTS** — UCT selection with uniform-random rollouts, arena-allocated nodes,
make/unmake rather than per-node board copies. Verified against the exhaustive
solver: on 4×4, where only 4 of 16 openings win, it selects a provably winning
move on 8 of 8 seeds.

**Alpha-beta** — negamax with a transposition table, iterative deepening, and
centre-first move ordering. It exists to be a fixed yardstick that never changes,
so agents from any later phase can be measured against a constant.

Hex has no material to count, so the evaluation is structural: the minimum number
of additional stones a player needs to complete a connection, where own stones
cost nothing to traverse, empty cells cost one, and opponent stones block. The
only weights are 0 and 1, so it is a deque-based 0-1 BFS rather than a Dijkstra.

Searching 4×4 to full depth reproduces the exhaustive solver's answer in
**147,873 nodes against the raw solver's 11,203,672** — a 76× reduction from
pruning, transpositions, and ordering.

Two measured results:

```
depth 4 beat depth 1 in 24/24 paired games (100%)
depth-4 alpha-beta won 19/20 against 1000-sim MCTS (95%)
```

The second is the more interesting one. Random rollouts are a poor value
estimator in Hex because the game is decided by connectivity structure, and
uniform-random play destroys exactly that signal — so a shallow search with a
structural evaluation outplays a far larger rollout budget. This is the concrete
argument for replacing rollouts with a learned value head rather than simply
buying more simulations.

The evaluation is deliberately simple and does not understand bridges, so it
undervalues positions a strong Hex player would read as already won. Shannon's
electrical-resistance evaluation and the two-distance metric are the natural
upgrades.

**PUCT** — the AlphaZero variant, where an evaluator supplies move priors and a
position value in place of rollouts. The evaluator is a template parameter, so
the identical tree code runs against a rollout, a heuristic, or a neural network.
Includes Dirichlet root noise and temperature sampling for self-play, and exposes
the root visit distribution as the policy training target.

Every sampler behind that — uniform, normal, gamma, Dirichlet — is hand-rolled,
because the `<random>` distribution classes are implementation-defined and would
break parity between the native and WebAssembly builds. Each is verified against
its analytic mean and variance.

How much search each evaluator needs to solve the same 4×4 opening:

```
heuristic @  2k: 8/8   rollout @  2k: 5/8   rollout @ 50k: 8/8
```

A structural evaluator reaches with 2,000 simulations what rollouts need 50,000
to match — a 25× difference in search budget from evaluation quality alone.

## Position encoding

Hex is not symmetric between players the way Go or chess are — Red connects top
to bottom, Blue connects left to right — so a position cannot be normalised by
swapping colours. But transposing `(r, c) → (c, r)` maps Red's goal onto Blue's,
which is the same isomorphism the swap rule is built on.

The encoder uses it to canonicalise: every position is presented from the
perspective of the player to move, oriented onto the top-bottom axis. The network
learns one goal direction instead of two, and a position and its mirror share an
evaluation exactly rather than approximately.

Three planes: the mover's stones, the opponent's stones, and a plane set when
swap is legal. That third plane is not redundant — after a swap the board holds
one stone at ply 2, which is indistinguishable in the first two planes from the
one-stone position at ply 1 where swap is still available.

The C++ driver and the Python trainer must produce byte-identical tensors, and a
mismatch is silent: loss falls smoothly while the network learns from misaligned
inputs. So `tests/test_encoding.cpp` writes a golden fixture of positions with
checksums over the resulting tensors, and `training/test_encoding.py` asserts the
Python encoder reproduces every one. Both run in CI.

## Network

A small residual tower with separate policy and value heads. The policy is
`N² + 1` wide — every cell plus the swap action — and the value is squashed to
`[-1, 1]` from the mover's perspective.

The Hex-specific choice is the convolution kernel. A cell on the rhombus has six
neighbours, at offsets `(-1,0) (-1,+1) (0,-1) (0,+1) (+1,-1) (+1,0)`. A standard
3×3 kernel spans eight, so two of its taps sit on cells that are not adjacent at
all. Those two weights are masked to zero, giving the network the real board
topology rather than asking it to learn that two of its inputs are meaningless.
The mask is a flag, so the choice can be ablated.

```
HexNet(board=9, channels=64, blocks=6, masked=True, params=464,637)
wrote hex9.onnx (1.78 MB)
onnx matches pytorch: policy max diff 7.82e-08, value max diff 4.47e-08
```

Export verification is not optional. A graph that silently changes behaviour —
BatchNorm left in training mode, an operator that folds differently — loads fine
in the browser and plays badly with nothing in the logs to explain it. The
exporter compares ONNX Runtime against PyTorch across several batch sizes before
writing, and refuses to emit a model whose weights spilled into a sidecar file,
since that deploys cleanly from disk and fails over HTTP.

## Training compute

There is no GPU in this project. Everything trains on CPU, so every number below
is reproducible by anyone who clones the repository.

Single-core ONNX Runtime throughput for the 9×9 network:

```
full  (464k params)  batch=1: 1,697   batch=32: 1,940   batch=128: 2,040 evals/sec
light (187k params)  batch=1: 4,226   batch=32: 4,497   batch=128: 4,810 evals/sec
```

Batching buys roughly 20% across a 128× range — the opposite of the GPU case,
where a single 9×9 position leaves the device nearly idle and batching is worth
an order of magnitude. A CPU core is already saturated by one convolution, so
batching only amortises call overhead.

That result determines the parallel design: independent self-play workers each
owning a single-threaded session, rather than a queue feeding one batching
evaluator. The batching machinery exists to keep a hungry accelerator fed, and
there is no accelerator here.

## Self-play

`tools/selfplay.cpp` plays games with PUCT and writes training records.

```
./build/selfplay --size=5 --heuristic --games=100 --sims=100 --out=shard.bin
./build/selfplay --size=5 --model=tiny.onnx --games=50 --sims=200 --out=shard.bin
```

ONNX Runtime is optional: pass `-DONNXRUNTIME_ROOT=<path>` to CMake to enable
network play. Without it the driver still runs against the connection-distance
heuristic, which is what the record tests use, so CI needs no extra dependency.

Records store move sequences plus per-move visit counts rather than encoded
tensors. Replaying is cheap, shards stay roughly twenty times smaller, and — the
real reason — the trainer reaches its inputs through exactly the same replay and
encode path that the golden fixture already pins. There is no second encoding to
drift.

The reader recomputes every game's winner with a flood fill. The engine derives
it incrementally through a union-find, so agreement between two independent
implementations is real evidence a shard is intact, rather than a value head
that quietly refuses to converge three days later.

## Training and the 5×5 gate

A falling loss is not evidence that a self-play trainer works. A value target with
the player-to-move sign inverted trains perfectly smoothly and then plays like
garbage, and so does a policy target misaligned with the canonical encoding. So
training is gated on positions whose true value and true winning moves are known.

Ground truth comes from sampled positions, not from whole small games. The empty
4×4 board costs 11.2M nodes, and solver cost rises about 12× per two stones
removed, which puts an empty 5×5 board several orders of magnitude out of reach.
Sampled positions are cheap and still mid-game — at 10 stones there are 15 empty
cells left:

```
5x5, stones placed     10       12       14       18
mean solve           0.18s     6ms      1ms    <1ms
mean nodes            2.2M     110k      19k     121
```

`tools/gate.cpp` solves 300 such positions for both their value and their full
set of winning moves; `training/gate.py` scores a checkpoint against them.

Two properties of that gate are load-bearing, because the obvious version of each
metric misleads.

**Value accuracy is balanced across outcome classes.** Random play leaves the
mover winning in about 73% of sampled positions, so a network that always answers
"winning" scores 73% raw. Averaging recall over the two classes scores that
strategy 50% — and a sign-inverted value head 0%, which is the number that
separates a broken trainer from a merely weak one.

**Policy accuracy is reported per board orientation.** A canonicalisation bug
corrupts only the positions that had to be transposed, so the aggregate stays
respectable and clears any single threshold. Measured, with canonicalisation
deliberately skipped:

```
                          aggregate   red as-is   blue mirrored
oracle                         100%        100%           100%
policy not canonicalised        65%        100%            20%
```

Won positions are also filtered for sharpness. About a third of randomly sampled
won positions have 90% or more of their legal moves winning, which drags the
policy metric's chance rate to 50% and makes a hit rate unreadable. Keeping only
positions at or below 0.30 puts chance near 14%.

Result after one generation of supervised training on heuristic self-play, and
the measured effect of data volume on it:

```
                             600 games    6000 games
value balanced accuracy          71.0%         84.8%   (always-majority 57.7%)
validation value loss             1.00          0.61
policy picks a winner            77.5%         89.6%   (chance 14.4%)
  red to move / blue to move  73.5/82.7     87.8/92.0
```

The value head is the part that needs data, and the reason is structural: a game
contributes one outcome shared by all of its positions, so 600 games of 13 plies
is 7,800 training rows but only 600 independent labels. At that volume the head
saturates its tanh and is confidently wrong 29% of the time; ten times the games
fixes it. The policy head, which gets a fresh target every ply, is far less
sensitive.

The train/validation split is by game rather than by position. Positions within
one game share almost their whole board, so a position-level split leaks and
produces a validation curve that tracks the training curve however badly the model
generalises.

### Gating the agent, not just the network

`gate.py` scores the raw policy and value. `build/gate_search` scores PUCT
*driving* that network, over the same fixture, which is the claim that actually
matters — search can repair a mediocre policy, and a mediocre policy can waste a
good search. At 200 simulations on the same checkpoint:

```
                          network alone   agent, 200 sims
value balanced accuracy           84.8%             97.7%
plays a winning move              89.6%             97.1%   (chance 14.4%)
  red to move / blue to move   87.8/92.0         95.9/98.7
```

That gap is the measured value of search on top of the policy, and it is the same
argument the rollout comparison makes from the other direction.

The search gate also covers a path nothing else did. The C++ encoder is pinned
against Python by the golden fixture, and PyTorch is pinned against ONNX Runtime
by the exporter, but `OnnxEvaluator`'s policy decode — softmax over legal actions,
then un-canonicalise back to board indices — sits between those two checks and was
covered by neither. A wrong mapping there collapses one board orientation and
leaves the other intact, which is exactly what the split rates would show.

`gate_search` needs an ONNX Runtime C++ distribution and is not built without one.
CI configures without it on purpose, so nothing in the required test path depends
on a 32 MB download. A separate CI job does install it, because otherwise the
ORT-only targets would be compiled by nothing and free to rot.

### The closed loop

`training/loop.py` runs the full cycle. Generation 0 bootstraps from the
connection-distance heuristic; every later generation plays its games with the
previous generation's ONNX export. Four generations, 1500 games each, 5×5:

```
gen   games from   network value   network policy   agent
  0    heuristic           76.7%            83.8%   96.0%
  1         gen0           79.0%            87.9%   96.5%
  2         gen1           81.7%            92.5%   98.3%
  3         gen2           82.4%            93.6%   97.7%
```

The network improves monotonically on both heads while learning only from games
its own predecessor played, which is the whole claim of self-play made
measurable.

The same loop at 9×9, four generations of 1500 games with the full 464k-parameter
network, measured over 50 colour-balanced pairs per match:

```
gen1 vs gen0   86-14 (86.0%)   elo +315 +/- 50
gen2 vs gen1   90-10 (90.0%)   elo +382 +/- 58
gen3 vs gen2   80-20 (80.0%)   elo +241 +/- 43
gen3 vs gen0   99-1  (99.0%)   elo +798 +/- 175
```

There is no gate at this size and there cannot be one: the solver reaches mid-game
positions on 5×5 but only nearly-decided endgames on 9×9, and a fixture of those would
pass trivially. Absolute correctness is established once at 5×5; progress at 9×9 is
relative, which is what the arena is for.

Worth noting the shape differs from 5×5. There the gains flattened — +413, +241, +196
— whereas at 9×9 the largest jump is the *second* step and the third is still +241.
The bigger board has not run out of room after four generations, so more of them are
worth buying. Total wall time was 2.5 hours on 15 cores.

The agent column in the 5×5 table is instructive for the opposite reason: it
saturates. Search at 200 simulations already solves 96% of these positions with a
generation-0 network, and the drop at generation 3 is one position out of 173.
That makes the search gate a good *correctness* check and a poor *progress*
metric — 5×5 is simply small enough that search covers for the policy. Reading
that column as a learning curve would be reading noise. Progress at 9×9 will need
arena matches between checkpoints, which is what Phase 3 adds.

Training restarts from scratch each generation over a sliding window of recent
shards rather than warm-starting. Warm-starting is faster and standard, but it
compounds whatever the previous generation got wrong, and at this size the retrain
costs minutes.

### Elo between checkpoints

The gate proves correctness but cannot measure progress, because search at a few
hundred simulations already solves 96% of the fixture with a generation-zero
network. Relative strength does not saturate, so `tools/arena.cpp` plays
checkpoints against each other:

```
gen1 vs gen0   183-17 (91.5%)   elo +413 +/- 44
gen2 vs gen1   160-40 (80.0%)   elo +241 +/- 31
gen3 vs gen2   151-49 (75.5%)   elo +196 +/- 29
gen3 vs gen0   196-4  (98.0%)   elo +676 +/- 88
```

Every generation separates from its predecessor by several standard errors, which
is the progress signal the gate could not give. The gains shrink — +413, +241,
+196 — matching the value-head plateau in the table above.

Two things this reports that are easy to get wrong.

**Colours alternate in pairs, and the pairing is not optional.** Red wins about
53% under uniform random play, so an unpaired match measures who moved first. Every
opening is played twice with colours exchanged and only the pair total counts. The
overall red win rate is printed as a check: it lands near 50% in these matches, and
a number far from it would mean colour rather than strength was driving the result.

**Elo carries a standard error and the tool refuses to imply significance without
one.** A match whose interval spans zero says so explicitly. And the numbers are
not additive: the consecutive gaps sum to 850 while gen3 against gen0 measures 676,
because a 98% win rate is near the resolution limit of 200 games. That last row is
better read as a lower bound than a point estimate.

Validation is a search-budget mismatch with one evaluator on both sides: 400
simulations against 25 gives 91.2% and +407 +/- 69, and reversing the sides gives
exactly 8.8% and -407. Identical agents cannot validate the arena, since every pair
then splits one win and one loss by construction.

## Parallel self-play

Self-play runs as independent games across a thread pool, each worker owning its
own single-threaded ONNX Runtime session. There is no shared tree, so there are no
atomics on node statistics and no virtual loss — the throughput comes from running
games side by side, which is what the flat CPU batching curve above implies.

```
./selfplay --size=9 --model=hex9.onnx --games=2000 --threads=15 --out=shard.bin
./bench_parallel --size=9 --model=hex9.onnx --games-per-worker=12 --sims=50
```

**Shards are bit-identical at any thread count.** Each game is seeded from
`(run seed, game index)` so its content depends only on its index, and completed
records are flushed in index order a block at a time. One thread or fifteen, the
same seed produces the same bytes — verified in CI across five thread and block
configurations, including the degenerate one game per block. Without that, every
replay buffer would quietly depend on scheduling.

### Scaling

9×9, full network, one session per worker, medians of three runs:

```
 threads    moves/sec   speedup   per thread   spread
       1         33.0     1.00x         100%       1%
       2         60.0     1.82x          91%       1%
       4        113.0     3.40x          85%       2%
       6        162.0     4.87x          81%       2%
       8        200.0     6.00x          75%       4%
      12        223.0     6.70x          56%      10%
      15        236.0     7.08x          47%       4%
```

**That 47% is not a contention result, and reporting it as one would be wrong.**
The machine is an M5 Pro: 15 logical cores, but 5 performance and 10 efficiency.
Pinning a single thread high versus to an efficiency core measures 30 against 9
moves/sec on this workload, a ratio near 3.3, so the achievable ceiling is
`5 + 10/3.3 ≈ 8×` rather than 15×. Against that denominator the driver reaches
about 88%. The ratio is workload-dependent — 5×5 with the heuristic evaluator gives
3.8 — so it is measured next to the curve rather than assumed.

### Workers or session threads

ONNX Runtime can parallelise inside a single inference call, so a fixed core budget
can buy many single-threaded workers or few multi-threaded ones. Same 15 cores:

```
 workers   intra-op    moves/sec   relative
      15          1        255.0      1.00x
       7          2        189.0      0.74x
       3          5         92.0      0.36x
       1         15         48.0      0.19x
```

Per-worker sessions win by **5.3×**, degrading monotonically as cores shift from
workers to intra-op threads. Put most sharply: intra-op parallelism turns 15 threads
into 1.45× on a 9×9 board (48 against 33 moves/sec single-threaded), while 15
independent games turn them into 7.08×. A 9×9 convolution is simply too small to
parallelise usefully.

The bias runs against the winner, which makes it a conservative result: fifteen
workers pay fifteen session constructions where one worker pays one.

### Parallelising one search

Self-play parallelises across games. When there is only one position to think about
— analysis, or a browser demo that wants a deeper tree per move — the threads have
to share. `include/hex/parallel_puct.hpp` implements the three standard answers so
they can be compared rather than argued about. 9×9, 4000 simulations, medians of
three:

```
mode    1 thread   15 threads   at 15 threads
root      456k       4548k         1.00x
tree      659k       2884k         0.63x
leaf      722k        817k         0.18x
```

Root-parallel wins on throughput, which is the right answer for a CPU: independent
trees share nothing and so synchronise nothing. Tree-parallel pays for one shared
tree and buys breadth. Leaf-parallel scales 1.13× across fifteen threads, because
only the evaluation is parallel and a 9×9 convolution is precisely the thing that
does not benefit — the same fact the batching curve reports from another angle.

Tree-parallel uses virtual loss to stop threads following each other down one
branch: a descent pretends to have lost, which is applied going down and removed
coming back up, so it steers selection without polluting the statistics that
outlive the search. Turning it off narrows the tree from 13,495 distinct nodes to
5,580 at the same budget.

Node statistics are atomics and expansion is claimed with a compare-exchange rather
than a lock; a thread that loses the claim backs up its value and moves on instead
of blocking. Value sums are fixed point, because `atomic<float>` has no `fetch_add`.

Read the absolute columns rather than the speedups: root-parallel's one-thread
figure is depressed by evaluation-counting instrumentation the other modes do not
pay, which inflates its speedup without affecting the 15-thread number. And
throughput is not strength — all three pick a provably winning 4×4 opening on 4 of
4 seeds, but ranking them by playing strength would need arena matches between
modes, which is not done.

### Verification

`tests/test_parallel.cpp` pins 34 checks over the self-play driver and 21 over the
search modes: reproducibility across thread counts and block sizes, in-order
flushing, statistics that agree with the records, solver agreement for all three
modes, and that virtual loss widens the tree. It runs under ASan and TSan in CI, so
the ordering argument in the headers is checked rather than trusted.

One test documents a precondition instead of a behaviour: reproducibility requires
the evaluator to be a pure function of the board. A stateful one — `RolloutEvaluator`
owns an RNG — makes a game's content depend on which worker claimed it. The test
demonstrates 35 of 40 games diverging, so the constraint stays visible rather than
being rediscovered.

Getting the solver-agreement test right was instructive in its own way. `Solver`
ignores the swap rule deliberately, so its winning moves are the swap-free ones.
Leave swap enabled and the search correctly learns to avoid exactly those moves,
because a strong opening under the swap rule is one the opponent simply takes: at
8000 simulations it puts over half its visits on a move the swap-free solver calls
losing, and it is right to. The test disables swap, as `test_puct.cpp` does.

Getting there turned up a memory-ordering result worth stating. The game-index
counter was `relaxed`, which is correct under the standard: the counter carries no
data, and `std::barrier` already orders a worker's writes before the completion step
that reads them. TSan reported six races anyway. A minimal repro with a per-slot
ownership check found zero actual collisions while the reports appeared and vanished
purely with the memory order, so libc++'s barrier does not give TSan that edge. The
counter is now `acq_rel`, which supplies an edge the tool tracks, costs one
read-modify-write per game against a game measured in milliseconds, and left
throughput unchanged. An unverifiable optimisation is worth less than a free one.

## Playing against it

```
cmake -B build && cmake --build build -j
./build/play                                    # 9x9 against MCTS
./build/play --opponent=alphabeta --depth=4
./build/play --size=7 --sims=50000 --colour=blue --swap
```

Moves use Hex notation — a column letter then a row number, such as `e5`. Also
accepted: `swap`, `hint`, `undo`, `quit`.

```
     a b c d e f g h i
  1  . . . . . . . . . 1
   2  . . . . . . . . . 2
    3  . . . . . . . . . 3
     4  . . . . B . . . . 4
      5  . . . . R . . . . 5
       6  . . . . . . . . . 6
        7  . . . . . . . . . 7
         8  . . . . . . . . . 8
          9  . . . . . . . . . 9
             a b c d e f g h i
```

Each row is indented one space further than the last, which is what makes the
six-neighbour adjacency legible: the cell below-left and below-right of a stone
are its neighbours, the cell directly below is not.

## Baseline throughput

Uniform-random playouts, single core. Native is `-O3 -march=native`; WebAssembly is
`-O3 -msimd128`, single-threaded, no SharedArrayBuffer:

```
                  native      wasm+simd
playouts/sec      403679        344784      85%
moves/sec          28.58M        24.48M
mean game length     71.0          71.0
red win rate        53.0%         53.0%
```

WebAssembly reaches 85% of native here. Identical game length and win rate on both
sides is the more reassuring half of that table: the hand-rolled samplers mean the
two builds play the same games, not merely a similar number of them.

**Both columns must be measured in one session on one machine.** An earlier revision
of this file published 135,054 playouts/sec, measured on different hardware; pairing
that with the WebAssembly figure above would have implied WebAssembly runs 2.5×
faster than native. The benchmark panel in phase 5 has to produce both numbers from
the same run for the same reason.

Design decisions and their rationale are in [docs/DESIGN.md](docs/DESIGN.md).

## Roadmap

- [x] Phase 0 — game core, rollback DSU, Zobrist, verification
- [x] Phase 1a — plain MCTS with UCT and random rollouts, verified against the
      exhaustive solver
- [x] Phase 1b — alpha-beta baseline with a connection-distance evaluation
- [x] Phase 1c — terminal CLI
- [ ] Phase 1d — 180° rotational symmetry in the transposition table
- [x] Phase 2a — PUCT search with pluggable evaluator, root noise, temperature
- [x] Phase 2b — canonical position encoding, verified across C++ and Python
- [x] Phase 2c — policy/value network with hex-masked convolutions, verified
      ONNX export
- [x] Phase 2d — self-play driver, record format, cross-validated reader
- [x] Phase 2e — training loop and the 5×5 gate against perfect play. One
      generation; closing the loop so the trained net generates the next batch of
      games needs an ONNX Runtime C++ distribution
- [x] Phase 3 — parallel self-play: a thread pool over independent games, one
      single-threaded ORT session per worker rather than a queue feeding a
      batching evaluator, virtual-loss tree search, clean under TSan. Scaling
      curve, workers-versus-session-threads comparison, tree/root/leaf parallel
      comparison, arena matches and Elo between checkpoints
- [ ] Phase 4 — Emscripten build, ONNX Runtime Web, board and tree UI
- [ ] Phase 5 — benchmark panel, checkpoint scrubber, deploy
