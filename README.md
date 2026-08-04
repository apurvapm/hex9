# hex9

[![ci](https://github.com/YOURUSER/hex9/actions/workflows/ci.yml/badge.svg)](https://github.com/YOURUSER/hex9/actions/workflows/ci.yml)

An AlphaZero-style agent for 9×9 Hex, trained offline and served entirely in the
browser. C++ engine compiled to WebAssembly, policy/value network exported to
ONNX, MCTS search tree visualised live as the agent thinks.

Status: **phase 1 in progress** — game core and search complete.

## Build

```
cmake -B build && cmake --build build -j
./build/test_board
./build/bench_board
```

Sanitiser build: `cmake -B build-asan -DHEX_SANITIZE=ON && cmake --build build-asan -j`

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

Uniform-random playouts, single core, `-O3 -march=native`:

```
playouts/sec    : 135054
moves/sec       : 9.59 M
mean game length: 71.0 plies
red win rate    : 53.0%   (first player, random play)
```

This is the reference number for the native-vs-WASM benchmark panel in phase 5.

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
- [ ] Phase 2e — training loop and the 5×5 gate against perfect play
- [ ] Phase 2 — self-play loop validated on 5×5 against exhaustive ground truth
- [ ] Phase 3 — parallel self-play: thread pool, bounded MPMC queue with
      backpressure, virtual-loss tree search, clean under TSan. Scaling curve
      and contention analysis. Arena matches and Elo between checkpoints
- [ ] Phase 4 — Emscripten build, ONNX Runtime Web, board and tree UI
- [ ] Phase 5 — benchmark panel, checkpoint scrubber, deploy
