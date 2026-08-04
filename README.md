# hex9

[![ci](https://github.com/apurvapm/hex9/actions/workflows/ci.yml/badge.svg)](https://github.com/apurvapm/hex9/actions/workflows/ci.yml)

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
- [ ] Phase 2b — network, self-play driver, training loop validated on 5×5
- [ ] Phase 2 — self-play loop validated on 5×5 against exhaustive ground truth
- [ ] Phase 3 — parallel self-play: thread pool, bounded MPMC queue with
      backpressure, virtual-loss tree search, clean under TSan. Scaling curve
      and contention analysis. Arena matches and Elo between checkpoints
- [ ] Phase 4 — Emscripten build, ONNX Runtime Web, board and tree UI
- [ ] Phase 5 — benchmark panel, checkpoint scrubber, deploy
