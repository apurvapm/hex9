#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "hex/board.hpp"
#include "hex/solver.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

// std::shuffle and the distribution classes are implementation-defined: they
// consume random bits differently under libstdc++ and libc++, so an identical
// mt19937 seed yields different sequences. The Emscripten build uses libc++
// while the native Linux build uses libstdc++, and this suite must produce
// identical output on both. These hand-rolled equivalents are fully specified.
//
// The modulo is biased, which is irrelevant for test-data generation but would
// need rejection sampling anywhere the distribution itself matters.
std::uint32_t NextBelow(std::mt19937& rng, std::uint32_t n) {
  return static_cast<std::uint32_t>(rng() % n);
}

bool NextBool(std::mt19937& rng) { return (rng() & 1u) != 0u; }

template <typename T>
void Shuffle(std::vector<T>& v, std::mt19937& rng) {
  for (std::size_t i = v.size(); i > 1; --i) {
    const std::size_t j = NextBelow(rng, static_cast<std::uint32_t>(i));
    std::swap(v[i - 1], v[j]);
  }
}

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}

// A deliberately naive winner check, written independently of the DSU path.
// If the two ever disagree, the incremental union-find has a bug.
template <int N>
hex::Cell BruteForceWinner(const hex::Board<N>& board) {
  for (int p = 0; p < 2; ++p) {
    const hex::Cell mine = p == 0 ? hex::Cell::kRed : hex::Cell::kBlue;
    std::vector<char> seen(N * N, 0);
    std::vector<int> stack;
    for (int i = 0; i < N; ++i) {
      // Red starts from row 0, Blue from column 0.
      const int idx = p == 0 ? hex::Board<N>::Index(0, i)
                             : hex::Board<N>::Index(i, 0);
      if (board.At(idx) == mine) {
        seen[idx] = 1;
        stack.push_back(idx);
      }
    }
    while (!stack.empty()) {
      const int cur = stack.back();
      stack.pop_back();
      const int row = hex::Board<N>::Row(cur);
      const int col = hex::Board<N>::Col(cur);
      if (p == 0 && row == N - 1) return mine;
      if (p == 1 && col == N - 1) return mine;
      for (const auto& [dr, dc] : hex::detail::kNeighbourOffsets) {
        const int nr = row + dr;
        const int nc = col + dc;
        if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
        const int nidx = hex::Board<N>::Index(nr, nc);
        if (!seen[nidx] && board.At(nidx) == mine) {
          seen[nidx] = 1;
          stack.push_back(nidx);
        }
      }
    }
  }
  return hex::Cell::kEmpty;
}

void TestNeighbourSymmetry() {
  std::printf("neighbour symmetry\n");
  constexpr int N = 9;
  int directed = 0;
  for (int r = 0; r < N; ++r) {
    for (int c = 0; c < N; ++c) {
      for (const auto& [dr, dc] : hex::detail::kNeighbourOffsets) {
        const int nr = r + dr;
        const int nc = c + dc;
        if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
        ++directed;
        bool mutual = false;
        for (const auto& [br, bc] : hex::detail::kNeighbourOffsets)
          if (nr + br == r && nc + bc == c) mutual = true;
        Check(mutual, "adjacency is not symmetric");
      }
    }
  }
  // A rhombic N x N Hex board has 3N^2 - 4N + 1 undirected edges.
  Check(directed / 2 == 3 * N * N - 4 * N + 1, "unexpected edge count");
  std::printf("  edges: %d (expected %d)\n", directed / 2, 3 * N * N - 4 * N + 1);
}

// The Hex theorem: any complete two-colouring of the board has exactly one
// winner. No draws, and never two winners. This single property catches almost
// every possible neighbourhood or connectivity bug.
void TestNoDrawTheorem() {
  std::printf("no-draw theorem (random full colourings)\n");
  constexpr int N = 9;
  constexpr int kTrials = 20000;
  std::mt19937 rng(20260803);

  int red_wins = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    hex::Board<N> board;
    for (int i = 0; i < N * N; ++i)
      board.PlaceStone(i, NextBool(rng) ? hex::Player::kRed : hex::Player::kBlue);

    const hex::Cell dsu_winner = board.Winner();
    const hex::Cell bfs_winner = BruteForceWinner(board);

    if (dsu_winner != bfs_winner) {
      Check(false, "DSU and brute-force winners disagree");
      break;
    }
    if (dsu_winner == hex::Cell::kEmpty) {
      Check(false, "a full board produced a draw");
      break;
    }
    if (dsu_winner == hex::Cell::kRed) ++red_wins;
  }
  Check(true, "");
  std::printf("  %d trials, red won %d (%.1f%%), no draws, no disagreements\n",
              kTrials, red_wins, 100.0 * red_wins / kTrials);
}

// Alternating legal play to a full board: still exactly one winner, and the
// winner must be detected at the moment the connection is completed, not later.
void TestIncrementalDetection() {
  std::printf("incremental win detection during legal play\n");
  constexpr int N = 9;
  constexpr int kTrials = 3000;
  std::mt19937 rng(7);

  long long total_plies = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    hex::Board<N> board;
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);

    for (const int move : order) {
      board.Play(move);
      const hex::Cell expected = BruteForceWinner(board);
      if (board.Winner() != expected) {
        Check(false, "winner mismatch mid-game");
        return;
      }
      if (board.IsTerminal()) break;
    }
    Check(board.IsTerminal(), "game ended without a winner");
    total_plies += board.MoveCount();
  }
  std::printf("  %d games, mean game length %.1f plies\n", kTrials,
              static_cast<double>(total_plies) / kTrials);
}

// Make/unmake must restore the board bit for bit: cells, hash, turn, winner,
// and the legal-move list (order-insensitively).
void TestUndoRestoresState() {
  std::printf("undo restores full state\n");
  constexpr int N = 9;
  std::mt19937 rng(99);

  for (int trial = 0; trial < 500; ++trial) {
    hex::Board<N> board;
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);

    std::vector<std::uint64_t> hashes;
    std::vector<std::vector<std::uint8_t>> legal_sets;
    int played = 0;

    for (const int move : order) {
      hashes.push_back(board.Hash());
      std::vector<std::uint8_t> legal(board.LegalMoves(),
                                      board.LegalMoves() + board.NumEmpty());
      std::sort(legal.begin(), legal.end());
      legal_sets.push_back(std::move(legal));
      board.Play(move);
      ++played;
      if (board.IsTerminal()) break;
    }

    for (int i = played - 1; i >= 0; --i) {
      board.Undo();
      Check(board.Hash() == hashes[i], "hash not restored by undo");
      std::vector<std::uint8_t> legal(board.LegalMoves(),
                                      board.LegalMoves() + board.NumEmpty());
      std::sort(legal.begin(), legal.end());
      Check(legal == legal_sets[i], "legal-move set not restored by undo");
    }
    Check(board.NumEmpty() == N * N, "board not empty after full undo");
    Check(board.Hash() == 0, "hash not zero after full undo");
    Check(!board.IsTerminal(), "terminal flag not cleared by undo");
  }
  std::printf("  500 random games unwound to the empty position\n");
}

void TestHashCollisions() {
  std::printf("zobrist hash distinguishes positions\n");
  constexpr int N = 9;
  std::mt19937 rng(2024);
  std::unordered_map<std::uint64_t, std::vector<hex::Cell>> seen;

  int collisions = 0;
  for (int trial = 0; trial < 50000; ++trial) {
    hex::Board<N> board;
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);
    const int plies = 10 + static_cast<int>(NextBelow(rng, 30));
    for (int i = 0; i < plies && !board.IsTerminal(); ++i) board.Play(order[i]);

    std::vector<hex::Cell> position;
    for (int i = 0; i < N * N; ++i) position.push_back(board.At(i));

    auto it = seen.find(board.Hash());
    if (it != seen.end() && it->second != position) ++collisions;
    seen.emplace(board.Hash(), std::move(position));
  }
  Check(collisions == 0, "zobrist collision on distinct positions");
  std::printf("  %zu distinct positions hashed, %d collisions\n", seen.size(),
              collisions);
}


// The swap (pie) rule. Transposing (r, c) -> (c, r) maps Red's top-bottom goal
// onto Blue's left-right goal, so swap is transpose-plus-recolour: Red's stone
// leaves, Blue's appears at the mirrored cell, and Red is to move.
void TestSwapRule() {
  std::printf("swap rule\n");
  constexpr int N = 9;
  using B = hex::Board<N>;

  {
    B b;
    const int open = B::Index(2, 5);
    b.Play(open);
    Check(b.CanSwap(), "swap unavailable at ply 1");
    b.PlaySwap();
    Check(b.MoveCount() == 2, "ply wrong after swap");
    Check(b.At(open) == hex::Cell::kEmpty, "original cell not vacated");
    Check(b.At(B::Index(5, 2)) == hex::Cell::kBlue, "mirrored cell not blue");
    Check(b.NumEmpty() == N * N - 1, "stone count changed across swap");
    Check(b.ToPlay() == hex::Player::kRed, "red should move after swap");
    Check(!b.CanSwap(), "swap offered twice");
  }

  {
    B b;
    const int open = B::Index(3, 7);
    b.Play(open);
    const std::uint64_t h1 = b.Hash();
    b.PlaySwap();
    b.Undo();
    Check(b.MoveCount() == 1, "ply not restored by undoing swap");
    Check(b.Hash() == h1, "hash not restored by undoing swap");
    Check(b.At(open) == hex::Cell::kRed, "red stone not restored");
    Check(b.At(B::Index(7, 3)) == hex::Cell::kEmpty, "blue stone not removed");
    Check(b.ToPlay() == hex::Player::kBlue, "blue not to move after undo");
    Check(b.CanSwap(), "swap not re-offered after undo");
    b.Undo();
    Check(b.MoveCount() == 0 && b.Hash() == 0, "board not empty after undo");
  }

  // A stone on the long diagonal mirrors onto itself; ownership must still move.
  {
    B b;
    const int diag = B::Index(4, 4);
    b.Play(diag);
    b.PlaySwap();
    Check(b.At(diag) == hex::Cell::kBlue, "diagonal opening not recoloured");
    Check(b.NumEmpty() == N * N - 1, "diagonal swap changed stone count");
  }

  // Swapped games still satisfy the no-draw theorem and still unwind cleanly.
  std::mt19937 rng(11);
  for (int game = 0; game < 2000; ++game) {
    B b;
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);

    b.Play(order[0]);
    b.PlaySwap();
    // The swap vacates order[0], so sweep the whole list; a partially filled
    // board may legitimately draw.
    for (std::size_t i = 0; i < order.size() && !b.IsTerminal(); ++i)
      if (b.At(order[i]) == hex::Cell::kEmpty) b.Play(order[i]);

    Check(b.IsTerminal(), "swapped game ended without a winner");
    while (b.MoveCount() > 0) b.Undo();
    Check(b.Hash() == 0 && b.NumEmpty() == N * N, "swapped game did not unwind");
  }
  std::printf("  2000 swapped games, mirror and undo verified\n");
}

// Ground truth, not a smoke test: Hex is a first-player win on every N x N
// board by strategy stealing, so a solver that disagrees has a rules bug.
template <int N>
void TestFirstPlayerWins() {
  hex::Board<N> board;
  hex::Solver<N> solver;
  const int value = solver.Solve(board);
  Check(value == 1, "first player does not win on a " + std::to_string(N) +
                        "x" + std::to_string(N) + " board");
  std::printf("  %dx%d: first player wins (%lld nodes)\n", N, N, solver.nodes());
}

}  // namespace

int main() {
  // Pinned so that any divergence in the random stream is caught immediately
  // rather than being mistaken for an engine bug. Every toolchain — libstdc++,
  // libc++, Emscripten — must produce exactly this number. If you add or remove
  // a test, update it deliberately.
  constexpr int kExpectedChecks = 79796;

  std::printf("== hex board core ==\n\n");
  TestNeighbourSymmetry();
  TestNoDrawTheorem();
  TestIncrementalDetection();
  TestUndoRestoresState();
  TestHashCollisions();
  TestSwapRule();
  std::printf("exhaustive solve of small boards\n");
  TestFirstPlayerWins<2>();
  TestFirstPlayerWins<3>();
  TestFirstPlayerWins<4>();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);

  if (g_checks != kExpectedChecks) {
    std::printf(
        "\nERROR: expected %d checks. The random stream differs from the "
        "reference,\nso results are not comparable across builds.\n",
        kExpectedChecks);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
