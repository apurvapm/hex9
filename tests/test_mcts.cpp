#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "hex/board.hpp"
#include "hex/mcts.hpp"
#include "hex/solver.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}

// Search must not disturb the position it was handed.
void TestSearchRestoresBoard() {
  std::printf("search leaves the board untouched\n");
  constexpr int N = 9;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(4, 4));
  board.Play(hex::Board<N>::Index(2, 6));
  board.Play(hex::Board<N>::Index(5, 3));

  const std::uint64_t hash = board.Hash();
  const int ply = board.MoveCount();
  const int empty = board.NumEmpty();

  hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{2000, 1.414, 7, true});
  const int move = mcts.Search(board);

  Check(board.Hash() == hash, "hash changed across search");
  Check(board.MoveCount() == ply, "ply changed across search");
  Check(board.NumEmpty() == empty, "empty count changed across search");
  Check(move >= 0 && move <= N * N, "search returned an invalid move");
  Check(board.At(move) == hex::Cell::kEmpty, "search chose an occupied cell");
}

// Identical seeds must produce identical searches, on every toolchain.
void TestDeterminism() {
  std::printf("search is deterministic under a fixed seed\n");
  constexpr int N = 9;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(4, 4));
  board.Play(hex::Board<N>::Index(3, 3));

  typename hex::Mcts<N>::Config config{1500, 1.414, 99, true};
  hex::Mcts<N> a(config);
  hex::Mcts<N> b(config);
  Check(a.Search(board) == b.Search(board), "same seed gave different moves");

  config.seed = 100;
  hex::Mcts<N> c(config);
  hex::Mcts<N> d(config);
  Check(c.Search(board) == d.Search(board), "same seed gave different moves");
}

// With one move left to complete a chain, search must find it.
void TestFindsImmediateWin() {
  std::printf("finds a win in one\n");
  constexpr int N = 5;
  hex::Board<N> board;

  // Red holds a near-complete column; the gap at (2, 0) finishes it.
  board.PlaceStone(hex::Board<N>::Index(0, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(4, 4), hex::Player::kBlue);
  board.PlaceStone(hex::Board<N>::Index(1, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(4, 3), hex::Player::kBlue);
  board.PlaceStone(hex::Board<N>::Index(3, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(4, 2), hex::Player::kBlue);
  board.PlaceStone(hex::Board<N>::Index(4, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(3, 3), hex::Player::kBlue);

  Check(board.ToPlay() == hex::Player::kRed, "red should be to move");
  Check(!board.IsTerminal(), "position should not already be won");

  hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{4000, 1.414, 3, false});
  const int move = mcts.Search(board);
  Check(move == hex::Board<N>::Index(2, 0), "missed the winning move");
}

// The strongest available check: on boards small enough to solve exactly,
// search must choose a move that perfect play agrees is winning.
template <int N>
void TestAgreesWithSolver(int simulations) {
  hex::Board<N> board;
  hex::Solver<N> solver;
  const std::vector<int> winning = solver.WinningMoves(board);
  Check(!winning.empty(), "solver found no winning opening");

  int agreements = 0;
  constexpr int kSeeds = 8;
  for (int seed = 0; seed < kSeeds; ++seed) {
    hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{
        simulations, 1.414, static_cast<std::uint64_t>(seed + 1), false});
    const int move = mcts.Search(board);
    if (std::find(winning.begin(), winning.end(), move) != winning.end())
      ++agreements;
  }
  Check(agreements == kSeeds, std::to_string(N) + "x" + std::to_string(N) +
                                  ": search chose a losing opening");
  std::printf("  %dx%d: %d/%d seeds chose a provably winning move "
              "(%zu of %d openings win)\n",
              N, N, agreements, kSeeds, winning.size(), N * N);
}

// Swap must appear as a legal action exactly once, at ply 1, and only when
// enabled.
void TestSwapIsSearchable() {
  std::printf("swap appears in the search tree\n");
  constexpr int N = 7;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(3, 3));  // a strong centre opening

  {
    hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{3000, 1.414, 5, true});
    mcts.Search(board);
    const auto stats = mcts.RootStats();
    const bool has_swap =
        std::any_of(stats.begin(), stats.end(), [](const auto& s) {
          return s.move == hex::Board<N>::kSwapMove;
        });
    Check(has_swap, "swap missing from the root when enabled");
    Check(static_cast<int>(stats.size()) == board.NumEmpty() + 1,
          "root child count wrong with swap enabled");
  }

  {
    hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{3000, 1.414, 5, false});
    mcts.Search(board);
    const auto stats = mcts.RootStats();
    const bool has_swap =
        std::any_of(stats.begin(), stats.end(), [](const auto& s) {
          return s.move == hex::Board<N>::kSwapMove;
        });
    Check(!has_swap, "swap present at the root when disabled");
    Check(static_cast<int>(stats.size()) == board.NumEmpty(),
          "root child count wrong with swap disabled");
  }
}

// More simulations should not make the agent worse. Play stronger against
// weaker over paired games with colours alternated, since Hex's first-player
// advantage would otherwise swamp the signal.
void TestMoreSimulationsIsStronger() {
  std::printf("more simulations beats fewer\n");
  constexpr int N = 7;
  constexpr int kPairs = 30;
  int strong_wins = 0;

  for (int pair = 0; pair < kPairs; ++pair) {
    for (int strong_is_red = 0; strong_is_red < 2; ++strong_is_red) {
      hex::Board<N> board;
      hex::Mcts<N> strong(typename hex::Mcts<N>::Config{
          800, 1.414, static_cast<std::uint64_t>(pair * 2 + 1), false});
      hex::Mcts<N> weak(typename hex::Mcts<N>::Config{
          50, 1.414, static_cast<std::uint64_t>(pair * 2 + 2), false});

      while (!board.IsTerminal()) {
        const bool red_to_play = board.ToPlay() == hex::Player::kRed;
        const bool strong_turn = (red_to_play == (strong_is_red == 1));
        board.Play(strong_turn ? strong.Search(board) : weak.Search(board));
      }
      const bool red_won = board.Winner() == hex::Cell::kRed;
      if (red_won == (strong_is_red == 1)) ++strong_wins;
    }
  }

  const int games = kPairs * 2;
  Check(strong_wins >= games * 3 / 4,
        "800-simulation search failed to dominate 50-simulation search");
  std::printf("  800 sims beat 50 sims in %d/%d paired games (%.0f%%)\n",
              strong_wins, games, 100.0 * strong_wins / games);
}

}  // namespace

int main() {
  constexpr int kExpectedChecks = 19;

  std::printf("== hex mcts ==\n\n");
  TestSearchRestoresBoard();
  TestDeterminism();
  TestFindsImmediateWin();
  std::printf("agrees with exhaustive solver\n");
  TestAgreesWithSolver<3>(2000);
  TestAgreesWithSolver<4>(20000);
  TestSwapIsSearchable();
  TestMoreSimulationsIsStronger();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_checks != kExpectedChecks) {
    std::printf("\nERROR: expected %d checks.\n", kExpectedChecks);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
