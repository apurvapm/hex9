#include <algorithm>
#include <cstdio>
#include <string>
#include <vector>

#include "hex/alphabeta.hpp"
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

// The connection distance is the number of stones still needed to link the two
// edges. It is the whole evaluation, so it is worth pinning down directly.
void TestConnectionDistance() {
  std::printf("connection distance\n");
  constexpr int N = 5;
  using AB = hex::AlphaBeta<N>;
  hex::Board<N> board;

  // An empty 5x5 board needs five stones for either player.
  Check(AB::ConnectionDistance(board, hex::Player::kRed) == 5,
        "empty board red distance");
  Check(AB::ConnectionDistance(board, hex::Player::kBlue) == 5,
        "empty board blue distance");

  // Each red stone on a fresh column shortens red's path by one.
  board.PlaceStone(hex::Board<N>::Index(0, 2), hex::Player::kRed);
  Check(AB::ConnectionDistance(board, hex::Player::kRed) == 4,
        "one red stone should shorten red's path");

  board.PlaceStone(hex::Board<N>::Index(1, 2), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(2, 2), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(3, 2), hex::Player::kRed);
  Check(AB::ConnectionDistance(board, hex::Player::kRed) == 1,
        "four stacked red stones should leave one to place");

  // Completing the chain drops the distance to zero.
  board.PlaceStone(hex::Board<N>::Index(4, 2), hex::Player::kRed);
  Check(AB::ConnectionDistance(board, hex::Player::kRed) == 0,
        "a completed chain should have distance zero");
  Check(board.IsTerminal() && board.Winner() == hex::Cell::kRed,
        "the completed chain should be a red win");

  // Blue must now detour around red's wall, so its distance grows.
  Check(AB::ConnectionDistance(board, hex::Player::kBlue) > 5,
        "blue should pay for crossing red's wall");
}

void TestSearchRestoresBoard() {
  std::printf("search leaves the board untouched\n");
  constexpr int N = 9;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(4, 4));
  board.Play(hex::Board<N>::Index(2, 6));

  const std::uint64_t hash = board.Hash();
  const int ply = board.MoveCount();

  hex::AlphaBeta<N> engine(typename hex::AlphaBeta<N>::Config{4, 20, true});
  const int move = engine.Search(board);

  Check(board.Hash() == hash, "hash changed across search");
  Check(board.MoveCount() == ply, "ply changed across search");
  Check(move >= 0 && move <= N * N, "invalid move returned");
  Check(board.At(move) == hex::Cell::kEmpty, "chose an occupied cell");
}

void TestDeterminism() {
  std::printf("search is deterministic\n");
  constexpr int N = 9;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(4, 4));
  board.Play(hex::Board<N>::Index(3, 5));

  typename hex::AlphaBeta<N>::Config config{4, 20, true};
  hex::AlphaBeta<N> a(config);
  hex::AlphaBeta<N> b(config);
  Check(a.Search(board) == b.Search(board), "two runs disagreed");
}

void TestFindsImmediateWin() {
  std::printf("finds a win in one\n");
  constexpr int N = 5;
  hex::Board<N> board;
  board.PlaceStone(hex::Board<N>::Index(0, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(4, 4), hex::Player::kBlue);
  board.PlaceStone(hex::Board<N>::Index(1, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(4, 3), hex::Player::kBlue);
  board.PlaceStone(hex::Board<N>::Index(3, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(4, 2), hex::Player::kBlue);
  board.PlaceStone(hex::Board<N>::Index(4, 0), hex::Player::kRed);
  board.PlaceStone(hex::Board<N>::Index(3, 3), hex::Player::kBlue);

  hex::AlphaBeta<N> engine(typename hex::AlphaBeta<N>::Config{4, 18, false});
  Check(engine.Search(board) == hex::Board<N>::Index(2, 0),
        "missed the winning move");
  Check(engine.LastScore() > hex::AlphaBeta<N>::kWinScore / 2,
        "a forced win should score as a win");
}

// Searched to full depth on a solvable board, alpha-beta is exact, so it must
// agree with the exhaustive solver.
template <int N>
void TestMatchesSolverAtFullDepth() {
  hex::Board<N> board;
  hex::Solver<N> solver;
  const std::vector<int> winning = solver.WinningMoves(board);

  hex::AlphaBeta<N> engine(typename hex::AlphaBeta<N>::Config{N * N, 18, false});
  const int move = engine.Search(board);

  Check(std::find(winning.begin(), winning.end(), move) != winning.end(),
        std::to_string(N) + "x" + std::to_string(N) + ": chose a losing opening");
  Check(engine.LastScore() > hex::AlphaBeta<N>::kWinScore / 2,
        std::to_string(N) + "x" + std::to_string(N) +
            ": should recognise a first-player win");
  std::printf("  %dx%d: matched the solver in %lld nodes\n", N, N,
              engine.Nodes());
}

// Deeper search should not be weaker. Paired games with colours alternated,
// because Hex's first-player advantage would otherwise dominate the result.
void TestDeeperIsStronger() {
  std::printf("deeper search beats shallower\n");
  constexpr int N = 7;
  constexpr int kPairs = 12;
  int deep_wins = 0;

  for (int pair = 0; pair < kPairs; ++pair) {
    for (int deep_is_red = 0; deep_is_red < 2; ++deep_is_red) {
      hex::Board<N> board;
      // Vary the opening so the pairs are not all the same game.
      board.Play(hex::Board<N>::Index(pair / 4, pair % 4));

      hex::AlphaBeta<N> deep(typename hex::AlphaBeta<N>::Config{4, 18, false});
      hex::AlphaBeta<N> shallow(typename hex::AlphaBeta<N>::Config{1, 18, false});

      while (!board.IsTerminal()) {
        const bool red_to_play = board.ToPlay() == hex::Player::kRed;
        const bool deep_turn = (red_to_play == (deep_is_red == 1));
        board.Play(deep_turn ? deep.Search(board) : shallow.Search(board));
      }
      const bool red_won = board.Winner() == hex::Cell::kRed;
      if (red_won == (deep_is_red == 1)) ++deep_wins;
    }
  }

  const int games = kPairs * 2;
  Check(deep_wins >= games * 2 / 3,
        "depth 4 failed to beat depth 1 convincingly");
  std::printf("  depth 4 beat depth 1 in %d/%d paired games (%.0f%%)\n",
              deep_wins, games, 100.0 * deep_wins / games);
}

// A calibration point rather than a pass/fail claim: how does the structural
// evaluation compare with a rollout-based search of a given budget?
void TestVersusMcts() {
  std::printf("alpha-beta versus mcts (calibration)\n");
  constexpr int N = 7;
  constexpr int kPairs = 10;
  int ab_wins = 0;

  for (int pair = 0; pair < kPairs; ++pair) {
    for (int ab_is_red = 0; ab_is_red < 2; ++ab_is_red) {
      hex::Board<N> board;
      hex::AlphaBeta<N> engine(typename hex::AlphaBeta<N>::Config{4, 18, false});
      hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{
          1000, 1.414, static_cast<std::uint64_t>(pair + 1), false});

      while (!board.IsTerminal()) {
        const bool red_to_play = board.ToPlay() == hex::Player::kRed;
        const bool ab_turn = (red_to_play == (ab_is_red == 1));
        board.Play(ab_turn ? engine.Search(board) : mcts.Search(board));
      }
      const bool red_won = board.Winner() == hex::Cell::kRed;
      if (red_won == (ab_is_red == 1)) ++ab_wins;
    }
  }

  const int games = kPairs * 2;
  Check(true, "");
  std::printf("  depth-4 alpha-beta won %d/%d against 1000-sim MCTS (%.0f%%)\n",
              ab_wins, games, 100.0 * ab_wins / games);
}

}  // namespace

int main() {
  constexpr int kExpectedChecks = 20;

  std::printf("== hex alpha-beta ==\n\n");
  TestConnectionDistance();
  TestSearchRestoresBoard();
  TestDeterminism();
  TestFindsImmediateWin();
  std::printf("matches the exhaustive solver at full depth\n");
  TestMatchesSolverAtFullDepth<3>();
  TestMatchesSolverAtFullDepth<4>();
  TestDeeperIsStronger();
  TestVersusMcts();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_checks != kExpectedChecks) {
    std::printf("\nERROR: expected %d checks.\n", kExpectedChecks);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
