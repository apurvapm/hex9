#include <algorithm>
#include <chrono>
#include <cstdio>
#include <numeric>
#include <random>
#include <vector>

#include "hex/board.hpp"

namespace {

constexpr int N = 9;

// A uniform-random playout from the current position to a terminal one.
// This is the inner loop of plain MCTS and the unit we benchmark against.
hex::Cell RandomPlayout(hex::Board<N>& board, std::mt19937& rng, int& plies) {
  const int start = board.MoveCount();
  while (!board.IsTerminal()) {
    const int num = board.NumEmpty();
    const int pick = static_cast<int>(rng() % static_cast<unsigned>(num));
    board.Play(board.LegalMoves()[pick]);
  }
  plies = board.MoveCount() - start;
  const hex::Cell winner = board.Winner();
  while (board.MoveCount() > start) board.Undo();
  return winner;
}

}  // namespace

int main() {
  std::mt19937 rng(12345);
  hex::Board<N> board;

  constexpr int kWarmup = 20000;
  int plies = 0;
  for (int i = 0; i < kWarmup; ++i) RandomPlayout(board, rng, plies);

  constexpr int kPlayouts = 400000;
  long long total_plies = 0;
  int red = 0;

  const auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < kPlayouts; ++i) {
    if (RandomPlayout(board, rng, plies) == hex::Cell::kRed) ++red;
    total_plies += plies;
  }
  const auto end = std::chrono::steady_clock::now();

  const double seconds =
      std::chrono::duration<double>(end - start).count();
  std::printf("playouts        : %d in %.3f s\n", kPlayouts, seconds);
  std::printf("playouts/sec    : %.0f\n", kPlayouts / seconds);
  std::printf("moves/sec       : %.2f M\n",
              total_plies / seconds / 1e6);
  std::printf("mean game length: %.1f plies\n",
              static_cast<double>(total_plies) / kPlayouts);
  std::printf("red win rate    : %.1f%% (first player, random play)\n",
              100.0 * red / kPlayouts);
  return 0;
}
