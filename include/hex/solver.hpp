#ifndef HEX_SOLVER_HPP
#define HEX_SOLVER_HPP

#include <cstdint>
#include <unordered_map>
#include <vector>

#include "hex/board.hpp"

namespace hex {

// Exhaustive negamax with a transposition table. Tractable only on tiny boards
// (up to 4x4 in about ten million nodes), which is exactly the point: it is the
// ground truth that search and, later, learned policies are measured against.
//
// The solver ignores the swap rule deliberately. It answers a question about
// the pure game — Hex is a first-player win on every N x N board by strategy
// stealing — and swap is a balancing convention layered on top of that fact.
template <int N>
class Solver {
 public:
  // +1 if the player to move wins with perfect play, -1 otherwise.
  // There are no draws in Hex, so those are the only two outcomes.
  int Solve(Board<N>& board) {
    ++nodes_;
    const auto it = table_.find(board.Hash());
    if (it != table_.end()) return it->second;

    int best = -1;
    const int num = board.NumEmpty();
    const std::vector<std::uint8_t> moves(board.LegalMoves(),
                                          board.LegalMoves() + num);
    for (const std::uint8_t move : moves) {
      board.Play(move);
      const int value = board.IsTerminal() ? 1 : -Solve(board);
      board.Undo();
      if (value > best) best = value;
      if (best == 1) break;  // a win is the best attainable value
    }
    table_.emplace(board.Hash(), best);
    return best;
  }

  // Every move that wins for the player to move. Unlike Solve() this cannot
  // short-circuit on the first win, so it costs strictly more.
  std::vector<int> WinningMoves(Board<N>& board) {
    std::vector<int> winning;
    const int num = board.NumEmpty();
    const std::vector<std::uint8_t> moves(board.LegalMoves(),
                                          board.LegalMoves() + num);
    for (const std::uint8_t move : moves) {
      board.Play(move);
      const int value = board.IsTerminal() ? 1 : -Solve(board);
      board.Undo();
      if (value == 1) winning.push_back(move);
    }
    return winning;
  }

  long long nodes() const { return nodes_; }
  void ResetCounters() { nodes_ = 0; }

 private:
  std::unordered_map<std::uint64_t, int> table_;
  long long nodes_ = 0;
};

}  // namespace hex

#endif  // HEX_SOLVER_HPP
