#ifndef HEX_MCTS_HPP
#define HEX_MCTS_HPP

#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

#include "hex/board.hpp"

namespace hex {

// Plain Monte Carlo tree search: UCT selection, uniform-random rollouts, no
// network. This is the phase 1 baseline. Phase 3 replaces the rollout with a
// value head and UCT with PUCT, but the tree machinery stays.
//
// Value convention: node.value_sum accumulates outcomes from the perspective of
// the player *to move at that node*. A child's value is therefore negated when
// viewed from its parent, which is where the sign flips in selection and backup.
template <int N>
class Mcts {
 public:
  struct Config {
    int simulations = 1000;
    // sqrt(2) is the textbook UCT constant for rewards in [-1, 1].
    double c_uct = 1.4142135623730951;
    std::uint64_t seed = 1;
    // The browser demo defaults to no swap because it confuses players who do
    // not know Hex. Training and arena play enable it.
    bool allow_swap = true;
  };

  struct MoveStat {
    int move;
    int visits;
    double value;  // from the perspective of the player to move at the root
  };

  explicit Mcts(Config config = {}) : config_(config), rng_(config.seed) {}

  // Returns the most-visited root move. The board is restored exactly.
  int Search(Board<N>& board) {
    const int entry_ply = board.MoveCount();
    const std::uint64_t entry_hash = board.Hash();

    nodes_.clear();
    nodes_.reserve(static_cast<std::size_t>(config_.simulations) * 2 + 64);
    nodes_.push_back(Node{-1, -1, -1, 0, 0, 0.0});

    for (int i = 0; i < config_.simulations; ++i) Simulate(board);

    assert(board.MoveCount() == entry_ply);
    assert(board.Hash() == entry_hash);
    (void)entry_ply;
    (void)entry_hash;

    return BestMove();
  }

  // Root statistics, for the demo's move table and for diagnostics.
  std::vector<MoveStat> RootStats() const {
    std::vector<MoveStat> stats;
    const Node& root = nodes_[0];
    for (int c = root.first_child; c < root.first_child + root.num_children;
         ++c) {
      const Node& child = nodes_[c];
      const double value =
          child.visits > 0 ? -child.value_sum / child.visits : 0.0;
      stats.push_back(MoveStat{child.move, child.visits, value});
    }
    return stats;
  }

  std::size_t TreeSize() const { return nodes_.size(); }

 private:
  struct Node {
    int move;         // move played to reach this node, -1 at the root
    int parent;       // arena index, -1 at the root
    int first_child;  // arena index, -1 while unexpanded
    int num_children;
    int visits;
    double value_sum;  // from the perspective of the player to move here
  };

  void ApplyMove(Board<N>& board, int move) const {
    if (move == Board<N>::kSwapMove) {
      board.PlaySwap();
    } else {
      board.Play(move);
    }
  }

  void Simulate(Board<N>& board) {
    int node = 0;
    int applied = 0;

    // Selection: descend while the node has children.
    while (nodes_[node].first_child >= 0) {
      node = SelectChild(node);
      ApplyMove(board, nodes_[node].move);
      ++applied;
    }

    double value;
    if (board.IsTerminal()) {
      // The opponent completed a chain on the previous move, so the player to
      // move at this node has already lost.
      value = -1.0;
    } else {
      Expand(node, board);
      const Node& parent = nodes_[node];
      const int child =
          parent.first_child +
          static_cast<int>(rng_() % static_cast<unsigned>(parent.num_children));
      node = child;
      ApplyMove(board, nodes_[node].move);
      ++applied;
      value = board.IsTerminal() ? -1.0 : Rollout(board);
    }

    // Backup: the sign flips at every level because the player to move
    // alternates.
    for (int cur = node; cur >= 0; cur = nodes_[cur].parent) {
      nodes_[cur].visits += 1;
      nodes_[cur].value_sum += value;
      value = -value;
    }

    while (applied-- > 0) board.Undo();
  }

  int SelectChild(int node) const {
    const Node& parent = nodes_[node];
    const double log_parent =
        std::log(static_cast<double>(parent.visits > 0 ? parent.visits : 1));

    int best = -1;
    double best_score = -1e30;
    for (int c = parent.first_child; c < parent.first_child + parent.num_children;
         ++c) {
      const Node& child = nodes_[c];
      double score;
      if (child.visits == 0) {
        // Unvisited children are explored first, in arena order.
        score = 1e29;
      } else {
        const double q = -child.value_sum / child.visits;
        const double u =
            config_.c_uct * std::sqrt(log_parent / child.visits);
        score = q + u;
      }
      if (score > best_score) {
        best_score = score;
        best = c;
      }
    }
    return best;
  }

  void Expand(int node, const Board<N>& board) {
    const int first = static_cast<int>(nodes_.size());
    const int num_empty = board.NumEmpty();
    for (int i = 0; i < num_empty; ++i)
      nodes_.push_back(Node{board.LegalMoves()[i], node, -1, 0, 0, 0.0});
    if (config_.allow_swap && board.CanSwap())
      nodes_.push_back(Node{Board<N>::kSwapMove, node, -1, 0, 0, 0.0});

    nodes_[node].first_child = first;
    nodes_[node].num_children = static_cast<int>(nodes_.size()) - first;
    assert(nodes_[node].num_children > 0);
  }

  // Uniform-random playout to a terminal position. Returns the outcome from the
  // perspective of the player to move when the rollout began.
  double Rollout(Board<N>& board) {
    const Player me = board.ToPlay();
    const int start = board.MoveCount();

    while (!board.IsTerminal()) {
      const int num_empty = board.NumEmpty();
      const int options =
          num_empty + (config_.allow_swap && board.CanSwap() ? 1 : 0);
      const int pick = static_cast<int>(rng_() % static_cast<unsigned>(options));
      if (pick == num_empty) {
        board.PlaySwap();
      } else {
        board.Play(board.LegalMoves()[pick]);
      }
    }

    const bool won = board.Winner() == ToCell(me);
    while (board.MoveCount() > start) board.Undo();
    return won ? 1.0 : -1.0;
  }

  // Most-visited rather than highest-value: visit counts are the statistic MCTS
  // actually concentrates, and a high-value child with two visits is noise.
  int BestMove() const {
    const Node& root = nodes_[0];
    int best = -1;
    int best_visits = -1;
    for (int c = root.first_child; c < root.first_child + root.num_children;
         ++c) {
      if (nodes_[c].visits > best_visits) {
        best_visits = nodes_[c].visits;
        best = nodes_[c].move;
      }
    }
    return best;
  }

  Config config_;
  std::mt19937 rng_;
  std::vector<Node> nodes_;
};

}  // namespace hex

#endif  // HEX_MCTS_HPP
