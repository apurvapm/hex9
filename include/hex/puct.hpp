#ifndef HEX_PUCT_HPP
#define HEX_PUCT_HPP

#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <random>
#include <utility>
#include <vector>

#include "hex/board.hpp"

namespace hex {

// What an evaluator returns for a position: a prior over every action
// (including the swap slot at index N*N) and a value in [-1, 1] from the
// perspective of the player to move.
template <int N>
struct Evaluation {
  std::array<float, N * N + 1> priors{};
  float value = 0.0f;
};

namespace detail {

// The <random> distribution classes are implementation-defined and differ
// between libstdc++ and libc++, so every sampler here is hand-rolled. Self-play
// must reproduce bit for bit across the native and WebAssembly builds.
inline double NextUnit(std::mt19937& rng) {
  // Half-open (0, 1): never exactly zero, so logs are always finite.
  return (static_cast<double>(rng()) + 0.5) / 4294967296.0;
}

inline double NextNormal(std::mt19937& rng) {
  const double u1 = NextUnit(rng);
  const double u2 = NextUnit(rng);
  return std::sqrt(-2.0 * std::log(u1)) * std::cos(6.283185307179586 * u2);
}

// Marsaglia and Tsang's method. For shape < 1 it samples at shape + 1 and
// scales down, which is the standard boost.
inline double NextGamma(std::mt19937& rng, double shape) {
  if (shape < 1.0) {
    const double u = NextUnit(rng);
    return NextGamma(rng, shape + 1.0) * std::pow(u, 1.0 / shape);
  }
  const double d = shape - 1.0 / 3.0;
  const double c = 1.0 / std::sqrt(9.0 * d);
  for (;;) {
    const double x = NextNormal(rng);
    const double t = 1.0 + c * x;
    if (t <= 0.0) continue;
    const double v = t * t * t;
    const double u = NextUnit(rng);
    if (std::log(u) < 0.5 * x * x + d - d * v + d * std::log(v)) return d * v;
  }
}

// Symmetric Dirichlet: independent gammas, normalised.
inline void SampleDirichlet(std::mt19937& rng, double alpha,
                            std::vector<double>& out) {
  double total = 0.0;
  for (double& value : out) {
    value = NextGamma(rng, alpha);
    total += value;
  }
  if (total <= 0.0) {
    std::fill(out.begin(), out.end(), 1.0 / static_cast<double>(out.size()));
    return;
  }
  for (double& value : out) value /= total;
}

}  // namespace detail

// AlphaZero-style search. The network supplies priors and a leaf value, so
// there is no rollout: a simulation descends to an unexpanded node, evaluates
// it once, and backs the value up.
template <int N>
class Puct {
 public:
  struct Config {
    int simulations = 800;
    float c_puct = 1.5f;
    // Root noise widens the opening during self-play. Set the weight to zero
    // for evaluation and for play against a human.
    float dirichlet_alpha = 0.3f;
    float dirichlet_weight = 0.25f;
    std::uint64_t seed = 1;
    bool allow_swap = true;
  };

  struct Result {
    int best_move = -1;
    // Visit counts per action. Normalised, this is the policy training target.
    std::vector<std::pair<int, int>> visits;
    float root_value = 0.0f;
  };

  explicit Puct(Config config = {}) : config_(config), rng_(config.seed) {}

  // Evaluator is any callable with signature
  //   void(const Board<N>&, Evaluation<N>&)
  // The board is restored exactly.
  template <typename Evaluator>
  Result Search(Board<N>& board, Evaluator&& evaluate) {
    const std::uint64_t entry_hash = board.Hash();

    nodes_.clear();
    nodes_.reserve(static_cast<std::size_t>(config_.simulations) * 2 + 64);
    nodes_.push_back(Node{-1, -1, -1, 0, 0, 0.0f, 0.0f});

    // Expand and evaluate the root before any simulation, so that noise can be
    // applied to real priors.
    Evaluation<N> root_eval;
    evaluate(board, root_eval);
    Expand(0, board, root_eval);
    if (config_.dirichlet_weight > 0.0f) ApplyRootNoise();

    for (int i = 0; i < config_.simulations; ++i)
      Simulate(board, evaluate);

    assert(board.Hash() == entry_hash);
    (void)entry_hash;

    Result result;
    const Node& root = nodes_[0];
    int best_visits = -1;
    for (int c = root.first_child; c < root.first_child + root.num_children;
         ++c) {
      result.visits.emplace_back(nodes_[c].move, nodes_[c].visits);
      if (nodes_[c].visits > best_visits) {
        best_visits = nodes_[c].visits;
        result.best_move = nodes_[c].move;
      }
    }
    result.root_value = root.visits > 0 ? root.value_sum / root.visits : 0.0f;
    return result;
  }

  // Self-play move selection. Temperature zero is greedy; higher temperatures
  // flatten the distribution and widen the opening book.
  int SampleMove(const Result& result, float temperature) {
    if (result.visits.empty()) return -1;
    if (temperature <= 0.0f) return result.best_move;

    std::vector<double> weights;
    weights.reserve(result.visits.size());
    double total = 0.0;
    for (const auto& [move, visits] : result.visits) {
      const double w =
          std::pow(static_cast<double>(visits), 1.0 / temperature);
      weights.push_back(w);
      total += w;
    }
    if (total <= 0.0) return result.best_move;

    double target = detail::NextUnit(rng_) * total;
    for (std::size_t i = 0; i < weights.size(); ++i) {
      target -= weights[i];
      if (target <= 0.0) return result.visits[i].first;
    }
    return result.visits.back().first;
  }

  std::size_t TreeSize() const { return nodes_.size(); }

 private:
  struct Node {
    int move;
    int parent;
    int first_child;
    int num_children;
    int visits;
    float value_sum;  // from the perspective of the player to move here
    float prior;
  };

  void ApplyMove(Board<N>& board, int move) const {
    if (move == Board<N>::kSwapMove) {
      board.PlaySwap();
    } else {
      board.Play(move);
    }
  }

  void Expand(int node, const Board<N>& board, const Evaluation<N>& eval) {
    const int first = static_cast<int>(nodes_.size());
    const int num_empty = board.NumEmpty();

    float total = 0.0f;
    for (int i = 0; i < num_empty; ++i)
      total += eval.priors[board.LegalMoves()[i]];
    if (config_.allow_swap && board.CanSwap())
      total += eval.priors[Board<N>::kSwapMove];

    // Renormalise over legal actions only. A network trained on masked targets
    // will already put near-zero mass on illegal cells, but never exactly zero.
    const int count = num_empty + (config_.allow_swap && board.CanSwap() ? 1 : 0);
    const float uniform = 1.0f / static_cast<float>(count);

    for (int i = 0; i < num_empty; ++i) {
      const int move = board.LegalMoves()[i];
      const float prior =
          total > 0.0f ? eval.priors[move] / total : uniform;
      nodes_.push_back(Node{move, node, -1, 0, 0, 0.0f, prior});
    }
    if (config_.allow_swap && board.CanSwap()) {
      const float prior =
          total > 0.0f ? eval.priors[Board<N>::kSwapMove] / total : uniform;
      nodes_.push_back(Node{Board<N>::kSwapMove, node, -1, 0, 0, 0.0f, prior});
    }

    nodes_[node].first_child = first;
    nodes_[node].num_children = static_cast<int>(nodes_.size()) - first;
    assert(nodes_[node].num_children > 0);
  }

  void ApplyRootNoise() {
    Node& root = nodes_[0];
    std::vector<double> noise(static_cast<std::size_t>(root.num_children));
    detail::SampleDirichlet(rng_, config_.dirichlet_alpha, noise);

    const float w = config_.dirichlet_weight;
    for (int i = 0; i < root.num_children; ++i) {
      Node& child = nodes_[root.first_child + i];
      child.prior = (1.0f - w) * child.prior +
                    w * static_cast<float>(noise[static_cast<std::size_t>(i)]);
    }
  }

  int SelectChild(int node) const {
    const Node& parent = nodes_[node];
    const double sqrt_total =
        std::sqrt(static_cast<double>(parent.visits > 0 ? parent.visits : 1));

    int best = -1;
    double best_score = -1e30;
    for (int c = parent.first_child; c < parent.first_child + parent.num_children;
         ++c) {
      const Node& child = nodes_[c];
      // Unvisited children score Q = 0 rather than infinity: the prior already
      // decides which of them to try first, so there is no need to sweep every
      // action before deepening. This is what lets PUCT search a 82-wide tree
      // usefully at 800 simulations.
      const double q =
          child.visits > 0
              ? -static_cast<double>(child.value_sum) / child.visits
              : 0.0;
      const double u = config_.c_puct * child.prior * sqrt_total /
                       (1.0 + child.visits);
      const double score = q + u;
      if (score > best_score) {
        best_score = score;
        best = c;
      }
    }
    return best;
  }

  template <typename Evaluator>
  void Simulate(Board<N>& board, Evaluator&& evaluate) {
    int node = 0;
    int applied = 0;

    while (nodes_[node].first_child >= 0) {
      node = SelectChild(node);
      ApplyMove(board, nodes_[node].move);
      ++applied;
    }

    float value;
    if (board.IsTerminal()) {
      // The opponent completed a chain on the previous move, so the player to
      // move at this node has already lost.
      value = -1.0f;
    } else {
      Evaluation<N> eval;
      evaluate(board, eval);
      Expand(node, board, eval);
      value = eval.value;
    }

    for (int cur = node; cur >= 0; cur = nodes_[cur].parent) {
      nodes_[cur].visits += 1;
      nodes_[cur].value_sum += value;
      value = -value;
    }

    while (applied-- > 0) board.Undo();
  }

  Config config_;
  std::mt19937 rng_;
  std::vector<Node> nodes_;
};

// A network-free evaluator: uniform priors and a value from one random
// playout. Useful as a baseline and for testing the search before any model
// exists.
template <int N>
class RolloutEvaluator {
 public:
  explicit RolloutEvaluator(std::uint64_t seed = 1) : rng_(seed) {}

  void operator()(const Board<N>& board, Evaluation<N>& out) {
    out.priors.fill(1.0f / static_cast<float>(N * N + 1));

    Board<N>& mutable_board = const_cast<Board<N>&>(board);
    const Player me = mutable_board.ToPlay();
    const int start = mutable_board.MoveCount();

    while (!mutable_board.IsTerminal()) {
      const int num_empty = mutable_board.NumEmpty();
      const int pick =
          static_cast<int>(rng_() % static_cast<unsigned>(num_empty));
      mutable_board.Play(mutable_board.LegalMoves()[pick]);
    }
    const bool won = mutable_board.Winner() == ToCell(me);
    while (mutable_board.MoveCount() > start) mutable_board.Undo();

    out.value = won ? 1.0f : -1.0f;
  }

 private:
  std::mt19937 rng_;
};

}  // namespace hex

#endif  // HEX_PUCT_HPP
