#ifndef HEX_SEARCH_TREE_HPP
#define HEX_SEARCH_TREE_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <utility>
#include <vector>

#include "hex/board.hpp"
#include "hex/puct.hpp"

// The shared search tree, split out from parallel_puct.hpp so it can be used
// without pulling in <thread> and <barrier>.
//
// That split exists for the browser. The WebAssembly build is single-threaded --
// GitHub Pages cannot set COOP/COEP, so there is no SharedArrayBuffer -- but it
// still needs a tree it can descend, evaluate and back up in separate steps,
// because ONNX Runtime Web returns a Promise and a synchronous search cannot wait
// on one without Asyncify. The descend/evaluate/backup split that leaf-parallel
// search needed for threads turns out to be exactly what an async evaluator needs
// for resumability, so the browser reuses tested code instead of adding a third
// search implementation.

namespace hex {

struct ParallelSearchConfig {
  int threads = 1;
  int simulations = 800;  // total across all threads
  float c_puct = 1.5f;
  bool allow_swap = true;
  std::uint64_t seed = 1;
  // Discourages threads from piling onto one branch by pretending a descent has
  // already lost there. Applied going down, removed coming back up, so it steers
  // selection without polluting the statistics that outlive the search. Zero
  // reproduces plain PUCT, which is what makes the mode ablatable.
  int virtual_loss = 3;
  int leaf_batch = 8;
};

struct ParallelSearchResult {
  int best_move = -1;
  std::vector<std::pair<int, int>> visits;
  float root_value = 0.0f;
  // Distinct positions handed to the evaluator. Tree-parallel should cover more
  // distinct nodes than root-parallel at the same budget -- that is the mechanism
  // by which it is supposed to win, so it is measured rather than assumed.
  long long evaluations = 0;
  int distinct_nodes = 0;
};

namespace detail {

// Values live in [-1, 1]. Stored as fixed point because atomic<float> has no
// fetch_add, and a compare-exchange loop on the hottest counter in the search is
// worse than integer scaling. Six digits is far more precision than a value head
// carries, and an int64 cannot overflow at any simulation count this project runs.
constexpr double kValueScale = 1000000.0;

enum : int { kUnexpanded = 0, kExpanding = 1, kExpanded = 2 };

template <int N>
struct SharedNode {
  int move = -1;
  int parent = -1;
  float prior = 0.0f;
  int first_child = -1;
  int num_children = 0;
  // Guards the two fields above. They are plain ints written by the single thread
  // that wins the claim and read only after state reads kExpanded with acquire, so
  // the release store below publishes them.
  std::atomic<int> state{kUnexpanded};
  std::atomic<int> visits{0};
  std::atomic<std::int64_t> value_sum{0};
  std::atomic<int> virtual_loss{0};
};

// One shared tree. Nodes live in a arena sized once at construction and handed out
// by a bump allocator, because SharedNode holds atomics and is therefore neither
// copyable nor movable: a growing vector could not reallocate it, and a reallocation
// would dangle the references concurrent selectors are holding.
template <int N>
class SharedTree {
 public:
  explicit SharedTree(const ParallelSearchConfig& config)
      : config_(config),
        // Worst case is one expansion per simulation, each adding every legal
        // action. Generous rather than clever: 800 simulations on 9x9 is about
        // 3 MB, and running out silently would corrupt the search.
        nodes_(static_cast<std::size_t>(config.simulations) *
                   static_cast<std::size_t>(N * N + 1) +
               64) {
    used_.store(1, std::memory_order_relaxed);  // node 0 is the root
  }

  int Root() const { return 0; }
  int NodesUsed() const { return used_.load(std::memory_order_relaxed); }

  bool IsExpanded(int node) const {
    return nodes_[static_cast<std::size_t>(node)].state.load(
               std::memory_order_acquire) == kExpanded;
  }

  // Returns true if this thread performed the expansion. A loser does not wait: it
  // has already evaluated the position, so backing that value up and moving on
  // costs one duplicated evaluation and no blocking.
  bool Expand(int node, const Board<N>& board, const Evaluation<N>& eval) {
    SharedNode<N>& parent = nodes_[static_cast<std::size_t>(node)];
    int expected = kUnexpanded;
    if (!parent.state.compare_exchange_strong(expected, kExpanding,
                                              std::memory_order_acq_rel))
      return false;

    const int num_empty = board.NumEmpty();
    const bool swap_ok = config_.allow_swap && board.CanSwap();
    const int count = num_empty + (swap_ok ? 1 : 0);

    const int first = used_.fetch_add(count, std::memory_order_relaxed);
    if (first + count > static_cast<int>(nodes_.size())) {
      // Out of arena. Leave the node unexpanded rather than writing past the end;
      // the search degrades to evaluating this leaf repeatedly, which is wrong but
      // bounded, and the capacity above makes it unreachable in practice.
      parent.state.store(kUnexpanded, std::memory_order_release);
      return false;
    }

    float total = 0.0f;
    for (int i = 0; i < num_empty; ++i)
      total += eval.priors[board.LegalMoves()[i]];
    if (swap_ok) total += eval.priors[Board<N>::kSwapMove];
    const float uniform = 1.0f / static_cast<float>(count);

    for (int i = 0; i < num_empty; ++i) {
      const int move = board.LegalMoves()[i];
      SharedNode<N>& child = nodes_[static_cast<std::size_t>(first + i)];
      child.move = move;
      child.parent = node;
      child.prior = total > 0.0f ? eval.priors[move] / total : uniform;
    }
    if (swap_ok) {
      SharedNode<N>& child =
          nodes_[static_cast<std::size_t>(first + num_empty)];
      child.move = Board<N>::kSwapMove;
      child.parent = node;
      child.prior = total > 0.0f ? eval.priors[Board<N>::kSwapMove] / total
                                 : uniform;
    }

    parent.first_child = first;
    parent.num_children = count;
    // Release: publishes first_child, num_children and every child initialised
    // above to any thread that later reads kExpanded with acquire.
    parent.state.store(kExpanded, std::memory_order_release);
    return true;
  }

  int SelectChild(int node) const {
    const SharedNode<N>& parent = nodes_[static_cast<std::size_t>(node)];
    const int first = parent.first_child;
    const int count = parent.num_children;
    const int parent_visits = parent.visits.load(std::memory_order_relaxed);
    const double sqrt_total =
        std::sqrt(static_cast<double>(parent_visits > 0 ? parent_visits : 1));

    int best = -1;
    double best_score = -1e30;
    for (int c = first; c < first + count; ++c) {
      const SharedNode<N>& child = nodes_[static_cast<std::size_t>(c)];
      const int visits = child.visits.load(std::memory_order_relaxed);
      const int loss = child.virtual_loss.load(std::memory_order_relaxed);
      const std::int64_t sum = child.value_sum.load(std::memory_order_relaxed);

      // Virtual loss enters as visits that already lost: it raises the
      // denominator and drags Q down, without ever touching value_sum. Removing
      // it on backup is therefore exact rather than approximate.
      const int effective = visits + loss;
      const double q =
          effective > 0
              ? -(static_cast<double>(sum) / kValueScale +
                  static_cast<double>(loss)) /
                    effective
              : 0.0;
      const double u =
          config_.c_puct * child.prior * sqrt_total / (1.0 + effective);
      const double score = q + u;
      if (score > best_score) {
        best_score = score;
        best = c;
      }
    }
    return best;
  }

  // Descends to an unexpanded node, applying virtual loss as it goes. Leaves the
  // board at the leaf; Backup restores it.
  int Descend(Board<N>& board, int& applied) {
    int node = Root();
    applied = 0;
    for (;;) {
      if (config_.virtual_loss > 0)
        nodes_[static_cast<std::size_t>(node)].virtual_loss.fetch_add(
            config_.virtual_loss, std::memory_order_relaxed);
      if (!IsExpanded(node) || board.IsTerminal()) return node;
      const int child = SelectChild(node);
      if (child < 0) return node;
      node = child;
      const int move = nodes_[static_cast<std::size_t>(node)].move;
      if (move == Board<N>::kSwapMove) {
        board.PlaySwap();
      } else {
        board.Play(move);
      }
      ++applied;
    }
  }

  void Backup(Board<N>& board, int node, float value, int applied) {
    for (int cur = node; cur >= 0;
         cur = nodes_[static_cast<std::size_t>(cur)].parent) {
      SharedNode<N>& n = nodes_[static_cast<std::size_t>(cur)];
      n.visits.fetch_add(1, std::memory_order_relaxed);
      n.value_sum.fetch_add(
          static_cast<std::int64_t>(static_cast<double>(value) * kValueScale),
          std::memory_order_relaxed);
      if (config_.virtual_loss > 0)
        n.virtual_loss.fetch_sub(config_.virtual_loss,
                                 std::memory_order_relaxed);
      value = -value;
    }
    while (applied-- > 0) board.Undo();
  }


  // A node as the UI needs to see it. The tree visualiser and the top-k move table
  // both read this rather than reaching into SharedNode, so the atomics and the
  // fixed-point value encoding stay an implementation detail.
  struct NodeView {
    int index = -1;
    int parent = -1;
    int move = -1;
    int visits = 0;
    float value = 0.0f;  // from the perspective of the player to move here
    float prior = 0.0f;
    int depth = 0;
  };

  int RootVisits() const {
    return nodes_[0].visits.load(std::memory_order_relaxed);
  }

  NodeView View(int index, int depth) const {
    const SharedNode<N>& n = nodes_[static_cast<std::size_t>(index)];
    const int visits = n.visits.load(std::memory_order_relaxed);
    NodeView view;
    view.index = index;
    view.parent = n.parent;
    view.move = n.move;
    view.visits = visits;
    view.value =
        visits > 0
            ? static_cast<float>(
                  static_cast<double>(
                      n.value_sum.load(std::memory_order_relaxed)) /
                  kValueScale / visits)
            : 0.0f;
    view.prior = n.prior;
    view.depth = depth;
    return view;
  }

  std::vector<NodeView> RootChildren() const {
    std::vector<NodeView> out;
    const SharedNode<N>& root = nodes_[0];
    if (root.state.load(std::memory_order_acquire) != kExpanded) return out;
    for (int c = root.first_child; c < root.first_child + root.num_children; ++c)
      out.push_back(View(c, 1));
    return out;
  }

  // Breadth-first, and within each level most-visited first, so truncating keeps the
  // part of the tree the search actually cared about. A visualiser that dropped the
  // principal variation to show unvisited siblings would be worse than useless.
  std::vector<NodeView> Snapshot(int max_nodes) const {
    std::vector<NodeView> out;
    if (max_nodes <= 0) return out;
    out.push_back(View(0, 0));

    std::vector<int> frontier{0};
    std::vector<int> depths{0};
    while (!frontier.empty() && static_cast<int>(out.size()) < max_nodes) {
      std::vector<int> next;
      std::vector<int> next_depths;
      for (std::size_t f = 0; f < frontier.size(); ++f) {
        const int node = frontier[f];
        const SharedNode<N>& n = nodes_[static_cast<std::size_t>(node)];
        if (n.state.load(std::memory_order_acquire) != kExpanded) continue;

        std::vector<NodeView> children;
        for (int c = n.first_child; c < n.first_child + n.num_children; ++c)
          children.push_back(View(c, depths[f] + 1));
        std::sort(children.begin(), children.end(),
                  [](const NodeView& a, const NodeView& b) {
                    return a.visits > b.visits;
                  });

        for (const NodeView& child : children) {
          if (static_cast<int>(out.size()) >= max_nodes) break;
          // Unvisited children are structure without information; including them
          // would crowd out visited nodes deeper in the principal variation.
          if (child.visits == 0) break;
          out.push_back(child);
          next.push_back(child.index);
          next_depths.push_back(child.depth);
        }
      }
      frontier = next;
      depths = next_depths;
    }
    return out;
  }

  ParallelSearchResult Collect() const {
    ParallelSearchResult result;
    const SharedNode<N>& root = nodes_[0];
    if (root.state.load(std::memory_order_acquire) == kExpanded) {
      int best_visits = -1;
      for (int c = root.first_child; c < root.first_child + root.num_children;
           ++c) {
        const SharedNode<N>& child = nodes_[static_cast<std::size_t>(c)];
        const int visits = child.visits.load(std::memory_order_relaxed);
        result.visits.emplace_back(child.move, visits);
        if (visits > best_visits) {
          best_visits = visits;
          result.best_move = child.move;
        }
      }
    }
    const int root_visits = root.visits.load(std::memory_order_relaxed);
    result.root_value =
        root_visits > 0
            ? static_cast<float>(
                  static_cast<double>(
                      root.value_sum.load(std::memory_order_relaxed)) /
                  kValueScale / root_visits)
            : 0.0f;
    result.distinct_nodes = NodesUsed();
    return result;
  }

 private:
  ParallelSearchConfig config_;
  std::vector<SharedNode<N>> nodes_;
  std::atomic<int> used_{1};
};

}  // namespace detail

}  // namespace hex

#endif  // HEX_SEARCH_TREE_HPP
