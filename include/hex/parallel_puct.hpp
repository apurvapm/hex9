#ifndef HEX_PARALLEL_PUCT_HPP
#define HEX_PARALLEL_PUCT_HPP

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cmath>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#include "hex/board.hpp"
#include "hex/puct.hpp"
#include "hex/search_tree.hpp"

namespace hex {

// Three ways to spend several threads on a single move, so they can be compared
// rather than argued about.
//
// Separate from puct.hpp on purpose. Puct is used by self-play, the arena, the CLI
// and the browser build, and tests/test_puct.cpp pins its check count; making its
// node statistics atomic would tax every one of those callers to serve a comparison
// only this phase needs. Root-parallel below reuses Puct verbatim, so the
// single-threaded search stays the one implementation everything else shares.
//
// What this is *not* for: self-play. That parallelises across independent games,
// which is strictly better on a CPU -- no shared state, no virtual loss, and
// scaling limited only by cores. Sharing a tree matters when there is one position
// to think about and idle cores, which is analysis and the browser demo.
enum class ParallelMode {
  // N independent trees from one root, visit counts summed. No shared state.
  kRoot,
  // N threads descending one tree, kept apart by virtual loss.
  kTree,
  // One descent at a time, leaves gathered and evaluated as a batch. Expected to
  // lose on CPU, where batching measured 20% rather than 20x.
  kLeaf,
};


// Root-parallel: N independent Puct searches, visit counts summed. Reuses Puct
// unchanged, so it inherits everything test_puct.cpp pins. Threads get distinct
// seeds, otherwise the trees would be identical and the extra threads idle.
template <int N, typename MakeEvaluator>
ParallelSearchResult SearchRootParallel(const Board<N>& board,
                                        const ParallelSearchConfig& config,
                                        MakeEvaluator make_evaluator) {
  const int threads = std::max(1, config.threads);
  const int per_thread = std::max(1, config.simulations / threads);

  std::vector<std::vector<std::pair<int, int>>> collected(
      static_cast<std::size_t>(threads));
  std::vector<float> values(static_cast<std::size_t>(threads), 0.0f);
  std::vector<long long> evals(static_cast<std::size_t>(threads), 0);
  std::vector<int> sizes(static_cast<std::size_t>(threads), 0);

  const auto worker = [&](int id) {
    auto evaluate = make_evaluator();
    long long evaluations = 0;
    const auto counting = [&evaluate, &evaluations](const Board<N>& b,
                                                    Evaluation<N>& out) {
      ++evaluations;
      evaluate(b, out);
    };

    typename Puct<N>::Config search_config{};
    search_config.simulations = per_thread;
    search_config.c_puct = config.c_puct;
    search_config.dirichlet_weight = 0.0f;
    search_config.allow_swap = config.allow_swap;
    search_config.seed =
        config.seed + static_cast<std::uint64_t>(id) * 0x9E3779B97F4A7C15ULL;

    Board<N> local = board;  // Puct mutates and restores in place
    Puct<N> search(search_config);
    const auto result = search.Search(local, counting);

    collected[static_cast<std::size_t>(id)] = result.visits;
    values[static_cast<std::size_t>(id)] = result.root_value;
    evals[static_cast<std::size_t>(id)] = evaluations;
    sizes[static_cast<std::size_t>(id)] = static_cast<int>(search.TreeSize());
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker, i);
  for (std::thread& thread : pool) thread.join();

  ParallelSearchResult result;
  std::vector<int> totals(static_cast<std::size_t>(N * N + 1), 0);
  for (const auto& visits : collected)
    for (const auto& [move, count] : visits)
      totals[static_cast<std::size_t>(move)] += count;

  int best_visits = -1;
  for (int move = 0; move <= N * N; ++move) {
    const int count = totals[static_cast<std::size_t>(move)];
    if (count <= 0) continue;
    result.visits.emplace_back(move, count);
    if (count > best_visits) {
      best_visits = count;
      result.best_move = move;
    }
  }
  double value = 0.0;
  for (const float v : values) value += v;
  result.root_value = static_cast<float>(value / threads);
  for (const long long e : evals) result.evaluations += e;
  // Summed across trees, and deliberately not deduplicated: N trees exploring the
  // same lines is exactly the redundancy tree-parallel exists to avoid, so the
  // number should look inflated when it is.
  for (const int s : sizes) result.distinct_nodes += s;
  return result;
}

// Tree-parallel: N threads descending one tree, kept apart by virtual loss.
template <int N, typename MakeEvaluator>
ParallelSearchResult SearchTreeParallel(const Board<N>& board,
                                        const ParallelSearchConfig& config,
                                        MakeEvaluator make_evaluator) {
  const int threads = std::max(1, config.threads);
  detail::SharedTree<N> tree(config);
  std::atomic<int> remaining{config.simulations};
  std::atomic<long long> evaluations{0};

  const auto worker = [&] {
    auto evaluate = make_evaluator();
    Board<N> local = board;
    long long local_evals = 0;

    for (;;) {
      if (remaining.fetch_sub(1, std::memory_order_relaxed) <= 0) break;

      int applied = 0;
      const int leaf = tree.Descend(local, applied);

      float value;
      if (local.IsTerminal()) {
        // The opponent completed a chain on the previous move, so the player to
        // move at this node has already lost.
        value = -1.0f;
      } else {
        Evaluation<N> eval;
        evaluate(local, eval);
        ++local_evals;
        tree.Expand(leaf, local, eval);
        value = eval.value;
      }
      tree.Backup(local, leaf, value, applied);
    }
    evaluations.fetch_add(local_evals, std::memory_order_relaxed);
  };

  // The root has to be expanded before any thread descends, or every thread would
  // race to expand it and the first simulations would be wasted.
  {
    auto evaluate = make_evaluator();
    Board<N> local = board;
    Evaluation<N> eval;
    evaluate(local, eval);
    tree.Expand(tree.Root(), local, eval);
    evaluations.fetch_add(1, std::memory_order_relaxed);
  }

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (std::thread& thread : pool) thread.join();

  ParallelSearchResult result = tree.Collect();
  result.evaluations = evaluations.load(std::memory_order_relaxed);
  return result;
}

// Leaf-parallel: one descent at a time, leaves gathered and evaluated together.
// Virtual loss makes successive descents in a batch pick different leaves.
template <int N, typename MakeEvaluator>
ParallelSearchResult SearchLeafParallel(const Board<N>& board,
                                       const ParallelSearchConfig& config,
                                       MakeEvaluator make_evaluator) {
  const int threads = std::max(1, config.threads);
  const int batch = std::max(1, config.leaf_batch);
  detail::SharedTree<N> tree(config);
  long long evaluations = 0;

  std::vector<std::decay_t<decltype(make_evaluator())>> evaluators;
  evaluators.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) evaluators.push_back(make_evaluator());

  Board<N> root_board = board;
  {
    Evaluation<N> eval;
    evaluators[0](root_board, eval);
    ++evaluations;
    tree.Expand(tree.Root(), root_board, eval);
  }

  std::vector<Board<N>> boards(static_cast<std::size_t>(batch));
  std::vector<int> leaves(static_cast<std::size_t>(batch), -1);
  std::vector<int> applied(static_cast<std::size_t>(batch), 0);
  std::vector<Evaluation<N>> evals(static_cast<std::size_t>(batch));
  std::vector<char> terminal(static_cast<std::size_t>(batch), 0);

  int size = 0;
  int done = 0;
  bool finished = false;

  // One barrier arrival per batch, and workers created once for the whole search.
  // Spawning a pool per batch measured five times slower than this at one thread,
  // which would have reported thread-creation cost as leaf-parallel's algorithmic
  // result. The same mistake was made and fixed in the self-play driver.
  //
  // The completion step runs on one thread with all others held, so it can back up
  // the batch that was just evaluated and descend the next one without any further
  // synchronisation.
  const auto prepare = [&] {
    if (size > 0) {
      for (int i = 0; i < size; ++i) {
        const std::size_t k = static_cast<std::size_t>(i);
        const float value = terminal[k] != 0 ? -1.0f : evals[k].value;
        if (terminal[k] == 0) {
          tree.Expand(leaves[k], boards[k], evals[k]);
          ++evaluations;
        }
        tree.Backup(boards[k], leaves[k], value, applied[k]);
      }
      done += size;
    }
    if (done >= config.simulations) {
      finished = true;
      size = 0;
      return;
    }
    size = std::min(batch, config.simulations - done);
    // Descents are serial by construction -- that is what makes this leaf-parallel
    // rather than tree-parallel. Virtual loss is what stops all of them landing on
    // the same leaf.
    for (int i = 0; i < size; ++i) {
      const std::size_t k = static_cast<std::size_t>(i);
      boards[k] = board;
      leaves[k] = tree.Descend(boards[k], applied[k]);
      terminal[k] = boards[k].IsTerminal() ? 1 : 0;
    }
  };

  std::barrier sync(threads, prepare);

  const auto worker = [&](int id) {
    for (;;) {
      sync.arrive_and_wait();
      if (finished) break;
      // Strided rather than a shared counter: the batch is small and a counter
      // would contend more than it balances.
      for (int i = id; i < size; i += threads) {
        const std::size_t k = static_cast<std::size_t>(i);
        if (terminal[k] != 0) continue;
        evaluators[static_cast<std::size_t>(id)](boards[k], evals[k]);
      }
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker, i);
  for (std::thread& thread : pool) thread.join();

  ParallelSearchResult result = tree.Collect();
  result.evaluations = evaluations;
  return result;
}

template <int N, typename MakeEvaluator>
ParallelSearchResult SearchParallel(const Board<N>& board, ParallelMode mode,
                                    const ParallelSearchConfig& config,
                                    MakeEvaluator make_evaluator) {
  switch (mode) {
    case ParallelMode::kRoot:
      return SearchRootParallel<N>(board, config, make_evaluator);
    case ParallelMode::kTree:
      return SearchTreeParallel<N>(board, config, make_evaluator);
    case ParallelMode::kLeaf:
      return SearchLeafParallel<N>(board, config, make_evaluator);
  }
  return {};
}

}  // namespace hex

#endif  // HEX_PARALLEL_PUCT_HPP
