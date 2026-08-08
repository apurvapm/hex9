#ifndef HEX_PARALLEL_SELFPLAY_HPP
#define HEX_PARALLEL_SELFPLAY_HPP

#include <algorithm>
#include <atomic>
#include <barrier>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "hex/board.hpp"
#include "hex/puct.hpp"
#include "hex/selfplay_record.hpp"

namespace hex {

struct ParallelConfig {
  int threads = 1;
  int games = 100;
  int simulations = 200;
  float c_puct = 1.5f;
  float dirichlet_alpha = 0.3f;
  float dirichlet_weight = 0.25f;
  int temperature_moves = 8;
  std::uint64_t seed = 1;
  bool allow_swap = true;
  // Games held in memory before being flushed in order. Zero picks a default
  // proportional to the pool, which is enough to keep every worker fed without
  // buffering a whole run: at 9x9 a record is roughly 20 KB.
  int block = 0;
};

namespace detail {

// Games buffered per flush when the caller does not choose. Large enough that the
// barrier below is amortised over real work, small enough that the buffer stays
// trivial: at 9x9 a record is roughly 20 KB, so 64 of them is about 1.3 MB.
inline int DefaultBlock(int threads) {
  const int scaled = threads * 8;
  return scaled > 64 ? scaled : 64;
}

}  // namespace detail

struct ParallelStats {
  long long total_plies = 0;
  int games = 0;
  int red_wins = 0;
  int swaps = 0;
};

// One self-play game. Seeded from (run seed, game index) rather than from a
// shared stream, so a game's content depends only on its index and never on
// which worker happened to run it.
template <int N, typename Evaluator>
SelfPlayRecord PlayOneGame(const ParallelConfig& config, int index,
                           Evaluator& evaluate) {
  Board<N> board;
  SelfPlayRecord record;

  const std::uint64_t game_seed =
      config.seed * 1000003ULL + static_cast<std::uint64_t>(index);
  Puct<N> search(typename Puct<N>::Config{
      config.simulations, config.c_puct, config.dirichlet_alpha,
      config.dirichlet_weight, game_seed, config.allow_swap});

  while (!board.IsTerminal()) {
    const auto result = search.Search(board, evaluate);

    std::vector<std::pair<int, std::uint16_t>> entries;
    for (const auto& [action, count] : result.visits)
      if (count > 0)
        entries.emplace_back(action, static_cast<std::uint16_t>(count));

    const float temperature =
        board.MoveCount() < config.temperature_moves ? 1.0f : 0.0f;
    const int move = search.SampleMove(result, temperature);

    record.moves.push_back(move);
    record.visits.push_back(std::move(entries));

    if (move == Board<N>::kSwapMove) {
      board.PlaySwap();
    } else {
      board.Play(move);
    }
  }

  record.winner = board.Winner() == Cell::kRed ? 1 : -1;
  return record;
}

// Plays independent games across a thread pool.
//
// There is no shared tree, so there are no atomics on node statistics and no
// virtual loss: each worker owns its Puct instance and Board outright. That
// follows from the measured batching result -- a CPU core is already saturated by
// one 9x9 convolution, so throughput comes from running games side by side rather
// than from driving one search harder.
//
// MakeEvaluator is invoked once per worker, not once per game, so each thread owns
// its own ORT session with intra_op_threads = 1. Sharing one session across
// threads would contend on its internal pool, which is the comparison the design
// notes ask to be measured rather than assumed.
//
// Output is written in game-index order at any thread count, so a shard is
// bit-identical from 1 thread or 15. Records are flushed a block at a time, which
// bounds the buffer instead of holding the whole run.
//
// That guarantee has a precondition: the evaluator must be a pure function of the
// board. HeuristicEvaluator and OnnxEvaluator are. RolloutEvaluator is not -- it
// owns an RNG, so its answer depends on how many positions that worker evaluated
// earlier, which depends on which games it happened to claim. Self-play with a
// stateful evaluator still produces valid games, but not reproducible ones, and no
// amount of index-ordered flushing recovers it. tests/test_parallel.cpp pins the
// pure case and demonstrates the stateful one diverging, so the distinction stays
// visible rather than being rediscovered.
//
// Multi-threaded ORT sessions were the suspected second way to lose this, on the
// theory that parallel reductions inside one inference call could vary run to run
// and flip an argmax. Measured on the 9x9 network and found not to: with
// --intra-threads=4, shards were byte-identical run to run, identical between 1 and
// 8 workers, and identical to intra-threads=1. Stated as tested rather than assumed,
// and not a general guarantee -- it covers stock ops on ORT 1.28 CPU. The reason to
// keep intra-op at 1 is throughput, where per-worker sessions win by 5.3x, not
// determinism.
//
// Workers are created once for the whole run, not once per block. An earlier
// version spawned and joined the pool per block, and at the default block size it
// reached 6.02x on 15 threads against 6.42x when the whole run was buffered as a
// single block. Persistent workers recover that gap while keeping the buffer
// bounded. Note the two candidate causes -- thread spawn cost and stragglers
// idling the pool at each block boundary -- are not separated by that
// measurement, so neither is claimed here as the culprit.
//
// Sink is invoked as sink(SelfPlayRecord&) once per game, in index order, from
// whichever worker runs the flush. It takes a sink rather than a RecordWriter so a
// benchmark can measure the search without a file in the timing path: the flush is
// serial, so counting I/O there would report storage latency as thread contention.
template <int N, typename MakeEvaluator, typename Sink>
ParallelStats RunParallelSelfPlay(const ParallelConfig& config,
                                  MakeEvaluator make_evaluator, Sink sink) {
  const int threads = std::max(1, config.threads);
  const int block =
      config.block > 0 ? config.block : detail::DefaultBlock(threads);
  ParallelStats stats;
  if (config.games <= 0) return stats;

  std::vector<SelfPlayRecord> records(static_cast<std::size_t>(block));
  int block_start = 0;
  int block_end = std::min(block, config.games);
  bool finished = false;
  std::atomic<int> next{0};

  // Runs on one worker once every worker has arrived at the barrier, and before
  // any of them is released. That makes it the right place to flush: every write
  // into `records` for this block happens-before the arrival, and no worker can
  // begin the next block until this returns. So block_start, block_end and
  // finished need no synchronisation of their own despite being plain values --
  // the barrier supplies it, which is what the TSan job is there to confirm.
  const auto flush_block = [&] {
    for (int index = block_start; index < block_end; ++index) {
      SelfPlayRecord& record =
          records[static_cast<std::size_t>(index - block_start)];
      stats.total_plies += static_cast<long long>(record.moves.size());
      if (record.winner > 0) ++stats.red_wins;
      for (const int move : record.moves)
        if (move == Board<N>::kSwapMove) ++stats.swaps;
      ++stats.games;
      sink(record);
      record = SelfPlayRecord{};
    }
    block_start = block_end;
    block_end = std::min(block_start + block, config.games);
    next.store(block_start, std::memory_order_release);
    finished = block_start >= config.games;
  };

  std::barrier sync(threads, flush_block);

  const auto worker = [&] {
    // Constructed inside the thread so the session belongs to it.
    auto evaluate = make_evaluator();
    for (;;) {
      const int start = block_start;
      const int end = block_end;
      for (;;) {
        // acq_rel, not relaxed. Relaxed is enough under the standard -- the
        // counter publishes no data, and std::barrier already guarantees that
        // everything a worker did before arriving happens-before the completion
        // step. But ThreadSanitizer does not derive that edge from libc++'s
        // barrier: with a relaxed counter it reported races between a worker
        // writing its slot and the completion reading it, and a minimal repro
        // confirmed the reports appear and vanish purely with the memory order
        // while a per-slot ownership check showed zero actual collisions.
        //
        // So the release here is what carries a worker's slot writes to the
        // thread that runs the completion, via that thread's own acquire. It
        // costs one read-modify-write per game -- against a game that takes
        // milliseconds -- and buys a race report that CI can believe. An
        // unverifiable ordering is worth less than a free one.
        const int index = next.fetch_add(1, std::memory_order_acq_rel);
        if (index >= end) break;
        records[static_cast<std::size_t>(index - start)] =
            PlayOneGame<N>(config, index, evaluate);
      }
      sync.arrive_and_wait();
      if (finished) break;
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (std::thread& thread : pool) thread.join();

  return stats;
}

}  // namespace hex

#endif  // HEX_PARALLEL_SELFPLAY_HPP
