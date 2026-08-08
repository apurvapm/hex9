// Generates self-play games and writes them as training records.
//
//   ./selfplay --size=5 --model=tiny.onnx --games=200 --sims=200 --out=shard.bin
//   ./selfplay --size=9 --model=hex9.onnx --games=2000 --threads=15 --out=shard.bin
//   ./selfplay --size=5 --heuristic --games=50 --out=shard.bin
//
// The heuristic mode needs no model and exists so the pipeline can be exercised
// before a network is trained, and so the record format can be tested in CI
// without an ONNX Runtime dependency.

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
#include "hex/parallel_selfplay.hpp"
#include "hex/puct.hpp"
#include "hex/selfplay_record.hpp"

#ifdef HEX_WITH_ONNX
#include "hex/onnx_evaluator.hpp"
#endif

namespace {

struct Options {
  int size = 5;
  int games = 100;
  int simulations = 200;
  float c_puct = 1.5f;
  float dirichlet_alpha = 0.3f;
  float dirichlet_weight = 0.25f;
  // Moves before the temperature drops to zero. Early randomness widens the
  // opening book; late greediness keeps the endgame sharp enough that the value
  // target reflects real play.
  int temperature_moves = 8;
  std::uint64_t seed = 1;
  bool allow_swap = true;
  bool heuristic = false;
  std::string model;
  std::string output = "shard.bin";
  // Defaults to one thread so the common invocation stays deterministic without
  // anyone having to ask for it. Parallelism is opt-in.
  int threads = 1;
  int block = 0;
  // ORT threads *inside* one inference call, as opposed to workers. Exists so the
  // reproducibility hazard can be tested rather than asserted; the measured
  // workers-versus-session-threads result argues against ever raising it.
  int intra_threads = 1;
};

// Stand-in for the network: uniform priors and a value from the
// connection-distance heuristic, squashed into the same range a value head
// produces.
template <int N>
struct HeuristicEvaluator {
  void operator()(const hex::Board<N>& board, hex::Evaluation<N>& out) const {
    out.priors.fill(1.0f);
    const hex::Player me = board.ToPlay();
    const int mine = hex::AlphaBeta<N>::ConnectionDistance(board, me);
    const int theirs =
        hex::AlphaBeta<N>::ConnectionDistance(board, hex::Opponent(me));
    out.value = std::tanh(0.5f * static_cast<float>(theirs - mine));
  }
};

// MakeEvaluator is a factory, not an evaluator: the parallel driver calls it once
// per worker so each thread owns its own ORT session. Passing a single evaluator
// would share one session across threads, which is the arrangement the design
// notes want measured rather than assumed.
template <int N, typename MakeEvaluator>
int PlayGames(const Options& options, MakeEvaluator make_evaluator) {
  hex::RecordWriter writer(options.output, N);
  if (!writer.ok()) {
    std::printf("could not open %s for writing\n", options.output.c_str());
    return 1;
  }

  const hex::ParallelConfig config{
      .threads = options.threads,
      .games = options.games,
      .simulations = options.simulations,
      .c_puct = options.c_puct,
      .dirichlet_alpha = options.dirichlet_alpha,
      .dirichlet_weight = options.dirichlet_weight,
      .temperature_moves = options.temperature_moves,
      .seed = options.seed,
      .allow_swap = options.allow_swap,
      .block = options.block,
  };

  const auto started = std::chrono::steady_clock::now();
  const hex::ParallelStats stats = hex::RunParallelSelfPlay<N>(
      config, make_evaluator,
      [&writer](hex::SelfPlayRecord& record) { writer.Write(record); });
  const double elapsed =
      std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
          .count();

  writer.Close();
  std::printf("wrote %u games to %s\n", writer.games(), options.output.c_str());
  std::printf("  threads     : %d\n", options.threads);
  std::printf("  mean length : %.1f plies\n",
              static_cast<double>(stats.total_plies) / stats.games);
  std::printf("  red wins    : %d/%d (%.0f%%)\n", stats.red_wins, stats.games,
              100.0 * stats.red_wins / stats.games);
  std::printf("  swaps taken : %d\n", stats.swaps);
  std::printf("  elapsed     : %.2fs\n", elapsed);
  // Reported per run because these are the numbers the phase 3 scaling curve is
  // built from; deriving them later from a wall-clock guess would not do.
  std::printf("  throughput  : %.1f games/sec, %.0f moves/sec\n",
              stats.games / elapsed,
              static_cast<double>(stats.total_plies) / elapsed);
  return 0;
}

template <int N>
int Run(const Options& options) {
  if (options.heuristic)
    return PlayGames<N>(options, [] { return HeuristicEvaluator<N>{}; });
#ifdef HEX_WITH_ONNX
  if (options.model.empty()) {
    std::printf("supply --model=<path.onnx> or pass --heuristic\n");
    return 1;
  }
  // One session per worker, each single-threaded. Returned as a prvalue so it is
  // constructed in place in the worker rather than moved into it.
  return PlayGames<N>(options, [&options] {
    return hex::OnnxEvaluator<N>(options.model, options.intra_threads);
  });
#else
  std::printf("built without ONNX Runtime; pass --heuristic\n");
  return 1;
#endif
}

bool MatchFlag(const char* arg, const char* name, std::string& value) {
  const std::size_t len = std::strlen(name);
  if (std::strncmp(arg, name, len) != 0 || arg[len] != '=') return false;
  value = arg + len + 1;
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string value;

  for (int i = 1; i < argc; ++i) {
    if (MatchFlag(argv[i], "--size", value)) {
      options.size = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--games", value)) {
      options.games = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--sims", value)) {
      options.simulations = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--cpuct", value)) {
      options.c_puct = std::strtof(value.c_str(), nullptr);
    } else if (MatchFlag(argv[i], "--noise", value)) {
      options.dirichlet_weight = std::strtof(value.c_str(), nullptr);
    } else if (MatchFlag(argv[i], "--temp-moves", value)) {
      options.temperature_moves = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--threads", value)) {
      options.threads = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--block", value)) {
      options.block = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--intra-threads", value)) {
      options.intra_threads = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--seed", value)) {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (MatchFlag(argv[i], "--model", value)) {
      options.model = value;
    } else if (MatchFlag(argv[i], "--out", value)) {
      options.output = value;
    } else if (std::strcmp(argv[i], "--heuristic") == 0) {
      options.heuristic = true;
    } else if (std::strcmp(argv[i], "--no-swap") == 0) {
      options.allow_swap = false;
    } else {
      std::printf("unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  switch (options.size) {
    case 5: return Run<5>(options);
    case 7: return Run<7>(options);
    case 9: return Run<9>(options);
    case 11: return Run<11>(options);
    default:
      std::printf("supported sizes are 5, 7, 9, 11\n");
      return 1;
  }
}
