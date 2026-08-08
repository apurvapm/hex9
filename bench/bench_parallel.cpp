// Scaling measurements for parallel self-play.
//
//   ./bench_parallel --size=9 --games-per-worker=12 --sims=200 --repeats=3
//   ./bench_parallel --size=9 --model=hex9.onnx --games-per-worker=12 --sims=50
//   ./bench_parallel --size=9 --search-modes --sims=4000 --threads=1,4,8
//
// Two things this measures that a single timed run cannot.
//
// First, medians over repeated samples. On a machine with asymmetric cores the
// scheduler decides which threads land on performance cores and which on
// efficiency cores, and it does not decide the same way every run. Single samples
// came out non-monotonic -- 10 threads beating 12 -- which is sampling noise, not a
// property of the code.
//
// Second, workers against session threads at a fixed total. ONNX Runtime can
// parallelise inside one inference call, so the same core budget can be spent on
// many single-threaded workers or few multi-threaded ones. The design notes expect
// per-worker sessions to win, because intra-op parallelism scales poorly across the
// small convolutions a 9x9 board produces while independent games are embarrassingly
// parallel. Expecting is not measuring, so this reports both.
//
// No file is written: the driver takes a sink and this passes one that discards, so
// the serial flush does not put storage latency inside the timing.

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <thread>
#include <vector>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
#include "hex/parallel_puct.hpp"
#include "hex/parallel_selfplay.hpp"

#ifdef HEX_WITH_ONNX
#include "hex/onnx_evaluator.hpp"
#endif

#ifdef __APPLE__
#include <sys/sysctl.h>
#endif

namespace {

struct Options {
  int size = 9;
  // Work is specified per worker, not per run, so total games scale with the
  // thread count. Two reasons, both learned from getting it wrong.
  //
  // Constructing an ORT session costs hundreds of milliseconds and happens inside
  // the timed region, because a session has to belong to the thread that uses it.
  // With a fixed total, 15 workers each played under three games and 15 setups
  // dominated the measurement: the 15-thread point came out depressed with a 99%
  // spread between its best and worst repeat. Fixing games *per worker* amortises
  // setup identically at every point on the curve.
  //
  // It also keeps wall time per point roughly constant, so a sweep costs the same
  // whether it starts at 1 thread or 15.
  int games_per_worker = 12;
  int simulations = 200;
  int repeats = 3;
  std::uint64_t seed = 1;
  std::string model;
  std::vector<int> thread_counts;
  bool search_modes = false;
};

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

// Reported so nobody reads a speedup against the wrong denominator. On an
// asymmetric machine the ideal is not the logical core count: it is the
// performance cores plus the efficiency cores discounted by their throughput
// ratio, which has to be measured separately (see docs/DESIGN.md).
void PrintTopology() {
  const unsigned logical = std::thread::hardware_concurrency();
#ifdef __APPLE__
  int performance = 0, efficiency = 0;
  std::size_t length = sizeof(int);
  if (sysctlbyname("hw.perflevel0.logicalcpu", &performance, &length, nullptr,
                   0) == 0 &&
      sysctlbyname("hw.perflevel1.logicalcpu", &efficiency, &length, nullptr,
                   0) == 0 &&
      efficiency > 0) {
    std::printf("cores: %u logical (%d performance, %d efficiency)\n", logical,
                performance, efficiency);
    std::printf(
        "  asymmetric: speedup against logical core count understates the\n"
        "  result. See docs/DESIGN.md for the achievable ceiling.\n");
    return;
  }
#endif
  std::printf("cores: %u logical\n", logical);
}

// Median rather than mean: one descheduled run should not move the number, and
// median over an even count averages the middle pair.
double Median(std::vector<double> samples) {
  if (samples.empty()) return 0.0;
  std::sort(samples.begin(), samples.end());
  const std::size_t middle = samples.size() / 2;
  if (samples.size() % 2 == 1) return samples[middle];
  return 0.5 * (samples[middle - 1] + samples[middle]);
}

struct Sample {
  double moves_per_sec = 0.0;
  double best = 0.0;
  double spread = 0.0;  // (best - worst) / best, as a noise indicator
};

template <int N, typename MakeEvaluator>
Sample Measure(const Options& options, int threads, MakeEvaluator make) {
  std::vector<double> samples;
  samples.reserve(static_cast<std::size_t>(options.repeats));

  for (int repeat = 0; repeat < options.repeats; ++repeat) {
    const hex::ParallelConfig config{
        .threads = threads,
        .games = threads * options.games_per_worker,
        .simulations = options.simulations,
        // Root noise off: it costs a Dirichlet sample per search and adds
        // variance without changing what is being measured.
        .dirichlet_weight = 0.0f,
        // Same seed every repeat, so repeats measure scheduling and timing noise
        // rather than the variance between different sets of games. Varying it
        // would mix the two and inflate the spread column with something that is
        // not jitter.
        .seed = options.seed,
    };
    (void)repeat;

    const auto started = std::chrono::steady_clock::now();
    const hex::ParallelStats stats = hex::RunParallelSelfPlay<N>(
        config, make, [](hex::SelfPlayRecord&) {});
    const double elapsed =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started)
            .count();
    samples.push_back(static_cast<double>(stats.total_plies) / elapsed);
  }

  Sample result;
  result.moves_per_sec = Median(samples);
  result.best = *std::max_element(samples.begin(), samples.end());
  const double worst = *std::min_element(samples.begin(), samples.end());
  result.spread = result.best > 0.0 ? (result.best - worst) / result.best : 0.0;
  return result;
}

template <int N, typename MakeEvaluator>
void Sweep(const Options& options, const char* label, MakeEvaluator make) {
  std::printf("\n%s: %d games per worker, %d simulations, median of %d\n", label,
              options.games_per_worker, options.simulations, options.repeats);
  std::printf("%8s %13s %10s %10s %8s\n", "threads", "moves/sec", "speedup",
              "per thread", "spread");

  double baseline = 0.0;
  for (const int threads : options.thread_counts) {
    const Sample sample = Measure<N>(options, threads, make);
    if (baseline == 0.0) baseline = sample.moves_per_sec;
    const double speedup = sample.moves_per_sec / baseline;
    // One decimal place: at 9x9 with a real network a single thread manages about
    // 33 moves/sec, and rounding that to an integer throws away precision right
    // where the target board size lives.
    std::printf("%8d %13.1f %9.2fx %9.0f%% %7.0f%%\n", threads,
                sample.moves_per_sec, speedup, 100.0 * speedup / threads,
                100.0 * sample.spread);
    std::fflush(stdout);
  }
}

#ifdef HEX_WITH_ONNX
// Same core budget, split differently between workers and intra-op threads.
template <int N>
void SessionThreadComparison(const Options& options, int budget) {
  // Note the setup bias runs against the configuration expected to win: 15 workers
  // pay fifteen session constructions, one worker pays one. A win for per-worker
  // sessions here is therefore a conservative one.
  std::printf(
      "\nworkers against session threads at a budget of %d: %d games per "
      "worker, median of %d\n",
      budget, options.games_per_worker, options.repeats);
  std::printf("%8s %8s %13s %10s\n", "workers", "intra", "moves/sec",
              "relative");

  double reference = 0.0;
  for (int workers = budget; workers >= 1; workers /= 2) {
    const int intra = budget / workers;
    if (intra < 1) continue;
    const std::string model = options.model;
    const Sample sample = Measure<N>(options, workers, [&model, intra] {
      return hex::OnnxEvaluator<N>(model, intra);
    });
    if (reference == 0.0) reference = sample.moves_per_sec;
    std::printf("%8d %8d %13.1f %9.2fx\n", workers, intra,
                sample.moves_per_sec, sample.moves_per_sec / reference);
    std::fflush(stdout);
    if (workers == 1) break;
  }
}
#endif

// Tree, root and leaf parallel on one position, same budget, same hardware.
//
// Reported together because they trade different things. Root-parallel duplicates
// work across independent trees, so it should be fastest per simulation and
// narrowest in coverage. Tree-parallel shares one tree and pays synchronisation for
// breadth. Leaf-parallel serialises the descent and only parallelises evaluation,
// which on a CPU is the one part that does not need it.
template <int N, typename MakeEvaluator>
void SearchModeComparison(const Options& options, MakeEvaluator make) {
  std::printf("\nsearch modes on one position: %d simulations, median of %d\n",
              options.simulations, options.repeats);
  std::printf("%6s %8s %12s %12s %8s\n", "mode", "threads", "sims/sec",
              "tree nodes", "move");

  hex::Board<N> board;
  // A few plies in, so the tree has real structure rather than a uniform opening.
  board.Play(N * N / 2);
  board.Play(N + 1);

  const char* names[] = {"root", "tree", "leaf"};
  const hex::ParallelMode modes[] = {
      hex::ParallelMode::kRoot, hex::ParallelMode::kTree,
      hex::ParallelMode::kLeaf};

  for (int m = 0; m < 3; ++m) {
    for (const int threads : options.thread_counts) {
      std::vector<double> samples;
      int nodes = 0;
      int move = -1;
      for (int repeat = 0; repeat < options.repeats; ++repeat) {
        hex::ParallelSearchConfig config;
        config.threads = threads;
        config.simulations = options.simulations;
        config.seed = options.seed;
        config.leaf_batch = std::max(2, threads);

        const auto started = std::chrono::steady_clock::now();
        const auto result =
            hex::SearchParallel<N>(board, modes[m], config, make);
        const double elapsed =
            std::chrono::duration<double>(std::chrono::steady_clock::now() -
                                          started)
                .count();
        samples.push_back(options.simulations / elapsed);
        nodes = result.distinct_nodes;
        move = result.best_move;
      }
      std::printf("%6s %8d %12.0f %12d %8d\n", names[m], threads,
                  Median(samples), nodes, move);
      std::fflush(stdout);
    }
  }
}

template <int N>
int Run(const Options& options) {
  if (options.search_modes) {
    if (options.model.empty()) {
      SearchModeComparison<N>(options, [] { return HeuristicEvaluator<N>{}; });
      return 0;
    }
#ifdef HEX_WITH_ONNX
    const std::string model = options.model;
    SearchModeComparison<N>(options,
                            [&model] { return hex::OnnxEvaluator<N>(model, 1); });
    return 0;
#else
    std::printf("built without ONNX Runtime; omit --model\n");
    return 1;
#endif
  }

  if (options.model.empty()) {
    Sweep<N>(options, "heuristic evaluator", [] {
      return HeuristicEvaluator<N>{};
    });
    return 0;
  }
#ifdef HEX_WITH_ONNX
  const std::string model = options.model;
  Sweep<N>(options, "network evaluator, one session per worker",
           [&model] { return hex::OnnxEvaluator<N>(model, 1); });
  const int budget = options.thread_counts.back();
  SessionThreadComparison<N>(options, budget);
  return 0;
#else
  std::printf("built without ONNX Runtime; omit --model\n");
  return 1;
#endif
}

bool Flag(const char* arg, const char* name, std::string& value) {
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
    if (Flag(argv[i], "--size", value)) {
      options.size = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--games-per-worker", value)) {
      options.games_per_worker = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--sims", value)) {
      options.simulations = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--repeats", value)) {
      options.repeats = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--seed", value)) {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (Flag(argv[i], "--model", value)) {
      options.model = value;
    } else if (Flag(argv[i], "--threads", value)) {
      options.thread_counts.clear();
      const char* cursor = value.c_str();
      while (*cursor != '\0') {
        options.thread_counts.push_back(std::atoi(cursor));
        while (*cursor != '\0' && *cursor != ',') ++cursor;
        if (*cursor == ',') ++cursor;
      }
    } else if (std::strcmp(argv[i], "--search-modes") == 0) {
      options.search_modes = true;
    } else {
      std::printf("unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  if (options.thread_counts.empty()) {
    const int logical =
        std::max(1, static_cast<int>(std::thread::hardware_concurrency()));
    for (const int candidate : {1, 2, 4, 6, 8, 12, 16, 24, 32})
      if (candidate <= logical) options.thread_counts.push_back(candidate);
    if (options.thread_counts.empty() ||
        options.thread_counts.back() != logical)
      options.thread_counts.push_back(logical);
  }

  PrintTopology();

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
