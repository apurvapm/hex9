// Head-to-head matches between checkpoints, reported as Elo.
//
//   ./arena --size=5 --a=gen3.onnx --b=gen0.onnx --pairs=50 --threads=8
//   ./arena --size=5 --a=gen3.onnx --heuristic-b --pairs=50
//   ./arena --size=5 --heuristic-a --heuristic-b --sims=400 --sims-b=25 --pairs=40
//
// This exists because the 5x5 gate saturates. Search at a few hundred simulations
// solves nearly every solved-position fixture case with even a first-generation
// network, so the gate confirms correctness and says almost nothing about progress.
// Relative strength between two checkpoints does not saturate, which makes it the
// metric that can still tell generation 3 from generation 4.
//
// Every match is colour-balanced in pairs. Red wins about 53% of games under
// uniform random play, so an unpaired result reports the colour assignment.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

#include "hex/alphabeta.hpp"
#include "hex/arena.hpp"
#include "hex/board.hpp"

#ifdef HEX_WITH_ONNX
#include "hex/onnx_evaluator.hpp"
#endif

namespace {

struct Options {
  int size = 5;
  int pairs = 50;
  int simulations = 200;
  int simulations_b = 0;
  float c_puct = 1.5f;
  int temperature_moves = 6;
  std::uint64_t seed = 1;
  int threads = 1;
  bool allow_swap = true;
  std::string model_a;
  std::string model_b;
  bool heuristic_a = false;
  bool heuristic_b = false;
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

void Report(const char* label_a, const char* label_b,
            const hex::ArenaResult& result) {
  const double rate = result.WinRate();
  const double elo = hex::EloDifference(rate);
  const double error = hex::EloStandardError(rate, result.games);

  std::printf("\n%s vs %s\n", label_a, label_b);
  std::printf("  games        : %d (%d colour-balanced pairs)\n", result.games,
              result.games / 2);
  std::printf("  record       : %d-%d (%.1f%%)\n", result.wins, result.losses,
              100.0 * rate);
  std::printf("  as red       : %d/%d\n", result.first_agent_as_red_wins,
              result.games / 2);
  std::printf("  as blue      : %d/%d\n", result.first_agent_as_blue_wins,
              result.games / 2);

  // Printed so a reader can see the pairing did its job. A number far from 50%
  // means colour, not strength, is driving the result.
  std::printf("  red won      : %d/%d (%.0f%%) across both agents\n",
              result.red_wins_overall, result.games,
              100.0 * result.red_wins_overall / result.games);

  if (rate <= 0.0 || rate >= 1.0) {
    std::printf("  elo          : %s%.0f (bound, clean sweep)\n",
                rate >= 1.0 ? "> +" : "< -", hex::kEloCap);
  } else {
    std::printf("  elo          : %+.0f +/- %.0f\n", elo, error);
  }

  // One standard error is not significance. Saying so beats leaving a reader to
  // assume a 30-Elo gap from 100 games means something.
  if (std::abs(elo) < 2.0 * error)
    std::printf(
        "  the interval spans zero: this match does not separate the two\n");
}

template <int N>
int Run(const Options& options) {
  hex::ArenaConfig config;
  config.pairs = options.pairs;
  config.simulations = options.simulations;
  config.simulations_second = options.simulations_b;
  config.c_puct = options.c_puct;
  config.temperature_moves = options.temperature_moves;
  config.seed = options.seed;
  config.allow_swap = options.allow_swap;
  config.threads = options.threads;

  const bool a_is_heuristic = options.heuristic_a || options.model_a.empty();
  const bool b_is_heuristic = options.heuristic_b || options.model_b.empty();

#ifndef HEX_WITH_ONNX
  if (!a_is_heuristic || !b_is_heuristic) {
    std::printf("built without ONNX Runtime; both sides must be --heuristic\n");
    return 1;
  }
#endif

  if (a_is_heuristic && b_is_heuristic) {
    const hex::ArenaResult result = hex::RunArena<N>(
        config, [] { return HeuristicEvaluator<N>{}; },
        [] { return HeuristicEvaluator<N>{}; });
    Report("heuristic", "heuristic", result);
    return 0;
  }

#ifdef HEX_WITH_ONNX
  const std::string model_a = options.model_a;
  const std::string model_b = options.model_b;

  if (a_is_heuristic) {
    const hex::ArenaResult result = hex::RunArena<N>(
        config, [] { return HeuristicEvaluator<N>{}; },
        [&model_b] { return hex::OnnxEvaluator<N>(model_b, 1); });
    Report("heuristic", model_b.c_str(), result);
    return 0;
  }
  if (b_is_heuristic) {
    const hex::ArenaResult result = hex::RunArena<N>(
        config, [&model_a] { return hex::OnnxEvaluator<N>(model_a, 1); },
        [] { return HeuristicEvaluator<N>{}; });
    Report(model_a.c_str(), "heuristic", result);
    return 0;
  }
  const hex::ArenaResult result = hex::RunArena<N>(
      config, [&model_a] { return hex::OnnxEvaluator<N>(model_a, 1); },
      [&model_b] { return hex::OnnxEvaluator<N>(model_b, 1); });
  Report(model_a.c_str(), model_b.c_str(), result);
  return 0;
#else
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
    } else if (Flag(argv[i], "--pairs", value)) {
      options.pairs = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--sims", value)) {
      options.simulations = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--sims-b", value)) {
      options.simulations_b = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--cpuct", value)) {
      options.c_puct = std::strtof(value.c_str(), nullptr);
    } else if (Flag(argv[i], "--temp-moves", value)) {
      options.temperature_moves = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--seed", value)) {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (Flag(argv[i], "--threads", value)) {
      options.threads = std::atoi(value.c_str());
    } else if (Flag(argv[i], "--a", value)) {
      options.model_a = value;
    } else if (Flag(argv[i], "--b", value)) {
      options.model_b = value;
    } else if (std::strcmp(argv[i], "--heuristic-a") == 0) {
      options.heuristic_a = true;
    } else if (std::strcmp(argv[i], "--heuristic-b") == 0) {
      options.heuristic_b = true;
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
