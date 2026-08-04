// Generates self-play games and writes them as training records.
//
//   ./selfplay --size=5 --model=tiny.onnx --games=200 --sims=200 --out=shard.bin
//   ./selfplay --size=5 --heuristic --games=50 --out=shard.bin
//
// The heuristic mode needs no model and exists so the pipeline can be exercised
// before a network is trained, and so the record format can be tested in CI
// without an ONNX Runtime dependency.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
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

template <int N, typename Evaluator>
int PlayGames(const Options& options, Evaluator&& evaluate) {
  hex::RecordWriter writer(options.output, N);
  if (!writer.ok()) {
    std::printf("could not open %s for writing\n", options.output.c_str());
    return 1;
  }

  long long total_plies = 0;
  int red_wins = 0;
  int swaps = 0;

  for (int game = 0; game < options.games; ++game) {
    hex::Board<N> board;
    hex::SelfPlayRecord record;

    // Seeding per game rather than from one shared stream keeps every game
    // independently reproducible, which is what makes a threaded run
    // debuggable later.
    const std::uint64_t game_seed =
        options.seed * 1000003ULL + static_cast<std::uint64_t>(game);
    hex::Puct<N> search(typename hex::Puct<N>::Config{
        options.simulations, options.c_puct, options.dirichlet_alpha,
        options.dirichlet_weight, game_seed, options.allow_swap});

    while (!board.IsTerminal()) {
      const auto result = search.Search(board, evaluate);

      std::vector<std::pair<int, std::uint16_t>> entries;
      for (const auto& [action, count] : result.visits)
        if (count > 0)
          entries.emplace_back(action, static_cast<std::uint16_t>(count));

      const float temperature =
          board.MoveCount() < options.temperature_moves ? 1.0f : 0.0f;
      const int move = search.SampleMove(result, temperature);

      record.moves.push_back(move);
      record.visits.push_back(std::move(entries));

      if (move == hex::Board<N>::kSwapMove) {
        board.PlaySwap();
        ++swaps;
      } else {
        board.Play(move);
      }
    }

    record.winner = board.Winner() == hex::Cell::kRed ? 1 : -1;
    if (record.winner > 0) ++red_wins;
    total_plies += static_cast<long long>(record.moves.size());
    writer.Write(record);

    if ((game + 1) % 50 == 0)
      std::printf("  %d/%d games\n", game + 1, options.games);
  }

  writer.Close();
  std::printf("wrote %u games to %s\n", writer.games(), options.output.c_str());
  std::printf("  mean length : %.1f plies\n",
              static_cast<double>(total_plies) / options.games);
  std::printf("  red wins    : %d/%d (%.0f%%)\n", red_wins, options.games,
              100.0 * red_wins / options.games);
  std::printf("  swaps taken : %d\n", swaps);
  return 0;
}

template <int N>
int Run(const Options& options) {
  if (options.heuristic) {
    HeuristicEvaluator<N> evaluator;
    return PlayGames<N>(options, evaluator);
  }
#ifdef HEX_WITH_ONNX
  if (options.model.empty()) {
    std::printf("supply --model=<path.onnx> or pass --heuristic\n");
    return 1;
  }
  hex::OnnxEvaluator<N> evaluator(options.model);
  return PlayGames<N>(options, evaluator);
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
