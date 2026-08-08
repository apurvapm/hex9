// Scores the agent -- PUCT driving a trained network -- against solved positions.
//
//   ./build/gate_search --model=hex5.onnx --sims=200
//
// training/gate.py scores the *network*: raw policy and value against ground
// truth. This scores the *agent*, which is the stronger and more relevant claim,
// because search can repair a mediocre policy and a mediocre policy can waste a
// good search. The two gates share one fixture, so there is no second ground
// truth to keep in step.
//
// It also exercises a path nothing else does. The C++ encoder is pinned against
// Python by the golden fixture, and PyTorch is pinned against ONNX Runtime by
// export_onnx.py, but OnnxEvaluator's policy decode -- softmax over legal actions,
// then un-canonicalise back to board indices -- sits between those two checks and
// was covered by neither. A wrong mapping there shows up here as a collapsed hit
// rate on exactly one of the two board orientations.
//
// Requires ONNX Runtime: CMake adds this target only when ONNXRUNTIME_ROOT is set.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include "hex/board.hpp"
#include "hex/encoding.hpp"
#include "hex/onnx_evaluator.hpp"
#include "hex/puct.hpp"

namespace {

struct Case {
  std::vector<int> moves;
  int mover_value = 0;
  std::vector<int> winning;
};

struct Fixture {
  int size = 0;
  std::vector<Case> cases;
};

// Mirrors training/gate.py's parser. Kept deliberately strict: a fixture that
// half-parses would produce a plausible score against the wrong positions.
bool LoadFixture(const std::string& path, Fixture& out) {
  std::ifstream file(path);
  if (!file) {
    std::printf("could not open %s\n", path.c_str());
    return false;
  }

  std::string line;
  bool have_header = false;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;
    std::istringstream stream(line);

    if (!have_header) {
      int policy_size = 0;
      if (!(stream >> out.size >> policy_size)) return false;
      if (policy_size != out.size * out.size + 1) {
        std::printf("fixture policy width %d does not include swap\n",
                    policy_size);
        return false;
      }
      have_header = true;
      continue;
    }

    Case entry;
    int token = 0;
    while (stream >> token && token != -1) entry.moves.push_back(token);
    if (token != -1) return false;

    int count = 0;
    if (!(stream >> entry.mover_value >> count)) return false;
    for (int i = 0; i < count; ++i) {
      if (!(stream >> token)) return false;
      entry.winning.push_back(token);
    }
    if ((entry.mover_value == 1) != !entry.winning.empty()) {
      std::printf("fixture value disagrees with its winning-move list\n");
      return false;
    }
    out.cases.push_back(std::move(entry));
  }
  return have_header && !out.cases.empty();
}

struct Tally {
  int hits = 0;
  int total = 0;
  double chance = 0.0;
};

template <int N>
int Run(const Fixture& fixture, const std::string& model, int simulations,
        float c_puct, double min_margin, double max_gap) {
  hex::OnnxEvaluator<N> evaluator(model);

  int value_correct_won = 0, value_total_won = 0;
  int value_correct_lost = 0, value_total_lost = 0;
  Tally direct, transposed;

  for (std::size_t index = 0; index < fixture.cases.size(); ++index) {
    const Case& entry = fixture.cases[index];

    hex::Board<N> board;
    for (const int move : entry.moves) {
      if (move == hex::Board<N>::kSwapMove) {
        board.PlaySwap();
      } else {
        board.Play(move);
      }
    }
    if (board.IsTerminal()) {
      std::printf("fixture case %zu is already terminal\n", index);
      return 1;
    }

    // Root noise off. It exists to widen self-play openings; here it would just
    // add variance to a measurement.
    hex::Puct<N> search(typename hex::Puct<N>::Config{
        simulations, c_puct, 0.3f, 0.0f,
        static_cast<std::uint64_t>(index) + 1, true});
    const auto result = search.Search(board, evaluator);

    const int predicted = result.root_value >= 0.0f ? 1 : -1;
    if (entry.mover_value == 1) {
      ++value_total_won;
      value_correct_won += predicted == 1;
    } else {
      ++value_total_lost;
      value_correct_lost += predicted == -1;
    }

    // Only won positions can be scored on move choice: where every move loses,
    // a hit rate would be zero by construction rather than by weakness.
    if (entry.mover_value != 1) continue;

    Tally& tally =
        hex::Encoder<N>::NeedsTranspose(board) ? transposed : direct;
    ++tally.total;
    for (const int move : entry.winning)
      if (move == result.best_move) {
        ++tally.hits;
        break;
      }
    tally.chance += static_cast<double>(entry.winning.size()) /
                    static_cast<double>(board.NumEmpty());
  }

  const auto rate = [](double hits, int total) {
    return total > 0 ? hits / total : 0.0;
  };
  const double recall_won = rate(value_correct_won, value_total_won);
  const double recall_lost = rate(value_correct_lost, value_total_lost);
  const double balanced = 0.5 * (recall_won + recall_lost);
  const int scored = direct.total + transposed.total;
  const double hit_rate = rate(direct.hits + transposed.hits, scored);
  const double chance = rate(direct.chance + transposed.chance, scored);
  const double hit_direct = rate(direct.hits, direct.total);
  const double hit_transposed = rate(transposed.hits, transposed.total);

  std::printf("search gate: %zu solved %dx%d positions at %d simulations\n",
              fixture.cases.size(), N, N, simulations);
  std::printf("  value balanced accuracy : %5.1f%%   (won %.1f%%, lost %.1f%%)\n",
              100.0 * balanced, 100.0 * recall_won, 100.0 * recall_lost);
  std::printf("  agent plays a winner    : %5.1f%%   (chance %.1f%% over %d)\n",
              100.0 * hit_rate, 100.0 * chance, scored);
  std::printf("    red to move, as-is    : %5.1f%%   (%d positions)\n",
              100.0 * hit_direct, direct.total);
  std::printf("    blue to move, mirrored: %5.1f%%   (%d positions)\n",
              100.0 * hit_transposed, transposed.total);
  std::printf("  network evaluations     : %lld\n", evaluator.evaluations());

  std::vector<std::string> failures;
  if (hit_rate < chance + min_margin) {
    char buffer[160];
    std::snprintf(buffer, sizeof(buffer),
                  "hit rate %.1f%% is below chance plus %.0f%% (%.1f%%)",
                  100.0 * hit_rate, 100.0 * min_margin,
                  100.0 * (chance + min_margin));
    failures.emplace_back(buffer);
  }
  const double gap = hit_direct > hit_transposed ? hit_direct - hit_transposed
                                                 : hit_transposed - hit_direct;
  if (gap > max_gap) {
    char buffer[200];
    std::snprintf(buffer, sizeof(buffer),
                  "red and blue hit rates differ by %.1f%% (%.1f%% vs %.1f%%); "
                  "the canonical action mapping is suspect",
                  100.0 * gap, 100.0 * hit_direct, 100.0 * hit_transposed);
    failures.emplace_back(buffer);
  }

  if (!failures.empty()) {
    std::printf("\nSEARCH GATE FAILED\n");
    for (const std::string& failure : failures)
      std::printf("  %s\n", failure.c_str());
    return 1;
  }
  std::printf("\nsearch gate passed\n");
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  std::string fixture_path = "training/gate_fixture.txt";
  std::string model;
  int simulations = 200;
  float c_puct = 1.5f;
  double min_margin = 0.25;
  double max_gap = 0.20;
  std::string value;

  const auto flag = [&value](const char* arg, const char* name) {
    const std::size_t len = std::strlen(name);
    if (std::strncmp(arg, name, len) != 0 || arg[len] != '=') return false;
    value = arg + len + 1;
    return true;
  };

  for (int i = 1; i < argc; ++i) {
    if (flag(argv[i], "--fixture")) {
      fixture_path = value;
    } else if (flag(argv[i], "--model")) {
      model = value;
    } else if (flag(argv[i], "--sims")) {
      simulations = std::atoi(value.c_str());
    } else if (flag(argv[i], "--cpuct")) {
      c_puct = std::strtof(value.c_str(), nullptr);
    } else if (flag(argv[i], "--min-margin")) {
      min_margin = std::strtod(value.c_str(), nullptr);
    } else if (flag(argv[i], "--max-gap")) {
      max_gap = std::strtod(value.c_str(), nullptr);
    } else {
      std::printf("unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  if (model.empty()) {
    std::printf("supply --model=<path.onnx>\n");
    return 1;
  }

  Fixture fixture;
  if (!LoadFixture(fixture_path, fixture)) return 1;

  switch (fixture.size) {
    case 5:
      return Run<5>(fixture, model, simulations, c_puct, min_margin, max_gap);
    case 7:
      return Run<7>(fixture, model, simulations, c_puct, min_margin, max_gap);
    case 9:
      return Run<9>(fixture, model, simulations, c_puct, min_margin, max_gap);
    default:
      std::printf("unsupported fixture board size %d\n", fixture.size);
      return 1;
  }
}
