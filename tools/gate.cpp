// Writes the 5x5 ground-truth fixture that gates the training loop.
//
//   ./build/gate --out=training/gate_fixture.txt
//
// Two failure modes in a self-play trainer are invisible in a loss curve: a
// value target with the player-to-move sign inverted, and a policy target
// misaligned with the canonical encoding. Both train smoothly and both play
// like garbage. A position whose true value and true winning moves are known is
// the only defence, so this tool produces them.
//
// Full 5x5 is not exhaustively solvable with Solver. Measured cost rises about
// 12x per two stones removed, and the empty 4x4 board already costs 11.2M
// nodes, which puts an empty 5x5 board out of reach by several orders of
// magnitude. Sampled positions from 10 stones up each solve in milliseconds and
// are still mid-game -- 15 empty cells, both sides uncommitted -- so that is
// where the fixture lives. Ground truth on a reachable position is worth more
// than an unreachable claim about the whole game.
//
// Positions are sampled at ply 10 and above, so swap is never legal in the
// fixture. Swap encoding is covered by tests/test_encoding.cpp instead.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <unordered_set>
#include <vector>

#include "hex/board.hpp"
#include "hex/solver.hpp"

namespace {

constexpr int kSize = 5;

struct Options {
  // The default has to reproduce the committed training/gate_fixture.txt, since
  // CI regenerates it with no arguments and diffs the result.
  int cases = 300;
  int min_stones = 10;
  int max_stones = 18;
  // A won position where most legal moves happen to win tests nothing: the
  // policy scores well by guessing. Only won positions whose winning moves are
  // this fraction of the legal moves or fewer are kept, which is what makes the
  // policy metric discriminating. Lost positions are never filtered -- there is
  // no sharpness to speak of when every move loses.
  // Measured on 5x5 at 10-18 stones, the distribution is bimodal: about a third
  // of won positions have 90% or more of their legal moves winning, which drags
  // the policy metric's chance rate to 50% and makes it unreadable. Roughly
  // 44% sit at or below 0.30, so that is the cut. It also happens to balance
  // the two value classes, which random play otherwise skews 73/27.
  double max_win_fraction = 0.30;
  std::uint64_t seed = 20260804;
  std::string output = "training/gate_fixture.txt";
};

// mt19937 is used directly with a hand-rolled range reduction: the <random>
// distribution classes differ between libstdc++ and libc++, and this fixture
// must be reproducible across every host that builds the project.
int NextBelow(std::mt19937_64& rng, int bound) {
  return static_cast<int>(rng() % static_cast<std::uint64_t>(bound));
}

struct Case {
  std::vector<int> moves;
  int mover_value;
  std::vector<int> winning;
};

// Random legal play to a target stone count, rejecting any line that ends the
// game early. Rejection rather than repair: a truncated game would bias the
// sample towards positions one move from a win, which are the easy ones.
bool SamplePosition(std::mt19937_64& rng, int stones, hex::Board<kSize>& board,
                    std::vector<int>& moves) {
  board.Reset();
  moves.clear();
  for (int i = 0; i < stones; ++i) {
    const int move = board.LegalMoves()[NextBelow(rng, board.NumEmpty())];
    board.Play(move);
    moves.push_back(move);
    if (board.IsTerminal()) return false;
  }
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  Options options;
  std::string value;

  const auto flag = [&value](const char* arg, const char* name) {
    const std::size_t len = std::strlen(name);
    if (std::strncmp(arg, name, len) != 0 || arg[len] != '=') return false;
    value = arg + len + 1;
    return true;
  };

  for (int i = 1; i < argc; ++i) {
    if (flag(argv[i], "--cases")) {
      options.cases = std::atoi(value.c_str());
    } else if (flag(argv[i], "--min-stones")) {
      options.min_stones = std::atoi(value.c_str());
    } else if (flag(argv[i], "--max-stones")) {
      options.max_stones = std::atoi(value.c_str());
    } else if (flag(argv[i], "--max-win-fraction")) {
      options.max_win_fraction = std::strtod(value.c_str(), nullptr);
    } else if (flag(argv[i], "--seed")) {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (flag(argv[i], "--out")) {
      options.output = value;
    } else {
      std::printf("unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  if (options.min_stones < 1 || options.max_stones >= kSize * kSize ||
      options.min_stones > options.max_stones) {
    std::printf("stone counts must satisfy 1 <= min <= max < %d\n",
                kSize * kSize);
    return 1;
  }

  std::mt19937_64 rng(options.seed);
  hex::Board<kSize> board;
  std::vector<int> moves;
  std::unordered_set<std::uint64_t> seen;
  std::vector<Case> cases;

  long long total_nodes = 0;
  int rejected = 0;
  int too_easy = 0;
  std::array<int, 10> win_fraction_histogram{};

  while (static_cast<int>(cases.size()) < options.cases) {
    const int span = options.max_stones - options.min_stones + 1;
    const int stones = options.min_stones + NextBelow(rng, span);
    if (!SamplePosition(rng, stones, board, moves)) {
      ++rejected;
      continue;
    }
    // Deduplicate by position, not by move order: two different orderings of
    // the same stones are one data point, and scoring it twice would silently
    // weight it double.
    if (!seen.insert(board.Hash()).second) continue;

    hex::Solver<kSize> solver;
    Case entry;
    entry.mover_value = solver.Solve(board);
    entry.winning = solver.WinningMoves(board);
    entry.moves = moves;
    total_nodes += solver.nodes();

    // The two solver queries have to agree: a winning position must expose at
    // least one winning move, and a lost one must expose none. Hex admits no
    // draws, so there is no third case to allow for.
    const bool consistent = (entry.mover_value == 1) == !entry.winning.empty();
    if (!consistent) {
      std::printf("solver disagrees with itself at %d stones\n", stones);
      return 1;
    }

    if (entry.mover_value == 1) {
      const double fraction = static_cast<double>(entry.winning.size()) /
                              static_cast<double>(board.NumEmpty());
      const int bucket = std::min(9, static_cast<int>(fraction * 10.0));
      ++win_fraction_histogram[static_cast<std::size_t>(bucket)];
      if (fraction > options.max_win_fraction) {
        ++too_easy;
        continue;
      }
    }
    cases.push_back(std::move(entry));
  }

  std::FILE* file = std::fopen(options.output.c_str(), "w");
  if (file == nullptr) {
    std::printf("could not open %s for writing\n", options.output.c_str());
    return 1;
  }

  std::fprintf(file, "# hex9 %dx%d gate fixture\n", kSize, kSize);
  std::fprintf(file, "# board_size policy_size\n");
  std::fprintf(file, "%d %d\n", kSize, kSize * kSize + 1);
  std::fprintf(file,
               "# per case: moves..., -1, mover_value, count, winning moves...\n"
               "# moves and winning moves are board indices, not canonical\n");

  int won = 0;
  for (const Case& entry : cases) {
    for (const int move : entry.moves) std::fprintf(file, "%d ", move);
    std::fprintf(file, "-1 %d %d", entry.mover_value,
                 static_cast<int>(entry.winning.size()));
    for (const int move : entry.winning) std::fprintf(file, " %d", move);
    std::fprintf(file, "\n");
    if (entry.mover_value == 1) ++won;
  }
  std::fclose(file);

  std::printf("wrote %d solved positions to %s\n",
              static_cast<int>(cases.size()), options.output.c_str());
  std::printf("  stones      : %d to %d\n", options.min_stones,
              options.max_stones);
  std::printf("  mover wins  : %d/%d (%.0f%%)\n", won,
              static_cast<int>(cases.size()),
              100.0 * won / static_cast<double>(cases.size()));
  std::printf("  solver nodes: %lld total, %lld mean\n", total_nodes,
              total_nodes / static_cast<long long>(cases.size()));
  std::printf("  rejected    : %d ended early, %d above the win fraction\n",
              rejected, too_easy);

  // The distribution the policy metric's null hypothesis comes from. Printed
  // unconditionally because a hit rate is meaningless without it.
  std::printf("  winning moves as a fraction of legal, over solved wins:\n");
  int histogram_total = 0;
  for (const int count : win_fraction_histogram) histogram_total += count;
  for (std::size_t bucket = 0; bucket < win_fraction_histogram.size(); ++bucket) {
    if (win_fraction_histogram[bucket] == 0) continue;
    std::printf("    %3zu-%3zu%% : %5d (%.0f%%)\n", bucket * 10,
                bucket * 10 + 10, win_fraction_histogram[bucket],
                100.0 * win_fraction_histogram[bucket] / histogram_total);
  }
  return 0;
}
