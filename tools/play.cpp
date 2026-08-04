// Terminal front end for playing against either engine.
//
//   ./play --opponent=alphabeta --depth=4
//   ./play --opponent=mcts --sims=20000 --colour=blue
//   ./play --size=7 --swap
//
// Commands during a game: a move in Hex notation such as "e5", plus
// "swap", "hint", "undo", "quit".

#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
#include "hex/mcts.hpp"

namespace {

struct Options {
  int size = 9;
  std::string opponent = "mcts";
  int simulations = 10000;
  int depth = 4;
  std::string human_colour = "red";
  bool allow_swap = false;
  std::uint64_t seed = 1;
};

// Hex notation: column letter, then one-based row. "e5" is (row 4, col 4).
template <int N>
int ParseMove(const std::string& token) {
  if (token.size() < 2) return -1;
  const char file = static_cast<char>(std::tolower(token[0]));
  if (file < 'a' || file >= 'a' + N) return -1;
  const int col = file - 'a';

  int row = 0;
  for (std::size_t i = 1; i < token.size(); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(token[i]))) return -1;
    row = row * 10 + (token[i] - '0');
  }
  if (row < 1 || row > N) return -1;
  return hex::Board<N>::Index(row - 1, col);
}

template <int N>
std::string MoveName(int move) {
  if (move == hex::Board<N>::kSwapMove) return "swap";
  const int row = hex::Board<N>::Row(move);
  const int col = hex::Board<N>::Col(move);
  return std::string(1, static_cast<char>('a' + col)) + std::to_string(row + 1);
}

// Each row is indented one space further than the last, which is what gives the
// board its rhombus shape and makes the six-neighbour adjacency visible.
template <int N>
void Render(const hex::Board<N>& board) {
  std::string header = "    ";
  for (int c = 0; c < N; ++c) {
    header += ' ';
    header += static_cast<char>('a' + c);
  }
  std::printf("\n%s\n", header.c_str());

  for (int row = 0; row < N; ++row) {
    std::string line(static_cast<std::size_t>(row), ' ');
    char buf[16];
    std::snprintf(buf, sizeof buf, "%3d ", row + 1);
    line += buf;
    for (int col = 0; col < N; ++col) {
      const hex::Cell cell = board.At(row, col);
      line += ' ';
      line += cell == hex::Cell::kEmpty ? '.'
              : cell == hex::Cell::kRed ? 'R'
                                        : 'B';
    }
    std::snprintf(buf, sizeof buf, " %d", row + 1);
    line += buf;
    std::printf("%s\n", line.c_str());
  }

  // The last row is indented N-1, plus four for the row-number gutter.
  std::string footer(static_cast<std::size_t>(N) + 3, ' ');
  for (int c = 0; c < N; ++c) {
    footer += ' ';
    footer += static_cast<char>('a' + c);
  }
  std::printf("%s\n", footer.c_str());

  // Stones each side still needs to link its two edges. A horizontal chain is
  // easy to mistake for a win, so make the actual progress explicit.
  std::printf("     R top-bottom: %d to go     B left-right: %d to go\n\n",
              hex::AlphaBeta<N>::ConnectionDistance(board, hex::Player::kRed),
              hex::AlphaBeta<N>::ConnectionDistance(board, hex::Player::kBlue));
}

template <int N>
void PrintTopMoves(const hex::Mcts<N>& mcts, int count) {
  std::vector<typename hex::Mcts<N>::MoveStat> stats = mcts.RootStats();
  std::sort(stats.begin(), stats.end(),
            [](const auto& a, const auto& b) { return a.visits > b.visits; });
  std::printf("  %-6s %8s %8s\n", "move", "visits", "value");
  for (int i = 0; i < count && i < static_cast<int>(stats.size()); ++i)
    std::printf("  %-6s %8d %8.3f\n", MoveName<N>(stats[i].move).c_str(),
                stats[i].visits, stats[i].value);
}

template <int N>
int Run(const Options& options) {
  hex::Board<N> board;

  hex::Mcts<N> mcts(typename hex::Mcts<N>::Config{
      options.simulations, 1.4142135623730951, options.seed,
      options.allow_swap});
  hex::AlphaBeta<N> engine(typename hex::AlphaBeta<N>::Config{
      options.depth, 20, options.allow_swap});

  const bool human_is_red = options.human_colour != "blue";
  const bool use_mcts = options.opponent != "alphabeta";

  std::printf("Hex %dx%d.  R connects top to bottom.  B connects left to right.\n",
              N, N);
  std::printf("You are %s. Opponent: %s", human_is_red ? "R" : "B",
              use_mcts ? "MCTS" : "alpha-beta");
  if (use_mcts) {
    std::printf(" (%d simulations)\n", options.simulations);
  } else {
    std::printf(" (depth %d)\n", options.depth);
  }
  std::printf("Swap rule: %s.  Commands: <move> | swap | hint | undo | quit\n",
              options.allow_swap ? "on" : "off");

  while (!board.IsTerminal()) {
    Render(board);

    const bool red_to_play = board.ToPlay() == hex::Player::kRed;
    const bool human_turn = red_to_play == human_is_red;

    if (!human_turn) {
      int move;
      if (use_mcts) {
        move = mcts.Search(board);
        std::printf("Engine plays %s\n", MoveName<N>(move).c_str());
      } else {
        move = engine.Search(board);
        std::printf("Engine plays %s  (score %d, %lld nodes)\n",
                    MoveName<N>(move).c_str(), engine.LastScore(),
                    engine.Nodes());
      }
      if (move == hex::Board<N>::kSwapMove) {
        board.PlaySwap();
      } else {
        board.Play(move);
      }
      continue;
    }

    std::printf("Your move (%s)%s: ", red_to_play ? "R" : "B",
                board.CanSwap() && options.allow_swap ? " [swap available]" : "");
    std::fflush(stdout);

    std::string token;
    if (!(std::cin >> token)) {
      std::printf("\nInput closed.\n");
      return 0;
    }

    if (token == "quit" || token == "q") return 0;

    if (token == "hint") {
      hex::Mcts<N> advisor(typename hex::Mcts<N>::Config{
          options.simulations, 1.4142135623730951, options.seed + 1,
          options.allow_swap});
      advisor.Search(board);
      PrintTopMoves(advisor, 5);
      continue;
    }

    if (token == "undo") {
      // Take back a full turn so the human is on move again.
      if (board.MoveCount() < 2) {
        std::printf("Nothing to undo.\n");
      } else {
        board.Undo();
        board.Undo();
      }
      continue;
    }

    if (token == "swap") {
      if (options.allow_swap && board.CanSwap()) {
        board.PlaySwap();
      } else {
        std::printf("Swap is not available here.\n");
      }
      continue;
    }

    const int move = ParseMove<N>(token);
    if (move < 0) {
      std::printf("Could not read '%s'. Use a letter then a number, like e5.\n",
                  token.c_str());
      continue;
    }
    if (board.At(move) != hex::Cell::kEmpty) {
      std::printf("%s is already occupied.\n", token.c_str());
      continue;
    }
    board.Play(move);
  }

  Render(board);
  const bool red_won = board.Winner() == hex::Cell::kRed;
  std::printf("%s wins in %d moves. %s\n", red_won ? "R" : "B",
              board.MoveCount(),
              red_won == human_is_red ? "You win." : "Engine wins.");
  return 0;
}

bool MatchFlag(const char* arg, const char* name, std::string& value) {
  const std::size_t len = std::strlen(name);
  if (std::strncmp(arg, name, len) != 0) return false;
  if (arg[len] != '=') return false;
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
    } else if (MatchFlag(argv[i], "--opponent", value)) {
      options.opponent = value;
    } else if (MatchFlag(argv[i], "--sims", value)) {
      options.simulations = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--depth", value)) {
      options.depth = std::atoi(value.c_str());
    } else if (MatchFlag(argv[i], "--colour", value) ||
               MatchFlag(argv[i], "--color", value)) {
      options.human_colour = value;
    } else if (MatchFlag(argv[i], "--seed", value)) {
      options.seed = std::strtoull(value.c_str(), nullptr, 10);
    } else if (std::strcmp(argv[i], "--swap") == 0) {
      options.allow_swap = true;
    } else {
      std::printf("Unknown option: %s\n", argv[i]);
      return 1;
    }
  }

  // The board size is a template parameter, so it is dispatched here rather
  // than being a runtime value.
  switch (options.size) {
    case 5: return Run<5>(options);
    case 7: return Run<7>(options);
    case 9: return Run<9>(options);
    case 11: return Run<11>(options);
    default:
      std::printf("Supported sizes are 5, 7, 9, 11.\n");
      return 1;
  }
}
