#ifndef HEX_ALPHABETA_HPP
#define HEX_ALPHABETA_HPP

#include <algorithm>
#include <array>
#include <cstdint>
#include <deque>
#include <limits>
#include <vector>

#include "hex/board.hpp"

namespace hex {

// Fixed-strength reference opponent: negamax with alpha-beta pruning, a
// transposition table, and iterative deepening. Its purpose is to be a yardstick
// that never changes, so that agents from any later phase can be compared
// against a constant.
//
// Hex has no material to count, so the evaluation is structural: the minimum
// number of additional stones a player needs to complete a connection. Own
// stones cost nothing to traverse, empty cells cost one, and opponent stones
// block. Because the only edge weights are 0 and 1, that shortest path is a
// deque-based 0-1 BFS rather than a full Dijkstra — O(N^2) with a tiny constant.
//
// This evaluation is deliberately simple. It does not understand bridges (two
// cells that guarantee a connection the opponent cannot sever), so it
// undervalues positions a strong Hex player would recognise as already won.
// Shannon's electrical-resistance evaluation and the two-distance metric both
// capture that structure and are the natural upgrades.
template <int N>
class AlphaBeta {
 public:
  static constexpr int kWinScore = 100000;
  static constexpr int kInfinity = 1000000;

  struct Config {
    int max_depth = 4;
    // 2^20 entries is roughly 25 MB and comfortably covers a 9x9 search.
    std::size_t table_bits = 20;
    bool allow_swap = true;
  };

  explicit AlphaBeta(Config config = {})
      : config_(config), table_(std::size_t{1} << config.table_bits) {}

  // Returns the best move for the player to move. The board is restored exactly.
  int Search(Board<N>& board) {
    nodes_ = 0;
    best_move_ = -1;
    score_ = 0;

    // Iterative deepening: each pass seeds the transposition table with a best
    // move for the next, which is what makes the ordering good enough for
    // alpha-beta to prune effectively.
    for (int depth = 1; depth <= config_.max_depth; ++depth) {
      const int score = Negamax(board, depth, -kInfinity, kInfinity);
      const int move = ProbeMove(board.Hash());
      if (move >= 0) {
        best_move_ = move;
        score_ = score;
      }
      // A forced result will not change with more depth.
      if (score >= kWinScore / 2 || score <= -kWinScore / 2) break;
    }

    if (best_move_ < 0) {
      const std::vector<int> moves = LegalMoves(board);
      best_move_ = moves.empty() ? -1 : moves.front();
    }
    return best_move_;
  }

  int LastScore() const { return score_; }
  long long Nodes() const { return nodes_; }

  // Exposed for testing: minimum stones the player still needs to connect.
  static int ConnectionDistance(const Board<N>& board, Player player) {
    constexpr int kUnreachable = std::numeric_limits<int>::max() / 4;
    const Cell mine = ToCell(player);
    const bool vertical = player == Player::kRed;

    std::array<int, N * N> dist;
    dist.fill(kUnreachable);
    std::deque<int> queue;

    // Seed every cell on the player's source edge that is not blocked.
    for (int i = 0; i < N; ++i) {
      const int idx = vertical ? Board<N>::Index(0, i) : Board<N>::Index(i, 0);
      const Cell cell = board.At(idx);
      if (cell != Cell::kEmpty && cell != mine) continue;
      const int cost = cell == mine ? 0 : 1;
      if (cost < dist[idx]) {
        dist[idx] = cost;
        if (cost == 0) {
          queue.push_front(idx);
        } else {
          queue.push_back(idx);
        }
      }
    }

    while (!queue.empty()) {
      const int u = queue.front();
      queue.pop_front();
      const int row = Board<N>::Row(u);
      const int col = Board<N>::Col(u);

      for (const auto& [dr, dc] : detail::kNeighbourOffsets) {
        const int nr = row + dr;
        const int nc = col + dc;
        if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
        const int v = Board<N>::Index(nr, nc);
        const Cell cell = board.At(v);
        if (cell != Cell::kEmpty && cell != mine) continue;

        const int weight = cell == mine ? 0 : 1;
        if (dist[u] + weight < dist[v]) {
          dist[v] = dist[u] + weight;
          if (weight == 0) {
            queue.push_front(v);
          } else {
            queue.push_back(v);
          }
        }
      }
    }

    int best = kUnreachable;
    for (int i = 0; i < N; ++i) {
      const int idx =
          vertical ? Board<N>::Index(N - 1, i) : Board<N>::Index(i, N - 1);
      best = std::min(best, dist[idx]);
    }
    return best;
  }

 private:
  enum class Bound : std::uint8_t { kExact, kLower, kUpper };

  struct Entry {
    std::uint64_t key = 0;
    int score = 0;
    int move = -1;
    std::int16_t depth = -1;
    Bound bound = Bound::kExact;
  };

  std::vector<int> LegalMoves(const Board<N>& board) const {
    std::vector<int> moves(board.LegalMoves(),
                           board.LegalMoves() + board.NumEmpty());
    if (config_.allow_swap && board.CanSwap())
      moves.push_back(Board<N>::kSwapMove);
    return moves;
  }

  void ApplyMove(Board<N>& board, int move) const {
    if (move == Board<N>::kSwapMove) {
      board.PlaySwap();
    } else {
      board.Play(move);
    }
  }

  // Cells nearer the centre are searched first. In Hex the centre touches more
  // potential connection paths, so this is a reasonable static ordering when the
  // transposition table has nothing better to offer.
  static const std::array<int, N * N>& CentreOrder() {
    static const auto order = [] {
      std::array<int, N * N> cells{};
      for (int i = 0; i < N * N; ++i) cells[i] = i;
      const double mid = (N - 1) / 2.0;
      std::stable_sort(cells.begin(), cells.end(), [mid](int a, int b) {
        const double da = std::abs(Board<N>::Row(a) - mid) +
                          std::abs(Board<N>::Col(a) - mid);
        const double db = std::abs(Board<N>::Row(b) - mid) +
                          std::abs(Board<N>::Col(b) - mid);
        return da < db;
      });
      return cells;
    }();
    return order;
  }

  void OrderMoves(std::vector<int>& moves, int tt_move) const {
    const auto& centre = CentreOrder();
    std::array<int, N * N> rank{};
    for (int i = 0; i < N * N; ++i) rank[centre[i]] = i;

    std::stable_sort(moves.begin(), moves.end(), [&](int a, int b) {
      if (a == tt_move) return true;
      if (b == tt_move) return false;
      const int ra = a == Board<N>::kSwapMove ? N * N : rank[a];
      const int rb = b == Board<N>::kSwapMove ? N * N : rank[b];
      return ra < rb;
    });
  }

  Entry& Slot(std::uint64_t hash) {
    return table_[hash & (table_.size() - 1)];
  }

  int ProbeMove(std::uint64_t hash) {
    const Entry& e = Slot(hash);
    return e.key == hash ? e.move : -1;
  }

  // Evaluation from the perspective of the player to move.
  int Evaluate(const Board<N>& board) const {
    const Player me = board.ToPlay();
    return ConnectionDistance(board, Opponent(me)) -
           ConnectionDistance(board, me);
  }

  int Negamax(Board<N>& board, int depth, int alpha, int beta) {
    ++nodes_;

    if (board.IsTerminal()) {
      // The opponent completed a chain on the previous move, so the player to
      // move has already lost. Adding the ply count makes a distant loss less
      // bad than an immediate one, so the engine resists rather than collapses.
      return -kWinScore + board.MoveCount();
    }
    if (depth == 0) return Evaluate(board);

    const std::uint64_t hash = board.Hash();
    const int alpha_original = alpha;
    int tt_move = -1;

    {
      const Entry& e = Slot(hash);
      if (e.key == hash) {
        tt_move = e.move;
        if (e.depth >= depth) {
          if (e.bound == Bound::kExact) return e.score;
          if (e.bound == Bound::kLower) alpha = std::max(alpha, e.score);
          if (e.bound == Bound::kUpper) beta = std::min(beta, e.score);
          if (alpha >= beta) return e.score;
        }
      }
    }

    std::vector<int> moves = LegalMoves(board);
    OrderMoves(moves, tt_move);

    int best = -kInfinity;
    int best_move = moves.empty() ? -1 : moves.front();

    for (const int move : moves) {
      ApplyMove(board, move);
      const int score = -Negamax(board, depth - 1, -beta, -alpha);
      board.Undo();

      if (score > best) {
        best = score;
        best_move = move;
      }
      alpha = std::max(alpha, best);
      if (alpha >= beta) break;  // this branch cannot affect the result
    }

    Entry& slot = Slot(hash);
    // Depth-preferred replacement: a deeper result is more valuable than a
    // shallower one for the same slot.
    if (slot.key != hash || slot.depth <= depth) {
      slot.key = hash;
      slot.score = best;
      slot.move = best_move;
      slot.depth = static_cast<std::int16_t>(depth);
      slot.bound = best <= alpha_original ? Bound::kUpper
                   : best >= beta         ? Bound::kLower
                                          : Bound::kExact;
    }
    return best;
  }

  Config config_;
  std::vector<Entry> table_;
  long long nodes_ = 0;
  int best_move_ = -1;
  int score_ = 0;
};

}  // namespace hex

#endif  // HEX_ALPHABETA_HPP
