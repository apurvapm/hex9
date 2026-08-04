#ifndef HEX_BOARD_HPP
#define HEX_BOARD_HPP

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

namespace hex {

// Red connects the top edge (row 0) to the bottom edge (row N-1).
// Blue connects the left edge (col 0) to the right edge (col N-1).
enum class Player : std::uint8_t { kRed = 0, kBlue = 1 };
enum class Cell : std::uint8_t { kEmpty = 0, kRed = 1, kBlue = 2 };

constexpr Player Opponent(Player p) {
  return p == Player::kRed ? Player::kBlue : Player::kRed;
}

constexpr Cell ToCell(Player p) {
  return p == Player::kRed ? Cell::kRed : Cell::kBlue;
}

namespace detail {

constexpr std::uint64_t SplitMix64(std::uint64_t& state) {
  state += 0x9E3779B97F4A7C15ULL;
  std::uint64_t z = state;
  z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
  z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
  return z ^ (z >> 31);
}

// The six axial neighbours of a cell on a rhombic Hex board.
constexpr std::array<std::pair<int, int>, 6> kNeighbourOffsets = {{
    {-1, 0}, {-1, +1}, {0, -1}, {0, +1}, {+1, -1}, {+1, 0},
}};

// Union-find without path compression so that unions can be rolled back.
// Union by size keeps find() at O(log n), which is ample here.
template <int kNodes>
class RollbackDsu {
 public:
  RollbackDsu() { Reset(); }

  void Reset() {
    for (int i = 0; i < kNodes; ++i) {
      parent_[i] = i;
      size_[i] = 1;
    }
    merges_.clear();
  }

  int Find(int x) const {
    while (parent_[x] != x) x = parent_[x];
    return x;
  }

  void Unite(int a, int b) {
    a = Find(a);
    b = Find(b);
    if (a == b) return;
    if (size_[a] < size_[b]) std::swap(a, b);
    parent_[b] = a;
    size_[a] += size_[b];
    merges_.push_back(b);
  }

  std::size_t Mark() const { return merges_.size(); }

  void RollbackTo(std::size_t mark) {
    while (merges_.size() > mark) {
      const int child = merges_.back();
      merges_.pop_back();
      const int root = parent_[child];
      size_[root] -= size_[child];
      parent_[child] = child;
    }
  }

 private:
  std::array<int, kNodes> parent_{};
  std::array<int, kNodes> size_{};
  std::vector<int> merges_;
};

}  // namespace detail

// Zobrist keys are generated from a fixed seed so that hashes are identical
// across runs, machines, and the native/WASM builds.
template <int N>
struct Zobrist {
  static constexpr int kCells = N * N;

  static const std::array<std::array<std::uint64_t, N * N>, 2>& Keys() {
    static const auto keys = [] {
      std::array<std::array<std::uint64_t, N * N>, 2> k{};
      std::uint64_t state = 0x1234567890ABCDEFULL ^ static_cast<std::uint64_t>(N);
      for (int player = 0; player < 2; ++player)
        for (int i = 0; i < kCells; ++i) k[player][i] = detail::SplitMix64(state);
      return k;
    }();
    return keys;
  }
};

template <int N>
class Board {
 public:
  static constexpr int kSize = N;
  static constexpr int kCells = N * N;
  static constexpr int kTop = kCells + 0;
  static constexpr int kBottom = kCells + 1;
  static constexpr int kLeft = kCells + 2;
  static constexpr int kRight = kCells + 3;

  // The swap (pie) rule. Blue may answer Red's opening by taking that stone
  // instead of replying. Encoded as a sentinel one past the last cell, so the
  // policy head is kCells + 1 wide and swap is a normal action to the network.
  static constexpr int kSwapMove = kCells;

  Board() { Reset(); }

  static constexpr int Index(int row, int col) { return row * N + col; }
  static constexpr int Row(int idx) { return idx / N; }
  static constexpr int Col(int idx) { return idx % N; }

  void Reset() {
    cells_.fill(Cell::kEmpty);
    dsu_.Reset();
    hash_ = 0;
    to_play_ = Player::kRed;
    winner_ = Cell::kEmpty;
    ply_ = 0;
    num_empty_ = kCells;
    for (int i = 0; i < kCells; ++i) {
      empty_[i] = static_cast<std::uint8_t>(i);
      empty_slot_[i] = static_cast<std::uint8_t>(i);
    }
    history_.clear();
  }

  Cell At(int idx) const { return cells_[idx]; }
  Cell At(int row, int col) const { return cells_[Index(row, col)]; }
  Player ToPlay() const { return to_play_; }
  std::uint64_t Hash() const { return hash_; }
  bool IsTerminal() const { return winner_ != Cell::kEmpty; }
  Cell Winner() const { return winner_; }
  int NumEmpty() const { return num_empty_; }
  int MoveCount() const { return ply_; }

  // Blue's swap is available only as the reply to Red's opening.
  bool CanSwap() const { return ply_ == 1; }

  // The legal moves are exactly the empty cells, in an unspecified order.
  const std::uint8_t* LegalMoves() const { return empty_.data(); }

  void Play(int idx) { PlaceStone(idx, to_play_); }

  // Places a stone for an explicit player, ignoring whose turn it is.
  // Used by tests and by position setup; Play() is the normal entry point.
  void PlaceStone(int idx, Player player) {
    assert(idx >= 0 && idx < kCells);
    assert(cells_[idx] == Cell::kEmpty);

    history_.push_back(UndoRecord{static_cast<std::uint8_t>(idx),
                                  empty_slot_[idx], winner_, to_play_,
                                  dsu_.Mark(), false, 0});

    cells_[idx] = ToCell(player);
    hash_ ^= Zobrist<N>::Keys()[static_cast<int>(player)][idx];
    RemoveFromEmpty(idx);

    const int row = Row(idx);
    const int col = Col(idx);
    const Cell mine = ToCell(player);

    for (const auto& [dr, dc] : detail::kNeighbourOffsets) {
      const int nr = row + dr;
      const int nc = col + dc;
      if (nr < 0 || nr >= N || nc < 0 || nc >= N) continue;
      const int nidx = Index(nr, nc);
      if (cells_[nidx] == mine) dsu_.Unite(idx, nidx);
    }

    if (player == Player::kRed) {
      if (row == 0) dsu_.Unite(idx, kTop);
      if (row == N - 1) dsu_.Unite(idx, kBottom);
      if (dsu_.Find(kTop) == dsu_.Find(kBottom)) winner_ = Cell::kRed;
    } else {
      if (col == 0) dsu_.Unite(idx, kLeft);
      if (col == N - 1) dsu_.Unite(idx, kRight);
      if (dsu_.Find(kLeft) == dsu_.Find(kRight)) winner_ = Cell::kBlue;
    }

    to_play_ = Opponent(player);
    ++ply_;
  }

  // Blue takes over Red's opening stone. Transposing (r, c) -> (c, r) maps
  // Red's top-bottom goal onto Blue's left-right goal, so transpose plus
  // recolour is an isomorphism of the game: the swapped position is exactly
  // the mirror of the original with the roles exchanged.
  void PlaySwap() {
    assert(CanSwap());
    const int original = history_.back().idx;
    Undo();
    const int mirrored = Index(Col(original), Row(original));
    PlaceStone(mirrored, Player::kBlue);
    history_.back().was_swap = true;
    history_.back().swap_from = static_cast<std::uint8_t>(original);
    ply_ = 2;
  }

  void Undo() {
    assert(ply_ > 0 && !history_.empty());
    const UndoRecord rec = history_.back();
    history_.pop_back();

    dsu_.RollbackTo(rec.dsu_mark);
    winner_ = rec.winner;
    to_play_ = rec.to_play;

    const int idx = rec.idx;
    hash_ ^= Zobrist<N>::Keys()[static_cast<int>(cells_[idx] == Cell::kRed
                                                     ? Player::kRed
                                                     : Player::kBlue)][idx];
    cells_[idx] = Cell::kEmpty;
    RestoreToEmpty(idx, rec.empty_slot);
    --ply_;

    if (rec.was_swap) {
      // Put Red's opening stone back. PlaceStone pushes its own record and
      // bumps ply_, so pin ply_ to 1: the position before Blue chose to swap.
      PlaceStone(rec.swap_from, Player::kRed);
      ply_ = 1;
    }
  }

 private:
  struct UndoRecord {
    std::uint8_t idx;
    std::uint8_t empty_slot;
    Cell winner;
    Player to_play;
    std::size_t dsu_mark;
    bool was_swap;
    std::uint8_t swap_from;
  };

  // Swap-removal that parks the removed cell at the end of the live region,
  // so undoing is a single swap back.
  void RemoveFromEmpty(int idx) {
    const int slot = empty_slot_[idx];
    const int last = num_empty_ - 1;
    const int moved = empty_[last];
    empty_[slot] = static_cast<std::uint8_t>(moved);
    empty_slot_[moved] = static_cast<std::uint8_t>(slot);
    empty_[last] = static_cast<std::uint8_t>(idx);
    empty_slot_[idx] = static_cast<std::uint8_t>(last);
    num_empty_ = last;
  }

  void RestoreToEmpty(int idx, int slot) {
    const int last = num_empty_;
    num_empty_ = last + 1;
    const int moved = empty_[slot];
    empty_[slot] = static_cast<std::uint8_t>(idx);
    empty_slot_[idx] = static_cast<std::uint8_t>(slot);
    empty_[last] = static_cast<std::uint8_t>(moved);
    empty_slot_[moved] = static_cast<std::uint8_t>(last);
  }

  std::array<Cell, kCells> cells_{};
  detail::RollbackDsu<kCells + 4> dsu_;
  std::array<std::uint8_t, kCells> empty_{};
  std::array<std::uint8_t, kCells> empty_slot_{};
  int num_empty_ = kCells;
  std::uint64_t hash_ = 0;
  Player to_play_ = Player::kRed;
  Cell winner_ = Cell::kEmpty;
  int ply_ = 0;
  std::vector<UndoRecord> history_;
};

}  // namespace hex

#endif  // HEX_BOARD_HPP
