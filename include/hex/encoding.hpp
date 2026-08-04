#ifndef HEX_ENCODING_HPP
#define HEX_ENCODING_HPP

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "hex/board.hpp"

namespace hex {

// Board to network input, and back again for the policy.
//
// Hex is not symmetric between players the way Go or chess are: Red connects
// top to bottom, Blue connects left to right, so a position cannot be
// normalised by simply swapping colours. But transposing (r, c) -> (c, r) maps
// Red's goal onto Blue's, so transpose-plus-recolour *is* an isomorphism — the
// same one the swap rule is built on.
//
// The encoder uses it to canonicalise: the position is always presented from
// the perspective of the player to move, oriented so that player connects top
// to bottom. The network therefore only ever learns one goal direction, which
// halves the effective input space and means a position and its mirror share
// their evaluation exactly rather than approximately.
//
// Planes:
//   0  stones belonging to the player to move
//   1  stones belonging to the opponent
//   2  all ones when the swap action is legal, all zeros otherwise
//
// Plane 2 is not redundant. After a swap the board holds exactly one stone at
// ply 2, which looks identical in planes 0 and 1 to the one-stone position at
// ply 1 where swap *is* still available. Stone count alone cannot separate them.
template <int N>
struct Encoder {
  static constexpr int kPlanes = 3;
  static constexpr int kCells = N * N;
  static constexpr int kInputSize = kPlanes * kCells;
  static constexpr int kPolicySize = kCells + 1;  // cells plus the swap action

  // Blue to move means the board must be transposed to put the player to move
  // on the top-bottom axis.
  static bool NeedsTranspose(const Board<N>& board) {
    return board.ToPlay() == Player::kBlue;
  }

  // Transposing is its own inverse, so this maps board indices to canonical
  // indices and back. The swap sentinel is unaffected.
  static int Canonicalise(bool transpose, int move) {
    if (!transpose || move == Board<N>::kSwapMove) return move;
    return Board<N>::Index(Board<N>::Col(move), Board<N>::Row(move));
  }

  static int ToCanonical(const Board<N>& board, int move) {
    return Canonicalise(NeedsTranspose(board), move);
  }

  static int FromCanonical(const Board<N>& board, int index) {
    return Canonicalise(NeedsTranspose(board), index);
  }

  // Writes kInputSize floats.
  static void Encode(const Board<N>& board, float* out) {
    std::fill(out, out + kInputSize, 0.0f);
    const bool transpose = NeedsTranspose(board);
    const Cell mine = ToCell(board.ToPlay());

    for (int row = 0; row < N; ++row) {
      for (int col = 0; col < N; ++col) {
        const Cell cell = board.At(row, col);
        if (cell == Cell::kEmpty) continue;
        const int destination =
            transpose ? Board<N>::Index(col, row) : Board<N>::Index(row, col);
        const int plane = cell == mine ? 0 : 1;
        out[plane * kCells + destination] = 1.0f;
      }
    }

    if (board.CanSwap())
      std::fill(out + 2 * kCells, out + 3 * kCells, 1.0f);
  }

  // Normalised visit counts in canonical action space: the policy target.
  // Writes kPolicySize floats.
  static void EncodePolicyTarget(
      const Board<N>& board,
      const std::vector<std::pair<int, int>>& visits, float* out) {
    std::fill(out, out + kPolicySize, 0.0f);
    const bool transpose = NeedsTranspose(board);

    double total = 0.0;
    for (const auto& [move, count] : visits) total += count;
    if (total <= 0.0) return;

    for (const auto& [move, count] : visits)
      out[Canonicalise(transpose, move)] =
          static_cast<float>(count / total);
  }

  // Mask of actions that are legal from this position, in canonical space.
  // Writes kPolicySize floats.
  static void EncodeLegalMask(const Board<N>& board, float* out) {
    std::fill(out, out + kPolicySize, 0.0f);
    const bool transpose = NeedsTranspose(board);
    for (int i = 0; i < board.NumEmpty(); ++i)
      out[Canonicalise(transpose, board.LegalMoves()[i])] = 1.0f;
    if (board.CanSwap()) out[Board<N>::kSwapMove] = 1.0f;
  }
};

}  // namespace hex

#endif  // HEX_ENCODING_HPP
