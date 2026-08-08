#ifndef HEX_POLICY_DECODE_HPP
#define HEX_POLICY_DECODE_HPP

#include <algorithm>
#include <array>
#include <cmath>

#include "hex/board.hpp"
#include "hex/encoding.hpp"
#include "hex/puct.hpp"

namespace hex {

// Turns raw network logits into priors the search can use.
//
// Extracted from OnnxEvaluator so the WebAssembly build shares it rather than
// reimplementing it. The browser runs inference in JavaScript through ONNX Runtime
// Web, so something has to map logits back to board actions on the other side of
// that call -- and if that something were JavaScript, the canonical action mapping
// would exist twice. It exists once, here, and tests/test_encoding.cpp already pins
// the mapping it depends on.
//
// Masking happens before the softmax rather than after. Zeroing illegal actions
// afterwards would leave their mass in the normaliser, so the legal moves would sum
// to less than one and every prior would be quietly scaled down.
template <int N>
void DecodePolicyLogits(const Board<N>& board, const float* logits,
                        Evaluation<N>& out) {
  using Enc = Encoder<N>;
  out.priors.fill(0.0f);
  const bool transpose = Enc::NeedsTranspose(board);

  // Fixed array rather than a vector: this runs once per evaluated position, which
  // is the hottest path in the search, and the action count has a compile-time
  // bound.
  std::array<int, N * N + 1> actions{};
  int count = 0;
  for (int i = 0; i < board.NumEmpty(); ++i)
    actions[static_cast<std::size_t>(count++)] = board.LegalMoves()[i];
  if (board.CanSwap()) actions[static_cast<std::size_t>(count++)] = Board<N>::kSwapMove;
  if (count == 0) return;

  // Subtract the max before exponentiating, so a confident network cannot overflow
  // the exponential.
  float best = -1e30f;
  for (int i = 0; i < count; ++i)
    best = std::max(
        best, logits[Enc::Canonicalise(transpose,
                                       actions[static_cast<std::size_t>(i)])]);

  float total = 0.0f;
  for (int i = 0; i < count; ++i) {
    const int action = actions[static_cast<std::size_t>(i)];
    const float weight =
        std::exp(logits[Enc::Canonicalise(transpose, action)] - best);
    out.priors[static_cast<std::size_t>(action)] = weight;
    total += weight;
  }
  if (total > 0.0f)
    for (int i = 0; i < count; ++i)
      out.priors[static_cast<std::size_t>(
          actions[static_cast<std::size_t>(i)])] /= total;
}

}  // namespace hex

#endif  // HEX_POLICY_DECODE_HPP
