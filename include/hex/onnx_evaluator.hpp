#ifndef HEX_ONNX_EVALUATOR_HPP
#define HEX_ONNX_EVALUATOR_HPP

#include <algorithm>
#include <cmath>
#include <memory>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "hex/board.hpp"
#include "hex/encoding.hpp"
#include "hex/policy_decode.hpp"
#include "hex/puct.hpp"

namespace hex {

// Wraps an exported HexNet so it can be handed to Puct::Search as an evaluator.
//
// The network is trained on canonical inputs — always from the mover's
// perspective, rotated onto the top-bottom axis — so this class owns both
// directions of that mapping. It encodes through Encoder before inference and
// un-canonicalises the returned policy before handing priors back to the
// search, which never sees canonical space at all.
template <int N>
class OnnxEvaluator {
 public:
  using Enc = Encoder<N>;

  explicit OnnxEvaluator(const std::string& model_path, int intra_op_threads = 1)
      : environment_(ORT_LOGGING_LEVEL_WARNING, "hex9"),
        memory_info_(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator,
                                                OrtMemTypeDefault)) {
    Ort::SessionOptions options;
    options.SetIntraOpNumThreads(intra_op_threads);
    options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
    session_ = std::make_unique<Ort::Session>(environment_, model_path.c_str(),
                                              options);
    input_.resize(static_cast<std::size_t>(Enc::kInputSize));
  }

  // Single-position evaluation. Convenient, and the only mode the browser needs.
  void operator()(const Board<N>& board, Evaluation<N>& out) {
    Enc::Encode(board, input_.data());

    const std::array<std::int64_t, 4> shape = {1, Enc::kPlanes, N, N};
    Ort::Value tensor = Ort::Value::CreateTensor<float>(
        memory_info_, input_.data(), input_.size(), shape.data(), shape.size());

    const char* input_names[] = {"board"};
    const char* output_names[] = {"policy", "value"};
    auto outputs = session_->Run(Ort::RunOptions{nullptr}, input_names, &tensor,
                                 1, output_names, 2);

    const float* logits = outputs[0].GetTensorData<float>();
    const float* value = outputs[1].GetTensorData<float>();

    DecodePolicyLogits<N>(board, logits, out);
    out.value = std::clamp(value[0], -1.0f, 1.0f);
    ++evaluations_;
  }

  long long evaluations() const { return evaluations_; }

 private:
  Ort::Env environment_;
  Ort::MemoryInfo memory_info_;
  std::unique_ptr<Ort::Session> session_;
  std::vector<float> input_;
  long long evaluations_ = 0;
};

}  // namespace hex

#endif  // HEX_ONNX_EVALUATOR_HPP
