// WebAssembly bindings for the 9x9 engine.
//
// The shape of this interface is forced by one fact: ONNX Runtime Web returns a
// Promise, and a synchronous C++ search cannot wait on one without Asyncify, which
// is both slow and invasive. So the search is resumable instead. JavaScript drives
// it:
//
//   beginSearch(...)
//   while (!searchComplete()) {
//     const n = prepareBatch(k);        // C++ descends k leaves
//     ...run the network over batchPlanes()...
//     submitBatch();                    // C++ expands and backs up
//     await yieldToEventLoop();         // keeps the UI responsive
//   }
//
// That loop is also the frame budget the design notes ask for: capping work per
// frame falls out of choosing k, with no scheduler to write.
//
// Everything that could drift lives on this side of the boundary. The canonical
// encoding, the legal-move mask, the softmax and the map from canonical action back
// to board cell are all C++, shared with the native build and pinned by the golden
// fixture. JavaScript only moves float arrays and draws pixels. A JavaScript
// reimplementation of the encoder would be a second implementation of the one thing
// this project spends its verification budget keeping single.

#include <algorithm>
#include <cstdint>
#include <memory>
#include <vector>

#include <emscripten/bind.h>
#include <emscripten/val.h>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
#include "hex/encoding.hpp"
#include "hex/policy_decode.hpp"
#include "hex/search_tree.hpp"

namespace {

constexpr int kSize = 9;
constexpr int kCells = kSize * kSize;
constexpr int kPolicy = kCells + 1;
constexpr int kPlanes = hex::Encoder<kSize>::kPlanes;
// Bounds the pre-allocated transfer buffers. Larger batches stop helping long before
// this on a CPU, and the buffers are cheap: 64 positions is 60 KB of planes.
constexpr int kMaxBatch = 64;

using Tree = hex::detail::SharedTree<kSize>;

class HexEngine {
 public:
  HexEngine()
      : planes_(static_cast<std::size_t>(kMaxBatch) * kPlanes * kCells, 0.0f),
        logits_(static_cast<std::size_t>(kMaxBatch) * kPolicy, 0.0f),
        values_(static_cast<std::size_t>(kMaxBatch), 0.0f) {}

  // --- game state -----------------------------------------------------------

  void Reset() {
    board_.Reset();
    tree_.reset();
    simulations_done_ = 0;
    pending_ = 0;
  }

  int Size() const { return kSize; }
  int PolicySize() const { return kPolicy; }
  int PlaneCount() const { return kPlanes; }
  int MaxBatch() const { return kMaxBatch; }

  int CellAt(int index) const {
    if (index < 0 || index >= kCells) return 0;
    return static_cast<int>(board_.At(index));
  }

  int ToPlay() const { return static_cast<int>(board_.ToPlay()); }
  bool IsTerminal() const { return board_.IsTerminal(); }
  int Winner() const { return static_cast<int>(board_.Winner()); }
  bool CanSwap() const { return board_.CanSwap(); }
  int MoveCount() const { return board_.MoveCount(); }
  int SwapAction() const { return hex::Board<kSize>::kSwapMove; }

  bool IsLegal(int action) const {
    if (board_.IsTerminal()) return false;
    if (action == hex::Board<kSize>::kSwapMove) return board_.CanSwap();
    if (action < 0 || action >= kCells) return false;
    return board_.At(action) == hex::Cell::kEmpty;
  }

  bool Play(int action) {
    if (!IsLegal(action)) return false;
    if (action == hex::Board<kSize>::kSwapMove) {
      board_.PlaySwap();
    } else {
      board_.Play(action);
    }
    // Any move invalidates the tree: it was built for a different position, and
    // reusing it silently would report the previous move's statistics.
    tree_.reset();
    simulations_done_ = 0;
    pending_ = 0;
    return true;
  }

  bool Undo() {
    if (board_.MoveCount() == 0) return false;
    board_.Undo();
    tree_.reset();
    simulations_done_ = 0;
    pending_ = 0;
    return true;
  }

  // Remaining stones each side needs to complete a connection. The CLI prints this
  // and the browser needs it for the same reason: a chain running left to right
  // looks like a win but is Blue's goal, and the indented rhombus makes that worse.
  // A rules paragraph does not fix it; a number under the board does.
  int ConnectionDistance(int player) const {
    const hex::Player who = player == 0 ? hex::Player::kRed : hex::Player::kBlue;
    return hex::AlphaBeta<kSize>::ConnectionDistance(board_, who);
  }

  emscripten::val LegalMoves() const {
    std::vector<int> moves(board_.LegalMoves(),
                           board_.LegalMoves() + board_.NumEmpty());
    if (board_.CanSwap()) moves.push_back(hex::Board<kSize>::kSwapMove);
    std::sort(moves.begin(), moves.end());
    return emscripten::val::array(moves);
  }

  // --- resumable search -----------------------------------------------------

  void BeginSearch(int simulations, float c_puct, int virtual_loss) {
    config_ = hex::ParallelSearchConfig{};
    config_.threads = 1;
    config_.simulations = std::max(1, simulations);
    config_.c_puct = c_puct;
    config_.allow_swap = true;
    config_.virtual_loss = std::max(0, virtual_loss);

    tree_ = std::make_unique<Tree>(config_);
    simulations_done_ = 0;
    pending_ = 0;
    root_needs_expanding_ = true;
  }

  bool SearchActive() const { return tree_ != nullptr; }
  bool SearchComplete() const {
    return tree_ == nullptr || simulations_done_ >= config_.simulations;
  }
  int SimulationsDone() const { return simulations_done_; }
  int SimulationBudget() const { return config_.simulations; }

  // Descends up to `max_leaves` times and returns how many positions need the
  // network. Terminal leaves are resolved here and excluded from the count: their
  // value is known exactly, so spending an inference on one would be waste.
  //
  // A return of zero does not mean the search is finished -- every descent in this
  // batch may have landed on a terminal node. Callers loop on searchComplete().
  int PrepareBatch(int max_leaves) {
    pending_ = 0;
    if (tree_ == nullptr || SearchComplete()) return 0;
    const int room = std::min(std::max(1, max_leaves), kMaxBatch);

    // The root is expanded on its own so the first descent has children to choose
    // between. It is the one position whose evaluation cannot be batched with
    // others, because everything else depends on it.
    if (root_needs_expanding_) {
      boards_[0] = board_;
      leaves_[0] = tree_->Root();
      applied_[0] = 0;
      pending_ = 1;
      root_needs_expanding_ = false;
      hex::Encoder<kSize>::Encode(boards_[0], planes_.data());
      return 1;
    }

    while (pending_ < room && simulations_done_ + pending_ < config_.simulations) {
      const std::size_t slot = static_cast<std::size_t>(pending_);
      boards_[slot] = board_;
      int applied = 0;
      const int leaf = tree_->Descend(boards_[slot], applied);

      if (boards_[slot].IsTerminal()) {
        // The opponent completed a chain on the previous move, so the player to
        // move here has already lost.
        tree_->Backup(boards_[slot], leaf, -1.0f, applied);
        ++simulations_done_;
        continue;
      }

      leaves_[slot] = leaf;
      applied_[slot] = applied;
      hex::Encoder<kSize>::Encode(
          boards_[slot],
          planes_.data() + slot * static_cast<std::size_t>(kPlanes) * kCells);
      ++pending_;
    }
    return pending_;
  }

  // Consumes whatever JavaScript wrote into batchLogits() and batchValues().
  void SubmitBatch() {
    if (tree_ == nullptr || pending_ == 0) return;
    for (int i = 0; i < pending_; ++i) {
      const std::size_t slot = static_cast<std::size_t>(i);
      hex::Evaluation<kSize> eval;
      hex::DecodePolicyLogits<kSize>(
          boards_[slot],
          logits_.data() + slot * static_cast<std::size_t>(kPolicy), eval);
      eval.value = std::clamp(values_[slot], -1.0f, 1.0f);

      tree_->Expand(leaves_[slot], boards_[slot], eval);
      tree_->Backup(boards_[slot], leaves_[slot], eval.value, applied_[slot]);
      ++simulations_done_;
    }
    pending_ = 0;
  }

  // Views into WebAssembly memory, valid until the heap grows. Every buffer is
  // allocated once in the constructor and never resized, so within a search loop
  // these stay valid -- but JavaScript should still re-acquire them each batch
  // rather than caching one across calls, because ALLOW_MEMORY_GROWTH means any
  // allocation anywhere can detach the old view.
  emscripten::val BatchPlanes() {
    return emscripten::val(emscripten::typed_memory_view(
        static_cast<std::size_t>(pending_) * kPlanes * kCells, planes_.data()));
  }
  emscripten::val BatchLogits() {
    return emscripten::val(emscripten::typed_memory_view(
        static_cast<std::size_t>(pending_) * kPolicy, logits_.data()));
  }
  emscripten::val BatchValues() {
    return emscripten::val(emscripten::typed_memory_view(
        static_cast<std::size_t>(pending_), values_.data()));
  }
  int PendingCount() const { return pending_; }

  // --- results --------------------------------------------------------------

  int BestMove() const {
    if (tree_ == nullptr) return -1;
    return tree_->Collect().best_move;
  }

  float RootValue() const {
    if (tree_ == nullptr) return 0.0f;
    return tree_->Collect().root_value;
  }

  // Sorted by visits, with the PUCT score each child would be selected on next.
  // Showing prior, visits, value and score together is what makes the search
  // legible: it is the difference between watching numbers move and seeing why.
  emscripten::val TopMoves(int how_many) const {
    emscripten::val out = emscripten::val::array();
    if (tree_ == nullptr) return out;

    std::vector<Tree::NodeView> children = tree_->RootChildren();
    std::sort(children.begin(), children.end(),
              [](const Tree::NodeView& a, const Tree::NodeView& b) {
                return a.visits > b.visits;
              });

    const int root_visits = std::max(1, tree_->RootVisits());
    const double sqrt_total = std::sqrt(static_cast<double>(root_visits));
    const int limit = std::min<int>(how_many, static_cast<int>(children.size()));
    for (int i = 0; i < limit; ++i) {
      const Tree::NodeView& child = children[static_cast<std::size_t>(i)];
      // Q is negated because a child's value is from its own mover's perspective,
      // and the root is choosing for the other side.
      const double q = child.visits > 0 ? -static_cast<double>(child.value) : 0.0;
      const double u = config_.c_puct * child.prior * sqrt_total /
                       (1.0 + child.visits);
      emscripten::val entry = emscripten::val::object();
      entry.set("move", child.move);
      entry.set("visits", child.visits);
      entry.set("prior", child.prior);
      entry.set("value", q);
      entry.set("puct", q + u);
      out.call<void>("push", entry);
    }
    return out;
  }

  // Normalised visit counts over the full action space, which is the same quantity
  // self-play writes as its policy target. Indexed by board action, so the swap
  // slot is the last entry.
  emscripten::val PolicyHeatmap() const {
    std::vector<float> mass(static_cast<std::size_t>(kPolicy), 0.0f);
    if (tree_ != nullptr) {
      const auto result = tree_->Collect();
      double total = 0.0;
      for (const auto& [move, visits] : result.visits) total += visits;
      if (total > 0.0)
        for (const auto& [move, visits] : result.visits)
          mass[static_cast<std::size_t>(move)] =
              static_cast<float>(visits / total);
    }
    return emscripten::val::array(mass);
  }

  emscripten::val TreeSnapshot(int max_nodes) const {
    emscripten::val out = emscripten::val::array();
    if (tree_ == nullptr) return out;
    for (const Tree::NodeView& node : tree_->Snapshot(max_nodes)) {
      emscripten::val entry = emscripten::val::object();
      entry.set("index", node.index);
      entry.set("parent", node.parent);
      entry.set("move", node.move);
      entry.set("visits", node.visits);
      entry.set("value", node.value);
      entry.set("prior", node.prior);
      entry.set("depth", node.depth);
      out.call<void>("push", entry);
    }
    return out;
  }

 private:
  hex::Board<kSize> board_;
  std::unique_ptr<Tree> tree_;
  hex::ParallelSearchConfig config_;

  int simulations_done_ = 0;
  int pending_ = 0;
  bool root_needs_expanding_ = false;

  hex::Board<kSize> boards_[kMaxBatch];
  int leaves_[kMaxBatch] = {};
  int applied_[kMaxBatch] = {};

  std::vector<float> planes_;
  std::vector<float> logits_;
  std::vector<float> values_;
};

}  // namespace

EMSCRIPTEN_BINDINGS(hex9) {
  emscripten::class_<HexEngine>("HexEngine")
      .constructor<>()
      .function("reset", &HexEngine::Reset)
      .function("size", &HexEngine::Size)
      .function("policySize", &HexEngine::PolicySize)
      .function("planeCount", &HexEngine::PlaneCount)
      .function("maxBatch", &HexEngine::MaxBatch)
      .function("cellAt", &HexEngine::CellAt)
      .function("toPlay", &HexEngine::ToPlay)
      .function("isTerminal", &HexEngine::IsTerminal)
      .function("winner", &HexEngine::Winner)
      .function("canSwap", &HexEngine::CanSwap)
      .function("moveCount", &HexEngine::MoveCount)
      .function("swapAction", &HexEngine::SwapAction)
      .function("isLegal", &HexEngine::IsLegal)
      .function("play", &HexEngine::Play)
      .function("undo", &HexEngine::Undo)
      .function("connectionDistance", &HexEngine::ConnectionDistance)
      .function("legalMoves", &HexEngine::LegalMoves)
      .function("beginSearch", &HexEngine::BeginSearch)
      .function("searchActive", &HexEngine::SearchActive)
      .function("searchComplete", &HexEngine::SearchComplete)
      .function("simulationsDone", &HexEngine::SimulationsDone)
      .function("simulationBudget", &HexEngine::SimulationBudget)
      .function("prepareBatch", &HexEngine::PrepareBatch)
      .function("submitBatch", &HexEngine::SubmitBatch)
      .function("batchPlanes", &HexEngine::BatchPlanes)
      .function("batchLogits", &HexEngine::BatchLogits)
      .function("batchValues", &HexEngine::BatchValues)
      .function("pendingCount", &HexEngine::PendingCount)
      .function("bestMove", &HexEngine::BestMove)
      .function("rootValue", &HexEngine::RootValue)
      .function("topMoves", &HexEngine::TopMoves)
      .function("policyHeatmap", &HexEngine::PolicyHeatmap)
      .function("treeSnapshot", &HexEngine::TreeSnapshot);
}
