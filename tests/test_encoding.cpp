#include <algorithm>
#include <cstdio>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "hex/board.hpp"
#include "hex/encoding.hpp"

namespace {

int g_failures = 0;
int g_checks = 0;

void Check(bool condition, const std::string& what) {
  ++g_checks;
  if (!condition) {
    ++g_failures;
    std::printf("  FAIL: %s\n", what.c_str());
  }
}

std::uint32_t NextBelow(std::mt19937& rng, std::uint32_t n) {
  return static_cast<std::uint32_t>(rng() % n);
}

template <typename T>
void Shuffle(std::vector<T>& v, std::mt19937& rng) {
  for (std::size_t i = v.size(); i > 1; --i)
    std::swap(v[i - 1], v[NextBelow(rng, static_cast<std::uint32_t>(i))]);
}

void TestRedIsNotTransposed() {
  std::printf("red to move is presented as-is\n");
  constexpr int N = 5;
  using E = hex::Encoder<N>;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(1, 3));  // red
  board.Play(hex::Board<N>::Index(2, 0));  // blue, red to move again

  Check(board.ToPlay() == hex::Player::kRed, "red should be to move");
  Check(!E::NeedsTranspose(board), "red to move must not transpose");

  std::vector<float> planes(E::kInputSize);
  E::Encode(board, planes.data());

  Check(planes[hex::Board<N>::Index(1, 3)] == 1.0f,
        "red stone missing from plane 0");
  Check(planes[E::kCells + hex::Board<N>::Index(2, 0)] == 1.0f,
        "blue stone missing from plane 1");
  Check(planes[hex::Board<N>::Index(2, 0)] == 0.0f,
        "blue stone leaked into plane 0");
}

void TestBlueIsTransposed() {
  std::printf("blue to move is transposed onto the top-bottom axis\n");
  constexpr int N = 5;
  using E = hex::Encoder<N>;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(1, 3));  // red, blue to move

  Check(board.ToPlay() == hex::Player::kBlue, "blue should be to move");
  Check(E::NeedsTranspose(board), "blue to move must transpose");

  std::vector<float> planes(E::kInputSize);
  E::Encode(board, planes.data());

  // Red's stone belongs to the opponent and lands transposed.
  Check(planes[E::kCells + hex::Board<N>::Index(3, 1)] == 1.0f,
        "red stone not transposed into plane 1");
  Check(planes[E::kCells + hex::Board<N>::Index(1, 3)] == 0.0f,
        "red stone left in its untransposed position");
}

// The load-bearing property. A position and its transpose-plus-recolour are the
// same game state with the roles exchanged, so both must encode to an identical
// tensor. If this fails, the network is being asked to learn two goal
// directions instead of one.
void TestIsomorphicPositionsEncodeIdentically() {
  std::printf("isomorphic positions encode identically\n");
  constexpr int N = 7;
  using E = hex::Encoder<N>;
  using B = hex::Board<N>;
  std::mt19937 rng(20260804);

  constexpr int kTrials = 3000;
  for (int trial = 0; trial < kTrials; ++trial) {
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);
    const int plies = 1 + static_cast<int>(NextBelow(rng, 20));

    // Build a position, then the mirror position with colours exchanged.
    hex::Board<N> direct;
    hex::Board<N> mirror;
    for (int i = 0; i < plies; ++i) {
      const int move = order[static_cast<std::size_t>(i)];
      const hex::Player mover =
          i % 2 == 0 ? hex::Player::kRed : hex::Player::kBlue;
      direct.PlaceStone(move, mover);
      mirror.PlaceStone(B::Index(B::Col(move), B::Row(move)),
                        hex::Opponent(mover));
      if (direct.IsTerminal() || mirror.IsTerminal()) break;
    }

    std::vector<float> a(E::kInputSize);
    std::vector<float> b(E::kInputSize);
    E::Encode(direct, a.data());
    E::Encode(mirror, b.data());

    if (a != b) {
      Check(false, "a position and its mirror encoded differently");
      return;
    }
  }
  Check(true, "");
  std::printf("  %d random position pairs encoded identically\n", kTrials);
}

void TestCanonicalRoundTrip() {
  std::printf("canonical action mapping round-trips\n");
  constexpr int N = 9;
  using E = hex::Encoder<N>;
  std::mt19937 rng(7);

  for (int trial = 0; trial < 2000; ++trial) {
    hex::Board<N> board;
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);
    const int plies = static_cast<int>(NextBelow(rng, 30));
    for (int i = 0; i < plies && !board.IsTerminal(); ++i) board.Play(order[i]);
    if (board.IsTerminal()) continue;

    for (int i = 0; i < board.NumEmpty(); ++i) {
      const int move = board.LegalMoves()[i];
      const int canonical = E::ToCanonical(board, move);
      Check(canonical >= 0 && canonical < E::kPolicySize,
            "canonical index out of range");
      Check(E::FromCanonical(board, canonical) == move,
            "canonical mapping did not round-trip");
      if (g_failures) return;
    }
    Check(E::ToCanonical(board, hex::Board<N>::kSwapMove) ==
              hex::Board<N>::kSwapMove,
          "swap index must be transpose-invariant");
    if (g_failures) return;
  }
  std::printf("  2000 positions, every legal action round-tripped\n");
}

// Stone count alone cannot tell ply 1 from a post-swap ply 2, so the swap plane
// carries information the other two do not.
void TestSwapPlane() {
  std::printf("swap plane distinguishes ply 1 from post-swap\n");
  constexpr int N = 7;
  using E = hex::Encoder<N>;
  std::vector<float> planes(E::kInputSize);

  hex::Board<N> before;
  before.Play(hex::Board<N>::Index(2, 4));
  Check(before.CanSwap(), "swap should be available at ply 1");
  E::Encode(before, planes.data());
  Check(planes[2 * E::kCells] == 1.0f, "swap plane should be set at ply 1");

  hex::Board<N> after;
  after.Play(hex::Board<N>::Index(2, 4));
  after.PlaySwap();
  Check(!after.CanSwap(), "swap should be gone after swapping");
  Check(after.NumEmpty() == before.NumEmpty(),
        "both positions should hold exactly one stone");
  E::Encode(after, planes.data());
  Check(planes[2 * E::kCells] == 0.0f, "swap plane should be clear after swap");

  std::vector<float> mask(E::kPolicySize);
  E::EncodeLegalMask(before, mask.data());
  Check(mask[hex::Board<N>::kSwapMove] == 1.0f, "mask should allow swap at ply 1");
  E::EncodeLegalMask(after, mask.data());
  Check(mask[hex::Board<N>::kSwapMove] == 0.0f,
        "mask should forbid swap after swapping");
}

void TestPolicyTarget() {
  std::printf("policy target is normalised and canonical\n");
  constexpr int N = 5;
  using E = hex::Encoder<N>;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(1, 3));  // blue to move, so transposed

  const std::vector<std::pair<int, int>> visits = {
      {hex::Board<N>::Index(0, 2), 30},
      {hex::Board<N>::Index(4, 1), 50},
      {hex::Board<N>::kSwapMove, 20},
  };

  std::vector<float> target(E::kPolicySize);
  E::EncodePolicyTarget(board, visits, target.data());

  const float total = std::accumulate(target.begin(), target.end(), 0.0f);
  Check(std::abs(total - 1.0f) < 1e-6f, "policy target should sum to one");
  Check(std::abs(target[hex::Board<N>::Index(2, 0)] - 0.3f) < 1e-6f,
        "first move not transposed into the target");
  Check(std::abs(target[hex::Board<N>::Index(1, 4)] - 0.5f) < 1e-6f,
        "second move not transposed into the target");
  Check(std::abs(target[hex::Board<N>::kSwapMove] - 0.2f) < 1e-6f,
        "swap mass misplaced");
}

// Golden fixture. The Python trainer must reproduce these tensors exactly from
// the same move sequences, or the two encoders have silently diverged.
void WriteFixture(const char* path) {
  constexpr int N = 9;
  using E = hex::Encoder<N>;
  std::mt19937 rng(31337);

  std::FILE* file = std::fopen(path, "w");
  if (file == nullptr) {
    Check(false, "could not open the fixture file for writing");
    return;
  }
  std::fprintf(file, "# hex9 encoding fixture\n");
  std::fprintf(file, "# board_size planes policy_size\n");
  std::fprintf(file, "%d %d %d\n", N, E::kPlanes, E::kPolicySize);
  std::fprintf(file, "# per case: moves..., then -1, then the plane sum and a "
                     "checksum over set indices\n");

  int cases = 0;
  for (int trial = 0; trial < 64; ++trial) {
    hex::Board<N> board;
    std::vector<int> order(N * N);
    std::iota(order.begin(), order.end(), 0);
    Shuffle(order, rng);
    const int plies = static_cast<int>(NextBelow(rng, 24));

    std::vector<int> played;
    for (int i = 0; i < plies && !board.IsTerminal(); ++i) {
      board.Play(order[static_cast<std::size_t>(i)]);
      played.push_back(order[static_cast<std::size_t>(i)]);
    }
    if (board.IsTerminal()) continue;

    std::vector<float> planes(E::kInputSize);
    E::Encode(board, planes.data());

    double sum = 0.0;
    std::uint64_t checksum = 1469598103934665603ULL;
    for (int i = 0; i < E::kInputSize; ++i) {
      sum += planes[i];
      if (planes[i] != 0.0f) {
        checksum ^= static_cast<std::uint64_t>(i);
        checksum *= 1099511628211ULL;
      }
    }

    for (const int move : played) std::fprintf(file, "%d ", move);
    std::fprintf(file, "-1 %.1f %llu\n", sum,
                 static_cast<unsigned long long>(checksum));
    ++cases;
  }
  std::fclose(file);
  Check(cases > 40, "fixture should contain a useful number of cases");
  std::printf("  wrote %d cases to %s\n", cases, path);
}

}  // namespace

int main(int argc, char** argv) {
  constexpr int kExpectedChecks = 268086;

  std::printf("== hex encoding ==\n\n");
  TestRedIsNotTransposed();
  TestBlueIsTransposed();
  TestIsomorphicPositionsEncodeIdentically();
  TestCanonicalRoundTrip();
  TestSwapPlane();
  TestPolicyTarget();

  std::printf("golden fixture for the python trainer\n");
  WriteFixture(argc > 1 ? argv[1] : "encoding_fixture.txt");

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_checks != kExpectedChecks) {
    std::printf("\nERROR: expected %d checks.\n", kExpectedChecks);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
