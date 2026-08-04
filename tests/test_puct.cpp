#include <algorithm>
#include <cmath>
#include <cstdio>
#include <numeric>
#include <string>
#include <vector>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
#include "hex/puct.hpp"
#include "hex/solver.hpp"

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

// Every sampler is hand-rolled for cross-toolchain determinism, so each one
// needs its distribution checked rather than taken on trust.
void TestSamplers() {
  std::printf("hand-rolled samplers\n");
  std::mt19937 rng(1234);

  double sum = 0.0;
  double min_seen = 1.0;
  double max_seen = 0.0;
  for (int i = 0; i < 200000; ++i) {
    const double u = hex::detail::NextUnit(rng);
    sum += u;
    min_seen = std::min(min_seen, u);
    max_seen = std::max(max_seen, u);
  }
  Check(std::abs(sum / 200000 - 0.5) < 0.005, "uniform mean should be 0.5");
  Check(min_seen > 0.0 && max_seen < 1.0, "uniform must stay open at both ends");

  double nsum = 0.0;
  double nsq = 0.0;
  for (int i = 0; i < 200000; ++i) {
    const double z = hex::detail::NextNormal(rng);
    nsum += z;
    nsq += z * z;
  }
  Check(std::abs(nsum / 200000) < 0.02, "normal mean should be 0");
  Check(std::abs(nsq / 200000 - 1.0) < 0.02, "normal variance should be 1");

  // Gamma(a, 1) has mean a and variance a. The shape below 1 exercises the
  // boost path, which is the branch most likely to be wrong.
  for (const double shape : {0.3, 1.0, 2.5}) {
    double gsum = 0.0;
    double gsq = 0.0;
    constexpr int kDraws = 200000;
    for (int i = 0; i < kDraws; ++i) {
      const double g = hex::detail::NextGamma(rng, shape);
      gsum += g;
      gsq += g * g;
    }
    const double mean = gsum / kDraws;
    const double var = gsq / kDraws - mean * mean;
    Check(std::abs(mean - shape) < 0.02 * std::max(1.0, shape),
          "gamma mean wrong for shape " + std::to_string(shape));
    Check(std::abs(var - shape) < 0.05 * std::max(1.0, shape),
          "gamma variance wrong for shape " + std::to_string(shape));
  }

  std::vector<double> dir(9);
  hex::detail::SampleDirichlet(rng, 0.3, dir);
  const double total = std::accumulate(dir.begin(), dir.end(), 0.0);
  Check(std::abs(total - 1.0) < 1e-9, "dirichlet should sum to one");
  Check(std::all_of(dir.begin(), dir.end(), [](double v) { return v >= 0.0; }),
        "dirichlet components must be non-negative");
  std::printf("  uniform, normal, gamma(0.3/1.0/2.5), dirichlet all in range\n");
}

void TestSearchRestoresBoard() {
  std::printf("search leaves the board untouched\n");
  constexpr int N = 9;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(4, 4));
  board.Play(hex::Board<N>::Index(2, 6));

  const std::uint64_t hash = board.Hash();
  const int ply = board.MoveCount();

  hex::Puct<N> puct(typename hex::Puct<N>::Config{400, 1.5f, 0.3f, 0.25f, 7, true});
  hex::RolloutEvaluator<N> evaluator(3);
  const auto result = puct.Search(board, evaluator);

  Check(board.Hash() == hash, "hash changed across search");
  Check(board.MoveCount() == ply, "ply changed across search");
  Check(result.best_move >= 0, "no move returned");
  Check(board.At(result.best_move) == hex::Cell::kEmpty, "chose an occupied cell");
}

// Visit counts are the policy training target, so their bookkeeping has to be
// exact.
void TestVisitDistribution() {
  std::printf("visit counts form a valid training target\n");
  constexpr int N = 7;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(3, 3));

  constexpr int kSimulations = 500;
  hex::Puct<N> puct(
      typename hex::Puct<N>::Config{kSimulations, 1.5f, 0.3f, 0.0f, 11, true});
  hex::RolloutEvaluator<N> evaluator(5);
  const auto result = puct.Search(board, evaluator);

  int total = 0;
  for (const auto& [move, visits] : result.visits) total += visits;
  Check(total == kSimulations, "root visits should sum to the simulation count");
  Check(static_cast<int>(result.visits.size()) == board.NumEmpty() + 1,
        "root should have one child per legal action including swap");
  Check(result.root_value >= -1.0f && result.root_value <= 1.0f,
        "root value out of range");
}

// A prior concentrated on one action must pull search towards it. This is the
// property that makes the policy head worth training at all.
void TestPriorsSteerSearch() {
  std::printf("priors steer the search\n");
  constexpr int N = 7;
  hex::Board<N> board;
  const int favoured = hex::Board<N>::Index(5, 5);

  auto biased = [favoured](const hex::Board<N>& b, hex::Evaluation<N>& out) {
    out.priors.fill(0.001f);
    out.priors[favoured] = 10.0f;
    out.value = 0.0f;
    (void)b;
  };
  auto flat = [](const hex::Board<N>& b, hex::Evaluation<N>& out) {
    out.priors.fill(1.0f);
    out.value = 0.0f;
    (void)b;
  };

  typename hex::Puct<N>::Config config{600, 1.5f, 0.3f, 0.0f, 2, false};
  hex::Puct<N> guided(config);
  hex::Puct<N> unguided(config);

  const auto with_prior = guided.Search(board, biased);
  const auto without = unguided.Search(board, flat);

  auto visits_for = [](const auto& result, int move) {
    for (const auto& [m, v] : result.visits)
      if (m == move) return v;
    return 0;
  };

  Check(with_prior.best_move == favoured,
        "a dominant prior should win the root");
  Check(visits_for(with_prior, favoured) > 5 * visits_for(without, favoured),
        "a dominant prior should concentrate visits");
  std::printf("  favoured action: %d visits with prior, %d without\n",
              visits_for(with_prior, favoured), visits_for(without, favoured));
}

void TestDeterminism() {
  std::printf("search is deterministic under a fixed seed\n");
  constexpr int N = 7;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(3, 3));

  typename hex::Puct<N>::Config config{400, 1.5f, 0.3f, 0.25f, 42, true};
  hex::Puct<N> a(config);
  hex::Puct<N> b(config);
  hex::RolloutEvaluator<N> ea(9);
  hex::RolloutEvaluator<N> eb(9);

  const auto ra = a.Search(board, ea);
  const auto rb = b.Search(board, eb);
  Check(ra.best_move == rb.best_move, "same seed gave different moves");
  Check(ra.visits == rb.visits, "same seed gave different visit counts");
}

// Root noise must perturb the search without being the only thing driving it.
void TestRootNoise() {
  std::printf("dirichlet root noise\n");
  constexpr int N = 7;
  hex::Board<N> board;

  auto flat = [](const hex::Board<N>& b, hex::Evaluation<N>& out) {
    out.priors.fill(1.0f);
    out.value = 0.0f;
    (void)b;
  };

  typename hex::Puct<N>::Config quiet{300, 1.5f, 0.3f, 0.0f, 5, false};
  hex::Puct<N> a(quiet);
  hex::Puct<N> b(quiet);
  Check(a.Search(board, flat).visits == b.Search(board, flat).visits,
        "zero noise weight should be reproducible");

  typename hex::Puct<N>::Config noisy_a{300, 1.5f, 0.3f, 0.25f, 5, false};
  typename hex::Puct<N>::Config noisy_b{300, 1.5f, 0.3f, 0.25f, 6, false};
  hex::Puct<N> c(noisy_a);
  hex::Puct<N> d(noisy_b);
  Check(c.Search(board, flat).visits != d.Search(board, flat).visits,
        "different seeds should give different noise");
}

void TestTemperature() {
  std::printf("temperature sampling\n");
  constexpr int N = 7;
  hex::Board<N> board;
  board.Play(hex::Board<N>::Index(3, 3));

  typename hex::Puct<N>::Config config{500, 1.5f, 0.3f, 0.0f, 13, false};
  hex::Puct<N> puct(config);
  hex::RolloutEvaluator<N> evaluator(4);
  const auto result = puct.Search(board, evaluator);

  for (int i = 0; i < 20; ++i)
    Check(puct.SampleMove(result, 0.0f) == result.best_move,
          "temperature zero must be greedy");

  std::vector<int> distinct;
  for (int i = 0; i < 200; ++i) {
    const int move = puct.SampleMove(result, 1.0f);
    if (std::find(distinct.begin(), distinct.end(), move) == distinct.end())
      distinct.push_back(move);
  }
  Check(distinct.size() > 3, "temperature one should explore several moves");
  std::printf("  temperature 1.0 produced %zu distinct moves in 200 draws\n",
              distinct.size());
}

// Stands in for the network until one exists: uniform priors and a value from
// the connection-distance heuristic. Squashed through tanh so it lands in the
// same [-1, 1] range a value head produces.
template <int N>
struct HeuristicEvaluator {
  void operator()(const hex::Board<N>& board, hex::Evaluation<N>& out) const {
    out.priors.fill(1.0f);
    const hex::Player me = board.ToPlay();
    const int mine = hex::AlphaBeta<N>::ConnectionDistance(board, me);
    const int theirs =
        hex::AlphaBeta<N>::ConnectionDistance(board, hex::Opponent(me));
    out.value = std::tanh(0.5f * static_cast<float>(theirs - mine));
  }
};

// The strongest available check: on boards small enough to solve exactly,
// search must pick a provably winning move. The heuristic evaluator is used
// here so the test measures the search rather than rollout variance.
template <int N>
void TestAgreesWithSolver(int simulations) {
  hex::Board<N> board;
  hex::Solver<N> solver;
  const std::vector<int> winning = solver.WinningMoves(board);

  int agreements = 0;
  constexpr int kSeeds = 8;
  for (int seed = 0; seed < kSeeds; ++seed) {
    hex::Puct<N> puct(typename hex::Puct<N>::Config{
        simulations, 1.5f, 0.3f, 0.0f, static_cast<std::uint64_t>(seed + 1),
        false});
    HeuristicEvaluator<N> evaluator;
    const auto result = puct.Search(board, evaluator);
    if (std::find(winning.begin(), winning.end(), result.best_move) !=
        winning.end())
      ++agreements;
  }
  Check(agreements == kSeeds, std::to_string(N) + "x" + std::to_string(N) +
                                  ": chose a losing opening");
  std::printf("  %dx%d: %d/%d seeds at %d simulations\n", N, N, agreements,
              kSeeds, simulations);
}

// How much search does each evaluator need to solve the same 4x4 opening? This
// is the argument for training a value head instead of buying more rollouts.
void TestEvaluatorQualityGap() {
  std::printf("evaluator quality (calibration)\n");
  constexpr int N = 4;
  hex::Board<N> board;
  hex::Solver<N> solver;
  const std::vector<int> winning = solver.WinningMoves(board);

  auto score = [&](int simulations, bool heuristic) {
    int agreements = 0;
    for (int seed = 0; seed < 8; ++seed) {
      hex::Puct<N> puct(typename hex::Puct<N>::Config{
          simulations, 1.5f, 0.3f, 0.0f, static_cast<std::uint64_t>(seed + 1),
          false});
      int move;
      if (heuristic) {
        HeuristicEvaluator<N> evaluator;
        move = puct.Search(board, evaluator).best_move;
      } else {
        hex::RolloutEvaluator<N> evaluator(static_cast<std::uint64_t>(seed + 100));
        move = puct.Search(board, evaluator).best_move;
      }
      if (std::find(winning.begin(), winning.end(), move) != winning.end())
        ++agreements;
    }
    return agreements;
  };

  const int heuristic_2k = score(2000, true);
  const int rollout_2k = score(2000, false);
  const int rollout_50k = score(50000, false);

  Check(heuristic_2k == 8, "heuristic evaluator should solve 4x4 at 2k");
  Check(heuristic_2k > rollout_2k,
        "heuristic should beat rollouts at equal simulation count");
  std::printf("  heuristic @  2k: %d/8   rollout @  2k: %d/8"
              "   rollout @ 50k: %d/8\n",
              heuristic_2k, rollout_2k, rollout_50k);
}

}  // namespace

int main() {
  constexpr int kExpectedChecks = 50;

  std::printf("== hex puct ==\n\n");
  TestSamplers();
  TestSearchRestoresBoard();
  TestVisitDistribution();
  TestPriorsSteerSearch();
  TestDeterminism();
  TestRootNoise();
  TestTemperature();
  std::printf("agrees with exhaustive solver\n");
  TestAgreesWithSolver<3>(500);
  TestAgreesWithSolver<4>(2000);
  TestEvaluatorQualityGap();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_checks != kExpectedChecks) {
    std::printf("\nERROR: expected %d checks.\n", kExpectedChecks);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
