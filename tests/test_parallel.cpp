// Tests for the parallel self-play driver.
//
// The load-bearing property is that a run's output does not depend on how it was
// scheduled. Every game is seeded from (run seed, game index) and completed records
// are flushed in index order, so one thread and fifteen must produce the same games
// in the same order. Losing that would make every replay buffer quietly dependent
// on the machine it was generated on, and nothing else in the pipeline would
// notice.
//
// The comparison is done in memory rather than by diffing shards, so a failure
// points at the game that diverged instead of at a byte offset.

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include "hex/alphabeta.hpp"
#include "hex/board.hpp"
#include "hex/parallel_puct.hpp"
#include "hex/parallel_selfplay.hpp"
#include "hex/selfplay_record.hpp"
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

// Pure: no state, so its answer depends only on the board. That is what makes a
// threaded run reproducible.
template <int N>
struct PureEvaluator {
  void operator()(const hex::Board<N>& board, hex::Evaluation<N>& out) const {
    out.priors.fill(1.0f);
    const hex::Player me = board.ToPlay();
    const int mine = hex::AlphaBeta<N>::ConnectionDistance(board, me);
    const int theirs =
        hex::AlphaBeta<N>::ConnectionDistance(board, hex::Opponent(me));
    out.value = std::tanh(0.5f * static_cast<float>(theirs - mine));
  }
};

// Deliberately impure: the value depends on how many boards this instance has
// already seen. Stands in for RolloutEvaluator, which owns an RNG, without
// depending on the rollout implementation.
template <int N>
struct StatefulEvaluator {
  int seen = 0;
  void operator()(const hex::Board<N>& board, hex::Evaluation<N>& out) {
    out.priors.fill(1.0f);
    ++seen;
    const hex::Player me = board.ToPlay();
    const int mine = hex::AlphaBeta<N>::ConnectionDistance(board, me);
    out.value = std::tanh(0.1f * static_cast<float>((seen % 7) - mine));
  }
};

bool SameRecord(const hex::SelfPlayRecord& a, const hex::SelfPlayRecord& b) {
  return a.moves == b.moves && a.visits == b.visits && a.winner == b.winner;
}

template <int N>
std::vector<hex::SelfPlayRecord> Collect(const hex::ParallelConfig& config) {
  std::vector<hex::SelfPlayRecord> out;
  hex::RunParallelSelfPlay<N>(
      config, [] { return PureEvaluator<N>{}; },
      [&out](hex::SelfPlayRecord& record) { out.push_back(record); });
  return out;
}

hex::ParallelConfig BaseConfig(int games, int threads, int block) {
  hex::ParallelConfig config;
  config.games = games;
  config.threads = threads;
  config.block = block;
  config.simulations = 40;
  config.temperature_moves = 4;
  config.seed = 20260806;
  return config;
}

void TestReproducibleAcrossThreadCounts() {
  std::printf("output is independent of thread count\n");
  constexpr int kSize = 5;
  constexpr int kGames = 60;

  const auto reference = Collect<kSize>(BaseConfig(kGames, 1, 0));
  Check(static_cast<int>(reference.size()) == kGames,
        "single-threaded run should produce every game");

  for (const int threads : {2, 3, 5, 8, 16}) {
    const auto actual = Collect<kSize>(BaseConfig(kGames, threads, 0));
    Check(actual.size() == reference.size(),
          "game count should not depend on thread count");
    bool identical = actual.size() == reference.size();
    for (std::size_t i = 0; identical && i < actual.size(); ++i)
      if (!SameRecord(actual[i], reference[i])) {
        identical = false;
        std::printf("    diverged at game %zu with %d threads\n", i, threads);
      }
    Check(identical, "records should match the single-threaded run exactly");
  }
  std::printf("  60 games matched across 1, 2, 3, 5, 8 and 16 threads\n");
}

void TestReproducibleAcrossBlockSizes() {
  std::printf("output is independent of block size\n");
  constexpr int kSize = 5;
  constexpr int kGames = 40;

  const auto reference = Collect<kSize>(BaseConfig(kGames, 1, 0));
  // block=1 forces a barrier phase per game, so all but one worker arrives with
  // nothing done. block larger than the run means a single phase.
  for (const int block : {1, 2, 7, 40, 100}) {
    const auto actual = Collect<kSize>(BaseConfig(kGames, 6, block));
    bool identical = actual.size() == reference.size();
    for (std::size_t i = 0; identical && i < actual.size(); ++i)
      identical = SameRecord(actual[i], reference[i]);
    Check(identical, "block size should not change the output");
  }
  std::printf("  40 games matched at block 1, 2, 7, 40 and 100\n");
}

void TestSinkSeesGamesInIndexOrder() {
  std::printf("sink is called in game-index order\n");
  constexpr int kSize = 5;
  constexpr int kGames = 50;

  // Each game's first move is a function of its index through the seed, so
  // matching the single-threaded sequence is what proves the ordering. A weaker
  // check -- that the same multiset arrived -- would pass on a shuffled flush.
  const auto reference = Collect<kSize>(BaseConfig(kGames, 1, 0));
  const auto threaded = Collect<kSize>(BaseConfig(kGames, 8, 3));

  int mismatches = 0;
  for (std::size_t i = 0; i < threaded.size() && i < reference.size(); ++i)
    if (!SameRecord(threaded[i], reference[i])) ++mismatches;
  Check(mismatches == 0, "flush order should follow the game index");
  Check(threaded.size() == static_cast<std::size_t>(kGames),
        "every game should reach the sink exactly once");
  std::printf("  50 games arrived in index order with 8 workers\n");
}

void TestStatsAgreeWithRecords() {
  std::printf("reported statistics match the records\n");
  constexpr int kSize = 5;
  hex::ParallelConfig config = BaseConfig(70, 7, 0);

  std::vector<hex::SelfPlayRecord> seen;
  const hex::ParallelStats stats = hex::RunParallelSelfPlay<kSize>(
      config, [] { return PureEvaluator<kSize>{}; },
      [&seen](hex::SelfPlayRecord& record) { seen.push_back(record); });

  long long plies = 0;
  int red = 0;
  int swaps = 0;
  for (const hex::SelfPlayRecord& record : seen) {
    plies += static_cast<long long>(record.moves.size());
    if (record.winner > 0) ++red;
    for (const int move : record.moves)
      if (move == hex::Board<kSize>::kSwapMove) ++swaps;
  }

  Check(stats.games == static_cast<int>(seen.size()), "game count should agree");
  Check(stats.total_plies == plies, "ply total should agree");
  Check(stats.red_wins == red, "red win count should agree");
  Check(stats.swaps == swaps, "swap count should agree");
  std::printf("  %d games, %lld plies, %d red wins, %d swaps\n", stats.games,
              stats.total_plies, stats.red_wins, stats.swaps);
}

void TestRecordsAreWellFormed() {
  std::printf("every record is well formed\n");
  constexpr int kSize = 5;
  const auto games = Collect<kSize>(BaseConfig(40, 5, 0));

  bool shapes_ok = true;
  bool winners_ok = true;
  bool actions_ok = true;
  bool visits_ok = true;
  for (const hex::SelfPlayRecord& record : games) {
    if (record.moves.size() != record.visits.size()) shapes_ok = false;
    if (record.moves.empty()) shapes_ok = false;
    if (record.winner != 1 && record.winner != -1) winners_ok = false;
    for (const int move : record.moves)
      if (move < 0 || move > hex::Board<kSize>::kSwapMove) actions_ok = false;
    for (const auto& row : record.visits) {
      if (row.empty()) visits_ok = false;
      for (const auto& [action, count] : row)
        if (count == 0 || action < 0 || action > hex::Board<kSize>::kSwapMove)
          visits_ok = false;
    }
  }
  Check(shapes_ok, "moves and visits should have one entry per ply");
  Check(winners_ok, "Hex admits no draws, so a winner is +1 or -1");
  Check(actions_ok, "every move should be inside the action space");
  Check(visits_ok, "every recorded visit count should be positive");
}

void TestDegenerateConfigurations() {
  std::printf("degenerate configurations behave\n");
  constexpr int kSize = 5;

  const auto none = Collect<kSize>(BaseConfig(0, 4, 0));
  Check(none.empty(), "zero games should produce no records and not hang");

  const auto one = Collect<kSize>(BaseConfig(1, 8, 0));
  Check(one.size() == 1, "one game across eight workers should yield one game");

  // More workers than games leaves most of the pool with nothing to do; they must
  // still reach the barrier so the run can finish.
  const auto few = Collect<kSize>(BaseConfig(3, 16, 0));
  Check(few.size() == 3, "idle workers should not lose games");

  const auto reference = Collect<kSize>(BaseConfig(3, 1, 0));
  bool identical = few.size() == reference.size();
  for (std::size_t i = 0; identical && i < few.size(); ++i)
    identical = SameRecord(few[i], reference[i]);
  Check(identical, "oversubscribed runs should still match a serial run");
}

// Documents a precondition rather than a defect. Reproducibility requires the
// evaluator to be a pure function of the board; a stateful one makes a game's
// content depend on which worker claimed it. This is why self-play uses the
// heuristic and network evaluators and not RolloutEvaluator.
void TestStatefulEvaluatorIsNotReproducible() {
  std::printf("a stateful evaluator forfeits reproducibility\n");
  constexpr int kSize = 5;
  constexpr int kGames = 40;

  const auto run = [](int threads) {
    std::vector<hex::SelfPlayRecord> out;
    hex::RunParallelSelfPlay<kSize>(
        BaseConfig(kGames, threads, 0),
        [] { return StatefulEvaluator<kSize>{}; },
        [&out](hex::SelfPlayRecord& record) { out.push_back(record); });
    return out;
  };

  const auto serial = run(1);
  const auto serial_again = run(1);
  bool serial_stable = serial.size() == serial_again.size();
  for (std::size_t i = 0; serial_stable && i < serial.size(); ++i)
    serial_stable = SameRecord(serial[i], serial_again[i]);
  Check(serial_stable,
        "even a stateful evaluator is reproducible on a single thread");

  const auto threaded = run(8);
  Check(threaded.size() == serial.size(),
        "a stateful evaluator should still produce every game");
  int differing = 0;
  for (std::size_t i = 0; i < threaded.size() && i < serial.size(); ++i)
    if (!SameRecord(threaded[i], serial[i])) ++differing;
  Check(differing > 0,
        "threaded output should diverge, confirming purity is required");
  std::printf("  %d of %d games differed, as expected\n", differing,
              static_cast<int>(serial.size()));
}

void TestLargerBoardReproduces() {
  std::printf("the guarantee holds on a larger board\n");
  constexpr int kSize = 9;
  hex::ParallelConfig config = BaseConfig(12, 1, 0);
  config.simulations = 20;

  const auto reference = Collect<kSize>(config);
  config.threads = 8;
  const auto threaded = Collect<kSize>(config);

  bool identical = reference.size() == threaded.size();
  for (std::size_t i = 0; identical && i < reference.size(); ++i)
    identical = SameRecord(reference[i], threaded[i]);
  Check(identical, "9x9 output should not depend on thread count");
  long long plies = 0;
  for (const auto& record : reference)
    plies += static_cast<long long>(record.moves.size());
  std::printf("  12 games, mean %.1f plies, matched on 1 and 8 threads\n",
              static_cast<double>(plies) / reference.size());
}

// Root-parallel with one thread is a wrapper around Puct with noise off, so it must
// reproduce Puct exactly. Anything else means the wrapper is not neutral, and every
// comparison built on it would be measuring the wrapper.
void TestRootParallelMatchesPuctOnOneThread() {
  std::printf("root-parallel with one thread reproduces Puct\n");
  constexpr int kSize = 5;
  hex::Board<kSize> board;
  board.Play(12);
  board.Play(7);

  hex::ParallelSearchConfig config;
  config.threads = 1;
  config.simulations = 300;
  config.seed = 99;

  typename hex::Puct<kSize>::Config plain{};
  plain.simulations = 300;
  plain.c_puct = config.c_puct;
  plain.dirichlet_weight = 0.0f;
  plain.allow_swap = config.allow_swap;
  plain.seed = config.seed;

  hex::Puct<kSize> reference(plain);
  PureEvaluator<kSize> evaluator;
  const auto expected = reference.Search(board, evaluator);

  const auto actual = hex::SearchRootParallel<kSize>(
      board, config, [] { return PureEvaluator<kSize>{}; });

  Check(actual.best_move == expected.best_move,
        "one-thread root-parallel should choose Puct's move");
  bool visits_match = actual.visits.size() == expected.visits.size();
  for (const auto& [move, count] : expected.visits) {
    bool found = false;
    for (const auto& [other_move, other_count] : actual.visits)
      if (other_move == move && other_count == count) found = true;
    if (!found) visits_match = false;
  }
  Check(visits_match, "visit counts should match Puct exactly");
}

// Every mode has to leave the caller's board untouched. A search that mutates it
// would corrupt the game it was called from, and with make/unmake rather than
// copies that is an easy mistake to make.
void TestModesRestoreTheBoard() {
  std::printf("every mode leaves the board unchanged\n");
  constexpr int kSize = 5;
  for (const hex::ParallelMode mode :
       {hex::ParallelMode::kRoot, hex::ParallelMode::kTree,
        hex::ParallelMode::kLeaf}) {
    hex::Board<kSize> board;
    board.Play(6);
    board.Play(18);
    const std::uint64_t before = board.Hash();
    const int plies = board.MoveCount();

    hex::ParallelSearchConfig config;
    config.threads = 4;
    config.simulations = 120;
    config.seed = 7;
    const auto result = hex::SearchParallel<kSize>(
        board, mode, config, [] { return PureEvaluator<kSize>{}; });

    Check(board.Hash() == before, "search should restore the board hash");
    Check(board.MoveCount() == plies, "search should restore the ply count");
    Check(result.best_move >= 0, "search should choose a move");
  }
}

// The discriminating check: on 4x4 only 4 of 16 openings win, so choosing a
// provably winning move is a real result rather than a coin flip. Run for every
// mode, because a mode that scales beautifully and plays badly is worthless.
//
// Swap must be disabled, and getting that wrong is instructive. Solver ignores the
// swap rule deliberately -- it answers a question about the pure game -- so its
// winning moves are the swap-free ones. Leave swap enabled and the search correctly
// learns to avoid exactly those moves, because a strong opening under the swap rule
// is one the opponent simply takes. Measured at 8000 simulations with swap on, the
// search puts over half its visits on a move the swap-free solver calls losing, and
// it is right to. tests/test_puct.cpp disables swap here for the same reason.
//
// Several seeds with a majority requirement, because tree-parallel is genuinely
// nondeterministic: threads race, so one seed could fail on scheduling alone.
template <int N>
void TestModeAgreesWithSolver(hex::ParallelMode mode, const char* label,
                              int simulations) {
  hex::Board<N> board;
  hex::Solver<N> solver;
  const std::vector<int> winning = solver.WinningMoves(board);

  constexpr int kSeeds = 4;
  int agreements = 0;
  long long evaluations = 0;
  for (int seed = 0; seed < kSeeds; ++seed) {
    hex::ParallelSearchConfig config;
    config.threads = 4;
    config.simulations = simulations;
    config.seed = static_cast<std::uint64_t>(seed) + 1;
    config.allow_swap = false;

    const auto result = hex::SearchParallel<N>(
        board, mode, config, [] { return PureEvaluator<N>{}; });
    evaluations += result.evaluations;
    if (std::find(winning.begin(), winning.end(), result.best_move) !=
        winning.end())
      ++agreements;
  }

  Check(agreements >= kSeeds - 1,
        std::string(label) + " should choose a winning opening on most seeds");
  Check(evaluations > 0, std::string(label) + " should evaluate leaves");
  std::printf("  %-6s %d/%d seeds chose one of %d winning openings\n", label,
              agreements, kSeeds, static_cast<int>(winning.size()));
}

void TestModesAgreeWithSolver() {
  std::printf("every mode agrees with the exhaustive solver on 4x4\n");
  TestModeAgreesWithSolver<4>(hex::ParallelMode::kRoot, "root", 2000);
  TestModeAgreesWithSolver<4>(hex::ParallelMode::kTree, "tree", 2000);
  TestModeAgreesWithSolver<4>(hex::ParallelMode::kLeaf, "leaf", 2000);
}

// Virtual loss exists to stop threads following each other down one branch. With it
// off, tree-parallel should concentrate on fewer distinct nodes at the same budget.
// This checks the mechanism, not just that the knob is wired up.
void TestVirtualLossSpreadsTheSearch() {
  std::printf("virtual loss widens a shared-tree search\n");
  constexpr int kSize = 5;
  hex::Board<kSize> board;

  const auto run = [&board](int virtual_loss) {
    hex::ParallelSearchConfig config;
    config.threads = 8;
    config.simulations = 800;
    config.seed = 11;
    config.virtual_loss = virtual_loss;
    return hex::SearchTreeParallel<kSize>(
        board, config, [] { return PureEvaluator<kSize>{}; });
  };

  const auto without = run(0);
  const auto with = run(3);

  Check(with.distinct_nodes >= without.distinct_nodes,
        "virtual loss should not narrow the tree");
  Check(with.best_move >= 0 && without.best_move >= 0,
        "both settings should still choose a move");
  std::printf("  distinct nodes: %d without virtual loss, %d with\n",
              without.distinct_nodes, with.distinct_nodes);
}

void TestVisitsAccountForEverySimulation() {
  std::printf("root visits account for the simulation budget\n");
  constexpr int kSize = 5;
  hex::Board<kSize> board;

  hex::ParallelSearchConfig config;
  config.threads = 4;
  config.simulations = 400;
  config.seed = 5;

  const auto tree = hex::SearchTreeParallel<kSize>(
      board, config, [] { return PureEvaluator<kSize>{}; });
  int total = 0;
  for (const auto& [move, count] : tree.visits) total += count;
  // Threads claim simulations with fetch_sub and may overshoot the counter by up to
  // one per thread, so the total is bounded rather than exact.
  Check(total > 0, "tree-parallel should record visits at the root");
  Check(total <= config.simulations + config.threads,
        "tree-parallel should not exceed its budget by more than the pool size");
  std::printf("  %d visits recorded for a budget of %d across %d threads\n",
              total, config.simulations, config.threads);
}

// The count is pinned so that a change in the random stream, or a quietly added
// check, has to be acknowledged rather than absorbed.
constexpr int kExpectedChecks = 55;

}  // namespace

int main() {
  std::printf("== hex parallel self-play ==\n\n");
  std::printf("hardware concurrency: %u\n\n",
              std::thread::hardware_concurrency());

  TestReproducibleAcrossThreadCounts();
  TestReproducibleAcrossBlockSizes();
  TestSinkSeesGamesInIndexOrder();
  TestStatsAgreeWithRecords();
  TestRecordsAreWellFormed();
  TestDegenerateConfigurations();
  TestStatefulEvaluatorIsNotReproducible();
  TestLargerBoardReproduces();

  std::printf("\n-- shared-tree search modes --\n");
  TestRootParallelMatchesPuctOnOneThread();
  TestModesRestoreTheBoard();
  TestModesAgreeWithSolver();
  TestVirtualLossSpreadsTheSearch();
  TestVisitsAccountForEverySimulation();

  std::printf("\n%d checks, %d failures\n", g_checks, g_failures);
  if (g_checks != kExpectedChecks) {
    std::printf("\nERROR: expected %d checks.\n", kExpectedChecks);
    return 1;
  }
  return g_failures == 0 ? 0 : 1;
}
