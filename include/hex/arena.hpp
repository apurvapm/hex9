#ifndef HEX_ARENA_HPP
#define HEX_ARENA_HPP

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <thread>
#include <vector>

#include "hex/board.hpp"
#include "hex/puct.hpp"

namespace hex {

// Head-to-head matches between two agents, and the Elo that follows.
//
// Colours are alternated in pairs and the pairing is not optional. Red wins about
// 53% of games under uniform random play, so an unpaired match measures which
// agent got the first move rather than which one is stronger: a 53% result would
// read as a 3% improvement. Every opening is therefore played twice, once with
// each agent as Red, and only the pair total is reported.
//
// Root noise is off. It exists to widen self-play openings; in evaluation it adds
// variance to the thing being measured. Opening variety instead comes from a
// per-pair temperature applied for the first few plies, so the two games of a pair
// share an opening line and differ only in who plays it.
struct ArenaConfig {
  int pairs = 50;
  int simulations = 200;
  // Search budget for the second agent; zero means the same as the first. Giving
  // the two sides different budgets is what makes the arena testable: with
  // identical agents every pair splits one win and one loss by construction, so a
  // 50% result proves the pairing arithmetic and nothing else. More search beating
  // less is a difference the arena must be able to detect.
  int simulations_second = 0;
  float c_puct = 1.5f;
  // Plies sampled at temperature 1 before play turns greedy. Without this every
  // pair would replay one identical opening and the match would measure a single
  // line rather than an agent.
  int temperature_moves = 6;
  std::uint64_t seed = 1;
  bool allow_swap = true;
  int threads = 1;
};

struct ArenaResult {
  int games = 0;
  int wins = 0;    // for the first agent
  int losses = 0;
  // Hex has no draws, so there is no draw field. A missing category is easier to
  // trust than one that is always zero.
  int first_agent_as_red_wins = 0;
  int first_agent_as_blue_wins = 0;
  int red_wins_overall = 0;

  double WinRate() const {
    return games > 0 ? static_cast<double>(wins) / games : 0.0;
  }
};

// Elo difference implied by a win rate. Positive means the first agent is stronger.
//
// Returns +/-kEloCap at a clean sweep rather than infinity: 20 games proving "at
// least this much stronger" is a real result, and an infinity in a results table
// is worse than a bound. The cap is reported as a bound by callers.
inline constexpr double kEloCap = 800.0;

inline double EloDifference(double win_rate) {
  if (win_rate <= 0.0) return -kEloCap;
  if (win_rate >= 1.0) return kEloCap;
  const double elo = -400.0 * std::log10(1.0 / win_rate - 1.0);
  return std::clamp(elo, -kEloCap, kEloCap);
}

// Standard error of the win rate, in Elo, from the normal approximation. Reported
// alongside every Elo number because a 50-game match has an error bar wide enough
// to swallow most of the differences worth caring about.
inline double EloStandardError(double win_rate, int games) {
  if (games <= 0) return kEloCap;
  const double clamped = std::clamp(win_rate, 0.5 / games, 1.0 - 0.5 / games);
  const double variance = clamped * (1.0 - clamped) / games;
  // d(Elo)/d(p) = 400 / (ln(10) * p * (1 - p))
  const double slope = 400.0 / (std::log(10.0) * clamped * (1.0 - clamped));
  return slope * std::sqrt(variance);
}

namespace detail {

// One game between two evaluators. `first_is_red` decides colours; the return is
// +1 when the first agent won.
template <int N, typename EvalA, typename EvalB>
int PlayArenaGame(const ArenaConfig& config, std::uint64_t game_seed,
                  bool first_is_red, EvalA& first, EvalB& second) {
  Board<N> board;

  typename Puct<N>::Config search_config{};
  search_config.simulations = config.simulations;
  search_config.c_puct = config.c_puct;
  search_config.dirichlet_weight = 0.0f;
  search_config.allow_swap = config.allow_swap;

  // Both agents draw their sampling from one generator seeded per game, so the two
  // games of a pair follow the same opening distribution.
  search_config.seed = game_seed;
  Puct<N> search_first(search_config);

  if (config.simulations_second > 0)
    search_config.simulations = config.simulations_second;
  Puct<N> search_second(search_config);

  while (!board.IsTerminal()) {
    const bool red_to_play = board.ToPlay() == Player::kRed;
    const bool first_to_play = red_to_play == first_is_red;
    const float temperature =
        board.MoveCount() < config.temperature_moves ? 1.0f : 0.0f;

    int move = -1;
    if (first_to_play) {
      const auto result = search_first.Search(board, first);
      move = search_first.SampleMove(result, temperature);
    } else {
      const auto result = search_second.Search(board, second);
      move = search_second.SampleMove(result, temperature);
    }

    if (move == Board<N>::kSwapMove) {
      board.PlaySwap();
    } else {
      board.Play(move);
    }
  }

  const bool red_won = board.Winner() == Cell::kRed;
  return red_won == first_is_red ? 1 : -1;
}

}  // namespace detail

// Plays `pairs` colour-balanced pairs. MakeA and MakeB are factories so each
// worker owns its own evaluator, matching the self-play driver: an ORT session
// belongs to the thread that runs it.
template <int N, typename MakeA, typename MakeB>
ArenaResult RunArena(const ArenaConfig& config, MakeA make_first,
                     MakeB make_second) {
  const int threads = std::max(1, config.threads);
  ArenaResult result;
  if (config.pairs <= 0) return result;

  // Two games per pair, indexed so that game 2k and 2k+1 are the same opening
  // seed with colours exchanged.
  const int total = config.pairs * 2;
  std::vector<int> outcomes(static_cast<std::size_t>(total), 0);
  std::vector<char> first_was_red(static_cast<std::size_t>(total), 0);
  std::vector<char> red_won(static_cast<std::size_t>(total), 0);
  std::atomic<int> next{0};

  const auto worker = [&] {
    auto first = make_first();
    auto second = make_second();
    for (;;) {
      const int index = next.fetch_add(1, std::memory_order_acq_rel);
      if (index >= total) break;

      const int pair = index / 2;
      const bool first_is_red = index % 2 == 0;
      const std::uint64_t game_seed =
          config.seed * 6364136223846793005ULL + static_cast<std::uint64_t>(pair);

      const int outcome = detail::PlayArenaGame<N>(config, game_seed,
                                                   first_is_red, first, second);
      outcomes[static_cast<std::size_t>(index)] = outcome;
      first_was_red[static_cast<std::size_t>(index)] = first_is_red ? 1 : 0;
      // The first agent won iff outcome is +1; it played Red iff first_is_red.
      // Red therefore won when those agree.
      red_won[static_cast<std::size_t>(index)] =
          (outcome == 1) == first_is_red ? 1 : 0;
    }
  };

  std::vector<std::thread> pool;
  pool.reserve(static_cast<std::size_t>(threads));
  for (int i = 0; i < threads; ++i) pool.emplace_back(worker);
  for (std::thread& thread : pool) thread.join();

  for (int index = 0; index < total; ++index) {
    const int outcome = outcomes[static_cast<std::size_t>(index)];
    ++result.games;
    if (outcome == 1) {
      ++result.wins;
      if (first_was_red[static_cast<std::size_t>(index)] != 0) {
        ++result.first_agent_as_red_wins;
      } else {
        ++result.first_agent_as_blue_wins;
      }
    } else {
      ++result.losses;
    }
    if (red_won[static_cast<std::size_t>(index)] != 0) ++result.red_wins_overall;
  }
  return result;
}

}  // namespace hex

#endif  // HEX_ARENA_HPP
