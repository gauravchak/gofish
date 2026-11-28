"""Headless bot-vs-bot benchmark arena (and optional self-play dataset generator).

Run examples:
  python3 -m scripts.evaluate --bot-a tree_2ply --bot-b heuristic --games 50
  python3 -m scripts.evaluate --bot-a tree_1ply --bot-b random --games 100
  python3 -m scripts.evaluate --bot-a tree_2ply --bot-b tree_1ply --games 100 --save-trajectories selfplay.jsonl
"""

import argparse
import math
import time

from bots import create_bot
from game import GameEngine, TrajectoryLogger


def compute_luck_adjusted_stats(pair_outcomes):
  """Computes luck-adjusted skill ('excess advantage') from paired duplicate deals.

  pair_outcomes: list of (winner_seat0, winner_seat1) where each element is
  'A', 'B', or 'Draw' on the exact same shuffled deck with seats swapped!
  """
  sweeps_a = sweeps_b = luck_splits = 0
  for w0, w1 in pair_outcomes:
    s_a = (1.0 if w0 == 'A' else (0.5 if w0 == 'Draw' else 0.0)) + (
        1.0 if w1 == 'A' else (0.5 if w1 == 'Draw' else 0.0)
    )
    if s_a > 1.0:
      sweeps_a += 1
    elif s_a < 1.0:
      sweeps_b += 1
    else:
      luck_splits += 1

  total_pairs = len(pair_outcomes)
  decisive = sweeps_a + sweeps_b
  decisive_pct_a = (100.0 * sweeps_a / decisive) if decisive > 0 else 50.0
  excess_per_100 = (
      100.0 * (sweeps_a - sweeps_b) / total_pairs if total_pairs > 0 else 0.0
  )
  mcnemar_z = (
      (sweeps_a - sweeps_b) / math.sqrt(decisive) if decisive > 0 else 0.0
  )
  return {
      'total_pairs': total_pairs,
      'sweeps_a': sweeps_a,
      'sweeps_b': sweeps_b,
      'luck_splits': luck_splits,
      'decisive_deals': decisive,
      'decisive_win_pct_a': round(decisive_pct_a, 1),
      'excess_per_100': round(excess_per_100, 2),
      'mcnemar_z': round(mcnemar_z, 2),
  }


def play_match(bot_a, bot_b, seed=None, logger=None, swap_seats=False):
  """Plays one headless 2-player game between bot_a and bot_b."""
  bots = [bot_b, bot_a] if swap_seats else [bot_a, bot_b]
  engine = GameEngine(logger=logger)
  engine.join_player('bot:0', bots[0].name, player_type=bots[0].player_type)
  engine.join_player('bot:1', bots[1].name, player_type=bots[1].player_type)
  engine.start_match(seed=seed)

  move_times = {0: [], 1: []}
  while engine.winner is None:
    p = engine.turn
    obs = engine.get_observation(p)
    t0 = time.perf_counter()
    if engine.phase == 'DRAW':
      action = bots[p].choose_draw(obs)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      res = engine.draw(p, action)
    else:
      action = bots[p].choose_discard(obs)
      elapsed_ms = (time.perf_counter() - t0) * 1000.0
      res = engine.discard(p, action, by_index=False)

    if not res['ok']:
      raise RuntimeError(
          f'{bots[p].name} made illegal {engine.phase} move {action}: {res}'
      )
    move_times[p].append(elapsed_ms)

  # Map winner back to 'A', 'B', or 'Draw' regardless of seat swap.
  if engine.winner_idx is None:
    winner = 'Draw'
  elif (engine.winner_idx == 0 and not swap_seats) or (
      engine.winner_idx == 1 and swap_seats
  ):
    winner = 'A'
  else:
    winner = 'B'

  a_times = move_times[1] if swap_seats else move_times[0]
  b_times = move_times[0] if swap_seats else move_times[1]
  return {
      'winner': winner,
      'turns': engine.turn_number,
      'a_ms': a_times,
      'b_ms': b_times,
  }


def benchmark(
    bot_a_key, bot_b_key, games=40, seed=42, save_path=None, duplicate=True
):
  """Runs N games (default: paired duplicate deals) and reports Luck-Adjusted Skill."""
  bot_a = create_bot(bot_a_key)
  bot_b = create_bot(bot_b_key)
  logger = TrajectoryLogger(save_path) if save_path else None
  if duplicate and games % 2 == 1:
    games += 1

  wins_a = wins_b = draws = 0
  total_turns = 0
  all_a_ms = []
  all_b_ms = []
  pair_outcomes = []
  first_winner = None

  for i in range(games):
    swap = (i % 2 == 1)
    deal_seed = seed + (i // 2) if duplicate else seed + i
    result = play_match(
        bot_a, bot_b, seed=deal_seed, logger=logger, swap_seats=swap
    )
    if result['winner'] == 'A':
      wins_a += 1
    elif result['winner'] == 'B':
      wins_b += 1
    else:
      draws += 1
    if duplicate:
      if not swap:
        first_winner = result['winner']
      else:
        pair_outcomes.append((first_winner, result['winner']))
    total_turns += result['turns']
    all_a_ms.extend(result['a_ms'])
    all_b_ms.extend(result['b_ms'])

  avg_turns = total_turns / max(games, 1)
  avg_a_ms = sum(all_a_ms) / max(len(all_a_ms), 1)
  avg_b_ms = sum(all_b_ms) / max(len(all_b_ms), 1)

  mode = 'paired duplicate deals' if duplicate else 'independent deals'
  print(f'=== {bot_a.name} vs {bot_b.name} ({games} games, {mode}) ===')
  print(f'  {bot_a.name:36s}: {wins_a:4d} wins ({100.0 * wins_a / games:5.1f}%) | {avg_a_ms:5.2f} ms/move')
  print(f'  {bot_b.name:36s}: {wins_b:4d} wins ({100.0 * wins_b / games:5.1f}%) | {avg_b_ms:5.2f} ms/move')
  print(f'  Draws                               : {draws:4d}      ({100.0 * draws / games:5.1f}%)')
  print(f'  Average turns per game              : {avg_turns:.1f}')
  if duplicate and pair_outcomes:
    stats = compute_luck_adjusted_stats(pair_outcomes)
    print(
        f'  Luck Splits (1-1, deal-dominated)   : {stats["luck_splits"]:4d} / '
        f'{stats["total_pairs"]} deals ({100.0 * stats["luck_splits"] / stats["total_pairs"]:.1f}%)'
    )
    print(
        f'  Skill Sweeps (2-0 vs 0-2)           : {stats["sweeps_a"]} vs {stats["sweeps_b"]} '
        f'(Decisive Skill Win%: {stats["decisive_win_pct_a"]:.1f}% | '
        f'Excess Advantage: {stats["excess_per_100"]:+.2f}/100 deals | '
        f'McNemar z={stats["mcnemar_z"]:+.2f})'
    )
  if save_path:
    print(f'  Saved trajectories to               : {save_path} ({logger.saved_games} games total)')


def main():
  parser = argparse.ArgumentParser(description='Benchmark GoFish computer players.')
  bot_choices = [
      'random',
      'novice',
      'heuristic',
      'tree_1ply',
      'tree_2ply',
      'rl_td_value',
      'rl_q_learning',
      'rl_policy_gradient',
      'rl_grpo_xray',
      'rl_distilled',
      'rl_ensemble_soft',
      'rl_ensemble_mode',
      'rl_alphazero',
  ]
  parser.add_argument('--bot-a', default='tree_1ply', choices=bot_choices)
  parser.add_argument('--bot-b', default='heuristic', choices=bot_choices)
  parser.add_argument('--games', type=int, default=40)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--no-duplicate', action='store_true')
  parser.add_argument('--save-trajectories', default=None)
  args = parser.parse_args()
  benchmark(
      args.bot_a,
      args.bot_b,
      games=args.games,
      seed=args.seed,
      save_path=args.save_trajectories,
      duplicate=not args.no_duplicate,
  )


if __name__ == '__main__':
  main()
