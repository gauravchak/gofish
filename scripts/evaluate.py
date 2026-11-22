"""Headless bot-vs-bot benchmark arena (and optional self-play dataset generator).

Run examples:
  python3 -m scripts.evaluate --bot-a tree_2ply --bot-b heuristic --games 50
  python3 -m scripts.evaluate --bot-a tree_1ply --bot-b random --games 100
  python3 -m scripts.evaluate --bot-a tree_2ply --bot-b tree_1ply --games 100 --save-trajectories selfplay.jsonl
"""

import argparse
import time

from bots import create_bot
from game import GameEngine, TrajectoryLogger


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


def benchmark(bot_a_key, bot_b_key, games=40, seed=42, save_path=None):
  """Runs N games (alternating who goes first) and prints win rates and timing."""
  bot_a = create_bot(bot_a_key)
  bot_b = create_bot(bot_b_key)
  logger = TrajectoryLogger(save_path) if save_path else None

  wins_a = wins_b = draws = 0
  total_turns = 0
  all_a_ms = []
  all_b_ms = []

  for i in range(games):
    # Alternate first-player advantage so the comparison is fair.
    result = play_match(
        bot_a, bot_b, seed=seed + i, logger=logger, swap_seats=(i % 2 == 1)
    )
    if result['winner'] == 'A':
      wins_a += 1
    elif result['winner'] == 'B':
      wins_b += 1
    else:
      draws += 1
    total_turns += result['turns']
    all_a_ms.extend(result['a_ms'])
    all_b_ms.extend(result['b_ms'])

  avg_turns = total_turns / max(games, 1)
  avg_a_ms = sum(all_a_ms) / max(len(all_a_ms), 1)
  avg_b_ms = sum(all_b_ms) / max(len(all_b_ms), 1)

  print(f'=== {bot_a.name} vs {bot_b.name} ({games} games, alternating seats) ===')
  print(f'  {bot_a.name:28s}: {wins_a:3d} wins ({100.0 * wins_a / games:5.1f}%) | {avg_a_ms:5.2f} ms/move')
  print(f'  {bot_b.name:28s}: {wins_b:3d} wins ({100.0 * wins_b / games:5.1f}%) | {avg_b_ms:5.2f} ms/move')
  print(f'  Draws                       : {draws:3d}      ({100.0 * draws / games:5.1f}%)')
  print(f'  Average turns per game      : {avg_turns:.1f}')
  if save_path:
    print(f'  Saved trajectories to       : {save_path} ({logger.saved_games} games total)')


def main():
  parser = argparse.ArgumentParser(description='Benchmark GoFish computer players.')
  bot_choices = ['random', 'novice', 'heuristic', 'tree_1ply', 'tree_2ply']
  parser.add_argument('--bot-a', default='tree_1ply', choices=bot_choices)
  parser.add_argument('--bot-b', default='heuristic', choices=bot_choices)
  parser.add_argument('--games', type=int, default=40)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument('--save-trajectories', default=None)
  args = parser.parse_args()
  benchmark(
      args.bot_a,
      args.bot_b,
      games=args.games,
      seed=args.seed,
      save_path=args.save_trajectories,
  )


if __name__ == '__main__':
  main()
