"""Official 5,000-Game Luck-Adjusted Strategy Advantage Leaderboard Generator.

Measures how much EACH strategy beats a fixed Standard Baseline Pool
(50% Heuristic Bot + 50% Tree Bot 1-Level) in EXCESS OF CARD LUCK using
paired duplicate deals (exact same shuffled deck played from both seats!).

For every strategy, reports:
  - Excess Advantage / 100 deals: 100 * (sweeps_2_0 - sweeps_0_2) / total_pairs
  - Decisive Skill Win%:          100 * sweeps_2_0 / (sweeps_2_0 + sweeps_0_2)
  - Luck Splits (1-1):            deals where card luck dominated regardless of bot
  - 5k Win% (+/- paired SE) & McNemar z-score

Saves results to models/advantage_leaderboard.json so http://localhost:7171/league
always displays the exact official Advantage Leaderboard!

Run:
  python3 -m scripts.benchmark_advantage --pairs-per-opp 1250
"""

import argparse
import json
import math
import multiprocessing as mp
import os
import time

import torch

from bots import MODELS_DIR, create_bot
from game import GameEngine

ADVANTAGE_LEADERBOARD_PATH = os.path.join(
    MODELS_DIR, 'advantage_leaderboard.json'
)


def _worker_eval_pairs(args):
  """Plays a chunk of paired duplicate deals in a worker process."""
  torch.set_num_threads(1)
  bot_key, opp_key, seed_start, n_pairs = args
  bot = create_bot(bot_key)
  opp = create_bot(opp_key)

  wins = losses = draws = 0
  sweeps_w = sweeps_l = splits = 0
  pair_scores = []

  for idx in range(n_pairs):
    deal_seed = seed_start + idx
    p_score = 0.0
    for swap in (0, 1):
      b0, b1 = (opp, bot) if swap else (bot, opp)
      eng = GameEngine(logger=None)
      eng.join_player('p0', b0.name, player_type=b0.player_type)
      eng.join_player('p1', b1.name, player_type=b1.player_type)
      eng.start_match(seed=deal_seed)
      while eng.winner is None:
        p = eng.turn
        obs = eng.get_observation(p)
        actor = b0 if p == 0 else b1
        if eng.phase == 'DRAW':
          eng.draw(p, actor.choose_draw(obs))
        else:
          eng.discard(p, actor.choose_discard(obs), by_index=False)
      seat = 1 if swap else 0
      if eng.winner_idx == seat:
        wins += 1
        p_score += 1.0
      elif eng.winner_idx is None:
        draws += 1
        p_score += 0.5
      else:
        losses += 1
    pair_scores.append(p_score / 2.0)
    if p_score > 1.0:
      sweeps_w += 1
    elif p_score < 1.0:
      sweeps_l += 1
    else:
      splits += 1

  return wins, losses, draws, sweeps_w, sweeps_l, splits, pair_scores


def _run_parallel_matchup(pool, bot_key, opp_key, n_pairs, seed_base, n_workers):
  """Evaluates bot_key vs opp_key over n_pairs mirror pairs across worker processes."""
  if bot_key == opp_key:
    # Exact symmetry: any bot against itself has 0.0 excess and 100% 1-1 splits.
    return {
        'wins': n_pairs,
        'losses': n_pairs,
        'draws': 0,
        'sweeps_w': 0,
        'sweeps_l': 0,
        'splits': n_pairs,
        'pair_scores': [0.5] * n_pairs,
    }

  chunk = n_pairs // n_workers
  rem = n_pairs % n_workers
  tasks = []
  cur = seed_base
  for i in range(n_workers):
    cnt = chunk + (1 if i < rem else 0)
    if cnt > 0:
      tasks.append((bot_key, opp_key, cur, cnt))
      cur += cnt

  parts = pool.map(_worker_eval_pairs, tasks)
  return {
      'wins': sum(x[0] for x in parts),
      'losses': sum(x[1] for x in parts),
      'draws': sum(x[2] for x in parts),
      'sweeps_w': sum(x[3] for x in parts),
      'sweeps_l': sum(x[4] for x in parts),
      'splits': sum(x[5] for x in parts),
      'pair_scores': [s for x in parts for s in x[6]],
  }


def _summarize_entry(bot_key, name, res_h, res_t):
  """Combines vs-Heuristic and vs-Tree1 duplicate results into Advantage metrics."""
  scores_h = res_h['pair_scores']
  scores_t = res_t['pair_scores']
  all_scores = scores_h + scores_t
  n_pairs = len(all_scores)
  games = n_pairs * 2

  mean_h = sum(scores_h) / max(len(scores_h), 1)
  mean_t = sum(scores_t) / max(len(scores_t), 1)
  mean_all = sum(all_scores) / max(n_pairs, 1)
  var_all = sum((s - mean_all) ** 2 for s in all_scores) / max(n_pairs - 1, 1)
  se_all = math.sqrt(var_all / max(n_pairs, 1))

  sweeps_w = res_h['sweeps_w'] + res_t['sweeps_w']
  sweeps_l = res_h['sweeps_l'] + res_t['sweeps_l']
  splits = res_h['splits'] + res_t['splits']
  decisive = sweeps_w + sweeps_l
  decisive_pct = (100.0 * sweeps_w / decisive) if decisive > 0 else 50.0
  excess_per_100 = 100.0 * (sweeps_w - sweeps_l) / max(n_pairs, 1)
  mcnemar_z = (sweeps_w - sweeps_l) / math.sqrt(decisive) if decisive > 0 else 0.0

  if bot_key.startswith('rl_ensemble'):
    category = 'ensemble'
  elif bot_key.startswith('rl_'):
    category = 'rl'
  elif bot_key in ('heuristic', 'tree_1ply'):
    category = 'baseline'
  else:
    category = 'starter'

  return {
      'key': bot_key,
      'name': name,
      'category': category,
      'games': games,
      'pairs': n_pairs,
      'win_pct': round(100.0 * mean_all, 2),
      'se_pct': round(100.0 * se_all, 2),
      'vs_heuristic_pct': round(100.0 * mean_h, 2),
      'vs_tree1_pct': round(100.0 * mean_t, 2),
      'sweeps_win': sweeps_w,
      'sweeps_loss': sweeps_l,
      'luck_splits': splits,
      'decisive_win_pct': round(decisive_pct, 2),
      'excess_per_100': round(excess_per_100, 2),
      'mcnemar_z': round(mcnemar_z, 2),
  }


def run_benchmark(pairs_per_opp=1250, seed=500000):
  """Runs the official 5,000-game Advantage Leaderboard benchmark and saves JSON."""
  t0 = time.time()
  n_workers = min(6, max(2, mp.cpu_count() - 2))

  bot_keys = [
      'rl_q_learning',
      'rl_td_value',
      'rl_ensemble_soft',
      'rl_ensemble_mode',
      'rl_grpo_xray',
      'rl_policy_gradient',
      'rl_distilled',
      'tree_1ply',
      'heuristic',
      'novice',
      'random',
  ]

  print(
      f'=== Official Strategy Advantage Leaderboard ({pairs_per_opp*4:,} games/bot: '
      f'{pairs_per_opp*2:,}g vs Heuristic + {pairs_per_opp*2:,}g vs Tree 1-Level, '
      f'{n_workers} workers) ===',
      flush=True,
  )

  entries = []
  with mp.Pool(processes=n_workers) as pool:
    # Compute Heuristic vs Tree1 once and mirror it cleanly!
    heur_vs_t1 = _run_parallel_matchup(
        pool, 'heuristic', 'tree_1ply', pairs_per_opp, seed + 100000, n_workers
    )
    t1_vs_heur = {
        'wins': heur_vs_t1['losses'],
        'losses': heur_vs_t1['wins'],
        'draws': heur_vs_t1['draws'],
        'sweeps_w': heur_vs_t1['sweeps_l'],
        'sweeps_l': heur_vs_t1['sweeps_w'],
        'splits': heur_vs_t1['splits'],
        'pair_scores': [1.0 - s for s in heur_vs_t1['pair_scores']],
    }
    self_tie = {
        'wins': pairs_per_opp,
        'losses': pairs_per_opp,
        'draws': 0,
        'sweeps_w': 0,
        'sweeps_l': 0,
        'splits': pairs_per_opp,
        'pair_scores': [0.5] * pairs_per_opp,
    }

    for key in bot_keys:
      tb = time.time()
      name = create_bot(key).name
      if key == 'heuristic':
        res_h, res_t = self_tie, heur_vs_t1
      elif key == 'tree_1ply':
        res_h, res_t = t1_vs_heur, self_tie
      else:
        res_h = _run_parallel_matchup(
            pool, key, 'heuristic', pairs_per_opp, seed, n_workers
        )
        res_t = _run_parallel_matchup(
            pool, key, 'tree_1ply', pairs_per_opp, seed + 100000, n_workers
        )
      entry = _summarize_entry(key, name, res_h, res_t)
      entries.append(entry)
      print(
          f'  {entry["name"]:40s} | Excess: {entry["excess_per_100"]:+6.2f}/100 | '
          f'Decisive: {entry["decisive_win_pct"]:5.1f}% ({entry["sweeps_win"]}-{entry["sweeps_loss"]}) | '
          f'5k: {entry["win_pct"]:5.2f}%±{entry["se_pct"]:.2f}% | z={entry["mcnemar_z"]:+.2f} '
          f'[{time.time()-tb:.1f}s]',
          flush=True,
      )

  # Rank strictly by Excess Advantage over Baseline (tie-break: 5k win_pct).
  entries.sort(key=lambda e: (e['excess_per_100'], e['win_pct']), reverse=True)
  for rank, e in enumerate(entries, 1):
    e['rank'] = rank

  payload = {
      'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
      'baseline_description': 'Standard Baseline Pool (50% Heuristic Bot + 50% Tree Bot 1-Level)',
      'games_per_bot': pairs_per_opp * 4,
      'pairs_per_bot': pairs_per_opp * 2,
      'entries': entries,
  }
  os.makedirs(MODELS_DIR, exist_ok=True)
  with open(ADVANTAGE_LEADERBOARD_PATH, 'w', encoding='utf-8') as f:
    json.dump(payload, f, indent=2)
    f.write('\n')

  print(f'\nSaved Official Advantage Leaderboard -> {ADVANTAGE_LEADERBOARD_PATH} ({time.time()-t0:.1f}s)')
  return payload


def main():
  parser = argparse.ArgumentParser(
      description='Generate the official 5,000-game Luck-Adjusted Advantage Leaderboard.'
  )
  parser.add_argument('--pairs-per-opp', type=int, default=1250)
  parser.add_argument('--seed', type=int, default=500000)
  args = parser.parse_args()
  run_benchmark(pairs_per_opp=args.pairs_per_opp, seed=args.seed)


if __name__ == '__main__':
  main()
