"""Train all three RL models via 10,000 progressive Elo self-play games.

Trains side-by-side:
  1. TDValueBot         (TD(lambda) afterstate value learning)
  2. QLearningBot       (DQN-style Q-learning with target networks)
  3. PolicyGradientBot  (Actor-Critic REINFORCE with value baseline)

Curriculum (10,000 games total by default):
  - Stage 1 (0%..20%):   Warm-up vs Random, Beginner, and peer RL learners
  - Stage 2 (20%..50%):  Climb vs Beginner, Heuristic, and peer RL learners
  - Stage 3 (50%..80%):  Top-Elo pool (Heuristic, 1-Level Tree, best RL peers)
  - Stage 4 (80%..100%): Champions league (top-3 Elo players + 1-Level Tree)

Run:
  python3 -m scripts.train_rl --games 10000 --save-trajectories selfplay_10k.jsonl.gz
"""

import argparse
import os
import random
import time

import torch

from bots import BOT_STARTING_ELO, RL_MODEL_PATHS, create_bot
from bots.heuristic import get_next_player_hints, get_unseen_mask
from game import GameEngine, TrajectoryLogger, card_to_id, expected_score
from ml.features import AFTERSTATE_DIM, extract_afterstate_features
from ml.policy_gradient import PolicyGradientBot
from ml.q_learning import QLearningBot
from ml.td_value import TDValueBot


def _update_elo(elos, key_a, key_b, winner_slot, k=24.0):
  """Updates Elo dictionary in place after one 2-player game (winner_slot: 0, 1, or None)."""
  ra, rb = elos[key_a], elos[key_b]
  ea = expected_score(ra, rb)
  score_a = 1.0 if winner_slot == 0 else (0.0 if winner_slot == 1 else 0.5)
  delta = k * (score_a - ea)
  elos[key_a] = ra + delta
  elos[key_b] = rb - delta


def _afterstate_from_discard(obs, discarded_card):
  """Extracts the 16-dim post-discard feature vector for any player's discard."""
  cid = card_to_id(discarded_card)
  hand7 = obs['hand_ids']
  idx = hand7.index(cid)
  h6 = hand7[:idx] + hand7[idx + 1 :]
  unseen_mask = get_unseen_mask(obs)
  next_wanted, next_discarded = get_next_player_hints(obs)
  return extract_afterstate_features(
      h6,
      unseen_mask,
      obs['deck_remaining'],
      next_wanted,
      next_discarded,
      discarded_id=cid,
      took_shown=obs.get('last_picked_shown_id') is not None,
  )


def _play_training_game(bot0, bot1, key0, key1, seed, logger, lr, actor_lr):
  """Plays one 2-player game and updates any RL learners participating."""
  engine = GameEngine(logger=logger)
  engine.join_player('p0', bot0.name, player_type=bot0.player_type)
  engine.join_player('p1', bot1.name, player_type=bot1.player_type)
  engine.start_match(seed=seed)

  bots = [bot0, bot1]
  has_td = any(isinstance(b, TDValueBot) for b in bots)
  has_pg = any(isinstance(b, PolicyGradientBot) for b in bots)

  # Per-seat episode buffers tailored to each RL algorithm.
  all_afterstates = [[], []]  # Balanced win/loss afterstates for TD Value & PG Critic
  q_transitions = [[], []]
  pg_steps = [[], []]

  while engine.winner is None:
    p = engine.turn
    bot = bots[p]
    obs = engine.get_observation(p)

    if engine.phase == 'DRAW':
      if isinstance(bot, QLearningBot):
        action, rec = bot.choose_draw(obs, return_record=True)
        q_transitions[p].append(rec)
      elif isinstance(bot, PolicyGradientBot):
        action, rec = bot.choose_draw(obs, return_record=True)
        pg_steps[p].append(rec)
      else:
        action = bot.choose_draw(obs)
      res = engine.draw(p, action)
    else:
      if isinstance(bot, TDValueBot):
        action, feats = bot.choose_discard(obs, return_features=True)
        all_afterstates[p].append(feats)
      elif isinstance(bot, QLearningBot):
        action, rec = bot.choose_discard(obs, return_record=True)
        q_transitions[p].append(rec)
        if has_td or has_pg:
          all_afterstates[p].append(rec['feats'][:AFTERSTATE_DIM])
      elif isinstance(bot, PolicyGradientBot):
        action, rec = bot.choose_discard(obs, return_record=True)
        pg_steps[p].append(rec)
        all_afterstates[p].append(rec['after_feats'])
      else:
        action = bot.choose_discard(obs)
        if has_td or has_pg:
          all_afterstates[p].append(_afterstate_from_discard(obs, action))
      res = engine.discard(p, action, by_index=False)

    if not res['ok']:
      raise RuntimeError(f'{bot.name} made illegal move {action}: {res}')

  # Perform RL updates for each participating RL learner.
  for seat, bot in enumerate(bots):
    other = 1 - seat
    if engine.winner_idx is None:
      outcome_01, reward_pm = 0.5, 0.0
      opp_01, opp_pm = 0.5, 0.0
    elif engine.winner_idx == seat:
      outcome_01, reward_pm = 1.0, 1.0
      opp_01, opp_pm = 0.0, -1.0
    else:
      outcome_01, reward_pm = 0.0, -1.0
      opp_01, opp_pm = 1.0, 1.0

    if isinstance(bot, TDValueBot):
      # Learn from BOTH seats (50% wins, 50% losses) to prevent pessimistic collapse!
      bot.learn_episode(all_afterstates[seat], outcome_01, lr=lr)
      bot.learn_episode(all_afterstates[other], opp_01, lr=lr)
    elif isinstance(bot, QLearningBot):
      bot.learn_episode(q_transitions[seat], reward_pm, lr=lr)
    elif isinstance(bot, PolicyGradientBot):
      # Train Critic on opponent's afterstates too for a well-calibrated baseline,
      # then run Advantage Actor-Critic on our own episode!
      bot.train_critic_only(all_afterstates[other], opp_pm, critic_lr=lr)
      bot.learn_episode(
          pg_steps[seat], reward_pm, actor_lr=actor_lr, critic_lr=lr
      )

  return engine.winner_idx


def _pick_opponent(stage, learner_key, live_elos, rng):
  """Selects a progressively harder opponent guided by live Elo rankings."""
  rl_peers = [
      k for k in ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient')
      if k != learner_key
  ]

  if stage == 1:
    # Stage 1: Welcoming warm-up vs Random, Beginner, and fellow learners.
    pool = ['random', 'novice', 'novice'] + rl_peers
    return rng.choice(pool)

  if stage == 2:
    # Stage 2: Step up to Beginner, Heuristic, and fellow learners.
    pool = ['novice', 'heuristic', 'heuristic', 'heuristic'] + rl_peers
    return rng.choice(pool)

  # Stage 3 & 4: Rank candidates by live Elo and bias toward the strongest players!
  candidates = ['heuristic', 'tree_1ply'] + rl_peers
  candidates.sort(key=lambda k: live_elos[k], reverse=True)

  if stage == 3:
    # Bias toward top Elo opponents (keep tree_1ply rate ~8% so 10k games stay snappy)
    if rng.random() < 0.08:
      return 'tree_1ply'
    fast_ranked = [k for k in candidates if k != 'tree_1ply']
    return fast_ranked[0] if rng.random() < 0.5 else rng.choice(fast_ranked)

  # Stage 4: Champions league — best Elo opponents + 12% 1-Level Tree sparring.
  if rng.random() < 0.12:
    return 'tree_1ply'
  fast_ranked = [k for k in candidates if k != 'tree_1ply']
  return fast_ranked[0] if rng.random() < 0.65 else rng.choice(fast_ranked[:2])


def run_eval_tournament(pool, keys, games_per_pair=30, seed=90000):
  """Fair round-robin tournament across all bots (alternating first player)."""
  # Freeze exploration during evaluation.
  for key in keys:
    bot = pool[key]
    if hasattr(bot, 'epsilon'):
      bot.epsilon = 0.0
    if hasattr(bot, 'stochastic'):
      bot.stochastic = False

  eval_elos = {k: 1000.0 for k in keys}
  wins = {k: 0 for k in keys}
  games_played = {k: 0 for k in keys}
  vs_heuristic = {k: [0, 0] for k in keys}  # [wins, total]

  match_idx = 0
  for i, key_a in enumerate(keys):
    for key_b in keys[i + 1 :]:
      # Keep 2-ply matches crisp (12 games each) and all others at games_per_pair.
      n_games = (
          min(games_per_pair, 12)
          if ('tree_2ply' in (key_a, key_b))
          else games_per_pair
      )
      for g in range(n_games):
        swap = (g % 2) == 1
        k0, k1 = (key_b, key_a) if swap else (key_a, key_b)
        eng = GameEngine(logger=None)
        eng.join_player('p0', pool[k0].name, player_type=pool[k0].player_type)
        eng.join_player('p1', pool[k1].name, player_type=pool[k1].player_type)
        eng.start_match(seed=seed + match_idx)
        match_idx += 1

        b0, b1 = pool[k0], pool[k1]
        while eng.winner is None:
          p = eng.turn
          obs = eng.get_observation(p)
          bot = b0 if p == 0 else b1
          if eng.phase == 'DRAW':
            eng.draw(p, bot.choose_draw(obs))
          else:
            eng.discard(p, bot.choose_discard(obs), by_index=False)

        _update_elo(eval_elos, k0, k1, eng.winner_idx, k=20.0)
        games_played[k0] += 1
        games_played[k1] += 1
        if eng.winner_idx == 0:
          wins[k0] += 1
        elif eng.winner_idx == 1:
          wins[k1] += 1

        if k0 == 'heuristic':
          vs_heuristic[k1][1] += 1
          if eng.winner_idx == 1:
            vs_heuristic[k1][0] += 1
        elif k1 == 'heuristic':
          vs_heuristic[k0][1] += 1
          if eng.winner_idx == 0:
            vs_heuristic[k0][0] += 1

  return eval_elos, wins, games_played, vs_heuristic


def train_all(total_games=10000, save_trajectories=None, seed=42):
  """Runs the 4-stage progressive curriculum for all three PyTorch RL models."""
  # For tiny per-move MLPs on CPU, 1 thread avoids thread-sync overhead (3-5x faster!).
  torch.set_num_threads(1)
  rng = random.Random(seed)
  t_start = time.time()

  # Fresh PyTorch RL learners initialized from scratch!
  td_bot = TDValueBot(hidden_dim=32, seed=seed + 1)
  q_bot = QLearningBot(hidden_dim=32, seed=seed + 2)
  pg_bot = PolicyGradientBot(hidden_dim=32, seed=seed + 3)

  pool = {
      'random': create_bot('random'),
      'novice': create_bot('novice'),
      'heuristic': create_bot('heuristic'),
      'tree_1ply': create_bot('tree_1ply'),
      'tree_2ply': create_bot('tree_2ply'),
      'rl_td_value': td_bot,
      'rl_q_learning': q_bot,
      'rl_policy_gradient': pg_bot,
  }

  live_elos = {
      'random': BOT_STARTING_ELO['bot:random'],
      'novice': BOT_STARTING_ELO['bot:novice'],
      'heuristic': BOT_STARTING_ELO['bot:heuristic'],
      'tree_1ply': BOT_STARTING_ELO['bot:tree_1ply'],
      'rl_td_value': 900.0,
      'rl_q_learning': 900.0,
      'rl_policy_gradient': 900.0,
  }

  if save_trajectories and os.path.exists(save_trajectories):
    os.remove(save_trajectories)
  logger = (
      TrajectoryLogger(save_trajectories, compact=True)
      if save_trajectories
      else None
  )

  learners = ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient')
  print(f'Starting {total_games:,}-game progressive Elo RL training...')

  for game_num in range(1, total_games + 1):
    progress = game_num / total_games
    # Adam learning-rate schedule across the 4 progressive curriculum stages.
    if progress <= 0.20:
      stage, eps, temp, lr, actor_lr = 1, 0.20, 1.20, 0.0040, 0.0030
    elif progress <= 0.50:
      stage, eps, temp, lr, actor_lr = 2, 0.12, 0.90, 0.0025, 0.0020
    elif progress <= 0.80:
      stage, eps, temp, lr, actor_lr = 3, 0.06, 0.65, 0.0012, 0.0010
    else:
      stage, eps, temp, lr, actor_lr = 4, 0.02, 0.45, 0.0005, 0.0004

    td_bot.epsilon = eps
    q_bot.epsilon = eps
    pg_bot.stochastic = True
    pg_bot.temperature = temp

    learner_key = learners[(game_num - 1) % len(learners)]
    opp_key = _pick_opponent(stage, learner_key, live_elos, rng)

    # Alternate starting seat so first-player advantage stays balanced.
    if game_num % 2 == 0:
      k0, k1 = learner_key, opp_key
    else:
      k0, k1 = opp_key, learner_key

    winner_slot = _play_training_game(
        pool[k0],
        pool[k1],
        k0,
        k1,
        seed=seed * 1000 + game_num,
        logger=logger,
        lr=lr,
        actor_lr=actor_lr,
    )
    _update_elo(live_elos, k0, k1, winner_slot, k=16.0)

    checkpoint = max(total_games // 10, 1)
    if game_num % checkpoint == 0 or game_num == total_games:
      elapsed = time.time() - t_start
      ranked = sorted(live_elos.items(), key=lambda item: item[1], reverse=True)
      summary = ' | '.join(f'{k}:{v:.0f}' for k, v in ranked[:5])
      print(
          f'  [Game {game_num:5d}/{total_games} | Stage {stage} | {elapsed:5.1f}s] '
          f'Top Elo -> {summary}',
          flush=True,
      )

  # Save all three trained models right away before running the tournament.
  for key, bot in (
      ('rl_td_value', td_bot),
      ('rl_q_learning', q_bot),
      ('rl_policy_gradient', pg_bot),
  ):
    bot.save(RL_MODEL_PATHS[key], metadata={'trained_games': total_games})

  # Run final fair round-robin evaluation tournament across all 8 bots.
  eval_keys = [
      'random',
      'novice',
      'heuristic',
      'tree_1ply',
      'tree_2ply',
      'rl_td_value',
      'rl_q_learning',
      'rl_policy_gradient',
  ]
  print('\nRunning final round-robin Elo tournament across all 8 players...', flush=True)
  eval_elos, wins, played, vs_heur = run_eval_tournament(
      pool, eval_keys, games_per_pair=24, seed=seed + 500000
  )

  print('\n=== FINAL ROUND-ROBIN ELO LEADERBOARD ===')
  ranked_final = sorted(eval_keys, key=lambda k: eval_elos[k], reverse=True)
  for rank, key in enumerate(ranked_final, 1):
    w = wins[key]
    gp = max(played[key], 1)
    vh_w, vh_g = vs_heur[key]
    vh_str = f'{100.0 * vh_w / vh_g:5.1f}%' if vh_g > 0 else '  n/a'
    print(
        f'  {rank}. {pool[key].name:28s} | Elo: {eval_elos[key]:6.1f} '
        f'| Overall Win%: {100.0 * w / gp:5.1f}% ({w:3d}/{gp:3d}) '
        f'| vs Heuristic: {vh_str}'
    )

  # Save all three trained RL models with tournament Elo metadata.
  for key, bot in (
      ('rl_td_value', td_bot),
      ('rl_q_learning', q_bot),
      ('rl_policy_gradient', pg_bot),
  ):
    bot.save(
        RL_MODEL_PATHS[key],
        metadata={
            'trained_games': total_games,
            'tournament_elo': round(eval_elos[key], 1),
            'overall_win_rate': round(
                wins[key] / max(played[key], 1), 3
            ),
        },
    )
    print(f'Saved {bot.name} -> {RL_MODEL_PATHS[key]}')

  if logger is not None:
    print(f'Saved {logger.saved_games:,} progressive self-play games -> {save_trajectories}')


def main():
  parser = argparse.ArgumentParser(
      description='Train all 3 RL bots via progressive Elo self-play.'
  )
  parser.add_argument('--games', type=int, default=10000)
  parser.add_argument('--save-trajectories', default='selfplay_10k.jsonl.gz')
  parser.add_argument('--seed', type=int, default=42)
  args = parser.parse_args()
  train_all(
      total_games=args.games,
      save_trajectories=args.save_trajectories or None,
      seed=args.seed,
  )


if __name__ == '__main__':
  main()
