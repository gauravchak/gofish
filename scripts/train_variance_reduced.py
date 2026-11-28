"""Variance-Reduced RL Trainer: X-Ray GRPO + Rollout Teacher Distillation.

Tests two modern variance-reduction techniques designed specifically for
high-luck imperfect-information card games:

  1. X-Ray Critic + Duplicate-Deal GRPO (`models/grpo_xray_model.pt`):
       - Plays paired mirror deals (same deck, seats swapped) in 5,000-game batches.
       - Omniscient X-Ray Critic sees both players' real hands during training only,
         while the Actor remains 100% fair!
  2. Rollout Teacher Distillation / Expert Iteration (`models/distilled_model.pt`):
       - Freezes real turns (using ONLY legally visible cards!), runs Common-Random-
         Number Monte-Carlo rollouts for every legal move to compute true EV(s, a),
         and distills the Teacher's probabilities into a fast 0.2 ms PyTorch Student!

Run:
  python3 -m scripts.train_variance_reduced --grpo-batches 4 --games-per-batch 5000
"""

import argparse
import random
import time

import torch

from bots import RL_MODEL_PATHS, create_bot
from game import GameEngine
from ml import (
    DistilledBot,
    XRayGRPOBot,
    evaluate_state_with_rollouts,
)


def _eval_h2h_duplicate(bot, opp, n_pairs=50, seed_base=85000):
  """Evaluates bot vs opp over n_pairs*2 paired duplicate deals (same deck, both seats)."""
  prev_stoch = getattr(bot, 'stochastic', False)
  if hasattr(bot, 'stochastic'):
    bot.stochastic = False
  w = l = d = 0
  for g in range(n_pairs * 2):
    swap = (g % 2) == 1
    b0, b1 = (opp, bot) if swap else (bot, opp)
    eng = GameEngine(logger=None)
    eng.join_player('p0', b0.name, player_type=b0.player_type)
    eng.join_player('p1', b1.name, player_type=b1.player_type)
    eng.start_match(seed=seed_base + g // 2)
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
      w += 1
    elif eng.winner_idx is None:
      d += 1
    else:
      l += 1
  if hasattr(bot, 'stochastic'):
    bot.stochastic = prev_stoch
  return w, l, d


def _play_grpo_mirror_pair(grpo_bot, opp_bot, deal_seed):
  """Plays ONE mirror pair (2 games, exact same shuffled deck, seats swapped)."""
  pair_data = []
  for bot_seat in (0, 1):
    eng = GameEngine(logger=None)
    b0, b1 = (grpo_bot, opp_bot) if bot_seat == 0 else (opp_bot, grpo_bot)
    eng.join_player('p0', b0.name, player_type=b0.player_type)
    eng.join_player('p1', b1.name, player_type=b1.player_type)
    eng.start_match(seed=deal_seed)

    steps = []
    while eng.winner is None:
      p = eng.turn
      obs = eng.get_observation(p)
      if p == bot_seat:
        opp_obs = eng.get_observation(1 - p)
        if eng.phase == 'DRAW':
          act, rec = grpo_bot.choose_draw(obs, return_record=True, opp_obs=opp_obs)
          steps.append(rec)
          eng.draw(p, act)
        else:
          act, rec = grpo_bot.choose_discard(obs, return_record=True, opp_obs=opp_obs)
          steps.append(rec)
          eng.discard(p, act, by_index=False)
      else:
        if eng.phase == 'DRAW':
          eng.draw(p, opp_bot.choose_draw(obs))
        else:
          eng.discard(p, opp_bot.choose_discard(obs), by_index=False)

    if eng.winner_idx == bot_seat:
      rew = 1.0
    elif eng.winner_idx is None:
      rew = 0.0
    else:
      rew = -1.0
    pair_data.append((steps, rew))

  return (pair_data[0][0], pair_data[0][1], pair_data[1][0], pair_data[1][1])


def train_xray_grpo(batches=4, games_per_batch=5000, seed=707):
  """Trains XRayGRPOBot in 5,000-game duplicate-deal batches."""
  torch.set_num_threads(1)
  rng = random.Random(seed)
  t0 = time.time()

  grpo = XRayGRPOBot(hidden_dim=64, seed=seed)
  heur = create_bot('heuristic')
  t1 = create_bot('tree_1ply')
  novice = create_bot('novice')
  td_opp = create_bot('rl_td_value')
  ddqn_opp = create_bot('rl_q_learning')
  opponents = [heur, heur, td_opp, ddqn_opp, t1, novice]

  print(
      f'\n=== EXPERIMENT 1: X-Ray Critic + Duplicate-Deal GRPO '
      f'({batches} batches x {games_per_batch:,} games) ===',
      flush=True,
  )

  best_score = -1.0
  pairs_per_batch = games_per_batch // 2
  deal_cursor = seed * 10000

  for batch_idx in range(1, batches + 1):
    decay = 1.0 / (1.0 + 0.25 * (batch_idx - 1))
    actor_lr = max(0.00025, 0.0010 * decay)
    critic_lr = max(0.00035, 0.0015 * decay)
    grpo.stochastic = True
    grpo.temperature = 1.0

    rollout_pairs = []
    for _ in range(pairs_per_batch):
      deal_cursor += 1
      opp = rng.choice(opponents) if batch_idx > 1 else rng.choice([heur, novice, td_opp])
      rollout_pairs.append(_play_grpo_mirror_pair(grpo, opp, deal_cursor))
      # Update every 32 mirror pairs (64 games) for low-variance GRPO steps!
      if len(rollout_pairs) >= 32:
        grpo.learn_grpo_batch(
            rollout_pairs, ppo_epochs=3, actor_lr=actor_lr, critic_lr=critic_lr
        )
        rollout_pairs.clear()

    if rollout_pairs:
      grpo.learn_grpo_batch(
          rollout_pairs, ppo_epochs=3, actor_lr=actor_lr, critic_lr=critic_lr
      )
      rollout_pairs.clear()

    wh, lh, dh = _eval_h2h_duplicate(grpo, heur, n_pairs=50, seed_base=85000)
    wt, lt, dt = _eval_h2h_duplicate(grpo, t1, n_pairs=50, seed_base=86000)
    pct_h = 100.0 * wh / max(wh + lh + dh, 1)
    pct_t = 100.0 * wt / max(wt + lt + dt, 1)
    combo = 0.5 * (pct_h + pct_t)
    promoted = ''
    if combo > best_score:
      best_score = combo
      grpo.save(
          RL_MODEL_PATHS['rl_grpo_xray'],
          metadata={
              'batch': batch_idx,
              'trained_games': batch_idx * games_per_batch,
              'vs_heuristic_pct': round(pct_h, 1),
              'vs_tree1_pct': round(pct_t, 1),
          },
      )
      promoted = ' [SAVED BEST]'
    print(
        f'  GRPO Batch {batch_idx} ({batch_idx * games_per_batch:5d}g | '
        f'{time.time() - t0:5.1f}s) | vs Heur: {wh:2d}-{lh:2d} ({pct_h:5.1f}%) | '
        f'vs Tree1: {wt:2d}-{lt:2d} ({pct_t:5.1f}%){promoted}',
        flush=True,
    )
  return XRayGRPOBot(model_path=RL_MODEL_PATHS['rl_grpo_xray'])


def _collect_teacher_states(rollout_bot, opp_bot, num_states, num_worlds, rng):
  """Plays games and runs Common-Random-Number MC rollouts on diverse turns."""
  dataset = []
  game_seed = rng.randint(100000, 900000)
  while len(dataset) < num_states:
    game_seed += 1
    eng = GameEngine(logger=None)
    eng.join_player('p0', rollout_bot.name, player_type=rollout_bot.player_type)
    eng.join_player('p1', opp_bot.name, player_type=opp_bot.player_type)
    eng.start_match(seed=game_seed)

    step_num = 0
    while eng.winner is None and len(dataset) < num_states:
      p = eng.turn
      obs = eng.get_observation(p)
      step_num += 1
      # Sample ~22% of non-trivial decisions across early/mid/late turns!
      if p == 0 and rng.random() < 0.22:
        sample = evaluate_state_with_rollouts(
            obs, rollout_bot, num_worlds=num_worlds, rng=rng
        )
        if sample is not None:
          evs = sample['evs']
          # Keep informative positions where actions actually differ in EV!
          if max(evs) - min(evs) >= 0.05:
            dataset.append(sample)

      actor = rollout_bot if p == 0 else opp_bot
      if eng.phase == 'DRAW':
        eng.draw(p, actor.choose_draw(obs))
      else:
        eng.discard(p, actor.choose_discard(obs), by_index=False)
  return dataset


def train_rollout_distillation(
    iterations=2, states_per_iter=450, num_worlds=12, seed=808
):
  """Trains DistilledBot via 2 rounds of Expert Iteration (MC Rollout Teacher)."""
  torch.set_num_threads(1)
  rng = random.Random(seed)
  t0 = time.time()

  heur = create_bot('heuristic')
  t1 = create_bot('tree_1ply')
  student = DistilledBot(hidden_dim=64, seed=seed)

  print(
      f'\n=== EXPERIMENT 2: Rollout Teacher Distillation / Expert Iteration '
      f'({iterations} iters x {states_per_iter} states x {num_worlds} worlds) ===',
      flush=True,
  )

  all_samples = []
  best_score = -1.0

  for it in range(1, iterations + 1):
    # Iter 1 uses Heuristic for ultra-fast strong rollouts; Iter 2 uses our Student!
    rollout_bot = heur if it == 1 else student
    opp_bot = heur if it == 1 else create_bot('rl_td_value')
    new_samples = _collect_teacher_states(
        rollout_bot, opp_bot, states_per_iter, num_worlds, rng
    )
    all_samples.extend(new_samples)

    loss = student.train_on_teacher_dataset(
        all_samples,
        epochs=35,
        batch_size=64,
        teacher_temp=0.07,
        lr=0.002 if it == 1 else 0.001,
        seed=seed + it,
    )

    wh, lh, dh = _eval_h2h_duplicate(student, heur, n_pairs=50, seed_base=85000)
    wt, lt, dt = _eval_h2h_duplicate(student, t1, n_pairs=50, seed_base=86000)
    pct_h = 100.0 * wh / max(wh + lh + dh, 1)
    pct_t = 100.0 * wt / max(wt + lt + dt, 1)
    combo = 0.5 * (pct_h + pct_t)
    promoted = ''
    if combo > best_score:
      best_score = combo
      student.save(
          RL_MODEL_PATHS['rl_distilled'],
          metadata={
              'iteration': it,
              'teacher_states': len(all_samples),
              'worlds_per_action': num_worlds,
              'vs_heuristic_pct': round(pct_h, 1),
              'vs_tree1_pct': round(pct_t, 1),
          },
      )
      promoted = ' [SAVED BEST]'
    print(
        f'  Distill Iter {it} ({len(all_samples):4d} states | loss={loss:.3f} | '
        f'{time.time() - t0:5.1f}s) | vs Heur: {wh:2d}-{lh:2d} ({pct_h:5.1f}%) | '
        f'vs Tree1: {wt:2d}-{lt:2d} ({pct_t:5.1f}%){promoted}',
        flush=True,
    )
  return DistilledBot(model_path=RL_MODEL_PATHS['rl_distilled'])


def run_grand_comparison():
  """Head-to-head benchmark of ALL 6 PyTorch RL models on 100 unseen duplicate deals!"""
  heur = create_bot('heuristic')
  t1 = create_bot('tree_1ply')
  print('\n=== GRAND COMPARISON: ALL 6 PYTORCH RL BOTS (100 UNSEEN DUPLICATE DEALS EACH) ===')
  for key in (
      'rl_grpo_xray',
      'rl_distilled',
      'rl_td_value',
      'rl_q_learning',
      'rl_policy_gradient',
      'rl_alphazero',
  ):
    bot = create_bot(key)
    n_pairs = 25 if key == 'rl_alphazero' else 50
    wh, lh, dh = _eval_h2h_duplicate(bot, heur, n_pairs=n_pairs, seed_base=95000)
    wt, lt, dt = _eval_h2h_duplicate(bot, t1, n_pairs=n_pairs, seed_base=96000)
    tot = 2 * n_pairs
    print(
        f'  {bot.name:40s} | vs Heur: {wh:2d}-{lh:2d} ({100*wh/tot:5.1f}%) | '
        f'vs Tree1: {wt:2d}-{lt:2d} ({100*wt/tot:5.1f}%)',
        flush=True,
    )


def main():
  parser = argparse.ArgumentParser(
      description='Train X-Ray GRPO and Rollout-Distilled Student bots.'
  )
  parser.add_argument('--grpo-batches', type=int, default=4)
  parser.add_argument('--games-per-batch', type=int, default=5000)
  parser.add_argument('--distill-iters', type=int, default=2)
  parser.add_argument('--distill-states', type=int, default=450)
  parser.add_argument('--distill-worlds', type=int, default=12)
  args = parser.parse_args()

  train_xray_grpo(batches=args.grpo_batches, games_per_batch=args.games_per_batch)
  train_rollout_distillation(
      iterations=args.distill_iters,
      states_per_iter=args.distill_states,
      num_worlds=args.distill_worlds,
  )
  run_grand_comparison()


if __name__ == '__main__':
  main()
