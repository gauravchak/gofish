"""Generational PyTorch Bot League: Live Replay, Double DQN, PPO & AlphaZero Self-Play.

Trains ALL 4 PyTorch RL bots in continuous 5,000-game batches against each other,
Hall-of-Fame snapshots, and our rules/tree bots — measuring rock-solid Elo with
paired duplicate-deal evaluation (same deck, seats swapped) and order-independent
Bradley-Terry Elo ratings!

Key ML upgrades:
  1. TD Value: Live target-network TD + MC replay (eliminates stale replay targets!)
  2. Double DQN: Online argmax + Target evaluation + MC blend + tau=0.005
  3. PPO: 32-episode rollout batches + GAE + Advantage Normalization + Clipped Loss
  4. True AlphaZero Self-Play: Policy + Value + Tree Search distilling search -> nets!
  5. Paired Duplicate-Deal Eval + --resume mode for continuous 5,000-game batches.

Run fresh:
  python3 -m scripts.league --generations 4 --games-per-gen 5000

Continue training in more 5,000-game batches:
  python3 -m scripts.league --resume --generations 4 --games-per-gen 5000
"""

import argparse
import gzip
import json
import os
import random
import time

import torch

from bots import MODELS_DIR, RL_MODEL_PATHS, create_bot
from bots.heuristic import get_next_player_hints, get_unseen_mask
from game import GameEngine, card_to_id, expected_score
from ml import (
    AFTERSTATE_DIM,
    AlphaZeroLiteBot,
    PolicyGradientBot,
    QLearningBot,
    ReplayBuffer,
    TDValueBot,
    extract_afterstate_features,
)

LEAGUE_HISTORY_PATH = os.path.join(MODELS_DIR, 'league_history.json')


def load_offline_td_samples(path):
  """Loads completed DISCARD steps from a .jsonl/.jsonl.gz file as (feats, outcome_01).

  Lets families mix their own real games from trajectories.jsonl right into the
  PyTorch Replay Buffer alongside self-play!
  """
  if not path or not os.path.exists(path):
    return []
  opener = (
      (lambda p: gzip.open(p, 'rt', encoding='utf-8'))
      if path.endswith('.gz')
      else (lambda p: open(p, 'r', encoding='utf-8'))
  )
  samples = []
  with opener(path) as f:
    for line in f:
      line = line.strip()
      if not line:
        continue
      try:
        rec = json.loads(line)
      except json.JSONDecodeError:
        continue
      if rec.get('phase') != 'DISCARD' or 'hand_ids' not in rec:
        continue
      cid = rec.get('action_id')
      hand7 = rec['hand_ids']
      if cid not in hand7 or len(hand7) != 7:
        continue
      idx = hand7.index(cid)
      h6 = hand7[:idx] + hand7[idx + 1 :]
      obs_view = {
          'player': rec['player'],
          'num_players': rec['num_players'],
          'hand_ids': hand7,
          'discard_pile_ids': rec.get('discard_pile_ids', []),
          'public_history': rec.get('public_history', []),
      }
      feats = extract_afterstate_features(
          h6,
          get_unseen_mask(obs_view),
          rec.get('deck_remaining', 20),
          *get_next_player_hints(obs_view),
          discarded_id=cid,
          took_shown=rec.get('last_picked_shown_id') is not None,
      )
      rew = float(rec.get('reward', 0.0))
      outcome_01 = 1.0 if rew > 0 else (0.0 if rew < 0 else 0.5)
      samples.append((feats, outcome_01))
  return samples


def _solve_balanced_elos(keys, all_results, iterations=40):
  """Computes order-independent Bradley-Terry Elo ratings (centered at 1000.0).

  Unlike single-pass sequential Elo (which depends heavily on match order!),
  simultaneous iterative updates converge to the exact maximum-likelihood Elo!
  """
  elos = {k: 1000.0 for k in keys}
  if not all_results:
    return elos

  # Aggregate pairwise scores once for fast, exact convergence.
  pair_stats = {}
  for k0, k1, winner_idx in all_results:
    score_0 = 1.0 if winner_idx == 0 else (0.0 if winner_idx == 1 else 0.5)
    a, b = (k0, k1) if k0 < k1 else (k1, k0)
    sa = score_0 if k0 < k1 else (1.0 - score_0)
    if (a, b) not in pair_stats:
      pair_stats[(a, b)] = [0.0, 0]
    pair_stats[(a, b)][0] += sa
    pair_stats[(a, b)][1] += 1

  for step in range(iterations):
    lr = 28.0 if step < 15 else 12.0
    deltas = {k: 0.0 for k in keys}
    counts = {k: 0 for k in keys}
    for (a, b), (score_a, n_games) in pair_stats.items():
      exp_a = expected_score(elos[a], elos[b]) * n_games
      diff = score_a - exp_a
      deltas[a] += diff
      deltas[b] -= diff
      counts[a] += n_games
      counts[b] += n_games

    for k in keys:
      if counts[k] > 0:
        # Damped gradient step proportional to average residual per game.
        elos[k] += lr * (deltas[k] / max(counts[k] ** 0.5, 1.0))

    # Re-center mean Elo at 1000.0 every iteration.
    mean_elo = sum(elos.values()) / len(elos)
    for k in keys:
      elos[k] -= mean_elo - 1000.0

  return elos


def _afterstate_from_discard(obs, discarded_card):
  cid = card_to_id(discarded_card)
  hand7 = obs['hand_ids']
  idx = hand7.index(cid)
  h6 = hand7[:idx] + hand7[idx + 1 :]
  return extract_afterstate_features(
      h6,
      get_unseen_mask(obs),
      obs['deck_remaining'],
      *get_next_player_hints(obs),
      discarded_id=cid,
      took_shown=obs.get('last_picked_shown_id') is not None,
  )


def _play_league_game(
    bot0,
    bot1,
    seed,
    td_replay,
    q_replay,
    pg_rollout,
    az_val_replay,
    az_pol_replay,
    lr,
    active_learners,
):
  """Plays one league training game, filling replay buffers & PPO rollouts."""
  eng = GameEngine(logger=None)
  eng.join_player('p0', bot0.name, player_type=bot0.player_type)
  eng.join_player('p1', bot1.name, player_type=bot1.player_type)
  eng.start_match(seed=seed)

  bots = [bot0, bot1]
  need_afterstates = any(
      b in active_learners and isinstance(b, (TDValueBot, PolicyGradientBot, AlphaZeroLiteBot))
      for b in bots
  )
  all_afterstates = [[], []]
  q_transitions = [[], []]
  pg_steps = [[], []]
  az_search = [[], []]

  while eng.winner is None:
    p = eng.turn
    bot = bots[p]
    is_active = bot in active_learners
    obs = eng.get_observation(p)

    if eng.phase == 'DRAW':
      if is_active and isinstance(bot, QLearningBot):
        action, rec = bot.choose_draw(obs, return_record=True)
        q_transitions[p].append(rec)
      elif is_active and isinstance(bot, PolicyGradientBot):
        action, rec = bot.choose_draw(obs, return_record=True)
        pg_steps[p].append(rec)
      elif is_active and isinstance(bot, AlphaZeroLiteBot):
        action, s_rec = bot.choose_draw(obs, return_search=True)
        az_search[p].append(s_rec)
      else:
        action = bot.choose_draw(obs)
      res = eng.draw(p, action)
    else:
      if is_active and isinstance(bot, TDValueBot):
        action, feats = bot.choose_discard(obs, return_features=True)
        all_afterstates[p].append(feats)
      elif is_active and isinstance(bot, QLearningBot):
        action, rec = bot.choose_discard(obs, return_record=True)
        q_transitions[p].append(rec)
        if need_afterstates:
          all_afterstates[p].append(rec['feats'][:AFTERSTATE_DIM])
      elif is_active and isinstance(bot, PolicyGradientBot):
        action, rec = bot.choose_discard(obs, return_record=True)
        pg_steps[p].append(rec)
        all_afterstates[p].append(rec['after_feats'])
      elif is_active and isinstance(bot, AlphaZeroLiteBot):
        action, s_rec = bot.choose_discard(obs, return_search=True)
        az_search[p].append(s_rec)
        all_afterstates[p].append(s_rec['after_feats'])
      else:
        action = bot.choose_discard(obs)
        if need_afterstates:
          all_afterstates[p].append(_afterstate_from_discard(obs, action))
      res = eng.discard(p, action, by_index=False)

    if not res['ok']:
      raise RuntimeError(f'{bot.name} made illegal move {action}: {res}')

  # Push experiences ONLY for active learners (never frozen Hall-of-Fame clones!).
  for seat, bot in enumerate(bots):
    if bot not in active_learners:
      continue
    other = 1 - seat
    if eng.winner_idx is None:
      out_01, rew_pm, opp_01, opp_pm = 0.5, 0.0, 0.5, 0.0
    elif eng.winner_idx == seat:
      out_01, rew_pm, opp_01, opp_pm = 1.0, 1.0, 0.0, -1.0
    else:
      out_01, rew_pm, opp_01, opp_pm = 0.0, -1.0, 1.0, 1.0

    if isinstance(bot, TDValueBot):
      td_replay.extend(bot.build_episode_samples(all_afterstates[seat], out_01))
      td_replay.extend(bot.build_episode_samples(all_afterstates[other], opp_01))
    elif isinstance(bot, QLearningBot):
      q_replay.extend(bot.build_replay_transitions(q_transitions[seat], rew_pm))
    elif isinstance(bot, PolicyGradientBot):
      bot.train_critic_only(all_afterstates[other], opp_pm, critic_lr=lr)
      pg_rollout.append((pg_steps[seat], rew_pm))
    elif isinstance(bot, AlphaZeroLiteBot):
      val_s, pol_s = bot.build_selfplay_samples(
          az_search[seat], all_afterstates[seat], out_01
      )
      az_val_replay.extend(val_s)
      az_val_replay.extend(
          bot.value_bot.build_episode_samples(all_afterstates[other], opp_01)
      )
      az_pol_replay.extend(pol_s)

  return eng.winner_idx


def _play_eval_pair(pool, key_a, key_b, n_games, seed_base):
  """Plays n_games PAIRED DUPLICATE-DEAL matches between key_a and key_b.

  Every pair (2k, 2k+1) deals the EXACT same shuffled deck with seats swapped!
  This cancels card-deal luck so Elo changes reflect genuine strategy gains.
  """
  results = []
  for g in range(n_games):
    swap = (g % 2) == 1
    k0, k1 = (key_b, key_a) if swap else (key_a, key_b)
    deal_seed = seed_base + (g // 2)
    eng = GameEngine(logger=None)
    eng.join_player('p0', pool[k0].name, player_type=pool[k0].player_type)
    eng.join_player('p1', pool[k1].name, player_type=pool[k1].player_type)
    eng.start_match(seed=deal_seed)

    b0, b1 = pool[k0], pool[k1]
    while eng.winner is None:
      p = eng.turn
      obs = eng.get_observation(p)
      bot = b0 if p == 0 else b1
      if eng.phase == 'DRAW':
        res = eng.draw(p, bot.choose_draw(obs))
      else:
        res = eng.discard(p, bot.choose_discard(obs), by_index=False)
      if not res['ok']:
        raise RuntimeError(f'{bot.name} made illegal eval move: {res}')
    results.append((k0, k1, eng.winner_idx))
  return results


def evaluate_generation(
    pool,
    keys,
    fixed_keys,
    cached_fixed_results,
    games_per_pair=32,
    seed=777000,
):
  """Fixed-seed duplicate-deal round-robin tournament across all 9 players."""
  # Switch all bots into deterministic full-strength eval mode.
  az_bot = pool.get('rl_alphazero')
  prev_az_cfg = None
  if az_bot is not None:
    prev_az_cfg = (az_bot.draw_sample, az_bot.top_k_discards, az_bot.stochastic)
    az_bot.draw_sample = 12
    az_bot.top_k_discards = 3
    az_bot.stochastic = False

  for key in keys:
    bot = pool[key]
    if hasattr(bot, 'epsilon'):
      bot.epsilon = 0.0
    if hasattr(bot, 'explore_temp'):
      bot.explore_temp = 0.0
    if hasattr(bot, 'stochastic'):
      bot.stochastic = False

  wins = {k: 0 for k in keys}
  played = {k: 0 for k in keys}
  vs_heur = {k: [0, 0] for k in keys}
  vs_tree1 = {k: [0, 0] for k in keys}

  slow_keys = {'tree_2ply', 'rl_alphazero'}
  all_results = []
  seed_cursor = seed

  for i, key_a in enumerate(keys):
    for key_b in keys[i + 1 :]:
      n_games = (
          min(games_per_pair, 20)
          if (key_a in slow_keys or key_b in slow_keys)
          else games_per_pair
      )
      # Ensure even number so every deck is played once from each seat!
      n_games = max(2, n_games - (n_games % 2))
      pair_key = (key_a, key_b)
      if key_a in fixed_keys and key_b in fixed_keys:
        if pair_key not in cached_fixed_results:
          cached_fixed_results[pair_key] = _play_eval_pair(
              pool, key_a, key_b, n_games, seed_cursor
          )
        pair_results = cached_fixed_results[pair_key]
      else:
        pair_results = _play_eval_pair(
            pool, key_a, key_b, n_games, seed_cursor
        )
      seed_cursor += n_games
      all_results.extend(pair_results)

  if az_bot is not None and prev_az_cfg is not None:
    az_bot.draw_sample, az_bot.top_k_discards, az_bot.stochastic = prev_az_cfg

  elos = _solve_balanced_elos(keys, all_results)
  for k0, k1, winner_idx in all_results:
    played[k0] += 1
    played[k1] += 1
    if winner_idx == 0:
      wins[k0] += 1
    elif winner_idx == 1:
      wins[k1] += 1

    for bench_key, tracker in (('heuristic', vs_heur), ('tree_1ply', vs_tree1)):
      if k0 == bench_key:
        tracker[k1][1] += 1
        if winner_idx == 1:
          tracker[k1][0] += 1
      elif k1 == bench_key:
        tracker[k0][1] += 1
        if winner_idx == 0:
          tracker[k0][0] += 1

  return elos, wins, played, vs_heur, vs_tree1


def _write_history(history, eval_pool, eval_keys, games_per_gen):
  os.makedirs(MODELS_DIR, exist_ok=True)
  with open(LEAGUE_HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(
        {
            'generations': len(history) - 1 if history else 0,
            'games_per_gen': games_per_gen,
            'player_names': {k: eval_pool[k].name for k in eval_keys},
            'history': history,
        },
        f,
        indent=2,
    )
    f.write('\n')


def run_league(
    generations=4,
    games_per_gen=5000,
    eval_games=32,
    seed=42,
    include_human=None,
    resume=False,
):
  """Trains all 4 PyTorch RL bots in continuous generational batches."""
  torch.set_num_threads(1)
  t_start = time.time()

  history = []
  start_gen = 0
  games_so_far = 0
  best_elo = {}

  if resume and os.path.exists(LEAGUE_HISTORY_PATH):
    with open(LEAGUE_HISTORY_PATH, 'r', encoding='utf-8') as f:
      prev = json.load(f)
    history = prev.get('history', [])
    if history:
      start_gen = history[-1]['generation']
      games_so_far = history[-1]['games_total']
      for k in (
          'rl_td_value',
          'rl_q_learning',
          'rl_policy_gradient',
          'rl_alphazero',
      ):
        prev_vals = [h['elos'][k] for h in history if k in h.get('elos', {})]
        if prev_vals:
          best_elo[k] = max(prev_vals)

  rng = random.Random(seed + start_gen * 1009)

  if resume:
    td_bot = TDValueBot(model_path=RL_MODEL_PATHS['rl_td_value'], hidden_dim=64, seed=seed + 1)
    q_bot = QLearningBot(model_path=RL_MODEL_PATHS['rl_q_learning'], hidden_dim=64, seed=seed + 2)
    pg_bot = PolicyGradientBot(model_path=RL_MODEL_PATHS['rl_policy_gradient'], hidden_dim=64, seed=seed + 3)
    az_bot = AlphaZeroLiteBot(model_path=RL_MODEL_PATHS['rl_alphazero'], hidden_dim=64, draw_sample=2, top_k_discards=2, seed=seed + 4)
  else:
    td_bot = TDValueBot(hidden_dim=64, seed=seed + 1)
    q_bot = QLearningBot(hidden_dim=64, seed=seed + 2)
    pg_bot = PolicyGradientBot(hidden_dim=64, seed=seed + 3)
    az_bot = AlphaZeroLiteBot(hidden_dim=64, draw_sample=2, top_k_discards=2, seed=seed + 4)

  td_replay = ReplayBuffer(capacity=50000)
  q_replay = ReplayBuffer(capacity=50000)
  az_val_replay = ReplayBuffer(capacity=40000)
  az_pol_replay = ReplayBuffer(capacity=40000)
  pg_rollout = []

  if include_human:
    human_samples = load_offline_td_samples(include_human)
    if human_samples:
      td_replay.extend(human_samples)
      az_val_replay.extend(human_samples)
      print(
          f'Seeded Replay Buffers with {len(human_samples)} human steps from {include_human}',
          flush=True,
      )

  hall_of_fame = []
  if resume:
    hall_of_fame.append(pg_bot.clone_frozen(f'pg_g{start_gen}'))
    hall_of_fame.append(q_bot.clone_frozen(f'ddqn_g{start_gen}'))

  fixed = {
      'random': create_bot('random'),
      'novice': create_bot('novice'),
      'heuristic': create_bot('heuristic'),
      'tree_1ply': create_bot('tree_1ply'),
      'tree_2ply': create_bot('tree_2ply'),
  }
  learners = {
      'rl_td_value': td_bot,
      'rl_q_learning': q_bot,
      'rl_policy_gradient': pg_bot,
      'rl_alphazero': az_bot,
  }
  active_learner_set = set(learners.values())
  for k in learners:
    best_elo.setdefault(k, -1.0)

  eval_pool = {**fixed, **learners}
  eval_keys = [
      'random',
      'novice',
      'heuristic',
      'tree_1ply',
      'tree_2ply',
      'rl_td_value',
      'rl_q_learning',
      'rl_policy_gradient',
      'rl_alphazero',
  ]
  fixed_keys = set(fixed.keys())
  cached_fixed_results = {}

  def _record_generation(gen_idx, games_total):
    elos, wins, played, vs_heur, vs_tree1 = evaluate_generation(
        eval_pool,
        eval_keys,
        fixed_keys=fixed_keys,
        cached_fixed_results=cached_fixed_results,
        games_per_pair=eval_games,
        seed=777000,
    )
    entry = {
        'generation': gen_idx,
        'games_total': games_total,
        'elapsed_s': round(time.time() - t_start, 1),
        'elos': {k: round(elos[k], 1) for k in eval_keys},
        'win_rates': {
            k: round(100.0 * wins[k] / max(played[k], 1), 1) for k in eval_keys
        },
        'vs_heuristic': {
            k: (
                round(100.0 * vs_heur[k][0] / vs_heur[k][1], 1)
                if vs_heur[k][1] > 0
                else None
            )
            for k in eval_keys
        },
        'vs_tree1': {
            k: (
                round(100.0 * vs_tree1[k][0] / vs_tree1[k][1], 1)
                if vs_tree1[k][1] > 0
                else None
            )
            for k in eval_keys
        },
    }
    history.append(entry)

    # King-of-the-Hill: promote checkpoints when a bot hits a new best Elo!
    promoted = []
    if gen_idx > 0:
      for k, bot in learners.items():
        if elos[k] > best_elo[k]:
          best_elo[k] = elos[k]
          bot.save(
              RL_MODEL_PATHS[k],
              metadata={
                  'generation': gen_idx,
                  'trained_games': games_total,
                  'tournament_elo': round(elos[k], 1),
                  'overall_win_rate': round(
                      wins[k] / max(played[k], 1), 3
                  ),
              },
          )
          promoted.append(f'{k}({elos[k]:.0f})')

      # Snapshot top non-search or search RL bot into Hall of Fame.
      top_rl = max(
          ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient'),
          key=lambda k: elos[k],
      )
      hall_of_fame.append(learners[top_rl].clone_frozen(f'{top_rl}_g{gen_idx}'))
      if len(hall_of_fame) > 5:
        hall_of_fame.pop(0)

    # Persist league_history.json live after EVERY batch so /league updates immediately!
    _write_history(history, eval_pool, eval_keys, games_per_gen)

    print(
        f'Gen {gen_idx:2d} ({games_total:6d} games | {entry["elapsed_s"]:6.1f}s) | '
        f'TD:{elos["rl_td_value"]:6.1f} | DDQN:{elos["rl_q_learning"]:6.1f} | '
        f'PPO:{elos["rl_policy_gradient"]:6.1f} | AZ:{elos["rl_alphazero"]:6.1f} | '
        f'Heur:{elos["heuristic"]:6.1f} | T1:{elos["tree_1ply"]:6.1f} | '
        f'T2:{elos["tree_2ply"]:6.1f}'
        + (f'  [Promoted: {", ".join(promoted)}]' if promoted else ''),
        flush=True,
    )
    return elos, wins, played, vs_heur, vs_tree1

  mode_str = f'RESUME from Gen {start_gen} ({games_so_far:,} games)' if resume else 'FRESH'
  print(
      f'=== PyTorch Bot League ({mode_str}): +{generations} Batches x {games_per_gen:,} games '
      f'(Duplicate-Deal Eval: {eval_games}g/pair) ===',
      flush=True,
  )
  if not history:
    _record_generation(0, 0)

  learner_keys = (
      'rl_td_value',
      'rl_q_learning',
      'rl_policy_gradient',
      'rl_alphazero',
  )
  ppo_batch_episodes = 64

  for batch_offset in range(1, generations + 1):
    gen = start_gen + batch_offset
    # Gentle decay with healthy floor so multi-batch --resume runs keep refining!
    decay = 1.0 / (1.0 + 0.30 * (gen - 1))
    lr = max(0.00025, 0.0012 * decay)
    actor_lr = max(0.00025, 0.0009 * decay)

    # Smart Boltzmann exploration for value/Q bots (never throw away completed sets!)
    td_bot.epsilon = 0.02
    td_bot.explore_temp = max(0.03, 0.08 * decay)
    q_bot.epsilon = 0.02
    q_bot.explore_temp = max(0.05, 0.14 * decay)
    # Standard PPO uses temperature=1.0; entropy bonus manages exploration cleanly.
    pg_bot.stochastic = True
    pg_bot.temperature = 1.0
    az_bot.stochastic = True
    az_bot.search_temp = max(0.03, 0.07 * decay)
    az_bot.draw_sample = 2
    az_bot.top_k_discards = 2

    for g in range(1, games_per_gen + 1):
      games_so_far += 1
      lk = learner_keys[(g - 1) % len(learner_keys)]
      fast_peers = [
          k for k in ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient')
          if k != lk
      ]

      r = rng.random()
      if gen == 1:
        if r < 0.45:
          opp = eval_pool[rng.choice(fast_peers)]
        elif r < 0.70:
          opp = fixed['heuristic']
        elif r < 0.90:
          opp = fixed['novice']
        else:
          opp = fixed['random']
      else:
        # Gen 2+: Keep 10% Novice anchor games so value/Q nets NEVER flatten!
        if r < 0.35:
          opp = eval_pool[rng.choice(fast_peers)]
        elif hall_of_fame and r < 0.55:
          opp = rng.choice(hall_of_fame)
        elif r < 0.75:
          opp = fixed['heuristic']
        elif r < 0.90:
          opp = fixed['tree_1ply']
        else:
          opp = fixed['novice']

      b0, b1 = (eval_pool[lk], opp) if (g % 2 == 0) else (opp, eval_pool[lk])
      _play_league_game(
          b0,
          b1,
          seed=seed * 100000 + games_so_far,
          td_replay=td_replay,
          q_replay=q_replay,
          pg_rollout=pg_rollout,
          az_val_replay=az_val_replay,
          az_pol_replay=az_pol_replay,
          lr=lr,
          active_learners=active_learner_set,
      )

      # 1. Batched PPO update every 64 episodes (low-variance GAE + clipped ratio!)
      if len(pg_rollout) >= ppo_batch_episodes:
        pg_bot.learn_ppo_batch(
            pg_rollout,
            ppo_epochs=3,
            actor_lr=actor_lr,
            critic_lr=lr,
        )
        pg_rollout.clear()

      # 2. Balanced 1:1 replay updates ONLY for bots that played this game (no overfitting!)
      played_bots = (b0, b1)
      if td_bot in played_bots and len(td_replay) >= 64:
        td_bot.train_batch(td_replay.sample(64, rng), lr=lr)
      if q_bot in played_bots and len(q_replay) >= 64:
        q_bot.train_batch(q_replay.sample(64, rng), lr=lr)
      if az_bot in played_bots:
        if len(az_val_replay) >= 64:
          az_bot.value_bot.train_batch(az_val_replay.sample(64, rng), lr=lr)
        if len(az_pol_replay) >= 64:
          az_bot.train_policy_batch(az_pol_replay.sample(64, rng), lr=lr)

    # Flush any remaining PPO rollout episodes at end of batch.
    if pg_rollout:
      pg_bot.learn_ppo_batch(
          pg_rollout,
          ppo_epochs=3,
          actor_lr=actor_lr,
          critic_lr=lr,
      )
      pg_rollout.clear()

    elos, wins, played, vs_heur, vs_tree1 = _record_generation(gen, games_so_far)

  print('\n=== FINAL BATCH BENCHMARK LEADERBOARD (9 PLAYERS, DUPLICATE DEALS) ===')
  for rank, key in enumerate(
      sorted(eval_keys, key=lambda k: elos[k], reverse=True), 1
  ):
    w, gp = wins[key], max(played[key], 1)
    vh_w, vh_g = vs_heur[key]
    vt_w, vt_g = vs_tree1[key]
    vh_str = f'{100.0 * vh_w / vh_g:5.1f}%' if vh_g > 0 else '  n/a'
    vt_str = f'{100.0 * vt_w / vt_g:5.1f}%' if vt_g > 0 else '  n/a'
    print(
        f'  {rank}. {eval_pool[key].name:38s} | Elo: {elos[key]:6.1f} '
        f'| Win%: {100.0 * w / gp:5.1f}% ({w:3d}/{gp:3d}) '
        f'| vs Heur: {vh_str} | vs T1: {vt_str}'
    )
  print(f'\nSaved league progression history -> {LEAGUE_HISTORY_PATH}')


def main():
  parser = argparse.ArgumentParser(
      description='Run the Generational PyTorch Bot League in 5,000-game batches.'
  )
  parser.add_argument('--generations', type=int, default=4)
  parser.add_argument('--games-per-gen', type=int, default=5000)
  parser.add_argument('--eval-games', type=int, default=32)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument(
      '--resume',
      action='store_true',
      help='Continue training from existing models/*.pt and append to league_history.json.',
  )
  parser.add_argument(
      '--include-human',
      default=None,
      help='Optional .jsonl/.jsonl.gz of family games to seed the Replay Buffers.',
  )
  args = parser.parse_args()
  run_league(
      generations=args.generations,
      games_per_gen=args.games_per_gen,
      eval_games=args.eval_games,
      seed=args.seed,
      include_human=args.include_human,
      resume=args.resume,
  )


if __name__ == '__main__':
  main()
