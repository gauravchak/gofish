"""Generational PyTorch Bot League: Experience Replay + Hall-of-Fame Self-Play.

Lets all 3 PyTorch RL bots play against EACH OTHER, past Hall-of-Fame snapshots,
and our rules/tree bots across multiple generations — tracking exact Elo progress
generation-by-generation and only promoting models that beat their previous best!

Key ML concepts demonstrated cleanly:
  1. Experience Replay Buffer (mini-batch size 64) for stable TD Value & DQN learning
  2. Hall-of-Fame opponent pool (prevents catastrophic forgetting in self-play)
  3. Fixed-seed Benchmark Tournament after every generation (apples-to-apples Elo)
  4. King-of-the-Hill checkpoint gating (only save new models when Elo improves!)
  5. AlphaZero-Lite hybrid evaluation (trained PyTorch TD Value Net + 1-Level Tree)

Run:
  python3 -m scripts.league --generations 6 --games-per-gen 2000
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


def _update_elo(elos, key_a, key_b, winner_slot, k=20.0):
  ra, rb = elos[key_a], elos[key_b]
  ea = expected_score(ra, rb)
  score_a = 1.0 if winner_slot == 0 else (0.0 if winner_slot == 1 else 0.5)
  delta = k * (score_a - ea)
  elos[key_a] = ra + delta
  elos[key_b] = rb - delta


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
    bot0, bot1, seed, td_replay, q_replay, lr, actor_lr
):
  """Plays one league training game, pushing to ReplayBuffers and updating A2C."""
  eng = GameEngine(logger=None)
  eng.join_player('p0', bot0.name, player_type=bot0.player_type)
  eng.join_player('p1', bot1.name, player_type=bot1.player_type)
  eng.start_match(seed=seed)

  bots = [bot0, bot1]
  has_td_or_pg = any(isinstance(b, (TDValueBot, PolicyGradientBot)) for b in bots)
  all_afterstates = [[], []]
  q_transitions = [[], []]
  pg_steps = [[], []]

  while eng.winner is None:
    p = eng.turn
    bot = bots[p]
    obs = eng.get_observation(p)

    if eng.phase == 'DRAW':
      if isinstance(bot, QLearningBot):
        action, rec = bot.choose_draw(obs, return_record=True)
        q_transitions[p].append(rec)
      elif isinstance(bot, PolicyGradientBot):
        action, rec = bot.choose_draw(obs, return_record=True)
        pg_steps[p].append(rec)
      else:
        action = bot.choose_draw(obs)
      res = eng.draw(p, action)
    else:
      if isinstance(bot, TDValueBot):
        action, feats = bot.choose_discard(obs, return_features=True)
        all_afterstates[p].append(feats)
      elif isinstance(bot, QLearningBot):
        action, rec = bot.choose_discard(obs, return_record=True)
        q_transitions[p].append(rec)
        if has_td_or_pg:
          all_afterstates[p].append(rec['feats'][:AFTERSTATE_DIM])
      elif isinstance(bot, PolicyGradientBot):
        action, rec = bot.choose_discard(obs, return_record=True)
        pg_steps[p].append(rec)
        all_afterstates[p].append(rec['after_feats'])
      else:
        action = bot.choose_discard(obs)
        if has_td_or_pg:
          all_afterstates[p].append(_afterstate_from_discard(obs, action))
      res = eng.discard(p, action, by_index=False)

    if not res['ok']:
      raise RuntimeError(f'{bot.name} made illegal move {action}: {res}')

  # Push off-policy experiences into ReplayBuffers & run on-policy A2C update.
  for seat, bot in enumerate(bots):
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
      bot.learn_episode(pg_steps[seat], rew_pm, actor_lr=actor_lr, critic_lr=lr)

  return eng.winner_idx


def _play_eval_pair(pool, key_a, key_b, n_games, seed_base):
  """Plays n_games alternating-seat evaluation matches between key_a and key_b."""
  results = []
  for g in range(n_games):
    swap = (g % 2) == 1
    k0, k1 = (key_b, key_a) if swap else (key_a, key_b)
    eng = GameEngine(logger=None)
    eng.join_player('p0', pool[k0].name, player_type=pool[k0].player_type)
    eng.join_player('p1', pool[k1].name, player_type=pool[k1].player_type)
    eng.start_match(seed=seed_base + g)

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
    games_per_pair=16,
    seed=777000,
):
  """Fixed-seed apples-to-apples round-robin tournament across all 9 players.

  Caches fixed-vs-fixed matches (since Random/Beginner/Heuristic/Tree never
  change across generations), making every generation eval ~5x faster!
  """
  for key in keys:
    bot = pool[key]
    if hasattr(bot, 'epsilon'):
      bot.epsilon = 0.0
    if hasattr(bot, 'stochastic'):
      bot.stochastic = False

  elos = {k: 1000.0 for k in keys}
  wins = {k: 0 for k in keys}
  played = {k: 0 for k in keys}
  vs_heur = {k: [0, 0] for k in keys}

  slow_keys = {'tree_2ply', 'rl_alphazero'}
  all_results = []
  seed_cursor = seed

  for i, key_a in enumerate(keys):
    for key_b in keys[i + 1 :]:
      n_games = (
          min(games_per_pair, 8)
          if (key_a in slow_keys or key_b in slow_keys)
          else games_per_pair
      )
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

  # Replay all pair results in deterministic order to compute balanced Elo ratings.
  for k0, k1, winner_idx in all_results:
    _update_elo(elos, k0, k1, winner_idx, k=20.0)
    played[k0] += 1
    played[k1] += 1
    if winner_idx == 0:
      wins[k0] += 1
    elif winner_idx == 1:
      wins[k1] += 1

    if k0 == 'heuristic':
      vs_heur[k1][1] += 1
      if winner_idx == 1:
        vs_heur[k1][0] += 1
    elif k1 == 'heuristic':
      vs_heur[k0][1] += 1
      if winner_idx == 0:
        vs_heur[k0][0] += 1

  return elos, wins, played, vs_heur


def run_league(generations=6, games_per_gen=2000, seed=42, include_human=None):
  """Trains all 3 PyTorch RL bots in a generational bot-vs-bot league."""
  torch.set_num_threads(1)
  rng = random.Random(seed)
  t_start = time.time()

  td_bot = TDValueBot(hidden_dim=32, seed=seed + 1)
  q_bot = QLearningBot(hidden_dim=32, seed=seed + 2)
  pg_bot = PolicyGradientBot(hidden_dim=32, seed=seed + 3)
  az_bot = AlphaZeroLiteBot(td_bot=td_bot, draw_sample=4)

  td_replay = ReplayBuffer(capacity=25000)
  q_replay = ReplayBuffer(capacity=25000)
  if include_human:
    human_samples = load_offline_td_samples(include_human)
    if human_samples:
      td_replay.extend(human_samples)
      print(
          f'Seeded TD Replay Buffer with {len(human_samples)} steps from {include_human}',
          flush=True,
      )
  hall_of_fame = []  # Frozen past champions for diverse league sparring

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
  }
  eval_pool = {**fixed, **learners, 'rl_alphazero': az_bot}
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
  best_elo = {k: -1.0 for k in learners}
  history = []

  def _record_generation(gen_idx, games_total):
    elos, wins, played, vs_heur = evaluate_generation(
        eval_pool,
        eval_keys,
        fixed_keys=fixed_keys,
        cached_fixed_results=cached_fixed_results,
        games_per_pair=16,
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
    }
    history.append(entry)

    # King-of-the-Hill: only promote checkpoints when a bot hits a new best Elo!
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

      # Add a frozen copy of this generation's top RL bot to the Hall of Fame.
      top_rl = max(learners.keys(), key=lambda k: elos[k])
      hall_of_fame.append(learners[top_rl].clone_frozen(f'{top_rl}_g{gen_idx}'))
      if len(hall_of_fame) > 4:
        hall_of_fame.pop(0)

    print(
        f'Gen {gen_idx} ({games_total:5d} games | {entry["elapsed_s"]:5.1f}s) | '
        f'TD:{elos["rl_td_value"]:6.1f} | Q:{elos["rl_q_learning"]:6.1f} | '
        f'PG:{elos["rl_policy_gradient"]:6.1f} | AZ:{elos["rl_alphazero"]:6.1f} | '
        f'Heur:{elos["heuristic"]:6.1f} | T1:{elos["tree_1ply"]:6.1f} | '
        f'T2:{elos["tree_2ply"]:6.1f}'
        + (f'  [Promoted: {", ".join(promoted)}]' if promoted else ''),
        flush=True,
    )
    return elos, wins, played, vs_heur

  print(
      f'=== PyTorch Bot League: {generations} Generations x {games_per_gen:,} games ===',
      flush=True,
  )
  _record_generation(0, 0)

  learner_keys = ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient')
  games_so_far = 0

  for gen in range(1, generations + 1):
    frac = gen / generations
    eps = max(0.03, 0.22 * (1.0 - 0.85 * frac))
    temp = max(0.45, 1.20 * (1.0 - 0.62 * frac))
    lr = max(0.0005, 0.0035 * (1.0 - 0.82 * frac))
    actor_lr = lr * 0.8

    td_bot.epsilon = eps
    q_bot.epsilon = eps
    pg_bot.stochastic = True
    pg_bot.temperature = temp

    for g in range(1, games_per_gen + 1):
      games_so_far += 1
      lk = learner_keys[(g - 1) % len(learner_keys)]
      peer_keys = [k for k in learner_keys if k != lk]

      # Progressive bot-vs-bot league matchmaking!
      r = rng.random()
      if gen == 1:
        if r < 0.45:
          opp = eval_pool[rng.choice(peer_keys)]
        elif r < 0.75:
          opp = fixed['novice']
        else:
          opp = fixed['random']
      elif gen == 2:
        if r < 0.45:
          opp = eval_pool[rng.choice(peer_keys)]
        elif r < 0.75:
          opp = fixed['heuristic']
        elif hall_of_fame and r < 0.90:
          opp = rng.choice(hall_of_fame)
        else:
          opp = fixed['novice']
      else:
        # Gen 3+: Heavy RL-vs-RL + Hall-of-Fame + Heuristic + 1-Level Tree!
        if r < 0.45:
          opp = eval_pool[rng.choice(peer_keys)]
        elif hall_of_fame and r < 0.68:
          opp = rng.choice(hall_of_fame)
        elif r < 0.90:
          opp = fixed['heuristic']
        else:
          opp = fixed['tree_1ply']

      b0, b1 = (eval_pool[lk], opp) if (g % 2 == 0) else (opp, eval_pool[lk])
      _play_league_game(
          b0,
          b1,
          seed=seed * 10000 + games_so_far,
          td_replay=td_replay,
          q_replay=q_replay,
          lr=lr,
          actor_lr=actor_lr,
      )

      # Mini-batch Experience Replay updates every 4 games (batch_size=64)!
      if g % 4 == 0:
        if len(td_replay) >= 64:
          for _ in range(2):
            td_bot.train_batch(td_replay.sample(64, rng), lr=lr)
        if len(q_replay) >= 64:
          for _ in range(2):
            q_bot.train_batch(q_replay.sample(64, rng), lr=lr)

    elos, wins, played, vs_heur = _record_generation(gen, games_so_far)

  # Save full generation-by-generation history for the /league chart!
  os.makedirs(MODELS_DIR, exist_ok=True)
  with open(LEAGUE_HISTORY_PATH, 'w', encoding='utf-8') as f:
    json.dump(
        {
            'generations': generations,
            'games_per_gen': games_per_gen,
            'player_names': {k: eval_pool[k].name for k in eval_keys},
            'history': history,
        },
        f,
        indent=2,
    )
    f.write('\n')

  print('\n=== FINAL GENERATION BENCHMARK LEADERBOARD (9 PLAYERS) ===')
  for rank, key in enumerate(
      sorted(eval_keys, key=lambda k: elos[k], reverse=True), 1
  ):
    w, gp = wins[key], max(played[key], 1)
    vh_w, vh_g = vs_heur[key]
    vh_str = f'{100.0 * vh_w / vh_g:5.1f}%' if vh_g > 0 else '  n/a'
    print(
        f'  {rank}. {eval_pool[key].name:34s} | Elo: {elos[key]:6.1f} '
        f'| Win%: {100.0 * w / gp:5.1f}% ({w:3d}/{gp:3d}) | vs Heuristic: {vh_str}'
    )
  print(f'\nSaved league progression history -> {LEAGUE_HISTORY_PATH}')


def main():
  parser = argparse.ArgumentParser(
      description='Run the Generational PyTorch Bot League.'
  )
  parser.add_argument('--generations', type=int, default=6)
  parser.add_argument('--games-per-gen', type=int, default=2000)
  parser.add_argument('--seed', type=int, default=42)
  parser.add_argument(
      '--include-human',
      default=None,
      help='Optional .jsonl/.jsonl.gz of family games to seed the Replay Buffer.',
  )
  args = parser.parse_args()
  run_league(
      generations=args.generations,
      games_per_gen=args.games_per_gen,
      seed=args.seed,
      include_human=args.include_human,
  )


if __name__ == '__main__':
  main()
