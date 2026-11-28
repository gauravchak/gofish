"""Unit tests for the PyTorch ML/RL stack (CardMLP, features, and 3 RL bots)."""

import os
import tempfile
import unittest

from bots import HAS_TORCH, create_bot
from game import GameEngine
from ml.features import (
    ACTION_DIM,
    AFTERSTATE_DIM,
    extract_action_features,
    extract_state_features,
)

if HAS_TORCH:
  import random
  import torch
  import torch.nn.functional as F
  from ml import (
      AlphaZeroLiteBot,
      CardMLP,
      PolicyGradientBot,
      QLearningBot,
      ReplayBuffer,
      TDValueBot,
      load_checkpoint,
      save_checkpoint,
  )


class TestFeatures(unittest.TestCase):

  def test_feature_dimensions(self):
    eng = GameEngine()
    eng.join_player('p0', 'A')
    eng.join_player('p1', 'B')
    eng.start_match(seed=12)

    obs_draw = eng.get_observation(0)
    self.assertEqual(len(extract_state_features(obs_draw)), AFTERSTATE_DIM)
    self.assertEqual(len(extract_action_features(obs_draw, 'T')), ACTION_DIM)
    self.assertEqual(len(extract_action_features(obs_draw, 'S')), ACTION_DIM)

    eng.draw(0, 'T')
    obs_disc = eng.get_observation(0)
    self.assertEqual(len(extract_state_features(obs_disc)), AFTERSTATE_DIM)
    cid = obs_disc['legal_action_ids'][0]
    self.assertEqual(len(extract_action_features(obs_disc, cid)), ACTION_DIM)


@unittest.skipUnless(HAS_TORCH, 'PyTorch not installed')
class TestPyTorchRLBots(unittest.TestCase):

  def test_card_mlp_learns_pattern_and_roundtrips_safe_checkpoint(self):
    torch.manual_seed(7)
    net = CardMLP(input_dim=2, hidden_dim=16, output_activation='sigmoid')
    opt = torch.optim.Adam(net.parameters(), lr=0.05)
    x = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]], dtype=torch.float32
    )
    y = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

    for _ in range(120):
      loss = F.binary_cross_entropy(net(x), y)
      opt.zero_grad()
      loss.backward()
      opt.step()

    self.assertGreater(net.predict_one([1.0, 0.0]), 0.85)
    self.assertLess(net.predict_one([0.0, 1.0]), 0.15)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'test_net.pt')
      save_checkpoint(
          path,
          {'state': net.state_dict(), 'metadata': {'note': 'unit-test'}},
      )
      ckpt = load_checkpoint(path)
      self.assertEqual(ckpt['metadata']['note'], 'unit-test')
      reloaded = CardMLP(input_dim=2, hidden_dim=16, output_activation='sigmoid')
      reloaded.load_state_dict(ckpt['state'])
      self.assertAlmostEqual(
          net.predict_one([1.0, 0.0]), reloaded.predict_one([1.0, 0.0]), places=5
      )

  def test_all_three_rl_bots_take_immediate_win_and_obey_rules(self):
    for key in ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient', 'rl_alphazero'):
      bot = create_bot(key)
      eng = GameEngine()
      eng.join_player('p0', bot.name, player_type=bot.player_type)
      eng.join_player('p1', 'Human', player_type='human')
      eng.start_match(seed=33)

      # 1 card away from winning with winning card on Shown
      eng.hands[0] = [
          [1, '♣'],
          [12, '♣'],
          [13, '♣'],
          [9, '♥'],
          [9, '♦'],
          [2, '♠'],
      ]
      eng.shown = [[9, '♠']]

      self.assertEqual(bot.choose_draw(eng.get_observation(0)), 'S', key)
      self.assertTrue(eng.draw(0, 'S')['ok'])
      disc = bot.choose_discard(eng.get_observation(0))
      self.assertEqual(disc, [2, '♠'], key)
      self.assertTrue(eng.discard(0, disc, by_index=False)['ok'])
      self.assertEqual(eng.winner_idx, 0)

  def test_all_three_rl_bots_complete_episode_and_save_reload(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      td = TDValueBot(hidden_dim=16, seed=1)
      ql = QLearningBot(hidden_dim=16, seed=2)
      pg = PolicyGradientBot(hidden_dim=16, seed=3)

      for bot, fname in (
          (td, 'td.pt'),
          (ql, 'ql.pt'),
          (pg, 'pg.pt'),
      ):
        eng = GameEngine()
        eng.join_player('p0', bot.name, player_type=bot.player_type)
        eng.join_player('p1', 'Other', player_type='bot:random')
        eng.start_match(seed=7)

        records = []
        while eng.winner is None:
          p = eng.turn
          obs = eng.get_observation(p)
          if p == 0:
            if eng.phase == 'DRAW':
              if isinstance(bot, (QLearningBot, PolicyGradientBot)):
                act, rec = bot.choose_draw(obs, return_record=True)
                records.append(rec)
              else:
                act = bot.choose_draw(obs)
              self.assertTrue(eng.draw(p, act)['ok'])
            else:
              if isinstance(bot, TDValueBot):
                act, feats = bot.choose_discard(obs, return_features=True)
                records.append(feats)
              else:
                act, rec = bot.choose_discard(obs, return_record=True)
                records.append(rec)
              self.assertTrue(eng.discard(p, act, by_index=False)['ok'])
          else:
            if eng.phase == 'DRAW':
              eng.draw(p, obs['legal_actions'][0])
            else:
              eng.discard(p, obs['legal_actions'][0], by_index=False)

        if isinstance(bot, TDValueBot):
          bot.learn_episode(records, 1.0, lr=0.003)
        elif isinstance(bot, QLearningBot):
          bot.learn_episode(records, 1.0, lr=0.003)
        else:
          bot.learn_episode(records, 1.0, actor_lr=0.002, critic_lr=0.003)

        path = os.path.join(tmpdir, fname)
        bot.save(path, metadata={'trained_games': 1})
        self.assertTrue(os.path.exists(path))
        reloaded = type(bot)(model_path=path)
        self.assertEqual(reloaded.metadata['trained_games'], 1)

  def test_replay_buffer_and_clone_frozen(self):
    buf = ReplayBuffer(capacity=5)
    for i in range(10):
      buf.push(i)
    self.assertEqual(len(buf), 5)
    sample = buf.sample(3, random.Random(1))
    self.assertEqual(len(sample), 3)
    self.assertTrue(all(v >= 5 for v in sample))

    td = TDValueBot(hidden_dim=16, seed=11)
    snap = td.clone_frozen('TD_G1')
    self.assertEqual(snap.name, 'TD_G1')
    az = AlphaZeroLiteBot(td_bot=snap, draw_sample=4)
    self.assertEqual(az.player_type, 'bot:rl_alphazero')

  def test_bot_thoughts_and_human_replay_loader(self):
    from scripts.league import load_offline_td_samples
    from game import TrajectoryLogger

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'human.jsonl')
      logger = TrajectoryLogger(path)
      eng = GameEngine(logger=logger)
      bots = [create_bot('rl_alphazero'), create_bot('rl_policy_gradient')]
      eng.join_player('b0', bots[0].name, player_type=bots[0].player_type)
      eng.join_player('b1', bots[1].name, player_type=bots[1].player_type)
      eng.start_match(seed=19)

      for _ in range(4):
        if eng.winner is not None:
          break
        p = eng.turn
        obs = eng.get_observation(p)
        if eng.phase == 'DRAW':
          act = bots[p].choose_draw(obs)
          eng.draw(p, act)
        else:
          act = bots[p].choose_discard(obs)
          eng.discard(p, act, by_index=False)
        self.assertTrue(len(bots[p].last_thought) > 0)

      # Finish match and verify offline human-trajectory loader reads DISCARD samples
      while eng.winner is None:
        p = eng.turn
        obs = eng.get_observation(p)
        if eng.phase == 'DRAW':
          eng.draw(p, bots[p].choose_draw(obs))
        else:
          eng.discard(p, bots[p].choose_discard(obs), by_index=False)

      samples = load_offline_td_samples(path)
      self.assertGreater(len(samples), 0)
      self.assertEqual(len(samples[0][0]), AFTERSTATE_DIM)

  def test_opponent_memory_danger_feature_and_ppo_ddqn_alphazero_training(self):
    from ml.features import AFTERSTATE_FEATURE_NAMES, extract_afterstate_features
    from scripts.league import _solve_balanced_elos

    # 1. Verify opp_danger_exact fires when discarding the 3rd card of opponent's Shown pair!
    # Opponent picked 7♥ (id=6) and 7♦ (id=19); discarding 7♣ (id=32) completes 7-7-7!
    danger_idx = AFTERSTATE_FEATURE_NAMES.index('opp_danger_exact')
    safe_idx = AFTERSTATE_FEATURE_NAMES.index('safe_discard_score')
    hand6 = [0, 1, 2, 10, 20, 30]
    unseen = ((1 << 52) - 1) ^ sum(1 << c for c in hand6)
    f_danger = extract_afterstate_features(
        hand6, unseen, 30, next_wanted={6, 19}, next_discarded=set(), discarded_id=32
    )
    f_safe = extract_afterstate_features(
        hand6, unseen, 30, next_wanted={6, 19}, next_discarded={40}, discarded_id=41
    )
    self.assertEqual(f_danger[danger_idx], 1.0)
    self.assertEqual(f_safe[danger_idx], 0.0)
    self.assertGreater(f_safe[safe_idx], f_danger[safe_idx])

    # 2. Verify PPO batch learning + AlphaZero self-play policy/value training + save/reload
    az = AlphaZeroLiteBot(hidden_dim=16, draw_sample=2, top_k_discards=2, seed=9)
    az.stochastic = True
    pg = PolicyGradientBot(hidden_dim=16, seed=8)
    pg.stochastic = True

    pg_eps = []
    az_vals, az_pols = [], []
    for s in (101, 102):
      eng = GameEngine()
      eng.join_player('p0', az.name, player_type=az.player_type)
      eng.join_player('p1', pg.name, player_type=pg.player_type)
      eng.start_match(seed=s)
      az_s, az_after, pg_s = [], [], []
      while eng.winner is None:
        p = eng.turn
        obs = eng.get_observation(p)
        if p == 0:
          if eng.phase == 'DRAW':
            act, rec = az.choose_draw(obs, return_search=True)
            az_s.append(rec)
            eng.draw(p, act)
          else:
            act, rec = az.choose_discard(obs, return_search=True)
            az_s.append(rec)
            az_after.append(rec['after_feats'])
            eng.discard(p, act, by_index=False)
        else:
          if eng.phase == 'DRAW':
            act, rec = pg.choose_draw(obs, return_record=True)
            pg_s.append(rec)
            eng.draw(p, act)
          else:
            act, rec = pg.choose_discard(obs, return_record=True)
            pg_s.append(rec)
            eng.discard(p, act, by_index=False)
      out_az = 1.0 if eng.winner_idx == 0 else (0.0 if eng.winner_idx == 1 else 0.5)
      rew_pg = 1.0 if eng.winner_idx == 1 else (-1.0 if eng.winner_idx == 0 else 0.0)
      vs, ps = az.build_selfplay_samples(az_s, az_after, out_az)
      az_vals.extend(vs)
      az_pols.extend(ps)
      pg_eps.append((pg_s, rew_pg))

    self.assertGreaterEqual(pg.learn_ppo_batch(pg_eps, ppo_epochs=2), 0.0)
    self.assertGreaterEqual(az.value_bot.train_batch(az_vals, lr=0.001), 0.0)
    self.assertGreaterEqual(az.train_policy_batch(az_pols, lr=0.001), 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
      az_path = os.path.join(tmpdir, 'az.pt')
      az.save(az_path, metadata={'generation': 1})
      reloaded = AlphaZeroLiteBot(model_path=az_path)
      self.assertEqual(reloaded.metadata['generation'], 1)

    # 3. Verify Bradley-Terry balanced Elo solver is order-independent!
    res = [('a', 'b', 0), ('a', 'b', 0), ('b', 'c', 0), ('a', 'c', 0)]
    e1 = _solve_balanced_elos(['a', 'b', 'c'], res)
    e2 = _solve_balanced_elos(['a', 'b', 'c'], list(reversed(res)))
    for k in ('a', 'b', 'c'):
      self.assertAlmostEqual(e1[k], e2[k], places=4)
    self.assertGreater(e1['a'], e1['b'])
    self.assertGreater(e1['b'], e1['c'])

  def test_xray_grpo_and_rollout_distillation_end_to_end(self):
    from ml import DistilledBot, XRayGRPOBot, evaluate_state_with_rollouts
    from scripts.train_variance_reduced import _play_grpo_mirror_pair

    # 1. X-Ray GRPO: paired mirror deal + omniscient critic update + save/reload
    grpo = XRayGRPOBot(hidden_dim=16, seed=11)
    grpo.stochastic = True
    opp = create_bot('heuristic')
    pair = _play_grpo_mirror_pair(grpo, opp, deal_seed=321)
    self.assertGreaterEqual(grpo.learn_grpo_batch([pair], ppo_epochs=2), 0.0)

    # 2. Fair Rollout Teacher + Soft-Target Distillation Student + save/reload
    eng = GameEngine()
    eng.join_player('p0', 'A')
    eng.join_player('p1', 'B')
    eng.start_match(seed=44)
    obs_draw = eng.get_observation(0)
    sample_draw = evaluate_state_with_rollouts(
        obs_draw, opp, num_worlds=3, rng=random.Random(1)
    )
    self.assertIsNotNone(sample_draw)
    self.assertEqual(len(sample_draw['evs']), 2)

    eng.draw(0, 'T')
    obs_disc = eng.get_observation(0)
    sample_disc = evaluate_state_with_rollouts(
        obs_disc, opp, num_worlds=3, rng=random.Random(2)
    )
    self.assertIsNotNone(sample_disc)
    self.assertEqual(len(sample_disc['evs']), 7)

    student = DistilledBot(hidden_dim=16, seed=12)
    loss = student.train_on_teacher_dataset([sample_draw, sample_disc], epochs=3)
    self.assertGreaterEqual(loss, 0.0)

    with tempfile.TemporaryDirectory() as tmpdir:
      gp = os.path.join(tmpdir, 'grpo.pt')
      dp = os.path.join(tmpdir, 'distill.pt')
      grpo.save(gp, metadata={'batch': 1})
      student.save(dp, metadata={'iteration': 1})
      self.assertEqual(XRayGRPOBot(model_path=gp).metadata['batch'], 1)
      self.assertEqual(DistilledBot(model_path=dp).metadata['iteration'], 1)

    # 3. Verify luck-adjusted skill ('excess advantage') math on paired duplicate deals
    from scripts.evaluate import compute_luck_adjusted_stats

    stats = compute_luck_adjusted_stats([
        ('A', 'A'),  # 2-0 skill sweep by A
        ('A', 'A'),  # 2-0 skill sweep by A
        ('A', 'A'),  # 2-0 skill sweep by A
        ('B', 'B'),  # 0-2 skill sweep by B
        ('A', 'B'),  # 1-1 luck split
        ('B', 'A'),  # 1-1 luck split
    ])
    self.assertEqual(stats['sweeps_a'], 3)
    self.assertEqual(stats['sweeps_b'], 1)
    self.assertEqual(stats['luck_splits'], 2)
    self.assertEqual(stats['decisive_win_pct_a'], 75.0)
    self.assertAlmostEqual(stats['excess_per_100'], 33.33, places=2)
    self.assertAlmostEqual(stats['mcnemar_z'], 1.0, places=2)


if __name__ == '__main__':
  unittest.main()
