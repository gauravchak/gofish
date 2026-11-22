"""Unit tests for the zero-dependency ML/RL stack (TinyMLP, features, and 3 RL bots)."""

import os
import tempfile
import unittest

from bots import create_bot
from game import GameEngine
from ml import (
    ACTION_DIM,
    AFTERSTATE_DIM,
    PolicyGradientBot,
    QLearningBot,
    TDValueBot,
    TinyMLP,
    extract_action_features,
    extract_state_features,
)


class TestTinyMLPAndFeatures(unittest.TestCase):

  def test_tiny_mlp_learns_simple_pattern_and_roundtrips_json(self):
    net = TinyMLP(input_dim=2, hidden_dim=8, output_activation='sigmoid', seed=7)
    data = [([1.0, 0.0], 1.0), ([0.0, 1.0], 0.0), ([1.0, 1.0], 1.0), ([0.0, 0.0], 0.0)]
    for _ in range(150):
      for x, y in data:
        net.train_step(x, y, lr=0.1, l2=0.0)
    self.assertGreater(net.predict([1.0, 0.0]), 0.85)
    self.assertLess(net.predict([0.0, 1.0]), 0.15)

    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'test_net.json')
      net.save(path, metadata={'note': 'unit-test'})
      loaded, meta = TinyMLP.load(path)
      self.assertEqual(meta['note'], 'unit-test')
      self.assertAlmostEqual(
          net.predict([1.0, 0.0]), loaded.predict([1.0, 0.0]), places=4
      )

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


class TestRLBots(unittest.TestCase):

  def test_all_three_rl_bots_take_immediate_win_and_obey_rules(self):
    for key in ('rl_td_value', 'rl_q_learning', 'rl_policy_gradient'):
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
      td = TDValueBot(hidden_dim=8, seed=1)
      ql = QLearningBot(hidden_dim=8, seed=2)
      pg = PolicyGradientBot(hidden_dim=8, seed=3)

      for bot, fname in (
          (td, 'td.json'),
          (ql, 'ql.json'),
          (pg, 'pg.json'),
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
          bot.learn_episode(records, 1.0, lr=0.01)
        elif isinstance(bot, QLearningBot):
          bot.learn_episode(records, 1.0, lr=0.01)
        else:
          bot.learn_episode(records, 1.0, actor_lr=0.01, critic_lr=0.01)

        path = os.path.join(tmpdir, fname)
        bot.save(path, metadata={'trained_games': 1})
        self.assertTrue(os.path.exists(path))
        reloaded = type(bot)(model_path=path)
        self.assertEqual(reloaded.metadata['trained_games'], 1)


if __name__ == '__main__':
  unittest.main()
