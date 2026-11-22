"""Unit tests for rules-based and 1-/2-level tree computer players."""

import json
import os
import tempfile
import unittest

from bots import BOT_OPTIONS, HeuristicBot, RandomBot, TreeBot, create_bot
from bots.heuristic import (
    PAIR_COMPLETER_MASKS,
    VALID_SET_MASKS,
    get_unseen_mask,
    is_winning_hand_ids,
    score_hand,
)
from game import GameEngine, SCHEMA_VERSION, TrajectoryLogger, card_to_id


class TestHeuristicTables(unittest.TestCase):

  def test_exact_count_of_valid_sets(self):
    # 52 three-of-a-kinds + 48 suited sequences (12 per suit) = 100 valid sets.
    self.assertEqual(len(VALID_SET_MASKS), 100)
    self.assertGreater(len(PAIR_COMPLETER_MASKS), 0)

  def test_winning_hand_scores_highest(self):
    win_hand = [
        card_to_id(c)
        for c in [
            [1, '♣'],
            [12, '♣'],
            [13, '♣'],
            [9, '♥'],
            [9, '♦'],
            [9, '♠'],
        ]
    ]
    almost_hand = [
        card_to_id(c)
        for c in [
            [1, '♣'],
            [12, '♣'],
            [13, '♣'],
            [9, '♥'],
            [9, '♦'],
            [2, '♠'],
        ]
    ]
    all_unseen = (1 << 52) - 1
    self.assertTrue(is_winning_hand_ids(win_hand))
    self.assertFalse(is_winning_hand_ids(almost_hand))
    self.assertGreater(
        score_hand(win_hand, all_unseen), score_hand(almost_hand, all_unseen)
    )


class TestBotsBehavior(unittest.TestCase):

  def test_registry_creates_all_ui_options(self):
    for opt in BOT_OPTIONS:
      bot = create_bot(opt['key'])
      self.assertTrue(bot.player_type.startswith('bot:'))

  def test_all_bots_take_immediate_winning_shown_card_and_win(self):
    for key in ('heuristic', 'tree_1ply', 'tree_2ply'):
      bot = create_bot(key)
      eng = GameEngine()
      eng.join_player('bot:0', bot.name, player_type=bot.player_type)
      eng.join_player('human:1', 'Human', player_type='human')
      eng.start_match(seed=10)

      # Give bot 1 card away from winning, with the winning card on Shown!
      eng.hands[0] = [
          [1, '♣'],
          [12, '♣'],
          [13, '♣'],
          [9, '♥'],
          [9, '♦'],
          [2, '♠'],
      ]
      eng.shown = [[9, '♠']]

      draw_action = bot.choose_draw(eng.get_observation(0))
      self.assertEqual(draw_action, 'S', f'{key} missed winning Shown card')
      self.assertTrue(eng.draw(0, draw_action)['ok'])

      discard_action = bot.choose_discard(eng.get_observation(0))
      self.assertEqual(
          discard_action, [2, '♠'], f'{key} missed winning discard'
      )
      res = eng.discard(0, discard_action, by_index=False)
      self.assertTrue(res['ok'])
      self.assertEqual(eng.winner_idx, 0)

  def test_bots_respect_forbidden_shown_rediscard(self):
    for key in ('random', 'heuristic', 'tree_1ply', 'tree_2ply'):
      bot = create_bot(key)
      eng = GameEngine()
      eng.join_player('bot:0', bot.name, player_type=bot.player_type)
      eng.join_player('human:1', 'Human', player_type='human')
      eng.start_match(seed=21)

      shown_top = list(eng.shown[-1])
      self.assertTrue(eng.draw(0, 'S')['ok'])
      obs = eng.get_observation(0)
      discard = bot.choose_discard(obs)
      self.assertNotEqual(discard, shown_top)
      self.assertTrue(eng.discard(0, discard, by_index=False)['ok'])

  def test_unseen_mask_tracks_opponent_public_shown_pick(self):
    eng = GameEngine()
    eng.join_player('p0', 'Alice', player_type='human')
    eng.join_player('p1', 'Bot', player_type='bot:tree_1ply')
    eng.start_match(seed=3)

    picked = list(eng.shown[-1])
    picked_id = card_to_id(picked)
    self.assertTrue(eng.draw(0, 'S')['ok'])
    other = eng.get_observation(0)['legal_actions'][0]
    self.assertTrue(eng.discard(0, other, by_index=False)['ok'])

    # Player 1 knows player 0 still holds `picked`, so it must NOT be in unseen_mask.
    obs1 = eng.get_observation(1)
    unseen_mask = get_unseen_mask(obs1)
    self.assertEqual(unseen_mask & (1 << picked_id), 0)

  def test_bot_vs_bot_match_logs_clean_v2_trajectories(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      log_path = os.path.join(tmpdir, 'bot_trajectories.jsonl')
      logger = TrajectoryLogger(log_path)
      eng = GameEngine(logger=logger)
      bots = [TreeBot(depth=1), HeuristicBot()]
      eng.join_player('bot:0', bots[0].name, player_type=bots[0].player_type)
      eng.join_player('bot:1', bots[1].name, player_type=bots[1].player_type)
      eng.start_match(seed=77)

      while eng.winner is None:
        p = eng.turn
        obs = eng.get_observation(p)
        if eng.phase == 'DRAW':
          act = bots[p].choose_draw(obs)
          self.assertTrue(eng.draw(p, act)['ok'])
        else:
          act = bots[p].choose_discard(obs)
          self.assertTrue(eng.discard(p, act, by_index=False)['ok'])

      self.assertEqual(logger.saved_games, 1)
      with open(log_path, 'r', encoding='utf-8') as f:
        rows = [json.loads(line) for line in f if line.strip()]
      self.assertGreater(len(rows), 0)
      for row in rows:
        self.assertEqual(row['schema_version'], SCHEMA_VERSION)
        self.assertEqual(row['player_types'], ['bot:tree_1ply', 'bot:heuristic'])
        self.assertIn(row['player_type'], ('bot:tree_1ply', 'bot:heuristic'))
        if row['phase'] == 'DRAW':
          self.assertEqual(row['shown_card'], row['discard_pile'][-1])

  def test_engine_remove_player_in_lobby(self):
    eng = GameEngine()
    eng.join_player('human:1', 'Alice', player_type='human')
    eng.join_player('bot:1', 'Tree Bot (1-Level)', player_type='bot:tree_1ply')
    self.assertEqual(len(eng.players), 2)
    self.assertTrue(eng.remove_player('bot:1')['ok'])
    self.assertEqual(len(eng.players), 1)
    self.assertFalse(eng.remove_player('missing')['ok'])


if __name__ == '__main__':
  unittest.main()
