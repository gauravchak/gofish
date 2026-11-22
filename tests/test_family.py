"""Unit tests for local family.json player profiles and Elo rating calculations."""

import json
import os
import tempfile
import unittest

from game import FamilyStore, GameEngine, expected_score


class TestFamilyStoreAndElo(unittest.TestCase):

  def test_expected_score_symmetry(self):
    self.assertAlmostEqual(expected_score(1000, 1000), 0.5)
    self.assertAlmostEqual(
        expected_score(1200, 1000) + expected_score(1000, 1200), 1.0
    )
    self.assertGreater(expected_score(1200, 1000), 0.75)

  def test_case_insensitive_name_resolution_and_persistence(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'family.json')
      store = FamilyStore(path)
      self.assertEqual(store.ensure_player('Alice', 'human'), 'Alice')
      self.assertEqual(store.ensure_player('  alice ', 'human'), 'Alice')
      self.assertEqual(store.ensure_player('Tree Bot (1-Level)', 'bot:tree_1ply'), 'Tree Bot (1-Level)')

      # Only human players appear in the lobby name dropdown.
      self.assertEqual(store.list_human_names(), ['Alice'])

      # Reloading from disk preserves saved profiles.
      reloaded = FamilyStore(path)
      self.assertEqual(reloaded.list_human_names(), ['Alice'])
      self.assertEqual(len(reloaded.leaderboard()), 2)

  def test_two_player_elo_update_is_exact_and_zero_sum(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'family.json')
      store = FamilyStore(path, initial_elo=1000.0, k_factor=32.0)

      deltas = store.record_match(
          'g1',
          [
              {'name': 'Alice', 'type': 'human'},
              {'name': 'Bob', 'type': 'human'},
          ],
          winner_idx=0,
      )
      self.assertEqual(deltas['Alice'], 16.0)
      self.assertEqual(deltas['Bob'], -16.0)
      self.assertEqual(store.data['players']['Alice']['elo'], 1016.0)
      self.assertEqual(store.data['players']['Bob']['elo'], 984.0)
      self.assertEqual(store.data['players']['Alice']['wins'], 1)
      self.assertEqual(store.data['players']['Bob']['losses'], 1)

      # Upset: lower-rated Bob beating Alice earns more than 16 points!
      deltas2 = store.record_match(
          'g2',
          [
              {'name': 'Alice', 'type': 'human'},
              {'name': 'Bob', 'type': 'human'},
          ],
          winner_idx=1,
      )
      self.assertGreater(deltas2['Bob'], 16.0)
      self.assertAlmostEqual(deltas2['Alice'] + deltas2['Bob'], 0.0, places=5)

  def test_multiplayer_elo_and_draw_are_zero_sum(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'family.json')
      store = FamilyStore(path, initial_elo=1000.0, k_factor=32.0)

      # 3-player game: equal ratings -> winner +16, each of 2 losers -8.
      deltas = store.record_match(
          'g3',
          [
              {'name': 'Alice', 'type': 'human'},
              {'name': 'Bob', 'type': 'human'},
              {'name': 'Charlie', 'type': 'human'},
          ],
          winner_idx=0,
      )
      self.assertEqual(deltas['Alice'], 16.0)
      self.assertEqual(deltas['Bob'], -8.0)
      self.assertEqual(deltas['Charlie'], -8.0)
      self.assertAlmostEqual(sum(deltas.values()), 0.0, places=5)

      # Draw game preserves total Elo and increments draw counts.
      draw_deltas = store.record_match(
          'g4',
          [
              {'name': 'Alice', 'type': 'human'},
              {'name': 'Bob', 'type': 'human'},
              {'name': 'Charlie', 'type': 'human'},
          ],
          winner_idx=None,
      )
      self.assertAlmostEqual(sum(draw_deltas.values()), 0.0, places=5)
      self.assertEqual(store.data['players']['Alice']['draws'], 1)
      self.assertEqual(store.data['players']['Bob']['draws'], 1)
      self.assertEqual(store.data['players']['Charlie']['draws'], 1)

  def test_engine_integration_records_elo_and_blocks_duplicate_human_name(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      path = os.path.join(tmpdir, 'family.json')
      store = FamilyStore(path)
      eng = GameEngine(family_store=store)

      self.assertTrue(eng.join_player('ip1', 'Alice', 'human')['ok'])
      # Another device cannot join with the exact same human name in one lobby.
      dup = eng.join_player('ip2', 'alice', 'human')
      self.assertFalse(dup['ok'])
      self.assertTrue(eng.join_player('bot:1', 'Tree Bot (1-Level)', 'bot:tree_1ply')['ok'])
      self.assertTrue(eng.start_match(seed=5))

      # Force an immediate Alice win
      eng.hands[0] = [
          [1, '♣'],
          [12, '♣'],
          [13, '♣'],
          [9, '♥'],
          [9, '♦'],
          [2, '♠'],
      ]
      eng.shown = [[9, '♠']]
      self.assertTrue(eng.draw(0, 'S')['ok'])
      self.assertTrue(eng.discard(0, [2, '♠'], by_index=False)['ok'])

      ui = eng.get_ui_state('ip1')
      self.assertEqual(ui['saved_human_names'], ['Alice'])
      self.assertEqual(len(ui['elo_updates']), 2)
      self.assertEqual(ui['elo_updates'][0]['name'], 'Alice')
      self.assertEqual(ui['elo_updates'][0]['delta'], 16.0)

      # Verify JSON file on disk is valid and matches
      with open(path, 'r', encoding='utf-8') as f:
        saved = json.load(f)
      self.assertEqual(saved['players']['Alice']['elo'], 1016.0)
      self.assertEqual(saved['players']['Tree Bot (1-Level)']['elo'], 984.0)


if __name__ == '__main__':
  unittest.main()
