"""Unit tests for rules, headless GameEngine, and leak-free TrajectoryLogger."""

import json
import os
import tempfile
import unittest

from game import (
    GameEngine,
    SCHEMA_VERSION,
    TrajectoryLogger,
    card_to_id,
    check_win,
    create_deck,
    find_winning_sets,
    id_to_card,
    is_valid_set,
    sort_cards,
)


class TestRules(unittest.TestCase):

  def test_card_id_bijection(self):
    deck = create_deck()
    self.assertEqual(len(deck), 52)
    ids = [card_to_id(c) for c in deck]
    self.assertEqual(sorted(ids), list(range(52)))
    for card_id in range(52):
      self.assertEqual(card_to_id(id_to_card(card_id)), card_id)

  def test_winning_hands(self):
    # 1. A-K-Q clubs + three 9s (shuffled order)
    hand_akq_999 = [[1, '♣'], [9, '♥'], [13, '♣'], [9, '♦'], [12, '♣'], [9, '♠']]
    self.assertTrue(check_win(hand_akq_999))
    sets = find_winning_sets(hand_akq_999)
    self.assertIsNotNone(sets)
    self.assertTrue(is_valid_set(sets[0]) and is_valid_set(sets[1]))

    # 2. Ace-low sequence + three Kings
    hand_a23_kkk = [[1, '♥'], [2, '♥'], [3, '♥'], [13, '♥'], [13, '♦'], [13, '♠']]
    self.assertTrue(check_win(hand_a23_kkk))

    # 3. Two suited sequences (Ace-high + mid-rank)
    hand_two_seqs = [[12, '♠'], [13, '♠'], [1, '♠'], [4, '♦'], [5, '♦'], [6, '♦']]
    self.assertTrue(check_win(hand_two_seqs))

    # 4. Two three-of-a-kinds
    hand_two_trips = [[7, '♥'], [7, '♦'], [7, '♣'], [2, '♠'], [2, '♥'], [2, '♣']]
    self.assertTrue(check_win(hand_two_trips))

    # 5. Overlapping rank/sequence split: {4♥,4♦,4♣} + {4♠,5♠,6♠}
    hand_overlap = [[4, '♥'], [4, '♦'], [4, '♠'], [5, '♠'], [6, '♠'], [4, '♣']]
    self.assertTrue(check_win(hand_overlap))

  def test_invalid_hands(self):
    # 6. K-A-2 wrap-around is invalid
    hand_ka2 = [[13, '♣'], [1, '♣'], [2, '♣'], [9, '♥'], [9, '♦'], [9, '♠']]
    self.assertFalse(check_win(hand_ka2))

    # 7. Mixed-suit sequence is invalid
    hand_mixed = [[1, '♣'], [13, '♦'], [12, '♣'], [9, '♥'], [9, '♦'], [9, '♠']]
    self.assertFalse(check_win(hand_mixed))

    # 8. 7-card hand should not win before discarding
    hand_seven = [
        [1, '♣'],
        [13, '♣'],
        [12, '♣'],
        [9, '♥'],
        [9, '♦'],
        [9, '♠'],
        [3, '♦'],
    ]
    self.assertFalse(check_win(hand_seven))

    # Duplicate cards cannot form a valid set
    self.assertFalse(is_valid_set([[9, '♥'], [9, '♥'], [9, '♠']]))


class TestEngineAndTrajectories(unittest.TestCase):

  def _make_two_player_engine(self, log_path=None, seed=42):
    logger = TrajectoryLogger(log_path) if log_path else None
    eng = GameEngine(logger=logger)
    self.assertTrue(eng.join_player('p1', 'Alice', player_type='human')['ok'])
    self.assertTrue(eng.join_player('p2', 'Bob', player_type='bot:tree_1ply')['ok'])
    self.assertTrue(eng.start_match(seed=seed))
    return eng, logger

  def test_forbid_immediate_shown_rediscard(self):
    eng, _ = self._make_two_player_engine(seed=7)
    shown_top = list(eng.shown[-1])

    # Player 0 picks from Shown
    res = eng.draw(0, 'S')
    self.assertTrue(res['ok'])
    self.assertEqual(res['card'], shown_top)

    # Legal discard list must exclude the exact shown card just picked
    obs = eng.get_observation(0)
    self.assertEqual(obs['last_picked_shown'], shown_top)
    self.assertNotIn(shown_top, obs['legal_actions'])
    self.assertEqual(len(obs['legal_actions']), 6)

    # Attempting to discard the exact just-picked shown card fails
    picked_idx = eng.hands[0].index(shown_top)
    bad = eng.discard(0, picked_idx, by_index=True)
    self.assertFalse(bad['ok'])

    # Discarding any other card succeeds
    other_idx = 0 if picked_idx != 0 else 1
    good = eng.discard(0, other_idx, by_index=True)
    self.assertTrue(good['ok'])

  def test_can_discard_just_drawn_hidden_card(self):
    eng, _ = self._make_two_player_engine(seed=7)
    res = eng.draw(0, 'T')
    self.assertTrue(res['ok'])
    drawn_card = res['card']

    obs = eng.get_observation(0)
    self.assertIsNone(obs['last_picked_shown'])
    self.assertIn(drawn_card, obs['legal_actions'])
    self.assertEqual(len(obs['legal_actions']), 7)

    # Discarding the card just drawn from Hidden Deck is legal
    res_d = eng.discard(0, drawn_card, by_index=False)
    self.assertTrue(res_d['ok'])

  def test_reorder_does_not_affect_canonical_observation(self):
    eng, _ = self._make_two_player_engine(seed=11)
    obs_before = eng.get_observation(0)
    self.assertTrue(eng.reorder_card(0, 5))
    obs_after = eng.get_observation(0)
    self.assertEqual(obs_before['hand'], obs_after['hand'])
    self.assertEqual(obs_before['hand_ids'], obs_after['hand_ids'])

  def test_input_validation(self):
    eng, _ = self._make_two_player_engine(seed=13)
    self.assertFalse(eng.draw(1, 'T')['ok'])  # Wrong player
    self.assertFalse(eng.draw(0, 'INVALID')['ok'])  # Invalid source
    self.assertTrue(eng.draw(0, 'T')['ok'])
    self.assertFalse(eng.discard(0, 99, by_index=True)['ok'])  # Out of bounds
    self.assertFalse(eng.discard(0, -1, by_index=True)['ok'])
    self.assertFalse(eng.discard(0, True, by_index=True)['ok'])

  def test_pre_action_logging_and_zero_leakage(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      log_path = os.path.join(tmpdir, 'trajectories.jsonl')
      eng, logger = self._make_two_player_engine(log_path=log_path, seed=100)

      # Set up deterministic hands so player 0 wins on turn 1 after picking Shown
      eng.hands[0] = [
          [1, '♣'],
          [12, '♣'],
          [13, '♣'],
          [9, '♥'],
          [9, '♦'],
          [2, '♠'],
      ]
      eng.shown = [[9, '♠']]
      initial_deck_len = len(eng.deck)

      # Turn 0: Player 0 picks Shown (9♠), then discards 2♠ to win immediately!
      self.assertTrue(eng.draw(0, 'S')['ok'])
      self.assertTrue(eng.discard(0, [2, '♠'], by_index=False)['ok'])

      self.assertEqual(eng.winner, 'Alice')
      self.assertEqual(logger.saved_games, 1)

      with open(log_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]

      self.assertEqual(len(records), 2)
      draw_rec, discard_rec = records

      # Check schema v2 & strict PRE-ACTION state on DRAW ('S')
      self.assertEqual(draw_rec['schema_version'], SCHEMA_VERSION)
      self.assertEqual(draw_rec['phase'], 'DRAW')
      self.assertEqual(draw_rec['action'], 'S')
      self.assertEqual(draw_rec['action_id'], 1)
      # Critical anti-leakage check: shown_card MUST still be in discard_pile!
      self.assertEqual(draw_rec['shown_card'], [9, '♠'])
      self.assertEqual(draw_rec['discard_pile'], [[9, '♠']])
      self.assertEqual(draw_rec['deck_remaining'], initial_deck_len)
      self.assertEqual(draw_rec['hand'], sort_cards(draw_rec['hand']))
      self.assertEqual(draw_rec['public_history'], [])
      self.assertEqual(draw_rec['player_types'], ['human', 'bot:tree_1ply'])
      self.assertTrue(draw_rec['won'])
      self.assertEqual(draw_rec['reward'], 1.0)
      self.assertEqual(draw_rec['steps_remaining'], 2)

      # Check strict PRE-ACTION state on DISCARD
      self.assertEqual(discard_rec['phase'], 'DISCARD')
      self.assertEqual(len(discard_rec['hand']), 7)
      self.assertEqual(discard_rec['hand'], sort_cards(discard_rec['hand']))
      self.assertEqual(discard_rec['action'], [2, '♠'])
      self.assertEqual(discard_rec['action_id'], card_to_id([2, '♠']))
      self.assertIn([2, '♠'], discard_rec['legal_actions'])
      self.assertNotIn([9, '♠'], discard_rec['legal_actions'])
      self.assertEqual(len(discard_rec['public_history']), 1)
      self.assertEqual(discard_rec['public_history'][0]['source'], 'S')
      self.assertEqual(discard_rec['public_history'][0]['card'], [9, '♠'])
      self.assertEqual(discard_rec['steps_remaining'], 1)

      # Verify TrajectoryLogger reloads cached count cleanly on new instance
      reloaded_logger = TrajectoryLogger(log_path)
      self.assertEqual(reloaded_logger.saved_games, 1)

  def test_full_game_hidden_draws_never_leak_in_public_history(self):
    with tempfile.TemporaryDirectory() as tmpdir:
      log_path = os.path.join(tmpdir, 'trajectories.jsonl')
      eng, logger = self._make_two_player_engine(log_path=log_path, seed=99)

      # Play until deck runs out or someone wins, always drawing Hidden ('T')
      while eng.winner is None:
        p = eng.turn
        deck_before = len(eng.deck)
        shown_before = list(eng.shown[-1])
        self.assertTrue(eng.draw(p, 'T')['ok'])
        # Last trajectory entry must reflect deck_before (not deck_before - 1)
        self.assertEqual(eng.trajectory[-1]['deck_remaining'], deck_before)
        self.assertEqual(eng.trajectory[-1]['shown_card'], shown_before)
        self.assertEqual(eng.trajectory[-1]['discard_pile'][-1], shown_before)
        # Hidden draw must not leak card identity in public_history
        self.assertIsNone(eng.public_history[-1]['card'])
        self.assertIsNone(eng.public_history[-1]['card_id'])

        obs = eng.get_observation(p)
        self.assertTrue(eng.discard(p, obs['legal_actions'][0], by_index=False)['ok'])

      self.assertEqual(logger.saved_games, 1)
      with open(log_path, 'r', encoding='utf-8') as f:
        records = [json.loads(line) for line in f if line.strip()]
      self.assertGreater(len(records), 0)
      for rec in records:
        self.assertEqual(rec['schema_version'], SCHEMA_VERSION)
        if eng.winner == 'Draw':
          self.assertEqual(rec['reward'], 0.0)
          self.assertFalse(rec['won'])

  def test_clone_is_isolated(self):
    eng, _ = self._make_two_player_engine(seed=55)
    cloned = eng.clone()
    self.assertTrue(cloned.draw(0, 'T')['ok'])
    self.assertEqual(eng.phase, 'DRAW')
    self.assertEqual(eng.step_count, 0)
    self.assertEqual(cloned.phase, 'DISCARD')
    self.assertEqual(cloned.step_count, 1)

  def test_rematch_preserves_players(self):
    eng, _ = self._make_two_player_engine(seed=1)
    first_id = eng.game_id
    self.assertTrue(eng.rematch(seed=2))
    self.assertNotEqual(eng.game_id, first_id)
    self.assertEqual(len(eng.players), 2)
    self.assertEqual(eng.player_name(0), 'Alice')
    self.assertEqual(eng.player_name(1), 'Bob')


if __name__ == '__main__':
  unittest.main()
