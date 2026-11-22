"""Beginner (Very Easy) computer player for family members' first solo games.

Plays like a very casual beginner just learning Two-Sets-of-3:
  - Still spots an immediate win (takes a winning Shown card or makes a winning
    discard when all 6 winning cards are right there).
  - Almost always draws from the Hidden Deck instead of watching Shown closely.
  - Sometimes protects a complete 3-card set (~60% of the time), but otherwise
    discards casually at random — never tracks pairs, dead discards, or what
    opponents are collecting.
"""

from itertools import combinations
import random

from bots.base import BotPolicy
from bots.heuristic import VALID_SET_MASKS, is_winning_hand_ids
from game.rules import id_to_card


def _cards_in_complete_sets(hand_ids):
  """Returns card IDs that already belong to at least one complete 3-card set."""
  protected = set()
  for a, b, c in combinations(hand_ids, 3):
    if ((1 << a) | (1 << b) | (1 << c)) in VALID_SET_MASKS:
      protected.update((a, b, c))
  return protected


class NoviceBot(BotPolicy):
  """Very easy bot: welcoming first opponent with a high human win rate."""

  name = 'Beginner Bot (Very Easy)'
  player_type = 'bot:novice'

  def __init__(self, seed=None, keep_set_prob=0.60, take_set_prob=0.20):
    self._rng = random.Random(seed)
    self.keep_set_prob = keep_set_prob
    self.take_set_prob = take_set_prob

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]

    # Always grab an immediate winning card from Shown.
    for i in range(6):  # Cannot re-discard shown_id (at index 6)
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        return 'S'

    # Only occasionally notice when Shown completes a first 3-card set.
    if shown_id in _cards_in_complete_sets(hand7):
      return 'S' if self._rng.random() < self.take_set_prob else 'T'
    return 'T'

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids = list(obs['legal_action_ids'])

    # Always take an immediate win if available.
    for i, cid in enumerate(hand7):
      if cid in legal_ids and is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        return id_to_card(cid)

    # Sometimes protect an existing 3-card set; otherwise discard casually at random.
    candidates = legal_ids
    if self._rng.random() < self.keep_set_prob:
      complete = _cards_in_complete_sets(hand7)
      outside_sets = [cid for cid in legal_ids if cid not in complete]
      if outside_sets:
        candidates = outside_sets

    return id_to_card(self._rng.choice(candidates))
