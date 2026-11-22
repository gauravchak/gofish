"""Beginner (Very Easy) computer player for family members' first solo games.

Plays like a casual beginner learning Two-Sets-of-3:
  - Never misses an obvious immediate win (takes a winning Shown card and makes
    a winning discard if available).
  - Keeps any complete 3-card set it already holds.
  - Otherwise only partially spots pairs/neighbors and makes occasional casual
    mistakes — never counts dead discard cards or tracks what opponents picked.
"""

from itertools import combinations
import random

from bots.base import BotPolicy
from bots.heuristic import VALID_SET_MASKS, is_winning_hand_ids, pair_completers
from game.rules import id_to_card


def _cards_in_complete_sets(hand_ids):
  """Returns card IDs that already belong to at least one complete 3-card set."""
  protected = set()
  for a, b, c in combinations(hand_ids, 3):
    if ((1 << a) | (1 << b) | (1 << c)) in VALID_SET_MASKS:
      protected.update((a, b, c))
  return protected


def _cards_with_partner(hand_ids):
  """Returns card IDs that form a pair or suited neighbor with another card in hand."""
  partnered = set()
  for a, b in combinations(hand_ids, 2):
    if pair_completers(a, b):
      partnered.update((a, b))
  return partnered


class NoviceBot(BotPolicy):
  """Very easy bot: fun first opponent that humans have an above-average chance to beat."""

  name = 'Beginner Bot (Very Easy)'
  player_type = 'bot:novice'

  def __init__(self, seed=None, smart_prob=0.45):
    self._rng = random.Random(seed)
    self.smart_prob = smart_prob

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]

    # Always grab an immediate winning card from Shown!
    for i in range(6):  # Cannot re-discard shown_id (at index 6)
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        return 'S'

    # Otherwise only occasionally take Shown when it completes a set or makes a pair.
    if shown_id in _cards_in_complete_sets(hand7):
      return 'S' if self._rng.random() < 0.7 else 'T'
    if shown_id in _cards_with_partner(hand7):
      return 'S' if self._rng.random() < 0.25 else 'T'
    return 'T'

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids = list(obs['legal_action_ids'])

    # Always take an immediate win if available.
    for i, cid in enumerate(hand7):
      if cid in legal_ids and is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        return id_to_card(cid)

    # Protect complete sets, and with `smart_prob` also protect pairs/neighbors.
    complete = _cards_in_complete_sets(hand7)
    candidates = [cid for cid in legal_ids if cid not in complete]
    if not candidates:
      candidates = legal_ids

    if self._rng.random() < self.smart_prob:
      partnered = _cards_with_partner(hand7)
      loose = [cid for cid in candidates if cid not in partnered]
      if loose:
        candidates = loose

    return id_to_card(self._rng.choice(candidates))
