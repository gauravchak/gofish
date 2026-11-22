"""Pure rules and card utilities for Two-Sets-of-3 (6-card Family Rummy)."""

from itertools import combinations

SUITS = ('♥', '♦', '♣', '♠')
SUIT_TO_IDX = {s: i for i, s in enumerate(SUITS)}
RANKS = tuple(range(1, 14))  # 1=Ace, 11=J, 12=Q, 13=K
RANK_NAMES = {1: 'A', 11: 'J', 12: 'Q', 13: 'K'}


def card_to_id(card):
  """Maps a [rank, suit] card to a canonical integer ID in 0..51."""
  rank, suit = card[0], card[1]
  return SUIT_TO_IDX[suit] * 13 + (rank - 1)


def id_to_card(card_id):
  """Maps an integer card ID in 0..51 back to [rank, suit]."""
  if not 0 <= card_id < 52:
    raise ValueError(f'Invalid card_id: {card_id}')
  suit = SUITS[card_id // 13]
  rank = (card_id % 13) + 1
  return [rank, suit]


def card_label(card):
  """Formats a [rank, suit] card as a short display string (e.g., 'A♠')."""
  return f'{RANK_NAMES.get(card[0], card[0])}{card[1]}'


def sort_cards(cards):
  """Returns a new canonically sorted list of [rank, suit] cards."""
  return [list(c) for c in sorted(cards, key=card_to_id)]


def create_deck():
  """Returns a standard unshuffled 52-card deck as [rank, suit] pairs."""
  return [[r, s] for s in SUITS for r in RANKS]


def is_valid_set(three_cards):
  """Returns True if 3 distinct cards form either:

  1. Three of a kind (same rank, e.g., 9♥, 9♦, 9♠)
  2. A suited sequence of 3 (e.g., A♣-2♣-3♣, 4♥-5♥-6♥, or Q♣-K♣-A♣)
  """
  if len(three_cards) != 3:
    return False

  if len({card_to_id(c) for c in three_cards}) != 3:
    return False

  ranks = sorted(c[0] for c in three_cards)
  suits = {c[1] for c in three_cards}

  # Case 1: Three of a kind
  if ranks[0] == ranks[1] == ranks[2]:
    return True

  # Case 2: Suited sequence of 3
  if len(suits) == 1:
    if ranks[0] + 1 == ranks[1] and ranks[1] + 1 == ranks[2]:
      return True
    # Ace-high sequence: Q (12), K (13), A (1)
    if ranks == [1, 12, 13]:
      return True

  return False


def find_winning_sets(hand):
  """Returns (set1, set2) if a 6-card hand splits into two valid sets of 3, else None."""
  if len(hand) != 6:
    return None

  all_indices = set(range(6))
  for idx_set1 in combinations(range(6), 3):
    # Break symmetry: element 0 always belongs to set1 so each partition is checked once
    if 0 not in idx_set1:
      continue
    idx_set2 = sorted(all_indices - set(idx_set1))
    set1 = [list(hand[i]) for i in idx_set1]
    set2 = [list(hand[i]) for i in idx_set2]
    if is_valid_set(set1) and is_valid_set(set2):
      return sort_cards(set1), sort_cards(set2)

  return None


def check_win(hand):
  """Returns True if a 6-card hand can be partitioned into two disjoint valid sets of 3."""
  return find_winning_sets(hand) is not None
