"""Fast, easy-to-read hand evaluator and rules-based (0-ply) computer player.

Key idea:
In a 52-card deck there are only 100 possible valid 3-card sets:
  - 52 three-of-a-kinds (13 ranks x 4 ways to pick 3 suits)
  - 48 suited sequences (4 suits x 12 runs: A-2-3 ... J-Q-K, Q-K-A)

Precomputing those 100 sets lets us score any 6-card hand in microseconds by:
  1. Splitting the 6 cards into two groups of 3 (only 10 ways to do this!).
  2. Checking whether each group is already a complete set (0 cards away),
     1 card away (and counting how many *live unseen* cards complete it),
     or 2 cards away.
  3. Tracking which cards the next player picked from Shown so we avoid
     discarding cards that help them complete a set.
"""

from itertools import combinations

from bots.base import BotPolicy
from game.rules import card_label, card_to_id, id_to_card

ALL_CARDS_MASK = (1 << 52) - 1

# Precompute all 10 unique ways to split 6 indices into two 3-card groups.
SIX_CARD_PARTITIONS = tuple(
    (g1, tuple(i for i in range(6) if i not in g1))
    for g1 in combinations(range(6), 3)
    if 0 in g1
)


def _build_lookup_tables():
  """Builds bitmask tables for all 100 valid 3-card sets and pair completers."""
  valid_set_masks = set()
  pair_completer_masks = {}  # (a, b) with a < b -> bitmask of cards c forming a valid set
  single_partner_masks = [0] * 52  # card a -> bitmask of cards that share a valid set with a

  all_sets = []

  # 1. Three-of-a-kind: 13 ranks x 4 suit combinations = 52 sets
  for rank_idx in range(13):
    same_rank_ids = [suit_idx * 13 + rank_idx for suit_idx in range(4)]
    for triple in combinations(same_rank_ids, 3):
      all_sets.append(triple)

  # 2. Suited sequences: 4 suits x 12 runs (A-2-3 .. J-Q-K and Q-K-A) = 48 sets
  for suit_idx in range(4):
    base = suit_idx * 13
    for start in range(11):  # A-2-3 (0,1,2) through J-Q-K (10,11,12)
      all_sets.append((base + start, base + start + 1, base + start + 2))
    # Ace-high sequence: Q (11), K (12), A (0)
    all_sets.append((base + 0, base + 11, base + 12))

  for a, b, c in all_sets:
    valid_set_masks.add((1 << a) | (1 << b) | (1 << c))
    for x, y, z in ((a, b, c), (a, c, b), (b, c, a)):
      pair = (x, y) if x < y else (y, x)
      pair_completer_masks[pair] = pair_completer_masks.get(pair, 0) | (1 << z)
      single_partner_masks[x] |= 1 << y

  return valid_set_masks, pair_completer_masks, tuple(single_partner_masks)


VALID_SET_MASKS, PAIR_COMPLETER_MASKS, SINGLE_PARTNER_MASKS = (
    _build_lookup_tables()
)

# Winning score is strictly above any non-winning hand (~0..160) while staying
# well-calibrated for expected-value averaging in tree search.
WIN_SCORE = 350.0


def pair_completers(a, b):
  """Returns bitmask of cards that complete a valid 3-card set with (a, b)."""
  return PAIR_COMPLETER_MASKS.get((a, b) if a < b else (b, a), 0)


def is_winning_hand_ids(hand_ids):
  """Fast check: returns True if 6 card IDs partition into two valid sets."""
  if len(hand_ids) != 6:
    return False
  for g1, g2 in SIX_CARD_PARTITIONS:
    m1 = (1 << hand_ids[g1[0]]) | (1 << hand_ids[g1[1]]) | (1 << hand_ids[g1[2]])
    if m1 not in VALID_SET_MASKS:
      continue
    m2 = (1 << hand_ids[g2[0]]) | (1 << hand_ids[g2[1]]) | (1 << hand_ids[g2[2]])
    if m2 in VALID_SET_MASKS:
      return True
  return False


def get_unseen_mask(obs):
  """Returns a 52-bit mask of cards still unseen from this player's perspective.

  Strictly fair: only excludes our own hand, dead cards buried in the discard
  pile, and cards we publicly saw opponents pick from Shown (and not yet drop).
  """
  seen_mask = 0
  for cid in obs['hand_ids']:
    seen_mask |= 1 << cid
  for cid in obs['discard_pile_ids']:
    seen_mask |= 1 << cid

  # Track cards opponents publicly picked from Shown and haven't discarded since.
  opponent_known = {}
  me = obs['player']
  for event in obs.get('public_history', ()):
    p = event['player']
    if p == me:
      continue
    if event['phase'] == 'DRAW' and event.get('source') == 'S':
      cid = event.get('card_id')
      if cid is not None:
        opponent_known.setdefault(p, set()).add(cid)
    elif event['phase'] == 'DISCARD':
      cid = event.get('card_id')
      if cid is not None and p in opponent_known:
        opponent_known[p].discard(cid)

  for cards in opponent_known.values():
    for cid in cards:
      seen_mask |= 1 << cid

  return ALL_CARDS_MASK ^ seen_mask


def mask_to_ids(mask):
  """Converts a 52-bit card bitmask to a list of card IDs."""
  ids = []
  while mask:
    lsb = mask & -mask
    ids.append(lsb.bit_length() - 1)
    mask ^= lsb
  return ids


def _score_triple(a, b, c, unseen_mask):
  """Scores a 3-card group (higher is closer to a complete set)."""
  triple_mask = (1 << a) | (1 << b) | (1 << c)
  if triple_mask in VALID_SET_MASKS:
    return 100.0

  # Check all 3 pairs inside this group to see how many live unseen cards finish a set.
  outs_mask = (
      pair_completers(a, b) | pair_completers(a, c) | pair_completers(b, c)
  ) & unseen_mask
  live_outs = outs_mask.bit_count()
  if live_outs > 0:
    # 1 card away from a valid set: reward more live outs (e.g. open-ended run > gutshot).
    return 20.0 + 6.0 * live_outs

  # If two cards form a proto-set whose completing cards are ALL already dead,
  # give only a tiny score so the bot prefers replacing dead pairs.
  if pair_completers(a, b) or pair_completers(a, c) or pair_completers(b, c):
    return 2.0

  # 2 cards away: small tie-breaker favoring cards with more live neighbors/same-rank cards.
  live_flex = (
      (SINGLE_PARTNER_MASKS[a] & unseen_mask).bit_count()
      + (SINGLE_PARTNER_MASKS[b] & unseen_mask).bit_count()
      + (SINGLE_PARTNER_MASKS[c] & unseen_mask).bit_count()
  )
  return 0.25 * live_flex


def score_hand(hand_ids, unseen_mask):
  """Evaluates a 6-card hand by checking all 10 ways to split it into two sets of 3."""
  best = -1.0
  for g1, g2 in SIX_CARD_PARTITIONS:
    s1 = _score_triple(
        hand_ids[g1[0]], hand_ids[g1[1]], hand_ids[g1[2]], unseen_mask
    )
    s2 = _score_triple(
        hand_ids[g2[0]], hand_ids[g2[1]], hand_ids[g2[2]], unseen_mask
    )
    if s1 >= 100.0 and s2 >= 100.0:
      return WIN_SCORE
    total = s1 + s2
    if total > best:
      best = total
  return best


def get_next_player_hints(obs):
  """Extracts cards the next player publicly wanted (picked from Shown) or rejected."""
  next_player = (obs['player'] + 1) % obs['num_players']
  wanted = set()
  discarded = set()
  for event in obs.get('public_history', ()):
    if event['player'] != next_player:
      continue
    cid = event.get('card_id')
    if cid is None:
      continue
    if event['phase'] == 'DRAW' and event.get('source') == 'S':
      wanted.add(cid)
    elif event['phase'] == 'DISCARD':
      wanted.discard(cid)
      discarded.add(cid)
  return wanted, discarded


def opponent_danger(card_id, next_wanted, next_discarded):
  """Estimates how risky it is to discard card_id to the next player."""
  danger = 0.0
  for wanted_id in next_wanted:
    # If card_id forms a pair or suited neighbor with a card the next player picked,
    # discarding it could directly feed them a winning set!
    if pair_completers(card_id, wanted_id):
      danger += 8.0
  for disc_id in next_discarded:
    # Slightly safer to discard cards adjacent to ones they already threw away.
    if pair_completers(card_id, disc_id):
      danger -= 0.5
  return danger


def best_discard_score(hand7_ids, unseen_mask, forbidden_id=None):
  """Fast helper: best heuristic score achievable after discarding 1 of 7 cards."""
  best = -1e9
  for i in range(7):
    if hand7_ids[i] == forbidden_id:
      continue
    hand6 = hand7_ids[:i] + hand7_ids[i + 1 :]
    sc = score_hand(hand6, unseen_mask)
    if sc > best:
      best = sc
  return best


class HeuristicBot(BotPolicy):
  """Rules-based (0-ply) bot: scores immediate hand strength + opponent danger."""

  name = 'Heuristic Bot (Easy)'
  player_type = 'bot:heuristic'

  def __init__(self, danger_weight=1.0, shown_info_cost=2.0):
    self.danger_weight = danger_weight
    self.shown_info_cost = shown_info_cost

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    unseen_mask = get_unseen_mask(obs)

    # Compare our current hand score against our best hand if we take Shown.
    current_score = score_hand(hand6, unseen_mask)
    if_take_shown = best_discard_score(
        hand6 + [shown_id], unseen_mask, forbidden_id=shown_id
    )

    if if_take_shown >= WIN_SCORE:
      self.last_thought = 'Took Shown — completes a winning hand!'
      return 'S'

    # Take Shown if it meaningfully improves our hand (beyond the info-reveal cost).
    if if_take_shown - self.shown_info_cost > current_score + 4.0:
      self.last_thought = (
          f'Took Shown (hand score {if_take_shown:.1f} vs current {current_score:.1f})'
      )
      return 'S'
    self.last_thought = (
        f'Drew Hidden (Shown would score {if_take_shown:.1f} vs current {current_score:.1f})'
    )
    return 'T'

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids = set(obs['legal_action_ids'])
    unseen_mask = get_unseen_mask(obs)
    next_wanted, next_discarded = get_next_player_hints(obs)

    scored = []
    for i, cid in enumerate(hand7):
      if cid not in legal_ids:
        continue
      hand6 = hand7[:i] + hand7[i + 1 :]
      hand_score = score_hand(hand6, unseen_mask)
      if hand_score >= WIN_SCORE:
        card = id_to_card(cid)
        self.last_thought = f'Put down {card_label(card)} — winning hand!'
        return card
      value = hand_score - self.danger_weight * opponent_danger(
          cid, next_wanted, next_discarded
      )
      scored.append((value, cid))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_val, best_cid = scored[0]
    card = id_to_card(best_cid)
    if len(scored) >= 2:
      alt_val, alt_cid = scored[1]
      self.last_thought = (
          f'Put down {card_label(card)} (score {best_val:.1f} vs '
          f'{card_label(id_to_card(alt_cid))} {alt_val:.1f})'
      )
    return card
