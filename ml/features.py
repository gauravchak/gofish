"""Shared, suit-invariant feature extractors for all three RL algorithms.

Every feature is computed strictly from information legally visible to the
acting player (hand, discard pile, public_history, deck_remaining).

Why suit-invariant features?
  - Suits (♥, ♦, ♣, ♠) are completely symmetric in Two-Sets-of-3.
  - Compressing a 6-card hand + table context into 16 clean numbers lets all
    three RL models learn strong strategy within a few thousand games (instead
    of needing hundreds of thousands of games on raw 52-bit vectors).
"""

from bots.heuristic import (
    SINGLE_PARTNER_MASKS,
    SIX_CARD_PARTITIONS,
    VALID_SET_MASKS,
    get_next_player_hints,
    get_unseen_mask,
    pair_completers,
)

AFTERSTATE_FEATURE_NAMES = (
    'bias',
    'is_win',
    'complete_sets',
    'has_one_complete_set',
    'one_away_groups',
    'total_live_outs',
    'min_group_live_outs',
    'max_group_live_outs',
    'has_super_wait',
    'dead_proto_groups',
    'single_flexibility',
    'deck_fraction',
    'discard_helps_next',
    'discard_matches_next_discard',
    'opp_shown_picks',
    'took_shown',
)

ACTION_FEATURE_NAMES = AFTERSTATE_FEATURE_NAMES + (
    'is_draw_phase',
    'action_is_take_shown',
    'delta_complete_sets',
    'delta_live_outs',
)

AFTERSTATE_DIM = len(AFTERSTATE_FEATURE_NAMES)
ACTION_DIM = len(ACTION_FEATURE_NAMES)


def _group_stats(a, b, c, unseen_mask):
  """Returns (is_complete, is_one_away, live_outs, is_dead_proto, single_flex)."""
  triple_mask = (1 << a) | (1 << b) | (1 << c)
  if triple_mask in VALID_SET_MASKS:
    return 1, 0, 0, 0, 0.0

  raw_outs = (
      pair_completers(a, b) | pair_completers(a, c) | pair_completers(b, c)
  )
  live_outs = (raw_outs & unseen_mask).bit_count()
  if live_outs > 0:
    return 0, 1, live_outs, 0, 0.0

  if raw_outs:
    return 0, 0, 0, 1, 0.0

  flex = (
      (SINGLE_PARTNER_MASKS[a] & unseen_mask).bit_count()
      + (SINGLE_PARTNER_MASKS[b] & unseen_mask).bit_count()
      + (SINGLE_PARTNER_MASKS[c] & unseen_mask).bit_count()
  )
  return 0, 0, 0, 0, float(flex)


def analyze_hand6(hand6_ids, unseen_mask):
  """Finds the strongest 2x3 partition of a 6-card hand and returns summary stats."""
  best_rank = (-1, -1, -1, -1.0)
  best_stats = None

  for g1, g2 in SIX_CARD_PARTITIONS:
    c1, o1, l1, d1, f1 = _group_stats(
        hand6_ids[g1[0]], hand6_ids[g1[1]], hand6_ids[g1[2]], unseen_mask
    )
    c2, o2, l2, d2, f2 = _group_stats(
        hand6_ids[g2[0]], hand6_ids[g2[1]], hand6_ids[g2[2]], unseen_mask
    )
    complete = c1 + c2
    one_away = o1 + o2
    total_live = l1 + l2
    flex = f1 + f2
    rank = (complete, one_away, total_live, flex - 2.0 * (d1 + d2))
    if rank > best_rank:
      best_rank = rank
      best_stats = (
          complete,
          one_away,
          total_live,
          min(l1, l2) if (c1 == 0 and c2 == 0) else max(l1, l2),
          max(l1, l2),
          d1 + d2,
          flex,
      )

  return best_stats


def extract_afterstate_features(
    hand6_ids,
    unseen_mask,
    deck_remaining,
    next_wanted,
    next_discarded,
    discarded_id=None,
    took_shown=False,
):
  """Builds the 16-element normalized feature vector for a 6-card afterstate."""
  complete, one_away, total_live, min_live, max_live, dead_proto, flex = (
      analyze_hand6(hand6_ids, unseen_mask)
  )

  helps_next = 0
  matches_next_disc = 0
  if discarded_id is not None:
    for wid in next_wanted:
      if pair_completers(discarded_id, wid):
        helps_next += 1
    for did in next_discarded:
      if pair_completers(discarded_id, did):
        matches_next_disc += 1

  return [
      1.0,                                     # bias
      1.0 if complete == 2 else 0.0,           # is_win
      complete / 2.0,                          # complete_sets
      1.0 if complete >= 1 else 0.0,           # has_one_complete_set
      one_away / 2.0,                          # one_away_groups
      min(total_live, 10) / 10.0,              # total_live_outs
      min(min_live, 4) / 4.0,                  # min_group_live_outs
      min(max_live, 4) / 4.0,                  # max_group_live_outs
      1.0 if max_live >= 3 else 0.0,           # has_super_wait
      dead_proto / 2.0,                        # dead_proto_groups
      min(flex, 30.0) / 30.0,                  # single_flexibility
      min(deck_remaining, 40) / 40.0,          # deck_fraction
      min(helps_next, 2) / 2.0,                # discard_helps_next
      min(matches_next_disc, 2) / 2.0,         # discard_matches_next_discard
      min(len(next_wanted), 3) / 3.0,          # opp_shown_picks
      1.0 if took_shown else 0.0,              # took_shown
  ]


def extract_state_features(obs):
  """Builds the 16-element feature vector for the player's current observation."""
  unseen_mask = get_unseen_mask(obs)
  next_wanted, next_discarded = get_next_player_hints(obs)
  hand_ids = obs['hand_ids']

  if len(hand_ids) == 6:
    return extract_afterstate_features(
        hand_ids,
        unseen_mask,
        obs['deck_remaining'],
        next_wanted,
        next_discarded,
        discarded_id=None,
        took_shown=False,
    )

  # During DISCARD (7 cards in hand), summarize the best legal 6-card hand.
  legal_ids = set(obs['legal_action_ids'])
  best_vec = None
  best_key = (-1.0, -1.0, -1.0)
  took_shown = obs.get('last_picked_shown_id') is not None
  for i, cid in enumerate(hand_ids):
    if cid not in legal_ids:
      continue
    h6 = hand_ids[:i] + hand_ids[i + 1 :]
    vec = extract_afterstate_features(
        h6,
        unseen_mask,
        obs['deck_remaining'],
        next_wanted,
        next_discarded,
        discarded_id=cid,
        took_shown=took_shown,
    )
    key = (vec[2], vec[4], vec[5])
    if key > best_key:
      best_key = key
      best_vec = vec
  return best_vec


def extract_action_features(
    obs, action, unseen_mask=None, next_wanted=None, next_discarded=None
):
  """Builds the 20-element (state, action) feature vector for Q-Learning & Policy Gradient.

  Works cleanly for BOTH phases:
    - DRAW: action is 'S' or 'T'
    - DISCARD: action is card_id (int)
  """
  if unseen_mask is None:
    unseen_mask = get_unseen_mask(obs)
  if next_wanted is None or next_discarded is None:
    next_wanted, next_discarded = get_next_player_hints(obs)
  deck_rem = obs['deck_remaining']

  if obs['phase'] == 'DRAW':
    hand6 = obs['hand_ids']
    base_vec = extract_afterstate_features(
        hand6, unseen_mask, deck_rem, next_wanted, next_discarded
    )
    if action == 'T':
      return base_vec + [1.0, 0.0, 0.0, 0.0]

    # Action == 'S': evaluate best legal 6-card hand we'd get by taking Shown.
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]
    best_vec = None
    best_key = (-1.0, -1.0, -1.0)
    for i in range(6):  # Index 6 is shown_id (forbidden to re-discard this turn)
      h6 = hand7[:i] + hand7[i + 1 :]
      vec = extract_afterstate_features(
          h6,
          unseen_mask,
          deck_rem,
          next_wanted,
          next_discarded,
          discarded_id=hand7[i],
          took_shown=True,
      )
      key = (vec[1], vec[2], vec[4] + vec[5])
      if key > best_key:
        best_key = key
        best_vec = vec

    delta_sets = best_vec[2] - base_vec[2]
    delta_outs = best_vec[5] - base_vec[5]
    return best_vec + [1.0, 1.0, delta_sets, delta_outs]

  # DISCARD phase: action is a card_id (int)
  cid = action if isinstance(action, int) else None
  hand7 = obs['hand_ids']
  idx = hand7.index(cid)
  h6 = hand7[:idx] + hand7[idx + 1 :]
  took_shown = obs.get('last_picked_shown_id') is not None
  after_vec = extract_afterstate_features(
      h6,
      unseen_mask,
      deck_rem,
      next_wanted,
      next_discarded,
      discarded_id=cid,
      took_shown=took_shown,
  )
  return after_vec + [0.0, 1.0 if took_shown else 0.0, 0.0, 0.0]


def extract_all_action_features(obs):
  """Computes (actions, feature_vectors) once per step without re-scanning public_history."""
  unseen_mask = get_unseen_mask(obs)
  next_wanted, next_discarded = get_next_player_hints(obs)
  actions = (
      list(obs['legal_actions'])
      if obs['phase'] == 'DRAW'
      else list(obs['legal_action_ids'])
  )
  feats = [
      extract_action_features(
          obs,
          act,
          unseen_mask=unseen_mask,
          next_wanted=next_wanted,
          next_discarded=next_discarded,
      )
      for act in actions
  ]
  return actions, feats
