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
    'cross_partition_win_outs',
    'is_ready_hand',
    'hand_connectedness',
    # Opponent Memory & Table Defense features (break through the 19-feature ceiling!)
    'opp_danger_exact',
    'opp_shown_pair_synergy',
    'dead_cards_fraction',
    'safe_discard_score',
    'second_group_live_outs',
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
  """Finds the strongest 2x3 partition AND cross-partition winning outs of a 6-card hand."""
  best_rank = (-1, -1, -1, -1.0)
  best_stats = None
  # Distinct unseen cards across ALL 10 partitions that would immediately win!
  win_outs_mask = 0

  for g1, g2 in SIX_CARD_PARTITIONS:
    a1, b1, c1_id = hand6_ids[g1[0]], hand6_ids[g1[1]], hand6_ids[g1[2]]
    a2, b2, c2_id = hand6_ids[g2[0]], hand6_ids[g2[1]], hand6_ids[g2[2]]
    c1, o1, l1, d1, f1 = _group_stats(a1, b1, c1_id, unseen_mask)
    c2, o2, l2, d2, f2 = _group_stats(a2, b2, c2_id, unseen_mask)

    # If one group is already complete, any live out for the other group is a WINNING out!
    if c1 == 1 and c2 == 0 and o2 == 1:
      win_outs_mask |= (
          pair_completers(a2, b2)
          | pair_completers(a2, c2_id)
          | pair_completers(b2, c2_id)
      ) & unseen_mask
    elif c2 == 1 and c1 == 0 and o1 == 1:
      win_outs_mask |= (
          pair_completers(a1, b1)
          | pair_completers(a1, c1_id)
          | pair_completers(b1, c1_id)
      ) & unseen_mask

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

  # Count how many pairs of cards in the 6-card hand have a live completing card.
  live_pairs = 0
  for i in range(6):
    ci = hand6_ids[i]
    for j in range(i + 1, 6):
      if pair_completers(ci, hand6_ids[j]) & unseen_mask:
        live_pairs += 1

  return best_stats + (win_outs_mask.bit_count(), live_pairs)


def extract_afterstate_features(
    hand6_ids,
    unseen_mask,
    deck_remaining,
    next_wanted,
    next_discarded,
    discarded_id=None,
    took_shown=False,
):
  """Builds the 24-element normalized feature vector for a 6-card afterstate."""
  (
      complete,
      one_away,
      total_live,
      min_live,
      max_live,
      dead_proto,
      flex,
      win_outs,
      live_pairs,
  ) = analyze_hand6(hand6_ids, unseen_mask)

  helps_next = 0
  matches_next_disc = 0
  opp_danger_exact = 0.0
  opp_shown_synergy = 0
  wanted_list = list(next_wanted)

  # Measure how synergistic the next opponent's publicly picked Shown cards are!
  nw_len = len(wanted_list)
  for i in range(nw_len):
    for j in range(i + 1, nw_len):
      completers = pair_completers(wanted_list[i], wanted_list[j])
      if completers:
        opp_shown_synergy += 1
        if discarded_id is not None and (completers & (1 << discarded_id)):
          # CRITICAL DEFENSE: this discard completes a known 2-card proto-set for opponent!
          opp_danger_exact = 1.0

  if discarded_id is not None:
    for wid in wanted_list:
      if pair_completers(discarded_id, wid):
        helps_next += 1
    for did in next_discarded:
      if pair_completers(discarded_id, did):
        matches_next_disc += 1

  # Safe discard: doesn't connect with any opponent Shown pickup, bonus if matching
  # cards the opponent already threw away.
  if discarded_id is None:
    safe_discard = 0.5
  elif opp_danger_exact > 0.0:
    safe_discard = 0.0
  elif helps_next == 0 and matches_next_disc > 0:
    safe_discard = 1.0
  elif helps_next == 0:
    safe_discard = 0.7
  else:
    safe_discard = 0.2

  # Fraction of the 52-card deck already permanently seen/dead (excluding our 6 cards).
  unseen_count = unseen_mask.bit_count()
  dead_cards_frac = max(0.0, min(1.0, (46 - unseen_count) / 40.0))

  # Live outs specifically for our weaker (second) group when we have 0 complete sets,
  # or for our remaining incomplete group when we have 1 complete set.
  second_group_outs = min_live if complete == 0 else max_live

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
      min(nw_len, 3) / 3.0,                    # opp_shown_picks
      1.0 if took_shown else 0.0,              # took_shown
      min(win_outs, 8) / 8.0,                  # cross_partition_win_outs
      1.0 if win_outs >= 1 else 0.0,           # is_ready_hand (1 draw from winning!)
      min(live_pairs, 6) / 6.0,                # hand_connectedness
      opp_danger_exact,                        # opp_danger_exact
      min(opp_shown_synergy, 3) / 3.0,         # opp_shown_pair_synergy
      dead_cards_frac,                         # dead_cards_fraction
      safe_discard,                            # safe_discard_score
      min(second_group_outs, 4) / 4.0,         # second_group_live_outs
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
    best_key = (-1.0,) * 6
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
      key = (vec[1], vec[2], vec[17], vec[16], vec[4] + vec[5], vec[22])
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
