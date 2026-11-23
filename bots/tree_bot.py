"""1-Level and 2-Level Expectimax Tree Computer Player.

How the tree thinks ahead:
  - 1-Level Tree (depth=1):
      * On DRAW: compares the exact result of taking the Shown card ('S')
        against the *average* best result across every unseen card we could
        draw from the Hidden Deck ('T').
      * On DISCARD: for each card we might put down, looks 1 draw ahead and
        averages our best resulting hand across all unseen next draws, while
        avoiding cards that help the next player.
  - 2-Level Tree (depth=2):
      * Keeps the top candidate moves from Level 1 and looks a *second* turn
        ahead (Draw -> Discard -> Next Draw -> Next Discard), discounting
        future turns slightly (gamma=0.85) so winning sooner is always better.
"""

from bots.base import BotPolicy
from bots.heuristic import (
    SINGLE_PARTNER_MASKS,
    WIN_SCORE,
    best_discard_score,
    get_next_player_hints,
    get_unseen_mask,
    mask_to_ids,
    opponent_danger,
    score_hand,
)
from game.rules import card_label, id_to_card


def _evenly_spaced_sample(items, max_count):
  """Deterministic subset of up to max_count items (keeps tree search reproducible)."""
  n = len(items)
  if n <= max_count:
    return items
  step = n / max_count
  return [items[int(i * step)] for i in range(max_count)]


def _expected_one_turn_value(hand6, unseen_ids, unseen_mask):
  """Level-1 chance node: exact average best hand score after 1 random unseen draw.

  Fast exact shortcut: any unseen card that shares no valid 3-card set with any
  card currently in hand6 is a 'brick' — the best move is to discard it right
  back, keeping our current hand score without re-scoring all 7 discards.
  """
  if not unseen_ids:
    return score_hand(hand6, unseen_mask)

  base_score = score_hand(hand6, unseen_mask)
  relevant_mask = 0
  for cid in hand6:
    relevant_mask |= SINGLE_PARTNER_MASKS[cid]

  total = 0.0
  for draw_id in unseen_ids:
    if not (relevant_mask & (1 << draw_id)):
      total += base_score
      continue
    next_unseen = unseen_mask ^ (1 << draw_id)
    val = best_discard_score(hand6 + [draw_id], next_unseen)
    total += val if val > base_score else base_score
  return total / len(unseen_ids)


def _best_discard_one_ply(hand7, unseen_ids, unseen_mask, forbidden_id=None):
  """Best 1-ply expected value after choosing 1 legal discard from 7 cards."""
  # Pre-rank the 7 discards with the fast heuristic and evaluate the top 3.
  ranked = []
  for i in range(7):
    cid = hand7[i]
    if cid == forbidden_id:
      continue
    hand6 = hand7[:i] + hand7[i + 1 :]
    imm = score_hand(hand6, unseen_mask)
    if imm >= WIN_SCORE:
      return WIN_SCORE
    ranked.append((imm, hand6))

  ranked.sort(key=lambda item: item[0], reverse=True)
  best_val = -1e9
  for _, hand6 in ranked[:3]:
    val = _expected_one_turn_value(hand6, unseen_ids, unseen_mask)
    if val > best_val:
      best_val = val
  return best_val


def _expected_two_turn_value(
    hand6, unseen_ids, unseen_mask, gamma=0.85, branch_sample=8
):
  """Level-2 chance node: looks 2 turns ahead (Draw1 -> Discard1 -> Draw2 -> Discard2).

  Checks ALL unseen cards for an immediate 1-turn win, and evaluates 2-turn
  continuation value across relevant unseen draws for speed.
  """
  if not unseen_ids:
    return score_hand(hand6, unseen_mask)

  # Exact 1-turn lookahead over every unseen card (never misses a 1-card win!)
  one_turn_val = _expected_one_turn_value(hand6, unseen_ids, unseen_mask)

  # 2nd-level lookahead: sample unseen draws for turn 1, pick the best discard,
  # and evaluate the exact 1-turn lookahead from that new hand.
  sampled_draws = _evenly_spaced_sample(unseen_ids, branch_sample)
  second_turn_total = 0.0
  for draw1_id in sampled_draws:
    next_unseen_mask = unseen_mask ^ (1 << draw1_id)
    hand7 = hand6 + [draw1_id]

    # Pick the best resulting 6-card hand after draw1.
    best_imm = -1e9
    best_h6 = None
    for i in range(7):
      h6_next = hand7[:i] + hand7[i + 1 :]
      sc = score_hand(h6_next, next_unseen_mask)
      if sc > best_imm:
        best_imm = sc
        best_h6 = h6_next

    if best_imm >= WIN_SCORE:
      second_turn_total += WIN_SCORE
      continue

    next_unseen_ids = [cid for cid in unseen_ids if cid != draw1_id]
    second_turn_total += _expected_one_turn_value(
        best_h6, next_unseen_ids, next_unseen_mask
    )

  two_turn_val = second_turn_total / len(sampled_draws)
  return one_turn_val + gamma * two_turn_val


class TreeBot(BotPolicy):
  """Expectimax lookahead computer player supporting depth=1 (1-ply) or depth=2 (2-ply)."""

  def __init__(
      self,
      depth=1,
      danger_weight=1.5,
      shown_info_cost=2.5,
      gamma=0.85,
  ):
    if depth not in (1, 2):
      raise ValueError('TreeBot supports depth=1 or depth=2.')
    self.depth = depth
    self.danger_weight = danger_weight
    self.shown_info_cost = shown_info_cost
    self.gamma = gamma
    if depth == 1:
      self.name = 'Tree Bot (1-Level)'
      self.player_type = 'bot:tree_1ply'
    else:
      self.name = 'Tree Bot (2-Level)'
      self.player_type = 'bot:tree_2ply'

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    unseen_mask = get_unseen_mask(obs)
    unseen_ids = mask_to_ids(unseen_mask)

    # Branch 1: Take the face-up Shown card (deterministic, cannot re-discard it).
    hand_with_shown = hand6 + [shown_id]
    imm_shown = best_discard_score(
        hand_with_shown, unseen_mask, forbidden_id=shown_id
    )
    if imm_shown >= WIN_SCORE:
      self.last_thought = 'Took Shown — completes a winning hand!'
      return 'S'

    if self.depth == 1:
      # Compare best immediate hand from Shown vs expected immediate hand from Hidden.
      val_shown = imm_shown
      val_hidden = _expected_one_turn_value(hand6, unseen_ids, unseen_mask)
    else:
      # 2-level comparison: evaluate the next turn after taking Shown vs Hidden.
      val_shown = _best_discard_one_ply(
          hand_with_shown, unseen_ids, unseen_mask, forbidden_id=shown_id
      )
      sampled_hidden = _evenly_spaced_sample(unseen_ids, 8)
      if sampled_hidden:
        hidden_total = 0.0
        for draw_id in sampled_hidden:
          next_mask = unseen_mask ^ (1 << draw_id)
          next_ids = [cid for cid in unseen_ids if cid != draw_id]
          hidden_total += _best_discard_one_ply(
              hand6 + [draw_id],
              next_ids,
              next_mask,
          )
        val_hidden = hidden_total / len(sampled_hidden)
      else:
        val_hidden = -1e9

    if val_shown - self.shown_info_cost > val_hidden:
      self.last_thought = (
          f'Took Shown ({self.depth}-ply expected {val_shown:.1f} vs Hidden {val_hidden:.1f})'
      )
      return 'S'
    self.last_thought = (
        f'Drew Hidden ({self.depth}-ply expected {val_hidden:.1f} vs Shown {val_shown:.1f})'
    )
    return 'T'

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids = set(obs['legal_action_ids'])
    unseen_mask = get_unseen_mask(obs)
    unseen_ids = mask_to_ids(unseen_mask)
    next_wanted, next_discarded = get_next_player_hints(obs)

    # Step 1: Score every legal discard with 1-level lookahead (and catch instant wins).
    scored_candidates = []
    for i, cid in enumerate(hand7):
      if cid not in legal_ids:
        continue
      hand6 = hand7[:i] + hand7[i + 1 :]
      imm = score_hand(hand6, unseen_mask)
      if imm >= WIN_SCORE:
        card = id_to_card(cid)
        self.last_thought = f'Put down {card_label(card)} — winning hand!'
        return card

      exp1 = _expected_one_turn_value(hand6, unseen_ids, unseen_mask)
      danger = opponent_danger(cid, next_wanted, next_discarded)
      val1 = exp1 - self.danger_weight * danger
      scored_candidates.append((val1, cid, hand6, danger))

    scored_candidates.sort(key=lambda item: item[0], reverse=True)
    if self.depth == 1:
      card = id_to_card(scored_candidates[0][1])
      if len(scored_candidates) >= 2:
        self.last_thought = (
            f'Put down {card_label(card)} (1-ply {scored_candidates[0][0]:.1f} vs '
            f'{card_label(id_to_card(scored_candidates[1][1]))} {scored_candidates[1][0]:.1f})'
        )
      return card

    # Step 2 (depth=2): Refine the top 3 discards with 2-level lookahead.
    refined = []
    for _, cid, hand6, danger in scored_candidates[:3]:
      val2 = _expected_two_turn_value(
          hand6, unseen_ids, unseen_mask, gamma=self.gamma
      ) - self.danger_weight * 1.5 * danger
      refined.append((val2, cid))

    refined.sort(key=lambda item: item[0], reverse=True)
    card = id_to_card(refined[0][1])
    if len(refined) >= 2:
      self.last_thought = (
          f'Put down {card_label(card)} (2-ply {refined[0][0]:.1f} vs '
          f'{card_label(id_to_card(refined[1][1]))} {refined[1][0]:.1f})'
      )
    return card
