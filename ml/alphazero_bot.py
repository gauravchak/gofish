"""AlphaZero-Lite Hybrid Bot: Learned PyTorch Value Network + 1-Level Tree Search.

Why combine Neural RL + Tree Search?
  - Pure TreeBot uses a hand-written heuristic score at the leaves.
  - Pure TDValueBot only looks 0 steps ahead on DISCARD (greedy V(s')).
  - AlphaZeroLiteBot plugs our trained PyTorch Value Network V(s) straight into
    a 1-level Expectimax lookahead tree — combining learned neural intuition
    with exact lookahead math!
"""

from bots.base import BotPolicy
from bots.heuristic import (
    SINGLE_PARTNER_MASKS,
    get_next_player_hints,
    get_unseen_mask,
    is_winning_hand_ids,
    mask_to_ids,
)
from game.rules import id_to_card
from ml.features import extract_afterstate_features
from ml.td_value import TDValueBot


def _evenly_spaced(items, max_count):
  n = len(items)
  if n <= max_count:
    return items
  step = n / max_count
  return [items[int(i * step)] for i in range(max_count)]


class AlphaZeroLiteBot(BotPolicy):
  """Hybrid bot: 1-level Expectimax tree evaluated by our trained PyTorch TD Value Net."""

  name = 'RL + Tree Bot (AlphaZero-Lite)'
  player_type = 'bot:rl_alphazero'

  def __init__(self, model_path=None, td_bot=None, draw_sample=4):
    self.value_bot = td_bot if td_bot is not None else TDValueBot(model_path=model_path)
    self.draw_sample = draw_sample

  def _expected_next_turn_value(
      self, hand6, unseen_ids, unseen_mask, deck_rem, next_wanted, next_discarded
  ):
    """1-level Expectimax chance node using batched PyTorch V(s) at the leaves."""
    if not unseen_ids:
      return self.value_bot.net.predict_one(
          extract_afterstate_features(
              hand6, unseen_mask, deck_rem, next_wanted, next_discarded
          )
      )

    base_val = self.value_bot.net.predict_one(
        extract_afterstate_features(
            hand6, unseen_mask, deck_rem, next_wanted, next_discarded
        )
    )
    relevant_mask = 0
    for cid in hand6:
      relevant_mask |= SINGLE_PARTNER_MASKS[cid]

    relevant_ids = [cid for cid in unseen_ids if relevant_mask & (1 << cid)]
    brick_count = len(unseen_ids) - len(relevant_ids)

    # Check ALL relevant unseen cards for immediate 1-turn wins (exact!),
    # and batch-evaluate a sample of non-winning draws through PyTorch V(s).
    win_draws = 0
    non_win_rel = []
    for draw_id in relevant_ids:
      hand7 = hand6 + [draw_id]
      won = False
      for i in range(7):
        if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
          won = True
          break
      if won:
        win_draws += 1
      else:
        non_win_rel.append(draw_id)

    sampled = _evenly_spaced(non_win_rel, self.draw_sample)
    rel_non_win_total = 0.0
    if sampled:
      # Build one flat batch across all sampled draws x 7 discards for fast PyTorch inference!
      batch_feats = []
      counts = []
      next_deck = max(deck_rem - 1, 0)
      for draw_id in sampled:
        next_mask = unseen_mask ^ (1 << draw_id)
        hand7 = hand6 + [draw_id]
        for i in range(7):
          h6 = hand7[:i] + hand7[i + 1 :]
          batch_feats.append(
              extract_afterstate_features(
                  h6,
                  next_mask,
                  next_deck,
                  next_wanted,
                  next_discarded,
                  discarded_id=hand7[i],
                  took_shown=False,
              )
          )
        counts.append(7)

      preds = self.value_bot.net.predict_batch(batch_feats)
      offset = 0
      for c in counts:
        best_v = max(preds[offset : offset + c])
        rel_non_win_total += best_v if best_v > base_val else base_val
        offset += c
      avg_non_win = rel_non_win_total / len(sampled)
    else:
      avg_non_win = base_val

    total_val = (
        brick_count * base_val
        + win_draws * 1.0
        + len(non_win_rel) * avg_non_win
    )
    return total_val / len(unseen_ids)

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    unseen_mask = get_unseen_mask(obs)
    unseen_ids = mask_to_ids(unseen_mask)
    next_wanted, next_discarded = get_next_player_hints(obs)
    deck_rem = obs['deck_remaining']

    # Branch 1: Take Shown ('S') — evaluate best legal discard using PyTorch V(s).
    hand7 = hand6 + [shown_id]
    shown_feats = []
    for i in range(6):  # Index 6 is shown_id (forbidden to re-discard this turn)
      h6 = hand7[:i] + hand7[i + 1 :]
      if is_winning_hand_ids(h6):
        return 'S'
      shown_feats.append(
          extract_afterstate_features(
              h6,
              unseen_mask,
              deck_rem,
              next_wanted,
              next_discarded,
              discarded_id=hand7[i],
              took_shown=True,
          )
      )
    val_shown = max(self.value_bot.net.predict_batch(shown_feats))

    # Branch 2: Draw Hidden ('T') — exact win-out check + batched PyTorch expectation.
    val_hidden = self._expected_next_turn_value(
        hand6, unseen_ids, unseen_mask, deck_rem, next_wanted, next_discarded
    )
    return 'S' if val_shown > val_hidden else 'T'

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids = list(obs['legal_action_ids'])
    unseen_mask = get_unseen_mask(obs)
    unseen_ids = mask_to_ids(unseen_mask)
    next_wanted, next_discarded = get_next_player_hints(obs)
    deck_rem = obs['deck_remaining']
    took_shown = obs.get('last_picked_shown_id') is not None

    # Step 1: Score all legal discards immediately with PyTorch V(s), catching instant wins.
    cids = []
    h6_list = []
    imm_feats = []
    for i, cid in enumerate(hand7):
      if cid not in legal_ids:
        continue
      h6 = hand7[:i] + hand7[i + 1 :]
      if is_winning_hand_ids(h6):
        return id_to_card(cid)
      cids.append(cid)
      h6_list.append(h6)
      imm_feats.append(
          extract_afterstate_features(
              h6,
              unseen_mask,
              deck_rem,
              next_wanted,
              next_discarded,
              discarded_id=cid,
              took_shown=took_shown,
          )
      )

    imm_vals = self.value_bot.net.predict_batch(imm_feats)
    ranked = sorted(
        range(len(cids)), key=lambda idx: imm_vals[idx], reverse=True
    )

    # Step 2: Run 1-level Expectimax lookahead on the top 2 candidate discards!
    best_cid = cids[ranked[0]]
    best_score = -1.0
    for idx in ranked[:2]:
      lookahead_v = self._expected_next_turn_value(
          h6_list[idx],
          unseen_ids,
          unseen_mask,
          deck_rem,
          next_wanted,
          next_discarded,
      )
      # Blend immediate neural value (includes opponent-danger feature!) with 1-ply lookahead.
      combined = 0.45 * imm_vals[idx] + 0.55 * lookahead_v
      if combined > best_score:
        best_score = combined
        best_cid = cids[idx]

    return id_to_card(best_cid)
