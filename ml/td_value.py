"""Model 1: TD(lambda) Value-Based RL Player (TD-Gammon / AlphaZero-lite style).

Core RL idea:
  - Instead of hand-writing a heuristic hand score, train a neural network
    V(s) -> (0, 1) to predict the probability of winning from any 6-card
    post-discard hand ('afterstate').
  - At play time:
      * DISCARD: pick the legal discard leaving the highest-value 6-card hand.
      * DRAW: compare taking Shown ('S') against the expected value of drawing
        a random unseen card from Hidden ('T').
  - Learning rule (TD(lambda) backward returns):
      target_t = (1 - lam) * gamma * V(s_{t+1}) + lam * target_{t+1}
      with terminal target = 1.0 (win), 0.5 (draw), 0.0 (loss).
"""

import json
import os
import random

from bots.base import BotPolicy
from bots.heuristic import (
    SINGLE_PARTNER_MASKS,
    get_next_player_hints,
    get_unseen_mask,
    is_winning_hand_ids,
    mask_to_ids,
)
from game.rules import id_to_card
from ml.features import AFTERSTATE_DIM, extract_afterstate_features
from ml.nn import TinyMLP


def _evenly_spaced(items, max_count):
  n = len(items)
  if n <= max_count:
    return items
  step = n / max_count
  return [items[int(i * step)] for i in range(max_count)]


class TDValueBot(BotPolicy):
  """RL Bot #1: learns a 6-card afterstate value function via TD(lambda)."""

  name = 'RL Bot — TD Value'
  player_type = 'bot:rl_td_value'

  def __init__(
      self,
      model_path=None,
      hidden_dim=16,
      gamma=0.96,
      lam=0.70,
      seed=101,
  ):
    self.gamma = gamma
    self.lam = lam
    self.epsilon = 0.0
    self._rng = random.Random(seed)
    self.metadata = {}

    if model_path and os.path.exists(model_path):
      self.net, self.metadata = TinyMLP.load(model_path)
    else:
      self.net = TinyMLP(
          input_dim=AFTERSTATE_DIM,
          hidden_dim=hidden_dim,
          output_activation='sigmoid',
          seed=seed,
      )

  def _best_afterstate_value(
      self,
      hand7,
      unseen_mask,
      deck_rem,
      next_wanted,
      next_discarded,
      forbidden_id=None,
      took_shown=False,
  ):
    best_val = -1.0
    for i in range(7):
      cid = hand7[i]
      if cid == forbidden_id:
        continue
      h6 = hand7[:i] + hand7[i + 1 :]
      if is_winning_hand_ids(h6):
        return 1.0
      feats = extract_afterstate_features(
          h6,
          unseen_mask,
          deck_rem,
          next_wanted,
          next_discarded,
          discarded_id=cid,
          took_shown=took_shown,
      )
      val = self.net.predict(feats)
      if val > best_val:
        best_val = val
    return best_val

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]
    if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
      return self._rng.choice(legal)

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    unseen_mask = get_unseen_mask(obs)
    next_wanted, next_discarded = get_next_player_hints(obs)
    deck_rem = obs['deck_remaining']

    # Branch 1: Take Shown ('S') — deterministic.
    val_shown = self._best_afterstate_value(
        hand6 + [shown_id],
        unseen_mask,
        deck_rem,
        next_wanted,
        next_discarded,
        forbidden_id=shown_id,
        took_shown=True,
    )
    if val_shown >= 1.0:
      return 'S'

    # Branch 2: Draw Hidden ('T') — expected value over unseen cards.
    unseen_ids = mask_to_ids(unseen_mask)
    if not unseen_ids:
      return 'S'

    base_val = self.net.predict(
        extract_afterstate_features(
            hand6, unseen_mask, deck_rem, next_wanted, next_discarded
        )
    )
    relevant_mask = 0
    for cid in hand6:
      relevant_mask |= SINGLE_PARTNER_MASKS[cid]

    relevant_ids = [cid for cid in unseen_ids if relevant_mask & (1 << cid)]
    brick_count = len(unseen_ids) - len(relevant_ids)

    # Sample up to 6 relevant unseen draws for fast self-play & live play.
    sampled_rel = _evenly_spaced(relevant_ids, 6)
    rel_total = 0.0
    for draw_id in sampled_rel:
      next_mask = unseen_mask ^ (1 << draw_id)
      val = self._best_afterstate_value(
          hand6 + [draw_id],
          next_mask,
          max(deck_rem - 1, 0),
          next_wanted,
          next_discarded,
          took_shown=False,
      )
      rel_total += val if val > base_val else base_val

    avg_rel = (rel_total / len(sampled_rel)) if sampled_rel else base_val
    val_hidden = (
        brick_count * base_val + len(relevant_ids) * avg_rel
    ) / len(unseen_ids)

    return 'S' if val_shown > val_hidden else 'T'

  def choose_discard(self, obs, return_features=False):
    hand7 = obs['hand_ids']
    legal_ids = list(obs['legal_action_ids'])
    unseen_mask = get_unseen_mask(obs)
    next_wanted, next_discarded = get_next_player_hints(obs)
    deck_rem = obs['deck_remaining']
    took_shown = obs.get('last_picked_shown_id') is not None

    # Always check immediate win first.
    candidates = []
    for i, cid in enumerate(hand7):
      if cid not in legal_ids:
        continue
      h6 = hand7[:i] + hand7[i + 1 :]
      feats = extract_afterstate_features(
          h6,
          unseen_mask,
          deck_rem,
          next_wanted,
          next_discarded,
          discarded_id=cid,
          took_shown=took_shown,
      )
      if is_winning_hand_ids(h6):
        card = id_to_card(cid)
        return (card, feats) if return_features else card
      candidates.append((cid, feats))

    if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
      cid, feats = self._rng.choice(candidates)
      card = id_to_card(cid)
      return (card, feats) if return_features else card

    best_cid, best_feats = candidates[0]
    best_val = -1.0
    for cid, feats in candidates:
      val = self.net.predict(feats)
      if val > best_val:
        best_val = val
        best_cid, best_feats = cid, feats

    card = id_to_card(best_cid)
    return (card, best_feats) if return_features else card

  def learn_episode(self, afterstate_features, outcome, lr=0.015, l2=1e-4):
    """Updates V(s) from one player's sequence of post-discard feature vectors.

    Args:
      afterstate_features: List of 16-dim feature vectors (one per turn).
      outcome: 1.0 for win, 0.0 for loss, 0.5 for draw.
    """
    if not afterstate_features:
      return 0.0

    # Precompute current value predictions for TD(lambda) targets.
    preds = [self.net.predict(f) for f in afterstate_features]
    n = len(afterstate_features)
    targets = [0.0] * n
    targets[-1] = outcome

    # Backward TD(lambda) pass: blends 1-step bootstrap with eventual outcome.
    for t in range(n - 2, -1, -1):
      # Keep win probability centered around 0.5 when discounting toward the future.
      next_v = 0.5 + self.gamma * (preds[t + 1] - 0.5)
      next_ret = 0.5 + self.gamma * (targets[t + 1] - 0.5)
      targets[t] = (1.0 - self.lam) * next_v + self.lam * next_ret

    total_loss = 0.0
    for feats, target in zip(afterstate_features, targets):
      total_loss += self.net.train_step(feats, target, lr=lr, l2=l2)
    return total_loss / n

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    self.net.save(path, metadata=self.metadata)
