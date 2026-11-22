"""Model 1: PyTorch TD(lambda) Value-Based RL Player (TD-Gammon / AlphaZero-lite).

Core RL idea:
  - Trains a PyTorch CardMLP V(s) -> (0, 1) to predict win probability from any
    6-card post-discard hand ('afterstate').
  - At play time:
      * DISCARD: scores all legal candidate 6-card hands in one PyTorch batch
        and picks the highest predicted win probability.
      * DRAW: compares taking Shown ('S') against the average predicted value
        of drawing an unseen card from Hidden ('T').
  - Learning rule: TD(lambda) backward returns + Binary Cross-Entropy + Adam.
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

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
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


def _evenly_spaced(items, max_count):
  n = len(items)
  if n <= max_count:
    return items
  step = n / max_count
  return [items[int(i * step)] for i in range(max_count)]


class TDValueBot(BotPolicy):
  """PyTorch RL Bot #1: learns a 6-card afterstate value network via TD(lambda)."""

  name = 'RL Bot — TD Value (PyTorch)'
  player_type = 'bot:rl_td_value'

  def __init__(
      self,
      model_path=None,
      hidden_dim=32,
      gamma=0.96,
      lam=0.70,
      lr=0.003,
      seed=101,
  ):
    self.gamma = gamma
    self.lam = lam
    self.epsilon = 0.0
    self._rng = random.Random(seed)
    self.metadata = {}

    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt['config']
      self.net = CardMLP(
          input_dim=cfg['input_dim'],
          hidden_dim=cfg['hidden_dim'],
          output_activation='sigmoid',
      )
      self.net.load_state_dict(ckpt['net_state'])
      self.metadata = ckpt.get('metadata', {})
    else:
      torch.manual_seed(seed)
      self.net = CardMLP(
          input_dim=AFTERSTATE_DIM,
          hidden_dim=hidden_dim,
          output_activation='sigmoid',
      )

    self.optimizer = torch.optim.Adam(
        self.net.parameters(), lr=lr, weight_decay=1e-4
    )

  def set_lr(self, lr):
    for group in self.optimizer.param_groups:
      group['lr'] = lr

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
    feat_rows = []
    for i in range(7):
      cid = hand7[i]
      if cid == forbidden_id:
        continue
      h6 = hand7[:i] + hand7[i + 1 :]
      if is_winning_hand_ids(h6):
        return 1.0
      feat_rows.append(
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
    return max(self.net.predict_batch(feat_rows))

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

    base_val = self.net.predict_one(
        extract_afterstate_features(
            hand6, unseen_mask, deck_rem, next_wanted, next_discarded
        )
    )
    relevant_mask = 0
    for cid in hand6:
      relevant_mask |= SINGLE_PARTNER_MASKS[cid]

    relevant_ids = [cid for cid in unseen_ids if relevant_mask & (1 << cid)]
    brick_count = len(unseen_ids) - len(relevant_ids)

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

    cids = []
    feat_rows = []
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
      cids.append(cid)
      feat_rows.append(feats)

    if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
      pick = self._rng.randrange(len(cids))
    else:
      vals = self.net.predict_batch(feat_rows)
      pick = max(range(len(vals)), key=lambda idx: vals[idx])

    card = id_to_card(cids[pick])
    return (card, feat_rows[pick]) if return_features else card

  def learn_episode(self, afterstate_features, outcome, lr=None):
    """Updates V(s) from one player's episode using PyTorch TD(lambda) + Adam."""
    if not afterstate_features:
      return 0.0
    if lr is not None:
      self.set_lr(lr)

    x = torch.tensor(afterstate_features, dtype=torch.float32)
    preds = self.net(x)

    # Build TD(lambda) targets without tracking gradients through the target.
    with torch.no_grad():
      n = len(afterstate_features)
      v_vals = preds.tolist()
      targets = [0.0] * n
      targets[-1] = float(outcome)
      for t in range(n - 2, -1, -1):
        next_v = 0.5 + self.gamma * (v_vals[t + 1] - 0.5)
        next_ret = 0.5 + self.gamma * (targets[t + 1] - 0.5)
        targets[t] = (1.0 - self.lam) * next_v + self.lam * next_ret
      target_tensor = torch.tensor(targets, dtype=torch.float32)

    loss = F.binary_cross_entropy(preds.clamp(1e-6, 1.0 - 1e-6), target_tensor)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
    self.optimizer.step()
    return float(loss.item())

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    save_checkpoint(
        path,
        {
            'config': {
                'input_dim': self.net.input_dim,
                'hidden_dim': self.net.hidden_dim,
            },
            'net_state': self.net.state_dict(),
            'metadata': self.metadata,
        },
    )
