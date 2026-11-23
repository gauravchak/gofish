"""Model 1: PyTorch TD(lambda) Value-Based RL Player with Live Target Network.

Core RL idea:
  - Trains a PyTorch CardMLP V(s) -> (0, 1) to predict win probability from any
    6-card post-discard hand ('afterstate').
  - Maintains a slow-moving Target Network (tau=0.005) so Experience Replay
    always recomputes FRESH bootstrap targets instead of training on stale
    targets from weaker earlier generations!
  - At play time:
      * DISCARD: scores all legal candidate 6-card hands in one PyTorch batch
        and picks the highest predicted win probability.
      * DRAW: compares taking Shown ('S') against the average predicted value
        of drawing an unseen card from Hidden ('T').
"""

import copy
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
from game.rules import card_label, id_to_card
from ml.features import AFTERSTATE_DIM, extract_afterstate_features
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


def _evenly_spaced(items, max_count):
  n = len(items)
  if n <= max_count:
    return items
  step = n / max_count
  return [items[int(i * step)] for i in range(max_count)]


class TDValueBot(BotPolicy):
  """PyTorch RL Bot #1: learns a 6-card afterstate value network via live TD + MC."""

  name = 'RL Bot — TD Value (PyTorch)'
  player_type = 'bot:rl_td_value'

  def __init__(
      self,
      model_path=None,
      hidden_dim=64,
      gamma=0.96,
      lam=0.35,
      target_tau=0.005,
      lr=0.0012,
      seed=101,
  ):
    self.gamma = gamma
    self.lam = lam
    self.target_tau = target_tau
    self.epsilon = 0.0
    self.explore_temp = 0.0
    self._rng = random.Random(seed)
    self.metadata = {}

    loaded = False
    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt.get('config', {})
      if cfg.get('input_dim') == AFTERSTATE_DIM and 'net_state' in ckpt:
        self.net = CardMLP(
            input_dim=cfg['input_dim'],
            hidden_dim=cfg['hidden_dim'],
            output_activation='sigmoid',
        )
        self.net.load_state_dict(ckpt['net_state'])
        self.metadata = ckpt.get('metadata', {})
        loaded = True

    if not loaded:
      torch.manual_seed(seed)
      self.net = CardMLP(
          input_dim=AFTERSTATE_DIM,
          hidden_dim=hidden_dim,
          output_activation='sigmoid',
      )

    # Soft-updated target network prevents stale replay targets & stabilizes TD!
    self.target_net = copy.deepcopy(self.net)
    self.optimizer = torch.optim.Adam(
        self.net.parameters(), lr=lr, weight_decay=1e-4
    )

  def set_lr(self, lr):
    for group in self.optimizer.param_groups:
      group['lr'] = lr

  @torch.no_grad()
  def _soft_update_target(self):
    for p_target, p_online in zip(
        self.target_net.parameters(), self.net.parameters()
    ):
      p_target.lerp_(p_online, self.target_tau)

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
      self.last_thought = 'Took Shown — completes a winning hand!'
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

    if val_shown > val_hidden:
      self.last_thought = (
          f'Took Shown (win prob {100*val_shown:.0f}% vs Hidden {100*val_hidden:.0f}%)'
      )
      return 'S'
    self.last_thought = (
        f'Drew Hidden (win prob {100*val_hidden:.0f}% vs Shown {100*val_shown:.0f}%)'
    )
    return 'T'

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
      if self.explore_temp > 0.0 and len(vals) > 1:
        # Boltzmann exploration: explores competitive discards without random blunders!
        probs = torch.softmax(
            torch.tensor(vals, dtype=torch.float32) / self.explore_temp, dim=-1
        ).tolist()
        r = self._rng.random()
        acc = 0.0
        pick = len(vals) - 1
        for idx, p in enumerate(probs):
          acc += p
          if r <= acc:
            pick = idx
            break
      else:
        ranked = sorted(range(len(vals)), key=lambda idx: vals[idx], reverse=True)
        pick = ranked[0]
        if len(ranked) >= 2:
          alt = ranked[1]
          self.last_thought = (
              f'Put down {card_label(id_to_card(cids[pick]))} '
              f'({100*vals[pick]:.0f}% win prob vs '
              f'{card_label(id_to_card(cids[alt]))} {100*vals[alt]:.0f}%)'
          )

    card = id_to_card(cids[pick])
    return (card, feat_rows[pick]) if return_features else card

  def build_episode_samples(self, afterstate_features, outcome):
    """Returns live replay transitions [(s_t, s_next_or_None, mc_target_t), ...].

    CRITICAL RL FIX: We store the next-state features s_{t+1} instead of baking
    in a frozen TD target from old weights. That way train_batch() always
    bootstraps from our LIVE target_net — zero stale targets in ReplayBuffer!
    """
    if not afterstate_features:
      return []
    n = len(afterstate_features)
    mc_targets = [0.0] * n
    mc_targets[-1] = float(outcome)
    for t in range(n - 2, -1, -1):
      mc_targets[t] = 0.5 + self.gamma * (mc_targets[t + 1] - 0.5)

    transitions = []
    for t in range(n):
      nxt = afterstate_features[t + 1] if t + 1 < n else None
      transitions.append((afterstate_features[t], nxt, mc_targets[t]))
    return transitions

  def train_batch(self, samples, lr=None):
    """Mini-batch PyTorch BCE update using LIVE target_net bootstrapping."""
    if not samples:
      return 0.0
    if lr is not None:
      self.set_lr(lr)

    x_rows = []
    targets = []
    next_indices = []
    next_rows = []

    for idx, item in enumerate(samples):
      if len(item) == 2:
        # Backward-compatible 2-tuple (feats, direct_target) e.g. from human logs
        x_rows.append(item[0])
        targets.append(float(item[1]))
      else:
        feats, nxt_feats, mc_target = item
        x_rows.append(feats)
        targets.append(float(mc_target))
        if nxt_feats is not None:
          next_indices.append(idx)
          next_rows.append(nxt_feats)

    if next_rows:
      with torch.no_grad():
        next_vs = self.target_net(
            torch.tensor(next_rows, dtype=torch.float32)
        ).tolist()
      for list_pos, sample_idx in enumerate(next_indices):
        boot_v = 0.5 + self.gamma * (next_vs[list_pos] - 0.5)
        targets[sample_idx] = (
            (1.0 - self.lam) * boot_v + self.lam * targets[sample_idx]
        )

    x = torch.tensor(x_rows, dtype=torch.float32)
    y = torch.tensor(targets, dtype=torch.float32).clamp(0.0, 1.0)
    preds = self.net(x)
    loss = F.binary_cross_entropy(preds.clamp(1e-6, 1.0 - 1e-6), y)
    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.net.parameters(), 2.0)
    self.optimizer.step()
    self._soft_update_target()
    return float(loss.item())

  def learn_episode(self, afterstate_features, outcome, lr=None):
    """Updates V(s) from one player's episode using live target-net TD + Adam."""
    samples = self.build_episode_samples(afterstate_features, outcome)
    return self.train_batch(samples, lr=lr)

  def clone_frozen(self, label=None):
    """Returns a frozen copy of this bot for Hall-of-Fame league sparring."""
    snap = TDValueBot(hidden_dim=self.net.hidden_dim)
    snap.net.load_state_dict(self.net.state_dict())
    snap.target_net.load_state_dict(self.net.state_dict())
    snap.epsilon = 0.0
    if label:
      snap.name = label
    return snap

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
