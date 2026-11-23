"""Model 2: PyTorch Q-Learning (DQN-style) RL Player.

Core RL idea:
  - Learns explicit action-values Q(s, a) in (-1, 1) directly for EVERY legal
    DRAW and DISCARD action — no draw simulator needed at play time!
  - Uses two PyTorch CardMLPs (draw_net, discard_net) + soft-updated target
    networks (DQN trick) + Huber loss (F.smooth_l1_loss) + Adam optimizer.
"""

import copy
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import id_to_card
from ml.features import ACTION_DIM, extract_all_action_features
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


class QLearningBot(BotPolicy):
  """PyTorch RL Bot #2: off-policy Q-Learning with target networks."""

  name = 'RL Bot — Q-Learning (PyTorch)'
  player_type = 'bot:rl_q_learning'

  def __init__(
      self,
      model_path=None,
      hidden_dim=32,
      gamma=0.95,
      target_tau=0.05,
      lr=0.003,
      seed=202,
  ):
    self.gamma = gamma
    self.target_tau = target_tau
    self.epsilon = 0.0
    self._rng = random.Random(seed)
    self.metadata = {}

    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt['config']
      self.draw_net = CardMLP(cfg['input_dim'], cfg['hidden_dim'], 'tanh')
      self.discard_net = CardMLP(cfg['input_dim'], cfg['hidden_dim'], 'tanh')
      self.draw_net.load_state_dict(ckpt['draw_state'])
      self.discard_net.load_state_dict(ckpt['discard_state'])
      self.metadata = ckpt.get('metadata', {})
    else:
      torch.manual_seed(seed)
      self.draw_net = CardMLP(ACTION_DIM, hidden_dim, 'tanh')
      self.discard_net = CardMLP(ACTION_DIM, hidden_dim, 'tanh')

    self.target_draw_net = copy.deepcopy(self.draw_net)
    self.target_discard_net = copy.deepcopy(self.discard_net)

    self.optimizer = torch.optim.Adam(
        list(self.draw_net.parameters()) + list(self.discard_net.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

  def set_lr(self, lr):
    for group in self.optimizer.param_groups:
      group['lr'] = lr

  @torch.no_grad()
  def _soft_update_targets(self):
    for p_target, p_online in zip(
        self.target_draw_net.parameters(), self.draw_net.parameters()
    ):
      p_target.lerp_(p_online, self.target_tau)
    for p_target, p_online in zip(
        self.target_discard_net.parameters(), self.discard_net.parameters()
    ):
      p_target.lerp_(p_online, self.target_tau)

  def choose_draw(self, obs, return_record=False):
    legal, all_feats = extract_all_action_features(obs)
    if len(legal) == 1:
      rec = {'phase': 'DRAW', 'feats': all_feats[0], 'all_feats': all_feats}
      return (legal[0], rec) if return_record else legal[0]

    # Spot obvious immediate winning card from Shown.
    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]
    for i in range(6):
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        s_idx = legal.index('S')
        rec = {'phase': 'DRAW', 'feats': all_feats[s_idx], 'all_feats': all_feats}
        return ('S', rec) if return_record else 'S'

    if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
      chosen_idx = self._rng.randrange(len(legal))
    else:
      q_vals = self.draw_net.predict_batch(all_feats)
      chosen_idx = max(range(len(q_vals)), key=lambda idx: q_vals[idx])

    rec = {'phase': 'DRAW', 'feats': all_feats[chosen_idx], 'all_feats': all_feats}
    return (legal[chosen_idx], rec) if return_record else legal[chosen_idx]

  def choose_discard(self, obs, return_record=False):
    hand7 = obs['hand_ids']
    legal_ids, all_feats = extract_all_action_features(obs)

    # Spot obvious immediate winning discard.
    for idx, cid in enumerate(legal_ids):
      h_idx = hand7.index(cid)
      if is_winning_hand_ids(hand7[:h_idx] + hand7[h_idx + 1 :]):
        card = id_to_card(cid)
        rec = {'phase': 'DISCARD', 'feats': all_feats[idx], 'all_feats': all_feats}
        return (card, rec) if return_record else card

    if self.epsilon > 0.0 and self._rng.random() < self.epsilon:
      chosen_idx = self._rng.randrange(len(legal_ids))
    else:
      q_vals = self.discard_net.predict_batch(all_feats)
      chosen_idx = max(range(len(q_vals)), key=lambda idx: q_vals[idx])

    card = id_to_card(legal_ids[chosen_idx])
    rec = {'phase': 'DISCARD', 'feats': all_feats[chosen_idx], 'all_feats': all_feats}
    return (card, rec) if return_record else card

  def learn_episode(self, transitions, reward, lr=None):
    """Applies batched Bellman Q-learning updates over one player's episode."""
    if not transitions:
      return 0.0
    if lr is not None:
      self.set_lr(lr)

    draw_x, draw_targets = [], []
    disc_x, disc_targets = [], []

    n = len(transitions)
    with torch.no_grad():
      for idx, step in enumerate(transitions):
        if idx == n - 1:
          target = float(reward)
        else:
          nxt = transitions[idx + 1]
          nxt_net = (
              self.target_draw_net
              if nxt['phase'] == 'DRAW'
              else self.target_discard_net
          )
          step_gamma = 0.99 if step['phase'] == 'DRAW' else self.gamma
          target = step_gamma * max(nxt_net.predict_batch(nxt['all_feats']))

        if step['phase'] == 'DRAW':
          draw_x.append(step['feats'])
          draw_targets.append(target)
        else:
          disc_x.append(step['feats'])
          disc_targets.append(target)

    loss = torch.tensor(0.0)
    if draw_x:
      preds = self.draw_net(torch.tensor(draw_x, dtype=torch.float32))
      loss = loss + F.smooth_l1_loss(
          preds, torch.tensor(draw_targets, dtype=torch.float32)
      )
    if disc_x:
      preds = self.discard_net(torch.tensor(disc_x, dtype=torch.float32))
      loss = loss + F.smooth_l1_loss(
          preds, torch.tensor(disc_targets, dtype=torch.float32)
      )

    self.optimizer.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(self.draw_net.parameters()) + list(self.discard_net.parameters()),
        2.0,
    )
    self.optimizer.step()
    self._soft_update_targets()
    return float(loss.item())

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    save_checkpoint(
        path,
        {
            'config': {
                'input_dim': self.draw_net.input_dim,
                'hidden_dim': self.draw_net.hidden_dim,
            },
            'draw_state': self.draw_net.state_dict(),
            'discard_state': self.discard_net.state_dict(),
            'metadata': self.metadata,
        },
    )
