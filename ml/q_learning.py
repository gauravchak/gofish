"""Model 2: PyTorch Double DQN (Q-Learning) RL Player.

Core RL improvements over vanilla DQN:
  1. Double DQN target selection: the ONLINE network picks the best next action
     a* = argmax Q_online(s', a'), while the TARGET network evaluates it
     Q_target(s', a*). This eliminates classic DQN max-overestimation bias!
  2. Slow soft target updates (tau=0.005) so Q-targets stay rock steady.
  3. Blended Double-DQN + discounted Monte-Carlo credit assignment so end-of-game
     +1/-1 rewards propagate cleanly all the way back to Turn 1!
"""

import copy
import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import card_label, id_to_card
from ml.features import ACTION_DIM, extract_all_action_features
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


class QLearningBot(BotPolicy):
  """PyTorch RL Bot #2: Double DQN with soft target networks & MC-blended returns."""

  name = 'RL Bot — Double DQN (PyTorch)'
  player_type = 'bot:rl_q_learning'

  def __init__(
      self,
      model_path=None,
      hidden_dim=64,
      gamma=0.95,
      target_tau=0.005,
      mc_blend=0.15,
      lr=0.0012,
      seed=202,
  ):
    self.gamma = gamma
    self.target_tau = target_tau
    self.mc_blend = mc_blend
    self.epsilon = 0.0
    self.explore_temp = 0.0
    self._rng = random.Random(seed)
    self.metadata = {}

    loaded = False
    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt.get('config', {})
      if (
          cfg.get('input_dim') == ACTION_DIM
          and 'draw_state' in ckpt
          and 'discard_state' in ckpt
      ):
        self.draw_net = CardMLP(cfg['input_dim'], cfg['hidden_dim'], 'tanh')
        self.discard_net = CardMLP(cfg['input_dim'], cfg['hidden_dim'], 'tanh')
        self.draw_net.load_state_dict(ckpt['draw_state'])
        self.discard_net.load_state_dict(ckpt['discard_state'])
        self.metadata = ckpt.get('metadata', {})
        loaded = True

    if not loaded:
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
      if len(legal) == 2:
        other_idx = 1 - chosen_idx
        self.last_thought = (
            f'Chose {"Shown" if legal[chosen_idx] == "S" else "Hidden"} '
            f'(Q={q_vals[chosen_idx]:+.2f} vs {q_vals[other_idx]:+.2f})'
        )

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
      if self.explore_temp > 0.0 and len(q_vals) > 1:
        # Boltzmann exploration: tries close Q moves without throwing away sets!
        probs = torch.softmax(
            torch.tensor(q_vals, dtype=torch.float32) / self.explore_temp, dim=-1
        ).tolist()
        r = self._rng.random()
        acc = 0.0
        chosen_idx = len(q_vals) - 1
        for idx, p in enumerate(probs):
          acc += p
          if r <= acc:
            chosen_idx = idx
            break
      else:
        ranked = sorted(range(len(q_vals)), key=lambda idx: q_vals[idx], reverse=True)
        chosen_idx = ranked[0]
        if len(ranked) >= 2:
          alt = ranked[1]
          self.last_thought = (
              f'Put down {card_label(id_to_card(legal_ids[chosen_idx]))} '
              f'(Q={q_vals[chosen_idx]:+.2f} vs '
              f'{card_label(id_to_card(legal_ids[alt]))} {q_vals[alt]:+.2f})'
          )

    card = id_to_card(legal_ids[chosen_idx])
    rec = {'phase': 'DISCARD', 'feats': all_feats[chosen_idx], 'all_feats': all_feats}
    return (card, rec) if return_record else card

  def build_replay_transitions(self, transitions, reward):
    """Converts one episode into Double-DQN replay tuples with MC returns."""
    n = len(transitions)
    if n == 0:
      return []

    mc_returns = [0.0] * n
    mc_returns[-1] = float(reward)
    for idx in range(n - 2, -1, -1):
      step_gamma = 0.99 if transitions[idx]['phase'] == 'DRAW' else self.gamma
      mc_returns[idx] = step_gamma * mc_returns[idx + 1]

    items = []
    for idx, step in enumerate(transitions):
      if idx == n - 1:
        items.append((
            step['phase'],
            step['feats'],
            None,
            None,
            float(reward),
            mc_returns[idx],
        ))
      else:
        nxt = transitions[idx + 1]
        items.append((
            step['phase'],
            step['feats'],
            nxt['phase'],
            nxt['all_feats'],
            None,
            mc_returns[idx],
        ))
    return items

  def train_batch(self, batch, lr=None):
    """Double-DQN mini-batch update from ReplayBuffer with MC-blended targets."""
    if not batch:
      return 0.0
    if lr is not None:
      self.set_lr(lr)

    draw_x, draw_targets = [], []
    disc_x, disc_targets = [], []

    with torch.no_grad():
      for item in batch:
        if len(item) == 5:
          phase, feats, nxt_phase, nxt_all_feats, term_reward = item
          mc_ret = None
        else:
          phase, feats, nxt_phase, nxt_all_feats, term_reward, mc_ret = item

        if term_reward is not None:
          target = term_reward
        else:
          online_net = (
              self.draw_net if nxt_phase == 'DRAW' else self.discard_net
          )
          target_net = (
              self.target_draw_net
              if nxt_phase == 'DRAW'
              else self.target_discard_net
          )
          nxt_tensor = torch.tensor(nxt_all_feats, dtype=torch.float32)
          # Double DQN: ONLINE net picks best action, TARGET net scores it!
          best_idx = int(torch.argmax(online_net(nxt_tensor)).item())
          ddqn_q = float(target_net(nxt_tensor)[best_idx].item())
          step_gamma = 0.99 if phase == 'DRAW' else self.gamma
          boot_target = step_gamma * ddqn_q
          if mc_ret is not None:
            target = (1.0 - self.mc_blend) * boot_target + self.mc_blend * mc_ret
          else:
            target = boot_target
          target = max(-1.0, min(1.0, target))

        if phase == 'DRAW':
          draw_x.append(feats)
          draw_targets.append(target)
        else:
          disc_x.append(feats)
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

  def learn_episode(self, transitions, reward, lr=None):
    """Applies Double-DQN updates over one player's episode."""
    return self.train_batch(
        self.build_replay_transitions(transitions, reward), lr=lr
    )

  def clone_frozen(self, label=None):
    """Returns a frozen copy of this bot for Hall-of-Fame league sparring."""
    snap = QLearningBot(hidden_dim=self.draw_net.hidden_dim)
    snap.draw_net.load_state_dict(self.draw_net.state_dict())
    snap.discard_net.load_state_dict(self.discard_net.state_dict())
    snap.target_draw_net.load_state_dict(self.draw_net.state_dict())
    snap.target_discard_net.load_state_dict(self.discard_net.state_dict())
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
                'input_dim': self.draw_net.input_dim,
                'hidden_dim': self.draw_net.hidden_dim,
            },
            'draw_state': self.draw_net.state_dict(),
            'discard_state': self.discard_net.state_dict(),
            'metadata': self.metadata,
        },
    )
