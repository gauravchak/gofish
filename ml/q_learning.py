"""Model 2: Q-Learning (DQN-style) RL Player.

Core RL idea:
  - Learns explicit action-values Q(s, a) in (-1, 1) directly for EVERY legal
    action — no draw simulator needed at play time!
  - Uses two small networks (one for DRAW, one for DISCARD) plus a soft-updated
    target network (just like Deep Q-Networks) for stable Bellman backups:
        target_t = R_final                              (if terminal)
        target_t = gamma * max_{a'} Q_target(s_{t+1}, a')  (otherwise)
"""

import json
import os
import random

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import id_to_card
from ml.features import ACTION_DIM, extract_all_action_features
from ml.nn import TinyMLP


class QLearningBot(BotPolicy):
  """RL Bot #2: off-policy Q-Learning with separate DRAW and DISCARD Q-networks."""

  name = 'RL Bot — Q-Learning'
  player_type = 'bot:rl_q_learning'

  def __init__(
      self,
      model_path=None,
      hidden_dim=16,
      gamma=0.95,
      target_tau=0.05,
      seed=202,
  ):
    self.gamma = gamma
    self.target_tau = target_tau
    self.epsilon = 0.0
    self._rng = random.Random(seed)
    self.metadata = {}

    if model_path and os.path.exists(model_path):
      with open(model_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
      self.draw_net = TinyMLP.from_dict(payload['draw_net'])
      self.discard_net = TinyMLP.from_dict(payload['discard_net'])
      self.metadata = payload.get('metadata', {})
    else:
      self.draw_net = TinyMLP(
          input_dim=ACTION_DIM,
          hidden_dim=hidden_dim,
          output_activation='tanh',
          seed=seed,
      )
      self.discard_net = TinyMLP(
          input_dim=ACTION_DIM,
          hidden_dim=hidden_dim,
          output_activation='tanh',
          seed=seed + 1,
      )

    # Target networks for stable Bellman updates (DQN trick).
    self.target_draw_net = TinyMLP.from_dict(self.draw_net.to_dict())
    self.target_discard_net = TinyMLP.from_dict(self.discard_net.to_dict())

  def _max_target_q_from_feats(self, phase, next_all_feats):
    """Computes max_{a' in legal(s')} Q_target(s', a') from cached action features."""
    net = self.target_draw_net if phase == 'DRAW' else self.target_discard_net
    best = -2.0
    for feats in next_all_feats:
      q = net.predict(feats)
      if q > best:
        best = q
    return best

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
      chosen_idx = 0
      best_q = -2.0
      for idx, feats in enumerate(all_feats):
        q = self.draw_net.predict(feats)
        if q > best_q:
          best_q = q
          chosen_idx = idx

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
      chosen_idx = 0
      best_q = -2.0
      for idx, feats in enumerate(all_feats):
        q = self.discard_net.predict(feats)
        if q > best_q:
          best_q = q
          chosen_idx = idx

    card = id_to_card(legal_ids[chosen_idx])
    rec = {'phase': 'DISCARD', 'feats': all_feats[chosen_idx], 'all_feats': all_feats}
    return (card, rec) if return_record else card

  def learn_episode(self, transitions, reward, lr=0.012, l2=1e-4):
    """Applies Bellman Q-learning updates over one player's episode transitions.

    Args:
      transitions: List of dicts {'phase': 'DRAW'|'DISCARD', 'feats': [...], 'all_feats': [...]}.
      reward: +1.0 for win, -1.0 for loss, 0.0 for draw.
    """
    if not transitions:
      return 0.0

    total_loss = 0.0
    n = len(transitions)
    for idx, step in enumerate(transitions):
      if idx == n - 1:
        target = reward
      else:
        nxt = transitions[idx + 1]
        step_gamma = 0.99 if step['phase'] == 'DRAW' else self.gamma
        target = step_gamma * self._max_target_q_from_feats(
            nxt['phase'], nxt['all_feats']
        )

      net = self.draw_net if step['phase'] == 'DRAW' else self.discard_net
      total_loss += net.train_step(step['feats'], target, lr=lr, l2=l2)

    # Soft-update target networks toward online networks.
    self.target_draw_net.copy_weights_from(self.draw_net, tau=self.target_tau)
    self.target_discard_net.copy_weights_from(
        self.discard_net, tau=self.target_tau
    )
    return total_loss / n

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        'draw_net': self.draw_net.to_dict(),
        'discard_net': self.discard_net.to_dict(),
        'metadata': self.metadata,
    }
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(payload, f, indent=2)
      f.write('\n')
