"""Model 3: PyTorch Advantage Actor-Critic (A2C) Policy Gradient RL Player.

Core RL idea:
  - Actor (Policy pi(a|s)): scores every legal action with a logit z(s, a) and
    builds a PyTorch Categorical distribution over legal actions.
  - Critic (Value baseline V(s)): estimates expected return from the PRE-ACTION
    state s_t (never peeking at the chosen action a_t!).
  - Advantage: A_t = target_t - V(s_t).
  - Actor loss:  -(dist.log_prob(chosen) * A_t.detach()) - entropy_coef * dist.entropy()
  - Critic loss: MSE(V(s_t), target_t)
"""

import os

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import card_label, id_to_card
from ml.features import ACTION_DIM, AFTERSTATE_DIM, extract_all_action_features
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


def _pre_action_state_feats(all_feats):
  """Returns an action-independent summary of the current state for the Critic.

  CRITICAL RL RULE: The Critic baseline V(s_t) must depend ONLY on the pre-action
  state s_t — NEVER on the chosen action a_t! Using the post-action features of
  a_t biases the advantage A_t = G_t - V(s_t, a_t) and can invert learning.
  """
  best = all_feats[0]
  best_key = (best[1], best[2], best[4], best[5])
  for f in all_feats[1:]:
    key = (f[1], f[2], f[4], f[5])
    if key > best_key:
      best_key = key
      best = f
  return best[:AFTERSTATE_DIM]


class PolicyGradientBot(BotPolicy):
  """PyTorch RL Bot #3: Advantage Actor-Critic (A2C) using torch.distributions.Categorical."""

  name = 'RL Bot — Policy Gradient (PyTorch)'
  player_type = 'bot:rl_policy_gradient'

  def __init__(
      self,
      model_path=None,
      hidden_dim=32,
      gamma=0.95,
      entropy_coef=0.015,
      actor_lr=0.002,
      critic_lr=0.003,
      seed=303,
  ):
    self.gamma = gamma
    self.entropy_coef = entropy_coef
    self.stochastic = False
    self.temperature = 1.0
    self.metadata = {}

    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt['config']
      self.draw_actor = CardMLP(cfg['action_dim'], cfg['hidden_dim'], 'linear')
      self.discard_actor = CardMLP(cfg['action_dim'], cfg['hidden_dim'], 'linear')
      self.critic = CardMLP(cfg['state_dim'], cfg['hidden_dim'], 'tanh')
      self.draw_actor.load_state_dict(ckpt['draw_actor_state'])
      self.discard_actor.load_state_dict(ckpt['discard_actor_state'])
      self.critic.load_state_dict(ckpt['critic_state'])
      self.metadata = ckpt.get('metadata', {})
    else:
      torch.manual_seed(seed)
      self.draw_actor = CardMLP(ACTION_DIM, hidden_dim, 'linear')
      self.discard_actor = CardMLP(ACTION_DIM, hidden_dim, 'linear')
      self.critic = CardMLP(AFTERSTATE_DIM, hidden_dim, 'tanh')

    self.actor_opt = torch.optim.Adam(
        list(self.draw_actor.parameters()) + list(self.discard_actor.parameters()),
        lr=actor_lr,
        weight_decay=1e-4,
    )
    self.critic_opt = torch.optim.Adam(
        self.critic.parameters(), lr=critic_lr, weight_decay=1e-4
    )

  def set_lrs(self, actor_lr, critic_lr):
    for group in self.actor_opt.param_groups:
      group['lr'] = actor_lr
    for group in self.critic_opt.param_groups:
      group['lr'] = critic_lr

  @torch.no_grad()
  def _pick_from_actor(self, actor, all_feats, labels=None):
    x = torch.tensor(all_feats, dtype=torch.float32)
    logits = actor(x)
    probs = torch.softmax(logits / max(self.temperature, 1e-3), dim=-1).tolist()
    if self.stochastic:
      dist = Categorical(logits=logits / max(self.temperature, 1e-3))
      chosen_idx = int(dist.sample().item())
    else:
      chosen_idx = int(torch.argmax(logits).item())

    if labels and len(labels) >= 2 and not self.stochastic:
      ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
      top, alt = ranked[0], ranked[1]
      self.last_thought = (
          f'Picked {labels[top]} ({100*probs[top]:.0f}% policy prob vs '
          f'{labels[alt]} {100*probs[alt]:.0f}%)'
      )
    return chosen_idx

  def choose_draw(self, obs, return_record=False):
    legal, all_feats = extract_all_action_features(obs)
    pre_state = all_feats[0][:AFTERSTATE_DIM]
    if len(legal) == 1:
      rec = {
          'phase': 'DRAW',
          'state_feats': pre_state,
          'all_action_feats': all_feats,
          'chosen_idx': 0,
      }
      return (legal[0], rec) if return_record else legal[0]

    # Always spot an immediate winning Shown card (matches TDValue & QLearning).
    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]
    for i in range(6):
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        s_idx = legal.index('S')
        rec = {
            'phase': 'DRAW',
            'state_feats': pre_state,
            'all_action_feats': all_feats,
            'chosen_idx': s_idx,
        }
        return ('S', rec) if return_record else 'S'

    draw_labels = ['Shown' if a == 'S' else 'Hidden' for a in legal]
    chosen_idx = self._pick_from_actor(
        self.draw_actor, all_feats, labels=draw_labels
    )
    rec = {
        'phase': 'DRAW',
        'state_feats': pre_state,
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
    }
    return (legal[chosen_idx], rec) if return_record else legal[chosen_idx]

  def choose_discard(self, obs, return_record=False):
    hand7 = obs['hand_ids']
    legal_ids, all_feats = extract_all_action_features(obs)
    pre_state = _pre_action_state_feats(all_feats)

    # Always spot an immediate winning discard (matches TDValue & QLearning).
    for idx, cid in enumerate(legal_ids):
      h_idx = hand7.index(cid)
      if is_winning_hand_ids(hand7[:h_idx] + hand7[h_idx + 1 :]):
        card = id_to_card(cid)
        rec = {
            'phase': 'DISCARD',
            'state_feats': pre_state,
            'after_feats': all_feats[idx][:AFTERSTATE_DIM],
            'all_action_feats': all_feats,
            'chosen_idx': idx,
        }
        return (card, rec) if return_record else card

    disc_labels = [card_label(id_to_card(cid)) for cid in legal_ids]
    chosen_idx = self._pick_from_actor(
        self.discard_actor, all_feats, labels=disc_labels
    )
    card = id_to_card(legal_ids[chosen_idx])
    rec = {
        'phase': 'DISCARD',
        'state_feats': pre_state,
        'after_feats': all_feats[chosen_idx][:AFTERSTATE_DIM],
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
    }
    return (card, rec) if return_record else card

  def train_critic_only(self, state_feats_list, reward, critic_lr=None):
    """Updates Critic V(s) on opponent afterstates for a balanced baseline."""
    if not state_feats_list:
      return 0.0
    if critic_lr is not None:
      for group in self.critic_opt.param_groups:
        group['lr'] = critic_lr

    x = torch.tensor(state_feats_list, dtype=torch.float32)
    preds = self.critic(x)
    with torch.no_grad():
      n = len(state_feats_list)
      v_vals = preds.tolist()
      targets = [0.0] * n
      targets[-1] = float(reward)
      lam = 0.65
      for t in range(n - 2, -1, -1):
        targets[t] = (1.0 - lam) * (self.gamma * v_vals[t + 1]) + lam * (
            self.gamma * targets[t + 1]
        )
      target_t = torch.tensor(targets, dtype=torch.float32)

    loss = F.mse_loss(preds, target_t)
    self.critic_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
    self.critic_opt.step()
    return float(loss.item())

  def learn_episode(self, steps, reward, actor_lr=None, critic_lr=None):
    """PyTorch Advantage Actor-Critic (A2C) update over one player's episode."""
    if not steps:
      return 0.0
    if actor_lr is not None and critic_lr is not None:
      self.set_lrs(actor_lr, critic_lr)

    state_x = torch.tensor([s['state_feats'] for s in steps], dtype=torch.float32)
    v_preds = self.critic(state_x)

    # Build TD(lambda) targets and low-variance TD advantages.
    with torch.no_grad():
      n = len(steps)
      v_list = v_preds.tolist()
      targets = [0.0] * n
      targets[-1] = float(reward)
      lam = 0.65
      for idx in range(n - 2, -1, -1):
        step_gamma = 0.99 if steps[idx]['phase'] == 'DRAW' else self.gamma
        targets[idx] = (1.0 - lam) * (step_gamma * v_list[idx + 1]) + lam * (
            step_gamma * targets[idx + 1]
        )
      target_t = torch.tensor(targets, dtype=torch.float32)
      advantages = target_t - v_preds.detach()

    # 1. Update Critic toward TD(lambda) targets.
    critic_loss = F.mse_loss(v_preds, target_t)
    self.critic_opt.zero_grad()
    critic_loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), 2.0)
    self.critic_opt.step()

    # 2. Update Actor using PyTorch Categorical log_prob + entropy bonus!
    actor_losses = []
    temp = max(self.temperature, 1e-3)
    for idx, step in enumerate(steps):
      all_feats = step['all_action_feats']
      if len(all_feats) <= 1:
        continue
      actor = (
          self.draw_actor if step['phase'] == 'DRAW' else self.discard_actor
      )
      logits = actor(torch.tensor(all_feats, dtype=torch.float32)) / temp
      dist = Categorical(logits=logits)
      chosen = torch.tensor(step['chosen_idx'], dtype=torch.long)
      step_loss = -(dist.log_prob(chosen) * advantages[idx]) - (
          self.entropy_coef * dist.entropy()
      )
      actor_losses.append(step_loss)

    if actor_losses:
      actor_loss = torch.stack(actor_losses).mean()
      self.actor_opt.zero_grad()
      actor_loss.backward()
      nn.utils.clip_grad_norm_(
          list(self.draw_actor.parameters())
          + list(self.discard_actor.parameters()),
          2.0,
      )
      self.actor_opt.step()

    return float(critic_loss.item())

  def clone_frozen(self, label=None):
    """Returns a frozen greedy copy of this bot for Hall-of-Fame league sparring."""
    snap = PolicyGradientBot(hidden_dim=self.critic.hidden_dim)
    snap.draw_actor.load_state_dict(self.draw_actor.state_dict())
    snap.discard_actor.load_state_dict(self.discard_actor.state_dict())
    snap.critic.load_state_dict(self.critic.state_dict())
    snap.stochastic = False
    if label:
      snap.name = label
    return snap

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    save_checkpoint(
        path,
        {
            'config': {
                'action_dim': self.draw_actor.input_dim,
                'state_dim': self.critic.input_dim,
                'hidden_dim': self.critic.hidden_dim,
            },
            'draw_actor_state': self.draw_actor.state_dict(),
            'discard_actor_state': self.discard_actor.state_dict(),
            'critic_state': self.critic.state_dict(),
            'metadata': self.metadata,
        },
    )
