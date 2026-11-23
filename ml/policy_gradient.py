"""Model 3: PyTorch PPO (Proximal Policy Optimization) RL Player.

Upgrades vanilla single-episode A2C to batched PyTorch PPO so Policy Gradient
climbs smoothly across thousands of games without oscillating:
  1. Rollout Batching: collects a batch of ~32 full games (~600 decisions)
     before updating, slashing single-deal card-luck variance.
  2. GAE (Generalized Advantage Estimation) + Advantage Normalization across
     the entire rollout batch.
  3. Clipped Surrogate Objective: prevents any noisy batch from pushing action
     probabilities too far in one step:
       L_clip = min(ratio * A, clip(ratio, 1 - eps, 1 + eps) * A)
  4. Vectorized Masked Categorical over legal actions (2 for DRAW, 6-7 for DISCARD).
"""

import os

import torch
from torch.distributions import Categorical
import torch.nn as nn
import torch.nn.functional as F

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import card_label, id_to_card
from ml.features import (
    ACTION_DIM,
    AFTERSTATE_DIM,
    extract_all_action_features,
    extract_state_features,
)
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


def _pre_action_state_feats(all_feats):
  """Returns an action-independent summary of the current state for the Critic.

  CRITICAL RL RULE: The Critic baseline V(s_t) must depend ONLY on the pre-action
  state s_t — NEVER on the chosen action a_t! Using the post-action features of
  a_t biases the advantage A_t = G_t - V(s_t, a_t) and can invert learning.
  """
  best = all_feats[0]
  best_key = (best[1], best[2], best[17], best[16], best[4], best[5])
  for f in all_feats[1:]:
    key = (f[1], f[2], f[17], f[16], f[4], f[5])
    if key > best_key:
      best_key = key
      best = f
  return best[:AFTERSTATE_DIM]


class PolicyGradientBot(BotPolicy):
  """PyTorch RL Bot #3: Batched PPO (Proximal Policy Optimization)."""

  name = 'RL Bot — PPO Policy Gradient (PyTorch)'
  player_type = 'bot:rl_policy_gradient'

  def __init__(
      self,
      model_path=None,
      hidden_dim=64,
      gamma=0.95,
      gae_lam=0.90,
      clip_eps=0.20,
      entropy_coef=0.003,
      actor_lr=0.001,
      critic_lr=0.0012,
      seed=303,
  ):
    self.gamma = gamma
    self.gae_lam = gae_lam
    self.clip_eps = clip_eps
    self.entropy_coef = entropy_coef
    self.stochastic = False
    self.temperature = 1.0
    self.metadata = {}

    loaded = False
    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt.get('config', {})
      if (
          cfg.get('action_dim') == ACTION_DIM
          and cfg.get('state_dim') == AFTERSTATE_DIM
          and 'draw_actor_state' in ckpt
      ):
        self.draw_actor = CardMLP(cfg['action_dim'], cfg['hidden_dim'], 'linear')
        self.discard_actor = CardMLP(cfg['action_dim'], cfg['hidden_dim'], 'linear')
        self.critic = CardMLP(cfg['state_dim'], cfg['hidden_dim'], 'tanh')
        self.draw_actor.load_state_dict(ckpt['draw_actor_state'])
        self.discard_actor.load_state_dict(ckpt['discard_actor_state'])
        self.critic.load_state_dict(ckpt['critic_state'])
        self.metadata = ckpt.get('metadata', {})
        loaded = True

    if not loaded:
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
    temp = max(self.temperature, 1e-3)
    x = torch.tensor(all_feats, dtype=torch.float32)
    logits = actor(x) / temp
    dist = Categorical(logits=logits)
    if self.stochastic:
      chosen_t = dist.sample()
      chosen_idx = int(chosen_t.item())
    else:
      chosen_idx = int(torch.argmax(logits).item())
      chosen_t = torch.tensor(chosen_idx, dtype=torch.long)

    old_log_prob = float(dist.log_prob(chosen_t).item())
    if labels and len(labels) >= 2 and not self.stochastic:
      probs = dist.probs.tolist()
      ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
      top, alt = ranked[0], ranked[1]
      self.last_thought = (
          f'Picked {labels[top]} ({100*probs[top]:.0f}% policy prob vs '
          f'{labels[alt]} {100*probs[alt]:.0f}%)'
      )
    return chosen_idx, old_log_prob

  def choose_draw(self, obs, return_record=False):
    legal, all_feats = extract_all_action_features(obs)
    pre_state = extract_state_features(obs)
    if len(legal) == 1:
      rec = {
          'phase': 'DRAW',
          'state_feats': pre_state,
          'all_action_feats': all_feats,
          'chosen_idx': 0,
          'old_log_prob': 0.0,
          'is_forced': True,
      }
      return (legal[0], rec) if return_record else legal[0]

    # Always spot an immediate winning Shown card (matches TDValue & Double DQN).
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
            'old_log_prob': 0.0,
            'is_forced': True,
        }
        return ('S', rec) if return_record else 'S'

    draw_labels = ['Shown' if a == 'S' else 'Hidden' for a in legal]
    chosen_idx, old_log_prob = self._pick_from_actor(
        self.draw_actor, all_feats, labels=draw_labels
    )
    rec = {
        'phase': 'DRAW',
        'state_feats': pre_state,
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
        'old_log_prob': old_log_prob,
        'is_forced': False,
    }
    return (legal[chosen_idx], rec) if return_record else legal[chosen_idx]

  def choose_discard(self, obs, return_record=False):
    hand7 = obs['hand_ids']
    legal_ids, all_feats = extract_all_action_features(obs)
    pre_state = _pre_action_state_feats(all_feats)

    # Always spot an immediate winning discard (matches TDValue & Double DQN).
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
            'old_log_prob': 0.0,
            'is_forced': True,
        }
        return (card, rec) if return_record else card

    disc_labels = [card_label(id_to_card(cid)) for cid in legal_ids]
    chosen_idx, old_log_prob = self._pick_from_actor(
        self.discard_actor, all_feats, labels=disc_labels
    )
    card = id_to_card(legal_ids[chosen_idx])
    rec = {
        'phase': 'DISCARD',
        'state_feats': pre_state,
        'after_feats': all_feats[chosen_idx][:AFTERSTATE_DIM],
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
        'old_log_prob': old_log_prob,
        'is_forced': False,
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
      lam = 0.70
      for t in range(n - 2, -1, -1):
        targets[t] = (1.0 - lam) * (self.gamma * v_vals[t + 1]) + lam * (
            self.gamma * targets[t + 1]
        )
      target_t = torch.tensor(targets, dtype=torch.float32)

    loss = F.smooth_l1_loss(preds, target_t)
    self.critic_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
    self.critic_opt.step()
    return float(loss.item())

  def _pack_phase_batch(self, items, max_actions):
    """Pads variable-length legal action lists into [B, max_actions, ACTION_DIM]."""
    zero_row = [0.0] * ACTION_DIM
    feats_batch, mask_batch, chosen_batch, old_lp_batch, adv_batch = [], [], [], [], []
    for step, adv in items:
      af = step['all_action_feats']
      k = len(af)
      padded = af + [zero_row] * (max_actions - k)
      mask = [True] * k + [False] * (max_actions - k)
      feats_batch.append(padded)
      mask_batch.append(mask)
      chosen_batch.append(step['chosen_idx'])
      old_lp_batch.append(step.get('old_log_prob', 0.0))
      adv_batch.append(adv)
    return (
        torch.tensor(feats_batch, dtype=torch.float32),
        torch.tensor(mask_batch, dtype=torch.bool),
        torch.tensor(chosen_batch, dtype=torch.long),
        torch.tensor(old_lp_batch, dtype=torch.float32),
        torch.tensor(adv_batch, dtype=torch.float32),
    )

  def learn_ppo_batch(
      self,
      episodes,
      ppo_epochs=4,
      actor_lr=None,
      critic_lr=None,
  ):
    """Vectorized PyTorch PPO update across a rollout batch of full episodes.

    episodes: list of (steps, reward) tuples collected under the current policy.
    """
    if not episodes:
      return 0.0
    if actor_lr is not None and critic_lr is not None:
      self.set_lrs(actor_lr, critic_lr)

    # 1. Compute GAE returns and advantages across all episodes in the rollout!
    all_states = []
    all_returns = []
    draw_items = []
    disc_items = []
    raw_advs = []

    with torch.no_grad():
      for steps, reward in episodes:
        if not steps:
          continue
        sx = torch.tensor([s['state_feats'] for s in steps], dtype=torch.float32)
        v_list = self.critic(sx).tolist()
        n = len(steps)
        gae = 0.0
        ep_advs = [0.0] * n
        ep_rets = [0.0] * n
        for t in range(n - 1, -1, -1):
          step_gamma = 0.99 if steps[t]['phase'] == 'DRAW' else self.gamma
          if t == n - 1:
            delta = float(reward) - v_list[t]
            gae = delta
          else:
            delta = step_gamma * v_list[t + 1] - v_list[t]
            gae = delta + step_gamma * self.gae_lam * gae
          ep_advs[t] = gae
          ep_rets[t] = max(-1.0, min(1.0, gae + v_list[t]))

        for t, step in enumerate(steps):
          all_states.append(step['state_feats'])
          all_returns.append(ep_rets[t])
          if step.get('is_forced') or len(step['all_action_feats']) <= 1:
            continue
          raw_advs.append(ep_advs[t])
          if step['phase'] == 'DRAW':
            draw_items.append((step, ep_advs[t]))
          else:
            disc_items.append((step, ep_advs[t]))

    if not all_states:
      return 0.0

    # Normalize advantages across the full rollout batch (crucial PPO trick!).
    if len(raw_advs) > 1:
      adv_t = torch.tensor(raw_advs, dtype=torch.float32)
      adv_mean = float(adv_t.mean().item())
      adv_std = float(adv_t.std().item()) + 1e-8
      draw_items = [(s, (a - adv_mean) / adv_std) for s, a in draw_items]
      disc_items = [(s, (a - adv_mean) / adv_std) for s, a in disc_items]

    state_x = torch.tensor(all_states, dtype=torch.float32)
    return_t = torch.tensor(all_returns, dtype=torch.float32)

    draw_pack = self._pack_phase_batch(draw_items, 2) if draw_items else None
    disc_pack = self._pack_phase_batch(disc_items, 7) if disc_items else None
    temp = max(self.temperature, 1e-3)
    last_critic_loss = 0.0

    # 2. Run multiple clipped PPO epochs over the collected rollout batch!
    for _ in range(ppo_epochs):
      v_preds = self.critic(state_x)
      critic_loss = F.smooth_l1_loss(v_preds, return_t)
      self.critic_opt.zero_grad()
      critic_loss.backward()
      nn.utils.clip_grad_norm_(self.critic.parameters(), 1.0)
      self.critic_opt.step()
      last_critic_loss = float(critic_loss.item())

      actor_loss_parts = []
      for actor, pack in (
          (self.draw_actor, draw_pack),
          (self.discard_actor, disc_pack),
      ):
        if pack is None:
          continue
        f_b, m_b, c_b, old_lp_b, adv_b = pack
        logits = actor(f_b) / temp
        logits = logits.masked_fill(~m_b, -1e9)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(c_b)
        ratio = torch.exp((new_lp - old_lp_b).clamp(-10.0, 10.0))
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
        p_loss = -torch.min(surr1, surr2).mean() - (
            self.entropy_coef * dist.entropy().mean()
        )
        actor_loss_parts.append(p_loss)

      if actor_loss_parts:
        actor_loss = sum(actor_loss_parts) / len(actor_loss_parts)
        self.actor_opt.zero_grad()
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.draw_actor.parameters())
            + list(self.discard_actor.parameters()),
            1.0,
        )
        self.actor_opt.step()

    return last_critic_loss

  def learn_episode(self, steps, reward, actor_lr=None, critic_lr=None):
    """Single-episode convenience wrapper (used by unit tests & quick demos)."""
    return self.learn_ppo_batch(
        [(steps, reward)],
        ppo_epochs=1,
        actor_lr=actor_lr,
        critic_lr=critic_lr,
    )

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
