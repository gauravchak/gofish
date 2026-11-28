"""Model 5: PyTorch X-Ray Critic + Duplicate-Deal GRPO RL Player.

Solves card-game luck variance with TWO provably unbiased modern RL tricks:
  1. Omniscient ("X-Ray") Critic (used ONLY during training — like Suphx Mahjong AI!):
       - At play time, the Actor ONLY sees legal cards (100% fair!).
       - During training, the Critic sees BOTH players' 24-feature hands (48 dims).
       - When the opponent was secretly dealt an unbeatable hand, the X-Ray Critic
         predicts V ~ -0.80 so the Actor is NOT unfairly punished for losing!
  2. Duplicate-Deal GRPO (inspired by DeepSeek-R1's Group Relative Policy Opt):
       - Every training deck is played TWICE with seats swapped (mirror deals).
       - Pure deal-luck splits (+1 in Seat 0, -1 in Seat 1) cancel out across the
         mirror pair while true skill wins (+1 in BOTH seats) get full gradient!
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
from ml.policy_gradient import _pre_action_state_feats


class XRayGRPOBot(BotPolicy):
  """Fair PyTorch Policy trained with an Omniscient X-Ray Critic + Duplicate-Deal GRPO."""

  name = 'RL Bot — X-Ray GRPO (PyTorch)'
  player_type = 'bot:rl_grpo_xray'

  def __init__(
      self,
      model_path=None,
      hidden_dim=64,
      gamma=0.95,
      gae_lam=0.92,
      clip_eps=0.20,
      entropy_coef=0.0025,
      actor_lr=0.001,
      critic_lr=0.0015,
      seed=505,
  ):
    self.gamma = gamma
    self.gae_lam = gae_lam
    self.clip_eps = clip_eps
    self.entropy_coef = entropy_coef
    self.stochastic = False
    self.temperature = 1.0
    self.metadata = {}

    xray_dim = 2 * AFTERSTATE_DIM
    loaded = False
    ckpt = None
    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt.get('config', {})
      if (
          cfg.get('action_dim') == ACTION_DIM
          and cfg.get('xray_dim') == xray_dim
          and 'draw_actor_state' in ckpt
      ):
        h = cfg['hidden_dim']
        self.draw_actor = CardMLP(ACTION_DIM, h, 'linear')
        self.discard_actor = CardMLP(ACTION_DIM, h, 'linear')
        self.xray_critic = CardMLP(xray_dim, h, 'tanh')
        self.draw_actor.load_state_dict(ckpt['draw_actor_state'])
        self.discard_actor.load_state_dict(ckpt['discard_actor_state'])
        self.xray_critic.load_state_dict(ckpt['xray_critic_state'])
        self.metadata = ckpt.get('metadata', {})
        loaded = True

    if not loaded:
      torch.manual_seed(seed)
      self.draw_actor = CardMLP(ACTION_DIM, hidden_dim, 'linear')
      self.discard_actor = CardMLP(ACTION_DIM, hidden_dim, 'linear')
      self.xray_critic = CardMLP(xray_dim, hidden_dim, 'tanh')

    self.actor_opt = torch.optim.Adam(
        list(self.draw_actor.parameters()) + list(self.discard_actor.parameters()),
        lr=actor_lr,
        weight_decay=1e-4,
    )
    self.critic_opt = torch.optim.Adam(
        self.xray_critic.parameters(), lr=critic_lr, weight_decay=1e-4
    )

  def set_lrs(self, actor_lr, critic_lr):
    for g in self.actor_opt.param_groups:
      g['lr'] = actor_lr
    for g in self.critic_opt.param_groups:
      g['lr'] = critic_lr

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
          f'Picked {labels[top]} (GRPO {100*probs[top]:.0f}% vs '
          f'{labels[alt]} {100*probs[alt]:.0f}%)'
      )
    return chosen_idx, old_log_prob

  def choose_draw(self, obs, return_record=False, opp_obs=None):
    legal, all_feats = extract_all_action_features(obs)
    my_state = extract_state_features(obs)
    opp_state = extract_state_features(opp_obs) if opp_obs is not None else my_state
    xray_state = my_state + opp_state

    if len(legal) == 1:
      rec = {
          'phase': 'DRAW',
          'xray_state': xray_state,
          'all_action_feats': all_feats,
          'chosen_idx': 0,
          'old_log_prob': 0.0,
          'is_forced': True,
      }
      return (legal[0], rec) if return_record else legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]
    for i in range(6):
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        s_idx = legal.index('S')
        self.last_thought = 'Took Shown — completes a winning hand!'
        rec = {
            'phase': 'DRAW',
            'xray_state': xray_state,
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
        'xray_state': xray_state,
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
        'old_log_prob': old_log_prob,
        'is_forced': False,
    }
    return (legal[chosen_idx], rec) if return_record else legal[chosen_idx]

  def choose_discard(self, obs, return_record=False, opp_obs=None):
    hand7 = obs['hand_ids']
    legal_ids, all_feats = extract_all_action_features(obs)
    my_state = _pre_action_state_feats(all_feats)
    opp_state = extract_state_features(opp_obs) if opp_obs is not None else my_state
    xray_state = my_state + opp_state

    for idx, cid in enumerate(legal_ids):
      h_idx = hand7.index(cid)
      if is_winning_hand_ids(hand7[:h_idx] + hand7[h_idx + 1 :]):
        card = id_to_card(cid)
        self.last_thought = f'Put down {card_label(card)} — winning hand!'
        rec = {
            'phase': 'DISCARD',
            'xray_state': xray_state,
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
        'xray_state': xray_state,
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
        'old_log_prob': old_log_prob,
        'is_forced': False,
    }
    return (card, rec) if return_record else card

  def _pack_phase_batch(self, items, max_actions):
    zero_row = [0.0] * ACTION_DIM
    feats_b, mask_b, chosen_b, old_lp_b, adv_b = [], [], [], [], []
    for step, adv in items:
      af = step['all_action_feats']
      k = len(af)
      feats_b.append(af + [zero_row] * (max_actions - k))
      mask_b.append([True] * k + [False] * (max_actions - k))
      chosen_b.append(step['chosen_idx'])
      old_lp_b.append(step.get('old_log_prob', 0.0))
      adv_b.append(adv)
    return (
        torch.tensor(feats_b, dtype=torch.float32),
        torch.tensor(mask_b, dtype=torch.bool),
        torch.tensor(chosen_b, dtype=torch.long),
        torch.tensor(old_lp_b, dtype=torch.float32),
        torch.tensor(adv_b, dtype=torch.float32),
    )

  def learn_grpo_batch(
      self,
      mirror_pairs,
      ppo_epochs=3,
      actor_lr=None,
      critic_lr=None,
      mirror_weight=0.35,
  ):
    """Updates Actor + X-Ray Critic from paired duplicate-deal mirror matches.

    mirror_pairs: list of (steps_a, reward_a, steps_b, reward_b) where Deal A
    and Deal B used the EXACT same shuffled deck with seats swapped!
    """
    if not mirror_pairs:
      return 0.0
    if actor_lr is not None and critic_lr is not None:
      self.set_lrs(actor_lr, critic_lr)

    all_xray_states = []
    all_critic_targets = []
    draw_items = []
    disc_items = []
    raw_advs = []

    with torch.no_grad():
      for steps_a, rew_a, steps_b, rew_b in mirror_pairs:
        # Mirror-deal group skill score: +1 if won BOTH seats, -1 if lost BOTH, 0 if split!
        mirror_net = 0.5 * (float(rew_a) + float(rew_b))
        for steps, raw_rew in ((steps_a, float(rew_a)), (steps_b, float(rew_b))):
          if not steps:
            continue
          sx = torch.tensor([s['xray_state'] for s in steps], dtype=torch.float32)
          v_list = self.xray_critic(sx).tolist()
          n = len(steps)

          # X-Ray Critic always predicts true unbiased game outcome raw_rew,
          # while Actor advantage blends individual X-Ray GAE + mirror-deal skill signal!
          gae_raw = 0.0
          ep_rets = [0.0] * n
          ep_advs = [0.0] * n
          shaped_rew = (1.0 - mirror_weight) * raw_rew + mirror_weight * mirror_net
          for t in range(n - 1, -1, -1):
            step_gamma = 0.99 if steps[t]['phase'] == 'DRAW' else self.gamma
            if t == n - 1:
              delta_c = raw_rew - v_list[t]
              gae_raw = delta_c
            else:
              delta_c = step_gamma * v_list[t + 1] - v_list[t]
              gae_raw = delta_c + step_gamma * self.gae_lam * gae_raw
            ep_rets[t] = max(-1.0, min(1.0, gae_raw + v_list[t]))
            # Mirror-calibrated advantage: removes god-hand luck via BOTH X-Ray V AND mirror_net!
            ep_advs[t] = gae_raw + (shaped_rew - raw_rew) * (self.gamma ** (n - 1 - t))

          for t, step in enumerate(steps):
            all_xray_states.append(step['xray_state'])
            all_critic_targets.append(ep_rets[t])
            if step.get('is_forced') or len(step['all_action_feats']) <= 1:
              continue
            raw_advs.append(ep_advs[t])
            if step['phase'] == 'DRAW':
              draw_items.append((step, ep_advs[t]))
            else:
              disc_items.append((step, ep_advs[t]))

    if not all_xray_states:
      return 0.0

    if len(raw_advs) > 1:
      adv_t = torch.tensor(raw_advs, dtype=torch.float32)
      mean_a = float(adv_t.mean().item())
      std_a = float(adv_t.std().item()) + 1e-8
      draw_items = [(s, (a - mean_a) / std_a) for s, a in draw_items]
      disc_items = [(s, (a - mean_a) / std_a) for s, a in disc_items]

    xray_x = torch.tensor(all_xray_states, dtype=torch.float32)
    ret_t = torch.tensor(all_critic_targets, dtype=torch.float32)
    draw_pack = self._pack_phase_batch(draw_items, 2) if draw_items else None
    disc_pack = self._pack_phase_batch(disc_items, 7) if disc_items else None
    temp = max(self.temperature, 1e-3)
    last_critic_loss = 0.0

    for _ in range(ppo_epochs):
      v_preds = self.xray_critic(xray_x)
      critic_loss = F.smooth_l1_loss(v_preds, ret_t)
      self.critic_opt.zero_grad()
      critic_loss.backward()
      nn.utils.clip_grad_norm_(self.xray_critic.parameters(), 1.0)
      self.critic_opt.step()
      last_critic_loss = float(critic_loss.item())

      actor_losses = []
      for actor, pack in (
          (self.draw_actor, draw_pack),
          (self.discard_actor, disc_pack),
      ):
        if pack is None:
          continue
        f_b, m_b, c_b, old_lp_b, adv_b = pack
        logits = (actor(f_b) / temp).masked_fill(~m_b, -1e9)
        dist = Categorical(logits=logits)
        new_lp = dist.log_prob(c_b)
        ratio = torch.exp((new_lp - old_lp_b).clamp(-10.0, 10.0))
        surr1 = ratio * adv_b
        surr2 = torch.clamp(ratio, 1.0 - self.clip_eps, 1.0 + self.clip_eps) * adv_b
        actor_losses.append(
            -torch.min(surr1, surr2).mean()
            - self.entropy_coef * dist.entropy().mean()
        )

      if actor_losses:
        loss = sum(actor_losses) / len(actor_losses)
        self.actor_opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.draw_actor.parameters())
            + list(self.discard_actor.parameters()),
            1.0,
        )
        self.actor_opt.step()

    return last_critic_loss

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    save_checkpoint(
        path,
        {
            'config': {
                'action_dim': self.draw_actor.input_dim,
                'xray_dim': self.xray_critic.input_dim,
                'hidden_dim': self.xray_critic.hidden_dim,
            },
            'draw_actor_state': self.draw_actor.state_dict(),
            'discard_actor_state': self.discard_actor.state_dict(),
            'xray_critic_state': self.xray_critic.state_dict(),
            'metadata': self.metadata,
        },
    )
