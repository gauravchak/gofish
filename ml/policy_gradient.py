"""Model 3: Actor-Critic Policy Gradient RL Player (REINFORCE with Value Baseline).

Core RL idea:
  - Actor (Policy pi(a|s)): scores every legal action with a logit z(s, a) and
    turns scores into probabilities via masked softmax.
  - Critic (Value baseline V(s)): estimates expected return from state s.
  - Advantage: A_t = G_t - V(s_t) ('did this move turn out better or worse than
    the Critic expected?').
      * If A_t > 0, nudge the chosen action's logit UP (and others slightly down).
      * If A_t < 0, nudge the chosen action's logit DOWN.
  - Plus a tiny entropy bonus during training so the policy keeps exploring!
"""

import json
import math
import os
import random

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import id_to_card
from ml.features import (
    ACTION_DIM,
    AFTERSTATE_DIM,
    extract_all_action_features,
    extract_state_features,
)
from ml.nn import TinyMLP


def _masked_softmax(logits, temperature=1.0):
  temp = max(temperature, 1e-3)
  scaled = [z / temp for z in logits]
  m = max(scaled)
  exps = [math.exp(max(min(v - m, 30.0), -30.0)) for v in scaled]
  total = sum(exps)
  return [e / total for e in exps]


def _sample_categorical(probs, rng):
  r = rng.random()
  cumulative = 0.0
  for idx, p in enumerate(probs):
    cumulative += p
    if r <= cumulative:
      return idx
  return len(probs) - 1


def _pre_action_state_feats(all_feats):
  """Returns an action-independent summary of the current state for the Critic.

  CRITICAL RL RULE: The Critic baseline V(s_t) must depend ONLY on the pre-action
  state s_t — NEVER on the chosen action a_t! Using the post-action features of
  a_t biases the advantage A_t = G_t - V(s_t, a_t) and can invert learning!
  """
  # Pick the strongest achievable legal afterstate as the canonical pre-action summary.
  best = all_feats[0]
  best_key = (best[1], best[2], best[4], best[5])
  for f in all_feats[1:]:
    key = (f[1], f[2], f[4], f[5])
    if key > best_key:
      best_key = key
      best = f
  return best[:AFTERSTATE_DIM]


class PolicyGradientBot(BotPolicy):
  """RL Bot #3: Actor-Critic Policy Gradient (separate DRAW & DISCARD actors + Critic)."""

  name = 'RL Bot — Policy Gradient'
  player_type = 'bot:rl_policy_gradient'

  def __init__(
      self,
      model_path=None,
      hidden_dim=16,
      gamma=0.95,
      entropy_coef=0.01,
      seed=303,
  ):
    self.gamma = gamma
    self.entropy_coef = entropy_coef
    self.stochastic = False  # True during training; greedy/low-temp at play time
    self.temperature = 1.0
    self._rng = random.Random(seed)
    self.metadata = {}

    if model_path and os.path.exists(model_path):
      with open(model_path, 'r', encoding='utf-8') as f:
        payload = json.load(f)
      self.draw_actor = TinyMLP.from_dict(payload['draw_actor'])
      self.discard_actor = TinyMLP.from_dict(payload['discard_actor'])
      self.critic = TinyMLP.from_dict(payload['critic'])
      self.metadata = payload.get('metadata', {})
    else:
      self.draw_actor = TinyMLP(
          input_dim=ACTION_DIM,
          hidden_dim=hidden_dim,
          output_activation='linear',
          seed=seed,
      )
      self.discard_actor = TinyMLP(
          input_dim=ACTION_DIM,
          hidden_dim=hidden_dim,
          output_activation='linear',
          seed=seed + 1,
      )
      self.critic = TinyMLP(
          input_dim=AFTERSTATE_DIM,
          hidden_dim=hidden_dim,
          output_activation='tanh',
          seed=seed + 2,
      )

  def _pick_from_actor(self, actor, all_feats):
    logits = [actor.predict(f) for f in all_feats]
    probs = _masked_softmax(logits, temperature=self.temperature)
    if self.stochastic:
      chosen_idx = _sample_categorical(probs, self._rng)
    else:
      chosen_idx = max(range(len(logits)), key=lambda i: logits[i])
    return chosen_idx, probs

  def choose_draw(self, obs, return_record=False):
    legal, all_feats = extract_all_action_features(obs)
    if len(legal) == 1:
      rec = {
          'phase': 'DRAW',
          'state_feats': all_feats[0][:AFTERSTATE_DIM],
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
            'state_feats': all_feats[0][:AFTERSTATE_DIM],
            'all_action_feats': all_feats,
            'chosen_idx': s_idx,
        }
        return ('S', rec) if return_record else 'S'

    chosen_idx, _ = self._pick_from_actor(self.draw_actor, all_feats)
    rec = {
        'phase': 'DRAW',
        'state_feats': all_feats[0][:AFTERSTATE_DIM],
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

    chosen_idx, _ = self._pick_from_actor(self.discard_actor, all_feats)
    card = id_to_card(legal_ids[chosen_idx])
    rec = {
        'phase': 'DISCARD',
        'state_feats': pre_state,
        'after_feats': all_feats[chosen_idx][:AFTERSTATE_DIM],
        'all_action_feats': all_feats,
        'chosen_idx': chosen_idx,
    }
    return (card, rec) if return_record else card

  def train_critic_only(self, state_feats_list, reward, critic_lr=0.015, l2=1e-4):
    """Updates Critic V(s) using TD(lambda) targets (can learn from both seats!)."""
    if not state_feats_list:
      return 0.0
    n = len(state_feats_list)
    preds = [self.critic.predict(f) for f in state_feats_list]
    targets = [0.0] * n
    targets[-1] = reward
    lam = 0.65
    for t in range(n - 2, -1, -1):
      targets[t] = (1.0 - lam) * (self.gamma * preds[t + 1]) + lam * (
          self.gamma * targets[t + 1]
      )
    loss = 0.0
    for f, target in zip(state_feats_list, targets):
      loss += self.critic.train_step(f, target, lr=critic_lr, l2=l2)
    return loss / n

  def learn_episode(self, steps, reward, actor_lr=0.012, critic_lr=0.018, l2=1e-4):
    """Advantage Actor-Critic (A2C) update over one player's episode.

    Uses TD advantage A_t = (gamma * V(s_{t+1}) - V(s_t)) blended with discounted
    Monte Carlo return so the Actor gets immediate step-by-step feedback even
    when a single game's final win/loss is noisy!
    """
    if not steps:
      return 0.0

    n = len(steps)
    v_preds = [self.critic.predict(s['state_feats']) for s in steps]

    # Compute TD(lambda) return targets for Critic & low-variance TD advantages for Actor.
    targets = [0.0] * n
    targets[-1] = reward
    lam = 0.65
    for idx in range(n - 2, -1, -1):
      step_gamma = 0.99 if steps[idx]['phase'] == 'DRAW' else self.gamma
      targets[idx] = (1.0 - lam) * (step_gamma * v_preds[idx + 1]) + lam * (
          step_gamma * targets[idx + 1]
      )

    total_critic_loss = 0.0
    for idx, step in enumerate(steps):
      state_feats = step['state_feats']
      v_pred = v_preds[idx]
      # TD Advantage: how much better/worse did this action's continuation turn out
      # than the Critic expected before the action?
      advantage = targets[idx] - v_pred

      # 1. Update Critic toward TD(lambda) target.
      total_critic_loss += self.critic.train_step(
          state_feats, targets[idx], lr=critic_lr, l2=l2
      )

      # 2. Update Actor using exact masked-softmax policy gradient:
      #    dL/dz_k = A_t * (pi_k - 1[k == chosen]) / temperature
      all_feats = step['all_action_feats']
      if len(all_feats) <= 1:
        continue
      actor = (
          self.draw_actor if step['phase'] == 'DRAW' else self.discard_actor
      )
      caches = [actor.forward_with_cache(f) for f in all_feats]
      logits = [out for out, _ in caches]
      probs = _masked_softmax(logits, temperature=self.temperature)
      chosen = step['chosen_idx']

      entropy_terms = [-(math.log(max(p, 1e-9)) + 1.0) for p in probs]
      mean_ent = sum(p * e for p, e in zip(probs, entropy_terms))

      for k, ((_, h_k), f_k, p_k) in enumerate(zip(caches, all_feats, probs)):
        indicator = 1.0 if k == chosen else 0.0
        d_pg = advantage * (p_k - indicator) / max(self.temperature, 1e-3)
        d_ent = -self.entropy_coef * p_k * (entropy_terms[k] - mean_ent)
        actor.apply_output_gradient(f_k, h_k, d_pg + d_ent, lr=actor_lr, l2=l2)

    return total_critic_loss / n

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {
        'draw_actor': self.draw_actor.to_dict(),
        'discard_actor': self.discard_actor.to_dict(),
        'critic': self.critic.to_dict(),
        'metadata': self.metadata,
    }
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(payload, f, indent=2)
      f.write('\n')
