"""Model 7: PyTorch RL Ensemble Bot (Soft Probability Averaging or Mode Voting).

Combines multiple independently trained PyTorch RL models (Double DQN, TD Value,
X-Ray GRPO, PPO, Rollout Distilled) into one consensus decision maker:
  - Shared feature extraction: computes the 28-dim action features ONCE per turn
    and feeds them to every ensemble member.
  - mode='soft': averages calibrated action probabilities across all members.
  - mode='vote': majority vote on DRAW, mode (most-voted card) on DISCARD.
"""

from collections import Counter

import torch

from bots.base import BotPolicy
from bots.heuristic import is_winning_hand_ids
from game.rules import card_label, id_to_card
from ml.features import AFTERSTATE_DIM, extract_all_action_features


def _member_probs(bot, obs, legal, all_feats, after_feats=None):
  """Returns calibrated action probabilities for one RL ensemble member."""
  phase = obs['phase']
  pt = bot.player_type
  if pt in ('bot:rl_policy_gradient', 'bot:rl_grpo_xray', 'bot:rl_distilled'):
    if phase == 'DRAW':
      net = bot.draw_net if pt == 'bot:rl_distilled' else bot.draw_actor
    else:
      net = bot.discard_net if pt == 'bot:rl_distilled' else bot.discard_actor
    with torch.no_grad():
      logits = net(torch.tensor(all_feats, dtype=torch.float32))
      return torch.softmax(logits, dim=-1).tolist()

  if pt == 'bot:rl_q_learning':
    net = bot.draw_net if phase == 'DRAW' else bot.discard_net
    q_vals = net.predict_batch(all_feats)
    return torch.softmax(
        torch.tensor(q_vals, dtype=torch.float32) / 0.12, dim=-1
    ).tolist()

  if pt == 'bot:rl_td_value':
    if phase == 'DRAW':
      act = bot.choose_draw(obs)
      return [0.85 if a == act else 0.15 for a in legal]
    vals = bot.net.predict_batch(after_feats)
    return torch.softmax(
        torch.tensor(vals, dtype=torch.float32) / 0.06, dim=-1
    ).tolist()

  raise ValueError(f'Unsupported ensemble member type: {pt}')


class EnsembleBot(BotPolicy):
  """Consensus ensemble over trained PyTorch RL bots ('soft' prob avg or 'vote' mode)."""

  def __init__(self, members, mode='soft', name=None, player_type=None):
    if mode not in ('soft', 'vote'):
      raise ValueError(f'Unsupported ensemble mode: {mode}')
    self.members = list(members)
    self.mode = mode
    self.name = name or (
        'RL Ensemble — Soft Prob (Top-4)'
        if mode == 'soft'
        else 'RL Ensemble — Mode Vote (Top-5)'
    )
    self.player_type = player_type or (
        'bot:rl_ensemble_soft' if mode == 'soft' else 'bot:rl_ensemble_mode'
    )

  def _aggregate(self, obs, legal, all_feats, after_feats=None):
    """Returns (chosen_idx, avg_probs, vote_counts_by_idx)."""
    n = len(legal)
    avg_probs = [0.0] * n
    votes = []
    for bot in self.members:
      probs = _member_probs(bot, obs, legal, all_feats, after_feats)
      for i, p in enumerate(probs):
        avg_probs[i] += p / len(self.members)
      votes.append(max(range(n), key=lambda i: probs[i]))

    counts = Counter(votes)
    if self.mode == 'soft':
      chosen_idx = max(range(n), key=lambda i: avg_probs[i])
    else:
      max_c = max(counts.values())
      # Deterministic tie-break: earliest member's vote among tied modes.
      chosen_idx = next(v for v in votes if counts[v] == max_c)
    return chosen_idx, avg_probs, counts

  def choose_draw(self, obs):
    legal = obs['legal_actions']
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]
    for i in range(6):
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        self.last_thought = 'Took Shown — completes a winning hand!'
        return 'S'

    _, all_feats = extract_all_action_features(obs)
    idx, avg_probs, counts = self._aggregate(obs, legal, all_feats)
    chosen = legal[idx]
    lbl = 'Shown' if chosen == 'S' else 'Hidden'
    self.last_thought = (
        f'Chose {lbl} ({counts[idx]}/{len(self.members)} votes, '
        f'{100*avg_probs[idx]:.0f}% consensus)'
    )
    return chosen

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids, all_feats = extract_all_action_features(obs)
    for cid in legal_ids:
      i = hand7.index(cid)
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        card = id_to_card(cid)
        self.last_thought = f'Put down {card_label(card)} — winning hand!'
        return card

    after_feats = [f[:AFTERSTATE_DIM] for f in all_feats]
    idx, avg_probs, counts = self._aggregate(obs, legal_ids, all_feats, after_feats)
    card = id_to_card(legal_ids[idx])
    self.last_thought = (
        f'Put down {card_label(card)} ({counts[idx]}/{len(self.members)} votes, '
        f'{100*avg_probs[idx]:.0f}% consensus)'
    )
    return card

  def score_discard_options(self, obs):
    """Returns [(card_id, avg_prob, vote_count), ...] for the Bot Brain Sandbox."""
    legal_ids, all_feats = extract_all_action_features(obs)
    after_feats = [f[:AFTERSTATE_DIM] for f in all_feats]
    _, avg_probs, counts = self._aggregate(obs, legal_ids, all_feats, after_feats)
    return [
        (cid, avg_probs[i], counts[i])
        for i, cid in enumerate(legal_ids)
    ]
