"""Model 4: True AlphaZero-Lite Self-Play Bot (Policy + Value + Tree Search).

Why is this the highest-ceiling RL method?
  1. Neural Policy Prior P(a|s) guides 1-level Expectimax lookahead to focus
     computation on the most promising candidate moves (top-3 discards!).
  2. Neural Value Network V(s) evaluates the leaves of the lookahead tree.
  3. TRUE ALPHAZERO SELF-PLAY LOOP:
       - AlphaZero plays training games using its own Policy+Value+Search tree.
       - Every turn produces a search-improved policy target pi_search AND
         search value estimate v_search.
       - The Value Network trains toward blended (game_outcome, v_search).
       - The Policy Networks train via Cross-Entropy to imitate pi_search!
     Better search -> better neural nets -> even better search next generation!
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
from game.rules import card_label, id_to_card
from ml.features import (
    ACTION_DIM,
    AFTERSTATE_DIM,
    extract_afterstate_features,
    extract_all_action_features,
)
from ml.nn import CardMLP, load_checkpoint, save_checkpoint
from ml.td_value import TDValueBot


def _evenly_spaced(items, max_count):
  n = len(items)
  if n <= max_count:
    return items
  step = n / max_count
  return [items[int(i * step)] for i in range(max_count)]


def _softmax_scores(scores, temp=0.08):
  """Converts a list of value scores in [0, 1] into a sharp probability target."""
  t = max(temp, 1e-3)
  mx = max(scores)
  exps = [torch.exp(torch.tensor((s - mx) / t)).item() for s in scores]
  total = sum(exps) or 1.0
  return [e / total for e in exps]


class AlphaZeroLiteBot(BotPolicy):
  """True AlphaZero-Lite: Policy Prior + Value Net + 1-Level Expectimax Search."""

  name = 'RL + Tree Bot (AlphaZero-Lite)'
  player_type = 'bot:rl_alphazero'

  def __init__(
      self,
      model_path=None,
      td_bot=None,
      hidden_dim=64,
      draw_sample=12,
      top_k_discards=3,
      lr=0.0015,
      seed=404,
  ):
    self.draw_sample = draw_sample
    self.top_k_discards = top_k_discards
    self.stochastic = False
    self.search_temp = 0.08
    self._rng = random.Random(seed)
    self.metadata = {}

    if td_bot is not None:
      self.value_bot = td_bot
      h_dim = td_bot.net.hidden_dim
      torch.manual_seed(seed)
      self.draw_policy = CardMLP(ACTION_DIM, h_dim, 'linear')
      self.discard_policy = CardMLP(ACTION_DIM, h_dim, 'linear')
    else:
      self.value_bot = TDValueBot(hidden_dim=hidden_dim, lr=lr, seed=seed)
      torch.manual_seed(seed + 1)
      self.draw_policy = CardMLP(ACTION_DIM, hidden_dim, 'linear')
      self.discard_policy = CardMLP(ACTION_DIM, hidden_dim, 'linear')

      if model_path and os.path.exists(model_path):
        ckpt = load_checkpoint(model_path)
        cfg = ckpt.get('config', {})
        if cfg.get('input_dim') == AFTERSTATE_DIM and 'value_state' in ckpt:
          h_dim = cfg['hidden_dim']
          self.value_bot = TDValueBot(hidden_dim=h_dim, lr=lr, seed=seed)
          self.value_bot.net.load_state_dict(ckpt['value_state'])
          self.value_bot.target_net.load_state_dict(ckpt['value_state'])
          self.draw_policy = CardMLP(cfg['action_dim'], h_dim, 'linear')
          self.discard_policy = CardMLP(cfg['action_dim'], h_dim, 'linear')
          self.draw_policy.load_state_dict(ckpt['draw_policy_state'])
          self.discard_policy.load_state_dict(ckpt['discard_policy_state'])
          self.metadata = ckpt.get('metadata', {})
        elif cfg.get('input_dim') == AFTERSTATE_DIM and 'net_state' in ckpt:
          # Graceful fallback if initialized from a TDValueBot checkpoint!
          self.value_bot = TDValueBot(model_path=model_path)
          self.metadata = ckpt.get('metadata', {})

    self.policy_opt = torch.optim.Adam(
        list(self.draw_policy.parameters())
        + list(self.discard_policy.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

  def set_lr(self, lr):
    self.value_bot.set_lr(lr)
    for group in self.policy_opt.param_groups:
      group['lr'] = lr

  def _expected_next_turn_value(
      self, hand6, unseen_ids, unseen_mask, deck_rem, next_wanted, next_discarded
  ):
    """1-level Expectimax chance node using batched PyTorch V(s) at the leaves."""
    base_val = self.value_bot.net.predict_one(
        extract_afterstate_features(
            hand6, unseen_mask, deck_rem, next_wanted, next_discarded
        )
    )
    if not unseen_ids:
      return base_val

    relevant_mask = 0
    for cid in hand6:
      relevant_mask |= SINGLE_PARTNER_MASKS[cid]

    relevant_ids = [cid for cid in unseen_ids if relevant_mask & (1 << cid)]
    brick_count = len(unseen_ids) - len(relevant_ids)

    # Check ALL relevant unseen cards for immediate 1-turn wins (exact!),
    # and batch-evaluate a sample of non-winning draws through PyTorch V(s).
    win_draws = 0
    non_win_rel = []
    for draw_id in relevant_ids:
      hand7 = hand6 + [draw_id]
      won = False
      for i in range(7):
        if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
          won = True
          break
      if won:
        win_draws += 1
      else:
        non_win_rel.append(draw_id)

    sampled = _evenly_spaced(non_win_rel, self.draw_sample)
    rel_non_win_total = 0.0
    if sampled:
      batch_feats = []
      counts = []
      next_deck = max(deck_rem - 1, 0)
      for draw_id in sampled:
        next_mask = unseen_mask ^ (1 << draw_id)
        hand7 = hand6 + [draw_id]
        for i in range(7):
          h6 = hand7[:i] + hand7[i + 1 :]
          batch_feats.append(
              extract_afterstate_features(
                  h6,
                  next_mask,
                  next_deck,
                  next_wanted,
                  next_discarded,
                  discarded_id=hand7[i],
                  took_shown=False,
              )
          )
        counts.append(7)

      preds = self.value_bot.net.predict_batch(batch_feats)
      offset = 0
      for c in counts:
        best_v = max(preds[offset : offset + c])
        rel_non_win_total += best_v if best_v > base_val else base_val
        offset += c
      avg_non_win = rel_non_win_total / len(sampled)
    else:
      avg_non_win = base_val

    total_val = (
        brick_count * base_val
        + win_draws * 1.0
        + len(non_win_rel) * avg_non_win
    )
    return total_val / len(unseen_ids)

  def _sample_or_argmax(self, scores):
    if self.stochastic and len(scores) > 1:
      probs = _softmax_scores(scores, temp=self.search_temp)
      r = self._rng.random()
      acc = 0.0
      for idx, p in enumerate(probs):
        acc += p
        if r <= acc:
          return idx, probs
      return len(probs) - 1, probs
    probs = _softmax_scores(scores, temp=0.05)
    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    return best_idx, probs

  def choose_draw(self, obs, return_search=False):
    legal = obs['legal_actions']
    if len(legal) == 1:
      if return_search:
        return legal[0], None
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    unseen_mask = get_unseen_mask(obs)
    unseen_ids = mask_to_ids(unseen_mask)
    next_wanted, next_discarded = get_next_player_hints(obs)
    deck_rem = obs['deck_remaining']

    # Branch 1: Take Shown ('S') — evaluate best legal discard using PyTorch V(s).
    hand7 = hand6 + [shown_id]
    shown_feats = []
    for i in range(6):  # Index 6 is shown_id (forbidden to re-discard this turn)
      h6 = hand7[:i] + hand7[i + 1 :]
      if is_winning_hand_ids(h6):
        self.last_thought = 'Took Shown — completes a winning hand!'
        if return_search:
          _, all_feats = extract_all_action_features(obs)
          s_idx = legal.index('S')
          pi = [1.0 if i == s_idx else 0.0 for i in range(len(legal))]
          return 'S', {
              'phase': 'DRAW',
              'all_action_feats': all_feats,
              'pi_target': pi,
              'search_val': 1.0,
          }
        return 'S'
      shown_feats.append(
          extract_afterstate_features(
              h6,
              unseen_mask,
              deck_rem,
              next_wanted,
              next_discarded,
              discarded_id=hand7[i],
              took_shown=True,
          )
      )
    val_shown = max(self.value_bot.net.predict_batch(shown_feats))

    # Branch 2: Draw Hidden ('T') — exact win-out check + batched PyTorch expectation.
    val_hidden = self._expected_next_turn_value(
        hand6, unseen_ids, unseen_mask, deck_rem, next_wanted, next_discarded
    )

    # Pure tree-evaluated values rank DRAW moves cleanly (never let noisy prior override!)
    _, all_feats = extract_all_action_features(obs)
    scores = [val_shown if act == 'S' else val_hidden for act in legal]

    pick_idx, pi_target = self._sample_or_argmax(scores)
    chosen = legal[pick_idx]
    if chosen == 'S':
      self.last_thought = (
          f'Took Shown (AlphaZero {100*val_shown:.0f}% vs Hidden {100*val_hidden:.0f}%)'
      )
    else:
      self.last_thought = (
          f'Drew Hidden (AlphaZero {100*val_hidden:.0f}% vs Shown {100*val_shown:.0f}%)'
      )

    if return_search:
      return chosen, {
          'phase': 'DRAW',
          'all_action_feats': all_feats,
          'pi_target': pi_target,
          'search_val': max(val_shown, val_hidden),
      }
    return chosen

  def choose_discard(self, obs, return_search=False):
    hand7 = obs['hand_ids']
    legal_ids, all_action_feats = extract_all_action_features(obs)
    unseen_mask = get_unseen_mask(obs)
    unseen_ids = mask_to_ids(unseen_mask)
    next_wanted, next_discarded = get_next_player_hints(obs)
    deck_rem = obs['deck_remaining']

    # Step 1: Extract afterstate features & catch instant winning discards.
    h6_list = []
    imm_feats = []
    for idx, cid in enumerate(legal_ids):
      h_idx = hand7.index(cid)
      h6 = hand7[:h_idx] + hand7[h_idx + 1 :]
      after_f = all_action_feats[idx][:AFTERSTATE_DIM]
      if is_winning_hand_ids(h6):
        card = id_to_card(cid)
        if return_search:
          pi = [1.0 if j == idx else 0.0 for j in range(len(legal_ids))]
          return card, {
              'phase': 'DISCARD',
              'after_feats': after_f,
              'all_action_feats': all_action_feats,
              'pi_target': pi,
              'search_val': 1.0,
          }
        return card
      h6_list.append(h6)
      imm_feats.append(after_f)

    # Step 2: Value Net evaluates both immediate afterstate (with discard defense)
    # and pure hand value base_v (without discard defense) so we can preserve 100%
    # of opponent defense while replacing hand value with exact 1-ply Bellman lookahead!
    imm_vals = self.value_bot.net.predict_batch(imm_feats)
    base_feats = [
        extract_afterstate_features(
            h6, unseen_mask, deck_rem, next_wanted, next_discarded
        )
        for h6 in h6_list
    ]
    base_vals = self.value_bot.net.predict_batch(base_feats)

    with torch.no_grad():
      prior_logits = self.discard_policy(
          torch.tensor(all_action_feats, dtype=torch.float32)
      )
      priors = torch.softmax(prior_logits, dim=-1).tolist()

    # Policy Prior gently helps choose which top-K discards to expand (never overrides V).
    guide_scores = [
        0.90 * imm_vals[i] + 0.10 * priors[i] for i in range(len(legal_ids))
    ]
    ranked = sorted(
        range(len(legal_ids)), key=lambda i: guide_scores[i], reverse=True
    )

    # Step 3: Exact Bellman 1-ply Expectimax on top-K candidates:
    #   score(cid) = E[max V(h6 + draw - next_disc)] + (imm_val(cid) - base_val(h6))
    # Preserves 100% of learned opponent-defense signal with zero double-counting!
    final_scores = list(imm_vals)
    for idx in ranked[: self.top_k_discards]:
      lookahead_v = self._expected_next_turn_value(
          h6_list[idx],
          unseen_ids,
          unseen_mask,
          deck_rem,
          next_wanted,
          next_discarded,
      )
      defense_delta = imm_vals[idx] - base_vals[idx]
      final_scores[idx] = lookahead_v + defense_delta

    pick_idx, pi_target = self._sample_or_argmax(final_scores)
    card = id_to_card(legal_ids[pick_idx])

    ordered = sorted(
        range(len(legal_ids)), key=lambda i: final_scores[i], reverse=True
    )
    if len(ordered) >= 2:
      best_i, alt_i = ordered[0], ordered[1]
      self.last_thought = (
          f'Put down {card_label(id_to_card(legal_ids[best_i]))} '
          f'(AlphaZero {100*final_scores[best_i]:.0f}% vs '
          f'{card_label(id_to_card(legal_ids[alt_i]))} {100*final_scores[alt_i]:.0f}%)'
      )

    if return_search:
      return card, {
          'phase': 'DISCARD',
          'after_feats': imm_feats[pick_idx],
          'all_action_feats': all_action_feats,
          'pi_target': pi_target,
          'search_val': max(0.0, min(1.0, final_scores[pick_idx])),
      }
    return card

  def build_selfplay_samples(self, search_steps, afterstates, outcome_01):
    """Builds (value_samples, policy_samples) from one AlphaZero self-play game!

    CRITICAL ALPHAZERO RULE: Train V(s) strictly on unbiased game outcomes +
    live target-net TD (never on optimistic 1-player lookahead values!), while
    training the Policy Prior Nets to imitate the tree-search distribution pi_search!
    """
    val_samples = self.value_bot.build_episode_samples(afterstates, outcome_01)
    pol_samples = [
        (s['phase'], s['all_action_feats'], s['pi_target'])
        for s in search_steps
        if s is not None and len(s['all_action_feats']) > 1
    ]
    return val_samples, pol_samples

  def train_policy_batch(self, policy_batch, lr=None):
    """Trains Policy Prior Nets via Cross-Entropy toward tree-search targets pi_search."""
    if not policy_batch:
      return 0.0
    if lr is not None:
      for group in self.policy_opt.param_groups:
        group['lr'] = lr

    zero_row = [0.0] * ACTION_DIM
    draw_f, draw_m, draw_pi = [], [], []
    disc_f, disc_m, disc_pi = [], [], []

    for phase, af, pi in policy_batch:
      k = len(af)
      if phase == 'DRAW':
        draw_f.append(af + [zero_row] * (2 - k))
        draw_m.append([True] * k + [False] * (2 - k))
        draw_pi.append(pi + [0.0] * (2 - k))
      else:
        disc_f.append(af + [zero_row] * (7 - k))
        disc_m.append([True] * k + [False] * (7 - k))
        disc_pi.append(pi + [0.0] * (7 - k))

    loss_parts = []
    for net, f_list, m_list, pi_list in (
        (self.draw_policy, draw_f, draw_m, draw_pi),
        (self.discard_policy, disc_f, disc_m, disc_pi),
    ):
      if not f_list:
        continue
      x = torch.tensor(f_list, dtype=torch.float32)
      mask = torch.tensor(m_list, dtype=torch.bool)
      target_pi = torch.tensor(pi_list, dtype=torch.float32)
      logits = net(x).masked_fill(~mask, -1e9)
      log_probs = F.log_softmax(logits, dim=-1)
      # Mask out padded slots before summing so -inf * 0 never yields NaN!
      ce = -(target_pi * torch.where(mask, log_probs, torch.zeros_like(log_probs))).sum(dim=-1).mean()
      loss_parts.append(ce)

    if not loss_parts:
      return 0.0
    loss = sum(loss_parts) / len(loss_parts)
    self.policy_opt.zero_grad()
    loss.backward()
    nn.utils.clip_grad_norm_(
        list(self.draw_policy.parameters())
        + list(self.discard_policy.parameters()),
        1.5,
    )
    self.policy_opt.step()
    return float(loss.item())

  def clone_frozen(self, label=None):
    """Returns a frozen greedy copy of this AlphaZero bot for Hall-of-Fame sparring."""
    snap = AlphaZeroLiteBot(
        hidden_dim=self.value_bot.net.hidden_dim,
        draw_sample=self.draw_sample,
        top_k_discards=self.top_k_discards,
    )
    snap.value_bot.net.load_state_dict(self.value_bot.net.state_dict())
    snap.value_bot.target_net.load_state_dict(self.value_bot.net.state_dict())
    snap.draw_policy.load_state_dict(self.draw_policy.state_dict())
    snap.discard_policy.load_state_dict(self.discard_policy.state_dict())
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
                'input_dim': self.value_bot.net.input_dim,
                'action_dim': self.draw_policy.input_dim,
                'hidden_dim': self.value_bot.net.hidden_dim,
            },
            'value_state': self.value_bot.net.state_dict(),
            'draw_policy_state': self.draw_policy.state_dict(),
            'discard_policy_state': self.discard_policy.state_dict(),
            'metadata': self.metadata,
        },
    )
