"""Model 6: PyTorch Rollout-Distilled Student Bot (Expert Iteration / LLM Distillation).

Core idea (just like distilling a slow reasoning LLM into a fast student model!):
  1. Freeze a real game decision using ONLY legally visible cards (zero deck peeking!).
  2. Sample K valid random worlds for the unseen deck & opponent's unknown cards
     (respecting every card the opponent publicly picked from Shown!).
  3. Using Common Random Numbers (the EXACT same K worlds for every candidate move!),
     roll out every legal action to the end of the game to measure its true EV(s, a)!
  4. Distill the Teacher's sharp move distribution pi_teacher = softmax(EV / temp)
     into a fast 0.2 ms PyTorch CardMLP via soft-target Cross-Entropy!
  5. Expert Iteration: plug the trained Student back into Step 3 as the rollout
     player to build an even stronger Teacher for the next generation!
"""

import os
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from bots.base import BotPolicy
from bots.heuristic import (
    get_next_player_hints,
    get_unseen_mask,
    is_winning_hand_ids,
    mask_to_ids,
)
from game.engine import GameEngine
from game.rules import card_label, id_to_card
from ml.features import ACTION_DIM, extract_all_action_features
from ml.nn import CardMLP, load_checkpoint, save_checkpoint


def evaluate_state_with_rollouts(
    obs, rollout_bot, num_worlds=12, rng=None, max_actions_to_eval=7
):
  """Fair Monte-Carlo Teacher: estimates true win-rate EV for every legal action.

  STRICTLY FAIR: builds K random valid worlds solely from `obs` (never peeks at
  the real hidden deck or opponent's secret cards), and uses Common Random Numbers
  so every candidate action faces the exact same K worlds!
  """
  rng = rng or random.Random(0)
  legal, all_feats = extract_all_action_features(obs)
  if len(legal) <= 1:
    return None

  phase = obs['phase']
  my_hand = list(obs['hand_ids'])
  discard_pile = list(obs['discard_pile_ids'])
  deck_rem = int(obs['deck_remaining'])
  unseen_mask = get_unseen_mask(obs)
  unseen_ids = mask_to_ids(unseen_mask)
  next_wanted, _ = get_next_player_hints(obs)

  # Keep only known opponent Shown pickups that aren't in our hand or discard pile.
  known_opp = [
      cid for cid in next_wanted
      if cid not in my_hand and cid not in discard_pile
  ][:6]
  need_opp = max(0, 6 - len(known_opp))

  # Pre-sample K valid random unseen worlds (Common Random Numbers!).
  worlds = []
  for _ in range(num_worlds):
    pool = list(unseen_ids)
    rng.shuffle(pool)
    opp_hand = known_opp + pool[:need_opp]
    deck_ids = pool[need_opp : need_opp + deck_rem]
    if len(opp_hand) == 6:
      worlds.append((opp_hand, deck_ids))
  if not worlds:
    return None

  # Evaluate every legal action across the exact same K worlds.
  evs = []
  for act in legal[:max_actions_to_eval]:
    score_sum = 0.0
    for opp_hand, deck_ids in worlds:
      eng = GameEngine(logger=None)
      eng.join_player('p0', 'Student', player_type='bot:rollout')
      eng.join_player('p1', 'Opp', player_type='bot:rollout')
      eng.start_match(seed=1)

      eng.hands[0] = [id_to_card(c) for c in my_hand]
      eng.hands[1] = [id_to_card(c) for c in opp_hand]
      eng.shown = [id_to_card(c) for c in discard_pile]
      eng.deck = [id_to_card(c) for c in deck_ids]
      eng.turn = 0
      eng.phase = phase
      eng.last_picked_shown_id = obs.get('last_picked_shown_id')
      eng.public_history = list(obs.get('public_history', []))

      if phase == 'DRAW':
        res = eng.draw(0, act)
      else:
        res = eng.discard(0, id_to_card(act), by_index=False)
      if not res['ok']:
        score_sum += 0.0
        continue

      while eng.winner is None:
        p = eng.turn
        o = eng.get_observation(p)
        if eng.phase == 'DRAW':
          eng.draw(p, rollout_bot.choose_draw(o))
        else:
          eng.discard(p, rollout_bot.choose_discard(o), by_index=False)

      if eng.winner_idx == 0:
        score_sum += 1.0
      elif eng.winner_idx is None:
        score_sum += 0.5
    evs.append(score_sum / len(worlds))

  return {
      'phase': phase,
      'all_action_feats': all_feats[: len(evs)],
      'evs': evs,
  }


class DistilledBot(BotPolicy):
  """Fast PyTorch Student distilled from a Multi-World Monte-Carlo Rollout Teacher."""

  name = 'RL Bot — Rollout Distilled (PyTorch)'
  player_type = 'bot:rl_distilled'

  def __init__(self, model_path=None, hidden_dim=64, lr=0.002, seed=606):
    self.metadata = {}
    loaded = False
    if model_path and os.path.exists(model_path):
      ckpt = load_checkpoint(model_path)
      cfg = ckpt.get('config', {})
      if cfg.get('action_dim') == ACTION_DIM and 'draw_state' in ckpt:
        h = cfg['hidden_dim']
        self.draw_net = CardMLP(ACTION_DIM, h, 'linear')
        self.discard_net = CardMLP(ACTION_DIM, h, 'linear')
        self.draw_net.load_state_dict(ckpt['draw_state'])
        self.discard_net.load_state_dict(ckpt['discard_state'])
        self.metadata = ckpt.get('metadata', {})
        loaded = True

    if not loaded:
      torch.manual_seed(seed)
      self.draw_net = CardMLP(ACTION_DIM, hidden_dim, 'linear')
      self.discard_net = CardMLP(ACTION_DIM, hidden_dim, 'linear')

    self.optimizer = torch.optim.Adam(
        list(self.draw_net.parameters()) + list(self.discard_net.parameters()),
        lr=lr,
        weight_decay=1e-4,
    )

  def choose_draw(self, obs):
    legal, all_feats = extract_all_action_features(obs)
    if len(legal) == 1:
      return legal[0]

    hand6 = obs['hand_ids']
    shown_id = obs['shown_card_id']
    hand7 = hand6 + [shown_id]
    for i in range(6):
      if is_winning_hand_ids(hand7[:i] + hand7[i + 1 :]):
        self.last_thought = 'Took Shown — completes a winning hand!'
        return 'S'

    with torch.no_grad():
      logits = self.draw_net(torch.tensor(all_feats, dtype=torch.float32))
      probs = torch.softmax(logits, dim=-1).tolist()
    best_idx = max(range(len(probs)), key=lambda i: probs[i])
    other_idx = 1 - best_idx
    lbl_best = 'Shown' if legal[best_idx] == 'S' else 'Hidden'
    lbl_alt = 'Shown' if legal[other_idx] == 'S' else 'Hidden'
    self.last_thought = (
        f'Picked {lbl_best} (Distilled {100*probs[best_idx]:.0f}% vs '
        f'{lbl_alt} {100*probs[other_idx]:.0f}%)'
    )
    return legal[best_idx]

  def choose_discard(self, obs):
    hand7 = obs['hand_ids']
    legal_ids, all_feats = extract_all_action_features(obs)

    for cid in legal_ids:
      h_idx = hand7.index(cid)
      if is_winning_hand_ids(hand7[:h_idx] + hand7[h_idx + 1 :]):
        card = id_to_card(cid)
        self.last_thought = f'Put down {card_label(card)} — winning hand!'
        return card

    with torch.no_grad():
      logits = self.discard_net(torch.tensor(all_feats, dtype=torch.float32))
      probs = torch.softmax(logits, dim=-1).tolist()
    ranked = sorted(range(len(probs)), key=lambda i: probs[i], reverse=True)
    best_idx = ranked[0]
    card = id_to_card(legal_ids[best_idx])
    if len(ranked) >= 2:
      alt_idx = ranked[1]
      self.last_thought = (
          f'Put down {card_label(card)} (Distilled {100*probs[best_idx]:.0f}% vs '
          f'{card_label(id_to_card(legal_ids[alt_idx]))} {100*probs[alt_idx]:.0f}%)'
      )
    return card

  def train_on_teacher_dataset(
      self, samples, epochs=25, batch_size=64, teacher_temp=0.08, lr=0.002, seed=7
  ):
    """Distills soft Teacher move distributions into draw_net & discard_net."""
    if not samples:
      return 0.0
    for g in self.optimizer.param_groups:
      g['lr'] = lr

    rng = random.Random(seed)
    zero_row = [0.0] * ACTION_DIM
    last_loss = 0.0

    for _ in range(epochs):
      shuffled = list(samples)
      rng.shuffle(shuffled)
      for start in range(0, len(shuffled), batch_size):
        chunk = shuffled[start : start + batch_size]
        draw_f, draw_m, draw_pi = [], [], []
        disc_f, disc_m, disc_pi = [], [], []
        for s in chunk:
          af = s['all_action_feats']
          evs = s['evs']
          k = len(af)
          if k <= 1:
            continue
          ev_t = torch.tensor(evs, dtype=torch.float32)
          pi = torch.softmax(ev_t / max(teacher_temp, 1e-3), dim=-1).tolist()
          if s['phase'] == 'DRAW':
            draw_f.append(af + [zero_row] * (2 - k))
            draw_m.append([True] * k + [False] * (2 - k))
            draw_pi.append(pi + [0.0] * (2 - k))
          else:
            disc_f.append(af + [zero_row] * (7 - k))
            disc_m.append([True] * k + [False] * (7 - k))
            disc_pi.append(pi + [0.0] * (7 - k))

        parts = []
        for net, fl, ml, pl in (
            (self.draw_net, draw_f, draw_m, draw_pi),
            (self.discard_net, disc_f, disc_m, disc_pi),
        ):
          if not fl:
            continue
          x = torch.tensor(fl, dtype=torch.float32)
          mask = torch.tensor(ml, dtype=torch.bool)
          target_pi = torch.tensor(pl, dtype=torch.float32)
          logits = net(x).masked_fill(~mask, -1e9)
          log_p = F.log_softmax(logits, dim=-1)
          ce = -(
              target_pi * torch.where(mask, log_p, torch.zeros_like(log_p))
          ).sum(dim=-1).mean()
          parts.append(ce)

        if parts:
          loss = sum(parts) / len(parts)
          self.optimizer.zero_grad()
          loss.backward()
          nn.utils.clip_grad_norm_(
              list(self.draw_net.parameters())
              + list(self.discard_net.parameters()),
              1.5,
          )
          self.optimizer.step()
          last_loss = float(loss.item())

    return last_loss

  def save(self, path, metadata=None):
    self.metadata = metadata or self.metadata
    save_checkpoint(
        path,
        {
            'config': {
                'action_dim': self.draw_net.input_dim,
                'hidden_dim': self.draw_net.hidden_dim,
            },
            'draw_state': self.draw_net.state_dict(),
            'discard_state': self.discard_net.state_dict(),
            'metadata': self.metadata,
        },
    )
