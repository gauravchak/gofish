"""Headless game engine for Two-Sets-of-3 (6-card Family Rummy).

Designed to serve human web play, tree-search bots, and ML policies through a
single unified observation/action interface with strict pre-action logging.
"""

import copy
from datetime import datetime, timezone
import random
import uuid

from game.logger import SCHEMA_VERSION
from game.rules import (
    card_label,
    card_to_id,
    create_deck,
    find_winning_sets,
    sort_cards,
)

DRAW_ACTION_IDS = {'T': 0, 'S': 1}


class GameEngine:
  """Pure Python state machine for a single Two-Sets-of-3 match."""

  def __init__(
      self,
      logger=None,
      forbid_same_turn_shown_discard=True,
      min_players=2,
      max_players=4,
  ):
    self.logger = logger
    self.forbid_same_turn_shown_discard = forbid_same_turn_shown_discard
    self.min_players = min_players
    self.max_players = max_players
    self.reset_lobby()

  def reset_lobby(self):
    """Resets to a 'Yet to Start' lobby where players can join."""
    self.started = False
    self.game_id = None
    self.players = []  # List of {'id': str, 'name': str, 'type': str}
    self.num_players = 0
    self.hands = []
    self.deck = []
    self.shown = []
    self.turn = 0
    self.turn_number = 0
    self.step_count = 0
    self.phase = 'DRAW'
    self.last_picked_shown = None
    self.last_drawn_card = None
    self.winner = None
    self.winner_idx = None
    self.winning_hand = []
    self.winning_sets = []
    self.saved = False
    self.public_history = []
    self.trajectory = []
    self.log = 'Waiting for players to join...'

  def get_player_index(self, client_id):
    for idx, player in enumerate(self.players):
      if player['id'] == client_id:
        return idx
    return None

  def player_name(self, idx):
    if idx is not None and 0 <= idx < len(self.players):
      return self.players[idx]['name']
    return f'Player {idx + 1}' if idx is not None else ''

  def player_type(self, idx):
    if idx is not None and 0 <= idx < len(self.players):
      return self.players[idx]['type']
    return 'unknown'

  def join_player(self, client_id, name='', player_type='human'):
    """Adds a human or bot player to the lobby before a match starts."""
    if self.started:
      return {'ok': False, 'error': 'Game already started.'}
    if self.get_player_index(client_id) is not None:
      return {'ok': False, 'error': 'Player already joined.'}
    if len(self.players) >= self.max_players:
      return {'ok': False, 'error': 'Lobby is full.'}

    idx = len(self.players)
    clean_name = (name or '').strip()[:16]
    display_name = clean_name if clean_name else f'Player {idx + 1}'
    self.players.append({
        'id': client_id,
        'name': display_name,
        'type': player_type,
    })
    self.log = f'{display_name} joined!'
    return {'ok': True, 'player_idx': idx}

  def start_match(self, seed=None):
    """Deals cards to everyone currently in the lobby and locks the roster."""
    if self.started or len(self.players) < self.min_players:
      return False

    rng = random.Random(seed) if seed is not None else random
    deck = create_deck()
    rng.shuffle(deck)

    self.num_players = len(self.players)
    self.game_id = str(uuid.uuid4())[:8]
    self.started = True
    self.deck = deck
    self.hands = [
        [self.deck.pop() for _ in range(6)] for _ in range(self.num_players)
    ]
    self.shown = [self.deck.pop()]
    self.turn = 0
    self.turn_number = 0
    self.step_count = 0
    self.phase = 'DRAW'
    self.last_picked_shown = None
    self.last_drawn_card = None
    self.winner = None
    self.winner_idx = None
    self.winning_hand = []
    self.winning_sets = []
    self.saved = False
    self.public_history = []
    self.trajectory = []
    self.log = (
        f'Game started with {self.num_players} players! '
        f'{self.player_name(0)} goes first.'
    )
    return True

  def rematch(self, seed=None):
    """Starts a fresh deal keeping the exact same player roster."""
    if len(self.players) < self.min_players:
      return False
    self.started = False
    return self.start_match(seed=seed)

  def legal_draw_actions(self):
    """Returns valid draw sources ('T', 'S') in the current state."""
    actions = []
    if self.deck:
      actions.append('T')
    if self.shown:
      actions.append('S')
    return actions

  def legal_discard_cards(self, player_idx):
    """Returns canonically sorted cards that player_idx is legally allowed to discard."""
    hand = sort_cards(self.hands[player_idx])
    if (
        self.forbid_same_turn_shown_discard
        and self.last_picked_shown is not None
        and len(hand) > 1
    ):
      forbidden_id = card_to_id(self.last_picked_shown)
      return [c for c in hand if card_to_id(c) != forbidden_id]
    return hand

  def get_observation(self, player_idx):
    """Builds the strict pre-action, information-set-faithful observation for player_idx.

    Used both for live bot/ML inference and for recording trajectories so that
    training and serving share the exact same schema with zero data leakage.
    """
    if not self.started or not (0 <= player_idx < self.num_players):
      raise ValueError(f'Invalid player_idx {player_idx} for current match.')

    canonical_hand = sort_cards(self.hands[player_idx])
    shown_top = list(self.shown[-1]) if self.shown else None
    discard_pile = [list(c) for c in self.shown]

    if self.phase == 'DRAW':
      legal_actions = self.legal_draw_actions()
      legal_action_ids = [DRAW_ACTION_IDS[a] for a in legal_actions]
    else:
      legal_actions = self.legal_discard_cards(player_idx)
      legal_action_ids = [card_to_id(c) for c in legal_actions]

    return {
        'schema_version': SCHEMA_VERSION,
        'game_id': self.game_id,
        'num_players': self.num_players,
        'player_types': [p['type'] for p in self.players],
        'step': self.step_count,
        'turn_number': self.turn_number,
        'player': player_idx,
        'player_type': self.player_type(player_idx),
        'phase': self.phase,
        'hand': canonical_hand,
        'hand_ids': [card_to_id(c) for c in canonical_hand],
        'shown_card': shown_top,
        'shown_card_id': card_to_id(shown_top) if shown_top else None,
        'discard_pile': discard_pile,
        'discard_pile_ids': [card_to_id(c) for c in discard_pile],
        'deck_remaining': len(self.deck),
        'last_picked_shown': (
            list(self.last_picked_shown) if self.last_picked_shown else None
        ),
        'last_picked_shown_id': (
            card_to_id(self.last_picked_shown)
            if self.last_picked_shown
            else None
        ),
        'public_history': copy.deepcopy(self.public_history),
        'legal_actions': legal_actions,
        'legal_action_ids': legal_action_ids,
    }

  def reorder_card(self, player_idx, idx):
    """Moves a card to the front of a player's UI hand (does not affect game state)."""
    if (
        not self.started
        or player_idx is None
        or not isinstance(idx, int)
        or not (0 <= idx < len(self.hands[player_idx]))
    ):
      return False
    card = self.hands[player_idx].pop(idx)
    self.hands[player_idx].insert(0, card)
    return True

  def draw(self, player_idx, source):
    """Executes a DRAW action ('T' = Hidden Deck, 'S' = Shown Pile)."""
    if (
        not self.started
        or player_idx is None
        or self.winner is not None
        or self.turn != player_idx
        or self.phase != 'DRAW'
    ):
      return {'ok': False, 'error': 'Not your draw turn.'}

    if source not in self.legal_draw_actions():
      return {'ok': False, 'error': f'Invalid draw source: {source}'}

    # Capture strict PRE-ACTION observation before mutating deck or shown pile.
    obs = self.get_observation(player_idx)

    if source == 'S':
      card = self.shown.pop()
      self.last_picked_shown = list(card)
    else:
      card = self.deck.pop()
      self.last_picked_shown = None

    self.trajectory.append({
        **obs,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': source,
        'action_id': DRAW_ACTION_IDS[source],
    })

    # Public history never reveals hidden draws (card is None when source == 'T').
    self.public_history.append({
        'step': self.step_count,
        'turn_number': self.turn_number,
        'player': player_idx,
        'phase': 'DRAW',
        'source': source,
        'card': list(card) if source == 'S' else None,
        'card_id': card_to_id(card) if source == 'S' else None,
    })

    self.step_count += 1
    self.hands[player_idx].append(card)
    self.last_drawn_card = list(card)
    self.phase = 'DISCARD'
    source_label = (
        f'Shown ({card_label(card)})' if source == 'S' else 'Hidden Deck'
    )
    self.log = f'{self.player_name(player_idx)} picked from {source_label}.'
    return {'ok': True, 'card': list(card)}

  def _resolve_discard_index(self, player_idx, card_or_index, by_index):
    hand = self.hands[player_idx]
    if by_index:
      if not isinstance(card_or_index, int) or isinstance(card_or_index, bool):
        return None
      if 0 <= card_or_index < len(hand):
        return card_or_index
      return None

    target_id = (
        card_or_index
        if isinstance(card_or_index, int) and not isinstance(card_or_index, bool)
        else card_to_id(card_or_index)
    )
    for i, c in enumerate(hand):
      if card_to_id(c) == target_id:
        return i
    return None

  def discard(self, player_idx, card_or_index, by_index=True):
    """Executes a DISCARD action (by hand index for UI, or by card/card_id for bots)."""
    if (
        not self.started
        or player_idx is None
        or self.winner is not None
        or self.turn != player_idx
        or self.phase != 'DISCARD'
    ):
      return {'ok': False, 'error': 'Not your discard turn.'}

    idx = self._resolve_discard_index(player_idx, card_or_index, by_index)
    if idx is None:
      return {'ok': False, 'error': 'Invalid card selection.'}

    candidate = self.hands[player_idx][idx]
    if (
        self.forbid_same_turn_shown_discard
        and self.last_picked_shown is not None
        and card_to_id(candidate) == card_to_id(self.last_picked_shown)
        and len(self.hands[player_idx]) > 1
    ):
      return {
          'ok': False,
          'error': 'Cannot discard the exact card you just picked from Shown.',
      }

    # Capture strict PRE-ACTION observation before removing card from hand.
    obs = self.get_observation(player_idx)

    card = self.hands[player_idx].pop(idx)
    card_id = card_to_id(card)

    self.trajectory.append({
        **obs,
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'action': list(card),
        'action_id': card_id,
    })

    self.public_history.append({
        'step': self.step_count,
        'turn_number': self.turn_number,
        'player': player_idx,
        'phase': 'DISCARD',
        'card': list(card),
        'card_id': card_id,
    })

    self.step_count += 1
    self.shown.append(card)
    self.last_picked_shown = None
    self.last_drawn_card = None
    self.log = f'{self.player_name(player_idx)} put down {card_label(card)}.'

    win_sets = find_winning_sets(self.hands[player_idx])
    if win_sets is not None:
      self.winner = self.player_name(player_idx)
      self.winner_idx = player_idx
      self.winning_sets = [win_sets[0], win_sets[1]]
      self.winning_hand = win_sets[0] + win_sets[1]
      self.log = f'🎉 {self.winner} WINS! (Trajectory saved)'
      self._finalize_trajectory(
          winner_idx=player_idx, winner_type=self.player_type(player_idx)
      )
    elif len(self.deck) == 0:
      self.winner = 'Draw'
      self.winner_idx = None
      self.log = 'Deck empty — Draw game! (Trajectory saved)'
      self._finalize_trajectory(winner_idx=None, winner_type=None)
    else:
      self.turn = (self.turn + 1) % self.num_players
      self.turn_number += 1
      self.phase = 'DRAW'

    return {'ok': True, 'card': list(card), 'winner': self.winner}

  def _finalize_trajectory(self, winner_idx, winner_type):
    if self.saved:
      return
    if self.logger is not None:
      self.logger.save_game(
          self.trajectory, winner_idx=winner_idx, winner_type=winner_type
      )
    self.saved = True

  def clone(self):
    """Returns a deep copy of the engine without a disk logger for tree search."""
    cloned = GameEngine(
        logger=None,
        forbid_same_turn_shown_discard=self.forbid_same_turn_shown_discard,
        min_players=self.min_players,
        max_players=self.max_players,
    )
    cloned.started = self.started
    cloned.game_id = self.game_id
    cloned.players = copy.deepcopy(self.players)
    cloned.num_players = self.num_players
    cloned.hands = copy.deepcopy(self.hands)
    cloned.deck = copy.deepcopy(self.deck)
    cloned.shown = copy.deepcopy(self.shown)
    cloned.turn = self.turn
    cloned.turn_number = self.turn_number
    cloned.step_count = self.step_count
    cloned.phase = self.phase
    cloned.last_picked_shown = copy.deepcopy(self.last_picked_shown)
    cloned.last_drawn_card = copy.deepcopy(self.last_drawn_card)
    cloned.winner = self.winner
    cloned.winner_idx = self.winner_idx
    cloned.winning_hand = copy.deepcopy(self.winning_hand)
    cloned.winning_sets = copy.deepcopy(self.winning_sets)
    cloned.saved = True  # Never persist simulated trajectories
    cloned.public_history = copy.deepcopy(self.public_history)
    cloned.trajectory = []
    cloned.log = self.log
    return cloned

  def get_ui_state(self, client_id):
    """Builds the JSON payload consumed by the browser client."""
    my_player = self.get_player_index(client_id)
    hand = self.hands[my_player] if (self.started and my_player is not None) else []
    is_my_discard = (
        self.started
        and self.winner is None
        and my_player is not None
        and self.turn == my_player
        and self.phase == 'DISCARD'
    )
    forbidden_discard_idx = None
    if (
        is_my_discard
        and self.forbid_same_turn_shown_discard
        and self.last_picked_shown is not None
    ):
      forbidden_id = card_to_id(self.last_picked_shown)
      for idx, card in enumerate(hand):
        if card_to_id(card) == forbidden_id:
          forbidden_discard_idx = idx
          break

    return {
        'started': self.started,
        'joined_count': len(self.players),
        'player_names': [p['name'] for p in self.players],
        'my_player': my_player,
        'my_name': self.player_name(my_player) if my_player is not None else None,
        'my_ip': client_id,
        'hand': [card_label(c) for c in hand],
        'forbidden_discard_idx': forbidden_discard_idx,
        'shown': card_label(self.shown[-1]) if self.shown else 'Empty',
        'deck_count': len(self.deck),
        'turn': self.turn,
        'turn_name': self.player_name(self.turn) if self.started else '',
        'phase': self.phase,
        'winner': self.winner,
        'winning_hand': [card_label(c) for c in self.winning_hand],
        'winning_sets': [
            [card_label(c) for c in s] for s in self.winning_sets
        ],
        'log': self.log,
        'saved_games': self.logger.saved_games if self.logger is not None else 0,
    }
