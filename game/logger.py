"""JSONL trajectory logger with cached game counts and schema v2 backfill."""

import json
import os

SCHEMA_VERSION = 2


class TrajectoryLogger:
  """Persists completed game trajectories to a JSONL file.

  Set compact=True for large bulk self-play runs (e.g., 10,000 games): omits
  repeated per-step public_history and redundant string lists, keeping files
  ~100x smaller while preserving every card ID, action, and outcome.
  """

  def __init__(self, log_file='trajectories.jsonl', compact=False):
    self.log_file = log_file
    self.compact = compact
    self._saved_games = self._count_existing_games()

  def _count_existing_games(self):
    if not self.log_file or not os.path.exists(self.log_file):
      return 0
    game_ids = set()
    with open(self.log_file, 'r', encoding='utf-8') as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          record = json.loads(line)
          if 'game_id' in record:
            game_ids.add(record['game_id'])
        except json.JSONDecodeError:
          continue
    return len(game_ids)

  @property
  def saved_games(self):
    return self._saved_games

  def save_game(self, trajectory, winner_idx, winner_type=None):
    """Backfills terminal outcomes onto every step and appends to disk."""
    total_steps = len(trajectory)
    if total_steps == 0:
      return 0

    with open(self.log_file, 'a', encoding='utf-8') as f:
      for i, step_record in enumerate(trajectory):
        if self.compact:
          record = {
              'schema_version': SCHEMA_VERSION,
              'game_id': step_record['game_id'],
              'num_players': step_record['num_players'],
              'player_types': step_record['player_types'],
              'step': step_record['step'],
              'turn_number': step_record['turn_number'],
              'player': step_record['player'],
              'player_type': step_record['player_type'],
              'phase': step_record['phase'],
              'hand_ids': step_record['hand_ids'],
              'shown_card_id': step_record['shown_card_id'],
              'discard_pile_ids': step_record['discard_pile_ids'],
              'deck_remaining': step_record['deck_remaining'],
              'last_picked_shown_id': step_record['last_picked_shown_id'],
              'legal_action_ids': step_record['legal_action_ids'],
              'action': step_record['action'],
              'action_id': step_record['action_id'],
          }
        else:
          record = dict(step_record)
          record['schema_version'] = SCHEMA_VERSION

        is_winner = (
            (record['player'] == winner_idx)
            if winner_idx is not None
            else False
        )
        record['won'] = is_winner
        record['reward'] = (
            0.0 if winner_idx is None else (1.0 if is_winner else -1.0)
        )
        record['winner'] = winner_idx
        record['winner_type'] = winner_type
        record['steps_remaining'] = total_steps - i
        f.write(json.dumps(record, ensure_ascii=False) + '\n')

    self._saved_games += 1
    return total_steps
