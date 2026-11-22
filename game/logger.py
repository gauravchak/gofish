"""JSONL trajectory logger with cached game counts and schema v2 backfill."""

import json
import os

SCHEMA_VERSION = 2


class TrajectoryLogger:
  """Persists completed game trajectories to a JSONL file."""

  def __init__(self, log_file='trajectories.jsonl'):
    self.log_file = log_file
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
        record = dict(step_record)
        is_winner = (
            (record['player'] == winner_idx)
            if winner_idx is not None
            else False
        )
        record['schema_version'] = SCHEMA_VERSION
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
