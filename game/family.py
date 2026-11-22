"""Local family player profiles and Elo rating tracker (stored in family.json).

Why keep family.json in .gitignore?
  Because it holds your real family names and match history! Keeping it local
  means you can safely share this repo publicly on GitHub without publishing
  personal info.

How Elo works here (K = 32, starting rating = 1000):
  - Expected win probability of A vs B:
        E(A, B) = 1 / (1 + 10 ** ((R_B - R_A) / 400))
  - In a 2-player game:
        delta_A = K * (score_A - E(A, B))   where score is 1 (win), 0.5 (draw), 0 (loss)
  - In a 3- or 4-player game:
        We compare the winner against each other player (or all pairs on a draw)
        and scale by 1 / (num_players - 1) so total Elo across all players is
        always conserved (zero-sum).
"""

from datetime import datetime, timezone
import json
import os
import tempfile

DEFAULT_ELO = 1000.0
DEFAULT_K_FACTOR = 32.0


def expected_score(rating_a, rating_b):
  """Returns standard Elo expected score of player A against player B (0..1)."""
  return 1.0 / (1.0 + 10.0 ** ((rating_b - rating_a) / 400.0))


class FamilyStore:
  """Persists player profiles, win/loss/draw stats, and Elo ratings to JSON."""

  def __init__(
      self,
      path='family.json',
      initial_elo=DEFAULT_ELO,
      k_factor=DEFAULT_K_FACTOR,
  ):
    self.path = path
    self.initial_elo = float(initial_elo)
    self.k_factor = float(k_factor)
    self.data = self._load()

  def _empty_data(self):
    return {
        'k_factor': self.k_factor,
        'initial_elo': self.initial_elo,
        'players': {},
        'recent_results': [],
    }

  def _load(self):
    if not self.path or not os.path.exists(self.path):
      return self._empty_data()
    try:
      with open(self.path, 'r', encoding='utf-8') as f:
        loaded = json.load(f)
      if not isinstance(loaded, dict):
        return self._empty_data()
      loaded.setdefault('k_factor', self.k_factor)
      loaded.setdefault('initial_elo', self.initial_elo)
      loaded.setdefault('players', {})
      loaded.setdefault('recent_results', [])
      return loaded
    except (OSError, json.JSONDecodeError):
      return self._empty_data()

  def _save(self):
    if not self.path:
      return
    parent = os.path.dirname(os.path.abspath(self.path))
    os.makedirs(parent, exist_ok=True)
    # Atomic write: write to a temp file in the same folder, then replace.
    fd, tmp_path = tempfile.mkstemp(prefix='.family_', suffix='.tmp', dir=parent)
    try:
      with os.fdopen(fd, 'w', encoding='utf-8') as f:
        json.dump(self.data, f, indent=2, ensure_ascii=False)
        f.write('\n')
      os.replace(tmp_path, self.path)
    finally:
      if os.path.exists(tmp_path):
        os.remove(tmp_path)

  def resolve_name(self, raw_name):
    """Matches existing saved names case-insensitively to prevent duplicate profiles."""
    cleaned = (raw_name or '').strip()[:24]
    if not cleaned:
      return ''
    lowered = cleaned.lower()
    for existing_name in self.data['players']:
      if existing_name.lower() == lowered:
        return existing_name
    return cleaned

  def ensure_player(self, name, player_type='human', save=True):
    """Finds or creates a player profile and returns its canonical name."""
    canonical = self.resolve_name(name)
    if not canonical:
      return ''
    players = self.data['players']
    now = datetime.now(timezone.utc).isoformat()
    if canonical not in players:
      players[canonical] = {
          'name': canonical,
          'type': player_type,
          'elo': round(self.initial_elo, 1),
          'games': 0,
          'wins': 0,
          'losses': 0,
          'draws': 0,
          'created_at': now,
          'last_played': None,
      }
      if save:
        self._save()
    return canonical

  def list_human_names(self):
    """Returns saved human player names sorted alphabetically for the lobby dropdown."""
    humans = [
        p['name']
        for p in self.data['players'].values()
        if not str(p.get('type', 'human')).startswith('bot:')
    ]
    return sorted(humans, key=lambda n: n.lower())

  def leaderboard(self):
    """Returns all saved players sorted by Elo descending (then wins, then name)."""
    entries = []
    for p in self.data['players'].values():
      entries.append({
          'name': p['name'],
          'type': p.get('type', 'human'),
          'is_bot': str(p.get('type', 'human')).startswith('bot:'),
          'elo': round(float(p.get('elo', self.initial_elo)), 1),
          'games': int(p.get('games', 0)),
          'wins': int(p.get('wins', 0)),
          'losses': int(p.get('losses', 0)),
          'draws': int(p.get('draws', 0)),
      })
    entries.sort(key=lambda e: (-e['elo'], -e['wins'], e['name'].lower()))
    return entries

  def record_match(self, game_id, match_players, winner_idx):
    """Updates W/L/D counts and Elo ratings after a completed match.

    Args:
      game_id: Short unique match ID.
      match_players: List of dicts {'name': str, 'type': str} in seat order.
      winner_idx: Winning seat index (0..N-1), or None for a Draw.

    Returns:
      Dict mapping player name -> rounded Elo delta (e.g., {'Alice': 16.0}).
    """
    n = len(match_players)
    if n < 2:
      return {}

    names = [
        self.ensure_player(p['name'], p.get('type', 'human'), save=False)
        for p in match_players
    ]
    players_db = self.data['players']
    old_ratings = [float(players_db[name]['elo']) for name in names]

    # Compute zero-sum pairwise Elo deltas scaled by 1 / (N - 1).
    scale = self.k_factor / (n - 1)
    seat_deltas = [0.0] * n

    if winner_idx is not None and 0 <= winner_idx < n:
      for j in range(n):
        if j == winner_idx or names[j] == names[winner_idx]:
          continue
        exp_w = expected_score(old_ratings[winner_idx], old_ratings[j])
        delta = round(scale * (1.0 - exp_w), 1)
        seat_deltas[winner_idx] += delta
        seat_deltas[j] -= delta
    else:
      # Draw: every unordered pair splits 0.5 / 0.5.
      for i in range(n):
        for j in range(i + 1, n):
          if names[i] == names[j]:
            continue
          exp_i = expected_score(old_ratings[i], old_ratings[j])
          delta = round(scale * (0.5 - exp_i), 1)
          seat_deltas[i] += delta
          seat_deltas[j] -= delta

    now = datetime.now(timezone.utc).isoformat()
    elo_deltas = {}
    for idx, name in enumerate(names):
      profile = players_db[name]
      delta = round(seat_deltas[idx], 1)
      profile['elo'] = round(float(profile['elo']) + delta, 1)
      profile['games'] = int(profile['games']) + 1
      if winner_idx is None:
        profile['draws'] = int(profile['draws']) + 1
      elif idx == winner_idx:
        profile['wins'] = int(profile['wins']) + 1
      else:
        profile['losses'] = int(profile['losses']) + 1
      profile['last_played'] = now
      elo_deltas[name] = round(elo_deltas.get(name, 0.0) + delta, 1)

    self.data['recent_results'] = (
        self.data['recent_results']
        + [{
            'game_id': game_id,
            'timestamp': now,
            'players': names,
            'winner': names[winner_idx] if winner_idx is not None else 'Draw',
            'elo_deltas': elo_deltas,
        }]
    )[-50:]

    self._save()
    return elo_deltas
