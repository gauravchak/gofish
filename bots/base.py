"""Common interface for all computer players (rules, tree search, and future ML).

Every bot implements the exact same two methods using the same observation dict
that gets saved to trajectories.jsonl. That means swapping a rules-based bot
for an ML model later requires zero changes to the game engine or web app.
"""

import random


class BotPolicy:
  """Base class for any computer player."""

  name = 'Bot'
  player_type = 'bot:base'

  def choose_draw(self, obs):
    """Returns 'S' (Shown pile) or 'T' (Hidden deck) from obs['legal_actions']."""
    raise NotImplementedError

  def choose_discard(self, obs):
    """Returns a [rank, suit] card from obs['legal_actions']."""
    raise NotImplementedError


class RandomBot(BotPolicy):
  """Picks uniformly at random among legal moves (useful baseline for benchmarks)."""

  name = 'Random Bot'
  player_type = 'bot:random'

  def __init__(self, seed=None):
    self._rng = random.Random(seed)

  def choose_draw(self, obs):
    return self._rng.choice(obs['legal_actions'])

  def choose_discard(self, obs):
    return list(self._rng.choice(obs['legal_actions']))
