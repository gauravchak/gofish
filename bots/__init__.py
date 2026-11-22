"""Bot registry so the web lobby, benchmarks, and future ML bots share one lookup."""

from bots.base import BotPolicy, RandomBot
from bots.heuristic import HeuristicBot
from bots.tree_bot import TreeBot

BOT_OPTIONS = (
    {'key': 'heuristic', 'label': 'Heuristic Bot (Easy)'},
    {'key': 'tree_1ply', 'label': 'Tree Bot — 1-Level (Medium)'},
    {'key': 'tree_2ply', 'label': 'Tree Bot — 2-Level (Hard)'},
)


def create_bot(bot_key):
  """Instantiates a computer player by key."""
  if bot_key == 'random':
    return RandomBot()
  if bot_key == 'heuristic':
    return HeuristicBot()
  if bot_key == 'tree_1ply':
    return TreeBot(depth=1)
  if bot_key == 'tree_2ply':
    return TreeBot(depth=2)
  raise ValueError(f'Unknown bot key: {bot_key}')


__all__ = [
    'BOT_OPTIONS',
    'BotPolicy',
    'HeuristicBot',
    'RandomBot',
    'TreeBot',
    'create_bot',
]
