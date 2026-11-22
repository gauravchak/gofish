"""Bot registry and Elo-based auto-matching for the web lobby and benchmarks."""

from bots.base import BotPolicy, RandomBot
from bots.heuristic import HeuristicBot
from bots.novice import NoviceBot
from bots.tree_bot import TreeBot

BOT_OPTIONS = (
    {'key': 'auto', 'label': 'Auto-Match by Elo (Recommended)'},
    {'key': 'random', 'label': 'Random Bot (Starter)'},
    {'key': 'novice', 'label': 'Beginner Bot (Very Easy)'},
    {'key': 'heuristic', 'label': 'Heuristic Bot (Easy)'},
    {'key': 'tree_1ply', 'label': 'Tree Bot — 1-Level (Medium)'},
    {'key': 'tree_2ply', 'label': 'Tree Bot — 2-Level (Hard)'},
)

# Starting Elo ratings for each computer tier in family.json so the leaderboard
# reflects relative bot difficulty right away.
BOT_STARTING_ELO = {
    'bot:random': 800.0,
    'bot:novice': 920.0,
    'bot:heuristic': 1020.0,
    'bot:tree_1ply': 1080.0,
    'bot:tree_2ply': 1140.0,
}

# Elo thresholds for Auto-Match: brand-new players (1000 Elo) start with a warm-up
# game against Random Bot (1 win -> ~1008 Elo graduates to Beginner Bot!).
AUTO_MATCH_TIERS = (
    (1007.0, 'random'),
    (1035.0, 'novice'),
    (1085.0, 'heuristic'),
    (1140.0, 'tree_1ply'),
)


def choose_bot_for_elo(elo):
  """Picks a computer player key appropriate for the given human Elo rating."""
  rating = float(elo)
  for ceiling, bot_key in AUTO_MATCH_TIERS:
    if rating < ceiling:
      return bot_key
  return 'tree_2ply'


def create_bot(bot_key):
  """Instantiates a computer player by key."""
  if bot_key == 'random':
    return RandomBot()
  if bot_key == 'novice':
    return NoviceBot()
  if bot_key == 'heuristic':
    return HeuristicBot()
  if bot_key == 'tree_1ply':
    return TreeBot(depth=1)
  if bot_key == 'tree_2ply':
    return TreeBot(depth=2)
  raise ValueError(f'Unknown bot key: {bot_key}')


__all__ = [
    'AUTO_MATCH_TIERS',
    'BOT_OPTIONS',
    'BOT_STARTING_ELO',
    'BotPolicy',
    'HeuristicBot',
    'NoviceBot',
    'RandomBot',
    'TreeBot',
    'choose_bot_for_elo',
    'create_bot',
]
