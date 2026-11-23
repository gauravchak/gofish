"""Bot registry and Elo-based auto-matching for the web lobby and benchmarks.

If PyTorch is installed (pip install -r requirements-ml.txt), the three trained
RL bots are automatically registered and loaded from models/*.pt. Otherwise the
game runs cleanly with human + Random/Beginner/Heuristic/Tree players.
"""

import os

from bots.base import BotPolicy, RandomBot
from bots.heuristic import HeuristicBot
from bots.novice import NoviceBot
from bots.tree_bot import TreeBot

try:
  from ml.policy_gradient import PolicyGradientBot
  from ml.q_learning import QLearningBot
  from ml.td_value import TDValueBot

  HAS_TORCH = True
except ImportError:
  PolicyGradientBot = None
  QLearningBot = None
  TDValueBot = None
  HAS_TORCH = False

MODELS_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models'
)

RL_MODEL_PATHS = {
    'rl_td_value': os.path.join(MODELS_DIR, 'td_value_model.pt'),
    'rl_q_learning': os.path.join(MODELS_DIR, 'q_learning_model.pt'),
    'rl_policy_gradient': os.path.join(MODELS_DIR, 'policy_gradient_model.pt'),
}

_BASE_BOT_OPTIONS = (
    {'key': 'auto', 'label': 'Auto-Match by Elo (Recommended)'},
    {'key': 'random', 'label': 'Random Bot (Starter)'},
    {'key': 'novice', 'label': 'Beginner Bot (Very Easy)'},
    {'key': 'heuristic', 'label': 'Heuristic Bot (Easy)'},
    {'key': 'tree_1ply', 'label': 'Tree Bot — 1-Level (Medium)'},
    {'key': 'tree_2ply', 'label': 'Tree Bot — 2-Level (Hard)'},
)

_RL_BOT_OPTIONS = (
    {'key': 'rl_td_value', 'label': '🧠 RL Bot — TD Value (PyTorch)'},
    {'key': 'rl_q_learning', 'label': '🧠 RL Bot — Q-Learning (PyTorch)'},
    {'key': 'rl_policy_gradient', 'label': '🧠 RL Bot — Policy Gradient (PyTorch)'},
)

BOT_OPTIONS = (
    _BASE_BOT_OPTIONS + _RL_BOT_OPTIONS if HAS_TORCH else _BASE_BOT_OPTIONS
)

# Starting Elo ratings for each computer tier in family.json so the leaderboard
# reflects relative bot difficulty right away.
BOT_STARTING_ELO = {
    'bot:random': 800.0,
    'bot:novice': 920.0,
    'bot:heuristic': 1020.0,
    'bot:tree_1ply': 1080.0,
    'bot:tree_2ply': 1140.0,
    'bot:rl_td_value': 1080.0,
    'bot:rl_q_learning': 1100.0,
    'bot:rl_policy_gradient': 1100.0,
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
  """Instantiates a computer player by key (rules, tree, or trained PyTorch RL)."""
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
  if bot_key in RL_MODEL_PATHS:
    if not HAS_TORCH:
      raise RuntimeError(
          'PyTorch is not installed. Run: pip install -r requirements-ml.txt'
      )
    if bot_key == 'rl_td_value':
      return TDValueBot(model_path=RL_MODEL_PATHS['rl_td_value'])
    if bot_key == 'rl_q_learning':
      return QLearningBot(model_path=RL_MODEL_PATHS['rl_q_learning'])
    if bot_key == 'rl_policy_gradient':
      return PolicyGradientBot(model_path=RL_MODEL_PATHS['rl_policy_gradient'])
  raise ValueError(f'Unknown bot key: {bot_key}')


__all__ = [
    'AUTO_MATCH_TIERS',
    'BOT_OPTIONS',
    'BOT_STARTING_ELO',
    'BotPolicy',
    'HAS_TORCH',
    'HeuristicBot',
    'NoviceBot',
    'PolicyGradientBot',
    'QLearningBot',
    'RL_MODEL_PATHS',
    'RandomBot',
    'TDValueBot',
    'TreeBot',
    'choose_bot_for_elo',
    'create_bot',
]
