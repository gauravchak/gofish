"""Simple Experience Replay Buffer for PyTorch RL training.

Why use a Replay Buffer?
  - Moves inside a single card game are strongly correlated.
  - Storing recent transitions in a fixed-size ring buffer and sampling random
    mini-batches (e.g., 64 at a time) breaks that correlation so PyTorch's Adam
    optimizer learns much more smoothly and reliably!
"""

from collections import deque


class ReplayBuffer:
  """Fixed-capacity ring buffer supporting uniform random mini-batch sampling."""

  def __init__(self, capacity=25000):
    self._buffer = deque(maxlen=capacity)

  def push(self, item):
    self._buffer.append(item)

  def extend(self, items):
    self._buffer.extend(items)

  def __len__(self):
    return len(self._buffer)

  def sample(self, batch_size, rng):
    """Returns up to batch_size randomly sampled items without replacement."""
    n = len(self._buffer)
    if n == 0:
      return []
    if n <= batch_size:
      return list(self._buffer)
    return rng.sample(self._buffer, batch_size)
