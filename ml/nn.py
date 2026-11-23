"""Clean, idiomatic PyTorch neural network building block for GoFish RL agents.

Every RL bot in ml/ uses this tiny CardMLP (torch.nn.Module) so kids can read
standard PyTorch patterns (nn.Sequential, nn.Linear, nn.Tanh, torch.save/load
with weights_only=True) without boilerplate.
"""

import os

import torch
import torch.nn as nn


class CardMLP(nn.Module):
  """Small 2-hidden-layer MLP: input_dim -> hidden -> hidden -> 1.

  Supports three output activations:
    - 'sigmoid': outputs in (0, 1)  (TD Value bot predicting win probability)
    - 'tanh':    outputs in (-1, 1) (Q-Learning & Critic predicting return)
    - 'linear':  unbounded logit    (Policy Gradient Actor before softmax)
  """

  def __init__(
      self,
      input_dim,
      hidden_dim=32,
      output_activation='linear',
  ):
    super().__init__()
    if output_activation not in ('linear', 'sigmoid', 'tanh'):
      raise ValueError(f'Unsupported activation: {output_activation}')
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_activation = output_activation

    self.net = nn.Sequential(
        nn.Linear(input_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, hidden_dim),
        nn.Tanh(),
        nn.Linear(hidden_dim, 1),
    )

  def forward(self, x):
    """Forward pass for a single feature vector [D] or a batch [N, D]."""
    out = self.net(x).squeeze(-1)
    if self.output_activation == 'sigmoid':
      return torch.sigmoid(out)
    if self.output_activation == 'tanh':
      return torch.tanh(out)
    return out

  @torch.no_grad()
  def predict_one(self, features):
    """Convenience helper returning a plain Python float for one feature list."""
    x = torch.tensor(features, dtype=torch.float32)
    return float(self.forward(x).item())

  @torch.no_grad()
  def predict_batch(self, feature_rows):
    """Evaluates a list of feature vectors in one batched PyTorch call."""
    if not feature_rows:
      return []
    x = torch.tensor(feature_rows, dtype=torch.float32)
    return self.forward(x).tolist()


def save_checkpoint(path, payload):
  """Saves a PyTorch checkpoint safely (tensors + plain Python config/metadata)."""
  parent = os.path.dirname(os.path.abspath(path))
  os.makedirs(parent, exist_ok=True)
  torch.save(payload, path)


def load_checkpoint(path):
  """Loads a PyTorch checkpoint safely using weights_only=True (no pickle exec)."""
  return torch.load(path, map_location='cpu', weights_only=True)
