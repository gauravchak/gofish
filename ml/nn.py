"""Tiny, zero-dependency 2-layer neural network in pure Python.

Why pure Python instead of PyTorch?
  1. Zero extra installs: works out of the box on any parent's or kid's laptop.
  2. Transparent math: forward pass, chain-rule backprop, and weight updates
     fit in ~100 readable lines so you can inspect every single equation.
"""

import json
import math
import os
import random


def _sigmoid(x):
  if x >= 0:
    z = math.exp(-min(x, 40.0))
    return 1.0 / (1.0 + z)
  z = math.exp(max(x, -40.0))
  return z / (1.0 + z)


def _clip(x, limit=5.0):
  if x > limit:
    return limit
  if x < -limit:
    return -limit
  return x


class TinyMLP:
  """1-hidden-layer MLP: input_dim -> hidden_dim (tanh) -> 1 output.

  Supports three output activations:
    - 'sigmoid': outputs in (0, 1)  (used by TD Value bot for win probability)
    - 'tanh':    outputs in (-1, 1) (used by Q-Learning & Critic for returns)
    - 'linear':  unbounded logit    (used by Policy Gradient Actor before softmax)
  """

  def __init__(
      self,
      input_dim,
      hidden_dim=16,
      output_activation='linear',
      seed=42,
  ):
    if output_activation not in ('linear', 'sigmoid', 'tanh'):
      raise ValueError(f'Unsupported activation: {output_activation}')
    self.input_dim = input_dim
    self.hidden_dim = hidden_dim
    self.output_activation = output_activation

    rng = random.Random(seed)
    scale1 = 1.0 / math.sqrt(max(input_dim, 1))
    scale2 = 1.0 / math.sqrt(max(hidden_dim, 1))
    self.w1 = [
        [rng.uniform(-scale1, scale1) for _ in range(input_dim)]
        for _ in range(hidden_dim)
    ]
    self.b1 = [0.0] * hidden_dim
    self.w2 = [rng.uniform(-scale2, scale2) for _ in range(hidden_dim)]
    self.b2 = 0.0

  def forward_with_cache(self, x):
    """Runs forward pass and returns (output, hidden_activations) for backprop."""
    h = []
    for j in range(self.hidden_dim):
      row = self.w1[j]
      total = self.b1[j]
      for i in range(self.input_dim):
        total += row[i] * x[i]
      h.append(math.tanh(total))

    z = self.b2
    for j in range(self.hidden_dim):
      z += self.w2[j] * h[j]

    if self.output_activation == 'sigmoid':
      out = _sigmoid(z)
    elif self.output_activation == 'tanh':
      out = math.tanh(z)
    else:
      out = z
    return out, h

  def predict(self, x):
    """Fast scalar prediction without keeping intermediate cache."""
    out, _ = self.forward_with_cache(x)
    return out

  def apply_output_gradient(self, x, h, d_loss_dz, lr=0.01, l2=1e-4):
    """Backpropagates scalar gradient dL/dz (pre-activation output) and updates weights."""
    dz = _clip(d_loss_dz, 5.0)
    w2 = self.w2
    w1 = self.w1
    b1 = self.b1

    # Save old w2 before updating so hidden gradients use consistent weights.
    old_w2 = list(w2)

    # 1. Update output layer weights (w2, b2)
    self.b2 -= lr * dz
    for j in range(self.hidden_dim):
      grad_w2 = dz * h[j] + l2 * w2[j]
      w2[j] -= lr * grad_w2

    # 2. Backprop through hidden tanh layer (dh = dz * w2 * (1 - h^2))
    for j in range(self.hidden_dim):
      hj = h[j]
      dh = _clip(dz * old_w2[j] * (1.0 - hj * hj), 5.0)
      b1[j] -= lr * dh
      row = w1[j]
      for i in range(self.input_dim):
        xi = x[i]
        if xi != 0.0:
          row[i] -= lr * (dh * xi + l2 * row[i])
        elif l2 > 0.0:
          row[i] -= lr * l2 * row[i]

  def train_step(self, x, target, lr=0.01, l2=1e-4):
    """Single SGD update toward scalar target; returns squared error."""
    out, h = self.forward_with_cache(x)
    err = out - target
    if self.output_activation == 'sigmoid':
      # Binary cross-entropy gradient w.r.t. logit z simplifies cleanly to (out - target)!
      d_loss_dz = err
    elif self.output_activation == 'tanh':
      # MSE gradient w.r.t. z: (out - target) * (1 - out^2)
      d_loss_dz = err * (1.0 - out * out)
    else:
      d_loss_dz = err

    self.apply_output_gradient(x, h, d_loss_dz, lr=lr, l2=l2)
    return err * err

  def copy_weights_from(self, other, tau=1.0):
    """Copies weights from another TinyMLP (tau=1.0 hard copy, tau<1.0 soft DQN update)."""
    one_m = 1.0 - tau
    self.b2 = one_m * self.b2 + tau * other.b2
    for j in range(self.hidden_dim):
      self.w2[j] = one_m * self.w2[j] + tau * other.w2[j]
      self.b1[j] = one_m * self.b1[j] + tau * other.b1[j]
      for i in range(self.input_dim):
        self.w1[j][i] = one_m * self.w1[j][i] + tau * other.w1[j][i]

  def to_dict(self):
    return {
        'input_dim': self.input_dim,
        'hidden_dim': self.hidden_dim,
        'output_activation': self.output_activation,
        'w1': [[round(v, 6) for v in row] for row in self.w1],
        'b1': [round(v, 6) for v in self.b1],
        'w2': [round(v, 6) for v in self.w2],
        'b2': round(self.b2, 6),
    }

  @classmethod
  def from_dict(cls, data):
    net = cls(
        input_dim=data['input_dim'],
        hidden_dim=data['hidden_dim'],
        output_activation=data['output_activation'],
    )
    net.w1 = [[float(v) for v in row] for row in data['w1']]
    net.b1 = [float(v) for v in data['b1']]
    net.w2 = [float(v) for v in data['w2']]
    net.b2 = float(data['b2'])
    return net

  def save(self, path, metadata=None):
    parent = os.path.dirname(os.path.abspath(path))
    os.makedirs(parent, exist_ok=True)
    payload = {'network': self.to_dict(), 'metadata': metadata or {}}
    with open(path, 'w', encoding='utf-8') as f:
      json.dump(payload, f, indent=2)
      f.write('\n')

  @classmethod
  def load(cls, path):
    with open(path, 'r', encoding='utf-8') as f:
      payload = json.load(f)
    return cls.from_dict(payload['network']), payload.get('metadata', {})
