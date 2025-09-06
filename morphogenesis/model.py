"""The Neural Cellular Automaton update rule.

Every cell holds a `channels`-dimensional state vector. Channels 0-3 are
visible RGBA (alpha doubles as the "alive" signal); the rest are hidden
channels the cells are free to use as chemical signals.

One CA step:
  1. Perception  — each cell concatenates its state with Sobel-x / Sobel-y
                   estimates of the local state gradient (fixed, not learned).
  2. Update rule — a small per-cell MLP (two 1x1 convolutions) maps the
                   perception vector to a state delta. The final layer is
                   zero-initialized so the untrained CA is the identity.
  3. Stochastic update — each cell applies its delta with prob `fire_rate`,
                   breaking global synchrony.
  4. Alive masking — cells with no mature neighbor (alpha > threshold in a
                   3x3 neighborhood) are zeroed: growth can only proceed
                   outward from living tissue.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _perception_kernels(channels: int) -> torch.Tensor:
    """Fixed depthwise kernels: identity, Sobel-x, Sobel-y for every channel."""
    ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
    sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]) / 8.0
    sobel_y = sobel_x.T
    kernels = torch.stack([ident, sobel_x, sobel_y])          # (3, 3, 3)
    kernels = kernels.repeat(channels, 1, 1)                  # (3*C, 3, 3)
    return kernels.unsqueeze(1)                               # (3*C, 1, 3, 3)


class NeuralCA(nn.Module):
    """Neural Cellular Automaton with Sobel perception and alive masking."""

    def __init__(
        self,
        channels: int = 16,
        hidden_dim: int = 128,
        fire_rate: float = 0.5,
        alive_threshold: float = 0.1,
    ):
        super().__init__()
        self.channels = channels
        self.fire_rate = fire_rate
        self.alive_threshold = alive_threshold

        self.register_buffer("perception_kernels", _perception_kernels(channels))

        self.update_rule = nn.Sequential(
            nn.Conv2d(channels * 3, hidden_dim, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, channels, kernel_size=1, bias=False),
        )
        # Zero-init the last layer: before training, a step changes nothing.
        nn.init.zeros_(self.update_rule[-1].weight)

    # ------------------------------------------------------------------
    def perceive(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, H, W) -> (B, 3C, H, W) perception vector per cell."""
        return F.conv2d(x, self.perception_kernels, padding=1, groups=self.channels)

    def alive_mask(self, x: torch.Tensor) -> torch.Tensor:
        """A cell is alive if any cell in its 3x3 neighborhood has alpha > threshold."""
        alpha = x[:, 3:4]
        return F.max_pool2d(alpha, kernel_size=3, stride=1, padding=1) > self.alive_threshold

    def forward(self, x: torch.Tensor, fire_rate: float | None = None) -> torch.Tensor:
        """One CA step."""
        pre_alive = self.alive_mask(x)

        dx = self.update_rule(self.perceive(x))

        rate = self.fire_rate if fire_rate is None else fire_rate
        update_mask = torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device) <= rate
        x = x + dx * update_mask

        post_alive = self.alive_mask(x)
        return x * (pre_alive & post_alive).to(x.dtype)

    def steps(self, x: torch.Tensor, n: int, fire_rate: float | None = None) -> torch.Tensor:
        """Run `n` CA steps."""
        for _ in range(n):
            x = self.forward(x, fire_rate)
        return x

    # ------------------------------------------------------------------
    @staticmethod
    def seed(grid_size: int, channels: int = 16, batch: int = 1,
             device: str | torch.device = "cpu") -> torch.Tensor:
        """A dead grid with a single living cell in the center."""
        x = torch.zeros(batch, channels, grid_size, grid_size, device=device)
        x[:, 3:, grid_size // 2, grid_size // 2] = 1.0
        return x

    @staticmethod
    def to_rgb(x: torch.Tensor) -> torch.Tensor:
        """Composite premultiplied RGBA state onto a white background: (B,3,H,W) in [0,1]."""
        rgb, a = x[:, :3], x[:, 3:4].clamp(0.0, 1.0)
        return (1.0 - a + rgb).clamp(0.0, 1.0)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
