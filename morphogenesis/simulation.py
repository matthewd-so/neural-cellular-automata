"""The simulation engine: a stateful grid of cells plus damage operations.

`Simulation` wraps a trained (or fresh) `NeuralCA` with a persistent grid
state, exposes step/damage/measure operations, and is what the CLI and the
interactive demo drive.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from morphogenesis.config import Config
from morphogenesis.metrics import state_ssim, state_to_rgb_array
from morphogenesis.model import NeuralCA
from morphogenesis.targets import get_target, pad_target


def resolve_device(device: str = "auto") -> torch.device:
    if device != "auto":
        return torch.device(device)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Damage operations (all act on (B, C, H, W) states, zeroing cell state)
# ---------------------------------------------------------------------------

def circle_mask(grid_size: int, cx: float, cy: float, radius: float,
                device: torch.device | str = "cpu") -> torch.Tensor:
    """(1, 1, H, W) boolean mask, True inside the circle."""
    yy, xx = torch.meshgrid(
        torch.arange(grid_size, device=device, dtype=torch.float32),
        torch.arange(grid_size, device=device, dtype=torch.float32),
        indexing="ij",
    )
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= radius ** 2)[None, None]


def damage_circle(x: torch.Tensor, cx: float, cy: float, radius: float) -> torch.Tensor:
    """Zero all cell state inside a circle (a localized blast)."""
    mask = circle_mask(x.shape[-1], cx, cy, radius, x.device)
    return x * (~mask).to(x.dtype)


def damage_random_cells(x: torch.Tensor, fraction: float,
                        generator: torch.Generator | None = None) -> torch.Tensor:
    """Destroy `fraction` of the *living* cells, chosen uniformly at random.

    Returns the damaged state. The surviving (1 - fraction) of cells keep
    their full state; destroyed cells are zeroed.
    """
    alive = x[:, 3:4] > 0.1
    noise = torch.rand(x.shape[0], 1, *x.shape[2:], device=x.device, generator=generator)
    kill = alive & (noise < fraction)
    return x * (~kill).to(x.dtype)


def damage_keep_fragment(x: torch.Tensor, keep_frac: float,
                         generator: torch.Generator | None = None) -> torch.Tensor:
    """Destroy everything except one contiguous circular fragment.

    The fragment is centered on a random living cell and sized so that it
    retains roughly `keep_frac` of the living cells.
    """
    out = torch.empty_like(x)
    for b in range(x.shape[0]):
        alive = (x[b, 3] > 0.1).nonzero()
        if len(alive) == 0:
            out[b] = x[b]
            continue
        idx = torch.randint(len(alive), (1,), device=x.device, generator=generator).item()
        cy, cx = alive[idx].tolist()
        # area of fragment ≈ keep_frac * alive area  =>  r = sqrt(keep * n / pi)
        radius = float(np.sqrt(keep_frac * len(alive) / np.pi))
        keep = circle_mask(x.shape[-1], cx, cy, radius, x.device)
        out[b] = x[b] * keep[0].to(x.dtype)
    return out


# ---------------------------------------------------------------------------

class Simulation:
    """A live morphogenesis simulation: one grid, one model, one target."""

    def __init__(self, model: NeuralCA, cfg: Config,
                 device: torch.device | str | None = None):
        self.cfg = cfg
        self.device = torch.device(device) if device is not None else resolve_device(cfg.device)
        self.model = model.to(self.device).eval()
        self.target = pad_target(get_target(cfg.target, cfg.target_size), cfg.grid_size)
        self.reset()

    # -- lifecycle ------------------------------------------------------
    def reset(self) -> None:
        """Back to a single seed cell."""
        self.x = NeuralCA.seed(self.cfg.grid_size, self.cfg.channels, device=self.device)
        self.step_count = 0

    @torch.no_grad()
    def step(self, n: int = 1) -> None:
        self.x = self.model.steps(self.x, n)
        self.step_count += n

    def grow(self, steps: int = 96) -> None:
        """Convenience: run the typical growth horizon."""
        self.step(steps)

    # -- damage ---------------------------------------------------------
    @torch.no_grad()
    def blast(self, cx: float, cy: float, radius: float) -> None:
        """Localized circular damage at (cx, cy) in grid coordinates."""
        self.x = damage_circle(self.x, cx, cy, radius)

    @torch.no_grad()
    def destroy_fraction(self, fraction: float, rng_seed: int | None = None) -> None:
        """Destroy a fraction of living cells uniformly at random."""
        gen = None
        if rng_seed is not None:
            gen = torch.Generator(device=self.x.device).manual_seed(rng_seed)
        self.x = damage_random_cells(self.x, fraction, gen)

    @torch.no_grad()
    def keep_fragment(self, keep_frac: float, rng_seed: int | None = None) -> None:
        """Destroy everything but one contiguous fragment."""
        gen = None
        if rng_seed is not None:
            gen = torch.Generator(device=self.x.device).manual_seed(rng_seed)
        self.x = damage_keep_fragment(self.x, keep_frac, gen)

    # -- measurement ----------------------------------------------------
    @property
    def n_cells(self) -> int:
        """Total number of cells being simulated."""
        return self.cfg.n_cells

    def alive_count(self) -> int:
        """Number of currently living cells (alpha > 0.1)."""
        return int((self.x[0, 3] > 0.1).sum().item())

    def ssim(self) -> float:
        """SSIM of the current state against the target."""
        return state_ssim(self.x, self.target)

    def rgb(self) -> np.ndarray:
        """(H, W, 3) float RGB rendering of the current state."""
        return state_to_rgb_array(self.x)

    # -- persistence ----------------------------------------------------
    @classmethod
    def from_checkpoint(cls, path: str | Path,
                        device: torch.device | str | None = None) -> "Simulation":
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        cfg = Config.from_dict(ckpt["config"])
        model = NeuralCA(cfg.channels, cfg.hidden_dim, cfg.fire_rate, cfg.alive_threshold)
        model.load_state_dict(ckpt["model"])
        return cls(model, cfg, device=device)
