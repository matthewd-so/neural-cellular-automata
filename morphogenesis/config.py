"""Experiment configuration: a single dataclass, loadable from YAML."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Config:
    """Everything that defines a morphogenesis experiment."""

    # --- target ---------------------------------------------------------
    target: str = "flower"          # built-in target name, or path to an RGBA image
    target_size: int = 40           # pattern is rendered at target_size x target_size
    grid_size: int = 128            # simulation grid (grid_size^2 cells)
    # The update rule is translation-invariant, so training can run on a
    # smaller cropped grid (the pattern plus padding) and the trained CA
    # transfers unchanged to the full simulation grid. None = grid_size.
    train_grid_size: int | None = 72

    # --- model ----------------------------------------------------------
    channels: int = 16              # state channels (RGBA + hidden)
    hidden_dim: int = 128           # width of the update-rule MLP
    fire_rate: float = 0.5          # per-cell stochastic update probability
    alive_threshold: float = 0.1    # alpha threshold for the alive mask

    # --- training -------------------------------------------------------
    pool_size: int = 1024           # persistent sample pool
    batch_size: int = 8
    steps_min: int = 64             # CA steps per training iteration (sampled)
    steps_max: int = 96
    iterations: int = 8000
    lr: float = 2e-3
    lr_decay_at: int = 2000         # iteration at which lr decays
    lr_decay: float = 0.1
    grad_clip: bool = True          # normalize gradients per-parameter

    # damage curriculum: how many of each batch get damaged before training
    damage_n: int = 3               # samples per batch hit with circular damage
    severe_damage_n: int = 1        # samples per batch reduced to a small fragment
    severe_keep_frac: float = 0.2   # fraction of the pattern the fragment keeps

    # --- bookkeeping ----------------------------------------------------
    seed: int = 0
    device: str = "auto"            # auto | cpu | cuda | mps
    checkpoint_dir: str = "checkpoints"
    log_every: int = 100

    @property
    def n_cells(self) -> int:
        return self.grid_size * self.grid_size

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Config":
        known = {f.name for f in dataclasses.fields(cls)}
        unknown = set(d) - known
        if unknown:
            raise ValueError(f"Unknown config keys: {sorted(unknown)}")
        return cls(**d)


def load_config(path: str | Path) -> Config:
    """Load a Config from a YAML file; missing keys fall back to defaults."""
    with open(path) as f:
        data = yaml.safe_load(f) or {}
    return Config.from_dict(data)


def save_config(cfg: Config, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.safe_dump(cfg.to_dict(), f, sort_keys=False)
