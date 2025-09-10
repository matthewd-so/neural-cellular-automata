"""The evaluation protocol behind the headline numbers.

For a trained checkpoint this measures, over `n_trials` independent runs:

  1. **Growth**: grow from a single seed for 96 steps; SSIM vs. target.
  2. **Random 80% destruction**: destroy 80% of living cells uniformly at
     random, run 200 recovery steps; SSIM vs. target.
  3. **Fragment regeneration**: keep only one contiguous fragment holding
     ~20% of the living cells (i.e. 80% destroyed), run 300 recovery steps;
     SSIM vs. target.

All SSIM values use scikit-image's reference implementation over the full
grid rendered on a white background.
"""

from __future__ import annotations

import statistics
from pathlib import Path

from morphogenesis.simulation import Simulation


def evaluate(checkpoint: str | Path, n_trials: int = 10,
             device: str | None = None) -> dict:
    sim = Simulation.from_checkpoint(checkpoint, device=device)

    growth, random80, fragment = [], [], []
    for trial in range(n_trials):
        # 1. growth from a single seed
        sim.reset()
        sim.grow(96)
        growth.append(sim.ssim())

        # 2. random 80% destruction -> recovery
        sim.destroy_fraction(0.8, rng_seed=1000 + trial)
        sim.step(200)
        random80.append(sim.ssim())

        # 3. regrow, then keep a single ~20% fragment -> recovery
        sim.reset()
        sim.grow(96)
        sim.keep_fragment(0.2, rng_seed=2000 + trial)
        sim.step(300)
        fragment.append(sim.ssim())

    def stats(xs: list[float]) -> dict:
        return {
            "mean": round(statistics.mean(xs), 4),
            "min": round(min(xs), 4),
            "max": round(max(xs), 4),
            "trials": [round(x, 4) for x in xs],
        }

    return {
        "checkpoint": str(checkpoint),
        "target": sim.cfg.target,
        "grid_size": sim.cfg.grid_size,
        "n_cells": sim.n_cells,
        "model_parameters": sim.model.num_parameters(),
        "n_trials": n_trials,
        "ssim_growth": stats(growth),
        "ssim_after_80pct_random_destruction": stats(random80),
        "ssim_after_80pct_fragment_only": stats(fragment),
    }
