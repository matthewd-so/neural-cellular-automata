"""Pool-based training with a damage curriculum.

The training recipe that makes regeneration work:

* A persistent **sample pool** of grid states, initialized to seeds. Each
  iteration trains on a random batch from the pool and writes the results
  back, so the CA learns to *persist* patterns over long horizons, not just
  reach them once.
* The worst sample in each batch is replaced with a fresh seed (so growing
  from scratch is never forgotten).
* A **damage curriculum**: some of the best samples are damaged before
  training (circular blasts, plus a severe variant that keeps only a small
  fragment), so the CA explicitly learns to regenerate from the kind of
  destruction we evaluate on.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from morphogenesis.config import Config
from morphogenesis.model import NeuralCA
from morphogenesis.simulation import (
    damage_circle,
    damage_keep_fragment,
    resolve_device,
)
from morphogenesis.targets import get_target, pad_target


class SamplePool:
    """Persistent pool of CA states stored on the training device."""

    def __init__(self, seed_state: torch.Tensor, size: int):
        self.states = seed_state.repeat(size, 1, 1, 1)

    def sample(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        idx = torch.randperm(self.states.shape[0])[:batch_size]
        return self.states[idx].clone(), idx

    def commit(self, idx: torch.Tensor, states: torch.Tensor) -> None:
        self.states[idx] = states.detach()


def _batch_loss(x: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Per-sample MSE between visible RGBA channels and the target."""
    return ((x[:, :4] - target) ** 2).mean(dim=(1, 2, 3))


def train(cfg: Config, out_dir: str | Path | None = None,
          progress: bool = True) -> tuple[NeuralCA, dict]:
    """Train a NeuralCA on cfg's target. Returns (model, history)."""
    device = resolve_device(cfg.device)
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)

    out_dir = Path(out_dir if out_dir is not None else cfg.checkpoint_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = out_dir / f"{Path(cfg.target).stem}.pt"

    # Training runs on a cropped grid (translation invariance means the
    # trained rule transfers unchanged to the full cfg.grid_size grid).
    grid = cfg.train_grid_size or cfg.grid_size
    target_np = pad_target(get_target(cfg.target, cfg.target_size), grid)
    target = torch.from_numpy(target_np).permute(2, 0, 1).unsqueeze(0).to(device)
    target_batch = target.repeat(cfg.batch_size, 1, 1, 1)

    model = NeuralCA(cfg.channels, cfg.hidden_dim, cfg.fire_rate, cfg.alive_threshold).to(device)
    seed = NeuralCA.seed(grid, cfg.channels, device=device)
    pool = SamplePool(seed, cfg.pool_size)

    optim = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    sched = torch.optim.lr_scheduler.MultiStepLR(
        optim, milestones=[cfg.lr_decay_at], gamma=cfg.lr_decay)

    history: dict = {"loss": [], "config": cfg.to_dict()}
    t0 = time.time()

    for it in range(1, cfg.iterations + 1):
        x, idx = pool.sample(cfg.batch_size)

        # rank batch by current loss (descending: worst first)
        with torch.no_grad():
            order = _batch_loss(x, target_batch).argsort(descending=True)
        x, idx = x[order], idx[order.cpu()]

        # worst sample -> fresh seed, so growth from scratch stays learned
        x[0] = seed[0]

        # damage curriculum on the best samples (end of the ranked batch)
        n_dmg, n_sev = cfg.damage_n, cfg.severe_damage_n
        if n_dmg > 0:
            sl = x[-(n_dmg + n_sev):len(x) - n_sev or None]
            for i in range(sl.shape[0]):
                r = np.random.uniform(grid * 0.1, grid * 0.25)
                cx = np.random.uniform(grid * 0.25, grid * 0.75)
                cy = np.random.uniform(grid * 0.25, grid * 0.75)
                sl[i:i + 1] = damage_circle(sl[i:i + 1], cx, cy, r)
        if n_sev > 0:
            keep = np.random.uniform(cfg.severe_keep_frac * 0.75, cfg.severe_keep_frac * 1.5)
            x[-n_sev:] = damage_keep_fragment(x[-n_sev:], keep)

        # unroll the CA and regress the visible channels onto the target
        n_steps = int(np.random.randint(cfg.steps_min, cfg.steps_max + 1))
        x = model.steps(x, n_steps)
        loss = _batch_loss(x, target_batch).mean()

        optim.zero_grad()
        loss.backward()
        if cfg.grad_clip:  # per-parameter gradient normalization
            for p in model.parameters():
                if p.grad is not None:
                    p.grad /= p.grad.norm() + 1e-8
        optim.step()
        sched.step()

        pool.commit(idx, x)
        history["loss"].append(float(loss.item()))

        if progress and it % cfg.log_every == 0:
            recent = float(np.mean(history["loss"][-cfg.log_every:]))
            print(f"iter {it:5d}/{cfg.iterations}  loss {recent:.5f}  "
                  f"lr {sched.get_last_lr()[0]:.1e}  {time.time() - t0:6.0f}s",
                  flush=True)
        if it % 1000 == 0 or it == cfg.iterations:
            save_checkpoint(model, cfg, ckpt_path, history)

    return model, history


def save_checkpoint(model: NeuralCA, cfg: Config, path: str | Path,
                    history: dict | None = None) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "config": cfg.to_dict()}, path)
    if history is not None:
        with open(path.with_suffix(".history.json"), "w") as f:
            json.dump({"loss": history["loss"], "config": history["config"]}, f)
