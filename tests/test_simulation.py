"""Unit tests for the simulation engine and damage operations."""

import torch

from morphogenesis.config import Config
from morphogenesis.model import NeuralCA
from morphogenesis.simulation import (
    Simulation,
    damage_circle,
    damage_keep_fragment,
    damage_random_cells,
)


def _full_state(grid: int = 64) -> torch.Tensor:
    """A fully-alive state with random content."""
    x = torch.rand(1, 16, grid, grid)
    x[:, 3] = 1.0
    return x


def test_damage_circle_zeroes_inside_only():
    x = _full_state()
    y = damage_circle(x, cx=32, cy=32, radius=10)
    assert torch.all(y[0, :, 32, 32] == 0)          # center destroyed
    assert torch.all(y[0, :, 32, 45] == x[0, :, 32, 45])  # outside untouched


def test_damage_random_cells_fraction():
    torch.manual_seed(0)
    x = _full_state(128)
    y = damage_random_cells(x, fraction=0.8)
    survivors = (y[0, 3] > 0.1).sum().item()
    total = 128 * 128
    assert abs(survivors / total - 0.2) < 0.02, "≈20% of cells should survive"
    # survivors keep their exact state
    mask = y[0, 3] > 0.1
    assert torch.allclose(y[0][:, mask], x[0][:, mask])


def test_damage_keep_fragment_is_contiguous_and_small():
    torch.manual_seed(0)
    x = _full_state(128)
    y = damage_keep_fragment(x, keep_frac=0.2)
    alive = (y[0, 3] > 0.1)
    frac = alive.sum().item() / (128 * 128)
    assert 0.05 < frac <= 0.25, f"fragment holds {frac:.0%}, expected ~20% or less"
    # contiguity: the fragment is one circle, so its bounding box is filled >75%
    rows, cols = alive.nonzero(as_tuple=True)
    box = (rows.max() - rows.min() + 1) * (cols.max() - cols.min() + 1)
    assert alive.sum() / box > 0.5


def test_damage_on_empty_grid_is_noop():
    x = torch.zeros(1, 16, 32, 32)
    assert torch.all(damage_keep_fragment(x, 0.2) == 0)
    assert torch.all(damage_random_cells(x, 0.8) == 0)


def test_simulation_lifecycle_untrained():
    cfg = Config(grid_size=64, target_size=40, device="cpu")
    sim = Simulation(NeuralCA(), cfg)
    assert sim.n_cells == 64 * 64
    assert sim.alive_count() == 1  # the seed
    sim.step(3)
    assert sim.step_count == 3
    sim.reset()
    assert sim.step_count == 0
    assert sim.alive_count() == 1
    rgb = sim.rgb()
    assert rgb.shape == (64, 64, 3)


def test_simulation_16k_cells():
    """The flagship configuration simulates more than 16,000 cells."""
    cfg = Config(grid_size=128, device="cpu")
    sim = Simulation(NeuralCA(), cfg)
    assert sim.n_cells == 16_384 > 16_000


def test_checkpoint_roundtrip(tmp_path):
    from morphogenesis.training import save_checkpoint
    cfg = Config(grid_size=64, target_size=40, device="cpu")
    model = NeuralCA()
    with torch.no_grad():
        model.update_rule[-1].weight.normal_()
    path = tmp_path / "m.pt"
    save_checkpoint(model, cfg, path)
    sim = Simulation.from_checkpoint(path, device="cpu")
    assert sim.cfg == cfg
    for p, q in zip(model.parameters(), sim.model.parameters()):
        assert torch.allclose(p, q.cpu())


def test_training_smoke(tmp_path):
    """Two iterations of the real training loop run end to end on CPU."""
    from morphogenesis.training import train
    cfg = Config(grid_size=48, target_size=32, train_grid_size=None,
                 iterations=2, pool_size=16, batch_size=4, damage_n=1,
                 severe_damage_n=1, steps_min=8, steps_max=12,
                 device="cpu", log_every=1)
    model, history = train(cfg, out_dir=tmp_path, progress=False)
    assert len(history["loss"]) == 2
    assert (tmp_path / "flower.pt").exists()
    assert all(l > 0 for l in history["loss"])
