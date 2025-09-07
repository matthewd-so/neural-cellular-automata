"""Unit tests for the NCA update rule (no trained weights needed)."""

import torch

from morphogenesis.model import NeuralCA


def test_seed_shape_and_content():
    x = NeuralCA.seed(64, channels=16, batch=2)
    assert x.shape == (2, 16, 64, 64)
    # exactly one living cell, at the center, alpha + hidden set to 1
    assert (x[:, 3] > 0).sum() == 2
    assert x[0, 3, 32, 32] == 1.0
    assert x[0, :3, 32, 32].sum() == 0.0  # RGB starts at zero


def test_step_preserves_shape():
    model = NeuralCA()
    x = NeuralCA.seed(48)
    assert model(x).shape == x.shape


def test_zero_init_is_identity_where_alive():
    """Untrained CA: the update delta is zero, so a step only applies alive masking."""
    torch.manual_seed(0)
    model = NeuralCA()
    x = NeuralCA.seed(32)
    y = model(x)
    assert torch.allclose(x, y), "zero-initialized update rule must not change the seed"


def test_dead_grid_stays_dead():
    """No spontaneous generation: an all-dead grid must remain all zeros."""
    model = NeuralCA()
    with torch.no_grad():
        model.update_rule[0].bias.fill_(1.0)  # even with a bias pushing updates
        model.update_rule[-1].weight.normal_()
    x = torch.zeros(1, 16, 32, 32)
    y = model.steps(x, 5)
    assert torch.all(y == 0), "cells with no living neighbors must stay dead"


def test_alive_mask_neighborhood():
    model = NeuralCA(alive_threshold=0.1)
    x = torch.zeros(1, 16, 9, 9)
    x[0, 3, 4, 4] = 1.0
    mask = model.alive_mask(x)[0, 0]
    # the 3x3 neighborhood of the living cell is alive, everything else dead
    assert mask[3:6, 3:6].all()
    assert mask.sum() == 9


def test_perception_identity_and_sobel():
    model = NeuralCA(channels=16)
    x = torch.rand(1, 16, 16, 16)
    p = model.perceive(x)
    assert p.shape == (1, 48, 16, 16)
    # every 3rd perception channel is the identity kernel -> equals the input
    assert torch.allclose(p[:, 0::3], x, atol=1e-6)
    # Sobel responses on a constant image are zero (interior; borders zero-padded)
    const = torch.ones(1, 16, 16, 16)
    pc = model.perceive(const)
    assert torch.allclose(pc[:, 1::3][..., 1:-1, 1:-1],
                          torch.zeros(1, 16, 14, 14), atol=1e-6)


def test_stochastic_update_masks_cells():
    """With fire_rate 0 nothing updates even with a non-trivial rule."""
    torch.manual_seed(0)
    model = NeuralCA()
    with torch.no_grad():
        model.update_rule[-1].weight.normal_(std=0.1)
    x = NeuralCA.seed(32)
    y = model(x, fire_rate=0.0)
    assert torch.allclose(x, y)
    z = model(x, fire_rate=1.0)
    assert not torch.allclose(x, z)


def test_parameter_count_is_small():
    """The whole organism's genome is a tiny network."""
    model = NeuralCA(channels=16, hidden_dim=128)
    assert model.num_parameters() == 16 * 3 * 128 + 128 + 128 * 16  # 8_320


def test_to_rgb_range():
    x = torch.rand(2, 16, 8, 8)
    rgb = NeuralCA.to_rgb(x)
    assert rgb.shape == (2, 3, 8, 8)
    assert rgb.min() >= 0.0 and rgb.max() <= 1.0
