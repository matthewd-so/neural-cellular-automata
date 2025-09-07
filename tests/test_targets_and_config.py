"""Unit tests for targets, padding, and configuration."""

import numpy as np
import pytest

from morphogenesis.config import Config, load_config, save_config
from morphogenesis.targets import get_target, list_targets, pad_target


def test_all_builtin_targets_render():
    for name in list_targets():
        t = get_target(name, size=40)
        assert t.shape == (40, 40, 4)
        assert t.dtype == np.float32
        assert 0.0 <= t.min() and t.max() <= 1.0
        assert t[..., 3].sum() > 0, f"target {name} is empty"


def test_targets_are_premultiplied():
    for name in list_targets():
        t = get_target(name, size=40)
        assert np.all(t[..., :3] <= t[..., 3:4] + 1e-6), \
            f"target {name} has rgb > alpha (not premultiplied)"


def test_multiple_targets_supported():
    assert len(list_targets()) >= 5


def test_custom_image_target(tmp_path):
    from PIL import Image
    path = tmp_path / "custom.png"
    Image.new("RGBA", (64, 64), (255, 0, 0, 255)).save(path)
    t = get_target(str(path), size=32)
    assert t.shape == (32, 32, 4)
    assert t[..., 0].mean() > 0.9  # red


def test_unknown_target_raises():
    with pytest.raises(ValueError, match="Unknown target"):
        get_target("not-a-target")


def test_pad_target_centers():
    t = get_target("ring", size=40)
    padded = pad_target(t, 128)
    assert padded.shape == (128, 128, 4)
    assert np.allclose(padded[44:84, 44:84], t)
    assert padded[..., 3].sum() == pytest.approx(t[..., 3].sum())


def test_pad_target_too_small_raises():
    with pytest.raises(ValueError, match="does not fit"):
        pad_target(get_target("ring", size=40), 32)


def test_config_roundtrip(tmp_path):
    cfg = Config(target="heart", grid_size=96, iterations=123)
    path = tmp_path / "cfg.yaml"
    save_config(cfg, path)
    loaded = load_config(path)
    assert loaded == cfg


def test_config_rejects_unknown_keys(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("target: flower\nnot_a_key: 1\n")
    with pytest.raises(ValueError, match="not_a_key"):
        load_config(path)


def test_config_cell_count():
    assert Config(grid_size=128).n_cells == 16_384
