"""Evaluation metrics for grown/regenerated patterns."""

from __future__ import annotations

import numpy as np
import torch
from skimage.metrics import structural_similarity

from morphogenesis.model import NeuralCA


def state_to_rgb_array(x: torch.Tensor) -> np.ndarray:
    """(1, C, H, W) state -> (H, W, 3) float RGB image on white background."""
    rgb = NeuralCA.to_rgb(x.detach().cpu())[0]
    return rgb.permute(1, 2, 0).numpy()


def target_to_rgb_array(target_rgba: np.ndarray) -> np.ndarray:
    """(H, W, 4) premultiplied RGBA target -> (H, W, 3) RGB on white background."""
    rgb, a = target_rgba[..., :3], target_rgba[..., 3:4]
    return np.clip(1.0 - a + rgb, 0.0, 1.0)


def ssim(image: np.ndarray, reference: np.ndarray) -> float:
    """Structural similarity between two (H, W, 3) float images in [0, 1].

    Uses scikit-image's reference implementation (Wang et al. 2004).
    """
    return float(
        structural_similarity(image, reference, channel_axis=-1, data_range=1.0)
    )


def state_ssim(x: torch.Tensor, target_rgba: np.ndarray) -> float:
    """SSIM between a CA state and the (padded) RGBA target."""
    return ssim(state_to_rgb_array(x), target_to_rgb_array(target_rgba))


def mse(x: torch.Tensor, target_rgba: np.ndarray) -> float:
    """Mean squared RGBA error between a CA state and the target."""
    tgt = torch.from_numpy(target_rgba).permute(2, 0, 1).unsqueeze(0)
    return float(((x[:, :4].detach().cpu() - tgt) ** 2).mean())
