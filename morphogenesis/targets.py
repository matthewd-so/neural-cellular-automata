"""Target patterns the cells learn to grow.

All targets are RGBA images in [0, 1] with *premultiplied* alpha (rgb * a),
rendered procedurally with PIL so the repo needs no downloaded assets.
Any external RGBA image can be used instead via `get_target(path)`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

# Procedural targets are drawn at high resolution then downsampled for
# clean anti-aliased edges.
_SUPERSAMPLE = 4


def _canvas(size: int) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    s = size * _SUPERSAMPLE
    img = Image.new("RGBA", (s, s), (0, 0, 0, 0))
    return img, ImageDraw.Draw(img), s


def _finish(img: Image.Image, size: int) -> np.ndarray:
    img = img.resize((size, size), Image.LANCZOS)
    rgba = np.asarray(img, dtype=np.float32) / 255.0
    rgba[..., :3] *= rgba[..., 3:4]  # premultiply
    return rgba


# ---------------------------------------------------------------------------
# Built-in targets
# ---------------------------------------------------------------------------

def _flower(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    cx = cy = s / 2
    petal_r, petal_d = s * 0.18, s * 0.26
    for k in range(8):
        ang = 2 * np.pi * k / 8
        px, py = cx + petal_d * np.cos(ang), cy + petal_d * np.sin(ang)
        color = (231, 84, 128, 255) if k % 2 == 0 else (246, 135, 179, 255)
        draw.ellipse([px - petal_r, py - petal_r, px + petal_r, py + petal_r], fill=color)
    core = s * 0.14
    draw.ellipse([cx - core, cy - core, cx + core, cy + core], fill=(255, 200, 60, 255))
    return _finish(img, size)


def _lizard(size: int) -> np.ndarray:
    """A gecko-ish lizard: curved body of overlapping discs, four legs, eyes."""
    img, draw, s = _canvas(size)
    cx = cy = s / 2
    green, dark = (86, 176, 80, 255), (56, 130, 56, 255)

    # body: discs along an S-curve, shrinking toward the tail
    t = np.linspace(-1.0, 1.0, 40)
    bx = cx + t * s * 0.30
    by = cy + np.sin(t * np.pi) * s * 0.14
    br = s * (0.10 - 0.065 * np.clip(t, 0, 1))  # head end thick, tail end thin
    for x, y, r in zip(bx, by, br):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=green)

    # head at the leading end
    hx, hy, hr = bx[0], by[0], s * 0.115
    draw.ellipse([hx - hr, hy - hr, hx + hr, hy + hr], fill=green)

    # legs: two pairs, splayed from the body
    for ti, spread in [(-0.55, 1.0), (0.35, 1.0)]:
        i = int((ti + 1) / 2 * (len(t) - 1))
        for side in (-1, 1):
            lx = bx[i] + side * s * 0.05
            ly = by[i] + side * s * 0.16 * spread
            draw.line([bx[i], by[i], lx, ly], fill=dark, width=int(s * 0.035))
            fr = s * 0.035
            draw.ellipse([lx - fr, ly - fr, lx + fr, ly + fr], fill=dark)

    # eyes
    for side in (-1, 1):
        ex, ey, er = hx - hr * 0.35, hy + side * hr * 0.5, s * 0.02
        draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(20, 20, 20, 255))
    return _finish(img, size)


def _heart(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    # classic parametric heart, filled as a polygon
    t = np.linspace(0, 2 * np.pi, 400)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    x = s / 2 + x / 36 * s
    y = s / 2 - y / 36 * s
    draw.polygon(list(zip(x, y)), fill=(220, 48, 68, 255))
    return _finish(img, size)


def _ring(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    cx = cy = s / 2
    r_out, r_in = s * 0.42, s * 0.24
    draw.ellipse([cx - r_out, cy - r_out, cx + r_out, cy + r_out], fill=(64, 120, 220, 255))
    draw.ellipse([cx - r_in, cy - r_in, cx + r_in, cy + r_in], fill=(0, 0, 0, 0))
    return _finish(img, size)


def _butterfly(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    cx = cy = s / 2
    orange, amber = (240, 130, 40, 255), (250, 180, 70, 255)
    for side in (-1, 1):
        # upper wing
        w = s * 0.30
        draw.ellipse([cx + side * s * 0.03 + min(side * w, 0), cy - s * 0.34,
                      cx + side * s * 0.03 + max(side * w, 0), cy + s * 0.02], fill=orange)
        # lower wing
        w2 = s * 0.22
        draw.ellipse([cx + side * s * 0.02 + min(side * w2, 0), cy - s * 0.02,
                      cx + side * s * 0.02 + max(side * w2, 0), cy + s * 0.28], fill=amber)
    # body + antennae
    draw.ellipse([cx - s * 0.035, cy - s * 0.30, cx + s * 0.035, cy + s * 0.26],
                 fill=(60, 40, 30, 255))
    for side in (-1, 1):
        draw.line([cx, cy - s * 0.28, cx + side * s * 0.10, cy - s * 0.42],
                  fill=(60, 40, 30, 255), width=int(s * 0.015))
    return _finish(img, size)


def _goose(size: int) -> np.ndarray:
    """A Canada goose (the University of Waterloo campus variety)."""
    img, draw, s = _canvas(size)
    body_brown, breast = (128, 108, 84, 255), (222, 214, 198, 255)
    black, white = (28, 28, 30, 255), (245, 245, 245, 255)

    # body: plump ellipse, goose facing left
    draw.ellipse([s * 0.28, s * 0.42, s * 0.88, s * 0.76], fill=body_brown)
    # pale breast/underside
    draw.ellipse([s * 0.30, 0.56 * s, s * 0.72, s * 0.78], fill=breast)
    # tail: black wedge at the rear
    draw.polygon([(s * 0.80, s * 0.52), (s * 0.98, s * 0.44),
                  (s * 0.94, s * 0.62), (s * 0.78, s * 0.66)], fill=black)
    # folded wing detail
    draw.ellipse([s * 0.42, s * 0.44, s * 0.84, s * 0.64],
                 fill=(108, 90, 70, 255))

    # neck: black discs along a gentle S from body to head
    t = np.linspace(0.0, 1.0, 24)
    nx = s * (0.34 - 0.06 * t + 0.02 * np.sin(t * np.pi))
    ny = s * (0.50 - 0.32 * t)
    nr = s * (0.045 - 0.008 * t)
    for x, y, r in zip(nx, ny, nr):
        draw.ellipse([x - r, y - r, x + r, y + r], fill=black)

    # head + bill
    hx, hy, hr = nx[-1], ny[-1], s * 0.055
    draw.ellipse([hx - hr * 1.35, hy - hr, hx + hr * 1.1, hy + hr], fill=black)
    draw.polygon([(hx - hr * 1.2, hy - s * 0.012), (hx - hr * 2.1, hy + s * 0.015),
                  (hx - hr * 1.2, hy + s * 0.035)], fill=black)
    # the white chinstrap
    draw.ellipse([hx - hr * 0.75, hy + hr * 0.15, hx + hr * 0.55, hy + hr * 1.05],
                 fill=white)
    # eye
    er = s * 0.011
    draw.ellipse([hx - hr * 0.6 - er, hy - hr * 0.35 - er,
                  hx - hr * 0.6 + er, hy - hr * 0.35 + er], fill=(210, 210, 210, 255))

    # legs
    for lx in (0.50, 0.62):
        draw.line([s * lx, s * 0.74, s * lx, s * 0.88], fill=black, width=int(s * 0.022))
        draw.polygon([(s * lx - s * 0.045, s * 0.90), (s * lx + s * 0.035, s * 0.90),
                      (s * lx, s * 0.855)], fill=black)
    return _finish(img, size)


def _fish(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    cx, cy = s * 0.44, s * 0.5
    blue, dark = (66, 148, 210, 255), (36, 96, 150, 255)
    draw.ellipse([cx - s * 0.30, cy - s * 0.18, cx + s * 0.30, cy + s * 0.18], fill=blue)
    # tail
    draw.polygon([(cx + s * 0.26, cy), (cx + s * 0.46, cy - s * 0.16),
                  (cx + s * 0.46, cy + s * 0.16)], fill=dark)
    # fins
    draw.polygon([(cx - s * 0.02, cy - s * 0.16), (cx + s * 0.10, cy - s * 0.30),
                  (cx + s * 0.12, cy - s * 0.14)], fill=dark)
    draw.polygon([(cx - s * 0.02, cy + s * 0.16), (cx + s * 0.10, cy + s * 0.30),
                  (cx + s * 0.12, cy + s * 0.14)], fill=dark)
    # eye
    ex, ey, er = cx - s * 0.18, cy - s * 0.05, s * 0.035
    draw.ellipse([ex - er, ey - er, ex + er, ey + er], fill=(255, 255, 255, 255))
    draw.ellipse([ex - er / 2, ey - er / 2, ex + er / 2, ey + er / 2], fill=(20, 20, 20, 255))
    return _finish(img, size)


def _maple_leaf(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    # stylized 5-lobe maple leaf: right half + mirrored left half + stem
    half = [
        (0.00, 0.92), (0.07, 0.62), (0.30, 0.74), (0.22, 0.44),
        (0.56, 0.56), (0.42, 0.24), (0.80, 0.16), (0.52, -0.06),
        (0.66, -0.36), (0.30, -0.24), (0.16, -0.44), (0.05, -0.36),
    ]
    pts = half + [(-x, y) for x, y in reversed(half[1:])]
    poly = [(s / 2 + x * s * 0.5, s / 2 - y * s * 0.5) for x, y in pts]
    draw.polygon(poly, fill=(206, 52, 40, 255))
    draw.line([s / 2, s / 2 + 0.36 * s * 0.5, s / 2, s / 2 + 0.9 * s * 0.5],
              fill=(206, 52, 40, 255), width=int(s * 0.04))
    return _finish(img, size)


def _mushroom(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    cx = s / 2
    # stem
    draw.rounded_rectangle([cx - s * 0.10, s * 0.42, cx + s * 0.10, s * 0.82],
                           radius=s * 0.08, fill=(238, 228, 210, 255))
    # cap: red dome
    draw.pieslice([cx - s * 0.34, s * 0.10, cx + s * 0.34, s * 0.78],
                  start=180, end=360, fill=(202, 46, 52, 255))
    # spots
    for (dx, dy, r) in [(-0.18, 0.32, 0.05), (0.0, 0.22, 0.06),
                        (0.18, 0.34, 0.045), (-0.05, 0.40, 0.035), (0.11, 0.42, 0.03)]:
        draw.ellipse([cx + dx * s - r * s, dy * s - r * s,
                      cx + dx * s + r * s, dy * s + r * s], fill=(250, 244, 232, 255))
    return _finish(img, size)


def _star(size: int) -> np.ndarray:
    img, draw, s = _canvas(size)
    cx = cy = s / 2
    pts = []
    for k in range(10):
        ang = -np.pi / 2 + k * np.pi / 5
        r = s * (0.46 if k % 2 == 0 else 0.20)
        pts.append((cx + r * np.cos(ang), cy + r * np.sin(ang)))
    draw.polygon(pts, fill=(250, 195, 50, 255))
    return _finish(img, size)


_BUILTINS = {
    "goose": _goose,
    "flower": _flower,
    "lizard": _lizard,
    "heart": _heart,
    "ring": _ring,
    "butterfly": _butterfly,
    "fish": _fish,
    "maple_leaf": _maple_leaf,
    "mushroom": _mushroom,
    "star": _star,
}


def list_targets() -> list[str]:
    """Built-in target names, gallery order (goose first, of course)."""
    return list(_BUILTINS)


def get_target(name: str, size: int = 56) -> np.ndarray:
    """Return an (size, size, 4) premultiplied-RGBA target in [0, 1].

    `name` is a built-in target name or a path to an image file.
    """
    if name in _BUILTINS:
        return _BUILTINS[name](size)
    path = Path(name)
    if path.exists():
        img = Image.open(path).convert("RGBA").resize((size, size), Image.LANCZOS)
        rgba = np.asarray(img, dtype=np.float32) / 255.0
        rgba[..., :3] *= rgba[..., 3:4]
        return rgba
    raise ValueError(f"Unknown target {name!r}; built-ins: {list_targets()}")


def pad_target(target: np.ndarray, grid_size: int) -> np.ndarray:
    """Center an (h, w, 4) target on a (grid_size, grid_size, 4) canvas."""
    h, w = target.shape[:2]
    if h > grid_size or w > grid_size:
        raise ValueError(f"target {h}x{w} does not fit grid {grid_size}")
    out = np.zeros((grid_size, grid_size, 4), dtype=np.float32)
    top, left = (grid_size - h) // 2, (grid_size - w) // 2
    out[top:top + h, left:left + w] = target
    return out
