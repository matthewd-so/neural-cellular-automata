"""Command-line interface for the morphogenesis engine.

    python -m morphogenesis train  --config configs/flower.yaml
    python -m morphogenesis grow   --checkpoint checkpoints/flower.pt
    python -m morphogenesis regen  --checkpoint checkpoints/flower.pt --destroy 0.8
    python -m morphogenesis eval   --checkpoint checkpoints/flower.pt
    python -m morphogenesis demo   --checkpoint checkpoints/flower.pt
    python -m morphogenesis targets
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from morphogenesis.config import Config, load_config
from morphogenesis.simulation import Simulation
from morphogenesis.targets import list_targets


def _save_png(rgb: np.ndarray, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(path)
    print(f"wrote {path}")


def _save_gif(frames: list[np.ndarray], path: str | Path, fps: int = 20) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imgs = [Image.fromarray((np.clip(f, 0, 1) * 255).astype(np.uint8)) for f in frames]
    imgs[0].save(path, save_all=True, append_images=imgs[1:],
                 duration=int(1000 / fps), loop=0)
    print(f"wrote {path}")


# ---------------------------------------------------------------------------

def cmd_train(args: argparse.Namespace) -> None:
    cfg = load_config(args.config) if args.config else Config()
    if args.target:
        cfg.target = args.target
    if args.iterations:
        cfg.iterations = args.iterations
    from morphogenesis.training import train  # deferred: heavy import
    print(f"training target={cfg.target!r} grid={cfg.grid_size} "
          f"({cfg.n_cells} cells) for {cfg.iterations} iterations")
    train(cfg)


def cmd_grow(args: argparse.Namespace) -> None:
    sim = Simulation.from_checkpoint(args.checkpoint, device=args.device)
    frames = [sim.rgb()]
    for _ in range(args.steps):
        sim.step()
        frames.append(sim.rgb())
    print(f"cells={sim.n_cells} alive={sim.alive_count()} "
          f"steps={sim.step_count} ssim={sim.ssim():.4f}")
    if args.gif:
        _save_gif(frames, args.gif)
    _save_png(sim.rgb(), args.out)


def cmd_regen(args: argparse.Namespace) -> None:
    sim = Simulation.from_checkpoint(args.checkpoint, device=args.device)
    sim.grow(args.steps)
    print(f"grown:     ssim={sim.ssim():.4f} alive={sim.alive_count()}")
    frames = [sim.rgb()]

    if args.fragment:
        sim.keep_fragment(1.0 - args.destroy, rng_seed=args.rng_seed)
    else:
        sim.destroy_fraction(args.destroy, rng_seed=args.rng_seed)
    print(f"damaged:   ssim={sim.ssim():.4f} alive={sim.alive_count()} "
          f"({args.destroy:.0%} destroyed)")
    frames.append(sim.rgb())

    for _ in range(args.recover_steps):
        sim.step()
        frames.append(sim.rgb())
    print(f"recovered: ssim={sim.ssim():.4f} alive={sim.alive_count()} "
          f"after {args.recover_steps} steps")
    if args.gif:
        _save_gif(frames, args.gif)
    _save_png(sim.rgb(), args.out)


def cmd_eval(args: argparse.Namespace) -> None:
    """Full evaluation protocol; prints a JSON report."""
    from morphogenesis.evaluate import evaluate
    report = evaluate(args.checkpoint, n_trials=args.trials, device=args.device,
                      seed=args.seed)
    print(json.dumps(report, indent=2))
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w") as f:
            json.dump(report, f, indent=2)
        print(f"wrote {args.out}")


def cmd_demo(args: argparse.Namespace) -> None:
    from morphogenesis.interactive import run_demo
    run_demo(args.checkpoint, device=args.device)


def cmd_targets(_: argparse.Namespace) -> None:
    for name in list_targets():
        print(name)


# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="morphogenesis",
                                description="Self-organizing morphogenesis engine (NCA)")
    sub = p.add_subparsers(dest="command", required=True)

    def add_common(sp: argparse.ArgumentParser) -> None:
        sp.add_argument("--checkpoint", required=True)
        sp.add_argument("--device", default=None, help="cpu | cuda | mps (default: auto)")

    t = sub.add_parser("train", help="train a CA on a target")
    t.add_argument("--config", help="YAML config (defaults used if omitted)")
    t.add_argument("--target", help="override config target")
    t.add_argument("--iterations", type=int, help="override config iterations")
    t.set_defaults(func=cmd_train)

    g = sub.add_parser("grow", help="grow the pattern from a single seed")
    add_common(g)
    g.add_argument("--steps", type=int, default=96)
    g.add_argument("--out", default="out/grow.png")
    g.add_argument("--gif", help="also write a growth GIF")
    g.set_defaults(func=cmd_grow)

    r = sub.add_parser("regen", help="grow, damage, and regenerate")
    add_common(r)
    r.add_argument("--steps", type=int, default=96, help="growth steps before damage")
    r.add_argument("--destroy", type=float, default=0.8,
                   help="fraction of living cells to destroy")
    r.add_argument("--fragment", action="store_true",
                   help="keep one contiguous fragment instead of random survivors")
    r.add_argument("--recover-steps", type=int, default=200)
    r.add_argument("--rng-seed", type=int, default=None)
    r.add_argument("--out", default="out/regen.png")
    r.add_argument("--gif", help="also write a regeneration GIF")
    r.set_defaults(func=cmd_regen)

    e = sub.add_parser("eval", help="run the full evaluation protocol")
    add_common(e)
    e.add_argument("--trials", type=int, default=10)
    e.add_argument("--seed", type=int, default=0,
                   help="seed for the CA dynamics (damage draws are seeded per trial)")
    e.add_argument("--out", help="write JSON report here")
    e.set_defaults(func=cmd_eval)

    d = sub.add_parser("demo", help="interactive demo: click to damage, watch it regrow")
    add_common(d)
    d.set_defaults(func=cmd_demo)

    sub.add_parser("targets", help="list built-in targets").set_defaults(func=cmd_targets)
    return p


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
