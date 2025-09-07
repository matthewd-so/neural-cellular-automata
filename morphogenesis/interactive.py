"""Interactive damage/regeneration demo.

Opens a matplotlib window running the CA live:

  * click (or drag)  — blast a hole where you click
  * D                — destroy 80% of living cells at random
  * F                — wipe everything but one ~20% fragment
  * R                — reset to a single seed
  * space            — pause / resume
  * Q                — quit

The title bar shows step count, living-cell count, and live SSIM.
"""

from __future__ import annotations

import matplotlib.pyplot as plt

from morphogenesis.simulation import Simulation

BLAST_RADIUS_FRAC = 0.12  # click blast radius, as a fraction of grid size
STEPS_PER_FRAME = 2


def run_demo(checkpoint: str, device: str | None = None) -> None:
    sim = Simulation.from_checkpoint(checkpoint, device=device)
    state = {"paused": False, "mouse_down": False}

    fig, ax = plt.subplots(figsize=(6, 6))
    fig.canvas.manager.set_window_title("Morphogenesis Engine")
    im = ax.imshow(sim.rgb(), interpolation="nearest")
    ax.set_axis_off()

    def blast_at(event) -> None:
        if event.inaxes is ax and event.xdata is not None:
            sim.blast(event.xdata, event.ydata, sim.cfg.grid_size * BLAST_RADIUS_FRAC)

    def on_press(event) -> None:
        state["mouse_down"] = True
        blast_at(event)

    def on_release(_) -> None:
        state["mouse_down"] = False

    def on_move(event) -> None:
        if state["mouse_down"]:
            blast_at(event)

    def on_key(event) -> None:
        if event.key == "r":
            sim.reset()
        elif event.key == "d":
            sim.destroy_fraction(0.8)
        elif event.key == "f":
            sim.keep_fragment(0.2)
        elif event.key == " ":
            state["paused"] = not state["paused"]
        elif event.key == "q":
            plt.close(fig)

    fig.canvas.mpl_connect("button_press_event", on_press)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_move)
    fig.canvas.mpl_connect("key_press_event", on_key)

    timer = fig.canvas.new_timer(interval=30)

    def tick() -> None:
        if not state["paused"]:
            sim.step(STEPS_PER_FRAME)
        im.set_data(sim.rgb())
        ax.set_title(
            f"step {sim.step_count}   alive {sim.alive_count()}/{sim.n_cells}   "
            f"SSIM {sim.ssim():.3f}",
            fontsize=10,
        )
        fig.canvas.draw_idle()

    timer.add_callback(tick)
    timer.start()
    plt.show()
