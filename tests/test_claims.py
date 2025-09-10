"""Integration tests against the trained flagship checkpoint.

These verify the headline claims end to end:

  * 16,000+ cells in the simulation grid
  * a target grows from a single seed cell
  * after 80% of the pattern is destroyed, the CA regenerates the whole
    image back to >= 0.90 SSIM, for random destruction and for
    single-surviving-fragment destruction
  * the pattern is stable (persists) over long horizons

They are skipped automatically if the checkpoint has not been trained yet.
"""

import pytest

pytestmark = pytest.mark.trained

GROWTH_STEPS = 96
RANDOM_RECOVERY_STEPS = 200
FRAGMENT_RECOVERY_STEPS = 300
SSIM_THRESHOLD = 0.90


def test_simulates_more_than_16000_cells(flagship_sim):
    assert flagship_sim.n_cells >= 16_000
    assert flagship_sim.n_cells == 16_384  # 128 x 128


def test_grows_target_from_single_seed(flagship_sim):
    sim = flagship_sim
    sim.reset()
    assert sim.alive_count() == 1, "growth must start from exactly one cell"
    sim.grow(GROWTH_STEPS)
    assert sim.ssim() >= SSIM_THRESHOLD, (
        f"grown pattern SSIM {sim.ssim():.4f} < {SSIM_THRESHOLD}"
    )
    assert sim.alive_count() > 100


@pytest.mark.parametrize("trial", range(5))
def test_regenerates_after_80pct_random_destruction(flagship_sim, trial):
    sim = flagship_sim
    sim.reset()
    sim.grow(GROWTH_STEPS)
    alive_before = sim.alive_count()

    sim.destroy_fraction(0.8, rng_seed=100 + trial)
    assert sim.alive_count() < 0.25 * alive_before, "damage must actually destroy ~80%"
    damaged_ssim = sim.ssim()

    sim.step(RANDOM_RECOVERY_STEPS)
    recovered = sim.ssim()
    assert recovered >= SSIM_THRESHOLD, (
        f"trial {trial}: SSIM after recovery {recovered:.4f} < {SSIM_THRESHOLD} "
        f"(damaged SSIM was {damaged_ssim:.4f})"
    )


@pytest.mark.parametrize("trial", range(5))
def test_regenerates_from_single_fragment(flagship_sim, trial):
    """80% destroyed, one contiguous ~20% fragment survives -> full image back."""
    sim = flagship_sim
    sim.reset()
    sim.grow(GROWTH_STEPS)
    alive_before = sim.alive_count()

    sim.keep_fragment(0.2, rng_seed=200 + trial)
    assert sim.alive_count() < 0.3 * alive_before

    sim.step(FRAGMENT_RECOVERY_STEPS)
    recovered = sim.ssim()
    assert recovered >= SSIM_THRESHOLD, (
        f"trial {trial}: SSIM after fragment regen {recovered:.4f} < {SSIM_THRESHOLD}"
    )


def test_pattern_is_persistent(flagship_sim):
    """The grown pattern must not decay or explode over a long horizon."""
    sim = flagship_sim
    sim.reset()
    sim.grow(GROWTH_STEPS)
    sim.step(400)
    assert sim.ssim() >= SSIM_THRESHOLD, (
        f"pattern decayed: SSIM {sim.ssim():.4f} after {sim.step_count} steps"
    )
