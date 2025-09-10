from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
FLAGSHIP_CHECKPOINT = REPO_ROOT / "checkpoints" / "flower.pt"


@pytest.fixture(scope="session")
def flagship_checkpoint() -> Path:
    if not FLAGSHIP_CHECKPOINT.exists():
        pytest.skip(
            "trained checkpoint missing, run: "
            "python -m morphogenesis train --config configs/flower.yaml"
        )
    return FLAGSHIP_CHECKPOINT


@pytest.fixture(scope="session")
def flagship_sim(flagship_checkpoint):
    from morphogenesis.simulation import Simulation

    return Simulation.from_checkpoint(flagship_checkpoint)
