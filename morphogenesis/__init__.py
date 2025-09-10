"""Self-Organizing Morphogenesis Engine: Neural Cellular Automata.

A PyTorch implementation of differentiable morphogenesis: a grid of cells,
each running an identical learned update rule, grows a target image from a
single seed and regenerates it from surviving fragments after damage.

A configurable engine with multiple targets, damage models, interactive
simulation, and a reproducible evaluation suite.
"""

from morphogenesis.config import Config, load_config
from morphogenesis.model import NeuralCA
from morphogenesis.simulation import Simulation
from morphogenesis.targets import get_target, list_targets

__version__ = "1.0.0"

__all__ = [
    "Config",
    "load_config",
    "NeuralCA",
    "Simulation",
    "get_target",
    "list_targets",
    "__version__",
]
