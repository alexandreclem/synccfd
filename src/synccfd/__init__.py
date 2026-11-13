"""SynCCFD: synthetic credit card transaction and fraud data simulator."""

from synccfd.simulator import DatasetGenerator, SimulationResult
from synccfd.simulator.location import LocationFeatures
from synccfd.utils import DataManager

__version__ = "1.0.0"

__all__ = [
    "DatasetGenerator",
    "SimulationResult",
    "LocationFeatures",
    "DataManager",
    "__version__",
]
