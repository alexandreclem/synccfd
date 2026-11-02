"""SynCCFD — synthetic credit card transaction & fraud data simulator."""

from synccfd.simulator import DatasetGenerator, FraudSimulator, SimulationResult
from synccfd.simulator.location import LocationFeatures
from synccfd.utils import DataManager

__version__ = "0.2.0"

__all__ = [
    "DatasetGenerator",
    "FraudSimulator",
    "SimulationResult",
    "LocationFeatures",
    "DataManager",
    "__version__",
]
