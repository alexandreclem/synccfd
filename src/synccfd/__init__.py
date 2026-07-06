"""SynCCFD — synthetic credit card transaction & fraud data simulator."""

from synccfd.simulator import DatasetGenerator, FraudSimulator
from synccfd.simulator.location import LocationFeatures
from synccfd.utils import DataManager

__version__ = "0.1.0"

__all__ = [
    "DatasetGenerator",
    "FraudSimulator",
    "LocationFeatures",
    "DataManager",
    "__version__",
]
