"""Synthetic transaction simulator."""

from . import presets
from .generator import DatasetGenerator, SimulationResult

__all__ = ["DatasetGenerator", "SimulationResult", "presets"]
