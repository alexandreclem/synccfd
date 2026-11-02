"""Backwards-compatible import path for the v0.1 fraud simulator."""

from .legacy.fraud import FraudSimulator

__all__ = ["FraudSimulator"]
