"""Named, independent random streams for the v2 engine.

All randomness of the v2 engine comes from local ``numpy.random.Generator`` objects
derived from one ``SeedSequence``; the global ``random``/``numpy.random`` state is
never used. Each subsystem has its own stream, at a fixed position, so enabling or
changing one subsystem never changes the draws of another.
"""

from typing import Optional

import numpy as np

# Fixed order: never reorder or remove names, only append, or existing streams
# (and therefore datasets) change.
STREAM_NAMES = (
    "customers",
    "terminals",
    "merchants",
    "transactions",
    "drift",
    "fraud",
)


class RandomStreams:
    """Deterministic family of random streams derived from one seed.

    ``generator(name)`` always returns a *fresh* generator at the start of the
    stream, and ``child_generators(name, n)`` the generators of ``n`` sub-streams
    (e.g. one per customer). Both are pure: calling them twice gives the same draws.
    """

    def __init__(self, seed: Optional[int]):
        self.entropy = np.random.SeedSequence(seed).entropy

    def _key(self, name: str) -> int:
        try:
            return STREAM_NAMES.index(name)
        except ValueError:
            raise KeyError(
                f"Unknown random stream {name!r}; expected one of {STREAM_NAMES}."
            ) from None

    def seed_sequence(self, name: str, *path: int) -> np.random.SeedSequence:
        """SeedSequence of stream ``name`` (or of its sub-stream ``path``)."""
        return np.random.SeedSequence(self.entropy, spawn_key=(self._key(name), *path))

    def generator(self, name: str, *path: int) -> np.random.Generator:
        """Fresh generator of stream ``name`` (or of its sub-stream ``path``)."""
        return np.random.default_rng(self.seed_sequence(name, *path))

    def child_generators(self, name: str, n: int) -> list[np.random.Generator]:
        """Generators of sub-streams ``0..n-1`` of stream ``name``."""
        return [self.generator(name, i) for i in range(n)]
