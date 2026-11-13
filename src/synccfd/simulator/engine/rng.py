"""Named, independent random streams."""

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
    """Random streams derived from one seed, one per subsystem."""

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
        """SeedSequence of a stream or of one of its sub-streams."""
        return np.random.SeedSequence(self.entropy, spawn_key=(self._key(name), *path))

    def generator(self, name: str, *path: int) -> np.random.Generator:
        """New generator at the start of a stream or sub-stream."""
        return np.random.default_rng(self.seed_sequence(name, *path))

    def child_generators(self, name: str, n: int) -> list[np.random.Generator]:
        """Generators of the first n sub-streams of a stream."""
        return [self.generator(name, i) for i in range(n)]
