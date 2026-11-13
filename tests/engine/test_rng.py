"""Named random streams."""

import numpy as np
import pytest

from synccfd.simulator.engine.rng import STREAM_NAMES, RandomStreams


def test_stream_names_are_fixed():
    # Appending is allowed; reordering or removing would change existing datasets.
    assert STREAM_NAMES[:6] == (
        "customers",
        "terminals",
        "merchants",
        "transactions",
        "drift",
        "fraud",
    )


def test_generators_are_pure():
    streams = RandomStreams(42)
    first = streams.generator("customers").random(5)
    again = streams.generator("customers").random(5)
    np.testing.assert_array_equal(first, again)
    children = [g.random(3) for g in streams.child_generators("transactions", 3)]
    children_again = [g.random(3) for g in streams.child_generators("transactions", 3)]
    np.testing.assert_array_equal(children, children_again)


def test_same_seed_same_draws_different_seed_different_draws():
    a = RandomStreams(1).generator("fraud", 2).random(10)
    b = RandomStreams(1).generator("fraud", 2).random(10)
    c = RandomStreams(2).generator("fraud", 2).random(10)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


def test_streams_are_distinct():
    streams = RandomStreams(0)
    draws = [streams.generator(name).random(4) for name in STREAM_NAMES]
    draws += [streams.generator("fraud", k).random(4) for k in range(1, 5)]
    draws += [g.random(4) for g in streams.child_generators("transactions", 4)]
    assert len({tuple(d) for d in draws}) == len(draws)


def test_child_streams_do_not_depend_on_their_number():
    streams = RandomStreams(3)
    few = [g.random(3) for g in streams.child_generators("transactions", 2)]
    many = [g.random(3) for g in streams.child_generators("transactions", 50)]
    np.testing.assert_array_equal(few, many[:2])


def test_unknown_stream_raises():
    with pytest.raises(KeyError):
        RandomStreams(0).generator("weather")
