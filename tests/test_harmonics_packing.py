"""The order in which the harmonics of a coefficient array form a vector."""
import numpy as np
import pytest

from planetmodel.harmonics import packing


def test_packing_lists_every_harmonic_once_in_its_order():
    L = 5
    s, l, m = packing(L)
    assert s.size == l.size == m.size == (L + 1) ** 2
    assert len(set(zip(s, l, m))) == (L + 1) ** 2
    assert np.all((m <= l) & (m >= s))               # no sine harmonic of order 0
    assert list(zip(s[:4], l[:4], m[:4])) == [(0, 0, 0), (0, 1, 0), (0, 1, 1),
                                              (1, 1, 1)]
    assert np.all(np.diff(l) >= 0)
    for deg in range(L + 1):
        here = l == deg
        assert list(s[here]) == [0] * (deg + 1) + [1] * deg
        assert list(m[here]) == list(range(deg + 1)) + list(range(1, deg + 1))


def test_packing_round_trips_a_coefficient_array():
    L = 6
    s, l, m = packing(L)
    rng = np.random.default_rng(0)
    vector = rng.standard_normal(((L + 1) ** 2, 3))      # a trailing radial axis
    c = np.zeros((2, L + 1, L + 1, 3))
    c[s, l, m] = vector
    assert np.array_equal(c[s, l, m], vector)
    assert np.count_nonzero(c) == vector.size
    assert packing(0)[0].size == 1
    with pytest.raises(ValueError):
        packing(-1)
