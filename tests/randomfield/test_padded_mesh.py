"""Padded meshes and the restriction to the physical interval."""
import numpy as np
import pytest

from planetmodel.mesh1d import Mesh1D
from planetmodel.randomfield import (RadialOperatorFamily, Restriction,
                                     matern_reach, padded_mesh, restriction)


def test_matern_reach():
    assert matern_reach(0.5, 3.0) == pytest.approx(6.0)
    for bad in ((0.0, 1.0), (1.0, 0.0), (-1.0, 1.0)):
        with pytest.raises(ValueError):
            matern_reach(*bad)


def test_padded_mesh_pins_the_interval_and_the_measure_decides_the_clamp():
    m = padded_mesh(0.5, 1.0, pad=(0.2, 0.3), ngll=4, drmax=0.1)
    assert m.left[0] == pytest.approx(0.3) and m.right[-1] == pytest.approx(1.3)
    assert 0.5 in m.left and 1.0 in m.right
    assert np.max(m.right - m.left) <= 0.1 * (1 + 1e-12)
    ball = padded_mesh(0.5, 1.0, pad=2.0)
    assert ball.left[0] == 0.0 and ball.right[-1] == 3.0
    assert ball.nspec == 3                          # one element per span
    line = padded_mesh(0.5, 1.0, pad=2.0, weight="one")
    assert line.left[0] == -1.5
    below = padded_mesh(-1.0, 1.0, pad=0.5, weight="one")
    assert below.left[0] == -1.5 and below.right[-1] == 1.5
    bare = padded_mesh(0.0, 1.0, pad=(0.0, 0.0))
    assert bare.nspec == 1 and (bare.left[0], bare.right[-1]) == (0.0, 1.0)
    for kw in (dict(pad=-0.1), dict(pad=(0.1, np.inf)), dict(pad=0.1, weight="r")):
        with pytest.raises(ValueError):
            padded_mesh(0.5, 1.0, **kw)
    with pytest.raises(ValueError, match="r1 >= 0"):
        padded_mesh(-1.0, 1.0, pad=0.1)
    with pytest.raises(ValueError, match="r1 < r2"):
        padded_mesh(1.0, 1.0, pad=0.1)


def test_restriction():
    m = padded_mesh(0.5, 1.0, pad=(0.2, 0.3), ngll=4, drmax=0.1)
    R = restriction(m, 0.5, 1.0)
    assert isinstance(R, Restriction) and R.interval == (0.5, 1.0)
    e0, e1 = R.elements
    assert m.left[e0] == 0.5 and m.right[e1 - 1] == 1.0
    assert np.array_equal(R.r, m.rglob[R.nodes])
    assert R.r[0] == 0.5 and R.r[-1] == 1.0
    assert R.r.size == (e1 - e0) * (m.ngll - 1) + 1
    with pytest.raises(ValueError):
        R.r[0] = 0.0
    whole = restriction(m, m.left[0], m.right[-1])
    assert whole.elements == (0, m.nspec) and whole.r.size == m.nglob
    # the polynomial view over the restriction's elements spans the interval
    pp = m.to_ppoly(np.ones(m.r.shape), elements=R.elements)
    assert (pp.x[0], pp.x[-1]) == (0.5, 1.0)
    with pytest.raises(ValueError, match="element boundaries"):
        restriction(m, 0.55, 1.0)
    with pytest.raises(ValueError, match="element boundaries"):
        restriction(Mesh1D([0.0, 1.0]), 0.0, 0.5)
    with pytest.raises(ValueError, match="r1 < r2"):
        restriction(m, 1.0, 0.5)


@pytest.mark.parametrize("a, b, weight", [(0.6, 1.0, "r2"), (-0.2, 0.2, "one")])
def test_padding_insensitivity_of_the_restricted_covariance(a, b, weight):
    """The covariance A^-p on the physical nodes barely notices the pad
    once it is a couple of reaches long, on a shell and on an interval of
    the line that crosses zero alike."""
    nu, lam = 1.0, 0.04
    p = nu + 0.5

    def covariance(pad_factor):
        mesh = padded_mesh(a, b, pad=pad_factor * matern_reach(nu, lam),
                           weight=weight, drmax=lam / 2.0)
        R = restriction(mesh, a, b)
        theta, Phi = RadialOperatorFamily(mesh, kappa=lam ** 2, weight=weight).eig(0)
        Phi = Phi[R.nodes]
        C = (Phi * theta ** -p) @ Phi.T
        d = np.sqrt(np.diag(C))
        return R.r, C / np.outer(d, d)

    r2, C2 = covariance(2.0)
    r4, C4 = covariance(4.0)
    assert np.allclose(r2, r4, rtol=0.0, atol=1e-14)
    assert np.max(np.abs(C2 - C4)) < 5e-4
