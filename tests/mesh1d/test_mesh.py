"""Mesh1D and RadialMesh: elements on breakpoints, layers, the exact view."""
import numpy as np
import pytest

from planetmodel import Geometry, Skeleton
from planetmodel.mesh1d import Mesh1D, RadialMesh

SK = Skeleton([0.0, 0.3, 0.7, 1.0])
HOLLOW = Skeleton([0.2, 0.7, 1.0])


def test_mesh1d_geometry():
    m = Mesh1D([0.0, 1.0, 3.0], ngll=4, drmax=0.5)
    assert m.nspec == 6                       # 2 + 4 elements
    assert m.nglob == 6 * 3 + 1
    assert np.allclose(m.right - m.left, 0.5)
    assert m.r.shape == (6, 4)
    assert np.array_equal(m.r[:, 0], m.left) and np.array_equal(m.r[:, -1], m.right)
    assert np.array_equal(m.rglob[m.gmap], m.r)
    assert Mesh1D([0.0, 1.0, 3.0]).nspec == 2
    with pytest.raises(ValueError, match="strictly increasing"):
        Mesh1D([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="drmax"):
        Mesh1D([0.0, 1.0], drmax=-1.0)


def test_element_of_and_clipping():
    m = Mesh1D([0.0, 1.0, 2.0], drmax=0.5)
    assert m.element_of(0.25) == 0
    assert m.element_of(0.5) == 1                # a boundary resolves upward
    assert m.element_of(2.0) == 3
    assert m.element_of(-1.0) == 0 and m.element_of(5.0) == 3


def test_to_ppoly_is_exact_on_a_polynomial():
    m = Mesh1D([0.0, 1.0, 2.5], ngll=5, drmax=0.4)
    f = lambda x: 3.0 * x ** 3 - x + 0.5
    P = m.to_ppoly(f(m.r))
    x = np.linspace(0.0, 2.5, 200)
    assert np.allclose(P(x), f(x), atol=1e-12)
    assert np.allclose(P.derivative()(x), 9.0 * x ** 2 - 1.0, atol=1e-10)
    jump = np.where(m.left[:, None] >= 1.0, 1.0, 0.0) + 0.0 * m.r
    Q = m.to_ppoly(jump)
    assert Q(0.5) == 0.0 and Q(1.5) == 1.0 and Q(1.0) == 1.0
    part = m.to_ppoly(f(m.r), elements=(0, 2))
    assert part.x[0] == m.left[0] and part.x[-1] == m.right[1]
    with pytest.raises(ValueError, match="shape"):
        m.to_ppoly(np.zeros(3))
    with pytest.raises(ValueError, match="in-range"):
        m.to_ppoly(f(m.r), elements=(2, 2))


def test_radial_mesh_honours_the_skeleton():
    m = RadialMesh(SK, drmax=0.25)
    for b in SK.boundaries:
        assert np.any(np.isclose(m.left, b)) or np.isclose(m.right[-1], b)
    for e in range(m.nspec):
        lo, hi = m.left[e], m.right[e]
        assert SK.interval(m.layer[e])[0] <= lo and hi <= SK.interval(m.layer[e])[1]
    assert RadialMesh(Geometry(SK), drmax=0.25).nspec == m.nspec
    assert RadialMesh(SK, lmax=4).drmax == pytest.approx(0.1 / 5)
    with pytest.raises(TypeError, match="Skeleton"):
        RadialMesh([0.0, 1.0], drmax=0.1)


def test_radial_mesh_on_a_hollow_skeleton():
    m = RadialMesh(HOLLOW, drmax=0.1)
    assert m.left[0] == 0.2 and m.right[-1] == 1.0
    assert set(m.layer) == {0, 1}


def test_range_and_sizing_refusals():
    with pytest.raises(ValueError, match="exactly one"):
        RadialMesh(SK)
    with pytest.raises(ValueError, match="exactly one"):
        RadialMesh(SK, drmax=0.1, lmax=3)
    with pytest.raises(ValueError, match="within the skeleton"):
        RadialMesh(SK, drmax=0.1, rmax=1.5)
    with pytest.raises(ValueError, match="within the skeleton"):
        RadialMesh(SK, drmax=0.1, rmin=0.5, rmax=0.4)
    with pytest.raises(ValueError, match="lmax"):
        RadialMesh(SK, lmax=-1)
    with pytest.raises(ValueError, match="drmax"):
        RadialMesh(SK, drmax=0.0)
    cut = RadialMesh(SK, drmax=0.1, rmin=0.5, rmax=0.9)
    assert cut.left[0] == 0.5 and cut.right[-1] == 0.9
    assert np.any(np.isclose(cut.left, 0.7))


def test_edges():
    m = RadialMesh(SK, edges=[0.0, 0.3, 0.5, 0.7, 1.0])
    assert m.nspec == 4 and list(m.layer) == [0, 1, 1, 2]
    with pytest.raises(ValueError, match="straddles"):
        RadialMesh(SK, edges=[0.0, 0.5, 1.0])
    with pytest.raises(ValueError, match="already fixes"):
        RadialMesh(SK, edges=[0.0, 0.3, 1.0], rmax=1.0)
    with pytest.raises(ValueError, match="within the skeleton"):
        RadialMesh(SK, edges=[0.0, 1.2])
    with pytest.raises(ValueError, match="at least two"):
        RadialMesh(SK, edges=[0.5])


def test_truncation_rule():
    m = RadialMesh(SK, drmax=0.05)
    assert m.truncation_radius(0) == pytest.approx(1e-8)
    r7 = m.truncation_radius(7)
    assert r7 == pytest.approx(1e-8 ** (1.0 / 8))
    assert m.element_at(r7) == m.start_element(7)
    assert m.element_at(0.3) == m.element_of(0.3)
    assert m.start_element(0) == 0
    with pytest.raises(ValueError, match="non-negative"):
        m.truncation_radius(-1)
    assert "RadialMesh" in repr(m)
