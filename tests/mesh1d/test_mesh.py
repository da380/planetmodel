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


def test_to_ppoly_with_a_trailing_shape_is_the_scalar_view_by_component():
    m = Mesh1D([0.0, 1.0, 2.5], ngll=6, drmax=0.4)
    v = np.random.default_rng(0).standard_normal(m.r.shape + (2, 3))
    P = m.to_ppoly(v, elements=(1, 4))
    assert P.c.shape == (6, 3, 2, 3)
    for i in range(2):
        for j in range(3):
            one = m.to_ppoly(v[..., i, j], elements=(1, 4))
            assert np.array_equal(P.c[..., i, j], one.c)
    x = np.linspace(m.left[1], m.right[3], 9)
    assert P(x).shape == (9, 2, 3)
    with pytest.raises(ValueError, match="shape"):
        m.to_ppoly(np.zeros((m.nspec, 2, m.ngll)))


def test_the_linear_rule_is_straight_lines_between_nodes():
    m = Mesh1D([0.0, 1.0, 2.5], ngll=4, drmax=0.6)
    v = np.random.default_rng(1).standard_normal((m.nspec, m.ngll, 2))
    P = m.to_ppoly(v, method="linear")
    assert P.c.shape == (2, m.nspec * (m.ngll - 1), 2)
    for e in range(m.nspec):
        x = np.linspace(m.left[e], m.right[e], 13)[:-1]
        assert np.allclose(P(x)[:, 1], np.interp(x, m.r[e], v[e, :, 1]), atol=1e-14)
    jump = np.where(m.left[:, None] >= 1.0, 1.0, 0.0) + 0.0 * m.r
    Q = m.to_ppoly(jump, method="linear")
    assert Q(0.5) == 0.0 and Q(1.0) == 1.0          # the upper element at a boundary
    part = m.to_ppoly(v, elements=(1, 3), method="linear")
    assert part.x[0] == m.left[1] and part.x[-1] == m.right[2]
    with pytest.raises(ValueError, match="method"):
        m.to_ppoly(v, method="cubic")


@pytest.mark.parametrize("method", ["polynomial", "linear"])
def test_interpolation_weights_give_the_exact_view(method):
    m = Mesh1D([0.0, 1.0, 2.5], ngll=5, drmax=0.4)
    rng = np.random.default_rng(2)
    u = rng.standard_normal(m.r.shape)
    x = np.concatenate([rng.uniform(-0.3, 2.8, (40,)), m.r.ravel(), m.left])
    e, W = m.interpolation_weights(x, method=method)
    assert e.shape == x.shape and W.shape == x.shape + (m.ngll,)
    assert np.allclose(np.einsum("pi,pi->p", W, u[e]), m.to_ppoly(u, method=method)(x),
                       atol=1e-12)
    assert np.allclose(W.sum(axis=-1), 1.0)
    assert np.array_equal(e[-m.nspec:], np.arange(m.nspec))   # boundaries resolve up
    assert e[x < 0.0].max(initial=0) == 0 and e[x > 2.5].min(initial=m.nspec - 1) == \
        m.nspec - 1
    e2, W2 = m.interpolation_weights(np.full((3, 2), 0.7), method=method)
    assert e2.shape == (3, 2) and W2.shape == (3, 2, m.ngll)
    with pytest.raises(ValueError, match="method"):
        m.interpolation_weights(x, method="spline")


def test_the_rules_agree_with_two_nodes_an_element():
    m = Mesh1D([0.0, 1.0, 2.5], ngll=2, drmax=0.3)
    u = np.random.default_rng(3).standard_normal(m.r.shape)
    x = np.linspace(-0.2, 2.7, 101)
    assert np.allclose(m.to_ppoly(u)(x), m.to_ppoly(u, method="linear")(x), atol=1e-13)
    assert np.allclose(m.interpolation_weights(x)[1],
                       m.interpolation_weights(x, method="linear")[1], atol=1e-14)


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
    assert m.element_of(r7) == m.start_element(7)
    assert m.start_element(0) == 0
    with pytest.raises(ValueError, match="non-negative"):
        m.truncation_radius(-1)
    assert "RadialMesh" in repr(m)


def _kinked(knots):
    from planetmodel.displacement import CallableDisplacement
    from planetmodel.mapping import RadialStretch

    def h(r, theta, phi):
        out = 0.0 * r
        for k in knots:
            out = out + 0.01 * np.maximum(r - k, 0.0) * np.cos(theta)
        return out

    return RadialStretch(CallableDisplacement(h, knots=knots), rmax=1.0)


def test_the_knots_of_a_model_are_pinned_as_of_its_geometry():
    from planetmodel import Model

    sk = Skeleton([0.0, 0.5, 1.0])
    g = Geometry(sk, mapping=_kinked((0.7, 0.8)))
    m = Model(g, [{}, {}])
    for source in (g, m):
        mesh = RadialMesh(source, drmax=0.25)
        assert {0.5, 0.7, 0.8} <= set(mesh.left.tolist())


def test_a_boundary_wins_over_a_knot_within_the_tolerance():
    sk = Skeleton([0.0, 0.5, 1.0])
    near = float(np.nextafter(0.5, 0.0))
    mesh = RadialMesh(Geometry(sk, mapping=_kinked((near,))), drmax=0.25)
    assert 0.5 in mesh.left and near not in mesh.left


def test_edges_are_held_to_the_skeleton_to_the_tolerance():
    sk = Skeleton([0.0, 0.5, 1.0])
    mesh = RadialMesh(sk, edges=[0.0, 0.5, float(np.nextafter(1.0, 2.0))])
    assert mesh.right[-1] == 1.0
    with pytest.raises(ValueError, match="within the skeleton"):
        RadialMesh(sk, edges=[0.0, 0.5, 1.01])


def test_a_pin_within_the_tolerance_of_a_boundary_is_the_boundary():
    sk = Skeleton([0.0, 0.5, 1.0])
    for near in (0.5 - 4e-10, 0.5 + 4e-10):
        mesh = RadialMesh(sk, edges=[0.0, 0.25, near, 1.0])
        assert list(mesh.left) == [0.0, 0.25, 0.5]
        assert list(mesh.right) == [0.25, 0.5, 1.0]
        assert list(mesh.layer) == [0, 0, 1]
    # the tolerance is rtol of the mesh's range
    part = RadialMesh(sk, edges=[0.5 + 4e-10, 0.75])
    assert part.left[0] == 0.5 + 4e-10 and list(part.layer) == [1]
    wide = RadialMesh(sk, edges=[0.0, 0.5 + 1e-6, 1.0], rtol=1e-5)
    assert list(wide.left) == [0.0, 0.5]
    # a pin beyond it is kept, and an element that then crosses is refused
    with pytest.raises(ValueError, match="straddles"):
        RadialMesh(sk, edges=[0.0, 0.25, 0.5 + 1e-6, 1.0])


def test_to_ppoly_keeps_complex_values():
    mesh = Mesh1D([0.0, 0.5, 1.0], ngll=5)
    u = (1.0 + 2.0j) * mesh.r ** 2
    x = np.array([0.3, 0.8])
    for method in ("polynomial", "linear"):
        element, w = mesh.interpolation_weights(x, method=method)
        direct = np.einsum("...i,...i->...", w, u[element])
        assert np.allclose(mesh.to_ppoly(u, method=method)(x), direct, rtol=1e-12)
    assert np.allclose(mesh.to_ppoly(u)(x), (1.0 + 2.0j) * x ** 2, rtol=1e-12)
