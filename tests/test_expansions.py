"""Fields and surface functions given by spherical-harmonic expansions.

A harmonic field passes the field contract, equals the per-point sum of
its harmonics, is exact in radius on polynomial pieces, and gridded on a
Gauss-Legendre grid of its band and read back by the harmonic rule it
returns to rounding.  A harmonic function is a relief whose displacement
and geometry pass their contracts.
"""
import numpy as np
import pytest
from scipy.interpolate import BSpline, CubicSpline, PPoly, make_interp_spline

from planetmodel import (Geometry, GriddedField, HarmonicField, HarmonicFunction,
                         RadialStretch, Skeleton, gauss_legendre, layer_linear,
                         sample, synthesise)
from planetmodel.catalogue import LayeredIsotropicElastic
from planetmodel.character import DENSITY, VECTOR
from planetmodel.expansions import ppoly_of
from planetmodel.harmonics import packing
from planetmodel.testing import (check_displacement, check_field, check_geometry,
                                 check_sample)

INTERVAL = (0.5, 1.0)


def random_coefficients(L, rng, *lead):
    """Coefficients of shape lead + (2, L + 1, L + 1), zero off the harmonics."""
    c = np.zeros(lead + (2, L + 1, L + 1))
    s, l, m = packing(L)
    c[..., s, l, m] = rng.standard_normal(lead + (s.size,)) / (l + 1.0)
    return c


def piecewise_cubic(L, rng):
    """A cubic spline of radius through six knots, values (2, L + 1, L + 1):
    continuous, as a field is within its layer."""
    r = np.linspace(*INTERVAL, 6)
    return CubicSpline(r, random_coefficients(L, rng, 6), axis=0)


def test_a_b_spline_is_taken_as_its_pieces():
    rng = np.random.default_rng(7)
    r = np.linspace(*INTERVAL, 7)
    spl = make_interp_spline(r, random_coefficients(3, rng, 7), k=3, axis=0)
    f = HarmonicField(INTERVAL, spl)
    x = rng.uniform(*INTERVAL, 20)
    assert np.allclose(np.moveaxis(f.coefficients_at(x), -1, 0), spl(x), atol=1e-12)


@pytest.mark.parametrize("degree", [0, 1, 2])
def test_a_b_spline_of_any_degree_is_its_pieces(degree):
    """Degree zero among them, whose knots are its breakpoints alone: the
    pieces are the spline's, and their values the spline's own."""
    rng = np.random.default_rng(8)
    inside = np.array([0.6, 0.75, 0.9])
    t = np.concatenate([[INTERVAL[0]] * (degree + 1), inside,
                        [INTERVAL[1]] * (degree + 1)])
    spl = BSpline(t, rng.standard_normal((t.size - degree - 1, 2)), degree)
    pp = ppoly_of(spl)
    assert np.array_equal(pp.x, np.concatenate([[INTERVAL[0]], inside,
                                                [INTERVAL[1]]]))
    assert pp.c.shape == (degree + 1, 4, 2)
    x = rng.uniform(*INTERVAL, 40)
    assert np.allclose(pp(x), spl(x), rtol=0, atol=1e-13)


def test_the_field_contract():
    rng = np.random.default_rng(0)
    f = HarmonicField(INTERVAL, piecewise_cubic(6, rng), name="dv")
    check_field(f)
    assert f.lmax == 6 and f.name == "dv" and not f.is_radial
    constant = HarmonicField(INTERVAL, random_coefficients(4, rng), character=DENSITY)
    check_field(constant)


def test_values_are_the_sum_of_the_harmonics_point_by_point():
    rng = np.random.default_rng(1)
    pp = piecewise_cubic(8, rng)
    f = HarmonicField(INTERVAL, pp)
    r = rng.uniform(*INTERVAL, 30)
    t, p = rng.uniform(0, np.pi, 30), rng.uniform(-4, 9, 30)
    want = [synthesise(pp(ri), ti, pi) for ri, ti, pi in zip(r, t, p)]
    assert np.allclose(f(r, t, p), want, rtol=0, atol=1e-13)
    assert f.coefficients_at(r).shape == (2, 9, 9, 30)
    assert f(r[:, None, None], t[None, :4, None], p[None, None, :3]).shape == (30, 4, 3)


def test_exact_in_radius_beyond_the_ends_and_rescaled():
    rng = np.random.default_rng(2)
    pp = piecewise_cubic(3, rng)
    f = HarmonicField(INTERVAL, pp)
    wide = f.on_interval(0.4, 1.1)
    assert np.allclose(wide(1.05, 1.0, 2.0), synthesise(pp(1.05), 1.0, 2.0))
    with pytest.raises(ValueError, match="outside the interval"):
        f(1.05, 1.0, 2.0)
    s = f.rescaled(k=2.0, v=3.0)
    r = rng.uniform(*INTERVAL, 10)
    t, p = rng.uniform(0, np.pi, 10), rng.uniform(0, 6, 10)
    assert s.interval == (1.0, 2.0)
    assert np.allclose(s(2.0 * r, t, p), 3.0 * f(r, t, p), atol=1e-12)
    assert f.renamed("q").name == "q" and f.name is None


def test_gridded_on_a_grid_of_its_band_and_read_back_by_the_harmonic_rule():
    rng = np.random.default_rng(3)
    pp = piecewise_cubic(10, rng)
    f = HarmonicField(INTERVAL, pp)
    g = GriddedField.sampled(f, gauss_legendre(10), breaks=np.unique(pp.x), ngll=4,
                             lateral="harmonic")
    r = rng.uniform(*INTERVAL, 50)
    t, p = rng.uniform(0, np.pi, 50), rng.uniform(0, 2 * np.pi, 50)
    assert np.allclose(g(r, t, p), f(r, t, p), rtol=0, atol=1e-12)


def test_a_harmonic_field_in_a_model_samples_and_validates():
    rng = np.random.default_rng(4)
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    c = random_coefficients(3, rng)
    c[0, 0, 0] = 4.0 * np.sqrt(4 * np.pi)
    m = m.with_field(1, "rho3d", HarmonicField(INTERVAL, c, character=DENSITY))
    s = sample(m, gauss_legendre(5), fields=["rho3d"], missing="nan", drmax=0.25)
    check_sample(s, m)


def test_refusals():
    rng = np.random.default_rng(5)
    c = random_coefficients(3, rng)
    with pytest.raises(ValueError, match="rank 0"):
        HarmonicField(INTERVAL, c, character=VECTOR)
    bad = c.copy()
    bad[1, 2, 0] = 1.0
    with pytest.raises(ValueError, match="belong to no harmonic"):
        HarmonicField(INTERVAL, bad)
    bad = c.copy()
    bad[0, 1, 3] = 1.0
    with pytest.raises(ValueError, match="belong to no harmonic"):
        HarmonicFunction(bad)
    with pytest.raises(ValueError, match="shape"):
        HarmonicField(INTERVAL, c[:, :, :2])
    with pytest.raises(TypeError, match="real"):
        HarmonicField(INTERVAL, c + 0j)
    with pytest.raises(ValueError, match="shape"):
        HarmonicFunction(c[None])


# -- HarmonicFunction -------------------------------------------------------------

def test_a_harmonic_function_and_its_gradient():
    rng = np.random.default_rng(6)
    c = random_coefficients(7, rng)
    f = HarmonicFunction(c)
    assert f.lmax == 7 and f.coefficients.shape == (2, 8, 8)
    t, p = rng.uniform(0.1, 3.0, 40), rng.uniform(0, 2 * np.pi, 40)
    assert np.allclose(f(t, p), synthesise(c, t, p), rtol=0, atol=1e-14)
    h = 1e-6
    dt, dp = f.angular_gradient(t, p)
    assert np.allclose(dt, (f(t + h, p) - f(t - h, p)) / (2 * h), atol=1e-8)
    assert np.allclose(dp, (f(t, p + h) - f(t, p - h)) / (2 * h), atol=1e-8)
    assert f.angular_gradient(0.0, 1.0)[1] == pytest.approx(0.0, abs=1e-14)


def test_a_harmonic_function_is_a_relief_whose_geometry_holds():
    c = np.zeros((2, 4, 4))
    c[0, 1, 0], c[0, 1, 1], c[1, 3, 2] = 0.01, 0.006, 0.004
    rel = HarmonicFunction(c)
    sk = Skeleton([0.0, 0.5, 1.0])
    h = layer_linear(sk, [None, rel, rel])
    check_displacement(h, sk)
    check_geometry(Geometry(sk, mapping=RadialStretch(h, rmax=1.0)))


def test_a_scaled_harmonic_relief_is_a_harmonic_function_still():
    rng = np.random.default_rng(11)
    c = 0.01 * random_coefficients(5, rng)
    rel = HarmonicFunction(c)
    k = 6.371e6
    big = rel.scaled(k)
    assert type(big) is HarmonicFunction and big.lmax == rel.lmax
    assert np.array_equal(big.coefficients, k * c)
    assert not big.coefficients.flags.writeable
    assert np.array_equal(rel.coefficients, c)                # the original is kept
    t, p = rng.uniform(0.0, np.pi, 30), rng.uniform(0, 2 * np.pi, 30)
    want = k * rel(t, p)                  # compared on the field's scale: it has zeros
    assert np.allclose(big(t, p), want, rtol=0, atol=1e-13 * np.abs(want).max())
    for got, want in zip(big.angular_gradient(t, p), rel.angular_gradient(t, p)):
        assert np.allclose(got, k * want, rtol=1e-12, atol=1e-12 * k * 0.01)
    assert big.mean() == pytest.approx(k * rel.mean(), rel=1e-14)
    assert np.allclose(big.scaled(1.0 / k).coefficients, c, rtol=1e-15, atol=0)
    # a stack of harmonic reliefs scaled with its body holds harmonic reliefs
    sk = Skeleton([0.0, 0.5, 1.0])
    h = layer_linear(sk, [None, rel, rel])
    stack = h.scaled(k)
    assert stack.reliefs[0] is None
    for scaled in stack.reliefs[1:]:
        assert type(scaled) is HarmonicFunction
        assert np.array_equal(scaled.coefficients, k * c)
    geometry = Geometry(sk, mapping=RadialStretch(h, rmax=1.0)).scaled(k)
    assert all(type(r) is HarmonicFunction for r in geometry.mapping.h.reliefs[1:])
    check_displacement(stack, Skeleton(k * sk.boundaries))
    check_geometry(geometry)


def test_on_grid_holds_the_radius_and_takes_ducc0_on_any_grid():
    from planetmodel import AngularGrid, gauss_legendre

    L = 6
    rng = np.random.default_rng(3)
    c = rng.standard_normal((2, L + 1, L + 1))
    c[:, np.arange(L + 1)[:, None] < np.arange(L + 1)[None, :]] = 0.0
    c[1, :, 0] = 0.0
    h = HarmonicField((0.5, 1.0), c)
    with pytest.raises(ValueError, match="outside the interval"):
        h.on_grid([5.0], gauss_legendre(L))
    custom = AngularGrid(np.linspace(0.1, 3.0, 7), np.linspace(0.0, 6.0, 9))
    for grid in (gauss_legendre(L), gauss_legendre(2), custom):
        want = h.evaluate(np.array([0.7])[:, None, None],
                          grid.colatitudes[None, :, None],
                          grid.longitudes[None, None, :])
        assert np.allclose(h.on_grid([0.7], grid), want, atol=1e-11)


def test_points_by_radius_and_by_term_are_the_coefficients_there():
    """A piece's points are synthesised a radius at a time where they have
    few radii, a term at a time where they have many; both are the
    expansion of the coefficients at each point's radius."""
    from planetmodel.harmonics import POINT_EPSILON

    rng = np.random.default_rng(14)
    L = 24
    c = rng.standard_normal((3, 2, 2, L + 1, L + 1))       # two quadratic pieces
    for l in range(L + 1):
        c[..., l, l + 1:] = 0.0
    c[..., 1, :, 0] = 0.0
    f = HarmonicField((1.0, 2.0), PPoly(c, [1.0, 1.5, 2.0]))
    n = 300
    theta, phi = np.arccos(rng.uniform(-1, 1, n)), rng.uniform(0, 2 * np.pi, n)
    r = np.where(np.arange(n) < 150, 1.2, rng.uniform(1.5, 2.0, n))   # one radius,
    got = f(r, theta, phi)                                  # then many
    want = np.array([synthesise(f.coefficients_at(ri), ti, pi)
                     for ri, ti, pi in zip(r, theta, phi)])
    scale = np.abs(want).max()
    assert np.allclose(got, want, rtol=0.0, atol=10 * POINT_EPSILON * scale)
    few = f(r[148:153], theta[148:153], phi[148:153])      # a radius at a time
    assert np.allclose(few, want[148:153], rtol=0.0, atol=10 * POINT_EPSILON * scale)
