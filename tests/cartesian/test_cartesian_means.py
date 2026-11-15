"""Horizontal means: a flat field, a perturbed field and a model reduced to
depth.

A field of depth alone is its own mean; a field without a mean of its own
takes the grid's quadrature, exact on the grid's band, of any rank; a
perturbed flat field combines its parts exactly when absolute or
relative and by quadrature of the exponential when logarithmic, on its
own grid or the one given; a model horizontally averaged re-derives what
its type derives; a mean of one system refuses an object of the other by
name.
"""
import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import (SCALAR, VECTOR, AnalyticField, PerturbedField, gauss_legendre,
                         perturbed, spherical_mean)
from planetmodel import cartesian as flat
from planetmodel.cartesian.means import quadrature_mean
from planetmodel.fields import ComposedField

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
IV = (0.0, 1.0)


def cosines(d, x, y):
    """A field of the band of an 8 x 6 grid: its mean 1 + d^2."""
    return (1.0 + d ** 2 + 0.3 * np.cos(np.pi * x) * d
            + 0.2 * np.sin(2 * np.pi * y) * np.cos(np.pi * x) + 0.0 * y)


def test_a_field_of_depth_alone_is_its_own_mean():
    rho = flat.DepthField(IV, 3.0)
    assert flat.horizontal_mean(rho) is rho
    v = flat.constant_field(IV, [1.0, 2.0, 3.0], character=VECTOR)
    assert flat.horizontal_mean(v) is v
    # a field of depth alone without a mean of its own is wrapped
    composed = ComposedField(lambda a: 2.0 * a, [rho], character=SCALAR)
    m = flat.horizontal_mean(composed)
    assert isinstance(m, flat.DepthField) and m(0.4) == pytest.approx(6.0)


def test_a_field_without_a_mean_of_its_own_needs_a_grid_and_is_exact_on_its_band():
    f = flat.AnalyticField(IV, cosines)
    with pytest.raises(ValueError, match="no horizontal mean of its own; pass grid="):
        flat.horizontal_mean(f)
    grid = flat.Grid(BOX, (8, 6))
    m = flat.horizontal_mean(f, grid=grid)
    d = np.linspace(0, 1, 7)
    assert isinstance(m, flat.DepthField)
    assert np.allclose(m(d), 1.0 + d ** 2, rtol=0, atol=1e-15)
    vec = flat.AnalyticField(IV, lambda d, x, y: np.stack(np.broadcast_arrays(
        cosines(d, x, y), 2.0 * cosines(d, x, y), np.cos(np.pi * x)), axis=-1),
        character=VECTOR)
    mv = flat.horizontal_mean(vec, grid=grid)
    assert mv.character == VECTOR
    want = np.stack([1 + d ** 2, 2 * (1 + d ** 2), 0 * d], axis=-1)
    assert np.allclose(mv.evaluate(d, 0.0, 0.0), want, rtol=0, atol=1e-15)
    with pytest.raises(TypeError, match="cartesian.Grid"):
        quadrature_mean(f, gauss_legendre(4))


@pytest.mark.parametrize("kind", ["absolute", "relative"])
def test_the_perturbed_kinds_combine_their_parts_exactly(kind):
    ref = flat.DepthField(IV, PPoly(np.array([[0.5], [3.0]]), np.array(IV)))
    grid = flat.Grid(BOX, (6, 6))
    t = grid.transform
    c = np.zeros((1, 1, t.size))
    c[0, 0, 0] = 0.02 * np.sqrt(grid.area)                      # mean 0.02
    c[0, 0, t.component_of([1, 1])] = 0.1
    dp = flat.FourierField(IV, grid, PPoly(c, np.array(IV)))
    p = PerturbedField(ref, dp, kind=kind)
    m = p.horizontal_mean()
    d = np.linspace(0, 1, 5)
    want = 3.0 + 0.5 * d + 0.02 if kind == "absolute" else (3.0 + 0.5 * d) * 1.02
    assert isinstance(m, flat.DepthField)
    assert np.allclose(m(d), want, rtol=0, atol=1e-14)
    assert np.allclose(flat.horizontal_mean(p)(d), want, rtol=0, atol=1e-14)


def test_a_log_perturbation_takes_the_mean_of_its_exponential():
    ref = flat.DepthField(IV, 2.0)
    grid = flat.Grid(BOX, (8, 6))
    dp = flat.GriddedField.sampled(flat.AnalyticField(IV, lambda d, x, y: 0.3 * np.cos(
        np.pi * x) + 0.0 * d), grid, ngll=2, lateral="fourier")
    p = PerturbedField(ref, dp, kind="log")
    m = p.horizontal_mean()                                # on the field's own grid
    ex = ComposedField(np.exp, [dp], character=SCALAR)
    assert np.allclose(m([0.2, 0.7]), 2.0 * quadrature_mean(ex, grid)([0.2, 0.7]),
                       rtol=1e-15)
    # the mean of exp(0.3 cos(u)) is the Bessel function I0(0.3)
    from scipy.special import i0
    fine = flat.Grid(BOX, (64, 4))
    assert np.allclose(p.horizontal_mean(grid=fine)([0.5]), 2.0 * i0(0.3), rtol=1e-12)
    bare = PerturbedField(ref, flat.AnalyticField(IV, lambda d, x, y: 0.1 * x),
                          kind="log")
    with pytest.raises(ValueError, match="pass grid=, a cartesian.Grid"):
        bare.horizontal_mean()
    with pytest.raises(ValueError, match="pass grid=, a cartesian.Grid"):
        bare.horizontal_mean(grid=gauss_legendre(4))
    radial = PerturbedField(ref, flat.DepthField(IV, 0.1), kind="log")
    assert radial.horizontal_mean()(0.5) == pytest.approx(2.0 * np.exp(0.1))


def test_a_model_horizontally_averaged_rederives_its_moduli():
    model = flat.CrustOverMantle(flat.Box([(0.0, 4e5), (0.0, 2e5)]))
    grid = flat.Grid(model.geometry.domain, (8, 4))
    dlnvs = flat.AnalyticField((100e3, 200e3), lambda d, x, y: 0.02 * np.cos(
        2 * np.pi * x / 4e5) + 0.0 * d)
    m = perturbed(model, "vs", dlnvs, kind="relative")
    assert not m.layer(-1)["vs"].is_radial
    avg = flat.horizontally_averaged(m, grid=grid)
    vs, L = avg.layer(-1)["vs"], avg.layer(-1)["L"]
    assert vs.is_radial and L.is_radial
    assert vs(150e3) == pytest.approx(4500.0, rel=1e-14)
    assert L(150e3) == pytest.approx(3300.0 * 4500.0 ** 2, rel=1e-12)
    avg.validate()


def test_a_mean_of_one_system_refuses_an_object_of_the_other_by_name():
    sphere_field = AnalyticField((0.5, 1.0), lambda r, t, p: r * np.cos(t))
    with pytest.raises(ValueError, match="a horizontal mean is defined in the "
                                         "coordinates 'flat' alone; this "
                                         "AnalyticField is in 'spherical'"):
        flat.horizontal_mean(sphere_field, grid=flat.Grid(BOX, 4))
    flat_field = flat.AnalyticField(IV, cosines)
    with pytest.raises(ValueError, match="a spherical mean is defined in the "
                                         "coordinates 'spherical' alone"):
        spherical_mean(flat_field, grid=gauss_legendre(4))
    p = PerturbedField(flat.DepthField(IV, 1.0), flat_field, kind="absolute")
    with pytest.raises(ValueError, match="a spherical mean"):
        p.spherical_mean(grid=gauss_legendre(4))
    from planetmodel import RadialField
    q = PerturbedField(RadialField((0.5, 1.0), 1.0), sphere_field, kind="absolute")
    with pytest.raises(ValueError, match="a horizontal mean is defined in the "
                                         "coordinates 'flat' alone"):
        q.horizontal_mean(grid=flat.Grid(BOX, 4))
    from planetmodel.catalogue import LayeredIsotropicElastic
    sphere = LayeredIsotropicElastic([0.0, 1.0], rho=[1.0], vp=[2.0], vs=[1.0])
    with pytest.raises(ValueError, match="a horizontal mean is defined"):
        flat.horizontally_averaged(sphere)
