"""The slice contracts hold the sphere's slices: points, layers and values
against the model, the labels and the normalisation; and its spectra."""
import dataclasses

import numpy as np
import pytest

from planetmodel import (VECTOR, AnalyticField, LayeredIsotropicElastic,
                         driscoll_healy, equiangular, flattening, gauss_legendre)
from planetmodel import slices
from planetmodel.frames import cartesian_points
from planetmodel.testing import check_slice, check_spectrum


def lateral(r, t, p):
    return 4.0 + 0.3 * r * np.cos(t) + 0.2 * r * np.sin(t) * np.cos(p)


def flow(r, t, p):
    X = cartesian_points(r, t, p)
    return np.stack([X[..., 1], -X[..., 0], 0.5 * X[..., 2]], axis=-1)


@pytest.fixture
def two():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    m = m.with_field(1, "vs", AnalyticField((0.5, 1.0), lateral, name="vs"),
                     replace=True)
    return m.with_field(1, "flow", AnalyticField((0.5, 1.0), flow, character=VECTOR,
                                                 frame="cartesian", name="flow"))


@pytest.mark.parametrize("stretch", [None, 0.05], ids=["identity", "flattened"])
def test_every_spherical_slice_meets_the_slice_contract(two, stretch):
    m = two if stretch is None else two.stretched(flattening(stretch))
    grid = gauss_legendre(6)
    for at in ("reference", "physical"):
        check_slice(slices.sphere(m, "vs", radius=0.8, grid=grid, at=at), m)
        check_slice(slices.section(m, "vs", (0.4, 0.1), end=(1.9, 2.0), n=4, m=7,
                                   at=at), m)
        check_slice(slices.profile(m, "rho", 0.3, 1.0, n=4, at=at), m)
    check_slice(slices.sphere(m, "rho", radius=0.5, grid=grid, side="below"), m)
    check_slice(slices.sphere(m, "flow", radius=0.8, grid=grid, component=1,
                              frame="spherical"), m)
    check_slice(slices.sphere(m, "vs", radius=0.8, grid=grid, relative_to="mean"), m)
    check_slice(slices.mean_profile(m, "vs", mean_grid=grid, n=4), m)
    check_slice(slices.integrated(m, "vs", grid=grid), m)
    check_slice(slices.cutaway(m, "vs", n=24, inner=0.3), m)
    check_spectrum(slices.spectrum(m, "vs", radius=0.8, lmax=4))


def test_the_slice_contract_holds_a_physical_section_to_the_mapped_body(two):
    m = two.stretched(flattening(0.1))
    # cut at the reference outer radius, the equator's raised surface left
    # out: refused as the whole body, held to the bounds it was given
    for cut in (slices.section(m, "vs", (0.4, 0.1), end=(1.9, 2.0), n=4, m=7,
                               at="physical", radii=(0.0, 1.0)),
                slices.profile(m, "vs", 1.5, 0.3, n=4, at="physical",
                               radii=(0.0, 1.0))):
        with pytest.raises(AssertionError, match="where the mapping puts"):
            check_slice(cut, m)
        check_slice(cut, m, bounds=(0.0, 1.0))
    # a layer within the bounds may not be left out
    outer = slices.section(m, "vs", (0.4, 0.1), end=(1.9, 2.0), n=4, m=7,
                           at="physical", radii=(0.6, 2.0))
    with pytest.raises(AssertionError, match="leaves it out"):
        check_slice(outer, m)
    check_slice(outer, m, bounds=(0.6, 2.0))


def _wind(r, t, p):
    """One vector everywhere, Cartesian: (1, 0.5, 0)."""
    shape = np.broadcast(r, t, p).shape
    return np.broadcast_to(np.array([1.0, 0.5, 0.0]), shape + (3,))


@pytest.fixture
def windy(two):
    """`two` with a constant vector field on both layers: its components in
    the spherical frame turn with the longitude a point is read at."""
    m = two
    for layer in range(2):
        m = m.with_field(layer, "wind", AnalyticField(
            m.skeleton.interval(layer), _wind, character=VECTOR, frame="cartesian",
            name="wind"))
    return m


def _polar_slices(m, k):
    """Slices of component `k` of the wind in the spherical frame with
    points that fix no longitude, or no angle: profiles at each pole,
    through the centre; sections starting at a pole, arriving at one and
    crossing one; spheres on grids holding a pole, or both."""
    kw = {"component": k, "frame": "spherical"}
    out = []
    for at in ("reference", "physical"):
        for theta, phi in ((0.0, 0.5 * np.pi), (0.0, -2.0), (np.pi, 0.7),
                           (0.3, 1.0)):
            out.append(slices.profile(m, "wind", theta, phi, n=3, at=at, **kw))
        for start, end in (((0.0, 0.3), (1.0, 2.0)), ((np.pi, 5.0), (1.0, 2.0)),
                           ((1.0, 2.0), (0.0, 0.3)), ((1.0, 2.0), (np.pi, 0.3))):
            out.append(slices.section(m, "wind", start, end=end, n=3, m=5, at=at,
                                      **kw))
        out.append(slices.section(m, "wind", (0.5, 1.0), azimuth=0.0, length=1.0,
                                  n=3, m=5, at=at, **kw))
        for grid in (driscoll_healy(4), equiangular(5, 8, poles=True)):
            out.append(slices.sphere(m, "wind", radius=0.8, grid=grid, at=at, **kw))
    return out


@pytest.mark.parametrize("stretch", [None, 0.05], ids=["identity", "flattened"])
def test_the_slice_contract_reads_a_point_that_fixes_no_angle_as_its_slice_does(
        windy, stretch):
    """A point on the polar axis fixes no longitude and the centre no angle,
    and the frame of a component there is that of the slice's own line or
    node: the check reads each such point at the angles its slice did."""
    m = windy if stretch is None else windy.stretched(flattening(stretch))
    for k in range(3):
        for cut in _polar_slices(m, k):
            # every point is checked, the polar ones among them
            size = sum(lay.values.size for lay in getattr(cut, "layers", ())) or (
                cut.values.size)
            for seed in range(3):
                check_slice(cut, m, n=size, rng=np.random.default_rng(seed))
    # the middle ray of the section crossing a pole is on the axis
    across = slices.section(m, "wind", (0.5, 1.0), azimuth=0.0, length=1.0, n=3, m=5,
                            component="norm")
    assert np.all(across.path[2, :2] == 0.0)
    theta, phi = across.lateral
    assert theta[2] == 0.0 and np.isclose(phi[2], 1.0)       # arriving up phi = 1
    assert slices.profile(m, "wind", 0.0, -2.0, n=3,
                          component="norm").lateral == (0.0, -2.0)
    mean = slices.mean_profile(m, "vs", mean_grid=gauss_legendre(4), n=3)
    assert mean.lateral is None


@pytest.mark.parametrize("at", ["reference", "physical"])
def test_the_slice_contract_refuses_values_read_at_another_longitude(windy, at):
    """A slice whose values on the polar axis are those of another longitude,
    zero say, is refused: of a profile at a pole, every value; of a section
    from a pole, its first ray; of a sphere on a grid holding a pole, that
    row."""
    m = windy.stretched(flattening(0.05))
    kw = {"component": 2, "frame": "spherical", "at": at}
    wrong = "differ from the model's field evaluated at its reference points"

    def refused(cut, bounds=None):
        for seed in range(20):
            try:
                check_slice(cut, m, rng=np.random.default_rng(seed), n=64,
                            bounds=bounds)
            except AssertionError as error:
                assert wrong in str(error)
                return True
        return False

    for theta in (0.0, np.pi):
        right = slices.profile(m, "wind", theta, 1.5, n=3, **kw)
        other = slices.profile(m, "wind", theta, 0.0, n=3, **kw)
        check_slice(right, m)
        assert all(np.allclose(a.reference, b.reference, rtol=0, atol=1e-15)
                   for a, b in zip(right.layers, other.layers))
        # the same points under the longitude asked for, with the values of
        # longitude zero
        assert refused(dataclasses.replace(other, phi=1.5))
        # points off the axis and away from the centre fix their angles,
        # whatever the profile says of itself; the centre fixes none
        off = slices.profile(m, "wind", 0.3, 1.0, n=3, radii=(0.2, 0.9), **kw)
        assert not refused(dataclasses.replace(off, phi=0.0), bounds=(0.2, 0.9))
        through = slices.profile(m, "wind", 0.3, 1.0, n=3, **kw)
        assert refused(dataclasses.replace(through, phi=0.0))
    sec = slices.section(m, "wind", (0.0, 0.3), end=(1.0, 2.0), n=3, m=5, **kw)
    zero = slices.profile(m, "wind", 0.0, 0.0, n=3, **kw)
    layers = []
    for lay, at_zero in zip(sec.layers, zero.layers):
        values = lay.values.copy()
        assert not np.allclose(values[:, 0], at_zero.values, rtol=0, atol=1e-3)
        values[:, 0] = at_zero.values
        layers.append(dataclasses.replace(lay, values=values))
    check_slice(sec, m)
    assert refused(dataclasses.replace(sec, layers=tuple(layers)))
    grid = equiangular(5, 8, poles=True)
    ball = slices.sphere(m, "wind", radius=0.8, grid=grid, **kw)
    values = ball.values.copy()
    values[0] = values[0, 0]                       # the pole's row at one longitude
    check_slice(ball, m)
    assert refused(dataclasses.replace(ball, values=values))
