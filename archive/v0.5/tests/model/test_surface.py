"""Surfaces: a topography placed at a radius, and attached to a body."""
import warnings

import numpy as np
import pytest

from planetmodel import PREM
from planetmodel.model.surface import (Surface, ellipsoid_surface,
                                  spherical_surface)
from planetmodel.model.topography import (AnalyticTopography, ZeroTopography)


@pytest.fixture(scope="module")
def relief():
    """Zero-mean relief: +/- 3 km, varying in both angles."""
    return AnalyticTopography(
        lambda t, p: 3000.0 * (np.cos(t) ** 2 - 1.0 / 3.0) * np.cos(p))


@pytest.fixture(scope="module")
def body():
    return (PREM(ocean=False)
            .name_interface(1, "cmb")
            .name_interface(-1, "surface"))


# ------------------------------------------------------------ placement

def test_radius_is_reference_plus_relief(relief):
    s = Surface(6.371e6, topography=relief)
    t, p = 0.7, 1.2
    assert s.radius(t, p) == pytest.approx(6.371e6 + relief(t, p))
    assert s.height(t, p) == pytest.approx(relief(t, p))


def test_a_bare_callable_is_adapted():
    s = Surface(1.0e6, topography=lambda t, p: np.zeros_like(
        np.asarray(t, dtype=float)))
    assert hasattr(s.topography, "gradient")
    assert s.radius(0.5, 0.5) == pytest.approx(1.0e6)


def test_default_relief_is_flat():
    s = Surface(2.0e6)
    assert isinstance(s.topography, ZeroTopography)
    assert s.radius(1.0, 1.0) == pytest.approx(2.0e6)
    assert spherical_surface(2.0e6).radius(1.0, 1.0) == pytest.approx(2.0e6)


def test_reference_radius_is_validated():
    with pytest.raises(ValueError, match="positive"):
        Surface(-1.0)
    with pytest.raises(ValueError, match="positive"):
        Surface(0.0)
    with pytest.raises(ValueError, match="finite"):
        Surface(np.inf)


def test_surfaces_are_frozen(relief):
    s = Surface(1e6, topography=relief)
    with pytest.raises(Exception):
        s.reference_radius = 2e6


def test_gradient_is_the_topographys(relief):
    s = Surface(6e6, topography=relief)
    t = np.linspace(0.2, 2.9, 20)
    p = np.linspace(-3.0, 3.0, 20)
    gt, gp = s.gradient(t, p)
    rt, rp = relief.gradient(t, p)
    assert np.allclose(gt, rt) and np.allclose(gp, rp)


def test_mean_radius_and_centring(relief):
    s = Surface(6.371e6, topography=relief)
    assert s.mean_radius() == pytest.approx(6.371e6, abs=50.0)
    assert s.is_centred(atol=50.0)

    off = Surface(6.371e6, topography=AnalyticTopography(
        lambda t, p: np.full_like(np.asarray(t, dtype=float), 1000.0)))
    assert not off.is_centred(atol=1.0)
    assert off.mean_radius() == pytest.approx(6.372e6, abs=1.0)

    fixed = off.centred()
    assert fixed.reference_radius == pytest.approx(6.372e6, abs=1.0)
    assert fixed.is_centred(atol=1.0)
    t, p = 1.0, 1.0
    assert fixed.radius(t, p) == pytest.approx(off.radius(t, p), abs=1e-6)


# ------------------------------------------------------------ arithmetic

def test_scaling_exaggerates_relief_not_radius(relief):
    """The exaggeration factor must not move the boundary itself."""
    s = Surface(6.371e6, topography=relief)
    big = s * 20.0
    assert big.reference_radius == s.reference_radius
    t, p = 0.9, -0.4
    assert big.height(t, p) == pytest.approx(20.0 * s.height(t, p))
    assert (20.0 * s).height(t, p) == pytest.approx(20.0 * s.height(t, p))


def test_adding_surfaces_at_one_radius(relief):
    a = Surface(6e6, topography=relief)
    b = Surface(6e6, topography=AnalyticTopography(lambda t, p: 100.0 * np.cos(t)))
    t, p = 1.1, 0.3
    assert (a + b).height(t, p) == pytest.approx(a.height(t, p) + b.height(t, p))


def test_adding_surfaces_at_different_radii_is_refused(relief):
    with pytest.raises(ValueError, match="different reference radii"):
        Surface(6e6, topography=relief) + Surface(7e6, topography=relief)


def test_adding_a_bare_topography(relief):
    s = Surface(6e6, topography=relief)
    t, p = 0.5, 0.5
    got = s + AnalyticTopography(lambda a, b: np.full_like(
        np.asarray(a, dtype=float), 10.0))
    assert got.height(t, p) == pytest.approx(s.height(t, p) + 10.0)


def test_surface_times_surface_is_refused(relief):
    s = Surface(6e6, topography=relief)
    with pytest.raises(TypeError, match="not defined"):
        s * s


def test_at_and_with_topography(relief):
    s = Surface(6e6, topography=relief, name="moho")
    moved = s.at(6.3e6)
    assert moved.reference_radius == pytest.approx(6.3e6)
    assert moved.name == "moho"
    assert moved.height(1.0, 1.0) == pytest.approx(s.height(1.0, 1.0))
    flat = s.with_topography(ZeroTopography())
    assert flat.height(1.0, 1.0) == 0.0


def test_bounds_come_from_the_relief():
    s = Surface(6e6, topography=AnalyticTopography(lambda t, p: np.cos(t)))
    assert s.bounds() is None            # analytic relief knows no bounds
    from planetmodel.model.topography import GriddedTopography
    lons, lats = np.arange(-179.0, 180.0, 2.0), np.arange(-89.0, 90.0, 2.0)
    v = np.linspace(-100.0, 100.0, lats.size)[:, None] * np.ones(lons.size)
    g = Surface(6e6, topography=GriddedTopography(lons, lats, v))
    lo, hi = g.bounds()
    assert (lo, hi) == pytest.approx((6e6 - 100.0, 6e6 + 100.0))


# ------------------------------------------------------------- ellipsoid

def test_ellipsoid_semi_axes_are_reproduced():
    a, b, c = 6378e3, 6370e3, 6357e3
    e = ellipsoid_surface(a, b, c)
    assert e.radius(np.pi / 2, 0.0) == pytest.approx(a, rel=1e-12)
    assert e.radius(np.pi / 2, np.pi / 2) == pytest.approx(b, rel=1e-12)
    assert e.radius(0.0, 0.0) == pytest.approx(c, rel=1e-12)


def test_ellipsoid_relief_has_zero_mean():
    """An ellipsoid is described as a sphere plus a departure from it."""
    e = ellipsoid_surface(6378e3, 6378e3, 6357e3)
    assert e.is_centred(atol=1.0)
    assert e.mean_radius() == pytest.approx(e.reference_radius, abs=1.0)


def test_the_ellipsoids_reference_radius_is_the_mean_radius():
    """WGS84's mean radius, against an independent midpoint rule.

    The reference radius an ellipsoid_surface reports is a quadrature
    result, so it is worth one check against a quadrature that shares no
    code with it: a 2000 x 4000 midpoint rule in (cos theta, phi) over
    the same radius function.  The published mean radius of WGS84,
    (2a + c)/3 = 6371008.7 m, is a different average and is not what
    this is; the area-weighted one is about fourteen metres below it.
    """
    a, c = 6378137.0, 6356752.0
    e = ellipsoid_surface(a, a, c)

    x = (np.arange(2000) + 0.5) / 1000.0 - 1.0             # midpoints in cos
    ph = (np.arange(4000) + 0.5) * (2.0 * np.pi / 4000.0) - np.pi
    theta = np.arccos(x)[:, None]
    r = e.radius(theta, ph[None, :])
    reference = float(r.mean())

    assert e.reference_radius == pytest.approx(reference, rel=1e-6)
    assert e.mean_radius() == pytest.approx(e.reference_radius, abs=1e-6)


def test_a_sphere_is_the_degenerate_ellipsoid():
    e = ellipsoid_surface(6e6, 6e6, 6e6)
    assert e.reference_radius == pytest.approx(6e6, rel=1e-9)
    for t, p in ((0.0, 0.0), (1.0, 2.0), (np.pi, 0.0)):
        assert e.radius(t, p) == pytest.approx(6e6, rel=1e-9)


def test_ellipsoid_semi_axes_are_validated():
    with pytest.raises(ValueError, match="semi-axis"):
        ellipsoid_surface(-1.0, 1.0, 1.0)


# ------------------------------------------------------------ attachment

def test_attaching_a_topography_uses_the_interface_radius(body, relief):
    b = body.with_surface("surface", relief)
    s = b.surface("surface")
    assert s.reference_radius == pytest.approx(body.interface("surface").radius)
    assert s.name == "surface"


def test_attaching_a_surface_requires_it_to_belong_there(body, relief):
    """The zero-mean contract, refused twice and accepted once.

    An interface radius is the boundary's mean radius, so a Surface may
    be attached to it only if it says the same thing: placed at that
    radius, carrying relief of zero mean.  A Surface placed elsewhere,
    or carrying its own mean, describes a different boundary, and the
    two disagreements are the ones that used to pass unnoticed.
    """
    cmb = body.interface("cmb").radius

    with pytest.raises(ValueError, match="reference radius"):
        body.with_surface("cmb", Surface(1.0e6, topography=relief))

    off = AnalyticTopography(lambda t, p: 1000.0 + relief(t, p))
    with pytest.raises(ValueError, match="relief of mean"):
        body.with_surface("cmb", Surface(cmb, topography=off))

    b = body.with_surface("cmb", Surface(cmb, topography=relief, name="cmb"))
    assert b.surface("cmb").reference_radius == pytest.approx(cmb)
    assert b.surface("cmb").is_centred(atol=1e-6)


def test_attaching_a_bare_topography_centres_it_and_says_so(body, relief):
    """A mean in the relief is a mean in the wrong place, and moves."""
    off = AnalyticTopography(lambda t, p: 1000.0 + relief(t, p))
    with pytest.warns(UserWarning, match="area-weighted mean of 1000"):
        b = body.with_surface("cmb", off)
    s = b.surface("cmb")
    assert s.reference_radius == pytest.approx(body.interface("cmb").radius)
    assert s.is_centred(atol=1e-6)
    t, p = 0.7, -1.1
    assert s.height(t, p) == pytest.approx(relief(t, p), abs=1e-6)


def test_centred_relief_is_attached_without_a_warning(body, relief):
    """Nothing to report when the shape already meets the contract."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        b = body.with_surface("cmb", relief)
    assert b.surface("cmb").is_centred(atol=1e-6)


def test_attachment_is_copy_on_write(body, relief):
    b = body.with_surface("surface", relief)
    assert body.surfaces == {}
    assert len(b.surfaces) == 1


def test_surfaces_survive_further_surgery(body, relief):
    b = body.with_surface("surface", relief).with_buffer(ratio=0.2)
    assert len(b.surfaces) == 1


def test_removing_a_surface(body, relief):
    b = body.with_surface("surface", relief)
    assert b.without_surface("surface").surfaces == {}


def test_surface_lookup_by_index_and_name(body, relief):
    b = body.with_surface(-1, relief)
    assert b.surface(-1) is not None
    assert b.surface("surface") is b.surface(-1)
    assert b.surface("cmb") is None


def test_attaching_nonsense_is_refused(body):
    with pytest.raises(TypeError, match="Surface or a Topography"):
        body.with_surface("surface", 3.0)


# ---------------------------------------- identity across surgery (review fix)

def test_surfaces_stay_on_their_interface_through_refinement(body, relief):
    """Regression: surfaces were keyed by interface index, and refined()
    renumbers indices, so a surface silently migrated onto the inserted
    interface -- surface('floor') returned the surface relief and
    surface('surface') returned None.  An interface's identity through
    surgery is its radius, and the store now keys on that.
    """
    b = body.with_surface("surface", relief)
    after = b.refined([6.0e6], names=["floor"], role="control")
    assert after.surface("floor") is None
    got = after.surface("surface")
    assert got is not None
    assert got.reference_radius == pytest.approx(
        body.interface("surface").radius)


def test_a_surface_on_a_truncated_boundary_is_dropped(body, relief):
    """Cutting the body below a surface removes the surface with it."""
    b = body.with_surface("surface", relief)
    cut = b.truncated(5.0e6)
    assert cut.surfaces == {}


def test_a_surface_survives_coarsening_elsewhere(body, relief):
    b = body.with_surface("surface", relief)
    coarse, _ = b.coarsened(drop=[0], state="fluid")
    (idx,) = coarse.surfaces
    assert coarse.surfaces[idx].reference_radius == pytest.approx(
        body.interface("surface").radius)
