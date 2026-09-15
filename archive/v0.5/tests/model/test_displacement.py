"""Radial displacements, and the layer_linear rule.

The structural properties asserted here are the ones the whole design
rests on: h is continuous everywhere, exactly linear within each span,
identically zero wherever no relief drives it, and exactly zero at a
buffer's outer boundary -- which is what the exterior coupling needs and
what a band-with-decay rule would only achieve by luck.
"""
import numpy as np
import pytest

from planetmodel import PREM, Skeleton
from planetmodel.model.displacement import (BlendDisplacement, CallableDisplacement,
                                       RadialDisplacement, SumDisplacement,
                                       ZeroDisplacement, as_displacement,
                                       layer_linear)
from planetmodel.model.topography import AnalyticTopography, ZeroTopography
from planetmodel.registry import lookup
from planetmodel.testing import check_displacement


def relief(amp, name=None):
    """Smooth relief of the given amplitude."""
    return AnalyticTopography(lambda t, p: amp * np.cos(t) * np.cos(p),
                              name=name)


@pytest.fixture(scope="module")
def body():
    """PREM with relief on the surface and the CMB, plus a buffer."""
    return (PREM(ocean=False)
            .name_interface(1, "cmb")
            .name_interface(-1, "surface")
            .with_surface("surface", relief(3000.0, "surface"))
            .with_surface("cmb", relief(1500.0, "cmb"))
            .with_buffer(ratio=0.2))


@pytest.fixture(scope="module")
def h(body):
    return layer_linear()(body)


# ------------------------------------------------------------- contracts

def test_layer_linear_satisfies_the_contract(body, h):
    check_displacement(h, body.skeleton)


def test_zero_displacement_satisfies_the_contract(body):
    check_displacement(ZeroDisplacement(), body.skeleton)


def test_protocol_is_structural():
    assert isinstance(ZeroDisplacement(), RadialDisplacement)
    assert isinstance(lambda r, t, p: r, RadialDisplacement)


def test_the_rule_is_registered():
    assert lookup("displacement_rule", "layer_linear") is layer_linear


# ------------------------------------------- the structural properties

def test_h_is_continuous_across_every_knot(body, h):
    """C0 everywhere, which is all the weak forms require.

    dh/dr genuinely jumps at a knot, so the two one-sided values differ
    by roughly eps times the larger slope; the test probes close enough
    that the difference is far below the metre scale of the relief.
    """
    for k in h.knots:
        if not body.skeleton.boundaries[0] < k < body.skeleton.boundaries[-1]:
            continue
        eps = 1e-4
        assert h(k - eps, 0.7, 0.3) == pytest.approx(h(k + eps, 0.7, 0.3),
                                                     abs=1e-3)


def test_relief_reaches_only_the_adjacent_spans(body, h):
    """A consequence of the rule worth stating: relief is local.

    With a finely layered skeleton the interpolation confines a
    boundary's relief to the spans either side of it, without anyone
    asking for confinement.  PREM's outermost span is 12 km thick, so
    surface relief does not reach the 6346 km interface below it.
    """
    b = body.skeleton.boundaries
    surf = float(b[-2])
    below = float(b[-3])
    assert h(surf, 0.0, 0.0) != 0.0
    assert h(below, 0.0, 0.0) == pytest.approx(0.0, abs=1e-12)
    assert h(0.5 * (below + surf), 0.0, 0.0) != 0.0


def test_h_is_exactly_linear_within_each_span(body, h):
    """Three collinear points per span: the blend is linear in r."""
    b = body.skeleton.boundaries
    for lo, hi in zip(b[:-1], b[1:]):
        a, m, z = lo + 0.2 * (hi - lo), 0.5 * (lo + hi), hi - 0.2 * (hi - lo)
        va, vm, vz = (float(h(x, 0.9, -0.4)) for x in (a, m, z))
        want = va + (vz - va) * (m - a) / (z - a)
        assert vm == pytest.approx(want, abs=1e-9 * max(1.0, abs(want)))


def test_dh_dr_is_piecewise_constant(body, h):
    """Exactly constant within a span, and it is the analytic slope."""
    b = body.skeleton.boundaries
    for lo, hi in zip(b[:-1], b[1:]):
        rs = np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), 7)
        d = h.radial_derivative(rs, np.full_like(rs, 0.9),
                                np.full_like(rs, -0.4))
        assert np.ptp(d) < 1e-12 * max(1.0, np.max(np.abs(d)))
        want = (float(h(hi, 0.9, -0.4)) - float(h(lo, 0.9, -0.4))) / (hi - lo)
        assert d[0] == pytest.approx(want, rel=1e-9, abs=1e-15)


def test_knots_are_the_skeleton_boundaries(body, h):
    """The whole smoothness argument: kinks land on element boundaries."""
    assert np.allclose(h.knots, body.skeleton.boundaries)


def test_h_vanishes_where_no_relief_drives_it(body, h):
    """A span bounded by two bare interfaces is untouched."""
    icb, cmb = 1221.5e3, 3480e3
    for r in np.linspace(1e3, icb, 20):
        assert h(r, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    assert h(0.5 * (0.0 + icb), 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    # the CMB carries relief, so just above it h is non-zero
    assert abs(h(cmb + 1e3, 0.0, 0.0)) > 0.0


def test_h_is_exactly_zero_at_the_buffer_boundary(body, h):
    """What the exterior coupling requires, obtained by construction."""
    b = float(body.skeleton.boundaries[-1])
    for t in np.linspace(0.0, np.pi, 9):
        assert h(b, t, 0.6) == 0.0


def test_h_reproduces_the_relief_at_its_own_interface(body, h):
    surf = float(body.skeleton.boundaries[-2])
    for t, p in ((0.3, 0.2), (1.5, -2.0), (2.9, 1.1)):
        assert h(surf, t, p) == pytest.approx(
            body.surface("surface").height(t, p), rel=1e-12)


def test_confinement_is_geometry_not_a_rule_parameter():
    """An interface with no relief is where the displacement dies.

    The design claim in full.  On a coarse body, surface relief reaches
    far down; inserting a bare interface stops it there, with no change
    to the rule and nothing to configure.
    """
    from planetmodel import ReferenceBody, Skeleton

    coarse = ReferenceBody.from_fields(Skeleton([0.0, 3480e3, 6371e3]), {})
    coarse = coarse.name_interface(-1, "surface").with_surface(
        "surface", relief(3000.0))
    floor = 6.0e6
    confined = coarse.refined([floor], names=["floor"], role="control")

    h_open = layer_linear()(coarse)
    h_shut = layer_linear()(confined)

    deep = 5.0e6
    assert abs(h_open(deep, 0.5, 0.5)) > 100.0          # reaches far down
    assert h_shut(deep, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)
    assert h_shut(floor, 0.5, 0.5) == pytest.approx(0.0, abs=1e-12)

    top = float(coarse.skeleton.boundaries[-1])         # unchanged at the top
    assert h_shut(top, 0.5, 0.5) == pytest.approx(h_open(top, 0.5, 0.5),
                                                  rel=1e-12)
    assert floor in h_shut.knots


def test_angular_gradient_blends_the_reliefs_gradients(body, h):
    surf = float(body.skeleton.boundaries[-2])
    gt, gp = h.angular_gradient(surf, 0.8, 0.5)
    rt, rp = body.surface("surface").gradient(0.8, 0.5)
    assert float(gt) == pytest.approx(float(rt), rel=1e-6)
    assert float(gp) == pytest.approx(float(rp), rel=1e-6)


# ------------------------------------------------------------- options

def test_inner_taper_vanishes_below_its_radius(body):
    taper = 2.0e6
    h = layer_linear(inner_taper_radius=taper)(body)
    assert taper in h.knots
    for r in np.linspace(1e4, taper, 15):
        assert h(r, 1.0, 1.0) == pytest.approx(0.0, abs=1e-12)
    check_displacement(h, body.skeleton)


def test_control_radii_add_knots_without_interfaces(body):
    extra = 4.0e6
    h = layer_linear(control_radii=(extra,))(body)
    assert extra in h.knots
    assert extra not in set(body.skeleton.boundaries)


def test_control_radii_must_lie_inside_the_body(body):
    with pytest.raises(ValueError, match="outside the body"):
        layer_linear(control_radii=(1e9,))(body)


def test_the_rule_is_a_frozen_dataclass():
    a, b = layer_linear(), layer_linear()
    assert a == b
    with pytest.raises(Exception):
        a.inner_taper_radius = 1.0


# ------------------------------------------------------------ adapters

def test_a_bare_callable_is_adapted():
    d = as_displacement(lambda r, t, p: 100.0 * np.cos(t))
    assert d.knots == ()
    r = np.linspace(1e5, 6e6, 20)
    assert np.allclose(d(r, 0.5, 0.5), 100.0 * np.cos(0.5))
    assert np.allclose(d.radial_derivative(r, 0.5, 0.5), 0.0, atol=1e-6)


def test_declared_knots_are_carried_through():
    d = as_displacement(lambda r, t, p: np.abs(r - 4e6), knots=(4e6,))
    assert d.knots == (4e6,)


def test_analytic_derivatives_are_used_when_given():
    d = CallableDisplacement(
        lambda r, t, p: r ** 2,
        radial_derivative=lambda r, t, p: 2.0 * r)
    r = np.linspace(1.0, 5.0, 10)
    assert np.allclose(d.radial_derivative(r, 0.0, 0.0), 2.0 * r, rtol=1e-15)


def test_as_displacement_preserves_a_rich_object(body, h):
    assert as_displacement(h) is h


def test_adapters_reject_non_callables():
    with pytest.raises(TypeError, match="callable"):
        as_displacement(42)


# ------------------------------------------------------------ algebra

def test_sum_and_scaling(body):
    a = layer_linear()(body)
    total = a + a
    assert total(5e6, 0.7, 0.2) == pytest.approx(2.0 * a(5e6, 0.7, 0.2))
    assert (3.0 * a)(5e6, 0.7, 0.2) == pytest.approx(3.0 * a(5e6, 0.7, 0.2))
    assert (-a)(5e6, 0.7, 0.2) == pytest.approx(-a(5e6, 0.7, 0.2))


def test_sum_unions_the_knots(body):
    a = layer_linear()(body)
    b = as_displacement(lambda r, t, p: np.zeros_like(np.asarray(r, float)),
                        knots=(1.234e6,))
    assert 1.234e6 in (a + b).knots
    assert set(a.knots) <= set((a + b).knots)


def test_displacement_times_displacement_is_refused(body, h):
    with pytest.raises(TypeError, match="not defined"):
        h * h


def test_empty_sum_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SumDisplacement(())


# ------------------------------------------------- the check itself bites

def test_check_catches_an_undeclared_kink(body):
    """A displacement that lies about its knots must not pass."""
    liar = CallableDisplacement(lambda r, t, p: np.abs(r - 4e6), knots=())
    with pytest.raises(AssertionError, match="knot list"):
        check_displacement(liar, body.skeleton)


def test_check_accepts_the_same_kink_once_declared(body):
    honest = CallableDisplacement(lambda r, t, p: np.abs(r - 4e6),
                                  knots=(4e6,))
    check_displacement(honest, body.skeleton)


def test_blend_validates_its_inputs():
    with pytest.raises(ValueError, match="at least two"):
        BlendDisplacement([1.0], [ZeroTopography()])
    with pytest.raises(ValueError, match="increasing"):
        BlendDisplacement([2.0, 1.0], [ZeroTopography()] * 2)
    with pytest.raises(ValueError, match="reliefs for"):
        BlendDisplacement([1.0, 2.0], [ZeroTopography()])


def test_bounds_come_from_the_reliefs():
    sk = Skeleton([0.0, 1.0e6, 2.0e6])
    from planetmodel.model.topography import GriddedTopography
    lons, lats = np.arange(-179.0, 180.0, 2.0), np.arange(-89.0, 90.0, 2.0)
    v = np.linspace(-500.0, 500.0, lats.size)[:, None] * np.ones(lons.size)
    g = GriddedTopography(lons, lats, v)
    d = BlendDisplacement(sk.boundaries, [ZeroTopography(), g, ZeroTopography()])
    lo, hi = d.bounds()
    assert (lo, hi) == pytest.approx((-500.0, 500.0))


def test_relief_drives_the_interface_it_is_attached_to():
    """A surface attached to an interface must belong at that radius.

    The relief a rule reads is the relief at the interface's own knot,
    so a Surface whose reference radius says otherwise is two answers to
    where the boundary is.  Attachment refuses it and names the fix; a
    surface placed where the interface is drives that knot and nothing
    below it.
    """
    from planetmodel import PREM
    from planetmodel.model.surface import Surface

    body = PREM(ocean=False).name_interface(1, "cmb")
    cmb = body.interface("cmb").radius

    odd = Surface(1.0e6, topography=relief(2000.0))          # bookkeeping says 1000 km
    with pytest.raises(ValueError, match="reference radius"):
        body.with_surface("cmb", odd)

    b = body.with_surface("cmb", odd.at(cmb))
    h = layer_linear()(b)
    assert h(cmb, 0.3, 0.2) == pytest.approx(odd.height(0.3, 0.2), rel=1e-12)
    assert h(1.0e6, 0.3, 0.2) == pytest.approx(0.0, abs=1e-12)


def test_a_taper_at_or_beyond_the_last_knot_is_refused():
    with pytest.raises(ValueError, match="taper radius"):
        BlendDisplacement([0.0, 1.0, 2.0], [ZeroTopography()] * 3, taper_radius=2.0)
    with pytest.raises(ValueError, match="taper radius"):
        layer_linear(inner_taper_radius=7.0e6)(PREM())


def test_wrapping_a_conforming_displacement_with_extras_is_refused():
    with pytest.raises(ValueError, match="already declares"):
        as_displacement(ZeroDisplacement(), knots=(1.0,))
