"""Dimensions, Scales, and the SI-at-rest / non-dimensional-at-work pipeline.

The oracles here are physical identities, not implementation echoes: G
becomes 1 under geophysical scales because the time scale is chosen to
make it so; F and J are dimensionless so a rescaled body must reproduce
them exactly; and the existing 1D gravity solver, untouched, must give
the same surface gravity from the non-dimensional body as from the SI
one, once re-dimensionalised.
"""

import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import (EARTH_MEAN_DENSITY, PREM, Dimensions, RadialField, Scales,
                         layer_linear, polynomial_layer)
from planetmodel.mesh1d import gravity
from planetmodel.model.fields.layer_function import rescale_layer_function
from planetmodel.model.topography import AnalyticTopography


@pytest.fixture(scope="module")
def prem():
    return PREM(ocean=False)


@pytest.fixture(scope="module")
def nd(prem):
    return prem.nondimensionalised()


# ------------------------------------------------------------- Dimensions

def test_dimension_algebra_is_the_physics():
    D = Dimensions
    assert D.VELOCITY == D.LENGTH / D.TIME
    assert D.MODULUS == D.DENSITY * D.VELOCITY ** 2
    assert D.GRAVITY == D.LENGTH / D.TIME ** 2
    assert D.GRAVITATIONAL_CONSTANT == D.LENGTH ** 3 / (D.MASS * D.TIME ** 2)
    assert D.DIMENSIONLESS.is_dimensionless
    assert not D.DENSITY.is_dimensionless


def test_dimensions_are_frozen_and_integer():
    with pytest.raises(TypeError, match="integer"):
        Dimensions(mass=0.5)
    with pytest.raises(Exception):
        Dimensions.DENSITY.mass = 2


def test_standard_fields_are_annotated(prem):
    D = Dimensions
    assert prem.rho.dimensions == D.DENSITY
    assert prem.vpv.dimensions == D.VELOCITY
    assert prem.qmu.dimensions == D.DIMENSIONLESS
    assert prem.A.dimensions == D.MODULUS
    assert prem["elastic_moduli"].dimensions == D.MODULUS


def test_dimensions_and_character_are_independent(prem):
    """The reason dimensions exist at all: Character cannot carry units."""
    assert prem.qmu.character == prem.vpv.character          # both SCALAR
    assert prem.qmu.dimensions != prem.vpv.dimensions


def test_derivative_divides_dimensions_by_length(prem):
    assert prem.rho.derivative().dimensions == (
        Dimensions.DENSITY / Dimensions.LENGTH)


# ----------------------------------------------------------------- Scales

def test_si_is_the_identity():
    s = Scales.SI
    assert s.is_si
    assert s.factor(Dimensions.MODULUS) == 1.0
    assert s.gravitational_constant == pytest.approx(6.6743e-11)


def test_geophysical_makes_G_one():
    s = Scales.geophysical(6.371e6)
    assert s.gravitational_constant == pytest.approx(1.0, abs=1e-14)
    assert s.density == pytest.approx(EARTH_MEAN_DENSITY)
    assert s.length == 6.371e6


def test_the_density_is_prescribed_not_computed(prem):
    """The convention is a choice; a test ties it to reality.

    PREM's actual mean density -- 3 * integral(rho r^2) / a^3 -- lands
    within a fraction of a percent of the prescribed 5515, which is why
    prescribing is safe as well as simple.
    """
    a = float(prem.skeleton.boundaries[-1])
    r = np.linspace(0.0, a, 200_001)
    mean = 3.0 * np.trapezoid(prem.rho.evaluate(r) * r ** 2, r) / a ** 3
    assert mean == pytest.approx(EARTH_MEAN_DENSITY, rel=5e-3)


def test_scales_validate():
    with pytest.raises(ValueError, match="positive finite"):
        Scales(length=-1.0)
    with pytest.raises(ValueError, match="positive"):
        Scales.geophysical(0.0)


# ------------------------------------------------------ layer functions

def test_ppoly_rescale_is_exact_and_stays_ppoly():
    f = polynomial_layer([1.0, 2.0, -0.5, 0.25], (1.0e6, 2.0e6), scale=6.371e6)
    k, vr = 1.0 / 6.371e6, 1.0 / 5515.0
    g = rescale_layer_function(f, k, vr)
    assert isinstance(g, PPoly)
    x = np.linspace(1.0e6 * k, 2.0e6 * k, 500)
    assert np.allclose(g(x), vr * f(x / k), rtol=1e-15)
    back = rescale_layer_function(g, 1.0 / k, 1.0 / vr)
    r = np.linspace(1.0e6, 2.0e6, 500)
    assert np.allclose(back(r), f(r), rtol=1e-14)


def test_wrapped_rescale_is_pointwise_exact():
    g = rescale_layer_function(lambda r: np.sin(np.asarray(r) / 1e6), 2.0, 3.0)
    assert g(np.array([2.0e6]))[0] == pytest.approx(3.0 * np.sin(1.0))


# ------------------------------------------------------------- the body

def test_nondimensionalised_body_is_pleasant(nd):
    assert nd.skeleton.boundaries[-1] == pytest.approx(1.0)
    assert float(nd.rho.evaluate(0.0)) == pytest.approx(2.373, abs=1e-3)
    assert 0.1 < float(nd.vpv.evaluate(0.8)) < 10.0
    assert 0.1 < float(nd.A.evaluate(0.8)) < 100.0


def test_construction_declares_and_rescaled_converts(prem, nd):
    """The declare/convert split, exercised."""
    s = nd.scales
    # declaration: same numbers, different meaning
    declared = prem  # SI by default
    assert declared.scales.is_si
    # conversion: values divided by the appropriate factor
    r_si, r_nd = 5.0e6, 5.0e6 / s.length
    assert float(nd.rho.evaluate(r_nd)) == pytest.approx(
        float(prem.rho.evaluate(r_si)) / s.density, rel=1e-14)
    assert float(nd.A.evaluate(r_nd)) == pytest.approx(
        float(prem.A.evaluate(r_si)) / s.modulus, rel=1e-14)
    assert float(nd.qmu.evaluate(r_nd)) == pytest.approx(
        float(prem.qmu.evaluate(r_si)), rel=1e-14)      # dimensionless


def test_round_trip_is_machine_precision(prem, nd):
    back = nd.redimensionalised()
    assert back.scales.is_si
    r = np.linspace(1e3, 6.36e6, 2000)
    for name in ("rho", "vpv", "A", "qmu"):
        want = prem[name].evaluate(r)
        got = back[name].evaluate(r)
        assert np.allclose(got, want, rtol=1e-14), name
    assert np.allclose(back.skeleton.boundaries, prem.skeleton.boundaries,
                       rtol=1e-15)


def test_exactness_survives(nd):
    """Non-dimensional PREM is still an exact polynomial model."""
    assert isinstance(nd.rho[0].function, PPoly)
    assert isinstance(nd.A[0].function, PPoly)


def test_velocity_views_rescale_through_their_sources(prem):
    """Composed views are dimensionally homogeneous, so rescaling the
    sources rescales the view."""
    stripped = PREM(ocean=False)
    for k in ("vpv", "vsv", "vph", "vsh", "eta"):
        stripped = stripped.without_field(k)
    from planetmodel.io.deck import attach_velocity_views
    attach_velocity_views(stripped)
    nd = stripped.nondimensionalised()
    s = nd.scales
    r_si = 5.0e6
    assert float(nd["vpv"].evaluate(r_si / s.length)) == pytest.approx(
        float(prem.vpv.evaluate(r_si)) / s.velocity, rel=1e-12)


def test_annotations_scales_and_surfaces_carry(prem):
    body = (prem.name_interface(-1, "surface")
            .with_surface("surface", AnalyticTopography(
                lambda t, p: 3.0e3 * np.cos(t)))
            .with_buffer(ratio=0.2))
    nd = body.nondimensionalised()
    s = nd.scales
    assert nd.layers[-1].is_vacuum                     # annotations survive
    assert nd.interfaces[-1].radius == pytest.approx(1.2 * 6.368e6 / s.length)
    surf = nd.surface("surface")
    assert surf.reference_radius == pytest.approx(1.0)
    assert float(surf.height(0.0, 0.0)) == pytest.approx(3.0e3 / s.length)
    # further surgery preserves scales
    assert nd.annotate(0, name="core").scales == nd.scales


def test_surgery_preserves_scales(prem):
    assert prem.with_buffer(ratio=0.2).scales.is_si
    nd = prem.nondimensionalised()
    assert nd.truncated(0.9).scales == nd.scales


def test_unannotated_fields_are_refused_by_name(prem):
    sk = prem.skeleton
    rogue = RadialField(sk, [polynomial_layer([1.0], sk.interval(i))
                             for i in range(sk.nlayers)], name="rogue")
    body = prem.annotate(0)  # a cheap copy
    body.add_field("rogue", rogue)
    with pytest.raises(ValueError, match="rogue.*no.*dimensions|dimensions"):
        body.nondimensionalised()


def test_dimensionless_is_the_opt_out(prem):
    sk = prem.skeleton
    fine = RadialField(sk, [polynomial_layer([1.0], sk.interval(i))
                            for i in range(sk.nlayers)], name="ok",
                       dimensions=Dimensions.DIMENSIONLESS)
    body = prem.annotate(0)
    body.add_field("ok", fine)
    nd = body.nondimensionalised()
    assert float(nd["ok"].evaluate(0.5)) == pytest.approx(1.0)


def test_nondimensionalised_requires_si(nd):
    with pytest.raises(ValueError, match="converts from SI"):
        nd.nondimensionalised()


def test_rescaled_is_identity_on_same_scales(nd):
    assert nd.rescaled(nd.scales) is nd


def test_nondimensionalised_length_defaults_to_the_solid_surface(prem):
    """A buffer must not become the length scale."""
    body = prem.with_buffer(ratio=0.2)
    nd = body.nondimensionalised()
    assert nd.scales.length == pytest.approx(6.368e6)   # not 1.2x
    assert nd.skeleton.boundaries[-1] == pytest.approx(1.2)


# ------------------------------------------------------- physics oracles

def test_gravity_redimensionalises_exactly(prem, nd):
    """The existing solver, untouched, on the nd body with G from the
    scales: the surface gravity must come back to the SI value."""
    g_si = float(gravity(prem, 6.368e6))
    g_nd = float(gravity(nd, 1.0, G=nd.scales.gravitational_constant))
    assert g_nd * nd.scales.gravity == pytest.approx(g_si, rel=1e-14)
    assert g_si == pytest.approx(9.83, abs=0.01)


def test_F_and_J_are_invariant_under_rescaling(prem):
    """Both are dimensionless, so the nd mapping must reproduce them."""
    body = (prem.name_interface(-1, "surface")
            .with_surface("surface", AnalyticTopography(
                lambda t, p: 3.0e3 * np.cos(t))))
    nd = body.nondimensionalised()
    m_si = body.mapping(rule=layer_linear())
    m_nd = nd.mapping(rule=layer_linear())

    rng = np.random.default_rng(9)
    v = rng.normal(size=(150, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    X_nd = v * rng.uniform(0.05, 0.999, size=(150, 1))
    X_si = X_nd * nd.scales.length

    assert np.allclose(m_nd.deformation_gradient(X_nd),
                       m_si.deformation_gradient(X_si), atol=1e-13)
    assert np.allclose(m_nd.jacobian(X_nd), m_si.jacobian(X_si), atol=1e-13)


def test_unit_strings_read_as_udunits_and_collapse_off_si():
    from planetmodel.model.units import unit_string
    assert Dimensions.DENSITY.unit_string() == "kg m-3"
    assert Dimensions.MODULUS.unit_string(si=True) == "kg m-1 s-2"
    assert Dimensions.DIMENSIONLESS.unit_string() == "1"
    assert Dimensions.VELOCITY.unit_string(si=False) == "1"
    assert unit_string(None, si=True) == "unknown"
    assert unit_string(Dimensions.TIME, si=True) == "s"
