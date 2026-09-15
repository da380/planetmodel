"""Dimensions and Scales: the algebra, the factors, the unit strings.

The oracles are physical identities: G becomes one under geophysical
scales because the time scale is chosen to make it so, and a factor of
a product is the product of the factors.
"""
import math

import pytest

from planetmodel import units
from planetmodel.units import (DIMENSIONLESS, EARTH_MEAN_DENSITY, G_SI,
                               GRAVITATIONAL_CONSTANT, GRAVITY, LENGTH, MASS,
                               MODULUS, TIME, VELOCITY, VISCOSITY, Dimensions,
                               Scales, unit_string)

# ------------------------------------------------------------- Dimensions


def test_dimension_algebra_is_the_physics():
    assert VELOCITY == LENGTH / TIME
    assert MODULUS == units.DENSITY * VELOCITY ** 2
    assert GRAVITY == LENGTH / TIME ** 2
    assert GRAVITATIONAL_CONSTANT == LENGTH ** 3 / (MASS * TIME ** 2)
    assert VISCOSITY == MODULUS * TIME
    assert DIMENSIONLESS.is_dimensionless
    assert not units.DENSITY.is_dimensionless
    assert (MASS / MASS).is_dimensionless


def test_dimensions_default_to_dimensionless_and_are_keyword_only():
    assert Dimensions() == DIMENSIONLESS
    assert Dimensions(length=2) == LENGTH ** 2
    with pytest.raises(TypeError):
        Dimensions(1, 0, 0)


def test_dimensions_are_frozen_and_integer():
    with pytest.raises(TypeError, match="integer"):
        Dimensions(mass=0.5)
    with pytest.raises(Exception):
        units.DENSITY.mass = 2
    with pytest.raises(TypeError):
        MASS ** 0.5


def test_dimensions_compare_by_value_and_hash():
    assert Dimensions(mass=1, length=-3) == units.DENSITY
    assert len({Dimensions(mass=1, length=-3), units.DENSITY}) == 1
    assert str(MODULUS) == "M^1 L^-1 T^-2"
    assert str(DIMENSIONLESS) == "dimensionless"


def test_unit_strings_read_as_udunits_and_collapse_off_si():
    assert units.DENSITY.unit_string() == "kg m-3"
    assert MODULUS.unit_string(si=True) == "kg m-1 s-2"
    assert DIMENSIONLESS.unit_string() == "1"
    assert VELOCITY.unit_string(si=False) == "1"
    assert TIME.unit_string() == "s"
    assert unit_string(None, si=True) == "unknown"
    assert unit_string(None, si=False) == "unknown"
    assert unit_string(TIME, si=True) == "s"
    assert unit_string(TIME, si=False) == "1"


# ----------------------------------------------------------------- Scales


def test_si_is_the_identity():
    s = Scales.SI
    assert s.is_si
    assert s == Scales()
    for d in (MODULUS, VELOCITY, units.DENSITY, GRAVITATIONAL_CONSTANT):
        assert s.factor(d) == 1.0
    assert repr(s) == "Scales.SI"


def test_geophysical_makes_G_one():
    s = Scales.geophysical(6.371e6)
    assert not s.is_si
    assert s.length == 6.371e6
    assert s.mass == pytest.approx(EARTH_MEAN_DENSITY * 6.371e6 ** 3)
    assert s.time == pytest.approx(1.0 / math.sqrt(G_SI * EARTH_MEAN_DENSITY))
    assert s.factor(units.DENSITY) == pytest.approx(EARTH_MEAN_DENSITY)
    assert s.factor(GRAVITATIONAL_CONSTANT) == pytest.approx(G_SI, rel=1e-14)
    assert G_SI / s.factor(GRAVITATIONAL_CONSTANT) == pytest.approx(
        1.0, abs=1e-14)


def test_geophysical_takes_a_prescribed_density():
    s = Scales.geophysical(1.0e6, density=3000.0)
    assert s.factor(units.DENSITY) == pytest.approx(3000.0)
    assert s.factor(GRAVITATIONAL_CONSTANT) == pytest.approx(G_SI, rel=1e-14)
    with pytest.raises(TypeError):
        Scales.geophysical(1.0e6, 3000.0)


def test_factors_compose():
    s = Scales(length=2.0, mass=3.0, time=5.0)
    assert s.factor(DIMENSIONLESS) == 1.0
    assert s.factor(LENGTH) == 2.0
    assert s.factor(MASS) == 3.0
    assert s.factor(TIME) == 5.0
    assert s.factor(VELOCITY) == pytest.approx(2.0 / 5.0)
    assert s.factor(units.DENSITY * VELOCITY ** 2) == pytest.approx(
        s.factor(units.DENSITY) * s.factor(VELOCITY) ** 2)
    assert s.factor(MODULUS) == pytest.approx(3.0 / (2.0 * 25.0))
    assert s.factor(MODULUS / units.DENSITY) == pytest.approx(
        s.factor(MODULUS) / s.factor(units.DENSITY))


def test_stored_is_si_over_factor():
    s = Scales.geophysical(6.371e6)
    rho_si = 13088.5
    assert rho_si / s.factor(units.DENSITY) == pytest.approx(
        rho_si / EARTH_MEAN_DENSITY)
    assert 6.371e6 / s.factor(LENGTH) == pytest.approx(1.0)


def test_scales_validate():
    with pytest.raises(ValueError, match="positive finite"):
        Scales(length=-1.0)
    with pytest.raises(ValueError, match="positive finite"):
        Scales(time=math.inf)
    with pytest.raises(ValueError, match="positive"):
        Scales.geophysical(0.0)
    with pytest.raises(ValueError, match="positive"):
        Scales.geophysical(1.0, density=-1.0)
    with pytest.raises(TypeError):
        Scales(2.0)


def test_scales_are_frozen_floats():
    s = Scales(length=2)
    assert isinstance(s.length, float)
    with pytest.raises(Exception):
        s.length = 3.0
