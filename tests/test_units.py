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


def test_temperature_is_the_fourth_base_dimension():
    from planetmodel.units import BASE, TEMPERATURE
    assert BASE == ("mass", "length", "time", "temperature")
    assert TEMPERATURE == Dimensions(temperature=1)
    assert TEMPERATURE.unit_string() == "K" and str(TEMPERATURE) == "Theta^1"
    assert (TEMPERATURE / Dimensions(time=1)).unit_string() == "s-1 K"
    assert not TEMPERATURE.is_dimensionless
    s = Scales(length=2.0, temperature=1500.0)
    assert s.factor(TEMPERATURE) == 1500.0 and not s.is_si
    assert "temperature=1500" in repr(s)
    assert Scales.SI.temperature == 1.0 and Scales() == Scales.SI
    g = Scales.geophysical(6.371e6)
    assert g.temperature == 1.0
    assert Scales.geophysical(6.371e6, temperature=1600.0).temperature == 1600.0
    with pytest.raises(ValueError, match="temperature"):
        Scales(temperature=0.0)


def test_a_temperature_field_converts():
    from planetmodel import LayeredIsotropicElastic, RadialField
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    hot = RadialField((0.5, 1.0), lambda r: 3000.0 - 2000.0 * r)
    m = m.with_field(1, "temperature", hot)
    c = m.converted(Scales(length=2.0, temperature=1000.0))
    want = (3000.0 - 2000.0 * 0.8) / 1000.0
    assert c.layer(1)["temperature"](0.4) == pytest.approx(want)
    back = c.converted(Scales.SI)
    assert back.layer(1)["temperature"](0.75) == pytest.approx(1500.0)


def test_numpy_scalars_are_numbers():
    import numpy as np

    assert Scales(length=np.int64(2), mass=np.float32(3.0)).length == 2.0
    assert Dimensions(mass=np.int64(1)) == MASS
    assert MASS ** np.int64(2) == Dimensions(mass=2)
    with pytest.raises(ValueError):
        Scales(length=True)


# ------------------------------------------------------------ unit strings

#: Every spelling of a value's unit in the EMC catalogue that the library can
#: hold, with its vertical axes' "kilometers" and "kilometer ", and what each
#: means: the dimensions and the SI size of one unit.
EMC_SPELLINGS = {
    "km.s-1": (VELOCITY, 1e3), "km/s": (VELOCITY, 1e3),
    "kilometers/second": (VELOCITY, 1e3), "kilometer sec^-1": (VELOCITY, 1e3),
    "kilometer per second": (VELOCITY, 1e3), "kmps": (VELOCITY, 1e3),
    "km s-1": (VELOCITY, 1e3),
    "%": (DIMENSIONLESS, 0.01), "percent": (DIMENSIONLESS, 0.01),
    "Percent": (DIMENSIONLESS, 0.01),
    "km": (LENGTH, 1e3), "kilometer": (LENGTH, 1e3), "kilometers": (LENGTH, 1e3),
    "kilometer ": (LENGTH, 1e3), "m": (LENGTH, 1.0),
    "count": (DIMENSIONLESS, 1.0), "1": (DIMENSIONLESS, 1.0),
    "dimensionless": (DIMENSIONLESS, 1.0), "ratio": (DIMENSIONLESS, 1.0),
    "kg.m-3": (units.DENSITY, 1.0), "kg/m^3": (units.DENSITY, 1.0),
    "kg/m3": (units.DENSITY, 1.0), "kilograms/meter^3": (units.DENSITY, 1.0),
    "g.cm-3": (units.DENSITY, 1e3), "g/cm^3": (units.DENSITY, 1e3),
    "gram centimeter^-3": (units.DENSITY, 1e3), "g.cc-1": (units.DENSITY, 1e3),
    "GPa": (MODULUS, 1e9),
    "degree": (DIMENSIONLESS, math.pi / 180), "degrees": (DIMENSIONLESS, math.pi / 180),
    "1/s": (units.FREQUENCY, 1.0), "s": (TIME, 1.0),
}

#: The catalogue's other spellings, each refused by name: an electric unit,
#: a logarithm, an offset, nothing at all, and a stray separator that reads
#: as a factor of -1.
EMC_REFUSED = {
    "S/m": "not a unit", "siemens.m-1": "not a unit", "ohm-m": "not a unit",
    "Ohm-m": "not a unit", "log10[S/m]": "logarithm", "Log10(Ohm-m)": "logarithm",
    "degrees C": "offset", "C": "offset", "degC": "offset", "": "empty",
    "km.s.-1": "factor",
}


@pytest.mark.parametrize("text", sorted(EMC_SPELLINGS))
def test_every_emc_spelling_reads(text):
    dims, factor = units.parse_unit(text)
    want_dims, want_factor = EMC_SPELLINGS[text]
    assert dims == want_dims
    assert factor == pytest.approx(want_factor, rel=1e-15)


@pytest.mark.parametrize("text", sorted(EMC_REFUSED))
def test_every_other_emc_spelling_is_refused_by_name(text):
    with pytest.raises(ValueError, match=EMC_REFUSED[text]):
        units.parse_unit(text)


@pytest.mark.parametrize("text", [
    "deg K", "degrees K", "degrees Kelvin", "degree kelvin", "Degrees K", "degK",
    "degree_K", "degrees_kelvin", "deg.K", "°K", "kelvin", "K"])
def test_a_degree_of_temperature_is_the_kelvin(text):
    assert units.parse_unit(text) == (units.TEMPERATURE, 1.0)


def test_a_degree_is_the_angle_unless_a_kelvin_follows_it():
    # inside a product the degree of temperature is still the kelvin
    conductivity = MASS * LENGTH / TIME ** 3 / units.TEMPERATURE
    for text in ("W m-1 degK-1", "W/m/deg K", "W.m-1.degrees_K-1"):
        assert units.parse_unit(text) == (conductivity, 1.0)
    # and the angle is the angle wherever no kelvin follows
    for text in ("deg", "degrees", "degree"):
        dims, factor = units.parse_unit(text)
        assert dims == DIMENSIONLESS and factor == pytest.approx(math.pi / 180)
    dims, factor = units.parse_unit("deg/Myr")
    assert dims == units.FREQUENCY
    assert factor == pytest.approx(math.pi / 180 / 31557600e6, rel=1e-15)
    dims, factor = units.parse_unit("K/deg")
    assert dims == units.TEMPERATURE and factor == pytest.approx(180 / math.pi)
    # an offset is still refused, alone or after a degree
    for text in ("deg C", "degrees Celsius", "deg F"):
        with pytest.raises(ValueError, match="offset"):
            units.parse_unit(text)


def test_a_mislabel_reads_as_written():
    # g.cm-1 labels densities in four EMC files: the parser reads what is
    # written, and the data set's entry corrects it.
    assert units.parse_unit("g.cm-1") == (MASS / LENGTH, pytest.approx(0.1))


def test_the_library_reads_its_own_unit_strings():
    scales = [Scales.SI, Scales.geophysical(6.371e6, temperature=1600.0)]
    for dims in (DIMENSIONLESS, LENGTH, TIME, VELOCITY, units.DENSITY, MODULUS,
                 VISCOSITY, GRAVITY, GRAVITATIONAL_CONSTANT, units.TEMPERATURE,
                 units.TEMPERATURE / TIME):
        assert units.parse_unit(dims.unit_string()) == (dims, 1.0)
        for s in scales:
            got_dims, got = units.parse_unit(units.scaled_unit(dims, s))
            assert got_dims == dims
            assert got == pytest.approx(s.factor(dims), rel=1e-9)


@pytest.mark.parametrize("text, dims, factor", [
    ("mm", LENGTH, 1e-3), ("µm", LENGTH, 1e-6), ("μm", LENGTH, 1e-6),
    ("MPa", MODULUS, 1e6), ("hPa", MODULUS, 1e2), ("ms", TIME, 1e-3),
    ("min", TIME, 60.0), ("h", TIME, 3600.0), ("Myr", TIME, 31557600e6),
    ("kyr", TIME, 31557600e3), ("Pa s", VISCOSITY, 1.0), ("Pa.s", VISCOSITY, 1.0),
    ("Pa*s", VISCOSITY, 1.0), ("kg m**-3", units.DENSITY, 1.0),
    ("m/s/s", GRAVITY, 1.0), ("kg/m s", MASS * TIME / LENGTH, 1.0),
    ("kilograms per cubic", None, None),
])
def test_prefixes_powers_and_quotients(text, dims, factor):
    if dims is None:
        with pytest.raises(ValueError, match="not a unit"):
            units.parse_unit(text)
        return
    got_dims, got = units.parse_unit(text)
    assert got_dims == dims and got == pytest.approx(factor, rel=1e-15)


def test_malformed_strings_are_refused_where_they_fail():
    for text, words in (("/s", "follows no factor"), ("km/", "ends in an operator"),
                        ("km..s", "follows no factor"), ("(km)", "cannot read"),
                        ("0 m", "factor"), ("unknown", "not a unit")):
        with pytest.raises(ValueError, match=words):
            units.parse_unit(text)
