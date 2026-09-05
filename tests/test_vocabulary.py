"""The field names the library speaks: one table, with the characters the
dimensions imply."""
import pytest

from planetmodel import units
from planetmodel.character import DENSITY, ELASTIC, SCALAR
from planetmodel.vocabulary import CONSTANTS, VOCABULARY, Constant, FieldSpec, spec

EXPECTED_NAMES = {
    "rho", "vp", "vs", "vpv", "vsv", "vph", "vsh", "eta", "qkappa", "qmu",
    "kappa", "mu", "A", "C", "F", "L", "N", "elastic_moduli", "viscosity",
}


def test_the_shipped_names():
    assert set(VOCABULARY) == EXPECTED_NAMES
    for name, entry in VOCABULARY.items():
        assert isinstance(entry, FieldSpec), name
        assert entry.meaning, name
        assert entry.dimensions is not None, name
        assert spec(name) is entry


def test_characters_and_dimensions_are_what_the_physics_says():
    assert spec("rho").character is DENSITY
    assert spec("rho").dimensions == units.DENSITY
    assert spec("vpv").character is SCALAR
    assert spec("vpv").dimensions == units.VELOCITY
    assert spec("qmu").dimensions == units.DIMENSIONLESS
    assert spec("eta").dimensions == units.DIMENSIONLESS
    assert spec("elastic_moduli").character is ELASTIC
    assert spec("elastic_moduli").dimensions == units.MODULUS
    assert spec("A").dimensions == units.MODULUS
    assert spec("kappa").dimensions == units.MODULUS
    assert spec("viscosity").dimensions == units.VISCOSITY
    assert spec("viscosity").character is SCALAR


def test_every_character_is_the_one_its_dimensions_imply():
    """Rank 0 everywhere but the tensor; weight 1 exactly for a density
    or a modulus."""
    for name, entry in VOCABULARY.items():
        c, d = entry.character, entry.dimensions
        if name == "elastic_moduli":
            assert c.rank == 4, name
        else:
            assert c.rank == 0, name
        weighted = d in (units.DENSITY, units.MODULUS)
        assert (c.weight == 1) == weighted, name
        assert c.voigt, name


def test_a_modulus_is_a_density_times_a_squared_velocity():
    """The product rule agrees with the table: rho * vs**2 has the
    character of a modulus."""
    rho, vs = spec("rho"), spec("vs")
    assert rho.character * vs.character * vs.character == spec("mu").character
    assert rho.dimensions * vs.dimensions ** 2 == spec("mu").dimensions


def test_specs_compare_by_value_and_meaning_is_keyword_only():
    assert FieldSpec(SCALAR, units.VELOCITY) == FieldSpec(SCALAR, units.VELOCITY)
    assert FieldSpec(SCALAR, None).dimensions is None
    with pytest.raises(TypeError):
        FieldSpec(SCALAR, units.VELOCITY, "a speed")
    with pytest.raises(Exception):
        spec("rho").meaning = "changed"


def test_a_name_outside_the_vocabulary_is_refused_by_name():
    with pytest.raises(KeyError, match="porosity"):
        spec("porosity")
    with pytest.raises(KeyError, match="rho"):
        spec("porosity")


def test_the_constants():
    assert set(CONSTANTS) == {"G"}
    G = CONSTANTS["G"]
    assert isinstance(G, Constant)
    assert G.value_si == units.G_SI
    assert G.dimensions == units.GRAVITATIONAL_CONSTANT
    assert G.meaning
    with pytest.raises(TypeError):
        Constant(1.0, units.DIMENSIONLESS, "one")


def test_G_is_one_in_geophysical_units():
    s = units.Scales.geophysical(6.371e6)
    G = CONSTANTS["G"]
    assert G.value_si / s.factor(G.dimensions) == pytest.approx(1.0, abs=1e-14)
