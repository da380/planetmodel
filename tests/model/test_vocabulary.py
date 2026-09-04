"""The field names the library itself speaks: one table, consulted everywhere."""
import pytest

from planetmodel import PREM
from planetmodel.io.deck import MINEOS_COLUMNS, ISOTROPIC_COLUMNS
from planetmodel.model.character import DENSITY, ELASTIC, SCALAR
from planetmodel.model.units import Dimensions
from planetmodel.model import vocabulary as voc


def test_every_deck_column_and_attached_field_is_in_the_vocabulary():
    for name in (*MINEOS_COLUMNS, *ISOTROPIC_COLUMNS, "A", "C", "F", "L", "N",
                 "kappa", "mu", "elastic_moduli", "viscoelastic_moduli"):
        assert name in voc.VOCABULARY
        assert voc.describe(name)


def test_characters_and_dimensions_are_what_the_physics_says():
    assert voc.character_of("rho") is DENSITY
    assert voc.dimensions_of("rho") == Dimensions.DENSITY
    assert voc.character_of("vpv") is SCALAR
    assert voc.dimensions_of("vpv") == Dimensions.VELOCITY
    assert voc.dimensions_of("qmu") == Dimensions.DIMENSIONLESS
    assert voc.character_of("elastic_moduli") is ELASTIC
    assert voc.dimensions_of("A") == Dimensions.MODULUS
    assert voc.dimensions_of("viscosity") == Dimensions.VISCOSITY


def test_prem_files_its_fields_under_the_vocabulary():
    prem = PREM()
    for name in prem.field_names:
        if name == "viscoelastic_moduli":
            continue
        assert prem[name].character == voc.character_of(name), name
        assert prem[name].dimensions == voc.dimensions_of(name), name


def test_a_name_outside_the_vocabulary_is_refused_by_name():
    with pytest.raises(KeyError, match="porosity"):
        voc.dimensions_of("porosity")
