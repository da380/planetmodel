"""Character and Symmetry: refusals, shapes, equality, the product rule."""
import pytest

from planetmodel.character import (DENSITY, ELASTIC, SCALAR, STRESS, VECTOR,
                                   Character, Symmetry)


def test_the_standard_characters():
    assert (SCALAR.rank, SCALAR.weight) == (0, 0)
    assert (DENSITY.rank, DENSITY.weight) == (0, 1)
    assert (VECTOR.rank, VECTOR.weight) == (1, 0)
    assert (STRESS.rank, STRESS.weight) == (2, 1)
    assert (ELASTIC.rank, ELASTIC.weight) == (4, 1)
    for c in (SCALAR, DENSITY, VECTOR, STRESS, ELASTIC):
        assert c.voigt


def test_only_scalar_weight_zero_is_invariant():
    """Invariant means it merely composes with the mapping."""
    assert SCALAR.is_invariant
    for c in (DENSITY, VECTOR, STRESS, ELASTIC):
        assert not c.is_invariant


def test_component_and_voigt_shapes():
    assert SCALAR.component_shape == ()
    assert VECTOR.component_shape == (3,)
    assert STRESS.component_shape == (3, 3)
    assert ELASTIC.component_shape == (3,) * 4
    assert STRESS.voigt_shape == (6,)
    assert ELASTIC.voigt_shape == (6, 6)
    assert SCALAR.voigt_shape is None
    assert VECTOR.voigt_shape is None


def test_no_voigt_form_is_a_different_character():
    first = Character(4, 1, voigt=False)
    assert first != ELASTIC
    assert first.voigt_shape is None
    assert first.component_shape == (3,) * 4
    assert "no Voigt form" in str(first)


def test_characters_compare_by_value_and_hash():
    assert Character(0, 1) == DENSITY
    assert len({Character(0, 1), DENSITY}) == 1
    assert Character(0, 1) != Character(0, 0)


def test_bad_characters_are_rejected():
    with pytest.raises(ValueError, match="rank"):
        Character(-1, 0)
    with pytest.raises(ValueError, match="weight"):
        Character(0, 2)
    with pytest.raises(ValueError, match="weight"):
        Character(0, -1)


def test_characters_are_frozen():
    with pytest.raises(Exception):
        DENSITY.rank = 3


def test_voigt_is_keyword_only():
    with pytest.raises(TypeError):
        Character(4, 1, False)


# ---------------------------------------------------------- product rule

def test_rank_zero_products_add_weights():
    assert SCALAR * SCALAR == SCALAR
    assert SCALAR * DENSITY == DENSITY
    assert DENSITY * SCALAR == DENSITY
    assert (SCALAR * DENSITY).voigt


def test_two_weight_one_fields_have_no_product():
    with pytest.raises(ValueError, match="weight"):
        DENSITY * DENSITY


def test_positive_rank_has_no_product():
    for c in (VECTOR, STRESS, ELASTIC):
        with pytest.raises(ValueError, match="rank"):
            c * SCALAR
        with pytest.raises(ValueError, match="rank"):
            SCALAR * c


def test_product_with_a_non_character_is_not_implemented():
    with pytest.raises(TypeError):
        SCALAR * 2


# -------------------------------------------------------------- Symmetry

def test_symmetry_counts():
    assert Symmetry.ISOTROPIC.n_independent == 2
    assert Symmetry.VTI.n_independent == 5
    assert set(Symmetry) == {Symmetry.ISOTROPIC, Symmetry.VTI}
    assert str(Symmetry.VTI) == "vti (5 moduli)"
