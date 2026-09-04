"""Character and Symmetry: the transformation vocabulary."""
import pytest

from planetmodel import (DENSITY, ELASTIC, SCALAR, STRESS, VECTOR, Character,
                    Symmetry)


def test_the_standard_characters():
    assert (SCALAR.rank, SCALAR.weight) == (0, 0)
    assert (DENSITY.rank, DENSITY.weight) == (0, 1)
    assert (VECTOR.rank, VECTOR.weight) == (1, 0)
    assert (STRESS.rank, STRESS.weight) == (2, 1)
    assert (ELASTIC.rank, ELASTIC.weight) == (4, 1)


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


def test_characters_compare_by_value_and_hash():
    assert Character(0, 1) == DENSITY
    assert len({Character(0, 1), DENSITY}) == 1


def test_bad_characters_are_rejected():
    with pytest.raises(ValueError, match="rank"):
        Character(-1, 0)
    with pytest.raises(ValueError, match="weight"):
        Character(0, 2)


def test_characters_are_frozen():
    with pytest.raises(Exception):
        DENSITY.rank = 3


def test_symmetry_counts():
    assert Symmetry.ISOTROPIC.n_independent == 2
    assert Symmetry.VTI.n_independent == 5
    assert Symmetry.ORTHOTROPIC.n_independent == 9
    assert Symmetry.GENERAL.n_independent == 21


def test_promote_takes_the_wider_class():
    S = Symmetry
    assert S.ISOTROPIC.promote(S.VTI) is S.VTI
    assert S.VTI.promote(S.ISOTROPIC) is S.VTI
    assert S.VTI.promote(S.GENERAL) is S.GENERAL
    assert S.ISOTROPIC.promote(S.ISOTROPIC) is S.ISOTROPIC


def test_promote_is_commutative_and_idempotent():
    for a in Symmetry:
        assert a.promote(a) is a
        for b in Symmetry:
            assert a.promote(b) is b.promote(a)


def test_promote_rejects_non_symmetries():
    with pytest.raises(TypeError, match="Symmetry"):
        Symmetry.VTI.promote("general")
