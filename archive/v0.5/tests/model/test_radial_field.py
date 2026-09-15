"""RadialField.evaluate: layer resolution, sides, broadcasting."""
import numpy as np
import pytest

from planetmodel import (DENSITY, PREM, SCALAR, Field, RadialField,
                    Skeleton, polynomial_layer)


@pytest.fixture(scope="module")
def prem():
    return PREM()


def test_satisfies_the_field_protocol(prem):
    """A RadialField is structurally a Field."""
    assert isinstance(prem.rho, RadialField)
    assert isinstance(prem.rho, Field)


def test_interior_agrees_with_layer_indexed_access(prem):
    """Away from boundaries, evaluate is the layer function."""
    b = prem.skeleton.boundaries
    r = np.linspace(b[0], b[-1], 20000)
    r = r[~np.isin(r, b)]
    want = [prem.rho[prem.skeleton.locate(x).layers[-1]](x) for x in r]
    assert np.allclose(prem.rho.evaluate(r), want)


def test_side_picks_the_right_layer_at_every_boundary(prem):
    """upper is the layer above a boundary, lower the one below.

    Regression: the searchsorted sides were once inverted, which made
    evaluate() return the fluid outer core's vsv on the mantle side of
    the CMB.  Both one-sided values are meaningful, so getting them the
    wrong way round is silent unless it is checked against the layers.
    """
    sk = prem.skeleton
    for j in range(1, sk.nlayers):
        r = float(sk.boundaries[j])
        assert prem.vsv.evaluate(r) == prem.vsv[j](r)
        assert prem.vsv.evaluate(r, side="lower") == prem.vsv[j - 1](r)


def test_the_cmb_has_a_fluid_side_and_a_solid_side(prem):
    """The physics the side convention has to reproduce."""
    cmb = 3480e3
    assert prem.vsv.evaluate(cmb) > 7000.0            # mantle
    assert prem.vsv.evaluate(cmb, side="lower") == 0.0  # outer core, fluid


def test_explicit_layer_overrides_side(prem):
    cmb = 3480e3
    assert prem.vsv.evaluate(cmb, layer=1) == prem.vsv[1](cmb)
    assert prem.vsv.evaluate(cmb, layer=2) == prem.vsv[2](cmb)


def test_domain_endpoints_resolve_on_both_sides(prem):
    b = prem.skeleton.boundaries
    for r in (float(b[0]), float(b[-1])):
        for side in ("upper", "lower"):
            assert np.isfinite(prem.rho.evaluate(r, side=side))


def test_shapes_and_broadcasting(prem):
    assert np.shape(prem.rho.evaluate(1e6)) == ()
    assert prem.rho.evaluate(np.zeros((3, 4)) + 1e6).shape == (3, 4)
    got = prem.rho.evaluate(np.full((2, 1), 1e6), theta=np.zeros(3))
    assert got.shape == (2, 3)


def test_radii_outside_the_skeleton_raise(prem):
    with pytest.raises(ValueError, match="outside"):
        prem.rho.evaluate(7e6)
    with pytest.raises(ValueError, match="outside"):
        prem.rho.evaluate(-1.0)


def test_bad_side_raises(prem):
    with pytest.raises(ValueError, match="side"):
        prem.rho.evaluate(1e6, side="sideways")


def test_character_defaults_to_scalar():
    """An unlabelled field is invariant: rank 0, weight 0."""
    sk = Skeleton([0.0, 1.0])
    f = RadialField(sk, [lambda r: np.zeros_like(np.asarray(r, float))])
    assert f.character is SCALAR
    assert f.character.is_invariant


def test_derivative_carries_the_character_through():
    """d/dr does not change how a field transforms."""
    sk = Skeleton([1.0, 2.0])
    f = RadialField(sk, [polynomial_layer([0.0, 0.0, 1.0], (1.0, 2.0))],
                    character=DENSITY)
    assert f.derivative().character is DENSITY


def test_both_frames_agree_and_a_third_is_refused(prem):
    """A scalar has the same components in every frame -- but `frame` is
    still an argument, not a decoration.

    The plan lets an implementation support only its natural frame; it
    does not let one accept a frame it has never heard of and return
    something anyway.
    """
    r = np.linspace(1e5, 6e6, 32)
    want = prem.rho.evaluate(r)
    assert np.array_equal(prem.rho.evaluate(r, frame="spherical"), want)
    assert np.array_equal(prem.rho.evaluate(r, frame="cartesian"), want)
    with pytest.raises(ValueError, match="unknown frame"):
        prem.rho.evaluate(r, frame="geographic")


def test_spherical_is_the_default_frame(prem):
    """The frame the coordinates imply: evaluate speaks (r, theta, phi)."""
    import inspect
    from planetmodel.model.fields.base import Field as FieldProtocol
    for fn in (RadialField.evaluate, FieldProtocol.evaluate,
               type(prem["elastic_moduli"]).evaluate):
        assert inspect.signature(fn).parameters["frame"].default == "spherical"


def test_a_radial_field_says_so(prem):
    assert prem.rho.is_radial is True


def test_layer_names_a_side_and_does_not_extrapolate():
    """`layer=` picks the side at a boundary; a radius outside the layer is
    refused rather than answered by that layer's function continued."""
    import pytest
    from planetmodel import RadialField, Skeleton
    sk = Skeleton([0.0, 1.0, 2.0])
    f = RadialField(sk, [lambda r: 1.0 + 0 * r, lambda r: 2.0 + 0 * r])
    assert f.evaluate(1.0, layer=0) == 1.0 and f.evaluate(1.0, layer=1) == 2.0
    with pytest.raises(ValueError, match="not in layer 0"):
        f.evaluate(1.5, layer=0)
    with pytest.raises(ValueError, match="not in layer 1"):
        f.evaluate(np.array([1.5, 0.5]), layer=1)
