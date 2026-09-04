"""Field algebra and lazy composite fields."""
import numpy as np
import pytest

from planetmodel import (DENSITY, PREM, SCALAR, ComposedField, Field,
                         RadialField, Skeleton, polynomial_layer)
from planetmodel.model import SumField
from planetmodel.io.deck import attach_velocity_views
from planetmodel.testing import check_field


@pytest.fixture(scope="module")
def prem():
    return PREM()


def const_field(sk, value, character=SCALAR, name=None):
    return RadialField(
        sk, [polynomial_layer([value], sk.interval(i))
             for i in range(sk.nlayers)], name=name, character=character)


# ----------------------------------------------------------------- algebra

def test_sum_adds_values_and_keeps_the_character(prem):
    s = prem.A + prem.C
    r = np.linspace(1e5, 6e6, 200)
    assert np.allclose(s.evaluate(r), prem.A.evaluate(r) + prem.C.evaluate(r))
    # A modulus is scalar-valued; the rank-4 ELASTIC character belongs to
    # the ElasticField assembled from the five, not to any one of them.
    assert s.character is prem.A.character is SCALAR


def test_difference_and_negation(prem):
    d = prem.A - prem.C
    r = np.linspace(1e5, 6e6, 50)
    assert np.allclose(d.evaluate(r), prem.A.evaluate(r) - prem.C.evaluate(r))
    assert np.allclose((-prem.A).evaluate(r), -prem.A.evaluate(r))


def test_scalar_multiplication_both_ways(prem):
    r = np.linspace(1e5, 6e6, 50)
    for f in (3.0 * prem.rho, prem.rho * 3.0):
        assert np.allclose(f.evaluate(r), 3.0 * prem.rho.evaluate(r))
        assert f.character is DENSITY
    assert np.allclose((prem.rho / 2.0).evaluate(r),
                       prem.rho.evaluate(r) / 2.0)


def test_adding_different_characters_is_an_error(prem):
    """Density plus a modulus is a mistake, not a broadcast."""
    with pytest.raises(ValueError, match="different character"):
        prem.rho + prem.A


def test_field_times_field_is_refused(prem):
    """A product of fields has no character in general."""
    with pytest.raises(TypeError, match="not defined"):
        prem.A * prem.rho
    with pytest.raises(TypeError, match="not defined"):
        prem.A / prem.rho


def test_adding_across_skeletons_is_an_error():
    a = const_field(Skeleton([0.0, 1.0]), 1.0)
    b = const_field(Skeleton([0.0, 2.0]), 1.0)
    with pytest.raises(ValueError, match="different skeletons"):
        a + b


def test_composites_are_fields(prem):
    for f in (prem.A + prem.C, 2.0 * prem.rho):
        assert isinstance(f, Field)
        check_field(f)


# --------------------------------------------------------------- composed

def test_composed_field_is_pointwise_exact(prem):
    """vs = sqrt(L / rho), evaluated rather than refitted."""
    vs = ComposedField(lambda L, rho: np.sqrt(L / rho),
                       (prem.L, prem.rho), name="vs")
    r = np.linspace(4e6, 6e6, 500)
    assert np.allclose(vs.evaluate(r),
                       np.sqrt(prem.L.evaluate(r) / prem.rho.evaluate(r)),
                       rtol=1e-15)


def test_composed_field_character_is_explicit(prem):
    assert ComposedField(lambda x: x, (prem.rho,)).character is SCALAR
    assert ComposedField(lambda x: x, (prem.rho,),
                         character=DENSITY).character is DENSITY


def test_composite_integration_is_quadrature_and_crosses_layers(prem):
    """Approximate by construction, and correct across discontinuities."""
    f = 2.0 * prem.rho
    lo, hi = 1e6, 5e6
    want = 2.0 * prem.rho.integrate(lo, hi)
    assert f.integrate(lo, hi) == pytest.approx(want, rel=1e-8)


def test_empty_composites_are_rejected(prem):
    with pytest.raises(ValueError, match="at least one"):
        SumField(())
    with pytest.raises(ValueError, match="at least one"):
        ComposedField(lambda: None, ())


def test_repr_is_informative(prem):
    assert "SumField" in repr(prem.A + prem.C)
    assert "ScaledField" in repr(2.0 * prem.rho)
    assert "ComposedField" in repr(
        ComposedField(lambda x: x, (prem.rho,), name="v"))


# -------------------------------------------------- velocities as views

def test_velocity_views_reconstruct_the_tabulated_values():
    """The round trip that justifies calling moduli canonical.

    Strip PREM's tabulated velocities, rebuild them from (rho, moduli),
    and they come back to machine precision -- so nothing was lost by
    storing moduli instead.
    """
    truth_model = PREM()
    r = np.linspace(1e5, 6.3e6, 2000)
    truth = {k: truth_model[k].evaluate(r)
             for k in ("vpv", "vsv", "vph", "vsh", "eta")}

    stripped = PREM()
    for k in truth:
        stripped = stripped.without_field(k)
    attach_velocity_views(stripped)

    for k, want in truth.items():
        got = stripped[k].evaluate(r)
        assert np.allclose(got, want, rtol=1e-12, atol=1e-9), k


def test_velocity_views_are_composed_not_refitted():
    stripped = PREM()
    for k in ("vpv", "vsv", "vph", "vsh", "eta"):
        stripped = stripped.without_field(k)
    views = attach_velocity_views(stripped)
    for f in views.values():
        assert isinstance(f, ComposedField)


def test_velocity_views_handle_the_fluid_core():
    """vs = 0 in the outer core, where L = 0, without dividing badly."""
    stripped = PREM()
    for k in ("vpv", "vsv", "vph", "vsh", "eta"):
        stripped = stripped.without_field(k)
    attach_velocity_views(stripped)
    r = np.linspace(1.3e6, 3.4e6, 100)      # inside the outer core
    assert np.allclose(stripped["vsv"].evaluate(r), 0.0)
    assert np.all(np.isfinite(stripped["vpv"].evaluate(r)))


def test_velocity_views_need_the_moduli():
    sk = Skeleton([0.0, 1.0])
    from planetmodel import ReferenceBody
    body = ReferenceBody.from_fields(sk, {"rho": const_field(sk, 1.0, DENSITY, "rho")})
    with pytest.raises(ValueError, match="need rho and the moduli"):
        attach_velocity_views(body)


def test_views_do_not_clobber_tabulated_columns(prem):
    """A model that tabulates velocities keeps the better ones."""
    before = prem["vpv"]
    attach_velocity_views(prem)
    assert prem["vpv"] is before


# ------------------------------------------------- the algebra is closed

def test_the_algebra_is_closed_over_the_composites(prem):
    """(a + b) + c and 2 * (a + b): a composite is an operand like any other.

    The operators once lived only on RadialField and
    ElasticField, so any expression deeper than one operation raised
    TypeError.  Each result is checked against the arithmetic of the
    leaves, which is the only thing the algebra is allowed to mean.
    """
    a, b, c = prem.A, prem.C, prem.F
    r = np.linspace(1e5, 6e6, 128)
    va, vb, vc = (f.evaluate(r) for f in (a, b, c))

    cases = {
        "(a + b) + c": ((a + b) + c, va + vb + vc),
        "a + (b + c)": (a + (b + c), va + vb + vc),
        "2 * (a + b)": (2.0 * (a + b), 2.0 * (va + vb)),
        "(a + b) * 2": ((a + b) * 2.0, 2.0 * (va + vb)),
        "(a + b) / 2": ((a + b) / 2.0, (va + vb) / 2.0),
        "-(a - b)": (-(a - b), -(va - vb)),
        "(a - b) - c": ((a - b) - c, va - vb - vc),
    }
    for label, (field, want) in cases.items():
        got = np.asarray(field.evaluate(r), dtype=float)
        assert np.allclose(got, want, rtol=1e-14, atol=1e-14 * np.max(np.abs(want))), (
            f"{label} disagrees with the arithmetic of its leaves")
        assert isinstance(field, Field)
        assert field.character is a.character


def test_every_shipped_field_inherits_field_base(prem):
    """The composites are FieldBase too; that is what closes the algebra."""
    from planetmodel import FieldBase

    for f in (prem.rho, prem["elastic_moduli"], prem.A + prem.C, 2.0 * prem.A,
              prem.rho / 2.0):
        assert isinstance(f, FieldBase)


def test_is_radial_is_true_for_radial_fields_and_their_composites(prem):
    """Radialness survives sums, scalings and pointwise views."""
    attach_velocity_views(prem, replace=True)
    assert prem.rho.is_radial
    assert prem["elastic_moduli"].is_radial
    assert (prem.A + prem.C).is_radial
    assert (2.0 * prem.A).is_radial
    assert prem["vph"].is_radial          # a ComposedField over radial sources


def test_is_radial_is_false_for_a_field_that_will_not_say(prem):
    """Silence is not a promise: a bare FieldBase does not claim to be radial."""
    from planetmodel import FieldBase
    assert FieldBase().is_radial is False


def test_composites_pass_the_frame_through(prem):
    """A sum of scalars is frame-agnostic, but the argument still travels."""
    s = prem.A + prem.C
    r = np.linspace(1e5, 6e6, 16)
    assert np.allclose(s.evaluate(r, frame="cartesian"), s.evaluate(r))
    with pytest.raises(ValueError, match="frame"):
        s.evaluate(r, frame="galactic")


# ------------------------------------------------------ the [extra] tier

def cartesian_points(radii, theta=0.7, phi=-1.1):
    """Cartesian points at the given radii, in one generic direction."""
    r = np.asarray(radii, dtype=float)
    return np.stack([r * np.sin(theta) * np.cos(phi),
                     r * np.sin(theta) * np.sin(phi),
                     r * np.cos(theta) * np.ones_like(r)], axis=-1)


def test_evaluate_at_agrees_with_evaluate_at_the_same_radii(prem):
    """A radial field: the Cartesian points only have to find the radius."""
    r = np.linspace(1.0e6, 6.3e6, 40)
    X = cartesian_points(r)
    got = prem.rho.evaluate_at(X)
    assert got.shape == r.shape
    assert np.allclose(got, prem.rho.evaluate(r), rtol=1e-12)


def test_evaluate_at_gives_cartesian_components_by_default(prem):
    """The frame follows the coordinates.

    Cartesian points in, Cartesian components out -- which for the
    elastic tensor is the Bond rotation of what evaluate returns, and
    emphatically not the same numbers.
    """
    r = np.linspace(1.0e6, 6.3e6, 12)
    X = cartesian_points(r)
    elastic = prem["elastic_moduli"]
    cart = elastic.evaluate(r, 0.7, -1.1, frame="cartesian")
    sph = elastic.evaluate(r, 0.7, -1.1)
    # The colatitude comes back through arccos, so the comparison is at
    # the level of the tensor rather than of its smallest entry.
    scale = float(np.max(np.abs(cart)))
    assert np.allclose(elastic.evaluate_at(X), cart, rtol=1e-9,
                       atol=1e-9 * scale)
    assert np.allclose(elastic.evaluate_at(X, frame="spherical"), sph,
                       rtol=1e-9, atol=1e-9 * scale)
    # PREM is isotropic below 220 km depth, where the rotation changes
    # nothing; the anisotropic shells are what make the two differ.
    assert np.max(np.abs(cart - sph)) > 1e-3 * scale


def test_evaluate_at_broadcasts_over_the_shape_of_the_points(prem):
    X = cartesian_points(np.linspace(2.0e6, 6.0e6, 6).reshape(3, 2))
    assert prem.rho.evaluate_at(X).shape == (3, 2)


def test_restricted_agrees_inside_the_layer(prem):
    """One layer's values, with no tie to break at either end."""
    j = prem.skeleton.nlayers - 2
    lo, hi = prem.skeleton.interval(j)
    rest = prem.rho.restricted(j)
    r = np.linspace(lo, hi, 50)
    assert np.allclose(rest.evaluate(r), prem.rho.evaluate(r, layer=j))
    # The ends belong to this layer, whichever side a tie-break prefers.
    assert np.allclose(rest.evaluate(hi), prem.rho[j](hi))
    assert np.allclose(rest.evaluate(lo), prem.rho[j](lo))


def test_restricted_refuses_radii_outside_its_layer(prem):
    j = 1
    lo, hi = prem.skeleton.interval(j)
    rest = prem.rho.restricted(j)
    for bad in (lo - 1.0e5, hi + 1.0e5):
        with pytest.raises(ValueError, match="outside"):
            rest.evaluate(bad)


def test_restricted_carries_the_layer_as_its_whole_skeleton(prem):
    j = 3
    rest = prem["elastic_moduli"].restricted(j)
    assert rest.skeleton.nlayers == 1
    assert rest.skeleton.boundaries == pytest.approx(
        np.array(prem.skeleton.interval(j)))
    assert rest.character is prem["elastic_moduli"].character
    assert rest.dimensions is prem["elastic_moduli"].dimensions
    assert rest.is_radial is prem["elastic_moduli"].is_radial
    check_field(rest)


def test_restricted_is_a_field_like_any_other(prem):
    """Including the algebra: a restriction is an operand."""
    a = prem.A.restricted(2)
    c = prem.C.restricted(2)
    lo, hi = prem.skeleton.interval(2)
    r = np.linspace(lo, hi, 20)
    assert np.allclose((a + c).evaluate(r),
                       prem.A.evaluate(r, layer=2) + prem.C.evaluate(r, layer=2))
    assert isinstance(a, Field)


def test_restricted_names_only_its_own_layer(prem):
    """Its skeleton has one layer, so the source's numbering is not it."""
    rest = prem.rho.restricted(4)
    with pytest.raises(IndexError, match="out of range"):
        rest.evaluate(0.5 * sum(prem.skeleton.interval(4)), layer=4)
