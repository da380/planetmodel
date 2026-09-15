"""The registry and the executable contracts of planetmodel.testing."""
import numpy as np
import pytest

from planetmodel import PREM, read_mineos_deck, Skeleton, RadialField
from planetmodel.io.deck import read_isotropic_deck
from planetmodel.registry import KINDS, lookup, register, registered
from planetmodel.testing import (check_field, check_layer_function, check_model,
                                 check_topography)


# --------------------------------------------------------------------- registry

def test_register_and_lookup_round_trip():
    obj = object()
    register("sizing", "test_only_round_trip", obj)
    assert lookup("sizing", "test_only_round_trip") is obj
    assert "test_only_round_trip" in registered("sizing")


def test_register_works_as_a_decorator():
    @register("state_rule", "test_only_decorated")
    def rule(body):
        return None
    assert lookup("state_rule", "test_only_decorated") is rule


def test_duplicate_registration_raises():
    register("topography", "test_only_dup", object())
    with pytest.raises(KeyError, match="already registered"):
        register("topography", "test_only_dup", object())


def test_unknown_kind_lists_the_known_ones():
    with pytest.raises(KeyError, match="unknown component kind"):
        register("not_a_kind", "x", object())
    with pytest.raises(KeyError, match="unknown component kind"):
        lookup("not_a_kind", "x")


def test_missing_name_lists_what_is_registered():
    with pytest.raises(KeyError, match="no sizing named"):
        lookup("sizing", "nope")


def test_every_kind_starts_present():
    for kind in KINDS:
        assert isinstance(registered(kind), tuple)


def test_the_shipped_components_are_registered():
    """A recipe can name what planetmodel ships, without importing Python.

    Registration happens at the class and function definitions, so it
    costs no import of the model layer from registry.py -- the table
    stays a leaf, which is what keeps it free of cycles.
    """
    for name in ("zero", "analytic", "gridded"):
        assert name in registered("topography")
    assert "fluid_where_vs_zero" in registered("state_rule")

    from planetmodel.model.body import fluid_where_vs_zero
    from planetmodel.model.topography import (AnalyticTopography, GriddedTopography,
                                         ZeroTopography)
    assert lookup("state_rule", "fluid_where_vs_zero") is fluid_where_vs_zero
    assert lookup("topography", "zero") is ZeroTopography
    assert lookup("topography", "analytic") is AnalyticTopography
    assert lookup("topography", "gridded") is GriddedTopography


# ------------------------------------------------------------ layer functions

def _shipped(model_name):
    """The three shipped readers, by short name."""
    if model_name == "prem":
        return PREM()
    if model_name == "deck":
        return read_mineos_deck("examples/prem.200")
    return read_isotropic_deck("tests/data/prem.nocrust")


@pytest.mark.parametrize("model_name", ["prem", "deck", "iso"])
def test_check_layer_function_over_shipped_models(model_name):
    """Every layer function of every radial field honours the contract.

    Not every Field is layer-indexed: an ElasticField is stored as its
    moduli and has no layer functions of its own, which is why the
    protocol asks only for evaluate() and treats the rest as optional.
    """
    model = _shipped(model_name)
    checked = 0
    for name in model.field_names:
        field = model[name]
        if not hasattr(field, "__getitem__"):
            continue
        for i in range(len(field)):
            check_layer_function(field[i], model.skeleton.interval(i))
            checked += 1
    assert checked > 0


def test_check_layer_function_catches_a_lying_derivative():
    """A derivative that does not match the function is rejected."""
    class Liar:
        def __call__(self, r):
            return np.asarray(r, dtype=float) ** 2

        def derivative(self, nu=1):
            return lambda r: np.zeros_like(np.asarray(r, dtype=float))

    with pytest.raises(AssertionError, match="derivative"):
        check_layer_function(Liar(), (1.0, 2.0))


# -------------------------------------------------------------------- fields

@pytest.mark.parametrize("model_name", ["prem", "deck", "iso"])
def test_check_field_over_shipped_models(model_name):
    """Every field of every shipped model, the elastic tensor included.

    The tensor is the field the extended contract bites on: it is the
    one whose Cartesian components differ from its spherical ones, and
    check_field compares them against an independent Bond rotation.
    """
    from planetmodel.testing import check_frequency_dependent_field
    model = _shipped(model_name)
    assert "elastic_moduli" in model.field_names
    for name in model.field_names:
        field = model[name]
        if getattr(field, "kind", "static") == "frequency":
            check_frequency_dependent_field(field, omegas=(0.5, 2 * np.pi, 9.0))
        else:
            check_field(field)


@pytest.mark.parametrize("model_name", ["prem", "deck", "iso"])
def test_check_model_over_shipped_models(model_name):
    """Every shipped reader returns a model class that keeps its guarantees."""
    check_model(_shipped(model_name))


def test_check_field_catches_a_rescale_that_does_not_round_trip():
    """A field whose rescaled() changes its values is rejected by name."""
    sk = Skeleton([1.0e6, 2.0e6])

    class Drifts(RadialField):
        def rescaled(self, convert, old, new):
            out = super().rescaled(convert, old, new)
            return out * 1.5

    from planetmodel.model.units import Dimensions
    f = Drifts(sk, [lambda r: np.ones_like(np.asarray(r, float))], name="drifts",
               dimensions=Dimensions.DIMENSIONLESS)
    with pytest.raises(AssertionError, match="there and back|rescal"):
        check_field(f)


def test_check_field_rejects_a_missing_attribute():
    class NotAField:
        skeleton = Skeleton([0.0, 1.0])
        name = None

        def evaluate(self, r, theta=None, phi=None, **kw):
            return np.zeros_like(np.asarray(r, dtype=float))

    with pytest.raises(AssertionError, match="character"):
        check_field(NotAField())


def test_check_field_rejects_unbounded_extrapolation():
    """A field that happily evaluates outside its skeleton fails."""
    sk = Skeleton([1.0, 2.0])

    class TooKeen(RadialField):
        def evaluate(self, r, theta=None, phi=None, **kw):
            return np.asarray(r, dtype=float) * 0.0

    f = TooKeen(sk, [lambda r: np.zeros_like(np.asarray(r, float))])
    with pytest.raises(AssertionError, match="outside the skeleton"):
        check_field(f)


CHECKS = ("check_layer_function", "check_field", "check_frequency_dependent_field",
          "check_time_dependent_field", "check_law", "check_model",
          "check_topography", "check_displacement", "check_mapping",
          "check_sample")


def test_every_protocol_check_is_implemented_and_exported():
    """No stubs: every contract is executable, exported, and documented."""
    from planetmodel import testing
    assert set(CHECKS) <= set(testing.__all__)
    for name in CHECKS:
        fn = getattr(testing, name)
        assert callable(fn)
        assert fn.__doc__ and "NotImplementedError" not in fn.__doc__
        assert name in testing.__doc__, f"{name} is not listed in the module doc"


def test_check_topography_over_a_centred_shape():
    """A centred shape keeps the continuity, gradient and mean contract."""
    from planetmodel.model.topography import AnalyticTopography, CentredTopography
    shape = AnalyticTopography(lambda t, p: 100.0 + 50.0 * np.cos(t) ** 2)
    centred = CentredTopography(shape, float(shape.mean()))
    check_topography(centred)
    assert abs(float(centred.mean())) < 1e-6


def test_check_field_catches_a_frame_argument_that_is_ignored():
    """The check with teeth: a rank-4 field that ignores `frame`.

    Returning the spherical components under the Cartesian name is the
    exact failure the Bond comparison exists to catch, and it is
    invisible to every shape and finiteness check.
    """
    prem = PREM()

    class Deaf:
        skeleton = prem.skeleton
        character = prem["elastic_moduli"].character
        dimensions = prem["elastic_moduli"].dimensions
        name = "deaf"
        is_radial = True

        def evaluate(self, r, theta=None, phi=None, *, frame="spherical", **kw):
            return prem["elastic_moduli"].evaluate(r, theta, phi, **kw)

    with pytest.raises(AssertionError, match="Bond rotation"):
        check_field(Deaf())


def test_name_of_an_unregistered_object_is_none():
    from planetmodel.registry import name_of
    assert name_of("sizing", object()) is None
    with pytest.raises(KeyError, match="unknown component kind"):
        name_of("nonsense", object())


def test_check_topography_sees_a_wrong_phi_gradient():
    from planetmodel.testing import check_topography

    class Wrong:
        def __call__(self, t, p):
            return np.sin(t) ** 2 * np.cos(2 * p)

        def gradient(self, t, p):
            return 2 * np.sin(t) * np.cos(t) * np.cos(2 * p), 0.0 * p

        def mean(self):
            return 0.0

    with pytest.raises(AssertionError, match="d/dphi"):
        check_topography(Wrong())
