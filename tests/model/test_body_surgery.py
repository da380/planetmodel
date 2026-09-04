"""Skeleton and ReferenceBody surgery: coarsening, cutting, growing.

The invariant behind all of it is that surgery is copy-on-write and that
fields keep their meaning.  Coarsening the geometry must not coarsen the
model: a merged layer still samples the fine structure inside it, which
is the whole difference between choosing a mesh and changing a planet.
"""
import warnings

import numpy as np
import pytest

from planetmodel import PREM, Skeleton
from planetmodel.io.deck import read_isotropic_deck
from planetmodel.model.body import Interface, Layer, ReferenceBody, fluid_where_vs_zero
from planetmodel.model.skeleton import CoarseningMap
from planetmodel.testing import check_field

ISO_DECK = "tests/data/prem.nocrust"


def _const(sk):
    from planetmodel import RadialField
    return RadialField(sk, [lambda r: 1.0 + 0.0 * r] * sk.nlayers, name="rho")


@pytest.fixture(scope="module")
def deck():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        return read_isotropic_deck(ISO_DECK).classify_states()


@pytest.fixture(scope="module")
def prem():
    return PREM()


# ------------------------------------------------------------ Skeleton

def test_coarsen_keeps_and_drops_are_complementary():
    sk = Skeleton(np.arange(6.0))            # 5 layers, 4 interior boundaries
    a, _ = sk.coarsen(drop=[1, 2])
    b, _ = sk.coarsen(keep=[0, 3])
    assert np.array_equal(a.boundaries, b.boundaries)


def test_coarsen_map_records_the_merge():
    sk = Skeleton(np.arange(6.0))
    coarse, cmap = sk.coarsen(drop=[1, 2])
    assert isinstance(cmap, CoarseningMap)
    assert coarse.nlayers == 3
    assert cmap.layers == ((0,), (1, 2, 3), (4,))
    assert cmap.kept_interfaces == (0, 3)
    assert cmap.dropped_interfaces == (1, 2)


def test_coarsen_map_finds_the_fine_layer():
    sk = Skeleton(np.arange(6.0))
    _, cmap = sk.coarsen(drop=[1, 2])
    assert cmap.fine_layer(2.5) == 2


def test_coarsen_rejects_ambiguous_and_out_of_range_requests():
    sk = Skeleton(np.arange(6.0))
    with pytest.raises(ValueError, match="exactly one"):
        sk.coarsen()
    with pytest.raises(ValueError, match="exactly one"):
        sk.coarsen(keep=[0], drop=[1])
    with pytest.raises(IndexError, match="interior boundary index"):
        sk.coarsen(drop=[4])


def test_negative_interior_indices_count_from_the_outside():
    sk = Skeleton(np.arange(6.0))
    a, _ = sk.coarsen(drop=[-1])
    b, _ = sk.coarsen(drop=[3])
    assert np.array_equal(a.boundaries, b.boundaries)


def test_refined_extended_truncated():
    sk = Skeleton([0.0, 1.0, 2.0])
    assert np.array_equal(sk.refined([0.5]).boundaries, [0.0, 0.5, 1.0, 2.0])
    assert np.array_equal(sk.extended([3.0]).boundaries, [0.0, 1.0, 2.0, 3.0])
    assert np.array_equal(sk.truncated(1.5).boundaries, [0.0, 1.0, 1.5])
    assert np.array_equal(sk.truncated(1.0).boundaries, [0.0, 1.0])


def test_surgery_rejects_radii_in_the_wrong_place():
    sk = Skeleton([0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="cannot insert"):
        sk.refined([3.0])
    with pytest.raises(ValueError, match="cannot append"):
        sk.extended([0.5])
    with pytest.raises(ValueError, match="already a boundary"):
        sk.refined([1.0])
    with pytest.raises(ValueError, match="beyond the outer boundary"):
        sk.truncated(3.0)
    with pytest.raises(ValueError, match="at or below the centre"):
        sk.truncated(0.0)


# --------------------------------------------------------- annotations

def test_default_annotations(prem):
    assert len(prem.layers) == prem.skeleton.nlayers
    assert len(prem.interfaces) == prem.skeleton.nlayers
    fluid = [lay.index for lay in prem.layers if lay.state == "fluid"]
    assert fluid[0] == 1 and fluid[1:] in ([], [prem.skeleton.nlayers - 1])
    assert prem.interfaces[-1].between[1] == -1     # nothing above the surface
    assert prem.interfaces[-1].radius == pytest.approx(prem.skeleton.boundaries[-1])


def test_classify_states_finds_exactly_the_outer_core(deck):
    """The default rule finds the outer core and nothing else."""
    fluid = [lay.index for lay in deck.layers if lay.state == "fluid"]
    assert fluid == [1]
    lo, hi = deck.skeleton.interval(1)
    assert (lo, hi) == pytest.approx((1221500.0, 3480000.0))


def test_classify_states_works_for_anisotropic_naming(prem):
    """PREM tabulates vsv rather than vs; the rule reads either.

    PREM by default carries its 3 km ocean, which is genuinely fluid, so
    the rule should find two fluid layers and not one: the outer core
    and the ocean.  Dropping the ocean leaves only the core.
    """
    fluid = [lay.index for lay in prem.classify_states().layers
             if lay.state == "fluid"]
    assert fluid == [1, 12]
    assert prem.skeleton.interval(12) == pytest.approx((6368e3, 6371e3))

    dry = [lay.index for lay in PREM(ocean=False).classify_states().layers
           if lay.state == "fluid"]
    assert dry == [1]


def test_overrides_beat_the_rule(deck):
    body = deck.classify_states(overrides={0: "fluid"})
    assert body.layer(0).state == "fluid"
    assert body.layer(1).state == "fluid"      # still found by the rule


def test_a_custom_rule_is_just_a_callable(deck):
    body = deck.classify_states(rule=lambda b, i: "fluid")
    assert all(lay.state == "fluid" for lay in body.layers)


def test_state_rule_has_no_opinion_without_a_shear_field():
    """Silence is not evidence: a layer with no vs is left as it was."""
    sk = Skeleton([0.0, 1.0])
    body = ReferenceBody.from_fields(sk, {})
    assert fluid_where_vs_zero(body, 0) is None
    assert body.classify_states().layer(0).state == "solid"
    assert body.annotate(0, state="fluid").classify_states().layer(0).state \
        == "fluid"


def test_rheology_is_no_longer_an_annotation(prem):
    """Stage three: what a layer is, its fields say (plan P11)."""
    with pytest.raises(TypeError):
        prem.annotate(0, rheology="viscoelastic")
    assert not hasattr(prem.layer(0), "rheology")


def test_layers_and_interfaces_resolve_by_name(prem):
    body = prem.name_interface(1, "cmb").annotate(2, name="lower_mantle")
    assert body.interface("cmb").radius == pytest.approx(3480e3)
    assert body.layer("lower_mantle").index == 2
    with pytest.raises(KeyError, match="no interface named"):
        body.interface("nope")
    with pytest.raises(KeyError, match="no layer named"):
        body.layer("nope")


def test_invalid_annotations_are_rejected():
    with pytest.raises(ValueError, match="state must be"):
        Layer(0, state="gaseous")
    with pytest.raises(ValueError, match="vacuum layer holds no fields"):
        Layer(0, interval=(0.0, 1.0), state="vacuum",
              fields={"rho": _const(Skeleton([0.0, 1.0]))})
    with pytest.raises(ValueError, match="role must be"):
        Interface(0, role="decorative")


# ------------------------------------------------------------- surgery

def test_coarsening_does_not_coarsen_the_model(deck):
    """The point of the CoarseningMap: merged layers keep fine structure."""
    coarse, _ = deck.coarsened(drop=[-1, -2, -3])
    assert coarse.skeleton.nlayers == deck.skeleton.nlayers - 3
    r = np.linspace(1e5, 6.34e6, 5000)
    for name in ("rho", "vp", "A", "L"):
        assert np.allclose(coarse[name].evaluate(r), deck[name].evaluate(r)), name


def test_coarsening_rebinds_composite_fields(deck):
    """An ElasticField is rebuilt from rebound moduli, not dropped."""
    coarse, _ = deck.coarsened(drop=[-1])
    assert "elastic_moduli" in coarse
    assert coarse["elastic_moduli"].skeleton == coarse.skeleton
    r = np.array([1e6, 5e6])
    assert np.allclose(coarse["elastic_moduli"].evaluate(r),
                       deck["elastic_moduli"].evaluate(r))


def test_truncated_cuts_geometry_and_keeps_material(deck):
    body = deck.truncated(6.3e6, name="cut")
    assert body.skeleton.boundaries[-1] == pytest.approx(6.3e6)
    assert body.interfaces[-1].name == "cut"
    r = np.linspace(1e5, 6.2e6, 500)
    assert np.allclose(body.rho.evaluate(r), deck.rho.evaluate(r))


def test_refined_inserts_a_boundary_without_changing_material(deck):
    body = deck.refined([6.0e6], names=["floor"], role="control")
    assert body.skeleton.nlayers == deck.skeleton.nlayers + 1
    assert body.interface("floor").role == "control"
    r = np.linspace(1e5, 6.3e6, 2000)
    assert np.allclose(body.rho.evaluate(r), deck.rho.evaluate(r))


def test_extended_is_empty_by_default(deck):
    """A new shell holds nothing unless told what to hold."""
    body = deck.extended([6.371e6])
    assert body.layers[-1].fields == {}
    assert body.layers[-1].fields == {}
    assert body.rho.domain == tuple(range(deck.skeleton.nlayers))


def test_extended_can_extrapolate(deck):
    body = deck.extended([6.371e6], fields="extrapolate")
    assert body.layers[-1].fields
    top = deck.skeleton.boundaries[-1]
    # The layer function continues; the piece it came from would refuse.
    assert body.rho.evaluate(top + 1e3) == pytest.approx(
        deck.rho[-1].function(top + 1e3))
    assert set(body.layers[-1].field_names) == set(deck.field_names)


def test_classify_states_leaves_an_empty_shell_alone(deck):
    """No fields is no evidence.

    extended(..., fields=None) grows a shell with nothing in it, so the
    default rule has no shear velocity to read there and says nothing.
    Classifying afterwards must leave the shell as it was declared --
    solid -- rather than reporting an ocean the model never claimed.
    """
    body = deck.extended([6.371e6], fields=None, names=["crust"])
    assert body.layers[-1].fields == {}
    assert body.layers[-1].state == "solid"
    after = body.classify_states()
    assert after.layers[-1].state == "solid"
    # ... while the layers that do have material are still classified.
    assert [lay.index for lay in after.layers if lay.state == "fluid"] == [1]


def test_overrides_reach_an_empty_shell_but_not_a_vacuum(deck):
    """An override is the user's authority, and reaches any layer that
    is not a void; a vacuum layer is left alone by rule and override
    alike, since there is nothing there to classify."""
    body = deck.extended([6.371e6], fields=None, names=["crust"])
    assert body.classify_states(overrides={-1: "fluid"}).layers[-1].state \
        == "fluid"
    assert body.annotate(-1, state="fluid").layers[-1].state == "fluid"
    buffered = body.with_buffer(ratio=0.1)
    assert buffered.classify_states(overrides={-1: "fluid"}).layers[-1].state \
        == "vacuum"


def test_extended_with_no_fields_marks_the_material_unspecified(deck):
    """The honest option for a shell the consumer will fill in."""
    body = deck.extended([6.371e6], fields=None, names=["surface"])
    assert body.layers[-1].fields == {}
    assert body.interfaces[-1].name == "surface"
    with pytest.raises(ValueError, match="not defined"):
        body.rho.evaluate(6.36e6)


def test_every_field_keeps_its_domain_after_growing(deck):
    """Nothing is zero-filled: a view has a domain, and check_field
    exercises it -- refusal outside, agreement with the pieces inside."""
    body = deck.extended([6.371e6], fields=None).with_buffer(ratio=0.2)
    n = deck.skeleton.nlayers
    for name in body.field_names:
        f = body[name]
        assert f.skeleton == body.skeleton
        assert f.domain == tuple(range(n)), name
        check_field(f)


def test_buffer_is_a_vacuum_shell(deck):
    body = deck.with_buffer(ratio=0.2)
    a = deck.skeleton.boundaries[-1]
    lay = body.layers[-1]
    assert body.skeleton.boundaries[-1] == pytest.approx(1.2 * a)
    assert lay.state == "vacuum" and lay.is_fluid
    assert lay.fields == {} and lay.is_vacuum
    with pytest.raises(ValueError, match="not defined"):
        body.rho.evaluate(1.1 * a)
    assert body.interfaces[-1].name == "buffer"


def test_buffer_by_explicit_radius(deck):
    body = deck.with_buffer(radius=8e6)
    assert body.skeleton.boundaries[-1] == pytest.approx(8e6)


def test_buffer_arguments_are_exclusive_and_checked(deck):
    with pytest.raises(ValueError, match="exactly one"):
        deck.with_buffer()
    with pytest.raises(ValueError, match="exactly one"):
        deck.with_buffer(ratio=0.2, radius=8e6)
    with pytest.raises(ValueError, match="must exceed"):
        deck.with_buffer(radius=1e6)


def test_surgery_is_copy_on_write(deck):
    """Nothing mutates the body it was called on."""
    n_layers, n_faces = len(deck.layers), len(deck.interfaces)
    deck.coarsened(drop=[0])
    deck.truncated(6e6)
    deck.refined([6e6])
    deck.extended([7e6])
    deck.with_buffer(ratio=0.2)
    deck.annotate(0, name="inner_core")
    assert len(deck.layers) == n_layers
    assert len(deck.interfaces) == n_faces
    assert deck.layer(0).name is None


def test_names_must_match_the_radii_given(deck):
    with pytest.raises(ValueError, match="names for"):
        deck.refined([6.0e6, 6.1e6], names=["only-one"])
    with pytest.raises(ValueError, match="names for"):
        deck.extended([7e6], names=["a", "b"])


def test_extended_rejects_an_unknown_fields_option(deck):
    with pytest.raises(ValueError, match='must be "extrapolate"'):
        deck.extended([7e6], fields="invent-something")


def test_coarsening_across_a_state_change_warns_unless_told(deck):
    """Merging a fluid with a solid names the merged state, or warns."""
    with pytest.warns(UserWarning, match="merges states"):
        coarse, _ = deck.coarsened(drop=[0])
    assert coarse.layer(0).state == deck.layer(1).state   # the outer part's
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        told, _ = deck.coarsened(drop=[0], state="fluid")
    assert told.layer(0).state == "fluid"
    fine = deck.refined([5.0e6])                           # the mantle in two
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        same, _ = fine.coarsened(drop=[2])                 # solid with solid
    assert same.layer(2).state == "solid"


def test_skeleton_layer_index_spans_and_contains():
    sk = Skeleton([0.0, 1.0, 3.0])
    assert sk.layer_index(-1) == 1 and sk.layer_index(0) == 0
    with pytest.raises(IndexError):
        sk.layer_index(2)
    assert sk.spans(0.0, 3.0) and not sk.spans(0.0, 1.0)
    assert sk.spans(1.0, 3.0, layer=1) and sk.spans(1.0, 3.0 + 1e-12, layer=-1)
    assert sk.contains(0.5, 2.5) and not sk.contains(-0.1, 2.5)
    assert sk.tolerance == pytest.approx(3e-9)


def test_the_constructor_takes_layers_and_from_fields_splits(deck):
    """One constructor; body-wide fields go in through from_fields."""
    rebuilt = ReferenceBody(deck.layers, meta=deck.meta, scales=deck.scales)
    assert rebuilt.skeleton == deck.skeleton
    assert rebuilt.field_names == deck.field_names
    with pytest.raises(TypeError):
        ReferenceBody(deck.skeleton, fields={"rho": deck.rho})
    split = ReferenceBody.from_fields(deck.skeleton, {"rho": deck.rho},
                                      layers=[Layer(index=i, name=f"L{i}")
                                              for i in range(deck.skeleton.nlayers)])
    assert split.field_names == ("rho",) and split.layer(1).name == "L1"
    assert split.layer(1)["rho"].skeleton.nlayers == 1
    assert split.validate() is None


def test_as_class_promotes_and_keeps_the_annotations(deck):
    from planetmodel.model.classes import ElasticModel
    b = ReferenceBody(deck.layers, meta={"name": "x"}).annotate(1, name="oc")
    m = b.as_class(ElasticModel)
    assert type(m) is ElasticModel and m.layer("oc").index == 1
    assert m.meta["name"] == "x"
    back = m.as_class(ReferenceBody, meta={"name": "y"})
    assert type(back) is ReferenceBody and back.meta == {"name": "y"}


def test_surfaces_travel_through_surgery_in_the_constructor(deck):
    """Attached surfaces are constructor state, not bolted on after."""
    from planetmodel import ZeroTopography
    b = deck.with_surface(-1, ZeroTopography())
    assert list(b.surfaces) == [len(b.interfaces) - 1]
    assert list(b.refined([4.0e6]).surfaces) == [len(b.interfaces)]
    assert b.without_surface(-1).surfaces == {}
    assert ReferenceBody(b.layers, interfaces=b.interfaces).surfaces == {}


# -- a three-layer toy body: solid core, fluid shell, solid mantle ----------

def toy_body():
    from planetmodel import RadialField
    from planetmodel.model.units import Dimensions
    sk = Skeleton([0.0, 1.0, 2.0, 3.0])
    rho = RadialField(sk, [lambda r: 3.0 + 0 * r, lambda r: 2.0 + 0 * r,
                           lambda r: 1.0 + 0 * r], name="rho",
                      dimensions=Dimensions.DENSITY)
    vs = RadialField(sk, [lambda r: 1.0 + 0 * r, lambda r: 0.0 * r,
                          lambda r: 1.0 + 0 * r], name="vs")
    return (ReferenceBody.from_fields(sk, {"rho": rho, "vs": vs})
            .annotate(0, name="core").annotate(1, name="oc")
            .annotate(2, name="mantle").name_interface(1, "cmb", role="control"))


def test_a_buffer_stays_vacuum_when_states_are_classified():
    b = toy_body().with_buffer(ratio=0.5).classify_states()
    assert [lay.state for lay in b.layers] == ["solid", "fluid", "solid", "vacuum"]


def test_an_override_is_not_asked_of_the_rule():
    calls = []

    def rule(b, i):
        calls.append(i)
        return "solid"

    toy_body().classify_states(rule=rule, overrides={"oc": "fluid"})
    assert 1 not in calls


def test_surgered_bodies_still_differentiate_and_integrate():
    buffered = toy_body().with_buffer(ratio=0.5)
    # The buffer holds no fields, so rho has no piece there to ask for.
    with pytest.raises(ValueError, match="not defined on layer"):
        buffered.rho[-1]
    assert buffered.rho.domain == (0, 1, 2)
    assert float(buffered.rho.derivative()[-2](2.5)) == 0.0

    coarse, _ = toy_body().coarsened(drop=[0])
    merged = coarse.rho[0]
    # The fine model's integral, split at the boundary it merged away.
    assert merged.integrate(0.0, 2.0) == pytest.approx(3.0 * 1.0 + 2.0 * 1.0)
    assert float(merged.derivative()(1.5)) == pytest.approx(0.0)


def test_coarsening_renumbers_between():
    coarse, _ = toy_body().coarsened(drop=[0], state="fluid")
    assert [f.between for f in coarse.interfaces] == [(0, 1), (1, -1)]


def test_refining_keeps_the_names_of_untouched_layers():
    fine = toy_body().refined([2.5])
    assert [lay.name for lay in fine.layers] == ["core", "oc", None, None]


def test_extending_with_a_dict_keeps_the_other_fields():
    from planetmodel import RadialField
    b = toy_body()
    sk = b.skeleton.extended([4.0])
    extra = RadialField(sk, [lambda r: 0 * r] * 4, name="q")
    ext = b.extended([4.0], fields={"q": extra})
    assert set(ext.field_names) == {"rho", "vs", "q"}
    assert ext.rho.domain == (0, 1, 2)          # the new shell holds no rho
    assert ext.q.domain == (0, 1, 2, 3)
    with pytest.raises(ValueError, match="not defined"):
        ext.rho.evaluate(3.5)


def test_a_derived_view_keeps_its_units_through_surgery():
    from planetmodel.model.fields.composite import ComposedField
    from planetmodel.model.units import Dimensions
    b = toy_body()
    view = ComposedField(lambda rho, vs: rho * vs, [b.rho, b.vs], name="p",
                         dimensions=Dimensions.DENSITY)
    b = ReferenceBody.from_fields(b.skeleton, {"rho": b.rho, "vs": b.vs,
                                               "p": view})
    assert b.refined([2.5])["p"].dimensions == Dimensions.DENSITY


def test_truncating_at_a_named_boundary_keeps_its_name_and_role():
    cut = toy_body().truncated(2.0)
    assert cut.interfaces[-1].name == "cmb"
    assert cut.interfaces[-1].role == "control"
    assert toy_body().truncated(2.0, name="top").interfaces[-1].name == "top"


def test_a_taper_at_or_beyond_the_last_knot_is_refused():
    from planetmodel import layer_linear
    with pytest.raises(ValueError, match="taper radius"):
        layer_linear(inner_taper_radius=3.0)(toy_body())
