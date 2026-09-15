"""A field belongs to one layer.

What a body stores is single-layer fields; what `body[name]` returns is a
view assembled from them, with a domain.  These tests pin the pieces,
the views, the assembly by type, and the surgery as list operations.
"""
import numpy as np
import pytest

from planetmodel import PREM, AnalyticField, RadialField, ReferenceBody, Skeleton
from planetmodel.model.body import Layer
from planetmodel.model.character import DENSITY, SCALAR, VECTOR
from planetmodel.model.fields.composite import ComposedField, RestrictedField
from planetmodel.model.fields.layerwise import LayerwiseField, assemble, split
from planetmodel.model.materials import ElasticField, Symmetry
from planetmodel.model.units import Dimensions
from planetmodel.testing import check_field


def const(sk, value, name, *, character=SCALAR, dims=Dimensions.DIMENSIONLESS):
    return RadialField(sk, [lambda r, v=value: v + 0.0 * r] * sk.nlayers,
                       name=name, character=character, dimensions=dims)


@pytest.fixture
def sk():
    return Skeleton([0.0, 1.0, 2.0, 3.0])


@pytest.fixture
def body(sk):
    rho = RadialField(sk, [lambda r: 3.0 + 0 * r, lambda r: 2.0 + 0 * r,
                           lambda r: 1.0 + 0 * r], name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    return ReferenceBody.from_fields(sk, {"rho": rho, "q": const(sk, 7.0, "q")})


# ----------------------------------------------------------- pieces

def test_a_layer_holds_single_layer_fields(body):
    lay = body.layer(1)
    assert lay.interval == (1.0, 2.0)
    assert lay.field_names == ("rho", "q")
    piece = lay["rho"]
    assert piece.skeleton.nlayers == 1
    assert tuple(piece.skeleton.boundaries) == (1.0, 2.0)
    assert piece(1.5) == 2.0                    # a single-layer field is callable
    assert lay["rho"] is piece
    assert "rho" in lay and "vs" not in lay


def test_indexing_a_view_gives_the_piece(body):
    rho = body.rho
    assert rho[1] is body.layer(1)["rho"]
    assert rho[1].function(1.5) == 2.0
    assert rho.restricted(1) is rho[1]
    assert [p(0.5 + i) for i, p in enumerate(rho)] == [3.0, 2.0, 1.0]


def test_a_layer_refuses_a_field_off_its_interval(sk):
    lay = Layer(0, interval=(0.0, 1.0))
    wrong = const(Skeleton([1.0, 2.0]), 1.0, "x")
    with pytest.raises(ValueError, match="not on this layer"):
        lay.with_field("x", wrong)
    with pytest.raises(ValueError, match="spans 3 layers"):
        lay.with_field("x", const(sk, 1.0, "x"))
    ok = lay.with_field("x", const(Skeleton([0.0, 1.0]), 1.0, "x"))
    assert ok.field_names == ("x",) and lay.field_names == ()   # copy-on-write


def test_a_vacuum_layer_holds_nothing():
    with pytest.raises(ValueError, match="vacuum layer holds no fields"):
        Layer(0, interval=(0.0, 1.0), state="vacuum",
              fields={"x": const(Skeleton([0.0, 1.0]), 1.0, "x")})
    lay = Layer(0, interval=(0.0, 1.0), state="vacuum")
    assert lay.is_vacuum and lay.is_fluid
    assert lay.fields == {}
    with pytest.raises(ValueError, match="vacuum"):
        lay.with_field("x", const(Skeleton([0.0, 1.0]), 1.0, "x"))


# ------------------------------------------------------------ views

def test_the_view_is_assembled_and_cached(body):
    v = body["rho"]
    assert isinstance(v, RadialField)
    assert v.skeleton == body.skeleton
    assert v.domain == (0, 1, 2)
    assert body["rho"] is v
    assert body.rho is v
    check_field(v)


def test_a_view_has_a_domain_and_refuses_outside_it(sk):
    rho = const(sk, 1.0, "rho", character=DENSITY, dims=Dimensions.DENSITY)
    b = ReferenceBody.from_fields(sk, {"rho": rho}).without_field("rho")
    b = b.with_field(0, "rho", rho[0]).with_field(2, "rho", rho[2])
    assert b.layers_with("rho") == (0, 2)
    v = b.rho
    assert v.domain == (0, 2)
    assert v.evaluate([0.5, 2.5]).tolist() == [1.0, 1.0]
    with pytest.raises(ValueError, match="not defined on layer 1"):
        v.evaluate(1.5)
    with pytest.raises(ValueError, match="not defined on layer 1"):
        v[1]
    with pytest.raises(ValueError, match="reaches into layer 1"):
        v.integrate(0.0, 3.0)
    assert v.integrate(0.0, 1.0) == pytest.approx(1.0)
    check_field(v)


def test_from_layers_builds_the_skeleton_from_the_intervals():
    a = Layer(0, interval=(0.0, 1.0),
              fields={"rho": const(Skeleton([0.0, 1.0]), 2.0, "rho")})
    b = Layer(1, interval=(1.0, 2.5), name="shell")
    body = ReferenceBody.from_layers([a, b])
    assert tuple(body.skeleton.boundaries) == (0.0, 1.0, 2.5)
    assert body.layer("shell").index == 1
    assert body.rho.domain == (0,)
    assert body.common_fields() == ()
    with pytest.raises(ValueError, match="must abut"):
        ReferenceBody.from_layers([a, Layer(1, interval=(1.5, 2.0))])
    with pytest.raises(ValueError, match="no interval"):
        ReferenceBody.from_layers([a, Layer(1)])


def test_add_field_splits_a_body_wide_field_and_places_a_piece(body, sk):
    v = body.add_field("w", const(sk, 4.0, "w"))
    assert all("w" in lay for lay in body.layers)
    assert v.domain == (0, 1, 2)
    body.add_field("z", const(Skeleton([2.0, 3.0]), 9.0, "z"))
    assert body.layers_with("z") == (2,)
    with pytest.raises(ValueError, match="matches no layer"):
        body.add_field("bad", const(Skeleton([0.5, 1.5]), 1.0, "bad"))
    with pytest.raises(ValueError, match="exists"):
        body.add_field("z", const(Skeleton([2.0, 3.0]), 9.0, "z"))
    # replace=True replaces every piece, dropping layers the new one lacks
    body.add_field("w", const(Skeleton([0.0, 1.0]), 5.0, "w"), replace=True)
    assert body.layers_with("w") == (0,)
    assert body.w(0.5) == 5.0


def test_field_names_are_the_union_in_first_appearance_order(body):
    body.add_field("z", const(Skeleton([2.0, 3.0]), 9.0, "z"))
    body.add_field("a", const(Skeleton([0.0, 1.0]), 9.0, "a"))
    # first appearance scanning the layers centre outward
    assert body.field_names == ("rho", "q", "a", "z")
    assert body.common_fields() == ("rho", "q")


# --------------------------------------------------- assembly by type

def test_an_elastic_field_reassembles_as_itself():
    prem = PREM()
    el = prem["elastic_moduli"]
    assert isinstance(el, ElasticField) and el.symmetry is Symmetry.VTI
    assert el.domain == tuple(range(prem.skeleton.nlayers))
    piece = prem.layer(3)["elastic_moduli"]
    assert isinstance(piece, ElasticField) and piece.skeleton.nlayers == 1
    r = np.array([4.0e6])
    assert np.allclose(piece.evaluate(r), el.evaluate(r))
    check_field(piece)


def test_composites_split_and_reassemble_as_themselves(body):
    view = ComposedField(lambda a, b: a * b, [body.rho, body.q], name="p")
    body.add_field("p", view)
    piece = body.layer(1)["p"]
    assert isinstance(piece, ComposedField)
    assert piece(1.5) == 14.0
    again = body["p"]
    assert isinstance(again, ComposedField)
    assert again.evaluate([0.5, 2.5]).tolist() == [21.0, 7.0]
    s = body.q + 2.0 * body.q
    body.add_field("s", s)
    assert body["s"].evaluate(1.5) == 21.0
    assert type(body["s"]).__name__ == "SumField"


def test_a_generic_field_split_into_a_body_comes_back_as_itself(sk):
    fn = lambda r, t, p: np.stack([r, np.cos(t), np.sin(p)], axis=-1)  # noqa: E731
    vec = AnalyticField(fn, sk, character=VECTOR, name="v")
    b = ReferenceBody.from_fields(sk, {"v": vec})
    assert isinstance(b.layer(0)["v"], AnalyticField)
    assert isinstance(b["v"], AnalyticField)
    assert b["v"].skeleton == sk

    class Opaque:
        """A Field that is only the protocol: no restrict, no assemble."""
        skeleton, character, name = sk, SCALAR, "o"
        dimensions = None

        def evaluate(self, r, theta=None, phi=None, *, layer=None,
                     side="upper", frame="spherical"):
            return np.asarray(r, dtype=float) * 2.0

    from planetmodel.model.fields.composite import FieldBase

    class OpaqueBase(FieldBase, Opaque):
        pass

    o = OpaqueBase()
    b.add_field("o", o)
    assert isinstance(b.layer(1)["o"], RestrictedField)
    assert b["o"] is o                          # restrictions of one source: the source
    half = b.without_field("o").with_field(2, "o", o.restricted(2))
    view = half["o"]
    assert isinstance(view, LayerwiseField)
    assert view.domain == (2,)
    assert view(2.5) == 5.0
    with pytest.raises(ValueError, match="not defined"):
        view(0.5)
    check_field(view)


def test_assemble_refuses_overlaps_and_strays(sk):
    a = const(Skeleton([0.0, 1.0]), 1.0, "a")
    with pytest.raises(ValueError, match="overlap"):
        assemble(sk, [a, const(Skeleton([0.5, 1.5]), 1.0, "a")])
    with pytest.raises(ValueError, match="outside the skeleton"):
        assemble(sk, [const(Skeleton([2.0, 4.0]), 1.0, "a")])
    with pytest.raises(ValueError, match="single-layer"):
        assemble(sk, [const(sk, 1.0, "a")])
    assert split(const(sk, 1.0, "a"))[1].skeleton.interval(0) == (1.0, 2.0)


# ---------------------------------------------------- surgery as lists

def test_coarsening_merges_pieces_and_keeps_the_fine_material():
    # Unequal fine layers, so no probe of check_field lands on the
    # boundary merged away (where the two fine values legitimately differ).
    sk = Skeleton([0.0, 1.0, 2.5, 3.5])
    rho = RadialField(sk, [lambda r: 3.0 + 0 * r, lambda r: 2.0 + 0 * r,
                           lambda r: 1.0 + 0 * r], name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    body = ReferenceBody.from_fields(sk, {"rho": rho})
    coarse, cmap = body.coarsened(drop=[0])
    lay = coarse.layer(0)
    assert lay.interval == (0.0, 2.5)
    merged = lay["rho"]
    assert isinstance(merged, RadialField) and merged.skeleton.nlayers == 1
    assert merged(0.5) == 3.0 and merged(1.5) == 2.0
    assert merged.integrate(0.0, 2.5) == pytest.approx(3.0 + 1.5 * 2.0)
    assert coarse.rho.evaluate([0.5, 1.5, 3.0]).tolist() == [3.0, 2.0, 1.0]
    check_field(coarse.rho)


def test_coarsening_warns_and_drops_a_one_sided_field(body):
    body.add_field("z", const(Skeleton([0.0, 1.0]), 9.0, "z"))
    with pytest.warns(UserWarning, match="drops \\['z'\\]"):
        coarse, _ = body.coarsened(drop=[0])
    assert "z" not in coarse
    assert coarse.layer(0).field_names == ("rho", "q")


def test_refining_and_truncating_clip_the_pieces(body):
    fine = body.refined([1.5])
    assert [lay.interval for lay in fine.layers] == [
        (0.0, 1.0), (1.0, 1.5), (1.5, 2.0), (2.0, 3.0)]
    assert fine.layer(2)["rho"](1.7) == 2.0
    with pytest.raises(ValueError, match="outside"):
        fine.layer(1)["rho"](1.7)
    cut = body.truncated(2.5)
    assert cut.layer(2).interval == (2.0, 2.5)
    assert cut.rho.evaluate(2.4) == 1.0
    with pytest.raises(ValueError, match="outside"):
        cut.rho.evaluate(2.6)


def test_extending_adds_empty_or_extrapolated_shells(body):
    empty = body.extended([4.0])
    assert empty.layer(3).fields == {}
    assert empty.rho.domain == (0, 1, 2)
    grown = body.extended([4.0], fields="extrapolate")
    assert grown.layer(3)["rho"](3.5) == 1.0
    assert grown.rho.domain == (0, 1, 2, 3)
    piece = const(Skeleton([3.0, 4.0]), 8.0, "e")
    given = body.extended([4.0], fields={"e": piece})
    assert given.layers_with("e") == (3,) and given.e(3.5) == 8.0


def test_a_generic_field_refuses_to_extrapolate(sk):
    from planetmodel.model.fields.composite import FieldBase

    class Opaque(FieldBase):
        skeleton, character, name, dimensions = sk, SCALAR, "o", None

        def evaluate(self, r, theta=None, phi=None, *, layer=None,
                     side="upper", frame="spherical"):
            return np.asarray(r, dtype=float)

    b = ReferenceBody.from_fields(sk, {"o": Opaque()})
    with pytest.raises(TypeError, match="cannot extrapolate 'o'"):
        b.extended([4.0], fields="extrapolate")
    # ... but it clips: truncation and refinement go through a restriction
    assert b.truncated(2.5).layer(2)["o"](2.2) == 2.2
    assert b.refined([2.5]).layer(3)["o"](2.7) == 2.7


def test_with_layer_and_annotate_are_copy_on_write(body):
    lay = body.layer(1).annotated(name="mid", state="fluid")
    b2 = body.with_layer(1, lay)
    assert b2.layer("mid").is_fluid and body.layer(1).name is None
    with pytest.raises(ValueError, match="replacement spans"):
        body.with_layer(1, Layer(1, interval=(0.0, 1.0)))
    b3 = body.annotate(0, name="core")
    assert b3.layer(0).name == "core" and b3.layer(0).field_names == ("rho", "q")


def test_rescaling_converts_each_layer(body):
    from planetmodel.model.units import Scales
    nd = body.rescaled(Scales.geophysical(3.0, density=1.0))
    assert tuple(nd.skeleton.boundaries) == pytest.approx((0.0, 1 / 3, 2 / 3, 1.0))
    assert nd.layer(1).interval == pytest.approx((1 / 3, 2 / 3))
    assert nd.rho(0.5) == pytest.approx(2.0)          # rho / density scale
    assert nd.q(0.5) == 7.0                           # dimensionless
    assert nd.redimensionalised().rho(1.5) == pytest.approx(2.0)


def test_prem_round_trips_through_the_layers():
    prem = PREM()
    assert all(lay.fields for lay in prem.layers)
    assert prem.common_fields() == prem.field_names
    again = ReferenceBody.from_layers(prem.layers, meta=prem.meta)
    assert again.skeleton == prem.skeleton
    r = np.linspace(1e5, 6.3e6, 300)
    for name in ("rho", "vpv", "A", "elastic_moduli"):
        assert np.allclose(again[name].evaluate(r), prem[name].evaluate(r))
