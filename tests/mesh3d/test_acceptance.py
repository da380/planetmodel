"""The acceptance structure: the brief's mesh, on a body the size of a test.

The meshing brief describes a PREM mesh with topography on the Moho and on the
surface, a control floor, a crust left to the consumer, a vacuum buffer
and order-2 elements, and lists what must be true of it.  Every one of
those statements is about the *mesher's handling of a body* -- numbering,
names, roles, flags, radii, validity, knots -- and the mesher does not
know or care whether the body came from a deck or from four numbers.  So
the body here is four numbers: three shells of unit radius, carrying the
full §4.2 structure, meshed in well under a second.  That PREM and
CRUST-1.0 produce a body of this shape is a separate question, answered
with no mesh at all in tests/model/test_prem_for_mesher.py.

The relief has exactly zero mean, which is what lets the interface
radius check stay at 1e-6: the boundaries carry their shape and still
average to the radii the model gives them.
"""
import numpy as np
import pytest

from planetmodel import DENSITY, RadialField, ReferenceBody, Skeleton, layer_linear
from planetmodel.model.units import Dimensions
from planetmodel.io import manifest as sc
from planetmodel.mesh3d import (BufferSpec, MeshSpec, UniformInterfaces,
                           build_layered_mesh)
from planetmodel.model.topography import AnalyticTopography

pytestmark = pytest.mark.gmsh

#: One mesh length unit, in metres.
RREF = 1.0e6

#: In the body's units, three times finer than the thinnest span.
COARSE = UniformInterfaces(0.15e6, 0.30e6, 0.30e6)


def relief(amplitude):
    """A degree-two shape, zonal plus sectoral, with exactly zero mean.

    P_2(cos t) integrates to zero over the sphere and so does
    sin^2 t cos 2p, so the boundary keeps the mean radius the model
    gives it while carrying relief in both angles -- which is what makes
    a 1e-6 check on the interface radii meaningful rather than lucky.
    """
    return AnalyticTopography(
        lambda t, p: amplitude * (0.5 * (3.0 * np.cos(t) ** 2 - 1.0)
                                  + np.sin(t) ** 2 * np.cos(2.0 * p)))


def deck():
    """A three-shell stand-in for a deck: it stops short of the Moho.

    It carries one field, a constant density, because since stage
    three a layer's material is what its fields say: a shell with no
    fields is one the consumer fills in, and the mantle here is not.
    """
    sk = Skeleton([0.0, 0.2e6, 0.55e6, 0.96e6])
    rho = RadialField(sk, [lambda r: 5.0e3 + 0.0 * r] * 3, name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    return (ReferenceBody.from_fields(sk, {"rho": rho})
            .annotate(0, name="core").annotate(1, name="mantle")
            .name_interface(0, "icb").name_interface(1, "cmb"))


def acceptance_spec(*, r_moho=0.95e6, r_surf=RREF, moho_amp=8.0e3,
                    surf_amp=3.0e3, exaggeration=1.0, order=2):
    """Plan §4.2's surgery, as one MeshSpec.

    Put the Moho where the crustal model says, insert the control floor
    300 km below it, add the crust back with its material left to the
    consumer, wrap it in a vacuum buffer, and hang the relief on both
    boundaries.
    """
    return MeshSpec(
        body=deck(), rref=RREF, order=order, sizing=COARSE,
        outer_radius=r_moho, outer_name="moho",
        insert_radii=[r_moho - 0.3e6], insert_names=["floor"],
        insert_role="control",
        extend_radii=[r_surf], extend_names=["surface"], extend_fields=None,
        buffers=[BufferSpec(ratio=0.2)],
        surfaces={"moho": relief(moho_amp) * exaggeration,
                  "surface": relief(surf_amp) * exaggeration},
        mapping_rule=layer_linear(), delivery="physical")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_layered_mesh(acceptance_spec(),
                              tmp_path_factory.mktemp("acc") / "shells")


# ------------------------------------------- the plan's five criteria

def test_the_attribute_structure_is_the_brief_s_convention(built):
    """1..N centre-outward, named, one buffer, control on exactly the floor."""
    card = sc.read(built.manifest_path)

    assert [lay["attribute"] for lay in card.layers] == \
        list(range(1, len(card.layers) + 1))
    assert [f["attribute"] for f in card.interfaces] == \
        list(range(1, len(card.interfaces) + 1))
    assert [f["name"] for f in card.interfaces] == \
        ["icb", "cmb", "floor", "moho", "surface", "buffer"]

    assert [lay["name"] for lay in card.layers if lay["is_vacuum"]] == ["buffer"]
    assert [f["name"] for f in card.interfaces
            if f["role"] == "control"] == ["floor"]
    # The crust is the consumer's to fill; the mantle below it is not.
    # What a layer has is what its fields say: the four model layers
    # carry the density, the crust and the buffer carry nothing.
    assert [lay["fields"] for lay in card.layers] == \
        [["rho"]] * 4 + [[], []]


def test_the_interfaces_sit_where_the_model_puts_them(built):
    """To 1e-6 non-dimensional -- a metre at the scale of the Earth."""
    assert built.validation.max_interface_radius_error < 1e-6


def test_every_validation_check_is_green(built):
    """Zero negative Jacobians, and nothing for MFEM to silently fix."""
    report = built.validation
    assert report.ok, report.failures
    assert report.negative_jacobians == 0
    assert report.negative_cells == 0
    assert report.inward_faces == 0
    assert report.knots_aligned
    assert sc.read(built.manifest_path).validation["wrong_orientation"] == 0


def test_the_element_count_is_of_a_sane_order(built):
    """Sanity, not tolerance: a unit body eight elements across is ~1e4."""
    assert 1e3 < built.counts["elements"] < 1e5
    assert built.counts["nodes"] > built.counts["elements"] / 10


def test_the_mapping_reaches_the_nodes_and_the_manifest(built):
    card = sc.read(built.manifest_path)
    assert card.delivery == "physical"
    assert card.mapping["rule"]["name"] == "layer_linear"
    assert card.mapping["applied_to_nodes"] is True
    assert card.provenance["perturbation"]["nodes"] == built.counts["nodes"]
    assert card.provenance["perturbation"]["validity_margin"] > 0.5
    # Every knot is a meshed interface: the floor, the Moho, the surface
    # and the buffer, so dh/dr never jumps inside an element.
    knots = card.mapping["knots_nd"]
    radii = [f["mean_radius_nd"] for f in card.interfaces]
    assert all(min(abs(np.asarray(radii) - k)) < 1e-9 for k in knots if k > 0)


# ------------------------------------------------------- the geometry

def test_the_body_grows_to_meet_a_boundary_above_it():
    """A crustal model may put the Moho above where the deck stops.

    CRUST-1.0's mean Moho is 6349.6 km and prem.nocrust ends at 6346.6,
    so a deck has to reach up to meet it.  Growing must merge the old
    boundary away: the material is the same extrapolated mantle on both
    sides, so an interface there is a discontinuity that is not one, and
    a sliver shell would dictate the element size of the whole crust.
    """
    from planetmodel.mesh3d.layered import _boundary_at
    body = deck()
    outer = float(body.skeleton.boundaries[-1])

    grown, did = _boundary_at(body, outer + 0.03e6, "moho")
    assert did
    assert grown.skeleton.boundaries[-1] == pytest.approx(outer + 0.03e6)
    assert outer not in list(grown.skeleton.boundaries)   # merged away
    assert grown.interfaces[-1].name == "moho"
    assert len(grown.interfaces) == len(body.interfaces)

    cut, did = _boundary_at(body, outer - 0.1e6, "moho")
    assert not did
    assert cut.skeleton.boundaries[-1] == pytest.approx(outer - 0.1e6)
    assert cut.interfaces[-1].name == "moho"


def test_relief_too_large_for_the_crust_is_refused_before_meshing(tmp_path):
    """The remedies the plan names reach the caller, and nothing is written.

    Refused analytically, before gmsh is asked anything: a mapping that
    folds on a lattice will certainly fold on the nodes, and finding
    that out after meshing is a wasted mesh.
    """
    spec = acceptance_spec(exaggeration=20.0)
    path = tmp_path / "folded"
    with pytest.raises(ValueError, match="orientation-preserving"):
        build_layered_mesh(spec, path)
    assert not path.with_suffix(".msh").exists()
    assert not path.with_suffix(".json").exists()
