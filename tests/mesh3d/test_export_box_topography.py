"""A box and a rectangle under topography through the MFEM export.

The reference mesh is deformed at the export, as the sphere's is: in
physical delivery the nodes are moved by the stretch, MFEM reads the
mesh as a consumer is told to, every Jacobian is positive, the free
surface's nodes lie on the topography and an interface's on its relief
to rounding, the base's and the sides' on their planes; in referential
delivery the displacement written is the stretch's, w e_z, at every
nodal dof.  A model's fields are referential at the dofs: the model's
own field at the dof's reference point, which the physical mesh places
where the mapping takes it, to the curved mesh's interpolation of the
mapping.  A stretch folding the box is refused by name, nothing
written.  A fault built in under the range has its nodes on the true
fault in the physical mesh MFEM holds, and AdGIA's range over its fault
at unit scale goes through whole.
"""
import numpy as np
import pytest

from planetmodel import DENSITY, SCALAR, Skeleton, cartesian
from planetmodel.fault import Fault
from planetmodel.layerfunction import polynomial_layer
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, export_mfem,
                                export_mfem_mesh, manifest as sc)
from planetmodel.mesh3d.export import MESH_READ_OPTIONS, _node_array
from planetmodel.vocabulary import FieldSpec

from test_box_topography import (LOCAL, SIZING, ZERO, Plane, adgia, box, ridge, root,
                                 topography)

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]


def rectangle():
    """The two layers over x in [0, 1], a rectangle, under the range's
    profile at y = 0 and the root's: the same at every y."""
    g = cartesian.Geometry(Skeleton([0.0, 0.4, 1.0]), cartesian.Rectangle((0.0, 1.0)),
                           layer_names=["crust", "mantle"],
                           interface_names=["surface", "moho"])
    return g.stretched(cartesian.layer_linear(
        g.skeleton, [lambda x, y: ridge(x, 0.0) + 0.0 * y,
                     lambda x, y: root(x, 0.0) + 0.0 * y, None]))


GEOMETRIES = {"box": topography, "rectangle": rectangle}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each geometry under topography built once at order 2."""
    root_ = tmp_path_factory.mktemp("topography")
    cache = {}

    def get(kind):
        if kind not in cache:
            cache[kind] = build_layered_mesh(MeshSpec(GEOMETRIES[kind](), SIZING,
                                                      order=2), root_ / kind)
        return cache[kind]
    return get


def load(path):
    o = MESH_READ_OPTIONS
    return mfem.Mesh(str(path), o["generate_edges"], o["refine"], o["fix_orientation"])


def lifted(X, card):
    """The physical points (n, 3) of points in the mesh's coordinates."""
    out = np.zeros((len(X), 3))
    out[:, list(card.mesh_axes(X.shape[1]))] = X
    return out


def boundary_nodes(mesh, attribute):
    """The positions of every nodal dof, high-order ones included, of the
    boundary elements of an attribute, in the mesh's coordinates."""
    nodes = np.asarray(_node_array(mesh.GetNodes()), dtype=float)
    fes = mesh.GetNodes().FESpace()
    dofs = [np.asarray(fes.GetBdrElementDofs(b), dtype=int)
            for b in range(mesh.GetNBE()) if mesh.GetBdrAttribute(b) == attribute]
    return nodes[np.unique(np.concatenate(dofs))]


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_physical_mesh_lies_on_the_topography_with_positive_jacobians(
        built, kind, tmp_path):
    """MFEM reads the moved mesh; no element folds; every nodal dof of the
    surface lies on the topography and of the Moho on its relief to
    rounding, of the base and the sides on their planes."""
    res = built(kind)
    export = export_mfem_mesh(res, tmp_path / kind, delivery="physical")
    card = sc.read(export.manifest_path)
    assert card.mesh["nodes"] == "physical" and card.mesh["displacement"] is None
    assert export.quality["negative_jacobians"] == 0
    assert 0.0 < export.quality["min_ratio"] < 1.0       # curved by the relief
    mesh = load(export.mesh_path)
    assert mesh.CheckElementOrientation(False) == 0
    assert mesh.CheckBdrElementOrientation(False) == 0
    for face, relief, depth in zip(card.interfaces, (ridge, root), (0.0, 0.4)):
        x = lifted(boundary_nodes(mesh, face["attribute"]), card)
        assert np.abs(x[:, 2] - (relief(x[:, 0], x[:, 1]) - depth)).max() < 1e-12
    for face in card.faces:
        axis = card.coordinates.index(face["axis"])
        x = boundary_nodes(mesh, face["attribute"])
        assert np.abs(x[:, axis] - face["value"]).max() < 1e-12
    if kind == "rectangle":
        assert mesh.SpaceDimension() == 2


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_referential_delivery_writes_the_stretch_s_displacement(built, kind,
                                                                    tmp_path):
    """The mesh as built, and beside it w e_z at every nodal dof, exactly
    the stretch's displacement there; adding it gives the physical mesh."""
    res = built(kind)
    ref = export_mfem_mesh(res, tmp_path / "ref", delivery="referential")
    phys = export_mfem_mesh(res, tmp_path / "phys", delivery="physical")
    card = sc.read(ref.manifest_path)
    assert card.mesh["nodes"] == "reference" and card.mesh["displacement"] == \
        "displacement"
    # every array read is copied while the MFEM object holding it lives
    mesh, moved = load(ref.mesh_path), load(phys.mesh_path)
    gf = mfem.GridFunction(mesh, str(ref.displacement_path))
    X = np.array(_node_array(mesh.GetNodes()), dtype=float)
    u = np.array(_node_array(gf), dtype=float)
    x = np.array(_node_array(moved.GetNodes()), dtype=float)
    P = lifted(X, card)
    want = res.mapping.displacement(P)[:, list(card.mesh_axes(X.shape[1]))]
    assert np.abs(u - want).max() < 1e-15
    assert np.all(u[:, :-1] == 0.0)                     # nothing moves laterally
    assert np.abs(x - (X + u)).max() < 1e-15


def model_on(geometry):
    """A flat model: a density of 1 and 2 by layer, and q = 2 depth."""
    specs = {"q": FieldSpec(SCALAR, dimensions=None)}
    layers = []
    for lay in geometry.layers:
        iv = lay.interval
        layers.append({
            "rho": cartesian.constant_field(iv, 1.0 + lay.index, character=DENSITY),
            "q": cartesian.DepthField(iv, polynomial_layer(iv, [0.0, 2.0]))})
    return cartesian.Model(geometry, layers, specs=specs)


def dof_points(mesh, fes):
    """Where every dof of `fes` sits, through MFEM's element transformations."""
    X = np.zeros((fes.GetNDofs(), mesh.SpaceDimension()))
    for e in range(mesh.GetNE()):
        pts = mfem.DenseMatrix()
        mesh.GetElementTransformation(e).Transform(fes.GetFE(e).GetNodes(), pts)
        X[np.asarray(fes.GetElementDofs(e), dtype=int)] = pts.GetDataArray().T
    return X


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_fields_at_the_deformed_dofs_are_the_model_s_at_their_reference_points(
        built, kind, tmp_path):
    """In physical delivery each dof of a field sits where the moved mesh
    puts it; its value is the model's own field at the dof's reference
    point, which the stretch takes there to the curved mesh's
    interpolation of it: the model evaluated at the inverse of the dof's
    physical point agrees to that interpolation error."""
    res = built(kind)
    model = model_on(res.geometry)
    export = export_mfem(res, tmp_path / kind, model=model, delivery="physical")
    reference = export_mfem(res, tmp_path / f"{kind}_ref", model=model,
                            delivery="referential")
    card = sc.read(export.manifest_path)
    moved, still = load(export.mesh_path), load(reference.mesh_path)
    gf = mfem.GridFunction(moved, str(export.field_paths["q"]))
    q = np.array(_node_array(gf), dtype=float)[:, 0]
    fes = gf.FESpace()                     # held by gf, which is kept alive
    x = lifted(dof_points(moved, fes), card)
    X = lifted(dof_points(still, fes), card)
    assert np.abs(q - 2.0 * -X[:, 2]).max() < 1e-12       # the model at X
    back = res.mapping.inverse(x)
    assert np.abs(back[:, :2] - X[:, :2]).max() < 1e-12
    # the moved mesh interpolates the stretch at its nodes, quadratically
    # between them: measured 4.6e-3 (box) and 1.5e-3 (rectangle) off m(X)
    # inside the elements of order 2, a few percent of an element, so q
    # by 8.9e-3 and 2.4e-3
    assert np.abs(res.mapping(X) - x).max() < 0.01
    assert np.abs(q - 2.0 * -back[:, 2]).max() < 0.02


def test_a_stretch_folding_the_box_is_refused_at_the_export(tmp_path):
    """A Moho raised above the surface over part of the box, the geometry
    built unchecked: the physical export refuses it by name and writes
    nothing; the referential export writes it, for a consumer to see."""
    g = box()
    over = cartesian.layer_linear(g.skeleton, [None, lambda x, y: 0.6 * np.exp(
        -((x - 0.5) ** 2 + y ** 2) / 0.02), None])
    folded = g.stretched(over, check=False)
    assert not folded.validity()
    res = build_layered_mesh(MeshSpec(folded, SIZING, order=1), tmp_path / "folded")
    with pytest.raises(ValueError, match=r"the mapping VerticalStretch\(.*\) folds \d+ "
                                         r"element\(s\).*Nothing was written"):
        export_mfem_mesh(res, tmp_path / "out")
    assert not (tmp_path / "out.mesh").exists()
    ref = export_mfem_mesh(res, tmp_path / "ref", delivery="referential")
    assert ref.quality["negative_jacobians"] == 0      # the reference mesh is fine


def test_a_fault_under_the_range_lies_on_the_true_fault_in_mfem_s_mesh(tmp_path):
    """A fault cut by the topography, exported physical: its boundary
    elements' nodal dofs, as MFEM holds them, lie on the plane to the
    pull-back's offset, and each face's plus cell is on its normal's
    side."""
    fault = Plane(LOCAL, [0.55, 0.0, -0.1], strike=0.0, dip=60.0, length=0.4,
                  width=0.5, slip=ZERO)
    res = build_layered_mesh(MeshSpec(topography(), SIZING, order=2,
                                      faults=[EmbeddedFault(Fault([fault]), name="f")]),
                             tmp_path / "faulted")
    export = export_mfem_mesh(res, tmp_path / "phys", delivery="physical")
    assert export.quality["negative_jacobians"] == 0
    card = sc.read(export.manifest_path)
    (record,) = card.faults
    mesh = load(export.mesh_path)
    x = boundary_nodes(mesh, record["attribute"])
    found = Fault([fault]).locate(x, strict=False)
    assert np.abs(found.distance).max() < 2.0 * record["offset"] + 1e-12
    xyz = np.asarray(mesh.GetVertexArray(), dtype=float)
    normal = fault.normal(0.0, 0.0)
    for b in range(mesh.GetNBE()):
        if mesh.GetBdrAttribute(b) != record["attribute"]:
            continue
        v = xyz[list(mesh.GetBdrElementVertices(b))]
        n = np.cross(v[1] - v[0], v[2] - v[0])
        assert n @ normal > 0.0


def test_adgia_s_range_goes_through_the_export(tmp_path):
    """AdGIA's range over its fault at unit scale: MFEM reads the physical
    mesh, no element folds, the surface's nodal dofs lie on the range."""
    g, fault, height = adgia()
    res = build_layered_mesh(
        MeshSpec(g, UniformInterfaces(0.08, 0.15, 0.2), order=2,
                 faults=[EmbeddedFault(Fault([fault]), name="fault",
                                       sizing=InterfaceSizing(0.04, 0.15, 0.1))]),
        tmp_path / "adgia")
    export = export_mfem_mesh(res, tmp_path / "phys")
    assert export.quality["negative_jacobians"] == 0
    # measured: 0.43, the steep front of the range curving its elements
    assert export.quality["min_ratio"] > 0.2
    card = sc.read(export.manifest_path)
    mesh = load(export.mesh_path)
    assert mesh.CheckElementOrientation(False) == 0
    x = boundary_nodes(mesh, card.interfaces[0]["attribute"])
    assert np.abs(x[:, 2] - height(x[:, 0], x[:, 1])).max() < 1e-12
    assert x[:, 2].max() > 0.08
