"""A box and a rectangle to MFEM: read back as a consumer reads them, the
attributes by layer and by face, every element and boundary element
where its attribute says, positive Jacobians at every order, the fields
of a flat model by layer, the mapping applied with the identity's
nodes as the reference, and what is refused."""
import numpy as np
import pytest

from planetmodel import DENSITY, SCALAR, VECTOR, Geometry, Model, Skeleton, cartesian
from planetmodel.fields import constant_field as spherical_constant
from planetmodel.layerfunction import polynomial_layer
from planetmodel.mapping import MappingBase
from planetmodel.mesh3d import (MeshSpec, build_layered_mesh, export_mfem,
                                export_mfem_mesh, manifest as sc)
from planetmodel.mesh3d.export import MESH_READ_OPTIONS, node_array, jacobian_report
from planetmodel.vocabulary import FieldSpec

from test_box import DEPTHS, GEOMETRIES, ORDERS, SIZING, Stretch

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

SPECS = {"q": FieldSpec(SCALAR, dimensions=None),
         "w": FieldSpec(VECTOR, dimensions=None)}


def model_on(geometry) -> cartesian.Model:
    """A flat model: a density constant in each layer (1, 2, 3 from the
    top), q = 2 depth, and a vector w = (depth, x, y) in Cartesian
    components."""
    layers = []
    for lay in geometry.layers:
        iv = lay.interval
        layers.append({
            "rho": cartesian.constant_field(iv, 1.0 + lay.index, character=DENSITY),
            "q": cartesian.DepthField(iv, polynomial_layer(iv, [0.0, 2.0])),
            "w": cartesian.AnalyticField(
                iv, lambda d, x, y: np.stack(np.broadcast_arrays(d, x, y), axis=-1),
                character=VECTOR, frame="cartesian")})
    return cartesian.Model(geometry, layers, specs=SPECS)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each geometry built once per order, the identity's and a stretch's."""
    root = tmp_path_factory.mktemp("flat")
    cache = {}

    def _built(kind, order=2, *, mapping=None):
        key = (kind, order, mapping is None)
        if key not in cache:
            g = GEOMETRIES[kind](mapping=mapping)
            name = f"{kind}_{order}_{'identity' if mapping is None else 'stretch'}"
            cache[key] = build_layered_mesh(MeshSpec(g, SIZING, order=order),
                                            root / name)
        return cache[key]

    return _built


def load(path):
    """The mesh a consumer loads, constructed the way the manifest says."""
    o = MESH_READ_OPTIONS
    return mfem.Mesh(str(path), o["generate_edges"], o["refine"], o["fix_orientation"])


def physical(X, card):
    """The physical points (n, 3) of points in the mesh's coordinates."""
    out = np.zeros((len(X), 3))
    out[:, list(card.mesh_axes(X.shape[1]))] = X
    return out


@pytest.mark.parametrize("delivery", ["physical", "referential"])
@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_mfem_reads_the_export_with_attributes_by_layer_and_face(built, kind,
                                                                 delivery, tmp_path):
    res = built(kind)
    export = export_mfem_mesh(res, tmp_path / kind, delivery=delivery)
    mesh = load(export.mesh_path)
    d = {"box": 3, "rectangle": 2}[kind]
    assert mesh.Dimension() == mesh.SpaceDimension() == d
    assert mesh.CheckElementOrientation(False) == 0
    assert mesh.CheckBdrElementOrientation(False) == 0
    card = sc.read(export.manifest_path)
    assert card.schema == sc.SCHEMA_FLAT and card.mesh["format"] == "mfem"
    assert card.mesh["nodes"] == "reference"
    assert list(mesh.attributes.ToList()) == [1, 2, 3]
    assert list(mesh.bdr_attributes.ToList()) == \
        list(range(1, len(card.interfaces) + len(card.faces) + 1))
    assert [e["name"] for e in card.faces] == \
        {"box": ["base", "x_min", "x_max", "y_min", "y_max"],
         "rectangle": ["base", "x_min", "x_max"]}[kind]
    if delivery == "referential":
        (entry,) = card.fields
        assert entry["vdim"] == d and export.displacement_path.exists()
        gf = mfem.GridFunction(mesh, str(export.displacement_path))
        assert np.abs(np.asarray(node_array(gf))).max() == 0.0
    # every element in its layer, every boundary element on its plane
    xyz = physical(np.asarray(mesh.GetVertexArray(), dtype=float), card)
    for e in range(mesh.GetNE()):
        lo, hi = card.layer_interval(card.layers[mesh.GetAttribute(e) - 1])
        depth = -xyz[list(mesh.GetElementVertices(e)), 2]
        assert depth.min() >= lo - 1e-12 and depth.max() <= hi + 1e-12
    where = {f["attribute"]: (2, -f["depth"]) for f in card.interfaces}
    where |= {f["attribute"]: (sc.AXES.index(f["axis"]), f["value"])
              for f in card.faces}
    for b in range(mesh.GetNBE()):
        axis, value = where[mesh.GetBdrAttribute(b)]
        at = xyz[list(mesh.GetBdrElementVertices(b)), axis]
        assert np.abs(at - value).max() <= 1e-12


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_positive_jacobians_at_every_order(built, kind, order, tmp_path):
    export = export_mfem_mesh(built(kind, order), tmp_path / kind)
    assert export.counts["order"] == order
    assert export.quality["negative_jacobians"] == 0
    # straight-sided cells: the Jacobian is constant in each
    assert export.quality["min_ratio"] == pytest.approx(1.0, abs=1e-9)
    report = jacobian_report(load(export.mesh_path))
    assert report["negative_jacobians"] == 0 and report["min_ratio"] > 0.99


def dof_points(mesh, fes) -> np.ndarray:
    """Where every dof of `fes` sits, through MFEM's element transformations."""
    X = np.zeros((fes.GetNDofs(), mesh.SpaceDimension()))
    for e in range(mesh.GetNE()):
        pts = mfem.DenseMatrix()
        mesh.GetElementTransformation(e).Transform(fes.GetFE(e).GetNodes(), pts)
        X[np.asarray(fes.GetElementDofs(e), dtype=int)] = pts.GetDataArray().T
    return X


def dof_attributes(mesh, fes) -> np.ndarray:
    """The attribute of the element each dof of `fes` belongs to."""
    a = np.zeros(fes.GetNDofs(), dtype=int)
    for e in range(mesh.GetNE()):
        a[np.asarray(fes.GetElementDofs(e), dtype=int)] = mesh.GetAttribute(e)
    return a


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_fields_of_a_flat_model_are_written_by_layer(built, kind, tmp_path):
    res = built(kind)
    model = model_on(res.geometry)
    export = export_mfem(res, tmp_path / kind, model=model, delivery="referential")
    mesh = load(export.mesh_path)
    card = sc.read(export.manifest_path)
    assert set(export.field_paths) == {"rho", "q", "w"}
    gf = {name: mfem.GridFunction(mesh, str(path))
          for name, path in export.field_paths.items()}
    scalar = mfem.FiniteElementSpace(mesh, gf["rho"].FESpace().FEColl(), 1)
    X = physical(dof_points(mesh, scalar), card)
    attributes = dof_attributes(mesh, scalar)
    depth = -X[:, 2]
    values = {name: np.array(node_array(g), dtype=float) for name, g in gf.items()}
    assert np.array_equal(values["rho"][:, 0], attributes.astype(float))
    assert np.abs(values["q"][:, 0] - 2.0 * depth).max() < 1e-12
    # Cartesian components (x, y, z) of w = (depth, x, y), over a rectangle too
    want = np.stack([depth, X[:, 0], X[:, 1]], axis=1)
    assert values["w"].shape == (len(X), 3)
    assert np.abs(values["w"] - want).max() < 1e-12
    if kind == "rectangle":
        assert np.all(X[:, 1] == 0.0)
    entries = {e["name"]: e for e in card.fields}
    assert entries["rho"]["unit"] == "kg m-3" and entries["rho"]["radial_degree"] == 0
    assert entries["q"]["radial_degree"] == 1 and entries["w"]["vdim"] == 3
    assert entries["displacement"]["vdim"] == mesh.Dimension()
    # the one-sided values at each interface, the deeper layer first
    assert [f["values"]["rho"] for f in card.interfaces] == \
        [[1.0, None], [2.0, 1.0], [3.0, 2.0]]
    for f, want in zip(card.interfaces, ([0.0, None], [0.6, 0.6], [1.2, 1.2])):
        assert f["values"]["q"] == pytest.approx(want, rel=1e-12, abs=1e-15)
    assert card.interfaces[0]["kind"] == "free"
    assert card.scales is not None and card.faces == sc.read(res.manifest_path).faces


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_mapping_is_applied_at_the_export(built, kind, tmp_path):
    """Under a vertical stretch the physical mesh is the reference mesh
    plus the displacement, z running to -1.2."""
    res = built(kind, mapping=Stretch())
    ref = export_mfem_mesh(res, tmp_path / "ref", delivery="referential")
    phys = export_mfem_mesh(res, tmp_path / "phys", delivery="physical")
    assert sc.read(phys.manifest_path).mesh["nodes"] == "physical"
    mesh = load(ref.mesh_path)
    X = np.array(node_array(mesh.GetNodes()), dtype=float)
    gf = mfem.GridFunction(mesh, str(ref.displacement_path))
    u = np.array(node_array(gf), dtype=float)
    card = sc.read(ref.manifest_path)
    want = (Stretch()(physical(X, card)) - physical(X, card))[
        :, list(card.mesh_axes(X.shape[1]))]
    assert np.abs(u - want).max() < 1e-12
    moved = load(phys.mesh_path)
    x = np.array(node_array(moved.GetNodes()), dtype=float)
    assert np.abs(x - (X + u)).max() < 1e-12
    assert x[:, -1].min() == pytest.approx(-1.2, abs=1e-12)
    assert phys.quality["negative_jacobians"] == 0


class ShiftedAlongY(MappingBase):
    """A translation along y: invariant along y, but out of a rectangle's plane."""

    def __call__(self, X):
        return np.asarray(X, dtype=float) + [0.0, 0.1, 0.0]

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(np.eye(3), X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        return np.ones(np.shape(X)[:-1])


@pytest.mark.netcdf
@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_a_flat_model_file_is_written_beside_the_mesh_and_named_in_the_manifest(
        built, kind, tmp_path):
    """The model of a flat mesh's export beside it as a flat model file,
    on the grid given or, for the fields of depth alone, on the one node
    of the domain; the manifest names it by its schema."""
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    from planetmodel.modelfile import SCHEMA_FLAT, read_model
    from planetmodel.testing import check_model_file
    res = built(kind)
    model = model_on(res.geometry)
    domain = res.geometry.domain
    export = export_mfem(res, tmp_path / "file", model=model,
                         model_file=cartesian.Grid(domain, 4))
    card = sc.read(export.manifest_path)
    assert card.is_flat
    assert card.model == {"file": "file.model.nc", "schema": SCHEMA_FLAT}
    assert export.model_path == tmp_path / "file.model.nc"
    check_model_file(export.model_path, model)
    # layer n of the file is the mesh's layer of own attribute n + 1
    back = read_model(export.model_path)
    assert back.geometry.domain == domain
    assert [lay.name for lay in back.geometry.layers] == [e["name"]
                                                          for e in card.layers]
    assert np.array_equal(back.skeleton.boundaries, card.boundaries)
    # True takes a model that does not vary laterally, on the smallest grid
    with pytest.raises(ValueError, match=r"depends on \(x, y\).*give the grid"):
        export_mfem(res, tmp_path / "varies", model=model, model_file=True)
    assert not (tmp_path / "varies.model.nc").exists()
    layered = export_mfem(res, tmp_path / "layered", model=model, fields=["rho", "q"],
                          model_file=True)
    from planetmodel.cartesian.modelfile import read_model_file
    data = read_model_file(layered.model_path)
    assert data.parts[0].grid == cartesian.Grid(domain, 1)
    assert sc.read(layered.manifest_path).model["schema"] == SCHEMA_FLAT
    q = read_model(layered.model_path).layer(1)["q"]
    assert float(q.evaluate(0.5 * sum(q.interval), 0.0, 0.0)) == pytest.approx(
        sum(q.interval))


def test_what_the_export_of_a_flat_mesh_refuses(built, tmp_path):
    res = built("box")
    spherical = Model(Geometry(Skeleton(DEPTHS)),
                      [{"rho": spherical_constant(iv, 1.0, character=DENSITY)}
                       for iv in zip(DEPTHS[:-1], DEPTHS[1:])])
    with pytest.raises(ValueError, match="in the coordinates 'spherical' and the mesh"):
        export_mfem(res, tmp_path / "sphere", model=spherical)
    shifted = GEOMETRIES["rectangle"](mapping=ShiftedAlongY())
    off = build_layered_mesh(MeshSpec(shifted, SIZING, order=1), tmp_path / "shifted")
    with pytest.raises(ValueError, match="keeps y = 0"):
        export_mfem_mesh(off, tmp_path / "off")
    assert not (tmp_path / "off.mesh").exists()
