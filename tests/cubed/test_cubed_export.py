"""A cubed sphere exported for MFEM: both deliveries under a mapping, a
fold refused, the fields of a model at the dofs of every layer and
their integrals against the model's own, and the model's records in
the manifest."""
import numpy as np
import pytest
from scipy.integrate import quad

from cubed_cases import (COARSE, SPECS, ball, build, equatorial_stretch, flattening,
                         shell)
from planetmodel import PREM
from planetmodel.character import DENSITY, ELASTIC
from planetmodel.fields import AnalyticField, constant_field
from planetmodel.mesh1d import gravity
from planetmodel.mesh3d import (CubedSphereSpec, Shell, UniformInterfaces,
                                build_cubed_sphere, export_mfem, export_mfem_mesh,
                                manifest as sc, read_mesh)
from planetmodel.mesh3d.export import (MESH_READ_OPTIONS, jacobian_report,
                                       node_array)
from planetmodel.model import Model

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = pytest.mark.mfem

#: name -> (geometry factory, dimension, order, shells)
CASES = {
    "ball3": (lambda: ball().with_mapping(flattening(0.05)), 3, 2, ()),
    "shell3": (lambda: shell().with_mapping(flattening(0.05, rmax=1.3)), 3, 3,
               (Shell(ratio=0.3),)),
    "ball2": (lambda: ball().with_mapping(equatorial_stretch(0.05)), 2, 2, ()),
    "shell2": (shell, 2, 1, ()),
}


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every case built once."""
    root = tmp_path_factory.mktemp("cubed")
    out = {}
    for name, (make, dimension, order, shells) in CASES.items():
        spec = SPECS[dimension](make(), 4, sizing=COARSE, order=order,
                                shells=shells)
        out[name] = build(spec, root / name)
    return out


def load(path):
    """The mesh a consumer loads, constructed the way the manifest says."""
    o = MESH_READ_OPTIONS
    mesh = mfem.Mesh(str(path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])
    if mesh.GetNodes() is None:
        mesh.SetCurvature(1)
    return mesh


def nodes_of(path):
    """The nodes of a written mesh, copied while the mesh is alive."""
    mesh = load(path)
    return np.array(node_array(mesh.GetNodes()), dtype=float, copy=True)


@pytest.mark.parametrize("name", list(CASES))
def test_both_deliveries(built, name, tmp_path):
    res = built[name]
    d = res.spec.dimension
    X = nodes_of(res.mesh_path)
    X3 = np.zeros((X.shape[0], 3))
    X3[:, :d] = X
    moved = np.asarray(res.mapping(X3), dtype=float)[:, :d]
    identity = res.geometry.is_identity

    ref = export_mfem_mesh(res, tmp_path / "ref", delivery="referential")
    card = sc.read(ref.manifest_path)
    assert card.mesh["nodes"] == "reference" and card.mesh["format"] == "mfem"
    assert card.mesh["displacement"] == "displacement"
    mesh = load(ref.mesh_path)
    np.testing.assert_allclose(node_array(mesh.GetNodes()), X, atol=1e-15)
    gf = mfem.GridFunction(mesh, str(ref.displacement_path))
    np.testing.assert_allclose(node_array(gf), moved - X, atol=1e-14)
    assert ref.counts["elements"] == res.counts["elements"]
    assert ref.counts["boundary_elements"] == res.counts["boundary_elements"]
    assert ref.counts["order"] == res.spec.order

    phys = export_mfem_mesh(res, tmp_path / "phys", delivery="physical")
    card = sc.read(phys.manifest_path)
    assert card.mesh["nodes"] == ("reference" if identity else "physical")
    assert phys.displacement_path is None and card.mesh["displacement"] is None
    np.testing.assert_allclose(nodes_of(phys.mesh_path), moved, atol=1e-14)
    assert phys.quality["negative_jacobians"] == 0
    assert jacobian_report(load(phys.mesh_path))["negative_jacobians"] == 0
    assert [e["attribute"] for e in card.layers] == list(
        range(1, res.counts["layers"] + 1))
    if not identity:
        assert np.abs(moved - X).max() > 1e-3


@pytest.mark.parametrize("name", list(CASES))
def test_the_boundary_elements_face_as_mfem_holds_them(built, name, tmp_path):
    """An interface's boundary elements point away from the centre and a
    shell's inner boundary towards it, out of the body, in the reference
    mesh as in its deliveries."""
    res = built[name]
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    d = mesh.dimension

    def facing(mesh):
        out = {}
        for attribute in np.unique(mesh.face_attributes):
            p = mesh.points[mesh.faces_with(attribute)]
            if d == 3:
                normal = np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0])
            else:
                t = p[:, 1] - p[:, 0]
                normal = np.stack([t[:, 1], -t[:, 0], np.zeros(len(t))], axis=1)
            away = np.einsum("ij,ij->i", normal, p.mean(axis=1)) > 0.0
            out[int(attribute)] = (int(away.sum()), int((~away).sum()))
        return out

    written = facing(mesh)
    for entry in card.interfaces:
        away, towards = written[entry["attribute"]]
        inner = entry["between_layers"][0] == -1
        assert (away, towards)[0 if inner else 1] == 0 and away + towards > 0
    for delivery in ("physical", "referential"):
        export = export_mfem_mesh(res, tmp_path / delivery, delivery=delivery)
        assert facing(read_mesh(export.mesh_path)) == written


def test_a_delivery_may_not_replace_its_reference_mesh(built, tmp_path):
    res = built["shell2"]
    for base in (res.mesh_path, res.mesh_path.with_suffix("")):
        with pytest.raises(ValueError, match="another basename"):
            export_mfem_mesh(res, base)
    assert load(res.mesh_path).GetNE() == res.counts["elements"]


def test_a_mapping_that_folds_the_mesh_is_refused_at_export(tmp_path):
    """A flattening of 3 turns the poles inside out; the geometry's own
    check is bypassed so that the exporter's Jacobian check is what
    catches it."""
    g = ball().with_mapping(flattening(3.0), check=False)
    res = build_cubed_sphere(CubedSphereSpec(g, 4, sizing=COARSE), tmp_path / "fold")
    assert res.validation.ok
    with pytest.raises(ValueError, match="folds .* element"):
        export_mfem_mesh(res, tmp_path / "nope", delivery="physical")
    assert not (tmp_path / "nope.mesh").exists()
    ref = export_mfem_mesh(res, tmp_path / "ref", delivery="referential")
    assert ref.quality["negative_jacobians"] == 0


# ------------------------------------------------------------- the fields

def model_on(geometry):
    """A model whose density varies with radius and direction and jumps
    between layers, with an elastic tensor on every layer but the first."""
    layers = []
    for lay in geometry.layers:
        iv = lay.interval
        scale = 3.0 - lay.index
        fields = {"rho": AnalyticField(
            iv, lambda r, t, p, scale=scale: scale * (1.0 + r * r)
            + 0.1 * np.sin(t) * np.cos(p), character=DENSITY, name="rho")}
        if lay.index > 0:
            fields["elastic_moduli"] = constant_field(
                iv, (lay.index + 1.0) * np.eye(6), character=ELASTIC,
                name="elastic_moduli")
        layers.append(fields)
    return Model(geometry, layers)


def dof_points(mesh, fes):
    """Where every dof of `fes` sits, through MFEM's element transformations."""
    X = np.zeros((fes.GetNDofs(), mesh.SpaceDimension()))
    for e in range(mesh.GetNE()):
        pts = mfem.DenseMatrix()
        mesh.GetElementTransformation(e).Transform(fes.GetFE(e).GetNodes(), pts)
        X[np.asarray(fes.GetElementDofs(e), dtype=int)] = pts.GetDataArray().T
    return X


def dof_attributes(mesh, fes):
    """The attribute of the element each dof of `fes` belongs to."""
    a = np.zeros(fes.GetNDofs(), dtype=int)
    for e in range(mesh.GetNE()):
        a[np.asarray(fes.GetElementDofs(e), dtype=int)] = mesh.GetAttribute(e)
    return a


@pytest.mark.parametrize("name", ["ball3", "shell3", "ball2"])
def test_every_field_reads_back_at_the_dof_coordinates(built, name, tmp_path):
    res = built[name]
    model = model_on(res.geometry)
    export = export_mfem(res, tmp_path / name, model=model, delivery="referential")
    mesh = load(export.mesh_path)
    card = sc.read(export.manifest_path)
    assert set(export.field_paths) == {"rho", "elastic_moduli"}
    entries = {e["name"]: e for e in card.fields}
    shells = [e["attribute"] for e in card.layers if not e["in_geometry"]]
    for field, path in export.field_paths.items():
        gf = mfem.GridFunction(mesh, str(path))
        fes = gf.FESpace()
        assert fes.FEColl().Name() == entries[field]["fe_space"] == (
            f"L2_{mesh.Dimension()}D_P{res.spec.order}")
        values = np.array(node_array(gf), dtype=float)
        scalar = mfem.FiniteElementSpace(mesh, fes.FEColl(), 1)
        X = dof_points(mesh, scalar)
        X3 = np.zeros((X.shape[0], 3))
        X3[:, :X.shape[1]] = X
        attributes = dof_attributes(mesh, scalar)
        holders = entries[field]["layers"]
        for entry in card.layers:
            dofs = np.flatnonzero(attributes == entry["attribute"])
            assert dofs.size
            if entry["attribute"] not in holders:
                assert np.all(values[dofs] == 0.0)
                continue
            layer = model.layer(entry["attribute"] - 1)
            r = np.linalg.norm(X3[dofs], axis=1)
            pulled = X3[dofs] * (np.clip(r, *layer.interval) / r)[:, None]
            want = layer[field].evaluate_at(pulled, frame="cartesian")
            np.testing.assert_allclose(values[dofs], want.reshape(dofs.size, -1),
                                       atol=1e-12)
        assert set(shells).isdisjoint(holders)
    assert entries["rho"]["layers"] == [e["attribute"] for e in card.layers
                                        if e["in_geometry"]]
    assert entries["elastic_moduli"]["layers"] == entries["rho"]["layers"][1:]
    assert [e["kind"] for e in card.interfaces][-1] == ("outer" if shells else "free")


class AxisDistanceSquared(mfem.PyCoefficient):
    """x^2 + y^2: the weight of the moment of inertia about z."""

    def EvalValue(self, x):
        return x[0] ** 2 + x[1] ** 2


def integral(gf, *, weight=None, geometry_order=1):
    """The integral of a scalar L2 GridFunction over its mesh, times
    `weight` where given, by a rule that reaches the curved geometry."""
    fes = gf.FESpace()
    coefficient = mfem.GridFunctionCoefficient(gf)
    if weight is not None:
        coefficient = mfem.ProductCoefficient(coefficient, weight)
    form = mfem.LinearForm(fes)
    form.AddDomainIntegrator(mfem.DomainLFIntegrator(coefficient, 2,
                                                     3 * geometry_order + 2))
    form.Assemble()
    return float(np.sum(form.GetDataArray()))


def test_the_mass_and_the_moment_of_inertia_are_the_model_s(tmp_path):
    """The exported density, integrated over the mesh, against the radial
    integrals of the model: every layer has its own material."""
    geometry = ball()
    scales = [3.0, 2.0, 1.0]
    model = Model(geometry, [
        {"rho": AnalyticField(lay.interval, lambda r, t, p, c=c: c * (1.0 + r * r),
                              character=DENSITY, name="rho")}
        for lay, c in zip(geometry.layers, scales)])
    mass = sum(4.0 * np.pi * quad(lambda r, c=c: c * (1 + r * r) * r ** 2, *iv)[0]
               for c, iv in zip(scales, (lay.interval for lay in geometry.layers)))
    moment = sum(8.0 * np.pi / 3.0
                 * quad(lambda r, c=c: c * (1 + r * r) * r ** 4, *iv)[0]
                 for c, iv in zip(scales, (lay.interval for lay in geometry.layers)))
    spec = CubedSphereSpec(geometry, 4, sizing=COARSE, order=3,
                           shells=(Shell(ratio=0.2),))
    res = build_cubed_sphere(spec, tmp_path / "built")
    export = export_mfem(res, tmp_path / "out", model=model, order=4)
    mesh = load(export.mesh_path)
    rho = mfem.GridFunction(mesh, str(export.field_paths["rho"]))
    assert integral(rho, geometry_order=3) == pytest.approx(mass, rel=1e-6)
    weight = AxisDistanceSquared()
    assert integral(rho, weight=weight, geometry_order=3) == pytest.approx(
        moment, rel=1e-6)
    # one layer's cells given another's density would change it by far more
    assert abs(mass * (scales[1] / scales[2] - 1.0)) > 1e-2 * mass


def test_prem_on_a_cubed_sphere(tmp_path):
    """PREM's thirteen layers, each at least a cell thick: the mass of the
    exported density is the model's own, and the manifest carries what
    the model says of its layers and interfaces."""
    model = PREM().nondimensionalised()
    spec = CubedSphereSpec(model.geometry, 3, sizing=UniformInterfaces(0.1, 0.1, 0.1),
                           order=3, shells=(Shell(ratio=0.2),))
    res = build_cubed_sphere(spec, tmp_path / "prem")
    assert res.validation.ok and res.counts["layers"] == 14
    assert any("times longer" in w for w in res.validation.warnings)
    export = export_mfem(res, tmp_path / "prem_out", model=model, fields=["rho"],
                         order=3)
    mesh = load(export.mesh_path)
    rho = mfem.GridFunction(mesh, str(export.field_paths["rho"]))
    assert integral(rho, geometry_order=3) == pytest.approx(gravity.mass(model),
                                                            rel=1e-5)
    card = sc.read(export.manifest_path)
    fluid = {e["name"]: e["fluid"] for e in card.layers}
    assert fluid["outer_core"] and fluid["ocean"] and not fluid["inner_core"]
    assert fluid["buffer"] is False
    kinds = [e["kind"] for e in card.interfaces]
    assert kinds[0] == kinds[1] == "fluid-solid"
    assert kinds[-2] == "free" and kinds[-1] == "outer"
    assert card.scales["length"] == pytest.approx(6.371e6)
    (entry,) = card.fields
    assert entry["name"] == "rho" and entry["radial_degree"] == 3
    assert entry["layers"] == list(range(1, 14))


def test_the_model_file_beside_a_cubed_sphere(tmp_path):
    pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")
    model = PREM().nondimensionalised()
    spec = CubedSphereSpec(model.geometry, 2, sizing=UniformInterfaces(0.3, 0.3, 0.3))
    res = build_cubed_sphere(spec, tmp_path / "prem")
    export = export_mfem(res, tmp_path / "prem_out", model=model, fields=["rho"],
                         model_file=True)
    card = sc.read(export.manifest_path)
    assert export.model_path.exists() and card.model["file"] == export.model_path.name
