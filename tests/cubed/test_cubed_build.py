"""A cubed sphere built: read back by MFEM as a consumer reads it, its
attributes, its interfaces on their spheres, its layers' volumes
converging with the order, its manifest, and what is refused."""
import subprocess
import sys

import numpy as np
import pytest

from cubed_cases import COARSE, ball, flattening, shell
from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.mesh3d import (CubedSphereSpec, Shell, UniformInterfaces,
                                build_cubed_sphere, manifest as sc)
from planetmodel.mesh3d import cubed
from planetmodel.mesh3d.cubed import cell_volumes, volume_tolerance
from planetmodel.mesh3d.export import (MESH_READ_OPTIONS, jacobian_report,
                                       node_array)

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = pytest.mark.mfem

#: name -> (geometry factory, dimension, order, shells)
CASES = {
    "ball3": (ball, 3, 2, ()),
    "shell3": (shell, 3, 2, (Shell(ratio=0.3),)),
    "ball2": (ball, 2, 3, (Shell(ratio=0.3),)),
    "shell2": (shell, 2, 1, ()),
    "straight3": (ball, 3, 1, ()),
}
EDGE = 4


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Every case built once."""
    root = tmp_path_factory.mktemp("cubed")
    out = {}
    for name, (make, dimension, order, shells) in CASES.items():
        spec = CubedSphereSpec(make(), EDGE, sizing=COARSE, dimension=dimension,
                               order=order, shells=shells)
        out[name] = build_cubed_sphere(spec, root / name)
    return out


def load(path):
    """The mesh a consumer loads, constructed the way the manifest says."""
    o = MESH_READ_OPTIONS
    return mfem.Mesh(str(path), o["generate_edges"], o["refine"], o["fix_orientation"])


def element_radii(mesh):
    """The least and greatest radius of each element's nodes."""
    nodes = mesh.GetNodes()
    fes = nodes.FESpace()
    X = np.array(node_array(nodes), dtype=float)
    r = np.linalg.norm(X, axis=1)
    lo, hi = np.empty(mesh.GetNE()), np.empty(mesh.GetNE())
    for e in range(mesh.GetNE()):
        dofs = np.asarray(fes.GetElementDofs(e), dtype=int)
        lo[e], hi[e] = r[dofs].min(), r[dofs].max()
    return lo, hi


@pytest.mark.parametrize("name", list(CASES))
def test_the_mesh_as_a_consumer_reads_it(built, name):
    res = built[name]
    spec = res.spec
    card = sc.read(res.manifest_path)
    assert res.mesh_path.suffix == ".mesh" and card.mesh["file"] == res.mesh_path.name
    assert card.mesh["format"] == "mfem" and card.mesh["nodes"] == "reference"
    assert card.mesh["read_options"] == MESH_READ_OPTIONS
    assert card.schema == sc.SCHEMA and not card.refinements and not card.fields
    mesh = load(res.mesh_path)
    assert mesh.CheckElementOrientation(False) == 0
    assert mesh.CheckBdrElementOrientation(False) == 0
    assert mesh.Dimension() == mesh.SpaceDimension() == spec.dimension
    assert mesh.GetNE() == spec.cells == res.counts["elements"]
    assert mesh.GetNodes().FESpace().GetOrder(0) == spec.order == res.counts["order"]
    assert mesh.GetNodes().FESpace().GetNDofs() == res.counts["nodes"]
    assert mesh.Conforming()
    assert jacobian_report(mesh)["negative_jacobians"] == 0
    # every element lies in the layer its attribute names
    attributes = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])
    assert sorted(set(attributes.tolist())) == [e["attribute"] for e in card.layers]
    lo, hi = element_radii(mesh)
    for entry in card.layers:
        mine = attributes == entry["attribute"]
        assert lo[mine].min() >= entry["r_inner"] - 1e-13
        assert hi[mine].max() <= entry["r_outer"] + 1e-13
    names = [lay.name for lay in spec.layers]
    assert [e["name"] for e in card.layers] == names
    assert [e["in_geometry"] for e in card.layers] == (
        [True] * spec.geometry.nlayers + [False] * len(spec.shells))


@pytest.mark.parametrize("name", list(CASES))
def test_the_interfaces_are_spheres_of_the_mesh(built, name):
    res = built[name]
    card = sc.read(res.manifest_path)
    mesh = load(res.mesh_path)
    d = mesh.Dimension()
    nodes = mesh.GetNodes()
    fes = nodes.FESpace()
    X = np.array(node_array(nodes), dtype=float)
    block = 2 * d * EDGE ** (d - 1)
    counts = np.bincount([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
    assert counts[1:].tolist() == [block] * len(card.interfaces)
    for b in range(mesh.GetNBE()):
        entry = card.interfaces[mesh.GetBdrAttribute(b) - 1]
        dofs = np.asarray(fes.GetBdrElementDofs(b), dtype=int)
        r = np.linalg.norm(X[dofs], axis=1)
        # every node of the face, not its corners alone, is on the sphere
        np.testing.assert_allclose(r, entry["radius"], rtol=1e-14)
        corners = np.array([mesh.GetVertexArray(v)
                            for v in mesh.GetBdrElementVertices(b)])
        centre = X[dofs].mean(axis=0)
        if d == 3:
            normal = np.cross(corners[1] - corners[0], corners[3] - corners[0])
            outward = float(normal @ centre)
        else:
            side = corners[1] - corners[0]
            outward = float(centre[0] * side[1] - centre[1] * side[0])
        inner = entry["between_layers"][0] == -1
        assert (outward < 0.0) if inner else (outward > 0.0)
    assert [e["between_layers"] for e in card.interfaces][-1][1] == -1


@pytest.mark.parametrize("name", list(CASES))
def test_the_report(built, name):
    res = built[name]
    v = res.validation
    assert v.ok and not v.failures and not v.warnings
    assert v.negative_cells == v.negative_jacobians == v.inward_faces == 0
    assert 0.3 < v.min_sicn <= 1.0
    assert v.max_interface_radius_error < 1e-15
    assert v.max_layer_volume_error < volume_tolerance(EDGE)
    assert 1.0 <= v.max_aspect_ratio < 10.0
    domain = res.spec.domain
    assert v.group_counts == {"layers": domain.nlayers,
                              "interfaces": len(domain.interfaces)}
    assert res.counts["layers"] == domain.nlayers
    assert res.counts["levels"] == res.spec.levels
    assert set(res.timings) == {"resolve", "cells", "curve", "validate", "write"}
    assert "elements" in res.summary() and res.mapping is domain.mapping


def exact_volumes(boundaries, d):
    b = np.asarray(boundaries, dtype=float)
    if d == 3:
        return 4.0 / 3.0 * np.pi * np.diff(b ** 3)
    return np.pi * np.diff(b ** 2)


@pytest.mark.parametrize(("make", "d"), [(ball, 3), (shell, 3), (ball, 2), (shell, 2)])
def test_the_layers_volumes_converge_with_the_order(make, d, tmp_path):
    errors = []
    for order in (1, 2, 3, 4):
        spec = CubedSphereSpec(make(), EDGE, sizing=COARSE, dimension=d, order=order)
        res = build_cubed_sphere(spec, tmp_path / f"o{order}")
        mesh = load(res.mesh_path)
        if mesh.GetNodes() is None:
            mesh.SetCurvature(1)
        attributes = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])
        volumes = np.bincount(attributes, weights=cell_volumes(mesh))[1:]
        exact = exact_volumes(spec.domain.skeleton.boundaries, d)
        error = np.abs(volumes / exact - 1.0)
        assert error.max() == pytest.approx(res.validation.max_layer_volume_error,
                                            rel=1e-6, abs=1e-14)
        errors.append(error.max())
    # a straight-sided mesh falls short by under the check's bound, and
    # each order gains at least a factor of thirty
    assert 0.2 * volume_tolerance(EDGE) < errors[0] < volume_tolerance(EDGE)
    for coarse, fine in zip(errors[:-1], errors[1:]):
        assert fine < coarse / 30.0
    assert errors[-1] < 1e-8


def test_the_volume_of_a_cell():
    """A box of 2 x 3 x 4 cells of sides 1, 1/2 and 1/4, and a square's."""
    mesh = mfem.Mesh.MakeCartesian3D(2, 3, 4, mfem.Element.HEXAHEDRON, 2.0, 1.5, 1.0)
    for order in (1, 2, 3):
        mesh.SetCurvature(order)
        np.testing.assert_allclose(cell_volumes(mesh), 0.125, rtol=1e-12)
    square = mfem.Mesh.MakeCartesian2D(2, 2, mfem.Element.QUADRILATERAL, False,
                                       2.0, 1.0)
    square.SetCurvature(2)
    np.testing.assert_allclose(cell_volumes(square), 0.5, rtol=1e-12)


def test_a_folded_cell_fails(tmp_path, monkeypatch):
    """A map that turns the cells by the surface inside out is caught by
    the Jacobian, at the nodes of the curved cell."""
    def folded(points, **kwargs):
        x = real(points, **kwargs)
        r = np.linalg.norm(x, axis=1, keepdims=True)
        return np.where(r > 0.9, x * (1.8 - r) / np.maximum(r, 1e-300), x)
    real = cubed.cubed_map
    monkeypatch.setattr(cubed, "cubed_map", folded)
    spec = CubedSphereSpec(ball(), 4, sizing=COARSE, validate=False)
    report = build_cubed_sphere(spec, tmp_path / "folded").validation
    assert not report.ok and report.negative_jacobians > 0
    assert any("cells fold" in failure for failure in report.failures)


def test_a_thin_layer_is_built_and_warned_of(tmp_path):
    thin = Geometry(Skeleton([0.0, 0.5, 0.99, 1.0]))
    res = build_cubed_sphere(CubedSphereSpec(thin, 4, sizing=COARSE), tmp_path / "thin")
    assert res.validation.ok and res.validation.max_aspect_ratio > 10.0
    (warning,) = res.validation.warnings
    assert "times longer one way than another" in warning


def test_a_failing_mesh_is_refused_or_written_with_its_failures(tmp_path, monkeypatch):
    monkeypatch.setattr(cubed, "volume_tolerance", lambda edge: 1e-9)
    spec = CubedSphereSpec(ball(), 4, sizing=COARSE, order=1)
    with pytest.raises(ValueError, match="failed validation(.|\n)*layer 1 has volume"):
        build_cubed_sphere(spec, tmp_path / "bad")
    assert not (tmp_path / "bad.mesh").exists()
    assert not (tmp_path / "bad.json").exists()
    kept = CubedSphereSpec(ball(), 4, sizing=COARSE, order=1, validate=False)
    res = build_cubed_sphere(kept, tmp_path / "kept")
    assert not res.validation.ok and len(res.validation.failures) == 3
    assert res.mesh_path.exists()


def test_the_mapping_must_cover_the_shells(tmp_path):
    """A mapping that turns the shell inside out is refused before anything
    is built, and a spherical outer boundary refuses one that moves it."""
    def h(r, theta, phi):
        return np.where(r > 1.0, -1.5 * (r - 1.0), 0.0)
    folding = RadialStretch(CallableDisplacement(h, knots=[1.0], name="fold"),
                            rmax=2.0)
    spec = CubedSphereSpec(ball().with_mapping(folding), 4, sizing=COARSE,
                           shells=(Shell(ratio=0.5),))
    with pytest.raises(ValueError, match="defined and orientation-preserving"):
        build_cubed_sphere(spec, tmp_path / "shells")
    stretched = ball().with_mapping(flattening(0.05))
    round_ = CubedSphereSpec(stretched, 4, sizing=COARSE, outer_boundary="spherical")
    with pytest.raises(ValueError, match="identity on the outer boundary"):
        build_cubed_sphere(round_, tmp_path / "round")
    assert not list(tmp_path.iterdir())
    free = CubedSphereSpec(stretched, 4, sizing=COARSE, shells=(Shell(ratio=0.5),))
    res = build_cubed_sphere(free, tmp_path / "free")
    assert res.validation.ok and res.mapping is stretched.mapping
    assert sc.read(res.manifest_path).outer_radius == pytest.approx(1.5)


def test_one_cell_an_edge_and_a_high_order(tmp_path):
    spec = CubedSphereSpec(Geometry(Skeleton([0.0, 1.0])), 1,
                           sizing=UniformInterfaces(1.0, 1.0, 1.0), order=6,
                           aspect=None)
    res = build_cubed_sphere(spec, tmp_path / "one")
    assert res.counts["elements"] == 7 and res.validation.ok
    assert res.validation.max_layer_volume_error < 1e-4


def test_the_meta_reaches_the_manifest(tmp_path):
    spec = CubedSphereSpec(shell(), 2, knots=[0.5, 0.8, 1.0],
                           meta={"run": "a", "n": 3})
    card = sc.read(build_cubed_sphere(spec, tmp_path / "meta").manifest_path)
    assert card.meta == {"run": "a", "n": 3}
    assert [e["radius"] for e in card.interfaces] == [0.5, 0.8, 1.0]


def test_a_cubed_sphere_needs_no_gmsh(tmp_path):
    code = f"""
import sys
sys.modules["gmsh"] = None
from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.mesh3d import (CubedSphereSpec, UniformInterfaces,
                                build_cubed_sphere, export_mfem_mesh)
spec = CubedSphereSpec(Geometry(Skeleton([0.0, 0.5, 1.0])), 2,
                       sizing=UniformInterfaces(0.3, 0.3, 0.3))
res = build_cubed_sphere(spec, {str(tmp_path / "free")!r})
export_mfem_mesh(res, {str(tmp_path / "delivery")!r})
assert sys.modules["gmsh"] is None
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
    assert (tmp_path / "delivery.mesh").exists()
