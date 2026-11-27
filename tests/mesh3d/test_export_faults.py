"""Faults through the MFEM export: interior boundary elements, their
attributes, their orientation, and the sides a consumer recovers.

MFEM reads a fault's faces as boundary elements of the fault's
attribute and, reading an interior boundary element, neither checks nor
changes the order of its vertices; so (v1 - v0) x (v2 - v0) of each,
in the mesh MFEM holds, still points from the footwall into the hanging
wall, and of the two elements sharing its face the plus (hanging-wall)
one is the one on that side.  The curved nodes of the physical delivery
lie on the true fault to the pull-back's offset.  The main fault is
sized by a number (`EmbeddedFault`'s rule), the splay is not.
"""
import numpy as np
import pytest

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                build_layered_mesh, export_mfem_mesh,
                                manifest as sc, read_mesh)

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

UNIT = GeographicFrame(1.0)
ZERO = Slip(np.zeros((1, 1, 3)))


def bump():
    def h(r, theta, phi):
        c = (np.cos(theta) * np.cos(np.deg2rad(60.0)) + np.sin(theta)
             * np.sin(np.deg2rad(60.0)) * np.cos(phi - np.deg2rad(40.0)))
        return 0.05 * r ** 2 * np.exp(-(1.0 - c) / 0.05)
    return RadialStretch(CallableDisplacement(h), rmax=1.0)


def splayed():
    """A main fault and a splay rising from its centre line, by name."""
    main = Plane.at_depth(UNIT, 30.0, 40.0, 0.2, strike=0.0, dip=60.0, length=0.4,
                          width=0.3, slip=ZERO)
    foot = main.cartesian(0.0, -0.1)
    sp = Plane(UNIT, UNIT.coordinates(foot), strike=0.0, dip=30.0, length=0.3,
               width=0.15, slip=ZERO)
    sp = Plane(UNIT, UNIT.coordinates(sp.cartesian(0.0, 0.0) + foot
                                      - sp.cartesian(0.0, 0.5)),
               strike=0.0, dip=30.0, length=0.3, width=0.15, slip=ZERO)
    return {"main": Fault([main]), "splay": Fault([sp])}


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    root = tmp_path_factory.mktemp("faults_mfem")
    faults = splayed()
    g = Geometry(Skeleton([0.0, 0.6, 1.0])).with_mapping(bump())
    res = build_layered_mesh(MeshSpec(g, UniformInterfaces(0.15, 0.3, 0.3), order=2,
                                      faults=[EmbeddedFault(f, name=n, sizing=size)
                                              for (n, f), size
                                              in zip(faults.items(), (0.08, None))]),
                             root / "built")
    out = {d: export_mfem_mesh(res, root / d, delivery=d)
           for d in ("physical", "referential")}
    return res, faults, out


def load(export):
    o = export.read_options
    return mfem.Mesh(str(export.mesh_path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])


def corners(mesh, be):
    """The corners of a boundary element, in its own order."""
    xyz = np.asarray(mesh.GetVertexArray())
    return xyz[np.asarray(mesh.GetBdrElementVertices(be))]


@pytest.mark.parametrize("delivery", ["physical", "referential"])
def test_the_faults_are_boundary_elements_of_their_attributes(exported, delivery):
    res, _, out = exported
    built = sc.read(res.manifest_path)
    card = sc.read(out[delivery].manifest_path)
    assert card.faults == built.faults
    mesh = load(out[delivery])
    for what in ("elements", "boundary_elements", "nodes"):
        assert out[delivery].counts[what] == res.counts[what]
    attributes = np.array([mesh.GetBdrAttribute(i) for i in range(mesh.GetNBE())])
    for record in card.faults:
        assert np.count_nonzero(attributes == record["attribute"]) > 0
    assert set(attributes) == {1, 2, 3, 4}


def test_the_sides_are_recovered_from_the_vertex_order(exported):
    """Physical delivery: each fault face's normal is the fault's, mapped;
    the format page's rule (`manifest.plus_cell`) on the vertices MFEM
    holds finds one of its two elements on each side, and the plus one's
    centre is on the hanging-wall side of the fault, the minus one's on
    the footwall's."""
    res, faults, out = exported
    card = sc.read(out["physical"].manifest_path)
    mesh = load(out["physical"])
    xyz = np.asarray(mesh.GetVertexArray())
    for record in card.faults:
        fault = faults[record["name"]]
        faces, pairs = [], []
        for be in range(mesh.GetNBE()):
            if mesh.GetBdrAttribute(be) != record["attribute"]:
                continue
            v = corners(mesh, be)
            n = np.cross(v[1] - v[0], v[2] - v[0])
            where = fault.locate(v.mean(axis=0)[None], strict=False)
            assert n @ fault.patches[0].normal(where.u, where.v)[0] > 0.0
            e1, e2 = mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(be))
            assert e1 >= 0 and e2 >= 0, "a fault face is interior"
            faces.append(list(mesh.GetBdrElementVertices(be)))
            pairs.append([list(mesh.GetElementVertices(e)) for e in (e1, e2)])
        faces, pairs = np.array(faces), np.array(pairs)
        plus = sc.plus_cell(xyz, faces, pairs)
        rows = np.arange(len(faces))
        centres = xyz[pairs].mean(axis=2)
        ahead = fault.locate(centres[rows, plus], strict=False).distance
        behind = fault.locate(centres[rows, 1 - plus], strict=False).distance
        assert np.all(ahead > 0.0) and np.all(behind < 0.0)
        assert len(faces) == len(_triangles(res, record["attribute"]))


def test_the_curved_nodes_of_the_physical_delivery_lie_on_the_fault(exported):
    res, faults, out = exported
    card = sc.read(out["physical"].manifest_path)
    mesh = load(out["physical"])
    fes = mesh.GetNodes().FESpace()
    assert fes.GetOrder(0) == 2
    for record in card.faults:
        points = []
        for be in range(mesh.GetNBE()):
            if mesh.GetBdrAttribute(be) != record["attribute"]:
                continue
            pts = mfem.DenseMatrix()
            mesh.GetBdrElementTransformation(be).Transform(fes.GetBE(be).GetNodes(),
                                                           pts)
            points.append(pts.GetDataArray().T.copy())
        points = np.concatenate(points)
        found = faults[record["name"]].locate(points, strict=False)
        # measured: 9e-8 and 2.5e-8 for offsets of 3.4e-7 and 2.4e-8, on
        # elements of 0.15; the nodes of the junction sit on OCC's curve
        # crossing two B-splines, a little beyond the sampled offset
        assert np.abs(found.distance).max() <= 2.0 * record["offset"] + 1e-12


def test_the_curves_are_recovered_from_mfem_s_boundary_elements(exported):
    """The format page's rule on the boundary elements MFEM holds gives
    the curve groups of the MSH file, edge for edge (the referential
    delivery keeps the reference coordinates both files share)."""
    res, _, out = exported
    card = sc.read(out["referential"].manifest_path)
    mesh = load(out["referential"])
    xyz = np.asarray(mesh.GetVertexArray())

    def key(a, b):
        return tuple(sorted((tuple(np.round(xyz[a], 10)), tuple(np.round(xyz[b], 10)))))

    faces = {}
    for be in range(mesh.GetNBE()):
        faces.setdefault(mesh.GetBdrAttribute(be), []).append(
            list(mesh.GetBdrElementVertices(be)))
    recovered = sc.fault_curves([faces[r["attribute"]] for r in card.faults],
                                faces[card.surface_attribute])
    built = read_mesh(res.mesh_path)
    at = np.round(built.points, 10)
    lines = {group: {tuple(sorted((tuple(at[a]), tuple(at[b]))))
                     for a, b in edges[:, :2].tolist()}
             for group, edges in built.curves.items()}
    for record, curves in zip(card.faults, recovered):
        for name in ("tip", "free_edge", "junction"):
            edges_ = {key(a, b) for a, b in getattr(curves, name)}
            group = record[name]
            assert lines.get(group, set()) == edges_, (record["name"], name)
            assert (group is None) == (not edges_)


def _triangles(res, attribute):
    return read_mesh(res.mesh_path).faces_with(attribute)
