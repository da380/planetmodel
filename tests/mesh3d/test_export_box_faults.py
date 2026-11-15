"""A faulted box through the MFEM export, under the identity mapping.

MFEM reads the box's layers and collars as element attributes, its
interfaces, faces and faults as boundary attributes, and keeps the
vertex order of a fault's faces, so the plus cell of each is on the
fault's normal side; the format page's rule on MFEM's boundary elements,
the box's faces given, recovers the MSH file's curve groups, the face
edge among them.  A split-node code built from the delivery and the
manifest alone finds sum_k chi_k s_k jumping by exactly s_k across fault
k's faces and nowhere else, whether it lets a fault's face edge slip or
pins it, in spaces of order 1 and 2; and a flat model's fields reach the
collars' elements.
"""
from collections import Counter

import numpy as np
import pytest

from planetmodel import DENSITY, cartesian
from planetmodel.mesh3d import (export_mfem, export_mfem_mesh, manifest as sc,
                                read_mesh)

from test_box_faults import CASES, box, build

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

#: The configurations exported: a fault through a side, one through the
#: surface and a side, one through the base, two faults, and a splay.
EXPORTED = ["side", "surface_side", "base", "two", "splay"]


@pytest.fixture(scope="module")
def delivered(tmp_path_factory):
    """Each configuration built at order 2 and exported in both deliveries,
    once: (the build, {delivery: (the export, its manifest)})."""
    root = tmp_path_factory.mktemp("box_faults_mfem")
    cache = {}

    def get(name):
        if name not in cache:
            res = build(root, name)
            out = {}
            for delivery in ("physical", "referential"):
                export = export_mfem_mesh(res, root / f"{name}_{delivery}",
                                          delivery=delivery)
                out[delivery] = (export, sc.read(export.manifest_path))
            cache[name] = (res, out)
        return cache[name]
    return get


def load(export):
    o = export.read_options
    return mfem.Mesh(str(export.mesh_path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])


@pytest.mark.parametrize("name", EXPORTED)
def test_mfem_reads_the_attributes_and_the_sides(delivered, name):
    """Element attributes are the layers' and the collars', boundary
    attributes the interfaces', the faces' and the faults'; every fault
    face's plus cell, by its vertex order, lies on the fault's normal
    side by position; the manifest stays schema 7 in either delivery."""
    res, out = delivered(name)
    export, card = out["physical"]
    assert card.schema == sc.SCHEMA_FLAT
    assert card.mesh["nodes"] == "reference"     # the identity moves nothing
    assert out["referential"][1].faults == card.faults
    mesh = load(export)
    built = read_mesh(res.msh_path)
    assert mesh.GetNE() == len(built.cells) and mesh.GetNBE() == len(built.faces)
    attributes = {mesh.GetAttribute(e) for e in range(mesh.GetNE())}
    assert attributes == set(card.element_attributes())
    boundary = {mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())}
    assert boundary == ({e["attribute"] for e in card.interfaces}
                        | set(card.face_attributes)
                        | {e["attribute"] for e in card.faults})
    xyz = np.asarray(mesh.GetVertexArray())
    planes = CASES[name][0]
    for record in card.faults:
        p = planes[record["name"]]
        rows = [b for b in range(mesh.GetNBE())
                if mesh.GetBdrAttribute(b) == record["attribute"]]
        faces = np.array([mesh.GetBdrElementVertices(b) for b in rows])
        cells = np.array([[mesh.GetElementVertices(e) for e in
                           mesh.GetFaceElements(mesh.GetBdrElementFaceIndex(b))]
                          for b in rows])
        plus = sc.plus_cell(xyz, faces, cells)
        centre = xyz[cells[np.arange(len(rows)), plus]].mean(axis=1)
        assert np.all((centre - p.cartesian(0.0, 0.0)) @ p.normal(0.0, 0.0) > 0.0)


@pytest.mark.parametrize("name", EXPORTED)
def test_the_curves_are_recovered_from_mfem_s_boundary_elements(delivered, name):
    """The rule on MFEM's boundary elements, the box's faces given, gives
    the MSH file's curve groups, edge for edge, the face edge among them."""
    res, out = delivered(name)
    export, card = out["referential"]
    mesh = load(export)
    xyz = np.asarray(mesh.GetVertexArray())

    def key(a, b):
        return tuple(sorted((tuple(np.round(xyz[a], 10)), tuple(np.round(xyz[b], 10)))))

    faces = {}
    for b in range(mesh.GetNBE()):
        faces.setdefault(mesh.GetBdrAttribute(b), []).append(
            list(mesh.GetBdrElementVertices(b)))
    recovered = sc.fault_curves(
        [faces[r["attribute"]] for r in card.faults], faces[card.surface_attribute],
        faces=[f for a in card.face_attributes for f in faces[a]])
    built = read_mesh(res.msh_path)
    at = np.round(built.points, 10)
    lines = {group: {tuple(sorted((tuple(at[a]), tuple(at[b]))))
                     for a, b in edges[:, :2].tolist()}
             for group, edges in built.curves.items()}
    for record, curves in zip(card.faults, recovered):
        for kind in card.fault_curves:
            edges = {key(a, b) for a, b in getattr(curves, kind)}
            group = record[kind]
            assert lines.get(group, set()) == edges, (record["name"], kind)
            assert (group is None) == (not edges)


def split_nodes(mesh, card, order, rng, *, pin_faces):
    """A split-node code's quantities from the mesh and the manifest alone
    (`test_split_nodes`), a fault's face edge, the edges it counts once on
    the box's faces, left to slip or pinned as `pin_faces` says."""
    fes = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(order, 3))
    battr = np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
    attrs = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])

    def edges(rows):
        return [e for b in rows for e in mesh.GetBdrElementEdges(int(b))[0]]

    on_surface = set(edges(np.flatnonzero(battr == card.surface_attribute)))
    on_faces = set(edges(np.flatnonzero(np.isin(battr, card.face_attributes))))
    slips, walls, faces = [], [], {}
    for k, record in enumerate(card.faults):
        rows = np.flatnonzero(battr == record["attribute"])
        count = Counter(edges(rows))
        free = on_surface | (set() if pin_faces else on_faces)
        pinned = {d for e, n in count.items() if n == 1 and e not in free
                  for d in fes.GetEdgeDofs(e)}
        carrying = sorted({d for b in rows for d in fes.GetBdrElementDofs(int(b))}
                          - pinned)
        s = np.zeros(fes.GetNDofs())
        s[carrying] = rng.uniform(1.0, 2.0, len(carrying))
        slips.append(s)
        walls.append(np.isin(attrs, record["hanging_wall"]).astype(float))
        faces.update({mesh.GetBdrElementFaceIndex(int(b)): (k, int(b)) for b in rows})
    return fes, np.stack(slips), np.stack(walls), faces


@pytest.mark.parametrize("pin_faces", [False, True])
@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("name", EXPORTED)
def test_chi_s_jumps_by_the_slip_on_the_faults_and_nowhere_else(delivered, name,
                                                                 order, pin_faces):
    """sum_k chi_k s_k jumps by exactly s_k across fault k's faces and by
    zero across every other face inside the box, the face edge slipping or
    pinned: the collars serve either choice."""
    _, out = delivered(name)
    export, card = out["physical"]
    mesh = load(export)
    fes, slips, walls, faces = split_nodes(mesh, card, order,
                                           np.random.default_rng(0),
                                           pin_faces=pin_faces)
    xyz = np.asarray(mesh.GetVertexArray())
    off = on = 0.0
    seen = 0
    for f in range(mesh.GetNumFaces()):
        e1, e2 = mesh.GetFaceElements(f)
        if e2 < 0:
            continue
        dofs = fes.GetFaceDofs(f)
        jump = (walls[:, e1] - walls[:, e2]) @ slips[:, dofs]
        if f in faces:
            k, b = faces[f]
            side = sc.fault_side(xyz, mesh.GetBdrElementVertices(b),
                                 mesh.GetElementVertices(e1))
            on = max(on, float(np.abs(jump - side * slips[k, dofs]).max()))
            seen += 1
        else:
            off = max(off, float(np.abs(jump).max()))
    assert seen == len(faces) and off == 0.0 and on == 0.0
    # a face edge left to slip carries slip at the box's faces
    if not pin_faces and any(r["face_edge"] is not None for r in card.faults):
        battr = np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
        sides = {d for b in np.flatnonzero(np.isin(battr, card.face_attributes))
                 for d in fes.GetBdrElementDofs(int(b))}
        assert np.any(slips[:, sorted(sides)] != 0.0)


def test_a_flat_model_s_fields_reach_the_collars(tmp_path):
    """`export_mfem` on a faulted box: every element holds its layer's
    value, a collar's among them, the layer read from the element's
    attribute through the manifest."""
    res = build(tmp_path, "crossing")
    g = box()
    layers = [{"rho": cartesian.constant_field(lay.interval, value, character=DENSITY)}
              for lay, value in zip(g.layers, (2.5, 3.3))]
    model = cartesian.Model(g, layers)
    export = export_mfem(res, tmp_path / "fields", model=model, fields=["rho"])
    card = sc.read(export.manifest_path)
    assert card.schema == sc.SCHEMA_FLAT
    assert card.element_attributes(layers=[1, 2]) == (1, 3, 2, 4)
    mesh = load(export)
    rho = mfem.GridFunction(mesh, str(export.field_paths["rho"]))
    fes = rho.FESpace()
    values = rho.GetDataArray()
    seen = set()
    for e in range(mesh.GetNE()):
        a = mesh.GetAttribute(e)
        want = {1: 2.5, 2: 3.3}[card.layer_of(a)["attribute"]]
        assert np.allclose(values[fes.GetElementDofs(e)], want, rtol=0, atol=1e-12)
        seen.add(a)
    assert seen == {1, 2, 3, 4}
    assert card.faults[0]["layers"] == [1, 2]
