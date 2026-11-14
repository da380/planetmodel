"""The split-node method's quantities built from an MFEM delivery alone.

A split-node code takes the physical displacement as u = u_w + sum_k
chi_k s_k: u_w continuous, s_k fault k's slip, a field of the
displacement space nonzero on its slip dofs alone, and chi_k the
indicator of fault k's hanging wall.  Here each is built as such a code
builds it, from the mesh MFEM reads and the manifest, with no geometry
and no walk over neighbours: chi_k from the element attributes of the
fault's `hanging_wall`; the slip dofs of fault k the dofs of its
boundary elements but those of its pinned edges, found by counting the
uses of each edge by its boundary elements (an edge used once and not
on the surface's boundary elements is its tip, or its junction where it
ends on another fault, and pinned).  Then sum_k chi_k s_k, element by
element, must jump across each face of fault m by exactly s_m, the
hanging-wall side less the footwall side (the sides by the format
page's rule on the face's vertex order), at every node of the face,
and not at all across any other face.  Lagrange traces agree where
their nodal values do, so the check is exact at the dofs of each face.
It holds for every configuration the builder accepts, in spaces of
order 1 and 2, and on the mesh refined once by MFEM, its elements
inheriting their attributes.
"""
from collections import Counter

import numpy as np
import pytest

from planetmodel.mesh3d import manifest as sc
from planetmodel.mesh3d import export_mfem_mesh

from test_collars import CASES, build

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]


@pytest.fixture(scope="module")
def delivered(tmp_path_factory):
    """Each configuration built at order 2 and exported in physical
    delivery, once: (the export, its manifest)."""
    root = tmp_path_factory.mktemp("split_nodes")
    cache = {}

    def get(name):
        if name not in cache:
            res = build(root, name, order=2)
            export = export_mfem_mesh(res, root / f"{name}_mfem", delivery="physical")
            cache[name] = (export, sc.read(export.manifest_path))
        return cache[name]
    return get


def load(export):
    o = export.read_options
    return mfem.Mesh(str(export.mesh_path), o["generate_edges"], o["refine"],
                     o["fix_orientation"])


def split_nodes(mesh, card, order, rng):
    """The split-node quantities from the mesh and the manifest alone:
    the slips (F, ndof), the hanging walls' indicators (F, ne), and the
    fault faces as {face: (fault, boundary element)}."""
    fes = mfem.FiniteElementSpace(mesh, mfem.H1_FECollection(order, 3))
    battr = np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
    attrs = np.array([mesh.GetAttribute(e) for e in range(mesh.GetNE())])

    def edges(rows):
        return [e for b in rows for e in mesh.GetBdrElementEdges(int(b))[0]]

    on_surface = set(edges(np.flatnonzero(battr == card.surface_attribute)))
    slips, walls, faces = [], [], {}
    for k, record in enumerate(card.faults):
        rows = np.flatnonzero(battr == record["attribute"])
        count = Counter(edges(rows))
        pinned = {d for e, n in count.items() if n == 1 and e not in on_surface
                  for d in fes.GetEdgeDofs(e)}
        carrying = sorted({d for b in rows for d in fes.GetBdrElementDofs(int(b))}
                          - pinned)
        assert carrying, record["name"]
        s = np.zeros(fes.GetNDofs())
        s[carrying] = rng.uniform(1.0, 2.0, len(carrying))
        slips.append(s)
        walls.append(np.isin(attrs, record["hanging_wall"]).astype(float))
        faces.update({mesh.GetBdrElementFaceIndex(int(b)): (k, int(b)) for b in rows})
    return fes, np.stack(slips), np.stack(walls), faces


def jumps(mesh, card, order, *, seed=0):
    """The largest jump of sum_k chi_k s_k across a face of no fault, the
    largest departure of its jump across a fault's face from that fault's
    slip, and the number of fault faces seen: over the faces inside the
    body, the geometry's layers, where the displacement lives (a shell
    outside it is not split, and its faces on the geometry's surface are
    the body's boundary, where the trace is free)."""
    fes, slips, walls, faces = split_nodes(mesh, card, order,
                                           np.random.default_rng(seed))
    xyz = np.asarray(mesh.GetVertexArray())
    body = np.isin([mesh.GetAttribute(e) for e in range(mesh.GetNE())],
                   card.element_attributes(layers=[e["attribute"] for e in card.layers
                                                   if e["in_geometry"]]))
    off = on = 0.0
    seen = 0
    for f in range(mesh.GetNumFaces()):
        e1, e2 = mesh.GetFaceElements(f)
        if e2 < 0 or not (body[e1] and body[e2]):
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
    return off, on, seen


@pytest.mark.parametrize("order", [1, 2])
@pytest.mark.parametrize("name", list(CASES))
def test_chi_s_jumps_by_the_slip_on_the_faults_and_nowhere_else(delivered, name,
                                                                 order):
    export, card = delivered(name)
    mesh = load(export)
    off, on, seen = jumps(mesh, card, order)
    assert seen == sum(np.count_nonzero(
        np.array([mesh.GetBdrAttribute(b) for b in range(mesh.GetNBE())])
        == r["attribute"]) for r in card.faults)
    assert off == 0.0 and on == 0.0


@pytest.mark.parametrize("name", ["surface", "splay", "crossing"])
def test_and_on_the_mesh_mfem_refines(delivered, name):
    """MFEM's uniform refinement: the children inherit the element and
    boundary attributes, and each child of a fault face its parent's
    orientation (else the sides by the vertex order would disagree with
    the attributes); the pinned edges are counted again on the refined
    faces."""
    export, card = delivered(name)
    mesh = load(export)
    faults = [r["attribute"] for r in card.faults]
    before = sum(mesh.GetBdrAttribute(b) in faults for b in range(mesh.GetNBE()))
    mesh.UniformRefinement()
    off, on, seen = jumps(mesh, card, 2)
    assert off == 0.0 and on == 0.0
    assert seen == 4 * before


def test_a_model_s_fields_reach_the_collars(tmp_path):
    """`export_mfem` on a mesh with faults: every element holds its
    layer's value, a collar's elements among them, the layer read from
    the element's attribute through the manifest; and the record names
    the layers by their own attributes."""
    from planetmodel import LayeredIsotropicElastic
    from planetmodel.mesh3d import export_mfem

    res = build(tmp_path, "crossing")
    model = LayeredIsotropicElastic([0.0, 0.6, 1.0], rho=[10.0, 4.0], vp=[9.0, 6.0],
                                    vs=[0.0, 3.5])
    export = export_mfem(res, tmp_path / "fields", model=model, fields=["rho"])
    card = sc.read(export.manifest_path)
    assert card.schema == sc.SCHEMA_WITH_FAULTS
    assert card.field_record("rho")["layers"] == [1, 2]
    assert card.element_attributes(layers=[1, 2]) == (1, 2, 3, 4, 5)
    mesh = load(export)
    rho = mfem.GridFunction(mesh, str(export.field_paths["rho"]))
    fes = rho.FESpace()
    values = rho.GetDataArray()
    seen = set()
    for e in range(mesh.GetNE()):
        a = mesh.GetAttribute(e)
        want = {1: 10.0, 2: 4.0}[card.layer_of(a)["attribute"]]
        assert np.allclose(values[fes.GetElementDofs(e)], want, rtol=0, atol=1e-12), a
        seen.add(a)
    assert seen == {1, 2, 3, 4, 5}
