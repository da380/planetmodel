"""The cross-check: what MFEM makes of the file the mesher wrote.

Everything planetmodel validates about a mesh -- Jacobian signs, face
orientations, interface radii -- it validates with gmsh's own numbers,
which is the same authority that produced them.  The consumer is MFEM,
and MFEM has its own opinion about whether a tetrahedron is
positively oriented and whether a boundary face points outwards.  Here
that opinion is asked for directly: load the written `.msh` with
PyMFEM and read back `CheckElementOrientation` and
`CheckBdrElementOrientation`.  Plan Appendix A records why the return
values are the only honest measure -- MFEM's "wrong orientation"
messages are compiled out of release builds -- and that the boundary
check judges only the outermost surface, interior faces being skipped
in both 2D and 3D.

The body is the acceptance structure of `test_acceptance.py`, unit
sized and a few elements across: five shells and a vacuum buffer, with
relief on the Moho and on the surface.  Four configurations are loaded,
each as its own test so a failure names the one that broke.

Two relations are pinned beyond the orientation counts.  The manifest's
`mesh.n_elements` is what `element_counts` writes: the cells of the
mesh dimension *plus* the faces of dimension d-1, which are what carry
the physical groups into the file.  MFEM splits that same set in two,
cells into `GetNE` and faces into `GetNBE` -- every interior interface
becomes a boundary element too -- so `GetNE() + GetNBE()` recovers the
manifest's count exactly.  And the total volume of a mesh depends only
on its outermost boundary, whatever relief the interior carries, so it
is the analytic ball of the buffer radius: at order 1 that is a
straight-sided polyhedron and about a percent low, at order 2 it is
right to a few parts in a million.
"""
import dataclasses
import math

import numpy as np
import pytest

from planetmodel.io import manifest as sc
from planetmodel.mesh3d import build_layered_mesh

from .test_acceptance import acceptance_spec

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

#: (dimension, order, mode): order 1 and 2 in 3D, both delivery modes,
#: and the 2D disc, which exercises the same code with faces that are
#: curves rather than surfaces.
CASES = [(3, 1, "physical"), (3, 2, "physical"),
         (3, 2, "referential"), (2, 2, "physical")]


@pytest.fixture(scope="module")
def load(tmp_path_factory):
    """Build a case once, load it with MFEM, and keep both.

    Module scoped and cached: the relief check wants the same physical
    order-2 mesh the case test loaded, and building it twice would
    double the most expensive configuration for nothing.
    """
    root = tmp_path_factory.mktemp("mfem")
    cache: dict = {}

    def _load(dimension, order, mode):
        key = (dimension, order, mode)
        if key not in cache:
            spec = dataclasses.replace(acceptance_spec(order=order),
                                       dimension=dimension, delivery=mode)
            built = build_layered_mesh(
                spec, root / f"{dimension}d_o{order}_{mode}")
            cache[key] = (mfem.Mesh(str(built.msh_path)),
                          sc.read(built.manifest_path))
        return cache[key]

    return _load


@pytest.mark.parametrize("dimension,order,mode", CASES)
def test_mfem_reads_the_mesh_and_finds_nothing_to_fix(dimension, order, mode,
                                                      load):
    """MFEM's own checks, its own attributes, its own element count."""
    mesh, card = load(dimension, order, mode)

    assert mesh.CheckElementOrientation(False) == 0
    assert mesh.CheckBdrElementOrientation(False) == 0

    # The brief's contract is the numbering, not the names: layers and
    # interfaces are 1..N centre-outward, and the acceptance body has
    # six of each -- four shells, the crust, and the vacuum buffer.
    assert list(mesh.attributes.ToList()) == \
        list(range(1, len(card.layers) + 1))
    assert list(mesh.bdr_attributes.ToList()) == \
        list(range(1, len(card.interfaces) + 1))

    assert mesh.GetNE() == card.mesh["n_elements"] - mesh.GetNBE()

    if mode == "referential":
        # Order 2 only: the straight-sided ball of order 1 is a percent
        # low by construction, which is a statement about polyhedra.
        r = card.interfaces[-1]["mean_radius_nd"]
        volume = sum(mesh.GetElementVolume(i) for i in range(mesh.GetNE()))
        assert volume == pytest.approx(4.0 / 3.0 * math.pi * r ** 3, rel=1e-4)


def test_the_relief_reached_the_file_mfem_read(load):
    """The surface MFEM sees is the shaped one, not the sphere.

    Mode A displaces the nodes before writing, so the check is on the
    coordinates that came back out of the file: every vertex of a
    boundary element carrying the surface attribute lies within one and
    a half amplitudes of the reference radius -- the relief's own range
    is `[-1.5, 1]` amplitudes -- and the spread across them is at least
    an amplitude, which a sphere could not manage.
    """
    mesh, card = load(3, 2, "physical")
    surface = card.interface_attribute("surface")
    amplitude = 3.0e3 / card.model["rref_m"]

    xyz = np.asarray(mesh.GetVertexArray())
    on_surface = sorted({v for i in range(mesh.GetNBE())
                         if mesh.GetBdrAttribute(i) == surface
                         for v in mesh.GetBdrElementVertices(i)})
    assert on_surface
    r = np.linalg.norm(xyz[on_surface], axis=1)

    assert r.min() >= 1.0 - 1.5 * amplitude
    assert r.max() <= 1.0 + 1.5 * amplitude
    assert r.max() - r.min() > amplitude
