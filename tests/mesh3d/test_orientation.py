"""Orientation, and the curving that can fold an element on its own.

The requirement is that a mesh loads into MFEM with no "wrong
orientation (fixed)" messages.  Two things stand between here and
there, and only one of them was expected.
"""
import pytest

import gmsh

from planetmodel.mesh3d._geometry import build_concentric
from planetmodel.mesh3d._orient import (element_quality, node_positions,
                                   orient_mesh, outward_dots, raise_order,
                                   signed_measures)
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._sizing import apply_mesh_options, apply_size_fields
from planetmodel.mesh3d._tagging import apply_physical_groups, identify
from planetmodel.mesh3d._writer import read_groups, write_msh
from planetmodel.mesh3d.spec import InterfaceSizing

pytestmark = pytest.mark.gmsh

RADII = (0.2, 0.55, 1.0)


def mesh_shells(dimension, size=0.15):
    g = build_concentric(RADII, dimension=dimension)
    t = identify(g, RADII)
    apply_physical_groups(t)
    apply_size_fields(t, {i: InterfaceSizing(size, size * 2, 0.35)
                          for i in range(len(RADII))})
    apply_mesh_options(order=1, algorithm_2d=6, algorithm_3d=1,
                       size_min=size, size_max=size * 2)
    gmsh.model.mesh.generate(dimension)
    return t


def negative_cells(dimension, pos=None):
    pos = node_positions() if pos is None else pos
    return sum(int((signed_measures(dimension, tag, pos)[1] < 0.0).sum())
               for _, tag in gmsh.model.getEntities(dimension))


def inward_faces(pos=None):
    pos = node_positions() if pos is None else pos
    return sum(int((outward_dots(tag, pos)[1] < 0.0).sum())
               for _, tag in gmsh.model.getEntities(2))


# ---------------------------------------------------------- orientation

@pytest.mark.parametrize("dimension", [2, 3])
def test_cells_come_out_positively_oriented(dimension):
    """gmsh 4.15 gets this right unaided -- checked, not assumed."""
    with session(name="cells"):
        mesh_shells(dimension)
        assert negative_cells(dimension) == 0


def test_occ_orients_an_interior_interface_inward():
    """The finding that makes boundary repair necessary.

    An interface bounding shells on both sides gets its orientation
    from OCC's own convention, and for the middle of three shells that
    points at the origin -- consistently, all 604 faces of it in the
    measured case, so it is a surface-level convention rather than
    scattered noise.
    """
    with session(name="inward"):
        mesh_shells(3)
        pos = node_positions()
        per_surface = {tag: int((outward_dots(tag, pos)[1] < 0.0).sum())
                       for _, tag in gmsh.model.getEntities(2)}
        totals = {tag: outward_dots(tag, pos)[1].size
                  for _, tag in gmsh.model.getEntities(2)}
        assert sum(per_surface.values()) > 0, "expected at least one flipped face"
        for tag, bad in per_surface.items():
            assert bad in (0, totals[tag]), (
                f"surface {tag} is inconsistently oriented ({bad} of "
                f"{totals[tag]}), which the per-surface story does not predict")


def test_orientation_repair_leaves_nothing_inward():
    with session(name="repair"):
        mesh_shells(3)
        report = orient_mesh(3)
        assert report.faces_flipped > 0
        assert inward_faces() == 0
        assert negative_cells(3) == 0


def test_repair_is_idempotent():
    with session(name="twice"):
        mesh_shells(3)
        first = orient_mesh(3)
        second = orient_mesh(3)
        assert first.faces_flipped > 0
        assert second.clean, "a second pass should find nothing to do"


@pytest.mark.parametrize("dimension", [2, 3])
def test_orientation_survives_the_msh_round_trip(dimension, tmp_path):
    with session(name="write"):
        mesh_shells(dimension)
        orient_mesh(dimension)
        path = write_msh(tmp_path / "o")
    with session(name="read"):
        read_groups(path)
        assert negative_cells(dimension) == 0
        if dimension == 3:
            assert inward_faces() == 0


# ------------------------------------------------- curving folds elements

def test_raising_the_order_can_fold_an_element_by_itself():
    """The unexpected half, and the reason raise_order exists.

    setOrder moves the new nodes onto the CAD surface.  Where an
    element is large compared with the local curvature that can invert
    it -- with no topography applied and nothing wrong with the
    straight mesh.  Measured here at order 2 on a three-shell sphere:
    one element in ~4000, sitting on a curved interface.
    """
    with session(name="fold"):
        mesh_shells(3)
        orient_mesh(3)
        gmsh.model.mesh.setOrder(2)
        worst, invalid, total = element_quality(3)
        assert invalid > 0, (
            "no element folded on curving; if a gmsh version stops doing "
            "this the optimiser step is merely belt-and-braces, which is "
            "worth knowing")
        assert worst < 0.0


@pytest.mark.parametrize("order", [2, 3])
def test_raise_order_repairs_the_fold(order):
    with session(name="raise"):
        mesh_shells(3)
        orient_mesh(3)
        report = raise_order(3, order)
        assert report["invalid_before"] > 0 and report["optimized"]
        assert report["invalid"] == 0
        assert report["min_sicn"] > 0.0


def test_refining_also_removes_the_fold():
    """It is curvature against element size, not a gmsh defect."""
    with session(name="fine"):
        mesh_shells(3, size=0.04)
        orient_mesh(3)
        gmsh.model.mesh.setOrder(2)
        _, invalid, _ = element_quality(3)
        assert invalid == 0


def test_order_one_needs_no_optimisation():
    with session(name="linear"):
        mesh_shells(3)
        orient_mesh(3)
        report = raise_order(3, 1)
        assert not report["optimized"] and report["invalid"] == 0


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_full_orientation_pipeline_is_clean(order):
    """Orientation, as the brief requires: nothing negative, nothing inward."""
    with session(name="pipeline"):
        mesh_shells(3)
        orient_mesh(3)
        report = raise_order(3, order)
        assert report["invalid"] == 0
        assert negative_cells(3) == 0
        assert inward_faces() == 0


def test_raise_order_validates_its_argument():
    with session(name="bad"):
        mesh_shells(2)
        with pytest.raises(ValueError, match="at least 1"):
            raise_order(2, 0)
