"""Concentric CAD, and matching entities to the geometry asked for.

The attribute numbers are the interface with every consumer, so the
tests that matter here are the ones about *identification* rather than
construction: that the answer does not depend on the order gmsh happens
to return entities in, and that a mismatch fails loudly instead of
producing a plausible wrong mesh.
"""
import numpy as np
import pytest

import gmsh

from planetmodel.mesh3d._geometry import (build_concentric, entity_radius,
                                     outer_face_of)
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._tagging import (apply_physical_groups, identify,
                                    mean_radius_of_entity)

pytestmark = pytest.mark.gmsh

RADII = (0.2, 0.55, 1.0)


@pytest.mark.parametrize("dimension", [2, 3])
def test_build_produces_one_cell_and_face_per_radius(dimension):
    with session(name="build"):
        g = build_concentric(RADII, dimension=dimension)
        assert g.n_layers == len(RADII)
        assert len(g.faces) == len(RADII)
        assert g.dimension == dimension


@pytest.mark.parametrize("dimension", [2, 3])
def test_bounding_boxes_give_the_radii_to_within_the_occ_padding(dimension):
    with session(name="radii"):
        g = build_concentric(RADII, dimension=dimension)
        measured = sorted(entity_radius(dimension - 1, t) for t in g.faces)
        assert np.allclose(measured, RADII, atol=2e-7)


def test_occ_pads_bounding_boxes_absolutely():
    """The measurement that sets the matching tolerance.

    OCC returns a bounding box grown by about 1e-7, and that padding is
    absolute: the same 1e-7 whether the radius is 0.2 or 6.371e6.  So a
    purely relative tolerance -- 1e-9 of the outer radius, say -- is far
    too tight in the non-dimensional units the mesher works in, where
    radii are of order 1.  This pins the finding so the tolerance is not
    quietly tightened back.
    """
    from planetmodel.mesh3d._tagging import default_atol

    with session(name="pad"):
        for scale in (1.0, 6.371e6):
            gmsh.model.occ.addSphere(0, 0, 0, scale)
            gmsh.model.occ.synchronize()
            pad = entity_radius(2, gmsh.model.getEntities(2)[-1][1]) - scale
            assert 0.0 < pad < 2e-7, f"padding {pad} at scale {scale}"
        assert default_atol(1.2) > 1e-7        # clears the pad in nd units
        assert default_atol(1.2) < 1e-3        # far tighter than a real span


@pytest.mark.parametrize("dimension", [2, 3])
def test_identification_orders_entities_centre_outward(dimension):
    with session(name="ident"):
        g = build_concentric(RADII, dimension=dimension)
        t = identify(g, RADII)
        assert np.allclose(t.radii, RADII, rtol=1e-12)   # as requested, unpadded
        assert len(set(t.cells)) == len(t.cells)      # a cell claimed once
        assert len(set(t.faces)) == len(t.faces)


@pytest.mark.parametrize("dimension", [2, 3])
def test_each_layer_is_bounded_by_its_own_interface(dimension):
    """The numbering convention, verified rather than assumed."""
    with session(name="bound"):
        g = build_concentric(RADII, dimension=dimension)
        t = identify(g, RADII)
        for i, cell in enumerate(t.cells):
            assert outer_face_of(dimension, cell) == t.faces[i]


def test_tag_order_carries_no_meaning():
    """gmsh returns entities in arbitrary order, and the probe that
    motivated match-not-sort shows it: the r = 1.0 surface came back
    with a lower tag than the r = 0.55 one."""
    with session(name="order"):
        g = build_concentric(RADII, dimension=3)
        by_tag = [entity_radius(2, t) for t in sorted(g.faces)]
        assert by_tag != sorted(by_tag), (
            "tags happened to be radially ordered here; the test cannot "
            "detect sorting-by-tag, though identify() still must not do it")


@pytest.mark.parametrize("dimension", [2, 3])
def test_identification_is_invariant_under_construction_order(dimension):
    """Building the shells in a different order must not renumber them."""
    def groups_for(order):
        with session(name="shuffle"):
            # build the tools in the given order, then fragment
            outer = (gmsh.model.occ.addSphere(0, 0, 0, RADII[-1])
                     if dimension == 3
                     else gmsh.model.occ.addDisk(0, 0, 0, RADII[-1], RADII[-1]))
            tools = []
            for r in order:
                tag = (gmsh.model.occ.addSphere(0, 0, 0, r) if dimension == 3
                       else gmsh.model.occ.addDisk(0, 0, 0, r, r))
                tools.append((dimension, tag))
            gmsh.model.occ.fragment([(dimension, outer)], tools)
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            from planetmodel.mesh3d._geometry import ConcentricGeometry
            g = ConcentricGeometry(
                dimension, RADII,
                tuple(t for _, t in gmsh.model.getEntities(dimension)),
                tuple(t for _, t in gmsh.model.getEntities(dimension - 1)))
            t = identify(g, RADII)
            return tuple(round(x, 12) for x in t.radii)

    assert groups_for(RADII[:-1]) == groups_for(tuple(reversed(RADII[:-1])))


@pytest.mark.parametrize("dimension", [2, 3])
def test_physical_groups_run_centre_outward(dimension):
    with session(name="groups"):
        g = build_concentric(RADII, dimension=dimension)
        t = identify(g, RADII)
        apply_physical_groups(t, layer_names=["core", "mantle", "crust"],
                              interface_names=["icb", "cmb", "surface"])

        cells = gmsh.model.getPhysicalGroups(dimension)
        faces = gmsh.model.getPhysicalGroups(dimension - 1)
        assert [tag for _, tag in cells] == [1, 2, 3]
        assert [tag for _, tag in faces] == [1, 2, 3]

        for attr, want in zip((1, 2, 3), RADII):
            (entity,) = gmsh.model.getEntitiesForPhysicalGroup(
                dimension - 1, attr)
            assert entity_radius(dimension - 1, entity) == pytest.approx(
                want, abs=2e-7)
        assert gmsh.model.getPhysicalName(dimension, 1) == "core"
        assert gmsh.model.getPhysicalName(dimension - 1, 3) == "surface"


def test_unnamed_groups_get_a_default_name():
    with session(name="names"):
        t = identify(build_concentric(RADII, dimension=3), RADII)
        apply_physical_groups(t)
        assert gmsh.model.getPhysicalName(3, 2) == "layer_2"
        assert gmsh.model.getPhysicalName(2, 2) == "interface_2"


# --------------------------------------------------------- loud failure

def test_a_wrong_expected_radius_fails_and_prints_both_lists():
    """Sorting would silently accept this; matching must not."""
    with session(name="bad"):
        g = build_concentric(RADII, dimension=3)
        with pytest.raises(RuntimeError) as exc:
            identify(g, (0.2, 0.7, 1.0))          # 0.55 was built, not 0.7
        msg = str(exc.value)
        assert "expected radii" in msg and "measured faces" in msg
        assert "0.7" in msg


def test_too_few_expected_radii_fails():
    with session(name="few"):
        g = build_concentric(RADII, dimension=3)
        with pytest.raises(RuntimeError) as exc:
            identify(g, (0.2, 0.55))
        assert "matched no requested radius" in str(exc.value)


def test_a_duplicate_radius_within_tolerance_fails():
    """Two faces matching one radius is ambiguous, not a coin toss."""
    with session(name="dup"):
        g = build_concentric(RADII, dimension=3)
        with pytest.raises(RuntimeError, match="matched 2 faces|matched 0"):
            identify(g, (0.2, 0.55, 1.0), atol=0.5)


# ------------------------------------------------------------ validation

def test_build_validates_its_radii():
    with session(name="validate"):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_concentric((1.0, 0.5), dimension=3)
        with pytest.raises(ValueError, match="dimension must be"):
            build_concentric((1.0,), dimension=4)
        with pytest.raises(ValueError, match="at least one non-zero"):
            build_concentric((0.0,), dimension=3)


def test_a_leading_zero_radius_is_the_centre_not_a_surface():
    """A skeleton starts at r = 0; that is a ball, not a hole."""
    with session(name="centre"):
        g = build_concentric((0.0, 0.5, 1.0), dimension=3)
        assert g.n_layers == 2
        assert np.allclose(sorted(entity_radius(2, t) for t in g.faces),
                           [0.5, 1.0], atol=2e-7)


def test_node_average_radius_is_the_fallback_path():
    """Used for the offset benchmark bodies, where a bbox says nothing."""
    with session(name="nodes"):
        g = build_concentric((1.0,), dimension=3)
        gmsh.option.setNumber("Mesh.MeshSizeMin", 0.2)
        gmsh.option.setNumber("Mesh.MeshSizeMax", 0.4)
        gmsh.model.mesh.generate(2)
        got = mean_radius_of_entity(2, g.faces[0])
        assert got == pytest.approx(1.0, rel=1e-2)
