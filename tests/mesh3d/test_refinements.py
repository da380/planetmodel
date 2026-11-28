"""Refinements: the size fields composed into the sizing by minimum.

The size function itself is plain numpy and is tested without a gmsh
session; the sessions below ask only whether gmsh honours it, on unit
geometries a few elements across.
"""
import dataclasses

import numpy as np
import pytest

import gmsh

from planetmodel.mesh3d import (InterfaceSizing, MeshSpec, NearPoints,
                                build_layered_mesh, manifest, near_curve)
from planetmodel.mesh3d._geometry import build_concentric
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._sizing import (apply_mesh_options, apply_refinements,
                                        apply_size_fields)
from planetmodel.mesh3d._tagging import apply_physical_groups, identify
from planetmodel.mesh3d.spec import QUALITY_FLOOR, check_sizing_scale
from planetmodel.testing import check_refinement

from conftest import COARSE, full_geometry

pytestmark = pytest.mark.gmsh


def ramp(distance, size, far, width):
    """The expected size at `distance`: `size` out to one `size` (the
    plateau), then the linear ramp, capped at far."""
    d = np.maximum(np.asarray(distance, dtype=float) - size, 0.0)
    return np.minimum(far, size + (far - size) * d / width)


def constant_refinement(value):
    """A bare function refinement returning one size everywhere."""
    def h(points):
        return np.full(len(points), float(value))
    h.size = float(value)
    h.far_size = float(value)
    return h


# ------------------------------------------------------- the size function

def test_near_points_is_the_ramp_of_the_distance_to_the_cloud():
    fine = NearPoints([[1.0, 0.0, 0.0]], 0.05, 0.3, 0.2)
    probes = np.array([[1.0, 0.0, 0.0],      # on the cloud
                       [1.05, 0.0, 0.0],     # at the plateau's edge
                       [1.15, 0.0, 0.0],     # mid-ramp
                       [1.25, 0.0, 0.0],     # where far_size is reached
                       [-1.0, 0.0, 0.0]])    # far beyond it
    got = fine(probes)
    assert got == pytest.approx(ramp([0.0, 0.05, 0.15, 0.25, 2.0],
                                     0.05, 0.3, 0.2))
    assert got[:2] == pytest.approx([0.05, 0.05])   # flat across the plateau
    assert got[2] == pytest.approx(0.175)           # halfway up the ramp
    assert got[3:] == pytest.approx([0.3, 0.3])


def test_the_refined_region_is_a_tube_about_the_curve():
    # a great circle in the xy plane: directly beneath it the size is
    # back to far_size one decay width down, so nothing fine is drilled
    # towards the centre
    phi = np.linspace(0.0, 2.0 * np.pi, 200, endpoint=False)
    cloud = np.stack([np.cos(phi), np.sin(phi), np.zeros_like(phi)], axis=1)
    fine = NearPoints(cloud, 0.02, 0.3, 0.1)
    under = np.array([[1.0 - 2.0 * 0.1, 0.0, 0.0], [0.0, 0.0, 0.0]])
    assert fine(under) == pytest.approx([0.3, 0.3])
    beside = np.array([[np.cos(0.01), np.sin(0.01), 0.0]])
    assert float(fine(beside)[0]) < 0.3


def test_near_points_validates_its_cloud_and_its_ramp():
    with pytest.raises(ValueError, match=r"shape \(n, 2\) or \(n, 3\)"):
        NearPoints([1.0, 0.0, 0.0], 0.05, 0.3, 0.2)
    with pytest.raises(ValueError, match=r"shape \(n, 2\) or \(n, 3\)"):
        NearPoints(np.zeros((0, 3)), 0.05, 0.3, 0.2)
    with pytest.raises(ValueError, match="finite"):
        NearPoints([[np.nan, 0.0, 0.0]], 0.05, 0.3, 0.2)
    with pytest.raises(ValueError, match="size must be positive"):
        NearPoints([[1.0, 0.0, 0.0]], -0.05, 0.3, 0.2)
    with pytest.raises(ValueError, match="refine away from the target"):
        NearPoints([[1.0, 0.0, 0.0]], 0.5, 0.3, 0.2)
    fine = NearPoints([[1.0, 0.0, 0.0]], 0.05, 0.3, 0.2)
    with pytest.raises(ValueError, match=r"shape \(n, 3\) like the cloud"):
        fine(np.zeros((4, 2)))


def test_near_points_is_frozen_comparable_and_printable():
    cloud = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    a = NearPoints(cloud, 0.05, 0.3, 0.2)
    assert a == NearPoints(np.asarray(cloud), 0.05, 0.3, 0.2)
    assert a != NearPoints(cloud, 0.06, 0.3, 0.2)
    assert a != NearPoints([[1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], 0.05, 0.3, 0.2)
    assert hash(a) == hash(NearPoints(cloud, 0.05, 0.3, 0.2))
    assert f"2 points in 3D [{a.digest}]" in repr(a)
    assert len(a.digest) == 16
    assert not a.points.flags.writeable
    with pytest.raises(dataclasses.FrozenInstanceError):
        a.size = 0.1
    # the stored cloud is a copy: mutating the caller's array changes nothing
    theirs = np.array(cloud)
    b = NearPoints(theirs, 0.05, 0.3, 0.2)
    theirs[0, 0] = 2.0
    assert b == a


def test_near_curve_resamples_each_segment_to_the_asked_spacing():
    fine = near_curve([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.7, 0.0]],
                      0.15, 0.5, 0.3)
    assert isinstance(fine, NearPoints)
    pts = fine.points
    assert np.allclose(pts[0], [0.0, 0.0, 0.0])
    assert np.allclose(pts[-1], [1.0, 0.7, 0.0])
    spacing = np.linalg.norm(np.diff(pts, axis=0), axis=1)
    assert np.all(spacing <= 0.15 + 1e-12)
    with pytest.raises(ValueError, match=r"shape \(m, 2\) or \(m, 3\)"):
        near_curve([[0.0, 0.0, 0.0]], 0.15, 0.5, 0.3)
    with pytest.raises(ValueError, match="size must be positive"):
        near_curve([[0.0, 0.0], [1.0, 0.0]], 0.0, 0.5, 0.3)


def test_near_curve_keeps_the_ripple_inside_the_plateau():
    # spacing <= size puts the distance ripple (at most half the
    # spacing) inside the plateau, so the size on the curve is exactly
    # the one asked, everywhere
    size, far, width = 0.5, 2.0, 1.0
    fine = near_curve([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], size, far, width)
    on_curve = np.stack([np.linspace(0.0, 10.0, 1001),
                         np.zeros(1001), np.zeros(1001)], axis=1)
    h = fine(on_curve)
    assert float(h.min()) == float(h.max()) == pytest.approx(size)


def test_check_refinement_holds_the_shipped_kinds_and_catches_violations():
    probes = np.concatenate([np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]),
                             np.random.default_rng(0).uniform(-1, 1, (30, 3))])
    check_refinement(NearPoints([[1.0, 0.0, 0.0]], 0.05, 0.3, 0.2), probes)
    check_refinement(near_curve([[0.0, 0.0], [1.0, 0.0]], 0.1, 0.4, 0.2),
                     probes[:, :2])
    check_refinement(constant_refinement(0.2), probes)

    with pytest.raises(AssertionError, match="positive number"):
        bare = lambda points: np.full(len(points), 0.2)   # noqa: E731
        check_refinement(bare, probes)
    breaks_floor = constant_refinement(0.2)
    breaks_floor.size = 0.5
    breaks_floor.far_size = 0.5
    with pytest.raises(AssertionError, match="below the declared floor"):
        check_refinement(breaks_floor, probes)

    def matrix(points):
        return np.full((len(points), 2), 0.2)
    matrix.size, matrix.far_size = 0.2, 0.2
    with pytest.raises(AssertionError, match="shape"):
        check_refinement(matrix, probes)


# ------------------------------------------------------- spec and checks

def test_mesh_spec_validates_its_refinements():
    g = full_geometry()
    fine = NearPoints([[1.0, 0.0, 0.0]], 0.05, 0.3, 0.2)
    spec = MeshSpec(g, COARSE, refinements=[fine, constant_refinement(0.2)])
    assert isinstance(spec.refinements, tuple) and spec.refinements[0] is fine
    assert MeshSpec(g, COARSE).refinements == ()
    with pytest.raises(TypeError, match=r"refinements\[0\].*not a callable"):
        MeshSpec(g, COARSE, refinements=[0.1])
    with pytest.raises(ValueError, match=r"refinements\[0\].*positive size"):
        bare = lambda points: points   # noqa: E731
        MeshSpec(g, COARSE, refinements=[bare])
    broken = constant_refinement(0.2)
    broken.far_size = 0.1
    with pytest.raises(ValueError, match=r"refinements\[1\].*far_size"):
        MeshSpec(g, COARSE, refinements=[fine, broken])


def test_the_scale_check_judges_refinements_too():
    sizes = {0: InterfaceSizing(0.15, 0.3, 0.3)}
    check_sizing_scale(1.0, sizes, refinements=[constant_refinement(0.1)])
    with pytest.raises(ValueError, match="not a resolution choice"):
        check_sizing_scale(1.0, sizes,
                           refinements=[constant_refinement(1e-7)])
    with pytest.raises(ValueError, match="nothing would be resolved"):
        check_sizing_scale(1.0, sizes,
                           refinements=[constant_refinement(5.0)])


# ------------------------------------------------------------ the manifest

def minimal_card(refinements=()):
    layers = [manifest.LayerEntry(attribute=1, name="ball", r_inner=0.0,
                                  r_outer=1.0, in_geometry=True)]
    faces = [manifest.InterfaceEntry(attribute=1, name="surface", radius=1.0,
                                     between_layers=[0, -1])]
    return manifest.MeshManifest.from_build(
        mesh=manifest.mesh_block("m.msh", format="msh", nodes="reference"),
        layers=layers, interfaces=faces, refinements=refinements)


def test_refinement_entries_record_what_a_refinement_exposes(tmp_path):
    fine = NearPoints([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 0.05, 0.3, 0.2)
    entry = manifest.RefinementEntry.from_refinement(fine, name="NearPoints")
    assert (entry.size, entry.far_size, entry.decay_width) == (0.05, 0.3, 0.2)
    assert entry.points == 2 and entry.digest == fine.digest

    bare = manifest.RefinementEntry.from_refinement(constant_refinement(0.2),
                                                    name="h")
    assert (bare.decay_width, bare.points, bare.digest) == (None, None, None)

    card = minimal_card([entry, bare])
    path = manifest.write(tmp_path / "m", card)
    again = manifest.read(path)
    assert again.refinements == card.refinements
    assert "refinements" in card.describe() and fine.digest in card.describe()


def test_the_structure_check_names_a_broken_refinement():
    card = minimal_card([manifest.RefinementEntry("h", 0.1, 0.2)])
    card.refinements[0]["far_size"] = 0.05
    with pytest.raises(ValueError, match=r"refinements\[0\].*far_size"):
        manifest.validate_structure(card)
    card = minimal_card([manifest.RefinementEntry("h", 0.1, 0.2)])
    card.refinements[0]["points"] = 0
    with pytest.raises(ValueError, match=r"refinements\[0\].points"):
        manifest.validate_structure(card)
    card = minimal_card([manifest.RefinementEntry("h", 0.1, 0.2)])
    card.refinements[0]["digest"] = 7
    with pytest.raises(ValueError, match=r"refinements\[0\].digest"):
        manifest.validate_structure(card)


# ---------------------------------------------------------- gmsh honours it

def mesh_ball(dimension, sizes, refinements=()):
    """The pipeline on one unit ball or disc, refinements included."""
    g = build_concentric((0.0, 1.0), dimension=dimension)
    t = identify(g, (1.0,))
    apply_physical_groups(t)
    apply_size_fields(t, sizes)
    apply_refinements(refinements, dimension=dimension)
    apply_mesh_options(order=1, algorithm_2d=6, algorithm_3d=1,
                       size_min=min([s.size for s in sizes.values()]
                                    + [f.size for f in refinements]),
                       size_max=max(s.far_size for s in sizes.values()))
    gmsh.model.mesh.generate(dimension)
    return t


def all_nodes():
    _, xyz, _ = gmsh.model.mesh.getNodes()
    return np.asarray(xyz).reshape(-1, 3)


def test_a_pass_through_callback_reproduces_the_background_mesh():
    """The fact the design rests on: gmsh hands the callback the size its
    other sources computed, and uses what the callback returns."""
    sizes = {0: InterfaceSizing(0.15, 0.3, 0.3)}
    with session(name="base"):
        mesh_ball(2, sizes)
        base = all_nodes()
    with session(name="hands-off"):
        g = build_concentric((0.0, 1.0), dimension=2)
        t = identify(g, (1.0,))
        apply_physical_groups(t)
        apply_size_fields(t, sizes)
        gmsh.model.mesh.setSizeCallback(
            lambda dim, tag, x, y, z, lc: lc)
        apply_mesh_options(order=1, algorithm_2d=6, algorithm_3d=1,
                           size_min=0.15, size_max=0.3)
        gmsh.model.mesh.generate(2)
        got = all_nodes()
    assert got.shape == base.shape
    order = np.lexsort(base.T)
    assert np.allclose(got[np.lexsort(got.T)], base[order], atol=1e-12)


def edge_midpoints_and_lengths(curve):
    tags, xyz, _ = gmsh.model.mesh.getNodes(1, curve, includeBoundary=True)
    pos = dict(zip(tags, np.asarray(xyz).reshape(-1, 3)))
    _, _, nodes = gmsh.model.mesh.getElements(1, curve)
    ends = np.asarray(nodes[0]).reshape(-1, 2)
    mids = np.array([0.5 * (pos[a] + pos[b]) for a, b in ends])
    lengths = np.array([np.linalg.norm(pos[a] - pos[b]) for a, b in ends])
    return mids, lengths


def test_the_measured_sizes_follow_the_ramp_on_a_disc():
    sizes = {0: InterfaceSizing(0.15, 0.3, 0.3)}
    fine = NearPoints([[1.0, 0.0]], 0.05, 0.3, 0.25)
    with session(name="disc"):
        t = mesh_ball(2, sizes, refinements=[fine])
        mids, lengths = edge_midpoints_and_lengths(t.faces[0])
    near = np.linalg.norm(mids - [1.0, 0.0, 0.0], axis=1) < 0.1
    far = np.linalg.norm(mids - [-1.0, 0.0, 0.0], axis=1) < 0.4
    assert near.any() and far.any()
    assert lengths[near].mean() == pytest.approx(0.05, rel=0.3)
    assert lengths[near].max() < 0.1
    assert lengths[far].mean() == pytest.approx(0.15, rel=0.25)


def test_the_refined_patch_is_denser_on_a_ball():
    sizes = {0: InterfaceSizing(0.15, 0.3, 0.3)}
    fine = NearPoints([[1.0, 0.0, 0.0]], 0.06, 0.3, 0.3)
    with session(name="ball"):
        t = mesh_ball(3, sizes, refinements=[fine])
        surface = t.faces[int(np.argmax(t.radii))]
        _, xyz, _ = gmsh.model.mesh.getNodes(2, surface, includeBoundary=True)
        nodes = np.asarray(xyz).reshape(-1, 3)
    refined = int(np.sum(np.linalg.norm(nodes - [1.0, 0.0, 0.0], axis=1) < 0.15))
    coarse = int(np.sum(np.linalg.norm(nodes + [1.0, 0.0, 0.0], axis=1) < 0.15))
    assert refined >= 3 * max(coarse, 1)


def test_a_refinement_no_finer_than_the_sizing_changes_nothing(tmp_path):
    """Minimum composition: a refinement can only refine."""
    g = full_geometry()
    base = build_layered_mesh(MeshSpec(g, COARSE, dimension=2),
                              tmp_path / "base")
    ceiling = NearPoints([[0.0, 1.0]], COARSE.far_size, COARSE.far_size, 0.3)
    same = build_layered_mesh(MeshSpec(g, COARSE, dimension=2,
                                       refinements=[ceiling]),
                              tmp_path / "same")
    assert same.counts == base.counts
    card = manifest.read(same.manifest_path)
    assert [e["name"] for e in card.refinements] == ["NearPoints"]


def test_a_feature_cloud_builds_checks_and_reaches_the_manifest(tmp_path):
    g = full_geometry()
    fine = NearPoints([[1.0, 0.0, 0.0]], 0.06, 0.3, 0.3)
    base = build_layered_mesh(MeshSpec(g, COARSE), tmp_path / "base")
    got = build_layered_mesh(MeshSpec(g, COARSE, refinements=[fine]),
                             tmp_path / "fine")
    assert got.validation.ok
    assert got.validation.min_sicn > QUALITY_FLOOR
    assert got.counts["elements"] > base.counts["elements"]
    card = manifest.read(got.manifest_path)
    [entry] = card.refinements
    assert entry["name"] == "NearPoints"
    assert (entry["size"], entry["far_size"], entry["decay_width"]) == \
        (0.06, 0.3, 0.3)
    assert entry["points"] == 1 and entry["digest"] == fine.digest
    assert manifest.read(base.manifest_path).refinements == []
