"""The private pipeline, one step at a time: session, CAD, tagging,
sizing, meshing, orientation, curving, displacement, writing."""
import numpy as np
import pytest

import gmsh

from planetmodel import IdentityMapping
from planetmodel.mesh3d import InterfaceSizing
from planetmodel.mesh3d._displace import apply_mapping
from planetmodel.mesh3d._geometry import (ConcentricGeometry, build_concentric,
                                          entity_radius, outer_face_of)
from planetmodel.mesh3d._orient import (element_quality, node_positions,
                                        orient_mesh, outward_dots, raise_order,
                                        signed_measures)
from planetmodel.mesh3d._session import is_active, session, set_options
from planetmodel.mesh3d._sizing import (apply_mesh_options, apply_size_fields,
                                        check_sizing_resolves_spans,
                                        check_sizing_scale)
from planetmodel.mesh3d._tagging import (apply_physical_groups, default_atol,
                                         identify, mean_radius_of_entity)
from planetmodel.mesh3d._validate import check_interface_radii, validate_mesh
from planetmodel.mesh3d._writer import element_counts, read_groups, write_msh

from conftest import flattening

pytestmark = pytest.mark.gmsh

FULL = (0.0, 0.4, 0.8, 1.0)          # skeleton boundaries of a full ball
HOLLOW = (0.5, 0.8, 1.0)             # and of a hollow one


def faces_of(boundaries):
    """The interface radii: every boundary but a centre at zero."""
    return tuple(r for r in boundaries if r > 0.0)


def uniform(n, size=0.15, far=0.3, decay=0.3):
    return {i: InterfaceSizing(size, far, decay) for i in range(n)}


def build_and_mesh(boundaries, dimension, *, sizes=None, order=1):
    """The pipeline as far as it goes: build, tag, size, mesh."""
    g = build_concentric(boundaries, dimension=dimension)
    t = identify(g, faces_of(boundaries))
    apply_physical_groups(t)
    sizes = uniform(len(t.faces)) if sizes is None else sizes
    apply_size_fields(t, sizes)
    apply_mesh_options(order=order, algorithm_2d=6, algorithm_3d=1,
                       size_min=min(s.size for s in sizes.values()),
                       size_max=max(s.far_size for s in sizes.values()))
    gmsh.model.mesh.generate(dimension)
    return t


# --------------------------------------------------------------- session

def test_session_initialises_finalizes_and_does_not_nest():
    assert not is_active()
    with session(name="test") as model:
        assert is_active()
        assert model.getCurrent() == "test"
        with pytest.raises(RuntimeError, match="does not nest"):
            with session(name="inner"):
                pass
    assert not is_active()
    with pytest.raises(RuntimeError, match="deliberate"):
        with session(name="test"):
            raise RuntimeError("deliberate")
    assert not is_active()


def test_set_options_dispatches_on_value_type():
    with session(name="test"):
        set_options({"Mesh.ElementOrder": 2, "General.ErrorFileName": "err.log"})
        assert gmsh.option.getNumber("Mesh.ElementOrder") == 2
        assert gmsh.option.getString("General.ErrorFileName") == "err.log"


# ------------------------------------------------------------------- CAD

@pytest.mark.parametrize("dimension", [2, 3])
def test_a_full_ball_has_one_cell_and_face_per_shell(dimension):
    with session(name="build"):
        g = build_concentric(FULL, dimension=dimension)
        assert not g.hollow and g.n_layers == 3 and len(g.faces) == 3
        assert g.radii == faces_of(FULL)
        measured = sorted(entity_radius(dimension - 1, t) for t in g.faces)
        assert np.allclose(measured, faces_of(FULL), atol=2e-7)


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_hollow_ball_keeps_its_inner_boundary_as_a_face(dimension):
    with session(name="hollow"):
        g = build_concentric(HOLLOW, dimension=dimension)
        assert g.hollow and g.n_layers == 2 and len(g.faces) == 3
        measured = sorted(entity_radius(dimension - 1, t) for t in g.faces)
        assert np.allclose(measured, HOLLOW, atol=2e-7)
        # no cell is the inner ball
        assert all(entity_radius(dimension, c) > 0.5 + 1e-6 for c in g.cells)


def test_build_validates_its_radii():
    with session(name="validate"):
        with pytest.raises(ValueError, match="strictly increasing"):
            build_concentric((1.0, 0.5), dimension=3)
        with pytest.raises(ValueError, match="dimension must be"):
            build_concentric((0.0, 1.0), dimension=4)
        with pytest.raises(ValueError, match="at least one shell"):
            build_concentric((0.0,), dimension=3)
        with pytest.raises(ValueError, match="at least one shell"):
            build_concentric((0.5,), dimension=3)


def test_occ_pads_bounding_boxes_absolutely():
    """The measurement behind the matching tolerance."""
    with session(name="pad"):
        for scale in (1.0, 6.371e6):
            gmsh.model.occ.addSphere(0, 0, 0, scale)
            gmsh.model.occ.synchronize()
            pad = entity_radius(2, gmsh.model.getEntities(2)[-1][1]) - scale
            assert 0.0 < pad < 2e-7, f"padding {pad} at scale {scale}"
        assert 1e-7 < default_atol(1.2) < 1e-3


# --------------------------------------------------------------- tagging

@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("boundaries", [FULL, HOLLOW])
def test_each_layer_is_bounded_above_by_its_own_interface(dimension, boundaries):
    with session(name="tag"):
        g = build_concentric(boundaries, dimension=dimension)
        t = identify(g, faces_of(boundaries))
        assert t.radii == faces_of(boundaries)
        assert t.hollow == (boundaries[0] > 0.0)
        first = 1 if t.hollow else 0
        assert len(t.cells) == len(t.faces) - first
        for i, cell in enumerate(t.cells):
            assert outer_face_of(dimension, cell) == t.faces[i + first]
        assert len(set(t.cells)) == len(t.cells)


@pytest.mark.parametrize("dimension", [2, 3])
def test_identification_is_invariant_under_construction_order(dimension):
    def radii_for(order):
        with session(name="shuffle"):
            add = (lambda r: gmsh.model.occ.addSphere(0, 0, 0, r) if dimension == 3
                   else gmsh.model.occ.addDisk(0, 0, 0, r, r))
            outer = add(1.0)
            tools = [(dimension, add(r)) for r in order]
            gmsh.model.occ.fragment([(dimension, outer)], tools)
            gmsh.model.occ.removeAllDuplicates()
            gmsh.model.occ.synchronize()
            g = ConcentricGeometry(
                dimension, faces_of(FULL),
                tuple(t for _, t in gmsh.model.getEntities(dimension)),
                tuple(t for _, t in gmsh.model.getEntities(dimension - 1)), False)
            t = identify(g, faces_of(FULL))
            return [round(entity_radius(dimension - 1, f), 6) for f in t.faces]

    assert radii_for((0.4, 0.8)) == radii_for((0.8, 0.4)) == [0.4, 0.8, 1.0]


def test_a_mismatch_fails_loudly_with_both_lists():
    with session(name="bad"):
        g = build_concentric(FULL, dimension=3)
        with pytest.raises(RuntimeError) as exc:
            identify(g, (0.4, 0.7, 1.0))
        assert "expected radii" in str(exc.value) and "0.7" in str(exc.value)
        with pytest.raises(RuntimeError, match="matched no requested radius"):
            identify(g, (0.4, 0.8))
        with pytest.raises(RuntimeError, match="matched 2 faces|matched 0"):
            identify(g, faces_of(FULL), atol=0.5)


@pytest.mark.parametrize("dimension", [2, 3])
def test_physical_groups_run_centre_outward_with_names(dimension):
    with session(name="groups"):
        t = identify(build_concentric(HOLLOW, dimension=dimension), HOLLOW)
        out = apply_physical_groups(t, layer_names=["lower", None],
                                    interface_names=["inner", "mid", "outer"])
        assert [tag for _, tag in gmsh.model.getPhysicalGroups(dimension)] == [1, 2]
        assert [tag for _, tag in gmsh.model.getPhysicalGroups(dimension - 1)] \
            == [1, 2, 3]
        assert gmsh.model.getPhysicalName(dimension, 1) == "lower"
        assert gmsh.model.getPhysicalName(dimension, 2) == "layer_2"
        assert gmsh.model.getPhysicalName(dimension - 1, 1) == "inner"
        for attr, want in zip((1, 2, 3), HOLLOW):
            (entity,) = gmsh.model.getEntitiesForPhysicalGroup(dimension - 1, attr)
            assert entity_radius(dimension - 1, entity) == pytest.approx(
                want, abs=2e-7)
        assert set(out) == {"layers", "interfaces"}


# ---------------------------------------------------------------- sizing

@pytest.mark.parametrize("dimension", [2, 3])
@pytest.mark.parametrize("boundaries", [FULL, HOLLOW])
def test_a_mesh_is_produced_with_positive_quality(dimension, boundaries):
    with session(name="mesh"):
        build_and_mesh(boundaries, dimension)
        counts = element_counts(dimension=dimension)
        assert counts["elements"] > 0 and counts["nodes"] > 0
        worst, invalid, total = element_quality(dimension)
        assert worst > 0.0 and invalid == 0 and total > 0


def test_sizing_is_honoured_per_interface():
    """Refining one interface must not refine all of them."""
    with session(name="coarse"):
        build_and_mesh(FULL, 2, sizes=uniform(3, size=0.1))
        coarse = element_counts(dimension=2)["elements"]
    with session(name="fine-middle"):
        sizes = uniform(3, size=0.1)
        sizes[1] = InterfaceSizing(0.02, 0.3, 0.3)
        build_and_mesh(FULL, 2, sizes=sizes)
        refined = element_counts(dimension=2)["elements"]
        _, coords, _ = gmsh.model.mesh.getNodes()
        r = np.linalg.norm(coords.reshape(-1, 3), axis=1)
        assert np.mean(np.abs(r - 0.8) < 0.05) > 0.3
    assert refined > 1.5 * coarse


def test_size_options_disable_the_competing_sources():
    with session(name="opts"):
        apply_mesh_options(order=2, algorithm_2d=6, algorithm_3d=1,
                           size_min=0.01, size_max=0.2)
        for opt in ("Mesh.MeshSizeExtendFromBoundary", "Mesh.MeshSizeFromPoints",
                    "Mesh.MeshSizeFromCurvature"):
            assert gmsh.option.getNumber(opt) == 0, opt
        assert gmsh.option.getNumber("Mesh.ElementOrder") == 2


def test_missing_sizing_is_refused():
    with session(name="missing"):
        t = identify(build_concentric(FULL, dimension=3), faces_of(FULL))
        with pytest.raises(ValueError, match="no sizing"):
            apply_size_fields(t, {})
        with pytest.raises(ValueError, match="no sizing for interface 2"):
            apply_size_fields(t, uniform(2))


def test_span_check_judges_each_layer_by_its_bounding_interfaces():
    fine = uniform(3, size=0.1)
    assert check_sizing_resolves_spans(FULL, fine, strict=False) == []
    thin = (0.0, 0.4, 0.98, 1.0)
    assert check_sizing_resolves_spans(thin, uniform(3, size=0.1), strict=False) == []
    problems = check_sizing_resolves_spans(thin, uniform(3, size=0.3), strict=False)
    assert len(problems) == 1 and "[0.98, 1]" in problems[0]
    with pytest.raises(ValueError, match="too coarse"):
        check_sizing_resolves_spans(thin, uniform(3, size=0.3))
    # a hollow domain: interface 0 sits on the innermost boundary
    hollow_thin = (0.5, 0.52, 1.0)
    assert check_sizing_resolves_spans(hollow_thin, uniform(3, size=0.1),
                                       strict=False) == []
    assert check_sizing_resolves_spans(hollow_thin, uniform(3, size=0.3),
                                       strict=False)


def test_scale_check_catches_sizes_at_the_wrong_scale():
    check_sizing_scale(1.0, uniform(1, size=0.1))
    with pytest.raises(ValueError, match="not a resolution choice"):
        check_sizing_scale(1.0, uniform(1, size=1e-7, far=1e-6, decay=1e-6))
    with pytest.raises(ValueError, match="nothing would be resolved"):
        check_sizing_scale(1.0, {0: InterfaceSizing(5.0, 6.0, 1.0)})


# ----------------------------------------------------------- orientation

def negative_cells(dimension, pos=None):
    pos = node_positions() if pos is None else pos
    return sum(int((signed_measures(dimension, tag, pos)[1] < 0.0).sum())
               for _, tag in gmsh.model.getEntities(dimension))


def inward_faces(pos=None):
    pos = node_positions() if pos is None else pos
    return sum(int((outward_dots(tag, pos)[1] < 0.0).sum())
               for _, tag in gmsh.model.getEntities(2))


def test_orientation_repair_leaves_nothing_inward_and_is_idempotent():
    with session(name="repair"):
        build_and_mesh(FULL, 3)
        assert negative_cells(3) == 0
        first = orient_mesh(3)
        assert inward_faces() == 0 and negative_cells(3) == 0
        assert orient_mesh(3).clean
        assert first.faces_checked > 0


@pytest.mark.parametrize("dimension", [2, 3])
def test_orientation_survives_the_msh_round_trip(dimension, tmp_path):
    with session(name="write"):
        build_and_mesh(FULL, dimension)
        orient_mesh(dimension)
        path = write_msh(tmp_path / "o")
    with session(name="read"):
        read_groups(path)
        assert negative_cells(dimension) == 0
        if dimension == 3:
            assert inward_faces() == 0


@pytest.mark.parametrize("order", [1, 2, 3])
def test_raise_order_leaves_every_element_valid(order):
    with session(name="raise"):
        build_and_mesh(FULL, 3)
        orient_mesh(3)
        report = raise_order(3, order)
        assert report["order"] == order and report["invalid"] == 0
        assert report["min_sicn"] > 0.0
        if order == 1:
            assert not report["optimized"]
        assert negative_cells(3) == 0 and inward_faces() == 0
        with pytest.raises(ValueError, match="at least 1"):
            raise_order(3, 0)


# ---------------------------------------------------------- displacement

def test_apply_mapping_moves_every_node_and_reports():
    with session(name="move"):
        build_and_mesh(FULL, 3)
        _, before, _ = gmsh.model.mesh.getNodes()
        X = before.reshape(-1, 3)
        report = apply_mapping(flattening(0.05))
        _, after, _ = gmsh.model.mesh.getNodes()
        x = after.reshape(-1, 3)
        assert report.nodes == X.shape[0]
        assert report.max_displacement == pytest.approx(0.05, rel=0.05)
        assert report.validity_margin > 0.0
        assert np.allclose(x, flattening(0.05)(X))
        assert apply_mapping(IdentityMapping()).max_displacement == 0.0


def test_a_folding_mapping_is_refused_before_any_node_moves():
    with session(name="fold"):
        build_and_mesh(FULL, 3)
        _, before, _ = gmsh.model.mesh.getNodes()
        with pytest.raises(ValueError, match="orientation-preserving"):
            apply_mapping(flattening(3.0))
        _, after, _ = gmsh.model.mesh.getNodes()
        assert np.array_equal(before, after)


# ------------------------------------------------------------ validation

@pytest.mark.parametrize("boundaries", [FULL, HOLLOW])
def test_validate_mesh_passes_a_clean_mesh(boundaries):
    with session(name="ok"):
        t = build_and_mesh(boundaries, 3)
        orient_mesh(3)
        raise_order(3, 2)
        rep = validate_mesh(t, expected_radii=faces_of(boundaries))
        assert rep.ok, rep.failures
        assert rep.negative_jacobians == 0 and rep.inward_faces == 0
        assert rep.max_interface_radius_error < 1e-3
        assert rep.group_counts == {"layers": len(t.cells),
                                    "interfaces": len(t.faces)}


def test_validate_mesh_names_a_wrong_radius_and_a_wrong_name():
    with session(name="wrong"):
        t = build_and_mesh(FULL, 2)
        worst, failures = check_interface_radii(t, (0.4, 0.7, 1.0))
        assert worst == pytest.approx(0.1, abs=1e-3) and len(failures) == 1
        rep = validate_mesh(t, expected_radii=(0.4, 0.7, 1.0),
                            layer_names=["core", "wrong", None])
        assert not rep.ok
        assert any("interface 2 has mean radius" in f for f in rep.failures)
        assert any("layer 2 is named 'layer_2'" in f for f in rep.failures)
        with pytest.raises(ValueError, match="failed validation"):
            rep.raise_if_failed()


def test_node_average_radius_is_the_fallback_measure():
    with session(name="nodes"):
        t = build_and_mesh((0.0, 1.0), 3)
        assert mean_radius_of_entity(2, t.faces[0]) == pytest.approx(1.0, rel=1e-2)


# ---------------------------------------------------------------- writer

@pytest.mark.parametrize("dimension", [2, 3])
def test_written_mesh_reads_back_with_its_groups(dimension, tmp_path):
    with session(name="write"):
        g = build_concentric(HOLLOW, dimension=dimension)
        t = identify(g, HOLLOW)
        apply_physical_groups(t, layer_names=["lower", "upper"],
                              interface_names=["inner", "mid", "outer"])
        apply_size_fields(t, uniform(3))
        apply_mesh_options(order=1, algorithm_2d=6, algorithm_3d=1,
                           size_min=0.15, size_max=0.3)
        gmsh.model.mesh.generate(dimension)
        path = write_msh(tmp_path / "m")
        written = element_counts(dimension=dimension)
    header = path.read_text(errors="ignore").splitlines()[:2]
    assert header[0].strip() == "$MeshFormat" and header[1].split()[0] == "2.2"
    with session(name="reread"):
        groups = read_groups(path)
        assert groups[dimension] == {1: "lower", 2: "upper"}
        assert groups[dimension - 1] == {1: "inner", 2: "mid", 3: "outer"}
        reread = element_counts(dimension=dimension)
        for d in (dimension, dimension - 1):
            assert reread[f"dim{d}"] == written[f"dim{d}"]
        assert reread["elements"] == written["elements"]
