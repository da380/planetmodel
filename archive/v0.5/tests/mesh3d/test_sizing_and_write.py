"""Per-interface sizing, first meshes, and the MSH 2.2 round trip."""
import numpy as np
import pytest

import gmsh

from planetmodel.mesh3d._geometry import build_concentric
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._sizing import apply_mesh_options, apply_size_fields
from planetmodel.mesh3d._tagging import apply_physical_groups, identify
from planetmodel.mesh3d._writer import element_counts, read_groups, write_msh
from planetmodel.mesh3d.spec import InterfaceSizing

pytestmark = pytest.mark.gmsh

RADII = (0.2, 0.55, 1.0)


def uniform(size=0.08, far=0.2, decay=0.3):
    return {i: InterfaceSizing(size, far, decay) for i in range(len(RADII))}


def build_and_mesh(dimension, sizes, order=1):
    """The pipeline as far as it goes: build, tag, size, mesh."""
    g = build_concentric(RADII, dimension=dimension)
    t = identify(g, RADII)
    apply_physical_groups(t)
    apply_size_fields(t, sizes)
    smallest = min(s.size for s in sizes.values())
    largest = max(s.far_size for s in sizes.values())
    apply_mesh_options(order=order, algorithm_2d=6, algorithm_3d=1,
                       size_min=smallest, size_max=largest)
    gmsh.model.mesh.generate(dimension)
    return t


@pytest.mark.parametrize("dimension", [2, 3])
def test_a_mesh_is_produced_with_sane_quality(dimension):
    with session(name="mesh"):
        build_and_mesh(dimension, uniform())
        counts = element_counts()
        assert counts["elements"] > 0 and counts["nodes"] > 0
        _, tags, _ = gmsh.model.mesh.getElements(dimension)
        q = gmsh.model.mesh.getElementQualities(np.concatenate(tags), "minSICN")
        assert q.min() > 0.0, "an inverted or degenerate element"


@pytest.mark.parametrize("dimension", [2, 3])
def test_sizing_is_honoured_per_interface(dimension):
    """The point of one field per interface: different boundaries can
    have different resolutions, and refining one must not refine all."""
    with session(name="coarse"):
        build_and_mesh(dimension, uniform(size=0.1))
        coarse = element_counts()["elements"]
    with session(name="fine-middle"):
        sizes = uniform(size=0.1)
        sizes[1] = InterfaceSizing(0.025, 0.2, 0.3)   # only the middle one
        build_and_mesh(dimension, sizes)
        refined = element_counts()["elements"]
    assert refined > 1.5 * coarse, (coarse, refined)


def test_elements_cluster_near_the_refined_interface():
    """Not just more elements -- more of them in the right place."""
    with session(name="cluster"):
        sizes = uniform(size=0.15)
        sizes[1] = InterfaceSizing(0.02, 0.25, 0.15)
        build_and_mesh(3, sizes)
        _, coords, _ = gmsh.model.mesh.getNodes()
        r = np.linalg.norm(coords.reshape(-1, 3), axis=1)
        near = np.abs(r - RADII[1]) < 0.05
        assert near.mean() > 0.2, (
            f"only {near.mean():.1%} of nodes lie near the refined interface")


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
        g = build_concentric(RADII, dimension=3)
        t = identify(g, RADII)
        with pytest.raises(ValueError, match="no sizing"):
            apply_size_fields(t, {})
        with pytest.raises(ValueError, match="no sizing for interface 2"):
            apply_size_fields(t, {0: InterfaceSizing(0.1, 0.2, 0.3),
                                  1: InterfaceSizing(0.1, 0.2, 0.3)})


# ------------------------------------------------------- the round trip

@pytest.mark.parametrize("dimension", [2, 3])
def test_written_mesh_reads_back_with_its_groups(dimension, tmp_path):
    """What is in memory must be what is in the file.

    gmsh's writer drops entities belonging to no physical group, so a
    numbering that exists in memory but not on disk would be a silent
    loss -- exactly the kind a consumer discovers by selecting the
    wrong material.
    """
    with session(name="write"):
        g = build_concentric(RADII, dimension=dimension)
        t = identify(g, RADII)
        apply_physical_groups(t, layer_names=["core", "mantle", "crust"],
                              interface_names=["icb", "cmb", "surface"])
        apply_size_fields(t, uniform())
        apply_mesh_options(order=1, algorithm_2d=6, algorithm_3d=1,
                           size_min=0.08, size_max=0.2)
        gmsh.model.mesh.generate(dimension)
        path = write_msh(tmp_path / "m")
        written = element_counts()
    assert path.exists()

    with session(name="reread"):
        groups = read_groups(path)
        assert sorted(groups[dimension]) == [1, 2, 3]
        assert sorted(groups[dimension - 1]) == [1, 2, 3]
        assert groups[dimension][1] == "core"
        assert groups[dimension - 1][3] == "surface"

        reread = element_counts()
        # The tagged dimensions survive exactly ...
        for d in (dimension, dimension - 1):
            assert reread[f"dim{d}"] == written[f"dim{d}"], f"dim {d}"
        # ... and the untagged ones are dropped, which is what we want:
        # gmsh writes only elements belonging to a physical group, so the
        # stray point and curve elements never reach the consumer.  It
        # also means the element *total* is not a round-trip invariant.
        for d in (0, 1, 2, 3):
            if d not in (dimension, dimension - 1):
                assert reread.get(f"dim{d}", 0) == 0


def test_the_file_is_msh_2_2(tmp_path):
    """MFEM's reader wants 2.2; gmsh defaults to 4.1."""
    with session(name="version"):
        build_and_mesh(3, uniform())
        path = write_msh(tmp_path / "v")
    header = path.read_text(errors="ignore").splitlines()[:2]
    assert header[0].strip() == "$MeshFormat"
    assert header[1].split()[0] == "2.2"


@pytest.mark.parametrize("order", [1, 2, 3])
def test_higher_order_meshes_write_and_read(order, tmp_path):
    with session(name="order"):
        build_and_mesh(3, uniform(size=0.2, far=0.4, decay=0.4), order=order)
        n = element_counts()["nodes"]
        path = write_msh(tmp_path / f"o{order}")
    with session(name="reread"):
        read_groups(path)
        assert element_counts()["nodes"] == n
    if order > 1:
        assert n > 0
