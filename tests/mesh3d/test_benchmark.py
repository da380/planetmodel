"""The offset two-body geometries: a ball inside a ball, off centre.

The geometry exists so a solver can be checked against an answer someone
wrote down, so these tests care most about the two things that would
make it useless: an inclusion tagged as the exterior, and a mesh that
passes validation while being folded.
"""
import json

import numpy as np
import pytest

from planetmodel.io import manifest as sc
from planetmodel.mesh3d import (InterfaceSizing, PerInterface,
                                UniformInterfaces, build_offset_mesh)
from planetmodel.mesh3d._orient import node_positions, outward_dots
from planetmodel.mesh3d._session import session

pytestmark = pytest.mark.gmsh

A, B, D = 0.3, 1.0, 0.35
COARSE = UniformInterfaces(0.12, 0.25, 0.3)


def mean_radius(a, d):
    """The surface average of |c + a n| over the sphere, |c| = d.

    (1/2) int_-1^1 sqrt(a^2 + d^2 + 2 a d u) du, which integrates to
    [(a+d)^3 - |a-d|^3] / 6ad -- the number the node average estimates,
    and the cheapest independent statement that the boundary tagged as
    the inclusion really is the offset sphere.
    """
    return ((a + d) ** 3 - abs(a - d) ** 3) / (6.0 * a * d)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp("bench") / "two_sphere"
    return build_offset_mesh(path, inner_radius=A, outer_radius=B,
                                offset=D, sizing=COARSE, order=2)


def test_the_benchmark_builds_and_validates(built):
    assert built.msh_path.exists() and built.manifest_path.exists()
    assert built.validation.ok
    assert built.validation.negative_jacobians == 0
    assert built.validation.negative_cells == 0
    assert built.validation.inward_faces == 0
    assert built.counts["layers"] == 2


def test_the_inclusion_is_the_boundary_tagged_first(built):
    """Attribute 1 is the inclusion, and it is where the offset puts it."""
    card = sc.read(built.manifest_path)
    inner, outer = card.interfaces
    assert inner["attribute"] == 1 and outer["attribute"] == 2
    assert outer["mean_radius_nd"] == pytest.approx(B, abs=1e-9)
    # 1% for a node average over an unstructured mesh; wrong tagging
    # would be out by 130%, which is the failure this is here to catch.
    assert inner["mean_radius_nd"] == pytest.approx(mean_radius(A, D), rel=0.01)


def test_the_concentric_case_is_the_same_generator(tmp_path):
    """Offset zero is legitimate, and then the mean radius is exactly a."""
    result = build_offset_mesh(tmp_path / "concentric", inner_radius=A,
                                  outer_radius=B, offset=0.0, sizing=COARSE)
    assert result.validation.ok
    inner = sc.read(result.manifest_path).interfaces[0]
    assert inner["mean_radius_nd"] == pytest.approx(A, abs=1e-9)


def test_two_dimensions_goes_through_the_same_path(tmp_path):
    result = build_offset_mesh(tmp_path / "two_disc", inner_radius=A,
                                  outer_radius=B, offset=D, dimension=2,
                                  sizing=UniformInterfaces(0.08, 0.2, 0.3))
    assert result.validation.ok
    card = sc.read(result.manifest_path)
    assert card.model["geometry"]["kind"] == "two_disc"
    assert card.mesh["dimension"] == 2


def test_the_manifest_says_where_the_inclusion_is(built):
    """The radii describe its size; only the offset says where it sits."""
    geometry = json.loads(built.manifest_path.read_text())["model"]["geometry"]
    assert geometry == {"kind": "two_sphere", "inner_radius_nd": A,
                        "outer_radius_nd": B, "offset_nd": D}
    card = sc.read(built.manifest_path)
    assert card.layers[0]["r_outer_nd"] == A       # the sphere it lies inside
    assert card.layers[0]["fields"] == []       # synthetic geometry, no model
    assert card.mapping is None and card.delivery == "physical"


def test_names_reach_the_mesh_file(built):
    from planetmodel.mesh3d._session import session
    from planetmodel.mesh3d._writer import read_groups
    with session(name="names"):
        groups = read_groups(built.msh_path)
    assert groups[3] == {1: "inclusion", 2: "matrix"}
    assert groups[2] == {1: "inclusion_boundary", 2: "surface"}


# ------------------------------------------------------------- refusals

def test_an_inclusion_touching_the_boundary_is_refused(tmp_path):
    with pytest.raises(ValueError, match="strictly enclosed"):
        build_offset_mesh(tmp_path / "x", inner_radius=0.3,
                             outer_radius=1.0, offset=0.75, sizing=COARSE)


def test_an_inclusion_larger_than_the_body_is_refused(tmp_path):
    with pytest.raises(ValueError, match="inner_radius < outer_radius"):
        build_offset_mesh(tmp_path / "x", inner_radius=1.2,
                             outer_radius=1.0, sizing=COARSE)


def test_sizing_is_not_optional(tmp_path):
    with pytest.raises(ValueError, match="no sizing given"):
        build_offset_mesh(tmp_path / "x", inner_radius=A, outer_radius=B)


def test_a_manifest_never_carries_a_nan(tmp_path):
    """NaN is not JSON, and a C++ reader would choke on the file."""
    card = sc.MeshManifest(model={"rref_m": float("nan")})
    with pytest.raises(ValueError, match="[Nn]a[Nn]|not allowed"):
        sc.write(tmp_path / "bad", card)


def test_a_mapping_without_a_validity_report_writes_null_not_nan(tmp_path):
    """The margin is unknown, which JSON spells null."""
    from planetmodel.io.manifest import _finite
    assert _finite(float("nan")) is None
    assert _finite(0.5) == 0.5
    assert json.dumps({"margin": _finite(float("nan"))}, allow_nan=False)


def test_an_offset_inclusion_is_oriented_about_its_own_centre(tmp_path):
    """With offset > radius the far cap points at the origin while
    pointing out of the inclusion; orienting about the origin reversed it."""
    import gmsh
    a, d = 0.3, 0.35
    result = build_offset_mesh(tmp_path / "off", inner_radius=a,
                               outer_radius=1.0, offset=d, sizing=COARSE,
                               order=1)
    with session(name="check"):
        gmsh.open(str(result.msh_path))
        pos = node_positions()
        # Physical group 1 of dimension 2 is the inclusion boundary.
        (face,) = gmsh.model.getEntitiesForPhysicalGroup(2, 1)
        _, about_centre = outward_dots(int(face), pos, centre=(0.0, 0.0, d))
        _, about_origin = outward_dots(int(face), pos)
    assert np.all(about_centre > 0.0)
    assert np.any(about_origin < 0.0)          # so the origin test would lie



def test_sizing_is_assigned_by_index_not_by_order(tmp_path):
    sizing = PerInterface({"surface": InterfaceSizing(0.25, 0.3, 0.3),
                           "inclusion_boundary": InterfaceSizing(0.1, 0.3, 0.3)})
    result = build_offset_mesh(tmp_path / "per", inner_radius=0.3,
                                  outer_radius=1.0, offset=0.2, sizing=sizing,
                                  order=1)
    per = sc.read(result.manifest_path).sizing["per_interface"]
    assert per[0]["attribute"] == 1 and per[0]["size_nd"] == 0.1
    assert sc.read(result.manifest_path).sizing["policy"] == "per_interface"



def test_an_offset_mesh_in_si_records_its_divisor(tmp_path):
    result = build_offset_mesh(tmp_path / "si", inner_radius=0.3e6,
                                  outer_radius=1.0e6, offset=0.2e6,
                                  sizing=UniformInterfaces(0.12e6, 0.25e6, 0.3e6),
                                  order=1, rref=1.0e6)
    units = sc.read(result.manifest_path).model["units"]
    assert units["rref_m"] == 1.0e6 and units["geometry_divisor"] == 1.0e6

