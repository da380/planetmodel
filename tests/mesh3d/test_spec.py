"""The specification objects and the sizing rules."""
import dataclasses

import pytest

from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (AngularResolution, InterfaceSizing, MeshResult,
                                MeshSpec, PerInterface, Shell, UniformInterfaces,
                                ValidationReport)
from planetmodel.mesh3d.spec import QUALITY_FLOOR

from conftest import COARSE, full_geometry, hollow_geometry

pytestmark = pytest.mark.gmsh


# --------------------------------------------------------------- shells

def test_shell_takes_exactly_one_of_ratio_and_radius():
    assert Shell(ratio=0.2).outer_radius(1.0) == pytest.approx(1.2)
    assert Shell(radius=1.5).outer_radius(1.0) == 1.5
    with pytest.raises(ValueError, match="exactly one"):
        Shell()
    with pytest.raises(ValueError, match="exactly one"):
        Shell(ratio=0.2, radius=1.5)
    with pytest.raises(ValueError, match="ratio must be positive"):
        Shell(ratio=-0.1)
    with pytest.raises(ValueError, match="radius must be positive"):
        Shell(radius=0.0)


def test_shells_chain_outward_from_the_geometry():
    spec = MeshSpec(full_geometry(), COARSE,
                    shells=[Shell(ratio=0.5, name="a"), Shell(ratio=0.5, name="b")])
    assert spec.shell_radii == pytest.approx((1.5, 2.25))
    assert spec.outer_radius == pytest.approx(2.25)
    assert spec.effective_divisor == pytest.approx(2.25)
    assert [lay.name for lay in spec.layers] == ["core", "mantle", "crust", "a", "b"]
    assert [f.name for f in spec.interfaces] == ["cmb", "moho", "surface", None, None]
    assert [f.between for f in spec.interfaces] == \
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, -1)]
    assert [lay.interval for lay in spec.layers][-1] == pytest.approx((1.5, 2.25))


def test_shell_refusals():
    g = full_geometry()
    with pytest.raises(ValueError, match="not above the boundary"):
        MeshSpec(g, COARSE, shells=[Shell(radius=0.9)])
    with pytest.raises(ValueError, match="not above the boundary"):
        MeshSpec(g, COARSE, shells=[Shell(radius=1.5, name="a"),
                                    Shell(radius=1.4, name="b")])
    with pytest.raises(ValueError, match="unique"):
        MeshSpec(g, COARSE, shells=[Shell(ratio=0.2), Shell(ratio=0.2)])
    with pytest.raises(TypeError, match="Shell instances"):
        MeshSpec(g, COARSE, shells=[0.2])


def test_a_hollow_geometry_keeps_its_inner_interface_with_shells():
    spec = MeshSpec(hollow_geometry(), COARSE, shells=[Shell(ratio=0.2)])
    assert [f.between for f in spec.interfaces] == [(-1, 0), (0, 1), (1, 2), (2, -1)]
    assert spec.domain.is_hollow
    assert spec.domain.skeleton.boundaries[0] == 0.5


def test_without_shells_the_domain_is_the_geometry():
    g = full_geometry()
    spec = MeshSpec(g, COARSE)
    assert spec.domain is g
    assert spec.outer_radius == 1.0
    assert spec.effective_divisor == 1.0
    assert MeshSpec(g, COARSE, divisor=2.0).effective_divisor == 2.0


# -------------------------------------------------------------- the spec

def test_mesh_spec_validates():
    g = full_geometry()
    MeshSpec(g, COARSE)
    with pytest.raises(TypeError, match="must be a Geometry"):
        MeshSpec(Skeleton([0.0, 1.0]), COARSE)
    with pytest.raises(TypeError, match="callable"):
        MeshSpec(g, 0.1)
    with pytest.raises(ValueError, match="dimension must be"):
        MeshSpec(g, COARSE, dimension=1)
    with pytest.raises(ValueError, match="element order"):
        MeshSpec(g, COARSE, order=4)
    with pytest.raises(ValueError, match="delivery must be"):
        MeshSpec(g, COARSE, delivery="halfway")
    with pytest.raises(ValueError, match="divisor must be positive"):
        MeshSpec(g, COARSE, divisor=0.0)


def test_mesh_spec_is_frozen_with_defaults():
    spec = MeshSpec(full_geometry(), COARSE)
    assert (spec.dimension, spec.order, spec.delivery) == (3, 2, "physical")
    assert spec.shells == () and spec.meta == {} and spec.validate
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.order = 3


def test_a_plain_function_is_a_valid_sizing_rule():
    def mine(interfaces, outer_radius):
        return {f.index: InterfaceSizing(0.1, 0.2, 0.2) for f in interfaces}

    spec = MeshSpec(full_geometry(), mine)
    sizes = spec.sizing(spec.interfaces, spec.outer_radius)
    assert len(sizes) == 3 and all(s.size == 0.1 for s in sizes.values())


# -------------------------------------------------------- sizing values

def test_interface_sizing_validates_and_scales():
    with pytest.raises(ValueError, match="size must be positive"):
        InterfaceSizing(-1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="far_size .* is smaller"):
        InterfaceSizing(10.0, 1.0, 3.0)
    s = InterfaceSizing(1e3, 1e4, 1e5).scaled(1e-3)
    assert (s.size, s.far_size, s.decay_width) == (1.0, 10.0, 100.0)


# --------------------------------------------------------- sizing rules

def test_angular_resolution_scales_with_radius():
    faces = full_geometry().interfaces
    rule = AngularResolution(h_ref=0.1, h_far=0.5)
    sizes = rule(faces, 1.0)
    assert sizes[0].size == pytest.approx(0.04)          # 0.1 * 0.4 / 1.0
    assert sizes[2].size == pytest.approx(0.1)
    assert sizes[0].decay_width == pytest.approx(0.2 * 0.4)
    # r_ref defaults to the outer radius given
    assert AngularResolution(0.1, 0.5)(faces, 2.0)[2].size == pytest.approx(0.05)
    assert AngularResolution(0.1, 0.5, r_ref=1.0)(faces, 2.0)[2].size == \
        pytest.approx(0.1)


def test_angular_resolution_never_exceeds_the_far_size():
    faces = full_geometry().interfaces
    for s in AngularResolution(h_ref=10.0, h_far=0.2)(faces, 1.0).values():
        assert s.size <= s.far_size


def test_uniform_interfaces_gives_one_size():
    sizes = UniformInterfaces(0.1, 0.2, 0.3)(full_geometry().interfaces, 1.0)
    assert {(s.size, s.far_size, s.decay_width) for s in sizes.values()} == \
        {(0.1, 0.2, 0.3)}


def test_per_interface_overrides_by_name_and_index():
    faces = full_geometry().interfaces
    fine = InterfaceSizing(0.02, 0.2, 0.2)
    sizes = PerInterface({"moho": fine, 0: fine}, base=COARSE)(faces, 1.0)
    assert sizes[1] is fine and sizes[0] is fine
    assert sizes[2].size == COARSE.h_min
    with pytest.raises(KeyError, match="no interface named"):
        PerInterface({"ocean": fine}, base=COARSE)(faces, 1.0)
    with pytest.raises(ValueError, match="no sizing for interfaces"):
        PerInterface({0: fine})(faces, 1.0)


def test_sizing_rules_are_frozen_and_comparable():
    assert AngularResolution(0.1, 0.5) == AngularResolution(0.1, 0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        AngularResolution(0.1, 0.5).h_ref = 1.0


# --------------------------------------------------------------- results

def test_validation_report_raises_with_every_failure():
    rep = ValidationReport(dimension=3)
    assert rep.ok and rep.raise_if_failed() is rep
    rep.failures += ["one", "two"]
    with pytest.raises(ValueError, match="one\n  - two"):
        rep.raise_if_failed()
    assert "2 FAILED" in repr(rep)
    assert 0.0 < QUALITY_FLOOR < 1.0


def test_mesh_result_constructs_by_hand(tmp_path):
    r = MeshResult(msh_path=tmp_path / "a.msh", manifest_path=tmp_path / "a.json",
                   geometry=None, counts={"elements": 3, "layers": 1},
                   validation=ValidationReport(dimension=2), timings={})
    assert r.spec is None and r.mapping is None and r.divisor == 1.0
    assert repr(r) == "MeshResult(a.msh, 3 elements, 1 layers)"


def test_the_geometry_type_is_the_core_one():
    assert isinstance(MeshSpec(full_geometry(), COARSE).geometry, Geometry)
