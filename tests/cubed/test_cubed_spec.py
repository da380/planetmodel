"""A cubed-sphere spec and a squared circle's: what they hold and
refuse, the computational domain they share with a MeshSpec, and the
spheres and the blend they give."""
import dataclasses
import subprocess
import sys

import numpy as np
import pytest

from cubed_cases import COARSE, SPECS, ball, shell
from planetmodel import Geometry, Skeleton, cartesian
from planetmodel.mesh3d import (AngularResolution, CubedSphereSpec, InterfaceSizing,
                                MeshSpec, PerInterface, Shell, SquaredCircleSpec,
                                UniformInterfaces)
from planetmodel.mesh3d.spec import CubedSpec
from planetmodel.mesh3d._cubed import blend_elongation, cubed_topology


def test_what_a_spec_holds():
    spec = CubedSphereSpec(ball(), 4, sizing=COARSE)
    assert (spec.dimension, spec.order, spec.cube, spec.elongation) == (3, 2, 0.45, 4.0)
    assert spec.outer_boundary == "free" and spec.validate and spec.meta == {}
    assert not spec.hollow and spec.first_sphere == 0.4
    assert spec.half_width == pytest.approx(0.18)
    assert spec.outer_radius == 1.0 and spec.knots is None
    assert [lay.name for lay in spec.layers] == ["core", "mantle", "crust"]
    assert [f.name for f in spec.interfaces] == ["cmb", "moho", "surface"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        spec.edge = 8
    hollow = CubedSphereSpec(shell(), 4, sizing=COARSE)
    assert hollow.hollow and hollow.first_sphere == 0.5
    assert hollow.half_width is None and hollow.blend is None


def test_the_dimension_is_the_class_s():
    """A cubed sphere is of three dimensions and a squared circle of two,
    with no keyword to say otherwise; what they share is not made."""
    g = ball()
    cubed = CubedSphereSpec(g, 4, sizing=COARSE)
    squared = SquaredCircleSpec(g, 4, sizing=COARSE)
    assert (cubed.dimension, squared.dimension) == (3, 2)
    assert isinstance(cubed, CubedSpec) and isinstance(squared, CubedSpec)
    assert "dimension" not in {f.name for f in dataclasses.fields(squared)}
    assert cubed != squared and cubed == CubedSphereSpec(g, 4, sizing=COARSE)
    assert cubed.cells == 4 ** 3 + 6 * 4 ** 2 * cubed.levels
    assert squared.cells == 4 ** 2 + 4 * 4 * squared.levels
    for make in (CubedSphereSpec, SquaredCircleSpec):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            make(ball(), 4, sizing=COARSE, dimension=2)
    with pytest.raises(TypeError, match="made as a CubedSphereSpec"):
        CubedSpec(ball(), 4, sizing=COARSE)


def test_the_domain_is_a_mesh_spec_s():
    shells = (Shell(ratio=0.5), Shell(radius=2.0, name="far"))
    cubed = CubedSphereSpec(ball(), 4, sizing=COARSE, shells=shells)
    tets = MeshSpec(ball(), COARSE, shells=shells)
    assert cubed.shell_radii == tets.shell_radii == (1.5, 2.0)
    assert cubed.shell_names == tets.shell_names == ["buffer_1", "far"]
    assert cubed.outer_radius == tets.outer_radius == 2.0
    assert cubed.layers == tets.layers and cubed.interfaces == tets.interfaces
    np.testing.assert_array_equal(cubed.domain.skeleton.boundaries,
                                  [0.0, 0.4, 0.8, 1.0, 1.5, 2.0])
    assert cubed.spheres[-1] == 2.0


def test_the_spheres_the_sizing_gives():
    spec = CubedSphereSpec(ball(), 4, sizing=UniformInterfaces(0.1, 0.1, 0.1))
    np.testing.assert_allclose(spec.spheres, [0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               atol=1e-13)
    assert {0.4, 0.8, 1.0} <= set(spec.spheres.tolist())
    assert not spec.spheres.flags.writeable and not spec.blend.flags.writeable
    # the blend, from the cube's half-width 0.18 to 0.4, takes three cells
    assert spec.blend.size == 4
    assert spec.levels == 3 + 6 and spec.cells == 4 ** 3 + 6 * 16 * 9
    assert spec.sizes[0] == InterfaceSizing(0.1, 0.1, 0.1)
    hollow = SquaredCircleSpec(shell(), 3, sizing=UniformInterfaces(0.1, 0.1, 0.1))
    np.testing.assert_allclose(hollow.spheres, [0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                               atol=1e-13)
    assert hollow.levels == 5 and hollow.cells == 4 * 3 * 5


@pytest.mark.parametrize("dimension", [3, 2])
@pytest.mark.parametrize("make", [ball, shell])
def test_the_cells_counted_are_the_topology_s(dimension, make):
    spec = SPECS[dimension](make(), 5, sizing=COARSE, shells=(Shell(ratio=0.3),))
    t = cubed_topology(spec.edge, spec.levels, dimension=dimension,
                       hollow=spec.hollow)
    assert spec.cells == t.cells.shape[0]


def test_fine_by_the_surface_and_coarse_below():
    rule = PerInterface({"surface": InterfaceSizing(0.02, 0.3, 0.3)},
                        base=UniformInterfaces(0.3, 0.3, 0.3))
    spec = CubedSphereSpec(ball(), 4, sizing=rule)
    crust = spec.spheres[spec.spheres >= 0.8]
    mantle = spec.spheres[(spec.spheres >= 0.4) & (spec.spheres <= 0.8)]
    assert np.diff(crust).min() < 0.05 and np.diff(mantle).max() > 0.1
    assert np.all(np.diff(np.diff(crust)) < 0.0)          # thinner towards the top


def test_knots_given_outright():
    spec = CubedSphereSpec(ball(), 8, knots=[0.4, 0.6, 0.8 + 1e-13, 1.0],
                           elongation=None)
    assert spec.knots == (0.4, 0.6, 0.8, 1.0)             # on the interface exactly
    assert spec.spheres.tolist() == [0.4, 0.6, 0.8, 1.0]
    assert spec.sizes is None and spec.blend.tolist() == [0.0, 1.0]
    capped = CubedSphereSpec(ball(), 32, knots=[0.4, 0.8, 1.0])
    assert capped.blend.size > 2
    assert blend_elongation(32, capped.blend, half_width=capped.half_width,
                            radius=0.4) <= capped.elongation
    with pytest.raises(ValueError, match="none is at r = 0.8"):
        CubedSphereSpec(ball(), 4, knots=[0.4, 0.7, 1.0])
    with pytest.raises(ValueError, match="none is at r = 1.5"):
        CubedSphereSpec(ball(), 4, knots=[0.4, 0.8, 1.0], shells=(Shell(ratio=0.5),))
    with pytest.raises(ValueError, match="to the outer boundary at r = 1,"):
        CubedSphereSpec(ball(), 4, knots=[0.4, 0.8, 1.0, 1.2])
    with pytest.raises(ValueError, match="increasing radii"):
        CubedSphereSpec(ball(), 4, knots=[0.4, 0.8, 0.8, 1.0])


@pytest.mark.parametrize("dimension", [3, 2])
def test_knots_below_the_first_sphere_divide_the_blend(dimension):
    """Each is the radius of a level along the ray through the centre of
    a face, from the cube's half-width 0.18 to the first sphere at 0.4;
    the blend is then as given, whatever the cap would have added."""
    make = SPECS[dimension]
    spec = make(ball(), 32, knots=[0.2, 0.29, 0.4, 0.8, 1.0])
    assert spec.knots == (0.2, 0.29, 0.4, 0.8, 1.0)
    assert spec.spheres.tolist() == [0.4, 0.8, 1.0]
    np.testing.assert_allclose(spec.blend, [0.0, 0.02 / 0.22, 0.5, 1.0], atol=1e-14)
    assert spec.levels == 3 + 2 and not spec.blend.flags.writeable
    # the cap would have divided it further
    capped = make(ball(), 32, knots=[0.4, 0.8, 1.0])
    assert capped.blend.size > spec.blend.size
    assert blend_elongation(32, spec.blend, half_width=0.18, radius=0.4,
                            dimension=dimension) > spec.elongation
    # a ball of one layer: the knots are all the radial structure there is
    whole = make(Geometry(Skeleton([0.0, 1.0])), 4, knots=[0.6, 0.8, 1.0])
    assert whole.spheres.tolist() == [1.0] and whole.levels == 3
    np.testing.assert_allclose(whole.blend, [0.0, 0.15 / 0.55, 0.35 / 0.55, 1.0],
                               atol=1e-14)
    for lowest in (0.18, 0.1):
        with pytest.raises(ValueError, match="knots lie above the cube, whose "
                                             "surface is at r = 0.18"):
            make(ball(), 4, knots=[lowest, 0.4, 0.8, 1.0])
    with pytest.raises(ValueError, match="none is at r = 0.4"):
        make(ball(), 4, knots=[0.3, 0.8, 1.0])
    # a level given twice to the geometry's tolerance, or one on the cube
    for knots in ([0.4, 0.8 - 1e-12, 0.8 + 1e-12, 1.0],
                  [0.3, 0.4 - 1e-13, 0.4, 0.8, 1.0],
                  [0.4, 0.6, 0.8, np.nextafter(0.8, 1.0), 1.0]):
        with pytest.raises(ValueError, match="are one radius to the geometry's "
                                             "tolerance"):
            make(ball(), 4, knots=knots)
    with pytest.raises(ValueError, match="knots lie above the cube"):
        make(ball(), 4, knots=[0.18 + 1e-14, 0.4, 0.8, 1.0])
    with pytest.raises(ValueError, match="a shell run from its inner boundary at "
                                         "r = 0.5, got one at r = 0.3"):
        make(shell(), 4, knots=[0.3, 0.5, 0.8, 1.0])


def test_what_a_spec_refuses():
    flat = cartesian.Geometry(Skeleton([0.0, 1.0]),
                              cartesian.Box([(0.0, 1.0), (0.0, 1.0)]))
    with pytest.raises(TypeError, match="a spherical geometry"):
        CubedSphereSpec(flat, 4, sizing=COARSE)
    for edge in (0, 2.5, True):
        with pytest.raises(ValueError, match="edge must be a whole number"):
            CubedSphereSpec(ball(), edge, sizing=COARSE)
    with pytest.raises(ValueError, match="exactly one of sizing and knots"):
        CubedSphereSpec(ball(), 4)
    with pytest.raises(ValueError, match="exactly one of sizing and knots"):
        CubedSphereSpec(ball(), 4, sizing=COARSE, knots=[0.4, 0.8, 1.0])
    with pytest.raises(TypeError, match="callable sizing rule"):
        CubedSphereSpec(ball(), 4, sizing=0.1)
    with pytest.raises(ValueError, match="at least 1"):
        CubedSphereSpec(ball(), 4, sizing=COARSE, order=0)
    with pytest.raises(ValueError, match="outer_boundary must be one of"):
        CubedSphereSpec(ball(), 4, sizing=COARSE, outer_boundary="round")
    with pytest.raises(TypeError, match="Shell instances"):
        CubedSphereSpec(ball(), 4, sizing=COARSE, shells=(1.5,))
    with pytest.raises(ValueError, match="cube must lie between"):
        CubedSphereSpec(ball(), 4, sizing=COARSE, cube=0.6)
    assert SquaredCircleSpec(ball(), 4, sizing=COARSE, cube=0.6).cube == 0.6
    with pytest.raises(ValueError, match="elongation must be positive"):
        CubedSphereSpec(ball(), 4, sizing=COARSE, elongation=0.0)


def test_a_sizing_at_the_wrong_scale_is_refused():
    spec = CubedSphereSpec(ball().scaled(6.371e6), 4,
                           sizing=UniformInterfaces(0.2, 0.2, 0.2))
    with pytest.raises(ValueError, match="not a resolution choice"):
        spec.spheres
    partial = CubedSphereSpec(ball(), 4, sizing=lambda faces, outer: {
        0: InterfaceSizing(0.2, 0.2, 0.2)})
    with pytest.raises(ValueError, match=r"no sizing for interfaces \[1, 2\]"):
        partial.spheres


def test_a_rule_that_scales_with_radius():
    spec = CubedSphereSpec(ball(), 8, sizing=AngularResolution(0.1, 0.3))
    steps = np.diff(spec.spheres)
    assert spec.spheres[0] == 0.4 and steps[0] < steps[-1] * 1.5
    assert spec.sizes[0].size == pytest.approx(0.04)


def test_what_needs_pymfem_says_so(tmp_path):
    """With PyMFEM unimportable a spec is still made and its knots read,
    and the builder and the reader of an MFEM file are refused, naming
    the `mfem` extra."""
    code = """
import sys
sys.modules["mfem"] = None
from planetmodel import Geometry, Skeleton
from planetmodel.mesh3d import (CubedSphereSpec, UniformInterfaces,
                                build_cubed_sphere, read_mesh)
spec = CubedSphereSpec(Geometry(Skeleton([0.0, 0.5, 1.0])), 2,
                       sizing=UniformInterfaces(0.3, 0.3, 0.3))
assert spec.cells == 8 + 24 * spec.levels
for attempt in ("build_cubed_sphere(spec, 'm')", "read_mesh('m.mesh')"):
    try:
        exec(attempt)
    except ImportError as error:
        assert "planetmodel[mfem]" in str(error), error
    else:
        raise AssertionError(attempt + " needed no PyMFEM")
"""
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                       cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    assert not list(tmp_path.iterdir())
