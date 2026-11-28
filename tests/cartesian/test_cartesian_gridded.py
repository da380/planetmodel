"""Stored fields that vary laterally in a box: the flat gridded field,
function and vertical displacement.

A flat gridded field passes the field contract for ranks 0 to 2, every
rule and both radial rules, over a box and a rectangle; equals the field
it was sampled from at every node; reproduces a band-limited field
exactly at points by the Fourier rule; has the exact mean of its rules;
continues its end elements and rescales.  A gridded function (a DEM on
its own grid) is a relief whose displacement, stretch and geometry pass
their contracts; a gridded vertical displacement reproduces the stretch
it samples and passes them too.
"""
import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import DENSITY, SCALAR, STRESS, VECTOR, RadialMesh, Skeleton
from planetmodel import cartesian as flat
from planetmodel.cartesian import FLAT
from planetmodel.cartesian.mapping import VerticalStretch
from planetmodel.character import Character
from planetmodel.mesh1d import Mesh1D
from planetmodel.testing import (check_field, check_geometry, check_mapping,
                                 check_vertical_displacement)

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
RECT = flat.Rectangle((0.0, 2.0))
INTERVAL = (0.2, 1.0)
TENSOR = Character(2, 0, voigt=False)


def smooth_for(char, *, rectangle=False):
    """A smooth periodic field of the character over BOX (or RECT), its
    components local."""
    shape = char.voigt_shape or char.component_shape

    def fn(d, x, y):
        d, x, y = np.broadcast_arrays(d, x, y)
        u, v = np.pi * x, (0.0 * y if rectangle else np.pi * y)
        base = (1.0 + d ** 2) * np.exp(0.3 * np.sin(u) - 0.2 * np.cos(v)
                                       + 0.1 * np.sin(u + v))
        if not char.rank:
            return base
        k = int(np.prod(shape))
        comps = [base * (1.0 + 0.1 * i) + 0.05 * i * np.cos(u) * d for i in range(k)]
        return np.stack(comps, axis=-1).reshape(d.shape + shape)
    return flat.AnalyticField(INTERVAL, fn, character=char)


def nodes_of(g):
    x, y = g.grid.nodes
    d = g.radii.reshape(g.radii.shape + (1,) * g.grid.dimension)
    return d, x[None, None], y[None, None]


@pytest.mark.parametrize("radial", ["polynomial", "linear"])
@pytest.mark.parametrize("lateral", ["linear", "cubic", "fourier"])
@pytest.mark.parametrize("char", [SCALAR, VECTOR, STRESS, TENSOR], ids=str)
def test_the_field_contract_for_every_rank_and_rule(char, lateral, radial):
    g = flat.GriddedField.sampled(smooth_for(char), flat.Grid(BOX, (10, 8)),
                                  breaks=[0.2, 0.6, 1.0], ngll=4, radial=radial,
                                  lateral=lateral)
    check_field(g)
    assert not g.is_radial and g.radial == radial and g.lateral == lateral
    assert g.coordinates is FLAT


@pytest.mark.parametrize("lateral", ["linear", "cubic", "fourier"])
@pytest.mark.parametrize("char", [SCALAR, VECTOR], ids=str)
def test_a_rectangle_s_gridded_field_holds_to_the_contract(char, lateral):
    g = flat.GriddedField.sampled(smooth_for(char, rectangle=True),
                                  flat.Grid(RECT, 12), breaks=[0.2, 1.0], ngll=3,
                                  lateral=lateral)
    check_field(g)
    assert g.values.shape == (1, 3, 12) + g.stored_shape
    assert np.array_equal(g(0.5, 0.3, -4.0), g(0.5, 0.3, 9.0))


@pytest.mark.parametrize("grid", [flat.Grid(BOX, (9, 7)), flat.Grid(RECT, 11)],
                         ids=repr)
@pytest.mark.parametrize("char", [SCALAR, VECTOR, STRESS], ids=str)
def test_sampled_equals_its_source_at_every_node(grid, char):
    src = smooth_for(char, rectangle=grid.dimension == 1)
    g = flat.GriddedField.sampled(src, grid, breaks=[0.2, 0.5, 1.0], ngll=3,
                                  lateral="cubic", name="q")
    d, x, y = nodes_of(g)
    want = src.evaluate(d, x, y)
    assert np.array_equal(g.values, want)
    assert np.allclose(g.evaluate(d, x, y), want, rtol=0, atol=1e-12)
    assert g.name == "q" and g.ngll == 3 and np.array_equal(g.breaks, [0.2, 0.5, 1.0])
    assert g.grid is grid and not g.values.flags.writeable


def band_limited(grid, char=SCALAR):
    """A Fourier field of the grid's band, cubic in depth on one piece."""
    t = grid.transform
    shape = char.voigt_shape or char.component_shape
    c = np.random.default_rng(3).standard_normal((4, 1, t.size) + shape)
    return flat.FourierField(INTERVAL, grid, PPoly(c, np.array(INTERVAL)),
                             character=char)


@pytest.mark.parametrize("grid", [flat.Grid(BOX, (8, 7)), flat.Grid(RECT, 9)],
                         ids=repr)
@pytest.mark.parametrize("char", [SCALAR, VECTOR], ids=str)
def test_a_band_limited_field_is_reproduced_exactly_on_the_grid_and_at_points(
        grid, char):
    """Sampled on its own grid at four GLL nodes (cubic in depth) and read
    by the Fourier rule: the field itself everywhere, to finufft's
    accuracy, and its mean exactly."""
    f = band_limited(grid, char)
    g = flat.GriddedField.sampled(f, grid, ngll=4, lateral="fourier")
    rng = np.random.default_rng(4)
    d = rng.uniform(0.2, 1.0, 400)
    x, y = rng.uniform(-6, 6, 400), rng.uniform(-6, 6, 400)
    want = f.evaluate(d, x, y)
    assert np.abs(g.evaluate(d, x, y) - want).max() < 1e-11 * np.abs(want).max()
    dd = np.linspace(0.2, 1.0, 7)
    scale = np.abs(want).max()
    assert np.allclose(g.horizontal_mean().evaluate(dd, 0, 0),
                       f.horizontal_mean().evaluate(dd, 0, 0), rtol=0,
                       atol=1e-12 * scale)
    assert np.allclose(f.on_grid(dd, grid), g.evaluate(*(
        dd.reshape((-1,) + (1,) * grid.dimension), *(v[None] for v in grid.nodes))),
        rtol=0, atol=1e-11 * scale)


@pytest.mark.parametrize("radial", ["polynomial", "linear"])
@pytest.mark.parametrize("lateral", ["linear", "cubic", "fourier"])
def test_the_mean_is_the_mean_of_the_rules(lateral, radial):
    """At every node the mean of the values, joined by the radial rule:
    against a fine quadrature of the field as its rules read it, at depths
    between the nodes too; a vector's mean is its components'."""
    grid = flat.Grid(BOX, (7, 5))
    for char in (SCALAR, VECTOR):
        g = flat.GriddedField.sampled(smooth_for(char), grid, breaks=[0.2, 0.55, 1.0],
                                      ngll=3, radial=radial, lateral=lateral)
        m = flat.horizontal_mean(g)
        assert isinstance(m, flat.DepthField) and m.character == char
        fine = flat.Grid(BOX, (7 * 24, 5 * 24))
        d = np.array([0.21, 0.37, 0.55, 0.8, 1.0])
        x, y = fine.nodes
        values = g.evaluate(d[:, None, None], x[None], y[None])
        quad = fine.mean(np.moveaxis(values, (1, 2), (-2, -1)))
        assert np.allclose(m.evaluate(d, 0.0, 0.0), quad, rtol=0,
                           atol=1e-12 * np.abs(values).max())
        pieces = m.radial_pieces()
        assert np.array_equal(pieces[0], g.radial_pieces()[0])


def test_on_interval_continues_the_end_elements_and_rescaled_scales_all():
    grid = flat.Grid(BOX, (6, 6))
    g = flat.GriddedField.sampled(smooth_for(SCALAR), grid, breaks=[0.2, 0.5, 1.0],
                                  ngll=5, lateral="cubic")
    wide = g.on_interval(0.1, 1.1)
    assert wide.interval == (0.1, 1.1)
    with pytest.raises(ValueError, match="depth 1.05 is outside the interval"):
        g(1.05, 1.0, 0.0)
    i, j = 3, 4
    pp = Mesh1D(g.breaks, ngll=5).to_ppoly(g.values[:, :, i, j])
    d = np.array([0.1, 0.15, 1.05, 1.1])
    assert np.allclose(wide(d, grid.x[i], grid.y[j]), pp(d), atol=1e-12)
    s = g.rescaled(k=2.0, v=3.0)
    assert np.allclose(s.interval, (0.4, 2.0)) and np.allclose(s.breaks, 2 * g.breaks)
    assert s.grid == grid.scaled(2.0)
    rng = np.random.default_rng(5)
    d, x, y = rng.uniform(0.2, 1, 20), rng.uniform(0, 2, 20), rng.uniform(-1, 1, 20)
    assert np.allclose(s(2 * d, 2 * x, 2 * y), 3 * g(d, x, y), atol=1e-12)
    assert g.renamed("x").name == "x" and g.name is None
    assert "cubic on Grid(6 x 6" in repr(g)


def test_a_gridded_field_in_a_flat_model_validates():
    model = flat.CrustOverMantle(flat.Box([(0.0, 4e5), (0.0, 2e5)]))
    iv = model.layer(1).interval
    grid = flat.Grid(model.geometry.domain, (16, 8))
    rho = flat.AnalyticField(iv, lambda d, x, y: 3300.0 + 20.0 * np.sin(
        2 * np.pi * x / 4e5) * np.cos(2 * np.pi * y / 2e5) + 0.0 * d, character=DENSITY)
    g = flat.GriddedField.sampled(rho, grid, ngll=3, lateral="cubic")
    m = model.with_field(1, "rho", g, replace=True)
    m.validate()
    assert np.allclose((g * 2.0)(iv[0] + 1, 1e5, 5e4), 2.0 * g(iv[0] + 1, 1e5, 5e4))


def test_refusals_name_what_is_wrong():
    grid = flat.Grid(BOX, (4, 5))
    v = np.ones((2, 3, 4, 5))
    with pytest.raises(ValueError, match="span"):
        flat.GriddedField(INTERVAL, [0.2, 0.5, 0.9], grid, v)
    with pytest.raises(ValueError, match="strictly increasing 1-d array of at least "
                                         "two values of the depth"):
        flat.GriddedField(INTERVAL, [0.2, 1.0, 0.8], grid, v)
    with pytest.raises(ValueError, match=r"values must have shape \(nelem, ngll, nx, "
                                         r"ny\)"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], grid, v[..., :-1])
    with pytest.raises(ValueError, match="ngll >= 2"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], grid, v[:, :1])
    with pytest.raises(ValueError, match="radial"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], grid, v, radial="spline")
    with pytest.raises(ValueError, match="lateral must be one of"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], grid, v, lateral="harmonic")
    with pytest.raises(TypeError, match="real"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], grid, v + 0j)
    with pytest.raises(ValueError, match="finite"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], grid, v * np.nan)
    with pytest.raises(TypeError, match="cartesian.Grid"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], (4, 5), v)
    with pytest.raises(ValueError, match=r"\(nelem, ngll, nx\)"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], flat.Grid(RECT, 4), v)
    with pytest.raises(ValueError, match="at least 3 nodes"):
        flat.GriddedField(INTERVAL, [0.2, 0.6, 1.0], flat.Grid(BOX, (4, 2)),
                          np.ones((2, 3, 4, 2)), lateral="cubic")


# -- the gridded function ---------------------------------------------------------

def dem(x, y):
    """A range and a basin, smooth and periodic over the unit box: heights,
    positive up."""
    u, v = 2 * np.pi * x, 2 * np.pi * (y + 0.5)
    return (0.05 * np.exp(np.cos(u - 3.6) - 1.0) * (1 + 0.3 * np.cos(v))
            - 0.01 * np.sin(u))


def dem_gradient(x, y):
    u, v = 2 * np.pi * x, 2 * np.pi * (y + 0.5)
    e = 0.05 * np.exp(np.cos(u - 3.6) - 1.0)
    return (-e * np.sin(u - 3.6) * (1 + 0.3 * np.cos(v)) * 2 * np.pi
            - 0.01 * np.cos(u) * 2 * np.pi,
            -e * 0.3 * np.sin(v) * 2 * np.pi)


UNIT = flat.Box([(0.0, 1.0), (-0.5, 0.5)])


@pytest.mark.parametrize("lateral", ["linear", "cubic", "fourier"])
def test_a_gridded_function_and_its_gradient(lateral):
    g = flat.Grid(UNIT, (48, 40))
    gf = flat.GriddedFunction.sampled(dem, g, lateral=lateral)
    assert gf.lateral == lateral and gf.values.shape == (48, 40) and gf.grid is g
    rng = np.random.default_rng(6)
    x, y = rng.uniform(0, 1, 300), rng.uniform(-0.5, 0.5, 300)
    tol = {"linear": 3e-4, "cubic": 3e-7, "fourier": 1e-12}[lateral]
    assert np.abs(gf(x, y) - dem(x, y)).max() < tol
    gx, gy = gf.lateral_gradient(x, y)
    h = 1e-6
    ok = np.ones(x.shape, dtype=bool)
    if lateral == "linear":
        for axis, v in zip(g.axes, (x, y)):
            ok &= np.abs(v[:, None] - axis[None]).min(axis=1) > 10 * h
    assert np.allclose(gx[ok], ((gf(x + h, y) - gf(x - h, y)) / (2 * h))[ok], atol=1e-7)
    assert np.allclose(gy[ok], ((gf(x, y + h) - gf(x, y - h)) / (2 * h))[ok], atol=1e-7)
    if lateral != "linear":
        ex, ey = dem_gradient(x, y)
        tol = 1e-10 if lateral == "fourier" else 1e-4
        assert np.abs(gx - ex).max() < tol and np.abs(gy - ey).max() < tol
    assert gf.mean() == pytest.approx(g.mean(gf.values), abs=1e-16)
    big = gf.scaled(3.0)
    assert np.allclose(big(3 * x, 3 * y), 3 * gf(x, y), rtol=0, atol=1e-13)
    assert lateral in repr(gf)


@pytest.mark.parametrize("lateral", ["cubic", "fourier"])
def test_a_dem_is_a_relief_whose_stretch_and_geometry_hold(lateral):
    """Topography from gridded data, the Moho a gridded root: the
    displacement, the stretch and the geometry contracts, and the surface
    mapped onto the gridded topography."""
    sk = Skeleton([0.0, 0.4, 1.0])
    g = flat.Grid(UNIT, (32, 24))
    surface = flat.GriddedFunction.sampled(dem, g, lateral=lateral)
    root = flat.GriddedFunction(g, -0.5 * surface.values, lateral=lateral)
    w = flat.layer_linear(sk, [surface, root, None])
    check_vertical_displacement(w, sk, bounds=((0.0, 1.0), (-0.5, 0.5)))
    geometry = flat.Geometry(sk, UNIT).stretched(w)
    check_geometry(geometry)
    _, x, y = geometry.lattice(n_c=3, n_a=5, n_b=4)
    d = np.array([0.05, 0.2, 0.35, 0.45, 0.7, 0.95])[:, None, None]
    check_mapping(geometry.mapping, FLAT.points(d, x, y).reshape(-1, 3))
    xs, ys = g.nodes
    top = geometry.mapping(FLAT.points(0.0, xs, ys))
    # the nodes read back exactly by the bicubic rule, by finufft by the series
    assert np.allclose(top[..., 2], surface.values, rtol=0, atol=1e-13)


def test_a_rectangle_s_dem_is_the_same_at_every_y():
    sk = Skeleton([0.0, 0.4, 1.0])
    rect = flat.Rectangle((0.0, 1.0))
    g = flat.Grid(rect, 32)
    surface = flat.GriddedFunction.sampled(lambda x, y: dem(x, 0.0), g)
    assert surface.values.shape == (32,)
    assert np.array_equal(surface.lateral_gradient(0.3, 1.0)[1], 0.0)
    check_geometry(flat.Geometry(sk, rect).stretched(
        flat.layer_linear(sk, [surface, None, None])))


def test_a_gridded_function_refuses_what_it_cannot_hold():
    g = flat.Grid(UNIT, (6, 5))
    with pytest.raises(ValueError, match=r"grid's shape \(nx, ny\) = \(6, 5\)"):
        flat.GriddedFunction(g, np.ones((5, 6)))
    with pytest.raises(ValueError, match="lateral must be one of"):
        flat.GriddedFunction(g, np.ones((6, 5)), lateral="quintic")
    with pytest.raises(TypeError, match="real"):
        flat.GriddedFunction(g, np.ones((6, 5)) + 0j)
    with pytest.raises(TypeError, match="cartesian.Grid"):
        flat.GriddedFunction(np.ones(3), np.ones((6, 5)))


# -- the gridded vertical displacement --------------------------------------------

def stack(sk):
    """A continuous stack of analytic reliefs to sample: the surface's DEM,
    a root on the Moho, the base held."""
    return flat.layer_linear(sk, [dem, lambda x, y: -0.5 * dem(x, y), None])


def test_a_gridded_displacement_reproduces_the_stretch_it_samples():
    sk = Skeleton([0.0, 0.4, 1.0])
    src = stack(sk)
    geometry = flat.Geometry(sk, UNIT)
    mesh = RadialMesh(geometry, ngll=3, drmax=0.25)
    gd = flat.GriddedDisplacement.sampled(src, mesh, flat.Grid(UNIT, (40, 32)))
    assert gd.knots == tuple(float(v) for v in mesh.left[1:])     # every element edge
    assert 0.4 in gd.knots and len(gd.fields) == 2 and gd.skeleton is sk
    assert gd.name == "gridded_displacement" and "2 layers" in repr(gd)
    rng = np.random.default_rng(7)
    d, x, y = rng.uniform(0, 1, 500), rng.uniform(0, 1, 500), rng.uniform(-.5, .5, 500)
    assert np.abs(gd(d, x, y) - src(d, x, y)).max() < 1e-6
    away = np.abs(d[:, None] - np.array(gd.knots)[None]).min(axis=1) > 1e-3
    assert np.abs(gd.depth_derivative(d, x, y) - src.depth_derivative(d, x, y))[
        away].max() < 1e-5
    for got, want in zip(gd.lateral_gradient(d, x, y), src.lateral_gradient(d, x, y)):
        assert np.abs(got - want).max() < 1e-4
    # at the nodes it is the source exactly
    xs, ys = gd.fields[0].grid.nodes
    assert np.allclose(gd(mesh.r[1, 1], xs, ys), src(mesh.r[1, 1], xs, ys), rtol=0,
                       atol=1e-15)
    check_vertical_displacement(gd, sk, bounds=((0.0, 1.0), (-0.5, 0.5)))
    stretch = VerticalStretch(gd, base=sk)
    _, lx, ly = geometry.lattice(n_c=3, n_a=5, n_b=4)
    inside = np.array([0.05, 0.15, 0.3, 0.45, 0.6, 0.9])[:, None, None]
    check_mapping(stretch, FLAT.points(inside, lx, ly).reshape(-1, 3))
    stretched = geometry.stretched(gd)
    check_geometry(stretched)
    assert stretched.knots == gd.knots


@pytest.mark.parametrize("radial", ["polynomial", "linear"])
def test_a_gridded_displacement_declares_where_its_depth_derivative_jumps(radial):
    """Read by straight lines between the nodes, a displacement's derivative
    in depth jumps at every node; read by the polynomial through an
    element's nodes, at the element boundaries alone.  The depths where it
    does jump are found from one-sided differences down a vertical, and
    are knots; a mesh over the stretched geometry has an edge at every
    knot."""
    sk = Skeleton([0.0, 0.4, 1.0])
    grid = flat.Grid(UNIT, (6, 6))

    def rise(d, x, y):
        return 0.05 * (1.0 - d) ** 2 * (1.0 + 0.2 * np.sin(2.0 * np.pi * x) + 0.0 * y)

    fields = [flat.GriddedField.sampled(
        flat.AnalyticField((lo, hi), rise), grid, breaks=[lo, 0.5 * (lo + hi), hi],
        ngll=4, radial=radial, lateral="cubic")
        for lo, hi in zip(sk.boundaries[:-1], sk.boundaries[1:])]
    gd = flat.GriddedDisplacement(sk, fields)
    nodes = np.unique(np.concatenate([f.radii.ravel() for f in fields]))
    nodes = nodes[(nodes > 0.0) & (nodes < 1.0)]
    assert nodes.size == 11
    want = nodes if radial == "linear" else np.array([0.2, 0.4, 0.7])
    assert np.allclose(gd.knots, want, rtol=0, atol=1e-14)
    assert np.allclose(gd.scaled(3.0).knots, 3.0 * want, rtol=0, atol=1e-13)
    x, y, h = 0.3, 0.1, 1e-6
    probes = np.concatenate([nodes, 0.5 * (nodes[:-1] + nodes[1:])])
    jump = np.abs((gd(probes + h, x, y) - 2.0 * gd(probes, x, y)
                   + gd(probes - h, x, y)) / h)
    jumping = probes[jump > 1e-4]
    assert all(np.min(np.abs(np.array(gd.knots) - d)) < 1e-12 for d in jumping)
    if radial == "linear":
        # a parabola in depth joined by straight lines turns at every node
        assert np.array_equal(np.sort(jumping), nodes)
    stretched = flat.Geometry(sk, UNIT).stretched(gd)
    assert stretched.knots == gd.knots
    mesh = RadialMesh(stretched, ngll=3, drmax=1.0)
    assert all(np.min(np.abs(mesh.left - k)) < 1e-12 for k in gd.knots)


def test_a_gridded_displacement_refuses_what_does_not_fit():
    sk = Skeleton([0.0, 0.4, 1.0])
    grid = flat.Grid(UNIT, (6, 6))
    on = flat.GriddedField.sampled(smooth_for(SCALAR), grid, ngll=2)    # on [0.2, 1]
    vec = flat.GriddedField.sampled(smooth_for(VECTOR), grid, ngll=2)
    with pytest.raises(ValueError, match="2 layers need 2"):
        flat.GriddedDisplacement(sk, [on])
    with pytest.raises(ValueError, match="not a flat gridded scalar field"):
        flat.GriddedDisplacement(sk, [vec, vec])
    with pytest.raises(ValueError, match="its layer"):
        flat.GriddedDisplacement(sk, [on, on])
