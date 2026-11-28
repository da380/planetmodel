"""Slices of a flat model: a plane, a section, a profile, the mean, a
spectrum by horizontal wavenumber and a block, held to the fields
evaluated directly at the slices' points and to the slice contracts."""
import numpy as np
import pytest

from planetmodel import VECTOR, LayeredIsotropicElastic, Scales, perturbed
from planetmodel import cartesian as flat
from planetmodel.cartesian import FLAT
from planetmodel.cartesian import slices
from planetmodel.mapping import MappingBase
from planetmodel.slices import normalised
from planetmodel.testing import check_slice, check_spectrum

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
GRID = flat.Grid(BOX, (8, 6))


def lateral(d, x, y):
    """Periodic over BOX, of the band of GRID."""
    return 4.0 + 0.3 * d * np.cos(np.pi * x) + 0.2 * d * np.sin(np.pi * y)


def flow(d, x, y):
    """A Cartesian vector field: (y, -x, 0.5 z), z = -d."""
    d, x, y = np.broadcast_arrays(d, x, y)
    return np.stack([y, -x, -0.5 * d], axis=-1)


@pytest.fixture
def two():
    """Two layers from the surface down: rho 2 then 3, jumping at depth 0.4;
    vs varying laterally in the deeper layer; a vector field there."""
    m = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], BOX, rho=[2.0, 3.0],
                                     vp=[5.0, 7.0], vs=[3.0, 4.0],
                                     layer_names=["upper", "lower"],
                                     interface_names=["surface", "moho"])
    m = m.with_field(1, "vs", flat.AnalyticField((0.4, 1.0), lateral, name="vs"),
                     replace=True)
    return m.with_field(1, "flow", flat.AnalyticField((0.4, 1.0), flow,
                                                      character=VECTOR,
                                                      frame="cartesian", name="flow"))


def ridge(grid, height=0.06):
    """A relief of a few low modes, heights up to about `height`."""
    x, y = grid.nodes
    h = height * (0.5 + 0.5 * np.cos(np.pi * x)) * (1.0 + 0.3 * np.cos(np.pi * y))
    return flat.FourierFunction(grid, grid.transform.analyse_grid(
        np.broadcast_to(h, grid.shape)))


@pytest.fixture
def raised(two):
    """`two` under topography and a Moho root of twice its depth."""
    g = flat.Grid(BOX, (16, 12))
    top = ridge(g)
    root = flat.FourierFunction(g, -2.0 * top.coefficients)
    return two.stretched(flat.layer_linear(two.skeleton, [top, root, None])), top


# -- planes ------------------------------------------------------------------------

def test_a_plane_is_the_field_at_its_points(two):
    p = slices.plane(two, "vs", depth=0.7, grid=GRID)
    x, y = np.broadcast_arrays(*GRID.nodes)
    assert p.values.shape == GRID.shape
    assert np.allclose(p.values, lateral(0.7, x, y), rtol=1e-14)
    assert np.array_equal(p.reference, FLAT.points(0.7, x, y))
    assert np.array_equal(p.physical, p.reference)
    assert p.at == "reference" and p.side is None and p.depth == 0.7
    assert p.label == "vs [m s-1]" and p.coordinates is FLAT
    assert slices.plane(two, "vs", depth=0.7).values.shape == (slices.PLANE_NODES,) * 2
    with pytest.raises(KeyError, match="nonesuch"):
        slices.plane(two, "nonesuch", depth=0.7)
    with pytest.raises(ValueError, match="outside the body"):
        slices.plane(two, "vs", depth=1.5, grid=GRID)
    with pytest.raises(TypeError, match="cartesian.Model"):
        slices.plane(LayeredIsotropicElastic([0.0, 1.0], rho=[1.0], vp=[2.0],
                                             vs=[1.0]), "rho", depth=0.5)


def test_on_an_interface_a_plane_takes_the_side_asked_for(two):
    with pytest.raises(ValueError, match="depth 0.4 lies on interface 'moho'.*"
                                         "side='below' or 'above'"):
        slices.plane(two, "rho", depth=0.4, grid=GRID)
    below = slices.plane(two, "rho", depth=0.4, grid=GRID, side="below")
    above = slices.plane(two, "rho", depth=0.4, grid=GRID, side="above")
    # below is deeper in a box, as the body is
    assert np.allclose(below.values, 3.0) and np.allclose(above.values, 2.0)
    assert below.side == "below" and above.side == "above"
    away = slices.plane(two, "rho", depth=0.2, grid=GRID, side="below")
    assert np.allclose(away.values, 2.0) and away.side is None
    assert np.allclose(slices.plane(two, "rho", depth=0.0, grid=GRID).values, 2.0)
    assert np.allclose(slices.plane(two, "rho", depth=1.0, grid=GRID).values, 3.0)
    lacking = slices.plane(two, "flow", depth=0.2, grid=GRID, component="norm")
    assert np.all(np.isnan(lacking.values))


def test_a_depth_on_an_interface_to_the_tolerance_is_read_on_it(two):
    """The geometry takes a depth within its tolerance of a boundary to lie
    on it; the layer asked for is then read at its end, the field never
    asked outside its interval."""
    x, y = np.broadcast_arrays(*GRID.nodes)
    tol = two.geometry.rtol * two.skeleton.span
    off = 0.8 * tol
    for at in ("reference", "physical"):
        above = slices.plane(two, "rho", depth=0.4 + off, grid=GRID, side="above",
                             at=at)
        assert np.allclose(above.values, 2.0) and above.side == "above"
        below = slices.plane(two, "vs", depth=0.4 - off, grid=GRID, side="below",
                             at=at)
        assert np.allclose(below.values, lateral(0.4, x, y), rtol=1e-12)
        base = slices.plane(two, "vs", depth=1.0 + off, grid=GRID, at=at)
        assert np.allclose(base.values, lateral(1.0, x, y), rtol=1e-12)
    assert slices.evaluate(two, "rho", 0, 0.4 + off, 0.3, 0.2) == 2.0
    with pytest.raises(ValueError, match="outside the body"):
        slices.plane(two, "rho", depth=1.0 + 1e3 * tol, grid=GRID)
    with pytest.raises(ValueError, match="outside the interval"):
        slices.evaluate(two, "rho", 0, 0.4 + 1e3 * tol, 0.3, 0.2)


def test_a_plane_of_a_rectangle_is_a_line_in_x():
    rect = flat.Rectangle((0.0, 2.0))
    m = flat.LayeredIsotropicElastic([0.0, 1.0], rect, rho=[1.0], vp=[2.0], vs=[1.0])
    m = m.with_field(0, "vs", flat.AnalyticField((0.0, 1.0), lateral, name="vs"),
                     replace=True)
    g = flat.Grid(rect, 8)
    p = slices.plane(m, "vs", depth=0.5, grid=g)
    assert p.values.shape == (8,)
    assert np.allclose(p.values, lateral(0.5, g.x, 0.0))
    sp = slices.spectrum_of(p)
    assert np.isclose(sp.power.sum(), g.mean(p.values ** 2), rtol=1e-13)
    sec = slices.section(m, "vs", (0.0, 0.3), end=(2.0, 0.3), n=4, m=5)
    assert np.allclose(sec.layers[0].values, lateral(sec.layers[0].radius,
                                                     sec.path[:, 0], 0.3))
    with pytest.raises(ValueError, match="a block is of a body over a Box"):
        slices.block(m, "vs")


# -- sections and profiles -------------------------------------------------------

def test_sections_and_profiles_keep_both_sides_of_an_interface(two):
    sec = slices.section(two, "rho", (0.1, -0.5), end=(1.9, 0.7), n=6, m=11)
    assert [lay.index for lay in sec.layers] == [0, 1]
    upper, lower = sec.layers
    assert upper.radius.shape == (6, 11) and upper.values.shape == (6, 11)
    assert np.allclose(upper.radius[-1], 0.4) and np.allclose(lower.radius[0], 0.4)
    assert np.allclose(upper.values[-1], 2.0) and np.allclose(lower.values[0], 3.0)
    prof = slices.profile(two, "rho", 1.2, 0.3, n=5)
    assert prof.x == 1.2 and prof.y == 0.3
    assert [lay.radius.shape for lay in prof.layers] == [(5,), (5,)]
    assert prof.layers[0].values[-1] == 2.0 and prof.layers[1].values[0] == 3.0
    part = slices.section(two, "rho", (0.1, -0.5), end=(1.9, 0.7), depths=(0.6, 1.0))
    assert [lay.index for lay in part.layers] == [1]
    assert np.isclose(part.layers[0].radius.min(), 0.6)


def test_a_section_is_the_field_along_its_line(two):
    sec = slices.section(two, "vs", (0.2, -0.8), end=(1.8, 0.4), n=4, m=21)
    assert sec.length == pytest.approx(2.0)
    assert np.allclose(sec.path[:, 2], 0.0)
    assert np.allclose(sec.plane @ sec.plane.T, np.eye(2), atol=1e-15)
    for lay in sec.layers:
        d, x, y, _ = FLAT.split(lay.reference)
        assert np.allclose(lay.radius, d)
        if lay.index == 1:
            assert np.allclose(lay.values, lateral(d, x, y), rtol=1e-14)
        # the section's own coordinates: (distance, depth)
        st = (lay.reference - sec.path[0]) @ sec.plane.T
        assert np.allclose(st[..., 0], sec.distance[None, :])
        assert np.allclose(st[..., 1], lay.radius)
    # an azimuth from north toward east
    north = slices.section(two, "rho", (0.5, -0.5), azimuth=0.0, length=1.0, m=3)
    assert np.allclose(north.path[:, :2], [[0.5, -0.5], [0.5, 0.0], [0.5, 0.5]])
    east = slices.section(two, "rho", (0.5, -0.5), azimuth=0.5 * np.pi, length=1.0,
                          m=3)
    assert np.allclose(east.path[:, :2], [[0.5, -0.5], [1.0, -0.5], [1.5, -0.5]])
    with pytest.raises(ValueError, match="the same point"):
        slices.section(two, "vs", (0.5, 0.0), end=(0.5, 0.0))
    with pytest.raises(ValueError, match="azimuth"):
        slices.section(two, "vs", (0.5, 0.0))
    with pytest.raises(ValueError, match="follows from its end"):
        slices.section(two, "vs", (0.5, 0.0), end=(1.0, 0.0), length=1.0)
    with pytest.raises(ValueError, match="lo < hi"):
        slices.section(two, "vs", (0.5, 0.0), end=(1.0, 0.0), depths=(0.5, 0.2))


def test_a_vector_is_sliced_by_component_in_a_frame_or_by_its_norm(two):
    x, y = np.broadcast_arrays(*GRID.nodes)
    u = flow(0.7, x, y)
    z = slices.plane(two, "flow", depth=0.7, grid=GRID, component=2,
                     frame="cartesian")
    assert np.allclose(z.values, u[..., 2])
    # the local frame's first component is upward: the Cartesian z
    up = slices.plane(two, "flow", depth=0.7, grid=GRID, component=0, frame="local")
    assert np.allclose(up.values, u[..., 2])
    ex = slices.plane(two, "flow", depth=0.7, grid=GRID, component=1, frame="local")
    assert np.allclose(ex.values, u[..., 0])
    norm = slices.plane(two, "flow", depth=0.7, grid=GRID, component="norm")
    assert np.allclose(norm.values, np.linalg.norm(u, axis=-1))
    assert norm.label == "|flow|" and up.label == "flow[0] (local)"
    with pytest.raises(ValueError, match="name frame=, one of \\('local', 'cartesian'"):
        slices.plane(two, "flow", depth=0.7, grid=GRID, component=1)
    with pytest.raises(ValueError, match="unknown frame 'spherical'"):
        slices.plane(two, "flow", depth=0.7, grid=GRID, component=1,
                     frame="spherical")
    with pytest.raises(ValueError, match="rank 1"):
        slices.plane(two, "flow", depth=0.7, grid=GRID)


# -- departures --------------------------------------------------------------------

def test_a_departure_from_the_mean_is_zero_in_mean(two):
    x, y = np.broadcast_arrays(*GRID.nodes)
    for departure in ("difference", "percent"):
        d = slices.plane(two, "vs", depth=0.9, grid=GRID, relative_to="mean",
                         departure=departure)
        assert abs(GRID.mean(d.values)) < 1e-13
    pct = slices.plane(two, "vs", depth=0.9, grid=GRID, relative_to="mean")
    assert np.allclose(pct.values, 100.0 * (lateral(0.9, x, y) / 4.0 - 1.0))
    assert pct.label == "vs, percent from the mean [%]" and pct.shown.is_departure
    # a section's mean is taken on the grid it is given
    with pytest.raises(ValueError, match="no horizontal mean of its own: pass "
                                         "mean_grid="):
        slices.section(two, "vs", (0.5, 0.0), end=(1.0, 1.0), relative_to="mean")
    sec = slices.section(two, "vs", (0.5, 0.0), end=(1.0, 1.0), relative_to="mean",
                         departure="difference", mean_grid=GRID, n=3, m=4)
    lay = sec.layers[1]
    d, x, y, _ = FLAT.split(lay.reference)
    assert np.allclose(lay.values, lateral(d, x, y) - 4.0)
    with pytest.raises(ValueError, match="relative_to"):
        slices.plane(two, "vs", depth=0.9, grid=GRID, departure="percent")


def test_a_departure_from_a_reference_or_another_model(two):
    x, y = np.broadcast_arrays(*GRID.nodes)
    base = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], BOX, rho=[2.0, 3.0],
                                        vp=[5.0, 7.0], vs=[3.0, 4.0]).named("base")
    dp = flat.AnalyticField((0.4, 1.0), lambda d, x, y: 0.02 * np.cos(np.pi * x) * d)
    m = perturbed(base, "vs", dp, kind="relative")
    rel = slices.plane(m, "vs", depth=0.8, grid=GRID, relative_to="reference")
    assert np.allclose(rel.values, 100.0 * 0.02 * np.cos(np.pi * x) * 0.8)
    assert rel.label == "vs, percent from the reference [%]"
    with pytest.raises(ValueError, match="not perturbed"):
        slices.plane(two, "vs", depth=0.8, grid=GRID, relative_to="reference")
    shallow = slices.plane(m, "vs", depth=0.2, grid=GRID, relative_to="reference")
    assert np.allclose(shallow.values, 0.0)
    vs = slices.plane(two, "vs", depth=0.8, grid=GRID, relative_to=base,
                      departure="log10")
    assert np.allclose(vs.values, np.log10(lateral(0.8, x, y) / 4.0))
    assert vs.label == "vs, log10 of the ratio to base"
    # on the interface the other model is read on the slice's own side
    rho = slices.section(two, "rho", (0.1, 0.0), end=(1.9, 0.0), relative_to=base,
                         departure="difference", n=3, m=4)
    assert all(np.allclose(lay.values, 0.0) for lay in rho.layers)
    sphere = LayeredIsotropicElastic([0.0, 0.4, 1.0], rho=[3.0, 2.0], vp=[7.0, 5.0],
                                     vs=[4.0, 3.0])
    with pytest.raises(ValueError, match="same coordinates; this model is in 'flat' "
                                         "and the other in 'spherical'"):
        slices.plane(two, "vs", depth=0.8, grid=GRID, relative_to=sphere)


def test_another_model_in_other_scales_is_compared_at_the_same_place(two):
    """A departure compares numbers in one unit at one place: another model
    in other scales is converted to the model's, so a model departs from
    itself in other units by nothing, and from another by what their SI
    numbers give."""
    x, y = np.broadcast_arrays(*GRID.nodes)
    plain = two.without_field("flow")
    scales = Scales(length=0.5, mass=1e-3, time=3.0)
    itself = plain.converted(scales)
    assert itself.skeleton.boundaries[-1] == 2.0
    for name, depth in (("rho", 0.8), ("vs", 0.8), ("rho", 0.2)):
        s = slices.plane(plain, name, depth=depth, grid=GRID, relative_to=itself)
        assert np.allclose(s.values, 0.0, rtol=0, atol=1e-12)
    base = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], BOX, rho=[4.0, 7.0],
                                        vp=[5.0, 7.0], vs=[3.0, 2.0]).named("base")
    other = base.converted(scales)
    rho = slices.plane(plain, "rho", depth=0.8, grid=GRID, relative_to=other)
    assert np.allclose(rho.values, 100.0 * (3.0 / 7.0 - 1.0), rtol=1e-12)
    vs = slices.plane(plain, "vs", depth=0.8, grid=GRID, relative_to=other,
                      departure="difference")
    assert np.allclose(vs.values, lateral(0.8, x, y) - 2.0, rtol=1e-12)
    sec = slices.section(plain, "rho", (0.1, 0.0), end=(1.9, 0.0), relative_to=other,
                         departure="difference", n=3, m=4)
    assert np.allclose(sec.layers[0].values, -2.0, rtol=1e-12)
    assert np.allclose(sec.layers[1].values, -4.0, rtol=1e-12)
    # two fields correlated: the other model read at the first one's depths,
    # where its spectrum is the first's own
    cr = slices.correlation(plain, "vs", itself, depths=[0.5, 0.9], grid=GRID)
    assert np.allclose(cr.total, 1.0)
    for first, second, depth in zip(cr.first, cr.second, cr.depths):
        assert np.isclose(first.power[1], depth ** 2 * (0.3 ** 2 + 0.2 ** 2) / 2.0)
        assert np.allclose(second.power, first.power, rtol=1e-12, atol=1e-14)
    with pytest.raises(ValueError, match="fields given by layer state no scales"):
        slices.correlation(plain, "vs", itself, depths=[0.5], grid=GRID,
                           other_fields={1: itself.layer(1)["vs"]})


# -- reference and physical --------------------------------------------------------

def test_slices_at_reference_points_give_their_physical_positions(raised):
    m, top = raised
    p = slices.plane(m, "vs", depth=0.7, grid=GRID)
    assert np.allclose(p.physical, m.geometry.mapping(p.reference))
    sec = slices.section(m, "rho", (0.0, 0.2), end=(2.0, 0.2), n=5, m=9)
    for lay in sec.layers:
        assert np.allclose(lay.physical, m.geometry.mapping(lay.reference))
    # the surface row is on the topography, z = the relief
    assert np.allclose(sec.layers[0].physical[0, :, 2], top(sec.path[:, 0], 0.2),
                       rtol=0, atol=1e-14)


def test_slices_at_physical_points_lie_where_they_were_asked_for(raised):
    m, top = raised
    mapping = m.geometry.mapping
    p = slices.plane(m, "vs", depth=0.7, grid=GRID, at="physical")
    assert np.allclose(p.physical[..., 2], -0.7)
    assert np.allclose(mapping(p.reference), p.physical, rtol=0, atol=1e-11)
    d, x, y, _ = FLAT.split(p.reference)
    assert np.allclose(p.values, lateral(d, x, y))
    # a physical plane above the reference surface is in the body only under
    # the topography, NaN elsewhere
    high = slices.plane(m, "rho", depth=-0.03, grid=GRID, at="physical")
    under = top(*np.broadcast_arrays(*GRID.nodes)) > 0.03
    assert np.all(np.isnan(high.values[~under]))
    assert np.allclose(high.values[under], 2.0)
    # a physical section: each layer between its mapped boundaries, the
    # surface on the topography
    sec = slices.section(m, "rho", (0.0, 0.2), end=(2.0, 0.2), n=5, m=9,
                         at="physical")
    for lay in sec.layers:
        assert np.allclose(-lay.physical[..., 2], lay.radius)
        assert np.allclose(mapping(lay.reference), lay.physical, rtol=0, atol=1e-11)
    xs = sec.path[:, 0]
    assert np.allclose(-sec.layers[0].radius[0], top(xs, 0.2), rtol=0, atol=1e-12)
    assert np.allclose(sec.layers[0].radius[-1], 0.4 + 2.0 * top(xs, 0.2), rtol=0,
                       atol=1e-12)
    assert np.allclose(sec.layers[1].radius[0], sec.layers[0].radius[-1])
    prof = slices.profile(m, "rho", 0.3, 0.2, at="physical", n=4)
    assert np.isclose(prof.layers[0].radius[0], -top(0.3, 0.2))


class _Shear(MappingBase):
    """A lateral shear growing with depth: x moves, so verticals do not
    stay vertical; no inverse."""

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        out = np.array(X)
        out[..., 0] = X[..., 0] - 0.01 * X[..., 2]
        return out


def test_a_mapping_without_an_inverse_or_off_the_verticals_is_refused(two):
    sheared = two.with_mapping(_Shear(), check=False)
    with pytest.raises(ValueError, match="does not invert itself"):
        slices.plane(sheared, "vs", depth=0.5, grid=GRID, at="physical")
    p = slices.plane(sheared, "vs", depth=0.5, grid=GRID)
    assert np.allclose(p.physical, _Shear()(p.reference))
    with pytest.raises(ValueError, match="off their verticals"):
        slices.section(sheared, "vs", (0.5, 0.0), end=(1.0, 1.0), at="physical")


# -- the mean and its spread -------------------------------------------------------

@pytest.mark.parametrize("domain, centre", [
    (flat.Box([(0.0, 2.0), (-1.0, 3.0)], padding=(0.5, 0.25)), (1.0, 1.0)),
    (flat.Rectangle((1.0, 2.0), padding=0.5), (1.5, 0.0))], ids=repr)
def test_a_mean_profile_stands_at_the_middle_of_the_extent(domain, centre):
    """Its points are at the middle of each axis the domain bounds, and at
    zero along one it does not."""
    m = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], domain, rho=[2.0, 3.0],
                                     vp=[5.0, 6.0], vs=[3.0, 4.0])
    mp = slices.mean_profile(m, "vs", mean_grid=flat.Grid(domain, 4), n=5)
    for lay, vs in zip(mp.layers, (3.0, 4.0)):
        _, x, y, _ = FLAT.split(lay.reference)
        assert np.all(x == centre[0]) and np.all(y == centre[1])
        assert np.allclose(lay.values, vs)


def test_the_mean_profile_and_its_spread(two):
    mp = slices.mean_profile(two, "vs", mean_grid=GRID, n=6)
    assert mp.x is None and mp.label == "vs, horizontal mean [m s-1]"
    assert mp.bands is None and mp.spread is None
    assert np.allclose(mp.layers[0].values, 3.0)
    assert np.allclose(mp.layers[1].values, 4.0, rtol=0, atol=1e-14)
    with pytest.raises(ValueError, match="mean_grid"):
        slices.mean_profile(two, "vs")
    with pytest.raises(ValueError, match="pass mean_grid="):
        slices.mean_profile(two, "vs", spread="range")
    with pytest.raises(ValueError, match="spread must be one of"):
        slices.mean_profile(two, "vs", spread="iqr", mean_grid=GRID)
    x, y = np.broadcast_arrays(*GRID.nodes)
    rng = slices.mean_profile(two, "vs", mean_grid=GRID, n=6, spread="range")
    std = slices.mean_profile(two, "vs", mean_grid=GRID, n=6, spread="std")
    d = rng.layers[1].radius
    v = lateral(d[:, None, None], x[None], y[None])
    lo, hi = rng.bands[1]
    assert np.allclose(lo, v.min(axis=(1, 2))) and np.allclose(hi, v.max(axis=(1, 2)))
    lo, hi = std.bands[1]
    dev = np.sqrt(GRID.mean((v - GRID.mean(v)[:, None, None]) ** 2))
    assert np.allclose(hi - std.layers[1].values, dev, rtol=1e-12)
    assert np.allclose(std.layers[1].values - lo, dev, rtol=1e-12)
    # a field of depth alone has no spread
    assert np.allclose(np.subtract(*std.bands[0]), 0.0)


@pytest.mark.parametrize("spread", ["range", "std"])
def test_a_normalised_mean_profile_keeps_its_bands_about_its_values(two, spread):
    """The bands are a spread about the values, so the number that divides
    the values divides them: the largest mean here, 4."""
    mp = slices.mean_profile(two, "vs", mean_grid=GRID, n=6, spread=spread)
    unit = normalised(mp)
    assert unit.spread == spread and unit.label == "vs, horizontal mean, normalised"
    assert np.isclose(max(np.max(np.abs(lay.values)) for lay in mp.layers), 4.0)
    for raw, got, was, lay in zip(mp.bands, unit.bands, mp.layers, unit.layers):
        assert np.allclose(lay.values, was.values / 4.0, rtol=1e-14)
        assert np.allclose(got[0], raw[0] / 4.0, rtol=1e-14)
        assert np.allclose(got[1], raw[1] / 4.0, rtol=1e-14)
        assert np.all(got[0] <= lay.values + 1e-14)
        assert np.all(lay.values <= got[1] + 1e-14)
    assert np.ptp(unit.bands[1][1] - unit.bands[1][0]) > 0.0
    # a layer lacking the field has no band, and none is made for it
    lacking = slices.mean_profile(two.without_field("vs", layers=[0]), "vs",
                                  mean_grid=GRID, n=6, spread=spread)
    assert normalised(lacking).bands[0] is None
    # a profile without a spread has none
    assert normalised(slices.mean_profile(two, "vs", mean_grid=GRID, n=6)).bands is None


def test_a_spread_is_taken_a_block_of_depths_at_a_time():
    big = flat.Grid(BOX, (1024, 1024))
    blocks = slices._blocks(np.arange(100.0), big)
    assert sum(b.size for b in blocks) == 100
    assert max(b.size for b in blocks) * big.shape[0] * big.shape[1] <= 1 << 22


# -- spectra -----------------------------------------------------------------------

def _mode_model(lengths, wavevector, *, phase=0):
    """One layer whose field is one Fourier mode of the box's grid."""
    box = flat.Box([(0.0, lengths[0]), (0.0, lengths[1])])
    g = flat.Grid(box, (16, 16))
    c = np.zeros(g.transform.size)
    c[g.transform.component_of(wavevector, phase=phase)] = 1.0
    f = flat.FourierField((0.0, 1.0), g, c, name="q")
    m = flat.LayeredIsotropicElastic([0.0, 1.0], box, rho=[1.0], vp=[2.0], vs=[1.0])
    return m.with_field(0, "q", f), g


def test_the_power_sums_to_the_mean_square(two):
    g = flat.Grid(BOX, (12, 10))
    c = np.random.default_rng(3).standard_normal(g.transform.size)
    m = two.with_field(1, "q", flat.FourierField((0.4, 1.0), g, c, name="q"))
    p = slices.plane(m, "q", depth=0.6, grid=g)
    sp = slices.spectrum_of(p)
    assert np.isclose(sp.power.sum(), g.mean(p.values ** 2), rtol=1e-13)
    assert np.allclose(sp.coefficients, c, rtol=0, atol=1e-11)
    bins, step = slices.wavenumber_bins(g)
    assert np.allclose(sp.power, np.bincount(bins, weights=c * c) / g.area)
    assert np.isclose(sp.spacing, step) and np.allclose(sp.wavenumbers,
                                                        sp.degrees * step)
    assert sp.wavelengths[0] == np.inf
    assert np.allclose(sp.wavelengths[1:], 2.0 * np.pi / sp.wavenumbers[1:])
    check_spectrum(sp)
    # a field the grid resolves, its Nyquist components zero, has the same
    # power on a finer grid: the mean square over the period
    nyquist = np.any(np.abs(g.transform.wavevectors)
                     == np.array(g.shape)[:, None] // 2, axis=0)
    c[nyquist] = 0.0
    m = two.with_field(1, "q", flat.FourierField((0.4, 1.0), g, c, name="q"))
    coarse = slices.spectrum(m, "q", depth=0.6, grid=g)
    finer = slices.spectrum(m, "q", depth=0.6, grid=flat.Grid(BOX, (24, 20)))
    # an empty bin holds the square of the transform's rounding, which the
    # coefficients above are held to
    assert np.allclose(finer.power[:coarse.power.size], coarse.power, rtol=1e-11,
                       atol=1e-11 ** 2)
    assert np.isclose(finer.power.sum(), coarse.power.sum(), rtol=1e-12)


@pytest.mark.parametrize(("lengths", "wavevector", "bin_"), [
    ((2.0, 2.0), [3, 0], 3), ((2.0, 2.0), [0, 2], 2), ((2.0, 2.0), [3, 4], 5),
    ((4.0, 2.0), [0, 1], 2), ((4.0, 2.0), [3, 0], 3)])
def test_a_single_mode_has_its_power_at_its_wavenumber(lengths, wavevector, bin_):
    m, g = _mode_model(lengths, wavevector)
    sp = slices.spectrum(m, "q", depth=0.5, grid=g)
    k = 2.0 * np.pi * np.hypot(wavevector[0] / lengths[0], wavevector[1] / lengths[1])
    assert int(np.argmax(sp.power)) == bin_
    assert np.isclose(sp.wavenumbers[bin_], k)
    assert np.isclose(sp.power[bin_], 1.0 / g.area)
    assert np.isclose(sp.power.sum() - sp.power[bin_], 0.0, atol=1e-25)


def test_over_equal_periods_the_bins_are_the_packing_s_degrees():
    g = flat.Grid(BOX, (12, 10))
    bins, step = slices.wavenumber_bins(g)
    assert np.array_equal(bins, g.transform.degrees)
    assert step == pytest.approx(np.pi)


def test_a_spectrum_wants_the_field_everywhere(raised):
    m, _ = raised
    with pytest.raises(ValueError, match="not defined everywhere"):
        slices.spectrum(m, "rho", depth=-0.03, grid=GRID, at="physical")


def test_a_field_correlates_to_one_with_itself_and_to_zero_with_an_orthogonal_one():
    a, g = _mode_model((2.0, 2.0), [2, 1])
    b, _ = _mode_model((2.0, 2.0), [1, 2], phase=1)    # the same bin, orthogonal
    both = a.with_field(0, "r", b.layer(0)["q"].renamed("r"))
    same = slices.correlation(a, "q", a, depths=[0.2, 0.7], grid=g)
    assert same.by_degree.shape == (2, same.degrees.size)
    assert np.allclose(same.by_degree[:, 2], 1.0)
    assert np.all(np.isnan(np.delete(same.by_degree, 2, axis=1)))
    assert np.allclose(same.total, 1.0) and same.label == "correlation of q"
    cross = slices.correlation(both, "q", both, other_fields="r", depths=[0.5],
                               grid=g)
    assert np.allclose(cross.by_degree[0, 2], 0.0, atol=1e-12)
    assert cross.label == "correlation of q with r"
    assert np.array_equal(cross.radii, cross.depths)


# -- surfaces, integrals, blocks ---------------------------------------------------

def test_a_function_on_an_interface_is_a_plane_at_its_depth(raised):
    m, top = raised
    s = slices.surface(m, top, "moho", name="height")
    assert s.grid == top.grid and s.depth == 0.4
    x, y = np.broadcast_arrays(*top.grid.nodes)
    assert np.allclose(s.values, top(x, y))
    assert np.allclose(s.physical, m.geometry.mapping(s.reference))
    assert s.label == "height, on the interface moho"
    g = flat.Grid(BOX, (8, 4))
    held = flat.GriddedFunction(g, np.arange(32.0).reshape(8, 4))
    assert np.array_equal(slices.surface(m, held, 0).values, held.values)


def test_an_integral_down_the_vertical_is_its_closed_form(two):
    x, y = np.broadcast_arrays(*GRID.nodes)
    s = slices.integrated(two, "vs", depths=(0.5, 0.9), grid=GRID)
    want = (4.0 * 0.4 + (0.3 * np.cos(np.pi * x) + 0.2 * np.sin(np.pi * y))
            * (0.9 ** 2 - 0.5 ** 2) / 2.0)
    assert np.allclose(s.values, want, rtol=1e-12)
    assert s.depth == 0.5
    assert s.label == "vs, integrated down the vertical, d depth [m2 s-1]"
    whole = slices.integrated(two, "rho", grid=GRID)
    assert np.allclose(whole.values, 2.0 * 0.4 + 3.0 * 0.6)


@pytest.mark.parametrize("kind", ["relative", "log"])
def test_an_integral_of_a_perturbed_field_is_of_its_values(two, kind):
    """A reference cubic in depth times one plus a cubic is of degree six,
    integrated here in closed form; times the exponential of one it is no
    polynomial, and `n` points on each piece hold it, held here to scipy's
    adaptive quadrature."""
    from numpy.polynomial import Polynomial
    from scipy.integrate import quad
    from scipy.interpolate import PPoly
    from planetmodel.layerfunction import PolynomialLayer
    iv = (0.4, 1.0)
    x, y = np.broadcast_arrays(*GRID.nodes)
    # in s = depth - 0.4: the reference 3 + 2 s - 4 s^2 + 5 s^3, the
    # perturbation 0.4 s^3 cos(pi x)
    ref = flat.DepthField(iv, PolynomialLayer(iv, PPoly(
        np.array([[5.0], [-4.0], [2.0], [3.0]]), np.array(iv))), name="vs",
        character=two.layer(1)["vs"].character)
    dp = flat.GriddedField.sampled(
        flat.AnalyticField(iv, lambda d, x, y: 0.4 * (d - 0.4) ** 3 * np.cos(np.pi * x)
                           + 0.0 * y), GRID, breaks=[0.4, 0.6, 1.0], ngll=4)
    m = perturbed(two.with_field(1, "vs", ref, replace=True), "vs", dp, kind=kind)
    reference, cubic = Polynomial([3.0, 2.0, -4.0, 5.0]), Polynomial([0, 0, 0, 0.4])
    if kind == "relative":
        want = (reference.integ()(0.6)
                + (reference * cubic).integ()(0.6) * np.cos(np.pi * x))
    else:
        want = np.array([quad(lambda s, c=c: reference(s) * np.exp(cubic(s) * c),
                              0.0, 0.6, epsabs=0.0, epsrel=1e-13)[0]
                         for c in np.cos(np.pi * x[:, 0])])
        want = np.broadcast_to(want[:, None], x.shape)
    got = slices.integrated(m, "vs", depths=iv, grid=GRID)
    assert np.allclose(got.values, want, rtol=1e-13, atol=0)
    if kind == "log":
        coarse = slices.integrated(m, "vs", depths=iv, grid=GRID, n=2)
        assert not np.allclose(coarse.values, got.values, rtol=1e-9, atol=0)
        assert np.allclose(coarse.values, got.values, rtol=1e-3, atol=0)


def test_a_block_is_the_top_and_two_sides_toward_its_corner(two):
    b = slices.block(two, "vs", n=4, m=5)
    assert b.faces == ("top", "south", "south", "east", "east")
    assert b.depths == (0.0, 1.0) and np.allclose(b.boundaries, [0.4])
    top = b.face("top")[0]
    assert top.values.shape == (5, 5) and np.allclose(top.radius, 0.0)
    assert np.allclose(top.values, 3.0)
    for lay in b.face("south"):
        assert np.allclose(lay.reference[..., 1], -1.0)
        assert np.allclose(lay.reference[0, :, 0], np.linspace(0.0, 2.0, 5))
    for lay in b.face("east"):
        assert np.allclose(lay.reference[..., 0], 2.0)
    deep = b.face("east")[1]
    d, x, y, _ = FLAT.split(deep.reference)
    assert np.allclose(deep.values, lateral(d, x, y))
    nw = slices.block(two, "vs", corner="northwest", depths=(0.4, 0.8), n=3, m=4)
    assert nw.faces == ("top", "north", "west")
    assert np.allclose(nw.face("top")[0].values, lateral(0.4, *FLAT.split(
        nw.face("top")[0].reference)[1:3]))
    with pytest.raises(ValueError, match="corner must be one of"):
        slices.block(two, "vs", corner="up")
    with pytest.raises(ValueError, match="name must be one of"):
        b.face("west")
    unit = normalised(b)
    assert np.isclose(max(np.nanmax(np.abs(lay.values)) for lay in unit.layers), 1.0)


def test_a_block_under_topography_wears_it(raised):
    m, top = raised
    b = slices.block(m, "rho", n=3, m=9)
    face = b.face("top")[0]
    assert np.allclose(face.physical[..., 2], top(face.reference[..., 0],
                                                  face.reference[..., 1]), atol=1e-14)


def test_evaluate_is_what_a_slice_shows(two):
    p = slices.plane(two, "vs", depth=0.7, grid=GRID, relative_to="mean")
    x, y = np.broadcast_arrays(*GRID.nodes)
    v = slices.evaluate(two, "vs", 1, 0.7, x, y, relative_to="mean", mean_grid=GRID)
    assert np.allclose(v, p.values)
    with pytest.raises(IndexError, match="no layer 2"):
        slices.evaluate(two, "vs", 2, 0.7, x, y)


# -- the contracts -----------------------------------------------------------------

def test_every_flat_slice_meets_the_slice_contract(two, raised):
    m, top = raised
    for model in (two, m):
        for at in ("reference", "physical"):
            check_slice(slices.plane(model, "vs", depth=0.7, grid=GRID, at=at), model)
            check_slice(slices.section(model, "vs", (0.1, -0.5), end=(1.9, 0.7), n=4,
                                       m=7, at=at), model)
            check_slice(slices.profile(model, "rho", 0.3, 0.2, n=4, at=at), model)
        check_slice(slices.plane(model, "flow", depth=0.7, grid=GRID, component=1,
                                 frame="local"), model)
        check_slice(slices.plane(model, "rho", depth=0.4, grid=GRID, side="below"),
                    model)
        check_slice(slices.mean_profile(model, "vs", mean_grid=GRID, n=4,
                                        spread="std"), model)
        check_slice(slices.block(model, "vs", n=3, m=5), model)
        check_slice(slices.integrated(model, "vs", grid=GRID), model)
        check_slice(slices.surface(model, top, "surface"), model)
        check_slice(slices.section(model, "vs", (0.0, 0.0), end=(2.0, 0.0),
                                   relative_to="mean", mean_grid=GRID, n=3, m=4), model)
    check_spectrum(slices.spectrum(two, "vs", depth=0.7, grid=GRID))


def test_the_slice_contract_holds_a_mean_profile_s_bands(two):
    """The bands stay about the values, one for each layer, and none is
    held where a layer has values."""
    from dataclasses import replace
    for spread in ("std", "range"):
        profile = slices.mean_profile(two, "vs", mean_grid=GRID, n=4, spread=spread)
        check_slice(profile, two)
        low, high = profile.bands[0]
        beside = (profile.bands[0][0] + 10.0, profile.bands[0][1] + 10.0)
        with pytest.raises(AssertionError, match="values leave its band"):
            check_slice(replace(profile, bands=(beside,) + profile.bands[1:]), two)
        with pytest.raises(AssertionError, match="has values and no band"):
            check_slice(replace(profile, bands=(None,) + profile.bands[1:]), two)
        with pytest.raises(AssertionError, match="1 bands for 2 layers"):
            check_slice(replace(profile, bands=profile.bands[:1]), two)
        with pytest.raises(AssertionError, match="has shapes"):
            check_slice(replace(profile, bands=((low[:2], high[:2]),)
                                + profile.bands[1:]), two)


def test_the_slice_contract_holds_a_physical_section_to_the_mapped_body(raised):
    m, _ = raised
    # cut at the reference surface, the topography above it left out:
    # refused as the whole body, held to the bounds it was given
    for cut in (slices.section(m, "vs", (0.1, -0.5), end=(1.9, 0.7), n=4, m=7,
                               at="physical", depths=(0.0, 1.0)),
                slices.profile(m, "vs", 0.3, 0.2, n=4, at="physical",
                               depths=(0.0, 1.0))):
        with pytest.raises(AssertionError, match="where the mapping puts"):
            check_slice(cut, m)
        check_slice(cut, m, bounds=(0.0, 1.0))
