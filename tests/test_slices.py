"""Slices: a model's field on a sphere, a section and a profile, held to
the fields evaluated directly at the slices' points."""
import numpy as np
import pytest

from planetmodel import (DENSITY, VECTOR, AnalyticField, LayeredIsotropicElastic,
                         Scales, flattening, gauss_legendre, perturbed)
from planetmodel import slices
from planetmodel.frames import cartesian_points, spherical_coordinates
from planetmodel.mapping import MappingBase


def lateral(r, t, p):
    return 4.0 + 0.3 * r * np.cos(t) + 0.2 * r * np.sin(t) * np.cos(p)


def flow(r, t, p):
    """A rotation and stretch, Cartesian."""
    X = cartesian_points(r, t, p)
    return np.stack([X[..., 1], -X[..., 0], 0.5 * X[..., 2]], axis=-1)


@pytest.fixture
def two():
    """Two layers: rho of the radius alone, jumping at r = 0.5; vs varying
    with direction in the outer layer; a vector field in the outer layer."""
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    m = m.with_field(1, "vs", AnalyticField((0.5, 1.0), lateral, name="vs"),
                     replace=True)
    return m.with_field(1, "flow", AnalyticField((0.5, 1.0), flow, character=VECTOR,
                                                 frame="cartesian", name="flow"))


def test_a_sphere_slice_is_the_field_at_its_points(two):
    grid = gauss_legendre(8)
    s = slices.sphere(two, "vs", radius=0.8, grid=grid)
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    assert s.values.shape == (grid.ntheta, grid.nphi)
    assert np.allclose(s.values, lateral(0.8, t, p), rtol=1e-14)
    assert np.allclose(s.reference, cartesian_points(0.8, t, p))
    assert np.array_equal(s.reference, s.physical)
    assert s.at == "reference" and s.side is None
    assert s.label == "vs [m s-1]"
    # depth below the outer boundary
    d = slices.sphere(two, "vs", depth=0.2, grid=grid)
    assert np.array_equal(d.values, s.values) and d.radius == pytest.approx(0.8)
    # the default grid: one degree
    assert slices.sphere(two, "vs", radius=0.8).values.shape == (180, 360)
    with pytest.raises(ValueError, match="exactly one"):
        slices.sphere(two, "vs", radius=0.8, depth=0.2)
    with pytest.raises(KeyError, match="nonesuch"):
        slices.sphere(two, "nonesuch", radius=0.8)


def test_on_an_interface_a_sphere_slice_takes_the_side_asked_for(two):
    grid = gauss_legendre(4)
    with pytest.raises(ValueError, match="side='below' or 'above'"):
        slices.sphere(two, "rho", radius=0.5, grid=grid)
    below = slices.sphere(two, "rho", radius=0.5, grid=grid, side="below")
    above = slices.sphere(two, "rho", radius=0.5, grid=grid, side="above")
    assert np.allclose(below.values, 10.0) and np.allclose(above.values, 4.0)
    assert below.side == "below" and above.side == "above"
    # elsewhere the side changes nothing, so a loop over radii names it once
    away = slices.sphere(two, "rho", radius=0.3, grid=grid, side="above")
    assert np.allclose(away.values, 10.0) and away.side is None
    # the outer boundary has one side
    assert np.allclose(slices.sphere(two, "rho", radius=1.0, grid=grid).values, 4.0)
    # a field a layer lacks is NaN there
    assert np.all(np.isnan(slices.sphere(two, "flow", radius=0.3, grid=grid,
                                         component="norm").values))


def test_sections_and_profiles_keep_both_sides_of_an_interface(two):
    sec = slices.section(two, "rho", (0.4, 0.1), end=(1.9, 2.0), n=6, m=11)
    assert [lay.index for lay in sec.layers] == [0, 1]
    inner, outer = sec.layers
    assert inner.radius.shape == (6, 11) and inner.values.shape == (6, 11)
    assert np.allclose(inner.radius[-1], 0.5) and np.allclose(outer.radius[0], 0.5)
    assert np.allclose(inner.values[-1], 10.0) and np.allclose(outer.values[0], 4.0)
    prof = slices.profile(two, "rho", 1.2, 0.3, n=5)
    assert [lay.radius.shape for lay in prof.layers] == [(5,), (5,)]
    assert prof.layers[0].values[-1] == 10.0 and prof.layers[1].values[0] == 4.0
    # radii bound the section
    part = slices.section(two, "rho", (0.4, 0.1), end=(1.9, 2.0), radii=(0.7, 1.0))
    assert [lay.index for lay in part.layers] == [1]
    assert np.isclose(part.layers[0].radius.min(), 0.7)


def test_a_section_is_the_field_along_its_great_circle(two):
    sec = slices.section(two, "vs", (0.6, 5.9), azimuth=0.0, length=np.pi, n=4, m=91)
    assert sec.length == pytest.approx(np.pi)
    # through the north pole and across the seam: every point on the circle,
    # the field read where it is, and continuous along the arc
    assert np.allclose(np.linalg.norm(sec.path, axis=-1), 1.0)
    assert np.allclose(sec.path @ np.cross(*sec.plane), 0.0, atol=1e-15)
    t = spherical_coordinates(sec.path)[1]
    assert t.min() < 1e-2
    lay = sec.layers[1]
    r, theta, phi, _ = spherical_coordinates(lay.reference)
    assert np.allclose(lay.values, lateral(r, theta, phi), rtol=1e-13)
    assert np.max(np.abs(np.diff(lay.values, axis=1))) < 0.05
    # the plane through the centre is a whole circle
    full = slices.section(two, "vs", (0.6, 0.0), azimuth=1.0, length=2 * np.pi, m=50)
    assert np.allclose(full.path[0], full.path[-1])
    with pytest.raises(ValueError, match="antipodal"):
        slices.section(two, "vs", (0.5, 0.0), end=(np.pi - 0.5, np.pi))
    with pytest.raises(ValueError, match="azimuth"):
        slices.section(two, "vs", (0.5, 0.0))


def test_a_vector_is_sliced_by_component_in_a_frame_or_by_its_norm(two):
    grid = gauss_legendre(6)
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    u = flow(0.7, t, p)
    z = slices.sphere(two, "flow", radius=0.7, grid=grid, component=2,
                      frame="cartesian")
    assert np.allclose(z.values, u[..., 2])
    from planetmodel.frames import spherical_frame
    ur = np.einsum("...ji,...j->...i", spherical_frame(t, p), u)[..., 0]
    radial = slices.sphere(two, "flow", radius=0.7, grid=grid, component=0,
                           frame="spherical")
    assert np.allclose(radial.values, ur)
    norm = slices.sphere(two, "flow", radius=0.7, grid=grid, component="norm")
    assert np.allclose(norm.values, np.linalg.norm(u, axis=-1))
    assert norm.label == "|flow|"
    assert z.label == "flow[2] (cartesian)"
    with pytest.raises(ValueError, match="frame"):
        slices.sphere(two, "flow", radius=0.7, grid=grid, component=1)
    with pytest.raises(ValueError, match="rank 1"):
        slices.sphere(two, "flow", radius=0.7, grid=grid)
    with pytest.raises(ValueError, match="scalar"):
        slices.sphere(two, "vs", radius=0.7, grid=grid, component=0, frame="spherical")


def test_a_departure_from_the_mean_is_zero_in_mean(two):
    grid = gauss_legendre(10)
    for departure in ("difference", "percent"):
        d = slices.sphere(two, "vs", radius=0.9, grid=grid, relative_to="mean",
                          departure=departure)
        assert abs(grid.mean(d.values)) < 1e-12
    pct = slices.sphere(two, "vs", radius=0.9, grid=grid, relative_to="mean")
    mean = grid.mean(lateral(0.9, *np.meshgrid(grid.colatitudes, grid.longitudes,
                                                indexing="ij")))
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    assert np.allclose(pct.values, 100.0 * (lateral(0.9, t, p) / mean - 1.0))
    assert pct.label == "vs, percent from the mean [%]"
    assert pct.shown.is_departure
    # a section's mean is taken on the grid it is given
    with pytest.raises(ValueError, match="mean_grid"):
        slices.section(two, "vs", (0.5, 0.0), end=(1.0, 1.0), relative_to="mean")
    sec = slices.section(two, "vs", (0.5, 0.0), end=(1.0, 1.0), relative_to="mean",
                         departure="difference", mean_grid=grid)
    assert np.all(np.isfinite(sec.layers[1].values))
    with pytest.raises(ValueError, match="relative_to"):
        slices.sphere(two, "vs", radius=0.9, grid=grid, departure="percent")


def test_a_departure_from_a_reference_or_another_model(two):
    grid = gauss_legendre(6)
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    base = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                   vs=[4.0, 3.0]).named("base")
    dp = AnalyticField((0.5, 1.0), lambda r, t, p: 0.02 * np.cos(t) * r)
    m = perturbed(base, "vs", dp, kind="relative")
    rel = slices.sphere(m, "vs", radius=0.8, grid=grid, relative_to="reference")
    assert np.allclose(rel.values, 100.0 * 0.02 * np.cos(t) * 0.8)
    assert rel.label == "vs, percent from the reference [%]"
    with pytest.raises(ValueError, match="not perturbed"):
        slices.sphere(two, "vs", radius=0.8, grid=grid, relative_to="reference")
    # the core is not perturbed: a radial field is its own reference
    core = slices.sphere(m, "vs", radius=0.3, grid=grid, relative_to="reference")
    assert np.allclose(core.values, 0.0)
    # against another model: the same field there, layer by layer
    vs = slices.sphere(two, "vs", radius=0.8, grid=grid, relative_to=base,
                       departure="log10")
    assert np.allclose(vs.values, np.log10(lateral(0.8, t, p) / 3.0))
    assert vs.label == "vs, log10 of the ratio to base"
    # on the interface the other model is read on the slice's own side
    rho = slices.section(two, "rho", (0.4, 0.1), end=(1.9, 2.0), relative_to=base,
                         departure="difference")
    assert all(np.allclose(lay.values, 0.0) for lay in rho.layers)


def test_slices_at_reference_points_give_their_physical_positions(two):
    flat = two.stretched(flattening(0.05))
    grid = gauss_legendre(6)
    s = slices.sphere(flat, "vs", radius=0.8, grid=grid)
    assert np.allclose(s.physical, flat.geometry.mapping(s.reference))
    assert not np.allclose(s.physical, s.reference)
    sec = slices.section(flat, "rho", (0.4, 0.1), end=(1.9, 2.0), n=5, m=7)
    for lay in sec.layers:
        assert np.allclose(lay.physical, flat.geometry.mapping(lay.reference))


def test_slices_at_physical_points_lie_where_they_were_asked_for(two):
    flat = two.stretched(flattening(0.05))
    mapping = flat.geometry.mapping
    grid = gauss_legendre(6)
    s = slices.sphere(flat, "vs", radius=0.8, grid=grid, at="physical")
    assert np.allclose(np.linalg.norm(s.physical, axis=-1), 0.8)
    assert np.allclose(mapping(s.reference), s.physical)
    r, t, p, _ = spherical_coordinates(s.reference)
    assert np.allclose(s.values, lateral(r, t, p))
    # a physical sphere crossing the flattened interface takes a side there
    # only where a point lies on it, and is NaN beyond the body
    rho = slices.sphere(flat, "rho", radius=0.5, grid=grid, at="physical")
    inner = np.linalg.norm(rho.reference, axis=-1) < 0.5
    assert np.allclose(rho.values[inner], 10.0) and np.allclose(rho.values[~inner], 4.0)
    # on an interface that stays spherical every point needs the side
    with pytest.raises(ValueError, match="side="):
        slices.sphere(two, "rho", radius=0.5, grid=grid, at="physical")
    on = slices.sphere(two, "rho", radius=0.5, grid=grid, at="physical", side="below")
    assert np.allclose(on.values, 10.0) and on.side == "below"
    assert slices.sphere(two, "rho", radius=0.3, grid=grid, at="physical",
                         side="below").side is None
    beyond = slices.sphere(flat, "rho", radius=0.99, grid=grid, at="physical")
    assert np.any(np.isnan(beyond.values)) and np.any(np.isfinite(beyond.values))
    # a physical section: each layer's run between its mapped boundaries
    sec = slices.section(flat, "rho", (0.4, 0.1), end=(1.9, 2.0), n=5, m=7,
                         at="physical")
    for lay in sec.layers:
        assert np.allclose(np.linalg.norm(lay.physical, axis=-1), lay.radius)
        assert np.allclose(mapping(lay.reference), lay.physical)
    ends = np.linalg.norm(mapping(0.5 * sec.path), axis=-1)
    assert np.allclose(sec.layers[0].radius[-1], ends)
    assert np.allclose(sec.layers[1].radius[0], ends)
    prof = slices.profile(flat, "rho", 0.3, 1.0, at="physical", n=4)
    assert np.isclose(prof.layers[0].radius[-1],
                      np.linalg.norm(mapping(cartesian_points(0.5, 0.3, 1.0))))


class _Twist(MappingBase):
    """A small rotation about the axis growing with radius: no inverse."""

    def __call__(self, X):
        X = np.asarray(X, dtype=float)
        a = 0.01 * np.linalg.norm(X, axis=-1)
        c, s = np.cos(a), np.sin(a)
        return np.stack([c * X[..., 0] - s * X[..., 1], s * X[..., 0] + c * X[..., 1],
                         X[..., 2]], axis=-1)


def test_a_mapping_without_an_inverse_is_refused_at_physical_points(two):
    twisted = two.with_mapping(_Twist(), check=False)
    grid = gauss_legendre(4)
    with pytest.raises(ValueError, match="does not invert itself"):
        slices.sphere(twisted, "vs", radius=0.8, grid=grid, at="physical")
    s = slices.sphere(twisted, "vs", radius=0.8, grid=grid)
    assert np.allclose(s.physical, _Twist()(s.reference))
    with pytest.raises(ValueError, match="off their rays"):
        slices.section(twisted, "vs", (0.5, 0.0), end=(1.0, 1.0), at="physical")


def test_the_mean_profile_and_the_labels_of_units(two):
    grid = gauss_legendre(10)
    mp = slices.mean_profile(two, "vs", mean_grid=grid, n=6)
    assert mp.theta is None and mp.label == "vs, spherical mean [m s-1]"
    assert np.allclose(mp.layers[0].values, 4.0)
    assert np.allclose(mp.layers[1].values[2], grid.mean(lateral(
        mp.layers[1].radius[2], *np.meshgrid(grid.colatitudes, grid.longitudes,
                                             indexing="ij"))))
    with pytest.raises(ValueError, match="mean_grid"):
        slices.mean_profile(two, "vs")
    scaled = two.without_field("flow").converted(Scales.geophysical(1.0))
    assert slices.sphere(scaled, "rho", radius=0.3,
                         grid=grid).label.startswith("rho [5515 kg m-3]")
    by_layer = slices.sphere(two, {1: AnalyticField((0.5, 1.0), lateral,
                                                    character=DENSITY)},
                             radius=0.8, grid=grid)
    assert by_layer.label == "field" and np.all(np.isfinite(by_layer.values))


# -- spectra ------------------------------------------------------------------

def _harmonic(coeffs):
    """The field of real orthonormal coefficients, by the scipy basis."""
    from planetmodel.harmonics import synthesise
    c = np.asarray(coeffs, dtype=float)
    return lambda r, t, p: r * synthesise(c, t, p)


def _with(model, layer, fn, name="vs"):
    lo, hi = model.skeleton.interval(layer)
    return model.with_field(layer, name, AnalyticField((lo, hi), fn, name=name),
                            replace=True)


@pytest.fixture
def harmonic_model(two):
    c = np.zeros((2, 5, 5))
    c[0, 0, 0], c[0, 2, 1], c[1, 3, 2] = 2.0, 0.5, -0.25
    return _with(two, 1, _harmonic(c)), c


def test_the_power_of_a_field_is_at_its_degrees_and_sums_to_its_mean_square(
        harmonic_model):
    m, c = harmonic_model
    sp = slices.spectrum(m, "vs", radius=0.8, lmax=6)
    want = np.zeros(7)
    want[0], want[2], want[3] = (0.8 * 2.0) ** 2, (0.8 * 0.5) ** 2, (0.8 * 0.25) ** 2
    assert np.allclose(sp.power, want / (4 * np.pi), atol=1e-14)
    assert np.allclose(sp.amplitude ** 2, sp.power)
    grid = gauss_legendre(12)
    values = slices.sphere(m, "vs", radius=0.8, grid=grid).values
    assert np.isclose(sp.power.sum(), grid.mean(values ** 2), rtol=1e-12)
    assert sp.degrees.tolist() == list(range(7)) and sp.label == "vs [m s-1]"
    # a departure from the mean has no mean
    dep = slices.spectrum(m, "vs", radius=0.8, lmax=6, relative_to="mean",
                          departure="difference")
    assert dep.power[0] < 1e-28 and np.allclose(dep.power[1:], sp.power[1:])


def test_the_coefficients_are_the_scipy_basis_by_quadrature(two):
    from planetmodel.harmonics import real_harmonics
    m = _with(two, 1, lateral)
    grid = gauss_legendre(8)
    sp = slices.spectrum(m, "vs", radius=0.7, lmax=4, grid=grid)
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    Y = real_harmonics(4, t, p)
    dphi = 2 * np.pi / grid.nphi
    direct = np.einsum("...ij,i->...", Y * lateral(0.7, t, p), grid.weights) * dphi
    assert np.allclose(sp.coefficients, direct, atol=1e-13)


def test_a_spectrum_wants_a_grid_with_a_band_and_the_field_everywhere(two):
    from planetmodel import AngularGrid
    with pytest.raises(ValueError, match="band"):
        slices.spectrum(two, "vs", radius=0.8, lmax=6, grid=gauss_legendre(4))
    with pytest.raises(ValueError, match="band"):
        slices.spectrum(two, "vs", radius=0.8, lmax=2,
                        grid=AngularGrid(np.linspace(0.1, 3.0, 6),
                                         np.linspace(0.0, 6.0, 12)))
    with pytest.raises(ValueError, match="everywhere"):
        slices.spectrum(two, "flow", radius=0.3, lmax=2, component="norm")


def test_a_field_correlates_to_one_with_itself_and_to_zero_with_an_orthogonal_one(
        two):
    a = np.zeros((2, 4, 4))
    b = np.zeros((2, 4, 4))
    a[0, 2, 1], a[0, 3, 0] = 1.0, 0.5
    b[1, 2, 1], b[0, 3, 0] = 0.7, -2.0
    ma = _with(two, 1, _harmonic(a)).named("a")
    mb = _with(two, 1, _harmonic(b)).named("b")
    same = slices.correlation(ma, "vs", ma, lmax=3, radii=[0.6, 0.9])
    assert np.allclose(same.by_degree[:, 2:], 1.0) and np.allclose(same.total, 1.0)
    assert np.all(np.isnan(same.by_degree[:, 1]))
    cr = slices.correlation(ma, "vs", mb, lmax=3, depths=[0.4])
    assert np.isclose(cr.radii[0], 0.6)
    assert np.isclose(cr.by_degree[0, 2], 0.0, atol=1e-13)
    assert np.isclose(cr.by_degree[0, 3], -1.0)
    assert np.isnan(cr.by_degree[0, 1])
    # the total over degrees 1 to lmax, weighted by power
    pa, pb = cr.first[0].power, cr.second[0].power
    want = (0.6 ** 2 * 0.5 * -2.0) / np.sqrt(pa[1:].sum() * pb[1:].sum()) / (4 * np.pi)
    assert np.isclose(cr.total[0], want)
    assert cr.label == "correlation of vs"
    # two fields of one model
    both = slices.correlation(ma, "vs", ma, other_fields="vs", lmax=3, radii=[0.8])
    assert np.allclose(both.by_degree[0, 2:], 1.0)
    with pytest.raises(ValueError, match="exactly one"):
        slices.correlation(ma, "vs", mb, lmax=3)


# -- kernels ------------------------------------------------------------------

def _cubic(r, t, p):
    """Cubic in r, signed across the sphere."""
    return (1.0 + r + r ** 3) * (np.cos(t) + 0.5 * np.sin(t) * np.cos(p))


@pytest.fixture
def kernel(two):
    """`two` with a kernel K, cubic in r on the outer layer and held there
    on gridded nodes that hold it exactly, of length^-2 by its spec."""
    from planetmodel import FieldSpec, GriddedField, SCALAR
    from planetmodel.units import Dimensions
    grid = gauss_legendre(6)
    k = GriddedField.sampled(AnalyticField((0.5, 1.0), _cubic), grid,
                             breaks=[0.5, 0.7, 1.0], ngll=4, lateral="linear")
    m = two.with_field(1, "K", k)
    return m.replaced(specs={**m.specs, "K": FieldSpec(SCALAR, Dimensions(length=-2))})


def test_an_integral_along_the_radius_is_its_closed_form(kernel):
    grid = gauss_legendre(6)
    t, p = np.meshgrid(grid.colatitudes, grid.longitudes, indexing="ij")
    angular = np.cos(t) + 0.5 * np.sin(t) * np.cos(p)

    def volume(a, b):
        return (b ** 3 / 3 + b ** 4 / 4 + b ** 6 / 6) - (a ** 3 / 3 + a ** 4 / 4
                                                        + a ** 6 / 6)

    whole = slices.integrated(kernel, "K", grid=grid)
    assert whole.radius == 1.0 and whole.grid == grid
    assert np.allclose(whole.values, volume(0.5, 1.0) * angular, rtol=0, atol=1e-14)
    assert whole.label == "K, integrated along the radius, r^2 dr [m]"
    part = slices.integrated(kernel, "K", grid=grid, radii=(0.6, 0.9))
    assert np.allclose(part.values, volume(0.6, 0.9) * angular, rtol=0, atol=1e-14)
    along = slices.integrated(kernel, "K", grid=grid, weight="radius")
    length = (1.0 + 1.0 / 2 + 1.0 / 4) - (0.5 + 0.125 + 0.5 ** 4 / 4)
    assert np.allclose(along.values, length * angular, rtol=0, atol=1e-14)
    assert along.label == "K, integrated along the radius, dr [m-1]"
    # a radial field of pieces, across both layers
    rho = slices.integrated(kernel, "rho", grid=grid)
    assert np.allclose(rho.values, (10.0 * 0.125 + 4.0 * 0.875) / 3, rtol=1e-14)
    # a field declaring no pieces, by n points on its layer
    smooth = slices.integrated(kernel, "vs", grid=grid, radii=(0.5, 1.0))
    exact = (4.0 * (1 - 0.125) / 3 + 0.3 * np.cos(t) * (1 - 0.5 ** 4) / 4
             + 0.2 * np.sin(t) * np.cos(p) * (1 - 0.5 ** 4) / 4)
    assert np.allclose(smooth.values, exact, rtol=1e-13)
    with pytest.raises(ValueError, match="weight"):
        slices.integrated(kernel, "K", grid=grid, weight="area")
    with pytest.raises(ValueError, match="no layer"):
        slices.integrated(kernel, "K", grid=grid, radii=(0.1, 0.4))


def test_the_total_profile_is_r2_times_the_integral_over_the_sphere(two):
    grid = gauss_legendre(10)
    mean = slices.mean_profile(two, "vs", mean_grid=grid, n=6)
    total = slices.mean_profile(two, "vs", mean_grid=grid, n=6, total=True)
    for a, b in zip(mean.layers, total.layers):
        assert np.allclose(b.values, 4.0 * np.pi * a.radius ** 2 * a.values,
                           rtol=1e-15)
    assert total.label == "vs, r^2 times the integral over the sphere [m3 s-1]"


def test_a_function_on_an_interface_is_a_sphere_slice_at_its_radius(kernel):
    from planetmodel import FieldSpec, GriddedFunction, SCALAR, flattening
    from planetmodel.units import Dimensions

    def shape(t, p):
        return np.sin(t) ** 2 * np.cos(2 * p) - 0.2 * np.cos(t)

    gf = GriddedFunction.sampled(shape, gauss_legendre(8))
    m = kernel.replaced(specs={**kernel.specs,
                               "K_shape": FieldSpec(SCALAR, Dimensions(length=-2))})
    s = slices.surface(m, gf, 0, name="K_shape")
    assert s.radius == 0.5 and s.grid == gf.grid
    assert np.array_equal(s.values, gf.values)
    assert s.label == "K_shape, on interface 0 [m-2]"
    assert np.allclose(np.linalg.norm(s.reference, axis=-1), 0.5)
    plain = slices.surface(m, shape, -1, grid=gauss_legendre(4))
    t, p = np.meshgrid(plain.grid.colatitudes, plain.grid.longitudes, indexing="ij")
    assert plain.radius == 1.0 and np.allclose(plain.values, shape(t, p))
    assert plain.label == "field, on interface 1"
    assert slices.surface(m, shape, 1).grid.ntheta == 180
    flat = m.stretched(flattening(0.1))
    moved = slices.surface(flat, gf, 1)
    assert not np.allclose(np.linalg.norm(moved.physical, axis=-1), 1.0)
    with pytest.raises(KeyError):
        slices.surface(m, shape, "cmb")


def test_any_sphere_slice_has_a_spectrum(kernel):
    from planetmodel import GriddedFunction
    from planetmodel.harmonics import synthesise
    c = np.zeros((2, 4, 4))
    c[0, 2, 1], c[1, 3, 3] = 1.5, -0.5
    gf = GriddedFunction.sampled(lambda t, p: synthesise(c, t, p), gauss_legendre(6))
    sp = slices.spectrum_of(slices.surface(kernel, gf, 1), lmax=3)
    assert np.allclose(sp.power, [0, 0, 1.5 ** 2 / (4 * np.pi), 0.25 / (4 * np.pi)],
                       atol=1e-14)
    column = slices.spectrum_of(slices.integrated(kernel, "K", grid=gauss_legendre(6)),
                                lmax=3)
    assert column.power[1] > 0 and np.allclose(column.power[2:], 0, atol=1e-28)
    with pytest.raises(ValueError, match="band"):
        slices.spectrum_of(slices.surface(kernel, gf, 1), lmax=7)


def test_a_normalised_slice_is_divided_by_its_largest_value(kernel):
    grid = gauss_legendre(6)
    s = slices.normalised(slices.integrated(kernel, "K", grid=grid))
    assert np.isclose(np.max(np.abs(s.values)), 1.0)
    assert s.label == "K, integrated along the radius, r^2 dr, normalised"
    sec = slices.normalised(slices.section(kernel, "K", (0.6, 0.2), azimuth=1.0,
                                           length=1.2, n=5, m=11))
    held = np.concatenate([lay.values[np.isfinite(lay.values)] for lay in sec.layers])
    assert np.isclose(np.max(np.abs(held)), 1.0) and sec.label == "K, normalised"
    dep = slices.normalised(slices.profile(kernel, "vs", 1.0, 2.0, n=5,
                                           relative_to="mean", mean_grid=grid))
    assert dep.label == "vs, percent from the mean, normalised"
    assert np.isclose(np.nanmax(np.abs(dep.layers[1].values)), 1.0)
    with pytest.raises(ValueError, match="zero"):
        slices.normalised(slices.surface(kernel, lambda t, p: 0.0 * t, 1, grid=grid))


# -- cutaways -----------------------------------------------------------------------

def _solid(q, outer, inner, longitudes, colatitudes):
    """Whether points lie in the cut body."""
    r = np.linalg.norm(q, axis=-1)
    theta = np.arccos(np.clip(q[..., 2] / np.where(r > 0, r, 1.0), -1.0, 1.0))
    phi = np.arctan2(q[..., 1], q[..., 0])
    start, end = longitudes
    in_cut = ((np.mod(phi - start, 2 * np.pi) < np.mod(end - start, 2 * np.pi))
              & (theta > colatitudes[0]) & (theta < colatitudes[1]))
    return (r <= outer) & ~((r > inner) & in_cut)


def _which_face(q, outer, inner, longitudes, colatitudes, tol):
    """The one face a point lies on, or None where that is not clear."""
    r = np.linalg.norm(q)
    theta = np.arccos(np.clip(q[2] / r, -1.0, 1.0))
    on = []
    if abs(r - outer) < tol:
        on.append("surface")
    if abs(r - inner) < tol:
        on.append("inner")
    for name, phi in zip(("first", "second"), longitudes):
        along = np.array([np.cos(phi), np.sin(phi), 0.0])
        normal = np.array([-np.sin(phi), np.cos(phi), 0.0])
        if abs(normal @ q) < tol and q @ along > tol:
            on.append(name)
    for name, lat in zip(("upper", "lower"), colatitudes):
        if abs(theta - lat) * r < tol:
            on.append(name)
    return on[0] if len(on) == 1 else None


@pytest.mark.parametrize("cut", [
    dict(longitudes=(0.3, 1.8), colatitudes=(0.0, np.pi), inner=0.0),
    dict(longitudes=(0.3, 1.8), colatitudes=(0.4, 2.5), inner=0.5),
    dict(longitudes=(-1.0, 2.8), colatitudes=(0.5 * np.pi, np.pi), inner=0.5),
    dict(longitudes=(2.0, 1.0), colatitudes=(0.0, 0.5 * np.pi), inner=0.3),
])
@pytest.mark.parametrize("view", [(1.1, 1.0), (0.5 * np.pi, 1.0), (2.4, 4.0),
                                  (0.0, 0.0)])
def test_each_pixel_of_a_cutaway_sees_the_first_point_of_the_cut_body(two, cut, view):
    """Against a march along each of a sample of rays, from the viewer
    into the body, to the first point of the cut body: the hit lies
    within a step of it, on the face the march's point is on."""
    outer, inner = 1.0, cut["inner"]
    c = slices.cutaway(two, "vs", view=view, n=40, **cut)
    assert c.face.shape == (40, 40) and set(np.unique(c.face)) <= set(range(7))
    v = c.view[2]
    rng = np.random.default_rng(0)
    steps = np.linspace(outer, -outer, 20001)
    checked = 0
    for i, j in rng.integers(0, 40, size=(60, 2)):
        X = c.reference[i, j]
        x = -outer + (j + 0.5) * 2.0 * outer / 40
        y = outer - (i + 0.5) * 2.0 * outer / 40
        p = x * c.view[0] + y * c.view[1]
        q = p[None] + steps[:, None] * v[None]
        solid = _solid(q, outer, inner, cut["longitudes"], cut["colatitudes"])
        if not np.any(solid):
            assert c.face[i, j] == 0 and np.all(np.isnan(X))
            continue
        first = q[np.argmax(solid)]
        if c.face[i, j] == 0:                         # a grazing ray
            assert np.linalg.norm(first - q[0]) < 2e-3 or not solid[-1]
            continue
        assert np.linalg.norm(X - first) < 2e-4
        face = _which_face(X, outer, inner, cut["longitudes"], cut["colatitudes"], 1e-6)
        if face is not None:
            assert slices.CUTAWAY_FACES[c.face[i, j]] == face
            checked += 1
    assert checked > 10


def test_a_cutaway_holds_the_field_at_the_points_each_pixel_sees(two):
    c = slices.cutaway(two, "vs", longitudes=(0.3, 1.8), colatitudes=(0.4, 2.5),
                       inner=0.5, n=60)
    assert c.label == "vs [m s-1]" and c.at == "reference" and c.side == "below"
    assert c.n == 60 and c.extent == 1.0 and c.outer == 1.0
    assert np.array_equal(c.boundaries, [0.0, 0.5, 1.0])
    seen = c.face > 0
    assert np.all(np.isnan(c.values[~seen])) and np.all(np.isnan(c.reference[~seen]))
    assert np.array_equal(c.physical[seen], c.reference[seen])
    r, t, p, _ = spherical_coordinates(c.reference[seen])
    assert np.all(r <= 1.0 + 1e-12)
    faces = (c.mask("first") | c.mask("second") | c.mask("upper") | c.mask("lower"))
    for name in ("first", "second", "upper", "lower", "inner", "surface"):
        assert np.any(c.mask(name)), name
    r, t, p, _ = spherical_coordinates(c.reference[faces])
    assert np.all((r > 0.5) & (r < 1.0))
    assert np.allclose(c.values[faces], lateral(r, t, p), rtol=1e-12)
    r, t, p, _ = spherical_coordinates(c.reference[c.mask("surface")])
    assert np.allclose(r, 1.0) and np.allclose(c.values[c.mask("surface")],
                                               lateral(1.0, t, p), rtol=1e-12)
    assert np.allclose(np.linalg.norm(c.reference[c.mask("inner")], axis=-1), 0.5)
    assert np.all(c.values[c.mask("inner")] == 4.0)      # the layer below
    above = slices.cutaway(two, "vs", longitudes=(0.3, 1.8), colatitudes=(0.4, 2.5),
                           inner=0.5, n=60, side="above")
    r, t, p, _ = spherical_coordinates(above.reference[above.mask("inner")])
    assert np.allclose(above.values[above.mask("inner")], lateral(0.5, t, p),
                       rtol=1e-12)
    assert above.side == "above"
    # the inner radius off an interface takes no side
    assert slices.cutaway(two, "vs", inner=0.7, n=20).side is None
    # an interface by name
    from planetmodel import PREM
    prem = PREM()
    assert slices.cutaway(prem, "vsv", inner="cmb", n=10).inner == 3480e3
    assert slices.cutaway(prem, "vsv", inner="icb", n=10).inner == 1221.5e3
    with pytest.raises(KeyError, match="no interface named"):
        slices.cutaway(prem, "vsv", inner="core", n=10)
    # a departure, a component, a normalised slice
    d = slices.cutaway(two, "vs", inner=0.5, n=30, relative_to="mean",
                       mean_grid=gauss_legendre(16))
    assert d.shown.is_departure and "percent from the mean" in d.label
    f = slices.cutaway(two, "flow", inner=0.5, n=30, component=2, frame="cartesian")
    assert "flow[2]" in f.label
    m = slices.normalised(c)
    assert m.label == "vs, normalised" and np.nanmax(np.abs(m.values)) == 1.0
    with pytest.raises(ValueError, match="name must be one of"):
        c.mask("core")


def test_a_cutaway_covers_the_disc_seen_from_the_equator_into_the_wedge(two):
    """With the whole wedge cut to the centre and the viewer in the
    equatorial plane at its middle, the faces fill the notch and the
    silhouette is the whole disc: the coverage sums to its area."""
    c = slices.cutaway(two, "rho", longitudes=(0.0, 0.5 * np.pi),
                       view=(0.5 * np.pi, 0.25 * np.pi), n=200)
    assert np.all((c.coverage >= 0.0) & (c.coverage <= 1.0))
    assert np.all(c.coverage[c.face > 0] > 0.0)
    area = c.coverage.sum() * (2.0 / 200) ** 2
    assert abs(area - np.pi) < 2e-3 * np.pi
    assert np.any((c.coverage > 0.0) & (c.coverage < 1.0))   # the limb
    assert np.isclose(c.coverage[100, 100], 1.0) and c.coverage[0, 0] == 0.0
    # what is seen: in front and not cut away, or on a face, and nothing behind
    v = c.view[2]
    assert not c.visible(v)                         # the surface in the wedge
    assert not c.visible(-v)
    assert c.visible(cartesian_points(1.0, 0.5 * np.pi, -0.5))   # outside the wedge
    assert c.visible(cartesian_points(0.8, 0.5 * np.pi, 0.0))    # on the first face
    assert not c.visible(cartesian_points(0.8, 0.5 * np.pi, np.pi))  # behind
    assert not c.visible(0.5 * v)                   # inside the body


def test_a_cutaway_refuses_what_cuts_nothing_or_cannot_be_seen(two):
    with pytest.raises(ValueError, match="cut nothing"):
        slices.cutaway(two, "vs", longitudes=(1.0, 1.0))
    with pytest.raises(ValueError, match="colatitudes"):
        slices.cutaway(two, "vs", colatitudes=(2.0, 1.0))
    with pytest.raises(ValueError, match="inner"):
        slices.cutaway(two, "vs", inner=1.0)
    with pytest.raises(ValueError, match="n must"):
        slices.cutaway(two, "vs", n=1)
    with pytest.raises(ValueError, match="side"):
        slices.cutaway(two, "vs", side="left")
    with pytest.raises(ValueError, match="side='below' or 'above'"):
        slices.cutaway(two, "vs", inner=0.5, n=10, side=None)
    # a view along the axis is well defined
    c = slices.cutaway(two, "vs", view=(0.0, 0.0), n=20)
    assert np.allclose(c.view[2], [0.0, 0.0, 1.0]) and np.any(c.face > 0)


def test_a_shell_cut_to_the_centre_shows_its_inner_surface():
    shell = LayeredIsotropicElastic([0.5, 1.0], rho=[4.0], vp=[7.0], vs=[3.0])
    shell = shell.with_field(0, "vs", AnalyticField((0.5, 1.0), lateral, name="vs"),
                             replace=True)
    c = slices.cutaway(shell, "vs", inner=0.0, n=60, longitudes=(0.3, 1.8))
    assert c.inner == 0.5 and c.side is None
    assert np.all(np.isfinite(c.values[c.face > 0]))
    core = c.mask("inner")
    assert np.any(core)
    r, t, p, _ = spherical_coordinates(c.reference[core])
    assert np.allclose(r, 0.5) and np.allclose(c.values[core], lateral(0.5, t, p))
