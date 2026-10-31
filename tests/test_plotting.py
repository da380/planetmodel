"""The radial profile helpers: radius upward, a joiner at every jump."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from planetmodel import PREM, LayeredIsotropicElastic, constant_field  # noqa: E402
from planetmodel.plotting import profile, radial_profile  # noqa: E402


def test_radius_is_vertical_and_every_jump_gets_a_joiner():
    m = LayeredIsotropicElastic([0.0, 0.5, 0.8, 1.0], rho=[3.0, 2.0, 2.0],
                                vp=[2.0, 2.0, 2.0], vs=[1.0, 0.0, 1.0])
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "rho", scale=2.0, value_scale=10.0, label="d")
    # three segments and two joiners (the second jump is zero-height, still drawn)
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert len(segments) == 3 and len(joiners) == 2
    x, y = segments[0].get_data()
    assert np.allclose(x, 30.0) and y[0] == 0.0 and np.isclose(y[-1], 1.0)
    jx, jy = joiners[0].get_data()
    assert np.allclose(jy, 1.0) and list(jx) == [30.0, 20.0]
    assert segments[0].get_label() == "d" and segments[1].get_label().startswith("_")
    assert len({ln.get_color() for ln in lines}) == 1
    assert all(ln.get_linestyle() == "-" for ln in lines)
    # the guides at the four boundaries sit beneath every profile line
    guides = [ln for ln in ax.get_lines() if ln not in lines]
    assert sorted(ln.get_ydata()[0] for ln in guides) == [0.0, 1.0, 1.6, 2.0]
    assert all(g.get_zorder() < ln.get_zorder() for g in guides for ln in lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    radial_profile(ax, m, "rho", boundaries=False)
    assert len(ax.get_lines()) == 5
    plt.close(fig)


def test_a_missing_layer_is_a_gap_without_a_joiner():
    m = PREM(ocean=False)
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "qmu", joiners=True)
    held = m.layers_with("qmu")
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert len(segments) == len(held)
    # no joiner across the outer core, which lacks qmu
    cmb = m.geometry.interface("cmb").radius
    icb = m.geometry.interface("icb").radius
    assert all(ln.get_data()[1][0] not in (icb, cmb) for ln in joiners)
    plt.close(fig)


def test_refusals():
    m = LayeredIsotropicElastic([0.0, 1.0], rho=[1.0], vp=[1.0], vs=[1.0])
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="complex"):
        profile(ax, [constant_field((0.0, 1.0), 1.0 + 1j)])
    with pytest.raises(ValueError, match="radial field of rank 0"):
        profile(ax, [m.elastic_moduli(0)])
    plt.close(fig)


# -- slices drawn -----------------------------------------------------------

from planetmodel import AnalyticField, flattening, gauss_legendre  # noqa: E402
from planetmodel import slices  # noqa: E402
from planetmodel.plotting import (BOUNDARY_STYLE, DIVERGING, SEQUENTIAL,  # noqa: E402
                                  colour_extend, colour_scale, section)


def _lateral(r, t, p):
    return 3.0 + 0.3 * r * np.cos(t) + 0.2 * r * np.sin(t) * np.cos(p)


@pytest.fixture
def layered():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    return m.with_field(1, "vs", AnalyticField((0.5, 1.0), _lateral, name="vs"),
                        replace=True)


def test_a_profile_slice_is_drawn_layer_by_layer_with_its_step(layered):
    prof = slices.profile(layered, "rho", 1.0, 2.0, n=7)
    fig, ax = plt.subplots()
    lines = profile(ax, prof, scale=2.0)
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    assert len(segments) == 2 and len(joiners) == 1
    x, y = segments[1].get_data()
    assert np.allclose(x, prof.layers[1].values)
    assert np.allclose(y, 2.0 * prof.layers[1].radius)
    assert list(joiners[0].get_data()[0]) == [10.0, 4.0]
    mean = slices.mean_profile(layered, "vs", mean_grid=gauss_legendre(8), n=5)
    assert len(profile(fig.add_subplot(2, 1, 2), mean, joiners=False)) == 2
    plt.close(fig)


def test_a_section_is_a_wedge_of_the_annulus_on_polar_axes(layered):
    sec = slices.section(layered, "vs", (0.6, 0.2), azimuth=1.0, length=1.2, n=5,
                         m=31, relative_to="mean", mean_grid=gauss_legendre(10))
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    meshes = section(ax, sec, scale=10.0, colorbar=True)
    assert len(meshes) == 2
    # the layers share one scale, symmetric about zero for a departure
    lims = {mesh.get_clim() for mesh in meshes}
    assert len(lims) == 1 and np.isclose(-lims.pop()[0], meshes[0].get_clim()[1])
    assert meshes[0].get_cmap().name == DIVERGING
    coords = meshes[1].get_coordinates()
    assert np.allclose(coords[..., 0], sec.distance[None, :] - 0.6)
    assert np.allclose(coords[..., 1], 10.0 * sec.layers[1].radius)
    assert np.allclose(np.asarray(meshes[1].get_array()).reshape(5, 31),
                       sec.layers[1].values)
    assert np.isclose(ax.get_thetamin(), -np.degrees(0.6))
    assert np.isclose(ax.get_thetamax(), np.degrees(0.6))
    assert np.allclose(ax.get_ylim(), (0.0, 10.0))
    # the interfaces are faint arcs, and the colour bar names the slice
    arcs = [ln for ln in ax.get_lines() if ln.get_color() == BOUNDARY_STYLE["color"]]
    assert len(arcs) == 4
    assert fig.axes[-1].get_ylabel() == sec.label
    plt.close(fig)


def test_a_whole_circle_is_a_disc_and_the_body_can_be_drawn_as_it_is(layered):
    full = slices.section(layered, "rho", (0.6, 0.0), azimuth=0.5, length=2 * np.pi,
                          n=4, m=40)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    section(ax, full)
    assert np.isclose(ax.get_thetamax() - ax.get_thetamin(), 360.0)
    assert section(ax, full)[0].get_cmap().name == SEQUENTIAL
    plt.close(fig)
    flat = layered.stretched(flattening(0.1))
    sec = slices.section(flat, "rho", (0.3, 0.0), end=(2.0, 0.0), n=4, m=21)
    fig, ax = plt.subplots()
    meshes = section(ax, sec, physical=True)
    xy = meshes[0].get_coordinates()
    assert np.allclose(np.hypot(xy[..., 0], xy[..., 1]),
                       np.linalg.norm(sec.layers[0].physical, axis=-1))
    assert ax.get_aspect() == 1.0
    with pytest.raises(ValueError, match="polar"):
        section(ax, sec)
    polar = fig.add_subplot(2, 1, 2, projection="polar")
    with pytest.raises(ValueError, match="polar"):
        section(polar, sec, physical=True)
    at = slices.section(flat, "rho", (0.3, 0.0), end=(2.0, 0.0), n=4, m=21,
                        at="physical")
    assert len(section(polar, at)) == 2
    with pytest.raises(ValueError, match="physical points already"):
        section(ax, at, physical=True)
    plt.close(fig)


def test_the_colour_scale_is_the_callers_or_by_what_the_values_are(layered):
    dep = slices.sphere(layered, "vs", radius=0.8, grid=gauss_legendre(4),
                        relative_to="mean")
    cmap, lo, hi = colour_scale([dep.values], dep.shown)
    assert cmap == DIVERGING and lo == -hi == -np.max(np.abs(dep.values))
    plain = slices.sphere(layered, "rho", radius=0.8, grid=gauss_legendre(4))
    cmap, lo, hi = colour_scale([plain.values], plain.shown)
    assert cmap == SEQUENTIAL and lo < 4.0 < hi
    assert colour_scale([plain.values], plain.shown, cmap="magma", vmin=0.0,
                        vmax=9.0) == ("magma", 0.0, 9.0)
    with pytest.raises(ValueError, match="no finite"):
        colour_scale([np.full(3, np.nan)], plain.shown)


def _signed(r, t, p):
    """Signed, with a spike near one point, as a kernel is beside a source."""
    return (r * (0.5 * np.cos(t) + np.sin(t) * np.sin(p))
            + 50.0 * np.exp(-((t - 0.5 * np.pi) ** 2 + p ** 2) / 1e-3))


def test_a_signed_field_is_drawn_symmetric_and_clipped_at_a_percentile(layered):
    m = layered.with_field(1, "K", AnalyticField((0.5, 1.0), _signed, name="K"))
    sec = slices.section(m, "K", (np.pi / 2, -0.6), azimuth=0.5 * np.pi,
                         length=1.2, n=5, m=41)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    meshes = section(ax, sec, diverging=True, colorbar=True)
    held = np.concatenate([lay.values[np.isfinite(lay.values)] for lay in sec.layers])
    lo, hi = meshes[0].get_clim()
    assert lo == -hi == -np.max(np.abs(held)) and meshes[0].get_cmap().name == DIVERGING
    assert fig.axes[-1].get_ylabel() == "K" and meshes[0].colorbar.extend == "neither"
    plt.close(fig)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    meshes = section(ax, sec, diverging=True, clip=90, colorbar=True)
    assert np.isclose(meshes[0].get_clim()[1], np.percentile(np.abs(held), 90))
    assert meshes[0].get_clim()[1] < 0.1 * np.max(np.abs(held))     # the spike cut
    assert fig.axes[-1].get_ylabel() == "K" and meshes[0].colorbar.extend == "max"
    plt.close(fig)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    meshes = section(ax, sec, normalised=True, diverging=True, colorbar=True)
    assert np.allclose(np.asarray(meshes[1].get_array()).reshape(5, 41),
                       sec.layers[1].values / np.max(np.abs(held)))
    assert meshes[0].get_clim() == (-1.0, 1.0)
    assert fig.axes[-1].get_ylabel() == "K, normalised"
    plt.close(fig)


def test_the_scale_clips_at_a_percentile_and_the_bar_says_where(layered):
    plain = slices.sphere(layered, "rho", radius=0.8, grid=gauss_legendre(4))
    values = np.concatenate([np.linspace(0.0, 1.0, 101), [1e3]])
    cmap, lo, hi = colour_scale([values], plain.shown, clip=95)
    assert cmap == SEQUENTIAL
    assert (lo, hi) == (np.percentile(values, 5), np.percentile(values, 95))
    cmap, lo, hi = colour_scale([values - 0.5], plain.shown, diverging=True, clip=95)
    assert cmap == DIVERGING and hi == -lo == np.percentile(np.abs(values - 0.5), 95)
    # most of a kernel is zero: a percentile of zero gives way to the largest
    spike = np.zeros(1000)
    spike[3] = -2.0
    assert colour_scale([spike], plain.shown, diverging=True, clip=99)[1:] == (-2.0,
                                                                               2.0)
    with pytest.raises(ValueError, match="percentile"):
        colour_scale([values], plain.shown, clip=40)
    assert colour_extend([values], 0.0, 2e3) == "neither"
    assert colour_extend([values], 0.1, 2e3) == "min"
    assert colour_extend([values], 0.0, 0.9) == "max"
    assert colour_extend([values, [np.nan]], 0.1, 0.9) == "both"


# -- spectra and correlations drawn --------------------------------------------

def _harmonic_model(layered, coeffs, name):
    from planetmodel.harmonics import synthesise
    c = np.asarray(coeffs, dtype=float)
    return layered.with_field(1, "vs", AnalyticField(
        (0.5, 1.0), lambda r, t, p: r * synthesise(c, t, p), name="vs"),
        replace=True).named(name)


def test_a_spectrum_is_a_line_of_power_by_degree(layered):
    from planetmodel.plotting import spectrum
    c = np.zeros((2, 4, 4))
    c[0, 0, 0], c[0, 2, 1], c[1, 3, 3] = 1.0, 0.5, 0.25
    sp = slices.spectrum(_harmonic_model(layered, c, "h"), "vs", radius=0.8, lmax=5)
    fig, ax = plt.subplots()
    line = spectrum(ax, sp, first=1, label="h")
    x, y = line.get_data()
    assert list(x) == [1, 2, 3, 4, 5] and np.allclose(y, sp.power[1:])
    assert ax.get_yscale() == "log" and line.get_label() == "h"
    assert np.allclose(spectrum(ax, sp, amplitude=True).get_data()[1], sp.amplitude)
    with pytest.raises(ValueError, match="first"):
        spectrum(ax, sp, first=9)
    plt.close(fig)


def test_a_correlation_is_an_image_by_degree_and_radius(layered):
    from planetmodel.plotting import correlation, correlation_total
    a, b = np.zeros((2, 4, 4)), np.zeros((2, 4, 4))
    a[0, 2, 1], a[0, 3, 0], a[0, 1, 1] = 1.0, 0.5, 0.3
    b[1, 2, 1], b[0, 3, 0], b[0, 1, 1] = 0.7, -2.0, 0.3
    cr = slices.correlation(_harmonic_model(layered, a, "a"), "vs",
                            _harmonic_model(layered, b, "b"), lmax=3,
                            radii=[0.9, 0.6, 0.75])
    fig, (ax, bx) = plt.subplots(1, 2)
    mesh = correlation(ax, cr, scale=10.0, colorbar=True)
    assert mesh.get_clim() == (-1.0, 1.0) and mesh.get_cmap().name == DIVERGING
    shown = np.ma.asarray(mesh.get_array()).reshape(3, 3)
    order = np.argsort(cr.radii)
    assert np.allclose(shown, cr.by_degree[order][:, 1:])
    xy = mesh.get_coordinates()
    assert np.allclose(xy[0, :, 0], [0.5, 1.5, 2.5, 3.5])
    assert np.allclose(np.sort(xy[:, 0, 1])[1:-1], 10.0 * np.array([0.675, 0.825]))
    line = correlation_total(bx, cr, scale=10.0)
    assert np.allclose(line.get_data()[0], cr.total[order])
    assert np.allclose(line.get_data()[1], 10.0 * cr.radii[order])
    assert fig.axes[-1].get_ylabel() == cr.label
    plt.close(fig)


# -- cutaways -----------------------------------------------------------------------

from planetmodel.plotting import cutaway  # noqa: E402


def test_a_cutaway_is_one_image_of_the_faces_with_the_edge_antialiased(layered):
    from matplotlib.colors import Normalize
    c = slices.cutaway(layered, "vs", inner=0.3, n=50, longitudes=(0.2, 1.6))
    fig, ax = plt.subplots()
    image = cutaway(ax, c, scale=2.0, colorbar=True)
    rgba = image.get_array()
    assert rgba.shape == (50, 50, 4)
    assert np.allclose(rgba[..., 3], c.coverage)
    assert np.allclose(image.get_extent(), (-2.0, 2.0, -2.0, 2.0))
    # every face shows the field on one colour scale
    seen = c.face > 0
    cmap, lo, hi = colour_scale([c.values[seen]], c.shown)
    norm = Normalize(lo, hi)
    assert np.allclose(rgba[seen][:, :3],
                       plt.get_cmap(cmap)(norm(c.values[seen]))[:, :3])
    assert ax.figure.axes[-1].get_ylabel() == c.label
    assert ax.get_aspect() == 1.0 and not ax.axison
    # the interfaces are faint arcs on the faces, drawn once
    lines = ax.get_lines()
    assert len(lines) == 1 and lines[0].get_color() == BOUNDARY_STYLE["color"]
    x, y = lines[0].get_data()
    assert np.any(np.isnan(x)) and np.isfinite(x).sum() > 50
    assert np.all(np.hypot(x, y)[np.isfinite(x)] <= 2.0 * 0.5 * (1 + 1e-12))
    plt.close(fig)

    # a plain surface and a plain inner sphere, shaded, never the field's colours
    fig, ax = plt.subplots()
    image = cutaway(ax, c, surface="0.9", inner="red", interfaces=False, cmap="gray",
                    vmin=0.0, vmax=10.0)
    rgba = image.get_array()
    top = rgba[c.mask("surface")]
    assert np.all(top[:, :3] <= 0.9) and np.all(top[:, :3] > 0.4)
    assert np.all(top[:, 0] == top[:, 1]) and np.any(top[:, 0] < 0.9)
    core = rgba[c.mask("inner")]
    assert np.all(core[:, 1:3] == 0.0) and np.all(core[:, 0] > 0.0)
    faces = c.mask("first") | c.mask("second")
    assert np.allclose(rgba[faces, 0], c.values[faces] / 10.0, atol=1e-2)
    assert not ax.get_lines()
    plt.close(fig)

    # the scale is of the field's pixels alone, diverging and clipped as asked
    fig, ax = plt.subplots()
    d = slices.cutaway(layered, "vs", inner=0.5, n=30, relative_to="mean",
                       mean_grid=gauss_legendre(16))
    image = cutaway(ax, d, surface="w", clip=90.0, normalised=True)
    assert image.get_array().shape == (30, 30, 4)
    plt.close(fig)


def test_discs_are_drawn_as_a_cutaway_is(layered):
    c = slices.discs(layered, "vs", poles=[(0.0, 0.0), (0.5 * np.pi, 0.0)], outer=0.3,
                     n=40)
    fig, ax = plt.subplots()
    image = cutaway(ax, c, surface="0.9", colorbar=True)
    rgba = image.get_array()
    assert np.allclose(rgba[..., 3], c.coverage)
    discs = c.mask("disc 1") | c.mask("disc 2")
    assert np.all(rgba[discs, 3] > 0.0) and np.any(rgba[discs, :3] != rgba[discs, :1])
    top = rgba[c.mask("surface")]
    assert np.all(top[:, 0] == top[:, 1]) and np.all(top[:, 0] <= 0.9)
    lines = ax.get_lines()
    assert len(lines) == 1                       # the interface circles on the discs
    x, y = lines[0].get_data()
    assert np.isfinite(x).sum() > 100 and np.all(np.hypot(x, y)[np.isfinite(x)] <= 0.5)
    plt.close(fig)


def test_a_cutaway_is_refused_on_the_wrong_axes_or_with_a_surface_it_cannot_draw(
        layered):
    c = slices.cutaway(layered, "vs", inner=0.5, n=20)
    fig = plt.figure()
    with pytest.raises(ValueError, match="plain axes"):
        cutaway(fig.add_subplot(projection="polar"), c)
    ax = fig.add_subplot()
    with pytest.raises(ValueError, match="surface is"):
        cutaway(ax, c, surface="nonesuch")
    with pytest.raises(ValueError, match="RGB"):
        cutaway(ax, c, surface=np.zeros((4, 8, 2)))
    with pytest.raises(ValueError, match="surface is"):
        cutaway(ax, c, surface=3.5)
    # an image of the surface, rows from the north pole, columns from 180 west
    image = np.zeros((2, 2, 3))
    image[0] = [1.0, 0.0, 0.0]                     # the northern hemisphere red
    image[1] = [0.0, 0.0, 1.0]                     # the southern blue
    im = cutaway(ax, c, surface=image, interfaces=False)
    rgba = im.get_array()
    top = c.mask("surface")
    north = top & (c.reference[..., 2] > 0.0)
    south = top & (c.reference[..., 2] < 0.0)
    assert np.all(rgba[north, 0] > 0.0) and np.all(rgba[north, 2] == 0.0)
    assert np.all(rgba[south, 2] > 0.0) and np.all(rgba[south, 0] == 0.0)
    eight = (255 * image).astype(np.uint8)
    assert np.allclose(cutaway(ax, c, surface=eight, interfaces=False).get_array(),
                       rgba)
    plt.close(fig)
