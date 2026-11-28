"""Flat slices drawn: sections flat and in the body's shape, interfaces over
the colours, profiles with their spread, spectra by horizontal
wavenumber, correlations and blocks, depth downward throughout; a
section of a model composes with the section of its mesh."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import PolyCollection, QuadMesh  # noqa: E402

from planetmodel import LayeredIsotropicElastic, Skeleton  # noqa: E402
from planetmodel import cartesian as flat  # noqa: E402
from planetmodel import slices as sphere_slices  # noqa: E402
from planetmodel.cartesian import slices  # noqa: E402
from planetmodel.plotting import (BOUNDARY_STYLE, block, correlation,  # noqa: E402
                                  correlation_total, plane_cut, profile, section,
                                  spectrum, vertical_section)

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
GRID = flat.Grid(BOX, (8, 6))


def lateral(d, x, y):
    return 4.0 + 0.3 * d * np.cos(np.pi * x) + 0.2 * d * np.sin(np.pi * y)


@pytest.fixture
def raised():
    """Two layers under a ridge on the surface and a root on the Moho."""
    m = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], BOX, rho=[2.0, 3.0],
                                     vp=[5.0, 7.0], vs=[3.0, 4.0])
    m = m.with_field(1, "vs", flat.AnalyticField((0.4, 1.0), lateral, name="vs"),
                     replace=True)
    g = flat.Grid(BOX, (16, 12))
    x, y = g.nodes
    h = np.broadcast_to(0.05 * (1.0 + np.cos(np.pi * x)) + 0.0 * y, g.shape)
    top = flat.FourierFunction(g, g.transform.analyse_grid(h))
    root = flat.FourierFunction(g, -2.0 * top.coefficients)
    return m.stretched(flat.layer_linear(m.skeleton, [top, root, None])), top


def test_a_section_is_drawn_in_its_own_coordinates_flat_or_as_the_body_is(raised):
    m, top = raised
    sec = slices.section(m, "vs", (0.0, 0.2), end=(2.0, 0.2), n=5, m=9,
                         relative_to="mean", mean_grid=GRID)
    fig, (a, b) = plt.subplots(2, 1)
    flat_meshes = vertical_section(a, sec, scale=10.0, colorbar=True)
    assert a.yaxis_inverted() and len(flat_meshes) == 2
    xy = flat_meshes[1].get_coordinates()
    assert np.allclose(xy[..., 0], 10.0 * sec.distance[None, :])
    assert np.allclose(xy[..., 1], 10.0 * sec.layers[1].radius)
    assert flat_meshes[0].get_cmap().name == "RdBu"
    assert fig.axes[-1].get_ylabel() == sec.label
    shaped = vertical_section(b, sec, physical=True)
    xy = shaped[0].get_coordinates()
    assert np.allclose(xy[0, :, 1], -top(sec.path[:, 0], 0.2), rtol=0, atol=1e-12)
    # the interfaces over the colours
    lines = [ln for ln in a.get_lines() if ln.get_color() == BOUNDARY_STYLE["color"]]
    assert len(lines) == 4
    assert all(ln.get_zorder() > mesh.get_zorder() for ln in lines
               for mesh in flat_meshes)
    with pytest.raises(ValueError, match="physical points already"):
        vertical_section(a, slices.section(m, "rho", (0.0, 0.2), end=(2.0, 0.2), n=3,
                                           m=4, at="physical"), physical=True)
    polar = fig.add_subplot(2, 2, 4, projection="polar")
    with pytest.raises(ValueError, match="plain axes"):
        vertical_section(polar, sec)
    with pytest.raises(ValueError, match="drawn by vertical_section"):
        section(polar, sec)
    plt.close(fig)


def test_a_spherical_section_draws_its_interfaces_over_the_colours():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    sec = sphere_slices.section(m, "rho", (0.6, 0.2), azimuth=1.0, length=1.2, n=4,
                                m=11)
    fig, ax = plt.subplots(subplot_kw={"projection": "polar"})
    meshes = section(ax, sec)
    arcs = [ln for ln in ax.get_lines() if ln.get_color() == BOUNDARY_STYLE["color"]]
    assert arcs and all(ln.get_zorder() > mesh.get_zorder() for ln in arcs
                        for mesh in meshes)
    # and over the axes' grid, beneath any curve
    assert BOUNDARY_STYLE["zorder"] > 1.5 and BOUNDARY_STYLE["zorder"] < 2.0
    plt.close(fig)


def test_a_flat_profile_runs_depth_downward_with_its_spread(raised):
    m, _ = raised
    prof = slices.profile(m, "rho", 0.5, 0.0, n=5, at="physical")
    fig, (a, b) = plt.subplots(1, 2)
    lines = profile(a, prof)
    assert a.yaxis_inverted()
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert np.allclose(segments[0].get_data()[1], prof.layers[0].radius)
    mean = slices.mean_profile(m, "vs", mean_grid=GRID, n=5, spread="range")
    profile(b, mean)
    bands = [c for c in b.collections if isinstance(c, PolyCollection)]
    assert len(bands) == 2 and b.yaxis_inverted()
    plt.close(fig)


def test_a_flat_spectrum_is_drawn_by_wavenumber_with_the_wavelength_on_top():
    g = flat.Grid(BOX, (8, 8))
    c = np.random.default_rng(1).standard_normal(g.transform.size)
    m = flat.LayeredIsotropicElastic([0.0, 1.0], BOX, rho=[1.0], vp=[2.0], vs=[1.0])
    m = m.with_field(0, "q", flat.FourierField((0.0, 1.0), g, c, name="q"))
    sp = slices.spectrum(m, "q", depth=0.5, grid=g)
    fig, ax = plt.subplots()
    line = spectrum(ax, sp, scale=0.5)
    x, y = line.get_data()
    assert np.allclose(x, sp.wavenumbers[1:] / 0.5) and np.allclose(y, sp.power[1:])
    assert ax.get_xscale() == "log" and ax.get_yscale() == "log"
    spectrum(ax, sp, amplitude=True, first=2, wavelengths="wavelength [km]")
    tops = [child for child in ax.child_axes]
    assert len(tops) == 1 and tops[0].get_xlabel() == "wavelength [km]"
    cor = slices.correlation(m, "q", m, depths=[0.2, 0.5, 0.8], grid=g)
    fig2, (a, b) = plt.subplots(1, 2)
    mesh = correlation(a, cor)
    assert a.yaxis_inverted() and mesh.get_array().shape[0] == 3
    correlation_total(b, cor)
    assert b.yaxis_inverted()
    plt.close(fig)
    plt.close(fig2)


def test_a_block_is_drawn_on_three_dimensional_axes(raised):
    m, top = raised
    b = slices.block(m, "vs", n=4, m=6, relative_to="mean", mean_grid=GRID)
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surfaces = block(ax, b, physical=True, colorbar=True)
    assert len(surfaces) == len(b.layers) and not ax.computed_zorder
    assert ax.zaxis.get_major_formatter()(-0.25, 0) == "0.25"
    assert fig.axes[-1].get_ylabel() == b.label
    plain = fig.add_subplot(2, 2, 1)
    with pytest.raises(ValueError, match="three-dimensional axes"):
        block(plain, b)
    plt.close(fig)


@pytest.mark.gmsh
def test_a_section_of_a_model_composes_with_the_section_of_its_mesh(tmp_path):
    """The mesh's interfaces and a fault cut by the section's own plane lie
    where the section draws them: the Moho at its depth, flat and as the
    body is, the fault's trace through the right points."""
    from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
    from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                    build_layered_mesh, manifest, read_mesh)
    from planetmodel.plotting import mesh_section
    box = flat.Box([(0.0, 1.0), (-0.5, 0.5)])
    geometry = flat.Geometry(Skeleton([0.0, 0.4, 1.0]), box,
                             interface_names=["surface", "moho"])
    fault = Plane.at_depth(CartesianFrame(), 0.5, 0.0, 0.6, strike=0.0, dip=50.0,
                           length=0.4, width=0.25, slip=Slip(np.zeros((1, 1, 3))))
    res = build_layered_mesh(MeshSpec(geometry, UniformInterfaces(0.15, 0.3, 0.3),
                                      faults=[EmbeddedFault(Fault([fault]),
                                                            name="f")]),
                             tmp_path / "box")
    mesh, card = read_mesh(res.mesh_path), manifest.read(res.manifest_path)
    model = flat.LayeredIsotropicElastic([0.0, 0.4, 1.0], box, rho=[2.0, 3.0],
                                         vp=[5.0, 7.0], vs=[3.0, 4.0])
    sec = slices.section(model, "rho", (0.0, 0.0), end=(1.0, 0.0), n=4, m=11)
    moho = mesh.faces_with(card.interfaces[1]["attribute"])[:, :3]
    cut = plane_cut(mesh.points, moho, sec.path[0], sec.plane)
    assert cut.pieces.size and np.allclose(cut.pieces[..., 1], 0.4)
    faces = mesh.faces_with(card.fault_record("f")["attribute"])[:, :3]
    trace = plane_cut(mesh.points, faces, sec.path[0], sec.plane)
    # the fault's trace in the section: through its centroid, dipping 50 degrees
    s, d = trace.pieces[..., 0].ravel(), trace.pieces[..., 1].ravel()
    assert np.allclose(d - 0.6, (s - 0.5) * np.tan(np.radians(50.0)), atol=1e-9)
    fig, ax = plt.subplots()
    vertical_section(ax, sec)
    mesh_section(ax, cut, colours="k")
    mesh_section(ax, trace, colours="b")
    assert ax.yaxis_inverted()
    assert any(isinstance(c, QuadMesh) for c in ax.collections)
    plt.close(fig)
