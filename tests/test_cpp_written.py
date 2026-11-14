"""The model files the C++ writer wrote, read by the Python: each
validates, reads as a model with its fields on interfaces, and holds the
values the C++ reader reads from it; a fixture the C++ read and wrote
back reads as the fixture does; the fault files the C++ wrote read by
the Python's strict reader, a fixture written back to the same fault
and metadata, evaluating number for number as the fixture does, and a
fault built in C++ to the C++'s own evaluations of it; and the kernel the example
`write_kernel` wrote reads as a kernel (tutorial 15, given its path,
compares it with the Python's own at every node, in the C++ job of CI).

The C++ tests leave the files in their build tree; PLANETMODEL_CPP_WRITTEN
names that directory (`build/cpp/tests/written` for the build of
CLAUDE.md), and without it these tests are skipped.
"""
import os
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.modelfile import (read_model, read_surfaces,  # noqa: E402
                                   validate_structure)

WRITTEN = os.environ.get("PLANETMODEL_CPP_WRITTEN")
FIXTURES = Path(__file__).resolve().parents[1] / "cpp" / "tests" / "fixtures"
needs_written = pytest.mark.skipif(
    WRITTEN is None,
    reason="PLANETMODEL_CPP_WRITTEN names no directory of C++-written files")


def _written(pattern: str) -> list[Path]:
    return [] if WRITTEN is None else sorted(Path(WRITTEN).glob(pattern))


def _check_expect(model, expect: Path, *, rtol: float, surfaces=None) -> int:
    """The lines of an expectation file met by a model and the functions
    on its interfaces: values at points in either frame, the mapping's
    displacement, and the fields on interfaces."""
    checked = 0
    for line in expect.read_text().splitlines():
        words = line.split()
        if words[0] == "value":
            layer, name, frame = int(words[1]), words[2], words[3]
            r, t, p = (float(x) for x in words[4:7])
            want = np.array([float(x) for x in words[7:]])
            got = np.asarray(model.layer(layer)[name].evaluate(
                r, t, p, frame="spherical" if frame == "s" else "cartesian"),
                dtype=float).ravel()
            scale = max(1.0, float(np.max(np.abs(want))))
            assert np.allclose(got, want, rtol=0, atol=rtol * scale), line
            checked += 1
        elif words[0] == "displacement":
            X = np.array([float(x) for x in words[1:4]])
            want = np.array([float(x) for x in words[4:7]])
            got = np.asarray(model.geometry.mapping.displacement(X[None, :]),
                             dtype=float)[0]
            outer = float(model.skeleton.boundaries[-1])
            assert np.allclose(got, want, rtol=0, atol=1e-12 * outer), line
            checked += 1
        elif words[0] in ("surface", "surface_gradient", "surface_absent"):
            k, name = int(words[1]), words[2]
            face = model.geometry.interface(k)
            on = surfaces.get(name, {})
            key = k if face.name is None else face.name
            if words[0] == "surface_absent":
                assert key not in on, line
                continue
            t, p = float(words[3]), float(words[4])
            want = np.array([float(x) for x in words[5:]])
            got = (np.atleast_1d(on[key](t, p)) if words[0] == "surface"
                   else np.array([float(d) for d in on[key].angular_gradient(t, p)]))
            scale = max(1.0, float(np.max(np.abs(want))))
            tol = rtol if words[0] == "surface" else 1e-9
            assert np.allclose(got, want, rtol=0, atol=tol * scale), line
            checked += 1
    return checked


@needs_written
def test_the_cpp_wrote_its_files():
    names = {p.name for p in _written("*.nc")}
    assert {"written_si.nc", "written_custom.nc", "written_tensors.nc",
            "written_surfaces.nc", "kernel.nc", "roundtrip_model_prem.nc",
            "roundtrip_model_harmonic.nc", "roundtrip_model_surfaces.nc",
            "surface_written.nc"} <= names


@needs_written
@pytest.mark.parametrize("path", _written("written_*.nc"), ids=lambda p: p.name)
def test_a_file_the_cpp_wrote_reads_as_the_cpp_reads_it(path):
    validate_structure(path)
    model = read_model(path)
    assert _check_expect(model, path.with_suffix(".expect"), rtol=1e-10,
                         surfaces=read_surfaces(path)) > 10


@needs_written
@pytest.mark.parametrize("path", _written("roundtrip_*.nc"), ids=lambda p: p.name)
def test_a_fixture_written_back_reads_as_the_fixture(path):
    validate_structure(path)
    expect = FIXTURES / (path.stem.removeprefix("roundtrip_") + ".expect")
    if not expect.exists():
        return                              # the harmonic copy: structure alone
    model = read_model(path)
    assert _check_expect(model, expect, rtol=1e-11, surfaces=read_surfaces(path)) > 10


@needs_written
def test_a_file_the_cpp_wrote_cites_what_it_was_given():
    assert read_model(Path(WRITTEN) / "written_si.nc").citation == (
        "A. Author (2026).  A written model.", "B. Author (2026).  Its source.")
    prem = read_model(Path(WRITTEN) / "roundtrip_model_prem.nc")
    assert prem.citation == read_model(FIXTURES / "model_prem.nc").citation != ()


@needs_written
def test_the_surface_file_the_cpp_wrote_holds_what_it_was_given():
    from planetmodel import Scales, gauss_legendre
    from planetmodel.surfacefile import SurfaceVariable, read_surface_file
    from planetmodel.testing import check_surface_file
    from planetmodel.units import LENGTH

    path = Path(WRITTEN) / "surface_written.nc"
    grid = read_surface_file(path).grid
    k = np.arange(3)[:, None, None].astype(float)
    i = np.arange(grid.ntheta)[None, :, None].astype(float)
    j = np.arange(grid.nphi)[None, None, :].astype(float)
    topography = (i + 0.01 * j)[0]
    topography.flat[1] = np.nan
    fields = {
        "topography": SurfaceVariable(topography, dimensions=LENGTH,
                                      meaning="present-day topography",
                                      poles=[1.5, -0.5]),
        "ice_thickness": SurfaceVariable(100.0 * k + i + 0.01 * j,
                                         support="node",
                                         time_interpolation="linear",
                                         dimensions=LENGTH),
        "rate": SurfaceVariable((10.0 * k + i - 0.1 * j)[:2],
                                support="interval"),
        "instantaneous": SurfaceVariable((1.0 * k + 2.0 * i + 0.2 * j)[:2],
                                         support="instant",
                                         lateral_interpolation="cubic"),
    }
    check_surface_file(path, grid, fields=fields, time=[0.0, 1.0, 2.5],
                       instant=[1.5, 2.5], radius=1.25,
                       scales=Scales(length=2.0, mass=3.0, time=5.0),
                       meta={"epoch": "present", "lmax": 4},
                       citation=("one", "two"))
    want = gauss_legendre(4)
    assert np.allclose(grid.colatitudes, want.colatitudes, rtol=0, atol=1e-12)
    assert np.allclose(grid.weights, want.weights, rtol=0, atol=1e-12)


@needs_written
def test_the_kernel_example_reads_as_a_kernel():
    from planetmodel import Dimensions, slices
    path = Path(WRITTEN) / "kernel.nc"
    validate_structure(path)
    model = read_model(path)
    assert model.name == "synthetic kernel"
    assert model.layers_with("K_vs") == (1, 2) and model.layers_with("rho") == (0, 1, 2)
    assert model.spec("K_vs").dimensions == Dimensions(length=-4, time=1)
    shape = read_surfaces(path)["K_d670"]
    assert list(shape) == ["d670"]
    assert model.spec("K_d670").dimensions == Dimensions(length=-3)
    # both signs, and the peak beside the receiver at the surface
    near = slices.sphere(model, "K_vs", depth=100e3)
    assert near.values.min() < 0.0 < near.values.max()
    i, j = np.unravel_index(np.argmax(near.values), near.values.shape)
    assert np.isclose(near.grid.colatitudes[i], np.pi / 3, atol=0.02)
    assert np.isclose(near.grid.longitudes[j], np.pi / 6, atol=0.02)
    assert shape["d670"](np.pi / 3, np.pi / 6) > 0.0


@needs_written
@pytest.mark.parametrize("name", ["fault_static", "fault_afterslip",
                                  "fault_cartesian"])
def test_a_fault_file_the_cpp_wrote_back_is_the_python_s(name):
    """The fixture the Python wrote, read and written back by the C++:
    the strict reader takes it, its metadata are the fixture's, and its
    fault evaluates as the fixture's, slip and quadrature number for
    number."""
    from planetmodel.fault import BASES, SQUARE
    from planetmodel.faultfile import read_fault_file
    from planetmodel.testing import check_fault_file

    path = Path(WRITTEN) / f"faultfile_back_{name}.nc"
    got, want = read_fault_file(path), read_fault_file(FIXTURES / f"{name}.nc")
    for key in ("schema", "names", "source", "created", "title", "citation"):
        assert getattr(got, key) == getattr(want, key), key
    assert dict(got.meta) == dict(want.meta)
    check_fault_file(path, want.fault, names=want.names, meta=want.meta,
                     citation=want.citation)
    a, b = got.fault, want.fault
    rng = np.random.default_rng(3)
    u, v = rng.uniform(*SQUARE, (2, 40))
    dates = [None] if b.time is None else list(b.time) + (
        [] if b.time_interpolation == "none" else [float(np.mean(b.time))])
    for date in dates:
        for k in range(len(b.patches)):
            for basis in BASES:
                assert np.array_equal(a.slip_vector(k, u, v, date=date, basis=basis),
                                      b.slip_vector(k, u, v, date=date, basis=basis))
        qa, qb = (f.quadrature(3, 2, points=2, date=date) for f in (a, b))
        for key in ("points", "positions", "normals", "weights", "slip", "patch"):
            assert np.array_equal(getattr(qa, key), getattr(qb, key)), key
    sa, sb = a.slice_quadrature(2, 2), b.slice_quadrature(2, 2)
    assert np.array_equal(sa.slip, sb.slip) and np.array_equal(sa.weights, sb.weights)


@needs_written
@pytest.mark.parametrize("name", ["faultfile_node", "faultfile_static"])
def test_a_fault_built_in_cpp_reads_as_the_cpp_evaluates_it(name):
    """A fault the C++ built and wrote, read by the Python's strict
    reader: its metadata as the C++ gave them, and its slip and quadrature
    the C++'s own evaluations, to rounding."""
    from planetmodel.faultfile import read_fault_file

    path = Path(WRITTEN) / f"{name}.nc"
    data = read_fault_file(path)
    fault = data.fault
    assert data.source == "planetmodel C++ tests"
    if name == "faultfile_node":
        assert data.names == ("main", None, "splay")
        assert data.title == "a fault written by the C++"
        assert dict(data.meta) == {"event": "test", "magnitude": 7.5, "count": 3}
        assert data.citation == ("A. Author (2026).  A fault.",
                                 "B. Author (2026).  Its slip.")
        assert (fault.support, fault.time_interpolation) == ("node", "linear")
        assert [p.kind for p in fault.patches] == ["plane", "structured", "structured"]
        assert fault.scales.length == 1e3 and fault.scales.time == 3.15576e7
    else:
        assert data.names == (None, None) and data.title is None
        assert data.meta == {} and data.citation == ()
        assert fault.support == "static" and fault.frame.kind == "cartesian"
    scale = max(1.0, max(float(np.max(np.abs(p.slip.values))) for p in fault.patches))
    checked = 0
    quadrature = None
    for line in path.with_suffix(".expect").read_text().splitlines():
        words = line.split()
        if words[0] == "slip":
            k, u, v = int(words[1]), float(words[2]), float(words[3])
            date = None if words[4] == "-" else float(words[4])
            want = np.array([float(x) for x in words[6:]])
            got = fault.slip_vector(k, u, v, date=date, basis=words[5])
            assert np.allclose(got, want, rtol=0, atol=1e-12 * scale), line
        elif words[0] == "quadrature":
            date = None if words[1] == "-" else float(words[1])
            quadrature = fault.quadrature(2, 3, points=2, date=date)
            continue
        elif words[0] == "qpoint":
            i, patch = int(words[1]), int(words[2])
            want = np.array([float(x) for x in words[3:]])
            q = quadrature
            assert q.patch[i] == patch, line
            assert np.allclose([q.u[i], q.v[i]], want[:2], rtol=0, atol=1e-15), line
            assert np.isclose(q.weights[i], want[2], rtol=1e-12, atol=0), line
            size = float(np.max(np.abs(want[3:6])))
            assert np.allclose(q.points[i], want[3:6], rtol=0, atol=1e-14 * size), line
            assert np.allclose(q.normals[i], want[6:9], rtol=0, atol=1e-12), line
            assert np.allclose(q.slip[i], want[9:12], rtol=0, atol=1e-12 * scale), line
        checked += 1
    assert checked > 50
