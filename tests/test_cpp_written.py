"""The model files the C++ writer wrote, read by the Python: each
validates, reads as a model with its fields on interfaces, and holds the
values the C++ reader reads from it; a fixture the C++ read and wrote
back reads as the fixture does; and the kernel the example
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
