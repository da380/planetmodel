"""Sampling: a model evaluated on radial times angular nodes.

The angular grids are the node sets they claim, Gauss-Legendre exact
to its band and equiangular at the midpoints; the radial nodes are the
mesh's per-element nodes with both one-sided values kept at every
interface; a direction-free field is stored on `(node,)` and says so,
a direction-dependent one equals its formula at every product point;
a layer lacking a field refuses or carries NaN; and a radial stretch
samples to `(h, 0, 0)` in the spherical frame.  `check_sample` runs on
every sample built and fails on tampered copies.  Meshes are a few
elements across, PREM included.
"""
import dataclasses
from math import factorial

import numpy as np
import pytest
from scipy.fft import next_fast_len
from scipy.special import lpmv

from planetmodel import RadialMesh, Skeleton
from planetmodel.catalogue import homogeneous, layered, prem
from planetmodel.character import DENSITY, VECTOR
from planetmodel.displacement import flattening
from planetmodel.fields import AnalyticField
from planetmodel.frames import spherical_frame
from planetmodel.sampling import (AngularGrid, Sample, equiangular, gauss_legendre,
                                  sample)
from planetmodel.testing import check_sample
from planetmodel.units import DENSITY as KG_M3
from planetmodel.units import Scales

A = 1.0
BOUNDS = [0.0, 0.3 * A, 0.7 * A, A]
RHO = [13.0, 5.0, 3.0]


def three_layers():
    return layered(BOUNDS, rho=RHO, vp=[11.0, 8.0, 6.0], vs=[3.5, 0.0, 3.0],
                   layer_names=["core", "shell", "crust"])


def scalar_fn(r, t, p):
    return 5.0 + 1.0 * (r / A) ** 2 * (1.0 + 0.1 * np.sin(t) * np.cos(p))


def vector_fn(r, t, p):
    r, t, p = np.broadcast_arrays(r, t, p)
    return np.stack([r / A + 0.0 * t, np.sin(t) * np.cos(p),
                     0.5 * np.cos(t) + 0.0 * p], axis=-1)


def with_angular_fields(model):
    """Every layer given a direction-dependent scalar and a vector."""
    for layer in model.layers:
        iv = layer.interval
        model = model.with_field(layer.index, "rho3d",
                                 AnalyticField(iv, scalar_fn, character=DENSITY))
        model = model.with_field(layer.index, "wind",
                                 AnalyticField(iv, vector_fn, character=VECTOR))
    return model


@pytest.fixture(scope="module")
def grid():
    return gauss_legendre(4)


@pytest.fixture(scope="module")
def plain(grid):
    return sample(three_layers(), grid, drmax=0.2 * A)


@pytest.fixture(scope="module")
def angular(grid):
    m = with_angular_fields(three_layers())
    return m, sample(m, grid, drmax=0.25 * A)


@pytest.fixture(scope="module")
def coarse_prem(grid):
    m = prem()
    return m, sample(m, grid, ngll=3, drmax=1e6)


@pytest.fixture(scope="module")
def stretched(grid):
    m = three_layers().stretched(flattening(0.01, rmax=A))
    return m, sample(m, grid, drmax=0.25 * A)


# ------------------------------------------------------------ AngularGrid

def test_gauss_legendre_nodes_and_band():
    lmax = 6
    g = gauss_legendre(lmax)
    assert g.kind == "gauss_legendre" and g.lmax == lmax
    assert g.ntheta == lmax + 1
    assert np.all(np.diff(g.colatitudes) > 0)
    assert 0 < g.colatitudes[0] and g.colatitudes[-1] < np.pi
    assert g.nphi >= 2 * lmax + 1 and g.nphi == next_fast_len(2 * lmax + 1)
    assert np.isclose(g.weights.sum(), 2.0, rtol=1e-14)
    assert g.longitudes[0] == 0.0 and np.allclose(
        np.diff(g.longitudes), 2 * np.pi / g.nphi)


def real_harmonic(l, m, theta, phi):
    """The orthonormal real spherical harmonic of degree l and order m >= 0."""
    norm = np.sqrt((2 * l + 1) / (4 * np.pi) * factorial(l - m) / factorial(l + m))
    P = lpmv(m, l, np.cos(theta))
    return norm * P * (np.sqrt(2.0) * np.cos(m * phi) if m else 1.0 + 0.0 * phi)


def test_gauss_legendre_integrates_harmonics_exactly_to_its_band():
    lmax = 5
    g = gauss_legendre(lmax)
    t, p = g.colatitudes[:, None], g.longitudes[None, :]
    w = g.weights[:, None] * (2 * np.pi / g.nphi)
    pairs = [(l, m) for l in range(lmax + 1) for m in range(l + 1)]
    for l1, m1 in pairs:
        for l2, m2 in pairs:
            got = np.sum(w * real_harmonic(l1, m1, t, p) * real_harmonic(l2, m2, t, p))
            want = 1.0 if (l1, m1) == (l2, m2) else 0.0
            assert abs(got - want) < 1e-13, (l1, m1, l2, m2)


def test_gauss_legendre_refuses_too_few_longitudes():
    with pytest.raises(ValueError, match="2 lmax \\+ 1"):
        gauss_legendre(4, nphi=8)
    assert gauss_legendre(4, nphi=9).nphi == 9
    with pytest.raises(ValueError, match="non-negative"):
        gauss_legendre(-1)


def test_equiangular_grid():
    g = equiangular(16, 32)
    assert g.kind == "equiangular" and g.lmax is None and g.weights is None
    assert g.ntheta == 16 and g.nphi == 32
    assert 0 < g.colatitudes[0] and g.colatitudes[-1] < np.pi
    assert np.allclose(g.colatitudes, np.pi * (np.arange(16) + 0.5) / 16)
    assert np.allclose(g.longitudes, 2 * np.pi * np.arange(32) / 32)
    with pytest.raises(ValueError, match="positive"):
        equiangular(0, 4)


@pytest.mark.parametrize("bad", [
    dict(colatitudes=[2.0, 1.0], longitudes=[0.0]),
    dict(colatitudes=[0.0, 1.0], longitudes=[0.0]),
    dict(colatitudes=[1.0, np.pi], longitudes=[0.0]),
    dict(colatitudes=[1.0], longitudes=[0.0, 2 * np.pi]),
    dict(colatitudes=[1.0], longitudes=[-0.1]),
    dict(colatitudes=[1.0], longitudes=[0.5, 0.2]),
    dict(colatitudes=[1.0], longitudes=[0.0], kind="lebedev"),
    dict(colatitudes=[1.0, 2.0], longitudes=[0.0], weights=[1.0]),
    dict(colatitudes=[1.0], longitudes=[0.0], lmax=-1),
    dict(colatitudes=[], longitudes=[0.0]),
])
def test_angular_grid_validation(bad):
    with pytest.raises(ValueError):
        AngularGrid(**bad)


def test_angular_grid_arrays_are_read_only():
    g = equiangular(4, 8)
    with pytest.raises(ValueError):
        g.colatitudes[0] = 0.5
    with pytest.raises(ValueError):
        gauss_legendre(3).weights[0] = 0.5
    assert "custom" in repr(AngularGrid([1.0], [0.0]))


# ------------------------------------------------------------ layout

def test_sample_shapes_and_marks(plain, grid):
    s = plain
    nnode = s.radial.nspec * s.radial.ngll
    assert isinstance(s, Sample) and s.nnode == nnode
    assert list(s.fields) == ["rho", "vp", "vs"]
    for name in s.fields:
        assert s.fields[name].shape == (nnode,)
        assert s.is_radial(name) and s.stored_shape(name) == ()
        assert s.fields[name].dtype == np.float64
        assert not s.fields[name].flags.writeable
    assert s.displacement is None
    assert s.dimensions["rho"] == KG_M3 and s.characters["rho"] == DENSITY
    assert s.scales == Scales.SI
    assert s.layer_names == ("core", "shell", "crust")
    with pytest.raises(TypeError):
        s.fields["rho"] = np.zeros(nnode)
    assert "rho" in repr(s)


def test_radial_layout_is_the_mesh_flattened(plain):
    s = plain
    assert np.array_equal(s.radius, s.radial.r.ravel())
    assert np.array_equal(s.element_layer, s.radial.layer)
    assert np.array_equal(s.node_layer, np.repeat(s.radial.layer, s.radial.ngll))
    starts = np.arange(1, s.radial.nspec) * s.radial.ngll
    assert np.array_equal(s.radius[starts - 1], s.radius[starts])


def test_both_one_sided_values_survive_at_interfaces(plain):
    s = plain
    m = three_layers()
    rho = s.fields["rho"]
    for i, rb in enumerate(BOUNDS[1:-1]):
        below = np.flatnonzero(s.radial.right == rb)[0]
        above = np.flatnonzero(s.radial.left == rb)[0]
        last = (below + 1) * s.radial.ngll - 1
        first = above * s.radial.ngll
        assert s.radius[last] == rb == s.radius[first]
        assert rho[last] == m.layer(i)["rho"](rb)
        assert rho[first] == m.layer(i + 1)["rho"](rb)
        assert rho[last] != rho[first]


def test_homogeneous_model_samples_to_constants(grid):
    m = homogeneous(A, rho=5.0, vp=8.0, vs=4.0, name="ball")
    s = sample(m, grid, drmax=0.5 * A)
    assert s.layer_names == ("ball",)
    assert np.all(s.fields["rho"] == 5.0) and np.all(s.fields["vs"] == 4.0)
    check_sample(s, m)


def test_prem_on_a_coarse_mesh(coarse_prem, grid):
    m, s = coarse_prem
    assert s.radial.ngll == 3 and s.radial.nspec < 40
    assert set(s.fields) == set(m.common_names())
    assert "qmu" not in s.fields and "rho" in s.fields
    for name in s.fields:
        assert s.fields[name].shape == (s.nnode,)
    assert s.layer_names[1] == "outer_core"
    cmb = m.skeleton.boundaries[2]
    below = np.flatnonzero(s.radial.right == cmb)[0]
    above = np.flatnonzero(s.radial.left == cmb)[0]
    rho = s.fields["rho"]
    assert rho[(below + 1) * 3 - 1] == m.layer("outer_core")["rho"](cmb)
    assert rho[above * 3] == m.layer("lowermost_mantle")["rho"](cmb)
    check_sample(s, m)


# ------------------------------------------------------------ direction

def test_angular_fields_get_the_angular_axes(angular, grid):
    m, s = angular
    nt, nph = grid.ntheta, grid.nphi
    assert s.fields["rho"].shape == (s.nnode,)
    assert s.fields["rho3d"].shape == (s.nnode, nt, nph)
    assert s.fields["wind"].shape == (s.nnode, nt, nph, 3)
    assert not s.is_radial("rho3d") and not s.is_radial("wind")
    assert s.stored_shape("wind") == (3,)
    assert s.dimensions["rho3d"] is None and s.characters["rho3d"] == DENSITY
    r = s.radius[:, None, None]
    t = grid.colatitudes[None, :, None]
    p = grid.longitudes[None, None, :]
    assert np.allclose(s.fields["rho3d"], scalar_fn(r, t, p), rtol=1e-14)
    assert np.allclose(s.fields["wind"], vector_fn(r, t, p), rtol=1e-14)
    check_sample(s, m)


def test_a_radial_vector_is_direction_free(grid):
    m = three_layers()
    for layer in m.layers:
        m = m.with_field(layer.index, "wind",
                         AnalyticField(layer.interval, vector_fn, character=VECTOR))
    from planetmodel.fields import RadialField
    radial = m.without_field("wind")
    for layer in radial.layers:
        lo, hi = layer.interval
        radial = radial.with_field(
            layer.index, "wind",
            RadialField((lo, hi), [lambda r: r / A, 0.0, 2.0], character=VECTOR))
    s = sample(radial, grid, fields=["wind"], drmax=0.5 * A)
    assert s.fields["wind"].shape == (s.nnode, 3) and s.is_radial("wind")
    assert np.allclose(s.fields["wind"][:, 0], s.radius / A)
    assert np.all(s.fields["wind"][:, 2] == 2.0)
    check_sample(s, radial)
    # one analytic layer among radial ones makes the whole name angular
    mixed = radial.with_field(1, "wind", m.layer(1)["wind"], replace=True)
    s = sample(mixed, grid, fields=["wind"], drmax=0.5 * A)
    assert s.fields["wind"].shape == (s.nnode, grid.ntheta, grid.nphi, 3)
    check_sample(s, mixed)


# ------------------------------------------------------------ missing

def test_fields_default_to_the_common_names(coarse_prem, grid):
    m, s = coarse_prem
    assert list(s.fields) == list(m.common_names())
    t = sample(three_layers(), grid, fields=["vs", "rho"], drmax=0.5 * A)
    assert list(t.fields) == ["vs", "rho"]
    with pytest.raises(TypeError, match="sequence"):
        sample(three_layers(), grid, fields="rho", drmax=0.5 * A)
    with pytest.raises(KeyError, match="nope"):
        sample(three_layers(), grid, fields=["nope"], drmax=0.5 * A)


def test_a_missing_field_is_refused_or_nan(coarse_prem, grid):
    m, _ = coarse_prem
    with pytest.raises(KeyError, match="outer_core.*qmu"):
        sample(m, grid, fields=["qmu"], ngll=3, drmax=1e6)
    with pytest.raises(ValueError, match="missing"):
        sample(m, grid, fields=["qmu"], ngll=3, drmax=1e6, missing="zero")
    s = sample(m, grid, fields=["qmu", "rho"], ngll=3, drmax=1e6, missing="nan")
    fluid = np.isin(s.node_layer, [m.layer("outer_core").index,
                                   m.layer("ocean").index])
    assert fluid.any() and not fluid.all()
    assert np.all(np.isnan(s.fields["qmu"][fluid]))
    assert np.all(np.isfinite(s.fields["qmu"][~fluid]))
    assert np.all(np.isfinite(s.fields["rho"]))
    check_sample(s, m)


# ------------------------------------------------------------ displacement

def test_a_radial_stretch_samples_to_h_e_r(stretched, grid):
    m, s = stretched
    u = s.displacement
    assert u.shape == (s.nnode, grid.ntheta, grid.nphi, 3)
    assert u.dtype == np.float64 and not u.flags.writeable
    assert np.allclose(u[..., 1:], 0.0, atol=1e-12 * A)
    r = s.radius[:, None, None]
    t = grid.colatitudes[None, :, None]
    p = grid.longitudes[None, None, :]
    h = m.geometry.mapping.h(r, t, p)
    assert np.allclose(u[..., 0], h, rtol=1e-12, atol=1e-12 * A)
    assert np.max(np.abs(h)) > 1e-3 * A
    R = spherical_frame(grid.colatitudes[:, None], grid.longitudes[None, :])
    X = r[..., None] * R[None, ..., :, 0]
    cart = np.einsum("tpij,ntpj->ntpi", R, u)
    assert np.allclose(cart, m.geometry.mapping(X) - X, rtol=1e-12, atol=1e-12 * A)
    check_sample(s, m)


def test_an_identity_geometry_has_no_displacement(plain):
    assert plain.displacement is None
    check_sample(plain, three_layers())


# ------------------------------------------------------------ arguments

def test_the_radial_mesh_is_used_as_given(grid):
    m = three_layers()
    mesh = RadialMesh(m.geometry, ngll=4, drmax=0.5 * A)
    s = sample(m, grid, radial=mesh, fields=["rho"])
    assert s.radial is mesh and s.radial.ngll == 4
    check_sample(s, m)
    with pytest.raises(ValueError, match="not both"):
        sample(m, grid, radial=mesh, drmax=0.1 * A)
    with pytest.raises(ValueError, match="another skeleton"):
        sample(m, grid, radial=RadialMesh(Skeleton([0.0, A]), drmax=0.5 * A))
    with pytest.raises(TypeError, match="RadialMesh"):
        sample(m, grid, radial=Skeleton([0.0, A]))


def test_a_partial_mesh_samples_the_layers_it_covers(grid):
    m = prem()
    solid = RadialMesh(m.geometry, ngll=3, drmax=1e6, rmin=4e6, rmax=6368e3)
    s = sample(m, grid, radial=solid, fields=["qmu"])
    assert np.all(np.isfinite(s.fields["qmu"]))
    check_sample(s, m)
    fluid = RadialMesh(m.geometry, ngll=3, drmax=1e6, rmin=1.3e6, rmax=3e6)
    with pytest.raises(KeyError, match="no layer the mesh covers"):
        sample(m, grid, radial=fluid, fields=["qmu"], missing="nan")


def test_the_band_sizes_the_radial_mesh(grid):
    s = sample(three_layers(), grid, fields=["rho"])
    assert s.radial.drmax == pytest.approx(0.1 * A / (grid.lmax + 1))
    assert s.radial.ngll == 5
    g = equiangular(4, 8)
    with pytest.raises(ValueError, match="carries no band"):
        sample(three_layers(), g, fields=["rho"])
    s = sample(three_layers(), g, fields=["rho"], drmax=0.5 * A)
    assert s.fields["rho"].shape == (s.nnode,)
    with pytest.raises(TypeError, match="AngularGrid"):
        sample(three_layers(), (g.colatitudes, g.longitudes), drmax=0.5 * A)
    with pytest.raises(TypeError, match="Model"):
        sample(three_layers().geometry, g, drmax=0.5 * A)


# ------------------------------------------------------------ check_sample

def frozen(arr):
    arr = np.array(arr)
    arr.setflags(write=False)
    return arr


def replaced(s, name, arr):
    fields = dict(s.fields)
    fields[name] = frozen(arr)
    return dataclasses.replace(s, fields=fields)


def test_check_sample_catches_a_wrong_value(angular):
    m, s = angular
    arr = s.fields["rho3d"].copy()
    arr[3, 1, 2] *= 1.0 + 1e-6
    with pytest.raises(AssertionError, match="differs from the model"):
        check_sample(replaced(s, "rho3d", arr), m, n=400)
    arr = s.fields["rho"].copy()
    arr[:] += 1.0
    with pytest.raises(AssertionError, match="'rho' at node"):
        check_sample(replaced(s, "rho", arr), m, n=8)


def test_check_sample_catches_a_wrong_shape(angular):
    m, s = angular
    with pytest.raises(AssertionError, match="shape"):
        check_sample(replaced(s, "wind", s.fields["wind"][..., :2]), m)
    with pytest.raises(AssertionError, match="shape"):
        check_sample(replaced(s, "rho", s.fields["rho"][:-1]), m)
    with pytest.raises(AssertionError, match="shape"):
        check_sample(replaced(s, "rho", np.broadcast_to(
            s.fields["rho"][:, None, None], s.fields["rho3d"].shape)), m)


def test_check_sample_catches_a_writable_array_and_bad_metadata(plain):
    m = three_layers()
    fields = dict(plain.fields)
    fields["rho"] = plain.fields["rho"].copy()
    with pytest.raises(AssertionError, match="writable"):
        check_sample(dataclasses.replace(plain, fields=fields), m)
    chars = dict(plain.characters)
    chars["rho"] = VECTOR
    with pytest.raises(AssertionError, match="character"):
        check_sample(dataclasses.replace(plain, characters=chars), m)
    dims = dict(plain.dimensions)
    dims["rho"] = None
    with pytest.raises(AssertionError, match="dimensions"):
        check_sample(dataclasses.replace(plain, dimensions=dims), m)
    with pytest.raises(AssertionError, match="layer_names"):
        check_sample(dataclasses.replace(plain, layer_names=("a", "b", "c")), m)
    with pytest.raises(AssertionError, match="scales"):
        check_sample(dataclasses.replace(plain, scales=Scales.geophysical(A)), m)
    with pytest.raises(AssertionError, match="another skeleton"):
        check_sample(plain, homogeneous(A, rho=1.0, vp=1.0, vs=1.0))


def test_check_sample_catches_a_filled_hole_and_a_vanished_value(coarse_prem, grid):
    m, _ = coarse_prem
    s = sample(m, grid, fields=["qmu"], ngll=3, drmax=1e6, missing="nan")
    arr = s.fields["qmu"].copy()
    arr[np.isnan(arr)] = 0.0
    with pytest.raises(AssertionError, match="not NaN"):
        check_sample(replaced(s, "qmu", arr), m)
    arr = s.fields["qmu"].copy()
    arr[0] = np.nan
    with pytest.raises(AssertionError, match="non-finite"):
        check_sample(replaced(s, "qmu", arr), m)


def test_check_sample_catches_a_wrong_displacement(stretched, plain):
    m, s = stretched
    u = s.displacement.copy()
    u[..., 1] = u[..., 0]
    u[..., 0] = 0.0
    with pytest.raises(AssertionError, match="R\\^T"):
        check_sample(dataclasses.replace(s, displacement=frozen(u)), m)
    with pytest.raises(AssertionError, match="displacement is None"):
        check_sample(dataclasses.replace(s, displacement=None), m)
    with pytest.raises(AssertionError, match="identity"):
        check_sample(dataclasses.replace(plain, displacement=frozen(u[:plain.nnode])),
                     three_layers())
    with pytest.raises(AssertionError, match="writable"):
        check_sample(dataclasses.replace(s, displacement=u), m)
