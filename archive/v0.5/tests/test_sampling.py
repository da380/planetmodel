"""Sampling: a body evaluated on radial times angular nodes.

What has to be true of a Sample is stated by plan §3.12 and checked
here on a three-layer body of order-one radius: the angular grids are
the node sets they claim (Gauss-Legendre exact to its band, midpoint
otherwise), the radial nodes are the mesh's per-element nodes with
both one-sided values kept at every interface, angular fields equal
their formula at every product point, a radial field is stored on
`(node,)` and says so, and a RadialStretch samples to `(h, 0, 0)` in
the spherical frame -- the fact the exporters rely on.  `check_sample`
is run on every sample built and shown to fail on tampered copies, so
the contract has teeth before the netCDF writer leans on it.

The last section adds a body with a hole: a density-only
core, an empty shell and a vacuum buffer.  The sample carries each
field's domain, holds NaN on the nodes outside it, and `check_sample`
fails both on a hole that filled itself and on a value that vanished.

Meshes are a few elements across; nothing here is Earth-sized.
"""
import dataclasses

import numpy as np
import pytest
from scipy.fft import next_fast_len

import planetmodel
from planetmodel import (DENSITY, VECTOR, AnalyticField, AnalyticTopography,
                    Dimensions, IdentityMapping, RadialField, RadialMesh,
                    ReferenceBody, Skeleton, Symmetry, layer_linear)
from planetmodel.model.materials import ElasticField
from planetmodel.model.frames import spherical_frame
from planetmodel.sampling import AngularGrid, Sample, SampleMetadata, sample_body
from planetmodel.testing import check_sample

A = 1.0e6
SK = Skeleton([0.0, 0.3 * A, 0.7 * A, A])
POLYS = ((13.0e3, -2.0e3), (5.0e3, -1.0e3), (3.0e3, 0.0))   # c0 + c1 (r/A)^2


def poly(i):
    c0, c1 = POLYS[i]
    return lambda r: c0 + c1 * (np.asarray(r, dtype=float) / A) ** 2


def const(v):
    return lambda r: np.full_like(np.asarray(r, dtype=float), v)


def scalar_fn(r, t, p):
    return 5.0e3 + 1.0e3 * (r / A) ** 2 * (1.0 + 0.1 * np.sin(t) * np.cos(p))


def vector_fn(r, t, p):
    r, t, p = np.broadcast_arrays(r, t, p)
    return np.stack([r / A + 0.0 * t, np.sin(t) * np.cos(p),
                     0.5 * np.cos(t) + 0.0 * p], axis=-1)


def body():
    rho = RadialField(SK, [poly(i) for i in range(3)], name="rho",
                      character=DENSITY, dimensions=Dimensions.DENSITY)
    scal = AnalyticField(scalar_fn, SK, character=DENSITY, name="rho3d")
    vec = AnalyticField(vector_fn, SK, character=VECTOR, name="v")
    elastic = ElasticField(Symmetry.ISOTROPIC,
                           {"kappa": RadialField(SK, [const(1.3e11)] * 3),
                            "mu": RadialField(SK, [const(6.7e10)] * 3)},
                           name="elastic_moduli")
    return (ReferenceBody.from_fields(SK, {"rho": rho, "rho3d": scal, "v": vec,
                               "elastic_moduli": elastic})
            .name_interface(-1, "surface"))


@pytest.fixture(scope="module")
def grid():
    return AngularGrid.gauss_legendre(4)


@pytest.fixture(scope="module")
def plain(grid):
    return body().sample(grid, drmax=0.2 * A)


@pytest.fixture(scope="module")
def mapped(grid):
    b = body().with_surface("surface", AnalyticTopography(
        lambda t, p: 2.0e4 * np.sin(t) * np.cos(p)))
    m = b.mapping(rule=layer_linear())
    return b.sample(grid, mapping=m, drmax=0.2 * A), m


# ------------------------------------------------------------ AngularGrid

def test_gauss_legendre_nodes_and_band():
    lmax = 6
    g = AngularGrid.gauss_legendre(lmax)
    assert g.kind == "gauss_legendre" and g.lmax == lmax
    assert g.ntheta == lmax + 1
    assert np.all(np.diff(g.colatitudes) > 0)
    assert 0 < g.colatitudes[0] and g.colatitudes[-1] < np.pi
    assert g.nphi >= 2 * lmax + 1 and g.nphi == next_fast_len(2 * lmax + 1)
    assert np.isclose(g.weights.sum(), 2.0, rtol=1e-14)
    assert g.longitudes[0] == 0.0 and np.allclose(
        np.diff(g.longitudes), 2 * np.pi / g.nphi)


def test_gauss_legendre_is_exact_to_its_band():
    lmax = 6
    g = AngularGrid.gauss_legendre(lmax)
    x = np.cos(g.colatitudes)
    for l in range(lmax + 1):
        P = np.polynomial.legendre.Legendre.basis(l)(x)
        assert np.isclose(np.sum(g.weights * P * P), 2.0 / (2 * l + 1),
                          rtol=1e-13)
    dphi = 2 * np.pi / g.nphi
    for m in range(1, lmax + 1):
        for k in range(1, lmax + 1):
            got = dphi * np.sum(np.cos(m * g.longitudes)
                                * np.cos(k * g.longitudes))
            assert np.isclose(got, np.pi if m == k else 0.0, atol=1e-13)


def test_gauss_legendre_refuses_too_few_longitudes():
    with pytest.raises(ValueError, match="2 lmax \\+ 1"):
        AngularGrid.gauss_legendre(4, nphi=8)
    assert AngularGrid.gauss_legendre(4, nphi=9).nphi == 9


def test_equiangular_grid():
    g = AngularGrid.equiangular(16, 32)
    assert g.kind == "equiangular" and g.lmax is None
    assert g.ntheta == 16 and g.nphi == 32
    assert 0 < g.colatitudes[0] and g.colatitudes[-1] < np.pi
    assert np.allclose(g.colatitudes, np.pi * (np.arange(16) + 0.5) / 16)
    assert np.allclose(g.weights, (np.pi / 16) * np.sin(g.colatitudes))
    coarse = abs(AngularGrid.equiangular(64, 4).weights.sum() - 2.0)
    fine = abs(AngularGrid.equiangular(256, 4).weights.sum() - 2.0)
    assert coarse < 1e-3 and fine < coarse / 4


@pytest.mark.parametrize("bad", [
    dict(colatitudes=[2.0, 1.0], longitudes=[0.0]),
    dict(colatitudes=[0.0, 1.0], longitudes=[0.0]),
    dict(colatitudes=[1.0, np.pi], longitudes=[0.0]),
    dict(colatitudes=[1.0], longitudes=[0.0, 2 * np.pi]),
    dict(colatitudes=[1.0], longitudes=[-0.1]),
    dict(colatitudes=[1.0], longitudes=[0.0], kind="lebedev"),
    dict(colatitudes=[1.0, 2.0], longitudes=[0.0], weights=[1.0]),
    dict(colatitudes=[1.0], longitudes=[0.0], lmax=-1),
])
def test_angular_grid_validation(bad):
    with pytest.raises(ValueError):
        AngularGrid(**bad)


def test_angular_grid_arrays_are_read_only():
    g = AngularGrid.equiangular(4, 8)
    with pytest.raises(ValueError):
        g.colatitudes[0] = 0.5


# ------------------------------------------------------------ the sample

def test_sample_shapes_and_marks(plain, grid):
    s = plain
    nnode = s.radial.nspec * s.radial.ngll
    assert s.nnode == nnode
    assert s.fields["rho"].shape == (nnode,)
    assert s.fields["rho3d"].shape == (nnode, grid.ntheta, grid.nphi)
    assert s.fields["v"].shape == (nnode, grid.ntheta, grid.nphi, 3)
    assert s.fields["elastic_moduli"].shape == (nnode, 6, 6)
    assert s.is_radial("rho") and s.is_radial("elastic_moduli")
    assert not s.is_radial("rho3d") and not s.is_radial("v")
    assert s.displacement is None
    assert set(s.metadata.characters) == {"rho", "rho3d", "v", "elastic_moduli"}
    assert s.metadata.dimensions["rho"] == Dimensions.DENSITY
    assert s.metadata.frames == {k: "spherical" for k in s.fields}
    assert s.metadata.skeleton == SK
    assert isinstance(s.metadata, SampleMetadata)
    for arr in s.fields.values():
        assert arr.flags.c_contiguous and arr.dtype == np.float64


def test_radial_layout_is_the_mesh_flattened(plain):
    s = plain
    assert np.array_equal(s.radius, s.radial.r.ravel())
    assert np.array_equal(s.element_start,
                          np.arange(s.radial.nspec + 1) * s.radial.ngll)
    assert np.array_equal(s.element_layer, s.radial.layer)
    # every element boundary is a repeated radius, one node each side
    inner = s.element_start[1:-1]
    assert np.array_equal(s.radius[inner - 1], s.radius[inner])


def test_both_one_sided_values_survive_at_interfaces(plain):
    s = plain
    rho = s.fields["rho"]
    for i, rb in enumerate(SK.inner_boundaries):
        below = np.flatnonzero(s.radial.right == rb)[0]
        above = np.flatnonzero(s.radial.left == rb)[0]
        last = s.element_start[below + 1] - 1
        first = s.element_start[above]
        assert s.radius[last] == rb == s.radius[first]
        assert np.isclose(rho[last], poly(i)(rb), rtol=1e-14)
        assert np.isclose(rho[first], poly(i + 1)(rb), rtol=1e-14)
        assert rho[last] != rho[first]


def test_angular_fields_equal_their_formula(plain, grid):
    s = plain
    r = s.radius[:, None, None]
    t = grid.colatitudes[None, :, None]
    p = grid.longitudes[None, None, :]
    assert np.allclose(s.fields["rho3d"], scalar_fn(r, t, p), rtol=1e-14)
    assert np.allclose(s.fields["v"], vector_fn(r, t, p), rtol=1e-14)


def test_check_sample_passes(plain, mapped):
    check_sample(plain)
    check_sample(mapped[0])


# ------------------------------------------------------------ displacement

def test_radial_stretch_samples_to_h_e_r(mapped, grid):
    s, m = mapped
    u = s.displacement
    assert u.shape == (s.nnode, grid.ntheta, grid.nphi, 3)
    assert np.allclose(u[..., 1:], 0.0, atol=1e-12 * A)
    r = s.radius[:, None, None]
    t = grid.colatitudes[None, :, None]
    p = grid.longitudes[None, None, :]
    h = m.h(r, t, p)
    assert np.allclose(u[..., 0], h, rtol=1e-12, atol=1e-12 * A)
    assert np.max(np.abs(h)) > 1.0e3            # the relief actually reached
    # R u_sph reproduces the Cartesian displacement at a few points
    R = spherical_frame(grid.colatitudes[:, None], grid.longitudes[None, :])
    X = r[..., None] * R[None, ..., :, 0]
    cart = np.einsum("tpij,ntpj->ntpi", R, u)
    assert np.allclose(cart, m.displacement(X), rtol=1e-12, atol=1e-12 * A)


def test_identity_mapping_samples_to_zero(grid):
    s = body().sample(grid, fields=["rho"], mapping=IdentityMapping(),
                      drmax=0.25 * A)
    assert s.displacement.shape == (s.nnode, grid.ntheta, grid.nphi, 3)
    assert np.all(s.displacement == 0.0)
    check_sample(s)


# ------------------------------------------------------------ arguments

def test_radial_mesh_is_used_as_given(grid):
    b = body()
    mesh = RadialMesh(b, ngll=4, drmax=0.5 * A)
    s = b.sample(grid, radial=mesh, fields=["rho"])
    assert s.radial is mesh and s.radial.ngll == 4
    with pytest.raises(ValueError, match="not both"):
        b.sample(grid, radial=mesh, drmax=0.1 * A)


def test_band_sizes_the_radial_mesh(grid):
    s = body().sample(grid, fields=["rho"])
    assert s.radial.drmax == pytest.approx(0.1 * A / (grid.lmax + 1))
    assert s.radial.ngll == 5


def test_custom_grid_needs_a_radial_size():
    g = AngularGrid.equiangular(4, 8)
    b = body()
    with pytest.raises(ValueError, match="carries no band"):
        b.sample(g, fields=["rho"])
    s = b.sample(g, fields=["rho"], drmax=0.5 * A)
    assert s.fields["rho"].shape == (s.nnode,)


def test_fields_by_name_and_by_dict(grid):
    b = body()
    s = b.sample(grid, fields=["v", "rho"], drmax=0.5 * A)
    assert list(s.fields) == ["v", "rho"]
    extra = AnalyticField(lambda r, t, p: r / A + np.cos(t) + 0.0 * p, SK,
                          name="extra")
    s = b.sample(grid, fields={"extra": extra, "rho": b["rho"]},
                 drmax=0.5 * A)
    assert list(s.fields) == ["extra", "rho"]
    assert s.source["extra"] is extra
    check_sample(s)
    with pytest.raises(KeyError):
        b.sample(grid, fields=["nope"], drmax=0.5 * A)
    with pytest.raises(TypeError):
        b.sample(grid, fields="rho", drmax=0.5 * A)
    other = AnalyticField(lambda r, t, p: r + 0.0 * t * p,
                          Skeleton([0.0, A]), name="other")
    with pytest.raises(ValueError, match="different skeleton"):
        b.sample(grid, fields={"other": other}, drmax=0.5 * A)
    with pytest.raises(ValueError, match="different skeleton"):
        b.sample(grid, fields=["rho"],
                 radial=RadialMesh(ReferenceBody.from_fields(Skeleton([0.0, A]), {}),
                                   drmax=0.5 * A))


def test_sample_body_is_the_method(grid):
    b = body()
    s1 = b.sample(grid, fields=["rho"], drmax=0.5 * A)
    s2 = sample_body(b, grid, fields=["rho"], drmax=0.5 * A)
    assert np.array_equal(s1.fields["rho"], s2.fields["rho"])
    assert isinstance(s1, Sample)


# ------------------------------------------------------------ check_sample

def test_check_sample_catches_a_wrong_value(plain):
    bad = dict(plain.fields)
    arr = bad["rho3d"].copy()
    arr[3, 1, 2] *= 1.0 + 1e-6
    bad["rho3d"] = arr
    with pytest.raises(AssertionError, match="differs from its source"):
        check_sample(dataclasses.replace(plain, fields=bad), n=400)


def test_check_sample_catches_a_wrong_radial_value(plain):
    bad = dict(plain.fields)
    arr = bad["rho"].copy()
    arr[:] += 1.0
    bad["rho"] = arr
    with pytest.raises(AssertionError, match="differs from its source"):
        check_sample(dataclasses.replace(plain, fields=bad), n=8)


def test_check_sample_catches_a_wrong_shape(plain, grid):
    bad = dict(plain.fields)
    bad["v"] = np.ascontiguousarray(bad["v"][..., :2])
    with pytest.raises(AssertionError, match="shape"):
        check_sample(dataclasses.replace(plain, fields=bad))


def test_check_sample_catches_a_wrong_frame(mapped):
    s, _ = mapped
    u = s.displacement.copy()
    u[..., 1] = u[..., 0]
    u[..., 0] = 0.0
    with pytest.raises(AssertionError, match="R\\^T"):
        check_sample(dataclasses.replace(s, displacement=u))


def test_check_sample_without_a_source_checks_layout_only(plain):
    bare = dataclasses.replace(plain, source=None, mapping=None)
    check_sample(bare)
    bad = dict(bare.fields)
    bad["rho"] = bad["rho"][:-1]
    with pytest.raises(AssertionError, match="shape"):
        check_sample(dataclasses.replace(bare, fields=bad))


# ------------------------------------------------------------ exports

def test_exports():
    assert planetmodel.AngularGrid is AngularGrid and planetmodel.Sample is Sample
    assert "AngularGrid" in planetmodel.__all__ and "Sample" in planetmodel.__all__
    assert "check_sample" in planetmodel.testing.__all__


# ------------------------------------------------------------ partial domains
#
# A field belongs to one layer, so a body may hold a
# density in its core and nothing above it.  The sample carries the
# domain per field and NaN outside it: a hole travels as a hole.

B = 1.0                       # a unit body: nothing here is Earth-sized
PART_SK = Skeleton([0.0, 0.5 * B, B])


def partial_body():
    """A density-only core, an empty solid shell and a vacuum buffer.

    `rho` is held by the core alone (a RadialField with None above it),
    `scal` by the two original layers, and the shell grown by
    `extended(fields=None)` and the buffer hold nothing at all.
    """
    rho = RadialField(PART_SK,
                      [lambda r: 5.0e3 - 2.0e3 * (np.asarray(r) / B) ** 2, None],
                      name="rho", character=DENSITY,
                      dimensions=Dimensions.DENSITY)
    scal = AnalyticField(lambda r, t, p: 1.0 + r / B + 0.1 * np.sin(t)
                         * np.cos(p), PART_SK, character=DENSITY, name="scal")
    body = ReferenceBody.from_fields(PART_SK, {"rho": rho, "scal": scal})
    return (body.extended([1.4 * B], fields=None, names=["crust"])
            .with_buffer(ratio=0.2))


@pytest.fixture(scope="module")
def partial(grid):
    return partial_body().sample(grid, drmax=0.3 * B)


def test_a_partial_field_samples_without_raising(partial, grid):
    s = partial
    assert s.metadata.domains == {"rho": (0,), "scal": (0, 1)}
    layer = np.repeat(s.element_layer, s.radial.ngll)
    assert set(np.unique(layer)) == {0, 1, 2, 3}
    for name, domain in s.metadata.domains.items():
        inside = np.isin(layer, domain)
        arr = s.fields[name]
        flat = arr.reshape(arr.shape[0], -1)
        assert np.all(np.isfinite(flat[inside]))
        assert np.all(np.isnan(flat[~inside]))
    # and the values inside are the field's own
    r = s.radius[layer == 0]
    assert np.allclose(s.fields["rho"][layer == 0],
                       partial_body()["rho"].evaluate(r, layer=0), rtol=1e-14)


def test_check_sample_is_green_on_a_partial_body(partial):
    check_sample(partial)


def test_check_sample_catches_a_filled_hole(partial):
    bad = dict(partial.fields)
    arr = bad["rho"].copy()
    arr[np.isnan(arr)] = 0.0           # zero-fill, which the library never does
    bad["rho"] = arr
    with pytest.raises(AssertionError, match="not NaN"):
        check_sample(dataclasses.replace(partial, fields=bad))


def test_check_sample_catches_a_vanished_value(partial):
    bad = dict(partial.fields)
    arr = bad["rho"].copy()
    arr[0] = np.nan
    bad["rho"] = arr
    with pytest.raises(AssertionError, match="non-finite"):
        check_sample(dataclasses.replace(partial, fields=bad))


# ---------------------------------------------- a sample at a chosen frequency

def dynamic_body():
    """The body's moduli under Maxwell relaxation, on every layer."""
    from planetmodel.model.rheology import maxwell
    b = body()
    eta = RadialField(SK, [const(1.0e21)] * 3, name="viscosity",
                      dimensions=Dimensions.VISCOSITY)
    b.add_field("viscosity", eta)
    b.add_field("viscoelastic_moduli", maxwell(b["elastic_moduli"], eta))
    return b


OMEGA = 1.0e-11        # rad/s: the Maxwell time here is ~1.6e10 s


def test_a_frequency_field_is_sampled_at_a_chosen_omega(grid):
    b = dynamic_body()
    s = b.sample(grid, fields=["viscoelastic_moduli", "rho"], drmax=0.5 * A,
                 omega=OMEGA)
    assert s.metadata.omegas == {"viscoelastic_moduli": OMEGA}
    assert s.fields["viscoelastic_moduli"].shape == (s.nnode, 6, 6, 2)
    assert s.fields["viscoelastic_moduli"].dtype == np.float64
    assert s.is_radial("viscoelastic_moduli")
    layer = np.repeat(s.element_layer, s.radial.ngll)
    want = b["viscoelastic_moduli"].evaluate(s.radius[layer == 1], layer=1,
                                        omega=OMEGA, part="complex")
    got = s.fields["viscoelastic_moduli"][layer == 1]
    assert np.array_equal(got[..., 0], np.real(want))
    assert np.array_equal(got[..., 1], np.imag(want))
    assert np.any(got[..., 1] != 0.0), "nothing relaxed: the test is blind"
    check_sample(s)


def test_an_omega_adds_the_frequency_fields_to_the_default_set(grid):
    b = dynamic_body()
    assert "viscoelastic_moduli" not in b.sample(grid, drmax=0.5 * A).fields
    s = b.sample(grid, drmax=0.5 * A, omega=OMEGA)
    assert "viscoelastic_moduli" in s.fields and "rho" in s.fields
    assert list(s.metadata.omegas) == ["viscoelastic_moduli"]
    check_sample(s)


def test_a_frequency_field_without_an_omega_is_refused(grid):
    b = dynamic_body()
    with pytest.raises(ValueError, match="pass omega="):
        b.sample(grid, fields=["viscoelastic_moduli"], drmax=0.5 * A)
    with pytest.raises(ValueError, match="scalar"):
        b.sample(grid, drmax=0.5 * A, omega=[1.0, 2.0])
    with pytest.raises(ValueError, match="omega must be real"):
        b.sample(grid, drmax=0.5 * A, omega=1.0e-11 + 1.0e-12j)


def test_check_sample_catches_swapped_real_and_imaginary_parts(grid):
    s = dynamic_body().sample(grid, fields=["viscoelastic_moduli"],
                              drmax=0.5 * A, omega=OMEGA)
    bad = {"viscoelastic_moduli": np.ascontiguousarray(
        s.fields["viscoelastic_moduli"][..., ::-1])}
    with pytest.raises(AssertionError, match="differs from its source"):
        check_sample(dataclasses.replace(s, fields=bad))


def test_a_mesh_says_which_field_and_which_layer():
    body = partial_body()
    core = RadialMesh(body, drmax=0.3 * B, rmax=0.5 * B)   # inside the domain
    assert np.allclose(core.nodal("rho"), body["rho"][0](core.r), rtol=1e-14)
    with pytest.raises(ValueError, match=r"'rho' is not defined on layer 1"):
        RadialMesh(body, drmax=0.3 * B).nodal("rho")
    with pytest.raises(ValueError, match="rho"):
        RadialMesh(body, drmax=0.3 * B).nodal_derivative("rho")


def test_a_vacuum_layer_is_fluid_to_the_mesh():
    body = partial_body()
    mesh = RadialMesh(body, drmax=0.3 * B)
    vacuum = np.array([body.layers[i].is_vacuum for i in mesh.layer])
    assert vacuum.any() and np.all(mesh.is_fluid[vacuum])
    assert not mesh.is_fluid[~vacuum].any()
