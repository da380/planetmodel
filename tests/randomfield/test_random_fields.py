"""Random fields: exact marginals, statistics, padding, the harmonics, the
samplers' views of a coefficient vector and of a batch of them, the Matern
covariance in a shell, and a layered sample as model fields."""
import numpy as np
import pytest
from scipy.special import eval_legendre, gamma, kv

from planetmodel import (PREM, Skeleton, gauss_legendre, real_harmonics, synthesise,
                         testing)
from planetmodel.expansions import HarmonicField
from planetmodel.fields import RadialField
from planetmodel.harmonics import synthesise_grid
from planetmodel.layerfunction import PolynomialLayer
from planetmodel.randomfield import LayeredGRF, RadialGRF, SphericalGRF


def matern(r, nu, lam):
    """The Matern correlation of smoothness nu and length scale lam, the
    covariance of (1 - lam^2 Laplacian)^-(nu + d / 2) white noise over
    its variance in any dimension d: 2^(1 - nu) / Gamma(nu) x^nu K_nu(x)
    with x = r / lam."""
    x = np.asarray(r, dtype=float) / lam
    safe = np.where(x > 0.0, x, 1.0)
    return np.where(x > 0.0, 2.0 ** (1.0 - nu) / gamma(nu) * safe ** nu
                    * kv(nu, safe), 1.0)


def cartesian(points):
    """Points (r, theta, phi) along the last axis as (x, y, z)."""
    r, theta, phi = np.moveaxis(np.asarray(points, dtype=float), -1, 0)
    return np.stack([r * np.sin(theta) * np.cos(phi),
                     r * np.sin(theta) * np.sin(phi), r * np.cos(theta)], axis=-1)


def test_radial_grf_normalisation_and_stats():
    sig = lambda r: 1.0 + r
    g = RadialGRF(0.5, 1.0, 1.5, 0.08, sigma=sig, tol=1e-10)
    B = g.factor()
    assert B.shape == (g.r.size, g.nmodes)
    assert np.max(np.abs(np.sqrt((B ** 2).sum(axis=1)) - sig(g.r))) < 1e-10
    assert np.max(np.abs(np.diag(g.covariance()) - sig(g.r) ** 2)) < 1e-10
    assert np.array_equal(g.std(), sig(g.r))

    c = g.sample(rng=0, size=8000)
    assert c.shape == (8000, g.nmodes) == (8000, g.size)
    # the coefficients have their standard deviations, the field its own
    assert np.max(np.abs(c.std(axis=0) / g.coefficient_std - 1.0)) < 0.05
    u = g.synthesise(c)
    assert u.shape == (8000, g.r.size)
    assert np.max(np.abs(u.std(axis=0) / sig(g.r) - 1.0)) < 0.06
    assert np.max(np.abs(u.mean(axis=0)) / sig(g.r)) < 0.05

    layer = g.to_layer(c[0])
    assert np.max(np.abs(layer(g.r) - u[0])) < 1e-10
    assert layer.interval == (0.5, 1.0)
    field = g.to_field(c[0], name="dv")
    testing.check_field(field)
    assert field.name == "dv" and np.max(np.abs(field(g.r) - u[0])) < 1e-10

    ball = RadialGRF(0.0, 1.0, 1.0, 0.2, sigma=2.0)
    assert ball.r[0] == 0.0                          # the axis is a physical node
    assert np.max(np.abs(np.sqrt((ball.factor() ** 2).sum(axis=1)) - 2.0)) < 1e-10
    assert np.isfinite(ball.synthesise(ball.sample(rng=4))[0])
    with pytest.raises(ValueError):
        RadialGRF(0.5, 0.4, 1.0, 0.1)
    with pytest.raises(ValueError):
        RadialGRF(0.5, 1.0, -1.0, 0.1)
    with pytest.raises(ValueError):
        g.to_layer(c[0, :3])


def test_a_radial_sample_is_the_coefficient_vector():
    g = RadialGRF(0.5, 1.0, 1.5, lambda r: 0.05 + 0.1 * r, sigma=lambda r: 1.0 + r)
    c = g.sample(rng=4)
    assert c.shape == (g.nmodes,) == (g.size,) == (g.basis.nmodes,)
    np.testing.assert_array_equal(c, g.sample(rng=4))
    assert not np.array_equal(c, g.sample(rng=5))
    # independent standard normals times theta^-beta, in the basis's order
    z = np.random.default_rng(4).standard_normal(g.size)
    np.testing.assert_allclose(c, z * g.basis.theta ** -g.beta, rtol=1e-14)
    np.testing.assert_array_equal(g.coefficient_std, g.basis.theta ** -g.beta)
    # the field of a vector: the basis's synthesis times the scale
    values = g.synthesise(c)
    assert values.shape == (g.r.size,)
    np.testing.assert_allclose(values, g.basis.synthesise(c) * g.scale, rtol=1e-13,
                               atol=1e-15)
    np.testing.assert_allclose(values, g.factor() @ z, rtol=0.0,
                               atol=1e-13 * np.abs(values).max())
    np.testing.assert_allclose(g.to_layer(c)(g.r), values, rtol=0.0,
                               atol=1e-12 * np.abs(values).max())
    for held in (g.sigma, g.scale, g.coefficient_std):
        assert not held.flags.writeable


def test_a_batch_is_successive_draws():
    """For every sampler and both white noises, a batch of `size` is
    `size` successive single draws from one generator: the same normals,
    so the same coefficients, and for the white noises the same samples
    to the rounding of the matrix product, which differs between a
    product with a vector and one with a matrix."""
    from planetmodel.randomfield import (SeparableOperatorFamily, SpectralBasis,
                                         padded_mesh)
    g = RadialGRF(0.5, 1.0, 1.0, 0.1)
    shell = SphericalGRF(0.5, 1.0, 1.0, 0.2, lmax=4, tol=1e-3)
    layered = LayeredGRF(Skeleton([0.0, 0.4, 0.7, 1.0]), 1.0, 0.1, layers=(0, 2))
    for sampler in (g, shell, layered):
        batch = sampler.sample(rng=4, size=3)
        assert batch.shape == (3, sampler.size)
        rng = np.random.default_rng(4)
        for row in batch:
            np.testing.assert_array_equal(row, sampler.sample(rng=rng))
        assert sampler.sample(rng=4, size=0).shape == (0, sampler.size)
    fam = SeparableOperatorFamily(padded_mesh(0.5, 1.0, pad=0.3, weight="r2", ngll=5,
                                           drmax=0.1), kappa=0.02, weight="r2")
    basis = SpectralBasis(fam, 2, nmodes=5)
    cases = ((lambda rng: basis.white_noise(rng=rng),
              lambda rng: basis.white_noise(rng=rng, size=3).T),
             (lambda rng: fam.white_noise(2, rng=rng),
              lambda rng: fam.white_noise(2, rng=rng, size=3).T))
    for single, batch in cases:
        rng = np.random.default_rng(4)
        each = np.array([single(rng) for _ in range(3)])
        assert np.allclose(batch(np.random.default_rng(4)), each, rtol=0.0,
                           atol=1e-13 * np.abs(each).max())


def test_a_batch_goes_through_every_method():
    """Whatever takes a coefficient vector takes a batch of them: an array
    comes back with the batch axis leading and a field as a tuple over
    the batch, each entry what the vector alone gives."""
    g = RadialGRF(0.5, 1.0, 1.0, 0.1, sigma=lambda r: 1.0 + r)
    c = g.sample(rng=1, size=3)
    u = g.synthesise(c)
    assert u.shape == (3, g.r.size)
    layers, fields = g.to_layer(c), g.to_field(c, name="dv")
    assert isinstance(layers, tuple) and isinstance(fields, tuple)
    assert len(layers) == len(fields) == 3
    for k in range(3):
        np.testing.assert_allclose(u[k], g.synthesise(c[k]), rtol=1e-13, atol=1e-15)
        assert isinstance(layers[k], PolynomialLayer)
        assert isinstance(fields[k], RadialField) and fields[k].name == "dv"
        np.testing.assert_array_equal(layers[k](g.r), g.to_layer(c[k])(g.r))
        np.testing.assert_array_equal(fields[k](g.r), g.to_field(c[k])(g.r))
    assert g.synthesise(c[:0]).shape == (0, g.r.size) and g.to_field(c[:0]) == ()

    s = SphericalGRF(0.5, 1.0, 1.0, 0.2, sigma=lambda r: 1.0 + r, lmax=4, tol=1e-3)
    grid = gauss_legendre(4)
    c = s.sample(rng=2, size=3)
    u, values, pp = s.synthesise(c), s.on_grid(c, grid), s.coefficient_functions(c)
    assert u.shape == (3, 2, 5, 5, s.r.size)
    assert values.shape == (3, s.r.size, grid.ntheta, grid.nphi)
    np.testing.assert_array_equal(s.sample_grid(grid, rng=2, size=3), values)
    assert pp(0.8).shape == (3, 2, 5, 5)
    fields = s.to_field(c, name="dv")
    assert isinstance(fields, tuple) and len(fields) == 3
    for k in range(3):
        np.testing.assert_array_equal(u[k], s.synthesise(c[k]))
        np.testing.assert_array_equal(values[k], s.on_grid(c[k], grid))
        np.testing.assert_array_equal(pp(0.8)[k], s.coefficient_functions(c[k])(0.8))
        assert isinstance(fields[k], HarmonicField) and fields[k].name == "dv"
        np.testing.assert_array_equal(fields[k].on_grid(s.r, grid),
                                      s.to_field(c[k]).on_grid(s.r, grid))
    assert s.on_grid(c[:0], grid).shape == (0, s.r.size, grid.ntheta, grid.nphi)

    sk = Skeleton([0.0, 0.4, 0.7, 1.0])
    layered = LayeredGRF(sk, 1.0, 0.1, layers=(0, 2), name="dv")
    c = layered.sample(rng=3, size=2)
    both = layered.to_fields(c)
    assert len(both) == 2 and all(len(fields) == 3 for fields in both)
    for k in range(2):
        alone = layered.to_fields(c[k])
        for i in range(3):
            r = np.linspace(*sk.interval(i), 7)
            np.testing.assert_array_equal(both[k][i](r), alone[i](r))

    # one vector or a batch of them, and nothing else
    for sampler, methods in ((g, ("synthesise", "to_layer", "to_field")),
                             (s, ("synthesise", "coefficient_functions", "to_field")),
                             (layered, ("to_fields",))):
        for method in methods:
            for wrong in (np.zeros(sampler.size + 1), np.zeros((2, 3, sampler.size)),
                          np.zeros((sampler.size, 2)), 0.0):
                with pytest.raises(ValueError, match="coefficient vector"):
                    getattr(sampler, method)(wrong)
    with pytest.raises(ValueError, match="coefficient vector"):
        s.on_grid(np.zeros(s.size + 1), grid)


def test_the_samplers_are_built_on_their_bases():
    """The samplers hold the bases they are made of, and their covariances
    match recorded reference values, chosen so that the sign of an
    eigenvector does not enter."""
    from planetmodel.randomfield import SpectralBasis, SphericalBasis

    g = RadialGRF(0.3, 1.0, 1.5, 0.2, sigma=lambda r: 1.0 + r, drmax=0.05)
    assert isinstance(g.basis, SpectralBasis) and g.basis.label == 0
    testing.check_spectral_basis(g.basis)
    assert g.basis.restriction is g.restriction and g.basis.r is g.r
    assert g.nmodes == g.basis.nmodes == 142
    C = g.covariance()
    assert np.allclose(C[[0, 10, 56], [0, 12, 40]],
                       [1.689999999999999, 2.0511116218346936, 2.6489316371911413],
                       rtol=1e-9, atol=0.0)
    raw = g.basis.pointwise_variance(2.0 * g.beta)
    B = g.basis.synthesise(np.diag(g.basis.theta ** -g.beta))
    assert np.allclose(g.factor(), (g.sigma / np.sqrt(raw))[:, None] * B, rtol=1e-12)
    assert np.allclose(g.scale, g.sigma / np.sqrt(raw), rtol=1e-12)

    b = RadialGRF(0.0, 1.0, 0.5, lambda r: 0.2 + 0.1 * r, drmax=0.06, robin="auto")
    Cb = b.covariance()
    assert b.nmodes == 132
    assert np.allclose(Cb[[0, 5, 68], [0, 9, 60]],
                       [1.0, 0.6276750404062182, 0.6686527605922784],
                       rtol=1e-9, atol=0.0)

    s = SphericalGRF(0.5, 1.0, 1.0, 0.25, lmax=4, lam_h=0.4, drmax=0.06)
    assert isinstance(s.basis, SphericalBasis) and s.basis.lmax == 4
    testing.check_spherical_basis(s.basis)
    assert list(s.nmodes) == list(s.basis.nmodes) == [64, 71, 79, 87, 94]
    C3 = s.factor(3) @ s.factor(3).T
    assert np.allclose(C3[[0, 10, 36], [0, 14, 30]],
                       [0.09763069928430478, 0.1526205920536206, 0.2459225804991355],
                       rtol=1e-9, atol=0.0)
    auto = SphericalGRF(0.0, 0.8, 1.5, 0.3, drmax=0.08, tol=1e-4)
    assert auto.lmax == 50 and auto.basis.lmax == 50


def test_padding_insensitivity():
    kw = dict(sigma=1.0, tol=1e-12)
    g2 = RadialGRF(0.6, 1.0, 1.0, 0.04, pad_factor=2.0, **kw)
    g4 = RadialGRF(0.6, 1.0, 1.0, 0.04, pad_factor=4.0, **kw)
    assert np.array_equal(g2.r, g4.r)
    assert np.max(np.abs(g2.covariance() - g4.covariance())) < 5e-4
    auto = RadialGRF(0.6, 1.0, 1.0, 0.04, robin="auto", **kw)
    assert auto.family is not None and auto.nmodes >= 1
    with pytest.raises(ValueError):
        RadialGRF(0.6, 1.0, 1.0, 0.04, robin="sideways")


def test_real_harmonics_are_orthonormal():
    L = 6
    x, w = np.polynomial.legendre.leggauss(40)
    theta = np.arccos(x)
    nphi = 64
    phi = np.arange(nphi) * 2 * np.pi / nphi
    Y = real_harmonics(L, theta[:, None], phi[None, :])
    assert Y.shape == (2, L + 1, L + 1, 40, nphi)
    W = w[:, None] * (2 * np.pi / nphi)
    G = np.einsum("almtp,bkntp,tp->almbkn", Y, Y, W)
    mask = np.zeros((2, L + 1, L + 1), bool)
    for l in range(L + 1):
        mask[0, l, :l + 1] = True
        if l:
            mask[1, l, 1:l + 1] = True
    Gm = G[mask][:, mask]
    assert np.max(np.abs(Gm - np.eye(Gm.shape[0]))) < 1e-12
    assert np.all(Y[~mask] == 0.0)
    # Y_00 = 1 / sqrt(4 pi); Y_10 = sqrt(3 / 4 pi) cos theta; no CS phase in Y_11
    assert np.allclose(Y[0, 0, 0], 1 / np.sqrt(4 * np.pi))
    assert np.allclose(Y[0, 1, 0], np.sqrt(3 / (4 * np.pi)) * np.cos(theta)[:, None])
    assert np.allclose(Y[0, 1, 1], np.sqrt(3 / (4 * np.pi)) * np.sin(theta)[:, None]
                       * np.cos(phi)[None, :])
    c = np.zeros((2, L + 1, L + 1))
    c[1, 3, 2] = 2.0
    assert np.allclose(synthesise(c, theta[:, None], phi[None, :]), 2.0 * Y[1, 3, 2])
    cc = np.broadcast_to(c[..., None, None], Y.shape)
    assert np.allclose(synthesise(cc, theta[:, None], phi[None, :]), 2.0 * Y[1, 3, 2])
    with pytest.raises(ValueError):
        synthesise(c[0], theta, phi)
    with pytest.raises(ValueError):
        real_harmonics(-1, 0.0, 0.0)


def test_spherical_grf():
    g = SphericalGRF(0.7, 1.0, 2.0, 0.15, sigma=1.3, lmax=12)
    assert g.nmodes.size == 13 and np.all(g.nmodes >= 1)
    assert np.array_equal(g.degrees, np.arange(13))
    v = g.sample(rng=3)
    assert v.shape == (g.size,) == (g.basis.size,)
    c = g.synthesise(v)
    assert c.shape == (2, 13, 13, g.r.size)
    assert np.all(c[1, :, 0] == 0.0)                 # no sine at m = 0
    l_idx = np.arange(13)
    tri = np.arange(13)[None, :] > l_idx[:, None]
    assert np.all(c[:, tri] == 0.0)                  # m <= l only
    assert g.factor(2).shape == (g.r.size, g.nmodes[2])

    # the variance at the pole is sigma^2 in law: u(pole) = sum_l c[0, l, 0] Y_l0(0)
    yl0 = np.sqrt((2 * l_idx + 1) / (4.0 * np.pi))
    rng = np.random.default_rng(7)
    j = g.r.size // 2
    vals = np.array([g.synthesise(g.sample(rng=rng))[0, :, 0, j] @ yl0
                     for _ in range(2500)])
    assert abs(vals.var() / 1.3 ** 2 - 1.0) < 0.15
    assert np.allclose(g.variance(), 1.3 ** 2) and np.allclose(g.std(), 1.3)
    # and exactly, as the sum over the degrees of their factors
    var = sum((2 * l + 1) / (4.0 * np.pi) * np.sum(g.factor(l) ** 2, axis=1)
              for l in g.degrees)
    np.testing.assert_allclose(var, 1.3 ** 2, rtol=1e-12)

    ball = SphericalGRF(0.0, 0.8, 2.0, 0.15, lmax=6)
    cb = ball.synthesise(ball.sample(rng=5))
    assert ball.r[0] == 0.0
    assert np.all(cb[:, 1:, :, 0] == 0.0)            # l >= 1 vanish at the centre
    assert cb[0, 0, 0, 0] != 0.0 and np.all(np.isfinite(cb))

    auto = SphericalGRF(0.7, 1.0, 2.0, 0.15, tol=1e-4)
    assert 2 < auto.lmax < 60                        # the automatic degree cut bites
    with pytest.raises(ValueError):
        SphericalGRF(0.7, 1.0, 2.0, 0.15, lmax=-1)


def test_a_shell_sample_is_the_coefficient_vector():
    g = SphericalGRF(0.6, 1.0, 1.5, 0.2, sigma=lambda r: 1.0 + r, lmax=5, lam_h=0.3)
    c = g.sample(rng=11)
    assert c.shape == (g.size,) == (g.basis.size,)
    np.testing.assert_array_equal(c, g.sample(rng=11))
    assert not np.array_equal(c, g.sample(rng=12))
    np.testing.assert_array_equal(g.coefficient_std, g.basis.theta ** -g.beta)
    for held in (g.sigma, g.scale, g.coefficient_std):
        assert not held.flags.writeable
    testing.check_spherical_basis(g.basis)
    # independent standard normals times theta^-beta, each normal of the
    # draw once: degree by degree, and within a degree mode by mode across
    # its harmonics, the cosines of orders 0 ... l and then the sines
    z = np.random.default_rng(11).standard_normal(g.size)
    at = 0
    for l, n in enumerate(g.nmodes):
        normals = z[at:at + (2 * l + 1) * n].reshape(n, 2 * l + 1)
        harmonics = [(0, m) for m in range(l + 1)] + [(1, m) for m in range(1, l + 1)]
        for h, (s, m) in enumerate(harmonics):
            np.testing.assert_allclose(
                c[g.basis.block(s, l, m)],
                normals[:, h] * g.basis[l].theta ** -g.beta, rtol=1e-14)
        at += (2 * l + 1) * n
    assert at == g.size


def test_the_views_of_a_shell_vector_agree():
    g = SphericalGRF(0.7, 1.0, 2.0, 0.15, sigma=lambda r: 1.0 + r, lmax=8, lam_h=0.4)
    c = g.sample(rng=3)
    u = g.synthesise(c)
    np.testing.assert_allclose(u, g.basis.synthesise(c) * g.scale, rtol=1e-14)
    grid = gauss_legendre(8)
    values = g.on_grid(c, grid)
    assert values.shape == (g.r.size, grid.ntheta, grid.nphi)
    np.testing.assert_array_equal(values, synthesise_grid(u, grid))
    np.testing.assert_array_equal(g.sample_grid(grid, rng=3), values)
    # a grid of a larger band holds the field; one of a smaller band is refused
    wide = gauss_legendre(12)
    assert g.on_grid(c, wide).shape == (g.r.size, wide.ntheta, wide.nphi)
    with pytest.raises(ValueError, match="band"):
        g.on_grid(c, gauss_legendre(6))
    # the field: a harmonic field on the interval, the values at the nodes
    field = g.to_field(c, name="dv")
    assert isinstance(field, HarmonicField) and field.name == "dv"
    assert field.interval == g.interval == (0.7, 1.0) and field.lmax == 8
    size = np.abs(values).max()
    np.testing.assert_allclose(field.on_grid(g.r, grid), values, rtol=0.0,
                               atol=1e-12 * size)
    np.testing.assert_allclose(
        field.evaluate(g.r[4], grid.colatitudes[:, None], grid.longitudes[None, :]),
        values[4], rtol=0.0, atol=1e-10 * size)
    # one block alone is one harmonic's coefficient function
    block = g.basis.block(1, 3, 2)
    one = np.zeros(g.size)
    one[block] = c[block]
    alone = g.synthesise(one)
    others = np.ones(alone.shape[:3], dtype=bool)
    others[1, 3, 2] = False
    np.testing.assert_array_equal(alone[others], 0.0)
    np.testing.assert_allclose(alone[1, 3, 2], u[1, 3, 2], rtol=1e-13,
                               atol=1e-16 * size)
    np.testing.assert_allclose(alone[1, 3, 2], g.factor(3) @ (c[block]
                               / g.coefficient_std[block]), rtol=0.0,
                               atol=1e-13 * size)


def test_shell_as_a_field():
    g = SphericalGRF(0.7, 1.0, 2.0, 0.15, sigma=1.0, lmax=8, lam_h=0.4)
    v = g.sample(rng=3)
    c = g.synthesise(v)
    pp = g.coefficient_functions(v)
    j = 3
    assert np.max(np.abs(pp(g.r[j]) - c[..., j])) < 1e-12
    field = g.to_field(v, name="dv")
    testing.check_field(field)
    assert isinstance(field, HarmonicField) and field.lmax == 8
    assert field.name == "dv" and field.interval == (0.7, 1.0)
    r = np.array([0.75, 0.9])
    theta = np.array([0.3, 1.2])
    phi = np.array([0.1, 2.0])
    got = field.evaluate(r, theta, phi)
    want = [synthesise(pp(ri), ti, pi) for ri, ti, pi in zip(r, theta, phi)]
    assert np.allclose(got, want)
    # the field's variance over the sphere at a node radius is sigma^2 by quadrature
    x, w = np.polynomial.legendre.leggauss(24)
    th = np.arccos(x)
    ph = np.arange(48) * 2 * np.pi / 48
    u = field.evaluate(g.r[j], th[:, None], ph[None, :])
    mean_sq = np.sum(u ** 2 * w[:, None]) * (2 * np.pi / 48) / (4 * np.pi)
    assert mean_sq == pytest.approx(float(np.sum(c[..., j] ** 2)) / (4 * np.pi))
    with pytest.raises(ValueError):
        g.coefficient_functions(v[:2])


def test_shell_sample_statistics_agree_with_the_marginal():
    """The sample variance over draws and the sphere at each node against
    sigma^2, to five standard errors of independent draws."""
    sig = lambda r: 1.0 + r
    g = SphericalGRF(0.6, 1.0, 1.5, 0.15, sigma=sig, lmax=6, tol=1e-4)
    grid = gauss_legendre(6)
    draws = 200
    values = g.sample_grid(grid, rng=7, size=draws)          # draws, nodes, grid
    assert values.shape == (draws, g.r.size, grid.ntheta, grid.nphi)
    weight = grid.weights[:, None] * (2.0 * np.pi / grid.nphi) / (4.0 * np.pi)
    var = np.mean(np.sum(values ** 2 * weight, axis=(2, 3)), axis=0)
    np.testing.assert_allclose(var, sig(g.r) ** 2, rtol=5.0 * np.sqrt(2.0 / draws))


# -- the Matern covariance of a shell --------------------------------------------

NU, LENGTH = 2.5, 0.08


@pytest.fixture(scope="module")
def shell_grf():
    """A field of length scale 0.08 in the shell from 0.5 to 1, the reach
    sqrt(8 nu) = 4.5 length scales: degrees to 64, elements one length
    scale deep of seven nodes."""
    return SphericalGRF(0.5, 1.0, NU, LENGTH, lmax=64, ngll=7, drmax=0.08)


@pytest.mark.parametrize("step", [
    (1.0, 0.0, 0.0),            # along the radius
    (0.0, 1.0, 0.0),            # along a meridian
    (0.0, 0.0, 1.0),            # along a parallel
    (0.6, 0.48, 0.64),          # oblique
])
@pytest.mark.parametrize("start", [
    (0.7, 1.1, 0.4),            # inside
    (0.52, 2.0, 5.0),           # by the inner boundary
    (1.0, 0.3, 1.0),            # on the surface
])
def test_shell_covariance_is_matern_of_the_separation_alone(shell_grf, start, step):
    """The exact covariance of the basis between two points against the
    d = 3 Matern function of the distance between them, to most of a
    reach.

    Measured, as a fraction of the variance: 1.7e-5 by the inner
    boundary, 5.5e-5 inside and 2.7e-4 on the surface.  The degree cut
    limits it: 6.0e-3 at lmax = 32 and 1.1e-3 at 48, the radial mesh
    unchanged.
    """
    a = np.asarray(start)
    inward = -1.0 if a[0] == 1.0 else 1.0
    s = np.linspace(0.0, 0.8 * np.sqrt(8.0 * NU) * LENGTH, 9)
    move = np.asarray(step) * [inward, 1.0 / a[0], 1.0 / (a[0] * np.sin(a[1]))]
    b = a + s[:, None] * move
    distance = np.linalg.norm(cartesian(b) - cartesian(a), axis=-1)
    C = shell_grf.covariance(a, b)
    assert C.shape == (s.size,)
    np.testing.assert_allclose(C, matern(distance, NU, LENGTH), rtol=0.0, atol=5e-4)
    assert abs(C[0] - 1.0) < 5e-6


def test_horizontal_scale_stretches_the_separation_in_a_shell():
    """A distinct horizontal scale: near a point the covariance is the
    Matern function of sqrt((radial / lam)^2 + (lateral / lam_h)^2), the
    lateral separation the chord at the mean radius, to the curvature of
    the shell over a horizontal length scale.  Measured 1.7e-3."""
    lam, lam_h = 0.04, 0.08
    g = SphericalGRF(0.5, 1.0, NU, lam, lam_h=lam_h, lmax=64, ngll=7, drmax=0.08)
    a = np.array([0.75, 1.1, 0.4])
    s = np.linspace(0.0, 0.15, 7)
    for radial, lateral in ((1.0, 0.0), (0.0, 1.0), (0.6, 0.8)):
        b = a + s[:, None] * np.array([radial, lateral / a[0], 0.0])
        chord = 2.0 * np.sqrt(a[0] * b[:, 0]) * np.sin(0.5 * (b[:, 1] - a[1]))
        stretched = np.hypot((b[:, 0] - a[0]) / lam, chord / lam_h)
        np.testing.assert_allclose(g.covariance(a, b), matern(stretched, NU, 1.0),
                                   rtol=0.0, atol=5e-3)


def test_shell_covariance_is_the_sum_over_degrees():
    """Between nodes of the radial mesh the covariance is the sum over
    degrees of (2 l + 1) / (4 pi) P_l(cos gamma) times the products of
    the degree's factor at the two radii, gamma the angle between the
    points: the addition theorem, with scipy's Legendre polynomials."""
    sig = lambda r: 1.0 + r
    g = SphericalGRF(0.6, 1.0, 1.5, 0.2, sigma=sig, lmax=6, lam_h=0.3)
    nodes = np.array([0, 3, g.r.size // 2, g.r.size - 1])
    a = np.stack([g.r[nodes], [0.0, 0.7, 1.9, np.pi], [0.0, 0.3, 4.0, 1.0]], axis=-1)
    b = np.stack([g.r[nodes[::-1]], [0.4, 2.2, 1.9, 0.1], [6.0, 0.3, 4.0, 2.0]],
                 axis=-1)
    C = g.covariance(a, b)
    assert C.shape == (4, 4)
    unit = lambda points: cartesian(points * [0.0, 1.0, 1.0] + [1.0, 0.0, 0.0])
    cosine = np.clip(unit(a) @ unit(b).T, -1.0, 1.0)
    want = np.zeros((4, 4))
    for l in g.degrees:
        B = g.factor(l)
        want += ((2 * l + 1) / (4.0 * np.pi) * eval_legendre(l, cosine)
                 * (B[nodes] @ B[nodes[::-1]].T))
    np.testing.assert_allclose(C, want, rtol=0.0, atol=1e-12 * sig(1.0) ** 2)
    # symmetric, and on the diagonal the marginal variance at the nodes
    np.testing.assert_allclose(g.covariance(b, a), C.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(np.diag(g.covariance(a, a)), sig(g.r[nodes]) ** 2,
                               rtol=1e-12)
    # the shapes of the points are kept
    assert g.covariance(a.reshape(2, 2, 3), b[0]).shape == (2, 2)
    with pytest.raises(ValueError, match="physical interval"):
        g.covariance(a, b + [0.5, 0.0, 0.0])
    with pytest.raises(ValueError, match="3 coordinates"):
        g.covariance(a[:, :2], b)
    with pytest.raises(ValueError, match="colatitudes"):
        g.covariance(a, b + [0.0, 4.0, 0.0])


def test_shell_covariance_is_that_of_the_fields():
    """Between points off the nodes, the covariance against the sum over
    the coefficients of their variance times the field of each one alone
    (`to_field` of a unit vector, evaluated through ducc0) at the two
    points.  Measured 5e-14."""
    g = SphericalGRF(0.6, 1.0, 1.5, 0.2, sigma=lambda r: 1.0 + r, lmax=3, lam_h=0.3,
                     tol=1e-2)
    points = np.array([[0.63, 0.4, 1.0], [0.8123, 2.0, 4.0], [1.0, 3.0, 0.2],
                       [0.6, 0.0, 0.0]])
    modes = np.array([field.evaluate(*points.T)
                      for field in g.to_field(np.eye(g.size))])
    want = (modes * g.coefficient_std[:, None] ** 2).T @ modes
    np.testing.assert_allclose(g.covariance(points, points), want, rtol=0.0,
                               atol=1e-11)


def test_layered_grf_as_model_fields():
    sk = Skeleton([0.0, 0.4, 0.7, 1.0])
    g = LayeredGRF(sk, 1.5, [0.10, 0.05, 0.03], sigma=[1.0, 2.0, 0.5], name="dv")
    assert g.layers == (0, 1, 2) and g.size == sum(g[i].size for i in range(3))
    assert g.block(0) == slice(0, g[0].size)
    assert g.block(2) == slice(g[0].size + g[1].size, g.size)
    c = g.sample(rng=11)
    assert c.shape == (g.size,)
    np.testing.assert_array_equal(c, g.sample(rng=11))
    rng = np.random.default_rng(11)
    for i in range(3):
        np.testing.assert_array_equal(c[g.block(i)], g[i].sample(rng=rng))
    fields = g.to_fields(c)
    assert len(fields) == 3 and all(f.name == "dv" for f in fields)
    for i, f in enumerate(fields):
        assert f.interval == sk.interval(i)
        testing.check_field(f)
        np.testing.assert_allclose(f(g[i].r), g[i].synthesise(c[g.block(i)]),
                                   rtol=0.0, atol=1e-12)
    assert fields[1](0.7) != fields[2](0.7)          # generic discontinuity
    assert np.isfinite(fields[0](0.0))               # the ball layer at the centre

    part = LayeredGRF(sk, 1.0, 0.05, layers=(1,))
    assert part.size == part[1].size and part.block(1) == slice(0, part.size)
    fp = part.to_fields(part.sample(rng=2))
    assert fp[0](0.2) == 0.0 and fp[2](0.9) == 0.0 and fp[1](0.55) != 0.0
    assert part[1].nmodes >= 1
    with pytest.raises(KeyError):
        part[0]
    with pytest.raises(KeyError):
        part.block(0)
    with pytest.raises(IndexError):
        LayeredGRF(sk, 1.0, 0.05, layers=(5,))
    with pytest.raises(ValueError):
        LayeredGRF(sk, 1.0, [0.05, 0.05])
    with pytest.raises(TypeError):
        LayeredGRF([0.0, 1.0], 1.0, 0.05)

    model = PREM(ocean=False)
    mantle = [i for i in range(model.nlayers) if model.layer(i).interval[0] >= 3480e3]
    layered = LayeredGRF(model, 1.5, 300e3, sigma=0.02, layers=mantle, name="delta")
    delta = layered.to_fields(layered.sample(rng=1))
    perturbed = model
    for i in mantle:
        rho = model.layer(i)["rho"]
        perturbed = perturbed.with_field(i, "rho", rho + rho * delta[i], replace=True)
    testing.check_model(perturbed)
    r = 5.0e6
    i = model.skeleton.locate(r).layer
    ratio = perturbed.layer(i)["rho"](r) / model.layer(i)["rho"](r) - 1.0
    assert ratio == pytest.approx(delta[i](r)) and abs(ratio) < 0.2
    assert perturbed.layer(0)["rho"](1e6) == model.layer(0)["rho"](1e6)


def test_a_seeded_draw_survives_a_change_of_mesh():
    """A seeded draw is the same field on two meshes differing by a few
    elements when the truncation is unchanged: the eigenfunctions keep
    their signs, so the draws differ only by the discretisation (measured
    at 2.3e-4 of the largest value for the radial field and 2.8e-4 for
    the shell, relative)."""
    r = np.linspace(0.5, 1.0, 50)
    g1 = RadialGRF(0.5, 1.0, 1.5, 0.08, drmax=0.02, tol=1e-4)
    g2 = RadialGRF(0.5, 1.0, 1.5, 0.08, drmax=0.0196, tol=1e-4)
    assert g1.mesh.nspec != g2.mesh.nspec and g1.nmodes == g2.nmodes
    u1, u2 = (g.to_layer(g.sample(rng=3))(r) for g in (g1, g2))
    assert np.abs(u1 - u2).max() < 1e-3 * np.abs(u1).max()

    s1 = SphericalGRF(0.5, 1.0, 1.5, 0.15, lmax=6, drmax=0.02, tol=1e-8)
    s2 = SphericalGRF(0.5, 1.0, 1.5, 0.15, lmax=6, drmax=0.0196, tol=1e-8)
    assert s1.mesh.nspec != s2.mesh.nspec and np.array_equal(s1.nmodes, s2.nmodes)
    c1, c2 = (s.coefficient_functions(s.sample(rng=5))(r) for s in (s1, s2))
    assert np.abs(c1 - c2).max() < 1e-3 * np.abs(c1).max()
