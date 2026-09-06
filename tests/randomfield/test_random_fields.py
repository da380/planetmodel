"""Random fields: exact marginals, statistics, padding, the harmonics, a
shell as a field, and a layered sample as model fields."""
import numpy as np
import pytest

from planetmodel import Skeleton, PREM, testing
from planetmodel.randomfield import (LayeredGRF, RadialGRF, SphericalGRF,
                                     real_harmonics, synthesise)


def test_radial_grf_normalisation_and_stats():
    sig = lambda r: 1.0 + r
    g = RadialGRF(0.5, 1.0, 1.5, 0.08, sigma=sig, tol=1e-10)
    assert np.max(np.abs(np.sqrt((g.factor ** 2).sum(axis=1)) - sig(g.r))) < 1e-10
    assert np.max(np.abs(np.diag(g.covariance()) - sig(g.r) ** 2)) < 1e-10
    assert np.array_equal(g.std(), sig(g.r))

    u = g.sample(rng=0, size=8000)
    assert u.shape == (8000, g.r.size)
    assert np.max(np.abs(u.std(axis=0) / sig(g.r) - 1.0)) < 0.06
    assert np.max(np.abs(u.mean(axis=0)) / sig(g.r)) < 0.05

    layer = g.to_layer(u[0])
    assert np.max(np.abs(layer(g.r) - u[0])) < 1e-10
    assert layer.interval == (0.5, 1.0)
    field = g.to_field(u[0], name="dv")
    testing.check_field(field)
    assert field.name == "dv" and np.max(np.abs(field(g.r) - u[0])) < 1e-10

    ball = RadialGRF(0.0, 1.0, 1.0, 0.2, sigma=2.0)
    assert ball.r[0] == 0.0                          # the axis is a physical node
    assert np.max(np.abs(np.sqrt((ball.factor ** 2).sum(axis=1)) - 2.0)) < 1e-10
    assert np.isfinite(ball.sample(rng=4)[0])
    with pytest.raises(ValueError):
        RadialGRF(0.5, 0.4, 1.0, 0.1)
    with pytest.raises(ValueError):
        RadialGRF(0.5, 1.0, -1.0, 0.1)
    with pytest.raises(ValueError):
        g.to_layer(u[0, :3])


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
    c = g.sample(rng=3)
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
    vals = np.array([g.sample(rng=rng)[0, :, 0, j] @ yl0 for _ in range(2500)])
    assert abs(vals.var() / 1.3 ** 2 - 1.0) < 0.15
    assert np.allclose(g.variance(), 1.3 ** 2) and np.allclose(g.std(), 1.3)

    ball = SphericalGRF(0.0, 0.8, 2.0, 0.15, lmax=6)
    cb = ball.sample(rng=5)
    assert ball.r[0] == 0.0
    assert np.all(cb[:, 1:, :, 0] == 0.0)            # l >= 1 vanish at the centre
    assert cb[0, 0, 0, 0] != 0.0 and np.all(np.isfinite(cb))

    auto = SphericalGRF(0.7, 1.0, 2.0, 0.15, tol=1e-4)
    assert 2 < auto.lmax < 60                        # the automatic degree cut bites
    with pytest.raises(ValueError):
        SphericalGRF(0.7, 1.0, 2.0, 0.15, lmax=-1)


def test_shell_as_a_field():
    g = SphericalGRF(0.7, 1.0, 2.0, 0.15, sigma=1.0, lmax=8, lam_h=0.4)
    c = g.sample(rng=3)
    pp = g.coefficient_functions(c)
    j = 3
    assert np.max(np.abs(pp(g.r[j]) - c[..., j])) < 1e-12
    field = g.to_field(c, name="dv")
    testing.check_field(field)
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
    v = field.evaluate(g.r[j], th[:, None], ph[None, :])
    mean_sq = np.sum(v ** 2 * w[:, None]) * (2 * np.pi / 48) / (4 * np.pi)
    assert mean_sq == pytest.approx(float(np.sum(c[..., j] ** 2)) / (4 * np.pi))
    with pytest.raises(ValueError):
        g.coefficient_functions(c[..., :2])


def test_layered_grf_as_model_fields():
    sk = Skeleton([0.0, 0.4, 0.7, 1.0])
    g = LayeredGRF(sk, 1.5, [0.10, 0.05, 0.03], sigma=[1.0, 2.0, 0.5], name="dv")
    fields = g.sample(rng=11)
    assert len(fields) == 3 and all(f.name == "dv" for f in fields)
    for i, f in enumerate(fields):
        assert f.interval == sk.interval(i)
        testing.check_field(f)
    assert fields[1](0.7) != fields[2](0.7)          # generic discontinuity
    assert np.isfinite(fields[0](0.0))               # the ball layer at the centre

    part = LayeredGRF(sk, 1.0, 0.05, layers=(1,))
    fp = part.sample(rng=2)
    assert fp[0](0.2) == 0.0 and fp[2](0.9) == 0.0 and fp[1](0.55) != 0.0
    assert part[1].nmodes >= 1
    with pytest.raises(KeyError):
        part[0]
    with pytest.raises(IndexError):
        LayeredGRF(sk, 1.0, 0.05, layers=(5,))
    with pytest.raises(ValueError):
        LayeredGRF(sk, 1.0, [0.05, 0.05])
    with pytest.raises(TypeError):
        LayeredGRF([0.0, 1.0], 1.0, 0.05)

    model = PREM(ocean=False)
    mantle = [i for i in range(model.nlayers) if model.layer(i).interval[0] >= 3480e3]
    layered = LayeredGRF(model, 1.5, 300e3, sigma=0.02, layers=mantle, name="delta")
    delta = layered.sample(rng=1)
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
