"""The horizontal grid and the flat lateral rules.

A grid is the nodes of its domain's transform; its mean is exact on its
band.  Each rule interpolates the nodes, is periodic along every axis,
converges at its order (bilinear 2, bicubic 4, measured and stated in
`cartesian.lateral`), the Fourier rule exactly on a field of the grid's
band; the gradients are the rules differentiated; the mean is the
interpolant's own, against a quadrature of it; a rectangle's grid has one
axis; what a grid cannot carry is refused by name.
"""
import numpy as np
import pytest

from planetmodel import cartesian as flat
from planetmodel.cartesian.lateral import (METHODS, LateralInterpolant, check_rule,
                                           interpolate)
from planetmodel.fourier import PaddedDomain, Transform

BOX = flat.Box([(0.0, 3.0), (-1.0, 1.0)])
RECT = flat.Rectangle((0.0, 3.0))


def band8(domain, shape=17, seed=11):
    """A Fourier function of band 8 along every axis of the domain, its
    coefficients N(0, 1) / (1 + |k|)."""
    g = flat.Grid(domain, shape)
    t = g.transform
    k = np.sqrt((t.wavevectors.astype(float) ** 2).sum(axis=0))
    c = np.random.default_rng(seed).standard_normal(t.size) / (1.0 + k)
    return flat.FourierFunction(g, c)


# -- the grid ------------------------------------------------------------------

def test_a_grid_is_the_nodes_of_its_domain_s_transform():
    g = flat.Grid(BOX, (8, 6))
    t = Transform((8, 6), BOX)
    assert g.shape == (8, 6) and g.dimension == 2 and g.domain is BOX
    assert np.array_equal(g.x, t.grid_axes[0]) and np.array_equal(g.y, t.grid_axes[1])
    assert g.x[0] == 0.0 and np.isclose(g.x[-1], 3.0 - 3.0 / 8)
    assert g.lengths == (3.0, 2.0) and g.spacing == (0.375, 1.0 / 3.0)
    assert g.area == 6.0 and g.transform.shape == (8, 6)
    x, y = g.nodes
    assert np.broadcast(x, y).shape == (8, 6)
    assert np.array_equal(g.points(x, y), t.grid_points)
    assert not g.x.flags.writeable
    # one source: the grid's nodes are the padded domain's
    padded = flat.Box([(0.0, 3.0), (-1.0, 1.0)], padding=0.5)
    gp = flat.Grid(padded, 10)
    assert gp.shape == (10, 10)
    assert np.array_equal(gp.x, PaddedDomain.of(padded).grid_axes((10, 10))[0])
    assert gp.x[0] == -0.5 and gp.lengths == (4.0, 3.0)
    assert g == flat.Grid(BOX, [8, 6]) and hash(g) == hash(flat.Grid(BOX, (8, 6)))
    assert g != flat.Grid(BOX, (8, 7)) and g != gp
    assert repr(g) == "Grid(8 x 6 over Box(((0, 3), (-1, 1))))"
    big = g.scaled(2.0)
    assert big.shape == g.shape and big.lengths == (6.0, 4.0)


def test_a_rectangle_s_grid_has_one_axis():
    g = flat.Grid(RECT, 12)
    assert g.shape == (12,) and g.dimension == 1 and g.y is None
    x, y = g.nodes
    assert np.array_equal(x, g.x) and np.all(y == 0.0)
    assert g.points(g.x, 5.0).shape == (12, 1)
    assert g.area == 3.0


def test_a_grid_mean_is_exact_on_its_band():
    for domain, shape in ((BOX, (9, 8)), (RECT, (10,))):
        g = flat.Grid(domain, shape)
        t = g.transform
        c = np.random.default_rng(1).standard_normal(t.size)
        values = t.synthesise_grid(c)
        assert np.isclose(g.mean(values), c[0] / np.sqrt(g.area), rtol=0, atol=1e-14)
        stacked = np.stack([values, 2.0 * values])
        assert g.mean(stacked).shape == (2,)
    with pytest.raises(ValueError, match="trailing shape"):
        flat.Grid(BOX, 4).mean(np.zeros((4, 5)))


def test_a_grid_refuses_what_it_cannot_be():
    with pytest.raises(TypeError, match="Box or a Rectangle"):
        flat.Grid(PaddedDomain([(0, 1), (0, 1)]), 4)
    with pytest.raises(ValueError, match="2 sizes, one per axis"):
        flat.Grid(BOX, (4, 4, 4))
    with pytest.raises(ValueError, match="1 size, one per axis"):
        flat.Grid(RECT, (4, 4))
    with pytest.raises(ValueError, match="at least one node"):
        flat.Grid(BOX, (4, 0))


# -- the rules ------------------------------------------------------------------

@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "rectangle"])
def test_every_rule_interpolates_the_nodes_and_is_periodic(domain, method):
    f = band8(domain)
    g = flat.Grid(domain, 24)
    v = f.on_grid(g)
    rule = LateralInterpolant(v, g, method=method)
    x, y = g.nodes
    assert np.allclose(rule(x, y), v, rtol=0, atol=1e-11 * np.abs(v).max())
    rng = np.random.default_rng(2)
    px, py = rng.uniform(0, 3, 50), rng.uniform(-1, 1, 50)
    here = rule(px, py)
    for sx, sy in ((3.0, 0.0), (-6.0, 2.0), (0.0, -4.0)):
        shifted = rule(px + sx, py + sy) if domain is BOX else rule(px + sx, py)
        assert np.allclose(shifted, here, rtol=0, atol=1e-12 * np.abs(v).max())
    if domain is RECT:
        assert np.array_equal(rule(px, 7.0 + py), here)


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "rectangle"])
@pytest.mark.parametrize("method, order", [("linear", 2), ("cubic", 4)])
def test_local_rules_converge_at_their_orders(domain, method, order):
    """The largest error at 4000 random points over the field's largest
    value, on grids of 32 to 256 nodes a side.  Measured (box): bilinear
    1.6e-1, 4.5e-2, 1.1e-2, 3.0e-3; bicubic 8.1e-3, 4.0e-4, 2.3e-5,
    1.5e-6; fitted orders 1.9 and 4.1, as `cartesian.lateral` states."""
    f = band8(domain)
    rng = np.random.default_rng(20261115)
    x, y = rng.uniform(0, 3, 4000), rng.uniform(-1, 1, 4000)
    ref = f(x, y)
    scale = np.abs(ref).max()
    n = np.array([32, 64, 128, 256])
    err = [np.abs(interpolate(f.on_grid(g), g, x, y, method=method) - ref).max() / scale
           for g in (flat.Grid(domain, int(m)) for m in n)]
    rate = -np.polyfit(np.log(n), np.log(err), 1)[0]
    assert abs(rate - order) < 0.25, (rate, err)
    if domain is BOX:
        bounds = {"linear": (0.2, 3.5e-3), "cubic": (1e-2, 2e-6)}[method]
        assert err[0] < bounds[0] and err[-1] < bounds[1], err


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "rectangle"])
def test_the_fourier_rule_is_exact_on_the_grid_s_band(domain):
    """Exact to finufft's accuracy once the grid holds the field's band (17
    nodes a side and more); on 16 the sines of wavenumber 8 are lost."""
    f = band8(domain)
    rng = np.random.default_rng(3)
    x, y = rng.uniform(-5, 5, 2000), rng.uniform(-5, 5, 2000)
    ref = f(x, y)
    scale = np.abs(ref).max()
    for n in (17, 18, 40):
        g = flat.Grid(domain, n)
        got = interpolate(f.on_grid(g), g, x, y, method="fourier")
        assert np.abs(got - ref).max() < 1e-11 * scale
    g = flat.Grid(domain, 16)
    assert np.abs(interpolate(f.on_grid(g), g, x, y, method="fourier")
                  - ref).max() > 1e-3 * scale


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "rectangle"])
@pytest.mark.parametrize("method", METHODS)
def test_the_gradient_is_the_rule_differentiated(domain, method):
    """Against central differences of the rule (away from the nodes'
    lines for the bilinear one, whose gradient is piecewise constant), and
    for the smooth rules against the series' own gradient."""
    f = band8(domain, seed=5)
    g = flat.Grid(domain, 64)
    rule = LateralInterpolant(f.on_grid(g), g, method=method)
    rng = np.random.default_rng(6)
    x, y = rng.uniform(0, 3, 300), rng.uniform(-1, 1, 300)
    gx, gy = rule.gradient(x, y)
    h = 1e-6
    ok = np.ones(x.shape, dtype=bool)
    if method == "linear":
        for axis, v in zip(g.axes, (x, y)):
            ok &= np.abs(v[:, None] - axis[None]).min(axis=1) > 10 * h
    fx = (rule(x + h, y) - rule(x - h, y)) / (2 * h)
    fy = (rule(x, y + h) - rule(x, y - h)) / (2 * h)
    scale = np.abs(fx).max()
    assert np.allclose(gx[ok], fx[ok], rtol=0, atol=1e-6 * scale)
    assert np.allclose(gy[ok], fy[ok], rtol=0, atol=1e-6 * scale)
    if domain is RECT:
        assert np.all(gy == 0.0)
    if method != "linear":
        ex, ey = f.lateral_gradient(x, y)
        # the bicubic rule's gradient converges as h^3: 2.4e-3 here
        tol = 1e-9 if method == "fourier" else 5e-3
        assert np.abs(gx - ex).max() < tol * np.abs(ex).max()
        if domain is BOX:
            assert np.abs(gy - ey).max() < tol * np.abs(ey).max()


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "rectangle"])
@pytest.mark.parametrize("method", METHODS)
def test_the_mean_is_the_interpolant_s_own(domain, method):
    """The mean of the values is the mean over the period of the rule's
    interpolant: against a fine trapezoid of the interpolant itself, on a
    grid of 7 by 5 nodes whose rules depart well from the field."""
    f = band8(domain, seed=7)
    g = flat.Grid(domain, (7, 5) if domain is BOX else (7,))
    v = np.stack([f.on_grid(g), 1.0 + f.on_grid(g) ** 2])
    rule = LateralInterpolant(v, g, method=method)
    fine = flat.Grid(domain, (7 * 40, 5 * 40) if domain is BOX else (7 * 40,))
    quadrature = fine.mean(rule(*fine.nodes))
    assert rule.mean().shape == (2,)
    assert np.allclose(rule.mean(), quadrature, rtol=0, atol=1e-12 * np.abs(v).max())
    assert np.allclose(rule.mean(), g.mean(v), rtol=0, atol=1e-15 * np.abs(v).max())


def test_the_leading_shape_is_carried_through():
    f = band8(BOX)
    g = flat.Grid(BOX, (12, 10))
    v = f.on_grid(g)
    stack = np.stack([v * k for k in range(1, 7)]).reshape(3, 2, *v.shape)
    rule = LateralInterpolant(stack, g, method="cubic")
    assert rule.shape == (3, 2) and rule.method == "cubic" and rule.grid is g
    out = rule(np.full((4, 5), 0.7), 0.2)
    assert out.shape == (3, 2, 4, 5)
    one = interpolate(v, g, 0.7, 0.2, method="cubic")
    assert one.shape == () and np.allclose(out[1, 0], 3.0 * one)
    dx, dy = rule.gradient(0.7, np.zeros(3))
    assert dx.shape == dy.shape == (3, 2, 3)
    assert "cubic" in repr(rule)


def test_refusals_say_what_is_wrong():
    g = flat.Grid(BOX, (6, 4))
    v = np.ones((6, 4))
    with pytest.raises(ValueError, match="method must be one of"):
        LateralInterpolant(v, g, method="harmonic")
    with pytest.raises(ValueError, match="at least 3 nodes along every axis"):
        check_rule(flat.Grid(BOX, (6, 2)), "cubic")
    check_rule(flat.Grid(BOX, (6, 2)), "linear")
    with pytest.raises(TypeError, match="cartesian.Grid"):
        LateralInterpolant(v, (6, 4), method="linear")
    with pytest.raises(ValueError, match="trailing shape"):
        LateralInterpolant(np.ones((4, 6)), g, method="linear")
    with pytest.raises(TypeError, match="real"):
        LateralInterpolant(v + 0j, g, method="linear")
    with pytest.raises(ValueError, match="finite"):
        LateralInterpolant(v * np.nan, g, method="linear")
    with pytest.raises(ValueError, match="finite"):
        LateralInterpolant(v, g, method="linear")(np.inf, 0.0)
