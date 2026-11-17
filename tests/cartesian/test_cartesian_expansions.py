"""Fields and functions given by Fourier series over a box or a rectangle.

A Fourier field passes the field contract for ranks 0 to 2 over a box and
over a rectangle; its values are the sum over the basis point by point (the
direct sum the oracle, finufft to its accuracy), exact in depth on its
pieces, on a grid by the fast transform or by synthesis alike, by depth or
by term alike; its mean is its constant term, exact; it is rescaled as a
field is.  A Fourier function's gradient is the series differentiated,
against differences and the closed form; it is a relief whose
displacement, stretch and geometry pass their contracts, over a box and a
rectangle.
"""
import numpy as np
import pytest
from scipy.interpolate import BSpline, PPoly

from planetmodel import DENSITY, SCALAR, STRESS, VECTOR, Skeleton, cartesian as flat
from planetmodel.cartesian import FLAT
from planetmodel.cartesian.mapping import VerticalStretch
from planetmodel.character import Character
from planetmodel.testing import (check_field, check_geometry, check_mapping,
                                 check_vertical_displacement)

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
RECT = flat.Rectangle((0.0, 2.0))
INTERVAL = (0.2, 1.0)
TENSOR = Character(2, 0, voigt=False)


def stored(char):
    return char.voigt_shape or char.component_shape


def random_field(grid, char=SCALAR, *, seed=0, breaks=(0.2, 0.5, 1.0), degree=2):
    """A Fourier field whose coefficients are random polynomials of depth on
    each piece, falling as 1 / (1 + |k|)."""
    t = grid.transform
    k = np.sqrt((t.wavevectors.astype(float) ** 2).sum(axis=0))
    shape = stored(char)
    c = np.random.default_rng(seed).standard_normal(
        (degree + 1, len(breaks) - 1, t.size) + shape)
    c /= (1.0 + k).reshape((1, 1, -1) + (1,) * len(shape))
    return flat.FourierField(INTERVAL, grid, PPoly(c, np.asarray(breaks, dtype=float)),
                             character=char, name="q")


GRIDS = [flat.Grid(BOX, (8, 6)), flat.Grid(BOX, (7, 5)), flat.Grid(RECT, 9)]


@pytest.mark.parametrize("grid", GRIDS, ids=repr)
@pytest.mark.parametrize("char", [SCALAR, VECTOR, STRESS, TENSOR], ids=str)
def test_the_field_contract_for_ranks_0_to_2(grid, char):
    f = random_field(grid, char)
    check_field(f)
    assert not f.is_radial and f.lateral == "fourier" and f.radial == "polynomial"
    assert f.coordinates is FLAT and f.grid is grid
    assert np.array_equal(f.radial_pieces()[0], [0.2, 0.5, 1.0])
    assert f.radial_pieces()[1] == 2


@pytest.mark.parametrize("grid", GRIDS, ids=repr)
def test_values_are_the_sum_over_the_basis_point_by_point(grid):
    """The direct sum over the basis at each point's depth, through the
    transform's own direct route: finufft to its accuracy."""
    f = random_field(grid, VECTOR)
    rng = np.random.default_rng(1)
    d = rng.uniform(0.2, 1.0, 300)
    x, y = rng.uniform(-4, 4, 300), rng.uniform(-4, 4, 300)
    got = f.evaluate(d, x, y)
    c = f.coefficients_at(d)                                  # (N, 3, P)
    basis = grid.transform.basis(grid.points(x, y))           # (P, N)
    want = np.einsum("pn,nip->pi", basis, c)
    assert got.shape == (300, 3)
    assert np.abs(got - want).max() < 1e-11 * np.abs(want).max()
    # the Cartesian components are the local ones permuted: one frame
    cart = f.evaluate(d, x, y, frame="cartesian")
    assert np.array_equal(cart, got[:, [1, 2, 0]])
    if grid.dimension == 1:
        assert np.array_equal(f.evaluate(d, x, 3.0 + y), got)


def test_by_depth_and_by_term_are_the_coefficients_there():
    """A few depths a piece take a synthesis a depth, many a synthesis a
    term: the two agree, and are exact in depth on the polynomial pieces."""
    g = flat.Grid(BOX, (8, 6))
    f = random_field(g, SCALAR, degree=3)
    rng = np.random.default_rng(2)
    x, y = rng.uniform(0, 2, 40), rng.uniform(-1, 1, 40)
    few = np.full(40, 0.31)
    many = rng.uniform(0.21, 0.49, 40)
    basis = g.transform.basis(g.points(x, y))                 # (P, N)
    for d in (few, many):
        want = np.einsum("pn,np->p", basis, f.coefficients_at(d))
        assert np.abs(f(d, x, y) - want).max() < 1e-11 * np.abs(want).max()
    # beyond the ends the end pieces continue, as on_interval asks of them
    wide = f.on_interval(0.0, 1.2)
    pp = f.coefficients
    out = np.array([0.05, 1.15])
    want = g.transform.synthesise(np.moveaxis(pp(out), 0, 0), g.points(0.3, 0.4))
    assert np.allclose(wide(out, 0.3, 0.4), want, rtol=0, atol=1e-12)
    with pytest.raises(ValueError, match="outside the interval"):
        f(1.1, 0.3, 0.4)


@pytest.mark.parametrize("char", [SCALAR, VECTOR], ids=str)
def test_on_grid_by_the_fast_transform_or_by_synthesis_alike(char):
    """On its own grid and on a finer grid of its domain (every mode held)
    through the fast transform; on a coarser grid or over another domain at
    the nodes; all equal to the field at the nodes."""
    g = flat.Grid(BOX, (8, 7))
    f = random_field(g, char)
    d = np.array([0.25, 0.6, 0.9])
    other = flat.Box([(0.0, 3.0), (-1.0, 1.0)])
    for target in (g, flat.Grid(BOX, (16, 9)), flat.Grid(BOX, (5, 4)),
                   flat.Grid(other, (6, 6))):
        got = f.on_grid(d, target)
        x, y = target.nodes
        want = f.evaluate(d[:, None, None], x[None], y[None])
        assert got.shape == (3,) + target.shape + stored(char)
        assert np.abs(got - want).max() < 1e-11 * np.abs(want).max()
    r = flat.Grid(RECT, 10)
    fr = random_field(r, char)
    assert np.allclose(fr.on_grid(d, flat.Grid(RECT, 21)),
                       fr.evaluate(d[:, None], flat.Grid(RECT, 21).x[None], 0.0),
                       rtol=0, atol=1e-11)


@pytest.mark.parametrize("grid", GRIDS, ids=repr)
@pytest.mark.parametrize("char", [SCALAR, VECTOR, STRESS], ids=str)
def test_the_horizontal_mean_is_the_constant_term_exactly(grid, char):
    """c_0 / sqrt(A): exact polynomial pieces of depth, of any rank, equal
    to the grid's quadrature of the field at every depth (exact for the
    band) and to that of a grid four times finer."""
    f = random_field(grid, char)
    m = flat.horizontal_mean(f)
    assert isinstance(m, flat.DepthField) and m.character == char and m.name == "q"
    d = np.linspace(0.2, 1.0, 9)
    for g in (grid, flat.Grid(grid.domain, tuple(4 * n for n in grid.shape))):
        v = f.on_grid(d, g)
        lead = 1 + g.dimension
        quad = g.mean(np.moveaxis(v, tuple(range(1, lead)),
                                  tuple(range(-g.dimension, 0))))
        assert np.allclose(m.evaluate(d, 0.0, 0.0), quad, rtol=0,
                           atol=1e-12 * np.abs(v).max())
    assert np.array_equal(m.radial_pieces()[0], f.radial_pieces()[0])


def test_rescaled_keeps_the_values_at_the_scaled_points():
    g = flat.Grid(BOX, (6, 5))
    f = random_field(g, VECTOR)
    s = f.rescaled(k=3.0, v=0.5)
    assert np.allclose(s.interval, (0.6, 3.0)) and s.grid == g.scaled(3.0)
    rng = np.random.default_rng(4)
    d, x, y = rng.uniform(0.2, 1, 30), rng.uniform(0, 2, 30), rng.uniform(-1, 1, 30)
    assert np.allclose(s(3 * d, 3 * x, 3 * y), 0.5 * f(d, x, y), rtol=0, atol=1e-11)
    assert np.allclose(flat.horizontal_mean(s)(3 * d, 0, 0),
                       0.5 * flat.horizontal_mean(f)(d, 0, 0), rtol=0, atol=1e-12)
    assert f.renamed("p").name == "p" and f.name == "q"
    assert "FourierField('q'" in repr(f)


def test_a_b_spline_and_a_constant_array_are_taken():
    g = flat.Grid(RECT, 5)
    t = np.array([0.2, 0.2, 0.2, 0.6, 1.0, 1.0, 1.0])
    c = np.random.default_rng(5).standard_normal((4, g.transform.size))
    spline = BSpline(t, c, 2)
    f = flat.FourierField(INTERVAL, g, spline)
    d = np.linspace(0.2, 1.0, 11)
    want = g.transform.synthesise(spline(d), g.points(0.7, 0.0))
    assert np.allclose(f(d, 0.7, 0.0), want, rtol=0, atol=1e-12)
    const = flat.FourierField(INTERVAL, g, c[0], character=DENSITY)
    assert np.allclose(const(d, 0.7, 0.0), g.transform.synthesise(c[0], [0.7]))
    assert const.horizontal_mean()(0.5) == pytest.approx(c[0, 0] / np.sqrt(2.0))


def test_a_fourier_field_in_a_flat_model_validates_and_its_algebra_holds():
    model = flat.CrustOverMantle(BOX.scaled(1e5))
    iv = model.layer(1).interval
    g = flat.Grid(model.geometry.domain, (6, 6))
    t = g.transform
    c = np.zeros(t.size)
    c[0] = 3300.0 * np.sqrt(g.area)
    c[t.component_of([1, 0])] = 30.0 * np.sqrt(g.area / 2)
    rho = flat.FourierField(iv, g, c, character=DENSITY)
    m = model.with_field(1, "rho", rho, replace=True)
    m.validate()
    x = np.linspace(0.0, 2e5, 5)
    assert np.allclose(rho(iv[0] + 1.0, x, 0.0),
                       3300.0 + 30.0 * np.cos(2 * np.pi * x / 2e5), rtol=1e-12)
    assert np.allclose((3.0 * rho - rho)(iv[0] + 1, x, 0.0),
                       2.0 * rho(iv[0] + 1, x, 0.0))


def test_refusals():
    g = flat.Grid(BOX, (4, 4))
    with pytest.raises(TypeError, match="cartesian.Grid"):
        flat.FourierField(INTERVAL, (4, 4), np.zeros(16))
    with pytest.raises(ValueError, match=r"values of shape \(N,\) \+ stored shape = "
                                         r"\(16, 3\)"):
        flat.FourierField(INTERVAL, g, np.zeros(16), character=VECTOR)
    with pytest.raises(TypeError, match="real"):
        flat.FourierField(INTERVAL, g, np.zeros(16) + 1j)
    with pytest.raises(ValueError, match="16 coefficients"):
        flat.FourierFunction(g, np.zeros(15))
    with pytest.raises(TypeError, match="real"):
        flat.FourierFunction(g, np.zeros(16) + 1j)
    with pytest.raises(ValueError, match="needs depth and x and y|needs x and y"):
        random_field(g)(0.5)


# -- the function -----------------------------------------------------------------

def ridge_series(grid):
    """A range along y: a few low modes, heights of about 0.08."""
    t = grid.transform
    c = np.zeros(t.size)
    root_a = np.sqrt(grid.area)
    along = [0] * (grid.dimension - 1)
    c[0] = 0.01 * root_a
    c[t.component_of([1] + along)] = 0.05 * root_a / np.sqrt(2)
    # the sine's basis function is -sqrt(2 / A) sin
    c[t.component_of([2] + along, phase=1)] = 0.02 * root_a / np.sqrt(2)
    if grid.dimension == 2:
        c[t.component_of([1, 1])] = 0.01 * root_a / np.sqrt(2)
    return flat.FourierFunction(grid, c)


def ridge_closed(x, y, grid):
    """`ridge_series` as a formula, and its gradient."""
    Lx = grid.lengths[0]
    u = 2 * np.pi * (x - grid.x[0]) / Lx
    h = 0.01 + 0.05 * np.cos(u) - 0.02 * np.sin(2 * u)
    gx = (-0.05 * np.sin(u) - 0.04 * np.cos(2 * u)) * 2 * np.pi / Lx
    gy = 0.0 * y
    if grid.dimension == 2:
        Ly = grid.lengths[1]
        v = 2 * np.pi * (y - grid.y[0]) / Ly
        h = h + 0.01 * np.cos(u + v)
        gx = gx - 0.01 * np.sin(u + v) * 2 * np.pi / Lx
        gy = -0.01 * np.sin(u + v) * 2 * np.pi / Ly
    return h, gx, gy


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "rectangle"])
def test_a_fourier_function_and_its_gradient(domain):
    g = flat.Grid(domain, 8)
    f = ridge_series(g)
    rng = np.random.default_rng(6)
    x, y = rng.uniform(-3, 3, 200), rng.uniform(-3, 3, 200)
    h, gx, gy = ridge_closed(x, y, g)
    assert np.allclose(f(x, y), h, rtol=0, atol=1e-13)
    fx, fy = f.lateral_gradient(x, y)
    assert np.allclose(fx, gx, rtol=0, atol=1e-12) and np.allclose(fy, gy, rtol=0,
                                                                     atol=1e-12)
    step = 1e-6
    assert np.allclose(fx, (f(x + step, y) - f(x - step, y)) / (2 * step), atol=1e-8)
    assert np.allclose(fy, (f(x, y + step) - f(x, y - step)) / (2 * step), atol=1e-8)
    assert f.mean() == pytest.approx(0.01, abs=1e-15)
    assert np.allclose(f.on_grid(g), ridge_closed(*g.nodes, g)[0], rtol=0, atol=1e-15)
    k = 2.5
    big = f.scaled(k)
    assert big.grid == g.scaled(k)
    assert np.allclose(big(k * x, k * y), k * f(x, y), rtol=0, atol=1e-13)
    assert np.allclose(big.lateral_gradient(k * x, k * y), f.lateral_gradient(x, y),
                       rtol=0, atol=1e-12)
    assert repr(f).startswith("FourierFunction(on Grid(")


def test_a_fourier_relief_drives_a_stretch_whose_contracts_hold():
    """Topography from a series on the surface and a root on the Moho, the
    base held: the displacement, the stretch and the geometry contracts."""
    sk = Skeleton([0.0, 0.4, 1.0])
    g = flat.Grid(flat.Box([(0.0, 1.0), (-0.5, 0.5)]), 12)
    surface = ridge_series(g)
    moho = flat.FourierFunction(g, -0.5 * surface.coefficients)
    w = flat.layer_linear(sk, [surface, moho, None])
    check_vertical_displacement(w, sk, bounds=((0.0, 1.0), (-0.5, 0.5)))
    geometry = flat.Geometry(sk, g.domain).stretched(w)
    check_geometry(geometry)
    _, x, y = geometry.lattice(n_c=3, n_a=5, n_b=4)
    d = np.array([0.05, 0.2, 0.35, 0.45, 0.7, 0.95])[:, None, None]
    check_mapping(VerticalStretch(w, base=sk), FLAT.points(d, x, y).reshape(-1, 3))
    top = geometry.mapping(FLAT.points(0.0, g.x, 0.1))
    assert np.allclose(top[:, 2], surface(g.x, 0.1), rtol=0, atol=1e-15)


def test_a_rectangle_s_relief_is_the_same_at_every_y():
    sk = Skeleton([0.0, 0.4, 1.0])
    rect = flat.Rectangle((0.0, 1.0))
    g = flat.Grid(rect, 10)
    w = flat.layer_linear(sk, [ridge_series(g), None, None])
    geometry = flat.Geometry(sk, rect).stretched(w)
    check_geometry(geometry)


def test_import_is_light_and_finufft_waits_for_a_transform_at_points():
    """`import planetmodel.cartesian` loads neither finufft nor ducc0; a grid,
    the fast transforms and the local rules load no finufft; a series read
    at points does."""
    import subprocess
    import sys
    code = (
        "import sys, numpy as np\n"
        "import planetmodel.cartesian as flat\n"
        "assert not [m for m in ('finufft', 'ducc0') if m in sys.modules]\n"
        "g = flat.Grid(flat.Box([(0, 1), (0, 1)]), 6)\n"
        "f = flat.GriddedFunction(g, np.ones((6, 6)))\n"
        "f(0.3, 0.2); g.transform.analyse_grid(np.ones((6, 6)))\n"
        "assert 'finufft' not in sys.modules, 'loaded early'\n"
        "flat.FourierFunction(g, np.ones(36))(0.3, 0.2)\n"
        "assert 'finufft' in sys.modules and 'ducc0' not in sys.modules\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr
