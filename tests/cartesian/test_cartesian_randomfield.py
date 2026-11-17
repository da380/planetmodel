"""Random fields of flat layers: the layer basis and its order, members
shared between wavevectors of equal length, the exact marginal standard
deviation, the Matern covariance along every direction of a box, and
the samplers' views of a coefficient vector."""
import numpy as np
import pytest
from scipy.special import gamma, kv

from planetmodel import cartesian as flat
from planetmodel import fourier, testing
from planetmodel.cartesian.randomfield import EIGENVALUE_RTOL, distinct_eigenvalues
from planetmodel.randomfield import (SeparableOperatorFamily, SpectralBasis,
                                     matern_reach, padded_mesh, restriction)
from planetmodel.randomfield.fields import matern_exponent


def matern(r, nu, lam):
    """The Matern correlation of smoothness nu and length scale lam, the
    covariance of (1 - lam^2 Laplacian)^-(nu + d / 2) white noise over
    its variance in any dimension d: 2^(1 - nu) / Gamma(nu) x^nu K_nu(x)
    with x = r / lam."""
    x = np.asarray(r, dtype=float) / lam
    safe = np.where(x > 0.0, x, 1.0)
    return np.where(x > 0.0, 2.0 ** (1.0 - nu) / gamma(nu) * safe ** nu
                    * kv(nu, safe), 1.0)


def small_family(lam=0.7, lam_h=None, n=4):
    mesh = padded_mesh(0.0, 2.0, pad=1.0, weight="one", ngll=4, drmax=0.5)
    family = SeparableOperatorFamily(
        mesh, kappa=lam ** 2, kappa_h=None if lam_h is None else lam_h ** 2,
        weight="one", lateral="one", index="value")
    return family, restriction(mesh, 0.0, 2.0)


# -- the basis ------------------------------------------------------------------

def test_distinct_eigenvalues_are_one_float_for_each_length():
    transform = fourier.Transform((12, 12), (3.0, 3.0))
    values, member = distinct_eigenvalues(transform)
    k = transform.wavevectors
    squared = k[0] ** 2 + k[1] ** 2                      # exact integers
    assert values.size == np.unique(squared).size
    assert np.all(np.diff(values) > 0.0)
    # one index for each integer |k|^2, whatever sum reached it
    for value in np.unique(squared):
        assert np.unique(member[squared == value]).size == 1
    np.testing.assert_allclose(values[member], transform.laplacian_eigenvalues,
                               rtol=4 * EIGENVALUE_RTOL)
    # unequal periods part the lengths a square box holds equal
    more, _ = distinct_eigenvalues(fourier.Transform((12, 12), (3.0, 3.7)))
    assert more.size > values.size


def test_equal_wavenumbers_share_one_member():
    family, R = small_family()
    transform = fourier.Transform((12, 12), (3.0, 3.0))
    basis = flat.LayerBasis.of(family, transform, restrict=R, nmodes=3)
    # |k|^2 = 25 by five routes, cosines and sines
    same = [transform.component_of(k, phase=s)
            for k in ((5, 0), (0, 5), (3, 4), (4, 3), (-3, 4)) for s in (0, 1)]
    first = basis.basis_of(same[0])
    assert all(basis.basis_of(c) is first for c in same)
    assert basis.basis_of(transform.component_of((5, 1))) is not first
    # twelve wavevectors of that length, six conjugate pairs: twelve components
    assert basis.multiplicity[basis.member[same[0]]] == 12
    # one entry of the family's cache for each distinct eigenvalue, under it
    assert family.cached() == list(basis.eigenvalues)
    assert len(basis) == len({i * i + j * j for i in range(7) for j in range(7)})
    assert basis.multiplicity.sum() == transform.size


@pytest.mark.parametrize("shape, periods", [((6, 5), (2.0, 3.0)), ((8, 8), (2.0, 2.0)),
                                            ((9,), (4.0,))])
def test_layer_basis_contract_and_order(shape, periods):
    family, R = small_family(lam_h=0.4)
    transform = fourier.Transform(shape, periods)
    basis = flat.LayerBasis.of(family, transform, restrict=R, power=2.5, tol=1e-3)
    testing.check_layer_basis(basis)
    assert basis.nmodes.min() >= 1 and basis.nmodes.max() < family.mesh.nglob
    # the order as documented: the transform's packing, then ascending theta
    at = 0
    for c in range(transform.size):
        own = basis[int(basis.member[c])]
        assert basis.block(c) == slice(at, at + own.nmodes)
        assert np.all(np.diff(basis.theta[basis.block(c)]) >= 0.0)
        at += own.nmodes
    assert at == basis.size == basis.theta.size
    assert np.array_equal(basis.r, R.r)
    # the constant comes first in the packing, the member lambda = 0
    assert basis.member[0] == 0 and basis.eigenvalues[0] == 0.0


def test_analysis_and_synthesis_are_inverse_and_projection():
    family, R = small_family()
    transform = fourier.Transform((6, 4), (2.0, 3.0))
    basis = flat.LayerBasis.of(family, transform, restrict=R, nmodes=5)
    rng = np.random.default_rng(1)
    v = rng.standard_normal(basis.size)
    np.testing.assert_allclose(basis.analyse(basis.synthesise(v, physical=False)), v,
                               rtol=0.0, atol=1e-12)
    # synthesis after analysis is the M-orthogonal projection onto the kept
    # modes: idempotent, and what it removes is M-orthogonal to every mode
    u = rng.standard_normal((transform.size, family.mesh.nglob))
    P = basis.synthesise(basis.analyse(u), physical=False)
    np.testing.assert_allclose(
        basis.synthesise(basis.analyse(P), physical=False), P, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(basis.analyse(u - P), 0.0, atol=1e-12)
    assert np.abs(u - P).max() > 0.1                    # a truncation, not the identity


def test_the_variance_rule_runs_over_the_whole_basis():
    family, R = small_family(lam=0.3)
    transform = fourier.Transform((16, 16), (4.0, 4.0))
    power, tol = 3.5, 1e-3
    basis = flat.LayerBasis.of(family, transform, restrict=R, power=power, tol=tol)
    every = np.concatenate([np.repeat(family.eigvalsh(lam), m)
                            for lam, m in zip(basis.eigenvalues, basis.multiplicity)])
    total = np.sum(every ** -power)
    omitted = total - np.sum(basis.theta ** -power)
    assert 0.0 <= omitted <= tol * total
    # the kept coefficients are those of smallest theta: every member's
    # first omitted eigenvalue is at least the cut, the largest one kept
    # by a member holding more than its one mode
    several = [b.theta[-1] for b in basis if b.nmodes > 1]
    cut = max(several)
    for b in basis:
        vals = family.eigvalsh(b.degree)
        if b.nmodes < vals.size:
            assert vals[b.nmodes] >= cut * (1.0 - 1e-12)
    # short wavelengths keep fewer modes than the constant, one at least
    assert basis.nmodes[-1] < basis.nmodes[0] and basis.nmodes.min() >= 1
    full = flat.LayerBasis.of(family, transform, restrict=R)
    assert np.all(full.nmodes == family.mesh.nglob)


def test_layer_basis_refusals():
    family, R = small_family()
    transform = fourier.Transform((4, 4), (2.0, 2.0))
    values, _ = distinct_eigenvalues(transform)
    bases = [SpectralBasis(family, float(v), restrict=R, nmodes=2) for v in values]
    flat.LayerBasis(transform, bases)
    with pytest.raises(ValueError, match="distinct lateral eigenvalues"):
        flat.LayerBasis(transform, bases[:-1])
    with pytest.raises(ValueError, match="lateral eigenvalue"):
        flat.LayerBasis(transform, bases[::-1])
    with pytest.raises(ValueError, match="at most one"):
        flat.LayerBasis.of(family, transform, nmodes=2, power=2.0, tol=1e-3)
    with pytest.raises(ValueError, match="together"):
        flat.LayerBasis.of(family, transform, power=2.0)
    degree = SeparableOperatorFamily(family.mesh, weight="one")
    with pytest.raises(ValueError, match="indexed by"):
        flat.LayerBasis(fourier.Transform((1,), (1.0,)), [SpectralBasis(degree, 0)])


# -- the Matern covariance: the Fourier and the depth directions agree ----------

NU, LENGTH = 2.5, 1.0


@pytest.fixture(scope="module")
def box_grf():
    """A field of unit length scale in a box 20 across and 6 deep, the
    reach sqrt(8 nu) = 4.5: 48 nodes a side, elements one length scale
    deep of seven nodes."""
    grid = flat.Grid(flat.Box([(0.0, 20.0), (0.0, 20.0)]), (48, 48))
    return flat.BoxGRF(0.0, 6.0, grid, NU, LENGTH, ngll=7, drmax=1.0)


@pytest.mark.parametrize("direction", [
    (0.0, 1.0, 0.0),            # lateral, along x
    (0.0, 0.6, 0.8),            # lateral, across the grid
    (1.0, 0.0, 0.0),            # along depth
    (0.6, 0.48, 0.64),          # oblique
])
@pytest.mark.parametrize("start", [
    (0.0, 3.0, 7.0),            # on the free surface
    (1.0, 19.5, 0.3),           # by a corner of the period
    (1.5, 10.123, 9.8),         # inside, off the nodes
])
def test_box_covariance_is_matern_of_the_separation_alone(box_grf, start, direction):
    """The exact covariance of the basis between two points against the
    d = 3 Matern function of their separation, to a reach.

    Measured, as a fraction of the variance: along depth 1.6e-5 to
    2.1e-5, obliquely 3.0e-5, laterally 2.7e-5 to 3.5e-5 inside and
    6.7e-5 on the free surface.  The lateral band limits it: 2.2e-4 on
    32 nodes a side and 1.3e-5 on 64, the box and the depth mesh
    unchanged, while a pad of 2.5 reaches for 1.5, or elements half as
    deep, change its third digit alone; the truncation tolerance, 1e-6,
    lies below it (at 1e-4 the error is 1.5e-4).
    """
    s = np.linspace(0.0, matern_reach(NU, LENGTH), 10)
    a = np.asarray(start)
    b = a + s[:, None] * np.asarray(direction)
    C = box_grf.covariance(a, b)
    assert C.shape == (s.size,)
    np.testing.assert_allclose(C, matern(s, NU, LENGTH), rtol=0.0, atol=2e-4)
    assert abs(C[0] - 1.0) < 2e-4


def test_box_covariance_is_symmetric_and_periodic(box_grf):
    a = np.array([[1.0, 2.0, 3.0], [4.0, 5.5, 6.5]])
    b = np.array([[0.5, 2.5, 3.5], [3.0, 5.0, 7.0], [6.0, 19.0, 1.0]])
    C = box_grf.covariance(a, b)
    assert C.shape == (2, 3)
    np.testing.assert_allclose(box_grf.covariance(b, a), C.T, rtol=0.0, atol=1e-14)
    np.testing.assert_allclose(box_grf.covariance(a + [0.0, 20.0, -40.0], b), C,
                               rtol=0.0, atol=1e-12)
    with pytest.raises(ValueError, match="physical interval"):
        box_grf.covariance(a, b + [7.0, 0.0, 0.0])


def test_horizontal_scale_stretches_the_separation():
    """A distinct horizontal scale: the covariance is the Matern function
    of sqrt((depth / lam)^2 + (lateral / lam_h)^2).  Measured 3.7e-5."""
    lam, lam_h = 0.5, 1.0
    grid = flat.Grid(flat.Box([(0.0, 20.0), (0.0, 20.0)]), (48, 48))
    grf = flat.BoxGRF(0.0, 3.0, grid, NU, lam, lam_h=lam_h, ngll=7, drmax=1.0)
    a = np.array([0.0, 4.0, 4.0])
    for direction in ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.6, 0.0, 0.8)):
        u = np.asarray(direction)
        s = np.linspace(0.0, 3.0, 7)
        stretched = s * np.hypot(u[0] / lam, np.hypot(u[1], u[2]) / lam_h)
        np.testing.assert_allclose(grf.covariance(a, a + s[:, None] * u),
                                   matern(stretched, NU, 1.0), rtol=0.0, atol=2e-4)


def test_rectangle_covariance_is_matern_in_two_dimensions():
    """One lateral axis: d = 2, the same Matern function.  Measured 2.0e-5
    along x on 96 nodes, 2.0e-6 along depth."""
    grid = flat.Grid(flat.Rectangle((0.0, 20.0)), 96)
    grf = flat.BoxGRF(0.0, 6.0, grid, NU, LENGTH, ngll=7, drmax=1.0)
    assert grf.beta == matern_exponent(NU, 2) == 0.5 * (NU + 1.0)
    a = np.array([1.0, 7.3])
    s = np.linspace(0.0, 4.0, 9)
    for direction in ((0.0, 1.0), (1.0, 0.0), (0.6, 0.8)):
        C = grf.covariance(a, a + s[:, None] * np.asarray(direction))
        np.testing.assert_allclose(C, matern(s, NU, LENGTH), rtol=0.0, atol=1e-4)
    c = grf.sample(rng=2)
    field = grf.to_field(c)
    values = grf.on_grid(c)
    assert values.shape == (grf.r.size, 96)
    np.testing.assert_allclose(field.evaluate(grf.r[:, None], grid.x[None, :], 3.3),
                               values, rtol=0.0, atol=1e-10 * np.abs(values).max())


def test_depth_covariance_is_matern_in_one_dimension():
    """A field of depth alone, from above the surface down.  Measured
    2.0e-5 at nu = 1.5, the pad of 1.5 reaches limiting it."""
    grf = flat.DepthGRF(-3.0, 9.0, 1.5, 1.0)
    assert grf.beta == matern_exponent(1.5, 1) == 1.0
    assert grf.mesh.rglob[0] < -3.0 and grf.mesh.rglob[-1] > 9.0
    C = grf.covariance()
    i = grf.r.size // 2
    np.testing.assert_allclose(C[i], matern(np.abs(grf.r - grf.r[i]), 1.5, 1.0),
                               rtol=0.0, atol=1e-4)


# -- the exact marginal standard deviation ----------------------------------------

def varying(depth):
    return 0.6 + 0.15 * np.asarray(depth)


def sigma_of(depth):
    return 0.02 * (1.0 + 0.5 * np.cos(np.asarray(depth)))


@pytest.mark.parametrize("lam", [0.8, varying], ids=["constant", "varying"])
@pytest.mark.parametrize("shape", [(8, 6), (7, 5)], ids=["even", "odd"])
def test_box_marginal_standard_deviation_is_exact_at_the_nodes(lam, shape):
    grid = flat.Grid(flat.Box([(0.0, 6.0), (1.0, 5.0)]), shape)
    grf = flat.BoxGRF(0.0, 4.0, grid, 1.5, lam, sigma=sigma_of, tol=1e-4)
    want = sigma_of(grf.r)
    np.testing.assert_allclose(grf.std(), want, rtol=1e-15)
    np.testing.assert_allclose(grf.variance(), want ** 2, rtol=1e-15)
    # the sum over the coefficients, each component of grid mean square 1 / V
    var = np.zeros(grf.r.size)
    for i, count in enumerate(grf.basis.multiplicity):
        var += count * np.sum(grf.factor(i) ** 2, axis=1)
    np.testing.assert_allclose(var / grid.area, want ** 2, rtol=1e-12)
    # and from the basis functions themselves, at nodes of the lateral grid
    x, y = grid.x[[0, 3, 4]], grid.y[[1, 0, 4]]
    for j in (0, grf.r.size // 3, grf.r.size - 1):
        points = np.stack([np.full(3, grf.r[j]), x, y], axis=-1)
        np.testing.assert_allclose(np.diag(grf.covariance(points, points)),
                                   want[j] ** 2, rtol=1e-12)
    # the coefficients' own variances, through the synthesis of each one
    c = grf.sample(rng=5)
    S = grf.on_grid(c)
    assert S.shape == (grf.r.size,) + shape
    assert np.all(np.isfinite(S))


def test_sample_statistics_agree_with_the_marginal():
    """The sample variance over draws and the lateral grid at each depth
    node against sigma^2, to five standard errors of independent
    draws."""
    grid = flat.Grid(flat.Box([(0.0, 6.0), (0.0, 6.0)]), (8, 8))
    grf = flat.BoxGRF(0.0, 3.0, grid, 1.5, 0.8, sigma=2.0, tol=1e-4)
    draws = 200
    c = grf.sample(rng=7, size=draws)
    values = np.stack([grf.on_grid(v) for v in c])           # draws, depth, x, y
    var = np.mean(values ** 2, axis=(0, 2, 3))
    np.testing.assert_allclose(var, 4.0, rtol=5.0 * np.sqrt(2.0 / draws))


@pytest.mark.parametrize("lam", [0.8, varying], ids=["constant", "varying"])
def test_depth_marginal_standard_deviation_is_exact_at_the_nodes(lam):
    grf = flat.DepthGRF(0.0, 4.0, 1.5, lam, sigma=sigma_of)
    testing.check_spectral_basis(grf.basis)
    np.testing.assert_allclose(np.sqrt(np.diag(grf.covariance())), sigma_of(grf.r),
                               rtol=1e-12)
    np.testing.assert_allclose(grf.std(), sigma_of(grf.r), rtol=1e-15)
    B = grf.factor()
    assert B.shape == (grf.r.size, grf.nmodes)
    z = np.random.default_rng(0).standard_normal(grf.nmodes)
    np.testing.assert_allclose(grf.synthesise(grf.coefficient_std * z), B @ z,
                               rtol=0.0, atol=1e-14)


# -- coefficients first: a sample and its views -----------------------------------

@pytest.fixture(scope="module")
def small_grf():
    grid = flat.Grid(flat.Box([(0.0, 6.0), (1.0, 5.0)]), (8, 6))
    return flat.BoxGRF(1.0, 4.0, grid, 1.5, varying, sigma=sigma_of, tol=1e-4)


def test_a_sample_is_the_coefficient_vector(small_grf):
    grf = small_grf
    c = grf.sample(rng=11)
    assert c.shape == (grf.size,) == (grf.basis.size,)
    np.testing.assert_array_equal(c, grf.sample(rng=11))
    assert not np.array_equal(c, grf.sample(rng=12))
    # independent standard normals times theta^-beta, in the basis's order
    z = np.random.default_rng(11).standard_normal(grf.size)
    np.testing.assert_allclose(c, z * grf.basis.theta ** -grf.beta, rtol=1e-14)
    np.testing.assert_array_equal(grf.coefficient_std, grf.basis.theta ** -grf.beta)
    # a batch is successive draws of one generator
    batch = grf.sample(rng=11, size=3)
    assert batch.shape == (3, grf.size)
    rng = np.random.default_rng(11)
    for row in batch:
        np.testing.assert_array_equal(row, grf.sample(rng=rng))
    testing.check_layer_basis(grf.basis)


def test_the_views_of_a_vector_agree(small_grf):
    grf, grid = small_grf, small_grf.grid
    c = grf.sample(rng=3)
    nodal = grf.synthesise(c)
    assert nodal.shape == (grid.transform.size, grf.r.size)
    np.testing.assert_allclose(nodal, grf.basis.synthesise(c) * grf.scale, rtol=1e-14)
    values = grf.on_grid(c)
    assert values.shape == (grf.r.size,) + grid.shape
    size = np.abs(values).max()
    np.testing.assert_allclose(values, grid.transform.synthesise_grid(nodal.T),
                               rtol=0.0, atol=1e-14 * size)
    np.testing.assert_array_equal(grf.sample_grid(rng=3), values)
    # the field: a Fourier field on the interval, the values at the nodes
    field = grf.to_field(c, name="dlnvs")
    assert isinstance(field, flat.FourierField) and field.name == "dlnvs"
    assert field.interval == grf.interval == (1.0, 4.0) and field.grid is grid
    testing.check_field(field)
    np.testing.assert_allclose(field.on_grid(grf.r, grid), values, rtol=0.0,
                               atol=1e-12 * size)
    x, y = grid.nodes
    np.testing.assert_allclose(field.evaluate(grf.r[4], x + 0.0, y + 0.0), values[4],
                               rtol=0.0, atol=1e-9 * size)
    # one block alone is one component's function of depth
    k = grid.transform.component_of((1, 2), phase=1)
    one = np.zeros(grf.size)
    one[grf.basis.block(k)] = c[grf.basis.block(k)]
    alone = grf.synthesise(one)
    np.testing.assert_array_equal(alone[np.arange(alone.shape[0]) != k], 0.0)
    np.testing.assert_allclose(alone[k], nodal[k], rtol=1e-13, atol=1e-16 * size)


def test_a_transform_or_a_padded_domain_names_the_lateral_grid():
    domain = fourier.PaddedDomain([(0.0, 4.0), (0.0, 3.0)], padding=(1.0, 0.5))
    transform = fourier.Transform((8, 6), domain)
    grf = flat.BoxGRF(0.0, 2.0, transform, 1.5, 0.8, tol=1e-3)
    assert grf.grid == flat.Grid(flat.Box([(0.0, 4.0), (0.0, 3.0)], padding=(1.0, 0.5)),
                                 (8, 6))
    assert grf.transform.domain == domain and grf.transform.lengths == (6.0, 4.0)
    line = flat.BoxGRF(0.0, 2.0, fourier.Transform((8,), (5.0,)), 1.5, 0.8, tol=1e-3)
    assert isinstance(line.grid.domain, flat.Rectangle)
    with pytest.raises(NotImplementedError, match="3 axes is not built"):
        flat.BoxGRF(0.0, 2.0, fourier.Transform((4, 4, 4), (1.0, 1.0, 1.0)), 1.5, 0.8)
    with pytest.raises(TypeError, match="lateral grid"):
        flat.BoxGRF(0.0, 2.0, (8, 6), 1.5, 0.8)


def test_robin_and_refusals():
    grid = flat.Grid(flat.Box([(0.0, 6.0), (0.0, 6.0)]), (6, 6))
    free = flat.BoxGRF(0.0, 2.0, grid, 1.5, 0.8, tol=1e-3, pad_factor=0.5)
    auto = flat.BoxGRF(0.0, 2.0, grid, 1.5, 0.8, tol=1e-3, pad_factor=0.5,
                       robin="auto")
    assert np.all(auto.basis.theta[:3] > free.basis.theta[:3])
    np.testing.assert_allclose(auto.std(), 1.0, rtol=1e-15)
    with pytest.raises(ValueError, match="depth1 < depth2"):
        flat.BoxGRF(2.0, 2.0, grid, 1.5, 0.8)
    with pytest.raises(ValueError, match="nu must be positive"):
        flat.DepthGRF(0.0, 2.0, 0.0, 0.8)
    with pytest.raises(ValueError, match="lambda must be positive"):
        flat.DepthGRF(0.0, 2.0, 1.5, lambda d: d - 1.0)
    with pytest.raises(ValueError, match="sigma must be non-negative"):
        flat.DepthGRF(0.0, 2.0, 1.5, 0.8, sigma=-1.0)


def test_depth_sample_and_field():
    grf = flat.DepthGRF(0.0, 4.0, 1.5, varying, sigma=sigma_of)
    c = grf.sample(rng=4)
    assert c.shape == (grf.nmodes,) == (grf.size,)
    np.testing.assert_array_equal(c, grf.sample(rng=4))
    batch = grf.sample(rng=4, size=2)
    np.testing.assert_array_equal(batch[0], c)
    values = grf.synthesise(c)
    np.testing.assert_allclose(grf.synthesise(batch)[0], values, rtol=1e-14)
    field = grf.to_field(c, name="f")
    assert isinstance(field, flat.DepthField) and field.interval == (0.0, 4.0)
    testing.check_field(field)
    np.testing.assert_allclose(field.evaluate(grf.r, 0.3, 0.4), values, rtol=0.0,
                               atol=1e-12 * np.abs(values).max())
    np.testing.assert_allclose(grf.to_layer(c)(grf.r), values, rtol=0.0,
                               atol=1e-12 * np.abs(values).max())


# -- the layers of a model ----------------------------------------------------------

def test_layered_fields_of_depth():
    model = flat.CrustOverMantle(flat.Box([(0.0, 400e3), (0.0, 300e3)]))
    nl = model.skeleton.nlayers
    grf = flat.LayeredGRF(model, 1.5, [8e3, 20e3, 30e3][:nl], sigma=0.03,
                          layers=[0, -1], name="dlnvs")
    assert grf.layers == (0, nl - 1) and grf.grid is None
    assert isinstance(grf[0], flat.DepthGRF)
    assert grf[0].interval == model.skeleton.interval(0)
    assert grf.size == grf[0].size + grf[nl - 1].size
    assert grf.block(0) == slice(0, grf[0].size)
    assert grf.block(nl - 1) == slice(grf[0].size, grf.size)
    with pytest.raises(KeyError):
        grf[1]
    c = grf.sample(rng=9)
    np.testing.assert_array_equal(c, grf.sample(rng=9))
    rng = np.random.default_rng(9)
    np.testing.assert_array_equal(c[grf.block(0)], grf[0].sample(rng=rng))
    np.testing.assert_array_equal(c[grf.block(nl - 1)], grf[nl - 1].sample(rng=rng))
    fields = grf.to_fields(c)
    assert len(fields) == nl and all(isinstance(f, flat.DepthField) for f in fields)
    assert all(f.name == "dlnvs" for f in fields)
    assert np.all(fields[1].evaluate(np.linspace(*fields[1].interval, 5), 0.0, 0.0)
                  == 0.0)
    d = grf[0].r
    np.testing.assert_allclose(fields[0].evaluate(d, 1e3, 2e3),
                               grf[0].synthesise(c[grf.block(0)]), rtol=0.0,
                               atol=1e-12)
    for i, field in enumerate(fields):
        model = model.with_field(i, "dlnvs", field)
    testing.check_model(model)
    with pytest.raises(IndexError):
        flat.LayeredGRF(model, 1.5, 8e3, layers=[nl])
    with pytest.raises(TypeError, match="Skeleton"):
        flat.LayeredGRF(object(), 1.5, 8e3)


def test_layered_fields_over_a_grid():
    model = flat.CrustOverMantle(flat.Box([(0.0, 400e3), (0.0, 300e3)]))
    nl = model.skeleton.nlayers
    grid = flat.Grid(model.geometry.domain, (8, 6))
    grf = flat.LayeredGRF(model.geometry, 1.5, 40e3, grid=grid, sigma=0.03,
                          lam_h=80e3, layers=[1], tol=1e-3, name="dlnvs")
    assert isinstance(grf[1], flat.BoxGRF) and grf[1].grid is grid
    assert grf.size == grf[1].size and grf.block(1) == slice(0, grf.size)
    c = grf.sample(rng=2)
    fields = grf.to_fields(c)
    assert len(fields) == nl and all(isinstance(f, flat.FourierField) for f in fields)
    x, y = grid.nodes
    assert np.all(fields[0].evaluate(1e3, x + 0.0, y + 0.0) == 0.0)
    values = grf[1].on_grid(c)
    np.testing.assert_allclose(fields[1].on_grid(grf[1].r, grid), values, rtol=0.0,
                               atol=1e-12 * np.abs(values).max())
    for i, field in enumerate(fields):
        model = model.with_field(i, "dlnvs", field)
    testing.check_model(model)
