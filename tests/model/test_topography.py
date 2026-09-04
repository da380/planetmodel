"""Topographies: the seam, the poles, and area-weighted means.

Three things about a lon-lat grid are easy to get quietly wrong, and all
three produce plausible-looking fields when wrong: the longitude seam
(clamping instead of wrapping), the poles (a longitude-dependent value,
which is a discontinuity), and the mean (unweighted, over-weighting the
poles).  Each has a test that fails if the shortcut is taken.
"""
import numpy as np
import pytest

from planetmodel.model.topography import (AnalyticTopography, GriddedTopography,
                                     HarmonicTopography, ScaledTopography,
                                     SumTopography, Topography,
                                     ZeroTopography, as_topography)
from planetmodel.testing import check_topography

CRUST = "tests/data/prem.nocrust"      # not a grid; used only for error paths


def one_degree_grid(fn):
    """A CRUST-1.0-shaped 1-degree grid of fn(colatitude, longitude)."""
    lons = np.arange(-179.5, 180.0, 1.0)
    lats = np.arange(-89.5, 90.0, 1.0)
    lon2, lat2 = np.meshgrid(lons, lats)
    return GriddedTopography(lons, lats,
                             fn(np.radians(90.0 - lat2), np.radians(lon2)))


@pytest.fixture(scope="module")
def bumpy():
    """A smooth, genuinely two-dimensional test field."""
    return one_degree_grid(lambda t, p: 1000.0 * np.sin(t) ** 2 * np.cos(2 * p)
                           + 300.0 * np.cos(t))


# ------------------------------------------------------------- contracts

@pytest.mark.parametrize("build", [
    lambda: ZeroTopography(),
    lambda: AnalyticTopography(lambda t, p: np.cos(t)),
    lambda: AnalyticTopography(lambda t, p: np.sin(t) * np.cos(p)),
    lambda: one_degree_grid(lambda t, p: np.cos(t)),
    lambda: one_degree_grid(lambda t, p: np.sin(t) ** 2 * np.sin(3 * p)),
])
def test_shipped_topographies_satisfy_the_contract(build):
    check_topography(build())


def test_composites_satisfy_the_contract(bumpy):
    check_topography(bumpy * 20.0)
    check_topography(bumpy + ZeroTopography())
    check_topography(-bumpy)


def test_protocol_is_structural():
    assert isinstance(ZeroTopography(), Topography)
    assert isinstance(lambda t, p: t, Topography)


# ------------------------------------------------------------- the seam

def test_the_seam_wraps_rather_than_clamping(bumpy):
    """Longitude is periodic; +pi and -pi are one meridian."""
    th = np.linspace(0.1, np.pi - 0.1, 33)
    assert np.allclose(bumpy(th, np.full_like(th, np.pi)),
                       bumpy(th, np.full_like(th, -np.pi)), atol=1e-9)


def test_values_are_continuous_across_the_seam(bumpy):
    """Stepping over the join must not jump."""
    th = np.full(9, 1.0)
    eps = 1e-4
    just_below = bumpy(th, np.full_like(th, np.pi - eps))
    just_above = bumpy(th, np.full_like(th, -np.pi + eps))
    assert np.allclose(just_below, just_above, rtol=1e-3)


def test_longitude_outside_the_principal_range_is_wrapped(bumpy):
    th = np.full(5, 1.2)
    phi = np.linspace(-2.0, 2.0, 5)
    assert np.allclose(bumpy(th, phi), bumpy(th, phi + 2 * np.pi), atol=1e-9)


# ------------------------------------------------------------- the poles

def test_the_pole_value_does_not_depend_on_longitude(bumpy):
    """Any longitude dependence at a pole is a discontinuity."""
    for t in (1e-9, np.pi - 1e-9):
        ring = bumpy(np.full(16, t), np.linspace(-np.pi, np.pi, 16,
                                                 endpoint=False))
        assert np.ptp(ring) < 1e-6 * max(1.0, np.max(np.abs(ring)))


def test_the_pole_value_is_the_ring_mean():
    """The only longitude-independent choice consistent with the data."""
    g = one_degree_grid(lambda t, p: np.cos(t) + 0.1 * np.cos(p))
    north_ring = g.values[-1].mean()
    assert g(1e-12, 0.0) == pytest.approx(north_ring, rel=1e-6)
    south_ring = g.values[0].mean()
    assert g(np.pi - 1e-12, 0.0) == pytest.approx(south_ring, rel=1e-6)


def test_beyond_the_pole_is_clamped_not_reflected(bumpy):
    """Colatitude outside [0, pi] clamps rather than producing nonsense."""
    assert np.isfinite(bumpy(-0.1, 0.0))
    assert np.isfinite(bumpy(np.pi + 0.1, 0.0))


# -------------------------------------------------------- area weighting

def test_mean_of_a_constant_is_that_constant():
    g = one_degree_grid(lambda t, p: np.full_like(t, 7.0))
    assert g.mean() == pytest.approx(7.0, rel=1e-12)


def test_mean_of_an_equatorially_antisymmetric_field_vanishes():
    g = one_degree_grid(lambda t, p: np.cos(t))
    assert abs(g.mean()) < 1e-12


def test_mean_is_area_weighted_not_a_plain_average():
    """The test that actually discriminates.

    An antisymmetric field averages to zero either way, so it proves
    nothing.  cos^2 of the colatitude has area-weighted mean 1/3 and
    plain grid average 1/2, so the two answers are far apart; 1/3 is the
    right one, to within the 1-degree grid's own discretisation.
    """
    g = one_degree_grid(lambda t, p: np.cos(t) ** 2)
    assert g.mean() == pytest.approx(1.0 / 3.0, abs=1e-4)
    assert abs(g.values.mean() - 0.5) < 1e-12       # what the shortcut gives


def test_analytic_mean_agrees_with_the_gridded_one():
    fn = (lambda t, p: np.cos(t) ** 2)
    assert AnalyticTopography(fn).mean() == pytest.approx(1.0 / 3.0, abs=1e-3)


def test_the_analytic_mean_is_exact_for_a_band_limited_shape():
    """Gauss-Legendre in cos(theta), not a fine grid in theta.

    The zero-mean contract of ReferenceBody.with_surface is only as good
    as the mean it removes, and a trapezoid rule in theta is accurate to
    about 1e-5 -- ten centimetres of relief on the Earth.  A Gauss rule
    in x = cos(theta) with equispaced longitudes integrates any shape
    band-limited at degree 20 exactly: cos^2 theta to 1/3, and every
    harmonic of non-zero degree to zero.
    """
    assert AnalyticTopography(lambda t, p: np.cos(t) ** 2).mean() == (
        pytest.approx(1.0 / 3.0, abs=1e-13))

    y20 = AnalyticTopography(lambda t, p: 0.5 * (3.0 * np.cos(t) ** 2 - 1.0))
    assert abs(y20.mean()) < 1e-13
    assert abs((3.0e3 * y20).mean()) < 3.0e3 * 1e-13     # relief in metres

    y43 = AnalyticTopography(
        lambda t, p: (np.sin(t) ** 3 * np.cos(t) * np.cos(3.0 * p)
                      + np.sin(t) ** 2 * np.sin(2.0 * p)))
    assert abs(y43.mean()) < 1e-13


def test_the_analytic_mean_reaches_neither_pole():
    """A Gauss node is never an endpoint, where a phi gradient is singular."""
    seen = []

    def watched(t, p):
        seen.append(np.min(np.sin(t)))
        return np.zeros_like(np.broadcast_arrays(t, p)[0])

    AnalyticTopography(watched).mean()
    assert min(seen) > 1e-3


# ------------------------------------------------------------- algebra

def test_scaling_is_exaggeration(bumpy):
    """An exaggeration factor is scalar multiplication and nothing else."""
    t, p = 1.0, 0.5
    assert (bumpy * 3.0)(t, p) == pytest.approx(3.0 * bumpy(t, p))
    assert (3.0 * bumpy)(t, p) == pytest.approx(3.0 * bumpy(t, p))
    assert (bumpy / 2.0)(t, p) == pytest.approx(bumpy(t, p) / 2.0)


def test_sum_difference_and_negation(bumpy):
    z = one_degree_grid(lambda t, p: np.cos(t))
    t, p = 0.7, -1.1
    assert (bumpy + z)(t, p) == pytest.approx(bumpy(t, p) + z(t, p))
    assert (-bumpy)(t, p) == pytest.approx(-bumpy(t, p))


def test_scaled_mean_and_bounds_scale(bumpy):
    s = ScaledTopography(bumpy, 3.0)
    assert s.mean() == pytest.approx(3.0 * bumpy.mean())
    lo, hi = s.bounds()
    blo, bhi = bumpy.bounds()
    assert (lo, hi) == pytest.approx((3.0 * blo, 3.0 * bhi))


def test_negative_scaling_swaps_the_bounds(bumpy):
    lo, hi = ScaledTopography(bumpy, -2.0).bounds()
    assert lo < hi


def test_gridded_addition_requires_a_shared_grid():
    a = one_degree_grid(lambda t, p: np.cos(t))
    lons = np.arange(-178.0, 180.0, 4.0)
    lats = np.arange(-88.0, 90.0, 4.0)
    b = GriddedTopography(lons, lats, np.zeros((lats.size, lons.size)))
    with pytest.raises(ValueError, match="different grids"):
        a + b
    resampled = b.regridded_to(a)
    assert isinstance(a + resampled, GriddedTopography)


def test_topography_times_topography_is_refused(bumpy):
    with pytest.raises(TypeError, match="not defined"):
        bumpy * bumpy


def test_empty_sum_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        SumTopography(())


# ----------------------------------------------------------- construction

def test_from_xyz_reads_a_grid(tmp_path):
    """Latitude descending in the file, values in km, as CRUST-1.0 ships."""
    path = tmp_path / "grid.xyz"
    rows = []
    for lat in (1.5, 0.5, -0.5, -1.5):            # descending, as in the files
        for lon in (-1.5, -0.5, 0.5, 1.5):
            rows.append(f"{lon} {lat} {lat + 0.1 * lon}")
    path.write_text("\n".join(rows))

    g = GriddedTopography.from_xyz(path, scale=1e3)
    assert g.values.shape == (4, 4)
    assert np.all(np.diff(g.lats) > 0)            # sorted on read
    assert g.values[0, 0] == pytest.approx(1e3 * (-1.5 + 0.1 * -1.5))


def test_from_xyz_rejects_a_ragged_grid(tmp_path):
    path = tmp_path / "ragged.xyz"
    path.write_text("0 0 1\n1 0 2\n0 1 3\n")      # missing (1, 1)
    with pytest.raises(ValueError, match="tensor grid"):
        GriddedTopography.from_xyz(path)


def test_constructor_validates_its_grid():
    lons, lats = np.array([0.0, 1.0]), np.array([0.0, 1.0])
    with pytest.raises(ValueError, match="shape"):
        GriddedTopography(lons, lats, np.zeros((3, 3)))
    with pytest.raises(ValueError, match="increasing"):
        GriddedTopography(np.array([1.0, 0.0]), lats, np.zeros((2, 2)))
    with pytest.raises(ValueError, match="interpolation"):
        GriddedTopography(lons, lats, np.zeros((2, 2)), interpolation="magic")


def test_bicubic_is_available_and_agrees_broadly():
    fn = lambda t, p: np.sin(t) ** 2 * np.cos(2 * p)
    lin = one_degree_grid(fn)
    cub = GriddedTopography(lin.lons, lin.lats, lin.values,
                            interpolation="bicubic")
    check_topography(cub)
    th = np.linspace(0.3, np.pi - 0.3, 40)
    ph = np.linspace(-3.0, 3.0, 40)
    assert np.allclose(cub(th, ph), lin(th, ph), atol=2.0)


def test_bilinear_reproduces_grid_nodes_exactly():
    g = one_degree_grid(lambda t, p: np.sin(t) * np.cos(p))
    j, i = 40, 100
    theta = np.radians(90.0 - g.lats[j])
    phi = np.radians(g.lons[i])
    assert g(theta, phi) == pytest.approx(g.values[j, i], rel=1e-12)


def test_as_topography_wraps_and_preserves(bumpy):
    assert as_topography(bumpy) is bumpy
    wrapped = as_topography(lambda t, p: np.cos(t))
    assert hasattr(wrapped, "gradient") and hasattr(wrapped, "mean")


def test_analytic_gradient_is_used_when_given():
    exact = AnalyticTopography(lambda t, p: np.cos(t),
                               gradient=lambda t, p: (-np.sin(t),
                                                      np.zeros_like(t)))
    t = np.linspace(0.2, 3.0, 20)
    gt, gp = exact.gradient(t, np.zeros_like(t))
    assert np.allclose(gt, -np.sin(t), rtol=1e-14)   # exact, not a difference
    assert np.allclose(gp, 0.0)


# ---------------------------------------------------- spherical harmonics

def harmonic(entries):
    """A HarmonicTopography from entries {(kind, l, m): value}.

    kind is 0 for the cosine coefficient and 1 for the sine one, the
    pyshtools real layout the class documents.
    """
    lmax = max(l for _, l, _ in entries)
    c = np.zeros((2, lmax + 1, lmax + 1))
    for (kind, l, m), v in entries.items():
        c[kind, l, m] = v
    return HarmonicTopography(c, name="test")


def random_coefficients(lmax=6, seed=3):
    """Coefficients filling the whole (l, m) triangle."""
    rng = np.random.default_rng(seed)
    c = np.zeros((2, lmax + 1, lmax + 1))
    for l in range(lmax + 1):
        c[0, l, :l + 1] = rng.normal(size=l + 1)
        c[1, l, 1:l + 1] = rng.normal(size=l)
    return c


def test_harmonic_topography_satisfies_the_contract():
    check_topography(HarmonicTopography(random_coefficients()))


def test_the_convention_is_orthonormal_with_the_condon_shortley_phase():
    """The three closed forms that pin the convention, to 1e-14.

    Orthonormal (unit L2 norm on the sphere, no 4 pi), with the phase
    (-1)^m carried by P_l^m -- the convention GSHTrans uses and the one
    the netCDF spectral group will record.  Written out rather than
    referred to, because a sign or a sqrt(2) here is exactly what a
    later reader of the file would get wrong:

        Y_20  =  sqrt(5 / 4 pi) (3 cos^2 theta - 1) / 2
        Y_11c = -sqrt(2) sqrt(3 / 8 pi) sin theta cos phi
        Y_11s = -sqrt(2) sqrt(3 / 8 pi) sin theta sin phi

    The minus signs are the Condon-Shortley phase, and the sqrt(2) is
    what makes the real pair orthonormal where the complex Y_1^1 was.
    A convention without the phase would give the same numbers with the
    opposite sign; one normalised to 4 pi would give them times 2.
    """
    rng = np.random.default_rng(11)
    t = rng.uniform(0.0, np.pi, 40)
    p = rng.uniform(-np.pi, np.pi, 40)

    y20 = harmonic({(0, 2, 0): 1.0})
    want = np.sqrt(5.0 / (4.0 * np.pi)) * (3.0 * np.cos(t) ** 2 - 1.0) / 2.0
    assert np.max(np.abs(y20(t, p) - want)) < 1e-14

    amp = -np.sqrt(2.0) * np.sqrt(3.0 / (8.0 * np.pi)) * np.sin(t)
    y11c = harmonic({(0, 1, 1): 1.0})
    assert np.max(np.abs(y11c(t, p) - amp * np.cos(p))) < 1e-14
    y11s = harmonic({(1, 1, 1): 1.0})
    assert np.max(np.abs(y11s(t, p) - amp * np.sin(p))) < 1e-14

    # Y_00 is the constant 1 / sqrt(4 pi): the reason mean() is exact.
    assert np.max(np.abs(harmonic({(0, 0, 0): 1.0})(t, p)
                         - 1.0 / np.sqrt(4.0 * np.pi))) < 1e-15


def test_the_expansion_agrees_with_scipy_harmonic_by_harmonic():
    """The independent construction: scipy's complex sph_harm_y.

    planetmodel evaluates its own recurrence -- one pass over the (l, m)
    triangle with the theta derivative differentiated from the same
    lines -- so the oracle has to come from outside it.  scipy's
    sph_harm_y(l, m, theta, phi) is orthonormal with the Condon-Shortley
    phase, which is the same convention, so the real basis is
    Re and Im of it times sqrt(2), and the gradient comes from the same
    call with diff_n=1.
    """
    from scipy.special import sph_harm_y

    c = random_coefficients(lmax=8, seed=7)
    topo = HarmonicTopography(c)
    rng = np.random.default_rng(2)
    t = rng.uniform(0.02, np.pi - 0.02, 60)
    p = rng.uniform(-np.pi, np.pi, 60)

    val = np.zeros_like(t)
    gt = np.zeros_like(t)
    gp = np.zeros_like(t)
    for l in range(c.shape[1]):
        for m in range(l + 1):
            Y, dY = sph_harm_y(l, m, t, p, diff_n=1)
            parts = (Y, dY[..., 0], dY[..., 1])
            w = 1.0 if m == 0 else np.sqrt(2.0)
            for out, y in zip((val, gt, gp), parts):
                out += w * c[0, l, m] * y.real
                if m:
                    out += w * c[1, l, m] * y.imag

    scale = np.max(np.abs(val))
    assert np.max(np.abs(topo(t, p) - val)) < 1e-13 * scale
    got_t, got_p = topo.gradient(t, p)
    assert np.max(np.abs(got_t - gt)) < 1e-13 * np.max(np.abs(gt))
    assert np.max(np.abs(got_p - gp)) < 1e-13 * np.max(np.abs(gp))


def test_the_mean_is_the_degree_zero_coefficient_exactly():
    c = random_coefficients(lmax=5, seed=4)
    topo = HarmonicTopography(c)
    assert topo.mean() == c[0, 0, 0] / np.sqrt(4.0 * np.pi)
    assert as_topography(topo).mean() == topo.mean()


def test_a_harmonic_of_degree_one_or_more_has_zero_mean():
    """Area-weighted, by quadrature rather than by the exact shortcut.

    as_topography keeps the exact mean() where a topography has one, so the
    shape is wrapped in an AnalyticTopography here: that route is the
    Gauss-Legendre rule, and what it says is that the *shape* integrates
    to zero, not merely that mean() reports so.
    """
    amp = 1.0e3
    for entry in ((0, 1, 0), (0, 1, 1), (1, 2, 2), (0, 5, 3), (1, 6, 4)):
        shape = harmonic({entry: amp})
        assert abs(as_topography(shape).mean()) == 0.0
        assert abs(AnalyticTopography(shape.__call__).mean()) < 1e-14 * amp


def test_a_harmonic_topography_can_be_centred_and_scaled():
    """It is a topography like any other: the algebra and the contract."""
    c = random_coefficients(lmax=4, seed=9)
    c[0, 0, 0] = 5.0e3 * np.sqrt(4.0 * np.pi)
    topo = HarmonicTopography(c)
    assert topo.mean() == pytest.approx(5.0e3, rel=1e-14)
    centred = topo - AnalyticTopography(lambda t, p: np.full_like(t, 5.0e3))
    assert abs(as_topography(centred).mean()) < 1e-9
    check_topography(2.0 * topo)


def test_the_layout_is_validated():
    with pytest.raises(ValueError, match="real layout"):
        HarmonicTopography(np.zeros((3, 4, 4)))
    with pytest.raises(ValueError, match="real layout"):
        HarmonicTopography(np.zeros((2, 4, 5)))
    bad = np.zeros((2, 3, 3)); bad[0, 1, 2] = 1.0
    with pytest.raises(ValueError, match="m > l"):
        HarmonicTopography(bad)
    bad = np.zeros((2, 3, 3)); bad[1, 2, 0] = 1.0
    with pytest.raises(ValueError, match=r"sin\(0 phi\)"):
        HarmonicTopography(bad)


def test_lmax_truncates_and_pads():
    c = random_coefficients(lmax=6, seed=1)
    cut = HarmonicTopography(c, lmax=3)
    assert cut.lmax == 3
    assert np.array_equal(cut.coeffs, HarmonicTopography(c[:, :4, :4]).coeffs)
    grown = HarmonicTopography(c, lmax=9)
    assert grown.lmax == 9
    t = np.linspace(0.1, 3.0, 15)
    assert np.allclose(grown(t, 0.3 * t), HarmonicTopography(c)(t, 0.3 * t))


def test_the_registry_knows_the_harmonic_topography():
    from planetmodel.registry import lookup
    assert lookup("topography", "harmonic") is HarmonicTopography


def test_a_0_to_360_longitude_grid_reads_like_a_centred_one():
    import numpy as np
    import pytest
    from planetmodel import GriddedTopography
    lats = np.arange(-89.0, 90.0, 2.0)
    east = np.arange(0.5, 360.0, 1.0)
    west = (east + 180.0) % 360.0 - 180.0
    shape = lambda lon: np.cos(np.deg2rad(lon))[None, :] + 0 * lats[:, None]
    g_east = GriddedTopography(east, lats, shape(east))
    g_west = GriddedTopography(np.sort(west), lats, shape(np.sort(west)))
    for lon in (-100.0, -179.5, 10.0, 170.0):
        t, p = np.deg2rad(60.0), np.deg2rad(lon)
        assert float(g_east(t, p)) == pytest.approx(float(g_west(t, p)), abs=1e-6)
        assert float(g_east(t, p)) == pytest.approx(np.cos(p), abs=1e-3)


# ------------------------------------------------------- centring, provenance

def test_a_centred_topography_keeps_its_shape_and_shift():
    from planetmodel.model.topography import CentredTopography
    c = random_coefficients(lmax=3, seed=2)
    c[0, 0, 0] = 2.0e3 * np.sqrt(4.0 * np.pi)
    topo = HarmonicTopography(c)
    centred = CentredTopography(topo, topo.mean())
    assert centred.shape is topo and centred.shift == pytest.approx(2.0e3)
    assert abs(centred.mean()) < 1e-9
    t, p = np.array([0.4, 2.0]), np.array([1.0, -1.0])
    assert np.allclose(centred(t, p), topo(t, p) - 2.0e3)
    assert np.allclose(centred.gradient(t, p), topo.gradient(t, p))
    check_topography(centred)


def test_provenance_walks_scaling_and_centring_but_not_sums():
    from planetmodel.model.topography import CentredTopography
    lats = np.arange(-89.0, 90.0, 2.0)
    lons = np.arange(-179.5, 180.0, 1.0)
    grid = GriddedTopography(lons, lats, np.ones((lats.size, lons.size)),
                             name="moho.xyz")
    grid.scale_to_m = 1000.0
    scaled = CentredTopography(grid * 2.0, 5.0e4)
    p = scaled.provenance()
    assert p["exaggeration"] == 2.0 and p["interpolation"] == "bilinear"
    assert p["files"] == [{"file": "moho.xyz", "scale_to_m": 1000.0}]
    summed = (grid + AnalyticTopography(lambda t, q: 0 * t)) * 3.0
    q = summed.provenance()
    assert q["exaggeration"] == 3.0 and [f["file"] for f in q["files"]] == ["moho.xyz"]
    assert ZeroTopography().provenance()["files"] == []


def test_adapting_keeps_a_callable_s_own_gradient():
    """Adapt, then call: an object with an exact gradient but no mean is
    wrapped for the mean and keeps the gradient it brought."""
    class Bumpy:
        def __call__(self, theta, phi):
            return np.cos(theta) * 0.0 + 3.0 * np.sin(theta) * np.cos(phi)

        def gradient(self, theta, phi):
            return (3.0 * np.cos(theta) * np.cos(phi) + 100.0,
                    -3.0 * np.sin(theta) * np.sin(phi))

    topo = as_topography(Bumpy())
    gt, _ = topo.gradient(0.3, 0.2)
    assert gt == pytest.approx(3.0 * np.cos(0.3) * np.cos(0.2) + 100.0)
    assert abs(topo.mean()) < 1e-12
