"""Real orthonormal spherical harmonics: synthesis, analysis, conventions.

The basis is real and orthonormal on the unit sphere: with theta the
colatitude, phi the longitude and P_l^m the associated Legendre
functions without the Condon-Shortley phase,

    Y_l0        = N_l0 P_l(cos theta),
    Y_lm^cos    = sqrt 2 N_lm P_l^m(cos theta) cos(m phi),   m > 0,
    Y_lm^sin    = sqrt 2 N_lm P_l^m(cos theta) sin(m phi),   m > 0,
    N_lm        = sqrt((2l + 1) / 4 pi  (l - m)! / (l + m)!),

so that a field u = sum_lm c_lm Y_lm has sum_lm c_lm^2 = int u^2 dS.
Coefficient arrays are laid out as `coeffs[0, l, m]` for the cosine
harmonic of order m and `coeffs[1, l, m]` for the sine one, zero for
m = 0 in the sine slot and for m > l.  The layout and the convention
are those of pyshtools under `normalization="ortho"` and `csphase=1`,
so a coefficient array here is a pyshtools `cilm` array as it stands.
Of its 2 (lmax + 1)^2 entries, (lmax + 1)^2 belong to a harmonic;
`packing` lists them in the one order in which they are laid out as a
vector: by degree, then the cosine harmonics of orders 0 ... l, then
the sine harmonics of orders 1 ... l.

**Through ducc0.**  Every transform is ducc0's, stable to degrees in
the tens of thousands.  `synthesise` and `synthesise_gradient` evaluate
a field and its two angular derivatives at arbitrary points through
ducc0's non-uniform synthesis, to about `POINT_EPSILON` of the field's
largest value; `real_harmonics` gives the basis itself at points,
through ducc0's Legendre transforms, to rounding; `synthesise_grid` and
`analyse_grid` go between coefficients and values on a grid of a
shipped kind -- Gauss-Legendre, Driscoll-Healy, or equiangular with or
without the poles, from `planetmodel.sampling` -- exactly for a field
band-limited to the grid's band.  The longitudes may start anywhere and
number anything from 2 lmax + 1 up, provided they are equally spaced.
ducc0 holds complex coefficients a_lm for m >= 0 in healpy's order,
with the Condon-Shortley phase; the real ones here are

    a_l0 = c[0, l, 0],    a_lm = (-1)^m (c[0, l, m] - i c[1, l, m]) / sqrt 2,

and that is the whole of the conversion.  ducc0 is imported where a
transform is made, so that `import planetmodel` does not load it.

**Many shells at once.**  A consumer holding a field on radial nodes
times a grid -- a hundred shells, one band -- crosses into ducc0 once
per shell.  `ShellTransform` instead tabulates the basis on one
Gauss-Legendre grid's meridian, once per grid, and carries every shell
together through two matrix products each way, agreeing with
`synthesise_grid` and `analyse_grid` to rounding.  Its table is
(lmax + 1)^3 numbers, so the grid transforms remain the route for a
single field, a grid of another kind, or degrees where the table would
not fit.

**Other conventions.**  Published coefficients come in other
normalisations and with or without the Condon-Shortley phase;
`from_convention` gives the library's coefficients from them and
`to_convention` the reverse, in the layout above and in pyshtools'
terms where they have one.  `normalization` is one of `NORMALIZATIONS`:

    "ortho"      the library's own: int Y^2 dOmega = 1
    "4pi"        geodesy's: int Y^2 dOmega = 4 pi, so c = sqrt(4 pi) C
    "schmidt"    Schmidt's semi-normalised: int Y^2 dOmega = 4 pi / (2l + 1)
    "unnorm"     the bare P_l^m(cos theta) cos(m phi) and sin(m phi)
    "woodhouse"  N_lm P_l^m(cos theta) cos(m phi) and sin(m phi), N_lm as
                 above with no sqrt 2: the real and imaginary parts of the
                 complex orthonormal harmonics, int Y^2 dOmega = 1/2 for m > 0

and `csphase` is 1 where the source's Legendre functions leave out the
Condon-Shortley phase (-1)^m and -1 where they include it.  With
`csphase=-1`, "woodhouse" is the form of the Woodhouse-Ritsema
coefficient files.  Entries that belong to no harmonic are left as they
are; the bare functions overflow at high degree, as the convention
does.
"""
from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import gammaln

from .frames import wrapped

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["real_harmonics", "synthesise", "synthesise_gradient", "synthesise_grid",
           "analyse_grid", "ShellTransform", "packing", "from_convention",
           "to_convention", "NORMALIZATIONS", "POINT_EPSILON"]

#: The relative accuracy asked of ducc0's synthesis at arbitrary points.
POINT_EPSILON = 1e-13

#: The normalisations of `from_convention` and `to_convention`.
NORMALIZATIONS = ("ortho", "4pi", "schmidt", "unnorm", "woodhouse")


def _coefficients(coeffs: ArrayLike) -> np.ndarray:
    """A coefficient array (2, L + 1, L + 1, ...) as floats, or the error."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, lmax + 1, lmax + 1, ...)")
    return c


def _convention_factors(lmax: int, normalization: str, csphase: int
                        ) -> np.ndarray:
    """c / C for every entry of a (2, L + 1, L + 1) array: a coefficient C
    of the convention times this is the library's; one where no harmonic
    is."""
    if normalization not in NORMALIZATIONS:
        raise ValueError(f"normalization must be one of {NORMALIZATIONS}, got "
                         f"{normalization!r}")
    if csphase not in (1, -1):
        raise ValueError(f"csphase must be 1 (without the Condon-Shortley phase) "
                         f"or -1 (with it), got {csphase!r}")
    l, m = np.meshgrid(np.arange(lmax + 1.0), np.arange(lmax + 1.0), indexing="ij")
    held = m <= l
    if normalization == "ortho":
        f = np.ones_like(l)
    elif normalization == "4pi":
        f = np.full_like(l, np.sqrt(4.0 * np.pi))
    elif normalization == "schmidt":
        f = np.sqrt(4.0 * np.pi / (2.0 * l + 1.0))
    elif normalization == "woodhouse":
        f = np.where(m == 0, 1.0, 1.0 / np.sqrt(2.0))
    else:                                           # "unnorm": 1 / (sqrt(2) N_lm)
        d = np.where(held, l - m, 0.0)
        log_n = 0.5 * (np.log((2.0 * l + 1.0) / (4.0 * np.pi))
                       + gammaln(d + 1.0) - gammaln(np.where(held, l + m, 0.0) + 1.0))
        f = np.exp(-log_n) / np.where(m == 0, 1.0, np.sqrt(2.0))
    if csphase == -1:
        f = f * (-1.0) ** m
    return np.stack([np.where(held, f, 1.0), np.where(held & (m > 0), f, 1.0)])


def from_convention(coefficients: ArrayLike, *, normalization: str,
                    csphase: int) -> np.ndarray:
    """The library's coefficients of the field whose coefficients in a
    source's convention are `coefficients`, of shape (2, L + 1, L + 1)
    followed by any extra axes: `normalization` one of `NORMALIZATIONS`,
    `csphase` 1 without the Condon-Shortley phase and -1 with it, as the
    module docstring states them."""
    c = _coefficients(coefficients)
    f = _convention_factors(c.shape[1] - 1, normalization, csphase)
    return c * f.reshape(f.shape + (1,) * (c.ndim - 3))


def to_convention(coefficients: ArrayLike, *, normalization: str,
                  csphase: int) -> np.ndarray:
    """The coefficients in a source's convention of the field whose
    library coefficients are `coefficients`: the inverse of
    `from_convention`."""
    c = _coefficients(coefficients)
    f = _convention_factors(c.shape[1] - 1, normalization, csphase)
    return c / f.reshape(f.shape + (1,) * (c.ndim - 3))


def packing(lmax: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Index arrays (s, l, m) of the (lmax + 1)^2 entries of a coefficient
    array that belong to a harmonic, in the order of a vector of them: by
    degree l, then s = 0 with m = 0 ... l, then s = 1 with m = 1 ... l.
    So `coeffs[s, l, m]` is the vector of an array, and
    `out[s, l, m] = vector` puts one back."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    s, l, m = [], [], []
    for deg in range(lmax + 1):
        s += [0] * (deg + 1) + [1] * deg
        l += [deg] * (2 * deg + 1)
        m += list(range(deg + 1)) + list(range(1, deg + 1))
    return np.array(s), np.array(l), np.array(m)


# -- through ducc0 --------------------------------------------------------------

def _sht() -> ModuleType:
    """ducc0's spherical-harmonic transforms, imported where they are used."""
    import ducc0
    return ducc0.sht


def _rings(grid: AngularGrid) -> str:
    """ducc0's name for the grid's ring layout, once its colatitudes are
    checked against its kind's own, or a ValueError naming what is wrong.
    An equiangular grid is Clenshaw-Curtis with its poles and Fejer's
    first rule without."""
    from .sampling import driscoll_healy, equiangular, gauss_legendre

    kinds = ("gauss_legendre", "driscoll_healy", "equiangular")
    if grid.kind not in kinds or grid.lmax is None:
        raise ValueError(
            f"grid synthesis and analysis take a grid of a shipped kind with a "
            f"band, one of {kinds}; got a {grid.kind!r} grid")
    nt = grid.ntheta
    if grid.kind == "gauss_legendre":
        name, own = "GL", gauss_legendre(nt - 1)
    elif grid.kind == "driscoll_healy":
        name, own = "DH", driscoll_healy(max(nt // 2 - 1, 0))
    elif grid.pole_nodes[0] and nt >= 2:
        name, own = "CC", equiangular(nt, grid.nphi, poles=True)
    else:
        name, own = "F1", equiangular(nt, grid.nphi)
    if own.ntheta != nt or not np.allclose(grid.colatitudes, own.colatitudes,
                                           rtol=0.0, atol=1e-12):
        raise ValueError(
            f"the {nt} colatitudes are not those of a {grid.kind!r} grid "
            f"({name} rings in ducc0's terms)")
    if grid.lmax > own.lmax:
        raise ValueError(
            f"a {grid.kind!r} grid of {nt} colatitudes resolves degrees up to "
            f"{own.lmax}, not the {grid.lmax} it claims")
    if grid.nphi < 2 * grid.lmax + 1:
        raise ValueError(
            f"a grid of band {grid.lmax} needs at least 2 lmax + 1 = "
            f"{2 * grid.lmax + 1} longitudes to resolve its orders; this grid "
            f"has {grid.nphi}")
    if not np.allclose(np.diff(grid.longitudes), 2.0 * np.pi / grid.nphi,
                       rtol=0.0, atol=1e-12):
        raise ValueError("the longitudes are not equally spaced around the circle")
    return name


def _healpy(lmax: int) -> tuple[np.ndarray, np.ndarray]:
    """(l, m) of each complex coefficient in healpy's order: m outermost,
    then l = m ... lmax."""
    m = np.concatenate([np.full(lmax + 1 - k, k) for k in range(lmax + 1)])
    l = np.concatenate([np.arange(k, lmax + 1) for k in range(lmax + 1)])
    return l, m


def _to_complex(c: np.ndarray) -> np.ndarray:
    """ducc0's coefficients of real ones of shape (2, L + 1, L + 1, K):
    shape (K, 1, nalm), one row per trailing index."""
    l, m = _healpy(c.shape[1] - 1)
    scale = np.where(m == 0, 1.0, (-1.0) ** m / np.sqrt(2.0))[:, None]
    a = scale * (c[0, l, m] - 1j * np.where(m == 0, 0.0, 1.0)[:, None] * c[1, l, m])
    return np.ascontiguousarray(a.T[:, None, :])


def _to_real(a: np.ndarray, lmax: int) -> np.ndarray:
    """Real coefficients (2, lmax + 1, lmax + 1, K) of ducc0's (K, nalm)."""
    l, m = _healpy(lmax)
    scale = np.where(m == 0, 1.0, (-1.0) ** m * np.sqrt(2.0))[:, None]
    c = np.zeros((2, lmax + 1, lmax + 1, a.shape[0]))
    c[0, l, m] = scale * a.T.real
    c[1, l, m] = np.where(m == 0, 0.0, -1.0)[:, None] * scale * a.T.imag
    return c


def synthesise_grid(coeffs: ArrayLike, grid: AngularGrid) -> np.ndarray:
    """Values on a grid of a shipped kind from coefficients, through ducc0.

    `coeffs` has shape (2, l + 1, l + 1) or that followed by any extra
    axes, a radial axis say, with l at most the grid's band; the result
    has the extra axes first and then (ntheta, nphi), the layout of a
    sample.  Exact for a field band-limited to the grid's degree.
    """
    sht = _sht()
    rings = _rings(grid)
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, l + 1, l + 1, ...)")
    L = c.shape[1] - 1
    if L > grid.lmax:
        raise ValueError(
            f"coefficients to degree {L} on a grid of band {grid.lmax}: the grid "
            f"cannot hold them; use a grid of band at least {L}")
    extra = c.shape[3:]
    alm = _to_complex(c.reshape(c.shape[:3] + (-1,)))
    out = np.empty((alm.shape[0], grid.ntheta, grid.nphi))
    for k in range(alm.shape[0]):
        out[k] = sht.synthesis_2d(alm=alm[k], spin=0, lmax=L, geometry=rings,
                                  ntheta=grid.ntheta, nphi=grid.nphi,
                                  phi0=float(grid.longitudes[0]))[0]
    return out.reshape(extra + (grid.ntheta, grid.nphi))


def analyse_grid(values: ArrayLike, grid: AngularGrid, *,
                 lmax: int | None = None) -> np.ndarray:
    """Coefficients from values on a grid of a shipped kind, through ducc0.

    `values` has shape (..., ntheta, nphi), any extra axes first; the
    result is (2, lmax + 1, lmax + 1) followed by the extra axes, with
    `lmax` the grid's band by default or a smaller degree to stop at.
    The inverse of `synthesise_grid` for a field band-limited to the
    grid; a projection otherwise.
    """
    sht = _sht()
    rings = _rings(grid)
    band = grid.lmax
    lmax = band if lmax is None else int(lmax)
    if not 0 <= lmax <= band:
        raise ValueError(f"lmax must lie in 0..{band}, got {lmax}")
    v = np.asarray(values, dtype=float)
    if v.ndim < 2 or v.shape[-2:] != (grid.ntheta, grid.nphi):
        raise ValueError(
            f"values must have trailing shape {(grid.ntheta, grid.nphi)}, "
            f"got {v.shape}")
    extra = v.shape[:-2]
    flat = v.reshape((-1, 1) + v.shape[-2:])
    a = np.empty((flat.shape[0], (lmax + 1) * (lmax + 2) // 2), dtype=complex)
    for k in range(flat.shape[0]):
        a[k] = sht.analysis_2d(map=np.ascontiguousarray(flat[k]), spin=0,
                               lmax=lmax, geometry=rings,
                               phi0=float(grid.longitudes[0]))[0]
    return _to_real(a, lmax).reshape((2, lmax + 1, lmax + 1) + extra)


def _points_and_coefficients(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike
                             ) -> tuple[np.ndarray, int, np.ndarray,
                                        tuple[int, ...], tuple[int, ...]]:
    """ducc0's coefficients (K, 1, nalm), their degree, and the points
    (P, 2) for a synthesis at arbitrary points, with the extra and point
    shapes to restore."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, l + 1, l + 1, ...)")
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        raise ValueError("colatitudes must lie in [0, pi]")
    loc = np.ascontiguousarray(np.stack([theta.ravel(), wrapped(phi.ravel())],
                                        axis=-1))
    alm = _to_complex(c.reshape(c.shape[:3] + (-1,)))
    return alm, c.shape[1] - 1, loc, c.shape[3:], theta.shape


def synthesise(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """sum_lm c_lm Y_lm at arbitrary points, through ducc0.

    `coeffs` has shape (2, l + 1, l + 1) or that followed by any extra
    axes; `theta` in [0, pi] and `phi`, in any range, broadcast
    together.  The result has the extra axes first and then the points'
    shape, to about `POINT_EPSILON` of the field's largest value, the
    accuracy asked of ducc0's non-uniform synthesis, at a cost nearly
    independent of the number of points.
    """
    sht = _sht()
    alm, L, loc, extra, points = _points_and_coefficients(coeffs, theta, phi)
    out = np.empty((alm.shape[0], loc.shape[0]))
    if loc.shape[0]:
        for k in range(alm.shape[0]):
            out[k] = sht.synthesis_general(alm=alm[k], spin=0, lmax=L, loc=loc,
                                           epsilon=POINT_EPSILON)[0]
    return out.reshape(extra + points)


def synthesise_gradient(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike
                        ) -> tuple[np.ndarray, np.ndarray]:
    """(d/dtheta, d/dphi) of sum_lm c_lm Y_lm at arbitrary points, through
    ducc0, in the layout and to the accuracy of `synthesise`.

    ducc0 gives the gradient's components d/dtheta and (1 / sin theta)
    d/dphi; the second is multiplied by sin theta here, so that at a pole
    d/dphi is zero, as it is for any field smooth there.
    """
    sht = _sht()
    alm, L, loc, extra, points = _points_and_coefficients(coeffs, theta, phi)
    dt = np.zeros((alm.shape[0], loc.shape[0]))
    dp = np.zeros((alm.shape[0], loc.shape[0]))
    if loc.shape[0] and L > 0:
        for k in range(alm.shape[0]):
            g = sht.synthesis_general(alm=alm[k], spin=1, lmax=L, loc=loc,
                                      epsilon=POINT_EPSILON, mode="DERIV1")
            dt[k], dp[k] = g[0], g[1] * np.sin(loc[:, 0])
    return dt.reshape(extra + points), dp.reshape(extra + points)


def real_harmonics(lmax: int, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """The basis up to degree `lmax` at colatitudes `theta` in [0, pi]
    and longitudes `phi` (broadcast together): shape (2, lmax + 1,
    lmax + 1) + points, zero where no harmonic is.  The normalised
    Legendre functions come from ducc0's transform from Legendre to
    harmonic coefficients, the adjoint of the step from coefficients to
    rings: a unit on one ring gives every N_lm P_l^m there, with the
    Condon-Shortley phase, which is removed.  Once per distinct
    colatitude, so that a grid's points cost a ring each."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    if np.any(theta < 0.0) or np.any(theta > np.pi):
        raise ValueError("colatitudes must lie in [0, pi]")
    sht = _sht()
    rings, which = np.unique(theta.ravel(), return_inverse=True)
    m = np.arange(lmax + 1)
    start = (m * (2 * lmax + 1 - m)) // 2                # healpy's index of (0, m)
    l, mm = np.tril_indices(lmax + 1)
    at = start[mm] + l
    unit = np.ones((1, 1, lmax + 1), dtype=complex)
    legendre = np.zeros((rings.size, lmax + 1, lmax + 1))
    # No index arrays: ducc0's default layout, every order up to lmax in
    # healpy's order, is the one `start` indexes.
    for k, t in enumerate(rings):
        a = sht.leg2alm(leg=unit, lmax=lmax, theta=np.array([t]), spin=0)
        legendre[k, l, mm] = a[0, at].real * (-1.0) ** mm
    lam = np.moveaxis(legendre[which], 0, -1)            # (l, m, points)
    angle = m[:, None] * wrapped(phi.ravel())[None, :]   # (m, points)
    root2 = np.where(m == 0, 1.0, np.sqrt(2.0))[:, None]
    out = np.stack([lam * (root2 * np.cos(angle))[None],
                    lam * (root2 * np.sin(angle))[None]])
    out[1, :, 0] = 0.0
    return out.reshape((2, lmax + 1, lmax + 1) + theta.shape)


# -- many shells at once ----------------------------------------------------------

class ShellTransform:
    """Synthesis and analysis on one Gauss-Legendre grid, for any number
    of radial shells at once.

    The transform is separable: the basis on the prime meridian -- the
    normalised Legendre functions, tabulated once through
    `real_harmonics` -- contracts the degrees of every shell in one
    matrix product, and the longitudes in a second; nothing loops over
    shells.  Construction costs the table, (lmax + 1)^3 numbers; each
    call after that is the matmuls.

    The grid is the minimal Gauss-Legendre one of its band,
    `gauss_legendre(lmax)`: lmax + 1 colatitudes and 2 lmax + 1
    longitudes, from any phi0.  Any other grid is refused.  On it,
    `synthesise` and `analyse` agree with `synthesise_grid` and
    `analyse_grid` to rounding: exact for a field band-limited to the
    grid's degree, a projection otherwise.
    """

    def __init__(self, grid: AngularGrid) -> None:
        """Tabulate the basis on the meridian of `grid`."""
        if (getattr(grid, "kind", None) != "gauss_legendre" or grid.lmax is None
                or grid.weights is None):
            raise ValueError(
                f"ShellTransform tabulates the basis of a Gauss-Legendre grid "
                f"and integrates with its weights; got a "
                f"{getattr(grid, 'kind', None)!r} grid")
        _rings(grid)            # the nodes are genuinely the kind's own
        lmax = int(grid.lmax)
        if grid.ntheta != lmax + 1 or grid.nphi != 2 * lmax + 1:
            raise ValueError(
                f"ShellTransform takes the minimal Gauss-Legendre grid of its "
                f"band, gauss_legendre({lmax}): {lmax + 1} colatitudes and "
                f"{2 * lmax + 1} longitudes, not {grid.ntheta} x {grid.nphi}")
        self._lmax = lmax
        # The basis on the prime meridian: the cosine harmonics there are the
        # normalised Legendre functions, and the sine ones share them.
        meridian = real_harmonics(lmax, grid.colatitudes, 0.0)[0]
        # (order, colatitude, degree), so that a matmul contracts the degree.
        self._legendre = np.ascontiguousarray(meridian.transpose(1, 2, 0))
        # (order, degree, colatitude), the quadrature weights folded in.
        self._weighted = np.ascontiguousarray(
            (meridian * grid.weights).transpose(1, 0, 2))
        angle = np.arange(lmax + 1)[:, None] * grid.longitudes[None, :]
        self._cosines = np.cos(angle)                      # (order, longitude)
        self._sines = np.sin(angle)
        self._cell = 2.0 * np.pi / grid.nphi

    @property
    def lmax(self) -> int:
        """The grid's band: the degree the table holds."""
        return self._lmax

    def synthesise(self, coefficients: ArrayLike) -> np.ndarray:
        """Values on every shell from coefficient functions.

        `coefficients` has shape (2, lmax + 1, lmax + 1, shells), a
        coefficient array with one radial axis; the result has shape
        (shells, ntheta, nphi), the layout of a sample.
        """
        c = np.asarray(coefficients, dtype=float)
        n = self._lmax + 1
        if c.ndim != 4 or c.shape[:3] != (2, n, n):
            raise ValueError(f"coefficients must have shape (2, {n}, {n}, "
                             f"shells), got {c.shape}")
        shells = c.shape[3]
        # (order, degree, shells) -> (order, colatitude, shells)
        even = np.matmul(self._legendre, c[0].transpose(1, 0, 2))
        odd = np.matmul(self._legendre, c[1].transpose(1, 0, 2))
        # (longitude, colatitude * shells)
        flat = self._cosines.T @ even.reshape(n, -1)
        flat += self._sines.T @ odd.reshape(n, -1)
        return np.ascontiguousarray(flat.reshape(-1, n, shells).transpose(2, 1, 0))

    def analyse(self, values: ArrayLike) -> np.ndarray:
        """Coefficient functions from values on every shell, by the grid's
        own quadrature.

        `values` has shape (shells, ntheta, nphi); the result has shape
        (2, lmax + 1, lmax + 1, shells), zero where no harmonic is.
        """
        v = np.asarray(values, dtype=float)
        n = self._lmax + 1
        if v.ndim != 3 or v.shape[1:] != (n, 2 * self._lmax + 1):
            raise ValueError(f"values must have shape (shells, {n}, "
                             f"{2 * self._lmax + 1}), got {v.shape}")
        shells = v.shape[0]
        flat = v.transpose(2, 1, 0).reshape(v.shape[2], -1)  # (longitude, ...)
        out = np.empty((2, n, n, shells))
        for part, waves in enumerate((self._cosines, self._sines)):
            fourier = (self._cell * waves) @ flat   # (order, colatitude * shells)
            fourier = fourier.reshape(n, n, shells)
            # (order, degree, colatitude) @ (order, colatitude, shells)
            out[part] = np.matmul(self._weighted, fourier).transpose(1, 0, 2)
        return out
