"""Real orthonormal spherical harmonics, and synthesis from coefficients.

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
m = 0 in the sine slot and for m > l.  Scipy's complex harmonics carry
the Condon-Shortley phase; it is removed here.  The layout and the
convention are those of pyshtools under `normalization="ortho"` and
`csphase=1`, so a coefficient array here is a pyshtools `cilm` array
as it stands.

Two routes.  `real_harmonics` and `synthesise` evaluate the basis at
arbitrary points through scipy, vectorised over the points: the
definition, and the oracle everything else is held to.  `synthesise_grid`
and `analyse_grid` go between coefficients and values on a
Gauss-Legendre `AngularGrid` through pyshtools, with the ducc0 backend
selected where it is installed: fast transforms, exact for a field
band-limited to the grid's degree, needing the `harmonics` extra.  The
grid is `gauss_legendre(lmax)` of `planetmodel.sampling`, whose nodes
are pyshtools' GLQ nodes and whose 2 lmax + 1 longitudes are the count
pyshtools takes.
"""
from __future__ import annotations

from types import ModuleType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import sph_harm_y

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["real_harmonics", "synthesise", "synthesise_grid", "analyse_grid"]


def real_harmonics(lmax: int, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """The basis up to degree `lmax` at colatitudes `theta` and longitudes
    `phi` (broadcast together): shape (2, lmax + 1, lmax + 1) + points."""
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("lmax must be non-negative")
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    shape = np.broadcast_shapes(theta.shape, phi.shape)
    theta = np.broadcast_to(theta, shape)
    phi = np.broadcast_to(phi, shape)
    out = np.zeros((2, lmax + 1, lmax + 1) + shape)
    for l in range(lmax + 1):
        m = np.arange(l + 1)
        y = sph_harm_y(l, m.reshape((-1,) + (1,) * len(shape)), theta[None], phi[None])
        sign = ((-1.0) ** m).reshape((-1,) + (1,) * len(shape))
        y = y * sign
        out[0, l, 0] = y[0].real
        if l:
            out[0, l, 1:l + 1] = np.sqrt(2.0) * y[1:].real
            out[1, l, 1:l + 1] = np.sqrt(2.0) * y[1:].imag
    return out


def synthesise(coeffs: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """sum_lm c_lm Y_lm(theta, phi) for coefficients of shape
    (2, lmax + 1, lmax + 1), or that shape followed by the points' shape
    when the coefficients vary from point to point."""
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, lmax + 1, lmax + 1, ...)")
    Y = real_harmonics(c.shape[1] - 1, theta, phi)
    if c.ndim == 3:
        c = np.broadcast_to(c.reshape(c.shape + (1,) * (Y.ndim - 3)), Y.shape)
    if c.shape != Y.shape:
        raise ValueError(f"coefficients of shape {c.shape} do not match points "
                         f"of shape {Y.shape[3:]}")
    return np.einsum("slm...,slm...->...", c, Y)


# -- grids, through pyshtools ---------------------------------------------------

def _pyshtools() -> ModuleType:
    """pyshtools with the ducc0 backend selected where present; imported
    here and nowhere else in planetmodel."""
    try:
        import pyshtools
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "the grid synthesis and analysis of planetmodel.harmonics need "
            "pyshtools.  Install it with:\n"
            "    pip install 'planetmodel[harmonics]'      "
            "(or: poetry install --extras harmonics)"
        ) from exc
    if pyshtools.backends.preferred_backend() != "ducc":
        try:
            import ducc0  # noqa: F401
            pyshtools.backends.select_preferred_backend("ducc")
        except ImportError:  # pragma: no cover - the shtools backend serves
            pass
    return pyshtools


def _glq(grid: AngularGrid) -> int:
    """The band of a Gauss-Legendre grid pyshtools can transform on, or a
    ValueError naming what is wrong."""
    if grid.kind != "gauss_legendre" or grid.lmax is None:
        raise ValueError(
            f"grid synthesis and analysis take a Gauss-Legendre grid with a "
            f"band, gauss_legendre(lmax); got a {grid.kind!r} grid")
    lmax = int(grid.lmax)
    if grid.nphi != 2 * lmax + 1:
        raise ValueError(
            f"a Gauss-Legendre grid of band {lmax} is transformed with "
            f"2 lmax + 1 = {2 * lmax + 1} longitudes; this grid has {grid.nphi}: "
            f"build it as gauss_legendre({lmax})")
    x = np.polynomial.legendre.leggauss(lmax + 1)[0]
    if not np.allclose(np.cos(grid.colatitudes), np.sort(x)[::-1], atol=1e-12):
        raise ValueError(
            f"the colatitudes are not the {lmax + 1} Gauss-Legendre nodes of "
            f"band {lmax}, north to south")
    return lmax


def synthesise_grid(coeffs: ArrayLike, grid: AngularGrid) -> np.ndarray:
    """Values on a Gauss-Legendre grid from coefficients, through pyshtools.

    `coeffs` has shape (2, l + 1, l + 1) or that followed by any extra
    axes, a radial axis say, with l at most the grid's band; the result
    has the extra axes first and then (ntheta, nphi), the layout of a
    sample.  Exact for a field band-limited to the grid's degree.
    """
    sh = _pyshtools()
    lmax = _glq(grid)
    c = np.asarray(coeffs, dtype=float)
    if c.ndim < 3 or c.shape[0] != 2 or c.shape[1] != c.shape[2]:
        raise ValueError("coefficients must have shape (2, l + 1, l + 1, ...)")
    if c.shape[1] - 1 > lmax:
        raise ValueError(
            f"coefficients to degree {c.shape[1] - 1} on a grid of band {lmax}: "
            "the grid cannot hold them; use gauss_legendre(l) with l at least "
            f"{c.shape[1] - 1}")
    extra = c.shape[3:]
    flat = c.reshape(c.shape[:3] + (-1,))
    out = np.empty((flat.shape[3], grid.ntheta, grid.nphi))
    for k in range(flat.shape[3]):
        out[k] = sh.expand.MakeGridGLQ(flat[..., k], lmax=lmax, norm=4, csphase=1)
    return out.reshape(extra + (grid.ntheta, grid.nphi))


def analyse_grid(values: ArrayLike, grid: AngularGrid, *,
                 lmax: int | None = None) -> np.ndarray:
    """Coefficients from values on a Gauss-Legendre grid, through pyshtools.

    `values` has shape (..., ntheta, nphi), any extra axes first; the
    result is (2, lmax + 1, lmax + 1) followed by the extra axes, with
    `lmax` the grid's band by default or a smaller degree to stop at.
    The inverse of `synthesise_grid` for a field band-limited to the
    grid; a projection otherwise.
    """
    sh = _pyshtools()
    band = _glq(grid)
    lmax = band if lmax is None else int(lmax)
    if not 0 <= lmax <= band:
        raise ValueError(f"lmax must lie in 0..{band}, got {lmax}")
    v = np.asarray(values, dtype=float)
    if v.ndim < 2 or v.shape[-2:] != (grid.ntheta, grid.nphi):
        raise ValueError(
            f"values must have trailing shape {(grid.ntheta, grid.nphi)}, "
            f"got {v.shape}")
    extra = v.shape[:-2]
    flat = v.reshape((-1,) + v.shape[-2:])
    zero, w = sh.expand.SHGLQ(band)
    out = np.empty((flat.shape[0], 2, lmax + 1, lmax + 1))
    for k in range(flat.shape[0]):
        out[k] = sh.expand.SHExpandGLQ(flat[k], w, zero, norm=4, csphase=1,
                                       lmax_calc=lmax)
    return np.moveaxis(out, 0, -1).reshape((2, lmax + 1, lmax + 1) + extra)
