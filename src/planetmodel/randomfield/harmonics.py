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
the Condon-Shortley phase; it is removed here.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from scipy.special import sph_harm_y

__all__ = ["real_harmonics", "synthesise"]


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
