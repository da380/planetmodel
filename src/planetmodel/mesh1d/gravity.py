"""gravity.py -- gravitational acceleration of a spherically symmetric model.

g(r) by layered quadrature of the enclosed mass; exact for polynomial
densities, so the PREM class gives g to machine precision.
"""
from __future__ import annotations

import numpy as np
from scipy.integrate import fixed_quad

from ..model.body import ReferenceBody

__all__ = ["G_NEWTON", "gravity"]


G_NEWTON = 6.6743e-11    # CODATA gravitational constant, m^3 kg^-1 s^-2


def gravity(model: ReferenceBody, radii, *, n: int = 8, G: float = G_NEWTON,
            panel: float = 5e4) -> np.ndarray:
    """Gravitational acceleration g(r) = 4 pi G M(r) / r^2 at the radii.

    The enclosed mass M(r) is accumulated layer by layer (radii are
    processed in sorted order) with n-point Gauss-Legendre quadrature on
    spans no wider than `panel` metres.  For polynomial densities of
    degree <= 2n - 3 -- PREM's are cubic -- this is exact regardless of
    panelling; for spline decks the panelling keeps it near machine
    accuracy.  g(0) = 0, and if the model's innermost boundary is
    positive the mass below it is taken to be zero.
    """
    sk = model.skeleton
    b = sk.boundaries
    rho = model["rho"]
    r = np.asarray(radii, dtype=float)
    flat = r.ravel()
    tol = 1e-6 * b[-1]
    if flat.size and (flat.min() < b[0] - tol or flat.max() > b[-1] + tol):
        raise ValueError("radii outside the model")
    rs = np.clip(flat, b[0], b[-1])
    order = np.argsort(rs, kind="stable")

    def span_mass(f, lo: float, hi: float) -> float:
        """Integral of rho s^2 over [lo, hi] within one layer."""
        if hi <= lo:
            return 0.0
        npan = max(1, int(np.ceil((hi - lo) / panel)))
        edges = np.linspace(lo, hi, npan + 1)
        return sum(fixed_quad(lambda s: f(s) * s * s, a2, b2, n=n)[0]
                   for a2, b2 in zip(edges[:-1], edges[1:]))

    M = np.empty(rs.size)
    total, prev, ilay = 0.0, float(b[0]), 0
    for j in order:
        rt = float(rs[j])
        while ilay < sk.nlayers - 1 and rt > b[ilay + 1]:
            total += span_mass(rho[ilay], prev, float(b[ilay + 1]))
            prev = float(b[ilay + 1])
            ilay += 1
        total += span_mass(rho[ilay], prev, rt)
        prev = rt
        M[j] = total
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(rs > 0.0, rs, 1.0)
        g = np.where(rs > 0.0, 4.0 * np.pi * G * M / safe**2, 0.0)
    return g.reshape(r.shape)
