"""sobolev.py -- self-adjoint radial operator families and their spectra.

The scalar elliptic operator  A = 1 - div(kappa grad)  on a spherical
annulus or ball, with kappa = kappa(r) (optionally anisotropic, with
distinct radial and horizontal coefficients), block-diagonalizes over
spherical harmonics into the family of radial operators

    A_l f = f - w^{-1} (kappa_r w f')' + kappa_h l(l+1) r^{-2} f,

one per degree l >= 0, acting on the weighted space L^2(w dr) with
w(r) = r^2.  RadialOperatorFamily discretizes every member of this
family at once on a single Mesh1D with the GLL spectral-element basis:
the mass matrix M is diagonal (GLL quadrature -- the standard SEM
lumping, here with one exact correction at r = 0, see below), the
centrifugal term is likewise diagonal because w / r^2 = 1, and the
radial stiffness is banded with half-bandwidth ngll - 1.  Boundary
conditions are natural (Neumann) unless Robin coefficients are given,
in which case  kappa_r f' -+ gamma f = 0  at the inner/outer boundary
adds gamma kappa_r w f g to the form there (Daon & Stadler 2018 use
such terms to reduce boundary artefacts of SPDE random fields).

Eigendecomposition.  Because M is diagonal, the generalized problem
A_l Phi = M Phi Theta symmetrizes into the banded ordinary problem
S = M^{-1/2} A_l M^{-1/2}, handed to LAPACK's banded symmetric drivers
through scipy.linalg.eig_banded / eigvals_banded, with select='v' for
spectral truncation.  Since A_l = M + K_l with K_l >= 0, every
eigenvalue satisfies theta >= 1: negative fractional powers of the
family are uniformly well behaved, which is what the random-field and
Sobolev applications rely on.  Eigenvectors are returned M-orthonormal
(Phi^T M Phi = I), so A_l^s = Phi Theta^s Phi^T M as an operator on
nodal vectors, and <f, A_l^s g> with the L^2(w dr) pairing is the H^s
inner product of the computational Sobolev scale built on A.

The centre of a ball needs care but no boundary condition: for l >= 1
the eigenfunctions vanish like r^l, so the axis dof is dropped
(f(0) = 0 imposed strongly); for l = 0 the axis value is genuine but
carries zero GLL mass (w_0 jac r_0^2 with r_0 = 0), making the pencil
singular there.  The massless row reads (A_0 f)_0 = 0 -- a discrete
regularity condition slaving f(0) to the first element's interior
nodes -- and is eliminated exactly by static condensation: the Schur
complement stays within the band, the reduced pencil has precisely the
finite spectrum of the singular one, and embed() reconstructs the axis
value of any eigenvector or sample.  Degrees are cached
independently, values-only and eigenpair caches separately, so a scan
over hundreds of degrees only pays for what it uses.
"""
from __future__ import annotations

import numpy as np
from scipy.linalg import eig_banded, eigvals_banded

from .mesh1d import Mesh1D

__all__ = ["RadialOperatorFamily"]


def _nodal_coefficient(mesh: Mesh1D, c) -> np.ndarray:
    """Coefficient c as a per-element nodal array of shape (nspec, ngll).

    Accepts a scalar, a callable of radius (vectorized or not), or an
    array already in nodal shape; per-element storage means a callable
    with jumps at breakpoints keeps both one-sided values.
    """
    if callable(c):
        out = np.asarray(c(mesh.r), dtype=float)
        out = np.broadcast_to(out, mesh.r.shape).copy()
    elif np.ndim(c) == 0:
        out = np.full_like(mesh.r, float(c))
    else:
        out = np.asarray(c, dtype=float)
        if out.shape != mesh.r.shape:
            raise ValueError(f"coefficient array must have nodal shape "
                             f"{mesh.r.shape}, got {out.shape}")
        out = out.copy()
    if not np.all(np.isfinite(out)):
        raise ValueError("coefficient must be finite everywhere on the mesh")
    return out


def _parse_robin(robin) -> tuple[float, float]:
    """Normalize the robin argument to (gamma_inner, gamma_outer)."""
    if robin is None:
        return 0.0, 0.0
    if np.ndim(robin) == 0:
        g = float(robin)
        gin, gout = g, g
    else:
        gin, gout = robin
        gin = 0.0 if gin is None else float(gin)
        gout = 0.0 if gout is None else float(gout)
    if gin < 0.0 or gout < 0.0:
        raise ValueError("Robin coefficients must be non-negative")
    return gin, gout


class RadialOperatorFamily:
    """The degree-indexed operators A_l on a 1-D GLL mesh, spectrally.

    Construction assembles, once, the diagonal mass M, the diagonal
    centrifugal factor and the banded radial stiffness on the given
    Mesh1D (a RadialMesh works too); pencil(l) then combines them into
    the symmetric banded matrix of A_l = M + K_r + l(l+1) K_h on the
    degree's active dofs.  kappa (radial) and kappa_h (horizontal,
    defaulting to kappa) may be scalars, callables of radius, or nodal
    arrays; weight selects the measure, "r2" for r^2 dr (annuli and
    balls) or "one" for plain dr on an interval with r > 0.  robin is
    None, a single gamma for both boundaries, or a pair
    (gamma_inner, gamma_outer) with None entries meaning Neumann; an
    inner Robin term is ignored on a ball, where r = 0 is not a
    boundary.

    Methods: eigvalsh(l) for eigenvalues only, eig(l, theta_max=None)
    for M-orthonormal eigenpairs (optionally truncated to
    theta <= theta_max), apply_power(l, s, v) for A_l^s v, inner for
    the H^s inner products, logdet, and the bookkeeping helpers
    ndof / nodes / mass / embed that account for the dropped axis dof
    of ball meshes at l >= 1.  Results are cached per degree.
    """

    def __init__(self, mesh: Mesh1D, *, kappa=1.0, kappa_h=None,
                 weight: str = "r2", robin=None) -> None:
        """Assemble the degree-independent parts on the mesh."""
        if weight not in ("r2", "one"):
            raise ValueError("weight must be 'r2' or 'one'")
        if not isinstance(mesh, Mesh1D):
            raise TypeError("mesh must be a Mesh1D (or RadialMesh)")
        if mesh.rglob[0] < 0.0:
            raise ValueError("mesh must lie in r >= 0")
        if weight == "one" and mesh.rglob[0] <= 0.0:
            raise ValueError("the plain measure needs r > 0 (the "
                             "centrifugal factor 1/r^2 is otherwise "
                             "singular); use weight='r2' for balls")

        self.mesh = mesh
        self.weight = weight
        self._ball = bool(mesh.rglob[0] == 0.0)

        kr = _nodal_coefficient(mesh, kappa)
        kh = kr if kappa_h is None else _nodal_coefficient(mesh, kappa_h)
        if np.any(kr <= 0.0) or np.any(kh < 0.0):
            raise ValueError("kappa must be positive (kappa_h non-negative)")

        r, jac, wq = mesh.r, mesh.jac, mesh.w
        D, g = mesh.deriv, mesh.gmap
        n, q = mesh.nglob, mesh.ngll - 1
        wgt = r * r if weight == "r2" else np.ones_like(r)

        # diagonal GLL mass and centrifugal factor
        M = np.zeros(n)
        np.add.at(M, g, wq[None, :] * jac[:, None] * wgt)
        Kh = np.zeros(n)
        rsafe = np.where(r > 0.0, r, 1.0)
        hloc = kh * (wq[None, :] * jac[:, None]) * (wgt / rsafe**2)
        np.add.at(Kh, g, hloc)

        # banded radial stiffness, upper symmetric storage
        # band[q + i - j, j] = K[i, j] for i <= j
        coef = wq[None, :] * kr * wgt / jac[:, None]
        Ke = np.einsum("qi,eq,qj->eij", D, coef, D)
        band = np.zeros((q + 1, n))
        GA, GB = g[:, :, None], g[:, None, :]
        sel = GA <= GB
        rows = np.broadcast_to(q + GA - GB, Ke.shape)
        cols = np.broadcast_to(GB, Ke.shape)
        np.add.at(band, (rows[sel], cols[sel]), Ke[sel])

        # Robin boundary terms (added to the stiffness diagonal)
        gin, gout = _parse_robin(robin)
        if gout > 0.0:
            band[q, -1] += gout * kr[-1, -1] * wgt[-1, -1]
        if gin > 0.0 and not self._ball:
            band[q, 0] += gin * kr[0, 0] * wgt[0, 0]

        # static condensation data for the massless ball centre: at l = 0
        # the axis row of the pencil reads  d f_0 + a . f_(1..q) = 0
        if self._ball:
            self._axis_a = np.array([band[q - j, j] for j in range(1, q + 1)])
            self._axis_d = float(band[q, 0])
            self._axis_c = -self._axis_a / self._axis_d
        if np.any(M[1 if self._ball else 0:] <= 0.0):
            raise RuntimeError("non-positive mass entry; degenerate mesh?")

        self._M, self._Kh, self._Kr, self._q = M, Kh, band, q
        self._vals: dict[int, np.ndarray] = {}
        self._pairs: dict[int, tuple[np.ndarray, np.ndarray, float]] = {}

    # -- degree bookkeeping -------------------------------------------------

    def _n0(self, l: int) -> int:
        """Number of leading dofs excluded at degree l (the ball axis).

        On a ball the centre is never an active dof: at l >= 1 its
        value is zero, at l = 0 it is slaved to the first element's
        interior nodes by static condensation.
        """
        if l < 0:
            raise ValueError("degree must be non-negative")
        return 1 if self._ball else 0

    def ndof(self, l: int) -> int:
        """Number of active dofs at degree l."""
        return self.mesh.nglob - self._n0(l)

    def nodes(self, l: int) -> np.ndarray:
        """Radii of the active dofs at degree l."""
        return self.mesh.rglob[self._n0(l):]

    def mass(self, l: int) -> np.ndarray:
        """Diagonal mass on the active dofs at degree l."""
        return self._M[self._n0(l):]

    def embed(self, l: int, values) -> np.ndarray:
        """Active-dof values completed to the full mesh (no-op off balls).

        The ball centre is filled with zero for l >= 1 and with the
        statically condensed value  f(0) = c . f_(first element)  for
        l = 0; works on (n,) vectors and (n, k) mode stacks alike.
        """
        v = np.asarray(values, dtype=float)
        n0 = self._n0(l)
        if n0 == 0:
            return v
        out = np.zeros((v.shape[0] + 1,) + v.shape[1:])
        out[1:] = v
        if l == 0:
            out[0] = self._axis_c @ v[:self._q]
        return out

    # -- the pencil and its spectrum ----------------------------------------

    def pencil(self, l: int) -> tuple[np.ndarray, np.ndarray]:
        """Banded A_l (upper storage) and diagonal mass, active dofs only.

        A_l = M + K_r + l(l+1) K_h.  On a ball the axis dof is removed
        at every degree: at l >= 1 its couplings are simply discarded
        (a strong f(0) = 0); at l = 0 the exact Schur complement of the
        massless axis row is folded into the first element's block
        first, so the reduced pencil carries precisely the finite
        spectrum of the singular one.
        """
        n0, q = self._n0(l), self._q
        band = self._Kr.copy()
        if n0 and l == 0:
            a, d = self._axis_a, self._axis_d
            for i in range(1, q + 1):
                for j in range(i, q + 1):
                    band[q + i - j, j] -= a[i - 1] * a[j - 1] / d
        band = band[:, n0:]
        if n0:
            for m in range(q):        # zero stale couplings to the axis dof
                j = q - 1 - m
                if j < band.shape[1]:
                    band[m, j] = 0.0
        mass = self._M[n0:]
        band[q] += mass + (l * (l + 1.0)) * self._Kh[n0:]
        return band, mass

    @staticmethod
    def _scaled(band: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Jacobi-symmetrized band: S[i, j] = s_i band[i, j] s_j."""
        q = band.shape[0] - 1
        n = band.shape[1]
        out = np.zeros_like(band)
        for m in range(q + 1):
            u = q - m
            if u < n:
                js = np.arange(u, n)
                out[m, u:] = band[m, u:] * s[js] * s[js - u]
        return out

    def eigvalsh(self, l: int) -> np.ndarray:
        """All eigenvalues of A_l (ascending, >= 1 up to roundoff), cached."""
        if l not in self._vals:
            band, mass = self.pencil(l)
            v = eigvals_banded(self._scaled(band, 1.0 / np.sqrt(mass)),
                               lower=False)
            v.setflags(write=False)
            self._vals[l] = v
        return self._vals[l]

    def eig(self, l: int, *, theta_max: float | None = None
            ) -> tuple[np.ndarray, np.ndarray]:
        """Eigenpairs (theta, Phi) of A_l Phi = M Phi Theta, cached.

        Phi has the active-dof eigenvectors in its columns, ascending in
        theta and M-orthonormal.  With theta_max, only the eigenvalues
        theta <= theta_max are computed (LAPACK's select-by-value path);
        the cache keeps the widest range requested so far and hands out
        truncated views of it.
        """
        want = np.inf if theta_max is None else float(theta_max)
        cached = self._pairs.get(l)
        if cached is None or cached[2] < want:
            band, mass = self.pencil(l)
            s = 1.0 / np.sqrt(mass)
            S = self._scaled(band, s)
            if np.isinf(want):
                theta, Y = eig_banded(S, lower=False)
                self._vals.setdefault(l, theta)
            else:
                theta, Y = eig_banded(S, lower=False, select="v",
                                      select_range=(0.0, want))
            Phi = s[:, None] * Y
            self._pairs[l] = (theta, Phi, want)
        theta, Phi, have = self._pairs[l]
        if want < have:
            k = int(np.searchsorted(theta, want, side="right"))
            return theta[:k], Phi[:, :k]
        return theta, Phi

    # -- functions of the operators -----------------------------------------

    def apply_power(self, l: int, s: float, v) -> np.ndarray:
        """A_l^s v on active-dof nodal values v of shape (n,) or (n, k).

        Uses the full eigendecomposition, A_l^s = Phi Theta^s Phi^T M;
        any real power is available, positive ones included, because
        the spectrum is bounded below by 1.
        """
        theta, Phi = self.eig(l)
        v = np.asarray(v, dtype=float)
        mass = self.mass(l)
        if v.shape[0] != mass.size:
            raise ValueError(f"expected {mass.size} active dofs, "
                             f"got {v.shape[0]}")
        if v.ndim == 1:
            return Phi @ (theta ** s * (Phi.T @ (mass * v)))
        c = Phi.T @ (mass[:, None] * v)
        return Phi @ (theta[:, None] ** s * c)

    def inner(self, l: int, s: float, u, v) -> float:
        """The H^s inner product <u, A_l^s v> with the L^2(w dr) pairing.

        s = 0 is the plain weighted L^2 inner product; the map
        v -> apply_power(l, -s, v) is the corresponding Riesz map.
        """
        u = np.asarray(u, dtype=float)
        return float(u @ (self.mass(l) * self.apply_power(l, s, v)))

    def logdet(self, l: int, *, s: float = 1.0) -> float:
        """log det A_l^s = s sum log theta, from the values-only cache."""
        return float(s * np.log(self.eigvalsh(l)).sum())

    def __repr__(self) -> str:
        """Compact summary of the family and its cache state."""
        kind = "ball" if self._ball else "annulus"
        return (f"RadialOperatorFamily({kind}, {self.mesh.nglob} nodes, "
                f"weight={self.weight!r}, cached degrees: "
                f"{sorted(set(self._vals) | set(self._pairs))})")
