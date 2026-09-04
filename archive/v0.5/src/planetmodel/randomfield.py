"""randomfield.py -- Whittle-Matern Gaussian random fields on annuli,
balls and layered models.

The SPDE construction (Whittle 1954; Lindgren, Rue & Lindstrom 2011):
a Matern-type field is the solution of  A^beta u = white noise  with
A = 1 - div(lambda^2 grad), so its covariance is C = A^{-2 beta}, and
matching the Matern smoothness parameter requires

    2 beta = nu + d / 2,

with d = 3 for a scalar field of the annulus or ball (SphericalGRF)
and d = 1 for a field of radius alone (RadialGRF -- the operator still
carries the r^2 dr measure of its spherical origin, but its eigenvalue
counting, hence the smoothness bookkeeping, is one-dimensional).  The
length-scale function lambda(r) enters through kappa = lambda^2 and
may vary with radius; an independent horizontal scale lambda_h feeds
the centrifugal term of the operator family, giving anisotropic fields
without breaking the block-diagonality over spherical harmonics.

Sampling is by the discrete Karhunen-Loeve expansion of each degree
block: with the M-orthonormal eigenpairs (Theta_l, Phi_l) of A_l from
sobolev.RadialOperatorFamily, discrete white noise is Phi z with
z ~ N(0, I), so

    u_lm(r) = Phi_l Theta_l^{-beta} z_lm ,   z_lm ~ N(0, I) iid over (l, m).

Modes are truncated where their contribution to the total variance
falls below a relative tolerance, and (for SphericalGRF with
lmax=None) the degree sum is truncated the same way.  The marginal
standard deviation is then normalized *exactly*: the raw pointwise
variance of the truncated expansion,

    sigma_raw^2(r) = sum_l (2l+1)/(4 pi) sum_j theta_lj^{-2 beta} phi_lj(r)^2,

is computed at the nodes and every sample is scaled by
sigma(r) / sigma_raw(r), so the discrete field has the requested
sigma(r) to machine precision (a pointwise rescaling of the SPDE
field, standard practice since the SPDE marginal variance otherwise
varies with position, boundary distance and truncation).

Artificial boundaries are pushed away from the physical domain by
fictitious padding: the computational interval extends beyond each
requested endpoint by pad_factor times the Matern effective range
sqrt(8 nu) lambda(boundary) (the distance at which the correlation has
decayed to about 0.1; cf. Khristenko et al. 2019 on boundary effects
in Matern SPDE sampling), with lambda continued constantly into the
pads.  The inner pad clamps at r = 0 -- the computational domain may
become a ball even when the physical annulus does not reach the
centre -- and r = 0 itself needs no padding or boundary condition.
Robin coefficients (Daon & Stadler 2018) can be added on top via
robin="auto", which sets gamma = 1 / lambda(boundary) at each
artificial boundary, or explicitly.

Conventions.  SphericalGRF samples are coefficient arrays with respect
to *real orthonormal* spherical harmonics (pyshtools
normalization='ortho', csphase=1), in the pyshtools real layout with a
trailing radial axis: coeffs[0, l, m, :] multiplies the cosine
harmonic of order m, coeffs[1, l, m, :] the sine harmonic, at the
physical nodes self.r.  Angular synthesis is left to the caller (e.g.
pyshtools); only radial coefficient functions are produced here.  All
fields are real and zero-mean.
"""
from __future__ import annotations

import warnings

import numpy as np
from scipy.interpolate import PPoly

from .model import RadialField, ReferenceBody, Skeleton
from .mesh1d import Mesh1D
from .sobolev import RadialOperatorFamily

__all__ = ["RadialGRF", "SphericalGRF", "LayeredGRF"]


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _as_fun(x):
    """A scalar or callable of radius as a vectorized callable."""
    if callable(x):
        return lambda r: np.broadcast_to(
            np.asarray(x(r), dtype=float), np.shape(r)).copy()
    v = float(x)
    return lambda r: np.full(np.shape(r), v)


def _padded_mesh(r1: float, r2: float, nu: float, lam_fun, ngll: int,
                 drmax: float | None, pad_factor: float
                 ) -> tuple[Mesh1D, float]:
    """The padded computational Mesh1D for a physical interval [r1, r2].

    Pads each side by pad_factor sqrt(8 nu) lambda(boundary), clamping
    the inner end at r = 0; r1 and r2 are pinned breakpoints, so the
    physical interval is always a whole number of elements.  Returns
    the mesh and the drmax actually used (default lambda_min / 2).
    """
    lam_s = lam_fun(np.linspace(r1, r2, 257))
    if np.any(lam_s <= 0.0) or not np.all(np.isfinite(lam_s)):
        raise ValueError("lambda must be positive and finite on [r1, r2]")
    if drmax is None:
        drmax = float(lam_s.min()) / 2.0
    reach = pad_factor * np.sqrt(8.0 * nu)
    lo = max(0.0, r1 - reach * float(lam_s[0]))
    hi = r2 + reach * float(lam_s[-1])
    breaks = np.unique([lo, r1, r2, hi])
    return Mesh1D(breaks, ngll=ngll, drmax=drmax), float(drmax)


def _phys_slice(mesh: Mesh1D, r1: float, r2: float
                ) -> tuple[int, int, slice]:
    """Element range and contiguous global-node slice of [r1, r2]."""
    e0 = int(np.searchsorted(mesh.left, r1, side="left"))
    e1 = int(np.searchsorted(mesh.right, r2, side="left"))
    if not (np.isclose(mesh.left[e0], r1) and np.isclose(mesh.right[e1], r2)):
        raise RuntimeError("physical endpoints are not mesh breakpoints")
    return e0, e1, slice(mesh.gmap[e0, 0], mesh.gmap[e1, -1] + 1)


def _truncated_modes(fam: RadialOperatorFamily, l: int, p: float,
                     tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Leading eigenpairs of A_l under the variance-trace criterion.

    Keeps the smallest set of modes whose omitted trace satisfies
    sum_tail theta^{-p} <= tol sum_all theta^{-p} (at least one mode is
    always kept), then fetches only those eigenvectors via the
    select-by-value path of the family's eig.
    """
    vals = fam.eigvalsh(l)
    wts = vals ** (-p)
    tail = np.cumsum(wts[::-1])[::-1]
    small = tail <= tol * tail[0]
    k = int(np.argmax(small)) if small.any() else vals.size
    k = max(k, 1)
    if k == vals.size:
        return fam.eig(l)
    return fam.eig(l, theta_max=float(vals[k - 1]) * (1.0 + 1e-12))


def _parse_robin_spec(robin, lam_fun, r1: float, r2: float):
    """Resolve robin="auto" into gamma = 1/lambda at each boundary."""
    if isinstance(robin, str):
        if robin != "auto":
            raise ValueError("robin must be None, 'auto', a number or a pair")
        return (1.0 / float(lam_fun(r1)), 1.0 / float(lam_fun(r2)))
    return robin


# ---------------------------------------------------------------------------
# fields of radius alone
# ---------------------------------------------------------------------------

class RadialGRF:
    """A real, zero-mean Gaussian random field of radius on [r1, r2].

    Matern-type with smoothness nu > 0, length scale lam (a scalar or
    a callable of radius) and marginal standard deviation sigma
    (likewise), via C = A^{-2 beta} with 2 beta = nu + 1/2; r1 = 0
    gives a field of the full ball.  The computational mesh pads
    [r1, r2] as described in the module docstring; ngll, drmax,
    pad_factor, robin and the truncation tolerance tol tune the
    discretization.  Samples are nodal values at the physical GLL
    nodes self.r; to_ppoly turns any such vector into an exact
    piecewise-polynomial callable on [r1, r2].
    """

    def __init__(self, r1: float, r2: float, nu: float, lam,
                 *, sigma=1.0, ngll: int = 5, drmax: float | None = None,
                 pad_factor: float = 1.5, robin=None,
                 tol: float = 1e-6) -> None:
        """Build the padded operator family and the sampling factor."""
        r1, r2 = float(r1), float(r2)
        if not 0.0 <= r1 < r2:
            raise ValueError("need 0 <= r1 < r2")
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        self.interval = (r1, r2)
        self.nu = float(nu)
        self.beta = 0.5 * (self.nu + 0.5)

        lam_f = _as_fun(lam)
        clipped = lambda r: lam_f(np.clip(r, r1, r2))
        self.mesh, self.drmax = _padded_mesh(r1, r2, self.nu, clipped,
                                             ngll, drmax, pad_factor)
        robin = _parse_robin_spec(robin, clipped, r1, r2)
        self.family = RadialOperatorFamily(
            self.mesh, kappa=lambda r: clipped(r) ** 2, weight="r2",
            robin=robin)

        e0, e1, sl = _phys_slice(self.mesh, r1, r2)
        self._elements, self._slice = (e0, e1 + 1), sl
        self.r = self.mesh.rglob[sl].copy()
        self.r.setflags(write=False)

        theta, Phi = _truncated_modes(self.family, 0, 2.0 * self.beta, tol)
        Phi = self.family.embed(0, Phi)[sl]   # reconstructs a ball's axis value
        var_raw = (Phi ** 2) @ theta ** (-2.0 * self.beta)
        if np.any(var_raw <= 0.0):
            raise RuntimeError("vanishing raw variance at a physical node")

        sig = _as_fun(sigma)(self.r)
        if np.any(sig < 0.0):
            raise ValueError("sigma must be non-negative")
        self.sigma = sig
        self.sigma.setflags(write=False)
        scale = sig / np.sqrt(var_raw)
        self._B = scale[:, None] * Phi * theta[None, :] ** (-self.beta)

    @property
    def nmodes(self) -> int:
        """Number of Karhunen-Loeve modes kept."""
        return self._B.shape[1]

    def sample(self, rng=None, size: int | None = None) -> np.ndarray:
        """Draw a sample (or `size` of them) at the physical nodes.

        rng is anything np.random.default_rng accepts.  Returns shape
        (r.size,) or (size, r.size).
        """
        rng = np.random.default_rng(rng)
        if size is None:
            return self._B @ rng.standard_normal(self.nmodes)
        return (self._B @ rng.standard_normal((self.nmodes, int(size)))).T

    def covariance(self) -> np.ndarray:
        """Dense covariance of the nodal values (diagonal = sigma^2)."""
        return self._B @ self._B.T

    def std(self) -> np.ndarray:
        """Marginal standard deviation at the nodes (exact by design)."""
        return self.sigma.copy()

    def to_ppoly(self, values) -> PPoly:
        """Exact PPoly on [r1, r2] of nodal values (e.g. one sample)."""
        v = np.asarray(values, dtype=float)
        if v.shape != self.r.shape:
            raise ValueError(f"expected {self.r.shape} nodal values, "
                             f"got {v.shape}")
        full = np.zeros(self.mesh.nglob)
        full[self._slice] = v
        return self.mesh.to_ppoly(full[self.mesh.gmap],
                                  elements=self._elements)

    def __repr__(self) -> str:
        """Compact summary of the field's setup."""
        r1, r2 = self.interval
        return (f"RadialGRF([{r1:g}, {r2:g}], nu={self.nu:g}, "
                f"{self.nmodes} modes on {self.r.size} nodes)")


# ---------------------------------------------------------------------------
# scalar fields of the annulus or ball
# ---------------------------------------------------------------------------

class SphericalGRF:
    """A real, zero-mean Gaussian random field of a spherical annulus
    (or ball, r1 = 0), delivered as spherical-harmonic radial
    coefficients.

    Matern-type with C = A^{-2 beta}, 2 beta = nu + 3/2; lam is the
    length scale (scalar or callable of radius), lam_h an optional
    distinct horizontal scale, sigma the marginal standard deviation.
    lmax=None chooses the degree cut automatically: degrees are added
    until their variance trace stays below tol times the running total
    for two degrees in a row (lmax_cap bounds the scan, with a warning
    if it bites).  sample() returns real orthonormal
    spherical-harmonic coefficients as an array of shape
    (2, lmax + 1, lmax + 1, r.size) -- see the module docstring for
    the layout -- and the pointwise standard deviation of the
    truncated synthesis is exactly sigma(r).
    """

    def __init__(self, r1: float, r2: float, nu: float, lam,
                 *, sigma=1.0, lmax: int | None = None, lam_h=None,
                 ngll: int = 5, drmax: float | None = None,
                 pad_factor: float = 1.5, robin=None, tol: float = 1e-6,
                 lmax_cap: int = 512) -> None:
        """Build the family, choose degrees, truncate and normalize."""
        r1, r2 = float(r1), float(r2)
        if not 0.0 <= r1 < r2:
            raise ValueError("need 0 <= r1 < r2")
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        self.interval = (r1, r2)
        self.nu = float(nu)
        self.beta = 0.5 * (self.nu + 1.5)
        p = 2.0 * self.beta

        lam_f = _as_fun(lam)
        clip_r = lambda r: lam_f(np.clip(r, r1, r2))
        lamh_f = clip_r if lam_h is None else (
            lambda r, f=_as_fun(lam_h): f(np.clip(r, r1, r2)))
        self.mesh, self.drmax = _padded_mesh(r1, r2, self.nu, clip_r,
                                             ngll, drmax, pad_factor)
        robin = _parse_robin_spec(robin, clip_r, r1, r2)
        self.family = RadialOperatorFamily(
            self.mesh, kappa=lambda r: clip_r(r) ** 2,
            kappa_h=lambda r: lamh_f(r) ** 2, weight="r2", robin=robin)

        if lmax is None:
            cum, small, l = 0.0, 0, 0
            while True:
                t = (2 * l + 1) / (4.0 * np.pi) * float(
                    np.sum(self.family.eigvalsh(l) ** (-p)))
                cum += t
                small = small + 1 if t <= tol * cum else 0
                if small >= 2:
                    break
                if l >= lmax_cap:
                    warnings.warn(
                        f"degree scan reached lmax_cap = {lmax_cap} before "
                        f"the variance tolerance was met; the angular "
                        f"truncation error may exceed tol", stacklevel=2)
                    break
                l += 1
            self.lmax = l
        else:
            if lmax < 0:
                raise ValueError("lmax must be non-negative")
            self.lmax = int(lmax)

        e0, e1, sl = _phys_slice(self.mesh, r1, r2)
        self._elements, self._slice = (e0, e1 + 1), sl
        self.r = self.mesh.rglob[sl].copy()
        self.r.setflags(write=False)

        var_raw = np.zeros(self.r.size)
        self._B: list[np.ndarray] = []
        for l in range(self.lmax + 1):
            theta, Phi = _truncated_modes(self.family, l, p, tol)
            Phi = self.family.embed(l, Phi)[sl]
            B = Phi * theta[None, :] ** (-self.beta)
            var_raw += (2 * l + 1) / (4.0 * np.pi) * (B ** 2).sum(axis=1)
            self._B.append(B)
        if np.any(var_raw <= 0.0):
            raise RuntimeError("vanishing raw variance at a physical node")

        sig = _as_fun(sigma)(self.r)
        if np.any(sig < 0.0):
            raise ValueError("sigma must be non-negative")
        self.sigma = sig
        self.sigma.setflags(write=False)
        self._scale = sig / np.sqrt(var_raw)
        self.nmodes = np.array([B.shape[1] for B in self._B])

    @property
    def degrees(self) -> np.ndarray:
        """The spherical-harmonic degrees included, 0 ... lmax."""
        return np.arange(self.lmax + 1)

    def sample(self, rng=None) -> np.ndarray:
        """One sample of the coefficient functions at the physical nodes.

        Returns shape (2, lmax + 1, lmax + 1, r.size) in the pyshtools
        real layout for *orthonormal* harmonics: [0, l, m] cosine,
        [1, l, m] sine (zero for m = 0).  For a ball, coefficients of
        every l >= 1 vanish at r = 0.  Memory is
        2 (lmax + 1)^2 r.size doubles; keep lmax moderate.
        """
        rng = np.random.default_rng(rng)
        L = self.lmax
        c = np.zeros((2, L + 1, L + 1, self.r.size))
        for l, B in enumerate(self._B):
            U = B @ rng.standard_normal((B.shape[1], 2 * l + 1))
            c[0, l, :l + 1] = U[:, :l + 1].T
            if l:
                c[1, l, 1:l + 1] = U[:, l + 1:].T
        c *= self._scale
        return c

    def variance(self) -> np.ndarray:
        """Pointwise variance sigma(r)^2 of the truncated field (exact)."""
        return self.sigma ** 2

    def std(self) -> np.ndarray:
        """Pointwise standard deviation sigma(r) (exact by design)."""
        return self.sigma.copy()

    def __repr__(self) -> str:
        """Compact summary of the field's setup."""
        r1, r2 = self.interval
        return (f"SphericalGRF([{r1:g}, {r2:g}], nu={self.nu:g}, "
                f"lmax={self.lmax}, {int(self.nmodes.sum())} radial modes "
                f"on {self.r.size} nodes)")


# ---------------------------------------------------------------------------
# layered fields: direct sums over a Skeleton
# ---------------------------------------------------------------------------

class LayeredGRF:
    """Independent radial Gaussian fields per layer of a Skeleton,
    assembled into a RadialField.

    Every included layer carries its own RadialGRF on its interval
    (each padded beyond the layer's boundaries -- the pads are
    fictitious and may overlap neighbours, reach past the surface, or
    clamp at the centre); the direct sum is generically discontinuous
    at layer boundaries, as befits per-layer material perturbations.
    nu, lam and sigma are shared across layers when scalars or
    callables, or per-layer when sequences of length skeleton.nlayers.
    layers restricts the sum to the given layer indices; excluded
    layers contribute exactly zero.  sample() returns a RadialField on the
    skeleton, ready for field arithmetic, plotting or model
    perturbation.
    """

    def __init__(self, skeleton, nu, lam, *, sigma=1.0, layers=None,
                 ngll: int = 5, drmax=None,
                 pad_factor: float = 1.5, robin=None, tol: float = 1e-6,
                 name: str | None = None) -> None:
        """Build one RadialGRF per included layer."""
        sk = (skeleton.skeleton if isinstance(skeleton, ReferenceBody)
              else skeleton)
        if not isinstance(sk, Skeleton):
            raise TypeError("expected a Skeleton or ReferenceBody")
        self.skeleton = sk
        self.name = name
        nl = sk.nlayers
        idx = range(nl) if layers is None else layers
        idx = tuple(sorted({i if i >= 0 else i + nl for i in idx}))
        if idx and not 0 <= idx[0] <= idx[-1] < nl:
            raise IndexError("layer index out of range")
        self.layers = idx

        def per_layer(spec, i):
            if isinstance(spec, (list, tuple)):
                if len(spec) != nl:
                    raise ValueError(f"per-layer sequences must have "
                                     f"length {nl}")
                return spec[i]
            return spec

        self._grfs = {
            i: RadialGRF(*sk.interval(i), per_layer(nu, i),
                         per_layer(lam, i), sigma=per_layer(sigma, i),
                         ngll=ngll, drmax=per_layer(drmax, i),
                         pad_factor=pad_factor, robin=robin, tol=tol)
            for i in idx}

    def __getitem__(self, i: int) -> RadialGRF:
        """The RadialGRF of layer i (KeyError if excluded)."""
        return self._grfs[i]

    def sample(self, rng=None) -> RadialField:
        """One sample as a RadialField on the skeleton (zero on excluded layers)."""
        rng = np.random.default_rng(rng)
        funcs = []
        for i in range(self.skeleton.nlayers):
            lo, hi = self.skeleton.interval(i)
            if i in self._grfs:
                g = self._grfs[i]
                funcs.append(g.to_ppoly(g.sample(rng)))
            else:
                funcs.append(PPoly(np.zeros((1, 1)), np.array([lo, hi])))
        return RadialField(self.skeleton, tuple(funcs), name=self.name)

    def __repr__(self) -> str:
        """Compact summary of the layered field."""
        return (f"LayeredGRF({len(self._grfs)}/{self.skeleton.nlayers} "
                f"layers active: {list(self.layers)})")
