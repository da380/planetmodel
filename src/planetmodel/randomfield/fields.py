"""Whittle-Matern Gaussian random fields on annuli, balls and layers.

The SPDE construction: a Matern-type field solves A^beta u = white
noise with A = 1 - div(lambda^2 grad), so its covariance is
C = A^-2 beta, and matching the Matern smoothness nu requires

    2 beta = nu + d / 2,

with d = 3 for a scalar field of the annulus or ball (`SphericalGRF`)
and d = 1 for a field of radius alone (`RadialGRF`; the operator still
carries the r^2 dr measure, but its eigenvalue counting, hence the
smoothness bookkeeping, is one-dimensional).  The length scale
lambda(r) enters through kappa = lambda^2 and may vary with radius; an
independent horizontal scale lambda_h feeds the centrifugal term,
giving anisotropic fields without breaking the block-diagonality over
spherical harmonics.

Sampling is the discrete Karhunen-Loeve expansion of each degree block:
with the M-orthonormal eigenpairs (Theta_l, Phi_l) of A_l from
`RadialOperatorFamily`, discrete white noise is Phi z with z ~ N(0, I),
so u_lm(r) = Phi_l Theta_l^-beta z_lm with z_lm iid over (l, m).  Modes
are truncated where their contribution to the total variance falls
below a relative tolerance, and for `SphericalGRF` with `lmax=None` the
degree sum is truncated the same way.  The marginal standard deviation
is then normalised exactly: the raw pointwise variance of the truncated
expansion,

    sigma_raw^2(r) = sum_l (2l+1)/(4 pi) sum_j theta_lj^-2 beta phi_lj(r)^2,

is computed at the nodes and every sample is scaled by
sigma(r) / sigma_raw(r), so the discrete field has the requested
sigma(r) to machine precision.

Artificial boundaries are pushed away from the physical domain by
fictitious padding: the computational interval extends beyond each
requested endpoint by `pad_factor` times the Matern effective range
sqrt(8 nu) lambda(boundary), with lambda continued constantly into the
pads.  The inner pad clamps at r = 0, which itself needs no padding or
boundary condition.  Robin coefficients can be added on top via
`robin="auto"`, which sets gamma = 1 / lambda(boundary) at each
artificial boundary, or explicitly.

`SphericalGRF` samples are coefficient arrays with respect to the real
orthonormal harmonics of `randomfield.harmonics`, with a trailing radial
axis over the physical nodes `r`; `to_field` wraps one as an
`AnalyticField` on the shell.  `LayeredGRF` draws an independent
`RadialGRF` on each chosen layer of a skeleton and returns one
`RadialField` per layer.  All fields are real and zero-mean.
"""
from __future__ import annotations

import warnings
from math import comb

import numpy as np
from scipy.interpolate import PPoly

from ..character import SCALAR, Character
from ..fields import AnalyticField, RadialField
from ..layerfunction import PolynomialLayer, constant_layer
from ..mesh1d import Mesh1D
from ..skeleton import Skeleton
from .harmonics import real_harmonics
from .operator import RadialOperatorFamily

__all__ = ["RadialGRF", "SphericalGRF", "LayeredGRF"]


# -- helpers ------------------------------------------------------------------

def _as_fun(x):
    """A scalar or callable of radius as a vectorised callable."""
    if callable(x):
        return lambda r: np.broadcast_to(np.asarray(x(r), dtype=float),
                                         np.shape(r)).copy()
    v = float(x)
    return lambda r: np.full(np.shape(r), v)


def _padded_mesh(r1: float, r2: float, nu: float, lam_fun, ngll: int,
                 drmax: float | None, pad_factor: float) -> tuple[Mesh1D, float]:
    """The padded computational mesh for a physical interval [r1, r2]:
    each side padded by pad_factor sqrt(8 nu) lambda(boundary), the
    inner end clamped at r = 0, r1 and r2 pinned as breakpoints.
    Returns the mesh and the drmax used (lambda_min / 2 by default)."""
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


def _phys_slice(mesh: Mesh1D, r1: float, r2: float) -> tuple[int, int, slice]:
    """Element range and contiguous global-node slice of [r1, r2]."""
    e0 = int(np.searchsorted(mesh.left, r1, side="left"))
    e1 = int(np.searchsorted(mesh.right, r2, side="left"))
    if not (np.isclose(mesh.left[e0], r1) and np.isclose(mesh.right[e1], r2)):
        raise RuntimeError("physical endpoints are not mesh breakpoints")
    return e0, e1, slice(mesh.gmap[e0, 0], mesh.gmap[e1, -1] + 1)


def _truncated_modes(fam: RadialOperatorFamily, l: int, p: float,
                     tol: float) -> tuple[np.ndarray, np.ndarray]:
    """Leading eigenpairs of A_l under the variance-trace criterion: the
    smallest set of modes whose omitted trace satisfies
    sum_tail theta^-p <= tol sum_all theta^-p, at least one kept."""
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
    """robin="auto" resolved into gamma = 1 / lambda at each boundary."""
    if isinstance(robin, str):
        if robin != "auto":
            raise ValueError("robin must be None, 'auto', a number or a pair")
        return (1.0 / float(lam_fun(r1)), 1.0 / float(lam_fun(r2)))
    return robin


def _ppoly(mesh: Mesh1D, nodal, elements: tuple[int, int]) -> PPoly:
    """The exact piecewise polynomial of nodal values with any trailing
    shape, over a half-open element range."""
    e0, e1 = elements
    n = mesh.ngll
    block = np.asarray(nodal, dtype=float)[e0:e1]
    ne, extra = block.shape[0], block.shape[2:]
    V = np.vander(mesh.xi, n, increasing=True)
    flat = block.reshape(ne, n, -1).transpose(1, 0, 2).reshape(n, -1)
    cxi = np.linalg.solve(V, flat)
    cxi = cxi.reshape(n, ne, -1)
    C = np.zeros((n, ne) + extra)
    ks = np.arange(n)
    for i, e in enumerate(range(e0, e1)):
        # xi = t / jac - 1 with t = x - left: xi^k = sum_j C(k, j) (-1)^(k-j) (t/jac)^j
        T = np.array([[comb(k, j) * (-1.0) ** (k - j) / mesh.jac[e] ** j
                       if j <= k else 0.0 for k in ks] for j in ks])
        C[::-1, i] = (T @ cxi[:, i]).reshape((n,) + extra)
    x = np.concatenate((mesh.left[e0:e1], mesh.right[e1 - 1:e1]))
    return PPoly(C, x)


# -- fields of radius alone -------------------------------------------------

class RadialGRF:
    """A real, zero-mean Gaussian random field of radius on [r1, r2].

    Matern-type with smoothness `nu` > 0, length scale `lam` (a scalar
    or a callable of radius) and marginal standard deviation `sigma`
    (likewise), via C = A^-2 beta with 2 beta = nu + 1/2; r1 = 0 gives
    a field of the whole ball.  `ngll`, `drmax`, `pad_factor`, `robin`
    and the truncation tolerance `tol` tune the discretisation.  Samples
    are nodal values at the physical GLL nodes `r`; `to_layer` turns any
    such vector into an exact `PolynomialLayer` on [r1, r2].
    """

    def __init__(self, r1: float, r2: float, nu: float, lam, *, sigma=1.0,
                 ngll: int = 5, drmax: float | None = None,
                 pad_factor: float = 1.5, robin=None, tol: float = 1e-6) -> None:
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
        self.mesh, self.drmax = _padded_mesh(r1, r2, self.nu, clipped, ngll,
                                             drmax, pad_factor)
        robin = _parse_robin_spec(robin, clipped, r1, r2)
        self.family = RadialOperatorFamily(
            self.mesh, kappa=lambda r: clipped(r) ** 2, weight="r2", robin=robin)

        e0, e1, sl = _phys_slice(self.mesh, r1, r2)
        self._elements, self._slice = (e0, e1 + 1), sl
        self.r = self.mesh.rglob[sl].copy()
        self.r.setflags(write=False)

        theta, Phi = _truncated_modes(self.family, 0, 2.0 * self.beta, tol)
        Phi = self.family.embed(0, Phi)[sl]
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

    @property
    def factor(self) -> np.ndarray:
        """The covariance factor B, shape (r.size, nmodes): samples are
        B z with z standard normal, and the covariance is B B^T."""
        return self._B

    def sample(self, *, rng=None, size: int | None = None) -> np.ndarray:
        """A sample (or `size` of them) at the physical nodes: shape
        (r.size,) or (size, r.size).  `rng` is anything
        `np.random.default_rng` accepts."""
        rng = np.random.default_rng(rng)
        if size is None:
            return self._B @ rng.standard_normal(self.nmodes)
        return (self._B @ rng.standard_normal((self.nmodes, int(size)))).T

    def covariance(self) -> np.ndarray:
        """Dense covariance of the nodal values (diagonal sigma^2)."""
        return self._B @ self._B.T

    def std(self) -> np.ndarray:
        """Marginal standard deviation at the nodes (exact by design)."""
        return self.sigma.copy()

    def to_layer(self, values) -> PolynomialLayer:
        """The exact `PolynomialLayer` on [r1, r2] of nodal values."""
        v = np.asarray(values, dtype=float)
        if v.shape != self.r.shape:
            raise ValueError(f"expected {self.r.shape} nodal values, got {v.shape}")
        full = np.zeros(self.mesh.nglob)
        full[self._slice] = v
        return PolynomialLayer(self.mesh.to_ppoly(full[self.mesh.gmap],
                                                  elements=self._elements))

    def to_field(self, values, *, character: Character = SCALAR,
                 name: str | None = None) -> RadialField:
        """A sample as a `RadialField` on [r1, r2]."""
        return RadialField(self.interval, self.to_layer(values),
                           character=character, name=name)

    def __repr__(self) -> str:
        r1, r2 = self.interval
        return (f"RadialGRF([{r1:g}, {r2:g}], nu={self.nu:g}, "
                f"{self.nmodes} modes on {self.r.size} nodes)")


# -- scalar fields of the annulus or ball ---------------------------------------

class SphericalGRF:
    """A real, zero-mean Gaussian random field of a spherical annulus (or
    ball, r1 = 0), delivered as spherical-harmonic coefficient functions.

    Matern-type with C = A^-2 beta, 2 beta = nu + 3/2; `lam` is the
    length scale (scalar or callable of radius), `lam_h` an optional
    distinct horizontal scale, `sigma` the marginal standard deviation.
    `lmax=None` chooses the degree cut automatically: degrees are added
    until their variance trace stays below `tol` times the running
    total for two degrees in a row (`lmax_cap` bounds the scan, with a
    warning if it bites).  `sample()` returns coefficients of shape
    (2, lmax + 1, lmax + 1, r.size) in the layout of
    `randomfield.harmonics`, and the pointwise standard deviation of the
    truncated synthesis is exactly sigma(r).
    """

    def __init__(self, r1: float, r2: float, nu: float, lam, *, sigma=1.0,
                 lmax: int | None = None, lam_h=None, ngll: int = 5,
                 drmax: float | None = None, pad_factor: float = 1.5,
                 robin=None, tol: float = 1e-6, lmax_cap: int = 512) -> None:
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
        self.mesh, self.drmax = _padded_mesh(r1, r2, self.nu, clip_r, ngll,
                                             drmax, pad_factor)
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
                        f"degree scan reached lmax_cap = {lmax_cap} before the "
                        f"variance tolerance was met; the angular truncation "
                        f"error may exceed tol", stacklevel=2)
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

    def factor(self, l: int) -> np.ndarray:
        """The covariance factor of degree l, shape (r.size, nmodes[l]),
        with the marginal normalisation applied: the coefficient
        functions of every order m of degree l are B_l z."""
        return self._scale[:, None] * self._B[l]

    def sample(self, *, rng=None) -> np.ndarray:
        """One sample of the coefficient functions at the physical nodes:
        shape (2, lmax + 1, lmax + 1, r.size).  For a ball, coefficients
        of every l >= 1 vanish at r = 0."""
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

    def coefficient_functions(self, coeffs) -> PPoly:
        """A coefficient sample as one `PPoly` of radius on [r1, r2] whose
        values have shape (2, lmax + 1, lmax + 1)."""
        c = np.asarray(coeffs, dtype=float)
        want = (2, self.lmax + 1, self.lmax + 1, self.r.size)
        if c.shape != want:
            raise ValueError(f"expected coefficients of shape {want}, got {c.shape}")
        full = np.zeros((self.mesh.nglob,) + want[:3])
        full[self._slice] = np.moveaxis(c, -1, 0)
        return _ppoly(self.mesh, full[self.mesh.gmap], self._elements)

    def to_field(self, coeffs, *, character: Character = SCALAR,
                 name: str | None = None) -> AnalyticField:
        """A coefficient sample as an `AnalyticField` on [r1, r2],
        synthesised from its harmonics at every point asked for."""
        pp = self.coefficient_functions(coeffs)
        L = self.lmax

        def fn(r, theta, phi):
            r, theta, phi = np.broadcast_arrays(np.asarray(r, dtype=float),
                                                np.asarray(theta, dtype=float),
                                                np.asarray(phi, dtype=float))
            c = pp(np.clip(r, *self.interval))
            Y = real_harmonics(L, theta, phi)
            return np.einsum("...slm,slm...->...", c, Y)

        return AnalyticField(self.interval, fn, character=character, name=name)

    def __repr__(self) -> str:
        r1, r2 = self.interval
        return (f"SphericalGRF([{r1:g}, {r2:g}], nu={self.nu:g}, "
                f"lmax={self.lmax}, {int(self.nmodes.sum())} radial modes "
                f"on {self.r.size} nodes)")


# -- layered fields: direct sums over a skeleton ----------------------------------

class LayeredGRF:
    """Independent radial Gaussian fields per layer of a skeleton.

    Every included layer carries its own `RadialGRF` on its interval
    (each padded beyond the layer's boundaries; the pads are fictitious
    and may overlap neighbours, reach past the surface, or clamp at the
    centre); the direct sum is generically discontinuous at layer
    boundaries.  `nu`, `lam`, `sigma` and `drmax` are shared across
    layers when scalars or callables, or per-layer when sequences of
    length `skeleton.nlayers`.  `layers` restricts the sum to the given
    layer indices; excluded layers contribute exactly zero.  `sample()`
    returns one `RadialField` per layer, ready for `Model.with_field`.
    A Geometry or Model is accepted in place of a skeleton.
    """

    def __init__(self, skeleton, nu, lam, *, sigma=1.0, layers=None,
                 ngll: int = 5, drmax=None, pad_factor: float = 1.5,
                 robin=None, tol: float = 1e-6, name: str | None = None) -> None:
        sk = getattr(skeleton, "skeleton", skeleton)
        if not isinstance(sk, Skeleton):
            raise TypeError("expected a Skeleton, a Geometry or a Model")
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
                    raise ValueError(f"per-layer sequences must have length {nl}")
                return spec[i]
            return spec

        self._grfs = {
            i: RadialGRF(*sk.interval(i), per_layer(nu, i), per_layer(lam, i),
                         sigma=per_layer(sigma, i), ngll=ngll,
                         drmax=per_layer(drmax, i), pad_factor=pad_factor,
                         robin=robin, tol=tol)
            for i in idx}

    def __getitem__(self, i: int) -> RadialGRF:
        """The `RadialGRF` of layer i (KeyError if excluded)."""
        return self._grfs[i]

    def sample(self, *, rng=None, character: Character = SCALAR
               ) -> tuple[RadialField, ...]:
        """One sample as a `RadialField` per layer of the skeleton, zero
        on the excluded layers."""
        rng = np.random.default_rng(rng)
        out = []
        for i in range(self.skeleton.nlayers):
            iv = self.skeleton.interval(i)
            if i in self._grfs:
                g = self._grfs[i]
                out.append(g.to_field(g.sample(rng=rng), character=character,
                                      name=self.name))
            else:
                out.append(RadialField(iv, constant_layer(0.0, iv),
                                       character=character, name=self.name))
        return tuple(out)

    def __repr__(self) -> str:
        return (f"LayeredGRF({len(self._grfs)}/{self.skeleton.nlayers} layers "
                f"active: {list(self.layers)})")
