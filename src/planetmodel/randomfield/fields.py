"""Whittle-Matern Gaussian random fields on annuli, balls and layers.

The SPDE construction: a Matern-type field solves A^beta u = white
noise with A = 1 - div(lambda^2 grad), so its covariance is
C = A^-2 beta, and matching the Matern smoothness nu requires ::

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
with the M-orthonormal eigenpairs (Theta_l, Phi_l) of A_l, held as the
`SpectralBasis` of each degree, discrete white noise is Phi z with
z ~ N(0, I), so u_lm(r) = Phi_l Theta_l^-beta z_lm with z_lm iid over
(l, m).  Modes are truncated where their contribution to the total
variance falls below a relative tolerance, and for `SphericalGRF` with
`lmax=None` the degree sum is truncated the same way.  The marginal
standard deviation is then normalised exactly: the raw pointwise
variance of the truncated expansion ::

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

**Coefficients first.**  A sample of `RadialGRF` or `SphericalGRF` is
the coefficient vector, in the order of its basis (a `SpectralBasis`, a
`SphericalBasis`), with independent entries of standard deviation
`coefficient_std`, theta^-beta.  The field of a vector is its synthesis
times `scale`, the factor sigma / sigma_raw of radius above.  The views
of a vector are derived from it: `synthesise` (the nodal values of a
field of radius; of a shell the coefficient functions with respect to
the real orthonormal harmonics of `planetmodel.harmonics`, with a
trailing radial axis over the physical nodes `r`), `to_field` (a
`RadialField`, or a `HarmonicField` on the shell) and, of a shell,
`on_grid` (values on an angular grid of a shipped kind, through ducc0's
grid transforms).  `LayeredGRF` holds an independent `RadialGRF` on each
chosen layer of a skeleton: its sample is their vectors end to end, and
`to_fields` gives one `RadialField` per layer.  All fields are real and
zero-mean.

**Batches.**  `sample(size=k)` draws k vectors, shape (k, size), the
k successive draws of the generator.  Every method that takes a vector
takes such a batch too: an array comes back with the batch axis leading,
and a field (`to_layer`, `to_field`, `to_fields`) as a tuple over the
batch.

The samplers of flat layers (`planetmodel.cartesian.randomfield`) have
the same methods with the same meanings and the same arguments.  One
difference is the geometry's: a sampler of a box holds its lateral grid,
which is also its band, while a shell's band is `lmax` and any angular
grid of that band or more will do, so `on_grid` and `sample_grid` of
`SphericalGRF` are given the grid.
"""
from __future__ import annotations

import warnings
from collections.abc import Callable, Iterable, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly

from ..character import SCALAR, Character
from ..expansions import HarmonicField
from ..fields import RadialField
from ..harmonics import packing, real_harmonics, synthesise_grid
from ..layerfunction import PolynomialLayer, constant_layer
from ..mesh1d import Mesh1D
from ..skeleton import Skeleton
from .basis import SpectralBasis, SphericalBasis
from .mesh import Restriction, matern_reach, padded_mesh, restriction
from .operator import Robin, SeparableOperatorFamily

if TYPE_CHECKING:
    from ..geometry import Geometry
    from ..model import Model
    from ..sampling import AngularGrid

__all__ = ["RadialGRF", "SphericalGRF", "LayeredGRF", "Profile",
           "profile_function", "sampler_mesh", "robin_spec", "matern_exponent",
           "marginal_scale", "coefficient_vectors", "nodal_ppoly",
           "layer_indices", "per_layer"]

#: A parameter of a field that may vary with radius: a number, or a
#: callable of the radius.
type Profile = float | Callable[[np.ndarray], ArrayLike]


# -- what every sampler shares ------------------------------------------------

def profile_function(x: Profile) -> Callable[[np.ndarray], np.ndarray]:
    """A `Profile`, a number or a callable of the layered coordinate, as a
    vectorised callable returning floats of its argument's shape."""
    if callable(x):
        return lambda r: np.broadcast_to(np.asarray(x(r), dtype=float),
                                         np.shape(r)).copy()
    v = float(x)
    return lambda r: np.full(np.shape(r), v)


def _length_scale(lam_fun: Callable[[np.ndarray], np.ndarray], r1: float,
                  r2: float) -> tuple[float, float, float]:
    """The length scale at r1, at r2 and at its smallest on [r1, r2], from
    a sampling of the interval; refuses one that is not positive and finite."""
    lam_s = lam_fun(np.linspace(r1, r2, 257))
    if np.any(lam_s <= 0.0) or not np.all(np.isfinite(lam_s)):
        raise ValueError("lambda must be positive and finite on [r1, r2]")
    return float(lam_s[0]), float(lam_s[-1]), float(lam_s.min())


def sampler_mesh(r1: float, r2: float, nu: float,
                 lam_fun: Callable[[np.ndarray], np.ndarray], ngll: int,
                 drmax: float | None, pad_factor: float, *, weight: str = "r2"
                 ) -> tuple[Mesh1D, float, Restriction]:
    """The samplers' computational mesh for a physical interval [r1, r2]:
    each side padded by `pad_factor` Matern reaches of the length scale at
    that boundary, elements no wider than `drmax`, by default half the
    smallest length scale.  `weight` is the measure the mesh is for, as
    `padded_mesh` takes it: under "r2" and "r" the lower pad stops at
    r = 0, under "one" it runs below r1 wherever that is.  Returns the
    mesh, the drmax used and the restriction to [r1, r2]."""
    lam1, lam2, lam_min = _length_scale(lam_fun, r1, r2)
    if drmax is None:
        drmax = lam_min / 2.0
    pad = (pad_factor * matern_reach(nu, lam1), pad_factor * matern_reach(nu, lam2))
    mesh = padded_mesh(r1, r2, pad=pad, weight=weight, ngll=ngll, drmax=drmax)
    return mesh, float(drmax), restriction(mesh, r1, r2)


def robin_spec(robin: Robin | str, lam_fun: Callable[[np.ndarray], np.ndarray],
               r1: float, r2: float) -> Robin:
    """A sampler's `robin` as a family takes it: "auto" resolved into
    gamma = 1 / lambda at each boundary, anything else as it stands."""
    if isinstance(robin, str):
        if robin != "auto":
            raise ValueError("robin must be None, 'auto', a number or a pair")
        return (1.0 / float(lam_fun(r1)), 1.0 / float(lam_fun(r2)))
    return robin


def matern_exponent(nu: float, d: int) -> float:
    """The beta of A^beta u = white noise that gives a field of `d`
    dimensions the Matern smoothness `nu`: 2 beta = nu + d / 2."""
    return 0.5 * (float(nu) + 0.5 * d)


def marginal_scale(sigma: Profile, r: np.ndarray, var_raw: np.ndarray
                   ) -> tuple[np.ndarray, np.ndarray]:
    """The exact normalisation of the marginal standard deviation: the
    wanted `sigma` at the nodes `r`, and the factor sigma / sqrt(var_raw)
    by which the nodal values of the truncated expansion, of raw
    pointwise variance `var_raw` there, are scaled so that their
    standard deviation is sigma to rounding; both read-only.  Refuses a
    raw variance that vanishes at a node and a negative sigma."""
    if np.any(var_raw <= 0.0):
        raise RuntimeError("vanishing raw variance at a physical node")
    sig = profile_function(sigma)(r)
    if np.any(sig < 0.0):
        raise ValueError("sigma must be non-negative")
    scale = sig / np.sqrt(var_raw)
    sig.setflags(write=False)
    scale.setflags(write=False)
    return sig, scale


def coefficient_vectors(vector: ArrayLike, size: int) -> np.ndarray:
    """What a sampler's method is given as coefficients, as floats: one
    vector, shape (size,), or a batch of them, shape (k, size), as
    `sample(size=k)` draws it.  Any other shape is refused."""
    c = np.asarray(vector, dtype=float)
    if c.ndim not in (1, 2) or c.shape[-1] != size:
        raise ValueError(f"expected a coefficient vector of length {size}, or a "
                         f"batch of them of shape (k, {size}), got shape {c.shape}")
    return c


def nodal_ppoly(mesh: Mesh1D, restriction: Restriction, nodal: np.ndarray) -> PPoly:
    """The piecewise polynomial of the layered coordinate on the physical
    interval through values at its nodes, leading axis the nodes: on
    each element the polynomial through the values at its nodes, its
    values of shape nodal.shape[1:]."""
    full = np.zeros((mesh.nglob,) + nodal.shape[1:])
    full[restriction.nodes] = nodal
    return mesh.to_ppoly(full[mesh.gmap], elements=restriction.elements)


def layer_indices(layers: Iterable[int] | None, nlayers: int) -> tuple[int, ...]:
    """The layers a layered sampler includes, ascending and without
    repeats: every one of `nlayers` when `layers` is None, a negative
    index counting from the last; one out of range is refused."""
    idx = range(nlayers) if layers is None else layers
    idx = tuple(sorted({i if i >= 0 else i + nlayers for i in idx}))
    if idx and not 0 <= idx[0] <= idx[-1] < nlayers:
        raise IndexError("layer index out of range")
    return idx


def per_layer(spec: Any, i: int, nlayers: int) -> Any:
    """A layered sampler's parameter for layer `i`: the entry of a list or
    tuple, which must have one for each of `nlayers`, or the parameter
    itself, shared by the layers."""
    if isinstance(spec, (list, tuple)):
        if len(spec) != nlayers:
            raise ValueError(f"per-layer sequences must have length {nlayers}")
        return spec[i]
    return spec


# -- fields of radius alone -------------------------------------------------

class RadialGRF:
    """A real, zero-mean Gaussian random field of radius on [r1, r2], in
    coefficients.

    Matern-type with smoothness `nu` > 0, length scale `lam` (a scalar
    or a callable of radius) and marginal standard deviation `sigma`
    (likewise), via C = A^-2 beta with 2 beta = nu + 1/2; r1 = 0 gives
    a field of the whole ball.  `ngll`, `drmax`, `pad_factor`, `robin`
    and the truncation tolerance `tol` (the omitted share of the
    variance trace) tune the discretisation.

    A sample is a vector of coefficients in `basis`, a `SpectralBasis`,
    independent with standard deviations `coefficient_std`; the field
    of a vector is `scale` times its synthesis, whose standard deviation
    at the nodes `r` is `sigma` to rounding.  Wherever a vector is taken
    a batch of them, shape (k, nmodes), is taken too.  Attributes:
    `interval`, `nu`, `beta`, `mesh`, `drmax`, `restriction`, `family`,
    `basis`, `r` (the physical nodes), `sigma`, `scale` and
    `coefficient_std`, the last three read-only.
    """

    def __init__(self, r1: float, r2: float, nu: float, lam: Profile, *,
                 sigma: Profile = 1.0, ngll: int = 5, drmax: float | None = None,
                 pad_factor: float = 1.5, robin: Robin | str = None,
                 tol: float = 1e-6) -> None:
        r1, r2 = float(r1), float(r2)
        if not 0.0 <= r1 < r2:
            raise ValueError("need 0 <= r1 < r2")
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        self.interval = (r1, r2)
        self.nu = float(nu)
        self.beta = matern_exponent(self.nu, 1)

        lam_f = profile_function(lam)
        clipped = lambda r: lam_f(np.clip(r, r1, r2))
        self.mesh, self.drmax, self.restriction = sampler_mesh(
            r1, r2, self.nu, clipped, ngll, drmax, pad_factor)
        robin = robin_spec(robin, clipped, r1, r2)
        self.family = SeparableOperatorFamily(
            self.mesh, kappa=lambda r: clipped(r) ** 2, weight="r2", robin=robin)

        self.r = self.restriction.r

        self.basis = SpectralBasis(self.family, 0, restrict=self.restriction,
                                   power=2.0 * self.beta, tol=tol)
        self.coefficient_std = self.basis.theta ** (-self.beta)
        self.coefficient_std.setflags(write=False)
        self.sigma, self.scale = marginal_scale(
            sigma, self.r, self.basis.pointwise_variance(2.0 * self.beta))

    @property
    def nmodes(self) -> int:
        """The number of coefficients: of Karhunen-Loeve modes kept."""
        return self.basis.nmodes

    @property
    def size(self) -> int:
        """The length of the coefficient vector, `nmodes`."""
        return self.basis.nmodes

    def sample(self, *, rng: Any = None, size: int | None = None) -> np.ndarray:
        """A coefficient vector (or `size` of them): shape (nmodes,) or
        (size, nmodes), a batch being `size` successive draws from the
        generator.  `rng` is anything `np.random.default_rng` accepts."""
        rng = np.random.default_rng(rng)
        shape = (self.nmodes,) if size is None else (int(size), self.nmodes)
        return rng.standard_normal(shape) * self.coefficient_std

    def synthesise(self, vector: ArrayLike) -> np.ndarray:
        """The field of a coefficient vector at the physical nodes, shape
        (r.size,), `scale` applied; of a batch, shape (k, r.size)."""
        c = coefficient_vectors(vector, self.nmodes)
        return (c @ self.basis.modes()[self.restriction.nodes].T) * self.scale

    def factor(self) -> np.ndarray:
        """The covariance factor B, shape (r.size, nmodes): the field at
        the nodes of standard normal z is B z, and its covariance B B^T."""
        return self.synthesise(np.diag(self.coefficient_std)).T

    def covariance(self) -> np.ndarray:
        """Dense covariance of the field at the nodes (diagonal sigma^2)."""
        B = self.factor()
        return B @ B.T

    def std(self) -> np.ndarray:
        """Marginal standard deviation at the nodes (exact by design)."""
        return self.sigma.copy()

    def to_layer(self, vector: ArrayLike
                 ) -> PolynomialLayer | tuple[PolynomialLayer, ...]:
        """The field of a coefficient vector as an exact `PolynomialLayer`
        on [r1, r2], through its nodal values; of a batch, one for each
        of its vectors."""
        c = coefficient_vectors(vector, self.nmodes)
        if c.ndim == 2:
            return tuple(self.to_layer(v) for v in c)
        return PolynomialLayer(self.interval, nodal_ppoly(
            self.mesh, self.restriction, self.synthesise(c)))

    def to_field(self, vector: ArrayLike, *, character: Character = SCALAR,
                 name: str | None = None) -> RadialField | tuple[RadialField, ...]:
        """The field of a coefficient vector as a `RadialField` on
        [r1, r2]; of a batch, one for each of its vectors."""
        c = coefficient_vectors(vector, self.nmodes)
        if c.ndim == 2:
            return tuple(self.to_field(v, character=character, name=name)
                         for v in c)
        return RadialField(self.interval, self.to_layer(c), character=character,
                           name=name)

    def __repr__(self) -> str:
        r1, r2 = self.interval
        return (f"RadialGRF([{r1:g}, {r2:g}], nu={self.nu:g}, "
                f"{self.nmodes} modes on {self.r.size} nodes)")


# -- scalar fields of the annulus or ball ---------------------------------------

class SphericalGRF:
    """A real, zero-mean Gaussian random field of a spherical annulus (or
    ball, r1 = 0), in coefficients.

    Matern-type with C = A^-2 beta, 2 beta = nu + 3/2; `lam` is the
    length scale (scalar or callable of radius), `lam_h` an optional
    distinct horizontal scale, `sigma` the marginal standard deviation.
    `lmax=None` chooses the degree cut automatically: degrees are added
    until their variance trace stays below `tol` times the running
    total for two degrees in a row (`lmax_cap` bounds the scan, with a
    warning if it bites).  `ngll`, `drmax`, `pad_factor` and `robin`
    are those of `RadialGRF`; `tol` is also the share of its variance
    trace each degree's truncation omits.

    A sample is a vector of coefficients in `basis`, a `SphericalBasis`
    (by degree, the cosine harmonics of orders 0 ... l and then the
    sine ones of orders 1 ... l, the radial modes ascending within each),
    independent with standard deviations `coefficient_std`; the field
    of a vector is `scale` (a factor of radius) times its synthesis,
    whose standard deviation at every point of a sphere through a node
    of `r` is `sigma` to rounding.  Wherever a vector is taken a batch
    of them, shape (k, size), is taken too.  Attributes: `interval`,
    `nu`, `beta`, `lmax`, `mesh`, `drmax`, `restriction`, `family`,
    `basis`, `nmodes` (the modes kept at each degree), `r` (the physical
    nodes), `sigma`, `scale` and `coefficient_std`, the last three
    read-only.
    """

    def __init__(self, r1: float, r2: float, nu: float, lam: Profile, *,
                 sigma: Profile = 1.0, lmax: int | None = None,
                 lam_h: Profile | None = None, ngll: int = 5,
                 drmax: float | None = None, pad_factor: float = 1.5,
                 robin: Robin | str = None, tol: float = 1e-6,
                 lmax_cap: int = 512) -> None:
        r1, r2 = float(r1), float(r2)
        if not 0.0 <= r1 < r2:
            raise ValueError("need 0 <= r1 < r2")
        if nu <= 0.0:
            raise ValueError("nu must be positive")
        self.interval = (r1, r2)
        self.nu = float(nu)
        self.beta = matern_exponent(self.nu, 3)
        p = 2.0 * self.beta

        lam_f = profile_function(lam)
        clip_r = lambda r: lam_f(np.clip(r, r1, r2))
        lamh_f = clip_r if lam_h is None else (
            lambda r, f=profile_function(lam_h): f(np.clip(r, r1, r2)))
        self.mesh, self.drmax, self.restriction = sampler_mesh(
            r1, r2, self.nu, clip_r, ngll, drmax, pad_factor)
        robin = robin_spec(robin, clip_r, r1, r2)
        self.family = SeparableOperatorFamily(
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

        self.r = self.restriction.r

        self.basis = SphericalBasis([
            SpectralBasis(self.family, l, restrict=self.restriction, power=p, tol=tol)
            for l in range(self.lmax + 1)])
        self.nmodes = self.basis.nmodes
        self.coefficient_std = self.basis.theta ** (-self.beta)
        self.coefficient_std.setflags(write=False)

        var_raw = np.zeros(self.r.size)
        for b in self.basis:
            var_raw += (2 * b.label + 1) / (4.0 * np.pi) * b.pointwise_variance(p)
        self.sigma, self.scale = marginal_scale(sigma, self.r, var_raw)

        #: for each entry of the vector, the normal of a draw that it takes
        draw, start = [], 0
        for l, n in enumerate(self.nmodes):
            count = (2 * l + 1) * int(n)
            draw.append(start + np.arange(count).reshape(int(n), 2 * l + 1).T.ravel())
            start += count
        self._draw = np.concatenate(draw)
        self._interpolant: PPoly | None = None

    @property
    def degrees(self) -> np.ndarray:
        """The spherical-harmonic degrees included, 0 ... lmax."""
        return np.arange(self.lmax + 1)

    @property
    def size(self) -> int:
        """The length of the coefficient vector."""
        return self.basis.size

    def sample(self, *, rng: Any = None, size: int | None = None) -> np.ndarray:
        """A coefficient vector (or `size` of them) in the order of
        `basis`: shape (basis.size,) or (size, basis.size), a batch being
        `size` successive draws from the generator.  `rng` is anything
        `np.random.default_rng` accepts.  A draw takes the generator's
        normals degree by degree and, within a degree, mode by mode
        across its harmonics."""
        rng = np.random.default_rng(rng)
        shape = (self.size,) if size is None else (int(size), self.size)
        return rng.standard_normal(shape)[..., self._draw] * self.coefficient_std

    def synthesise(self, vector: ArrayLike) -> np.ndarray:
        """The coefficient functions of the field of a coefficient vector
        at the physical nodes: shape (2, lmax + 1, lmax + 1, r.size) in
        the layout of `planetmodel.harmonics`, the radial axis last,
        `scale` applied; of a batch, that with a leading axis over it.
        For a ball, those of every l >= 1 vanish at r = 0."""
        c = coefficient_vectors(vector, self.size)
        rows = c.reshape(-1, self.size)
        L = self.lmax
        out = np.empty((rows.shape[0], 2, L + 1, L + 1, self.r.size))
        for k, row in enumerate(rows):
            out[k] = self.basis.synthesise(row)
        out *= self.scale
        return out.reshape(c.shape[:-1] + out.shape[1:])

    def on_grid(self, vector: ArrayLike, grid: AngularGrid) -> np.ndarray:
        """The field of a coefficient vector at the nodes `r` and the
        nodes of an angular grid of a shipped kind and band at least
        `lmax`: shape (r.size, ntheta, nphi), through `synthesise_grid`;
        of a batch, that with a leading axis over it.  The grid is an
        argument because the sampler holds a band, `lmax`, and no grid."""
        c = self.synthesise(vector)
        return synthesise_grid(np.moveaxis(c, 0, 3) if c.ndim == 5 else c, grid)

    def sample_grid(self, grid: AngularGrid, *, rng: Any = None,
                    size: int | None = None) -> np.ndarray:
        """One sample (or `size` of them) at the nodes `r` and the nodes
        of an angular grid, `on_grid` of `sample`: shape
        (r.size, ntheta, nphi), or that with a leading axis of `size`."""
        return self.on_grid(self.sample(rng=rng, size=size), grid)

    def factor(self, l: int) -> np.ndarray:
        """The covariance factor of degree l, shape (r.size, nmodes[l]),
        with the marginal normalisation applied: the coefficient function
        of every harmonic of degree l is B_l z with z standard normal."""
        b = self.basis[l]
        return (self.scale[:, None] * b.modes()[self.restriction.nodes]
                * b.theta[None, :] ** (-self.beta))

    def variance(self) -> np.ndarray:
        """Pointwise variance sigma(r)^2 of the truncated field at the
        nodes (exact)."""
        return self.sigma ** 2

    def std(self) -> np.ndarray:
        """Pointwise standard deviation sigma(r) at the nodes (exact by
        design)."""
        return self.sigma.copy()

    def coefficient_functions(self, vector: ArrayLike) -> PPoly:
        """The coefficient functions of the field of a coefficient vector
        as one `PPoly` of radius on [r1, r2] whose values have shape
        (2, lmax + 1, lmax + 1), or (k, 2, lmax + 1, lmax + 1) of a batch:
        the polynomial of each element through `synthesise`'s values at
        its nodes."""
        return nodal_ppoly(self.mesh, self.restriction,
                           np.moveaxis(self.synthesise(vector), -1, 0))

    def to_field(self, vector: ArrayLike, *, character: Character = SCALAR,
                 name: str | None = None
                 ) -> HarmonicField | tuple[HarmonicField, ...]:
        """The field of a coefficient vector as a `HarmonicField` on
        [r1, r2], its coefficients the piecewise polynomials of
        `coefficient_functions`; of a batch, one for each of its vectors."""
        c = coefficient_vectors(vector, self.size)
        if c.ndim == 2:
            return tuple(self.to_field(v, character=character, name=name)
                         for v in c)
        return HarmonicField(self.interval, self.coefficient_functions(c),
                             character=character, name=name)

    def _at_radii(self, r: np.ndarray) -> np.ndarray:
        """The weights (r.size, self.r.size) that read the polynomial
        through nodal values at radii of the interval."""
        if self._interpolant is None:
            self._interpolant = nodal_ppoly(self.mesh, self.restriction,
                                            np.eye(self.r.size))
        return self._interpolant(r)

    def covariance(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """The covariance of the field (as `to_field` gives it) between
        points `a` and points `b`, each of shape (..., 3) holding
        (r, theta, phi), radii within the interval and colatitudes in
        [0, pi]: shape a.shape[:-1] + b.shape[:-1].

        Exact, from the basis: the sum over the coefficients of their
        variance times the synthesised mode at the two points, the
        harmonics by `harmonics.real_harmonics`.  It costs the harmonics
        to `lmax` for each point and is meant for a few points; the
        marginal variance at the nodes is `variance`.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        for pts in (a, b):
            if pts.ndim == 0 or pts.shape[-1] != 3:
                raise ValueError(f"points have 3 coordinates each, (r, theta, phi), "
                                 f"along the last axis; got shape {pts.shape}")
        r1, r2 = self.interval
        fa, fb = a.reshape(-1, 3), b.reshape(-1, 3)
        slack = 1e-12 * (r2 - r1)
        for pts in (fa, fb):
            if np.any(pts[:, 0] < r1 - slack) or np.any(pts[:, 0] > r2 + slack):
                raise ValueError(f"radii must lie in the physical interval "
                                 f"[{r1:g}, {r2:g}]")
        Wa = self._at_radii(np.clip(fa[:, 0], r1, r2))
        Wb = self._at_radii(np.clip(fb[:, 0], r1, r2))
        harmonic = packing(self.lmax)
        Ya = real_harmonics(self.lmax, fa[:, 1], fa[:, 2])[harmonic]
        Yb = real_harmonics(self.lmax, fb[:, 1], fb[:, 2])[harmonic]
        out = np.zeros((fa.shape[0], fb.shape[0]))
        for l in range(self.lmax + 1):
            B = self.factor(l)
            which = slice(l * l, (l + 1) * (l + 1))
            out += ((Wa @ B) @ (Wb @ B).T) * (Ya[which].T @ Yb[which])
        return out.reshape(a.shape[:-1] + b.shape[:-1])

    def __repr__(self) -> str:
        r1, r2 = self.interval
        return (f"SphericalGRF([{r1:g}, {r2:g}], nu={self.nu:g}, "
                f"lmax={self.lmax}, {self.size} coefficients on {self.r.size} "
                f"nodes)")


# -- layered fields: direct sums over a skeleton ----------------------------------

class LayeredGRF:
    """Independent radial Gaussian fields on the layers of a skeleton, in
    coefficients.

    Every included layer carries its own `RadialGRF` on its interval
    (each padded beyond the layer's boundaries; the pads are fictitious
    and may overlap neighbours, reach past the surface, or clamp at the
    centre); the direct sum is generically discontinuous at layer
    boundaries.  `nu`, `lam`, `sigma` and `drmax` are shared across
    layers when scalars or callables, or per-layer when sequences of
    length `skeleton.nlayers`.  `layers` restricts the sum to the given
    layer indices.  A Geometry or Model is accepted in place of a
    skeleton.

    A sample is one vector, the coefficient vectors of the included
    layers in ascending order of layer (`block`), of length `size`;
    `to_fields` gives the field of a vector on every layer of the
    skeleton, zero on the excluded ones, ready for `Model.with_field`.
    Wherever a vector is taken a batch of them, shape (k, size), is
    taken too.
    """

    def __init__(self, skeleton: Skeleton | Geometry | Model,
                 nu: float | Sequence[float], lam: Profile | Sequence[Profile], *,
                 sigma: Profile | Sequence[Profile] = 1.0,
                 layers: Iterable[int] | None = None, ngll: int = 5,
                 drmax: float | Sequence[float | None] | None = None,
                 pad_factor: float = 1.5, robin: Robin | str = None,
                 tol: float = 1e-6, name: str | None = None) -> None:
        sk = getattr(skeleton, "skeleton", skeleton)
        if not isinstance(sk, Skeleton):
            raise TypeError("expected a Skeleton, a Geometry or a Model")
        self.skeleton = sk
        self.name = name
        nl = sk.nlayers
        self.layers = layer_indices(layers, nl)
        self._grfs = {
            i: RadialGRF(*sk.interval(i), per_layer(nu, i, nl), per_layer(lam, i, nl),
                         sigma=per_layer(sigma, i, nl), ngll=ngll,
                         drmax=per_layer(drmax, i, nl), pad_factor=pad_factor,
                         robin=robin, tol=tol)
            for i in self.layers}
        sizes = [self._grfs[i].size for i in self.layers]
        self._starts = dict(zip(self.layers, np.cumsum([0] + sizes[:-1]).tolist()))
        self.size = int(sum(sizes))

    def __getitem__(self, i: int) -> RadialGRF:
        """The `RadialGRF` of layer i (KeyError if excluded)."""
        return self._grfs[i]

    def block(self, i: int) -> slice:
        """Where the coefficient vector of layer i sits in the vector
        (KeyError if excluded)."""
        start = self._starts[i]
        return slice(start, start + self._grfs[i].size)

    def sample(self, *, rng: Any = None, size: int | None = None) -> np.ndarray:
        """One coefficient vector (or `size` of them), shape (self.size,)
        or (size, self.size): the layers' samples in ascending order of
        layer, drawn in that order from the generator, a batch being
        `size` successive draws.  `rng` is anything
        `np.random.default_rng` accepts."""
        rng = np.random.default_rng(rng)
        out = np.empty((1 if size is None else int(size), self.size))
        for row in out:
            for i in self.layers:
                row[self.block(i)] = self._grfs[i].sample(rng=rng)
        return out[0] if size is None else out

    def to_fields(self, vector: ArrayLike, *, character: Character = SCALAR
                  ) -> tuple[RadialField, ...] | tuple[tuple[RadialField, ...], ...]:
        """The field of a coefficient vector on every layer of the
        skeleton: a `RadialField` each, zero on the excluded layers.  Of
        a batch, one such tuple for each of its vectors."""
        v = coefficient_vectors(vector, self.size)
        if v.ndim == 2:
            return tuple(self.to_fields(row, character=character) for row in v)
        out = []
        for i in range(self.skeleton.nlayers):
            iv = self.skeleton.interval(i)
            if i in self._grfs:
                out.append(self._grfs[i].to_field(v[self.block(i)],
                                                  character=character,
                                                  name=self.name))
            else:
                out.append(RadialField(iv, constant_layer(iv, 0.0),
                                       character=character, name=self.name))
        return tuple(out)

    def __repr__(self) -> str:
        return (f"LayeredGRF({len(self._grfs)}/{self.skeleton.nlayers} layers "
                f"active: {list(self.layers)})")
