"""Whittle-Matern Gaussian random fields of flat layers, in coefficients.

The flat counterpart of `planetmodel.randomfield`'s samplers.  A
Matern-type field solves A^beta u = white noise with
A = 1 - div(lambda^2 grad), so that its covariance is C = A^-2 beta
with 2 beta = nu + d / 2.  Over a periodic lateral box A separates in
the real Fourier basis of a `fourier.Transform`: on the component of
wavevector k it is the member of lateral eigenvalue `|k|^2` of the flat
`SeparableOperatorFamily` (the plain measure in depth, the lateral
profile one, indexed by value)::

    A_k f = f - (kappa f')' + |k|^2 kappa_h f,

with kappa = lambda^2 along depth and kappa_h = lambda_h^2 across it,
each a function of depth alone.  Depth is the layered coordinate,
positive downward.

**The basis.**  `LayerBasis` holds one `SpectralBasis` for each
distinct `|k|^2` of the transform and the map from every packed Fourier
component to its own: components of equal `|k|^2` share one member of
the family, hence one eigenproblem and one entry of its cache.  The
coordinates of a field are the coefficients of each component's
function of depth in the eigenfunctions of its member, as one vector:
the components in the order of the transform's packing, and within a
component the modes in ascending theta.  The number of modes differs
between members, so the vector is ragged; `block` says where a
component sits.  In these coordinates every function of A is
diagonal, as on the sphere.

**Truncation.**  Every coefficient of a field of covariance A^-p has
variance theta^-p, and every component the same mean square over the
box, 1 / V, so the fewest coefficients for a given share of the
variance are those of smallest theta over the whole basis.
`LayerBasis.of` keeps them: the fewest whose omitted share of
sum theta^-p over every component and mode is at most `tol`, and never
fewer than one mode of a member.  Since `theta >= 1 + |k|^2 min kappa_h`,
the members of short wavelength keep few modes, and only the kept
eigenvectors are computed and held.

**Coefficients first.**  A sample of `DepthGRF` or `BoxGRF` is the
coefficient vector, in the order of its basis, with independent
entries of standard deviation `coefficient_std`, theta^-beta.  The
field of a vector is its synthesis times a factor of depth, `scale`,
that makes the marginal standard deviation exact: the raw pointwise
variance of the truncated expansion at a node of the lateral grid::

    sigma_raw^2(depth) = 1 / V sum_k sum_j theta_kj^-2 beta phi_kj(depth)^2,

the sum over every packed component (each of grid mean square 1 / V,
which is what the packing's measures state) in place of the sphere's
(2 l + 1) / (4 pi) over degrees, is computed at the depth nodes and
`scale` is sigma / sigma_raw there.  The views of a vector are derived
from it: `synthesise` (the functions of depth at the nodes),
`to_field` (a `FourierField` or a `DepthField`), `on_grid` (values at
the nodes of the lateral grid).  Between the nodes of the lateral grid
the variance is as exact but for the Nyquist cosines of an even axis,
whose square is not constant off the grid.

**Batches.**  `sample(size=k)` draws k vectors, shape (k, size), the
k successive draws of the generator.  Every method that takes a vector
takes such a batch too: an array comes back with the batch axis leading,
and a field (`to_layer`, `to_field`, `to_fields`) as a tuple over the
batch.

**Padding.**  The mesh in depth extends beyond both ends of the
physical interval by `pad_factor` times the Matern reach
sqrt(8 nu) lambda at that end, above the free surface (negative depth)
as below the base, with lambda continued constantly; the plain measure
has no special point.  Its ends are Neumann unless `robin` is given
("auto" for gamma = 1 / lambda there, a number or a pair).  Laterally
the field is periodic over the period of its grid's domain: a model's
box, or a `Box` or `fourier.PaddedDomain` with padding where a domain
proper is wanted clear of its periodic images.

`DepthGRF` is a field of depth alone (d = 1), `BoxGRF` a field of a
box (d = 3) or of a rectangle (d = 2: one lateral axis, the same at
every y), and `LayeredGRF` independent fields on the layers of a flat
skeleton.  The horizontal scale is one, `lam_h`; distinct scales along
x and y are not built.  All fields are real and zero-mean.  The
methods are those of `randomfield.RadialGRF`, `SphericalGRF` and
`LayeredGRF`, with the same meanings and arguments, but that a sampler
of a box holds its lateral grid, so `on_grid` and `sample_grid` take
none.

::

    grf = cartesian.BoxGRF(0.0, 50.0, grid, 1.5, 4.0, sigma=0.02)
    c = grf.sample(rng=rng)                  # the coefficient vector
    grf.to_field(c)                          # a FourierField
    grf.on_grid(c)                           # depth nodes x grid nodes
    grf.on_grid(grf.sample(rng=rng, size=8)) # eight of them, the batch first
    grf.basis.block(grid.transform.component_of((1, 0)))
"""
from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly

from ..character import SCALAR, Character
from ..fourier import PaddedDomain, Transform
from ..layerfunction import PolynomialLayer, constant_layer
from ..randomfield.basis import SpectralBasis
from ..randomfield.fields import (Profile, coefficient_vectors, layer_indices,
                                  marginal_scale, matern_exponent, nodal_ppoly,
                                  per_layer, profile_function, robin_spec,
                                  sampler_mesh)
from ..randomfield.mesh import Restriction
from ..randomfield.operator import CACHE_BYTES, Robin, SeparableOperatorFamily
from ..skeleton import Skeleton
from .domains import Box, Rectangle
from .expansions import FourierField
from .fields import DepthField
from .grid import Grid

if TYPE_CHECKING:
    from .geometry import Geometry
    from .model import Model

__all__ = ["LayerBasis", "DepthGRF", "BoxGRF", "LayeredGRF", "distinct_eigenvalues",
           "EIGENVALUE_RTOL"]

#: How closely, relative to the larger, two lateral eigenvalues of a
#: transform must agree to be one: wavevectors of equal length reach
#: their `|k|^2` by different sums, which differ in rounding.
EIGENVALUE_RTOL = 1e-12


def distinct_eigenvalues(transform: Transform, *, rtol: float = EIGENVALUE_RTOL
                         ) -> tuple[np.ndarray, np.ndarray]:
    """The distinct lateral eigenvalues `|k|^2` of a transform's components,
    ascending, and the index into them of each packed component.

    Values within `rtol` of one another are one, represented by the
    smallest of them, so that every component of one `|k|^2` names its
    member of a family by the same float.
    """
    lam = transform.laplacian_eigenvalues
    order = np.argsort(lam, kind="stable")
    ordered = lam[order]
    new = np.concatenate(([True], np.diff(ordered) > rtol * ordered[1:]))
    member = np.empty(lam.size, dtype=int)
    member[order] = np.cumsum(new) - 1
    return ordered[new].copy(), member


def _readonly(a: np.ndarray) -> np.ndarray:
    a.setflags(write=False)
    return a


# -- the basis ----------------------------------------------------------------

class LayerBasis:
    """The bases of every lateral eigenvalue of a transform on one family
    and one restriction, and the order of their coefficients as one
    vector: the flat counterpart of `randomfield.SphericalBasis`.

    `transform` is the `fourier.Transform` of the lateral grid and
    `bases` one `SpectralBasis` for each of its distinct eigenvalues,
    ascending, each labelled by that eigenvalue as `distinct_eigenvalues`
    gives it; `of` builds them.  The family is indexed by value.  The
    vector runs over the packed Fourier components in the transform's
    order, and within a component over the modes of its basis in
    ascending theta.

    Attributes: `transform`; `eigenvalues` (the distinct `|k|^2`,
    ascending); `member` (for each packed component the index of its
    basis); `multiplicity` (the components sharing each basis); `nmodes`
    (the modes kept by each basis); `size` (the length of the vector,
    sum multiplicity * nmodes); `theta` (the eigenvalue belonging to
    each entry of the vector, of which a diagonal metric or covariance
    is a power); and `r`, the physical nodes, which are depths.  The
    arrays are read-only.
    """

    def __init__(self, transform: Transform, bases: Sequence[SpectralBasis]) -> None:
        if not isinstance(transform, Transform):
            raise TypeError("transform must be a fourier.Transform")
        bases = tuple(bases)
        values, member = distinct_eigenvalues(transform)
        if len(bases) != values.size:
            raise ValueError(f"the transform has {values.size} distinct lateral "
                             f"eigenvalues, got {len(bases)} bases")
        first = bases[0]
        for i, b in enumerate(bases):
            if not isinstance(b, SpectralBasis):
                raise TypeError("bases must be SpectralBasis objects")
            if b.family is not first.family:
                raise ValueError("the bases must share one family")
            if b.family.index != "value":
                raise ValueError(f"a layer basis is of a family indexed by "
                                 f"'value', not by {b.family.index!r}")
            if b.label != values[i]:
                raise ValueError(f"bases[{i}] has the lateral eigenvalue "
                                 f"{b.label!r}, the transform's is {values[i]!r}")
            if (b.restriction.interval != first.restriction.interval
                    or b.restriction.nodes != first.restriction.nodes):
                raise ValueError("the bases must share one restriction")
        self._bases = bases
        self.transform = transform
        self.eigenvalues = _readonly(values)
        self.member = _readonly(member)
        self.multiplicity = _readonly(np.bincount(member, minlength=values.size))
        self.nmodes = _readonly(np.array([b.nmodes for b in bases]))
        counts = self.nmodes[member]
        self._starts = np.concatenate(([0], np.cumsum(counts)))
        self.size = int(self._starts[-1])
        self.r = first.r
        order = np.argsort(member, kind="stable")
        ends = np.cumsum(self.multiplicity)
        #: for each basis, the packed components that share it, ascending
        self._components = tuple(order[e - m:e] for e, m in zip(ends,
                                                               self.multiplicity))
        theta = np.empty(self.size)
        for i, b in enumerate(bases):
            theta[self._entries(i)] = b.theta
        self.theta = _readonly(theta)

    @classmethod
    def of(cls, family: SeparableOperatorFamily, transform: Transform, *,
           restrict: Restriction | None = None, nmodes: int | None = None,
           power: float | None = None, tol: float | None = None) -> LayerBasis:
        """The layer basis of `family` over `transform`, truncated by at
        most one rule: `nmodes` leading modes of every member, or the pair
        `power`, `tol`, the fewest coefficients of the whole basis, in
        ascending theta, whose omitted share of sum theta^-power over
        every component and mode is at most `tol`, with never fewer than
        one mode of a member; with neither every mode is kept.  `restrict`
        is the physical interval, the whole mesh when None.  Each distinct
        eigenvalue is passed to the family once, as one float."""
        if (power is None) != (tol is None):
            raise ValueError("power and tol are given together")
        if nmodes is not None and power is not None:
            raise ValueError("give at most one of nmodes and (power, tol)")
        values, member = distinct_eigenvalues(transform)
        if power is None:
            counts = [nmodes] * values.size
        else:
            if not 0.0 <= tol < 1.0:
                raise ValueError("tol must lie in [0, 1)")
            counts = cls._counts(family, values, np.bincount(member), float(power),
                                 float(tol))
        return cls(transform, [
            SpectralBasis(family, float(lam), restrict=restrict,
                          nmodes=None if k is None else int(k))
            for lam, k in zip(values, counts)])

    @staticmethod
    def _counts(family: SeparableOperatorFamily, values: np.ndarray,
                multiplicity: np.ndarray, power: float, tol: float) -> np.ndarray:
        """The modes each member keeps under the variance rule over the
        whole basis."""
        theta = np.concatenate([family.eigvalsh(float(lam)) for lam in values])
        each = theta.size // values.size
        owner = np.repeat(np.arange(values.size), each)
        order = np.argsort(theta, kind="stable")
        share = (multiplicity[owner] * theta ** (-power))[order]
        tail = np.cumsum(share[::-1])[::-1]
        small = tail <= tol * tail[0]
        kept = int(np.argmax(small)) if small.any() else theta.size
        counts = np.bincount(owner[order[:kept]], minlength=values.size)
        return np.maximum(counts, 1)

    @property
    def family(self) -> SeparableOperatorFamily:
        """The family the bases are of."""
        return self._bases[0].family

    @property
    def restriction(self) -> Restriction:
        """The physical interval the bases share."""
        return self._bases[0].restriction

    def __len__(self) -> int:
        """The number of bases: of distinct lateral eigenvalues."""
        return len(self._bases)

    def __getitem__(self, i: int) -> SpectralBasis:
        """The basis of the i-th distinct eigenvalue, ascending."""
        return self._bases[i]

    def __iter__(self) -> Iterator[SpectralBasis]:
        """The bases in ascending order of their eigenvalue."""
        return iter(self._bases)

    def basis_of(self, component: int) -> SpectralBasis:
        """The basis of one packed component: `self[member[component]]`."""
        return self._bases[int(self.member[self._component(component)])]

    def components(self, i: int) -> np.ndarray:
        """The packed components that share the i-th basis, ascending."""
        return self._components[i]

    def _component(self, component: int) -> int:
        c = int(component)
        if not 0 <= c < self.member.size:
            raise IndexError(f"component {component} is outside "
                             f"0..{self.member.size - 1}")
        return c

    def block(self, component: int) -> slice:
        """Where the coefficients of one packed component sit in the
        vector, as many as its basis has modes; `Transform.component_of`
        gives the component of a wavevector and phase."""
        c = self._component(component)
        return slice(int(self._starts[c]), int(self._starts[c + 1]))

    def _entries(self, i: int) -> np.ndarray:
        """The entries of the vector that belong to the i-th basis, shape
        (multiplicity[i], nmodes[i])."""
        return (self._starts[self._components[i]][:, None]
                + np.arange(int(self.nmodes[i]))[None, :])

    def synthesise(self, vector: ArrayLike, *, physical: bool = True) -> np.ndarray:
        """The functions of depth of a vector, shape (N, n) with N the
        transform's components in its order and depth last: at the
        physical nodes `r`, or with `physical=False` at every node of the
        mesh, which is what `analyse` takes."""
        v = np.asarray(vector, dtype=float)
        if v.shape != (self.size,):
            raise ValueError(f"expected a vector of length {self.size}, got "
                             f"shape {v.shape}")
        n = self.r.size if physical else self.family.mesh.nglob
        out = np.empty((self.member.size, n))
        for i, b in enumerate(self._bases):
            out[self._components[i]] = b.synthesise(v[self._entries(i)].T,
                                                    physical=physical).T
        return out

    def analyse(self, coeffs: ArrayLike) -> np.ndarray:
        """The vector of functions of depth given at every node of the
        mesh, shape (N, nglob): each component's function projected onto
        the basis of its eigenvalue."""
        c = np.asarray(coeffs, dtype=float)
        want = (self.member.size, self.family.mesh.nglob)
        if c.shape != want:
            raise ValueError(f"expected functions of depth of shape {want}, got "
                             f"{c.shape}")
        out = np.empty(self.size)
        for i, b in enumerate(self._bases):
            out[self._entries(i)] = b.analyse(c[self._components[i]].T).T
        return out

    def __repr__(self) -> str:
        r1, r2 = self.restriction.interval
        return (f"LayerBasis({self.transform.shape}, {len(self)} lateral "
                f"eigenvalues, {self.size} coefficients, on [{r1:g}, {r2:g}])")


# -- what the samplers share ----------------------------------------------------

def _clipped(spec: Profile, d1: float, d2: float) -> Any:
    """A profile of depth continued constantly beyond [d1, d2]."""
    f = profile_function(spec)
    return lambda d: f(np.clip(d, d1, d2))


def _interval(depth1: float, depth2: float, nu: float) -> tuple[float, float]:
    d1, d2 = float(depth1), float(depth2)
    if not (np.isfinite(d1) and np.isfinite(d2) and d1 < d2):
        raise ValueError("need finite depth1 < depth2")
    if not nu > 0.0:
        raise ValueError("nu must be positive")
    return d1, d2


# -- fields of depth alone ------------------------------------------------------

class DepthGRF:
    """A real, zero-mean Gaussian random field of depth on
    [depth1, depth2]: the flat counterpart of `randomfield.RadialGRF`.

    Matern-type with smoothness `nu` > 0, length scale `lam` (a number
    or a callable of depth) and marginal standard deviation `sigma`
    (likewise), via C = A^-2 beta with 2 beta = nu + 1/2 under the
    plain measure; the interval may lie anywhere.  `ngll`, `drmax`,
    `pad_factor`, `robin` and the truncation tolerance `tol` (the
    omitted share of the variance trace) tune the discretisation as
    they do `RadialGRF`'s.

    A sample is a vector of coefficients in `basis`, a `SpectralBasis`,
    independent with standard deviations `coefficient_std`; the field
    of a vector is `scale` times its synthesis, whose standard deviation
    at the nodes `r` is `sigma` to rounding.  Wherever a vector is taken
    a batch of them, shape (k, nmodes), is taken too.  Attributes:
    `interval`, `nu`, `beta`, `mesh`, `drmax`, `restriction`, `family`,
    `basis`, `r` (the physical nodes, depths), `sigma`, `scale` and
    `coefficient_std`, the last three read-only.
    """

    def __init__(self, depth1: float, depth2: float, nu: float, lam: Profile, *,
                 sigma: Profile = 1.0, ngll: int = 5, drmax: float | None = None,
                 pad_factor: float = 1.5, robin: Robin | str = None,
                 tol: float = 1e-6) -> None:
        d1, d2 = _interval(depth1, depth2, nu)
        self.interval = (d1, d2)
        self.nu = float(nu)
        self.beta = matern_exponent(self.nu, 1)
        lam_f = _clipped(lam, d1, d2)
        self.mesh, self.drmax, self.restriction = sampler_mesh(
            d1, d2, self.nu, lam_f, ngll, drmax, pad_factor, weight="one")
        self.family = SeparableOperatorFamily(
            self.mesh, kappa=lambda d: lam_f(d) ** 2, weight="one", lateral="one",
            index="value", robin=robin_spec(robin, lam_f, d1, d2))
        self.r = self.restriction.r
        self.basis = SpectralBasis(self.family, 0.0, restrict=self.restriction,
                                   power=2.0 * self.beta, tol=tol)
        self.coefficient_std = _readonly(self.basis.theta ** (-self.beta))
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
        on [depth1, depth2], through its nodal values; of a batch, one
        for each of its vectors."""
        c = coefficient_vectors(vector, self.nmodes)
        if c.ndim == 2:
            return tuple(self.to_layer(v) for v in c)
        return PolynomialLayer(self.interval, nodal_ppoly(
            self.mesh, self.restriction, self.synthesise(c)))

    def to_field(self, vector: ArrayLike, *, character: Character = SCALAR,
                 name: str | None = None) -> DepthField | tuple[DepthField, ...]:
        """The field of a coefficient vector as a `DepthField` on
        [depth1, depth2]; of a batch, one for each of its vectors."""
        c = coefficient_vectors(vector, self.nmodes)
        if c.ndim == 2:
            return tuple(self.to_field(v, character=character, name=name)
                         for v in c)
        return DepthField(self.interval, self.to_layer(c), character=character,
                          name=name)

    def __repr__(self) -> str:
        d1, d2 = self.interval
        return (f"DepthGRF([{d1:g}, {d2:g}], nu={self.nu:g}, {self.nmodes} modes "
                f"on {self.r.size} nodes)")


# -- fields of a box or a rectangle ---------------------------------------------

def _lateral_grid(lateral: Grid | Transform) -> Grid:
    """A sampler's lateral grid: itself, or the grid of a transform's
    shape over its domain as a `Box` or a `Rectangle`."""
    if isinstance(lateral, Grid):
        return lateral
    if not isinstance(lateral, Transform):
        raise TypeError(f"the lateral grid is a cartesian.Grid or a "
                        f"fourier.Transform, got {type(lateral).__name__}")
    domain: PaddedDomain = lateral.domain
    # the one branch on the number of axes: it names the kind of lateral domain
    kind = {len(kind.axes): kind for kind in (Box, Rectangle)}.get(domain.dimension)
    if kind is None:
        raise NotImplementedError(f"a random field over a lateral domain of "
                                  f"{domain.dimension} axes is not built")
    return Grid(kind(domain.bounds, padding=domain.padding), lateral.shape)


class BoxGRF:
    """A real, zero-mean Gaussian random field of a box on
    [depth1, depth2], in coefficients: the flat counterpart of
    `randomfield.SphericalGRF`.

    `grid` is the lateral `cartesian.Grid` (or a `fourier.Transform`,
    read as the grid of its shape over its domain), whose period the
    field has and whose shape is the lateral band.  Matern-type with
    C = A^-2 beta, 2 beta = nu + d / 2, d = 3 over a `Box` and d = 2
    over a `Rectangle`, whose field is the same at every y; `lam` is the
    length scale (a number or a callable of depth), `lam_h` an optional
    distinct horizontal one, `sigma` the marginal standard deviation.
    `ngll`, `drmax`, `pad_factor` and `robin` are those of `DepthGRF`;
    `tol` is the share of the variance trace the truncation of the
    basis omits (`LayerBasis.of`), and `cache_bytes` the budget of the
    family's cache.

    A sample is a vector of coefficients in `basis`, a `LayerBasis`,
    independent with standard deviations `coefficient_std`; the field
    of a vector is `scale` (a factor of depth) times its synthesis,
    whose standard deviation at the depth nodes `r` and the nodes of
    the lateral grid is `sigma` to rounding.  Wherever a vector is taken
    a batch of them, shape (k, size), is taken too.  Attributes:
    `interval`, `grid`, `transform`, `nu`, `beta`, `mesh`, `drmax`,
    `restriction`, `family`, `basis`, `r` (the physical nodes, depths),
    `sigma`, `scale` and `coefficient_std`, the last three read-only.
    """

    def __init__(self, depth1: float, depth2: float, grid: Grid | Transform,
                 nu: float, lam: Profile, *, sigma: Profile = 1.0,
                 lam_h: Profile | None = None, ngll: int = 5,
                 drmax: float | None = None, pad_factor: float = 1.5,
                 robin: Robin | str = None, tol: float = 1e-6,
                 cache_bytes: int | None = CACHE_BYTES) -> None:
        d1, d2 = _interval(depth1, depth2, nu)
        self.grid = _lateral_grid(grid)
        self.transform = self.grid.transform
        self.interval = (d1, d2)
        self.nu = float(nu)
        self.beta = matern_exponent(self.nu, self.transform.dimension + 1)
        p = 2.0 * self.beta

        lam_f = _clipped(lam, d1, d2)
        lamh_f = lam_f if lam_h is None else _clipped(lam_h, d1, d2)
        self.mesh, self.drmax, self.restriction = sampler_mesh(
            d1, d2, self.nu, lam_f, ngll, drmax, pad_factor, weight="one")
        self.family = SeparableOperatorFamily(
            self.mesh, kappa=lambda d: lam_f(d) ** 2,
            kappa_h=lambda d: lamh_f(d) ** 2, weight="one", lateral="one",
            index="value", robin=robin_spec(robin, lam_f, d1, d2),
            cache_bytes=cache_bytes)
        self.r = self.restriction.r
        self.basis = LayerBasis.of(self.family, self.transform,
                                   restrict=self.restriction, power=p, tol=tol)
        self.coefficient_std = _readonly(self.basis.theta ** (-self.beta))

        var_raw = np.zeros(self.r.size)
        for b, count in zip(self.basis, self.basis.multiplicity):
            var_raw += count * b.pointwise_variance(p)
        self.sigma, self.scale = marginal_scale(sigma, self.r,
                                                var_raw / self.transform.volume)
        self._interpolant: PPoly | None = None

    @property
    def nmodes(self) -> np.ndarray:
        """The modes kept by the basis of each distinct lateral eigenvalue."""
        return self.basis.nmodes

    @property
    def size(self) -> int:
        """The length of the coefficient vector."""
        return self.basis.size

    def sample(self, *, rng: Any = None, size: int | None = None) -> np.ndarray:
        """A coefficient vector (or `size` of them) in the order of
        `basis`: shape (basis.size,) or (size, basis.size), a batch being
        `size` successive draws from the generator.  `rng` is anything
        `np.random.default_rng` accepts."""
        rng = np.random.default_rng(rng)
        shape = (self.size,) if size is None else (int(size), self.size)
        return rng.standard_normal(shape) * self.coefficient_std

    def synthesise(self, vector: ArrayLike) -> np.ndarray:
        """The Fourier components of the field of a coefficient vector as
        functions of depth at the physical nodes: shape (N, r.size), the
        transform's components in its order and depth last, `scale`
        applied; of a batch, that with a leading axis over it."""
        c = coefficient_vectors(vector, self.size)
        rows = c.reshape(-1, self.size)
        out = np.empty((rows.shape[0], self.transform.size, self.r.size))
        for k, row in enumerate(rows):
            out[k] = self.basis.synthesise(row)
        out *= self.scale
        return out.reshape(c.shape[:-1] + out.shape[1:])

    def on_grid(self, vector: ArrayLike) -> np.ndarray:
        """The field of a coefficient vector at the depth nodes `r` and
        the nodes of the lateral grid: shape (r.size,) + grid.shape,
        through the transform's grid synthesis; of a batch, that with a
        leading axis over it.  The grid is the sampler's own, which is
        also its lateral band."""
        return self.transform.synthesise_grid(
            np.swapaxes(self.synthesise(vector), -1, -2))

    def sample_grid(self, *, rng: Any = None, size: int | None = None
                    ) -> np.ndarray:
        """One sample (or `size` of them) at the depth nodes `r` and the
        nodes of the lateral grid, `on_grid` of `sample`: shape
        (r.size,) + grid.shape, or that with a leading axis of `size`."""
        return self.on_grid(self.sample(rng=rng, size=size))

    def factor(self, i: int) -> np.ndarray:
        """The covariance factor of the i-th distinct lateral eigenvalue,
        shape (r.size, nmodes[i]), with the marginal normalisation
        applied: the function of depth of every component sharing it is
        B_i z with z standard normal."""
        b = self.basis[i]
        return (self.scale[:, None] * b.modes()[self.restriction.nodes]
                * b.theta[None, :] ** (-self.beta))

    def variance(self) -> np.ndarray:
        """Pointwise variance sigma(depth)^2 of the truncated field at the
        nodes (exact)."""
        return self.sigma ** 2

    def std(self) -> np.ndarray:
        """Pointwise standard deviation sigma(depth) at the nodes (exact
        by design)."""
        return self.sigma.copy()

    def coefficient_functions(self, vector: ArrayLike) -> PPoly:
        """The Fourier components of the field of a coefficient vector as
        one `PPoly` of depth on [depth1, depth2] whose values have shape
        (N,), or (k, N) of a batch: the polynomial of each element
        through `synthesise`'s values at its nodes."""
        return nodal_ppoly(self.mesh, self.restriction,
                           np.moveaxis(self.synthesise(vector), -1, 0))

    def to_field(self, vector: ArrayLike, *, character: Character = SCALAR,
                 name: str | None = None
                 ) -> FourierField | tuple[FourierField, ...]:
        """The field of a coefficient vector as a `FourierField` on
        [depth1, depth2] over `grid`, its coefficients the piecewise
        polynomials of `coefficient_functions`; of a batch, one for each
        of its vectors."""
        c = coefficient_vectors(vector, self.size)
        if c.ndim == 2:
            return tuple(self.to_field(v, character=character, name=name)
                         for v in c)
        return FourierField(self.interval, self.grid, self.coefficient_functions(c),
                            character=character, name=name)

    def _at_depths(self, depth: np.ndarray) -> np.ndarray:
        """The weights (depth.size, r.size) that read the polynomial
        through nodal values at depths of the interval."""
        if self._interpolant is None:
            self._interpolant = nodal_ppoly(self.mesh, self.restriction,
                                            np.eye(self.r.size))
        return self._interpolant(depth)

    def covariance(self, a: ArrayLike, b: ArrayLike) -> np.ndarray:
        """The covariance of the field (as `to_field` gives it) between
        points `a` and points `b`, each of shape (..., 1 + D) holding
        (depth, x, y) over a box and (depth, x) over a rectangle, depths
        within the interval: shape a.shape[:-1] + b.shape[:-1].

        Exact, from the basis: the sum over the coefficients of their
        variance times the synthesised mode at the two points, the
        lateral basis by its formulas (`Transform.basis`).  It costs the
        size of the lateral grid for each point and is meant for a few
        points; the marginal variance at the nodes is `variance`.
        """
        a = np.asarray(a, dtype=float)
        b = np.asarray(b, dtype=float)
        d = self.transform.dimension
        for pts in (a, b):
            if pts.ndim == 0 or pts.shape[-1] != 1 + d:
                raise ValueError(f"points have {1 + d} coordinates each, depth "
                                 f"first, along the last axis; got shape {pts.shape}")
        d1, d2 = self.interval
        fa, fb = a.reshape(-1, 1 + d), b.reshape(-1, 1 + d)
        slack = 1e-12 * (d2 - d1)
        for pts in (fa, fb):
            if np.any(pts[:, 0] < d1 - slack) or np.any(pts[:, 0] > d2 + slack):
                raise ValueError(f"depths must lie in the physical interval "
                                 f"[{d1:g}, {d2:g}]")
        Wa = self._at_depths(np.clip(fa[:, 0], d1, d2))
        Wb = self._at_depths(np.clip(fb[:, 0], d1, d2))
        La = self.transform.basis(fa[:, 1:])
        Lb = self.transform.basis(fb[:, 1:])
        out = np.zeros((fa.shape[0], fb.shape[0]))
        for i in range(len(self.basis)):
            B = self.factor(i)
            which = self.basis.components(i)
            out += ((Wa @ B) @ (Wb @ B).T) * (La[:, which] @ Lb[:, which].T)
        return out.reshape(a.shape[:-1] + b.shape[:-1])

    def __repr__(self) -> str:
        d1, d2 = self.interval
        return (f"BoxGRF([{d1:g}, {d2:g}], nu={self.nu:g}, {self.grid!r}, "
                f"{self.size} coefficients on {self.r.size} depth nodes)")


# -- layered fields: direct sums over a flat skeleton -----------------------------

class LayeredGRF:
    """Independent Gaussian fields on the layers of a flat skeleton, in
    coefficients: the flat counterpart of `randomfield.LayeredGRF`.

    Every included layer carries its own sampler on its interval of
    depth, a `DepthGRF` (fields of depth alone) when `grid` is None and
    a `BoxGRF` over `grid` otherwise, each padded beyond the layer's
    boundaries; the pads are fictitious and may overlap neighbours and
    reach above the surface, and the direct sum is generically
    discontinuous at layer boundaries.  `nu`, `lam`, `sigma`, `lam_h`
    and `drmax` are shared across layers when numbers or callables, or
    per-layer when sequences of length `skeleton.nlayers`; `lam_h` is
    read with a grid alone.  `layers` restricts the sum to the given
    layer indices.  A flat Geometry or Model is accepted in place of a
    skeleton; over a model the grid is one over its domain, the model's
    box being the period.

    A sample is one vector, the coefficient vectors of the included
    layers in ascending order of layer (`block`), of length `size`;
    `to_fields` gives the field of a vector on every layer of the
    skeleton, zero on the excluded ones, ready for `Model.with_field`.
    Wherever a vector is taken a batch of them, shape (k, size), is
    taken too.
    """

    def __init__(self, skeleton: Skeleton | Geometry | Model,
                 nu: float | Sequence[float], lam: Profile | Sequence[Profile], *,
                 grid: Grid | Transform | None = None,
                 sigma: Profile | Sequence[Profile] = 1.0,
                 lam_h: Profile | Sequence[Profile | None] | None = None,
                 layers: Iterable[int] | None = None, ngll: int = 5,
                 drmax: float | Sequence[float | None] | None = None,
                 pad_factor: float = 1.5, robin: Robin | str = None,
                 tol: float = 1e-6, name: str | None = None) -> None:
        sk = getattr(skeleton, "skeleton", skeleton)
        if not isinstance(sk, Skeleton):
            raise TypeError("expected a Skeleton, a flat Geometry or a flat Model")
        self.skeleton = sk
        self.name = name
        self.grid = None if grid is None else _lateral_grid(grid)
        nl = sk.nlayers
        self.layers = layer_indices(layers, nl)

        def made(i: int) -> DepthGRF | BoxGRF:
            shared = dict(sigma=per_layer(sigma, i, nl), ngll=ngll,
                          drmax=per_layer(drmax, i, nl), pad_factor=pad_factor,
                          robin=robin, tol=tol)
            d1, d2 = sk.interval(i)
            if self.grid is None:
                return DepthGRF(d1, d2, per_layer(nu, i, nl), per_layer(lam, i, nl),
                                **shared)
            return BoxGRF(d1, d2, self.grid, per_layer(nu, i, nl),
                          per_layer(lam, i, nl), lam_h=per_layer(lam_h, i, nl),
                          **shared)

        self._grfs = {i: made(i) for i in self.layers}
        sizes = [self._grfs[i].size for i in self.layers]
        self._starts = dict(zip(self.layers, np.cumsum([0] + sizes[:-1]).tolist()))
        self.size = int(sum(sizes))

    def __getitem__(self, i: int) -> DepthGRF | BoxGRF:
        """The sampler of layer i (KeyError if excluded)."""
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
                  ) -> (tuple[DepthField | FourierField, ...]
                        | tuple[tuple[DepthField | FourierField, ...], ...]):
        """The field of a coefficient vector on every layer of the
        skeleton: a `DepthField` each, or over a grid a `FourierField`
        each, zero on the excluded layers.  Of a batch, one such tuple
        for each of its vectors."""
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
            elif self.grid is None:
                out.append(DepthField(iv, constant_layer(iv, 0.0), character=character,
                                      name=self.name))
            else:
                out.append(FourierField(iv, self.grid,
                                        np.zeros(self.grid.transform.size),
                                        character=character, name=self.name))
        return tuple(out)

    def __repr__(self) -> str:
        kind = "depth alone" if self.grid is None else repr(self.grid)
        return (f"LayeredGRF({len(self._grfs)}/{self.skeleton.nlayers} layers "
                f"active: {list(self.layers)}, {kind})")
