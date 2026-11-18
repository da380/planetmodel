"""Real Fourier series on a periodic box: conventions, packing, transforms.

**The box.**  A field is periodic on a box of D axes with periods L_1
... L_D and volume V = L_1 ... L_D, sampled on a grid of `shape` (n_1,
..., n_D), the points origin_i + j L_i / n_i for j = 0 ... n_i - 1.
`PaddedDomain` states the box: a domain proper of `bounds` with `padding`
on each side of each axis, the box being the bounds so widened (`extent`)
and its origin their lower corner.  A `cartesian.Box` or `Rectangle` is
read as one through its own `bounds` and `padding`, and the lengths of a
torus give the box itself from the origin, with no padding.

**The basis** is real and orthonormal in L2 over the box.  With ::

    theta_k(x) = sum_i 2 pi k_i (x_i - origin_i) / L_i

for an integer wavevector k, a mode and its conjugate -k form a pair
unless they coincide modulo the shape.  A pair contributes two
components::

    sqrt(2 / V) cos theta_k   and   -sqrt(2 / V) sin theta_k,

the sign that of rfftn's exp(-i theta); a self-conjugate mode -- the
constant, and the modes whose every wavenumber is zero or the Nyquist
n_i / 2 of an axis of even size -- contributes one, cos theta_k / sqrt V.

**The Nyquist rule.**  A cosine at a Nyquist wavenumber is self-conjugate
and stored once, as the real FFT stores it, with its norm on the grid:
the grid's quadrature, V / N times the sum over its N points, gives it
norm one, while over the continuous box its squared norm is a half.  The
basis is orthonormal under the grid's quadrature exactly, and in L2 over
the box for every mode but these.  The grid's convention is kept so that
a pointwise product of two grid functions is exactly self-adjoint; where
such a mode is carried to a grid on which it is a pair (`transfer`), the
function is kept, not its coefficient, the ratio of the two norms
carried.

**Packing.**  The N = n_1 ... n_D components of a grid are ordered as
`Packing` lists them: the self-conjugate modes first, in the flat order of
rfftn's half spectrum (of shape n_1, ..., n_{D-1}, n_D // 2 + 1), then
each pair's cosine and sine together, in the same order of the pair's
representative, the member of lower flat index where both lie in the half
spectrum.  A component is named by its signed integer wavevector,
wavenumbers above n_i / 2 taken as k_i - n_i so that a Nyquist wavenumber
is +n_i / 2, and its phase, 0 for a cosine or a self-conjugate mode and 1
for a sine; `component_of` finds one, modulo the grid.  The
representative's wavevector is the one the basis is evaluated with off
the grid.  `degrees` is `|k|` rounded down in grid units, which group
modes of one spatial scale only where the periods are equal;
`laplacian_eigenvalues` are the physical sum_i (2 pi k_i / L_i)^2.  A
grid holds exactly its packing's components: its band is its shape.

**Transforms** (`Transform`).  Leading axes are batches throughout: grid
values `(..., *shape)`, components `(..., N)`, rfftn arrays
`(..., *rfft_shape)`, and points `(..., D)`, whose values have the batch axes and
then the points' own.  `analyse_grid` and `synthesise_grid` go between
grid values and components through scipy's rfftn and irfftn, each the
other's inverse to rounding; `from_rfft` and `to_rfft` are the same
steps without the FFT, between components and the plain unnormalised
rfftn array numpy and scipy share, a component being the real or
imaginary part of its entry times sqrt(V) / N, and times a further
sqrt 2 for a pair.  `derivative` is d/dx_i in components: a pair's
cosine and sine rotate into one another, and a self-conjugate mode, whose
derivative the grid does not hold, goes to zero.  `basis` is the basis
at points, by the formulas above, to rounding.  `synthesise`,
`synthesise_gradient` and `accumulate` are a series at arbitrary points,
its gradient (d/dx_i along each axis, in the domain's lengths), and the
adjoint of evaluation, sum_j w_j phi(x_j) for every basis function phi:
through finufft's type-2 and type-1 non-uniform FFTs, to about
`POINT_EPSILON` of the field's size, or by the direct sum over the basis
with `route="direct"`, to rounding at a cost of N per point.  `at`
prepares a set of points once, for transforms applied there many times:
its `PointTransform` holds the points' angles in the box and, on
finufft's route, finufft's plans with the points sorted, and applies the
three to the same results with no work on the points; each of the
methods that take points is one application of such a preparation.  The
spectrum handed to finufft is two larger than the grid along every axis,
so that +k and -k are distinct entries for every mode, Nyquist included,
and the self-conjugate modes and the pairs need no separate handling.
finufft transforms one to three dimensions: on a box of more axes the
transforms at points go by the direct route, asked for by name, and
finufft's is refused.  finufft is imported where a non-uniform transform
is made, so that `import planetmodel` does not load it.

**Points** are any real coordinates: a series is periodic and is
evaluated at a point's image in the box.  Where the padding is not part
of the domain, `PaddedDomain.contains` says which points lie in the
domain proper, and `PaddedDomain.checked` refuses the others by name.
`mask` is the grid's points in the domain proper, and `taper` the
raised-cosine window that is one there and falls to zero at the far edge
of the padding, for a field held at its boundary values across the
padding: a step there would ring through a truncated series.
"""
from __future__ import annotations

import math
import threading
from collections.abc import Sequence
from functools import cached_property, lru_cache
from types import ModuleType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.fft import irfftn, rfftn

if TYPE_CHECKING:
    from .cartesian.domains import Domain

__all__ = ["Packing", "PaddedDomain", "Transform", "PointTransform", "DomainLike",
           "POINT_EPSILON", "POSITION_RTOL", "NUFFT_THREADS", "ROUTES"]

#: The relative accuracy asked of finufft at arbitrary points.
POINT_EPSILON = 1e-12

#: finufft's threads unless more are asked for: its own default, every core,
#: costs more in threading than it saves at the sizes of a grid's transform,
#: and oversubscribes inside an outer parallel loop.
NUFFT_THREADS = 1

#: The relative allowance at the bounds of a domain, of each axis's width: a
#: point this close outside lies on the boundary.
POSITION_RTOL = 1e-12

#: The routes of the transforms at points: finufft's non-uniform FFTs, or the
#: direct sum over the basis.
ROUTES = ("nufft", "direct")

#: What a transform is built over: a `PaddedDomain`, a `cartesian.Box` or
#: `Rectangle` (read through its `bounds` and `padding`), or the periods of a
#: torus from the origin, one per axis.
type DomainLike = PaddedDomain | Domain | Sequence[float] | float


def _shape(shape: Sequence[int]) -> tuple[int, ...]:
    """A grid shape as a tuple of positive ints, or the error."""
    try:
        out = tuple(int(n) for n in shape)
    except TypeError:
        raise ValueError(f"a grid shape is a sequence of sizes, got "
                         f"{shape!r}") from None
    if not out or any(n < 1 for n in out):
        raise ValueError(f"a grid shape has one or more positive sizes, got {shape!r}")
    return out


def _readonly(a: np.ndarray) -> np.ndarray:
    """The array, marked read-only: a packing's tables are shared."""
    a.setflags(write=False)
    return a


# -- the packing -------------------------------------------------------------------

class Packing:
    """The real components of a grid of `shape`, and where each sits in
    rfftn's half spectrum; see the module docstring for the order.

    Each conjugate orbit of the grid's modes is enumerated once: an entry
    of the half spectrum whose conjugate is itself is one component, and
    a pair two.  The count is the number of grid points, as a real field
    on the grid has that many degrees of freedom; that identity is
    asserted on construction.
    """

    def __init__(self, shape: Sequence[int]) -> None:
        shape = _shape(shape)
        rfft_shape = shape[:-1] + (shape[-1] // 2 + 1,)
        grids = np.meshgrid(*[np.arange(n) for n in rfft_shape], indexing="ij")
        flat = [g.ravel() for g in grids]
        conjugate = [(-k) % n for k, n in zip(flat, shape)]
        # A conjugate lies in the half spectrum exactly when its last index
        # does; in one dimension that is the zero and Nyquist entries alone,
        # in more whole slices of the half spectrum pair up inside it.
        inside = conjugate[-1] < rfft_shape[-1]
        linear = np.ravel_multi_index(flat, rfft_shape)
        partner = np.full(linear.size, -1, dtype=int)
        if np.any(inside):
            partner[inside] = np.ravel_multi_index([c[inside] for c in conjugate],
                                                   rfft_shape)
        fixed = partner == linear
        # One representative per pair: the lower flat index where both members
        # are in the half spectrum, the entry itself where its partner is not.
        paired = ~fixed & ((partner < 0) | (linear < partner))
        self._shape = shape
        self._rfft_shape = rfft_shape
        self._fixed = _readonly(np.flatnonzero(fixed))
        self._paired = _readonly(np.flatnonzero(paired))
        self._partners = _readonly(partner[self._paired])
        size = int(np.prod(shape))
        self._size = size
        if self._fixed.size + 2 * self._paired.size != size:
            raise AssertionError(
                f"the packing of shape {shape} has {self._fixed.size} self-conjugate "
                f"modes and {self._paired.size} pairs, not {size} components")
        signed = [np.where(k > n // 2, k - n, k) for k, n in zip(flat, shape)]
        self._wavevectors = _readonly(np.stack(
            [np.concatenate([w[self._fixed], np.repeat(w[self._paired], 2)])
             for w in signed]))
        self._phases = _readonly(np.concatenate(
            [np.zeros(self._fixed.size, dtype=int),
             np.tile(np.array([0, 1]), self._paired.size)]))
        keys = self._keys(self._wavevectors, self._phases)
        order = np.argsort(keys)
        self._sorted = (_readonly(keys[order]), _readonly(order))

    @property
    def shape(self) -> tuple[int, ...]:
        """The grid's points along each axis."""
        return self._shape

    @property
    def rfft_shape(self) -> tuple[int, ...]:
        """The shape of rfftn's half spectrum of the grid: the last axis
        halved, n // 2 + 1."""
        return self._rfft_shape

    @property
    def size(self) -> int:
        """The number of components, the grid's number of points."""
        return self._size

    @property
    def dimension(self) -> int:
        """The number of axes."""
        return len(self._shape)

    @property
    def fixed_indices(self) -> np.ndarray:
        """Flat indices into the half spectrum of the self-conjugate modes,
        in the order of the first components."""
        return self._fixed

    @property
    def paired_indices(self) -> np.ndarray:
        """Flat indices into the half spectrum of each pair's representative,
        in the order of the pairs."""
        return self._paired

    @property
    def paired_partners(self) -> np.ndarray:
        """The flat index of each representative's conjugate where it lies in
        the half spectrum too, and -1 where it does not: irfftn imposes the
        symmetry across the dropped half alone, so these are written."""
        return self._partners

    @property
    def wavevectors(self) -> np.ndarray:
        """The signed integer wavevector of each component, shape (D, N);
        a pair's cosine and sine share theirs."""
        return self._wavevectors

    @property
    def phases(self) -> np.ndarray:
        """Zero for a cosine or a self-conjugate mode, one for a sine."""
        return self._phases

    @cached_property
    def degrees(self) -> np.ndarray:
        """`|k|` of each component's integer wavevector, rounded down."""
        k = self._wavevectors.astype(float)
        return _readonly(np.floor(np.sqrt(np.sum(k ** 2, axis=0))).astype(int))

    @cached_property
    def measures(self) -> np.ndarray:
        """The squared grid norm of each component's bare cosine or sine, as
        a fraction of the volume: one for a self-conjugate mode, a half for
        a member of a pair (the Nyquist rule)."""
        count = self._fixed.size
        return _readonly(np.concatenate([np.ones(count),
                                         np.full(self.size - count, 0.5)]))

    def _keys(self, wavevector: np.ndarray, phase: np.ndarray) -> np.ndarray:
        """One integer per (wavevector, phase), the wavevector taken modulo
        the grid."""
        reduced = [np.mod(np.asarray(wavevector[axis], dtype=int), n)
                   for axis, n in enumerate(self._shape)]
        return np.ravel_multi_index([*reduced, np.asarray(phase, dtype=int)],
                                    (*self._shape, 2))

    def component_of(self, wavevector: ArrayLike, *, phase: ArrayLike = 0
                     ) -> int | np.ndarray:
        """The position of the component of a wavevector and phase.

        One wavevector of D integers gives an int; an array of shape (D, m)
        gives m positions, `phase` a scalar or one per column.  Wavevectors
        are taken modulo the grid, so a mode may be named by either sign of
        an aliased wavenumber.  A label the grid does not hold, a sine of a
        self-conjugate mode, is refused.
        """
        k = np.asarray(wavevector, dtype=int)
        if k.ndim == 0 or k.shape[0] != self.dimension:
            raise ValueError(f"a wavevector on a grid of {self.dimension} axes has "
                             f"{self.dimension} components, got shape {k.shape}")
        p = np.broadcast_to(np.asarray(phase, dtype=int), k.shape[1:])
        if np.any((p != 0) & (p != 1)):
            raise ValueError("a phase is 0 (cosine) or 1 (sine)")
        keys = self._keys(k, p)
        sorted_keys, order = self._sorted
        found = np.minimum(np.searchsorted(sorted_keys, keys), sorted_keys.size - 1)
        if np.any(sorted_keys[found] != keys):
            raise ValueError("no component has that wavevector and phase: a "
                             "self-conjugate mode has no sine")
        position = order[found]
        return int(position) if position.ndim == 0 else position

    def matching(self, target: Packing) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(here, there, scale): the components of this packing and of
        `target` that share a signed wavevector and phase, and the factor
        that carries a coefficient from here to there keeping its function,
        sqrt(target measure / measure here): 1 / sqrt 2 for a Nyquist
        cosine here that is a pair's cosine there, sqrt 2 the other way, one
        otherwise.  Components without a match are absent from both."""
        if target.dimension != self.dimension:
            raise ValueError(f"a grid of {self.dimension} axes matches no grid of "
                             f"{target.dimension}")
        bound = [max(a, b) for a, b in zip(self._shape, target.shape)]
        dims = tuple(2 * b + 1 for b in bound) + (2,)

        def keys(p: Packing) -> np.ndarray:
            return np.ravel_multi_index(
                [*(p.wavevectors[i] + bound[i] for i in range(self.dimension)),
                 p.phases], dims)

        _, here, there = np.intersect1d(keys(self), keys(target), assume_unique=True,
                                        return_indices=True)
        scale = np.sqrt(target.measures[there] / self.measures[here])
        return here, there, scale

    def __repr__(self) -> str:
        return f"Packing({self._shape})"


@lru_cache(maxsize=64)
def _packing(shape: tuple[int, ...]) -> Packing:
    """The packing of a shape, built once: its tables are read-only."""
    return Packing(shape)


# -- the domain --------------------------------------------------------------------

def _pairs(bounds: Sequence[Sequence[float]]) -> tuple[tuple[float, float], ...]:
    """Finite increasing (lower, upper) pairs, one per axis, or the error."""
    try:
        pairs = tuple((float(lo), float(hi)) for lo, hi in bounds)
    except (TypeError, ValueError):
        raise ValueError(f"bounds are one (lower, upper) pair per axis, got "
                         f"{bounds!r}") from None
    if not pairs:
        raise ValueError("a domain has one axis or more")
    for axis, (lo, hi) in enumerate(pairs):
        if not (np.isfinite(lo) and np.isfinite(hi) and hi > lo):
            raise ValueError(f"the bounds of axis {axis} must be finite and increase, "
                             f"got ({lo:g}, {hi:g})")
    return pairs


class PaddedDomain:
    """A domain proper of `bounds`, one (lower, upper) pair per axis, with
    `padding` on each side of each axis (one length for every axis, or one
    each, zero by default), inside the periodic box of the bounds so
    widened.  Two are equal when their bounds and padding are.

    `of` reads a `cartesian.Box` or `Rectangle`, or makes a torus of its
    periods; see the module docstring.
    """

    def __init__(self, bounds: Sequence[Sequence[float]], *,
                 padding: float | Sequence[float] = 0.0) -> None:
        self._bounds = _pairs(bounds)
        d = len(self._bounds)
        pads = (padding,) * d if np.ndim(padding) == 0 else tuple(padding)
        if len(pads) != d:
            raise ValueError(f"a domain of {d} axes takes one padding or {d}, got "
                             f"{padding!r}")
        self._padding = tuple(float(p) for p in pads)
        if not all(np.isfinite(p) and p >= 0.0 for p in self._padding):
            raise ValueError(f"the padding must be finite and non-negative, got "
                             f"{padding!r}")

    @classmethod
    def of(cls, domain: DomainLike) -> PaddedDomain:
        """The padded domain of `domain`: itself; a `cartesian.Box` or
        `Rectangle`, or anything with `bounds` and `padding` in either's
        form, read through them; or the periods of a torus, a number or one
        per axis, whose domain proper is the whole box from the origin."""
        if isinstance(domain, PaddedDomain):
            return domain
        bounds = getattr(domain, "bounds", None)
        if bounds is None:
            lengths = np.atleast_1d(np.asarray(domain, dtype=float))
            if lengths.ndim != 1 or not np.all(np.isfinite(lengths) & (lengths > 0.0)):
                raise ValueError(f"the periods of a torus are positive lengths, one "
                                 f"per axis, got {domain!r}")
            return cls([(0.0, float(length)) for length in lengths])
        if np.ndim(bounds) == 1:                     # one axis, a Rectangle's
            return cls([bounds], padding=domain.padding)
        return cls(bounds, padding=domain.padding)

    @property
    def dimension(self) -> int:
        """The number of axes."""
        return len(self._bounds)

    @property
    def bounds(self) -> tuple[tuple[float, float], ...]:
        """The domain proper: one (lower, upper) pair per axis."""
        return self._bounds

    @property
    def padding(self) -> tuple[float, ...]:
        """The margin on each side of each axis."""
        return self._padding

    @property
    def extent(self) -> tuple[tuple[float, float], ...]:
        """The bounds widened by the padding on each side: the periodic box."""
        return tuple((lo - p, hi + p) for (lo, hi), p in zip(self._bounds,
                                                           self._padding))

    @property
    def lengths(self) -> tuple[float, ...]:
        """The periods: each extent's width."""
        return tuple(hi - lo for lo, hi in self.extent)

    @property
    def origin(self) -> tuple[float, ...]:
        """The box's lower corner, where a grid's first point is."""
        return tuple(lo for lo, _ in self.extent)

    @property
    def volume(self) -> float:
        """The box's volume, the product of the periods."""
        return float(np.prod(self.lengths))

    @property
    def domain_volume(self) -> float:
        """The volume of the domain proper, the padding left out."""
        return float(np.prod([hi - lo for lo, hi in self._bounds]))

    def _points(self, points: ArrayLike) -> np.ndarray:
        """Points as floats with D coordinates along the last axis, or the
        error."""
        x = np.asarray(points, dtype=float)
        if x.ndim == 0 or x.shape[-1] != self.dimension:
            raise ValueError(f"points have {self.dimension} coordinates each, along "
                             f"the last axis; got shape {x.shape}")
        return x

    def contains(self, points: ArrayLike, *, rtol: float = POSITION_RTOL) -> np.ndarray:
        """Whether each point, of shape (..., D), lies in the domain proper,
        to `rtol` of each axis's width; the padding is outside."""
        x = self._points(points)
        inside = np.ones(x.shape[:-1], dtype=bool)
        for axis, (lo, hi) in enumerate(self._bounds):
            tol = rtol * (hi - lo)
            inside &= (x[..., axis] >= lo - tol) & (x[..., axis] <= hi + tol)
        return inside

    def checked(self, points: ArrayLike) -> np.ndarray:
        """The points as floats of shape (..., D), every one in the domain
        proper to `POSITION_RTOL`; a point outside is refused, naming the
        domain and the first such point, for a caller to whom the padding is
        not part of the domain and a point past it a different place."""
        x = self._points(points)
        outside = ~self.contains(x)
        if np.any(outside):
            box = " x ".join(f"[{lo:g}, {hi:g}]" for lo, hi in self._bounds)
            first = ", ".join(f"{v:g}" for v in x[outside][0])
            raise ValueError(f"every point must lie in the domain, {box}; one is at "
                             f"({first})")
        return x

    def grid_axes(self, shape: Sequence[int]) -> tuple[np.ndarray, ...]:
        """The coordinates of a grid of `shape` along each axis: from the
        box's lower corner in steps of the period over the size, so that
        the domain proper sits in the middle of each."""
        shape = self._grid_shape(shape)
        return tuple(lo + np.arange(n) * length / n
                     for (lo, _), n, length in zip(self.extent, shape, self.lengths))

    def mask(self, shape: Sequence[int]) -> np.ndarray:
        """A boolean array of `shape`: true at the grid's points in the
        domain proper, to `POSITION_RTOL`, false in the padding."""
        out = np.ones(self._grid_shape(shape), dtype=bool)
        for axis, (values, (lo, hi)) in enumerate(zip(self.grid_axes(shape),
                                                      self._bounds)):
            tol = POSITION_RTOL * (hi - lo)
            out &= self._along(axis, (values >= lo - tol) & (values <= hi + tol))
        return out

    def taper(self, shape: Sequence[int]) -> np.ndarray:
        """The raised-cosine window on a grid of `shape`: one on the domain
        proper and, across the padding of each axis, 0.5 (1 + cos(pi s))
        with s the distance from the nearer bound over the padding, zero at
        the far edge; the product over the axes."""
        out = np.ones(self._grid_shape(shape))
        for axis, (values, (lo, hi), pad) in enumerate(
                zip(self.grid_axes(shape), self._bounds, self._padding)):
            window = np.ones_like(values)
            if pad > 0.0:
                below, above = values < lo, values > hi
                window[below] = 0.5 * (1.0 + np.cos(
                    np.pi * np.clip((lo - values[below]) / pad, 0.0, 1.0)))
                window[above] = 0.5 * (1.0 + np.cos(
                    np.pi * np.clip((values[above] - hi) / pad, 0.0, 1.0)))
            out = out * self._along(axis, window)
        return out

    def _grid_shape(self, shape: Sequence[int]) -> tuple[int, ...]:
        """A grid shape of this domain's number of axes, or the error."""
        shape = _shape(shape)
        if len(shape) != self.dimension:
            raise ValueError(f"a grid on a domain of {self.dimension} axes has "
                             f"{self.dimension} sizes, got {shape}")
        return shape

    def _along(self, axis: int, values: np.ndarray) -> np.ndarray:
        """A one-axis array shaped to broadcast along `axis` of a grid."""
        shape = [1] * self.dimension
        shape[axis] = values.size
        return values.reshape(shape)

    def __eq__(self, other: object) -> bool:
        return (type(other) is PaddedDomain and self._bounds == other._bounds
                and self._padding == other._padding)

    def __hash__(self) -> int:
        return hash((PaddedDomain, self._bounds, self._padding))

    def __repr__(self) -> str:
        bounds = ", ".join(f"({lo:g}, {hi:g})" for lo, hi in self._bounds)
        bounds += "," if self.dimension == 1 else ""
        pad = ("" if not any(self._padding) else
               ", padding=(" + ", ".join(f"{p:g}" for p in self._padding) + ")")
        return f"PaddedDomain(({bounds}){pad})"


# -- the transforms ----------------------------------------------------------------

def _finufft() -> ModuleType:
    """finufft, imported where a non-uniform transform is made."""
    import finufft
    return finufft


class Transform:
    """The real Fourier transforms of a grid of `shape` on the periodic box
    of `domain` (a `PaddedDomain`, a `cartesian.Box` or `Rectangle`, or the
    periods of a torus); see the module docstring for the conventions.

    Grid analysis and synthesis, the interchange with plain rfftn arrays,
    the basis at points, and synthesis, its gradient and the adjoint at
    arbitrary points, through finufft or by the direct sum.  The packing is
    built once per shape and the non-uniform layout once per transform.
    """

    def __init__(self, shape: Sequence[int], domain: DomainLike) -> None:
        self._domain = PaddedDomain.of(domain)
        self._packing = _packing(self._domain._grid_shape(shape))

    @property
    def shape(self) -> tuple[int, ...]:
        """The grid's points along each axis."""
        return self._packing.shape

    @property
    def domain(self) -> PaddedDomain:
        """The domain proper and the periodic box it sits in."""
        return self._domain

    @property
    def packing(self) -> Packing:
        """The components and their place in rfftn's half spectrum."""
        return self._packing

    @property
    def dimension(self) -> int:
        """The number of axes."""
        return self._packing.dimension

    @property
    def size(self) -> int:
        """The number of components, the grid's number of points."""
        return self._packing.size

    @property
    def lengths(self) -> tuple[float, ...]:
        """The periods along each axis."""
        return self._domain.lengths

    @property
    def origin(self) -> tuple[float, ...]:
        """The box's lower corner, the grid's first point."""
        return self._domain.origin

    @property
    def volume(self) -> float:
        """The box's volume, V."""
        return self._domain.volume

    @property
    def wavevectors(self) -> np.ndarray:
        """The signed integer wavevector of each component, shape (D, N)."""
        return self._packing.wavevectors

    @property
    def phases(self) -> np.ndarray:
        """Zero for a cosine or a self-conjugate mode, one for a sine."""
        return self._packing.phases

    @property
    def degrees(self) -> np.ndarray:
        """`|k|` of each component's integer wavevector, rounded down."""
        return self._packing.degrees

    @cached_property
    def wavenumbers(self) -> np.ndarray:
        """The physical wavevector 2 pi k_i / L_i of each component, shape
        (D, N)."""
        k = self._packing.wavevectors
        return _readonly(np.stack([2.0 * np.pi * k[i] / length
                                   for i, length in enumerate(self.lengths)]))

    @cached_property
    def laplacian_eigenvalues(self) -> np.ndarray:
        """sum_i (2 pi k_i / L_i)^2 of each component, the eigenvalue of
        minus the Laplacian on its basis function."""
        k = self._packing.wavevectors
        squared = np.zeros(self.size)
        for axis, length in enumerate(self.lengths):
            squared += (2.0 * np.pi * k[axis] / length) ** 2
        return _readonly(squared)

    def component_of(self, wavevector: ArrayLike, *, phase: ArrayLike = 0
                     ) -> int | np.ndarray:
        """The position of the component of a wavevector and phase:
        `Packing.component_of`."""
        return self._packing.component_of(wavevector, phase=phase)

    @cached_property
    def grid_axes(self) -> tuple[np.ndarray, ...]:
        """The grid's coordinates along each axis, from the box's lower
        corner: `PaddedDomain.grid_axes`."""
        return tuple(_readonly(a) for a in self._domain.grid_axes(self.shape))

    @property
    def grid_points(self) -> np.ndarray:
        """The grid's points, shape `(*shape, D)`."""
        return np.stack(np.meshgrid(*self.grid_axes, indexing="ij"), axis=-1)

    @cached_property
    def mask(self) -> np.ndarray:
        """The grid's points in the domain proper: `PaddedDomain.mask`."""
        return _readonly(self._domain.mask(self.shape))

    @cached_property
    def taper(self) -> np.ndarray:
        """The raised-cosine window over the padding: `PaddedDomain.taper`."""
        return _readonly(self._domain.taper(self.shape))

    # -- components and grids ------------------------------------------------------

    @cached_property
    def _scale(self) -> float:
        """sqrt(V) / N: a self-conjugate component over the real part of its
        rfftn entry, and a pair's over its parts with a further sqrt 2."""
        return float(np.sqrt(self.volume) / self.size)

    def _components(self, components: ArrayLike) -> np.ndarray:
        """Components as floats with N along the last axis, or the error."""
        c = np.asarray(components, dtype=float)
        if c.ndim == 0 or c.shape[-1] != self.size:
            raise ValueError(f"components of a grid of shape {self.shape} have "
                             f"{self.size} entries along the last axis; got shape "
                             f"{c.shape}")
        return c

    @cached_property
    def _interchange(self) -> tuple[int, int, np.float64, np.ndarray, np.ndarray]:
        """(entries, count, pair, retained, written): the entries of rfftn's
        half spectrum, the number of self-conjugate modes, a pair's
        component over its parts, sqrt(V) sqrt(2) / N, and the pairs whose
        partner lies in the half spectrum with the flat index of each
        partner."""
        p = self._packing
        retained = np.flatnonzero(p.paired_partners >= 0)
        return (int(np.prod(p.rfft_shape)), p.fixed_indices.size,
                self._scale * np.sqrt(2.0), _readonly(retained),
                _readonly(p.paired_partners[retained]))

    def from_rfft(self, coefficients: ArrayLike) -> np.ndarray:
        """Components from an rfftn array of shape `(..., *rfft_shape)`, on the
        unnormalised scale numpy and scipy share.  A self-conjugate entry's
        imaginary part, zero for a real field, is not read, nor is a pair's
        entry other than its representative."""
        p = self._packing
        a = np.asarray(coefficients)
        _, count, pair, _, _ = self._interchange
        if a.shape == p.rfft_shape:            # one array: the steps below, unbatched
            flat = a.reshape(-1)
            out = np.empty(self.size)
            out[:count] = flat[p.fixed_indices].real * self._scale
            paired = flat[p.paired_indices] * pair
            out[count::2] = paired.real
            out[count + 1::2] = paired.imag
            return out
        d = self.dimension
        if a.ndim < d or a.shape[a.ndim - d:] != p.rfft_shape:
            raise ValueError(f"an rfftn array of a grid of shape {self.shape} has "
                             f"trailing shape {p.rfft_shape}; got {a.shape}")
        flat = a.reshape(a.shape[:a.ndim - d] + (-1,))
        out = np.empty(flat.shape[:-1] + (self.size,))
        out[..., :count] = flat[..., p.fixed_indices].real * self._scale
        paired = flat[..., p.paired_indices] * pair
        out[..., count::2] = paired.real
        out[..., count + 1::2] = paired.imag
        return out

    def to_rfft(self, components: ArrayLike) -> np.ndarray:
        """The rfftn array, of shape `(..., *rfft_shape)`, of the field with
        these components: the inverse of `from_rfft`, a pair's partners in
        the half spectrum written as its conjugate."""
        p = self._packing
        c = self._components(components)
        entries, count, pair, retained, written = self._interchange
        if c.ndim == 1:                        # one vector: the steps below, unbatched
            out = np.zeros(entries, dtype=complex)
            out[p.fixed_indices] = c[:count] / self._scale
            values = c[count::2] / pair + 1j * (c[count + 1::2] / pair)
            out[p.paired_indices] = values
            out[written] = np.conj(values[retained])
            return out.reshape(p.rfft_shape)
        batch = c.shape[:-1]
        out = np.zeros(batch + (entries,), dtype=complex)
        out[..., p.fixed_indices] = c[..., :count] / self._scale
        values = c[..., count::2] / pair + 1j * (c[..., count + 1::2] / pair)
        out[..., p.paired_indices] = values
        out[..., written] = np.conj(values[..., retained])
        return out.reshape(batch + p.rfft_shape)

    def analyse_grid(self, values: ArrayLike) -> np.ndarray:
        """Components, shape `(..., N)`, of values on the grid, shape
        `(..., *shape)`, through scipy's rfftn: the inverse of `synthesise_grid`."""
        v = np.asarray(values, dtype=float)
        d = self.dimension
        if v.ndim < d or v.shape[v.ndim - d:] != self.shape:
            raise ValueError(f"values on a grid of shape {self.shape} have that "
                             f"trailing shape; got {v.shape}")
        return self.from_rfft(rfftn(v, s=self.shape, axes=tuple(range(-d, 0))))

    def synthesise_grid(self, components: ArrayLike) -> np.ndarray:
        """Values on the grid, shape `(..., *shape)`, of components, shape
        `(..., N)`, through scipy's irfftn."""
        d = self.dimension
        return irfftn(self.to_rfft(components), s=self.shape, axes=tuple(range(-d, 0)))

    def derivative(self, components: ArrayLike, axis: int) -> np.ndarray:
        """The components of d/dx along `axis` of the series: each pair's
        cosine and sine rotate into one another times 2 pi k / L, and a
        self-conjugate mode goes to zero (the constant's derivative, and a
        Nyquist cosine's, a sine the grid does not hold)."""
        if not 0 <= int(axis) < self.dimension:
            raise ValueError(f"a box of {self.dimension} axes has no axis {axis}")
        c = self._components(components)
        count = self._packing.fixed_indices.size
        rate = self.wavenumbers[int(axis)][count::2]
        out = np.zeros_like(c)
        out[..., count::2] = -rate * c[..., count + 1::2]
        out[..., count + 1::2] = rate * c[..., count::2]
        return out

    def transfer(self, components: ArrayLike, target: Transform) -> np.ndarray:
        """The components on `target`, a grid of another shape on the same
        domain, of the series with these: matched by signed wavevector and
        phase (`Packing.matching`), the modes `target` lacks dropped and
        those it alone has zero, a Nyquist cosine keeping its function."""
        if target.domain != self._domain:
            raise ValueError(f"a transfer is between grids of one domain; "
                             f"{target.domain!r} is not {self._domain!r}")
        c = self._components(components)
        here, there, scale = self._packing.matching(target.packing)
        out = np.zeros(c.shape[:-1] + (target.size,))
        out[..., there] = scale * c[..., here]
        return out

    # -- points --------------------------------------------------------------------

    def _angles(self, points: ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
        """The points' angles 2 pi (x - origin) / L in [0, 2 pi], shape
        (P, D), with the points' own shape."""
        x = self._domain._points(points)
        lengths = np.asarray(self.lengths)
        theta = 2.0 * np.pi * (np.mod(x - np.asarray(self.origin), lengths) / lengths)
        return theta.reshape(-1, self.dimension), x.shape[:-1]

    def basis(self, points: ArrayLike) -> np.ndarray:
        """Every basis function at points of shape (..., D): shape (..., N),
        to rounding.  A self-conjugate mode is cos(theta_k) / sqrt V and a
        pair sqrt(2 / V) (cos theta_k, -sin theta_k), theta_k with the
        representative's wavevector."""
        theta, shape = self._angles(points)
        return self._basis(theta).reshape(shape + (self.size,))

    def _basis(self, theta: np.ndarray) -> np.ndarray:
        """The basis at angles (P, D): shape (P, N)."""
        phase = theta @ self.wavevectors
        count = self._packing.fixed_indices.size
        out = np.empty_like(phase)
        out[:, :count] = np.cos(phase[:, :count]) / np.sqrt(self.volume)
        amplitude = np.sqrt(2.0 / self.volume)
        out[:, count::2] = amplitude * np.cos(phase[:, count::2])
        out[:, count + 1::2] = -amplitude * np.sin(phase[:, count + 1::2])
        return out

    def _basis_gradient(self, theta: np.ndarray) -> np.ndarray:
        """d/dx_i of the basis at angles (P, D): shape (P, D, N)."""
        phase = theta @ self.wavevectors
        count = self._packing.fixed_indices.size
        slope = np.empty_like(phase)                 # d/dtheta_k of each function
        slope[:, :count] = -np.sin(phase[:, :count]) / np.sqrt(self.volume)
        amplitude = np.sqrt(2.0 / self.volume)
        slope[:, count::2] = -amplitude * np.sin(phase[:, count::2])
        slope[:, count + 1::2] = -amplitude * np.cos(phase[:, count + 1::2])
        return slope[:, None, :] * self.wavenumbers[None, :, :]

    def _route(self, route: str) -> str:
        """The route, checked: finufft's goes to three dimensions."""
        if route not in ROUTES:
            raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
        if route == "nufft" and self.dimension > 3:
            raise ValueError(f"finufft transforms one to three dimensions; a box of "
                             f"{self.dimension} axes takes route='direct'")
        return route

    @cached_property
    def _layout(self) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
        """(sizes, plus, minus, amplitude): finufft's spectrum, two larger
        than the grid along every axis so that a Nyquist wavenumber +n / 2
        and its conjugate -n / 2 are distinct entries (folding the first
        onto the second changes its phase off the grid), and the flat
        entries of +k and -k and the basis amplitude of each mode, the
        self-conjugate modes and then the pairs' representatives."""
        p = self._packing
        sizes = tuple(n + 2 for n in self.shape)
        offsets = [m // 2 for m in sizes]
        count = p.fixed_indices.size
        unique = np.concatenate([np.arange(count), np.arange(count, self.size, 2)])
        k = p.wavevectors[:, unique]
        d = self.dimension
        plus = np.ravel_multi_index([k[i] + offsets[i] for i in range(d)], sizes)
        minus = np.ravel_multi_index([offsets[i] - k[i] for i in range(d)], sizes)
        amplitude = np.concatenate([np.full(count, 1.0 / np.sqrt(self.volume)),
                                    np.full(unique.size - count,
                                            np.sqrt(2.0 / self.volume))])
        return sizes, plus, minus, amplitude

    def _spectrum(self, c: np.ndarray) -> np.ndarray:
        """finufft's spectrum, shape (B, *sizes), of components (B, N):
        half of each mode's complex amplitude at +k and its conjugate at -k,
        which coincide for the constant alone."""
        sizes, plus, minus, amplitude = self._layout
        count = self._packing.fixed_indices.size
        real = np.concatenate([c[:, :count], c[:, count::2]], axis=1)
        imaginary = np.concatenate([np.zeros((c.shape[0], count)),
                                    c[:, count + 1::2]], axis=1)
        weights = amplitude * (real + 1j * imaginary)
        spectrum = np.zeros((c.shape[0], int(np.prod(sizes))), dtype=complex)
        spectrum[:, plus] = 0.5 * weights
        spectrum[:, minus] += 0.5 * np.conj(weights)
        return spectrum.reshape((c.shape[0],) + sizes)

    def _flat(self, components: ArrayLike) -> tuple[np.ndarray, tuple[int, ...]]:
        """Components as (B, N), with the batch shape."""
        c = self._components(components)
        return c.reshape(-1, self.size), c.shape[:-1]

    def at(self, points: ArrayLike, *, route: str = "nufft",
           epsilon: float = POINT_EPSILON, nthreads: int = NUFFT_THREADS
           ) -> PointTransform:
        """The transforms at points of shape (..., D), prepared once to be
        applied many times: a `PointTransform`, whose `synthesise`,
        `synthesise_gradient` and `accumulate` take no points and give the
        results of the methods of those names here, by `route`, to
        `epsilon` with `nthreads` threads on finufft's."""
        return PointTransform(self, points, route=route, epsilon=epsilon,
                              nthreads=nthreads)

    def synthesise(self, components: ArrayLike, points: ArrayLike, *,
                   route: str = "nufft", epsilon: float = POINT_EPSILON,
                   nthreads: int = NUFFT_THREADS) -> np.ndarray:
        """The series of `components`, shape (..., N), at points of shape
        (..., D): the batch axes and then the points' shape.

        By finufft's type-2 transform, to about `epsilon` of the field's
        size with `nthreads` threads, or with `route="direct"` by the sum
        over `basis`, to rounding at N per point.  Points used more than
        once are prepared once by `at`.
        """
        route = self._route(route)
        c, batch = self._flat(components)
        prepared = self.at(points, route=route, epsilon=epsilon, nthreads=nthreads)
        return prepared._values(c).reshape(batch + prepared.shape)

    def synthesise_gradient(self, components: ArrayLike, points: ArrayLike, *,
                            route: str = "nufft", epsilon: float = POINT_EPSILON,
                            nthreads: int = NUFFT_THREADS) -> np.ndarray:
        """d/dx_i of the series along each axis at points of shape (..., D):
        the batch axes, the points' shape, then D.  The spectrum times
        i 2 pi k_i / L_i, every axis in one batched type-2 transform, or by
        the direct sum; to the accuracy of `synthesise`.  Points used more
        than once are prepared once by `at`."""
        route = self._route(route)
        c, batch = self._flat(components)
        prepared = self.at(points, route=route, epsilon=epsilon, nthreads=nthreads)
        return prepared._gradient(c).reshape(batch + prepared.shape
                                             + (self.dimension,))

    def accumulate(self, weights: ArrayLike, points: ArrayLike, *,
                   route: str = "nufft", epsilon: float = POINT_EPSILON,
                   nthreads: int = NUFFT_THREADS) -> np.ndarray:
        """sum_j w_j phi(x_j) for every basis function phi: the adjoint of
        `synthesise`, components of shape `(..., N)` from `weights` of shape
        `(..., *P)` at points of shape `(*P, D)`.

        By finufft's type-1 transform, the adjoint of the type 2 that
        `synthesise` makes, to about `epsilon`, or by the direct sum.
        Points used more than once are prepared once by `at`.
        """
        return self.at(points, route=route, epsilon=epsilon,
                       nthreads=nthreads).accumulate(weights)

    def __repr__(self) -> str:
        return f"Transform({self.shape}, {self._domain!r})"


class PointTransform:
    """The transforms of a `Transform` at one set of points, prepared once
    to be applied many times (`Transform.at`): `synthesise`,
    `synthesise_gradient` and `accumulate` without the points, giving the
    results of the transform's methods of those names, which are each one
    application of such a preparation.

    The points, of shape (..., D), are taken to their angles in the box on
    construction.  On finufft's route the first application of each type
    of transform (2 for the series and its gradient, 1 for the adjoint) to
    each number of fields at once makes finufft's plan for it, the points
    sorted and its upsampled grid allocated (about 2^D times the spectrum,
    complex, for each thread), and keeps it.  On the direct route the
    angles are all that is kept, the basis summed at each application, N
    per point.  Applications from several threads take turns on the plans;
    a copy or a pickle leaves them behind, to be made again on first use.
    """

    def __init__(self, transform: Transform, points: ArrayLike, *,
                 route: str = "nufft", epsilon: float = POINT_EPSILON,
                 nthreads: int = NUFFT_THREADS) -> None:
        self._transform = transform
        self._route = transform._route(route)
        theta, self._shape = transform._angles(points)
        self._theta = _readonly(theta)
        self._axes = (tuple(_readonly(np.ascontiguousarray(theta[:, i]))
                            for i in range(transform.dimension))
                      if self._route == "nufft" else ())
        self._epsilon = epsilon
        self._nthreads = nthreads
        self._plans: dict[tuple[int, int], Any] = {}
        self._lock = threading.Lock()

    @property
    def transform(self) -> Transform:
        """The transform whose series are taken at the points."""
        return self._transform

    @property
    def route(self) -> str:
        """The route of every application: one of `ROUTES`."""
        return self._route

    @property
    def shape(self) -> tuple[int, ...]:
        """The points' shape, their coordinates' axis left out."""
        return self._shape

    def synthesise(self, components: ArrayLike) -> np.ndarray:
        """The series of `components`, shape (..., N), at the points: the
        batch axes and then the points' shape (`Transform.synthesise`)."""
        c, batch = self._transform._flat(components)
        return self._values(c).reshape(batch + self._shape)

    def synthesise_gradient(self, components: ArrayLike) -> np.ndarray:
        """d/dx_i of the series of `components` along each axis at the
        points: the batch axes, the points' shape, then D
        (`Transform.synthesise_gradient`)."""
        c, batch = self._transform._flat(components)
        return self._gradient(c).reshape(batch + self._shape
                                         + (self._transform.dimension,))

    def accumulate(self, weights: ArrayLike) -> np.ndarray:
        """sum_j w_j phi(x_j) for every basis function phi, components of
        shape `(..., N)` from `weights` of shape `(..., *P)`, P the points'
        shape: the adjoint of `synthesise` (`Transform.accumulate`)."""
        w = np.asarray(weights, dtype=float)
        lead = w.ndim - len(self._shape)
        if lead < 0 or w.shape[lead:] != self._shape:
            raise ValueError(f"weights at points of shape {self._shape} have that "
                             f"trailing shape; got {w.shape}")
        batch = w.shape[:lead]
        w = w.reshape(math.prod(batch), self._theta.shape[0])
        return self._sums(w).reshape(batch + (self._transform.size,))

    def _values(self, c: np.ndarray) -> np.ndarray:
        """The series of components (B, N) at the points: shape (B, P)."""
        t = self._transform
        if self._theta.shape[0] == 0:
            return np.zeros((c.shape[0], 0))
        if self._route == "direct":
            return c @ t._basis(self._theta).T
        return self._type2(t._spectrum(c))

    def _gradient(self, c: np.ndarray) -> np.ndarray:
        """d/dx_i of the series of components (B, N) at the points: shape
        (B, P, D), the spectrum times i 2 pi k_i / L_i, every axis in one
        batched type-2 transform."""
        t = self._transform
        d = t.dimension
        if self._theta.shape[0] == 0:
            return np.zeros((c.shape[0], 0, d))
        if self._route == "direct":
            return np.einsum("bn,pdn->bpd", c, t._basis_gradient(self._theta))
        sizes = t._layout[0]
        spectrum = t._spectrum(c)
        stacked = np.empty((c.shape[0], d) + sizes, dtype=complex)
        for axis in range(d):
            k = np.arange(sizes[axis]) - sizes[axis] // 2
            factor = (1j * 2.0 * np.pi / t.lengths[axis]) * k
            along = [1] * d
            along[axis] = sizes[axis]
            stacked[:, axis] = spectrum * factor.reshape(along)
        values = self._type2(stacked.reshape((-1,) + sizes))
        return values.reshape(c.shape[0], d, -1).transpose(0, 2, 1)

    def _sums(self, w: np.ndarray) -> np.ndarray:
        """sum_j w_j phi(x_j) of weights (B, P): components (B, N)."""
        t = self._transform
        if self._theta.shape[0] == 0:
            return np.zeros((w.shape[0], t.size))
        if self._route == "direct":
            return w @ t._basis(self._theta)
        return self._type1(w)

    def _plan(self, kind: int, count: int) -> Any:
        """finufft's plan of type `kind`, 2 with exp(+i k . theta) or 1 with
        exp(-i k . theta), for `count` transforms at once at the points:
        made on first use and kept.  The caller holds the lock."""
        plan = self._plans.get((kind, count))
        if plan is None:
            plan = _finufft().Plan(kind, self._transform._layout[0], n_trans=count,
                                   eps=float(self._epsilon),
                                   isign=1 if kind == 2 else -1,
                                   nthreads=int(self._nthreads))
            plan.setpts(*self._axes)
            self._plans[kind, count] = plan
        return plan

    def _type2(self, spectra: np.ndarray) -> np.ndarray:
        """finufft's type 2, sum_k f_k exp(+i k . theta), of spectra (B,
        *sizes) at the points: shape (B, P), real parts."""
        with self._lock:
            values = self._plan(2, spectra.shape[0]).execute(
                np.ascontiguousarray(spectra))
        return np.asarray(values).reshape(spectra.shape[0], -1).real

    def _type1(self, w: np.ndarray) -> np.ndarray:
        """Components (B, N) of finufft's type 1, sum_j w_j exp(-i k .
        theta_j), read at each mode's +k: a pair's cosine from the real part
        and its sine from the imaginary part, the sine's sign and the
        transform's cancelling."""
        t = self._transform
        _, plus, _, amplitude = t._layout
        with self._lock:
            spectrum = self._plan(1, w.shape[0]).execute(
                np.ascontiguousarray(w, dtype=complex))
        at = np.asarray(spectrum).reshape(w.shape[0], -1)[:, plus]
        count = t.packing.fixed_indices.size
        out = np.empty((w.shape[0], t.size))
        out[:, :count] = amplitude[:count] * at[:, :count].real
        out[:, count::2] = amplitude[count:] * at[:, count:].real
        out[:, count + 1::2] = amplitude[count:] * at[:, count:].imag
        return out

    def __getstate__(self) -> dict[str, Any]:
        """The state a copy or a pickle takes: finufft's plans and the lock
        left behind."""
        state = self.__dict__.copy()
        state["_plans"] = {}
        del state["_lock"]
        return state

    def __setstate__(self, state: dict[str, Any]) -> None:
        """The state restored, with a lock of its own and no plans yet."""
        self.__dict__.update(state)
        self._lock = threading.Lock()

    def __repr__(self) -> str:
        return (f"PointTransform({self._transform!r}, points of shape {self._shape}, "
                f"route={self._route!r})")
