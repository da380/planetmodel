"""Slices of a flat model: its field on the points one looks at it by.

The counterpart of `planetmodel.slices` for a model of
`planetmodel.cartesian`: a horizontal plane at a depth on the nodes of a
`cartesian.Grid` (`plane`), a vertical section along a straight line,
distance along it by depth (`section`, to an `end=` or at an
`azimuth=` for a `length=`), a profile down the vertical at a point
(`profile`), the horizontal mean as a profile, with its spread over the
plane at each depth (`mean_profile`), a function of (x, y) on one of the
model's interfaces (`surface`), a field integrated down each vertical
(`integrated`), and the box seen from outside, its top and the two
sides meeting at a corner (`block`, the counterpart of the sphere's
cutaway).  Drawing them is `planetmodel.plotting`'s business; nothing
here draws.

**What is shared.**  What a slice is of and how it departs is the same
concept in either system and is `planetmodel.slices`' own: what is
sliced (a field by name, or fields by layer), a vector's or a tensor's
component in a frame (here `"local"`, (e_up, e_x, e_y), or
`"cartesian"`) or a vector's `"norm"`, and `relative_to=` with
`departure=` (`slices.evaluator`); what a slice shows (`Shown`); one
layer's part of a section or a profile (`SliceLayer`, whose `radius`
holds the layered coordinate, here the depth); the side of an interface
(`slices.layers_of`); the passage between reference and physical points
(`slices.to_reference`, `slices.to_physical`); each layer's part of a
section or a profile (`slices.runs`, down verticals here and along rays
on the sphere); and `slices.normalised`,
which takes these slices as it takes the sphere's.  A departure from
the mean is from the field's horizontal mean at each point's reference
depth (`cartesian.horizontal_mean`), its own where it has one and by
the quadrature of `mean_grid` (a `cartesian.Grid`) otherwise, a plane
taking its own grid for that by default; `"reference"` and another
`cartesian.Model` are as on the sphere.  Every slice states its
coordinates (`FLAT`), which is how the plotting knows to draw depth
downward.

**Points.**  Points are z up and the depth is -z (`FLAT`); lengths are
the domain's.  A plane's points are a grid's nodes at one depth, over a
`Rectangle` the nodes of x alone, the body being the same at every y.
A section runs from `start` (x, y) to `end`, or from `start` at an
`azimuth`, measured from north (+y) toward east (+x) as on a map, for a
`length`; `distance` is the length along it.  `path` holds its points
on the reference surface z = 0 and `plane` the orthonormal pair (along
the line, down) spanning its vertical plane, so that
(X - path[0]) @ plane.T is (distance, depth) for any point X in it:
the coordinates `plotting.vertical_section` draws in and
`plotting.plane_cut(points, cells, path[0], plane)` cuts a mesh in, so
that a section of a model and the section of its mesh compose.

**Reference or physical.**  As on the sphere: a slice taken at reference
points (`at="reference"`) evaluates the fields where they live and
gives the points' physical positions for drawing the body as it is; at
physical points (`at="physical"`: a true plane z = -depth, a true
vertical) each point is pulled back through the mapping's inverse, and
a point above the physical surface or below the base is NaN.  A
physical section or profile needs the mapping to keep each vertical (x
and y), as a vertical stretch does, so that each layer's part of a
vertical is one run of depths, between where the mapping puts the
layer's boundaries on it, as on the sphere; under topography its first
row is the physical surface, raised or lowered, nothing above it
sampled.

**Interfaces.**  A section or a profile keeps both sides of an
interface, each layer's numbers including its own ends.  A plane is
asked for `side="below"` (the deeper layer) or `"above"` (the
shallower), below and above as the body is, required where its depth
lies on an interface and of no effect elsewhere.

**Spectra.**  `spectrum` analyses a plane slice into the real basis of
its grid, orthonormal over the period (`fourier.Transform.analyse_grid`,
the components c_i), and gives the power in bins of horizontal
wavenumber |k|,

    P_j = sum over the components i of bin j of c_i^2 / A,

A the period's area (its length over a rectangle), so that the sum over
the bins is the mean square of the slice over the grid's nodes
(`Grid.mean` of its square), exactly: Parseval under the grid's
quadrature.  For a field the grid resolves, its Nyquist components
zero, that is the mean square over the period.  It is the counterpart
of the sphere's P_l = sum_m c_lm^2 / (4 pi).  Bin j holds the components
with j <= |k| / dk < j + 1, dk = 2 pi / L the fundamental wavenumber of
the longest period L (`wavenumber_bins`): over equal periods, the
integer wavevector's length rounded down, `Packing.degrees`.  Its
`wavenumbers` are j dk, and its wavelength 2 pi / (j dk).  A field not
limited to the grid's band aliases into it, so the grid is taken finer
than the field's band.  `correlation` gives two fields' correlation by
bin at each of a list of depths, r_j = sum a_i b_i / sqrt(sum a_i^2 sum
b_i^2) over the components of bin j, and over the bins from 1 up, NaN
where a bin has no power in either (`slices.NO_POWER` of the field's
amplitude).

**Blocks.**  `block` is the box from its top at the depth its span
starts at (the free surface by default) to the depth it ends at, seen
from the `corner` toward the viewer: the top and the two sides meeting
at that corner, each a set of `SliceLayer`s, the sides layer by layer
so that a jump at an interface is a sharp edge.  It is taken at
reference points and gives their physical positions, so that under
topography it is drawn in the body's shape.

Nothing here knows about units, and memory is bounded: a spread over a
grid is taken a block of depths at a time.
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from ..coordinates import Coordinates
from ..fields import Field
from ..slices import (AT, NO_POWER, SIDES, Fields, MeanOf, RelativeTo, Shown,
                      SliceLayer, Values, evaluate_layers, evaluator, field_of,
                      layer_quadrature, layers_of, runs, to_physical, to_reference)
from ..units import Dimensions
from .coordinates import FLAT
from .domains import Box
from .grid import Grid
from .means import horizontal_mean
from .model import Model

__all__ = ["PLANE_NODES", "SPREADS", "CORNERS", "PlaneSlice", "SectionSlice",
           "ProfileSlice", "BlockSlice", "Spectrum", "Correlation", "plane",
           "section", "profile", "mean_profile", "surface", "integrated", "block",
           "evaluate", "spectrum", "spectrum_of", "correlation", "wavenumber_bins"]

#: The nodes along each axis of a plane's grid when none is given.
PLANE_NODES = 128

#: The spread of a mean profile: one standard deviation about the mean, or
#: the least and the greatest value, over the plane at each depth.
SPREADS = ("std", "range")

#: The corners a block may be seen from, by compass: x east and y north.
CORNERS = ("southeast", "southwest", "northeast", "northwest")

#: The most values a spread over a grid evaluates at once (32 MB of doubles).
_BLOCK_VALUES = 1 << 22


# -- the slices -------------------------------------------------------------------

@dataclass(frozen=True)
class PlaneSlice:
    """A field on a horizontal plane: `values` (grid.shape) on `grid`'s nodes
    at `depth`.

    `depth` is a reference depth, or a physical one when `at` is
    `"physical"`; `reference` and `physical` are the points' Cartesian
    positions, grid.shape + (3,), in the two bodies; `side` the side of
    an interface the slice was taken on, where it lies on one.
    """

    depth: float
    grid: Grid
    values: np.ndarray
    reference: np.ndarray
    physical: np.ndarray
    shown: Shown
    _: KW_ONLY
    at: str = "reference"
    side: str | None = None

    #: The coordinate system the slice's points are in.
    coordinates: ClassVar[Coordinates] = FLAT

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label


@dataclass(frozen=True)
class SectionSlice:
    """A field on a vertical section along a straight line, layer by layer.

    `distance` (m,) is the length along the line from its start; `path`
    (m, 3) its points on the reference surface z = 0; `plane` (2, 3) the
    direction along the line and the downward vertical, the orthonormal
    pair spanning its plane, so that (X - path[0]) @ plane.T is
    (distance, depth); `layers` each layer's points and values, their
    `radius` the depths, (n, m).
    """

    distance: np.ndarray
    path: np.ndarray
    plane: np.ndarray
    layers: tuple[SliceLayer, ...]
    shown: Shown
    _: KW_ONLY
    at: str = "reference"

    #: The coordinate system the slice's points are in.
    coordinates: ClassVar[Coordinates] = FLAT

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label

    @property
    def length(self) -> float:
        """The section's length."""
        return float(self.distance[-1])


@dataclass(frozen=True)
class ProfileSlice:
    """A field down the vertical at (x, y), layer by layer, the layers'
    `radius` the depths; for a mean profile `x` and `y` are None, the
    values are the horizontal means, and `spread` names the spread of
    `bands`, each layer's (low, high) at its depths, or None for none."""

    x: float | None
    y: float | None
    layers: tuple[SliceLayer, ...]
    shown: Shown
    _: KW_ONLY
    at: str = "reference"
    spread: str | None = None
    bands: tuple[tuple[np.ndarray, np.ndarray] | None, ...] | None = None

    #: The coordinate system the slice's points are in.
    coordinates: ClassVar[Coordinates] = FLAT

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label


@dataclass(frozen=True)
class BlockSlice:
    """A field on the faces of a box seen from outside: its top and the two
    sides meeting at `corner`, from the depth `depths[0]` to `depths[1]`.

    `layers` are the pieces of the faces, each a `SliceLayer`, and
    `faces` names the face of each, `"top"` or a side by compass
    (`"south"`, the side y = y0, `"east"`, the side x = x1, and so on):
    the top first, (m, m) points over x by y at the top depth, both ends
    of each axis included, then each side layer by layer from the top
    down, (n, m) points by depth along the side.  `boundaries` are the
    depths of the model's boundaries within the span, for drawing them
    on the sides.
    """

    corner: str
    depths: tuple[float, float]
    faces: tuple[str, ...]
    layers: tuple[SliceLayer, ...]
    boundaries: np.ndarray
    shown: Shown
    _: KW_ONLY
    at: str = "reference"

    #: The coordinate system the slice's points are in.
    coordinates: ClassVar[Coordinates] = FLAT

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label

    def face(self, name: str) -> tuple[SliceLayer, ...]:
        """The pieces of the face `name`, one of `faces`."""
        if name not in self.faces:
            raise ValueError(f"name must be one of {tuple(dict.fromkeys(self.faces))}, "
                             f"got {name!r}")
        return tuple(lay for f, lay in zip(self.faces, self.layers) if f == name)


@dataclass(frozen=True)
class Spectrum:
    """A field's power by bin of horizontal wavenumber on a plane.

    `degrees` are the bins 0 .. J, and `wavenumbers` their |k|, j dk;
    `power` P_j, summing to the mean square over the grid's nodes;
    `coefficients` the slice's components on `grid` (N,), the real
    orthonormal ones of `grid.transform`; `depth`, `side` and `at` as the
    plane slice they were analysed from.
    """

    depth: float
    degrees: np.ndarray
    wavenumbers: np.ndarray
    power: np.ndarray
    coefficients: np.ndarray
    grid: Grid
    shown: Shown
    _: KW_ONLY
    at: str = "reference"
    side: str | None = None

    #: The coordinate system the slice's points are in.
    coordinates: ClassVar[Coordinates] = FLAT

    @property
    def amplitude(self) -> np.ndarray:
        """The square root of the power, by bin."""
        return np.sqrt(self.power)

    @property
    def spacing(self) -> float:
        """dk, the width of a bin: 2 pi over the longest period."""
        return float(2.0 * np.pi / max(self.grid.lengths))

    @property
    def wavelengths(self) -> np.ndarray:
        """2 pi / |k| of each bin, infinite for the mean's."""
        with np.errstate(divide="ignore"):
            return np.where(self.wavenumbers > 0.0, 2.0 * np.pi / self.wavenumbers,
                            np.inf)

    @property
    def label(self) -> str:
        """The label of what the spectrum is of."""
        return self.shown.label


@dataclass(frozen=True)
class Correlation:
    """Two fields' correlation by bin of horizontal wavenumber and by depth.

    `depths` (k,), `degrees` the bins and `wavenumbers` their |k|;
    `by_degree` (k, J + 1), r_j at each depth; `total` (k,), over the
    bins from 1 up; `first` and `second` the two fields' spectra at each
    depth.
    """

    depths: np.ndarray
    degrees: np.ndarray
    wavenumbers: np.ndarray
    by_degree: np.ndarray
    total: np.ndarray
    first: tuple[Spectrum, ...]
    second: tuple[Spectrum, ...]

    #: The coordinate system the slice's points are in.
    coordinates: ClassVar[Coordinates] = FLAT

    @property
    def radii(self) -> np.ndarray:
        """The depths, under the name the sphere's correlation gives its
        layered coordinate."""
        return self.depths

    @property
    def label(self) -> str:
        """What is correlated with what, by name."""
        a = self.first[0].shown.name or "field"
        b = self.second[0].shown.name or "field"
        return f"correlation of {a}" if a == b else f"correlation of {a} with {b}"


# -- what is evaluated ------------------------------------------------------------

def _flat(model: Model) -> None:
    """Refuse anything but a flat model, by name."""
    if not isinstance(model, Model):
        raise TypeError(f"expected a cartesian.Model, got {type(model).__name__}")


def _horizontal_means(mean_grid: Grid | None) -> MeanOf:
    """The horizontal mean of a layer's field, its own or by `mean_grid`'s
    quadrature, a field with neither refused by name."""
    def mean(field: Field, layer: int) -> Field:
        if (mean_grid is None and getattr(field, "horizontal_mean", None) is None
                and not getattr(field, "is_radial", False)):
            raise ValueError(f"layer {layer}'s {field!r} has no horizontal mean of its "
                             "own: pass mean_grid=, a cartesian.Grid, to take it by "
                             "quadrature")
        return horizontal_mean(field, grid=mean_grid)
    return mean


def _values(model: Model, fields: Fields, *, component: int | str | None,
            frame: str | None, relative_to: RelativeTo, departure: str | None,
            mean_grid: Grid | None) -> tuple[Values, Shown]:
    """`slices.evaluator` with the horizontal mean."""
    _flat(model)
    return evaluator(model, fields, component=component, frame=frame,
                     relative_to=relative_to, departure=departure,
                     mean=_horizontal_means(mean_grid))


def _check(at: str, side: str | None) -> None:
    if at not in AT:
        raise ValueError(f"at must be one of {AT}, got {at!r}")
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")


def _grid_of(model: Model, grid: Grid | None) -> Grid:
    """The grid given, or `PLANE_NODES` along each axis of the model's
    domain; anything but a flat model refused."""
    _flat(model)
    if grid is None:
        return Grid(model.geometry.domain, PLANE_NODES)
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    return grid


def _nodes(grid: Grid) -> tuple[np.ndarray, np.ndarray]:
    """x and y at every node, each of the grid's shape."""
    x, y = grid.nodes
    x, y = np.broadcast_arrays(x, y)
    return np.array(x), np.array(y)


# -- the slices -------------------------------------------------------------------

def plane(model: Model, fields: Fields, *, depth: float, grid: Grid | None = None,
          side: str | None = None, at: str = "reference",
          component: int | str | None = None, frame: str | None = None,
          relative_to: RelativeTo = None, departure: str | None = None,
          mean_grid: Grid | None = None) -> PlaneSlice:
    """A field on the horizontal plane at `depth`, by the rules of the module
    docstring.

    `grid` is a `cartesian.Grid`, `PLANE_NODES` along each axis of the
    model's domain by default; `side` the side of an interface the depth
    lies on; `at` reference or physical points; the component, frame
    and departure as `planetmodel.slices` takes them, the mean of a field
    without one of its own taken on `mean_grid`, `grid` itself by
    default.
    """
    _check(at, side)
    grid = _grid_of(model, grid)
    values, shown = _values(model, fields, component=component, frame=frame,
                            relative_to=relative_to, departure=departure,
                            mean_grid=grid if mean_grid is None else mean_grid)
    d = float(depth)
    x, y = _nodes(grid)
    points = FLAT.points(d, x, y)
    if at == "reference":
        layers, sided = layers_of(model, np.array([d]), side)
        if layers[0] < 0:
            b = model.skeleton.boundaries
            raise ValueError(f"depth {d:g} lies outside the body, [{b[0]:g}, "
                             f"{b[-1]:g}]")
        v = values(int(layers[0]), np.full(x.shape, d), x, y)
        reference, physical = points, to_physical(model, points)
    else:
        physical = points
        reference = to_reference(model, points)
        layers, sided = layers_of(model, FLAT.split(reference)[0], side)
        if np.all(layers < 0):
            raise ValueError(f"the physical plane at depth {d:g} lies outside the body")
        v = evaluate_layers(values, layers, reference, FLAT)
    return PlaneSlice(d, grid, v, reference, physical, shown, at=at,
                      side=side if sided else None)


def _line(start: tuple[float, float], end: tuple[float, float] | None,
          azimuth: float | None, length: float | None, m: int
          ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distance along the line, its points on the reference surface and
    the plane's pair (along the line, down)."""
    a = np.array([float(v) for v in start])
    if a.shape != (2,):
        raise ValueError(f"a section starts at a point (x, y), got {start!r}")
    if (end is None) == (azimuth is None):
        raise ValueError("give the section's end, or its azimuth= and length=")
    if end is not None:
        if length is not None:
            raise ValueError("the section's length follows from its end; give one")
        b = np.array([float(v) for v in end])
        if b.shape != (2,):
            raise ValueError(f"a section ends at a point (x, y), got {end!r}")
        span = float(np.linalg.norm(b - a))
        if not span > 0.0:
            raise ValueError("the section's ends are the same point, so no line is "
                             "defined between them; give azimuth= and length=")
        t = (b - a) / span
    else:
        if length is None or not float(length) > 0.0:
            raise ValueError("a section at a given azimuth needs length= > 0")
        span = float(length)
        t = np.array([np.sin(float(azimuth)), np.cos(float(azimuth))])
    if int(m) < 2:
        raise ValueError(f"m must be at least 2, got {m}")
    distance = np.linspace(0.0, span, int(m))
    xy = a[None, :] + distance[:, None] * t[None, :]
    path = np.stack([xy[:, 0], xy[:, 1], np.zeros(int(m))], axis=-1)
    plane_ = np.array([[t[0], t[1], 0.0], [0.0, 0.0, -1.0]])
    return distance, path, plane_


def _keeps_verticals(model: Model, X: np.ndarray) -> None:
    """Refuse a physical section or profile of a mapping that moves points
    off their verticals."""
    x = to_physical(model, X)
    size = max(model.skeleton.span, float(np.max(np.abs(X[..., :2]), initial=0.0)))
    if np.any(np.abs(x[..., :2] - X[..., :2]) > 1e-9 * size):
        raise ValueError("the mapping moves points off their verticals, so a physical "
                         "section or profile has no run of depths per layer; slice it "
                         "at reference points (at='reference')")


def _runs(model: Model, values: Values, path: np.ndarray,
          depths: tuple[float, float] | None, n: int, at: str
          ) -> tuple[SliceLayer, ...]:
    """Each layer's points and values down the verticals through `path`
    (m, 3): `slices.runs` in a box."""
    if depths is not None and not float(depths[0]) < float(depths[1]):
        raise ValueError(f"depths must be (lo, hi) with lo < hi, got {depths}")
    x, y = path[None, :, 0], path[None, :, 1]
    return runs(model, values, path, lambda d: FLAT.points(d, x, y), depths, n, at,
                keeps=_keeps_verticals)


def section(model: Model, fields: Fields, start: tuple[float, float], *,
            end: tuple[float, float] | None = None, azimuth: float | None = None,
            length: float | None = None, depths: tuple[float, float] | None = None,
            n: int = 50, m: int = 181, at: str = "reference",
            component: int | str | None = None, frame: str | None = None,
            relative_to: RelativeTo = None, departure: str | None = None,
            mean_grid: Grid | None = None) -> SectionSlice:
    """A field on a vertical section, by the rules of the module docstring.

    The line runs from `start` (x, y) to `end`, or from `start` at
    `azimuth` (in radians from north, +y, toward east, +x) for `length`,
    the one or the other given; `depths` (lo, hi) bounds it, the whole
    body by default; `n` points per layer, its ends included, and `m`
    along the line; `at`, the component, frame and departure as
    `plane` takes them, the mean of a field without one of its own taken
    on `mean_grid`.
    """
    _check(at, None)
    values, shown = _values(model, fields, component=component, frame=frame,
                            relative_to=relative_to, departure=departure,
                            mean_grid=mean_grid)
    distance, path, plane_ = _line(start, end, azimuth, length, m)
    layers = _runs(model, values, path, depths, n, at)
    return SectionSlice(distance, path, plane_, layers, shown, at=at)


def profile(model: Model, fields: Fields, x: float, y: float, *, n: int = 100,
            depths: tuple[float, float] | None = None, at: str = "reference",
            component: int | str | None = None, frame: str | None = None,
            relative_to: RelativeTo = None, departure: str | None = None,
            mean_grid: Grid | None = None) -> ProfileSlice:
    """A field down the vertical at (x, y), by the rules of the module
    docstring: `n` points per layer, its ends included, within `depths`
    (lo, hi), the whole body by default."""
    _check(at, None)
    values, shown = _values(model, fields, component=component, frame=frame,
                            relative_to=relative_to, departure=departure,
                            mean_grid=mean_grid)
    path = np.array([[float(x), float(y), 0.0]])
    layers = tuple(SliceLayer(lay.index, lay.radius[:, 0], lay.values[:, 0],
                              lay.reference[:, 0], lay.physical[:, 0])
                   for lay in _runs(model, values, path, depths, n, at))
    return ProfileSlice(float(x), float(y), layers, shown, at=at)


def _blocks(depths: np.ndarray, grid: Grid) -> list[np.ndarray]:
    """The depths cut into blocks whose values on `grid` number at most
    `_BLOCK_VALUES`."""
    per = max(1, _BLOCK_VALUES // int(np.prod(grid.shape)))
    return [depths[k:k + per] for k in range(0, depths.size, per)]


def _spread(field: Field, depths: np.ndarray, grid: Grid, mean: np.ndarray,
            spread: str) -> tuple[np.ndarray, np.ndarray]:
    """The band about a mean profile at `depths`: one standard deviation over
    the grid's nodes either side of `mean`, or the least and greatest value
    there."""
    x, y = grid.nodes
    d = grid.dimension
    low, high = np.empty(depths.size), np.empty(depths.size)
    at = 0
    for block in _blocks(depths, grid):
        v = np.asarray(field.evaluate(block.reshape((-1,) + (1,) * d), x[None],
                                      y[None]))
        if np.iscomplexobj(v):
            raise TypeError(f"{field!r} is complex; slice its real or imaginary part "
                            "as a field of its own")
        v = np.broadcast_to(v, (block.size,) + grid.shape)
        k = slice(at, at + block.size)
        if spread == "range":
            low[k] = v.min(axis=tuple(range(1, 1 + d)))
            high[k] = v.max(axis=tuple(range(1, 1 + d)))
        else:
            centre = grid.mean(v)
            dev = np.sqrt(grid.mean((v - centre.reshape((-1,) + (1,) * d)) ** 2))
            low[k], high[k] = mean[k] - dev, mean[k] + dev
        at += block.size
    return low, high


def mean_profile(model: Model, fields: Fields, *, n: int = 100,
                 mean_grid: Grid | None = None,
                 spread: str | None = None) -> ProfileSlice:
    """The horizontal mean of a scalar field down the depth, `n` points per
    layer, its ends included; a field without a mean of its own takes it
    on `mean_grid`.  With `spread`, one of `SPREADS`, the band about it
    over the nodes of `mean_grid`, which it then needs: one standard
    deviation either side of the mean, or the least and greatest values,
    at each depth."""
    _flat(model)
    if int(n) < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if spread is not None and spread not in SPREADS:
        raise ValueError(f"spread must be one of {SPREADS}, got {spread!r}")
    if spread is not None and not isinstance(mean_grid, Grid):
        raise ValueError("a spread is taken over the nodes of a grid: pass mean_grid=, "
                         "a cartesian.Grid")
    name = fields if isinstance(fields, str) else None
    if name is not None and not model.layers_with(name):
        raise KeyError(f"no layer holds a field {name!r}")
    spec = None if name is None else model.spec(name)
    shown = Shown(name, statistic="horizontal mean",
                  dimensions=None if spec is None else spec.dimensions,
                  scales=model.scales)
    means = _horizontal_means(mean_grid)
    (x0, x1), (y0, y1) = _extent(model)
    centre = (0.5 * (x0 + x1), 0.5 * (y0 + y1))
    out, bands = [], []
    for layer in range(model.nlayers):
        field = field_of(model, fields, layer)
        lo, hi = model.skeleton.interval(layer)
        d = np.linspace(lo, hi, int(n))
        X = FLAT.points(d, *centre)
        if field is None:
            out.append(SliceLayer(layer, d, np.full(d.shape, np.nan), X, X))
            bands.append(None)
            continue
        if field.character.rank:
            raise ValueError(f"layer {layer}'s field is of rank "
                             f"{field.character.rank}; a horizontal mean profile is "
                             "of a scalar")
        v = np.asarray(means(field, layer).evaluate(d, None, None))
        if np.iscomplexobj(v):
            raise TypeError(f"{field!r} is complex; slice its real or imaginary "
                            "part as a field of its own")
        v = np.array(np.broadcast_to(v, d.shape), dtype=float)
        out.append(SliceLayer(layer, d, v, X, X))
        bands.append(None if spread is None
                     else _spread(field, d, mean_grid, v, spread))
    return ProfileSlice(None, None, tuple(out), shown, spread=spread,
                        bands=None if spread is None else tuple(bands))


def _extent(model: Model) -> tuple[tuple[float, float], tuple[float, float]]:
    """The lateral extent of the model's body, a rectangle's y taken as
    (0, 0)."""
    extent = model.geometry.domain.extent
    if isinstance(model.geometry.domain, Box):
        return extent
    return extent, (0.0, 0.0)


def surface(model: Model, function: Callable[[np.ndarray, np.ndarray], ArrayLike],
            interface: int | str, *, name: str | None = None,
            grid: Grid | None = None) -> PlaneSlice:
    """A function of (x, y) on one of the model's interfaces, by index or
    name, as a plane slice at its reference depth: on `grid`, by default
    the function's own where it has one (a `GriddedFunction`'s, whose
    values the slice then holds at its nodes, or a `FourierFunction`'s)
    and `PLANE_NODES` along each axis otherwise; `name` the function's,
    its unit the model's spec of it."""
    _flat(model)
    face = model.geometry.interface(interface)
    own = getattr(function, "grid", None)
    grid = _grid_of(model, own if grid is None else grid)
    x, y = _nodes(grid)
    held = getattr(function, "values", None)
    v = (np.asarray(held) if own is not None and held is not None and own == grid
         else np.asarray(function(x, y)))
    if np.iscomplexobj(v):
        raise TypeError("the function is complex; slice its real or imaginary part")
    v = np.array(np.broadcast_to(v.astype(float, copy=False), x.shape))
    spec = None if name is None else model.spec(name)
    label = f"the interface {face.name}" if face.name else f"interface {face.index}"
    shown = Shown(name, statistic=f"on {label}",
                  dimensions=None if spec is None else spec.dimensions,
                  scales=model.scales)
    points = FLAT.points(face.radius, x, y)
    return PlaneSlice(face.radius, grid, v, points, to_physical(model, points), shown)


def integrated(model: Model, fields: Fields, *,
               depths: tuple[float, float] | None = None, grid: Grid | None = None,
               n: int = 32, component: int | str | None = None,
               frame: str | None = None) -> PlaneSlice:
    """A map of a field integrated down each vertical, the integral of f
    d depth over the reference depths `depths` (lo, hi), the whole body by
    default, on `grid`: within a layer by Gauss-Legendre on each of the
    field's pieces in depth, exact for a piecewise polynomial of their
    degree, or `n` points on the whole layer for a field that declares
    none; a layer lacking the field adds nothing.  The slice's depth is
    the span's top, its points there."""
    if int(n) < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    b = model.skeleton.boundaries
    lo, hi = (float(b[0]), float(b[-1])) if depths is None else map(float, depths)
    if not hi > lo:
        raise ValueError(f"depths must be (lo, hi) with lo < hi, got {depths}")
    grid = _grid_of(model, grid)
    values, shown = _values(model, fields, component=component, frame=frame,
                            relative_to=None, departure=None, mean_grid=None)
    x, y = _nodes(grid)
    total = np.zeros(x.shape)
    held = False
    for layer in range(model.nlayers):
        a, c = model.skeleton.interval(layer)
        a, c = max(a, lo), min(c, hi)
        field = field_of(model, fields, layer)
        if field is None or not c > a:
            continue
        held = True
        for d, w in layer_quadrature(field, a, c, n):
            for dk, wk in zip(d, w):
                total += wk * values(layer, np.full(x.shape, dk), x, y)
    if not held:
        raise ValueError(f"no layer within [{lo}, {hi}] holds the field")
    dims = shown.dimensions
    if dims is not None:
        dims = dims * Dimensions(length=1)
    shown = Shown(shown.name, component=component, frame=frame,
                  statistic="integrated down the vertical, d depth", dimensions=dims,
                  scales=model.scales)
    points = FLAT.points(lo, x, y)
    return PlaneSlice(lo, grid, total, points, to_physical(model, points), shown)


def block(model: Model, fields: Fields, *, depths: tuple[float, float] | None = None,
          corner: str = "southeast", n: int = 40, m: int = 81,
          component: int | str | None = None, frame: str | None = None,
          relative_to: RelativeTo = None, departure: str | None = None,
          mean_grid: Grid | None = None) -> BlockSlice:
    """The box from the depth `depths[0]` to `depths[1]` (the whole body by
    default) seen from `corner`, one of `CORNERS`, by the rules of the
    module docstring: its top, m by m points over the domain's extent,
    and the two sides meeting at the corner, `n` points per layer down
    and `m` along; the component, frame and departure as `plane` takes
    them, the mean of a field without one of its own taken on
    `mean_grid`.  A model over a `Rectangle` has no block."""
    _flat(model)
    domain = model.geometry.domain
    if not isinstance(domain, Box):
        raise ValueError("a block is of a body over a Box; a rectangle's body is "
                         "drawn by its vertical section")
    if corner not in CORNERS:
        raise ValueError(f"corner must be one of {CORNERS}, got {corner!r}")
    if int(m) < 2:
        raise ValueError(f"m must be at least 2, got {m}")
    values, shown = _values(model, fields, component=component, frame=frame,
                            relative_to=relative_to, departure=departure,
                            mean_grid=mean_grid)
    b = model.skeleton.boundaries
    lo, hi = (float(b[0]), float(b[-1])) if depths is None else map(float, depths)
    if not lo < hi:
        raise ValueError(f"depths must be (lo, hi) with lo < hi, got {depths}")
    (x0, x1), (y0, y1) = domain.extent
    layers, _ = layers_of(model, np.array([lo]), "below")
    if layers[0] < 0:
        raise ValueError(f"the block's top at depth {lo:g} lies outside the body")
    x, y = np.meshgrid(np.linspace(x0, x1, int(m)), np.linspace(y0, y1, int(m)),
                       indexing="ij")
    top = FLAT.points(lo, x, y)
    pieces = [SliceLayer(int(layers[0]), np.full(x.shape, lo),
                         values(int(layers[0]), np.full(x.shape, lo), x, y), top,
                         to_physical(model, top))]
    faces = ["top"]
    east, south = corner in ("southeast", "northeast"), corner.startswith("south")
    cx, cy = (x1 if east else x0), (y0 if south else y1)
    sides = ((("south" if south else "north"), (x0, cy), (x1, cy)),
             (("east" if east else "west"), (cx, y0), (cx, y1)))
    for name, start, end in sides:
        _, path, _ = _line(start, end, None, None, m)
        for lay in _runs(model, values, path, (lo, hi), n, "reference"):
            pieces.append(lay)
            faces.append(name)
    inside = b[(b > lo) & (b < hi)]
    return BlockSlice(corner, (lo, hi), tuple(faces), tuple(pieces),
                      np.array(inside), shown)


def evaluate(model: Model, fields: Fields, layer: int, depth: ArrayLike, x: ArrayLike,
             y: ArrayLike, *, component: int | str | None = None,
             frame: str | None = None, relative_to: RelativeTo = None,
             departure: str | None = None, mean_grid: Grid | None = None
             ) -> np.ndarray:
    """What a slice of the same choices shows, at reference points (depth,
    x, y) of a layer, broadcast: for a set of points of the caller's own;
    NaN where the layer lacks the field."""
    values, _ = _values(model, fields, component=component, frame=frame,
                        relative_to=relative_to, departure=departure,
                        mean_grid=mean_grid)
    if not 0 <= int(layer) < model.nlayers:
        raise IndexError(f"the model has {model.nlayers} layers, so no layer {layer}")
    depth, x, y = np.broadcast_arrays(np.asarray(depth, dtype=float),
                                      np.asarray(x, dtype=float),
                                      np.asarray(y, dtype=float))
    return values(int(layer), depth, x, y)


# -- spectra ----------------------------------------------------------------------

def wavenumber_bins(grid: Grid) -> tuple[np.ndarray, float]:
    """The bin of horizontal wavenumber of each of a grid's components, (N,),
    and the bins' width dk = 2 pi / L, L the longest period: bin j holds
    j <= |k| / dk < j + 1, the integer wavevector's length rounded down
    over equal periods (`fourier.Packing.degrees`)."""
    if not isinstance(grid, Grid):
        raise TypeError(f"grid must be a cartesian.Grid, got {type(grid).__name__}")
    longest = max(grid.lengths)
    k = grid.transform.wavevectors.astype(float)
    q = np.sqrt(sum((k[i] * (longest / length)) ** 2
                    for i, length in enumerate(grid.lengths)))
    # a length that is a whole number of dk is one to rounding
    return np.floor(q + 1e-9).astype(int), float(2.0 * np.pi / longest)


def spectrum(model: Model, fields: Fields, *, depth: float, grid: Grid | None = None,
             side: str | None = None, at: str = "reference",
             component: int | str | None = None, frame: str | None = None,
             relative_to: RelativeTo = None, departure: str | None = None,
             mean_grid: Grid | None = None) -> Spectrum:
    """A field's power by bin of horizontal wavenumber on the plane at
    `depth`, by the rules of the module docstring; the plane and the
    field's choices as `plane` takes them.  A slice with NaN in it (a
    layer lacking the field, a physical plane leaving the body) is
    refused."""
    return spectrum_of(plane(model, fields, depth=depth, grid=grid, side=side, at=at,
                             component=component, frame=frame,
                             relative_to=relative_to, departure=departure,
                             mean_grid=mean_grid))


def spectrum_of(slice_: PlaneSlice) -> Spectrum:
    """A plane slice's power by bin of horizontal wavenumber, analysed on
    its own grid; a slice with NaN in it is refused."""
    if not np.all(np.isfinite(slice_.values)):
        raise ValueError(f"{slice_.label} is not defined everywhere on the plane at "
                         f"depth {slice_.depth:g}, so it has no spectrum there")
    grid = slice_.grid
    c = grid.transform.analyse_grid(slice_.values)
    bins, step = wavenumber_bins(grid)
    power = np.bincount(bins, weights=c * c) / grid.area
    degrees = np.arange(power.size)
    return Spectrum(slice_.depth, degrees, degrees * step, power, c, grid,
                    slice_.shown, at=slice_.at, side=slice_.side)


def correlation(model: Model, fields: Fields, other: Model, *,
                other_fields: Fields | None = None, depths: ArrayLike,
                grid: Grid | None = None, side: str | None = None,
                at: str = "reference", component: int | str | None = None,
                frame: str | None = None, relative_to: RelativeTo = None,
                departure: str | None = None,
                mean_grid: Grid | None = None) -> Correlation:
    """Two fields' correlation by bin of horizontal wavenumber and by depth,
    by the rules of the module docstring: `fields` of `model` against
    `other_fields` (the same by default) of `other`, which may be
    `model` itself, at each of `depths`, both on one grid, `grid` or the
    first model's default, with the same choices."""
    ds = np.atleast_1d(np.asarray(depths, dtype=float))
    grid = _grid_of(model, grid)
    other_fields = fields if other_fields is None else other_fields
    kw = dict(grid=grid, side=side, at=at, component=component, frame=frame,
              relative_to=relative_to, departure=departure, mean_grid=mean_grid)
    first = tuple(spectrum(model, fields, depth=float(d), **kw) for d in ds)
    second = tuple(spectrum(other, other_fields, depth=float(d), **kw) for d in ds)
    bins, step = wavenumber_bins(grid)
    size = int(bins.max()) + 1
    by_degree, total = [], []
    for a, b in zip(first, second):
        ca, cb = a.coefficients, b.coefficients
        cross = np.bincount(bins, weights=ca * cb, minlength=size)
        pa = np.bincount(bins, weights=ca * ca, minlength=size)
        pb = np.bincount(bins, weights=cb * cb, minlength=size)
        heard = (pa > NO_POWER ** 2 * np.sum(pa)) & (pb > NO_POWER ** 2 * np.sum(pb))
        with np.errstate(divide="ignore", invalid="ignore"):
            by_degree.append(np.where(heard, cross / np.sqrt(pa * pb), np.nan))
        quiet = (np.sum(pa[1:]) <= NO_POWER ** 2 * np.sum(pa)
                 or np.sum(pb[1:]) <= NO_POWER ** 2 * np.sum(pb))
        whole = np.sqrt(np.sum(pa[1:]) * np.sum(pb[1:]))
        total.append(np.nan if quiet else np.sum(cross[1:]) / whole)
    degrees = np.arange(size)
    return Correlation(ds, degrees, degrees * step, np.array(by_degree),
                       np.array(total), first, second)
