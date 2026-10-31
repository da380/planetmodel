"""Slices: a model's field on the points one looks at it by.

A slice is a field evaluated on a set of points chosen for looking at,
as plain numbers, for any model: a sphere at one radius (`sphere`), a
great-circle section, distance along the arc by radius (`section`, to
an `end=` or at an `azimuth=` for a `length=`; a whole circle is the
plane through the centre), a profile at a point,
radius alone (`profile`), and the spherical mean as a profile
(`mean_profile`).  Drawing them is `planetmodel.plotting`'s business;
nothing here draws or needs more than numpy and scipy.

**What is sliced.**  A field by name, from each layer holding it, NaN on
a layer that lacks it, or a mapping from layer indices to fields, for a
quantity the model does not hold by name.  A scalar is sliced as it is;
a vector or tensor by `component=` (an index into its stored components,
flattened in C order: the Voigt slot for a Voigt form) and `frame=`
together, one without the other refused, or `component="norm"` for a
vector's length.  `relative_to=` makes the slice a departure: from the
field's spherical mean at each point's reference radius (`"mean"`, by
`means.spherical_mean`, consistent with the field's own rules), from a
perturbed field's own reference (`"reference"`), or from the same field
of another model (a `Model`), as `departure="percent"` (100 (v / ref -
1), the default), `"difference"` (v - ref) or `"log10"` (log10(v /
ref)).  Every slice records what it shows (`Shown`) and labels itself.

**Reference or physical.**  A slice is taken at reference points, where
the fields live (`at="reference"`), and gives besides where the mapping
takes each of them, so that it can be drawn as the body is.  At physical
points (`at="physical"`: a true sphere |x| = r, a true plane, a true
radius) each point is pulled back through the mapping's inverse and the
field evaluated there, which is offered where the mapping inverts
itself and refused by name where it does not; a physical section or
profile also needs the mapping to keep each ray, as a radial stretch
does, so that each layer's part of a ray is one run of radii.  Points
are given as the library gives them, colatitude and longitude in
radians; `frames.from_geographic` converts latitudes and longitudes.

**Spectra.**  `spectrum` analyses a sphere slice into the library's
real orthonormal harmonics (`planetmodel.harmonics`: the integral of
Y_lm^2 over the unit sphere is one, with no Condon-Shortley phase),
and gives the power by degree

    P_l = sum_m c_lm^2 / (4 pi),

so that the sum over degrees is the field's mean square over the sphere.
The slice is taken on `gauss_legendre(2 lmax)` unless a grid is given,
so that a field not limited to the band does not alias into it.
`correlation` gives, at each of a list of radii, two fields' correlation
by degree, r_l = sum_m a_lm b_lm / sqrt(sum_m a_lm^2 sum_m b_lm^2), and
over the degrees from 1 to `lmax`, NaN where a degree has no power in
either (its amplitude below `NO_POWER` of the field's); the fields may be
two models' or two of one model's.

**Kernels and other signed fields.**  `integrated` maps a field
integrated along each radius over a span of reference radii, the
integral of f r^2 dr (`weight="volume"`, the share of each direction in
the field's volume integral: a sensitivity kernel's column) or of f dr
(`weight="radius"`); within a layer it is Gauss-Legendre on each of the
field's radial pieces, exact for a piecewise polynomial of their degree
with the weight, or on the whole layer for a field that declares none,
and a layer lacking the field adds nothing.  `mean_profile(...,
total=True)` gives r^2 times the field's integral over the sphere, the
one-dimensional kernel K_1D whose integral with dm dr is that of K dm dV
for a radial dm.  `surface` is a sphere slice of a function of the
angles on one of the model's interfaces (a field on interfaces from a
model file, a shift of a boundary), on its own grid where it has one;
`spectrum_of` analyses any sphere slice, a surface's or an integral's
among them.  `normalised` divides a slice by its largest absolute
value, its label saying so.

**Cutaways.**  `cutaway` is the body seen from outside with a wedge
cut from it, the field on the faces of the cut and on what remains of
the surface: the classic picture of a planet's interior.  The wedge is
a span of longitudes at the radii above `inner`, so that its faces are
two meridional half-planes and, where `inner` is a radius within the
body, the sphere of that radius is left standing in the cut.  The
picture is an orthographic view toward a point given by its colatitude
and longitude, north up, rendered pixel by pixel: each pixel's ray is
cast through the cut body and the field is evaluated where it first
lands, on the surface, a face or the inner sphere, so that what hides
what is settled exactly.  The slice keeps which face each pixel sees
(`CutawaySlice.face`, by `CUTAWAY_FACES`), the point and the value,
and the fraction of each pixel the body covers, for the edge of the
picture; `visible` and `project` put anything else, a coastline say,
into the same view.  A cutaway is taken at reference points.

**Interfaces.**  A section or a profile keeps both sides of an
interface, each layer's numbers including its own ends.  A sphere slice
is asked for one side, `side="below"` or `"above"`, required where its
radius lies on an interface to the geometry's tolerance and of no effect
elsewhere, so that a loop over radii names it once; a physical sphere's
points that fall on an interface take the side the same way, and its
points outside the body are NaN.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .fields import Field, check_frame, stored_shape
from .frames import cartesian_points, spherical_coordinates, wrapped
from .means import spherical_mean
from .model import Model
from .units import Dimensions, Scales, scaled_unit

if TYPE_CHECKING:
    from .geometry import InterfaceInfo
    from .sampling import AngularGrid

__all__ = ["AT", "SIDES", "DEPARTURES", "WEIGHTS", "NO_POWER", "Fields", "RelativeTo",
           "Slice", "Shown", "SliceLayer", "SphereSlice", "SectionSlice",
           "ProfileSlice", "CutawaySlice", "CUTAWAY_FACES", "Spectrum",
           "Correlation", "sphere", "section", "profile", "mean_profile",
           "integrated", "surface", "cutaway", "normalised", "evaluate",
           "spectrum", "spectrum_of", "correlation"]

#: Where a slice's points are: in the reference body, or the physical one.
AT = ("reference", "physical")

#: The sides of an interface a sphere slice may be taken on.
SIDES = ("below", "above")

#: How a departure is expressed.
DEPARTURES = ("percent", "difference", "log10")

#: The weights of an integral along the radius: r^2 dr and dr.
WEIGHTS = ("volume", "radius")

#: The amplitude of a degree, relative to the whole field's, below which a
#: correlation takes the degree to have no power: what the analysis gives
#: a field with nothing at that degree.
NO_POWER = 1e-10

#: What is sliced: a field by name, or fields by layer index.
type Fields = str | Mapping[int, Field]

#: What a departure is from: the mean, the reference, or another model.
type RelativeTo = str | Model | None

#: A layer's values at points, from (layer, r, theta, phi).
type _Values = Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]

#: One field's numbers at points, from (r, theta, phi).
type _PointValues = Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]


# -- what a slice shows -----------------------------------------------------------

@dataclass(frozen=True)
class Shown:
    """What a slice's values are: the field, the component, the departure.

    `name` is the field's name (None for fields given by layer);
    `component` an index into the stored components or `"norm"`, and
    `frame` the frame of a component; `relative_to` what a departure is
    from (`"mean"`, `"reference"`, or the other model's name) and
    `departure` how it is expressed; `statistic` `"spherical mean"` for a
    mean profile; `dimensions` and `scales` the field's and the model's,
    so that the label states the unit; `normalised` that the values were
    divided by their largest absolute value, which leaves them no unit.
    """

    name: str | None
    _: KW_ONLY
    component: int | str | None = None
    frame: str | None = None
    relative_to: str | None = None
    departure: str | None = None
    statistic: str | None = None
    dimensions: Dimensions | None = None
    scales: Scales = Scales.SI
    normalised: bool = False

    @property
    def label(self) -> str:
        """A label for an axis or a colour bar: "vs, percent from the mean
        [%]", "rho [kg m-3]", "|u| [m]", "K, normalised"."""
        base = self.name or "field"
        if self.component == "norm":
            base = f"|{base}|"
        elif self.component is not None:
            base = f"{base}[{self.component}] ({self.frame})"
        if self.statistic:
            base = f"{base}, {self.statistic}"
        unit = None if self.dimensions is None else scaled_unit(self.dimensions,
                                                                 self.scales)
        if self.relative_to is not None:
            ref = self.relative_to
            if ref in ("mean", "reference"):
                ref = f"the {ref}"
            if self.departure == "percent":
                base, unit = f"{base}, percent from {ref}", "%"
            elif self.departure == "log10":
                base, unit = f"{base}, log10 of the ratio to {ref}", None
            else:
                base = f"{base}, difference from {ref}"
        if self.normalised:
            return f"{base}, normalised"
        return base if unit is None else f"{base} [{unit}]"

    @property
    def is_departure(self) -> bool:
        """Whether the values are a departure, centred on zero."""
        return self.relative_to is not None


# -- the slices -----------------------------------------------------------------

@dataclass(frozen=True)
class SphereSlice:
    """A field on a sphere: `values` (ntheta, nphi) on `grid` at `radius`.

    `radius` is a reference radius, or a physical one when `at` is
    `"physical"`; `reference` and `physical` are the points' Cartesian
    positions (ntheta, nphi, 3) in the two bodies; `side` the side of an
    interface the slice was taken on, where it lies on one.
    """

    radius: float
    grid: AngularGrid
    values: np.ndarray
    reference: np.ndarray
    physical: np.ndarray
    shown: Shown
    _: KW_ONLY
    at: str = "reference"
    side: str | None = None

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label


@dataclass(frozen=True)
class SliceLayer:
    """One layer's part of a section or a profile.

    `radius` holds the points' radii, reference or physical as the
    slice's `at`, and `values` the field there, both of the points'
    shape: (n, m), n radii by m points along the arc, for a section, and
    (n,) for a profile; `reference` and `physical` are the points'
    Cartesian positions, that shape followed by 3.  Each layer's points
    include its own ends, so that an interface carries both sides.
    """

    index: int
    radius: np.ndarray
    values: np.ndarray
    reference: np.ndarray
    physical: np.ndarray


@dataclass(frozen=True)
class SectionSlice:
    """A field on a great-circle section, layer by layer.

    `distance` (m,) is the angle along the arc from its start, in
    radians; `path` (m, 3) the unit vectors of the arc's points; `plane`
    (2, 3) the start and the direction of travel there, an orthonormal
    basis of the section's plane, for drawing positions in it; `layers`
    each layer's points and values.
    """

    distance: np.ndarray
    path: np.ndarray
    plane: np.ndarray
    layers: tuple[SliceLayer, ...]
    shown: Shown
    _: KW_ONLY
    at: str = "reference"

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label

    @property
    def length(self) -> float:
        """The arc's length, in radians."""
        return float(self.distance[-1])


@dataclass(frozen=True)
class ProfileSlice:
    """A field along a radius at (theta, phi), layer by layer; for a mean
    profile `theta` and `phi` are None and the values are the spherical
    means."""

    theta: float | None
    phi: float | None
    layers: tuple[SliceLayer, ...]
    shown: Shown
    _: KW_ONLY
    at: str = "reference"

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label


#: What a pixel of a cutaway looks at, by its code in `CutawaySlice.face`:
#: nothing, the surface, the meridional faces at the cut's first and second
#: longitudes, the faces at its first and second colatitudes (the equatorial
#: plane for a cut stopping at the equator, a cone otherwise), and the inner
#: sphere left standing in the cut.
CUTAWAY_FACES = ("none", "surface", "first", "second", "upper", "lower", "inner")

#: Subsamples across each side of a pixel when a cutaway's coverage is
#: measured.
_COVERAGE_SUBSAMPLES = 4

#: How far, relative to the outer radius, a point may lie in front of the
#: first thing its ray meets and still count as seen.
_VISIBLE_RTOL = 1e-9


@dataclass(frozen=True)
class CutawaySlice:
    """A field on the faces of the body with a wedge cut from it, as the
    pixels of an orthographic view.

    The wedge is the longitudes from `longitudes[0]` east to
    `longitudes[1]` and the colatitudes between `colatitudes[0]` and
    `colatitudes[1]`, in radians, at radii between `inner` and `outer`,
    the body's outer boundary.  `view` (3, 3) holds, as rows, the unit
    vectors of the picture's right, its up and the direction toward the
    viewer, in reference Cartesian coordinates; the picture spans
    [-extent, extent] in both directions, `n` pixels across, rows from
    the top.  `face` (n, n) says what each pixel looks at by its index
    into `CUTAWAY_FACES`; `reference` and `physical` (n, n, 3) the point
    it looks at in the two bodies, NaN where it looks at nothing;
    `values` (n, n) the field there, NaN likewise; `coverage` (n, n) the
    fraction of each pixel the body covers, 1 inside the picture of the
    body and 0 outside; `boundaries` the radii of the model's interfaces,
    for drawing them on the faces.  `side` is the side of an interface
    the inner sphere was taken on, where it lies on one.
    """

    longitudes: tuple[float, float]
    colatitudes: tuple[float, float]
    inner: float
    outer: float
    boundaries: np.ndarray
    view: np.ndarray
    extent: float
    face: np.ndarray
    reference: np.ndarray
    physical: np.ndarray
    values: np.ndarray
    coverage: np.ndarray
    shown: Shown
    _: KW_ONLY
    at: str = "reference"
    side: str | None = None

    @property
    def label(self) -> str:
        """The label of what the slice shows."""
        return self.shown.label

    @property
    def n(self) -> int:
        """The pixels across the picture."""
        return int(self.face.shape[0])

    def mask(self, name: str) -> np.ndarray:
        """The pixels looking at the face `name`, one of `CUTAWAY_FACES`."""
        if name not in CUTAWAY_FACES:
            raise ValueError(f"name must be one of {CUTAWAY_FACES}, got {name!r}")
        return self.face == CUTAWAY_FACES.index(name)

    def project(self, points: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
        """The positions (x, y) in the picture of reference Cartesian
        points (..., 3): their components along its right and its up."""
        X = np.asarray(points, dtype=float)
        return X @ self.view[0], X @ self.view[1]

    def visible(self, points: ArrayLike) -> np.ndarray:
        """Whether each reference Cartesian point (..., 3) is seen: it is
        the first thing its ray meets, within `_VISIBLE_RTOL` of the
        outer radius, so a point of the surface within the wedge, or
        behind the body, or inside it, is not."""
        X = np.asarray(points, dtype=float)
        depth = X @ self.view[2]
        t, _, _ = _cutaway_hits(X, self.view, self.outer, self.inner, self.longitudes,
                                self.colatitudes)
        return np.isfinite(t) & (np.abs(t - depth) <= _VISIBLE_RTOL * self.outer)


#: A slice of any form, as `normalised` takes one.
type Slice = SphereSlice | SectionSlice | ProfileSlice | CutawaySlice


@dataclass(frozen=True)
class Spectrum:
    """A field's power by degree on a sphere.

    `degrees` 0 .. lmax; `power` P_l, summing to the mean square over the
    sphere; `coefficients` the real orthonormal ones, (2, lmax + 1, lmax +
    1) as `planetmodel.harmonics` lays them out; `radius`, `side` and
    `at` as the sphere slice they were analysed from.
    """

    radius: float
    degrees: np.ndarray
    power: np.ndarray
    coefficients: np.ndarray
    shown: Shown
    _: KW_ONLY
    at: str = "reference"
    side: str | None = None

    @property
    def amplitude(self) -> np.ndarray:
        """The square root of the power, by degree."""
        return np.sqrt(self.power)

    @property
    def label(self) -> str:
        """The label of what the spectrum is of."""
        return self.shown.label


@dataclass(frozen=True)
class Correlation:
    """Two fields' correlation by degree and radius.

    `radii` (k,) and `degrees` (lmax + 1,); `by_degree` (k, lmax + 1),
    r_l at each radius; `total` (k,), over the degrees from 1 to lmax;
    `first` and `second` the two fields' spectra at each radius.
    """

    radii: np.ndarray
    degrees: np.ndarray
    by_degree: np.ndarray
    total: np.ndarray
    first: tuple[Spectrum, ...]
    second: tuple[Spectrum, ...]

    @property
    def label(self) -> str:
        """What is correlated with what, by name: "correlation of vs", or
        "correlation of vs with rho"."""
        a = self.first[0].shown.name or "field"
        b = self.second[0].shown.name or "field"
        return f"correlation of {a}" if a == b else f"correlation of {a} with {b}"


# -- what is evaluated --------------------------------------------------------------

def _field_of(model: Model, fields: Fields, layer: int) -> Field | None:
    """The field of a layer, or None where it has none."""
    if isinstance(fields, str):
        held = model.layer(layer)
        return held[fields] if fields in held else None
    return fields.get(layer)


def _real(values: np.ndarray, field: Field) -> np.ndarray:
    v = np.asarray(values)
    if np.iscomplexobj(v):
        raise TypeError(f"{field!r} is complex; slice its real or imaginary part as "
                        "a field of its own")
    return v.astype(float, copy=False)


def _components(field: Field, layer: int, component: int | str | None,
                frame: str | None) -> _PointValues:
    """A layer's field as a function of points giving one number each."""
    rank = field.character.rank
    if rank == 0:
        if component is not None or frame is not None:
            raise ValueError(f"layer {layer}'s {field.name or 'field'} is a scalar; "
                             "it has no component or frame")
        return lambda r, t, p: _real(field.evaluate(r, t, p), field)
    if component is None:
        raise ValueError(f"layer {layer}'s {field.name or 'field'} is of rank {rank}: "
                         "name component= (an index, with frame=) or 'norm'")
    if component == "norm":
        if rank != 1:
            raise ValueError("component='norm' is a vector's length; this field is "
                             f"of rank {rank}")
        return lambda r, t, p: np.linalg.norm(_real(field.evaluate(r, t, p), field),
                                              axis=-1)
    if frame is None:
        raise ValueError("a component is of a frame: name frame='spherical' or "
                         "'cartesian' with component=")
    check_frame(frame)
    size = int(np.prod(stored_shape(field.character)))
    if not (isinstance(component, (int, np.integer)) and 0 <= component < size):
        raise ValueError(f"component must be an index in 0..{size - 1} or 'norm', "
                         f"got {component!r}")
    shape = stored_shape(field.character)

    def one(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        v = _real(field.evaluate(r, t, p, frame=frame), field)
        return v.reshape(v.shape[:v.ndim - len(shape)] + (size,))[..., component]
    return one


def _other_layer(model: Model, r: np.ndarray, toward: float) -> np.ndarray:
    """The layer of another model holding each radius, nudged toward
    `toward`, a radius inside the slice's own layer, so that a radius on
    an interface of the other model is read on the side of the slice's
    layer."""
    b = model.skeleton.boundaries
    tol = model.geometry.rtol * model.skeleton.span
    nudged = r + 2.0 * tol * np.sign(toward - r)
    return np.clip(np.searchsorted(b, nudged, side="right") - 1, 0, b.size - 2)


def _evaluator(model: Model, fields: Fields, *, component: int | str | None,
               frame: str | None, relative_to: RelativeTo, departure: str | None,
               mean_grid: AngularGrid | None) -> tuple[_Values, Shown]:
    """The values of a layer at points, with the departure asked for, and
    what they show."""
    if departure is not None and departure not in DEPARTURES:
        raise ValueError(f"departure must be one of {DEPARTURES}, got {departure!r}")
    if relative_to is None and departure is not None:
        raise ValueError("departure= says how a departure is expressed; name what "
                         "it is from with relative_to=")
    if relative_to is not None and departure is None:
        departure = "percent"
    if isinstance(relative_to, str) and relative_to not in ("mean", "reference"):
        raise ValueError(f"relative_to must be 'mean', 'reference' or a Model, got "
                         f"{relative_to!r}")
    if isinstance(relative_to, Model) and not isinstance(fields, str):
        raise ValueError("a departure from another model is of a field by name")
    name = fields if isinstance(fields, str) else None
    if name is not None and not model.layers_with(name):
        raise KeyError(f"no layer holds a field {name!r}")
    spec = None if name is None else model.spec(name)
    shown = Shown(name, component=component, frame=frame,
                  relative_to=(relative_to if relative_to is None
                               or isinstance(relative_to, str)
                               else relative_to.name or "the other model"),
                  departure=departure,
                  dimensions=None if spec is None else spec.dimensions,
                  scales=model.scales)
    own: dict[int, _PointValues | None] = {}
    against: dict[int, _PointValues] = {}

    def value(layer: int) -> _PointValues | None:
        if layer not in own:
            field = _field_of(model, fields, layer)
            own[layer] = (None if field is None
                          else _components(field, layer, component, frame))
        return own[layer]

    def reference_of(layer: int) -> _PointValues:
        """The values a departure on a layer is from."""
        if layer in against:
            return against[layer]
        field = _field_of(model, fields, layer)
        if relative_to in ("mean", "reference") and (
                field.character.rank or component is not None):
            raise ValueError(f"a departure from the {relative_to} is of a field of "
                             f"rank 0; layer {layer}'s is of rank "
                             f"{field.character.rank}")
        if relative_to == "mean":
            if (mean_grid is None and getattr(field, "spherical_mean", None) is None
                    and not getattr(field, "is_radial", False)):
                raise ValueError(f"layer {layer}'s {field!r} has no spherical mean of "
                                 "its own: pass mean_grid=, an angular grid with "
                                 "weights to take it by quadrature")
            mean = spherical_mean(field, grid=mean_grid)
            fn = lambda r, t, p: _real(mean.evaluate(r, None, None), mean)
        elif relative_to == "reference":
            ref = getattr(field, "reference", None)
            if ref is None and getattr(field, "is_radial", False):
                ref = field
            if ref is None:
                raise ValueError(f"layer {layer}'s {field!r} is not perturbed, so it "
                                 "has no reference to depart from")
            fn = lambda r, t, p: _real(ref.evaluate(r, None, None), ref)
        else:
            other = relative_to
            lo, hi = model.skeleton.interval(layer)
            mid = 0.5 * (lo + hi)

            def fn(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
                r, t, p = np.broadcast_arrays(r, t, p)
                out = np.full(r.shape, np.nan)
                which = _other_layer(other, r, mid)
                for j in np.unique(which):
                    held = other.layer(int(j))
                    if name not in held:
                        continue
                    k = which == j
                    f = _components(held[name], int(j), component, frame)
                    out[k] = f(r[k], t[k], p[k])
                return out
        against[layer] = fn
        return fn

    def values(layer: int, r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        r, t, p = np.broadcast_arrays(r, t, p)
        f = value(layer)
        if f is None:
            return np.full(r.shape, np.nan)
        v = np.broadcast_to(f(r, t, p), r.shape)
        if relative_to is None:
            return np.array(v, dtype=float)
        ref = np.broadcast_to(reference_of(layer)(r, t, p), r.shape)
        with np.errstate(divide="ignore", invalid="ignore"):
            if departure == "percent":
                return 100.0 * (v / ref - 1.0)
            if departure == "log10":
                return np.log10(v / ref)
            return v - ref
    return values, shown


# -- where the points are -----------------------------------------------------------

def _check(at: str, side: str | None) -> None:
    if at not in AT:
        raise ValueError(f"at must be one of {AT}, got {at!r}")
    if side is not None and side not in SIDES:
        raise ValueError(f"side must be one of {SIDES}, got {side!r}")


def _pulled_back(model: Model, x: np.ndarray) -> np.ndarray:
    """Reference points X with m(X) = x, or the refusal of a mapping
    without an inverse."""
    mapping = model.geometry.mapping
    if model.geometry.is_identity:
        return np.array(x, dtype=float)
    try:
        return np.asarray(mapping.inverse(x), dtype=float)
    except NotImplementedError:
        raise ValueError(
            f"the mapping ({type(mapping).__name__}) does not invert itself, so the "
            "model cannot be sliced at physical points; slice it at reference ones "
            "(at='reference'), which give their physical positions too") from None


def _forward(model: Model, X: np.ndarray) -> np.ndarray:
    """m(X), the physical positions of reference points."""
    if model.geometry.is_identity:
        return np.array(X, dtype=float)
    return np.asarray(model.geometry.mapping(X), dtype=float)


def _interface_name(model: Model, boundary: int) -> str:
    """The interface at a boundary, as an error names it."""
    radius = float(model.skeleton.boundaries[boundary])
    face = min(model.geometry.interfaces, key=lambda f: abs(f.radius - radius))
    return f"interface {face.name!r}" if face.name else f"interface {face.index}"


def _sided(model: Model, boundary: int, side: str | None, radius: float) -> int:
    """The layer a side of an interior boundary names, or the refusal."""
    if side is None:
        raise ValueError(
            f"radius {radius} lies on {_interface_name(model, boundary)}: name the "
            "side to slice, side='below' or 'above'")
    return boundary - 1 if side == "below" else boundary


def _layers_of(model: Model, s: np.ndarray, side: str | None
               ) -> tuple[np.ndarray, bool]:
    """The layer holding each reference radius, the side taken where it
    lies on an interior boundary and -1 outside the body, and whether a
    side was taken."""
    b = model.skeleton.boundaries
    tol = model.geometry.rtol * model.skeleton.span
    out = np.clip(np.searchsorted(b, s, side="right") - 1, 0, b.size - 2)
    nearest = np.argmin(np.abs(s[..., None] - b), axis=-1)
    on = np.abs(b[nearest] - s) <= tol
    interior = on & (nearest > 0) & (nearest < b.size - 1)
    if np.any(interior):
        k = tuple(np.argwhere(interior)[0])
        _sided(model, int(nearest[k]), side, float(s[k]))
        out = np.where(interior, nearest - 1 if side == "below" else nearest, out)
    out = np.where(on & (nearest == 0), 0, out)
    out = np.where(on & (nearest == b.size - 1), b.size - 2, out)
    outside = (s < b[0] - tol) | (s > b[-1] + tol)
    return np.where(outside, -1, out), bool(np.any(interior))


def _evaluate_layers(values: _Values, layers: np.ndarray, X: np.ndarray) -> np.ndarray:
    """The values at reference points X, each from its layer, NaN at -1."""
    s, t, p, _ = spherical_coordinates(X)
    out = np.full(s.shape, np.nan)
    for layer in np.unique(layers):
        if layer < 0:
            continue
        k = layers == layer
        out[k] = values(int(layer), s[k], t[k], p[k])
    return out


# -- the slices ---------------------------------------------------------------------

def sphere(model: Model, fields: Fields, *, radius: float | None = None,
           depth: float | None = None, grid: AngularGrid | None = None,
           side: str | None = None, at: str = "reference",
           component: int | str | None = None, frame: str | None = None,
           relative_to: RelativeTo = None, departure: str | None = None,
           mean_grid: AngularGrid | None = None) -> SphereSlice:
    """A field on a sphere, by the rules of the module docstring.

    Exactly one of `radius` and `depth`, the depth below the reference
    body's outer boundary; `grid` the angular grid, equiangular at one
    degree by default (180 x 360 midpoints); `side` the side of an
    interface the radius lies on; `at` reference or physical points; the
    component, frame and departure as the module docstring says, the
    mean of a field without one of its own taken on `mean_grid` or, where
    it has weights, on `grid`.
    """
    from .sampling import equiangular
    _check(at, side)
    if (radius is None) == (depth is None):
        raise ValueError("give exactly one of radius= and depth=")
    outer = float(model.skeleton.boundaries[-1])
    r = float(outer - depth if radius is None else radius)
    grid = equiangular(180, 360) if grid is None else grid
    if mean_grid is None and grid.weights is not None:
        mean_grid = grid
    values, shown = _evaluator(model, fields, component=component, frame=frame,
                               relative_to=relative_to, departure=departure,
                               mean_grid=mean_grid)
    theta = grid.colatitudes[:, None]
    phi = grid.longitudes[None, :]
    points = cartesian_points(r, theta, phi)
    taken = None
    if at == "reference":
        location = model.skeleton.locate(r, rtol=model.geometry.rtol)
        if len(location.layers) == 1:
            layer = location.layers[0]
        else:
            layer = _sided(model, int(location.boundary), side, r)
            taken = side
        th, ph = np.broadcast_arrays(theta, phi)
        v = values(layer, np.full(th.shape, r), th, ph)
        reference, physical = points, _forward(model, points)
    else:
        physical = points
        reference = _pulled_back(model, points)
        s = np.linalg.norm(reference, axis=-1)
        layers, sided = _layers_of(model, s, side)
        if np.all(layers < 0):
            raise ValueError(f"the physical sphere of radius {r} lies outside the body")
        v = _evaluate_layers(values, layers, reference)
        taken = side if sided else None
    return SphereSlice(r, grid, v, reference, physical, shown, at=at, side=taken)


def _arc(start: tuple[float, float], end: tuple[float, float] | None,
         azimuth: float | None, length: float | None, m: int
         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """The distance along the arc, its points' unit vectors and the plane's
    basis (the start and the direction of travel there)."""
    a = cartesian_points(1.0, start[0], start[1])
    if (end is None) == (azimuth is None):
        raise ValueError("give the arc's end, or its azimuth= and length=")
    if end is not None:
        if length is not None:
            raise ValueError("the arc's length follows from its end; give one")
        b = cartesian_points(1.0, end[0], end[1])
        along = b - np.dot(a, b) * a
        span = float(np.arctan2(np.linalg.norm(along), np.dot(a, b)))
        if np.linalg.norm(along) <= 1e-12:
            raise ValueError("the arc's ends are the same point or antipodal, so no "
                             "great circle is defined between them; give azimuth= "
                             "and length=")
        t = along / np.linalg.norm(along)
    else:
        if length is None or not 0.0 < float(length) <= 2.0 * np.pi:
            raise ValueError("an arc of a given azimuth needs length= in (0, 2 pi]")
        span = float(length)
        theta, phi = start
        north = -np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi),
                           -np.sin(theta)])
        east = np.array([-np.sin(phi), np.cos(phi), 0.0])
        t = np.cos(azimuth) * north + np.sin(azimuth) * east
        if np.linalg.norm(t) < 0.5:
            raise ValueError("at a pole the azimuth has no north to count from; start "
                             "the arc elsewhere or give its end")
        t = t / np.linalg.norm(t)
    if int(m) < 2:
        raise ValueError(f"m must be at least 2, got {m}")
    distance = np.linspace(0.0, span, int(m))
    path = (np.cos(distance)[:, None] * a[None, :]
            + np.sin(distance)[:, None] * t[None, :])
    return distance, path, np.stack([a, t])


def _keeps_rays(model: Model, X: np.ndarray) -> None:
    """Refuse a physical section or profile of a mapping that moves points
    off their rays."""
    x = _forward(model, X)
    rx = np.linalg.norm(x, axis=-1, keepdims=True)
    rX = np.linalg.norm(X, axis=-1, keepdims=True)
    ok = (rX[..., 0] == 0.0) | (np.linalg.norm(x / np.where(rx > 0, rx, 1.0)
                                               - X / np.where(rX > 0, rX, 1.0),
                                               axis=-1) <= 1e-9)
    if not np.all(ok):
        raise ValueError("the mapping moves points off their rays, so a physical "
                         "section or profile has no run of radii per layer; slice it "
                         "at reference points (at='reference')")


def _runs(model: Model, values: _Values, path: np.ndarray,
          radii: tuple[float, float] | None, n: int, at: str
          ) -> tuple[SliceLayer, ...]:
    """Each layer's points and values along the rays of `path` (m, 3)."""
    if int(n) < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    b = model.skeleton.boundaries
    if at == "physical" and model.geometry.is_identity:
        at = "reference"
    lo, hi = (float(b[0]), float(b[-1])) if radii is None else map(float, radii)
    if not lo < hi:
        raise ValueError(f"radii must be (lo, hi) with lo < hi, got {radii}")
    t, p = spherical_coordinates(path)[1:3]
    out = []
    for layer in range(model.nlayers):
        b0, b1 = float(b[layer]), float(b[layer + 1])
        if at == "reference":
            r0, r1 = max(lo, b0), min(hi, b1)
            if not r0 < r1:
                continue
            r = np.linspace(r0, r1, int(n))[:, None] * np.ones((1, path.shape[0]))
            X = r[..., None] * path[None, :, :]
            v = values(layer, r, np.broadcast_to(t, r.shape),
                       np.broadcast_to(p, r.shape))
            out.append(SliceLayer(layer, r, v, X, _forward(model, X)))
            continue
        ends = np.stack([b0 * path, b1 * path])
        _keeps_rays(model, ends)
        bounds = np.linalg.norm(_forward(model, ends), axis=-1)
        r0, r1 = np.maximum(bounds[0], lo), np.minimum(bounds[1], hi)
        if not np.any(r0 < r1):
            continue
        frac = np.linspace(0.0, 1.0, int(n))[:, None]
        rho = r0[None, :] + frac * np.maximum(r1 - r0, 0.0)[None, :]
        x = rho[..., None] * path[None, :, :]
        X = _pulled_back(model, x)
        s, tt, pp, _ = spherical_coordinates(X)
        v = values(layer, np.clip(s, b0, b1), tt, pp)
        v = np.where((r1 > r0)[None, :], v, np.nan)
        out.append(SliceLayer(layer, rho, v, X, x))
    return tuple(out)


def section(model: Model, fields: Fields, start: tuple[float, float], *,
            end: tuple[float, float] | None = None, azimuth: float | None = None,
            length: float | None = None, radii: tuple[float, float] | None = None,
            n: int = 50, m: int = 181, at: str = "reference",
            component: int | str | None = None, frame: str | None = None,
            relative_to: RelativeTo = None, departure: str | None = None,
            mean_grid: AngularGrid | None = None) -> SectionSlice:
    """A field on a great-circle section, by the rules of the module
    docstring.

    The arc runs from `start` (theta, phi) along the minor arc to `end`,
    or from `start` at `azimuth` (from north toward east, in radians) for
    `length` radians, up to a whole circle, the one or the other given;
    `radii` (lo, hi) bounds it, the whole body by default; `n` points per
    layer, its ends included, and `m` along the arc; `at`, the component,
    frame and departure as the module docstring says.
    """
    _check(at, None)
    values, shown = _evaluator(model, fields, component=component, frame=frame,
                               relative_to=relative_to, departure=departure,
                               mean_grid=mean_grid)
    distance, path, plane = _arc(start, end, azimuth, length, m)
    layers = _runs(model, values, path, radii, n, at)
    return SectionSlice(distance, path, plane, layers, shown, at=at)


def profile(model: Model, fields: Fields, theta: float, phi: float, *, n: int = 100,
            radii: tuple[float, float] | None = None, at: str = "reference",
            component: int | str | None = None, frame: str | None = None,
            relative_to: RelativeTo = None, departure: str | None = None,
            mean_grid: AngularGrid | None = None) -> ProfileSlice:
    """A field along the radius at (theta, phi), by the rules of the
    module docstring: `n` points per layer, its ends included, within
    `radii` (lo, hi), the whole body by default."""
    _check(at, None)
    values, shown = _evaluator(model, fields, component=component, frame=frame,
                               relative_to=relative_to, departure=departure,
                               mean_grid=mean_grid)
    path = cartesian_points(1.0, theta, phi)[None, :]
    layers = tuple(SliceLayer(lay.index, lay.radius[:, 0], lay.values[:, 0],
                              lay.reference[:, 0], lay.physical[:, 0])
                   for lay in _runs(model, values, path, radii, n, at))
    return ProfileSlice(float(theta), float(phi), layers, shown, at=at)


def mean_profile(model: Model, fields: Fields, *, n: int = 100,
                 mean_grid: AngularGrid | None = None,
                 total: bool = False) -> ProfileSlice:
    """The spherical mean of a scalar field along the radius, `n` points
    per layer, its ends included; a field without a mean of its own takes
    it on `mean_grid`.  With `total`, r^2 times the integral over the
    sphere, 4 pi r^2 times the mean: the one-dimensional kernel of the
    module docstring."""
    if int(n) < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    name = fields if isinstance(fields, str) else None
    if name is not None and not model.layers_with(name):
        raise KeyError(f"no layer holds a field {name!r}")
    spec = None if name is None else model.spec(name)
    dims = None if spec is None or spec.dimensions is None else spec.dimensions
    if total and dims is not None:
        dims = dims * Dimensions(length=2)
    shown = Shown(name, statistic=("r^2 times the integral over the sphere" if total
                                   else "spherical mean"),
                  dimensions=dims, scales=model.scales)
    out = []
    for layer in range(model.nlayers):
        field = _field_of(model, fields, layer)
        lo, hi = model.skeleton.interval(layer)
        r = np.linspace(lo, hi, int(n))
        X = r[:, None] * np.array([0.0, 0.0, 1.0])[None, :]
        if field is None:
            v = np.full(r.shape, np.nan)
        else:
            if field.character.rank:
                raise ValueError(f"layer {layer}'s field is of rank "
                                 f"{field.character.rank}; a spherical mean is of a "
                                 "scalar")
            if (mean_grid is None and getattr(field, "spherical_mean", None) is None
                    and not getattr(field, "is_radial", False)):
                raise ValueError(f"layer {layer}'s {field!r} has no spherical mean of "
                                 "its own: pass mean_grid=")
            mean = spherical_mean(field, grid=mean_grid)
            v = _real(mean.evaluate(r, None, None), mean)
            if total:
                v = 4.0 * np.pi * r ** 2 * v
        out.append(SliceLayer(layer, r, v, X, X))
    return ProfileSlice(None, None, tuple(out), shown)


def _pieces(field: Field, lo: float, hi: float, n: int
            ) -> list[tuple[np.ndarray, np.ndarray]]:
    """Gauss-Legendre radii and weights on [lo, hi] within a field's
    layer: on each of its radial pieces, with points enough to be exact
    for their degree times r^2, or `n` points on the whole span."""
    pieces = getattr(field, "radial_pieces", lambda: None)()
    if pieces is None:
        breaks, count = np.array([lo, hi]), int(n)
    else:
        b, degree = pieces
        inside = np.asarray(b, dtype=float)
        breaks = np.unique(np.concatenate([[lo, hi], inside[(inside > lo)
                                                            & (inside < hi)]]))
        count = (int(degree) + 2) // 2 + 1
    x, w = np.polynomial.legendre.leggauss(count)
    out = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        out.append((0.5 * (b - a) * x + 0.5 * (a + b), 0.5 * (b - a) * w))
    return out


def integrated(model: Model, fields: Fields, *,
               radii: tuple[float, float] | None = None, weight: str = "volume",
               grid: AngularGrid | None = None, n: int = 32,
               component: int | str | None = None,
               frame: str | None = None) -> SphereSlice:
    """A map of a field integrated along each radius, by the rule of the
    module docstring: over the reference radii `radii` (lo, hi), the whole
    body by default, with `weight` one of `WEIGHTS`, on `grid`
    (equiangular at one degree by default); `n` Gauss-Legendre points on
    a layer whose field declares no radial pieces; a component and frame
    as for a sphere.  The slice's radius is the span's outer end, its
    points there."""
    from .sampling import equiangular
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}, got {weight!r}")
    if int(n) < 1:
        raise ValueError(f"n must be at least 1, got {n}")
    b = model.skeleton.boundaries
    lo, hi = (float(b[0]), float(b[-1])) if radii is None else map(float, radii)
    if not hi > lo:
        raise ValueError(f"radii must be (lo, hi) with lo < hi, got {radii}")
    grid = equiangular(180, 360) if grid is None else grid
    values, shown = _evaluator(model, fields, component=component, frame=frame,
                               relative_to=None, departure=None, mean_grid=None)
    th, ph = np.broadcast_arrays(grid.colatitudes[:, None], grid.longitudes[None, :])
    total = np.zeros(th.shape)
    held = False
    for layer in range(model.nlayers):
        a, c = model.skeleton.interval(layer)
        a, c = max(a, lo), min(c, hi)
        field = _field_of(model, fields, layer)
        if field is None or not c > a:
            continue
        held = True
        for r, w in _pieces(field, a, c, n):
            factor = w * (r ** 2 if weight == "volume" else 1.0)
            for rk, fk in zip(r, factor):
                total += fk * values(layer, np.full(th.shape, rk), th, ph)
    if not held:
        raise ValueError(f"no layer within [{lo}, {hi}] holds the field")
    dims = shown.dimensions
    if dims is not None:
        dims = dims * Dimensions(length=3 if weight == "volume" else 1)
    shown = Shown(shown.name, component=component, frame=frame,
                  statistic=f"integrated along the radius, "
                            f"{'r^2 dr' if weight == 'volume' else 'dr'}",
                  dimensions=dims, scales=model.scales)
    points = cartesian_points(hi, grid.colatitudes[:, None], grid.longitudes[None, :])
    return SphereSlice(hi, grid, total, points, _forward(model, points), shown)


def surface(model: Model, function: Callable[[np.ndarray, np.ndarray], ArrayLike],
            interface: int | str, *, name: str | None = None,
            grid: AngularGrid | None = None) -> SphereSlice:
    """A function of (theta, phi) on one of the model's interfaces, by
    index or name, as a sphere slice at its reference radius: on `grid`,
    by default the function's own where it has one (a `GriddedFunction`'s,
    whose values the slice then holds at its nodes) and equiangular at
    one degree otherwise; `name` the function's, its unit the model's
    spec of it."""
    from .sampling import equiangular
    face = model.geometry.interface(interface)
    own = getattr(function, "grid", None)
    grid = own if grid is None else grid
    grid = equiangular(180, 360) if grid is None else grid
    th, ph = np.broadcast_arrays(grid.colatitudes[:, None], grid.longitudes[None, :])
    held = getattr(function, "values", None)
    v = (np.asarray(held) if own is not None and held is not None and own == grid
         else np.asarray(function(th, ph)))
    if np.iscomplexobj(v):
        raise TypeError("the function is complex; slice its real or imaginary part")
    v = np.array(np.broadcast_to(v.astype(float, copy=False), th.shape))
    spec = None if name is None else model.spec(name)
    shown = Shown(name, statistic=f"on {_face_label(face)}",
                  dimensions=None if spec is None else spec.dimensions,
                  scales=model.scales)
    points = cartesian_points(face.radius, th, ph)
    return SphereSlice(face.radius, grid, v, points, _forward(model, points), shown)


def _face_label(face: InterfaceInfo) -> str:
    """An interface as a label names it."""
    return (f"the interface {face.name}" if face.name
            else f"interface {face.index}")


def _in_wedge(phi: np.ndarray, longitudes: tuple[float, float]) -> np.ndarray:
    """Whether longitudes lie in the wedge from `longitudes[0]` east to
    `longitudes[1]`, closed at its start and open at its end."""
    start, end = longitudes
    return wrapped(phi - start) < wrapped(end - start)


def _in_cut(q: np.ndarray, longitudes: tuple[float, float],
            colatitudes: tuple[float, float]) -> np.ndarray:
    """Whether the directions of points (..., 3) lie within the cut's span
    of longitudes and, strictly, of colatitudes."""
    r = np.linalg.norm(q, axis=-1)
    theta = np.arccos(np.clip(q[..., 2] / np.where(r > 0.0, r, 1.0), -1.0, 1.0))
    phi = np.arctan2(q[..., 1], q[..., 0])
    return (_in_wedge(phi, longitudes) & (theta > colatitudes[0])
            & (theta < colatitudes[1]))


def _cutaway_hits(points: np.ndarray, view: np.ndarray, outer: float, inner: float,
                  longitudes: tuple[float, float], colatitudes: tuple[float, float]
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """What the rays toward the viewer through reference points (..., 3)
    meet first on the cut body: the distance t along `view[2]` from the
    picture's plane of the first point met from the viewer's side (-inf
    where none), its face by `CUTAWAY_FACES`, and the point (NaN where
    none).  Each ray is cast against every piece of the body's boundary
    as cut, the surface where it is not removed, the inner sphere where
    it is exposed, the meridional faces and the faces at the
    colatitudes between the radii, and the piece nearest the viewer
    wins."""
    v = view[2]
    X = np.asarray(points, dtype=float)
    p = X - (X @ v)[..., None] * v                    # into the picture's plane
    d2 = np.einsum("...i,...i->...", p, p)
    best_t = np.full(d2.shape, -np.inf)
    best_face = np.zeros(d2.shape, dtype=np.int8)
    best_q = np.full(d2.shape + (3,), np.nan)

    def consider(t: np.ndarray, q: np.ndarray, ok: np.ndarray, code: int) -> None:
        take = ok & (t > best_t)
        best_t[take] = t[take]
        best_face[take] = code
        best_q[take] = q[take]

    def between(q: np.ndarray) -> np.ndarray:
        r = np.linalg.norm(q, axis=-1)
        return (r > inner) & (r < outer)

    for radius, code, exposed in ((outer, 1, False), (inner, 6, True)):
        if not radius > 0.0:
            continue
        ok = d2 < radius * radius
        t = np.sqrt(np.where(ok, radius * radius - d2, 0.0))
        q = p + t[..., None] * v
        removed = _in_cut(q, longitudes, colatitudes)
        consider(t, q, ok & (removed if exposed else ~removed), code)
    for code, phi in zip((2, 3), longitudes):
        along = np.array([np.cos(phi), np.sin(phi), 0.0])
        normal = np.array([-np.sin(phi), np.cos(phi), 0.0])
        denominator = float(normal @ v)
        if abs(denominator) < 1e-15:
            continue                                   # the face seen edge-on
        t = -(p @ normal) / denominator
        q = p + t[..., None] * v
        r = np.linalg.norm(q, axis=-1)
        theta = np.arccos(np.clip(q[..., 2] / np.where(r > 0.0, r, 1.0), -1.0, 1.0))
        consider(t, q, (q @ along > 0.0) & between(q)
                 & (theta > colatitudes[0]) & (theta < colatitudes[1]), code)
    for code, theta in zip((4, 5), colatitudes):
        if not 0.0 < theta < np.pi:
            continue                                   # a pole: no face
        if abs(theta - 0.5 * np.pi) < 1e-12:           # the equatorial plane
            if abs(v[2]) < 1e-15:
                continue
            roots = [-p[..., 2] / v[2]]
        else:                                          # the cone of the colatitude
            T2 = np.tan(theta) ** 2
            a = float(v[0] ** 2 + v[1] ** 2 - T2 * v[2] ** 2)
            b = 2.0 * (p[..., 0] * v[0] + p[..., 1] * v[1] - T2 * p[..., 2] * v[2])
            c = p[..., 0] ** 2 + p[..., 1] ** 2 - T2 * p[..., 2] ** 2
            with np.errstate(divide="ignore", invalid="ignore"):
                if abs(a) < 1e-14:                     # the ray along the cone
                    roots = [np.where(b != 0.0, -c / b, np.nan)]
                else:
                    disc = b * b - 4.0 * a * c
                    root = np.sqrt(np.where(disc >= 0.0, disc, np.nan))
                    roots = [(-b - root) / (2.0 * a), (-b + root) / (2.0 * a)]
        for t in roots:
            ok = np.isfinite(t)
            t = np.where(ok, t, 0.0)
            q = p + t[..., None] * v
            phi = np.arctan2(q[..., 1], q[..., 0])
            nappe = q[..., 2] * np.cos(theta) >= 0.0
            consider(t, q, ok & nappe & between(q) & _in_wedge(phi, longitudes), code)
    return best_t, best_face, best_q


def _view_frame(theta: float, phi: float) -> np.ndarray:
    """The picture's right, up and toward-the-viewer unit vectors, rows,
    for a viewer in the direction (theta, phi): north up, or for a view
    along the axis the direction opposite the view's longitude up."""
    v = cartesian_points(1.0, theta, phi)
    up = np.array([0.0, 0.0, 1.0]) - v[2] * v
    if np.linalg.norm(up) < 1e-9:
        up = -np.array([np.cos(phi), np.sin(phi), 0.0])
    up = up / np.linalg.norm(up)
    return np.stack([np.cross(up, v), up, v])


def cutaway(model: Model, fields: Fields, *,
            longitudes: tuple[float, float] = (0.0, 0.5 * np.pi),
            colatitudes: tuple[float, float] = (0.0, np.pi),
            inner: float | str = 0.0, view: tuple[float, float] | None = None,
            n: int = 400, side: str | None = "below",
            component: int | str | None = None, frame: str | None = None,
            relative_to: RelativeTo = None, departure: str | None = None,
            mean_grid: AngularGrid | None = None) -> CutawaySlice:
    """The body with a wedge cut from it, seen from outside, by the rules
    of the module docstring.

    `longitudes` is the wedge, from its first longitude east to its
    second, in radians, and `colatitudes` the span of colatitudes cut,
    the whole by default, (pi / 2, pi) for the octant south of the
    equator; `inner` the radius the cut goes down to, or the name of
    the interface there (`"cmb"`, `"icb"`), zero for the centre, so that
    the sphere of that radius is left standing in it, or the body's
    inner boundary where that is higher (a shell shows its inner
    surface);
    `view` the colatitude and longitude of the direction toward the
    viewer, in radians, by default 25 degrees toward the pole of the
    hemisphere the cut lies in (north where it straddles the equator)
    and 20 degrees east of the cut's first longitude (a quarter of the
    opening where that is narrower), so that the cut sits at the upper
    right of the picture, its second face nearly square-on and its first
    a sliver, as the classic picture has it; `n` the
    pixels across the picture, which spans the body's outer radius each
    way; `side` the side of an interface the inner sphere is taken on,
    where it lies on one; the component, frame and departure as the
    module docstring says.
    """
    _check("reference", side)
    start, end = (float(x) for x in longitudes)
    if not wrapped(end - start) > 0.0:
        raise ValueError("the wedge is from its first longitude east to its second; "
                         f"longitudes {longitudes} cut nothing")
    north, south = (float(x) for x in colatitudes)
    if not 0.0 <= north < south <= np.pi:
        raise ValueError("colatitudes are a span within [0, pi], the first less "
                         f"than the second, got {colatitudes}")
    outer = float(model.skeleton.boundaries[-1])
    if isinstance(inner, str):
        inner = model.geometry.interface(inner).radius
    inner = float(inner)
    if not 0.0 <= inner < outer:
        raise ValueError(f"inner must lie in [0, {outer:g}), got {inner}")
    inner = max(inner, float(model.skeleton.boundaries[0]))
    n = int(n)
    if n < 2:
        raise ValueError(f"n must be at least 2, got {n}")
    if view is None:
        elevation = np.radians(25.0)
        southern = 0.5 * (north + south) > 0.5 * np.pi
        view = (0.5 * np.pi + (elevation if southern else -elevation),
                start + min(np.radians(20.0), 0.25 * float(wrapped(end - start))))
    frame_ = _view_frame(float(view[0]), float(view[1]))
    values, shown = _evaluator(model, fields, component=component, frame=frame,
                               relative_to=relative_to, departure=departure,
                               mean_grid=mean_grid)

    def rays(count: int) -> np.ndarray:
        """The points of the picture's plane at the centres of `count`
        pixels across, rows from the top, shape (count, count, 3)."""
        c = -outer + (np.arange(count) + 0.5) * (2.0 * outer / count)
        return (c[None, :, None] * frame_[0] - c[:, None, None] * frame_[1])

    wedge, span = (start, end), (north, south)
    _, face, X = _cutaway_hits(rays(n), frame_, outer, inner, wedge, span)
    hit = face > 0
    s, _, _, _ = spherical_coordinates(X[hit])
    layers, sided = _layers_of(model, s, side)
    v = np.full((n, n), np.nan)
    v[hit] = _evaluate_layers(values, layers, X[hit])
    physical = np.full((n, n, 3), np.nan)
    physical[hit] = _forward(model, X[hit])
    k = _COVERAGE_SUBSAMPLES
    t, _, _ = _cutaway_hits(rays(n * k), frame_, outer, inner, wedge, span)
    coverage = np.isfinite(t).reshape(n, k, n, k).mean(axis=(1, 3))
    taken = side if sided and np.any(face == CUTAWAY_FACES.index("inner")) else None
    return CutawaySlice(wedge, span, inner, outer, np.array(model.skeleton.boundaries),
                        frame_, outer, face, X, physical, v, coverage, shown,
                        side=taken)


def normalised(slice_: Slice) -> Slice:
    """The slice divided by its largest absolute finite value, its label
    saying so; a slice whose values are all zero is refused."""
    from dataclasses import replace
    if isinstance(slice_, (SphereSlice, CutawaySlice)):
        parts = [slice_.values]
    else:
        parts = [lay.values for lay in slice_.layers]
    finite = np.concatenate([v[np.isfinite(v)].ravel() for v in parts])
    scale = float(np.max(np.abs(finite))) if finite.size else 0.0
    if not scale > 0.0:
        raise ValueError(f"{slice_.label} has no value but zero to be normalised by")
    shown = replace(slice_.shown, normalised=True)
    if isinstance(slice_, (SphereSlice, CutawaySlice)):
        return replace(slice_, values=slice_.values / scale, shown=shown)
    layers = tuple(replace(lay, values=lay.values / scale) for lay in slice_.layers)
    return replace(slice_, layers=layers, shown=shown)


def evaluate(model: Model, fields: Fields, layer: int, r: ArrayLike, theta: ArrayLike,
             phi: ArrayLike, *, component: int | str | None = None,
             frame: str | None = None, relative_to: RelativeTo = None,
             departure: str | None = None,
             mean_grid: AngularGrid | None = None) -> np.ndarray:
    """What a slice of the same choices shows, at reference points (r,
    theta, phi) of a layer, broadcast: for a set of points of the caller's
    own, as the VTK writer's departures are; NaN where the layer lacks the
    field."""
    values, _ = _evaluator(model, fields, component=component, frame=frame,
                           relative_to=relative_to, departure=departure,
                           mean_grid=mean_grid)
    if not 0 <= int(layer) < model.nlayers:
        raise IndexError(f"the model has {model.nlayers} layers, so no layer {layer}")
    r, theta, phi = np.broadcast_arrays(np.asarray(r, dtype=float),
                                        np.asarray(theta, dtype=float),
                                        np.asarray(phi, dtype=float))
    return values(int(layer), r, theta, phi)


# -- spectra ------------------------------------------------------------------------

def _band_grid(lmax: int, grid: AngularGrid | None) -> AngularGrid:
    """The grid a spectrum's slice is taken on: the caller's, of a shipped
    kind and a band of `lmax` at least, or Gauss-Legendre of twice the
    band."""
    from .sampling import gauss_legendre
    if int(lmax) < 0:
        raise ValueError(f"lmax must be non-negative, got {lmax}")
    if grid is None:
        return gauss_legendre(2 * int(lmax))
    if grid.kind == "custom" or grid.lmax is None or grid.lmax < int(lmax):
        raise ValueError(f"a spectrum to degree {lmax} is analysed on a grid of a "
                         f"shipped kind with a band of {lmax} at least; this one is "
                         f"{grid!r}")
    return grid


def _power(coefficients: np.ndarray) -> np.ndarray:
    """P_l = sum_m c_lm^2 / (4 pi) of (2, L + 1, L + 1) coefficients."""
    return np.einsum("ilm,ilm->l", coefficients, coefficients) / (4.0 * np.pi)


def spectrum(model: Model, fields: Fields, *, lmax: int, radius: float | None = None,
             depth: float | None = None, grid: AngularGrid | None = None,
             side: str | None = None, at: str = "reference",
             component: int | str | None = None, frame: str | None = None,
             relative_to: RelativeTo = None, departure: str | None = None,
             mean_grid: AngularGrid | None = None) -> Spectrum:
    """A field's power by degree to `lmax` on a sphere, by the rules of the
    module docstring; the sphere and the field's choices as `sphere`
    takes them.  A slice with NaN in it (a layer lacking the field, a
    physical sphere leaving the body) is refused."""
    grid = _band_grid(lmax, grid)
    s = sphere(model, fields, radius=radius, depth=depth, grid=grid, side=side, at=at,
               component=component, frame=frame, relative_to=relative_to,
               departure=departure, mean_grid=mean_grid)
    return spectrum_of(s, lmax=lmax)


def spectrum_of(slice_: SphereSlice, *, lmax: int) -> Spectrum:
    """A sphere slice's power by degree to `lmax`, analysed on its own
    grid, which is of a shipped kind with a band of `lmax` at least (twice
    that, for a field not limited to the band, keeps it from aliasing);
    a slice with NaN in it is refused."""
    from .harmonics import analyse_grid
    grid = _band_grid(lmax, slice_.grid)
    if not np.all(np.isfinite(slice_.values)):
        raise ValueError(f"{slice_.label} is not defined everywhere on the sphere of "
                         f"radius {slice_.radius}, so it has no spectrum there")
    c = analyse_grid(slice_.values, grid, lmax=int(lmax))
    return Spectrum(slice_.radius, np.arange(int(lmax) + 1), _power(c), c,
                    slice_.shown, at=slice_.at, side=slice_.side)


def correlation(model: Model, fields: Fields, other: Model, *,
                other_fields: Fields | None = None, lmax: int,
                radii: ArrayLike | None = None, depths: ArrayLike | None = None,
                grid: AngularGrid | None = None, side: str | None = None,
                at: str = "reference", component: int | str | None = None,
                frame: str | None = None, relative_to: RelativeTo = None,
                departure: str | None = None,
                mean_grid: AngularGrid | None = None) -> Correlation:
    """Two fields' correlation by degree and radius, by the rules of the
    module docstring: `fields` of `model` against `other_fields` (the
    same by default) of `other`, which may be `model` itself, at each of
    `radii` or `depths`; both sliced with the same choices."""
    if (radii is None) == (depths is None):
        raise ValueError("give exactly one of radii= and depths=")
    outer = float(model.skeleton.boundaries[-1])
    rs = (np.atleast_1d(np.asarray(radii, dtype=float)) if depths is None
          else outer - np.atleast_1d(np.asarray(depths, dtype=float)))
    grid = _band_grid(lmax, grid)
    other_fields = fields if other_fields is None else other_fields
    kw = dict(lmax=lmax, grid=grid, side=side, at=at, component=component,
              frame=frame, relative_to=relative_to, departure=departure,
              mean_grid=mean_grid)
    first = tuple(spectrum(model, fields, radius=float(r), **kw) for r in rs)
    second = tuple(spectrum(other, other_fields, radius=float(r), **kw) for r in rs)
    by_degree, total = [], []
    for a, b in zip(first, second):
        ca, cb = a.coefficients, b.coefficients
        cross = np.einsum("ilm,ilm->l", ca, cb)
        pa, pb = np.einsum("ilm,ilm->l", ca, ca), np.einsum("ilm,ilm->l", cb, cb)
        heard = ((pa > NO_POWER ** 2 * np.sum(pa)) & (pb > NO_POWER ** 2 * np.sum(pb)))
        with np.errstate(divide="ignore", invalid="ignore"):
            by_degree.append(np.where(heard, cross / np.sqrt(pa * pb), np.nan))
        whole = np.sqrt(np.sum(pa[1:]) * np.sum(pb[1:]))
        quiet = (np.sum(pa[1:]) <= NO_POWER ** 2 * np.sum(pa)
                 or np.sum(pb[1:]) <= NO_POWER ** 2 * np.sum(pb))
        total.append(np.nan if quiet else np.sum(cross[1:]) / whole)
    return Correlation(rs, np.arange(int(lmax) + 1), np.array(by_degree),
                       np.array(total), first, second)
