"""Processing: a data set's fields made fit for a model of one's own.

A data set's perturbations are put onto a model of one's own in steps,
each a function of fields and models, so that every step can be looked
at and replaced.

`rereferenced(source, onto)` gives, for each name, a `Perturbation`: a
kind and the fields that tile the source's support.  Under
`mode="relative"`, the usual practice, the source's own perturbation is
kept as it is (a data set of values made a perturbation of its own
reference first, `source_reference`); under `mode="absolute"` the
source's values are kept, so that the perturbation is relative to
`onto`'s field (v / R_onto - 1 for a relative kind, log(v / R_onto) for
a log one, v - R_onto for an absolute one), and where the two references
differ, at a discontinuity of one the other lacks, the result shows it,
split at `onto`'s boundaries inside the support.  A reference holding a
transversely isotropic pair where the name is isotropic gives its Voigt
average.

`spread(perturbation, onto)` puts one onto whole layers of `onto`, a
field per layer: as it lies by radius, refusing a layer it covers in
part and naming the remedies, which are `ends=(inner, outer)` (the
support mapped affinely onto the radii between two of `onto`'s
interfaces), `extend=True` (the end values continued to the ends of the
layers covered in part) and `taper=` (a width, or a bottom and a top
one, over which the perturbation goes to zero inside each end of its
support, by `raised_cosine`, and zero beyond).  A layer that one
segment covers as it lies, untapered, keeps the segment's own field, a
harmonic field harmonic.

`filtered(field, lmax=)` weights a field by degree: coefficients cut
above `lmax` and below `lmin`, or rolled off by the raised cosine over
`roll_off` degrees below `lmax`.  A harmonic field gives a harmonic
field; a gridded field on a grid of a shipped kind is analysed at each
node, weighted and synthesised back onto its own grid or `grid=`; a
perturbed field is filtered in its perturbation;
a field composed from a gridded one (a data set of values made relative
to a reference) is first sampled onto that field's grid and radial
nodes (`resampled`).  `demeaned` is a field less its spherical mean at
each radius, exact for a harmonic field (its degree 0 gone) and for a
gridded one (the mean of its lateral rule taken at every node).

`merged(model, regional, name=)` blends a `RegionalGrid` into the
field `name` of a model, both expressed relative to the model's own
reference (a regional grid of values made a perturbation of it):
f = g + w (r - g) with w the raised cosine of the great-circle distance
inside the region's edge over `width` (an angle), and of radius over
`radial_width` inside its depth ends (zero for a sharp end, as at the
surface), w zero where the regional grid has no value; on each layer
the region reaches, a `GriddedField` on `grid` between the regional
grid's slices, its poles evaluated, attached through `with_field`, so
that a model type's own rules run.  The result is the global field
resampled on `grid` outside the region, equal to it at the grid's nodes;
a layer whose reference vanishes (a fluid's shear velocity) is left as
it is.

`raised_cosine` is the one taper all of them use.

Every length and every value is in the units of the model the result
goes onto.  A source model, and its reference, in other scales than
that model are converted to its scales first (`rereferenced`, and so
`transplanted`); a regional grid is SI, and its radii and values are
put in the model's units (`merged`); a `Perturbation` carries no
scales, so the fields given to `spread` are in the model's units, as
`rereferenced` gives them.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly

from ..character import SCALAR
from ..expansions import HarmonicField
from ..fields import AnalyticField, ComposedField, Field
from ..gridded import GriddedField
from ..layerfunction import same_interval
from ..model import Model
from ..perturbation import KINDS, PerturbedField
from ..sampling import AngularGrid
from ..units import LENGTH
from .grids import RegionalGrid
from .models import reference_field

__all__ = ["raised_cosine", "Perturbation", "MODES", "rereferenced", "spread",
           "filtered", "resampled", "demeaned", "merged"]

#: What re-referencing keeps: the source's perturbation, or its values.
MODES = ("relative", "absolute")


def raised_cosine(t: ArrayLike) -> np.ndarray:
    """0 for t <= 0, 1 for t >= 1, (1 - cos(pi t)) / 2 between: the taper of
    every step here, continuous with its first derivative."""
    t = np.clip(np.asarray(t, dtype=float), 0.0, 1.0)
    return 0.5 * (1.0 - np.cos(np.pi * t))


@dataclass(frozen=True)
class Perturbation:
    """A perturbation of the field `name`, of one of `perturbation.KINDS`,
    as the fields that tile its support from the centre out."""

    name: str
    kind: str
    segments: tuple[Field, ...]

    def __post_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {self.kind!r}")
        if not self.segments:
            raise ValueError(f"the perturbation of {self.name!r} has no fields")
        object.__setattr__(self, "segments", tuple(self.segments))

    @property
    def support(self) -> tuple[float, float]:
        """The radii its segments run between."""
        return (float(self.segments[0].interval[0]),
                float(self.segments[-1].interval[1]))


#: How a value v and a reference R make a perturbation of each kind.
_AGAINST: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    "relative": lambda v, R: v / R - 1.0,
    "log": lambda v, R: np.log(v / R),
    "absolute": lambda v, R: v - R,
}


def _layer_at(model: Model, r: float) -> Any:
    """The layer of a model holding radius r."""
    return model.layer(model.skeleton.locate(r).layer)


def _pieces(holders: list[Any], *, split: Sequence[float],
            span: tuple[float, float] | None = None) -> list[tuple[float, float, Any]]:
    """(lo, hi, layer) for every holder layer cut at the radii of `split`,
    within `span` where one is given."""
    out = []
    for layer in holders:
        lo, hi = (float(x) for x in layer.interval)
        if span is not None:
            lo, hi = max(lo, span[0]), min(hi, span[1])
            if not hi > lo:
                continue
        cuts = sorted({lo, hi} | {float(r) for r in split if lo < r < hi})
        out += [(a, b, layer) for a, b in zip(cuts, cuts[1:])]
    return out


def rereferenced(source: Model, onto: Model, *, names: Sequence[str] | None = None,
                 mode: str = "relative", source_reference: Model | None = None,
                 kind: str = "relative",
                 span: tuple[float, float] | None = None) -> dict[str, Perturbation]:
    """Each name's perturbation, by the rule of the module docstring:
    `names` the source's names to carry (every one it holds by default),
    `source_reference` the reference of a data set of values under
    `mode="relative"`, and `kind` the kind a data set of values is
    expressed in (a perturbed source keeps its own), and `span` the radii
    (lo, hi) of the source taken, all of its support by default.  The
    perturbations are in `onto`'s units, `span` too: a source or a
    `source_reference` in other scales is converted to `onto`'s first."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if source.scales != onto.scales:
        source = source.converted(onto.scales)
    if source_reference is not None and source_reference.scales != onto.scales:
        source_reference = source_reference.converted(onto.scales)
    if names is None:
        names = sorted({n for layer in source.layers for n in layer.names})
    out = {}
    for name in names:
        holders = [layer for layer in source.layers if name in layer]
        if not holders:
            raise ValueError(f"the source holds no {name!r}")
        perturbed = [getattr(layer[name], "perturbation", None) is not None
                     for layer in holders]
        if any(perturbed) and not all(perturbed):
            raise ValueError(f"{name!r} is perturbed on some of the source's layers "
                             "and not on others")
        kinds = {layer[name].kind for layer in holders} if all(perturbed) else {kind}
        if len(kinds) != 1:
            raise ValueError(f"{name!r} is perturbed of several kinds: {kinds}")
        (out_kind,) = kinds
        against = _AGAINST[out_kind]
        if all(perturbed) and mode == "relative":
            pieces = []
            for layer in holders:
                p = layer[name].perturbation
                lo, hi = (float(x) for x in p.interval)
                if span is not None:
                    lo, hi = max(lo, span[0]), min(hi, span[1])
                    if not hi > lo:
                        continue
                pieces.append(p if same_interval(p.interval, (lo, hi))
                              else p.on_interval(lo, hi))
            out[name] = Perturbation(name, out_kind, tuple(pieces))
            continue
        if mode == "relative":
            if source_reference is None:
                raise ValueError(f"the source holds {name!r} as values: under "
                                 "mode='relative' it needs source_reference=")
            ref_model, split = source_reference, source_reference.skeleton.boundaries
        else:
            ref_model, split = onto, onto.skeleton.boundaries
        pieces = []
        for a, b, layer in _pieces(holders, split=split, span=span):
            v = layer[name]
            v = v if same_interval(v.interval, (a, b)) else v.on_interval(a, b)
            R = reference_field(_layer_at(ref_model, 0.5 * (a + b)), name)
            R = R if same_interval(R.interval, (a, b)) else R.on_interval(a, b)
            character = SCALAR if out_kind in ("relative", "log") else v.character
            pieces.append(ComposedField(against, (v, R), character=character,
                                        name=name))
        out[name] = Perturbation(name, out_kind, tuple(pieces))
    return out


def _interface_radius(model: Model, which: int | str) -> float:
    """The radius of an interface of a model, by name or by index."""
    faces = model.geometry.interfaces
    for i, face in enumerate(faces):
        if which == face.name or which == i:
            return float(face.radius)
    raise ValueError(f"the model has no interface {which!r}; it has "
                     f"{[f.name for f in faces]}")


def _widths(taper: float | tuple[float, float] | None) -> tuple[float, float]:
    """A taper's bottom and top widths."""
    if taper is None:
        return 0.0, 0.0
    if np.ndim(taper) == 0:
        return float(taper), float(taper)
    lo, hi = taper
    return float(lo), float(hi)


def spread(perturbation: Perturbation, onto: Model, *,
           layers: Sequence[int | str] | None = None,
           ends: tuple[int | str, int | str] | None = None, extend: bool = False,
           taper: float | tuple[float, float] | None = None) -> dict[int, Field]:
    """The perturbation on whole layers of `onto`, by layer index, by the
    rule of the module docstring.  The perturbation's fields, and the
    widths of `taper`, are in `onto`'s units, as `rereferenced` gives
    them: a `Perturbation` carries no scales to convert it by."""
    segs = perturbation.segments
    lo, hi = perturbation.support
    if ends is not None:
        t_lo, t_hi = _interface_radius(onto, ends[0]), _interface_radius(onto, ends[1])
        if not t_hi > t_lo:
            raise ValueError(f"the interfaces {ends} do not bound a span from the "
                             "inside out")
        scale = (hi - lo) / (t_hi - t_lo)
    else:
        t_lo, t_hi, scale = lo, hi, 1.0
    w_lo, w_hi = _widths(taper)
    tol = onto.geometry.rtol * onto.skeleton.span
    if layers is None:
        chosen = [layer for layer in onto.layers
                  if layer.interval[1] > t_lo + tol and layer.interval[0] < t_hi - tol]
    else:
        chosen = [onto.layer(w) for w in layers]
    if not chosen:
        raise ValueError(f"the perturbation of {perturbation.name!r} on "
                         f"[{t_lo:g}, {t_hi:g}] reaches no layer of the model")

    def to_source(r: np.ndarray) -> np.ndarray:
        return lo + (r - t_lo) * scale

    def weight(r: np.ndarray) -> np.ndarray:
        w = np.ones_like(r)
        if w_lo > 0.0:
            w = w * raised_cosine((r - t_lo) / w_lo)
        if w_hi > 0.0:
            w = w * raised_cosine((t_hi - r) / w_hi)
        return w

    def fn(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        r, theta, phi = np.broadcast_arrays(np.asarray(r, dtype=float), theta, phi)
        rs = to_source(r)
        if extend:
            rs = np.clip(rs, lo, hi)
        else:                                   # the ends, to the model's tolerance
            near = (r >= t_lo - tol) & (r <= t_hi + tol)
            rs = np.where(near, np.clip(rs, lo, hi), rs)
        out = np.zeros(r.shape + tuple(segs[0].stored_shape))
        for k, seg in enumerate(segs):
            a, b = (float(x) for x in seg.interval)
            last = k == len(segs) - 1
            sel = (rs >= a) & ((rs <= b) if last else (rs < b))
            if sel.any():
                out[sel] = seg.evaluate(rs[sel], theta[sel], phi[sel])
        if w_lo <= 0.0 and w_hi <= 0.0:
            return out
        w = weight(r)
        return out * w.reshape(w.shape + (1,) * (out.ndim - w.ndim))

    result = {}
    for layer in chosen:
        a, b = (float(x) for x in layer.interval)
        if b <= t_lo + tol or a >= t_hi - tol:
            raise ValueError(f"the perturbation of {perturbation.name!r} on "
                             f"[{t_lo:g}, {t_hi:g}] does not reach layer {layer.index} "
                             f"({layer.name})")
        covered = a >= t_lo - tol and b <= t_hi + tol
        if not covered and not (extend or taper is not None):
            raise ValueError(
                f"the perturbation of {perturbation.name!r} on [{t_lo:g}, {t_hi:g}] "
                f"covers layer {layer.index} ({layer.name}) on [{a:g}, {b:g}] only in "
                "part; ends=(inner, outer) maps it onto two of the model's "
                "interfaces, extend=True continues its end values, taper= takes it "
                "to zero inside its ends")
        whole = [s for s in segs if float(s.interval[0]) <= a + tol
                 and float(s.interval[1]) >= b - tol]
        if ends is None and covered and taper is None and whole:
            seg = whole[0]
            result[layer.index] = (seg if same_interval(seg.interval, (a, b))
                                   else seg.on_interval(a, b))
        else:
            result[layer.index] = AnalyticField((a, b), fn, character=segs[0].character,
                                                name=perturbation.name)
    return result


def _degree_weights(lmax: int, lmin: int, roll_off: int | None) -> np.ndarray:
    """w_l for l = 0 ... lmax: zero below lmin, one up to lmax - roll_off,
    the raised cosine down to lmax."""
    l = np.arange(lmax + 1)
    w = np.where(l >= lmin, 1.0, 0.0)
    if roll_off:
        k = int(roll_off)
        fall = l > lmax - k
        w = np.where(fall, w * raised_cosine((lmax + 1 - l) / (k + 1)), w)
    return w


def _gridded_source(field: Field) -> GriddedField | None:
    """The gridded field a field is composed from, if any."""
    if isinstance(field, GriddedField):
        return field
    for source in getattr(field, "sources", ()):
        found = _gridded_source(source)
        if found is not None:
            return found
    return None


def resampled(field: Field, *, grid: AngularGrid | None = None) -> GriddedField:
    """A field composed from a gridded one, sampled onto that field's grid
    (or `grid`) at the nodes of its radial elements within the field's
    interval, with its rules: a gridded field equal to the field at every
    node and pole."""
    source = _gridded_source(field)
    if source is None:
        raise TypeError(f"{field!r} is composed from no gridded field, so it has no "
                        "grid or radial nodes of its own to be sampled on")
    lo, hi = (float(x) for x in field.interval)
    inner = source.breaks[(source.breaks > lo) & (source.breaks < hi)]
    return GriddedField.sampled(field, source.grid if grid is None else grid,
                                breaks=np.concatenate([[lo], inner, [hi]]),
                                ngll=source.ngll, radial=source.radial,
                                lateral=source.lateral)


def demeaned(field: Field) -> Field:
    """The field less its spherical mean at each radius, by the rule of the
    module docstring; a field composed from a gridded one is resampled
    first."""
    pert = getattr(field, "perturbation", None)
    if pert is not None:
        return PerturbedField(field.reference, demeaned(pert), kind=field.kind,
                              name=field.name)
    if isinstance(field, HarmonicField):
        return filtered(field, lmax=field.lmax, lmin=1)
    if not isinstance(field, GriddedField):
        field = resampled(field)
    mean = np.asarray(field.spherical_mean()(field.radii))
    shape = mean.shape + (1,) * (field.values.ndim - mean.ndim)
    poles = None if field.poles is None else field.poles - mean[..., None]
    return GriddedField(field.interval, field.breaks, field.grid,
                        field.values - mean.reshape(shape), character=field.character,
                        name=field.name, radial=field.radial, lateral=field.lateral,
                        poles=poles)


def filtered(field: Field, *, lmax: int, lmin: int = 0, roll_off: int | None = None,
             grid: AngularGrid | None = None) -> Field:
    """The field weighted by degree, by the rule of the module docstring."""
    lmax, lmin = int(lmax), int(lmin)
    if not 0 <= lmin <= lmax:
        raise ValueError(f"the band must have 0 <= lmin <= lmax, got {lmin}, {lmax}")
    if roll_off is not None and not 0 < int(roll_off) <= lmax:
        raise ValueError(f"roll_off must be a number of degrees in 1 ... lmax, got "
                         f"{roll_off}")
    w = _degree_weights(lmax, lmin, roll_off)
    pert = getattr(field, "perturbation", None)
    if pert is not None:
        return PerturbedField(field.reference,
                              filtered(pert, lmax=lmax, lmin=lmin, roll_off=roll_off,
                                       grid=grid),
                              kind=field.kind, name=field.name)
    if isinstance(field, HarmonicField):
        pp = field.coefficients
        L = min(lmax, field.lmax)
        c = np.array(pp.c[..., :L + 1, :L + 1])
        c *= w[:L + 1].reshape((L + 1, 1))
        return HarmonicField(field.interval, PPoly(c, pp.x), character=field.character,
                             name=field.name)
    if isinstance(field, GriddedField):
        from ..harmonics import analyse_grid, synthesise_grid

        source = field.grid
        target = source if grid is None else grid
        for g, what in ((source, "the field's grid"), (target, "grid=")):
            if g.kind == "custom" or g.lmax is None or g.lmax < lmax:
                raise ValueError(f"filtering a gridded field to degree {lmax} needs "
                                 f"{what} of a shipped kind with a band of {lmax} at "
                                 f"least; it is {g!r}")
        c = analyse_grid(field.values, source, lmax=lmax)
        c *= w.reshape((1, lmax + 1, 1) + (1,) * (c.ndim - 3))
        values = synthesise_grid(c, target)
        return GriddedField(field.interval, field.breaks, target, values,
                            character=field.character, name=field.name,
                            radial=field.radial, lateral=field.lateral)
    if _gridded_source(field) is not None:
        return filtered(resampled(field), lmax=lmax, lmin=lmin, roll_off=roll_off,
                        grid=grid)
    raise TypeError(f"filtering by degree takes a harmonic or a gridded field (or one "
                    f"perturbed by one, or composed from one); {field!r} is none: "
                    "sample it onto a grid first (GriddedField.sampled)")


def _unit(model: Model, name: str) -> float:
    """One of the model's units of its field `name`, in SI: the factor of
    the dimensions its spec gives the name, refused where a model in other
    scales than SI gives it none."""
    if model.scales.is_si:
        return 1.0
    spec = model.spec(name)
    if spec is None or spec.dimensions is None:
        raise ValueError(f"{name!r} has no dimensions in the model, so a regional "
                         "grid's SI values cannot be put in the model's units: give "
                         "it a spec")
    return model.scales.factor(spec.dimensions)


def _edge_weight(regional: RegionalGrid, theta: np.ndarray, phi: np.ndarray,
                 width: float) -> np.ndarray:
    """The raised cosine of the angle inside the region's edge
    (`RegionalGrid.edge_distance`) over `width`, zero outside."""
    d = regional.edge_distance(theta, phi)
    w = raised_cosine(d / width) if width > 0.0 else np.ones_like(d)
    return np.where(regional.contains(theta, phi), w, 0.0)


def merged(model: Model, regional: RegionalGrid, *, name: str, width: float,
           grid: AngularGrid, radial_width: float | tuple[float, float] = 0.0,
           source_name: str | None = None, lateral: str = "linear") -> Model:
    """The model with the regional grid's `source_name` (`name` by default)
    blended into its field `name`, by the rule of the module docstring:
    `width` an angle in radians, `radial_width` a length in the model's
    units (a bottom and a top one), `grid` the angular grid of the result.
    The regional grid is SI; its radii, and its values or its absolute
    perturbation, are put in the model's units by the model's scales."""
    source_name = name if source_name is None else source_name
    if source_name not in regional.names:
        raise ValueError(f"the regional grid holds {regional.names}, not "
                         f"{source_name!r}")
    if not width >= 0.0:
        raise ValueError(f"width must be a non-negative angle, got {width}")
    regional_kind = regional.kinds[source_name]
    length = model.scales.factor(LENGTH)
    unit = 1.0 if regional_kind in ("relative", "log") else _unit(model, name)
    r0, r1 = (x / length for x in regional.extent[0])
    w_lo, w_hi = _widths(radial_width)
    tol = model.geometry.rtol * model.skeleton.span
    out = model.with_citation(*regional.citation)
    reached = False
    for layer in model.layers:
        a, b = (float(x) for x in layer.interval)
        if b <= r0 + tol or a >= r1 - tol or name not in layer:
            continue
        f = layer[name]
        pert = getattr(f, "perturbation", None)
        ref = f.reference if pert is not None else f
        kind = f.kind if pert is not None else "relative"
        if not getattr(ref, "is_radial", False):
            raise ValueError(f"{name!r} on layer {layer.index} is not a radial "
                             "reference or a perturbation of one")
        if kind != "absolute" and np.any(np.asarray(
                ref.evaluate(np.array([a, 0.5 * (a + b), b]), np.zeros(3),
                             np.zeros(3))) == 0.0):
            continue                                    # a fluid's shear velocity
        reached = True
        if regional_kind is not None and regional_kind != kind:
            raise ValueError(f"the regional grid's {source_name!r} is a "
                             f"{regional_kind} perturbation; the model's {name!r} is "
                             f"{kind}")
        against = _AGAINST[kind]

        def blend(r: np.ndarray, theta: np.ndarray, phi: np.ndarray,
                  *, ref: Field = ref, pert: Field | None = pert,
                  against: Callable[..., np.ndarray] = against) -> np.ndarray:
            r, theta, phi = np.broadcast_arrays(np.asarray(r, dtype=float), theta, phi)
            g = (pert.evaluate(r, theta, phi) if pert is not None
                 else np.zeros(r.shape))
            v = regional(source_name, r * length, theta, phi) / unit
            local = (v if regional_kind is not None
                     else against(v, ref.evaluate(r, theta, phi)))
            w = _edge_weight(regional, theta, phi, width)
            if w_lo > 0.0:
                w = w * raised_cosine((r - r0) / w_lo)
            if w_hi > 0.0:
                w = w * raised_cosine((r1 - r) / w_hi)
            w = np.where(np.isfinite(local) & (r >= r0) & (r <= r1), w, 0.0)
            return g + w * np.where(np.isfinite(local), local - g, 0.0)

        radii = regional.radii / length
        breaks = np.unique(np.concatenate([[a, b], radii[(radii > a) & (radii < b)]]))
        sampled = GriddedField.sampled(AnalyticField((a, b), blend, name=name), grid,
                                       breaks=breaks, ngll=2, radial="linear",
                                       lateral=lateral)
        out = out.with_field(layer.index, name,
                             PerturbedField(ref, sampled, kind=kind, name=name),
                             replace=True)
    if not reached:
        raise ValueError(f"the regional grid on [{r0:g}, {r1:g}] reaches no layer of "
                         f"the model holding {name!r} (a fluid's shear velocity "
                         "aside)")
    return out
