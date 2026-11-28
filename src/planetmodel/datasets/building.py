"""Models built from data sets: a data set's perturbations on a model of one's own.

`transplanted(model, source)` is the steps of `datasets.processing` in
their order, for every name the source carries:

1. re-referenced (`rereferenced`): the source's perturbation kept, or
   under `mode="absolute"` its values against the model's own field; a
   data set of values made a perturbation of its own reference first,
   `source_reference` or, for a data set given by its entry, the
   reference the entry names;
2. its spherical mean at each radius removed when `remove_mean=True`
   (degree 0; on any field, no extra needed but for a resampling);
3. filtered by degree when `lmax=` (or `lmin=`) asks, rolled off over
   `roll_off` degrees;
4. spread onto whole layers (`ends=`, `extend=`, `taper=`, `layers=`);
5. attached by `perturbed`, so that a model type's own rules run: on an
   `Elastic` model a perturbed velocity re-derives the Love moduli.

A source name the model holds lands on itself; an isotropic name (`vs`,
`vp`) lands on the transversely isotropic pair the model holds where the
source does not carry that pair itself (`vs` on `vsv` and `vsh`, as
SPECFEM applies S40RTS to PREM), and is otherwise left out; `names=`
maps each source name to the model's names explicitly.  Conversions,
velocity-to-density scalings among them, are not made here.  The model
cites what it cited and then what the source does.

Strata (`datasets.strata`) go onto a model by one of two routes
(`ROUTES`).  **As geometry**, chosen interfaces of the model move to
chosen surfaces of the strata (`interfaces`, each model interface to an
index into `Strata.surfaces`, a stratum's name for its top, or one of
the surfaces itself, deeper interfaces to deeper surfaces): each
interface's shift is its `relief`, the surface's radius in the model's
units less the interface's, held on the surfaces' grid, and the model is
stretched by `layer_linear` over exactly these reliefs, every other
boundary staying spherical.  `reliefs` gives the shifts alone, keyed as
asked, for a boundary-perturbation method that takes them rather than a
deformed model.  A layer of the model taken to zero or negative
thickness anywhere on the grid is refused, so a stratum that thins to
nothing is merged with its neighbours into one layer, never a layer of
its own.  The one model layer between two moved interfaces holds the
strata between their surfaces, merged: each property the
thickness-weighted mean over them (`take` names a stratum whose own
values a layer holds instead), varying with the angles and constant in
radius, exact on the grid.  **As fields**, the skeleton stays
spherical, and each point of the chosen layers (by default those the
strata reach) takes the property of the stratum its radius falls in,
linear in radius within it, `Strata.below` beneath the last surface
where it holds the property and the model's own field elsewhere,
sampled on the grid at radial elements `spacing` apart: smeared across
the strata's surfaces by nature.  A property lands on the names the
layer holds as a source name does (`vp` on `vpv` and `vph`), or as
`properties` maps it, in the model's units; on an `Elastic` model the
moduli follow.  Either way the model cites the strata after its own.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..displacement import layer_linear
from ..gridded import GriddedField, GriddedFunction
from ..model import Layer, Model
from ..perturbation import perturbed
from ..sampling import AngularGrid
from .entries import DatasetLike, as_dataset
from .fetch import load
from .models import reference_of
from .processing import demeaned, filtered, rereferenced, spread
from .strata import Strata
from .surfaces import Surface

__all__ = ["transplanted", "PAIRS", "ROUTES", "SurfaceOf", "relief", "reliefs",
           "with_strata", "moved"]

#: The routes by which `with_strata` puts strata on a model.
ROUTES = ("geometry", "fields")

#: Which surface of strata an interface moves to: an index into
#: `Strata.surfaces` (from the top down), a stratum's name (its top), or
#: one of the surfaces itself.
type SurfaceOf = int | str | Surface

#: The transversely isotropic pair an isotropic name lands on.
PAIRS = {"vs": ("vsv", "vsh"), "vp": ("vpv", "vph")}


def _held(model: Model) -> set[str]:
    """Every name some layer of the model holds."""
    return {n for layer in model.layers for n in layer.names}


def _landing(source: Model, model: Model,
             names: Mapping[str, str | Sequence[str]] | None
             ) -> dict[str, tuple[str, ...]]:
    """Each source name and the model names it lands on."""
    if names is not None:
        return {k: (v,) if isinstance(v, str) else tuple(v) for k, v in names.items()}
    carried, held = _held(source), _held(model)
    out: dict[str, tuple[str, ...]] = {}
    for name in sorted(carried):
        if name in held:
            out[name] = (name,)
        elif (name in PAIRS and all(p in held for p in PAIRS[name])
              and not any(p in carried for p in PAIRS[name])):
            out[name] = PAIRS[name]
    if not out:
        raise ValueError(f"the source carries {sorted(carried)}, none of which the "
                         f"model holds (it holds {sorted(held)}); names= maps them")
    return out


def transplanted(model: Model, source: Model | DatasetLike, *,
                 names: Mapping[str, str | Sequence[str]] | None = None,
                 mode: str = "relative", remove_mean: bool = False,
                 layers: Sequence[int | str] | None = None,
                 ends: tuple[int | str, int | str] | None = None, extend: bool = False,
                 taper: float | tuple[float, float] | None = None,
                 lmax: int | None = None, lmin: int = 0, roll_off: int | None = None,
                 source_reference: Model | None = None,
                 kind: str = "relative") -> Model:
    """The model with the source's perturbations on it, by the steps of the
    module docstring: `source` a data set's model or its entry (loaded),
    the rest each step's own arguments.  A source, or a
    `source_reference`, in other scales than the model is converted to
    the model's first (`rereferenced`), and every length given (`taper`)
    is in the model's units."""
    if not isinstance(source, Model):
        d = as_dataset(source)
        loaded = load(d)
        if not isinstance(loaded, Model):
            raise TypeError(f"{d.name} reads as {type(loaded).__name__}, not a model "
                            "to transplant (a regional grid is merged)")
        if source_reference is None and d.reference is not None and mode == "relative":
            source_reference = reference_of(d)
        source = loaded
    landing = _landing(source, model, names)
    span = None
    if layers is not None:
        chosen = [model.layer(k).interval for k in layers]
        span = (min(lo for lo, _ in chosen), max(hi for _, hi in chosen))
    perturbations = rereferenced(source, model, names=list(landing), mode=mode,
                                 source_reference=source_reference, kind=kind,
                                 span=span)
    out = model.with_citation(*source.citation, *(
        () if source_reference is None else source_reference.citation))
    for name, p in perturbations.items():
        segments = p.segments
        if remove_mean:
            segments = tuple(demeaned(s) for s in segments)
        if lmax is not None:
            segments = tuple(filtered(s, lmax=lmax, lmin=lmin, roll_off=roll_off)
                             for s in segments)
        elif lmin:
            raise ValueError("lmin= filters by degree, and needs lmax= beside it")
        p = type(p)(p.name, p.kind, segments)
        on = spread(p, out, layers=layers, ends=ends, extend=extend, taper=taper)
        for target in landing[name]:
            for index, field in on.items():
                out = perturbed(out, target, field, kind=p.kind, layers=[index])
    return out


# -- strata ----------------------------------------------------------------------

def _gridded_on(surfaces: Sequence[Surface], grid: AngularGrid | None
                ) -> AngularGrid:
    """`grid`, or the grid the surfaces' functions share."""
    own = [s.function for s in surfaces if isinstance(s.function, GriddedFunction)]
    if grid is not None:
        return grid
    if len(own) == len(surfaces) and all(f.grid == own[0].grid for f in own):
        return own[0].grid
    raise ValueError("the surfaces are not held on one grid of their own: grid= "
                     "gives the angular grid of the reliefs")


def _own_lateral(strata: Strata) -> str:
    """The lateral rule the strata's own functions are read by, linear
    where they have none."""
    for s in strata.surfaces:
        if isinstance(s.function, GriddedFunction):
            return s.function.lateral
    return "linear"


def _where(theta: float, phi: float) -> str:
    lat, lon = 90.0 - np.degrees(theta), np.degrees(phi)
    lon = lon - 360.0 if lon > 180.0 else lon
    return (f"{abs(lat):.2f} {'N' if lat >= 0 else 'S'}, "
            f"{abs(lon):.2f} {'E' if lon >= 0 else 'W'}")


def relief(surface: Surface, model: Model, interface: int | str, *,
           grid: AngularGrid | None = None, lateral: str = "cubic"
           ) -> GriddedFunction:
    """The surface as the relief of the model's `interface`, by the rule of
    the module docstring: its radius in the model's units less the
    interface's, on `grid` (the surface's own where its function is
    gridded), read by `lateral`, bicubic by default so that a mapping
    built on it has a gradient everywhere, the poles included."""
    grid = _gridded_on([surface], grid)
    radius = model.geometry.interface(interface).radius
    length = model.scales.length

    def shift(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        return surface.radius(theta, phi) / length - radius

    return GriddedFunction.sampled(shift, grid, lateral=lateral)


def _moves(strata: Strata, model: Model, interfaces: Mapping[int | str, SurfaceOf]
           ) -> list[tuple[int | str, int, int]]:
    """(key, interface index, surface index) of every move, centre outward,
    the surfaces in the same order, from the bottom up."""
    if not interfaces:
        raise ValueError("interfaces= names at least one interface and its surface")
    surfaces = strata.surfaces
    out = []
    for key, which in interfaces.items():
        face = model.geometry.interface(key).index
        if isinstance(which, Surface):
            found = [k for k, s in enumerate(surfaces) if s is which]
            if not found:
                raise ValueError(f"{which!r} is not a surface of {strata!r}")
            k = found[0]
        elif isinstance(which, str):
            k = strata.names.index(strata[which].name)
        else:
            k = range(len(surfaces))[int(which)]
        out.append((key, face, k))
    out.sort(key=lambda m: m[1])
    for (key0, f0, k0), (key1, f1, k1) in zip(out[:-1], out[1:]):
        if f0 == f1:
            raise ValueError(f"the interfaces {key0!r} and {key1!r} are one")
        if not k1 < k0:
            raise ValueError(
                f"{key1!r} lies above {key0!r} in the model, but its surface "
                f"{surfaces[k1].name or k1!r} is not above {surfaces[k0].name or k0!r} "
                "in the strata")
    return out


def _stack(model: Model, held: Mapping[int, GriddedFunction]
           ) -> list[GriddedFunction | None]:
    """One relief per boundary of the skeleton from the reliefs of the
    interfaces `held`, None elsewhere."""
    b = model.skeleton.boundaries
    out: list[GriddedFunction | None] = [None] * len(b)
    for face, h in held.items():
        out[int(np.argmin(np.abs(b - model.geometry.interface(face).radius)))] = h
    return out


def _check_thickness(model: Model, stack: Sequence[GriddedFunction | None],
                     grid: AngularGrid) -> None:
    """Refuse a layer taken to zero or negative thickness at a node or a
    pole of the grid, or half way between nodes, where a bicubic relief
    may overshoot, naming it, where, and what may be done."""
    t = np.concatenate([grid.colatitudes, 0.5 * (grid.colatitudes[:-1]
                                                 + grid.colatitudes[1:])])
    f = grid.longitudes
    f = np.concatenate([f, 0.5 * (f + np.append(f[1:], f[0] + 2.0 * np.pi))])
    at_node = np.concatenate([
        np.outer(np.arange(t.size) < grid.ntheta, np.arange(f.size) < grid.nphi)
        .ravel(), [True, True]])
    theta = np.concatenate([np.repeat(t, f.size), [0.0, np.pi]])
    phi = np.concatenate([np.tile(f, t.size), [0.0, 0.0]])
    moved = np.array([b + (np.zeros_like(theta) if h is None else h(theta, phi))
                      for b, h in zip(model.skeleton.boundaries, stack)])
    thick = np.diff(moved, axis=0)
    for i in range(thick.shape[0]):
        j = int(np.argmin(thick[i]))
        if not thick[i, j] > 0.0:
            layer = model.layer(i)
            label = f"{i} ({layer.name})" if layer.name else str(i)
            place = "a node" if at_node[j] else "a point half way between nodes"
            raise ValueError(
                f"layer {label} would be {thick[i, j]:.6g} thick at "
                f"{_where(theta[j], phi[j])}, {place}; a layer is never taken to "
                "zero or negative thickness.  A moved interface may cross one of "
                "the model's own there (move that one too, or remove it with "
                "coarsened), a stratum may thin to nothing (merge it with its "
                "neighbours between two moved interfaces), or a bicubic relief "
                "may overshoot between nodes (lateral='linear' never does)")


def reliefs(strata: Strata, model: Model, *, interfaces: Mapping[int | str, SurfaceOf],
            grid: AngularGrid | None = None, lateral: str = "cubic"
            ) -> dict[int | str, GriddedFunction]:
    """Each interface's relief from its surface of the strata (`relief`),
    keyed as `interfaces` is, centre outward, on one grid (`grid`, or the surfaces' own)
    read by `lateral`, refused where a layer would be taken to zero or
    negative thickness, as the module docstring says."""
    moves = _moves(strata, model, interfaces)
    grid = _gridded_on([strata.surfaces[k] for _, _, k in moves], grid)
    out = {key: relief(strata.surfaces[k], model, face, grid=grid, lateral=lateral)
           for key, face, k in moves}
    _check_thickness(model, _stack(model, {face: out[key] for key, face, _ in moves}),
                     grid)
    return out


def _on_nodes(fn: object, grid: AngularGrid) -> tuple[np.ndarray, np.ndarray]:
    """A function of the angles at the grid's nodes (ntheta, nphi) and its
    poles (2,)."""
    nodes = np.broadcast_to(np.asarray(fn(grid.colatitudes[:, None],
                                          grid.longitudes[None, :]), dtype=float),
                            (grid.ntheta, grid.nphi))
    poles = np.asarray(fn(np.array([0.0, np.pi]), np.zeros(2)), dtype=float)
    return np.array(nodes), poles


def _lands(layer: Layer, properties: Mapping[str, str | Sequence[str]] | None,
           strata_held: Sequence[str]) -> dict[str, tuple[str, ...]]:
    """Each property every stratum holds and the names of the layer it
    lands on, as `properties` maps them or as a source name lands."""
    if properties is not None:
        out = {k: (v,) if isinstance(v, str) else tuple(v)
               for k, v in properties.items()}
        for k in out:
            if k not in strata_held:
                raise ValueError(f"the strata do not all hold {k!r}; they hold "
                                 f"{sorted(strata_held)}")
        return out
    held = set(layer.names)
    out = {}
    for prop in strata_held:
        if prop in held:
            out[prop] = (prop,)
        elif (prop in PAIRS and all(n in held for n in PAIRS[prop])
              and not any(n in strata_held for n in PAIRS[prop])):
            out[prop] = PAIRS[prop]
    return out


def _held_by_all(strata: Strata, which: Sequence[int]) -> list[str]:
    """The properties every stratum of `which` holds."""
    sets = [set(strata.strata[k].properties) for k in which]
    return sorted(set.intersection(*sets)) if sets else []


def _attached(model: Model, index: int, name: str, breaks: np.ndarray,
              grid: AngularGrid, values: np.ndarray, poles: np.ndarray,
              lateral: str) -> Model:
    """The model with the field of `name` on layer `index` replaced by the
    SI values (nelem, 2, ntheta, nphi) and poles (nelem, 2, 2) in its units."""
    spec = model.spec(name)
    k = model.scales.factor(spec.dimensions)
    a, b = model.layer(index).interval
    field = GriddedField((a, b), breaks, grid, values / k, character=spec.character,
                         name=name, radial="linear", lateral=lateral, poles=poles / k)
    return model.with_field(index, name, field, replace=True)


def _as_geometry(model: Model, strata: Strata,
                 interfaces: Mapping[int | str, SurfaceOf],
                 properties: Mapping[str, str | Sequence[str]] | None,
                 take: Mapping[int | str, str] | None, grid: AngularGrid | None,
                 lateral: str | None) -> Model:
    moves = _moves(strata, model, interfaces)
    shifts = reliefs(strata, model, interfaces=interfaces, grid=grid,
                     lateral="cubic" if lateral is None else lateral)
    grid = next(iter(shifts.values())).grid
    lateral = _own_lateral(strata)
    take = {model.layer(k).index: v for k, v in (take or {}).items()}
    groups = []
    for (key0, f0, k0), (key1, f1, k1) in zip(moves[:-1], moves[1:]):
        below = model.geometry.interface(f0).between[1]
        above = model.geometry.interface(f1).between[0]
        if above != below:
            inner = [model.geometry.interface(f).name or f for f in range(f0 + 1, f1)]
            raise ValueError(
                f"between {key0!r} and {key1!r} the model has {above - below + 1} "
                f"layers; the strata between two moved interfaces fill one layer, "
                f"so move the interfaces {inner} too or remove them first "
                "(model.coarsened(drop=...))")
        groups.append((below, list(range(k1, k0))))
    for index in take:
        if index not in [g for g, _ in groups]:
            raise ValueError(f"take= names layer {index}, which lies between no two "
                             "moved interfaces")
    out = model.stretched(layer_linear(model.skeleton, _stack(
        model, {face: shifts[key] for key, face, _ in moves})), name="strata")
    radii = [_on_nodes(s.radius, grid) for s in strata.surfaces]
    for index, which in groups:
        held = _held_by_all(strata, which)
        lands = _lands(model.layer(index), properties, held)
        weights = [radii[k][0] - radii[k + 1][0] for k in which]
        weights_poles = [radii[k][1] - radii[k + 1][1] for k in which]
        if index in take:
            name = take[index]
            if strata[name].name not in [strata.strata[k].name for k in which]:
                raise ValueError(f"take= gives layer {index} the stratum {name!r}, "
                                 "which lies outside its strata")
        a, b = model.layer(index).interval
        for prop, names in lands.items():
            means = []
            for k in which:
                top, bottom = strata.strata[k].properties[prop]
                t, tp = _on_nodes(top, grid)
                u, up = _on_nodes(bottom, grid)
                means.append((0.5 * (t + u), 0.5 * (tp + up)))
            if index in take:
                at = [strata.strata[k].name for k in which].index(strata[take[index]]
                                                                   .name)
                nodes, poles = means[at]
            else:
                total = sum(weights)
                nodes = sum(w * m for w, (m, _) in zip(weights, means)) / total
                total = sum(weights_poles)
                poles = sum(w * m for w, (_, m) in zip(weights_poles, means)) / total
            values = np.broadcast_to(nodes, (1, 2) + nodes.shape)
            pole_values = np.broadcast_to(poles, (1, 2, 2))
            for name in names:
                out = _attached(out, index, name, np.array([a, b]), grid, values,
                                pole_values, lateral)
    return out


def _as_fields(model: Model, strata: Strata,
               properties: Mapping[str, str | Sequence[str]] | None,
               layers: Sequence[int | str] | None, grid: AngularGrid | None,
               lateral: str | None, spacing: float | None) -> Model:
    if spacing is None or not spacing > 0.0:
        raise ValueError("the fields route samples at radial elements spacing= apart "
                         "(a length in the model's units)")
    grid = _gridded_on(strata.surfaces, grid)
    lateral = _own_lateral(strata) if lateral is None else lateral
    length = model.scales.length
    radii = [_on_nodes(s.radius, grid) for s in strata.surfaces]
    if layers is None:
        lo = min(float(np.min(radii[-1][0])), float(np.min(radii[-1][1]))) / length
        hi = max(float(np.max(radii[0][0])), float(np.max(radii[0][1]))) / length
        indices = [lay.index for lay in model.layers
                   if lay.interval[1] > lo and lay.interval[0] < hi]
    else:
        indices = sorted({model.layer(k).index for k in layers})
    held = _held_by_all(strata, range(len(strata)))
    surf = np.stack([r for r, _ in radii])                        # (n + 1, nt, np)
    surf_poles = np.stack([p for _, p in radii])                  # (n + 1, 2)
    tops, bottoms = {}, {}
    for prop in held:
        pairs = [(_on_nodes(s.properties[prop][0], grid),
                  _on_nodes(s.properties[prop][1], grid)) for s in strata.strata]
        tops[prop] = (np.stack([t for (t, _), _ in pairs]),
                      np.stack([tp for (_, tp), _ in pairs]))
        bottoms[prop] = (np.stack([u for _, (u, _) in pairs]),
                         np.stack([up for _, (_, up) in pairs]))
    below = {prop: _on_nodes(fn, grid) for prop, fn in strata.below.items()}
    theta = grid.colatitudes[None, None, :, None]
    phi = grid.longitudes[None, None, None, :]

    def pick(r: np.ndarray, R: np.ndarray, top: np.ndarray, bottom: np.ndarray,
             under: np.ndarray | None, own: np.ndarray) -> np.ndarray:
        """Values at radii r (metres) over the points of R (n + 1, ...)."""
        out = np.array(own, dtype=float)
        for k in range(R.shape[0] - 1):
            inside = (r >= R[k + 1]) & (r < R[k])
            with np.errstate(divide="ignore", invalid="ignore"):
                s = np.where(inside, (R[k] - r) / (R[k] - R[k + 1]), 0.0)
            out = np.where(inside, top[k] + s * (bottom[k] - top[k]), out)
        if under is not None:
            out = np.where(r < R[-1], under, out)
        return out

    out = model
    for index in indices:
        layer = model.layer(index)
        lands = _lands(layer, properties, held)
        a, b = layer.interval
        n = max(1, int(np.ceil((b - a) / spacing - 1e-9)))
        breaks = np.linspace(a, b, n + 1)
        r = np.stack([breaks[:-1], breaks[1:]], axis=1)              # (nelem, 2)
        for prop, names in lands.items():
            for name in names:
                if name not in layer:
                    raise ValueError(f"layer {index} holds no {name!r} to replace")
                k = model.scales.factor(model.spec(name).dimensions)
                field = layer[name]
                own = k * np.broadcast_to(np.asarray(field.evaluate(
                    r[:, :, None, None], theta, phi), dtype=float),
                    r.shape + (grid.ntheta, grid.nphi))
                own_poles = k * np.broadcast_to(np.asarray(field.evaluate(
                    r[:, :, None], np.array([0.0, np.pi]), np.zeros(2),
                    frame="cartesian"), dtype=float), r.shape + (2,))
                under = below.get(prop)
                values = pick(length * r[:, :, None, None], surf[:, None, None],
                              tops[prop][0][:, None, None],
                              bottoms[prop][0][:, None, None],
                              None if under is None else under[0], own)
                poles = pick(length * r[:, :, None], surf_poles[:, None, None],
                             tops[prop][1][:, None, None],
                             bottoms[prop][1][:, None, None],
                             None if under is None else under[1], own_poles)
                out = _attached(out, index, name, breaks, grid, values, poles, lateral)
    return out


def with_strata(model: Model, strata: Strata, *, route: str = "geometry",
                interfaces: Mapping[int | str, SurfaceOf] | None = None,
                properties: Mapping[str, str | Sequence[str]] | None = None,
                take: Mapping[int | str, str] | None = None,
                layers: Sequence[int | str] | None = None,
                grid: AngularGrid | None = None, lateral: str | None = None,
                spacing: float | None = None) -> Model:
    """The model with the strata on it by `route`, one of `ROUTES`, by the
    rules of the module docstring: as geometry, `interfaces` moved to their
    surfaces and the layers between them filled (`take` a layer's one
    stratum), the reliefs read by `lateral` (bicubic by default) and the
    fields by the strata's own rule; as fields, `layers` sampled at radial
    elements `spacing` apart, read by `lateral` (the strata's own rule by
    default); on `grid` (the surfaces' own by default), `properties`
    mapping a property onto the model's names."""
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}, got {route!r}")
    if route == "geometry":
        for word, value in (("layers", layers), ("spacing", spacing)):
            if value is not None:
                raise ValueError(f"{word}= belongs to the fields route; the geometry "
                                 "route fills the layers between moved interfaces")
        if interfaces is None:
            raise ValueError("the geometry route needs interfaces=, each interface "
                             "of the model and the surface of the strata it moves to")
        out = _as_geometry(model, strata, interfaces, properties, take, grid, lateral)
    else:
        for word, value in (("interfaces", interfaces), ("take", take)):
            if value is not None:
                raise ValueError(f"{word}= belongs to the geometry route; the fields "
                                 "route keeps the skeleton spherical")
        out = _as_fields(model, strata, properties, layers, grid, lateral, spacing)
    return out.with_citation(*strata.citation)


def moved(model: Model, surfaces: Mapping[int | str, Surface], *,
          grid: AngularGrid | None = None, lateral: str = "cubic") -> Model:
    """The model with each interface named moved onto its surface: the
    model stretched by `layer_linear` over their reliefs (`relief`), on
    one grid (`grid`, or the surfaces' own) read by `lateral`, refused
    where a layer would be taken to zero or negative thickness, as
    `reliefs` refuses, and citing the surfaces after its own works."""
    if not surfaces:
        raise ValueError("surfaces names at least one interface and its surface")
    grid = _gridded_on(list(surfaces.values()), grid)
    shifts = {model.geometry.interface(k).index: relief(f, model, k, grid=grid,
                                                        lateral=lateral)
              for k, f in surfaces.items()}
    stack = _stack(model, shifts)
    _check_thickness(model, stack, grid)
    out = model.stretched(layer_linear(model.skeleton, stack), name="moved")
    return out.with_citation(*[w for f in surfaces.values() for w in f.citation])
