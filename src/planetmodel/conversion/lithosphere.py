"""The base of the lithosphere, found in a model's fields.

The lithosphere is what lies above the first radius, going down a
radial line from the top of the chosen layers, at which a field crosses
to the hot, weak or fast-relaxing side of a threshold:

- `isotherm(model, temperature)`: the absolute temperature rises to the
  isotherm (1300 C, say, 1573 K);
- `viscosity_threshold(model, viscosity)`: the viscosity falls to the
  threshold (1e23 Pa s, say);
- `maxwell_threshold(model, loading_time)`: the Maxwell time eta / mu,
  with mu the layer's Voigt shear modulus, falls to the loading time, so
  that above the base the rock answers a load of that time scale
  elastically.

Each samples the field at radii no more than `step` apart through each
chosen layer from its top down, on the nodes of `grid` and at both
poles, takes the first sample on the far side of the threshold, and
bisects between it and the one above to `rtol` of the radius; a crossing
at a boundary between layers, where the field may jump, is the boundary.
A line hot already at the top has its base at the top; one that never
crosses has it at the bottom of the chosen layers.  The thresholds are
in SI, the field's values converted at the boundary.  The result is a
`datasets.Surface` measuring a radius, read bilinearly between the
grid's nodes, for `datasets.relief` and `datasets.building.moved` (the
base as geometry) or `with_lid` (the base as a field: the viscosity
above it raised to a lid value, kept a log perturbation of the layer's
radial viscosity, which a file stores and a code reads as it reads any
viscosity; a fully 3D viscosity, which has no radial reference, becomes
a field of its own, the lid above and the viscosity below).  Radial
lines, an angular grid and a surface measuring a radius are the
sphere's: a flat model is refused by name.
"""
from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import TYPE_CHECKING

import numpy as np

from ..coordinates import SPHERICAL, require_coordinates
from ..fields import Field
from ..gridded import GriddedFunction
from ..materials import kappa_mu
from ..model import Model
from ..perturbation import PerturbedField
from ..units import TEMPERATURE, TIME, VISCOSITY

if TYPE_CHECKING:
    from ..datasets.surfaces import Surface
    from ..sampling import AngularGrid

__all__ = ["isotherm", "viscosity_threshold", "maxwell_threshold", "with_lid"]

#: How a layer's field is read: values at (r, theta, phi) in SI.
type Reading = Callable[[int, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _layers(model: Model, layers: Iterable[int | str] | None, holds: str
            ) -> list[int]:
    """The chosen layers from the outside in, by default those holding
    `holds`."""
    which = ([lay.index for lay in model.layers if holds in lay] if layers is None
             else sorted({model.layer(k).index for k in layers}))
    if not which:
        raise ValueError(f"no layer holds {holds!r} to find a base in")
    return which[::-1]


def _base(model: Model, read: Reading, threshold: float, hot_above: bool,
          which: list[int], grid: AngularGrid, step: float, rtol: float,
          name: str) -> Surface:
    """The base by the rule of the module docstring, as a Surface; a model
    off the sphere is refused."""
    from ..datasets.surfaces import Surface
    require_coordinates(model, SPHERICAL, "the base of the lithosphere on radial lines")
    length = model.scales.length
    theta = np.concatenate([np.repeat(grid.colatitudes, grid.nphi), [0.0, np.pi]])
    phi = np.concatenate([np.tile(grid.longitudes, grid.ntheta), [0.0, 0.0]])

    def hot(values: np.ndarray) -> np.ndarray:
        return values >= threshold if hot_above else values <= threshold

    # a line never crossing has its base at the bottom of the chosen layers
    base = np.full(theta.shape, model.layer(which[-1]).interval[0])
    open_ = np.ones(theta.shape, dtype=bool)
    for i in which:
        if not open_.any():
            break
        lo, hi = model.layer(i).interval
        n = max(2, int(np.ceil((hi - lo) * length / step)) + 1)
        radii = np.linspace(hi, lo, n)
        k = np.flatnonzero(open_)
        found = hot(np.stack([read(i, np.full(k.size, r), theta[k], phi[k])
                              for r in radii]))
        crossed = found.any(axis=0)
        first = found.argmax(axis=0)
        # hot at the layer's top: the base is that top (a boundary, or the
        # top of the chosen layers)
        at_top = crossed & (first == 0)
        base[k[at_top]] = hi
        inside = crossed & (first > 0)
        if inside.any():
            m, f = k[inside], first[inside]
            a, b = radii[f - 1], radii[f]                  # a cold, b hot
            while np.any(a - b > rtol * hi):
                mid = 0.5 * (a + b)
                h = hot(read(i, mid, theta[m], phi[m]))
                b, a = np.where(h, mid, b), np.where(h, a, mid)
            base[m] = 0.5 * (a + b)
        open_[k[crossed]] = False
    values = base * length
    nodes = values[:-2].reshape(grid.ntheta, grid.nphi)
    return Surface(GriddedFunction(grid, nodes, lateral="linear", poles=values[-2:]),
                   "radius", name=name)


def _reading(model: Model, name: str) -> Reading:
    """A layer's field `name` at points, in SI."""
    k = model.scales.factor(model.spec(name).dimensions)

    def read(i: int, r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return k * np.asarray(model.layer(i)[name](r, t, p), dtype=float)

    return read


def isotherm(model: Model, temperature: float, *, grid: AngularGrid,
             layers: Iterable[int | str] | None = None, name: str = "temperature",
             step: float = 2e3, rtol: float = 1e-10) -> Surface:
    """The base where the field `name`, an absolute temperature, rises to
    `temperature` (K), by the rule of the module docstring."""
    if model.spec(name) is None or model.spec(name).dimensions != TEMPERATURE:
        raise ValueError(f"{name!r} is not a temperature in this model")
    return _base(model, _reading(model, name), float(temperature), True,
                 _layers(model, layers, name), grid, step, rtol,
                 f"the {temperature:g} K isotherm")


def viscosity_threshold(model: Model, viscosity: float, *, grid: AngularGrid,
                        layers: Iterable[int | str] | None = None,
                        name: str = "viscosity", step: float = 2e3,
                        rtol: float = 1e-10) -> Surface:
    """The base where the viscosity falls to `viscosity` (Pa s)."""
    if model.spec(name) is None or model.spec(name).dimensions != VISCOSITY:
        raise ValueError(f"{name!r} is not a viscosity in this model")
    return _base(model, _reading(model, name), float(viscosity), False,
                 _layers(model, layers, name), grid, step, rtol,
                 f"viscosity {viscosity:g} Pa s")


def maxwell_threshold(model: Model, loading_time: float, *, grid: AngularGrid,
                      layers: Iterable[int | str] | None = None,
                      name: str = "viscosity", step: float = 2e3,
                      rtol: float = 1e-10) -> Surface:
    """The base where the Maxwell time eta / mu falls to `loading_time`
    (s), mu the layer's Voigt shear modulus."""
    kt = model.scales.factor(TIME)
    mus: dict[int, Field] = {}

    def read(i: int, r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        lay = model.layer(i)
        if i not in mus:
            mus[i] = kappa_mu(lay)[1]
        eta = np.asarray(lay[name](r, t, p), dtype=float)
        mu = np.asarray(mus[i](r, t, p), dtype=float)
        return kt * eta / mu                  # the model's time, in seconds

    return _base(model, read, float(loading_time), False,
                 _layers(model, layers, name), grid, step, rtol,
                 f"Maxwell time {loading_time:g} s")


def with_lid(model: Model, base: Surface, *, viscosity: float,
             name: str = "viscosity") -> Model:
    """The model with the viscosity of every layer holding it raised to
    `viscosity` (Pa s) above `base`, a log perturbation of the layer's
    radial viscosity (or of a perturbed one's reference) or, for a fully
    3D one, a field of its own, by the rule of the module docstring; a
    layer wholly below the base untouched; a model off the sphere is
    refused, the base being a surface measuring a radius."""
    require_coordinates(model, SPHERICAL, "a lid below a surface measuring a radius")
    length = model.scales.length
    lid = float(viscosity) / model.scales.factor(VISCOSITY)
    f = base.function
    lowest = min(float(np.min(f.values)), float(np.min(f.poles)))
    out = model
    for lay in model.layers:
        lo, hi = lay.interval
        if name not in lay or hi * length <= lowest:
            continue
        out = out.with_field(lay.index, name, _lidded(lay[name], base, lid, length,
                                                       (lo, hi), name), replace=True)
    return out.with_citation(*base.citation)


def _lidded(f: Field, base: Surface, lid: float, length: float,
            interval: tuple[float, float], name: str) -> Field:
    """The field below `base` and the lid above it: a log perturbation of
    the field's radial reference where it has one."""
    from ..fields import AnalyticField
    ref = getattr(f, "reference", None)
    ref = f if ref is None else ref
    if not getattr(ref, "is_radial", False):
        def whole(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
            above = np.asarray(r) * length > base.radius(t, p)
            return np.where(above, lid, np.asarray(f(r, t, p), dtype=float))
        return AnalyticField(interval, whole, name=name)
    pert, kind = getattr(f, "perturbation", None), getattr(f, "kind", None)

    def dln(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        r0 = np.asarray(ref(r), dtype=float)
        if pert is None:
            own = np.zeros(np.broadcast(r, t, p).shape)
        elif kind == "log":
            own = np.asarray(pert(r, t, p), dtype=float)
        elif kind == "relative":
            own = np.log1p(np.asarray(pert(r, t, p), dtype=float))
        else:
            own = np.log1p(np.asarray(pert(r, t, p), dtype=float) / r0)
        above = np.asarray(r) * length > base.radius(t, p)
        return np.where(above, np.log(lid / r0), own)

    return PerturbedField(ref, AnalyticField(interval, dln, name=name), kind="log",
                          name=name)
