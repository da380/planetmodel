"""Gravity, mass and pressure of a spherically symmetric model, through
its fields.

For a spherically symmetric density the acceleration is
g(r) = G M(r) / r^2 with M(r) = 4 pi int_0^r rho(s) s^2 ds, the mass
inside r accumulated layer by layer.  Each layer's integral is asked of
the field algebra: `rho * s^2` is a field whose layer function
integrates itself, exactly when rho is a polynomial and by quadrature
otherwise.  This is the gravity of the reference body: rho must be a
radial field on every layer, a density that depends on direction is
refused by name, and the geometry's mapping does not enter, so a model
placed in the physical world by a non-trivial mapping is answered with
the gravity of its spherical reference.  Nothing here is a number with
a unit: `G` is the model's, the radii are in the model's length, and a
hollow model has no mass inside its inner boundary.  Gravity from the
mass inside a radius is defined on the sphere alone: a model of another
system (a flat one, `planetmodel.cartesian`) is refused by name, a flat
model's gravity being prescribed, not computed.

The hydrostatic pressure of the same body is P(r) = int_r^a rho(s) g(s)
ds, zero at its outer radius a and continuous across every boundary.
rho g is smooth on each layer (a polynomial times G M(s) / s^2, with
M(s) a polynomial on the layer), so each layer's integral is taken by
Gauss-Legendre quadrature of `PRESSURE_NODES` points on the span asked
for, which is exact to rounding for a polynomial density; dP/dr = -rho g
is stated exactly.  A pressure is a stress, so its field has a density's
character, as the moduli and the equilibrium stress have.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..character import DENSITY
from ..coordinates import SPHERICAL, require_coordinates
from ..fields import Field, RadialField
from ..layerfunction import NumericLayer, polynomial_layer

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["gravity", "mass", "layer_masses", "gravity_fields", "pressure",
           "pressure_fields", "radii_within", "PRESSURE_NODES"]

#: The Gauss-Legendre points of each pressure integral.
PRESSURE_NODES = 24


def _shell_masses(model: Model) -> tuple[list[Field], np.ndarray]:
    """The mass of every layer and the `rho s^2` field of each, a field
    that integrates itself; a model off the sphere is refused.

    Where the product of the density with s^2 cannot integrate itself,
    the density being radial without being a function of the radius the
    algebra combines (a composition of radial fields, say), the density
    is read as a callable of the radius and the product integrates by
    quadrature."""
    _require_sphere(model)
    fields, masses = [], []
    for layer in model.layers:
        if "rho" not in layer:
            raise KeyError(
                f"gravity needs 'rho' on every layer; layer {layer.index} "
                f"({layer.name!r}) holds {list(layer.names)}")
        rho = layer["rho"]
        if not getattr(rho, "is_radial", False):
            raise ValueError(
                f"'rho' on layer {layer.index} ({layer.name!r}) depends on "
                "direction; gravity here is that of a spherically symmetric "
                "density and reads radial fields only")
        lo, hi = layer.interval
        s2 = RadialField((lo, hi), polynomial_layer((lo, hi), [0.0, 0.0, 1.0]))
        f = rho * s2
        if not hasattr(f, "integrate"):
            f = _of_radius(rho) * s2
        fields.append(f)
        masses.append(4.0 * np.pi * float(f.integrate(lo, hi)))
    return fields, np.asarray(masses)


def _of_radius(field: Field) -> RadialField:
    """A radial field of rank 0 as a `RadialField` over a callable of the
    radius: the field evaluated in one direction, every direction giving
    the same."""
    return RadialField(
        field.interval,
        NumericLayer(field.interval, lambda r: field.evaluate(r, 0.0, 0.0)),
        character=field.character, name=field.name)


def _require_sphere(model: Model) -> None:
    """Refuse a model whose coordinates are not the sphere's, by name."""
    require_coordinates(model, SPHERICAL, "gravity from the mass inside a radius")


def radii_within(model: Model, r: np.ndarray) -> np.ndarray:
    """The radii `r` as radii of `model`, of the same shape: each held to
    the model's range to the geometry's tolerance, `rtol` of that range,
    and clipped onto it.  A radius beyond the tolerance, or NaN, is
    refused, as is a model off the sphere."""
    _require_sphere(model)
    b = model.skeleton.boundaries
    tol = model.geometry.rtol * (b[-1] - b[0])
    if r.size and (np.isnan(r).any() or np.any(r < b[0] - tol)
                   or np.any(r > b[-1] + tol)):
        raise ValueError(f"radii must lie in the model [{b[0]:g}, {b[-1]:g}]")
    return np.clip(r, b[0], b[-1])


def layer_masses(model: Model) -> np.ndarray:
    """The mass of every layer, from the centre outward."""
    return _shell_masses(model)[1]


def mass(model: Model, *, radius: float | None = None) -> float:
    """The mass inside `radius`, the outer boundary by default."""
    b = model.skeleton.boundaries
    r = float(b[-1]) if radius is None else float(radius)
    r = float(radii_within(model, np.array(r)))
    fields, masses = _shell_masses(model)
    total = 0.0
    for i, f in enumerate(fields):
        lo, hi = f.interval
        if r >= hi:
            total += masses[i]
        elif r > lo:
            total += 4.0 * np.pi * float(f.integrate(lo, r))
    return total


def gravity(model: Model, radii: ArrayLike) -> np.ndarray:
    """g(r) = G M(r) / r^2 at `radii`, in the model's units.

    `radii` broadcast to any shape and must lie in the model, to the
    geometry's tolerance; g is zero at the centre and on the inner
    boundary of a hollow model.  M(r) is the antiderivative of each
    layer's `rho s^2`, as `gravity_fields` has it.
    """
    r = radii_within(model, np.asarray(radii, dtype=float))
    b = model.skeleton.boundaries
    fields, masses = _shell_masses(model)
    below = np.concatenate(([0.0], np.cumsum(masses)))
    flat = r.reshape(-1)
    M = np.empty(flat.shape)
    idx = np.clip(np.searchsorted(b, flat, side="right") - 1, 0, len(fields) - 1)
    for i in np.unique(idx):
        m = idx == i
        lo, hi = fields[i].interval
        M[m] = below[i] + _cumulative_mass(fields[i], lo)(np.minimum(flat[m], hi))
    g = np.zeros(flat.shape)
    positive = flat > 0.0
    g[positive] = model.G * M[positive] / flat[positive] ** 2
    return g.reshape(r.shape)


def _cumulative_mass(f: Field, lo: float) -> Callable[[np.ndarray], np.ndarray]:
    """4 pi times the integral of `f` from `lo` to each of an array of
    radii: through the antiderivative where `f` holds one layer function
    that is a polynomial, else by `f.integrate` point by point."""
    ppoly = getattr(getattr(f, "function", None), "ppoly", None)
    if ppoly is not None:
        P = ppoly.antiderivative()
        base = P(lo)
        return lambda r: 4.0 * np.pi * (P(r) - base)
    return lambda r: 4.0 * np.pi * np.array([f.integrate(lo, float(x)) for x in
                                             np.asarray(r, dtype=float).ravel()]
                                            ).reshape(np.shape(r))


def gravity_fields(model: Model) -> tuple[RadialField, ...]:
    """g(r) = G M(r) / r^2 as one radial field per layer, named `g`.

    Each field is exact in value and in its first derivative where the
    layer's density is polynomial: M(r) is the polynomial antiderivative
    of 4 pi rho r^2 plus the mass below the layer, and
    dg/dr = G (4 pi rho - 2 M / r^3).  The fields are continuous across
    every boundary and vanish at the centre; a model attaches them by
    `with_field(i, "g", field)`.
    """
    fields, masses = _shell_masses(model)
    below = np.concatenate(([0.0], np.cumsum(masses)))
    G = model.G
    out = []
    for i, (layer, f) in enumerate(zip(model.layers, fields)):
        lo, hi = layer.interval
        cumulative = _cumulative_mass(f, lo)
        mass_below = float(below[i])
        rho = layer["rho"]

        def g_of(r: np.ndarray, *, cumulative: Callable[[np.ndarray], np.ndarray]
                 = cumulative, mass_below: float = mass_below) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            M = mass_below + cumulative(r)
            safe = np.where(r > 0.0, r, 1.0)
            return np.where(r > 0.0, G * M / safe ** 2, 0.0)

        def dg_dr(r: np.ndarray, *, cumulative: Callable[[np.ndarray], np.ndarray]
                  = cumulative, mass_below: float = mass_below,
                  rho: Field = rho) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            M = mass_below + cumulative(r)
            safe = np.where(r > 0.0, r, 1.0)
            # g -> (4 pi G rho(0) / 3) r as r -> 0, so dg/dr -> 4 pi G rho(0) / 3
            centre = 4.0 * np.pi * G * rho(r) / 3.0
            return np.where(r > 0.0, G * (4.0 * np.pi * rho(r) - 2.0 * M / safe ** 3),
                            centre)

        out.append(RadialField((lo, hi), NumericLayer((lo, hi), g_of, derivative=dg_dr),
                               name="g"))
    return tuple(out)


def pressure_fields(model: Model) -> tuple[RadialField, ...]:
    """P(r) = int_r^a rho g ds as one radial field per layer, named
    `pressure`, by the rule of the module docstring: zero at the outer
    boundary, continuous across every other, its derivative -rho g."""
    gs = gravity_fields(model)
    x, w = np.polynomial.legendre.leggauss(PRESSURE_NODES)
    n = len(gs)

    def integral(i: int, r: np.ndarray, hi: float) -> np.ndarray:
        """int_r^hi rho g ds on layer i, for an array of radii."""
        rho, g = model.layer(i)["rho"], gs[i]
        half = 0.5 * (hi - r)[..., None]
        s = r[..., None] + half * (x + 1.0)
        return np.sum(w * np.asarray(rho(s), dtype=float)
                      * np.asarray(g(s), dtype=float), axis=-1) * half[..., 0]

    tops = np.zeros(n)
    for i in range(n - 1, 0, -1):
        lo, hi = model.layer(i).interval
        tops[i - 1] = tops[i] + float(integral(i, np.array(lo), hi))
    out = []
    for i in range(n):
        lo, hi = model.layer(i).interval
        rho, g = model.layer(i)["rho"], gs[i]

        def p_of(r: np.ndarray, *, i: int = i, hi: float = hi,
                 top: float = float(tops[i])) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            return top + integral(i, r, hi)

        def dp_dr(r: np.ndarray, *, rho: Field = rho, g: Field = g) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            return -np.asarray(rho(r), dtype=float) * np.asarray(g(r), dtype=float)

        out.append(RadialField((lo, hi), NumericLayer((lo, hi), p_of, derivative=dp_dr),
                               character=DENSITY, name="pressure"))
    return tuple(out)


def pressure(model: Model, radii: ArrayLike) -> np.ndarray:
    """The hydrostatic pressure at `radii`, in the model's units, by the
    rule of the module docstring; `radii` as `gravity` takes them, a
    radius on a boundary read from the layer below, where it is the same."""
    r = radii_within(model, np.asarray(radii, dtype=float))
    b = model.skeleton.boundaries
    fields = pressure_fields(model)
    flat = r.reshape(-1)
    idx = np.clip(np.searchsorted(b, flat, side="left") - 1, 0, len(fields) - 1)
    out = np.empty(flat.shape)
    for i in np.unique(idx):
        m = idx == i
        out[m] = fields[i](flat[m])
    return out.reshape(r.shape)
