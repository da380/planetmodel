"""Gravity and mass of a spherically symmetric model, through its fields.

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
hollow model has no mass inside its inner boundary.
"""
from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..fields import Field, RadialField
from ..layerfunction import NumericLayer, polynomial_layer

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["gravity", "mass", "gravity_fields"]


def _shell_masses(model: Model) -> tuple[list[Field], np.ndarray]:
    """The mass of every layer and the `rho s^2` field of each."""
    fields, masses = [], []
    for layer in model.layers:
        if "rho" not in layer:
            raise KeyError(
                f"gravity needs 'rho' on every layer; layer {layer.index} "
                f"({layer.name!r}) holds {list(layer.names)}")
        if not getattr(layer["rho"], "is_radial", False):
            raise ValueError(
                f"'rho' on layer {layer.index} ({layer.name!r}) depends on "
                "direction; gravity here is that of a spherically symmetric "
                "density and reads radial fields only")
        lo, hi = layer.interval
        s2 = RadialField((lo, hi), polynomial_layer([0.0, 0.0, 1.0], (lo, hi)))
        f = layer["rho"] * s2
        fields.append(f)
        masses.append(4.0 * np.pi * float(f.integrate(lo, hi)))
    return fields, np.asarray(masses)


def mass(model: Model, *, radius: float | None = None) -> float:
    """The mass inside `radius`, the outer boundary by default."""
    b = model.skeleton.boundaries
    r = float(b[-1]) if radius is None else float(radius)
    if not b[0] <= r <= b[-1]:
        raise ValueError(f"radius {r:g} is outside the model [{b[0]:g}, {b[-1]:g}]")
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

    `radii` broadcast to any shape and must lie in the model; g is zero
    at the centre and on the inner boundary of a hollow model.
    """
    r = np.asarray(radii, dtype=float)
    b = model.skeleton.boundaries
    if r.size and (np.any(r < b[0]) or np.any(r > b[-1])):
        raise ValueError(f"radii must lie in the model [{b[0]:g}, {b[-1]:g}]")
    fields, masses = _shell_masses(model)
    below = np.concatenate(([0.0], np.cumsum(masses)))
    flat = r.reshape(-1)
    M = np.empty(flat.shape)
    idx = np.clip(np.searchsorted(b, flat, side="right") - 1, 0, len(fields) - 1)
    for i in np.unique(idx):
        m = idx == i
        lo, hi = fields[i].interval
        part = np.array([fields[i].integrate(lo, min(x, hi)) for x in flat[m]])
        M[m] = below[i] + 4.0 * np.pi * part
    g = np.zeros(flat.shape)
    positive = flat > 0.0
    g[positive] = model.G * M[positive] / flat[positive] ** 2
    return g.reshape(r.shape)


def _cumulative_mass(f: Field, lo: float) -> Callable[[np.ndarray], np.ndarray]:
    """4 pi times the integral of `f` from `lo` to each of an array of
    radii: through the antiderivative where the layer function is a
    polynomial, else by quadrature point by point."""
    fn = f.function
    ppoly = getattr(fn, "ppoly", None)
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

        out.append(RadialField((lo, hi), NumericLayer(g_of, (lo, hi), derivative=dg_dr),
                               name="g"))
    return tuple(out)
