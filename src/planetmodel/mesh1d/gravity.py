"""Gravity and mass of a spherically layered model, through its fields.

For a spherically symmetric density the acceleration is
g(r) = G M(r) / r^2 with M(r) = 4 pi int_0^r rho(s) s^2 ds, the mass
inside r accumulated layer by layer.  Each layer's integral is asked of
the field algebra: `rho * s^2` is a field whose layer function
integrates itself, exactly when rho is a polynomial and by quadrature
otherwise.  Nothing here is a number with a unit: `G` is the model's,
the radii are in the model's length, and a hollow model has no mass
inside its inner boundary.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..fields import Field, RadialField
from ..layerfunction import polynomial_layer

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["gravity", "mass"]


def _shell_masses(model: Model) -> tuple[list[Field], np.ndarray]:
    """The mass of every layer and the `rho s^2` field of each."""
    fields, masses = [], []
    for layer in model.layers:
        if "rho" not in layer:
            raise KeyError(
                f"gravity needs 'rho' on every layer; layer {layer.index} "
                f"({layer.name!r}) holds {list(layer.names)}")
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
