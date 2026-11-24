"""The stratification of a spherically symmetric model: the Bullen
parameter and the buoyancy frequency.

A layer in hydrostatic equilibrium is neutrally stratified where its
density follows the adiabat, d rho / d r = -rho^2 g / kappa, and stably
stratified where it falls with radius faster than that.  The Bullen
parameter states which,

    eta(r) = -kappa rho'(r) / (rho^2 g),

one on the adiabat, greater where the layer is stable and less where it
is unstable; its excess over one is the name `bullen_excess` of the
vocabulary.  The square of the buoyancy (Brunt-Vaisala) frequency is the
same statement with the dimensions of a frequency squared,

    N^2(r) = -g (rho' / rho + rho g / kappa) = (rho g^2 / kappa) (eta - 1),

and vanishes with g at the centre whatever the stratification, which is
why the Bullen parameter and not N^2 is what describes a layer there.

Everything is read off the fields: rho and its radial derivative, the
bulk modulus as `materials.kappa_mu` has it, and the gravity of
`mesh1d.gravity`.  The derivative is exact where the density is a
polynomial.  Where g vanishes the Bullen parameter is the limit of the
ratio: where rho' vanishes too (the centre of a density that is smooth
there) it is -kappa rho'' / (rho^2 g'), and elsewhere it is infinite,
with the sign of -rho'.  A radius on an interior boundary is answered
from the layer above it, and the parameter jumps there with the fields.
Nothing here is a number with a unit.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from .. import materials
from .gravity import _within, gravity, gravity_fields

if TYPE_CHECKING:
    from ..model import Model

__all__ = ["bullen_parameter", "bullen_excess", "brunt_vaisala_squared"]


def _on_layers(model: Model, radii: ArrayLike
               ) -> tuple[np.ndarray, list[tuple[np.ndarray, int]]]:
    """The radii held to the model, flattened, and for each layer that
    holds some of them the mask of those it holds and its index."""
    r = _within(model, np.asarray(radii, dtype=float))
    flat = r.reshape(-1)
    b = model.skeleton.boundaries
    idx = np.clip(np.searchsorted(b, flat, side="right") - 1, 0, model.nlayers - 1)
    return r, [(idx == i, int(i)) for i in np.unique(idx)]


def _density_and_modulus(model: Model, i: int, r: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """rho, rho', rho'' and kappa of layer `i` at `r`."""
    layer = model.layers[i]
    rho = layer["rho"]
    if not hasattr(rho, "derivative"):
        raise ValueError(
            f"layer {layer.index} ({layer.name!r}) holds a rho that cannot "
            "state its radial derivative, so its stratification cannot be "
            "formed")
    kappa = materials.kappa_mu(layer)[0]
    return (np.asarray(rho(r), dtype=float),
            np.asarray(rho.derivative(nu=1)(r), dtype=float),
            np.asarray(rho.derivative(nu=2)(r), dtype=float),
            np.asarray(kappa(r), dtype=float))


def bullen_parameter(model: Model, radii: ArrayLike) -> np.ndarray:
    """The Bullen parameter eta(r) = -kappa rho' / (rho^2 g) at `radii`.

    `radii` broadcast to any shape and must lie in the model, to the
    geometry's tolerance.  One where the layer is neutrally stratified,
    greater where it is stable, less where it is unstable.  Where g
    vanishes the value is the limit of the ratio (the module docstring):
    finite at the centre of a density that is smooth there, infinite
    where rho' does not vanish with g.
    """
    r, layers = _on_layers(model, radii)
    flat = r.reshape(-1)
    g = gravity(model, flat)
    fields = None
    out = np.empty(flat.shape)
    for mask, i in layers:
        x = flat[mask]
        rho, d1, d2, kappa = _density_and_modulus(model, i, x)
        gx = g[mask]
        still = gx == 0.0
        eta = -kappa * d1 / (rho ** 2 * np.where(still, 1.0, gx))
        if still.any():
            if fields is None:
                fields = gravity_fields(model)
            dg = np.asarray(fields[i].derivative(nu=1)(x), dtype=float)
            flat_density = d1 == 0.0
            limit = -kappa * d2 / (rho ** 2 * np.where(dg != 0.0, dg, 1.0))
            unbounded = np.where(d1 < 0.0, np.inf, -np.inf)
            eta = np.where(still, np.where(flat_density, limit, unbounded), eta)
        out[mask] = eta
    return out.reshape(r.shape)


def bullen_excess(model: Model, radii: ArrayLike) -> np.ndarray:
    """The excess of the Bullen parameter over one at `radii`: zero where
    the layer is neutrally stratified, positive where it is stable and
    negative where it is unstable (`bullen_parameter`)."""
    return bullen_parameter(model, radii) - 1.0


def brunt_vaisala_squared(model: Model, radii: ArrayLike) -> np.ndarray:
    """The square of the buoyancy frequency,
    N^2(r) = -g (rho' / rho + rho g / kappa), at `radii`.

    `radii` broadcast to any shape and must lie in the model.  Positive
    where the layer is stably stratified and negative where it is
    unstable; zero where g vanishes, the centre among them.
    """
    r, layers = _on_layers(model, radii)
    flat = r.reshape(-1)
    g = gravity(model, flat)
    out = np.empty(flat.shape)
    for mask, i in layers:
        rho, d1, _, kappa = _density_and_modulus(model, i, flat[mask])
        out[mask] = -g[mask] * (d1 / rho + rho * g[mask] / kappa)
    return out.reshape(r.shape)
