"""Spherical means: a field reduced to a function of radius.

The spherical mean of a scalar field f on its layer is

    fbar(r) = (1 / 4 pi) integral over S^2 of f(r, theta, phi) dOmega,

a `RadialField` of the field's character.  A field that can give it
exactly is asked (`spherical_mean`, a method): a radial field is its own
mean; a harmonic field's is c_00(r) / sqrt(4 pi), exact polynomial
pieces; a gridded field gives at each of its nodes the exact mean of
its lateral rule's interpolant, joined by its radial rule, the mean of
the field as its rules define it; a
perturbed field combines its parts, exactly from the perturbation's
mean when the perturbation is absolute or relative, and from the mean
of its exponential, by quadrature, when it is logarithmic.  A radial
field without a mean of its own is wrapped as one.  Any other field
needs `grid=`, and its mean is that grid's quadrature at each
radius, a `NumericLayer`.  Fields of rank 1 and above have no mean here.

`spherically_averaged(model)` puts the mean of every field that is not
radial in its place, through `with_field(replace=True)` in each layer's
order, so that a model type's own rule runs: on an `Elastic` model the
mean of a velocity re-derives the Love moduli, rather than each modulus
being averaged apart.  The mean and the reference of a perturbed field
differ in general; `unperturbed` gives the second.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .fields import Field, RadialField
from .layerfunction import NumericLayer

if TYPE_CHECKING:
    from .model import Model
    from .sampling import AngularGrid

__all__ = ["spherical_mean", "quadrature_mean", "spherically_averaged"]

# The most field values a quadrature mean evaluates at once (32 MB of doubles).
_BLOCK_VALUES = 1 << 22


def quadrature_mean(field: Field, grid: AngularGrid) -> RadialField:
    """The mean of a scalar field by `grid`'s quadrature at every radius it
    is asked at: a `RadialField` over a `NumericLayer`."""
    theta = grid.colatitudes[None, :, None]
    phi = grid.longitudes[None, None, :]

    # The field is evaluated on the whole grid at each DISTINCT radius, a
    # bounded number of radii at a time: the work is the number of distinct
    # radii times the grid's size, whatever the shape or repetition of r.
    per_block = max(1, _BLOCK_VALUES // (grid.ntheta * grid.nphi))

    def fbar(r: np.ndarray) -> np.ndarray:
        r = np.asarray(r, dtype=float)
        radii, inverse = np.unique(r.ravel(), return_inverse=True)
        means = np.empty(radii.shape)
        for start in range(0, radii.size, per_block):
            block = radii[start:start + per_block]
            values = field.evaluate(block[:, None, None], theta, phi)
            means[start:start + per_block] = grid.mean(values)
        return means[inverse.ravel()].reshape(r.shape)

    grid.mean(np.zeros((grid.ntheta, grid.nphi)))       # refuses a grid without weights
    return RadialField(field.interval, NumericLayer(field.interval, fbar),
                       character=field.character, name=field.name)


def spherical_mean(field: Field, *, grid: AngularGrid | None = None) -> RadialField:
    """The spherical mean of a scalar field, by the rule of the module
    docstring: the field's own when it has one, else quadrature on `grid`."""
    if field.character.rank:
        raise ValueError(f"a spherical mean is taken of a field of rank 0, not "
                         f"{field.character}")
    own = getattr(field, "spherical_mean", None)
    if own is not None:
        return own(grid=grid)
    if getattr(field, "is_radial", False):
        return RadialField(field.interval,
                           NumericLayer(field.interval,
                                        lambda r: field.evaluate(r, None, None)),
                           character=field.character, name=field.name)
    if grid is None:
        raise ValueError(f"{field!r} has no spherical mean of its own; pass "
                         "grid=, an angular grid with weights to take it by "
                         "quadrature")
    return quadrature_mean(field, grid)


def spherically_averaged(model: Model, *, grid: AngularGrid | None = None) -> Model:
    """`model` with the spherical mean of every field that is not radial in
    its place, through `with_field(replace=True)`, layer by layer in each
    layer's order of names; a field made radial by an earlier
    replacement is left as it now is."""
    out = model
    for layer in model.layers:
        for name in layer.names:
            current = out.layer(layer.index)
            if name not in current or getattr(current[name], "is_radial", False):
                continue
            field = current[name]
            out = out.with_field(layer.index, name,
                                 spherical_mean(field, grid=grid), replace=True)
    return out
