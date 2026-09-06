"""The free functions as methods: an adaptor, and stateless mixins.

A model type is a class derived from `Model` alone; what it exposes
beyond the base it gets by wrapping the free functions of the library,
never by reimplementing them and never through a hierarchy of model
types.  Two forms of wrapping are here.

`layer_method(fn)` turns a function of a layer, such as `moduli(layer)`,
into a method `model.moduli(which)` that resolves `which` (an index or a
name) through `model.layer`.  A function of a model, such as
`gravity(model, radii)` or `frozen(model, omega)`, needs no adaptor:
assigned in a class body it is already a method.

The mixins bundle the wrapped methods a kind of model exposes.  Each is
a class body of such assignments and nothing else: no state, no
constructor, so `class PREM(Elastic, SelfGravitating, Viscoelastic,
Model)` is still one model type derived from `Model` alone.  The
transformations `isotropic` and `elastic` return copies of the same
class through `Model.replaced`.
"""
from __future__ import annotations

import functools
from collections.abc import Callable
from typing import TYPE_CHECKING

from . import materials, rheology
from .mesh1d.gravity import gravity, mass

if TYPE_CHECKING:
    from .model import Model

__all__ = ["layer_method", "Elastic", "SelfGravitating", "Viscoelastic"]


def layer_method[T](fn: Callable[..., T]) -> Callable[..., T]:
    """`fn(layer, *args, **kwargs)` as a method `model.fn(which, *args, **kwargs)`.

    `which` is a layer index (negatives counting back) or a layer name,
    resolved by `model.layer`; the docstring and name are the function's.
    """
    @functools.wraps(fn)
    def method(self: Model, which: int | str, *args: object, **kwargs: object) -> T:
        return fn(self.layer(which), *args, **kwargs)
    return method


class Elastic:
    """The elastic behaviour: fluidity, the moduli and the Voigt average
    by layer, and the isotropic re-description of the whole model."""

    is_fluid = layer_method(materials.is_fluid)
    moduli = layer_method(materials.moduli)
    elastic_moduli = layer_method(materials.elastic_moduli)
    kappa_mu = layer_method(materials.kappa_mu)

    def isotropic(self: Model) -> Model:
        """The model with every elastic description replaced by its Voigt
        average: each layer keeps rho and everything that is not an
        elastic name, and holds kappa and mu in place of whatever it
        described its medium by.  Exact on polynomial layers.  A layer
        holding no elastic description is left alone.
        """
        layers = []
        for layer in self.layers:
            fields = {k: f for k, f in layer.fields.items()
                      if k not in materials.ELASTIC_NAMES}
            if any(n in layer for n in materials.ELASTIC_NAMES):
                kappa, mu = materials.kappa_mu(layer)
                fields["kappa"], fields["mu"] = kappa, mu
            layers.append(fields)
        return self.replaced(layers=layers)


class SelfGravitating:
    """The gravity of a model with density on every layer."""

    gravity = gravity
    mass = mass


class Viscoelastic:
    """The rheology read from the fields: a layer's viscoelasticity, the
    model frozen at a frequency, and the elastic model that drops the
    rheology fields."""

    is_viscoelastic = layer_method(rheology.is_viscoelastic)
    frozen = rheology.frozen

    def elastic(self: Model) -> Model:
        """The model without its rheology fields (`RHEOLOGY_NAMES`), so
        that every layer is elastic; a frozen model keeps its complex
        moduli, which are what its rheology became."""
        layers = [{k: f for k, f in layer.fields.items()
                   if k not in rheology.RHEOLOGY_NAMES} for layer in self.layers]
        return self.replaced(layers=layers, check=False)
