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
from .mesh1d.gravity import gravity, gravity_fields, mass

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
    by layer, and the isotropic re-description of the whole model.

    `elastic_moduli(which)` is the layer's tensor of any symmetry, its
    own where it holds one; `moduli(which)` is the spherically symmetric
    reading, the five transversely isotropic moduli, and `kappa_mu` and
    `isotropic` go through them, so a layer holding a general anisotropic
    tensor is refused by those three until the general Voigt average is
    written.
    """

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
    """The gravity and mass of a spherically symmetric model with a radial
    density on every layer: the reference body's, computed on each call
    and never stored, as numbers by `gravity` and `mass`, as one radial
    field per layer by `gravity_fields`, and as a copy of the model
    holding that field under the vocabulary name `g` by `with_gravity`.
    A density that depends on direction is refused, and the geometry's
    mapping does not enter (see `mesh1d.gravity`)."""

    gravity = gravity
    mass = mass
    gravity_fields = gravity_fields

    def with_gravity(self: Model, *, name: str = "g", replace: bool = False) -> Model:
        """The model with its gravity attached to every layer as a radial
        field under `name`, exact where the density is polynomial; a
        layer already holding `name` is refused unless `replace`."""
        fields = gravity_fields(self)
        layers = []
        for layer, field in zip(self.layers, fields):
            if name in layer and not replace:
                raise ValueError(
                    f"layer {layer.index} ({layer.name!r}) already holds {name!r}; "
                    "pass replace=True to replace it")
            layers.append({**layer.fields, name: field})
        return self.replaced(layers=layers)


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
