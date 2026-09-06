"""The free functions as methods: an adaptor, and stateless mixins.

A model type is a class derived from `Model` alone.  What its
constructor is given is its base information, PREM's polynomial
coefficients say; the model is that information and everything derived
from it, fields and behaviour alike, and the mixins here are the common
derivations written once: the elastic description completed in both
directions (moduli from velocities and velocities from moduli), the
elastic tensor and its averages, gravity, and the linear rheologies
built from their static parts.  A model type gets them by mixing in,
never by reimplementing them and never through a hierarchy of model
types.  Two forms of wrapping are here.

`layer_method(fn)` turns a function of a layer, such as `moduli(layer)`,
into a method `model.moduli(which)` that resolves `which` (an index or a
name) through `model.layer`.  A function of a model, such as
`gravity(model, radii)` or `frozen(model, omega)`, needs no adaptor:
assigned in a class body it is already a method.

The mixins bundle the wrapped methods a kind of model exposes.  Each is
a class body of such assignments and nothing else, with one exception:
`Elastic` has a constructor hook that attaches the five Love moduli to
every layer as first-class fields, since a spherically symmetric
elastic medium is those five and there is no reason to hide them behind
a call.  No mixin holds state of its own, so `class PREM(Elastic,
ConstantQ, SelfGravitating, Viscoelastic, Model)` is still one model
type derived from `Model` alone.  The transformations `isotropic`,
`elastic` and `with_gravity` return copies of the same class through
`Model.replaced`.
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Mapping
from typing import TYPE_CHECKING, Any

import numpy as np

from . import materials, rheology
from .character import SCALAR
from .fields import ComposedField, Field
from .mesh1d.gravity import gravity, gravity_fields, mass

if TYPE_CHECKING:
    from .model import Model

__all__ = ["layer_method", "with_moduli", "with_velocities", "Elastic", "ConstantQ",
           "SelfGravitating", "Viscoelastic"]

#: The velocity names an elastic description may be given in.
VELOCITY_NAMES = ("vp", "vs", "vpv", "vph", "vsv", "vsh", "eta")


def layer_method[T](fn: Callable[..., T]) -> Callable[..., T]:
    """`fn(layer, *args, **kwargs)` as a method `model.fn(which, *args, **kwargs)`.

    `which` is a layer index (negatives counting back) or a layer name,
    resolved by `model.layer`; the docstring and name are the function's.
    """
    @functools.wraps(fn)
    def method(self: Model, which: int | str, *args: object, **kwargs: object) -> T:
        return fn(self.layer(which), *args, **kwargs)
    return method


def with_moduli(fields: Mapping[str, Field]) -> dict[str, Field]:
    """A layer's fields with the five Love moduli A, C, F, L, N added,
    read from whatever elastic description it holds; unchanged where it
    already holds the five, holds its tensor directly, or holds no
    elastic description at all."""
    out = dict(fields)
    five = materials.MODULI_NAMES[materials.Symmetry.VTI]
    if (all(n in out for n in five) or "elastic_moduli" in out
            or not any(n in out for n in materials.ELASTIC_NAMES)):
        return out
    out.update(materials.moduli(out))
    return out


def with_velocities(fields: Mapping[str, Field]) -> dict[str, Field]:
    """A layer's fields with the velocities added from rho and the five
    Love moduli: vp and vs where the five are isotropic, else vpv, vph,
    vsv, vsh and eta.  Square roots are taken pointwise, so these are
    composed fields, not polynomials.  Unchanged where the layer already
    holds any velocity, lacks rho or the five, or holds its tensor
    directly."""
    out = dict(fields)
    five = materials.MODULI_NAMES[materials.Symmetry.VTI]
    if (any(n in out for n in VELOCITY_NAMES) or "rho" not in out
            or not all(n in out for n in five) or "elastic_moduli" in out):
        return out
    rho = out["rho"]
    A, C, F, L, N = (out[n] for n in five)

    def speed(modulus: Field, name: str) -> Field:
        return ComposedField(lambda m, d: np.sqrt(m / d), (modulus, rho),
                             character=SCALAR, name=name)

    symmetry, _ = materials._independent_moduli(out)
    if symmetry is materials.Symmetry.ISOTROPIC:
        out["vp"] = speed(C, "vp")
        out["vs"] = speed(L, "vs")
        return out
    out["vpv"], out["vph"] = speed(C, "vpv"), speed(A, "vph")
    out["vsv"], out["vsh"] = speed(L, "vsv"), speed(N, "vsh")
    out["eta"] = ComposedField(lambda f, a, l: f / (a - 2.0 * l), (F, A, L),
                               character=SCALAR, name="eta")
    return out


class Elastic:
    """The elastic behaviour of a spherically symmetric model.

    On construction the elastic description of every layer is completed
    in both directions: the five Love moduli A, C, F, L, N are attached
    where a layer was given rho with velocities or kappa and mu, exact
    where those are polynomial, and the velocities where a layer was
    given rho with the five (`with_moduli`, `with_velocities`); a layer
    already holding both, or its tensor under `elastic_moduli`, is left
    as it is.  The methods then read the fields: `moduli(which)` the
    five, `elastic_moduli` the tensor (the layer's own of any symmetry
    where it holds one), `kappa_mu` the Voigt average, `is_fluid` the
    vanishing of shear, and `isotropic()` the model re-described by its
    Voigt average, kappa and mu with the five and the velocities
    recomputed from them.  A layer holding a general anisotropic tensor
    is refused by the three that read the five, until the general Voigt
    average is written.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        from .model import Layer
        self._layers = tuple(
            Layer(layer.info, with_velocities(with_moduli(layer.fields)))
            for layer in self._layers)

    is_fluid = layer_method(materials.is_fluid)
    moduli = layer_method(materials.moduli)
    elastic_moduli = layer_method(materials.elastic_moduli)
    kappa_mu = layer_method(materials.kappa_mu)

    def isotropic(self: Model) -> Model:
        """The model with every elastic description replaced by its Voigt
        average: each layer keeps rho and everything that is not an
        elastic name, and holds kappa and mu, and the five recomputed
        from them, in place of whatever it described its medium by.
        Exact on polynomial layers.  A layer holding no elastic
        description is left alone.
        """
        layers = []
        for layer in self.layers:
            fields = {k: f for k, f in layer.fields.items()
                      if k not in materials.ELASTIC_NAMES}
            if any(n in layer for n in materials.ELASTIC_NAMES):
                kappa, mu = materials.kappa_mu(layer)
                fields["kappa"], fields["mu"] = kappa, mu
                fields = with_velocities(with_moduli(fields))
            layers.append(fields)
        return self.replaced(layers=layers)


class ConstantQ:
    """The Love moduli at a frequency under the constant-Q absorption
    band: kappa dispersed by `qkappa` and mu by `qmu` about a reference
    frequency, following the logarithmic dispersion relation, so that a
    layer's five at angular frequency omega are complex fields.  The
    model is not changed; `Viscoelastic.frozen` is the model that is.
    The reference frequency is the model's constant `omega_ref` where
    it declares one, else 2 pi rad/s, a period of one second, in the
    model's units (`reference_omega`).
    """

    reference_omega = rheology.reference_omega

    def moduli_at(self: Model, which: int | str, omega: float, *,
                  reference_omega: float | None = None) -> dict[str, Field]:
        """The five moduli of a layer at `omega`, complex where the layer
        holds a Q, else its static five."""
        ref = self.reference_omega() if reference_omega is None else reference_omega
        return rheology.dispersive_moduli(self.layer(which), omega,
                                          reference_omega=ref)

    def elastic_moduli_at(self: Model, which: int | str, omega: float, *,
                          reference_omega: float | None = None
                          ) -> materials.ElasticField:
        """The transversely isotropic tensor of a layer at `omega`."""
        five = self.moduli_at(which, omega, reference_omega=reference_omega)
        return materials.ElasticField(materials.Symmetry.VTI, five,
                                      name="elastic_moduli")


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
