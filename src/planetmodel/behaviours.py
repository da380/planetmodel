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

`Elastic`, `ConstantQ` and `Viscoelastic` read and build fields point by
point, so they serve a flat model (`planetmodel.cartesian`) as they
serve a spherical one, whose layer's depth stands where the radius
does.  `SelfGravitating` computes gravity from the mass inside a
radius, which is the sphere's alone: on a flat model its methods refuse
by name, and a flat model's gravity is a prescribed field
(`cartesian.Model.with_uniform_gravity`).
"""
from __future__ import annotations

import functools
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike

from . import materials, rheology
from .character import DENSITY, SCALAR
from .fields import ComposedField, Field, RadialField
from .layerfunction import (
    LayerFunctionLike,
    NumericLayer,
    as_layer_function,
    constant_layer,
)
from .mesh1d.gravity import (
    gravity,
    gravity_fields,
    layer_masses,
    mass,
    pressure,
    pressure_fields,
)
from .mesh1d.stratification import (
    BULLEN_EXCESS,
    Stabilisation,
    brunt_vaisala_squared,
    bullen_excess,
    bullen_parameter,
    is_neutral,
    stable_layer,
    stratified_density,
)

if TYPE_CHECKING:
    from .geometry import Names
    from .model import CoarseningFit, Layer, Model

__all__ = ["layer_method", "with_moduli", "with_velocities", "Elastic", "ConstantQ",
           "SelfGravitating", "Viscoelastic", "ELASTIC_FAMILIES", "elastic_family"]

#: The velocity names an elastic description may be given in.
VELOCITY_NAMES = materials.VELOCITY_NAMES

#: The ways a layer describes its elastic medium, each a family of names
#: that is complete on its own: the isotropic and the transversely
#: isotropic velocities beside rho, the isotropic moduli, the five Love
#: moduli, and the tensor stored under `elastic_moduli`.
ELASTIC_FAMILIES = (materials.VELOCITY_SETS[materials.Symmetry.ISOTROPIC],
                    materials.VELOCITY_SETS[materials.Symmetry.VTI],
                    materials.MODULI_NAMES[materials.Symmetry.ISOTROPIC],
                    materials.MODULI_NAMES[materials.Symmetry.VTI],
                    ("elastic_moduli",))


def elastic_family(name: str) -> tuple[str, ...]:
    """The family of `ELASTIC_FAMILIES` an elastic name belongs to."""
    for family in ELASTIC_FAMILIES:
        if name in family:
            return family
    raise KeyError(f"{name!r} is not an elastic name; those are "
                   f"{list(materials.ELASTIC_NAMES)}")


#: The fields a frozen layer's complex moduli were derived from.
_FROZEN_FROM = ("rho",) + materials.ELASTIC_NAMES + rheology.RHEOLOGY_NAMES


def _refuse_frozen(model: Model, why: str) -> None:
    """Refuse a change that would leave a frozen model's moduli stale."""
    if rheology.is_frozen(model):
        raise ValueError(
            f"the model is frozen at omega = {float(model.constant('omega')):g} "
            f"and {why}, so the frozen moduli would go stale; make the change "
            "on the elastic model and freeze it again")


def _refuse_other_symmetry(layer: Layer, name: str) -> None:
    """Refuse a velocity of one symmetry on a layer holding velocities of
    the other, naming what the layer holds and the way out."""
    sets = materials.VELOCITY_SETS
    own = next((s for s, names in sets.items() if name in names), None)
    held = [n for s, names in sets.items() if s is not own for n in names
            if n in layer]
    if own is None or not held:
        return
    if own is materials.Symmetry.ISOTROPIC:
        remedy = ("replace one of those, or describe the layer by vp and vs "
                  "with isotropic() first")
    else:
        remedy = (f"no one velocity completes {list(sets[own])}; replace one of "
                  "the Love moduli A, C, F, L, N instead, and the velocities are "
                  "derived again from the five")
    where = f"layer {layer.index}" + (f" ({layer.name!r})" if layer.name else "")
    raise ValueError(f"{where} holds the velocities {held}, so {name!r} would "
                     f"contradict them; {remedy}")


def _description_family(model: Model, group: Sequence[int]) -> tuple[str, ...]:
    """The elastic names a merged group is refit through: the first
    description every layer of the group holds, velocities preferred
    since their products keep the derived moduli exact polynomials."""
    held = set.intersection(*(set(model.layer(i).names) for i in group))
    if "rho" in held and ({"vp", "vs"} <= held or {"vpv", "vsv"} <= held):
        return tuple(nm for nm in VELOCITY_NAMES if nm in held)
    five = materials.MODULI_NAMES[materials.Symmetry.VTI]
    if set(five) <= held:
        return five
    iso = materials.MODULI_NAMES[materials.Symmetry.ISOTROPIC]
    if set(iso) <= held:
        return iso
    raise ValueError(
        f"merged layers {list(group)} share no complete elastic description "
        "in velocities or moduli (perhaps only 'elastic_moduli'); keep the "
        "interface, or re-describe the layers first")


#: The elastic names that are moduli, not velocities.
_MODULUS_NAMES = tuple(n for n in materials.ELASTIC_NAMES if n not in VELOCITY_NAMES)


def _label(layer: Layer) -> str:
    """A layer by its index, and its name where it has one."""
    return f"layer {layer.index}" + (f" ({layer.name!r})" if layer.name else "")


def _excess_field(model: Model, layer: Layer, excess: Field | LayerFunctionLike
                  ) -> RadialField:
    """`excess` as the Bullen excess of a fluid layer: a radial scalar
    field on its interval that is nowhere negative.  A layer that is not
    fluid, an excess that depends on direction and one that is negative
    somewhere are refused."""
    where = _label(layer)
    try:
        fluid = materials.is_fluid(layer)
    except KeyError:
        fluid = False
    if not fluid:
        raise ValueError(
            f"{where} is not a fluid layer; only a fluid is described by its "
            "stratification, and a solid's density is its own description")
    if isinstance(excess, Field):
        if not isinstance(excess, RadialField) or excess.character.rank != 0:
            raise ValueError(
                f"the Bullen excess of {where} must be a radial scalar field")
        excess = excess.function
    field = model.geometry.radial_field_type(
        layer.interval, as_layer_function(layer.interval, excess), character=SCALAR,
        name=BULLEN_EXCESS)
    pieces = field.radial_pieces()
    breaks = np.asarray(layer.interval if pieces is None else pieces[0], dtype=float)
    x = 0.5 * (1.0 - np.cos(np.pi * np.arange(17) / 16.0))        # in [0, 1]
    r = (breaks[:-1, None] + (breaks[1:] - breaks[:-1])[:, None] * x).ravel()
    values = np.asarray(field(r), dtype=float)
    if np.min(values) < -1e-12 * (1.0 + np.max(np.abs(values))):
        raise ValueError(
            f"the Bullen excess given for {where} is negative in places (as low "
            f"as {np.min(values):g}): the layer would be unstably stratified, "
            "and a description by stratification is of a stable layer")
    return field


def _p_stiffness(layer: Layer) -> tuple[str, LayerFunctionLike]:
    """What resists compression in a fluid layer, as `stratified_density`
    takes it: its P velocity where the layer holds one as a radial field
    of its own, the velocities then being its description, and its bulk
    modulus otherwise."""
    for name in ("vp", "vpv"):
        field = layer.fields.get(name)
        if isinstance(field, RadialField):
            return "vp", field.function
    kappa = materials.kappa_mu(layer)[0]
    return "kappa", kappa.function if isinstance(kappa, RadialField) else kappa


def _derived(model: Model, j: int, *, rtol: float = 1e-12,
             **anchor: float) -> Model:
    """`model` with the density of layer `j` derived from the Bullen
    excess it holds, under the gravity of the mass beneath it, and its
    elastic description completed again: the moduli follow the new
    density where the velocities describe the layer, the velocities
    where the moduli do.  `anchor` is the one number `stratified_density`
    is given."""
    layer = model.layers[j]
    kind, stiffness = _p_stiffness(layer)
    below = float(np.sum(layer_masses(model)[:j]))
    density = stratified_density(
        layer.interval, layer[BULLEN_EXCESS].function, G=model.G, mass_below=below,
        rtol=rtol, **{kind: stiffness}, **anchor)
    drop = _MODULUS_NAMES if kind == "vp" else VELOCITY_NAMES
    fields = {k: f for k, f in layer.fields.items() if k not in drop}
    fields["rho"] = model.geometry.radial_field_type(
        layer.interval, density, character=DENSITY, name="rho")
    layers = [dict(other.fields) for other in model.layers]
    layers[j] = with_velocities(with_moduli(fields))
    return model.replaced(layers=layers)


def _derived_above(model: Model, i: int, masses: np.ndarray) -> Model:
    """`model` with the density of every layer above layer `i` that holds
    a Bullen excess derived again, each keeping the mass `masses` gives
    it: what a change of the mass inside them asks for, gravity in them
    having changed."""
    for j in range(i + 1, model.nlayers):
        if BULLEN_EXCESS in model.layers[j]:
            model = _derived(model, j, mass=float(masses[j]))
    return model


def _stratification_kept(before: Model, after: Model, i: int, name: str,
                         replace: bool) -> Model:
    """`after`, the model `before` with `name` attached to layer `i`, with
    the densities that are derived from a Bullen excess derived again
    where the change asks for it.

    A Bullen excess attached to a layer derives its density, the layer
    keeping its mass.  A density attached to a layer becomes its
    description: the layer's Bullen excess is dropped, and the layers
    above that hold one are derived again, each keeping its mass.  An
    elastic field replaced on a layer holding a Bullen excess derives
    that layer's density again, the layer keeping its mass.  A model
    that is not self-gravitating is returned as it is.
    """
    if not isinstance(after, SelfGravitating):
        return after
    if name == BULLEN_EXCESS:
        layer = after.layers[i]
        layers = [dict(other.fields) for other in after.layers]
        layers[i][BULLEN_EXCESS] = _excess_field(after, layer, layer[BULLEN_EXCESS])
        held = after.replaced(layers=layers)
        return _derived(held, i, mass=float(layer_masses(before)[i]))
    if not any(BULLEN_EXCESS in layer for layer in after.layers):
        return after
    if name == "rho":
        masses = layer_masses(before)
        if BULLEN_EXCESS in after.layers[i]:
            after = after.without_field(BULLEN_EXCESS, layers=[i])
        return _derived_above(after, i, masses)
    if replace and name in materials.ELASTIC_NAMES and BULLEN_EXCESS in after.layers[i]:
        return _derived(after, i, mass=float(layer_masses(before)[i]))
    return after


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

    symmetry, _ = materials.independent_moduli(out)
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
    is refused by the three that read the five.

    The derived fields are kept in step by `with_field`: replacing an
    elastic field on a layer makes the family it belongs to (rho with
    the isotropic or the transversely isotropic velocities, kappa and
    mu, or the five) the description, drops the other elastic names from
    that layer, and completes the description again, so the five follow
    a new `vs` and the velocities a new `L`; a field added beside the
    existing ones is attached as it is, and a velocity of the other
    symmetry than the layer's is refused, as is any change to what a
    frozen model's moduli were derived from.  A change that goes through
    `replaced` directly, as the library's own transformations do, is the
    caller's to keep consistent.

    On a model that is also `SelfGravitating`, `with_field` keeps in step
    the densities that are derived from a Bullen excess
    (`SelfGravitating.with_bullen_excess`): see `with_field`.
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

    def with_field(self: Model, which: int | str, name: str, field: Field, *,
                   replace: bool = False) -> Model:
        """A copy with `field` attached to one layer under `name`.

        Replacing an elastic field makes its family the layer's
        description and derives the other elastic fields from it again;
        replacing rho on a layer holding velocities makes the velocities
        the description, so the moduli follow the new density.  A field
        added beside the existing ones, or a family too incomplete to
        describe the medium, is attached as it is.  A velocity of one
        symmetry on a layer holding velocities of the other is refused,
        added or replaced: no single velocity completes the other set,
        and `isotropic()` re-describes a layer by vp and vs.  On a frozen
        model (`rheology.is_frozen`) rho, the elastic names and the
        rheology names are refused, the frozen moduli having been derived
        from them.

        On a self-gravitating model the densities derived from a Bullen
        excess follow.  A Bullen excess attached to a fluid layer derives
        the layer's density from it, the layer keeping its mass.  An
        elastic field replaced on a layer that holds a Bullen excess
        derives its density again, the layer keeping its mass.  A density
        attached to a layer is that layer's description from then on: a
        Bullen excess it held is dropped, and since the mass inside the
        layers above may have changed, each of those that holds a Bullen
        excess has its density derived again, keeping its own mass.
        """
        from .model import ModelBase
        if (name in _FROZEN_FROM or name == BULLEN_EXCESS
                or rheology.is_rheology_name(name)):
            _refuse_frozen(self, f"its moduli were derived from {name!r}"
                           if name != BULLEN_EXCESS else
                           "the density is derived from the Bullen excess")
        _refuse_other_symmetry(self.layer(which), name)
        out = ModelBase.with_field(self, which, name, field, replace=replace)
        i = self.layer(which).index
        return _stratification_kept(self, self._completed(out, i, name, replace),
                                    i, name, replace)

    def _completed(self: Model, out: Model, i: int, name: str,
                   replace: bool) -> Model:
        """`out`, this model with `name` attached to layer `i`, with the
        elastic description of that layer completed again from the family
        the change makes its description."""
        if name == "rho" and any(n in out.layer(i) for n in VELOCITY_NAMES):
            keep = VELOCITY_NAMES
        elif not replace or name not in materials.ELASTIC_NAMES:
            return out
        else:
            keep = elastic_family(name)
        layers = [dict(layer.fields) for layer in out.layers]
        fields = {k: f for k, f in layers[i].items()
                  if k not in materials.ELASTIC_NAMES or k in keep}
        try:
            layers[i] = with_velocities(with_moduli(fields))
        except KeyError:
            return out
        return out.replaced(layers=layers)

    def coarsened(self: Model, *, keep: Iterable[int | str] | None = None,
                  drop: Iterable[int | str] | None = None, degree: int = 3,
                  n: int = 64) -> tuple[Model, CoarseningFit]:
        """`Model.coarsened` with the elastic description kept consistent.

        Each merged group is refit through its base description alone,
        rho with the velocities where every merged layer holds them
        (the five then follow as exact polynomial products of the
        fits), else the five Love moduli, else kappa and mu; the other
        elastic names are dropped before the fit and derived again
        from the fitted description, so the coarse model satisfies the
        elastic relations exactly.  A group whose layers share no such
        description (one speaking only through `elastic_moduli`, say)
        is refused, as is a frozen model.  Layers left whole are
        untouched, but for the density of a layer that holds a Bullen
        excess and lies above a merged group: the refit changes the mass
        beneath it, so it is derived again, the layer keeping its mass.
        A group that includes a layer holding a Bullen excess is refused:
        a fit of the density is not a density derived from a fit of the
        excess.
        """
        from .model import ModelBase
        _refuse_frozen(self, "coarsening refits the elastic description")
        _, cmap = self._coarsening(keep=keep, drop=drop)
        layers = [dict(layer.fields) for layer in self.layers]
        merged = [group for group in cmap.layers if len(group) > 1]
        for group in merged:
            held = [i for i in group if BULLEN_EXCESS in self.layers[i]]
            if held:
                stratified = _label(self.layers[held[0]])
                raise ValueError(
                    f"merged layers {list(group)} include {stratified}, whose "
                    f"density is derived from its {BULLEN_EXCESS!r}; keep its "
                    "interfaces, or drop the field with without_field first")
            family = _description_family(self, group)
            for i in group:
                layers[i] = {k: f for k, f in layers[i].items()
                             if k not in materials.ELASTIC_NAMES or k in family}
        stripped = self.replaced(layers=layers, check=False)
        coarse, fit = ModelBase.coarsened(stripped, keep=keep, drop=drop,
                                          degree=degree, n=n)
        out = [dict(layer.fields) for layer in coarse.layers]
        lowest = None
        for j, group in enumerate(fit.map.layers):
            if len(group) > 1:
                out[j] = with_velocities(with_moduli(out[j]))
                lowest = j if lowest is None else lowest
        result = coarse.replaced(layers=out)
        if lowest is not None and isinstance(result, SelfGravitating):
            result = _derived_above(result, lowest, layer_masses(coarse))
        return result, fit

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


def _with_radial(model: Model, fields: Sequence[Field], name: str,
                 replace: bool) -> Model:
    """The model with one field per layer attached under `name`, a layer
    already holding it refused unless `replace`."""
    layers = []
    for layer, field in zip(model.layers, fields):
        if name in layer and not replace:
            raise ValueError(
                f"layer {layer.index} ({layer.name!r}) already holds {name!r}; "
                "pass replace=True to replace it")
        layers.append({**layer.fields, name: field})
    return model.replaced(layers=layers)


class SelfGravitating:
    """The gravity, mass and pressure of a spherically symmetric model with
    a radial density on every layer: the reference body's, computed on
    each call and never stored, as numbers by `gravity`, `mass` and
    `pressure`, as one radial field per layer by `gravity_fields` and
    `pressure_fields`, and as a copy of the model holding that field
    under the vocabulary name `g` by `with_gravity`, `pressure` by
    `with_pressure`;
    `with_stratification` attaches d rho / d Phi_0 = rho'(r) / g(r) to
    the fluid layers.  `bullen_parameter`, `bullen_excess` and
    `brunt_vaisala_squared` say how each layer is stratified
    (`mesh1d.stratification`).

    A fluid layer may be described by its stratification in place of its
    density: `with_bullen_excess` gives it a Bullen excess, held under
    that name, and derives its density from it, and `is_neutral` says
    whether the excess a layer holds vanishes.  `stabilised` makes the
    unstably stratified fluid layers stable, each becoming such a layer.
    On a model that is also
    `Elastic` such a density is kept in step with what it was derived
    from (`Elastic.with_field`).  Cutting the inside of the model away
    (`hollowed`) removes the mass under which those densities were
    derived, so the layers keep their densities and lose their Bullen
    excess, as does a layer made by continuing another outward
    (`extended` by "extrapolate" or "hold").

    A density that depends on direction is refused,
    and the geometry's mapping does not enter (see `mesh1d.gravity`).
    A model off the sphere (a flat one) is refused by name."""

    gravity = gravity
    mass = mass
    gravity_fields = gravity_fields
    pressure = pressure
    pressure_fields = pressure_fields
    bullen_parameter = bullen_parameter
    bullen_excess = bullen_excess
    brunt_vaisala_squared = brunt_vaisala_squared
    layer_masses = layer_masses
    is_neutral = layer_method(is_neutral)

    def with_bullen_excess(self: Model, which: int | str,
                           excess: Field | LayerFunctionLike, *,
                           mass: float | None = None,
                           rho_bottom: float | None = None,
                           rho_top: float | None = None,
                           rtol: float = 1e-12) -> Model:
        """The model with one fluid layer described by its stratification.

        `excess` is the Bullen excess of the layer: a radial field, a
        layer function, a number, or a callable of radius, nowhere
        negative; zero where the layer is neutrally stratified.  It is
        attached under the vocabulary name `bullen_excess`, and the
        layer's density is derived from it and from what the layer holds
        to resist compression, its P velocity where the velocities
        describe it and its bulk modulus otherwise, under the gravity of
        the mass beneath (`mesh1d.stratification.stratified_density`, to
        `rtol`).  One number fixes the density: `mass`, the mass of the
        layer; `rho_bottom`, its density at the inner radius; or
        `rho_top`, at the outer.  With none given the layer keeps the
        mass it has, and gravity above it is unchanged.  The elastic
        description of the layer is completed again from the new
        density.  A layer that is not fluid, and a frozen model, are
        refused.
        """
        _refuse_frozen(self, "the density is derived from the Bullen excess")
        layer = self.layer(which)
        i = layer.index
        given = {k: v for k, v in (("mass", mass), ("rho_bottom", rho_bottom),
                                   ("rho_top", rho_top)) if v is not None}
        if not given:
            given = {"mass": float(layer_masses(self)[i])}
        layers = [dict(other.fields) for other in self.layers]
        layers[i][BULLEN_EXCESS] = _excess_field(self, layer, excess)
        return _derived(self.replaced(layers=layers), i, rtol=rtol, **given)

    def stabilised(self: Model, *, layers: Iterable[int | str] | None = None,
                   inertia: bool = True, neutral_below: float = 0.0,
                   rtol: float = 1e-12) -> tuple[Model, Stabilisation]:
        """The model with its unstably stratified fluid made stable, and
        what was done.

        Each fluid layer of `layers`, every fluid layer by default, that
        is unstable in places is replaced by the stable layer nearest to
        it (`mesh1d.stratification.stable_layer`): its P velocity, its
        radii and its mass are kept, and with `inertia` its moment of
        inertia, so that nothing above or below it changes and the
        model's mass and moment of inertia are what they were.  The
        layer is then described by its stratification, holding its
        Bullen excess, zero on the intervals made neutral, and the
        density derived from it (`with_bullen_excess`).  A layer
        described by its bulk modulus is described by its P velocity
        from then on, that being what is kept.  With `neutral_below`
        positive a stretch whose excess stays under it is made neutral
        as well.

        A layer that is stable as it stands, or already holds a Bullen
        excess, is left alone, and a model of such layers is returned as
        it is.  A layer that cannot keep its moment of inertia, being
        unstable throughout as a uniform layer is, is refused by name:
        leave it out of `layers`, or pass `inertia=False`.  A layer of
        `layers` that is not fluid, and a frozen model, are refused.
        """
        _refuse_frozen(self, "stabilising changes the density")
        if layers is None:
            which = []
            for layer in self.layers:
                try:
                    if materials.is_fluid(layer):
                        which.append(layer.index)
                except KeyError:
                    pass
        else:
            which = sorted({self.layer(w).index for w in layers})
            for i in which:
                _excess_field(self, self.layers[i], 0.0)     # refuses a solid
        masses = layer_masses(self)
        out = [dict(layer.fields) for layer in self.layers]
        found = {}
        make = self.geometry.radial_field_type
        for i in which:
            layer = self.layers[i]
            if BULLEN_EXCESS in layer:
                continue
            kind, stiffness = _p_stiffness(layer)
            rho = layer["rho"]
            if kind == "kappa":
                kappa = materials.kappa_mu(layer)[0]
                stiffness = NumericLayer(
                    layer.interval,
                    lambda r, kappa=kappa, rho=rho: np.sqrt(kappa(r) / rho(r)))
            try:
                result = stable_layer(
                    layer.interval, rho.function, stiffness, G=self.G,
                    mass_below=float(np.sum(masses[:i])), inertia=inertia,
                    neutral_below=neutral_below, rtol=rtol)
            except ValueError as exc:
                raise ValueError(f"{_label(layer)}: {exc}") from exc
            if result is None:
                continue
            if kind == "kappa":
                fields = {k: f for k, f in layer.fields.items()
                          if k not in materials.ELASTIC_NAMES}
                fields["vp"] = make(layer.interval, stiffness, name="vp")
                fields["vs"] = make(layer.interval,
                                    constant_layer(layer.interval, 0.0), name="vs")
            else:
                fields = {k: f for k, f in layer.fields.items()
                          if k not in _MODULUS_NAMES}
            fields["rho"] = make(layer.interval, result.density, character=DENSITY,
                                 name="rho")
            fields[BULLEN_EXCESS] = make(layer.interval, result.excess,
                                         character=SCALAR, name=BULLEN_EXCESS)
            out[i] = with_velocities(with_moduli(fields))
            found[i] = result
        if not found:
            return self, Stabilisation({})
        return self.replaced(layers=out), Stabilisation(found)

    def hollowed(self: Model, radius: float, *, name: str | None = None) -> Model:
        """`Model.hollowed`; the layers that hold a Bullen excess keep
        their densities and lose it, the mass under which the densities
        were derived having been cut away."""
        out = super().hollowed(radius, name=name)
        if any(BULLEN_EXCESS in layer for layer in out.layers):
            out = out.without_field(BULLEN_EXCESS)
        return out

    def extended(self: Model, radii: ArrayLike, *,
                 fields: str | Sequence[Mapping[str, Field]] | None = None,
                 names: Names = None, interface_names: Names = None) -> Model:
        """`Model.extended`; a layer made by continuing the last one
        outward ("extrapolate" or "hold") does not take its Bullen
        excess, its density being a continuation and not derived."""
        out = super().extended(radii, fields=fields, names=names,
                               interface_names=interface_names)
        new = range(self.nlayers, out.nlayers)
        if isinstance(fields, str) and any(BULLEN_EXCESS in out.layers[j]
                                           for j in new):
            out = out.without_field(BULLEN_EXCESS, layers=new)
        return out

    def with_gravity(self: Model, *, name: str = "g", replace: bool = False) -> Model:
        """The model with its gravity attached to every layer as a radial
        field under `name`, exact where the density is polynomial; a
        layer already holding `name` is refused unless `replace`."""
        return _with_radial(self, gravity_fields(self), name, replace)

    def with_pressure(self: Model, *, name: str = "pressure",
                      replace: bool = False) -> Model:
        """The model with its hydrostatic pressure attached to every layer
        as a radial field under `name` (`mesh1d.gravity`); a layer already
        holding `name` is refused unless `replace`."""
        return _with_radial(self, pressure_fields(self), name, replace)

    def with_stratification(self: Model, *, name: str = "drho_dphi0",
                            replace: bool = False) -> Model:
        """The model with the stratification of its fluid layers,
        d rho / d Phi_0 = rho'(r) / g(r), attached to each of them
        under `name`; solid layers, and layers whose fluidity cannot
        be read, are left alone, and a model with no fluid layer is
        refused.  rho' is exact where the density is polynomial; the
        field is zero where g vanishes (the centre).  On a layer that
        holds its Bullen excess e the field is -(1 + e) rho^2 / kappa,
        with no derivative taken.  A fluid layer already holding `name`
        is refused unless `replace`."""
        fields = gravity_fields(self)
        layers = []
        n_fluid = 0
        for layer, g in zip(self.layers, fields):
            try:
                fluid = materials.is_fluid(layer)
            except KeyError:
                fluid = False
            if not fluid:
                layers.append(dict(layer.fields))
                continue
            if name in layer and not replace:
                raise ValueError(
                    f"layer {layer.index} ({layer.name!r}) already holds {name!r}; "
                    "pass replace=True to replace it")
            rho = layer["rho"]
            if BULLEN_EXCESS in layer:
                n_fluid += 1
                strat = ComposedField(
                    lambda e, d, k: -(1.0 + e) * d * d / k,
                    (layer[BULLEN_EXCESS], rho, materials.kappa_mu(layer)[0]),
                    character=DENSITY, name=name)
                layers.append({**layer.fields, name: strat})
                continue
            if not hasattr(rho, "derivative"):
                raise ValueError(
                    f"layer {layer.index} ({layer.name!r}) holds a rho that "
                    "cannot state its radial derivative, so its "
                    "stratification cannot be formed")
            n_fluid += 1
            strat = ComposedField(
                lambda d, a: np.where(a != 0.0, d, 0.0)
                / np.where(a != 0.0, a, 1.0),
                (rho.derivative(nu=1), g), character=DENSITY, name=name)
            layers.append({**layer.fields, name: strat})
        if not n_fluid:
            raise ValueError(
                "no fluid layer: the stratification d rho / d Phi_0 is a "
                "property of fluids, and no layer of this model is one")
        return self.replaced(layers=layers)


class Viscoelastic:
    """The rheology read from the fields: a layer's viscoelasticity, the
    model frozen at a frequency, and the elastic model that drops the
    rheology fields."""

    is_viscoelastic = layer_method(rheology.is_viscoelastic)
    frozen = rheology.frozen

    def elastic(self: Model) -> Model:
        """The model without its rheology fields (`rheology.is_rheology_name`), so
        that every layer is elastic; a frozen model keeps its complex
        moduli, which are what its rheology became."""
        layers = [{k: f for k, f in layer.fields.items()
                   if not rheology.is_rheology_name(k)} for layer in self.layers]
        return self.replaced(layers=layers, check=False)
