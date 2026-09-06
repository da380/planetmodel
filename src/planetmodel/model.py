"""A model: a geometry with a bag of fields on every layer.

The third level of the design.  A `Model` holds a `Geometry`, one
mapping of name to field per layer, the `Scales` its numbers are in,
and the specs and constants that say what the names mean.  Everything
a field is asked goes through its layer: `model.layer(i)["rho"]`, or
`model.layer("mantle")["rho"]`.  A discontinuity is two layers asked
separately.

What a name means comes from the vocabulary, or from `specs=` for a
name outside it; a field attached under a name with a spec must have
the spec's character, and a name with no spec is accepted, carries no
dimensions, and is refused by name on conversion.  Nothing is required:
the function that needs a field asks the layer for it.  Behaviour
shared by groups of models is a free function of a layer or a model.

Units live here and nowhere else.  A field is numbers; the model's
`scales` say what one stored unit of length, mass and time is in SI,
and `converted` re-expresses the whole model, geometry included, in
other scales, by name and exactly for polynomial layers.  A constant is
read in the model's units, `G` included.

Surgery goes through the geometry's own and carries the fields: a
split layer's fields are re-stated on each part, a cut layer's on its
remainder, and appended shells hold the fields they are given.  Every
change is a copy that validates again.

A model type is a class derived from `Model` alone, plus the stateless
behaviour mixins of `planetmodel.behaviours` that wrap the free
functions as methods; there is no hierarchy of model types.  Every copy
goes through `replaced`, a shallow copy with the changed parts swapped
in, never through `__init__`, so a model type may give itself whatever
constructor its construction needs and every surgery, conversion and
freezing returns an instance of the same class carrying the same
instance state.
"""
from __future__ import annotations

import copy
from collections import abc
from collections.abc import Iterable, Iterator, Sequence
from types import MappingProxyType

import numpy as np
from numpy.typing import ArrayLike

from .displacement import RadialDisplacement, SphericalFunction
from .fields import Field
from .geometry import Geometry, LayerInfo, Names, Renames
from .layerfunction import same_interval
from .mapping import Mapping
from .skeleton import Skeleton
from .units import EARTH_MEAN_DENSITY, LENGTH, Scales
from .vocabulary import CONSTANTS, VOCABULARY, Constant, FieldSpec

__all__ = ["Layer", "Model"]


class Layer:
    """One layer of a model: its place in the geometry and its fields.

    `index`, `interval` and `name` are the geometry's; `fields` is a
    read-only mapping of name to field, also reached by `layer[name]`,
    `name in layer`, and `names`.
    """

    def __init__(self, info: LayerInfo, fields: abc.Mapping[str, Field]) -> None:
        self._info = info
        self._fields = MappingProxyType(dict(fields))

    @property
    def index(self) -> int:
        return self._info.index

    @property
    def interval(self) -> tuple[float, float]:
        return self._info.interval

    @property
    def name(self) -> str | None:
        return self._info.name

    @property
    def info(self) -> LayerInfo:
        return self._info

    @property
    def fields(self) -> abc.Mapping[str, Field]:
        return self._fields

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._fields)

    def __getitem__(self, name: str) -> Field:
        try:
            return self._fields[name]
        except KeyError:
            raise KeyError(
                f"layer {self._label()} holds no field {name!r}; it holds "
                f"{list(self._fields)}") from None

    def __contains__(self, name: object) -> bool:
        return name in self._fields

    def __iter__(self) -> Iterator[str]:
        return iter(self._fields)

    def __len__(self) -> int:
        return len(self._fields)

    def _label(self) -> str:
        return (f"{self._info.index} ({self._info.name!r})"
                if self._info.name is not None else str(self._info.index))

    def __repr__(self) -> str:
        lo, hi = self._info.interval
        return (f"Layer({self._label()}, [{lo:g}, {hi:g}], "
                f"fields={list(self._fields)})")


class Model:
    """A geometry with fields on every layer; see the module docstring.

    `layers` is one mapping of name to field per geometry layer, empty
    where a layer has no material.  `specs` gives a `FieldSpec` for
    names outside the vocabulary (or overriding it) and `constants` a
    `Constant` beyond the shipped ones.  `check=False` skips validation
    for fields already known to fit.
    """

    def __init__(self, geometry: Geometry, layers: Iterable[abc.Mapping[str, Field]],
                 *, scales: Scales = Scales.SI,
                 specs: abc.Mapping[str, FieldSpec] | None = None,
                 constants: abc.Mapping[str, Constant] | None = None,
                 check: bool = True) -> None:
        if not isinstance(geometry, Geometry):
            raise TypeError(f"expected a Geometry, got {type(geometry).__name__}")
        layers = [dict(m) for m in layers]
        if len(layers) != geometry.nlayers:
            raise ValueError(
                f"got fields for {len(layers)} layers; the geometry has "
                f"{geometry.nlayers}")
        if not isinstance(scales, Scales):
            raise TypeError(f"expected Scales, got {type(scales).__name__}")
        self._geometry = geometry
        self._scales = scales
        self._specs = MappingProxyType({**VOCABULARY, **dict(specs or {})})
        self._constants = MappingProxyType({**CONSTANTS, **dict(constants or {})})
        self._layers = tuple(Layer(info, fields)
                             for info, fields in zip(geometry.layers, layers))
        if check:
            self.validate()

    # -- validation ---------------------------------------------------------

    def validate(self) -> None:
        """Every field a Field on its layer's interval, with its spec's character."""
        for spec in self._specs.values():
            if not isinstance(spec, FieldSpec):
                raise TypeError(f"{spec!r} is not a FieldSpec")
        for c in self._constants.values():
            if not isinstance(c, Constant):
                raise TypeError(f"{c!r} is not a Constant")
        rtol = self._geometry.rtol
        for layer in self._layers:
            for name, field in layer.fields.items():
                if not isinstance(field, Field):
                    raise TypeError(
                        f"{name!r} on layer {layer._label()} is not a Field: "
                        f"{field!r}")
                if not same_interval(field.interval, layer.interval, rtol=rtol):
                    raise ValueError(
                        f"{name!r} on layer {layer._label()} lives on "
                        f"{field.interval}, not the layer's {layer.interval}")
                spec = self._specs.get(name)
                if spec is not None and field.character != spec.character:
                    raise ValueError(
                        f"{name!r} on layer {layer._label()} has character "
                        f"{field.character}; its spec says {spec.character}")

    # -- what it is ---------------------------------------------------------

    @property
    def geometry(self) -> Geometry:
        return self._geometry

    @property
    def skeleton(self) -> Skeleton:
        return self._geometry.skeleton

    @property
    def scales(self) -> Scales:
        return self._scales

    @property
    def nlayers(self) -> int:
        return len(self._layers)

    @property
    def layers(self) -> tuple[Layer, ...]:
        return self._layers

    @property
    def specs(self) -> abc.Mapping[str, FieldSpec]:
        """Every name with a meaning: the vocabulary, then `specs=`."""
        return self._specs

    @property
    def constants(self) -> abc.Mapping[str, Constant]:
        return self._constants

    def layer(self, which: int | str) -> Layer:
        """A layer by index (negatives allowed) or by name."""
        return self._layers[self._geometry.layer(which).index]

    def field_names(self) -> tuple[str, ...]:
        """Every name any layer holds, in first-appearance order."""
        seen = []
        for layer in self._layers:
            for name in layer.names:
                if name not in seen:
                    seen.append(name)
        return tuple(seen)

    def common_names(self) -> tuple[str, ...]:
        """The names every layer holds."""
        return tuple(n for n in self.field_names()
                     if all(n in layer for layer in self._layers))

    def layers_with(self, name: str) -> tuple[int, ...]:
        """The indices of the layers holding `name`."""
        return tuple(layer.index for layer in self._layers if name in layer)

    def spec(self, name: str) -> FieldSpec | None:
        """The spec of a name, or None when it has none."""
        return self._specs.get(name)

    def constant(self, name: str) -> float:
        """A declared constant in the model's units."""
        try:
            c = self._constants[name]
        except KeyError:
            raise KeyError(f"no constant {name!r}; the model knows "
                           f"{list(self._constants)}") from None
        return c.value_si / self._scales.factor(c.dimensions)

    @property
    def G(self) -> float:
        """The gravitational constant in the model's units."""
        return self.constant("G")

    # -- copies -------------------------------------------------------------

    def replaced(self, *, geometry: Geometry | None = None,
                 layers: Sequence[abc.Mapping[str, Field]] | None = None,
                 scales: Scales | None = None,
                 specs: abc.Mapping[str, FieldSpec] | None = None,
                 constants: abc.Mapping[str, Constant] | None = None,
                 check: bool = True) -> "Model":
        """A shallow copy with the given parts replaced, validated unless
        `check=False`: the one path every copy of a model takes.

        `layers` is one mapping of name to field per layer of the (new)
        geometry; `specs` and `constants` replace the model's own
        tables, the vocabulary and the shipped constants being merged in
        again.  The class and every other instance attribute are kept.
        """
        out = copy.copy(self)
        if geometry is not None:
            if not isinstance(geometry, Geometry):
                raise TypeError(f"expected a Geometry, got {type(geometry).__name__}")
            out._geometry = geometry
        if geometry is not None or layers is not None:
            fields = ([layer.fields for layer in self._layers] if layers is None
                      else [dict(m) for m in layers])
            if len(fields) != out._geometry.nlayers:
                raise ValueError(
                    f"got fields for {len(fields)} layers; the geometry has "
                    f"{out._geometry.nlayers}")
            out._layers = tuple(Layer(info, f)
                                for info, f in zip(out._geometry.layers, fields))
        if scales is not None:
            if not isinstance(scales, Scales):
                raise TypeError(f"expected Scales, got {type(scales).__name__}")
            out._scales = scales
        if specs is not None:
            out._specs = MappingProxyType({**VOCABULARY, **dict(specs)})
        if constants is not None:
            out._constants = MappingProxyType({**CONSTANTS, **dict(constants)})
        if check:
            out.validate()
        return out

    def with_field(self, which: int | str, name: str, field: Field, *,
                   replace: bool = False) -> "Model":
        """A copy with `field` attached to one layer under `name`."""
        i = self._geometry.layer(which).index
        layers = [dict(layer.fields) for layer in self._layers]
        if name in layers[i] and not replace:
            raise ValueError(
                f"layer {self._layers[i]._label()} already holds {name!r}; "
                "pass replace=True to replace it")
        layers[i][name] = field
        return self.replaced(layers=layers)

    def without_field(self, name: str, *,
                      layers: Iterable[int | str] | None = None) -> "Model":
        """A copy without `name` on every layer, or on the layers given."""
        which = (range(self.nlayers) if layers is None
                 else [self._geometry.layer(w).index for w in layers])
        out = [dict(layer.fields) for layer in self._layers]
        for i in which:
            out[i].pop(name, None)
        return self.replaced(layers=out, check=False)

    def with_geometry(self, geometry: Geometry) -> "Model":
        """A copy on another geometry over the same skeleton."""
        a, b = self.skeleton.boundaries, geometry.skeleton.boundaries
        if a.size != b.size or not np.allclose(a, b, rtol=self._geometry.rtol,
                                               atol=0.0):
            raise ValueError("the new geometry has another skeleton; use the "
                             "surgery methods to change the layering")
        return self.replaced(geometry=geometry)

    def renamed(self, *, layers: Renames = None, interfaces: Renames = None) -> "Model":
        return self.replaced(geometry=self._geometry.renamed(layers=layers,
                                                          interfaces=interfaces),
                          check=False)

    def with_mapping(self, mapping: Mapping, *, check: bool = True) -> "Model":
        return self.replaced(geometry=self._geometry.with_mapping(mapping, check=check),
                          check=False)

    def stretched(self, h: RadialDisplacement | SphericalFunction, *,
                  name: str | None = None, check: bool = True) -> "Model":
        return self.replaced(geometry=self._geometry.stretched(h, name=name,
                                                            check=check),
                          check=False)

    # -- surgery ------------------------------------------------------------

    def _carried(self, geometry: Geometry, *,
                 shells: Sequence[abc.Mapping[str, Field]] | None = None
                 ) -> list[dict[str, Field]]:
        """The fields of every layer of `geometry`, taken from this model.

        A new layer inside the old skeleton takes the old layer around
        its midpoint, re-stated on its interval; a layer outside takes
        the next mapping of `shells`.
        """
        old = self.skeleton
        out = []
        shells = list(shells or [])
        for lo, hi in (geometry.skeleton.interval(i) for i in range(geometry.nlayers)):
            mid = 0.5 * (lo + hi)
            if old.contains(mid, mid, rtol=geometry.rtol):
                src = self._layers[old.locate(mid).layer]
                out.append({name: (f if same_interval(f.interval, (lo, hi),
                                                       rtol=geometry.rtol)
                                   else f.on_interval(lo, hi))
                            for name, f in src.fields.items()})
            else:
                out.append(dict(shells.pop(0)) if shells else {})
        return out

    def refined(self, radii: ArrayLike, *, names: Names = None) -> "Model":
        """Interior boundaries inserted; a split layer's fields on each part."""
        g = self._geometry.refined(radii, names=names)
        return self.replaced(geometry=g, layers=self._carried(g))

    def truncated(self, radius: float, *, name: str | None = None) -> "Model":
        """The model cut at `radius`; the cut layer's fields on its remainder."""
        g = self._geometry.truncated(radius, name=name)
        return self.replaced(geometry=g, layers=self._carried(g))

    def hollowed(self, radius: float, *, name: str | None = None) -> "Model":
        """The model cut from below at `radius`."""
        g = self._geometry.hollowed(radius, name=name)
        return self.replaced(geometry=g, layers=self._carried(g))

    def extended(self, radii: ArrayLike, *,
                 fields: str | Sequence[abc.Mapping[str, Field]] | None = None,
                 names: Names = None, interface_names: Names = None) -> "Model":
        """Shells appended outside, holding the fields given.

        `fields` is None for empty shells, "extrapolate" to re-state the
        outermost layer's fields on each shell by their own rule, or one
        mapping of name to field per shell.
        """
        g = self._geometry.extended(radii, names=names,
                                    interface_names=interface_names)
        n_new = g.nlayers - self.nlayers
        if fields is None:
            shells = [{} for _ in range(n_new)]
        elif isinstance(fields, str):
            if fields != "extrapolate":
                raise ValueError(f"fields must be None, 'extrapolate' or "
                                 f"mappings, got {fields!r}")
            outer = self._layers[-1].fields
            shells = [{name: f.on_interval(*g.skeleton.interval(self.nlayers + j))
                       for name, f in outer.items()} for j in range(n_new)]
        else:
            shells = [dict(m) for m in fields]
            if len(shells) != n_new:
                raise ValueError(f"got fields for {len(shells)} shells, "
                                 f"expected {n_new}")
        return self.replaced(geometry=g, layers=self._carried(g, shells=shells))

    # -- units --------------------------------------------------------------

    def converted(self, scales: Scales) -> "Model":
        """The model in other scales: every length and every field by name."""
        if not isinstance(scales, Scales):
            raise TypeError(f"expected Scales, got {type(scales).__name__}")
        old, new = self._scales, scales
        k = old.factor(LENGTH) / new.factor(LENGTH)
        layers = []
        for layer in self._layers:
            out = {}
            for name, f in layer.fields.items():
                spec = self._specs.get(name)
                if spec is None or spec.dimensions is None:
                    raise ValueError(
                        f"{name!r} on layer {layer._label()} has no dimensions, "
                        "so its numbers cannot be converted: give it a spec or "
                        "drop it with without_field")
                v = old.factor(spec.dimensions) / new.factor(spec.dimensions)
                out[name] = f.rescaled(k=k, v=v)
            layers.append(out)
        return self.replaced(geometry=self._geometry.scaled(k), layers=layers,
                          scales=new)

    def nondimensionalised(self, *, density: float = EARTH_MEAN_DENSITY,
                           length: float | None = None) -> "Model":
        """The model in geophysical scales, with G equal to one.

        `length` is the outer radius by default; the model must be in SI.
        """
        if not self._scales.is_si:
            raise ValueError("nondimensionalised expects a model in SI; "
                             "convert with in_si first")
        if length is None:
            length = float(self.skeleton.boundaries[-1])
        return self.converted(Scales.geophysical(length, density=density))

    def in_si(self) -> "Model":
        return self.converted(Scales.SI)

    def __repr__(self) -> str:
        names = ", ".join(self.field_names())
        return (f"{type(self).__name__}({self.nlayers} layers, fields [{names}], "
                f"scales={self._scales!r})")
