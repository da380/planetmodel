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
"""
from __future__ import annotations

from types import MappingProxyType
from typing import Mapping

import numpy as np

from .fields import Field
from .geometry import Geometry, LayerInfo
from .layerfunction import same_interval
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

    def __init__(self, info: LayerInfo, fields: Mapping[str, Field]) -> None:
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
    def fields(self) -> Mapping[str, Field]:
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

    def __contains__(self, name) -> bool:
        return name in self._fields

    def __iter__(self):
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

    def __init__(self, geometry, layers, *, scales: Scales = Scales.SI,
                 specs=None, constants=None, check: bool = True) -> None:
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
    def specs(self) -> Mapping[str, FieldSpec]:
        """Every name with a meaning: the vocabulary, then `specs=`."""
        return self._specs

    @property
    def constants(self) -> Mapping[str, Constant]:
        return self._constants

    def layer(self, which) -> Layer:
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

    def _copy(self, *, geometry=None, layers=None, scales=None,
              check: bool = True) -> "Model":
        """A copy of the same class; a subclass keeps the constructor's signature."""
        return type(self)(
            self._geometry if geometry is None else geometry,
            [layer.fields for layer in self._layers] if layers is None else layers,
            scales=self._scales if scales is None else scales,
            specs=self._specs, constants=self._constants, check=check)

    def with_field(self, which, name: str, field, *, replace: bool = False) -> "Model":
        """A copy with `field` attached to one layer under `name`."""
        i = self._geometry.layer(which).index
        layers = [dict(layer.fields) for layer in self._layers]
        if name in layers[i] and not replace:
            raise ValueError(
                f"layer {self._layers[i]._label()} already holds {name!r}; "
                "pass replace=True to replace it")
        layers[i][name] = field
        return self._copy(layers=layers)

    def without_field(self, name: str, *, layers=None) -> "Model":
        """A copy without `name` on every layer, or on the layers given."""
        which = (range(self.nlayers) if layers is None
                 else [self._geometry.layer(w).index for w in layers])
        out = [dict(layer.fields) for layer in self._layers]
        for i in which:
            out[i].pop(name, None)
        return self._copy(layers=out, check=False)

    def with_geometry(self, geometry) -> "Model":
        """A copy on another geometry over the same skeleton."""
        a, b = self.skeleton.boundaries, geometry.skeleton.boundaries
        if a.size != b.size or not np.allclose(a, b, rtol=self._geometry.rtol,
                                               atol=0.0):
            raise ValueError("the new geometry has another skeleton; use the "
                             "surgery methods to change the layering")
        return self._copy(geometry=geometry)

    def renamed(self, *, layers=None, interfaces=None) -> "Model":
        return self._copy(geometry=self._geometry.renamed(layers=layers,
                                                          interfaces=interfaces),
                          check=False)

    def with_mapping(self, mapping, *, check: bool = True) -> "Model":
        return self._copy(geometry=self._geometry.with_mapping(mapping, check=check),
                          check=False)

    def stretched(self, h, *, name: str | None = None,
                  check: bool = True) -> "Model":
        return self._copy(geometry=self._geometry.stretched(h, name=name,
                                                            check=check),
                          check=False)

    # -- surgery ------------------------------------------------------------

    def _carried(self, geometry: Geometry, *, shells=None) -> list[dict]:
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

    def refined(self, radii, *, names=None) -> "Model":
        """Interior boundaries inserted; a split layer's fields on each part."""
        g = self._geometry.refined(radii, names=names)
        return self._copy(geometry=g, layers=self._carried(g))

    def truncated(self, radius, *, name: str | None = None) -> "Model":
        """The model cut at `radius`; the cut layer's fields on its remainder."""
        g = self._geometry.truncated(radius, name=name)
        return self._copy(geometry=g, layers=self._carried(g))

    def hollowed(self, radius, *, name: str | None = None) -> "Model":
        """The model cut from below at `radius`."""
        g = self._geometry.hollowed(radius, name=name)
        return self._copy(geometry=g, layers=self._carried(g))

    def extended(self, radii, *, fields=None, names=None,
                 interface_names=None) -> "Model":
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
        return self._copy(geometry=g, layers=self._carried(g, shells=shells))

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
        return self._copy(geometry=self._geometry.scaled(k), layers=layers,
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
