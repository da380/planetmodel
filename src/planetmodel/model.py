"""A model: a geometry with a bag of fields on every layer.

A model holds a geometry, one
mapping of name to field per layer, the `Scales` its numbers are in,
and the specs and constants that say what the names mean.  Everything
a field is asked goes through its layer: `model.layer(i)["rho"]`, or
`model.layer("mantle")["rho"]`.  A discontinuity is two layers asked
separately.  `ModelBase` is all of that; a leaf names the geometry it
is built on: `Model` a spherical `Geometry`, `planetmodel.cartesian.Model`
a flat one.  Every field is in the geometry's coordinates.

What a name means comes from the vocabulary, or from `specs=` for a
name outside it; a field attached under a name with a spec must have
the spec's character, and a name with no spec is accepted, carries no
dimensions, and is refused by name on conversion.  Nothing is required:
the function that needs a field asks the layer for it.  Behaviour
shared by groups of models is a free function of a layer or a model.

Units live here and nowhere else.  A field is numbers; the model's
`scales` say what one stored unit of each base dimension is in SI,
and `converted` re-expresses the whole model, geometry included, in
other scales, by name and exactly for polynomial layers.  A constant is
read in the model's units, `G` included.

Surgery goes through the geometry's own and carries the fields: a
split layer's fields are re-stated on each part, a cut layer's on its
remainder, and appended shells hold the fields they are given, the last
layer's by its own rule, or the last layer's held at their values on
the last boundary (`fields.held`), which pads a box below its base.
Every change is a copy, and every change that can break the model's
invariants (its fields, its layering) validates the copy again.

A model cites the works it rests on (`citation`, one work a string;
`cite` gives them a line each): PREM its paper, a published model what
its source asks to be cited.  Every copy keeps them, and a model built
from others adds theirs (`with_citation`), so that a model of PREM with
S40RTS on it, written to a file and read back, still says what to cite.

A model type is a class derived from one leaf alone (`Model` on the
sphere), plus the stateless
behaviour mixins of `planetmodel.behaviours` that wrap the free
functions as methods; there is no hierarchy of model types.  Every copy
goes through `replaced`, a shallow copy with the changed parts swapped
in, never through `__init__`, so a model type may give itself whatever
constructor its construction needs and every surgery, conversion and
freezing returns an instance of the same class carrying the same
instance state.  `behaviours` names the mixins of a model's class, which
a model file states so that a reader gives them back, and `restored` is
how a reader makes a model of a type from the parts a file holds, with
no constructor called.
"""
from __future__ import annotations

import copy
from collections import abc
from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .character import Character
from .coordinates import Coordinates, coordinates_of
from .displacement import RadialDisplacement, SphericalFunction
from .fields import Field, held
from .geometry import Geometry, GeometryBase, LayerInfo, Names, Renames
from .layerfunction import LayerFunction, polynomial_fit, same_interval
from .mapping import Mapping
from .materials import is_fluid
from .skeleton import CoarseningMap, Skeleton
from .units import EARTH_MEAN_DENSITY, LENGTH, Scales
from .vocabulary import CONSTANTS, VOCABULARY, Constant, FieldSpec, family_spec

__all__ = ["Layer", "ModelBase", "Model", "CoarseningFit", "cited", "restored"]

#: What `extended` restates the last layer's fields by on the new layers.
_EXTENSIONS = ("extrapolate", "hold")


def cited(works: Iterable[str], more: Iterable[str]) -> tuple[str, ...]:
    """`works` followed by those of `more` they lack, each stripped, the
    empty ones dropped, a work one line: how a model, a data set's surface,
    strata or grid, and whatever is built from several of them, gather
    what they cite."""
    if isinstance(more, str) or isinstance(works, str):
        raise TypeError("a citation is a sequence of works, not one string")
    out: list[str] = []
    for w in (*works, *more):
        if not isinstance(w, str):
            raise TypeError(f"a work cited is a string, got {w!r}")
        w = w.strip()
        if "\n" in w or "\r" in w:
            raise ValueError(f"a work cited is one line, got {w!r}")
        if w and w not in out:
            out.append(w)
    return tuple(out)


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
        """The layer's position in the skeleton's order: 0 at the centre of
        a ball, at the top of a box."""
        return self._info.index

    @property
    def interval(self) -> tuple[float, float]:
        """The (lo, hi) span of the layered coordinate the layer occupies:
        of radius on the sphere, of depth in a box."""
        return self._info.interval

    @property
    def name(self) -> str | None:
        """The layer's name, or None."""
        return self._info.name

    @property
    def info(self) -> LayerInfo:
        """The geometry's record of this layer: index, interval, name."""
        return self._info

    @property
    def fields(self) -> abc.Mapping[str, Field]:
        """The layer's fields by name, read-only."""
        return self._fields

    @property
    def names(self) -> tuple[str, ...]:
        """The names of the fields this layer holds."""
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


@dataclass(frozen=True)
class CoarseningFit:
    """What a model coarsening did: the map, and each fit's misfit.

    `residuals` maps (coarse layer index, field name) to the fit's
    relative root-mean-square misfit against the fine model over the
    merged interval, the absolute RMS where the data vanish, measured
    on a mesh four times finer than the fit's nodes.  A layer left
    whole keeps its fields and appears in no entry.
    """

    map: CoarseningMap
    residuals: abc.Mapping[tuple[int, str], float]

    @property
    def worst(self) -> float:
        """The largest residual; zero where nothing was fitted."""
        return max(self.residuals.values(), default=0.0)

    def describe(self) -> str:
        """One line per fit, the worst first."""
        lines = [f"  layer {j} {name!r}: {res:.3g}"
                 for (j, name), res in sorted(self.residuals.items(),
                                              key=lambda kv: -kv[1])]
        return "\n".join([repr(self)] + lines)

    def __repr__(self) -> str:
        return (f"CoarseningFit({self.map.fine.nlayers} -> "
                f"{self.map.coarse.nlayers} layers, {len(self.residuals)} "
                f"fits, worst {self.worst:.3g})")


def _misfit(data: Callable[[np.ndarray], np.ndarray], fitted: LayerFunction,
            interval: tuple[float, float], n: int) -> float:
    """The fit's relative RMS misfit against the data on the interval,
    measured at 4 n equispaced radii; absolute where the data vanish."""
    lo, hi = interval
    r = np.linspace(lo, hi, 4 * n)
    y = np.asarray(data(r))
    err = float(np.sqrt(np.mean(np.abs(np.asarray(fitted(r)) - y) ** 2)))
    scale = float(np.sqrt(np.mean(np.abs(y) ** 2)))
    return err / scale if scale > 0.0 else err


def _qualified(cls: type) -> str:
    """A class by its module and name, which tells two leaves apart."""
    return f"{cls.__module__}.{cls.__qualname__}"


class ModelBase:
    """A geometry with fields on every layer; see the module docstring.

    `layers` is one mapping of name to field per geometry layer, empty
    where a layer has no material.  `specs` gives a `FieldSpec` for
    names outside the vocabulary (or overriding it) and `constants` a
    `Constant` beyond the shipped ones.  `name` is what the model is
    called, in its repr and in the files that write it; it survives
    every copy.  `citation` holds the works the model rests on, as the
    module docstring says.  `check=False` skips validation for fields
    already known to fit.  A leaf names the class of geometry it is
    built on as `geometry_type`; any other is refused.
    """

    #: The class of geometry this leaf is built on.
    geometry_type: ClassVar[type[GeometryBase]] = GeometryBase

    #: The behaviours a file named for the model, kept by `restored`;
    #: empty for a model made any other way.
    _named_behaviours: tuple[str, ...] = ()

    def __init__(self, geometry: GeometryBase,
                 layers: Iterable[abc.Mapping[str, Field]], *,
                 scales: Scales = Scales.SI,
                 specs: abc.Mapping[str, FieldSpec] | None = None,
                 constants: abc.Mapping[str, Constant] | None = None,
                 name: str | None = None, citation: Iterable[str] = (),
                 check: bool = True) -> None:
        self._require_geometry(geometry)
        layers = [dict(m) for m in layers]
        if len(layers) != geometry.nlayers:
            raise ValueError(
                f"got fields for {len(layers)} layers; the geometry has "
                f"{geometry.nlayers}")
        if not isinstance(scales, Scales):
            raise TypeError(f"expected Scales, got {type(scales).__name__}")
        self._geometry = geometry
        self._scales = scales
        self._name = None if name is None else str(name)
        self._citation = cited((), citation)
        self._specs = MappingProxyType({**VOCABULARY, **dict(specs or {})})
        self._constants = MappingProxyType({**CONSTANTS, **dict(constants or {})})
        self._layers = tuple(Layer(info, fields)
                             for info, fields in zip(geometry.layers, layers))
        if check:
            self.validate()

    # -- validation ---------------------------------------------------------

    def _require_geometry(self, geometry: object) -> None:
        """Refuse a geometry that is not of the leaf's class, naming both."""
        if not isinstance(geometry, self.geometry_type):
            raise TypeError(f"expected a {_qualified(self.geometry_type)}, got "
                            f"{_qualified(type(geometry))}")

    def validate(self) -> None:
        """Every field a Field on its layer's interval, in the geometry's
        coordinates, with its spec's character."""
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
                system = coordinates_of(field)
                if system != self._geometry.coordinates:
                    raise ValueError(
                        f"{name!r} on layer {layer._label()} is in the coordinates "
                        f"{system.name!r}, the geometry in "
                        f"{self._geometry.coordinates.name!r}")
                spec = self.spec(name)
                if spec is not None and field.character != spec.character:
                    raise ValueError(
                        f"{name!r} on layer {layer._label()} has character "
                        f"{field.character}; its spec says {spec.character}")

    # -- what it is ---------------------------------------------------------

    @property
    def geometry(self) -> GeometryBase:
        """The geometry the fields hang on."""
        return self._geometry

    @property
    def skeleton(self) -> Skeleton:
        """The geometry's boundaries: radii on the sphere, depths in a box."""
        return self._geometry.skeleton

    @property
    def coordinates(self) -> Coordinates:
        """The coordinate system of the geometry and of every field:
        `SPHERICAL` on the sphere, `cartesian.FLAT` in a box."""
        return self._geometry.coordinates

    @property
    def scales(self) -> Scales:
        """What one stored unit of each base dimension is in SI."""
        return self._scales

    @property
    def name(self) -> str | None:
        """What the model is called, or None."""
        return self._name

    @property
    def citation(self) -> tuple[str, ...]:
        """The works the model rests on, one a string, in the order added."""
        return self._citation

    @property
    def behaviours(self) -> tuple[str, ...]:
        """The names of the model's behaviours: the mixins of its class
        (`planetmodel.behaviours`, or a model type's own), in the order of
        its bases, which a model file states so that a reader gives them
        back.  A model read from a file that names a behaviour the
        reading program lacks has the methods of the others and keeps the
        file's names whole, that one in its place, so that it is written
        again as it was read."""
        return self._named_behaviours or tuple(
            c.__name__ for c in type(self).__mro__
            if c is not object and not issubclass(c, ModelBase))

    def cite(self) -> str:
        """The works the model rests on, one a line: `print(model.cite())`.
        Empty for a model that records none."""
        return "\n".join(self._citation)

    def with_citation(self, *works: str) -> "ModelBase":
        """A copy citing `works` too, after its own, each work once."""
        out = self.replaced(check=False)
        out._citation = cited(self._citation, works)
        return out

    @property
    def nlayers(self) -> int:
        """The number of layers."""
        return len(self._layers)

    @property
    def layers(self) -> tuple[Layer, ...]:
        """The layers in the skeleton's order, each a bag of fields: centre
        outward on the sphere, from the surface down in a box."""
        return self._layers

    @property
    def specs(self) -> abc.Mapping[str, FieldSpec]:
        """Every name with a meaning: the vocabulary, then `specs=`."""
        return self._specs

    @property
    def constants(self) -> abc.Mapping[str, Constant]:
        """The declared constants by name; `constant` reads their values."""
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
        """The spec of a name, the model's own or the vocabulary's (a
        family's for a member of one), or None when it has none."""
        own = self._specs.get(name)
        return own if own is not None else family_spec(name)

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

    def replaced(self, *, geometry: GeometryBase | None = None,
                 layers: Sequence[abc.Mapping[str, Field]] | None = None,
                 scales: Scales | None = None,
                 specs: abc.Mapping[str, FieldSpec] | None = None,
                 constants: abc.Mapping[str, Constant] | None = None,
                 name: str | None = None, citation: Iterable[str] | None = None,
                 check: bool = True) -> "ModelBase":
        """A shallow copy with the given parts replaced, validated unless
        `check=False`: the one path every copy of a model takes.

        `layers` is one mapping of name to field per layer of the (new)
        geometry; `specs` and `constants` replace the model's own
        tables, the vocabulary and the shipped constants being merged in
        again; `citation` replaces the works it cites.  The class and
        every other instance attribute are kept, the name and the
        citation included unless given others.
        """
        out = copy.copy(self)
        if geometry is not None:
            self._require_geometry(geometry)
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
        if name is not None:
            out._name = str(name)
        if citation is not None:
            out._citation = cited((), citation)
        if check:
            out.validate()
        return out

    def named(self, name: str | None) -> "ModelBase":
        """A copy called `name`; None unnames it."""
        out = self.replaced(check=False)
        out._name = None if name is None else str(name)
        return out

    def with_field(self, which: int | str, name: str, field: Field, *,
                   replace: bool = False) -> "ModelBase":
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
                      layers: Iterable[int | str] | None = None) -> "ModelBase":
        """A copy without `name` on every layer, or on the layers given."""
        which = (range(self.nlayers) if layers is None
                 else [self._geometry.layer(w).index for w in layers])
        out = [dict(layer.fields) for layer in self._layers]
        for i in which:
            out[i].pop(name, None)
        return self.replaced(layers=out, check=False)

    def with_geometry(self, geometry: GeometryBase) -> "ModelBase":
        """A copy on another geometry over the same skeleton."""
        a, b = self.skeleton.boundaries, geometry.skeleton.boundaries
        if a.size != b.size or not np.allclose(a, b, rtol=self._geometry.rtol,
                                               atol=0.0):
            raise ValueError("the new geometry has another skeleton; use the "
                             "surgery methods to change the layering")
        return self.replaced(geometry=geometry)

    def renamed(self, *, layers: Renames = None,
                interfaces: Renames = None) -> "ModelBase":
        """The model with layers or interfaces renamed; see
        `GeometryBase.renamed`."""
        return self.replaced(geometry=self._geometry.renamed(layers=layers,
                                                          interfaces=interfaces),
                          check=False)

    def with_mapping(self, mapping: Mapping, *, check: bool = True) -> "ModelBase":
        """The model placed by `mapping`, over the same skeleton."""
        return self.replaced(geometry=self._geometry.with_mapping(mapping, check=check),
                          check=False)

    # -- surgery ------------------------------------------------------------

    def _carried(self, geometry: GeometryBase, *,
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

    def refined(self, radii: ArrayLike, *,
                interface_names: Names = None) -> "ModelBase":
        """Interior boundaries inserted, named by `interface_names`; a split
        layer's fields on each part."""
        g = self._geometry.refined(radii, interface_names=interface_names)
        return self.replaced(geometry=g, layers=self._carried(g))

    def truncated(self, radius: float, *, name: str | None = None) -> "ModelBase":
        """The model cut at `radius`, the part beyond it dropped (outside a
        ball, below a box); the cut layer's fields on its remainder."""
        g = self._geometry.truncated(radius, name=name)
        return self.replaced(geometry=g, layers=self._carried(g))

    def hollowed(self, radius: float, *, name: str | None = None) -> "ModelBase":
        """The model cut at `radius`, the part before it dropped (the
        centre of a ball, the top of a box); the cut layer's fields on its
        remainder."""
        g = self._geometry.hollowed(radius, name=name)
        return self.replaced(geometry=g, layers=self._carried(g))

    def extended(self, radii: ArrayLike, *,
                 fields: str | Sequence[abc.Mapping[str, Field]] | None = None,
                 layer_names: Names = None,
                 interface_names: Names = None) -> "ModelBase":
        """Layers appended beyond the last boundary (outside a ball, below a
        box), holding the fields given.

        `fields` is None for empty layers; "extrapolate" to re-state the
        last layer's fields on each new layer by their own rule; "hold" to
        hold them at their values on the last boundary, constant along the
        layered coordinate (`fields.held`, exact for a radial field), which
        pads a box below its base with the lowest properties; or one
        mapping of name to field per new layer.  `layer_names` name the
        new layers and `interface_names` the interfaces they bring.
        """
        g = self._geometry.extended(radii, layer_names=layer_names,
                                    interface_names=interface_names)
        n_new = g.nlayers - self.nlayers
        if fields is None:
            shells = [{} for _ in range(n_new)]
        elif isinstance(fields, str):
            if fields not in _EXTENSIONS:
                raise ValueError(f"fields must be None, one of {_EXTENSIONS} or "
                                 f"mappings, got {fields!r}")
            last = self._layers[-1].fields
            at = float(self.skeleton.boundaries[-1])
            intervals = [g.skeleton.interval(self.nlayers + j) for j in range(n_new)]
            if fields == "extrapolate":
                shells = [{name: f.on_interval(*iv) for name, f in last.items()}
                          for iv in intervals]
            else:
                shells = [{name: held(iv, f, at) for name, f in last.items()}
                          for iv in intervals]
        else:
            shells = [dict(m) for m in fields]
            if len(shells) != n_new:
                raise ValueError(f"got fields for {len(shells)} shells, "
                                 f"expected {n_new}")
        return self.replaced(geometry=g, layers=self._carried(g, shells=shells))

    def coarsened(self, *, keep: Iterable[int | str] | None = None,
                  drop: Iterable[int | str] | None = None, degree: int = 3,
                  n: int = 64) -> tuple["ModelBase", CoarseningFit]:
        """Interior interfaces removed, each merged layer's fields refit.

        Exactly one of `keep` and `drop` selects interior interfaces by
        name or index; the outermost interface (the surface of a ball or
        of a box), and the inner boundary of a hollow ball, are not
        selectable.  A layer left whole
        keeps its field objects.  A merged layer holds, for every name
        all its fine layers carry, the least-squares polynomial of
        `degree` through `n` Chebyshev samples of the fine model, a
        field of the geometry's `radial_field_type`, and
        the returned CoarseningFit reports each fit's relative misfit
        for the caller to judge.  Merging a fluid layer with a solid
        one, a name only some of the merged layers hold, and a field
        that is not a scalar of the layered coordinate alone are refused
        by name.
        The identity mapping is required, as for every surgery; the
        fits are independent, so derived fields (an attached gravity, a
        modulus beside its velocities) are more consistently derived
        again from the coarse model.
        """
        geometry, cmap = self._geometry.coarsened(keep=keep, drop=drop)
        layers: list[dict[str, Field]] = []
        residuals: dict[tuple[int, str], float] = {}
        for j, group in enumerate(cmap.layers):
            if len(group) == 1:
                layers.append(dict(self._layers[group[0]].fields))
                continue
            self._require_mergeable(group)
            interval = geometry.skeleton.interval(j)
            fields: dict[str, Field] = {}
            for name in self._merged_names(group):
                character = self._one_character(group, name)
                data = self._across(group, name)
                fitted = polynomial_fit(interval, data, degree=degree, n=n)
                fields[name] = geometry.radial_field_type(interval, fitted,
                                                          character=character,
                                                          name=name)
                residuals[(j, name)] = _misfit(data, fitted, interval, n)
            layers.append(fields)
        coarse = self.replaced(geometry=geometry, layers=layers)
        return coarse, CoarseningFit(cmap, MappingProxyType(residuals))

    def _require_mergeable(self, group: Sequence[int]) -> None:
        """Refuse a merge across the fluid-solid character."""
        fluidity = {}
        for i in group:
            try:
                fluidity[i] = is_fluid(self._layers[i])
            except KeyError:
                continue
        if True in fluidity.values() and False in fluidity.values():
            fluid = [i for i, f in fluidity.items() if f]
            solid = [i for i, f in fluidity.items() if not f]
            raise ValueError(
                f"cannot merge fluid layers {fluid} with solid layers "
                f"{solid}: keep the interface between them")

    def _merged_names(self, group: Sequence[int]) -> tuple[str, ...]:
        """Every name all layers of `group` hold, in the first layer's
        order; a name only some hold is refused."""
        held = [set(self._layers[i].names) for i in group]
        common = set.intersection(*held)
        partial = sorted(set.union(*held) - common)
        if partial:
            raise ValueError(
                f"cannot merge layers {list(group)}: {partial} are held on "
                "some of them and not others; drop them first, or keep the "
                "interface")
        return tuple(nm for nm in self._layers[group[0]].names if nm in common)

    def _one_character(self, group: Sequence[int], name: str) -> Character:
        """The one character `name` has across `group`, each layer's
        field a scalar of the layered coordinate alone; anything else is
        refused."""
        characters = set()
        for i in group:
            f = self._layers[i][name]
            if f.character.rank != 0 or not getattr(f, "is_radial", False):
                raise ValueError(
                    f"{name!r} on layer {self._layers[i]._label()} is not a "
                    f"scalar of the {self._geometry.coordinates.labels[0]} alone, "
                    "so it cannot be refit; drop it first, or keep the interface")
            characters.add(f.character)
        if len(characters) != 1:
            raise ValueError(
                f"{name!r} has characters {sorted(map(str, characters))} "
                f"across layers {list(group)}; one field holds one character")
        return characters.pop()

    def _across(self, group: Sequence[int], name: str
                ) -> Callable[[np.ndarray], np.ndarray]:
        """`name` of the fine model as one function of r over `group`,
        each radius answered by the fine layer containing it."""
        b = self.skeleton.boundaries
        lo_i, hi_i = int(group[0]), int(group[-1])

        def data(r: np.ndarray) -> np.ndarray:
            r = np.asarray(r, dtype=float)
            idx = np.clip(np.searchsorted(b, r, side="right") - 1, lo_i, hi_i)
            parts = {int(i): self._layers[int(i)][name](r[idx == i])
                     for i in np.unique(idx)}
            out = np.empty(r.shape, dtype=np.result_type(
                *(p.dtype for p in parts.values())))
            for i, vals in parts.items():
                out[idx == i] = vals
            return out

        return data

    # -- units --------------------------------------------------------------

    def converted(self, scales: Scales) -> "ModelBase":
        """The model in other scales: every length and every field by name."""
        if not isinstance(scales, Scales):
            raise TypeError(f"expected Scales, got {type(scales).__name__}")
        old, new = self._scales, scales
        k = old.factor(LENGTH) / new.factor(LENGTH)
        layers = []
        for layer in self._layers:
            out = {}
            for name, f in layer.fields.items():
                spec = self.spec(name)
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
                           length: float | None = None) -> "ModelBase":
        """The model in geophysical scales, with G equal to one.

        `length` is the skeleton's last boundary by default, the outer
        radius of a ball or the depth of a box's base; the model must be
        in SI.
        """
        if not self._scales.is_si:
            raise ValueError("nondimensionalised expects a model in SI; "
                             "convert with in_si first")
        if length is None:
            length = float(self.skeleton.boundaries[-1])
        return self.converted(Scales.geophysical(length, density=density))

    def in_si(self) -> "ModelBase":
        """The model converted to SI scales."""
        return self.converted(Scales.SI)

    def __repr__(self) -> str:
        names = ", ".join(self.field_names())
        called = f"{self._name!r}, " if self._name is not None else ""
        return (f"{type(self).__name__}({called}{self.nlayers} layers, "
                f"fields [{names}], scales={self._scales!r})")


class Model(ModelBase):
    """A spherically layered model: a `ModelBase` on a spherical
    `Geometry`, placed by any mapping or by the radial stretch
    (`stretched`); see the module docstring.  Every model type of the
    sphere (`PREM`, a user's own) derives from it.
    """

    geometry_type: ClassVar[type[GeometryBase]] = Geometry

    def stretched(self, h: RadialDisplacement | SphericalFunction, *,
                  name: str | None = None, check: bool = True) -> "Model":
        """The model under the radial stretch driven by `h`; see
        `Geometry.stretched`."""
        return self.replaced(geometry=self._geometry.stretched(h, name=name,
                                                            check=check),
                             check=False)


def restored(cls: type[ModelBase], geometry: GeometryBase,
             layers: Iterable[abc.Mapping[str, Field]], *,
             scales: Scales = Scales.SI,
             specs: abc.Mapping[str, FieldSpec] | None = None,
             constants: abc.Mapping[str, Constant] | None = None,
             name: str | None = None, citation: Iterable[str] = (),
             behaviours: Iterable[str] = (), check: bool = True) -> ModelBase:
    """A model of the type `cls` holding the parts given, made without the
    type's own constructor, as `replaced` makes a copy: what a reader makes
    of a file, which cannot call a constructor written for other arguments.

    The parts are `ModelBase`'s.  The model holds the fields given and no
    others, nothing being derived as a constructor may derive it (the Love
    moduli `behaviours.Elastic` completes), and none of the instance state
    the type's constructor sets.  `behaviours` are the names a file stated
    for the model, kept as `ModelBase.behaviours` gives them where they
    name one the type lacks; they name the mixins of `cls` in its order,
    with any others between.
    """
    if not (isinstance(cls, type) and issubclass(cls, ModelBase)):
        raise TypeError(f"expected a model type, got {cls!r}")
    out = object.__new__(cls)
    ModelBase.__init__(out, geometry, layers, scales=scales, specs=specs,
                       constants=constants, name=name, citation=citation, check=check)
    named = tuple(str(b) for b in behaviours)
    own = out.behaviours
    if named and tuple(b for b in named if b in own) != own:
        raise ValueError(f"the behaviours {named} do not name those of "
                         f"{cls.__name__}, {own}, in its order")
    out._named_behaviours = named
    return out
