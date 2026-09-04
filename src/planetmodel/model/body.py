"""body.py -- the reference body: layers, their fields, and annotations.

A ReferenceBody is a list of Layers.  Each Layer is an interval of
radius, the single-layer Fields it holds by name, and the annotations
that fields cannot carry: a name, and whether the layer is solid,
fluid or vacuum.  The Skeleton is the intervals laid end to end, and
the Interfaces between them carry names and a role.

A field belongs to one layer.  A body-wide
field, `body["rho"]`, is a *view* the body assembles from the pieces
its layers hold (`fields.layerwise.assemble`), defined on the layers
that have one -- its domain -- and refusing radii elsewhere.  Nothing
is zero-filled: a layer with no fields is empty, and says so.

The base class does no file IO; concrete constructors own their own
provenance (planetmodel.io.deck reads tabulated decks, planetmodel.catalogue.prem
builds the exact published polynomial model with no file at all).

All surgery is copy-on-write and is a list operation on the layers.
Nothing mutates a body in place except `add_field`, so the Skeleton a
set of fields was built against cannot change underneath them: that the
Skeleton is invariant under perturbation is a fact about the types
rather than a convention.
"""
from __future__ import annotations

import warnings
from collections.abc import Mapping as _MappingABC, Sequence

from dataclasses import KW_ONLY, dataclass, field as _dc_field, replace as _replace

import numpy as np

from .fields.base import Field
from .fields.layerwise import assemble, split
from .skeleton import CoarseningMap, Skeleton
from .displacement import as_displacement
from .mapping import IdentityMapping, Mapping, RadialStretch
from .surface import Surface
from .topography import CentredTopography, as_topography
from .units import EARTH_MEAN_DENSITY, Scales
from ..registry import register

__all__ = ["Layer", "Interface", "ReferenceBody", "fluid_where_vs_zero"]


STATES = ("solid", "fluid", "vacuum")
ROLES = ("material", "control")


def _same_interval(a, b, *, tol: float) -> bool:
    return (abs(float(a[0]) - float(b[0])) <= tol
            and abs(float(a[1]) - float(b[1])) <= tol)


@dataclass(frozen=True)
class Layer:
    """One shell: its interval, the fields it holds, and its annotations.

    `fields` maps names to single-layer Fields whose skeleton is this
    layer's interval.  `state` is the one classification a mesh needs
    and the fields cannot supply: whether the layer is solid, fluid or
    vacuum.  Vacuum is a *void* -- MMA26's buffer region, or the space
    outside a planet -- and holds no fields; a solid layer with no
    fields is a shell whose material a consumer will supply.

    A Layer is a value.  It does not know which body it is in, and
    every change returns a new one.
    """

    index: int
    _: KW_ONLY
    interval: tuple[float, float] | None = None
    name: str | None = None
    state: str = "solid"          # solid | fluid | vacuum
    fields: _MappingABC[str, Field] = _dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        """Reject annotations no consumer could interpret."""
        if self.state not in STATES:
            raise ValueError(
                f"state must be one of {STATES}, got {self.state!r}")
        if self.state == "vacuum" and self.fields:
            raise ValueError(
                f"a vacuum layer holds no fields; got {list(self.fields)}")
        if self.interval is not None:
            lo, hi = (float(x) for x in self.interval)
            if not hi > lo:
                raise ValueError(f"interval must increase, got {self.interval}")
            object.__setattr__(self, "interval", (lo, hi))
            for name, f in self.fields.items():
                self._check(name, f)
        object.__setattr__(self, "fields", dict(self.fields))

    def _check(self, name: str, f) -> None:
        """A field held by a layer lives on exactly that layer."""
        if not isinstance(f, Field):
            raise TypeError(
                f"field {name!r}: expected a Field (an object with skeleton, "
                f"character, name and evaluate); got {type(f).__name__}")
        sk = f.skeleton
        if sk.nlayers != 1:
            raise ValueError(
                f"field {name!r} spans {sk.nlayers} layers; a layer holds "
                "single-layer fields (index or restrict it first)")
        if self.interval is not None:
            lo, hi = self.interval
            if not _same_interval(sk.boundaries, (lo, hi), tol=1e-9 * (hi - lo)):
                b = sk.boundaries
                raise ValueError(
                    f"field {name!r} lives on [{b[0]:.6g}, {b[-1]:.6g}], not "
                    f"on this layer [{lo:.6g}, {hi:.6g}]")

    # -- what the layer has -----------------------------------------------

    @property
    def field_names(self) -> tuple[str, ...]:
        """The names of the fields held, in insertion order."""
        return tuple(self.fields)

    @property
    def is_vacuum(self) -> bool:
        """Whether the layer is a void."""
        return self.state == "vacuum"

    @property
    def is_fluid(self) -> bool:
        """Whether the layer supports no shear stress."""
        return self.state in ("fluid", "vacuum")

    def __getitem__(self, name: str) -> Field:
        """The field held under `name` (KeyError if absent)."""
        return self.fields[name]

    def __contains__(self, name: str) -> bool:
        return name in self.fields


    # -- copy-on-write ------------------------------------------------------

    def with_field(self, name: str, f: Field, *, replace: bool = False) -> "Layer":
        """A copy holding `f` under `name`."""
        if name in self.fields and not replace:
            raise ValueError(f"layer {self.index} already holds {name!r} "
                             "(pass replace=True)")
        if self.is_vacuum:
            raise ValueError(f"layer {self.index} is vacuum and holds no fields")
        self._check(name, f)
        return _replace(self, fields={**self.fields, name: f})

    def without_field(self, name: str) -> "Layer":
        """A copy without the named field."""
        return _replace(self, fields={k: v for k, v in self.fields.items()
                                      if k != name})

    def annotated(self, **kw) -> "Layer":
        """A copy with annotations changed (name, state)."""
        return _replace(self, **kw)

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        iv = ("" if self.interval is None
              else f", [{self.interval[0]:.4g}, {self.interval[1]:.4g}]")
        return (f"Layer({self.index}{nm}{iv}, {self.state}, "
                f"fields={list(self.fields)})")


@dataclass(frozen=True)
class Interface:
    """A boundary between layers, or the outer boundary."""

    index: int                     # 0 = innermost interior boundary
    _: KW_ONLY
    name: str | None = None
    radius: float = 0.0
    between: tuple[int, int] = (-1, -1)   # (below, above); -1 = none
    role: str = "material"         # material | control

    def __post_init__(self) -> None:
        if self.role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {self.role!r}")


@register("state_rule", "fluid_where_vs_zero")
def fluid_where_vs_zero(body, layer: int) -> str | None:
    """The default state rule: a layer with no shear velocity is fluid.

    Reads vsv, or vs, at the layer midpoint.  Verified to select
    exactly the outer core for PREM and for prem.nocrust.  A layer that
    holds no shear-velocity field is left alone (None: no opinion),
    since silence is not evidence of a fluid.
    """
    lay = body.layer(layer)
    name = next((n for n in ("vsv", "vs") if n in lay), None)
    if name is None:
        return None
    lo, hi = lay.interval
    mid = 0.5 * (lo + hi)
    return "fluid" if float(lay[name](mid)) == 0.0 else "solid"


class ReferenceBody:
    """A spherically layered body: Layers carrying fields, and annotations.

    `ReferenceBody(layers)` takes Layers whose intervals abut centre
    outward, each holding its single-layer fields; the skeleton is
    theirs.  `from_fields(skeleton, fields)` builds one from body-wide
    fields, splitting each into the pieces the layers hold.  Interfaces
    (names and roles for the boundaries), attached surfaces, `meta` and
    `scales` are keyword-only and default sensibly.

    A field is reached as a *view*: `body["rho"]` is one field on the
    body's skeleton assembled from the pieces its layers hold
    (`fields.layerwise.assemble`), defined on the layers that have one
    -- its `domain` -- and refusing radii elsewhere.  Nothing is
    zero-filled: a layer with no fields is empty, and says so.

    Surgery is copy-on-write and returns a body of the same class,
    re-validated by that class's constructor; `add_field` is the one
    operation that changes a body in place.

    `scales` DECLARES what the stored values are: Scales.SI (the
    default) says they are SI, anything else says they are already
    non-dimensional relative to those scales.  Construction never
    converts -- that is rescaled()'s job, and keeping declaration and
    conversion separate is what denies silently wrong numbers a way in.
    """

    def __init__(self, layers: Sequence[Layer], *, interfaces=None,
                 surfaces=None, meta: dict | None = None,
                 scales: Scales | None = None) -> None:
        self._layers, self._sk = self._layers_and_skeleton(tuple(layers))
        self._views: dict[str, Field] = {}
        self.scales = Scales.SI if scales is None else scales
        self.meta = dict(meta or {})
        self._interfaces = (self._default_interfaces() if interfaces is None
                            else tuple(interfaces))
        self._surfaces = dict(surfaces or {})

    @classmethod
    def from_layers(cls, layers: Sequence[Layer], **kw) -> "ReferenceBody":
        """A body from Layers carrying intervals and fields."""
        return cls(tuple(layers), **kw)

    @classmethod
    def from_fields(cls, skeleton: Skeleton, fields: _MappingABC[str, Field],
                    *, layers=None, **kw) -> "ReferenceBody":
        """A body from body-wide fields on a skeleton, split per layer.

        `layers` may annotate the shells (names, states); intervals they
        carry must agree with the skeleton's.  The fields are attached
        before the class validates, so a model class can be built this
        way too.
        """
        given = (tuple(Layer(index=i) for i in range(skeleton.nlayers))
                 if layers is None else tuple(layers))
        body = ReferenceBody(cls._placed(given, skeleton), **kw)
        for k, v in fields.items():
            body.add_field(k, v)
        return body if cls is ReferenceBody else body.as_class(cls)

    def as_class(self, cls, **kw) -> "ReferenceBody":
        """This body as `cls`, its layers, annotations and surfaces kept.

        `cls` is `ReferenceBody` or a model class; the class's
        constructor validates what it guarantees.  Keywords override
        `meta`, `interfaces`, `surfaces` or `scales`.
        """
        given = {"meta": dict(self.meta), "interfaces": self._interfaces,
                 "surfaces": dict(self._surfaces), "scales": self.scales}
        given.update(kw)
        return cls(self._layers, **given)

    # -- assembly -----------------------------------------------------------

    @staticmethod
    def _placed(layers: tuple[Layer, ...], sk: Skeleton) -> tuple[Layer, ...]:
        """Layers given against a skeleton: indexed, with their intervals."""
        if len(layers) != sk.nlayers:
            raise ValueError(
                f"got {len(layers)} layers for a skeleton of {sk.nlayers}")
        out = []
        for i, lay in enumerate(layers):
            lo, hi = sk.interval(i)
            if lay.interval is not None and not _same_interval(
                    lay.interval, (lo, hi), tol=1e-9 * (hi - lo)):
                raise ValueError(
                    f"layer {i} says it is [{lay.interval[0]:.6g}, "
                    f"{lay.interval[1]:.6g}], the skeleton says [{lo:.6g}, "
                    f"{hi:.6g}]")
            out.append(_replace(lay, index=i, interval=(lo, hi)))
        return tuple(out)

    @staticmethod
    def _layers_and_skeleton(layers: tuple[Layer, ...]
                             ) -> tuple[tuple[Layer, ...], Skeleton]:
        """Layers with intervals, abutting centre outward, and their skeleton."""
        if not layers:
            raise ValueError("a body needs at least one layer")
        b = []
        for k, lay in enumerate(layers):
            if lay.interval is None:
                raise ValueError(f"layer {k} has no interval")
            lo, hi = lay.interval
            if b and abs(lo - b[-1]) > 1e-9 * (hi - lo):
                raise ValueError(
                    f"layer {k} starts at {lo:.6g}, but the layer below ends "
                    f"at {b[-1]:.6g}: layers must abut")
            if not b:
                b.append(lo)
            b.append(hi)
        sk = Skeleton(b)
        return ReferenceBody._placed(layers, sk), sk

    def _default_interfaces(self) -> tuple[Interface, ...]:
        """One Interface per boundary above the centre, outward.

        The outer boundary is included -- it is where the exterior
        coupling lives -- and reports -1 as the layer above it.
        """
        b = self._sk.boundaries
        out = []
        for i in range(1, b.size):
            above = i if i < self._sk.nlayers else -1
            out.append(Interface(index=i - 1, radius=float(b[i]),
                                 between=(i - 1, above)))
        return tuple(out)

    def _rebuilt(self, *, layers, interfaces, meta=None, surfaces=None
                 ) -> "ReferenceBody":
        """A new body of this class from new layers and interfaces,
        re-validated by the class's constructor."""
        return type(self)(
            tuple(layers), meta=meta if meta is not None else dict(self.meta),
            interfaces=tuple(interfaces),
            surfaces=dict(self._surfaces if surfaces is None else surfaces),
            scales=self.scales)

    def validate(self) -> None:
        """What this class guarantees; nothing, for a plain body.

        A model class overrides it to check its fields layer by layer,
        and every constructor and `add_field` call it.
        """

    def _after_change(self) -> None:
        """Called after `add_field`; a model class re-validates here."""

    # -- annotations --------------------------------------------------------

    @property
    def layers(self) -> tuple[Layer, ...]:
        """Per-layer annotations and fields, centre outward."""
        return self._layers

    @property
    def interfaces(self) -> tuple[Interface, ...]:
        """Per-boundary annotations, centre outward (outer boundary last)."""
        return self._interfaces

    def layer(self, which) -> Layer:
        """A Layer by index or name."""
        return self._layers[self._resolve_layer(which)]

    def interface(self, which) -> Interface:
        """An Interface by index or name."""
        return self._interfaces[self._resolve_interface(which)]

    def _resolve_layer(self, which) -> int:
        """Layer index from an index or a name."""
        if isinstance(which, str):
            hits = [i for i, lay in enumerate(self._layers) if lay.name == which]
            if not hits:
                named = [lay.name for lay in self._layers if lay.name]
                raise KeyError(f"no layer named {which!r}; named layers: {named}")
            return hits[0]
        return self._sk.layer_index(int(which))

    def _resolve_interface(self, which) -> int:
        """Interface index from an index or a name."""
        if isinstance(which, str):
            hits = [i for i, f in enumerate(self._interfaces) if f.name == which]
            if not hits:
                named = [f.name for f in self._interfaces if f.name]
                raise KeyError(
                    f"no interface named {which!r}; named interfaces: {named}")
            return hits[0]
        i = int(which)
        n = len(self._interfaces)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"interface index {which} out of range for {n}")
        return i

    def annotate(self, which, **kw) -> "ReferenceBody":
        """A copy with one layer's annotations (name, state) changed."""
        i = self._resolve_layer(which)
        layers = list(self._layers)
        layers[i] = layers[i].annotated(**kw)
        return self._rebuilt(layers=layers, interfaces=self._interfaces)

    def with_layer(self, which, layer: Layer) -> "ReferenceBody":
        """A copy with one layer replaced; its interval must be the same."""
        i = self._resolve_layer(which)
        lo, hi = self._sk.interval(i)
        if layer.interval is not None and not _same_interval(
                layer.interval, (lo, hi), tol=1e-9 * (hi - lo)):
            raise ValueError(
                f"the replacement spans [{layer.interval[0]:.6g}, "
                f"{layer.interval[1]:.6g}], layer {i} is [{lo:.6g}, {hi:.6g}]")
        layers = list(self._layers)
        layers[i] = _replace(layer, index=i, interval=(lo, hi))
        return self._rebuilt(layers=layers, interfaces=self._interfaces)

    def with_field(self, which, name: str, f: Field, *, replace: bool = False
                   ) -> "ReferenceBody":
        """A copy with a single-layer field attached to one layer."""
        i = self._resolve_layer(which)
        return self.with_layer(i, self._layers[i].with_field(name, f,
                                                             replace=replace))

    def name_interface(self, which, name: str, *, role=None) -> "ReferenceBody":
        """A copy with one interface named, and optionally re-roled."""
        i = self._resolve_interface(which)
        faces = list(self._interfaces)
        kw = {"name": name} if role is None else {"name": name, "role": role}
        faces[i] = _replace(faces[i], **kw)
        return self._rebuilt(layers=self._layers, interfaces=tuple(faces))

    def classify_states(self, *, rule=fluid_where_vs_zero, overrides=None
                        ) -> "ReferenceBody":
        """A copy with each layer's state set by `rule`, overrides winning.

        `rule(body, layer) -> state | None` is any callable; the default
        reads the shear velocity at the layer midpoint and returns None
        for a layer holding none, which leaves that layer as it was: a
        layer with no evidence is not reclassified, and an empty shell
        awaiting a consumer's material is not read as a fluid.
        `overrides` maps a layer index or name to a state and always
        beats the rule, because a rule derived from tabulated numbers is
        evidence, not authority.  Vacuum layers are left alone: there
        is nothing there to classify.
        """
        overrides = overrides or {}
        resolved = {self._resolve_layer(k): v for k, v in overrides.items()}
        layers = []
        for i, lay in enumerate(self._layers):
            if lay.is_vacuum:
                layers.append(lay)
                continue
            state = resolved[i] if i in resolved else rule(self, i)
            layers.append(lay if state is None else _replace(lay, state=state))
        return self._rebuilt(layers=layers, interfaces=self._interfaces)

    @property
    def skeleton(self) -> Skeleton:
        """The Skeleton: the layers' intervals laid end to end."""
        return self._sk

    # -- fields ---------------------------------------------------------------

    @property
    def field_names(self) -> tuple[str, ...]:
        """Every field name held by some layer, in order of first appearance."""
        seen: dict[str, None] = {}
        for lay in self._layers:
            for name in lay.fields:
                seen.setdefault(name, None)
        return tuple(seen)

    def layers_with(self, name: str) -> tuple[int, ...]:
        """The layers holding a field of that name."""
        return tuple(i for i, lay in enumerate(self._layers) if name in lay.fields)

    def common_fields(self) -> tuple[str, ...]:
        """The names held by every non-vacuum layer."""
        return tuple(n for n in self.field_names
                     if all(n in lay.fields for lay in self._layers
                            if not lay.is_vacuum))

    def __getitem__(self, name: str) -> Field:
        """The body-wide view of a field: assembled from the layers' pieces.

        Defined on the layers that hold the name (its `domain`) and
        refusing radii elsewhere.  KeyError if no layer holds it.
        """
        try:
            return self._views[name]
        except KeyError:
            pass
        pieces = [lay.fields[name] for lay in self._layers if name in lay.fields]
        if not pieces:
            raise KeyError(name)
        view = assemble(self._sk, pieces, name=name)
        self._views[name] = view
        return view

    def __contains__(self, name: str) -> bool:
        """Whether some layer holds a field of that name."""
        return any(name in lay.fields for lay in self._layers)

    def __getattr__(self, name: str) -> Field:
        """Fallback attribute access into the field views.

        Declared attributes and methods win (this is only consulted
        when normal lookup fails); guarded so that unpickling and
        hasattr() probes never recurse.
        """
        try:
            layers = object.__getattribute__(self, "_layers")
        except AttributeError:
            raise AttributeError(name) from None
        if any(name in lay.fields for lay in layers):
            return self[name]
        raise AttributeError(
            f"{type(self).__name__} has no attribute or field {name!r}")

    def __dir__(self):
        """Standard attributes plus the field names, for tab completion."""
        return sorted(set(super().__dir__()) | set(self.field_names))

    def add_field(self, name: str, f: Field, *, replace: bool = False) -> Field:
        """Attach a field under `name` and return the body-wide view.

        A field on the body's skeleton is split into the pieces its
        layers hold, one per layer of its domain; a single-layer field
        is attached to the layer whose interval it spans.  Existing
        names are protected unless replace=True, in which case every
        layer's piece of that name is replaced (or dropped, where the
        new field has no piece).  Names that collide with an attribute
        are stored but reachable only via body[name], and a warning
        says so.  This is the one operation that changes a body in
        place.
        """
        if not isinstance(f, Field):
            raise TypeError(
                f"expected a Field (an object with skeleton, character, name "
                f"and evaluate); got {type(f).__name__}")
        if name in self and not replace:
            raise ValueError(f"field {name!r} exists (pass replace=True)")
        if hasattr(type(self), name) or name in self.__dict__:
            warnings.warn(f"field name {name!r} is shadowed by an existing "
                          f"attribute; reach it via model[{name!r}]")
        pieces = self._pieces_of(name, f)
        layers = []
        for i, lay in enumerate(self._layers):
            if i in pieces:
                layers.append(lay.with_field(name, pieces[i], replace=True))
            elif replace and name in lay.fields:
                layers.append(lay.without_field(name))
            else:
                layers.append(lay)
        self._layers = tuple(layers)
        self._views.pop(name, None)
        self._after_change()
        return self[name]

    def _pieces_of(self, name: str, f: Field) -> dict[int, Field]:
        """Where a field goes: one piece per layer it spans."""
        sk = f.skeleton
        if sk == self._sk:
            return dict(zip(f.domain, split(f)))
        if sk.nlayers == 1:
            b = sk.boundaries
            for i, lay in enumerate(self._layers):
                lo, hi = lay.interval
                if _same_interval(b, (lo, hi), tol=1e-9 * (hi - lo)):
                    return {i: f}
            raise ValueError(
                f"field {name!r} on [{b[0]:.6g}, {b[-1]:.6g}] matches no layer "
                f"of this body; its layers are "
                f"{[lay.interval for lay in self._layers]}")
        raise ValueError(
            f"field {name!r} lives on {sk!r}, not on this body's "
            f"{self._sk!r} or one of its layers")

    def without_field(self, name: str) -> "ReferenceBody":
        """A copy with the named field dropped from every layer."""
        if name not in self:
            raise KeyError(name)
        layers = [lay.without_field(name) if name in lay.fields else lay
                  for lay in self._layers]
        return self._rebuilt(layers=layers, interfaces=self._interfaces)

    # -- geometric surgery --------------------------------------------------

    @staticmethod
    def _clipped(lay: Layer, lo: float, hi: float, *, index: int,
                 name=None) -> Layer:
        """A layer's fields re-stated on [lo, hi] inside its interval."""
        fields = {k: f.on_interval(lo, hi) for k, f in lay.fields.items()}
        return _replace(lay, index=index, interval=(lo, hi), name=name,
                        fields=fields)

    def coarsened(self, *, keep=None, drop=None, state=None
                  ) -> tuple["ReferenceBody", CoarseningMap]:
        """A copy on a coarser skeleton, merging interior boundaries.

        The *fields keep their original resolution*: a merged layer
        holds, for each name every merged layer had, one field that
        answers from whichever fine piece contains the radius.
        Coarsening the mesh geometry is not coarsening the model.  A
        name held on only some of the merged layers is dropped from the
        merged layer, with a warning saying which.

        Layers of different `state` merge into the outermost one's,
        with a warning naming the states, unless `state` says what the
        merged layer is: a fluid and a solid make neither, and a mesh
        reads the answer.

        `keep` and `drop` index the *interior* boundaries, 0 to
        nlayers - 2 from the centre outward: the boundary between layers
        i and i + 1 is boundary i.  The outer boundary cannot be merged.
        """
        coarse, cmap = self._sk.coarsen(keep=keep, drop=drop)
        layers = []
        for i, fine in enumerate(cmap.layers):
            lo, hi = coarse.interval(i)
            if len(fine) == 1:
                layers.append(_replace(self._layers[fine[0]], index=i))
                continue
            parts = [self._layers[j] for j in fine]
            names = [n for n in parts[0].fields
                     if all(n in p.fields for p in parts)]
            lost = sorted({n for p in parts for n in p.fields} - set(names))
            if lost:
                warnings.warn(
                    f"coarsening layers {tuple(fine)} into one drops {lost}: "
                    "held on some of them but not all, so the merged layer "
                    "cannot carry them", UserWarning, stacklevel=2)
            fields = {n: assemble(Skeleton([lo, hi]), [p.fields[n] for p in parts],
                                  name=n) for n in names}
            states = {p.state for p in parts}
            if len(states) == 1:
                merged = states.pop()
            elif state is not None:
                merged = state
            else:
                merged = parts[-1].state
                warnings.warn(
                    f"coarsening layers {tuple(fine)} into one merges states "
                    f"{sorted(states)}; the merged layer is {merged!r}, the "
                    "outermost part's. Pass state= to say otherwise.",
                    UserWarning, stacklevel=2)
            layers.append(Layer(index=i, interval=(lo, hi), state=merged,
                                fields=fields))
        kept = set(cmap.kept_interfaces)
        faces = [f for i, f in enumerate(self._interfaces)
                 if i in kept or i == len(self._interfaces) - 1]
        faces = tuple(
            _replace(f, index=i, between=(i, i + 1 if i < len(faces) - 1 else -1))
            for i, f in enumerate(faces))
        return self._rebuilt(layers=layers, interfaces=faces), cmap

    def truncated(self, radius: float, *, name=None) -> "ReferenceBody":
        """A copy cut at `radius`, which becomes the new outer boundary."""
        sk = self._sk.truncated(radius)
        n = sk.nlayers
        layers = []
        for i in range(n):
            lay = self._layers[i]
            lo, hi = sk.interval(i)
            if _same_interval(lay.interval, (lo, hi), tol=1e-9 * (hi - lo)):
                layers.append(_replace(lay, index=i))
            else:
                layers.append(self._clipped(lay, lo, hi, index=i, name=lay.name))
        outer = float(sk.boundaries[-1])
        faces = list(self._interfaces[:n - 1])
        # Cutting exactly at an existing boundary keeps what it was called
        # and what it was for, unless the caller renames it.
        old = next((f for f in self._interfaces if f.radius == outer), None)
        faces.append(Interface(
            index=n - 1, radius=outer, between=(n - 1, -1),
            name=name if name is not None else (old.name if old else None),
            role=old.role if old else "material"))
        return self._rebuilt(layers=layers, interfaces=tuple(faces))

    def refined(self, radii, *, names=None, role="material") -> "ReferenceBody":
        """A copy with extra interior boundaries inserted.

        The material is unchanged: each new layer holds the fields of
        the layer it was cut from, re-stated on its part of it, so
        inserting a boundary is a purely geometric act.  `names` name
        the new *interfaces*, one per radius; the layers either side
        keep their names, and `annotate` renames a layer.  `role="control"`
        marks a boundary inserted only to shape a mapping (see
        displacement.py), which a consumer may merge across.
        """
        radii = list(np.atleast_1d(np.asarray(radii, dtype=float)))
        names = list(names or [None] * len(radii))
        if len(names) != len(radii):
            raise ValueError(f"got {len(names)} names for {len(radii)} radii")
        sk = self._sk.refined(radii)

        layers, faces = [], []
        b = sk.boundaries
        for i in range(sk.nlayers):
            src = self._sk.locate(0.5 * (b[i] + b[i + 1])).layers[-1]
            lay = self._layers[src]
            lo, hi = sk.interval(i)
            if _same_interval(lay.interval, (lo, hi), tol=1e-9 * (hi - lo)):
                layers.append(_replace(lay, index=i))
            else:
                layers.append(self._clipped(lay, lo, hi, index=i))
        for i in range(1, b.size):
            r = float(b[i])
            hit = [j for j, x in enumerate(radii) if x == r]
            if hit:
                faces.append(Interface(index=i - 1, name=names[hit[0]], radius=r,
                                       between=(i - 1, i if i < sk.nlayers else -1),
                                       role=role))
            else:
                old = next(f for f in self._interfaces if f.radius == r)
                faces.append(_replace(old, index=i - 1,
                                      between=(i - 1, i if i < sk.nlayers else -1)))
        return self._rebuilt(layers=layers, interfaces=tuple(faces))

    def extended(self, radii, *, fields=None, state="solid",
                 names=None, role="material") -> "ReferenceBody":
        """A copy with shells appended beyond the outer boundary.

        `fields` decides what the new shells hold:

          None           nothing: an empty shell (the default).  With
                         state="vacuum" it is a void; solid, it is a
                         shell whose material a consumer supplies.
          "extrapolate"  every field of the outermost layer, re-stated
                         on the new shell by its own rule of evaluation
                         (a layer function or formula continues; a
                         field that cannot is refused by name)
          a dict         Fields to attach: a body-wide field on the
                         extended skeleton replaces that name outright,
                         a single-layer field goes to the shell it spans

        Empty is the honest option for, say, a crustal shell whose
        properties the consumer will supply by attribute.  Extrapolating
        mantle values into a crust would be a quieter kind of wrong.
        """
        radii = list(np.atleast_1d(np.asarray(radii, dtype=float)))
        names = list(names or [None] * len(radii))
        if len(names) != len(radii):
            raise ValueError(f"got {len(names)} names for {len(radii)} radii")
        sk = self._sk.extended(radii)
        n_old = self._sk.nlayers
        n_new = sk.nlayers - n_old
        if not (fields is None or fields == "extrapolate"
                or isinstance(fields, _MappingABC)):
            raise ValueError(
                'fields must be "extrapolate", a dict of Fields, or None')

        last = self._layers[-1]
        layers = list(self._layers)
        for k in range(n_new):
            i = n_old + k
            lo, hi = sk.interval(i)
            held = {}
            if fields == "extrapolate":
                for name, f in last.fields.items():
                    try:
                        held[name] = f.on_interval(lo, hi)
                    except TypeError as exc:
                        raise TypeError(
                            f"cannot extrapolate {name!r} into the new shell "
                            f"[{lo:.6g}, {hi:.6g}]: {exc}") from None
            layers.append(Layer(index=i, interval=(lo, hi), state=state,
                                fields=held))
        faces = list(self._interfaces)
        b = sk.boundaries
        for k in range(n_new):
            i = n_old + k
            faces[-1] = _replace(faces[-1], between=(faces[-1].between[0], i))
            faces.append(Interface(
                index=i, name=names[k], radius=float(b[i + 1]),
                between=(i, -1), role=role))
        body = self._rebuilt(layers=layers, interfaces=tuple(faces))
        if isinstance(fields, _MappingABC):
            for key, f in fields.items():
                body.add_field(key, f, replace=True)
        return body

    def with_buffer(self, *, ratio=None, radius=None, name="buffer"
                    ) -> "ReferenceBody":
        """A copy with a vacuum shell appended outside the surface.

        Give `ratio` = b/a - 1 (so 0.2 means b/a = 1.2) or `radius` = b
        directly.  The shell holds no fields and is vacuum: MMA26's
        reference domain B, a ball strictly enclosing the planet with
        the mapping the identity on its boundary.
        """
        if (ratio is None) == (radius is None):
            raise ValueError("give exactly one of ratio and radius")
        a = float(self._sk.boundaries[-1])
        b = a * (1.0 + float(ratio)) if ratio is not None else float(radius)
        if b <= a:
            raise ValueError(
                f"buffer outer radius {b} must exceed the surface radius {a}")
        return self.extended([b], fields=None, state="vacuum",
                             names=[name]).annotate(-1, name=name)

    # -- units --------------------------------------------------------------

    def rescaled(self, new_scales: Scales) -> "ReferenceBody":
        """The same body, its values re-expressed in different scales.

        Converts three kinds of thing, because all three carry units:
        the field values, the skeleton radii, and the attached surfaces.
        Each field converts itself (`Field.rescaled`), a composite
        through its operands and a law through its constants;
        piecewise-polynomial layer functions convert exactly -- one
        multiply per coefficient -- so an exact model stays exact
        through non-dimensionalisation and back.

        A field that cannot re-express itself is refused by name, as is
        a leaf with no declared dimensions: silently leaving a modulus
        unscaled produces a wrong answer that looks entirely plausible,
        and Dimensions.DIMENSIONLESS is the explicit way to say a field
        genuinely has none.
        """
        old, new = self.scales, new_scales
        if old == new:
            return self
        k = old.length / new.length

        seen: dict[int, Field] = {}

        def convert(field):
            """The field in the new scales, each object converted once."""
            if id(field) in seen:           # shared substructure stays shared
                return seen[id(field)]
            if hasattr(field, "rescaled"):
                out = field.rescaled(convert, old, new)
            else:
                raise TypeError(
                    f"cannot rescale {type(field).__name__} "
                    f"{getattr(field, 'name', None)!r}: it does not say how to "
                    "re-express itself in other scales (no `rescaled`). "
                    "Rebuild it after the rescale.")
            seen[id(field)] = out
            return out

        layers = []
        for lay in self._layers:
            lo, hi = lay.interval
            fields = {n: convert(f) for n, f in lay.fields.items()}
            layers.append(_replace(lay, interval=(lo * k, hi * k), fields=fields))
        faces = tuple(_replace(f, radius=f.radius * k)
                      for f in self._interfaces)
        surfaces = {radius * k: Surface(s.reference_radius * k,
                                        topography=s.topography * k, name=s.name)
                    for radius, s in self._surfaces.items()}
        return type(self)(tuple(layers), meta=dict(self.meta),
                          interfaces=faces, surfaces=surfaces, scales=new)

    def nondimensionalised(self, *, density: float = EARTH_MEAN_DENSITY,
                           length: float | None = None) -> "ReferenceBody":
        """The recommended working form: geophysical scales, G = 1.

        `density` is prescribed, never computed -- conventionally the
        Earth's mean -- and `length` defaults to the outermost
        non-vacuum boundary.  Valid only from SI: a body already in some
        other scales should say what it wants with rescaled() directly,
        rather than have defaults derived from already-scaled radii.
        """
        if not self.scales.is_si:
            raise ValueError(
                "nondimensionalised() converts from SI, and this body's "
                f"scales are {self.scales!r}; use rescaled() to move "
                "between non-SI scales explicitly")
        if length is None:
            solid = [lay for lay in self._layers if not lay.is_vacuum]
            if not solid:
                raise ValueError("the body is all vacuum; give length=")
            length = float(self._sk.boundaries[solid[-1].index + 1])
        return self.rescaled(Scales.geophysical(length, density=density))

    def redimensionalised(self) -> "ReferenceBody":
        """Back to SI: the output half of the pipeline, named."""
        return self.rescaled(Scales.SI)

    # -- boundary shapes ----------------------------------------------------
    #
    # Attached shapes are stored keyed by the interface RADIUS, not the
    # interface index.  Indices are positional and surgery renumbers
    # them -- refined() shifts everything above the insertion up by one
    # -- so an index key silently migrates a surface onto the wrong
    # interface.  A radius is the identity an interface keeps through
    # surgery: it survives renumbering exactly (the same float travels
    # through every skeleton operation), and a surface whose radius is
    # no longer an interface (dropped by coarsening, cut off by
    # truncation) simply stops being listed, which is the right answer.

    @property
    def surfaces(self) -> dict[int, Surface]:
        """Attached boundary shapes, keyed by current interface index."""
        return {f.index: self._surfaces[f.radius]
                for f in self._interfaces if f.radius in self._surfaces}

    def with_surface(self, which, shape, *, atol: float | None = None
                     ) -> "ReferenceBody":
        """A copy with a boundary shape attached to one interface.

        **The zero-mean contract.**  An `Interface.radius` *is* the
        boundary's area-weighted mean radius -- that is what the manifest
        and the netCDF format record under that name -- so relief
        attached to it must be a departure from that radius, of zero
        mean.  Attachment is where that is enforced, because it is the
        only place where the two halves of the statement, the radius and
        the shape, are in the same room.

        A bare Topography is centred here: its mean is computed and,
        unless it is already zero to within `atol`, removed and reported
        by a warning; the shape then sits at the interface's own radius.
        A Surface is accepted only if it already satisfies the contract
        -- placed at the interface's radius, carrying centred relief --
        and refused otherwise, naming `Surface.centred()`, which moves a
        mean into the radius, as the fix.  Refusing is the point: raw
        CRUST-1.0 depths attached at the mean Moho radius put the
        physical Moho twice as deep as the data says, and nothing
        downstream could notice.

        `atol` is the tolerance for both tests, in the body's units, and
        defaults to `1e-9 * face.radius`, which is float64 round-off on a
        radius rather than a modelling tolerance.
        """
        i = self._resolve_interface(which)
        face = self._interfaces[i]
        tol = 1e-9 * abs(face.radius) if atol is None else abs(float(atol))
        if isinstance(shape, Surface):
            if abs(shape.reference_radius - face.radius) > tol:
                raise ValueError(
                    f"the surface offered for interface {face.name!r} sits at "
                    f"reference radius {shape.reference_radius:.6g}, but that "
                    f"interface is at {face.radius:.6g}: an interface radius "
                    "is the boundary's mean radius, so the two must agree. "
                    "Place the surface at the interface's radius, move the "
                    "interface to the surface's, or attach the topography "
                    "itself and let with_surface centre it.")
            mean = float(shape.topography.mean())
            if abs(mean) > tol:
                raise ValueError(
                    f"the surface offered for interface {face.name!r} carries "
                    f"relief of mean {mean:.6g}, not zero, so the boundary it "
                    f"describes does not lie at {face.radius:.6g} on average. "
                    "Use Surface.centred() to move that mean into the "
                    "reference radius, or attach the topography itself and "
                    "let with_surface centre it.")
            surface = shape if shape.name else _replace(shape, name=face.name)
        elif callable(shape):
            topography = as_topography(shape)
            mean = float(topography.mean())
            if abs(mean) > tol:
                # Within tolerance the shape is passed through untouched,
                # to the same tolerance a Surface is accepted on: a shape
                # already centred keeps its own type, its bounds and its
                # file names, and a walk of the tree still reads its
                # outermost scaling as the exaggeration it is
                # (Topography.provenance()).
                topography = CentredTopography(topography, mean)
                warnings.warn(
                    f"relief attached to interface {face.name!r} had an "
                    f"area-weighted mean of {mean:.6g} removed, so that the "
                    f"interface radius {face.radius:.6g} is the boundary's "
                    "mean radius as it is defined to be. If that mean was "
                    "meant to place the boundary, the interface radius and "
                    "the data disagree: move the interface to the data's own "
                    "mean radius.", UserWarning, stacklevel=2)
            surface = Surface(face.radius, topography=topography, name=face.name)
        else:
            raise TypeError(
                f"expected a Surface or a Topography, got {type(shape).__name__}")

        surfaces = dict(self._surfaces)
        surfaces[face.radius] = surface
        return self._rebuilt(layers=self._layers, interfaces=self._interfaces,
                             surfaces=surfaces)

    def without_surface(self, which) -> "ReferenceBody":
        """A copy with one interface's shape removed."""
        radius = self._interfaces[self._resolve_interface(which)].radius
        surfaces = {k: v for k, v in self._surfaces.items() if k != radius}
        return self._rebuilt(layers=self._layers, interfaces=self._interfaces,
                             surfaces=surfaces)

    def mapping(self, *, rule=None, displacement=None) -> Mapping:
        """The body's mapping m : reference -> physical.

        This is where the two consumer archetypes converge on one
        object.  Mode A hands the result to the mesher, which applies
        it to nodes; Mode B hands it to a solver, which contracts its
        F and J into a weak form.  Same mapping either way.

        `rule` is a displacement rule -- any callable of the body
        returning a RadialDisplacement, layer_linear() being the
        shipped one -- and reads the attached surfaces.  `displacement`
        is a RadialDisplacement (or bare callable) supplied directly,
        for a prescribed volumetric h.  With neither, the body is
        spherically symmetric and the mapping is the identity, which is
        why nothing downstream special-cases the 1D world.
        """
        if rule is not None and displacement is not None:
            raise ValueError(
                "give a rule or a displacement, not both: a rule builds the "
                "displacement from the attached surfaces, so passing another "
                "one alongside would be two answers to one question")
        rmax = float(self._sk.boundaries[-1])
        if rule is not None:
            return RadialStretch(rule(self), rmax=rmax)
        if displacement is not None:
            return RadialStretch(as_displacement(displacement), rmax=rmax)
        return IdentityMapping()

    def sample(self, grid, *, fields=None, mapping=None, radial=None,
               ngll: int = 5, drmax=None, omega=None):
        """The body on a consumer's nodes: a Sample.

        `grid` is an AngularGrid; the radial nodes are `radial` when
        given, a RadialMesh of this body, and otherwise built from the
        grid's band (`lmax`) or from `drmax` with `ngll` nodes per
        element.  `fields` names what to sample (default: every static
        field), and `mapping`, when given, adds its displacement in the
        spherical frame.  `omega` samples the frequency-dependent
        fields there too, as (real, imaginary) pairs.  The work is done
        by `planetmodel.sampling`, which imports the 1D mesh and so is
        reached from here lazily.
        """
        from ..sampling import sample_body
        return sample_body(self, grid, fields=fields, mapping=mapping,
                           radial=radial, ngll=ngll, drmax=drmax, omega=omega)

    def surface(self, which) -> Surface | None:
        """The shape attached to an interface, or None."""
        radius = self._interfaces[self._resolve_interface(which)].radius
        return self._surfaces.get(radius)

    def __repr__(self) -> str:
        """Compact summary: optional name, layer count and field names."""
        nm = self.meta.get("name")
        head = f"{type(self).__name__}({nm!r}, " if nm else f"{type(self).__name__}("
        return head + f"{self._sk.nlayers} layers, fields={list(self.field_names)})"

