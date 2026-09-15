"""spec.py -- what to mesh, and what came back.

A MeshSpec is the whole description of a wanted mesh: which body, which
of its interfaces survive, what to add around it, how finely to resolve
each boundary, and which delivery -- the physical mesh, or the
reference mesh plus the mapping.  build_layered_mesh turns one into a
MeshResult.

Sizing is a callable, not a class hierarchy: a rule takes the
interfaces and the reference radius and returns one InterfaceSizing per
interface.  The three shipped rules are frozen dataclasses whose
instances *are* the callable, so they print, compare and serialise, and
anyone wanting something else writes a function.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..registry import register

__all__ = [
    "BufferSpec", "InterfaceSizing", "SizingRule", "AngularResolution",
    "UniformInterfaces", "PerInterface", "MeshSpec", "MeshResult",
]


@dataclass(frozen=True)
class BufferSpec:
    """A vacuum shell outside the body, for exterior coupling.

    `ratio` is b/a - 1, so 0.2 means an outer radius 1.2 times the
    surface; `radius` gives b directly.  Exactly one of them.
    """

    _: KW_ONLY
    ratio: float | None = None
    radius: float | None = None
    name: str = "buffer"

    def __post_init__(self) -> None:
        if (self.ratio is None) == (self.radius is None):
            raise ValueError("give exactly one of ratio and radius")


@dataclass(frozen=True)
class InterfaceSizing:
    """Target element size at an interface, and how it relaxes away.

    All three are in the same units as the geometry they describe --
    the mesher non-dimensionalises them alongside the radii, once.
    """

    size: float
    far_size: float
    decay_width: float

    def __post_init__(self) -> None:
        for name, value in (("size", self.size), ("far_size", self.far_size),
                            ("decay_width", self.decay_width)):
            if not value > 0.0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.far_size < self.size:
            raise ValueError(
                f"far_size ({self.far_size}) is smaller than size "
                f"({self.size}): the mesh would refine away from the "
                "interface rather than towards it")

    def scaled(self, factor: float) -> "InterfaceSizing":
        """The same sizing in units scaled by `factor`."""
        return InterfaceSizing(self.size * factor, self.far_size * factor,
                               self.decay_width * factor)


#: A sizing rule maps the interfaces of a body to a sizing for each.
SizingRule = Callable[[Sequence, float], dict]


@register("sizing", "angular_resolution")
@dataclass(frozen=True)
class AngularResolution:
    """Equal angular resolution on every interface: h_i = h_ref r_i/r_ref.

    The default.  A deep interface is smaller than a shallow one, so
    giving both the same absolute element size resolves the deep one far
    more finely in angle than the shallow one -- usually not what is
    wanted, and expensive where it is not.  Scaling with radius gives
    every boundary comparable triangle counts.

    The decay width scales with radius too, so the transition away from
    each interface occupies a comparable fraction of the body.
    """

    h_ref: float
    r_ref: float
    h_far: float
    _: KW_ONLY
    fraction: float = 0.2

    def __call__(self, interfaces, rref: float) -> dict:
        out = {}
        for face in interfaces:
            h = self.h_ref * face.radius / self.r_ref
            out[face.index] = InterfaceSizing(
                size=min(h, self.h_far),
                far_size=self.h_far,
                decay_width=self.fraction * face.radius)
        return out


@register("sizing", "uniform_interfaces")
@dataclass(frozen=True)
class UniformInterfaces:
    """The same absolute size at every interface.

    Simplest, and right when the structure of interest is all at one
    depth.
    """

    h_min: float
    h_max: float
    decay_width: float

    def __call__(self, interfaces, rref: float) -> dict:
        return {face.index: InterfaceSizing(self.h_min, self.h_max,
                                            self.decay_width)
                for face in interfaces}


@register("sizing", "per_interface")
@dataclass(frozen=True)
class PerInterface:
    """Explicit sizing for named or indexed interfaces, with a fallback.

    For when the default reading of "resolve this boundary well" is
    wrong and the user knows better.  Interfaces not named fall through
    to `base`; without a base they must all be named.
    """

    sizes_by: dict
    _: KW_ONLY
    base: SizingRule | None = None

    def __call__(self, interfaces, rref: float) -> dict:
        out = dict(self.base(interfaces, rref)) if self.base is not None else {}
        by_name = {f.name: f.index for f in interfaces if f.name}
        for key, sizing in self.sizes_by.items():
            if isinstance(key, str):
                if key not in by_name:
                    raise KeyError(
                        f"no interface named {key!r}; named interfaces are "
                        f"{sorted(by_name)}")
                out[by_name[key]] = sizing
            else:
                out[int(key)] = sizing
        missing = [f.index for f in interfaces if f.index not in out]
        if missing:
            raise ValueError(
                f"no sizing for interfaces {missing}; name them or give a base "
                "rule")
        return out


@dataclass(frozen=True)
class MeshSpec:
    """The complete description of a mesh to build.

    The geometry steps happen in the order listed here and in
    build_layered_mesh: coarsen, set the outer boundary, refine, extend,
    buffer, and then the surfaces are attached.  That order is not
    arbitrary -- cutting after refining would drop inserted interfaces,
    buffering before extending would bury the buffer inside the body,
    and a surface attaches to an interface that has to exist first.

    `delivery` is `"physical"` (the nodes carry the mapping) or
    `"referential"` (the mesh stays spherical and the mapping travels
    with it for the consumer to apply).
    """

    body: object                                # ReferenceBody
    sizing: SizingRule
    _: KW_ONLY
    rref: float | None = None                   # required only for an SI body
    dimension: int = 3                          # 3 spheres, 2 discs
    order: int = 2                              # element order, 1..3

    keep_interfaces: Sequence[int] | None = None
    drop_interfaces: Sequence[int] | None = None
    outer_radius: float | None = None           # outer boundary: cut, or grown, to here
    outer_name: str | None = None               # what that boundary is called
    insert_radii: Sequence[float] = ()
    insert_names: Sequence[str | None] = ()
    insert_role: str = "material"
    extend_radii: Sequence[float] = ()
    extend_names: Sequence[str | None] = ()
    extend_fields: object = "extrapolate"
    extend_role: str = "material"
    buffers: Sequence[BufferSpec] = ()

    #: Boundary shapes to attach *after* the surgery, by interface name
    #: or index.  Attaching beforehand is impossible for the interesting
    #: case -- a Moho relief belongs on the boundary that truncation
    #: creates, which does not exist until the surgery has run.
    surfaces: dict = field(default_factory=dict)

    mapping: object | None = None               # Mapping, or None
    #: A displacement rule (layer_linear() and friends), from which the
    #: mapping is built once the resolved body exists.  A rule reads the
    #: attached surfaces, so a caller who lets the spec attach them
    #: cannot build the mapping itself; this is the either/or that
    #: ReferenceBody.mapping already draws, lifted to the spec.
    mapping_rule: object | None = None
    delivery: str = "physical"                  # physical | referential

    algorithm_2d: int = 6
    algorithm_3d: int = 1
    validate: bool = True
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.dimension not in (2, 3):
            raise ValueError(f"dimension must be 2 or 3, got {self.dimension}")
        if not 1 <= self.order <= 3:
            raise ValueError(f"element order must be 1..3, got {self.order}")
        if self.delivery not in ("physical", "referential"):
            raise ValueError(
                "delivery must be 'physical' or 'referential', got "
                f"{self.delivery!r}")
        if self.keep_interfaces is not None and self.drop_interfaces is not None:
            raise ValueError("give keep_interfaces or drop_interfaces, not both")
        if self.mapping is not None and self.mapping_rule is not None:
            raise ValueError(
                "give mapping or mapping_rule, not both: a rule builds the "
                "mapping from the resolved body's surfaces, so passing "
                "another one alongside would be two answers to one question")
        if self.outer_name is not None and self.outer_radius is None:
            raise ValueError(
                "outer_name names the boundary outer_radius creates, but "
                "outer_radius is not set")
        if self.rref is not None and not self.rref > 0.0:
            raise ValueError(f"rref must be positive, got {self.rref}")
        if self.surfaces and self.mapping_rule is None:
            raise ValueError(
                "surfaces are attached but no mapping_rule builds a mapping "
                "from them, so the relief would never reach the mesh"
                + ("; an explicit mapping does not read them"
                   if self.mapping is not None else ""))
        if self.delivery == "referential" and self.mapping is None \
                and self.mapping_rule is None:
            raise ValueError(
                "a referential delivery hands the consumer the mapping to "
                "apply, and none is given; a spherical body wants "
                "delivery='physical'")
        for radii, names, what in ((self.insert_radii, self.insert_names,
                                    "insert"),
                                   (self.extend_radii, self.extend_names,
                                    "extend")):
            if names and len(names) != len(radii):
                raise ValueError(
                    f"{what}_names has {len(names)} entries for "
                    f"{len(radii)} radii")


@dataclass(frozen=True)
class MeshResult:
    """What a build produced.

    `body` is the *resolved* body -- after coarsening, the outer
    boundary, insertion and buffering -- which is what the manifest describes and
    what a caller needs to interpret the attribute numbers.

    The last three are what the MFEM exporter (`mesh3d/export.py`) needs
    and cannot recover from the files: the spec it was built from, the
    `MeshUnits` relating mesh lengths to the body's, and the mapping in
    the *body's* coordinates, which is the one whose F and J carry the
    fields across.  They are keyword-only with defaults so a MeshResult
    written by hand, in a test or a script, still constructs.
    """

    msh_path: Path
    manifest_path: Path
    body: object
    counts: dict
    validation: object
    timings: dict
    _: KW_ONLY
    spec: object = None
    units: object = None
    mapping: object = None

    def __repr__(self) -> str:
        n = self.counts.get("elements", "?")
        return (f"MeshResult({self.msh_path.name}, {n} elements, "
                f"{self.counts.get('layers', '?')} layers)")
