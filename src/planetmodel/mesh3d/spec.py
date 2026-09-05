"""What to mesh, and what came back.

A MeshSpec is the whole description of a wanted mesh: the geometry, the
sizing of every interface, the shells appended outside the geometry,
the dimension and element order, and the delivery.  The computational
domain is the geometry's layers followed by the shells, numbered from
the centre; `MeshSpec.domain` is that domain as a Geometry under the
geometry's own mapping, and is what the builder meshes.

Sizing is a callable, not a class hierarchy: a rule takes the
interfaces of the computational domain and its outer radius, in the
geometry's own lengths, and returns one InterfaceSizing per interface
index.  The three shipped rules are frozen dataclasses whose instances
are the callable, so they print, compare and serialise; anything else
is a function.

Nothing here knows about units.  Every length is a number in the
geometry's own lengths, and the builder hands those numbers to gmsh
unchanged: a geometry of radius 6.4e6 is meshed at radius 6.4e6.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import Callable, Sequence

from ..geometry import Geometry, InterfaceInfo, LayerInfo
from ..mapping import Mapping

__all__ = [
    "Shell", "InterfaceSizing", "SizingRule", "AngularResolution",
    "UniformInterfaces", "PerInterface", "MeshSpec", "MeshResult",
    "ValidationReport", "DELIVERIES", "QUALITY_FLOOR",
]

#: The two deliveries: nodes moved by the mapping, or the reference mesh
#: with the mapping recorded for the consumer to apply.
DELIVERIES = ("physical", "referential")

#: The minSICN below which an element is poorly shaped: the level at
#: which `raise_order` runs gmsh's high-order optimiser and at which
#: `validate_mesh` warns.  Elements at or below zero are folded and fail.
QUALITY_FLOOR = 0.05


@dataclass(frozen=True)
class Shell:
    """A layer appended outside the geometry, numbered after its layers.

    Exactly one of `ratio` and `radius` gives the outer radius b of the
    shell: `ratio` is b / a - 1 with a the radius of the boundary the
    shell is attached to (the geometry's outer boundary, or the previous
    shell's), `radius` is b itself.  `name` names the layer; its outer
    boundary takes the default interface name.
    """

    _: KW_ONLY
    #: b / a - 1, the thickness as a fraction of the radius it sits on.
    ratio: float | None = None
    #: b, the outer radius of the shell, in the geometry's lengths.
    radius: float | None = None
    #: The layer's name in the mesh and the manifest.
    name: str = "buffer"

    def __post_init__(self) -> None:
        if (self.ratio is None) == (self.radius is None):
            raise ValueError("give exactly one of ratio and radius")
        if self.ratio is not None and not self.ratio > 0.0:
            raise ValueError(f"a shell's ratio must be positive, got {self.ratio}")
        if self.radius is not None and not self.radius > 0.0:
            raise ValueError(f"a shell's radius must be positive, got {self.radius}")

    def outer_radius(self, inner: float) -> float:
        """The shell's outer radius when attached to a boundary at `inner`."""
        if self.radius is not None:
            return float(self.radius)
        return float(inner) * (1.0 + float(self.ratio))


@dataclass(frozen=True)
class InterfaceSizing:
    """Target element size at an interface, and how it relaxes away.

    All three are lengths in the geometry's own units, handed to gmsh as
    they are.
    """

    #: The element size on the interface.
    size: float
    #: The element size far from it.
    far_size: float
    #: The distance over which the size grows from `size` to `far_size`.
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
        """The same sizing with every length multiplied by `factor`."""
        return InterfaceSizing(self.size * factor, self.far_size * factor,
                               self.decay_width * factor)


#: A sizing rule: (interfaces of the computational domain, its outer
#: radius) -> {interface index: InterfaceSizing}.  The interfaces are
#: objects with `index`, `radius` and `name`.
SizingRule = Callable[[Sequence[InterfaceInfo], float], dict]


@dataclass(frozen=True)
class AngularResolution:
    """Equal angular resolution on every interface: h_i = h_ref r_i / r_ref.

    A deep interface is smaller than a shallow one, so one absolute size
    everywhere resolves the deep one far more finely in angle than the
    shallow one.  Scaling with radius gives every boundary comparable
    element counts, capped at `h_far`; the decay width is `fraction` of
    each interface's radius.  `r_ref` defaults to the outer radius the
    rule is called with.
    """

    #: The element size at r_ref.
    h_ref: float
    #: The element size far from every interface, and the cap on h_i.
    h_far: float
    _: KW_ONLY
    #: The radius at which the size is h_ref; the outer radius if None.
    r_ref: float | None = None
    #: The decay width as a fraction of each interface's radius.
    fraction: float = 0.2

    def __call__(self, interfaces, outer_radius: float) -> dict:
        r_ref = float(outer_radius) if self.r_ref is None else float(self.r_ref)
        out = {}
        for face in interfaces:
            h = self.h_ref * face.radius / r_ref
            out[face.index] = InterfaceSizing(
                size=min(h, self.h_far), far_size=self.h_far,
                decay_width=self.fraction * face.radius)
        return out


@dataclass(frozen=True)
class UniformInterfaces:
    """The same absolute size at every interface."""

    #: The element size on every interface.
    h_min: float
    #: The element size far from every interface.
    h_max: float
    #: The distance over which the size grows from h_min to h_max.
    decay_width: float

    def __call__(self, interfaces, outer_radius: float) -> dict:
        return {face.index: InterfaceSizing(self.h_min, self.h_max,
                                            self.decay_width)
                for face in interfaces}


@dataclass(frozen=True)
class PerInterface:
    """Explicit sizing for named or indexed interfaces, with a fallback.

    Interfaces not named in `sizes_by` fall through to `base`; without a
    base they must all be named.
    """

    #: {interface name or index: InterfaceSizing}.
    sizes_by: dict
    _: KW_ONLY
    #: The rule for the interfaces `sizes_by` does not name.
    base: SizingRule | None = None

    def __call__(self, interfaces, outer_radius: float) -> dict:
        out = (dict(self.base(interfaces, outer_radius))
               if self.base is not None else {})
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

    Surgery is done on the geometry before it comes here.  The
    computational domain is the geometry followed by the shells; its
    layers are numbered 1..N from the centre and its interfaces 1..M in
    the order of `geometry.interfaces`, then the shells' outer
    boundaries.
    """

    #: The geometry to mesh: a skeleton, a mapping and names.
    geometry: Geometry
    #: The rule giving every interface of the computational domain its sizing.
    sizing: SizingRule
    _: KW_ONLY
    #: 3 for balls, 2 for discs.
    dimension: int = 3
    #: The element order, 1..3.
    order: int = 2
    #: Shells appended outside the geometry, innermost first.
    shells: Sequence[Shell] = ()
    #: "physical": the nodes are moved by the mapping; "referential": the
    #: mesh stays spherical and the mapping is recorded.
    delivery: str = "physical"
    #: gmsh's Mesh.Algorithm.
    algorithm_2d: int = 6
    #: gmsh's Mesh.Algorithm3D.
    algorithm_3d: int = 1
    #: False writes a failing mesh with the failures recorded in the manifest.
    validate: bool = True
    #: Copied into the manifest's provenance block.
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, Geometry):
            raise TypeError(
                f"geometry must be a Geometry, got {type(self.geometry).__name__}")
        if not callable(self.sizing):
            raise TypeError("sizing must be a callable sizing rule")
        if self.dimension not in (2, 3):
            raise ValueError(f"dimension must be 2 or 3, got {self.dimension}")
        if not 1 <= self.order <= 3:
            raise ValueError(f"element order must be 1..3, got {self.order}")
        if self.delivery not in DELIVERIES:
            raise ValueError(
                f"delivery must be one of {DELIVERIES}, got {self.delivery!r}")
        object.__setattr__(self, "shells", tuple(self.shells))
        for shell in self.shells:
            if not isinstance(shell, Shell):
                raise TypeError(
                    f"shells must be Shell instances, got {type(shell).__name__}")
        inner = float(self.geometry.skeleton.boundaries[-1])
        for shell in self.shells:
            outer = shell.outer_radius(inner)
            if not outer > inner:
                raise ValueError(
                    f"shell {shell.name!r} has outer radius {outer:g}, not above "
                    f"the boundary it is attached to at {inner:g}")
            inner = outer
        self.domain     # names must be unique across the geometry and the shells

    @property
    def shell_radii(self) -> tuple[float, ...]:
        """The outer radii of the shells, innermost first."""
        radii = []
        inner = float(self.geometry.skeleton.boundaries[-1])
        for shell in self.shells:
            inner = shell.outer_radius(inner)
            radii.append(inner)
        return tuple(radii)

    @property
    def outer_radius(self) -> float:
        """The outer radius of the computational domain."""
        radii = self.shell_radii
        return radii[-1] if radii else float(self.geometry.skeleton.boundaries[-1])

    @property
    def domain(self) -> Geometry:
        """The computational domain: the geometry then the shells, unchecked.

        A Geometry under the geometry's own mapping, whose layers and
        interfaces carry the numbering the mesh will.  Whether the mapping
        is defined on the shells is the builder's check, not this one's.
        """
        g = self.geometry
        if not self.shells:
            return g
        sk = g.skeleton.extended(self.shell_radii)
        return Geometry(
            sk, mapping=g.mapping,
            layer_names=[lay.name for lay in g.layers] + [s.name for s in self.shells],
            interface_names=[f.name for f in g.interfaces] + [None] * len(self.shells),
            rtol=g.rtol, check=False)

    @property
    def layers(self) -> tuple[LayerInfo, ...]:
        """The layers of the computational domain, centre outward."""
        return self.domain.layers

    @property
    def interfaces(self) -> tuple[InterfaceInfo, ...]:
        """The interfaces of the computational domain, centre outward."""
        return self.domain.interfaces


@dataclass
class ValidationReport:
    """The outcome of every check of a built mesh, whether or not any failed."""

    #: The mesh dimension.
    dimension: int
    _: KW_ONLY
    #: Elements whose minSICN is not positive.
    negative_jacobians: int = 0
    #: The worst minSICN over the cells.
    min_sicn: float = 0.0
    #: Cells with negative signed volume or area.
    negative_cells: int = 0
    #: Boundary faces whose normal points towards the centre they enclose.
    inward_faces: int = 0
    #: The largest distance of a tagged interface's mean radius from the
    #: radius asked for, in the mesh's lengths.
    max_interface_radius_error: float = 0.0
    #: The number of physical groups found, by "layers" and "interfaces".
    group_counts: dict = field(default_factory=dict)
    #: Every failed check, in words.
    failures: list = field(default_factory=list)
    #: Every warning, in words.
    warnings: list = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.failures

    def raise_if_failed(self) -> "ValidationReport":
        """Raise ValueError listing every failure, or return self."""
        if self.failures:
            raise ValueError(
                "the generated mesh failed validation:\n  - "
                + "\n  - ".join(self.failures))
        return self

    def __repr__(self) -> str:
        state = "ok" if self.ok else f"{len(self.failures)} FAILED"
        return (f"ValidationReport({state}, minSICN {self.min_sicn:.4g}, "
                f"max interface error {self.max_interface_radius_error:.3g})")


@dataclass(frozen=True)
class MeshResult:
    """What a build produced.

    `mapping` is the geometry's own mapping, which acts on the mesh's
    coordinates as they are and is what the exporter applies to the
    nodes it reads back; it is None for a mesh that was not built from
    a geometry.  The keyword fields have defaults so a MeshResult
    written by hand still constructs.
    """

    #: The MSH 2.2 file.
    msh_path: Path
    #: The JSON manifest beside it.
    manifest_path: Path
    #: The geometry the mesh was built from; None for an offset mesh.
    geometry: Geometry | None
    #: Element, node, layer and interface counts.
    counts: dict
    #: The checks the mesh passed or failed.
    validation: ValidationReport
    #: Seconds spent in each stage of the build.
    timings: dict
    _: KW_ONLY
    #: The spec the mesh was built from, if any.
    spec: MeshSpec | None = None
    #: The geometry's mapping, or None where there was no geometry.
    mapping: Mapping | None = None

    def __repr__(self) -> str:
        n = self.counts.get("elements", "?")
        return (f"MeshResult({self.msh_path.name}, {n} elements, "
                f"{self.counts.get('layers', '?')} layers)")
