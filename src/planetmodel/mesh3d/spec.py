"""What to mesh, and what came back.

A MeshSpec is the whole description of a wanted mesh: the geometry, the
sizing of every interface, the shells appended outside the geometry,
the dimension and element order, and what the mapping may do to the
outer boundary.  The mesh built from it is always the reference one;
the mapping is applied at the MFEM export.  The computational
domain is the geometry's layers followed by the shells, numbered from
the centre; `MeshSpec.domain` is that domain as a Geometry under the
geometry's own mapping, and is what the builder meshes.

A flat geometry (`planetmodel.cartesian.Geometry`) is meshed as the
box over its lateral domain, cut by the planes of its interfaces, or,
over a `Rectangle`, as its section in the (x, z) plane: three or two
dimensions, which the geometry decides.  Its layers and interfaces are
numbered from the surface down, as the geometry numbers them, and its
base and sides are faces of their own (the manifest's `faces[]`).  A
box approximates an unbounded body, so it takes no shells and no
spherical outer boundary.  Faults in a box are built under any
mapping that inverts itself (topography, a `cartesian.VerticalStretch`),
pulled back through it as on the sphere; in a section they are not
built.

Sizing is a callable, not a class hierarchy: a rule takes the
interfaces of the computational domain and its last boundary (the
outer radius of a ball, the depth of a box's base), in the geometry's
own lengths, and returns one InterfaceSizing per interface index.  The
shipped rules are frozen dataclasses whose instances
are the callable, so they print, compare and serialise; anything else
is a function.  `UniformInterfaces` and `PerInterface` size any
geometry; `AngularResolution` and `CappedInterfaces` scale with radius,
and refuse an interface at radius zero, such as a box's surface.  A
refinement is a second kind of callable, a size
field over the reference domain composed into the sizing by minimum,
so it can only refine, never coarsen; NearPoints and near_curve are
the shipped kinds, and anything with the Refinement protocol's two
attributes serves.  Its points are in the mesh's coordinates: (x, z)
for a section.

Faults are built into the mesh from the start (`EmbeddedFault`): each
is pulled back through the geometry's mapping, fragmented into the
shells (or the boxes of a box's layers) with them so that the mesh
conforms to it, and labelled with a boundary attribute of its own after
the interfaces' and a box's faces' (the manifest's `faults[]`).  A
fault is a surface of a 3D domain, so a spec with faults is
three-dimensional, and its faults are in the geometry's frame: a
geographic one on a sphere, a Cartesian one in a box.

Nothing here knows about units.  Every length is a number in the
geometry's own lengths, and the builder hands those numbers to gmsh
unchanged: a geometry of radius 6.4e6 is meshed at radius 6.4e6.  A
fault's positions are taken in those same numbers.
"""
from __future__ import annotations

import hashlib
from collections.abc import Callable, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.spatial import KDTree

from ..fault import Fault
from ..geometry import Geometry, GeometryBase, InterfaceInfo, LayerInfo
from ..mapping import Mapping
from ._layout import layout_of

if TYPE_CHECKING:
    from ..fault import Preimage

__all__ = [
    "Shell", "InterfaceSizing", "SizingRule", "AngularResolution",
    "UniformInterfaces", "CappedInterfaces", "PerInterface", "Refinement",
    "NearPoints", "near_curve", "EmbeddedFault", "MeshSpec",
    "MeshResult", "ValidationReport", "optimise_methods", "DELIVERIES",
    "OUTER_BOUNDARIES", "QUALITY_FLOOR",
]

#: The two deliveries of the MFEM export: the nodes moved by the mapping,
#: or the reference mesh with the displacement written beside it as a
#: field for the consumer to apply.
DELIVERIES = ("physical", "referential")

#: What the mapping may do to the outer boundary of the computational
#: domain: "free" lets it carry topography like any other boundary;
#: "spherical" requires the mapping to be the identity there, so that
#: a far-field condition can be applied on a sphere.
OUTER_BOUNDARIES = ("free", "spherical")

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
    shell's), `radius` is b itself.  `name` names the layer: by default
    `buffer`, or `buffer_<k>` for the k-th shell, counted from one, when
    a spec has more than one; its outer boundary takes the default
    interface name.
    """

    _: KW_ONLY
    #: b / a - 1, the thickness as a fraction of the radius it sits on.
    ratio: float | None = None
    #: b, the outer radius of the shell, in the geometry's lengths.
    radius: float | None = None
    #: The layer's name in the mesh and the manifest; None for the default.
    name: str | None = None

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


def _require_ramp(size: float, far_size: float, decay_width: float) -> None:
    """Refuse ramp lengths that are not a refinement towards the target."""
    for name, value in (("size", size), ("far_size", far_size),
                        ("decay_width", decay_width)):
        if not value > 0.0:
            raise ValueError(f"{name} must be positive, got {value}")
    if far_size < size:
        raise ValueError(
            f"far_size ({far_size}) is smaller than size ({size}): the mesh "
            "would refine away from the target rather than towards it")


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
        _require_ramp(self.size, self.far_size, self.decay_width)

    def scaled(self, factor: float) -> "InterfaceSizing":
        """The same sizing with every length multiplied by `factor`."""
        return InterfaceSizing(self.size * factor, self.far_size * factor,
                               self.decay_width * factor)


#: A sizing rule: (interfaces of the computational domain, its outer
#: radius) -> {interface index: InterfaceSizing}.  The interfaces are
#: objects with `index`, `radius`, `between` and `name`.
type SizingRule = Callable[[Sequence[InterfaceInfo], float],
                           dict[int, InterfaceSizing]]


class Refinement(Protocol):
    """A size field over the reference domain, composed in by minimum.

    A refinement maps points of shape (n, d), in reference coordinates
    and the geometry's own lengths, to the element sizes wanted there,
    shape (n,).  `size` is the floor of what it returns and `far_size`
    the ceiling: they let the builder bound gmsh's global sizes and the
    scale check judge the refinement without evaluating it.  The builder
    takes the minimum over the per-interface sizing and every
    refinement at each point, so a refinement can only refine, never
    coarsen, and the sizing's own guarantees stand everywhere.
    `planetmodel.testing.check_refinement` holds an implementation to
    this contract.
    """

    #: The smallest size the refinement returns.
    size: float
    #: The largest size the refinement returns.
    far_size: float

    def __call__(self, points: np.ndarray) -> np.ndarray:
        """The element sizes at (n, d) reference points, shape (n,)."""
        ...


def _ramp(distance: ArrayLike, size: float, far_size: float,
          decay_width: float) -> np.ndarray:
    """The element size at `distance` from a refined target: `size` out
    to one `size` of distance, then growing linearly to `far_size` over
    `decay_width`.

    The plateau is what lets the asked size be realised.  The mesher
    settles where an element's own length agrees with the size field
    across the element, so against a ramp of gradient g starting at the
    target the realised size there is about size / (1 - g/2) — twice
    the ask already at g = 1.  Held flat for one element's width, the
    field gives the mesher room to place that element before the ramp
    bites.  An interface needs no plateau: it is meshed as a surface,
    on which its distance is exactly zero.
    """
    d = np.maximum(np.asarray(distance, dtype=float) - size, 0.0)
    return np.minimum(far_size, size + (far_size - size) * (d / decay_width))


@dataclass(frozen=True, eq=False, repr=False)
class NearPoints:
    """Refine towards a cloud of points.

    The points are positions in the reference domain, shape (n, 2) or
    (n, 3) in the geometry's own lengths; a cloud on an interface is the
    caller's unit directions times its radius.  The size is `size` out
    to one `size` of Euclidean distance from the cloud (a KD-tree
    query) — the plateau that lets the mesher realise the asked size,
    see `_ramp` — and grows linearly to `far_size` over the next
    `decay_width`.  Distance grows with depth under a surface cloud as
    fast as it grows laterally, so the refined region around a densely
    sampled curve is a tube of radius about `size + decay_width`, and
    roughly `length * decay_width**2 / size**3` extra tetrahedra (in
    2D, `length * decay_width / size**2` triangles) come with it, which
    is where the element budget goes when the cloud's `size` is far
    below the `far_size` around it.

    The distance to a cloud sampled from a curve is zero at the points
    and about half their spacing between them; spacing at most `size`
    keeps that ripple inside the plateau, where it changes nothing.
    `near_curve` guarantees the spacing by construction, while a cloud
    of genuinely isolated points needs no such rule.  The array is
    stored read-only and a copy; equality, hashing and the manifest go
    through `digest` and the scalars rather than the points themselves.
    """

    #: The cloud, (n, d) reference coordinates, stored read-only.
    points: np.ndarray
    #: The element size on the cloud.
    size: float
    #: The element size far from it.
    far_size: float
    #: The distance over which the size grows from `size` to `far_size`.
    decay_width: float

    def __post_init__(self) -> None:
        pts = np.array(self.points, dtype=float)
        if pts.ndim != 2 or pts.shape[1] not in (2, 3) or pts.shape[0] == 0:
            raise ValueError(
                "points must have shape (n, 2) or (n, 3) with n >= 1, got "
                f"shape {np.shape(self.points)}")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must be finite")
        _require_ramp(self.size, self.far_size, self.decay_width)
        pts.setflags(write=False)
        object.__setattr__(self, "points", pts)
        object.__setattr__(self, "_tree", KDTree(pts))

    def __call__(self, points: ArrayLike) -> np.ndarray:
        """The element sizes at `points`, shaped (n, d) like the cloud."""
        pts = np.asarray(points, dtype=float)
        d = self.points.shape[1]
        if pts.ndim != 2 or pts.shape[1] != d:
            raise ValueError(
                f"points must have shape (n, {d}) like the cloud, got "
                f"{pts.shape}")
        distance, _ = self._tree.query(pts)
        return _ramp(distance, self.size, self.far_size, self.decay_width)

    @property
    def digest(self) -> str:
        """A hex digest of the cloud's shape and bytes: its identity."""
        h = hashlib.sha256(str(self.points.shape).encode())
        h.update(np.ascontiguousarray(self.points).tobytes())
        return h.hexdigest()[:16]

    def _key(self) -> tuple:
        return (self.points.shape, self.digest, self.size, self.far_size,
                self.decay_width)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, NearPoints):
            return NotImplemented
        return self._key() == other._key()

    def __hash__(self) -> int:
        return hash(self._key())

    def __repr__(self) -> str:
        n, d = self.points.shape
        return (f"NearPoints({n} points in {d}D [{self.digest}], "
                f"size={self.size!r}, far_size={self.far_size!r}, "
                f"decay_width={self.decay_width!r})")


def near_curve(vertices: ArrayLike, size: float, far_size: float,
               decay_width: float) -> NearPoints:
    """Refine towards a polyline: NearPoints on a resampling of it.

    `vertices` is (m, 2) or (m, 3), the polyline's vertices in order; a
    closed curve repeats its first vertex at the end.  Every segment is
    subdivided to chord spacing at most `size`, which keeps the ripple
    in the distance to the sampled cloud inside the ramp's plateau,
    where it changes no size, whatever the vertex spacing given.  The
    other three arguments are NearPoints's.
    """
    v = np.asarray(vertices, dtype=float)
    if v.ndim != 2 or v.shape[1] not in (2, 3) or v.shape[0] < 2:
        raise ValueError(
            "vertices must have shape (m, 2) or (m, 3) with m >= 2, got "
            f"shape {v.shape}")
    if not size > 0.0:
        raise ValueError(f"size must be positive, got {size}")
    points = [v[:1]]
    for a, b in zip(v[:-1], v[1:]):
        k = max(int(np.ceil(np.linalg.norm(b - a) / float(size))), 1)
        t = (np.arange(1, k + 1) / k)[:, None]
        points.append(a + t * (b - a))
    return NearPoints(np.concatenate(points), size, far_size, decay_width)


def _require_radii(rule: object, interfaces: Sequence[InterfaceInfo]) -> None:
    """Refuse an interface at radius zero or below for a rule that scales
    with radius: a box's surface, at depth zero, has none."""
    for face in interfaces:
        if not face.radius > 0.0:
            raise ValueError(
                f"interface {face.index} lies at {face.radius:g}, where "
                f"{type(rule).__name__}, which scales with radius, gives no size: "
                "it sizes a spherical geometry; size a flat one with "
                "UniformInterfaces or PerInterface")


def optimise_methods(optimise: str | Sequence[str] | None) -> tuple[str, ...]:
    """The gmsh optimisation methods `optimise` names, as a tuple.

    A single method name, a sequence of names applied in order, or None
    for no optimisation; each is a method `gmsh.model.mesh.optimize`
    accepts ("Netgen", "Relocate3D", ...).
    """
    if optimise is None:
        return ()
    if isinstance(optimise, str):
        return (optimise,)
    try:
        methods = tuple(optimise)
    except TypeError:
        raise TypeError(
            "optimise must be a method name, a sequence of names or None, "
            f"got {type(optimise).__name__}") from None
    for m in methods:
        if not isinstance(m, str):
            raise TypeError(f"optimise methods are names, got {m!r}")
    return methods


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

    def __call__(self, interfaces: Sequence[InterfaceInfo],
                 outer_radius: float) -> dict[int, InterfaceSizing]:
        """One sizing per interface: size h_ref r / r_ref capped at h_far;
        an interface at radius zero is refused."""
        _require_radii(self, interfaces)
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

    def __call__(self, interfaces: Sequence[InterfaceInfo],
                 outer_radius: float) -> dict[int, InterfaceSizing]:
        """The same sizing for every interface."""
        return {face.index: InterfaceSizing(self.h_min, self.h_max,
                                            self.decay_width)
                for face in interfaces}


@dataclass(frozen=True)
class CappedInterfaces:
    """One absolute size, capped in angle and against thin layers.

    Each interface takes `size = min(h, angular * r, thin * s)`, with
    `r` its radius and `s` the thinner of the layer spans it bounds:
    a small sphere is resolved in angle however deep it sits (0.3 rad
    is about twenty elements round a great circle), and a thin layer
    takes elements that fit inside it rather than failing in gmsh a
    long way from the cause.  The far size is `max(h_far, size)`, and
    the decay width the smaller of `decay_width` and `decay_factor`
    times the size, so a size capped far below `h` relaxes over a
    distance of its own scale instead of holding a fine mesh through
    the body.
    """

    #: The element size at every interface, before the caps.
    h: float
    #: The element size far from every interface.
    h_far: float
    #: The distance over which an uncapped size grows to `h_far`.
    decay_width: float
    _: KW_ONLY
    #: The cap in angle: at most `angular * radius` on each interface,
    #: in radians.
    angular: float = 0.3
    #: The cap against thin layers: at most `thin` times the thinner
    #: adjacent layer span.
    thin: float = 4.0
    #: A capped size decays over at most `decay_factor` times itself.
    decay_factor: float = 10.0

    def __call__(self, interfaces: Sequence[InterfaceInfo],
                 outer_radius: float) -> dict[int, InterfaceSizing]:
        """One sizing per interface, capped as the class describes; an
        interface at radius zero is refused."""
        _require_radii(self, interfaces)
        faces = sorted(interfaces, key=lambda f: f.radius)
        radii = [f.radius for f in faces]
        out = {}
        for k, face in enumerate(faces):
            below, above = face.between
            spans = []
            if below != -1:
                spans.append(radii[k] - (radii[k - 1] if k else 0.0))
            if above != -1 and k + 1 < len(radii):
                spans.append(radii[k + 1] - radii[k])
            size = min(self.h, self.angular * face.radius,
                       self.thin * min(spans))
            out[face.index] = InterfaceSizing(
                size=size, far_size=max(self.h_far, size),
                decay_width=min(self.decay_width, self.decay_factor * size))
        return out


@dataclass(frozen=True)
class PerInterface:
    """Explicit sizing for named or indexed interfaces, with a fallback.

    Interfaces not named in `sizes_by` fall through to `base`; without a
    base they must all be named.
    """

    #: {interface name or index: InterfaceSizing}.
    sizes_by: dict[str | int, InterfaceSizing]
    _: KW_ONLY
    #: The rule for the interfaces `sizes_by` does not name.
    base: SizingRule | None = None

    def __call__(self, interfaces: Sequence[InterfaceInfo],
                 outer_radius: float) -> dict[int, InterfaceSizing]:
        """The named sizings over the base rule's, every interface covered."""
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
class EmbeddedFault:
    """A fault to build into the mesh: its faces conform to it, carry a
    boundary attribute of their own and are oriented with its normal.

    The fault is where it physically is, in the geometry's lengths, and
    the mesh is the reference one, so each patch is pulled back through
    the geometry's mapping first (`fault.pull_back`) through `controls`
    nodes each way, or, where `controls` is None, through the fewest of
    `fault.PULL_BACK_CONTROLS` that bring it within `rtol` of its size.
    A patch crossing a knot of the mapping (an interface where a stack
    of reliefs changes slope), whose preimage is creased along the
    crossing, is cut there into pieces (`fault.Region`), each within one
    layer and pulled back as a patch is, all through one grid, their
    join made one curve inside the fault: one fault in the mesh, its
    records and its curves as any (a crossing that does not run along
    the patch's strike, from end to end, is refused).
    An edge of a patch whose preimage lies nowhere more than `reach`
    below the geometry's free surface (its outer sphere, or a box's top)
    is carried across it and trimmed there, so that the fault meets the
    surface exactly, the part carried the preimage of the patch's own
    continuation past the edge; None takes for `reach` the relief of the
    surface over the edge (the largest displacement the mapping gives the
    surface's points above it, along the layered coordinate) plus the
    CAD kernel's tolerance, so that under the identity only an edge at
    the surface or above it is carried across.  In a box, which stands
    for an unbounded body, a fault may leave through the base or a side
    too: an edge whose preimage lies at one of them, or beyond, or
    within the distance the mapping moves that face along its own
    coordinate (none at a side under a vertical stretch), is carried
    across it, the part outside trimmed, and the edge left there is the
    fault's face edge, not its tip.
    A fault crosses every interface it reaches or stops clear of it (a
    tenth of the element size there), and every face of a box: one that
    comes closer without crossing, passes through a sphere's centre,
    leaves a hollow sphere through its inner boundary, lies wholly
    outside the geometry, cuts a layer or an interface right through, or
    overlaps another fault on a surface is refused by name.
    `sizing` refines the mesh towards the fault, `size` on it growing to
    `far_size` over `decay_width`; None leaves the sizing to the
    interfaces' rule, the fault's faces conforming to it at whatever
    size that gives.  `name` names the fault in the mesh and the
    manifest: by default `fault_<k>`, k counted from one in the spec's
    order.  Every patch must give its map as a spline (`spline`), and
    the frame must be the geometry's, which the spec checks: a
    `fault.GeographicFrame` on a sphere, a `fault.CartesianFrame` in a
    box (`planetmodel.cartesian`), whose points (x, y, z), z up, are the
    box's own.
    """

    #: The fault, in the geometry's lengths.
    fault: Fault
    _: KW_ONLY
    #: The fault's name in the mesh and the manifest; None for the default.
    name: str | None = None
    #: The refinement towards the fault, or None for none.
    sizing: InterfaceSizing | None = None
    #: The pull-back's control nodes each way on every patch, or on every
    #: piece of one cut at a knot; None to choose them by `rtol`.
    controls: int | None = None
    #: The pull-back's offset, as a fraction of each patch's size, met
    #: when `controls` is None.
    rtol: float = 1e-6
    #: How far below the outer boundary an edge may lie and still be
    #: carried across it; None for the mapping's relief there.
    reach: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.fault, Fault):
            raise TypeError(f"fault must be a planetmodel.fault.Fault, got "
                            f"{type(self.fault).__name__}")
        for k, patch in enumerate(self.fault.patches):
            if not callable(getattr(patch, "spline", None)):
                raise TypeError(f"patch {k} ({type(patch).__name__}) gives no "
                                "spline of its map to build the mesh from")
        if self.name is not None and (not isinstance(self.name, str)
                                      or not self.name):
            raise ValueError(f"a fault's name is a non-empty string, got "
                             f"{self.name!r}")
        if self.sizing is not None and not isinstance(self.sizing, InterfaceSizing):
            raise TypeError(f"sizing must be an InterfaceSizing, got "
                            f"{type(self.sizing).__name__}")
        if self.controls is not None and (isinstance(self.controls, bool)
                                          or not isinstance(self.controls, int)
                                          or self.controls < 4):
            raise ValueError(f"controls must be an integer of at least 4 (a "
                             f"bicubic patch), got {self.controls!r}")
        if not self.rtol > 0.0:
            raise ValueError(f"rtol must be positive, got {self.rtol}")
        if self.reach is not None and not self.reach >= 0.0:
            raise ValueError(f"reach must not be negative, got {self.reach}")


@dataclass(frozen=True)
class MeshSpec:
    """The complete description of a mesh to build.

    Surgery is done on the geometry before it comes here.  The
    computational domain is the geometry followed by the shells; its
    layers are numbered 1..N from the centre and its interfaces 1..M in
    the order of `geometry.interfaces`, then the shells' outer
    boundaries.  The faults' faces take boundary attributes M + 1..M + F
    in the order of `faults`.  A flat geometry's layers and interfaces
    are numbered from the surface down, and its faces (`manifest.FACES`)
    take M + 1 onward, its faults the attributes after them; it takes no
    shells, a 'free' outer boundary, and faults over a `Box` alone
    (module docstring).
    """

    #: The geometry to mesh: a skeleton, a mapping and names, spherical
    #: (`Geometry`) or flat (`cartesian.Geometry`).
    geometry: GeometryBase
    #: The rule giving every interface of the computational domain its sizing.
    sizing: SizingRule
    _: KW_ONLY
    #: 3 for balls and boxes, 2 for discs and sections; None for the
    #: geometry's own: 3, but 2 over a `Rectangle`.
    dimension: int | None = None
    #: The element order, 1..3.
    order: int = 2
    #: Shells appended outside the geometry, innermost first.
    shells: Sequence[Shell] = ()
    #: Extra size fields composed into the sizing by minimum; each can
    #: only refine, never coarsen.
    refinements: Sequence[Refinement] = ()
    #: Faults built into the mesh, each within the geometry's layers.
    faults: Sequence[EmbeddedFault] = ()
    #: "free": the outer boundary of the computational domain may carry
    #: topography; "spherical": the mapping must be the identity on it.
    outer_boundary: str = "free"
    #: gmsh's Mesh.Algorithm.
    algorithm_2d: int = 6
    #: gmsh's Mesh.Algorithm3D.
    algorithm_3d: int = 1
    #: The gmsh optimisation of the linear volume mesh before it is
    #: curved: a method name, a sequence applied in order, or None for
    #: none.  3D only; the Delaunay mesher leaves slivers between
    #: interfaces that a Netgen pass removes without moving their nodes.
    #: With faults the Netgen pass is skipped, with a warning: it crashes
    #: on a solid with a surface embedded in it.
    optimise: str | Sequence[str] | None = "Netgen"
    #: False writes a failing mesh; the failures are on the result's report.
    validate: bool = True
    #: Copied into the manifest's `meta` block, for the consumer's own use.
    meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.geometry, GeometryBase):
            raise TypeError(
                "geometry must be a Geometry (spherical or cartesian), got "
                f"{type(self.geometry).__name__}")
        layout = layout_of(self.geometry)
        if not callable(self.sizing):
            raise TypeError("sizing must be a callable sizing rule")
        if self.dimension is None:
            object.__setattr__(self, "dimension", layout.dimensions[0])
        if self.dimension not in layout.dimensions:
            allowed = " or ".join(map(str, sorted(layout.dimensions)))
            raise ValueError(f"dimension must be {allowed} for {layout.what}, got "
                             f"{self.dimension}")
        if not 1 <= self.order <= 3:
            raise ValueError(f"element order must be 1..3, got {self.order}")
        if self.outer_boundary not in OUTER_BOUNDARIES:
            raise ValueError(
                f"outer_boundary must be one of {OUTER_BOUNDARIES}, "
                f"got {self.outer_boundary!r}")
        object.__setattr__(self, "optimise", optimise_methods(self.optimise))
        object.__setattr__(self, "shells", tuple(self.shells))
        for shell in self.shells:
            if not isinstance(shell, Shell):
                raise TypeError(
                    f"shells must be Shell instances, got {type(shell).__name__}")
        object.__setattr__(self, "faults", tuple(self.faults))
        layout.check(self)
        object.__setattr__(self, "refinements", tuple(self.refinements))
        for i, fine in enumerate(self.refinements):
            if not callable(fine):
                raise TypeError(
                    f"refinements[{i}] is {type(fine).__name__}, not a callable "
                    "size field")
            for name in ("size", "far_size"):
                value = getattr(fine, name, None)
                if not isinstance(value, (int, float)) or not value > 0.0:
                    raise ValueError(
                        f"refinements[{i}] must carry a positive {name}, "
                        f"got {value!r}")
            if fine.far_size < fine.size:
                raise ValueError(
                    f"refinements[{i}] has far_size ({fine.far_size}) below its "
                    f"size ({fine.size}): it would coarsen towards its target")
            try:
                probe = np.shape(fine(np.zeros((1, self.dimension))))
            except Exception as error:
                raise ValueError(
                    f"refinements[{i}] cannot be evaluated at points of a "
                    f"{self.dimension}-dimensional mesh: {error}") from error
            if probe != (1,):
                raise ValueError(
                    f"refinements[{i}] returns shape {probe} for one point; a "
                    "refinement returns one size per point")
        inner = float(self.geometry.skeleton.boundaries[-1])
        for name, shell in zip(self.shell_names, self.shells):
            outer = shell.outer_radius(inner)
            if not outer > inner:
                raise ValueError(
                    f"shell {name!r} has outer radius {outer:g}, not above "
                    f"the boundary it is attached to at {inner:g}")
            inner = outer
        domain = self.domain    # names must be unique across geometry and shells
        for i, item in enumerate(self.faults):
            if not isinstance(item, EmbeddedFault):
                raise TypeError(
                    f"faults[{i}] is {type(item).__name__}; wrap a Fault in an "
                    "EmbeddedFault")
        if self.faults and self.dimension != 3:
            raise ValueError("faults are surfaces of a 3D domain; a 2D mesh "
                             "takes none")
        names = self.fault_names
        if len(set(names)) != len(names):
            raise ValueError(f"fault names must be unique, got {names}")
        taken = {f.name for f in domain.interfaces if f.name} | {
            f"interface_{k + 1}" for k in range(len(domain.interfaces))}
        clash = sorted(taken.intersection(names))
        if clash:
            raise ValueError(f"faults named {clash} would share a name with an "
                             "interface, and both label boundary elements")
        layout.check_faults(self)

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
    def shell_names(self) -> list[str]:
        """The layer names of the shells, the defaults of `Shell` filled in."""
        many = len(self.shells) > 1
        return [shell.name if shell.name is not None
                else (f"buffer_{k}" if many else "buffer")
                for k, shell in enumerate(self.shells, start=1)]

    @property
    def fault_names(self) -> list[str]:
        """The names of the faults, the default `fault_<k>` filled in."""
        return [item.name if item.name is not None else f"fault_{k}"
                for k, item in enumerate(self.faults, start=1)]

    @property
    def outer_radius(self) -> float:
        """The outer radius of the computational domain: its last boundary,
        which for a box is the depth of its base."""
        radii = self.shell_radii
        return radii[-1] if radii else float(self.geometry.skeleton.boundaries[-1])

    @property
    def domain(self) -> GeometryBase:
        """The computational domain: the geometry then the shells, unchecked.

        A Geometry under the geometry's own mapping, whose layers and
        interfaces carry the numbering the mesh will; a flat geometry,
        which takes no shells, is its own.  Whether the mapping is
        defined on the shells is the builder's check, not this one's.
        """
        g = self.geometry
        if not self.shells:
            return g
        sk = g.skeleton.extended(self.shell_radii)
        return Geometry(
            sk, mapping=g.mapping,
            layer_names=[lay.name for lay in g.layers] + self.shell_names,
            interface_names=[f.name for f in g.interfaces] + [None] * len(self.shells),
            rtol=g.rtol, check=False)

    @property
    def layers(self) -> tuple[LayerInfo, ...]:
        """The layers of the computational domain, in the skeleton's order:
        centre outward, or from the surface down."""
        return self.domain.layers

    @property
    def interfaces(self) -> tuple[InterfaceInfo, ...]:
        """The interfaces of the computational domain, in the skeleton's
        order: centre outward, or from the surface down."""
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
    #: Boundary faces whose normal points towards the centre they enclose,
    #: or against the outward direction a plane of a box states.
    inward_faces: int = 0
    #: Faces of a fault whose normal does not run from its footwall into
    #: its hanging wall.
    misoriented_fault_faces: int = 0
    #: Faces of a fault that are not a face of exactly two cells.
    nonconforming_fault_faces: int = 0
    #: The largest distance of a tagged interface's mean radius (across a
    #: box, its mean depth) from the one asked for, in the mesh's lengths.
    max_interface_radius_error: float = 0.0
    #: The number of physical groups found, by "layers" and "interfaces",
    #: and "faces" and "faults" where the mesh has them.
    group_counts: dict[str, int] = field(default_factory=dict)
    #: Every failed check, in words.
    failures: list[str] = field(default_factory=list)
    #: Every warning, in words.
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        """Whether every check passed."""
        return not self.failures

    def raise_if_failed(self) -> "ValidationReport":
        """Raise ValueError listing every failure, or return self."""
        if self.failures:
            raise ValueError(
                "the generated mesh failed validation:\n  - "
                + "\n  - ".join(self.failures))
        return self

    def _faults_text(self) -> str:
        """The faults' checks and the warnings, for a mesh with faults: ""
        for one without."""
        faults = self.group_counts.get("faults", 0)
        if not faults:
            return ""
        text = (f"; {faults} fault{'s' if faults > 1 else ''}, "
                f"{self.misoriented_fault_faces} misoriented and "
                f"{self.nonconforming_fault_faces} nonconforming faces")
        return text + "".join(f"; warning: {w}" for w in self.warnings)

    def __repr__(self) -> str:
        state = "ok" if self.ok else f"{len(self.failures)} FAILED"
        return (f"ValidationReport({state}, minSICN {self.min_sicn:.4g}, "
                f"max interface error {self.max_interface_radius_error:.3g}"
                f"{self._faults_text()})")


@dataclass(frozen=True)
class MeshResult:
    """What a build produced.

    The mesh on disk is the reference mesh.  `mapping` is the geometry's
    own mapping, which acts on the mesh's coordinates as they are and is
    what the exporter applies to the nodes it reads back; it is None
    for a mesh that was not built from a geometry.  The keyword fields
    have defaults so a MeshResult written by hand still constructs.
    """

    #: The MSH 2.2 file.
    msh_path: Path
    #: The JSON manifest beside it.
    manifest_path: Path
    #: The geometry the mesh was built from; None for an offset mesh.
    geometry: GeometryBase | None
    #: Element, node, layer and interface counts.
    counts: dict[str, int]
    #: The checks the mesh passed or failed.
    validation: ValidationReport
    #: Seconds spent in each stage of the build.
    timings: dict[str, float]
    _: KW_ONLY
    #: The spec the mesh was built from, if any.
    spec: MeshSpec | None = None
    #: The geometry's mapping, or None where there was no geometry.
    mapping: Mapping | None = None
    #: Each fault's name -> the preimages of the surfaces built in: one a
    #: patch, or one a piece of a patch where it was cut at a knot of the
    #: mapping or continued past an edge (`fault.Region`), in the order
    #: of the patches.
    preimages: dict[str, tuple[Preimage, ...]] = field(default_factory=dict)

    def summary(self) -> str:
        """The build in one line: counts, worst quality, worst interface
        error, for a mesh with faults the faults' checks and the warnings,
        and seconds in all and by stage."""
        v = self.validation
        stages = ", ".join(f"{k} {t:.2g}" for k, t in self.timings.items())
        total = sum(self.timings.values())
        return (f"{self.counts.get('elements', '?')} elements, "
                f"{self.counts.get('nodes', '?')} nodes in "
                f"{self.counts.get('layers', '?')} layers; "
                f"minSICN {v.min_sicn:.3g}, max interface error "
                f"{v.max_interface_radius_error:.3g}{v._faults_text()}; "
                f"{total:.3g} s ({stages})")

    def __repr__(self) -> str:
        n = self.counts.get("elements", "?")
        return (f"MeshResult({self.msh_path.name}, {n} elements, "
                f"{self.counts.get('layers', '?')} layers)")
