"""How a geometry is meshed: concentric spheres about the origin, or a box
cut by horizontal planes.

The spec and the builder ask the layout of the geometry they mesh
(`layout_of`) for whatever depends on the body's shape: the dimensions
it is meshed in and what a spec of it may hold (`check`, and of its
faults `check_faults`), its CAD and the identification of its entities
(`build`, giving a `Tagging`, which then answers for each interface's
distance, each boundary's outward direction and an interface's
measured place), the length the sizing's scale is judged against
(`scale`), the tolerance an interface's place is held to (`tolerance`),
the records of its manifest (`records`), and for a fault built into it
the curves the mesh labels (`fault_curves`), the parts of its boundary
a fault may leave it through (`exits`) and what else a fault may not
reach (`check_reach`).

A geometry over a lateral domain (`planetmodel.cartesian.Geometry`,
whose `domain` is a `Box` or a `Rectangle`) is meshed as the box over
the domain's extent and the skeleton's depths, or as its cross-section, by
`Planar`: points (x, y, z) with z up, layer k between the planes
z = -depth of its top and bottom, the mesh three-dimensional over a
box and two-dimensional over a rectangle, whose coordinates are then
(x, z).  Any other geometry is a ball, a shell or a disc of concentric
spheres about the origin, with the spec's shells outside, by
`Concentric`.  Nothing here imports gmsh: what builds the CAD is
imported where it is built.

A fault leaves a body through its free surface, where its edge is its
free edge, its sides free: the outer sphere of a spherical geometry,
the top of a box.  A box stands for an unbounded body, so a fault may
leave it through its base or a side too, where its edge is its face
edge (`manifest.FLAT_FAULT_CURVES`), an artificial boundary; a
spherical geometry has no other way out, and a fault reaching its
centre or below a hollow one's inner boundary is refused.  Each way out
is an `Exit`: a level of one of the geometry's coordinates (c, a, b)
(`Coordinates`), the radius of the outer sphere, the depth of the top
or the base, or the x or y of a side, so that where a point lies
against it is read through the geometry's own coordinate system, the
one source of what a depth is.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol

import numpy as np
from numpy.typing import ArrayLike

from ..coordinates import Coordinates, increases_outward
from . import manifest

if TYPE_CHECKING:
    from ..cartesian.domains import Domain
    from ..geometry import GeometryBase
    from ._box import BoxGeometry
    from ._faults import CollarTagging
    from ._geometry import ConcentricGeometry
    from ._tagging import Tagging
    from .spec import MeshSpec

__all__ = ["Layout", "Concentric", "Planar", "layout_of", "PlanarFace", "Exit"]

#: A face of a flat mesh as the builder makes it: (name, axis, outward,
#: value), a row of `manifest.FACES` with the coordinate of its plane.
type PlanarFace = tuple[str, str, int, float]


@dataclass(frozen=True)
class Exit:
    """A part of the boundary of a body that a fault may leave it through
    (module docstring): the level `value` of coordinate `axis` of
    `coordinates`, (c, a, b) numbered 0, 1, 2, the body on the side of it
    opposite `outward`, +1 where the coordinate grows out of the body
    there and -1 where it shrinks.

    `interface` is the index, among the geometry's interfaces, of the
    interface it is, the free surface, where a fault's edge is its free
    edge; None for a face of a box, its base or a side, where a fault's
    edge is its face edge.  `name` names it in messages: the free
    surface's interface, or the face's own name (`manifest.FACES`).
    """

    #: The interface's or the face's name.
    name: str
    #: The coordinate system its level is read in: the geometry's.
    coordinates: Coordinates
    #: Which of the coordinates (c, a, b) it is a level of: 0, 1 or 2.
    axis: int
    #: Its level of that coordinate.
    value: float
    #: +1 or -1: the sense, along the coordinate, out of the body.
    outward: float
    #: The index of the interface it is (the free surface), or None (a face).
    interface: int | None

    @property
    def free(self) -> bool:
        """Whether it is the free surface, where a fault's edge is its free
        edge, rather than an artificial boundary."""
        return self.interface is not None

    @property
    def label(self) -> str:
        """The name of the coordinate it is a level of, for messages."""
        return self.coordinates.labels[self.axis]

    def level(self, points: ArrayLike) -> np.ndarray:
        """Its coordinate of points (..., 3), (...)."""
        return np.asarray(self.coordinates.split(points)[self.axis], dtype=float)

    def inside(self, points: ArrayLike) -> np.ndarray:
        """How far points (..., 3) lie inside it, along its coordinate:
        positive inside the body, negative beyond it."""
        return self.outward * (self.value - self.level(points))

    def beyond(self, points: ArrayLike, distance: float) -> np.ndarray:
        """Whether points (..., 3) lie beyond it, along its coordinate, by
        `distance` or more."""
        return (self.outward * self.level(points)
                >= self.outward * (self.value + self.outward * distance))

    def onto(self, points: ArrayLike) -> np.ndarray:
        """Points (..., 3) moved along its coordinate onto it, the others
        kept."""
        c, a, b, _ = self.coordinates.split(points)
        parts = [np.asarray(c, dtype=float), np.asarray(a, dtype=float),
                 np.asarray(b, dtype=float)]
        parts[self.axis] = np.full_like(parts[self.axis], self.value)
        return self.coordinates.points(*parts)

    def where(self) -> str:
        """Where it lies, for messages: "interface 'surface' (radius = 1)",
        "face 'x_max' (x = 1)"."""
        what = "interface" if self.free else "face"
        return f"{what} {self.name!r} ({self.label} = {self.value:g})"


class Layout(Protocol):
    """What the spec and the builder ask of the shape of a body (module
    docstring)."""

    #: The dimensions the body may be meshed in, the default first.
    dimensions: tuple[int, ...]
    #: What the body is, for messages.
    what: str
    #: The curves of a fault the mesh labels, in the order of their
    #: dimension-1 groups (`manifest.fault_curve_groups`).
    fault_curves: tuple[str, ...]

    def check(self, spec: MeshSpec) -> None:
        """Refuse, by name, what a spec of this body may not hold."""
        ...

    def check_faults(self, spec: MeshSpec) -> None:
        """Refuse, by name, a fault of the spec this body cannot take: one
        in a frame other than its own."""
        ...

    def scale(self, spec: MeshSpec) -> float:
        """The length the sizing's scale is judged against."""
        ...

    def tolerance(self, spec: MeshSpec) -> float | None:
        """The distance an interface's measured place may lie from the one
        asked for; None for `_validate.check_interface_radii`'s own."""
        ...

    def exits(self, spec: MeshSpec) -> tuple[Exit, ...]:
        """The parts of the body's boundary a fault may leave it through,
        the free surface first (module docstring)."""
        ...

    def check_reach(self, spec: MeshSpec, levels: np.ndarray, *, atol: float,
                    what: str) -> None:
        """Refuse a fault whose points lie at `levels` of the layered
        coordinate where no fault may reach, naming it by `what`."""
        ...

    def build(self, spec: MeshSpec, *, surfaces: Sequence[int] = ()
              ) -> tuple[ConcentricGeometry | BoxGeometry, Tagging]:
        """The CAD of the computational domain in the open gmsh session,
        the surfaces of faults fragmented in, and its entities identified."""
        ...

    def records(self, spec: MeshSpec, *, collared: CollarTagging | None
                ) -> dict[str, Any]:
        """The layers, interfaces and, of a flat mesh, faces and coordinates
        of the manifest, as `manifest.MeshManifest.from_build` takes them."""
        ...


def _outward(coordinates: Coordinates) -> float:
    """The sense of the layered coordinate out of the body through its
    free surface: +1 where it grows outward (a radius), -1 where it is a
    depth (`coordinates.increases_outward`)."""
    return 1.0 if increases_outward(coordinates) else -1.0


def _frames_of(spec: MeshSpec, reference: str, frame: str, what: str) -> None:
    """Refuse a fault of the spec whose frame has no `reference`, the
    attribute of the frame (`frame`) a body of this kind (`what`) takes
    faults in."""
    for name, item in zip(spec.fault_names, spec.faults):
        given = item.fault.frame
        if getattr(given, reference, None) is None:
            raise ValueError(
                f"fault {name!r} is in a {type(given).__name__}: {what} takes faults "
                f"in a {frame}; a fault in a Cartesian frame belongs in a regional "
                "box (a geometry of planetmodel.cartesian), and one in a "
                "GeographicFrame on a sphere")


class Concentric:
    """A ball, a shell or a disc of concentric spheres or circles about the
    origin, with the spec's shells outside: the mesh of a spherical
    geometry, three-dimensional or its equatorial disc."""

    dimensions: ClassVar[tuple[int, ...]] = (3, 2)
    what: ClassVar[str] = "a spherical geometry"
    #: The curves of a fault the mesh labels.
    fault_curves: ClassVar[tuple[str, ...]] = manifest.FAULT_CURVES

    def check(self, spec: MeshSpec) -> None:
        """Nothing beyond the spec's own checks."""

    def check_faults(self, spec: MeshSpec) -> None:
        """Refuse a fault in a frame without a reference sphere: a sphere
        takes faults in a `fault.GeographicFrame`."""
        _frames_of(spec, "radius", "GeographicFrame", self.what)

    def scale(self, spec: MeshSpec) -> float:
        """The outer radius of the computational domain."""
        return spec.outer_radius

    def tolerance(self, spec: MeshSpec) -> float | None:
        """None: a twentieth of the thinnest span, the check's own."""
        return None

    def exits(self, spec: MeshSpec) -> tuple[Exit, ...]:
        """The geometry's outer surface alone, its last interface: a sphere
        of the geometry's outer radius, the shells outside it holding no
        fault."""
        g = spec.geometry
        k = len(g.interfaces) - 1
        return (Exit(name=g.interfaces[k].name or f"interface_{k + 1}",
                     coordinates=g.coordinates, axis=0,
                     value=float(g.skeleton.boundaries[-1]),
                     outward=_outward(g.coordinates), interface=k),)

    def check_reach(self, spec: MeshSpec, levels: np.ndarray, *, atol: float,
                    what: str) -> None:
        """Refuse a fault through the centre (radius within `atol` of
        zero), where the spherical frame has no direction, and one reaching
        below the inner boundary of a hollow domain: a fault leaves a
        sphere through its outer surface alone."""
        lo = float(np.min(levels))
        inner = float(spec.geometry.skeleton.boundaries[0])
        if lo <= atol:
            raise ValueError(f"{what} passes through the centre, where the spherical "
                             "frame has no direction: a fault of a layered mesh "
                             "keeps clear of it")
        if inner > 0.0 and lo < inner:
            raise ValueError(f"{what} reaches r = {lo:.6g}, below the inner boundary "
                             f"of the hollow domain at r = {inner:g}: a fault leaves "
                             "the domain through its outer surface alone")

    def build(self, spec: MeshSpec, *, surfaces: Sequence[int] = ()
              ) -> tuple[ConcentricGeometry, Tagging]:
        """Concentric shells of the domain's radii, the surfaces trimmed at
        the geometry's outer boundary (`_geometry.build_concentric`),
        matched to the radii asked for (`_tagging.identify`)."""
        from ._geometry import build_concentric
        from ._tagging import identify
        domain = spec.domain
        cad = build_concentric(domain.skeleton.boundaries, dimension=spec.dimension,
                               surfaces=surfaces,
                               trim=float(spec.geometry.skeleton.boundaries[-1]))
        return cad, identify(cad, [f.radius for f in domain.interfaces])

    def records(self, spec: MeshSpec, *, collared: CollarTagging | None
                ) -> dict[str, Any]:
        """The layers by radius, shells flagged, with their collars'
        attributes where the mesh has faults, and the interfaces."""
        domain = spec.domain
        b = domain.skeleton.boundaries
        n_geometry = spec.geometry.nlayers
        layers = [manifest.LayerEntry.from_layer(
            lay, attribute=i + 1, r_inner=b[i], r_outer=b[i + 1],
            in_geometry=i < n_geometry,
            attributes=(None if collared is None
                        else collared.numbering.layer_attributes(i + 1)))
            for i, lay in enumerate(domain.layers)]
        interfaces = [manifest.InterfaceEntry.from_interface(
            face, attribute=k + 1, radius=face.radius)
            for k, face in enumerate(domain.interfaces)]
        return {"layers": layers, "interfaces": interfaces}


class Planar:
    """A box cut by horizontal planes, or the cross-section of one: the mesh of a
    flat geometry over its lateral domain (module docstring).

    Its coordinates are the domain's lateral axes and then z, (x, y, z)
    over a `Box` and (x, z) over a `Rectangle`, one dimension each; its
    faces are the rows of `manifest.FACES` whose axes it has, the base at
    z = -(the last depth) and each side at an end of the domain's extent,
    planes of the reference box which the mapping moves at the export (a
    vertical stretch keeps the sides, and moves the base where it gives
    the base a relief).  A fault, over a `Box` alone and under any
    mapping that inverts itself, through which it is pulled back,
    leaves the body through its top, the free surface, or through any of
    its faces.
    """

    def __init__(self, domain: Domain) -> None:
        self._domain = domain

    @property
    def coordinates(self) -> tuple[str, ...]:
        """The physical axes of the mesh's coordinates, in order."""
        return tuple(self._domain.axes) + ("z",)

    @property
    def dimensions(self) -> tuple[int, ...]:
        """The one dimension of the mesh: 3 over a box, 2 over a rectangle."""
        return (len(self.coordinates),)

    @property
    def what(self) -> str:
        """What the body is, for messages."""
        return ("a flat geometry over a Box" if len(self.coordinates) == 3
                else "a flat geometry over a Rectangle")

    def bounds(self) -> dict[str, tuple[float, float]]:
        """The extent of the domain along each lateral axis: the period,
        padding included, which the body a geometry over it holds."""
        extent = np.reshape(np.asarray(self._domain.extent, dtype=float), (-1, 2))
        return {axis: (float(lo), float(hi))
                for axis, (lo, hi) in zip(self._domain.axes, extent)}

    def faces(self, spec: MeshSpec) -> tuple[PlanarFace, ...]:
        """The faces of the mesh in the order of their attributes, each with
        the coordinate of its plane."""
        base = -float(spec.geometry.skeleton.boundaries[-1])
        bounds = self.bounds()
        out = []
        for name, axis, outward in manifest.FACES:
            if axis == "z":
                out.append((name, axis, outward, base))
            elif axis in bounds:
                out.append((name, axis, outward, bounds[axis][outward > 0]))
        return tuple(out)

    #: The curves of a fault the mesh labels: with its face edge, where it
    #: leaves the box through its base or a side.
    fault_curves: ClassVar[tuple[str, ...]] = manifest.FLAT_FAULT_CURVES

    def check(self, spec: MeshSpec) -> None:
        """Refuse shells (concentric layers outside a sphere: a box is
        padded below its base by its model), a spherical outer boundary,
        faults over a rectangle (not built), and an interface named as a
        face."""
        if spec.shells:
            raise ValueError(
                f"{self.what} takes no shells, the concentric layers outside a "
                "sphere: pad a model below its base with "
                "Model.extended(depths, fields='hold'), or widen its domain")
        if spec.outer_boundary != "free":
            raise ValueError(
                f"{self.what} has no spherical outer boundary: its faces are "
                "labelled, and carry no conditions; leave outer_boundary 'free'")
        if spec.faults and len(self.coordinates) != 3:
            raise ValueError(
                "faults in a flat geometry over a Rectangle are not "
                "built: a fault there is a segment "
                "of the (x, z) plane, which needs a fault of one parameter, the "
                "manifest's rules of a fault's faces in two dimensions (its sides "
                "by a segment's direction, its tips points, its walls of "
                "triangles) and the segment fragmented into the rectangle; mesh "
                "the box over a Box instead")
        names = {f.name for f in spec.geometry.interfaces if f.name}
        clash = sorted(names.intersection(name for name, _, _ in manifest.FACES))
        if clash:
            raise ValueError(f"interfaces named {clash} would share a name with a "
                             "face of the box, and both label boundary elements")

    def check_faults(self, spec: MeshSpec) -> None:
        """Refuse a fault in a frame without a reference plane, a box taking
        faults in a `fault.CartesianFrame` (points (x, y, z) with z up, the
        mesh's own), and a fault named as a face of the box."""
        _frames_of(spec, "surface", "CartesianFrame", self.what)
        clash = sorted(set(spec.fault_names).intersection(
            name for name, _, _ in manifest.FACES))
        if clash:
            raise ValueError(f"faults named {clash} would share a name with a face "
                             "of the box, and both label boundary elements")

    def scale(self, spec: MeshSpec) -> float:
        """The largest side of the body: its depth span or a lateral width."""
        b = spec.geometry.skeleton.boundaries
        return max([float(b[-1] - b[0])]
                   + [hi - lo for lo, hi in self.bounds().values()])

    def tolerance(self, spec: MeshSpec) -> float:
        """A twentieth of the thinnest layer, the base's included."""
        return 0.05 * float(np.min(np.diff(spec.geometry.skeleton.boundaries)))

    def exits(self, spec: MeshSpec) -> tuple[Exit, ...]:
        """The free surface, the top of the box at the skeleton's first
        depth, then its faces in the order of `manifest.FACES`: the base
        at its last depth, and each side at an end of the domain's extent,
        a level of the lateral coordinate of its axis."""
        g = spec.geometry
        coords = g.coordinates
        b = g.skeleton.boundaries
        up = _outward(coords)
        top = Exit(name=g.interfaces[0].name or "interface_1", coordinates=coords,
                   axis=0, value=float(b[0]), outward=up, interface=0)
        faces = []
        for name, axis, outward, value in self.faces(spec):
            if axis == "z":
                faces.append(Exit(name=name, coordinates=coords, axis=0,
                                  value=float(b[-1]), outward=-up, interface=None))
            else:
                faces.append(Exit(name=name, coordinates=coords,
                                  axis=coords.labels.index(axis), value=float(value),
                                  outward=float(outward), interface=None))
        return (top, *faces)

    def check_reach(self, spec: MeshSpec, levels: np.ndarray, *, atol: float,
                    what: str) -> None:
        """Nothing: every part of a box's boundary is a way out."""

    def build(self, spec: MeshSpec, *, surfaces: Sequence[int] = ()
              ) -> tuple[BoxGeometry, Tagging]:
        """A box of each layer, fragmented into one body with the surfaces
        of faults, which are trimmed at its boundary, its entities matched
        to the planes asked for (`_box.build_box`)."""
        from ._box import build_box
        return build_box(spec.geometry.skeleton.boundaries,
                         coordinates=self.coordinates, bounds=self.bounds(),
                         faces=self.faces(spec), scale=self.scale(spec),
                         surfaces=surfaces)

    def records(self, spec: MeshSpec, *, collared: CollarTagging | None
                ) -> dict[str, Any]:
        """The layers and interfaces by depth, the faces and the
        coordinates of a flat mesh (schema 7), with the attributes of the
        layers' collars where the mesh has faults."""
        g = spec.geometry
        b = g.skeleton.boundaries
        layers = [manifest.FlatLayerEntry.from_layer(
            lay, attribute=i + 1, depth_top=b[i], depth_bottom=b[i + 1],
            attributes=(None if collared is None
                        else collared.numbering.layer_attributes(i + 1)))
            for i, lay in enumerate(g.layers)]
        interfaces = [manifest.FlatInterfaceEntry.from_interface(
            face, attribute=k + 1, depth=face.radius)
            for k, face in enumerate(g.interfaces)]
        m = len(interfaces)
        faces = [manifest.FaceEntry(attribute=m + k + 1, name=name, axis=axis,
                                    value=float(value), outward=int(outward))
                 for k, (name, axis, outward, value) in enumerate(self.faces(spec))]
        return {"layers": layers, "interfaces": interfaces, "faces": faces,
                "coordinates": self.coordinates}


def layout_of(geometry: GeometryBase) -> Layout:
    """The layout of a geometry's mesh: over the lateral domain of a
    geometry that has one (`cartesian.Geometry`), the box or rectangle
    `Planar`; of any other, the concentric spheres `Concentric`."""
    domain = getattr(geometry, "domain", None)
    return Concentric() if domain is None else Planar(domain)
