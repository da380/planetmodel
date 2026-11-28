"""A skeleton placed in the physical world by one continuous mapping.

A geometry is a Skeleton, a Mapping from the reference body to the
physical one, and the names of the layers and interfaces.  It is the
object a mesher takes and the object on which fields are later hung.
`GeometryBase` is what every geometry shares; a leaf states its
coordinate system, which boundaries are interfaces, and the lattice
covering its body.  `Geometry` is the spherical leaf, a ball or a shell
in (r, theta, phi); `planetmodel.cartesian.Geometry` is the flat one,
layers in depth over a lateral domain.

The mapping is one continuous map of the whole reference domain,
orientation-preserving, whose gradient may jump only at the values of
the layered coordinate the mapping declares, its knots: skeleton
boundaries, or values inside the layers where a relief sits on a
surface that is no boundary of the model or a mapping read from a mesh
or a grid has an element edge.  Those are the invariants a geometry
checks on construction: the mapping satisfies the protocol, its
Jacobian is positive on a lattice covering every layer and the whole
body, and it is continuous across every interior boundary.  The
geometry's knots are those of the mapping's within the skeleton; a
mapping may be defined on a larger body, and a truncated or hollowed
geometry keeps its mapping.  A piecewise construction is therefore
legitimate as long as the pieces agree where they meet.

Interfaces are the boundaries that separate two layers, plus those of
the skeleton's ends the leaf counts, numbered in the skeleton's order;
`between` is (the layer below, the layer above), -1 standing for the
outside.  On the sphere the outer boundary is an interface and the
centre is not: for a full geometry (innermost radius zero) interface k
is skeleton boundary k + 1, and `between = (k, k + 1)` with -1 standing
for the outside.  For a hollow geometry the inner boundary is interface
0 with `between = (-1, 0)`, so interface k is skeleton boundary k.

The lattice of those checks is the geometry's `lattice`, in its
`coordinates` (`SPHERICAL`'s lattice on the sphere), and is the sample
the mapping's `validity(sample=..., coordinates=...)` takes; the
messages, and the repr's skeleton, name the system's coordinates.  A
leaf states the class of a field of its layered coordinate alone
(`radial_field_type`: `RadialField` on the sphere, `cartesian.DepthField`
in a box), which whatever builds such a field for a geometry's layers
asks for: a model's coarsening, a deck read, a constant put on a layer.

Nothing here knows about units: radii are numbers and every tolerance
is `rtol` times the skeleton's span.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import KW_ONLY, dataclass
from typing import ClassVar

import numpy as np
from numpy.typing import ArrayLike

from .coordinates import SPHERICAL, Coordinates, increases_outward
from .displacement import RadialDisplacement, SphericalFunction
from .fields import RadialField
from .mapping import (IdentityMapping, Mapping, RadialStretch, SphericalSample,
                      ValidityReport, scaled_mapping, validity)
from .skeleton import CoarseningMap, Skeleton

__all__ = ["GeometryBase", "Geometry", "LayerInfo", "InterfaceInfo", "Names",
           "Renames"]

#: Optional names for the layers or interfaces of a geometry, one per item
#: in order, None where an item is unnamed; None for no names at all.
type Names = Sequence[str | None] | None

#: What `renamed` takes: a full sequence of names, or a dict from index or
#: current name to the new name (None to unname).
type Renames = Sequence[str | None] | dict[int | str, str | None] | None


@dataclass(frozen=True)
class LayerInfo:
    """One layer of a geometry: its index, interval and optional name."""

    index: int
    interval: tuple[float, float]
    _: KW_ONLY
    name: str | None = None


@dataclass(frozen=True)
class InterfaceInfo:
    """One interface of a geometry.

    `radius` is the boundary it sits on, a value of the layered
    coordinate: a radius on the sphere, a depth in a box.  `between` is
    (layer below, layer above); -1 stands for the outside of the
    geometry, so on the sphere the outer interface has `between = (L - 1,
    -1)` and a hollow geometry's inner interface has `between = (-1, 0)`,
    and in a box the surface has `between = (0, -1)`.
    """

    index: int
    radius: float
    between: tuple[int, int]
    _: KW_ONLY
    name: str | None = None


def _names(names: Names, n: int, what: str) -> tuple[str | None, ...]:
    """A tuple of n optional names, unique among those given."""
    if names is None:
        return (None,) * n
    names = tuple(None if s is None else str(s) for s in names)
    if len(names) != n:
        raise ValueError(f"got {len(names)} {what} names for {n} {what}s")
    given = [s for s in names if s is not None]
    if len(given) != len(set(given)):
        raise ValueError(f"{what} names must be unique; got {given}")
    return names


class GeometryBase:
    """What every geometry shares: a skeleton, a mapping and names; see
    the module docstring.

    A leaf states its `_coordinates` and `radial_field_type`, which
    boundaries are interfaces and the layers on either side
    (`_face_boundaries`, `_between`), the lattice covering its body
    (`_lattice`) and how a copy is built (`_made`); everything else is
    here.  `check=False` skips the invariant checks, for a mapping already
    known to satisfy them.
    """

    _coordinates: Coordinates

    #: The class of a field of the layered coordinate alone in the
    #: geometry's system, built without naming the system.
    radial_field_type: ClassVar[type[RadialField]]

    def __init__(self, skeleton: Skeleton, *, mapping: Mapping | None = None,
                 layer_names: Names = None, interface_names: Names = None,
                 rtol: float = 1e-9, check: bool = True) -> None:
        if not isinstance(skeleton, Skeleton):
            raise TypeError(f"expected a Skeleton, got {type(skeleton).__name__}")
        self._sk = skeleton
        self._m = IdentityMapping() if mapping is None else mapping
        self._rtol = float(rtol)
        if not self._rtol > 0.0:
            raise ValueError(f"rtol must be positive, got {rtol}")
        self._layer_names = _names(layer_names, skeleton.nlayers, "layer")
        self._face_names = _names(interface_names, self._n_interfaces(), "interface")
        if check:
            self._check_mapping()

    # -- what a leaf states -------------------------------------------------

    def _face_boundaries(self, skeleton: Skeleton) -> range:
        """The indices of the boundaries of `skeleton` that are interfaces,
        in increasing order: interface k sits on the k-th of them."""
        raise NotImplementedError

    def _between(self, j: int, nlayers: int) -> tuple[int, int]:
        """(layer below, layer above) of the interface on boundary j of a
        skeleton of `nlayers` layers, -1 for the outside."""
        raise NotImplementedError

    def _lattice(self, **sizes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """The lattice of `lattice`, with the sizes given."""
        raise NotImplementedError

    def _made(self, skeleton: Skeleton, *, k: float, mapping: Mapping,
              layer_names: Names, interface_names: Names, rtol: float,
              check: bool) -> "GeometryBase":
        """A geometry of this leaf over `skeleton`, whatever else it holds
        scaled by `k` (its lateral domain, in a box)."""
        raise NotImplementedError

    # -- construction checks ------------------------------------------------

    def _n_interfaces(self) -> int:
        return len(self._face_boundaries(self._sk))

    def _check_mapping(self) -> None:
        """The invariants: protocol, validity, continuity."""
        m = self._m
        if not isinstance(m, Mapping):
            raise TypeError(
                f"{type(m).__name__} is not a Mapping: it needs __call__, "
                "deformation_gradient and jacobian on (..., 3) points")
        tol = self._rtol * self._sk.span
        report = validity(m, sample=self.lattice(), coordinates=self.coordinates)
        if not report:
            raise ValueError(f"the mapping does not preserve orientation: {report!r}")
        self._check_continuity(tol)

    def _check_continuity(self, tol: float) -> None:
        """m agrees across every interior boundary to tol, on a set of directions."""
        coords = self.coordinates
        c, a, b = coords.labels
        _, theta, phi = self.lattice()
        theta = theta.reshape(-1)[:, None]
        phi = phi.reshape(-1)[None, :]
        for r in self._sk.inner_boundaries:
            below = coords.points(r - tol, theta, phi)
            above = coords.points(r + tol, theta, phi)
            gap = np.linalg.norm(np.asarray(self._m(above), dtype=float)
                                 - np.asarray(self._m(below), dtype=float), axis=-1)
            # the images of two points 2 tol apart differ by 2 tol times the
            # stretch when m is continuous; allow a generous factor for it
            worst = float(np.max(gap))
            if worst > 100.0 * tol:
                k = int(np.argmax(gap))
                th = float(np.broadcast_to(theta, gap.shape).reshape(-1)[k])
                ph = float(np.broadcast_to(phi, gap.shape).reshape(-1)[k])
                raise ValueError(
                    f"the mapping is discontinuous across the boundary at "
                    f"{c} {float(r):g}: images differ by {worst:.3g} at "
                    f"({a}, {b}) = ({th:.3f}, {ph:.3f})")

    # -- what it is ---------------------------------------------------------

    @property
    def skeleton(self) -> Skeleton:
        """The boundaries the geometry is built over: radii on the sphere,
        depths in a box."""
        return self._sk

    @property
    def coordinates(self) -> Coordinates:
        """The coordinate system of the skeleton, the checks and the
        fields hung on the geometry: `SPHERICAL` for a `Geometry`."""
        return self._coordinates

    @property
    def mapping(self) -> Mapping:
        """The mapping taking the reference body to the physical one."""
        return self._m

    @property
    def rtol(self) -> float:
        """The relative tolerance every check of this geometry uses."""
        return self._rtol

    @property
    def nlayers(self) -> int:
        """The number of layers."""
        return self._sk.nlayers

    @property
    def is_identity(self) -> bool:
        """Whether the mapping is known to move nothing."""
        return bool(getattr(self._m, "is_identity", False))

    @property
    def is_hollow(self) -> bool:
        """Whether the skeleton's first boundary is above zero: a ball with
        a cavity, a box whose top lies below the free surface."""
        return self._sk.is_hollow

    @property
    def layers(self) -> tuple[LayerInfo, ...]:
        """Index, interval and name of every layer, in the skeleton's order:
        centre outward on the sphere, from the surface down in a box."""
        return tuple(LayerInfo(i, self._sk.interval(i), name=self._layer_names[i])
                     for i in range(self._sk.nlayers))

    @property
    def interfaces(self) -> tuple[InterfaceInfo, ...]:
        """Index, place, `between` and name of every interface, in the
        skeleton's order; see the module docstring for the numbering."""
        b = self._sk.boundaries
        L = self._sk.nlayers
        return tuple(InterfaceInfo(k, float(b[j]), self._between(j, L),
                                   name=self._face_names[k])
                     for k, j in enumerate(self._face_boundaries(self._sk)))

    def layer(self, which: int | str) -> LayerInfo:
        """A layer by index (negatives count back) or by name."""
        return self.layers[self._resolve(which, self._layer_names, "layer")]

    def interface(self, which: int | str) -> InterfaceInfo:
        """An interface by index (negatives count back) or by name."""
        return self.interfaces[self._resolve(which, self._face_names, "interface")]

    @staticmethod
    def _resolve(which: int | str, names: Sequence[str | None], what: str) -> int:
        if isinstance(which, str):
            if which not in names:
                named = [s for s in names if s is not None]
                raise KeyError(f"no {what} named {which!r}; named {what}s: {named}")
            return names.index(which)
        n = len(names)
        i = int(which)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"{what} index out of range for {n} {what}s")
        return i

    @property
    def knots(self) -> tuple[float, ...]:
        """The values of the layered coordinate within the skeleton, to
        `rtol` of its span, where the mapping declares its gradient may
        jump, in increasing order."""
        tol = self._rtol * self._sk.span
        b = self._sk.boundaries
        return tuple(sorted(float(k) for k in getattr(self._m, "knots", ())
                            if b[0] - tol <= float(k) <= b[-1] + tol))

    def lattice(self, *, n_c: int | None = None, n_a: int | None = None,
                n_b: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(c, a, b) broadcastable arrays covering every layer and the whole
        body, in the geometry's coordinates: the sample of the invariant
        checks and of `validity`.  `n_c` values lie strictly inside each
        layer and `n_a` by `n_b` lateral ones cover the body, each leaf
        holding its own defaults: on the sphere `SPHERICAL.lattice`, over
        a box its lateral domain's."""
        sizes = {key: int(n) for key, n in (("n_c", n_c), ("n_a", n_a), ("n_b", n_b))
                 if n is not None}
        return self._lattice(**sizes)

    def validity(self, *, sample: SphericalSample | None = None) -> ValidityReport:
        """The mapping's validity report on `sample`, coordinates of the
        geometry's system, or on its `lattice`."""
        lattice = self.lattice() if sample is None else sample
        return validity(self._m, sample=lattice, coordinates=self.coordinates)

    # -- copies -------------------------------------------------------------

    def _copy(self, *, skeleton: Skeleton | None = None,
              mapping: Mapping | None = None, layer_names: Names = None,
              interface_names: Names = None, check: bool = False,
              k: float = 1.0) -> "GeometryBase":
        return self._made(
            self._sk if skeleton is None else skeleton, k=k,
            mapping=self._m if mapping is None else mapping,
            layer_names=self._layer_names if layer_names is None else layer_names,
            interface_names=(self._face_names if interface_names is None
                             else interface_names),
            rtol=self._rtol, check=check)

    def _face_values(self, skeleton: Skeleton) -> list[float]:
        """The boundaries of `skeleton` that are interfaces, in order."""
        return [float(skeleton.boundaries[j]) for j in self._face_boundaries(skeleton)]

    def _name_cut(self, skeleton: Skeleton, j: int, names: list[str | None],
                  name: str | None) -> None:
        """`name`, when given, on the interface a cut made at boundary j of
        `skeleton`; refused where that boundary is no interface."""
        if name is None:
            return
        faces = list(self._face_boundaries(skeleton))
        if j not in faces:
            raise ValueError(
                f"the cut at {self.coordinates.labels[0]} "
                f"{float(skeleton.boundaries[j]):g} is no interface of the result, "
                f"so it takes no name; got {name!r}")
        names[faces.index(j)] = name

    def renamed(self, *, layers: Renames = None,
                interfaces: Renames = None) -> "GeometryBase":
        """A copy with layer and interface names replaced.

        Each argument is a full sequence of names, or a mapping from
        index or current name to the new name.
        """
        return self._copy(
            layer_names=_updated(self._layer_names, layers),
            interface_names=_updated(self._face_names, interfaces))

    def with_mapping(self, mapping: Mapping, *, check: bool = True) -> "GeometryBase":
        """A copy with another mapping, checked unless told otherwise."""
        return self._copy(mapping=mapping, check=check)

    def scaled(self, k: float) -> "GeometryBase":
        """The same geometry with every length multiplied by k."""
        k = float(k)
        sk = Skeleton(self._sk.boundaries * k)
        m = self._m if self.is_identity else scaled_mapping(self._m, k)
        return self._copy(skeleton=sk, mapping=m, k=k)

    # -- surgery ------------------------------------------------------------

    def refined(self, radii: ArrayLike, *,
                interface_names: Names = None) -> "GeometryBase":
        """Interior boundaries inserted; the mapping is kept.

        A split layer loses its name; `interface_names` name the new
        interfaces, one per radius in the order given.
        """
        radii = [float(r) for r in np.atleast_1d(np.asarray(radii, dtype=float))]
        sk = self._sk.refined(radii)
        new_face_names = _names(interface_names, len(radii), "interface")
        by_radius = dict(zip(sorted(radii), [new_face_names[radii.index(r)]
                                             for r in sorted(radii)]))
        old_faces = {f.radius: f.name for f in self.interfaces}
        layer_names = []
        for i in range(sk.nlayers):
            lo, hi = sk.interval(i)
            src = self._sk.locate(0.5 * (lo + hi)).layer
            olo, ohi = self._sk.interval(src)
            layer_names.append(self._layer_names[src]
                               if (lo == olo and hi == ohi) else None)
        face_names = [by_radius.get(r, old_faces.get(r)) for r in self._face_values(sk)]
        return self._copy(skeleton=sk, layer_names=layer_names,
                          interface_names=face_names)

    def truncated(self, radius: float, *, name: str | None = None) -> "GeometryBase":
        """The geometry cut at `radius`, the part beyond it dropped; the
        mapping is kept.

        On the sphere the cut is the new outer interface; a cut on an
        existing boundary keeps that interface's name unless `name` is
        given; the shortened layer keeps its name.  In a box the cut is
        the new base, a face and no interface, which takes no name.
        """
        sk = self._sk.truncated(radius)
        old_faces = {f.radius: f.name for f in self.interfaces}
        layer_names = list(self._layer_names[:sk.nlayers])
        face_names = [old_faces.get(r) for r in self._face_values(sk)]
        self._name_cut(sk, sk.nlayers, face_names, name)
        return self._copy(skeleton=sk, layer_names=layer_names,
                          interface_names=face_names)

    def hollowed(self, radius: float, *, name: str | None = None) -> "GeometryBase":
        """The geometry cut at `radius`, the part before it dropped; the
        mapping is kept.

        The result is hollow, with the cut as its first interface: the
        inner one on the sphere, the top in a box.  A cut on an existing
        boundary keeps that interface's name unless `name` is given; the
        shortened layer keeps its name.
        """
        sk = self._sk.hollowed(radius)
        old_faces = {f.radius: f.name for f in self.interfaces}
        dropped = self._sk.nlayers - sk.nlayers
        layer_names = list(self._layer_names[dropped:])
        face_names = [old_faces.get(r) for r in self._face_values(sk)]
        self._name_cut(sk, 0, face_names, name)
        return self._copy(skeleton=sk, layer_names=layer_names,
                          interface_names=face_names)

    def extended(self, radii: ArrayLike, *, layer_names: Names = None,
                 interface_names: Names = None) -> "GeometryBase":
        """Layers appended beyond the last boundary; identity mapping only.

        `layer_names` name the new layers and `interface_names` the interfaces
        they bring, one each: the new layers' outer boundaries on the
        sphere, their tops in a box (the old base first).  A geometry
        whose mapping is not the identity is refused: extend the
        skeleton first and build the mapping after.
        """
        self._require_identity("extended")
        radii = [float(r) for r in np.atleast_1d(np.asarray(radii, dtype=float))]
        sk = self._sk.extended(radii)
        n = len(radii)
        return self._copy(
            skeleton=sk,
            layer_names=self._layer_names + _names(layer_names, n, "layer"),
            interface_names=self._face_names + _names(interface_names, n, "interface"))

    def coarsened(self, *, keep: Iterable[int | str] | None = None,
                  drop: Iterable[int | str] | None = None
                  ) -> tuple["GeometryBase", CoarseningMap]:
        """Interior interfaces removed; identity mapping only.

        Exactly one of `keep` and `drop` selects interior interfaces by
        index or name, as `interface` takes them; the outermost interface
        (the surface of a ball or of a box), and the inner boundary of a
        hollow ball, are not selectable.  A merged layer's name is None;
        kept interfaces keep theirs.
        """
        if (keep is None) == (drop is None):
            raise ValueError("give exactly one of keep and drop")
        which = [self._interior(w) for w in (drop if keep is None else keep)]
        self._require_identity("coarsened")
        sk, cmap = (self._sk.coarsened(drop=which) if keep is None
                    else self._sk.coarsened(keep=which))
        layer_names = [self._layer_names[fine[0]] if len(fine) == 1 else None
                       for fine in cmap.layers]
        old_faces = {f.radius: f.name for f in self.interfaces}
        face_names = [old_faces.get(r) for r in self._face_values(sk)]
        return (self._copy(skeleton=sk, layer_names=layer_names,
                           interface_names=face_names), cmap)

    def _interior(self, which: int | str) -> int:
        """Interface `which`, by index or name, as an index into the
        skeleton's interior boundaries; an end of the skeleton is refused."""
        face = self.interface(which)
        b = self._sk.boundaries
        L = self._sk.nlayers
        j = int(np.flatnonzero(b == face.radius)[0])
        if j == (L if increases_outward(self.coordinates) else 0):
            raise ValueError(
                f"interface {face.name or face.index!r} is the outermost "
                "boundary and cannot be selected for coarsening")
        if not 0 < j < L:
            raise ValueError(
                f"interface {face.name or face.index!r} is the inner "
                "boundary of a hollow geometry and cannot be selected")
        return j - 1

    def _require_identity(self, verb: str) -> None:
        if not self.is_identity:
            raise ValueError(
                f"a geometry can only be {verb} while its mapping is the "
                "identity: do the surgery on the skeleton first and build "
                "the mapping for the result")


class Geometry(GeometryBase):
    """A spherical geometry, a ball or a shell in (r, theta, phi): a
    skeleton of radii, a mapping and names; see the module docstring.

    `check=False` skips the invariant checks, for a mapping already known
    to satisfy them.
    """

    _coordinates: Coordinates = SPHERICAL
    radial_field_type: ClassVar[type[RadialField]] = RadialField

    def _face_boundaries(self, skeleton: Skeleton) -> range:
        """Every boundary but the centre: from boundary 1 on a full skeleton,
        from boundary 0 on a hollow one."""
        return range(0 if skeleton.is_hollow else 1, skeleton.boundaries.size)

    def _between(self, j: int, nlayers: int) -> tuple[int, int]:
        """(j - 1, j), the outside -1: the layer below is the inner one."""
        return (j - 1, j if j < nlayers else -1)

    def _lattice(self, **sizes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.coordinates.lattice(self._sk, **sizes)

    def _made(self, skeleton: Skeleton, *, k: float, mapping: Mapping,
              layer_names: Names, interface_names: Names, rtol: float,
              check: bool) -> "Geometry":
        return Geometry(skeleton, mapping=mapping, layer_names=layer_names,
                        interface_names=interface_names, rtol=rtol, check=check)

    def stretched(self, h: RadialDisplacement | SphericalFunction, *,
                  name: str | None = None, check: bool = True) -> "Geometry":
        """A copy whose mapping is the radial stretch driven by `h`.

        `h` is any radial displacement; the stretch's `rmax` is this
        geometry's outer radius.
        """
        return self.with_mapping(RadialStretch(h, rmax=self._sk, name=name),
                                 check=check)

    def __repr__(self) -> str:
        m = "identity" if self.is_identity else repr(self._m)
        return (f"Geometry({self._sk.describe(self.coordinates.labels[0])}, "
                f"mapping={m})")


def _updated(current: tuple[str | None, ...], new: Renames) -> tuple[str | None, ...]:
    """Names after applying `new`: a full sequence, or a mapping of changes."""
    if new is None:
        return current
    if isinstance(new, dict):
        out = list(current)
        for key, value in new.items():
            i = current.index(key) if isinstance(key, str) else int(key)
            out[i] = None if value is None else str(value)
        return tuple(out)
    return tuple(new)
