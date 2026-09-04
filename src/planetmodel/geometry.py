"""A skeleton placed in the physical world by one continuous mapping.

A Geometry is a Skeleton, a Mapping from the reference ball to the
physical body, and the names of the layers and interfaces.  It is the
object a mesher takes and the object on which fields are later hung.

The mapping is one continuous map of the whole reference domain,
orientation-preserving, whose gradient may jump only across skeleton
boundaries.  Those are the invariants a Geometry checks on
construction: the mapping satisfies the protocol, every knot it declares
lies on a boundary, its Jacobian is positive on a lattice covering every
layer and the whole sphere, and it is continuous across every interior
boundary.  A piecewise construction is therefore legitimate as long as
the pieces agree where they meet.

Interfaces are the boundaries that separate two layers plus the outer
boundary.  For a full geometry (innermost radius zero) interface k is
skeleton boundary k + 1, and `between = (k, k + 1)` with -1 standing for
the outside.  For a hollow geometry the inner boundary is interface 0
with `between = (-1, 0)`, so interface k is skeleton boundary k.

Nothing here knows about units: radii are numbers and every tolerance
is `rtol` times the skeleton's span.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import numpy as np

from .frames import cartesian_points
from .mapping import IdentityMapping, Mapping, ScaledMapping, validity_lattice
from .skeleton import CoarseningMap, Skeleton

__all__ = ["Geometry", "LayerInfo", "InterfaceInfo"]


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

    `between` is (layer below, layer above); -1 stands for the outside of
    the geometry, so the outer interface has `between = (L - 1, -1)` and a
    hollow geometry's inner interface has `between = (-1, 0)`.
    """

    index: int
    radius: float
    between: tuple[int, int]
    _: KW_ONLY
    name: str | None = None


def _names(names, n: int, what: str) -> tuple[str | None, ...]:
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


class Geometry:
    """A skeleton, a mapping and names; see the module docstring.

    `check=False` skips the invariant checks, for a mapping already known
    to satisfy them.
    """

    def __init__(self, skeleton, *, mapping=None, layer_names=None,
                 interface_names=None, rtol: float = 1e-9,
                 check: bool = True) -> None:
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

    # -- construction checks ------------------------------------------------

    def _n_interfaces(self) -> int:
        return self._sk.nlayers + (1 if self._sk.is_hollow else 0)

    def _check_mapping(self) -> None:
        """The invariants: protocol, knots on boundaries, validity, continuity."""
        m = self._m
        if not isinstance(m, Mapping):
            raise TypeError(
                f"{type(m).__name__} is not a Mapping: it needs __call__, "
                "deformation_gradient and jacobian on (..., 3) points")
        tol = self._rtol * self._sk.span
        b = self._sk.boundaries
        for k in getattr(m, "knots", ()):
            if np.min(np.abs(b - float(k))) > tol:
                raise ValueError(
                    f"the mapping declares a kink at r = {float(k):g}, which is "
                    f"not a boundary of {self._sk!r}; refine the skeleton there "
                    "or move the kink")
        lattice = validity_lattice(self._sk)
        report = (m.is_valid(sample=lattice) if hasattr(m, "is_valid")
                  else _generic_validity(m, lattice))
        if not report:
            raise ValueError(f"the mapping does not preserve orientation: {report!r}")
        self._check_continuity(tol)

    def _check_continuity(self, tol: float) -> None:
        """m agrees across every interior boundary to tol, on a set of directions."""
        _, theta, phi = validity_lattice(self._sk)
        theta = theta.reshape(-1)[:, None]
        phi = phi.reshape(-1)[None, :]
        for r in self._sk.inner_boundaries:
            below = cartesian_points(r - tol, theta, phi)
            above = cartesian_points(r + tol, theta, phi)
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
                    f"r = {float(r):g}: images differ by {worst:.3g} at "
                    f"(theta, phi) = ({th:.3f}, {ph:.3f})")

    # -- what it is ---------------------------------------------------------

    @property
    def skeleton(self) -> Skeleton:
        return self._sk

    @property
    def mapping(self):
        return self._m

    @property
    def rtol(self) -> float:
        """The relative tolerance every check of this geometry uses."""
        return self._rtol

    @property
    def nlayers(self) -> int:
        return self._sk.nlayers

    @property
    def is_identity(self) -> bool:
        """Whether the mapping is known to move nothing."""
        return bool(getattr(self._m, "is_identity", False))

    @property
    def is_hollow(self) -> bool:
        return self._sk.is_hollow

    @property
    def layers(self) -> tuple[LayerInfo, ...]:
        return tuple(LayerInfo(i, self._sk.interval(i), name=self._layer_names[i])
                     for i in range(self._sk.nlayers))

    @property
    def interfaces(self) -> tuple[InterfaceInfo, ...]:
        b = self._sk.boundaries
        L = self._sk.nlayers
        faces = []
        first = 0 if self._sk.is_hollow else 1
        for k, j in enumerate(range(first, b.size)):
            below = j - 1
            above = j if j < L else -1
            faces.append(InterfaceInfo(k, float(b[j]), (below, above),
                                       name=self._face_names[k]))
        return tuple(faces)

    def layer(self, which) -> LayerInfo:
        """A layer by index (negatives count back) or by name."""
        return self.layers[self._resolve(which, self._layer_names, "layer")]

    def interface(self, which) -> InterfaceInfo:
        """An interface by index (negatives count back) or by name."""
        return self.interfaces[self._resolve(which, self._face_names, "interface")]

    @staticmethod
    def _resolve(which, names, what: str) -> int:
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

    def knots(self) -> tuple[float, ...]:
        """The radii where the mapping declares its gradient may jump."""
        return tuple(float(k) for k in getattr(self._m, "knots", ()))

    def validity(self, *, sample=None):
        """The mapping's validity report on `sample`, or on the lattice."""
        lattice = validity_lattice(self._sk) if sample is None else sample
        if hasattr(self._m, "is_valid"):
            return self._m.is_valid(sample=lattice)
        return _generic_validity(self._m, lattice)

    # -- copies -------------------------------------------------------------

    def _copy(self, *, skeleton=None, mapping=None, layer_names=None,
              interface_names=None, check: bool = False) -> "Geometry":
        return Geometry(
            self._sk if skeleton is None else skeleton,
            mapping=self._m if mapping is None else mapping,
            layer_names=self._layer_names if layer_names is None else layer_names,
            interface_names=(self._face_names if interface_names is None
                             else interface_names),
            rtol=self._rtol, check=check)

    def renamed(self, *, layers=None, interfaces=None) -> "Geometry":
        """A copy with layer and interface names replaced.

        Each argument is a full sequence of names, or a mapping from
        index or current name to the new name.
        """
        return self._copy(
            layer_names=_updated(self._layer_names, layers),
            interface_names=_updated(self._face_names, interfaces))

    def with_mapping(self, mapping, *, check: bool = True) -> "Geometry":
        """A copy with another mapping, checked unless told otherwise."""
        return self._copy(mapping=mapping, check=check)

    def scaled(self, k: float) -> "Geometry":
        """The same geometry with every length multiplied by k."""
        k = float(k)
        sk = Skeleton(self._sk.boundaries * k)
        m = self._m if self.is_identity else ScaledMapping(self._m, k)
        return self._copy(skeleton=sk, mapping=m)

    # -- surgery ------------------------------------------------------------

    def refined(self, radii, *, names=None) -> "Geometry":
        """Interior boundaries inserted; the mapping is kept.

        A split layer loses its name; `names` name the new interfaces.
        """
        radii = [float(r) for r in np.atleast_1d(np.asarray(radii, dtype=float))]
        sk = self._sk.refined(radii)
        new_face_names = _names(names, len(radii), "interface")
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
        face_names = []
        first = 0 if sk.is_hollow else 1
        for j in range(first, sk.boundaries.size):
            r = float(sk.boundaries[j])
            face_names.append(by_radius.get(r, old_faces.get(r)))
        return self._copy(skeleton=sk, layer_names=layer_names,
                          interface_names=face_names)

    def truncated(self, radius, *, name=None) -> "Geometry":
        """The geometry cut at `radius`; the mapping is kept.

        A cut on an existing boundary keeps that interface's name unless
        `name` is given; the shortened layer keeps its name.
        """
        sk = self._sk.truncated(radius)
        old_faces = {f.radius: f.name for f in self.interfaces}
        layer_names = list(self._layer_names[:sk.nlayers])
        face_names = []
        first = 0 if sk.is_hollow else 1
        for j in range(first, sk.boundaries.size):
            face_names.append(old_faces.get(float(sk.boundaries[j])))
        if name is not None or face_names[-1] is None:
            face_names[-1] = name
        return self._copy(skeleton=sk, layer_names=layer_names,
                          interface_names=face_names)

    def hollowed(self, radius, *, name=None) -> "Geometry":
        """The geometry cut at `radius` from below; the mapping is kept.

        The result is hollow, with the cut as its inner interface.  A cut
        on an existing boundary keeps that interface's name unless `name`
        is given; the shortened layer keeps its name.
        """
        sk = self._sk.hollowed(radius)
        old_faces = {f.radius: f.name for f in self.interfaces}
        dropped = self._sk.nlayers - sk.nlayers
        layer_names = list(self._layer_names[dropped:])
        face_names = [old_faces.get(float(b)) for b in sk.boundaries]
        if name is not None or face_names[0] is None:
            face_names[0] = name
        return self._copy(skeleton=sk, layer_names=layer_names,
                          interface_names=face_names)

    def extended(self, radii, *, names=None, interface_names=None) -> "Geometry":
        """Layers appended beyond the outer boundary; identity mapping only.

        `names` name the new layers, `interface_names` their outer
        boundaries.  A geometry whose mapping is not the identity is
        refused: extend the skeleton first and build the mapping after.
        """
        self._require_identity("extended")
        radii = [float(r) for r in np.atleast_1d(np.asarray(radii, dtype=float))]
        sk = self._sk.extended(radii)
        n = len(radii)
        return self._copy(
            skeleton=sk,
            layer_names=self._layer_names + _names(names, n, "layer"),
            interface_names=self._face_names + _names(interface_names, n, "interface"))

    def coarsened(self, *, keep=None, drop=None) -> tuple["Geometry", CoarseningMap]:
        """Interior boundaries removed; identity mapping only.

        A merged layer's name is None; kept interfaces keep theirs.
        """
        self._require_identity("coarsened")
        sk, cmap = self._sk.coarsen(keep=keep, drop=drop)
        layer_names = [self._layer_names[fine[0]] if len(fine) == 1 else None
                       for fine in cmap.layers]
        old_faces = {f.radius: f.name for f in self.interfaces}
        first = 0 if sk.is_hollow else 1
        face_names = [old_faces.get(float(sk.boundaries[j]))
                      for j in range(first, sk.boundaries.size)]
        return (self._copy(skeleton=sk, layer_names=layer_names,
                           interface_names=face_names), cmap)

    def _require_identity(self, verb: str) -> None:
        if not self.is_identity:
            raise ValueError(
                f"a geometry can only be {verb} while its mapping is the "
                "identity: do the surgery on the skeleton first and build "
                "the mapping for the result")

    def __repr__(self) -> str:
        m = "identity" if self.is_identity else repr(self._m)
        return f"Geometry({self._sk!r}, mapping={m})"


def _updated(current, new):
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


def _generic_validity(m, lattice):
    """J > 0 on the lattice for a mapping without `is_valid`."""
    from .mapping import MappingBase
    class _Wrapped(MappingBase):
        def __call__(self, X):
            return m(X)

        def deformation_gradient(self, X):
            return m.deformation_gradient(X)

        def jacobian(self, X):
            return m.jacobian(X)
    return _Wrapped().is_valid(sample=lattice)
