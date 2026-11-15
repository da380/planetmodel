"""A flat-layered body: a skeleton of depths over a lateral domain, placed
by one continuous mapping.

`Geometry` is the flat leaf of `geometry.GeometryBase`: everything the
spherical geometry does (names, the invariant checks, knots, validity,
copies, scaling and surgery) it does in `FLAT`, (depth, x, y), over its
lateral `domain`, a `Box` or a `Rectangle`.  The skeleton's boundaries
are depths, from the free surface (depth 0) down; the body is the
domain's extent times the skeleton's span.

The interfaces are the free surface, or the top of a hollowed body, and
every boundary between two layers, numbered from the surface down:
interface k is skeleton boundary k, the top of layer k, with
`between = (k, k - 1)`, the layer below it first and -1 for the outside
above the surface.  The base of the box, the skeleton's last boundary,
is a face of the box and no interface, as the centre of a ball is none;
nothing carries boundary physics there.  So `extended` makes the old
base an interface, the top of the first new layer, and `truncated`
makes a new base, which takes no name.

The mapping is checked as on the sphere, on the lattice the domain lays
over the body (`lattice`): orientation preserved, continuity across
every interior boundary.  Over a `Rectangle` the body is invariant
along y, and the mapping is held to it: m(X + s e_y) = m(X) + s e_y
across the lattice, s the rectangle's length.  Any mapping is accepted;
the identity is the default, and `stretched` places the geometry under
the vertical stretch of a displacement (`cartesian.VerticalStretch`),
topography as a stack of reliefs (`cartesian.layer_linear`), heights
positive up (`cartesian.displacement` states the sign).  A field of
depth alone built for its
layers is a `DepthField` (`radial_field_type`).  Nothing here knows
about units: depths are numbers and every tolerance is `rtol` times the
skeleton's span.
"""
from __future__ import annotations

from typing import ClassVar

import numpy as np

from ..fields import RadialField
from ..frames import SphericalFunction
from ..geometry import GeometryBase, Names
from ..mapping import Mapping
from ..skeleton import Skeleton
from .coordinates import FLAT
from .displacement import VerticalDisplacement
from .domains import Domain
from .fields import DepthField
from .mapping import VerticalStretch

__all__ = ["Geometry"]


class Geometry(GeometryBase):
    """A skeleton of depths over a lateral `domain`, a mapping and names;
    see the module docstring.

    `check=False` skips the invariant checks, for a mapping already known
    to satisfy them.
    """

    _coordinates = FLAT
    radial_field_type: ClassVar[type[RadialField]] = DepthField

    def __init__(self, skeleton: Skeleton, domain: Domain, *,
                 mapping: Mapping | None = None, layer_names: Names = None,
                 interface_names: Names = None, rtol: float = 1e-9,
                 check: bool = True) -> None:
        if not isinstance(domain, Domain.__value__):
            raise TypeError(
                f"expected a Box or a Rectangle as the lateral domain, got "
                f"{type(domain).__name__}")
        self._domain = domain
        super().__init__(skeleton, mapping=mapping, layer_names=layer_names,
                         interface_names=interface_names, rtol=rtol, check=check)

    @property
    def domain(self) -> Domain:
        """The lateral domain the body extends over."""
        return self._domain

    def _face_boundaries(self, skeleton: Skeleton) -> range:
        """Every boundary but the base: the top, then those between layers."""
        return range(0, skeleton.nlayers)

    def _between(self, j: int, nlayers: int) -> tuple[int, int]:
        """(j, j - 1), the outside -1: the layer below is the deeper one."""
        return (j, j - 1)

    def _lattice(self, **sizes: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self._domain.lattice(self._sk, **sizes)

    def _made(self, skeleton: Skeleton, *, k: float, mapping: Mapping,
              layer_names: Names, interface_names: Names, rtol: float,
              check: bool) -> "Geometry":
        domain = self._domain if k == 1.0 else self._domain.scaled(k)
        return Geometry(skeleton, domain, mapping=mapping, layer_names=layer_names,
                        interface_names=interface_names, rtol=rtol, check=check)

    def _check_mapping(self) -> None:
        """The invariants of every geometry, then, over a domain that leaves
        y unbounded, the mapping's invariance along y."""
        super()._check_mapping()
        if "y" not in self._domain.axes:
            self._check_invariance()

    def _check_invariance(self) -> None:
        """m(X + s e_y) = m(X) + s e_y on the lattice, to `rtol` of the body's
        size, s the domain's length."""
        depth, x, y = self.lattice()
        X = FLAT.points(depth, x, y)
        s = float(self._domain.length)
        shift = np.array([0.0, s, 0.0])
        gap = np.linalg.norm(np.asarray(self._m(X + shift), dtype=float)
                             - np.asarray(self._m(X), dtype=float) - shift, axis=-1)
        size = max(self._sk.span, s)
        worst = float(np.max(gap))
        if worst > 100.0 * self._rtol * size:
            i = np.unravel_index(int(np.argmax(gap)), gap.shape)
            d, a, b = (float(np.broadcast_to(v, gap.shape)[i]) for v in (depth, x, y))
            raise ValueError(
                f"the mapping of a section must not vary along y: shifting by "
                f"{s:g} along y moves the image by {worst:.3g} more at "
                f"({', '.join(FLAT.labels)}) = ({d:g}, {a:g}, {b:g})")

    def stretched(self, w: VerticalDisplacement | SphericalFunction, *,
                  name: str | None = None, check: bool = True) -> "Geometry":
        """A copy whose mapping is the vertical stretch driven by `w`, the
        upward displacement (heights, positive up).

        `w` is any vertical displacement, a `cartesian.layer_linear` stack
        of reliefs for topography; the stretch's `base` is this
        geometry's base depth.  The copy is checked unless told
        otherwise: a stretch folding the body, or over a `Rectangle`
        varying along y, is refused by name.
        """
        return self.with_mapping(VerticalStretch(w, base=self._sk, name=name),
                                 check=check)

    def __repr__(self) -> str:
        m = "identity" if self.is_identity else repr(self._m)
        return (f"cartesian.Geometry({self._sk.describe(FLAT.labels[0])}, "
                f"{self._domain!r}, mapping={m})")
