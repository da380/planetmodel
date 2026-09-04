"""skeleton.py -- the reference geometry: ordered shell boundaries.

Skeleton is pure geometry.  It holds the boundary radii
b0 < b1 < ... < bL defining L shells and answers questions about
intervals and membership; it carries no fields, no names, no physics.
Those live on the ReferenceBody, which keeps Skeleton.__eq__ usable as
the compatibility test between anything defined over the same geometry.

locate(r) *flags* boundary ambiguity through Location rather than
resolving it: at an interior discontinuity both adjacent layers are
candidates and the choice of side belongs to the caller.
"""
from __future__ import annotations

from dataclasses import KW_ONLY, dataclass

import numpy as np

__all__ = ["Location", "Skeleton", "CoarseningMap"]


@dataclass(frozen=True)
class Location:
    """Result of Skeleton.locate: candidate layer indices plus a boundary flag.

    Interior point            -> layers=(i,),    boundary=None
    On interior boundary j    -> layers=(j-1,j), boundary=j
    At centre / outer surface -> layers=(i,),    boundary=0 or L
    """
    layers: tuple[int, ...]
    _: KW_ONLY
    boundary: int | None = None

    @property
    def is_boundary(self) -> bool:
        """Whether the radius coincides with a skeleton boundary."""
        return self.boundary is not None

    @property
    def layer(self) -> int:
        """The unique layer index, raising when genuinely ambiguous."""
        if len(self.layers) == 1:
            return self.layers[0]
        raise ValueError(
            f"radius lies on interior boundary {self.boundary}; candidate "
            f"layers {self.layers} -- choose a side explicitly")


class Skeleton:
    """Ordered shell boundaries; geometry only, no fields, no physics."""

    def __init__(self, boundaries) -> None:
        """Validate and freeze a strictly increasing 1-d boundary array."""
        b = np.array(boundaries, dtype=float)
        if b.ndim != 1 or b.size < 2:
            raise ValueError("need a 1-d array of at least two boundary radii")
        if not np.all(np.diff(b) > 0.0):
            raise ValueError("boundary radii must be strictly increasing")
        b.setflags(write=False)
        self._b = b

    @property
    def boundaries(self) -> np.ndarray:
        """The read-only boundary radii b0 < b1 < ... < bL."""
        return self._b

    @property
    def nlayers(self) -> int:
        """The number of layers, L."""
        return self._b.size - 1

    @property
    def inner_boundaries(self) -> np.ndarray:
        """The interior discontinuity radii b1 ... b_{L-1}."""
        return self._b[1:-1]

    def layer_index(self, i: int) -> int:
        """A layer index normalised to 0 <= i < nlayers (negatives count back).

        IndexError outside that range.  Every method taking a layer
        index accepts what this accepts, so `-1` is the outermost layer
        everywhere.
        """
        n = self.nlayers
        i = int(i)
        if i < 0:
            i += n
        if not 0 <= i < n:
            raise IndexError(f"layer index out of range for {n} layers")
        return i


    def interval(self, i: int) -> tuple[float, float]:
        """The radial interval (b_i, b_{i+1}) of layer i (negatives allowed)."""
        i = self.layer_index(i)
        return float(self._b[i]), float(self._b[i + 1])

    @property
    def tolerance(self) -> float:
        """The round-off allowance on a radius: 1e-9 of the span.

        Every "is this interval that layer" test in the library uses it,
        so a field re-stated on an interval and the layer it was cut
        from agree to the same tolerance everywhere.
        """
        return 1e-9 * float(self._b[-1] - self._b[0])

    def spans(self, lo: float, hi: float, *, layer: int | None = None) -> bool:
        """Whether [lo, hi] is this skeleton's span, or one layer's, to round-off.

        With `layer` given, compares against that layer's interval;
        otherwise against the whole skeleton.
        """
        a, b = (self.interval(layer) if layer is not None
                else (float(self._b[0]), float(self._b[-1])))
        tol = self.tolerance
        return abs(float(lo) - a) <= tol and abs(float(hi) - b) <= tol

    def contains(self, lo: float, hi: float) -> bool:
        """Whether [lo, hi] lies inside this skeleton's span, to round-off."""
        tol = self.tolerance
        return (float(lo) >= float(self._b[0]) - tol
                and float(hi) <= float(self._b[-1]) + tol)

    def locate(self, r: float, *, atol: float = 0.0) -> Location:
        """Locate radius r, flagging boundary ambiguity via Location.

        atol > 0 snaps radii within that distance onto a boundary,
        which is useful for computed radii; the default demands exact
        coincidence.  Radii outside the skeleton raise ValueError.
        """
        r = float(r)
        b = self._b
        if r < b[0] - atol or r > b[-1] + atol:
            raise ValueError(f"radius {r} outside [{b[0]}, {b[-1]}]")
        j = int(np.argmin(np.abs(b - r)))
        if abs(b[j] - r) <= atol:
            if j == 0:
                return Location((0,), boundary=0)
            if j == b.size - 1:
                return Location((self.nlayers - 1,), boundary=j)
            return Location((j - 1, j), boundary=j)
        i = int(np.searchsorted(b, r, side="right")) - 1
        return Location((i,), boundary=None)

    def coarsen(self, *, keep=None, drop=None) -> tuple["Skeleton", "CoarseningMap"]:
        """A skeleton retaining a subset of the interior boundaries.

        Exactly one of `keep` or `drop` is given, indexing the *interior*
        boundaries as they appear in inner_boundaries (0-based).  The
        centre and the outer boundary are not removable: they bound the
        body rather than divide it.

        Returns the coarse skeleton and a CoarseningMap recording which
        fine layers each coarse layer merges -- which is what lets a
        consumer keep sampling the full model inside a merged layer, and
        what a mesh manifest records so the choice is never implicit.
        """
        n_inner = self._b.size - 2
        if (keep is None) == (drop is None):
            raise ValueError("give exactly one of keep and drop")
        if drop is not None:
            drop = [self._norm_inner(i, n_inner) for i in drop]
            keep = [i for i in range(n_inner) if i not in set(drop)]
        else:
            keep = sorted({self._norm_inner(i, n_inner) for i in keep})

        kept = np.array([0, *[i + 1 for i in keep], self._b.size - 1], dtype=int)
        coarse = Skeleton(self._b[kept])
        layers = tuple(tuple(range(kept[j], kept[j + 1]))
                       for j in range(kept.size - 1))
        return coarse, CoarseningMap(self, coarse, layers, tuple(int(k) for k in kept))

    def _norm_inner(self, i: int, n_inner: int) -> int:
        """Normalize an interior-boundary index, or say why it is invalid."""
        j = i + n_inner if i < 0 else i
        if not 0 <= j < n_inner:
            raise IndexError(
                f"interior boundary index {i} out of range; this skeleton has "
                f"{n_inner} interior boundaries")
        return j

    def refined(self, radii) -> "Skeleton":
        """A skeleton with extra interior boundaries inserted."""
        return self._with_radii(radii, "insert", lambda r: self._b[0] < r < self._b[-1])

    def extended(self, radii) -> "Skeleton":
        """A skeleton with extra shells appended beyond the outer boundary."""
        return self._with_radii(radii, "append", lambda r: r > self._b[-1])

    def truncated(self, radius: float) -> "Skeleton":
        """The skeleton cut at `radius`, which becomes the outer boundary.

        Boundaries at or above the cut are dropped.  A cut that
        coincides with an existing boundary simply removes what lies
        above it; otherwise the outermost surviving layer is shortened.
        """
        radius = float(radius)
        if radius <= self._b[0]:
            raise ValueError(
                f"cannot truncate at {radius}: it is at or below the centre "
                f"{self._b[0]}")
        if radius > self._b[-1]:
            raise ValueError(
                f"cannot truncate at {radius}: beyond the outer boundary "
                f"{self._b[-1]}; use extended() to grow the skeleton")
        kept = self._b[self._b < radius]
        return Skeleton(np.concatenate([kept, [radius]]))

    def _with_radii(self, radii, verb: str, ok) -> "Skeleton":
        """Merge extra radii into the boundary array, checking each one."""
        extra = np.atleast_1d(np.asarray(radii, dtype=float))
        for r in extra:
            if not ok(float(r)):
                raise ValueError(
                    f"cannot {verb} radius {float(r)} into a skeleton spanning "
                    f"[{self._b[0]}, {self._b[-1]}]")
            if np.any(self._b == r):
                raise ValueError(f"radius {float(r)} is already a boundary")
        merged = np.union1d(self._b, extra)
        if merged.size != self._b.size + extra.size:
            raise ValueError(f"duplicate radii among {list(map(float, extra))}")
        return Skeleton(merged)

    def __eq__(self, other) -> bool:
        """Skeletons are equal iff their boundary arrays are identical."""
        return isinstance(other, Skeleton) and np.array_equal(self._b, other._b)

    def __repr__(self) -> str:
        """Compact summary: layer count and radial range."""
        return (f"Skeleton({self.nlayers} layers, "
                f"r in [{self._b[0]:g}, {self._b[-1]:g}])")


@dataclass(frozen=True)
class CoarseningMap:
    """What a coarsening did: which fine layers each coarse layer merges.

    Recorded rather than inferred, because the alternative -- dropping
    boundaries by position and hoping the consumer guesses the same
    thing -- is exactly the implicit convention this replaces.
    """

    fine: Skeleton
    coarse: Skeleton
    layers: tuple[tuple[int, ...], ...]   # coarse layer -> fine layers merged
    boundaries: tuple[int, ...]           # coarse boundary -> fine index

    @property
    def kept_interfaces(self) -> tuple[int, ...]:
        """Interior boundaries retained, indexed on the fine skeleton."""
        return tuple(i - 1 for i in self.boundaries[1:-1])

    @property
    def dropped_interfaces(self) -> tuple[int, ...]:
        """Interior boundaries removed, indexed on the fine skeleton."""
        kept = set(self.kept_interfaces)
        return tuple(i for i in range(self.fine.boundaries.size - 2)
                     if i not in kept)

    def fine_layer(self, r: float) -> int:
        """The fine layer containing r -- the one that still has the fields."""
        return self.fine.locate(float(r)).layers[-1]

    def __repr__(self) -> str:
        return (f"CoarseningMap({self.fine.nlayers} -> {self.coarse.nlayers} "
                f"layers, dropped {list(self.dropped_interfaces)})")
