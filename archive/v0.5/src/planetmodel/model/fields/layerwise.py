"""layerwise.py -- fields assembled from single-layer pieces.

A field belongs to one layer.  What a body stores is single-layer
fields, and what a caller asking for `body["rho"]` receives is a
*view*: one field on the body's skeleton, assembled from the pieces its
layers hold.  This module is the assembly.

`assemble(skeleton, pieces)` returns the best field it can:

- the piece itself, when there is one and it spans the skeleton;
- the pieces' common source, when every piece is a RestrictedField of
  one field on that skeleton and together they cover its domain -- so a
  pulled-back or pushed-forward field split into a body comes back as
  the object it was;
- the pieces' own type, when they share one that knows how to
  reassemble itself (`Assemblable.assembled`: a RadialField from its
  functions, an ElasticField from assembled moduli, a composite from
  assembled operands, an AnalyticField from its one formula);
- otherwise the generic view, which dispatches each radius to the
  piece that owns it: a `LayerwiseField` for static pieces, a
  `LayerwiseDependentField` (dependent.py) for frequency- or
  time-dependent ones.  Pieces of different kinds are refused.

A view has a *domain*: the layers with a piece.  A radius in any other
layer is refused by name.  Nothing is ever filled in.

`LayerwiseField` also serves the coarsened body: its pieces need not
align with the layers of the skeleton it presents, only tile part of
it, so one coarse layer can hold several fine pieces without anything
downstream knowing.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from ..skeleton import Skeleton
from .composite import FieldBase, RestrictedField

__all__ = ["LayerwiseField", "assemble", "split"]


def _span(field) -> tuple[float, float]:
    b = field.skeleton.boundaries
    return float(b[0]), float(b[-1])


def split(field) -> list:
    """The single-layer pieces of a field, one per layer of its domain."""
    if field.skeleton.nlayers == 1:
        return [field]
    return [field.restricted(i) for i in field.domain]


def assemble(skeleton: Skeleton, pieces: Sequence, *, name: str | None = None):
    """One field on `skeleton` from single-layer pieces inside it.

    See the module docstring for what comes back.  Every piece must be
    a single-layer field lying inside the skeleton's span, and no two
    may overlap.
    """
    pieces = list(pieces)
    if not pieces:
        raise ValueError("nothing to assemble: no pieces")
    lo, hi = float(skeleton.boundaries[0]), float(skeleton.boundaries[-1])
    tol = skeleton.tolerance
    for p in pieces:
        if p.skeleton.nlayers != 1:
            raise ValueError(f"{p!r} is not a single-layer field")
        a, b = _span(p)
        if not skeleton.contains(a, b):
            raise ValueError(
                f"{p!r} on [{a:.6g}, {b:.6g}] lies outside the skeleton "
                f"[{lo:.6g}, {hi:.6g}]")
    pieces.sort(key=lambda p: _span(p)[0])
    for u, v in zip(pieces[:-1], pieces[1:]):
        if _span(v)[0] < _span(u)[1] - tol:
            raise ValueError(f"{u!r} and {v!r} overlap")

    if len(pieces) == 1 and skeleton.nlayers == 1:
        p = pieces[0]
        if skeleton.spans(*_span(p)):
            return p if name is None or p.name == name else _renamed(p, name)

    # Restrictions of one source that together cover its domain: the
    # source is the view, and nothing is wrapped around it.
    if all(isinstance(p, RestrictedField) for p in pieces):
        src = pieces[0].source
        if (all(p.source is src for p in pieces) and src.skeleton == skeleton
                and sorted(p.layer for p in pieces) == sorted(src.domain)
                and all(skeleton.spans(*_span(p), layer=p.layer)
                        for p in pieces)):
            return src

    kinds = {type(p) for p in pieces}
    if len(kinds) == 1:
        cls = next(iter(kinds))
        hook = getattr(cls, "assembled", None) or getattr(cls, "_assembled", None)
        if hook is not None:
            out = hook(skeleton, pieces, name=name)
            if out is not NotImplemented:
                return out

    field_kinds = {getattr(p, "kind", "static") for p in pieces}
    if field_kinds == {"static"}:
        return LayerwiseField(skeleton, pieces, name=name)
    if len(field_kinds) != 1:
        raise ValueError(
            f"the view {name!r} mixes kinds {sorted(field_kinds)}: a view is "
            "one kind of field, so lift the static pieces first")
    from .dependent import LayerwiseDependentField
    return LayerwiseDependentField(skeleton, pieces, name=name)


def _renamed(field, name: str):
    """A shallow copy of a field under another name, where possible."""
    import copy
    try:
        out = copy.copy(field)
        out.name = name
        return out
    except Exception:        # a field that will not be copied keeps its name
        return field


class LayerwiseField(FieldBase):
    """The generic view: single-layer pieces tiling part of a skeleton.

    Each radius goes to the piece whose span contains it; on a shared
    edge `side` decides, as it does everywhere else.  The `domain` is
    the set of layers of `skeleton` that the pieces cover exactly, and
    a radius in a layer they do not cover is refused by name.

    `field[i]` is the piece on layer i (or, for a coarse layer tiled by
    several fine pieces, a LayerwiseField of those on that one layer),
    so the `field[i](r)` idiom works on a view as on a RadialField.
    """

    def __init__(self, skeleton: Skeleton, pieces: Sequence,
                 *, name: str | None = None) -> None:
        pieces = sorted(pieces, key=lambda p: _span(p)[0])
        if not pieces:
            raise ValueError("a LayerwiseField needs at least one piece")
        char = pieces[0].character
        for p in pieces[1:]:
            if p.character != char:
                raise ValueError(
                    f"pieces of different character: {char} and {p.character}")
        dims = {getattr(p, "dimensions", None) for p in pieces}
        dims.discard(None)
        if len(dims) > 1:
            raise ValueError(
                f"pieces of different dimensions: {sorted(map(str, dims))}")
        self.skeleton = skeleton
        self._pieces = tuple(pieces)
        self._edges = np.array([_span(p)[0] for p in pieces]
                               + [_span(pieces[-1])[1]], dtype=float)
        self._hi = np.array([_span(p)[1] for p in pieces], dtype=float)
        self.character = char
        self.dimensions = dims.pop() if dims else None
        self.name = name if name is not None else pieces[0].name

    @property
    def pieces(self) -> tuple:
        """The single-layer pieces, centre outward."""
        return self._pieces

    @property
    def is_radial(self) -> bool:
        return all(bool(getattr(p, "is_radial", False)) for p in self._pieces)

    @property
    def domain(self) -> tuple[int, ...]:
        """The layers of the skeleton that the pieces cover exactly."""
        out = []
        for i in range(self.skeleton.nlayers):
            if self._pieces_on(i) is not None:
                out.append(i)
        return tuple(out)

    def _pieces_on(self, i: int):
        """The pieces tiling layer i exactly, or None."""
        lo, hi = self.skeleton.interval(i)
        tol = self.skeleton.tolerance
        inside = [p for p in self._pieces
                  if _span(p)[0] >= lo - tol and _span(p)[1] <= hi + tol]
        if not inside:
            return None
        edge = lo
        for p in inside:
            a, b = _span(p)
            if abs(a - edge) > tol:
                return None
            edge = b
        return inside if abs(edge - hi) <= tol else None

    def __len__(self) -> int:
        return self.skeleton.nlayers

    def __getitem__(self, i: int):
        """The field on layer i (negatives allowed)."""
        return self.restricted(i)

    def __iter__(self):
        return (self[i] for i in self.domain)

    def restricted(self, layer):
        """The piece on one layer, or the pieces tiling it as one field."""
        i = self.skeleton.layer_index(layer)
        inside = self._pieces_on(i)
        if inside is None:
            raise ValueError(
                f"{self!r} is not defined on layer {i}: its domain is "
                f"{self.domain}")
        if len(inside) == 1:
            return inside[0]
        return assemble(Skeleton(self.skeleton.interval(i)), inside,
                        name=self.name)

    def on_interval(self, lo: float, hi: float):
        """Clip to [lo, hi] inside a single layer; never extrapolate."""
        if self.skeleton.nlayers != 1:
            raise ValueError(
                f"{self!r} spans {self.skeleton.nlayers} layers; restrict "
                "to one first")
        if not self.skeleton.contains(lo, hi):
            raise TypeError(
                f"{self!r} cannot be extrapolated to [{lo:.6g}, {hi:.6g}]")
        tol = self.skeleton.tolerance
        kept = []
        for p in self._pieces:
            pa, pb = _span(p)
            qa, qb = max(pa, lo), min(pb, hi)
            if qb - qa > tol:
                kept.append(p if (abs(qa - pa) <= tol and abs(qb - pb) <= tol)
                            else p.on_interval(qa, qb))
        return assemble(Skeleton([lo, hi]), kept, name=self.name)

    def rescaled(self, convert, old, new):
        """The same view of the converted pieces, on the scaled skeleton."""
        k = old.length / new.length
        return type(self)(Skeleton(self.skeleton.boundaries * k),
                          [convert(p) for p in self._pieces], name=self.name)

    def _index(self, flat: np.ndarray, *, side: str) -> np.ndarray:
        """The piece index of each radius, refusing gaps by name."""
        idx = np.searchsorted(self._edges, flat,
                              side="right" if side == "upper" else "left") - 1
        np.clip(idx, 0, len(self._pieces) - 1, out=idx)
        tol = self.skeleton.tolerance
        bad = (flat < self._edges[idx] - tol) | (flat > self._hi[idx] + tol)
        if np.any(bad):
            r = float(flat[bad][0])
            layer = self.skeleton.locate(r).layers[-1]
            raise ValueError(
                f"{self!r} is not defined at radius {r:.6g} (layer {layer}): "
                f"its domain is {self.domain}")
        return idx

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """Each radius answered by the piece that owns it."""
        return self._dispatch(r, theta, phi, layer=layer, side=side,
                              frame=frame)

    _NO_ARG = object()

    def _dispatch(self, r, theta=None, phi=None, *, layer=None,
                  side: str = "upper", frame: str = "spherical", arg=_NO_ARG):
        """Each radius answered by the piece that owns it.

        With `arg` given the pieces are dependent fields and are asked
        through `evaluate_with` at that argument.
        """
        if side not in ("upper", "lower"):
            raise ValueError("side must be 'upper' or 'lower'")
        b = self.skeleton.boundaries
        r = np.asarray(r, dtype=float)
        angles = theta is not None or phi is not None
        if angles:
            if theta is None or phi is None:
                raise ValueError("give both theta and phi, or neither")
            r, theta, phi = np.broadcast_arrays(
                r, np.asarray(theta, dtype=float), np.asarray(phi, dtype=float))
        shape = r.shape
        flat = np.atleast_1d(r).ravel()
        if flat.size and (flat.min() < b[0] or flat.max() > b[-1]):
            bad = flat[(flat < b[0]) | (flat > b[-1])][0]
            raise ValueError(f"radius {bad} outside [{b[0]}, {b[-1]}]")
        if layer is not None:
            i = self.skeleton.layer_index(layer)
            lo, hi = self.skeleton.interval(i)
            tol = self.skeleton.tolerance
            if flat.size and (flat.min() < lo - tol or flat.max() > hi + tol):
                raise ValueError(
                    f"radius outside layer {i} [{lo:.6g}, {hi:.6g}]")
            if self._pieces_on(i) is None:
                raise ValueError(
                    f"{self!r} is not defined on layer {i}: its domain is "
                    f"{self.domain}")
            # Inside a layer the tie-break at its ends is the layer's own
            # piece either way; between fine pieces `side` still applies.
            idx = self._index(np.clip(flat, lo, hi), side=side)
        else:
            idx = self._index(flat, side=side)
        th = np.atleast_1d(theta).ravel() if angles else None
        ph = np.atleast_1d(phi).ravel() if angles else None
        out = None
        for k in np.unique(idx):
            m = idx == k
            p = self._pieces[int(k)]
            if arg is not self._NO_ARG:
                v = np.asarray(p.evaluate_with(
                    flat[m], None if th is None else th[m],
                    None if ph is None else ph[m], arg,
                    layer=None, side=side, frame=frame))
            else:
                v = np.asarray(p.evaluate(
                    flat[m], None if th is None else th[m],
                    None if ph is None else ph[m], frame=frame))
            if out is None:
                out = np.empty(flat.shape + v.shape[1:], dtype=v.dtype)
            out[m] = v
        if out is None:
            return np.empty(shape + self.character.component_shape)
        return out.reshape(shape + out.shape[1:]) if shape else out[0]

    def derivative(self, nu: int = 1):
        """Piece by piece, where every piece differentiates."""
        return assemble(self.skeleton, [p.derivative(nu) for p in self._pieces],
                        name=None if self.name is None else self.name + "'" * nu)

    def integrate(self, a: float, b: float) -> float:
        """Piece by piece over the overlap; a gap inside [a, b] is refused."""
        a, b = float(a), float(b)
        if a == b:
            return 0.0
        sign = 1.0
        if a > b:
            a, b, sign = b, a, -1.0
        tol = self.skeleton.tolerance
        total, edge = 0.0, a
        for p in self._pieces:
            pa, pb = _span(p)
            lo, hi = max(a, pa), min(b, pb)
            if hi > lo:
                if lo > edge + tol:
                    raise ValueError(
                        f"[{a}, {b}] crosses a gap in {self!r} at "
                        f"[{edge:.6g}, {lo:.6g}]")
                total += float(p.integrate(lo, hi))
                edge = hi
        if edge < b - tol:
            raise ValueError(
                f"[{a}, {b}] crosses a gap in {self!r} at [{edge:.6g}, {b:.6g}]")
        return sign * total

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        n = self.skeleton.nlayers
        d = len(self.domain)
        g = f", {n - d} without" if d != n else ""
        return f"{type(self).__name__}({len(self._pieces)} pieces on {n} layers{g}{nm})"
