"""Tabulated radial models: decks, their formats, and fields by interpolation.

A deck is a table of knots, radius first, one column per named field,
in which a repeated radius marks a boundary: the layering comes from
that structure, so a boundary is recorded even where the tabulated
values happen to agree across it.  What the columns are called and what
the header lines mean is a `DeckFormat`: the column names (by row
width, where a format admits more than one), the number of header
lines, and how to read and write them.  `MINEOS` is the format of the
mineos and PREM decks, three header lines and nine or six columns; any
other table is a format of its own, given its column names.

`read_deck` turns a file into a `Deck`, numbers and a header, nothing
else.  `deck_layers` turns a deck into the base fields of a model: the
knots of every layer are interpolated by a piecewise polynomial (a
cubic spline by default), which is a `PolynomialLayer`, so the field
algebra downstream is exact on it and the moduli of a deck model are
products of piecewise polynomials, never refits.  A column that is NaN
throughout a layer is absent from that layer, the way a fluid layer
holds no shear quality factor; NaN on part of a layer is refused.  From
those base fields the rest of a model is built the normal way, by the
mixins of `planetmodel.behaviours`.

`Tabulated` is the mixin of a model type built from a deck: it keeps
the knots per layer and the header as `knots` and `header`, and gives
the model back as a deck by `to_deck`, sampled on those knots, so that
a model that was read can be written.
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, PPoly

from .character import SCALAR, Character
from .fields import Field, RadialField
from .layerfunction import PolynomialLayer
from .skeleton import Skeleton
from .vocabulary import VOCABULARY, FieldSpec

if TYPE_CHECKING:
    from .model import Model

__all__ = ["Deck", "DeckFormat", "MINEOS", "MINEOS_COLUMNS", "MINEOS_ISOTROPIC_COLUMNS",
           "read_deck", "write_deck", "deck_layers", "deck_knots", "mineos_names",
           "Tabulated", "KINDS"]

#: The interpolants a deck's knots may be joined by, each a piecewise
#: polynomial: a not-a-knot cubic spline, a shape-preserving cubic
#: (pchip), Akima's cubic, or straight lines.  Two knots give a line
#: under every kind.
KINDS = ("cubic", "pchip", "akima", "linear")

#: The columns of a transversely isotropic mineos deck, after the radius.
MINEOS_COLUMNS = ("rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta")

#: The columns of an isotropic mineos deck, after the radius.
MINEOS_ISOTROPIC_COLUMNS = ("rho", "vp", "vs", "qkappa", "qmu")


# -- the deck and its format ---------------------------------------------------

@dataclass(frozen=True)
class DeckFormat:
    """What a family of deck files looks like.

    `columns` names the columns after the radius, either one tuple or a
    mapping from row width to tuple where the format admits several;
    `header_lines` is how many lines precede the table, `parse_header`
    turns them into the header mapping a `Deck` carries and
    `write_header` turns a deck back into them.  Without a parser the
    header lines are kept verbatim under "lines".
    """

    columns: tuple[str, ...] | Mapping[int, tuple[str, ...]] = field(hash=False)
    _: KW_ONLY
    name: str = "custom"
    header_lines: int = 0
    parse_header: Callable[[Sequence[str]], Mapping[str, Any]] | None = None
    write_header: Callable[["Deck"], Sequence[str]] | None = None

    def names(self, width: int) -> tuple[str, ...]:
        """The column names of a row with `width` values after the radius."""
        if isinstance(self.columns, Mapping):
            try:
                return tuple(self.columns[width])
            except KeyError:
                raise ValueError(
                    f"a {self.name} deck has {sorted(self.columns)} columns after "
                    f"the radius, not {width}") from None
        if len(self.columns) != width:
            raise ValueError(
                f"a {self.name} deck has {len(self.columns)} columns after the "
                f"radius, {list(self.columns)}; the table has {width}")
        return tuple(self.columns)

    def header(self, lines: Sequence[str]) -> Mapping[str, Any]:
        """The header mapping of the given header lines."""
        if self.parse_header is None:
            return {"lines": tuple(lines)}
        return self.parse_header(lines)


@dataclass(frozen=True)
class Deck:
    """A table of knots: radii, one array per named column, and a header.

    `radius` is non-decreasing with every repeat a boundary; `columns`
    maps each name to its values at the knots, NaN where a value is
    absent; `header` is what the format read from the header lines.
    Arrays are read-only.
    """

    radius: np.ndarray
    columns: Mapping[str, np.ndarray]
    _: KW_ONLY
    header: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        r = np.array(self.radius, dtype=float)
        if r.ndim != 1 or r.size < 2:
            raise ValueError("a deck needs a 1-d array of at least two knot radii")
        if np.any(np.diff(r) < 0.0):
            raise ValueError("knot radii must be non-decreasing")
        r.setflags(write=False)
        cols = {}
        for name, values in self.columns.items():
            v = np.array(values, dtype=float)
            if v.shape != r.shape:
                raise ValueError(f"column {name!r} has shape {v.shape}, the radii "
                                 f"{r.shape}")
            v.setflags(write=False)
            cols[str(name)] = v
        object.__setattr__(self, "radius", r)
        object.__setattr__(self, "columns", MappingProxyType(cols))
        object.__setattr__(self, "header", MappingProxyType(dict(self.header)))
        self.layers()                      # the structure is checked on construction

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self.columns)

    @property
    def nknots(self) -> int:
        return self.radius.size

    def __getitem__(self, name: str) -> np.ndarray:
        try:
            return self.columns[name]
        except KeyError:
            raise KeyError(f"the deck has no column {name!r}; it has "
                           f"{list(self.columns)}") from None

    def __contains__(self, name: object) -> bool:
        return name in self.columns

    def layers(self) -> list[slice]:
        """The rows of every layer, centre outward: one slice per span
        between repeated radii, each of at least two strictly increasing
        knots."""
        r = self.radius
        cuts = np.flatnonzero(np.diff(r) == 0.0)
        if cuts.size and np.any(np.diff(cuts) == 1):
            raise ValueError("a radius repeats more than twice: a layer of no "
                             "thickness")
        starts = np.concatenate(([0], cuts + 1))
        stops = np.concatenate((cuts + 1, [r.size]))
        slices = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
        for s in slices:
            if s.stop - s.start < 2:
                raise ValueError(f"the layer starting at r = {r[s.start]:g} has one "
                                 "knot; every layer needs at least two")
        return slices

    @property
    def boundaries(self) -> np.ndarray:
        """The skeleton's boundary radii: the ends and every repeated radius."""
        r = self.radius
        cuts = np.flatnonzero(np.diff(r) == 0.0)
        return np.concatenate(([r[0]], r[cuts], [r[-1]]))

    @property
    def nlayers(self) -> int:
        return len(self.layers())

    def __repr__(self) -> str:
        return (f"Deck({self.nknots} knots, {self.nlayers} layers, columns "
                f"{list(self.columns)})")


def read_deck(source: str | os.PathLike[str] | Iterable[str],
              format: DeckFormat | Sequence[str], *,
              header_lines: int | None = None) -> Deck:
    """A deck from a file, or from lines: the header lines by the format,
    then whitespace-separated rows, radius first.

    `format` is a `DeckFormat`, or the column names after the radius for
    a table whose header is `header_lines` verbatim lines (none by
    default).  Blank lines in the table are skipped.
    """
    if isinstance(source, (str, os.PathLike)):
        path: str | None = os.fspath(source)
        with open(source) as fh:
            lines = fh.read().splitlines()
    else:
        path = None
        lines = list(source)
    if not isinstance(format, DeckFormat):
        format = DeckFormat(tuple(format), header_lines=header_lines or 0)
    elif header_lines is not None:
        raise ValueError("header_lines is set by the DeckFormat")
    n = format.header_lines
    if len(lines) < n:
        raise ValueError(f"{path or 'the deck'}: {len(lines)} lines, fewer than the "
                         f"{n} header lines of a {format.name} deck")
    header = format.header(lines[:n])
    body = [ln for ln in lines[n:] if ln.strip()]
    if not body:
        raise ValueError(f"{path or 'the deck'}: no knots after the header")
    table = np.loadtxt(body, dtype=float, ndmin=2)
    names = format.names(table.shape[1] - 1)
    return Deck(table[:, 0], {nm: table[:, j + 1] for j, nm in enumerate(names)},
                header=header, source=path)


def write_deck(path: str | os.PathLike[str], deck: Deck, format: DeckFormat) -> Path:
    """Write a deck in a format: its header lines, then one row per knot,
    radius first and the format's columns in order.  A column the deck
    lacks is refused; an absent value is written as nan."""
    names = format.names(len(deck.columns)) if not isinstance(format.columns, Mapping) \
        else next((c for c in format.columns.values() if set(c) <= set(deck.columns)),
                  None)
    if names is None or any(n not in deck for n in names):
        raise ValueError(f"a {format.name} deck needs columns {format.columns}; the "
                         f"deck has {list(deck.columns)}")
    if format.write_header is not None:
        header = list(format.write_header(deck))
    else:
        header = list(deck.header.get("lines", ()))
    if len(header) != format.header_lines:
        raise ValueError(f"a {format.name} deck has {format.header_lines} header "
                         f"lines; {len(header)} were produced")
    rows = np.column_stack([deck.radius] + [deck[n] for n in names])
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for line in header:
            fh.write(line.rstrip("\n") + "\n")
        for row in rows:
            fh.write(f"{row[0]:14.3f}" + "".join(f"{v:16.8g}" for v in row[1:]) + "\n")
    return out


# -- the mineos format -------------------------------------------------------------

def _mineos_header(lines: Sequence[str]) -> dict[str, Any]:
    """The three header lines of a mineos deck: a title; `ifanis tref
    ifdeck`; `nknot nic noc`.  Trailing tokens are kept under "extras"."""
    title = lines[0].strip()
    h2, h3 = lines[1].split(), lines[2].split()
    if len(h2) < 3 or len(h3) < 3:
        raise ValueError("a mineos deck's second and third header lines carry "
                         "`ifanis tref ifdeck` and `nknot nic noc`")
    return {"name": title, "ifanis": int(h2[0]), "tref": float(h2[1]),
            "ifdeck": int(h2[2]), "nknot": int(h3[0]), "nic": int(h3[1]),
            "noc": int(h3[2]), "extras": (tuple(h2[3:]), tuple(h3[3:]))}


def _mineos_header_lines(deck: Deck) -> list[str]:
    """The three header lines for a deck: the title from its header, the
    flags from its header where present, the counts from its knots."""
    h = deck.header
    cuts = np.flatnonzero(np.diff(deck.radius) == 0.0)
    nic = int(cuts[0] + 1) if cuts.size else deck.nknots
    noc = int(cuts[1] + 1) if cuts.size > 1 else nic
    ifanis = int(h.get("ifanis", 1 if "vph" in deck else 0))
    tref = float(h.get("tref", 1.0))
    ifdeck = int(h.get("ifdeck", 1))
    return [str(h.get("name", deck.source or "deck")),
            f"{ifanis} {tref:g} {ifdeck}",
            f"{deck.nknots} {h.get('nic', nic)} {h.get('noc', noc)}"]


#: The mineos deck: three header lines, then `r rho vpv vsv qkappa qmu vph
#: vsh eta` in SI (nine columns) or `r rho vp vs qkappa qmu` (six).
MINEOS = DeckFormat({8: MINEOS_COLUMNS, 5: MINEOS_ISOTROPIC_COLUMNS}, name="mineos",
                    header_lines=3, parse_header=_mineos_header,
                    write_header=_mineos_header_lines)


def mineos_names(deck: Deck) -> tuple[list[str | None], list[str | None]]:
    """Layer and interface names a mineos header implies: `nic` and `noc`
    count the knots of the inner core and to the top of the outer core,
    so where each sits on a repeated radius the first two layers are
    `inner_core` and `outer_core` under `icb` and `cmb`; the outermost
    interface is `surface`.  Every other name is None.  A header whose
    counts do not sit on repeated radii is warned about and ignored."""
    layers: list[str | None] = [None] * deck.nlayers
    faces: list[str | None] = [None] * (deck.nlayers + (1 if deck.radius[0] > 0 else 0))
    faces[-1] = "surface"
    r, h = deck.radius, deck.header
    first = 1 if deck.radius[0] > 0 else 0
    cores = (("nic", "inner_core", "icb"), ("noc", "outer_core", "cmb"))
    for key, layer, face in cores:
        i = int(h.get(key, 0))
        if not (1 <= i < r.size and r[i - 1] == r[i]):
            if key in h:
                warnings.warn(f"the header's {key} = {i} does not sit on a repeated "
                              "radius; the layer is left unnamed", stacklevel=2)
            continue
        k = int(np.searchsorted(deck.boundaries, r[i]))      # the boundary index
        if 0 < k < deck.nlayers:
            layers[k - 1] = layer
            faces[k - 1 + first] = face
    if "nknot" in h and h["nknot"] != deck.nknots:
        warnings.warn(f"the header says {h['nknot']} knots, the table has "
                      f"{deck.nknots}; the table is taken as the truth", stacklevel=2)
    return layers, faces


# -- fields by interpolation ---------------------------------------------------

def _interpolant(r: np.ndarray, y: np.ndarray, kind: str) -> PolynomialLayer:
    """The piecewise polynomial through the knots (r, y) of one layer."""
    if kind not in KINDS:
        raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
    if not np.all(np.diff(r) > 0.0):
        raise ValueError("the knots of a layer must increase strictly")
    if r.size == 2 or kind == "linear":
        slopes = np.diff(y) / np.diff(r)
        ppoly = PPoly(np.vstack([slopes, y[:-1]]), r)
    elif kind == "cubic":
        ppoly = CubicSpline(r, y)
    elif kind == "pchip":
        ppoly = PchipInterpolator(r, y)
    else:
        ppoly = Akima1DInterpolator(r, y)
    return PolynomialLayer(ppoly, interval=(float(r[0]), float(r[-1])))


def deck_layers(deck: Deck, *, kind: str = "cubic",
                specs: Mapping[str, FieldSpec] | None = None
                ) -> tuple[Skeleton, list[dict[str, Field]]]:
    """The skeleton of a deck and the base fields of every layer.

    Every column becomes one `RadialField` per layer on the piecewise
    polynomial of `kind` through that layer's knots, with the character
    the vocabulary or `specs` give the name (SCALAR for a name neither
    knows).  A column that is NaN throughout a layer is absent from that
    layer; NaN on part of one is refused.
    """
    known = {**VOCABULARY, **dict(specs or {})}
    slices = deck.layers()
    skeleton = Skeleton(deck.boundaries)
    layers: list[dict[str, Field]] = []
    for i, s in enumerate(slices):
        r = deck.radius[s]
        fields: dict[str, Field] = {}
        for name, values in deck.columns.items():
            y = values[s]
            bad = np.isnan(y)
            if bad.all():
                continue
            if bad.any():
                raise ValueError(f"column {name!r} is NaN on part of layer {i}; a "
                                 "column is absent from a layer only when NaN "
                                 "throughout it")
            spec = known.get(name)
            character: Character = SCALAR if spec is None else spec.character
            fields[name] = RadialField(skeleton.interval(i), _interpolant(r, y, kind),
                                       character=character, name=name)
        layers.append(fields)
    return skeleton, layers


# -- the mixin of a model built from a deck ----------------------------------

class Tabulated:
    """The behaviour of a model type built from a deck.

    The type's constructor sets `knots`, one array of knot radii per
    layer, and `header`, the deck's header; both survive every copy.
    `to_deck` samples the model on those knots, every rank-0 radial
    field by name, NaN where a layer lacks one, so that a model that
    was read can be written by `write_deck`, and a model whose fields
    were changed can be written as the deck it now is.  A model whose
    layering no longer matches its knots (after surgery) is refused.
    """

    knots: tuple[np.ndarray, ...] = ()
    header: Mapping[str, Any] = MappingProxyType({})

    def to_deck(self: Model, *, columns: Sequence[str] | None = None) -> Deck:
        """The model as a deck on its knots: `columns` are the names to
        tabulate, every rank-0 radial real field by default."""
        knots = self.knots
        intervals = [layer.interval for layer in self.layers]
        if len(knots) != self.nlayers or any(
                not (np.isclose(k[0], lo) and np.isclose(k[-1], hi))
                for k, (lo, hi) in zip(knots, intervals)):
            raise ValueError("the model's layering no longer matches its knots; "
                             "a deck is written on the layering it was read with")
        if columns is None:
            columns = [name for name in self.field_names()
                       if all(_tabulable(self.layer(i)[name])
                              for i in self.layers_with(name))]
        radius = np.concatenate(knots)
        table = {}
        for name in columns:
            parts = []
            for layer, k in zip(self.layers, knots):
                if name in layer:
                    f = layer[name]
                    if not _tabulable(f):
                        raise ValueError(f"{name!r} on layer {layer.index} is not a "
                                         "real radial field of rank 0")
                    parts.append(np.asarray(f(k), dtype=float))
                else:
                    parts.append(np.full(k.size, np.nan))
            table[name] = np.concatenate(parts)
        return Deck(radius, table, header=dict(self.header))


def _tabulable(f: Field) -> bool:
    return (bool(getattr(f, "is_radial", False)) and f.character.rank == 0
            and f.dtype == np.float64)


def deck_knots(deck: Deck) -> tuple[np.ndarray, ...]:
    """The knot radii of every layer of a deck, read-only copies."""
    out = []
    for s in deck.layers():
        k = np.array(deck.radius[s])
        k.setflags(write=False)
        out.append(k)
    return tuple(out)

