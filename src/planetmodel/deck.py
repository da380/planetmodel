"""Decks: a table of knots brought in as a model, and a model written as one.

The form of every model is the model file (`planetmodel.modelfile`),
which holds a model's fields with what they mean.  A deck is a text
table for the codes that read one, and this module does two things
with it: `model_from_deck` makes an ordinary model of a table, to be
written to a model file like any other, and `sample_deck` and
`write_deck` write a model as a table.

A deck is a table of knots, the layered coordinate first, one column
per named field, in which a repeated knot marks a boundary: the
layering comes from that structure, so a boundary is recorded even
where the tabulated values agree across it.  The coordinate increases
down the table: a radius from the centre outward (`RADIUS`), or a depth
from the free surface down (`cartesian.DEPTH`), a `DeckCoordinate`
saying which.  A table is dense enough in knots that the choice of
interpolation between them does not matter in practice.

A `DeckFormat` is what a family of files looks like: the column names
(by row width, where a format admits more than one), how many header
lines precede the table, the header a file of it is written under, and
what is written for a column a layer lacks.  `MINEOS` is the format of
the mineos and PREM decks, three header lines and nine or six columns
in SI, an absent quality factor written as 0, mineos's no loss.  `ND`
is TauP's named discontinuities, depth first from the surface in km,
km/s and g/cm^3 with label lines naming the regions below them, and
`TVEL` its tabled velocities under two header lines.  Any other table
is a format of its own, given its column names.  A format whose table
is not rows of the knot and the columns in SI reads and writes it
through hooks of its own, the deck being SI and its knot first
whatever the file.

`read_deck` turns a file into a `Deck`: the numbers, and the header
lines as they stand, which nothing reads unless asked to.  What a
header means is for the caller to say: `mineos_header` reads a mineos
header's values, `mineos_names` the names of the core its counts imply
and `mineos_constants` its reference period as the constant
`omega_ref`; `nd_names` the names an `.nd` table's labels imply.  Each
is called by whoever wants it and its result passed on.

`model_from_deck` builds the model: the layers from the repeated knots,
each column a field of every layer on the function through that layer's
knots (`deck_layers`), with the behaviours, the names of the layers and
interfaces, the constants and the scales the caller gives.  The knots
of a layer are joined by a kind of `KINDS` (a cubic spline by default,
another kind for a column, or for every member of a family, where
asked), a piecewise polynomial, a `PolynomialLayer`, so the field
algebra downstream is exact on it and the moduli of the model are
products of piecewise polynomials, never refits.  A kind of
`LOG_KINDS` (`log-linear`, `log-cubic`, ...) joins the logarithms of a
column's values by that kind instead, the value their exponential, an
`ExponentialLayer` with exact derivatives: for a column positive at
every knot that changes by orders of magnitude within a layer, a
viscosity or a relaxation time, where a cubic through the values
overshoots and can go negative and straight lines through them are far
above the geometric mean between knots.  A column constant through a
layer is that constant under every kind.  A column that is NaN
throughout a layer is absent from that layer, the way a fluid layer
holds no shear quality factor; NaN on part of a layer is refused.

`sample_deck` goes the other way, any model to a deck in SI, its knots
the model's layered coordinate: each column is the layer's field of
that name, or a velocity its density and Love moduli give, on knots
given (`deck_knots` gives those of a table read) or placed at every
breakpoint of the fields and no further apart than a fraction of the
model's span.  `write_deck` writes a deck in a format under the
format's own header, generated from the table and from what the model
said of itself when it was sampled (its name, its reference period,
the names of its interfaces); under lines the caller gives, written as
they are; or under none.
"""
from __future__ import annotations

import os
import re
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from scipy.interpolate import Akima1DInterpolator, CubicSpline, PchipInterpolator, PPoly

from numpy.typing import ArrayLike

from . import materials
from .behaviours import model_type
from .character import SCALAR, Character
from .fields import Field, RadialField
from .geometry import Geometry, GeometryBase, Names
from .layerfunction import (ExponentialLayer, PolynomialLayer, constant_layer,
                            same_interval)
from .model import Layer, Model, ModelBase
from .rheology import omega_ref_constant
from .skeleton import Skeleton
from .units import LENGTH, Scales
from .vocabulary import VOCABULARY, Constant, FieldSpec, family_spec, member

if TYPE_CHECKING:
    from .cartesian.domains import Domain

__all__ = ["Deck", "DeckFormat", "DeckCoordinate", "RADIUS", "HeaderNames", "Table",
           "MINEOS", "MINEOS_COLUMNS", "MINEOS_ISOTROPIC_COLUMNS", "ND", "TVEL",
           "ND_COLUMNS", "ND_LABELS", "read_deck", "write_deck", "deck_layers",
           "deck_knots", "model_from_deck", "mineos_header", "mineos_names",
           "mineos_constants", "nd_names", "sample_deck", "SAMPLE_SPACING", "KINDS",
           "LOG_KINDS"]

#: The interpolants a deck's knots may be joined by, each a piecewise
#: polynomial: a not-a-knot cubic spline, the parabola through three
#: knots; a natural cubic spline, as the old Fortran codes take; a
#: shape-preserving cubic (pchip), Akima's cubic, or straight lines.  Two
#: knots give a line under every kind.
KINDS = ("cubic", "natural", "pchip", "akima", "linear")

#: The kinds of `KINDS` applied to the logarithm of a column: the
#: logarithms of the knot values joined by the kind, the value their
#: exponential, so that `log-linear` is geometric between knots,
#: y_i (y_{i+1} / y_i)^t.  The values of a layer so joined are positive
#: and finite at every knot.  Two knots give `log-linear` under every one.
LOG_KINDS = tuple(f"log-{kind}" for kind in KINDS)

#: The columns of a transversely isotropic mineos deck, after the radius.
MINEOS_COLUMNS = ("rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta")

#: The columns of an isotropic mineos deck, after the radius.
MINEOS_ISOTROPIC_COLUMNS = ("rho", "vp", "vs", "qkappa", "qmu")

#: What a format's own table reader gives: the knots, the columns at
#: them, both SI, and what the table adds to the deck's header.
type Table = Callable[[Sequence[str]],
                      tuple[np.ndarray, Mapping[str, np.ndarray], Mapping[str, Any]]]

#: Names for a deck's layers and for its interfaces, by layer and by
#: interface, None where one is unnamed: what `mineos_names` and
#: `nd_names` read out of a header, and `model_from_deck` takes.
type HeaderNames = tuple[list[str | None], list[str | None]]


# -- the coordinate of a deck's knots ----------------------------------------------

@dataclass(frozen=True)
class DeckCoordinate:
    """The coordinate of a deck's knots, increasing down the table.

    `word` names it in a message about a knot (`at r = 0`); `base` is the
    leaf model type a table of such knots is a model of, whose geometry's
    `radial_field_type` is the class of the fields its columns become
    (`field_type`); and `geometry` builds that model's geometry, called
    `geometry(skeleton, domain, layer_names=..., interface_names=...)`
    with the lateral domain `model_from_deck` was given, None for none,
    and refusing one it has no use for or the lack of one it needs.
    `RADIUS` is the radius from the centre outward; `cartesian.DEPTH`
    the depth from the free surface down.
    """

    word: str
    base: type[ModelBase]
    geometry: Callable[..., GeometryBase]

    @property
    def field_type(self) -> type[RadialField]:
        """The class of the fields a deck's columns become."""
        return self.base.geometry_type.radial_field_type


def _sphere(skeleton: Skeleton, domain: Domain | None, *, layer_names: Names,
            interface_names: Names) -> Geometry:
    """The geometry of a table of radii, which has no lateral domain."""
    if domain is not None:
        raise ValueError(f"a table of radii is a model of the whole sphere and takes "
                         f"no lateral domain; got {domain!r} (a table of depths is "
                         "read with coordinate=cartesian.DEPTH)")
    return Geometry(skeleton, layer_names=layer_names, interface_names=interface_names)


#: The radius from the centre outward: `r` in a message, the knots of a
#: `Model` on a `Geometry`, the fields each a `RadialField`, the interfaces
#: one per layer and one more at the inner boundary of a hollow body.
RADIUS = DeckCoordinate("r", Model, _sphere)


# -- the deck and its format ---------------------------------------------------

@dataclass(frozen=True)
class DeckFormat:
    """What a family of deck files looks like.

    `columns` names the columns after the knot's, either one tuple or a
    mapping from row width to tuple where the format admits several (a
    deck is written with the widest it holds, `written`); `header_lines`
    is how many lines precede the table, kept as they stand by a deck
    read.  `write_header` gives the header a file of the format is
    written under where the caller gives none, from the deck and the
    columns written; without one the format has no header of its own.
    `fill` is the value written for a column where a layer lacks it, NaN
    for a column it does not name.  A table that is not rows of the knot
    and SI values has `read_table`, from the lines after the header to
    (knots, columns, more of the header) in SI and the knot first, and
    `write_table`, from a deck and the columns to write to its lines.
    """

    columns: tuple[str, ...] | Mapping[int, tuple[str, ...]] = field(hash=False)
    _: KW_ONLY
    name: str = "custom"
    header_lines: int = 0
    write_header: Callable[["Deck", tuple[str, ...]], Sequence[str]] | None = None
    fill: Mapping[str, float] = field(default_factory=dict, hash=False)
    read_table: Table | None = None
    write_table: Callable[["Deck", tuple[str, ...]], Sequence[str]] | None = None

    def names(self, width: int) -> tuple[str, ...]:
        """The column names of a row with `width` values after the knot."""
        if isinstance(self.columns, Mapping):
            try:
                return tuple(self.columns[width])
            except KeyError:
                raise ValueError(
                    f"a {self.name} deck has {sorted(self.columns)} columns after "
                    f"the knot, not {width}") from None
        if len(self.columns) != width:
            raise ValueError(
                f"a {self.name} deck has {len(self.columns)} columns after the "
                f"knot, {list(self.columns)}; the table has {width}")
        return tuple(self.columns)

    def written(self, deck: Deck) -> tuple[str, ...]:
        """The columns `write_deck` writes `deck` with, in order: the
        widest tuple of a format of several the deck holds, the one tuple
        of any other; a deck lacking a column of them is refused."""
        names: tuple[str, ...] | None
        if isinstance(self.columns, Mapping):
            widest = sorted(self.columns.items(), key=lambda item: -item[0])
            names = next((tuple(c) for _, c in widest if set(c) <= set(deck.columns)),
                         None)
        else:
            names = tuple(self.columns)
        if names is None or any(n not in deck for n in names):
            raise ValueError(f"a {self.name} deck needs columns {self.columns}; the "
                             f"deck has {list(deck.columns)}")
        return names


@dataclass(frozen=True, eq=False)
class Deck:
    """A table of knots: values of the layered coordinate, one array per
    named column, and a header.

    `radius` holds the knots, non-decreasing with every repeat a boundary:
    radii, or depths in a deck of depths (`cartesian.DEPTH`), the deck
    being the same table either way; `columns` maps each name to its
    values at the knots, NaN where a value is absent.  `header` is what
    is known beside the table, which the library reads only where asked:
    of a deck read, the header lines as they stand under "lines" (and an
    `.nd` table's labels under "labels"); of a deck sampled from a
    model, what a format's own header is written from, the model's
    "name", its reference period "tref" and the names of its "layers"
    and "interfaces".  `source` is the path of a deck read.  Arrays are
    read-only.  A deck holds arrays, so two decks are equal only where
    they are the same object, and a deck hashes by its identity.
    """

    radius: np.ndarray
    columns: Mapping[str, np.ndarray]
    _: KW_ONLY
    header: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    def __post_init__(self) -> None:
        r = np.array(self.radius, dtype=float)
        if r.ndim != 1 or r.size < 2:
            raise ValueError("a deck needs a 1-d array of at least two knots")
        if np.any(np.diff(r) < 0.0):
            raise ValueError("the knots must be non-decreasing")
        r.setflags(write=False)
        cols = {}
        for name, values in self.columns.items():
            v = np.array(values, dtype=float)
            if v.shape != r.shape:
                raise ValueError(f"column {name!r} has shape {v.shape}, the knots "
                                 f"{r.shape}")
            v.setflags(write=False)
            cols[str(name)] = v
        object.__setattr__(self, "radius", r)
        object.__setattr__(self, "columns", MappingProxyType(cols))
        object.__setattr__(self, "header", MappingProxyType(dict(self.header)))
        self.layers()                      # the structure is checked on construction

    @property
    def names(self) -> tuple[str, ...]:
        """The column names, in the deck's order."""
        return tuple(self.columns)

    @property
    def nknots(self) -> int:
        """The number of knots, a repeated knot counted twice."""
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
        """The rows of every layer, in the order of the table (centre
        outward in a deck of radii, from the surface down in one of
        depths): one slice per span between repeated knots, each of at
        least two strictly increasing knots."""
        r = self.radius
        cuts = np.flatnonzero(np.diff(r) == 0.0)
        if cuts.size and np.any(np.diff(cuts) == 1):
            raise ValueError("a knot repeats more than twice: a layer of no "
                             "thickness")
        starts = np.concatenate(([0], cuts + 1))
        stops = np.concatenate((cuts + 1, [r.size]))
        slices = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
        for s in slices:
            if s.stop - s.start < 2:
                raise ValueError(f"the layer starting at the knot {r[s.start]:g} has "
                                 "one knot; every layer needs at least two")
        return slices

    @property
    def boundaries(self) -> np.ndarray:
        """The skeleton's boundaries: the ends and every repeated knot."""
        r = self.radius
        cuts = np.flatnonzero(np.diff(r) == 0.0)
        return np.concatenate(([r[0]], r[cuts], [r[-1]]))

    @property
    def nlayers(self) -> int:
        """The number of layers the repeated knots cut the deck into."""
        return len(self.layers())

    def __repr__(self) -> str:
        return (f"Deck({self.nknots} knots, {self.nlayers} layers, columns "
                f"{list(self.columns)})")


def _as_format(format: DeckFormat | Sequence[str], header_lines: int) -> DeckFormat:
    """`format`, or the format of a table of the column names given under
    `header_lines` lines."""
    if isinstance(format, DeckFormat):
        return format
    if isinstance(format, str):
        raise TypeError(f"a format is a DeckFormat or the column names after the "
                        f"knot, a sequence of them; got the string {format!r}")
    return DeckFormat(tuple(format), header_lines=header_lines)


def read_deck(source: str | os.PathLike[str] | Iterable[str],
              format: DeckFormat | Sequence[str], *,
              header_lines: int | None = None) -> Deck:
    """A deck from a file, or from lines: the format's header lines, kept
    as they stand under "lines" in the deck's header and read no further,
    then whitespace-separated rows, the knot first.

    `format` is a `DeckFormat`, or the column names after the knot for a
    table under `header_lines` lines (none by default).  Blank lines and
    `#` comments in the table are skipped.
    """
    if isinstance(source, (str, os.PathLike)):
        path: str | None = os.fspath(source)
        with open(source) as fh:
            lines = fh.read().splitlines()
    else:
        path = None
        lines = list(source)
    if isinstance(format, DeckFormat) and header_lines is not None:
        raise ValueError("header_lines is set by the DeckFormat")
    format = _as_format(format, header_lines or 0)
    n = format.header_lines
    if len(lines) < n:
        raise ValueError(f"{path or 'the deck'}: {len(lines)} lines, fewer than the "
                         f"{n} header lines of a {format.name} deck")
    header: dict[str, Any] = {"lines": tuple(lines[:n])}
    if format.read_table is not None:
        radius, columns, more = format.read_table(lines[n:])
        return Deck(radius, columns, header={**header, **more}, source=path)
    body = [ln for ln in lines[n:] if ln.strip()]
    if not body:
        raise ValueError(f"{path or 'the deck'}: no knots after the header")
    table = np.loadtxt(body, dtype=float, ndmin=2)
    names = format.names(table.shape[1] - 1)
    return Deck(table[:, 0], {nm: table[:, j + 1] for j, nm in enumerate(names)},
                header=header, source=path)


def write_deck(path: str | os.PathLike[str], deck: Deck,
               format: DeckFormat | Sequence[str], *,
               header: Sequence[str] | bool = True) -> Path:
    """Write a deck in a format: a header, then one row per knot, the knot
    first and the format's columns in order (`DeckFormat.written`), the
    knot as `%14.3f` and values as `%16.8g`.

    `format` is a `DeckFormat`, or the column names after the knot for a
    plain table of them.  `header` is True for the format's own header,
    generated from the deck (`MINEOS`: its three lines; `TVEL`: its two
    title lines; none for a format without one, a plain table among
    them); lines, written as they are in place of it, whatever their
    number; or False for none.  A column the deck lacks is refused, a
    column the format does not name is not written, and an absent value
    is written as the format's `fill` for its column, else as nan.
    """
    format = _as_format(format, 0)
    names = format.written(deck)
    if isinstance(header, (bool, np.bool_)):
        own = header and format.write_header is not None
        lines = list(format.write_header(deck, names)) if own else []
    elif isinstance(header, str):
        raise TypeError(f"header is True, False or the header's lines, a sequence of "
                        f"them; got the string {header!r}")
    else:
        lines = [str(line) for line in header]
    if format.fill:
        filled = {n: np.where(np.isnan(deck[n]), format.fill[n], deck[n])
                  for n in names if n in format.fill}
        deck = Deck(deck.radius, {**deck.columns, **filled}, header=deck.header,
                    source=deck.source)
    if format.write_table is not None:
        body = list(format.write_table(deck, names))
    else:
        rows = np.column_stack([deck.radius] + [deck[n] for n in names])
        body = [f"{row[0]:14.3f}" + "".join(f"{v:16.8g}" for v in row[1:])
                for row in rows]
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as fh:
        for line in lines + body:
            fh.write(line.rstrip("\n") + "\n")
    return out


# -- the mineos format -------------------------------------------------------------

def mineos_header(deck: Deck) -> dict[str, Any]:
    """What the three header lines of a mineos deck say, for a caller who
    wants it: the title under "name"; `ifanis tref ifdeck` of the second
    line; `nknot nic noc` of the third.  `tref` is the period in seconds
    the elastic values are given at, 0 for none; `nic` and `noc` count
    the knots of the inner core and to the top of the outer core.  A deck
    whose header lines are not those is refused."""
    lines = tuple(deck.header.get("lines", ()))
    rule = ("a mineos deck's header is a title line, then `ifanis tref ifdeck` and "
            "`nknot nic noc`")
    if len(lines) < 3:
        raise ValueError(f"{rule}; the deck holds {len(lines)} header lines")
    h2, h3 = lines[1].split(), lines[2].split()
    try:
        return {"name": lines[0].strip(), "ifanis": int(h2[0]), "tref": float(h2[1]),
                "ifdeck": int(h2[2]), "nknot": int(h3[0]), "nic": int(h3[1]),
                "noc": int(h3[2])}
    except (IndexError, ValueError):
        raise ValueError(f"{rule}; got {lines[1]!r} and {lines[2]!r}") from None


def mineos_constants(deck: Deck) -> dict[str, Constant]:
    """The constants a mineos header declares, to give a model of the
    deck: `omega_ref`, the angular frequency of its reference period
    `tref`, which a constant-Q band disperses the elastic values about;
    none where `tref` is 0."""
    tref = mineos_header(deck)["tref"]
    if not tref > 0.0:
        return {}
    return {"omega_ref": omega_ref_constant(tref)}


def _core_counts(deck: Deck, names: tuple[str, ...]) -> tuple[int, int]:
    """(nic, noc) of a table: the outer core is the innermost layer whose
    S velocity is zero throughout, `noc` the knots up to its top and
    `nic` those beneath it, 0 where no solid layer lies beneath; (0, 0)
    for a body with no fluid layer."""
    vs = deck["vsv" if "vsv" in names else "vs"]
    for s in deck.layers():
        if np.all(vs[s] == 0.0):
            return s.start, s.stop
    return 0, 0


def _mineos_header_lines(deck: Deck, names: tuple[str, ...]) -> list[str]:
    """The three header lines of a mineos deck: the title, the deck's
    "name"; `ifanis tref ifdeck`, the flag 1 where the columns are the
    transversely isotropic ones and `tref` the deck's, 0 for none; and
    `nknot nic noc`, counted from the table (`_core_counts`)."""
    h = deck.header
    nic, noc = _core_counts(deck, names)
    return [str(h.get("name") or deck.source or "deck"),
            f"{int('vph' in names)} {float(h.get('tref', 0.0)):g} 1",
            f"{deck.nknots} {nic} {noc}"]


#: The mineos deck: three header lines, then `r rho vpv vsv qkappa qmu vph
#: vsh eta` in SI (nine columns) or `r rho vp vs qkappa qmu` (six); an
#: absent quality factor is written as 0, which mineos reads as no loss.
#: Its own header is a title, the model's name; `ifanis tref ifdeck`,
#: `tref` the period of the model's reference frequency and 0 where it
#: declares none; and `nknot nic noc`, the counts those of the table, the
#: outer core its innermost layer whose S velocity is zero throughout
#: (`0 0` for a body with no fluid layer).
MINEOS = DeckFormat({len(MINEOS_COLUMNS): MINEOS_COLUMNS,
                     len(MINEOS_ISOTROPIC_COLUMNS): MINEOS_ISOTROPIC_COLUMNS},
                    name="mineos", header_lines=3, write_header=_mineos_header_lines,
                    fill={"qkappa": 0.0, "qmu": 0.0})


# -- TauP's named discontinuities ----------------------------------------------

#: The columns of an `.nd` or `.tvel` row after the depth, by how many there
#: are: TauP's Qs is the shear quality factor, `qmu`, its Qp `qp`.
ND_COLUMNS = {5: ("vp", "vs", "rho", "qp", "qmu"), 4: ("vp", "vs", "rho", "qp"),
              3: ("vp", "vs", "rho"), 2: ("vp", "vs")}

#: The SI size of one unit of each `.nd` column: km/s, g/cm^3, pure numbers.
_ND_UNITS = {"vp": 1e3, "vs": 1e3, "rho": 1e3, "qp": 1.0, "qmu": 1.0}

#: TauP's labels of the boundaries every planetary model has, and the
#: interface names they give: a label names the region below it.  The
#: first label of each interface is the one written for it.
ND_LABELS = {"mantle": "moho", "moho": "moho", "outer-core": "cmb", "cmb": "cmb",
             "inner-core": "icb", "icocb": "icb"}


def _nd_label(line: str) -> str:
    """A label line in TauP's normal form: lower case, words joined by '-'."""
    return "-".join(line.lower().split())


def _nd_table(lines: Sequence[str]) -> tuple[np.ndarray, dict[str, np.ndarray],
                                              dict[str, Any]]:
    """The rows of an `.nd` table: comments (`#`, `//`, `/* */`) dropped,
    rows `depth vp vs [rho [qp [qs]]]` of one width, a label line before
    the first row (the outermost region) or between the two rows of a
    repeated depth (the region below the boundary there); radii from the
    deepest depth, which is the body's radius, in SI, from the centre."""
    text = re.sub(r"/\*.*?\*/", " ", "\n".join(lines), flags=re.S)
    rows: list[list[float]] = []
    at: list[tuple[int, str]] = []
    for raw in text.splitlines():
        line = re.split(r"#|//", raw, maxsplit=1)[0].strip()
        if not line:
            continue
        tokens = line.split()
        try:
            float(tokens[0])
        except ValueError:
            at.append((len(rows), _nd_label(line)))
            continue
        try:
            rows.append([float(t) for t in tokens])
        except ValueError:
            raise ValueError(f"cannot read the row {line!r}: a row is numbers, a "
                             "label starts with a letter") from None
    widths = sorted({len(r) for r in rows})
    if not rows or len(widths) != 1 or widths[0] - 1 not in ND_COLUMNS:
        raise ValueError(f"the rows of an .nd table are depth vp vs [rho [qp [qs]]], "
                         f"three to six numbers, the same on every row; got widths "
                         f"{widths or 'none'}")
    table = np.array(rows)
    depth = table[:, 0]
    if np.any(np.diff(depth) < 0.0):
        raise ValueError("the depths of an .nd table must not decrease")
    surface = float(depth[-1])
    labels = []
    for k, label in at:
        if not (k == 0 or (0 < k < depth.size and depth[k - 1] == depth[k])):
            raise ValueError(f"the label {label!r} sits at no boundary: a label comes "
                             "before the first row or between the two rows of a "
                             "repeated depth")
        labels.append(((surface - float(depth[k if k < depth.size else -1])) * 1e3,
                       label))
    names = ND_COLUMNS[widths[0] - 1]
    radius = (surface - depth[::-1]) * 1e3
    columns = {n: table[::-1, j + 1] * _ND_UNITS[n] for j, n in enumerate(names)}
    return radius, columns, {"labels": tuple(labels)}


def _nd_labels(deck: Deck) -> dict[float, str]:
    """The labels an `.nd` table of a deck is written with, by the radius
    they sit at: those the deck was read with, or else the ones the names
    of its interfaces give, `ND_LABELS`'s first for `moho`, `cmb` and
    `icb`, each naming the region below its boundary."""
    if "labels" in deck.header:
        return {float(r): label for r, label in deck.header["labels"]}
    label_of = {face: label for label, face in reversed(ND_LABELS.items())}
    b = deck.boundaries
    faces = tuple(deck.header.get("interfaces", ()))
    if not faces:
        return {}
    first = len(faces) - deck.nlayers          # one more for a hollow body's inner
    if first not in (0, 1):
        raise ValueError(f"the deck's header names {len(faces)} interfaces; its "
                         f"table has {deck.nlayers} layers, an interface each and one "
                         "more for a hollow body's inner boundary")
    return {float(b[k]): label_of[faces[k - 1 + first]]
            for k in range(1, b.size - 1) if faces[k - 1 + first] in label_of}


def _nd_rows(deck: Deck, names: tuple[str, ...], labels: Mapping[float, str]
             ) -> list[str]:
    """The rows of an `.nd` table from a deck, from the surface down, each
    of `labels` between the two rows of its boundary (before the first row
    for the surface's); an absent value is refused."""
    radius, top = deck.radius, float(deck.radius[-1])
    values = np.column_stack([deck[n] / _ND_UNITS[n] for n in names])
    if np.isnan(values).any():
        raise ValueError("an .nd table holds every column on every row; the deck "
                         "has absent values")
    out = [labels[top]] if top in labels else []
    for i in range(radius.size - 1, -1, -1):
        r = float(radius[i])
        if i < radius.size - 1 and r == float(radius[i + 1]) and r in labels:
            out.append(labels[r])
        depth = (top - r) / 1e3
        out.append(" ".join([f"{depth:.10g}"] + [f"{v:.10g}" for v in values[i]]))
    return out


def _nd_table_lines(deck: Deck, names: tuple[str, ...]) -> list[str]:
    """An `.nd` table of a deck: its rows with its labels (`_nd_labels`)."""
    return _nd_rows(deck, names, _nd_labels(deck))


def _tvel_table_lines(deck: Deck, names: tuple[str, ...]) -> list[str]:
    """A `.tvel` table of a deck: its rows, a `.tvel` holding no labels."""
    return _nd_rows(deck, names, {})


def _tvel_header_lines(deck: Deck, names: tuple[str, ...]) -> list[str]:
    """The two title lines of a `.tvel` table, which TauP skips: the
    deck's name, marked P and S as TauP's own tables are."""
    title = str(deck.header.get("name") or deck.source or "deck")
    return [f"{title} - P", f"{title} - S"]


#: TauP's named discontinuities: no header, rows `depth vp vs [rho [qp
#: [qs]]]` from the surface down in km, km/s and g/cm^3, a repeated depth a
#: boundary, a label line naming the region below it.  A table is written
#: with the labels TauP knows the Moho and the core by (`mantle`,
#: `outer-core`, `inner-core`) at the interfaces named `moho`, `cmb` and
#: `icb`, or with the labels it was read with.
ND = DeckFormat(ND_COLUMNS, name="nd", read_table=_nd_table,
                write_table=_nd_table_lines)

#: TauP's tabled velocities: two title lines, then `.nd` rows without
#: labels.
TVEL = DeckFormat(ND_COLUMNS, name="tvel", header_lines=2, read_table=_nd_table,
                  write_header=_tvel_header_lines, write_table=_tvel_table_lines)


def nd_names(deck: Deck) -> HeaderNames:
    """Layer and interface names an `.nd` table's labels imply: each label
    names the layer below it (its words joined by '_'), and the labels of
    `ND_LABELS` name their interfaces too (`moho`, `cmb`, `icb`); the
    outermost interface is `surface`.  Every other name is None.  A label
    at the innermost boundary, of a solid body or a hollow one, is refused:
    no layer lies below it."""
    b = deck.boundaries
    first = 1 if b[0] > 0 else 0
    layers: list[str | None] = [None] * deck.nlayers
    faces: list[str | None] = [None] * (deck.nlayers - 1 + first) + ["surface"]
    for r, label in deck.header.get("labels", ()):
        k = int(np.argmin(np.abs(b - float(r))))
        if k == 0:
            raise ValueError(f"the label {label!r} sits at the innermost boundary, "
                             "where no layer lies below it")
        layers[k - 1] = label.replace("-", "_")
        if label in ND_LABELS and k < b.size - 1:
            faces[k - 1 + first] = ND_LABELS[label]
    return layers, faces


def mineos_names(deck: Deck) -> HeaderNames:
    """Layer and interface names a mineos header implies (`mineos_header`):
    `nic` and `noc` count the knots of the inner core and to the top of
    the outer core, so where each sits on a repeated radius the layer
    beneath it is `inner_core` or `outer_core` under `icb` or `cmb`; the
    outermost interface is `surface`.  Every other name is None.  A count
    of zero says there is no such core; a header whose other counts do
    not sit on repeated radii, or whose `nknot` is not the table's, is
    warned about and the table taken as the truth."""
    layers: list[str | None] = [None] * deck.nlayers
    faces: list[str | None] = [None] * (deck.nlayers + (1 if deck.radius[0] > 0 else 0))
    faces[-1] = "surface"
    r, h = deck.radius, mineos_header(deck)
    first = 1 if deck.radius[0] > 0 else 0
    cores = (("nic", "inner_core", "icb"), ("noc", "outer_core", "cmb"))
    for key, layer, face in cores:
        i = h[key]
        if not (1 <= i < r.size and r[i - 1] == r[i]):
            if i != 0:
                warnings.warn(f"the header's {key} = {i} does not sit on a repeated "
                              "radius; the layer is left unnamed", stacklevel=2)
            continue
        k = int(np.searchsorted(deck.boundaries, r[i]))      # the boundary index
        if 0 < k < deck.nlayers:
            layers[k - 1] = layer
            faces[k - 1 + first] = face
    if h["nknot"] != deck.nknots:
        warnings.warn(f"the header says {h['nknot']} knots, the table has "
                      f"{deck.nknots}; the table is taken as the truth", stacklevel=2)
    return layers, faces


# -- fields by interpolation ---------------------------------------------------

def _check_kind(kind: str) -> None:
    """Refuse a kind of neither `KINDS` nor `LOG_KINDS`."""
    if kind not in KINDS and kind not in LOG_KINDS:
        raise ValueError(f"kind must be one of {KINDS + LOG_KINDS}, got {kind!r}")


def _interpolant(r: np.ndarray, y: np.ndarray, kind: str
                 ) -> PolynomialLayer | ExponentialLayer:
    """The function through the knots (r, y) of one layer: the piecewise
    polynomial of `kind`, or for a kind of `LOG_KINDS` y_0 exp(L), L the
    polynomial of its kind through log(y_i / y_0), the values being
    positive and finite and their ratios within the range of a float; a
    constant y is that constant under every kind."""
    _check_kind(kind)
    if not np.all(np.diff(r) > 0.0):
        raise ValueError("the knots of a layer must increase strictly")
    if kind in LOG_KINDS:
        if not np.all(np.isfinite(y) & (y > 0.0)):
            raise ValueError(f"a column joined by {kind} is positive and finite at "
                             "every knot")
        interval = (float(r[0]), float(r[-1]))
        if np.all(y == y[0]):
            return constant_layer(interval, float(y[0]))
        with np.errstate(over="ignore", under="ignore", divide="ignore"):
            ratios = np.log(y / y[0])
        if not np.all(np.isfinite(ratios)):
            raise ValueError(f"a column joined by {kind} spans more than the range of "
                             "a float within a layer: the ratio of two of its values "
                             "is not finite")
        log = _interpolant(r, ratios, kind.removeprefix("log-"))
        return ExponentialLayer(interval, log, factor=float(y[0]))
    if r.size == 2 or kind == "linear":
        slopes = np.diff(y) / np.diff(r)
        ppoly = PPoly(np.vstack([slopes, y[:-1]]), r)
    elif kind == "cubic":
        ppoly = CubicSpline(r, y)
    elif kind == "natural":
        ppoly = CubicSpline(r, y, bc_type="natural")
    elif kind == "pchip":
        ppoly = PchipInterpolator(r, y)
    else:
        ppoly = Akima1DInterpolator(r, y)
    return PolynomialLayer((float(r[0]), float(r[-1])), ppoly)


def deck_layers(deck: Deck, *, kind: str = "cubic",
                kinds: Mapping[str, str] | None = None,
                specs: Mapping[str, FieldSpec] | None = None,
                coordinate: DeckCoordinate = RADIUS
                ) -> tuple[Skeleton, list[dict[str, Field]]]:
    """The skeleton of a deck and the base fields of every layer.

    Every column becomes one field per layer, of the `field_type` of the
    knots' `coordinate` (a `RadialField` of the radius by default, a
    `cartesian.DepthField` for `cartesian.DEPTH`), on the function
    through that layer's knots of the kind `kinds` gives the column, or
    else the stem of its family (`vocabulary.member`), or else of
    `kind`, with the character the vocabulary or `specs` give the name
    (SCALAR for a name neither knows).  A kind of `KINDS` gives the
    piecewise polynomial, a `PolynomialLayer`; one of `LOG_KINDS` the
    exponential of that kind through the logarithms of the values, an
    `ExponentialLayer`, or the constant where the values are all equal,
    a `PolynomialLayer`.  A column that is NaN throughout a layer is
    absent from that layer; NaN on part of one is refused, as are a
    column joined by a kind of `LOG_KINDS` that is not positive and
    finite at every knot of a layer holding it, a kind of neither tuple
    and a name of `kinds` that is neither a column of the deck nor the
    stem of a family it holds a member of, each naming the knot by the
    coordinate's word.
    """
    kinds = dict(kinds or {})
    for k in (kind, *kinds.values()):
        _check_kind(k)
    stems = {found[0] for found in map(member, deck.names) if found is not None}
    lacking = [name for name in kinds if name not in deck and name not in stems]
    if lacking:
        raise ValueError(f"kinds names columns the deck lacks, {lacking}; it has "
                         f"{list(deck.names)}")

    def kind_of(name: str) -> str:
        """The kind a column is joined by."""
        found = member(name)
        if name not in kinds and found is not None and found[0] in kinds:
            return kinds[found[0]]
        return kinds.get(name, kind)

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
            joined = kind_of(name)
            off = ~(np.isfinite(y) & (y > 0.0))
            if joined in LOG_KINDS and off.any():
                j = int(np.argmax(off))
                raise ValueError(f"column {name!r} is {y[j]:g} at {coordinate.word} = "
                                 f"{r[j]:g} on "
                                 f"layer {i}; a column joined by {joined}, its "
                                 "logarithm interpolated, is positive and finite at "
                                 "every knot of a layer holding it")
            spec = known.get(name) or family_spec(name)
            character: Character = SCALAR if spec is None else spec.character
            fields[name] = coordinate.field_type(skeleton.interval(i),
                                                 _interpolant(r, y, joined),
                                                 character=character, name=name)
        layers.append(fields)
    return skeleton, layers


def model_from_deck(deck: Deck, *, behaviours: Iterable[type] = (),
                    kind: str = "cubic", kinds: Mapping[str, str] | None = None,
                    specs: Mapping[str, FieldSpec] | None = None,
                    layer_names: Names = None, interface_names: Names = None,
                    constants: Mapping[str, Constant] | None = None,
                    scales: Scales = Scales.SI, name: str | None = None,
                    citation: Iterable[str] = (), coordinate: DeckCoordinate = RADIUS,
                    domain: Domain | None = None) -> ModelBase:
    """The model of a deck: an ordinary model, to be used, changed and
    written to a model file like any other.

    Its layers are the deck's, cut at the repeated knots, and the fields
    of each its columns, joined by `kind`, and by the kind `kinds` gives
    a column or the stem of a family where it gives one (`deck_layers`:
    `kinds={"viscosity": "log-linear"}` joins a viscosity in its
    logarithm), with the characters the vocabulary or `specs` give the
    names.  Everything else is the caller's to say, nothing being read
    from the deck's header: `behaviours` are the mixins of
    `planetmodel.behaviours` the model has (`behaviours.BEHAVIOURS` for
    all four; `Elastic` completes the Love moduli beside the velocities,
    and wants every layer's elastic description whole), its class
    `behaviours.model_type` of them; `layer_names` and
    `interface_names` name the layers and interfaces, in the deck's
    order; `constants` are the model's own (`omega_ref`, say); `scales`
    are the units the table's numbers are in, SI as every format here
    reads them; `name` and `citation` are the model's.  `mineos_names`,
    `mineos_constants`, `mineos_header` and `nd_names` read such things
    out of a header for a caller who trusts it.

    `coordinate` says what the knots are: radii from the centre outward
    by default, a model of the sphere; with `cartesian.DEPTH` depths
    from the free surface down, a flat model over the lateral `domain`
    (a `cartesian.Box` or `Rectangle`), which a table does not hold and
    the caller gives.
    """
    skeleton, layers = deck_layers(deck, kind=kind, kinds=kinds, specs=specs,
                                   coordinate=coordinate)
    geometry = coordinate.geometry(skeleton, domain, layer_names=layer_names,
                                   interface_names=interface_names)
    return model_type(coordinate.base, behaviours)(
        geometry, layers, scales=scales, specs=specs, constants=constants, name=name,
        citation=citation)


# -- any model as a deck ---------------------------------------------------------

#: The widest gap `sample_deck` leaves between knots by default, as a
#: fraction of the model's span, its skeleton's last boundary less its
#: first: about 32 km on the Earth, the density of PREM's own tabulation,
#: and 1 km in a box 200 km deep.
SAMPLE_SPACING = 1.0 / 200.0

#: The fewest knots `sample_deck` puts on a polynomial piece of a layer,
#: so that a not-a-knot cubic through them reproduces a cubic piece.
_PIECE_KNOTS = 4


def sample_deck(model: ModelBase, columns: Sequence[str], *,
                knots: Sequence[ArrayLike] | None = None,
                spacing: float = SAMPLE_SPACING) -> Deck:
    """A model as a deck, in SI.

    `columns` are the names to tabulate, in order.  On each layer a
    column is the layer's field of that name, which must be a real
    radial field of rank 0; or a velocity the layer lacks (`vpv`, `vph`,
    `vsv`, `vsh`, `eta`, and `vp` and `vs` of an isotropic layer), from
    its density and Love moduli by `materials.velocities_from_moduli`;
    NaN where the layer has neither.  The knots are values of the model's
    layered coordinate (radii, or depths in a flat model).  `knots` are
    one array per layer in the model's units, each increasing from one
    end of the layer to the other (`deck_knots` gives a table's).
    Without them each layer is cut at every breakpoint of the fields
    sampled on it, and each piece evenly, to gaps no wider than `spacing`
    times the model's span and at least four knots, so that a not-a-knot
    cubic through them reproduces a cubic piece exactly.  The header
    holds what a format's own header is written from (`write_deck`): the
    model's name where it has one, the names of its layers and
    interfaces under "layers" and "interfaces", and the period of a
    declared `omega_ref` under "tref", in seconds.  A model in other
    scales is sampled in SI.
    """
    if not model.scales.is_si:
        if knots is not None:
            unit = model.scales.factor(LENGTH)
            knots = [np.asarray(k, dtype=float) * unit for k in knots]
        return sample_deck(model.in_si(), columns, knots=knots, spacing=spacing)
    columns = tuple(columns)
    if len(set(columns)) != len(columns) or not columns:
        raise ValueError(f"name each column once, and at least one; got "
                         f"{list(columns)}")
    if not spacing > 0.0:
        raise ValueError(f"spacing is a positive fraction of the model's span, got "
                         f"{spacing}")
    if knots is None:
        step = spacing * model.skeleton.span
        per_layer = [_layer_knots(layer, columns, step, model.geometry.rtol)
                     for layer in model.layers]
    else:
        per_layer = [np.asarray(k, dtype=float) for k in knots]
        rtol = model.geometry.rtol
        if len(per_layer) != model.nlayers or any(
                k.ndim != 1 or k.size < 2
                or not same_interval((k[0], k[-1]), layer.interval, rtol=rtol)
                for k, layer in zip(per_layer, model.layers)):
            raise ValueError("the model's layering no longer matches the knots: they "
                             "are one array per layer, each spanning its layer")
        if any(np.any(np.diff(k) <= 0.0) for k in per_layer):
            raise ValueError("the knots of a layer must increase strictly")
    table: dict[str, list[np.ndarray]] = {name: [] for name in columns}
    for layer, k in zip(model.layers, per_layer):
        for name, values in _sampled(layer, columns, k).items():
            table[name].append(values)
    header: dict[str, Any] = {
        "layers": tuple(layer.name for layer in model.layers),
        "interfaces": tuple(face.name for face in model.geometry.interfaces)}
    if model.name:
        header["name"] = model.name
    if "omega_ref" in model.constants:
        header["tref"] = 2.0 * np.pi / float(model.constant("omega_ref"))
    return Deck(np.concatenate(per_layer),
                {name: np.concatenate(parts) for name, parts in table.items()},
                header=header)


def _layer_knots(layer: Layer, columns: tuple[str, ...], step: float,
                 rtol: float) -> np.ndarray:
    """A layer's default knots: its ends and every breakpoint of the
    fields sampled on it (those of the columns, and rho and the moduli
    where a velocity is derived), each piece cut evenly to gaps no wider
    than `step`, with at least `_PIECE_KNOTS` knots."""
    lo, hi = layer.interval
    fields = [layer[name] for name in columns if name in layer]
    if any(name in materials.VELOCITY_NAMES and name not in layer for name in columns):
        fields += [layer[name] for name in ("rho", "A", "C", "F", "L", "N")
                   if name in layer]
    breaks = [np.array([lo, hi])]
    for f in fields:
        pieces = getattr(f, "radial_pieces", None)
        found = pieces() if pieces is not None else None
        if found is not None:
            breaks.append(np.asarray(found[0], dtype=float))
    points = np.unique(np.clip(np.concatenate(breaks), lo, hi))
    close = rtol * max(abs(lo), abs(hi))
    kept = [points[0]]
    for x in points[1:-1]:
        if x - kept[-1] > close and hi - x > close:
            kept.append(x)
    kept.append(hi)
    out = []
    for a, b in zip(kept[:-1], kept[1:]):
        n = max(_PIECE_KNOTS, int(np.ceil((b - a) / step)) + 1)
        out.append(np.linspace(a, b, n)[:-1])
    out.append(np.array([hi]))
    return np.concatenate(out)


def _sampled(layer: Layer, columns: tuple[str, ...], r: np.ndarray
             ) -> dict[str, np.ndarray]:
    """Each column on one layer at the radii `r`: its field, a velocity
    its density and moduli give, or NaN."""
    out: dict[str, np.ndarray] = {}
    derived: dict[str, np.ndarray] | None = None
    for name in columns:
        if name in layer:
            f = layer[name]
            if not _tabulable(f):
                raise ValueError(f"{name!r} on layer {layer.index} is not a real "
                                 "radial field of rank 0")
            out[name] = np.asarray(f(r), dtype=float)
            continue
        if name in materials.VELOCITY_NAMES:
            if derived is None:
                derived = _velocities(layer, r)
            if name in derived:
                out[name] = derived[name]
                continue
            isotropic = materials.VELOCITY_SETS[materials.Symmetry.ISOTROPIC]
            if derived and name in isotropic:
                raise ValueError(
                    f"{name!r} asked of layer {layer.index}, which is transversely "
                    "isotropic; ask for vpv, vph, vsv, vsh and eta, or make it "
                    "isotropic() first")
        out[name] = np.full(r.size, np.nan)
    return out


def _velocities(layer: Layer, r: np.ndarray) -> dict[str, np.ndarray]:
    """The velocities a layer's density and Love moduli give at `r`, vp
    and vs too where the layer is isotropic; none where it holds no
    density or no elastic description."""
    if "rho" not in layer or not any(n in layer for n in materials.ELASTIC_NAMES):
        return {}
    try:
        five = materials.moduli(layer)
        symmetry, _ = materials.independent_moduli(layer)
    except KeyError as err:
        raise ValueError(f"no velocities of layer {layer.index}: "
                         f"{err.args[0]}") from None
    if any(f.dtype != np.float64 for f in (layer["rho"], *five.values())):
        raise ValueError(f"the moduli of layer {layer.index} are not real (a model "
                         "frozen at a frequency, say); a deck holds real fields")
    out = materials.velocities_from_moduli(layer["rho"](r), *(five[n](r) for n in
                                                            ("A", "C", "F", "L", "N")))
    if symmetry is materials.Symmetry.ISOTROPIC:
        out["vp"], out["vs"] = out["vpv"], out["vsv"]
    return out


def _tabulable(f: Field) -> bool:
    return (bool(getattr(f, "is_radial", False)) and f.character.rank == 0
            and f.dtype == np.float64)


def deck_knots(deck: Deck) -> tuple[np.ndarray, ...]:
    """The knots of every layer of a deck, read-only copies: what
    `sample_deck` takes as `knots` to tabulate a model where a table has
    its own."""
    out = []
    for s in deck.layers():
        k = np.array(deck.radius[s])
        k.setflags(write=False)
        out.append(k)
    return tuple(out)
