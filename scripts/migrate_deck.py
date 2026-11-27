"""A Maxwell, prony or burgers deck of planetmodel 2 as a model file.

planetmodel 2 wrote and read a viscoelastic body as a text deck: a title
line; a line naming the columns, the first word `r` for knots that are
radii from the centre outward or `depth` for depths from the free
surface down; optionally a `# layers` and a `# interfaces` line straight
after it, naming them in the order of the table with `-` for none; then
one row a knot, in SI, `nan` where a layer lacks a column, a repeated
knot a boundary.  The columns after the knot were ::

    Maxwell   rho vpv vsv vph vsh eta viscosity
    prony     rho vpv vsv vph vsh eta, then relaxation_strength_k
              relaxation_time_k for k = 1 to N
    burgers   the Maxwell deck's, then kelvin_viscosity
              kelvin_modulus_fraction
    elastic   rho vpv vsv qkappa qmu vph vsh eta, or rho vp vs qkappa
              qmu (of depths only)

This script reads such a file as a table of named columns and writes
the model of it to a model file, which holds what the deck held and
says what it means:

    python scripts/migrate_deck.py earth.deck earth.nc
    python scripts/migrate_deck.py box.deck box.nc --domain=0,400e3,-200e3,200e3

The model is the one planetmodel 2 read from the deck, and the file
holds its fields on the deck's own knots.  Every column is
joined through its knots by a cubic spline (`--kind` for another), but
a viscosity, a Kelvin element's viscosity and a relaxation time in
their logarithm by straight lines, and a relaxation strength and a
Kelvin modulus fraction by straight lines.  A layer of a prony deck
holds its branches up to the last whose strength is not zero throughout
it; a layer of a burgers deck holding a Kelvin element is a Burgers
body, held as its two Prony branches (`conversion.prony.burgers_branches`),
and one holding a viscosity alone a Maxwell body.  The layers and
interfaces take the names of the naming lines; without a line the
layers are unnamed and the interfaces too, but for the free surface
(the outermost of a table of radii, the top of a table of depths that
starts at 0).  The title is the model's name.  The model has the
behaviours its columns allow: `Elastic` with a velocity, `ConstantQ`
with a quality factor, `SelfGravitating` where the knots are radii,
`Viscoelastic` beside `Elastic`.

A table of depths holds no lateral extent, so `--domain` gives the box
the flat model is built over, `x0,x1,y0,y1` in metres, or the rectangle,
`x0,x1`.

A bare table, rows under header lines that name nothing, is read by
naming its columns after the knot: `--columns rho vpv vsv vph vsh eta
viscosity --header-lines 2`.  Its knots are radii, or depths where
`--domain` is given, and `--name` names the model.

Reading the file back is `planetmodel.modelfile.read_model(path)`, which
gives the model with its behaviours.
"""
from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from planetmodel import (ConstantQ, Deck, Elastic, RadialMesh, SelfGravitating,
                         Viscoelastic, cartesian, kappa_mu, model_from_deck, read_deck)
from planetmodel.conversion.prony import burgers_branches
from planetmodel.deck import RADIUS, DeckCoordinate
from planetmodel.materials import VELOCITY_NAMES
from planetmodel.model import ModelBase
from planetmodel.modelfile import write_model
from planetmodel.rheology import BRANCH_STEMS
from planetmodel.vocabulary import member

#: The coordinate of the knots, by the first word of a column line.
COORDINATES: dict[str, DeckCoordinate] = {"r": RADIUS, "depth": cartesian.DEPTH}

#: The keywords of the naming lines.
NAMING = ("layers", "interfaces")

#: The kinds the columns of the decks were joined by where not the
#: default: by column, or by the stem of a family of Prony branches.
KINDS = {"viscosity": "log-linear", "kelvin_viscosity": "log-linear",
         "kelvin_modulus_fraction": "linear", BRANCH_STEMS[0]: "linear",
         BRANCH_STEMS[1]: "log-linear"}

#: The Kelvin element's columns of a burgers deck.
KELVIN = ("kelvin_viscosity", "kelvin_modulus_fraction")

#: The nodes of an element of the file written, which spans two knots of
#: the deck: between them every field is one smooth function, a modulus a
#: product of cubics, of degree nine, which ten nodes hold exactly, and a
#: Burgers layer's branches to about 1e-11.
NGLL = 10

#: What a table names its layers and interfaces by: a name, or None.
type Names = list[str | None]


def header_of(lines: Sequence[str], source: str
              ) -> tuple[str, tuple[str, ...], dict[str, Names], int]:
    """The header of a deck of named columns: its title, the words of its
    column line before any `#` comment (the knot's word first), the names
    of the naming lines straight after it by their keyword, `-` read as
    None, and the number of header lines."""
    if len(lines) < 2:
        raise ValueError(f"{source}: a deck has a title line and a line naming its "
                         f"columns; the file has {len(lines)} lines")
    words = tuple(lines[1].split("#", 1)[0].split())
    if not words or words[0] not in COORDINATES:
        raise ValueError(
            f"{source}: the second line names the columns, `r` or `depth` first; got "
            f"{lines[1].strip()!r} (a bare table is read with --columns)")
    names: dict[str, Names] = {}
    n = 2
    while n < len(lines) and lines[n].strip().startswith("#"):
        key, *given = lines[n].strip()[1:].split() or [""]
        if key not in NAMING:
            break
        if key in names:
            raise ValueError(f"{source}: one `# {key}` line; the deck has two")
        names[key] = [None if word == "-" else word for word in given]
        n += 1
    return lines[0].strip(), words, names, n


def default_names(deck: Deck, coordinate: DeckCoordinate) -> dict[str, Names]:
    """The names of a deck without naming lines: the layers unnamed, and
    the interfaces too but for the free surface, `surface`."""
    n = deck.nlayers
    if coordinate is RADIUS:
        hollow = 1 if deck.radius[0] > 0.0 else 0
        faces: Names = [None] * (n - 1 + hollow) + ["surface"]
    else:
        faces = ["surface" if deck.radius[0] == 0.0 else None] + [None] * (n - 1)
    return {"layers": [None] * n, "interfaces": faces}


def behaviours_of(columns: Sequence[str], coordinate: DeckCoordinate
                  ) -> tuple[type, ...]:
    """The behaviours a table's columns allow, in the library's order."""
    elastic = any(name in VELOCITY_NAMES for name in columns)
    wanted = ((Elastic, elastic), (ConstantQ, elastic and ("qkappa" in columns
                                                           or "qmu" in columns)),
              (SelfGravitating, coordinate is RADIUS and "rho" in columns),
              (Viscoelastic, elastic))
    return tuple(behaviour for behaviour, has in wanted if has)


def with_bodies(model: ModelBase, deck: Deck) -> ModelBase:
    """The model of a deck's columns as planetmodel 2 held it: each layer
    without the branches after the last whose strength is not zero at
    every knot of it, and a layer holding a Kelvin element holding, in
    its place and the viscosity's, the two Prony branches of the Burgers
    body."""
    strength = BRANCH_STEMS[0]
    counts = [found[1] for found in map(member, deck.names)
              if found is not None and found[0] == strength]
    layers = []
    for layer, rows in zip(model.layers, deck.layers()):
        fields = dict(layer.fields)
        used = max((k for k in counts if np.any(deck[f"{strength}_{k}"][rows] != 0.0)),
                   default=0)
        for k in counts:
            if k > used:
                for stem in BRANCH_STEMS:
                    fields.pop(f"{stem}_{k}", None)
        if any(name in fields for name in KELVIN):
            if not all(name in fields for name in ("viscosity", *KELVIN)):
                raise ValueError(
                    f"layer {layer.index} holds {[n for n in KELVIN if n in fields]} "
                    "without the rest of a Burgers body: a viscosity and both of "
                    f"{list(KELVIN)}")
            kelvin = {name: fields.pop(name) for name in KELVIN}
            fields.update(burgers_branches(kappa_mu(layer)[1], fields.pop("viscosity"),
                                           **kelvin))
        layers.append(fields)
    return model.replaced(layers=layers)


def built(path: str | os.PathLike[str], *,
          domain: cartesian.Box | cartesian.Rectangle | None = None,
          kind: str = "cubic", columns: Sequence[str] | None = None,
          header_lines: int = 0, name: str | None = None) -> tuple[ModelBase, Deck]:
    """The model of the deck at `path`, as the module docstring says, and
    the deck as a table: of a deck of planetmodel 2 by its own header, or
    of a bare table of `columns` under `header_lines` lines, named `name`."""
    source = os.fspath(path)
    names: dict[str, Names] = {}
    if columns is None:
        with open(path) as fh:
            name, words, names, header_lines = header_of(fh.read().splitlines(),
                                                         source)
        coordinate = COORDINATES[words[0]]
        try:
            deck = read_deck(path, words[1:], header_lines=header_lines)
        except ValueError as err:
            raise ValueError(f"{source}: {err} (a table whose column line says more "
                             "than its columns is read with --columns and "
                             "--header-lines)") from None
    else:
        coordinate = RADIUS if domain is None else cartesian.DEPTH
        deck = read_deck(path, tuple(columns), header_lines=header_lines)
    names = {**default_names(deck, coordinate), **names}
    stems = {found[0] for found in map(member, deck.names) if found is not None}
    model = model_from_deck(
        deck, behaviours=behaviours_of(deck.names, coordinate), kind=kind,
        kinds={key: joined for key, joined in KINDS.items()
               if key in deck or key in stems},
        layer_names=names["layers"], interface_names=names["interfaces"],
        name=name or None, coordinate=coordinate, domain=domain)
    return with_bodies(model, deck), deck


def model_of(path: str | os.PathLike[str], *,
             domain: cartesian.Box | cartesian.Rectangle | None = None,
             kind: str = "cubic", columns: Sequence[str] | None = None,
             header_lines: int = 0, name: str | None = None) -> ModelBase:
    """The model of the deck at `path` (`built`)."""
    return built(path, domain=domain, kind=kind, columns=columns,
                 header_lines=header_lines, name=name)[0]


def migrate(path: str | os.PathLike[str], out: str | os.PathLike[str], *,
            domain: cartesian.Box | cartesian.Rectangle | None = None,
            kind: str = "cubic", columns: Sequence[str] | None = None,
            header_lines: int = 0, name: str | None = None) -> ModelBase:
    """Write the model of the deck at `path` (`built`) to the model file
    `out`, and return it.  The file's elements are the spans between the
    deck's knots, `NGLL` nodes each, and it records the deck's file name
    as `meta_deck`."""
    model, deck = built(path, domain=domain, kind=kind, columns=columns,
                        header_lines=header_lines, name=name)
    mesh = RadialMesh(model.skeleton, edges=np.unique(deck.radius), ngll=NGLL)
    write_model(out, model, radial=mesh, meta={"deck": Path(path).name})
    return model


def domain_of(text: str | None) -> cartesian.Box | cartesian.Rectangle | None:
    """The lateral domain of `--domain`: a box of `x0,x1,y0,y1`, a
    rectangle of `x0,x1`, None for none."""
    if text is None:
        return None
    bounds = [float(word) for word in text.split(",")]
    if len(bounds) == 2:
        return cartesian.Rectangle(tuple(bounds))
    if len(bounds) == 4:
        return cartesian.Box([tuple(bounds[:2]), tuple(bounds[2:])])
    raise ValueError(f"--domain takes x0,x1 or x0,x1,y0,y1; got {len(bounds)} numbers")


def main(argv: Sequence[str] | None = None) -> None:
    """Migrate the deck named on the command line and say what was written."""
    parser = argparse.ArgumentParser(
        description="Write a Maxwell, prony or burgers deck of planetmodel 2 as a "
                    "model file.")
    parser.add_argument("deck", type=Path, help="the deck to read")
    parser.add_argument("out", type=Path, help="the model file to write")
    parser.add_argument("--domain", metavar="BOUNDS",
                        help="the lateral domain of a table of depths, in metres: "
                             "x0,x1,y0,y1 for a box, x0,x1 for a rectangle (written "
                             "--domain=... where the first bound is negative)")
    parser.add_argument("--kind", default="cubic",
                        help="the kind the columns are joined by where the deck's "
                             "format names none (default: cubic)")
    parser.add_argument("--columns", nargs="+", metavar="NAME",
                        help="the columns after the knot of a bare table")
    parser.add_argument("--header-lines", type=int, default=0,
                        help="the lines above the rows of a bare table (default: 0)")
    parser.add_argument("--name", help="the name of the model of a bare table")
    args = parser.parse_args(argv)
    model = migrate(args.deck, args.out, domain=domain_of(args.domain), kind=args.kind,
                    columns=args.columns, header_lines=args.header_lines,
                    name=args.name)
    print(f"{args.deck} -> {args.out}")
    print(f"  {model.name or '(unnamed)'}: {model.nlayers} layers, behaviours "
          f"{' '.join(model.behaviours) or 'none'}")
    for layer in model.layers:
        print(f"  {layer.index:3d} {layer.name or '-':24s} {' '.join(layer.names)}")


if __name__ == "__main__":
    main()
