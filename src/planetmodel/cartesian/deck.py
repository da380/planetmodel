"""Decks of depths: the tables a flat model is read from and written to.

A flat model's deck is a table of knots whose first column is the
depth, increasing down the file from the free surface at 0, one column
per named field, a repeated depth a boundary between two layers; the
layers run from the surface down, as the model's do, and the last knot
is the base, a face of the box and no interface.  It is a `deck.Deck`
like any other, its `radius` holding the depths, read by
`planetmodel.read_deck`, written by `write_deck`, and sampled from a
model by `sample_deck`, all the library's.

The formats are those of the radius-first decks stated for `DEPTH`, the
depth from the free surface down, each one definition for either
coordinate (`deck.elastic_format`, `maxwell_format`, `prony_format`,
`burgers_format`): the column names, the title, the column line and the
`# layers` and `# interfaces` lines, the kinds that join a column
(`MAXWELL_KINDS`, `PRONY_KINDS`, `BURGERS_KINDS`, the viscosities and
times in their logarithm), and the completion and rules of the prony
and burgers decks are theirs.  What differs is said by `DEPTH`: the
column line names the knots `depth`, a message names a knot's depth,
the columns become fields, each a `DepthField` (`deck_layers(deck,
coordinate=DEPTH)`), and the naming lines name one interface per layer
from the surface down, the top first (`depth_names`), where a deck of
radii names one more for a hollow body's inner boundary.  A deck holds
no lateral extent: the box a deck's model is built over is given
beside the deck (`cartesian.catalogue`).  An elastic deck of a crust
over a mantle, its quality factors absent (rows read as whitespace-
separated numbers; written with the knot as `%14.3f` and each value as
`%16.8g`)::

    a crust over a mantle
    depth rho vp vs qkappa qmu
    # layers crust mantle
    # interfaces surface moho
    0       2800  6400  3700  nan  nan
    30000   2800  6400  3700  nan  nan
    30000   3300  8000  4500  nan  nan
    200000  3300  8000  4500  nan  nan

::

  DEPTH        the depth from the free surface down
  ELASTIC      depth rho vpv vsv qkappa qmu vph vsh eta, or
               depth rho vp vs qkappa qmu
  MAXWELL      depth rho vpv vsv vph vsh eta viscosity
  PRONY        depth rho vpv vsv vph vsh eta, then the branches'
               relaxation_strength_k relaxation_time_k, k = 1 to N
  BURGERS      the Maxwell deck's, then kelvin_viscosity
               kelvin_modulus_fraction
  depth_names  the layer and interface names a header gives
"""
from __future__ import annotations

from ..deck import (Deck, DeckCoordinate, HeaderNames, burgers_format, elastic_format,
                    header_names, maxwell_format, prony_format)
from .coordinates import FLAT
from .fields import DepthField

__all__ = ["DEPTH", "ELASTIC", "MAXWELL", "PRONY", "BURGERS", "depth_names"]


def depth_names(deck: Deck) -> HeaderNames:
    """Layer and interface names a header of depths gives: those of its
    `# layers` and `# interfaces` lines, from the surface down, None for
    `-`.  Without a line the layers are unnamed, and the interfaces too
    but for the top, `surface` where the deck starts at depth 0.  A count
    other than the deck's layers, or its interfaces (one per layer, the
    base being none), is refused."""
    top = "surface" if float(deck.radius[0]) == 0.0 else None
    return header_names(deck, [top] + [None] * (deck.nlayers - 1))


#: The depth from the free surface down: `depth` in a column line, the
#: layered coordinate of `FLAT`, the fields each a `DepthField`, one interface
#: per layer, the top first.
DEPTH = DeckCoordinate("depth", FLAT, DepthField, depth_names)

#: The elastic deck of depths: a title line, the column line `depth rho
#: vpv vsv qkappa qmu vph vsh eta` or `depth rho vp vs qkappa qmu`,
#: optionally `# layers` and `# interfaces` lines, then the rows in SI,
#: NaN where a layer lacks a field.
ELASTIC = elastic_format(DEPTH)

#: The Maxwell deck of depths: a title line, the column line `depth rho
#: vpv vsv vph vsh eta viscosity`, optionally the naming lines, then the
#: rows in SI, NaN where a layer lacks a field.
MAXWELL = maxwell_format(DEPTH)

#: The prony deck of depths: the Maxwell deck's, the viscosity replaced by
#: the pairs `relaxation_strength_k relaxation_time_k` for k = 1 to N.
PRONY = prony_format(DEPTH)

#: The burgers deck of depths: the Maxwell deck's, then the Kelvin
#: element's `kelvin_viscosity` and `kelvin_modulus_fraction`.
BURGERS = burgers_format(DEPTH)
