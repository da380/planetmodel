"""Decks of depths: a table brought in as a flat model, and a flat model
written as one.

A flat model's deck is a table of knots whose first column is the
depth, increasing down the table from the free surface at 0, one column
per named field, a repeated depth a boundary between two layers; the
layers run from the surface down, as the model's do, and the last knot
is the base, a face of the box and no interface.  It is a `deck.Deck`
like any other, its `radius` holding the depths, read by
`planetmodel.read_deck`, made a model by `model_from_deck`, sampled
from a model by `sample_deck` and written by `write_deck`, all the
library's.

What differs from a table of radii is said by `DEPTH`: the columns
become fields of depth, each a `DepthField`, a message names a knot's
depth, and the model is a `cartesian.Model` over a lateral domain,
which a table does not hold and `model_from_deck` is given beside it.
The names given it name one interface per layer from the surface down,
the top first, where a table of radii has one more for a hollow body's
inner boundary.  A crust over a mantle, as a table of `rho`, `vp` and
`vs` under one line of the caller's own, and the model of it::

    depth rho vp vs
    0       2800  6400  3700
    30000   2800  6400  3700
    30000   3300  8000  4500
    200000  3300  8000  4500

    deck = read_deck(path, ["rho", "vp", "vs"], header_lines=1)
    model = model_from_deck(deck, coordinate=DEPTH, domain=box,
                            behaviours=[Elastic],
                            layer_names=["crust", "mantle"],
                            interface_names=["surface", "moho"])

::

  DEPTH        the depth from the free surface down
"""
from __future__ import annotations

from ..deck import DeckCoordinate
from ..geometry import Names
from ..skeleton import Skeleton
from .domains import Domain
from .geometry import Geometry
from .model import Model

__all__ = ["DEPTH"]


def _box(skeleton: Skeleton, domain: Domain | None, *, layer_names: Names,
         interface_names: Names) -> Geometry:
    """The geometry of a table of depths over the lateral `domain`, which
    the table does not hold."""
    if domain is None:
        raise ValueError("a table of depths holds no lateral extent: give the box or "
                         "the rectangle the model is built over, domain=")
    return Geometry(skeleton, domain, layer_names=layer_names,
                    interface_names=interface_names)


#: The depth from the free surface down: `depth` in a message, the knots of
#: a `cartesian.Model` over a lateral domain, the fields each a `DepthField`,
#: one interface per layer, the top first.
DEPTH = DeckCoordinate("depth", Model, _box)
