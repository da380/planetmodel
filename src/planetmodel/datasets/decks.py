"""Radial models published as decks, read into models.

A radial reference (STW105, ak135-F, iasp91, the Moon's models) is a
table in some format of `planetmodel.deck`: a mineos deck, TauP's named
discontinuities, its tabled velocities.  `DeckReader` is the reader of
such a data set: the table read by its format and made a model by
`deck.model_from_deck`, its columns joined layer by layer by the
interpolation the source intends (cubic splines for a mineos deck,
straight lines for TauP's tables, which TauP itself interpolates
linearly), with what the entry says of it: the behaviours the model
has; how its layers and interfaces are named, by a rule that reads the
file's header (`deck.mineos_names`, `deck.nd_names`) or by the names
themselves; and the rule that reads its constants out of the header
(`deck.mineos_constants`).  The model is named as the data set is.  SI,
as decks are, or the caller's scales.
"""
from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from ..deck import Deck, DeckFormat, HeaderNames, model_from_deck, read_deck
from ..model import Model
from ..units import Scales
from ..vocabulary import Constant

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["DeckReader", "Names"]

#: A rule naming a deck's layers and interfaces, as `deck.mineos_names` does.
type Names = Callable[[Deck], HeaderNames]


@dataclass(frozen=True)
class DeckReader:
    """The reader of a data set that is one deck in `format`, the knots
    joined by `kind`: a model with `behaviours` (mixins of
    `planetmodel.behaviours`; a plain `Model` by default), its layers and
    interfaces named by `names`, a rule applied to the deck or the two
    sequences of names themselves (unnamed by default), and holding the
    constants the rule `constants` reads out of the deck (none by
    default)."""

    format: DeckFormat
    _: KW_ONLY
    behaviours: tuple[type, ...] = ()
    names: Names | HeaderNames | None = None
    constants: Callable[[Deck], Mapping[str, Constant]] | None = None
    kind: str = "cubic"

    def read(self, paths: tuple[Path, ...], dataset: Dataset, *,
             kind: str | None = None, scales: Scales | None = None) -> Model:
        """The data set's model: its one file read as a deck, joined by
        `kind` (the reader's own by default), in SI or `scales`."""
        if len(paths) != 1:
            raise ValueError(f"{dataset.name}: a deck is one file, given {len(paths)}")
        deck = read_deck(paths[0], self.format)
        named = self.names(deck) if callable(self.names) else self.names
        layer_names, interface_names = (None, None) if named is None else named
        model = model_from_deck(
            deck, behaviours=self.behaviours,
            kind=self.kind if kind is None else kind, layer_names=layer_names,
            interface_names=interface_names,
            constants=None if self.constants is None else self.constants(deck),
            name=dataset.name, citation=dataset.citation)
        return model if scales is None else model.converted(scales)
