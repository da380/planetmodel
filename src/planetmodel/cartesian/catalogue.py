"""Named flat model types: a layered isotropic model, a crust over a
mantle, and the models of the decks of depths.

Each is a class derived from `cartesian.Model` alone, with the
behaviours of `planetmodel.behaviours` that are pointwise mixed in:
`Elastic` attaches the five Love moduli beside the velocities a layer
is given (or the other way round), `ConstantQ` disperses them where a
layer holds quality factors, and `Viscoelastic` freezes the model at a
frequency where a layer holds a viscosity or Prony branches.  None is
`SelfGravitating`: gravity from the mass inside a radius is the
sphere's, and a flat model's gravity is prescribed
(`Model.with_uniform_gravity`).  Every model is in SI, converted to
other scales by `converted`, and built over a lateral `Box` or
`Rectangle` given to its constructor, a deck holding no lateral extent:
the box is as large as the problem it serves needs, which no default
could know.  Below its base a model is padded by `extended(depths,
fields="hold")`.

`LayeredIsotropicElastic` is the flat counterpart of the spherical one:
`rho`, `vp` and `vs` per layer, a constant or the coefficients of a
polynomial in depth / scale (`catalogue.isotropic_layers`), a fluid
layer one with vs = 0; `homogeneous` is a single layer from the surface
down.

`CrustOverMantle` is the reference flat model of the tests and
examples: round values typical of a continental crust over the
uppermost mantle, in SI::

    layer                 depth (km)   rho (kg/m^3)   vp (m/s)   vs (m/s)
    crust                 0 - 30       2800           6400       3700
    lithospheric_mantle   30 - 100     3300           8000       4500
    asthenosphere         100 - base   3300           8000       4500

with the interfaces `surface`, `moho` and `lab` (the base of the
lithosphere) and the base at 200 km unless `base` says otherwise; with
`viscosity` (Pa s) the asthenosphere is a Maxwell body, its elastic
values unrelaxed.  It holds no quality factors; one put on a layer
(`with_field`) disperses its moduli by `moduli_at`.

`ElasticDeckModel`, `MaxwellDeckModel`, `PronyDeckModel` and
`BurgersDeckModel` are the models of the decks of depths
(`cartesian.ELASTIC`, `MAXWELL`, `PRONY`, `BURGERS`), built as their
spherical counterparts are from the same rules (`deck.deck_from`,
`deck.prony_layers`, `conversion.prony.burgers_layers`), the deck kept
so that `to_deck` writes the model back, character for character on
its own knots.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

from ..behaviours import ConstantQ, Elastic, Viscoelastic
from ..catalogue import LayerValue, isotropic_layers
from ..deck import (BURGERS_KINDS, MAXWELL_KINDS, PRONY_KINDS, Deck, DeckFormat,
                    Tabulated, deck_from, deck_layers, prony_layers)
from ..fields import Field
from ..layerfunction import constant_layer
from ..skeleton import Skeleton
from ..units import Scales
from .deck import BURGERS, DEPTH, ELASTIC, MAXWELL, PRONY, depth_names
from .domains import Domain
from .geometry import Geometry
from .model import Model

__all__ = ["LayeredIsotropicElastic", "CrustOverMantle", "ElasticDeckModel",
           "MaxwellDeckModel", "PronyDeckModel", "BurgersDeckModel",
           "CRUST_OVER_MANTLE"]

#: `CrustOverMantle`'s layers from the surface down: name, top depth (m),
#: and rho (kg/m^3), vp and vs (m/s).
CRUST_OVER_MANTLE = (("crust", 0.0, 2800.0, 6400.0, 3700.0),
                     ("lithospheric_mantle", 30e3, 3300.0, 8000.0, 4500.0),
                     ("asthenosphere", 100e3, 3300.0, 8000.0, 4500.0))

#: `CrustOverMantle`'s interfaces from the surface down.
_CRUST_OVER_MANTLE_INTERFACES = ("surface", "moho", "lab")


class LayeredIsotropicElastic(Elastic, Model):
    """An isotropic flat model of exact polynomial layers between `depths`
    over the lateral `domain`.

    `depths` are the skeleton's, from the surface down (a first depth
    below zero is refused, one above it gives a body whose top lies
    below the free surface); `rho`, `vp` and `vs` give one entry per
    layer, a number for a constant or a sequence of ascending polynomial
    coefficients in depth / scale, and a layer with vs = 0 throughout is
    fluid.  `name` is the model's.
    """

    def __init__(self, depths: Sequence[float], domain: Domain, *,
                 rho: Sequence[LayerValue], vp: Sequence[LayerValue],
                 vs: Sequence[LayerValue], scale: float = 1.0,
                 layer_names: Sequence[str | None] | None = None,
                 interface_names: Sequence[str | None] | None = None,
                 scales: Scales = Scales.SI, name: str | None = None) -> None:
        geometry = Geometry(Skeleton(depths), domain, layer_names=layer_names,
                            interface_names=interface_names)
        layers = isotropic_layers(geometry, rho=rho, vp=vp, vs=vs, scale=scale)
        super().__init__(geometry, layers, scales=scales, name=name)

    @classmethod
    def homogeneous(cls, depth: float, domain: Domain, *, rho: float, vp: float,
                    vs: float, name: str | None = None,
                    scales: Scales = Scales.SI) -> "LayeredIsotropicElastic":
        """A uniform isotropic slab from the surface to `depth` over
        `domain`: constant rho, vp and vs; `name` names the model and its
        one layer."""
        return cls([0.0, depth], domain, rho=[rho], vp=[vp], vs=[vs],
                   layer_names=None if name is None else [name], scales=scales,
                   name=name)


class CrustOverMantle(Elastic, ConstantQ, Viscoelastic, Model):
    """A crust over the uppermost mantle, over the lateral `domain`, in SI;
    see the module docstring for its values.

    `base` is the depth of the base in metres, below the base of the
    lithosphere at 100 km; `viscosity`, in Pa s, makes the asthenosphere
    a Maxwell body of that constant viscosity, None leaving the model
    elastic.
    """

    def __init__(self, domain: Domain, *, base: float = 200e3,
                 viscosity: float | None = None) -> None:
        tops = [top for _, top, *_ in CRUST_OVER_MANTLE]
        if not float(base) > tops[-1]:
            raise ValueError(f"the base lies below the base of the lithosphere at "
                             f"{tops[-1]:g} m; got {base:g}")
        if viscosity is not None and not float(viscosity) > 0.0:
            raise ValueError(f"the viscosity is positive, got {viscosity!r}")
        geometry = Geometry(Skeleton(tops + [float(base)]), domain,
                            layer_names=[row[0] for row in CRUST_OVER_MANTLE],
                            interface_names=_CRUST_OVER_MANTLE_INTERFACES)
        layers = isotropic_layers(geometry,
                                  rho=[row[2] for row in CRUST_OVER_MANTLE],
                                  vp=[row[3] for row in CRUST_OVER_MANTLE],
                                  vs=[row[4] for row in CRUST_OVER_MANTLE])
        name = "crust over mantle"
        if viscosity is not None:
            iv = geometry.skeleton.interval(-1)
            layers[-1]["viscosity"] = Geometry.radial_field_type(
                iv, constant_layer(iv, float(viscosity)), name="viscosity")
            name += " (Maxwell asthenosphere)"
        super().__init__(geometry, layers, scales=Scales.SI, name=name)


def _read(model: Tabulated, source: str | os.PathLike[str] | Deck, domain: Domain,
          format: DeckFormat, *, kind: str, kinds: Mapping[str, str] | None
          ) -> tuple[Geometry, list[dict[str, Field]], Deck]:
    """A deck of depths in `format` as a geometry over `domain` and the base
    fields of its layers, the deck kept on `model`, the one being built."""
    deck = deck_from(source, format)
    skeleton, layers = deck_layers(deck, kind=kind, kinds=kinds, coordinate=DEPTH)
    layer_names, interface_names = depth_names(deck)
    geometry = Geometry(skeleton, domain, layer_names=layer_names,
                        interface_names=interface_names)
    model.keep_deck(deck)
    return geometry, layers, deck


def _title(deck: Deck) -> str | None:
    """A deck's title as a model's name, None for none."""
    return str(deck.header.get("name", "")) or None


class ElasticDeckModel(Elastic, ConstantQ, Viscoelastic, Tabulated, Model):
    """The model of an elastic deck of depths (`cartesian.ELASTIC`) over the
    lateral `domain`: `depth rho vpv vsv qkappa qmu vph vsh eta` in SI, or
    `depth rho vp vs qkappa qmu`, under a title, the column line and the
    naming lines, NaN where a layer lacks a field.

    The columns are interpolated layer by layer by `kind` (see
    `planetmodel.deck.KINDS`); a quality factor NaN throughout a layer is
    absent from it, and `moduli_at` disperses the moduli of a layer
    holding one about a period of one second.  The naming lines name the
    layers and interfaces (`depth_names`), the title the model.  The
    deck is kept as `deck`, its knots per layer as `knots`, its header
    as `header` and its columns as `columns`, so `to_deck` writes the
    model back out.  The model is in SI, as the deck is.
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, domain: Domain, *,
                 kind: str = "cubic") -> None:
        geometry, layers, deck = _read(self, source, domain, ELASTIC, kind=kind,
                                       kinds=None)
        super().__init__(geometry, layers, name=_title(deck))


class MaxwellDeckModel(Elastic, Viscoelastic, Tabulated, Model):
    """The model of a Maxwell deck of depths (`cartesian.MAXWELL`) over the
    lateral `domain`: `depth rho vpv vsv vph vsh eta viscosity` in SI
    under a title, the column line and the naming lines, NaN where a
    layer lacks a field.

    As the spherical `MaxwellDeckModel`: the columns are interpolated by
    `kind`, and by `kinds` where it names a column, the viscosity in its
    logarithm by straight lines by default (`MAXWELL_KINDS`); the elastic
    values are the Maxwell body's unrelaxed ones, which `frozen(omega)`
    relaxes with tau = viscosity / mu.  The deck is kept and `to_deck`
    writes the model back out.
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, domain: Domain, *,
                 kind: str = "cubic", kinds: Mapping[str, str] = MAXWELL_KINDS) -> None:
        geometry, layers, deck = _read(self, source, domain, MAXWELL, kind=kind,
                                       kinds=kinds)
        super().__init__(geometry, layers, name=_title(deck))


class PronyDeckModel(Elastic, Viscoelastic, Tabulated, Model):
    """The model of a prony deck of depths (`cartesian.PRONY`) over the
    lateral `domain`: the Maxwell deck's columns with the viscosity
    replaced by the pairs `relaxation_strength_k relaxation_time_k`.

    As the spherical `PronyDeckModel`: by default the strengths are joined
    by straight lines and the times in their logarithm (`PRONY_KINDS`),
    each layer holds the branches it uses (`deck.prony_layers`), a deck
    given rather than read is completed and held to the format, and
    `to_deck` writes the model back out as a prony deck of depths on its
    knots (`conversion.prony.prony_deck`).
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, domain: Domain, *,
                 kind: str = "cubic", kinds: Mapping[str, str] = PRONY_KINDS) -> None:
        geometry, layers, deck = _read(self, source, domain, PRONY, kind=kind,
                                       kinds=kinds)
        super().__init__(geometry, prony_layers(deck, layers), name=_title(deck))

    def to_deck(self, *, columns: Sequence[str] | None = None) -> Deck:
        """The model as a prony deck of depths on its knots, in SI, by
        `conversion.prony.prony_deck`, under the header it was read with
        (`Tabulated.restated`); with `columns`, those fields as
        `Tabulated.to_deck` samples them."""
        if columns is not None:
            return super().to_deck(columns=columns)
        from ..conversion.prony import prony_deck
        return self.restated(prony_deck(self, knots=self.knots, format=PRONY))


class BurgersDeckModel(Elastic, Viscoelastic, Tabulated, Model):
    """The model of a burgers deck of depths (`cartesian.BURGERS`) over the
    lateral `domain`: the Maxwell deck's columns, then the Kelvin
    element's `kelvin_viscosity` and `kelvin_modulus_fraction`.

    As the spherical `BurgersDeckModel`: by default the two viscosities
    are joined in their logarithm and the modulus fraction by straight
    lines (`BURGERS_KINDS`), a Burgers layer holds its two Prony branches
    (`conversion.prony.burgers_layers`) and a Maxwell layer its
    viscosity, and `to_deck` writes the model back out as a burgers deck
    of depths on its knots (`conversion.prony.burgers_deck`).
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, domain: Domain, *,
                 kind: str = "cubic", kinds: Mapping[str, str] = BURGERS_KINDS) -> None:
        from ..conversion.prony import burgers_layers
        geometry, layers, deck = _read(self, source, domain, BURGERS, kind=kind,
                                       kinds=kinds)
        super().__init__(geometry, burgers_layers(layers), name=_title(deck))

    def to_deck(self, *, columns: Sequence[str] | None = None) -> Deck:
        """The model as a burgers deck of depths on its knots, in SI, by
        `conversion.prony.burgers_deck`, under the header it was read with
        (`Tabulated.restated`); with `columns`, those fields as
        `Tabulated.to_deck` samples them."""
        if columns is not None:
            return super().to_deck(columns=columns)
        from ..conversion.prony import burgers_deck
        return self.restated(burgers_deck(self, knots=self.knots, format=BURGERS))
