"""Model classes: what a body guarantees, layer by layer.

A ReferenceBody is the general container.  A model class is a body
that guarantees which fields its layers hold and exposes them under
stable names: ElasticModel (density and static moduli),
ViscoelasticModel (those, plus frequency-dependent moduli),
ViscousModel (density and viscosity).  Validation is per layer: every
layer that holds any of the required fields must hold them all, a
vacuum layer passes, and at least one layer must be complete.  Surgery
preserves the class.  In a ViscoelasticModel a layer holding only
static moduli is elastic: its `viscoelastic_moduli` is the static
tensor lifted at view time, and nothing is stored for it.

A class is a mixin of aspects, each declaring the fields it requires
and the property it exposes them under, so a new class is a few lines.
This script exercises the three shipped classes and defines one.
"""
from pathlib import Path

import numpy as np

from planetmodel import (DENSITY, Dimensions, ElasticModel, RadialField,
                         ReferenceBody, Skeleton, ViscoelasticModel, ViscousModel,
                         maxwell, read_isotropic_deck, register, testing)
from planetmodel.model.classes import HasDensity, ModelBase
from planetmodel.registry import lookup

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"

# -- a deck read as the class its fields warrant --------------------------------
deck = read_isotropic_deck(DATA / "prem.nocrust")
assert type(deck) is ElasticModel                     # Q columns, but no period given
assert ElasticModel.required_fields() == ("rho", "elastic_moduli")
assert deck.guaranteed_layers == tuple(range(deck.skeleton.nlayers))
assert deck.rho is deck["rho"] and deck.elastic_moduli is deck["elastic_moduli"]
assert deck.symmetry.name == "ISOTROPIC"
assert deck.layers[1].state == "fluid"                # classified on read
print(deck)

# Calibrated at a period, the same deck is viscoelastic under constant Q.
visco = read_isotropic_deck(DATA / "prem.nocrust", reference_period=1.0)
assert type(visco) is ViscoelasticModel
assert all("viscoelastic_moduli" in lay for lay in visco.layers)

# -- surgery preserves the class -------------------------------------------------
cut = deck.truncated(6.0e6)
assert type(cut) is ElasticModel
grown = deck.extended([6.6e6])                        # an empty shell: allowed
assert type(grown) is ElasticModel and grown.guaranteed_layers == deck.guaranteed_layers

# -- what validate() refuses -------------------------------------------------------
sk = Skeleton([0.0, 1.0, 2.0])
rho = RadialField(sk, [lambda r: 5.0e3 + 0.0 * r] * 2, name="rho",
                  character=DENSITY, dimensions=Dimensions.DENSITY)
density_only = ReferenceBody.from_fields(sk, {"rho": rho})
try:
    density_only.as_class(ElasticModel)
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("a density-only body is not an ElasticModel")

# A partial guarantee is refused too: moduli on one layer, density on both.
half = ReferenceBody(deck.layers).without_field("elastic_moduli").with_field(
    0, "elastic_moduli", deck.layers[0]["elastic_moduli"])
try:
    half.as_class(ElasticModel)
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("every layer holding any required field must hold all")

# -- ViscousModel ------------------------------------------------------------------
eta = RadialField(sk, [lambda r: 1.0e21 + 0.0 * r] * 2, name="viscosity",
                  dimensions=Dimensions.VISCOSITY)
viscous = ReferenceBody.from_fields(sk, {"rho": rho, "viscosity": eta}).as_class(
    ViscousModel)
assert type(viscous) is ViscousModel and viscous.viscosity is viscous["viscosity"]

# -- view-time lifting in a ViscoelasticModel ----------------------------------------
mantle = deck.layers[2]
law = maxwell(mantle["elastic_moduli"], eta.restricted(0).on_interval(*mantle.interval))
mixed = ReferenceBody(deck.layers).with_field(2, "viscoelastic_moduli", law).as_class(
    ViscoelasticModel)
assert "viscoelastic_moduli" in mixed.layers[2]        # the Maxwell mantle
assert "viscoelastic_moduli" not in mixed.layers[0]    # an elastic core: nothing stored
view = mixed.viscoelastic_moduli                        # ... but the view covers it
assert view.domain == mixed.elastic_moduli.domain
r = np.array([0.5e6])
assert np.allclose(view.evaluate(r, omega=1.0), mixed.elastic_moduli.evaluate(r))
assert "viscoelastic_moduli" not in mixed.layers[0]    # still nothing stored
frozen = mixed.moduli_at(1.0e-11)                       # a static complex tensor
assert frozen.evaluate(r).dtype == np.complex128

# -- a class of your own --------------------------------------------------------
class HasPorosity:
    """The layer holds `porosity`."""

    REQUIRES = ("porosity",)

    @property
    def porosity(self):
        return self["porosity"]


@register("model_class", "PorousModel")
class PorousModel(HasDensity, HasPorosity, ModelBase):
    """Density and porosity: what a fluid-flow calculation needs."""

    ASPECTS = (HasDensity, HasPorosity)


phi = RadialField(sk, [lambda r: 0.1 + 0.0 * r] * 2, name="porosity",
                  dimensions=Dimensions.DIMENSIONLESS)
porous = ReferenceBody.from_fields(sk, {"rho": rho, "porosity": phi}).as_class(
    PorousModel)
assert porous.porosity is porous["porosity"]
assert type(porous.refined([0.5])) is PorousModel
assert lookup("model_class", "PorousModel") is PorousModel   # a file can name it
testing.check_model(porous)
testing.check_model(mixed)

print("ok: the classes guarantee their fields per layer and survive surgery")
