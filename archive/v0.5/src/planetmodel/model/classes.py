"""classes.py -- what a body guarantees: aspects, and the model classes.

A `ReferenceBody` is the general container: layers, their fields,
annotations.  A *model class* is a body that guarantees which fields
its layers hold and offers the methods that meaning licenses, and it
says so in the most general terms that still mean something -- density
and moduli; those plus moduli that depend on frequency; density and
viscosity.  Nothing is named for an application, for a law, or with a
word that means different things to different readers.

An **aspect** declares the fields it requires, by name, and the stable
property it exposes them under; a model class combines aspects.
Validation is *per layer*: a class guarantees its required fields on
every layer that holds any of them, a layer holding none (an empty
shell awaiting a consumer's material, a vacuum) passes, and at least
one layer must hold them all -- a guarantee of nothing is refused.

Every model class accepts the generic `ReferenceBody` constructor and
re-validates, so surgery preserves the class: a truncated
`ViscoelasticModel` is still one.  Each is registered under the kind
"model_class" so a file can name it and a reader rebuild it.

The classes check; they do not complete.  A `ViscoelasticModel` whose
layer holds static moduli alone is an elastic layer, and the class's
`viscoelastic_moduli` view lifts those moduli when it is asked for,
storing nothing: which law built a layer's `viscoelastic_moduli`, or
whether one did, is the builder's business and survives only as the
record on the field.
"""
from __future__ import annotations

from ..registry import register
from .body import ReferenceBody
from .fields.frequency import at_frequency, lifted_to_frequency
from .fields.layerwise import assemble

__all__ = ["ModelBase", "HasDensity", "HasModuli", "HasViscosity",
           "ElasticModel", "ViscoelasticModel", "ViscousModel"]


# ---------------------------------------------------------------------------
# aspects
# ---------------------------------------------------------------------------

class HasDensity:
    """The layer holds `rho`."""

    REQUIRES = ("rho",)

    @property
    def rho(self):
        """The density view, on the layers that hold it."""
        return self["rho"]


class HasModuli:
    """The layer holds `elastic_moduli`, the second elasticity tensor."""

    REQUIRES = ("elastic_moduli",)

    @property
    def elastic_moduli(self):
        """The static moduli: the `elastic_moduli` view, an ElasticField."""
        return self["elastic_moduli"]

    @property
    def symmetry(self):
        """The symmetry class of the moduli."""
        return self.elastic_moduli.symmetry


class HasViscosity:
    """The layer holds `viscosity`."""

    REQUIRES = ("viscosity",)

    @property
    def viscosity(self):
        """The viscosity view, on the layers that hold it."""
        return self["viscosity"]


# ---------------------------------------------------------------------------
# the base, and the three classes
# ---------------------------------------------------------------------------

class ModelBase(ReferenceBody):
    """A ReferenceBody that guarantees its aspects' fields, layer by layer."""

    #: The aspects this class combines, in order.
    ASPECTS: tuple[type, ...] = ()

    def __init__(self, layers, **kw) -> None:
        super().__init__(layers, **kw)
        self.validate()

    @classmethod
    def required_fields(cls) -> tuple[str, ...]:
        """Every field name the class's aspects require."""
        seen: dict[str, None] = {}
        for aspect in cls.ASPECTS:
            for name in aspect.REQUIRES:
                seen.setdefault(name, None)
        return tuple(seen)

    def validate(self) -> None:
        """Every layer holding any required field holds them all."""
        required = self.required_fields()
        complete = 0
        for lay in self.layers:
            if lay.is_vacuum:
                continue
            held = [n for n in required if n in lay.fields]
            if not held:
                continue
            missing = [n for n in required if n not in lay.fields]
            if missing:
                raise ValueError(
                    f"{type(self).__name__} guarantees {required} on every "
                    f"layer that holds any of them; layer {lay.index} holds "
                    f"{held} but not {missing}")
            complete += 1
        if complete == 0:
            raise ValueError(
                f"{type(self).__name__} guarantees {required}, and no layer "
                "holds them: a model class that guarantees nothing is a "
                "plain ReferenceBody")

    def _after_change(self) -> None:
        self.validate()

    @property
    def guaranteed_layers(self) -> tuple[int, ...]:
        """The layers holding every required field."""
        required = self.required_fields()
        return tuple(i for i, lay in enumerate(self.layers)
                     if all(n in lay.fields for n in required))

    def __repr__(self) -> str:
        nm = self.meta.get("name")
        head = f"{type(self).__name__}({nm!r}, " if nm else f"{type(self).__name__}("
        return head + (f"{self.skeleton.nlayers} layers, guarantees "
                       f"{list(self.required_fields())} on layers "
                       f"{list(self.guaranteed_layers)})")


@register("model_class", "ElasticModel")
class ElasticModel(HasDensity, HasModuli, ModelBase):
    """Density and static moduli: what a static elastic calculation needs."""

    ASPECTS = (HasDensity, HasModuli)


@register("model_class", "ViscoelasticModel")
class ViscoelasticModel(HasDensity, HasModuli, ModelBase):
    """Density, static moduli, and moduli that depend on frequency.

    A layer holding `elastic_moduli` may also hold `viscoelastic_moduli`,
    a frequency-dependent rank-4 field built from them by a law; a
    layer holding the static moduli alone is elastic, a contribution
    that does not depend on frequency.  The `viscoelastic_moduli` view
    is one field across the layers with moduli: each layer's own where
    it has one, its `elastic_moduli` lifted where it has not.  Nothing
    is stored for the lift.
    """

    ASPECTS = (HasDensity, HasModuli)
    #: The name of the frequency-dependent moduli a layer may hold.
    VISCOELASTIC = "viscoelastic_moduli"

    def validate(self) -> None:
        super().validate()
        for lay in self.layers:
            if self.VISCOELASTIC not in lay.fields:
                continue
            f = lay[self.VISCOELASTIC]
            if getattr(f, "kind", "static") != "frequency":
                raise ValueError(
                    f"{type(self).__name__}: layer {lay.index}'s "
                    f"{self.VISCOELASTIC!r} is {getattr(f, 'kind', 'static')}, "
                    "not a frequency-dependent field")
            if f.character.rank != 4:
                raise ValueError(
                    f"{type(self).__name__}: layer {lay.index}'s "
                    f"{self.VISCOELASTIC!r} has character {f.character}, not a "
                    "rank-4 elastic tensor")
            if "elastic_moduli" not in lay.fields:
                raise ValueError(
                    f"{type(self).__name__}: layer {lay.index} holds "
                    f"{self.VISCOELASTIC!r} without the static "
                    "'elastic_moduli' it departs from")

    def _after_change(self) -> None:
        self._views.pop(self.VISCOELASTIC, None)
        super()._after_change()

    @property
    def viscoelastic_moduli(self):
        """The frequency-dependent moduli: one view across the layers with moduli.

        A layer's own `viscoelastic_moduli` where it holds one, its
        `elastic_moduli` lifted where it does not.  Built on demand and
        cached until a field changes.
        """
        try:
            return self._views[self.VISCOELASTIC]
        except KeyError:
            pass
        pieces = []
        for lay in self.layers:
            if self.VISCOELASTIC in lay.fields:
                pieces.append(lay[self.VISCOELASTIC])
            elif "elastic_moduli" in lay.fields:
                pieces.append(lifted_to_frequency(lay["elastic_moduli"],
                                                  name=self.VISCOELASTIC))
        view = assemble(self.skeleton, pieces, name=self.VISCOELASTIC)
        self._views[self.VISCOELASTIC] = view
        return view

    def __getitem__(self, name: str):
        """The stitched view; `viscoelastic_moduli` includes the lifted layers."""
        if name == self.VISCOELASTIC:
            return self.viscoelastic_moduli
        return super().__getitem__(name)

    def moduli_at(self, omega, *, part: str = "complex"):
        """The moduli at one omega, as a static ELASTIC field.

        Complex by default, the complex tensor being the object; `part`
        "real" or "imag" for one part as float64.
        """
        return at_frequency(self.viscoelastic_moduli, omega, part=part)


@register("model_class", "ViscousModel")
class ViscousModel(HasDensity, HasViscosity, ModelBase):
    """Density and viscosity: what a convection calculation needs."""

    ASPECTS = (HasDensity, HasViscosity)
