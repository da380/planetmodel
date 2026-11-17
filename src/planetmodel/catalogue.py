"""Named model types: PREM from its polynomials, and a simple layered
model for use and for testing.

Each is a class derived from `Model` alone, with the elastic, gravity
and rheology behaviours of `planetmodel.behaviours` mixed in: every
layer holds the five Love moduli A, C, F, L, N as exact fields beside
the velocities it was built from, `PREM().moduli_at("lower_mantle",
omega)` gives them dispersed by the constant-Q band, `PREM().gravity(r)`
and `PREM().frozen(omega)` are the free functions of the library as
methods.  Their constructors build the fields; every copy, surgery,
conversion and freezing keeps the class.

`PREM` is the Preliminary Reference Earth Model of Dziewonski and
Anderson (1981, Phys. Earth Planet. Inter. 25, 297-356), built from the
low-order polynomials in x = r / a of their Table I with a = 6371 km,
so every field is an exact piecewise polynomial and evaluation,
derivatives, moduli and integrals carry no interpolation error.  The
table below holds the coefficients in the paper's units (g/cm^3, km/s,
dimensionless) and the model is in SI.  Regions with one P and one S
velocity are isotropic and get vph = vpv, vsh = vsv and eta = 1 as
exact constants; the fluid outer core and ocean hold no qmu; the
elastic values are those at a reference period of 1 s.

`LayeredIsotropicElastic` builds an isotropic model of exact layers
from a few numbers per layer, a constant or the coefficients of a
polynomial in r / scale, a fluid layer being one with vs = 0
(`isotropic_layers`, which builds them for any geometry); its
`homogeneous` classmethod is the one-layer case.

`MineosModel` is the model of a mineos deck, PREM's own tabulation
(`examples/data/prem.200`) or any other: the table's columns become the
base fields by interpolation (`planetmodel.deck`), the header names
the core layers and sets the reference period of the constant-Q band,
and the mixins add the rest.  `MaxwellDeckModel` is the model of a
Maxwell deck, a radial Earth for glacial isostatic adjustment: density
and the unrelaxed elastic velocities beside a viscosity, a Maxwell body
in shear, with no quality factors, since a constant-Q dispersion about
a seismic reference period would otherwise reach glacial periods.
`PronyDeckModel` is the model of a prony deck, the same with Prony
branches in place of the viscosity, a generalised Maxwell body, and
`BurgersDeckModel` that of a burgers deck, each layer a Maxwell or a
Burgers body, the latter held as its two branches.  What a deck's
model holds of its columns is said once, beside the format
(`deck.deck_from`, `deck.prony_layers`, `conversion.prony.burgers_layers`),
and the flat catalogue (`planetmodel.cartesian`) builds its own deck
models from the same.
"""
from __future__ import annotations

import os
from collections.abc import Mapping, Sequence

import numpy as np

from .behaviours import ConstantQ, Elastic, SelfGravitating, Viscoelastic
from .character import DENSITY, SCALAR
from .deck import (BURGERS, BURGERS_KINDS, MAXWELL, MAXWELL_KINDS, MINEOS, PRONY,
                   PRONY_KINDS, Deck, Tabulated, deck_from, deck_layers, maxwell_names,
                   mineos_names, prony_layers)
from .fields import Field, RadialField
from .geometry import Geometry, GeometryBase
from .layerfunction import constant_layer, polynomial_layer
from .model import Model
from .skeleton import Skeleton
from .units import FREQUENCY, Scales
from .vocabulary import Constant

__all__ = ["PREM", "LayeredIsotropicElastic", "MineosModel", "MaxwellDeckModel",
           "PronyDeckModel", "BurgersDeckModel", "PREM_RADIUS", "isotropic_layers",
           "LayerValue"]

#: One layer's value of an isotropic model: a constant, or the ascending
#: coefficients of a polynomial in the layered coordinate over a scale.
type LayerValue = float | Sequence[float]

#: The normalisation radius a of PREM's polynomials, metres.
PREM_RADIUS = 6371e3

_PREM_CITATION = ("Dziewonski, A. M. and D. L. Anderson (1981).  Preliminary "
                  "reference Earth model.  Phys. Earth Planet. Inter. 25, 297-356, "
                  "doi:10.1016/0031-9201(81)90046-7")

#: Table I of Dziewonski and Anderson (1981), centre outward: the radii
#: of each region in metres and the polynomial coefficients of each
#: column in the paper's units, ascending in x = r / a.
_PREM_REGIONS = (
    ("inner_core", 0.0, 1221500.0, {
        "rho": (13.0885, 0.0, -8.8381),
        "vpv": (11.2622, 0.0, -6.3640),
        "vsv": (3.6678, 0.0, -4.4475),
        "qkappa": (1327.7,), "qmu": (84.6,),
    }),
    ("outer_core", 1221500.0, 3480000.0, {
        "rho": (12.5815, -1.2638, -3.6426, -5.5281),
        "vpv": (11.0487, -4.0362, 4.8023, -13.5732),
        "vsv": (0.0,),
        "qkappa": (57823.0,),
    }),
    ("lowermost_mantle", 3480000.0, 3630000.0, {
        "rho": (7.9565, -6.4761, 5.5283, -3.0807),
        "vpv": (15.3891, -5.3181, 5.5242, -2.5514),
        "vsv": (6.9254, 1.4672, -2.0834, 0.9783),
        "qkappa": (57823.0,), "qmu": (312.0,),
    }),
    ("lower_mantle", 3630000.0, 5600000.0, {
        "rho": (7.9565, -6.4761, 5.5283, -3.0807),
        "vpv": (24.9520, -40.4673, 51.4832, -26.6419),
        "vsv": (11.1671, -13.7818, 17.4575, -9.2777),
        "qkappa": (57823.0,), "qmu": (312.0,),
    }),
    ("upper_lower_mantle", 5600000.0, 5701000.0, {
        "rho": (7.9565, -6.4761, 5.5283, -3.0807),
        "vpv": (29.2766, -23.6027, 5.5242, -2.5514),
        "vsv": (22.3459, -17.2473, -2.0834, 0.9783),
        "qkappa": (57823.0,), "qmu": (312.0,),
    }),
    ("transition_zone_lower", 5701000.0, 5771000.0, {
        "rho": (5.3197, -1.4836),
        "vpv": (19.0957, -9.8672),
        "vsv": (9.9839, -4.9324),
        "qkappa": (57823.0,), "qmu": (143.0,),
    }),
    ("transition_zone_middle", 5771000.0, 5971000.0, {
        "rho": (11.2494, -8.0298),
        "vpv": (39.7027, -32.6166),
        "vsv": (22.3512, -18.5856),
        "qkappa": (57823.0,), "qmu": (143.0,),
    }),
    ("transition_zone_upper", 5971000.0, 6151000.0, {
        "rho": (7.1089, -3.8045),
        "vpv": (20.3926, -12.2569),
        "vsv": (8.9496, -4.4597),
        "qkappa": (57823.0,), "qmu": (143.0,),
    }),
    ("low_velocity_zone", 6151000.0, 6291000.0, {
        "rho": (2.6910, 0.6924),
        "vpv": (0.8317, 7.2180),
        "vsv": (5.8582, -1.4678),
        "vph": (3.5908, 4.6172),
        "vsh": (-1.0839, 5.7176),
        "eta": (3.3687, -2.4778),
        "qkappa": (57823.0,), "qmu": (80.0,),
    }),
    ("lid", 6291000.0, 6346600.0, {
        "rho": (2.6910, 0.6924),
        "vpv": (0.8317, 7.2180),
        "vsv": (5.8582, -1.4678),
        "vph": (3.5908, 4.6172),
        "vsh": (-1.0839, 5.7176),
        "eta": (3.3687, -2.4778),
        "qkappa": (57823.0,), "qmu": (600.0,),
    }),
    ("lower_crust", 6346600.0, 6356000.0, {
        "rho": (2.900,), "vpv": (6.800,), "vsv": (3.900,),
        "qkappa": (57823.0,), "qmu": (600.0,),
    }),
    ("upper_crust", 6356000.0, 6368000.0, {
        "rho": (2.600,), "vpv": (5.800,), "vsv": (3.200,),
        "qkappa": (57823.0,), "qmu": (600.0,),
    }),
    ("ocean", 6368000.0, 6371000.0, {
        "rho": (1.020,), "vpv": (1.450,), "vsv": (0.0,),
        "qkappa": (57823.0,),
    }),
)

#: The interfaces of the full model, centre outward, named by the
#: usual abbreviation or by their depth in kilometres.
_PREM_INTERFACES = ("icb", "cmb", "d2741", "d771", "d670", "d600", "d400",
                    "d220", "d80", "moho", "d15", "ocean_floor", "surface")

#: SI units per unit of the paper's.
_PREM_SI = {"rho": 1e3, "vpv": 1e3, "vsv": 1e3, "vph": 1e3, "vsh": 1e3,
            "eta": 1.0, "qkappa": 1.0, "qmu": 1.0}


def _prem_coefficients(name: str, region: Mapping[str, tuple[float, ...]]
                       ) -> tuple[float, ...] | None:
    """A column's coefficients in one region, the isotropic defaults filled
    in, or None where the region has no such field."""
    if name in region:
        return region[name]
    if name == "vph":
        return region["vpv"]
    if name == "vsh":
        return region["vsv"]
    if name == "eta":
        return (1.0,)
    return None


class PREM(Elastic, ConstantQ, SelfGravitating, Viscoelastic, Model):
    """PREM as an exact polynomial model in SI; see the module docstring.

    `ocean=False` drops the 3 km water layer, so the model ends at the
    top of the upper crust and that interface is the surface; the
    polynomials keep their normalisation radius.
    """

    def __init__(self, *, ocean: bool = True) -> None:
        regions = _PREM_REGIONS[:-1] if not ocean else _PREM_REGIONS
        boundaries = [lo for _, lo, _, _ in regions] + [regions[-1][2]]
        names = [name for name, _, _, _ in regions]
        faces = list(_PREM_INTERFACES[:len(regions)])
        faces[-1] = "surface"
        geometry = Geometry(Skeleton(boundaries), layer_names=names,
                            interface_names=faces)
        layers = []
        for _, lo, hi, table in regions:
            fields = {}
            for name, si in _PREM_SI.items():
                coeffs = _prem_coefficients(name, table)
                if coeffs is None:
                    continue
                fields[name] = RadialField(
                    (lo, hi),
                    polynomial_layer((lo, hi), np.asarray(coeffs) * si,
                                     scale=PREM_RADIUS),
                    character=DENSITY if name == "rho" else SCALAR,
                    name=name)
            layers.append(fields)
        super().__init__(geometry, layers, scales=Scales.SI,
                         name="PREM" if ocean else "PREM (no ocean)",
                         citation=(_PREM_CITATION,))


def isotropic_layers(geometry: GeometryBase, *, rho: Sequence[LayerValue],
                     vp: Sequence[LayerValue], vs: Sequence[LayerValue],
                     scale: float = 1.0) -> list[dict[str, Field]]:
    """`rho`, `vp` and `vs` on every layer of `geometry`, one entry each
    per layer: a number for a constant, or a sequence of ascending
    polynomial coefficients in the layered coordinate over `scale`, each
    an exact field of the geometry's `radial_field_type`.  A layer with
    vs = 0 throughout is fluid."""
    sk = geometry.skeleton
    values = {"rho": rho, "vp": vp, "vs": vs}
    for key, seq in values.items():
        if len(seq) != sk.nlayers:
            raise ValueError(f"{key} needs {sk.nlayers} values, got {len(seq)}")
    make = geometry.radial_field_type
    layers = []
    for i in range(sk.nlayers):
        iv = sk.interval(i)
        fields: dict[str, Field] = {}
        for key, character in (("rho", DENSITY), ("vp", SCALAR), ("vs", SCALAR)):
            value = values[key][i]
            function = (constant_layer(iv, float(value)) if np.isscalar(value)
                        else polynomial_layer(iv, value, scale=scale))
            fields[key] = make(iv, function, character=character, name=key)
        layers.append(fields)
    return layers


class LayeredIsotropicElastic(Elastic, SelfGravitating, Model):
    """An isotropic model of exact polynomial layers between `boundaries`.

    `boundaries` are the skeleton's, centre outward (an inner radius
    above zero gives a hollow model); `rho`, `vp` and `vs` give one
    entry per layer, a number for a constant or a sequence of ascending
    polynomial coefficients in r / scale, and a layer with vs = 0
    throughout is fluid.  `name` is the model's.
    """

    def __init__(self, boundaries: Sequence[float], *,
                 rho: Sequence[LayerValue], vp: Sequence[LayerValue],
                 vs: Sequence[LayerValue], scale: float = 1.0,
                 layer_names: Sequence[str | None] | None = None,
                 interface_names: Sequence[str | None] | None = None,
                 scales: Scales = Scales.SI, name: str | None = None) -> None:
        geometry = Geometry(Skeleton(boundaries), layer_names=layer_names,
                            interface_names=interface_names)
        layers = isotropic_layers(geometry, rho=rho, vp=vp, vs=vs, scale=scale)
        super().__init__(geometry, layers, scales=scales, name=name)

    @classmethod
    def homogeneous(cls, radius: float, *, rho: float, vp: float, vs: float,
                    name: str | None = None,
                    scales: Scales = Scales.SI) -> "LayeredIsotropicElastic":
        """A uniform isotropic sphere of `radius`: constant rho, vp and vs;
        `name` names the model and its one layer."""
        return cls([0.0, radius], rho=[rho], vp=[vp], vs=[vs],
                   layer_names=None if name is None else [name], scales=scales,
                   name=name)


class MineosModel(Elastic, ConstantQ, SelfGravitating, Viscoelastic, Tabulated, Model):
    """The model of a mineos deck: `r rho vpv vsv qkappa qmu vph vsh eta`
    in SI, or the six-column isotropic form, under three header lines.

    The columns are interpolated layer by layer by `kind` (see
    `planetmodel.deck.KINDS`); a `qmu` of zero on a layer marks no shear
    loss and is kept as read.  The header's `nic` and `noc` name the
    inner and outer core and their boundaries, its `tref`, in seconds,
    becomes the constant `omega_ref` that `moduli_at` and `frozen`
    disperse about, and its title is the model's name.  The deck is
    kept as `deck`, its knots per layer as `knots`, its header as
    `header` and its columns as `columns`, so `to_deck` writes the model
    back out.  The model is in SI, as the deck is; `converted` gives it
    in other scales.
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, *,
                 kind: str = "cubic") -> None:
        deck = deck_from(source, MINEOS)
        skeleton, layers = deck_layers(deck, kind=kind)
        layer_names, interface_names = mineos_names(deck)
        geometry = Geometry(skeleton, layer_names=layer_names,
                            interface_names=interface_names)
        constants = {}
        tref = float(deck.header.get("tref", 0.0))
        if tref > 0.0:
            constants["omega_ref"] = Constant(
                2.0 * np.pi / tref, FREQUENCY,
                meaning="reference angular frequency of the deck's elastic values")
        self.keep_deck(deck)
        super().__init__(geometry, layers, constants=constants,
                         name=str(deck.header.get("name", "")) or None)


class MaxwellDeckModel(Elastic, SelfGravitating, Viscoelastic, Tabulated, Model):
    """The model of a Maxwell deck: `r rho vpv vsv vph vsh eta viscosity`
    in SI under a title and the line naming the columns, NaN where a
    layer lacks a field.

    The columns are interpolated layer by layer by `kind`, and by the
    kind `kinds` gives a column where it gives one (see
    `planetmodel.deck.KINDS` and `LOG_KINDS`); by default the viscosity
    is joined in its logarithm by straight lines, `MAXWELL_KINDS`,
    positive and geometric between knots, and `kinds={"viscosity":
    "linear"}` joins its values by straight lines instead.  The elastic
    values are the unrelaxed moduli of the Maxwell body, so
    `frozen(omega)` relaxes each layer holding a viscosity with tau =
    viscosity / mu, and the deck holds no quality factors.  The
    `# layers` and `# interfaces` lines of the header name the layers
    and interfaces (`maxwell_names`); its title is the model's name.
    The deck is kept as `deck`, its knots per layer as `knots`, its
    header as `header` and its columns as `columns`, so `to_deck` writes
    the model back out.  The model is in SI, as the deck is; `converted`
    gives it in other scales.
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, *, kind: str = "cubic",
                 kinds: Mapping[str, str] = MAXWELL_KINDS) -> None:
        deck = deck_from(source, MAXWELL)
        skeleton, layers = deck_layers(deck, kind=kind, kinds=kinds)
        layer_names, interface_names = maxwell_names(deck)
        geometry = Geometry(skeleton, layer_names=layer_names,
                            interface_names=interface_names)
        self.keep_deck(deck)
        super().__init__(geometry, layers,
                         name=str(deck.header.get("name", "")) or None)


class PronyDeckModel(Elastic, SelfGravitating, Viscoelastic, Tabulated, Model):
    """The model of a prony deck: `r rho vpv vsv vph vsh eta` and the pairs
    `relaxation_strength_k relaxation_time_k`, k = 1 to N, in SI under a
    title and the line naming the columns, NaN where a layer lacks a field.

    The columns are interpolated layer by layer by `kind`, and by the
    kind `kinds` gives a column or a family's stem (see
    `planetmodel.deck.KINDS` and `LOG_KINDS`); by default the strengths
    are joined by straight lines and the times in their logarithm by
    straight lines, `PRONY_KINDS`.  The elastic values are the
    unrelaxed moduli, and each layer holds the Prony branches it uses as
    `relaxation_strength_k` and `relaxation_time_k`, a generalised
    Maxwell body in shear that `frozen` relaxes: the branches after the
    last whose strength is not zero at every knot of the layer are not
    made fields of it, so a layer the writer completed holds the
    branches it was written from.  An elastic layer holds none.  A deck
    given rather than read is completed and held to the format as one
    written would be.  Its header names the layers and interfaces
    (`maxwell_names`) and the model; the deck is kept as `deck`, its
    knots per layer as `knots`, its header as `header` and its columns
    as `columns`, and `to_deck` writes the model back out as a prony
    deck on those knots (`conversion.prony.prony_deck`).  The model is in
    SI, as the deck is; `converted` gives it in other scales.
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, *, kind: str = "cubic",
                 kinds: Mapping[str, str] = PRONY_KINDS) -> None:
        deck = deck_from(source, PRONY)
        skeleton, layers = deck_layers(deck, kind=kind, kinds=kinds)
        layers = prony_layers(deck, layers)
        layer_names, interface_names = maxwell_names(deck)
        geometry = Geometry(skeleton, layer_names=layer_names,
                            interface_names=interface_names)
        self.keep_deck(deck)
        super().__init__(geometry, layers,
                         name=str(deck.header.get("name", "")) or None)

    def to_deck(self, *, columns: Sequence[str] | None = None) -> Deck:
        """The model as a prony deck on its knots, in SI, by
        `conversion.prony.prony_deck`, the header the deck's with the
        model's name and the names of its layers and interfaces as they
        now are; with `columns`, those fields as `Tabulated.to_deck`
        samples them."""
        if columns is not None:
            return super().to_deck(columns=columns)
        from .conversion.prony import prony_deck
        return self.restated(prony_deck(self, knots=self.knots))


class BurgersDeckModel(Elastic, SelfGravitating, Viscoelastic, Tabulated, Model):
    """The model of a burgers deck: `r rho vpv vsv vph vsh eta viscosity
    kelvin_viscosity kelvin_modulus_fraction` in SI under a title and the
    line naming the columns, NaN where a layer lacks a field.

    The columns are interpolated layer by layer by `kind`, and by the
    kind `kinds` gives a column (see `planetmodel.deck.KINDS` and
    `LOG_KINDS`); by default the two viscosities are joined in their
    logarithm by straight lines and the Kelvin modulus fraction by
    straight lines, `BURGERS_KINDS`.  The elastic values are the
    unrelaxed moduli.  A layer holding a viscosity alone is a Maxwell
    body and holds it, as a `MaxwellDeckModel`'s layer does; a layer
    holding the Kelvin element too is a Burgers body and holds its two
    Prony branches (`conversion.prony.burgers_branches`), fields of its
    elastic (Voigt) shear modulus and the three interpolated columns,
    which `frozen` relaxes, as `with_burgers` makes them; an elastic
    layer holds none.  A deck given rather than read is held to the
    format as one written would be.  Its header names the layers and
    interfaces (`maxwell_names`) and the model; the deck is kept as
    `deck`, its knots per layer as `knots`, its header as `header` and
    its columns as `columns`, and `to_deck` writes the model back out as
    a burgers deck on those knots (`conversion.prony.burgers_deck`, the
    Kelvin element recovered from the branches).  The model is in SI, as
    the deck is; `converted` gives it in other scales.
    """

    def __init__(self, source: str | os.PathLike[str] | Deck, *, kind: str = "cubic",
                 kinds: Mapping[str, str] = BURGERS_KINDS) -> None:
        from .conversion.prony import burgers_layers
        deck = deck_from(source, BURGERS)
        skeleton, layers = deck_layers(deck, kind=kind, kinds=kinds)
        layers = burgers_layers(layers)
        layer_names, interface_names = maxwell_names(deck)
        geometry = Geometry(skeleton, layer_names=layer_names,
                            interface_names=interface_names)
        self.keep_deck(deck)
        super().__init__(geometry, layers,
                         name=str(deck.header.get("name", "")) or None)

    def to_deck(self, *, columns: Sequence[str] | None = None) -> Deck:
        """The model as a burgers deck on its knots, in SI, by
        `conversion.prony.burgers_deck`, the header the deck's with the
        model's name and the names of its layers and interfaces as they
        now are; with `columns`, those fields as `Tabulated.to_deck`
        samples them."""
        if columns is not None:
            return super().to_deck(columns=columns)
        from .conversion.prony import burgers_deck
        return self.restated(burgers_deck(self, knots=self.knots))
