# %% [markdown]
# # 11. Models from decks
#
# A deck is a plain-text table describing a spherically symmetric model:
# one row per knot, with the radius first and then one column per
# field, the rows ordered by radius. A radius that appears twice marks a
# boundary: the first of the two rows belongs to the layer below it and
# the second to the layer above. Decks are how PREM and the models used
# with the mineos normal-mode code are published.
#
# What the columns are called and what the header lines mean is a
# `DeckFormat`; `MINEOS` is the format of mineos and PREM decks, and
# `MAXWELL` that of a viscoelastic model with a viscosity column.
# `read_deck` turns a file into a `Deck`, which is the numbers and the
# header and nothing else. `deck_layers` interpolates every column,
# layer by layer, with a piecewise polynomial through the knots, so the
# base fields are polynomials and the arithmetic of the earlier
# tutorials is exact on them; the moduli are products of them. The
# mixins then add everything else in the usual way.
#
# This tutorial reads PREM's own 200-knot deck as a `MineosModel` and
# compares it with the polynomial `PREM`. It then writes PREM with a
# viscous mantle as a `MAXWELL` deck, of the kind used for glacial
# isostatic adjustment, and reads it back as a `MaxwellDeckModel`.
# Last, it defines a format and a model type of its own.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import (MAXWELL, MINEOS, PREM, Deck, DeckFormat, Elastic, Geometry,
                         MaxwellDeckModel, MineosModel, Model, SelfGravitating,
                         Tabulated, constant_field, deck_layers, gravity, read_deck,
                         sample_deck, testing, write_deck)

DATA = Path(__file__).resolve().parent.parent / "data"
FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A mineos deck
#
# `read_deck` with the `MINEOS` format reads three header lines, a
# title, then `ifanis tref ifdeck`, then `nknot nic noc`, and nine
# columns in SI units: the radius, the density, the vertical and
# horizontal P and S velocities, the two quality factors and the
# anisotropy parameter eta. The layering comes from the repeated radii,
# thirteen layers here. The header's `nic` and `noc` say which knots end
# the inner and the outer core. `layers` returns a slice of the knots
# for each layer; the outer core's S velocity is zero on every knot, as
# a fluid's must be.

# %%
deck = read_deck(DATA / "prem.200", MINEOS)
print(deck)
print("header:", dict(deck.header))
print("boundaries (km):", np.round(deck.boundaries / 1e3, 1))
s = deck.layers()[1]
print("outer core knots:", s, "| vsv there:", np.unique(deck["vsv"][s]))

# %% [markdown]
# ## The model of a deck
#
# `MineosModel` is the model type of that format. The columns become the
# base fields by a cubic spline through each layer's knots. `Elastic`
# adds the five moduli beside them, the header names the two core layers
# and their boundaries and sets the reference frequency of the
# attenuation model, and the model can do everything a `PREM` can. It
# keeps its deck, its knots and its header, and `to_deck` writes it
# back out.

# %%
model = MineosModel(deck)
print(model)
print("layers:", [layer.name for layer in model.layers][:3], "...",
      "| interfaces:", [f.name for f in model.geometry.interfaces][:2])
print("the outer core holds:", model.layer("outer_core").names)
print("reference period:", 2 * np.pi / model.reference_omega(), "s")
testing.check_model(model)
print("check_model passes")

# %% [markdown]
# The deck is a 200-knot sampling of the polynomial PREM, and the spline
# through it agrees with the polynomial to about a part in a million.
# The moduli are products of splines and agree to the same order, and
# so do the mass and the gravity that follows from the density.

# %%
prem = PREM()
for x in (1000e3, 3000e3, 5000e3, 6300e3):
    i = model.skeleton.locate(x).layer
    d, p = model.layer(i), prem.layer(i)
    print(f"r = {x / 1e3:5.0f} km: rho {d['rho'](x):9.2f} vs {p['rho'](x):9.2f}   "
          f"A/A_prem - 1 = {d['A'](x) / p['A'](x) - 1:+.1e}")
print(f"mass: deck {model.mass():.7e}  prem {prem.mass():.7e}")
for x in (3480e3, 6371e3):
    print(f"g at {x / 1e3:4.0f} km: deck {gravity(model, x):.6f}  "
          f"prem {gravity(prem, x):.6f}")

# %% [markdown]
# ## A Maxwell deck
#
# The library ships a second format, `MAXWELL`, for the models used in
# glacial isostatic adjustment. It has seven columns after the radius:
# the density, the five elastic velocities of a transversely isotropic
# medium, and a viscosity. The header is a title and a line naming the
# columns, and after them, optionally, a `# layers` line and a
# `# interfaces` line giving their names from the centre outward, with
# `-` for one that has no name. A layer with a viscosity is a Maxwell
# body in shear. A layer without one has NaN in that column; a column
# that is NaN on every knot of a layer is absent from that layer's
# fields, which is how a rectangular table carries a field that some
# layers do not have.
#
# `sample_deck` turns any model into a deck. It samples the columns you
# ask for on each layer, putting knots at the layer's ends, at every
# breakpoint of its fields, and in between no further apart than a
# two-hundredth of the outer radius, about 32 km for the Earth. A
# velocity that a layer does not hold is computed from its density and
# moduli. The header gets the model's name and the names of its layers
# and interfaces, which `write_deck` puts on the naming lines.
#
# The model written here is PREM without attenuation and without its
# ocean, with a viscosity of 10^22 Pa s below the 670 km discontinuity
# and 10^21 Pa s above it, up to the top of the low-velocity zone. The
# lid, the crust and the inner core stay elastic.

# %%
viscosity = {"lowermost_mantle": 1e22, "lower_mantle": 1e22, "upper_lower_mantle": 1e22,
             "transition_zone_lower": 1e21, "transition_zone_middle": 1e21,
             "transition_zone_upper": 1e21, "low_velocity_zone": 1e21}
viscous = prem.elastic().truncated(6368e3, name="surface")   # no Q, no ocean
for name, eta in viscosity.items():
    layer = viscous.layer(name)
    viscous = viscous.with_field(name, "viscosity",
                                 constant_field(layer.interval, eta, name="viscosity"))
maxwell_deck = sample_deck(viscous, MAXWELL.columns)
workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))
path = write_deck(workdir / "prem_maxwell.deck", maxwell_deck, MAXWELL)
print(path.name, "written;", maxwell_deck)
print("\n".join(line[:96] for line in path.read_text().splitlines()[:5]))

# %% [markdown]
# `MaxwellDeckModel` is the model type of that format. It reads the
# deck, names the layers and interfaces from the naming lines, and joins
# each layer's knots by a cubic spline, except the viscosity's, which
# are joined by straight lines: a viscosity can change by orders of
# magnitude across a layer, and a cubic through such values overshoots,
# even below zero. The elastic, gravity and viscoelastic mixins are the
# same as `PREM`'s, and `frozen` gives the model with complex moduli at
# a chosen period. The shear modulus of the lower mantle keeps its
# elastic value at ten years, and has relaxed most of the way to zero at
# a hundred thousand years, well past the Maxwell time of about two
# thousand years.

# %%
maxwell = MaxwellDeckModel(path)
print(maxwell)
print("viscoelastic layers:",
      [lay.name for lay in maxwell.layers if maxwell.is_viscoelastic(lay.index)])
print("the outer core holds no viscosity:",
      "viscosity" not in maxwell.layer("outer_core"))
year = 3.15576e7
x = 4500e3
print(f"elastic L at {x / 1e3:.0f} km: "
      f"{maxwell.layer('lower_mantle')['L'](x) / 1e9:.2f} GPa")
for period in (10.0, 1e3, 1e5):
    at_period = maxwell.frozen(2 * np.pi / (period * year))
    L = at_period.layer("lower_mantle")["L"](x) / 1e9
    print(f"L at {period:8.0f} years: {L.real:+8.2f} {L.imag:+8.2f}i GPa   "
          f"({type(at_period).__name__})")
testing.check_model(maxwell)
print("check_model passes")

# %% [markdown]
# ## A format of your own
#
# A format of your own is a `DeckFormat` given its column names and the
# number of header lines. Without functions to read and write the
# header, the header lines are kept as they are. The model type for it
# is a short class: its constructor reads the deck, interpolates the
# layers, keeps the deck with `keep_deck`, a method of the `Tabulated`
# mixin, and passes the geometry and the layers to `Model`. Because the
# deck is kept, `to_deck` gives the model back as a deck on the same
# knots.
#
# Here the format holds the density and the isotropic velocities under
# one title line, and the deck is PREM's isotropic average sampled on
# six evenly spaced knots per layer.

# %%
SIMPLE = DeckFormat(("rho", "vp", "vs"), name="simple", header_lines=1)


class SimpleDeckModel(Elastic, SelfGravitating, Tabulated, Model):
    """An isotropic model from a deck of density and two velocities."""

    def __init__(self, source):
        d = read_deck(source, SIMPLE)
        skeleton, layers = deck_layers(d)
        self.keep_deck(d)
        super().__init__(Geometry(skeleton), layers, name=d.header["lines"][0])


iso = prem.isotropic()
sampled = sample_deck(iso, SIMPLE.columns,
                      knots=[np.linspace(*layer.interval, 6) for layer in iso.layers])
simple = Deck(sampled.radius, sampled.columns, header={"lines": ("PREM, isotropic",)})
own = SimpleDeckModel(write_deck(workdir / "prem_simple.deck", simple, SIMPLE))
print(own)
print(f"mass: {own.mass():.6e}, PREM {prem.mass():.6e}")
print("its deck again:", own.to_deck())

# %% [markdown]
# ## A picture

# %%
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

from planetmodel.plotting import radial_profile  # noqa: E402

fig, (left, middle, right) = plt.subplots(1, 3, figsize=(14, 6), sharey=True)
radial_profile(left, prem, "rho", scale=1e-3, value_scale=1e-3, color="0.6", lw=2.5,
               label="PREM, polynomial")
radial_profile(left, model, "rho", scale=1e-3, value_scale=1e-3, color="C3", lw=0.9,
               label="prem.200, spline")
for s in deck.layers():
    left.plot(deck["rho"][s] / 1e3, deck.radius[s] / 1e3, "k.", ms=2)
left.set_xlabel("rho (g/cm^3)"); left.set_ylabel("r (km)"); left.legend()
left.set_title("the deck's knots and its spline")
radial_profile(middle, model, "L", scale=1e-3, value_scale=1e-9, lw=1.2)
middle.set_xlabel("L (GPa)"); middle.set_title("a modulus, product of splines")
radial_profile(right, maxwell, "viscosity", scale=1e-3, lw=1.2)
right.set_xscale("log"); right.set_xlabel("viscosity (Pa s)")
right.set_title("a Maxwell deck's viscosity, absent where elastic")
fig.tight_layout()
out = FIGURES / "tutorial_11_models_from_decks.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
