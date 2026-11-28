# %% [markdown]
# # 11. Models from decks
#
# A deck is a plain-text table describing a spherically symmetric model:
# one row per knot, with the radius first and then one column per
# field, the rows ordered by radius. A knot is a radius at which the
# table gives values. A radius that appears twice marks a boundary: the
# first of the two rows belongs to the layer below it and the second to
# the layer above. Decks are how PREM and the models used with the
# mineos normal-mode code are published, and many codes read one.
#
# The library does two things with a deck. It brings one in: the table
# becomes an ordinary model, the same kind of object as `PREM`. And it
# writes one out: any model becomes a table, for a code that reads one.
# In between, a model is kept in a *model file*, a netCDF file that
# holds the model's fields together with what they mean: their names
# and units, the names of the layers and interfaces, and what the model
# can do. Tutorial 12 looks inside a model file; here it is only
# written and read.
#
# This tutorial reads PREM's own 220-knot deck, makes a model of it and
# compares the model with the polynomial `PREM`. It saves the model as a
# model file, reads it back, and writes it as a mineos deck again. Last,
# it gives PREM a viscous mantle and takes that model through a file and
# through a table with a column of its own.
#
# This tutorial writes model files and plots, so it needs the `netcdf`
# and `plot` extras (netCDF4 and matplotlib). The figure is written to
# `examples/figures/`.

# %%
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import (MINEOS, PREM, constant_field, gravity, model_from_deck,
                         read_deck, sample_deck, testing, write_deck)
from planetmodel.behaviours import BEHAVIOURS, Elastic, SelfGravitating, Viscoelastic
from planetmodel.deck import deck_knots, mineos_constants, mineos_header, mineos_names
from planetmodel.modelfile import read_model, write_model

DATA = Path(__file__).resolve().parent.parent / "data"
FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## A table
#
# What a kind of deck file looks like is a `DeckFormat`: the names of
# its columns and the number of header lines above the rows. `MINEOS` is
# the format of mineos and PREM decks: three header lines, then nine
# columns in SI units, which are the radius, the density, the vertical
# P and S velocities, the two quality factors, the horizontal P and S
# velocities and the anisotropy parameter eta.
#
# `read_deck` turns a file into a `Deck`, which holds the numbers. The
# layering comes from the repeated radii, thirteen layers here. `layers`
# returns the rows of each layer; the outer core's S velocity is zero on
# every knot, as a fluid's must be. The header lines are kept exactly as
# they are in the file, and the library reads nothing from them.

# %%
deck = read_deck(DATA / "prem.200", MINEOS)
print(deck)
print("header lines:", deck.header["lines"])
print("boundaries (km):", np.round(deck.boundaries / 1e3, 1))
rows = deck.layers()[1]
print("outer core rows:", rows, "| vsv there:", np.unique(deck["vsv"][rows]))

# %% [markdown]
# ## What the header says
#
# A header often says things that the rows do not, and three functions
# read a mineos header for you when you want them to. `mineos_header`
# gives its values: the title, then `ifanis tref ifdeck`, then
# `nknot nic noc`. `tref` is the period, in seconds, at which the
# velocities are given. `nic` and `noc` count the knots up to the top of
# the inner core and of the outer core. `mineos_names` turns those two
# counts into names for the two core layers and their boundaries, and
# `mineos_constants` turns `tref` into the model's reference frequency.
#
# These are your choices to make. A header can be wrong, and you may
# know better names than it implies.

# %%
header = mineos_header(deck)
print("header values:", header)
layer_names, interface_names = mineos_names(deck)
print("layer names:", layer_names[:3], "...")
print("interface names:", interface_names[:2], "...", interface_names[-1])
constants = mineos_constants(deck)
print("constants:", {name: c.value_si for name, c in constants.items()})

# %% [markdown]
# ## The model of a table
#
# `model_from_deck` makes the model. Each column becomes one field on
# each layer, a cubic spline through that layer's knots. Everything
# else is given by you: the names and the constants from the step
# above, the model's name, and its *behaviours*. A behaviour is a group
# of methods that a model has. `Elastic` adds the five elastic moduli
# beside the velocities, `ConstantQ` reads the quality factors as
# attenuation, `SelfGravitating` computes mass and gravity, and
# `Viscoelastic` gives the model at a chosen frequency. `BEHAVIOURS` is
# all four, which is what `PREM` has. With no behaviours the result is a
# plain model that holds the columns and nothing more.

# %%
model = model_from_deck(deck, behaviours=BEHAVIOURS, layer_names=layer_names,
                        interface_names=interface_names, constants=constants,
                        name=header["name"])
print(model)
print("behaviours:", model.behaviours)
print("layers:", [layer.name for layer in model.layers][:3], "...",
      "| interfaces:", [f.name for f in model.geometry.interfaces][:2])
print("the outer core holds:", model.layer("outer_core").names)
print("reference period:", 2 * np.pi / model.reference_omega(), "s")
testing.check_model(model)
print("check_model passes")

# %% [markdown]
# The deck is a 220-knot sampling of the polynomial PREM, and the spline
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
# ## Saving the model
#
# `write_model` writes the model to a model file. The file holds every
# field, the names of the layers and interfaces, the constants, the
# model's name and its behaviours, so nothing has to be remembered
# beside it. `read_model` reads it back. The model that comes back is of
# the same type, with the same methods, and its fields are the same
# functions to rounding error, because a spline is stored exactly.

# %%
workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))
path = write_model(workdir / "prem_200.nc", model)
back = read_model(path)
print(path.name, f"written, {path.stat().st_size / 1e3:.0f} kB")
print(back)
print("same type:", type(back) is type(model), "| behaviours:", back.behaviours)
print("reference period:", 2 * np.pi / back.reference_omega(), "s")
r = np.linspace(3480e3, 3630e3, 7)
print("largest difference in vsv on the lowermost mantle:",
      np.max(np.abs(back.layer(2)["vsv"](r) / model.layer(2)["vsv"](r) - 1.0)))
print(f"mass: read back {back.mass():.7e}  as built {model.mass():.7e}")

# %% [markdown]
# ## Writing a table
#
# A code that reads a deck needs the model as a table again.
# `sample_deck` tabulates the columns you name. Given no knots it places
# its own, at every boundary and no further apart than a two-hundredth
# of the radius. Here it is given the knots of the table the model came
# from, with `deck_knots`, so the numbers can be compared.
#
# `write_deck` writes the table in a format. By default it writes the
# header that format needs, made from the table and the model: for
# `MINEOS` the title is the model's name, `tref` is the period of the
# model's reference frequency (0 if it has none), and `nknot nic noc`
# are counted from the table, the outer core being the innermost layer
# whose S velocity is zero. `header=` takes lines of your own, which are
# written as they are, or `False` for no header at all.

# %%
table = sample_deck(back, deck.names, knots=deck_knots(deck))
out = write_deck(workdir / "prem_again.200", table, MINEOS)
print("\n".join(line[:72] for line in out.read_text().splitlines()[:5]))
again = read_deck(out, MINEOS)
print("header values again:", mineos_header(again))
print("largest difference from the table read first:",
      max(float(np.max(np.abs(again[name] - deck[name]) / (np.abs(deck[name]) + 1.0)))
          for name in deck.names))
mine = write_deck(workdir / "prem_mine.200", table, MINEOS,
                  header=["PREM, from my model file", " 1 1.0 1", " 220 40 112"])
print("under lines of my own:", mine.read_text().splitlines()[0])

# %% [markdown]
# ## A model with a rheology
#
# A model used for glacial isostatic adjustment has a viscosity in its
# mantle: a layer with a viscosity is a Maxwell body in shear, which is
# elastic over short times and flows over long ones. The model built
# here is PREM without attenuation and without its ocean, with a
# viscosity of 10^22 Pa s below the 670 km discontinuity and 10^21 Pa s
# above it, up to the top of the low-velocity zone. The lid, the crust
# and the core have none.
#
# A viscosity is a field like any other, so the model goes to a model
# file and comes back with no extra step. The model read back is
# viscoelastic: `frozen` gives it at a chosen period, with complex
# moduli. The shear modulus of the lower mantle keeps its elastic value
# at ten years, and has relaxed most of the way to zero at a hundred
# thousand years, well past the Maxwell time of about 1300 years.

# %%
viscosity = {"lowermost_mantle": 1e22, "lower_mantle": 1e22, "upper_lower_mantle": 1e22,
             "transition_zone_lower": 1e21, "transition_zone_middle": 1e21,
             "transition_zone_upper": 1e21, "low_velocity_zone": 1e21}
viscous = prem.elastic().truncated(6368e3, name="surface")   # no Q, no ocean
for name, eta in viscosity.items():
    layer = viscous.layer(name)
    viscous = viscous.with_field(name, "viscosity",
                                 constant_field(layer.interval, eta, name="viscosity"))
maxwell = read_model(write_model(workdir / "prem_maxwell.nc", viscous))
print(maxwell)
print("viscoelastic layers:",
      [lay.name for lay in maxwell.layers if maxwell.is_viscoelastic(lay.index)])
year = 3.15576e7
x = 4500e3
print(f"elastic L at {x / 1e3:.0f} km: "
      f"{maxwell.layer('lower_mantle')['L'](x) / 1e9:.2f} GPa")
for period in (10.0, 1e3, 1e5):
    at_period = maxwell.frozen(2 * np.pi / (period * year))
    L = at_period.layer("lower_mantle")["L"](x) / 1e9
    print(f"L at {period:8.0f} years: {L.real:+8.2f} {L.imag:+8.2f}i GPa")

# %% [markdown]
# ## A table with columns of your own
#
# No published format has a viscosity column, and none is needed: a
# table with any columns is written and read by naming them. A layer
# without a viscosity has NaN in that column, and a column that is NaN
# on every knot of a layer is left out of that layer's fields, which is
# how a rectangular table carries a field that some layers do not have.
#
# Reading it back, `kinds` says how a column is joined between its
# knots where a cubic spline is the wrong choice. A viscosity can change
# by orders of magnitude across a layer, and a cubic through such values
# overshoots, even below zero. `log-linear` joins the logarithms of the
# values by straight lines instead. The names are given again, since a
# plain table has no header to hold them; here they are taken from the
# model the table was made from.

# %%
columns = ("rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity")
sampled = sample_deck(maxwell, columns)
plain = write_deck(workdir / "prem_maxwell.txt", sampled, columns,
                   header=["# r " + " ".join(columns)])
print("\n".join(line[:96] for line in plain.read_text().splitlines()[:3]))
own = model_from_deck(read_deck(plain, columns),
                      behaviours=[Elastic, SelfGravitating, Viscoelastic],
                      kinds={"viscosity": "log-linear"},
                      layer_names=[layer.name for layer in maxwell.layers],
                      interface_names=[f.name for f in maxwell.geometry.interfaces],
                      name=maxwell.name)
print(own)
print("the outer core holds no viscosity:", "viscosity" not in own.layer("outer_core"))
print(f"mass: from the table {own.mass():.6e}, from the file {maxwell.mass():.6e}")
testing.check_model(own)
print("check_model passes")

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
radial_profile(left, back, "rho", scale=1e-3, value_scale=1e-3, color="C3", lw=0.9,
               label="prem.200, from its model file")
for rows in deck.layers():
    left.plot(deck["rho"][rows] / 1e3, deck.radius[rows] / 1e3, "k.", ms=2)
left.set_xlabel("rho (g/cm^3)"); left.set_ylabel("r (km)"); left.legend()
left.set_title("the deck's knots and its spline")
radial_profile(middle, back, "L", scale=1e-3, value_scale=1e-9, lw=1.2)
middle.set_xlabel("L (GPa)"); middle.set_title("a modulus, product of splines")
radial_profile(right, maxwell, "viscosity", scale=1e-3, lw=1.2)
right.set_xscale("log"); right.set_xlabel("viscosity (Pa s)")
right.set_title("a viscosity, absent where elastic")
fig.tight_layout()
out = FIGURES / "tutorial_11_models_from_decks.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
