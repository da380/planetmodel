# %% [markdown]
# # 12. Models from decks
#
# A deck is a plain-text table describing a spherically symmetric model:
# one row per knot, with the radius first and then one column per
# field, the rows ordered by radius. A radius that appears twice marks a
# boundary: the first of the two rows belongs to the layer below it and
# the second to the layer above. Decks are how PREM and the models used
# with the mineos normal-mode code are published.
#
# What the columns are called and what the header lines mean is a
# `DeckFormat`; `MINEOS` is the format of mineos and PREM decks.
# `read_deck` turns a file into a `Deck`, which is the numbers and the
# header and nothing else. `deck_layers` interpolates every column,
# layer by layer, with a piecewise polynomial through the knots, so the
# base fields are polynomials and the arithmetic of the earlier
# tutorials is exact on them; the moduli are products of them. The
# mixins then add everything else in the usual way.
#
# This tutorial reads PREM's own 200-knot deck as a `MineosModel` and
# compares it with the polynomial `PREM`. It then writes a deck in a
# format of its own, PREM's elastic part with a viscosity column on the
# solid layers, and reads it back as a model type of its own, of the
# kind used for glacial isostatic adjustment.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
import tempfile
from pathlib import Path

import numpy as np

from planetmodel import (PREM, Deck, DeckFormat, Elastic, Geometry, MINEOS,
                         MineosModel, Model, RadialMesh, SelfGravitating, Tabulated,
                         Viscoelastic, deck_layers, read_deck, testing, write_deck)
from planetmodel.deck import deck_knots
from planetmodel.loading import love_numbers

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
# The moduli are products of splines and agree to the same order. The
# mass and the degree-2 load Love numbers agree to a part in ten
# million; for the Love numbers both models are cut at the top of the
# crust, since a fluid surface cannot be loaded.

# %%
prem = PREM()
for x in (1000e3, 3000e3, 5000e3, 6300e3):
    i = model.skeleton.locate(x).layer
    d, p = model.layer(i), prem.layer(i)
    print(f"r = {x / 1e3:5.0f} km: rho {d['rho'](x):9.2f} vs {p['rho'](x):9.2f}   "
          f"A/A_prem - 1 = {d['A'](x) / p['A'](x) - 1:+.1e}")
print(f"mass: deck {model.mass():.7e}  prem {prem.mass():.7e}")
mesh_d = RadialMesh(model.truncated(6368e3), ngll=5, lmax=8)
mesh_p = RadialMesh(prem.truncated(6368e3), ngll=5, lmax=8)
ld = love_numbers(model.truncated(6368e3), 2, mesh=mesh_d).conventional()
lp = love_numbers(prem.truncated(6368e3), 2, mesh=mesh_p).conventional()
print("degree-2 load Love numbers h', k':", ld["h"][2], ld["k"][2],
      "| prem:", lp["h"][2], lp["k"][2])

# %% [markdown]
# ## A deck of your own
#
# A format is a set of column names and a description of the header.
# The format here has seven columns after the radius, PREM's elastic
# fields and a viscosity, and a header of two lines: a title, and the
# column names. Its two functions say how the header lines are read
# into a mapping and written back from one.
#
# The deck itself is built from PREM's elastic part, with no attenuation
# and no ocean, sampled on eight knots per layer. The viscosity column
# is set on the solid layers and NaN on the fluid outer core. A column
# that is NaN on every knot of a layer is absent from that layer's
# fields, which is how a rectangular table carries a field that some
# layers do not have.

# %%
GIA = DeckFormat(
    ("rho", "vpv", "vsv", "vph", "vsh", "eta", "viscosity"), name="gia",
    header_lines=2,
    parse_header=lambda lines: {"title": lines[0].strip(),
                                "columns": tuple(lines[1].split()[1:])},
    write_header=lambda d: [d.header["title"], "r " + " ".join(d.names)])

elastic = prem.elastic().truncated(6368e3, name="surface")   # no Q, no ocean
viscosity = {"lower_mantle": 1e22, "upper_lower_mantle": 1e22,
             "lowermost_mantle": 1e22, "inner_core": 1e21}
radii, columns = [], {name: [] for name in GIA.columns}
for layer in elastic.layers:
    lo, hi = layer.interval
    r = np.linspace(lo, hi, 8)
    radii.append(r)
    for name in GIA.columns:
        if name == "viscosity":
            eta = np.nan if elastic.is_fluid(layer.index) else viscosity.get(layer.name,
                                                                            1e21)
            columns[name].append(np.full(r.size, eta))
        else:
            columns[name].append(layer[name](r))
gia_deck = Deck(np.concatenate(radii),
                {n: np.concatenate(v) for n, v in columns.items()},
                header={"title": "PREM, elastic, with a Maxwell viscosity"})
workdir = Path(tempfile.mkdtemp(prefix="planetmodel_"))
path = write_deck(workdir / "prem_gia.deck", gia_deck, GIA)
print(path.name, "written;", gia_deck)
print("\n".join(path.read_text().splitlines()[:4]))


# %% [markdown]
# The model type for that format is a short class. Its constructor
# reads the deck, interpolates the layers, keeps the knots and the
# header as the `Tabulated` mixin expects, and passes the geometry and
# the layers to `Model`. The layer names are taken from the model the
# deck was written from, since this format's header does not record
# them. The elastic, gravity and viscoelastic mixins are the same as
# before. A layer with a viscosity is a Maxwell body in shear, so
# `frozen` gives the model with complex moduli at a chosen period, and
# the loading solver gives the complex tidal Love number at that period:
# the elastic value at ten years, and most of the way to the fluid limit
# at a hundred thousand years.

# %%
class GIAModel(Elastic, SelfGravitating, Viscoelastic, Tabulated, Model):
    """PREM's elastic part with a Maxwell viscosity on the solid layers."""

    def __init__(self, source, *, kind="cubic"):
        d = read_deck(source, GIA)
        skeleton, layers = deck_layers(d, kind=kind)
        names = [lay.name for lay in elastic.layers]          # the same layering
        self.knots, self.header = deck_knots(d), d.header
        super().__init__(Geometry(skeleton, layer_names=names), layers)


gia = GIAModel(path)
print(gia, "|", gia.header["title"])
print("viscoelastic layers:",
      [lay.name for lay in gia.layers if gia.is_viscoelastic(lay.index)])
print("the outer core holds no viscosity:", "viscosity" not in gia.layer("outer_core"))
year = 3.15576e7
elastic_love = love_numbers(gia, 2, mesh=RadialMesh(gia, ngll=5, lmax=8)).tidal()
print("elastic tidal k2:", elastic_love["k"][2])
for period in (10.0, 1e3, 1e5):
    at_period = gia.frozen(2 * np.pi / (period * year))
    mesh = RadialMesh(at_period, ngll=5, lmax=8)
    k2 = love_numbers(at_period, 2, mesh=mesh).tidal()["k"][2]
    print(f"k2 at {period:8.0f} years: {k2.real:+.4f} {k2.imag:+.4f}i   "
          f"({type(at_period).__name__})")
testing.check_model(gia)
print("check_model passes; the deck it now is:",
      gia.to_deck(columns=["rho", "viscosity"]))

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
radial_profile(right, gia, "viscosity", scale=1e-3, lw=1.2)
right.set_xscale("log"); right.set_xlabel("viscosity (Pa s)")
right.set_title("the custom column, absent in the fluid")
fig.tight_layout()
out = FIGURES / "tutorial_12_models_from_decks.png"
fig.savefig(out, dpi=120)
print("figure written to", out)
