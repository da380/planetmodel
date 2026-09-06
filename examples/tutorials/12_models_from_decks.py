# %% [markdown]
# # 12. Models from decks
#
# A deck is a table of knots, radius first, one column per field, in which
# a repeated radius marks a boundary. What the columns are called and what
# the header lines mean is a `DeckFormat`; `MINEOS` is the format of the
# mineos and PREM decks. Reading one gives a `Deck`, numbers and a header.
# `deck_layers` interpolates every column layer by layer with a piecewise
# polynomial, so the base fields are exact for the algebra downstream and
# the moduli are products of them, and the mixins build the rest in the
# normal way. This tutorial reads PREM's own 200-knot tabulation as a
# `MineosModel` and compares it with the polynomial `PREM`, then writes a
# deck of its own, PREM's elastic part with a viscosity in the solid
# regions, and reads it back as a GIA-style model type.
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
# `read_deck` with the `MINEOS` format: three header lines (a title;
# `ifanis tref ifdeck`; `nknot nic noc`) and nine columns in SI. The
# layering comes from the repeated radii, thirteen layers here; the header
# says which are the inner and outer core.

# %%
deck = read_deck(DATA / "prem.200", MINEOS)
print(deck)
print("header:", dict(deck.header))
print("boundaries (km):", np.round(deck.boundaries / 1e3, 1))
s = deck.layers()[1]
print("outer core knots:", s, "| vsv there:", set(deck["vsv"][s]))

# %% [markdown]
# ## The model of a deck
#
# `MineosModel` is the model type of that format: the columns become the
# base fields by a cubic spline through each layer's knots, `Elastic`
# attaches the five moduli beside them, the header names the cores and
# sets the reference frequency of the constant-Q band, and everything a
# `PREM` can do this can too. It keeps its deck, its knots and its header,
# and `to_deck` writes it back out.

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
# Against the polynomial PREM the deck is a 200-knot sampling, and the
# spline through it agrees to a few parts in a million; the moduli are
# products of the splines and so agree to the same order, the mass to a
# part in ten million, and the degree-2 Love numbers to the same.

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
# A format is its column names and its header. Here PREM's elastic part
# sampled on eight knots per layer with a viscosity added on the solid
# layers, NaN on the fluid ones: a column that is NaN throughout a layer
# is absent from that layer, the way a fluid layer holds no `qmu`. The
# header is a title and the column names, and the format says how to
# read and write them.

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
# The model type of that format: a constructor that reads the deck and
# hands the interpolated layers to `Model`, the `Tabulated` mixin keeping
# the knots and the header, and the elastic, gravity and viscoelastic
# behaviours as before. A viscosity on a layer makes it a Maxwell body in
# shear, so `frozen(omega)` gives its complex moduli at a period, and the
# loading solver gives the Love numbers of the relaxed body.

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
    frozen = gia.frozen(2 * np.pi / (period * year))
    mesh = RadialMesh(frozen, ngll=5, lmax=8)
    k2 = love_numbers(frozen, 2, mesh=mesh).tidal()["k"][2]
    print(f"k2 at {period:8.0f} years: {k2.real:+.4f} {k2.imag:+.4f}i   "
          f"({type(frozen).__name__})")
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
