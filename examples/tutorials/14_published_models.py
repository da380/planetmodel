# %% [markdown]
# # 14. Published models
#
# Most models of the Earth's interior are published by their authors as
# files: tables of coefficients, grids of values, tables of a radial
# model. `planetmodel.datasets` knows where a set of them are kept, fetches
# them when they are first asked for, checks that each file is the one its
# entry records, and reads it into the library's own objects, so that a
# published model can be looked at, compared with another, and put onto a
# model of your own.
#
# This tutorial goes through those steps with real models: the
# shear-velocity model S40RTS, the model S362ANI+M given as velocities on
# a grid, a regional model of the North Atlantic, one of Antarctica given
# on a polar map, and a model of the Moon.
# It needs the network the first time it runs (the files are a few
# megabytes, and are kept in a cache afterwards), the `netcdf` extra
# (netCDF4) for the gridded models, and the `maps` extra (cartopy) for the
# maps. The figures are written to `examples/figures/`.

# %%
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

from planetmodel import PREM, gauss_legendre, equiangular, from_geographic, slices
from planetmodel.frames import cartesian_points
from planetmodel import datasets
from planetmodel.modelfile import read_model, write_model
from planetmodel.plotting import (correlation, cutaway, profile, section, spectrum,
                                  sphere_map)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The registry
#
# A *data set* is a published model the library knows about. Its *entry*
# says where each of its files is served, the file's checksum (a
# fingerprint that changes if a single byte of the file does), how the
# file is read, which model its values are relative to, and how its
# authors ask to be cited, with its licence in their own words. The
# library holds the entries, never the data: the files come from their
# authors' own hosts. `REGISTRY` holds the shipped entries by name.

# %%
print(sorted(datasets.REGISTRY))
print()
print(datasets.describe("S40RTS"))

# %% [markdown]
# `fetch` brings a data set's files into a cache, a directory on your
# machine (`PLANETMODEL_DATA` if you set it, else your platform's usual
# place for caches), and gives their paths. It downloads a file only when
# it is not already there, and checks its checksum every time; a file
# that does not match is refused. `cache_directory()` says where the cache
# is.

# %%
print(datasets.fetch("S40RTS"))
print("the cache:", datasets.cache_directory())

# %% [markdown]
# ## A data set as the model it is
#
# S40RTS (Ritsema and others, 2011) is a model of the shear velocity of
# the mantle, published as coefficients of spherical harmonics up to
# degree 40 on 21 radial splines between the core-mantle boundary and the
# Moho. What it describes is a *relative perturbation*: how much faster
# or slower than PREM the shear velocity is at each point, as a fraction
# of PREM's velocity there.
#
# `load` reads it as a model holding exactly that: a model on PREM's
# layers from the core-mantle boundary to the surface, whose field `vs` on
# each mantle layer is PREM's shear velocity times one plus the
# perturbation, and whose crust holds nothing. (PREM's shear velocity in
# its upper mantle differs between the two polarisations; `vs` is their
# Voigt average.) The perturbation is held exactly, as the harmonic
# expansion the file is. Everything the library does with a model works
# on it as it stands: here a map of the perturbation 1000 km deep, as
# its authors draw it.

# %%
s40 = datasets.load("S40RTS")
print(s40)
deep = slices.sphere(s40, "vs", depth=1000e3, relative_to="reference")
print(deep.label)

fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
sphere_map(ax, deep, colorbar=True, coastlines=True)
ax.set_title("S40RTS, vs 1000 km deep")
across = slices.section(s40, "vs", from_geographic(0.0, -60.0), azimuth=0.5 * np.pi,
                        length=np.pi, radii=(3480e3, 6346.6e3), relative_to="reference")
ax = fig.add_subplot(1, 2, 2, projection="polar")
section(ax, across, scale=1e-3, colorbar=True)
ax.set_rticks([4000, 5000, 6000])
ax.set_title("the mantle along the equator, from 60 degrees west")
fig.tight_layout()
out = FIGURES / "tutorial_14_s40rts.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# A cutaway is the picture such models were first shown in: the Earth
# seen from outside with an octant removed down to the core-mantle
# boundary, the perturbation on the faces of the cut, the core standing
# in it, and the surface wearing Natural Earth's shaded relief with the
# coastlines drawn where they are seen. Here the octant north of the
# equator from 60 degrees west to 30 degrees east is removed, and the
# viewer is where the default puts them, 25 degrees north and a little
# east of the cut's first face, over the Atlantic; on the right the same cut
# with the perturbation at the surface too and the coastlines over it.
# `n` is the number of pixels across the picture, and the figure is saved
# at a resolution that keeps them.

# %%
octant = slices.cutaway(s40, "vs", longitudes=(np.radians(-60.0), np.radians(30.0)),
                        colatitudes=(0.0, 0.5 * np.pi), inner=3480e3, n=1200,
                        relative_to="reference")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
cutaway(ax, octant, scale=1e-3, surface="image", inner="0.8", coastlines=True,
        clip=99.5, colorbar=True)
ax.set_title("S40RTS, the northern octant removed")
cutaway(bx, octant, scale=1e-3, inner="0.8", coastlines=True, clip=99.5)
bx.set_title("the perturbation at the surface too")
out = FIGURES / "tutorial_14_cutaway.png"
fig.savefig(out, dpi=200)
print("figure written to", out)

# %% [markdown]
# The surface can be dressed in several ways, and the cut faces always
# show the field. `surface=` is `"field"`, `"image"` for Natural Earth's
# shaded relief, `"land"` for the land and the sea in two plain
# colours, or any matplotlib colour; `coastlines=True` adds the
# coastlines to any of them. `inner=` dresses the sphere left standing
# in the cut the same way: a colour, or `"field"` for the perturbation
# at the core-mantle boundary. Plain colours and the relief are lit
# from the upper left so that the sphere reads as one; the field's
# colours are never shaded.

# %%
dressings = [
    ("the field at the surface", dict()),
    ("the field with coastlines", dict(coastlines=True)),
    ("Natural Earth's relief", dict(surface="image")),
    ("the relief with coastlines", dict(surface="image", coastlines=True)),
    ("land and sea", dict(surface="land")),
    ("land and sea with coastlines", dict(surface="land", coastlines=True)),
    ("a plain surface with coastlines", dict(surface="0.92", coastlines=True)),
    ("the field on the core, the surface plain",
     dict(surface="0.92", inner="field", coastlines=True)),
]
fig, axs = plt.subplots(2, 4, figsize=(24, 12), layout="constrained")
for ax, (title, style) in zip(axs.ravel(), dressings):
    style.setdefault("inner", "0.8")
    cutaway(ax, octant, scale=1e-3, clip=99.5, **style)
    ax.set_title(title)
fig.suptitle("S40RTS with the northern octant removed: what the surface can show")
out = FIGURES / "tutorial_14_cutaway_surfaces.png"
fig.savefig(out, dpi=200)
print("figure written to", out)

# %% [markdown]
# The cut need not be an octant. Left to their defaults the colatitudes
# span the whole body, so the wedge goes from pole to pole, and `inner`
# is the centre; S40RTS is a model of the mantle alone, so the cut stops
# at its inner boundary, the core-mantle boundary, whose sphere is then
# seen in the cut. Here a wedge of a hundred degrees is removed, the
# relief on the surface on the left, and on the right the field on the
# surface and on the core-mantle boundary.

# %%
wedge = slices.cutaway(s40, "vs", longitudes=(np.radians(-100.0), 0.0), n=1200,
                       relative_to="reference")
print("the cut stops at", wedge.inner / 1e3, "km")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(14, 7), layout="constrained")
cutaway(ax, wedge, scale=1e-3, clip=99.5, surface="image", inner="0.8", coastlines=True,
        colorbar=True)
ax.set_title("a wedge of 100 degrees cut to the core")
cutaway(bx, wedge, scale=1e-3, clip=99.5, coastlines=True)
bx.set_title("the field on the surface and on the core-mantle boundary")
out = FIGURES / "tutorial_14_cutaway_wedge.png"
fig.savefig(out, dpi=200)
print("figure written to", out)

# %% [markdown]
# ## A published model on a model of your own
#
# `transplanted` puts a data set's perturbation onto a model of your own.
# The perturbation is kept as a percentage (`mode="relative"`, the usual
# practice), so that your model's own radial structure stays as it is.
# Because PREM holds two shear velocities, `vsv` and `vsh`, and S40RTS
# perturbs one isotropic `vs`, the perturbation goes onto both, as
# SPECFEM applies S40RTS to PREM. PREM's model type then recomputes its
# elastic moduli from the perturbed velocities.
#
# `lmax=` keeps the perturbation's degrees up to a band; `roll_off=`
# brings the highest of them down smoothly rather than cutting them off.
# The spectra show what that does: the power of the perturbation at each
# degree, at one depth, before and after. Above degree 12 the filtered
# perturbation has no power at all, so its curve leaves the bottom of the
# picture.

# %%
prem = PREM()
on_prem = datasets.transplanted(prem, "S40RTS")
smooth = datasets.transplanted(prem, "S40RTS", lmax=12, roll_off=4)
lay = on_prem.layer("lower_mantle")
r, theta, phi = 5.0e6, 1.0, 2.0
L = lay["rho"](r) * lay["vsv"](r, theta, phi) ** 2
again = np.allclose(lay["L"](r, theta, phi), L)
print("vsv on the lower mantle:", type(lay["vsv"]).__name__,
      "| L there is rho vsv^2 again:", again)

fig, ax = plt.subplots(figsize=(7, 4))
for model, label in ((on_prem, "degree 40"), (smooth, "to degree 12, rolled off")):
    sp = slices.spectrum(model, "vsv", lmax=40, depth=1000e3, relative_to="reference")
    spectrum(ax, sp, label=label)
ax.set_ylim(1e-7, 1e-1)                 # above degree 12 the filtered power is zero
ax.set_xlabel("degree")
ax.set_ylabel("power")
ax.legend()
ax.set_title("the perturbation of vsv 1000 km deep, by degree")
fig.tight_layout()
out = FIGURES / "tutorial_14_spectra.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# On PREM the model has a core, so a cutaway can go down to the
# core-mantle boundary, to the inner-core boundary, or to the centre:
# `inner=` is a radius, or the name of one of the model's interfaces.
# The sphere left standing in the cut is drawn plain, or with the field
# on it; at an interface the field has two sides, and `side=` says
# which, `"below"` for the core's values and `"above"` for the mantle's
# at the base of the mantle. Cut to the centre, the faces run through
# the core, where the perturbation is zero, so the two deeper cuts show
# the velocity itself: zero in the liquid outer core, and the solid
# inner core standing at the centre.

# %%
cuts = [
    ("to the core-mantle boundary", dict(inner="cmb", relative_to="reference"),
     dict(inner="0.8", clip=99.5)),
    ("to the core-mantle boundary, the mantle side on it",
     dict(inner="cmb", side="above", relative_to="reference"),
     dict(inner="field", clip=99.5)),
    ("to the inner-core boundary, the velocity itself", dict(inner="icb"),
     dict(inner="0.8")),
    ("to the centre, the velocity itself", dict(inner=0.0), dict()),
]
fig, axs = plt.subplots(2, 2, figsize=(14, 14), layout="constrained")
for ax, (title, cut, style) in zip(axs.ravel(), cuts):
    deep = slices.cutaway(on_prem, "vsv", colatitudes=(0.0, 0.5 * np.pi), n=1000,
                          longitudes=(np.radians(-60.0), np.radians(30.0)), **cut)
    cutaway(ax, deep, scale=1e-3, surface="image", coastlines=True, **style)
    ax.set_title(title)
fig.suptitle("S40RTS on PREM, cut to each depth")
out = FIGURES / "tutorial_14_cutaway_depths.png"
fig.savefig(out, dpi=200)
print("figure written to", out)

# %% [markdown]
# ## A model given as velocities
#
# S362ANI+M (Moulik and Ekström, 2014) is offered by the Earth Model
# Collaboration as velocities in km/s on a grid of latitude, longitude
# and depth. Read as it is, it is a model of values, not of
# perturbations. Its authors' reference is the radial model STW105, which
# the registry also holds; `transplanted` makes the velocities relative to
# STW105 first, and carries that percentage onto PREM.
#
# The grid runs from 25 to 2890 km deep, so it reaches neither PREM's
# Moho (24.4 km) nor its core-mantle boundary (2891 km), and covers two of
# PREM's layers only in part. Rather than guess, the library refuses, and
# says how to choose: `ends=("cmb", "moho")` stretches the model's depth
# range onto the span between two of PREM's interfaces, `extend=True`
# carries its end values out to the layers' ends, and `taper=` takes the
# perturbation to zero inside its ends.

# %%
try:
    datasets.transplanted(prem, "S362ANI+M")
except ValueError as exc:
    print("refused:", str(exc)[:240], "...")
s362 = datasets.transplanted(prem, "S362ANI+M", ends=("cmb", "moho"))

# %% [markdown]
# ## Comparing two models
#
# Two models of the same mantle agree more at some scales and depths than
# at others. The correlation of their perturbations by degree and depth
# shows where: near one where they agree, near zero where they are
# unrelated. The two models were made from different data by different
# methods. They agree best at the largest scales, and least at the smaller
# scales between about 5500 and 6100 km radius, in and around the
# transition zone.

# %%
compared = slices.correlation(on_prem, "vsv", s362, lmax=18,
                              depths=np.linspace(150e3, 2800e3, 25),
                              relative_to="reference")
fig, ax = plt.subplots(figsize=(7, 5))
correlation(ax, compared, scale=1e-3, colorbar=True)
ax.set_xlabel("degree")
ax.set_ylabel("radius [km]")
ax.set_title("S40RTS against S362ANI+M on PREM, vsv")
fig.tight_layout()
out = FIGURES / "tutorial_14_correlation.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A regional model inside a global one
#
# A regional model covers part of the Earth in more detail. The Earth
# Model Collaboration's North Atlantic part of the Collaborative Seismic
# Earth Model covers latitudes 20 to 75 degrees north and longitudes 55
# degrees west to 10 east, down to 500 km. `load` reads it as a regional
# grid, which can be asked for its values inside its box. `merged` blends
# it into a global model: inside the box, the regional values; outside,
# the global ones; between them, over a band of the width asked for, a
# smooth change from one to the other, and again over 50 km at the base
# of the regional model. The result is held on the angular grid given.

# %%
atlantic = datasets.load("CSEM-North-Atlantic")
print("a regional grid of", atlantic.names)
blended = on_prem
for name in ("vsv", "vsh"):
    blended = datasets.merged(blended, atlantic, name=name, width=np.radians(5.0),
                              grid=equiangular(180, 360), radial_width=(50e3, 0.0))

fig = plt.figure(figsize=(12, 5))
view = ccrs.Orthographic(-18.0, 55.0)
for k, (model, title) in enumerate(((on_prem, "S40RTS on PREM"),
                                    (blended, "with the North Atlantic merged"))):
    ax = fig.add_subplot(1, 2, k + 1, projection=view)
    at = slices.sphere(model, "vsv", depth=150e3, relative_to="mean")
    sphere_map(ax, at, colorbar=True, coastlines=True, vmin=-6.0, vmax=6.0)
    ax.set_title(f"{title}: vsv 150 km deep")
fig.tight_layout()
out = FIGURES / "tutorial_14_merged.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A regional model on a polar map
#
# Models of the polar regions are often published on a flat map of the
# pole, a *polar stereographic projection*, rather than in latitude and
# longitude. ANT-20 (Lloyd and others, 2020), a shear-velocity model of
# Antarctica and the surrounding oceans, is given on the standard one for
# Antarctica, EPSG:3031: distances in kilometres from the South Pole, x
# toward 90 degrees east and y toward the Greenwich meridian. `load` reads
# it as a regional grid that knows its projection, and `merged` blends it
# into a global model as it did the North Atlantic, measuring the band at
# its edge from where the model's values stop.
#
# At 100 km depth, East Antarctica, an old and cold continent, is fast,
# and West Antarctica, younger and warmer, is slow: in ANT-20 about 5
# percent faster and 2 percent slower than PREM. S40RTS shows the same
# pattern more smoothly. Had the projection's axes been read the wrong
# way round, the slow region would lie beneath East Antarctica, or out in
# the ocean, and the two maps would disagree.

# %%
antarctica = datasets.load("ANT-20")
print("ANT-20 on", antarctica.projection)
polar = on_prem
for name in ("vsv", "vsh"):
    polar = datasets.merged(polar, antarctica, name=name, source_name="vs",
                            width=np.radians(5.0), grid=equiangular(90, 180),
                            radial_width=(50e3, 0.0))

fig = plt.figure(figsize=(12, 5.5))
view = ccrs.SouthPolarStereo()
for k, (model, title) in enumerate(((on_prem, "S40RTS on PREM"),
                                    (polar, "with ANT-20 merged"))):
    ax = fig.add_subplot(1, 2, k + 1, projection=view)
    at = slices.sphere(model, "vsv", depth=100e3, relative_to="mean")
    sphere_map(ax, at, colorbar=True, coastlines=True, vmin=-6.0, vmax=6.0)
    ax.set_extent([-180.0, 180.0, -90.0, -55.0], ccrs.PlateCarree())
    for label, (lon, lat) in (("West\nAntarctica", (-120.0, -79.0)),
                              ("East\nAntarctica", (75.0, -78.0))):
        ax.text(lon, lat, label, transform=ccrs.PlateCarree(), ha="center",
                va="center", fontsize=9)
    ax.set_title(f"{title}: vsv 100 km deep")
fig.tight_layout()
out = FIGURES / "tutorial_14_antarctica.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## The crust as geometry
#
# CRUST1.0 describes the crust as eight layers, called *strata* here
# (water, ice, three of sediments and three of crystalline crust), between
# nine surfaces, on cells one degree across. `with_strata` puts strata on
# a model. Its first route treats them as *geometry*: chosen interfaces of
# the model move to chosen surfaces of the strata, and the one layer of
# the model between two moved interfaces holds the strata between their
# surfaces, each property averaged over them by thickness. The model's
# mapping stretches each point along its radius, so the Moho of the model
# lies where CRUST1.0 puts it, and a mesh of the model follows it.
#
# PREM's crust has two layers. Merging them first (`coarsened`) leaves
# one layer between the Moho and the surface; the Moho then moves to the
# bottom of CRUST1.0's crust (its last surface, index -1) and the surface
# to the top of its sediments, the solid surface beneath any water or
# ice. The shift of each moved interface is its *relief*, which
# `reliefs` gives on its own, for a method that works with the shifts of
# boundaries rather than a deformed model. The map shows the Moho's
# shift, downward (red) under the continents and most of all under Tibet
# and the Andes; the section, drawn at physical points so that the body
# is shown as it is, crosses the Andes from the Pacific, the depth
# stretched for clarity.

# %%
crust1 = datasets.load("CRUST1.0")
print(crust1)
one_crust, _ = PREM(ocean=False).coarsened(drop=["d15"])
crusted = datasets.with_strata(one_crust, crust1,
                               interfaces={"moho": -1, "surface": "upper_sediments"})
moho = datasets.reliefs(crust1, one_crust, interfaces={"moho": -1})["moho"]

fig = plt.figure(figsize=(13, 5.0))
panels = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.15])
ax = fig.add_subplot(panels[0], projection=ccrs.Robinson())
mesh = sphere_map(ax, slices.surface(one_crust, moho, "moho"), coastlines=True,
                  diverging=True)
fig.colorbar(mesh, ax=ax, orientation="horizontal", shrink=0.8,
             label="the Moho's shift from PREM's, m")
ax.set_title("the Moho moved to CRUST1.0's")
ax = fig.add_subplot(panels[1], projection="polar")
andes = slices.section(crusted, "vsv", from_geographic(-22.0, -80.0),
                       end=from_geographic(-22.0, -55.0), radii=(6301e3, 6377e3),
                       n=77, m=241, at="physical")
section(ax, andes, colorbar=True, scale=1e-3)
ax.set_rorigin(6301.0 - 100.0)                      # the depth stretched
ax.set_title("vsv across the Andes at 22 S, as the body is (km)")
fig.tight_layout()
out = FIGURES / "tutorial_14_crust.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## A lithosphere's base
#
# A code that models the lithosphere as one layer can move its base to a
# published one in the same way. LITHO1.0 gives the base of the
# lithosphere (the bottom of its last stratum, `lid`), between 9 and 320
# km deep. Here PREM's layers from 400 km to its surface are merged, an
# interface is placed at 100 km depth, and that interface moves to
# LITHO1.0's base. LITHO1.0's base changes sharply from one cell to the
# next, so its relief is read bilinearly (`lateral="linear"`): a bicubic
# reading, the default, overshoots between nodes here, which `reliefs`
# finds and refuses, naming where.

# %%
litho1 = datasets.load("LITHO1.0")
lithosphere, _ = PREM(ocean=False).coarsened(drop=["d220", "d80", "moho", "d15"])
lithosphere = lithosphere.refined([6271e3], names=["lab"])
base = datasets.reliefs(litho1, lithosphere, interfaces={"lab": litho1["lid"].bottom},
                        lateral="linear")["lab"]
print(f"the lithosphere's base moves by {base.values.min() / 1e3:.0f} to "
      f"{base.values.max() / 1e3:.0f} km from 100 km deep")
with_base = datasets.with_strata(lithosphere, litho1,
                                 interfaces={"lab": litho1["lid"].bottom},
                                 lateral="linear")
craton = from_geographic(-25.0, 125.0)                   # Western Australia
lab = with_base.geometry.interface("lab").radius
moved = with_base.geometry.mapping(cartesian_points(lab, *craton))
print(f"under Western Australia the base lies "
      f"{(6371e3 - np.linalg.norm(moved)) / 1e3:.0f} km deep")

# %% [markdown]
# ## What to cite
#
# A model records the works it rests on, and `cite()` gives them one to a
# line. PREM cites its paper, a model loaded from the registry cites what
# its source asks to be cited, and a model built from others cites theirs
# as well, in the order they were added. The model above was built from
# PREM, S40RTS and ANT-20, so it cites all three.

# %%
print(polar.cite())

# %% [markdown]
# ## Writing the result
#
# A model built this way is written to a model file like any other, and
# read back; this is the file a code such as AdGIA or sl3dpp reads.
# Writing samples the fields on the angular grid given; a grid about four
# times finer than the highest degree held keeps the fields accurate
# between its nodes.

# %%
with tempfile.TemporaryDirectory() as tmp:
    path = write_model(Path(tmp) / "prem_s40rts.nc", smooth, grid=gauss_legendre(48),
                       fields=["rho", "vsv", "vsh"])
    back = read_model(path)
    written = float(smooth.layer("lower_mantle")["vsv"](r, theta, phi))
    read = float(back.layer("lower_mantle")["vsv"](r, theta, phi))
    print(f"vsv at one point: written {written:.3f}, read back {read:.3f} m/s")
    print("the file cites what the model does:", back.citation == smooth.citation)

# %% [markdown]
# ## Another body
#
# Nothing here is particular to the Earth. The registry holds models of
# the Moon from the compilation of Garcia and others (2019), in the
# format of the TauP toolkit, which lists depths with the velocities and
# density below them and names the boundaries. VPREMOON is one: a model
# 1737.1 km in radius, with a crust, a mantle and a liquid outer core.

# %%
moon = datasets.load("VPREMOON")
print(moon)
fig, ax = plt.subplots(figsize=(5, 6))
profile(ax, [lay["vp"] for lay in moon.layers], scale=1e-3, value_scale=1e-3,
        label="vp")
profile(ax, [lay["vs"] for lay in moon.layers], scale=1e-3, value_scale=1e-3,
        label="vs")
ax.set_xlabel("velocity [km/s]")
ax.set_ylabel("radius [km]")
ax.legend()
ax.set_title("VPREMOON")
fig.tight_layout()
out = FIGURES / "tutorial_14_moon.png"
fig.savefig(out, dpi=100)
print("figure written to", out)
