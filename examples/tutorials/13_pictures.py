# %% [markdown]
# # 13. Pictures of a model
#
# A picture of a three-dimensional model is made in two steps. The first
# is to choose the points to look at and evaluate a field there: a
# sphere at one depth, a cross-section through the planet, or a line
# down one radius. The result is a *slice*, plain numbers that know what
# they are. The second step is to draw the slice. The library keeps the
# two apart. `planetmodel.slices` makes slices of any model and needs
# nothing but numpy and scipy; `planetmodel.plotting` draws them with
# matplotlib, and maps with cartopy.
#
# This tutorial builds PREM with a perturbation of its shear velocity in
# the lower mantle, and then draws it in every way the library offers: a
# map at one depth, cross-sections, profiles, the spectrum of the
# perturbation, a comparison with a second model, and a cutaway, the
# body with an octant removed. It ends by writing
# the model as files for ParaView, a program for looking at
# three-dimensional data interactively, and says how to start with it.
#
# It needs the `plot` extra (matplotlib) and the `maps` extra (cartopy)
# for the maps. The figures are written to `examples/figures/`, and the
# ParaView files to
# `examples/figures/tutorial_13_vtk/`.

# %%
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import CubicSpline

from planetmodel import PREM, HarmonicField, flattening, gauss_legendre, perturbed
from planetmodel import slices
from planetmodel.harmonics import packing
from planetmodel.plotting import (correlation, correlation_total, cutaway, profile,
                                  section, spectrum, sphere_map, write_vtk)

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The model
#
# The perturbation is a *relative* one: the perturbed velocity is `vsv *
# (1 + delta)`, where `delta` is a sum of spherical harmonics up to
# degree 12 whose coefficients vary with radius. It spans the three
# layers of PREM's lower mantle, from the core-mantle boundary at 3480
# km radius up to 5701 km, and fades to nothing at both ends. The
# amplitude falls with the degree, so the large-scale pattern dominates,
# as it does in the Earth's lower mantle.
#
# A second model, *B*, shares the same pattern up to degree 6 and has
# independent structure above it. It stands in for a second published
# model of the same region, and is used at the end for a comparison.

# %%
prem = PREM()
lo, hi = 3480e3, 5701e3
L = 12
s, l, m = packing(L)
rng = np.random.default_rng(13)
knots = np.linspace(lo, hi, 6)
taper = np.sin(np.linspace(0.0, np.pi, 6))


def coefficients(pattern):
    """A pattern of coefficients at every knot, tapered to zero at the ends."""
    return CubicSpline(knots, taper[:, None, None, None] * pattern, axis=0)


pattern = np.zeros((2, L + 1, L + 1))
pattern[s, l, m] = 0.012 * rng.standard_normal(s.size) / (1.0 + l)
pattern[0, 0, 0] = 0.0                             # no change to the mean
other = pattern.copy()
high = l > 6
other[s[high], l[high], m[high]] = (0.012 * rng.standard_normal(high.sum())
                                    / (1.0 + l[high]))

model = perturbed(prem, "vsv", HarmonicField((lo, hi), coefficients(pattern)),
                  kind="relative")
model_b = perturbed(prem, "vsv", HarmonicField((lo, hi), coefficients(other)),
                    kind="relative").named("model B")
print(model)

# %% [markdown]
# ## A map at one depth
#
# `slices.sphere` evaluates a field on a sphere. Here the sphere lies
# 1500 km below the surface, and the field is shown as its *departure*
# from the spherical mean at that depth, in percent, which is how maps
# of seismic velocity are usually drawn. The angular grid of points is
# one degree in latitude and longitude unless another is given. The
# slice carries its values, its grid, the positions of its points, and
# a label saying what it shows.
#
# `plotting.sphere_map` draws the slice on a cartopy map. Any cartopy
# projection works; here the Robinson projection for the whole globe and
# an orthographic view over the north pole. A departure is drawn on a
# red-blue colour map, with the limits symmetric about zero, so that
# white means no departure; negative values, slower than average, are
# red, as is usual for seismic velocity. `coastlines=True` would add
# the coastlines, which cartopy downloads the first time; they are left
# out here because the pattern is not the Earth's.

# %%
depth = 1500e3
mapped = slices.sphere(model, "vsv", depth=depth, relative_to="mean")
print("a sphere slice:", mapped.values.shape, "values |", mapped.label)

fig = plt.figure(figsize=(12, 4.5))
ax = fig.add_subplot(1, 2, 1, projection=ccrs.Robinson())
sphere_map(ax, mapped, colorbar=True)
ax.set_title(f"vsv {depth / 1e3:.0f} km deep")
ax = fig.add_subplot(1, 2, 2, projection=ccrs.Orthographic(0, 60))
sphere_map(ax, mapped)
ax.set_title("the same, over the north pole")
fig.tight_layout()
out = FIGURES / "tutorial_13_maps.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# A sphere at the radius of an interface has two sets of values, one
# just below it and one just above. At the core-mantle boundary the
# liquid outer core lies below and the rock of the mantle above, and
# the shear velocity is zero on one side and seven kilometres a second
# on the other. A slice there must be told which side to take, and is
# refused when it is not. Away from an interface the side has no effect.

# %%
try:
    slices.sphere(model, "vsv", radius=3480e3)
except ValueError as exc:
    print("refused:", exc)
below = slices.sphere(model, "vsv", radius=3480e3, side="below")
above = slices.sphere(model, "vsv", radius=3480e3, side="above")
print(f"vsv just below the core-mantle boundary {below.values.mean():.0f} m/s, "
      f"just above {above.values.mean():.0f} m/s")

# %% [markdown]
# ## Cross-sections
#
# `slices.section` evaluates a field on a cross-section through the
# centre, along a great circle: the path on the surface that a plane
# through the centre cuts. The section is given by a starting point and
# either an end point or a direction and a length, all angles in
# radians, with the starting point as a colatitude and a longitude.
# Here it starts on the equator at longitude 0 and heads east for half a
# turn. `plotting.section` draws it on matplotlib's polar axes as a
# wedge of the planet, the distance along the path across and the radius
# outward, with each layer drawn on its own so that a jump in the field
# at an interface stays sharp. A section of a whole turn is a disc: the
# second panel is the plane through the north pole and the equator,
# showing the absolute velocity, with the liquid outer core at zero.

# %%
equator, east = (0.5 * np.pi, 0.0), 0.5 * np.pi
mantle = slices.section(model, "vsv", equator, azimuth=east, length=np.pi,
                        radii=(lo, 6371e3), relative_to="mean")
disc = slices.section(model, "vsv", equator, azimuth=0.0, length=2 * np.pi,
                      m=361)
fig = plt.figure(figsize=(12, 5))
ax = fig.add_subplot(1, 2, 1, projection="polar")
section(ax, mantle, scale=1e-3, colorbar=True)
ax.set_rticks([4000, 5000, 6000])
ax.set_title("the mantle along the equator, heading east")
ax = fig.add_subplot(1, 2, 2, projection="polar")
section(ax, disc, scale=1e-3, colorbar=True)
ax.set_title("a whole disc through the poles")
fig.tight_layout()
out = FIGURES / "tutorial_13_sections.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# A model's fields live on its *reference* body, the sphere it was built
# on; a mapping carries the reference body to the physical one, which
# may be flattened or carry topography on its boundaries. A slice is
# taken at reference points unless asked otherwise, and gives besides
# where the mapping takes each of its points. With `physical=True` the
# section is drawn where those points are, so that the body appears in
# its real shape. The flattening here is fifty times the Earth's, to be
# seen; the outer boundary and the core-mantle boundary both become
# ellipses.

# %%
flat = model.stretched(flattening(50.0 / 298.25))
pole_to_equator = slices.section(flat, "vsv", (1e-3, 0.0), end=(0.5 * np.pi, 0.0),
                                 m=91)
fig, ax = plt.subplots(figsize=(6, 5))
section(ax, pole_to_equator, scale=1e-3, physical=True, colorbar=True)
ax.set_title("pole to equator, the body flattened")
fig.tight_layout()
out = FIGURES / "tutorial_13_flattened.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## Profiles
#
# `slices.profile` evaluates a field along one radius, from the centre to
# the surface at a chosen point, and `slices.mean_profile` gives the
# spherical mean at every radius instead. `plotting.profile` draws either
# with the radius upward and a horizontal step at every interface. A
# perturbation of a percent is too small to see against the velocity
# itself, so the second panel shows the departure from the mean along
# the same radius, which is zero outside the lower mantle.

# %%
point = slices.profile(model, "vsv", 1.0, 2.0)
mean = slices.mean_profile(model, "vsv")
departure = slices.profile(model, "vsv", 1.0, 2.0, relative_to="mean")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(9, 6), sharey=True)
profile(ax, mean, scale=1e-3, label="spherical mean", color="0.5", lw=3)
profile(ax, point, scale=1e-3, label="at colatitude 1, longitude 2")
ax.set_xlabel("vsv [m/s]")
ax.set_ylabel("radius [km]")
ax.legend(loc="lower right")
profile(bx, departure, scale=1e-3)
bx.set_xlabel(departure.label)
fig.tight_layout()
out = FIGURES / "tutorial_13_profiles.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## Spectra
#
# A field on a sphere can be written as a sum of spherical harmonics,
# the analogue of a Fourier series on the sphere, and its *spectrum*
# says how much of it lies at each degree: degree 0 is the mean, degree 1
# a pattern with one half faster than the other, and higher degrees
# finer patterns. `slices.spectrum` takes a field on a sphere to degree
# `lmax` and gives its *power* at each degree, which adds up over the
# degrees to the average of the field's square over the sphere. The
# library's harmonics are orthonormal, each integrating to one over the
# sphere when squared. The spectra here are of the departure from the
# mean, at three depths, up to degree 12, where the perturbation stops.
# Each coefficient's amplitude was made to fall as 1 / (1 + l), but a
# degree has 2l + 1 coefficients, so the power of a degree falls only
# slowly after the first few. The depth of 1800 km, midway through the
# lower mantle, has the most power, as the taper makes it.

# %%
fig, ax = plt.subplots(figsize=(6, 4))
for d in (1000e3, 1800e3, 2500e3):
    sp = slices.spectrum(model, "vsv", depth=d, lmax=12, relative_to="mean")
    spectrum(ax, sp, first=1, label=f"{d / 1e3:.0f} km", marker="o", ms=3)
    print(f"{d / 1e3:.0f} km: power summing to {sp.power.sum():.3g} (percent squared)")
ax.set_xlabel("degree")
ax.set_ylabel("power [percent squared]")
ax.legend()
fig.tight_layout()
out = FIGURES / "tutorial_13_spectra.png"
fig.savefig(out, dpi=100)
print("figure written to", out)

# %% [markdown]
# ## Comparing two models
#
# `slices.correlation` compares a field of two models at a list of radii:
# at each radius and degree it gives the correlation of the two patterns,
# 1 where they are the same up to a factor, 0 where they are unrelated
# and -1 where one is the other reversed. Model B shares the pattern up
# to degree 6 and nothing above, so the picture shows correlation near 1
# for the low degrees and scattered values above. The total over all
# degrees, weighted by their power, is drawn beside it against radius.

# %%
radii = np.linspace(lo + 100e3, hi - 100e3, 12)
compared = slices.correlation(model, "vsv", model_b, lmax=12, radii=radii,
                              relative_to="mean")
fig, (ax, bx) = plt.subplots(1, 2, figsize=(11, 4.5), width_ratios=[3, 1])
correlation(ax, compared, scale=1e-3, colorbar=True)
ax.set_xlabel("degree")
ax.set_ylabel("radius [km]")
correlation_total(bx, compared, scale=1e-3, marker="o", ms=3)
bx.set_xlabel("total correlation")
fig.tight_layout()
out = FIGURES / "tutorial_13_comparison.png"
fig.savefig(out, dpi=100)
print("figure written to", out)
print("correlation at degrees 1-6:", np.nanmean(compared.by_degree[:, 1:7]).round(3),
      "| at degrees 7-12:", np.nanmean(compared.by_degree[:, 7:]).round(3))

# %% [markdown]
# ## A cutaway
#
# The classic picture of a planet's interior is the body seen from
# outside with a wedge cut from it, the field on the faces of the cut
# and the surface around them. `slices.cutaway` makes one: the wedge is
# a span of longitudes, here a quarter turn, and a span of colatitudes,
# here the northern hemisphere, so that an octant is removed; the cut
# stops at a radius, here the core-mantle boundary, so that the core is
# left standing in it as a sphere. The picture is an orthographic view
# toward a point given by its colatitude and longitude, north up,
# rendered pixel by pixel: each pixel's ray is cast into the cut body
# and the field is evaluated where it first lands, so that the slice
# knows which face every pixel sees, and what hides what is settled
# exactly. `plotting.cutaway` draws it on plain axes. The faces always
# show the field; the surface shows the field too (the departure is
# zero there, so it is white), or a plain colour, with the interfaces as
# faint arcs on the faces and the core in a plain grey. Unless `view=`
# says otherwise the viewer is 25 degrees north of the equator and a
# little east of the cut's first face, which puts the cut at the upper
# right of the picture with one face nearly square-on, the other a
# sliver, and the face on the equator seen from above. For the Earth
# the surface can also wear Natural Earth's relief, or its land and sea,
# with the coastlines, which tutorial 14 shows on S40RTS. `n` is the
# number of pixels across the picture, and the figure is saved at a
# resolution that keeps them.

# %%
cut = slices.cutaway(model, "vsv", longitudes=(0.0, 0.5 * np.pi),
                     colatitudes=(0.0, 0.5 * np.pi), inner=lo, n=1200,
                     relative_to="mean", mean_grid=gauss_legendre(32))
print(cut.label, "| faces seen:",
      [name for name in slices.CUTAWAY_FACES if cut.mask(name).any()])
fig, (ax, bx) = plt.subplots(1, 2, figsize=(13, 6), layout="constrained")
cutaway(ax, cut, scale=1e-3, inner="0.7", colorbar=True)
ax.set_title("the northern octant removed, the field at the surface")
cutaway(bx, cut, scale=1e-3, surface="0.9", inner="0.7")
bx.set_title("the surface plain")
out = FIGURES / "tutorial_13_cutaway.png"
fig.savefig(out, dpi=200)
print("figure written to", out)

# %% [markdown]
# ## Files for ParaView
#
# ParaView is a free program for looking at three-dimensional data: it
# cuts a model open, shows surfaces of equal value, and turns the result
# in the hand. `plotting.write_vtk` writes a model in VTK's format, which
# ParaView reads: one file per layer, gathered by one `.vtm` file. Each
# layer is a grid of points, its radii by an angular grid, with the
# fields at every point, a `layer` number for picking layers out, and
# here the velocity's departure from its mean as a field of its own. The
# points are written where the mapping puts them, so a flattened model
# is written flattened. A coarse grid keeps the files small.
#
# To look at the model in ParaView:
#
# 1. *File > Open*, choose `tutorial_13.vtm`, and press *Apply* in the
#    Properties panel. The planet appears as a solid ball.
# 2. In the toolbar above the view, choose `vsv_percent_from_mean` to
#    colour by, and press the *Rescale to data range* button beside it.
# 3. *Filters > Common > Clip* cuts the ball open with a plane; press
#    *Apply*, and drag the plane in the view, or untick *Show Plane* to
#    hide it.
# 4. *Filters > Common > Threshold* on `layer` keeps a range of layers:
#    layers 2 to 4 are the lower mantle, where the perturbation is.
# 5. *Filters > Common > Slice* shows the field on a plane through the
#    centre, a cross-section like the ones above, which can be turned
#    freely.
#
# Each filter acts on whatever is selected in the Pipeline Browser on
# the left, so filters can be applied one after another, and the eye
# icon beside each step shows or hides it.

# %%
vtk = write_vtk(FIGURES / "tutorial_13_vtk" / "tutorial_13", model,
                grid=gauss_legendre(24), n=6, fields=["vsv", "rho"],
                departures={"vsv": "mean"})
print("ParaView file written to", vtk)
