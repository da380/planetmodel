# %% [markdown]
# # viscosity3d: a 3D viscosity model for a GIA code
#
# From S40RTS to a model file a GIA code reads: PREM given a radial
# viscosity and an elastic lid, S40RTS put on its mantle without degree
# 0, the shear-speed anomaly taken through sl3d's chain to a viscosity
# (or a published viscosity model in its place), a thicker lid where the
# viscosity stays high, power-law creep and a Burgers body's Prony
# branches if asked, the model written with the works it rests on, read
# back, drawn, and read by the C++ library's example reader when its
# path is given. See README.md.
#
# Needs: netcdf, maps
# Network: first run (S40RTS, 3 MB)
# Run: poetry run python examples/miniapps/viscosity3d/viscosity3d.py

# %%
import argparse
import subprocess
import tempfile
from pathlib import Path

import cartopy.crs as ccrs
import matplotlib.pyplot as plt
import numpy as np

import planetmodel as pm
from planetmodel import datasets
from planetmodel.conversion import building, lithosphere, parameters, prony
from planetmodel.mesh1d import RadialMesh
from planetmodel.modelfile import read_model, write_model
from planetmodel.plotting import sphere_map

FIGURES = Path(__file__).resolve().parents[2] / "figures"
LAWS = {"sl3d": parameters.SL3D_VISCOSITY, "milne": parameters.MILNE_2018_VISCOSITY}

parser = argparse.ArgumentParser(description="S40RTS to a 3D viscosity model file")
parser.add_argument("--law", choices=sorted(LAWS), default="sl3d",
                    help="eps of the exponential law: sl3d's 0.01 or Milne's 0.04 "
                         "per K")
parser.add_argument("--renormalise", action="store_true",
                    help="keep each depth's mean viscosity the radial one")
parser.add_argument("--burgers", action="store_true",
                    help="a Burgers body (Kelvin modulus fraction 0.2, viscosity "
                         "ratio 5) in place of the Maxwell viscosity")
parser.add_argument("--lmax", type=int, default=None,
                    help="filter S40RTS to this degree (all 40 by default)")
parser.add_argument("--published", default=None,
                    help="a registry viscosity model in place of S40RTS's chain, as "
                         "GLAD-M25-viscosity or SMEAN2-VM5a-b1")
parser.add_argument("--lid", type=float, default=None,
                    help="a viscosity (Pa s): above the depth, within 400 km, where "
                         "the viscosity falls to it, a lid of 1e30 Pa s")
parser.add_argument("--power-law", type=float, nargs=2, default=None,
                    metavar=("N", "SIGMA"),
                    help="power-law creep: its stress exponent and transition stress "
                         "(Pa)")
parser.add_argument("--band", type=int, default=96,
                    help="the band of the Gauss-Legendre grid the file holds (96)")
parser.add_argument("--out", type=Path, default=None,
                    help="the model file to write (a temporary one by default)")
parser.add_argument("--cpp", type=Path, default=None,
                    help="the C++ example planetmodel_read_model, to read the file")
args = parser.parse_args()

# %% [markdown]
# ## The radial model
#
# PREM without its ocean; viscosity 2e21 Pa s below 670 km and 5e20 Pa s
# above, up to the base of PREM's lid, 80 km deep. The lid and the crust
# hold no viscosity, so they stay elastic.

# %%
model = pm.PREM(ocean=False)
for lay in model.layers:
    lo, hi = lay.interval
    if 3480e3 <= lo and hi <= 6291e3:
        model = model.with_field(lay.index, "viscosity", pm.constant_field(
            (lo, hi), 2e21 if hi <= 5701e3 else 5e20, name="viscosity"))
mantle = [lay.index for lay in model.layers if "viscosity" in lay]
print("viscous layers:", [model.layer(i).name for i in mantle])

# %% [markdown]
# ## S40RTS on the mantle, then the viscosity
#
# Degree 0 is removed, so the radial model's mean stays its own.

# %%
if args.published is None:
    model = datasets.transplanted(model, "S40RTS", remove_mean=True, layers=mantle,
                                  lmax=args.lmax)
    dt = building.temperature_anomaly(building.velocity_anomaly(model, layers=mantle),
                                      scaling=parameters.SL3D_DENSITY_SCALING,
                                      expansivity=parameters.SL3D_EXPANSIVITY)
    model = building.with_viscosity(model, dt, LAWS[args.law],
                                    renormalise=pm.gauss_legendre(32)
                                    if args.renormalise else None)
else:
    model = datasets.transplanted(model, args.published,
                                  names={"viscosity": "viscosity"}, mode="absolute",
                                  kind="log", layers=mantle, extend=True)

# %% [markdown]
# ## A thicker lid, and power-law creep, if asked
#
# Where the viscosity stays above `--lid` going down from the lid's
# base, as under the old continents, the lithosphere goes on: the
# viscosity above the depth where it first falls to that value, within
# 400 km, is raised to 1e30 Pa s.

# %%
if args.lid is not None:
    upper = [i for i in mantle if model.layer(i).interval[0] >= 5971e3]
    base = lithosphere.viscosity_threshold(model, args.lid, layers=upper,
                                           grid=pm.equiangular(90, 180))
    depth = (6371e3 - base.function.values) / 1e3
    print(f"the lid's base from {depth.min():.0f} to {depth.max():.0f} km deep")
    model = lithosphere.with_lid(model, base, viscosity=1e30)
if args.power_law is not None:
    model = building.with_power_law(model, exponent=args.power_law[0],
                                    transition_stress=args.power_law[1])
names = ["rho", "vpv", "vph", "vsv", "vsh", "eta", "viscosity"]
if args.power_law is not None:
    names += ["stress_exponent", "transition_stress"]
if args.burgers:
    model = prony.with_branches(model, prony.burgers_series(
        kelvin_modulus_fraction=0.2, viscosity_ratio=5.0))
    names = [n for n in names if n != "viscosity"] + [
        f"relaxation_{k}_{n}" for n in (1, 2) for k in ("strength", "time")]

# %% [markdown]
# ## The model file, and back
#
# The laterally varying fields are sampled on a Gauss-Legendre grid of
# band 96 at the nodes of 50 km elements and read bicubically: S40RTS
# goes to degree 40, and its exponential is sharper still, so a bilinear
# rule on band 64 misses the viscosity by up to 40 %, where this holds
# log10 eta to about 0.002 (README.md). The file keeps the works the model
# rests on.

# %%
lateral = [n for n in names if n in ("vsv", "vsh", "viscosity")
           or n.startswith("relaxation_time")]
with tempfile.TemporaryDirectory() as tmp:
    path = args.out or Path(tmp) / "viscosity3d.nc"
    write_model(path, model, grid=pm.gauss_legendre(args.band), fields=names,
                radial=RadialMesh(model.geometry, drmax=50e3, ngll=5),
                interpolation={n: ("polynomial", "cubic") for n in lateral})
    print(f"wrote {path} ({path.stat().st_size / 1e6:.0f} MB)")
    back = read_model(path)
    i = model.layer("lower_mantle").index
    rng = np.random.default_rng(1)
    lo, hi = model.layer(i).interval
    r, t = rng.uniform(lo, hi, 500), np.arccos(rng.uniform(-1, 1, 500))
    p = rng.uniform(0, 2 * np.pi, 500)
    name = "relaxation_time_1" if args.burgers else "viscosity"
    a, b = model.layer(i)[name](r, t, p), back.layer(i)[name](r, t, p)
    print(f"{name} read back in the lower mantle: log10 within "
          f"{np.max(np.abs(np.log10(b / a))):.1e} of the model at 500 points")
    print("the file cites:", *back.citation, sep="\n  ")
    if args.cpp is not None:
        out = subprocess.run([str(args.cpp), str(path)], capture_output=True,
                             text=True, check=True).stdout
        print("the C++ reader says:", *out.splitlines()[:6], sep="\n  ")

# %% [markdown]
# ## Maps
#
# log10 of the viscosity over the radial one at three depths.

# %%
if not args.burgers:
    FIGURES.mkdir(exist_ok=True)
    fig = plt.figure(figsize=(12, 4))
    for k, depth in enumerate((250e3, 1000e3, 2500e3)):
        ax = fig.add_subplot(1, 3, k + 1, projection=ccrs.Robinson())
        at = pm.slices.sphere(model, "viscosity", depth=depth,
                              relative_to="reference", departure="log10")
        sphere_map(ax, at, colorbar=True, diverging=True, coastlines=True)
        ax.set_title(f"{depth / 1e3:.0f} km deep")
    fig.tight_layout()
    fig.savefig(FIGURES / "viscosity3d.png", dpi=100)
    print("figure written to", FIGURES / "viscosity3d.png")
