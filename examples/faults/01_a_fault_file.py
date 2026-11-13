# %% [markdown]
# # A fault file
#
# A fault of two patches, written to a fault file and read back: a
# megathrust swept from an arcuate trench at a dip steepening with
# depth, with a tapered thrust and two slices of afterslip down dip of
# it, and a plane, a normal fault in the outer rise. The slip is held in
# fault components (strike-slip, dip-slip, opening), so the vector it
# rebuilds at each point follows the curved interface: drawn as arrows,
# on the map and in a section down dip, it lies in the surface.
#
# Needs: netcdf, plot
# Network: none
# Run: poetry run python examples/faults/01_a_fault_file.py

# %%
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from planetmodel import Scales
from planetmodel.fault import (Fault, GeographicFrame, Plane, Slip, SmallCircle,
                               StructuredPatch)
from planetmodel.faultfile import read_fault_file, write_fault

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
EARTH = GeographicFrame(6371e3)
YEAR = 3.15576e7

# %% [markdown]
# The megathrust: a trench 8 degrees from a pole at (38N, 150E), 60
# degrees of turn about it, the interface descending towards the pole at
# 10 degrees at the trench steepening to 30 at 60 km. Its slip, cumulative
# at 0, 1 and 5 years: an 8 m thrust tapered to nothing at its ends and
# its base, then afterslip of 1.5 m and 3 m centred down dip of it. The
# plane: 2 m of normal slip, at once, on a fault seaward of the trench.

# %%
U, V = np.meshgrid(np.linspace(-0.5, 0.5, 9), np.linspace(-0.5, 0.5, 7), indexing="ij")
taper = np.cos(np.pi * U) ** 2
coseismic = 8.0 * taper * np.cos(0.5 * np.pi * (V + 0.5)) ** 2
after = taper * np.exp(-(((V - 0.25) / 0.15) ** 2))
dip_slip = np.stack([coseismic, coseismic + 1.5 * after, coseismic + 3.0 * after])
trace = SmallCircle((38.0, 150.0), 8.0, 230.0, 290.0)
slab = StructuredPatch.swept(EARTH, trace, dip=lambda z: 10.0 + z / 3e3, bottom=60e3,
                             n_u=9, n_v=7,
                             slip=Slip.from_rake(90.0, dip_slip, rule="cubic"))
lat, lon, _ = EARTH.coordinates(slab.cartesian(0.0, -0.5))
outer = Plane.at_depth(EARTH, lat - 0.25, lon - 0.9, 8e3, strike=160.0, dip=60.0,
                       length=100e3, width=16e3,
                       slip=Slip.from_rake(-90.0, np.full((3, 1, 1), 2.0)))
fault = Fault([slab, outer], support="node", time_interpolation="linear",
              time=[0.0, 1.0, 5.0], scales=Scales(time=YEAR))

# %% [markdown]
# Written and read back: the same numbers, so the same slip anywhere.

# %%
with tempfile.TemporaryDirectory() as tmp:
    path = write_fault(Path(tmp) / "fault.nc", fault, source="planetmodel example",
                       names=["megathrust", "outer rise"],
                       meta={"time_origin": "the mainshock"})
    data = read_fault_file(path)
back = data.fault
rng = np.random.default_rng(0)
u, v = rng.uniform(-0.5, 0.5, (2, 1000))
for k in range(2):
    assert np.array_equal(back.slip_vector(k, u, v, date=3.0),
                          fault.slip_vector(k, u, v, date=3.0))
print(data.names, back)
slip = back.slip_vector(0, u, v, date=3.0, basis="cartesian")
print("largest slip along the megathrust's normal (m):",
      f"{np.max(np.abs(np.einsum('pi,pi->p', slip, slab.normal(u, v)))):.1e}")

# %% [markdown]
# On the map, the dip-slip at five years and the slip's horizontal part
# as arrows; down dip of the middle of the trench, a section through
# the interface with the slip in that plane, at 0 and 5 years; and the
# fault components along that section in time.

# %%
fig = plt.figure(figsize=(12, 7.5), layout="constrained")
grid = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0])
ax = fig.add_subplot(grid[:, 0])
Us, Vs = np.meshgrid(np.linspace(-0.5, 0.5, 61), np.linspace(-0.5, 0.5, 31),
                     indexing="ij")
pos = back.patches[0].position(Us, Vs)
mesh = ax.pcolormesh(pos[..., 1], pos[..., 0],
                     back.slip_vector(0, Us, Vs, date=5.0, basis="fault")[..., 1],
                     cmap="Blues", shading="gouraud")
fig.colorbar(mesh, ax=ax, label="dip-slip at 5 yr (m)", shrink=0.6)
ax.plot(pos[:, 0, 1], pos[:, 0, 0], color="#0b0b0b", lw=2, label="trench")
a, b = np.meshgrid(np.linspace(-0.4, 0.4, 9), np.linspace(-0.45, 0.45, 7),
                   indexing="ij")
at = back.patches[0].position(a, b)
east, north, _ = np.moveaxis(back.slip_vector(0, a, b, date=5.0), -1, 0)
ax.quiver(at[..., 1], at[..., 0], east, north, color="#d64b2a", scale=60,
          label="slip, east and north")
corners = back.patches[1].position([-0.5, 0.5, 0.5, -0.5, -0.5],
                                   [-0.5, -0.5, 0.5, 0.5, -0.5])
ax.plot(corners[:, 1], corners[:, 0], color="#52514e", label="outer-rise plane")
ax.set(xlabel="longitude", ylabel="latitude", aspect=1 / np.cos(np.deg2rad(lat)))
ax.legend(loc="lower left")

ax = fig.add_subplot(grid[0, 1])
w = np.linspace(-0.5, 0.5, 201)
x = back.patches[0].cartesian(0.0 * w, w)
up0 = x[0] / np.linalg.norm(x[0])
across = back.patches[0].fault_basis(0.0, -0.5)[:, 0]
across = np.cross(across, up0)
section = lambda p: np.stack([(p - x[0]) @ across, (p - x[0]) @ up0], axis=-1) / 1e3
curve = section(x)
ax.plot(curve[:, 0], curve[:, 1], color="#0b0b0b", lw=1.5, label="the interface")
for date, colour in ((0.0, "#2a78d6"), (5.0, "#d64b2a")):
    vs = np.linspace(-0.45, 0.45, 10)
    s = back.slip_vector(0, 0.0 * vs, vs, date=date, basis="cartesian")
    base = section(back.patches[0].cartesian(0.0 * vs, vs))
    ax.quiver(base[:, 0], base[:, 1], s @ across, s @ up0, color=colour, scale=25,
              angles="xy", label=f"slip at {date:g} yr")
ax.set(xlabel="distance from the trench (km)", ylabel="height (km)", aspect="equal")
ax.legend(loc="lower left", fontsize="small")

ax = fig.add_subplot(grid[1, 1])
for date, colour in ((0.0, "#2a78d6"), (1.0, "#7a9cc6"), (5.0, "#d64b2a")):
    comp = back.slip_vector(0, 0.0 * w, w, date=date, basis="fault")
    ax.plot(w, comp[:, 1], color=colour, label=f"dip-slip at {date:g} yr")
ax.plot(w, comp[:, 0], color="#a3a29b", ls="--", label="strike-slip and opening")
ax.plot(w, comp[:, 2], color="#a3a29b", ls="--")
ax.set(xlabel="v, down dip", ylabel="slip (m)")
ax.legend(fontsize="small")
fig.savefig(FIGURES / "faults_01.png", dpi=110)
print("figure written to", FIGURES / "faults_01.png")
