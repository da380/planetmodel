# %% [markdown]
# # A random field in a box, its length scale growing with depth
#
# A Matern random field of a box 320 km across and 200 km deep
# (`cartesian.BoxGRF`), short-scaled under the surface and long-scaled
# at the base: a Fourier series laterally, over the box's period, and in
# depth the eigenfunctions of the operator each lateral wavenumber sees,
# on a mesh padded above the surface and below the base. A sample is the
# vector of its coefficients; the field on the grid's nodes and the
# `FourierField` are views of it. Here a section through one sample, the
# planes at two depths, and the power by horizontal wavenumber at those
# depths, the sample's against its expectation, which the basis states.
#
# Needs: plot
# Network: none
# Run: poetry run python examples/cartesian/11_a_random_field.py

# %%
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from planetmodel import cartesian as flat

FIGURES = Path(__file__).resolve().parents[1] / "figures"
FIGURES.mkdir(exist_ok=True)
KM = 1e-3

# %% [markdown]
# The box and its grid, 48 nodes a side, and the field: smoothness 1.5,
# a length scale of 12 km at the surface growing linearly to 36 km at
# 200 km, a standard deviation of 2 % at every depth. The mesh in depth
# has elements 10 km deep and runs 1.5 reaches beyond each end, the
# reach sqrt(8 nu) times the length scale there; the basis keeps the
# coefficients that carry all but 1e-6 of the variance.

# %%
box = flat.Box([(0.0, 320e3), (0.0, 320e3)])
grid = flat.Grid(box, (48, 48))
base = 200e3


def length_scale(depth: np.ndarray) -> np.ndarray:
    return 12e3 + 24e3 * np.asarray(depth) / base


grf = flat.BoxGRF(0.0, base, grid, 1.5, length_scale, sigma=0.02, drmax=10e3)
basis = grf.basis
print(grf)
print(f"the mesh in depth: {grf.mesh.nglob} nodes on "
      f"[{grf.mesh.rglob[0] * KM:.0f}, {grf.mesh.rglob[-1] * KM:.0f}] km, "
      f"{grf.r.size} of them in the box")
print(f"{grid.transform.size} Fourier components share {len(basis)} lateral "
      f"eigenvalues; modes kept by each: {basis.nmodes.min()} to "
      f"{basis.nmodes.max()}")

# %% [markdown]
# One sample: the coefficient vector, in the basis's order (the
# transform's components, then each one's modes in depth), and its
# values at the depth nodes over the grid. The standard deviation over
# each plane is the 2 % asked for, to the scatter of one draw.

# %%
vector = grf.sample(rng=20)
values = grf.on_grid(vector)                    # depth nodes, x, y
spread = values.std(axis=(1, 2))
print(f"{vector.size} coefficients; the standard deviation over a plane lies "
      f"between {spread.min():.4f} and {spread.max():.4f}")

# %% [markdown]
# A section along x at y = 160 km, and the planes at 10 km and 180 km:
# fine structure above, broad below.

# %%
row = int(np.argmin(np.abs(grid.y - 160e3)))
shallow = int(np.argmin(np.abs(grf.r - 10e3)))
deep = int(np.argmin(np.abs(grf.r - 180e3)))
limit = 3 * 0.02
fig, axes = plt.subplot_mosaic([["top", "top"], ["left", "right"]],
                               figsize=(11, 7.4), constrained_layout=True)
top, left, right = axes["top"], axes["left"], axes["right"]
image = top.pcolormesh(grid.x * KM, grf.r * KM, values[:, :, row], cmap="RdBu_r",
                       vmin=-limit, vmax=limit, shading="gouraud")
top.invert_yaxis()
top.set(xlabel="x [km]", ylabel="depth [km]",
        title=f"a sample, y = {grid.y[row] * KM:.0f} km")
for ax, node in ((left, shallow), (right, deep)):
    ax.pcolormesh(grid.x * KM, grid.y * KM, values[node].T, cmap="RdBu_r",
                  vmin=-limit, vmax=limit, shading="gouraud")
    ax.set(xlabel="x [km]", ylabel="y [km]", aspect="equal",
           title=f"at {grf.r[node] * KM:.0f} km depth")
fig.colorbar(image, ax=[top, left, right], label="the field", shrink=0.7)
fig.savefig(FIGURES / "cartesian_11_section.png", dpi=120)

# %% [markdown]
# The lateral spectrum. The field's Fourier components at a depth are
# independent, and those of one horizontal wavenumber share a variance,
# which the basis states: the squares of that wavenumber's covariance
# factor at the depth, summed over its modes. Against it, the sample's
# own components, their squares averaged over the components sharing a
# wavenumber and then over bins of |k|. The power moves to long
# wavelengths with depth.

# %%
components = grf.synthesise(vector)             # Fourier components, depth nodes
wavenumber = np.sqrt(basis.eigenvalues)
edges = np.linspace(0.0, wavenumber[-1], 25)
centres = 0.5 * (edges[:-1] + edges[1:])
which = np.digitize(wavenumber, edges[1:-1])
fig, ax = plt.subplots(figsize=(7, 4.4), constrained_layout=True)
for node, colour in ((shallow, "#2a78d6"), (deep, "#eb6834")):
    expected = np.array([np.sum(grf.factor(i)[node] ** 2)
                         for i in range(len(basis))])
    drawn = np.array([np.mean(components[basis.components(i), node] ** 2)
                      for i in range(len(basis))])
    count = basis.multiplicity
    binned = (np.bincount(which, weights=count * drawn, minlength=centres.size)
              / np.bincount(which, weights=count, minlength=centres.size))
    ax.semilogy(wavenumber[1:] / KM, expected[1:], color=colour,
                label=f"expected, {grf.r[node] * KM:.0f} km")
    ax.semilogy(centres / KM, binned, "o", ms=4, color=colour,
                label=f"the sample, {grf.r[node] * KM:.0f} km")
    total = np.sum(count * expected) / grid.area
    print(f"at {grf.r[node] * KM:.0f} km the expected power sums to a variance "
          f"of {total:.6g}, sigma^2 = {0.02 ** 2:.6g}")
ax.set(xlabel="|k| [rad/km]", ylabel="variance of a Fourier component",
       title="the lateral spectrum at two depths")
ax.legend()
fig.savefig(FIGURES / "cartesian_11_spectrum.png", dpi=120)

# %% [markdown]
# The same vector as a field of the library: a `FourierField` on the
# depth interval, exact at the nodes and a polynomial between them.

# %%
field = grf.to_field(vector, name="dlnvs")
x, y = grid.nodes
miss = np.abs(field.evaluate(grf.r[deep], x + 0.0, y + 0.0) - values[deep]).max()
print(f"{field!r} agrees with the grid values at {grf.r[deep] * KM:.0f} km "
      f"to {miss:.1e}")
