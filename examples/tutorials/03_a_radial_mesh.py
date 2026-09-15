# %% [markdown]
# # 3. A radial mesh
#
# A one-dimensional spectral-element mesh is one way a skeleton is handed
# to a solver. It is a chain of elements along the radius, each carrying
# a set of Gauss-Lobatto-Legendre (GLL) nodes, with every skeleton
# boundary an element boundary so that no element straddles a
# discontinuity. The mesh is geometry only. Tutorial 7 shows how a
# model's fields are evaluated on its nodes.
#
# This tutorial plots, so it needs the `plot` extra (matplotlib). The
# figure is written to `examples/figures/`.

# %%
from pathlib import Path

import numpy as np

from planetmodel import RadialMesh, Skeleton
from planetmodel.mesh1d import Mesh1D, gll_points_weights

FIGURES = Path(__file__).resolve().parent.parent / "figures"
FIGURES.mkdir(exist_ok=True)

# %% [markdown]
# ## The reference element
#
# Every element is a copy of the reference interval `[-1, 1]`, which
# carries `ngll` GLL nodes: the two endpoints plus the zeros of the
# derivative of a Legendre polynomial. Each node has a quadrature weight,
# and the weighted sum of a polynomial's values at the nodes gives its
# integral exactly up to degree `2 ngll - 3`.

# %%
xi, w = gll_points_weights(5)
print("nodes  :", np.round(xi, 4))
print("weights:", np.round(w, 4), "| sum", w.sum())

# %% [markdown]
# ## A mesh over a skeleton
#
# `RadialMesh` lays elements over a skeleton. Exactly one of three
# arguments sets the element size: `drmax`, the largest element width
# allowed; `lmax`, a spherical harmonic degree, from which
# `drmax = 0.1 rmax / (lmax + 1)` follows; or `edges`, the element
# boundaries themselves. Every skeleton boundary is an element boundary
# whichever is given, and each element records the layer it lies in.

# %%
sk = Skeleton([0.0, 0.19, 0.55, 0.99, 1.0])
mesh = RadialMesh(sk, ngll=5, drmax=0.1)
print(mesh)
print("elements per layer:", np.bincount(mesh.layer))
print("first element:", mesh.left[0], "to", mesh.right[0])
print("thin crust gets its own element:", mesh.left[-1], "to", mesh.right[-1])

# %% [markdown]
# Values on the nodes are stored per element, in arrays of shape
# `(nspec, ngll)` with `nspec` the number of elements. A node shared by
# two elements therefore appears twice, once from each side, which is
# what lets a field that jumps at a boundary keep both of its values
# there. `gmap` gives each node a single global number, which a solver
# uses when it assembles one system of equations over the whole mesh.

# %%
print("nodal array shape:", mesh.r.shape, "| global nodes:", mesh.nglob)
e = mesh.element_at(0.55)
print(f"element at the CMB: {e}, its first node r = {mesh.r[e, 0]}")
print(f"the element below ends at r = {mesh.r[e - 1, -1]}")
print("global index of that shared node:", mesh.gmap[e - 1, -1], mesh.gmap[e, 0])

# %% [markdown]
# ## The exact polynomial view
#
# Values on the nodes define, element by element, the polynomial that
# passes through them. `to_ppoly` returns that piecewise polynomial as a
# scipy `PPoly`, which can be evaluated, differentiated and integrated
# exactly. What a spectral-element solver computes with is exactly what
# you get back.
#
# The example is a function that jumps at the core-mantle boundary. It
# is evaluated element by element, using the formula of each element's
# own layer, so the node shared at the boundary gets a value from each
# side.

# %%
core = lambda r: 12.0 - 8.0 * r**2
mantle = lambda r: 5.5 - 2.0 * r
in_core = (mesh.layer < 2)[:, None]
nodal = np.where(in_core, core(mesh.r), mantle(mesh.r))
print("both sides of the CMB:", nodal[e - 1, -1], nodal[e, 0])

P = mesh.to_ppoly(nodal)
r = np.array([0.1, 0.3, 0.5, 0.549, 0.551, 0.7, 0.9])
exact = np.where(r < 0.55, core(r), mantle(r))
print("P(r)  :", np.round(P(r), 4))
print("exact :", np.round(exact, 4))
print(
    "integral over the mantle:",
    P.integrate(0.55, 0.99),
    "| exact:",
    5.5 * 0.44 - (0.99**2 - 0.55**2),
)

# %% [markdown]
# ## Truncation for a degree-l solve
#
# A solution of spherical harmonic degree `l` decays towards the centre
# like `(r / a)^(l + 1)`, with `a` the outer radius. `truncation_radius(l)`
# is the radius at which that factor falls below a small threshold, 1e-8
# by default, and `start_element(l)` is the first element a degree-`l`
# solve needs to include. At high degree most of the mesh can be skipped.

# %%
fine = RadialMesh(sk, ngll=5, lmax=64)
print(fine)
for l in (2, 8, 32, 64):
    print(
        f"l = {l:3d}: start at r = {fine.truncation_radius(l):.3f}, "
        f"element {fine.start_element(l)} of {fine.nspec}"
    )

# %% [markdown]
# ## A mesh without a skeleton
#
# `Mesh1D` is the same machinery on a bare interval with chosen
# breakpoints, for a solver that has no skeleton at all.

# %%
print(Mesh1D([0.0, 1.0, 3.0], ngll=4, drmax=0.5))

# %% [markdown]
# ## A picture
#
# The elements along the radius, coloured by layer, with their GLL nodes,
# and the piecewise polynomial through the jumping function above.

# %%
try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:
    print("matplotlib is not installed; no figure")
    raise SystemExit(0)

fig, (top, bottom) = plt.subplots(
    2, 1, figsize=(9, 5), sharex=True, gridspec_kw={"height_ratios": [1, 2]}
)
colours = plt.rcParams["axes.prop_cycle"].by_key()["color"]
for e in range(mesh.nspec):
    c = colours[mesh.layer[e] % len(colours)]
    top.plot(
        [mesh.left[e], mesh.right[e]], [0, 0], color=c, lw=6, solid_capstyle="butt"
    )
    top.plot(mesh.r[e], np.zeros(mesh.ngll), "k.", ms=3)
for b in sk.boundaries:
    top.axvline(b, color="0.6", lw=0.8)
top.set_yticks([])
top.set_title("elements by layer, GLL nodes as dots")

rr = np.linspace(0.0, 1.0, 600)
bottom.plot(rr, P(rr), lw=1.5, label="to_ppoly")
bottom.plot(mesh.r.ravel(), nodal.ravel(), "k.", ms=3, label="nodal values")
for b in sk.boundaries:
    bottom.axvline(b, color="0.6", lw=0.8)
bottom.set_xlabel("r")
bottom.legend()
fig.tight_layout()
fig.savefig(FIGURES / "tutorial_03_radial_mesh.png", dpi=110)
print("figure written to", FIGURES / "tutorial_03_radial_mesh.png")

# %% [markdown]
# The next tutorial hands a geometry to the 3D mesher.
