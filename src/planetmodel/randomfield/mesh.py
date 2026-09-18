"""Padded computational meshes, and the physical interval inside one.

A field wanted on a physical interval [r1, r2] is computed on a longer
one.  The operator A = 1 - div(kappa grad) needs a boundary condition
at each end of its mesh, and whichever is chosen distorts the field
within about a correlation length of the boundary; extending the mesh
beyond r1 and r2 by a fictitious pad moves the distortion out of the
interval that matters.  `padded_mesh` builds such a mesh with r1 and r2
pinned as breakpoints, so that the physical interval is a run of whole
elements and a contiguous slice of the global nodes; `restriction`
finds that run and slice, and a `Restriction` holds them.  Everything
"physical" is then the full-mesh quantity followed by the slice: there
is no second mesh and no second set of eigenfunctions.

How long a pad should be is the caller's business.  For a Matern field
of smoothness nu and length scale lambda the natural unit is the
effective range sqrt(8 nu) lambda, the distance at which the
correlation has fallen to about a tenth, which `matern_reach` returns.

The measure decides what happens at the centre.  With the r^2 dr measure
of an annulus or ball the inner pad stops at r = 0, which is a regular
point of the operator and needs neither padding nor a boundary
condition; with the plain measure dr the mesh is an interval of the
line and may extend anywhere.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..mesh1d import Mesh1D
from .operator import WEIGHTS

__all__ = ["matern_reach", "padded_mesh", "Restriction", "restriction"]

#: How closely, relative to the length of the physical interval, its
#: endpoints must sit on element boundaries of the mesh.
_ENDPOINT_RTOL = 1e-12


def matern_reach(nu: float, lam: float) -> float:
    """The effective range sqrt(8 nu) lam of a Matern field of smoothness
    `nu` and length scale `lam`."""
    if nu <= 0.0 or lam <= 0.0:
        raise ValueError("nu and lam must be positive")
    return float(np.sqrt(8.0 * nu) * lam)


def padded_mesh(r1: float, r2: float, *, pad: float | tuple[float, float],
                weight: str = "r2", ngll: int = 5,
                drmax: float | None = None) -> Mesh1D:
    """The mesh of [r1 - pad_lo, r2 + pad_hi] with r1 and r2 pinned.

    `pad` is one length for both ends or a pair (lower, upper); a zero
    pad leaves that end of the physical interval as the mesh boundary.
    `weight` is the measure the mesh is for, "r2" or "one" as
    `RadialOperatorFamily` takes it: under "r2" the interval must lie in
    r >= 0 and the lower pad stops at r = 0.  `ngll` and `drmax` are
    those of `Mesh1D`, `drmax=None` meaning one element per span.
    """
    if weight not in WEIGHTS:
        raise ValueError(f"weight must be one of {WEIGHTS}")
    r1, r2 = float(r1), float(r2)
    if not r1 < r2:
        raise ValueError("need r1 < r2")
    lo, hi = (pad, pad) if np.ndim(pad) == 0 else pad
    lo, hi = float(lo), float(hi)
    if lo < 0.0 or hi < 0.0 or not np.isfinite(lo + hi):
        raise ValueError("pads must be non-negative and finite")
    a, b = r1 - lo, r2 + hi
    if weight == "r2":
        if r1 < 0.0:
            raise ValueError("the r^2 measure needs r1 >= 0")
        a = max(0.0, a)
    return Mesh1D(np.unique([a, r1, r2, b]), ngll=ngll, drmax=drmax)


@dataclass(frozen=True, eq=False)
class Restriction:
    """The physical interval inside a padded mesh.

    `interval` is (r1, r2); `elements` the half-open range of the mesh's
    elements that make it up, as `Mesh1D.to_ppoly` takes it; `nodes` the
    contiguous slice of the global nodes lying in it; `r` those nodes'
    radii, read-only.
    """

    interval: tuple[float, float]
    elements: tuple[int, int]
    nodes: slice
    r: np.ndarray


def restriction(mesh: Mesh1D, r1: float, r2: float) -> Restriction:
    """The `Restriction` of `mesh` to [r1, r2], whose endpoints must be
    element boundaries of the mesh (as `padded_mesh` makes them)."""
    r1, r2 = float(r1), float(r2)
    if not r1 < r2:
        raise ValueError("need r1 < r2")
    e0 = int(np.argmin(np.abs(mesh.left - r1)))
    e1 = int(np.argmin(np.abs(mesh.right - r2)))
    tol = _ENDPOINT_RTOL * (r2 - r1)
    if abs(mesh.left[e0] - r1) > tol or abs(mesh.right[e1] - r2) > tol:
        raise ValueError(f"the endpoints {r1:g} and {r2:g} are not both element "
                         f"boundaries of the mesh")
    nodes = slice(int(mesh.gmap[e0, 0]), int(mesh.gmap[e1, -1]) + 1)
    r = mesh.rglob[nodes].copy()
    r.setflags(write=False)
    return Restriction((r1, r2), (e0, e1 + 1), nodes, r)
