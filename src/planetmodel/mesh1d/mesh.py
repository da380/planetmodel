"""Conforming GLL meshes over an interval and over a skeleton.

Mesh1D is pure geometry on an interval: pinned breakpoints, each span
between them cut uniformly into elements no wider than `drmax`, `ngll`
Gauss-Lobatto-Legendre nodes per element, a global node numbering, the
reference-element data a weak form assembles with, and an exact
piecewise-polynomial view of nodal values.  Nodal arrays are stored per
element, shape (nspec, ngll), so a shared node carries one value from
each side.

RadialMesh lays such a mesh over a Skeleton so that every skeleton
boundary is an element boundary, and records the layer each element
lies in.  Given a model, `nodal` evaluates the field of each element's
own layer at the element's nodes, so a shared node carries both
one-sided values, and `nodal_gravity` the model's gravity there.
Everything is numbers: the mesh knows nothing about units.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PPoly

from ..skeleton import Skeleton
from .gll import gll_points_weights, lagrange_derivative_matrix

__all__ = ["Mesh1D", "RadialMesh"]


class Mesh1D:
    """A conforming GLL mesh over an interval with pinned breakpoints.

    Each span between consecutive breakpoints is subdivided uniformly
    into elements no wider than `drmax` (one element per span when
    `drmax` is None).  Attributes: `nspec` elements with `left`,
    `right` and `jac` (half-widths) of shape (nspec,); node coordinates
    `r` of shape (nspec, ngll); the global numbering `gmap` of shape
    (nspec, ngll) into `nglob = nspec (ngll - 1) + 1` shared nodes with
    coordinates `rglob`; and the reference-element data `xi`, `w`,
    `deriv` (see `lagrange_derivative_matrix` for the index convention).
    """

    def __init__(self, breakpoints, *, ngll: int = 5,
                 drmax: float | None = None) -> None:
        b = np.array(breakpoints, dtype=float)
        if b.ndim != 1 or b.size < 2:
            raise ValueError("need a 1-d array of at least two breakpoints")
        if not np.all(np.diff(b) > 0.0):
            raise ValueError("breakpoints must be strictly increasing")
        if drmax is not None and drmax <= 0.0:
            raise ValueError("drmax must be positive")
        step = np.inf if drmax is None else float(drmax)

        left, right = [], []
        for lo, hi in zip(b[:-1], b[1:]):
            nel = max(1, int(np.ceil((hi - lo) / step)))
            edges = np.linspace(lo, hi, nel + 1)
            left.extend(edges[:-1])
            right.extend(edges[1:])

        b.setflags(write=False)
        self.breakpoints = b
        self.ngll = int(ngll)
        self.drmax = step
        self.left = np.asarray(left)
        self.right = np.asarray(right)
        self.nspec = self.left.size
        self.jac = 0.5 * (self.right - self.left)

        self.xi, self.w = gll_points_weights(self.ngll)
        self.deriv = lagrange_derivative_matrix(self.xi)

        self.r = self.left[:, None] + (self.xi[None, :] + 1.0) * self.jac[:, None]
        self.r[:, 0] = self.left      # exact shared endpoints
        self.r[:, -1] = self.right

        idx = (np.arange(self.nspec)[:, None] * (self.ngll - 1)
               + np.arange(self.ngll)[None, :])
        self.gmap = idx
        self.nglob = self.nspec * (self.ngll - 1) + 1
        self.rglob = np.empty(self.nglob)
        self.rglob[self.gmap] = self.r

    def element_of(self, x: float) -> int:
        """The element whose interval contains x.

        A coordinate on a shared boundary resolves to the element above
        it; a coordinate outside the mesh clips to the first or last
        element.
        """
        i = int(np.searchsorted(self.left, x, side="right")) - 1
        return min(max(i, 0), self.nspec - 1)

    def to_ppoly(self, nodal, *, elements: tuple[int, int] | None = None) -> PPoly:
        """The exact piecewise-polynomial (scipy PPoly) view of nodal values.

        `nodal` has shape (nspec, ngll); `elements`, when given, is a
        half-open (start, stop) element range restricting the view.
        Within each element the result is the interpolating polynomial
        through the GLL nodes, so its derivative and integral are those
        of the spectral-element function.  Discontinuous data is
        representable; at a shared breakpoint the upper element's
        polynomial is evaluated.
        """
        nodal = np.asarray(nodal, dtype=float)
        if nodal.shape != (self.nspec, self.ngll):
            raise ValueError(f"nodal values must have shape "
                             f"{(self.nspec, self.ngll)}, got {nodal.shape}")
        e0, e1 = (0, self.nspec) if elements is None else map(int, elements)
        if not 0 <= e0 < e1 <= self.nspec:
            raise ValueError("elements must be a non-empty in-range interval")
        n = self.ngll
        V = np.vander(self.xi, n, increasing=True)
        cxi = np.linalg.solve(V, nodal[e0:e1].T)      # coefficients in xi
        C = np.zeros((n, e1 - e0))
        for k, e in enumerate(range(e0, e1)):
            # substitute xi = (x - left) / jac - 1 into the xi-polynomial
            p = Polynomial(cxi[:, k])(Polynomial([-1.0, 1.0 / self.jac[e]]))
            C[n - p.coef.size:, k] = p.coef[::-1]     # PPoly: degree-descending
        x = np.concatenate((self.left[e0:e1], self.right[e1 - 1:e1]))
        return PPoly(C, x)

    def __repr__(self) -> str:
        return (f"Mesh1D({self.nspec} elements x {self.ngll} GLL, "
                f"{self.nglob} nodes, "
                f"x in [{self.left[0]:g}, {self.right[-1]:g}])")


class RadialMesh(Mesh1D):
    """A GLL mesh over [rmin, rmax] of a skeleton.

    Exactly one of `drmax`, `lmax` and `edges` sizes the mesh.  With
    `drmax` or `lmax` the element boundaries are every skeleton boundary
    strictly inside the range plus a uniform subdivision no coarser than
    `drmax`, where `lmax` sets `drmax = 0.1 rmax / (lmax + 1)`; `rmin`
    and `rmax` default to the skeleton's ends.  With `edges` the element
    boundaries are taken as given, must lie inside the skeleton, and no
    element may straddle a skeleton boundary (to `rtol` of the range).
    `layer` records the skeleton layer of every element.  A Geometry is
    accepted in place of a skeleton.
    """

    def __init__(self, skeleton, *, ngll: int = 5,
                 drmax: float | None = None, lmax: int | None = None,
                 rmin: float | None = None, rmax: float | None = None,
                 edges=None, rtol: float = 1e-9) -> None:
        sk = getattr(skeleton, "skeleton", skeleton)
        if not isinstance(sk, Skeleton):
            raise TypeError(f"expected a Skeleton or a Geometry, got "
                            f"{type(skeleton).__name__}")
        given = sum(x is not None for x in (drmax, lmax, edges))
        if given != 1:
            raise ValueError("give exactly one of drmax, lmax and edges")
        b = sk.boundaries
        if edges is not None:
            if rmin is not None or rmax is not None:
                raise ValueError(
                    "edges= already fixes the range: give rmin/rmax with "
                    "drmax or lmax instead")
            e = np.array(edges, dtype=float)
            if e.ndim != 1 or e.size < 2:
                raise ValueError("edges must be a 1-d array of at least two "
                                 "element boundaries")
            if not (b[0] <= e[0] < e[-1] <= b[-1]):
                raise ValueError("mesh range must lie within the skeleton")
            tol = rtol * (e[-1] - e[0])
            for lo, hi in zip(e[:-1], e[1:]):
                inside = b[(b > lo + tol) & (b < hi - tol)]
                if inside.size:
                    raise ValueError(
                        f"the element [{lo:.6g}, {hi:.6g}] straddles the "
                        f"skeleton boundary at {inside[0]:.6g}")
            super().__init__(e, ngll=ngll)
        else:
            r0 = float(b[0] if rmin is None else rmin)
            r1 = float(b[-1] if rmax is None else rmax)
            if not (b[0] <= r0 < r1 <= b[-1]):
                raise ValueError("mesh range must lie within the skeleton")
            if drmax is None:
                if lmax < 0:
                    raise ValueError("lmax must be non-negative")
                drmax = 0.1 * r1 / (lmax + 1)
            if drmax <= 0.0:
                raise ValueError("drmax must be positive")
            inner = b[(b > r0) & (b < r1)]
            super().__init__(np.concatenate(([r0], inner, [r1])),
                             ngll=ngll, drmax=drmax)

        self.skeleton = sk
        mid = 0.5 * (self.left + self.right)
        self.layer = np.array([sk.locate(m).layer for m in mid], dtype=int)

    def truncation_radius(self, l: int, *, eps: float = 1e-8) -> float:
        """The radius below which a degree-l solution is negligible.

        With the interior decay (r / rmax)^(l + 1), the returned radius
        satisfies (r / rmax)^(l + 1) = eps.
        """
        if l < 0:
            raise ValueError("degree must be non-negative")
        return float(self.right[-1] * eps ** (1.0 / (l + 1)))

    def element_at(self, r: float) -> int:
        """The element whose interval contains r; a boundary resolves upward."""
        return self.element_of(r)

    def start_element(self, l: int, *, eps: float = 1e-8) -> int:
        """The first element of the sub-mesh for a degree-l solve."""
        return self.element_at(self.truncation_radius(l, eps=eps))

    def nodal(self, model, name: str, *, nu: int = 0,
              missing: str = "refuse") -> np.ndarray:
        """A radial field of a model at the nodes, per element: (nspec, ngll)
        for rank 0, with the stored components appended for higher rank.

        Each element takes the field of its own layer, so a node on a
        skeleton boundary carries both one-sided values; `nu` asks for
        the nu-th radial derivative.  An element whose layer lacks the
        name is refused by name, or filled with NaN when
        `missing="nan"`.  Complex where any layer's field is complex.
        A new array each call.
        """
        if missing not in ("refuse", "nan"):
            raise ValueError(f"missing must be 'refuse' or 'nan', got {missing!r}")
        pieces = {}
        for i in np.unique(self.layer):
            layer = model.layer(int(i))
            if name not in layer:
                if missing == "nan":
                    continue
                raise KeyError(
                    f"layer {int(i)} ({layer.name!r}) holds no field {name!r}; "
                    "pass missing='nan' to fill NaN there")
            field = layer[name]
            if not getattr(field, "is_radial", False):
                raise ValueError(
                    f"{name!r} on layer {int(i)} depends on direction; nodal "
                    "values are for radial fields")
            if nu:
                field = field.derivative(nu=nu)
            pieces[int(i)] = field.evaluate(self.r[self.layer == i], 0.0, 0.0)
        if not pieces:
            raise KeyError(f"no layer of the mesh holds {name!r}")
        first = next(iter(pieces.values()))
        dtype = np.result_type(*(v.dtype for v in pieces.values()))
        out = np.full((self.nspec, self.ngll) + first.shape[2:], np.nan, dtype=dtype)
        for i, values in pieces.items():
            out[self.layer == i] = values
        return out

    def nodal_gravity(self, model) -> np.ndarray:
        """The model's gravity at the nodes, per element, over the whole model."""
        from .gravity import gravity
        return gravity(model, self.r)

    def __repr__(self) -> str:
        return (f"RadialMesh({self.nspec} elements x {self.ngll} GLL, "
                f"{self.nglob} nodes, r in [{self.left[0]:g}, {self.right[-1]:g}])")
