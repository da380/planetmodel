"""mesh.py -- conforming GLL meshes over an interval and over a model.

Mesh1D is pure geometry: elements, nodes, global numbering and reference
data, with pinned breakpoints and an exact piecewise-polynomial view of
nodal functions.  RadialMesh lays such a mesh over (part of) a model so
that no element straddles a Skeleton boundary, carrying the containing
layer index per element -- which is what preserves the two one-sided
values at every discontinuity, since nodal arrays are stored per element
rather than per global node.

Everything here is dimensional SI; non-dimensionalisation belongs to the
assembly stage.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial import Polynomial
from scipy.interpolate import PPoly

from ..model.body import ReferenceBody
from .gll import gll_points_weights, lagrange_derivative_matrix
from .gravity import G_NEWTON, gravity

__all__ = ["Mesh1D", "RadialMesh"]


class Mesh1D:
    """A conforming GLL mesh over an interval with pinned breakpoints.

    The span between each pair of consecutive breakpoints is subdivided
    uniformly into elements no wider than drmax (one element per span
    when drmax is None), so every breakpoint is an element boundary and
    no element straddles one.

    Geometry attributes: nspec elements with arrays left, right, jac
    (= half-widths) of shape (nspec,), node coordinates r of shape
    (nspec, ngll), and a global numbering gmap of shape (nspec, ngll)
    into nglob = nspec (ngll - 1) + 1 shared nodes with coordinates
    rglob.  Reference-element data: xi, w, deriv (see
    lagrange_derivative_matrix for the index convention).

    This class is pure geometry; RadialMesh layers a ReferenceBody's Skeleton,
    per-layer material evaluation and gravity on top of it.
    """

    def __init__(self, breakpoints, *, ngll: int = 5,
                 drmax: float | None = None) -> None:
        """Build the mesh from strictly increasing breakpoints."""
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
        """Index of the element whose interval contains coordinate x.

        Coordinates on a shared boundary resolve to the element above
        it, so the returned element's left endpoint is <= x; arguments
        outside the mesh clip to the first or last element.
        """
        i = int(np.searchsorted(self.left, x, side="right")) - 1
        return min(max(i, 0), self.nspec - 1)

    def to_ppoly(self, nodal, *, elements: tuple[int, int] | None = None) -> PPoly:
        """Exact piecewise-polynomial (scipy PPoly) view of nodal values.

        `nodal` has shape (nspec, ngll); `elements`, when given, is a
        contiguous half-open (start, stop) element range restricting
        the view.  Within each element the result is the interpolating
        polynomial through the GLL nodes, so it reproduces the
        spectral-element function exactly and its .derivative /
        .integrate / .antiderivative are those of the SEM interpolant.
        Nodal arrays are per element, so discontinuous data is
        representable; at a shared breakpoint PPoly evaluates the upper
        element's polynomial (except at the final endpoint, which is
        evaluated from the last element).
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
        """Compact summary: elements, nodes and coordinate range."""
        return (f"Mesh1D({self.nspec} elements x {self.ngll} GLL, "
                f"{self.nglob} nodes, "
                f"x in [{self.left[0]:g}, {self.right[-1]:g}])")


class RadialMesh(Mesh1D):
    """A GLL mesh over [rmin, rmax] of a radial ReferenceBody.

    Element boundaries comprise every Skeleton boundary inside the
    range plus a uniform subdivision of each layer into elements no
    wider than drmax, so no element straddles a discontinuity and each
    carries the index of its containing layer.  Passing lmax instead of
    drmax applies the Al-Attar & Tromp resolution rule
    drmax = 0.1 rmax / (lmax + 1), sized for the largest degree of an
    intended sweep.  rmax below the surface builds a truncated model
    (e.g. rmax = 6368 km strips PREM's ocean and makes the upper crust
    the loaded surface).

    Geometry (left, right, jac, r, gmap, nglob, rglob, xi, w, deriv)
    is inherited from Mesh1D; on top of it each element records the
    index of its model layer in the array `layer`.

    Nodal material arrays are per element, shape (nspec, ngll), so the
    two one-sided values of a discontinuous property at a shared node
    are both retained -- adjoining elements evaluate their own layer's
    function there.

    `edges=` is the third way to say where the elements are, and the one
    a *reader* needs: a file records the element boundaries it used, not
    the rule that produced them, so `planetmodel.io.netcdf.read` hands the
    edges over verbatim rather than trying to invert drmax.
    """

    def __init__(self, model: ReferenceBody, *, ngll: int = 5,
                 drmax: float | None = None, lmax: int | None = None,
                 rmin: float | None = None, rmax: float | None = None,
                 edges=None) -> None:
        """Build the mesh; give exactly one of drmax, lmax or edges."""
        given = sum(x is not None for x in (drmax, lmax, edges))
        if given != 1:
            raise ValueError("give exactly one of drmax, lmax and edges")
        sk = model.skeleton
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
                raise ValueError("mesh range must lie within the model")
            # No element may straddle a discontinuity: that is the one
            # invariant the drmax path gets for free and this one must
            # check, since it is what makes `layer` well defined.
            tol = 1e-9 * (e[-1] - e[0])
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
                raise ValueError("mesh range must lie within the model")
            if drmax is None:
                if lmax < 0:
                    raise ValueError("lmax must be non-negative")
                drmax = 0.1 * r1 / (lmax + 1)
            if drmax <= 0.0:
                raise ValueError("drmax must be positive")

            inner = b[(b > r0) & (b < r1)]
            super().__init__(np.concatenate(([r0], inner, [r1])),
                             ngll=ngll, drmax=drmax)

        self.model = model
        mid = 0.5 * (self.left + self.right)
        self.layer = np.array([sk.locate(m).layer for m in mid], dtype=int)
        self._nodal: dict = {}
        self._gravity: tuple[float, np.ndarray] | None = None

    # -- material fields ----------------------------------------------------

    def nodal(self, name: str) -> np.ndarray:
        """Nodal values of model field `name`, shape (nspec, ngll), cached.

        Each element evaluates its own layer's function, so one-sided
        values at discontinuities are preserved on the shared nodes.
        """
        if name not in self._nodal:
            self._nodal[name] = self._per_element(self.model[name], name)
        return self._nodal[name]

    def nodal_derivative(self, name: str, *, nu: int = 1) -> np.ndarray:
        """Nodal values of the nu-th radial derivative of field `name`."""
        key = ("d", name, nu)
        if key not in self._nodal:
            self._nodal[key] = self._per_element(
                self.model[name].derivative(nu), f"d{nu} {name} / dr{nu}")
        return self._nodal[key]

    def _per_element(self, fld, name: str) -> np.ndarray:
        """`fld` on every element's nodes, each from its own layer's piece.

        A field belongs to the layers of its domain, and a mesh may
        reach layers it does not cover -- a body whose
        core alone carries a density, meshed whole.  The piece is asked
        for by layer, and the refusal is restated here so that it names
        the field, the element and the layer rather than only the
        field's domain: what the caller must change is which mesh or
        which field, and both are in the message.
        """
        out = np.empty((self.nspec, self.ngll))
        for e in range(self.nspec):
            try:
                piece = fld[self.layer[e]]
            except ValueError as exc:
                raise ValueError(
                    f"field {name!r} is not defined on layer "
                    f"{int(self.layer[e])}, which element {e} of this mesh "
                    f"lies in ([{self.left[e]:.6g}, {self.right[e]:.6g}]): "
                    f"{exc}") from None
            out[e] = piece(self.r[e])
        out.setflags(write=False)
        return out

    def nodal_gravity(self, *, n: int = 8, G: float = G_NEWTON) -> np.ndarray:
        """Nodal g(r), shape (nspec, ngll), cached.

        Computed with gravity() over the *full* model, so a truncated
        mesh (rmin > 0 or rmax below the surface) still sees the
        gravitational field of everything beneath it -- though note
        that stripping layers by rmax means their mass is genuinely
        absent from the model only if the model itself was rebuilt.
        Here the model is used as-is: mass below/above the mesh range
        that belongs to the model still contributes to M(r).
        """
        key = (float(G), int(n))
        if self._gravity is None or self._gravity[0] != key:
            g = gravity(self.model, self.r, n=n, G=G)
            g.setflags(write=False)
            self._gravity = (key, g)
        return self._gravity[1]

    @property
    def is_fluid(self) -> np.ndarray:
        """Per-element fluid flags.

        An element is fluid where its layer supports no shear stress:
        `Layer.is_fluid`, which is fluid *or* vacuum, a void carrying no
        shear either.  The layer's state is the one source: readers
        classify it on construction and `classify_states` overrides it,
        so nothing here re-derives it from a velocity.
        """
        key = ("is_fluid",)
        if key not in self._nodal:
            fluid = np.array([lay.is_fluid for lay in self.model.layers],
                             dtype=bool)
            flags = fluid[self.layer]
            flags.setflags(write=False)
            self._nodal[key] = flags
        return self._nodal[key]

    # -- degree-dependent truncation ---------------------------------------

    def truncation_radius(self, l: int, *, eps: float = 1e-8) -> float:
        """Radius below which a degree-l solution is negligible.

        Uses the interior decay phi ~ (r/rmax)^(l+1): the returned
        radius satisfies (r/rmax)^(l+1) = eps, matching the
        spheroidal_start rule of the reference Fortran implementation.
        """
        if l < 0:
            raise ValueError("degree must be non-negative")
        return float(self.right[-1] * eps ** (1.0 / (l + 1)))

    def element_at(self, r: float) -> int:
        """Index of the element whose interval contains radius r.

        Radii at a shared boundary resolve to the element above it, so
        the returned element's left endpoint is <= r; truncating a
        solve there errs on the side of a slightly larger domain.
        """
        return self.element_of(r)

    def start_element(self, l: int, *, eps: float = 1e-8) -> int:
        """First element of the sub-mesh used for a degree-l solve."""
        return self.element_at(self.truncation_radius(l, eps=eps))

    #: The older names, kept for `loading.py` until it leaves.
    first_element = element_at
    first_element_for = start_element

    def __repr__(self) -> str:
        """Compact summary: elements, nodes, range, and fluid count."""
        nf = int(self.is_fluid.sum())
        return (f"RadialMesh({self.nspec} elements x {self.ngll} GLL, "
                f"{self.nglob} nodes, r in [{self.left[0]:g}, {self.right[-1]:g}], "
                f"{nf} fluid elements)")
