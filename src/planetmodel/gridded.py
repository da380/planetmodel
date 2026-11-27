"""Stored fields that vary with direction: values on a grid at GLL nodes.

A `GriddedField` is a field on one interval held as numbers: the
interval is cut into radial elements at `breaks`, each element carries
`ngll` Gauss-Lobatto-Legendre nodes, and at every node the field is
given on one `AngularGrid`, so that `values` has shape ::

    (nelem, ngll, ntheta, nphi) + stored shape,

the stored components in the spherical frame at each node (Voigt for
the ranks that have it), which is a `Sample`'s layout cut to one layer.
Between the nodes it is read radially by one of `mesh1d.RADIAL_METHODS`
within the element holding the radius (the Lagrange polynomial through
its GLL nodes, or straight lines between consecutive nodes), and
laterally by one of the rules of `planetmodel.lateral` at every node of
that element.  A table of depth slices joined by straight
lines is `ngll = 2` with the breaks at the slices.  The end elements
continue beyond the interval, as a polynomial layer's end pieces do,
which is what `on_interval` asks of them.

Components of rank 1 and above are read in the Cartesian frame, where
a field is single-valued at a pole, and turned back into the frame
asked for at the point.  The values at the poles follow the rule of
`lateral.polar_values` and have one source each: a pole that is a node
of the grid is its own row, and a pole the grid lacks is given in
`poles`, of shape (nelem, ngll, 2) + stored shape with the components
Cartesian, or found from the grid's expansion when the grid has a band;
`poles` holds both poles once resolved.  `GriddedField.sampled` builds
one from any field, evaluating its values at the poles exactly.

A `GriddedFunction` is the same thing for a function of (theta, phi)
alone: values on a grid, its values at the poles, a lateral rule, and
the two angular derivatives, so that it serves as a relief of
`layer_linear` and as a surface quantity such as the thickness of a
lithosphere.  A `GriddedDisplacement` is a mapping held the same way:
its displacement m(X) - X a gridded vector field on every layer of a
skeleton, its deformation gradient the derivatives of the rules that
read it, and every radius its radial rule's derivative may jump at
declared as a knot: the breakpoints of each field's `radial_pieces`,
the element boundaries under the polynomial rule and every node under
the linear one.

`GriddedBase` is the radial half, shared with the gridded fields of a
box (`cartesian.GriddedField`): the elements, the radial rule, the
values' checks, the derivatives and the means are its, the grid, the
lateral rule over it and the frame of the components the leaf's.

Everything is numbers and every stored array is read-only.
"""
from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly

from .character import SCALAR, VECTOR, Character
from .displacement import Relief
from .fields import Field, FieldBase, RadialField, rotate_stored, stored_shape
from .frames import cartesian_points, spherical_coordinates, spherical_frame
from .layerfunction import PolynomialLayer, as_interval
from .lateral import METHODS, LateralInterpolant, check_rule, polar_values
from .mapping import R_FLOOR_FRACTION, SIN_FLOOR, Mapping, MappingBase
from .mesh1d.mesh import RADIAL_METHODS, Mesh1D, RadialMesh
from .skeleton import Skeleton

if TYPE_CHECKING:
    from .sampling import AngularGrid

__all__ = ["GriddedBase", "GriddedField", "GriddedFunction", "GriddedDisplacement"]


def _readonly(a: np.ndarray) -> np.ndarray:
    """A read-only float64 copy."""
    a = np.array(a, dtype=float)
    a.setflags(write=False)
    return a


def _grid_frames(grid: AngularGrid) -> np.ndarray:
    """The spherical frame at every node of the grid, (ntheta, nphi, 3, 3)."""
    return spherical_frame(grid.colatitudes[:, None], grid.longitudes[None, :])


class GriddedBase(FieldBase):
    """The radial half every gridded field shares, on the sphere or in a
    box: the interval cut into elements at `breaks`, `ngll` GLL nodes
    each, a radial rule of `mesh1d.RADIAL_METHODS` along the layered
    coordinate, and at every node of an element the leaf's lateral rule
    over its grid.

    A leaf's constructor calls `_hold`, which checks and stores the
    interval, the breaks and the values, and supplies `_interpolant(e)`,
    the lateral rule over the nodes of element e built once: called at
    the two lateral coordinates it gives `(ngll,) + stored shape + P`,
    and its `gradient` the two lateral derivatives alike, and `mean()`
    its mean over the lateral domain.  `derivatives` and the means are
    the radial rule's weights against them.  The values the rule reads
    are the leaf's: Cartesian components on the sphere, where the local
    frame turns, and the local ones in a box, where it does not.
    """

    def _hold(self, interval: tuple[float, float], breaks: ArrayLike,
              values: ArrayLike, lateral: tuple[int, ...], *,
              axes: tuple[str, ...], character: Character, name: str | None,
              radial: str, rtol: float) -> np.ndarray:
        """Check and store the interval, the breaks, the radial rule and the
        values, of shape (nelem, ngll) + `lateral` + stored shape, `axes`
        naming the lateral axes in a refusal; the values as float64."""
        if radial not in RADIAL_METHODS:
            raise ValueError(f"radial must be one of {RADIAL_METHODS}, got {radial!r}")
        self._interval = as_interval(interval)
        self._character = character
        self._name = name
        self._rtol = float(rtol)
        lo, hi = self._interval
        b = np.array(breaks, dtype=float)
        if b.ndim != 1 or b.size < 2 or not np.all(np.diff(b) > 0.0):
            raise ValueError("breaks must be a strictly increasing 1-d array of at "
                             f"least two values of the {self.coordinates.labels[0]}")
        tol = self._rtol * (hi - lo)
        if abs(b[0] - lo) > tol or abs(b[-1] - hi) > tol:
            raise ValueError(f"breaks from {b[0]:g} to {b[-1]:g} do not span the "
                             f"interval [{lo:g}, {hi:g}]")
        if np.iscomplexobj(values):
            raise TypeError("a gridded field holds real values")
        v = np.asarray(values, dtype=float)
        c = stored_shape(character)
        lead = (b.size - 1,)
        if v.ndim != 2 + len(lateral) + len(c) or v.shape[:1] != lead \
                or v.shape[1] < 2 or v.shape[2:] != tuple(lateral) + c:
            raise ValueError(
                f"values must have shape (nelem, ngll, {', '.join(axes)}) + {c} = "
                f"{lead + ('ngll',) + tuple(lateral) + c} with ngll >= 2, "
                f"got {v.shape}")
        if not np.all(np.isfinite(v)):
            raise ValueError("a gridded field holds finite values")
        self._breaks = _readonly(b)
        self._data = _readonly(v)
        self._radial = radial
        self._mesh = Mesh1D(b, ngll=v.shape[1])
        self._interpolants: dict[int, object] = {}
        return v

    def _interpolant(self, e: int) -> LateralInterpolant:
        """The lateral rule over the nodes of element e, built once."""
        raise NotImplementedError

    @property
    def breaks(self) -> np.ndarray:
        """The element boundaries along the layered coordinate."""
        return self._breaks

    @property
    def values(self) -> np.ndarray:
        """The stored components at the nodes, (nelem, ngll) + the grid's
        shape + stored shape, in the local frame."""
        return self._data

    @property
    def radial(self) -> str:
        """The radial rule, one of `mesh1d.RADIAL_METHODS`."""
        return self._radial

    @property
    def lateral(self) -> str:
        """The lateral rule, one of the leaf's lateral methods."""
        return self._lateral

    @property
    def ngll(self) -> int:
        """The GLL nodes per element."""
        return self._data.shape[1]

    @property
    def radii(self) -> np.ndarray:
        """The layered coordinate of every node, (nelem, ngll): its radius on
        the sphere, its depth in a box."""
        return self._mesh.r

    def _values_at(self, r: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """The components the lateral rule reads, at flat points, (P,) +
        stored shape."""
        e, W = self._mesh.interpolation_weights(r, method=self._radial)
        out = np.empty((r.size,) + self.stored_shape)
        for k in np.unique(e):
            m = e == k
            lat = self._interpolant(int(k))(a[m], b[m])          # (ngll, *c, Pk)
            out[m] = np.einsum("pi,i...p->p...", W[m], lat)
        return out

    def derivatives(self, r: ArrayLike, a: ArrayLike, b: ArrayLike
                    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """The components the lateral rule reads (Cartesian on the sphere,
        local in a box) at flat points, and their derivatives along the
        layered coordinate and the two lateral ones, each (P,) + stored
        shape: the radial rule's weights and their derivatives against the
        lateral rule's values and gradient at the nodes of the point's
        element.  Values beyond the breaks continue the end elements;
        nothing is checked against the interval."""
        r, a, b = (np.asarray(v, dtype=float).ravel()
                   for v in np.broadcast_arrays(r, a, b))
        e, W = self._mesh.interpolation_weights(r, method=self._radial)
        _, dW = self._mesh.interpolation_weights(r, method=self._radial, nu=1)
        out = [np.empty((r.size,) + self.stored_shape) for _ in range(4)]
        for k in np.unique(e):
            m = e == k
            f = self._interpolant(int(k))
            lat = f(a[m], b[m])                               # (ngll, *c, Pk)
            ga, gb = f.gradient(a[m], b[m])
            out[0][m] = np.einsum("pi,i...p->p...", W[m], lat)
            out[1][m] = np.einsum("pi,i...p->p...", dW[m], lat)
            out[2][m] = np.einsum("pi,i...p->p...", W[m], ga)
            out[3][m] = np.einsum("pi,i...p->p...", W[m], gb)
        return out[0], out[1], out[2], out[3]

    def radial_pieces(self) -> tuple[np.ndarray, int]:
        """The breakpoints and degree that hold the field exactly on its
        interval: the breaks inside it and ngll - 1 under the polynomial
        rule, the nodes inside it and one under the linear rule, with the
        interval's ends."""
        linear = self._radial == "linear"
        x = np.unique(self._mesh.r) if linear else self._breaks
        lo, hi = self._interval
        tol = self._rtol * (hi - lo)
        inside = x[(x > lo + tol) & (x < hi - tol)]
        return np.concatenate([[lo], inside, [hi]]), 1 if linear else self.ngll - 1

    def _lateral_mean(self, make: type[RadialField]) -> RadialField:
        """The mean over the lateral domain as exact polynomial pieces, a
        field of the class `make`: at every node the exact mean of the
        lateral rule's interpolant, joined by the radial rule, one layer
        function per stored component."""
        nodal = np.stack([np.asarray(self._interpolant(e).mean())
                          for e in range(self._breaks.size - 1)])
        pp = self._mesh.to_ppoly(nodal, method=self._radial)
        shape = self.stored_shape
        if not shape:
            function = PolynomialLayer(self._interval, pp)
        else:
            function = np.empty(shape, dtype=object)
            for idx in np.ndindex(shape):
                function[idx] = PolynomialLayer(
                    self._interval, PPoly(pp.c[(...,) + idx], pp.x))
        return make(self._interval, function, character=self._character,
                    name=self._name)

    def on_interval(self, lo: float, hi: float) -> "GriddedBase":
        """The same elements on [lo, hi]: beyond the breaks the end
        elements continue."""
        out = copy.copy(self)
        out._interval = as_interval((lo, hi))
        out._interpolants = self._interpolants
        return out

    def renamed(self, name: str | None) -> "GriddedBase":
        """The same field under `name`."""
        out = copy.copy(self)
        out._name = name
        return out


class GriddedField(GriddedBase):
    """A field held on an angular grid at the GLL nodes of radial elements.

    `breaks` are the element boundaries, strictly increasing and
    spanning `interval` (within `rtol` of its width at each end), and
    `values` has shape `(nelem, ngll, ntheta, nphi) + stored shape`, the
    components in the spherical frame at the nodes of the elements, with
    `ngll` at least two.  `radial` is one of `mesh1d.RADIAL_METHODS` and
    `lateral` one of `lateral.METHODS`.  `poles`, of shape
    `(nelem, ngll, 2) + stored shape` with components in the Cartesian
    frame, gives the values at the poles the grid lacks, by the rule of
    `lateral.polar_values` with `rtol` of the values' largest; when it is
    None they are found from the grid's expansion, which needs a band,
    and refused otherwise.  Values are real.
    """

    def __init__(self, interval: tuple[float, float], breaks: ArrayLike,
                 grid: AngularGrid, values: ArrayLike, *,
                 character: Character = SCALAR, name: str | None = None,
                 radial: str = "polynomial", lateral: str = "linear",
                 poles: ArrayLike | None = None, rtol: float = 1e-9) -> None:
        from .sampling import AngularGrid
        if not isinstance(grid, AngularGrid):
            raise TypeError(f"grid must be an AngularGrid, got {type(grid).__name__}")
        if radial not in RADIAL_METHODS:
            raise ValueError(f"radial must be one of {RADIAL_METHODS}, got {radial!r}")
        if lateral not in METHODS:
            raise ValueError(f"lateral must be one of {METHODS}, got {lateral!r}")
        check_rule(grid, lateral)
        if poles is not None and np.iscomplexobj(poles):
            raise TypeError("a gridded field holds real values")
        v = self._hold(interval, breaks, values, (grid.ntheta, grid.nphi),
                       axes=("ntheta", "nphi"), character=character, name=name,
                       radial=radial, rtol=rtol)
        self._grid = grid
        self._lateral = lateral
        c = stored_shape(character)
        lead = (self._breaks.size - 1,)
        # the components laid out for the lateral rules: Cartesian, with
        # the grid's two axes last
        cart = v if not character.rank else rotate_stored(v, _grid_frames(grid),
                                                          character)
        self._cartesian = np.moveaxis(cart, (2, 3), (-2, -1))
        if poles is not None:
            want = lead + (v.shape[1], 2) + c
            p = np.asarray(poles, dtype=float)
            if p.shape != want:
                raise ValueError(f"poles must have shape (nelem, ngll, 2) + {c} = "
                                 f"{want}, got {p.shape}")
            poles = np.moveaxis(p, 2, -1)
        resolved = polar_values(self._cartesian, grid, poles=poles, rtol=self._rtol)
        self._poles = _readonly(np.moveaxis(resolved, -1, 2))

    @classmethod
    def sampled(cls, field: Field, grid: AngularGrid, *,
                breaks: ArrayLike | None = None, ngll: int = 5,
                radial: str = "polynomial", lateral: str = "linear",
                name: str | None = None) -> "GriddedField":
        """`field` evaluated on `grid` at the GLL nodes of elements with
        `breaks` (the ends of its interval by default) and `ngll` nodes
        each, with its values at both poles evaluated exactly: a gridded
        field that equals `field` at every node.  The name is the field's
        unless `name` is given."""
        lo, hi = (float(x) for x in field.interval)
        b = np.array([lo, hi] if breaks is None else breaks, dtype=float)
        mesh = Mesh1D(b, ngll=int(ngll))
        r = mesh.r[:, :, None, None]
        theta = grid.colatitudes[None, None, :, None]
        phi = grid.longitudes[None, None, None, :]
        values = field.evaluate(r, theta, phi)
        poles = field.evaluate(mesh.r[:, :, None], np.array([0.0, np.pi]),
                               np.zeros(2), frame="cartesian")
        return cls(field.interval, b, grid, values, character=field.character,
                   name=field.name if name is None else name, radial=radial,
                   lateral=lateral, poles=poles, rtol=getattr(field, "rtol", 1e-9))

    @property
    def grid(self) -> AngularGrid:
        """The angular grid at every node."""
        return self._grid

    @property
    def poles(self) -> np.ndarray:
        """The values at the north and south poles at every node,
        (nelem, ngll, 2) + stored shape, components in the Cartesian frame."""
        return self._poles

    def _interpolant(self, e: int) -> LateralInterpolant:
        """The lateral rule over the nodes of element e, built once."""
        f = self._interpolants.get(e)
        if f is None:
            f = LateralInterpolant(self._cartesian[e], self._grid,
                                   method=self._lateral,
                                   poles=np.moveaxis(self._poles[e], 1, -1),
                                   rtol=self._rtol)
            self._interpolants[e] = f
        return f

    def cartesian_derivatives(self, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                                         np.ndarray]:
        """The Cartesian components at flat points and their derivatives in
        r, theta and phi, each (P,) + stored shape: `derivatives`, the
        components the lateral rules read being Cartesian here."""
        return self.derivatives(r, theta, phi)

    def _values(self, r: np.ndarray, theta: np.ndarray | None,
                phi: np.ndarray | None) -> np.ndarray:
        shape = r.shape
        out = self._values_at(r.ravel(), theta.ravel(), phi.ravel())
        out = out.reshape(shape + self.stored_shape)
        if self._character.rank:
            R = spherical_frame(theta, phi)
            out = rotate_stored(out, np.swapaxes(R, -1, -2), self._character)
        return out

    def spherical_mean(self, *, grid: AngularGrid | None = None) -> RadialField:
        """The spherical mean as exact polynomial pieces: at every node the
        exact mean of the lateral rule's interpolant, joined by the radial
        rule, so the mean of the field as its rules define it; `grid` is
        not needed."""
        if self._character.rank:
            raise ValueError(f"a spherical mean is taken of a field of rank 0, "
                             f"not {self._character}")
        return self._lateral_mean(RadialField)

    def rescaled(self, *, k: float, v: float) -> "GriddedField":
        """v f(r / k, theta, phi): the breaks and interval scaled by k, the
        values and the polar values by v."""
        k, v = float(k), float(v)
        lo, hi = self._interval
        breaks = k * self._breaks
        out = GriddedField((breaks[0], breaks[-1]), breaks, self._grid,
                           v * self._data, character=self._character,
                           name=self._name, radial=self._radial,
                           lateral=self._lateral, poles=v * self._poles,
                           rtol=self._rtol)
        return out.on_interval(k * lo, k * hi)

    def __repr__(self) -> str:
        nm = f"{self._name!r}, " if self._name else ""
        lo, hi = self._interval
        return (f"GriddedField({nm}{self._character} on [{lo:g}, {hi:g}]: "
                f"{self._breaks.size - 1} x {self.ngll} nodes, {self._radial} x "
                f"{self._lateral} on {self._grid!r})")


class GriddedFunction:
    """A function of (theta, phi) held on an angular grid.

    `values` has shape (ntheta, nphi); `lateral` is one of
    `lateral.METHODS`, the bicubic one by default; `poles`, shape (2,),
    gives the values at the poles the grid lacks, by the rule of
    `lateral.polar_values` with `rtol`, and when it is None they are
    found from the grid's expansion or refused.  Calling it gives the
    values at points; `angular_gradient` gives (d/dtheta, d/dphi), so
    that it is a relief for `layer_linear`.
    """

    def __init__(self, grid: AngularGrid, values: ArrayLike, *,
                 lateral: str = "cubic", poles: ArrayLike | None = None,
                 rtol: float = 1e-9) -> None:
        if np.iscomplexobj(values):
            raise TypeError("a gridded function holds real values")
        v = np.asarray(values, dtype=float)
        if v.ndim != 2:
            raise ValueError(f"values must have shape (ntheta, nphi), got {v.shape}")
        if lateral not in METHODS:
            raise ValueError(f"lateral must be one of {METHODS}, got {lateral!r}")
        check_rule(grid, lateral)
        self._rtol = float(rtol)
        self._poles = _readonly(polar_values(v, grid, poles=poles, rtol=self._rtol))
        self._f = LateralInterpolant(v, grid, method=lateral, poles=self._poles,
                                     rtol=self._rtol)
        self._data = _readonly(v)

    @classmethod
    def sampled(cls, fn: Relief, grid: AngularGrid, *,
                lateral: str = "cubic") -> "GriddedFunction":
        """`fn(theta, phi)` evaluated on `grid` and, exactly, at both poles."""
        values = np.asarray(fn(grid.colatitudes[:, None], grid.longitudes[None, :]),
                            dtype=float)
        poles = np.asarray(fn(np.array([0.0, np.pi]), np.zeros(2)), dtype=float)
        return cls(grid, np.broadcast_to(values, (grid.ntheta, grid.nphi)),
                   lateral=lateral, poles=poles)

    @property
    def grid(self) -> AngularGrid:
        """The angular grid the values are on."""
        return self._f.grid

    @property
    def values(self) -> np.ndarray:
        """The values at the nodes, (ntheta, nphi)."""
        return self._data

    @property
    def poles(self) -> np.ndarray:
        """The values at the north and south poles."""
        return self._poles

    @property
    def lateral(self) -> str:
        """The lateral rule, one of `lateral.METHODS`."""
        return self._f.method

    def __call__(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The values at (theta, phi), broadcast together."""
        return self._f(theta, phi)

    def angular_gradient(self, theta: ArrayLike, phi: ArrayLike
                         ) -> tuple[np.ndarray, np.ndarray]:
        """(d/dtheta, d/dphi) at (theta, phi), broadcast together."""
        return self._f.gradient(theta, phi)

    def mean(self) -> float:
        """The mean over the sphere of the function as its rule defines it."""
        return float(self._f.mean())

    def scaled(self, k: float) -> "GriddedFunction":
        """k times the function, on the same grid by the same rule and with
        the same `rtol`: the relief of a body k times the size."""
        k = float(k)
        return GriddedFunction(self.grid, k * self._data, lateral=self.lateral,
                               poles=k * self._poles, rtol=self._rtol)

    def __repr__(self) -> str:
        return f"GriddedFunction({self.lateral} on {self.grid!r})"


class GriddedDisplacement(MappingBase):
    """The mapping m(X) = X + u(X), with u held on grids.

    `fields` holds one `GriddedField` of character `VECTOR` per layer of
    `skeleton`, in order, each on its layer's interval: the displacement
    m(X) - X in the spherical frame at X, as a sample holds it.  The
    deformation gradient is I + grad u from the radial and lateral
    rules' derivatives, the e_phi entry taken on the polar axis as its
    limit along the meridian; every breakpoint of every field's
    `radial_pieces` within the skeleton is a knot, the element boundaries
    of a field read by the polynomial rule and all the nodes of one read
    by the linear rule, where its radial derivative jumps.
    `GriddedDisplacement.sampled` builds one from any mapping.
    """

    def __init__(self, skeleton: Skeleton, fields: Sequence[GriddedField]) -> None:
        fields = tuple(fields)
        b = skeleton.boundaries
        if len(fields) != b.size - 1:
            raise ValueError(f"{b.size - 1} layers need {b.size - 1} gridded "
                             f"fields, got {len(fields)}")
        tol = 1e-9 * skeleton.span
        for i, f in enumerate(fields):
            if not isinstance(f, GriddedField) or f.character != VECTOR:
                raise ValueError(f"field {i} is not a gridded vector field")
            lo, hi = f.interval
            if abs(lo - b[i]) > tol or abs(hi - b[i + 1]) > tol:
                raise ValueError(f"field {i} lies on [{lo:g}, {hi:g}], its layer "
                                 f"on [{b[i]:g}, {b[i + 1]:g}]")
        self._sk, self._fields = skeleton, fields
        pieces = np.unique(np.concatenate([f.radial_pieces()[0] for f in fields]))
        self.knots = tuple(float(k) for k in pieces[1:-1])
        self.rmax = float(b[-1])
        self.name = "gridded_displacement"

    @classmethod
    def sampled(cls, mapping: Mapping, mesh: RadialMesh, grid: AngularGrid, *,
                lateral: str = "cubic") -> "GriddedDisplacement":
        """`mapping`'s displacement sampled at the nodes of `mesh`'s
        elements on `grid`, layer by layer, and exactly at both poles."""
        sk = mesh.skeleton
        disp = getattr(mapping, "displacement", None)

        def u(X: np.ndarray) -> np.ndarray:
            return np.asarray(disp(X) if disp is not None else mapping(X) - X,
                              dtype=float)

        R = _grid_frames(grid)                              # (nt, np, 3, 3)
        fields = []
        for i in range(sk.nlayers):
            elems = np.flatnonzero(mesh.layer == i)
            r = mesh.r[elems]                               # (ne, ngll)
            breaks = np.append(mesh.left[elems], mesh.right[elems[-1]])
            X = r[:, :, None, None, None] * R[None, None, ..., :, 0]
            values = np.einsum("tpji,egtpj->egtpi", R, u(X))
            poles = u(cartesian_points(r[:, :, None], np.array([0.0, np.pi]),
                                       np.zeros(2)))
            fields.append(GriddedField(tuple(sk.boundaries[i:i + 2]), breaks, grid,
                                       values, character=VECTOR, lateral=lateral,
                                       poles=poles))
        return cls(sk, fields)

    @property
    def skeleton(self) -> Skeleton:
        """The skeleton the displacement is held over."""
        return self._sk

    @property
    def fields(self) -> tuple[GriddedField, ...]:
        """The displacement's gridded field on each layer."""
        return self._fields

    def _derivatives(self, r: np.ndarray, theta: np.ndarray, phi: np.ndarray
                     ) -> list[np.ndarray]:
        """[u, du/dr, du/dtheta, du/dphi] at flat (r, theta, phi), each
        (P, 3), components Cartesian; a radius on a boundary is read from
        the layer above."""
        b = self._sk.boundaries
        layer = np.clip(np.searchsorted(b, r, side="right") - 1, 0, b.size - 2)
        parts = [np.empty((r.size, 3)) for _ in range(4)]
        for i in np.unique(layer):
            m = layer == i
            for out, got in zip(parts, self._fields[i].cartesian_derivatives(
                    r[m], theta[m], phi[m])):
                out[m] = got
        return parts

    def _parts(self, X: ArrayLike) -> tuple[np.ndarray, ...]:
        """(shape, r, theta, phi, R, u, du/dr, du/dtheta, du/dphi) at X, flat."""
        X = np.asarray(X, dtype=float)
        r, theta, phi, R = spherical_coordinates(X)
        shape = r.shape
        r, theta, phi = r.ravel(), theta.ravel(), phi.ravel()
        return (shape, r, theta, phi, R.reshape(-1, 3, 3),
                *self._derivatives(r, theta, phi))

    def displacement(self, X: ArrayLike) -> np.ndarray:
        """u(X) = m(X) - X, Cartesian, of the shape of X."""
        shape, *_, u, _, _, _ = self._parts(X)
        return u.reshape(shape + (3,))

    def __call__(self, X: ArrayLike) -> np.ndarray:
        """m(X) = X + u(X)."""
        X = np.asarray(X, dtype=float)
        return X + self.displacement(X)

    def deformation_gradient(self, X: ArrayLike) -> np.ndarray:
        """F = I + du/dr e_r + (1/r) du/dtheta e_theta + (1/(r sin theta))
        du/dphi e_phi, each a Cartesian column against a frame vector."""
        shape, r, theta, phi, R, _, dr, dt, dp = self._parts(X)
        rs = np.maximum(r, R_FLOOR_FRACTION * self.rmax)[:, None]
        sin_t = np.sin(theta)
        per_sin = dp / np.maximum(sin_t, SIN_FLOOR)[:, None]
        axis = sin_t < SIN_FLOOR
        if np.any(axis):
            # on the polar axis the e_phi derivative is the theta derivative
            # a quarter turn on, negated at the south pole
            q = self._derivatives(r[axis], theta[axis], phi[axis] + 0.5 * np.pi)[2]
            per_sin[axis] = np.where(np.cos(theta[axis]) > 0.0, 1.0, -1.0)[:, None] * q
        G = (np.einsum("pi,pj->pij", dr, R[:, :, 0])
             + np.einsum("pi,pj->pij", dt / rs, R[:, :, 1])
             + np.einsum("pi,pj->pij", per_sin / rs, R[:, :, 2]))
        return (np.eye(3) + G).reshape(shape + (3, 3))

    def jacobian(self, X: ArrayLike) -> np.ndarray:
        """J = det F."""
        return np.linalg.det(self.deformation_gradient(X))

    def scaled(self, k: float) -> "GriddedDisplacement":
        """k m(X / k): the skeleton and the fields' radii multiplied by k,
        and the displacement with them."""
        k = float(k)
        return GriddedDisplacement(Skeleton(k * self._sk.boundaries),
                                   [f.rescaled(k=k, v=k) for f in self._fields])

    def __repr__(self) -> str:
        return (f"GriddedDisplacement({len(self._fields)} layers, "
                f"{len(self.knots)} knots)")
