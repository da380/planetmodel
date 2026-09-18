"""Truncated eigenbases of the radial operators, as coordinates for fields.

The eigenfunctions of one member A_l of a `RadialOperatorFamily`,
M-orthonormal on the mesh, are an orthonormal basis of the discrete
weighted space L^2(w dr), and the leading ones a finite-dimensional
subspace of it.  A `SpectralBasis` is that subspace made usable: a
function is its vector of coefficients c, and

    synthesis      u = Phi c            nodal values from coefficients,
    analysis       c = Phi^T M u        the M-orthogonal projection back,
    evaluation     u(r) = phi(r) . c    at any radius, through the exact
                                        polynomial of each eigenfunction,
    integration    int u w dr = q . u   with q the quadrature weights,
                                        the mass of the elements integrated
                                        over.

In these coordinates every function of the operator is diagonal: the
H^s inner product of the Sobolev scale built on A is sum theta^s c^2, a
covariance A^-p is theta^-p, and white noise of the space is a vector of
independent standard normal coefficients.  The basis therefore carries
`theta` and leaves the powers to whoever needs them.

The mesh is usually longer than the interval that matters, padded so
that the boundary conditions act at a distance (the `mesh` module beside
this one).
The basis holds the `Restriction` to the physical interval; synthesis,
evaluation and the pointwise variance are the full-mesh quantities read
on the physical nodes, and integration runs over the physical elements
alone, while analysis takes values on the whole mesh, since a
projection needs the function everywhere the eigenfunctions live.  The
product of two fields is thus formed on the full mesh and analysed: a
product of nodal values, projected onto the kept modes.

Truncation is chosen once, when the basis is built: a number of modes,
a largest eigenvalue, or the variance-trace rule of the samplers, which
keeps the fewest modes whose omitted share of trace(A_l^-power) is at
most `tol`.

On an annulus or ball a scalar field has one radial coefficient
function per real spherical harmonic, and its coordinates are the
coefficients of each of those functions in the basis of its degree.
`SphericalBasis` holds the bases of degrees 0 ... lmax together and fixes
the order of that vector: by degree l, then the cosine harmonics of
orders 0 ... l, then the sine harmonics of orders 1 ... l (the order of
`planetmodel.harmonics.packing`), and within each harmonic the modes j
in ascending theta.  The number of modes may differ between degrees, so
the vector is ragged in l; `block` says where each harmonic sits.
"""
from __future__ import annotations

from collections.abc import Iterator, Sequence
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly

from .mesh import Restriction, restriction
from .operator import RadialOperatorFamily

__all__ = ["SpectralBasis", "SphericalBasis"]

#: Relative margin by which a select-by-value eigenvalue bound exceeds the
#: last eigenvalue wanted, so that roundoff cannot drop it.
_SELECT_RTOL = 1e-12

#: How far outside the physical interval, relative to its length, a radius
#: handed to `evaluate` may fall and still count as an endpoint.
_INTERVAL_RTOL = 1e-12


class SpectralBasis:
    """The kept eigenpairs of one A_l, with the restriction to the
    physical interval.

    `family` and the degree `l` name the operator; `restrict` is the
    physical interval inside the family's mesh, the whole mesh when
    None.  The truncation is at most one of `nmodes` (that many leading
    modes), `theta_max` (the modes with theta <= theta_max) or the pair
    `power`, `tol` (the fewest leading modes whose omitted share of
    trace(A_l^-power) is at most `tol`, never fewer than one); with none
    of them every mode is kept.

    Attributes: `family`, `degree`, `restriction`, `theta` (the kept
    eigenvalues, ascending, read-only), `nmodes`, and `r`, the physical
    nodes.
    """

    def __init__(self, family: RadialOperatorFamily, l: int, *,
                 restrict: Restriction | None = None, nmodes: int | None = None,
                 theta_max: float | None = None, power: float | None = None,
                 tol: float | None = None) -> None:
        if not isinstance(family, RadialOperatorFamily):
            raise TypeError("family must be a RadialOperatorFamily")
        if (power is None) != (tol is None):
            raise ValueError("power and tol are given together")
        rules = sum(x is not None for x in (nmodes, theta_max, power))
        if rules > 1:
            raise ValueError("give at most one of nmodes, theta_max and "
                             "(power, tol)")
        mesh = family.mesh
        if restrict is None:
            restrict = restriction(mesh, mesh.left[0], mesh.right[-1])
        elif (restrict.nodes.stop > mesh.nglob
              or not np.array_equal(restrict.r, mesh.rglob[restrict.nodes])):
            raise ValueError("the restriction is not one of the family's mesh")

        self.family = family
        self.degree = int(l)
        self.restriction = restrict
        self.r = restrict.r

        theta, Phi = self._truncate(nmodes, theta_max, power, tol)
        self.theta = theta.view()
        self.theta.setflags(write=False)
        self._Phi = Phi                               # active dofs
        self._modes = family.embed(self.degree, Phi).view()
        self._modes.setflags(write=False)
        self._ppoly: PPoly | None = None

    def _truncate(self, nmodes: int | None, theta_max: float | None,
                  power: float | None, tol: float | None
                  ) -> tuple[np.ndarray, np.ndarray]:
        """The kept eigenpairs under the rule given."""
        fam, l = self.family, self.degree
        if theta_max is not None:
            theta, Phi = fam.eig(l, theta_max=float(theta_max))
            if theta.size == 0:
                raise ValueError(f"no eigenvalue of degree {l} lies below "
                                 f"theta_max = {theta_max:g}; the smallest is "
                                 f"{fam.eigvalsh(l)[0]:g}")
            return theta, Phi
        if nmodes is None and power is None:
            return fam.eig(l)
        vals = fam.eigvalsh(l)
        if nmodes is not None:
            k = int(nmodes)
            if not 1 <= k <= vals.size:
                raise ValueError(f"nmodes must lie in 1..{vals.size}, got {nmodes}")
        else:
            if not 0.0 <= tol < 1.0:
                raise ValueError("tol must lie in [0, 1)")
            wts = vals ** (-power)
            tail = np.cumsum(wts[::-1])[::-1]
            small = tail <= tol * tail[0]
            k = max(int(np.argmax(small)) if small.any() else vals.size, 1)
        if k == vals.size:
            return fam.eig(l)
        theta, Phi = fam.eig(l, theta_max=float(vals[k - 1]) * (1.0 + _SELECT_RTOL))
        return theta[:k], Phi[:, :k]

    @property
    def nmodes(self) -> int:
        """The number of modes kept."""
        return int(self.theta.size)

    def modes(self) -> np.ndarray:
        """The kept eigenfunctions at the nodes of the full mesh, shape
        (nglob, nmodes), read-only: M-orthonormal columns, ascending in
        theta.  At the centre of a ball those of degree l >= 1 vanish."""
        return self._modes

    def synthesise(self, coeffs: ArrayLike, *, physical: bool = True) -> np.ndarray:
        """Nodal values Phi c of coefficients of shape (nmodes,) or
        (nmodes, k): at the physical nodes `r`, or with `physical=False`
        at every node of the mesh, which is what `analyse` takes."""
        c = np.asarray(coeffs, dtype=float)
        if c.ndim not in (1, 2) or c.shape[0] != self.nmodes:
            raise ValueError(f"expected {self.nmodes} coefficients along the "
                             f"first axis, got shape {c.shape}")
        Phi = self._modes[self.restriction.nodes] if physical else self._modes
        return Phi @ c

    def analyse(self, nodal: ArrayLike) -> np.ndarray:
        """The coefficients Phi^T M v of nodal values on the full mesh,
        shape (nglob,) or (nglob, k).

        The M-orthogonal projection onto the kept modes: the inverse of
        `synthesise(..., physical=False)` on their span, and the nearest
        element of it, in the weighted L^2 norm, to anything else.
        """
        v = np.asarray(nodal, dtype=float)
        nglob = self.family.mesh.nglob
        if v.ndim not in (1, 2) or v.shape[0] != nglob:
            raise ValueError(f"expected values at the {nglob} nodes of the mesh "
                             f"along the first axis, got shape {v.shape}")
        mass = self.family.mass(self.degree)
        v = v[nglob - mass.size:]             # the centre of a ball is massless
        return self._Phi.T @ (mass * v if v.ndim == 1 else mass[:, None] * v)

    def evaluate(self, r: ArrayLike) -> np.ndarray:
        """The kept eigenfunctions at radii of the physical interval:
        shape r.shape + (nmodes,), so that `evaluate(r) @ c` is the field
        of coefficients c there.  Exact: each eigenfunction is a piecewise
        polynomial, and its polynomial is what is evaluated."""
        r = np.asarray(r, dtype=float)
        r1, r2 = self.restriction.interval
        slack = _INTERVAL_RTOL * (r2 - r1)
        if np.any(r < r1 - slack) or np.any(r > r2 + slack):
            raise ValueError(f"radii must lie in the physical interval "
                             f"[{r1:g}, {r2:g}]")
        if self._ppoly is None:
            mesh = self.family.mesh
            self._ppoly = mesh.to_ppoly(self._modes[mesh.gmap],
                                        elements=self.restriction.elements)
        return self._ppoly(np.clip(r, r1, r2))

    def weights(self) -> np.ndarray:
        """Quadrature weights at the physical nodes, so that
        `weights() @ synthesise(c)` is the integral of the field over the
        physical interval against the family's measure: the mass assembled
        from the physical elements alone.  At r1 and r2 that is less than
        the diagonal mass, which counts the neighbouring pad element too."""
        mesh, R = self.family.mesh, self.restriction
        e0, e1 = R.elements
        out = np.zeros(self.r.size)
        np.add.at(out, mesh.gmap[e0:e1] - R.nodes.start,
                  self.family.element_mass()[e0:e1])
        return out

    def pointwise_variance(self, power: float) -> np.ndarray:
        """sum_j theta_j^-power phi_j(r)^2 at the physical nodes: the
        variance there of a field of this degree with covariance
        A_l^-power, truncated to the kept modes."""
        Phi = self._modes[self.restriction.nodes]
        return (Phi ** 2) @ self.theta ** (-float(power))

    def white_noise(self, *, rng: Any = None, size: int | None = None) -> np.ndarray:
        """Coefficients of white noise of the space, independent standard
        normals of shape (nmodes,) or (nmodes, size).  `rng` is anything
        `np.random.default_rng` accepts."""
        rng = np.random.default_rng(rng)
        shape = self.nmodes if size is None else (self.nmodes, int(size))
        return rng.standard_normal(shape)

    def __repr__(self) -> str:
        r1, r2 = self.restriction.interval
        return (f"SpectralBasis(l={self.degree}, {self.nmodes} modes, "
                f"theta <= {self.theta[-1]:g}, on [{r1:g}, {r2:g}])")


class SphericalBasis:
    """The bases of degrees 0 ... lmax of one family on one restriction,
    and the order of their coefficients as one vector.

    `bases[l]` is the `SpectralBasis` of degree l.  The vector runs over
    the degrees; within a degree over the cosine harmonics of orders
    0 ... l and then the sine harmonics of orders 1 ... l; within a
    harmonic over the modes in ascending theta.

    Attributes: `lmax`, `nmodes` (the modes kept at each degree), `size`
    (the length of the vector, sum_l (2 l + 1) nmodes[l]), `theta` (the
    eigenvalue belonging to each entry of the vector, read-only, of which
    a diagonal metric or covariance is a power), and `r`, the physical
    nodes.
    """

    def __init__(self, bases: Sequence[SpectralBasis]) -> None:
        bases = tuple(bases)
        if not bases:
            raise ValueError("need the basis of degree 0 at least")
        first = bases[0]
        for l, b in enumerate(bases):
            if not isinstance(b, SpectralBasis):
                raise TypeError("bases must be SpectralBasis objects")
            if b.degree != l:
                raise ValueError(f"bases[{l}] has degree {b.degree}")
            if b.family is not first.family:
                raise ValueError("the bases must share one family")
            if (b.restriction.interval != first.restriction.interval
                    or b.restriction.nodes != first.restriction.nodes):
                raise ValueError("the bases must share one restriction")
        self._bases = bases
        self.lmax = len(bases) - 1
        self.nmodes = np.array([b.nmodes for b in bases])
        self.nmodes.setflags(write=False)
        counts = (2 * np.arange(self.lmax + 1) + 1) * self.nmodes
        self._offsets = np.concatenate(([0], np.cumsum(counts)))
        self.size = int(self._offsets[-1])
        self.r = first.r
        self.theta = np.concatenate([np.tile(b.theta, 2 * l + 1)
                                     for l, b in enumerate(bases)])
        self.theta.setflags(write=False)

    @property
    def family(self) -> RadialOperatorFamily:
        """The family the bases are of."""
        return self._bases[0].family

    @property
    def restriction(self) -> Restriction:
        """The physical interval the bases share."""
        return self._bases[0].restriction

    def __len__(self) -> int:
        return self.lmax + 1

    def __getitem__(self, l: int) -> SpectralBasis:
        """The basis of degree l."""
        return self._bases[l]

    def __iter__(self) -> Iterator[SpectralBasis]:
        """The bases in order of degree."""
        return iter(self._bases)

    def block(self, s: int, l: int, m: int) -> slice:
        """Where the nmodes[l] coefficients of one harmonic sit in the
        vector: s = 0 for the cosine harmonic of order m, 0 <= m <= l, and
        s = 1 for the sine one, 1 <= m <= l."""
        if not 0 <= l <= self.lmax:
            raise IndexError(f"degree {l} is outside 0..{self.lmax}")
        if s not in (0, 1) or not s <= m <= l:
            raise IndexError(f"no harmonic (s, l, m) = ({s}, {l}, {m})")
        n = int(self.nmodes[l])
        start = int(self._offsets[l]) + (m if s == 0 else l + m) * n
        return slice(start, start + n)

    def _degree_block(self, l: int) -> slice:
        return slice(int(self._offsets[l]), int(self._offsets[l + 1]))

    def synthesise(self, vector: ArrayLike, *, physical: bool = True) -> np.ndarray:
        """The coefficient functions of a vector, shape
        (2, lmax + 1, lmax + 1, n) in the layout of `planetmodel.harmonics`
        with the radial axis last: at the physical nodes `r`, or with
        `physical=False` at every node of the mesh, which is what `analyse`
        takes."""
        v = np.asarray(vector, dtype=float)
        if v.shape != (self.size,):
            raise ValueError(f"expected a vector of length {self.size}, got "
                             f"shape {v.shape}")
        n = self.r.size if physical else self.family.mesh.nglob
        out = np.zeros((2, self.lmax + 1, self.lmax + 1, n))
        for l, b in enumerate(self._bases):
            c = v[self._degree_block(l)].reshape(2 * l + 1, b.nmodes)
            U = b.synthesise(c.T, physical=physical)           # (n, 2 l + 1)
            out[0, l, :l + 1] = U[:, :l + 1].T
            out[1, l, 1:l + 1] = U[:, l + 1:].T
        return out

    def analyse(self, coeffs: ArrayLike) -> np.ndarray:
        """The vector of coefficient functions given at every node of the
        mesh, shape (2, lmax + 1, lmax + 1, nglob): each harmonic's function
        projected onto the basis of its degree.  Entries of `coeffs` that
        belong to no harmonic (m > l, the sine slot of m = 0) are ignored."""
        c = np.asarray(coeffs, dtype=float)
        want = (2, self.lmax + 1, self.lmax + 1, self.family.mesh.nglob)
        if c.shape != want:
            raise ValueError(f"expected coefficient functions of shape {want}, "
                             f"got {c.shape}")
        out = np.empty(self.size)
        for l, b in enumerate(self._bases):
            U = np.concatenate((c[0, l, :l + 1], c[1, l, 1:l + 1]))   # (2 l + 1, nglob)
            out[self._degree_block(l)] = b.analyse(U.T).T.reshape(-1)
        return out

    def __repr__(self) -> str:
        r1, r2 = self.restriction.interval
        return (f"SphericalBasis(lmax={self.lmax}, {self.size} coefficients, "
                f"on [{r1:g}, {r2:g}])")
