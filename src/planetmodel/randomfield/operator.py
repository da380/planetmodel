"""Self-adjoint separable operator families and their spectra.

The scalar elliptic operator A = 1 - div(kappa grad), with kappa a
function of one coordinate x alone (optionally anisotropic, with
distinct coefficients along x and across it), separates wherever the
other coordinates carry a complete set of eigenfunctions of their own
Laplacian: it block-diagonalises into a family of ordinary operators
in x::

    A_lambda f = f - w^-1 (kappa w f')' + lambda kappa_h p f,

one per lateral eigenvalue lambda >= 0, each self-adjoint on the
weighted space L^2(w dx).  Three statements name the geometry::

    weight    the measure w:  "r2" for x^2, "r" for x, "one" for 1;
    lateral   the profile p of the lateral term:  "inverse_square" for
              1 / x^2, "one" for 1;
    index     how a member is labelled:  "degree" for an integer l with
              lambda = l (l + 1), "order" for an integer m with
              lambda = m^2, "value" for lambda itself, a real number.

A ball or a spherical annulus is ("r2", "inverse_square", "degree"), x
the radius and the members those of the spherical harmonics; a disk or
a plane annulus ("r", "inverse_square", "order"), the members those of
the circle's Fourier modes; a flat layer ("one", "one", "value"), x the
depth and lambda the squared length of a lateral wavevector; an
interval of the line ("one", "inverse_square", "degree"), below.  The index "value"
goes with every weight and profile, "degree" with the weights "r2" and
"one" and "order" with the weight "r", both under the inverse-square
profile: a degree or an order is the label of an angular harmonic, and
a family that is not of a sphere or a circle states its lambda.

`SeparableOperatorFamily` discretises every member at once on one
`Mesh1D` with the GLL spectral-element basis: the mass matrix M is
diagonal (GLL quadrature, with one exact correction at a centre,
below), the lateral term K_h = int kappa_h phi phi w p dx is likewise
diagonal, and the stiffness K_r along x is banded with half-bandwidth
ngll - 1, so that A_lambda = M + K_r + lambda K_h.  Boundary conditions
are natural (Neumann) unless Robin coefficients are given, in which
case f' -+ gamma f = 0 at the inner/outer boundary adds
gamma kappa w f g to the form there; gamma is an inverse length,
whatever kappa is.

Because M is diagonal, the generalised problem A Phi = M Phi Theta
symmetrises into the banded ordinary problem S = M^-1/2 A M^-1/2 for
LAPACK's banded symmetric drivers, with select-by-value for spectral
truncation.  Since A_lambda = M + K with K >= 0, every eigenvalue
satisfies theta >= 1, so negative fractional powers of the family are
uniformly well behaved.  Eigenvectors are M-orthonormal
(Phi^T M Phi = I) and of a fixed sign, the first component above
`SIGN_RTOL` of the largest positive, so that the same eigenfunction
discretised on two meshes has one sign; A^s = Phi Theta^s Phi^T M on
nodal vectors and <f, A^s g> with the L^2(w dx) pairing is the H^s
inner product of the Sobolev scale built on A.  White noise of the
space is M^-1/2 z with z standard normal, whose covariance is M^-1.

A mesh that starts at x = 0 under the measure x^2 or x has a centre: a
ball's or a disk's.  It needs care but no boundary condition, and one
rule serves both.  Under the inverse-square profile the eigenfunctions
of a member with lambda > 0 vanish there, so the axis dof is dropped
(f(0) = 0 imposed strongly); at lambda = 0 the axis value is genuine
but carries zero GLL mass, making the pencil singular there.  The
massless row reads (A_0 f)_0 = 0, a discrete regularity condition
slaving f(0) to the first element's interior nodes, and is eliminated
exactly by static condensation: the Schur complement stays within the
band, the reduced pencil has precisely the finite spectrum of the
singular one, and `embed` reconstructs the axis value.  Under the
profile one the lateral term is as massless at the centre as the mass,
and every member is condensed as the member lambda = 0 is.

With the plain measure w = 1 there is no centre: x = 0, if the mesh
reaches it, is an ordinary node with its own mass, and the mesh may lie
anywhere.  Under the profile one that is the whole story.  Under the
inverse-square profile the family is that of an interval of the line:
its member lambda = 0, f - (kappa f')', is the one-dimensional operator
and asks nothing of where the interval lies; the lateral term alone
knows about x = 0, so the members with lambda > 0 exist only on a mesh
in x > 0 and are refused on any other.

What is computed for a member (its pencil, its eigenvalues, its
eigenpairs) is cached under its lambda, within a budget of memory: when
the cache outgrows `cache_bytes` the members least recently used are
forgotten whole, and a forgotten member is computed again when next
asked for, by the same arithmetic.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.linalg import eig_banded, eigvals_banded, solveh_banded

from ..mesh1d import Mesh1D

__all__ = ["SeparableOperatorFamily", "Coefficient",
           "Robin", "Label", "WEIGHTS", "LATERALS", "INDICES", "CACHE_BYTES",
           "SIGN_RTOL"]

#: The size, relative to an eigenvector's largest component, from which a
#: component counts as nonzero when the vector's sign is fixed.
SIGN_RTOL = 1e-8

#: The measures a family may carry: "r2" for x^2 dx, the radial part of the
#: volume element of an annulus or ball, "r" for x dx, that of the area
#: element of a disk or plane annulus, and "one" for plain dx.
WEIGHTS: tuple[str, ...] = ("r2", "r", "one")

#: The profiles of a family's lateral term: "inverse_square" for 1 / x^2,
#: the angular part of a Laplacian about a centre, and "one" for 1, the
#: lateral part of a Laplacian in Cartesian coordinates.
LATERALS: tuple[str, ...] = ("inverse_square", "one")

#: The ways a family labels its members: "degree" by an integer l with
#: lateral eigenvalue l (l + 1), "order" by an integer m with m^2, and
#: "value" by the lateral eigenvalue itself.
INDICES: tuple[str, ...] = ("degree", "order", "value")

#: The weights each index of `INDICES` goes with under the inverse-square
#: profile; "value" goes with every weight and profile.
_INDEX_WEIGHTS: dict[str, tuple[str, ...]] = {
    "degree": ("r2", "one"), "order": ("r",), "value": WEIGHTS}

#: The default budget of a family's cache, in bytes (1 GiB).
CACHE_BYTES: int = 2 ** 30

#: A coefficient of the operator: a number, a callable of the coordinate,
#: or a nodal array of the mesh's shape (nspec, ngll).
type Coefficient = ArrayLike | Callable[[np.ndarray], ArrayLike]

#: Robin coefficients: None for Neumann on both boundaries, one gamma for
#: both, or a pair (inner, outer) with None meaning Neumann there.
type Robin = float | tuple[float | None, float | None] | None

#: The label of a member of a family: an integer degree or order, or the
#: lateral eigenvalue itself, as the family's index says.
type Label = int | float


def _nodal_coefficient(mesh: Mesh1D, c: Coefficient) -> np.ndarray:
    """A coefficient as a per-element nodal array of shape (nspec, ngll):
    a scalar, a callable of radius, or an array already in that shape."""
    if callable(c):
        out = np.asarray(c(mesh.r), dtype=float)
        out = np.broadcast_to(out, mesh.r.shape).copy()
    elif np.ndim(c) == 0:
        out = np.full_like(mesh.r, float(c))
    else:
        out = np.asarray(c, dtype=float)
        if out.shape != mesh.r.shape:
            raise ValueError(f"coefficient array must have nodal shape "
                             f"{mesh.r.shape}, got {out.shape}")
        out = out.copy()
    if not np.all(np.isfinite(out)):
        raise ValueError("coefficient must be finite everywhere on the mesh")
    return out


def _signed(Phi: np.ndarray) -> np.ndarray:
    """The columns of Phi, each with its first component larger in size
    than `SIGN_RTOL` of its largest made positive."""
    size = np.abs(Phi)
    big = size > SIGN_RTOL * size.max(axis=0, keepdims=True)
    first = np.argmax(big, axis=0)
    lead = Phi[first, np.arange(Phi.shape[1])]
    return Phi * np.where(lead < 0.0, -1.0, 1.0)


def _parse_robin(robin: Robin) -> tuple[float, float]:
    """(gamma_inner, gamma_outer) from None, one number, or a pair with
    None entries meaning Neumann."""
    if robin is None:
        return 0.0, 0.0
    if np.ndim(robin) == 0:
        g = float(robin)
        gin, gout = g, g
    else:
        gin, gout = robin
        gin = 0.0 if gin is None else float(gin)
        gout = 0.0 if gout is None else float(gout)
    if gin < 0.0 or gout < 0.0:
        raise ValueError("Robin coefficients must be non-negative")
    return gin, gout


class _Member:
    """What a family's cache holds of one member: its label, and whichever
    of its pencil (band, mass), its eigenvalues and its eigenpairs (theta,
    Phi, the bound they reach) have been computed."""

    __slots__ = ("label", "pencil", "vals", "pairs", "nbytes")

    def __init__(self, label: Label) -> None:
        self.label = label
        self.pencil: tuple[np.ndarray, np.ndarray] | None = None
        self.vals: np.ndarray | None = None
        self.pairs: tuple[np.ndarray, np.ndarray, float] | None = None
        self.nbytes = 0

    def measure(self) -> int:
        """The bytes of the arrays held, each counted once; the mass, which
        is the family's own, is not counted."""
        held = {}
        if self.pencil is not None:
            held[id(self.pencil[0])] = self.pencil[0]
        if self.vals is not None:
            held[id(self.vals)] = self.vals
        if self.pairs is not None:
            held[id(self.pairs[0])] = self.pairs[0]
            held[id(self.pairs[1])] = self.pairs[1]
        return int(sum(a.nbytes for a in held.values()))


class SeparableOperatorFamily:
    """The operators A_lambda = M + K_r + lambda K_h on a 1-D GLL mesh,
    spectrally.

    Construction assembles, once, the diagonal mass M, the diagonal
    lateral factor K_h and the banded stiffness K_r on the given `Mesh1D`
    (a `RadialMesh` works too); `pencil(l)` combines them into the
    symmetric banded matrix of the member labelled l on its active dofs.
    `kappa` (along the mesh) and `kappa_h` (lateral, defaulting to kappa)
    may be scalars, callables of the coordinate, or nodal arrays.

    `weight` is the measure, one of `WEIGHTS`: "r2" for x^2 dx (balls and
    spherical annuli) and "r" for x dx (disks and plane annuli), each on
    a mesh in x >= 0, or "one" for plain dx on any interval.  `lateral`
    is the profile of the lateral term, one of `LATERALS`:
    "inverse_square" for 1 / x^2 or "one" for 1.  `index` says what the
    label of a member is, one of `INDICES`: "degree", an integer l >= 0
    with lambda = l (l + 1); "order", an integer m >= 0 with
    lambda = m^2; or "value", lambda itself, a real number >= 0.  The
    defaults are a ball's or a spherical annulus's.  "value" goes with
    every weight and profile; "degree" goes with the weights "r2" and
    "one" and "order" with the weight "r", both under the inverse-square
    profile, and any other combination is refused: the lateral
    eigenvalue wanted is then stated through "value".  Under the plain
    measure with the inverse-square profile the members with lambda > 0
    need the mesh to lie in x > 0.

    `robin` is None, one gamma for both boundaries, or a pair (inner,
    outer) with None meaning Neumann; an inner Robin term is ignored at
    a centre.  `cache_bytes` is the budget of the cache of members, by
    default `CACHE_BYTES`, or None for no bound: when what is held
    exceeds it the members least recently used are forgotten, the one in
    use never, and a forgotten member is computed again when asked for.
    Members are cached under their lambda, so equal labels share one
    entry; with the index "value" the number given is the key as it
    stands, and one lateral eigenvalue reached by two routes must be
    passed as the same float to be the same member.

    The spectral methods are `eigvalsh(l)`, `eig(l, *, theta_max, count)`,
    `apply_power(l, s, v)`, `inner(l, s, u, v)` and `logdet(l)`; the
    direct ones `apply(l, v)` for A v and `solve(l, b)` for A^-1 b;
    `white_noise(l, rng=...)` draws M^-1/2 z; `ndof`, `nodes`, `mass`,
    `embed` account for the axis dof removed at a centre, and
    `element_mass` is the mass before assembly.  `label(l)` is a label
    as the family holds it and `lateral_eigenvalue(l)` its lambda.

    Attributes: `mesh`, `weight`, `lateral`, `index`, `cache_bytes`.
    """

    def __init__(self, mesh: Mesh1D, *, kappa: Coefficient = 1.0,
                 kappa_h: Coefficient | None = None, weight: str = "r2",
                 lateral: str = "inverse_square", index: str = "degree",
                 robin: Robin = None, cache_bytes: int | None = CACHE_BYTES
                 ) -> None:
        if weight not in WEIGHTS:
            raise ValueError(f"weight must be one of {WEIGHTS}")
        if lateral not in LATERALS:
            raise ValueError(f"lateral must be one of {LATERALS}")
        if index not in INDICES:
            raise ValueError(f"index must be one of {INDICES}")
        if index != "value" and lateral != "inverse_square":
            raise ValueError(
                f"index {index!r} labels an angular harmonic, whose lateral "
                f"term has the profile 'inverse_square'; with lateral="
                f"{lateral!r} the lateral eigenvalue is given itself, "
                f"index='value'")
        if weight not in _INDEX_WEIGHTS[index]:
            raise ValueError(
                f"index {index!r} goes with weight "
                f"{' or '.join(repr(w) for w in _INDEX_WEIGHTS[index])}, not "
                f"with weight {weight!r}; index='value' gives any lateral "
                f"eigenvalue under any weight")
        if not isinstance(mesh, Mesh1D):
            raise TypeError("mesh must be a Mesh1D (or RadialMesh)")
        if weight == "r2" and mesh.rglob[0] < 0.0:
            raise ValueError("the r^2 measure needs a mesh in r >= 0")
        if weight == "r" and mesh.rglob[0] < 0.0:
            raise ValueError("the r measure needs a mesh in r >= 0")
        if cache_bytes is not None and not cache_bytes > 0:
            raise ValueError("cache_bytes must be positive, or None for no bound")

        self.mesh = mesh
        self.weight = weight
        self.lateral = lateral
        self.index = index
        self.cache_bytes = cache_bytes
        inverse = lateral == "inverse_square"
        self._centre = bool(weight != "one" and mesh.rglob[0] == 0.0)
        self._drops = self._centre and inverse
        self._zero_only = bool(weight == "one" and inverse
                               and mesh.rglob[0] <= 0.0)

        kr = _nodal_coefficient(mesh, kappa)
        kh = kr if kappa_h is None else _nodal_coefficient(mesh, kappa_h)
        if np.any(kr <= 0.0) or np.any(kh < 0.0):
            raise ValueError("kappa must be positive (kappa_h non-negative)")

        r, jac, wq = mesh.r, mesh.jac, mesh.w
        D, g = mesh.deriv, mesh.gmap
        n, q = mesh.nglob, mesh.ngll - 1
        wgt = (r * r if weight == "r2" else r.copy() if weight == "r"
               else np.ones_like(r))

        Me = wq[None, :] * jac[:, None] * wgt
        Me.setflags(write=False)
        M = np.zeros(n)
        np.add.at(M, g, Me)
        Kh = np.zeros(n)
        if inverse:
            rsafe = np.where(r != 0.0, r, 1.0)
            profile = wgt / rsafe ** 2
        else:
            profile = wgt
        np.add.at(Kh, g, kh * (wq[None, :] * jac[:, None]) * profile)

        # banded stiffness, upper symmetric storage:
        # band[q + i - j, j] = K[i, j] for i <= j
        coef = wq[None, :] * kr * wgt / jac[:, None]
        Ke = np.einsum("qi,eq,qj->eij", D, coef, D)
        band = np.zeros((q + 1, n))
        GA, GB = g[:, :, None], g[:, None, :]
        sel = GA <= GB
        rows = np.broadcast_to(q + GA - GB, Ke.shape)
        cols = np.broadcast_to(GB, Ke.shape)
        np.add.at(band, (rows[sel], cols[sel]), Ke[sel])

        gin, gout = _parse_robin(robin)
        if gout > 0.0:
            band[q, -1] += gout * kr[-1, -1] * wgt[-1, -1]
        if gin > 0.0 and not self._centre:
            band[q, 0] += gin * kr[0, 0] * wgt[0, 0]

        # static condensation data for the massless centre: where the axis
        # dof is condensed its row of the pencil reads
        # d f_0 + a . f_(1..q) = 0
        if self._centre:
            self._axis_a = np.array([band[q - j, j] for j in range(1, q + 1)])
            self._axis_d = float(band[q, 0])
            self._axis_c = -self._axis_a / self._axis_d
        if np.any(M[1 if self._centre else 0:] <= 0.0):
            raise RuntimeError("non-positive mass entry; degenerate mesh?")

        self._Me = Me
        self._M, self._Kh, self._Kr, self._q = M, Kh, band, q
        self._members: OrderedDict[float, _Member] = OrderedDict()
        self._nbytes = 0

    # -- labels and the cache -------------------------------------------------

    def label(self, l: Label) -> Label:
        """A member's label as the family holds it: the integer of a degree
        or an order, the float of a lateral eigenvalue.  Refuses a degree
        or order that is negative or not an integer, and a lateral
        eigenvalue that is negative or not finite."""
        if self.index == "value":
            lam = float(l)
            if not (lam >= 0.0 and np.isfinite(lam)):
                raise ValueError(f"a lateral eigenvalue must be non-negative "
                                 f"and finite, got {l!r}")
            return lam
        i = int(l)
        if i != l:
            raise ValueError(f"{self.index} must be an integer, got {l!r}")
        if i < 0:
            raise ValueError(f"{self.index} must be non-negative")
        return i

    def lateral_eigenvalue(self, l: Label) -> float:
        """The lambda of the member labelled l: l (l + 1) of a degree, m^2
        of an order, the value itself otherwise."""
        l = self.label(l)
        if self.index == "degree":
            return float(l * (l + 1))
        if self.index == "order":
            return float(l * l)
        return l

    def _lam(self, l: Label) -> float:
        """The lambda of a member the family has, refusing one it has not:
        lambda > 0 on a plain-measure mesh reaching x <= 0 under the
        inverse-square profile."""
        lam = self.lateral_eigenvalue(l)
        if lam > 0.0 and self._zero_only:
            if self.index == "degree":
                raise ValueError("degree l >= 1 needs r > 0 on a plain-measure mesh")
            raise ValueError("a lateral eigenvalue above zero needs x > 0 on a "
                             "plain-measure mesh under the inverse-square "
                             "profile")
        return lam

    def _member(self, l: Label) -> _Member:
        """The cache's entry for a member, made if absent and marked as the
        most recently used."""
        lam = self._lam(l)
        member = self._members.get(lam)
        if member is None:
            member = self._members[lam] = _Member(self.label(l))
        else:
            self._members.move_to_end(lam)
        return member

    def _account(self, member: _Member) -> None:
        """Count what a member now holds and, over the budget, forget the
        members least recently used, the most recent never."""
        size = member.measure()
        self._nbytes += size - member.nbytes
        member.nbytes = size
        if self.cache_bytes is None:
            return
        while self._nbytes > self.cache_bytes and len(self._members) > 1:
            _, old = self._members.popitem(last=False)
            self._nbytes -= old.nbytes

    @property
    def cache_nbytes(self) -> int:
        """The bytes the cache of members holds now: the banded pencils,
        eigenvalues and eigenvectors of the members remembered."""
        return self._nbytes

    def cached(self) -> list[Label]:
        """The labels of the members whose eigenvalues or eigenpairs the
        cache holds, ascending."""
        return sorted(m.label for m in self._members.values()
                      if m.vals is not None or m.pairs is not None)

    # -- active dofs ----------------------------------------------------------

    @property
    def has_centre(self) -> bool:
        """Whether the mesh starts at a centre: x = 0 under the measure
        x^2 or x, a ball's or a disk's."""
        return self._centre

    @property
    def is_ball(self) -> bool:
        """Whether the family is that of a ball: the r^2 measure on a mesh
        that starts at r = 0."""
        return self._centre and self.weight == "r2"

    def _n0(self, l: Label) -> int:
        """Leading dofs excluded from the member labelled l: the axis at a
        centre."""
        self._lam(l)
        return 1 if self._centre else 0

    def ndof(self, l: Label) -> int:
        """Number of active dofs of the member labelled l."""
        return self.mesh.nglob - self._n0(l)

    def nodes(self, l: Label) -> np.ndarray:
        """Coordinates of the active dofs of the member labelled l."""
        return self.mesh.rglob[self._n0(l):]

    def mass(self, l: Label) -> np.ndarray:
        """Diagonal mass on the active dofs of the member labelled l."""
        return self._M[self._n0(l):]

    def element_mass(self) -> np.ndarray:
        """The mass of each element at its own nodes, shape (nspec, ngll),
        read-only: quadrature weight times Jacobian times the measure.
        Summed over all the elements sharing a node it is the diagonal
        mass; summed over a run of elements it is the weights of an
        integral over that run alone, the same for every member."""
        return self._Me

    def embed(self, l: Label, values: ArrayLike) -> np.ndarray:
        """Active-dof values completed to the full mesh (a no-op without a
        centre): the centre is zero where the axis dof is dropped, the
        members with lambda > 0 under the inverse-square profile, and the
        statically condensed value f(0) = c . f_(first element) otherwise;
        works on (n,) vectors and (n, k) stacks alike."""
        v = np.asarray(values, dtype=float)
        lam = self._lam(l)
        if not self._centre:
            return v
        out = np.zeros((v.shape[0] + 1,) + v.shape[1:])
        out[1:] = v
        if not (self._drops and lam > 0.0):
            out[0] = self._axis_c @ v[:self._q]
        return out

    # -- the pencil and its spectrum ----------------------------------------

    def pencil(self, l: Label) -> tuple[np.ndarray, np.ndarray]:
        """Banded A_lambda (upper storage) and diagonal mass of the member
        labelled l, active dofs only.

        At a centre the axis dof is removed from every member: where
        lambda > 0 under the inverse-square profile its couplings are
        discarded (a strong f(0) = 0); otherwise the exact Schur
        complement of the massless axis row is folded into the first
        element's block, so the reduced pencil carries precisely the
        finite spectrum of the singular one.
        """
        member = self._member(l)
        if member.pencil is not None:
            return member.pencil
        lam = self._lam(l)
        n0, q = (1 if self._centre else 0), self._q
        band = self._Kr.copy()
        if n0 and not (self._drops and lam > 0.0):
            a, d = self._axis_a, self._axis_d
            for i in range(1, q + 1):
                for j in range(i, q + 1):
                    band[q + i - j, j] -= a[i - 1] * a[j - 1] / d
        band = band[:, n0:]
        if n0:
            for m in range(q):
                j = q - 1 - m
                if j < band.shape[1]:
                    band[m, j] = 0.0
        mass = self._M[n0:]
        band[q] += mass + lam * self._Kh[n0:]
        band.setflags(write=False)
        member.pencil = (band, mass)
        self._account(member)
        return band, mass

    @staticmethod
    def _scaled(band: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Jacobi-symmetrised band: S[i, j] = s_i band[i, j] s_j."""
        q = band.shape[0] - 1
        n = band.shape[1]
        out = np.zeros_like(band)
        for m in range(q + 1):
            u = q - m
            if u < n:
                js = np.arange(u, n)
                out[m, u:] = band[m, u:] * s[js] * s[js - u]
        return out

    def eigvalsh(self, l: Label) -> np.ndarray:
        """All eigenvalues of the member labelled l (ascending, >= 1 up to
        roundoff), cached."""
        member = self._member(l)
        if member.vals is None:
            band, mass = self.pencil(l)
            v = eigvals_banded(self._scaled(band, 1.0 / np.sqrt(mass)), lower=False)
            v.setflags(write=False)
            member.vals = v
            self._account(member)
        return member.vals

    def eig(self, l: Label, *, theta_max: float | None = None,
            count: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Eigenpairs (theta, Phi) of A Phi = M Phi Theta for the member
        labelled l, cached.

        Phi has the active-dof eigenvectors in its columns, ascending in
        theta, M-orthonormal and each of a fixed sign: its first component
        above `SIGN_RTOL` of its largest is positive, so that the same
        eigenfunction on two meshes has the same sign (LAPACK's is
        arbitrary).  With `theta_max`, only the eigenvalues
        theta <= theta_max are computed; with `count`, the `count`
        smallest, selected by index so that none is lost to the rounding
        of a bound.  The cache keeps the widest range requested so far
        and hands out truncated views of it.
        """
        if theta_max is not None and count is not None:
            raise ValueError("give at most one of theta_max and count")
        member = self._member(l)
        cached = member.pairs
        if count is not None:
            count = int(count)
            if count < 1:
                raise ValueError(f"count must be positive, got {count}")
            if cached is None or (cached[0].size < count and not np.isinf(cached[2])):
                band, mass = self.pencil(l)
                s = 1.0 / np.sqrt(mass)
                S = self._scaled(band, s)
                if count >= S.shape[1]:
                    theta, Y = eig_banded(S, lower=False)
                    if member.vals is None:
                        member.vals = theta
                    have = np.inf
                else:
                    theta, Y = eig_banded(S, lower=False, select="i",
                                          select_range=(0, count - 1))
                    have = float(theta[-1])
                member.pairs = (theta, _signed(s[:, None] * Y), have)
                self._account(member)
            theta, Phi, _ = member.pairs
            return theta[:count], Phi[:, :count]
        want = np.inf if theta_max is None else float(theta_max)
        if cached is None or cached[2] < want:
            band, mass = self.pencil(l)
            s = 1.0 / np.sqrt(mass)
            S = self._scaled(band, s)
            if np.isinf(want):
                theta, Y = eig_banded(S, lower=False)
                if member.vals is None:
                    member.vals = theta
            else:
                theta, Y = eig_banded(S, lower=False, select="v",
                                      select_range=(0.0, want))
            Phi = _signed(s[:, None] * Y)
            member.pairs = (theta, Phi, want)
            self._account(member)
        theta, Phi, have = member.pairs
        if want < have:
            k = int(np.searchsorted(theta, want, side="right"))
            return theta[:k], Phi[:, :k]
        return theta, Phi

    # -- the operators themselves --------------------------------------------

    def _check(self, l: Label, v: ArrayLike) -> np.ndarray:
        v = np.asarray(v, dtype=float)
        if v.shape[0] != self.ndof(l):
            raise ValueError(f"expected {self.ndof(l)} active dofs, got {v.shape[0]}")
        return v

    def apply(self, l: Label, v: ArrayLike) -> np.ndarray:
        """A v for the member labelled l on active-dof values v of shape
        (n,) or (n, k): the operator of the weighted space, M^-1 K v with
        K the banded pencil, so that `inner(l, 0, u, apply(l, v))` is the
        bilinear form u^T K v."""
        v = self._check(l, v)
        band, mass = self.pencil(l)
        q, n = self._q, band.shape[1]
        out = band[q][:, None] * v.reshape(n, -1) if v.ndim == 2 else band[q] * v
        for m in range(q):
            u = q - m
            if u < n:
                b = band[m, u:]
                if v.ndim == 2:
                    out[u:] += b[:, None] * v[:n - u]
                    out[:n - u] += b[:, None] * v[u:]
                else:
                    out[u:] += b * v[:n - u]
                    out[:n - u] += b * v[u:]
        return out / mass[:, None] if v.ndim == 2 else out / mass

    def solve(self, l: Label, b: ArrayLike) -> np.ndarray:
        """A^-1 b for the member labelled l, the inverse of `apply`, by
        banded Cholesky of the pencil, for b of shape (n,) or (n, k)."""
        b = self._check(l, b)
        band, mass = self.pencil(l)
        rhs = mass[:, None] * b if b.ndim == 2 else mass * b
        return solveh_banded(band, rhs, lower=False)

    def white_noise(self, l: Label, *, rng: Any = None,
                    size: int | None = None) -> np.ndarray:
        """White noise of L^2(w dx) on the active dofs: M^-1/2 z with z
        standard normal, of shape (n,) or (n, size), column j of a batch
        being the j-th of `size` successive draws from the generator.  Its
        covariance is M^-1, so <f, white> is `N(0, |f|^2)` for every nodal
        f.  `rng` is anything `np.random.default_rng` accepts."""
        rng = np.random.default_rng(rng)
        s = 1.0 / np.sqrt(self.mass(l))
        if size is None:
            return s * rng.standard_normal(s.size)
        return s[:, None] * rng.standard_normal((int(size), s.size)).T

    def apply_power(self, l: Label, s: float, v: ArrayLike) -> np.ndarray:
        """A^s v for the member labelled l on active-dof values v of shape
        (n,) or (n, k), through the full eigendecomposition
        A^s = Phi Theta^s Phi^T M; any real power, the spectrum being
        bounded below by 1."""
        theta, Phi = self.eig(l)
        v = self._check(l, v)
        mass = self.mass(l)
        if v.ndim == 1:
            return Phi @ (theta ** s * (Phi.T @ (mass * v)))
        c = Phi.T @ (mass[:, None] * v)
        return Phi @ (theta[:, None] ** s * c)

    def inner(self, l: Label, s: float, u: ArrayLike, v: ArrayLike) -> float:
        """The H^s inner product <u, A^s v> of the member labelled l with
        the L^2(w dx) pairing; s = 0 is the weighted L^2 inner product."""
        u = np.asarray(u, dtype=float)
        return float(u @ (self.mass(l) * self.apply_power(l, s, v)))

    def logdet(self, l: Label, *, s: float = 1.0) -> float:
        """log det A^s = s sum log theta for the member labelled l, from
        the values-only cache."""
        return float(s * np.log(self.eigvalsh(l)).sum())

    def __repr__(self) -> str:
        kind = ("interval" if self.weight == "one"
                else "annulus" if not self._centre
                else "ball" if self.weight == "r2" else "disk")
        return (f"SeparableOperatorFamily({kind}, {self.mesh.nglob} nodes, "
                f"weight={self.weight!r}, lateral={self.lateral!r}, "
                f"index={self.index!r}, cached {self.index}s: {self.cached()})")

