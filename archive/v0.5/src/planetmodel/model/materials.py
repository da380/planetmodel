"""materials.py -- moduli as the canonical elastic description.

Velocities have no transformation law (see character.py), so what a
model stores is (rho, A): a density and an elastic tensor.  Deck readers
convert on load, and velocities become derived views.

The transversely isotropic parameterisation, with the symmetry axis
radial (the seismological convention, "VTI"), is the five moduli

    A = rho vph^2      C = rho vpv^2      L = rho vsv^2
    N = rho vsh^2      F = eta (A - 2 L)

which collapse to the isotropic case at vph = vpv, vsh = vsv, eta = 1,
where A = C = kappa + 4 mu / 3, L = N = mu and F = A - 2 L = kappa -
2 mu / 3 = lambda.

The two elasticity tensors
-------------------------

planetmodel distinguishes them, because they are different objects:

* The **second** elasticity tensor has the full minor and major
  symmetries, so Voigt 6x6 is faithful and the Symmetry classes count
  its independent moduli.  It is what decks provide and what
  ElasticField stores.  Push-forward wraps all four slots identically
  and preserves these symmetries -- the result is still a second
  elasticity tensor, though its symmetry group is conjugated, so a
  physically isotropic tensor is no longer isotropic with respect to
  the Euclidean metric.

* The **first** elasticity tensor is what referential weak forms
  consume.  Its factors of F land asymmetrically across the slots and,
  in a pre-stressed configuration, equilibrium-stress terms enter, so
  in general only the major symmetry survives and there is no Voigt
  form.

The storage rule follows: **store the second, compute the first's
action on demand**.  The first tensor is `firstelastic.FirstElasticField`
(Appendix B.8.4; Maitra & Al-Attar 2021, GJI 225, 378-415), built from
an ElasticField and a mapping and consumed through its action on a
displacement gradient rather than by materialising 81 components.

Frames and Voigt ordering
-------------------------

Voigt ordering here is the usual (11, 22, 33, 23, 13, 12), and the
matrix holds tensor components with **no** engineering-strain factors --
which is why the Voigt matrix of an isotropic medium reads mu, not 2 mu,
on the shear diagonal.  Components are given in the frame the
coordinates imply: `evaluate` returns the spherical frame, in which
1, 2, 3 = r, theta, phi and the VTI symmetry axis is index 1, e_r; the
Cartesian form is its Bond rotation (bond_matrix, Appendix B.9) by
R = [e_r, e_theta, e_phi] as columns (frames.spherical_frame).

Complex moduli and the time convention
--------------------------------------

A viscoelastic modulus is a complex number, and every routine here that
builds or reshapes a tensor keeps the dtype it is given, so the Voigt
matrix, the expansion to full components and the Bond rotation all
accept complex moduli unchanged.  The sign convention throughout
planetmodel is a time dependence exp(+i omega t), so the Laplace
variable is s = i omega and a lossy modulus has a **positive**
imaginary part: Im mu(omega) > 0 for omega > 0.  A consumer working
with exp(-i omega t) conjugates on the way in.
"""
from __future__ import annotations

import numpy as np

from .character import ELASTIC, Symmetry
from .fields.composite import FieldBase
from .fields.layer_function import combine_layer_functions
from .frames import spherical_frame
from .units import Dimensions

__all__ = [
    "moduli_from_velocities", "velocities_from_moduli",
    "voigt_matrix", "bond_matrix", "kappa_mu_from_moduli", "ElasticField",
    "voigt_to_tensor", "tensor_to_voigt", "MODULI_NAMES", "check_frame",
]

#: The frames a field's components may be asked for in.
FRAMES = ("spherical", "cartesian")


def check_frame(frame: str) -> None:
    """Refuse a frame name that is neither 'spherical' nor 'cartesian'."""
    if frame not in FRAMES:
        raise ValueError(
            f"unknown frame {frame!r}: values are given in the 'spherical' "
            "frame or the 'cartesian' one")


def _as_array(v) -> np.ndarray:
    """An array that is float64 when real and complex128 when complex."""
    v = np.asarray(v)
    return v if np.iscomplexobj(v) else np.asarray(v, dtype=float)

#: The five TI moduli, in the order used throughout.
MODULI_NAMES = ("A", "C", "F", "L", "N")

#: Voigt index pairs, in order.
_VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def moduli_from_velocities(rho, vpv, vsv, *, vph=None, vsh=None, eta=None):
    """The TI moduli (A, C, F, L, N) from density and velocities.

    With vph, vsh and eta omitted the medium is isotropic: vph = vpv,
    vsh = vsv, eta = 1, giving A = C and F = A - 2 L.  All arguments
    broadcast; the result is a dict keyed by MODULI_NAMES.
    """
    rho = np.asarray(rho, dtype=float)
    vpv = np.asarray(vpv, dtype=float)
    vsv = np.asarray(vsv, dtype=float)
    vph = vpv if vph is None else np.asarray(vph, dtype=float)
    vsh = vsv if vsh is None else np.asarray(vsh, dtype=float)
    eta = 1.0 if eta is None else np.asarray(eta, dtype=float)

    A = rho * vph ** 2
    C = rho * vpv ** 2
    L = rho * vsv ** 2
    N = rho * vsh ** 2
    F = eta * (A - 2.0 * L)
    return {"A": A, "C": C, "F": F, "L": L, "N": N}


def velocities_from_moduli(rho, A, C, F, L, N):
    """Velocities and eta from density and the TI moduli.

    The inverse of moduli_from_velocities, and the reason velocities are
    *derived*: the square roots make this non-polynomial, so an exact
    polynomial model stays exact in the moduli and becomes merely
    pointwise-exact in the velocities.
    """
    rho = np.asarray(rho, dtype=float)
    A, C = np.asarray(A, dtype=float), np.asarray(C, dtype=float)
    F, L = np.asarray(F, dtype=float), np.asarray(L, dtype=float)
    N = np.asarray(N, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        safe = np.where(rho > 0.0, rho, 1.0)
        denom = A - 2.0 * L
        return {
            "vph": np.sqrt(np.maximum(A / safe, 0.0)),
            "vpv": np.sqrt(np.maximum(C / safe, 0.0)),
            "vsv": np.sqrt(np.maximum(L / safe, 0.0)),
            "vsh": np.sqrt(np.maximum(N / safe, 0.0)),
            "eta": np.where(denom != 0.0, F / np.where(denom != 0.0, denom, 1.0),
                            1.0),
        }


def kappa_mu_from_moduli(A, C, F, L, N):
    """The Voigt-averaged isotropic (kappa, mu) of a TI medium.

        kappa = (4A + C + 4F - 4N) / 9
        mu    = (A + C - 2F + 5N + 6L) / 15

    which reduce exactly to rho(vp^2 - 4 vs^2 / 3) and rho vs^2 where
    the medium is isotropic.
    """
    A, C = np.asarray(A, dtype=float), np.asarray(C, dtype=float)
    F, L = np.asarray(F, dtype=float), np.asarray(L, dtype=float)
    N = np.asarray(N, dtype=float)
    kappa = (4.0 * A + C + 4.0 * F - 4.0 * N) / 9.0
    mu = (A + C - 2.0 * F + 5.0 * N + 6.0 * L) / 15.0
    return kappa, mu


def voigt_matrix(symmetry: Symmetry, components: dict) -> np.ndarray:
    """The Voigt 6x6 of a second elasticity tensor.

    `components` holds the independent moduli of the class: (kappa, mu)
    for ISOTROPIC, (A, C, F, L, N) for VTI.  Values broadcast, and the
    result has shape broadcast_shape + (6, 6).

    The VTI form in this library's frame, (1, 2, 3) = (r, theta, phi),
    so the symmetry axis -- radial -- is index **1**, and in the Voigt
    order (11, 22, 33, 23, 13, 12) = (rr, tt, pp, tp, rp, rt):

        [ C      F      F    .  .  . ]
        [ F      A      A-2N .  .  . ]
        [ F      A-2N   A    .  .  . ]
        [ .      .      .    N  .  . ]
        [ .      .      .    .  L  . ]
        [ .      .      .    .  .  L ]

    The seismological tables put the axis at index 3 because their
    third axis is vertical; here the first axis is.  The layout is
    pinned against the invariant form of Appendix B.8.3 with n = e_r.
    """
    if symmetry is Symmetry.ISOTROPIC:
        kappa = _as_array(components["kappa"])
        mu = _as_array(components["mu"])
        lam = kappa - 2.0 * mu / 3.0
        shape = np.broadcast(kappa, mu).shape
        out = np.zeros(shape + (6, 6), dtype=np.result_type(kappa, mu))
        for i in range(3):
            for j in range(3):
                out[..., i, j] = lam
            out[..., i, i] = lam + 2.0 * mu
            out[..., 3 + i, 3 + i] = mu
        return out

    if symmetry is Symmetry.VTI:
        A, C, F, L, N = (_as_array(components[k]) for k in MODULI_NAMES)
        shape = np.broadcast(A, C, F, L, N).shape
        out = np.zeros(shape + (6, 6), dtype=np.result_type(A, C, F, L, N))
        out[..., 0, 0] = C
        out[..., 1, 1] = out[..., 2, 2] = A
        out[..., 0, 1] = out[..., 1, 0] = F
        out[..., 0, 2] = out[..., 2, 0] = F
        out[..., 1, 2] = out[..., 2, 1] = A - 2.0 * N
        out[..., 3, 3] = N
        out[..., 4, 4] = out[..., 5, 5] = L
        return out

    raise NotImplementedError(
        f"voigt_matrix does not yet build {symmetry}; ISOTROPIC and VTI are "
        "supported, ORTHOTROPIC and GENERAL arrive with the 3D field work")


def bond_matrix(R) -> np.ndarray:
    """The 6x6 Bond matrix of an orthogonal R (Appendix B.9).

    A frame change with orthogonal R takes tensor components as

        c'_{ijkl} = R_{ia} R_{jb} R_{kc} R_{ld} c_{abcd},

    and the Voigt matrix of a second elasticity tensor -- holding those
    same components, with no engineering-strain factors -- therefore
    transforms as V' = M V M^T with

        M[a, b] = R[i, k] R[j, l] + R[i, l] R[j, k]   (k != l)
                = R[i, k] R[j, k]                     (k == l)

    for a <-> (i, j) and b <-> (k, l) in the order (11, 22, 33, 23, 13,
    12).  The single expression reproduces both blocks of the usual
    tabulation, factors of two included: at i = j and k != l it gives
    2 R[i, k] R[i, l].  A rank-2 Voigt vector transforms by the same M
    applied once, v' = M v.

    R broadcasts: the result has R.shape[:-2] + (6, 6).  For the frame
    of a spherical-to-Cartesian change, R = [e_r, e_theta, e_phi] as
    columns (frames.spherical_frame).
    """
    R = np.asarray(R, dtype=float)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotations of shape (..., 3, 3), got {R.shape}")
    M = np.empty(R.shape[:-2] + (6, 6))
    for a, (i, j) in enumerate(_VOIGT_PAIRS):
        for b, (k, m) in enumerate(_VOIGT_PAIRS):
            term = R[..., i, k] * R[..., j, m]
            if k != m:
                term = term + R[..., i, m] * R[..., j, k]
            M[..., a, b] = term
    return M


def voigt_to_tensor(v, *, rank: int = 4) -> np.ndarray:
    """Expand a Voigt vector or matrix to full components.

    rank 4: (..., 6, 6) -> (..., 3, 3, 3, 3), every symmetry-related
    slot filled with the same number, so the result has the full minor
    and major symmetries by construction.
    rank 2: (..., 6) -> (..., 3, 3), symmetric.

    `rank` is explicit rather than inferred because the trailing shapes
    do not distinguish themselves: an array of shape (6, 6) is a Voigt
    matrix at one point and six Voigt vectors at six points, and only
    the caller's character says which.  The generic push-forward wants
    full components (a Voigt matrix is not a tensor to contract F
    against), so this is the door it goes through.
    """
    v = _as_array(v)
    if rank == 4:
        if v.shape[-2:] != (6, 6):
            raise ValueError(
                f"a rank-4 Voigt matrix has trailing shape (6, 6), got {v.shape}")
        out = np.zeros(v.shape[:-2] + (3, 3, 3, 3), dtype=v.dtype)
        for a, (i, j) in enumerate(_VOIGT_PAIRS):
            for b, (k, m) in enumerate(_VOIGT_PAIRS):
                val = v[..., a, b]
                for ii, jj in ((i, j), (j, i)):
                    for kk, mm in ((k, m), (m, k)):
                        out[..., ii, jj, kk, mm] = val
        return out
    if rank == 2:
        if v.shape[-1:] != (6,):
            raise ValueError(
                f"a rank-2 Voigt vector has trailing shape (6,), got {v.shape}")
        out = np.zeros(v.shape[:-1] + (3, 3), dtype=v.dtype)
        for a, (i, j) in enumerate(_VOIGT_PAIRS):
            out[..., i, j] = out[..., j, i] = v[..., a]
        return out
    raise ValueError(f"only ranks 2 and 4 are Voigt-reducible, got {rank}")


def tensor_to_voigt(t, *, rank: int = 4) -> np.ndarray:
    """Reduce full components to Voigt, by reading the six index pairs.

    The inverse of voigt_to_tensor on tensors that have the symmetries;
    on anything else it is a *projection*, silently keeping one slot out
    of each symmetry class, so the caller is responsible for knowing
    that the reduction is faithful.  Push-forward wraps every slot
    identically and so preserves the symmetries (Appendix B.8.1), which
    is exactly what licenses reducing a pushed-forward second elasticity
    tensor; pushforward.check_tensor_symmetries checks that where it
    matters.
    """
    t = _as_array(t)
    if rank == 4:
        if t.shape[-4:] != (3, 3, 3, 3):
            raise ValueError(
                f"a rank-4 tensor has trailing shape (3, 3, 3, 3), got {t.shape}")
        out = np.empty(t.shape[:-4] + (6, 6), dtype=t.dtype)
        for a, (i, j) in enumerate(_VOIGT_PAIRS):
            for b, (k, m) in enumerate(_VOIGT_PAIRS):
                out[..., a, b] = t[..., i, j, k, m]
        return out
    if rank == 2:
        if t.shape[-2:] != (3, 3):
            raise ValueError(
                f"a rank-2 tensor has trailing shape (3, 3), got {t.shape}")
        out = np.empty(t.shape[:-2] + (6,), dtype=t.dtype)
        for a, (i, j) in enumerate(_VOIGT_PAIRS):
            out[..., a] = t[..., i, j]
        return out
    raise ValueError(f"only ranks 2 and 4 are Voigt-reducible, got {rank}")


def _combine(terms, name: str):
    """A RadialField that is a fixed linear combination of other fields.

    `terms` is a sequence of (field, coefficient) pairs -- a sequence
    rather than a mapping because the same field may legitimately appear
    twice, as it does whenever a medium has kappa and mu equal.

    Layer by layer, on the layer functions themselves, so an exact model
    stays exact: the combination is done on polynomial coefficients
    wherever the operands allow it.  The field algebra (`kappa + 4/3 *
    mu`) would build a lazy composite instead, which evaluates the same
    numbers but has no exact derivative or integral to offer a mesh.
    """
    from .fields.radial import RadialField

    sk = terms[0][0].skeleton
    funcs = [None if any(f.functions[i] is None for f, _ in terms)
             else combine_layer_functions([(c, f.functions[i]) for f, c in terms])
             for i in range(sk.nlayers)]
    return RadialField(sk, funcs, name=name, character=terms[0][0].character,
                       dimensions=getattr(terms[0][0], "dimensions", None))


class ElasticField(FieldBase):
    """A second elasticity tensor: character ELASTIC, stored by its moduli.

    The independent moduli of the symmetry class are held as Fields --
    two for ISOTROPIC, five for VTI -- and the Voigt 6x6 is built only
    when evaluate() is called.  Storage stays at the size the physics
    justifies; nothing materialises 21 components unless asked.

    evaluate(..., voigt=False) returns the full (..., 3, 3, 3, 3)
    tensor, which is what a generic push-forward contracts against.
    """

    character = ELASTIC

    def __init__(self, symmetry: Symmetry, components: dict,
                 *, name: str | None = None) -> None:
        """Bind the independent moduli fields of a symmetry class."""
        expected = {Symmetry.ISOTROPIC: ("kappa", "mu"),
                    Symmetry.VTI: MODULI_NAMES}.get(symmetry)
        if expected is None:
            raise NotImplementedError(
                f"ElasticField does not yet store {symmetry}; ISOTROPIC and "
                "VTI are supported")
        missing = [k for k in expected if k not in components]
        if missing:
            raise ValueError(f"{symmetry} needs {list(expected)}; missing {missing}")
        extra = [k for k in components if k not in expected]
        if extra:
            raise ValueError(f"{symmetry} takes only {list(expected)}; got {extra}")

        skeletons = [f.skeleton for f in components.values()]
        if any(s != skeletons[0] for s in skeletons[1:]):
            raise ValueError("all moduli must share one skeleton")

        self.symmetry = symmetry
        self.components = dict(components)
        self.name = name
        self._sk = skeletons[0]

    @property
    def skeleton(self):
        """The Skeleton shared by every modulus."""
        return self._sk

    @property
    def dimensions(self):
        """An elastic tensor's components are moduli: pressure."""
        return Dimensions.MODULUS

    @property
    def moduli_names(self) -> tuple[str, ...]:
        """The independent moduli held, in canonical order."""
        return tuple(self.components)

    @property
    def domain(self) -> tuple[int, ...]:
        """The layers every modulus is defined on."""
        doms = [set(getattr(f, "domain", range(self._sk.nlayers)))
                for f in self.components.values()]
        return tuple(sorted(set.intersection(*doms)))

    def restricted(self, layer) -> "ElasticField":
        """The tensor on one layer: the same symmetry of restricted moduli."""
        i = self._sk.layer_index(layer)
        return ElasticField(self.symmetry,
                            {k: f.restricted(i) for k, f in self.components.items()},
                            name=self.name)

    def on_interval(self, lo: float, hi: float) -> "ElasticField":
        """The same symmetry of moduli re-stated on [lo, hi]."""
        return ElasticField(self.symmetry,
                            {k: f.on_interval(lo, hi)
                             for k, f in self.components.items()},
                            name=self.name)

    def rescaled(self, convert, old, new):
        """The same symmetry of moduli, each converted."""
        return ElasticField(self.symmetry,
                            {k: convert(f) for k, f in self.components.items()},
                            name=self.name)

    @classmethod
    def _assembled(cls, skeleton, pieces, *, name=None):
        """One tensor from per-layer tensors of one symmetry class."""
        from .fields.layerwise import assemble
        first = pieces[0]
        if not all(isinstance(p, ElasticField) and p.symmetry is first.symmetry
                   and p.moduli_names == first.moduli_names for p in pieces):
            return NotImplemented
        comps = {k: assemble(skeleton, [p.components[k] for p in pieces])
                 for k in first.moduli_names}
        return cls(first.symmetry, comps,
                   name=name if name is not None else first.name)

    @property
    def is_radial(self) -> bool:
        """Whether the stored moduli are functions of radius alone.

        A statement about the tensor's components *in its own frame*.
        A VTI body whose moduli are radial is radial in that sense, and
        its Cartesian components still vary with direction, because the
        frame does -- which is exactly why evaluate(frame="cartesian")
        insists on the angles.
        """
        return all(bool(getattr(f, "is_radial", False))
                   for f in self.components.values())

    def evaluate_full(self, r, theta=None, phi=None, *, layer=None,
                      side: str = "upper", frame: str = "spherical"):
        """The full (..., 3, 3, 3, 3) tensor: `evaluate` with `voigt=False`."""
        return self.evaluate(r, theta, phi, layer=layer, side=side,
                             frame=frame, voigt=False)

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical",
                 voigt: bool = True):
        """The elastic tensor at the given points, in the chosen frame.

        Voigt 6x6 by default; voigt=False expands to the full rank-4
        tensor.  Both are built on demand from the stored moduli.

        The Voigt matrix is native to the *spherical* frame, where the
        VTI symmetry axis is index 1 = e_r and the moduli have their
        seismological meaning.  frame="cartesian" returns its Bond
        rotation M V M^T with M = bond_matrix(R) and R the local frame
        [e_r, e_theta, e_phi] as columns (Appendix B.9), so the angles
        are then required: the components depend on direction even where
        the moduli do not.  A physically isotropic tensor is the one
        exception -- it is the same matrix in every frame, so the angles
        may be omitted and the values are returned unrotated rather than
        rotated and rounded.

        voigt=False expands *after* the rotation, which is the same
        tensor as rotating the expansion: the Bond transformation is the
        Voigt shadow of the four-slot rotation, by construction.
        """
        check_frame(frame)
        vals = {k: f.evaluate(r, theta, phi, layer=layer, side=side)
                for k, f in self.components.items()}
        v = voigt_matrix(self.symmetry, vals)

        if frame == "cartesian" and self.symmetry is not Symmetry.ISOTROPIC:
            if theta is None or phi is None:
                raise ValueError(
                    f"frame='cartesian' needs theta and phi: the Cartesian "
                    f"components of a {self.symmetry.name.lower()} tensor "
                    "depend on direction, because the frame carrying its "
                    "symmetry axis does")
            R = spherical_frame(theta, phi)
            R = np.broadcast_to(R, v.shape[:-2] + (3, 3))
            M = bond_matrix(R)
            v = M @ v @ np.swapaxes(M, -1, -2)

        return v if voigt else voigt_to_tensor(v)

    def as_symmetry(self, symmetry: Symmetry) -> "ElasticField":
        """The same medium, described in a wider symmetry class.

        Promotion only: a VTI medium is a special case of a more general
        class, but a general medium is not a VTI one, so narrowing is
        refused rather than silently projected.
        """
        if symmetry is self.symmetry:
            return self
        if symmetry.n_independent < self.symmetry.n_independent:
            raise ValueError(
                f"cannot narrow {self.symmetry} to {symmetry}; that would "
                "discard moduli rather than re-describe them")
        if self.symmetry is Symmetry.ISOTROPIC and symmetry is Symmetry.VTI:
            kappa, mu = self.components["kappa"], self.components["mu"]
            A = _combine([(kappa, 1.0), (mu, 4.0 / 3.0)], "A")
            F = _combine([(kappa, 1.0), (mu, -2.0 / 3.0)], "F")
            return ElasticField(Symmetry.VTI, {
                "A": A, "C": A, "F": F, "L": mu, "N": mu}, name=self.name)
        raise NotImplementedError(
            f"promotion {self.symmetry} -> {symmetry} arrives with the 3D "
            "field work")

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        return (f"ElasticField({self.symmetry.name.lower()}{nm}, "
                f"moduli={list(self.components)})")
