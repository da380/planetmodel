"""firstelastic.py -- the first elasticity tensor, and its action.

planetmodel carries two elasticity tensors and refuses to confuse them.
The **second** tensor CC is what a deck provides and what `ElasticField`
stores: rank 4, weight 1, with the full minor and major symmetries, so
Voigt 6x6 is faithful and `Symmetry` counts its moduli.  The **first**
tensor is what a *referential* weak form contracts against.  Its factors
of the deformation gradient land asymmetrically across the slots and, in
a pre-stressed body, the equilibrium stress enters as well, so only the
major symmetry survives and there is no Voigt form.  Hence the storage
rule: **store the second, compute the first's action on demand.**

Appendix B.8.4, verbatim.  With `F_iA = dm_i/dX_A`, `J = det F`, CC the
stored referential second elasticity tensor and `S_AB` the referential
equilibrium stress -- the **second Piola-Kirchhoff** stress, forced by
its character (rank 2, weight 1) pushing it to the Cauchy stress
`sigma = F S F^T / J` -- the referential first elasticity tensor is

    A_iAjB = d_ij S_AB + F_iC F_jD CC_CADB

with slot order `(i, A, j, B)`: physical, reference, physical,
reference.  Its action on a referential displacement gradient
`G_jB = du_j/dX_B` is

    (A G)_iA = (G S)_iA + (F K)_iA,     K_CA = CC_CADB (F^T G)_DB

and `apply` computes it that way -- form `M = F^T G`, symmetrise it,
contract with the stored Voigt matrix, then one multiply by F and one by
S -- so no 81-component array is ever built on the hot path.  `evaluate`
materialises the tensor and is the oracle `apply` is measured against.

Referred to the equilibrium configuration -- both reference slots pushed
forward -- this is the seismological tensor of Maitra & Al-Attar (2021,
GJI 225, 378-415),

    LAM_ijkl = c_ijkl + d_ik sigma_jl,   c = push_forward(CC),

which `equilibrium_form` returns.  Its minor symmetry fails by exactly
`d_ik sigma_jl - d_jk sigma_il` and by nothing else: the two-tensor
distinction, made quantitative.

Frames.  The algebra runs in Cartesian components, the frame the mapping
speaks: CC is read from the source with `frame="cartesian"`, F comes
from `mapping.deformation_gradient(X)`, and S is read in its own
(spherical) frame and rotated with the local frame `R(X)`.
`frame="spherical"` then rotates **every** slot back with `R(X)^T`,
physical and reference alike.  For a radial mapping direction is
preserved, so the frame at X is also the frame at m(X); for a general
mapping the physical slots are reported in the frame at the *reference*
point, a choice stated here so it is visible.  The value has no Voigt
form, so `evaluate` returns `(..., 3, 3, 3, 3)` and there is no `voigt=`
argument, which is what `FIRST_ELASTIC = Character(4, 1, voigt=False)`
records.
"""
from __future__ import annotations

import numpy as np

from .character import ELASTIC, FIRST_ELASTIC, STRESS
from .fields.composite import FieldBase
from .frames import cartesian_points, rotate_slots, spherical_frame
from .materials import check_frame, tensor_to_voigt, voigt_to_tensor
from .pushforward import full_components, push_forward
from .units import Dimensions

__all__ = ["FirstElasticField"]

#: The identity, built once: it appears in both d_ij S_AB and d_ik sigma_jl.
_EYE = np.eye(3)


class FirstElasticField(FieldBase):
    """The referential first elasticity tensor of a body under a mapping.

    `source` is the stored **second** elasticity tensor: an
    `ElasticField`, a `PulledBackElasticField`, or any rank-4 Field
    whose components at full rank can be had in the Cartesian frame
    (through `evaluate_full`, or by expanding its Voigt `evaluate`).
    `mapping` is the mapping from the reference body to the physical
    one.  `stress` is optional: the referential equilibrium stress,
    character `STRESS`, understood as the second Piola-Kirchhoff stress
    in the source's frame convention -- spherical components, Voigt
    `(..., 6)` or full `(..., 3, 3)`, both accepted.  Omitted, the body
    is unstressed and only the elastic term survives.

    Nothing is precomputed: the field holds its three operands and does
    the algebra in `evaluate`, `apply` and `equilibrium_form`, so a
    change to any of them is seen at the next call.  All three
    coordinates are required -- F depends on direction whatever the
    moduli do -- and `is_radial` is False for the same reason.
    """

    character = FIRST_ELASTIC

    def __init__(self, source, mapping, *, stress=None,
                 name: str | None = None) -> None:
        """Bind the second elasticity tensor, the mapping and the stress."""
        char = getattr(source, "character", None)
        if char is None or char.rank != 4 or not char.voigt:
            raise ValueError(
                "source must be a *second* elasticity tensor -- rank 4 with "
                "the full minor and major symmetries, character ELASTIC -- "
                f"since the first tensor is built from it; got {char}")
        if stress is not None:
            schar = getattr(stress, "character", None)
            if schar != STRESS:
                raise ValueError(
                    "stress must be the referential equilibrium stress, "
                    f"character STRESS ({STRESS}); got {schar}.  It is the "
                    "second Piola-Kirchhoff stress: that is what rank 2, "
                    "weight 1 means, since that rule pushes it forward to "
                    "the Cauchy stress F S F^T / J")
        self.source = source
        self.mapping = mapping
        self.stress = stress
        self.skeleton = source.skeleton
        src_name = getattr(source, "name", None)
        self.name = name if name is not None else (
            f"{src_name}_first" if src_name else None)

    @property
    def dimensions(self):
        """An elasticity tensor's components are moduli: pressure.

        The stress term carries the same dimensions, which is why it can
        be added to the elastic one at all.
        """
        return Dimensions.MODULUS

    @property
    def is_radial(self) -> bool:
        """False: F depends on direction even where the moduli do not."""
        return False

    # -- geometry ----------------------------------------------------------

    def _points(self, r, theta, phi):
        """(r, theta, phi) broadcast, the Cartesian points X, and R(X)."""
        if theta is None or phi is None:
            raise ValueError(
                f"{type(self).__name__} needs theta and phi as well as r: "
                "every term carries a factor of F, and F depends on "
                "direction even where the moduli do not")
        r, theta, phi = np.broadcast_arrays(
            np.asarray(r, dtype=float), np.asarray(theta, dtype=float),
            np.asarray(phi, dtype=float))
        return (r, theta, phi, cartesian_points(r, theta, phi),
                spherical_frame(theta, phi))

    def _stress_cartesian(self, r, theta, phi, R, *, layer, side):
        """S in Cartesian components, or None where there is no stress.

        The stress is read in its own frame -- spherical, as everywhere
        else in the model layer -- and rotated here with R(X), rather
        than asked for `frame="cartesian"`, so that one rule serves
        `evaluate`, `apply` and `equilibrium_form` and they cannot drift
        apart.  Voigt `(..., 6)` and full `(..., 3, 3)` are both
        accepted, as in pullback.py; the trailing shapes distinguish
        themselves.
        """
        if self.stress is None:
            return None
        vals = np.asarray(
            self.stress.evaluate(r, theta, phi, layer=layer, side=side),
            dtype=float)
        if vals.shape[-1:] == (6,):
            vals = voigt_to_tensor(vals, rank=2)
        elif vals.shape[-2:] != (3, 3):
            raise ValueError(
                "a stress value has trailing shape (6,) in Voigt form or "
                f"(3, 3) in full, got {vals.shape}")
        return rotate_slots(vals, R, 2)             # spherical -> Cartesian

    def _F(self, X):
        """The deformation gradient at the reference points."""
        return np.asarray(self.mapping.deformation_gradient(X), dtype=float)

    def _second(self, r, theta, phi, *, layer, side):
        """CC at full rank, in Cartesian components."""
        return full_components(self.source, self.source.character,
                               r, theta, phi, layer=layer, side=side,
                               frame="cartesian")

    # -- the tensor, the action, and the equilibrium form -------------------

    def evaluate(self, r, theta=None, phi=None, *, layer=None,
                 side: str = "upper", frame: str = "spherical"):
        """The materialised A_iAjB = d_ij S_AB + F_iC F_jD CC_CADB.

        Shape: the broadcast shape of (r, theta, phi) followed by
        (3, 3, 3, 3), with slot order (i, A, j, B) -- physical,
        reference, physical, reference.  There is no Voigt form to ask
        for: the minor symmetries fail whenever S is non-zero or F is
        not the identity, which is precisely the two-tensor distinction,
        so a Voigt reduction would silently keep one slot of each pair
        and throw the evidence away.

        This is the reference implementation, not the hot path.  A
        consumer assembling a weak form wants `apply`, which contracts
        the same tensor against a displacement gradient without ever
        building 81 components per point.
        """
        check_frame(frame)
        r, theta, phi, X, R = self._points(r, theta, phi)

        # The source first, so a radius outside the skeleton is refused
        # by the field that owns the skeleton rather than silently used.
        CC = self._second(r, theta, phi, layer=layer, side=side)
        F = self._F(X)
        out = np.einsum("...iC,...jD,...CADB->...iAjB", F, F, CC,
                        optimize=True)

        S = self._stress_cartesian(r, theta, phi, R, layer=layer, side=side)
        if S is not None:
            out = out + np.einsum("ij,...AB->...iAjB", _EYE, S)

        if frame == "spherical":
            out = rotate_slots(out, np.swapaxes(R, -1, -2), 4)
        return out

    evaluate_full = evaluate

    def apply(self, grad_u, r, theta=None, phi=None, *, layer=None,
              side: str = "upper", frame: str = "spherical"):
        """The action (A G)_iA on a referential displacement gradient.

        `grad_u` is `G_jB = du_j/dX_B`, shape (..., 3, 3), broadcasting
        with the points and given in the frame `frame` names; the result
        `(A G)_iA` has the same shape and is returned in that frame.
        Appendix B.8.4's route, exactly:

            M = F^T G,  symmetrised;  K_CA = CC_CADB M_DB;
            (A G) = F K + G S.

        The contraction with CC is done through the stored Voigt matrix
        in the source's **own** frame, so no Bond rotation is needed:
        the symmetric M is rotated into the spherical frame, reduced to
        a Voigt 6-vector with its three shear components doubled -- the
        matrix holds tensor components and each off-diagonal pair is
        counted twice in the sum -- multiplied by the 6x6, expanded back
        to (3, 3) and rotated to Cartesian.  Six multiply-adds per point
        in place of eighty-one, and nothing of rank 4 is allocated.

        Only sym M reaches CC, because CC is symmetric in the (D, B)
        pair it contracts; the antisymmetric part of the displacement
        gradient is an infinitesimal rotation and does not strain the
        material.  The stress term keeps the *full* G, which is exactly
        where the minor symmetry of A goes.
        """
        check_frame(frame)
        r, theta, phi, X, R = self._points(r, theta, phi)
        G = np.asarray(grad_u, dtype=float)
        if G.shape[-2:] != (3, 3):
            raise ValueError(
                "a displacement gradient G_jB = du_j/dX_B has trailing shape "
                f"(3, 3), got {G.shape}")
        if frame == "spherical":
            G = rotate_slots(G, R, 2)               # spherical -> Cartesian

        # The Voigt matrix in the frame it is native to, and the source
        # evaluated first so it owns the skeleton check.
        V = np.asarray(
            self.source.evaluate(r, theta, phi, layer=layer, side=side,
                                 frame="spherical"), dtype=float)
        F = self._F(X)

        M = np.einsum("...kC,...kB->...CB", F, G)        # (F^T G)_CB
        M = 0.5 * (M + np.swapaxes(M, -1, -2))
        M = rotate_slots(M, np.swapaxes(R, -1, -2), 2)   # -> spherical
        m = tensor_to_voigt(M, rank=2)
        m = np.concatenate([m[..., :3], 2.0 * m[..., 3:]], axis=-1)
        K = voigt_to_tensor(np.einsum("...ab,...b->...a", V, m), rank=2)
        K = rotate_slots(K, R, 2)                        # -> Cartesian

        out = np.einsum("...iC,...CA->...iA", F, K)
        S = self._stress_cartesian(r, theta, phi, R, layer=layer, side=side)
        if S is not None:
            out = out + np.einsum("...ij,...jA->...iA", G, S)

        if frame == "spherical":
            out = rotate_slots(out, np.swapaxes(R, -1, -2), 2)
        return out

    def equilibrium_form(self, r, theta=None, phi=None, *, layer=None,
                         side: str = "upper", frame: str = "spherical"):
        """LAM_ijkl = c_ijkl + d_ik sigma_jl [extra].

        The first elasticity tensor referred to the equilibrium
        configuration -- both reference slots pushed forward,
        `LAM_ijkl = F_jA F_lB A_iAkB / J` -- which is the seismological
        tensor of Maitra & Al-Attar (2021, GJI 225, 378-415).  It
        assembles from the pushed-forward second tensor
        `c = push_forward(CC, F, J, ELASTIC)` and the Cauchy stress
        `sigma = F S F^T / J`, and both routes agree; that agreement is
        the test.

        Like A it has the major symmetry and only the major symmetry.
        The violation is exact and worth stating, because it is the
        whole content of the two-tensor distinction:

            LAM_ijkl - LAM_jikl = d_ik sigma_jl - d_jk sigma_il

        -- c is minor-symmetric, so every departure comes from the
        stress term, and an unstressed body has none.  Shape is the
        broadcast shape of the points followed by (3, 3, 3, 3), all four
        slots physical, in the frame asked for.
        """
        check_frame(frame)
        r, theta, phi, X, R = self._points(r, theta, phi)

        CC = self._second(r, theta, phi, layer=layer, side=side)
        F = self._F(X)
        J = np.asarray(self.mapping.jacobian(X), dtype=float)
        out = push_forward(CC, F, J, ELASTIC)

        S = self._stress_cartesian(r, theta, phi, R, layer=layer, side=side)
        if S is not None:
            sigma = (np.einsum("...iA,...AB,...jB->...ij", F, S, F)
                     / J[..., None, None])
            out = out + np.einsum("ik,...jl->...ijkl", _EYE, sigma)

        if frame == "spherical":
            out = rotate_slots(out, np.swapaxes(R, -1, -2), 4)
        return out

    def __repr__(self) -> str:
        nm = f" {self.name!r}" if self.name else ""
        stress = "" if self.stress is None else f", stress={self.stress!r}"
        return (f"FirstElasticField({self.source!r}, {self.mapping!r}"
                f"{stress}{nm})")
