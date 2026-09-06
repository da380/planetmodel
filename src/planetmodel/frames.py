"""The local spherical frame, and tensors moved between frames.

Fields speak (r, theta, phi) and give their components in the local
orthonormal frame (e_r, e_theta, e_phi); mappings speak Cartesian points
and give Cartesian components.  This module is where the two meet.  With
theta the colatitude and phi the longitude, both in radians,

    e_r     = (sin th cos ph,  sin th sin ph,  cos th)
    e_theta = (cos th cos ph,  cos th sin ph, -sin th)
    e_phi   = (-sin ph,        cos ph,         0)
    R       = [e_r, e_theta, e_phi]      as columns, so R is orthogonal

and R takes spherical-frame components to Cartesian ones: a vector as
R v, a rank-2 tensor as R V R^T, and in general one factor of R on every
slot, which is what `rotate_slots` does.  The inverse direction is R^T.

Symmetric rank-2 and rank-4 tensors are also carried in Voigt form, the
six index pairs in the order (11, 22, 33, 23, 13, 12) with no
engineering-strain factors; `bond_matrix` is the (6, 6) matrix that
moves such components between frames, `voigt_to_tensor` and
`tensor_to_voigt` go between the two forms.

Nothing here depends on the rest of the package.
"""
from __future__ import annotations

from collections.abc import Callable

import numpy as np
from numpy.typing import ArrayLike

__all__ = ["spherical_frame", "spherical_coordinates", "cartesian_points",
           "rotate_slots", "rotation_subscripts", "MAX_RANK",
           "VOIGT_PAIRS", "bond_matrix", "voigt_to_tensor", "tensor_to_voigt",
           "SphericalFunction"]

#: A function of (r, theta, phi), given broadcast float arrays and returning
#: something array-like at each point: a displacement, a formula, a relief's
#: parent.  Every callable the library takes in spherical coordinates has
#: this shape.
type SphericalFunction = Callable[[np.ndarray, np.ndarray, np.ndarray], ArrayLike]

#: Index letters for the output (lowercase) and contracted (uppercase)
#: slots of `rotate_slots`; their length is the rank ceiling.
_OUT = "ijklmn"
_IN = "ABCDEF"

#: The rank ceiling of `rotate_slots`: above it the contraction has
#: 3**(2 rank) intermediate entries and is refused rather than allocated.
MAX_RANK = len(_OUT)


def spherical_frame(theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """The local orthonormal frame at (theta, phi), basis vectors as columns.

    R[..., :, 0] = e_r, R[..., :, 1] = e_theta, R[..., :, 2] = e_phi, in
    Cartesian components.  theta and phi broadcast; the result has the
    broadcast shape followed by (3, 3).
    """
    theta, phi = np.broadcast_arrays(np.asarray(theta, dtype=float),
                                     np.asarray(phi, dtype=float))
    st, ct = np.sin(theta), np.cos(theta)
    sp, cp = np.sin(phi), np.cos(phi)
    e_r = np.stack([st * cp, st * sp, ct], axis=-1)
    e_th = np.stack([ct * cp, ct * sp, -st], axis=-1)
    e_ph = np.stack([-sp, cp, np.zeros_like(sp)], axis=-1)
    return np.stack([e_r, e_th, e_ph], axis=-1)


def spherical_coordinates(X: ArrayLike
                          ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cartesian points -> (r, theta, phi, R), with R the frame there.

    X has shape (..., 3).  At the origin the direction is undefined and
    (theta, phi) = (pi/2, 0) is returned with the frame that implies;
    a caller for whom the origin is special tests r itself.  No length
    floor is involved, so the function means the same thing in metres
    and in non-dimensional units.
    """
    X = np.asarray(X, dtype=float)
    if X.shape[-1] != 3:
        raise ValueError(f"expected points of shape (..., 3), got {X.shape}")
    x, y, z = X[..., 0], X[..., 1], X[..., 2]
    r = np.sqrt(x * x + y * y + z * z)
    safe = np.where(r > 0.0, r, 1.0)
    theta = np.arccos(np.clip(z / safe, -1.0, 1.0))
    phi = np.arctan2(y, x)
    return r, theta, phi, spherical_frame(theta, phi)


def cartesian_points(r: ArrayLike, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
    """The Cartesian points X = r e_r(theta, phi), shape broadcast + (3,)."""
    r, theta, phi = np.broadcast_arrays(np.asarray(r, dtype=float),
                                        np.asarray(theta, dtype=float),
                                        np.asarray(phi, dtype=float))
    st, ct = np.sin(theta), np.cos(theta)
    return np.stack([r * st * np.cos(phi), r * st * np.sin(phi), r * ct],
                    axis=-1)


def rotation_subscripts(rank: int) -> str:
    """The einsum string that puts one matrix factor on every slot.

    Each factor is written [out, contracted], so the same string serves
    a rotation R (new components from old), a deformation gradient F
    (physical from reference) and their inverses without transposing:

        rank 1   ...iA,...A->...i
        rank 2   ...iA,...jB,...AB->...ij
        rank 4   ...iA,...jB,...kC,...lD,...ABCD->...ijkl
    """
    if rank > MAX_RANK:
        raise ValueError(
            f"rank {rank} is beyond what rotate_slots will build: the "
            f"contraction has {rank + 1} operands and 3**{2 * rank} "
            f"intermediate entries; ranks up to {MAX_RANK} are allowed")
    terms = [f"...{_OUT[k]}{_IN[k]}" for k in range(rank)]
    terms.append("..." + _IN[:rank])
    return ",".join(terms) + "->..." + _OUT[:rank]


def rotate_slots(values: ArrayLike, M: ArrayLike, rank: int) -> np.ndarray:
    """One factor of M on every slot: out_{i..} = M_{iA} ... T_{A..}.

    `values` has trailing shape (3,) * rank and `M` is (..., 3, 3), both
    broadcasting over leading axes.  With M = R this takes spherical
    components to Cartesian ones; with M = R^T it takes them back; with
    M = F it is the tensor part of a push-forward.  Complex values stay
    complex.
    """
    if rank == 0:
        return values
    M = np.asarray(M, dtype=float)
    if M.shape[-2:] != (3, 3):
        raise ValueError(f"expected (..., 3, 3) matrices, got {M.shape}")
    return np.einsum(rotation_subscripts(rank), *([M] * rank), values,
                     optimize=True)


#: The index pairs behind the six Voigt slots, in Voigt order.
VOIGT_PAIRS = ((0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1))


def bond_matrix(R: ArrayLike) -> np.ndarray:
    """The (6, 6) Bond matrix of an orthogonal R, shape R.shape[:-2] + (6, 6).

    A frame change with orthogonal R takes tensor components as
    c'_ijkl = R_ia R_jb R_kc R_ld c_abcd, and the Voigt matrix holding
    those same components, with no engineering-strain factors, therefore
    transforms as V' = M V M^T with

        M[a, b] = R[i, k] R[j, l] + R[i, l] R[j, k]   (k != l)
                = R[i, k] R[j, k]                     (k == l)

    for a <-> (i, j) and b <-> (k, l) in Voigt order.  A rank-2 Voigt
    vector transforms by the same M applied once, v' = M v.  With
    R = spherical_frame(theta, phi) this takes spherical Voigt
    components to Cartesian ones; the inverse is bond_matrix of R^T.
    """
    R = np.asarray(R, dtype=float)
    if R.shape[-2:] != (3, 3):
        raise ValueError(f"expected rotations of shape (..., 3, 3), got {R.shape}")
    M = np.empty(R.shape[:-2] + (6, 6))
    for a, (i, j) in enumerate(VOIGT_PAIRS):
        for b, (k, m) in enumerate(VOIGT_PAIRS):
            term = R[..., i, k] * R[..., j, m]
            if k != m:
                term = term + R[..., i, m] * R[..., j, k]
            M[..., a, b] = term
    return M


def voigt_to_tensor(v: ArrayLike, *, rank: int = 4) -> np.ndarray:
    """Expand Voigt components to full ones: (..., 6, 6) -> (..., 3, 3, 3, 3)
    for rank 4, (..., 6) -> (..., 3, 3) for rank 2.

    Every symmetry-related slot receives the same number, so the result
    has the minor and major symmetries by construction.  `rank` is
    explicit because a (6, 6) array is one Voigt matrix or six Voigt
    vectors, and only the caller's character says which.
    """
    v = np.asarray(v)
    if rank == 4:
        if v.shape[-2:] != (6, 6):
            raise ValueError(
                f"a rank-4 Voigt matrix has trailing shape (6, 6), got {v.shape}")
        out = np.zeros(v.shape[:-2] + (3, 3, 3, 3), dtype=v.dtype)
        for a, (i, j) in enumerate(VOIGT_PAIRS):
            for b, (k, m) in enumerate(VOIGT_PAIRS):
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
        for a, (i, j) in enumerate(VOIGT_PAIRS):
            out[..., i, j] = out[..., j, i] = v[..., a]
        return out
    raise ValueError(f"only ranks 2 and 4 have a Voigt form, got {rank}")


def tensor_to_voigt(t: ArrayLike, *, rank: int = 4) -> np.ndarray:
    """Reduce full components to Voigt by reading the six index pairs.

    The inverse of voigt_to_tensor on a tensor with the symmetries; on
    any other it is a projection that keeps one slot of each symmetry
    class, so the caller is responsible for knowing the reduction is
    faithful.
    """
    t = np.asarray(t)
    if rank == 4:
        if t.shape[-4:] != (3, 3, 3, 3):
            raise ValueError(
                f"a rank-4 tensor has trailing shape (3, 3, 3, 3), got {t.shape}")
        out = np.empty(t.shape[:-4] + (6, 6), dtype=t.dtype)
        for a, (i, j) in enumerate(VOIGT_PAIRS):
            for b, (k, m) in enumerate(VOIGT_PAIRS):
                out[..., a, b] = t[..., i, j, k, m]
        return out
    if rank == 2:
        if t.shape[-2:] != (3, 3):
            raise ValueError(
                f"a rank-2 tensor has trailing shape (3, 3), got {t.shape}")
        out = np.empty(t.shape[:-2] + (6,), dtype=t.dtype)
        for a, (i, j) in enumerate(VOIGT_PAIRS):
            out[..., a] = t[..., i, j]
        return out
    raise ValueError(f"only ranks 2 and 4 have a Voigt form, got {rank}")
