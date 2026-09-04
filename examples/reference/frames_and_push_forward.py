"""Frames, the Bond rotation, and the push-forward of tensors.

Components are given in the frame the coordinates imply: `evaluate(r,
theta, phi)` returns them in the local spherical frame `(e_r, e_theta,
e_phi)`, and `frame="cartesian"` rotates them with the frame matrix
`R = [e_r, e_theta, e_phi]` (as columns), one factor of R per tensor
slot.  A Voigt matrix rotates by the Bond matrix of R.  A mapping from
the reference body to the physical one carries a field across by the
push-forward its character dictates: a weight-w rank-k tensor picks up
k factors of the deformation gradient F and a factor J^-w.  The
pull-back is the inverse rule, and the two compose to the identity.

This script checks these rules on random frames and on a random
deformation gradient, for a density, a vector and an elastic tensor.
"""
import numpy as np

from planetmodel import (DENSITY, ELASTIC, PREM, SCALAR, VECTOR, Character,
                         IdentityMapping, bond_matrix, push_forward, pull_back,
                         push_forward_field)
from planetmodel.model.frames import (cartesian_points, rotate_slots,
                                      spherical_coordinates, spherical_frame)
from planetmodel.model.materials import voigt_matrix, voigt_to_tensor, tensor_to_voigt
from planetmodel.model.character import Symmetry

rng = np.random.default_rng(0)
theta, phi = rng.uniform(0.2, 3.0, 4), rng.uniform(-3.0, 3.0, 4)

# -- the frame ----------------------------------------------------------------
R = spherical_frame(theta, phi)                     # (4, 3, 3), columns e_r e_th e_ph
assert np.allclose(np.swapaxes(R, -1, -2) @ R, np.eye(3))       # orthogonal
assert np.allclose(np.linalg.det(R), 1.0)                       # right-handed
X = cartesian_points(1.0, theta, phi)
assert np.allclose(R[..., :, 0], X)                             # e_r is X / |X|
r_back, th_back, ph_back, _ = spherical_coordinates(X)
assert np.allclose(r_back, 1.0) and np.allclose(th_back, theta)
assert np.allclose(np.angle(np.exp(1j * (ph_back - phi))), 0.0)

# A vector's spherical components become Cartesian by one factor of R.
v_sph = np.array([1.0, 0.0, 0.0])                   # e_r in every frame
assert np.allclose(rotate_slots(np.broadcast_to(v_sph, (4, 3)), R, 1), X)

# -- the Bond rotation of a Voigt matrix ---------------------------------------
kappa, mu = 1.3e11, 6.0e10
V = voigt_matrix(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu})   # (6, 6)
M = bond_matrix(R[0])
rotated = M @ V @ M.T
assert np.allclose(rotated, V, atol=1e-10 * np.abs(V).max())   # isotropy: invariant
# For a general tensor the Bond form equals the slot-by-slot rotation.
G = rng.normal(size=(6, 6))
G = 0.5 * (G + G.T)
full = rotate_slots(voigt_to_tensor(G), R[0], 4)
assert np.allclose(tensor_to_voigt(full), bond_matrix(R[0]) @ G @ bond_matrix(R[0]).T)
# The inverse rotation is bond(R^T), not M^T: M is not orthogonal.
assert np.allclose(bond_matrix(R[0].T) @ M, np.eye(6))

# -- the push-forward rules ----------------------------------------------------
F = np.eye(3) + 0.1 * rng.normal(size=(3, 3))       # a deformation gradient
J = np.linalg.det(F)
assert J > 0

rho = 5.0e3
assert np.isclose(push_forward(rho, F, J, DENSITY), rho / J)    # weight 1: mass kept
assert np.isclose(push_forward(rho, F, J, SCALAR), rho)         # weight 0: a value
v = rng.normal(size=3)
assert np.allclose(push_forward(v, F, J, VECTOR), F @ v)
C = voigt_to_tensor(V)                                          # (3, 3, 3, 3)
c = push_forward(C, F, J, ELASTIC)
want = np.einsum("iA,jB,kC,lD,ABCD->ijkl", F, F, F, F, C) / J
close = dict(rtol=1e-12, atol=1e-12 * np.abs(C).max())         # values are ~1e11
assert np.allclose(c, want, **close)
assert np.allclose(pull_back(c, F, J, ELASTIC), C, **close)     # round trip
assert np.allclose(pull_back(push_forward(v, F, J, VECTOR), F, J, VECTOR), v)

# A stress-like rank-2 weight-1 tensor: sigma = F S F^T / J.
S = rng.normal(size=(3, 3))
S = 0.5 * (S + S.T)
assert np.allclose(push_forward(S, F, J, Character(2, 1)), F @ S @ F.T / J)

# -- fields cross a mapping the same way ------------------------------------------
prem = PREM(ocean=False)
same = push_forward_field(prem["rho"], IdentityMapping())
assert same is prem["rho"]                          # nothing to do, nothing done
pushed = push_forward_field(prem.elastic_moduli, IdentityMapping())
r, th, ph = np.array([5.0e6]), np.array([1.0]), np.array([0.5])
assert np.allclose(pushed.evaluate(r, th, ph, frame="cartesian"),
                   prem.elastic_moduli.evaluate(r, th, ph, frame="cartesian"))

print("ok: frames rotate slot by slot; push-forward and pull-back invert each other")
