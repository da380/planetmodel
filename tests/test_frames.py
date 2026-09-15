"""The local spherical frame and rotations of tensor slots."""
import numpy as np
import pytest

from planetmodel.frames import (MAX_RANK, cartesian_points, rotate_slots,
                                rotation_subscripts, spherical_coordinates,
                                spherical_frame)


def test_frame_is_orthonormal_and_right_handed():
    theta = np.linspace(0.0, np.pi, 7)[:, None]
    phi = np.linspace(-np.pi, np.pi, 6, endpoint=False)[None, :]
    R = spherical_frame(theta, phi)
    assert R.shape == (7, 6, 3, 3)
    eye = np.einsum("...ki,...kj->...ij", R, R)
    assert np.allclose(eye, np.eye(3))
    assert np.allclose(np.linalg.det(R), 1.0)
    e_r = R[..., :, 0]
    assert np.allclose(e_r, cartesian_points(1.0, theta, phi))


def test_coordinates_round_trip():
    r = np.array([0.5, 1.0, 2.0])[:, None]
    theta = np.array([0.3, 1.2])[None, :]
    phi = 0.7
    X = cartesian_points(r, theta, phi)
    rr, tt, pp, R = spherical_coordinates(X)
    assert np.allclose(rr, np.broadcast_to(r, rr.shape))
    assert np.allclose(tt, np.broadcast_to(theta, tt.shape))
    assert np.allclose(pp, phi)
    assert np.allclose(R, spherical_frame(theta, phi))
    r0, t0, p0, _ = spherical_coordinates(np.zeros(3))
    assert r0 == 0.0 and t0 == pytest.approx(np.pi / 2) and p0 == 0.0
    with pytest.raises(ValueError, match=r"\(\.\.\., 3\)"):
        spherical_coordinates(np.zeros((4, 2)))


def test_rotate_slots_on_vectors_and_tensors():
    R = spherical_frame(0.4, 1.1)
    v = np.array([1.0, 2.0, 3.0])
    assert np.allclose(rotate_slots(v, R, 1), R @ v)
    T = np.arange(9.0).reshape(3, 3)
    assert np.allclose(rotate_slots(T, R, 2), R @ T @ R.T)
    back = rotate_slots(rotate_slots(T, R, 2), R.T, 2)
    assert np.allclose(back, T)
    assert rotate_slots(5.0, R, 0) == 5.0
    C = np.arange(81.0).reshape(3, 3, 3, 3)
    want = np.einsum("iA,jB,kC,lD,ABCD->ijkl", R, R, R, R, C)
    assert np.allclose(rotate_slots(C, R, 4), want)
    assert np.iscomplexobj(rotate_slots(v + 1j, R, 1))
    with pytest.raises(ValueError, match=r"\(\.\.\., 3, 3\)"):
        rotate_slots(v, np.eye(2), 1)


def test_rotation_subscripts():
    assert rotation_subscripts(1) == "...iA,...A->...i"
    assert rotation_subscripts(2) == "...iA,...jB,...AB->...ij"
    with pytest.raises(ValueError, match="rank"):
        rotation_subscripts(MAX_RANK + 1)
