"""A patch's spline form, its continuation, and its preimage under a mapping.

The oracles: the spline against the patch's own map, for a plane and a
structured patch of either rule; the continued spline against the
original inside the square and against scipy's polynomial continuation
of its end pieces outside; the preimage against the mapping itself,
forward(inverse(x)) = x at the control nodes, and its offset falling at
the fourth order of the control spacing between them, the rate of a
bicubic interpolant.
"""
import numpy as np
import pytest
from scipy.interpolate import NdBSpline

from planetmodel import CallableDisplacement, RadialStretch
from planetmodel.fault import (PULL_BACK_CONTROLS, GeographicFrame, GreatCircle,
                               Plane, Slip, SmallCircle, StructuredPatch,
                               TensorSpline, pull_back)
from planetmodel.mapping import IdentityMapping, MappingBase, ScaledMapping

UNIT = GeographicFrame(1.0)
ZERO = Slip(np.zeros((1, 1, 3)))
RNG = np.random.default_rng(21)
U_, V_ = RNG.uniform(-0.5, 0.5, (2, 300))


def bump(amplitude=0.05):
    """A radial relief decaying with depth, a smooth bump over (30N, 40E)."""
    def h(r, theta, phi):
        c = (np.cos(theta) * np.cos(np.deg2rad(60.0)) + np.sin(theta)
             * np.sin(np.deg2rad(60.0)) * np.cos(phi - np.deg2rad(40.0)))
        return amplitude * r ** 2 * np.exp(-(1.0 - c) / 0.05)
    return RadialStretch(CallableDisplacement(h), rmax=1.0)


def plane():
    return Plane.at_depth(UNIT, 30.0, 40.0, 0.2, strike=20.0, dip=50.0, length=0.3,
                          width=0.2, slip=ZERO)


def curved(interpolation="cubic"):
    return StructuredPatch.swept(UNIT, SmallCircle([60.0, 10.0], 30.0, 160.0, 200.0),
                                 dip=lambda z: 15.0 + 100.0 * z, width=0.3, top=0.02,
                                 n_u=7, n_v=6, slip=ZERO, interpolation=interpolation)


# -------------------------------------------------------------- the spline form

@pytest.mark.parametrize("make", [plane, curved, lambda: curved("linear")])
def test_the_spline_is_the_patch(make):
    patch = make()
    spline = patch.spline()
    assert spline.bounds == ((-0.5, 0.5), (-0.5, 0.5))
    # measured: 2.2e-16 on a plane, 0 on a structured patch (its own rule)
    assert np.allclose(spline(U_, V_), patch.cartesian(U_, V_), rtol=0, atol=1e-14)
    assert spline(U_[:6].reshape(2, 3), V_[:6].reshape(2, 3)).shape == (2, 3, 3)


def test_a_plane_s_spline_is_bilinear_through_its_corners():
    p = plane()
    spline = p.spline()
    assert spline.degree == (1, 1) and spline.coefficients.shape == (2, 2, 3)
    assert np.allclose(spline.coefficients[1, 0], p.cartesian(0.5, -0.5), rtol=0,
                       atol=1e-15)


@pytest.mark.parametrize("make", [plane, curved])
def test_the_continued_spline_is_the_same_map_carried_on(make):
    spline = make().spline()
    wide = spline.extended(u=(0.1, 0.05), v=(0.3, 0.0))
    assert wide.bounds == ((-0.6, 0.55), (-0.8, 0.5))
    assert wide.degree == spline.degree
    # inside the old bounds nothing changes: measured 5.6e-16
    assert np.allclose(wide(U_, V_), spline(U_, V_), rtol=0, atol=1e-13)
    # outside, the end pieces' polynomials, as scipy continues them
    u = RNG.uniform(-0.6, 0.55, 50)
    v = RNG.uniform(-0.8, -0.5, 50)
    own = NdBSpline(spline.knots, spline.coefficients, spline.degree,
                    extrapolate=True)(np.stack([u, v], axis=-1))
    assert np.allclose(wide(u, v), own, rtol=0, atol=1e-12)
    # clamped at the new ends: the new edges are curves of the map
    tu, tv = wide.knots
    k = wide.degree[1]
    assert np.all(tv[:k + 1] == -0.8) and np.all(tu[-(k + 1):] == 0.55)


def test_a_plane_continues_as_the_plane():
    p = plane()
    wide = p.spline().extended(v=(0.4, 0.0))
    x0, xu, xv = p.cartesian(0.0, 0.0), *p.tangents(0.0, 0.0)
    want = x0 + 0.2 * xu - 0.85 * xv
    assert np.allclose(wide(0.2, -0.85), want, rtol=0, atol=1e-14)


def test_a_spline_refuses_what_is_not_one():
    spline = plane().spline()
    with pytest.raises(ValueError, match="knots number"):
        TensorSpline((1, 1), (np.array([-0.5, 0.5]), spline.knots[1]),
                     spline.coefficients)
    with pytest.raises(ValueError, match="repeat each end"):
        TensorSpline((1, 1), (np.array([-0.5, 0.1, 0.2, 0.5]), spline.knots[1]),
                     spline.coefficients)
    with pytest.raises(ValueError, match="non-negative"):
        spline.extended(u=(-0.1, 0.0))
    with pytest.raises(ValueError, match="coefficients"):
        TensorSpline((1, 1), spline.knots, np.zeros((2, 2, 2, 3)))


# -------------------------------------------------------------- the preimage

def test_under_the_identity_a_patch_is_its_own_preimage():
    p = plane()
    pre = pull_back(p, IdentityMapping())
    assert pre.patch is p and pre.controls is None and pre.offset == 0.0


@pytest.mark.parametrize("make", [plane, curved])
def test_the_preimage_maps_onto_the_patch_at_its_control_nodes(make):
    """forward(inverse(x)) = x: the oracle of the pull-back, at the nodes."""
    patch, m = make(), bump()
    pre = pull_back(patch, m, controls=9)
    assert isinstance(pre.patch, StructuredPatch)
    assert pre.patch.frame == patch.frame and pre.patch.interpolation == "cubic"
    assert pre.patch.positions.shape == (9, 9, 3) and pre.controls == (9, 9)
    nodes = np.linspace(-0.5, 0.5, 9)
    U, V = np.meshgrid(nodes, nodes, indexing="ij")
    x = patch.cartesian(U, V)
    # measured: 2.2e-16 on coordinates of one
    assert np.allclose(m(pre.patch.cartesian(U, V)), x, rtol=0, atol=1e-12)
    # the preimage lies in the reference body, inside its image here
    assert np.all(np.linalg.norm(pre.patch.cartesian(U, V), axis=-1)
                  <= np.linalg.norm(x, axis=-1) + 1e-15)
    assert np.all(pre.patch.slip.values == 0.0)


@pytest.mark.parametrize("make", [plane, curved])
def test_the_offset_falls_at_the_fourth_order(make):
    patch, m = make(), bump()
    offsets = [pull_back(patch, m, controls=n).offset for n in (5, 9, 17, 33)]
    ratios = np.array(offsets[:-1]) / np.array(offsets[1:])
    # measured: plane 63, 17, 10; curved 16, 16, 17 (the bicubic's 16)
    assert np.all(ratios > 8.0), ratios
    # and it is what it says: the preimage mapped forward, located
    pre = pull_back(patch, m, controls=5)
    x = m(pre.patch.cartesian(U_, V_))
    assert np.max(np.abs(patch.locate(x).distance)) <= pre.offset * (1 + 1e-9)


def test_without_controls_the_fewest_that_meet_rtol_are_taken():
    patch, m = plane(), bump()
    pre = pull_back(patch, m, rtol=1e-6)
    size = np.linalg.norm(patch.cartesian(0.5, 0.5) - patch.cartesian(-0.5, -0.5))
    assert pre.offset <= 1e-6 * size
    n = pre.controls[0]
    assert n in PULL_BACK_CONTROLS and n > PULL_BACK_CONTROLS[0]
    fewer = PULL_BACK_CONTROLS[PULL_BACK_CONTROLS.index(n) - 1]
    assert pull_back(patch, m, controls=fewer).offset > 1e-6 * size
    with pytest.raises(ValueError, match="short of rtol"):
        pull_back(patch, m, rtol=1e-17)


def test_a_scaled_radial_mapping_pulls_back_through_its_own_inverse():
    k = 6.371e6
    m = ScaledMapping(bump(), k)
    frame = GeographicFrame(k)
    patch = Plane.at_depth(frame, 30.0, 40.0, 0.2 * k, strike=20.0, dip=50.0,
                           length=0.3 * k, width=0.2 * k, slip=ZERO)
    pre = pull_back(patch, m, controls=9)
    small = pull_back(plane(), bump(), controls=9)
    assert pre.offset == pytest.approx(k * small.offset, rel=1e-6)


def test_refusals():
    class Squash(MappingBase):
        def __call__(self, X):
            return np.asarray(X, dtype=float) * 1.01

        def deformation_gradient(self, X):
            X = np.asarray(X, dtype=float)
            return np.broadcast_to(1.01 * np.eye(3), X.shape[:-1] + (3, 3)).copy()

        def jacobian(self, X):
            return np.full(np.shape(X)[:-1], 1.01 ** 3)

    with pytest.raises(ValueError, match="does not invert itself"):
        pull_back(plane(), Squash())
    with pytest.raises(ValueError, match="controls must be an integer of at least 4"):
        pull_back(plane(), bump(), controls=3)
    with pytest.raises(ValueError, match="count or"):
        pull_back(plane(), bump(), controls=(5, 5, 5))
    with pytest.raises(TypeError, match="Patch"):
        pull_back("plane", bump())
    with pytest.raises(ValueError, match="rtol"):
        pull_back(plane(), bump(), rtol=0.0)


def test_a_rectangular_control_grid():
    patch = StructuredPatch.swept(UNIT, GreatCircle([25.0, 30.0], [35.0, 50.0]),
                                  dip=40.0, width=0.2, top=0.05, n_u=7, n_v=4,
                                  slip=ZERO)
    pre = pull_back(patch, bump(), controls=(9, 5))
    assert pre.controls == (9, 5) and pre.patch.positions.shape == (9, 5, 3)
