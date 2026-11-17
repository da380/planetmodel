"""Topography as a mapping: the vertical displacement, its stack of reliefs
and the vertical stretch, held to the contracts the spherical mappings
are held to, their closed forms against finite differences, the sign
(heights, positive up), folds refused by name, rectangles held to
invariance along y, fields carried across by the shared machinery, and
a fault pulled back through the stretch."""
import numpy as np
import pytest

from planetmodel import DENSITY, VECTOR, Skeleton, cartesian, testing
from planetmodel.cartesian import FLAT
from planetmodel.cartesian.mapping import VerticalStretch
from planetmodel.character import Character
from planetmodel.fault import PULL_BACK_CONTROLS, CartesianFrame, Plane, Slip, pull_back
from planetmodel.mapping import ScaledMapping
from planetmodel.pushforward import PulledBackField, PushedForwardField

DEPTHS = [0.0, 0.4, 1.0]
BOX = cartesian.Box([(0.0, 1.0), (-0.5, 0.5)])
BOUNDS = ((0.0, 1.0), (-0.5, 0.5))
LOCAL = CartesianFrame()
ZERO = Slip(np.zeros((1, 1, 3)))


def ridge(x, y):
    """A range along y over x = 0.6, a little higher towards y = 0, with
    a shallow basin in front of it: a height, positive up."""
    return (0.08 * np.exp(-((x - 0.6) / 0.15) ** 2) * (1.0 + 0.3 * np.cos(np.pi * y))
            - 0.02 * np.exp(-((x - 0.3) / 0.1) ** 2))


def ridge_gradient(x, y):
    g = np.exp(-((x - 0.6) / 0.15) ** 2)
    b = np.exp(-((x - 0.3) / 0.1) ** 2)
    dx = (0.08 * g * (-2.0 * (x - 0.6) / 0.15 ** 2) * (1.0 + 0.3 * np.cos(np.pi * y))
          - 0.02 * b * (-2.0 * (x - 0.3) / 0.1 ** 2))
    dy = -0.08 * g * 0.3 * np.pi * np.sin(np.pi * y)
    return dx, dy


ridge.lateral_gradient = ridge_gradient


def root(x, y):
    """The Moho's relief: a root under the range, deeper there."""
    return -0.5 * 0.08 * np.exp(-((x - 0.6) / 0.2) ** 2)


def box(**kw):
    return cartesian.Geometry(Skeleton(DEPTHS), BOX, layer_names=["crust", "mantle"],
                              interface_names=["surface", "moho"], **kw)


def stack(surface=ridge, moho=root, base=None):
    return cartesian.layer_linear(Skeleton(DEPTHS), [surface, moho, base])


def lattice_points(geometry):
    return FLAT.points(*geometry.lattice(n_c=3, n_a=6, n_b=5)).reshape(-1, 3)


# ------------------------------------------------------------ the displacements

@pytest.mark.parametrize("w", [
    stack(),
    stack(surface=lambda x, y: ridge(x, y), base=lambda x, y: 0.01 * x * y),
    cartesian.CallableDisplacement(lambda d, x, y: 0.05 * np.sin(3 * x) * (1 - d) ** 2),
    cartesian.CallableDisplacement(
        lambda d, x, y: 0.05 * np.sin(3 * x) * (1 - d) ** 2,
        depth_derivative=lambda d, x, y: -0.1 * np.sin(3 * x) * (1 - d),
        lateral_gradient=lambda d, x, y: (0.15 * np.cos(3 * x) * (1 - d) ** 2, 0 * y)),
    cartesian.ZeroDisplacement(),
])
def test_every_displacement_holds_to_its_contract(w):
    testing.check_vertical_displacement(w, Skeleton(DEPTHS), bounds=BOUNDS)


def test_the_sign_is_heights_positive_up():
    """Every boundary takes its relief as a height: the surface rises to
    the ridge, the Moho sinks into its root, the base is held; within a
    layer the displacement is linear in depth."""
    w = stack()
    x, y = np.linspace(0.0, 1.0, 11), 0.2
    assert np.allclose(w(0.0, x, y), ridge(x, y), rtol=0.0, atol=1e-15)
    assert np.allclose(w(0.4, x, y), root(x, y), rtol=0.0, atol=1e-15)
    assert np.all(w(1.0, x, y) == 0.0)
    assert np.allclose(w(0.1, x, y), 0.75 * ridge(x, y) + 0.25 * root(x, y))
    m = VerticalStretch(w, base=1.0)
    top = m(FLAT.points(0.0, x, y))
    assert np.allclose(top[:, 2], ridge(x, y), rtol=0.0, atol=1e-15)
    assert top[np.argmin(np.abs(x - 0.6)), 2] > 0.09        # the range stands up
    moho = m(FLAT.points(0.4, x, y))
    assert np.all(-moho[:, 2] >= 0.4) and -moho[6, 2] > 0.43  # the root is deeper
    assert np.all(m(FLAT.points(1.0, x, y))[:, 2] == -1.0)
    assert w.knots == (0.4,) and m.knots == (0.4,)
    assert w.reliefs[2] is None and np.array_equal(w.boundaries, DEPTHS)


def test_layer_linear_refuses_a_wrong_count_or_a_non_callable():
    with pytest.raises(ValueError, match="3 boundaries need 3 reliefs, the free "
                                         "surface first and the base last"):
        cartesian.layer_linear(Skeleton(DEPTHS), [ridge])
    with pytest.raises(TypeError, match="callable of \\(x, y\\) or None"):
        cartesian.layer_linear(Skeleton(DEPTHS), [ridge, 1.0, None])
    with pytest.raises(ValueError, match="already declares"):
        cartesian.as_displacement(stack(), knots=[0.5])
    assert cartesian.as_displacement(stack()).knots == (0.4,)


# ------------------------------------------------------------ the stretch

@pytest.mark.parametrize("w", [
    stack(), stack(surface=lambda x, y: ridge(x, y)),
    cartesian.CallableDisplacement(lambda d, x, y: 0.05 * np.sin(3 * x + y) * (1 - d))])
def test_the_vertical_stretch_holds_to_the_mapping_contract(w):
    """F and J against central differences of m, the inverse round trip,
    m(X) - X, and the scaled stretch, on points inside every layer."""
    m = VerticalStretch(w, base=1.0)
    g = box(mapping=m)
    _, x, y = g.lattice(n_c=3, n_a=6, n_b=5)
    d = np.array([0.05, 0.2, 0.35, 0.45, 0.7, 0.95])[:, None, None]
    testing.check_mapping(m, FLAT.points(d, x, y).reshape(-1, 3))
    assert m.is_valid(sample=g.lattice(), coordinates=FLAT)
    assert isinstance(m.scaled(2.0), VerticalStretch) and m.scaled(2.0).base == 2.0


def test_the_closed_forms_of_the_stretch():
    """F = I + e_z grad w with d/dz = -d/d depth, J = 1 - dw/d depth, the
    local F the same with the vertical row first; x and y kept."""
    w = stack()
    m = VerticalStretch(w, base=1.0)
    X = lattice_points(box())
    d, x, y, _ = FLAT.split(X)
    F = m.deformation_gradient(X)
    gx, gy = w.lateral_gradient(d, x, y)
    dd = w.depth_derivative(d, x, y)
    assert np.allclose(F[:, 2, 0], gx) and np.allclose(F[:, 2, 1], gy)
    assert np.allclose(F[:, 2, 2], 1.0 - dd)
    assert np.array_equal(F[:, :2], np.broadcast_to(np.eye(3)[:2], F[:, :2].shape))
    assert np.allclose(m.jacobian(X), np.linalg.det(F), rtol=1e-14)
    R = cartesian.coordinates.LOCAL_FRAME
    assert np.allclose(m.deformation_gradient_local(X), R.T @ F @ R)
    x_ = m(X)
    assert np.array_equal(x_[:, :2], X[:, :2])
    assert np.allclose(m.displacement(X), x_ - X, rtol=0.0, atol=1e-16)
    assert not m.is_identity
    assert VerticalStretch(cartesian.ZeroDisplacement(), base=1.0).is_identity
    assert repr(m) == "VerticalStretch(cartesian.LayerLinear(2 reliefs on 3 " \
                      "boundaries), base=1)"


def test_the_inverse_is_a_column_root_find_to_rounding():
    """Points of the physical body, above the reference surface under the
    ridge, below the base, and on the piecewise-linear stack's knots,
    come back to their reference points; Newton cycling between two
    layers gives way to the bisection."""
    m = VerticalStretch(stack(), base=1.0)
    rng = np.random.default_rng(1)
    X = FLAT.points(rng.uniform(-0.2, 1.3, 400), rng.uniform(0.0, 1.0, 400),
                    rng.uniform(-0.5, 0.5, 400))
    X = np.concatenate([X, FLAT.points(np.array([0.0, 0.4, 1.0]), 0.6, 0.0)])
    back = m.inverse(m(X))
    assert np.abs(back - X).max() < 1e-12
    assert back.shape == X.shape and np.array_equal(back[:, :2], X[:, :2])
    with pytest.raises(TypeError):
        VerticalStretch(stack())
    with pytest.raises(ValueError, match="depth of the base must be positive"):
        VerticalStretch(stack(), base=0.0)
    assert VerticalStretch(stack(), base=box()).base == 1.0
    assert VerticalStretch(stack(), base=Skeleton([0.0, 3.0])).base == 3.0


def test_linearise_against_differences_of_the_stretch():
    """dF and dJ of a perturbation of w equal the central difference of F
    and J between the stretches of w plus and minus a small multiple of
    it."""
    w = stack()
    delta = cartesian.CallableDisplacement(
        lambda d, x, y: 0.03 * np.cos(2 * x) * np.sin(3 * y) * (1 - d) ** 2)
    m = VerticalStretch(w, base=1.0)
    X = lattice_points(box())
    eps = 1e-6

    def moved(s):
        return VerticalStretch(cartesian.CallableDisplacement(
            lambda d, x, y: w(d, x, y) + s * delta(d, x, y),
            depth_derivative=lambda d, x, y: (w.depth_derivative(d, x, y)
                                              + s * delta.depth_derivative(d, x, y)),
            lateral_gradient=lambda d, x, y: tuple(
                a + s * b for a, b in zip(w.lateral_gradient(d, x, y),
                                          delta.lateral_gradient(d, x, y)))),
            base=1.0)

    lin = m.linearise(delta, X=X)
    up, down = moved(eps), moved(-eps)
    dF = (up.deformation_gradient(X) - down.deformation_gradient(X)) / (2 * eps)
    dJ = (up.jacobian(X) - down.jacobian(X)) / (2 * eps)
    assert np.allclose(lin.dF, dF, rtol=0.0, atol=1e-7)
    assert np.allclose(lin.dJ, dJ, rtol=0.0, atol=1e-7)


# ------------------------------------------------------------ the geometry

def test_a_stretched_geometry_holds_to_the_geometry_contract():
    """A box, a hollowed box and a rectangle under reliefs; the knots are the
    interior boundaries; scaling keeps the stretch a stretch."""
    g = box().stretched(stack(), name="topography")
    assert isinstance(g.mapping, VerticalStretch) and g.mapping.base == 1.0
    assert g.knots() == (0.4,) and g.mapping.name == "topography"
    testing.check_geometry(g)
    testing.check_geometry(g.hollowed(0.1))
    testing.check_geometry(g.refined([0.7]))
    big = g.scaled(1e4)
    assert isinstance(big.mapping, VerticalStretch) and big.mapping.base == 1e4
    assert not isinstance(big.mapping, ScaledMapping)
    X = lattice_points(box())
    assert np.allclose(big.mapping(1e4 * X), 1e4 * g.mapping(X), rtol=0.0, atol=1e-10)
    rectangle = cartesian.Geometry(Skeleton(DEPTHS), cartesian.Rectangle((0.0, 1.0)))
    testing.check_geometry(rectangle.stretched(stack(surface=lambda x, y: ridge(x, 0.0),
                                                     moho=lambda x, y: root(x, 0.0))))
    assert repr(g).startswith("cartesian.Geometry(Skeleton(2 layers, depth in [0, 1])")


def test_a_rectangle_takes_a_stretch_the_same_at_every_y_and_never_moves_y():
    """Over a Rectangle the stretch moves nothing along y, and a relief that
    varies along y is refused by name."""
    rectangle = cartesian.Geometry(Skeleton(DEPTHS), cartesian.Rectangle((0.0, 1.0)))
    flat_y = rectangle.stretched(stack(surface=lambda x, y: ridge(x, 0.0) + 0.0 * y,
                                       moho=None))
    X = FLAT.points(*flat_y.lattice(n_c=3, n_a=7, n_b=4)).reshape(-1, 3)
    x = flat_y.mapping(X)
    assert np.array_equal(x[:, 1], X[:, 1])
    shifted = X + [0.0, 0.37, 0.0]
    assert np.array_equal(flat_y.mapping(shifted)[:, 2], x[:, 2])
    with pytest.raises(ValueError,
                       match="mapping over a rectangle must not vary along y"):
        rectangle.stretched(stack())


def test_a_folding_stack_is_refused_by_name():
    """A Moho raised above the surface over part of the box turns the crust
    inside out: refused at the geometry, naming the condition and where."""
    over = stack(surface=None, moho=lambda x, y: 0.5 * np.exp(-((x - 0.5) / 0.1) ** 2))
    with pytest.raises(ValueError, match=r"does not preserve orientation: "
                                         r"ValidityReport\(INVALID: 1 - dw/d depth = "
                                         r".* is not positive, so the mapping folds "
                                         r"vertically.*at \(depth, x, y\)"):
        box().stretched(over)
    report = box().stretched(over, check=False).validity()
    assert not report and report.labels == ("depth", "x", "y")
    assert 0.0 < report.worst_point[0] < 0.4 and abs(report.worst_point[1] - 0.5) < 0.07
    m = VerticalStretch(over, base=1.0)
    assert not m.is_valid(X=FLAT.points(0.2, 0.5, 0.0)[None])
    assert m.is_valid(X=FLAT.points(0.6, 0.5, 0.0)[None])      # the mantle is fine
    # a sample without coordinates is spherical, as the protocol has it
    assert m.is_valid(sample=(np.array([0.9]), np.array([0.0]), np.array([0.0])))
    with pytest.raises(ValueError, match="give either points X"):
        m.is_valid()


def test_a_stretched_model_holds_to_the_model_contract():
    """The flat catalogue's model under a range of 2 km and a Moho root of
    5 km, the base held: the model contract, surgery included."""
    model = cartesian.CrustOverMantle(cartesian.Box([(0.0, 400e3), (0.0, 200e3)]))
    sk = model.skeleton
    reliefs = [lambda x, y: 2e3 * np.exp(-((x - 200e3) / 50e3) ** 2),
               lambda x, y: -5e3 * np.exp(-((x - 200e3) / 60e3) ** 2)]
    reliefs += [None] * (sk.boundaries.size - 2)
    stretched = model.stretched(cartesian.layer_linear(sk, reliefs))
    assert type(stretched) is type(model) and isinstance(stretched.geometry.mapping,
                                                         VerticalStretch)
    testing.check_model(stretched)
    top = stretched.geometry.mapping(FLAT.points(0.0, 200e3, 50e3))
    assert top[2] == pytest.approx(2e3, rel=1e-15)


# ------------------------------------------------------------ fields carried across

def _difference_of_m(m, X, eps=1e-7):
    """F at X by central differences of m, (n, 3, 3)."""
    F = np.empty(X.shape + (3,))
    for j in range(3):
        e = np.zeros(3)
        e[j] = eps
        F[..., :, j] = (m(X + e) - m(X - e)) / (2 * eps)
    return F


def test_fields_are_pushed_forward_by_the_shared_machinery():
    """Under the stretch a referential vector v becomes F v, the difference
    of m along it; a density rho becomes rho / J, J the determinant of
    the differenced F; a rank-2 tensor S becomes F S F^T / J: all three
    against finite differences of m, in the local frame at m(X)."""
    m = VerticalStretch(stack(), base=1.0)
    iv = (0.0, 0.4)
    v = cartesian.AnalyticField(iv, lambda d, x, y: np.stack(
        np.broadcast_arrays(1.0 + d, np.sin(x), 0.5 * y), axis=-1), character=VECTOR)
    rho = cartesian.AnalyticField(iv, lambda d, x, y: 2.0 + d * x, character=DENSITY)
    stress = Character(2, 1)
    S = cartesian.AnalyticField(iv, lambda d, x, y: np.stack(np.broadcast_arrays(
        1.0 + d, 2.0, 3.0 + x, 0.1 * y, 0.2, 0.3 * d), axis=-1), character=stress)
    rng = np.random.default_rng(3)
    d, x, y = (rng.uniform(0.02, 0.38, 20), rng.uniform(0.0, 1.0, 20),
               rng.uniform(-0.5, 0.5, 20))
    X = FLAT.points(d, x, y)
    F = _difference_of_m(m, X)
    J = np.linalg.det(F)
    R = cartesian.coordinates.LOCAL_FRAME

    vc = v.evaluate(d, x, y, frame="cartesian")
    eps = 1e-7
    along = (m(X + eps * vc) - m(X - eps * vc)) / (2 * eps)
    got = PushedForwardField(v, m).evaluate(d, x, y)          # local frame at m(X)
    assert np.allclose(got @ R.T, along, rtol=0.0, atol=1e-7)
    assert np.allclose(PushedForwardField(rho, m).evaluate(d, x, y),
                       rho.evaluate(d, x, y) / J, rtol=1e-7)
    from planetmodel.frames import voigt_to_tensor
    Sc = voigt_to_tensor(S.evaluate(d, x, y, frame="cartesian"), rank=2)
    want = np.einsum("pik,pkl,pjl->pij", F, Sc, F) / J[:, None, None]
    got = voigt_to_tensor(PushedForwardField(S, m).evaluate(d, x, y, frame="cartesian"),
                          rank=2)
    assert np.allclose(got, want, rtol=0.0, atol=1e-6)


def test_a_physical_field_pulled_back_pushes_forward_onto_itself():
    """A vector known on the physical body, pulled back through the
    stretch and carried forward again by differences of m, is the
    physical vector at m(X); a physical density pulled back is it times
    the differenced J."""
    m = VerticalStretch(stack(), base=1.0)
    iv = (0.4, 1.0)

    def physical(d, x, y):     # local components (up, x, y) at the physical point
        return np.stack(np.broadcast_arrays(-1.0 - d, x * y, 0.3), axis=-1)

    back = PulledBackField(iv, physical, m, character=VECTOR, coordinates=FLAT)
    rng = np.random.default_rng(4)
    d, x, y = (rng.uniform(0.42, 0.98, 20), rng.uniform(0.0, 1.0, 20),
               rng.uniform(-0.5, 0.5, 20))
    X = FLAT.points(d, x, y)
    ref = back.evaluate(d, x, y, frame="cartesian")
    eps = 1e-7
    forward = (m(X + eps * ref) - m(X - eps * ref)) / (2 * eps)
    pd, px, py, _ = FLAT.split(m(X))
    R = cartesian.coordinates.LOCAL_FRAME
    assert np.allclose(forward, physical(pd, px, py) @ R.T, rtol=0.0, atol=1e-7)
    density = PulledBackField(iv, lambda d, x, y: 3.0 + 0.0 * d, m, character=DENSITY,
                              coordinates=FLAT)
    J = np.linalg.det(_difference_of_m(m, X))
    assert np.allclose(density.evaluate(d, x, y), 3.0 * J, rtol=1e-7)


# ------------------------------------------------------------ the fault's pull-back

def test_a_fault_is_pulled_back_through_the_stretch():
    """`fault.pull_back` asks the stretch for its inverse alone: the
    preimage keeps every control node's x and y (a column), mapped
    forward it lies on the plane to its measured offset, which falls as
    the fourth power of the control spacing, and the default grid meets
    rtol."""
    m = VerticalStretch(stack(), base=1.0)
    p = Plane.at_depth(LOCAL, 0.55, 0.0, 0.25, strike=0.0, dip=40.0, length=0.6,
                       width=0.4, slip=ZERO)
    offsets = []
    for n in (5, 9, 17):
        pre = pull_back(p, m, controls=n)
        assert pre.controls == (n, n)
        U, V = np.meshgrid(np.linspace(-0.5, 0.5, n), np.linspace(-0.5, 0.5, n),
                           indexing="ij")
        nodes = pre.patch.cartesian(U, V)
        assert np.allclose(nodes[..., :2], p.cartesian(U, V)[..., :2], rtol=0.0,
                           atol=1e-15)
        assert np.abs(m(nodes) - p.cartesian(U, V)).max() < 1e-13
        uu, vv = np.meshgrid(np.linspace(-0.5, 0.5, 41), np.linspace(-0.5, 0.5, 41))
        off = np.abs(p.locate(m(pre.patch.cartesian(uu, vv)).reshape(-1, 3),
                              strict=False).distance).max()
        # the offset is the largest distance found on its own samples, the
        # scale of the error rather than a bound
        assert pre.offset <= off < 1.5 * pre.offset
        offsets.append(pre.offset)
    # measured: 1.3e-4, 2.1e-5, 2.0e-6 (and 1.4e-7 at 33): the ratio rising
    # towards 2^4 per halving of the spacing, a cubic's
    assert offsets[0] / offsets[1] > 5 and offsets[1] / offsets[2] > 8
    pre = pull_back(p, m)
    assert pre.controls == (33, 33) and pre.controls[0] in PULL_BACK_CONTROLS
    assert pre.offset <= 1e-6 * np.hypot(0.6, 0.4)
    assert pull_back(p, box().mapping).patch is p
