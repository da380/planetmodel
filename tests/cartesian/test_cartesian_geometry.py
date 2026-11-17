"""The flat geometry: numbering from the surface down with the base no
interface, the lateral domains, mappings checked over the domain, and
the contract held as on the sphere."""
import numpy as np
import pytest

from planetmodel import Skeleton
from planetmodel import cartesian as flat
from planetmodel.geometry import Geometry as SphericalGeometry
from planetmodel.geometry import GeometryBase
from planetmodel.mapping import MappingBase, ScaledMapping, validity
from planetmodel.testing import check_geometry, check_mapping

FLAT = flat.FLAT
DEPTHS = [0.0, 10.0, 30.0, 50.0]
BOX = flat.Box([(0.0, 100.0), (-20.0, 40.0)])


class Linear(MappingBase):
    """m(X) = A X + t: not the identity."""

    def __init__(self, A, t=(0.0, 0.0, 0.0)):
        self.A, self.t = np.asarray(A, dtype=float), np.asarray(t, dtype=float)

    def __call__(self, X):
        return np.asarray(X, dtype=float) @ self.A.T + self.t

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(self.A, X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        X = np.asarray(X, dtype=float)
        return np.full(X.shape[:-1], np.linalg.det(self.A))


SHEAR = Linear([[1.0, 0.1, 0.05], [0.0, 1.1, -0.1], [0.02, 0.0, 0.9]], (1.0, 2.0, 0.5))


class Folded(MappingBase):
    """Reflects z below a depth: J < 0 there, and torn across that depth."""

    def __init__(self, depth):
        self.depth = depth

    def _below(self, X):
        return np.asarray(X, dtype=float)[..., 2] < -self.depth

    def __call__(self, X):
        X = np.array(X, dtype=float)
        below = self._below(X)
        X[..., 2] = np.where(below, -X[..., 2], X[..., 2])
        return X

    def deformation_gradient(self, X):
        F = np.broadcast_to(np.eye(3), np.shape(X)[:-1] + (3, 3)).copy()
        F[..., 2, 2] = np.where(self._below(X), -1.0, 1.0)
        return F

    def jacobian(self, X):
        return np.where(self._below(X), -1.0, 1.0)


def box_geometry(**kw):
    return flat.Geometry(Skeleton(DEPTHS), BOX, layer_names=["crust", "mantle", "deep"],
                         interface_names=["surface", "moho", "d2"], **kw)


# ------------------------------------------------------------ numbering

def test_interfaces_are_numbered_from_the_surface_down_and_the_base_is_none():
    g = box_geometry()
    assert isinstance(g, GeometryBase) and not isinstance(g, SphericalGeometry)
    assert g.coordinates is FLAT and g.domain == BOX and g.is_identity
    faces = g.interfaces
    assert [f.radius for f in faces] == [0.0, 10.0, 30.0]      # 50, the base, is none
    assert [f.between for f in faces] == [(0, -1), (1, 0), (2, 1)]
    assert [f.name for f in faces] == ["surface", "moho", "d2"]
    assert g.interface("surface").index == 0 and g.interface(-1).name == "d2"
    assert [lay.interval for lay in g.layers] == [(0.0, 10.0), (10.0, 30.0),
                                                  (30.0, 50.0)]
    with pytest.raises(ValueError, match="got 4 interface names for 3 interfaces"):
        flat.Geometry(Skeleton(DEPTHS), BOX, interface_names=["a", "b", "c", "base"])


def test_surgery_keeps_the_numbering_and_the_base_takes_no_name():
    g = box_geometry()
    top = g.hollowed(10.0)
    assert top.is_hollow and [f.name for f in top.interfaces] == ["moho", "d2"]
    assert top.interfaces[0].between == (0, -1)
    assert g.hollowed(20.0, name="cut").interfaces[0].name == "cut"
    cut = g.truncated(30.0)
    assert [f.radius for f in cut.interfaces] == [0.0, 10.0]
    assert cut.skeleton.boundaries[-1] == 30.0
    with pytest.raises(ValueError, match="cut at depth 40 is no interface"):
        g.truncated(40.0, name="base")
    longer = g.extended([80.0, 120.0], names=["pad1", "pad2"],
                        interface_names=["old base", "between pads"])
    assert [f.radius for f in longer.interfaces] == [0.0, 10.0, 30.0, 50.0, 80.0]
    assert longer.interfaces[3].between == (3, 2)
    assert longer.interface("old base").radius == 50.0
    finer = g.refined([20.0], names=["mid"])
    assert [f.name for f in finer.interfaces] == ["surface", "moho", "mid", "d2"]
    assert finer.layer(1).name is None and finer.layer(0).name == "crust"
    coarse, _ = g.coarsened(drop=[1])
    assert [f.name for f in coarse.interfaces] == ["surface", "moho"]
    assert coarse.layers[1].interval == (10.0, 50.0)


def test_the_contract_holds_flat_geometries_with_any_mapping():
    for g in (box_geometry(), box_geometry(mapping=SHEAR),
              box_geometry().hollowed(10.0),
              flat.Geometry(Skeleton(DEPTHS), flat.Rectangle((0.0, 60.0), padding=5.0)),
              flat.Geometry(Skeleton([0.0, 2.0]), flat.Box([(0, 1), (0, 1)],
                                                           padding=0.25))):
        check_geometry(g)


def test_scaling_scales_the_domain_and_the_mapping():
    g = box_geometry(mapping=SHEAR)
    big = g.scaled(1000.0)
    assert big.domain == flat.Box([(0.0, 1e5), (-2e4, 4e4)])
    assert np.array_equal(big.skeleton.boundaries, 1000.0 * np.array(DEPTHS))
    assert isinstance(big.mapping, ScaledMapping)
    X = FLAT.points(5.0, 3.0, 4.0)
    assert np.allclose(big.mapping(1000.0 * X), 1000.0 * SHEAR(X), rtol=1e-14)
    assert repr(g).startswith("cartesian.Geometry(")


# ------------------------------------------------------------ mappings

def test_a_folding_mapping_is_refused_naming_depth_and_x_and_y():
    with pytest.raises(ValueError, match=r"J = -1 is not positive at \(depth, x, y\)"):
        box_geometry(mapping=Folded(40.0))
    report = box_geometry(mapping=Folded(40.0), check=False).validity()
    assert not report and report.worst_point[0] > 40.0
    assert report.labels == ("depth", "x", "y")


def test_a_torn_mapping_is_refused_at_its_depth():
    torn = Linear(np.eye(3))
    torn_below = type("Torn", (Linear,), {
        "__call__": lambda self, X: np.asarray(X, dtype=float) + np.where(
            np.asarray(X)[..., 2:3] < -10.0, 1.0, 0.0)})
    with pytest.raises(ValueError, match=r"discontinuous across the boundary at depth "
                                         r"10: .* at \(x, y\) = "):
        box_geometry(mapping=torn_below(np.eye(3)))
    assert box_geometry(mapping=torn).is_identity is False


def test_validity_and_the_scaled_mapping_go_through_the_flat_system():
    g = box_geometry(mapping=Folded(40.0), check=False)
    d, x, y = g.lattice(n_c=2, n_a=3, n_b=2)
    assert x.min() == 0.0 and x.max() == 100.0 and y.min() == -20.0 and y.max() == 40.0
    assert d.max() < 50.0 and d.min() > 0.0
    report = validity(g.mapping, sample=(d, x, y), coordinates=FLAT)
    assert not report and report.worst_point[0] > 40.0
    # the scaled mapping reads the sample of a body k times the size at X / k
    scaled = ScaledMapping(Folded(40.0), 2.0)
    assert scaled.is_valid(sample=(np.array([70.0]), 0.0, 0.0), coordinates=FLAT)
    bad = scaled.is_valid(sample=(np.array([90.0]), 3.0, 4.0), coordinates=FLAT)
    assert not bad and np.allclose(bad.worst_point, (45.0, 1.5, 2.0))
    check_mapping(SHEAR, FLAT.points(*g.lattice(n_c=2, n_a=3, n_b=2)).reshape(-1, 3))


# ------------------------------------------------------------ domains

def test_a_box_has_bounds_padding_and_a_period():
    b = flat.Box([(0, 100), (-20, 40)], padding=(5.0, 2.5))
    assert b.bounds == ((0.0, 100.0), (-20.0, 40.0)) and b.padding == (5.0, 2.5)
    assert b.extent == ((-5.0, 105.0), (-22.5, 42.5)) and b.lengths == (110.0, 65.0)
    assert flat.Box([(0, 1), (0, 2)], padding=0.5).padding == (0.5, 0.5)
    assert b == flat.Box([(0.0, 100.0), (-20.0, 40.0)], padding=[5.0, 2.5])
    assert b != BOX and hash(b) == hash(flat.Box([(0, 100), (-20, 40)],
                                                 padding=(5, 2.5)))
    inside = b.contains([50.0, 101.0, 50.0, -1.0], [0.0, 0.0, 41.0, -20.0])
    assert inside.tolist() == [True, False, False, False]
    _, x, y = b.lattice(Skeleton([0.0, 1.0]), n_a=3, n_b=3)
    assert x.ravel().tolist() == [-5.0, 50.0, 105.0]           # the padding is covered
    assert y.ravel().tolist() == [-22.5, 10.0, 42.5]


def test_a_rectangle_bounds_x_alone():
    r = flat.Rectangle((10, 70), padding=5.0)
    assert r.bounds == (10.0, 70.0) and r.extent == (5.0, 75.0) and r.length == 70.0
    assert r.axes == ("x",) and flat.Box([(0, 1), (0, 1)]).axes == ("x", "y")
    assert r.contains([20.0, 71.0], [1e9, 0.0]).tolist() == [True, False]
    assert r == flat.Rectangle((10.0, 70.0), padding=5) and r.scaled(2.0) == (
        flat.Rectangle((20.0, 140.0), padding=10.0))
    _, x, y = r.lattice(Skeleton([0.0, 1.0]))
    assert x.min() == 5.0 and x.max() == 75.0 and np.isclose(y.min(), -35.0)


def test_domains_refuse_what_is_no_domain():
    with pytest.raises(ValueError, match="bounds of y must be finite and increase"):
        flat.Box([(0, 1), (2, 2)])
    with pytest.raises(ValueError, match="bounds of x and of y"):
        flat.Box([(0, 1)])
    with pytest.raises(ValueError, match="padding of y must be finite and non-neg"):
        flat.Box([(0, 1), (0, 1)], padding=(0.0, -1.0))
    with pytest.raises(ValueError, match="bounds of x are a"):
        flat.Rectangle(5.0)
    with pytest.raises(ValueError, match="scale factor must be positive"):
        BOX.scaled(0.0)
    with pytest.raises(TypeError, match="expected a Box or a Rectangle"):
        flat.Geometry(Skeleton(DEPTHS), ((0, 1), (0, 1)))


def test_a_rectangle_holds_its_mapping_to_invariance_along_y():
    sk = Skeleton(DEPTHS)
    rect = flat.Rectangle((0.0, 60.0))
    # shears in the (x, z) plane, keeping y: a rectangle's mapping
    plane = Linear([[1.0, 0.0, 0.1], [0.0, 1.0, 0.0], [0.05, 0.0, 1.1]],
                   (2.0, 0.0, 1.0))
    check_geometry(flat.Geometry(sk, rect, mapping=plane))
    with pytest.raises(ValueError, match="must not vary along y"):
        flat.Geometry(sk, rect, mapping=Linear([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                                [0.0, 0.1, 1.0]]))
    # the box takes the same mapping
    flat.Geometry(sk, BOX, mapping=Linear([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                                           [0.0, 0.1, 1.0]]))
