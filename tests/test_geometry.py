"""Geometry: invariants, names, numbering, surgery, scaling."""
import numpy as np
import pytest

from planetmodel import (CallableDisplacement, Geometry, IdentityMapping,
                         InterfaceInfo, LayerInfo, MappingBase, RadialStretch,
                         Skeleton, testing)

SK = Skeleton([0.0, 0.4, 0.8, 1.0])
NAMES = dict(layer_names=["core", "mantle", "crust"],
             interface_names=["cmb", "moho", "surface"])


def stretch(amp=0.05, *, knot=None):
    def h(r, theta, phi):
        base = -amp * r * 0.5 * (3.0 * np.cos(theta) ** 2 - 1.0)
        if knot is None:
            return base
        return base + amp * np.maximum(r - knot, 0.0) * np.cos(phi) * np.sin(theta) ** 2

    return RadialStretch(CallableDisplacement(h, knots=() if knot is None else [knot]),
                         rmax=1.0)


def test_identity_geometry_and_names():
    g = Geometry(SK, **NAMES)
    assert g.is_identity and not g.is_hollow and g.nlayers == 3
    assert isinstance(g.mapping, IdentityMapping)
    assert g.layer("mantle") == LayerInfo(1, (0.4, 0.8), name="mantle")
    assert g.layer(-1).name == "crust"
    assert g.interface("cmb") == InterfaceInfo(0, 0.4, (0, 1), name="cmb")
    assert g.interface("surface").between == (2, -1)
    assert [f.radius for f in g.interfaces] == [0.4, 0.8, 1.0]
    with pytest.raises(KeyError, match="no layer named"):
        g.layer("ocean")
    with pytest.raises(IndexError):
        g.interface(3)
    assert g.knots == ()
    assert g.validity()
    testing.check_geometry(g)


def test_hollow_geometry_numbering():
    g = Geometry(Skeleton([0.5, 0.8, 1.0]), interface_names=["inner", "mid", "outer"])
    assert g.is_hollow
    assert g.interface("inner") == InterfaceInfo(0, 0.5, (-1, 0), name="inner")
    assert g.interface("mid").between == (0, 1)
    assert g.interface("outer").between == (1, -1)
    testing.check_geometry(g)


def test_name_refusals():
    with pytest.raises(ValueError, match="unique"):
        Geometry(SK, layer_names=["a", "a", None])
    with pytest.raises(ValueError, match="got 2 layer names"):
        Geometry(SK, layer_names=["a", "b"])
    with pytest.raises(TypeError, match="Skeleton"):
        Geometry([0.0, 1.0])
    with pytest.raises(ValueError, match="rtol"):
        Geometry(SK, rtol=0.0)


def test_renamed():
    g = Geometry(SK, **NAMES)
    h = g.renamed(layers={"core": "inner_core", 2: None}, interfaces=["a", "b", "c"])
    assert [lay.name for lay in h.layers] == ["inner_core", "mantle", None]
    assert [f.name for f in h.interfaces] == ["a", "b", "c"]
    assert [lay.name for lay in g.layers] == NAMES["layer_names"]


def test_a_valid_mapping_with_a_kink_on_a_boundary_is_accepted():
    g = Geometry(SK, mapping=stretch(knot=0.8), **NAMES)
    assert not g.is_identity
    assert g.knots == (0.8,)
    assert g.validity().margin > 0
    testing.check_geometry(g)


def test_invariant_refusals():
    assert Geometry(SK, mapping=stretch(knot=0.7)).knots == (0.7,)
    # a knot beyond the skeleton is the mapping's business, not the body's
    assert Geometry(SK, mapping=stretch(knot=1.7)).knots == ()
    truncated = Geometry(SK, mapping=stretch(knot=0.8)).truncated(0.7)
    assert truncated.knots == ()
    testing.check_geometry(truncated)
    with pytest.raises(ValueError, match="does not preserve orientation"):
        Geometry(SK, mapping=stretch(amp=-3.0))
    with pytest.raises(TypeError, match="not a Mapping"):
        Geometry(SK, mapping=object())

    class Jump(MappingBase):
        """Continuous below r = 0.8, shifted above it."""

        def __call__(self, X):
            X = np.asarray(X, dtype=float)
            r = np.linalg.norm(X, axis=-1)
            return X * np.where(r > 0.8, 1.05, 1.0)[..., None]

        def deformation_gradient(self, X):
            X = np.asarray(X, dtype=float)
            r = np.linalg.norm(X, axis=-1)
            s = np.where(r > 0.8, 1.05, 1.0)
            return s[..., None, None] * np.eye(3)

        def jacobian(self, X):
            return self.deformation_gradient(X)[..., 0, 0] ** 3

    with pytest.raises(ValueError, match="discontinuous across the boundary"):
        Geometry(SK, mapping=Jump())
    Geometry(SK, mapping=Jump(), check=False)       # the caller's assertion


def test_with_mapping_and_scaled():
    g = Geometry(SK, **NAMES)
    d = g.with_mapping(stretch())
    assert d.layer("crust").name == "crust" and not d.is_identity
    big = d.scaled(6.371e6)
    assert np.allclose(big.skeleton.boundaries, 6.371e6 * SK.boundaries)
    assert big.interface("moho").radius == pytest.approx(0.8 * 6.371e6)
    X = np.array([[0.0, 0.0, 0.5], [0.3, 0.2, 0.1]])
    assert np.allclose(big.mapping(6.371e6 * X), 6.371e6 * d.mapping(X))
    assert g.scaled(2.0).is_identity
    testing.check_geometry(big)


def test_stretched():
    g = Geometry(SK, **NAMES)
    d = g.stretched(lambda r, t, p: -0.05 * r * 0.5 * (3 * np.cos(t) ** 2 - 1),
                    name="flat")
    assert isinstance(d.mapping, RadialStretch) and d.mapping.rmax == 1.0
    assert d.mapping.name == "flat" and not d.is_identity
    assert d.layer("crust").name == "crust"
    testing.check_geometry(d)


def test_refined_keeps_the_mapping_and_carries_names():
    g = Geometry(SK, mapping=stretch(knot=0.8), **NAMES)
    fine = g.refined([0.6], interface_names=["floor"])
    assert fine.nlayers == 4
    assert [lay.name for lay in fine.layers] == ["core", None, None, "crust"]
    assert [f.name for f in fine.interfaces] == ["cmb", "floor", "moho", "surface"]
    assert fine.mapping is g.mapping
    testing.check_geometry(fine)


def test_truncated_keeps_the_mapping():
    g = Geometry(SK, mapping=stretch(), **NAMES)
    cut = g.truncated(0.9)
    assert cut.nlayers == 3 and cut.interface(-1).radius == 0.9
    assert cut.interface(-1).name is None
    assert cut.truncated(0.85, name="top").interface("top").radius == 0.85
    on_boundary = g.truncated(0.8)
    assert [f.name for f in on_boundary.interfaces] == ["cmb", "moho"]
    assert [lay.name for lay in on_boundary.layers] == ["core", "mantle"]
    testing.check_geometry(cut)


def test_hollowed_keeps_the_mapping():
    g = Geometry(SK, mapping=stretch(), **NAMES)
    inner = g.hollowed(0.5)
    assert inner.is_hollow and inner.nlayers == 2
    assert inner.interface(0) == InterfaceInfo(0, 0.5, (-1, 0))
    assert [lay.name for lay in inner.layers] == ["mantle", "crust"]
    assert inner.mapping is g.mapping
    on_boundary = g.hollowed(0.4, name="bottom")
    assert [lay.name for lay in on_boundary.layers] == ["mantle", "crust"]
    assert [f.name for f in on_boundary.interfaces] == ["bottom", "moho", "surface"]
    assert g.hollowed(0.4).interface(0).name == "cmb"
    testing.check_geometry(inner)


def test_extended_and_coarsened_need_the_identity():
    g = Geometry(SK, **NAMES)
    grown = g.extended([1.2], layer_names=["buffer"], interface_names=["outer"])
    assert grown.layer("buffer").interval == (1.0, 1.2)
    assert grown.interface("outer").between == (3, -1)
    assert grown.interface("surface").between == (2, 3)
    coarse, cmap = g.coarsened(drop=[0])
    assert [lay.name for lay in coarse.layers] == [None, "crust"]
    assert [f.name for f in coarse.interfaces] == ["moho", "surface"]
    assert cmap.dropped_interior == (0,)
    deformed = g.with_mapping(stretch())
    with pytest.raises(ValueError, match="only be extended while"):
        deformed.extended([1.2])
    with pytest.raises(ValueError, match="only be coarsened while"):
        deformed.coarsened(drop=[0])


# -- knots anywhere ---------------------------------------------------------------

def test_a_relief_inside_a_layer_declares_a_knot_the_geometry_admits():
    """Reliefs on surfaces that are no boundary of the model: a layer_linear
    on a finer skeleton, its knots inside the model's layers."""
    from planetmodel import RadialMesh, RadialStretch, layer_linear

    fine = Skeleton([0.0, 0.25, 0.4, 0.6, 0.8, 1.0])

    def tilt(t, p):
        return 0.01 * np.sin(t) * np.cos(p) + 0.005 * np.cos(t)

    h = layer_linear(fine, [None, tilt, None, tilt, None, tilt])
    assert set(h.knots) == {0.25, 0.4, 0.6, 0.8}
    testing.check_displacement(h, SK)
    g = Geometry(SK, mapping=RadialStretch(h, rmax=1.0), **NAMES)
    assert set(g.knots) == {0.25, 0.4, 0.6, 0.8}
    testing.check_geometry(g)
    mesh = RadialMesh(g, ngll=3, drmax=0.3)
    edges = np.append(mesh.left, mesh.right[-1])
    for k in (0.25, 0.6):
        assert np.min(np.abs(edges - k)) < 1e-12        # a knot is an element edge


def test_an_undeclared_kink_is_still_caught():
    from planetmodel import CallableDisplacement
    kinked = CallableDisplacement(
        lambda r, t, p: 0.05 * np.maximum(r - 0.6, 0.0) * np.cos(t) + 0.0 * p)
    with pytest.raises(AssertionError, match="not declared as a knot"):
        testing.check_displacement(kinked, SK)
    declared = CallableDisplacement(
        lambda r, t, p: 0.05 * np.maximum(r - 0.6, 0.0) * np.cos(t) + 0.0 * p,
        knots=[0.6])
    testing.check_displacement(declared, SK)
