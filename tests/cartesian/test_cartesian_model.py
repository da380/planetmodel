"""The flat model: the model contract on a box and a rectangle, refusals of
the other system by name, and padding below the base by holding the
lowest properties, exactly."""
import numpy as np
import pytest

from planetmodel import (DENSITY, SCALAR, LayeredIsotropicElastic, RadialField,
                         Scales, Skeleton)
from planetmodel import cartesian as flat
from planetmodel.fields import HeldField
from planetmodel.geometry import Geometry as SphericalGeometry
from planetmodel.layerfunction import polynomial_layer
from planetmodel.mapping import MappingBase
from planetmodel.model import Model as SphericalModel
from planetmodel.model import ModelBase
from planetmodel.testing import check_model
from planetmodel.vocabulary import FieldSpec

DEPTHS = [0.0, 20e3, 60e3, 100e3]
BOX = flat.Box([(0.0, 400e3), (0.0, 200e3)])
SPECS = {"q": FieldSpec(SCALAR, dimensions=None)}


class Tilt(MappingBase):
    """A small affine tilt of the body: not the identity."""

    A = np.array([[1.0, 0.0, 0.02], [0.0, 1.0, 0.0], [-0.01, 0.0, 1.05]])

    def __call__(self, X):
        return np.asarray(X, dtype=float) @ self.A.T

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(self.A, X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        return np.full(np.shape(X)[:-1], np.linalg.det(self.A))


def three_layers(domain=BOX, **kw):
    """Crust, mantle and a deep layer in SI: density of depth alone, with a
    lateral variation in the mantle."""
    g = flat.Geometry(Skeleton(DEPTHS), domain, layer_names=["crust", "mantle", "deep"],
                      interface_names=["surface", "moho", "d60"], **kw)
    crust = (0.0, 20e3)
    mantle = (20e3, 60e3)
    deep = (60e3, 100e3)
    rho = [flat.DepthField(crust, polynomial_layer(crust, [2700.0, 1e-2]),
                           character=DENSITY),
           flat.AnalyticField(mantle, lambda d, x, y: 3300.0 + 2e-3 * d
                              + 10.0 * np.sin(2 * np.pi * x / 400e3),
                              character=DENSITY),
           flat.DepthField(deep, polynomial_layer(deep, [3350.0, 1e-3]),
                           character=DENSITY)]
    return flat.Model(g, [{"rho": r} for r in rho], name="three layers")


def test_a_flat_model_holds_to_the_model_contract():
    for m in (three_layers(), three_layers(mapping=Tilt()),
              three_layers(domain=flat.Rectangle((0.0, 400e3))),
              three_layers().hollowed(20e3)):
        assert isinstance(m, ModelBase) and not isinstance(m, SphericalModel)
        check_model(m)
    m = three_layers()
    assert [lay.name for lay in m.layers] == ["crust", "mantle", "deep"]
    assert m.layer("crust").index == 0 and m.layer(0).interval == (0.0, 20e3)


def test_conversion_scales_the_box_with_the_depths():
    m = three_layers()
    km = m.converted(Scales(length=1e3, mass=1.0, time=1.0, temperature=1.0))
    assert km.geometry.domain == flat.Box([(0.0, 400.0), (0.0, 200.0)])
    assert np.allclose(km.skeleton.boundaries, [0.0, 20.0, 60.0, 100.0])
    assert type(km) is flat.Model and type(km.layer(0)["rho"]) is flat.DepthField


def test_the_other_system_is_refused_by_name():
    g = flat.Geometry(Skeleton([0.0, 1.0]), flat.Box([(0, 1), (0, 1)]))
    spherical = SphericalGeometry(Skeleton([0.0, 1.0]))
    with pytest.raises(ValueError, match="is in the coordinates 'spherical', the "
                                         "geometry in 'flat'"):
        flat.Model(g, [{"q": RadialField((0.0, 1.0), 1.0)}], specs=SPECS)
    with pytest.raises(ValueError, match="is in the coordinates 'flat', the "
                                         "geometry in 'spherical'"):
        SphericalModel(spherical, [{"q": flat.constant_field((0.0, 1.0), 1.0)}],
                       specs=SPECS)
    with pytest.raises(TypeError, match="expected a planetmodel.cartesian.geometry."
                                        "Geometry, got planetmodel.geometry.Geometry"):
        flat.Model(spherical, [{}])
    with pytest.raises(TypeError, match="expected a planetmodel.geometry.Geometry, "
                                        "got planetmodel.cartesian.geometry.Geometry"):
        SphericalModel(g, [{}])
    m = flat.Model(g, [{}])
    with pytest.raises(TypeError, match="expected a planetmodel.cartesian"):
        m.replaced(geometry=spherical)
    with pytest.raises(ValueError, match="is in the coordinates 'spherical'"):
        m.with_field(0, "q", RadialField((0.0, 1.0), 1.0))


def test_holding_pads_below_the_base_with_the_lowest_properties_exactly():
    m = three_layers()
    rho = m.layer(-1)["rho"]
    base = float(rho(100e3))
    padded = m.extended([150e3, 250e3], fields="hold",
                        layer_names=["pad", "far"],
                        interface_names=["base", "pad"])
    assert type(padded) is flat.Model and padded.nlayers == 5
    assert [f.radius for f in padded.geometry.interfaces] == [0.0, 20e3, 60e3, 100e3,
                                                             150e3]
    assert padded.geometry.interface("base").between == (3, 2)
    d = np.linspace(100e3, 250e3, 31)
    for layer in padded.layers[3:]:
        held = layer["rho"]
        assert type(held) is flat.DepthField and held.radial_degree == 0
        lo, hi = layer.interval
        assert np.all(held(d[(d >= lo) & (d <= hi)]) == base)
    # the field above is untouched and the padding joins it without a step
    assert padded.layer(2)["rho"] is rho
    assert padded.layer(3)["rho"](100e3) == rho(100e3)
    check_model(padded)


def test_holding_a_lateral_field_keeps_its_lateral_variation():
    m = three_layers().truncated(60e3)
    padded = m.extended([90e3], fields="hold")
    held = padded.layer(-1)["rho"]
    assert isinstance(held, HeldField) and held.at == 60e3
    rng = np.random.default_rng(7)
    x, y = rng.uniform(0, 400e3, 40), rng.uniform(0, 200e3, 40)
    want = m.layer(-1)["rho"].evaluate(np.full(40, 60e3), x, y)
    for depth in (60e3, 75e3, 90e3):
        assert np.array_equal(held.evaluate(np.full(40, depth), x, y), want)


def test_holding_works_on_the_sphere_too():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[3.0, 2.0], vp=[2.0, 1.5],
                                vs=[1.0, 0.8])
    shell = m.extended([1.2], fields="hold")
    assert type(shell.layer(-1)["rho"]) is RadialField
    assert shell.layer(-1)["rho"](1.1) == m.layer(-1)["rho"](1.0)
    with pytest.raises(ValueError, match=r"one of \('extrapolate', 'hold'\)"):
        m.extended([1.2], fields="keep")


def test_coarsening_a_flat_model_keeps_the_surface():
    m = three_layers()
    flatter = m.with_field(1, "rho", flat.DepthField(
        (20e3, 60e3), polynomial_layer((20e3, 60e3), [3300.0, 2e-3]),
        character=DENSITY), replace=True)
    with pytest.raises(ValueError, match="'surface' is the outermost boundary"):
        flatter.coarsened(drop=["surface"])
    with pytest.raises(ValueError, match="is not a scalar of the depth alone"):
        m.coarsened(drop=["d60"])
    coarse, fit = flatter.coarsened(drop=["d60"])
    assert [f.name for f in coarse.geometry.interfaces] == ["surface", "moho"]
    assert coarse.layer(1).interval == (20e3, 100e3) and fit.worst < 1e-2
    # an integer is an interface on the model and on its geometry alike
    for selection in ({"drop": [2]}, {"keep": [1]}, {"drop": ["d60"]}):
        again, refit = flatter.coarsened(**selection)
        g, cmap = flatter.geometry.coarsened(**selection)
        assert again.skeleton == coarse.skeleton == g.skeleton
        assert cmap.layers == refit.map.layers == fit.map.layers
    with pytest.raises(ValueError, match="'surface' is the outermost boundary"):
        flatter.geometry.coarsened(drop=[0])
