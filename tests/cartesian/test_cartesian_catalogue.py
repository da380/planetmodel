"""The flat catalogue: the layered isotropic model and the reference crust
over a mantle, their values, names and options, held to the model
contract; a prescribed gravity; and the surgery of a flat model
returning flat classes, its repr naming the depth."""
import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic as SphericalLayered
from planetmodel import Scales
from planetmodel import cartesian as flat
from planetmodel.cartesian.catalogue import CRUST_OVER_MANTLE
from planetmodel.fields import RadialField
from planetmodel.testing import check_geometry, check_model

BOX = flat.Box([(0.0, 400e3), (-200e3, 200e3)])
RECTANGLE = flat.Rectangle((-500e3, 500e3))


def test_the_crust_over_a_mantle_holds_its_documented_values():
    m = flat.CrustOverMantle(BOX)
    assert m.name == "crust over mantle" and m.scales == Scales.SI
    assert [lay.name for lay in m.layers] == ["crust", "lithospheric_mantle",
                                              "asthenosphere"]
    assert [f.name for f in m.geometry.interfaces] == ["surface", "moho", "lab"]
    assert list(m.skeleton.boundaries) == [0.0, 30e3, 100e3, 200e3]
    for layer, (name, top, rho, vp, vs) in zip(m.layers, CRUST_OVER_MANTLE):
        lo, hi = layer.interval
        assert lo == top and layer.name == name
        d = np.linspace(lo, hi, 7)
        assert np.all(layer["rho"](d) == rho) and np.all(layer["vp"](d) == vp)
        assert np.all(layer["vs"](d) == vs)
        assert np.allclose(layer["L"](d), rho * vs ** 2, rtol=1e-15, atol=0)
        assert np.allclose(layer["C"](d), rho * vp ** 2, rtol=1e-15, atol=0)
        assert all(type(f) is flat.DepthField for f in layer.fields.values())
        assert not m.is_viscoelastic(layer.index)
    check_model(m)
    check_geometry(m.geometry)


def test_the_crust_over_a_mantle_takes_a_maxwell_asthenosphere_and_a_base():
    m = flat.CrustOverMantle(RECTANGLE, viscosity=1e20, base=400e3)
    assert m.skeleton.boundaries[-1] == 400e3 and m.geometry.domain == RECTANGLE
    assert m.name == "crust over mantle (Maxwell asthenosphere)"
    eta = m.layer("asthenosphere")["viscosity"]
    assert type(eta) is flat.DepthField and eta(300e3) == 1e20
    assert [m.is_viscoelastic(i) for i in range(3)] == [False, False, True]
    assert "viscosity" not in m.layer("crust")
    check_model(m)
    with pytest.raises(ValueError, match="below the base of the lithosphere"):
        flat.CrustOverMantle(BOX, base=100e3)
    with pytest.raises(ValueError, match="the viscosity is positive"):
        flat.CrustOverMantle(BOX, viscosity=0.0)
    km = m.converted(Scales(length=1e3, mass=1.0, time=1.0, temperature=1.0))
    assert type(km) is flat.CrustOverMantle and km.skeleton.boundaries[-1] == 400.0
    assert km.geometry.domain == RECTANGLE.scaled(1e-3)


def test_the_layered_model_is_the_spherical_one_by_depth():
    rho, vp, vs = [2800.0, [3300.0, 1e-3]], [6400.0, 8000.0], [3700.0, [4500.0, 2e-3]]
    m = flat.LayeredIsotropicElastic([0.0, 30e3, 200e3], BOX, rho=rho, vp=vp, vs=vs,
                                     scale=1e3, layer_names=["crust", "mantle"],
                                     interface_names=["surface", "moho"], name="two")
    sphere = SphericalLayered([0.0, 30e3, 200e3], rho=rho, vp=vp, vs=vs, scale=1e3)
    d = np.linspace(30e3, 200e3, 9)
    for name in ("rho", "vp", "vs", "A", "C", "F", "L", "N"):
        a, b = m.layer("mantle")[name], sphere.layer(1)[name]
        assert type(a) is flat.DepthField and type(b) is RadialField
        assert np.allclose(a(d), b(d), rtol=1e-15, atol=0), name
    assert np.isclose(m.layer("mantle")["rho"](100e3), 3300.1, rtol=1e-15)
    check_model(m)
    slab = flat.LayeredIsotropicElastic.homogeneous(50e3, RECTANGLE, rho=2700.0,
                                                    vp=6000.0, vs=3500.0, name="slab")
    assert slab.nlayers == 1 and slab.layer("slab").interval == (0.0, 50e3)
    with pytest.raises(ValueError, match="rho needs 2 values, got 1"):
        flat.LayeredIsotropicElastic([0.0, 1.0, 2.0], BOX, rho=[1.0], vp=[1.0, 1.0],
                                     vs=[1.0, 1.0])
    fluid = flat.LayeredIsotropicElastic([0.0, 4e3, 30e3], BOX, rho=[1020.0, 2800.0],
                                         vp=[1450.0, 6400.0], vs=[0.0, 3700.0])
    assert fluid.is_fluid(0) and not fluid.is_fluid(1)


def test_a_prescribed_gravity_is_a_plain_field_of_every_layer():
    m = flat.CrustOverMantle(BOX).with_uniform_gravity(9.81)
    for layer in m.layers:
        g = layer["g"]
        assert type(g) is flat.DepthField and g.radial_degree == 0
        assert g(layer.interval[1]) == 9.81
    km = m.converted(Scales(length=1e3, mass=1.0, time=1.0, temperature=1.0))
    assert np.isclose(km.layer(0)["g"](10.0), 9.81e-3, rtol=1e-15)
    padded = m.extended([500e3], fields="hold")
    assert padded.layer(-1)["g"](400e3) == 9.81
    check_model(m)
    with pytest.raises(ValueError, match="already holds 'g'"):
        m.with_uniform_gravity(9.8)
    assert m.with_uniform_gravity(9.8, replace=True).layer(0)["g"](0.0) == 9.8
    with pytest.raises(ValueError, match="finite magnitude"):
        m.with_uniform_gravity(-1.0, replace=True)


def test_surgery_on_a_flat_model_returns_flat_classes():
    m = flat.CrustOverMantle(BOX)
    made = {"refined": m.refined([60e3], names=["d60"]),
            "truncated": m.truncated(150e3),
            "hollowed": m.hollowed(10e3),
            "extrapolated": m.extended([300e3], fields="extrapolate"),
            "held": m.extended([300e3], fields="hold"),
            "coarsened": m.coarsened(drop=["lab"])[0]}
    for verb, out in made.items():
        assert type(out) is flat.CrustOverMantle, verb
        assert type(out.geometry) is flat.Geometry, verb
        for layer in out.layers:
            for name, f in layer.fields.items():
                assert type(f) is flat.DepthField, (verb, name)
        check_model(out)
    coarse = made["coarsened"]
    assert [f.name for f in coarse.geometry.interfaces] == ["surface", "moho"]
    assert coarse.layer(1).interval == (30e3, 200e3)
    assert np.allclose(coarse.layer(1)["vs"](120e3), 4500.0, rtol=1e-12)
    assert "depth in [0, 200000]" in repr(m.geometry)
    assert "radius" not in repr(m.geometry) and " r in" not in repr(m.geometry)
    assert repr(m.skeleton) == "Skeleton(3 layers, [0, 200000])"
    assert m.coordinates is flat.FLAT
