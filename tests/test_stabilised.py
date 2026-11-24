"""Unstably stratified fluid made stable: the nearest stable layer, of
the same velocity, mass and moment of inertia."""
import numpy as np
import pytest

from planetmodel import Elastic, Geometry, Model, SelfGravitating, Skeleton, testing
from planetmodel.catalogue import LayeredIsotropicElastic, PREM
from planetmodel.character import DENSITY
from planetmodel.fields import RadialField
from planetmodel.layerfunction import polynomial_layer
from planetmodel.mesh1d.stratification import (
    BULLEN_EXCESS,
    Stabilisation,
    StableLayer,
    stable_layer,
)

KM = 1e3


def inertia_of(layer):
    """The moment of inertia of a layer about the centre, exactly."""
    lo, hi = layer.interval
    r4 = RadialField((lo, hi), polynomial_layer((lo, hi), [0, 0, 0, 0, 1.0]))
    return 8.0 * np.pi / 3.0 * float((layer["rho"] * r4).integrate(lo, hi))


@pytest.fixture(scope="module")
def prem():
    return PREM()


@pytest.fixture(scope="module")
def stable_prem(prem):
    return prem.stabilised(layers=["outer_core"])


def test_prems_outer_core_is_made_stable(prem, stable_prem):
    stable, report = stable_prem
    assert type(stable) is type(prem)
    core, before = stable.layer("outer_core"), prem.layer("outer_core")
    i = core.index
    lo, hi = core.interval
    r = np.linspace(lo, hi, 2001)[1:-1]
    assert np.min(stable.bullen_parameter(r)) >= 1.0 - 1e-8
    assert np.min(core[BULLEN_EXCESS](r)) >= 0.0
    # the velocity, the mass and the moment of inertia are the layer's own
    assert core["vpv"] is before["vpv"]
    assert np.allclose(stable.layer_masses(), prem.layer_masses(), rtol=1e-11)
    assert np.isclose(inertia_of(core), inertia_of(before), rtol=1e-11)
    # so nothing else has changed
    for layer in prem.layers:
        if layer.index != i:
            assert stable.layers[layer.index].fields == layer.fields
    outside = np.linspace(hi, prem.skeleton.boundaries[-1], 7)
    assert np.allclose(stable.gravity(outside), prem.gravity(outside), rtol=1e-11)
    # by a few parts in ten thousand of the density
    ends = np.linspace(lo, hi, 2001)
    change = np.max(np.abs(core["rho"](ends) / before["rho"](ends) - 1.0))
    assert 2e-4 < change < 4e-4
    found = report.layers[i]
    assert isinstance(report, Stabilisation) and isinstance(found, StableLayer)
    assert report.changed == (i,)
    assert np.isclose(found.density_change, change, rtol=1e-2)
    assert abs(found.inertia_change) < 1e-11
    # neutral from 1403 to 2643 km and from 2826 km to the top
    assert np.allclose(np.array(found.neutral) / KM,
                       [[1403.0, 2643.0], [2825.0, 3480.0]], atol=3.0)
    assert "layer 1" in report.describe()
    testing.check_model(stable)


def test_a_stable_model_is_returned_as_it_is(stable_prem):
    stable, _ = stable_prem
    again, report = stable.stabilised(layers=["outer_core"])
    assert again is stable
    assert report.changed == () and "stable as it stood" in report.describe()
    # a layer that was never unstable is not described anew
    plain = LayeredIsotropicElastic(
        [0.0, 3480e3, 6371e3], rho=[(13000.0, -8000.0), 4500.0], scale=6371e3,
        vp=[14000.0, 11000.0], vs=[0.0, 6000.0])
    same, report = plain.stabilised()
    assert same is plain and report.changed == ()


def test_holding_the_mass_alone(prem):
    stable, report = prem.stabilised(layers=["outer_core"], inertia=False)
    core, before = stable.layer("outer_core"), prem.layer("outer_core")
    found = report.layers[core.index]
    assert found.tilt == 0.0
    assert np.isclose(stable.layer_masses()[core.index],
                      prem.layer_masses()[core.index], rtol=1e-11)
    assert np.isclose(inertia_of(core) / inertia_of(before) - 1.0,
                      found.inertia_change, atol=1e-12)
    assert -1e-5 < found.inertia_change < -1e-6
    # outside the pooled intervals the density is what it was, to the
    # accuracy of the grid the pooling is done on
    lo = core.interval[0]
    r = np.linspace(lo, found.neutral[0][0], 9)[:-1]
    assert np.allclose(core["rho"](r), before["rho"](r), rtol=1e-6)
    # with a threshold the weak stratification between the two goes too
    wider, report = prem.stabilised(layers=["outer_core"], inertia=False,
                                    neutral_below=0.01)
    neutral = report.layers[core.index].neutral
    assert len(neutral) == 1 and np.isclose(neutral[0][1], core.interval[1])


def test_a_layer_that_cannot_keep_its_moment_of_inertia_is_refused(prem):
    # a uniform layer is unstable throughout
    with pytest.raises(ValueError, match="ocean.*cannot keep its moment of inertia"):
        prem.stabilised(layers=["ocean"])
    with pytest.raises(ValueError, match="cannot keep its moment of inertia"):
        prem.stabilised()
    stable, report = prem.stabilised(layers=["ocean"], inertia=False)
    ocean = stable.layer("ocean")
    assert stable.is_neutral("ocean")
    lo, hi = ocean.interval
    assert ocean["rho"](lo) > 1020.0 > ocean["rho"](hi)
    assert report.layers[ocean.index].neutral == ((lo, hi),)
    with pytest.raises(ValueError, match="neutral throughout once"):
        prem.stabilised(layers=["outer_core"], neutral_below=0.05)
    with pytest.raises(ValueError, match="is not a fluid layer"):
        prem.stabilised(layers=["inner_core"])


class Planet(Elastic, SelfGravitating, Model):
    """A model given by rho, kappa and mu."""


def test_a_layer_described_by_its_modulus_is_described_by_its_velocity():
    sk = Skeleton([0.0, 3480e3, 6371e3])

    def field(i, value, name):
        iv = sk.interval(i)
        return RadialField(iv, polynomial_layer(iv, [value]), character=DENSITY,
                           name=name)

    m = Planet(Geometry(sk), [
        {"rho": field(0, 11000.0, "rho"), "kappa": field(0, 9e11, "kappa"),
         "mu": field(0, 0.0, "mu")},
        {"rho": field(1, 4500.0, "rho"), "kappa": field(1, 4e11, "kappa"),
         "mu": field(1, 1.6e11, "mu")}])
    stable, report = m.stabilised(inertia=False)
    core = stable.layer(0)
    assert report.changed == (0,) and stable.is_neutral(0)
    r = np.linspace(0.0, 3480e3, 9)[1:-1]
    # the velocity of the uniform layer, sqrt(kappa / rho), is what is kept
    assert isinstance(core["vp"], RadialField)
    assert np.allclose(core["vp"](r), np.sqrt(9e11 / 11000.0))
    assert np.allclose(stable.bullen_excess(r), 0.0, atol=1e-8)
    assert np.isclose(stable.layer_masses()[0], m.layer_masses()[0], rtol=1e-11)
    testing.check_model(stable)


def test_the_function_of_a_layer():
    # a layer stable as it stands needs nothing; a cusp at the centre of an
    # unstable one is refused
    interval = (0.0, 1.0)
    assert stable_layer(interval, polynomial_layer(interval, [2.0, 0.0, -1.0]), 5.0,
                        G=1.0) is None
    with pytest.raises(ValueError, match="slope at the centre"):
        stable_layer(interval, polynomial_layer(interval, [2.0, 0.5]), 1.0, G=1.0)
