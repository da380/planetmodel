"""Nodal values of a model's fields on the radial mesh."""
import numpy as np
import pytest

from planetmodel import RadialMesh
from planetmodel.catalogue import PREM
from planetmodel.character import VECTOR
from planetmodel.fields import AnalyticField, RadialField
from planetmodel.mesh1d.gravity import gravity


@pytest.fixture(scope="module")
def model():
    return PREM()


@pytest.fixture(scope="module")
def mesh(model):
    return RadialMesh(model, ngll=4, drmax=400e3)


def test_nodal_values_are_each_elements_own_layer(model, mesh):
    rho = mesh.nodal(model, "rho")
    assert rho.shape == (mesh.nspec, mesh.ngll) and rho.dtype == np.float64
    for e in range(mesh.nspec):
        want = model.layer(int(mesh.layer[e]))["rho"](mesh.r[e])
        assert np.allclose(rho[e], want, rtol=1e-14)
    cmb = 3480e3
    e_below = mesh.element_at(cmb) - 1
    e_above = mesh.element_at(cmb)
    assert rho[e_below, -1] == model.layer("outer_core")["rho"](cmb)
    assert rho[e_above, 0] == model.layer("lowermost_mantle")["rho"](cmb)
    assert rho[e_below, -1] != rho[e_above, 0]


def test_nodal_derivative(model, mesh):
    d = mesh.nodal(model, "rho", nu=1)
    for e in range(mesh.nspec):
        f = model.layer(int(mesh.layer[e]))["rho"].derivative()
        assert np.allclose(d[e], f(mesh.r[e]), rtol=1e-13)


def test_missing_is_refused_or_nan(model, mesh):
    with pytest.raises(KeyError, match="holds no field 'qmu'"):
        mesh.nodal(model, "qmu")
    q = mesh.nodal(model, "qmu", missing="nan")
    fluid = np.isin(mesh.layer, (1, 12))
    assert np.all(np.isnan(q[fluid])) and np.all(np.isfinite(q[~fluid]))
    with pytest.raises(ValueError, match="missing"):
        mesh.nodal(model, "rho", missing="zero")
    with pytest.raises(KeyError, match="no layer"):
        mesh.nodal(model, "nothing", missing="nan")


def test_a_radial_vector_field_gets_its_components(model, mesh):
    iv = model.layer(3).interval
    v = RadialField(iv, [1.0, 2.0, 3.0], character=VECTOR)
    m = model.with_field(3, "v", v)
    out = mesh.nodal(m, "v", missing="nan")
    assert out.shape == (mesh.nspec, mesh.ngll, 3)
    assert np.allclose(out[mesh.layer == 3], [1.0, 2.0, 3.0])


def test_a_direction_dependent_field_is_refused(model, mesh):
    iv = model.layer(3).interval
    m = model.with_field(3, "a", AnalyticField(iv, lambda r, t, p: np.cos(t)))
    with pytest.raises(ValueError, match="depends on direction"):
        mesh.nodal(m, "a", missing="nan")


def test_nodal_gravity_is_over_the_whole_model(model):
    mesh = RadialMesh(model, ngll=4, drmax=400e3, rmin=5000e3)
    g = mesh.nodal_gravity(model)
    assert g.shape == (mesh.nspec, mesh.ngll)
    assert np.allclose(g, gravity(model, mesh.r), rtol=1e-14)
    assert g[0, 0] > 9.0
