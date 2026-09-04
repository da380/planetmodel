"""The mesher's half of the units story.

These need planetmodel.mesh3d importable, whose package guard requires gmsh
-- even though _units.py itself is pure numpy -- so they live with the
gmsh-marked tests rather than in the core suite.  The core CI job runs
without the meshing extra precisely to catch a stray import like that,
and it did.
"""
import numpy as np
import pytest

pytest.importorskip("gmsh", reason="needs the planetmodel[meshing] extra")

from planetmodel import PREM, layer_linear  # noqa: E402
from planetmodel.model.topography import AnalyticTopography  # noqa: E402

pytestmark = pytest.mark.gmsh


@pytest.fixture(scope="module")
def prem():
    return PREM(ocean=False)


@pytest.fixture(scope="module")
def nd(prem):
    return prem.nondimensionalised()

def test_mesh_units_for_an_si_body(prem):
    from planetmodel.mesh3d._units import resolve_mesh_units

    units = resolve_mesh_units(prem, 6.368e6)
    assert units.divisor == pytest.approx(6.368e6)
    assert units.rref_m == pytest.approx(6.368e6)
    with pytest.raises(ValueError, match="needs rref"):
        resolve_mesh_units(prem, None)


def test_mesh_units_for_an_nd_body(nd):
    from planetmodel.mesh3d._units import resolve_mesh_units

    units = resolve_mesh_units(nd, None)
    assert units.divisor == 1.0
    assert units.rref_m == pytest.approx(nd.scales.length)
    with pytest.raises(ValueError, match="one answer, not two"):
        resolve_mesh_units(nd, 5.0e6)


def test_unpleasant_coordinates_warn(prem):
    from planetmodel.mesh3d._units import resolve_mesh_units

    with pytest.warns(UserWarning, match="tuned for coordinates"):
        resolve_mesh_units(prem, 1.0)          # outer radius 6.4e6 mesh units


def test_geometry_scaled_mapping_conjugates(prem):
    from planetmodel.mesh3d._units import GeometryScaledMapping

    body = (prem.name_interface(-1, "surface")
            .with_surface("surface", AnalyticTopography(
                lambda t, p: 3.0e3 * np.cos(t))))
    m = body.mapping(rule=layer_linear())
    L = 6.368e6
    g = GeometryScaledMapping(m, L)

    X_nd = np.array([[0.5, 0.3, 0.2], [0.0, 0.0, 0.9]])
    assert np.allclose(g(X_nd), m(X_nd * L) / L, rtol=1e-14)
    # F and J pass through unchanged: dimensionless
    assert np.allclose(g.deformation_gradient(X_nd),
                       m.deformation_gradient(X_nd * L))
    assert np.allclose(g.jacobian(X_nd), m.jacobian(X_nd * L))
    # knots arrive in mesh units
    assert max(g.knots) == pytest.approx(1.0)
