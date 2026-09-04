"""Mode B, end to end: the gravity-solver archetype, with no gmsh anywhere.

The claim: a consumer in the style of Myhill, Maitra &
Al-Attar (2026) gets everything it needs -- the reference body, the
referential density, and an evaluable mapping with F, J and a
linearisation -- without a mesh ever being built or the meshing extra
ever being installed.
"""
import subprocess
import sys

import numpy as np
import pytest

from planetmodel import PREM, layer_linear
from planetmodel.model.mapping import validity_lattice
from planetmodel.model.topography import AnalyticTopography
from planetmodel.testing import check_mapping


@pytest.fixture(scope="module")
def body():
    """PREM in a buffer -- MMA26's reference domain B -- with relief on
    the CMB and the surface."""
    cmb_topo = AnalyticTopography(
        lambda t, p: 1.5e3 * (np.cos(t) ** 2 - 1.0 / 3.0))
    surf_topo = AnalyticTopography(
        lambda t, p: 3.0e3 * np.sin(t) ** 2 * np.cos(2.0 * p))
    return (PREM(ocean=False)
            .name_interface(1, "cmb")
            .name_interface(-1, "surface")
            .with_buffer(ratio=0.3)
            .with_surface("cmb", cmb_topo)
            .with_surface("surface", surf_topo))


@pytest.fixture(scope="module")
def m(body):
    return body.mapping(rule=layer_linear())


@pytest.fixture(scope="module")
def X(body):
    rng = np.random.default_rng(21)
    v = rng.normal(size=(300, 3))
    v /= np.linalg.norm(v, axis=1)[:, None]
    rmax = float(body.skeleton.boundaries[-1])
    return v * rng.uniform(0.05 * rmax, 0.98 * rmax, size=(300, 1))


def test_the_whole_flow_never_touches_gmsh():
    """The model layer never imports gmsh.

    Run in a subprocess: the mesher tests import gmsh into the session,
    so an in-process check on sys.modules would report on the test run
    rather than on Mode B.  What matters is that this flow
    pulls gmsh in a *fresh* interpreter, which is what a consumer gets.
    """
    code = (
        "import sys\n"
        "import numpy as np\n"
        "from planetmodel import PREM, layer_linear\n"
        "from planetmodel.model.topography import AnalyticTopography\n"
        "body = (PREM(ocean=False).name_interface(-1, 'surface')\n"
        "        .with_buffer(ratio=0.3)\n"
        "        .with_surface('surface',\n"
        "                      AnalyticTopography(lambda t, p: 3e3*np.cos(t))))\n"
        "m = body.mapping(rule=layer_linear())\n"
        "X = np.array([[4e6, 1e6, -2e6], [1e6, 2e6, 3e6]])\n"
        "m(X); m.deformation_gradient(X); m.jacobian(X)\n"
        "body.rho.evaluate(np.linspace(1e5, 6e6, 50))\n"
        "assert 'gmsh' not in sys.modules, 'Mode B pulled in gmsh'\n"
    )
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True)
    assert r.returncode == 0, r.stderr


def test_the_model_interface_is_rho_and_the_mapping(body, m, X):
    """MMA26 §2: the entire model interface for the gravity solver."""
    rho = body.rho                                  # referential density
    assert rho.character.weight == 1
    F = m.deformation_gradient(X)
    J = m.jacobian(X)
    assert F.shape == (300, 3, 3) and J.shape == (300,)
    assert np.all(J > 0.0)
    # the solver forms a = J C^-1 itself, in one line, from F and J:
    C = np.einsum("...ki,...kj->...ij", F, F)
    a = J[..., None, None] * np.linalg.inv(C)
    # ... and planetmodel's diagnostic version agrees with that one line
    assert np.allclose(a, m.gravity_tensor(X), rtol=1e-10)


def test_the_mapping_is_the_identity_on_the_buffer_boundary(body, m):
    """m fixes the outer boundary of B, as MMA26 requires."""
    b = float(body.skeleton.boundaries[-1])
    th = np.linspace(0.1, np.pi - 0.1, 7)
    ring = np.stack([b * np.sin(th), np.zeros_like(th), b * np.cos(th)],
                    axis=-1)
    assert np.allclose(m(ring), ring, rtol=0, atol=1e-6)


def test_the_mapping_satisfies_its_contract(m, X):
    check_mapping(m, X, rtol=1e-4)      # layer_linear h: smooth per span


def test_the_mapping_is_valid_everywhere(body, m):
    rep = m.is_valid(sample=validity_lattice(body.skeleton))
    assert rep, repr(rep)


def test_the_adjoint_side_gets_its_linearisation(body, m, X):
    """delta F and delta J from a perturbation of the displacement --
    what MMA26 eq. (37) contracts to form delta a."""
    from planetmodel.model.displacement import CallableDisplacement

    delta = CallableDisplacement(
        lambda r, t, p: 100.0 * np.cos(t)
        * np.asarray(r, float) / 6.371e6)
    lin = m.linearise(delta, X=X)
    assert lin.dF.shape == (300, 3, 3) and lin.dJ.shape == (300,)
    assert np.all(np.isfinite(lin.dF)) and np.all(np.isfinite(lin.dJ))


def test_density_pushes_forward_as_rho_over_J(body, m):
    """rho_phys = rho_ref / J at reference points -- the identity the
    two papers share, on the real model and the real mapping."""
    from planetmodel.model.character import DENSITY

    r = np.linspace(3.6e6, 6.3e6, 40)
    th = np.full_like(r, 1.1)
    ph = np.full_like(r, -0.7)
    Xr = np.stack([r * np.sin(th) * np.cos(ph), r * np.sin(th) * np.sin(ph),
                   r * np.cos(th)], axis=-1)
    rho = body.rho.evaluate(r)
    J = m.jacobian(Xr)
    assert np.allclose(m.push_forward(rho, Xr, DENSITY), rho / J, rtol=1e-14)


def test_a_bare_body_maps_by_the_identity(body):
    from planetmodel.model.mapping import IdentityMapping

    assert isinstance(PREM(ocean=False).mapping(), IdentityMapping)


def test_a_prescribed_displacement_is_accepted_directly(body):
    """Answer 3 of the design review: the callable is the interface."""
    m = body.mapping(displacement=lambda r, t, p: np.zeros_like(
        np.asarray(r, dtype=float)))
    X = np.array([[4e6, 1e6, -2e6]])
    assert np.allclose(m(X), X)


def test_rule_and_displacement_together_are_refused(body):
    with pytest.raises(ValueError, match="not both"):
        body.mapping(rule=layer_linear(), displacement=lambda r, t, p: r)
