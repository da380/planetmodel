"""Every shipped implementation passes its contract."""
import numpy as np
import pytest

from planetmodel import (CallableDisplacement, Geometry, IdentityMapping, RadialStretch,
                         ScaledMapping, Skeleton, ZeroDisplacement, testing,
                         validity_lattice)
from planetmodel.frames import cartesian_points

SK = Skeleton([0.0, 0.3, 0.7, 1.0])
HOLLOW = Skeleton([0.2, 0.7, 1.0])


def bump(r, theta, phi):
    return 0.02 * r * np.sin(theta) ** 2 * np.cos(3.0 * phi)


DISPLACEMENTS = [ZeroDisplacement(), CallableDisplacement(bump)]
MAPPINGS = [IdentityMapping(), RadialStretch(bump, rmax=1.0),
            ScaledMapping(RadialStretch(bump, rmax=1.0), 3.0)]


@pytest.mark.parametrize("h", DISPLACEMENTS)
def test_displacements(h):
    testing.check_displacement(h, SK)


@pytest.mark.parametrize("m", MAPPINGS)
def test_mappings(m):
    k = 3.0 if isinstance(m, ScaledMapping) else 1.0
    X = k * cartesian_points(*validity_lattice(SK, n_r=2, n_theta=4, n_phi=3))
    testing.check_mapping(m, X.reshape(-1, 3))


@pytest.mark.parametrize("sk", [SK, HOLLOW])
@pytest.mark.parametrize("mapping", [None, RadialStretch(bump, rmax=1.0)])
def test_geometries(sk, mapping):
    testing.check_geometry(Geometry(sk, mapping=mapping))


# ------------------------------------------------------------ stage-2 objects

def test_catalogue_models_and_their_fields():
    from planetmodel import PREM, gauss_legendre, sample
    m = PREM()
    testing.check_model(m)
    for layer in m.layers:
        for f in layer.fields.values():
            testing.check_field(f)
    s = sample(m, gauss_legendre(4), ngll=3, drmax=1.5e6)
    testing.check_sample(s, m)


def test_shipped_displacements():
    from planetmodel import flattening, layer_linear
    testing.check_displacement(flattening(0.01, rmax=1.0), SK)
    h = layer_linear(SK, [None, lambda t, p: 0.02 * np.sin(t) ** 2 * np.cos(3.0 * p),
                          None, lambda t, p: 0.01 * np.cos(t)])
    testing.check_displacement(h, SK)
    testing.check_geometry(Geometry(SK, mapping=RadialStretch(h, rmax=1.0)))
