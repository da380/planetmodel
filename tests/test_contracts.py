"""Every shipped implementation passes its contract."""
import numpy as np
import pytest

from planetmodel import (CallableDisplacement, Geometry, IdentityMapping,
                         RadialStretch, ScaledMapping, Skeleton, ZeroDisplacement,
                         testing, validity_lattice)
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
