"""Dimensions, scales, non-dimensionalisation, and the field vocabulary.

Every field may declare its physical dimensions as exponents of mass,
length and time.  A body carries Scales saying what its numbers are:
SI by default, or a length, mass and time that make them
dimensionless.  `nondimensionalised()` converts every field exactly
(a polynomial layer function converts coefficient by coefficient) and
`redimensionalised()` brings it back; a field that declares no
dimensions refuses to convert rather than keeping numbers in the wrong
units.  The vocabulary is the table of canonical field names the
library reads, writes and guarantees, each with its character and
dimensions; a user's own names are free.

This script shows the arithmetic of scales on PREM and reads the
vocabulary.
"""
import numpy as np

from planetmodel import PREM, Dimensions, RadialField, ReferenceBody, Scales, Skeleton
from planetmodel.model import vocabulary
from planetmodel.model.units import EARTH_MEAN_DENSITY, unit_string

# -- dimensions ----------------------------------------------------------------
rho_dims = Dimensions(mass=1, length=-3)
assert rho_dims == Dimensions.DENSITY
assert Dimensions.MODULUS == Dimensions(mass=1, length=-1, time=-2)
assert Dimensions.VISCOSITY == Dimensions(mass=1, length=-1, time=-1)
print("unit strings:", Dimensions.DENSITY.unit_string(),
      Dimensions.MODULUS.unit_string(), unit_string(None))

# -- scales ----------------------------------------------------------------------
si = Scales.SI
assert si.is_si and si.factor(Dimensions.DENSITY) == 1.0
geo = Scales.geophysical(6371.0e3)                   # length, Earth's mean density, G
assert np.isclose(geo.length, 6371.0e3)
assert np.isclose(geo.density, EARTH_MEAN_DENSITY)
assert np.isclose(geo.gravitational_constant, 1.0)   # G = 1 in these units
print("one scaled modulus is", geo.modulus, "Pa; one scaled second is", geo.time, "s")

# -- a body converts exactly and comes back ------------------------------------------
prem = PREM(ocean=False)
nd = prem.nondimensionalised()
assert not nd.scales.is_si
assert np.isclose(nd.scales.length, prem.skeleton.boundaries[-1])
assert np.isclose(nd.skeleton.boundaries[-1], 1.0)
r_si = np.array([3.0e6, 5.0e6])
r_nd = r_si / nd.scales.length
assert np.allclose(nd["rho"].evaluate(r_nd) * nd.scales.density,
                   prem["rho"].evaluate(r_si))
assert np.allclose(nd["vsv"].evaluate(r_nd) * nd.scales.velocity,
                   prem["vsv"].evaluate(r_si))
back = nd.redimensionalised()
assert back.scales.is_si
assert np.allclose(back["rho"].evaluate(r_si), prem["rho"].evaluate(r_si), rtol=1e-14)

# The class and the laws come through: the reference period is now in
# scaled seconds.
assert type(nd).__name__ == "ViscoelasticModel"
period = nd.layers[3]["viscoelastic_moduli"].law.constants["reference_period"]
assert np.isclose(period * nd.scales.time, 1.0)

# -- a field without dimensions refuses to convert ----------------------------------
sk = Skeleton([0.0, 1.0e6])
anonymous = RadialField(sk, [lambda r: 1.0 + 0.0 * r], name="anonymous")
body = ReferenceBody.from_fields(sk, {"anonymous": anonymous})
try:
    body.nondimensionalised()
except ValueError as exc:
    print("refused as expected:", exc)
else:
    raise AssertionError("a field with no dimensions must not be rescaled silently")

# -- the vocabulary -----------------------------------------------------------------
assert "rho" in vocabulary.names() and "elastic_moduli" in vocabulary.names()
assert vocabulary.dimensions_of("rho") == Dimensions.DENSITY
assert vocabulary.character_of("rho").weight == 1          # a density is weight 1
assert vocabulary.character_of("vsv").weight == 0
assert vocabulary.character_of("elastic_moduli").rank == 4
for name in ("rho", "vsv", "qmu", "elastic_moduli", "viscoelastic_moduli"):
    print(f"  {name:22s} {vocabulary.describe(name)}")

print("ok: scales convert exactly and refuse the undeclared; one vocabulary table")
