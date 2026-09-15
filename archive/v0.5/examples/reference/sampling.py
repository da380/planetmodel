"""Sampling a body on radial times angular nodes.

Export is sampling at a consumer's degrees of freedom.  A RadialMesh
is a spectral-element mesh in radius honouring the skeleton, with
Gauss-Lobatto-Legendre nodes in each element; an AngularGrid is a set
of colatitudes and longitudes, Gauss-Legendre in colatitude for a
spectral consumer or equiangular.  `body.sample(grid)` evaluates every
static field on the product of the two and returns a Sample: one array
per field, with a trailing component shape for tensors, NaN on the
nodes of any element outside the field's domain, and metadata saying
what each array is.

This script samples a small body with a gap, inspects the arrays and
their metadata, and runs the sample's contract check.
"""
import numpy as np

from planetmodel import (DENSITY, AngularGrid, Dimensions, RadialField,
                         RadialMesh, ReferenceBody, Skeleton, at_frequency,
                         testing)
from planetmodel.model.fields.frequency import ComposedFrequencyField

sk = Skeleton([0.0, 1.0e6, 2.0e6, 3.0e6])
rho = RadialField(sk, [lambda r: 5.0e3 + 0.0 * r, None, lambda r: 3.0e3 + 0.0 * r],
                  name="rho", character=DENSITY, dimensions=Dimensions.DENSITY)
vs = RadialField(sk, [lambda r: 3.0e3 + 0.0 * r] * 3, name="vs",
                 dimensions=Dimensions.VELOCITY)
body = ReferenceBody.from_fields(sk, {"rho": rho, "vs": vs})

# -- the grids -------------------------------------------------------------------
grid = AngularGrid.gauss_legendre(lmax=4)          # enough for degree 4 exactly
assert grid.kind == "gauss_legendre" and grid.weights is not None
square = AngularGrid.equiangular(6, 12)
assert square.ntheta == 6 and square.nphi == 12

radial = RadialMesh(body, ngll=5, drmax=0.5e6)     # elements no longer than 500 km
assert radial.ngll == 5 and radial.nspec == 6      # two per layer
assert np.isclose(radial.rglob[-1], 3.0e6)
print("radial mesh:", radial.nspec, "elements,", radial.nglob, "distinct nodes")

# -- a sample ----------------------------------------------------------------------
sample = body.sample(grid, radial=radial)
assert set(sample.fields) == {"rho", "vs"}
assert sample.fields["rho"].shape == (sample.nnode,)      # radial: no angular axes
assert sample.nnode == radial.r.size                     # element by element
assert sample.metadata.domains["rho"] == (0, 2)
assert sample.metadata.domains["vs"] == (0, 1, 2)
assert sample.metadata.characters["rho"] == DENSITY
assert sample.metadata.dimensions["vs"] == Dimensions.VELOCITY

# The gap is carried as NaN on the middle layer's nodes, never filled.
inside_gap = (sample.radius > 1.0e6) & (sample.radius < 2.0e6)
assert np.all(np.isnan(sample.fields["rho"][inside_gap]))
assert not np.any(np.isnan(sample.fields["vs"]))
assert np.allclose(sample.fields["rho"][sample.radius < 1.0e6], 5.0e3)

# A band-limited grid sizes the radial mesh itself when none is given.
auto = body.sample(grid)
assert auto.radial.nspec >= 3

# -- a frequency-dependent field is sampled frozen at one omega ---------------------
kv = ComposedFrequencyField(lambda omega, v: v * (1.0 + 1j * omega), [vs],
                            character=vs.character, dimensions=vs.dimensions,
                            name="vs_complex")
body.add_field("vs_complex", kv)
static_only = body.sample(grid, radial=radial)
assert "vs_complex" not in static_only.fields             # skipped by default
chosen = body.sample(grid, radial=radial, omega=2.0)
assert chosen.metadata.omegas == {"vs_complex": 2.0}
assert chosen.fields["vs_complex"].shape == (sample.nnode, 2)   # (real, imaginary)
assert np.allclose(chosen.fields["vs_complex"][:, 1], 2.0 * chosen.fields["vs"])
frozen = at_frequency(kv, 2.0)
assert np.allclose(frozen.evaluate(sample.radius).real,
                   chosen.fields["vs_complex"][:, 0])

# -- the contract ------------------------------------------------------------------
testing.check_sample(sample)
testing.check_sample(chosen)
print("ok: samples carry every field's domain, and holes are NaN, never filled")
