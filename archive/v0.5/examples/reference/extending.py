"""Extending the library: a field type, a law, a model class, a mapping.

The framework is a set of small contracts.  A field is anything with a
skeleton, a character, dimensions, a name and `evaluate`; subclassing
FieldBase adds the algebra, restriction and Cartesian evaluation, and
a `rescaled` method lets a body convert it between unit systems.  A
law is a registered function that returns a frequency-dependent field
carrying a LawRecord, with a `from_record` attribute so a file can
rebuild it.  A model class is a mixin of aspects registered under
"model_class".  A mapping subclasses MappingBase and supplies the map,
its deformation gradient and its Jacobian.  Each contract has an
executable check in `planetmodel.testing`.

This script writes one of each outside the library and runs it through
the checks, a body, and the netCDF file.
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np

from planetmodel import (DENSITY, SCALAR, Dimensions, FieldBase, LawRecord,
                         RadialField, ReferenceBody, Skeleton, ViscoelasticModel,
                         register, testing)
from planetmodel.model.classes import HasDensity, ModelBase
from planetmodel.model.fields.frequency import ComposedFrequencyField
from planetmodel.model.mapping import MappingBase
from planetmodel.model.rheology import law_record_of, rebuild

sk = Skeleton([0.0, 1.0e6, 2.0e6])
rho = RadialField(sk, [lambda r: 5.0e3 + 0.0 * r] * 2, name="rho",
                  character=DENSITY, dimensions=Dimensions.DENSITY)
mu = RadialField(sk, [lambda r: 6.0e10 + 0.0 * r] * 2, name="mu",
                 dimensions=Dimensions.MODULUS)


# -- 1. a field type: values tabulated in radius, on one layer ------------------
class TabulatedField(FieldBase):
    """Linear interpolation through (radius, value) pairs on one layer."""

    def __init__(self, skeleton, radii, values, *, dimensions=None, name=None):
        self.skeleton = skeleton
        self.character = SCALAR
        self.dimensions = dimensions
        self.name = name
        self.radii = np.asarray(radii, float)
        self.values = np.asarray(values, float)

    def evaluate(self, r, theta=None, phi=None, *, layer=None, side="upper",
                 frame="spherical"):
        r = np.asarray(r, float)
        if theta is not None:                  # broadcast with the angles, ignored
            r = np.broadcast_arrays(r, np.asarray(theta, float),
                                    np.asarray(phi, float))[0]
        lo, hi = self.skeleton.boundaries[0], self.skeleton.boundaries[-1]
        if np.any(r < lo - self.skeleton.tolerance) or \
                np.any(r > hi + self.skeleton.tolerance):
            raise ValueError(f"{self.name!r}: radius outside [{lo}, {hi}]")
        return np.interp(r, self.radii, self.values)

    def rescaled(self, convert, old, new):
        """Radii scale as lengths, values by their own dimensions."""
        k = old.length / new.length
        v = old.factor(self.dimensions) / new.factor(self.dimensions)
        return TabulatedField(Skeleton(self.skeleton.boundaries * k),
                              self.radii * k, self.values * v,
                              dimensions=self.dimensions, name=self.name)


layer1 = Skeleton(sk.interval(1))
porosity = TabulatedField(layer1, np.linspace(1.0e6, 2.0e6, 5),
                          [0.30, 0.25, 0.20, 0.15, 0.10],
                          dimensions=Dimensions.DIMENSIONLESS, name="porosity")
testing.check_field(porosity)                          # the contract, in one call
assert np.isclose((2.0 * porosity)(1.5e6), 0.40)       # the algebra comes for free

body = ReferenceBody.from_fields(sk, {"rho": rho, "mu": mu})
body = body.with_field(1, "porosity", porosity)
assert body["porosity"].domain == (1,)
nd = body.nondimensionalised()                          # rescaled() makes this work
assert np.isclose(nd.skeleton.boundaries[-1], 1.0)       # radii now in outer radii
assert np.isclose(nd["porosity"].evaluate(0.75), 0.20)  # dimensionless: unchanged


# -- 2. a law: a power-law Q, registered with its inverse ------------------------
@register("rheology", "power_law_q")
def power_law_q(modulus, q, *, reference_period, alpha=0.3, name=None):
    """mu(omega) = mu_0 (1 + i / Q (omega / omega_0)^alpha), a scalar law."""
    omega_0 = 2.0 * np.pi / reference_period

    def fn(omega, m, qq):                      # the argument first, then values
        return m * (1.0 + 1j * (omega / omega_0) ** alpha / qq)

    return ComposedFrequencyField(
        fn, [modulus, q], character=SCALAR, dimensions=modulus.dimensions,
        name=name or "mu_complex", omega_domain="real",
        law=LawRecord("power_law_q", parameters=(modulus.name, q.name),
                      constants={"reference_period": reference_period,
                                 "alpha": alpha}))


def _from_record(record, fields):
    modulus, q = (fields[n] for n in record.parameters)
    return power_law_q(modulus, q, **record.constants)


power_law_q.from_record = _from_record
power_law_q.constant_dimensions = {"reference_period": Dimensions.TIME,
                                   "alpha": Dimensions.DIMENSIONLESS}

q = RadialField(sk, [lambda r: 300.0 + 0.0 * r] * 2, name="q",
                dimensions=Dimensions.DIMENSIONLESS)
law = power_law_q(mu, q, reference_period=1.0)
r = np.array([0.5e6, 1.5e6])
testing.check_frequency_dependent_field(law, omegas=[0.1, 1.0, 10.0])
again = rebuild(law_record_of(law), {"mu": mu, "q": q})   # what a reader does
assert np.allclose(again.evaluate(r, omega=3.0), law.evaluate(r, omega=3.0))


# -- 3. a model class ---------------------------------------------------------------
class HasPorosity:
    REQUIRES = ("porosity",)

    @property
    def porosity(self):
        return self["porosity"]


@register("model_class", "PorousModel")
class PorousModel(HasDensity, HasPorosity, ModelBase):
    ASPECTS = (HasDensity, HasPorosity)


full = TabulatedField(sk, np.linspace(0.0, 2.0e6, 5), [0.4, 0.3, 0.2, 0.1, 0.05],
                      dimensions=Dimensions.DIMENSIONLESS, name="porosity")
porous = ReferenceBody.from_fields(sk, {"rho": rho, "porosity": full}).as_class(
    PorousModel)
testing.check_model(porous)
assert type(porous.truncated(1.5e6)) is PorousModel


# -- 4. a mapping: a uniform squash along z ---------------------------------------
class Squash(MappingBase):
    """x -> (x, y, c z): affine, so its gradient is constant."""

    knots = ()

    def __init__(self, skeleton, c):
        self.skeleton = skeleton
        self.c = float(c)

    def __call__(self, X):
        return np.asarray(X, float) * np.array([1.0, 1.0, self.c])

    def deformation_gradient(self, X, *, frame="cartesian"):
        X = np.asarray(X, float)
        F = np.diag([1.0, 1.0, self.c])
        return np.broadcast_to(F, X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        return np.full(np.asarray(X, float).shape[:-1], self.c)

    def inverse(self, x, **kw):
        return np.asarray(x, float) / np.array([1.0, 1.0, self.c])


squash = Squash(sk, 0.9)
points = np.random.default_rng(0).normal(size=(20, 3)) * 0.5e6 + [0, 0, 1.0e6]
testing.check_mapping(squash, points)
assert np.allclose(squash.inverse(squash(points)), points)


# -- 5. all of it through a file ------------------------------------------------------
try:
    import netCDF4  # noqa: F401
except ImportError:
    print("netCDF4 is not installed; the round trip is skipped")
else:
    from planetmodel import AngularGrid, ElasticField, Symmetry, read_model, write_model
    kappa = RadialField(sk, [lambda r: 1.3e11 + 0.0 * r] * 2, name="kappa",
                        dimensions=Dimensions.MODULUS)
    el = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu},
                      name="elastic_moduli")
    # The file restores an elastic tensor from its named component fields,
    # so kappa and mu travel beside it.
    visco = (ReferenceBody.from_fields(sk, {"rho": rho, "kappa": kappa, "mu": mu,
                                            "q": q, "elastic_moduli": el})
             .with_field(1, "mu_complex", law.restricted(1))
             .as_class(ViscoelasticModel))
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "user.nc"
        write_model(visco, visco.sample(AngularGrid.gauss_legendre(2)), path)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            back, _ = read_model(path)
        assert type(back) is ViscoelasticModel
        assert back.layers[1]["mu_complex"].law.law == "power_law_q"   # rebuilt by name
        assert np.allclose(back["mu_complex"].evaluate(np.array([1.5e6]), omega=3.0),
                           law.evaluate(np.array([1.5e6]), omega=3.0), rtol=1e-10)
        print("read back:", type(back).__name__, "with the user law rebuilt")

print("ok: a field type, a law, a class and a mapping written outside the library work")
