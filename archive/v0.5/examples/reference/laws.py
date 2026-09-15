"""Rheological laws: functions from static fields to frequency-dependent ones.

A law is a registered function that takes a layer's static fields and
constants and returns a frequency-dependent field, built by
composition.  The field it returns carries a LawRecord -- the law's
name, the names of the fields it read, its constants and its
convention -- so a file can say how the field was made and a reader
can rebuild it by calling the law again.  Three laws ship: constant Q
(the absorption band, with the moduli holding at a reference period),
Maxwell (unrelaxed moduli and a viscosity; the bulk modulus never
relaxes) and a Prony series (generalised Maxwell).

This script applies each law to a small isotropic body and to PREM,
checks the identities the laws obey, and rebuilds a field from its
record through the registry.
"""
import numpy as np

from planetmodel import (PREM, Dimensions, ElasticField, RadialField,
                         ReferenceBody, Skeleton, Symmetry, constant_q, maxwell,
                         prony)
from planetmodel.model.rheology import law_record_of, rebuild
from planetmodel.registry import registered

# -- an isotropic body: kappa, mu, viscosity on two solid layers -------------
sk = Skeleton([0.0, 1.0e6, 2.0e6])
kappa = RadialField(sk, [lambda r: 1.3e11 + 0.0 * r] * 2, name="kappa",
                    dimensions=Dimensions.MODULUS)
mu = RadialField(sk, [lambda r: 6.0e10 + 0.0 * r, lambda r: 7.0e10 + 0.0 * r],
                 name="mu", dimensions=Dimensions.MODULUS)
eta = RadialField(sk, [lambda r: 1.0e21 + 0.0 * r, lambda r: 3.0e20 + 0.0 * r],
                  name="viscosity", dimensions=Dimensions.VISCOSITY)
moduli = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu},
                      name="elastic_moduli")
r = np.array([0.5e6, 1.5e6])

# -- Maxwell: mu(s) = mu_0 s tau / (1 + s tau), s = i omega, tau = eta / mu_0 --
mx = maxwell(moduli, eta)
assert mx.kind == "frequency" and mx.omega_domain == "complex"
tau = eta.evaluate(r) / mu.evaluate(r)
for omega in (1.0e-12, 1.0e-10):
    s = 1j * omega
    got = mx.evaluate(r, omega=omega)                  # Voigt (2, 6, 6)
    want_mu = mu.evaluate(r) * s * tau / (1.0 + s * tau)
    assert np.allclose(got[:, 3, 3], want_mu, rtol=1e-12)              # shear entry
    assert np.allclose(got[:, 0, 0] - 4.0 / 3.0 * got[:, 3, 3],         # bulk entry
                       kappa.evaluate(r), rtol=1e-12)
    assert np.all(np.imag(got[:, 3, 3]) > 0.0)         # loss for omega > 0
# Off the real axis: the Laplace variable of a time-domain code.
laplace = mx.evaluate(r, omega=-1j * 1.0e-11)          # s = 1e-11, real
assert np.allclose(np.imag(laplace), 0.0)

record = mx.law                                        # the provenance it carries
assert record.law == "maxwell" and record.parameters == ("elastic_moduli", "viscosity")
print("maxwell record:", record)

# -- Prony: one term with no long-time modulus is Maxwell -----------------------
tau_field = RadialField(sk, [lambda r: 1.0e21 / 6.0e10 + 0.0 * r,
                             lambda r: 3.0e20 / 7.0e10 + 0.0 * r],
                        name="tau1", dimensions=Dimensions.TIME)
pr = prony(moduli, [tau_field], [mu])
assert np.allclose(pr.evaluate(r, omega=1.0e-11), mx.evaluate(r, omega=1.0e-11),
                   rtol=1e-12)
assert pr.law.law == "prony" and pr.law.constants == {"terms": 1}

# -- rebuilding through the registry ------------------------------------------
assert "maxwell" in registered("rheology")
fields = {"elastic_moduli": moduli, "viscosity": eta}
again = rebuild(law_record_of(mx), fields)
assert np.array_equal(again.evaluate(r, omega=1.0e-11), mx.evaluate(r, omega=1.0e-11))

# -- constant Q on PREM: the moduli hold at the reference period -------------
prem = PREM(ocean=False)
lay = prem.layers[5]                                    # a mantle layer
cq = constant_q(lay["elastic_moduli"], lay["qkappa"], lay["qmu"],
                reference_period=1.0)
assert cq.omega_domain == "real"
rr = np.array([5.72e6, 5.76e6])
at_ref = cq.evaluate(rr, omega=2.0 * np.pi / 1.0)
static = lay["elastic_moduli"].evaluate(rr)
assert np.allclose(np.real(at_ref), static, rtol=1e-14)   # M_0 at omega_0
assert np.allclose(np.imag(at_ref)[:, 3, 3], static[:, 3, 3] / lay["qmu"](rr))
slow = cq.evaluate(rr, omega=2.0 * np.pi / 100.0)          # 100 s: softer
assert np.all(np.real(slow)[:, 3, 3] < static[:, 3, 3])
print("PREM mantle mu at 1 s and 100 s:", static[0, 3, 3], np.real(slow)[0, 3, 3])
assert prem.layers[5]["viscoelastic_moduli"].law.constants == {"reference_period": 1.0}

# -- rescaling converts a law's constants with their dimensions ----------------
body = ReferenceBody.from_fields(sk, {"kappa": kappa, "mu": mu, "viscosity": eta,
                                      "elastic_moduli": moduli})
body.add_field("viscoelastic_moduli", mx)
nd = body.nondimensionalised()
scaled = nd.layers[0]["viscoelastic_moduli"]
assert scaled.law.law == "maxwell"
# The same physics in the new units: s tau is dimensionless, so the
# shear factor at a matching frequency agrees.
omega_si = 1.0e-11
omega_nd = omega_si * nd.scales.time
r0 = r[:1]                                              # a radius in layer 0
factor_si = mx.evaluate(r0, omega=omega_si)[:, 3, 3] / mu.evaluate(r0)
factor_nd = (scaled.evaluate(r0 / nd.scales.length, omega=omega_nd)[:, 3, 3]
             / nd.layers[0]["mu"].evaluate(r0 / nd.scales.length))
assert np.allclose(factor_nd, factor_si, rtol=1e-10)

print("ok: the laws obey their formulas, carry their records, and rebuild by name")
