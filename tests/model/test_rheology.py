"""Laws: constant Q, Maxwell and the Prony series, and their provenance.

Every oracle is the Appendix B formula written out pointwise on the
parameter fields' values, independently of `rheology.py`.
"""
import numpy as np
import pytest

from planetmodel import PREM, RadialField, ReferenceBody, Skeleton, bond_matrix
from planetmodel.io.deck import read_isotropic_deck
from planetmodel.model.character import ELASTIC, SCALAR
from planetmodel.model.fields.frequency import (ComposedFrequencyField,
                                                lifted_to_frequency)
from planetmodel.model.frames import spherical_frame
from planetmodel.model.materials import ElasticField, Symmetry
from planetmodel.model.rheology import (STATIC, LawRecord, constant_dimensions_of,
                                        constant_q, constant_q_scalar,
                                        law_record_of, maxwell, prony, rebuild)
from planetmodel.model.units import Dimensions
from planetmodel.registry import lookup, register
from planetmodel.testing import check_frequency_dependent_field, check_law

T0 = 1.0
OMEGA0 = 2.0 * np.pi / T0
OMEGAS = (0.02, OMEGA0, 25.0)


def factor(omega):
    return (2.0 / np.pi) * np.log(omega / OMEGA0) + 1j


def band(omega, M0, Q):
    """B.9.1 for one modulus; Q = 0 attenuates nothing."""
    M0, Q = np.asarray(M0, dtype=float), np.asarray(Q, dtype=float)
    out = M0.astype(complex)
    live = Q != 0.0
    out[live] = M0[live] * (1.0 + factor(omega) / Q[live])
    return out


def voigt_vti(A, C, F, L, N):
    out = np.zeros(np.shape(A) + (6, 6), dtype=complex)
    out[..., 0, 0] = C
    out[..., 1, 1] = out[..., 2, 2] = A
    out[..., 0, 1] = out[..., 1, 0] = out[..., 0, 2] = out[..., 2, 0] = F
    out[..., 1, 2] = out[..., 2, 1] = A - 2.0 * N
    out[..., 3, 3] = N
    out[..., 4, 4] = out[..., 5, 5] = L
    return out


def voigt_iso(kappa, mu):
    lam = kappa - 2.0 * mu / 3.0
    out = np.zeros(np.shape(kappa) + (6, 6), dtype=complex)
    for i in range(3):
        for j in range(3):
            out[..., i, j] = lam
        out[..., i, i] = lam + 2.0 * mu
        out[..., 3 + i, 3 + i] = mu
    return out


@pytest.fixture(scope="module")
def deck():
    """prem.nocrust: isotropic, with a fluid outer core where mu = Q_mu = 0."""
    return read_isotropic_deck("tests/data/prem.nocrust")


@pytest.fixture(scope="module")
def law(deck):
    return constant_q(deck["elastic_moduli"], deck["qkappa"], deck["qmu"],
                      reference_period=T0)


def test_constant_q_is_its_formula(deck, law):
    kappa, mu = deck["kappa"], deck["mu"]
    qk, qm = deck["qkappa"], deck["qmu"]

    def oracle(omega, r, theta, phi):
        return voigt_iso(band(omega, kappa.evaluate(r), qk.evaluate(r)),
                         band(omega, mu.evaluate(r), qm.evaluate(r)))

    assert law.kind == "frequency" and law.omega_domain == "real"
    assert law.character == ELASTIC and law.dimensions == Dimensions.MODULUS
    check_law(law, omegas=OMEGAS, oracle=oracle)


def test_at_the_reference_period_the_real_part_is_the_static_moduli(deck, law):
    r = np.linspace(1.0e5, 6.3e6, 400)
    static = deck["elastic_moduli"].evaluate(r)
    dynamic = law.evaluate(r, omega=OMEGA0)
    assert np.allclose(np.real(dynamic), static, rtol=1e-14,
                       atol=1e-14 * np.max(static))
    # ... and the imaginary part is M0 / Q, modulus by modulus.
    mu, qm = deck["mu"].evaluate(r), deck["qmu"].evaluate(r)
    live = mu != 0.0
    assert np.allclose(np.imag(dynamic[..., 3, 3])[live], (mu / qm)[live],
                       rtol=1e-14)


def test_the_fluid_core_evaluates_with_a_zero_shear_modulus(deck, law):
    core = [i for i, lay in enumerate(deck.layers)
            if float(deck["mu"][i](0.5 * sum(lay.interval))) == 0.0]
    assert core, "prem.nocrust has a fluid outer core"
    lo, hi = deck.layer(core[0]).interval
    r = np.array([0.5 * (lo + hi)])
    v = law.evaluate(r, omega=3.0)
    assert v[0, 3, 3] == 0.0 and np.isfinite(v).all()
    assert np.imag(v[0, 0, 0]) > 0.0                       # bulk loss survives


def test_a_zero_q_leaves_that_modulus_undispersed():
    """A zero Q contributes nothing: that modulus is undispersed."""
    sk = Skeleton([0.0, 1.0])
    kappa = RadialField(sk, [lambda r: 2.0 + 0 * r], name="kappa",
                        dimensions=Dimensions.MODULUS)
    mu = RadialField(sk, [lambda r: 1.0 + 0 * r], name="mu",
                     dimensions=Dimensions.MODULUS)
    el = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu}, name="el")
    qk = RadialField(sk, [lambda r: 100.0 + 0 * r], name="qkappa")
    q0 = RadialField(sk, [lambda r: 0.0 * r], name="qmu")
    f = constant_q(el, qk, q0, reference_period=T0)
    v = f.evaluate(0.5, omega=3.0)
    assert v[3, 3] == 1.0 + 0.0j                       # mu untouched
    assert np.imag(v[0, 0]) == pytest.approx(2.0 / 100.0)   # kappa attenuated
    with pytest.raises(ValueError, match="omega > 0"):
        f.evaluate(0.5, omega=-1.0)
    with pytest.raises(ValueError, match="omega must be real"):
        f.evaluate(0.5, omega=1.0 + 1.0j)
    with pytest.raises(ValueError, match="reference_period"):
        constant_q(el, qk, q0, reference_period=0.0)


def test_a_real_omega_given_as_a_complex_number_is_accepted(deck, law):
    """A real-axis law takes complex(3, 0) as the real number 3."""
    r = np.array([4.0e6])
    assert np.array_equal(law.evaluate(r, omega=complex(3.0, 0.0)),
                          law.evaluate(r, omega=3.0))


def test_vti_constant_q_is_voigt_averages_formula():
    """B.9.1 for A, C, F, L, N: the band on the Voigt-average kappa and mu,
    the anisotropic residual undispersed."""
    prem = PREM()
    law = constant_q(prem["elastic_moduli"], prem["qkappa"], prem["qmu"],
                     reference_period=prem.meta["tref"])
    assert law.symmetry is Symmetry.VTI and law.law.convention == "voigt_average"
    mod = {k: prem[k] for k in ("A", "C", "F", "L", "N")}
    qk, qm = prem["qkappa"], prem["qmu"]

    def oracle(omega, r, theta, phi):
        A, C, F, L, N = (mod[k].evaluate(r) for k in ("A", "C", "F", "L", "N"))
        kappa = (C + 4.0 * (A - N + F)) / 9.0
        mu = (C + A + 6.0 * L + 5.0 * N - 2.0 * F) / 15.0
        rk = band(omega, kappa, qk.evaluate(r)) - kappa      # f kappa / Qkappa
        rm = band(omega, mu, qm.evaluate(r)) - mu
        return voigt_vti(A + rk + (4.0 / 3.0) * rm, C + rk + (4.0 / 3.0) * rm,
                         F + rk - (2.0 / 3.0) * rm, L + rm, N + rm)

    check_law(law, omegas=OMEGAS, oracle=oracle)
    r = np.linspace(1.0e5, 6.3e6, 300)
    assert np.allclose(np.real(law.evaluate(r, omega=OMEGA0)),
                       prem["elastic_moduli"].evaluate(r), rtol=1e-14)
    # Cartesian components are the Bond rotation, as for the static tensor
    # (check_field on the frozen field, inside check_law, pins this too).
    th, ph = 0.7, 0.4
    M = bond_matrix(spherical_frame(np.array(th), np.array(ph)))
    sph = law.evaluate(4.0e6, th, ph, omega=3.0)
    cart = law.evaluate(4.0e6, th, ph, omega=3.0, frame="cartesian")
    assert np.allclose(cart, M @ sph @ M.T)


def test_the_vti_law_on_an_isotropic_medium_is_the_isotropic_law(deck, law):
    """The isotropic limit of the voigt_average convention is the one-modulus
    formula on kappa and mu: the oracle B.9.1 names."""
    vti = deck["elastic_moduli"].as_symmetry(Symmetry.VTI)
    wide = constant_q(vti, deck["qkappa"], deck["qmu"], reference_period=T0)
    r = np.linspace(1.0e5, 6.3e6, 300)
    for omega in OMEGAS:
        a, b = wide.evaluate(r, omega=omega), law.evaluate(r, omega=omega)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12 * np.max(np.abs(b)))


def test_provenance_is_a_registered_law_record(law, deck):
    record = law.law
    assert isinstance(record, LawRecord)
    assert record.law == "constant_q" and record.convention == "voigt_average"
    assert record.parameters == ("elastic_moduli", "qkappa", "qmu")
    assert record.constants == {"reference_period": T0}
    assert lookup("rheology", "constant_q") is constant_q
    # carried through restriction and reassembly by a body
    body = ReferenceBody.from_fields(deck.skeleton, {n: deck[n] for n in
                                                     ("kappa", "mu", "qkappa",
                                                      "qmu", "elastic_moduli")})
    body.add_field("dyn", law)
    piece = body.layer(3)["dyn"]
    assert piece.law == record
    view = body["dyn"]
    assert view.law == record and view.omega_domain == "real"
    r = np.array([4.0e6])
    assert np.allclose(view.evaluate(r, omega=3.0), law.evaluate(r, omega=3.0))


def test_a_record_restates_its_parameters_and_knows_its_dimensions(law):
    record = law.law
    renamed = record.with_parameters(("moduli", "Qk", "Qm"))
    assert renamed.parameters == ("moduli", "Qk", "Qm")
    assert renamed.constants == record.constants and renamed.law == record.law
    with pytest.raises(ValueError, match="read 3 fields"):
        record.with_parameters(("moduli",))
    assert constant_dimensions_of(record) == {"reference_period": Dimensions.TIME}
    assert constant_dimensions_of(LawRecord(STATIC, parameters=("el",))) == {}
    assert constant_dimensions_of(LawRecord("prony", constants={"terms": 2})) \
        == {"terms": Dimensions.DIMENSIONLESS}


def test_one_modulus_law(deck):
    f = constant_q_scalar(deck["mu"], deck["qmu"], reference_period=T0)
    assert f.character == SCALAR and f.law.law == "constant_q_scalar"
    assert f.law.parameters == ("mu", "qmu")

    def oracle(omega, r, theta, phi):
        return band(omega, deck["mu"].evaluate(r), deck["qmu"].evaluate(r))

    check_law(f, omegas=OMEGAS, oracle=oracle)
    check_frequency_dependent_field(f, omegas=OMEGAS)


def test_a_law_takes_static_operands_only(deck):
    inner = ComposedFrequencyField(lambda omega, m: m * (1.0 + 1j / omega),
                                   [deck["mu"]], character=SCALAR)
    with pytest.raises(TypeError, match="static fields"):
        constant_q_scalar(inner, deck["qmu"], reference_period=T0)
    # the lift of a static field is that static field, as restriction
    # hands operands back lifted
    f = constant_q_scalar(lifted_to_frequency(deck["mu"]), deck["qmu"],
                          reference_period=T0)
    r = np.array([4.0e6])
    assert np.array_equal(f.evaluate(r, omega=3.0),
                          constant_q_scalar(deck["mu"], deck["qmu"],
                                            reference_period=T0)
                          .evaluate(r, omega=3.0))


def test_a_lifted_elastic_layer_beside_a_constant_q_one(deck, law):
    """The composite the model classes check: static where lifted."""
    layers = []
    for i, lay in enumerate(deck.layers):
        dyn = (lifted_to_frequency(lay["elastic_moduli"]) if i == 0
               else law.restricted(i))
        layers.append(lay.with_field("dyn", dyn))
    body = ReferenceBody.from_layers(layers)
    view = body["dyn"]
    assert view.kind == "frequency" and view.omega_domain == "real"
    r0 = np.array([0.5 * deck.layer(0).interval[1]])
    assert np.all(np.imag(view.evaluate(r0, omega=3.0)) == 0.0)
    r3 = np.array([0.5 * sum(deck.layer(3).interval)])
    assert np.allclose(view.evaluate(r3, omega=3.0), law.evaluate(r3, omega=3.0))
    check_frequency_dependent_field(view, omegas=OMEGAS)


# ------------------------------------------------ B.9.2 Maxwell, B.9.3 Prony

def iso_medium():
    """A three-layer isotropic body with a viscosity and a fluid middle."""
    sk = Skeleton([0.0, 1.0, 2.0, 3.0])
    kappa = RadialField(sk, [lambda r: 3.0e10 + 0 * r] * 3, name="kappa",
                        dimensions=Dimensions.MODULUS)
    mu = RadialField(sk, [lambda r: 6.0e10 + 1e9 * r, lambda r: 0.0 * r,
                          lambda r: 7.0e10 + 0 * r], name="mu",
                     dimensions=Dimensions.MODULUS)
    eta = RadialField(sk, [lambda r: 1.0e21 + 0 * r, lambda r: 0.0 * r,
                           lambda r: 3.0e20 + 0 * r], name="viscosity",
                      dimensions=Dimensions.VISCOSITY)
    el = ElasticField(Symmetry.ISOTROPIC, {"kappa": kappa, "mu": mu}, name="el")
    return sk, kappa, mu, eta, el, maxwell(el, eta)


SMALL = (1.0e-13, 3.0e-12, 1.0e-10)    # rad/s: the Maxwell time is ~1e10-1e11 s


def test_maxwell_is_its_formula_on_and_off_the_real_axis():
    sk, kappa, mu, eta, el, law = iso_medium()
    assert law.omega_domain == "complex" and law.law.law == "maxwell"

    def oracle(omega, r, theta, phi):
        s = 1j * omega
        m, e = mu.evaluate(r), eta.evaluate(r)
        tau = np.where(m != 0, e / np.where(m != 0, m, 1.0), 0.0)
        mus = np.where(m != 0, m * s * tau / (1.0 + s * tau), 0.0)
        return voigt_iso(kappa.evaluate(r).astype(complex), mus)

    check_law(law, omegas=SMALL, oracle=oracle)
    r = np.array([0.5, 2.5])
    for s in (2.0e-11, 5.0e-12 + 3.0e-12j):        # Laplace variable, s = i omega
        omega = -1j * s
        got = law.evaluate(r, omega=omega)[..., 3, 3]
        m, e = mu.evaluate(r), eta.evaluate(r)
        assert np.allclose(got, m * s * (e / m) / (1.0 + s * e / m))


def test_maxwell_against_the_laplace_transform_of_the_relaxation_function():
    """B.9.2's oracle: s times the transform of mu_0 exp(-t / tau)."""
    from scipy.integrate import quad
    sk, kappa, mu, eta, el, law = iso_medium()
    r = 2.5
    m, e = float(mu.evaluate(r)), float(eta.evaluate(r))
    tau = e / m
    for s in (0.7 / tau, 3.0 / tau, (0.5 + 2.0j) / tau):
        f = lambda t: m * np.exp(-t / tau) * np.exp(-s * t)  # noqa: E731
        re = quad(lambda t: np.real(f(t)), 0.0, 60.0 * tau, limit=400)[0]
        im = quad(lambda t: np.imag(f(t)), 0.0, 60.0 * tau, limit=400)[0]
        want = s * (re + 1j * im)
        got = law.evaluate(r, omega=-1j * s)[3, 3]
        assert abs(got - want) <= 1e-8 * abs(want)


def test_maxwell_limits_and_the_fluid_layer():
    sk, kappa, mu, eta, el, law = iso_medium()
    r = np.array([0.5, 1.5, 2.5])
    tau = eta.evaluate(r) / np.where(mu.evaluate(r) != 0, mu.evaluate(r), 1.0)
    fast = law.evaluate(r, omega=1.0e6 / np.max(tau))     # s -> inf: unrelaxed
    assert np.allclose(fast[..., 3, 3], mu.evaluate(r), rtol=1e-5)
    slow = law.evaluate(r, omega=1.0e-6 / np.max(tau))    # s -> 0: relaxed away
    assert np.all(np.abs(slow[..., 3, 3]) <= 1e-5 * np.max(mu.evaluate(r)))
    assert np.allclose(np.real(fast[..., 0, 0] - 4.0 / 3.0 * fast[..., 3, 3]),
                       kappa.evaluate(r))                # kappa unrelaxed
    assert law.evaluate(1.5, omega=3.0e-12)[3, 3] == 0.0   # fluid: mu = 0
    check_frequency_dependent_field(law, omegas=SMALL)


def test_the_relaxing_laws_are_lossy_for_positive_omega():
    """The time convention: exp(+i omega t), so Im mu > 0 where mu relaxes."""
    sk, kappa, mu, eta, el, law = iso_medium()
    tau = RadialField(sk, [lambda r: 1.0e10 + 0 * r] * 3, name="tau_1",
                      dimensions=Dimensions.TIME)
    series = prony(el, [tau], [mu])
    r = np.array([0.5, 2.5])
    for omega in SMALL:
        assert np.all(np.imag(law.evaluate(r, omega=omega)[..., 3, 3]) > 0.0)
        assert np.all(np.imag(series.evaluate(r, omega=omega)[..., 3, 3]) > 0.0)


def test_prony_one_term_is_maxwell_and_its_limits():
    sk, kappa, mu, eta, el, law = iso_medium()
    tau = RadialField(sk, [lambda r: 1.0e21 / (6.0e10 + 1e9 * r),
                           lambda r: 1.0 + 0 * r, lambda r: 3.0e20 / 7.0e10 + 0 * r],
                      name="tau_1", dimensions=Dimensions.TIME)
    series = prony(el, [tau], [mu])
    assert series.law.law == "prony" and series.law.constants == {"terms": 1.0}
    assert series.law.parameters == ("el", "tau_1", "mu")
    r = np.array([0.5, 2.5])
    for omega in SMALL + (2.0e-12 + 1.0e-12j,):
        assert np.allclose(series.evaluate(r, omega=omega),
                           law.evaluate(r, omega=omega), rtol=1e-14)
    # two terms with a long-time modulus: the two limits of B.9.3
    tau2 = RadialField(sk, [lambda r: 1.0e9 + 0 * r] * 3, name="tau_2",
                       dimensions=Dimensions.TIME)
    M2 = RadialField(sk, [lambda r: 2.0e10 + 0 * r] * 3, name="M_2",
                     dimensions=Dimensions.MODULUS)
    Minf = RadialField(sk, [lambda r: 5.0e9 + 0 * r] * 3, name="M_inf",
                       dimensions=Dimensions.MODULUS)
    two = prony(el, [tau, tau2], [mu, M2], long_time_modulus=Minf)
    assert two.law.parameters == ("el", "tau_1", "tau_2", "mu", "M_2", "M_inf")
    hi = two.evaluate(r, omega=1.0e3)[..., 3, 3]
    assert np.allclose(hi, 5.0e9 + mu.evaluate(r) + 2.0e10, rtol=1e-6)
    lo = two.evaluate(r, omega=1.0e-25)[..., 3, 3]
    assert np.allclose(lo, 5.0e9, rtol=1e-6)
    check_frequency_dependent_field(two, omegas=(1.0e-12, 1.0e-9, 1.0e-6))

    def oracle(omega, r, theta, phi):
        s = 1j * omega
        t1, t2 = tau.evaluate(r), tau2.evaluate(r)
        m = (Minf.evaluate(r) + mu.evaluate(r) * s * t1 / (1 + s * t1)
             + M2.evaluate(r) * s * t2 / (1 + s * t2))
        return voigt_iso(kappa.evaluate(r).astype(complex), m)

    check_law(two, omegas=(1.0e-12, 1.0e-9, 1.0e-6), oracle=oracle)
    with pytest.raises(ValueError, match="equally many"):
        prony(el, [tau], [])


def test_vti_maxwell_relaxes_everything_but_the_bulk_part():
    """C(s) = K I(x)I + g (C - K I(x)I) with one tau = eta / mu, K and mu the
    Voigt averages: the deviatoric stress relaxes completely and the bulk
    modulus never moves."""
    prem = PREM()
    eta = RadialField(prem.skeleton,
                      [lambda r: 1.0e21 + 0 * r] * prem.skeleton.nlayers,
                      name="viscosity", dimensions=Dimensions.VISCOSITY)
    law = maxwell(prem["elastic_moduli"], eta)
    assert law.symmetry is Symmetry.VTI and law.omega_domain == "complex"
    mod = {k: prem[k] for k in ("A", "C", "F", "L", "N")}

    def averages(r):
        A, C, F, L, N = (mod[k].evaluate(r) for k in ("A", "C", "F", "L", "N"))
        K = (C + 4.0 * (A - N + F)) / 9.0
        mu = (C + A + 6.0 * L + 5.0 * N - 2.0 * F) / 15.0
        return A, C, F, L, N, K, mu

    def oracle(omega, r, theta, phi):
        A, C, F, L, N, K, mu = averages(r)
        tau = np.where(mu != 0, 1.0e21 / np.where(mu != 0, mu, 1.0), 0.0)
        s = 1j * omega
        g = np.where(tau != 0, s * tau / (1.0 + s * tau), 0.0)
        return voigt_vti(K + g * (A - K), K + g * (C - K), K + g * (F - K),
                         g * L, g * N)

    check_law(law, omegas=SMALL, oracle=oracle)

    # The property: left alone (s -> 0) any strain gives pure pressure.
    r = np.array([4.0e6, 6.0e6, 6.35e6])
    relaxed = law.evaluate(r, omega=1.0e-30)
    A, C, F, L, N, K, mu = averages(r)
    rng = np.random.default_rng(3)
    strain = rng.normal(size=(3, 6)) * np.array([1, 1, 1, 2, 2, 2])  # Voigt
    stress = np.einsum("nij,nj->ni", relaxed, strain)
    p = stress[:, :3].mean(axis=1)
    assert np.allclose(stress[:, :3], p[:, None], rtol=1e-6)      # isotropic
    assert np.allclose(stress[:, 3:], 0.0, atol=1e-6 * np.abs(p).max())
    assert np.allclose(np.real(relaxed[:, 0, 0]), K, rtol=1e-6)   # K unmoved
    unrelaxed = law.evaluate(r, omega=1.0e30)
    assert np.allclose(np.real(unrelaxed), prem["elastic_moduli"].evaluate(r),
                       rtol=1e-6)

    # a Prony series with one term and no long-time modulus is Maxwell
    mu_eq = (prem["C"] + prem["A"] + 6.0 * prem["L"] + 5.0 * prem["N"]
             - 2.0 * prem["F"]) / 15.0

    def tau_fn(i):
        def fn(r):
            m = np.asarray(mu_eq.evaluate(r, layer=i), dtype=float)
            return np.where(m != 0.0, 1.0e21 / np.where(m != 0.0, m, 1.0), 0.0)
        return fn
    tau = RadialField(prem.skeleton,
                      [tau_fn(i) for i in range(prem.skeleton.nlayers)],
                      name="tau_1", dimensions=Dimensions.TIME)
    one = prony(prem["elastic_moduli"], [tau], [prem["L"]])
    for omega in SMALL:
        assert np.allclose(one.evaluate(r, omega=omega), law.evaluate(r, omega=omega),
                           rtol=1e-12)


def test_the_vti_relaxation_on_an_isotropic_medium_is_the_isotropic_law():
    sk, kappa, mu, eta, el, law = iso_medium()
    wide = maxwell(el.as_symmetry(Symmetry.VTI), eta)
    r = np.array([0.5, 1.5, 2.5])
    for omega in SMALL + (2.0e-12 + 1.0e-12j,):
        a, b = wide.evaluate(r, omega=omega), law.evaluate(r, omega=omega)
        assert np.allclose(a, b, rtol=1e-12, atol=1e-12 * np.max(np.abs(b)))


# ------------------------------------------ a law rebuilt from its provenance

def test_rebuild_is_the_law_again_on_prems_layers():
    """PREM's every layer holds `viscoelastic_moduli` under constant Q, and
    the file carries nothing of it but the `LawRecord`; a consumer with
    the layer's basic fields and that record gets the same field back."""
    prem = PREM()
    for lay in prem.layers:
        piece = lay["viscoelastic_moduli"]
        record = law_record_of(piece)
        assert record.law == "constant_q"
        assert record.parameters == ("elastic_moduli", "qkappa", "qmu")
        again = rebuild(record, lay.fields)
        lo, hi = lay.interval
        r = np.linspace(lo + 0.01 * (hi - lo), hi - 0.01 * (hi - lo), 17)
        for omega in OMEGAS:
            want = piece.evaluate(r, omega=omega)
            got = again.evaluate(r, omega=omega)
            assert np.allclose(got, want, rtol=1e-12,
                               atol=1e-12 * np.max(np.abs(want)))


def test_rebuild_of_maxwell_and_prony_and_of_a_lift():
    sk, kappa, mu, eta, el, law = iso_medium()
    fields = {"el": el, "kappa": kappa, "mu": mu, "viscosity": eta}
    r = np.array([0.5, 1.5, 2.5])

    again = rebuild(law_record_of(law), fields)
    for omega in SMALL + (2.0e-12 + 1.0e-12j,):
        assert np.array_equal(again.evaluate(r, omega=omega),
                              law.evaluate(r, omega=omega))

    tau = RadialField(sk, [lambda r: 1.0e10 + 0 * r] * 3, name="tau_1",
                      dimensions=Dimensions.TIME)
    tau2 = RadialField(sk, [lambda r: 1.0e9 + 0 * r] * 3, name="tau_2",
                       dimensions=Dimensions.TIME)
    M2 = RadialField(sk, [lambda r: 2.0e10 + 0 * r] * 3, name="M_2",
                     dimensions=Dimensions.MODULUS)
    Minf = RadialField(sk, [lambda r: 5.0e9 + 0 * r] * 3, name="M_inf",
                       dimensions=Dimensions.MODULUS)
    fields.update({"tau_1": tau, "tau_2": tau2, "M_2": M2, "M_inf": Minf})
    for n, series in ((1, prony(el, [tau], [mu])),
                      (2, prony(el, [tau, tau2], [mu, M2],
                                long_time_modulus=Minf))):
        record = law_record_of(series)
        assert record.constants == {"terms": float(n)}
        again = rebuild(record, fields)
        for omega in SMALL + (1.0e-9,):
            assert np.array_equal(again.evaluate(r, omega=omega),
                                  series.evaluate(r, omega=omega))

    # a lift: no law at all, recorded by the name of what it lifts
    lifted = lifted_to_frequency(el)
    assert law_record_of(lifted, source_name="el") == LawRecord(STATIC,
                                                                parameters=("el",))
    again = rebuild(law_record_of(lifted, source_name="el"), fields)
    for omega in SMALL:
        assert np.array_equal(again.evaluate(r, omega=omega),
                              lifted.evaluate(r, omega=omega))


def test_a_rescaled_prony_body_rebuilds_from_its_record():
    """Rescaling keeps every constant a record needs, in the new units."""
    sk, kappa, mu, eta, el, law = iso_medium()
    tau = RadialField(sk, [lambda r: 1.0e10 + 0 * r] * 3, name="tau_1",
                      dimensions=Dimensions.TIME)
    body = ReferenceBody.from_fields(
        sk, {"el": el, "kappa": kappa, "mu": mu, "tau_1": tau,
             "viscosity": eta})
    body.add_field("dyn", prony(el, [tau], [mu]))
    body.add_field("mx", maxwell(el, eta))
    nd = body.nondimensionalised()
    s = nd.scales
    for name in ("dyn", "mx"):
        for lay in nd.layers:
            piece = lay[name]
            record = law_record_of(piece)
            again = rebuild(record, lay.fields)
            lo, hi = lay.interval
            r = np.linspace(lo + 0.1 * (hi - lo), hi - 0.1 * (hi - lo), 5)
            for omega in (1.0e-12 * s.time, 1.0e-10 * s.time):
                want = piece.evaluate(r, omega=omega)
                assert np.allclose(again.evaluate(r, omega=omega), want,
                                   rtol=1e-12, atol=1e-12 * np.max(np.abs(want)))
    assert nd.layer(0)["dyn"].law.constants == {"terms": 1.0}
    # and the rescaled series is the SI series re-expressed
    r_si = np.array([0.5, 2.5])
    omega_si = 3.0e-12
    want = body["dyn"].evaluate(r_si, omega=omega_si)
    got = nd["dyn"].evaluate(r_si / s.length, omega=omega_si * s.time)
    assert np.allclose(got * s.modulus, want, rtol=1e-12)


def test_a_user_law_rebuilds_through_its_own_from_record():
    """A law registered outside the library carries its own inverse."""
    sk, kappa, mu, eta, el, law = iso_medium()

    def kelvin_voigt(moduli, viscosity, *, name=None):
        def fn(omega, kappa, mu, eta):
            return {"kappa": kappa + 0.0j, "mu": mu + 1j * omega * eta}
        return ComposedFrequencyField(
            lambda omega, k, m, e: fn(omega, k, m, e)["mu"],
            [moduli.components["kappa"], moduli.components["mu"], viscosity],
            character=SCALAR, name=name or "kelvin_voigt",
            law=LawRecord("test_kelvin_voigt",
                          parameters=("el", "kappa", "mu", "viscosity")))

    def from_record(record, fields):
        el_, _, _, eta_ = (fields[n] for n in record.parameters)
        return kelvin_voigt(el_, eta_)

    kelvin_voigt.from_record = from_record
    register("rheology", "test_kelvin_voigt", kelvin_voigt)

    def orphan(moduli, viscosity):
        return kelvin_voigt(moduli, viscosity, name="orphan")
    register("rheology", "test_orphan", orphan)

    fields = {"el": el, "kappa": kappa, "mu": mu, "viscosity": eta}
    f = kelvin_voigt(el, eta)
    again = rebuild(law_record_of(f), fields)
    r = np.array([0.5, 2.5])
    assert np.array_equal(again.evaluate(r, omega=2.0), f.evaluate(r, omega=2.0))
    with pytest.raises(ValueError, match="from_record"):
        rebuild(LawRecord("test_orphan", parameters=("el", "viscosity")), fields)


def test_rebuild_refuses_what_it_cannot_do_by_name():
    from planetmodel.model.fields.composite import ComposedField
    sk, kappa, mu, eta, el, law = iso_medium()
    with pytest.raises(ValueError, match="no field named 'viscosity'"):
        rebuild(law_record_of(law), {"el": el})
    with pytest.raises(ValueError, match="no rheology law named 'kelvin'"):
        rebuild(LawRecord("kelvin", parameters=("el",)), {"el": el})
    with pytest.raises(TypeError):
        rebuild("constant_q", {})
    # a composition written by hand has no provenance and says so
    hand = ComposedField(lambda k: 2.0 * k, (kappa,), name="twice")
    assert law_record_of(hand) is None
    assert law_record_of(kappa) is None
