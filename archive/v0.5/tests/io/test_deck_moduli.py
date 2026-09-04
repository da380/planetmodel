"""Decks convert velocities to moduli on load, without losing exactness.

The claim under test is narrow and important: PREM's fields are exact
polynomials, the moduli are products of them, and forming those products
on the coefficients keeps the result exact.  If the conversion ever falls
back to sample-and-refit, PREM quietly stops being an exact model -- so
these tests check the values against the defining formula at many radii,
and check the integrals, rather than trusting that no warning appeared.
"""
import warnings

import numpy as np
import pytest

from planetmodel import PREM, Dimensions, Symmetry
from planetmodel.io.deck import read_isotropic_deck, read_mineos_deck
from planetmodel.model.materials import ElasticField, voigt_matrix
from planetmodel.testing import check_field

DECK = "examples/prem.200"
ISO_DECK = "tests/data/prem.nocrust"


@pytest.fixture(scope="module")
def prem():
    return PREM()


@pytest.fixture(scope="module")
def deck():
    return read_mineos_deck(DECK)


@pytest.fixture(scope="module")
def iso():
    with warnings.catch_warnings():
        # the file's header says 220 knots and carries 214: the crustal
        # knots were removed without updating it.  The table is ground
        # truth, so the reader warns and proceeds.
        warnings.simplefilter("ignore", UserWarning)
        return read_isotropic_deck(ISO_DECK)


# ------------------------------------------------------- exactness (the DoD)

def test_prem_moduli_are_exact_at_ten_thousand_radii_per_layer(prem):
    """A = rho vph^2 and friends, to machine precision, layer by layer."""
    sk = prem.skeleton
    for i in range(sk.nlayers):
        lo, hi = sk.interval(i)
        r = np.linspace(lo, hi, 10_000)
        rho = prem.rho[i](r)
        want = {
            "A": rho * prem.vph[i](r) ** 2,
            "C": rho * prem.vpv[i](r) ** 2,
            "L": rho * prem.vsv[i](r) ** 2,
            "N": rho * prem.vsh[i](r) ** 2,
        }
        want["F"] = prem.eta[i](r) * (want["A"] - 2.0 * want["L"])
        for name, target in want.items():
            got = prem[name][i](r)
            scale = max(1.0, float(np.max(np.abs(target))))
            assert np.allclose(got, target, rtol=1e-13, atol=1e-13 * scale), (
                f"layer {i}, modulus {name}")


def test_prem_moduli_integrals_match_quadrature(prem):
    """.integrate on a modulus agrees with fine quadrature to 1e-14."""
    sk = prem.skeleton
    for i in range(sk.nlayers):
        lo, hi = sk.interval(i)
        fine = np.linspace(lo, hi, 100_001)
        for name in ("A", "C", "F", "L", "N"):
            want = np.trapezoid(prem[name][i](fine), fine)
            got = prem[name][i].integrate(lo, hi)
            if abs(want) > 0.0:
                assert abs(got - want) <= 1e-8 * abs(want), f"layer {i}, {name}"


def test_conversion_never_refits(prem):
    """No sample-and-refit anywhere in the conversion, for any reader."""
    for build in (PREM,
                  lambda: read_mineos_deck(DECK),
                  lambda: read_isotropic_deck(ISO_DECK)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            build()
        refits = [str(x.message) for x in w if "refitted" in str(x.message)]
        assert not refits, refits


# ------------------------------------------------------------- what is built

@pytest.mark.parametrize("name, symmetry", [
    ("prem", Symmetry.VTI), ("deck", Symmetry.VTI),
    ("iso", Symmetry.ISOTROPIC)])
def test_every_reader_attaches_moduli_and_an_elastic_field(name, symmetry,
                                                           request):
    """The five moduli in every case; the symmetry the deck actually has.

    A deck tabulating vp and vs alone describes an isotropic medium, and
    the ElasticField says so rather than presenting a VTI
    tensor with degenerate moduli.  The five moduli stay attached in
    both cases -- the 1D solver and the velocity views read them by
    name -- as the derived five-moduli form of the same medium.
    """
    model = request.getfixturevalue(name)
    for modulus in ("A", "C", "F", "L", "N"):
        assert modulus in model
    elastic = model["elastic_moduli"]
    assert isinstance(elastic, ElasticField)
    assert elastic.symmetry is symmetry
    assert elastic.character.rank == 4 and elastic.character.weight == 1


@pytest.mark.parametrize("name", ["prem", "deck", "iso"])
def test_moduli_fields_satisfy_the_field_contract(name, request):
    model = request.getfixturevalue(name)
    for modulus in ("A", "C", "F", "L", "N"):
        check_field(model[modulus])


def test_velocity_columns_are_untouched(deck):
    """Conversion adds; it does not replace what the file said."""
    for column in ("rho", "vpv", "vsv", "qkappa", "qmu", "vph", "vsh", "eta"):
        assert column in deck


def test_elastic_field_agrees_with_the_moduli(prem):
    r = np.array([1e6, 3e6, 5e6, 6.3e6])
    want = voigt_matrix(Symmetry.VTI,
                        {k: prem[k].evaluate(r) for k in "ACFLN"})
    assert np.allclose(prem["elastic_moduli"].evaluate(r), want, rtol=1e-14)


def test_density_carries_weight_one(prem):
    """rho is weight 1; the rheological columns are invariant."""
    assert prem.rho.character.weight == 1
    assert prem.qmu.character.weight == 0


# ------------------------------------------------------------ read_isotropic_deck

def test_isotropic_deck_reads_prem_nocrust(iso):
    assert iso.skeleton.nlayers == 10
    assert iso.skeleton.boundaries[-1] == pytest.approx(6346600.0)
    assert iso.meta["name"] == "prem.200noiso"
    for column in ("rho", "vp", "vs", "qkappa", "qmu"):
        assert column in iso


def test_isotropy_relations_hold(iso):
    """A = C, L = N and F = A - 2L, with no eta column in the file.

    A and C come from the identical computation on the identical vp
    field, so they agree bit for bit, and likewise L and N.  F is built
    by combining polynomial coefficients while the check recombines
    evaluated values, so the two differ by a few ulps of rounding --
    measured at most 1.14e-15 relative across every layer -- which is
    what 1e-14 allows for and 1e-15 would not.
    """
    sk = iso.skeleton
    for i in range(sk.nlayers):
        lo, hi = sk.interval(i)
        r = np.linspace(lo, hi, 500)
        A, C = iso.A[i](r), iso.C[i](r)
        L, N = iso.L[i](r), iso.N[i](r)
        F = iso.F[i](r)
        assert np.array_equal(A, C), f"layer {i}: A != C"
        assert np.array_equal(L, N), f"layer {i}: L != N"
        assert np.allclose(F, A - 2.0 * L, rtol=1e-14), f"layer {i}: F != A - 2L"


def test_exactly_one_fluid_layer_and_it_is_the_outer_core(iso):
    """The brief's claim about prem.nocrust, verified.

    vs == 0 picks out the outer core alone, which is what makes it a
    usable default rule for classifying layers.
    """
    sk = iso.skeleton
    fluid = [i for i in range(sk.nlayers)
             if iso.vs[i](np.mean(sk.interval(i))) == 0.0]
    assert fluid == [1]
    lo, hi = sk.interval(1)
    assert lo == pytest.approx(1221500.0)   # ICB
    assert hi == pytest.approx(3480000.0)   # CMB


def test_header_mismatch_warns_rather_than_failing():
    """The table is ground truth; a stale header is only a warning."""
    with pytest.warns(UserWarning, match="header says 220 knots"):
        read_isotropic_deck(ISO_DECK)


# --------------------------------------- the solver reads stored moduli

def test_ti_moduli_reads_the_stored_fields(prem):
    """Since M2 this is a lookup, not a conversion."""
    from planetmodel import RadialMesh
    from planetmodel.loading import ti_moduli

    mesh = RadialMesh(prem, ngll=5, drmax=200e3)
    got = ti_moduli(mesh)
    for arr, name in zip(got, ("A", "C", "F", "L", "N")):
        assert np.array_equal(arr, mesh.nodal(name))


def test_ti_moduli_falls_back_for_a_model_without_moduli(prem):
    """A hand-built model with only velocities still works."""
    from planetmodel import RadialMesh, ReferenceBody
    from planetmodel.loading import ti_moduli

    bare = ReferenceBody.from_fields(
        prem.skeleton,
        {n: prem[n] for n in ("rho", "vpv", "vsv", "vph", "vsh", "eta")},
        meta=dict(prem.meta))
    assert "A" not in bare

    mesh_bare = RadialMesh(bare, ngll=5, drmax=200e3)
    mesh_full = RadialMesh(prem, ngll=5, drmax=200e3)
    for a, b in zip(ti_moduli(mesh_bare), ti_moduli(mesh_full)):
        assert np.allclose(a, b, rtol=1e-12)


def test_voigt_moduli_now_derives_from_ti_moduli(prem):
    """One conversion path, so the two cannot drift apart."""
    from planetmodel import RadialMesh
    from planetmodel.loading import ti_moduli, voigt_moduli

    mesh = RadialMesh(prem, ngll=5, drmax=200e3)
    A, C, F, L, N = ti_moduli(mesh)
    kappa, mu = voigt_moduli(mesh)
    assert np.allclose(kappa, (4 * A + C + 4 * F - 4 * N) / 9, rtol=1e-14)
    assert np.allclose(mu, (A + C - 2 * F + 5 * N + 6 * L) / 15, rtol=1e-14)


def test_the_isotropic_deck_stores_kappa_and_mu(iso):
    """kappa = rho(vp^2 - 4 vs^2 / 3) and mu = rho vs^2, on the deck's own knots."""
    assert "kappa" in iso and "mu" in iso
    assert iso["elastic_moduli"].moduli_names == ("kappa", "mu")
    r = np.linspace(*iso.skeleton.interval(5), 200)
    rho, vp, vs = (iso[n].evaluate(r) for n in ("rho", "vp", "vs"))
    assert np.allclose(iso.kappa.evaluate(r), rho * (vp ** 2 - 4.0 * vs ** 2 / 3.0),
                       rtol=1e-13)
    assert np.allclose(iso.mu.evaluate(r), rho * vs ** 2, rtol=1e-14)
    for name in ("kappa", "mu"):
        assert iso[name].dimensions is Dimensions.MODULUS
        assert iso[name].character.rank == 0


def test_widening_the_isotropic_field_returns_the_attached_five_moduli(iso):
    """as_symmetry(VTI) is a re-description, not a second opinion.

    kappa and mu were built from A and L on the same layer functions, so
    widening them back must land on A, C, F, L, N as attached -- to
    rounding, since the two routes recombine polynomial coefficients in
    a different order.
    """
    vti = iso["elastic_moduli"].as_symmetry(Symmetry.VTI)
    assert vti.symmetry is Symmetry.VTI
    r = np.linspace(1e5, 6.3e6, 501)
    for name in ("A", "C", "F", "L", "N"):
        got = np.asarray(vti.components[name].evaluate(r), dtype=float)
        want = np.asarray(iso[name].evaluate(r), dtype=float)
        err = np.max(np.abs(got - want)) / np.max(np.abs(want))
        assert err < 1e-13, f"{name}: relative error {err:.3e}"
    assert np.allclose(vti.evaluate(r), iso["elastic_moduli"].evaluate(r), rtol=1e-13)


def test_the_isotropic_tensor_is_the_same_in_both_frames(iso):
    """Isotropy is a statement about every frame; the angles are optional."""
    r = np.array([2e6, 5e6, 6.3e6])
    theta = np.array([0.3, 1.1, 2.7])
    phi = np.array([-2.0, 0.4, 3.0])
    sph = iso["elastic_moduli"].evaluate(r, theta, phi)
    assert np.allclose(iso["elastic_moduli"].evaluate(r, theta, phi, frame="cartesian"),
                       sph, rtol=1e-13)
    assert np.allclose(iso["elastic_moduli"].evaluate(r, frame="cartesian"), sph)


def test_a_vti_tensor_refuses_cartesian_without_angles(prem):
    """The frame carrying the symmetry axis is a function of direction."""
    with pytest.raises(ValueError, match="depend on direction"):
        prem["elastic_moduli"].evaluate(5e6, frame="cartesian")
    with pytest.raises(ValueError, match="unknown frame"):
        prem["elastic_moduli"].evaluate(5e6, frame="galactic")


def test_the_readers_classify_the_outer_core_as_fluid():
    """A layer with no shear velocity is fluid from the moment it is read."""
    iso = read_isotropic_deck("tests/data/prem.nocrust")
    assert [lay.state for lay in iso.layers][:3] == ["solid", "fluid", "solid"]
    mineos = read_mineos_deck("examples/prem.200")
    assert mineos.layers[1].state == "fluid"
    assert all(lay.state == "solid" for lay in mineos.layers[2:-1])
    prem = PREM()
    assert prem.layers[1].state == "fluid" and prem.layers[-1].state == "fluid"
