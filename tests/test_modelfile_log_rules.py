"""The radial rules in the logarithm: a positive field whose logarithm is
a piecewise polynomial, written exactly and read back.

A Maxwell deck's viscosity changing by one, two and three decades
between knots comes back to rounding, values and derivatives, on its
knots alone when written alone and beside every other field, a layer of
one viscosity to the bit and the layers without one without; the
polynomial rule remains by asking, an approximation.  A prony deck's
relaxation times and a burgers deck's viscosity come back so too.  A
variable asking for a rule in the logarithm and for another, or varying
with direction, is refused by name, and parts of their own hold each
exactly.  A viscosity perturbed in its logarithm about a graded
reference, on a model placed by boundary reliefs, comes back exactly.
The schema is the second exactly where a variable states such a rule;
the readers refuse a file breaking that or the rule's terms, and read a
variable by another rule where asked.
"""
import numpy as np
import pytest

from planetmodel import (AnalyticField, BurgersDeckModel, Deck, GriddedField,
                         GriddedFunction, MaxwellDeckModel, PronyDeckModel,
                         RadialField, flattening, gauss_legendre, layer_linear,
                         perturbed)
from planetmodel.character import DENSITY, VECTOR
from planetmodel.conversion import prony
from planetmodel.layerfunction import ExponentialLayer, PolynomialLayer
from planetmodel.testing import check_field, check_model_file

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.modelfile import (LOG_RULES, RADIAL_RULES, SCHEMA,  # noqa: E402
                                   SCHEMA_WITH_LOG_RULES, read_model, read_model_file,
                                   schema_of, write_model)

#: The knots of the deck's four layers: a fluid core, a mantle whose
#: viscosity varies, a mantle of one viscosity, an elastic crust.
KNOTS = [np.linspace(0.0, 1.2e6, 3), np.linspace(1.2e6, 5.0e6, 6),
         np.linspace(5.0e6, 6.2e6, 4), np.linspace(6.2e6, 6.371e6, 2)]

#: The viscosity of the third layer, throughout it.
ETA = 1e21


def maxwell_deck(decades: float) -> Deck:
    """A Maxwell deck whose second layer's viscosity goes up and down by
    `decades` between its knots, 760 km apart."""
    ends = {"rho": [(11000, 10000), (5500, 4500), (4000, 3500), (2900, 2700)],
            "vp": [(9000, 8500), (13000, 11000), (10000, 8500), (6800, 6500)],
            "vs": [(0, 0), (7000, 6000), (5500, 4700), (3900, 3700)]}
    columns = {}
    for name, column in (("rho", "rho"), ("vpv", "vp"), ("vsv", "vs"),
                         ("vph", "vp"), ("vsh", "vs")):
        columns[name] = np.concatenate([np.linspace(a, b, r.size)
                                        for (a, b), r in zip(ends[column], KNOTS)])
    columns["eta"] = np.ones(sum(r.size for r in KNOTS))
    columns["viscosity"] = np.concatenate([
        np.full(3, np.nan), 10.0 ** (20.0 + decades * np.array([0, 1, 0, 1, 0, 1])),
        np.full(4, ETA), np.full(2, np.nan)])
    return Deck(np.concatenate(KNOTS), columns)


def same_function(a, b, *, rtol, drtol):
    """b is a to `rtol` and its derivative to `drtol`, point by point."""
    r = np.linspace(*a.interval, 2001)
    assert np.allclose(b(r), a(r), rtol=rtol, atol=0)
    assert np.allclose(b.derivative()(r), a.derivative()(r), rtol=drtol, atol=0)


def one_viscosity(field):
    """A layer of one viscosity, to the bit and without slope."""
    r = np.linspace(*field.interval, 101)
    assert np.array_equal(field(r), np.full(r.size, ETA))
    assert np.array_equal(field.derivative()(r), np.zeros(r.size))


def rules_of(path):
    """Every variable's radial rule, by name, over the parts."""
    return {name: var.attributes.get("radial_interpolation")
            for part in read_model_file(path).parts
            for name, var in part.variables.items() if not name.endswith("_poles")}


@pytest.mark.parametrize("decades", [1.0, 2.0, 3.0])
def test_a_graded_viscosity_comes_back_on_its_knots(decades, tmp_path):
    """Written alone, the viscosity's elements are its knots, two nodes
    each; beside every field, ten nodes each; either way it comes back
    to rounding with its derivative, the layer of one viscosity to the
    bit.  The polynomial rule, asked for, misses it between knots."""
    m = MaxwellDeckModel(maxwell_deck(decades))
    assert isinstance(m.layer(1)["viscosity"].function, ExponentialLayer)
    path = write_model(tmp_path / "eta.nc", m, grid=gauss_legendre(2),
                       fields=["viscosity"])
    check_model_file(path, m)
    data = read_model_file(path)
    mesh = data.parts[0].mesh
    assert data.schema == schema_of(path) == SCHEMA_WITH_LOG_RULES
    assert rules_of(path) == {"viscosity": "log-polynomial"}
    assert mesh.ngll == 2
    graded = mesh.layer == 1
    assert np.array_equal(np.append(mesh.left[graded], mesh.right[graded][-1]),
                          m.knots[1])
    back = read_model(path)
    assert back.layers_with("viscosity") == m.layers_with("viscosity") == (1, 2)
    assert isinstance(back.layer(1)["viscosity"].function, ExponentialLayer)
    same_function(m.layer(1)["viscosity"], back.layer(1)["viscosity"], rtol=1e-14,
                  drtol=1e-14)
    one_viscosity(back.layer(2)["viscosity"])

    path = write_model(tmp_path / "all.nc", m, grid=gauss_legendre(2))
    check_model_file(path, m)
    assert read_model_file(path).parts[0].mesh.ngll == 10
    assert {rule for name, rule in rules_of(path).items()
            if name != "viscosity"} == {"polynomial"}
    back = read_model(path)
    same_function(m.layer(1)["viscosity"], back.layer(1)["viscosity"], rtol=1e-13,
                  drtol=5e-13)
    one_viscosity(back.layer(2)["viscosity"])
    for name in ("rho", "vsv", "L"):
        same_function(m.layer(1)[name], back.layer(1)[name], rtol=1e-13, drtol=1e-10)

    path = write_model(tmp_path / "poly.nc", m, grid=gauss_legendre(2),
                       fields=["viscosity"],
                       interpolation={"viscosity": ("polynomial", "linear")})
    assert read_model_file(path).schema == SCHEMA
    assert rules_of(path) == {"viscosity": "polynomial"}
    a, b = m.layer(1)["viscosity"], read_model(path).layer(1)["viscosity"]
    r = np.linspace(*a.interval, 2001)
    assert np.max(np.abs(b(r) / a(r) - 1.0)) > 0.04 * decades


def test_a_prony_deck_s_relaxation_times_come_back(tmp_path):
    """The times joined in their logarithm come back by the rule in the
    logarithm, the strengths, straight lines, by the polynomial one."""
    p = PronyDeckModel(prony.prony_deck(MaxwellDeckModel(maxwell_deck(3.0))))
    path = write_model(tmp_path / "prony.nc", p, grid=gauss_legendre(2))
    check_model_file(path, p)
    rules = rules_of(path)
    assert rules["relaxation_time_1"] == "log-polynomial"
    assert rules["relaxation_strength_1"] == "polynomial"
    back = read_model(path)
    for i in (1, 2):
        assert isinstance(p.layer(i)["relaxation_time_1"].function, ExponentialLayer)
        same_function(p.layer(i)["relaxation_time_1"],
                      back.layer(i)["relaxation_time_1"], rtol=1e-13, drtol=5e-13)
        r = np.linspace(*p.layer(i).interval, 101)
        assert np.allclose(back.layer(i)["relaxation_strength_1"](r),
                           p.layer(i)["relaxation_strength_1"](r), rtol=1e-14, atol=0)


def test_a_burgers_deck_s_viscosity_comes_back_and_its_branches_as_sampled(tmp_path):
    """A Maxwell layer's graded viscosity comes back by the rule in the
    logarithm; a Burgers layer's branches, functions of its columns that
    declare no pieces, by the polynomial rule on the generic layout."""
    maxwell = MaxwellDeckModel(maxwell_deck(3.0))
    b = BurgersDeckModel(prony.burgers_deck(prony.with_burgers(
        maxwell, [2], kelvin_viscosity=1e20, kelvin_modulus_fraction=0.5)))
    assert b.layers_with("viscosity") == (1,)
    assert b.layers_with("relaxation_time_2") == (2,)
    path = write_model(tmp_path / "burgers.nc", b, grid=gauss_legendre(2))
    check_model_file(path, b)
    rules = rules_of(path)
    assert rules["viscosity"] == "log-polynomial"
    assert rules["relaxation_time_1"] == rules["relaxation_time_2"] == "polynomial"
    back = read_model(path)
    same_function(b.layer(1)["viscosity"], back.layer(1)["viscosity"], rtol=1e-13,
                  drtol=5e-13)
    r = np.linspace(*b.layer(2).interval, 401)
    for name in ("relaxation_time_1", "relaxation_time_2"):
        assert np.allclose(back.layer(2)[name](r), b.layer(2)[name](r), rtol=1e-12)


def test_a_variable_asking_for_two_radial_rules_is_refused_by_name(tmp_path):
    """The graded viscosity beside a viscosity that is a line, or beside
    one that varies with direction, in one part: no one rule holds both,
    so the writer names the layers and the ways out.  Asked for one rule,
    it writes it, approximately where it does not hold; in parts of their
    own, each layer is held exactly."""
    from scipy.interpolate import PPoly
    m = MaxwellDeckModel(maxwell_deck(2.0))
    lo, hi = m.layer(2).interval
    line = RadialField((lo, hi), PPoly([[1e15], [ETA]], [lo, hi]), name="viscosity")
    mixed = m.with_field(2, "viscosity", line, replace=True)
    with pytest.raises(ValueError, match=r"'viscosity' is the exponential of a "
                       r"piecewise polynomial on layers \[1\].*in its values on "
                       r"layers \[2\].*interpolation=.*parts of their own"):
        write_model(tmp_path / "x.nc", mixed, grid=gauss_legendre(2))
    asked = write_model(tmp_path / "asked.nc", mixed, grid=gauss_legendre(2),
                        fields=["viscosity"],
                        interpolation={"viscosity": ("polynomial", "linear")})
    assert rules_of(asked) == {"viscosity": "polynomial"}
    r = np.linspace(lo, hi, 101)
    assert np.allclose(read_model(asked).layer(2)["viscosity"](r), line(r), rtol=1e-14)
    grid = {0: gauss_legendre(2), 1: gauss_legendre(2), 2: gauss_legendre(4),
            3: gauss_legendre(4)}
    apart = write_model(tmp_path / "apart.nc", mixed, grid=grid, fields=["viscosity"])
    data = read_model_file(apart)
    assert data.schema == SCHEMA_WITH_LOG_RULES
    assert [p.variables["viscosity"].attributes["radial_interpolation"]
            for p in data.parts] == ["log-polynomial", "polynomial"]
    back = read_model(apart)
    same_function(m.layer(1)["viscosity"], back.layer(1)["viscosity"], rtol=1e-14,
                  drtol=1e-14)
    assert np.allclose(back.layer(2)["viscosity"](r), line(r), rtol=1e-14)

    gridded = GriddedField.sampled(AnalyticField(
        (lo, hi), lambda r, t, p: ETA * (2.0 + np.cos(t))), gauss_legendre(4), ngll=3)
    lateral = m.with_field(2, "viscosity", gridded.renamed("viscosity"), replace=True)
    with pytest.raises(ValueError,
                       match=r"layers \[1\].*in its values on layers \[2\]"):
        write_model(tmp_path / "x.nc", lateral, grid=gauss_legendre(4))
    unstated = m.with_field(2, "viscosity", AnalyticField(
        (lo, hi), lambda r, t, p: ETA * (2.0 + np.cos(t)), name="viscosity"),
        replace=True)
    with pytest.raises(ValueError, match=r"varies with direction on layers \[2\]"):
        write_model(tmp_path / "x.nc", unstated, grid=gauss_legendre(4))
    path = write_model(tmp_path / "parts.nc", lateral, fields=["viscosity"], grid=grid)
    check_model_file(path, lateral)


def test_a_rule_in_the_logarithm_is_refused_where_it_cannot_read(tmp_path):
    m = MaxwellDeckModel(maxwell_deck(1.0))
    with pytest.raises(ValueError, match="rules must be one of"):
        write_model(tmp_path / "x.nc", m, grid=gauss_legendre(2),
                    interpolation={"viscosity": ("log-cubic", "linear")})
    with pytest.raises(ValueError, match=r"'vsv' is 0 at r = 0 on layer 0; a variable "
                       "read by log-linear, its logarithm interpolated, is positive"):
        write_model(tmp_path / "x.nc", m, grid=gauss_legendre(2), fields=["vsv"],
                    interpolation={"vsv": ("log-linear", "linear")})
    with pytest.raises(ValueError, match="'displacement': log-polynomial reads the "
                       "logarithm of a number"):
        write_model(tmp_path / "x.nc", m.stretched(flattening(0.01)),
                    grid=gauss_legendre(2), fields=["rho"],
                    interpolation={"displacement": ("log-polynomial", "cubic")})
    lo, hi = m.layer(1).interval
    flow = m.with_field(1, "flow", RadialField(
        (lo, hi), [lambda r: 1.0 + r, 0.0, 0.0], character=VECTOR, name="flow"))
    with pytest.raises(ValueError, match="'flow': log-linear reads the logarithm"):
        write_model(tmp_path / "x.nc", flow, grid=gauss_legendre(2), fields=["flow"],
                    interpolation={"flow": ("log-linear", "linear")})
    gridded = m.with_field(1, "rho", GriddedField.sampled(AnalyticField(
        (lo, hi), lambda r, t, p: 5000.0 + np.cos(t), character=DENSITY),
        gauss_legendre(4), ngll=3), replace=True)
    with pytest.raises(ValueError, match="'rho': log-polynomial reads the logarithm"):
        write_model(tmp_path / "x.nc", gridded, grid=gauss_legendre(4), fields=["rho"],
                    interpolation={"rho": ("log-polynomial", "linear")})
    assert not list(tmp_path.iterdir())
    # a positive polynomial, by asking, is held in its logarithm,
    # approximately, and comes back so
    path = write_model(tmp_path / "rho.nc", m, grid=gauss_legendre(2), fields=["rho"],
                       interpolation={"rho": ("log-polynomial", "linear")})
    check_model_file(path, m)
    assert rules_of(path) == {"rho": "log-polynomial"}
    r = np.linspace(*m.layer(1).interval, 101)
    assert np.allclose(read_model(path).layer(1)["rho"](r), m.layer(1)["rho"](r),
                       rtol=1e-6)


def test_a_viscosity_perturbed_in_its_logarithm_about_a_graded_one_comes_back(tmp_path):
    """A 3D viscosity, the graded reference times the exponential of a
    gridded perturbation, on a model placed by boundary reliefs: the
    reference states the rule in the logarithm, the perturbation its own,
    and both come back exactly, the field at points between every node to
    rounding; asked for a rule in the logarithm by name, the reference
    takes it and the perturbation the rule it names in its values."""
    m = MaxwellDeckModel(maxwell_deck(3.0))
    lo, hi = m.layer(1).interval
    grid = gauss_legendre(4)
    def lneta(r, t, p):
        return 0.3 * np.sin(t) ** 2 * np.cos(p) * (r - lo) / (hi - lo)

    dlneta = GriddedField.sampled(AnalyticField((lo, hi), lneta), grid, ngll=3,
                                  lateral="cubic")
    relief = GriddedFunction.sampled(lambda t, p: 2e3 * np.sin(t) ** 2 * np.cos(2 * p),
                                     gauss_legendre(8))
    model = perturbed(m, "viscosity", dlneta, kind="log").stretched(
        layer_linear(m.skeleton, [None, relief, None, None, relief]))
    path = write_model(tmp_path / "3d.nc", model, grid=grid,
                       fields=["rho", "viscosity"])
    check_model_file(path, model)
    data = read_model_file(path)
    assert data.schema == SCHEMA_WITH_LOG_RULES and data.geometry is not None
    var = data.parts[0].variables
    assert var["viscosity"].attributes["radial_interpolation"] == "log-polynomial"
    pert = var["viscosity_perturbation"].attributes
    assert pert["radial_interpolation"] == "polynomial"
    assert pert["lateral_interpolation"] == "cubic"
    back = read_model(path)
    rng = np.random.default_rng(5)
    r = rng.uniform(lo, hi, 200)
    t, p = rng.uniform(0.0, np.pi, 200), rng.uniform(0.0, 2.0 * np.pi, 200)
    want = model.layer(1)["viscosity"].evaluate(r, t, p)
    assert np.allclose(back.layer(1)["viscosity"].evaluate(r, t, p), want, rtol=1e-13,
                       atol=0)
    one_viscosity(back.layer(2)["viscosity"])
    asked = write_model(tmp_path / "asked.nc", model, grid=grid, fields=["viscosity"],
                        interpolation={"viscosity": ("log-polynomial", "cubic")})
    var = read_model_file(asked).parts[0].variables
    assert var["viscosity"].attributes["radial_interpolation"] == "log-polynomial"
    pert = var["viscosity_perturbation"].attributes
    assert pert["radial_interpolation"] == "polynomial"
    again = read_model(asked, interpolation={"viscosity": ("log-polynomial", "cubic")})
    assert np.allclose(again.layer(1)["viscosity"].evaluate(r, t, p), want, rtol=1e-13,
                       atol=0)


def test_a_field_declares_its_logarithm_s_pieces():
    m = MaxwellDeckModel(maxwell_deck(2.0))
    graded, constant = m.layer(1)["viscosity"], m.layer(2)["viscosity"]
    breaks, degree = graded.log_radial_pieces()
    assert np.array_equal(breaks, m.knots[1]) and degree == 1
    assert graded.radial_pieces() is None
    assert constant.log_radial_pieces()[1] == 0 and constant.radial_pieces()[1] == 0
    assert m.layer(1)["rho"].log_radial_pieces() is None
    check_field(graded)
    check_field(constant)


def test_the_schema_is_the_second_exactly_where_a_rule_in_the_logarithm_is_stated(
        tmp_path):
    """A deck model joined in its values states no rule in the logarithm
    and its file is the first schema, every rule one of the first two; a
    reader refuses a rule in the logarithm under the first schema, the
    second schema without one, the rule on a variable that varies with
    direction or holds a value that is not positive, and a schema it does
    not know, naming both it reads."""
    plain = MaxwellDeckModel(maxwell_deck(2.0), kinds={"viscosity": "linear"})
    path = write_model(tmp_path / "plain.nc", plain, grid=gauss_legendre(2))
    check_model_file(path, plain)
    assert schema_of(path) == SCHEMA
    assert set(rules_of(path).values()) == {"polynomial"}
    assert RADIAL_RULES[:2] == ("polynomial", "linear")
    assert LOG_RULES == ("log-polynomial", "log-linear")
    logged = write_model(tmp_path / "logged.nc", MaxwellDeckModel(maxwell_deck(2.0)),
                         grid=gauss_legendre(2))

    def refused(source, edit, message):
        target = tmp_path / f"edited_{source.name}"
        target.write_bytes(source.read_bytes())
        with netCDF4.Dataset(target, "a") as ds:
            edit(ds)
        with pytest.raises(ValueError, match=message):
            read_model_file(target)

    refused(logged, lambda ds: ds.setncattr("schema", SCHEMA),
            r"part_0/viscosity states a rule in the logarithm, which a file of "
            r"schema 'planetmodel.model/1' does not hold: such a file is "
            r"'planetmodel.model/2'")
    refused(path, lambda ds: ds.setncattr("schema", SCHEMA_WITH_LOG_RULES),
            "and none does")
    refused(path, lambda ds: ds.setncattr("schema", "planetmodel.model/3"),
            "this reader reads 'planetmodel.model/1' and 'planetmodel.model/2'")

    def negative(ds):
        var = ds.groups["part_0"].variables["viscosity"]
        held = np.flatnonzero(~np.isnan(np.ma.filled(var[:], np.nan)))
        var[int(held[0])] = -1.0

    refused(logged, negative, "holds a value that is not positive")
    lateral = perturbed(plain, "rho", AnalyticField(
        plain.layer(1).interval, lambda r, t, p: 0.01 * np.cos(t)), kind="relative")
    three_d = write_model(tmp_path / "3d.nc", lateral, grid=gauss_legendre(4),
                          fields=["rho"])
    refused(three_d, lambda ds: ds.groups["part_0"].variables[
        "rho_perturbation"].setncattr("radial_interpolation", "log-linear"),
        "rank 0 of the radius alone")


def test_a_reader_reads_by_another_rule_where_asked(tmp_path):
    """A variable stating a rule in the logarithm read by the polynomial
    rule through its values, and a positive one stating the polynomial
    rule read in its logarithm, through its nodes and near it between
    them; a rule in the logarithm asked of a
    variable that varies with direction, or of values that are not
    positive, is refused."""
    m = MaxwellDeckModel(maxwell_deck(1.0))
    path = write_model(tmp_path / "m.nc", m, grid=gauss_legendre(2),
                       fields=["viscosity", "rho", "vsv"])
    back = read_model(path, interpolation={"viscosity": ("polynomial", None),
                                           "rho": ("log-polynomial", None)})
    assert isinstance(back.layer(1)["viscosity"].function, PolynomialLayer)
    assert isinstance(back.layer(1)["rho"].function, ExponentialLayer)
    r = np.linspace(*m.layer(1).interval, 101)
    nodes = read_model_file(path).parts[0].mesh.r.ravel()
    nodes = nodes[(nodes >= r[0]) & (nodes <= r[-1])]
    assert np.allclose(back.layer(1)["rho"](nodes), m.layer(1)["rho"](nodes),
                       rtol=1e-14, atol=0)
    assert np.allclose(back.layer(1)["rho"](r), m.layer(1)["rho"](r), rtol=1e-6, atol=0)
    with pytest.raises(ValueError, match="not positive"):
        read_model(path, interpolation={"vsv": ("log-polynomial", None)})
    lateral = m.with_field(1, "rho", AnalyticField(
        m.layer(1).interval, lambda r, t, p: 5000.0 + np.cos(t), character=DENSITY),
        replace=True)
    three_d = write_model(tmp_path / "3d.nc", lateral, grid=gauss_legendre(4),
                          fields=["rho"])
    with pytest.raises(ValueError, match="'rho' varies with direction on layer 1, and "
                       "log-linear reads a variable of the radius alone"):
        read_model(three_d, interpolation={"rho": ("log-linear", "linear")})
