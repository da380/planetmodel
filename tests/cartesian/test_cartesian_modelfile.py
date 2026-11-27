"""The flat model file: a model of `planetmodel.cartesian` written,
validated and read back.

A box and a rectangle, padded, with fields of depth alone, gridded fields
of ranks 0 to 2 under each lateral rule, a Fourier field and a Fourier
perturbation, with and without a relief, a stretch held as its upward
displacement, a prescribed gravity, fields on interfaces, a rule in the
logarithm and other scales: each comes back at the file's nodes, and
between them by the stated rules, the domain's bounds and padding to the
bit and the mapped points of the body, held to the file by
`check_model_file`.  A model of depth alone is written with no grid
given, and a model read back has the behaviours its file names.  What
the flat file cannot hold is refused on
writing; a damaged flat file is refused by name; xarray opens it as CF
tools do.
"""
import dataclasses
import warnings

import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import (SCALAR, STRESS, VECTOR, ConstantQ, Elastic, Scales,
                         SelfGravitating, Viscoelastic, gauss_legendre, perturbed)
from planetmodel import cartesian as flat
from planetmodel.behaviours import model_type
from planetmodel.cartesian import FLAT
from planetmodel.character import Character
from planetmodel.testing import check_geometry, check_model, check_model_file

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel.cartesian.modelfile import read_model_file  # noqa: E402
from planetmodel.modelfile import (SCHEMA_FLAT, radial_grid, read_model,  # noqa: E402
                                   read_surfaces, schema_of, write_model)

BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)], padding=(0.25, 0.5))
RECT = flat.Rectangle((0.0, 2.0), padding=0.3)
TENSOR = Character(2, 0, voigt=False)
MANTLE = (0.4, 1.0)


def layered(domain, **kw):
    """Two exact layers, a crust over a mantle whose density and speed
    grow with depth, with the Love moduli beside them."""
    return flat.LayeredIsotropicElastic(
        [0.0, 0.4, 1.0], domain, rho=[2.8, (3.3, 0.1)], vp=[6.0, (8.0, 0.5)],
        vs=[3.5, 4.5], layer_names=["crust", "mantle"],
        interface_names=["surface", "moho"], name="two layers", **kw)


def smooth(char, *, rectangle=False):
    """A smooth periodic field of the character on the mantle, local
    components, over the period of BOX (or of RECT)."""
    shape = char.voigt_shape or char.component_shape
    (lx, ly) = (2.5, 3.0) if not rectangle else (2.6, 1.0)

    def fn(d, x, y):
        d, x, y = np.broadcast_arrays(d, x, y)
        u, v = 2 * np.pi * x / lx, (0.0 * y if rectangle else 2 * np.pi * y / ly)
        base = (1.0 + d ** 2) * np.exp(0.3 * np.sin(u) - 0.2 * np.cos(v)
                                       + 0.1 * np.sin(u + v))
        if not char.rank:
            return base
        k = int(np.prod(shape))
        comps = [base * (1.0 + 0.1 * i) + 0.05 * i * np.cos(u) * d for i in range(k)]
        return np.stack(comps, axis=-1).reshape(d.shape + shape)
    return flat.AnalyticField(MANTLE, fn, character=char)


def series(grid, interval, char=SCALAR, *, seed=3):
    """A Fourier field of the grid's band, cubic in depth on one piece."""
    shape = char.voigt_shape or char.component_shape
    c = np.random.default_rng(seed).standard_normal((4, 1, grid.transform.size)
                                                    + shape)
    c[:, :, 1:] /= 4.0
    return flat.FourierField(interval, grid, PPoly(c, np.array(interval)),
                             character=char)


def three_d(domain, grid, *, rectangle=False):
    """The layered model with, on the mantle, gridded fields of every rank
    under each lateral rule on `grid`, and a Fourier field; on the crust a
    relative Fourier perturbation of vs."""
    m = layered(domain)
    for name, char, rule in (("s_linear", SCALAR, "linear"),
                             ("v_cubic", VECTOR, "cubic"),
                             ("t_fourier", STRESS, "fourier"),
                             ("u_cubic", TENSOR, "cubic")):
        m = m.with_field(1, name, flat.GriddedField.sampled(
            smooth(char, rectangle=rectangle), grid, breaks=[0.4, 0.7, 1.0], ngll=4,
            lateral=rule, name=name))
    m = m.with_field(1, "f_series", series(grid, MANTLE, VECTOR).renamed("f_series"))
    return perturbed(m, "vs", series(grid, (0.0, 0.4), seed=5).rescaled(k=1.0, v=0.01),
                     kind="relative")


def random_points(domain, interval, n=300, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = interval
    d = lo + (hi - lo) * rng.uniform(0.001, 0.999, n)
    (x0, x1) = domain.extent[0]
    x = rng.uniform(x0 - 1.0, x1 + 1.0, n)         # beyond the period: periodic
    y = rng.uniform(-3.0, 3.0, n)
    return d, x, y


def worst(model, back, *, n=300):
    """The largest difference between every field of the two models at
    random points, relative to the field's largest value there."""
    out = {}
    for layer in model.layers:
        d, x, y = random_points(model.geometry.domain, layer.interval,
                                seed=layer.index, n=n)
        for name in layer.names:
            want = layer[name].evaluate(d, x, y)
            got = back.layer(layer.index)[name].evaluate(d, x, y)
            out[(layer.index, name)] = float(np.abs(got - want).max()
                                             / max(np.abs(want).max(), 1e-300))
    return out


def written(tmp_path, model, stem="m.nc", **kw):
    path = write_model(tmp_path / stem, model, **kw)
    assert schema_of(path) == SCHEMA_FLAT
    return path


@pytest.mark.parametrize("domain", [BOX, RECT], ids=repr)
def test_a_flat_model_comes_back_from_its_file(tmp_path, domain):
    """Every field at the file's nodes, held by `check_model_file`; between
    them, the gridded fields and the Fourier ones by their own rules
    (stated, and the same), and the fields of depth alone exactly; the
    names, domain and scales the model's."""
    rectangle = domain is RECT
    grid = flat.Grid(domain, (14, 12) if not rectangle else 14)
    model = three_d(domain, grid, rectangle=rectangle)
    path = written(tmp_path, model, grid=grid)
    check_model_file(path, model)
    back = read_model(path)
    assert isinstance(back, flat.Model) and back.geometry.domain == domain
    assert back.geometry.domain.bounds == domain.bounds
    assert back.geometry.domain.padding == domain.padding
    assert [lay.name for lay in back.layers] == ["crust", "mantle"]
    assert [f.name for f in back.geometry.interfaces] == ["surface", "moho"]
    assert back.scales == model.scales and back.name == "two layers"
    check_model(back)
    errors = worst(model, back)
    exact = ("rho", "vp", "vs", "s_linear", "v_cubic", "u_cubic")
    assert max(errors[(i, n)] for (i, n) in errors if n in exact) < 1e-12, errors
    assert errors[(1, "t_fourier")] < 1e-10 and errors[(1, "f_series")] < 1e-10
    assert errors[(0, "vs")] < 1e-12
    with netCDF4.Dataset(path) as ds:
        # the domain as the file states it: its kind, then bounds and padding
        # axis after axis
        kind, bounds, padding = (("box", [0.0, 2.0, -1.0, 1.0], [0.25, 0.5])
                                 if not rectangle else ("rectangle", [0.0, 2.0], [0.3]))
        assert ds.getncattr("domain") == kind
        for key, want in (("domain_bounds", bounds), ("domain_padding", padding)):
            stored = np.atleast_1d(ds.getncattr(key))
            assert stored.dtype == np.float64 and stored.tolist() == want
        assert np.ravel(domain.bounds).tolist() == bounds
        assert list(domain.padding) == padding
        part = ds["part_0"]
        assert part["t_fourier"].lateral_interpolation == "fourier"
        assert part["f_series"].lateral_interpolation == "fourier"
        assert part["v_cubic"].lateral_interpolation == "cubic"
        assert part["vs_perturbation"].lateral_interpolation == "fourier"
        assert part["rho"].dimensions == ("node",)
        assert part["v_cubic"].dimensions == (("node", "x", "y", "component")
                                              if not rectangle else
                                              ("node", "x", "component"))


def test_a_fourier_field_on_a_grid_short_of_its_modes_states_the_default(tmp_path):
    """A series is held by the Fourier rule only on a grid holding its
    modes; on a coarser one the grid takes its values at the nodes, read
    by the bilinear default."""
    fine = flat.Grid(BOX, (16, 12))
    model = layered(BOX).with_field(1, "q", series(fine, MANTLE).renamed("q"))
    coarse = flat.Grid(BOX, (8, 6))
    path = written(tmp_path, model, grid=coarse)
    check_model_file(path, model)
    with netCDF4.Dataset(path) as ds:
        assert ds["part_0"]["q"].lateral_interpolation == "linear"
    path = written(tmp_path, model, "fine.nc", grid=flat.Grid(BOX, (32, 25)))
    with netCDF4.Dataset(path) as ds:
        assert ds["part_0"]["q"].lateral_interpolation == "fourier"
    assert worst(model, read_model(path))[(1, "q")] < 1e-10


def dem(x, y):
    return 0.02 * np.cos(2 * np.pi * x / 2.5) * np.cos(2 * np.pi * y / 3.0) + 0.01


@pytest.mark.parametrize("domain", [BOX, RECT], ids=repr)
def test_a_relief_comes_back_as_the_geometry_group(tmp_path, domain):
    """Gridded reliefs on the surface and the Moho, held as heights on
    their own grid, read back the same functions: the mapped points of the
    body are the model's to rounding, and the geometry passes its contract."""
    rgrid = flat.Grid(domain, (24, 20) if domain is BOX else 24)
    top = flat.GriddedFunction.sampled(dem, rgrid)
    moho = flat.GriddedFunction.sampled(lambda x, y: -3.0 * dem(x, y), rgrid)
    model = layered(domain)
    model = model.stretched(flat.layer_linear(model.skeleton, [top, moho, None]))
    grid = flat.Grid(domain, (6, 5) if domain is BOX else 6)
    path = written(tmp_path, model, grid=grid)
    check_model_file(path, model)
    with netCDF4.Dataset(path) as ds:
        g = ds["geometry"]
        assert g["boundary_displacement"].positive == "up"
        assert g["boundary_displacement"].lateral_interpolation == "cubic"
        assert np.array_equal(g["boundary_depth"][:], [0.0, 0.4, 1.0])
        assert np.array_equal(g["boundary_displacement"][2], np.zeros(rgrid.shape))
        assert "displacement" not in ds["part_0"].variables
    back = read_model(path)
    assert isinstance(back.geometry.mapping, flat.VerticalStretch)
    assert back.geometry.mapping.w.reliefs[2] is None
    d, x, y = random_points(domain, (0.0, 1.0), n=500)
    X = FLAT.points(d, x, y)
    gap = np.abs(back.geometry.mapping(X) - model.geometry.mapping(X)).max()
    assert gap < 1e-15, gap
    check_geometry(back.geometry)


def test_fourier_reliefs_are_held_by_the_fourier_rule(tmp_path):
    """A topography and its root as Fourier functions on one grid: held on
    that grid by their own rule, so the body comes back to the accuracy of
    the synthesis."""
    grid = flat.Grid(BOX, (12, 10))
    x, y = grid.nodes
    top = flat.FourierFunction(grid, grid.transform.analyse_grid(
        np.broadcast_to(dem(x, y), grid.shape)))
    root = flat.FourierFunction(grid, -4.0 * top.coefficients)
    model = layered(BOX)
    model = model.stretched(flat.layer_linear(model.skeleton, [top, root, None]))
    path = written(tmp_path, model, grid=flat.Grid(BOX, 4))
    check_model_file(path, model)
    data = read_model_file(path)
    assert data.geometry.lateral == "fourier" and data.geometry.grid == grid
    d, x, y = random_points(BOX, (0.0, 1.0), n=500)
    X = FLAT.points(d, x, y)
    assert np.abs(read_model(path).geometry.mapping(X)
                  - model.geometry.mapping(X)).max() < 1e-13


def test_another_stretch_comes_back_as_its_displacement(tmp_path):
    """A stretch driven by a gridded displacement is held as `displacement`
    in every part, w at the nodes, positive up, and read back as a gridded
    displacement giving the same body; an analytic one to its sampling."""
    model = layered(BOX)
    grid = flat.Grid(BOX, (20, 16))
    stack = flat.layer_linear(model.skeleton, [dem, lambda x, y: -3.0 * dem(x, y),
                                               None])
    from planetmodel import RadialMesh
    mesh = RadialMesh(model.geometry, ngll=3, drmax=0.25)
    gd = flat.GriddedDisplacement.sampled(stack, mesh, grid)
    stretched = model.stretched(gd)
    path = written(tmp_path, stretched, grid=grid)
    check_model_file(path, stretched)
    with netCDF4.Dataset(path) as ds:
        disp = ds["part_0"]["displacement"]
        assert disp.dimensions == ("node", "x", "y") and disp.positive == "up"
        assert "geometry" not in ds.groups
    back = read_model(path)
    assert isinstance(back.geometry.mapping.w, flat.GriddedDisplacement)
    d, x, y = random_points(BOX, (0.0, 1.0), n=500)
    X = FLAT.points(d, x, y)
    gap = np.abs(back.geometry.mapping(X) - stretched.geometry.mapping(X)).max()
    assert gap < 1e-14, gap
    check_geometry(back.geometry)
    analytic = model.stretched(flat.CallableDisplacement(
        lambda d, x, y: dem(x, y) * (1.0 - d), knots=[0.4]))
    path = written(tmp_path, analytic, "analytic.nc", grid=grid)
    check_model_file(path, analytic)
    gap = np.abs(read_model(path).geometry.mapping(X) - analytic.geometry.mapping(X))
    assert gap.max() < 1e-4, gap.max()


def test_a_prescribed_gravity_and_other_scales_come_back(tmp_path):
    """The uniform gravity a model carries is a field of depth alone, held
    exactly; a model in geophysical scales is read back in them, and in SI
    where asked."""
    model = layered(BOX).with_uniform_gravity(9.81)
    path = written(tmp_path, model, grid=flat.Grid(BOX, 1))
    check_model_file(path, model)
    back = read_model(path)
    d = np.linspace(0.0, 1.0, 11)
    for layer in back.layers:
        assert np.all(layer["g"].evaluate(np.clip(d, *layer.interval), 0, 0) == 9.81)
    scales = Scales.geophysical(2.0)
    other = layered(BOX).converted(scales)
    assert other.geometry.domain == BOX.scaled(0.5)
    path = written(tmp_path, other, "geo.nc", grid=flat.Grid(other.geometry.domain, 1))
    check_model_file(path, other)
    assert read_model(path).scales == scales
    si = read_model(path, scales=Scales.SI)
    assert si.geometry.domain == BOX and si.scales == Scales.SI


def test_a_rule_in_the_logarithm_keeps_the_flat_schema(tmp_path):
    """The viscosity of a flat model built from a table of depths, joined
    in its logarithm, states log-polynomial under the one flat schema and
    comes back the same function, the model with its behaviours."""
    from planetmodel import Elastic, Viscoelastic, model_from_deck, sample_deck
    box = flat.Box([(0.0, 400e3), (0.0, 300e3)])
    deck = sample_deck(flat.CrustOverMantle(box, viscosity=1e20),
                       ("rho", "vp", "vs", "viscosity"))
    eta = np.array(deck.columns["viscosity"], dtype=float)
    held = np.isfinite(eta) & (eta > 0)
    eta[held] = 1e20 * 10.0 ** np.linspace(0.0, 2.0, int(held.sum()))
    deck = dataclasses.replace(deck, columns={**deck.columns, "viscosity": eta})
    model = model_from_deck(deck, behaviours=[Elastic, Viscoelastic],
                            kinds={"viscosity": "log-linear"}, coordinate=flat.DEPTH,
                            domain=box)
    path = written(tmp_path, model, grid=flat.Grid(box, 1), fields=["viscosity", "rho"])
    check_model_file(path, model)
    with netCDF4.Dataset(path) as ds:
        assert ds["part_0"]["viscosity"].radial_interpolation == "log-polynomial"
    back = read_model(path)
    assert type(back) is type(model)
    assert back.behaviours == ("Elastic", "Viscoelastic")
    for layer in model.layers:
        if "viscosity" not in layer:
            continue
        d = np.linspace(*layer.interval, 17)
        assert np.allclose(back.layer(layer.index)["viscosity"](d),
                           layer["viscosity"](d), rtol=1e-12, atol=0)


@pytest.mark.parametrize("domain", [BOX, RECT], ids=repr)
def test_a_model_of_depth_alone_is_written_with_no_grid(tmp_path, same_netcdf,
                                                        domain):
    """A flat model whose fields are all of depth alone, under the
    identity, needs no grid: the file is the one the grid of one node over
    its domain writes, to the bit, and reads back to rounding; a field
    that varies laterally, and a relief, are refused by name."""
    model = layered(domain).with_uniform_gravity(9.8)
    path = written(tmp_path, model)
    check_model_file(path, model)
    one = flat.Grid(domain, 1)
    assert radial_grid(model) == one
    same_netcdf(path, written(tmp_path, model, "given.nc", grid=one))
    data = read_model_file(path)
    assert len(data.parts) == 1 and data.parts[0].grid == one
    assert all(var.values.ndim == 1 for var in data.parts[0].variables.values())
    back = read_model(path)
    assert back.geometry.domain == domain
    assert max(worst(model, back).values()) < 1e-13
    assert [lay.name for lay in back.layers] == ["crust", "mantle"]
    grid = flat.Grid(domain, 8)
    lateral = three_d(domain, grid, rectangle=domain is RECT)
    with pytest.raises(ValueError, match=r"'vs' on layer 0 \('crust'\) depends on "
                                         r"\(x, y\).*of the depth alone"):
        written(tmp_path, lateral, "x.nc")
    alone = written(tmp_path, lateral, "names.nc", fields=["rho", "vp"])
    assert read_model(alone).field_names() == ("rho", "vp")
    assert read_model_file(alone).parts[0].grid == one
    relief = flat.GriddedFunction.sampled(dem, grid)
    with pytest.raises(ValueError, match="the model is placed by"):
        written(tmp_path, model.stretched(
            flat.layer_linear(model.skeleton, [relief, None, None])), "x.nc")
    assert not (tmp_path / "x.nc").exists()


class Creeping:
    """A behaviour of a program's own over a flat model."""

    def viscous_layers(self):
        return self.layers_with("viscosity")


class Lithosphere(Elastic, Creeping, Viscoelastic, flat.Model):
    """A flat model type of a program's own."""


def test_a_flat_model_read_back_has_its_behaviours(tmp_path):
    """A flat file names the behaviours of the model written and the model
    read back has them over `cartesian.Model`; a name the program lacks is
    kept with a warning, and `model_type` gives the caller's own type."""
    box = flat.Box([(0.0, 400e3), (0.0, 300e3)])
    model = flat.CrustOverMantle(box, viscosity=1e20)
    names = ("Elastic", "ConstantQ", "Viscoelastic")
    assert model.behaviours == names
    path = written(tmp_path, model)
    check_model_file(path, model)
    with netCDF4.Dataset(path) as ds:
        assert ds.getncattr("behaviours") == " ".join(names)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        back = read_model(path)
    assert type(back) is model_type(flat.Model, (Elastic, ConstantQ, Viscoelastic))
    assert isinstance(back, flat.Model) and not isinstance(back, SelfGravitating)
    assert back.behaviours == names
    check_model(back)
    assert max(worst(model, back).values()) < 1e-13
    omega = 2.0 * np.pi / 3.0e10
    frozen, want = back.frozen(omega), model.frozen(omega)
    d = np.linspace(*model.layer(-1).interval, 9)
    for i in range(model.nlayers):
        assert back.is_viscoelastic(i) == model.is_viscoelastic(i)
        assert back.is_fluid(i) == model.is_fluid(i)
    for name, field in want.layer(-1).fields.items():
        assert np.allclose(frozen.layer(-1)[name](d), field(d), rtol=1e-12), name
    for name, field in model.moduli(-1).items():
        assert np.allclose(back.moduli(-1)[name](d), field(d), rtol=1e-12), name
    assert back.elastic().field_names() == model.elastic().field_names()

    own = Lithosphere(model.geometry, [layer.fields for layer in model.layers])
    path = written(tmp_path, own, "own.nc")
    check_model_file(path, own)
    with pytest.warns(UserWarning, match="names the behaviour 'Creeping'"):
        back = read_model(path)
    assert type(back) is model_type(flat.Model, (Elastic, Viscoelastic))
    assert back.behaviours == ("Elastic", "Creeping", "Viscoelastic")
    with netCDF4.Dataset(written(tmp_path, back, "again.nc")) as ds:
        assert ds.getncattr("behaviours") == "Elastic Creeping Viscoelastic"
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mine = read_model(path, model_type=Lithosphere)
        plain = read_model(path, model_type=())
    assert type(mine) is Lithosphere and mine.viscous_layers() == (2,)
    assert type(plain) is flat.Model and plain.behaviours == ()
    from planetmodel import Model
    with pytest.raises(TypeError, match=r"holds a planetmodel\.cartesian\.model\.Model"
                                        ".*is not a model type over it"):
        read_model(path, model_type=Model)
    with netCDF4.Dataset(path, "a") as ds:
        ds.delncattr("behaviours")
    assert type(read_model(path)) is flat.Model


def test_fields_on_interfaces_come_back_as_flat_gridded_functions(tmp_path):
    """A kernel on the surface and on the Moho, on a grid of its own, read
    back by interface name as `cartesian.GriddedFunction`s; slicing the
    model at the surface shows it at its own nodes."""
    from planetmodel.cartesian import slices
    model = layered(BOX)
    sgrid = flat.Grid(BOX, (10, 8))
    kernel = {"surface": flat.GriddedFunction.sampled(dem, sgrid, lateral="linear"),
              "moho": flat.GriddedFunction.sampled(lambda x, y: 2.0 * dem(x, y), sgrid,
                                                   lateral="linear")}
    path = written(tmp_path, model, grid=flat.Grid(BOX, 1),
                   surfaces={"K_shape": kernel})
    check_model_file(path, model, surfaces={"K_shape": kernel})
    read = read_surfaces(path)
    fn = read["K_shape"]["moho"]
    assert isinstance(fn, flat.GriddedFunction) and fn.grid == sgrid
    assert np.array_equal(fn.values, kernel["moho"].values)
    shown = slices.surface(model, fn, "moho")
    assert shown.grid == sgrid and np.array_equal(shown.values, fn.values)


def test_what_the_flat_file_cannot_hold_is_refused(tmp_path):
    model = layered(BOX)
    with pytest.raises(TypeError, match="not a cartesian.Grid"):
        write_model(tmp_path / "x.nc", model, grid=gauss_legendre(2))
    with pytest.raises(ValueError, match="covers the model's domain"):
        write_model(tmp_path / "x.nc", model, grid=flat.Grid(RECT, 4))
    with pytest.raises(ValueError, match="keeps for itself"):
        write_model(tmp_path / "x.nc",
                    model.with_field(0, "depth", model.layer(0)["rho"]),
                    grid=flat.Grid(BOX, 2))
    # a spherical file's variable, which a reader refuses in a flat file
    for name in ("radius", "latitude", "colatitude_weights"):
        with pytest.raises(ValueError, match=f"'{name}' is a name the model file "
                                             "keeps for itself"):
            write_model(tmp_path / "x.nc",
                        model.with_field(0, name, model.layer(0)["rho"]),
                        grid=flat.Grid(BOX, 2), fields=[name])
        with pytest.raises(ValueError, match=f"'{name}' is a name the model file "
                                             "keeps for itself"):
            write_model(tmp_path / "x.nc", model, grid=flat.Grid(BOX, 2),
                        surfaces={name: {0: lambda x, y: 1.0 + 0.0 * x + 0.0 * y}})
    from planetmodel.mapping import IdentityMapping

    class Shear(IdentityMapping):
        """A mapping that is no vertical stretch."""

        @property
        def is_identity(self) -> bool:
            return False

    sheared = model.with_mapping(Shear(), check=False)
    with pytest.raises(ValueError, match="no other flat mapping has a form"):
        write_model(tmp_path / "x.nc", sheared, grid=flat.Grid(BOX, 2))
    g1, g2 = flat.Grid(BOX, 6), flat.Grid(BOX, 8)
    apart = model.stretched(flat.layer_linear(model.skeleton, [
        flat.GriddedFunction.sampled(dem, g1), flat.GriddedFunction.sampled(dem, g2),
        None]))
    with pytest.raises(ValueError, match="one grid of their own"):
        write_model(tmp_path / "x.nc", apart, grid=g1)
    with pytest.raises(TypeError, match="function of \\(x, y\\)"):
        write_model(tmp_path / "x.nc", model, grid=g1, surfaces={"k": {0: 1.0}})
    assert not (tmp_path / "x.nc").exists()


@pytest.fixture
def damaged(tmp_path):
    """A flat file with a relief and a perturbation, and a way to damage a
    copy of it."""
    grid = flat.Grid(BOX, (8, 6))
    model = three_d(BOX, grid)
    model = model.stretched(flat.layer_linear(model.skeleton, [
        flat.GriddedFunction.sampled(dem, grid), None, None]))
    good = written(tmp_path, model, "good.nc", grid=grid)
    check_model_file(good, model)

    def damage(edit):
        bad = tmp_path / "bad.nc"
        bad.write_bytes(good.read_bytes())
        with netCDF4.Dataset(bad, "a") as ds:
            edit(ds)
        return bad
    return model, damage


def add_variable(group, name, dims, *, sizes=None):
    for dim, size in (sizes or {}).items():
        group.createDimension(dim, size)
    group.createVariable(name, "f8", dims)


def relabel_an_element(ds):
    """The first element of a layer of several given to the layer above it:
    `element_layer` holds the part's layers still, and gives that element a
    layer its bounds lie outside."""
    el = ds["part_0"]["element_layer"]
    layers = np.asarray(el[:], dtype=int)
    e = next(e for e in range(1, layers.size) if layers[e] != layers[e - 1]
             and np.sum(layers == layers[e]) > 1)
    el[e] = layers[e - 1]


@pytest.mark.parametrize("edit, message", [
    (lambda ds: ds.delncattr("coordinate_system"), "coordinate_system is None"),
    (lambda ds: ds.setncattr("local_frame", "down x y"), "local_frame is 'down x y'"),
    (lambda ds: ds.setncattr("domain", "disc"), "domain is 'disc'"),
    (lambda ds: ds.setncattr("domain_bounds", np.array([0.0, 2.0])),
     "a box states 4 bounds"),
    (lambda ds: ds.setncattr("domain_padding", np.array([-1.0, 0.0])),
     "the domain is malformed"),
    (lambda ds: ds.delncattr("domain_padding"), "domain_padding is missing"),
    (lambda ds: ds["part_0"]["x"].__setitem__(0, 0.125), "x is not the nodes"),
    (lambda ds: ds["part_0"].setncattr("grid_kind", "gauss_legendre"),
     "grid_kind is a spherical file's"),
    (lambda ds: add_variable(ds["part_0"], "latitude", ("lat",), sizes={"lat": 3}),
     "latitude is a spherical file's"),
    (lambda ds: add_variable(ds["part_0"], "rho_poles", ("node",)),
     "rho_poles holds polar values"),
    (lambda ds: add_variable(ds, "interface_radius", ("interface",)),
     "interface_radius is a spherical file's"),
    (lambda ds: ds["part_0"]["s_linear"].setncattr("lateral_interpolation",
                                                   "harmonic"),
     "states no known lateral rule"),
    (lambda ds: ds["geometry"]["boundary_displacement"].setncattr("positive", "down"),
     "positive up"),
    (lambda ds: ds["geometry"]["x"].__setitem__(1, 9.0), "geometry: x is not"),
    (lambda ds: ds.setncattr("schema", "planetmodel.model/9"), "schema is"),
    (lambda ds: ds.setncattr("schema", "planetmodel.model/1"),
     "no variable 'interface_radius'"),
    (lambda ds: ds["interface_depth"].__setitem__(1, 0.5),
     "interface_depth does not hold the 2 interfaces"),
    (lambda ds: ds.setncattr("scale_mass", np.inf), "scale_mass is missing or"),
    (lambda ds: ds["part_0"].setncattr("ngll", "four"), "ngll is 'four'"),
    (relabel_an_element, "part_0/element_layer gives element . to layer 0, which "
                         "its bounds do not lie in"),
    (lambda ds: ds["geometry"]["boundary_depth"].__setitem__(0, -0.5),
     "geometry: boundary_depth starts above the free surface"),
    (lambda ds: ds["geometry"]["boundary_depth"].__setitem__(1, 2.0),
     "geometry: boundary_depth does not increase strictly"),
    (lambda ds: ds["geometry"]["boundary_depth"].__setitem__(2, 0.4),
     "geometry: boundary_depth does not increase strictly"),
])
def test_a_damaged_flat_file_is_refused_by_name(damaged, edit, message):
    model, damage = damaged
    with pytest.raises(ValueError, match=message):
        check_model_file(damage(edit), model)


@pytest.mark.parametrize("edit, message", [
    (lambda ds: ds.setncattr("name", "another"), "names its model"),
    (lambda ds: ds.setncattr("constant_G", 1.0), "constant G"),
])
def test_the_check_holds_a_flat_file_s_name_and_constants(damaged, edit, message):
    model, damage = damaged
    with pytest.raises(AssertionError, match=message):
        check_model_file(damage(edit), model)


def test_a_rectangle_s_grid_has_no_y(tmp_path):
    model = layered(RECT)
    path = written(tmp_path, model, grid=flat.Grid(RECT, 5))
    with netCDF4.Dataset(path, "a") as ds:
        ds["part_0"].createDimension("y", 2)
    with pytest.raises(ValueError, match="one axis x"):
        read_model_file(path)


def test_xarray_opens_a_flat_file(tmp_path):
    xr = pytest.importorskip("xarray")
    grid = flat.Grid(BOX, (8, 6))
    path = written(tmp_path, three_d(BOX, grid), grid=grid)
    tree = xr.open_datatree(path)
    part = tree["part_0"]
    assert "x" in part.coords and "y" in part.coords
    assert part["s_linear"].dims == ("node", "x", "y")
    assert part["depth"].attrs["positive"] == "down"
    assert tree.attrs["schema"] == SCHEMA_FLAT and tree.attrs["domain"] == "box"
    assert np.array_equal(part["x"].values, grid.x)
