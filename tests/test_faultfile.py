"""The fault file: written, read back, refused.

Faults of planes and structured patches in both frames, static and on
time nodes, under every slip rule and time rule, round-trip to the same
evaluations at random (u, v, t) (`check_fault_file`); what the format
cannot hold is refused on writing; a malformed file, and a file naming
anything the schema does not, is refused on reading, naming what is
wrong; and `ncdump -h` of the format page's examples is what the page
shows.
"""
import re
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest

from planetmodel import Scales
from planetmodel.fault import (CartesianFrame, Fault, GeographicFrame, Plane, Slip,
                               StructuredPatch)
from planetmodel.faultfile import (SCHEMA, FaultFile, malformed, read_fault,
                                   read_fault_file, schema_of, validate_structure,
                                   write_fault)
from planetmodel.testing import check_fault_file

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

R = 6371e3
EARTH = GeographicFrame(R)
LOCAL = CartesianFrame(surface=1.5)
PAGE = Path(__file__).resolve().parents[1] / "docs" / "source" / "formats" \
    / "fault_file.rst"


def _grid(n_u: int, n_v: int) -> tuple[np.ndarray, np.ndarray]:
    return np.meshgrid(np.linspace(-0.5, 0.5, n_u), np.linspace(-0.5, 0.5, n_v),
                       indexing="ij")


def _bent(frame, n_u: int, n_v: int, slip: Slip, interpolation: str
          ) -> StructuredPatch:
    """A structured patch bent out of a plane, in either frame."""
    U, V = _grid(n_u, n_v)
    if isinstance(frame, GeographicFrame):
        base = Plane.at_depth(frame, 34.2, 139.7, 30e3, strike=210.0, dip=25.0,
                              length=60e3, width=40e3,
                              slip=Slip(np.zeros((1, 1, 3))))
        bump = np.stack([0.02 * U * V, 0.03 * U ** 2, -4e3 * V ** 2 + 1e3 * U],
                        axis=-1)
        return StructuredPatch(frame, base.position(U, V) + bump, slip=slip,
                               interpolation=interpolation)
    pos = np.stack(np.broadcast_arrays(3.0 * U, 2.0 * V,
                                       -4.0 - 0.3 * V ** 2 + 0.2 * U * V), -1)
    return StructuredPatch(frame, pos, slip=slip, interpolation=interpolation)


def _faults() -> dict[str, tuple[Fault, dict]]:
    """Faults of every kind, frame, support and rule, with what they are
    written with."""
    rng = np.random.default_rng(3)
    time = np.array([0.0, 1.0, 2.5, 7.0])
    out = {}
    plane = Plane.at_depth(EARTH, 38.3, 142.4, 20e3, strike=195.0, dip=14.0,
                           length=400e3, width=160e3,
                           slip=Slip.from_rake(rng.uniform(60, 120, (5, 3)),
                                               rng.uniform(0, 30, (5, 3))))
    out["static_plane"] = (Fault([plane]), {})
    sill = Plane(LOCAL, [0.0, 0.0, -1.0], strike=30.0, dip=0.0, length=2.0,
                 width=2.0, slip=Slip.from_rake(0.0, 0.0, opening=0.4))
    out["cartesian_sill"] = (Fault([sill], scales=Scales(length=1e3)),
                             {"names": ["sill"]})
    for rule in ("linear", "previous", "none"):
        a = Plane.at_depth(EARTH, 35.0, 140.0, 20e3, strike=200.0, dip=15.0,
                           length=100e3, width=50e3,
                           slip=Slip(rng.normal(size=(4, 3, 2, 3)), rule="linear"))
        b = _bent(EARTH, 6, 5, Slip(rng.normal(size=(4, 4, 5, 3)), rule="cubic"),
                  "cubic")
        c = _bent(EARTH, 3, 2, Slip(rng.normal(size=(4, 2, 2, 3))), "linear")
        out[f"node_{rule}"] = (Fault([a, b, c], support="node",
                                     time_interpolation=rule, time=time,
                                     scales=Scales.geophysical(R)),
                               {"names": ["main", None, "splay"],
                                "meta": {"event": "test", "magnitude": 9.0},
                                "citation": ("one work", "another")})
    d = _bent(LOCAL, 4, 4, Slip(rng.normal(size=(2, 3, 3, 3)), rule="cell"),
              "cubic")
    e = Plane(LOCAL, [1.0, 2.0, -3.0], strike=250.0, dip=90.0, length=2.0,
              width=1.0, slip=Slip(rng.normal(size=(2, 4, 4, 3)), rule="cubic"))
    out["cartesian_node"] = (Fault([d, e], support="node",
                                   time_interpolation="linear",
                                   time=np.array([-1.0, 4.0])), {})
    return out


@pytest.mark.parametrize("case", sorted(_faults()))
def test_round_trip(tmp_path: Path, case: str):
    fault, kw = _faults()[case]
    path = write_fault(tmp_path / f"{case}.nc", fault, source="planetmodel tests",
                       **kw)
    check_fault_file(path, fault, **kw)
    data = read_fault_file(path)
    assert isinstance(data, FaultFile) and data.schema == SCHEMA
    # the same numbers, so the same evaluations, exactly
    rng = np.random.default_rng(1)
    u, v = rng.uniform(-0.5, 0.5, (2, 40))
    back = read_fault(path)
    date = None if fault.time is None else float(fault.time[-1])
    for k in range(len(fault.patches)):
        assert np.array_equal(back.slip_vector(k, u, v, date=date, basis="ambient"),
                              fault.slip_vector(k, u, v, date=date, basis="ambient"))


def test_a_writer_stamps_this_library_as_its_source_unless_given_one(tmp_path: Path):
    """`source` is the model file's: this library and its version unless
    given."""
    import planetmodel

    fault, kw = _faults()["node_linear"]
    path = write_fault(tmp_path / "stamped.nc", fault, **kw)
    assert read_fault_file(path).source == f"planetmodel {planetmodel.__version__}"
    path = write_fault(tmp_path / "named.nc", fault, source="my code 1.0", **kw)
    assert read_fault_file(path).source == "my code 1.0"


def test_xarray_opens_it_as_it_stands(tmp_path: Path):
    xr = pytest.importorskip("xarray")
    fault, kw = _faults()["node_linear"]
    path = write_fault(tmp_path / "f.nc", fault, source="planetmodel tests", **kw)
    with xr.open_dataset(path, group="patch_1") as ds:
        assert ds["slip"].dims == ("time", "su", "sv", "component")
        assert ds["latitude"].attrs["units"] == "degrees_north"


def test_writing_refuses_what_the_format_cannot_hold(tmp_path: Path):
    p = Plane(LOCAL, [0.0, 0.0, -5.0], strike=10.0, dip=40.0, length=6.0, width=3.0,
              slip=Slip(np.zeros((2, 1, 1, 3))))
    for support, axes in (("interval", {"time": [0.0, 1.0, 2.0]}),
                          ("instant", {"instant": [0.0, 1.0]})):
        fault = Fault([p], support=support, **axes)
        with pytest.raises(ValueError, match=f"a '{support}' slip is not part of "
                                             "planetmodel.fault/1"):
            write_fault(tmp_path / "x.nc", fault, source="t")
    one = Fault([Plane(LOCAL, [0.0, 0.0, -5.0], strike=10.0, dip=40.0, length=6.0,
                       width=3.0, slip=Slip(np.zeros((1, 1, 3))))])
    for source in (" ", "", 3):
        with pytest.raises(ValueError, match="source names the writing code"):
            write_fault(tmp_path / "x.nc", one, source=source)
    with pytest.raises(ValueError, match="2 names for 1"):
        write_fault(tmp_path / "x.nc", one, source="t", names=["a", "b"])
    with pytest.raises(TypeError, match="a string or None"):
        write_fault(tmp_path / "x.nc", one, source="t", names=[3])
    with pytest.raises(TypeError, match="meta"):
        write_fault(tmp_path / "x.nc", one, source="t", meta={"k": [1, 2]})
    with pytest.raises(TypeError, match="Fault"):
        write_fault(tmp_path / "x.nc", one.patches[0], source="t")
    with pytest.raises(TypeError, match="not one string"):
        write_fault(tmp_path / "x.nc", one, source="t", citation="Smith 2020")
    with pytest.raises(ValueError, match="is one line"):
        write_fault(tmp_path / "x.nc", one, source="t", citation=("Smith\n2020",))
    assert not list(tmp_path.iterdir())


# -- reading refuses ---------------------------------------------------------------

def _written(tmp_path: Path, case: str) -> Path:
    fault, kw = _faults()[case]
    return write_fault(tmp_path / f"{case}.nc", fault, source="planetmodel tests",
                       **kw)


def _set(key, value, *, group=None, variable=None):
    def edit(ds):
        g = ds if group is None else ds.groups[group]
        (g if variable is None else g.variables[variable]).setncattr(key, value)
    return edit


def _delete(key, *, group=None, variable=None):
    def edit(ds):
        g = ds if group is None else ds.groups[group]
        (g if variable is None else g.variables[variable]).delncattr(key)
    return edit


def _store(index, value, *, variable, group=None):
    def edit(ds):
        (ds if group is None else ds.groups[group]).variables[variable][index] = value
        if variable == "y" and index == slice(None):     # flat along v too
            ds.groups[group].variables["z"][:] = value
    return edit


def _add_dimension(name, *, group=None):
    return lambda ds: (ds if group is None else ds.groups[group]).createDimension(
        name, 2)


def _add_variable(name, *, group=None):
    def edit(ds):
        g = ds if group is None else ds.groups[group]
        g.createDimension(f"{name}_n", 2)
        g.createVariable(name, "f8", (f"{name}_n",))
    return edit


def _rename(old, new, *, group):
    return lambda ds: ds.groups[group].renameVariable(old, new)


#: Malformed files: (the case written, an edit by netCDF4, where the
#: refusal says the fault lies, what it says).
REFUSALS = {
    "schema": ("static_plane", _set("schema", "planetmodel.fault/0"), "schema",
               "this reader reads planetmodel.fault/1"),
    "no_schema": ("static_plane", _delete("schema"), "schema",
                  "not a planetmodel fault file"),
    "root_attribute": ("static_plane", _set("rupture_velocity", 2.5),
                       "rupture_velocity", "planetmodel.fault/1 does not name it"),
    "no_created": ("static_plane", _delete("created"), "created",
                   "is missing or empty"),
    "scale": ("static_plane", _set("scale_time", -1.0), "scale_time",
              "is missing or not positive"),
    "scale_infinite": ("static_plane", _set("scale_length", np.inf), "scale_length",
                       "is missing or not positive and finite"),
    "frame": ("static_plane", _set("frame", "spherical"), "frame",
              "not one of geographic, cartesian"),
    "radius": ("static_plane", _set("reference_radius", 0.0), "reference_radius",
               "not a positive number"),
    "both_references": ("static_plane", _set("reference_surface", 0.0),
                        "reference_surface",
                        "not an attribute of a geographic fault file"),
    "no_surface": ("cartesian_sill", _delete("reference_surface"),
                   "reference_surface", "is missing or not a finite number"),
    "interval": ("node_linear", _set("time_support", "interval"), "time_support",
                 "is interval, which is not part of planetmodel.fault/1"),
    "instant": ("node_linear", _set("time_support", "instant"), "time_support",
                "is instant, which is not part of planetmodel.fault/1"),
    "support": ("static_plane", _set("time_support", "always"), "time_support",
                "not one of static, node"),
    "time_rule": ("static_plane", _set("time_interpolation", "linear"),
                  "time_interpolation", "is not sampled between times"),
    "patches_text": ("static_plane", _set("patches", "one"), "patches",
                     "not a positive integer"),
    "patches_more": ("static_plane", _set("patches", np.int32(2)), "patch_1",
                     "is missing: the file states 2 patches"),
    "patches_fewer": ("node_linear", _set("patches", np.int32(2)), "patch_2",
                      "is not one of the 2 patches"),
    "root_dimension": ("static_plane", _add_dimension("time"), "time",
                       "is not a dimension of a static fault file"),
    "root_variable": ("node_linear", _add_variable("depth"), "depth_n",
                      "is not a dimension of a node fault file"),
    "time_order": ("node_linear", _store(2, 0.5, variable="time"), "time",
                   "not finite and strictly increasing"),
    "time_attribute": ("node_linear", _set("calendar", "julian", variable="time"),
                       "time.calendar", "does not name it"),
    "kind": ("static_plane", _set("kind", "triangulated", group="patch_0"),
             "patch_0.kind", "not one of plane, structured"),
    "patch_attribute": ("static_plane", _set("rake", 90.0, group="patch_0"),
                        "patch_0.rake", "not an attribute of a plane patch"),
    "plane_interpolation": ("static_plane",
                            _set("interpolation", "cubic", group="patch_0"),
                            "patch_0.interpolation",
                            "not an attribute of a plane patch"),
    "name": ("static_plane", _set("name", 3.0, group="patch_0"), "patch_0.name",
             "is not text"),
    "patch_dimension": ("static_plane", _add_dimension("nu", group="patch_0"),
                        "patch_0/nu", "not a dimension of a plane patch"),
    "centroid": ("static_plane", _set("centroid", [91.0, 0.0, 1e6], group="patch_0"),
                 "patch_0.centroid", "latitude outside [-90, 90]"),
    "centroid_size": ("static_plane", _set("centroid", [1.0, 2.0], group="patch_0"),
                      "patch_0.centroid", "not three finite numbers"),
    "dip": ("static_plane", _set("dip", 95.0, group="patch_0"), "patch_0.dip",
            "not in [0, 90] degrees"),
    "width": ("static_plane", _set("width", 0.0, group="patch_0"), "patch_0.width",
              "not a positive number"),
    "strike": ("static_plane", _delete("strike", group="patch_0"), "patch_0.strike",
               "is missing or not a finite number"),
    "interpolation": ("node_linear", _set("interpolation", "quintic",
                                          group="patch_1"),
                      "patch_1.interpolation", "not one of linear, cubic"),
    "nodes": ("node_linear", _set("interpolation", "cubic", group="patch_2"),
              "patch_2", "the cubic rule needs at least 4 each way"),
    "position_missing": ("node_linear", _rename("radius", "r", group="patch_1"),
                         "patch_1/r", "not a variable of a structured patch"),
    "position_nan": ("node_linear", _store((1, 2), np.nan, variable="radius",
                                           group="patch_1"),
                     "patch_1", "control positions that are not finite"),
    "position_latitude": ("node_linear", _store((0, 0), -91.0, variable="latitude",
                                                group="patch_1"),
                          "patch_1", "latitude outside [-90, 90]"),
    "position_attribute": ("node_linear", _set("axis", "Y", group="patch_1",
                                               variable="latitude"),
                           "patch_1/latitude.axis", "does not name it"),
    "degenerate": ("cartesian_node", _store(slice(None), 0.0, variable="y",
                                            group="patch_0"),
                   "patch_0", "is degenerate"),
    "slip_missing": ("static_plane", _rename("slip", "slip_east", group="patch_0"),
                     "patch_0/slip_east", "not a variable of a plane patch"),
    "slip_components": ("static_plane", _set("components", "east north up",
                                             group="patch_0", variable="slip"),
                        "patch_0/slip.components",
                        "not \"strike_slip dip_slip opening\""),
    "slip_rule": ("static_plane", _set("rule", "nearest", group="patch_0",
                                       variable="slip"),
                  "patch_0/slip.rule", "not one of cell, linear, cubic"),
    "slip_nodes": ("static_plane", _set("rule", "cubic", group="patch_0",
                                        variable="slip"),
                   "patch_0/slip", "the cubic rule needs at least 4 nodes"),
    "slip_nan": ("node_linear", _store((1, 0, 1, 2), np.nan, variable="slip",
                                       group="patch_0"),
                 "patch_0/slip", "values that are not finite"),
    "slip_attribute": ("static_plane", _set("rake", 90.0, group="patch_0",
                                            variable="slip"),
                       "patch_0/slip.rake", "does not name it"),
}


@pytest.mark.parametrize("label", sorted(REFUSALS))
def test_a_malformed_file_is_refused(tmp_path: Path, label: str):
    case, edit, where, phrase = REFUSALS[label]
    path = _written(tmp_path, case)
    with netCDF4.Dataset(path, "a") as ds:
        edit(ds)
    with pytest.raises(ValueError) as caught:
        read_fault_file(path)
    message = str(caught.value)
    assert message.startswith(f"malformed fault file {path}: {where}: "), message
    assert phrase in message, message


def _copy(source: Path, target: Path, *, hook=None, sizes=None) -> Path:
    """A copy of a netCDF file, group by group: `hook(where, name, var)`
    gives a variable's (dimensions, values) or None to drop it; `sizes`
    maps "where/dimension" to a new length."""
    def copy(src, dst, where):
        for key in src.ncattrs():
            dst.setncattr(key, src.getncattr(key))
        for name, dim in src.dimensions.items():
            dst.createDimension(name, (sizes or {}).get(f"{where}/{name}", len(dim)))
        for name, var in src.variables.items():
            got = (tuple(var.dimensions), var[:]) if hook is None else hook(
                where, name, var)
            if got is None:
                continue
            out = dst.createVariable(name, "f8", got[0])
            out[:] = got[1]
            for key in var.ncattrs():
                if key != "_FillValue":
                    out.setncattr(key, var.getncattr(key))
        for name, group in src.groups.items():
            copy(group, dst.createGroup(name), name)
    with netCDF4.Dataset(source) as src, netCDF4.Dataset(target, "w") as dst:
        copy(src, dst, "")
    return target


def _dropping(where, name):
    def hook(w, n, var):
        return None if (w, n) == (where, name) else (tuple(var.dimensions), var[:])
    return hook


@pytest.mark.parametrize("label", ["slip_dims", "components_4", "no_slip",
                                   "no_position", "no_time"])
def test_a_file_missing_or_misshaping_a_variable_is_refused(tmp_path: Path,
                                                             label: str):
    def swapped(w, n, var):
        if (w, n) == ("patch_0", "slip"):
            return ("sv", "su", "component"), np.zeros((3, 5, 3))
        return tuple(var.dimensions), var[:]

    def wider(w, n, var):
        if (w, n) == ("patch_0", "slip"):
            return tuple(var.dimensions), np.zeros((5, 3, 4))
        return tuple(var.dimensions), var[:]

    cases = {
        "slip_dims": ("static_plane", dict(hook=swapped), "patch_0/slip",
                      "does not run over (su, sv, component)"),
        "components_4": ("static_plane",
                         dict(hook=wider, sizes={"patch_0/component": 4}),
                         "patch_0/slip", "has 4 components, not 3"),
        "no_slip": ("static_plane", dict(hook=_dropping("patch_0", "slip")),
                    "patch_0/slip", "is missing"),
        "no_position": ("node_linear", dict(hook=_dropping("patch_1", "radius")),
                        "patch_1/radius", "is missing"),
        "no_time": ("node_linear", dict(hook=_dropping("", "time")), "time",
                    "is missing: a node slip needs its time axis"),
    }
    case, kw, where, phrase = cases[label]
    path = _copy(_written(tmp_path, case), tmp_path / "copy.nc", **kw)
    with pytest.raises(ValueError) as caught:
        read_fault_file(path)
    message = str(caught.value)
    assert message.startswith(f"malformed fault file {path}: {where}: "), message
    assert phrase in message, message


def test_a_group_the_schema_does_not_name_is_refused(tmp_path: Path):
    path = _written(tmp_path, "static_plane")
    with netCDF4.Dataset(path, "a") as ds:
        ds.groups["patch_0"].createGroup("extra")
    with pytest.raises(ValueError, match="patch_0/extra: is not a group of a patch"):
        read_fault_file(path)
    path = _written(tmp_path, "static_plane")
    with netCDF4.Dataset(path, "a") as ds:
        ds.createGroup("hypocentre")
    with pytest.raises(ValueError, match="hypocentre: is not one of the 1 patches"):
        read_fault_file(path)


# -- the structure alone, the schema alone, the refusal -----------------------------

@pytest.mark.parametrize("case", sorted(_faults()))
def test_a_written_file_has_the_structure(tmp_path: Path, case: str):
    path = _written(tmp_path, case)
    assert validate_structure(path) is None
    assert validate_structure(str(path)) is None
    with netCDF4.Dataset(path, "r") as ds:
        assert validate_structure(ds) is None


@pytest.mark.parametrize("label", sorted(REFUSALS))
def test_the_structure_is_refused_as_the_reader_refuses_it(tmp_path: Path, label: str):
    """By path or as an open dataset, in the reader's words."""
    case, edit, where, phrase = REFUSALS[label]
    path = _written(tmp_path, case)
    with netCDF4.Dataset(path, "a") as ds:
        edit(ds)
    with pytest.raises(ValueError) as read:
        read_fault_file(path)
    with pytest.raises(ValueError) as validated:
        validate_structure(path)
    assert str(validated.value) == str(read.value)
    assert str(read.value).startswith(f"malformed fault file {path}: {where}: ")
    with netCDF4.Dataset(path, "r") as ds:
        with pytest.raises(ValueError) as opened:
            validate_structure(ds)
    assert str(opened.value) == str(read.value)


def test_the_schema_is_read_without_the_rest_of_the_file(tmp_path: Path):
    path = _written(tmp_path, "static_plane")
    assert schema_of(path) == SCHEMA == read_fault_file(path).schema
    with netCDF4.Dataset(path, "a") as ds:
        ds.setncattr("schema", "planetmodel.fault/0")
        ds.delncattr("created")
        ds.groups["patch_0"].delncattr("kind")
    assert schema_of(path) == "planetmodel.fault/0"
    with netCDF4.Dataset(path, "a") as ds:
        ds.delncattr("schema")
    with pytest.raises(ValueError) as caught:
        schema_of(path)
    assert str(caught.value) == (f"malformed fault file {path}: schema: is missing: "
                                 "not a planetmodel fault file")
    with pytest.raises(ValueError) as read:
        read_fault_file(path)
    assert str(read.value) == str(caught.value)


def test_malformed_is_the_refusal_of_a_file():
    with pytest.raises(ValueError) as caught:
        malformed("a.nc", "patch_0.kind: is missing")
    assert str(caught.value) == "malformed fault file a.nc: patch_0.kind: is missing"
    where = Path("b") / "c.nc"
    with pytest.raises(ValueError) as caught:
        malformed(where, "nothing in it")
    assert str(caught.value) == f"malformed fault file {where}: nothing in it"


# -- the format page's examples ----------------------------------------------------

def page_examples() -> dict[str, tuple[Fault, dict]]:
    """The faults the format page shows `ncdump -h` of: a rupture in
    metres, and afterslip in kilometres and years."""
    rupture = Plane.at_depth(EARTH, 38.3, 142.4, 20e3, strike=195.0, dip=14.0,
                             length=400e3, width=160e3,
                             slip=Slip.from_rake(np.full((20, 8), 85.0),
                                                 np.ones((20, 8))))
    km = GeographicFrame(6371.0)
    main = Plane.at_depth(km, 38.3, 142.4, 20.0, strike=195.0, dip=14.0,
                          length=400.0, width=160.0,
                          slip=Slip(np.zeros((3, 10, 5, 3)), rule="linear"))
    U, V = _grid(9, 6)
    splay = Plane.at_depth(km, 38.9, 142.9, 10.0, strike=195.0, dip=35.0,
                           length=150.0, width=40.0, slip=Slip(np.zeros((1, 1, 3))))
    bend = np.stack([0.0 * U, 0.0 * U, 2.0 * V ** 2], axis=-1)
    splay = StructuredPatch(km, splay.position(U, V) - bend,
                            slip=Slip(np.zeros((3, 4, 4, 3)), rule="cubic"))
    return {"rupture": (Fault([rupture]), {}),
            "afterslip": (Fault([main, splay], support="node",
                                time_interpolation="linear",
                                time=np.array([0.0, 0.5, 2.0]),
                                scales=Scales(length=1e3, time=3.15576e7)),
                          {"names": ["main", "splay"]})}


def _normalised(text: str) -> list[str]:
    """ncdump's lines, stripped, the creation time and the file's name set
    aside."""
    text = re.sub(r':created = "[^"]*"', ':created = "..."', text)
    text = re.sub(r"^netcdf \S+ \{", "netcdf {", text.strip())
    return [line.strip() for line in text.splitlines() if line.strip()]


def test_ncdump_of_the_examples_is_the_format_pages(tmp_path: Path):
    ncdump = shutil.which("ncdump")
    if ncdump is None:
        pytest.skip("ncdump is not installed")
    page = PAGE.read_text()
    for name, (fault, kw) in page_examples().items():
        path = write_fault(tmp_path / f"{name}.nc", fault,
                           source="planetmodel 1.6", **kw)
        dumped = subprocess.run([ncdump, "-h", str(path)], capture_output=True,
                                text=True, check=True).stdout
        block = re.search(rf"\.\. code-block:: text\n   :caption: {name}\.nc\n\n"
                          rf"((?:   .*\n|\n)+)", page)
        assert block is not None, f"the page shows no {name}.nc"
        shown = "\n".join(line[3:] for line in block.group(1).splitlines())
        assert _normalised(dumped) == _normalised(shown), name
