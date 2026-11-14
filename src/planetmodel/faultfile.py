"""The fault file: a fault, its surfaces and the slip on them in time, in
one netCDF file.

One netCDF-4 file holds one `fault.Fault`: its patches, each a map of
the parameter square (u, v) in [-1/2, 1/2] x [-1/2, 1/2] with the slip
on it, in one frame, with one time axis for every patch.  It stores
what evaluating the fault anywhere takes and no more: no quadrature
points, no cell counts for a consumer, no rigidity, no moment.  It is
CF-styled, so that ncdump, xarray and Panoply open it as it stands.
The Python here is the definition of the format and the reference the
C++ reader and writer are held to; `docs/source/formats/fault_file.rst` describes
it for consumers, and `fault`'s module docstring the conventions
(frames, the parameter square, the normal and the fault basis, the
slip in fault components and its rules).

**Global attributes**: `schema` (`SCHEMA`, checked by every reader),
`Conventions`, `source` (the writing code and version) and `created`
(ISO 8601), both required; `title`, `citation` (one work a line) and
`meta_<key>` as the surface file has them; `scale_length`,
`scale_mass`, `scale_time` and `scale_temperature` (what one stored
unit is in SI: lengths and times are in them); `frame`, one of
`FRAMES`, with `reference_radius` (the sphere of a `geographic` frame)
or `reference_surface` (the height z of a `cartesian` frame's reference
plane); `time_support`, one of `SUPPORTS`, and `time_interpolation`, by
the surface file's rules (`surfacefile.check_time_rule`); and
`patches`, their number.

**Time**: a `static` file has no time axis; a `node` file has the
coordinate `time(time)`, finite and strictly increasing in the file's
time unit, and its slip is read between nodes by `time_interpolation`
(`surfacefile.time_rows`).  The slip stored is always the cumulative
slip at a node; `interval` and `instant` supports are not part of
this version and are refused by name.

**A patch** is the group `patch_<k>`, k = 0 .. patches - 1, with the
attribute `kind` (one of `KINDS`) and an optional `name`.  A `plane`
states its parameters as attributes: `centroid` (three numbers in the
frame's coordinates), `strike` and `dip` (degrees, the dip in
[0, 90]), `length` (along strike) and `width` (down dip).  A
`structured` patch has the dimensions `nu` and `nv`, its control
positions as three variables over `(nu, nv)` (`latitude`, `longitude`
and `radius` in a geographic frame, `x`, `y` and `z` in a Cartesian
one; `POSITIONS`), and the attribute `interpolation`, one of
`fault.INTERPOLATIONS`.  Every patch holds its slip: the dimensions
`su`, `sv` and `component` (always 3), and the variable `slip` over the
support's leading dimension (none, or `time`) then `(su, sv,
component)`, the fault components (`fault.SLIP_COMPONENTS`, named in
its attribute `components`) on its own grid, with the attribute
`rule`, one of `fault.SLIP_RULES`, and `units`, `dimensions` and
`long_name`.

**Strictness.**  The reader refuses any group, dimension, variable or
attribute this module does not name (beyond netCDF's own `_FillValue`
and the free `meta_<key>`), so that an extension of the format is a
schema bump and never a silent misreading.

`write_fault` writes a fault, whole or not at all; `read_fault_file`
gives a `FaultFile` (the fault and the file's metadata), the file
checked first; `read_fault` the fault alone.  netCDF4 (the `netcdf`
extra) is imported where a file is opened.
"""
from __future__ import annotations

import numbers
import os
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Any, NoReturn

import numpy as np

from .fault import (INTERPOLATIONS, MIN_NODES, SLIP_COMPONENTS, SLIP_RULES,
                    CartesianFrame, Fault, Frame, GeographicFrame, Patch, Plane,
                    Slip, StructuredPatch)
from .files import written_whole
from .modelfile import CONVENTIONS, netcdf_module, put_variable, variable_values
from .surfacefile import check_time_rule
from .units import BASE, LENGTH, TIME, Dimensions, Scales, scaled_unit

__all__ = ["SCHEMA", "FRAMES", "SUPPORTS", "KINDS", "POSITIONS", "FaultFile",
           "write_fault", "read_fault_file", "read_fault"]

#: The schema string every reader checks; bumped only for an incompatible change.
SCHEMA = "planetmodel.fault/1"

#: The frames a fault file states: `fault.GeographicFrame`'s and
#: `fault.CartesianFrame`'s `kind`.
FRAMES = ("geographic", "cartesian")

#: The time supports of this version, of `surfacefile.TIME_SUPPORTS`.
SUPPORTS = ("static", "node")

#: The kinds of patch: `fault.Plane`'s and `fault.StructuredPatch`'s `kind`.
KINDS = ("plane", "structured")

#: A structured patch's position variables in each frame, in the order of
#: the frame's coordinates.
POSITIONS = {"geographic": ("latitude", "longitude", "radius"),
             "cartesian": ("x", "y", "z")}

#: What the slip variable's `long_name` says.
_SLIP_MEANING = "displacement of the hanging wall relative to the footwall"

#: netCDF's own attribute, admitted on any variable.
_FILL = "_FillValue"

#: The global attributes the format names, besides `meta_<key>`.
_ROOT_ATTRIBUTES = ("schema", "Conventions", "source", "created", "title",
                    "citation", "scale_length", "scale_mass", "scale_time",
                    "scale_temperature", "frame", "reference_radius",
                    "reference_surface", "time_support", "time_interpolation",
                    "patches")

#: The reference each frame states, the other refused.
_REFERENCE = {"geographic": "reference_radius", "cartesian": "reference_surface"}

#: The attributes of a patch group, by kind.
_PATCH_ATTRIBUTES = {
    "plane": ("kind", "name", "centroid", "strike", "dip", "length", "width"),
    "structured": ("kind", "name", "interpolation")}

#: The attributes of each variable.
_VARIABLE_ATTRIBUTES = {
    "time": ("long_name", "units", "dimensions", _FILL),
    "latitude": ("long_name", "units", "standard_name", _FILL),
    "longitude": ("long_name", "units", "standard_name", _FILL),
    "radius": ("long_name", "units", "dimensions", _FILL),
    "x": ("long_name", "units", "dimensions", _FILL),
    "y": ("long_name", "units", "dimensions", _FILL),
    "z": ("long_name", "units", "dimensions", _FILL),
    "slip": ("long_name", "units", "dimensions", "rule", "components", _FILL)}

#: The leading dimensions of the slip, by support.
_LEADING = {"static": (), "node": ("time",)}


def _listed(names: Sequence[str]) -> str:
    """Names as a message lists them: `a, b, c`."""
    return ", ".join(names)


@dataclass(frozen=True)
class FaultFile:
    """A fault file as `read_fault_file` gives it: the fault, the patches'
    names (None where a patch has none) and the file's metadata."""

    schema: str
    fault: Fault
    _: KW_ONLY
    names: tuple[str | None, ...] = ()
    source: str = ""
    created: str = ""
    title: str | None = None
    meta: Mapping[str, Any] = MappingProxyType({})
    citation: tuple[str, ...] = ()


# -- writing ------------------------------------------------------------------------

def write_fault(path: str | os.PathLike[str], fault: Fault, *, source: str,
                names: Sequence[str | None] | None = None,
                title: str | None = None, meta: Mapping[str, Any] | None = None,
                citation: Sequence[str] = ()) -> Path:
    """Write `fault` to a fault file at `path`, returned.

    `source` names the writing code and version; `names` gives each
    patch an optional name (a segment's label); `meta` becomes
    `meta_<key>` attributes.  Every number is written in the fault's
    `scales`.  A fault whose slip is on intervals or at instants, a frame
    or a kind of patch the format does not name, is refused.  The file
    is written whole or not at all, with the permissions a plain
    `open(path, "w")` would give it (`files.written_whole`).
    """
    nc = netcdf_module()
    if not isinstance(fault, Fault):
        raise TypeError(f"expected a Fault, got {type(fault).__name__}")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source names the writing code and version")
    if fault.support not in SUPPORTS:
        raise ValueError(f"a {fault.support!r} slip is not part of {SCHEMA}: its "
                         f"supports are {_listed(SUPPORTS)}, the slip cumulative "
                         "at each node")
    frame = getattr(fault.frame, "kind", None)
    if frame not in FRAMES:
        raise ValueError(f"the frame {fault.frame!r} is not one of {SCHEMA}'s, "
                         f"{_listed(FRAMES)}")
    patches = fault.patches
    for k, p in enumerate(patches):
        if getattr(p, "kind", None) not in KINDS:
            raise ValueError(f"patch {k} is a {type(p).__name__}, not a kind of "
                             f"{SCHEMA}: {_listed(KINDS)}")
    names = (None,) * len(patches) if names is None else tuple(names)
    if len(names) != len(patches):
        raise ValueError(f"names gives {len(names)} names for {len(patches)} patches")
    for name in names:
        if name is not None and not isinstance(name, str):
            raise TypeError(f"a patch's name is a string or None, got {name!r}")
    meta = dict(meta or {})
    for key, value in meta.items():
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
            raise TypeError(f"meta[{key!r}] is {value!r}; a netCDF attribute is "
                            "a string or a number")
    path = Path(path)
    with written_whole(path) as partial:
        _write_file(nc, partial, fault, frame, names, source, title, meta,
                    tuple(citation))
    return path


def _exponents(dims: Dimensions) -> np.ndarray:
    """Dimensions as the integer exponents a `dimensions` attribute holds."""
    return np.asarray([getattr(dims, b) for b in BASE], dtype=np.int32)


def _write_file(nc: Any, path: Path, fault: Fault, frame: str,
                names: tuple[str | None, ...], source: str, title: str | None,
                meta: Mapping[str, Any], citation: tuple[str, ...]) -> None:
    """The file of `write_fault`, at a path of its own until it is whole."""
    scales = fault.scales
    length = scaled_unit(LENGTH, scales)
    with nc.Dataset(path, "w", format="NETCDF4") as root:
        root.setncattr("schema", SCHEMA)
        root.setncattr("Conventions", CONVENTIONS)
        if title:
            root.setncattr("title", title)
        root.setncattr("source", source)
        root.setncattr("created",
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
        for base in ("length", "mass", "time", "temperature"):
            root.setncattr(f"scale_{base}", float(getattr(scales, base)))
        if citation:
            root.setncattr("citation", "\n".join(citation))
        for key, value in meta.items():
            root.setncattr(f"meta_{key}", value)
        root.setncattr("frame", frame)
        if frame == "geographic":
            root.setncattr("reference_radius", float(fault.frame.radius))
        else:
            root.setncattr("reference_surface", float(fault.frame.surface))
        root.setncattr("time_support", fault.support)
        root.setncattr("time_interpolation", fault.time_interpolation)
        root.setncattr("patches", np.int32(len(fault.patches)))
        if fault.support == "node":
            root.createDimension("time", fault.time.size)
            put_variable(root, "time", ("time",), fault.time,
                         {"long_name": "time", "units": scaled_unit(TIME, scales),
                          "dimensions": _exponents(TIME)})
        for k, (patch, name) in enumerate(zip(fault.patches, names)):
            group = root.createGroup(f"patch_{k}")
            group.setncattr("kind", patch.kind)
            if name is not None:
                group.setncattr("name", name)
            if patch.kind == "plane":
                group.setncattr("centroid", np.asarray(patch.centroid, dtype=float))
                for key in ("strike", "dip", "length", "width"):
                    group.setncattr(key, float(getattr(patch, key)))
            else:
                n_u, n_v = patch.positions.shape[:2]
                group.createDimension("nu", n_u)
                group.createDimension("nv", n_v)
                group.setncattr("interpolation", patch.interpolation)
                _write_positions(group, frame, patch.positions, length)
            slip = patch.slip
            n_u, n_v = slip.shape
            group.createDimension("su", n_u)
            group.createDimension("sv", n_v)
            group.createDimension("component", 3)
            values = slip.values if fault.support == "node" else slip.values[0]
            put_variable(group, "slip", _LEADING[fault.support] + ("su", "sv",
                                                                   "component"),
                         values, {"long_name": _SLIP_MEANING, "units": length,
                                  "dimensions": _exponents(LENGTH),
                                  "rule": slip.rule,
                                  "components": " ".join(SLIP_COMPONENTS)})


def _write_positions(group: Any, frame: str, positions: np.ndarray,
                     length: str) -> None:
    """A structured patch's control positions, one variable a coordinate."""
    lengths = {"long_name": None, "units": length, "dimensions": _exponents(LENGTH)}
    attributes = {
        "latitude": {"long_name": "latitude of the control positions",
                     "units": "degrees_north", "standard_name": "latitude"},
        "longitude": {"long_name": "longitude of the control positions",
                      "units": "degrees_east", "standard_name": "longitude"},
        "radius": lengths | {"long_name": "radius of the control positions"},
        "x": lengths | {"long_name": "x (east) of the control positions"},
        "y": lengths | {"long_name": "y (north) of the control positions"},
        "z": lengths | {"long_name": "z (up) of the control positions"}}
    for i, name in enumerate(POSITIONS[frame]):
        put_variable(group, name, ("nu", "nv"), positions[..., i], attributes[name])


# -- reading ------------------------------------------------------------------------

def _fail(path: object, where: str, msg: str) -> NoReturn:
    """The refusal of a file, naming the place of the fault."""
    raise ValueError(f"malformed fault file {path}: "
                     f"{f'{where}: ' if where else ''}{msg}")


def _text(value: object) -> str | None:
    """An attribute's text, or None where it is not text."""
    return value if isinstance(value, str) else None


def _numbers(value: object) -> np.ndarray | None:
    """An attribute's numbers as float64, or None where it is text."""
    if isinstance(value, str):
        return None
    out = np.atleast_1d(np.asarray(value))
    if out.dtype.kind not in "iuf":
        return None
    return out.astype(float)


def _number(value: object) -> float | None:
    """An attribute's one number, or None where it is text or many."""
    out = _numbers(value)
    return None if out is None or out.size != 1 else float(out[0])


def _count(value: object) -> int | None:
    """An attribute's one integer, or None where it is not one."""
    v = None if isinstance(value, str) else np.atleast_1d(np.asarray(value))
    if v is None or v.size != 1 or v.dtype.kind not in "iu":
        return None
    return int(v[0])


def _unnamed(path: object, where: str, present: Sequence[str],
             named: Sequence[str], what: str) -> None:
    """Refuse the first of `present` that is not among `named`."""
    for key in present:
        if key not in named:
            _fail(path, f"{where}{key}", f"is not {what}: {SCHEMA} does not "
                                         "name it")


def read_fault_file(path: str | os.PathLike[str]) -> FaultFile:
    """A fault file's fault and metadata, every check of the module
    docstring made first, the first fault named."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        return _read(ds, path)


def read_fault(path: str | os.PathLike[str]) -> Fault:
    """The fault of a fault file (`read_fault_file`)."""
    return read_fault_file(path).fault


def _read(ds: Any, path: object) -> FaultFile:
    """The fault and metadata of an open file, checked in the order the
    C++ reader checks them."""
    attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
    schema = _text(attrs.get("schema"))
    if schema is None:
        _fail(path, "schema", "is missing: not a planetmodel fault file")
    if schema != SCHEMA:
        _fail(path, "schema", f"is {schema}, this reader reads {SCHEMA}")
    _unnamed(path, "", [k for k in attrs if not k.startswith("meta_")],
             _ROOT_ATTRIBUTES, "an attribute of a fault file")
    for key in ("source", "created"):
        text = _text(attrs.get(key))
        if text is None or not text.strip():
            _fail(path, key, "is missing or empty")
    for key in ("title", "citation"):
        if key in attrs and _text(attrs[key]) is None:
            _fail(path, key, "is not text")
    scales = {}
    for base in ("length", "mass", "time", "temperature"):
        value = _number(attrs.get(f"scale_{base}"))
        if value is None or not value > 0.0:
            _fail(path, f"scale_{base}", "is missing or not positive")
        scales[base] = value
    frame_name = _text(attrs.get("frame"))
    if frame_name not in FRAMES:
        _fail(path, "frame", f"is missing or not one of {_listed(FRAMES)}")
    frame = _frame(path, frame_name, attrs)
    support = _text(attrs.get("time_support"))
    if support in ("interval", "instant"):
        _fail(path, "time_support", f"is {support}, which is not part of {SCHEMA}: "
                                    f"its supports are {_listed(SUPPORTS)}")
    if support not in SUPPORTS:
        _fail(path, "time_support", f"is missing or not one of {_listed(SUPPORTS)}")
    rule = _text(attrs.get("time_interpolation"))
    try:
        check_time_rule(support, "" if rule is None else rule, name="slip")
    except ValueError as error:
        _fail(path, "time_interpolation", str(error))
    count = _count(attrs.get("patches"))
    if count is None or count < 1:
        _fail(path, "patches", "is missing or not a positive integer")
    leading = _LEADING[support]
    _unnamed(path, "", list(ds.dimensions), leading,
             f"a dimension of a {support} fault file")
    _unnamed(path, "", list(ds.variables), leading,
             f"a variable of a {support} fault file")
    time = None
    if support == "node":
        if "time" not in ds.variables:
            _fail(path, "time", "is missing: a node slip needs its time axis")
        var = ds.variables["time"]
        if tuple(var.dimensions) != ("time",):
            _fail(path, "time", "does not run over its own dimension")
        _unnamed(path, "time.", var.ncattrs(), _VARIABLE_ATTRIBUTES["time"],
                 "an attribute of the time axis")
        time = variable_values(var)
        if not np.all(np.isfinite(time)) or not np.all(np.diff(time) > 0.0):
            _fail(path, "time", "is not finite and strictly increasing")
    expected = [f"patch_{k}" for k in range(count)]
    _unnamed(path, "", list(ds.groups), expected, f"one of the {count} patches")
    for name in expected:
        if name not in ds.groups:
            _fail(path, name, f"is missing: the file states {count} patches")
    patches, names = [], []
    nslices = 1 if time is None else time.size
    for k, group_name in enumerate(expected):
        patch, name = _patch(path, group_name, ds.groups[group_name], frame,
                             frame_name, leading, nslices)
        patches.append(patch)
        names.append(name)
    fault = Fault(patches, support=support, time_interpolation=rule, time=time,
                  scales=Scales(**scales))
    meta = {k[len("meta_"):]: v.item() if isinstance(v, np.generic) else v
            for k, v in attrs.items() if k.startswith("meta_")}
    title = attrs.get("title")
    return FaultFile(schema, fault, names=tuple(names), source=str(attrs["source"]),
                     created=str(attrs["created"]),
                     title=None if title is None else str(title),
                     meta=MappingProxyType(meta),
                     citation=tuple(str(attrs.get("citation", "")).splitlines()))


def _frame(path: object, name: str, attrs: Mapping[str, Any]) -> Frame:
    """The frame the root states, its reference checked and the other
    frame's refused."""
    other = _REFERENCE["cartesian" if name == "geographic" else "geographic"]
    if other in attrs:
        _fail(path, other, f"is not an attribute of a {name} fault file")
    key = _REFERENCE[name]
    value = _number(attrs.get(key))
    if name == "geographic":
        if value is None or not (np.isfinite(value) and value > 0.0):
            _fail(path, key, "is missing or not a positive number")
        return GeographicFrame(value)
    if value is None or not np.isfinite(value):
        _fail(path, key, "is missing or not a finite number")
    return CartesianFrame(surface=value)


def _geographic_positions(c: np.ndarray) -> bool:
    """Whether positions (..., 3) have latitudes in [-90, 90] and positive
    radii."""
    return bool(np.all(np.abs(c[..., 0]) <= 90.0) and np.all(c[..., 2] > 0.0))


def _patch(path: object, where: str, group: Any, frame: Frame, frame_name: str,
           leading: tuple[str, ...], nslices: int) -> tuple[Patch, str | None]:
    """One patch group read and checked: the patch and its name."""
    attrs = {k: group.getncattr(k) for k in group.ncattrs()}
    _unnamed(path, f"{where}/", list(group.groups), (), "a group of a patch")
    kind = _text(attrs.get("kind"))
    if kind not in KINDS:
        _fail(path, f"{where}.kind", f"is missing or not one of {_listed(KINDS)}")
    _unnamed(path, f"{where}.", list(attrs), _PATCH_ATTRIBUTES[kind],
             f"an attribute of a {kind} patch")
    name = attrs.get("name")
    if name is not None and _text(name) is None:
        _fail(path, f"{where}.name", "is not text")
    dims = ("su", "sv", "component") + (("nu", "nv") if kind == "structured" else ())
    _unnamed(path, f"{where}/", list(group.dimensions), dims,
             f"a dimension of a {kind} patch")
    variables = ("slip",) + (POSITIONS[frame_name] if kind == "structured" else ())
    _unnamed(path, f"{where}/", list(group.variables), variables,
             f"a variable of a {kind} patch")
    plane, positions, interpolation = None, None, None
    if kind == "plane":
        plane = _plane_parameters(path, where, attrs, frame_name)
    else:
        interpolation = _text(attrs.get("interpolation"))
        if interpolation not in INTERPOLATIONS:
            _fail(path, f"{where}.interpolation",
                  f"is missing or not one of {_listed(INTERPOLATIONS)}")
        positions = _positions(path, where, group, frame_name, interpolation)
    slip = _slip(path, where, group, leading, nslices)
    if kind == "plane":
        return Plane(frame, plane["centroid"], strike=plane["strike"],
                     dip=plane["dip"], length=plane["length"],
                     width=plane["width"], slip=slip), name
    try:
        patch = StructuredPatch(frame, positions, slip=slip,
                                interpolation=interpolation)
    except ValueError:
        _fail(path, where, "is degenerate: its area element is not positive at "
                           "every control node")
    return patch, name


def _plane_parameters(path: object, where: str, attrs: Mapping[str, Any],
                      frame_name: str) -> dict[str, Any]:
    """A plane's parameters, checked."""
    centroid = _numbers(attrs.get("centroid"))
    if centroid is None or centroid.size != 3 or not np.all(np.isfinite(centroid)):
        _fail(path, f"{where}.centroid", "is missing or not three finite numbers")
    if frame_name == "geographic" and not _geographic_positions(centroid):
        _fail(path, f"{where}.centroid", "has a latitude outside [-90, 90] or a "
                                         "radius that is not positive")
    out: dict[str, Any] = {"centroid": centroid}
    strike = _number(attrs.get("strike"))
    if strike is None or not np.isfinite(strike):
        _fail(path, f"{where}.strike", "is missing or not a finite number")
    dip = _number(attrs.get("dip"))
    if dip is None or not 0.0 <= dip <= 90.0:
        _fail(path, f"{where}.dip", "is missing or not in [0, 90] degrees")
    out["strike"], out["dip"] = strike, dip
    for key in ("length", "width"):
        value = _number(attrs.get(key))
        if value is None or not (np.isfinite(value) and value > 0.0):
            _fail(path, f"{where}.{key}", "is missing or not a positive number")
        out[key] = value
    return out


def _positions(path: object, where: str, group: Any, frame_name: str,
               interpolation: str) -> np.ndarray:
    """A structured patch's control positions, checked."""
    n_u, n_v = (len(group.dimensions[d]) if d in group.dimensions else 0
                for d in ("nu", "nv"))
    least = MIN_NODES[interpolation]
    if min(n_u, n_v) < least:
        _fail(path, where, f"has {n_u} by {n_v} control nodes, and the "
                           f"{interpolation} rule needs at least {least} each way")
    columns = []
    for name in POSITIONS[frame_name]:
        if name not in group.variables:
            _fail(path, f"{where}/{name}", "is missing")
        var = group.variables[name]
        if tuple(var.dimensions) != ("nu", "nv"):
            _fail(path, f"{where}/{name}", "does not run over (nu, nv)")
        _unnamed(path, f"{where}/{name}.", var.ncattrs(), _VARIABLE_ATTRIBUTES[name],
                 "an attribute of a control position")
        columns.append(variable_values(var))
    positions = np.stack(columns, axis=-1)
    if not np.all(np.isfinite(positions)):
        _fail(path, where, "has control positions that are not finite")
    if frame_name == "geographic" and not _geographic_positions(positions):
        _fail(path, where, "has a control position with a latitude outside "
                           "[-90, 90] or a radius that is not positive")
    return positions


def _slip(path: object, where: str, group: Any, leading: tuple[str, ...],
          nslices: int) -> Slip:
    """A patch's slip, checked."""
    at = f"{where}/slip"
    if "slip" not in group.variables:
        _fail(path, at, "is missing")
    var = group.variables["slip"]
    want = leading + ("su", "sv", "component")
    if tuple(var.dimensions) != want:
        _fail(path, at, f"does not run over ({_listed(want)})")
    if len(group.dimensions["component"]) != 3:
        _fail(path, at, f"has {len(group.dimensions['component'])} components, "
                        "not 3")
    _unnamed(path, f"{at}.", var.ncattrs(), _VARIABLE_ATTRIBUTES["slip"],
             "an attribute of the slip")
    attrs = {k: var.getncattr(k) for k in var.ncattrs()}
    if _text(attrs.get("components")) != " ".join(SLIP_COMPONENTS):
        _fail(path, f"{at}.components",
              f"is missing or not \"{' '.join(SLIP_COMPONENTS)}\"")
    rule = _text(attrs.get("rule"))
    if rule not in SLIP_RULES:
        _fail(path, f"{at}.rule", f"is missing or not one of {_listed(SLIP_RULES)}")
    n_u, n_v = len(group.dimensions["su"]), len(group.dimensions["sv"])
    least = MIN_NODES[rule]
    if min(n_u, n_v) < least:
        _fail(path, at, f"has {n_u} by {n_v} values, and the {rule} rule needs at "
                        f"least {least} {'cells' if rule == 'cell' else 'nodes'} "
                        "each way")
    values = variable_values(var)
    if not np.all(np.isfinite(values)):
        _fail(path, at, "has values that are not finite: slip is defined "
                        "everywhere on its patch")
    return Slip(values.reshape((nslices, n_u, n_v, 3)), rule=rule)
