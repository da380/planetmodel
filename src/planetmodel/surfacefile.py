"""The surface file: fields on one spherical surface, static or in time.

One netCDF-4 file holds named scalar fields on one angular grid and one
surface — an ice-thickness history, a present-day topography, the
surface kernels of a measurement — in the writer's own units with its
scales recorded, CF-styled so that xarray, Panoply and ncview open it
as it stands.  The Python here is the definition of the format and the
reference the C++ reader is held to;
`docs/source/formats/surface_file.rst` describes it for consumers.  This
docstring describes the spherical file; fields on a horizontal plane of
a flat body are a file of schema `SCHEMA_FLAT`, which
`cartesian.surfacefile` describes where it differs.  What the surface
decides of the file is a `SurfaceLayout` (`SphericalSurfaceLayout`, and
the flat `cartesian.surfacefile.FlatSurfaceLayout`), asked by the writer
(`write_with`) and the readers (`read_with`), which hold the rest in
common.

**Global attributes**: `schema` (`SCHEMA`, checked by every reader),
`Conventions`, `source` (the writing code and version) and `created`
(when, ISO 8601), both required; `title` where one is given;
`scale_length`, `scale_mass`, `scale_time` and `scale_temperature`
(what one stored unit is in SI); `radius`, the radius of the surface in
the file's units, where the writer states it (absent, the fields live
on the outer surface of whatever model the reader pairs the file with);
`citation` (one work a line) and `meta_<key>` for the writer's `meta`;
and the grid's `grid_kind` and `grid_lmax`.

**The grid**, at the root: the CF coordinate variables `latitude`
(degrees north, decreasing) and `longitude` (degrees east, in
[0, 360)), and the variable `colatitude_weights(latitude)` where the
grid has weights.  One grid per file, and one surface per file.

**Time**: the optional coordinate `time(time)`, strictly increasing in
the file's units, and the optional coordinate `instant(instant)`, a
second strictly increasing set for fields tied to observation times; a
file holding a `node` or an `interval` field has `time`, and one holding
an `instant` field has `instant`.  The dimension `interval`, present
where any field needs it, is of size one less than `time`: interval k
spans `time[k]` to `time[k + 1]`.  How the stored times map onto a
calendar (years before present, say) is the writer's provenance, not
the format's.

**Fields**: one float64 variable per name, NaN marking the points a
field does not cover, its dimensions by its `time_support` attribute —
`static` `(latitude, longitude)`, `node` `(time, latitude, longitude)`,
`interval` `(interval, latitude, longitude)`, `instant`
`(instant, latitude, longitude)` — with the attributes `long_name`,
`units` (for CF tools, never read back), `dimensions` (the exponents of
`units.BASE`, where declared), `time_interpolation` (`linear` or
`previous` for a `node` field a reader may sample between nodes,
`none` otherwise, and always `none` for the other supports) and
`lateral_interpolation`, a rule the grid can carry (`lateral.check_rule`:
`harmonic` only on a grid of a shipped kind with its band).  Where the
grid lacks either pole, a field may carry `name_poles`, north then
south, over its support's leading dimension and the dimension `pole`
(two); a reader without it falls back to its lateral rule.  On a grid
that holds both poles a field carries none: the writer refuses them, and
a reader a file that has them.  Values are real; complex fields are
refused.

`write_surface` writes a file from variables (`SurfaceVariable`) and
`read_surface_file` gives one back as a `SurfaceFile` of numbers and
metadata, nothing interpolated or converted, refusing a flat file by its
schema; `validate_structure` checks a file of either kind's structure
alone, and `schema_of` reads its schema.  `read_surface` gives a file of
either kind as a function of time (`HistoryBase`), a spherical one as a
`SurfaceHistory`: each field's values
at a date by the field's own time rule — a `node` field interpolated
between its nodes (`linear`), read from the latest node not after the
date (`previous`), or served at its stored nodes alone (`none`); an
`interval` field from the interval holding the date, the last closed
at its end; an `instant` field at its stored instants alone; a
`static` field at any date.  Outside its span a `linear` or `previous`
`node` field and an `interval` field refuse the date, or hold their
nearest value where the caller asks; a `none` `node` field and an
`instant` field refuse any date that is not one of their stored times,
held or not.  A date within a tolerance of a stored time is that time:
`time_rtol` (`TIME_RTOL` by default, zero for exact matching) times the
scale of the axis, the largest of its span (its last time less its
first) and the magnitudes of its first and last times.  A date formed
by arithmetic on times of that size is a rounding error relative to
them from the time it stands for, however short the span; an axis of
one time has the magnitude of that time as its scale, and an axis whose
one time is zero a scale of zero, matching exactly.  So a date a
rounding error from a node reads that node alone under every rule
(`none` and `instant` serve it, `previous` and `interval` read from
it), and a date that close beyond an end of the span is that end; the
nearest stored time within the tolerance is taken, the earlier on a
tie.  That
arithmetic is `time_rows`, and `time_axis`, `check_time_rule` and
`leading_shape` check an axis, a support's rule and a support's shape:
whatever else in the library holds slices in time states and reads
them by these.  netCDF4 (the `netcdf` extra) is imported where a file
is opened.
"""
from __future__ import annotations

import numbers
import os
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import numpy as np
from numpy.typing import ArrayLike

from .files import written_whole
from .lateral import METHODS as LATERAL_METHODS
from .lateral import check_rule
from .modelfile import (CONVENTIONS, FileVariable, attribute_number,
                        attribute_text, grid_degrees, grid_of_group, netcdf_module,
                        put_variable, stamped_source, variable_attributes,
                        variable_values)
from .sampling import KINDS as GRID_KINDS
from .sampling import AngularGrid
from .units import BASE, TIME, Dimensions, Scales, scaled_unit

if TYPE_CHECKING:
    from .gridded import GriddedFunction

__all__ = ["SCHEMA", "SCHEMA_FLAT", "TIME_SUPPORTS", "TIME_METHODS", "EXTRAPOLATIONS",
           "TIME_RTOL", "TimeRows", "SurfaceVariable", "SurfaceFile", "SurfaceLayout",
           "SphericalSurfaceLayout", "HistoryBase", "SurfaceHistory",
           "write_surface", "write_with", "read_surface_file", "read_with",
           "read_surface", "validate_structure", "schema_of", "malformed", "time_axis",
           "check_time_rule", "leading_shape", "time_rows"]

#: The schema string every reader checks; bumped only for an incompatible change.
SCHEMA = "planetmodel.surface/1"

#: The schema of a flat surface file, fields on a horizontal plane of a
#: body of `planetmodel.cartesian` (`cartesian.surfacefile` describes it),
#: which a reader of `SCHEMA` refuses by its schema.
SCHEMA_FLAT = "planetmodel.surface/2"

#: The reader of each kind of file, named where a reader of the other kind
#: refuses one.
_READERS = {SCHEMA: "planetmodel.surfacefile.read_surface_file",
            SCHEMA_FLAT: "planetmodel.cartesian.surfacefile.read_surface_file"}

#: How a field sits in time: its dimensions follow (module docstring).
TIME_SUPPORTS = ("static", "node", "interval", "instant")

#: The time rules a `node` field may state; every other support states "none".
TIME_METHODS = ("linear", "previous", "none")

#: Names the format gives its own variables, which no field may take.
_RESERVED = ("latitude", "longitude", "colatitude_weights", "time", "instant")

#: The leading dimensions of each support's variables.
_LEADING = {"static": (), "node": ("time",), "interval": ("interval",),
            "instant": ("instant",)}

#: The coordinate variable the values of each support in time are read
#: against.
_AXIS = {"node": "time", "interval": "time", "instant": "instant"}


@dataclass(frozen=True)
class SurfaceVariable:
    """One field to write: its values on the grid, and how they sit in time.

    `values` run over the support's leading shape then the grid,
    `(ntheta, nphi)` for `static`, `(ntime, ntheta, nphi)` for `node`,
    one fewer for `interval`, `(ninstant, ntheta, nphi)` for `instant`;
    NaN marks the points the field does not cover.  `poles`, where the
    grid lacks a pole, are the field's values at the north then the
    south pole on the leading shape alone.  `dimensions` are the
    field's, in the file's scales; `meaning` becomes its `long_name`.
    """

    values: ArrayLike
    _: KW_ONLY
    support: str = "static"
    time_interpolation: str = "none"
    lateral_interpolation: str = "linear"
    dimensions: Dimensions | None = None
    meaning: str = ""
    poles: ArrayLike | None = None


@dataclass(frozen=True)
class SurfaceFile:
    """A surface file's numbers and metadata, as `read_surface_file`
    gives them: nothing is interpolated or converted."""

    schema: str
    grid: AngularGrid
    variables: Mapping[str, FileVariable]
    _: KW_ONLY
    scales: Scales = Scales.SI
    time: np.ndarray | None = None
    instant: np.ndarray | None = None
    #: The radius of the surface in the file's units, where it states one.
    radius: float | None = None
    source: str = ""
    created: str = ""
    title: str | None = None
    meta: Mapping[str, Any] = MappingProxyType({})
    citation: tuple[str, ...] = ()


class SurfaceLayout(Protocol):
    """What a surface file's surface decides, asked by the writer and the
    readers of this module, which hold the rest of the format in common
    (the time axes and rules, the fields and their attributes, the units):
    `SphericalSurfaceLayout`, of `SCHEMA`, and the flat
    `cartesian.surfacefile.FlatSurfaceLayout`, of `SCHEMA_FLAT`.  A layout
    names its schema, the grid it takes and its rules, where the surface
    is (`place`: a radius, a depth), and how the grid is stated and read
    back; it builds the file and the history the numbers are read as.
    `context` is what `validate_grid` reads of the root for the rest of
    the reading."""

    #: The schema the layout writes and reads.
    schema: str
    #: Names the format keeps for itself, which no field may take.
    reserved: tuple[str, ...]
    #: The lateral rules a field may state.
    methods: tuple[str, ...]
    #: The attribute stating where the surface is, where the writer states it.
    place: str

    def check_grid(self, grid: object) -> None:
        """Refuse, by name, a grid the layout does not take."""
        ...

    def check_rule(self, grid: Any, rule: str) -> None:
        """Refuse, by name, a lateral rule the grid cannot carry."""
        ...

    def check_place(self, value: object) -> None:
        """Refuse, by name, a place that is not one."""
        ...

    def check_poles(self, grid: Any, name: str) -> None:
        """Refuse, by name, polar values given for the field `name` on a
        grid that takes none."""
        ...

    def lateral_dims(self, grid: Any) -> tuple[str, ...]:
        """The grid's dimensions, in a variable's order."""
        ...

    def lateral_shape(self, grid: Any) -> tuple[int, ...]:
        """The grid's shape, in a variable's order."""
        ...

    def write_grid(self, root: Any, grid: Any, scales: Scales) -> None:
        """The grid's statements, dimensions and coordinate variables."""
        ...

    def validate_place(self, ds: Any, path: object) -> None:
        """The checks on the statement of the surface's place."""
        ...

    def validate_grid(self, ds: Any, path: object) -> Any:
        """The checks on the grid's statements, giving the context of the
        rest of the reading."""
        ...

    def axes(self, context: Any) -> tuple[str, ...]:
        """The grid's dimensions in the file read."""
        ...

    def read_grid(self, ds: Any, context: Any) -> Any:
        """The grid the root holds."""
        ...

    def file(self, schema: str, grid: Any, variables: Mapping[str, FileVariable],
             place: float | None, **metadata: Any) -> Any:
        """The file's numbers and metadata, `metadata` the shared ones of
        `SurfaceFile` by name."""
        ...

    def history(self, file: Any) -> HistoryBase:
        """The file as a function of time."""
        ...


class SphericalSurfaceLayout:
    """The spherical surface file's `SurfaceLayout`: the module docstring's
    angular grid, its polar values and the surface's radius."""

    schema = SCHEMA
    reserved = _RESERVED
    methods = tuple(LATERAL_METHODS)
    place = "radius"

    def check_grid(self, grid: object) -> None:
        """An `AngularGrid`."""
        if not isinstance(grid, AngularGrid):
            raise TypeError(f"expected an AngularGrid, got {type(grid).__name__}")

    def check_rule(self, grid: AngularGrid, rule: str) -> None:
        """`lateral.check_rule`."""
        check_rule(grid, rule)

    def check_place(self, value: object) -> None:
        """A positive finite radius."""
        if not (isinstance(value, numbers.Real) and np.isfinite(value)
                and value > 0.0):
            raise ValueError(f"radius must be positive and finite, got {value!r}")

    def check_poles(self, grid: AngularGrid, name: str) -> None:
        """Refused where the grid holds both poles already."""
        if all(grid.pole_nodes):
            raise ValueError(f"{name!r} gives polar values, and the grid "
                             "holds both poles already")

    def lateral_dims(self, grid: AngularGrid) -> tuple[str, ...]:
        """(latitude, longitude)."""
        return ("latitude", "longitude")

    def lateral_shape(self, grid: AngularGrid) -> tuple[int, ...]:
        """(ntheta, nphi)."""
        return (grid.ntheta, grid.nphi)

    def write_grid(self, root: Any, grid: AngularGrid, scales: Scales) -> None:
        """`latitude` and `longitude`, the grid's kind and band, and its
        colatitude weights where it has them."""
        root.createDimension("latitude", grid.ntheta)
        root.createDimension("longitude", grid.nphi)
        root.setncattr("grid_kind", grid.kind)
        if grid.lmax is not None:
            root.setncattr("grid_lmax", np.int32(grid.lmax))
        lat, lon = grid_degrees(grid)
        var = root.createVariable("latitude", "f8", ("latitude",))
        var[...] = lat
        var.units, var.standard_name = "degrees_north", "latitude"
        var.long_name = "latitude"
        var = root.createVariable("longitude", "f8", ("longitude",))
        var[...] = lon
        var.units, var.standard_name = "degrees_east", "longitude"
        var.long_name = "longitude"
        if grid.weights is not None:
            put_variable(root, "colatitude_weights", ("latitude",), grid.weights,
                         {"long_name": "colatitude quadrature weights against "
                                       "sin(theta) dtheta", "units": "1"})

    def validate_place(self, ds: Any, path: object) -> None:
        """`radius`, where stated, a positive number."""
        if "radius" in ds.ncattrs():
            radius = ds.getncattr("radius")
            if (isinstance(radius, str) or np.size(radius) != 1
                    or not np.isfinite(float(radius)) or not float(radius) > 0.0):
                _fail(path, f"radius is {radius!r}, not a positive number")

    def validate_grid(self, ds: Any, path: object) -> None:
        """`grid_kind` one of `sampling.KINDS`, and the coordinates."""
        if attribute_text(ds, "grid_kind") not in GRID_KINDS:
            _fail(path, f"grid_kind is missing or not a kind of grid, one of "
                        f"{GRID_KINDS}")
        for name in ("latitude", "longitude"):
            if name not in ds.variables or tuple(
                    ds.variables[name].dimensions) != (name,):
                _fail(path, f"no coordinate variable {name!r}")

    def axes(self, context: Any) -> tuple[str, ...]:
        """(latitude, longitude)."""
        return ("latitude", "longitude")

    def read_grid(self, ds: Any, context: Any) -> AngularGrid:
        """`modelfile.grid_of_group`."""
        return grid_of_group(ds)

    def file(self, schema: str, grid: AngularGrid,
             variables: Mapping[str, FileVariable], place: float | None,
             **metadata: Any) -> SurfaceFile:
        """A `SurfaceFile`, `place` its radius."""
        return SurfaceFile(schema, grid, variables, radius=place, **metadata)

    def history(self, file: SurfaceFile) -> SurfaceHistory:
        """A `SurfaceHistory`."""
        return SurfaceHistory(file)


#: The spherical layout, the one of every file of `SCHEMA`.
_SPHERICAL = SphericalSurfaceLayout()


def _layouts() -> Iterator[SurfaceLayout]:
    """Every layout, the spherical one first and the flat one imported
    where it is first asked for."""
    yield _SPHERICAL
    from .cartesian.surfacefile import LAYOUT
    yield LAYOUT


def _real_values(values: ArrayLike, shape: tuple[int, ...], name: str,
                 what: str) -> np.ndarray:
    """A field's numbers, real and of `shape`; NaN is allowed."""
    v = np.asarray(values)
    if np.iscomplexobj(v):
        raise TypeError(f"{name!r}: {what} is complex; a surface file holds "
                        "real values")
    try:
        return np.broadcast_to(v.astype(float, copy=False), shape)
    except ValueError:
        raise ValueError(f"{name!r}: {what} has shape {v.shape}, not {shape}"
                         ) from None


def time_axis(values: ArrayLike | None, name: str) -> np.ndarray | None:
    """A time axis checked: None where none is given, otherwise one
    dimensional, finite and strictly increasing, as float64."""
    if values is None:
        return None
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} is a one-dimensional sequence of times")
    if not np.all(np.isfinite(v)) or not np.all(np.diff(v) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return v


def check_time_rule(support: str, rule: str, *, name: str) -> None:
    """Refuse, naming the field `name`, a support not among
    `TIME_SUPPORTS`, a rule not among `TIME_METHODS`, and any rule but
    `none` for a support other than `node`."""
    if support not in TIME_SUPPORTS:
        raise ValueError(f"{name!r}: time_support {support!r} is not one "
                         f"of {TIME_SUPPORTS}")
    if rule not in TIME_METHODS:
        raise ValueError(f"{name!r}: time_interpolation {rule!r} is not one of "
                         f"{TIME_METHODS}")
    if support != "node" and rule != "none":
        raise ValueError(f"{name!r}: a {support!r} field is not sampled "
                         "between times; its time_interpolation is 'none'")


def _checked(name: str, field: SurfaceVariable, layout: SurfaceLayout) -> None:
    """The per-field statements `write_surface` refuses early."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"a field's name is a string, got {name!r}")
    if name in layout.reserved or name.endswith("_poles"):
        raise ValueError(f"{name!r} is a name the surface file keeps for itself")
    if not isinstance(field, SurfaceVariable):
        raise TypeError(f"{name!r} is not a SurfaceVariable")
    check_time_rule(field.support, field.time_interpolation, name=name)
    if field.lateral_interpolation not in layout.methods:
        raise ValueError(f"{name!r}: lateral_interpolation "
                         f"{field.lateral_interpolation!r} is not one of "
                         f"{layout.methods}")
    if field.dimensions is not None and not isinstance(field.dimensions,
                                                       Dimensions):
        raise TypeError(f"{name!r}: dimensions are a units.Dimensions")


def leading_shape(support: str, *, time: np.ndarray | None,
                  instant: np.ndarray | None, name: str) -> tuple[int, ...]:
    """The leading shape a support's values carry, its axis required: ()
    for `static`, (ntime,) for `node`, one fewer for `interval`,
    (ninstant,) for `instant`; the field `name` is named in a refusal."""
    if support == "static":
        return ()
    if support in ("node", "interval"):
        if time is None:
            raise ValueError(f"{name!r} is on the time axis, and no time is given")
        if support == "interval" and time.size < 2:
            raise ValueError(f"{name!r} is on intervals, and one time makes none")
        return (time.size,) if support == "node" else (time.size - 1,)
    if instant is None:
        raise ValueError(f"{name!r} is at instants, and no instant is given")
    return (instant.size,)


def write_surface(path: str | os.PathLike[str], grid: AngularGrid, *,
                  fields: Mapping[str, SurfaceVariable],
                  source: str | None = None, time: ArrayLike | None = None,
                  instant: ArrayLike | None = None,
                  radius: float | None = None, scales: Scales = Scales.SI,
                  title: str | None = None,
                  meta: Mapping[str, Any] | None = None,
                  citation: Sequence[str] = ()) -> Path:
    """Write `fields` on `grid` to a surface file at `path`, returned.

    `source` names the writing code and version, this library and its
    version unless given (`modelfile.stamped_source`); `time` and `instant`
    are the axes of the module docstring, given where a field needs
    one; `radius` is the surface's radius and every number is in
    `scales`' units; `meta` becomes `meta_<key>` attributes, and
    `citation` is the works cited, one line each (`model.cited`).  The
    file is written whole or not at all, with the permissions a plain
    `open(path, "w")` would give it (`files.written_whole`).  A flat
    surface file is written by `cartesian.surfacefile.write_surface`.
    """
    return write_with(path, grid, _SPHERICAL, fields=fields, source=source,
                      time=time, instant=instant, place=radius, scales=scales,
                      title=title, meta=meta, citation=citation)


def write_with(path: str | os.PathLike[str], grid: Any, layout: SurfaceLayout, *,
               fields: Mapping[str, SurfaceVariable], source: str | None = None,
               time: ArrayLike | None = None, instant: ArrayLike | None = None,
               place: float | None = None, scales: Scales = Scales.SI,
               title: str | None = None, meta: Mapping[str, Any] | None = None,
               citation: Sequence[str] = ()) -> Path:
    """Write `fields` on `grid` to a surface file of `layout` at `path`,
    returned: `write_surface`'s, the surface's `place` the layout's (a
    sphere's radius, a plane's depth)."""
    from .model import cited

    nc = netcdf_module()
    layout.check_grid(grid)
    source = stamped_source(source)
    citation = cited(citation, ())
    if not isinstance(scales, Scales):
        raise TypeError(f"expected Scales, got {type(scales).__name__}")
    if isinstance(fields, str) or not isinstance(fields, Mapping) or not fields:
        raise TypeError("fields maps names to SurfaceVariables, at least one")
    t, s = time_axis(time, "time"), time_axis(instant, "instant")
    if place is not None:
        layout.check_place(place)
    meta = dict(meta or {})
    for key, value in meta.items():
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
            raise TypeError(f"meta[{key!r}] is {value!r}; a netCDF attribute is "
                            "a string or a number")
    prepared: dict[str, tuple[SurfaceVariable, np.ndarray, np.ndarray | None]] = {}
    for name, field in fields.items():
        _checked(name, field, layout)
        try:
            layout.check_rule(grid, field.lateral_interpolation)
        except ValueError as error:
            raise ValueError(f"{name!r}: {error}") from None
        leading = leading_shape(field.support, time=t, instant=s, name=name)
        values = _real_values(field.values, leading + layout.lateral_shape(grid),
                              name, "its values")
        poles = None
        if field.poles is not None:
            layout.check_poles(grid, name)
            poles = _real_values(field.poles, leading + (2,), name, "its poles")
        prepared[name] = (field, values, poles)
    needs_interval = any(f.support == "interval" for f, _, _ in prepared.values())
    needs_pole = any(p is not None for _, _, p in prepared.values())
    path = Path(path)
    with written_whole(path) as partial:
        _write_file(nc, partial, grid, prepared, source, t, s,
                    needs_interval, needs_pole, place, scales, title, meta,
                    citation, layout)
    return path


def _write_file(nc: Any, path: Path, grid: Any,
                prepared: Mapping[str, tuple[SurfaceVariable, np.ndarray,
                                             np.ndarray | None]],
                source: str, time: np.ndarray | None,
                instant: np.ndarray | None, needs_interval: bool,
                needs_pole: bool, place: float | None, scales: Scales,
                title: str | None, meta: Mapping[str, Any],
                citation: tuple[str, ...], layout: SurfaceLayout) -> None:
    """The file of `write_with`, at a path of its own until it is whole."""
    time_unit = scaled_unit(TIME, scales)
    time_dims = np.asarray([getattr(TIME, b) for b in BASE], dtype=np.int32)
    with nc.Dataset(path, "w", format="NETCDF4") as root:
        root.setncattr("schema", layout.schema)
        root.setncattr("Conventions", CONVENTIONS)
        if title:
            root.setncattr("title", title)
        root.setncattr("source", source)
        root.setncattr("created",
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
        for base in ("length", "mass", "time", "temperature"):
            root.setncattr(f"scale_{base}", float(getattr(scales, base)))
        if place is not None:
            root.setncattr(layout.place, float(place))
        if citation:
            root.setncattr("citation", "\n".join(citation))
        for key, value in meta.items():
            root.setncattr(f"meta_{key}", value)
        layout.write_grid(root, grid, scales)
        if time is not None:
            root.createDimension("time", time.size)
            put_variable(root, "time", ("time",), time,
                         {"long_name": "time", "units": time_unit,
                          "dimensions": time_dims})
        if needs_interval:
            root.createDimension("interval", time.size - 1)
        if instant is not None:
            root.createDimension("instant", instant.size)
            put_variable(root, "instant", ("instant",), instant,
                         {"long_name": "observation times", "units": time_unit,
                          "dimensions": time_dims})
        if needs_pole:
            root.createDimension("pole", 2)
            root.setncattr("poles", "north south")
        for name, (field, values, poles) in prepared.items():
            dims = _LEADING[field.support] + layout.lateral_dims(grid)
            attrs: dict[str, Any] = {
                "long_name": field.meaning or name,
                "units": scaled_unit(field.dimensions, scales),
                "time_support": field.support,
                "time_interpolation": field.time_interpolation,
                "lateral_interpolation": field.lateral_interpolation,
            }
            if field.dimensions is not None:
                attrs["dimensions"] = np.asarray(
                    [getattr(field.dimensions, b) for b in BASE], dtype=np.int32)
            put_variable(root, name, dims, values, attrs)
            if poles is not None:
                put_variable(root, f"{name}_poles",
                             _LEADING[field.support] + ("pole",), poles,
                             {"long_name": f"{name} at the north and south poles",
                              "units": attrs["units"]})


# -- reading --------------------------------------------------------------------

def malformed(path: object, msg: str) -> NoReturn:
    """Refuse a malformed surface file, naming it and the first thing wrong."""
    raise ValueError(f"malformed surface file {path}: {msg}")


_fail = malformed


def _exponents(value: Any) -> Dimensions | None:
    """The dimensions a field's `dimensions` attribute states: four whole
    numbers, the exponents of `units.BASE` in its order; None where the
    attribute holds anything else."""
    stated = np.asarray(value)
    if (stated.dtype.kind not in "iuf" or stated.shape != (len(BASE),)
            or not np.all(stated == np.round(stated))):
        return None
    return Dimensions(**{b: int(e) for b, e in zip(BASE, stated)})


def validate_structure(source: str | os.PathLike[str] | Any) -> None:
    """Check that a file, by path or as an open netCDF4 Dataset, has the
    structure its schema defines (the module docstring's for a spherical
    surface, `cartesian.surfacefile`'s for a flat one), raising ValueError
    naming the first thing wrong."""
    if isinstance(source, (str, os.PathLike)):
        nc = netcdf_module()
        with nc.Dataset(source, "r") as ds:
            _validate(ds, source, _layouts())
    else:
        _validate(source, getattr(source, "filepath", lambda: "<dataset>")(),
                  _layouts())


def schema_of(path: str | os.PathLike[str]) -> str:
    """The schema a surface file declares, its global attribute `schema`,
    read without the rest of the file; ValueError where it declares none."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        if "schema" not in ds.ncattrs():
            _fail(path, "no schema attribute: not a planetmodel surface file")
        return str(ds.getncattr("schema"))


def _layout_of_file(ds: Any, path: object, layouts: Iterable[SurfaceLayout]
                    ) -> SurfaceLayout:
    """The layout among `layouts` of the file's schema, or the error naming
    the schemas they read and, for a file of another kind, its reader."""
    if "schema" not in ds.ncattrs():
        _fail(path, "no schema attribute: not a planetmodel surface file")
    schema = ds.getncattr("schema")
    text = isinstance(schema, str)
    seen = []
    for layout in layouts:
        if text and schema == layout.schema:
            return layout
        seen.append(repr(layout.schema))
    other = (f"; a file of {schema!r} is read by {_READERS[schema]}"
             if text and schema in _READERS else "")
    _fail(path, f"schema is {schema!r}, this reader reads "
                f"{', '.join(seen[:-1]) + ' and ' if len(seen) > 1 else ''}"
                f"{seen[-1]}{other}")


def _validate(ds: Any, path: object, layouts: Iterable[SurfaceLayout]
              ) -> tuple[SurfaceLayout, Any]:
    """The checks of `validate_structure` on an open dataset, by the layout
    among `layouts` whose schema it states; that layout, and the grid the
    file holds."""
    layout = _layout_of_file(ds, path, layouts)
    for key in ("source", "created"):
        if not (attribute_text(ds, key) or "").strip():
            _fail(path, f"{key} is missing or empty")
    for base in ("length", "mass", "time", "temperature"):
        key = f"scale_{base}"
        scale = attribute_number(ds, key)
        if scale is None or not scale > 0.0:
            _fail(path, f"{key} is missing or not positive and finite")
    layout.validate_place(ds, path)
    context = layout.validate_grid(ds, path)
    try:
        grid = layout.read_grid(ds, context)
    except (TypeError, ValueError) as error:
        _fail(path, f"the grid is malformed: {error}")
    ntime = None
    for name in ("time", "instant"):
        if name not in ds.variables:
            continue
        if tuple(ds.variables[name].dimensions) != (name,):
            _fail(path, f"{name} does not run over its own dimension")
        values = np.asarray(ds.variables[name][:], dtype=float)
        if not np.all(np.isfinite(values)) or not np.all(np.diff(values) > 0.0):
            _fail(path, f"{name} is not finite and strictly increasing")
        if name == "time":
            ntime = values.size
    if "interval" in ds.dimensions:
        if ntime is None or len(ds.dimensions["interval"]) != ntime - 1:
            _fail(path, "the interval dimension is not one shorter than time")
    if "pole" in ds.dimensions and len(ds.dimensions["pole"]) != 2:
        _fail(path, "the pole dimension does not run over the two poles")
    for name, var in ds.variables.items():
        if name in layout.reserved:
            continue
        if name.endswith("_poles"):
            base_name = name[:-len("_poles")]
            if (base_name not in ds.variables or base_name in layout.reserved
                    or base_name.endswith("_poles")):
                _fail(path, f"{name} accompanies no field {base_name!r}")
            try:
                layout.check_poles(grid, base_name)
            except ValueError as error:
                _fail(path, f"{name}: {error}")
            continue
        support = attribute_text(var, "time_support")
        if support not in TIME_SUPPORTS:
            _fail(path, f"{name} states no time support, one of {TIME_SUPPORTS}")
        want = _LEADING[support] + layout.axes(context)
        if tuple(var.dimensions) != want:
            _fail(path, f"{name} is {support!r} and does not run over {want}")
        axis = _AXIS.get(support)
        if axis is not None and axis not in ds.variables:
            _fail(path, f"{name} is {support!r}, and the file has no coordinate "
                        f"variable {axis!r}")
        lateral = attribute_text(var, "lateral_interpolation")
        if lateral not in layout.methods:
            _fail(path, f"{name} states no known lateral rule")
        try:
            layout.check_rule(grid, lateral)
        except ValueError as error:
            _fail(path, f"{name} states the lateral rule {lateral!r}, and {error}")
        rule = attribute_text(var, "time_interpolation")
        if rule not in TIME_METHODS or (support != "node" and rule != "none"):
            _fail(path, f"{name} is {support!r} and states the time rule "
                        f"{rule!r}")
        want = _LEADING[support] + ("pole",)
        if f"{name}_poles" in ds.variables and tuple(
                ds.variables[f"{name}_poles"].dimensions) != want:
            _fail(path, f"{name}_poles does not run over {want}")
        if "dimensions" in var.ncattrs() and _exponents(
                var.getncattr("dimensions")) is None:
            _fail(path, f"{name} declares dimensions that are not four exponents")
    return layout, grid


def read_surface_file(path: str | os.PathLike[str]) -> SurfaceFile:
    """A spherical surface file's numbers and metadata, its structure
    validated first; a flat file is refused by its schema, naming its
    reader (`cartesian.surfacefile.read_surface_file`)."""
    return read_with(path, (_SPHERICAL,))


def read_with(path: str | os.PathLike[str], layouts: Iterable[SurfaceLayout]) -> Any:
    """A surface file's numbers and metadata, read by the layout among
    `layouts` whose schema the file states, its structure validated
    first: the layout's file (`SurfaceFile` on the sphere)."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        layout, grid = _validate(ds, path, layouts)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        scales = Scales(**{b: float(attrs[f"scale_{b}"])
                           for b in ("length", "mass", "time", "temperature")})
        meta = {k[len("meta_"):]: v.item() if isinstance(v, np.generic) else v
                for k, v in attrs.items() if k.startswith("meta_")}
        time = (variable_values(ds.variables["time"])
                if "time" in ds.variables else None)
        instant = (variable_values(ds.variables["instant"])
                   if "instant" in ds.variables else None)
        variables = {name: FileVariable(variable_values(var),
                                        variable_attributes(var))
                     for name, var in ds.variables.items()
                     if name not in layout.reserved}
    place = attrs.get(layout.place)
    title = attrs.get("title")
    return layout.file(str(attrs["schema"]), grid, MappingProxyType(variables),
                       None if place is None else float(place),
                       scales=scales, time=time, instant=instant,
                       source=str(attrs["source"]),
                       created=str(attrs["created"]),
                       title=None if title is None else str(title),
                       meta=MappingProxyType(meta),
                       citation=tuple(str(attrs.get("citation", "")).splitlines()))


# -- the file as a function of time -----------------------------------------------

#: What a field read between nodes does with a date outside its span: refuse
#: it, or hold the nearest value; a field served at its stored times alone
#: refuses any other date either way.
EXTRAPOLATIONS = ("refuse", "hold")

#: The tolerance of a date on a stored time by default, relative to the
#: scale of the axis (module docstring).
TIME_RTOL = 1e-12

#: The leading rows a date reads and their weights, ((k0, w0), (k1, w1)):
#: the value at the date is w0 times row k0 plus w1 times row k1.
type TimeRows = tuple[tuple[int, float], tuple[int, float]]


def _stored_time_at(axis: np.ndarray, date: float, tolerance: float) -> int | None:
    """The index of the stored time nearest `date` within `tolerance` of it,
    the earlier on a tie; None where none is that close."""
    after = int(np.searchsorted(axis, date, side="right"))
    best, least = None, tolerance
    for k in (after - 1, after):
        if not 0 <= k < axis.size:
            continue
        gap = abs(float(axis[k]) - date)
        if gap <= least and not (best is not None and gap == least):
            best, least = k, gap
    return best


def _time_tolerance(axis: np.ndarray, rtol: float) -> float:
    """A date's tolerance on an axis: `rtol` times its scale, the largest
    of its span and the magnitudes of its first and last times."""
    first, last = float(axis[0]), float(axis[-1])
    return rtol * max(last - first, abs(first), abs(last))


def time_rows(support: str, rule: str, date: float | None, *,
              time: np.ndarray | None = None, instant: np.ndarray | None = None,
              extrapolation: str = "refuse", name: str = "the field",
              time_rtol: float = TIME_RTOL) -> TimeRows | None:
    """The leading rows a date reads, with their weights, by a support's
    time rule (module docstring); None for a `static` field, whose values
    are the answer whole.

    `time` is the axis of `node` and `interval` values, `instant` that of
    `instant` values; `extrapolation` is one of `EXTRAPOLATIONS`, `hold`
    clamping a date outside the span of a `linear`, `previous` or
    `interval` field to its nearer end (a `none` or `instant` field
    refuses a date not stored, held or not); `time_rtol` is the tolerance
    of a date on a stored time, relative to the scale of the axis (module
    docstring), zero matching exactly.  A date on a stored node, or within the tolerance
    of one, reads that node alone, its weight one and the other row's
    zero and the same row, so that a NaN beside it stays out.  The field
    `name` is named in a refusal.
    """
    if extrapolation not in EXTRAPOLATIONS:
        raise ValueError(f"extrapolation is one of {EXTRAPOLATIONS}, "
                         f"got {extrapolation!r}")
    if isinstance(time_rtol, bool) or not isinstance(time_rtol, numbers.Real) \
            or not (np.isfinite(time_rtol) and time_rtol >= 0.0):
        raise ValueError(f"time_rtol must be a finite number, not negative, got "
                         f"{time_rtol!r}")
    if support == "static":
        return None
    if date is None:
        raise ValueError(f"{name!r} is a {support} field and needs a date")
    date = float(date)
    hold = extrapolation == "hold"
    if support == "instant":
        k = _stored_time_at(instant, date, _time_tolerance(instant, time_rtol))
        if k is None:
            raise ValueError(f"{name!r} is held at its stored instants "
                             f"alone, and {date!r} is not one")
        return ((k, 1.0), (k, 0.0))
    axis = time
    if support == "node" and rule == "none":
        k = _stored_time_at(axis, date, _time_tolerance(axis, time_rtol))
        if k is None:
            raise ValueError(f"{name!r} states no time rule and is held "
                             f"at its stored nodes alone; {date!r} is not one")
        return ((k, 1.0), (k, 0.0))
    k = _stored_time_at(axis, date, _time_tolerance(axis, time_rtol))
    if k is not None:
        date = float(axis[k])
    if date < axis[0] or date > axis[-1]:
        if not hold:
            raise ValueError(f"{date!r} lies outside {name!r}'s span "
                             f"[{axis[0]!r}, {axis[-1]!r}]")
        date = min(max(date, float(axis[0])), float(axis[-1]))
    if support == "interval":
        k = min(int(np.searchsorted(axis, date, side="right")) - 1,
                axis.size - 2)
        return ((k, 1.0), (k, 0.0))
    if rule == "previous":
        k = int(np.searchsorted(axis, date, side="right")) - 1
        return ((k, 1.0), (k, 0.0))
    if axis.size == 1:  # a single node: the span is that date alone
        return ((0, 1.0), (0, 0.0))
    k = min(int(np.searchsorted(axis, date, side="right")) - 1, axis.size - 2)
    w = (date - float(axis[k])) / (float(axis[k + 1]) - float(axis[k]))
    if w == 0.0:  # a stored node: its own values, a NaN beside it kept out
        return ((k, 1.0), (k, 0.0))
    if w == 1.0:
        return ((k + 1, 1.0), (k + 1, 0.0))
    return ((k, 1.0 - w), (k + 1, w))


class HistoryBase:
    """A surface file as a function of time (module docstring): each
    field's values at a date, by the field's own time rule.  A leaf gives
    a field at a date as a gridded function of its grid: `SurfaceHistory`
    on the sphere, `cartesian.surfacefile.SurfaceHistory` on a plane."""

    def __init__(self, file: Any) -> None:
        self._file = file

    @property
    def file(self) -> Any:
        """The file's numbers and metadata, as read."""
        return self._file

    @property
    def grid(self) -> Any:
        """The grid every field is on."""
        return self._file.grid

    @property
    def time(self) -> np.ndarray | None:
        """The time nodes, where the file has them."""
        return self._file.time

    @property
    def instant(self) -> np.ndarray | None:
        """The observation times, where the file has them."""
        return self._file.instant

    def names(self) -> tuple[str, ...]:
        """The fields, polar values not counted among them."""
        return tuple(n for n in self._file.variables if not n.endswith("_poles"))

    def support(self, name: str) -> str:
        """How a field sits in time, one of `TIME_SUPPORTS`."""
        return str(self.variable(name).attributes["time_support"])

    def dimensions(self, name: str) -> Dimensions | None:
        """A field's dimensions, where the file declares them."""
        stated = self.variable(name).attributes.get("dimensions")
        return None if stated is None else _exponents(stated)

    def values(self, name: str, *, date: float | None = None,
               extrapolation: str = "refuse",
               time_rtol: float = TIME_RTOL) -> np.ndarray:
        """A field's values on the grid at `date`, (ntheta, nphi), by its
        own time rule (module docstring); NaN stays where the field does
        not cover a point.  `extrapolation` is `refuse` (the default) or
        `hold`, which serves the nearest value of a `linear`, `previous`
        or `interval` field outside its span; `time_rtol` is the date's
        tolerance on a stored time (`time_rows`)."""
        rows = self._rows(name, date, extrapolation, time_rtol)
        values = self.variable(name).values
        if rows is None:
            return values.copy()
        (k0, w0), (k1, w1) = rows
        return w0 * values[k0] + w1 * values[k1]

    def variable(self, name: str) -> FileVariable:
        """The variable of a field, its values as stored and its attributes;
        KeyError for polar values and for a name the file does not hold."""
        if name.endswith("_poles") or name not in self._file.variables:
            raise KeyError(f"the file holds no field {name!r}")
        return self._file.variables[name]

    def _rows(self, name: str, date: float | None, extrapolation: str,
              time_rtol: float) -> TimeRows | None:
        """The leading rows a date reads, with their weights (`time_rows`);
        None for a static field, whose values are the answer whole."""
        variable = self.variable(name)
        return time_rows(self.support(name),
                         str(variable.attributes["time_interpolation"]), date,
                         time=self._file.time, instant=self._file.instant,
                         extrapolation=extrapolation, name=name,
                         time_rtol=time_rtol)


class SurfaceHistory(HistoryBase):
    """A spherical surface file as a function of time (`HistoryBase`), a
    field at a date a `GriddedFunction` by its lateral rule."""

    def __init__(self, file: SurfaceFile) -> None:
        super().__init__(file)

    @property
    def file(self) -> SurfaceFile:
        """The file's numbers and metadata, as read."""
        return self._file

    @property
    def grid(self) -> AngularGrid:
        """The angular grid every field is on."""
        return self._file.grid

    def function(self, name: str, *, date: float | None = None,
                 extrapolation: str = "refuse", rtol: float = 1e-9,
                 time_rtol: float = TIME_RTOL) -> GriddedFunction:
        """The field at `date` as a `GriddedFunction` by its stated lateral
        rule, its stored polar values carried along where the file holds
        them; a field with NaN cannot be one.  `rtol` is the function's,
        `time_rtol` the date's (`time_rows`)."""
        from .gridded import GriddedFunction

        lateral = str(self.variable(name).attributes["lateral_interpolation"])
        poles = None
        held = self._file.variables.get(f"{name}_poles")
        if held is not None:
            rows = self._rows(name, date, extrapolation, time_rtol)
            if rows is None:
                poles = np.asarray(held.values, dtype=float)
            else:
                (k0, w0), (k1, w1) = rows
                poles = w0 * held.values[k0] + w1 * held.values[k1]
        return GriddedFunction(self.grid, self.values(name, date=date,
                                                      extrapolation=extrapolation,
                                                      time_rtol=time_rtol),
                               lateral=lateral, poles=poles, rtol=rtol)


def read_surface(path: str | os.PathLike[str]) -> HistoryBase:
    """A surface file as a function of time, its structure validated
    first: a `SurfaceHistory` of a spherical file, or a
    `cartesian.surfacefile.SurfaceHistory` of a flat one, by what its
    schema states."""
    file = read_with(path, _layouts())
    for layout in _layouts():
        if file.schema == layout.schema:
            return layout.history(file)
    raise ValueError(f"no layout reads schema {file.schema!r}")
