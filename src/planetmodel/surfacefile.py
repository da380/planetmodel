"""The surface file: fields on one spherical surface, static or in time.

One netCDF-4 file holds named scalar fields on one angular grid and one
surface — an ice-thickness history, a present-day topography, the
surface kernels of a measurement — in the writer's own units with its
scales recorded, CF-styled so that xarray, Panoply and ncview open it
as it stands.  The Python here is the definition of the format and the
reference the C++ reader is held to;
`docs/source/formats/surface_file.rst` describes it for consumers.

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
second strictly increasing set for fields tied to observation times.
The dimension `interval`, present where any field needs it, is of size
one less than `time`: interval k spans `time[k]` to `time[k + 1]`.  How
the stored times map onto a calendar (years before present, say) is the
writer's provenance, not the format's.

**Fields**: one float64 variable per name, NaN marking the points a
field does not cover, its dimensions by its `time_support` attribute —
`static` `(latitude, longitude)`, `node` `(time, latitude, longitude)`,
`interval` `(interval, latitude, longitude)`, `instant`
`(instant, latitude, longitude)` — with the attributes `long_name`,
`units` (for CF tools, never read back), `dimensions` (the exponents of
`units.BASE`, where declared), `time_interpolation` (`linear` or
`previous` for a `node` field a reader may sample between nodes,
`none` otherwise, and always `none` for the other supports) and
`lateral_interpolation`.  Where the grid lacks either pole, a field
may carry `name_poles`, north then south, on its leading time shape;
a reader without it falls back to its lateral rule.  Values are real;
complex fields are refused.

`write_surface` writes a file from `SurfaceVariable`s and
`read_surface_file` gives one back as a `SurfaceFile` of numbers and
metadata, nothing interpolated or converted; `validate_structure`
checks a file's structure alone.  `read_surface` gives a
`SurfaceHistory`, the file as a function of time: each field's values
at a date by the field's own time rule — a `node` field interpolated
between its nodes (`linear`), read from the latest node not after the
date (`previous`), or served at its stored nodes alone (`none`); an
`interval` field from the interval holding the date, the last closed
at its end; an `instant` field at its stored instants alone; a
`static` field at any date.  Outside its span a field refuses the
date, or holds its nearest value where the caller asks.  netCDF4 (the
`netcdf` extra) is imported where a file is opened.
"""
from __future__ import annotations

import numbers
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike

from .lateral import METHODS as LATERAL_METHODS
from .modelfile import (CONVENTIONS, FileVariable, grid_degrees, grid_of_group,
                        netcdf_module, put_variable, variable_attributes,
                        variable_values)
from .sampling import KINDS as GRID_KINDS
from .sampling import AngularGrid
from .units import BASE, TIME, Dimensions, Scales, scaled_unit

if TYPE_CHECKING:
    from .gridded import GriddedFunction

__all__ = ["SCHEMA", "TIME_SUPPORTS", "TIME_METHODS", "SurfaceVariable",
           "SurfaceFile", "SurfaceHistory", "write_surface",
           "read_surface_file", "read_surface", "validate_structure"]

#: The schema string every reader checks; bumped only for an incompatible change.
SCHEMA = "planetmodel.surface/1"

#: How a field sits in time: its dimensions follow (module docstring).
TIME_SUPPORTS = ("static", "node", "interval", "instant")

#: The time rules a `node` field may state; every other support states "none".
TIME_METHODS = ("linear", "previous", "none")

#: Names the format gives its own variables, which no field may take.
_RESERVED = ("latitude", "longitude", "colatitude_weights", "time", "instant")

#: The leading dimensions of each support's variables.
_LEADING = {"static": (), "node": ("time",), "interval": ("interval",),
            "instant": ("instant",)}


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


def _axis(values: ArrayLike | None, name: str) -> np.ndarray | None:
    """A time axis: one dimensional, finite and strictly increasing."""
    if values is None:
        return None
    v = np.asarray(values, dtype=float)
    if v.ndim != 1 or v.size == 0:
        raise ValueError(f"{name} is a one-dimensional sequence of times")
    if not np.all(np.isfinite(v)) or not np.all(np.diff(v) > 0.0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    return v


def _checked(name: str, field: SurfaceVariable) -> None:
    """The per-field statements `write_surface` refuses early."""
    if not isinstance(name, str) or not name:
        raise TypeError(f"a field's name is a string, got {name!r}")
    if name in _RESERVED or name.endswith("_poles"):
        raise ValueError(f"{name!r} is a name the surface file keeps for itself")
    if not isinstance(field, SurfaceVariable):
        raise TypeError(f"{name!r} is not a SurfaceVariable")
    if field.support not in TIME_SUPPORTS:
        raise ValueError(f"{name!r}: time_support {field.support!r} is not one "
                         f"of {TIME_SUPPORTS}")
    if field.time_interpolation not in TIME_METHODS:
        raise ValueError(f"{name!r}: time_interpolation "
                         f"{field.time_interpolation!r} is not one of "
                         f"{TIME_METHODS}")
    if field.support != "node" and field.time_interpolation != "none":
        raise ValueError(f"{name!r}: a {field.support!r} field is not sampled "
                         "between times; its time_interpolation is 'none'")
    if field.lateral_interpolation not in LATERAL_METHODS:
        raise ValueError(f"{name!r}: lateral_interpolation "
                         f"{field.lateral_interpolation!r} is not one of "
                         f"{tuple(LATERAL_METHODS)}")
    if field.dimensions is not None and not isinstance(field.dimensions,
                                                       Dimensions):
        raise TypeError(f"{name!r}: dimensions are a units.Dimensions")


def _leading_shape(support: str, time: np.ndarray | None,
                   instant: np.ndarray | None, name: str) -> tuple[int, ...]:
    """The leading shape a support's values carry, its axis required."""
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
                  fields: Mapping[str, SurfaceVariable], source: str,
                  time: ArrayLike | None = None,
                  instant: ArrayLike | None = None,
                  radius: float | None = None, scales: Scales = Scales.SI,
                  title: str | None = None,
                  meta: Mapping[str, Any] | None = None,
                  citation: Sequence[str] = ()) -> Path:
    """Write `fields` on `grid` to a surface file at `path`, returned.

    `source` names the writing code and version; `time` and `instant`
    are the axes of the module docstring, given where a field needs
    one; `radius` is the surface's radius and every number is in
    `scales`' units; `meta` becomes `meta_<key>` attributes.  The file
    is written whole or not at all.
    """
    nc = netcdf_module()
    if not isinstance(grid, AngularGrid):
        raise TypeError(f"expected an AngularGrid, got {type(grid).__name__}")
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source names the writing code and version")
    if not isinstance(scales, Scales):
        raise TypeError(f"expected Scales, got {type(scales).__name__}")
    if isinstance(fields, str) or not isinstance(fields, Mapping) or not fields:
        raise TypeError("fields maps names to SurfaceVariables, at least one")
    t, s = _axis(time, "time"), _axis(instant, "instant")
    if radius is not None and not (isinstance(radius, numbers.Real)
                                   and np.isfinite(radius) and radius > 0.0):
        raise ValueError(f"radius must be positive and finite, got {radius!r}")
    meta = dict(meta or {})
    for key, value in meta.items():
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
            raise TypeError(f"meta[{key!r}] is {value!r}; a netCDF attribute is "
                            "a string or a number")
    has_poles = all(grid.pole_nodes)
    prepared: dict[str, tuple[SurfaceVariable, np.ndarray, np.ndarray | None]] = {}
    for name, field in fields.items():
        _checked(name, field)
        leading = _leading_shape(field.support, t, s, name)
        values = _real_values(field.values, leading + (grid.ntheta, grid.nphi),
                              name, "its values")
        poles = None
        if field.poles is not None:
            if has_poles:
                raise ValueError(f"{name!r} gives polar values, and the grid "
                                 "holds both poles already")
            poles = _real_values(field.poles, leading + (2,), name, "its poles")
        prepared[name] = (field, values, poles)
    needs_interval = any(f.support == "interval" for f, _, _ in prepared.values())
    needs_pole = any(p is not None for _, _, p in prepared.values())
    path = Path(path)
    fd, partial = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent or None)
    os.close(fd)
    try:
        _write_file(nc, Path(partial), grid, prepared, source, t, s,
                    needs_interval, needs_pole, radius, scales, title, meta,
                    tuple(citation))
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return path


def _write_file(nc: Any, path: Path, grid: AngularGrid,
                prepared: Mapping[str, tuple[SurfaceVariable, np.ndarray,
                                             np.ndarray | None]],
                source: str, time: np.ndarray | None,
                instant: np.ndarray | None, needs_interval: bool,
                needs_pole: bool, radius: float | None, scales: Scales,
                title: str | None, meta: Mapping[str, Any],
                citation: tuple[str, ...]) -> None:
    """The file of `write_surface`, at a path of its own until it is whole."""
    time_unit = scaled_unit(TIME, scales)
    time_dims = np.asarray([getattr(TIME, b) for b in BASE], dtype=np.int32)
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
        if radius is not None:
            root.setncattr("radius", float(radius))
        if citation:
            root.setncattr("citation", "\n".join(citation))
        for key, value in meta.items():
            root.setncattr(f"meta_{key}", value)
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
            dims = _LEADING[field.support] + ("latitude", "longitude")
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

def _fail(path: object, msg: str) -> NoReturn:
    raise ValueError(f"malformed surface file {path}: {msg}")


def validate_structure(source: str | os.PathLike[str] | Any) -> None:
    """Check that a file, by path or as an open netCDF4 Dataset, has the
    structure the module docstring defines, raising ValueError naming
    the first thing wrong."""
    if isinstance(source, (str, os.PathLike)):
        nc = netcdf_module()
        with nc.Dataset(source, "r") as ds:
            _validate(ds, source)
    else:
        _validate(source, getattr(source, "filepath", lambda: "<dataset>")())


def _validate(ds: Any, path: object) -> None:
    """The checks of `validate_structure` on an open dataset."""
    attrs = set(ds.ncattrs())
    if "schema" not in attrs:
        _fail(path, "no schema attribute: not a planetmodel surface file")
    if ds.getncattr("schema") != SCHEMA:
        _fail(path, f"schema is {ds.getncattr('schema')!r}, this reader reads "
                    f"{SCHEMA!r}")
    for key in ("source", "created"):
        if key not in attrs or not str(ds.getncattr(key)).strip():
            _fail(path, f"{key} is missing or empty")
    for base in ("length", "mass", "time", "temperature"):
        key = f"scale_{base}"
        if key not in attrs or not float(ds.getncattr(key)) > 0.0:
            _fail(path, f"{key} is missing or not positive")
    if "radius" in attrs:
        radius = ds.getncattr("radius")
        if (isinstance(radius, str) or np.size(radius) != 1
                or not np.isfinite(float(radius)) or not float(radius) > 0.0):
            _fail(path, f"radius is {radius!r}, not a positive number")
    if "grid_kind" not in attrs or ds.getncattr("grid_kind") not in GRID_KINDS:
        _fail(path, f"grid_kind is missing or not a kind of grid, one of "
                    f"{GRID_KINDS}")
    for name in ("latitude", "longitude"):
        if name not in ds.variables or tuple(
                ds.variables[name].dimensions) != (name,):
            _fail(path, f"no coordinate variable {name!r}")
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
    has_pole_dim = "pole" in ds.dimensions
    for name, var in ds.variables.items():
        if name in _RESERVED:
            continue
        if name.endswith("_poles"):
            base_name = name[:-len("_poles")]
            if base_name not in ds.variables or base_name in _RESERVED:
                _fail(path, f"{name} accompanies no field {base_name!r}")
            if not has_pole_dim or var.dimensions[-1] != "pole":
                _fail(path, f"{name} does not end on the pole dimension")
            continue
        a = set(var.ncattrs())
        if "time_support" not in a or var.getncattr(
                "time_support") not in TIME_SUPPORTS:
            _fail(path, f"{name} states no time support, one of {TIME_SUPPORTS}")
        support = str(var.getncattr("time_support"))
        want = _LEADING[support] + ("latitude", "longitude")
        if tuple(var.dimensions) != want:
            _fail(path, f"{name} is {support!r} and does not run over {want}")
        if "lateral_interpolation" not in a or var.getncattr(
                "lateral_interpolation") not in LATERAL_METHODS:
            _fail(path, f"{name} states no known lateral rule")
        rule = var.getncattr("time_interpolation") if "time_interpolation" in a \
            else None
        if rule not in TIME_METHODS or (support != "node" and rule != "none"):
            _fail(path, f"{name} is {support!r} and states the time rule "
                        f"{rule!r}")


def read_surface_file(path: str | os.PathLike[str]) -> SurfaceFile:
    """A surface file's numbers and metadata, its structure validated first."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        _validate(ds, path)
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
                     if name not in _RESERVED}
        grid = grid_of_group(ds)
    radius = attrs.get("radius")
    title = attrs.get("title")
    return SurfaceFile(str(attrs["schema"]), grid, MappingProxyType(variables),
                       scales=scales, time=time, instant=instant,
                       radius=None if radius is None else float(radius),
                       source=str(attrs["source"]),
                       created=str(attrs["created"]),
                       title=None if title is None else str(title),
                       meta=MappingProxyType(meta),
                       citation=tuple(str(attrs.get("citation", "")).splitlines()))


# -- the file as a function of time -----------------------------------------------

#: What a field does with a date outside its span.
_EXTRAPOLATIONS = ("refuse", "hold")


class SurfaceHistory:
    """A surface file as a function of time (module docstring): each
    field's values at a date, by the field's own time rule, and as a
    `GriddedFunction` by its lateral one."""

    def __init__(self, file: SurfaceFile) -> None:
        self._file = file

    @property
    def file(self) -> SurfaceFile:
        """The file's numbers and metadata, as read."""
        return self._file

    @property
    def grid(self) -> AngularGrid:
        """The angular grid every field is on."""
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
        return str(self._variable(name).attributes["time_support"])

    def dimensions(self, name: str) -> Dimensions | None:
        """A field's dimensions, where the file declares them."""
        exps = self._variable(name).attributes.get("dimensions")
        if exps is None:
            return None
        return Dimensions(**{b: int(e) for b, e in zip(BASE, np.atleast_1d(exps))})

    def values(self, name: str, *, date: float | None = None,
               extrapolation: str = "refuse") -> np.ndarray:
        """A field's values on the grid at `date`, (ntheta, nphi), by its
        own time rule (module docstring); NaN stays where the field does
        not cover a point.  `extrapolation` is `refuse` (the default) or
        `hold`, which serves the nearest value of a `linear`, `previous`
        or `interval` field outside its span."""
        rows = self._rows(name, date, extrapolation)
        values = self._variable(name).values
        if rows is None:
            return values.copy()
        (k0, w0), (k1, w1) = rows
        return w0 * values[k0] + w1 * values[k1]

    def function(self, name: str, *, date: float | None = None,
                 extrapolation: str = "refuse",
                 rtol: float = 1e-9) -> GriddedFunction:
        """The field at `date` as a `GriddedFunction` by its stated lateral
        rule (`harmonic` read as `cubic`, this library evaluating no
        harmonic rule), its stored polar values carried along where the
        file holds them; a field with NaN cannot be one."""
        from .gridded import GriddedFunction

        lateral = str(self._variable(name).attributes["lateral_interpolation"])
        poles = None
        held = self._file.variables.get(f"{name}_poles")
        if held is not None:
            rows = self._rows(name, date, extrapolation)
            if rows is None:
                poles = np.asarray(held.values, dtype=float)
            else:
                (k0, w0), (k1, w1) = rows
                poles = w0 * held.values[k0] + w1 * held.values[k1]
        return GriddedFunction(self.grid, self.values(name, date=date,
                                                      extrapolation=extrapolation),
                               lateral="cubic" if lateral == "harmonic" else lateral,
                               poles=poles, rtol=rtol)

    def _variable(self, name: str) -> FileVariable:
        """The variable of a field, polar values and reserved names refused."""
        if name.endswith("_poles") or name not in self._file.variables:
            raise KeyError(f"the file holds no field {name!r}")
        return self._file.variables[name]

    def _rows(self, name: str, date: float | None, extrapolation: str
              ) -> tuple[tuple[int, float], tuple[int, float]] | None:
        """The leading rows a date reads, with their weights; None for a
        static field, whose values are the answer whole."""
        if extrapolation not in _EXTRAPOLATIONS:
            raise ValueError(f"extrapolation is one of {_EXTRAPOLATIONS}, "
                             f"got {extrapolation!r}")
        support = self.support(name)
        if support == "static":
            return None
        if date is None:
            raise ValueError(f"{name!r} is a {support} field and needs a date")
        date = float(date)
        hold = extrapolation == "hold"
        if support == "instant":
            axis = self._file.instant
            hits = np.flatnonzero(axis == date)
            if hits.size == 0:
                raise ValueError(f"{name!r} is held at its stored instants "
                                 f"alone, and {date!r} is not one")
            return ((int(hits[0]), 1.0), (int(hits[0]), 0.0))
        axis = self._file.time
        rule = str(self._variable(name).attributes["time_interpolation"])
        if support == "node" and rule == "none":
            hits = np.flatnonzero(axis == date)
            if hits.size == 0:
                raise ValueError(f"{name!r} states no time rule and is held "
                                 f"at its stored nodes alone; {date!r} is not one")
            return ((int(hits[0]), 1.0), (int(hits[0]), 0.0))
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
        k = min(int(np.searchsorted(axis, date, side="right")) - 1, axis.size - 2)
        w = (date - float(axis[k])) / (float(axis[k + 1]) - float(axis[k]))
        if w == 0.0:  # a stored node: its own values, a NaN beside it kept out
            return ((k, 1.0), (k, 0.0))
        if w == 1.0:
            return ((k + 1, 1.0), (k + 1, 0.0))
        return ((k, 1.0 - w), (k + 1, w))


def read_surface(path: str | os.PathLike[str]) -> SurfaceHistory:
    """A surface file as a function of time, its structure validated
    first."""
    return SurfaceHistory(read_surface_file(path))
