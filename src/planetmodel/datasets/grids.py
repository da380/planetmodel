"""Gridded sources: netCDF grids of latitude and longitude, in the library's form.

The Earth Model Collaboration's files, and COARDS grids such as GMT's,
hold fields on a grid of latitude and longitude, with or without a
vertical axis, in conventions that vary from file to file.  `read_grid`
reads one into a `GridData` in the library's form:

- the axes are found by their coordinate variables' names (latitude,
  lat; longitude, lon; x and y; depth; radius; elevation, height), or
  failing those by `standard_name`, units and `positive`, in any order
  of dimensions;
- a vertical axis of depth (positive down), radius or elevation becomes
  radius in metres on the source's surface radius (its global
  `radius_in_meters`, else `surface=`, the Earth's 6371 km by default),
  increasing; two slices within `VERTICAL_RTOL` of the surface radius of
  each other, and the pairs `pairs=` names, are made one radius: a
  discontinuity, the two slices its sides, as a repeated radius is in a
  deck;
- latitudes become colatitudes increasing from the north, longitudes
  are reduced into one turn and sorted, a seam column repeated at the
  end made one with the first; equally spaced nodes are
  snapped to their exact positions, and `regular=True` rebuilds a
  glitched coordinate as the regular one of its count and ends, refused
  where a node is off by more than half a step;
- missing values (`_FillValue`, `missing_value`, NaN) become NaN, and
  `valid_range` is ignored, being in some files the rounded range of
  the data;
- a seam column repeated at the end, and a pole row, are checked to hold
  one value to `SAME_RTOL` of the field's largest (the precision of a file
  written to five figures) and made exactly that; one that does not is
  refused, or averaged where the entry says so (`seam="mean"`,
  `pole_rows="mean"`: a source whose own gridding left two columns on
  one meridian);
- `stride=k` keeps every k-th latitude and longitude of the file's own
  order, for a grid too fine to hold whole, and `vertical=(lo, hi)` the
  slices whose vertical coordinate lies in that window (in the file's
  own units, either end open as None), for a source whose part beyond it
  is not wanted; both are applied as the file is read, and the file's
  own precision kept until the grid is in its order, so that a large
  grid is copied only once it is cut down.

`read_table` reads a text table of longitude, latitude and values, a
node a row in any order (LithoRef18's), into the same form: the rows
must be every node of a product of latitudes and longitudes, each once,
and the coordinates then follow the rules above.

A grid on a projected plane, x and y (a polar stereographic grid of the
Antarctic, EPSG:3031), is read as `ProjectedData`: its northing and
easting in metres, increasing (their unit a length, km or m), its values
on (radius, y, x), and the projection, `projection=` where the entry
gives it and else the file's `proj4text`
(`projections.PolarStereographic`); it is always regional.

A grid whose longitudes go round the circle and whose latitudes reach
within a step and a half of each pole is global, on an `AngularGrid`:
the shipped equiangular kind where its nodes are one (from wherever its
longitudes start), a custom grid otherwise.  Any other is regional,
kept on its own coordinates.  Values stay in the file's units, beside
its unit strings.

`grid_model` makes the model of a global grid with a vertical axis:
each variable under the vocabulary name it is mapped to, in SI by its
unit string (`units.parse_unit`; a mislabel corrected by `units=`), as
values (kind None) or as a perturbation of one of `perturbation.KINDS`
over a reference; depth slices missing everywhere dropped from the ends
of the support and refused inside it, scattered missing values refused
unless `fill=` gives them; gridded fields (`GriddedField`) with `ngll = 2` between the
slices, radially linear, laterally `lateral` (linear by default), one
per segment between discontinuities; at a pole the grid lacks, the mean
of its nearest ring, the reader's rule, right to second order in the
ring's distance from the pole.  `RegionalGrid` holds a regional grid's
values in SI and evaluates them inside its box.  `GridReader` is the
reader of a data set of one such file: a model of a global grid, a
`RegionalGrid` of a regional one.  netCDF4 is imported where a file is
read (the `netcdf` extra).
"""
from __future__ import annotations

import os
import warnings
from collections.abc import Iterable, Mapping
from dataclasses import KW_ONLY, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import RegularGridInterpolator

from ..character import SCALAR
from ..gridded import GriddedField
from ..model import Model, cited
from ..perturbation import KINDS
from ..sampling import AngularGrid, equiangular
from ..units import DIMENSIONLESS, LENGTH, Scales, parse_unit
from ..vocabulary import spec
from .models import over_reference, reference_of, standalone
from .projections import PolarStereographic

if TYPE_CHECKING:
    from .registry import Dataset

__all__ = ["VERTICAL_RTOL", "SAME_RTOL", "EARTH_RADIUS", "Variables", "GridData",
           "ProjectedData", "Gridded", "read_grid", "read_table", "ring_poles",
           "grid_model", "RegionalGrid", "GridReader"]

#: Two slices this close, relative to the surface radius, are one radius.
VERTICAL_RTOL = 1e-5

#: A pole row, or a seam column and its repeat, hold one value to this
#: fraction of the field's largest: the precision of five figures.
SAME_RTOL = 1e-4

#: The radius of a depth of zero unless a file or its entry says otherwise.
EARTH_RADIUS = 6371.0e3

#: How the variables of a file are read: each file name to the vocabulary
#: name it is and its kind, None for values, else a perturbation's kind.
type Variables = Mapping[str, tuple[str, str | None]]

_ROLES = {"latitude": ("latitude", "lat"), "longitude": ("longitude", "lon"),
          "x": ("x", "projection_x_coordinate"), "y": ("y", "projection_y_coordinate"),
          "depth": ("depth",), "radius": ("radius",),
          "elevation": ("elevation", "height", "altitude")}
_VERTICAL = ("depth", "radius", "elevation")


@dataclass(frozen=True, eq=False)
class GridData:
    """A gridded source in the library's form: `theta` colatitudes
    increasing and `phi` longitudes (radians; in one turn and increasing
    for a global grid, contiguous for a regional one), `radius` (metres,
    non-decreasing, a repeat a discontinuity; None for a surface),
    `values` by the file's names, shaped (radius, theta, phi) or (theta,
    phi), in the file's `units`, NaN where missing; a global grid's
    `grid`."""

    theta: np.ndarray
    phi: np.ndarray
    radius: np.ndarray | None
    values: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    _: KW_ONLY
    grid: AngularGrid | None = None
    surface: float = EARTH_RADIUS
    long_names: Mapping[str, str] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def is_global(self) -> bool:
        """Whether the grid covers the sphere, and so has an `AngularGrid`."""
        return self.grid is not None


@dataclass(frozen=True, eq=False)
class ProjectedData:
    """A gridded source on a projected plane in the library's form: `y`
    and `x` its northing and easting (metres, increasing), `radius` as a
    `GridData`'s, `values` by the file's names shaped (radius, y, x) or
    (y, x), in the file's `units`, NaN where missing, and the
    `projection` that takes the library's angles to (x, y)."""

    y: np.ndarray
    x: np.ndarray
    radius: np.ndarray | None
    values: Mapping[str, np.ndarray]
    units: Mapping[str, str]
    projection: PolarStereographic
    _: KW_ONLY
    surface: float = EARTH_RADIUS
    long_names: Mapping[str, str] = field(default_factory=dict)
    attributes: Mapping[str, Any] = field(default_factory=dict)
    source: str | None = None

    @property
    def is_global(self) -> bool:
        """A projected grid is regional."""
        return False


#: What `read_grid` gives: a grid of latitude and longitude, or a projected one.
type Gridded = GridData | ProjectedData


def _roles(ds: Any) -> dict[str, str]:
    """The coordinate variable of each axis a file has, by its role."""
    found: dict[str, str] = {}
    for name, var in ds.variables.items():
        if var.ndim != 1 or var.dimensions != (name,):
            continue
        low = name.lower()
        std = str(getattr(var, "standard_name", "")).lower()
        unit = str(getattr(var, "units", "")).strip().lower()
        role = next((r for r, names in _ROLES.items() if low in names or std in names),
                    None)
        if role is None and unit.startswith(("degrees_north", "degree_north")):
            role = "latitude"
        elif role is None and unit.startswith(("degrees_east", "degree_east")):
            role = "longitude"
        if role is not None and role not in found:
            found[role] = name
    geographic = "latitude" in found and "longitude" in found
    if not geographic and not ("x" in found and "y" in found):
        raise ValueError("the file has neither latitude and longitude axes nor x and "
                         "y ones")
    for role in (("x", "y") if geographic else ("latitude", "longitude")):
        found.pop(role, None)
    vertical = [r for r in _VERTICAL if r in found]
    if len(vertical) > 1:
        raise ValueError(f"the file has more than one vertical axis: {vertical}")
    return found


def _regular(values: np.ndarray, what: str) -> np.ndarray:
    """A coordinate rebuilt as the regular one of its count and ends,
    refused where a node is off by more than half a step."""
    rebuilt = np.linspace(values[0], values[-1], values.size)
    step = abs(rebuilt[1] - rebuilt[0]) if values.size > 1 else 1.0
    if np.max(np.abs(values - rebuilt)) > 0.5 * step:
        raise ValueError(f"the {what} are not a regular coordinate with glitches: a "
                         "node is off by more than half a step")
    return rebuilt


def _unwrapped(reduced: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Sorted longitudes of one turn, in degrees, made contiguous and
    increasing from after the widest gap around the circle; with the
    permutation that takes them there."""
    gaps = np.diff(np.concatenate([reduced, [reduced[0] + 360.0]]))
    start = (int(np.argmax(gaps)) + 1) % reduced.size
    order = np.concatenate([np.arange(start, reduced.size), np.arange(start)])
    return np.concatenate([reduced[start:], reduced[:start] + 360.0]), order


def _horizontal(path: str, lat: np.ndarray, lon: np.ndarray,
                values: dict[str, np.ndarray], *, regular: bool, seam: str,
                pole_rows: str
                ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray],
                           AngularGrid | None]:
    """A source's latitudes and longitudes, and its values on them (their
    last two axes), in the library's form by the rules of the module
    docstring: the colatitudes, the longitudes, the values reordered, and
    a global grid's `AngularGrid` (None for a regional one)."""
    if regular:
        lat, lon = _regular(lat, "latitudes"), _regular(lon, "longitudes")
    # colatitude increasing from the north
    if np.any(np.abs(lat) > 90.0 + 1e-6):
        raise ValueError(f"{path}: latitudes beyond +-90 degrees")
    lat = np.clip(lat, -90.0, 90.0)
    order = np.argsort(-lat, kind="stable")
    lat = lat[order]
    values = {n: np.take(v, order, axis=-2) for n, v in values.items()}
    # longitudes: the seam dropped, one turn, sorted
    reduced = np.mod(lon, 360.0)
    reduced[np.isclose(reduced, 360.0, rtol=0.0, atol=1e-9)] = 0.0
    keep = np.ones(lon.size, dtype=bool)
    for i in range(lon.size):
        same = np.flatnonzero(np.isclose(reduced[:i], reduced[i], rtol=0.0, atol=1e-6))
        if same.size:
            j = int(same[0])
            for n, v in values.items():
                scale = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
                worst = float(np.nanmax(np.abs(v[..., i] - v[..., j]), initial=0.0))
                if seam == "refuse" and worst > SAME_RTOL * scale:
                    raise ValueError(
                        f"{path}: {n!r} differs by {worst:.3g} between the two columns "
                        f"at longitude {reduced[i]:g}, the seam repeated; seam='mean' "
                        "averages them")
                v[..., j] = 0.5 * (v[..., i] + v[..., j])
            keep[i] = False
    lon, reduced = lon[keep], reduced[keep]
    values = {n: v[..., keep] for n, v in values.items()}
    order = np.argsort(reduced, kind="stable")
    reduced = reduced[order]
    values = {n: v[..., order] for n, v in values.items()}
    values = {n: np.array(v, dtype=float) for n, v in values.items()}
    theta = np.radians(90.0 - lat)
    # global or regional
    lon_gaps = np.diff(np.concatenate([reduced, [reduced[0] + 360.0]]))
    lat_step = np.median(np.abs(np.diff(lat))) if lat.size > 1 else 180.0
    reaches = 90.0 - lat[0] <= 1.5 * lat_step and lat[-1] + 90.0 <= 1.5 * lat_step
    is_global = bool(lon_gaps.max() <= 1.5 * np.median(lon_gaps) and reaches)
    grid = None
    if is_global:
        n, m = theta.size, reduced.size
        phi0 = np.radians(reduced[0])
        even_lon = np.allclose(lon_gaps, 360.0 / m, rtol=0.0, atol=1e-6)
        for poles in (False, True):
            if not even_lon or (poles and n < 2):
                continue
            candidate = equiangular(n, m, poles=poles, phi0=phi0)
            if (np.allclose(candidate.colatitudes, theta, rtol=0.0, atol=1e-8)
                    and np.allclose(candidate.longitudes, np.radians(reduced),
                                    rtol=0.0, atol=1e-8)):
                grid = candidate
                break
        if grid is None:
            phi = (np.sort(np.radians(reduced) % (2 * np.pi)) if not even_lon else
                   np.radians(reduced[0]) + 2 * np.pi * np.arange(m) / m)
            grid = AngularGrid(theta, phi)
        theta, phi = grid.colatitudes, grid.longitudes
        # the pole rows, one value each
        for k, at in ((0, 0.0), (-1, np.pi)):
            if abs(theta[k] - at) > 1e-12:
                continue
            for name, v in values.items():
                row = v[..., k, :]
                with warnings.catch_warnings():         # a slice missing everywhere
                    warnings.simplefilter("ignore", RuntimeWarning)
                    spread = np.nanmax(row, axis=-1) - np.nanmin(row, axis=-1)
                    mean = np.nanmean(row, axis=-1)
                scale = float(np.nanmax(np.abs(v))) if np.isfinite(v).any() else 1.0
                worst = float(np.nanmax(spread, initial=0.0))
                if pole_rows == "refuse" and worst > SAME_RTOL * scale:
                    raise ValueError(
                        f"{path}: {name!r} varies by {worst:.3g} along the "
                        f"pole row at colatitude {at:g}; pole_rows='mean' averages it")
                v[..., k, :] = mean[..., None]
    else:
        contiguous, order = _unwrapped(reduced)
        phi = np.radians(contiguous)
        values = {n: v[..., order] for n, v in values.items()}
    return theta, phi, values, grid


def _plane(path: str, coords: Mapping[str, np.ndarray], unit: Mapping[str, str],
           values: dict[str, np.ndarray], *, regular: bool
           ) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """A projected grid's northing and easting in metres, increasing, and
    its values reordered to match (their last two axes)."""
    out = []
    for role, axis in (("y", -2), ("x", -1)):
        c = np.asarray(coords[role], dtype=float)
        dims, factor = parse_unit(unit[role] or "m")
        if dims != LENGTH:
            raise ValueError(f"{path}: the {role} axis's unit {unit[role]!r} is not a "
                             "length")
        if regular:
            c = _regular(c, f"{role} coordinates")
        order = np.argsort(c, kind="stable")
        c = factor * c[order]
        if np.any(np.diff(c) <= 0.0):
            raise ValueError(f"{path}: the {role} coordinates repeat")
        values = {n: np.take(v, order, axis=axis) for n, v in values.items()}
        out.append(c)
    values = {n: np.array(v, dtype=float) for n, v in values.items()}
    return out[0], out[1], values


def read_grid(source: str | os.PathLike[str], *,
              variables: tuple[str, ...] | None = None,
              surface: float | None = None, vertical_units: str | None = None,
              pairs: tuple[tuple[float, float], ...] = (), regular: bool = False,
              seam: str = "refuse", pole_rows: str = "refuse", stride: int = 1,
              vertical: tuple[float | None, float | None] | None = None,
              projection: PolarStereographic | None = None) -> Gridded:
    """A netCDF grid in the library's form, by the rules of the module
    docstring: `variables` the names to read (every field on the grid's
    axes by default), `surface` the radius of a depth of zero in metres
    where the file states none, `vertical_units` the vertical axis's unit
    where it states none, `pairs` the values of the vertical coordinate
    of slices that are the two sides of one discontinuity, `projection`
    a projected grid's, in place of its `proj4text`."""
    import netCDF4

    for what, rule in (("seam", seam), ("pole_rows", pole_rows)):
        if rule not in ("refuse", "mean"):
            raise ValueError(f"{what} must be 'refuse' or 'mean', got {rule!r}")
    path = os.fspath(source)
    with netCDF4.Dataset(path) as ds:
        ds.set_auto_mask(False)
        roles = _roles(ds)
        projected = "x" in roles
        horizontal = ["y", "x"] if projected else ["latitude", "longitude"]
        axes = [r for r in _VERTICAL if r in roles] + horizontal
        dims = [roles[r] for r in axes]
        names = (tuple(n for n, v in ds.variables.items()
                       if n not in dims and set(v.dimensions) == set(dims))
                 if variables is None else tuple(variables))
        if not names:
            raise ValueError(f"{path}: no field lies on the grid's axes {dims}")
        coords = {r: np.asarray(ds.variables[roles[r]][:], dtype=float) for r in axes}
        vunit = {r: str(getattr(ds.variables[roles[r]], "units", "")) for r in axes}
        attributes = {k: ds.getncattr(k) for k in ds.ncattrs()}
        axis = next((r for r in _VERTICAL if r in coords), None)
        stride = int(stride)
        if stride < 1:
            raise ValueError(f"stride must be a positive integer, got {stride}")
        take = {roles[horizontal[0]]: slice(None, None, stride),
                roles[horizontal[1]]: slice(None, None, stride)}
        if vertical is not None:
            if axis is None:
                raise ValueError(f"{path}: vertical= cuts a grid with a vertical axis")
            lo, hi = vertical
            c = coords[axis]
            keep = np.ones(c.shape, dtype=bool)
            if lo is not None:
                keep &= c >= lo
            if hi is not None:
                keep &= c <= hi
            if keep.sum() < 2:
                raise ValueError(f"{path}: the window {vertical} keeps fewer than "
                                 "two slices")
            first, last = int(np.argmax(keep)), int(keep.size - np.argmax(keep[::-1]))
            take[roles[axis]] = slice(first, last)
        for r in axes:
            coords[r] = coords[r][take.get(roles[r], slice(None))]
        values, units, long_names = {}, {}, {}
        for n in names:
            var = ds.variables[n]
            if set(var.dimensions) != set(dims):
                raise ValueError(f"{path}: {n!r} lies on {var.dimensions}, not the "
                                 f"grid's axes {dims}")
            a = np.asarray(var[tuple(take.get(d, slice(None)) for d in var.dimensions)])
            a = a.astype(np.float32 if a.dtype == np.float32 else float, copy=False)
            for attr in ("_FillValue", "missing_value"):
                if attr in var.ncattrs():
                    fill = np.asarray(var.getncattr(attr), dtype=var.dtype).ravel()[0]
                    a[a == fill] = np.nan
            values[n] = np.transpose(a, [var.dimensions.index(d) for d in dims])
            units[n] = str(getattr(var, "units", ""))
            long_names[n] = str(getattr(var, "long_name", ""))
    if projected:
        y, x, values = _plane(path, coords, vunit, values, regular=regular)
        if projection is None:
            text = attributes.get("proj4text")
            if text is None:
                raise ValueError(f"{path}: a projected grid with no proj4text; give "
                                 "projection=")
            projection = PolarStereographic.from_proj4(str(text))
    else:
        theta, phi, values, grid = _horizontal(path, coords["latitude"],
                                               coords["longitude"], values,
                                               regular=regular, seam=seam,
                                               pole_rows=pole_rows)
    radius = None
    vertical = axis
    if vertical is not None:
        unit = vertical_units or vunit[vertical]
        if not unit.strip():
            raise ValueError(f"{path}: the {vertical} axis states no unit; give "
                             "vertical_units=")
        dims_, factor = parse_unit(unit)
        if dims_ != LENGTH:
            raise ValueError(f"{path}: the {vertical} axis's unit {unit!r} is not a "
                             "length")
        top = float(attributes.get("radius_in_meters", surface or EARTH_RADIUS))
        c = coords[vertical]
        r = {"depth": top - c * factor, "radius": c * factor,
             "elevation": top + c * factor}[vertical]
        order = np.argsort(r, kind="stable")
        r, c = r[order], c[order]
        values = {n: v[order] for n, v in values.items()}
        for a, b in pairs:
            i = np.flatnonzero(np.isclose(c, a, rtol=0.0, atol=1e-9 * max(1.0, abs(a))))
            j = np.flatnonzero(np.isclose(c, b, rtol=0.0, atol=1e-9 * max(1.0, abs(b))))
            if i.size != 1 or j.size != 1 or abs(int(i[0]) - int(j[0])) != 1:
                raise ValueError(f"{path}: the pair ({a:g}, {b:g}) is not two adjacent "
                                 f"slices of the {vertical} axis")
            lo, hi = sorted((int(i[0]), int(j[0])))
            r[lo] = r[hi] = 0.5 * (r[lo] + r[hi])
        close = np.flatnonzero(np.diff(r) < VERTICAL_RTOL * top)
        if np.any(np.diff(close) == 1):
            raise ValueError(f"{path}: three slices meet at one radius")
        for i in close:
            r[i] = r[i + 1] = 0.5 * (r[i] + r[i + 1])
        radius = r
        surface = top
    for v in values.values():
        v.setflags(write=False)
    if projected:
        return ProjectedData(y, x, radius, MappingProxyType(values),
                             MappingProxyType(units), projection,
                             surface=float(surface or EARTH_RADIUS),
                             long_names=MappingProxyType(long_names),
                             attributes=MappingProxyType(attributes), source=path)
    return GridData(theta, phi, radius, MappingProxyType(values),
                    MappingProxyType(units), grid=grid,
                    surface=float(surface or EARTH_RADIUS),
                    long_names=MappingProxyType(long_names),
                    attributes=MappingProxyType(attributes), source=path)


def read_table(source: str | os.PathLike[str], *,
               names: tuple[str, ...] | None = None, longitude: str = "LONG",
               latitude: str = "LAT", columns: tuple[str, ...] | None = None,
               units: Mapping[str, str] | None = None, regular: bool = False,
               seam: str = "refuse", pole_rows: str = "refuse") -> GridData:
    """A text table of a surface's values in the library's form, by the
    rules of the module docstring: `names` the columns' names, by default
    the words of the last line before the first row of numbers;
    `longitude` and `latitude` the names of those two; `columns` the
    names to read, every other column by default; `units` each one's
    unit string, which a table does not state.  Whitespace or commas
    separate the values; lines before the first row of numbers are its
    header."""
    path = os.fspath(source)
    rows = Path(path).read_text(encoding="utf-8", errors="replace").splitlines()
    first = None
    for i, line in enumerate(rows):
        words = line.replace(",", " ").split()
        try:
            if len(words) >= 3 and all(np.isfinite(float(w)) for w in words):
                first = i
                break
        except ValueError:
            continue
    if first is None:
        raise ValueError(f"{path}: no row of numbers")
    if names is None:
        heads = [line for line in rows[:first] if line.strip()]
        if not heads:
            raise ValueError(f"{path}: no line of column names; give names=")
        names = tuple(heads[-1].replace(",", " ").split())
    table = np.loadtxt([r.replace(",", " ") for r in rows[first:] if r.strip()],
                       ndmin=2)
    if table.shape[1] != len(names):
        raise ValueError(f"{path}: {table.shape[1]} columns, named {list(names)}")
    for want in (longitude, latitude):
        if want not in names:
            raise KeyError(f"{path}: no column {want!r}; the columns are {list(names)}")
    picked = (tuple(n for n in names if n not in (longitude, latitude))
              if columns is None else tuple(columns))
    for n in picked:
        if n not in names:
            raise KeyError(f"{path}: no column {n!r}; the columns are {list(names)}")
    lon_all = table[:, names.index(longitude)]
    lat_all = table[:, names.index(latitude)]
    lat, i = np.unique(lat_all, return_inverse=True)
    lon, j = np.unique(lon_all, return_inverse=True)
    seen = np.zeros((lat.size, lon.size), dtype=int)
    np.add.at(seen, (i, j), 1)
    if np.any(seen != 1):
        raise ValueError(f"{path}: the rows are not every node of a grid of "
                         f"{lat.size} latitudes and {lon.size} longitudes, each once")
    values = {}
    for n in picked:
        v = np.empty((lat.size, lon.size))
        v[i, j] = table[:, names.index(n)]
        values[n] = v
    theta, phi, values, grid = _horizontal(path, lat, lon, values, regular=regular,
                                           seam=seam, pole_rows=pole_rows)
    for v in values.values():
        v.setflags(write=False)
    units = dict(units or {})
    return GridData(theta, phi, None, MappingProxyType(values),
                    MappingProxyType({n: units.get(n, "") for n in picked}), grid=grid,
                    source=path)


def _si(data: Gridded, file_name: str, name: str, kind: str | None,
        units: Mapping[str, str] | None) -> np.ndarray:
    """A variable's values in SI, its unit checked against what it is."""
    unit = (units or {}).get(file_name, data.units[file_name])
    dims, factor = parse_unit(unit)
    if kind in ("relative", "log"):
        want = DIMENSIONLESS
    else:
        want = spec(name).dimensions
    if dims != want:
        raise ValueError(f"{file_name!r} is in {unit!r}, of dimensions {dims}; as "
                         f"{name!r} ({'values' if kind is None else kind}) it must be "
                         f"of {want}; units= corrects a mislabel")
    return data.values[file_name] * factor


def _support(v: np.ndarray, name: str) -> slice:
    """The slices of a variable not missing everywhere, refused where a
    missing one lies inside them."""
    present = np.isfinite(v).any(axis=(-2, -1))
    if not present.any():
        raise ValueError(f"{name!r} is missing everywhere")
    first, last = int(np.argmax(present)), int(present.size - np.argmax(present[::-1]))
    if not present[first:last].all():
        raise ValueError(f"{name!r} is missing on a whole slice inside its support")
    return slice(first, last)


def _segments(radius: np.ndarray) -> list[slice]:
    """The runs of slices between discontinuities (repeated radii), each of
    at least two slices; a lone slice at an end, the far side of a boundary
    the source stops at, is left out."""
    cuts = np.flatnonzero(np.diff(radius) == 0.0)
    starts = np.concatenate([[0], cuts + 1])
    stops = np.concatenate([cuts + 1, [radius.size]])
    out = [slice(int(a), int(b)) for a, b in zip(starts, stops)]
    while len(out) > 1 and out[0].stop - out[0].start < 2:
        out.pop(0)
    while len(out) > 1 and out[-1].stop - out[-1].start < 2:
        out.pop()
    for s in out:
        if s.stop - s.start < 2:
            raise ValueError(f"the segment at radius {radius[s.start]:g} m has one "
                             "slice; a segment between discontinuities needs two")
    return out


def ring_poles(v: np.ndarray, grid: AngularGrid) -> np.ndarray:
    """The values at the poles, (..., 2) for values (..., ntheta, nphi) on
    `grid`: a pole node's own, else the mean of the nearest ring, the
    readers' rule for a grid without its poles."""
    with warnings.catch_warnings():                     # a slice missing everywhere
        warnings.simplefilter("ignore", RuntimeWarning)
        north = np.nanmean(v[..., 0, :], axis=-1)
        south = np.nanmean(v[..., -1, :], axis=-1)
    return np.stack([north, south], axis=-1)


def grid_model(data: GridData, variables: Variables, *, reference: Model | None = None,
               units: Mapping[str, str] | None = None, lateral: str = "linear",
               fill: float | None = None, name: str | None = None,
               citation: Iterable[str] = ()) -> Model:
    """The model of a global grid with a vertical axis, by the rules of
    the module docstring: `variables` maps the file's names to (name,
    kind); values make a model of their own, perturbations need
    `reference`; citing `citation` (and what the reference cites)."""
    if not data.is_global or data.radius is None:
        raise ValueError("a model is made of a global grid with a vertical axis; "
                         "this grid is " + ("regional" if not data.is_global
                                            else "a surface"))
    kinds = {k for _, k in variables.values()}
    if len(kinds) != 1:
        raise ValueError(f"the variables of one model share a kind, got {kinds}")
    (kind,) = kinds
    if kind is not None and kind not in KINDS:
        raise ValueError(f"kind must be None or one of {KINDS}, got {kind!r}")
    grid = data.grid
    converted = {}
    support = None
    for file_name, (vocab, _) in variables.items():
        v = _si(data, file_name, vocab, kind, units)
        here = _support(v, file_name)
        if support is not None and here != support:
            raise ValueError("the variables are missing on different slices")
        support = here
        converted[vocab] = v
    radius = data.radius[support]
    segments = _segments(radius)
    fields: dict[str, list[GriddedField]] = {}
    for vocab, v in converted.items():
        v = np.array(v[support])
        missing = ~np.isfinite(v)
        if missing.any():
            if fill is None:
                raise ValueError(f"{vocab!r} is missing at {int(missing.sum())} nodes "
                                 "inside its support; fill= says what they are")
            v[missing] = fill
        character = (spec(vocab).character if kind in (None, "absolute")
                     else SCALAR)
        poles = ring_poles(v, grid)
        for k, node in enumerate(grid.pole_nodes):
            if node:
                poles[..., k] = np.nan
        segs = []
        for s in segments:
            r = radius[s]
            vals = np.stack([v[s][:-1], v[s][1:]], axis=1)
            pls = np.stack([poles[s][:-1], poles[s][1:]], axis=1)
            segs.append(GriddedField((float(r[0]), float(r[-1])), r, grid, vals,
                                     character=character, name=vocab,
                                     radial="linear", lateral=lateral, poles=pls))
        fields[vocab] = segs
    if kind is None:
        return standalone(fields, surface=data.surface, name=name, citation=citation)
    if reference is None:
        raise ValueError(f"perturbations of kind {kind!r} need reference=")
    return over_reference(reference, fields, kind=kind, name=name, citation=citation)


class RegionalGrid:
    """A regional grid's variables in SI by their vocabulary names, each
    with its kind, evaluated inside its region: linear in radius between
    the slices of each segment (a radius on a discontinuity reads the side
    above) and in its own two horizontal coordinates, NaN outside the
    region or the support.  Its coordinates are colatitude and longitude
    for a box in latitude and longitude (`GridData`), northing and easting
    for a projected grid (`ProjectedData`), reached from the library's
    angles by the grid's map, the identity or the projection; asked for
    the angle of a point inside its region's edge (`edge_distance`), a box
    measures it along great circles to its sides, a projected grid on its
    plane to the edge of the nodes holding a value at some radius, divided
    by the projection's scale there.  `reference` names what a
    perturbation is relative to, and `citation` holds the works it rests
    on."""

    def __init__(self, data: Gridded, variables: Variables, *,
                 units: Mapping[str, str] | None = None,
                 reference: str | None = None, citation: Iterable[str] = ()) -> None:
        if data.is_global or data.radius is None:
            raise ValueError("a RegionalGrid holds a regional grid with a vertical "
                             "axis")
        self._data = data
        self._projected = isinstance(data, ProjectedData)
        self._kinds = {vocab: kind for vocab, kind in variables.values()}
        self._values = {vocab: _si(data, f, vocab, kind, units)
                        for f, (vocab, kind) in variables.items()}
        self._segments = _segments(data.radius)
        self.reference = reference
        self._citation = cited((), citation)
        self._interp: dict[tuple[str, int], RegularGridInterpolator] = {}
        self._edge: RegularGridInterpolator | None = None

    @property
    def names(self) -> tuple[str, ...]:
        """The vocabulary names held."""
        return tuple(self._values)

    @property
    def citation(self) -> tuple[str, ...]:
        """The works it rests on, one a string."""
        return self._citation

    def cite(self) -> str:
        """The works it rests on, one a line."""
        return "\n".join(self._citation)

    @property
    def radii(self) -> np.ndarray:
        """The radii of its slices (m), a repeat at each discontinuity."""
        return self._data.radius

    @property
    def kinds(self) -> Mapping[str, str | None]:
        """Each name's kind: None for values, else a perturbation's."""
        return MappingProxyType(self._kinds)

    @property
    def projection(self) -> PolarStereographic | None:
        """A projected grid's projection; None for a box."""
        return self._data.projection if self._projected else None

    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        d = self._data
        return (d.y, d.x) if self._projected else (d.theta, d.phi)

    @property
    def extent(self) -> tuple[tuple[float, float], tuple[float, float],
                              tuple[float, float]]:
        """The grid's box in its own coordinates: radii (m), then
        colatitudes and longitudes (radians, the longitudes contiguous) for
        a box, northings and eastings (m) for a projected grid, each (low,
        high)."""
        a, b = self._axes()
        r = self._data.radius
        return ((float(r[0]), float(r[-1])), (float(a[0]), float(a[-1])),
                (float(b[0]), float(b[-1])))

    def coordinates(self, theta: ArrayLike, phi: ArrayLike
                    ) -> tuple[np.ndarray, np.ndarray]:
        """The grid's own coordinates of points: colatitude, and longitude
        reduced into the turn the box's longitudes start; or northing and
        easting by the projection."""
        if self._projected:
            x, y = self._data.projection.from_angles(theta, phi)
            return y, x
        start = float(self._data.phi[0])
        phi = start + np.mod(np.asarray(phi, dtype=float) - start, 2 * np.pi)
        return np.broadcast_arrays(np.asarray(theta, dtype=float), phi)

    def _in_box(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        (_, _), (a0, a1), (b0, b1) = self.extent
        return (a >= a0) & (a <= a1) & (b >= b0) & (b <= b1)

    def contains(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """Whether points lie inside the region: the box, or the part of a
        projected grid's plane where it holds values."""
        if self._projected:
            return self.edge_distance(theta, phi) > 0.0
        return self._in_box(*self.coordinates(theta, phi))

    def edge_distance(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The angle (radians) of points inside the region's edge, by the
        rule of the class docstring, zero outside it; a box's edge at a
        pole, or its longitudes round the whole circle, is no edge
        (infinity where nothing else is)."""
        if self._projected:
            return self._footprint_distance(theta, phi)
        (_, _), (t0, t1), (p0, p1) = self.extent
        t, p = self.coordinates(theta, phi)
        d = np.full(t.shape, np.inf)
        if t0 > 1e-12:
            d = np.minimum(d, t - t0)
        if t1 < np.pi - 1e-12:
            d = np.minimum(d, t1 - t)
        if p1 - p0 < 2.0 * np.pi * (1.0 - 1e-6):
            sn = np.sin(t)
            d = np.minimum(d, np.arcsin(np.clip(sn * np.abs(np.sin(p - p0)), 0.0, 1.0)))
            d = np.minimum(d, np.arcsin(np.clip(sn * np.abs(np.sin(p1 - p)), 0.0, 1.0)))
        return np.where(self._in_box(t, p), d, 0.0)

    def _footprint_distance(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """A projected grid's edge distance: on its plane, the distance of
        each node holding a value at some radius to the nearest one that
        holds none or to the plane's border, read bilinearly at the point
        and divided by the projection's scale and the semi-major axis."""
        from scipy.ndimage import distance_transform_edt
        y, x = self._axes()
        if self._edge is None:
            held = np.zeros((y.size, x.size), dtype=bool)
            for v in self._values.values():
                held |= np.isfinite(v).any(axis=0)
            padded = np.pad(held, 1)
            dy, dx = float(np.median(np.diff(y))), float(np.median(np.diff(x)))
            d = distance_transform_edt(padded, sampling=(dy, dx))[1:-1, 1:-1]
            self._edge = RegularGridInterpolator((y, x), d, bounds_error=False,
                                                 fill_value=0.0)
        a, b = self.coordinates(theta, phi)
        flat = self._edge(np.stack([np.ravel(a), np.ravel(b)], axis=-1))
        proj = self._data.projection
        lat = proj.geodetic_latitude(np.broadcast_to(np.asarray(theta, dtype=float),
                                                     a.shape))
        flat = flat.reshape(a.shape)
        return flat / (proj.scale_factor(lat) * proj.semi_major_axis)

    def __call__(self, name: str, r: ArrayLike, theta: ArrayLike, phi: ArrayLike
                 ) -> np.ndarray:
        """The variable `name` at points, NaN outside the region or support."""
        if name not in self._values:
            raise KeyError(f"the grid holds {self.names}, not {name!r}")
        a, b = self.coordinates(theta, phi)
        r, a, b = np.broadcast_arrays(np.asarray(r, dtype=float), a, b)
        out = np.full(r.shape, np.nan)
        radius = self._data.radius
        first, second = self._axes()
        for k, s in enumerate(self._segments):
            lo, hi = radius[s][0], radius[s][-1]
            top = k == len(self._segments) - 1
            inside = (r >= lo) & ((r <= hi) if top else (r < hi))
            if not inside.any():
                continue
            key = (name, k)
            if key not in self._interp:
                self._interp[key] = RegularGridInterpolator(
                    (radius[s], first, second), self._values[name][s],
                    bounds_error=False, fill_value=np.nan)
            out[inside] = self._interp[key](np.stack([r[inside], a[inside], b[inside]],
                                                     axis=-1))
        return out


@dataclass(frozen=True)
class GridReader:
    """The reader of a data set of one gridded file: `variables` maps its
    names to (vocabulary name, kind); the other fields go to `read_grid`
    and `grid_model` as the entry knows them (a mislabelled unit, the
    surface radius, discontinuity pairs, a glitched coordinate, a seam or
    pole rows to average, a fill, the lateral rule, a stride, a window of
    the vertical axis, a projected grid's projection where its file states
    it wrongly or not at all); `log10` the variables that hold the base-10
    logarithm of their quantity, in the unit `units` gives it, and `clip`
    a floor and a cap in SI for a variable whose source leaves its values
    unbounded, both applied as the file is read."""

    variables: Variables
    _: KW_ONLY
    units: Mapping[str, str] | None = None
    surface: float | None = None
    vertical_units: str | None = None
    pairs: tuple[tuple[float, float], ...] = ()
    regular: bool = False
    seam: str = "refuse"
    pole_rows: str = "refuse"
    fill: float | None = None
    lateral: str = "linear"
    stride: int = 1
    vertical: tuple[float | None, float | None] | None = None
    projection: PolarStereographic | None = None
    log10: tuple[str, ...] = ()
    clip: Mapping[str, tuple[float, float]] | None = None

    def _prepared(self, data: Gridded) -> Gridded:
        """The data with `log10` and `clip` applied."""
        if not self.log10 and not self.clip:
            return data
        values = dict(data.values)
        for key in self.log10:
            values[key] = 10.0 ** np.asarray(values[key], dtype=float)
        for key, (lo, hi) in (self.clip or {}).items():
            if not lo < hi:
                raise ValueError(f"the floor of {key!r} is not below its cap")
            factor = parse_unit((self.units or {}).get(key, data.units[key]))[1]
            values[key] = np.clip(values[key], lo / factor, hi / factor)
        return replace(data, values=values)

    def read(self, paths: tuple[Path, ...], dataset: Dataset, *,
             reference: Model | None = None, scales: Scales | None = None,
             stride: int | None = None) -> Model | RegionalGrid:
        """A global grid's model (over its reference when it holds
        perturbations), in SI or `scales`; a regional grid's
        `RegionalGrid`."""
        if len(paths) != 1:
            raise ValueError(f"{dataset.name}: a grid is one file, given {len(paths)}")
        data = read_grid(paths[0], variables=tuple(self.variables),
                         surface=self.surface, vertical_units=self.vertical_units,
                         pairs=self.pairs, regular=self.regular, seam=self.seam,
                         pole_rows=self.pole_rows,
                         stride=self.stride if stride is None else stride,
                         vertical=self.vertical, projection=self.projection)
        data = self._prepared(data)
        if not data.is_global:
            return RegionalGrid(data, self.variables, units=self.units,
                                reference=dataset.reference, citation=dataset.citation)
        kinds = {k for _, k in self.variables.values()}
        ref = None if kinds == {None} else reference_of(dataset, reference=reference)
        model = grid_model(data, self.variables, reference=ref, units=self.units,
                           lateral=self.lateral, fill=self.fill, name=dataset.name,
                           citation=dataset.citation)
        return model if scales is None else model.converted(scales)
