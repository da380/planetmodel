"""Strata: layered surfaces with the properties between them.

A crustal or lithospheric model (CRUST1.0, LITHO1.0, LithoRef18) is a
stack of strata from the top down, each between two surfaces, with
properties inside it.  `Stratum` holds one: its `top` and `bottom`, each
a `Surface`, and each property as the pair of its values at the top and
at the bottom, functions of the angles in SI, the property linear in
radius between (a pair of one function twice: constant); `value` reads a
property at a radius inside it.  `Strata` holds the stack, each
stratum's bottom the next one's top (`Strata.from_surfaces` builds it
from the surfaces and the properties), and the properties of what lies
beneath the last (`below`: CRUST1.0's mantle beneath the Moho).

A stratum absent from a cell has zero thickness there, its surfaces
coinciding; its properties there are those of the next present stratum
below at its top, so that a property read by the lateral rule into the
thin edge of a stratum stays between the values either side, never
mixing in the zeros or blanks a source gives for what is not there.

- `read_crust1`: CRUST1.0's own files, nine surfaces as elevations from
  sea level (the Earth's mean radius, 6371 km), eight strata (water, ice,
  three sediments, three crystalline crust) with vp, vs and rho, and the
  mantle beneath the Moho, on its 1-degree cells (`CRUST1_STRATA`);
- `litho1_strata`: LITHO1.0 from the Earth Model Collaboration's 1-degree
  grid (`read_grid` with its glitched latitudes rebuilt), the same
  strata and the lithospheric mantle, the lid, whose bottom is the
  lithosphere-asthenosphere boundary, as depths below sea level, with
  the eight properties of a mineos card at each stratum's top and bottom
  (`LITHO1_STRATA`, `LITHO1_PROPERTIES`); each surface is its stratum's
  top, held no shallower than the one above, since the grid's
  interpolation leaves a few cells where a stratum's bottom lies below
  the next one's top;
- a table's columns (`TableStrataReader`, LithoRef18's).

The surfaces are each a `GriddedFunction` on the source's grid, read
bilinearly between its cell centres, with the mean of the nearest ring
at each pole (`grids.ring_poles`).  `Crust1Reader`, `Litho1Reader` and
`TableStrataReader` are the registry's readers, whose strata and every
surface of them cite the entry's works.
"""
from __future__ import annotations

import os
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import ArrayLike

from ..gridded import GriddedFunction
from ..model import cited
from ..sampling import AngularGrid, equiangular
from ..units import parse_unit
from .grids import EARTH_RADIUS, GridData, read_grid, read_table, ring_poles
from .surfaces import Surface

if TYPE_CHECKING:
    from ..displacement import Relief
    from .registry import Dataset

__all__ = ["CRUST1_STRATA", "LITHO1_STRATA", "LITHO1_PROPERTIES", "Property",
           "Stratum", "Strata", "read_crust1", "litho1_strata", "Crust1Reader",
           "Litho1Reader", "TableStrataReader"]

#: CRUST1.0's strata, from the top down.
CRUST1_STRATA = ("water", "ice", "upper_sediments", "middle_sediments",
                 "lower_sediments", "upper_crust", "middle_crust", "lower_crust")

#: LITHO1.0's strata, from the top down: CRUST1.0's and the lid.
LITHO1_STRATA = CRUST1_STRATA + ("lid",)

#: LITHO1.0's properties by the vocabulary names they are: its density,
#: its velocities (vertical, then horizontal, as a mineos card's), its
#: quality factors and eta.
LITHO1_PROPERTIES = MappingProxyType({
    "density": "rho", "vp": "vpv", "vs": "vsv", "vp2": "vph", "vs2": "vsh",
    "qkappa": "qkappa", "qmu": "qmu", "eta": "eta"})

#: A property of a stratum: its values at the top and at the bottom.
type Property = tuple[Relief, Relief]


@dataclass(frozen=True, eq=False)
class Stratum:
    """One stratum, by the rules of the module docstring: `name`, its `top`
    and `bottom` surfaces, and its `properties`, each a pair (top,
    bottom) of functions of the angles in SI."""

    name: str
    top: Surface
    bottom: Surface
    _: KW_ONLY
    properties: Mapping[str, Property] = field(default_factory=dict)

    def thickness(self, theta: ArrayLike, phi: ArrayLike) -> np.ndarray:
        """The top's radius less the bottom's, at (theta, phi)."""
        return self.top.radius(theta, phi) - self.bottom.radius(theta, phi)

    def value(self, name: str, r: ArrayLike, theta: ArrayLike,
              phi: ArrayLike) -> np.ndarray:
        """The property `name` at radius `r` and (theta, phi), broadcast
        together: linear in radius between its values at the top and the
        bottom, the top's where the stratum has no thickness."""
        if name not in self.properties:
            raise KeyError(f"{self.name} has no property {name!r}; it has "
                           f"{sorted(self.properties)}")
        r, t, p = np.broadcast_arrays(np.asarray(r, dtype=float),
                                      np.asarray(theta, dtype=float),
                                      np.asarray(phi, dtype=float))
        at_top, at_bottom = self.properties[name]
        upper = np.asarray(at_top(t, p), dtype=float)
        if at_bottom is at_top:
            return np.broadcast_to(upper, r.shape).copy()
        lower = np.asarray(at_bottom(t, p), dtype=float)
        a, b = self.top.radius(t, p), self.bottom.radius(t, p)
        span = a - b
        with np.errstate(divide="ignore", invalid="ignore"):
            s = np.where(span > 0.0, (a - r) / span, 0.0)
        return upper + s * (lower - upper)

    def __repr__(self) -> str:
        return f"Stratum({self.name}: {sorted(self.properties)})"


@dataclass(frozen=True, eq=False)
class Strata:
    """Strata from the top down, each one's bottom the next one's top, and
    `below`, the properties beneath the last, by the rules of the module
    docstring; `name` what they are, `citation` the works they rest on."""

    strata: tuple[Stratum, ...]
    _: KW_ONLY
    below: Mapping[str, Relief] = field(default_factory=dict)
    name: str | None = None
    citation: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        strata = tuple(self.strata)
        if not strata:
            raise ValueError("strata hold one stratum or more")
        names = [s.name for s in strata]
        if len(set(names)) != len(names):
            raise ValueError(f"the strata's names repeat: {names}")
        for upper, lower in zip(strata[:-1], strata[1:]):
            if upper.bottom is not lower.top:
                raise ValueError(f"{upper.name}'s bottom is not {lower.name}'s top")
        object.__setattr__(self, "strata", strata)
        object.__setattr__(self, "below", MappingProxyType(dict(self.below)))
        object.__setattr__(self, "citation", cited((), self.citation))

    @classmethod
    def from_surfaces(cls, surfaces: Sequence[Surface], names: Sequence[str],
                      properties: Mapping[str, Mapping[str, Property]], *,
                      below: Mapping[str, Relief] | None = None,
                      name: str | None = None) -> Strata:
        """Strata between `surfaces` (one more than `names`), from the top
        down, each stratum's properties from `properties` by its name."""
        surfaces = tuple(surfaces)
        if len(surfaces) != len(names) + 1:
            raise ValueError(f"{len(names)} strata lie between {len(names) + 1} "
                             f"surfaces, not {len(surfaces)}")
        strata = tuple(Stratum(n, surfaces[k], surfaces[k + 1],
                               properties=dict(properties.get(n, {})))
                       for k, n in enumerate(names))
        return cls(strata, below=dict(below or {}), name=name)

    @property
    def names(self) -> tuple[str, ...]:
        """The strata's names, from the top down."""
        return tuple(s.name for s in self.strata)

    @property
    def surfaces(self) -> tuple[Surface, ...]:
        """Every stratum's top and the last one's bottom, from the top down."""
        return tuple(s.top for s in self.strata) + (self.strata[-1].bottom,)

    def __getitem__(self, which: int | str) -> Stratum:
        """A stratum by index or name."""
        if isinstance(which, str):
            for s in self.strata:
                if s.name == which:
                    return s
            raise KeyError(f"no stratum {which!r}; the strata are {list(self.names)}")
        return self.strata[which]

    def cite(self) -> str:
        """The works the strata rest on, one a line."""
        return "\n".join(self.citation)

    def with_citation(self, *works: str) -> Strata:
        """The same strata citing `works` too, after their own, and every
        surface of them too."""
        surfaces = tuple(s.with_citation(*works) for s in self.surfaces)
        strata = tuple(Stratum(s.name, surfaces[k], surfaces[k + 1],
                               properties=s.properties)
                       for k, s in enumerate(self.strata))
        return replace(self, strata=strata, below=dict(self.below),
                       citation=cited(self.citation, works))

    def __len__(self) -> int:
        return len(self.strata)

    def __iter__(self) -> Iterator[Stratum]:
        return iter(self.strata)

    def __repr__(self) -> str:
        below = f", below: {sorted(self.below)}" if self.below else ""
        return f"Strata({self.name or 'unnamed'}: {', '.join(self.names)}{below})"


# -- building strata from values on a grid ---------------------------------------

def _gridded(grid: AngularGrid, values: np.ndarray) -> GriddedFunction:
    """Values on a grid without its poles as a bilinear gridded function."""
    return GriddedFunction(grid, values, lateral="linear",
                           poles=ring_poles(values, grid))


def _filled(tops: np.ndarray, bottoms: np.ndarray, absent: np.ndarray
            ) -> tuple[np.ndarray, np.ndarray]:
    """(strata, ...) values at the tops and bottoms, an absent stratum's
    both the next present one's at its top (the last stratum's own where
    none is below)."""
    tops, bottoms = np.array(tops, dtype=float), np.array(bottoms, dtype=float)
    for k in range(tops.shape[0] - 2, -1, -1):
        tops[k] = np.where(absent[k], tops[k + 1], tops[k])
        bottoms[k] = np.where(absent[k], tops[k + 1], bottoms[k])
    return tops, bottoms


def _strata_on_grid(grid: AngularGrid, levels: np.ndarray, measures: str,
                    names: Sequence[str],
                    properties: Mapping[str, tuple[np.ndarray, np.ndarray]],
                    below: Mapping[str, np.ndarray], *, name: str) -> Strata:
    """Strata from the surfaces' values (surfaces, ntheta, nphi) in metres
    measured by `measures` from sea level, and each property's values at
    the strata's tops and bottoms (strata, ntheta, nphi) in SI."""
    surfaces = tuple(Surface(_gridded(grid, v), measures, datum=EARTH_RADIUS,
                             name=f"{name}: {where}")
                     for v, where in zip(levels, [f"top of {n}" for n in names]
                                         + [f"bottom of {names[-1]}"]))
    absent = np.diff(levels, axis=0) == 0.0
    props: dict[str, dict[str, Property]] = {n: {} for n in names}
    for prop, (tops, bottoms) in properties.items():
        tops, bottoms = _filled(tops, bottoms, absent)
        for k, n in enumerate(names):
            if not (np.all(np.isfinite(tops[k])) and np.all(np.isfinite(bottoms[k]))):
                raise ValueError(f"{name}: {prop} of {n} is missing where {n} is "
                                 "present")
            upper = _gridded(grid, tops[k])
            same = np.array_equal(tops[k], bottoms[k])
            props[n][prop] = (upper, upper if same else _gridded(grid, bottoms[k]))
    return Strata.from_surfaces(surfaces, names, props,
                                below={p: _gridded(grid, v) for p, v in below.items()},
                                name=name)


# -- CRUST1.0 ---------------------------------------------------------------------

def read_crust1(paths: Sequence[str | os.PathLike[str]], *,
                name: str = "CRUST1.0") -> Strata:
    """CRUST1.0's strata from its own files, `crust1.bnds`, `.vp`, `.vs`
    and `.rho`, found among `paths` by their suffixes: each 64800 lines of
    nine values, a cell a line from (89.5 N, 179.5 W), longitude inner,
    west to east, then rows north to south; boundaries in km as
    elevations, positive up, refused where they rise downward; vp and vs
    in km/s, rho in g/cm^3; the ninth column of each property the mantle
    beneath the Moho.  In SI, by the rules of the module docstring."""
    found = {Path(os.fspath(p)).suffix: Path(os.fspath(p)) for p in paths}
    for suffix in (".bnds", ".vp", ".vs", ".rho"):
        if suffix not in found:
            raise ValueError(f"CRUST1.0's {suffix} file is not among the paths "
                             f"{[os.fspath(p) for p in paths]}")
    tables = {}
    for suffix in (".bnds", ".vp", ".vs", ".rho"):
        t = np.loadtxt(found[suffix], ndmin=2)
        if t.shape != (64800, 9):
            raise ValueError(f"{found[suffix]}: {t.shape[0]} lines of {t.shape[1]} "
                             "values, not CRUST1.0's 64800 of 9")
        tables[suffix] = t.reshape(180, 360, 9)
    lon = -179.5 + np.arange(360.0)
    order = np.argsort(np.mod(lon, 360.0), kind="stable")
    grid = equiangular(180, 360, phi0=np.radians(lon[0]))
    data = {k: np.moveaxis(t[:, order], -1, 0) for k, t in tables.items()}
    levels = 1e3 * data[".bnds"]                         # (9, 180, 360), metres
    if np.any(np.diff(levels, axis=0) > 0.0):
        raise ValueError(f"{found['.bnds']}: a boundary lies above the one before it")
    scale = {".vp": ("vp", 1e3), ".vs": ("vs", 1e3), ".rho": ("rho", 1e3)}
    properties, below = {}, {}
    for suffix, (prop, factor) in scale.items():
        v = factor * data[suffix]
        properties[prop] = (v[:8], v[:8])
        below[prop] = v[8]
    return _strata_on_grid(grid, levels, "elevation", CRUST1_STRATA, properties,
                           below, name=name)


class Crust1Reader:
    """The reader of CRUST1.0's files: `read(paths, dataset)` gives its
    `Strata` by `read_crust1`."""

    def read(self, paths: tuple[Path, ...], dataset: Dataset) -> Strata:
        """CRUST1.0's strata."""
        return read_crust1(paths, name=dataset.name).with_citation(*dataset.citation)

    def __repr__(self) -> str:
        return "Crust1Reader()"


# -- LITHO1.0 -----------------------------------------------------------------------

def litho1_strata(data: GridData, *, name: str = "LITHO1.0") -> Strata:
    """LITHO1.0's strata from the Earth Model Collaboration's 1-degree
    grid as `read_grid` gives it (`<stratum>_{top,bottom}_<property>`,
    depths in km positive down, missing where a stratum is absent), by
    the rules of the module docstring: surfaces as depths in metres below
    sea level, each its stratum's top and no shallower than the one
    above, the last the lid's bottom; properties in SI by their units."""
    if data.grid is None or data.radius is not None:
        raise ValueError("LITHO1.0 is a global grid of surfaces")

    def read(var: str) -> np.ndarray:
        if var not in data.values:
            raise KeyError(f"{data.source}: no variable {var!r}")
        unit = data.units.get(var, "")
        factor = parse_unit(unit)[1] if unit.strip() else 1.0
        return factor * np.asarray(data.values[var], dtype=float)

    tops = np.stack([read(f"{n}_top_depth") for n in LITHO1_STRATA])
    bottom = read("lid_bottom_depth")
    absent = np.isnan(tops)
    if absent[-1].any() or np.isnan(bottom).any():
        raise ValueError(f"{data.source}: the lid is absent somewhere")
    levels = np.concatenate([tops, bottom[None]])
    for k in range(levels.shape[0] - 2, -1, -1):          # absent: the next top
        levels[k] = np.where(np.isnan(levels[k]), levels[k + 1], levels[k])
    levels = np.maximum.accumulate(levels, axis=0)       # none shallower above
    properties = {}
    for source, vocabulary in LITHO1_PROPERTIES.items():
        properties[vocabulary] = tuple(
            np.stack([read(f"{n}_{end}_{source}") for n in LITHO1_STRATA])
            for end in ("top", "bottom"))
    below = {vocabulary: read(f"asthenospheric_mantle_top_{source}")
             for source, vocabulary in LITHO1_PROPERTIES.items()}
    return _strata_on_grid(data.grid, levels, "depth", LITHO1_STRATA, properties,
                           below, name=name)


class Litho1Reader:
    """The reader of LITHO1.0's 1-degree grid from the Earth Model
    Collaboration: `read(paths, dataset)` gives its `Strata` by
    `litho1_strata`, the grid's glitched latitudes rebuilt as the regular
    ones (netCDF4, the `netcdf` extra)."""

    def read(self, paths: tuple[Path, ...], dataset: Dataset) -> Strata:
        """LITHO1.0's strata."""
        strata = litho1_strata(read_grid(paths[0], regular=True), name=dataset.name)
        return strata.with_citation(*dataset.citation)

    def __repr__(self) -> str:
        return "Litho1Reader()"


# -- tables -------------------------------------------------------------------------

class TableStrataReader:
    """The reader of strata given as a table's columns (`grids.read_table`):
    `surfaces` the columns of the surfaces from the top down, measuring
    `measures` (elevation or depth) from sea level; `names` the strata
    between them; `properties` each stratum's properties, {stratum:
    {property: column}}, constant through it; `below` the properties
    beneath the last, {property: column}; `units` each column's unit
    string, which a table does not state; `longitude` and `latitude` the
    coordinates' columns."""

    def __init__(self, *, surfaces: tuple[str, ...], names: tuple[str, ...],
                 measures: str, properties: Mapping[str, Mapping[str, str]],
                 below: Mapping[str, str] | None = None,
                 units: Mapping[str, str] | None = None, longitude: str = "LONG",
                 latitude: str = "LAT") -> None:
        if measures not in ("elevation", "depth"):
            raise ValueError(f"measures must be 'elevation' or 'depth', got "
                             f"{measures!r}")
        if len(surfaces) != len(names) + 1:
            raise ValueError(f"{len(names)} strata lie between {len(names) + 1} "
                             f"surfaces, not {len(surfaces)}")
        self.surfaces, self.names = tuple(surfaces), tuple(names)
        self.measures = measures
        self.properties = {n: dict(p) for n, p in properties.items()}
        self.below = dict(below or {})
        self.units = dict(units or {})
        self.longitude, self.latitude = longitude, latitude

    def read(self, paths: tuple[Path, ...], dataset: Dataset) -> Strata:
        """The strata the table holds."""
        columns = tuple(dict.fromkeys(
            self.surfaces + tuple(c for p in self.properties.values()
                                  for c in p.values()) + tuple(self.below.values())))
        data = read_table(paths[0], columns=columns, units=self.units,
                          longitude=self.longitude, latitude=self.latitude)
        if data.grid is None:
            raise ValueError(f"{paths[0]}: the table does not cover the sphere")

        def si(column: str) -> np.ndarray:
            unit = data.units.get(column, "")
            factor = parse_unit(unit)[1] if unit.strip() else 1.0
            return factor * np.asarray(data.values[column], dtype=float)

        levels = np.stack([si(c) for c in self.surfaces])
        step = np.diff(levels, axis=0)
        if np.any(step > 0.0 if self.measures == "elevation" else step < 0.0):
            raise ValueError(f"{paths[0]}: a surface lies above the one before it")
        props: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        names = sorted({p for each in self.properties.values() for p in each})
        for prop in names:
            values = np.stack([si(self.properties[n][prop]) if prop in
                               self.properties.get(n, {}) else
                               np.zeros(levels.shape[1:]) for n in self.names])
            props[prop] = (values, values)
        strata = _strata_on_grid(data.grid, levels, self.measures, self.names, props,
                                 {p: si(c) for p, c in self.below.items()},
                                 name=dataset.name)
        return strata.with_citation(*dataset.citation)

    def __repr__(self) -> str:
        return f"TableStrataReader({', '.join(self.names)})"
