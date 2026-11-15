"""The flat surface file: fields on a horizontal plane of a flat body,
static or in time.

The counterpart of the surface file (`planetmodel.surfacefile`) for a
body of `planetmodel.cartesian`: one netCDF-4 file of schema
`surfacefile.SCHEMA_FLAT`, `planetmodel.surface/2`, holds named scalar
fields on one horizontal grid of one plane (a topography, an ice
thickness history, the kernels of a measurement on the free surface),
in the writer's units with its scales recorded, CF-styled.  Everything
`surfacefile` describes holds here (the global attributes `source`,
`created`, `title`, the scales, `citation` and `meta_<key>`; the time
axes `time`, `instant` and `interval`; each field's `time_support`,
`time_interpolation`, `lateral_interpolation`, `dimensions`,
`long_name` and `units`; the time rules, their tolerance and
extrapolation, `surfacefile.time_rows` and `HistoryBase` being shared),
except where the surface differs, which is this module's layout,
`FlatSurfaceLayout`; the Python here and in `surfacefile` is the
definition of the format, and `docs/source/formats/surface_file.rst`
describes it for consumers.

**The plane and its grid.**  `coordinate_system` is `flat`; `depth`,
where the writer states it, is the depth of the plane in the reference
body, in the file's units, and absent the fields lie on the free
surface.  The lateral domain is stated as the flat model file states it
(`domain`, `domain_bounds`, `domain_padding`; `cartesian.modelfile`),
and the grid is `cartesian.Grid` over it, its nodes the CF coordinate
variables `x(x)` and, over a box, `y(y)`.  A field is `name(x, y)`
(`name(x)` over a section) after its support's leading dimension; its
lateral rule is one of the flat rules (`cartesian.lateral`: `linear`,
`cubic`, `fourier`).  A flat grid has no poles: polar values, a grid's
kind, band and weights, `latitude`, `longitude` and `radius` are a
spherical file's, refused on writing and on reading.

`write_surface` writes a file from `surfacefile.SurfaceVariable`s;
`read_surface_file` gives one back as a `SurfaceFile` of numbers and
metadata, and refuses a spherical file by its schema;
`surfacefile.read_surface` gives either kind as a function of time, a
flat file as a `SurfaceHistory`, whose `function` is a
`cartesian.GriddedFunction` on the file's grid by the field's lateral
rule.  netCDF4 is imported where a file is opened.
"""
from __future__ import annotations

import numbers
import os
from collections.abc import Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike

from ..modelfile import FileVariable
from ..surfacefile import (SCHEMA_FLAT, TIME_RTOL, HistoryBase, SurfaceVariable,
                           malformed, read_with, write_with)
from ..units import LENGTH, Scales, scaled_unit
from .coordinates import FLAT
from .grid import Grid
from .lateral import METHODS
from .modelfile import (axes_of, check_grid_variables, grid_variables, read_domain,
                        refuse_spherical, write_domain)

if TYPE_CHECKING:
    from .gridded import GriddedFunction

__all__ = ["SurfaceFile", "FlatSurfaceLayout", "LAYOUT", "SurfaceHistory",
           "write_surface", "read_surface_file"]


@dataclass(frozen=True)
class SurfaceFile:
    """A flat surface file's numbers and metadata, as `read_surface_file`
    gives them: nothing is interpolated or converted."""

    schema: str
    grid: Grid
    variables: Mapping[str, FileVariable]
    _: KW_ONLY
    scales: Scales = Scales.SI
    time: np.ndarray | None = None
    instant: np.ndarray | None = None
    #: The depth of the plane in the file's units, where it states one; the
    #: fields lie on the free surface where it states none.
    depth: float | None = None
    source: str = ""
    created: str = ""
    title: str | None = None
    meta: Mapping[str, Any] = MappingProxyType({})
    citation: tuple[str, ...] = ()


class FlatSurfaceLayout:
    """The flat surface file's `surfacefile.SurfaceLayout`: the module
    docstring's plane and grid.  Its context is the file's grid."""

    schema = SCHEMA_FLAT
    reserved = ("x", "y", "time", "instant")
    methods = METHODS
    place = "depth"

    def check_grid(self, grid: object) -> None:
        """A `cartesian.Grid`."""
        if not isinstance(grid, Grid):
            raise TypeError(f"expected a cartesian.Grid, got {type(grid).__name__}")

    def check_place(self, value: object) -> None:
        """A finite depth, not negative."""
        if not (isinstance(value, numbers.Real) and np.isfinite(value)
                and value >= 0.0):
            raise ValueError(f"depth must be finite and not negative, got {value!r}")

    def check_poles(self, grid: Grid, name: str) -> None:
        """Refused: a flat grid has no poles."""
        raise ValueError(f"{name!r} gives polar values, and a flat grid has no poles")

    def lateral_dims(self, grid: Grid) -> tuple[str, ...]:
        """(x, y) over a box, (x,) over a section."""
        return axes_of(grid.domain)

    def lateral_shape(self, grid: Grid) -> tuple[int, ...]:
        """The grid's shape."""
        return grid.shape

    def write_grid(self, root: Any, grid: Grid, scales: Scales) -> None:
        """`coordinate_system`, the domain, and the grid's coordinates."""
        root.setncattr("coordinate_system", FLAT.name)
        write_domain(root, grid.domain)
        for axis, n in zip(axes_of(grid.domain), grid.shape):
            root.createDimension(axis, n)
        grid_variables(root, grid, scaled_unit(LENGTH, scales))

    def validate_place(self, ds: Any, path: object) -> None:
        """`depth`, where stated, a finite number, not negative."""
        if "depth" in ds.ncattrs():
            depth = ds.getncattr("depth")
            if (isinstance(depth, str) or np.size(depth) != 1
                    or not np.isfinite(float(depth)) or float(depth) < 0.0):
                malformed(path, f"depth is {depth!r}, not a depth")

    def validate_grid(self, ds: Any, path: object) -> Grid:
        """`coordinate_system` `flat`, no spherical statement, the domain,
        and coordinates that are a grid's over it, which is the context."""
        def fail(msg: str) -> NoReturn:
            malformed(path, msg)

        got = ds.getncattr("coordinate_system") if (
            "coordinate_system" in ds.ncattrs()) else None
        if got != FLAT.name:
            fail(f"coordinate_system is {got!r}, and a flat surface file states "
                 f"{FLAT.name!r}")
        refuse_spherical(ds, fail)
        return check_grid_variables(ds, read_domain(ds, fail), fail)

    def axes(self, context: Grid) -> tuple[str, ...]:
        """The grid's dimensions."""
        return axes_of(context.domain)

    def read_grid(self, ds: Any, context: Grid) -> Grid:
        """The grid `validate_grid` read."""
        return context

    def file(self, schema: str, grid: Grid, variables: Mapping[str, FileVariable],
             place: float | None, **metadata: Any) -> SurfaceFile:
        """A flat `SurfaceFile`, `place` its depth."""
        return SurfaceFile(schema, grid, variables, depth=place, **metadata)

    def history(self, file: SurfaceFile) -> SurfaceHistory:
        """A flat `SurfaceHistory`."""
        return SurfaceHistory(file)


#: The flat surface layout, the one of every file of `SCHEMA_FLAT`.
LAYOUT = FlatSurfaceLayout()


class SurfaceHistory(HistoryBase):
    """A flat surface file as a function of time (`surfacefile.HistoryBase`),
    a field at a date a `cartesian.GriddedFunction` by its lateral rule."""

    def __init__(self, file: SurfaceFile) -> None:
        super().__init__(file)

    @property
    def file(self) -> SurfaceFile:
        """The file's numbers and metadata, as read."""
        return self._file

    @property
    def grid(self) -> Grid:
        """The horizontal grid every field is on."""
        return self._file.grid

    def function(self, name: str, *, date: float | None = None,
                 extrapolation: str = "refuse",
                 time_rtol: float = TIME_RTOL) -> GriddedFunction:
        """The field at `date` as a `cartesian.GriddedFunction` on the
        file's grid by its stated lateral rule; a field with NaN cannot be
        one.  `extrapolation` and `time_rtol` are `values`'."""
        from .gridded import GriddedFunction
        lateral = str(self._variable(name).attributes["lateral_interpolation"])
        return GriddedFunction(self.grid, self.values(name, date=date,
                                                     extrapolation=extrapolation,
                                                     time_rtol=time_rtol),
                               lateral=lateral)


def write_surface(path: str | os.PathLike[str], grid: Grid, *,
                  fields: Mapping[str, SurfaceVariable], source: str,
                  time: ArrayLike | None = None, instant: ArrayLike | None = None,
                  depth: float | None = None, scales: Scales = Scales.SI,
                  title: str | None = None, meta: Mapping[str, Any] | None = None,
                  citation: Sequence[str] = ()) -> Path:
    """Write `fields` on the horizontal `grid` (a `cartesian.Grid`) to a
    flat surface file at `path`, returned: `surfacefile.write_surface`'s
    arguments, the plane's `depth` in place of a sphere's radius (the free
    surface where it is None), and no polar values, which are refused."""
    return write_with(path, grid, LAYOUT, fields=fields, source=source, time=time,
                      instant=instant, place=depth, scales=scales, title=title,
                      meta=meta, citation=citation)


def read_surface_file(path: str | os.PathLike[str]) -> SurfaceFile:
    """A flat surface file's numbers and metadata, its structure validated
    first; a spherical file is refused by its schema, naming its reader
    (`surfacefile.read_surface_file`)."""
    return read_with(path, (LAYOUT,))
