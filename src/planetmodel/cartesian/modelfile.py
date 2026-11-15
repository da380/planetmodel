"""The flat model file: a model of `planetmodel.cartesian` in one netCDF file.

A flat model is written to a model file by `modelfile.write_model`, as a
spherical one is, and read back by `modelfile.read_model` as a
`cartesian.Model`; the file is of schema `modelfile.SCHEMA_FLAT`,
`planetmodel.model/3`, which the readers of the spherical schemas refuse
by name.  Everything `modelfile` describes for the spherical file holds
here (the global attributes, units and scales, constants, citation and
meta; parts of contiguous layers at the GLL nodes of their elements;
fields of every rank with their characters, dimensions and rules;
references with perturbations; the radial rules, those in the logarithm
among them; the exact layout; the groups `geometry` and `surfaces`)
except where the geometry differs, which is this module's: the layout
`FlatLayout` the shared writer and readers ask.  The Python here and
in `modelfile` is the definition of the format;
`docs/source/formats/model_file.rst` describes it for consumers.

**The root** states the system and the lateral domain beside the shared
attributes: `coordinate_system`, `flat` (`FLAT.name`); `local_frame`,
`up x y`, the local frame (e_up, e_x, e_y) every component is in;
`domain`, `box` (a `Box`) or `section` (a `Rectangle`, the section of a
body invariant along y); `domain_bounds`, (x0, x1, y0, y1) of a box and
(x0, x1) of a section, and `domain_padding`, (px, py) or (px,), exactly
as the domain holds them, so that its extent, the period of every
Fourier series on it, is rebuilt to the bit.  The skeleton is in depth
from the free surface down: `layer_bounds(layer, bound)` the depths of
each layer's top and bottom, `interface_depth(interface)` the depths of
the interfaces as `cartesian.Geometry` numbers them, interface 0 the
free surface (or a hollowed body's top) and interface k the top of layer
k, the base a face of the box and no interface; parts run from the
surface down, `part_0` holding the top layer.

**A part** holds `depth(node)` (an auxiliary coordinate, CF
`standard_name` `depth` with `positive` `down`) in place of the radius
and the horizontal grid as `cartesian.Grid` states it: the CF coordinate
variables `x(x)` and, over a box, `y(y)`, in the model's unit of length,
the nodes x_i = x0 - px + i Lx / nx of the domain's extent (`Grid`), the
last one step short of the far end; a section has no `y`.  A field is
`name(node)` of depth alone or `name(node, x, y)` (`name(node, x)` over
a section), with the trailing axes of its components as on the sphere,
the components in the local frame, the same at every point; its
`lateral_interpolation` is one of the flat rules (`cartesian.lateral`:
`linear`, `cubic`, `fourier`), and its `coordinates` attribute `depth`.
No grid has poles, a kind, a band or weights: none of `latitude`,
`longitude`, `colatitude_weights`, `grid_kind`, `grid_lmax`, `pole`,
`poles` or `*_poles` is in a flat file, nor `radius` nor
`interface_radius`, and a reader refuses a file that has any of them.

**Fourier fields** are held as the sphere holds a harmonic field: their
values on the part's grid, stating the rule `fourier` where the grid
holds every mode of the series (`Grid.holds`: the same domain, a packing
containing the field's), so that the analysis of the stored values gives
back its coefficients and the series is read between the nodes exactly,
and the default `linear` on a grid that does not; in depth the layout
puts element edges at the breakpoints of the coefficients' piecewise
polynomial, so the field read back is the same function.  A Fourier
function on an interface or as a relief is held the same way.

**The mapping** is the vertical stretch (`cartesian.VerticalStretch`),
in one of two forms, never both; the identity is neither, and a mapping
of any other kind has no form in the file.  A stretch of boundary
reliefs (`cartesian.layer_linear`) is the group `geometry`:
`boundary_depth(boundary)`, the reference depths of the surfaces, the
free surface first, and `boundary_displacement(boundary, x, y)`, each
surface's upward displacement, its height above its reference place
(attribute `positive` `up`: the sign of `cartesian.displacement`), on
the group's own grid, with `lateral_interpolation` and `between`
(`linear`).  Any other stretch is the variable `displacement(node, x,
y)` of every part, its upward displacement w at the node, read back as a
`cartesian.GriddedDisplacement`.

**The surfaces group** holds functions of (x, y) on interfaces, as on
the sphere, on a grid of its own over the domain, read back as
`cartesian.GriddedFunction`s.

`read_model_file` gives a flat file's numbers and metadata as a
`modelfile.ModelFile`, its `domain` the file's, and refuses a spherical
file by its schema; `modelfile.read_model_file` refuses a flat one.
`FlatLayout` is the layout, `LAYOUT` its instance; `write_domain`,
`read_domain`, `grid_variables`, `check_grid_variables`,
`refuse_spherical` and `on_grid` are the statements of a flat grid and
domain that the flat surface file shares.  netCDF4 is imported where a
file is opened.
"""
from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike

from ..character import Character
from ..fields import Field, RadialField
from ..modelfile import (SCHEMA_FLAT, FileGeometry, Interpolation, ModelFile,
                         displacement_fields, malformed, put_variable, read_file,
                         real_values, relief_grid, variable_values)
from ..skeleton import Skeleton
from ..units import BASE, LENGTH, scaled_unit
from ..vocabulary import FieldSpec
from .coordinates import FLAT
from .domains import Box, Domain, Rectangle
from .grid import Grid
from .lateral import METHODS, check_rule

__all__ = ["FlatLayout", "LAYOUT", "LOCAL_FRAME_AXES", "DOMAINS", "SPHERICAL_NAMES",
           "read_model_file", "write_domain", "read_domain", "grid_variables",
           "check_grid_variables", "refuse_spherical", "on_grid", "axes_of"]

#: The axes of the local frame every component is in, as `local_frame`
#: states them: (e_up, e_x, e_y).
LOCAL_FRAME_AXES = ("up", "x", "y")

#: The kinds of lateral domain a flat file states, by `domain`.
DOMAINS = ("box", "section")

#: The statements of a spherical file a flat one never makes: dimensions,
#: attributes and variables of a grid on the sphere, and of the radius.
SPHERICAL_NAMES = {
    "dimensions": ("latitude", "longitude", "pole"),
    "attributes": ("grid_kind", "grid_lmax", "poles", "radius"),
    "variables": ("latitude", "longitude", "colatitude_weights", "radius",
                  "interface_radius", "boundary_radius"),
}

#: Names the flat model file gives its own variables, which no field may take.
_RESERVED = ("depth", "element_layer", "element_bounds", "x", "y", "displacement",
             "layer_bounds", "layer_name", "interface_depth", "interface_name",
             "layer_part", "surface_interface")

#: The rules a displacement states when the mapping states none.
_DISPLACEMENT_RULES = ("polynomial", "cubic")


# -- the domain and the grid, shared with the flat surface file -----------------------

def write_domain(group: Any, domain: Domain) -> None:
    """The statements of the lateral domain: `domain`, `box` or `section`,
    and `domain_bounds` and `domain_padding` as the domain holds them."""
    group.setncattr("domain", DOMAINS[0] if domain.axes == ("x", "y") else DOMAINS[1])
    group.setncattr("domain_bounds", np.ravel(np.asarray(domain.bounds, dtype=float)))
    group.setncattr("domain_padding",
                    np.atleast_1d(np.asarray(domain.padding, dtype=float)))


def read_domain(group: Any, fail: Callable[[str], NoReturn]) -> Domain:
    """The lateral domain a group's attributes state, `fail` called with
    what is wrong where they state none or one that is malformed."""
    attrs = set(group.ncattrs())
    for key in ("domain", "domain_bounds", "domain_padding"):
        if key not in attrs:
            fail(f"{key} is missing: a flat file states its lateral domain")
    kind = group.getncattr("domain")
    if isinstance(kind, np.ndarray) or kind not in DOMAINS:
        fail(f"domain is {kind!r}, not one of {DOMAINS}")
    try:
        bounds = np.atleast_1d(np.asarray(group.getncattr("domain_bounds"),
                                          dtype=float))
        padding = np.atleast_1d(np.asarray(group.getncattr("domain_padding"),
                                           dtype=float))
    except (TypeError, ValueError):
        fail("domain_bounds and domain_padding are numbers")
    sizes = (4, 2) if kind == DOMAINS[0] else (2, 1)
    if (bounds.size, padding.size) != sizes:
        fail(f"a {kind} states {sizes[0]} bounds and {sizes[1]} padding, got "
             f"{bounds.size} and {padding.size}")
    try:
        if kind == DOMAINS[0]:
            return Box([bounds[:2], bounds[2:]], padding=tuple(padding))
        return Rectangle(bounds, padding=float(padding[0]))
    except ValueError as error:
        fail(f"the domain is malformed: {error}")


def axes_of(domain: Domain) -> tuple[str, ...]:
    """The lateral dimensions of a grid over the domain: (x, y) over a box
    and (x,) over a section."""
    return tuple(domain.axes)


def grid_variables(group: Any, grid: Grid, length: str) -> None:
    """The CF coordinate variables of a grid, `x(x)` and over a box `y(y)`,
    its nodes along each axis in the unit `length`; their dimensions are
    the caller's."""
    for axis, nodes in zip(axes_of(grid.domain), grid.axes):
        var = group.createVariable(axis, "f8", (axis,))
        var[...] = nodes
        var.units, var.long_name = length, axis
        var.axis = axis.upper()


def check_grid_variables(group: Any, domain: Domain,
                         fail: Callable[[str], NoReturn]) -> Grid:
    """The grid a group's dimensions hold over `domain`, its coordinate
    variables checked to be the grid's nodes (to 1e-12 of the period),
    `fail` called with what is wrong otherwise; a group over a section
    with a `y` is refused."""
    axes = axes_of(domain)
    if "y" not in axes and ("y" in group.dimensions or "y" in group.variables):
        fail("a section's grid has the one axis x, and the group states y")
    for axis in axes:
        if axis not in group.dimensions or axis not in group.variables:
            fail(f"no coordinate {axis!r}")
        if tuple(group.variables[axis].dimensions) != (axis,):
            fail(f"{axis} does not run over its own dimension")
    try:
        grid = Grid(domain, tuple(len(group.dimensions[a]) for a in axes))
    except ValueError as error:
        fail(f"the grid is malformed: {error}")
    for axis, nodes, period in zip(axes, grid.axes, grid.lengths):
        stored = variable_values(group.variables[axis])
        if not np.allclose(stored, nodes, rtol=0.0, atol=1e-12 * period):
            fail(f"{axis} is not the nodes of a grid of {nodes.size} over the "
                 f"domain's extent")
    return grid


def refuse_spherical(group: Any, fail: Callable[[str], NoReturn]) -> None:
    """Refuse, by name, a statement of a spherical file in a flat file's
    group (`SPHERICAL_NAMES`, and any polar values)."""
    for kind, names in SPHERICAL_NAMES.items():
        held = {"dimensions": group.dimensions, "attributes": group.ncattrs(),
                "variables": group.variables}[kind]
        for name in names:
            if name in held:
                fail(f"{name} is a spherical file's, and a flat file states no "
                     f"{name}")
    poles = [n for n in group.variables if n.endswith("_poles")]
    if poles:
        fail(f"{poles[0]} holds polar values, which a flat grid has none of")


def on_grid(fn: Callable[[np.ndarray, np.ndarray], ArrayLike], grid: Grid
            ) -> np.ndarray:
    """A function of (x, y) on the grid's nodes: the values it holds where
    it is held on that grid, through its own `on_grid` where it has one,
    and called at the nodes otherwise."""
    held = getattr(fn, "values", None)
    if getattr(fn, "grid", None) == grid and held is not None:
        return np.asarray(held)
    own = getattr(fn, "on_grid", None)
    if callable(own):
        return np.asarray(own(grid))
    x, y = grid.nodes
    return np.asarray(fn(x, y))


# -- the layout ------------------------------------------------------------------

class FlatLayout:
    """The flat model file's `modelfile.Layout`: the module docstring's
    statements, asked by the shared writer and readers.  Its context is
    the file's lateral domain."""

    system = FLAT.name
    schemas = (SCHEMA_FLAT,)
    coordinate = "depth"
    bounds_meaning = "depths of the layer's top and bottom below the free surface"
    interface_meaning = "depth of the interface below the free surface"
    element_meaning = "depths of the element's top and bottom"
    reserved = _RESERVED
    methods = METHODS
    spectral = "fourier"
    arguments = "(x, y)"
    part_attributes = ("ngll",)
    displacement_shape = ()

    # -- writing

    def schema(self, logged: bool) -> str:
        """`SCHEMA_FLAT`, whether a variable states a rule in the logarithm
        or none does."""
        return SCHEMA_FLAT

    def check_schema(self, schema: str, logged: Sequence[str], path: object) -> None:
        """A flat file states rules in the logarithm or none under its one
        schema."""

    def check_grid(self, model: Any, grid: object, layer: int) -> None:
        """A `cartesian.Grid` over the model's domain for every layer."""
        if not isinstance(grid, Grid):
            raise TypeError(f"the grid of layer {layer} is not a cartesian.Grid")
        domain = model.geometry.domain
        if grid.domain != domain:
            raise ValueError(f"the grid of layer {layer} lies over {grid.domain!r}, "
                             f"the model over {domain!r}: every grid of a flat file "
                             "covers the model's domain")

    def mapping_form(self, model: Any) -> str:
        """A vertical stretch of boundary reliefs is the stack, any other
        vertical stretch a displacement field; a mapping that is no
        vertical stretch is refused."""
        geometry = model.geometry
        if geometry.is_identity:
            return "none"
        w = getattr(geometry.mapping, "w", None)
        if w is None:
            raise ValueError(
                f"the flat model file holds a vertical stretch "
                f"(cartesian.VerticalStretch), as boundary reliefs or as its upward "
                f"displacement; {geometry.mapping!r} is not one, and no other flat "
                "mapping has a form in the file")
        if (getattr(w, "reliefs", None) is not None
                and getattr(w, "boundaries", None) is not None):
            return "stack"
        return "field"

    def outer(self, parts: Sequence[tuple[list[int], Any]]) -> Grid:
        """The first part's grid, the part at the free surface."""
        return parts[0][1]

    def check_rule(self, grid: Grid, rule: str) -> None:
        """`cartesian.lateral.check_rule`."""
        check_rule(grid, rule)

    def spectral_holds(self, grid: Grid, fields: Sequence[object]) -> bool:
        """Whether the grid holds every mode of the grid of every field that
        varies laterally (`Grid.holds`)."""
        for f in fields:
            if getattr(f, "is_radial", False):
                continue
            own = getattr(f, "grid", None)
            if not isinstance(own, Grid) or not grid.holds(own):
                return False
        return True

    def inexact_step(self, model: Any, grid: Grid) -> float:
        """The grid's finest node spacing, the step `cartesian.sample` takes."""
        return float(min(grid.spacing))

    def write_root(self, root: Any, model: Any) -> None:
        """`coordinate_system`, `local_frame` and the domain."""
        root.setncattr("coordinate_system", self.system)
        root.setncattr("local_frame", " ".join(LOCAL_FRAME_AXES))
        write_domain(root, model.geometry.domain)

    def grid_dimensions(self, group: Any, grid: Grid) -> None:
        """`x` and, over a box, `y`."""
        for axis, n in zip(axes_of(grid.domain), grid.shape):
            group.createDimension(axis, n)

    def grid_attributes(self, group: Any, grid: Grid) -> None:
        """None: the domain and the coordinates state the grid."""

    def grid_variables(self, group: Any, grid: Grid, length: str) -> None:
        """`grid_variables`."""
        grid_variables(group, grid, length)

    def node_attributes(self, length: str) -> dict[str, Any]:
        """`depth(node)`, CF's depth, positive down."""
        return {"long_name": "depth of the node below the free surface; an element "
                             "boundary is two nodes", "units": length,
                "standard_name": "depth", "positive": "down"}

    def lacks_pole(self, grid: Grid) -> bool:
        """Never: a flat grid has no poles."""
        return False

    def lateral_dims(self, grid: Grid) -> tuple[str, ...]:
        """(x, y) over a box, (x,) over a section."""
        return axes_of(grid.domain)

    def lateral_shape(self, grid: Grid) -> tuple[int, ...]:
        """The grid's shape."""
        return grid.shape

    def evaluate(self, field: Field, r: np.ndarray, grid: Grid, *,
                 radial: bool) -> np.ndarray:
        """A field of depth alone at the depths, its components the same at
        every point; any other on the grid, through its own `on_grid` where
        it has one (a Fourier field's transform)."""
        if radial:
            if field.character.rank == 0:
                return real_values(field.evaluate(r, None, None), field)
            return real_values(field.evaluate(r, 0.0, 0.0), field)
        own = getattr(field, "on_grid", None)
        if own is not None:
            return real_values(own(r, grid), field)
        x, y = grid.nodes
        depth = r.reshape((-1,) + (1,) * grid.dimension)
        return real_values(field.evaluate(depth, x[None], y[None]), field)

    def poles_of(self, field: Field, r: np.ndarray) -> np.ndarray:
        """Refused: a flat grid has no poles."""
        raise ValueError("a flat grid has no poles")

    def on_grid(self, fn: Callable[[np.ndarray, np.ndarray], ArrayLike],
                grid: Grid) -> ArrayLike:
        """`on_grid`."""
        return on_grid(fn, grid)

    def at_poles(self, fn: Callable[[np.ndarray, np.ndarray], ArrayLike]) -> None:
        """None: a flat grid has no poles."""
        return None

    def write_displacement(self, group: Any, model: Any, r: np.ndarray, grid: Grid,
                           rules: tuple[str, str], *, lacks_pole: bool,
                           length: str) -> None:
        """The stretch's upward displacement w at the part's nodes."""
        w = model.geometry.mapping.w
        x, y = grid.nodes
        depth = r.reshape((-1,) + (1,) * grid.dimension)
        values = np.broadcast_to(np.asarray(w(depth, x[None], y[None]), dtype=float),
                                 (r.size,) + grid.shape)
        attrs = {"long_name": "upward displacement of the vertical stretch, a height "
                              "above the node's reference place", "units": length,
                 "dimensions": np.asarray([getattr(LENGTH, b) for b in BASE],
                                          dtype=np.int32),
                 "radial_interpolation": rules[0],
                 "lateral_interpolation": rules[1], "coordinates": "depth",
                 "positive": "up"}
        put_variable(group, "displacement", ("node",) + axes_of(grid.domain), values,
                     attrs)

    def write_stack(self, root: Any, model: Any,
                    parts: Sequence[tuple[list[int], Any]],
                    interpolation: Interpolation) -> None:
        """The group `geometry`: the stretch's boundary reliefs, heights."""
        w = model.geometry.mapping.w
        reliefs, depths = list(w.reliefs), np.asarray(w.boundaries, dtype=float)
        grid, lateral = relief_grid(reliefs, self.outer(parts), interpolation)
        if not isinstance(grid, Grid) or grid.domain != model.geometry.domain:
            raise ValueError(f"the reliefs are held on {grid!r}; the geometry group "
                             "holds them on a cartesian.Grid over the model's domain")
        check_rule(grid, lateral)
        g = root.createGroup("geometry")
        g.createDimension("boundary", depths.size)
        self.grid_dimensions(g, grid)
        length = scaled_unit(LENGTH, model.scales)
        put_variable(g, "boundary_depth", ("boundary",), depths,
                     {"long_name": "reference depth of the surface the displacement "
                                   "belongs to", "units": length})
        grid_variables(g, grid, length)
        values = np.zeros((depths.size,) + grid.shape)
        for k, rel in enumerate(reliefs):
            if rel is not None:
                values[k] = np.broadcast_to(np.asarray(on_grid(rel, grid), dtype=float),
                                            grid.shape)
        put_variable(g, "boundary_displacement", ("boundary",) + axes_of(grid.domain),
                     values,
                     {"long_name": "upward displacement of the surface, its height "
                                   "above its reference place", "units": length,
                      "positive": "up", "lateral_interpolation": lateral,
                      "between": "linear"})

    # -- reading

    def validate_root(self, ds: Any, path: object) -> Domain:
        """`coordinate_system` and `local_frame` as a flat file states them,
        no spherical statement, and the domain, which is the context."""
        def fail(msg: str) -> NoReturn:
            malformed(path, msg)

        for key, want in (("coordinate_system", self.system),
                          ("local_frame", " ".join(LOCAL_FRAME_AXES))):
            got = ds.getncattr(key) if key in ds.ncattrs() else None
            if got != want:
                fail(f"{key} is {got!r}, and a flat model file states {want!r}")
        refuse_spherical(ds, fail)
        return read_domain(ds, fail)

    def axes(self, context: Domain) -> tuple[str, ...]:
        """The domain's axes."""
        return axes_of(context)

    def grid_names(self, context: Domain) -> tuple[str, ...]:
        """The coordinates of the domain's axes."""
        return axes_of(context)

    def validate_grid(self, group: Any, where: str, path: object,
                      context: Domain) -> None:
        """No spherical statement, and coordinates that are the nodes of a
        grid over the domain."""
        def fail(msg: str) -> NoReturn:
            malformed(path, f"{where}: {msg}")

        refuse_spherical(group, fail)
        check_grid_variables(group, context, fail)

    def read_grid(self, group: Any, context: Domain) -> Grid:
        """The grid over the domain of the group's shape."""
        return Grid(context, tuple(len(group.dimensions[a]) for a in axes_of(context)))

    def validate_stack(self, group: Any, path: object, context: Domain) -> None:
        """The geometry group: the reference depths, the heights on a grid
        over the domain, positive up, a flat rule and linear between."""
        def fail(msg: str) -> NoReturn:
            malformed(path, f"geometry: {msg}")

        refuse_spherical(group, fail)
        axes = axes_of(context)
        for name in ("boundary_depth", "boundary_displacement"):
            if name not in group.variables:
                fail(f"no variable {name!r}")
        check_grid_variables(group, context, fail)
        var = group.variables["boundary_displacement"]
        if tuple(var.dimensions) != ("boundary",) + axes:
            fail(f"boundary_displacement does not run over {('boundary',) + axes}")
        a = set(var.ncattrs())
        if "lateral_interpolation" not in a or var.getncattr(
                "lateral_interpolation") not in METHODS:
            fail("boundary_displacement states no known lateral rule")
        if "between" not in a or var.getncattr("between") != "linear":
            fail("boundary_displacement states an unknown rule between surfaces")
        if "positive" not in a or var.getncattr("positive") != "up":
            fail("boundary_displacement is a height, positive up, and states "
                 "otherwise")

    def read_stack(self, group: Any, context: Domain) -> FileGeometry:
        """The surfaces' depths, grid and heights; no polar values."""
        var = group.variables["boundary_displacement"]
        return FileGeometry(
            np.asarray(group.variables["boundary_depth"][:], dtype=float),
            self.read_grid(group, context), np.asarray(var[:], dtype=float), None,
            str(var.getncattr("lateral_interpolation")), str(var.getncattr("between")))

    def radial_field(self, interval: tuple[float, float], function: Any, *,
                     character: Character, name: str | None) -> RadialField:
        """A `DepthField`."""
        from .fields import DepthField
        return DepthField(interval, function, character=character, name=name)

    def gridded_field(self, interval: tuple[float, float], breaks: np.ndarray,
                      grid: Grid, values: np.ndarray, poles: np.ndarray | None, *,
                      character: Character, name: str | None, radial: str,
                      lateral: str) -> Field:
        """A `cartesian.GriddedField`; a flat file holds no polar values."""
        from .gridded import GriddedField
        return GriddedField(interval, breaks, grid, values, character=character,
                            name=name, radial=radial, lateral=lateral)

    def gridded_function(self, grid: Grid, values: np.ndarray, *, lateral: str,
                         poles: np.ndarray | None) -> Any:
        """A `cartesian.GriddedFunction`."""
        from .gridded import GriddedFunction
        return GriddedFunction(grid, values, lateral=lateral)

    def geometry(self, data: ModelFile, interpolation: Interpolation) -> Any:
        """A `cartesian.Geometry` over the file's domain, its mapping the
        vertical stretch of the stack's reliefs or of a
        `cartesian.GriddedDisplacement`."""
        from ..character import SCALAR
        from .displacement import LayerLinear
        from .geometry import Geometry
        from .gridded import GriddedDisplacement, GriddedFunction
        from .mapping import VerticalStretch
        sk = data.skeleton
        mapping = None
        if data.geometry is not None:
            geo = data.geometry
            lateral = interpolation.get("boundary_displacement", (None, geo.lateral))[1]
            reliefs = [GriddedFunction(geo.grid, v, lateral=lateral) if np.any(v)
                       else None for v in geo.values]
            mapping = VerticalStretch(LayerLinear(Skeleton(geo.radii), reliefs),
                                      base=sk)
        elif any("displacement" in part.variables for part in data.parts):
            fields = displacement_fields(data, interpolation, self, character=SCALAR)
            mapping = VerticalStretch(GriddedDisplacement(sk, fields), base=sk)
        return Geometry(sk, data.domain, mapping=mapping,
                        layer_names=list(data.layer_names),
                        interface_names=list(data.interface_names), rtol=data.rtol)

    def model(self, geometry: Any, layers: list[dict[str, Field]], *,
              data: ModelFile, specs: Mapping[str, FieldSpec]) -> Any:
        """A plain `cartesian.Model`."""
        from .model import Model
        return Model(geometry, layers, scales=data.scales, specs=specs,
                     name=data.name, constants=data.declared, citation=data.citation)


#: The flat layout, the one of every file of `SCHEMA_FLAT`.
LAYOUT = FlatLayout()


def read_model_file(path: str | os.PathLike[str]) -> ModelFile:
    """A flat model file's numbers and metadata, its structure validated
    first, its `domain` the file's; a spherical file is refused by its
    schema, naming its reader (`modelfile.read_model_file`)."""
    return read_file(path, (LAYOUT,))
