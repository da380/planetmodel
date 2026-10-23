"""The model file: a model in one netCDF file, on its reference skeleton.

One netCDF-4 file holds a model as numbers: its skeleton, its fields at
the nodes of a radial layout and an angular grid, and its mapping, in
the model's own units with its scales recorded, CF-styled so that
xarray, Panoply and ncview open it as it stands.  The Python here is the
definition of the format and the reference the C++ readers are held to;
`docs/source/formats/model_file.rst` describes it for consumers.

**Global attributes**: `schema` (`SCHEMA`, checked by every reader),
`Conventions`, `source` (the writing code and version) and `created`
(when, ISO 8601) — both stamped by the writers, and empty on reading a
file from before they were — the model's `name`, `scale_length`, `scale_mass`,
`scale_time` and `scale_temperature` (what one stored unit is in SI),
`rtol` (the geometry's tolerance, the default where a file has none),
`constant_<name>` for every constant in the model's units, `citation`
(the works the model cites, one a line, where it cites any), and
`meta_<key>` for the writer's `meta`.  **The skeleton**, at the root:
dimensions `layer`, `interface` and `bound` (two), `layer_bounds(layer,
bound)`, `layer_name(layer)`, `interface_radius(interface)`,
`interface_name(interface)` and `layer_part(layer)`.

**Parts.**  The layers are cut into parts, contiguous runs of whole
layers, one group each (`part_0`, `part_1`, ... from the centre), each
on an angular grid of its own.  A part holds the `Sample` layout of its
layers: dimension `node`, the per-element GLL nodes flattened element by
element, so that an interface is two nodes at one radius; `radius(node)`
(an auxiliary coordinate, since it repeats), `element_layer(element)`,
`element_bounds(element, bound)` and the attribute `ngll`; the grid as
the CF coordinate variables `latitude` (degrees north, decreasing) and
`longitude` (degrees east, in [0, 360)), with the attributes `grid_kind`
and `grid_lmax` and the variable `colatitude_weights(latitude)` where it
has them.  Then one variable per field, `name(node)` when every layer
holding it is radial and `name(node, latitude, longitude)` otherwise,
the stored components of rank 1 and above on trailing axes (`component`,
`voigt`, or row and column pairs of them) in the spherical frame at the
node, NaN on the nodes of the layers that lack it; attributes
`dimensions` (the exponents of `units.BASE`), `units` (for CF tools,
never read back), `long_name`, `rank`, `weight`, `voigt`, `layers`,
`radial_interpolation` and, laterally, `lateral_interpolation`.  A
field that is a radial reference with a perturbation on every layer of
the part that holds it perturbed is two variables: `name(node)`, the
reference, with attributes `perturbation` and `perturbation_kind`, and
`name_perturbation`.  Where the grid lacks either pole, every variable
that varies with direction has `name_poles(node, pole, ...)`, north then
south, its components Cartesian, evaluated exactly by the writer.

**The mapping** is held in one of two forms.  A mapping that offers the
reliefs of its surfaces (a radial stretch by `layer_linear`) is the
group `geometry`: `boundary_radius(boundary)`, the radii of its
surfaces, and `boundary_displacement(boundary, latitude, longitude)` on
a grid of its own, with `boundary_displacement_poles` where that grid
lacks a pole and the attributes `lateral_interpolation` and `between`
(the rule between surfaces, `linear`).  Any other mapping is the field
`displacement(node, latitude, longitude, component)` of every part, m(X)
- X in the spherical frame at X, as a sample holds it, read back as a
`GriddedDisplacement`.  The identity is neither.

**Fields on interfaces**, functions of the angles on the skeleton's own
interfaces (a shape kernel, the shift of a boundary), are the optional
group `surfaces`: dimension `surface` over the interfaces that hold any,
`surface_interface(surface)` their indices in the root's `interface`,
increasing; a grid of its own, as a part's is; and one variable per
name, `name(surface, latitude, longitude)`, NaN on an interface that
lacks it, with the attributes `long_name`, `units`, `dimensions` (where
declared), `lateral_interpolation` and `interfaces` (those holding it),
and `name_poles(surface, pole)` where the grid lacks a pole.  A name is
a layer field's or a surface field's, never both, and a surface field
is a number at each point, its spec's dimensions its own.  A file
without the group reads as one with none.

**The radial layout** is exact where it can be: the writer builds, per
part, a radial mesh whose element edges are the layer boundaries, every
breakpoint a field declares (`radial_pieces`) and every knot of the
mapping, with `ngll` one more than the highest declared degree; a field
that declares no polynomial description subdivides the elements of its
layer to the spacing a sample takes from the grid's band (every layer,
for a mapping written as a displacement field), and raises `ngll` to at
least 5.  So a deck model
written and read back is the same polynomials, and a band-limited field
on a grid of its band is held exactly.  `radial=` gives the layout
instead.  **Every variable states its intended interpolation**: a
gridded field its own rules, a harmonic field `polynomial` and
`harmonic`, any other field the defaults (`polynomial`, and `linear`
laterally) unless `interpolation=` names others; a reader uses the
stated rules unless its caller overrides them.

`read_model_file` gives the numbers and the metadata as a `ModelFile`,
the way `read_deck` gives a deck; `file_layers`, `file_geometry` and
`file_surfaces` turn them into fields, a geometry and the functions on
interfaces; `read_model` is the plain model, and `read_surfaces` the
functions beside it.
netCDF4 (the `netcdf` extra) is imported where a file is opened.  Real
fields only; a model is never converted on writing, and is read back in
the file's scales unless `scales=` asks for others.
"""
from __future__ import annotations

import numbers
import os
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, NoReturn

import numpy as np
from numpy.typing import ArrayLike

from .character import SCALAR, VECTOR, Character
from .fields import Field, RadialField, stored_shape
from .frames import from_geographic, geographic, spherical_frame
from .layerfunction import PolynomialLayer
from .lateral import METHODS as LATERAL_METHODS
from .lateral import check_rule
from .mesh1d.mesh import RADIAL_METHODS, Mesh1D, RadialMesh
from .perturbation import KINDS as PERTURBATION_KINDS
from .sampling import KINDS as GRID_KINDS
from .skeleton import DEFAULT_RTOL, Skeleton
from .units import BASE, Dimensions, Scales, scaled_unit
from .vocabulary import Constant, FieldSpec, shipped

if TYPE_CHECKING:
    from .displacement import Relief
    from .geometry import Geometry
    from .gridded import GriddedFunction
    from .model import Model
    from .sampling import AngularGrid

__all__ = ["SCHEMA", "CONVENTIONS", "DEFAULT_INTERPOLATION", "Grids",
           "Interpolation", "Surfaces", "FileVariable",
           "FilePart", "FileGeometry", "FileSurfaces", "ModelFile", "write_model",
           "read_model_file", "file_layers", "file_geometry", "file_surfaces",
           "read_model", "read_surfaces", "validate_structure",
           "netcdf_module", "put_variable", "variable_attributes",
           "variable_values", "grid_degrees", "grid_of_group"]

#: The schema string every reader checks; bumped only for an incompatible change.
SCHEMA = "planetmodel.model/1"

#: The CF conventions the file follows; groups are admitted from 1.8.
CONVENTIONS = "CF-1.8"

#: The rules a variable states when its field states none: (radial, lateral).
DEFAULT_INTERPOLATION = ("polynomial", "linear")

#: The rules a displacement states when the mapping states none: the
#: bicubic rule, so that its deformation gradient is continuous.
_DISPLACEMENT_INTERPOLATION = ("polynomial", "cubic")

#: Names the format gives its own variables, which no field may take.
_RESERVED = ("radius", "element_layer", "element_bounds", "latitude",
             "longitude", "colatitude_weights", "displacement",
             "displacement_poles", "layer_bounds", "layer_name",
             "interface_radius", "interface_name", "layer_part",
             "surface_interface")

#: The names of the trailing axes of a stored shape.
_AXES = {(): (), (3,): ("component",), (6,): ("voigt",),
         (3, 3): ("component_row", "component_column"),
         (6, 6): ("voigt_row", "voigt_column")}

#: The netCDF-4 compression applied to every variable.
_ZLIB = {"zlib": True, "complevel": 4}

#: The rules asked for by name: name -> (radial, lateral).
type Interpolation = Mapping[str, tuple[str, str]]

#: The angular grid of a whole model, or one per layer by index or name.
type Grids = AngularGrid | Mapping[int | str, AngularGrid]

#: Functions of (theta, phi) on interfaces: name -> {interface, by index
#: or name -> function}.
type Surfaces = Mapping[str, Mapping[int | str, Relief]]


def netcdf_module() -> ModuleType:
    """netCDF4, imported here and nowhere else in planetmodel."""
    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "the model file needs netCDF4.  Install it with:\n"
            "    pip install 'planetmodel[netcdf]'      "
            "(or: poetry install --extras netcdf)") from exc
    return netCDF4


# -- what a file holds ----------------------------------------------------------

@dataclass(frozen=True)
class FileVariable:
    """One variable of a part: its values, as stored, and its attributes."""

    values: np.ndarray
    attributes: Mapping[str, Any]


@dataclass(frozen=True)
class FilePart:
    """One part: its run of layers, their radial layout as a mesh over the
    skeleton, the angular grid, and the variables, by name."""

    layers: tuple[int, ...]
    mesh: RadialMesh
    grid: AngularGrid
    variables: Mapping[str, FileVariable]


@dataclass(frozen=True)
class FileGeometry:
    """The mapping as boundary displacements: the surfaces' radii, their
    grid, the displacements (boundary, ntheta, nphi) and their polar
    values (boundary, 2) where the grid lacks a pole, and the rules."""

    radii: np.ndarray
    grid: AngularGrid
    values: np.ndarray
    poles: np.ndarray | None
    lateral: str
    between: str


@dataclass(frozen=True)
class FileSurfaces:
    """The fields on interfaces: the interfaces holding any (indices,
    increasing), their grid, and the variables by name, each's values
    (surface, ntheta, nphi) and, as `name_poles`, (surface, 2)."""

    interfaces: tuple[int, ...]
    grid: AngularGrid
    variables: Mapping[str, FileVariable]


@dataclass(frozen=True)
class ModelFile:
    """A model file's numbers and metadata, as `read_model_file` gives
    them: nothing is interpolated or converted."""

    schema: str
    name: str | None
    scales: Scales
    constants: Mapping[str, float]
    meta: Mapping[str, Any]
    skeleton: Skeleton
    layer_names: tuple[str | None, ...]
    interface_names: tuple[str | None, ...]
    parts: tuple[FilePart, ...]
    geometry: FileGeometry | None
    _: KW_ONLY
    #: The constants whose dimensions the file states, as a model holds them.
    declared: Mapping[str, Constant] = MappingProxyType({})
    #: The tolerance of the geometry the file was written from, relative
    #: to the skeleton's span; the default where the file states none.
    rtol: float = DEFAULT_RTOL
    #: The group `surfaces`, where the file has it.
    surfaces: FileSurfaces | None = None
    #: The works the model cites, one a line of the attribute `citation`.
    citation: tuple[str, ...] = ()
    #: The code that wrote the file (the attribute `source`); empty in
    #: files written before the writers stamped it.
    source: str = ""
    #: When the file was written, ISO 8601 (the attribute `created`);
    #: empty in files written before the writers stamped it.
    created: str = ""

    def part_of(self, layer: int) -> FilePart:
        """The part holding `layer`."""
        for part in self.parts:
            if layer in part.layers:
                return part
        raise IndexError(f"no part holds layer {layer}")


# -- writing --------------------------------------------------------------------

def _parts_of(model: Model, grid: Grids) -> list[tuple[list[int], AngularGrid]]:
    """The runs of contiguous layers on one grid, or the error."""
    from .sampling import AngularGrid
    n = model.nlayers
    if isinstance(grid, AngularGrid):
        per = [grid] * n
    else:
        per = [None] * n
        for key, g in grid.items():
            per[model.geometry.layer(key).index] = g
        missing = [model.layers[i].name or str(i) for i in range(n) if per[i] is None]
        if missing:
            raise ValueError(f"no grid given for layers {missing}")
    parts: list[tuple[list[int], AngularGrid]] = []
    for i, g in enumerate(per):
        if not isinstance(g, AngularGrid):
            raise TypeError(f"the grid of layer {i} is not an AngularGrid")
        if parts and parts[-1][1] == g:
            parts[-1][0].append(i)
        else:
            if any(p[1] == g for p in parts):
                raise ValueError(
                    f"layer {i} is on the grid of layers that are not next to "
                    "it; a part is a contiguous run of layers, so give each run "
                    "its own grid")
            parts.append(([i], g))
    return parts


def _mapping_form(model: Model) -> str:
    """'none', 'stack' (boundary reliefs) or 'field' (a displacement)."""
    geometry = model.geometry
    if geometry.is_identity:
        return "none"
    h = getattr(geometry.mapping, "h", None)
    if (getattr(h, "reliefs", None) is not None
            and getattr(h, "boundaries", None) is not None):
        return "stack"
    return "field"


def _part_mesh(model: Model, layers: list[int], grid: AngularGrid,
               names: Sequence[str], form: str) -> RadialMesh:
    """The exact radial layout of a part, by the rule of the module docstring."""
    b = model.skeleton.boundaries
    lo, hi = float(b[layers[0]]), float(b[layers[-1] + 1])
    edges, degree = [b[layers[0]:layers[-1] + 2]], 1
    inexact = set(layers) if form == "field" else set()
    for i in layers:
        layer = model.layer(i)
        for name in names:
            if name not in layer:
                continue
            pieces = getattr(layer[name], "radial_pieces", lambda: None)()
            if pieces is None:
                inexact.add(i)
                continue
            edges.append(pieces[0])
            degree = max(degree, pieces[1])
    knots = np.asarray(model.geometry.knots(), dtype=float)
    edges.append(knots[(knots > lo) & (knots < hi)])
    tol = model.geometry.rtol * model.skeleton.span
    pins = np.unique(np.concatenate(edges))
    pins = pins[np.concatenate([[True], np.diff(pins) > tol])]
    pins[0], pins[-1] = lo, hi
    ngll = degree + 1
    if inexact:
        if grid.lmax is None:
            raise ValueError(
                "a field or the mapping declares no polynomial description in "
                f"radius, and the {grid.kind!r} grid carries no band to size "
                "the elements by: pass radial=, a RadialMesh over the skeleton")
        ngll = max(ngll, 5)
        drmax = 0.1 * float(b[-1]) / (grid.lmax + 1)
        runs = []
        for i in layers:
            inside = pins[(pins > b[i] + tol) & (pins < b[i + 1] - tol)]
            run = np.concatenate([[b[i]], inside, [b[i + 1]]])
            if i in inexact:
                cut = Mesh1D(run, ngll=2, drmax=drmax)
                run = np.append(cut.left, cut.right[-1])
            runs.append(run[:-1])
        pins = np.append(np.concatenate(runs), hi)
    return RadialMesh(model.skeleton, ngll=ngll, edges=pins)


def _restricted(mesh: RadialMesh, layers: list[int]) -> RadialMesh:
    """The given mesh's elements in a run of layers, as a mesh of its own."""
    elems = np.flatnonzero(np.isin(mesh.layer, layers))
    if elems.size == 0 or not np.array_equal(np.unique(mesh.layer[elems]),
                                             np.asarray(layers)):
        raise ValueError(f"radial= does not cover layers {layers}")
    return RadialMesh(mesh.skeleton, ngll=mesh.ngll,
                      edges=np.append(mesh.left[elems], mesh.right[elems[-1]]))


def _real(values: ArrayLike, field: Field) -> np.ndarray:
    """Values as float64, a complex field refused: the file holds reals."""
    v = np.asarray(values)
    if np.iscomplexobj(v):
        raise TypeError(f"{field!r} is complex; a model file holds real fields, so "
                        "write its real and imaginary parts under names of their own")
    return v.astype(float, copy=False)


def _evaluate(field: Field, r: np.ndarray, grid: AngularGrid, *,
              radial: bool) -> np.ndarray:
    """A field at the nodes: (n,) + c when radial, (n, nt, np) + c else."""
    if radial:
        if field.character.rank == 0:
            return _real(field.evaluate(r, None, None), field)
        return _real(field.evaluate(r, 0.5 * np.pi, 0.0), field)
    on_grid = getattr(field, "on_grid", None)
    if on_grid is not None:
        return _real(on_grid(r, grid), field)
    return _real(field.evaluate(r[:, None, None], grid.colatitudes[None, :, None],
                                grid.longitudes[None, None, :]), field)


def _poles_of(field: Field, r: np.ndarray) -> np.ndarray:
    """A field at both poles at the radii, Cartesian: (n, 2) + c."""
    return _real(field.evaluate(r[:, None], np.array([0.0, np.pi]), np.zeros(2),
                                frame="cartesian"), field)


def _stated(fields: Sequence[object], name: str, interpolation: Interpolation,
            default: tuple[str, str], grid: AngularGrid) -> tuple[str, str]:
    """The rules a variable states: the caller's, else the one its fields
    agree on (a field of the radius alone has no lateral rule to state),
    else the default.  Fields that state different rules are refused by
    name.  A field's harmonic rule is stated only on a grid whose
    transforms reproduce it, a shipped kind of band at least its own,
    and the default lateral rule otherwise; the caller's rule must be
    one the grid can carry."""
    if name in interpolation:
        radial, lateral = interpolation[name]
        if radial not in RADIAL_METHODS or lateral not in LATERAL_METHODS:
            raise ValueError(
                f"{name!r}: the rules must be one of {RADIAL_METHODS} and one of "
                f"{LATERAL_METHODS}, got ({radial!r}, {lateral!r})")
        if not all(getattr(f, "is_radial", False) for f in fields):
            check_rule(grid, lateral)
        return radial, lateral
    radials = {getattr(f, "radial", None) for f in fields} - {None}
    laterals = {getattr(f, "lateral", None) for f in fields
                if not getattr(f, "is_radial", False)} - {None}
    if len(radials) > 1 or len(laterals) > 1:
        raise ValueError(
            f"{name!r}: its layers read by different rules, radially "
            f"{sorted(radials)} and laterally {sorted(laterals)}, and one variable "
            f"states one: give interpolation={{{name!r}: (radial, lateral)}}, or "
            "put the layers in parts of their own")
    radial = radials.pop() if radials else default[0]
    lateral = laterals.pop() if laterals else default[1]
    if lateral == "harmonic":
        bands = [getattr(f, "lmax", None) for f in fields
                 if not getattr(f, "is_radial", False)]
        if (grid.kind == "custom" or grid.lmax is None
                or any(b is None or b > grid.lmax for b in bands)):
            lateral = default[1]
    return radial, lateral


def _attributes(name: str, model: Model, character: Character, layers: list[int],
                rules: tuple[str, str], *, lateral: bool) -> dict[str, Any]:
    """A field variable's attributes."""
    spec = model.spec(name)
    dims = None if spec is None else spec.dimensions
    out: dict[str, Any] = {
        "long_name": (spec.meaning if spec is not None and spec.meaning else name),
        "units": scaled_unit(dims, model.scales),
        "rank": np.int32(character.rank), "weight": np.int32(character.weight),
        "voigt": np.int32(bool(character.voigt)),
        "layers": np.asarray(layers, dtype=np.int32),
        "radial_interpolation": rules[0],
    }
    if dims is not None:
        out["dimensions"] = np.asarray([getattr(dims, b) for b in BASE], dtype=np.int32)
    if lateral:
        out["lateral_interpolation"] = rules[1]
    return out


def _mark_radial_layers(attributes: dict[str, Any],
                        fields: Mapping[int, Field]) -> None:
    """Name, in the attribute `radial_layers`, the layers on which a
    lateral variable holds a radial field, its values repeated over the
    grid; nothing where there is none."""
    radial = sorted(i for i, f in fields.items() if getattr(f, "is_radial", False))
    if radial:
        attributes["radial_layers"] = np.asarray(radial, dtype=np.int32)


def _radial_layers(attributes: Mapping[str, Any]) -> set[int]:
    """The layers a lateral variable names as holding a radial field."""
    return {int(i) for i in np.atleast_1d(attributes.get("radial_layers", []))}


def put_variable(group: Any, name: str, dims: tuple[str, ...], values: np.ndarray,
         attributes: Mapping[str, Any]) -> None:
    """A float64 variable with NaN as its fill, and its attributes."""
    var = group.createVariable(name, "f8", dims, fill_value=np.nan, **_ZLIB)
    var[:] = values
    for key, value in attributes.items():
        var.setncattr(key, value)


def _axes(group: Any, character: Character) -> tuple[str, ...]:
    """The trailing axis names of a character's stored shape, created as
    dimensions of the group where they are new."""
    shape = stored_shape(character)
    if shape not in _AXES:
        raise ValueError(f"the model file holds no field of stored shape {shape}")
    for axis, size in zip(_AXES[shape], shape):
        if axis not in group.dimensions:
            group.createDimension(axis, size)
    return _AXES[shape]


def grid_degrees(grid: AngularGrid) -> tuple[np.ndarray, np.ndarray]:
    """The grid's latitudes and longitudes in degrees, each on its own axis."""
    lat, _ = geographic(grid.colatitudes, np.zeros_like(grid.colatitudes))
    _, lon = geographic(np.full_like(grid.longitudes, 0.5 * np.pi), grid.longitudes)
    return lat, lon


def _write_part(group: Any, model: Model, index: int, layers: list[int],
                grid: AngularGrid, mesh: RadialMesh, names: Sequence[str],
                interpolation: Interpolation, form: str) -> None:
    """One part's layout, grid and variables."""
    group.createDimension("node", mesh.nspec * mesh.ngll)
    group.createDimension("element", mesh.nspec)
    group.createDimension("bound", 2)
    group.createDimension("latitude", grid.ntheta)
    group.createDimension("longitude", grid.nphi)
    group.setncattr("ngll", np.int32(mesh.ngll))
    group.setncattr("layers", np.asarray(layers, dtype=np.int32))
    group.setncattr("grid_kind", grid.kind)
    if grid.lmax is not None:
        group.setncattr("grid_lmax", np.int32(grid.lmax))
    length = scaled_unit(Dimensions(length=1), model.scales)
    r = mesh.r.ravel()
    node_layer = np.repeat(mesh.layer, mesh.ngll)
    put_variable(group, "radius", ("node",), r,
         {"long_name": "radius of the node; an element boundary is two nodes",
          "units": length})
    var = group.createVariable("element_layer", "i4", ("element",))
    var[...] = mesh.layer
    var.long_name = "skeleton layer of the element"
    put_variable(group, "element_bounds", ("element", "bound"),
         np.stack([mesh.left, mesh.right], axis=-1),
         {"long_name": "radii of the element's ends", "units": length})
    lat, lon = grid_degrees(grid)
    var = group.createVariable("latitude", "f8", ("latitude",))
    var[...] = lat
    var.units, var.standard_name = "degrees_north", "latitude"
    var.long_name = "latitude"
    var = group.createVariable("longitude", "f8", ("longitude",))
    var[...] = lon
    var.units, var.standard_name = "degrees_east", "longitude"
    var.long_name = "longitude"
    if grid.weights is not None:
        put_variable(group, "colatitude_weights", ("latitude",), grid.weights,
             {"long_name": "colatitude quadrature weights against sin(theta) dtheta",
              "units": "1"})
    lacks_pole = not all(grid.pole_nodes)
    if lacks_pole:
        group.createDimension("pole", 2)
        group.setncattr("poles", "north south")

    for name in names:
        holders = [i for i in layers if name in model.layer(i)]
        if not holders:
            continue
        fields = {i: model.layer(i)[name] for i in holders}
        char = next(iter(fields.values())).character
        axes = _axes(group, char)
        c = stored_shape(char)
        kinds = {getattr(f, "kind", None) for f in fields.values()
                 if getattr(f, "reference", None) is not None}
        split = (len(kinds) == 1 and all(
            getattr(f, "reference", None) is not None or getattr(f, "is_radial", False)
            for f in fields.values()))
        if split:
            _write_perturbed(group, model, name, fields, mesh, node_layer, grid,
                             interpolation, kinds.pop(), lacks_pole)
            continue
        radial = all(getattr(f, "is_radial", False) for f in fields.values())
        shape = (r.size,) + (() if radial else (grid.ntheta, grid.nphi)) + c
        values = np.full(shape, np.nan)
        poles = None if radial or not lacks_pole else np.full((r.size, 2) + c, np.nan)
        rules = _stated(list(fields.values()), name, interpolation,
                        DEFAULT_INTERPOLATION, grid)
        for i, f in fields.items():
            m = node_layer == i
            values[m] = _evaluate(f, r[m], grid, radial=radial)
            if poles is not None:
                poles[m] = _poles_of(f, r[m])
        dims = ("node",) if radial else ("node", "latitude", "longitude")
        attrs = _attributes(name, model, char, holders, rules, lateral=not radial)
        attrs["coordinates"] = "radius"
        if not radial:
            _mark_radial_layers(attrs, fields)
        put_variable(group, name, dims + axes, values, attrs)
        if poles is not None:
            put_variable(group, f"{name}_poles", ("node", "pole") + axes, poles,
                 {"long_name": f"{name} at the north and south poles, Cartesian "
                               "components", "units": attrs["units"]})
    if form == "field":
        axes = _axes(group, VECTOR)
        mapping = model.geometry.mapping
        R = spherical_frame(grid.colatitudes[:, None], grid.longitudes[None, :])
        X = r[:, None, None, None] * R[None, ..., :, 0]
        u = np.asarray(mapping(X), dtype=float) - X
        values = np.einsum("tpji,ntpj->ntpi", R, u)
        rules = _stated([mapping], "displacement", interpolation,
                        _DISPLACEMENT_INTERPOLATION, grid)
        attrs = {"long_name": "displacement of the mapping, m(X) - X, spherical "
                              "components at X", "units": length,
                 "dimensions": np.asarray([getattr(Dimensions(length=1), b)
                                           for b in BASE], dtype=np.int32),
                 "radial_interpolation": rules[0],
                 "lateral_interpolation": rules[1], "coordinates": "radius"}
        put_variable(group, "displacement", ("node", "latitude", "longitude") + axes,
             values, attrs)
        if lacks_pole:
            from .frames import cartesian_points
            P = cartesian_points(r[:, None], np.array([0.0, np.pi]), np.zeros(2))
            put_variable(group, "displacement_poles", ("node", "pole") + axes,
                 np.asarray(mapping(P), dtype=float) - P,
                 {"long_name": "displacement at the north and south poles, "
                               "Cartesian components", "units": length})


def _write_perturbed(group: Any, model: Model, name: str,
                     fields: Mapping[int, Field], mesh: RadialMesh,
                     node_layer: np.ndarray, grid: AngularGrid,
                     interpolation: Interpolation, kind: str,
                     lacks_pole: bool) -> None:
    """A field held as a radial reference with a perturbation: two variables."""
    r = mesh.r.ravel()
    any_field = next(iter(fields.values()))
    char = any_field.character
    pert = [getattr(f, "perturbation", None) for f in fields.values()]
    pert_radial = all(p is None or getattr(p, "is_radial", False) for p in pert)
    ref = np.full((r.size,), np.nan)
    dp_shape = (r.size,) + (() if pert_radial else (grid.ntheta, grid.nphi))
    dp = np.full(dp_shape, np.nan)
    poles = None if pert_radial or not lacks_pole else np.full((r.size, 2), np.nan)
    pert_char = None
    references, perturbations = [], []
    for i, f in fields.items():
        m = node_layer == i
        reference = getattr(f, "reference", None)
        if reference is None:
            ref[m] = _evaluate(f, r[m], grid, radial=True)
            references.append(f)
            continue
        ref[m] = _evaluate(reference, r[m], grid, radial=True)
        dp[m] = _evaluate(f.perturbation, r[m], grid, radial=pert_radial)
        if poles is not None:
            poles[m] = _poles_of(f.perturbation, r[m])
        references.append(reference)
        perturbations.append(f.perturbation)
        pert_char = f.perturbation.character
    ref_rules = _stated(references, name, interpolation, DEFAULT_INTERPOLATION, grid)
    rules = _stated(perturbations, name, interpolation, DEFAULT_INTERPOLATION, grid)
    holders = sorted(fields)
    attrs = _attributes(name, model, char, holders, ref_rules, lateral=False)
    attrs.update({"perturbation": f"{name}_perturbation", "perturbation_kind": kind,
                  "coordinates": "radius"})
    put_variable(group, name, ("node",), ref, attrs)
    perturbed = [i for i, f in fields.items()
                 if getattr(f, "reference", None) is not None]
    pattrs = _attributes(name, model, pert_char, perturbed, rules,
                         lateral=not pert_radial)
    pattrs["long_name"] = f"{kind} perturbation of {name}"
    if kind != "absolute":
        pattrs["units"] = "1"
        pattrs["dimensions"] = np.zeros(len(BASE), dtype=np.int32)
    pattrs["coordinates"] = "radius"
    if not pert_radial:
        _mark_radial_layers(pattrs, {i: f.perturbation for i, f in fields.items()
                                     if getattr(f, "reference", None) is not None})
    dims = ("node",) if pert_radial else ("node", "latitude", "longitude")
    put_variable(group, f"{name}_perturbation", dims, dp, pattrs)
    if poles is not None:
        put_variable(group, f"{name}_perturbation_poles", ("node", "pole"), poles,
             {"long_name": f"{kind} perturbation of {name} at the poles",
              "units": pattrs["units"]})


def _write_stack(root: Any, model: Model, parts: list[tuple[list[int], AngularGrid]],
                 interpolation: Interpolation) -> None:
    """The group `geometry`: the mapping's boundary reliefs, stacked."""
    h = model.geometry.mapping.h
    reliefs, radii = list(h.reliefs), np.asarray(h.boundaries, dtype=float)
    given = [rel for rel in reliefs if rel is not None]
    grids = [getattr(rel, "grid", None) for rel in given]
    if not given:
        grid = parts[-1][1]
    elif all(g is not None for g in grids) and all(grids[0] == g for g in grids):
        grid = grids[0]
    else:
        raise ValueError(
            "the mapping's reliefs are not held on one grid of their own, which "
            "the geometry group takes: sample them with GriddedFunction.sampled "
            "on the grid the file should hold them on")
    laterals = {getattr(rel, "lateral", None) for rel in given}
    lateral = laterals.pop() if len(laterals) == 1 and None not in laterals else "cubic"
    if "boundary_displacement" in interpolation:
        lateral = interpolation["boundary_displacement"][1]
    g = root.createGroup("geometry")
    g.createDimension("boundary", radii.size)
    g.createDimension("latitude", grid.ntheta)
    g.createDimension("longitude", grid.nphi)
    g.setncattr("grid_kind", grid.kind)
    if grid.lmax is not None:
        g.setncattr("grid_lmax", np.int32(grid.lmax))
    length = scaled_unit(Dimensions(length=1), model.scales)
    put_variable(g, "boundary_radius", ("boundary",), radii,
         {"long_name": "radius of the surface the displacement belongs to",
          "units": length})
    lat, lon = grid_degrees(grid)
    var = g.createVariable("latitude", "f8", ("latitude",))
    var[...] = lat
    var.units, var.standard_name = "degrees_north", "latitude"
    var = g.createVariable("longitude", "f8", ("longitude",))
    var[...] = lon
    var.units, var.standard_name = "degrees_east", "longitude"
    if grid.weights is not None:
        put_variable(g, "colatitude_weights", ("latitude",), grid.weights,
                     {"units": "1"})
    values = np.zeros((radii.size, grid.ntheta, grid.nphi))
    poles = np.zeros((radii.size, 2))
    for k, rel in enumerate(reliefs):
        if rel is None:
            continue
        values[k] = np.broadcast_to(
            np.asarray(rel(grid.colatitudes[:, None], grid.longitudes[None, :]),
                       dtype=float), (grid.ntheta, grid.nphi))
        poles[k] = np.asarray(rel(np.array([0.0, np.pi]), np.zeros(2)), dtype=float)
    put_variable(g, "boundary_displacement",
                 ("boundary", "latitude", "longitude"), values,
                 {"long_name": "radial displacement of the surface", "units": length,
                  "lateral_interpolation": lateral, "between": "linear"})
    if not all(grid.pole_nodes):
        g.createDimension("pole", 2)
        put_variable(g, "boundary_displacement_poles", ("boundary", "pole"), poles,
             {"long_name": "radial displacement at the north and south poles",
              "units": length})


def _surface_holders(model: Model, surfaces: Surfaces, written: Sequence[str]
                     ) -> dict[str, dict[int, Relief]]:
    """Each surface field's functions by interface index, increasing: a
    name the format keeps or one written as a layer field, an interface
    given twice, and a name on no interface are refused."""
    if isinstance(surfaces, str) or not isinstance(surfaces, Mapping):
        raise TypeError("surfaces maps names to {interface: function of (theta, phi)}")
    out: dict[str, dict[int, Relief]] = {}
    for name, on in surfaces.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"a surface field's name is a string, got {name!r}")
        if name in _RESERVED or name.endswith(("_perturbation", "_poles")):
            raise ValueError(f"{name!r} is a name the model file keeps for itself")
        if name in written:
            raise ValueError(f"{name!r} is written as a field of the layers; a name "
                             "is a layer field's or a surface field's, never both")
        spec = model.spec(name)
        if spec is not None and spec.character.rank != 0:
            raise ValueError(f"{name!r} is of rank {spec.character.rank} by its spec, "
                             "and a surface field is a number at each point")
        held: dict[int, Relief] = {}
        for which, fn in on.items():
            face = model.geometry.interface(which)
            if face.index in held:
                raise ValueError(f"{name!r} is given twice on interface {face.index}")
            if not callable(fn):
                raise TypeError(f"{name!r} on interface {which!r} is not a function "
                                "of (theta, phi)")
            held[face.index] = fn
        if not held:
            raise ValueError(f"{name!r} is held on no interface")
        out[name] = dict(sorted(held.items()))
    return out


def _surface_grid(held: Mapping[str, Mapping[int, Relief]],
                  parts: list[tuple[list[int], AngularGrid]]) -> AngularGrid:
    """The grid the functions share, else the outermost part's."""
    grids = [getattr(fn, "grid", None) for on in held.values() for fn in on.values()]
    if all(g is not None for g in grids) and all(grids[0] == g for g in grids):
        return grids[0]
    return parts[-1][1]


def _on_angles(values: ArrayLike, shape: tuple[int, ...], name: str,
               interface: int) -> np.ndarray:
    """A surface function's values, real, finite and of `shape`."""
    v = np.asarray(values)
    if np.iscomplexobj(v):
        raise TypeError(f"{name!r} on interface {interface} is complex; a model file "
                        "holds real values")
    v = np.broadcast_to(v.astype(float, copy=False), shape)
    if not np.all(np.isfinite(v)):
        raise ValueError(f"{name!r} on interface {interface} is not finite at every "
                         "node; NaN is the file's mark of an interface that lacks it")
    return v


def _write_surfaces(root: Any, model: Model, held: Mapping[str, Mapping[int, Relief]],
                    grid: AngularGrid, interpolation: Interpolation) -> None:
    """The group `surfaces`: each name's functions on its interfaces."""
    faces = sorted({k for on in held.values() for k in on})
    g = root.createGroup("surfaces")
    g.createDimension("surface", len(faces))
    g.createDimension("latitude", grid.ntheta)
    g.createDimension("longitude", grid.nphi)
    g.setncattr("grid_kind", grid.kind)
    if grid.lmax is not None:
        g.setncattr("grid_lmax", np.int32(grid.lmax))
    var = g.createVariable("surface_interface", "i4", ("surface",))
    var[...] = np.asarray(faces, dtype=np.int32)
    var.long_name = "the interface the surface is, by its index at the root"
    lat, lon = grid_degrees(grid)
    var = g.createVariable("latitude", "f8", ("latitude",))
    var[...] = lat
    var.units, var.standard_name = "degrees_north", "latitude"
    var.long_name = "latitude"
    var = g.createVariable("longitude", "f8", ("longitude",))
    var[...] = lon
    var.units, var.standard_name = "degrees_east", "longitude"
    var.long_name = "longitude"
    if grid.weights is not None:
        put_variable(g, "colatitude_weights", ("latitude",), grid.weights,
             {"long_name": "colatitude quadrature weights against sin(theta) dtheta",
              "units": "1"})
    lacks_pole = not all(grid.pole_nodes)
    if lacks_pole:
        g.createDimension("pole", 2)
        g.setncattr("poles", "north south")
    row = {k: i for i, k in enumerate(faces)}
    theta, phi = grid.colatitudes[:, None], grid.longitudes[None, :]
    for name, on in held.items():
        values = np.full((len(faces), grid.ntheta, grid.nphi), np.nan)
        poles = np.full((len(faces), 2), np.nan)
        for k, fn in on.items():
            values[row[k]] = _on_angles(fn(theta, phi), (grid.ntheta, grid.nphi),
                                        name, k)
            poles[row[k]] = _on_angles(fn(np.array([0.0, np.pi]), np.zeros(2)), (2,),
                                       name, k)
        rules = _stated(list(on.values()), name, interpolation, DEFAULT_INTERPOLATION,
                        grid)
        spec = model.spec(name)
        dims = None if spec is None else spec.dimensions
        attrs: dict[str, Any] = {
            "long_name": spec.meaning if spec is not None and spec.meaning else name,
            "units": scaled_unit(dims, model.scales),
            "lateral_interpolation": rules[1],
            "interfaces": np.asarray(list(on), dtype=np.int32),
        }
        if dims is not None:
            attrs["dimensions"] = np.asarray([getattr(dims, b) for b in BASE],
                                             dtype=np.int32)
        put_variable(g, name, ("surface", "latitude", "longitude"), values, attrs)
        if lacks_pole:
            put_variable(g, f"{name}_poles", ("surface", "pole"), poles,
                 {"long_name": f"{name} at the north and south poles",
                  "units": attrs["units"]})


def write_model(path: str | os.PathLike[str], model: Model, *, grid: Grids,
                radial: RadialMesh | None = None,
                fields: Iterable[str] | None = None,
                interpolation: Interpolation | None = None,
                meta: Mapping[str, Any] | None = None,
                surfaces: Surfaces | None = None,
                surface_grid: AngularGrid | None = None,
                source: str | None = None) -> Path:
    """Write `model` to a model file at `path`, which is returned.

    `grid` is one `AngularGrid` for every layer, or a mapping from layers
    (by index or name) to grids, every layer given one: contiguous layers
    on one grid form one part, and a grid's layers that are not
    contiguous are refused.  `radial` is a `RadialMesh` over the model's
    skeleton to lay the nodes out by, in place of the exact layout of the
    module docstring.  `fields` are the names to write, every name the
    model holds by default; `interpolation` maps names (and
    `displacement`, `boundary_displacement`) to the (radial, lateral)
    rules to state in place of the fields' own; `meta` becomes
    `meta_<key>` attributes.  `surfaces` maps names to functions of
    (theta, phi) on interfaces (by index or name), written to the group
    `surfaces` on `surface_grid`: by default the grid the functions are
    held on where they share one, else the outermost part's.  A surface
    field's dimensions and meaning are the model's spec of its name, and
    `interpolation` names its lateral rule as for a field.  `source`
    names the writing code and version in the attribute of that name,
    this library and its version unless given.  The model is
    written in its own units.
    """
    from .model import Model
    if not isinstance(model, Model):
        raise TypeError(f"expected a Model, got {type(model).__name__}")
    nc = netcdf_module()
    interpolation = dict(interpolation or {})
    names = model.field_names() if fields is None else tuple(fields)
    if isinstance(names, str):
        raise TypeError("fields is a sequence of names, not one string")
    known = model.field_names()
    for name in names:
        if name not in known:
            raise KeyError(f"no layer holds a field {name!r}")
        if name in _RESERVED or name.endswith(("_perturbation", "_poles")):
            raise ValueError(f"{name!r} is a name the model file keeps for itself")
    parts = _parts_of(model, grid)
    form = _mapping_form(model)
    if form == "field" and any(g != parts[0][1] for _, g in parts):
        raise ValueError(
            "a mapping that is not a stack of reliefs is held as a displacement "
            "in every part, read by the part's own rule, so parts on different "
            "grids would disagree between their nodes where they meet: give "
            "every layer one grid, or, for a mapping linear in r within each "
            "layer (a flattening, say), give it as boundary reliefs through "
            "layer_linear with GriddedFunction reliefs, which any grids take")
    meta = dict(meta or {})
    for key, value in meta.items():
        if isinstance(value, bool) or not isinstance(value, (str, numbers.Real)):
            raise TypeError(f"meta[{key!r}] is {value!r}; a netCDF attribute is "
                            "a string or a number")
    if surfaces is None and surface_grid is not None:
        raise ValueError("surface_grid is given with no surfaces to hold on it")
    held = {} if surfaces is None else _surface_holders(model, surfaces, names)
    on = None
    if held:
        on = _surface_grid(held, parts) if surface_grid is None else surface_grid
    path = Path(path)
    fd, partial = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent or None)
    os.close(fd)
    try:
        _write_file(nc, Path(partial), model, parts, form, names, radial,
                    interpolation, meta, held, on, source)
        os.replace(partial, path)
    finally:
        if os.path.exists(partial):
            os.remove(partial)
    return path


def _write_file(nc: ModuleType, path: Path, model: Model,
                parts: list[tuple[list[int], AngularGrid]], form: str,
                names: Sequence[str], radial: RadialMesh | None,
                interpolation: Interpolation, meta: Mapping[str, Any],
                held: Mapping[str, Mapping[int, Relief]],
                surface_grid: AngularGrid | None,
                source: str | None) -> None:
    """The file of `write_model`, at a path of its own until it is whole."""
    from datetime import datetime, timezone

    from . import __version__
    with nc.Dataset(path, "w", format="NETCDF4") as root:
        root.setncattr("schema", SCHEMA)
        root.setncattr("Conventions", CONVENTIONS)
        root.setncattr("title", f"planetmodel model {model.name or ''}".strip())
        root.setncattr("source", source or f"planetmodel {__version__}")
        root.setncattr("created",
                       datetime.now(timezone.utc).isoformat(timespec="seconds"))
        if model.name:
            root.setncattr("name", model.name)
        for base in ("length", "mass", "time", "temperature"):
            root.setncattr(f"scale_{base}", float(getattr(model.scales, base)))
        root.setncattr("rtol", float(model.geometry.rtol))
        for cname, c in model.constants.items():
            root.setncattr(f"constant_{cname}", float(model.constant(cname)))
            root.setncattr(f"constant_{cname}_dimensions",
                           np.asarray([getattr(c.dimensions, b) for b in BASE],
                                      dtype=np.int32))
            if c.meaning:
                root.setncattr(f"constant_{cname}_meaning", c.meaning)
        if model.citation:
            root.setncattr("citation", "\n".join(model.citation))
        for key, value in meta.items():
            root.setncattr(f"meta_{key}", value)
        root.createDimension("layer", model.nlayers)
        root.createDimension("interface", len(model.geometry.interfaces))
        root.createDimension("bound", 2)
        b = model.skeleton.boundaries
        length = scaled_unit(Dimensions(length=1), model.scales)
        put_variable(root, "layer_bounds", ("layer", "bound"),
                     np.stack([b[:-1], b[1:]], -1),
                     {"long_name": "radii of the layer's inner and outer boundaries",
                      "units": length})
        var = root.createVariable("layer_name", str, ("layer",))
        for i, layer in enumerate(model.geometry.layers):
            var[i] = layer.name or ""
        faces = model.geometry.interfaces
        put_variable(root, "interface_radius", ("interface",),
             np.array([f.radius for f in faces], dtype=float),
             {"long_name": "radius of the interface", "units": length})
        var = root.createVariable("interface_name", str, ("interface",))
        for k, face in enumerate(faces):
            var[k] = face.name or ""
        var = root.createVariable("layer_part", "i4", ("layer",))
        part_of = np.empty(model.nlayers, dtype=np.int32)
        for k, (layers, _) in enumerate(parts):
            part_of[layers] = k
        var[...] = part_of
        var.long_name = "the part, part_<k>, holding the layer"
        for k, (layers, g) in enumerate(parts):
            mesh = (_part_mesh(model, layers, g, names, form) if radial is None
                    else _restricted(radial, layers))
            _write_part(root.createGroup(f"part_{k}"), model, k, layers, g, mesh,
                        names, interpolation, form)
        if form == "stack":
            _write_stack(root, model, parts, interpolation)
        if held:
            _write_surfaces(root, model, held, surface_grid, interpolation)


# -- reading --------------------------------------------------------------------

def variable_attributes(var: Any) -> dict[str, Any]:
    """A netCDF variable's attributes, numpy scalars made plain."""
    out = {}
    for key in var.ncattrs():
        value = var.getncattr(key)
        if isinstance(value, np.ndarray) and value.ndim == 0:
            value = value.item()
        elif isinstance(value, np.generic):
            value = value.item()
        out[key] = value
    return out


def variable_values(var: Any) -> np.ndarray:
    """A variable's numbers as float64, NaN where it holds its fill."""
    v = var[:]
    return (v.filled(np.nan) if np.ma.isMaskedArray(v)
            else np.asarray(v, dtype=float)).astype(float)


def grid_of_group(group: Any) -> AngularGrid:
    """The angular grid a group's coordinates and attributes hold."""
    from .sampling import AngularGrid
    lat = np.asarray(group.variables["latitude"][:], dtype=float)
    lon = np.asarray(group.variables["longitude"][:], dtype=float)
    theta, _ = from_geographic(lat, np.zeros_like(lat))
    _, phi = from_geographic(np.zeros_like(lon), lon)
    attrs = {k: group.getncattr(k) for k in group.ncattrs()}
    weights = (np.asarray(group.variables["colatitude_weights"][:], dtype=float)
               if "colatitude_weights" in group.variables else None)
    lmax = attrs.get("grid_lmax")
    return AngularGrid(theta, phi, kind=str(attrs.get("grid_kind", "custom")),
                       lmax=None if lmax is None else int(lmax), weights=weights)


def _fail(path: object, msg: str) -> NoReturn:
    raise ValueError(f"malformed model file {path}: {msg}")


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
        _fail(path, "no schema attribute: not a planetmodel model file")
    if ds.getncattr("schema") != SCHEMA:
        _fail(path, f"schema is {ds.getncattr('schema')!r}, this reader reads "
                    f"{SCHEMA!r}")
    for base in ("length", "mass", "time", "temperature"):
        key = f"scale_{base}"
        if key not in attrs or not float(ds.getncattr(key)) > 0.0:
            _fail(path, f"{key} is missing or not positive")
    if "rtol" in attrs:
        rtol = ds.getncattr("rtol")
        if (isinstance(rtol, str) or np.size(rtol) != 1
                or not 0.0 < float(rtol) < 1.0):
            _fail(path, f"rtol is {rtol!r}, not a tolerance in (0, 1)")
    for name, dims in (("layer_bounds", ("layer", "bound")),
                       ("layer_name", ("layer",)),
                       ("interface_radius", ("interface",)),
                       ("interface_name", ("interface",)),
                       ("layer_part", ("layer",))):
        if name not in ds.variables:
            _fail(path, f"no variable {name!r} at the root")
        if tuple(ds.variables[name].dimensions) != dims:
            _fail(path, f"{name} has dimensions {ds.variables[name].dimensions}, "
                        f"not {dims}")
    bounds = np.asarray(ds.variables["layer_bounds"][:], dtype=float)
    if not np.all(bounds[:, 1] > bounds[:, 0]) or not np.allclose(
            bounds[1:, 0], bounds[:-1, 1], rtol=0, atol=0):
        _fail(path, "layer_bounds do not make a skeleton of contiguous layers")
    layer_part = np.asarray(ds.variables["layer_part"][:], dtype=int)
    for k in np.unique(layer_part):
        if f"part_{k}" not in ds.groups:
            _fail(path, f"layer_part names part_{k}, which the file lacks")
        layers = np.flatnonzero(layer_part == k)
        if not np.array_equal(layers, np.arange(layers[0], layers[-1] + 1)):
            _fail(path, f"the layers of part {k} are not contiguous")
    for k in np.unique(layer_part):
        layers = np.flatnonzero(layer_part == k)
        _validate_part(ds.groups[f"part_{k}"], f"part_{k}", layers, bounds, path)
    has_stack = "geometry" in ds.groups
    holding = [k for k in np.unique(layer_part)
               if "displacement" in ds.groups[f"part_{k}"].variables]
    has_field = bool(holding)
    if has_field and len(holding) != np.unique(layer_part).size:
        _fail(path, f"the displacement is held in parts {holding} only; a mapping "
                    "held as a field is held in every part")
    if has_stack and has_field:
        _fail(path, "the mapping is held twice, as the geometry group and as a "
                    "displacement field")
    if has_stack:
        _validate_stack(ds.groups["geometry"], path)
    if "surfaces" in ds.groups:
        taken = {name for k in np.unique(layer_part)
                 for name in ds.groups[f"part_{k}"].variables}
        _validate_surfaces(ds.groups["surfaces"], path, len(ds.dimensions["interface"]),
                           taken)


def _validate_part(g: Any, where: str, layers: np.ndarray, bounds: np.ndarray,
                   path: object) -> None:
    """The checks on one part."""
    for dim in ("node", "element", "bound", "latitude", "longitude"):
        if dim not in g.dimensions:
            _fail(path, f"{where} has no dimension {dim!r}")
    for name in ("radius", "element_layer", "element_bounds", "latitude",
                 "longitude"):
        if name not in g.variables:
            _fail(path, f"{where} has no variable {name!r}")
    if "ngll" not in g.ncattrs() or "grid_kind" not in g.ncattrs():
        _fail(path, f"{where} lacks the attribute ngll or grid_kind")
    if g.getncattr("grid_kind") not in GRID_KINDS:
        _fail(path, f"{where}: grid_kind {g.getncattr('grid_kind')!r} is not one of "
                    f"{GRID_KINDS}")
    ngll = int(g.getncattr("ngll"))
    nelem, nnode = len(g.dimensions["element"]), len(g.dimensions["node"])
    if ngll < 2 or nelem * ngll != nnode:
        _fail(path, f"{where}: {nnode} nodes are not {nelem} elements of ngll = {ngll}")
    el = np.asarray(g.variables["element_layer"][:], dtype=int)
    if not np.array_equal(np.unique(el), layers):
        _fail(path, f"{where}: its elements lie in layers {np.unique(el).tolist()}, "
                    f"not {layers.tolist()}")
    eb = np.asarray(g.variables["element_bounds"][:], dtype=float)
    span = bounds[layers[-1], 1] - bounds[layers[0], 0]
    if not np.allclose(eb[1:, 0], eb[:-1, 1], rtol=0, atol=1e-12 * span) or not (
            np.isclose(eb[0, 0], bounds[layers[0], 0], rtol=0, atol=1e-12 * span)
            and np.isclose(eb[-1, 1], bounds[layers[-1], 1], rtol=0,
                           atol=1e-12 * span)):
        _fail(path, f"{where}: element_bounds do not tile its layers")
    lacks_pole = "pole" in g.dimensions
    own = {"radius", "element_layer", "element_bounds", "latitude", "longitude",
           "colatitude_weights"}
    for name, var in g.variables.items():
        if name in own or name.endswith("_poles"):
            continue
        a = set(var.ncattrs())
        dims = tuple(var.dimensions)
        if dims[:1] != ("node",):
            _fail(path, f"{where}/{name} does not run over the nodes")
        lateral = dims[1:3] == ("latitude", "longitude")
        need = ({"radial_interpolation"} if name == "displacement"
                else {"radial_interpolation", "rank", "weight", "voigt"})
        if not need <= a:
            _fail(path, f"{where}/{name} lacks the attributes "
                        f"{sorted(need - a)}")
        if name == "displacement":
            want: tuple[int, ...] = (3,)
        else:
            try:
                want = stored_shape(_character(variable_attributes(var)))
            except (ValueError, TypeError) as error:
                _fail(path, f"{where}/{name} states no character: {error}")
        tail = var.shape[3 if lateral else 1:]
        if tuple(tail) != tuple(want):
            _fail(path, f"{where}/{name} has trailing axes {tuple(tail)}, but its "
                        f"character stores {tuple(want)}")
        if "radial_interpolation" in a and var.getncattr(
                "radial_interpolation") not in RADIAL_METHODS:
            _fail(path, f"{where}/{name} states an unknown radial rule")
        if "radial_layers" in a:
            listed = _radial_layers(variable_attributes(var))
            held = (set(int(i) for i in np.atleast_1d(var.getncattr("layers")))
                    if "layers" in a else set(int(i) for i in layers))
            if not lateral or not listed <= held:
                _fail(path, f"{where}/{name} names radial_layers "
                            f"{sorted(listed)}, which must be layers a lateral "
                            "variable holds")
        if lateral:
            if "lateral_interpolation" not in a or var.getncattr(
                    "lateral_interpolation") not in LATERAL_METHODS:
                _fail(path, f"{where}/{name} states no known lateral rule")
            if lacks_pole and f"{name}_poles" not in g.variables:
                _fail(path, f"{where}/{name} varies with direction on a grid "
                            f"without its poles, and {name}_poles is missing")
        if "perturbation" in a:
            other = var.getncattr("perturbation")
            if other not in g.variables:
                _fail(path, f"{where}/{name} names its perturbation {other!r}, "
                            "which the part lacks")
            if ("perturbation_kind" not in a
                    or var.getncattr("perturbation_kind") not in PERTURBATION_KINDS):
                _fail(path, f"{where}/{name} states no known perturbation_kind, one "
                            f"of {PERTURBATION_KINDS}")


def _validate_stack(g: Any, path: object) -> None:
    """The checks on the geometry group."""
    for name in ("boundary_radius", "boundary_displacement", "latitude",
                 "longitude"):
        if name not in g.variables:
            _fail(path, f"geometry has no variable {name!r}")
    var = g.variables["boundary_displacement"]
    if tuple(var.dimensions) != ("boundary", "latitude", "longitude"):
        _fail(path, "geometry/boundary_displacement has the wrong dimensions")
    a = set(var.ncattrs())
    if ("lateral_interpolation" not in a
            or var.getncattr("lateral_interpolation") not in LATERAL_METHODS):
        _fail(path, "geometry/boundary_displacement states no known lateral rule")
    if "between" not in a or var.getncattr("between") != "linear":
        _fail(path, "geometry/boundary_displacement states an unknown rule between "
                    "surfaces")


#: The variables of the surfaces group that are not fields.
_SURFACE_OWN = ("surface_interface", "latitude", "longitude", "colatitude_weights")


def _validate_surfaces(g: Any, path: object, ninterface: int, taken: set[str]) -> None:
    """The checks on the surfaces group; `taken` holds the parts' names."""
    for dim in ("surface", "latitude", "longitude"):
        if dim not in g.dimensions:
            _fail(path, f"surfaces has no dimension {dim!r}")
    for name in ("surface_interface", "latitude", "longitude"):
        if name not in g.variables:
            _fail(path, f"surfaces has no variable {name!r}")
    if "grid_kind" not in g.ncattrs() or g.getncattr("grid_kind") not in GRID_KINDS:
        _fail(path, f"surfaces.grid_kind is missing or not a kind of grid, one of "
                    f"{GRID_KINDS}")
    var = g.variables["surface_interface"]
    faces = np.asarray(var[:], dtype=float)
    if (tuple(var.dimensions) != ("surface",) or np.any(faces != np.round(faces))
            or np.any(faces < 0) or np.any(faces >= ninterface)
            or np.any(np.diff(faces) <= 0)):
        _fail(path, "surfaces/surface_interface is not interfaces of the file, "
                    "increasing")
    lacks_pole = "pole" in g.dimensions
    for name, v in g.variables.items():
        if name in _SURFACE_OWN or name.endswith("_poles"):
            continue
        where = f"surfaces/{name}"
        if tuple(v.dimensions) != ("surface", "latitude", "longitude"):
            _fail(path, f"{where} does not run over the surfaces and the grid")
        if name in taken:
            _fail(path, f"{where} is a layer field's name too; a name is a layer "
                        "field's or a surface field's, never both")
        a = set(v.ncattrs())
        if "lateral_interpolation" not in a or v.getncattr(
                "lateral_interpolation") not in LATERAL_METHODS:
            _fail(path, f"{where} states no known lateral rule")
        if "interfaces" in a and not {float(k) for k in np.atleast_1d(
                v.getncattr("interfaces"))} <= set(faces.tolist()):
            _fail(path, f"{where} names interfaces that are not surfaces of the group")
        if lacks_pole and (f"{name}_poles" not in g.variables or tuple(
                g.variables[f"{name}_poles"].dimensions) != ("surface", "pole")):
            _fail(path, f"{where} is on a grid without its poles, and {name}_poles "
                        "is missing")


def read_model_file(path: str | os.PathLike[str]) -> ModelFile:
    """A model file's numbers and metadata, its structure validated first."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        _validate(ds, path)
        attrs = {k: ds.getncattr(k) for k in ds.ncattrs()}
        scales = Scales(**{b: float(attrs[f"scale_{b}"])
                           for b in ("length", "mass", "time", "temperature")})
        constants, declared = {}, {}
        for key, value in attrs.items():
            if not key.startswith("constant_") or key.endswith(("_dimensions",
                                                                 "_meaning")):
                continue
            cname = key[len("constant_"):]
            constants[cname] = float(value)
            exps = attrs.get(f"{key}_dimensions")
            if exps is not None:
                dims = Dimensions(**{b: int(e) for b, e in
                                     zip(BASE, np.atleast_1d(exps))})
                declared[cname] = Constant(
                    float(value) * scales.factor(dims), dims,
                    meaning=str(attrs.get(f"{key}_meaning", "")))
        meta = {k[len("meta_"):]: v.item() if isinstance(v, np.generic) else v
                for k, v in attrs.items() if k.startswith("meta_")}
        bounds = np.asarray(ds.variables["layer_bounds"][:], dtype=float)
        skeleton = Skeleton(np.append(bounds[:, 0], bounds[-1, 1]))
        layer_names = tuple(str(x) or None for x in ds.variables["layer_name"][:])
        interface_names = tuple(str(x) or None
                                for x in ds.variables["interface_name"][:])
        layer_part = np.asarray(ds.variables["layer_part"][:], dtype=int)
        parts = []
        for k in np.unique(layer_part):
            g = ds.groups[f"part_{k}"]
            layers = tuple(int(i) for i in np.flatnonzero(layer_part == k))
            eb = np.asarray(g.variables["element_bounds"][:], dtype=float)
            mesh = RadialMesh(skeleton, ngll=int(g.getncattr("ngll")),
                              edges=np.append(eb[:, 0], eb[-1, 1]))
            stored = np.asarray(g.variables["radius"][:], dtype=float)
            if not np.allclose(stored, mesh.r.ravel(), rtol=1e-12, atol=0):
                _fail(path, f"part_{k}: radius is not the GLL nodes of its elements")
            variables = {
                name: FileVariable(variable_values(var), variable_attributes(var))
                for name, var in g.variables.items()
                if name not in ("radius", "element_layer", "element_bounds",
                                "latitude", "longitude", "colatitude_weights")}
            parts.append(FilePart(layers, mesh, grid_of_group(g),
                                  MappingProxyType(variables)))
        geometry = None
        if "geometry" in ds.groups:
            g = ds.groups["geometry"]
            var = g.variables["boundary_displacement"]
            poles = (np.asarray(g.variables["boundary_displacement_poles"][:],
                                dtype=float)
                     if "boundary_displacement_poles" in g.variables else None)
            geometry = FileGeometry(
                np.asarray(g.variables["boundary_radius"][:], dtype=float),
                grid_of_group(g), np.asarray(var[:], dtype=float), poles,
                str(var.getncattr("lateral_interpolation")),
                str(var.getncattr("between")))
        surfaces = None
        if "surfaces" in ds.groups:
            g = ds.groups["surfaces"]
            faces = tuple(int(k) for k in
                          np.asarray(g.variables["surface_interface"][:]))
            variables = {name: FileVariable(variable_values(var),
                                            variable_attributes(var))
                         for name, var in g.variables.items()
                         if name not in _SURFACE_OWN}
            surfaces = FileSurfaces(faces, grid_of_group(g),
                                    MappingProxyType(variables))
    return ModelFile(schema=str(attrs["schema"]), name=attrs.get("name"),
                     scales=scales, constants=MappingProxyType(constants),
                     meta=MappingProxyType(meta), skeleton=skeleton,
                     layer_names=layer_names, interface_names=interface_names,
                     parts=tuple(parts), geometry=geometry,
                     declared=MappingProxyType(declared),
                     rtol=float(attrs.get("rtol", DEFAULT_RTOL)), surfaces=surfaces,
                     citation=tuple(str(attrs.get("citation", "")).splitlines()),
                     source=str(attrs.get("source", "")),
                     created=str(attrs.get("created", "")))


def _character(attrs: Mapping[str, Any]) -> Character:
    return Character(int(attrs["rank"]), int(attrs["weight"]),
                     voigt=bool(int(attrs["voigt"])))


def _rules(name: str, attrs: Mapping[str, Any],
           interpolation: Interpolation) -> tuple[str, str | None]:
    """The rules a reader uses: the caller's, else the stated ones."""
    if name in interpolation:
        return interpolation[name]
    return (str(attrs["radial_interpolation"]),
            None if "lateral_interpolation" not in attrs
            else str(attrs["lateral_interpolation"]))


def _layer_mesh(part: FilePart, layer: int) -> tuple[np.ndarray, np.ndarray, Mesh1D]:
    """The node mask, element indices and Mesh1D of one layer of a part."""
    elems = np.flatnonzero(part.mesh.layer == layer)
    mask = np.repeat(part.mesh.layer == layer, part.mesh.ngll)
    breaks = np.append(part.mesh.left[elems], part.mesh.right[elems[-1]])
    return mask, elems, Mesh1D(breaks, ngll=part.mesh.ngll)


def _radial_field(interval: tuple[float, float], nodal: np.ndarray, mesh: Mesh1D,
                  method: str, character: Character, name: str | None) -> RadialField:
    """Nodal values (nelem, ngll) + c as a radial field of exact pieces."""
    pp = mesh.to_ppoly(nodal, method=method)
    c = nodal.shape[2:]
    if not c:
        return RadialField(interval, PolynomialLayer(interval, pp),
                           character=character, name=name)
    functions = np.empty(c, dtype=object)
    for idx in np.ndindex(c):
        functions[idx] = PolynomialLayer(interval, type(pp)(pp.c[(...,) + idx], pp.x))
    return RadialField(interval, functions, character=character, name=name)


def _gridded_field(interval: tuple[float, float], part: FilePart, mask: np.ndarray,
                   elems: np.ndarray, mesh: Mesh1D, var: FileVariable,
                   poles: FileVariable | None, rules: tuple[str, str],
                   character: Character, name: str | None) -> Field:
    """A variable's values on one layer as a gridded field."""
    from .gridded import GriddedField
    grid = part.grid
    v = var.values[mask].reshape((elems.size, mesh.ngll) + var.values.shape[1:])
    p = None if poles is None else poles.values[mask].reshape(
        (elems.size, mesh.ngll) + poles.values.shape[1:])
    return GriddedField(interval, mesh.breakpoints, grid, v, character=character,
                        name=name, radial=rules[0], lateral=rules[1], poles=p)


def _variable_field(data: ModelFile, part: FilePart, layer: int, name: str,
                    var: FileVariable, interpolation: Interpolation, *,
                    character: Character | None = None) -> Field | None:
    """One variable on one layer as a field, or None where it is NaN there."""
    mask, elems, mesh = _layer_mesh(part, layer)
    values = var.values[mask]
    if np.all(np.isnan(values)):
        return None
    interval = tuple(float(x) for x in data.skeleton.boundaries[layer:layer + 2])
    char = _character(var.attributes) if character is None else character
    rules = _rules(name, var.attributes, interpolation)
    radial = var.values.ndim == 1 + len(stored_shape(char))
    if not radial and layer in _radial_layers(var.attributes):
        values, radial = values[:, 0, 0], True      # every column is the same
    if radial:
        nodal = values.reshape((elems.size, mesh.ngll) + values.shape[1:])
        return _radial_field(interval, nodal, mesh, rules[0], char, name)
    lateral = rules[1] or DEFAULT_INTERPOLATION[1]
    poles = part.variables.get(f"{name}_poles")
    return _gridded_field(interval, part, mask, elems, mesh, var, poles,
                          (rules[0], lateral), char, name)


def file_layers(data: ModelFile, *, interpolation: Interpolation | None = None
                ) -> list[dict[str, Field]]:
    """One mapping of name to field per layer: radial variables as exact
    polynomial layers through the nodes, the others as gridded fields,
    and references with perturbations as perturbed fields, read by the
    stated rules unless `interpolation` names others."""
    from .perturbation import PerturbedField
    interpolation = dict(interpolation or {})
    out: list[dict[str, Field]] = [{} for _ in range(data.skeleton.nlayers)]
    for part in data.parts:
        for name, var in part.variables.items():
            if (name.endswith(("_poles", "_perturbation")) or name == "displacement"):
                continue
            pert_name = var.attributes.get("perturbation")
            for layer in part.layers:
                field = _variable_field(data, part, layer, name, var, interpolation)
                if field is None:
                    continue
                if pert_name is not None:
                    # the perturbation is read by the rules asked for its
                    # field, or by its own stated ones
                    asked = ({pert_name: interpolation[name]}
                             if name in interpolation else {})
                    dp = _variable_field(data, part, layer, pert_name,
                                         part.variables[pert_name], asked)
                    if dp is not None:
                        field = PerturbedField(field, dp.renamed(None),
                                               kind=var.attributes["perturbation_kind"],
                                               name=name)
                out[layer][name] = field
    return out


def file_geometry(data: ModelFile, *, interpolation: Interpolation | None = None
                  ) -> Geometry:
    """The geometry a file holds: its skeleton and names, and its mapping
    from the geometry group or the displacement field, read by the stated
    rules unless `interpolation` names others."""
    from .displacement import LayerLinear
    from .geometry import Geometry
    from .gridded import GriddedDisplacement, GriddedFunction
    from .mapping import RadialStretch
    interpolation = dict(interpolation or {})
    sk = data.skeleton
    mapping = None
    if data.geometry is not None:
        geo = data.geometry
        lateral = interpolation.get("boundary_displacement", (None, geo.lateral))[1]
        reliefs = []
        for k in range(geo.radii.size):
            v = geo.values[k]
            p = None if geo.poles is None else geo.poles[k]
            if not np.any(v) and (p is None or not np.any(p)):
                reliefs.append(None)
            else:
                reliefs.append(GriddedFunction(geo.grid, v, lateral=lateral, poles=p))
        mapping = RadialStretch(LayerLinear(Skeleton(geo.radii), reliefs),
                                rmax=float(sk.boundaries[-1]))
    elif any("displacement" in part.variables for part in data.parts):
        fields = []
        for layer in range(sk.nlayers):
            part = data.part_of(layer)
            var = part.variables["displacement"]
            fields.append(_variable_field(data, part, layer, "displacement", var,
                                          interpolation, character=VECTOR))
        mapping = GriddedDisplacement(sk, fields)
    return Geometry(sk, mapping=mapping, layer_names=list(data.layer_names),
                    interface_names=list(data.interface_names), rtol=data.rtol)


def file_surfaces(data: ModelFile, *, interpolation: Interpolation | None = None
                  ) -> dict[str, dict[int | str, GriddedFunction]]:
    """The functions on interfaces a file holds, name -> {interface ->
    `GriddedFunction`}, an interface keyed by its name, or by its index
    where it has none; each read by its stated lateral rule unless
    `interpolation` names another, and empty for a file without them."""
    from .gridded import GriddedFunction
    interpolation = dict(interpolation or {})
    held = data.surfaces
    if held is None:
        return {}
    out: dict[str, dict[int | str, GriddedFunction]] = {}
    for name, var in held.variables.items():
        if name.endswith("_poles"):
            continue
        lateral = (interpolation[name][1] if name in interpolation
                   else str(var.attributes["lateral_interpolation"]))
        poles = held.variables.get(f"{name}_poles")
        on: dict[int | str, GriddedFunction] = {}
        for row, k in enumerate(held.interfaces):
            if np.all(np.isnan(var.values[row])):
                continue
            key = data.interface_names[k]
            on[k if key is None else key] = GriddedFunction(
                held.grid, var.values[row], lateral=lateral,
                poles=None if poles is None else poles.values[row])
        out[name] = on
    return out


def _dimensions(attributes: Mapping[str, Any]) -> Dimensions | None:
    """The dimensions a variable states, or None where it states none."""
    if "dimensions" not in attributes:
        return None
    return Dimensions(**dict(zip(BASE, (int(x) for x in np.atleast_1d(
        attributes["dimensions"])))))


def _specs(data: ModelFile) -> dict[str, FieldSpec]:
    """The specs the file states for names outside the vocabulary, its
    statements for vocabulary names checked against the vocabulary; a
    surface field's character is a number's."""
    stated: list[tuple[str, FieldSpec]] = []
    for part in data.parts:
        for name, var in part.variables.items():
            if name.endswith(("_poles", "_perturbation")) or name == "displacement":
                continue
            a = var.attributes
            stated.append((name, FieldSpec(_character(a), _dimensions(a),
                                           meaning=str(a.get("long_name", "")))))
    if data.surfaces is not None:
        for name, var in data.surfaces.variables.items():
            if not name.endswith("_poles"):
                a = var.attributes
                stated.append((name, FieldSpec(SCALAR, _dimensions(a),
                                               meaning=str(a.get("long_name", "")))))
    out: dict[str, FieldSpec] = {}
    for name, spec in stated:
        known = shipped(name)
        if known is None:
            out[name] = spec
            continue
        surface = data.surfaces is not None and name in data.surfaces.variables
        same = (known.character.rank == 0 if surface
                else known.character == spec.character)
        if not same or (spec.dimensions is not None
                        and known.dimensions != spec.dimensions):
            raise ValueError(
                f"the file states {name!r} as {spec.character}, {spec.dimensions}, "
                f"where the vocabulary has {known.character}, {known.dimensions}")
    return out


def read_model(path: str | os.PathLike[str], *,
               interpolation: Interpolation | None = None,
               scales: Scales | None = None) -> Model:
    """The model a file holds, as a plain `Model` with the file's specs,
    in the file's scales or, with `scales=`, converted to those."""
    from .model import Model
    data = read_model_file(path)
    model = Model(file_geometry(data, interpolation=interpolation),
                  file_layers(data, interpolation=interpolation),
                  scales=data.scales, specs=_specs(data), name=data.name,
                  constants=data.declared, citation=data.citation)
    return model if scales is None else model.converted(scales)


def read_surfaces(path: str | os.PathLike[str], *,
                  interpolation: Interpolation | None = None
                  ) -> dict[str, dict[int | str, GriddedFunction]]:
    """The functions on interfaces a file holds, as `file_surfaces` gives
    them, in the file's scales (the model's specs, from `read_model`,
    give their dimensions); empty for a file without them."""
    return file_surfaces(read_model_file(path), interpolation=interpolation)
