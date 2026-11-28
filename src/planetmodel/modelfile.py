"""The model file: a model in one netCDF file, on its reference skeleton.

One netCDF-4 file holds a model as numbers: its skeleton, its fields at
the nodes of a radial layout and an angular grid, and its mapping, in
the model's own units with its scales recorded, CF-styled so that
xarray, Panoply and ncview open it as it stands.  The Python here is the
definition of the format and the reference the C++ readers are held to;
`docs/source/formats/model_file.rst` describes it for consumers.  This
docstring describes the spherical file; a flat model
(`planetmodel.cartesian`) is written to a file of schema `SCHEMA_FLAT`,
which `cartesian.modelfile` describes where it differs.

**Global attributes**: `schema` (`SCHEMA`, or `SCHEMA_WITH_LOG_RULES`
for a file that states a radial rule in the logarithm; checked by every
reader), `Conventions`, `source` (the writing code and version) and `created`
(when, ISO 8601), both stamped by the writers and optional on reading,
empty where a file states none; the model's `name`, `scale_length`, `scale_mass`,
`scale_time` and `scale_temperature` (what one stored unit is in SI),
`rtol` (the geometry's tolerance, the default where a file has none),
`constant_<name>` for every constant in the model's units, `citation`
(the works the model cites, one a line, where it cites any),
`behaviours` (the names of the model's behaviours, `Model.behaviours`,
separated by spaces, where it has any: what kind of model the numbers
are, below), and `meta_<key>` for the writer's `meta`.  **The skeleton**, at the root:
dimensions `layer`, `interface` and `bound` (two), `layer_bounds(layer,
bound)`, `layer_name(layer)`, `interface_radius(interface)`,
`interface_name(interface)` and `layer_part(layer)`.  The interfaces are
those of a `Geometry` over the skeleton `layer_bounds` make, every
boundary but a centre, and `interface_radius` holds their radii: a
reader refuses a file in which it does not.

**Parts.**  The layers are cut into parts, contiguous runs of whole
layers, one group each (`part_0`, `part_1`, ... from the centre), each
on an angular grid of its own; a radial model, every field of the radius
alone under the identity, is one part on the grid of one node that
`write_model` supplies where it is given none.  A part holds the
`Sample` layout of its layers: dimension `node`, the per-element GLL
nodes flattened element by element, so that an interface is two nodes at
one radius; `radius(node)`
(an auxiliary coordinate, since it repeats), `element_layer(element)`,
`element_bounds(element, bound)` and the attribute `ngll`, the elements
tiling the part's layers, each within the layer `element_layer` gives
it (a reader refuses a part in which they do not); the grid as
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
south, its components Cartesian, evaluated exactly by the writer; a
reader asks the grid's nodes whether it lacks one (a first latitude that
is not 90, or a last that is not -90), and refuses a part in which such
a variable is without them.

**The mapping** is held in one of two forms.  A mapping that offers the
reliefs of its surfaces (a radial stretch by `layer_linear`) is the
group `geometry`: `boundary_radius(boundary)`, the radii of its
surfaces, two or more and strictly increasing, and
`boundary_displacement(boundary, latitude, longitude)` on a grid of its
own, with `boundary_displacement_poles(boundary, pole)` where that grid
lacks a pole and the attributes `lateral_interpolation` and `between`
(the rule between surfaces, `linear`).  Any other mapping is the field
`displacement(node, latitude, longitude, component)` of every part, m(X)
- X in the spherical frame at X, as a sample holds it, with
`displacement_poles` as any variable that varies with direction has
them, read back as a `GriddedDisplacement`.  The identity is neither.

**Fields on interfaces**, functions of the angles on the skeleton's own
interfaces (a shape kernel, the shift of a boundary), are the optional
group `surfaces`: dimension `surface` over the interfaces that hold any,
`surface_interface(surface)` their indices in the root's `interface`,
increasing; a grid of its own, as a part's is; and one variable per
name, `name(surface, latitude, longitude)`, NaN on an interface that
lacks it, with the attributes `long_name`, `units`, `dimensions` (where
declared), `lateral_interpolation` and `interfaces` (those holding it),
and `name_poles(surface, pole)` where the grid lacks a pole, without
which a reader refuses the group.  A name is
a layer field's or a surface field's, never both, and a surface field
is a number at each point, its spec's dimensions its own.  A file
without the group reads as one with none.

**Radial rules.**  `polynomial` and `linear` read the values between a
part's nodes (`mesh1d.RADIAL_METHODS`).  The rules in the logarithm,
`log-polynomial` and `log-linear` (`LOG_RULES`), read the logarithms of
the values by those rules and take the exponential, y_e exp(P(r)) on
element e, y_e its first value and P the rule's polynomial through the
log(y / y_e) of its nodes, so that an element of one value reads that
value to the bit; a variable of rank 0 of the radius alone, positive at
every node of the layers it holds, states them, and a file in which one
does is of schema `SCHEMA_WITH_LOG_RULES`, which a reader of `SCHEMA`
alone refuses.

**The radial layout** is exact where it can be: the writer builds, per
part, a radial mesh whose element edges are the layer boundaries, every
breakpoint a field declares for its variable's rule (`radial_pieces`, or
under a rule in the logarithm `log_radial_pieces`, its logarithm's) and
every knot of the mapping, with `ngll` one more than the highest
declared degree; a field that declares no such description subdivides
the elements of its layer to the spacing a sample takes from the grid's
band (every layer, for a mapping written as a displacement field), and
raises `ngll` to at least 5, which holds it to that sampling's accuracy
and no better.  So a field declaring a piecewise polynomial, or the
exponential of one, written and read back is the same function, the
fields of a model built from a deck among them, and a band-limited field
on a grid of its band is held exactly.  `radial=` gives the layout instead.
**Every variable states its intended interpolation**: a gridded field
its own rules, a harmonic field `polynomial` and `harmonic`, a variable
of the radius alone some of whose fields are the exponentials of
piecewise polynomials, declaring `log_radial_pieces` and not
`radial_pieces`, and none of which declares the opposite,
`log-polynomial`, any other the defaults (`polynomial`, and `linear`
laterally), unless `interpolation=` names others.  A variable some of
whose layers ask for a rule in the logarithm and others for another, or
that asks for one and varies with direction, is refused by name:
`interpolation=` reads it by one rule, approximately where that rule
does not hold a field.  A reader uses the stated rules unless its
caller overrides them, and refuses a file in which a variable states a
lateral rule the grid of its group cannot carry (`lateral.check_rule`:
the harmonic rule on a grid that is not of a shipped kind with its
band, `grid_kind` and `grid_lmax`).

**The two kinds.**  What a coordinate system decides of the file is a
`Layout`, asked by the writer and the readers here, which hold the rest
in common: `SphericalLayout`, of `SCHEMAS`, and the flat
`cartesian.modelfile.FlatLayout`, of `SCHEMA_FLAT`.  `write_model` takes
the layout of the model's coordinate system; a reader takes the one of
the schema the file states.

**What the model is.**  A model's behaviours are the mixins of its class
(`planetmodel.behaviours`: `Elastic`, `ConstantQ`, `SelfGravitating`,
`Viscoelastic`, or a model type's own), methods over its fields that
hold no numbers, so a file names them and carries nothing more: the
attribute `behaviours`, the class names in the order of the bases,
absent for a plain model.  `read_model` gives back a model with those of
`behaviours.BEHAVIOURS` that the file names, a plain model from a file
naming none, so that a file from before the attribute reads as it did.
A name that is none of those is a behaviour of another program's own,
whose code a file cannot carry: the model read lacks its methods and
keeps its name, which `write_model` states again in its place, and the
reading warns, naming it.  The fields are what the file holds either
way: nothing is derived on reading.

`read_model_file` gives a spherical file's numbers and metadata as a
`ModelFile`, the way `read_deck` gives a deck, refusing a flat one by
its schema (`cartesian.modelfile.read_model_file` reads that, and refuses
a spherical one); `file_layers`, `file_geometry` and `file_surfaces` turn
the numbers of either kind into fields, a geometry and the functions on
interfaces; `read_model` is the model of either kind, by what the
file states, and `read_surfaces` the functions beside it; `schema_of`
reads the schema alone.  A variable
read by a rule in the logarithm is an `ExponentialLayer` on each layer,
or the constant where the layer holds one value.
netCDF4 (the `netcdf` extra) is imported where a file is opened.  Real
fields only; a model is never converted on writing, and is read back in
the file's scales unless `scales=` asks for others.
"""
from __future__ import annotations

import numbers
import os
import warnings
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import KW_ONLY, dataclass
from pathlib import Path
from types import MappingProxyType, ModuleType
from typing import TYPE_CHECKING, Any, NoReturn, Protocol

import numpy as np
from numpy.typing import ArrayLike
from scipy.interpolate import PPoly

from .character import SCALAR, VECTOR, Character
from .fields import Field, RadialField, from_inside, stored_shape
from .files import written_whole
from .frames import from_geographic, geographic, spherical_frame
from .layerfunction import ExponentialLayer, PolynomialLayer, constant_layer
from .lateral import METHODS as LATERAL_METHODS
from .lateral import check_rule
from .mesh1d.mesh import RADIAL_METHODS, Mesh1D, RadialMesh
from .perturbation import KINDS as PERTURBATION_KINDS
from .sampling import KINDS as GRID_KINDS
from .skeleton import DEFAULT_RTOL, Skeleton
from .units import BASE, Dimensions, Scales, scaled_unit
from .vocabulary import Constant, FieldSpec, shipped

if TYPE_CHECKING:
    from .cartesian.domains import Domain
    from .displacement import Relief
    from .geometry import GeometryBase
    from .gridded import GriddedFunction
    from .model import ModelBase
    from .sampling import AngularGrid, LateralGrid

__all__ = ["SCHEMA", "SCHEMA_WITH_LOG_RULES", "SCHEMA_FLAT", "SCHEMAS", "LOG_RULES",
           "RADIAL_RULES", "CONVENTIONS", "DEFAULT_INTERPOLATION", "Grids",
           "Interpolation", "Surfaces", "ModelType", "FileVariable",
           "FilePart", "FileGeometry", "FileSurfaces", "ModelFile", "Layout",
           "SphericalLayout", "write_model", "read_model_file", "file_layers",
           "file_geometry", "file_surfaces", "read_model", "read_surfaces",
           "validate_structure", "schema_of", "read_with", "relief_grid",
           "radial_grid",
           "displacement_fields", "malformed", "real_values", "netcdf_module",
           "put_variable", "variable_attributes", "variable_values", "grid_degrees",
           "grid_of_group", "attribute_text", "attribute_number", "stamped_source"]

#: The schema string every reader checks, of a file in which no variable
#: states a rule in the logarithm; bumped only for an incompatible change.
SCHEMA = "planetmodel.model/1"

#: The schema of a file in which some variable states a rule in the
#: logarithm (`LOG_RULES`), which a reader of `SCHEMA` alone does not
#: know, and so refuses the file by its schema before reading a value.
SCHEMA_WITH_LOG_RULES = "planetmodel.model/2"

#: The schema of a flat model file, a model of `planetmodel.cartesian`
#: (`cartesian.modelfile` describes it), which a reader of the spherical
#: schemas refuses by its schema; a flat file states a rule in the
#: logarithm under this one schema, its readers knowing them all.
SCHEMA_FLAT = "planetmodel.model/3"

#: The schemas of a spherical model file, which `read_model_file` reads.
SCHEMAS = (SCHEMA, SCHEMA_WITH_LOG_RULES)

#: The reader of the numbers of each kind of file, named where a reader of
#: the other kind refuses one.
_READERS = {SCHEMA: "planetmodel.modelfile.read_model_file",
            SCHEMA_WITH_LOG_RULES: "planetmodel.modelfile.read_model_file",
            SCHEMA_FLAT: "planetmodel.cartesian.modelfile.read_model_file"}

#: The radial rules in the logarithm: the logarithms of the values read
#: by the rule of `mesh1d.RADIAL_METHODS` they name, the value their
#: exponential, so that a positive field whose logarithm is a piecewise
#: polynomial is held exactly.
LOG_RULES = tuple(f"log-{method}" for method in RADIAL_METHODS)

#: Every radial rule a variable may state.
RADIAL_RULES = RADIAL_METHODS + LOG_RULES

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

#: What `read_model` is asked to give back: a model type, or the
#: behaviours of one, mixin classes over the leaf of the file's kind.
type ModelType = type[ModelBase] | Iterable[type]


def netcdf_module() -> ModuleType:
    """netCDF4, imported where a model file, a surface file or a fault file
    is opened: the module, or an ImportError naming the `netcdf` extra
    that provides it."""
    try:
        import netCDF4
    except ImportError as exc:  # pragma: no cover - exercised by the extra
        raise ImportError(
            "reading and writing a netCDF file needs netCDF4.  Install it with:\n"
            "    pip install 'planetmodel[netcdf]'      "
            "(or: poetry install --extras netcdf)") from exc
    return netCDF4


def stamped_source(source: str | None) -> str:
    """The attribute `source` a writer of a model file, a surface file or
    a fault file stamps: `source`, naming the writing code and version,
    or this library and its version where it is None; a source that is
    not text, or is blank, is refused."""
    if source is None:
        from . import __version__
        return f"planetmodel {__version__}"
    if not isinstance(source, str) or not source.strip():
        raise ValueError("source names the writing code and version, got "
                         f"{source!r}")
    return source


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
    #: The names of the model's behaviours, as the attribute `behaviours`
    #: states them; empty where the file states none.
    behaviours: tuple[str, ...] = ()
    #: The code that wrote the file (the attribute `source`); empty where
    #: the file states none.
    source: str = ""
    #: When the file was written, ISO 8601 (the attribute `created`);
    #: empty where the file states none.
    created: str = ""
    #: The lateral domain of a flat file, a `cartesian.Box` or
    #: `cartesian.Rectangle`; None for a spherical one.
    domain: Domain | None = None

    def part_of(self, layer: int) -> FilePart:
        """The part holding `layer`."""
        for part in self.parts:
            if layer in part.layers:
                return part
        raise IndexError(f"no part holds layer {layer}")


# -- what a coordinate system decides ---------------------------------------------

class Layout(Protocol):
    """What a model file's coordinate system decides, asked by the writer
    and the readers of this module, which hold the rest of the format in
    common: `SphericalLayout`, of `SCHEMAS`, and the flat
    `cartesian.modelfile.FlatLayout`, of `SCHEMA_FLAT`.

    A layout names the system whose models it writes, the schemas it
    reads, its layered coordinate and the names it keeps; it says which
    boundaries of a skeleton are interfaces; it states the lateral grid
    of a part and of the surfaces group and reads it back, with the
    lateral rules the grid carries; it evaluates a field at a part's
    nodes; it holds the mapping in the forms it has; and it builds the
    fields, functions and geometry the numbers are read back as, and names
the leaf of the model they make.
    `context` is what `validate_root` reads of a file's root for the
    rest of its reading (a flat file's domain; nothing on the sphere).
    """

    #: The coordinate system of the models written in the layout,
    #: `Coordinates.name`.
    system: str
    #: The schemas the layout reads.
    schemas: tuple[str, ...]
    #: The layered coordinate: a part's node variable, and the suffix of
    #: the root's `interface_<coordinate>` and the geometry group's
    #: `boundary_<coordinate>`.
    coordinate: str
    #: The `long_name` of `layer_bounds`, of `interface_<coordinate>` and of
    #: a part's `element_bounds`.
    bounds_meaning: str
    interface_meaning: str
    element_meaning: str
    #: Names the format keeps for itself, which no field may take.
    reserved: tuple[str, ...]
    #: The lateral rules a variable may state.
    methods: tuple[str, ...]
    #: The lateral rule that analyses the grid, stated only where it holds
    #: the fields exactly (`spectral_holds`).
    spectral: str
    #: How a function of the lateral coordinates is called, in a message.
    arguments: str
    #: The attributes every part states.
    part_attributes: tuple[str, ...]
    #: The trailing shape of the variable `displacement`.
    displacement_shape: tuple[int, ...]

    def schema(self, logged: bool) -> str:
        """The schema of a file in which some variable states a rule in
        the logarithm (`logged`), or none does."""
        ...

    def check_schema(self, schema: str, logged: Sequence[str], path: object) -> None:
        """Refuse a file whose schema disagrees with whether a variable
        states a rule in the logarithm, `logged` naming those that do."""
        ...

    def check_grid(self, model: ModelBase, grid: object, layer: int) -> None:
        """Refuse, by name, a grid the layout does not take for `layer`."""
        ...

    def mapping_form(self, model: ModelBase) -> str:
        """How the model's mapping is held: 'none' (the identity), 'stack'
        (the geometry group) or 'field' (a displacement in every part);
        one the layout cannot hold is refused by name."""
        ...

    def radial_grid(self, model: ModelBase) -> Any:
        """The grid of one node that `write_model` supplies where it is
        given none, for a model of the layered coordinate alone."""
        ...

    def outer(self, parts: Sequence[tuple[list[int], Any]]) -> Any:
        """The grid of the part at the model's outer or free surface."""
        ...

    def check_rule(self, grid: Any, rule: str) -> None:
        """Refuse, by name, a lateral rule the grid cannot carry."""
        ...

    def spectral_holds(self, grid: Any, fields: Sequence[object]) -> bool:
        """Whether the spectral rule on `grid` holds every one of `fields`
        that varies laterally exactly."""
        ...

    def inexact_step(self, model: ModelBase, grid: Any) -> float:
        """The widest element in the layered coordinate of a layer whose
        field declares no polynomial description, as a sample of the grid
        takes it; refused by name where the grid sets none."""
        ...

    def write_root(self, root: Any, model: ModelBase) -> None:
        """The root's statements of the layout, after the shared ones."""
        ...

    def grid_dimensions(self, group: Any, grid: Any) -> None:
        """The grid's dimensions in a group."""
        ...

    def grid_attributes(self, group: Any, grid: Any) -> None:
        """The grid's attributes of a group."""
        ...

    def grid_variables(self, group: Any, grid: Any, length: str) -> None:
        """The grid's coordinate variables in a group, `length` the unit of
        a length."""
        ...

    def node_attributes(self, length: str) -> dict[str, Any]:
        """The attributes of a part's node variable."""
        ...

    def lacks_pole(self, grid: Any) -> bool:
        """Whether a variable on the grid needs its polar values beside it."""
        ...

    def lateral_dims(self, grid: Any) -> tuple[str, ...]:
        """The grid's dimensions, in a variable's order."""
        ...

    def lateral_shape(self, grid: Any) -> tuple[int, ...]:
        """The grid's shape, in a variable's order."""
        ...

    def evaluate(self, field: Field, r: np.ndarray, grid: Any, *,
                 radial: bool) -> np.ndarray:
        """A field at the nodes `r`: (n,) + c when `radial`, (n,) + the
        grid's shape + c otherwise, real."""
        ...

    def poles_of(self, field: Field, r: np.ndarray) -> np.ndarray:
        """A field at both poles at the nodes `r`, (n, 2) + c, where the
        grid lacks one."""
        ...

    def on_grid(self, fn: Callable[[np.ndarray, np.ndarray], ArrayLike],
                grid: Any) -> ArrayLike:
        """A function of the lateral coordinates on the grid's nodes."""
        ...

    def at_poles(self, fn: Callable[[np.ndarray, np.ndarray], ArrayLike]
                 ) -> ArrayLike | None:
        """A function of the lateral coordinates at both poles, where the
        layout has any."""
        ...

    def write_displacement(self, group: Any, model: ModelBase, r: np.ndarray,
                           grid: Any, rules: tuple[str, str], *,
                           lacks_pole: bool, length: str) -> None:
        """The mapping held as a displacement in one part, stating the
        (radial, lateral) `rules`."""
        ...

    def write_stack(self, root: Any, model: ModelBase,
                    parts: Sequence[tuple[list[int], Any]],
                    interpolation: Interpolation) -> None:
        """The mapping held as the geometry group."""
        ...

    def validate_root(self, ds: Any, path: object) -> Any:
        """The checks on the root's statements of the layout, giving the
        context of the rest of the reading."""
        ...

    def interfaces(self, skeleton: Skeleton, context: Any) -> np.ndarray:
        """The layered coordinate of every interface of `skeleton`, in the
        order of the root's `interface` dimension."""
        ...

    def axes(self, context: Any) -> tuple[str, ...]:
        """The lateral dimensions of every grid of the file."""
        ...

    def grid_names(self, context: Any) -> tuple[str, ...]:
        """The variables a group's grid takes."""
        ...

    def validate_grid(self, group: Any, where: str, path: object,
                      context: Any) -> None:
        """The checks on a part's or the surfaces group's grid, beyond its
        dimensions and coordinate variables."""
        ...

    def read_grid(self, group: Any, context: Any) -> Any:
        """The grid a group's coordinates and attributes hold."""
        ...

    def validate_stack(self, group: Any, path: object, context: Any) -> None:
        """The checks on the geometry group."""
        ...

    def read_stack(self, group: Any, context: Any) -> FileGeometry:
        """The geometry group's numbers."""
        ...

    def radial_field(self, interval: tuple[float, float], function: Any, *,
                     character: Character, name: str | None) -> RadialField:
        """A field of the layered coordinate alone in the layout's system."""
        ...

    def gridded_field(self, interval: tuple[float, float], breaks: np.ndarray,
                      grid: Any, values: np.ndarray, poles: np.ndarray | None, *,
                      character: Character, name: str | None, radial: str,
                      lateral: str) -> Field:
        """Values on a grid at GLL nodes as a gridded field."""
        ...

    def gridded_function(self, grid: Any, values: np.ndarray, *, lateral: str,
                         poles: np.ndarray | None) -> Any:
        """Values on a grid as a gridded function of the lateral coordinates."""
        ...

    def geometry(self, data: ModelFile, interpolation: Interpolation
                 ) -> GeometryBase:
        """The geometry a file holds, its mapping read by the stated rules
        unless `interpolation` names others."""
        ...

    def leaf(self) -> type[ModelBase]:
        """The leaf a model of the layout's kind is built on."""
        ...


def relief_grid(reliefs: Sequence[object], default: Any,
                interpolation: Interpolation) -> tuple[Any, str]:
    """The grid and lateral rule the geometry group holds a stack's
    `reliefs` (None where a surface keeps its place) by: the one grid every
    relief given is held on, `default` where none is given, and the rule
    they share, the bicubic one where they share none, unless
    `interpolation` names one for `boundary_displacement`.  Reliefs not
    held on one grid of their own are refused by name."""
    given = [rel for rel in reliefs if rel is not None]
    grids = [getattr(rel, "grid", None) for rel in given]
    if not given:
        grid = default
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
    return grid, lateral


class SphericalLayout:
    """The spherical model file's `Layout`: the module docstring's parts on
    angular grids, their polar values, components in the spherical frame,
    the mapping as boundary displacements or a displacement field, read
    back as a `Model` or a model type over it."""

    system = "spherical"
    schemas = SCHEMAS
    coordinate = "radius"
    bounds_meaning = "radii of the layer's inner and outer boundaries"
    interface_meaning = "radius of the interface"
    element_meaning = "radii of the element's ends"
    reserved = _RESERVED
    methods = LATERAL_METHODS
    spectral = "harmonic"
    arguments = "(theta, phi)"
    part_attributes = ("ngll", "grid_kind")
    displacement_shape = (3,)

    # -- writing

    def schema(self, logged: bool) -> str:
        """`SCHEMA_WITH_LOG_RULES` where a variable states a rule in the
        logarithm, `SCHEMA` otherwise."""
        return SCHEMA_WITH_LOG_RULES if logged else SCHEMA

    def check_schema(self, schema: str, logged: Sequence[str], path: object) -> None:
        """A rule in the logarithm is stated in a file of
        `SCHEMA_WITH_LOG_RULES` alone, and such a file states one."""
        if logged and schema != SCHEMA_WITH_LOG_RULES:
            _fail(path, f"{logged[0]} states a rule in the logarithm, which a file "
                        f"of schema {schema!r} does not hold: such a file is "
                        f"{SCHEMA_WITH_LOG_RULES!r}")
        if not logged and schema == SCHEMA_WITH_LOG_RULES:
            _fail(path, f"schema is {schema!r}, a file in which a variable states "
                        "a rule in the logarithm, and none does")

    def check_grid(self, model: ModelBase, grid: object, layer: int) -> None:
        """An `AngularGrid` for every layer."""
        from .sampling import AngularGrid
        if not isinstance(grid, AngularGrid):
            raise TypeError(f"the grid of layer {layer} is not an AngularGrid")

    def mapping_form(self, model: ModelBase) -> str:
        """A radial stretch of boundary reliefs is the stack; any other
        mapping a displacement field."""
        geometry = model.geometry
        if geometry.is_identity:
            return "none"
        h = getattr(geometry.mapping, "h", None)
        if (getattr(h, "reliefs", None) is not None
                and getattr(h, "boundaries", None) is not None):
            return "stack"
        return "field"

    def radial_grid(self, model: ModelBase) -> AngularGrid:
        """The Gauss-Legendre grid of band zero, `gauss_legendre(0)`: the
        equator at longitude zero."""
        from .sampling import gauss_legendre
        return gauss_legendre(0)

    def outer(self, parts: Sequence[tuple[list[int], Any]]) -> Any:
        """The last part's grid, the parts running from the centre."""
        return parts[-1][1]

    def check_rule(self, grid: AngularGrid, rule: str) -> None:
        """`lateral.check_rule`."""
        check_rule(grid, rule)

    def spectral_holds(self, grid: AngularGrid, fields: Sequence[object]) -> bool:
        """A grid of a shipped kind whose band is at least that of every
        field varying with direction: the field's own `lmax`, or, of a
        field read by the harmonic rule on a grid of a shipped kind, that
        grid's band, the band of what the rule reads."""
        if grid.kind == "custom" or grid.lmax is None:
            return False
        for f in fields:
            if getattr(f, "is_radial", False):
                continue
            band = getattr(f, "lmax", None)
            own = getattr(f, "grid", None)
            if (band is None and getattr(f, "lateral", None) == self.spectral
                    and getattr(own, "kind", "custom") != "custom"):
                band = getattr(own, "lmax", None)
            if band is None or band > grid.lmax:
                return False
        return True

    def inexact_step(self, model: ModelBase, grid: AngularGrid) -> float:
        """0.1 / (lmax + 1) of the outer radius, from the grid's band."""
        if grid.lmax is None:
            raise ValueError(
                "a field or the mapping declares no polynomial description in "
                f"radius, and the {grid.kind!r} grid carries no band to size "
                "the elements by: pass radial=, a RadialMesh over the skeleton")
        return 0.1 * float(model.skeleton.boundaries[-1]) / (grid.lmax + 1)

    def write_root(self, root: Any, model: ModelBase) -> None:
        """Nothing beyond the shared statements."""

    def grid_dimensions(self, group: Any, grid: AngularGrid) -> None:
        """`latitude` and `longitude`."""
        group.createDimension("latitude", grid.ntheta)
        group.createDimension("longitude", grid.nphi)

    def grid_attributes(self, group: Any, grid: AngularGrid) -> None:
        """`grid_kind`, and `grid_lmax` where the grid has a band."""
        group.setncattr("grid_kind", grid.kind)
        if grid.lmax is not None:
            group.setncattr("grid_lmax", np.int32(grid.lmax))

    def grid_variables(self, group: Any, grid: AngularGrid, length: str) -> None:
        """The CF coordinates `latitude` and `longitude`, and the colatitude
        weights where the grid has them."""
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
                 {"long_name": "colatitude quadrature weights against sin(theta) "
                               "dtheta", "units": "1"})

    def node_attributes(self, length: str) -> dict[str, Any]:
        """`radius(node)`: an element boundary is two nodes."""
        return {"long_name": "radius of the node; an element boundary is two nodes",
                "units": length}

    def lacks_pole(self, grid: AngularGrid) -> bool:
        """Whether either pole is not a node."""
        return not all(grid.pole_nodes)

    def lateral_dims(self, grid: AngularGrid) -> tuple[str, ...]:
        """(latitude, longitude)."""
        return ("latitude", "longitude")

    def lateral_shape(self, grid: AngularGrid) -> tuple[int, ...]:
        """(ntheta, nphi)."""
        return (grid.ntheta, grid.nphi)

    def evaluate(self, field: Field, r: np.ndarray, grid: AngularGrid, *,
                 radial: bool) -> np.ndarray:
        """`_evaluate`: a radial field's local components at the equator,
        any other's on the grid, through its own `on_grid` where it has one."""
        return _evaluate(field, r, grid, radial=radial)

    def poles_of(self, field: Field, r: np.ndarray) -> np.ndarray:
        """The field at both poles, Cartesian components."""
        return _poles_of(field, r)

    def on_grid(self, fn: Callable[[np.ndarray, np.ndarray], ArrayLike],
                grid: AngularGrid) -> ArrayLike:
        """fn(theta, phi) on the grid's nodes."""
        return fn(grid.colatitudes[:, None], grid.longitudes[None, :])

    def at_poles(self, fn: Callable[[np.ndarray, np.ndarray], ArrayLike]
                 ) -> ArrayLike:
        """fn at the north then the south pole."""
        return fn(np.array([0.0, np.pi]), np.zeros(2))

    def write_displacement(self, group: Any, model: ModelBase, r: np.ndarray,
                           grid: AngularGrid, rules: tuple[str, str], *,
                           lacks_pole: bool, length: str) -> None:
        """m(X) - X in the spherical frame at X, and at the poles the grid
        lacks in Cartesian components."""
        axes = _axes(group, VECTOR)
        mapping = model.geometry.mapping
        R = spherical_frame(grid.colatitudes[:, None], grid.longitudes[None, :])
        X = r[:, None, None, None] * R[None, ..., :, 0]
        u = np.asarray(mapping(X), dtype=float) - X
        values = np.einsum("tpji,ntpj->ntpi", R, u)
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

    def write_stack(self, root: Any, model: ModelBase,
                    parts: Sequence[tuple[list[int], Any]],
                    interpolation: Interpolation) -> None:
        """The group `geometry`: the radial stretch's boundary reliefs."""
        h = model.geometry.mapping.h
        reliefs, radii = list(h.reliefs), np.asarray(h.boundaries, dtype=float)
        grid, lateral = relief_grid(reliefs, self.outer(parts), interpolation)
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
            poles[k] = np.asarray(rel(np.array([0.0, np.pi]), np.zeros(2)),
                                  dtype=float)
        put_variable(g, "boundary_displacement",
                     ("boundary", "latitude", "longitude"), values,
                     {"long_name": "radial displacement of the surface",
                      "units": length, "lateral_interpolation": lateral,
                      "between": "linear"})
        if not all(grid.pole_nodes):
            g.createDimension("pole", 2)
            put_variable(g, "boundary_displacement_poles", ("boundary", "pole"), poles,
                 {"long_name": "radial displacement at the north and south poles",
                  "units": length})

    # -- reading

    def validate_root(self, ds: Any, path: object) -> None:
        """Nothing beyond the shared statements."""

    def interfaces(self, skeleton: Skeleton, context: Any) -> np.ndarray:
        """The radii of the interfaces of a `Geometry` over the skeleton."""
        from .geometry import Geometry
        faces = Geometry(skeleton, check=False).interfaces
        return np.array([face.radius for face in faces], dtype=float)

    def axes(self, context: Any) -> tuple[str, ...]:
        """(latitude, longitude)."""
        return ("latitude", "longitude")

    def grid_names(self, context: Any) -> tuple[str, ...]:
        """The coordinates and the colatitude weights."""
        return ("latitude", "longitude", "colatitude_weights")

    def validate_grid(self, group: Any, where: str, path: object,
                      context: Any) -> None:
        """`grid_kind` one of `sampling.KINDS`."""
        kind = attribute_text(group, "grid_kind")
        if kind is None:
            _fail(path, f"{where}.grid_kind is missing or not a kind of grid, one of "
                        f"{GRID_KINDS}")
        if kind not in GRID_KINDS:
            _fail(path, f"{where}: grid_kind {kind!r} is not one of {GRID_KINDS}")

    def read_grid(self, group: Any, context: Any) -> AngularGrid:
        """`grid_of_group`."""
        return grid_of_group(group)

    def validate_stack(self, group: Any, path: object, context: Any) -> None:
        """`_validate_stack`."""
        _validate_stack(group, path)

    def read_stack(self, group: Any, context: Any) -> FileGeometry:
        """The surfaces' radii, grid, displacements and polar values."""
        var = group.variables["boundary_displacement"]
        poles = (np.asarray(group.variables["boundary_displacement_poles"][:],
                            dtype=float)
                 if "boundary_displacement_poles" in group.variables else None)
        return FileGeometry(
            np.asarray(group.variables["boundary_radius"][:], dtype=float),
            grid_of_group(group), np.asarray(var[:], dtype=float), poles,
            str(var.getncattr("lateral_interpolation")),
            str(var.getncattr("between")))

    def radial_field(self, interval: tuple[float, float], function: Any, *,
                     character: Character, name: str | None) -> RadialField:
        """A `RadialField`."""
        return RadialField(interval, function, character=character, name=name)

    def gridded_field(self, interval: tuple[float, float], breaks: np.ndarray,
                      grid: AngularGrid, values: np.ndarray,
                      poles: np.ndarray | None, *, character: Character,
                      name: str | None, radial: str, lateral: str) -> Field:
        """A `gridded.GriddedField`, its polar values the stored ones."""
        from .gridded import GriddedField
        return GriddedField(interval, breaks, grid, values, character=character,
                            name=name, radial=radial, lateral=lateral, poles=poles)

    def gridded_function(self, grid: AngularGrid, values: np.ndarray, *,
                         lateral: str, poles: np.ndarray | None) -> GriddedFunction:
        """A `gridded.GriddedFunction`, its polar values the stored ones."""
        from .gridded import GriddedFunction
        return GriddedFunction(grid, values, lateral=lateral, poles=poles)

    def geometry(self, data: ModelFile, interpolation: Interpolation
                 ) -> GeometryBase:
        """A `Geometry`, its mapping a radial stretch of the stack or a
        `GriddedDisplacement`."""
        from .displacement import LayerLinear
        from .geometry import Geometry
        from .gridded import GriddedDisplacement, GriddedFunction
        from .mapping import RadialStretch
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
                    reliefs.append(GriddedFunction(geo.grid, v, lateral=lateral,
                                                   poles=p))
            mapping = RadialStretch(LayerLinear(Skeleton(geo.radii), reliefs),
                                    rmax=float(sk.boundaries[-1]))
        elif any("displacement" in part.variables for part in data.parts):
            mapping = GriddedDisplacement(sk, displacement_fields(
                data, interpolation, self, character=VECTOR))
        return Geometry(sk, mapping=mapping, layer_names=list(data.layer_names),
                        interface_names=list(data.interface_names), rtol=data.rtol)

    def leaf(self) -> type[ModelBase]:
        """`Model`."""
        from .model import Model
        return Model


#: The spherical layout, the one of every file of `SCHEMAS`.
_SPHERICAL = SphericalLayout()


def _layouts() -> Iterator[Layout]:
    """Every layout, the spherical one first and the flat one imported
    where it is first asked for."""
    yield _SPHERICAL
    from .cartesian.modelfile import LAYOUT
    yield LAYOUT


def _layout_of_model(model: ModelBase) -> Layout:
    """The layout of the model's coordinate system, refused by name where
    none is."""
    system = model.coordinates.name
    for layout in _layouts():
        if layout.system == system:
            return layout
    raise ValueError(f"a model in the coordinates {system!r} has no model file")


def _listed(items: Sequence[str]) -> str:
    """'a', 'a and b', 'a, b and c', each quoted."""
    quoted = [repr(x) for x in items]
    return quoted[0] if len(quoted) == 1 else (
        f"{', '.join(quoted[:-1])} and {quoted[-1]}")


def _layout_of_file(ds: Any, path: object, layouts: Iterable[Layout]) -> Layout:
    """The layout among `layouts` whose schemas hold the file's, or the
    error naming the schemas they read and, for a file of a schema they do
    not read, the reader that does."""
    if "schema" not in ds.ncattrs():
        _fail(path, "no schema attribute: not a planetmodel model file")
    schema = ds.getncattr("schema")
    text = not isinstance(schema, np.ndarray)
    seen: list[Layout] = []
    for layout in layouts:
        if text and schema in layout.schemas:
            return layout
        seen.append(layout)
    known = [s for layout in seen for s in layout.schemas]
    other = (f"; a file of {schema!r} is read by {_READERS[schema]}"
             if text and schema in _READERS else "")
    _fail(path, f"schema is {schema!r}, this reader reads {_listed(known)}{other}")


# -- writing --------------------------------------------------------------------

def _parts_of(model: ModelBase, grid: Grids, layout: Layout
              ) -> list[tuple[list[int], Any]]:
    """The runs of contiguous layers on one grid, or the error."""
    n = model.nlayers
    if not isinstance(grid, Mapping):
        per = [grid] * n
    else:
        per = [None] * n
        for key, g in grid.items():
            per[model.geometry.layer(key).index] = g
        missing = [model.layers[i].name or str(i) for i in range(n) if per[i] is None]
        if missing:
            raise ValueError(f"no grid given for layers {missing}")
    parts: list[tuple[list[int], Any]] = []
    for i, g in enumerate(per):
        layout.check_grid(model, g, i)
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


def _pieces(field: object, rule: str) -> tuple[np.ndarray, int] | None:
    """The pieces a field declares that hold it exactly by a radial rule:
    its logarithm's under a rule in the logarithm, its own otherwise."""
    method = "log_radial_pieces" if rule in LOG_RULES else "radial_pieces"
    return getattr(field, method, lambda: None)()


def _held_pieces(field: Field, name: str, rules: Mapping[str, tuple[str, str]]
                 ) -> tuple[np.ndarray, int] | None:
    """The pieces that hold one layer's field by the rules its variables
    state: a reference and its perturbation written apart each by its
    own variable's, both or neither."""
    split = f"{name}_perturbation"
    if split in rules and getattr(field, "reference", None) is not None:
        parts = [_pieces(field.reference, rules[name][0]),
                 _pieces(field.perturbation, rules[split][0])]
        if any(p is None for p in parts):
            return None
        return (np.unique(np.concatenate([p[0] for p in parts])),
                max(p[1] for p in parts))
    return _pieces(field, rules[name][0])


def _part_mesh(model: ModelBase, layers: list[int], grid: Any,
               names: Sequence[str], form: str,
               rules: Mapping[str, tuple[str, str]], layout: Layout) -> RadialMesh:
    """The exact radial layout of a part, by the rule of the module
    docstring, each field's pieces those its variables' `rules` hold."""
    b = model.skeleton.boundaries
    lo, hi = float(b[layers[0]]), float(b[layers[-1] + 1])
    edges, degree = [b[layers[0]:layers[-1] + 2]], 1
    inexact = set(layers) if form == "field" else set()
    for i in layers:
        layer = model.layer(i)
        for name in names:
            if name not in layer:
                continue
            pieces = _held_pieces(layer[name], name, rules)
            if pieces is None:
                inexact.add(i)
                continue
            edges.append(pieces[0])
            degree = max(degree, pieces[1])
    knots = np.asarray(model.geometry.knots, dtype=float)
    edges.append(knots[(knots > lo) & (knots < hi)])
    tol = model.geometry.rtol * model.skeleton.span
    pins = np.unique(np.concatenate(edges))
    pins = pins[np.concatenate([[True], np.diff(pins) > tol])]
    pins[0], pins[-1] = lo, hi
    ngll = degree + 1
    if inexact:
        drmax = layout.inexact_step(model, grid)
        ngll = max(ngll, 5)
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


def real_values(values: ArrayLike, field: Field) -> np.ndarray:
    """A field's values as float64, a complex field refused: the file
    holds reals."""
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
            return real_values(field.evaluate(r, None, None), field)
        return real_values(field.evaluate(r, 0.5 * np.pi, 0.0), field)
    on_grid = getattr(field, "on_grid", None)
    if on_grid is not None:
        return real_values(on_grid(r, grid), field)
    return real_values(field.evaluate(r[:, None, None],
                                      grid.colatitudes[None, :, None],
                                      grid.longitudes[None, None, :]), field)


def _poles_of(field: Field, r: np.ndarray) -> np.ndarray:
    """A field at both poles at the radii, Cartesian: (n, 2) + c."""
    return real_values(field.evaluate(r[:, None], np.array([0.0, np.pi]),
                                      np.zeros(2), frame="cartesian"), field)


def _stated(fields: Sequence[object], name: str, interpolation: Interpolation,
            default: tuple[str, str], grid: Any, layout: Layout) -> tuple[str, str]:
    """The rules a variable states: the caller's, else the one its fields
    agree on (a field of the radius alone has no lateral rule to state),
    else the default.  Fields that state different rules are refused by
    name.  A field's spectral rule (harmonic on the sphere) is stated only
    on a grid whose transforms reproduce it (on the sphere a shipped kind
    of band at least its own; `Layout.spectral_holds`), and the default
    lateral rule otherwise; the caller's rule must be one the grid can
    carry, and a rule in the logarithm one of a variable of rank 0 of the
    radius alone."""
    if name in interpolation:
        radial, lateral = interpolation[name]
        if radial not in RADIAL_RULES or lateral not in layout.methods:
            raise ValueError(
                f"{name!r}: the rules must be one of {RADIAL_RULES} and one of "
                f"{layout.methods}, got ({radial!r}, {lateral!r})")
        if radial in LOG_RULES and not all(
                getattr(f, "is_radial", False)
                and getattr(getattr(f, "character", None), "rank", None) == 0
                for f in fields):
            raise ValueError(
                f"{name!r}: {radial} reads the logarithm of a number, so a variable "
                "of rank 0 of the radius alone states it, and this one varies with "
                "direction or has components")
        if not all(getattr(f, "is_radial", False) for f in fields):
            layout.check_rule(grid, lateral)
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
    if lateral == layout.spectral and not layout.spectral_holds(grid, fields):
        lateral = default[1]
    return radial, lateral


def _asks_log(field: object) -> bool:
    """Whether a rule in the logarithm alone holds a field exactly: its
    logarithm declares pieces and it declares none of its own."""
    return (_pieces(field, LOG_RULES[0]) is not None
            and _pieces(field, RADIAL_METHODS[0]) is None)


def _asks_values(field: object) -> bool:
    """Whether a rule in the values alone holds a field exactly: it states
    one, or declares pieces of its own and its logarithm none."""
    if getattr(field, "radial", None) is not None:
        return True
    return (_pieces(field, RADIAL_METHODS[0]) is not None
            and _pieces(field, LOG_RULES[0]) is None)


def _field_rules(fields: Mapping[int, object], name: str, interpolation: Interpolation,
                 grid: Any, layout: Layout) -> tuple[str, str]:
    """The rules a variable of the fields on its layers states: the
    caller's; else `log-polynomial` where a field asks for a rule in the
    logarithm (`_asks_log`) and none for one in its values
    (`_asks_values`), a field held exactly by either (a positive
    constant) or by neither (no pieces declared) following; else
    `_stated`'s.  A variable whose layers ask for both, or that asks for a
    rule in the logarithm and varies with direction, is refused by name."""
    rules = _stated(list(fields.values()), name, interpolation, DEFAULT_INTERPOLATION,
                    grid, layout)
    if name in interpolation:
        return rules
    logs = sorted(i for i, f in fields.items() if _asks_log(f))
    if not logs:
        return rules
    values = sorted(i for i, f in fields.items() if _asks_values(f))
    varying = sorted(i for i, f in fields.items() if not getattr(f, "is_radial", False))
    if values or varying:
        why = (f"is held exactly by a rule in its values on layers {values}" if values
               else f"varies with direction on layers {varying}, where a rule in the "
                    "logarithm is stated by a variable of the radius alone")
        raise ValueError(
            f"{name!r} is the exponential of a piecewise polynomial on layers {logs}, "
            f"held exactly by a rule in the logarithm, and {why}; one variable states "
            f"one radial rule: give interpolation={{{name!r}: (radial, lateral)}} to "
            "read every layer by one, approximately where it does not hold the "
            "field, or put the layers in parts of their own")
    return LOG_RULES[0], rules[1]


def _is_split(fields: Mapping[int, Field]) -> bool:
    """Whether a variable's fields are written as a radial reference and a
    perturbation: every layer perturbed, by one kind, or radial."""
    kinds = {getattr(f, "kind", None) for f in fields.values()
             if getattr(f, "reference", None) is not None}
    return len(kinds) == 1 and all(
        getattr(f, "reference", None) is not None or getattr(f, "is_radial", False)
        for f in fields.values())


def _in_values(interpolation: Interpolation, name: str) -> Interpolation:
    """`interpolation` with the radial rule asked for `name` taken in the
    values, a rule in the logarithm by the rule it names."""
    if name not in interpolation:
        return interpolation
    radial, lateral = interpolation[name]
    return {**interpolation, name: (str(radial).removeprefix("log-"), lateral)}


def _part_rules(model: ModelBase, layers: list[int], grid: Any,
                names: Sequence[str], interpolation: Interpolation, layout: Layout
                ) -> dict[str, tuple[str, str]]:
    """The rules of every variable a part holds, by `_field_rules`: a
    field's under its name, and a field written as a reference and a
    perturbation, the reference's under its name and the perturbation's
    under `<name>_perturbation`.  A rule in the logarithm the caller asks
    for such a field is its reference's; the perturbation, of either sign,
    is read by the rule it names in its values."""
    out: dict[str, tuple[str, str]] = {}
    for name in names:
        fields = {i: model.layer(i)[name] for i in layers if name in model.layer(i)}
        if not fields:
            continue
        if not _is_split(fields):
            out[name] = _field_rules(fields, name, interpolation, grid, layout)
            continue
        references = {i: f if getattr(f, "reference", None) is None else f.reference
                      for i, f in fields.items()}
        perturbations = {i: f.perturbation for i, f in fields.items()
                         if getattr(f, "reference", None) is not None}
        out[name] = _field_rules(references, name, interpolation, grid, layout)
        out[f"{name}_perturbation"] = _field_rules(
            perturbations, name, _in_values(interpolation, name), grid, layout)
    return out


def _check_positive(values: np.ndarray, r: np.ndarray, name: str, layer: int,
                    rule: str) -> None:
    """Refuse a layer's values that a rule in the logarithm cannot read,
    naming the first that is not positive and finite."""
    if rule not in LOG_RULES:
        return
    bad = ~(np.isfinite(values) & (values > 0.0))
    if bad.any():
        j = np.unravel_index(int(np.argmax(bad)), bad.shape)
        raise ValueError(
            f"{name!r} is {values[j]:g} at r = {r[j[0]]:g} on layer {layer}; a "
            f"variable read by {rule}, its logarithm interpolated, is positive and "
            "finite at every node of a layer holding it")


def _attributes(name: str, model: ModelBase, character: Character, layers: list[int],
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


def _write_part(group: Any, model: ModelBase, index: int, layers: list[int],
                grid: Any, mesh: RadialMesh, names: Sequence[str],
                rules: Mapping[str, tuple[str, str]],
                interpolation: Interpolation, form: str, layout: Layout) -> None:
    """One part's layout, grid and variables, each stating its `rules`."""
    group.createDimension("node", mesh.nspec * mesh.ngll)
    group.createDimension("element", mesh.nspec)
    group.createDimension("bound", 2)
    layout.grid_dimensions(group, grid)
    group.setncattr("ngll", np.int32(mesh.ngll))
    group.setncattr("layers", np.asarray(layers, dtype=np.int32))
    layout.grid_attributes(group, grid)
    length = scaled_unit(Dimensions(length=1), model.scales)
    r = mesh.r.ravel()
    node_layer = np.repeat(mesh.layer, mesh.ngll)
    # the last node of each element: where a field jumps at an element
    # edge inside a layer, it holds the value from inside its element
    outer_end = np.arange(r.size) % mesh.ngll == mesh.ngll - 1
    put_variable(group, layout.coordinate, ("node",), r, layout.node_attributes(length))
    var = group.createVariable("element_layer", "i4", ("element",))
    var[...] = mesh.layer
    var.long_name = "skeleton layer of the element"
    put_variable(group, "element_bounds", ("element", "bound"),
         np.stack([mesh.left, mesh.right], axis=-1),
         {"long_name": layout.element_meaning, "units": length})
    layout.grid_variables(group, grid, length)
    lacks_pole = layout.lacks_pole(grid)
    if lacks_pole:
        group.createDimension("pole", 2)
        group.setncattr("poles", "north south")
    lateral_dims, lateral_shape = layout.lateral_dims(grid), layout.lateral_shape(grid)

    for name in names:
        holders = [i for i in layers if name in model.layer(i)]
        if not holders:
            continue
        fields = {i: model.layer(i)[name] for i in holders}
        char = next(iter(fields.values())).character
        axes = _axes(group, char)
        c = stored_shape(char)
        if _is_split(fields):
            kind = next(getattr(f, "kind", None) for f in fields.values()
                        if getattr(f, "reference", None) is not None)
            _write_perturbed(group, model, name, fields, mesh, node_layer, grid,
                             rules[name], rules[f"{name}_perturbation"], kind,
                             lacks_pole, layout)
            continue
        radial = all(getattr(f, "is_radial", False) for f in fields.values())
        shape = (r.size,) + (() if radial else lateral_shape) + c
        values = np.full(shape, np.nan)
        poles = None if radial or not lacks_pole else np.full((r.size, 2) + c, np.nan)
        for i, f in fields.items():
            m = node_layer == i
            values[m] = layout.evaluate(f, r[m], grid, radial=radial)
            if radial:
                values[m] = from_inside(f, r[m], values[m], outer_end[m])
            _check_positive(values[m], r[m], name, i, rules[name][0])
            if poles is not None:
                poles[m] = layout.poles_of(f, r[m])
        dims = ("node",) if radial else ("node",) + lateral_dims
        attrs = _attributes(name, model, char, holders, rules[name], lateral=not radial)
        attrs["coordinates"] = layout.coordinate
        if not radial:
            _mark_radial_layers(attrs, fields)
        put_variable(group, name, dims + axes, values, attrs)
        if poles is not None:
            put_variable(group, f"{name}_poles", ("node", "pole") + axes, poles,
                 {"long_name": f"{name} at the north and south poles, Cartesian "
                               "components", "units": attrs["units"]})
    if form == "field":
        stated = _stated([model.geometry.mapping], "displacement", interpolation,
                         _DISPLACEMENT_INTERPOLATION, grid, layout)
        layout.write_displacement(group, model, r, grid, stated,
                                  lacks_pole=lacks_pole, length=length)


def _write_perturbed(group: Any, model: ModelBase, name: str,
                     fields: Mapping[int, Field], mesh: RadialMesh,
                     node_layer: np.ndarray, grid: Any,
                     ref_rules: tuple[str, str], rules: tuple[str, str], kind: str,
                     lacks_pole: bool, layout: Layout) -> None:
    """A field held as a radial reference with a perturbation: two
    variables, stating `ref_rules` and `rules`."""
    r = mesh.r.ravel()
    outer_end = np.arange(r.size) % mesh.ngll == mesh.ngll - 1
    any_field = next(iter(fields.values()))
    char = any_field.character
    pert = [getattr(f, "perturbation", None) for f in fields.values()]
    pert_radial = all(p is None or getattr(p, "is_radial", False) for p in pert)
    ref = np.full((r.size,), np.nan)
    dp_shape = (r.size,) + (() if pert_radial else layout.lateral_shape(grid))
    dp = np.full(dp_shape, np.nan)
    poles = None if pert_radial or not lacks_pole else np.full((r.size, 2), np.nan)
    pert_char = None
    for i, f in fields.items():
        m = node_layer == i
        reference = getattr(f, "reference", None)
        source = f if reference is None else reference
        ref[m] = layout.evaluate(source, r[m], grid, radial=True)
        ref[m] = from_inside(source, r[m], ref[m], outer_end[m])
        _check_positive(ref[m], r[m], name, i, ref_rules[0])
        if reference is None:
            continue
        dp[m] = layout.evaluate(f.perturbation, r[m], grid, radial=pert_radial)
        _check_positive(dp[m], r[m], f"{name}_perturbation", i, rules[0])
        if poles is not None:
            poles[m] = layout.poles_of(f.perturbation, r[m])
        pert_char = f.perturbation.character
    holders = sorted(fields)
    attrs = _attributes(name, model, char, holders, ref_rules, lateral=False)
    attrs.update({"perturbation": f"{name}_perturbation", "perturbation_kind": kind,
                  "coordinates": layout.coordinate})
    put_variable(group, name, ("node",), ref, attrs)
    perturbed = [i for i, f in fields.items()
                 if getattr(f, "reference", None) is not None]
    pattrs = _attributes(name, model, pert_char, perturbed, rules,
                         lateral=not pert_radial)
    pattrs["long_name"] = f"{kind} perturbation of {name}"
    if kind != "absolute":
        pattrs["units"] = "1"
        pattrs["dimensions"] = np.zeros(len(BASE), dtype=np.int32)
    pattrs["coordinates"] = layout.coordinate
    if not pert_radial:
        _mark_radial_layers(pattrs, {i: f.perturbation for i, f in fields.items()
                                     if getattr(f, "reference", None) is not None})
    dims = ("node",) if pert_radial else ("node",) + layout.lateral_dims(grid)
    put_variable(group, f"{name}_perturbation", dims, dp, pattrs)
    if poles is not None:
        put_variable(group, f"{name}_perturbation_poles", ("node", "pole"), poles,
             {"long_name": f"{kind} perturbation of {name} at the poles",
              "units": pattrs["units"]})


def _surface_holders(model: ModelBase, surfaces: Surfaces, written: Sequence[str],
                     layout: Layout) -> dict[str, dict[int, Relief]]:
    """Each surface field's functions by interface index, increasing: a
    name the format keeps or one written as a layer field, an interface
    given twice, and a name on no interface are refused."""
    if isinstance(surfaces, str) or not isinstance(surfaces, Mapping):
        raise TypeError("surfaces maps names to {interface: function of "
                        f"{layout.arguments}}}")
    out: dict[str, dict[int, Relief]] = {}
    for name, on in surfaces.items():
        if not isinstance(name, str) or not name:
            raise TypeError(f"a surface field's name is a string, got {name!r}")
        if name in layout.reserved or name.endswith(("_perturbation", "_poles")):
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
                                f"of {layout.arguments}")
            held[face.index] = fn
        if not held:
            raise ValueError(f"{name!r} is held on no interface")
        out[name] = dict(sorted(held.items()))
    return out


def _surface_grid(held: Mapping[str, Mapping[int, Relief]]) -> Any:
    """The grid the functions share, each held on it; None where they
    share none."""
    grids = [getattr(fn, "grid", None) for on in held.values() for fn in on.values()]
    if all(g is not None for g in grids) and all(grids[0] == g for g in grids):
        return grids[0]
    return None


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


def _write_surfaces(root: Any, model: ModelBase,
                    held: Mapping[str, Mapping[int, Relief]], grid: Any,
                    interpolation: Interpolation, layout: Layout) -> None:
    """The group `surfaces`: each name's functions on its interfaces."""
    faces = sorted({k for on in held.values() for k in on})
    g = root.createGroup("surfaces")
    g.createDimension("surface", len(faces))
    layout.grid_dimensions(g, grid)
    layout.grid_attributes(g, grid)
    var = g.createVariable("surface_interface", "i4", ("surface",))
    var[...] = np.asarray(faces, dtype=np.int32)
    var.long_name = "the interface the surface is, by its index at the root"
    layout.grid_variables(g, grid, scaled_unit(Dimensions(length=1), model.scales))
    lacks_pole = layout.lacks_pole(grid)
    if lacks_pole:
        g.createDimension("pole", 2)
        g.setncattr("poles", "north south")
    row = {k: i for i, k in enumerate(faces)}
    shape = layout.lateral_shape(grid)
    for name, on in held.items():
        values = np.full((len(faces),) + shape, np.nan)
        poles = np.full((len(faces), 2), np.nan)
        for k, fn in on.items():
            values[row[k]] = _on_angles(layout.on_grid(fn, grid), shape, name, k)
            at = layout.at_poles(fn)
            if at is not None:
                poles[row[k]] = _on_angles(at, (2,), name, k)
        rules = _stated(list(on.values()), name, interpolation, DEFAULT_INTERPOLATION,
                        grid, layout)
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
        put_variable(g, name, ("surface",) + layout.lateral_dims(grid), values, attrs)
        if lacks_pole:
            put_variable(g, f"{name}_poles", ("surface", "pole"), poles,
                 {"long_name": f"{name} at the north and south poles",
                  "units": attrs["units"]})


def radial_grid(model: ModelBase, *, fields: Iterable[str] | None = None
                ) -> LateralGrid:
    """The grid a model given none is written on: the grid of one node,
    `gauss_legendre(0)` on the sphere and `cartesian.Grid(domain, 1)` over
    a flat model's domain, for a model whose `fields` (every name it
    holds by default) are all of the layered coordinate alone and whose
    mapping is the identity.  Any other model is refused, naming the
    first field that is not, or the mapping."""
    layout = _layout_of_model(model)
    names = model.field_names() if fields is None else tuple(fields)
    only = (f"only a model whose fields are all of the {layout.coordinate} alone, "
            "under the identity, is written with no grid given, on the grid of one "
            "node; give the grid to hold it on")
    for layer in model.layers:
        for name in names:
            if name in layer and not getattr(layer[name], "is_radial", False):
                raise ValueError(
                    f"{name!r} on layer {layer.index} ({layer.name!r}) depends on "
                    f"{layout.arguments}: {only}")
    if not model.geometry.is_identity:
        raise ValueError(f"the model is placed by {model.geometry.mapping!r}: {only}")
    return layout.radial_grid(model)


def write_model(path: str | os.PathLike[str], model: ModelBase, *,
                grid: Grids | None = None,
                radial: RadialMesh | None = None,
                fields: Iterable[str] | None = None,
                interpolation: Interpolation | None = None,
                meta: Mapping[str, Any] | None = None,
                surfaces: Surfaces | None = None,
                surface_grid: LateralGrid | None = None,
                source: str | None = None) -> Path:
    """Write `model` to a model file at `path`, which is returned.

    A spherical model is written as the module docstring says, and a flat
    one (`planetmodel.cartesian`) as `cartesian.modelfile` says, the
    arguments the same: the layout is the model's coordinate system's.
    `grid` is one `AngularGrid` for every layer (a `cartesian.Grid` over
    the domain for a flat model), or a mapping from layers
    (by index or name) to grids, every layer given one: contiguous layers
    on one grid form one part, and a grid's layers that are not
    contiguous are refused.  A radial model needs none: where every field
    written is of the radius alone (of depth alone in a flat model) and
    the mapping is the identity, the file is the one written on the grid
    of one node, `gauss_legendre(0)` (`cartesian.Grid(domain, 1)` over a
    flat model's domain), one part holding every field as `name(node)`,
    a field that declares no piecewise polynomial laid out as that grid
    asks, on elements a tenth of the outer radius long at most (the
    shorter side of a flat domain's extent), unless `radial` lays the
    nodes out; any other model given no grid is refused, naming the
    field that varies laterally or the mapping (`radial_grid`), as are
    surface fields held on no grid of their own and given no
    `surface_grid`.  `radial` is a `RadialMesh` over the model's
    skeleton to lay the nodes out by, in place of the exact layout of the
    module docstring.  `fields` are the names to write, every name the
    model holds by default; `interpolation` maps names (and
    `displacement`, `boundary_displacement`) to the (radial, lateral)
    rules to state in place of the fields' own, the radial one of
    `RADIAL_RULES` (a rule in the logarithm asked for a field written as
    a reference and a perturbation is the reference's, the perturbation
    read by the rule it names in its values); `meta` becomes
    `meta_<key>` attributes.  `surfaces` maps names to functions of
    (theta, phi) on interfaces (of (x, y) on a flat model's), by index or
    name, written to the group `surfaces` on `surface_grid`: by default
    the grid the functions are held on where they share one, else the
    outermost part's (the top part's of a flat model).  A surface
    field's dimensions and meaning are the model's spec of its name, and
    `interpolation` names its lateral rule as for a field.  `source`
    names the writing code and version in the attribute of that name,
    this library and its version unless given (`stamped_source`).  The
    model is
    written in its own units, and the file whole or not at all, with the
    permissions a plain `open(path, "w")` would give it
    (`files.written_whole`); a spherical file is of schema
    `SCHEMA_WITH_LOG_RULES` where a variable states a rule in the
    logarithm, and `SCHEMA` otherwise, and a flat one of `SCHEMA_FLAT`.
    """
    from .model import ModelBase
    if not isinstance(model, ModelBase):
        raise TypeError(f"expected a Model, got {type(model).__name__}")
    layout = _layout_of_model(model)
    nc = netcdf_module()
    interpolation = dict(interpolation or {})
    names = model.field_names() if fields is None else tuple(fields)
    if isinstance(names, str):
        raise TypeError("fields is a sequence of names, not one string")
    known = model.field_names()
    for name in names:
        if name not in known:
            raise KeyError(f"no layer holds a field {name!r}")
        if name in layout.reserved or name.endswith(("_perturbation", "_poles")):
            raise ValueError(f"{name!r} is a name the model file keeps for itself")
    supplied = grid is None
    if supplied:
        grid = radial_grid(model, fields=names)
    parts = _parts_of(model, grid, layout)
    form = layout.mapping_form(model)
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
    source = stamped_source(source)
    if surfaces is None and surface_grid is not None:
        raise ValueError("surface_grid is given with no surfaces to hold on it")
    held = {} if surfaces is None else _surface_holders(model, surfaces, names, layout)
    on = None
    if held:
        on = _surface_grid(held) if surface_grid is None else surface_grid
        if on is None and supplied:
            raise ValueError(
                f"the surface fields {sorted(held)} are functions of "
                f"{layout.arguments}, held on no grid of their own, and no grid is "
                "given to hold them on: give surface_grid=, or grid=")
        if on is None:
            on = layout.outer(parts)
    path = Path(path)
    with written_whole(path) as partial:
        _write_file(nc, partial, model, parts, form, names, radial,
                    interpolation, meta, held, on, source, layout)
    return path


def _write_file(nc: ModuleType, path: Path, model: ModelBase,
                parts: list[tuple[list[int], Any]], form: str,
                names: Sequence[str], radial: RadialMesh | None,
                interpolation: Interpolation, meta: Mapping[str, Any],
                held: Mapping[str, Mapping[int, Relief]],
                surface_grid: Any, source: str, layout: Layout) -> None:
    """The file of `write_model`, at a path of its own until it is whole."""
    from datetime import datetime, timezone

    plans = []
    for layers, g in parts:
        rules = _part_rules(model, layers, g, names, interpolation, layout)
        mesh = (_part_mesh(model, layers, g, names, form, rules, layout)
                if radial is None else _restricted(radial, layers))
        plans.append((layers, g, mesh, rules))
    logged = any(rule[0] in LOG_RULES for *_, rules in plans for rule in rules.values())
    with nc.Dataset(path, "w", format="NETCDF4") as root:
        root.setncattr("schema", layout.schema(logged))
        root.setncattr("Conventions", CONVENTIONS)
        root.setncattr("title", f"planetmodel model {model.name or ''}".strip())
        root.setncattr("source", source)
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
        if model.behaviours:
            root.setncattr("behaviours", " ".join(model.behaviours))
        for key, value in meta.items():
            root.setncattr(f"meta_{key}", value)
        layout.write_root(root, model)
        root.createDimension("layer", model.nlayers)
        root.createDimension("interface", len(model.geometry.interfaces))
        root.createDimension("bound", 2)
        b = model.skeleton.boundaries
        length = scaled_unit(Dimensions(length=1), model.scales)
        put_variable(root, "layer_bounds", ("layer", "bound"),
                     np.stack([b[:-1], b[1:]], -1),
                     {"long_name": layout.bounds_meaning, "units": length})
        var = root.createVariable("layer_name", str, ("layer",))
        for i, layer in enumerate(model.geometry.layers):
            var[i] = layer.name or ""
        faces = model.geometry.interfaces
        put_variable(root, f"interface_{layout.coordinate}", ("interface",),
             np.array([f.radius for f in faces], dtype=float),
             {"long_name": layout.interface_meaning, "units": length})
        var = root.createVariable("interface_name", str, ("interface",))
        for k, face in enumerate(faces):
            var[k] = face.name or ""
        var = root.createVariable("layer_part", "i4", ("layer",))
        part_of = np.empty(model.nlayers, dtype=np.int32)
        for k, (layers, _) in enumerate(parts):
            part_of[layers] = k
        var[...] = part_of
        var.long_name = "the part, part_<k>, holding the layer"
        for k, (layers, g, mesh, rules) in enumerate(plans):
            _write_part(root.createGroup(f"part_{k}"), model, k, layers, g, mesh,
                        names, rules, interpolation, form, layout)
        if form == "stack":
            layout.write_stack(root, model, parts, interpolation)
        if held:
            _write_surfaces(root, model, held, surface_grid, interpolation, layout)


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


def malformed(path: object, msg: str) -> NoReturn:
    """Refuse a malformed model file, naming it and the first thing wrong."""
    raise ValueError(f"malformed model file {path}: {msg}")


_fail = malformed


def attribute_text(holder: Any, key: str) -> str | None:
    """The attribute `key` of a netCDF group or variable where it is text,
    None where it is absent or holds anything else."""
    if key not in holder.ncattrs():
        return None
    value = holder.getncattr(key)
    return value if isinstance(value, str) else None


def attribute_number(holder: Any, key: str) -> float | None:
    """The attribute `key` of a netCDF group or variable where it is one
    finite number, None where it is absent or holds anything else."""
    if key not in holder.ncattrs():
        return None
    value = np.asarray(holder.getncattr(key))
    if value.size != 1 or value.dtype.kind not in "iuf":
        return None
    number = float(value.item())
    return number if np.isfinite(number) else None


def _group_grid(group: Any, where: str, path: object, layout: Layout,
                context: Any) -> Any:
    """The grid a group's statements hold, by the layout; a group whose
    statements make no grid is refused."""
    try:
        return layout.read_grid(group, context)
    except (TypeError, ValueError) as error:
        _fail(path, f"{where}: the grid is malformed: {error}")


def _check_carried(grid: Any, rule: str, where: str, path: object,
                   layout: Layout) -> None:
    """Refuse the variable `where` stating a lateral rule the grid of its
    group cannot carry (`Layout.check_rule`)."""
    try:
        layout.check_rule(grid, rule)
    except ValueError as error:
        _fail(path, f"{where} states the lateral rule {rule!r}, and {error}")


def validate_structure(source: str | os.PathLike[str] | Any) -> None:
    """Check that a file, by path or as an open netCDF4 Dataset, has the
    structure its schema defines (the module docstring's for a spherical
    file, `cartesian.modelfile`'s for a flat one), raising ValueError
    naming the first thing wrong."""
    if isinstance(source, (str, os.PathLike)):
        nc = netcdf_module()
        with nc.Dataset(source, "r") as ds:
            _validate(ds, source, _layouts())
    else:
        _validate(source, getattr(source, "filepath", lambda: "<dataset>")(),
                  _layouts())


def schema_of(path: str | os.PathLike[str]) -> str:
    """The schema a model file declares, its global attribute `schema`,
    read without the rest of the file; ValueError where it declares none."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        if "schema" not in ds.ncattrs():
            _fail(path, "no schema attribute: not a planetmodel model file")
        return str(ds.getncattr("schema"))


def _validate(ds: Any, path: object, layouts: Iterable[Layout]) -> tuple[Layout, Any]:
    """The checks of `validate_structure` on an open dataset, by the layout
    among `layouts` whose schema it states; that layout, and the context
    of its reading."""
    layout = _layout_of_file(ds, path, layouts)
    attrs = set(ds.ncattrs())
    schema = ds.getncattr("schema")
    for base in ("length", "mass", "time", "temperature"):
        key = f"scale_{base}"
        scale = attribute_number(ds, key)
        if scale is None or not scale > 0.0:
            _fail(path, f"{key} is missing or not positive and finite")
    if "rtol" in attrs:
        rtol = ds.getncattr("rtol")
        if (isinstance(rtol, str) or np.size(rtol) != 1
                or not 0.0 < float(rtol) < 1.0):
            _fail(path, f"rtol is {rtol!r}, not a tolerance in (0, 1)")
    if "behaviours" in attrs:
        named = ds.getncattr("behaviours")
        if not isinstance(named, str) or len(set(named.split())) != len(
                named.split()):
            _fail(path, f"behaviours is {named!r}, not the names of the model's "
                        "behaviours, each once, separated by spaces")
    context = layout.validate_root(ds, path)
    for name, dims in (("layer_bounds", ("layer", "bound")),
                       ("layer_name", ("layer",)),
                       (f"interface_{layout.coordinate}", ("interface",)),
                       ("interface_name", ("interface",)),
                       ("layer_part", ("layer",))):
        if name not in ds.variables:
            _fail(path, f"no variable {name!r} at the root")
        if tuple(ds.variables[name].dimensions) != dims:
            _fail(path, f"{name} has dimensions {ds.variables[name].dimensions}, "
                        f"not {dims}")
    bounds = np.asarray(ds.variables["layer_bounds"][:], dtype=float)
    if (not bounds.size or not np.all(bounds[:, 1] > bounds[:, 0])
            or not np.allclose(bounds[1:, 0], bounds[:-1, 1], rtol=0, atol=0)):
        _fail(path, "layer_bounds do not make a skeleton of contiguous layers")
    edges = np.append(bounds[:, 0], bounds[-1, 1])
    stated = f"interface_{layout.coordinate}"
    try:
        faces = layout.interfaces(Skeleton(edges), context)
    except ValueError as error:
        _fail(path, f"layer_bounds do not make a skeleton: {error}")
    held = variable_values(ds.variables[stated])
    if held.shape != faces.shape or not np.allclose(
            held, faces, rtol=0, atol=1e-12 * (edges[-1] - edges[0])):
        _fail(path, f"{stated} does not hold the {faces.size} interfaces of the "
                    "skeleton that layer_bounds make")
    layer_part = np.asarray(ds.variables["layer_part"][:], dtype=int)
    for k in np.unique(layer_part):
        if f"part_{k}" not in ds.groups:
            _fail(path, f"layer_part names part_{k}, which the file lacks")
        layers = np.flatnonzero(layer_part == k)
        if not np.array_equal(layers, np.arange(layers[0], layers[-1] + 1)):
            _fail(path, f"the layers of part {k} are not contiguous")
    logged = []
    for k in np.unique(layer_part):
        layers = np.flatnonzero(layer_part == k)
        logged += _validate_part(ds.groups[f"part_{k}"], f"part_{k}", layers, bounds,
                                 path, layout, context)
    layout.check_schema(schema, logged, path)
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
        layout.validate_stack(ds.groups["geometry"], path, context)
    if "surfaces" in ds.groups:
        taken = {name for k in np.unique(layer_part)
                 for name in ds.groups[f"part_{k}"].variables}
        _validate_surfaces(ds.groups["surfaces"], path, len(ds.dimensions["interface"]),
                           taken, layout, context)
    return layout, context


def _validate_part(g: Any, where: str, layers: np.ndarray, bounds: np.ndarray,
                   path: object, layout: Layout, context: Any) -> list[str]:
    """The checks on one part; the variables stating a rule in the
    logarithm, as `part_<k>/<name>`."""
    logged: list[str] = []
    axes = layout.axes(context)
    for dim in ("node", "element", "bound") + axes:
        if dim not in g.dimensions:
            _fail(path, f"{where} has no dimension {dim!r}")
    for name in (layout.coordinate, "element_layer", "element_bounds") + axes:
        if name not in g.variables:
            _fail(path, f"{where} has no variable {name!r}")
    if not set(layout.part_attributes) <= set(g.ncattrs()):
        _fail(path, f"{where} lacks the attribute "
                    f"{' or '.join(layout.part_attributes)}")
    layout.validate_grid(g, where, path, context)
    grid = _group_grid(g, where, path, layout, context)
    count = attribute_number(g, "ngll")
    if count is None or not count.is_integer():
        _fail(path, f"{where}: ngll is {g.getncattr('ngll')!r}, not a whole number")
    ngll = int(count)
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
    inside = ((eb[:, 0] >= bounds[el, 0] - 1e-12 * span)
              & (eb[:, 1] <= bounds[el, 1] + 1e-12 * span))
    if not np.all(inside):
        e = int(np.flatnonzero(~inside)[0])
        _fail(path, f"{where}/element_layer gives element {e} to layer {el[e]}, "
                    "which its bounds do not lie in")
    lacks_pole = layout.lacks_pole(grid)
    own = {layout.coordinate, "element_layer", "element_bounds",
           *layout.grid_names(context)}
    for name, var in g.variables.items():
        if name in own or name.endswith("_poles"):
            continue
        a = set(var.ncattrs())
        dims = tuple(var.dimensions)
        if dims[:1] != ("node",):
            _fail(path, f"{where}/{name} does not run over the nodes")
        lateral = dims[1:1 + len(axes)] == axes
        need = ({"radial_interpolation"} if name == "displacement"
                else {"radial_interpolation", "rank", "weight", "voigt"})
        if not need <= a:
            _fail(path, f"{where}/{name} lacks the attributes "
                        f"{sorted(need - a)}")
        if name == "displacement":
            want: tuple[int, ...] = layout.displacement_shape
        else:
            try:
                want = stored_shape(_character(variable_attributes(var)))
            except (ValueError, TypeError) as error:
                _fail(path, f"{where}/{name} states no character: {error}")
        tail = var.shape[1 + len(axes) if lateral else 1:]
        if tuple(tail) != tuple(want):
            _fail(path, f"{where}/{name} has trailing axes {tuple(tail)}, but its "
                        f"character stores {tuple(want)}")
        rule = var.getncattr("radial_interpolation")
        if not isinstance(rule, str) or rule not in RADIAL_RULES:
            _fail(path, f"{where}/{name} states an unknown radial rule")
        if rule in LOG_RULES:
            if lateral or want != () or name == "displacement":
                _fail(path, f"{where}/{name} states {rule}, a rule in the logarithm, "
                            "which a variable of rank 0 of the radius alone states")
            v = variable_values(var)
            held = ~np.isnan(v)
            if not np.all(np.isfinite(v[held]) & (v[held] > 0.0)):
                _fail(path, f"{where}/{name} states {rule}, a rule in the logarithm, "
                            "and holds a value that is not positive")
            logged.append(f"{where}/{name}")
        if "radial_layers" in a:
            listed = _radial_layers(variable_attributes(var))
            held = (set(int(i) for i in np.atleast_1d(var.getncattr("layers")))
                    if "layers" in a else set(int(i) for i in layers))
            if not lateral or not listed <= held:
                _fail(path, f"{where}/{name} names radial_layers "
                            f"{sorted(listed)}, which must be layers a lateral "
                            "variable holds")
        if lateral:
            stated = attribute_text(var, "lateral_interpolation")
            if stated not in layout.methods:
                _fail(path, f"{where}/{name} states no known lateral rule")
            _check_carried(grid, stated, f"{where}/{name}", path, layout)
            if lacks_pole:
                if f"{name}_poles" not in g.variables:
                    _fail(path, f"{where}/{name} varies with direction on a grid "
                                f"without its poles, and {name}_poles is missing")
                poles = g.variables[f"{name}_poles"]
                if (tuple(poles.dimensions[:2]) != ("node", "pole")
                        or tuple(poles.shape) != (nnode, 2) + tuple(want)):
                    _fail(path, f"{where}/{name}_poles does not hold both poles "
                                "at every node")
        if "perturbation" in a:
            other = attribute_text(var, "perturbation")
            if other is None or other not in g.variables:
                _fail(path, f"{where}/{name} names its perturbation "
                            f"{var.getncattr('perturbation')!r}, which the part lacks")
            if attribute_text(var, "perturbation_kind") not in PERTURBATION_KINDS:
                _fail(path, f"{where}/{name} states no known perturbation_kind, one "
                            f"of {PERTURBATION_KINDS}")
    return logged


def _validate_stack(g: Any, path: object) -> None:
    """The checks on the geometry group: its variables and their
    dimensions, known rules the grid can carry, radii of two surfaces or
    more that increase strictly, and the polar values of every surface
    where the grid lacks a pole."""
    for name in ("boundary_radius", "boundary_displacement", "latitude",
                 "longitude"):
        if name not in g.variables:
            _fail(path, f"geometry has no variable {name!r}")
    var = g.variables["boundary_displacement"]
    if tuple(var.dimensions) != ("boundary", "latitude", "longitude"):
        _fail(path, "geometry/boundary_displacement has the wrong dimensions")
    poles = g.variables.get("boundary_displacement_poles")
    if poles is not None and tuple(poles.dimensions) != ("boundary", "pole"):
        _fail(path, "geometry/boundary_displacement_poles has the wrong dimensions")
    stated = attribute_text(var, "lateral_interpolation")
    if stated not in LATERAL_METHODS:
        _fail(path, "geometry/boundary_displacement states no known lateral rule")
    if attribute_text(var, "between") != "linear":
        _fail(path, "geometry/boundary_displacement states an unknown rule between "
                    "surfaces")
    try:
        grid = grid_of_group(g)
    except (TypeError, ValueError) as error:
        _fail(path, f"geometry: the grid is malformed: {error}")
    try:
        check_rule(grid, stated)
    except ValueError as error:
        _fail(path, f"geometry/boundary_displacement states the lateral rule "
                    f"{stated!r}, and {error}")
    radii = np.ravel(variable_values(g.variables["boundary_radius"]))
    if not np.all(np.diff(radii) > 0.0):
        _fail(path, "geometry/boundary_radius does not increase strictly")
    if radii.size < 2:
        _fail(path, "geometry/boundary_radius is fewer than two surfaces")
    if not all(grid.pole_nodes) and (
            poles is None or tuple(poles.shape) != (radii.size, 2)):
        _fail(path, "geometry/boundary_displacement varies with direction on a "
                    "grid without its poles, and boundary_displacement_poles is "
                    "missing")


def _validate_surfaces(g: Any, path: object, ninterface: int, taken: set[str],
                       layout: Layout, context: Any) -> None:
    """The checks on the surfaces group; `taken` holds the parts' names."""
    axes = layout.axes(context)
    for dim in ("surface",) + axes:
        if dim not in g.dimensions:
            _fail(path, f"surfaces has no dimension {dim!r}")
    for name in ("surface_interface",) + axes:
        if name not in g.variables:
            _fail(path, f"surfaces has no variable {name!r}")
    layout.validate_grid(g, "surfaces", path, context)
    grid = _group_grid(g, "surfaces", path, layout, context)
    var = g.variables["surface_interface"]
    faces = np.asarray(var[:], dtype=float)
    if (tuple(var.dimensions) != ("surface",) or np.any(faces != np.round(faces))
            or np.any(faces < 0) or np.any(faces >= ninterface)
            or np.any(np.diff(faces) <= 0)):
        _fail(path, "surfaces/surface_interface is not interfaces of the file, "
                    "increasing")
    lacks_pole = layout.lacks_pole(grid)
    own = ("surface_interface",) + layout.grid_names(context)
    for name, v in g.variables.items():
        if name in own or name.endswith("_poles"):
            continue
        where = f"surfaces/{name}"
        if tuple(v.dimensions) != ("surface",) + axes:
            _fail(path, f"{where} does not run over the surfaces and the grid")
        if name in taken:
            _fail(path, f"{where} is a layer field's name too; a name is a layer "
                        "field's or a surface field's, never both")
        a = set(v.ncattrs())
        stated = attribute_text(v, "lateral_interpolation")
        if stated not in layout.methods:
            _fail(path, f"{where} states no known lateral rule")
        _check_carried(grid, stated, where, path, layout)
        if "interfaces" in a and not {float(k) for k in np.atleast_1d(
                v.getncattr("interfaces"))} <= set(faces.tolist()):
            _fail(path, f"{where} names interfaces that are not surfaces of the group")
        poles = g.variables.get(f"{name}_poles")
        if lacks_pole and (poles is None
                           or tuple(poles.dimensions) != ("surface", "pole")
                           or tuple(poles.shape) != (faces.size, 2)):
            _fail(path, f"{where} is on a grid without its poles, and {name}_poles "
                        "is missing")


def read_model_file(path: str | os.PathLike[str]) -> ModelFile:
    """A spherical model file's numbers and metadata, its structure
    validated first; a flat file is refused by its schema, naming its
    reader (`cartesian.modelfile.read_model_file`)."""
    return read_with(path, (_SPHERICAL,))


def read_with(path: str | os.PathLike[str], layouts: Iterable[Layout]) -> ModelFile:
    """A model file's numbers and metadata, read by the layout among
    `layouts` whose schema the file states, its structure validated
    first; a file of any other schema is refused by it, naming the schemas
    `layouts` read and the reader of the file's where there is one."""
    nc = netcdf_module()
    with nc.Dataset(path, "r") as ds:
        layout, context = _validate(ds, path, layouts)
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
            coordinate = layout.coordinate
            stored = np.asarray(g.variables[coordinate][:], dtype=float)
            if not np.allclose(stored, mesh.r.ravel(), rtol=1e-12, atol=0):
                _fail(path, f"part_{k}: {coordinate} is not the GLL nodes of its "
                            "elements")
            own = (coordinate, "element_layer", "element_bounds",
                   *layout.grid_names(context))
            variables = {
                name: FileVariable(variable_values(var), variable_attributes(var))
                for name, var in g.variables.items() if name not in own}
            parts.append(FilePart(layers, mesh, layout.read_grid(g, context),
                                  MappingProxyType(variables)))
        geometry = None
        if "geometry" in ds.groups:
            geometry = layout.read_stack(ds.groups["geometry"], context)
        surfaces = None
        if "surfaces" in ds.groups:
            g = ds.groups["surfaces"]
            faces = tuple(int(k) for k in
                          np.asarray(g.variables["surface_interface"][:]))
            own = ("surface_interface", *layout.grid_names(context))
            variables = {name: FileVariable(variable_values(var),
                                            variable_attributes(var))
                         for name, var in g.variables.items() if name not in own}
            surfaces = FileSurfaces(faces, layout.read_grid(g, context),
                                    MappingProxyType(variables))
    return ModelFile(schema=str(attrs["schema"]), name=attrs.get("name"),
                     scales=scales, constants=MappingProxyType(constants),
                     meta=MappingProxyType(meta), skeleton=skeleton,
                     layer_names=layer_names, interface_names=interface_names,
                     parts=tuple(parts), geometry=geometry,
                     declared=MappingProxyType(declared),
                     rtol=float(attrs.get("rtol", DEFAULT_RTOL)), surfaces=surfaces,
                     citation=tuple(str(attrs.get("citation", "")).splitlines()),
                     behaviours=tuple(str(attrs.get("behaviours", "")).split()),
                     source=str(attrs.get("source", "")),
                     created=str(attrs.get("created", "")), domain=context)


def _layout_of_data(data: ModelFile) -> Layout:
    """The layout of a file read, by its schema."""
    for layout in _layouts():
        if data.schema in layout.schemas:
            return layout
    raise ValueError(f"no layout reads schema {data.schema!r}")


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


def _radial_log_field(interval: tuple[float, float], nodal: np.ndarray, mesh: Mesh1D,
                      method: str, character: Character, name: str | None,
                      layout: Layout) -> RadialField:
    """Nodal values (nelem, ngll) of a rank-0 variable read by a rule in
    the logarithm: the constant where they are one value, else the
    exponential layer y_e exp(P_e) on each element, y_e its first value
    and P_e the rule's polynomial through log(y / y_e) at its nodes."""
    y = np.asarray(nodal, dtype=float)
    if character.rank or y.ndim != 2:
        raise ValueError(f"{name!r}: {method} reads the logarithm of a number, and "
                         f"the field is of rank {character.rank}")
    if not np.all(np.isfinite(y) & (y > 0.0)):
        raise ValueError(f"{name!r} is read by {method}, its logarithm interpolated, "
                         "and holds a value that is not positive and finite")
    if np.all(y == y.flat[0]):
        return layout.radial_field(interval, constant_layer(interval, float(y.flat[0])),
                                   character=character, name=name)
    first = y[:, :1]
    exponent = mesh.to_ppoly(np.log(y / first), method=method.removeprefix("log-"))
    factor = PPoly(first.T.copy(), mesh.breakpoints)
    return layout.radial_field(interval,
                               ExponentialLayer(interval, exponent, factor=factor),
                               character=character, name=name)


def _radial_field(interval: tuple[float, float], nodal: np.ndarray, mesh: Mesh1D,
                  method: str, character: Character, name: str | None,
                  layout: Layout) -> RadialField:
    """Nodal values (nelem, ngll) + c as a field of the layered coordinate
    alone, of exact pieces, read by a rule in the logarithm by
    `_radial_log_field`."""
    if method in LOG_RULES:
        return _radial_log_field(interval, nodal, mesh, method, character, name,
                                 layout)
    pp = mesh.to_ppoly(nodal, method=method)
    c = nodal.shape[2:]
    if not c:
        return layout.radial_field(interval, PolynomialLayer(interval, pp),
                                   character=character, name=name)
    functions = np.empty(c, dtype=object)
    for idx in np.ndindex(c):
        functions[idx] = PolynomialLayer(interval, type(pp)(pp.c[(...,) + idx], pp.x))
    return layout.radial_field(interval, functions, character=character, name=name)


def _variable_field(data: ModelFile, part: FilePart, layer: int, name: str,
                    var: FileVariable, interpolation: Interpolation, layout: Layout,
                    *, character: Character | None = None) -> Field | None:
    """One variable on one layer as a field, or None where it is NaN there."""
    mask, elems, mesh = _layer_mesh(part, layer)
    values = var.values[mask]
    if np.all(np.isnan(values)):
        return None
    interval = tuple(float(x) for x in data.skeleton.boundaries[layer:layer + 2])
    char = _character(var.attributes) if character is None else character
    rules = _rules(name, var.attributes, interpolation)
    c = stored_shape(char)
    radial = var.values.ndim == 1 + len(c)
    if not radial and layer in _radial_layers(var.attributes):
        # every column is the same: the first node of the grid's
        lateral_axes = values.ndim - 1 - len(c)
        values, radial = values[(slice(None),) + (0,) * lateral_axes], True
    if radial:
        nodal = values.reshape((elems.size, mesh.ngll) + values.shape[1:])
        return _radial_field(interval, nodal, mesh, rules[0], char, name, layout)
    if rules[0] in LOG_RULES:
        raise ValueError(f"{name!r} varies with direction on layer {layer}, and "
                         f"{rules[0]} reads a variable of the radius alone")
    lateral = rules[1] or DEFAULT_INTERPOLATION[1]
    poles = part.variables.get(f"{name}_poles")
    v = values.reshape((elems.size, mesh.ngll) + values.shape[1:])
    p = None if poles is None else poles.values[mask].reshape(
        (elems.size, mesh.ngll) + poles.values.shape[1:])
    return layout.gridded_field(interval, mesh.breakpoints, part.grid, v, p,
                                character=char, name=name, radial=rules[0],
                                lateral=lateral)


def file_layers(data: ModelFile, *, interpolation: Interpolation | None = None
                ) -> list[dict[str, Field]]:
    """One mapping of name to field per layer: radial variables as exact
    polynomial layers through the nodes (exponential layers under a rule
    in the logarithm), the others as gridded fields, and references with
    perturbations as perturbed fields, read by the stated rules unless
    `interpolation` names others; the fields of the file's coordinate
    system, by its schema."""
    from .perturbation import PerturbedField
    layout = _layout_of_data(data)
    interpolation = dict(interpolation or {})
    out: list[dict[str, Field]] = [{} for _ in range(data.skeleton.nlayers)]
    for part in data.parts:
        for name, var in part.variables.items():
            if (name.endswith(("_poles", "_perturbation")) or name == "displacement"):
                continue
            pert_name = var.attributes.get("perturbation")
            for layer in part.layers:
                field = _variable_field(data, part, layer, name, var, interpolation,
                                        layout)
                if field is None:
                    continue
                if pert_name is not None:
                    # the perturbation is read by the rules asked for its
                    # field, a rule in the logarithm by the one it names
                    # in the values, or by its own stated ones
                    asked = ({pert_name: _in_values(interpolation, name)[name]}
                             if name in interpolation else {})
                    dp = _variable_field(data, part, layer, pert_name,
                                         part.variables[pert_name], asked, layout)
                    if dp is not None:
                        field = PerturbedField(field, dp.renamed(None),
                                               kind=var.attributes["perturbation_kind"],
                                               name=name)
                out[layer][name] = field
    return out


def file_geometry(data: ModelFile, *, interpolation: Interpolation | None = None
                  ) -> GeometryBase:
    """The geometry a file holds: its skeleton and names, and its mapping
    from the geometry group or the displacement field, read by the stated
    rules unless `interpolation` names others; a spherical `Geometry` or
    a `cartesian.Geometry` over the file's domain, by its schema."""
    return _layout_of_data(data).geometry(data, dict(interpolation or {}))


def displacement_fields(data: ModelFile, interpolation: Interpolation,
                        layout: Layout, *, character: Character) -> list[Field]:
    """The variable `displacement` of every part, a field of `character` on
    each layer, read by its stated rules unless `interpolation` names
    others: the mapping a file holds as a field, layer by layer."""
    fields = []
    for layer in range(data.skeleton.nlayers):
        part = data.part_of(layer)
        var = part.variables["displacement"]
        fields.append(_variable_field(data, part, layer, "displacement", var,
                                      interpolation, layout, character=character))
    return fields


def file_surfaces(data: ModelFile, *, interpolation: Interpolation | None = None
                  ) -> dict[str, dict[int | str, Any]]:
    """The functions on interfaces a file holds, name -> {interface ->
    gridded function} (`GriddedFunction`, or a `cartesian.GriddedFunction`
    in a flat file), an interface keyed by its name, or by its index
    where it has none; each read by its stated lateral rule unless
    `interpolation` names another, and empty for a file without them."""
    layout = _layout_of_data(data)
    interpolation = dict(interpolation or {})
    held = data.surfaces
    if held is None:
        return {}
    out: dict[str, dict[int | str, Any]] = {}
    for name, var in held.variables.items():
        if name.endswith("_poles"):
            continue
        lateral = (interpolation[name][1] if name in interpolation
                   else str(var.attributes["lateral_interpolation"]))
        poles = held.variables.get(f"{name}_poles")
        on: dict[int | str, Any] = {}
        for row, k in enumerate(held.interfaces):
            if np.all(np.isnan(var.values[row])):
                continue
            key = data.interface_names[k]
            on[k if key is None else key] = layout.gridded_function(
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


def _model_type(data: ModelFile, leaf: type[ModelBase], asked: ModelType | None,
                path: object) -> type[ModelBase]:
    """The type of the model a file is read as: the one `asked` for, a
    model type over `leaf` or the behaviours of one; else the behaviours
    of `behaviours.BEHAVIOURS` the file names, over `leaf`, with a warning
    naming those it names that are none of them."""
    from .behaviours import BEHAVIOURS, model_type
    if asked is None:
        known = {b.__name__: b for b in BEHAVIOURS}
        other = [name for name in data.behaviours if name not in known]
        if other:
            s = "s" if len(other) > 1 else ""
            warnings.warn(
                f"{path}: the file names the behaviour{s} {_listed(other)}, which "
                f"this program lacks; the model read keeps the name{s}, to be "
                "written again, and is without the methods: read_model(..., "
                "model_type=) takes a type that has them",
                UserWarning, stacklevel=3)
        return model_type(leaf, [known[name] for name in data.behaviours
                                 if name in known])
    if not isinstance(asked, type):
        return model_type(leaf, asked)
    if not issubclass(asked, leaf):
        raise TypeError(
            f"{path} holds a {leaf.__module__}.{leaf.__name__}, and "
            f"{asked.__module__}.{asked.__qualname__} is not a model type over "
            "it; behaviours are given as a sequence of mixin classes")
    return asked


def read_model(path: str | os.PathLike[str], *,
               interpolation: Interpolation | None = None,
               scales: Scales | None = None,
               model_type: ModelType | None = None) -> ModelBase:
    """The model a file holds: a `Model`, or a `cartesian.Model` from a
    flat file, by what its schema states, with the file's specs, in the
    file's scales or, with `scales=`, converted to those.

    The model has the behaviours the file names (the module docstring's
    `behaviours`): its class is `behaviours.model_type` of those of
    `behaviours.BEHAVIOURS` among them over the leaf, so that `moduli`,
    `gravity`, `frozen` and the rest are called on it as on the model
    written, and a file naming none gives a plain model.  A name that is
    none of those is kept (`Model.behaviours`) and written again by
    `write_model`, and a UserWarning names it.  `model_type` says what to
    give back instead, whatever the file names: a model type of the
    caller's own, or the behaviours of one, an iterable of mixin classes
    (the caller's own among them; empty for a plain model).

    No constructor is called (`model.restored`): the model holds the
    fields the file holds and no others, nothing derived on reading, and
    none of the instance state a type's own constructor sets.
    """
    from .model import restored
    data = read_with(path, _layouts())
    layout = _layout_of_data(data)
    cls = _model_type(data, layout.leaf(), model_type, path)
    model = restored(cls, file_geometry(data, interpolation=interpolation),
                     file_layers(data, interpolation=interpolation),
                     scales=data.scales, specs=_specs(data), name=data.name,
                     constants=data.declared, citation=data.citation,
                     behaviours=data.behaviours if model_type is None else ())
    return model if scales is None else model.converted(scales)


def read_surfaces(path: str | os.PathLike[str], *,
                  interpolation: Interpolation | None = None
                  ) -> dict[str, dict[int | str, Any]]:
    """The functions on interfaces a file holds, as `file_surfaces` gives
    them, a spherical file's or a flat one's by its schema, in the file's
    scales (the model's specs, from `read_model`, give their dimensions);
    empty for a file without them."""
    return file_surfaces(read_with(path, _layouts()), interpolation=interpolation)
