"""Write the fixtures the C++ readers are held to: model files and
interpolation cases, from planetmodel's Python, which defines both.

Run as a script it writes into `fixtures/` beside it; `write_fixtures`
writes into any directory and returns the paths.  The fixtures are
written in CI, not committed.

**Model files** (`model_*.nc`), each with the model's fields at its own
nodes: PREM (radial, one part); PREM with a relative perturbation of
`vsv` over the lower mantle; a two-layer model with a laterally varying
density and a vector field; PREM in three parts on three kinds of grid;
a model placed by boundary reliefs (the `geometry` group); one placed
by a flattening (the `displacement` field); and a two-layer model with
fields on both its interfaces (the `surfaces` group, a shape kernel by
the bilinear rule and a shift by the bicubic one, on a grid without its
poles); and PREM given a viscosity and converted by sl3d's chain from a
perturbation of its shear speeds (`model_converted.nc`, the viscosity a
log perturbation and the model's citation the chain's), and the same
with that viscosity made a Burgers body's two Prony branches and
power-law creep beside them (`model_prony.nc`); and PREM with its
outer core made stable, described by its Bullen excess, zero on the
stretches made neutral, with the density that excess derives
(`model_stratified.nc`); and a two-layer model of the radius alone,
written with no grid given, on the grid of one node, whose type has a
behaviour of its own beside the library's (`model_radial.nc`, its
`behaviours` attribute naming `Tidal` between them).  A Maxwell model
built from a table whose viscosity changes by three decades between
knots, joined in its logarithm, one viscosity through another layer and
none in the core and crust, every field written (`model_log.nc`, of
`planetmodel.model/2`, the viscosity stating `log-polynomial`); the
same with each viscosity made its one Prony branch and tabulated on the
knots, a relaxation time so joined (`model_log_prony.nc`); and the
Maxwell model with that viscosity perturbed in its logarithm by a
gridded field and placed by a flattening (`model_log_lateral.nc`, the
reference stating `log-polynomial`).  The Maxwell model with its layer
of one viscosity made a Burgers body, written with no grid, the branches
functions of the layer's fields held as sampled beside the graded
viscosity (`model_log_burgers.nc`); and the one whose viscosity varies
laterally with that layer a Burgers body, each relaxation time a
reference asked to state `log-polynomial` and a perturbation in its
logarithm (`model_log_burgers_lateral.nc`).  PREM's file again with the
absent values of one variable marked by a fill that is not NaN
(`model_fill.nc`, the `qmu` of its part holding `_FillValue = -999` on
the layers that lack it).

Beside them, `model_kinds.nc` (an absolute perturbation of `rho` and a
log one of `viscosity`), `model_tensors.nc` (a rank-2 `stress`, radial
on one layer and gridded on the other, and a rank-4 `elastic`) and
`model_harmonic.nc` (a copy of the lateral one stating the harmonic
rule, which the C++ refuses); and for every model file but that one
`<model>.expect`, from the Python's own reading of the file, one fact
per line: `scales`, `rtol`, `name`, `citation` (its works a line each),
`behaviours` (the names, a space between them) and the `layer_name` and
`interface_name`s (UTF-8 in hexadecimal, `-` for none), each `constant`
with its dimensions, `value <layer> <name> <s|c> r theta phi` and the
stored components in the spherical or Cartesian frame, `gradient
<layer> <name> r theta phi` and the derivatives in r, theta and phi of
a scalar, `absent <layer> <name>` for a variable a layer lacks,
`displacement x y z` and u(X) for a mapping, and for a field on
interfaces `surface <interface> <name> theta phi` and its value,
`surface_gradient <interface> <name> theta phi` and its derivatives in
theta and phi, or `surface_absent <interface> <name>`.  **Model files to refuse**
(`model_cases/bad_*.nc`): malformed copies, listed in `bad.txt` as the
case, a tab, the place the fault lies, a tab, and a phrase the refusal
says; a structural variable holding its fill, an element given to a
layer its bounds lie outside, a scale that is not finite, interface radii
that are not the skeleton's, a fractional `ngll`, a `behaviours` that
is a number or names a behaviour twice, a lateral rule the
grid of its group cannot carry (the harmonic rule on a grid that is custom
or states no band, and the bicubic rule on `model_cases/thin.nc`, a valid
file on a grid of two longitudes), polar values that are absent, of the
wrong size or over the wrong dimensions on a grid lacking a pole (of a
variable of a part, of a displacement, of the stack and of a field on
interfaces), a stack of one surface or of radii that do not increase,
and layer bounds that are negative, not finite or none at all are among
them; a copy that no edit in place makes (a variable or a dimension
left out, a dimension made shorter) is rebuilt by `_Rebuilding`.  **GLL points**
(`gll.txt`): `gll <ngll>` and the points on [-1, 1].

**Surface files** (`surface_basic.nc`): one surface file with a field
of every time support on a Gauss-Legendre grid, polar values on three of
them (a static field, and two node fields, one read by the bilinear rule
and one by the bicubic), a NaN-marked point, geophysical scales, a
radius, meta and a citation; beside it `surface_basic.expect`, one fact
per line:
`scales`, `radius`, `grid <kind> <lmax> <ntheta> <nphi>`,
`time <k> <value>` and `instant <k> <value>`, per field
`variable <name> <support> <time rule> <lateral rule> <m l t T | ->`,
sampled entries `value <name> <k|-> <i> <j> <value>` (`absent` where
the point is NaN), `poles <name> <k|-> <north> <south>`, and the
history's reading
`sample <name> <date|-> <refuse|hold> <i> <j> <value>`, from
`SurfaceHistory.values` at dates between nodes, at nodes, held outside
the span, inside and closing an interval, and at an instant, and
`function <name> <date|-> <refuse|hold> <theta> <phi> <value> <dtheta>
<dphi>`, from `SurfaceHistory.function`, a field at a date read by its
lateral rule with its stored polar values, at the poles, the seam, a
node and between nodes.  Beside it `surface_poles.nc`, a static field
and a node field on an equiangular grid holding both poles, which carry
no polar values, in SI and without a radius, with its expectations.
**Surface files to refuse** (`surface_cases/bad_*.nc`): copies of the
first with a field
that does not run over its support's dimension and the grid's (the grid's
two transposed, another dimension as long as `time`), polar values
that do not run over their field's, a scale that is not finite, and a
field stating the harmonic rule on a grid that is custom or states no
band, and copies of the second with polar values on its grid, which holds
both poles, and with a dimension `pole` of three, listed in `bad.txt` as
the model cases are.

**Interpolation cases** (`case_lateral_*.nc`, `case_radial_*.nc`), one
netCDF file each: for a lateral case the grid (`colatitude`,
`longitude`, radians), its `kind`, the `values` at its nodes and the
`poles`, the `method`, the query points `query_colatitude` and
`query_longitude`, and the `expected` values with their derivatives
`expected_dtheta` and `expected_dphi`; for a radial case the element
`breaks`, `ngll`, the node `radius` and the `values` there (element by
element), the `method`, the query radii and the `expected` values and
derivatives, the rules in the logarithm on positive values spanning
decades with one element of one value.
Every case covers the poles, the longitude seam and points between
nodes.  The local rules only: the C++ side does no spherical-harmonic
work.

**Grids** (`grids.txt`): for each shipped grid, named as the C++ makes
it (`gl:<lmax>[:<nphi>]`, `dh:<lmax>:<sampling>`,
`eq:<ntheta>:<nphi>:<poles>`), its `band`, `colatitudes`, `longitudes`
and `weights`, one line each.  **The vocabulary** (`vocabulary.txt`):
each name, its rank, weight and Voigt form, its four exponents, and its
meaning's UTF-8 in hexadecimal; then each family's stem the same way,
after the word `family`.

**Frames** (`frames.txt`): `frame <theta> <phi>` and the nine entries
of the frame matrix by rows, and `wrapped <x> <start> <value>`, the
longitude reduction, the seam's rounding case among them.

**JSON** (`json_*.json`, and `manifests/<case>/*.json`, the manifests
the mesher and the MFEM exports write on unit geometries, which need
gmsh and PyMFEM and are left out with a note without them), each with a
walk beside it (`<file>.walk`): one line per value in document order,
its path from the root as the C++ reader writes it, a tab, its type
(`null`, `bool`, `number`, `string`, `array`, `object`), a tab, and its
value as Python's `json` reads it (`true` or `false`; a number written
to round-trip; a string's UTF-8 bytes in hexadecimal; the length of an
array or an object).  **Manifests of cubed meshes**
(`cubed_manifests/<case>/`, which need PyMFEM and no gmsh, and are left
out with a note without it): the manifest `build_cubed_sphere` writes
beside its mesh of hexahedra for a unit ball of three layers
(`cubed_sphere`), and the one `build_squared_circle` writes for a hollow
disc of two layers under a shell, at order two (`squared_circle`), each
with its walk.

**Manifests to refuse** (`manifest_cases/`): a valid manifest, the
format page's example with a refinement and a model file, as
`base.json` (schema 5); variants the reader accepts (`accept_*.json`:
the keys added within the schema left out, an unknown key added, and
two faults meeting along a junction added with their collars, schema
6); and one malformed copy per check `validate_structure` makes
(`bad_*.json`, the faults' and collars' checks on the copy with
faults), listed in `bad.txt`, one per line, as the case, a tab, the
JSON path the fault lies at (empty for the document itself), a tab,
and a phrase the refusal says.  None needs gmsh.

**Meshes with faults** (`fault_meshes/<case>/`, which need gmsh and are
left out with a note without it): a unit ball of two layers under a
bump of relief, meshed at order one, with a plane reaching the surface
(`surface`: a tip and a free edge), with a splay ending on its main
fault (`splay`: a junction, on the splay's boundary and inside the
main fault), and with two planes crossing (`crossing`: a junction inside
both), each held to `testing.check_fault_mesh` first.  Beside each
manifest (`<case>.json`), `<case>.mesh.txt` holds
the mesh as plain arrays (as `mesh3d.read_mesh` reads it), one per
line: `vertex x y z` (the physical positions, the mapping applied, in
index order), `cell a b c d attribute` (each tetrahedron's corners and
its element attribute, a collar's where it is in one)
and `boundary a b c attribute` (each boundary triangle's corners in
the order the mesh holds them);
and `<case>.expect` the answers of the Python's rules
(`mesh3d.manifest.plus_cell`, `fault_curves`, `fault_walls`): `pair
<boundary> <cell> <cell> <0|1>` for each face of a fault, its two cells
(indices among the `cell` lines, the smaller first) and which of them
is the plus one; `edge <fault> <tip|free_edge|junction|junction_end> a
b` for each edge on a fault's curves; `ends <fault> <fault>` for each
fault a fault ends on; and `wall <fault> <cell> <1|-1>` for each cell
in a wall of a fault, every other cell in neither.

**Fault files** (`fault_*.nc`): a static plane in a geographic frame
with a `cell` slip (`fault_static`); a fault in kilometres and years of
a plane, a bent bicubic patch and a bilinear patch across the date
line, its slip on three time nodes read `linear` (`fault_afterslip`);
and a Cartesian fault of a twisted bilinear patch, a vertical plane and
a sill, on two nodes read `previous`, far from the time origin against
their span (`fault_cartesian`); every slip rule among them.  Beside
each, `<fault>.expect`, the Python's answers,
one fact per line: `fault <frame> <reference> <support> <rule>
<patches> <slices>`, `scales`, `time <k> <value>`, `patch <k> <kind>
<name in hexadecimal | ->`; at stated (u, v) of each patch, corners,
edges, knots and cell boundaries among them, `point`, `position`,
`tangents` (x_u then x_v), `second` (x_uu, x_uv, x_vv), `normal`,
`area` and `basis` (the fault basis by rows) as `<what> <k> <u> <v>`
and the numbers; `slip <k> <u> <v> <date|-> <refuse|hold>
<fault|ambient|cartesian>` and the vector, or `refused` in its place;
`quadrature <points> <date|-> <n_u,...> <n_v,...> <size>` and a
`qpoint <i> <patch> <u> <v> <weight>` line, then the point, position,
normal and slip, for each of its points; `slices <points> <n_u,...>
<n_v,...> <size> <slices>` and an `spoint <i> <patch> <u> <v> <weight>`
line, then the point, position, normal and the slip of every slice in
turn, for each point of the slice quadrature; the dates read include
some within the default tolerance of a node or an end of the span and
one beyond it; `locate <x> <y> <z> <patch>
<u> <v> <distance> <inside> <converged>` for physical points beside
the patches, beyond their edges and corners and near a bilinear
patch's creases; and `slip_at <max_distance> <date|-> <basis> <x> <y>
<z>` and the vector, or `slip_at_refused <max_distance> <x> <y> <z>`.
**Fault files to refuse** (`fault_cases/bad_*.nc`): malformed copies,
listed in `bad.txt` as the model cases are.  **Slip from a rake**
(`slip_rake.txt`): for each case `case <nslices> <n_u> <n_v> <rule>`,
then the `rake`, `magnitude` and `opening` given to `Slip.from_rake`
(one value, one a point or one a slice and point) and the `values` it
makes, (nslices, n_u, n_v, 3) in C order.

**Burgers bodies** (`burgers.txt`): `terms <modulus> <viscosity>
<kelvin_viscosity> <kelvin_modulus_fraction>` and the two strengths and
two times `conversion.prony.burgers_terms` gives, the faster first, for
bodies across a mantle's ranges and to the extremes of the Kelvin
element; and `parameters <modulus> <w_1> <w_2> <tau_1> <tau_2>` and the
viscosity, Kelvin viscosity and Kelvin modulus fraction
`burgers_parameters` gives, for each pair of branches in both orders.

**Flat fixtures** (`flat_*`), of `planetmodel.cartesian`, a body layered
in depth over a box or a rectangle, each kind as its spherical counterpart
above with these differences.  **Flat model files**
(`flat_model_<case>.nc`, schema `planetmodel.model/3`): fields of depth
alone over a box and over a rectangle (`layered`, `layered_rectangle`);
lateral fields of ranks 0, 1 and 2 on the bilinear and the bicubic rule,
a perturbed field, in two parts on two grids over a padded box
(`lateral`), and over a rectangle, whose grid has the one axis (`rectangle`);
a Fourier field, stating the rule the C++ refuses (`fourier`, without
expectations); topography as a stack of reliefs over a box and over a
rectangle (`stacked`, `stacked_rectangle`, the `geometry` group) and a
vertical stretch that is no stack (`stretched`, the `displacement`
variable); fields on interfaces (`surfaces`); a model padded below its
base with its lowest properties held (`held`); a viscosity read in
its logarithm (`log`); and fields of depth alone written with no grid
given, on the grid of one node, the model's type having a behaviour of
its own (`depth`).  Their expectations have `domain <box|rectangle>
<bounds> <padding>`, a point is `depth x y`, a frame `l` (local, up x y)
or `c` (Cartesian, z up), a `gradient` the derivatives in depth, x and y,
`displacement x y z` the stretch's u(X), and a field on an interface is
of `x y`.  **Flat model files to refuse** (`flat_model_cases/`), listed
in `bad.txt` as the model cases are, an element given to a layer its
bounds lie outside, a stack whose first surface lies above the free
surface, whose depths do not increase or which has one surface, interface
depths that are not the skeleton's and the bicubic rule
on `flat_model_cases/thin.nc`, a valid file on a grid of two nodes along
y, among them.  **Flat interpolation cases**
(`flat_case_lateral_<grid>_<rule>.nc`): the domain's `bounds` and
`padding`, the grid's `x` (and `y`), the `values`, the query points
`query_x` and `query_y` over several periods and at the seam, and the
`expected` values with `expected_dx` and `expected_dy`.  **Flat surface
files** (`flat_surface_box.nc` below the surface, `flat_surface_rectangle.nc`
on it, schema `planetmodel.surface/2`), their expectations those of the
surface file with `depth`, `domain`, `grid <shape>`, an index `i j` into
(nx, ny) (`j` 0 over a rectangle), and `function <name> <date|-> x y value
dx dy`, a field at a date read by its lateral rule; and the malformed
copies (`flat_surface_cases/`).  **Flat manifests**
(schema 7): `flat_manifest_cases/`, a base (a box exported with a model),
accepted variants (a rectangle, a top below the surface, faults with a
face edge) and malformed copies listed in `bad.txt`, none needing gmsh;
`flat_manifests/<case>/`, the manifests the mesher and the MFEM export
write for a box, a rectangle, a box with a fault leaving through a side,
and flat models with their model files, each with its walk; and
`flat_fault_meshes/<case>/`, a box with a fault reaching the surface and
leaving through a side, and one with a thrust, a splay ending on it and
a buried fault, each mesh as plain arrays with the answers of the
Python's rules, an `edge` being also of the kind `face_edge`.

**The facts the C++ restates** (`constants.txt`), one per line, a key
and its value separated by spaces: the version, `G_SI`,
`EARTH_MEAN_DENSITY`, and for each of a few scales, `scales <label>`
with its four values, `factor <label> <m l t T> <value>` and
`unit <label> <m l t T> <string>` for dimensions across the vocabulary,
floats written to round-trip exactly.
"""
from __future__ import annotations

import json
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from scipy.interpolate import PPoly

import planetmodel as pm
from planetmodel import units
from planetmodel.behaviours import model_type
from planetmodel.deck import deck_knots
from planetmodel.lateral import LateralInterpolant
from planetmodel.layerfunction import ExponentialLayer
from planetmodel.mesh1d import Mesh1D
from planetmodel.modelfile import LOG_RULES, validate_structure, write_model
from planetmodel.surfacefile import TIME_METHODS


def smooth(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """A smooth function on the sphere, the exponential of a Cartesian one."""
    st = np.sin(theta)
    return np.exp(st * np.cos(phi) + 0.5 * st * np.sin(phi) - 0.3 * np.cos(theta))


def _queries(grid: pm.AngularGrid, rng: np.random.Generator
             ) -> tuple[np.ndarray, np.ndarray]:
    """Random points, the poles, the seam and the nodes' own positions."""
    t = list(rng.uniform(0.0, np.pi, 40)) + [0.0, 0.0, np.pi, np.pi]
    p = list(rng.uniform(-np.pi, 3.0 * np.pi, 40)) + [0.3, 4.0, 1.1, 5.9]
    phi0 = float(grid.longitudes[0])
    t += [1.0, 1.0, float(grid.colatitudes[1])]
    p += [phi0 - 1e-9, phi0 + 1e-9, float(grid.longitudes[2])]
    return np.array(t), np.array(p)


def _write_case(path: Path, arrays: dict[str, np.ndarray],
                attributes: dict[str, object]) -> Path:
    """One case file: every array on its own dimension of its length."""
    import netCDF4
    with netCDF4.Dataset(path, "w") as ds:
        for key, value in attributes.items():
            ds.setncattr(key, value)
        for name, value in arrays.items():
            value = np.asarray(value, dtype=float)
            dims = tuple(f"{name}_{k}" for k in range(value.ndim))
            for d, n in zip(dims, value.shape):
                ds.createDimension(d, n)
            ds.createVariable(name, "f8", dims)[:] = value
    return path


def lateral_cases(directory: Path) -> list[Path]:
    """The lateral rules on every kind of grid."""
    rng = np.random.default_rng(20261018)
    offset = pm.AngularGrid(np.linspace(0.2, 2.9, 9),
                            np.sort(rng.uniform(0.3, 6.1, 11)))
    grids = {"gauss_legendre": pm.gauss_legendre(6),
             "driscoll_healy": pm.driscoll_healy(5),
             "equiangular": pm.equiangular(9, 16),
             "equiangular_poles": pm.equiangular(9, 16, poles=True),
             "custom_offset": offset}
    out = []
    for label, grid in grids.items():
        values = smooth(grid.colatitudes[:, None], grid.longitudes[None, :])
        poles = smooth(np.array([0.0, np.pi]), np.zeros(2))
        t, p = _queries(grid, rng)
        for method in ("linear", "cubic"):
            f = LateralInterpolant(values, grid, method=method, poles=poles)
            dt, dp = f.gradient(t, p)
            out.append(_write_case(
                directory / f"case_lateral_{label}_{method}.nc",
                {"colatitude": grid.colatitudes, "longitude": grid.longitudes,
                 "values": values, "poles": poles, "query_colatitude": t,
                 "query_longitude": p, "expected": f(t, p),
                 "expected_dtheta": dt, "expected_dphi": dp},
                {"kind": grid.kind, "method": method, "mean": float(f.mean())}))
    return out


def radial_cases(directory: Path) -> list[Path]:
    """The radial rules on elements of unequal widths."""
    mesh = Mesh1D([0.0, 0.35, 0.5, 1.0], ngll=5)
    values = np.sin(3.0 * mesh.r) + mesh.r ** 2
    r = np.concatenate([np.linspace(0.0, 1.0, 41), mesh.left])
    out = []
    for method in ("polynomial", "linear"):
        pp = mesh.to_ppoly(values, method=method)
        out.append(_write_case(
            directory / f"case_radial_{method}.nc",
            {"breaks": mesh.breakpoints, "radius": mesh.r, "values": values,
             "query_radius": r,
             "expected": pp(r), "expected_dr": pp.derivative()(r)},
            {"method": method, "ngll": np.int32(mesh.ngll)}))
    # positive values over decades, the middle element of one value: the
    # exponential of each element's polynomial through log(y / y_e)
    positive = 1e21 * np.exp(4.0 * np.sin(5.0 * mesh.r) - 3.0 * mesh.r)
    positive[1] = 2.5e21
    first = positive[:, :1]
    for method in LOG_RULES:
        exponent = mesh.to_ppoly(np.log(positive / first),
                                 method=method.removeprefix("log-"))
        f = ExponentialLayer((0.0, 1.0), exponent,
                             factor=PPoly(first.T.copy(), mesh.breakpoints))
        out.append(_write_case(
            directory / f"case_radial_{method}.nc",
            {"breaks": mesh.breakpoints, "radius": mesh.r, "values": positive,
             "query_radius": r,
             "expected": f(r), "expected_dr": f.derivative()(r)},
            {"method": method, "ngll": np.int32(mesh.ngll)}))
    return out


class Tidal:
    """A behaviour of a program's own: a model file names it, and a reader
    that lacks it carries the name."""


def model_files(directory: Path) -> list[Path]:
    """The model files of the module docstring."""
    prem = pm.PREM()
    lo, hi = 3480e3, 5701e3

    def dlnvs(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.02 * np.sin(t) ** 2 * np.cos(2 * p) * (r - lo) / (hi - lo)

    perturbed = pm.perturbed(prem, "vsv", pm.AnalyticField((lo, hi), dlnvs),
                             kind="relative")
    base = ["rho", "vpv", "vsv", "vph", "vsh", "eta"]
    two = pm.LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0],
                                     vp=[9.0, 7.0], vs=[4.0, 3.0])
    grid = pm.gauss_legendre(8)

    def rho3d(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 4.0 + 0.2 * r * smooth(t, p)

    def wind(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        r, t, p = np.broadcast_arrays(r, t, p)
        return np.stack([r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p),
                         r * np.cos(t) + 1.0], axis=-1)

    lateral = two.with_field(1, "rho", pm.GriddedField.sampled(
        pm.AnalyticField((0.5, 1.0), rho3d, character=pm.DENSITY), grid,
        breaks=[0.5, 0.75, 1.0], ngll=4, lateral="cubic"), replace=True)
    lateral = lateral.with_field(1, "flow", pm.GriddedField.sampled(
        pm.AnalyticField((0.5, 1.0), wind, character=pm.VECTOR, frame="cartesian"),
        grid, ngll=3, lateral="linear"))
    parts = {i: pm.gauss_legendre(4) for i in range(2)}
    dh, eq = pm.driscoll_healy(5), pm.equiangular(9, 16, poles=True)
    parts.update({i: dh for i in range(2, 9)})
    parts.update({i: eq for i in range(9, 13)})

    def bump(t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.02 * np.sin(t) ** 3 * np.cos(3 * p) + 0.01 * np.cos(t)

    relief = pm.GriddedFunction.sampled(bump, pm.gauss_legendre(16))
    stacked = two.stretched(pm.layer_linear(two.skeleton, [None, relief, relief]))
    flattened = two.stretched(pm.flattening(0.01))
    kinds = two.with_field(1, "viscosity", pm.constant_field((0.5, 1.0), 1e21,
                                                             name="viscosity"))
    kinds = pm.perturbed(kinds, "viscosity", pm.AnalyticField(
        (0.5, 1.0), lambda r, t, p: 0.3 * np.sin(t) ** 2 * np.cos(p) * r), kind="log")
    kinds = pm.perturbed(kinds, "rho", pm.AnalyticField(
        (0.5, 1.0), lambda r, t, p: 0.1 * r * smooth(t, p), character=pm.DENSITY),
        kind="absolute")

    def stress(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        r, t, p = np.broadcast_arrays(r, t, p)
        x, y, z = r * np.sin(t) * np.cos(p), r * np.sin(t) * np.sin(p), r * np.cos(t)
        return np.stack([1 + x * y, 2 + z, 3 - x, 0.5 * y * z, 0.2 * x, 0.1 + y], -1)

    def elastic(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        r, t, p = np.broadcast_arrays(r, t, p)
        base = np.eye(6) * 3.0 + 0.5
        bump = (0.1 * r * smooth(t, p))[..., None, None]
        return base + bump * np.add.outer(np.arange(6), np.arange(6)) / 10.0

    radial_stress = pm.RadialField((0.0, 0.5), [
        lambda r: 1.0 + r, lambda r: 2.0 - r, lambda r: 3.0 + r ** 2,
        lambda r: 0.1 * r, lambda r: 0.2 * r, lambda r: 0.3 * r], character=pm.STRESS,
        name="stress")
    tensors = two.with_field(0, "stress", radial_stress)
    tensors = tensors.with_field(1, "stress", pm.GriddedField.sampled(
        pm.AnalyticField((0.5, 1.0), stress, character=pm.STRESS, frame="cartesian"),
        pm.gauss_legendre(4), ngll=3, lateral="cubic"))
    tensors = tensors.with_field(1, "elastic", pm.GriddedField.sampled(
        pm.AnalyticField((0.5, 1.0), elastic, character=pm.ELASTIC, frame="cartesian"),
        pm.gauss_legendre(4), ngll=3, lateral="linear"))
    kernel = two.replaced(specs={**two.specs, "K_shape": pm.FieldSpec(
        pm.SCALAR, pm.Dimensions(length=-2), meaning="a shape kernel")})
    on_faces = {"K_shape": {0: lambda t, p: smooth(t, p) - 0.4 * np.cos(t), 1: smooth},
                "shift": {1: pm.GriddedFunction.sampled(bump, pm.gauss_legendre(8))}}
    radial = model_type(pm.Model, (pm.Elastic, Tidal, pm.SelfGravitating))(
        two.geometry, [layer.fields for layer in two.layers], name="radial")
    return [
        write_model(directory / "model_radial.nc", radial.with_gravity()),
        write_model(directory / "model_prem.nc", prem.with_pressure(),
                    grid=pm.gauss_legendre(2)),
        write_model(directory / "model_kinds.nc", kinds, grid=pm.gauss_legendre(6)),
        write_model(directory / "model_tensors.nc", tensors, grid=pm.gauss_legendre(4),
                    fields=["rho", "stress", "elastic"]),
        write_model(directory / "model_prem_perturbed.nc", perturbed,
                    grid=pm.gauss_legendre(8), fields=base,
                    interpolation={"vsv": ("polynomial", "cubic")}),
        write_model(directory / "model_lateral.nc", lateral, grid=grid),
        write_model(directory / "model_parts.nc", perturbed, grid=parts,
                    fields=base),
        write_model(directory / "model_stacked.nc", stacked,
                    grid=pm.gauss_legendre(4)),
        write_model(directory / "model_flattened.nc", flattened,
                    grid=pm.gauss_legendre(8)),
        write_model(directory / "model_surfaces.nc", kernel, grid=pm.gauss_legendre(4),
                    fields=["rho", "vp"], surfaces=on_faces),
        write_model(directory / "model_converted.nc", converted(),
                    grid=pm.gauss_legendre(6),
                    fields=["rho", "vsv", "vsh", "viscosity"]),
        write_model(directory / "model_prony.nc", with_burgers(converted()),
                    grid=pm.gauss_legendre(6)),
        write_model(directory / "model_stratified.nc",
                    prem.stabilised(layers=["outer_core"])[0],
                    grid=pm.gauss_legendre(2),
                    fields=["rho", "vpv", "vsv", "bullen_excess"]),
    ] + log_model_files(directory) + [
        _refilled(directory / "model_prem.nc", directory / "model_fill.nc",
                  group="part_0", variable="qmu", fill=-999.0)]


def _refilled(source: Path, target: Path, *, group: str, variable: str,
              fill: float) -> Path:
    """A copy of a netCDF file in which one variable of a group marks its
    absent values by `fill` and not by NaN: `fill` is its `_FillValue`, and
    the value stored wherever the source holds NaN."""
    import netCDF4

    def copy(old: Any, new: Any, marked: bool) -> None:
        new.setncatts({key: old.getncattr(key) for key in old.ncattrs()})
        for name, dim in old.dimensions.items():
            new.createDimension(name, len(dim))
        for name, var in old.variables.items():
            var.set_auto_mask(False)
            values = var[...]
            attrs = {key: var.getncattr(key) for key in var.ncattrs()}
            kept = attrs.pop("_FillValue", None)
            if marked and name == variable:
                absent = np.isnan(values)
                if not absent.any() or absent.all():
                    raise AssertionError(f"{variable} is absent from no layer, or all")
                values, kept = np.where(absent, fill, values), fill
            out = new.createVariable(name, var.dtype, var.dimensions, fill_value=kept)
            out.set_auto_mask(False)
            out[...] = values
            out.setncatts(attrs)
        for name, sub in old.groups.items():
            copy(sub, new.createGroup(name), name == group)

    with netCDF4.Dataset(source) as old, netCDF4.Dataset(target, "w") as new:
        copy(old, new, False)
    return target


#: The behaviours of the models built from tables of the graded viscosity.
_VISCOELASTIC = (pm.Elastic, pm.SelfGravitating, pm.Viscoelastic)

#: The interface names of those models: the outermost alone is named.
_GRADED_FACES = (None, None, None, "surface")

#: The elastic columns of those tables.
_ELASTIC_COLUMNS = ("rho", "vpv", "vsv", "vph", "vsh", "eta")

#: The Kelvin element of the layers of those models made Burgers bodies.
_KELVIN = {"kelvin_viscosity": 1e20, "kelvin_modulus_fraction": 0.5}

#: The two Prony branches of a Burgers body, as a model holds them.
_BRANCHES = ("relaxation_strength_1", "relaxation_time_1",
             "relaxation_strength_2", "relaxation_time_2")


def graded_maxwell() -> tuple[pm.Model, tuple[np.ndarray, ...]]:
    """A Maxwell model built from a table, and the table's knots by layer:
    a fluid core, a mantle whose viscosity changes by three decades
    between knots, up and down, joined in its logarithm, a mantle of one
    viscosity and an elastic crust."""
    knots = [np.linspace(0.0, 1.2e6, 3), np.linspace(1.2e6, 5.0e6, 6),
             np.linspace(5.0e6, 6.2e6, 4), np.linspace(6.2e6, 6.371e6, 2)]
    ends = {"rho": [(11000, 10000), (5500, 4500), (4000, 3500), (2900, 2700)],
            "vp": [(9000, 8500), (13000, 11000), (10000, 8500), (6800, 6500)],
            "vs": [(0, 0), (7000, 6000), (5500, 4700), (3900, 3700)]}
    columns = {}
    for name, column in (("rho", "rho"), ("vpv", "vp"), ("vsv", "vs"),
                         ("vph", "vp"), ("vsh", "vs")):
        columns[name] = np.concatenate([np.linspace(a, b, r.size)
                                        for (a, b), r in zip(ends[column], knots)])
    columns["eta"] = np.ones(sum(r.size for r in knots))
    columns["viscosity"] = np.concatenate([
        np.full(3, np.nan), 10.0 ** (20.0 + 3.0 * np.array([0, 1, 0, 1, 0, 1])),
        np.full(4, 1e21), np.full(2, np.nan)])
    deck = pm.Deck(np.concatenate(knots), columns)
    model = pm.model_from_deck(deck, behaviours=_VISCOELASTIC,
                               kinds={"viscosity": "log-linear"},
                               interface_names=_GRADED_FACES)
    return model, deck_knots(deck)


def graded_prony(maxwell: pm.Model, knots: tuple[np.ndarray, ...]) -> pm.Model:
    """`maxwell` with each viscosity made the one Prony branch it is,
    tabulated on `knots` and brought in again as a table: the strength
    joined by straight lines and the relaxation time, eta / mu at the
    knots, in its logarithm."""
    from planetmodel.conversion import prony
    branched = prony.with_branches(maxwell, prony.maxwell_series())
    branch = ("relaxation_strength_1", "relaxation_time_1")
    deck = pm.sample_deck(branched, _ELASTIC_COLUMNS + branch, knots=knots)
    return pm.model_from_deck(deck, behaviours=_VISCOELASTIC,
                              kinds={"relaxation_strength": "linear",
                                     "relaxation_time": "log-linear"},
                              interface_names=_GRADED_FACES)


def log_model_files(directory: Path) -> list[Path]:
    """The model files stating a rule in the logarithm."""
    from planetmodel.conversion import prony
    maxwell, knots = graded_maxwell()
    lo, hi = maxwell.layer(1).interval

    def dlneta(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.3 * np.sin(t) ** 2 * np.cos(p) * (r - lo) / (hi - lo)

    grid = pm.gauss_legendre(4)
    lateral = pm.perturbed(maxwell, "viscosity", pm.GriddedField.sampled(
        pm.AnalyticField((lo, hi), dlneta), grid, ngll=3, lateral="cubic"), kind="log")
    relief = pm.GriddedFunction.sampled(
        lambda t, p: 2e3 * np.sin(t) ** 2 * np.cos(2 * p), pm.gauss_legendre(8))
    lateral = lateral.stretched(pm.layer_linear(
        lateral.skeleton, [None, relief, None, None, relief]))
    return [
        write_model(directory / "model_log.nc", maxwell, grid=pm.gauss_legendre(2)),
        write_model(directory / "model_log_prony.nc", graded_prony(maxwell, knots),
                    grid=pm.gauss_legendre(2)),
        write_model(directory / "model_log_lateral.nc", lateral, grid=grid,
                    fields=["rho", "vsv", "viscosity"]),
        write_model(directory / "model_log_burgers.nc",
                    prony.with_burgers(maxwell, [2], **_KELVIN)),
        write_model(directory / "model_log_burgers_lateral.nc",
                    prony.with_burgers(lateral, [1], **_KELVIN), grid=grid,
                    fields=["rho", "vsv", "viscosity", *_BRANCHES],
                    interpolation={name: ("log-polynomial", "cubic")
                                   for name in _BRANCHES if "time" in name}),
    ]


def with_burgers(model: pm.Model) -> pm.Model:
    """The model with its viscosity made a Burgers body's two branches."""
    from planetmodel.conversion import building, prony
    burgers = prony.with_branches(model, prony.burgers_series(
        kelvin_modulus_fraction=0.2, viscosity_ratio=5.0))
    return building.with_power_law(burgers, exponent=3.5, transition_stress=1e6)


def converted() -> pm.Model:
    """PREM with a viscosity of 2e21 Pa s below 670 km and 5e20 above, its
    vsv and vsh perturbed over the lower mantle, through sl3d's chain."""
    from planetmodel.conversion import building, parameters
    prem = pm.PREM()
    lo, hi = 3480e3, 5701e3
    for lay in prem.layers:
        a, b = lay.interval
        if lo <= a and b <= 6291e3:
            prem = prem.with_field(lay.index, "viscosity", pm.constant_field(
                (a, b), 2e21 if b <= hi else 5e20, name="viscosity"))

    def dlnvs(r: np.ndarray, t: np.ndarray, p: np.ndarray) -> np.ndarray:
        return 0.01 * np.sin(t) ** 2 * np.cos(2 * p) * (0.5 + (r - lo) / (hi - lo))

    for name in ("vsv", "vsh"):
        prem = pm.perturbed(prem, name, pm.AnalyticField((lo, hi), dlnvs),
                            kind="relative")
    dt = building.temperature_anomaly(
        building.velocity_anomaly(prem), scaling=parameters.SL3D_DENSITY_SCALING,
        expansivity=parameters.SL3D_EXPANSIVITY)
    return building.with_viscosity(prem, dt, parameters.SL3D_VISCOSITY)


#: The JSON documents of the walk fixtures, as bytes on disk: the
#: grammar's corners, and a nesting deep but within the reader's limit.
_JSON_DOCUMENTS = {
    "corners": (
        '{"unicode": "caf\\u00e9 \\ud83d\\ude00", "raw": "caf\u00e9 \U0001f600",\n'
        ' "escapes": "a\\"b\\\\c\\/d\\b\\f\\n\\r\\t",\n'
        ' "numbers": [0, -0.0, 1e300, 5e-324, 0.1, 123456789012, -1.5E-3, 1E+2,\n'
        '             2e-1],\n'
        ' "empty": {}, "list": [], "nested": [[[{"x": null}]]],\n'
        ' "keys": {"a b": 1, "x.y": 2, "": 3, "q\\"uote": 4, "9lives": 5, "_ok9": 6},\n'
        ' "bools": [true, false], "spaced": [ 1 ,\t2\r\n, 3 ]}\n'),
    "deep": "[" * 200 + "7" + "]" * 200,
    "scalar": "  -12.5e3  ",
}


def _path(path: str, key: str | int) -> str:
    """The path of an element or a member, by the C++ reader's rule."""
    if isinstance(key, int):
        return f"{path}[{key}]"
    if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
        return key if not path else f"{path}.{key}"
    quoted = key.replace("\\", "\\\\").replace('"', '\\"')
    return f'{path}["{quoted}"]'


def _walk(value: Any, path: str = "") -> list[str]:
    """The walk lines of a parsed JSON value, by the module docstring."""
    if value is None:
        lines = [f"{path}\tnull\t"]
    elif isinstance(value, bool):
        lines = [f"{path}\tbool\t{'true' if value else 'false'}"]
    elif isinstance(value, (int, float)):
        lines = [f"{path}\tnumber\t{float(value)!r}"]
    elif isinstance(value, str):
        lines = [f"{path}\tstring\t{value.encode('utf-8').hex()}"]
    elif isinstance(value, list):
        lines = [f"{path}\tarray\t{len(value)}"]
        for i, v in enumerate(value):
            lines += _walk(v, _path(path, i))
    else:
        lines = [f"{path}\tobject\t{len(value)}"]
        for k, v in value.items():
            lines += _walk(v, _path(path, k))
    return lines


def _write_walk(path: Path) -> Path:
    """The walk of the JSON document at `path`, beside it."""
    walk = path.with_name(path.name + ".walk")
    walk.write_text("\n".join(_walk(json.loads(path.read_bytes()))) + "\n",
                    encoding="utf-8")
    return walk


def json_documents(directory: Path) -> list[Path]:
    """The documents of `_JSON_DOCUMENTS`, each with its walk."""
    out = []
    for label, text in _JSON_DOCUMENTS.items():
        path = directory / f"json_{label}.json"
        path.write_bytes(text.encode("utf-8"))
        out += [path, _write_walk(path)]
    return out


def manifests(directory: Path) -> list[Path]:
    """The manifests the Python writes, each with its walk: the mesher's
    (3D, and 2D sized towards a point cloud), `export_mfem_mesh`'s with no
    model (and a hollow 2D mesh with a shell), and `export_mfem`'s of an
    elastic model with a fluid core in both deliveries, one with a model
    file beside it and one non-dimensional.  None without gmsh and
    PyMFEM."""
    try:
        import mfem.ser  # noqa: F401

        from planetmodel.mesh3d import (MeshSpec, NearPoints, Shell, UniformInterfaces,
                                        build_layered_mesh, export_mfem,
                                        export_mfem_mesh)
    except ImportError as error:
        print(f"no manifests without gmsh and PyMFEM: {error}", file=sys.stderr)
        return []
    root = directory / "manifests"
    shutil.rmtree(root, ignore_errors=True)
    model = pm.LayeredIsotropicElastic([0.0, 0.4, 0.8, 1.0], rho=[12.0, 5.0, 3.0],
                                       vp=[10.0, 9.0, 6.0], vs=[0.0, 5.0, 3.5])
    model = model.with_geometry(model.geometry.renamed(
        layers=["core", "mantle", "crust"], interfaces=["cmb", "moho", "surface"]))
    flat = model.stretched(pm.flattening(0.02))
    coarse = UniformInterfaces(0.3, 0.5, 0.5)
    hollow = pm.Geometry(pm.Skeleton([0.5, 0.8, 1.0]), layer_names=["lower", "upper"],
                         interface_names=["inner", "mid", "outer"])
    cloud = NearPoints(np.array([[1.0, 0.0]]), 0.1, 0.5, 0.3)
    written = []
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)

        def built(label: str, spec: MeshSpec) -> Any:
            return build_layered_mesh(spec, work / label / label)

        sphere = built("mesher", MeshSpec(model.geometry, coarse, dimension=3))
        oblate = built("oblate", MeshSpec(flat.geometry, coarse, dimension=3, order=2))
        shell = built("shell", MeshSpec(hollow, coarse, dimension=2,
                                        shells=(Shell(ratio=0.3),)))
        refined = built("mesher_refined", MeshSpec(model.geometry, coarse, dimension=2,
                                                   refinements=[cloud]))
        cases = {
            "mesher": sphere.manifest_path,
            "mesher_refined": refined.manifest_path,
            "mesh_only": export_mfem_mesh(sphere, work / "mesh_only" / "mesh_only"
                                          ).manifest_path,
            "hollow_shell": export_mfem_mesh(
                shell, work / "hollow_shell" / "hollow_shell").manifest_path,
            "referential": export_mfem(oblate, work / "referential" / "referential",
                                       model=flat, delivery="referential",
                                       model_file=pm.gauss_legendre(4)).manifest_path,
            "physical": export_mfem(oblate, work / "physical" / "physical", model=flat,
                                    delivery="physical").manifest_path,
            "nondimensional": export_mfem(
                sphere, work / "nondimensional" / "nondimensional",
                model=model.nondimensionalised()).manifest_path,
        }
        for label, manifest in cases.items():
            target = root / label
            shutil.copytree(manifest.parent, target)
            path = target / manifest.name
            written += [path, _write_walk(path)]
    return written


def cubed_manifests(directory: Path) -> list[Path]:
    """The manifests the builders of cubed meshes write, each with its mesh
    and its walk: a cubed sphere's, of a unit ball of three layers, and a
    squared circle's, of a hollow disc of two layers under a shell at order
    two.  None without PyMFEM; gmsh is not needed."""
    try:
        import mfem.ser  # noqa: F401

        from planetmodel.mesh3d import (CubedSphereSpec, Shell, SquaredCircleSpec,
                                        UniformInterfaces, build_cubed_sphere,
                                        build_squared_circle)
    except ImportError as error:
        print(f"no manifests of cubed meshes without PyMFEM: {error}", file=sys.stderr)
        return []
    root = directory / "cubed_manifests"
    shutil.rmtree(root, ignore_errors=True)
    ball = pm.Geometry(pm.Skeleton([0.0, 0.4, 0.8, 1.0]),
                       layer_names=["core", "mantle", "crust"],
                       interface_names=["cmb", "moho", "surface"])
    hollow = pm.Geometry(pm.Skeleton([0.5, 0.8, 1.0]), layer_names=["lower", "upper"],
                         interface_names=["inner", "mid", "outer"])
    coarse = UniformInterfaces(0.5, 0.5, 0.5)
    written = []
    for label, build, spec in (
            ("cubed_sphere", build_cubed_sphere,
             CubedSphereSpec(ball, 2, sizing=coarse)),
            ("squared_circle", build_squared_circle,
             SquaredCircleSpec(hollow, 2, sizing=coarse, order=2,
                               shells=(Shell(ratio=0.3),)))):
        path = build(spec, root / label / label).manifest_path
        written += [path, _write_walk(path)]
    return written


def _fault_cases() -> dict[str, dict[str, Any]]:
    """The faults of the meshes with faults, by case and name: planes and
    a bilinear patch on a unit sphere, with no slip."""
    from planetmodel.fault import Fault, GeographicFrame, Plane, Slip, StructuredPatch

    unit, zero = GeographicFrame(1.0), Slip(np.zeros((1, 1, 3)))

    def plane(depth: float, **kw: Any) -> Any:
        return Plane.at_depth(unit, 30.0, 40.0, depth, slip=zero, **kw)

    # a plane whose up-dip edge is at the reference surface
    trace = plane(0.15 * np.sin(np.deg2rad(60.0)), strike=20.0, dip=60.0, length=0.3,
                  width=0.3)
    # a main fault, and a splay whose down-dip edge lies in it exactly: a
    # bilinear patch through two of its points and the same two moved up
    # dip and into its hanging wall
    main = plane(0.2, strike=0.0, dip=60.0, length=0.4, width=0.3)
    foot = main.cartesian(np.array([-0.3, 0.3]), np.array([0.1, 0.1]))
    rise = (0.08 * main.normal(0.0, 0.0)
            + 0.2 * (main.cartesian(0.0, -0.5) - main.cartesian(0.0, 0.5)))
    splay = StructuredPatch(unit, unit.coordinates(np.stack([foot + rise, foot],
                                                            axis=1)),
                            slip=zero, interpolation="linear")
    # two planes crossing at their centres
    a = plane(0.2, strike=0.0, dip=60.0, length=0.3, width=0.3)
    b = Plane(unit, a.centroid, strike=90.0, dip=60.0, length=0.3, width=0.3,
              slip=zero)
    return {"surface": {"trace": Fault([trace])},
            "splay": {"main": Fault([main]), "splay": Fault([splay])},
            "crossing": {"a": Fault([a]), "b": Fault([b])}}


def fault_meshes(directory: Path) -> list[Path]:
    """The meshes with faults of the module docstring, under
    `fault_meshes/`, each held to `testing.check_fault_mesh` and the
    answers of the Python's rules written beside it.  None without
    gmsh."""
    from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                    UniformInterfaces, read_mesh)
    from planetmodel.mesh3d import manifest as mf
    try:
        from planetmodel.mesh3d import build_layered_mesh
    except ImportError as error:
        print(f"no meshes with faults without gmsh: {error}", file=sys.stderr)
        return []
    from planetmodel import testing

    def h(r: np.ndarray, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
        c = (np.cos(theta) * np.cos(np.deg2rad(60.0)) + np.sin(theta)
             * np.sin(np.deg2rad(60.0)) * np.cos(phi - np.deg2rad(40.0)))
        return 0.05 * r ** 2 * np.exp(-(1.0 - c) / 0.05)

    geometry = pm.Geometry(pm.Skeleton([0.0, 0.6, 1.0]), layer_names=["core", "mantle"],
                           interface_names=["cmb", "surface"]).with_mapping(
        pm.RadialStretch(pm.CallableDisplacement(h, name="bump"), rmax=1.0))
    near = InterfaceSizing(0.07, 0.3, 0.1)
    root = directory / "fault_meshes"
    shutil.rmtree(root, ignore_errors=True)
    written = []
    with tempfile.TemporaryDirectory() as scratch:
        for label, faults in _fault_cases().items():
            spec = MeshSpec(geometry, UniformInterfaces(0.15, 0.3, 0.3), order=1,
                            faults=[EmbeddedFault(f, name=n, sizing=near)
                                    for n, f in faults.items()])
            res = build_layered_mesh(spec, Path(scratch) / label / label)
            offset = max(p.offset for v in res.preimages.values() for p in v)
            testing.check_fault_mesh(res.mesh_path, faults, mapping=res.mapping,
                                     distance=2.0 * offset + 1e-12)
            card = mf.read(res.manifest_path)
            mesh = read_mesh(res.mesh_path)
            cells, layers = mesh.cells, mesh.cell_attributes
            boundary, attributes = mesh.faces, mesh.face_attributes
            x = np.asarray(res.mapping(mesh.points), dtype=float)
            faces = np.flatnonzero(np.isin(attributes, [f["attribute"]
                                                         for f in card.faults]))
            pairs = mesh.face_cells(boundary[faces])
            plus = mf.plus_cell(x, boundary[faces], cells[pairs])
            curves = mf.fault_curves([boundary[attributes == f["attribute"]]
                                      for f in card.faults],
                                     boundary[attributes == card.surface_attribute])
            walls = mesh.walls(card)

            target = root / label
            target.mkdir(parents=True)
            manifest_path = target / f"{label}.json"
            shutil.copyfile(res.manifest_path, manifest_path)
            lines = [f"vertex {a!r} {b!r} {c!r}" for a, b, c in x.tolist()]
            lines += [f"cell {a} {b} {c} {d} {g}"
                      for (a, b, c, d), g in zip(cells.tolist(), layers.tolist())]
            lines += [f"boundary {a} {b} {c} {g}"
                      for (a, b, c), g in zip(boundary.tolist(), attributes.tolist())]
            mesh_path = target / f"{label}.mesh.txt"
            mesh_path.write_text("\n".join(lines) + "\n")
            expect = [f"pair {b} {p} {q} {k}"
                      for b, (p, q), k in zip(faces.tolist(), pairs.tolist(),
                                              plus.tolist())]
            for record, edges, wall in zip(card.faults, curves, walls):
                for key in ("tip", "free_edge", "junction", "junction_end"):
                    expect += [f"edge {record['name']} {key} {a} {b}"
                               for a, b in getattr(edges, key).tolist()]
                expect += [f"ends {record['name']} {card.faults[j]['name']}"
                           for j in edges.ends_on]
                expect += [f"wall {record['name']} {c} {w}"
                           for c, w in zip(np.flatnonzero(wall).tolist(),
                                           wall[wall != 0].tolist())]
            expect_path = target / f"{label}.expect"
            expect_path.write_text("\n".join(expect) + "\n")
            written += [manifest_path, mesh_path, expect_path]
    return written


#: The valid manifest the refusal cases are made from.
_BASE_MANIFEST: dict[str, Any] = {
    "mesh": {"file": "earth.mesh", "format": "mfem", "nodes": "reference",
             "read_options": {"generate_edges": 1, "refine": 0,
                              "fix_orientation": False},
             "displacement": "displacement"},
    "layers": [
        {"attribute": 1, "name": "core", "r_inner": 0.0, "r_outer": 0.546,
         "in_geometry": True, "fluid": True},
        {"attribute": 2, "name": "mantle", "r_inner": 0.546, "r_outer": 1.0,
         "in_geometry": True, "fluid": False},
        {"attribute": 3, "name": "buffer", "r_inner": 1.0, "r_outer": 1.5,
         "in_geometry": False, "fluid": False}],
    "interfaces": [
        {"attribute": 1, "name": "cmb", "radius": 0.546, "between_layers": [0, 1],
         "kind": "fluid-solid", "values": {"rho": [1.7157, 0.7655]}},
        {"attribute": 2, "name": "surface", "radius": 1.0, "between_layers": [1, 2],
         "kind": "free", "values": {"rho": [0.4642, None]}},
        {"attribute": 3, "name": "interface_3", "radius": 1.5,
         "between_layers": [2, -1], "kind": "outer", "values": None}],
    "refinements": [
        {"name": "NearPoints", "size": 0.05, "far_size": 0.3, "decay_width": 0.2,
         "points": 12, "digest": "3f2a"}],
    "fields": [
        {"name": "displacement", "file": "earth.displacement.gf",
         "fe_space": "H1_3D_P2", "vdim": 3, "ordering": "byNODES", "rank": 1,
         "weight": 0, "voigt": False, "unit": "1", "layers": [1, 2, 3],
         "radial_degree": None},
        {"name": "rho", "file": "earth.rho.gf", "fe_space": "L2_3D_P2", "vdim": 1,
         "ordering": "byNODES", "rank": 0, "weight": 1, "voigt": False,
         "unit": "1", "layers": [1, 2], "radial_degree": 3}],
    "scales": {"length": 6371000.0, "mass": 1.4262e+24, "time": 1647.5,
               "temperature": 1.0},
    "constants": {"G": 1.0},
    "model": {"file": "earth.model.nc", "schema": "planetmodel.model/1"},
    "meta": {"kind": "test"},
    "schema": "planetmodel.mesh.manifest/5",
}


def _edited(edit: Any) -> dict[str, Any]:
    """A deep copy of the base manifest, changed by `edit(copy)`."""
    card = json.loads(json.dumps(_BASE_MANIFEST))
    edit(card)
    return card


def _set(*keys: Any, value: Any) -> Any:
    """An edit setting the value at a path of keys and indices."""
    def edit(card: dict[str, Any]) -> None:
        target = card
        for key in keys[:-1]:
            target = target[key]
        target[keys[-1]] = value
    return edit


def _drop(*keys: Any) -> Any:
    """An edit removing the key or element at a path."""
    def edit(card: dict[str, Any]) -> None:
        target = card
        for key in keys[:-1]:
            target = target[key]
        del target[keys[-1]]
    return edit


def _both(*edits: Any) -> Any:
    """An edit doing several in turn."""
    def edit(card: dict[str, Any]) -> None:
        for e in edits:
            e(card)
    return edit


#: Two faults of the base manifest, meeting along a junction: a fault
#: reaching the surface, pulled back and refined towards, and a splay of
#: two patches.
_FAULTS: list[dict[str, Any]] = [
    {"name": "megathrust", "attribute": 4, "layers": [2], "hanging_wall": [5, 6],
     "tip": 1, "free_edge": 2, "junction": 3, "meets": ["splay"], "ends_on": [],
     "controls": [[17, 17]], "offset": 3.3e-08,
     "sizing": {"size": 0.05, "far_size": 0.3, "decay_width": 0.2}},
    {"name": "splay", "attribute": 5, "layers": [1, 2], "hanging_wall": [4, 6, 7],
     "tip": 4, "free_edge": None, "junction": 6, "meets": ["megathrust"],
     "ends_on": ["megathrust"], "controls": [[9, 9], [5, 17]], "offset": 0,
     "sizing": None}]


#: The element attributes the base's layers carry beside `_FAULTS`: the
#: splay's collar in the core (4), and in the mantle the megathrust's (5),
#: both's (6) and the splay's (7).
_COLLARS = [[1, 4], [2, 5, 6, 7], [3]]


def _faulted(*edits: Any) -> Any:
    """An edit adding `_FAULTS` and their collars to the base, under the
    schema of a manifest with faults, then doing `edits` in turn."""
    def edit(card: dict[str, Any]) -> None:
        card["faults"] = json.loads(json.dumps(_FAULTS))
        card["schema"] = "planetmodel.mesh.manifest/6"
        for layer, attributes in zip(card["layers"], _COLLARS):
            layer["attributes"] = list(attributes)
        for e in edits:
            e(card)
    return edit


#: The variants the reader accepts: the keys added within the schema left
#: out, an unknown key added, and faults added.
_ACCEPTED_MANIFESTS = {
    "optional_keys_absent": _both(_drop("refinements"), _drop("model"),
                                  _drop("scales", "temperature"), _drop("meta"),
                                  _drop("constants")),
    "unknown_key": _set("extra", value={"anything": [1, 2]}),
    "no_model_said": _both(*(_set("layers", i, "fluid", value=None)
                             for i in range(3)),
                           *(_set("interfaces", i, "kind", value=None)
                             for i in range(3)),
                           *(_set("interfaces", i, "values", value=None)
                             for i in range(3)),
                           _set("scales", value=None), _set("constants", value={})),
    "faults": _faulted(),
}

#: The malformed variants: (edit, where the fault lies, what the refusal says).
_BAD_MANIFESTS = {
    "schema_4": (_set("schema", value="planetmodel.mesh.manifest/4"), "schema",
                 "regenerate the manifest"),
    "schema_other": (_set("schema", value="something/1"), "schema",
                     "not a mesh manifest"),
    "schema_null": (lambda card: card.clear() or card.update({"schema": None}),
                      "schema", "expected a string"),
    "format": (_set("mesh", "format", value="vtk"), "mesh.format", "not msh or mfem"),
    "nodes": (_set("mesh", "nodes", value="moved"), "mesh.nodes",
              "not reference or physical"),
    "no_read_options": (_drop("mesh", "read_options"), "mesh",
                        'no member "read_options"'),
    "displacement_type": (_set("mesh", "displacement", value=3), "mesh.displacement",
                          "expected a string"),
    "no_layers": (_set("layers", value=[]), "layers", "lists no layers"),
    "layers_not_a_list": (_set("layers", value={}), "layers", "expected an array"),
    "layer_attribute": (_set("layers", 1, "attribute", value=3),
                        "layers[1].attribute", "expected 2"),
    "layer_attribute_float": (_set("layers", 0, "attribute", value=1.0),
                              "layers[0].attribute", "expected an integer"),
    "layer_radii": (_set("layers", 0, "r_outer", value=0.0), "layers[0]",
                    "not 0 <= r_inner < r_outer"),
    "layer_gap": (_set("layers", 1, "r_inner", value=0.55), "layers[1]",
                  "the layer below ends at"),
    "layer_fluid": (_set("layers", 0, "fluid", value="yes"), "layers[0].fluid",
                    "expected a boolean"),
    "interface_count": (_drop("interfaces", 2), "interfaces", "a full mesh has one"),
    "hollow_without_face": (_set("layers", 0, "r_inner", value=0.1), "interfaces",
                            "a hollow one has a face"),
    "interface_attribute": (_set("interfaces", 1, "attribute", value=5),
                            "interfaces[1].attribute", "expected 2"),
    "between_layers": (_set("interfaces", 0, "between_layers", value=[0, 2]),
                       "interfaces[0].between_layers", "expected [0, 1]"),
    "kind": (_set("interfaces", 0, "kind", value="liquid"), "interfaces[0].kind",
             "not a kind of interface"),
    "values_pair": (_set("interfaces", 0, "values", "rho", value=[1.0]),
                    "interfaces[0].values.rho", "not [below, above]"),
    "values_unknown_field": (_set("interfaces", 0, "values", value={"mu": [1.0, 2.0]}),
                             "interfaces[0].values", "fields holds no record"),
    "refinement_size": (_set("refinements", 0, "far_size", value=0.01),
                        "refinements[0]", "not 0 < size <= far_size"),
    "refinement_decay": (_set("refinements", 0, "decay_width", value=-1.0),
                         "refinements[0].decay_width", "not positive"),
    "refinement_points": (_set("refinements", 0, "points", value=0),
                          "refinements[0].points", "not a count"),
    "field_repeated": (_set("fields", 1, "name", value="displacement"),
                       "fields[0].name", "is repeated"),
    "field_layers": (_set("fields", 1, "layers", value=[0]), "fields[1].layers",
                     "not an attribute in 1..3"),
    "field_degree": (_set("fields", 1, "radial_degree", value=-1),
                     "fields[1].radial_degree", "not a degree"),
    "field_weight": (_set("fields", 1, "weight", value=2), "fields[1]",
                     "has no character"),
    "field_vdim": (_set("fields", 1, "vdim", value=2), "fields[1].vdim",
                   "stores 1 components"),
    "field_missing_key": (_drop("fields", 1, "unit"), "fields[1]",
                          'no member "unit"'),
    "displacement_unknown": (_set("mesh", "displacement", value="u"),
                             "mesh.displacement", "fields holds no record"),
    "displacement_physical": (_set("mesh", "nodes", value="physical"),
                              "mesh.displacement", "physical nodes"),
    "model_record": (_drop("model", "schema"), "model", 'no member "schema"'),
    "scales_missing": (_drop("scales", "time"), "scales", 'no member "time"'),
    "scales_temperature": (_set("scales", "temperature", value="K"),
                           "scales.temperature", "expected a number"),
    "constants_value": (_set("constants", "G", value="one"), "constants.G",
                        "expected a number"),
    "meta_not_an_object": (_set("meta", value=[]), "meta", "expected an object"),
    "faults_not_a_list": (_set("faults", value={}), "faults", "expected an array"),
    "fault_missing_key": (_faulted(_drop("faults", 1, "meets")), "faults[1]",
                          'no member "meets"'),
    "fault_name_type": (_faulted(_set("faults", 0, "name", value=4)), "faults[0].name",
                        "expected a string"),
    "fault_name_repeated": (_faulted(_set("faults", 1, "name", value="megathrust")),
                            "faults[0].name", "is repeated"),
    "fault_name_interface": (_faulted(_set("faults", 1, "name", value="cmb")),
                             "faults[1].name", "shares its name with an interface"),
    "fault_attribute": (_faulted(_set("faults", 0, "attribute", value=3)),
                        "faults[0].attribute", "expected 4, after the 3 interfaces"),
    "fault_attributes_swapped": (_faulted(_set("faults", 0, "attribute", value=5),
                                          _set("faults", 1, "attribute", value=4)),
                                 "faults[0].attribute", "expected 4"),
    "fault_layers_shell": (_faulted(_set("faults", 0, "layers", value=[2, 3])),
                           "faults[0].layers", "not attributes of layers of the"),
    "fault_layers_collar": (_faulted(_set("faults", 0, "layers", value=[5])),
                            "faults[0].layers", "not attributes of layers of the"),
    "fault_layers_empty": (_faulted(_set("faults", 0, "layers", value=[])),
                           "faults[0].layers", "not attributes of layers of the"),
    "fault_layers_float": (_faulted(_set("faults", 0, "layers", value=[2.0])),
                           "faults[0].layers[0]", "expected an integer"),
    "fault_tip_group": (_faulted(_set("faults", 1, "tip", value=1)), "faults[1].tip",
                        "is 1, expected 4 or null"),
    "fault_tip_type": (_faulted(_set("faults", 0, "tip", value="1")), "faults[0].tip",
                       "expected a number"),
    "fault_free_edge_group": (_faulted(_set("faults", 0, "free_edge", value=5)),
                              "faults[0].free_edge", "expected 2 or null"),
    "fault_junction_group": (_faulted(_set("faults", 0, "junction", value=6)),
                             "faults[0].junction", "expected 3 or null"),
    "fault_meets_unknown": (_faulted(_set("faults", 0, "meets", value=["slab"])),
                            "faults[0].meets", "not the names of other faults"),
    "fault_meets_itself": (_faulted(_set("faults", 0, "meets", value=["megathrust"])),
                           "faults[0].meets", "not the names of other faults"),
    "fault_meets_type": (_faulted(_set("faults", 0, "meets", value=[1])),
                         "faults[0].meets[0]", "expected a string"),
    "fault_meets_one_way": (_faulted(_set("faults", 1, "meets", value=[]),
                                     _set("faults", 1, "junction", value=None)),
                            "faults[0].meets", "not every one of them meets"),
    "fault_junction_alone": (_faulted(_set("faults", 0, "meets", value=[]),
                                      _set("faults", 1, "meets", value=[])),
                             "faults[0].junction", "a junction is where it meets"),
    "fault_meets_without_junction": (
        _faulted(_set("faults", 0, "junction", value=None)), "faults[0].junction",
        "a junction is where it meets"),
    "fault_controls_small": (
        _faulted(_set("faults", 1, "controls", value=[[9, 9], [3, 17]])),
        "faults[1].controls", "each at least 4"),
    "fault_controls_shape": (_faulted(_set("faults", 0, "controls", value=[[17]])),
                             "faults[0].controls", "not [n_u, n_v] per patch"),
    "fault_controls_float": (
        _faulted(_set("faults", 0, "controls", value=[[17.0, 17]])),
        "faults[0].controls[0][0]", "expected an integer"),
    "fault_offset": (_faulted(_set("faults", 0, "offset", value=-1e-9)),
                     "faults[0].offset", "not a distance"),
    "fault_offset_type": (_faulted(_set("faults", 0, "offset", value="0")),
                          "faults[0].offset", "expected a number"),
    "fault_sizing": (_faulted(_set("faults", 0, "sizing", "size", value=0.5)),
                     "faults[0].sizing", "not 0 < size <= far_size"),
    "fault_sizing_decay": (
        _faulted(_set("faults", 0, "sizing", "decay_width", value=0)),
        "faults[0].sizing", "with a positive decay_width"),
    "fault_sizing_missing": (_faulted(_drop("faults", 0, "sizing", "far_size")),
                             "faults[0].sizing", 'no member "far_size"'),
    "schema_5_with_faults": (_faulted(_set("schema",
                                           value="planetmodel.mesh.manifest/5")),
                             "faults", "a manifest with faults is"),
    "schema_6_without_faults": (_set("schema", value="planetmodel.mesh.manifest/6"),
                                "faults", "has faults"),
    "fault_ends_on_unmet": (_faulted(_set("faults", 0, "ends_on", value=["slab"])),
                            "faults[0].ends_on", "not faults it meets"),
    "fault_ends_on_twice": (_faulted(_set("faults", 1, "ends_on",
                                          value=["megathrust", "megathrust"])),
                            "faults[1].ends_on", "each once"),
    "fault_ends_on_type": (_faulted(_set("faults", 1, "ends_on", value=[4])),
                           "faults[1].ends_on[0]", "expected a string"),
    "fault_hanging_wall_missing": (_faulted(_drop("faults", 0, "hanging_wall")),
                                   "faults[0]", 'no member "hanging_wall"'),
    "fault_hanging_wall_layer": (_faulted(_set("faults", 0, "hanging_wall",
                                               value=[2, 5])),
                                 "faults[0].hanging_wall", "not collar attributes"),
    "fault_hanging_wall_order": (_faulted(_set("faults", 1, "hanging_wall",
                                               value=[6, 4, 7])),
                                 "faults[1].hanging_wall", "in increasing order"),
    "layer_attributes_missing": (_faulted(_drop("layers", 1, "attributes")),
                                 "layers[1]", 'no member "attributes"'),
    "layer_attributes_own": (_faulted(_set("layers", 1, "attributes",
                                           value=[5, 2, 6, 7])),
                             "layers[1].attributes", "not its own attribute 2"),
    "layer_attributes_order": (_faulted(_set("layers", 1, "attributes",
                                             value=[2, 6, 5, 7])),
                               "layers[1].attributes", "in increasing order"),
    "layer_attributes_type": (_faulted(_set("layers", 1, "attributes",
                                            value=[2, 5.0, 6, 7])),
                              "layers[1].attributes[1]", "expected an integer"),
    "shell_collar": (_faulted(_set("layers", 2, "attributes", value=[3, 8])),
                     "layers[2].attributes", "a shell holds no fault"),
    "collar_twice": (_faulted(_set("layers", 0, "attributes", value=[1, 4, 5])),
                     "layers[1].attributes", "which layers[0] lists too"),
    "collar_gap": (_faulted(_set("layers", 1, "attributes", value=[2, 5, 6, 8]),
                            _set("faults", 1, "hanging_wall", value=[4, 6, 8])),
                   "layers", "not 4..7"),
    "collar_in_no_wall": (_faulted(_set("faults", 1, "hanging_wall", value=[4, 6])),
                          "layers[1].attributes", "in no fault's hanging wall"),
    "collar_order": (_faulted(_set("faults", 0, "hanging_wall", value=[6, 7]),
                              _set("faults", 1, "hanging_wall", value=[4, 5, 7])),
                     "layers[1].attributes", "out of the order of layer and faults"),
}


def manifest_cases(directory: Path) -> list[Path]:
    """The cases of the module docstring, under `manifest_cases/`."""
    root = directory / "manifest_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir(parents=True)

    def dump(name: str, card: dict[str, Any]) -> Path:
        path = root / f"{name}.json"
        path.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
        return path

    out = [dump("base", _BASE_MANIFEST)]
    out += [dump(f"accept_{label}", _edited(edit))
            for label, edit in _ACCEPTED_MANIFESTS.items()]
    lines = []
    for label, (edit, where, phrase) in _BAD_MANIFESTS.items():
        out.append(dump(f"bad_{label}", _edited(edit)))
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


#: The dimensions the units fixture covers: every named one, and powers
#: and products the named ones do not reach.
_DIMENSIONS = (units.DIMENSIONLESS, units.MASS, units.LENGTH, units.TIME,
               units.FREQUENCY, units.DENSITY, units.VELOCITY, units.GRAVITY,
               units.MODULUS, units.VISCOSITY, units.GRAVITATIONAL_CONSTANT,
               units.TEMPERATURE, units.VELOCITY ** 2,
               units.DENSITY * units.TEMPERATURE ** -1, units.LENGTH ** -7)


def frames(directory: Path) -> list[Path]:
    """The frames and reductions of the module docstring."""
    from planetmodel.frames import spherical_frame, wrapped
    rng = np.random.default_rng(3)
    theta = np.concatenate(([0.0, np.pi, 0.5 * np.pi], rng.uniform(0.0, np.pi, 6)))
    phi = np.concatenate(([0.0, 1.0, -2.0], rng.uniform(-7.0, 7.0, 6)))
    lines = []
    for t, p in zip(theta, phi):
        R = spherical_frame(t, p)
        lines.append(f"frame {float(t)!r} {float(p)!r} "
                     + " ".join(repr(float(x)) for x in R.ravel()))
    for x, start in ((-1e-17, 0.0), (2.0 * np.pi, 0.0), (-7.0, 0.3), (19.0, 0.3),
                     (0.3, 0.3), (0.3 - 1e-16, 0.3), (-100.5, 1.7)):
        lines.append(f"wrapped {x!r} {start!r} {float(wrapped(x, start=start))!r}")
    path = directory / "frames.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


def constants(directory: Path) -> list[Path]:
    """The facts of the module docstring, under the SI scales, the Earth's
    geophysical ones, and scales of no physical meaning with a
    temperature."""
    scales = {"si": pm.Scales.SI,
              "earth": pm.Scales.geophysical(6371e3),
              "odd": pm.Scales(length=3.5e4, mass=2.25e19, time=17.0,
                               temperature=273.15)}
    lines = [f"version {pm.__version__}", f"G_SI {units.G_SI!r}",
             f"EARTH_MEAN_DENSITY {units.EARTH_MEAN_DENSITY!r}"]
    for label, s in scales.items():
        lines.append(f"scales {label} {s.length!r} {s.mass!r} {s.time!r} "
                     f"{s.temperature!r}")
        for d in _DIMENSIONS:
            exps = f"{d.mass} {d.length} {d.time} {d.temperature}"
            lines.append(f"factor {label} {exps} {s.factor(d)!r}")
            lines.append(f"unit {label} {exps} {units.scaled_unit(d, s)}")
    path = directory / "constants.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


def _scalar_gradient(f: Any, r: float, t: float, p: float) -> np.ndarray:
    """(d/dr, d/dtheta, d/dphi) of a scalar field the reader builds, from
    the exact derivatives of its parts."""
    from planetmodel.gridded import GriddedField
    if getattr(f, "reference", None) is not None:
        ref, dp = f.reference, f.perturbation
        v0, g0 = float(ref.evaluate(r, t, p)), _scalar_gradient(ref, r, t, p)
        v1, g1 = float(dp.evaluate(r, t, p)), _scalar_gradient(dp, r, t, p)
        if f.kind == "absolute":
            return g0 + g1
        if f.kind == "relative":
            return g0 * (1.0 + v1) + v0 * g1
        return np.exp(v1) * (g0 + v0 * g1)
    if isinstance(f, GriddedField):
        _, dr, dt, dph = f.cartesian_derivatives(r, t, p)
        return np.array([dr.item(), dt.item(), dph.item()])
    return np.array([float(f.derivative().evaluate(r, t, p)), 0.0, 0.0])


def _expect_model(path: Path) -> Path:
    """The expectations of a model file, by the module docstring."""
    from planetmodel.modelfile import read_model, read_model_file, read_surfaces
    data, model = read_model_file(path), read_model(path, model_type=())
    hexed = (lambda x: "-" if not x else x.encode("utf-8").hex())
    s = data.scales
    lines = [f"scales {s.length!r} {s.mass!r} {s.time!r} {s.temperature!r}",
             f"rtol {data.rtol!r}", f"name {hexed(data.name)}",
             f"citation {hexed(chr(10).join(data.citation))}",
             f"behaviours {hexed(' '.join(data.behaviours))}"]
    lines += [f"layer_name {i} {hexed(n)}" for i, n in enumerate(data.layer_names)]
    lines += [f"interface_name {i} {hexed(n)}"
              for i, n in enumerate(data.interface_names)]
    for cname, value in data.constants.items():
        c = data.declared.get(cname)
        dims = "" if c is None else " " + " ".join(
            str(getattr(c.dimensions, b))
            for b in ("mass", "length", "time", "temperature"))
        lines.append(f"constant {cname} {value!r}{dims}")
    names = sorted({n for part in data.parts for n in part.variables
                    if not n.endswith(("_poles", "_perturbation"))
                    and n != "displacement"})
    for i, layer in enumerate(model.layers):
        lo, hi = layer.interval
        rs = (lo, lo + 0.3 * (hi - lo), lo + 0.77 * (hi - lo), hi)
        ts = (0.0, 0.7, np.pi, 2.2)
        ps = (-1.0, 0.3, 4.0, 9.5)
        for name in names:
            if name not in layer:
                lines.append(f"absent {i} {name}")
                continue
            f = layer[name]
            for r, t, p in zip(rs, ts, ps):
                for frame, tag in (("spherical", "s"), ("cartesian", "c")):
                    v = np.asarray(f.evaluate(r, t, p, frame=frame),
                                   dtype=float).ravel()
                    lines.append(f"value {i} {name} {tag} {r!r} {t!r} {p!r} "
                                 + " ".join(repr(float(x)) for x in v))
                if f.character.rank == 0:
                    g = _scalar_gradient(f, r, t, p)
                    lines.append(f"gradient {i} {name} {r!r} {t!r} {p!r} "
                                 + " ".join(repr(float(x)) for x in g))
    mapping = model.geometry.mapping
    if not model.geometry.is_identity:
        rng = np.random.default_rng(11)
        top = float(model.skeleton.boundaries[-1])
        radii = np.concatenate((rng.uniform(0.0, top, 6), model.skeleton.boundaries))
        t = np.arccos(rng.uniform(-1, 1, radii.size))
        ph = rng.uniform(-np.pi, np.pi, radii.size)
        X = np.stack([radii * np.sin(t) * np.cos(ph), radii * np.sin(t) * np.sin(ph),
                      radii * np.cos(t)], -1)
        U = mapping.displacement(X)
        for x, u in zip(X, U):
            lines.append("displacement " + " ".join(repr(float(a)) for a in (*x, *u)))
    for name, on in sorted(read_surfaces(path).items()):
        for face in model.geometry.interfaces:
            k = face.index
            f = on.get(k if face.name is None else face.name)
            if f is None:
                lines.append(f"surface_absent {k} {name}")
                continue
            for t, p in ((0.0, -1.0), (0.7, 0.3), (np.pi, 4.0), (2.2, 9.5),
                         (1.3, 2 * np.pi), (0.05, 1.0)):
                lines.append(f"surface {k} {name} {t!r} {p!r} {float(f(t, p))!r}")
            for t, p in ((0.7, 0.3), (2.2, 9.5), (1.3, 2 * np.pi)):
                dt, dp = (float(d) for d in f.angular_gradient(t, p))
                lines.append(f"surface_gradient {k} {name} {t!r} {p!r} {dt!r} {dp!r}")
    out = path.with_name(path.name.removesuffix(".nc") + ".expect")
    out.write_text("\n".join(lines) + "\n")
    return out


def _setting(key: str, value: Any, *, group: str | None = None,
             variable: str | None = None) -> Any:
    """An edit setting one attribute: of the root, a group, or a variable."""
    def edit(ds: Any) -> None:
        g = ds if group is None else ds.groups[group]
        (g if variable is None else g.variables[variable]).setncattr(key, value)
    return edit


def _deleting(key: str, *, group: str | None = None,
              variable: str | None = None) -> Any:
    """An edit deleting one attribute: of the root, a group, or a variable."""
    def edit(ds: Any) -> None:
        g = ds if group is None else ds.groups[group]
        (g if variable is None else g.variables[variable]).delncattr(key)
    return edit


def _storing(index: tuple[int, ...], value: float, *, variable: str,
             group: str | None = None, relative: bool = False) -> Any:
    """An edit storing one value of a variable, or scaling it."""
    def edit(ds: Any) -> None:
        var = (ds if group is None else ds.groups[group]).variables[variable]
        var[index] = value * float(var[index]) if relative else value
    return edit


def _adding_half(key: str, *, group: str) -> Any:
    """An edit adding a half to a whole number a group states."""
    def edit(ds: Any) -> None:
        g = ds.groups[group]
        g.setncattr(key, float(g.getncattr(key)) + 0.5)
    return edit


def _renaming(old: str, new: str, *, group: str) -> Any:
    """An edit renaming a variable of a group."""
    return lambda ds: ds.groups[group].renameVariable(old, new)


def _negating(*, variable: str, group: str) -> Any:
    """An edit storing -1 at the first node holding a variable."""
    def edit(ds: Any) -> None:
        var = ds.groups[group].variables[variable]
        held = np.flatnonzero(~np.isnan(np.ma.filled(var[:], np.nan)))
        var[int(held[0])] = -1.0
    return edit


def _relabelling(*, group: str) -> Any:
    """An edit giving the first element of a part's second layer to its
    first: `element_layer` stays sorted and holds the part's layers, and
    gives that element a layer its bounds lie outside."""
    def edit(ds: Any) -> None:
        var = ds.groups[group].variables["element_layer"]
        layers = np.asarray(var[:], dtype=int)
        second = np.flatnonzero(layers == np.unique(layers)[1])
        if second.size < 2:
            raise AssertionError(f"the second layer of {group} has one element")
        var[int(second[0])] = layers[0]
    return edit


def _unwriting(index: tuple[int, ...], *, variable: str, group: str | None = None
               ) -> Any:
    """An edit storing a variable's fill at one place, as if no value had
    been written there."""
    def edit(ds: Any) -> None:
        import netCDF4
        var = (ds if group is None else ds.groups[group]).variables[variable]
        var.set_auto_mask(False)
        var[index] = (var.getncattr("_FillValue") if "_FillValue" in var.ncattrs()
                      else netCDF4.default_fillvals[var.dtype.str[1:]])
    return edit


def _creating(name: str, dims: tuple[str, ...], *, group: str | None = None,
              attributes: dict[str, Any] | None = None,
              sizes: dict[str, int] | None = None) -> Any:
    """An edit creating a real variable of zeros over dimensions of the
    root or a group, those of `sizes` created first."""
    def edit(ds: Any) -> None:
        g = ds if group is None else ds.groups[group]
        for dim, n in (sizes or {}).items():
            g.createDimension(dim, n)
        var = g.createVariable(name, "f8", dims)
        var[...] = 0.0
        for key, value in (attributes or {}).items():
            var.setncattr(key, value)
    return edit


def _adding_dimension(name: str, *, group: str | None = None, size: int = 2) -> Any:
    """An edit adding a dimension of `size` to the root or a group."""
    return lambda ds: (ds if group is None else ds.groups[group]).createDimension(
        name, size)


class _Rebuilding:
    """A malformed copy that no edit in place makes, netCDF holding on to
    what a file has: the file copied by netCDF4 without the variables and
    dimensions `without` names (`<group>/<name>`, or `<name>` at the
    root), each dimension of `sizes` (named the same way) made that long
    and the variables over it cut to fit, and `then`, an edit, applied to
    the copy.  A length of zero makes the dimension unlimited and empty."""

    def __init__(self, *without: str, sizes: dict[str, int] | None = None,
                 then: Any = None) -> None:
        self.without, self.sizes, self.then = set(without), dict(sizes or {}), then

    def __call__(self, source: Path, target: Path) -> None:
        import netCDF4

        def copy(old: Any, new: Any, prefix: str) -> None:
            new.setncatts({key: old.getncattr(key) for key in old.ncattrs()})
            for name, dim in old.dimensions.items():
                if prefix + name not in self.without:
                    new.createDimension(name, self.sizes.get(prefix + name, len(dim)))
            for name, var in old.variables.items():
                if prefix + name in self.without:
                    continue
                var.set_auto_mask(False)
                attrs = {key: var.getncattr(key) for key in var.ncattrs()}
                out = new.createVariable(name, var.dtype, var.dimensions,
                                         fill_value=attrs.pop("_FillValue", None))
                out.set_auto_mask(False)
                kept = tuple(slice(0, self.sizes.get(prefix + d, len(old.dimensions[d]))
                                   if d in old.dimensions else None)
                             for d in var.dimensions)
                out[...] = var[...][kept]
                out.setncatts(attrs)
            for name, sub in old.groups.items():
                copy(sub, new.createGroup(name), f"{prefix}{name}/")

        with netCDF4.Dataset(source) as old, netCDF4.Dataset(target, "w") as new:
            copy(old, new, "")
            if self.then is not None:
                self.then(new)


_P = {"group": "part_0"}
_RHO = {"group": "part_0", "variable": "rho"}
_BETWEEN = {"group": "geometry", "variable": "boundary_displacement"}
_S = {"group": "surfaces"}

#: What the refusal of a grid with a latitude beyond a pole says.
_LATITUDES = "the grid is malformed: latitudes must lie in [-90, 90]"

#: What the refusal of a field's `dimensions` that are not four whole
#: numbers says.
_EXPONENTS = "declares dimensions that are not four exponents"

#: The malformed model files: (the file copied, an edit by netCDF4, where
#: the fault lies, what the refusal says).
_BAD_MODELS: dict[str, tuple[str, Any, str, str]] = {
    "schema": ("model_lateral.nc", _setting("schema", "planetmodel.model/0"),
               "schema", "this reader reads"),
    "no_schema": ("model_lateral.nc", _deleting("schema"), "schema",
                  "not a planetmodel model file"),
    "rtol_negative": ("model_lateral.nc", _setting("rtol", -1.0), "rtol",
                      "not a tolerance"),
    "rtol_text": ("model_lateral.nc", _setting("rtol", "loose"), "rtol",
                  "not a tolerance"),
    "behaviours_number": ("model_lateral.nc", _setting("behaviours", np.int32(4)),
                          "behaviours", "not the names of the model's behaviours"),
    "behaviours_twice": ("model_lateral.nc",
                         _setting("behaviours", "Elastic Tidal Elastic"),
                         "behaviours", "each once"),
    "no_scale_mass": ("model_lateral.nc", _deleting("scale_mass"), "scale_mass",
                      "missing or not positive"),
    "layer_bounds": ("model_lateral.nc",
                     _storing((1, 0), 0.55, variable="layer_bounds"),
                     "layer_bounds", "contiguous"),
    "layer_part": ("model_lateral.nc", _storing((0,), 3, variable="layer_part"),
                   "layer_part", "part_3"),
    "grid_kind": ("model_lateral.nc", _setting("grid_kind", "healpix", **_P),
                  "part_0.grid_kind", "not a kind of grid"),
    "ngll": ("model_lateral.nc", _setting("ngll", np.int32(7), **_P), "part_0",
             "elements of ngll"),
    "element_bounds": ("model_lateral.nc",
                       _storing((0, 1), 0.3, variable="element_bounds", **_P),
                       "part_0", "do not tile"),
    "radius": ("model_lateral.nc",
               _storing((1,), 1.001, variable="radius", relative=True, **_P),
               "part_0/radius", "Gauss-Lobatto-Legendre"),
    "element_bounds_end": ("model_lateral.nc",
                           _storing((-1, 1), 1.0 + 1e-7, variable="element_bounds",
                                    relative=True, **_P),
                           "part_0", "do not tile"),
    "grid_kind_number": ("model_lateral.nc", _setting("grid_kind", np.int32(3), **_P),
                         "part_0.grid_kind", "not a kind of grid"),
    "radial_rule": ("model_lateral.nc", _deleting("radial_interpolation", **_RHO),
                    "part_0/rho", "lacks the attribute"),
    "radial_rule_number": ("model_lateral.nc",
                           _setting("radial_interpolation", np.int32(1), **_RHO),
                           "part_0/rho", "unknown radial rule"),
    "rank_text": ("model_lateral.nc", _setting("rank", "two", **_RHO), "part_0/rho",
                  "states no character"),
    "displacement_rule": ("model_flattened.nc",
                          _deleting("radial_interpolation", group="part_0",
                                    variable="displacement"),
                          "part_0/displacement", "lacks the attribute"),
    "perturbation_number": ("model_kinds.nc",
                            _setting("perturbation", np.int32(1), **_RHO),
                            "part_0/rho", "which the part lacks"),
    "lateral_rule": ("model_lateral.nc",
                     _setting("lateral_interpolation", "quintic", **_RHO),
                     "part_0/rho", "no known lateral rule"),
    "rank": ("model_lateral.nc", _setting("rank", np.int32(1), **_RHO), "part_0/rho",
             "trailing axes"),
    "radial_layers_unheld": ("model_lateral.nc",
                             _setting("radial_layers", np.int32([5]), **_RHO),
                             "part_0/rho", "radial_layers"),
    "radial_layers_radial": ("model_lateral.nc",
                             _setting("radial_layers", np.int32([0]), group="part_0",
                                      variable="vp"),
                             "part_0/vp", "radial_layers"),
    "poles_missing": ("model_lateral.nc", _renaming("rho_poles", "rho_p", **_P),
                      "part_0/rho", "rho_poles is missing"),
    "perturbation_kind": ("model_kinds.nc",
                          _setting("perturbation_kind", "squared", **_RHO),
                          "part_0/rho", "perturbation_kind"),
    "between": ("model_stacked.nc", _setting("between", "cubic", **_BETWEEN),
                "geometry/boundary_displacement", "between surfaces"),
    "surface_interface": ("model_surfaces.nc",
                          _storing((1,), 5, variable="surface_interface", **_S),
                          "surfaces/surface_interface", "not interfaces of the file"),
    "surface_order": ("model_surfaces.nc",
                      _storing((0,), 1, variable="surface_interface", **_S),
                      "surfaces/surface_interface", "not interfaces of the file"),
    "surface_grid_kind": ("model_surfaces.nc", _setting("grid_kind", "healpix", **_S),
                          "surfaces.grid_kind", "not a kind of grid"),
    "surface_rule": ("model_surfaces.nc",
                     _setting("lateral_interpolation", "quintic", **_S,
                              variable="K_shape"),
                     "surfaces/K_shape", "no known lateral rule"),
    "surface_holders": ("model_surfaces.nc",
                        _setting("interfaces", np.int32([0, 3]), **_S,
                                 variable="shift"),
                        "surfaces/shift", "not surfaces of the group"),
    "surface_poles_missing": ("model_surfaces.nc",
                              _renaming("K_shape_poles", "K_shape_p", **_S),
                              "surfaces/K_shape", "K_shape_poles is missing"),
    "surface_name_taken": ("model_surfaces.nc", _renaming("shift", "rho", **_S),
                           "surfaces/rho", "never both"),
    "log_rule_schema": ("model_log.nc", _setting("schema", "planetmodel.model/1"),
                        "part_0/viscosity", "does not hold"),
    "log_schema_unused": ("model_lateral.nc", _setting("schema", "planetmodel.model/2"),
                          "schema", "and none does"),
    "log_rule_lateral": ("model_lateral.nc",
                         _setting("radial_interpolation", "log-polynomial", **_RHO),
                         "part_0/rho", "rank 0 of the radius alone"),
    "log_rule_value": ("model_log.nc", _negating(variable="viscosity", group="part_0"),
                       "part_0/viscosity", "not positive"),
    "layer_part_fill": ("model_prem.nc", _unwriting((0,), variable="layer_part"),
                        "layer_part", "which the file lacks"),
    "element_layer_fill": ("model_prem.nc",
                           _unwriting((0,), variable="element_layer", **_P),
                           "part_0", "do not lie in its layers"),
    "element_layer": ("model_prem.nc", _relabelling(**_P), "part_0/element_layer",
                      "which its bounds do not lie in"),
    "scale_infinite": ("model_lateral.nc", _setting("scale_length", np.inf),
                       "scale_length", "not positive and finite"),
    "interface_radius": ("model_lateral.nc",
                         _storing((0,), 0.25, variable="interface_radius"),
                         "interface_radius", "interfaces of the skeleton"),
    "ngll_fraction": ("model_lateral.nc", _adding_half("ngll", **_P), "part_0.ngll",
                      "whole number"),
    "harmonic_custom": ("model_lateral.nc",
                        _both(_setting("lateral_interpolation", "harmonic", **_RHO),
                              _setting("grid_kind", "custom", **_P)),
                        "part_0/rho", "states the lateral rule harmonic"),
    "harmonic_band": ("model_lateral.nc",
                      _both(_setting("lateral_interpolation", "harmonic", **_RHO),
                            _deleting("grid_lmax", **_P)),
                      "part_0/rho", "states no grid_lmax"),
    "stack_harmonic": ("model_stacked.nc",
                       _both(_setting("lateral_interpolation", "harmonic", **_BETWEEN),
                             _setting("grid_kind", "custom", group="geometry")),
                       "geometry/boundary_displacement",
                       "states the lateral rule harmonic"),
    "surface_harmonic": ("model_surfaces.nc",
                         _both(_setting("lateral_interpolation", "harmonic", **_S,
                                        variable="K_shape"),
                               _setting("grid_kind", "custom", **_S)),
                         "surfaces/K_shape", "states the lateral rule harmonic"),
    "cubic_thin": ("model_cases/thin.nc",
                   _setting("lateral_interpolation", "cubic", **_RHO),
                   "part_0/rho", "states the lateral rule cubic"),
    "poles_absent": ("model_cases/thin.nc",
                     _Rebuilding("part_0/rho_poles", "part_0/pole"),
                     "part_0/rho", "rho_poles is missing"),
    "poles_size": ("model_cases/thin.nc",
                   _Rebuilding("part_0/rho_poles", "part_0/pole",
                               then=_creating("rho_poles", ("node", "pole"),
                                              sizes={"pole": 3}, **_P)),
                   "part_0/rho_poles", "does not hold both poles at every node"),
    "poles_dimensions": ("model_cases/thin.nc",
                         _Rebuilding("part_0/rho_poles",
                                     then=_creating("rho_poles", ("pole", "node"),
                                                    **_P)),
                         "part_0/rho_poles", "does not hold both poles at every node"),
    "displacement_poles_absent": ("model_flattened.nc",
                                  _Rebuilding("part_0/displacement_poles",
                                              "part_0/pole"),
                                  "part_0/displacement",
                                  "displacement_poles is missing"),
    "stack_poles_absent": ("model_stacked.nc",
                           _Rebuilding("geometry/boundary_displacement_poles",
                                       "geometry/pole"),
                           "geometry/boundary_displacement",
                           "boundary_displacement_poles is missing"),
    "stack_poles_size": ("model_stacked.nc",
                         _Rebuilding("geometry/boundary_displacement_poles",
                                     "geometry/pole",
                                     then=_creating("boundary_displacement_poles",
                                                    ("boundary", "pole"),
                                                    sizes={"pole": 1},
                                                    group="geometry")),
                         "geometry/boundary_displacement",
                         "boundary_displacement_poles is missing"),
    "stack_poles_dimensions": ("model_stacked.nc",
                               _Rebuilding("geometry/boundary_displacement_poles",
                                           then=_creating(
                                               "boundary_displacement_poles",
                                               ("pole", "boundary"),
                                               group="geometry")),
                               "geometry/boundary_displacement_poles",
                               "has the wrong dimensions"),
    "surface_poles_absent": ("model_surfaces.nc",
                             _Rebuilding("surfaces/K_shape_poles",
                                         "surfaces/shift_poles", "surfaces/pole"),
                             "surfaces/K_shape", "K_shape_poles is missing"),
    "surface_poles_size": ("model_surfaces.nc",
                           _Rebuilding("surfaces/K_shape_poles",
                                       "surfaces/shift_poles", "surfaces/pole",
                                       then=_both(
                                           _creating("K_shape_poles",
                                                     ("surface", "pole"),
                                                     sizes={"pole": 3}, **_S),
                                           _creating("shift_poles",
                                                     ("surface", "pole"), **_S))),
                           "surfaces/K_shape", "K_shape_poles is missing"),
    "stack_radius_order": ("model_stacked.nc",
                           _storing((1,), 1.5, variable="boundary_radius",
                                    group="geometry"),
                           "geometry/boundary_radius", "does not increase strictly"),
    "stack_one_surface": ("model_stacked.nc",
                          _Rebuilding(sizes={"geometry/boundary": 1}),
                          "geometry/boundary_radius", "fewer than two surfaces"),
    "layer_bounds_negative": ("model_lateral.nc",
                              _storing((0, 0), -0.1, variable="layer_bounds"),
                              "layer_bounds", "do not make a skeleton"),
    "layer_bounds_infinite": ("model_lateral.nc",
                              _storing((-1, 1), np.inf, variable="layer_bounds"),
                              "layer_bounds", "do not make a skeleton"),
    "no_layers": ("model_lateral.nc", _Rebuilding(sizes={"layer": 0}),
                  "layer_bounds", "do not make a skeleton of contiguous layers"),
    "latitude_range": ("model_lateral.nc",
                       _storing((0,), 90.5, variable="latitude", **_P),
                       "part_0", _LATITUDES),
    "stack_latitude_range": ("model_stacked.nc",
                             _storing((-1,), -90.5, variable="latitude",
                                      group="geometry"),
                             "geometry", _LATITUDES),
    "surface_latitude_range": ("model_surfaces.nc",
                               _storing((0,), 90.5, variable="latitude", **_S),
                               "surfaces", _LATITUDES),
}


def _thin_model(path: Path) -> Path:
    """A two-layer model file whose density varies with direction, on a grid
    of two longitudes and stating the bilinear rule: valid, and on a grid
    that cannot carry the bicubic rule."""
    grid = pm.equiangular(6, 2)
    two = pm.LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0],
                                     vp=[9.0, 7.0], vs=[4.0, 3.0])
    rho = pm.GriddedField.sampled(pm.AnalyticField(
        (0.5, 1.0), lambda r, t, p: 4.0 + 0.1 * r * np.cos(t), character=pm.DENSITY),
        grid, ngll=3)
    write_model(path, two.with_field(1, "rho", rho, replace=True), grid=grid,
                fields=["rho"])
    validate_structure(path)
    return path


def _malformed(directory: Path, root: Path,
               cases: dict[str, tuple[str, Any, str, str]]) -> list[Path]:
    """The malformed copies `bad_<label>.nc` of `cases` in `root`, each a
    file of `directory` edited by netCDF4 in place, or rebuilt by a
    `_Rebuilding`, and their index `bad.txt`: a case, a tab, where the
    fault lies, a tab, and what the refusal says."""
    import netCDF4
    out, lines = [], []
    for label, (base, edit, where, phrase) in cases.items():
        path = root / f"bad_{label}.nc"
        if isinstance(edit, _Rebuilding):
            edit(directory / base, path)
        else:
            shutil.copyfile(directory / base, path)
            with netCDF4.Dataset(path, "a") as ds:
                edit(ds)
        out.append(path)
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


def model_cases(directory: Path) -> list[Path]:
    """The harmonic copy, the expectations, and the malformed copies of the
    model files already in `directory` and of `model_cases/thin.nc`, written
    here."""
    import netCDF4
    out = []
    harmonic = directory / "model_harmonic.nc"
    shutil.copyfile(directory / "model_lateral.nc", harmonic)
    with netCDF4.Dataset(harmonic, "a") as ds:
        _setting("lateral_interpolation", "harmonic", **_RHO)(ds)
    out.append(harmonic)
    for path in sorted(directory.glob("model_*.nc")):
        if path != harmonic:
            out.append(_expect_model(path))
    root = directory / "model_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    out.append(_thin_model(root / "thin.nc"))
    return out + _malformed(directory, root, _BAD_MODELS)


def gll(directory: Path) -> list[Path]:
    """The Gauss-Lobatto-Legendre points of the module docstring."""
    from planetmodel.mesh1d.gll import gll_points_weights
    lines = [f"gll {n} " + " ".join(repr(float(x)) for x in gll_points_weights(n)[0])
             for n in range(2, 14)]
    path = directory / "gll.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


#: The shipped grids `grids.txt` holds, by the C++ name of each.
_GRIDS = {
    "gl:0": lambda: pm.gauss_legendre(0),
    "gl:1": lambda: pm.gauss_legendre(1),
    "gl:5": lambda: pm.gauss_legendre(5),
    "gl:5:16": lambda: pm.gauss_legendre(5, nphi=16),
    "gl:64": lambda: pm.gauss_legendre(64),
    "gl:255": lambda: pm.gauss_legendre(255),
    "dh:0:1": lambda: pm.driscoll_healy(0),
    "dh:6:1": lambda: pm.driscoll_healy(6),
    "dh:6:2": lambda: pm.driscoll_healy(6, sampling=2),
    "dh:63:1": lambda: pm.driscoll_healy(63),
    "eq:1:1:0": lambda: pm.equiangular(1, 1),
    "eq:8:16:0": lambda: pm.equiangular(8, 16),
    "eq:9:4:0": lambda: pm.equiangular(9, 4),
    "eq:2:3:1": lambda: pm.equiangular(2, 3, poles=True),
    "eq:9:16:1": lambda: pm.equiangular(9, 16, poles=True),
    "eq:64:128:1": lambda: pm.equiangular(64, 128, poles=True)}


def grids(directory: Path) -> list[Path]:
    """The shipped grids of the module docstring."""
    lines = []
    for label, make in _GRIDS.items():
        g = make()
        lines.append(f"band {label} {g.kind} {g.lmax}")
        for key, values in (("colatitudes", g.colatitudes),
                            ("longitudes", g.longitudes), ("weights", g.weights)):
            lines.append(f"{key} {label} " + " ".join(repr(float(x)) for x in values))
    path = directory / "grids.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


def vocabulary(directory: Path) -> list[Path]:
    """The vocabulary of the module docstring."""
    from planetmodel.vocabulary import FAMILIES, VOCABULARY
    lines = []
    for prefix, table in (("", VOCABULARY), ("family ", FAMILIES)):
        for name, spec in table.items():
            c, d = spec.character, spec.dimensions
            lines.append(f"{prefix}{name} {c.rank} {c.weight} {int(bool(c.voigt))} "
                         f"{d.mass} {d.length} {d.time} {d.temperature} "
                         f"{spec.meaning.encode('utf-8').hex()}")
    path = directory / "vocabulary.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


def _as_long_as(new: str, old: str) -> Any:
    """An edit adding the root a dimension `new` of the length of `old`."""
    return lambda ds: ds.createDimension(new, len(ds.dimensions[old]))


#: The attributes of a static field of a surface file.
_STATIC = {"time_support": "static", "time_interpolation": "none",
           "lateral_interpolation": "linear"}

#: The malformed surface files: (the file copied, an edit by netCDF4, where
#: the fault lies, what the refusal says).
_BAD_SURFACES: dict[str, tuple[str, Any, str, str]] = {
    "transposed": ("surface_basic.nc",
                   _creating("turned", ("longitude", "latitude"), attributes=_STATIC),
                   "turned", "does not run over its support and the grid"),
    "other_axis": ("surface_basic.nc",
                   _both(_as_long_as("epoch", "time"),
                         _creating("drift", ("epoch", "latitude", "longitude"),
                                   attributes={**_STATIC, "time_support": "node"})),
                   "drift", "does not run over its support and the grid"),
    "poles_axis": ("surface_basic.nc",
                   _both(_as_long_as("epoch", "time"),
                         _creating("load_nodes_poles", ("epoch", "pole"))),
                   "load_nodes_poles", "does not run over the support of load_nodes"),
    "scale_infinite": ("surface_basic.nc", _setting("scale_time", np.inf),
                       "scale_time", "not positive and finite"),
    "harmonic_custom": ("surface_basic.nc",
                        _both(_setting("lateral_interpolation", "harmonic",
                                       variable="topography"),
                              _setting("grid_kind", "custom")),
                        "topography", "states the lateral rule harmonic"),
    "harmonic_band": ("surface_basic.nc",
                      _both(_setting("lateral_interpolation", "harmonic",
                                     variable="topography"),
                            _deleting("grid_lmax")),
                      "topography", "states no grid_lmax"),
    "poles_on_both": ("surface_poles.nc",
                      _creating("relief_poles", ("pole",), sizes={"pole": 2}),
                      "relief_poles",
                      "gives polar values on a grid holding both poles"),
    "pole_length": ("surface_poles.nc", _adding_dimension("pole", size=3), "pole",
                    "does not run over the two poles"),
    "latitude_range": ("surface_basic.nc",
                       _storing((0,), 90.5, variable="latitude"), "", _LATITUDES),
    "source_number": ("surface_basic.nc", _setting("source", 3.0), "source",
                      "is missing or empty"),
    "created_number": ("surface_basic.nc", _setting("created", np.int32(2026)),
                       "created", "is missing or empty"),
    "dimensions_three": ("surface_basic.nc",
                         _setting("dimensions", np.asarray([0, 1, 0], dtype=np.int32),
                                  variable="topography"),
                         "topography", _EXPONENTS),
    "dimensions_text": ("surface_basic.nc",
                        _setting("dimensions", "length", variable="topography"),
                        "topography", _EXPONENTS),
    "dimensions_fraction": ("surface_basic.nc",
                            _setting("dimensions", np.asarray([0.0, 0.5, 0.0, 0.0]),
                                     variable="topography"),
                            "topography", _EXPONENTS),
}


def _expect_surface(path: Path, fields: dict[str, Any], *,
                    dated: tuple[tuple[str, float | None, str], ...],
                    functions: tuple[tuple[str, float | None, str], ...]) -> Path:
    """The expectations of the surface file at `path`, written from
    `fields`, beside it (module docstring): the Python's reading of the
    file, its history's values at each (name, date, extrapolation) of
    `dated`, and the field of each of `functions` there as a function of
    (theta, phi), with its two derivatives."""
    from planetmodel import surfacefile

    data = surfacefile.read_surface_file(path)
    grid, scales = data.grid, data.scales
    lines = [f"scales {scales.length!r} {scales.mass!r} {scales.time!r} "
             f"{scales.temperature!r}"]
    if data.radius is not None:
        lines.append(f"radius {data.radius!r}")
    lines.append(f"grid {grid.kind} {grid.lmax} {grid.ntheta} {grid.nphi}")
    for axis in ("time", "instant"):
        if getattr(data, axis) is not None:
            lines += [f"{axis} {k} {float(t)!r}"
                      for k, t in enumerate(getattr(data, axis))]
    samples = ((0, 1), (grid.ntheta // 2, grid.nphi // 3),
               (grid.ntheta - 1, grid.nphi - 1), (0, 0))
    for name, field in fields.items():
        a = data.variables[name].attributes
        exps = ("-" if "dimensions" not in a else
                " ".join(str(int(e)) for e in np.atleast_1d(a["dimensions"])))
        lines.append(f"variable {name} {a['time_support']} "
                     f"{a['time_interpolation']} {a['lateral_interpolation']} "
                     f"{exps}")
        values = data.variables[name].values
        leading = [None] if values.ndim == 2 else [0, values.shape[0] - 1]
        for k in leading:
            v = values if k is None else values[k]
            tag = "-" if k is None else str(k)
            for i, j in samples:
                if np.isnan(v[i, j]):
                    lines.append(f"absent {name} {tag} {i} {j}")
                else:
                    lines.append(f"value {name} {tag} {i} {j} {float(v[i, j])!r}")
        poles_name = f"{name}_poles"
        if poles_name in data.variables:
            p = np.atleast_2d(data.variables[poles_name].values)
            for k in ([None] if field.support == "static" else range(p.shape[0])):
                row = p[0] if k is None else p[k]
                tag = "-" if k is None else str(k)
                lines.append(f"poles {name} {tag} {float(row[0])!r} {float(row[1])!r}")
    history = surfacefile.read_surface(path)
    for name, date, extrapolation in dated:
        at = history.values(name, date=date, extrapolation=extrapolation)
        tag = "-" if date is None else repr(float(date))
        for i, j in samples:
            if np.isfinite(at[i, j]):
                lines.append(f"sample {name} {tag} {extrapolation} {i} {j} "
                             f"{float(at[i, j])!r}")
    # the fields as functions of (theta, phi): both poles, the seam, a node
    # and points between nodes
    phi0 = float(grid.longitudes[0])
    points = ((0.0, 0.3), (np.pi, 1.1), (1.0, phi0 - 1e-9), (0.7, 2.0), (2.9, -1.0),
              (0.05, 5.0), (float(grid.colatitudes[1]), float(grid.longitudes[2])))
    for name, date, extrapolation in functions:
        f = history.function(name, date=date, extrapolation=extrapolation)
        tag = "-" if date is None else repr(float(date))
        for theta, phi in points:
            dt, dp = (float(v) for v in f.angular_gradient(theta, phi))
            lines.append(f"function {name} {tag} {extrapolation} {theta!r} {phi!r} "
                         f"{float(f(theta, phi))!r} {dt!r} {dp!r}")
    expect = path.with_name(path.name.removesuffix(".nc") + ".expect")
    expect.write_text("\n".join(lines) + "\n")
    return expect


def surface_files(directory: Path) -> list[Path]:
    """The surface files of the module docstring, their expectations, and
    the malformed copies of them (`surface_cases/`), each refused by the
    Python's reader as it is written."""
    import netCDF4

    from planetmodel import surfacefile

    grid = pm.gauss_legendre(8)
    rng = np.random.default_rng(11)
    time = np.array([0.0, 1.0, 2.5, 4.0])
    instant = np.array([2.5, 4.0])
    shape = (grid.ntheta, grid.nphi)
    topography = rng.normal(size=shape)
    topography[0, 0] = np.nan
    fields = {
        "topography": surfacefile.SurfaceVariable(
            topography, dimensions=units.LENGTH,
            meaning="present-day topography", poles=[1.0, -2.0]),
        "ice_thickness": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node",
            time_interpolation="linear", dimensions=units.LENGTH,
            poles=rng.normal(size=(time.size, 2))),
        "load_previous": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node",
            time_interpolation="previous"),
        "load_nodes": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node"),
        "rate": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size - 1,) + shape), support="interval"),
        "instantaneous": surfacefile.SurfaceVariable(
            rng.normal(size=(instant.size,) + shape), support="instant",
            lateral_interpolation="cubic"),
        "geoid": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node",
            time_interpolation="linear", lateral_interpolation="cubic",
            poles=rng.normal(size=(time.size, 2))),
    }
    scales = pm.Scales.geophysical(6.371e6)
    path = directory / "surface_basic.nc"
    surfacefile.write_surface(path, grid, fields=fields,
                              source="planetmodel fixtures", time=time,
                              instant=instant, radius=1.0, scales=scales,
                              meta={"epoch": "present"}, citation=("a work",))
    surfacefile.validate_structure(path)
    out = [path, _expect_surface(
        path, fields,
        dated=(("topography", None, "refuse"), ("ice_thickness", 1.75, "refuse"),
               ("ice_thickness", 2.5, "refuse"), ("ice_thickness", 9.0, "hold"),
               ("load_previous", 1.75, "refuse"), ("load_previous", -3.0, "hold"),
               ("load_nodes", 2.5, "refuse"), ("rate", 3.0, "refuse"),
               ("rate", 4.0, "refuse"), ("instantaneous", 2.5, "refuse")),
        functions=(("ice_thickness", 1.75, "refuse"), ("ice_thickness", 2.5, "refuse"),
                   ("ice_thickness", 9.0, "hold"), ("geoid", 1.75, "refuse"),
                   ("geoid", -3.0, "hold")))]
    # a grid holding both poles, whose fields carry no polar values
    both = pm.equiangular(7, 12, poles=True)
    theta, phi = both.nodes
    relief = smooth(theta, phi)
    held = {
        "relief": surfacefile.SurfaceVariable(
            relief, dimensions=units.LENGTH, lateral_interpolation="cubic"),
        "ice_thickness": surfacefile.SurfaceVariable(
            np.stack([relief, 2.0 * relief + np.cos(theta)]), support="node",
            time_interpolation="linear", dimensions=units.LENGTH),
    }
    path = directory / "surface_poles.nc"
    surfacefile.write_surface(path, both, fields=held, time=[0.0, 2.0])
    surfacefile.validate_structure(path)
    out += [path, _expect_surface(
        path, held,
        dated=(("relief", None, "refuse"), ("ice_thickness", 0.5, "refuse")),
        functions=(("relief", None, "refuse"), ("ice_thickness", 0.5, "refuse"),
                   ("ice_thickness", 5.0, "hold")))]
    root = directory / "surface_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    lines = []
    for label, (base, edit, where, phrase) in _BAD_SURFACES.items():
        bad = root / f"bad_{label}.nc"
        shutil.copyfile(directory / base, bad)
        with netCDF4.Dataset(bad, "a") as ds:
            edit(ds)
        try:
            surfacefile.read_surface_file(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"{label}: the Python reads the malformed file")
        out.append(bad)
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


def _fault_examples() -> dict[str, tuple[Any, dict[str, Any]]]:
    """The faults of the module docstring, with what they are written with."""
    from planetmodel.fault import (CartesianFrame, Fault, GeographicFrame, Plane,
                                   Slip, StructuredPatch)

    rng = np.random.default_rng(23)
    earth = GeographicFrame(6371e3)
    rupture = Plane.at_depth(earth, 38.3, 142.4, 20e3, strike=195.0, dip=14.0,
                             length=400e3, width=160e3,
                             slip=Slip.from_rake([[80.0, 95.0], [90.0, 110.0],
                                                  [70.0, 85.0]],
                                                 [[3.0, 1.0], [8.0, 4.0], [2.0, 0.5]],
                                                 opening=[[0.0, 0.0], [0.3, 0.0],
                                                          [0.0, 0.0]]))
    km = GeographicFrame(6371.0)
    main = Plane.at_depth(km, 35.0, 140.0, 20.0, strike=200.0, dip=15.0,
                          length=100.0, width=50.0,
                          slip=Slip(rng.normal(size=(3, 4, 3, 3)), rule="linear"))
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 6), np.linspace(-0.5, 0.5, 5),
                       indexing="ij")
    base = Plane.at_depth(km, 34.2, 139.7, 30.0, strike=210.0, dip=25.0, length=60.0,
                          width=40.0, slip=Slip(np.zeros((1, 1, 3))))
    bend = np.stack([0.02 * U * V, 0.03 * U ** 2, -4.0 * V ** 2 + 1.0 * U], axis=-1)
    bent = StructuredPatch(km, base.position(U, V) + bend,
                           slip=Slip(rng.normal(size=(3, 4, 4, 3)), rule="cubic"),
                           interpolation="cubic")
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 4), np.linspace(-0.5, 0.5, 3),
                       indexing="ij")
    lon = 179.6 + 0.8 * (U + 0.5)
    lon = np.where(lon > 180.0, lon - 360.0, lon)
    crossing = StructuredPatch(
        km, np.stack([-10.0 - 0.3 * (V + 0.5) + 0.05 * U ** 2, lon,
                      6371.0 - 5.0 - 30.0 * (V + 0.5) - 3.0 * (U * V)], axis=-1),
        slip=Slip(rng.normal(size=(3, 2, 2, 3))), interpolation="linear")
    local = CartesianFrame(surface=1.5)
    U, V = np.meshgrid(np.linspace(-0.5, 0.5, 3), np.linspace(-0.5, 0.5, 3),
                       indexing="ij")
    twisted = StructuredPatch(local, np.stack([3.0 * U, 2.0 * V, -4.0 + 1.6 * U * V
                                               - 0.4 * V], axis=-1),
                              slip=Slip(rng.normal(size=(2, 3, 3, 3)), rule="linear"),
                              interpolation="linear")
    wall = Plane(local, [1.0, 2.0, -3.0], strike=250.0, dip=90.0, length=2.0,
                 width=1.0, slip=Slip(rng.normal(size=(2, 4, 4, 3)), rule="cubic"))
    sill = Plane(local, [0.0, 0.0, -1.0], strike=30.0, dip=0.0, length=2.0, width=2.0,
                 slip=Slip.from_rake(0.0, [[[0.0]], [[0.0]]],
                                     opening=[[[0.4]], [[0.7]]]))
    return {
        "fault_static": (Fault([rupture]), {}),
        "fault_afterslip": (Fault([main, bent, crossing], support="node",
                                  time_interpolation="linear",
                                  time=np.array([0.0, 1.0, 2.5]),
                                  scales=pm.Scales(length=1e3, time=3.15576e7)),
                            {"names": ["main", None, "splay"],
                             "meta": {"event": "fixture"}, "citation": ("a work",)}),
        "fault_cartesian": (Fault([twisted, wall, sill], support="node",
                                  time_interpolation="previous",
                                  time=np.array([-1004.0, -999.0])),
                            {"names": ["twisted", "wall", "sill"]}),
    }


#: The (u, v) at which every patch is evaluated: corners, edges, the
#: centre, knots of the grids above and cell boundaries among them.
_FAULT_UV = ((-0.5, -0.5), (0.5, 0.5), (0.5, -0.5), (-0.5, 0.5), (0.0, 0.0),
             (0.13, -0.37), (-0.42, 0.29), (0.1, -0.25), (-0.3, 0.0),
             (0.5, 0.11), (-0.1666, 0.5), (1 / 6, 1 / 3), (0.25, 0.4999))


def _fault_points(fault: Any, rng: np.random.Generator) -> list[np.ndarray]:
    """Physical points about a fault: beside each patch along its normal,
    beyond its edges and corners in its tangent plane, and scattered."""
    out = []
    for p in fault.patches:
        size = float(np.linalg.norm(p.cartesian(0.5, 0.5) - p.cartesian(-0.5, -0.5)))
        for u, v, off in ((0.1, 0.2, 0.01), (-0.3, -0.1, -0.02), (0.37, 0.41, 0.005),
                          (0.0, -0.5, 0.0), (-0.1, 0.05, 0.08)):
            out.append(p.cartesian(u, v) + off * size * p.normal(u, v))
        for u, v, du, dv in ((0.5, 0.0, 1.0, 0.0), (-0.5, 0.2, -1.0, 0.0),
                             (0.1, 0.5, 0.0, 1.0), (0.5, 0.5, 1.0, 1.0),
                             (-0.5, -0.5, -1.0, -0.6)):
            xu, xv = p.tangents(u, v)
            out.append(p.cartesian(u, v) + 0.07 * (du * xu + dv * xv)
                       + 0.01 * size * p.normal(u, v))
        for _ in range(3):
            u, v = rng.uniform(-0.45, 0.45, 2)
            out.append(p.cartesian(u, v) + 0.03 * size * rng.normal(size=3))
    return out


def _vector(values: Any) -> str:
    return " ".join(repr(float(x)) for x in np.ravel(values))


def _expect_fault(fault: Any, names: Any, rng: np.random.Generator) -> list[str]:
    """The Python's answers about a fault, as the module docstring lists them."""
    from planetmodel.fault import BASES

    frame = fault.frame
    reference = frame.radius if frame.kind == "geographic" else frame.surface
    lines = [f"fault {frame.kind} {reference!r} {fault.support} "
             f"{fault.time_interpolation} {len(fault.patches)} {fault.nslices}"]
    s = fault.scales
    lines.append(f"scales {s.length!r} {s.mass!r} {s.time!r} {s.temperature!r}")
    if fault.time is not None:
        lines += [f"time {k} {float(t)!r}" for k, t in enumerate(fault.time)]
    names = names or [None] * len(fault.patches)
    for k, (p, name) in enumerate(zip(fault.patches, names)):
        tag = "-" if name is None else name.encode("utf-8").hex()
        lines.append(f"patch {k} {p.kind} {tag}")
        for u, v in _FAULT_UV:
            head = f"{k} {u!r} {v!r}"
            xu, xv = p.tangents(u, v)
            lines += [f"point {head} {_vector(p.cartesian(u, v))}",
                      f"position {head} {_vector(p.position(u, v))}",
                      f"tangents {head} {_vector([xu, xv])}",
                      f"second {head} {_vector(p.second_derivatives(u, v))}",
                      f"normal {head} {_vector(p.normal(u, v))}",
                      f"area {head} {float(p.area_element(u, v))!r}",
                      f"basis {head} {_vector(p.fault_basis(u, v))}"]
    if fault.time is None:
        dates = [(None, "refuse")]
    else:
        t = fault.time
        # within the default tolerance of a node and of the ends, and
        # beyond it, the two languages' tolerances held to one another: the
        # tolerance is relative to the axis's scale, its span or the
        # magnitude of its first or last time, whichever is largest
        tol = 1e-12 * max(float(t[-1] - t[0]), abs(float(t[0])), abs(float(t[-1])))
        dates = [(float(t[0]), "refuse"), (float(t[-1]), "refuse"),
                 (float(0.6 * t[0] + 0.4 * t[-1]), "refuse"),
                 (float(t[-1] + 1.0), "refuse"), (float(t[-1] + 1.0), "hold"),
                 (float(t[-1] + 0.5 * tol), "refuse"),
                 (float(t[-1] + 3.0 * tol), "refuse"),
                 (float(t[-1] - 0.4 * tol), "refuse"),
                 (float(t[0] - 0.4 * tol), "refuse"),
                 (float(t[0] - 2.0), "hold")]
        if t.size > 2:
            dates += [(float(t[1]), "refuse"), (float(t[1] - 0.3 * tol), "refuse")]
    for k in range(len(fault.patches)):
        for u, v in _FAULT_UV[:9]:
            for date, extrapolation in dates:
                for basis in BASES:
                    tag = "-" if date is None else repr(date)
                    head = f"slip {k} {u!r} {v!r} {tag} {extrapolation} {basis}"
                    try:
                        got = fault.slip_vector(k, u, v, date=date, basis=basis,
                                                extrapolation=extrapolation)
                    except ValueError:
                        lines.append(f"{head} refused")
                        continue
                    lines.append(f"{head} {_vector(got)}")
    n = len(fault.patches)
    for points, n_u, n_v, (date, extrapolation) in (
            (1, [2] * n, [3] * n, dates[0]),
            (3, list(range(1, n + 1)), list(range(n, 0, -1)), dates[-1])):
        q = fault.quadrature(n_u, n_v, points=points, date=date,
                             extrapolation=extrapolation)
        lines.append(f"quadrature {points} {'-' if date is None else repr(date)} "
                     f"{extrapolation} {','.join(map(str, n_u))} "
                     f"{','.join(map(str, n_v))} {q.size}")
        for i in range(q.size):
            lines.append(f"qpoint {i} {int(q.patch[i])} {float(q.u[i])!r} "
                         f"{float(q.v[i])!r} {float(q.weights[i])!r} "
                         f"{_vector(q.points[i])} {_vector(q.positions[i])} "
                         f"{_vector(q.normals[i])} {_vector(q.slip[i])}")
    for points, n_u, n_v in ((2, [2] * n, [1] * n),
                             (1, list(range(1, n + 1)), [2] * n)):
        sq = fault.slice_quadrature(n_u, n_v, points=points)
        lines.append(f"slices {points} {','.join(map(str, n_u))} "
                     f"{','.join(map(str, n_v))} {sq.size} {sq.nslices}")
        for i in range(sq.size):
            lines.append(f"spoint {i} {int(sq.patch[i])} {float(sq.u[i])!r} "
                         f"{float(sq.v[i])!r} {float(sq.weights[i])!r} "
                         f"{_vector(sq.points[i])} {_vector(sq.positions[i])} "
                         f"{_vector(sq.normals[i])} {_vector(sq.slip[:, i])}")
    points = _fault_points(fault, rng)
    for x in points:
        found = fault.locate(x)
        lines.append(f"locate {_vector(x)} {int(found.patch)} {float(found.u)!r} "
                     f"{float(found.v)!r} {float(found.distance)!r} "
                     f"{int(bool(found.inside))} {int(bool(found.converged))}")
    near = np.array(points)
    distance = np.abs(fault.locate(near).distance)
    limit = float(np.max(distance)) * 1.01
    date = dates[min(2, len(dates) - 1)][0]
    for basis in BASES:
        got = fault.slip_at(near, max_distance=limit, date=date, basis=basis)
        for x, value in zip(near, got):
            lines.append(f"slip_at {limit!r} {'-' if date is None else repr(date)} "
                         f"{basis} {_vector(x)} {_vector(value)}")
    tight = float(np.median(distance))
    lines.append(f"slip_at_refused {tight!r} {_vector(near[np.argmax(distance)])}")
    return lines


def _flattening(group: str) -> Any:
    """An edit collapsing a Cartesian structured patch onto a line."""
    def edit(ds: Any) -> None:
        for name in ("y", "z"):
            ds.groups[group].variables[name][:] = 0.0
    return edit


_P0 = {"group": "patch_0"}
_SLIP0 = {"group": "patch_0", "variable": "slip"}

#: The malformed fault files: (the file copied, an edit by netCDF4, where
#: the fault lies, what the refusal says).
_BAD_FAULTS: dict[str, tuple[str, Any, str, str]] = {
    "schema": ("fault_static.nc", _setting("schema", "planetmodel.fault/0"), "schema",
               "this reader reads planetmodel.fault/1"),
    "no_schema": ("fault_static.nc", _deleting("schema"), "schema",
                  "not a planetmodel fault file"),
    "root_attribute": ("fault_static.nc", _setting("rupture_velocity", 2.5),
                       "rupture_velocity", "planetmodel.fault/1 does not name it"),
    "no_created": ("fault_static.nc", _deleting("created"), "created",
                   "is missing or empty"),
    "source_empty": ("fault_static.nc", _setting("source", ""), "source",
                     "is missing or empty"),
    "scale": ("fault_static.nc", _setting("scale_time", -1.0), "scale_time",
              "is missing or not positive"),
    "scale_infinite": ("fault_static.nc", _setting("scale_time", np.inf), "scale_time",
                       "is missing or not positive and finite"),
    "frame": ("fault_static.nc", _setting("frame", "spherical"), "frame",
              "not one of geographic, cartesian"),
    "radius": ("fault_static.nc", _setting("reference_radius", 0.0), "reference_radius",
               "not a positive number"),
    "both_references": ("fault_static.nc", _setting("reference_surface", 0.0),
                        "reference_surface",
                        "not an attribute of a geographic fault file"),
    "no_surface": ("fault_cartesian.nc", _deleting("reference_surface"),
                   "reference_surface", "is missing or not a finite number"),
    "interval": ("fault_afterslip.nc", _setting("time_support", "interval"),
                 "time_support",
                 "is interval, which is not part of planetmodel.fault/1"),
    "instant": ("fault_afterslip.nc", _setting("time_support", "instant"),
                "time_support", "is instant, which is not part of planetmodel.fault/1"),
    "support": ("fault_static.nc", _setting("time_support", "always"), "time_support",
                "not one of static, node"),
    "time_rule": ("fault_static.nc", _setting("time_interpolation", "linear"),
                  "time_interpolation", "is not sampled between times"),
    "time_rule_unknown": ("fault_afterslip.nc", _setting("time_interpolation", "cubic"),
                          "time_interpolation",
                          "'slip': time_interpolation 'cubic' is not one of "
                          f"{TIME_METHODS}"),
    "patches_text": ("fault_static.nc", _setting("patches", "one"), "patches",
                     "not a positive integer"),
    "patches_float": ("fault_static.nc", _setting("patches", 1.0), "patches",
                      "not a positive integer"),
    "patches_more": ("fault_static.nc", _setting("patches", np.int32(2)), "patch_1",
                     "is missing: the file states 2 patches"),
    "patches_fewer": ("fault_afterslip.nc", _setting("patches", np.int32(2)), "patch_2",
                      "is not one of the 2 patches"),
    "root_dimension": ("fault_static.nc", _adding_dimension("time"), "time",
                       "is not a dimension of a static fault file"),
    "time_order": ("fault_afterslip.nc", _storing((2,), 0.5, variable="time"), "time",
                   "not finite and strictly increasing"),
    "time_attribute": ("fault_afterslip.nc",
                       _setting("calendar", "julian", variable="time"),
                       "time.calendar", "does not name it"),
    "kind": ("fault_static.nc", _setting("kind", "triangulated", **_P0),
             "patch_0.kind", "not one of plane, structured"),
    "patch_attribute": ("fault_static.nc", _setting("rake", 90.0, **_P0),
                        "patch_0.rake", "not an attribute of a plane patch"),
    "name": ("fault_static.nc", _setting("name", 3.0, **_P0), "patch_0.name",
             "is not text"),
    "patch_dimension": ("fault_static.nc", _adding_dimension("nu", group="patch_0"),
                        "patch_0/nu", "not a dimension of a plane patch"),
    "centroid": ("fault_static.nc", _setting("centroid", [91.0, 0.0, 1e6], **_P0),
                 "patch_0.centroid", "latitude outside [-90, 90]"),
    "centroid_size": ("fault_static.nc", _setting("centroid", [1.0, 2.0], **_P0),
                      "patch_0.centroid", "not three finite numbers"),
    "dip": ("fault_static.nc", _setting("dip", 95.0, **_P0), "patch_0.dip",
            "not in [0, 90] degrees"),
    "width": ("fault_static.nc", _setting("width", 0.0, **_P0), "patch_0.width",
              "not a positive number"),
    "strike": ("fault_static.nc", _deleting("strike", **_P0), "patch_0.strike",
               "is missing or not a finite number"),
    "interpolation": ("fault_afterslip.nc",
                      _setting("interpolation", "quintic", group="patch_1"),
                      "patch_1.interpolation", "not one of linear, cubic"),
    "nodes": ("fault_afterslip.nc", _setting("interpolation", "cubic", group="patch_2"),
              "patch_2", "the cubic rule needs at least 4 each way"),
    "position_renamed": ("fault_afterslip.nc",
                         _renaming("radius", "r", group="patch_1"),
                         "patch_1/r", "not a variable of a structured patch"),
    "position_nan": ("fault_afterslip.nc",
                     _storing((1, 2), np.nan, variable="radius", group="patch_1"),
                     "patch_1", "control positions that are not finite"),
    "position_latitude": ("fault_afterslip.nc",
                          _storing((0, 0), -91.0, variable="latitude", group="patch_1"),
                          "patch_1", "latitude outside [-90, 90]"),
    "position_attribute": ("fault_afterslip.nc",
                           _setting("axis", "Y", group="patch_1", variable="latitude"),
                           "patch_1/latitude.axis", "does not name it"),
    "degenerate": ("fault_cartesian.nc", _flattening("patch_0"), "patch_0",
                   "is degenerate"),
    "slip_renamed": ("fault_static.nc", _renaming("slip", "slip_east", **_P0),
                     "patch_0/slip_east", "not a variable of a plane patch"),
    "slip_components": ("fault_static.nc",
                        _setting("components", "east north up", **_SLIP0),
                        "patch_0/slip.components",
                        "not \"strike_slip dip_slip opening\""),
    "slip_rule": ("fault_static.nc", _setting("rule", "nearest", **_SLIP0),
                  "patch_0/slip.rule", "not one of cell, linear, cubic"),
    "slip_nodes": ("fault_static.nc", _setting("rule", "cubic", **_SLIP0),
                   "patch_0/slip", "the cubic rule needs at least 4 nodes"),
    "slip_nan": ("fault_afterslip.nc",
                 _storing((1, 0, 1, 2), np.nan, variable="slip", group="patch_0"),
                 "patch_0/slip", "values that are not finite"),
    "slip_attribute": ("fault_static.nc", _setting("rake", 90.0, **_SLIP0),
                       "patch_0/slip.rake", "does not name it"),
}


def fault_files(directory: Path) -> list[Path]:
    """The fault files of the module docstring, their expectations, and
    the malformed copies."""
    import netCDF4

    from planetmodel.faultfile import read_fault_file, write_fault
    from planetmodel.testing import check_fault_file

    rng = np.random.default_rng(29)
    out = []
    for name, (fault, kw) in _fault_examples().items():
        path = write_fault(directory / f"{name}.nc", fault,
                           source="planetmodel fixtures", **kw)
        check_fault_file(path, fault, **kw)
        back = read_fault_file(path).fault
        expect = directory / f"{name}.expect"
        expect.write_text("\n".join(_expect_fault(back, kw.get("names"), rng)) + "\n")
        out += [path, expect]
    root = directory / "fault_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    lines = []
    for label, (base, edit, where, phrase) in _BAD_FAULTS.items():
        path = root / f"bad_{label}.nc"
        shutil.copyfile(directory / base, path)
        with netCDF4.Dataset(path, "a") as ds:
            edit(ds)
        try:
            read_fault_file(path)
        except ValueError as error:
            said = str(error)
            if f": {where}: " not in said or phrase not in said:
                raise AssertionError(f"{label}: the Python says {said!r}") from None
        else:
            raise AssertionError(f"{label}: the Python reads the malformed file")
        out.append(path)
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


def slip_from_rake(directory: Path) -> list[Path]:
    """`Slip.from_rake` on grids of every rule, its arguments one value,
    one a point or one a slice and point, as the module docstring lists
    them (`slip_rake.txt`)."""
    from planetmodel.fault import Slip

    rng = np.random.default_rng(37)
    cases = [
        (1, 1, 1, "cell", 45.0, 3.0, 0.0),
        (1, 2, 3, "cell", rng.uniform(-180.0, 180.0, (2, 3)), 2.5, 0.1),
        (3, 2, 2, "linear", rng.uniform(0.0, 360.0, (2, 2)),
         rng.uniform(0.0, 4.0, (3, 2, 2)), 0.0),
        (2, 4, 4, "cubic",
         np.array([90.0, 180.0, -90.0, 270.0, 1e-3, 359.99, 0.0, -180.0,
                   30.0, 60.0, 120.0, 150.0, 210.0, 240.0, 300.0, 330.0]
                  * 2).reshape(2, 4, 4),
         rng.uniform(-1.0, 5.0, (2, 4, 4)), rng.uniform(0.0, 0.5, (2, 4, 4))),
    ]
    lines = []
    for nslices, n_u, n_v, rule, rake, magnitude, opening in cases:
        slip = Slip.from_rake(rake, magnitude, opening=opening, rule=rule)
        lines += [f"case {nslices} {n_u} {n_v} {rule}",
                  f"rake {_vector(rake)}", f"magnitude {_vector(magnitude)}",
                  f"opening {_vector(opening)}", f"values {_vector(slip.values)}"]
    path = directory / "slip_rake.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


def burgers(directory: Path) -> list[Path]:
    """`burgers_terms` and `burgers_parameters` (`burgers.txt`): `terms
    <modulus> <viscosity> <kelvin_viscosity> <kelvin_modulus_fraction>`
    and the strengths and times, faster first, for bodies across the
    ranges a mantle spans, the Kelvin element far faster and far slower
    than the Maxwell one and the roots nearly equal among them; and
    `parameters <modulus> <w_1> <w_2> <tau_1> <tau_2>` and the viscosity
    and the Kelvin element, for those branches in both orders."""
    from planetmodel.conversion.prony import burgers_parameters, burgers_terms

    rng = np.random.default_rng(41)
    bodies = [(7e10, 1e21, 1e20, 0.5), (7e10, 1e21, 2e20, 1.0),
              (3e10, 1e19, 1e18, 1.0), (7e10, 1e21, 1e25, 1e-3),
              (7e10, 1e21, 1e15, 1e3), (1.0, 1.0, 1.0, 1.0),
              (7e10, 1e21, 1e21, 1e-6), (7e10, 1e18, 1e24, 1e6), (2.0, 3.0, 5.0, 7.0)]
    for _ in range(12):
        bodies.append((float(10.0 ** rng.uniform(9.0, 12.0)),
                       float(10.0 ** rng.uniform(17.0, 24.0)),
                       float(10.0 ** rng.uniform(15.0, 25.0)),
                       float(10.0 ** rng.uniform(-3.0, 3.0))))
    lines = []
    for mu, eta, eta_k, f in bodies:
        w, tau = burgers_terms(mu, eta, kelvin_viscosity=eta_k,
                               kelvin_modulus_fraction=f)
        lines.append(f"terms {mu!r} {eta!r} {eta_k!r} {f!r} {_vector(w)} "
                     f"{_vector(tau)}")
        for order in ((0, 1), (1, 0)):
            ws, ts = w[list(order)], tau[list(order)]
            back = burgers_parameters(mu, ws, ts)
            lines.append(f"parameters {mu!r} {_vector(ws)} {_vector(ts)} "
                         f"{_vector(back)}")
    path = directory / "burgers.txt"
    path.write_text("\n".join(lines) + "\n")
    return [path]


# -- the flat fixtures: planetmodel.cartesian ---------------------------------------

def _flat_domains() -> dict[str, Any]:
    """The lateral domains of the flat fixtures: a box padded along x, a
    box that is its own period, and a rectangle padded on each side."""
    from planetmodel import cartesian as flat
    return {"padded": flat.Box([(0.0, 400e3), (0.0, 300e3)], padding=(50e3, 0.0)),
            "box": flat.Box([(0.0, 400e3), (0.0, 300e3)]),
            "rectangle": flat.Rectangle((0.0, 400e3), padding=25e3)}


def _wave(grid: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """A smooth function of (x, y) with the periods of a grid's domain, the
    exponential of sines; of x alone over a rectangle."""
    x0 = grid.domain.extent[0][0]
    lx = grid.lengths[0]
    a = 2.0 * np.pi * (np.asarray(x, dtype=float) - x0) / lx
    out = np.sin(a) + 0.3 * np.cos(2.0 * a + 0.4)
    if grid.dimension == 2:
        b = 2.0 * np.pi * (np.asarray(y, dtype=float) - grid.domain.extent[1][0]) \
            / grid.lengths[1]
        out = out + 0.5 * np.cos(b + 0.3) * (1.0 + 0.2 * np.sin(a))
    return np.exp(0.5 * out)


def flat_lateral_cases(directory: Path) -> list[Path]:
    """The flat lateral rules, doubly periodic, on a box's grid and a
    rectangle's, each on a grid of the fewest nodes the cubic rule takes
    too: `flat_case_lateral_<grid>_<rule>.nc`."""
    from planetmodel import cartesian as flat
    from planetmodel.cartesian.lateral import LateralInterpolant
    rng = np.random.default_rng(20261117)
    box = flat.Box([(-1.0, 3.0), (2.0, 5.0)], padding=(0.5, 0.0))
    rectangle = flat.Rectangle((10.0, 14.0), padding=0.25)
    grids = {"box": flat.Grid(box, (9, 12)), "box_small": flat.Grid(box, (3, 4)),
             "rectangle": flat.Grid(rectangle, (11,)),
             "rectangle_small": flat.Grid(rectangle, (3,))}
    out = []
    for label, grid in grids.items():
        gx, gy = grid.nodes
        values = np.broadcast_to(_wave(grid, gx, gy), grid.shape)
        two = grid.dimension == 2
        (x0, x1) = grid.domain.extent[0]
        lx = grid.lengths[0]
        y0, ly = (grid.domain.extent[1][0], grid.lengths[1]) if two else (0.0, 1.0)
        # random points over three periods, the seam, the nodes, the far end
        qx = list(rng.uniform(x0 - lx, x1 + lx, 40)) + [
            x0 - 1e-9 * lx, x0 + 1e-9 * lx, float(grid.x[1]), float(grid.x[-1]),
            x1, x0 + 0.37 * lx]
        qy = list(rng.uniform(y0 - ly, y0 + 2.0 * ly, 40)) + [
            y0 + 0.21 * ly, y0 + 0.83 * ly, y0 + 0.5 * ly, y0 - 1e-9 * ly,
            y0 + 0.4 * ly, float(grid.y[2]) if two else 0.0]
        qx, qy = np.array(qx), np.array(qy)
        for method in ("linear", "cubic"):
            f = LateralInterpolant(values, grid, method=method)
            dx, dy = f.gradient(qx, qy)
            arrays = {"x": grid.x, "values": values, "query_x": qx, "query_y": qy,
                      "expected": f(qx, qy), "expected_dx": dx, "expected_dy": dy}
            if two:
                arrays["y"] = grid.y
            out.append(_write_case(
                directory / f"flat_case_lateral_{label}_{method}.nc", arrays,
                {"domain": "box" if two else "rectangle",
                 "bounds": np.ravel(np.asarray(grid.domain.bounds, dtype=float)),
                 "padding": np.asarray(grid.domain.padding, dtype=float),
                 "method": method, "mean": float(f.mean())}))
    return out


def _flat_models() -> dict[str, tuple[Any, dict[str, Any]]]:
    """The flat models of the fixtures, each with the arguments it is
    written with."""
    from scipy.interpolate import PPoly as Pieces

    from planetmodel import cartesian as flat
    domains = _flat_domains()
    box, padded = domains["box"], domains["padded"]
    rectangle = domains["rectangle"]
    out: dict[str, tuple[Any, dict[str, Any]]] = {}

    # layered only: fields of depth alone, over a box and over a rectangle
    layered = flat.CrustOverMantle(box, viscosity=1e21).with_uniform_gravity(9.8)
    out["layered"] = (layered, {"grid": flat.Grid(box, (4, 3))})
    out["layered_rectangle"] = (flat.CrustOverMantle(rectangle),
                                {"grid": flat.Grid(rectangle, (5,))})
    # the same, of a type with a behaviour of its own, with no grid given
    own = model_type(flat.Model, (pm.Elastic, Tidal, pm.Viscoelastic))
    out["depth"] = (own(layered.geometry, [layer.fields for layer in layered.layers],
                        name="depth alone"), {})

    # lateral fields on each rule, of every rank, in two parts on two grids
    base = flat.CrustOverMantle(padded)
    coarse, fine = flat.Grid(padded, (10, 6)), flat.Grid(padded, (16, 12))

    def rho3d(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 3300.0 * (1.0 + 1e-7 * d * (_wave(fine, x, y) - 1.0))

    def wind(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d, x, y = np.broadcast_arrays(d, x, y)
        w = _wave(fine, x, y)
        return np.stack([1e-5 * d * w, 2.0 - w, 0.5 * w + 1e-5 * d], axis=-1)

    def stress(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d, x, y = np.broadcast_arrays(d, x, y)
        w = _wave(coarse, x, y)
        return np.stack([1.0 + w, 2.0 + 1e-5 * d, 3.0 - w, 0.5 * w, 0.2 + 1e-6 * d * w,
                         0.1 * w], axis=-1)

    lateral = base.with_field(1, "rho", flat.GriddedField.sampled(
        flat.AnalyticField((30e3, 100e3), rho3d, character=pm.DENSITY), fine,
        breaks=[30e3, 60e3, 100e3], ngll=4, lateral="cubic"), replace=True)
    lateral = lateral.with_field(1, "flow", flat.GriddedField.sampled(
        flat.AnalyticField((30e3, 100e3), wind, character=pm.VECTOR), fine, ngll=3,
        lateral="linear"))
    lateral = lateral.with_field(0, "stress", flat.DepthField((0.0, 30e3), [
        lambda d: 1.0 + 1e-5 * d, lambda d: 2.0 - 1e-5 * d,
        lambda d: 3.0 + 1e-10 * d ** 2, lambda d: 1e-6 * d, lambda d: 2e-6 * d,
        lambda d: 3e-6 * d],
        character=pm.STRESS, name="stress"))
    lateral = lateral.with_field(2, "stress", flat.GriddedField.sampled(
        flat.AnalyticField((100e3, 200e3), stress, character=pm.STRESS,
                           frame="cartesian"), coarse, ngll=3, lateral="cubic"))
    lateral = pm.perturbed(lateral, "vs", flat.AnalyticField(
        (100e3, 200e3), lambda d, x, y: 0.02 * (_wave(coarse, x, y) - 1.0)
        * (200e3 - d) / 100e3), kind="relative")
    lateral = lateral.with_geometry(lateral.geometry.renamed(
        interfaces=["surface", "moho", None]))
    out["lateral"] = (lateral, {"grid": {0: fine, 1: fine, 2: coarse},
                                "interpolation": {"vs": ("polynomial", "cubic")}})

    # the same kinds over a rectangle, whose grid has the one axis
    line = flat.Grid(rectangle, (14,))
    two = flat.CrustOverMantle(rectangle)
    cut = two.with_field(1, "rho", flat.GriddedField.sampled(
        flat.AnalyticField((30e3, 100e3), lambda d, x, y: 3300.0 * (
            1.0 + 1e-7 * d * (_wave(line, x, y) - 1.0)), character=pm.DENSITY),
        line, breaks=[30e3, 60e3, 100e3], ngll=4, lateral="cubic"), replace=True)

    def drift(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d, x, y = np.broadcast_arrays(d, x, y)
        w = _wave(line, x, y)
        return np.stack([1e-5 * d * w, 2.0 - w, 0.5 * w + 1e-5 * d], axis=-1)

    cut = cut.with_field(2, "flow", flat.GriddedField.sampled(
        flat.AnalyticField((100e3, 200e3), drift, character=pm.VECTOR,
                           frame="cartesian"), line, ngll=3, lateral="linear"))
    cut = pm.perturbed(cut, "vs", flat.AnalyticField(
        (0.0, 30e3), lambda d, x, y: 0.01 * _wave(line, x, y) * d / 30e3),
        kind="absolute")
    out["rectangle"] = (cut, {"grid": line,
                              "interpolation": {"vs": ("polynomial", "cubic")}})

    # a Fourier field, which states the rule the C++ refuses by name
    grid = flat.Grid(box, (8, 6))
    gx, gy = grid.nodes
    c = grid.transform.analyse_grid(0.03 * (np.broadcast_to(_wave(grid, gx, gy),
                                                            grid.shape) - 1.0))
    fade = Pieces(np.stack([-c / 200e3, c])[:, None], [0.0, 200e3])
    fourier = pm.perturbed(flat.CrustOverMantle(box), "vs",
                           flat.FourierField((0.0, 200e3), grid, fade),
                           kind="relative")
    out["fourier"] = (fourier, {"grid": grid, "fields": ["rho", "vs"]})

    # topography as a stack of reliefs, over a box and over a rectangle
    def hill(g: Any) -> Any:
        return flat.GriddedFunction.sampled(
            lambda x, y: 2e3 * (_wave(g, x, y) - 1.0), g)

    def root(g: Any) -> Any:
        return flat.GriddedFunction.sampled(
            lambda x, y: -6e3 * (_wave(g, x, y) - 1.0), g, lateral="linear")

    reliefs = flat.Grid(padded, (20, 12))
    stacked = base.stretched(flat.layer_linear(
        base.skeleton, [hill(reliefs), root(reliefs), None, None]))
    out["stacked"] = (stacked, {"grid": coarse, "fields": ["rho", "vp"]})
    out["stacked_rectangle"] = (two.stretched(flat.layer_linear(
        two.skeleton, [hill(line), hill(line), None, None])),
        {"grid": line, "fields": ["rho"]})

    # a vertical stretch that is no stack: held as its displacement
    def w(d: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 1.5e3 * (_wave(coarse, x, y) - 1.0) * (1.0 - d / 200e3) ** 2

    smooth_stretch = base.stretched(flat.CallableDisplacement(w, name="taper"))
    out["stretched"] = (smooth_stretch, {"grid": coarse, "fields": ["rho"]})

    # fields on interfaces
    kernel = base.replaced(specs={**base.specs, "K_shape": pm.FieldSpec(
        pm.SCALAR, pm.Dimensions(length=-2), meaning="a shape kernel")})
    on_faces = {"K_shape": {0: lambda x, y: _wave(coarse, x, y),
                            "moho": lambda x, y: 2.0 - _wave(coarse, x, y)},
                "shift": {1: flat.GriddedFunction.sampled(
                    lambda x, y: 30.0 * _wave(fine, x, y), fine)}}
    out["surfaces"] = (kernel, {"grid": coarse, "fields": ["rho", "vp"],
                                "surfaces": on_faces})

    # padded below the base, the lowest properties held
    held = lateral.extended([260e3, 400e3], fields="hold",
                            layer_names=["padding", "deep_padding"])
    out["held"] = (held, {"grid": coarse, "fields": ["rho", "vs", "stress"],
                          "interpolation": {"vs": ("polynomial", "cubic")}})

    # a viscosity joined in its logarithm, 1e21 at 100 km and three decades
    # less at the base, and a constant above it
    logged = base.with_field(2, "viscosity", flat.DepthField(
        (100e3, 200e3), pm.layerfunction.ExponentialLayer(
            (100e3, 200e3), Pieces(np.array([[-3.0 * np.log(10.0) / 100e3],
                                             [np.log(1e21)]]), [100e3, 200e3])),
        name="viscosity"))
    logged = logged.with_field(1, "viscosity", flat.constant_field(
        (30e3, 100e3), 1e23, name="viscosity"))
    out["log"] = (logged, {"grid": flat.Grid(padded, (3, 3)),
                           "fields": ["rho", "viscosity"],
                           "interpolation": {"viscosity": ("log-polynomial",
                                                           "linear")}})
    return out


def _flat_gradient(f: Any, d: float, x: float, y: float) -> np.ndarray:
    """(d/d depth, d/dx, d/dy) of a scalar field the flat reader builds,
    from the exact derivatives of its parts."""
    from planetmodel.cartesian import GriddedField
    if getattr(f, "reference", None) is not None:
        ref, dp = f.reference, f.perturbation
        v0, g0 = float(ref.evaluate(d, x, y)), _flat_gradient(ref, d, x, y)
        v1, g1 = float(dp.evaluate(d, x, y)), _flat_gradient(dp, d, x, y)
        if f.kind == "absolute":
            return g0 + g1
        if f.kind == "relative":
            return g0 * (1.0 + v1) + v0 * g1
        return np.exp(v1) * (g0 + v0 * g1)
    if isinstance(f, GriddedField):
        _, dd, dx, dy = f.derivatives(d, x, y)
        return np.array([dd.item(), dx.item(), dy.item()])
    return np.array([float(f.derivative().evaluate(d, x, y)), 0.0, 0.0])


def _expect_flat_model(path: Path) -> Path:
    """The expectations of a flat model file: the lines of
    `_expect_model`, a point being (depth, x, y) and a frame `l` (local)
    or `c` (Cartesian), and `domain <kind> <bounds> <padding>`."""
    from planetmodel.cartesian import FLAT
    from planetmodel.cartesian.modelfile import read_model_file
    from planetmodel.modelfile import read_model, read_surfaces
    data, model = read_model_file(path), read_model(path, model_type=())
    hexed = (lambda x: "-" if not x else x.encode("utf-8").hex())
    s = data.scales
    domain = data.domain
    two = len(domain.axes) == 2
    lines = [f"scales {s.length!r} {s.mass!r} {s.time!r} {s.temperature!r}",
             f"rtol {data.rtol!r}", f"name {hexed(data.name)}",
             f"citation {hexed(chr(10).join(data.citation))}",
             f"behaviours {hexed(' '.join(data.behaviours))}",
             f"domain {'box' if two else 'rectangle'} "
             + " ".join(repr(float(v)) for v in (
                 *np.ravel(np.asarray(domain.bounds, dtype=float)),
                 *domain.padding))]
    lines += [f"layer_name {i} {hexed(n)}" for i, n in enumerate(data.layer_names)]
    lines += [f"interface_name {i} {hexed(n)}"
              for i, n in enumerate(data.interface_names)]
    for cname, value in data.constants.items():
        c = data.declared.get(cname)
        dims = "" if c is None else " " + " ".join(
            str(getattr(c.dimensions, b))
            for b in ("mass", "length", "time", "temperature"))
        lines.append(f"constant {cname} {value!r}{dims}")
    names = sorted({n for part in data.parts for n in part.variables
                    if not n.endswith("_perturbation") and n != "displacement"})
    (x0, x1), (y0, y1) = (*domain.extent, (0.0, 1.0))[:2]
    lx, ly = x1 - x0, y1 - y0
    for i, layer in enumerate(model.layers):
        lo, hi = layer.interval
        ds = (lo, lo + 0.3 * (hi - lo), lo + 0.77 * (hi - lo), hi)
        xs = (x0 - 0.2 * lx, x0 + 0.31 * lx, x1, x0 + 1.9 * lx)
        ys = (y0 + 0.1 * ly, y0 - 0.45 * ly, y0 + 0.62 * ly, y0 + 2.2 * ly)
        for name in names:
            if name not in layer:
                lines.append(f"absent {i} {name}")
                continue
            f = layer[name]
            for d, x, y in zip(ds, xs, ys):
                for frame, tag in (("local", "l"), ("cartesian", "c")):
                    v = np.asarray(f.evaluate(d, x, y, frame=frame),
                                   dtype=float).ravel()
                    lines.append(f"value {i} {name} {tag} {d!r} {x!r} {y!r} "
                                 + " ".join(repr(float(a)) for a in v))
                if f.character.rank == 0:
                    g = _flat_gradient(f, d, x, y)
                    lines.append(f"gradient {i} {name} {d!r} {x!r} {y!r} "
                                 + " ".join(repr(float(a)) for a in g))
    mapping = model.geometry.mapping
    if not model.geometry.is_identity:
        rng = np.random.default_rng(11)
        b = model.skeleton.boundaries
        depths = np.concatenate((rng.uniform(b[0], b[-1], 8), b))
        X = FLAT.points(depths, rng.uniform(x0 - lx, x1 + lx, depths.size),
                        rng.uniform(y0 - ly, y1 + ly, depths.size))
        U = mapping.displacement(X)
        for x, u in zip(X, U):
            lines.append("displacement " + " ".join(repr(float(a)) for a in (*x, *u)))
    for name, on in sorted(read_surfaces(path).items()):
        for face in model.geometry.interfaces:
            k = face.index
            f = on.get(k if face.name is None else face.name)
            if f is None:
                lines.append(f"surface_absent {k} {name}")
                continue
            points = ((x0 - 0.2 * lx, y0 + 0.1 * ly), (x0 + 0.31 * lx, y0 - 0.45 * ly),
                      (x1, y0 + 0.62 * ly), (x0 + 1.9 * lx, y0 + 2.2 * ly),
                      (x0, y0), (x0 + 0.5 * lx, y1))
            for x, y in points:
                lines.append(f"surface {k} {name} {x!r} {y!r} {float(f(x, y))!r}")
            for x, y in points[:3]:
                dx, dy = (float(v) for v in f.lateral_gradient(x, y))
                lines.append(f"surface_gradient {k} {name} {x!r} {y!r} {dx!r} {dy!r}")
    out = path.with_name(path.name.removesuffix(".nc") + ".expect")
    out.write_text("\n".join(lines) + "\n")
    return out


def flat_model_files(directory: Path) -> list[Path]:
    """The flat model files of the module docstring, `flat_model_<case>.nc`,
    each validated, and for every one but the Fourier one its
    expectations."""
    out = []
    for label, (model, arguments) in _flat_models().items():
        path = write_model(directory / f"flat_model_{label}.nc", model, **arguments)
        validate_structure(path)
        out.append(path)
        if label != "fourier":
            out.append(_expect_flat_model(path))
    return out


_FLAT_RHO = {"group": "part_0", "variable": "rho"}
_FLAT_VS = {"group": "part_1", "variable": "vs"}
_FLAT_STACK = {"group": "geometry", "variable": "boundary_displacement"}

#: The malformed flat model files: (the file copied, an edit by netCDF4,
#: where the fault lies, what the refusal says).
_BAD_FLAT_MODELS: dict[str, tuple[str, Any, str, str]] = {
    "schema_spherical": ("flat_model_lateral.nc",
                         _setting("schema", "planetmodel.model/1"), "schema",
                         "read by planetmodel::ReadModelFile"),
    "schema": ("flat_model_lateral.nc", _setting("schema", "planetmodel.model/9"),
               "schema", "this reader reads planetmodel.model/3"),
    "no_schema": ("flat_model_lateral.nc", _deleting("schema"), "schema",
                  "not a planetmodel model file"),
    "rtol_negative": ("flat_model_lateral.nc", _setting("rtol", -1.0), "rtol",
                      "not a tolerance"),
    "behaviours_number": ("flat_model_lateral.nc",
                          _setting("behaviours", np.int32(4)), "behaviours",
                          "not the names of the model's behaviours"),
    "behaviours_twice": ("flat_model_lateral.nc",
                         _setting("behaviours", "Elastic Tidal Elastic"),
                         "behaviours", "each once"),
    "no_scale_mass": ("flat_model_lateral.nc", _deleting("scale_mass"), "scale_mass",
                      "missing or not positive"),
    "coordinate_system": ("flat_model_lateral.nc",
                          _setting("coordinate_system", "spherical"),
                          "coordinate_system", "a flat model file states flat"),
    "no_local_frame": ("flat_model_lateral.nc", _deleting("local_frame"),
                       "local_frame", "a flat model file states up x y"),
    "local_frame": ("flat_model_lateral.nc", _setting("local_frame", "x y up"),
                    "local_frame", "a flat model file states up x y"),
    "no_domain": ("flat_model_lateral.nc", _deleting("domain"), "domain",
                  "states its lateral domain"),
    "domain_kind": ("flat_model_lateral.nc", _setting("domain", "disc"), "domain",
                    "not box or rectangle"),
    "domain_bounds": ("flat_model_lateral.nc",
                      _setting("domain_bounds", np.array([0.0, 400e3])),
                      "domain_bounds", "a box states 4 bounds"),
    "domain_inverted": ("flat_model_lateral.nc",
                        _setting("domain_bounds",
                                 np.array([400e3, 0.0, 0.0, 300e3])),
                        "domain", "malformed"),
    "spherical_attribute": ("flat_model_lateral.nc",
                            _setting("grid_kind", "custom", group="part_0"),
                            "part_0/grid_kind", "a spherical file's"),
    "spherical_variable": ("flat_model_lateral.nc",
                           _creating("interface_radius", ("interface",)),
                           "interface_radius", "a spherical file's"),
    "poles": ("flat_model_lateral.nc",
              _creating("rho_poles", ("node", "pole"), group="part_0",
                        sizes={"pole": 2}),
              "part_0/pole", "a spherical file's"),
    "polar_values": ("flat_model_lateral.nc",
                     _creating("rho_poles", ("node",), group="part_0"),
                     "part_0/rho_poles", "polar values"),
    "grid_nodes": ("flat_model_lateral.nc",
                   _storing((1,), 1.01, variable="x", group="part_0", relative=True),
                   "part_0/x", "not the nodes of a grid"),
    "rectangle_y": ("flat_model_rectangle.nc",
                    _creating("y", ("y",), group="part_0", sizes={"y": 3}),
                    "part_0/y", "the one axis x"),
    "layer_bounds": ("flat_model_lateral.nc",
                     _storing((1, 0), 31e3, variable="layer_bounds"),
                     "layer_bounds", "contiguous"),
    "layer_part": ("flat_model_lateral.nc", _storing((0,), 3, variable="layer_part"),
                   "layer_part", "which the file lacks"),
    "ngll": ("flat_model_lateral.nc", _setting("ngll", np.int32(1), group="part_0"),
             "part_0.ngll", "two or more"),
    "element_bounds": ("flat_model_lateral.nc",
                       _storing((0, 1), 0.9, variable="element_bounds",
                                group="part_0", relative=True),
                       "part_0", "do not tile"),
    "depth_nodes": ("flat_model_lateral.nc",
                    _storing((1,), 1.001, variable="depth", group="part_0",
                             relative=True),
                    "part_0/depth", "Gauss-Lobatto-Legendre"),
    "radial_rule": ("flat_model_lateral.nc",
                    _setting("radial_interpolation", "spline", **_FLAT_RHO),
                    "part_0/rho", "unknown radial rule"),
    "lateral_rule": ("flat_model_lateral.nc",
                     _setting("lateral_interpolation", "harmonic", **_FLAT_RHO),
                     "part_0/rho", "no known lateral rule"),
    "no_rank": ("flat_model_lateral.nc", _deleting("rank", **_FLAT_RHO),
                "part_0/rho", "lacks the attribute rank"),
    "log_lateral": ("flat_model_lateral.nc",
                    _setting("radial_interpolation", "log-polynomial", **_FLAT_RHO),
                    "part_0/rho", "a rule in the logarithm"),
    "radial_layers": ("flat_model_lateral.nc",
                      _setting("radial_layers", np.array([1], dtype=np.int32),
                               group="part_0", variable="vp"),
                      "part_0/vp", "names radial_layers"),
    "perturbation": ("flat_model_lateral.nc",
                     _setting("perturbation", "nonesuch", **_FLAT_VS),
                     "part_1/vs", "which the part lacks"),
    "perturbation_kind": ("flat_model_lateral.nc",
                          _setting("perturbation_kind", "odd", **_FLAT_VS),
                          "part_1/vs", "no known perturbation_kind"),
    "stack_between": ("flat_model_stacked.nc",
                      _setting("between", "cubic", **_FLAT_STACK),
                      "geometry/boundary_displacement", "rule between surfaces"),
    "stack_positive": ("flat_model_stacked.nc",
                       _setting("positive", "down", **_FLAT_STACK),
                       "geometry/boundary_displacement", "positive up"),
    "stack_rule": ("flat_model_stacked.nc",
                   _setting("lateral_interpolation", "harmonic", **_FLAT_STACK),
                   "geometry/boundary_displacement", "no known lateral rule"),
    "stack_depth": ("flat_model_stacked.nc",
                    _renaming("boundary_depth", "boundary_radius", group="geometry"),
                    "geometry/boundary_radius", "a spherical file's"),
    "mapping_twice": ("flat_model_stacked.nc",
                      _creating("displacement", ("node",), group="part_0",
                                attributes={"radial_interpolation": "polynomial"}),
                      "geometry", "held twice"),
    "surface_interface": ("flat_model_surfaces.nc",
                          _storing((1,), 7, variable="surface_interface",
                                   group="surfaces"),
                          "surfaces/surface_interface", "interfaces of the file"),
    "surface_name": ("flat_model_surfaces.nc",
                     _renaming("K_shape", "rho", group="surfaces"),
                     "surfaces/rho", "a layer field's name too"),
    "surface_rule": ("flat_model_surfaces.nc",
                     _setting("lateral_interpolation", "harmonic", group="surfaces",
                              variable="shift"),
                     "surfaces/shift", "no known lateral rule"),
    "element_layer": ("flat_model_lateral.nc", _relabelling(group="part_0"),
                      "part_0/element_layer", "which its bounds do not lie in"),
    "stack_above": ("flat_model_stacked.nc",
                    _storing((0,), -0.5, variable="boundary_depth", group="geometry"),
                    "geometry/boundary_depth", "above the free surface"),
    "scale_infinite": ("flat_model_lateral.nc", _setting("scale_mass", np.inf),
                       "scale_mass", "not positive and finite"),
    "interface_depth": ("flat_model_lateral.nc",
                        _storing((1,), 31e3, variable="interface_depth"),
                        "interface_depth", "interfaces of the skeleton"),
    "ngll_fraction": ("flat_model_lateral.nc", _adding_half("ngll", group="part_0"),
                      "part_0.ngll", "whole number"),
    "cubic_thin": ("flat_model_cases/thin.nc",
                   _setting("lateral_interpolation", "cubic", **_FLAT_RHO),
                   "part_0/rho", "states the lateral rule cubic"),
    "stack_order": ("flat_model_stacked.nc",
                    _storing((1,), 1e9, variable="boundary_depth", group="geometry"),
                    "geometry/boundary_depth", "does not increase strictly"),
    "stack_one_surface": ("flat_model_stacked.nc",
                          _Rebuilding(sizes={"geometry/boundary": 1}),
                          "geometry/boundary_depth", "fewer than two surfaces"),
}


def _thin_flat_model(path: Path) -> Path:
    """A two-layer flat model file whose density varies laterally, on a grid
    of two nodes along y and stating the bilinear rule: valid, and on a
    grid that cannot carry the bicubic rule."""
    from planetmodel import cartesian as flat
    box = _flat_domains()["box"]
    grid = flat.Grid(box, (4, 2))
    two = flat.LayeredIsotropicElastic([0.0, 30e3, 200e3], box, rho=[2800.0, 3300.0],
                                       vp=[6400.0, 8000.0], vs=[3700.0, 4500.0])
    rho = flat.GriddedField.sampled(flat.AnalyticField(
        (30e3, 200e3), lambda d, x, y: 3300.0 + 10.0 * _wave(grid, x, y) + 0.0 * d,
        character=pm.DENSITY), grid, ngll=3)
    write_model(path, two.with_field(1, "rho", rho, replace=True), grid=grid,
                fields=["rho"])
    validate_structure(path)
    return path


def flat_model_cases(directory: Path) -> list[Path]:
    """The malformed copies of the flat model files already in `directory`
    and of `flat_model_cases/thin.nc`, written here: `bad_*.nc`, listed in
    `bad.txt` as the model cases are."""
    root = directory / "flat_model_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    return [_thin_flat_model(root / "thin.nc")] + _malformed(directory, root,
                                                             _BAD_FLAT_MODELS)


def _flat_surface(directory: Path, label: str, grid: Any, *,
                  depth: float | None) -> list[Path]:
    """One flat surface file on `grid`, a field of every time support on
    it, and its expectations."""
    from planetmodel import surfacefile
    from planetmodel.cartesian.surfacefile import read_surface_file, write_surface

    rng = np.random.default_rng(11 + grid.dimension)
    time = np.array([0.0, 1.0, 2.5, 4.0])
    instant = np.array([2.5, 4.0])
    shape = grid.shape
    gx, gy = grid.nodes
    topography = rng.normal(size=shape)
    topography.flat[0] = np.nan
    fields = {
        "topography": surfacefile.SurfaceVariable(
            topography, dimensions=units.LENGTH, meaning="present-day topography"),
        "relief": surfacefile.SurfaceVariable(
            np.broadcast_to(2e3 * (_wave(grid, gx, gy) - 1.0), shape),
            dimensions=units.LENGTH, lateral_interpolation="cubic"),
        "ice_thickness": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node",
            time_interpolation="linear", dimensions=units.LENGTH),
        "load_previous": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node",
            time_interpolation="previous"),
        "load_nodes": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size,) + shape), support="node"),
        "rate": surfacefile.SurfaceVariable(
            rng.normal(size=(time.size - 1,) + shape), support="interval"),
        "instantaneous": surfacefile.SurfaceVariable(
            rng.normal(size=(instant.size,) + shape), support="instant",
            lateral_interpolation="cubic"),
    }
    scales = pm.Scales(length=1e3, mass=1e12, time=3.15576e10)
    path = directory / f"flat_surface_{label}.nc"
    write_surface(path, grid, fields=fields, source="planetmodel fixtures", time=time,
                  instant=instant, depth=depth, scales=scales,
                  meta={"epoch": "present"}, citation=("a work",))
    surfacefile.validate_structure(path)
    data = read_surface_file(path)
    domain = grid.domain
    two = grid.dimension == 2
    lines = [f"scales {scales.length!r} {scales.mass!r} {scales.time!r} "
             f"{scales.temperature!r}",
             f"depth {'-' if data.depth is None else repr(data.depth)}",
             f"domain {'box' if two else 'rectangle'} "
             + " ".join(repr(float(v)) for v in (
                 *np.ravel(np.asarray(domain.bounds, dtype=float)),
                 *domain.padding)),
             "grid " + " ".join(str(n) for n in shape)]
    lines += [f"time {k} {float(t)!r}" for k, t in enumerate(data.time)]
    lines += [f"instant {k} {float(t)!r}" for k, t in enumerate(data.instant)]
    ny = shape[1] if two else 1
    samples = ((0, 1 % ny), (shape[0] // 2, ny // 3), (shape[0] - 1, ny - 1), (0, 0))

    def at(v: np.ndarray, i: int, j: int) -> float:
        return float(v[i, j] if two else v[i])

    for name in fields:
        a = data.variables[name].attributes
        exps = ("-" if "dimensions" not in a else
                " ".join(str(int(e)) for e in np.atleast_1d(a["dimensions"])))
        lines.append(f"variable {name} {a['time_support']} "
                     f"{a['time_interpolation']} {a['lateral_interpolation']} "
                     f"{exps}")
        values = data.variables[name].values
        leading = [None] if values.ndim == len(shape) else [0, values.shape[0] - 1]
        for k in leading:
            v = values if k is None else values[k]
            tag = "-" if k is None else str(k)
            for i, j in samples:
                if np.isnan(at(v, i, j)):
                    lines.append(f"absent {name} {tag} {i} {j}")
                else:
                    lines.append(f"value {name} {tag} {i} {j} {at(v, i, j)!r}")
    history = surfacefile.read_surface(path)
    dated = (("topography", None, "refuse"), ("ice_thickness", 1.75, "refuse"),
             ("ice_thickness", 2.5, "refuse"), ("ice_thickness", 9.0, "hold"),
             ("load_previous", 1.75, "refuse"), ("load_previous", -3.0, "hold"),
             ("load_nodes", 2.5, "refuse"), ("rate", 3.0, "refuse"),
             ("rate", 4.0, "refuse"), ("instantaneous", 2.5, "refuse"))
    for name, date, extrapolation in dated:
        got = history.values(name, date=date, extrapolation=extrapolation)
        tag = "-" if date is None else repr(float(date))
        for i, j in samples:
            if np.isfinite(at(got, i, j)):
                lines.append(f"sample {name} {tag} {extrapolation} {i} {j} "
                             f"{at(got, i, j)!r}")
    # the fields as functions of (x, y), by their lateral rules
    x0, lx = float(grid.x[0]), grid.lengths[0]
    y0, ly = (float(grid.y[0]), grid.lengths[1]) if two else (0.0, 1.0)
    points = ((x0 + 0.31 * lx, y0 + 0.62 * ly), (x0 - 0.4 * lx, y0 + 1.7 * ly),
              (x0 + lx, y0), (float(grid.x[2]), y0 + 0.25 * ly))
    for name, date in (("relief", None), ("ice_thickness", 1.75),
                       ("instantaneous", 4.0)):
        f = history.function(name, date=date)
        tag = "-" if date is None else repr(float(date))
        for x, y in points:
            dx, dy = (float(v) for v in f.lateral_gradient(x, y))
            lines.append(f"function {name} {tag} {x!r} {y!r} {float(f(x, y))!r} "
                         f"{dx!r} {dy!r}")
    expect = directory / f"flat_surface_{label}.expect"
    expect.write_text("\n".join(lines) + "\n")
    return [path, expect]


#: The malformed flat surface files: (an edit by netCDF4, where the fault
#: lies, what the refusal says), each of a copy of `flat_surface_box.nc`.
_BAD_FLAT_SURFACES: dict[str, tuple[Any, str, str]] = {
    "schema_spherical": (_setting("schema", "planetmodel.surface/1"), "schema",
                         "read by planetmodel::ReadSurfaceFile"),
    "schema": (_setting("schema", "planetmodel.surface/9"), "schema",
               "this reader reads planetmodel.surface/2"),
    "no_source": (_deleting("source"), "source", "missing or empty"),
    "coordinate_system": (_setting("coordinate_system", "spherical"),
                          "coordinate_system", "a flat surface file states flat"),
    "no_domain_bounds": (_deleting("domain_bounds"), "domain_bounds",
                         "states its lateral domain"),
    "depth_negative": (_setting("depth", -1.0), "depth", "not a depth"),
    "radius": (_setting("radius", 1.0), "radius", "a spherical file's"),
    "polar_values": (_creating("relief_poles", ("x",)), "relief_poles",
                     "polar values"),
    "grid_nodes": (_storing((2,), 1.02, variable="y", relative=True), "y",
                   "not the nodes of a grid"),
    "time_axis": (_storing((1,), 9.0, variable="time"), "time",
                  "strictly increasing"),
    "support": (_setting("time_support", "sometimes", variable="rate"), "rate",
                "states no time support"),
    "time_rule": (_setting("time_interpolation", "linear", variable="rate"), "rate",
                  "states the time rule"),
    "lateral_rule": (_setting("lateral_interpolation", "harmonic",
                              variable="relief"), "relief",
                     "no known lateral rule"),
    "dimensions": (_setting("time_support", "node", variable="relief"), "relief",
                   "does not run over"),
    "scale_infinite": (_setting("scale_time", np.inf), "scale_time",
                       "not positive and finite"),
    "source_number": (_setting("source", 3.0), "source", "is missing or empty"),
    "created_number": (_setting("created", np.int32(2026)), "created",
                       "is missing or empty"),
    "dimensions_three": (_setting("dimensions", np.asarray([0, 1, 0], dtype=np.int32),
                                  variable="relief"), "relief", _EXPONENTS),
    "dimensions_text": (_setting("dimensions", "length", variable="relief"),
                        "relief", _EXPONENTS),
    "dimensions_fraction": (_setting("dimensions", np.asarray([0.0, 0.5, 0.0, 0.0]),
                                     variable="relief"), "relief", _EXPONENTS),
}


def flat_surface_files(directory: Path) -> list[Path]:
    """The flat surface files of the module docstring, a box's below the
    surface and a rectangle's on it, each with its expectations, and the
    malformed copies of the box's (`flat_surface_cases/`)."""
    import netCDF4

    from planetmodel import cartesian as flat
    domains = _flat_domains()
    out = _flat_surface(directory, "box", flat.Grid(domains["padded"], (10, 6)),
                        depth=2.5)
    out += _flat_surface(directory, "rectangle", flat.Grid(domains["rectangle"], (9,)),
                         depth=None)
    root = directory / "flat_surface_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    lines = []
    for label, (edit, where, phrase) in _BAD_FLAT_SURFACES.items():
        path = root / f"bad_{label}.nc"
        shutil.copyfile(directory / "flat_surface_box.nc", path)
        with netCDF4.Dataset(path, "a") as ds:
            edit(ds)
        out.append(path)
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


#: The valid flat manifest the flat refusal cases are made from: a box of
#: two layers exported for MFEM with a model, its model file beside it.
_FLAT_BASE_MANIFEST: dict[str, Any] = {
    "mesh": {"file": "box.mesh", "format": "mfem", "nodes": "reference",
             "read_options": {"generate_edges": 1, "refine": 0,
                              "fix_orientation": False},
             "displacement": "displacement"},
    "coordinates": ["x", "y", "z"],
    "layers": [
        {"attribute": 1, "name": "crust", "depth_top": 0.0, "depth_bottom": 30.0,
         "in_geometry": True, "fluid": False},
        {"attribute": 2, "name": "mantle", "depth_top": 30.0, "depth_bottom": 200.0,
         "in_geometry": True, "fluid": False}],
    "interfaces": [
        {"attribute": 1, "name": "surface", "depth": 0.0, "between_layers": [0, -1],
         "kind": "free", "values": {"rho": [2800.0, None]}},
        {"attribute": 2, "name": "moho", "depth": 30.0, "between_layers": [1, 0],
         "kind": "solid-solid", "values": {"rho": [3300.0, 2800.0]}}],
    "faces": [
        {"attribute": 3, "name": "base", "axis": "z", "value": -200.0, "outward": -1},
        {"attribute": 4, "name": "x_min", "axis": "x", "value": -50.0, "outward": -1},
        {"attribute": 5, "name": "x_max", "axis": "x", "value": 450.0, "outward": 1},
        {"attribute": 6, "name": "y_min", "axis": "y", "value": 0.0, "outward": -1},
        {"attribute": 7, "name": "y_max", "axis": "y", "value": 300.0, "outward": 1}],
    "refinements": [
        {"name": "NearPoints", "size": 5.0, "far_size": 30.0, "decay_width": 20.0,
         "points": 12, "digest": "3f2a"}],
    "fields": [
        {"name": "displacement", "file": "box.displacement.gf",
         "fe_space": "H1_3D_P2", "vdim": 3, "ordering": "byNODES", "rank": 1,
         "weight": 0, "voigt": False, "unit": "km", "layers": [1, 2],
         "radial_degree": None},
        {"name": "rho", "file": "box.rho.gf", "fe_space": "L2_3D_P2", "vdim": 1,
         "ordering": "byNODES", "rank": 0, "weight": 1, "voigt": False,
         "unit": "kg km-3", "layers": [1, 2], "radial_degree": 0}],
    "scales": {"length": 1000.0, "mass": 1.0, "time": 1.0, "temperature": 1.0},
    "constants": {"g": 9.8e-3},
    "model": {"file": "box.model.nc", "schema": "planetmodel.model/3"},
    "meta": {"kind": "test"},
    "schema": "planetmodel.mesh.manifest/7",
}

#: Two faults of the flat base manifest, meeting along a junction: a thrust
#: reaching the surface and leaving through a side, and a splay ending on
#: it; fault k's curve groups are 4k - 3 to 4k.
_FLAT_FAULTS: list[dict[str, Any]] = [
    {"name": "thrust", "attribute": 8, "layers": [1, 2], "hanging_wall": [3, 4, 5],
     "tip": 1, "free_edge": 2, "junction": 3, "meets": ["splay"], "ends_on": [],
     "controls": [[17, 17]], "offset": 3.3e-08,
     "sizing": {"size": 5.0, "far_size": 30.0, "decay_width": 20.0},
     "face_edge": 4},
    {"name": "splay", "attribute": 9, "layers": [1], "hanging_wall": [4],
     "tip": 5, "free_edge": None, "junction": 7, "meets": ["thrust"],
     "ends_on": ["thrust"], "controls": None, "offset": 0, "sizing": None,
     "face_edge": None}]

#: The element attributes the flat base's layers carry beside
#: `_FLAT_FAULTS`: in the crust the thrust's collar (3) and both's (4), in
#: the mantle the thrust's (5).
_FLAT_COLLARS = [[1, 3, 4], [2, 5]]


def _flat_faulted(*edits: Any) -> Any:
    """An edit adding `_FLAT_FAULTS` and their collars to the flat base,
    then doing `edits` in turn."""
    def edit(card: dict[str, Any]) -> None:
        card["faults"] = json.loads(json.dumps(_FLAT_FAULTS))
        for layer, attributes in zip(card["layers"], _FLAT_COLLARS):
            layer["attributes"] = list(attributes)
        for e in edits:
            e(card)
    return edit


def _flat_rectangle(*edits: Any) -> Any:
    """An edit making the flat base the manifest of a rectangle: its
    coordinates (x, z), no y faces, a mesh of two coordinates."""
    def edit(card: dict[str, Any]) -> None:
        card["coordinates"] = ["x", "z"]
        del card["faces"][3:]
        card["fields"][0].update(fe_space="H1_2D_P2", vdim=2)
        card["fields"][1]["fe_space"] = "L2_2D_P2"
        for e in edits:
            e(card)
    return edit


#: The variants of the flat base the reader accepts.
_ACCEPTED_FLAT_MANIFESTS = {
    "optional_keys_absent": _both(_drop("refinements"), _drop("model"),
                                  _drop("scales", "temperature"), _drop("meta"),
                                  _drop("constants")),
    "unknown_key": _set("extra", value={"anything": [1, 2]}),
    "no_model_said": _both(*(_set("layers", i, "fluid", value=None)
                             for i in range(2)),
                           *(_set("interfaces", i, "kind", value=None)
                             for i in range(2)),
                           *(_set("interfaces", i, "values", value=None)
                             for i in range(2)),
                           _set("scales", value=None), _set("constants", value={})),
    "rectangle": _flat_rectangle(),
    "buried_top": _both(_set("layers", 0, "depth_top", value=5.0),
                        _set("interfaces", 0, "depth", value=5.0)),
    "faults": _flat_faulted(),
}

#: The malformed variants of the flat base: (edit, where the fault lies,
#: what the refusal says).
_BAD_FLAT_MANIFESTS = {
    "schema_5": (_set("schema", value="planetmodel.mesh.manifest/5"), "schema",
                 "read it with planetmodel::MeshManifest"),
    "schema_6": (_set("schema", value="planetmodel.mesh.manifest/6"), "schema",
                 "a spherical mesh's"),
    "schema_8": (_set("schema", value="planetmodel.mesh.manifest/8"), "schema",
                 "regenerate the manifest"),
    "schema_other": (_set("schema", value="something/1"), "schema",
                     "not a mesh manifest"),
    "format": (_set("mesh", "format", value="vtk"), "mesh.format", "not msh or mfem"),
    "no_coordinates": (_drop("coordinates"), "", 'no member "coordinates"'),
    "coordinates": (_set("coordinates", value=["x", "y"]), "coordinates",
                    "not [x, y, z] or [x, z]"),
    "no_layers": (_set("layers", value=[]), "layers", "lists no layers"),
    "layer_attribute": (_set("layers", 1, "attribute", value=3),
                        "layers[1].attribute", "expected 2"),
    "layer_radii": (_both(_drop("layers", 0, "depth_top"),
                          _set("layers", 0, "r_inner", value=0.0)),
                    "layers[0]", 'no member "depth_top"'),
    "layer_upside_down": (_set("layers", 0, "depth_bottom", value=-30.0),
                          "layers[0]", "not 0 <= depth_top < depth_bottom"),
    "layer_gap": (_set("layers", 1, "depth_top", value=31.0), "layers[1]",
                  "the layer above ends at 30"),
    "layer_shell": (_set("layers", 1, "in_geometry", value=False),
                    "layers[1].in_geometry", "every layer of a flat mesh"),
    "interface_count": (_drop("interfaces", 1), "interfaces",
                        "one per layer, its top"),
    "interface_depth": (_set("interfaces", 1, "depth", value=31.0),
                        "interfaces[1].depth", "not the top of layers[1]"),
    "interface_between": (_set("interfaces", 1, "between_layers", value=[0, 1]),
                          "interfaces[1].between_layers", "expected [1, 0]"),
    "interface_kind": (_set("interfaces", 0, "kind", value="rigid"),
                       "interfaces[0].kind", "not a kind of interface"),
    "no_faces": (_drop("faces"), "", 'no member "faces"'),
    "faces_missing": (_drop("faces", 4), "faces",
                      "not [base, x_min, x_max, y_min, y_max]"),
    "faces_of_a_box_over_a_rectangle": (_set("coordinates", value=["x", "z"]),
                                        "faces", "not [base, x_min, x_max]"),
    "face_attribute": (_set("faces", 2, "attribute", value=9), "faces[2].attribute",
                       "expected 5, after the 2 interfaces"),
    "face_outward": (_set("faces", 1, "outward", value=1), "faces[1]",
                     "not x with -1"),
    "face_base": (_set("faces", 0, "value", value=-150.0), "faces[0].value",
                  "not the last layer's bottom"),
    "face_sides": (_set("faces", 3, "value", value=400.0), "faces[3].value",
                   "not the first below the second"),
    "face_name": (_set("interfaces", 1, "name", value="x_min"), "faces[1].name",
                  "shares its name with an interface"),
    "refinement_size": (_set("refinements", 0, "size", value=0.0), "refinements[0]",
                        "not 0 < size <= far_size"),
    "field_layers": (_set("fields", 1, "layers", value=[1, 3]), "fields[1].layers",
                     "not an attribute in 1..2"),
    "displacement": (_set("mesh", "displacement", value="shift"),
                     "mesh.displacement", "which fields holds no record for"),
    "attributes_without_faults": (_set("layers", 0, "attributes", value=[1]),
                                  "layers[0].attributes", "with its faults or neither"),
    "faults_without_attributes": (_flat_faulted(_drop("layers", 1, "attributes")),
                                  "layers[1]", 'no member "attributes"'),
    "fault_attribute": (_flat_faulted(_set("faults", 0, "attribute", value=3)),
                        "faults[0].attribute",
                        "expected 8, after the 2 interfaces and the 5 faces"),
    "fault_face_name": (_flat_faulted(_set("faults", 1, "name", value="base"),
                                      _set("faults", 0, "meets", value=["base"])),
                        "faults[1].name", "shares its name with a face of the box"),
    "fault_no_face_edge": (_flat_faulted(_drop("faults", 0, "face_edge")),
                           "faults[0]", 'no member "face_edge"'),
    "fault_tip_group": (_flat_faulted(_set("faults", 1, "tip", value=4)),
                        "faults[1].tip", "expected 5 or null"),
    "fault_junction_group": (_flat_faulted(_set("faults", 1, "junction", value=6)),
                             "faults[1].junction", "expected 7 or null"),
    "fault_face_edge_group": (_flat_faulted(_set("faults", 0, "face_edge", value=5)),
                              "faults[0].face_edge", "expected 4 or null"),
    "fault_layers": (_flat_faulted(_set("faults", 0, "layers", value=[3])),
                     "faults[0].layers", "not attributes of layers of the geometry"),
    "fault_meets": (_flat_faulted(_set("faults", 1, "meets", value=[])),
                    "faults[0].meets", "not every one of them meets thrust"),
    "collar_order": (_flat_faulted(_set("layers", 0, "attributes", value=[1, 4, 3])),
                     "layers[0].attributes", "in increasing order"),
    "collar_unowned": (_flat_faulted(_set("faults", 0, "hanging_wall", value=[3, 4]),
                                     _set("faults", 1, "hanging_wall", value=[4])),
                       "layers[1].attributes", "in no fault's hanging wall"),
}


def flat_manifest_cases(directory: Path) -> list[Path]:
    """The flat manifests of the module docstring, under
    `flat_manifest_cases/`: the base, the accepted variants and the
    malformed ones, with `bad.txt`.  None needs gmsh."""
    root = directory / "flat_manifest_cases"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()

    def edited(edit: Any) -> dict[str, Any]:
        card = json.loads(json.dumps(_FLAT_BASE_MANIFEST))
        edit(card)
        return card

    def dump(name: str, card: dict[str, Any]) -> Path:
        path = root / f"{name}.json"
        path.write_text(json.dumps(card, indent=2, allow_nan=False) + "\n")
        return path

    out = [dump("base", _FLAT_BASE_MANIFEST)]
    out += [dump(f"accept_{label}", edited(edit))
            for label, edit in _ACCEPTED_FLAT_MANIFESTS.items()]
    lines = []
    for label, (edit, where, phrase) in _BAD_FLAT_MANIFESTS.items():
        out.append(dump(f"bad_{label}", edited(edit)))
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


def _flat_fault_cases() -> dict[str, dict[str, Any]]:
    """The faults of the flat meshes with faults, by case and name, in the
    unit box of `_flat_mesh_geometry`: a plane reaching the free surface
    and leaving through the side x = 1 (a tip, a free edge and a face
    edge), and a thrust with a splay ending on it and a buried fault
    beside them (junctions, on the splay's boundary and inside the
    thrust; a fault of a tip alone)."""
    from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
    local, zero = CartesianFrame(), Slip(np.zeros((1, 1, 3)))
    dip, width = 50.0, 0.25
    side = Plane.at_depth(local, 0.9, 0.0, 0.5 * width * np.sin(np.deg2rad(dip)),
                          strike=90.0, dip=dip, length=0.4, width=width, slip=zero)
    dip, width = 45.0, 0.3
    thrust = Plane.at_depth(local, 0.3 + 0.5 * width * np.cos(np.deg2rad(dip)), 0.35,
                            0.5 * width * np.sin(np.deg2rad(dip)), strike=0.0, dip=dip,
                            length=0.5, width=width, slip=zero)
    foot = thrust.cartesian(-0.1, 0.1)
    splay = Plane(local, foot, strike=180.0, dip=60.0, length=0.25, width=0.1,
                  slip=zero)
    splay = Plane(local, 2.0 * foot - splay.cartesian(0.0, 0.5), strike=180.0,
                  dip=60.0, length=0.25, width=0.1, slip=zero)
    buried = Plane.at_depth(local, 0.7, -0.25, 0.6, strike=60.0, dip=75.0, length=0.25,
                            width=0.2, slip=zero)
    return {"side": {"fault": Fault([side])},
            "several": {"thrust": Fault([thrust]), "splay": Fault([splay]),
                        "buried": Fault([buried])}}


def _flat_mesh_geometry(kind: str = "box") -> Any:
    """The unit box of the flat meshes, a crust over a mantle, or its
    rectangle."""
    from planetmodel import cartesian as flat
    domain = (flat.Box([(0.0, 1.0), (-0.5, 0.5)]) if kind == "box"
              else flat.Rectangle((0.0, 1.0)))
    return flat.Geometry(pm.Skeleton([0.0, 0.4, 1.0]), domain,
                         layer_names=["crust", "mantle"],
                         interface_names=["surface", "moho"])


def flat_manifests(directory: Path) -> list[Path]:
    """The flat manifests the Python writes (schema 7), each with its walk,
    under `flat_manifests/`: the mesher's of a box and of a rectangle, of a
    box with a fault leaving through a side, and `export_mfem`'s of a flat
    model under topography with its model file beside it (referential) and
    of a rectangle's (physical).  None without gmsh and PyMFEM."""
    try:
        import mfem.ser  # noqa: F401

        from planetmodel.mesh3d import (EmbeddedFault, MeshSpec, UniformInterfaces,
                                        build_layered_mesh, export_mfem)
    except ImportError as error:
        print(f"no flat manifests without gmsh and PyMFEM: {error}", file=sys.stderr)
        return []
    from planetmodel import cartesian as flat
    root = directory / "flat_manifests"
    shutil.rmtree(root, ignore_errors=True)
    box, cut = _flat_mesh_geometry("box"), _flat_mesh_geometry("rectangle")
    coarse = UniformInterfaces(0.3, 0.5, 0.5)

    def model_on(geometry: Any) -> Any:
        layers = []
        for lay, rho, vs in zip(geometry.layers, (2.8, 3.3), (3.7, 4.5)):
            iv = lay.interval
            layers.append({"rho": flat.constant_field(iv, rho, character=pm.DENSITY),
                           "vs": flat.constant_field(iv, vs)})
        return flat.Model(geometry, layers)

    def ridge(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 0.025 * (1.0 + np.cos(2.0 * np.pi * (x - 0.6))) * (
            1.0 + 0.25 * np.cos(2.0 * np.pi * y))

    raised = box.stretched(flat.layer_linear(box.skeleton, [
        flat.GriddedFunction.sampled(ridge, flat.Grid(box.domain, (16, 16))), None,
        None]))
    written = []
    with tempfile.TemporaryDirectory() as scratch:
        work = Path(scratch)

        def built(label: str, spec: MeshSpec) -> Any:
            return build_layered_mesh(spec, work / label / label)

        plain = built("box", MeshSpec(box, coarse, order=1))
        rectangle = built("rectangle", MeshSpec(cut, coarse, order=2))
        relief = built("relief", MeshSpec(raised, coarse, order=2))
        faulted = built("faulted", MeshSpec(
            box, UniformInterfaces(0.2, 0.4, 0.4), order=1,
            faults=[EmbeddedFault(f, name=n, sizing=0.08)
                    for n, f in _flat_fault_cases()["side"].items()]))
        cases = {
            "box": plain.manifest_path,
            "rectangle": rectangle.manifest_path,
            "faulted": faulted.manifest_path,
            "referential": export_mfem(
                relief, work / "referential" / "referential", model=model_on(raised),
                delivery="referential",
                model_file=flat.Grid(raised.domain, (8, 8))).manifest_path,
            "physical_rectangle": export_mfem(
                rectangle, work / "physical_rectangle" / "physical_rectangle",
                model=model_on(cut), delivery="physical",
                model_file=True).manifest_path,
        }
        for label, manifest in cases.items():
            target = root / label
            shutil.copytree(manifest.parent, target)
            path = target / manifest.name
            written += [path, _write_walk(path)]
    return written


def flat_fault_meshes(directory: Path) -> list[Path]:
    """The flat meshes with faults of the module docstring, under
    `flat_fault_meshes/`, each held to `testing.check_fault_mesh` and the
    answers of the Python's rules written beside it, the box's faces
    given.  None without gmsh."""
    from planetmodel.mesh3d import EmbeddedFault, MeshSpec, UniformInterfaces, read_mesh
    from planetmodel.mesh3d import manifest as mf
    try:
        from planetmodel.mesh3d import build_layered_mesh
    except ImportError as error:
        print(f"no flat meshes with faults without gmsh: {error}", file=sys.stderr)
        return []
    from planetmodel import testing

    geometry = _flat_mesh_geometry("box")
    root = directory / "flat_fault_meshes"
    shutil.rmtree(root, ignore_errors=True)
    written = []
    with tempfile.TemporaryDirectory() as scratch:
        for label, faults in _flat_fault_cases().items():
            spec = MeshSpec(geometry, UniformInterfaces(0.2, 0.4, 0.4), order=1,
                            faults=[EmbeddedFault(f, name=n, sizing=0.08)
                                    for n, f in faults.items()])
            res = build_layered_mesh(spec, Path(scratch) / label / label)
            testing.check_fault_mesh(res.mesh_path, faults, mapping=res.mapping,
                                     distance=1e-9)
            card = mf.read(res.manifest_path)
            mesh = read_mesh(res.mesh_path)
            cells, layers = mesh.cells, mesh.cell_attributes
            boundary, attributes = mesh.faces, mesh.face_attributes
            x = np.asarray(mesh.points, dtype=float)
            faces = np.flatnonzero(np.isin(attributes, [f["attribute"]
                                                         for f in card.faults]))
            pairs = mesh.face_cells(boundary[faces])
            plus = mf.plus_cell(x, boundary[faces], cells[pairs])
            curves = mf.fault_curves([boundary[attributes == f["attribute"]]
                                      for f in card.faults],
                                     boundary[attributes == card.surface_attribute],
                                     faces=mesh.boundary_of(card.face_attributes))
            walls = mesh.walls(card)

            target = root / label
            target.mkdir(parents=True)
            manifest_path = target / f"{label}.json"
            shutil.copyfile(res.manifest_path, manifest_path)
            lines = [f"vertex {a!r} {b!r} {c!r}" for a, b, c in x.tolist()]
            lines += [f"cell {a} {b} {c} {d} {g}"
                      for (a, b, c, d), g in zip(cells.tolist(), layers.tolist())]
            lines += [f"boundary {a} {b} {c} {g}"
                      for (a, b, c), g in zip(boundary.tolist(), attributes.tolist())]
            mesh_path = target / f"{label}.mesh.txt"
            mesh_path.write_text("\n".join(lines) + "\n")
            expect = [f"pair {b} {p} {q} {k}"
                      for b, (p, q), k in zip(faces.tolist(), pairs.tolist(),
                                              plus.tolist())]
            for record, edges, wall in zip(card.faults, curves, walls):
                for key in ("tip", "free_edge", "junction", "junction_end",
                            "face_edge"):
                    expect += [f"edge {record['name']} {key} {a} {b}"
                               for a, b in getattr(edges, key).tolist()]
                expect += [f"ends {record['name']} {card.faults[j]['name']}"
                           for j in edges.ends_on]
                expect += [f"wall {record['name']} {c} {w}"
                           for c, w in zip(np.flatnonzero(wall).tolist(),
                                           wall[wall != 0].tolist())]
            expect_path = target / f"{label}.expect"
            expect_path.write_text("\n".join(expect) + "\n")
            written += [manifest_path, mesh_path, expect_path]
    return written


def write_fixtures(directory: str | Path, *, meshes: bool = True) -> list[Path]:
    """Every fixture into `directory`, each model file validated, the
    spherical ones and then the flat ones (`planetmodel.cartesian`); the
    manifests and the meshes with faults, which build meshes, only where
    `meshes` is true."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = model_files(directory)
    for path in models:
        validate_structure(path)
    return (models + model_cases(directory) + gll(directory) + grids(directory)
            + vocabulary(directory) + surface_files(directory)
            + fault_files(directory) + slip_from_rake(directory) + burgers(directory)
            + lateral_cases(directory) + radial_cases(directory)
            + constants(directory) + frames(directory) + json_documents(directory)
            + manifest_cases(directory)
            + flat_model_files(directory) + flat_model_cases(directory)
            + flat_lateral_cases(directory) + flat_surface_files(directory)
            + flat_manifest_cases(directory)
            + (manifests(directory) + cubed_manifests(directory)
               + fault_meshes(directory)
               + flat_manifests(directory) + flat_fault_meshes(directory)
               if meshes else []))


if __name__ == "__main__":
    here = Path(__file__).parent / "fixtures"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    for written in write_fixtures(target):
        print(written)
