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
power-law creep beside them (`model_prony.nc`).

Beside them, `model_kinds.nc` (an absolute perturbation of `rho` and a
log one of `viscosity`), `model_tensors.nc` (a rank-2 `stress`, radial
on one layer and gridded on the other, and a rank-4 `elastic`) and
`model_harmonic.nc` (a copy of the lateral one stating the harmonic
rule, which the C++ refuses); and for every model file but that one
`<model>.expect`, from the Python's own reading of the file, one fact
per line: `scales`, `rtol`, `name`, `citation` (its works a line each)
and the `layer_name` and `interface_name`s (UTF-8 in hexadecimal, `-`
for none), each `constant`
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
says.  **GLL points** (`gll.txt`): `gll <ngll>` and the points on [-1,
1].

**Surface files** (`surface_basic.nc`): one surface file with a field
of every time support on a Gauss-Legendre grid, polar values on two of
them, a NaN-marked point, geophysical scales, a radius, meta and a
citation; beside it `surface_basic.expect`, one fact per line:
`scales`, `radius`, `grid <kind> <lmax> <ntheta> <nphi>`,
`time <k> <value>` and `instant <k> <value>`, per field
`variable <name> <support> <time rule> <lateral rule> <m l t T | ->`,
sampled entries `value <name> <k|-> <i> <j> <value>` (`absent` where
the point is NaN), `poles <name> <k|-> <north> <south>`, and the
history's reading
`sample <name> <date|-> <refuse|hold> <i> <j> <value>`, from
`SurfaceHistory.values` at dates between nodes, at nodes, held outside
the span, inside and closing an interval, and at an instant.

**Interpolation cases** (`case_lateral_*.nc`, `case_radial_*.nc`), one
netCDF file each: for a lateral case the grid (`colatitude`,
`longitude`, radians), its `kind`, the `values` at its nodes and the
`poles`, the `method`, the query points `query_colatitude` and
`query_longitude`, and the `expected` values with their derivatives
`expected_dtheta` and `expected_dphi`; for a radial case the element
`breaks`, `ngll`, the node `radius` and the `values` there (element by
element), the `method`, the query radii and the `expected` values and
derivatives.
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
array or an object).

**Decks** (`deck_*.txt`): the two shipped mineos decks as they are
(nine and six columns), a GIA deck, and a small custom one with layers
of two, three, four and six knots; each with `<deck>.expect` beside it,
one fact per line: `format`, `knots`, `boundaries`, the mineos
`header` fields (a string in hexadecimal), the `layer_name` and
`interface_name` of every layer and interface that `mineos_names`
gives (`-` for none) and the count of what it `ignored`, and for each
kind the C++ reads, `value <kind> <layer> <column> <r> <value>
<derivative>` at the layer's ends and between its knots, or `absent
<layer> <column>`.

**Manifests to refuse** (`manifest_cases/`): a valid manifest, the
format page's example with a refinement and a model file, as
`base.json`; variants the reader accepts (`accept_*.json`: the keys
added within the schema left out, an unknown key added); and one
malformed copy per check `validate_structure` makes (`bad_*.json`),
listed in `bad.txt`, one per line, as the case, a tab, the JSON path
the fault lies at (empty for the document itself), a tab, and a phrase
the refusal says.  None needs gmsh.

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

import planetmodel as pm
from planetmodel import units
from planetmodel.lateral import LateralInterpolant
from planetmodel.mesh1d import Mesh1D
from planetmodel.modelfile import validate_structure, write_model


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
    return out


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
    return [
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
    ]


def with_burgers(model: pm.Model) -> pm.Model:
    """The model with its viscosity made a Burgers body's two branches."""
    from planetmodel.conversion import building, prony
    burgers = prony.with_branches(model, prony.burgers_series(strength=5.0,
                                                              viscosity_ratio=5.0))
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


#: The kinds of deck interpolant the C++ reads, of `deck.KINDS`.
_CPP_KINDS = ("cubic", "natural", "linear")


def _expect_deck(path: Path, deck: Any, *, mineos: bool) -> Path:
    """The expectation file of a deck, by the module docstring."""
    import warnings

    from planetmodel.deck import deck_layers, mineos_names
    lines = [f"knots {deck.nknots}",
             "boundaries " + " ".join(repr(float(b)) for b in deck.boundaries)]
    if mineos:
        h = deck.header
        lines.append(f"header name {h['name'].encode('utf-8').hex()}")
        lines += [f"header {k} {h[k]!r}" for k in
                  ("ifanis", "tref", "ifdeck", "nknot", "nic", "noc")]
        lines.append("extras2 " + " ".join(h["extras"][0]))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            layers, faces = mineos_names(deck)
        lines += [f"layer_name {i} {n or '-'}" for i, n in enumerate(layers)]
        lines += [f"interface_name {i} {n or '-'}" for i, n in enumerate(faces)]
        lines.append(f"ignored {len(caught)}")
    rng = np.random.default_rng(len(deck.radius))
    for kind in _CPP_KINDS:
        _, fields = deck_layers(deck, kind=kind)
        for i, s in enumerate(deck.layers()):
            r = deck.radius[s]
            between = np.sort(rng.uniform(r[0], r[-1], 3))
            radii = np.concatenate(([r[0]], between, [r[-1]]))
            for name in deck.names:
                if name not in fields[i]:
                    lines.append(f"absent {i} {name}")
                    continue
                f, df = fields[i][name], fields[i][name].derivative()
                for x in radii:
                    lines.append(f"value {kind} {i} {name} {float(x)!r} "
                                 f"{float(f(x))!r} {float(df(x))!r}")
    out = path.with_name(path.name.removesuffix(".txt") + ".expect")
    out.write_text("\n".join(lines) + "\n")
    return out


def decks(directory: Path) -> list[Path]:
    """The decks of the module docstring, each with its expectations."""
    from planetmodel.deck import GIA, MINEOS, Deck, DeckFormat, read_deck, write_deck
    data = Path(__file__).resolve().parents[2] / "examples" / "data"
    out = []
    for label, source in (("prem200", "prem.200"), ("nocrust", "prem.nocrust")):
        path = directory / f"deck_{label}.txt"
        shutil.copyfile(data / source, path)
        out += [path, _expect_deck(path, read_deck(path, MINEOS), mineos=True)]
    prem = read_deck(data / "prem.200", MINEOS)
    viscosity = np.full(prem.nknots, np.nan)
    for i, s in enumerate(prem.layers()):
        if i >= 2:
            viscosity[s] = 1e21 * (1.0 + 0.5 * (prem.radius[s] / 6371e3))
    gia = Deck(prem.radius, {**{n: prem[n] for n in ("rho", "vpv", "vsv", "vph", "vsh",
                                                      "eta")},
                             "viscosity": viscosity}, header={"title": "a GIA deck"})
    path = write_deck(directory / "deck_gia.txt", gia, GIA)
    out += [path, _expect_deck(path, read_deck(path, GIA), mineos=False)]
    knots = Deck([0.0, 0.4, 0.4, 0.5, 0.7, 0.7, 0.75, 0.8, 0.9, 0.9, 0.92, 0.94, 0.95,
                  0.97, 0.98, 1.0],
                 {"rho": [9.0, 8.5, 6.0, 5.8, 5.5, 4.0, 3.9, 3.8, 3.6, 3.0, 2.95, 2.9,
                          2.85, 2.8, 2.7, 2.6],
                  "vp": [10.0, 9.9, 8.0, 8.1, 7.9, 7.0, 7.05, 6.9, 6.8, 6.0, 6.01, 5.99,
                         5.97, 5.9, 5.8, 5.7]},
                 header={"lines": ("layers of two, three, four and six knots",)})
    custom = DeckFormat(("rho", "vp"), header_lines=1)
    path = write_deck(directory / "deck_knots.txt", knots, custom)
    out += [path, _expect_deck(path, read_deck(path, ("rho", "vp"), header_lines=1),
                               mineos=False)]
    return out


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


#: The variants the reader accepts: the keys added within the schema left
#: out, and an unknown key added.
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
    data, model = read_model_file(path), read_model(path)
    hexed = (lambda x: "-" if not x else x.encode("utf-8").hex())
    s = data.scales
    lines = [f"scales {s.length!r} {s.mass!r} {s.time!r} {s.temperature!r}",
             f"rtol {data.rtol!r}", f"name {hexed(data.name)}",
             f"citation {hexed(chr(10).join(data.citation))}"]
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


def _renaming(old: str, new: str, *, group: str) -> Any:
    """An edit renaming a variable of a group."""
    return lambda ds: ds.groups[group].renameVariable(old, new)


_P = {"group": "part_0"}
_RHO = {"group": "part_0", "variable": "rho"}
_BETWEEN = {"group": "geometry", "variable": "boundary_displacement"}
_S = {"group": "surfaces"}

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
}


def model_cases(directory: Path) -> list[Path]:
    """The harmonic copy, the expectations, and the malformed copies, of
    the model files already in `directory`."""
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
    lines = []
    for label, (base, edit, where, phrase) in _BAD_MODELS.items():
        path = root / f"bad_{label}.nc"
        shutil.copyfile(directory / base, path)
        with netCDF4.Dataset(path, "a") as ds:
            edit(ds)
        out.append(path)
        lines.append(f"{label}\t{where}\t{phrase}")
    index = root / "bad.txt"
    index.write_text("\n".join(lines) + "\n")
    return out + [index]


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


def surface_files(directory: Path) -> list[Path]:
    """The surface file of the module docstring, and its expectations."""
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
    }
    scales = pm.Scales.geophysical(6.371e6)
    path = directory / "surface_basic.nc"
    surfacefile.write_surface(path, grid, fields=fields,
                              source="planetmodel fixtures", time=time,
                              instant=instant, radius=1.0, scales=scales,
                              meta={"epoch": "present"}, citation=("a work",))
    surfacefile.validate_structure(path)
    data = surfacefile.read_surface_file(path)
    lines = [f"scales {scales.length!r} {scales.mass!r} {scales.time!r} "
             f"{scales.temperature!r}",
             f"radius {data.radius!r}",
             f"grid {grid.kind} {grid.lmax} {grid.ntheta} {grid.nphi}"]
    lines += [f"time {k} {float(t)!r}" for k, t in enumerate(data.time)]
    lines += [f"instant {k} {float(t)!r}" for k, t in enumerate(data.instant)]
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
    for name, date, extrapolation in (("topography", None, "refuse"),
                                      ("ice_thickness", 1.75, "refuse"),
                                      ("ice_thickness", 2.5, "refuse"),
                                      ("ice_thickness", 9.0, "hold"),
                                      ("load_previous", 1.75, "refuse"),
                                      ("load_previous", -3.0, "hold"),
                                      ("load_nodes", 2.5, "refuse"),
                                      ("rate", 3.0, "refuse"),
                                      ("rate", 4.0, "refuse"),
                                      ("instantaneous", 2.5, "refuse")):
        at = history.values(name, date=date, extrapolation=extrapolation)
        tag = "-" if date is None else repr(float(date))
        for i, j in samples:
            if np.isfinite(at[i, j]):
                lines.append(f"sample {name} {tag} {extrapolation} {i} {j} "
                             f"{float(at[i, j])!r}")
    expect = directory / "surface_basic.expect"
    expect.write_text("\n".join(lines) + "\n")
    return [path, expect]


def write_fixtures(directory: str | Path, *, meshes: bool = True) -> list[Path]:
    """Every fixture into `directory`, each model file validated; the
    manifests, which build meshes, only where `meshes` is true."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    models = model_files(directory)
    for path in models:
        validate_structure(path)
    return (models + model_cases(directory) + gll(directory) + grids(directory)
            + vocabulary(directory) + surface_files(directory)
            + lateral_cases(directory) + radial_cases(directory)
            + constants(directory) + frames(directory) + json_documents(directory)
            + manifest_cases(directory) + decks(directory)
            + (manifests(directory) if meshes else []))


if __name__ == "__main__":
    here = Path(__file__).parent / "fixtures"
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else here
    for written in write_fixtures(target):
        print(written)
