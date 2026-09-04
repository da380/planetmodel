"""recipe.py -- a TOML file that says what mesh to build.

The API stays primary: a recipe constructs a `MeshSpec` and nothing
else, so anything a recipe can express, a script can express, and the
recipe is a convenience the manifest echoes back.  What it buys is a
mesh that is one command and one small file from its inputs, with a
hash of that file recorded beside the mesh.

A recipe says six things -- the model, the geometry surgery, the
sizing, the surfaces by file, the mapping rule, and where the mesh
goes -- and three conventions carry the rest.

**Lengths wear their units.**  A key ending `_km` or `_m` is read in
those units and the suffix stripped, so `h_ref_km = 20` reaches an
`AngularResolution` as 20000.0 and `radius_km = 950` a boundary as
950000.0.  The model layer is SI and a recipe should not have to be.

**Components are named, not imported.**  `policy`, `reader` and `rule`
resolve through the registry, so a recipe says
`rule = "layer_linear"` and any keys left over in its table become that
component's keyword arguments.  A component that is not registered
cannot be named here, which is the price of a file format that never
executes Python.

**Radii are numbers.**  Every boundary the surgery creates -- the
outer boundary, an inserted interface, an extended shell -- is placed
by a number in metres, and the boundary the truncation makes is named
by `truncate_name` rather than inferred.  A radius a data set decides,
a mean Moho say, is computed by the script that writes the recipe.

Surfaces are built at the model's reference radius and then *centred*,
because an interface radius is by definition the boundary's mean radius
and the relief hung on it has zero mean.  So a
depth-to-Moho grid becomes a boundary at the Moho's own mean radius,
and the recipe's number for that boundary -- its `truncate_at`, or the
`radius_m` of the interface it inserts -- must be that same mean radius.
`read()` checks the two agree to `1e-6 * rref` and says so naming both
numbers if they do not, since the alternative is a refusal from
`with_surface` much later that never mentions the recipe.  Inside that
tolerance the surface is placed at the recipe's radius exactly, so the
boundary sits where the file says it sits.

An unknown key is an error, not a shrug.  A silently ignored
`h_far_kmm` produces a mesh that is wrong in a way nothing downstream
can detect.
"""
from __future__ import annotations

import tomllib
from dataclasses import dataclass
from pathlib import Path

from ..registry import lookup
from .manifest import file_digest

__all__ = ["Recipe", "read", "build"]

#: Length suffixes a key may wear, and what they mean in metres.
_UNITS = {"_km": 1.0e3, "_m": 1.0}

_SECTIONS = {"model", "geometry", "mesh", "sizing", "surfaces", "mapping",
             "output"}

#: "Nothing said yet", distinct from `fields=None`, which says something.
_UNSET = object()


@dataclass(frozen=True)
class Recipe:
    """A parsed recipe: the spec it describes and where it goes."""

    spec: object                    # MeshSpec
    output: Path
    source: Path
    digest: str

    @property
    def command(self) -> str:
        return f"python -m planetmodel.mesh3d {self.source.name}"

    def build(self, **kw):
        """Build the mesh this recipe describes."""
        from ..mesh3d import build_layered_mesh
        return build_layered_mesh(self.spec, self.output, **kw)

    def __repr__(self) -> str:
        return f"Recipe({self.source.name} -> {self.output})"


# -- small parsing helpers -------------------------------------------------

def _check_keys(table: dict, known, where: str) -> None:
    """Reject keys the format does not define, naming the ones it does."""
    unknown = [k for k in table if k not in known]
    if unknown:
        raise ValueError(
            f"[{where}]: unknown key(s) {sorted(unknown)}; this section takes "
            + ", ".join(sorted(known)))


def _lengths(table: dict) -> dict:
    """Strip length suffixes, converting to metres.

    `h_ref_km = 20` becomes `h_ref = 20000.0`.  A key wearing no suffix
    passes through untouched, since not every parameter is a length.
    """
    out = {}
    for key, value in table.items():
        name, converted = key, value
        for suffix, factor in _UNITS.items():
            if not key.endswith(suffix) or isinstance(value, bool):
                continue
            if isinstance(value, (int, float)):
                name, converted = key[: -len(suffix)], float(value) * factor
                break
            if isinstance(value, list):
                name = key[: -len(suffix)]
                converted = tuple(float(v) * factor for v in value)
                break
        if name in out:
            raise ValueError(
                f"{name!r} is given twice, once with a unit suffix and once "
                "without; say it one way")
        out[name] = converted
    return out


def _component(kind: str, table: dict, where: str, *, drop=()):
    """Build a registered component from `name` plus the rest of its table."""
    table = dict(table)
    for key in drop:
        table.pop(key, None)
    name = table.pop("policy", None) or table.pop("rule", None)
    if name is None:
        raise ValueError(f"[{where}]: no component named; give a "
                         f"{'policy' if kind == 'sizing' else 'rule'}")
    cls = lookup(kind, name)
    kw = _lengths(table)
    fields = getattr(cls, "__dataclass_fields__", None)
    if fields is not None:
        # A component's parameters are recipe keys like any other, so a
        # typo in one gets the same answer as a typo in a section key
        # rather than a TypeError from a constructor the author never saw.
        _check_keys(kw, set(fields), where)
    try:
        return cls(**kw)
    except TypeError as exc:
        raise ValueError(f"[{where}]: {name!r} rejected {sorted(kw)}: "
                         f"{exc}") from None


def _metres(value, where: str) -> float:
    """A length in metres, as a recipe is allowed to say it.

    Which is: a number.  The unit suffix on the key has already been
    applied by `_lengths`, so `radius_km = 950` arrives here as 950000.0
    and there is nothing left to interpret.  A radius that a data set
    decides is computed by the script that writes the recipe, which is
    the one place that has the data in hand.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(
            f"{where}: cannot read {value!r} as a length; write a number in "
            "metres, or put a _km or _m suffix on the key. A radius a data "
            "set decides -- a mean Moho, say -- is computed by the script "
            "that writes the recipe.")
    return float(value)


# -- the sections ----------------------------------------------------------

def _read_model(table: dict, root: Path):
    """Load the reference body, and the reference radius that scales it."""
    from .deck import read_deck

    _check_keys(table, {"source", "reader", "columns", "header_lines",
                        "rref_m"}, "model")
    if "source" not in table:
        raise ValueError("[model]: no source file given")
    path = (root / table["source"]).resolve()
    columns = table.get("columns")
    header_lines = table.get("header_lines")

    reader = table.get("reader")
    if reader is not None:
        read = lookup("deck_reader", reader)
        if header_lines is not None and header_lines != 3:
            raise ValueError(
                f"[model]: reader {reader!r} reads three header lines, but the "
                f"recipe says {header_lines}; drop the key or use the generic "
                "reader by giving columns without a reader")
        if columns is not None and tuple(columns) != tuple(read.columns):
            raise ValueError(
                f"[model]: reader {reader!r} reads columns "
                f"{list(read.columns)}, but the recipe says {list(columns)}; "
                "drop the columns key or use a reader that matches the file")
        body = read(path)
    elif columns is not None:
        body = read_deck(path, columns, header_lines=header_lines or 0,
                         meta={"source": str(path)})
    else:
        raise ValueError(
            "[model]: give a reader, or columns for the generic deck reader")

    rref = table.get("rref_m")
    return body, (float(rref) if rref is not None else None)


def _read_surfaces(entries, root: Path, rref: float | None) -> tuple:
    """Build each named surface, centred on its own mean radius.

    Several files sum, which is how a crustal thickness and a depth to
    the Moho make a surface elevation.  The values are read as a
    departure from the model's reference radius, so `rref_m` is what
    turns a depth field into a boundary.

    The files are returned alongside because summing loses them: two
    grids on one lat-lon mesh add their values into a single
    GriddedTopography, and no walk of the finished shape can recover
    which files made it.  The manifest wants them for its hashes, and
    the recipe is the only place that still knows.
    """
    from ..model.topography import GriddedTopography

    surfaces, sources = {}, {}
    for i, entry in enumerate(entries):
        where = f"surfaces[{i}]"
        _check_keys(entry, {"name", "files", "units", "scale"}, where)
        name = entry.get("name")
        if not name:
            raise ValueError(f"[[{where}]]: every surface needs a name")
        files = entry.get("files")
        if not files:
            raise ValueError(f"[[{where}]] {name!r}: no files given")
        if rref is None:
            raise ValueError(
                f"[[{where}]] {name!r}: surfaces are read as a departure from "
                "the model's reference radius, so [model] needs rref_m")

        units = entry.get("units", "m")
        try:
            factor = {"m": 1.0, "km": 1.0e3}[units]
        except KeyError:
            raise ValueError(
                f"[[{where}]] {name!r}: units must be \"m\" or \"km\", got "
                f"{units!r}") from None
        factor *= float(entry.get("scale", 1.0))

        shape, paths = None, []
        for f in files:
            path = (root / f).resolve()
            paths.append({"file": str(path), "units": units,
                          "scale_to_m": float(factor)})
            piece = GriddedTopography.from_xyz(path, scale=factor)
            shape = piece if shape is None else shape + piece
        surfaces[name] = _centred(rref, shape, name)
        sources[name] = paths
    return surfaces, sources


def _centred(rref: float, shape, name: str):
    """A Surface at its own mean radius, carrying zero-mean relief."""
    from ..model.surface import Surface
    return Surface(float(rref), topography=shape, name=name).centred()


def _read_geometry(table: dict, body) -> dict:
    """The surgery: what to drop, where to cut, what to add back.

    Radii arrive as numbers, in metres once `_lengths` has stripped any
    `_km`.  The truncation carries `truncate_name` because the boundary
    it makes is a new one that nothing else can name, and `[[surfaces]]`
    attaches by name.
    """
    table = _lengths(table)
    known = {"drop_outermost_interfaces", "drop_interfaces", "keep_interfaces",
             "truncate_at", "truncate_name", "insert", "buffer"}
    _check_keys(table, known | set(table.get("insert", ())), "geometry")
    out: dict = {}

    n = len(body.interfaces)
    chosen = [k for k in ("drop_outermost_interfaces", "drop_interfaces",
                          "keep_interfaces") if k in table]
    if len(chosen) > 1:
        raise ValueError(
            f"[geometry]: {chosen} say the same thing three ways; give one")
    if "drop_outermost_interfaces" in table:
        k = int(table["drop_outermost_interfaces"])
        # The outermost interface is the body's own boundary and cannot
        # be merged away, so the k dropped are the interior ones below it.
        if not 0 <= k < n - 1:
            raise ValueError(
                f"[geometry]: cannot drop the outermost {k} interfaces of a "
                f"body with {n} (the outer boundary is not one of them)")
        out["drop_interfaces"] = list(range(n - 1 - k, n - 1))
    elif "drop_interfaces" in table:
        out["drop_interfaces"] = [int(i) for i in table["drop_interfaces"]]
    elif "keep_interfaces" in table:
        out["keep_interfaces"] = [int(i) for i in table["keep_interfaces"]]

    outer = float(body.skeleton.boundaries[-1])
    if "truncate_at" in table:
        radius = _metres(table["truncate_at"], "[geometry] truncate_at")
        out["outer_radius"] = radius
        if "truncate_name" in table:
            out["outer_name"] = str(table["truncate_name"])
        outer = radius
    elif "truncate_name" in table:
        raise ValueError(
            "[geometry]: truncate_name names a boundary nothing truncates "
            "to; give truncate_at (or truncate_at_km) as well, or drop the "
            "name")

    inserts, extends = [], []
    fields = _UNSET
    for name in table.get("insert", ()):
        entry = table.get(name)
        if entry is None:
            raise ValueError(
                f"[geometry]: insert lists {name!r} but there is no "
                f"[geometry.{name}] table saying where it goes")
        where = f"[geometry.{name}]"
        entry = _lengths(entry)
        _check_keys(entry, {"radius", "role", "fields"}, f"geometry.{name}")
        if "radius" not in entry:
            raise ValueError(
                f"{where}: no radius; give radius_m or radius_km saying where "
                "the boundary goes")
        radius = _metres(entry["radius"], where)
        role = entry.get("role", "material")
        if radius <= outer and "fields" in entry:
            raise ValueError(
                f"{where}: fields applies to a shell added outside the body; "
                "an inserted boundary splits a layer and keeps its model")
        if radius > outer:
            extends.append((radius, name, role))
            how = entry.get("fields", "extrapolate")
            if how not in ("extrapolate", "none"):
                raise ValueError(
                    f"{where}: fields must be \"extrapolate\" (the layer "
                    f"below's fields continued into the shell) or \"none\" "
                    f"(a shell holding no fields), got {how!r}")
            want = None if how == "none" else "extrapolate"
            if fields is not _UNSET and fields != want:
                raise ValueError(
                    "[geometry]: extended shells disagree about their fields; "
                    "one recipe cannot both extrapolate and hold none")
            fields = want
        else:
            inserts.append((radius, name, role))

    for key, group in (("insert", inserts), ("extend", extends)):
        group.sort()
        if group:
            out[f"{key}_radii"] = [r for r, _, _ in group]
            out[f"{key}_names"] = [nm for _, nm, _ in group]
            roles = {role for _, _, role in group}
            if len(roles) > 1:
                raise ValueError(
                    f"[geometry]: {key}ed boundaries disagree about role "
                    f"({sorted(roles)}); planetmodel carries one role per group")
            out[f"{key}_role"] = roles.pop()
    if fields is not _UNSET:
        out["extend_fields"] = fields

    buffers = table.get("buffer")
    if buffers is not None:
        from ..mesh3d.spec import BufferSpec
        if isinstance(buffers, dict):
            buffers = [buffers]
        specs = []
        for i, buf in enumerate(buffers):
            _check_keys(buf, {"ratio", "radius", "radius_km", "radius_m",
                              "name"}, f"geometry.buffer[{i}]")
            kw = _lengths(buf)
            specs.append(BufferSpec(**kw))
        out["buffers"] = specs
    return out


def _read_mapping(table: dict, surfaces: dict) -> tuple[dict, dict]:
    """The displacement rule, the mode, and any relief exaggeration."""
    known = {"mode", "rule", "exaggeration"}
    _check_keys(table, known | {k for k in table if k.startswith("inner_")
                                or k.startswith("control_")}, "mapping")
    out = {"delivery": table.get("mode", "physical")}
    if "rule" in table:
        out["mapping_rule"] = _component("displacement_rule", table, "mapping",
                                         drop=("mode", "exaggeration"))

    factor = float(table.get("exaggeration", 1.0))
    if factor != 1.0:
        # The relief is exaggerated, not the boundary: each surface is
        # already centred, so scaling leaves every mean radius alone and
        # the geometry the recipe asked for still holds.
        surfaces = {name: surf * factor for name, surf in surfaces.items()}
    return out, surfaces


# -- the format ------------------------------------------------------------

def read(path) -> Recipe:
    """Parse a recipe into the MeshSpec it describes.

    Paths inside the file -- the model deck, the surface grids, the
    output -- are relative to the recipe itself, so a recipe and its
    data move together.
    """
    from ..mesh3d.spec import MeshSpec       # registers the sizing policies

    path = Path(path)
    with open(path, "rb") as fh:
        doc = tomllib.load(fh)
    _check_keys(doc, _SECTIONS, path.name)
    root = path.parent

    body, rref = _read_model(doc.get("model", {}), root)
    surfaces, sources = _read_surfaces(doc.get("surfaces", ()), root, rref)

    mapping_kw, surfaces = _read_mapping(doc.get("mapping", {}), surfaces)
    geometry = _read_geometry(doc.get("geometry", {}), body)

    mesh = dict(doc.get("mesh", {}))
    _check_keys(mesh, {"dimension", "order", "algorithm_2d", "algorithm_3d",
                       "validate"}, "mesh")

    sizing_table = doc.get("sizing")
    if not sizing_table:
        raise ValueError("[sizing]: every mesh needs a sizing policy")
    sizing = _component("sizing", sizing_table, "sizing")

    output = doc.get("output", {})
    _check_keys(output, {"path"}, "output")
    if "path" not in output:
        raise ValueError("[output]: no path given for the mesh")

    for name, surface in list(surfaces.items()):
        radius = _placement(name, geometry, body)
        if radius is None:
            raise ValueError(
                f"[[surfaces]] {name!r} attaches to no boundary: nothing is "
                f"truncated at, inserted at or extended to a boundary called "
                f"{name!r}, and the body has no interface of that name. A "
                "surface that only exists to be measured from still has to sit "
                "somewhere; give the boundary its name")
        surfaces[name] = _placed(name, surface, radius, rref)
    if surfaces and "mapping_rule" not in mapping_kw:
        raise ValueError(
            "[[surfaces]] are defined but [mapping] names no rule, so the "
            "relief would never reach the mesh; add rule = \"layer_linear\"")

    spec = MeshSpec(body=body, sizing=sizing, rref=rref, surfaces=surfaces,
                    meta={"recipe": path.name,
                          "recipe_sha256": file_digest(path),
                          "command": f"python -m planetmodel.mesh3d {path.name}",
                          "surface_sources": sources},
                    **geometry, **mesh, **mapping_kw)
    return Recipe(spec=spec, output=(root / output["path"]).resolve(),
                  source=path, digest=file_digest(path))


def _placement(name: str, geometry: dict, body) -> float | None:
    """The radius the recipe gives the boundary a surface attaches to.

    The surgery's boundaries first, then the body's own interfaces:
    truncation, insertion and extension all put a boundary somewhere the
    recipe chose, and an interface the deck already carries is where the
    deck put it.  `None` says the name matches no boundary at all.
    """
    if name == geometry.get("outer_name"):
        return float(geometry["outer_radius"])
    for key in ("insert", "extend"):
        names = list(geometry.get(f"{key}_names", ()))
        if name in names:
            return float(geometry[f"{key}_radii"][names.index(name)])
    for face in body.interfaces:
        if face.name == name:
            return float(face.radius)
    return None


def _placed(name: str, surface, radius: float, rref: float):
    """The surface, checked against the radius the recipe put it at.

    A centred surface's reference radius *is* the data's mean radius, and
    the boundary it hangs on is placed by a number the recipe author
    wrote.  The two are statements about the same thing, so a
    disagreement is an error here, where both numbers can be shown,
    rather than a refusal from `with_surface` two layers down that has
    never heard of a recipe.  Within tolerance the recipe's number wins:
    the boundary is where the file says it is, and the last bits of the
    quadrature do not move it.
    """
    tol = 1.0e-6 * abs(float(rref))
    if abs(surface.reference_radius - radius) > tol:
        raise ValueError(
            f"[[surfaces]] {name!r} is centred on mean radius "
            f"{surface.reference_radius:.10g} m -- that is what its files "
            f"say -- but the recipe places the boundary called {name!r} at "
            f"{radius:.10g} m, a difference of "
            f"{surface.reference_radius - radius:.6g} m. An interface radius "
            "is the boundary's mean radius, so the two have to agree: write "
            "the data's own mean radius here, computing it in the script that "
            "writes the recipe if the data decides it.")
    return surface.at(radius)


def build(path, **kw):
    """Read a recipe and build its mesh."""
    return read(path).build(**kw)
