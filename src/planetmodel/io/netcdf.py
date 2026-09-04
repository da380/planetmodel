"""netcdf.py -- the reference-model format, `planetmodel.model/1`.

One file serves the spectral codes, planetmodel's own storage, and the
input form for real 3D models.  It is netCDF-4 with dimension order
`(node, colatitude, longitude, components...)`, and it writes what a
model *is* -- fields on the reference body, the mapping as a
displacement -- and nothing a consumer can form for itself.  The
reader-side description of the layout is `docs/formats/model_netcdf.md`.

The layout is the `Sample` of `planetmodel.sampling`: `/radial` is the
sample's `RadialMesh` flattened element by element, every interface a
repeated radius with one node each side; `/angular` is the node arrays
of the `AngularGrid`; each variable under `/fields` is `(node,
colatitude, longitude, c...)`, or `(node, c...)` for a field that does
not depend on direction, carrying its character, physical dimensions,
frame and `layers` (the layer indices it is defined on) as attributes,
with the `_FillValue` on the nodes of every other layer;
`/mapping/displacement` is `m(X) - X` in the spherical frame.
Components are in the spherical frame `(r, theta, phi)`, ranks 2 and 4
Voigt-reduced.  `/spectral` is reserved.

Rheology is provenance, not values.  A frequency-dependent field has no
numbers until a frequency is chosen, so `/rheology/<name>` holds, per
layer, the `LawRecord` of the field that layer holds: the law's
registered name (`static` for a layer whose moduli do not depend on
frequency), the fields it read in the order it took them, its constants
with their dimensions, and its convention; a consumer calls the same
law on the same fields in its own code.  The root attribute
`model_class` names what the body guarantees and `reference_period`
the period the laws' moduli hold at, where they have one.  A *sample*
at a chosen omega may be stored beside that: `/fields/<name>` with a
trailing dimension `part` of length 2, real then imaginary, and the
attributes `omega` and `part = "complex"`.

Lengths, masses and times are in the body's own scales, recorded as
root attributes; nothing is converted on the way in or out.  A field
of no declared dimensions is written with `units = "unknown"`.

`read` restores the sample whole and the body from what the file can
state: the radial scalar fields as the piecewise Lagrange interpolants
through their nodal values (exact for polynomial layer functions of
degree below the GLL order, PREM included), the elastic tensors from
the components their variables name, and each frequency-dependent
field by calling the law its `/rheology` row names.  A field the body
cannot take back -- one with angular dependence, or of rank above zero
that is not an elastic tensor -- stays in the sample and is warned
about by name.  Surfaces are not restored.

netCDF4 is an optional extra and is imported only inside the functions
here, so `import planetmodel` stays light.
"""
from __future__ import annotations

import json
import warnings
from datetime import datetime, timezone

import numpy as np

from ..mesh1d.mesh import RadialMesh
from ..model.body import Interface, Layer, ReferenceBody
from ..model.character import SCALAR, Character
from ..model.fields.radial import RadialField
from ..model.mapping import IdentityMapping, RadialStretch
from ..model.materials import ElasticField, Symmetry
from ..model.rheology import LawRecord, STATIC, law_record_of, rebuild
from ..model.skeleton import Skeleton
from ..model.units import Dimensions, Scales, unit_string
from ..registry import lookup, name_of, registered
from ..sampling import AngularGrid, Sample, SampleMetadata
from .manifest import provenance_of

__all__ = ["SCHEMA", "write", "read"]

SCHEMA = "planetmodel.model/1"

#: Dimensions of a law's constants where its record does not say.
_CONSTANT_DIMENSIONS = {"reference_period": Dimensions.TIME}


def _put(variable, values) -> None:
    """Assign a whole variable.

    netCDF4 reshapes a multi-dimensional array in place on assignment,
    which NumPy 2.5 deprecates; that is the library's to fix, and the
    warning is silenced here so a writer's own warnings stay visible.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", "Setting the shape on a NumPy array",
                                DeprecationWarning)
        variable[...] = values


def _component_dims(character: Character) -> tuple[str, ...]:
    """Dimension names for the trailing component axes of a field."""
    if character.voigt_shape is not None:
        return {1: ("voigt",), 2: ("voigt_i", "voigt_j")}[len(character.voigt_shape)]
    return tuple(f"component_{s}" for s in "ijkl"[:character.rank])


def _string_variable(group, name: str, dim: str, values) -> None:
    """A vlen-string variable, one entry per element of the dimension."""
    v = group.createVariable(name, str, (dim,))
    for i, s in enumerate(values):
        v[i] = str(s)


def _mapping_rule(mapping) -> dict:
    """What can be said about how the displacement was built.

    A `RadialStretch` carries its displacement `h`; a rule built by a
    registered name (`layer_linear`) names itself, so that a consumer
    reconstructing the mapping on its side knows which one
    and can rebuild it from the knots and the surfaces in the file.
    Anything else is recorded by its repr and reproducibility falls back
    honestly to the writer's script.
    """
    h = getattr(mapping, "h", None)
    if h is None:
        return {"name": type(mapping).__name__, "registered": False,
                "repr": repr(mapping)}
    name = getattr(h, "name", None) or type(h).__name__
    try:
        lookup("displacement_rule", name)
        registered = True
    except KeyError:
        registered = False
    block = {"name": name, "registered": registered,
             "knots": [float(k) for k in getattr(h, "knots", ())]}
    taper = getattr(h, "taper_radius", getattr(h, "_taper", None))
    if taper is not None:
        block["inner_taper_radius"] = float(taper)
    if not registered:
        block["repr"] = repr(h)
    return block


def _model_class(body) -> str:
    """The registered name of what the body guarantees, or "".

    A plain `ReferenceBody` guarantees nothing and writes an empty
    string; a model class writes the name it is registered under, which
    is what lets a reader hand back the same class rather than a body
    that claims less than the file does.
    """
    return name_of("model_class", type(body)) or ""


def _elastic_of(sample: Sample, body, name: str):
    """The `ElasticField` a stored field came from, or None.

    A rank-4 Voigt 6x6 does not say which moduli it was built from, so
    the writer records the symmetry and the component names beside it
    and the reader rebuilds the tensor from the components it
    restored.  The sample's `source` answers where it has one; a sample
    read back from a file carries none, and the body it came with does,
    so a file written from a read-back pair says the same thing as the
    file it came from.
    """
    fld = (sample.source or {}).get(name)
    if fld is None and name in body:
        fld = body[name]
    return fld if isinstance(fld, ElasticField) else None


def _lifted_source_name(layer, field) -> str | None:
    """The name a layer holds a lifted field's static source under.

    A lift has no law; its provenance is the name of the field it
    lifts (`rheology.law_record_of`), and only the layer knows that name.
    """
    src = getattr(field, "source", None)
    if src is None:
        return None
    for name, other in layer.fields.items():
        if other is src:
            return name
    return getattr(src, "name", None)


def _constant_dimensions(record: LawRecord) -> dict[str, list[int]]:
    """The exponent triple of each constant, from the record where it
    says, else from the names the library knows."""
    declared = dict(getattr(record, "constant_dimensions", None) or {})
    out = {}
    for name in record.constants:
        d = declared.get(name, _CONSTANT_DIMENSIONS.get(name))
        if d is not None:
            out[name] = [int(d.mass), int(d.length), int(d.time)]
    return out


def _record_row(record: LawRecord | None) -> tuple[str, str, str, str, str]:
    """One layer's row of `/rheology/<name>`: law, parameters, constants,
    constant_dimensions, convention.  An empty `law` is a layer that holds
    nothing to rebuild."""
    if record is None:
        return "", "[]", "{}", "{}", ""
    return (record.law, json.dumps(list(record.parameters)),
            json.dumps(dict(record.constants)),
            json.dumps(_constant_dimensions(record)), record.convention or "")


def _rheology_rows(body, layers, fname: str, *, guaranteed: bool) -> list:
    """The `/rheology/<fname>` rows, and the check that each can be written.

    A layer holding the field contributes its record; a frequency-
    dependent field with no record cannot be rebuilt by anyone and is
    refused by name rather than written as a blank.  Where `fname` is
    the field a model class guarantees, a layer holding `elastic_moduli`
    and no such field is an elastic layer, and its row says `static`.
    """
    rows = []
    for lay in layers:
        f = lay.fields.get(fname)
        if f is None:
            if guaranteed and "elastic_moduli" in lay.fields:
                rows.append(_record_row(LawRecord(STATIC,
                                                  parameters=("elastic_moduli",))))
            else:
                rows.append(_record_row(None))
            continue
        record = law_record_of(f, source_name=_lifted_source_name(lay, f))
        if record is None:
            raise ValueError(
                f"field {fname!r} on layer {lay.index} depends on frequency "
                "but carries no LawRecord, so no file can say how to rebuild "
                "it; build it with a registered law, or drop it before "
                "writing")
        rows.append(_record_row(record))
    return rows


def _reference_period(rows_by_field: dict) -> float | None:
    """The one reference period the laws hold at, or None.

    Every record that names a `reference_period` must name the same
    one: two would make the root attribute a lie about one of them.
    """
    periods = set()
    for rows in rows_by_field.values():
        for row in rows:
            constants = json.loads(row[2])
            if "reference_period" in constants:
                periods.add(float(constants["reference_period"]))
    if len(periods) > 1:
        raise ValueError(
            f"the laws hold at different reference periods {sorted(periods)}; "
            "one file records one reference_period")
    return periods.pop() if periods else None


def _mapping_kind(mapping) -> str:
    if getattr(mapping, "is_identity", False) or isinstance(mapping,
                                                             IdentityMapping):
        return "identity"
    if isinstance(mapping, RadialStretch):
        return "radial_stretch"
    return "general"


def write(body, sample: Sample, path) -> None:
    """Write a body and a sample of it as a `planetmodel.model/1` file.

    `sample` is what `body.sample(...)` returned for this body; the
    skeletons must agree.  The file carries the skeleton and its
    annotations, the radial and angular node sets, every sampled field,
    the mapping's displacement when the sample has one (with `dh/dr`
    beside it where the displacement provides it exactly), and the
    relief of every surface attached to the body, sampled on the same
    angular grid.  Nothing is converted: the numbers are in the body's
    scales, which the root attributes record.

    The root says what the body *is* -- `model_class`, the registered
    name of its class, empty for a plain `ReferenceBody` -- and
    `/rheology/<name>` carries, per layer, the `LawRecord` of every
    frequency-dependent field the body holds; `reference_period` is the
    period those records agree on.  A frequency-dependent field with no
    record is refused by name.  A field sampled at a chosen `omega` is
    stored with a trailing dimension `part` of length 2 and the
    attributes `omega` and `part = "complex"`.
    """
    from netCDF4 import Dataset

    if not isinstance(sample, Sample):
        raise TypeError(f"sample must be a Sample, got {type(sample).__name__}")
    if sample.metadata.skeleton != body.skeleton:
        raise ValueError("the sample was taken on a different skeleton from "
                         "the body being written")
    from .. import __version__

    mesh, grid = sample.radial, sample.angular
    sk = body.skeleton
    scales = body.scales
    si = scales.is_si
    theta, phi = grid.colatitudes, grid.longitudes
    nnode = sample.nnode
    layers = body.layers

    # The rheology rows are formed first: a field that cannot be
    # recorded refuses the whole write, and the root's reference period
    # is what the records agree on.
    guaranteed = getattr(body, "VISCOELASTIC", None)
    dynamic = [n for n in body.field_names
               if any(getattr(lay.fields[n], "kind", "static") == "frequency"
                      for lay in layers if n in lay.fields)]
    if (guaranteed is not None and guaranteed not in dynamic
            and "elastic_moduli" in body.field_names):
        dynamic.append(guaranteed)
    rows_by_field = {n: _rheology_rows(body, layers, n, guaranteed=(n == guaranteed))
                     for n in dynamic}
    reference_period = _reference_period(rows_by_field)

    with Dataset(str(path), "w", format="NETCDF4") as ds:
        # -- root ------------------------------------------------------------
        name = body.meta.get("name", "unnamed")
        ds.setncatts({
            "schema": SCHEMA,
            "title": name,
            "source": body.meta.get("source") or name,
            "history": (datetime.now(timezone.utc).isoformat(timespec="seconds")
                        + " planetmodel.io.netcdf.write"),
            "planetmodel_version": __version__,
            "Conventions": "CF-1.10 where applicable",
            "scales_length_m": float(scales.length),
            "scales_mass_kg": float(scales.mass),
            "scales_time_s": float(scales.time),
            "gravitational_constant": float(scales.gravitational_constant),
            "frame": "spherical (r, theta, phi); theta is colatitude",
            "layout": "(node, colatitude, longitude, components...)",
            "model_class": _model_class(body),
            "index_base": 0,
        })
        if reference_period is not None:
            # In the file's own time unit, like every other number here:
            # a law's constants are already in the body's scales.
            ds.setncattr("reference_period", float(reference_period))
        ds.createDimension("node", nnode)
        ds.createDimension("element", mesh.nspec)
        ds.createDimension("element_edge", mesh.nspec + 1)
        ds.createDimension("boundary", sk.boundaries.size)
        ds.createDimension("layer", sk.nlayers)
        ds.createDimension("interface", len(body.interfaces))
        ds.createDimension("colatitude", theta.size)
        ds.createDimension("longitude", phi.size)
        ds.createDimension("component", 3)

        # -- skeleton --------------------------------------------------------
        g = ds.createGroup("skeleton")
        v = g.createVariable("boundaries", "f8", ("boundary",))
        v[:] = sk.boundaries
        v.units = unit_string(Dimensions.LENGTH, si=si)
        v.long_name = "radii of the layer boundaries, centre outward"
        _string_variable(g, "layer_name", "layer",
                         [lay.name or f"layer_{i}"
                          for i, lay in enumerate(layers)])
        _string_variable(g, "layer_state", "layer", [lay.state for lay in layers])
        # What a layer has is what its fields say: the names are the
        # record, and a layer with an empty list holds nothing.
        _string_variable(g, "layer_fields", "layer",
                         [json.dumps(list(lay.field_names)) for lay in layers])
        v = g.createVariable("layer_is_vacuum", "i1", ("layer",))
        v[:] = np.array([bool(lay.is_vacuum) for lay in layers], dtype="i1")
        v.long_name = "1 where the layer is a void holding no fields"
        faces = body.interfaces
        _string_variable(g, "interface_name", "interface",
                         [f.name or f"interface_{i}"
                          for i, f in enumerate(faces)])
        _string_variable(g, "interface_role", "interface",
                         [f.role for f in faces])
        v = g.createVariable("interface_radius", "f8", ("interface",))
        v[:] = np.array([f.radius for f in faces], dtype=float)
        v.units = unit_string(Dimensions.LENGTH, si=si)
        v.long_name = "mean radius of each interface (the boundary it sits on)"

        # -- radial ----------------------------------------------------------
        g = ds.createGroup("radial")
        g.setncatts({"gll_order": int(mesh.ngll - 1), "n_gll": int(mesh.ngll),
                     "note": ("per-element GLL nodes, flattened; an element "
                              "boundary is a repeated radius, one node on "
                              "each side")})
        v = g.createVariable("radius", "f8", ("node",))
        v[:] = sample.radius
        v.units = unit_string(Dimensions.LENGTH, si=si)
        v = g.createVariable("element_start", "i8", ("element_edge",))
        v[:] = sample.element_start
        v.long_name = "first node of each element, and the total"
        v = g.createVariable("element_layer", "i8", ("element",))
        v[:] = sample.element_layer
        v.long_name = "skeleton layer index of each element (0-based)"
        v = g.createVariable("weights", "f8", ("node",))
        v[:] = (mesh.w[None, :] * mesh.jac[:, None]).ravel()
        v.units = unit_string(Dimensions.LENGTH, si=si)
        v.long_name = "GLL quadrature weights in r, per node"

        # -- angular ---------------------------------------------------------
        g = ds.createGroup("angular")
        g.setncatts({"kind": grid.kind,
                     "weights_note": ("integral over the sphere of f = sum_i "
                                      "weights_i sum_j (2 pi / nphi) f_ij")})
        if grid.lmax is not None:
            g.setncattr("lmax", int(grid.lmax))
        v = g.createVariable("colatitude", "f8", ("colatitude",))
        v[:] = theta
        v.units = "radian"
        v = g.createVariable("longitude", "f8", ("longitude",))
        v[:] = phi
        v.units = "radian"
        if grid.weights is not None:
            v = g.createVariable("weights", "f8", ("colatitude",))
            v[:] = grid.weights
            v.long_name = "colatitude quadrature weights against sin(theta) dtheta"

        # -- fields ----------------------------------------------------------
        g = ds.createGroup("fields")
        meta = sample.metadata
        for fname, arr in sample.fields.items():
            ch = meta.characters[fname]
            cdims = _component_dims(ch)
            # A field sampled at a chosen omega is complex, and is stored
            # as two real numbers per component on a trailing dimension
            # `part`: real first, then imaginary.
            omega = meta.omegas.get(fname)
            if omega is not None:
                cdims = cdims + ("part",)
            for d, n in zip(cdims, arr.shape[-len(cdims):] if cdims else ()):
                if d not in ds.dimensions:
                    ds.createDimension(d, int(n))
            base = ("node",) if sample.is_radial(fname) else (
                "node", "colatitude", "longitude")
            # A field belongs to the layers of its domain; the nodes of
            # every other element take the fill value, which is the NaN
            # the sample already carries.  A consumer that ignores
            # `layers` sees fill and nothing else changes.
            v = g.createVariable(fname, "f8", base + cdims,
                                 fill_value=np.nan)
            _put(v, arr)
            dims = meta.dimensions[fname]
            v.setncatts({
                "layers": json.dumps([int(i) for i in meta.domains[fname]]),
                "character_rank": int(ch.rank),
                "character_weight": int(ch.weight),
                "voigt": int(ch.voigt_shape is not None),
                "physical_dimensions": (np.array([dims.mass, dims.length, dims.time],
                                        dtype="i8") if dims is not None
                               else np.array([0, 0, 0], dtype="i8")),
                "physical_dimensions_declared": int(dims is not None),
                "units": unit_string(dims, si=si),
                "frame": meta.frames[fname],
                "long_name": fname,
                "representation": "referential",
                "radial": int(sample.is_radial(fname)),
                "sampled_from": (type(sample.source[fname]).__name__
                                 if sample.source and fname in sample.source
                                 else "unknown"),
            })
            if omega is not None:
                v.setncatts({
                    "omega": float(omega),
                    "part": "complex",
                    "part_note": ("index 0 is the real part, index 1 the "
                                  "imaginary part"),
                })
            elastic = _elastic_of(sample, body, fname)
            if elastic is not None:
                # The moduli names are not recoverable from a 6x6, so the
                # tensor says which symmetry class it is and which fields
                # of this file its components are.
                v.setncatts({
                    "symmetry": elastic.symmetry.name,
                    "components": json.dumps(list(elastic.moduli_names)),
                })

        # -- rheology --------------------------------------------------------
        # Frequency-dependent fields are never stored as values; what the
        # file carries is the provenance of each, layer by layer, so that
        # a consumer rebuilds the field from the basic fields itself.
        if rows_by_field:
            g = ds.createGroup("rheology")
            for fname, rows in rows_by_field.items():
                view = body[fname]
                sub = g.createGroup(fname)
                sub.setncatts({
                    "kind": "frequency",
                    "omega_domain": str(getattr(view, "omega_domain",
                                                "complex")),
                    "character_rank": int(view.character.rank),
                    "character_weight": int(view.character.weight),
                })
                for j, (nm, note) in enumerate((
                        ("law", "registered rheology law; 'static' where the "
                                "layer's moduli do not depend on frequency; "
                                "empty where the layer holds nothing to rebuild"),
                        ("parameters", "JSON list: the fields the law read, "
                                       "in the order it took them"),
                        ("constants", "JSON object: the law's numbers, in "
                                      "this file's units"),
                        ("constant_dimensions", "JSON object: each constant's "
                                                "(mass, length, time) exponents"),
                        ("convention", "the named variant of the law, empty "
                                       "where it has none"))):
                    _string_variable(sub, nm, "layer", [row[j] for row in rows])
                    sub.variables[nm].long_name = note

        # -- mapping ---------------------------------------------------------
        u = sample.displacement
        if u is not None:
            g = ds.createGroup("mapping")
            mapping = sample.mapping
            g.setncatts({"kind": _mapping_kind(mapping),
                         "rule": json.dumps(_mapping_rule(mapping))})
            v = g.createVariable("displacement", "f8",
                                 ("node", "colatitude", "longitude", "component"))
            _put(v, u)
            v.setncatts({"units": unit_string(Dimensions.LENGTH, si=si),
                         "frame": "spherical",
                         "long_name": "m(X) - X, components (r, theta, phi)"})
            h = getattr(mapping, "h", None)
            knots = tuple(getattr(h, "knots", ()))
            if knots:
                ds.createDimension("knot", len(knots))
                v = g.createVariable("knots", "f8", ("knot",))
                v[:] = np.array(knots, dtype=float)
                v.units = unit_string(Dimensions.LENGTH, si=si)
                v.long_name = "radii where dh/dr may jump"
            if h is not None and hasattr(h, "radial_derivative"):
                r = sample.radius[:, None, None]
                dh = np.broadcast_to(np.asarray(
                    h.radial_derivative(r, theta[None, :, None],
                                        phi[None, None, :]), dtype=float),
                    (nnode, theta.size, phi.size))
                v = g.createVariable("dh_dr", "f8",
                                     ("node", "colatitude", "longitude"))
                _put(v, dh)
                v.setncatts({"units": "1",
                             "long_name": "d h / d r, exact from the displacement"})

        # -- surfaces --------------------------------------------------------
        surfaces = body.surfaces
        if surfaces:
            g = ds.createGroup("surfaces")
            for i in sorted(surfaces):
                face = body.interfaces[i]
                surf = surfaces[i]
                sg = g.createGroup(face.name or f"interface_{i}")
                v = sg.createVariable("relief", "f8", ("colatitude", "longitude"))
                _put(v, np.broadcast_to(surf.height(theta[:, None], phi[None, :]),
                                        (theta.size, phi.size)))
                v.units = unit_string(Dimensions.LENGTH, si=si)
                v.long_name = "departure from the interface's mean radius"
                provenance = provenance_of(surf.topography)
                sg.setncatts({
                    "interface": int(i),
                    "interface_name": face.name or f"interface_{i}",
                    "reference_radius": float(face.radius),
                    "topography": (getattr(surf.topography, "name", None)
                                   or type(surf.topography).__name__),
                    "sources": json.dumps(provenance),
                })

        # -- spectral (reserved) --------------------------------------------
        g = ds.createGroup("spectral")
        g.setncatts({
            "status": "reserved",
            "convention": ("per radial node, spherical-harmonic coefficients "
                           "orthonormal with the Condon-Shortley phase, upper "
                           "index N, l-major m-fastest block layout (GSHTrans)"),
        })


# ---------------------------------------------------------------------------
# reading
# ---------------------------------------------------------------------------

def _scales_of(ds) -> Scales:
    """The body's own units, from the root attributes.

    `gravitational_constant` is written for a consumer to read and is
    derived here from the scales rather than trusted, so that a file
    whose three scales and whose G disagree is read as its scales say.
    """
    return Scales(length=float(ds.scales_length_m),
                  mass=float(ds.scales_mass_kg),
                  time=float(ds.scales_time_s))


def _att(v, name: str):
    """One attribute of a variable, asked for by name and not by dot.

    `v.dimensions` on a netCDF4 Variable is the library's own property,
    the tuple of dimension names, and a per-field attribute of that
    name would be shadowed by it.
    Every field attribute is read through `getncattr` for that reason,
    so no reader here can be caught by the next such collision.
    """
    return v.getncattr(name)


def _character_of(v) -> Character:
    """The Character of a stored field, from its three attributes.

    `voigt` is the file's answer to "is this stored Voigt-reduced",
    which is a statement about ranks 2 and 4 alone; for every other rank
    the flag is written as 0 whatever the character said, so it is not
    read back, and the Character's own default (True) stands.  Nothing
    downstream distinguishes `Character(0, 1)` from `Character(0, 1,
    voigt=False)`, and SCALAR, DENSITY and VECTOR are all the default.
    """
    rank = int(_att(v, "character_rank"))
    weight = int(_att(v, "character_weight"))
    voigt = bool(int(_att(v, "voigt"))) if rank in (2, 4) else True
    return Character(rank, weight, voigt=voigt)


def _dimensions_of(v) -> Dimensions | None:
    """Physical dimensions, or None where the writer declared none."""
    if not int(_att(v, "physical_dimensions_declared")):
        return None
    mass, length, time = (int(x) for x in _att(v, "physical_dimensions"))
    return Dimensions(mass=mass, length=length, time=time)


def _names(values, prefix: str) -> list[str | None]:
    """Strings from a vlen variable, the default spellings back to None.

    The writer spells an unnamed layer `layer_3` and an unnamed
    interface `interface_5`, 0-based like every other index in the
    file.  A name that is exactly its own index's default
    is not a name, so it reads back as None and a body written, read and
    written again spells it the same way.
    """
    out: list[str | None] = []
    for i, s in enumerate(values):
        s = str(s)
        out.append(None if s == f"{prefix}{i}" else s)
    return out


def _mesh_edges(g, ngll: int) -> np.ndarray:
    """Element boundaries from the flat node radii and `element_start`.

    A node array is per element and an element boundary is a repeated
    radius, so the edges are the first node of every element with the
    last node of the last one appended: exactly the numbers the mesh was
    built from, not a rule guessed backwards from them.
    """
    radius = np.asarray(g.variables["radius"][:], dtype=float)
    start = np.asarray(g.variables["element_start"][:], dtype=int)
    if start.size < 2 or radius.size != (start.size - 1) * ngll:
        raise ValueError(
            f"/radial holds {radius.size} nodes but element_start says "
            f"{start.size - 1} elements of {ngll}: the file is inconsistent")
    return np.concatenate((radius[start[:-1]], radius[-1:]))


def _interpolants(nodal: np.ndarray, mesh: RadialMesh, domain, nlayers: int):
    """One layer function per layer: the SEM interpolant of the nodes.

    Within an element the restored function is the polynomial through
    the element's GLL nodes -- `Mesh1D.to_ppoly`, a scipy `PPoly` in `r`
    -- and a layer's function is those polynomials laid end to end, so
    `derivative` and `integrate` are exact and `rescaled` works.  For a
    model whose layer functions are polynomials of degree below the GLL
    order, PREM included, the interpolant *is* the original function.

    A layer of the domain that the file's mesh does not reach (a
    truncated mesh: `rmin` above the centre, `rmax` below the surface)
    gets None, since there are no nodes there to interpolate.
    """
    layer = np.asarray(mesh.layer, dtype=int)
    funcs: list = [None] * nlayers
    for i in domain:
        elements = np.flatnonzero(layer == i)
        if elements.size == 0:
            continue
        funcs[i] = mesh.to_ppoly(
            nodal, elements=(int(elements[0]), int(elements[-1]) + 1))
    return funcs


def _rheology_of(ds, nlayers: int) -> dict[str, list]:
    """The `/rheology` group as one `LawRecord` (or None) per layer per field.

    An empty `law` is a layer with nothing to rebuild; `static` is an
    elastic layer, whose row is read back as None too, since the class
    lifts such a layer's moduli at view time and nothing is stored.
    """
    out: dict[str, list] = {}
    if "rheology" not in ds.groups:
        return out
    for fname, sub in ds.groups["rheology"].groups.items():
        laws = [str(s) for s in sub.variables["law"][:]]
        params = [str(s) for s in sub.variables["parameters"][:]]
        consts = [str(s) for s in sub.variables["constants"][:]]
        convs = [str(s) for s in sub.variables["convention"][:]]
        if "constant_dimensions" in sub.variables:
            cdims = [str(s) for s in sub.variables["constant_dimensions"][:]]
        else:
            cdims = ["{}"] * len(laws)
        if len(laws) != nlayers:
            raise ValueError(
                f"/rheology/{fname} has {len(laws)} entries for a skeleton of "
                f"{nlayers} layers: the file is inconsistent")
        out[fname] = [
            None if (not law or law == STATIC) else _record(
                law, json.loads(p), json.loads(c), json.loads(d), conv or None)
            for law, p, c, d, conv in zip(laws, params, consts, cdims, convs)]
    return out


def _record(law, parameters, constants, constant_dimensions, convention):
    """A `LawRecord` from one row, carrying the constants' dimensions
    where the record type can hold them."""
    record = LawRecord(law, parameters=tuple(parameters), constants=constants,
                       convention=convention)
    dims = {k: Dimensions(mass=int(m), length=int(l), time=int(t))
            for k, (m, l, t) in constant_dimensions.items()}
    if dims and hasattr(record, "constant_dimensions"):
        try:
            object.__setattr__(record, "constant_dimensions", dims)
        except (AttributeError, TypeError):
            pass
    return record


def _components_from_voigt(symmetry: Symmetry, V: np.ndarray) -> dict:
    """The independent moduli read off a sampled Voigt matrix (..., 6, 6).

    The inverse of `materials.voigt_matrix` in this library's frame, where
    the symmetry axis is the first Voigt index: isotropic mu is the
    (3, 3) entry and kappa follows from (0, 0) = kappa + 4 mu / 3; the
    transversely isotropic C, A, F, N, L sit at (0, 0), (1, 1), (0, 1),
    (3, 3), (4, 4).
    """
    if symmetry is Symmetry.ISOTROPIC:
        mu = V[..., 3, 3]
        return {"kappa": V[..., 0, 0] - 4.0 * mu / 3.0, "mu": mu}
    if symmetry is Symmetry.VTI:
        return {"A": V[..., 1, 1], "C": V[..., 0, 0], "F": V[..., 0, 1],
                "L": V[..., 4, 4], "N": V[..., 3, 3]}
    raise ValueError(f"cannot read the components of a {symmetry.name} tensor")


def _rebuilt_body(body, moduli: dict, rheology: dict, *, derived=None):
    """The body with its tensors and its rheology back on the layers.

    Two things the file states rather than stores.  An `ElasticField` is
    rebuilt from the components its `/fields` variable names, on every
    layer where all of them were restored -- a 6x6 does not say which
    moduli it was built from, but the attributes `symmetry` and
    `components` do.  A frequency-dependent field is rebuilt by calling
    the law its `/rheology` row names on that layer's own fields
    (`rheology.rebuild`), which is why the moduli go on first: a law's
    parameters are fields, and `constant_q` reads the tensor.

    A law naming a field the file does not carry is refused by name: the
    alternative is a body quietly claiming less than its own root
    attribute says, which is what `model_class` was given a reader for.
    """
    if not moduli and not rheology:
        return body
    derived = derived or {}
    layers = list(body.layers)
    for i, lay in enumerate(layers):
        if lay.is_vacuum:
            continue
        for fname, (symmetry, components) in moduli.items():
            # A component the body holds by name is the source; one it does
            # not is read off the tensor's own sample, so a body that held
            # the tensor alone comes back holding the tensor alone.
            spare = derived.get(fname, {})
            parts = {}
            for c in components:
                if c in lay.fields:
                    parts[c] = lay[c]
                elif c in spare and i in spare[c].domain:
                    parts[c] = spare[c].restricted(i)
            if len(parts) != len(components):
                continue
            lay = lay.with_field(
                fname, ElasticField(symmetry, parts, name=fname), replace=True)
        for fname, specs in rheology.items():
            spec = specs[i]
            if spec is None:
                continue
            try:
                f = rebuild(spec, lay.fields)
            except ValueError as exc:
                raise ValueError(
                    f"layer {i}: the file's /rheology says {fname!r} was built "
                    f"by {spec.law!r} from {list(spec.parameters)}, and that "
                    f"cannot be done from what the file carries -- {exc}. Only "
                    "the radial scalar fields and the elastic tensors of "
                    "/fields are restored on a body, so a model whose laws "
                    "are to be rebuilt must have its parameter fields "
                    "sampled") from None
            lay = lay.with_field(fname, f, replace=True)
        layers[i] = lay
    return ReferenceBody(layers, meta=dict(body.meta), interfaces=body.interfaces,
                         scales=body.scales)


def _typed(body, model_class: str):
    """The body as the class the file names, or a ValueError naming it."""
    try:
        cls = lookup("model_class", model_class)
    except KeyError:
        raise ValueError(
            f"this file declares model_class={model_class!r}, which is not "
            f"registered here; registered: {list(registered('model_class'))}"
        ) from None
    return body.as_class(cls)


def read(path):
    """Read a `planetmodel.model/1` file: `(body, sample)`.

    The **sample** is restored in full: the `RadialMesh` on the file's
    own element edges, the `AngularGrid`, every field with the fill
    values turned back into NaN, the displacement, and the metadata.
    Its `source` and `mapping` are None, since a file holds numbers, not
    the fields that made them.

    The **body** carries the skeleton, the layer names and states, the
    interfaces, the scales and the title, and the fields the file can
    state: each radial scalar field as the piecewise Lagrange
    interpolant through its nodal values, an `ElasticField` on every
    layer whose named components were restored, and each
    frequency-dependent field of `/rheology` rebuilt by calling the law
    its record names on that layer's own fields.  A `static` row is an
    elastic layer and gets nothing: the model class supplies the lift at
    view time.  Every other stored field -- anything with angular
    dependence, or of rank above zero that is not an elastic tensor --
    stays in the sample alone and is warned about by name.  Surfaces are
    not restored.

    A file whose `model_class` is not empty comes back as that class; a
    name this build does not know, or a law whose parameter fields the
    file does not carry, is a `ValueError` naming it.
    """
    from netCDF4 import Dataset

    with Dataset(str(path)) as ds:
        ds.set_auto_mask(False)
        schema = getattr(ds, "schema", None)
        if schema != SCHEMA:
            raise ValueError(f"not a {SCHEMA} file: schema is {schema!r}")
        model_class = str(getattr(ds, "model_class", ""))
        scales = _scales_of(ds)

        # -- the geometry and its annotations --------------------------------
        gs = ds.groups["skeleton"]
        skeleton = Skeleton(np.asarray(gs.variables["boundaries"][:],
                                       dtype=float))
        nlayers = skeleton.nlayers
        layer_names = _names(gs.variables["layer_name"][:], "layer_")
        layer_states = [str(s) for s in gs.variables["layer_state"][:]]
        layers = tuple(Layer(index=i, name=layer_names[i], state=layer_states[i])
                       for i in range(nlayers))
        face_names = _names(gs.variables["interface_name"][:], "interface_")
        face_roles = [str(s) for s in gs.variables["interface_role"][:]]
        radii = np.asarray(gs.variables["interface_radius"][:], dtype=float)
        faces = tuple(
            Interface(index=i, name=face_names[i], radius=float(radii[i]),
                      between=(i, i + 1 if i + 1 < nlayers else -1),
                      role=face_roles[i])
            for i in range(len(face_roles)))

        meta: dict = {}
        for attr, key in (("title", "name"), ("source", "source"),
                          ("history", "history")):
            value = getattr(ds, attr, None)
            if value is not None:
                meta[key] = str(value)
        body = ReferenceBody.from_fields(skeleton, {}, layers=layers,
                                         interfaces=faces, meta=meta,
                                         scales=scales)

        # -- the radial and angular node sets --------------------------------
        gr = ds.groups["radial"]
        ngll = int(gr.n_gll)
        mesh = RadialMesh(body, ngll=ngll, edges=_mesh_edges(gr, ngll))
        ga = ds.groups["angular"]
        weights = (np.asarray(ga.variables["weights"][:], dtype=float)
                   if "weights" in ga.variables else None)
        grid = AngularGrid(
            np.asarray(ga.variables["colatitude"][:], dtype=float),
            np.asarray(ga.variables["longitude"][:], dtype=float),
            kind=str(ga.kind),
            lmax=int(ga.lmax) if "lmax" in ga.ncattrs() else None,
            weights=weights)

        # -- the fields ------------------------------------------------------
        arrays, chars, dims, frames, doms, omegas = {}, {}, {}, {}, {}, {}
        radial_fields: dict[str, RadialField] = {}
        moduli: dict[str, tuple] = {}
        for name, v in ds.groups["fields"].variables.items():
            arrays[name] = np.ascontiguousarray(np.asarray(v[...], dtype=float))
            chars[name] = _character_of(v)
            dims[name] = _dimensions_of(v)
            frames[name] = str(_att(v, "frame"))
            doms[name] = tuple(int(i) for i in json.loads(_att(v, "layers")))
            atts = v.ncattrs()
            if "part" in atts and str(_att(v, "part")) == "complex":
                # A sample of a frequency-dependent field at one omega:
                # the array keeps its trailing (real, imaginary) axis and
                # the field itself is rebuilt from /rheology, not here.
                omegas[name] = float(_att(v, "omega"))
            if "symmetry" in atts:
                symmetry = str(_att(v, "symmetry"))
                try:
                    moduli[name] = (Symmetry[symmetry],
                                    tuple(json.loads(_att(v, "components"))))
                except KeyError:
                    raise ValueError(
                        f"field {name!r} declares symmetry {symmetry!r}, which "
                        f"is not a Symmetry here; known: "
                        f"{[s.name for s in Symmetry]}") from None
            if name in omegas:
                continue
            if not (int(_att(v, "radial")) and chars[name].rank == 0):
                continue
            # A vacuum layer holds no fields, so a domain reaching into
            # one is trimmed here rather than refused: the sample keeps
            # the file's own domain, which is what its NaN pattern says.
            domain = [i for i in doms[name] if not body.layers[i].is_vacuum]
            funcs = _interpolants(arrays[name].reshape(mesh.nspec, ngll),
                                  mesh, domain, nlayers)
            if any(f is not None for f in funcs):
                radial_fields[name] = RadialField(
                    skeleton, funcs, name=name, character=chars[name],
                    dimensions=dims[name])

        # A radial tensor whose named components the file does not carry
        # as fields of their own is rebuilt from its sample.
        derived: dict[str, dict[str, RadialField]] = {}
        for fname, (symmetry, components) in moduli.items():
            v = ds.groups["fields"].variables[fname]
            if (fname in omegas or not int(_att(v, "radial"))
                    or chars[fname].rank != 4
                    or all(c in radial_fields for c in components)):
                continue
            nodal = _components_from_voigt(symmetry, arrays[fname])
            domain = [i for i in doms[fname] if not body.layers[i].is_vacuum]
            for c, values in nodal.items():
                if c in radial_fields:
                    continue
                funcs = _interpolants(values.reshape(mesh.nspec, ngll), mesh,
                                      domain, nlayers)
                if any(f is not None for f in funcs):
                    derived.setdefault(fname, {})[c] = RadialField(
                        skeleton, funcs, name=c, character=SCALAR,
                        dimensions=dims[fname])

        # -- the rheology, and the mapping's displacement ---------------------
        rheology = _rheology_of(ds, nlayers)
        displacement = None
        if "mapping" in ds.groups:
            gm = ds.groups["mapping"]
            if "displacement" in gm.variables:
                displacement = np.ascontiguousarray(
                    np.asarray(gm.variables["displacement"][...], dtype=float))

    for name, fld in radial_fields.items():
        body.add_field(name, fld)
    dropped = [n for n in arrays
               if n not in radial_fields and n not in moduli and n not in omegas]
    if dropped:
        warnings.warn(
            f"{path}: the fields {dropped} depend on direction or are tensors "
            "the file does not state how to rebuild, so they are in the sample "
            "and not on the body", UserWarning, stacklevel=2)
    body = _rebuilt_body(body, moduli, rheology, derived=derived)
    if model_class:
        body = _typed(body, model_class)
    metadata = SampleMetadata(characters=chars, dimensions=dims, frames=frames,
                              domains=doms, scales=scales, skeleton=skeleton,
                              omegas=omegas)
    sample = Sample(radial=mesh, angular=grid, fields=arrays,
                    displacement=displacement, metadata=metadata)
    return body, sample
