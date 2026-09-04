"""A recipe file builds a mesh, and the manifest says which recipe.

The recipes here are the plan's §4.2 shape at a thousandth of the size:
a five-knot deck on a 960 km planet, two ten-degree relief grids written
by the test itself, and a sizing that meshes in under a second.  The
questions are all about the *format* -- do the units convert, does a
boundary have to agree with the surface hung on it, does an unknown key
fail loudly, does the manifest echo the file -- and none of them is
answered better by a real Earth, which is why no reference data appears
here at all.

The one thing the trimmed format asks of an author is on show in
`recipe_dir`: a boundary carrying a surface is placed by a *number*, and
the number is the surface's own mean radius, so the script that writes
the recipe computes it from the files.  That is the whole of what the
`"mean_<surface>"` sugar used to do, done where the data is.

These sit behind the gmsh guard because parsing a recipe builds a
MeshSpec, and planetmodel.mesh3d is where MeshSpec lives.
"""
from pathlib import Path

import numpy as np
import pytest

from planetmodel.io import recipe as rp
from planetmodel.io import manifest as sc
from planetmodel.mesh3d import __main__ as cli
from planetmodel.model.surface import Surface
from planetmodel.model.topography import GriddedTopography

pytestmark = pytest.mark.gmsh

#: One mesh length unit, in metres.  A small planet, meshed coarsely.
RREF = 1.0e6


def write_deck(path):
    """A five-knot isotropic deck: three shells out to 960 km."""
    rows = []
    for lo, hi in ((0.0, 0.2e6), (0.2e6, 0.6e6), (0.6e6, 0.96e6)):
        for r in np.linspace(lo, hi, 5):
            rows.append(f"{r:12.1f} {5000.0:9.1f} {8000.0:9.1f} "
                        f"{4500.0:9.1f} {1000.0:9.1f} {600.0:9.1f}")
    path.write_text("small planet\n 1 1.0 1\n"
                    f" {len(rows)} 5 10\n" + "\n".join(rows) + "\n")
    return path


def mean_radius(*paths):
    """The mean radius of the boundary those files describe, in metres.

    The reader's own arithmetic, run by the recipe's author: read each
    grid as a departure from the reference radius in kilometres, sum
    them, and ask where the result sits on average.  A recipe writes
    this number down; nothing in the format computes it.
    """
    shape = None
    for path in paths:
        piece = GriddedTopography.from_xyz(path, scale=1.0e3)
        shape = piece if shape is None else shape + piece
    return Surface(RREF, topography=shape).centred().reference_radius


RECIPE = """
[model]
source = "planet.deck"
reader = "isotropic_deck"
rref_m = 1.0e6

[geometry]
drop_outermost_interfaces = 1
truncate_at = %(moho_m).10f
truncate_name = "moho"
insert = ["floor", "surface"]
buffer = { ratio = 0.2, name = "buffer" }

  [geometry.floor]
  radius_km = 650
  role = "control"

  [geometry.surface]
  radius_m = %(surface_m).10f
  fields = "none"

[mesh]
dimension = 3
order = 2

[sizing]
policy = "uniform_interfaces"
h_min_km = 150
h_max_km = 300
decay_width_km = 300

[[surfaces]]
name = "moho"
files = ["moho.xyz"]
units = "km"

[[surfaces]]
name = "surface"
files = ["thickness.xyz", "moho.xyz"]
units = "km"

[mapping]
mode = "physical"
rule = "layer_linear"
exaggeration = 2

[output]
path = "mesh/small"
"""


@pytest.fixture
def recipe_dir(tmp_path, write_relief_xyz):
    """A directory holding a complete recipe and the data it names.

    The Moho grid is a depth about the reference radius, 50 km down with
    5 km of degree-two relief; the crustal thickness partly cancels it,
    as isostasy would, so the surface the two sum to carries 2 km.  Both
    mean radii are computed here and written into the recipe, which is
    the workflow the format now expects.
    """
    write_deck(tmp_path / "planet.deck")
    moho = write_relief_xyz(tmp_path / "moho.xyz", offset_km=-50.0,
                            amplitude_km=5.0)
    thickness = write_relief_xyz(tmp_path / "thickness.xyz", offset_km=50.0,
                                 amplitude_km=-3.0)
    (tmp_path / "recipe.toml").write_text(
        RECIPE % {"moho_m": mean_radius(moho),
                  "surface_m": mean_radius(thickness, moho)})
    return tmp_path


def truncation_line(recipe_dir) -> str:
    """The recipe's `truncate_at = ...` line, whatever number it holds."""
    text = (recipe_dir / "recipe.toml").read_text()
    return next(ln for ln in text.splitlines() if ln.startswith("truncate_at ="))


def edited(recipe_dir, old, new):
    """The same recipe with one line changed, re-read."""
    text = (recipe_dir / "recipe.toml").read_text()
    assert old in text, f"{old!r} is not in the recipe"
    path = recipe_dir / "edited.toml"
    path.write_text(text.replace(old, new))
    return path


# --------------------------------------------------------------- parsing

def test_the_plan_s_recipe_shape_resolves(recipe_dir):
    """Every §4.2 construction lands in the spec it describes."""
    card = rp.read(recipe_dir / "recipe.toml")
    spec = card.spec

    assert spec.rref == pytest.approx(RREF)
    assert spec.outer_radius == pytest.approx(0.95e6, rel=1e-3)
    assert spec.outer_name == "moho"          # so a surface can attach
    assert spec.insert_radii == [pytest.approx(0.65e6)]
    assert spec.insert_names == ["floor"]
    assert spec.insert_role == "control"
    assert spec.extend_radii == [pytest.approx(1.0e6, rel=1e-3)]
    assert spec.extend_fields is None            # fields = "none"
    assert [b.ratio for b in spec.buffers] == [0.2]
    assert card.output == recipe_dir / "mesh" / "small"


def test_lengths_wear_their_units(recipe_dir):
    """h_min_km = 150 reaches the sizing rule as 150000 metres."""
    card = rp.read(recipe_dir / "recipe.toml")
    assert card.spec.sizing.h_min == pytest.approx(0.15e6)
    assert card.spec.sizing.h_max == pytest.approx(0.30e6)
    # And a radius says its unit the same way: 650 km is where the floor
    # went, written as radius_km in the recipe above.
    assert card.spec.insert_radii == [pytest.approx(0.65e6)]


def test_a_radius_is_a_number_and_says_so_when_it_is_not(recipe_dir):
    """The dataset-specific sugar is gone, and its absence is explained."""
    with pytest.raises(ValueError, match="cannot read 'mean_moho'"):
        rp.read(edited(recipe_dir, truncation_line(recipe_dir),
                       'truncate_at = "mean_moho"'))


def test_surfaces_are_centred_on_their_own_mean_radius(recipe_dir):
    """A depth field becomes a boundary at the depth's own mean radius."""
    surfaces = rp.read(recipe_dir / "recipe.toml").spec.surfaces
    moho = surfaces["moho"]
    assert moho.reference_radius == pytest.approx(0.95e6, rel=1e-3)
    assert moho.is_centred()                     # the mean moved into the radius
    assert surfaces["surface"].reference_radius == pytest.approx(1.0e6, rel=1e-3)


def test_a_boundary_and_the_surface_on_it_must_agree(recipe_dir):
    """Both numbers are named, because either one could be the wrong one.

    An interface radius is the boundary's mean radius, so a recipe that
    puts the boundary somewhere its data does not sit is describing two
    different Mohos.  Caught here, where the recipe is still in view,
    rather than in with_surface, which has never heard of one.
    """
    moho = mean_radius(recipe_dir / "moho.xyz")
    with pytest.raises(ValueError, match=r"\[\[surfaces\]\] 'moho' is centred"):
        rp.read(edited(recipe_dir, truncation_line(recipe_dir),
                       f"truncate_at = {moho - 2.0e3:.10f}"))


def test_exaggeration_scales_the_relief_and_not_the_radius(recipe_dir):
    """The boundary stays where the geometry put it; only relief grows."""
    plain = rp.read(edited(recipe_dir, "exaggeration = 2",
                           "exaggeration = 1")).spec.surfaces["moho"]
    loud = rp.read(recipe_dir / "recipe.toml").spec.surfaces["moho"]
    theta = np.array([0.3, 1.2, 2.9])
    assert loud.reference_radius == pytest.approx(plain.reference_radius)
    assert loud.height(theta, 0.0) == pytest.approx(
        2.0 * plain.height(theta, 0.0))


def test_paths_are_relative_to_the_recipe(recipe_dir, tmp_path):
    """A recipe and its data move together."""
    moved = tmp_path / "elsewhere"
    moved.mkdir()
    (moved / "run.toml").write_text((recipe_dir / "recipe.toml").read_text())
    with pytest.raises((OSError, ValueError)):
        rp.read(moved / "run.toml")              # data is not beside it


# ------------------------------------------------------------- refusals

def test_an_unknown_key_is_an_error_not_a_shrug(recipe_dir):
    with pytest.raises(ValueError, match="unknown key"):
        rp.read(edited(recipe_dir, "h_max_km = 300", "h_max_kmm = 300"))


def test_the_offset_sugar_is_gone_and_reads_as_an_unknown_key(recipe_dir):
    """`radius_below = { surface = ..., km = ... }` no longer parses."""
    with pytest.raises(ValueError, match="unknown key"):
        rp.read(edited(recipe_dir, "radius_km = 650",
                       'radius_below = { surface = "moho", km = 300 }'))


def test_an_unregistered_component_lists_the_registered_ones(recipe_dir):
    with pytest.raises(KeyError, match="uniform_interfaces"):
        rp.read(edited(recipe_dir, 'policy = "uniform_interfaces"',
                       'policy = "angular_resolutions"'))


def test_a_reader_that_disagrees_with_the_columns_is_refused(recipe_dir):
    with pytest.raises(ValueError, match="columns"):
        rp.read(edited(recipe_dir, 'reader = "isotropic_deck"',
                       'reader = "isotropic_deck"\ncolumns = ["rho", "vp"]'))


def test_a_surface_with_no_boundary_of_its_name_is_refused(recipe_dir):
    """Renaming the cut leaves the Moho surface nowhere to hang."""
    with pytest.raises(ValueError, match="'moho' attaches to no boundary"):
        rp.read(edited(recipe_dir, 'truncate_name = "moho"',
                       'truncate_name = "crust_base"'))


def test_naming_a_truncation_that_does_not_happen_is_refused(recipe_dir):
    with pytest.raises(ValueError, match="nothing truncates to"):
        rp.read(edited(recipe_dir, truncation_line(recipe_dir) + "\n", ""))


def test_an_insert_without_a_table_says_so(recipe_dir):
    with pytest.raises(ValueError, match="no .geometry.lid. table"):
        rp.read(edited(recipe_dir, 'insert = ["floor", "surface"]',
                       'insert = ["floor", "surface", "lid"]'))


def test_an_inserted_boundary_needs_a_radius(recipe_dir):
    with pytest.raises(ValueError, match="no radius; give radius_m"):
        rp.read(edited(recipe_dir, "  radius_km = 650\n", ""))


# -------------------------------------------------------- the round trip

def test_the_recipe_builds_and_the_manifest_echoes_it(recipe_dir):
    """recipe -> spec -> mesh -> manifest naming the recipe that made it."""
    card = rp.read(recipe_dir / "recipe.toml")
    result = card.build()
    assert result.msh_path.exists()
    assert result.validation.ok

    manifest = sc.read(result.manifest_path)
    assert manifest.provenance["recipe"] == "recipe.toml"
    assert manifest.provenance["recipe_sha256"] == card.digest
    assert manifest.provenance["command"] == "python -m planetmodel.mesh3d recipe.toml"

    # The names a recipe uses are the names the manifest records, so the
    # two can be read side by side.
    assert manifest.sizing["policy"] == "uniform_interfaces"
    assert manifest.mapping["rule"]["name"] == "layer_linear"
    assert manifest.interface_attribute("moho")   # truncation named it
    assert manifest.vacuum_attributes == (len(manifest.layers),)


def test_the_manifest_can_rebuild_the_relief_it_used(recipe_dir):
    """Summed and exaggerated relief still names its files and its factor.

    Mode B hands the reference mesh over and expects the consumer to
    apply the mapping, so a manifest that lost the grids behind a
    ScaledTopography would be describing a mapping nobody can rebuild.
    """
    result = rp.build(recipe_dir / "recipe.toml")
    surfaces = {s["name"]: s for s in
                sc.read(result.manifest_path).mapping["surfaces"]}

    assert surfaces["moho"]["exaggeration"] == 2.0
    # In the order the recipe sums them, which is the order that
    # reconstructs the same relief.
    assert [Path(s["file"]).name
            for s in surfaces["surface"]["sources"]] == ["thickness.xyz",
                                                         "moho.xyz"]
    digests = {s["sha256"] for s in surfaces["moho"]["sources"]}
    assert digests == {sc.file_digest(recipe_dir / "moho.xyz")}


def rebuild_from(card):
    """The mapping, reconstructed from the manifest and the files it names.

    This is the Mode B consumer's job done in Python: read the skeleton
    off the layers, read each surface off the files the manifest names,
    scale as it says, centre it on its own mean, look the rule up by
    name, and ask the body for its mapping.  Nothing here consults the
    objects that built the mesh -- if the manifest has left anything out,
    this cannot compensate for it.
    """
    from planetmodel.model.body import ReferenceBody
    from planetmodel.model.skeleton import Skeleton
    from planetmodel.registry import lookup

    edges = ([card.layers[0]["r_inner_nd"]]
             + [lay["r_outer_nd"] for lay in card.layers])
    body = ReferenceBody.from_fields(Skeleton(edges), {})

    for entry in card.mapping["surfaces"]:
        shape = None
        for s in entry["sources"]:
            # Metres per file unit, over metres per mesh unit: the relief
            # arrives in the same units as the skeleton above.
            piece = GriddedTopography.from_xyz(
                s["file"], scale=s["scale_to_m"] / card.rref_m,
                interpolation=entry["interpolation"])
            shape = piece if shape is None else shape + piece
        surface = Surface(1.0, topography=shape).centred() * entry["exaggeration"]
        body = body.with_surface(entry["interface"] - 1, surface)

    rule = card.mapping["rule"]
    assert rule["registered"], "an unregistered rule cannot be rebuilt by name"
    return body.mapping(rule=lookup("displacement_rule", rule["name"])(
        inner_taper_radius=rule["inner_taper_radius_nd"],
        control_radii=tuple(rule["control_radii_nd"])))


def test_the_manifest_alone_rebuilds_the_mapping(recipe_dir):
    """The Mode B promise: hand over the reference mesh and this file.

    A consumer that applies the mapping itself has nothing else -- not
    the recipe, not the body, not the Python that made them -- so if the
    manifest is short of anything the mapping is not reconstructable and
    Mode B does not work.  Checked to 1e-12 against the live object.
    """
    from planetmodel.mesh3d._units import GeometryScaledMapping
    from planetmodel.mesh3d.layered import resolve_body

    card = rp.read(recipe_dir / "recipe.toml")
    result = card.spec  # the live side, built the way the mesher builds it
    body, _ = resolve_body(result)
    live = GeometryScaledMapping(body.mapping(rule=result.mapping_rule),
                                 result.rref)

    built = rp.build(recipe_dir / "recipe.toml")
    rebuilt = rebuild_from(sc.read(built.manifest_path))

    rng = np.random.default_rng(20260901)
    r = rng.uniform(0.05, 1.19, 400)
    theta = np.arccos(rng.uniform(-1.0, 1.0, 400))
    phi = rng.uniform(-np.pi, np.pi, 400)
    X = np.c_[r * np.sin(theta) * np.cos(phi),
              r * np.sin(theta) * np.sin(phi), r * np.cos(theta)]

    assert rebuilt(X) == pytest.approx(live(X), abs=1e-12)
    assert rebuilt.jacobian(X) == pytest.approx(live.jacobian(X), rel=1e-12)


def test_the_surgered_boundaries_carry_their_relief(recipe_dir):
    """A surface attaches to the boundary the surgery created."""
    result = rp.build(recipe_dir / "recipe.toml")
    names = {i: face.name for i, face in enumerate(result.body.interfaces)}
    attached = {names[i] for i in result.body.surfaces}
    assert attached == {"moho", "surface"}


# ---------------------------------------------------------- the command

def test_check_reports_without_meshing(recipe_dir, capsys):
    assert cli.main([str(recipe_dir / "recipe.toml"), "--check"]) == 0
    out = capsys.readouterr().out
    assert "outer      boundary at 9" in out and "named 'moho'" in out
    assert not (recipe_dir / "mesh").exists()


def test_the_command_builds_the_mesh(recipe_dir, capsys):
    assert cli.main([str(recipe_dir / "recipe.toml")]) == 0
    assert "wrote" in capsys.readouterr().out
    assert (recipe_dir / "mesh" / "small.msh").exists()
    assert (recipe_dir / "mesh" / "small.json").exists()


def test_a_broken_recipe_is_reported_not_raised(recipe_dir, capsys):
    """A recipe is a file a person wrote; a traceback helps nobody."""
    path = edited(recipe_dir, "h_max_km = 300", "h_max_kmm = 300")
    assert cli.main([str(path)]) == 2
    assert "unknown key" in capsys.readouterr().err
