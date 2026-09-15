"""The manifest: assembly, the round trip, and `validate_structure` on
every block."""
import copy
import dataclasses
import json

import pytest

from planetmodel import LayeredIsotropicElastic, Scales
from planetmodel.character import DENSITY, VECTOR
from planetmodel.units import DENSITY as DENSITY_DIMS, G_SI, LENGTH
from planetmodel.mesh3d import manifest as sc

from conftest import full_geometry, hollow_geometry

pytestmark = pytest.mark.gmsh


def card_for(geometry, *, mesh=None, meta=None):
    """A manifest assembled by hand from a geometry, without a mesh."""
    b = geometry.skeleton.boundaries
    layers = [sc.LayerEntry.from_layer(lay, attribute=i + 1, r_inner=b[i],
                                       r_outer=b[i + 1], in_geometry=True)
              for i, lay in enumerate(geometry.layers)]
    faces = [sc.InterfaceEntry.from_interface(f, attribute=k + 1, radius=f.radius)
             for k, f in enumerate(geometry.interfaces)]
    mesh = mesh or sc.mesh_block("m.msh", format="msh", nodes="reference")
    return sc.MeshManifest.from_build(mesh=mesh, layers=layers, interfaces=faces,
                                      meta=meta)


READ_OPTIONS = {"generate_edges": 1, "refine": 0, "fix_orientation": False}


def displacement_entry(n_layers):
    return sc.FieldEntry.from_field(
        "displacement", "m.displacement.gf", fe_space="H1_3D_P2", vdim=3,
        ordering="byNODES", character=VECTOR, dimensions=LENGTH, si=False,
        layers=range(1, n_layers + 1))


def exported_card(geometry, *, with_field=True):
    """A manifest as the MFEM export leaves it, with a displacement and a field."""
    n = geometry.nlayers
    card = card_for(geometry)
    card.mesh = sc.mesh_block("m.mesh", format="mfem", nodes="reference",
                              read_options=READ_OPTIONS, displacement="displacement")
    card.fields = [dataclasses.asdict(displacement_entry(n))]
    if with_field:
        rho = sc.FieldEntry.from_field(
            "rho", "m.rho.gf", fe_space="L2_3D_P2", vdim=1, ordering="byNODES",
            character=DENSITY, dimensions=DENSITY_DIMS, si=True,
            layers=range(1, n + 1))
        card.fields.append(dataclasses.asdict(rho))
        card.scales = sc.scales_block(Scales.SI)
        card.constants = {"G": G_SI}
    sc.validate_structure(card)
    return card


# ------------------------------------------------------------- assembly

def test_from_build_assembles_the_skeleton_and_the_mesh():
    card = card_for(full_geometry(), meta={"run": 1})
    assert card.schema == sc.SCHEMA == "planetmodel.mesh.manifest/4"
    assert card.mesh == {"file": "m.msh", "format": "msh", "nodes": "reference",
                         "read_options": {}, "displacement": None}
    assert [lay["name"] for lay in card.layers] == ["core", "mantle", "crust"]
    assert card.layers[0]["in_geometry"] is True
    assert [f["between_layers"] for f in card.interfaces] == [[0, 1], [1, 2], [2, -1]]
    assert [f["radius"] for f in card.interfaces] == [0.4, 0.8, 1.0]
    assert card.fields == [] and card.scales is None and card.constants == {}
    assert card.meta == {"run": 1}
    assert (card.inner_radius, card.outer_radius) == (0.0, 1.0)
    assert card.layer_attribute("mantle") == 2
    assert card.interface_attribute("surface") == 3
    assert card.shell_attributes == ()
    with pytest.raises(KeyError, match="no interface named"):
        card.interface_attribute("ocean")
    with pytest.raises(KeyError, match="no field named"):
        card.field_record("rho")


def test_a_hollow_geometry_has_one_more_interface_than_layers():
    card = card_for(hollow_geometry())
    assert card.inner_radius == 0.5
    assert [f["between_layers"] for f in card.interfaces] == [[-1, 0], [0, 1], [1, -1]]
    assert [f["name"] for f in card.interfaces] == ["inner", "mid", "outer"]


def test_mesh_block_refuses_unknown_formats_and_nodes():
    block = sc.mesh_block("dir/m.mesh", format="mfem", nodes="physical",
                          read_options=READ_OPTIONS)
    assert block["file"] == "m.mesh" and block["read_options"] == READ_OPTIONS
    with pytest.raises(ValueError, match="format must be"):
        sc.mesh_block("m.vtk", format="vtk", nodes="reference")
    with pytest.raises(ValueError, match="nodes must be"):
        sc.mesh_block("m.msh", format="msh", nodes="moved")


def test_field_entries_carry_character_unit_and_layers():
    e = displacement_entry(3)
    assert dataclasses.asdict(e) == {
        "name": "displacement", "file": "m.displacement.gf",
        "fe_space": "H1_3D_P2", "vdim": 3, "ordering": "byNODES", "rank": 1,
        "weight": 0, "voigt": False, "unit": "1", "layers": [1, 2, 3]}
    rho = sc.FieldEntry.from_field(
        "rho", "d/m.rho.gf", fe_space="L2_3D_P2", vdim=1, ordering="byNODES",
        character=DENSITY, dimensions=DENSITY_DIMS, si=True, layers=[1, 2])
    assert rho.file == "m.rho.gf" and rho.unit == "kg m-3" and rho.weight == 1
    assert sc.FieldEntry.from_field(
        "foo", "m.foo.gf", fe_space="L2_3D_P2", vdim=1, ordering="byNODES",
        character=DENSITY, dimensions=None, si=True, layers=[1]).unit == "unknown"


def test_scales_and_constants_blocks_come_from_the_model():
    model = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[2.0, 1.0], vp=[3.0, 2.0],
                                    vs=[1.0, 0.0],
                                    scales=Scales(length=1.0, mass=2.0, time=0.5))
    assert sc.scales_block(model.scales) == {"length": 1.0, "mass": 2.0,
                                             "time": 0.5}
    assert sc.constants_block(model) == {"G": pytest.approx(G_SI / (0.5 * 4.0))}


def test_default_names_reach_the_entries():
    g = full_geometry().renamed(layers=[None] * 3, interfaces=[None] * 3)
    card = card_for(g)
    assert card.layers[1]["name"] == "layer_2"
    assert card.interfaces[2]["name"] == "interface_3"


# ------------------------------------------------------------ round trip

def test_write_read_round_trip_and_beside(tmp_path):
    card = exported_card(full_geometry())
    path = sc.write(tmp_path / "run.v1.5", card)
    assert path.name == "run.v1.5.json"
    assert sc.beside(tmp_path / "a.msh", ".json").name == "a.json"
    assert sc.beside(tmp_path / "a.mesh", ".displacement.gf").name == \
        "a.displacement.gf"
    back = sc.read(path)
    assert dataclasses.asdict(back) == dataclasses.asdict(card)
    assert back.mesh["displacement"] == "displacement"
    assert back.field_record("rho")["unit"] == "kg m-3"


def test_read_rejects_a_foreign_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "something/else"}')
    with pytest.raises(ValueError, match="schema"):
        sc.read(path)


def test_write_refuses_nan(tmp_path):
    card = card_for(full_geometry())
    card.layers[0]["r_outer"] = float("nan")
    with pytest.raises(ValueError):
        sc.write(tmp_path / "nan", card)


# ------------------------------------------------- validate_structure

def broken(card, block, key, value):
    """A copy of `card` with one key of one block replaced."""
    bad = copy.deepcopy(card)
    target = getattr(bad, block)
    if key is None:
        setattr(bad, block, value)
    else:
        target[key] = value
    return bad


BREAKAGES = [
    ("mesh", "file", None),
    ("mesh", "format", "vtk"),
    ("mesh", "nodes", "moved"),
    ("mesh", "read_options", []),
    ("mesh", "displacement", "u"),
    ("mesh", None, []),
    ("fields", None, "rho"),
    ("fields", None, [{"name": "rho"}]),
    ("scales", None, {"length": 1.0}),
    ("scales", "mass", "two"),
    ("constants", None, []),
    ("constants", "G", "big"),
    ("meta", None, []),
]


@pytest.mark.parametrize("block,key,value", BREAKAGES)
def test_validate_structure_names_the_broken_key(block, key, value):
    card = exported_card(full_geometry())
    sc.validate_structure(card)
    with pytest.raises(ValueError, match=f"malformed manifest: .*{block}"):
        sc.validate_structure(broken(card, block, key, value))


def test_validate_structure_checks_the_entries_and_their_consistency():
    card = exported_card(full_geometry())
    bad = copy.deepcopy(card)
    bad.layers[1]["r_inner"] = "0.4"
    with pytest.raises(ValueError, match=r"layers\[1\].r_inner"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.layers[1]["r_inner"] = 0.5
    with pytest.raises(ValueError, match="layer below ends at"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.layers[0]["in_geometry"] = 1
    with pytest.raises(ValueError, match=r"layers\[0\].in_geometry"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.interfaces[0]["between_layers"] = [1, 0]
    with pytest.raises(ValueError, match=r"interfaces\[0\].between_layers"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.interfaces[2]["attribute"] = 4
    with pytest.raises(ValueError, match=r"interfaces\[2\].attribute"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.interfaces.pop()
    with pytest.raises(ValueError, match="2 interfaces for 3 layers"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.layers = "layers"
    with pytest.raises(ValueError, match="list of objects"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.layers = []
    with pytest.raises(ValueError, match="no layers"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.fields[1]["layers"] = [0]
    with pytest.raises(ValueError, match=r"fields\[1\].layers"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.fields[1]["name"] = "displacement"
    with pytest.raises(ValueError, match="field names repeat"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.fields.pop(0)
    with pytest.raises(ValueError, match="mesh.displacement names"):
        sc.validate_structure(bad)


def test_validate_against_catches_a_mismatch():
    card = card_for(full_geometry())
    sc.validate_against(card, layer_count=3, interface_count=3,
                        groups={"layers": [1, 2, 3], "interfaces": [1, 2, 3]})
    with pytest.raises(ValueError, match="lists 3 layers"):
        sc.validate_against(card, layer_count=9, interface_count=3)
    with pytest.raises(ValueError, match="lists 3 interfaces"):
        sc.validate_against(card, layer_count=3, interface_count=2)
    with pytest.raises(ValueError, match="physical groups are"):
        sc.validate_against(card, layer_count=3, interface_count=3,
                            groups={"layers": [1, 2, 4]})


def test_the_json_is_flat_numbers_and_strings(tmp_path):
    """Every length is a number, and the file carries the schema string."""
    path = sc.write(tmp_path / "flat", card_for(hollow_geometry()))
    data = json.loads(path.read_text())
    assert data["schema"] == sc.SCHEMA
    assert set(data) == {"schema", "mesh", "layers", "interfaces", "fields",
                         "scales", "constants", "meta"}
    assert isinstance(data["layers"][0]["r_inner"], float)
    assert data["fields"] == [] and data["scales"] is None


# ------------------------------------------------------------- describe

def test_describe_is_a_readable_summary_and_str():
    card = exported_card(hollow_geometry())
    card.meta = {"run": 1}
    text = card.describe()
    assert str(card) == text
    lines = text.splitlines()
    assert lines[0] == sc.SCHEMA
    assert "mesh        m.mesh (mfem), reference nodes, read options " \
        "generate_edges 1, refine 0, fix_orientation False, displacement " \
        "displacement" in text
    assert "    1  lower  [0.5, 0.8]  in geometry" in lines
    assert "    2  upper  [0.8, 1]    in geometry" in lines
    assert "    1  inner  radius 0.5  between layers [-1, 0]" in lines
    assert "    3  outer  radius 1    between layers [1, -1]" in lines
    assert "    displacement  m.displacement.gf  H1_3D_P2 vdim 3 byNODES  " \
        "rank 1 weight 0  1       layers [1, 2]" in lines
    assert "    rho           m.rho.gf           L2_3D_P2 vdim 1 byNODES  " \
        "rank 0 weight 1  kg m-3  layers [1, 2]" in lines
    assert "scales      length 1, mass 1, time 1, constants G 6.6743e-11" in text
    assert "meta        run 1" in text
    # a bare mesh lists no fields and no scales
    bare = card_for(full_geometry())
    assert "fields" not in bare.describe() and "scales" not in bare.describe()
    assert "mesh        m.msh (msh), reference nodes" in bare.describe()
    # repr stays one line
    assert repr(card) == \
        "MeshManifest(planetmodel.mesh.manifest/4, m.mesh, 2 layers, " \
        "3 interfaces, 2 fields)"
