"""The manifest: assembly, the round trip, and `validate_structure` on
every block."""
import copy
import dataclasses
import json

import pytest

from planetmodel import IdentityMapping, RadialStretch, ZeroDisplacement
from planetmodel.mesh3d import InterfaceSizing, ValidationReport, manifest as sc
from planetmodel.mesh3d._orient import OrientationReport
from planetmodel.mesh3d.spec import QUALITY_FLOOR

from conftest import full_geometry, hollow_geometry

pytestmark = pytest.mark.gmsh


def card_for(geometry, *, hollow_inner=None, files=None, delivery="physical"):
    """A manifest assembled by hand from a geometry, without a mesh."""
    b = geometry.skeleton.boundaries
    layers = [sc.LayerEntry.from_layer(lay, attribute=i + 1, r_inner_nd=b[i],
                                       r_outer_nd=b[i + 1], in_geometry=True)
              for i, lay in enumerate(geometry.layers)]
    faces = [sc.InterfaceEntry.from_interface(f, attribute=k + 1,
                                              mean_radius_nd=f.radius)
             for k, f in enumerate(geometry.interfaces)]
    sizes = {k: InterfaceSizing(0.1, 0.2, 0.3) for k in range(len(faces))}
    report = ValidationReport(dimension=3, min_sicn=0.4)
    card = sc.MeshManifest.from_build(
        geometry=sc.geometry_block(divisor=2.0, outer_radius_nd=b[-1],
                                   inner_radius_nd=b[0], n_layers=len(layers)),
        mesh=sc.mesh_block(dimension=3, order=2, gmsh_version="4.15",
                           algorithm_2d=6, algorithm_3d=1,
                           counts={"nodes": 10, "elements": 4},
                           curving={"optimized": False}),
        delivery=delivery, layers=layers, interfaces=faces,
        mapping=sc.mapping_block(None, applied_to_nodes=False),
        sizing=sc.sizing_block(policy="UniformInterfaces", sizes=sizes),
        validation=sc.validation_block(report, OrientationReport()),
        provenance=sc.provenance_block(mesh_file="m.msh", meta={"run": 1}))
    card.files = files
    return card


FILES = {"mesh": "m.mesh", "mesh_read_options": {"generate_edges": 1, "refine": 0,
                                                 "fix_orientation": False},
         "grid_functions": [{"kind": "displacement", "name": "displacement",
                             "file": "m.displacement.gf", "fe_space": "H1_3D_P2",
                             "vdim": 3, "ordering": "byNODES"}]}


# ------------------------------------------------------------- assembly

def test_from_build_assembles_every_block():
    card = card_for(full_geometry(), files=FILES)
    assert card.schema == sc.SCHEMA == "planetmodel.mesh.manifest/2"
    assert card.geometry == {"divisor": 2.0, "outer_radius_nd": 1.0,
                             "inner_radius_nd": 0.0, "n_layers": 3}
    assert [lay["name"] for lay in card.layers] == ["core", "mantle", "crust"]
    assert card.layers[0]["in_geometry"] is True
    assert [f["between_layers"] for f in card.interfaces] == [[0, 1], [1, 2], [2, -1]]
    assert card.mapping == {"kind": "IdentityMapping", "repr": "IdentityMapping()",
                            "knots_nd": [], "applied_to_nodes": False}
    assert card.sizing["policy"] == "UniformInterfaces"
    assert card.sizing["per_interface"][1] == {
        "attribute": 2, "size_nd": 0.1, "far_size_nd": 0.2, "decay_width_nd": 0.3}
    assert card.validation["min_sicn"] == 0.4 and card.validation["warnings"] == []
    assert card.provenance["meta"] == {"run": 1}
    assert card.provenance["perturbation"] is None
    assert card.provenance["planetmodel_version"] == sc.planetmodel_version()
    assert card.mesh["msh_version"] == 2.2 and card.mesh["n_elements"] == 4
    assert card.layer_attribute("mantle") == 2
    assert card.interface_attribute("surface") == 3
    assert card.shell_attributes == ()
    with pytest.raises(KeyError, match="no interface named"):
        card.interface_attribute("ocean")


def test_a_hollow_geometry_has_one_more_interface_than_layers():
    card = card_for(hollow_geometry())
    assert card.geometry["inner_radius_nd"] == 0.5
    assert [f["between_layers"] for f in card.interfaces] == [[-1, 0], [0, 1], [1, -1]]
    assert [f["name"] for f in card.interfaces] == ["inner", "mid", "outer"]


def test_mapping_block_records_class_repr_and_knots():
    m = RadialStretch(ZeroDisplacement(), rmax=1.0)
    block = sc.mapping_block(m, knots_nd=[0.5], applied_to_nodes=True)
    assert block["kind"] == "RadialStretch" and block["repr"] == repr(m)
    assert block["knots_nd"] == [0.5] and block["applied_to_nodes"] is True
    assert sc.mapping_block(IdentityMapping(), applied_to_nodes=False)["kind"] == \
        "IdentityMapping"


def test_default_names_reach_the_entries():
    g = full_geometry().renamed(layers=[None] * 3, interfaces=[None] * 3)
    card = card_for(g)
    assert card.layers[1]["name"] == "layer_2"
    assert card.interfaces[2]["name"] == "interface_3"


# ------------------------------------------------------------ round trip

def test_write_read_round_trip_and_beside(tmp_path):
    card = card_for(full_geometry(), files=FILES, delivery="referential")
    path = sc.write(tmp_path / "run.v1.5", card)
    assert path.name == "run.v1.5.json"
    assert sc.beside(tmp_path / "a.msh", ".json").name == "a.json"
    assert sc.beside(tmp_path / "a.mesh", ".displacement.gf").name == \
        "a.displacement.gf"
    back = sc.read(path)
    assert dataclasses.asdict(back) == dataclasses.asdict(card) | {
        "mesh": card.mesh | {"date": back.mesh["date"]}}
    assert back.delivery == "referential" and back.files == FILES
    assert sc.file_digest(path) == sc.file_digest(path)
    assert sc.file_digest(tmp_path / "missing") is None


def test_read_rejects_a_foreign_schema(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text('{"schema": "something/else"}')
    with pytest.raises(ValueError, match="schema"):
        sc.read(path)


def test_write_refuses_nan(tmp_path):
    card = card_for(full_geometry())
    card.validation["min_sicn"] = float("nan")
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
    ("geometry", "divisor", "two"),
    ("geometry", "n_layers", 2.5),
    ("geometry", None, []),
    ("mesh", "dimension", 4),
    ("mesh", "element_order", "2"),
    ("mesh", "high_order_optimised", 1),
    ("mesh", "gmsh_version", 4.15),
    ("mapping", "kind", None),
    ("mapping", "knots_nd", [0.5, "x"]),
    ("mapping", "applied_to_nodes", 0),
    ("sizing", "policy", 3),
    ("sizing", "per_interface", [{"attribute": 1}]),
    ("validation", "negative_jacobians", 0.0),
    ("validation", "warnings", ["ok", 3]),
    ("provenance", "mesh_file", None),
    ("provenance", "meta", []),
    ("provenance", "perturbation", "none"),
    ("delivery", None, "halfway"),
    ("files", None, {"mesh": "m.mesh"}),
    ("files", None, {"mesh": "m.mesh", "mesh_read_options": {},
                     "grid_functions": []}),
    ("files", None, {"mesh": "m.mesh", "mesh_read_options": {"refine": 0},
                     "grid_functions": [{"name": "u"}]}),
]


@pytest.mark.parametrize("block,key,value", BREAKAGES)
def test_validate_structure_names_the_broken_key(block, key, value):
    card = card_for(full_geometry(), files=FILES)
    sc.validate_structure(card)
    with pytest.raises(ValueError, match=f"malformed manifest: .*{block}"):
        sc.validate_structure(broken(card, block, key, value))


def test_validate_structure_checks_the_entries_and_their_consistency():
    card = card_for(full_geometry())
    bad = copy.deepcopy(card)
    bad.layers[1]["r_inner_nd"] = "0.4"
    with pytest.raises(ValueError, match=r"layers\[1\].r_inner_nd"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.layers[1]["r_inner_nd"] = 0.5
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
    bad.geometry["n_layers"] = 2
    with pytest.raises(ValueError, match="n_layers is 2"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.geometry["outer_radius_nd"] = 1.5
    with pytest.raises(ValueError, match="outer_radius_nd"):
        sc.validate_structure(bad)
    bad = copy.deepcopy(card)
    bad.layers = "layers"
    with pytest.raises(ValueError, match="list of objects"):
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
    assert isinstance(data["layers"][0]["r_inner_nd"], float)
    assert data["files"] is None
    assert data["validation"]["min_sicn"] >= QUALITY_FLOOR
