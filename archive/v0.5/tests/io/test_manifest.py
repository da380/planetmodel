"""The mesh manifest: built from typed entries, validated, round-tripped.

Nothing here needs gmsh: the manifest is the JSON a mesh travels with,
and its shape is checked on a hand-built body with a fluid, a solid and
a vacuum layer.
"""
import json
from types import SimpleNamespace

import numpy as np
import pytest

from planetmodel import RadialField, ReferenceBody, Skeleton
from planetmodel.io import manifest as mf


def three_layer_body():
    """Solid core, fluid shell, vacuum outside; density on the first two."""
    sk = Skeleton([0.0, 0.5, 1.0])
    rho = RadialField(sk, [lambda r: 0 * r + 2.0, lambda r: 0 * r + 1.0],
                      name="rho")
    body = (ReferenceBody.from_fields(sk, {"rho": rho})
            .annotate(0, name="core").annotate(1, name="ocean", state="fluid")
            .name_interface(0, "cmb").name_interface(1, "surface")
            .with_buffer(ratio=0.5))
    return body


def report_and_orientation():
    report = SimpleNamespace(negative_jacobians=0, min_sicn=0.4,
                             negative_cells=0, inward_faces=0,
                             max_interface_radius_error=1e-9,
                             knots_aligned=True, warnings=["one warning"])
    orientation = SimpleNamespace(faces_flipped=3)
    return report, orientation


def built_card(body):
    b = np.asarray(body.skeleton.boundaries)
    layers = [mf.LayerEntry.from_layer(lay, attribute=i + 1, r_inner_nd=b[i],
                                       r_outer_nd=b[i + 1])
              for i, lay in enumerate(body.layers)]
    interfaces = [mf.InterfaceEntry.from_interface(
        face, attribute=i + 1, mean_radius_nd=face.radius)
        for i, face in enumerate(body.interfaces)]
    report, orientation = report_and_orientation()
    sizing = SimpleNamespace(size=0.1, far_size=0.3, decay_width=0.2)
    return mf.MeshManifest.from_build(
        model={"name": "three", "source": None, "sha256": None, "rref_m": 1.0,
               "units": mf.units_block(body.scales, 1.0, 1.0)},
        mesh=mf.mesh_block(dimension=3, order=2, gmsh_version="4.15",
                           algorithm_2d=6, algorithm_3d=1,
                           counts={"nodes": 10, "elements": 20},
                           curving={"optimized": False}),
        delivery="physical", layers=layers, interfaces=interfaces,
        sizing=mf.sizing_block(policy="uniform_interfaces",
                               sizes={0: sizing, 1: sizing, 2: sizing}),
        validation=mf.validation_block(report, orientation),
        provenance=mf.provenance_block(mesh_file="three.msh"))


def test_the_entries_say_what_each_layer_is():
    body = three_layer_body()
    card = built_card(body)
    assert [lay["name"] for lay in card.layers] == ["core", "ocean", "buffer"]
    assert [lay["state"] for lay in card.layers] == ["solid", "fluid", "vacuum"]
    assert [lay["is_vacuum"] for lay in card.layers] == [False, False, True]
    assert [lay["fields"] for lay in card.layers] == [["rho"], ["rho"], []]
    # No moduli anywhere: no law to name.
    assert [lay["law"] for lay in card.layers] == [None, None, None]
    assert card.layers[1]["r_inner_nd"] == 0.5 and card.layers[2]["r_outer_nd"] == 1.5
    assert [f["between_layers"] for f in card.interfaces] == [[0, 1], [1, 2], [2, -1]]
    assert [f["name"] for f in card.interfaces] == ["cmb", "surface", "buffer"]
    assert card.vacuum_attributes == (3,)
    assert card.layer_attribute("ocean") == 2 and card.interface_attribute("cmb") == 1
    assert card.validation["faces_reoriented"] == 3
    assert card.validation["warnings"] == ["one warning"]
    assert card.provenance["planetmodel_version"] == mf.planetmodel_version()
    assert card.provenance["perturbation"] is None
    assert card.mesh["n_elements"] == 20 and card.sizing["per_interface"][2] == {
        "attribute": 3, "size_nd": 0.1, "far_size_nd": 0.3, "decay_width_nd": 0.2}


def test_the_law_is_read_from_the_field_the_layer_holds():
    from planetmodel import PREM
    prem = PREM(ocean=False)
    assert mf.law_name_of(prem.layers[3]) == "constant_q"
    # The static moduli alone: no frequency dependence to name.
    static = prem.layers[3].without_field("viscoelastic_moduli")
    assert mf.law_name_of(static) == "static"
    # A field with no record and no source is not something a file can rebuild.
    handmade = SimpleNamespace(fields={"elastic_moduli": object(),
                                      "viscoelastic_moduli": object()})
    assert mf.law_name_of(handmade) is None
    assert mf.law_name_of(three_layer_body().layers[0]) is None


def test_write_and_read_round_trip(tmp_path):
    card = built_card(three_layer_body())
    path = mf.write(tmp_path / "three", card)
    assert path.name == "three.json"
    back = mf.read(path)
    assert back.layers == card.layers and back.interfaces == card.interfaces
    assert back.delivery == "physical" and back.schema == mf.SCHEMA
    assert "date" in back.mesh
    with open(path) as fh:
        raw = json.load(fh)
    assert set(raw) >= {"schema", "model", "mesh", "delivery", "layers",
                        "interfaces", "sizing", "validation", "provenance"}


def test_from_build_and_read_validate_the_same_structure(tmp_path):
    card = built_card(three_layer_body())
    mf.validate_structure(card)

    def broken(mutate):
        bad = built_card(three_layer_body())
        mutate(bad)
        return bad

    for mutate, message in (
            (lambda c: setattr(c, "delivery", "banana"), "delivery"),
            (lambda c: c.layers[1].__setitem__("r_inner_nd", 0.4), "starts at 0.4"),
            (lambda c: c.layers[0].__setitem__("fields", "rho"), "fields"),
            (lambda c: c.layers[0].pop("is_vacuum"), "is_vacuum"),
            (lambda c: c.layers[0].__setitem__("is_vacuum", 1), "is_vacuum"),
            (lambda c: c.layers[0].__setitem__("law", 3), r"layers\[0\].law"),
            (lambda c: c.interfaces[0].__setitem__("between_layers", [5, 9]),
             "between_layers"),
            (lambda c: c.interfaces[0].__setitem__("role", "odd"), "role"),
            (lambda c: c.model.pop("rref_m"), "rref_m")):
        with pytest.raises(ValueError, match=message):
            mf.validate_structure(broken(mutate))
    for law in ("constant_q", "static", None):
        ok = broken(lambda c: c.layers[0].__setitem__("law", law))
        mf.validate_structure(ok)

    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"schema": mf.SCHEMA, "interfaces": "oops"}))
    with pytest.raises(ValueError, match="malformed"):
        mf.read(path)
    path.write_text(json.dumps({"schema": "somebody.else/9"}))
    with pytest.raises(ValueError, match="schema"):
        mf.read(path)


def test_the_files_block_is_checked_entry_by_entry():
    ok = built_card(three_layer_body())
    ok.files = {"mesh": "m.mesh", "mesh_read_options": {"refine": 0},
                "grid_functions": [{
                    "kind": "field", "name": "rho", "file": "m.rho.gf",
                    "fe_space": "L2_3D_P2", "vdim": 1, "ordering": "byNODES",
                    "frame": "cartesian", "units": "kg m-3",
                    "character_rank": 0, "character_weight": 1,
                    "physical_dimensions": [1, -3, 0], "layers": [0]}]}
    mf.validate_structure(ok)
    for key, value, message in (
            ("mesh", None, "files.mesh"),
            ("mesh_read_options", {}, "mesh_read_options"),
            ("grid_functions", {}, "grid_functions")):
        bad = built_card(three_layer_body())
        bad.files = dict(ok.files, **{key: value})
        with pytest.raises(ValueError, match=message):
            mf.validate_structure(bad)
    for key, value in (("vdim", "one"), ("layers", "0"),
                       ("physical_dimensions", [1, 2])):
        bad = built_card(three_layer_body())
        bad.files = dict(ok.files, grid_functions=[
            dict(ok.files["grid_functions"][0], **{key: value})])
        with pytest.raises(ValueError, match=key):
            mf.validate_structure(bad)


def test_the_mesh_s_groups_are_compared_with_the_manifest():
    card = built_card(three_layer_body())
    mf.validate_against(card, layer_count=3, interface_count=3,
                        groups={"layers": [1, 2, 3], "interfaces": [1, 2, 3]})
    with pytest.raises(ValueError, match="layers physical groups"):
        mf.validate_against(card, layer_count=3, interface_count=3,
                            groups={"layers": [1, 2], "interfaces": [1, 2, 3]})
    with pytest.raises(ValueError, match="lists 3 layers"):
        mf.validate_against(card, layer_count=2, interface_count=3)
    card.layers[0]["attribute"] = 7
    with pytest.raises(ValueError, match="attributes are"):
        mf.validate_against(card, layer_count=3, interface_count=3)


def test_a_dotted_basename_survives(tmp_path):
    assert mf.beside(tmp_path / "run.v1.5", ".json").name == "run.v1.5.json"
    assert mf.beside(tmp_path / "run.msh", ".json").name == "run.json"
