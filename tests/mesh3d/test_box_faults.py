"""Faults built into a box: fragmented in, trimmed at the box, labelled.

Unit boxes a few elements across, faults in a Cartesian frame (z up, the
box's own points).  Every built mesh is held to `testing.check_fault_mesh`,
the contract a consumer relies on, as on the sphere: the faces conform
and are oriented footwall to hanging wall, every node of them lies on
the fault to rounding (the mapping is the identity, so the forward
evaluation is the reference one), the curve groups are the edges of the
faces by the format page's rule, and the collars are the classification
of the walls.  A fault may end inside the box, cross an interface, reach
the free surface (its free edge, the trace), and leave through a side or
the base (its face edge, a curve of its own, not pinned); several faults
share a box.  The collar of a plane lies on its normal's side by
position, and the footwall's cells on the other.  The manifest is schema
7, validated, round-tripped and held to its rules; what the box cannot
take is refused by name.  AdGIA's faulted box (`meshes/faulted_box.py`,
its flat variant) is reproduced at unit scale.
"""
import copy
import json

import numpy as np
import pytest

from planetmodel import Skeleton, cartesian, testing
from planetmodel.fault import CartesianFrame, Fault, GeographicFrame, Plane, Slip
from planetmodel.mapping import MappingBase
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, manifest as sc,
                                read_mesh)

from test_collars import boundary_faces, kuhn

pytestmark = pytest.mark.gmsh

LOCAL = CartesianFrame()
ZERO = Slip(np.zeros((1, 1, 3)))
DEPTHS = [0.0, 0.4, 1.0]
#: Element size on the interfaces: a few elements across the unit box.
H = 0.15
SIZING = UniformInterfaces(H, 0.3, 0.3)


def box(**kw):
    """Two layers of a unit box, x in [0, 1] and y in [-0.5, 0.5]."""
    return cartesian.Geometry(Skeleton(DEPTHS),
                              cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                              layer_names=["upper", "lower"],
                              interface_names=["surface", "moho"], **kw)


def plane(x, y, depth, **kw):
    """A plane of centroid (x, y) at `depth`, striking along y and dipping
    50 degrees towards +x by default."""
    kw = {"strike": 0.0, "dip": 50.0, "length": 0.4, "width": 0.25} | kw
    return Plane.at_depth(LOCAL, x, y, depth, slip=ZERO, **kw)


def top_at_surface(dip=50.0, width=0.25):
    """The centroid depth of a plane whose up-dip edge is at the surface."""
    return 0.5 * width * np.sin(np.deg2rad(dip))


def splayed():
    """A main fault and a splay rising from its centre line, ending on it."""
    main = plane(0.45, 0.0, 0.6, dip=60.0, length=0.5, width=0.3)
    foot = main.cartesian(0.0, -0.1)
    sp = Plane(LOCAL, foot, strike=0.0, dip=30.0, length=0.3, width=0.15, slip=ZERO)
    sp = Plane(LOCAL, sp.cartesian(0.0, 0.0) + foot - sp.cartesian(0.0, 0.5),
               strike=0.0, dip=30.0, length=0.3, width=0.15, slip=ZERO)
    return {"main": main, "splay": sp}


#: Each configuration: its faults by name, and the curves each fault has.
CASES = {
    "buried": ({"f": plane(0.5, 0.0, 0.7)}, {"f": {"tip"}}),
    "crossing": ({"f": plane(0.5, 0.0, 0.4, width=0.3)}, {"f": {"tip"}}),
    "surface": ({"f": plane(0.5, 0.0, top_at_surface())},
                {"f": {"tip", "free_edge"}}),
    # through the side x = 1, the part beyond it trimmed
    "side": ({"f": plane(0.9, 0.0, 0.6, strike=90.0)}, {"f": {"tip", "face_edge"}}),
    # its edge on the side x = 1: carried across it and trimmed
    "side_edge": ({"f": plane(0.8, 0.0, 0.6, strike=90.0)},
                  {"f": {"tip", "face_edge"}}),
    # its down-dip edge on the base: carried across it
    "base": ({"f": plane(0.5, 0.0, 1.0 - top_at_surface(70.0, 0.3), dip=70.0,
                         width=0.3)}, {"f": {"tip", "face_edge"}}),
    # rupturing the surface and leaving through the side x = 1
    "surface_side": ({"f": plane(0.9, 0.0, top_at_surface(), strike=90.0)},
                     {"f": {"tip", "free_edge", "face_edge"}}),
    "two": ({"a": plane(0.3, 0.0, 0.7), "b": plane(0.75, 0.1, 0.25, strike=90.0,
                                                    length=0.6)},
            {"a": {"tip"}, "b": {"tip", "face_edge"}}),
    "splay": (splayed(), {"main": {"tip", "junction"},
                          "splay": {"tip", "junction"}}),
}


def build(root, name, *, order=2, geometry=None, **spec):
    """Build a configuration of `CASES` (or {name: Plane} given) in a box."""
    faults = CASES[name][0] if isinstance(name, str) else name
    tag = name if isinstance(name, str) else "_".join(faults)
    items = [EmbeddedFault(Fault([p]), name=n) for n, p in faults.items()]
    return build_layered_mesh(MeshSpec(geometry or box(), SIZING, faults=items,
                                       order=order, **spec), root / f"{tag}_{order}")


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each configuration built once at order 2."""
    root = tmp_path_factory.mktemp("box_faults")
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = build(root, name)
        return cache[name]
    return get


def physical_planes(card):
    """Each face of the box and its top: name -> (axis, value)."""
    out = {e["name"]: (sc.AXES.index(e["axis"]), e["value"]) for e in card.faces}
    out["top"] = (2, -card.interfaces[0]["depth"])
    return out


# ------------------------------------------------------------ the contract

@pytest.mark.parametrize("name", list(CASES))
def test_every_configuration_holds_to_the_contract(built, name):
    """The spherical contract on a box: built, validated, held to
    `check_fault_mesh` with its nodes on the fault to rounding."""
    res = built(name)
    v = res.validation
    assert v.ok, v.failures
    assert v.misoriented_fault_faces == 0 and v.nonconforming_fault_faces == 0
    assert v.inward_faces == 0 and v.negative_cells == 0 and v.min_sicn > 0.0
    faults = {n: Fault([p]) for n, p in CASES[name][0].items()}
    testing.check_fault_mesh(res.msh_path, faults, distance=1e-12)
    card = sc.read(res.manifest_path)
    assert card.schema == sc.SCHEMA_FLAT
    assert res.counts["faults"] == len(faults) and res.counts["faces"] == 5
    assert 100 < res.counts["dim3"] < 5000


@pytest.mark.parametrize("name", list(CASES))
def test_each_fault_has_the_curves_where_it_meets_the_box(built, name):
    """A fault's curve groups are the ones its place says, numbered
    4k - 3 to 4k, and each lies where it should: its free edge on the
    top, its face edge on the side or base it leaves through, its tip
    inside the box."""
    res = built(name)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    planes = physical_planes(card)
    for k, record in enumerate(card.faults, start=1):
        want = CASES[name][1][record["name"]]
        got = {key for key in sc.FLAT_FAULT_CURVES if record[key] is not None}
        assert got == want, record["name"]
        for key, group in zip(sc.FLAT_FAULT_CURVES,
                              sc.fault_curve_groups(k, curves=sc.FLAT_FAULT_CURVES)):
            if record[key] is None:
                continue
            assert record[key] == group == 4 * k - 3 + sc.FLAT_FAULT_CURVES.index(key)
            x = mesh.points[np.unique(mesh.curves[group])]
            on = {where for where, (axis, value) in planes.items()
                  if np.all(np.abs(x[:, axis] - value) <= 1e-12)}
            if key == "free_edge":
                assert on == {"top"}
            elif key == "face_edge":
                assert len(on) == 1 and on <= {"base", "x_min", "x_max", "y_min",
                                               "y_max"}
            elif key == "tip":
                inside = np.ones(len(x), dtype=bool)
                for axis, value in planes.values():
                    inside &= np.abs(x[:, axis] - value) > 1e-12
                assert np.any(inside)
        assert mesh.names[(2, record["attribute"])] == record["name"]
        for key in sc.FLAT_FAULT_CURVES:
            if record[key] is not None:
                assert mesh.names[(1, record[key])] == f"{record['name']}_{key}"


def test_a_fault_reaches_the_surface_where_its_edge_was(built):
    """The trace of a surface-rupturing plane is its up-dip edge, on z = 0,
    along y over the plane's length; it carries slip, and the walls of
    its free edge are free: the collar reaches the surface."""
    res = built("surface")
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    trace = mesh.points[np.unique(mesh.curves[card.faults[0]["free_edge"]])]
    p = CASES["surface"][0]["f"]
    assert np.abs(trace[:, 2]).max() <= 1e-12
    assert np.allclose(trace[:, 0], p.cartesian(0.0, -0.5)[0], rtol=0.0, atol=1e-12)
    assert trace[:, 1].min() == pytest.approx(-0.2, abs=1e-12)
    assert trace[:, 1].max() == pytest.approx(0.2, abs=1e-12)
    collar = np.isin(mesh.cell_attributes, card.faults[0]["hanging_wall"])
    on_top = mesh.points[mesh.cells[collar, :4], 2].max(axis=1) >= -1e-12
    assert np.any(on_top)


@pytest.mark.parametrize("name", ["side", "side_edge", "base", "surface_side"])
def test_a_face_edge_carries_slip_and_its_cells_are_in_the_walls(built, name):
    """Where a fault leaves the box the rule finds its face edge, not a
    tip: nothing pinned there, and the cells touching its vertices there
    are in its walls.  Without the box's faces the same edges would be
    taken for its tip."""
    res = built(name)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    faces = mesh.faces_with(card.faults[0]["attribute"])[:, :3]
    surface = mesh.faces_with(card.surface_attribute)[:, :3]
    box_faces = mesh.boundary_of(card.face_attributes)
    (curves,) = sc.fault_curves([faces], surface, faces=box_faces)
    assert len(curves.face_edge)
    pinned = {tuple(e) for e in curves.pinned.tolist()}
    assert not pinned & {tuple(e) for e in curves.face_edge.tolist()}
    (blind,) = sc.fault_curves([faces], surface)
    assert {tuple(e) for e in blind.tip.tolist()} == (
        {tuple(e) for e in curves.tip.tolist()}
        | {tuple(e) for e in curves.face_edge.tolist()})
    walls = mesh.walls(card)[0]
    free = np.setdiff1d(np.unique(curves.face_edge), np.unique(curves.pinned))
    on_edge = np.isin(mesh.cells[:, :4], free).any(axis=1)
    assert free.size and np.all(walls[on_edge] != 0)


def test_two_faults_and_a_splay_share_a_box(built):
    """Two faults apart keep their own groups and collars; a splay ending
    on its main fault meets it along a junction in both faults' groups,
    ends on it, and is pinned there."""
    card = sc.read(built("two").manifest_path)
    a, b = card.faults
    assert (a["attribute"], b["attribute"]) == (8, 9)
    assert a["meets"] == b["meets"] == [] and a["junction"] is b["junction"] is None
    assert set(a["hanging_wall"]).isdisjoint(b["hanging_wall"])
    card = sc.read(built("splay").manifest_path)
    main, splay = card.faults
    assert main["meets"] == ["splay"] and splay["meets"] == ["main"]
    assert splay["ends_on"] == ["main"] and main["ends_on"] == []
    assert (main["junction"], splay["junction"]) == (3, 7)


@pytest.mark.parametrize("name", ["buried", "crossing", "side", "surface_side",
                                  "base"])
def test_the_collar_lies_on_the_hanging_wall_side_by_position(built, name):
    """For a plane, every cell of its hanging wall lies on its normal's side
    of the plane and every cell of its footwall on the other, by their
    centroids' signed distance from the plane; the collar attributes are
    those of the hanging wall, in the layers its faces cross."""
    res = built(name)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    p = CASES[name][0]["f"]
    centroid = mesh.points[mesh.cells[:, :4]].mean(axis=1)
    side = (centroid - p.cartesian(0.0, 0.0)) @ p.normal(0.0, 0.0)
    walls = mesh.walls(card)[0]
    assert np.count_nonzero(walls == 1) and np.count_nonzero(walls == -1)
    assert np.all(side[walls == 1] > 0.0) and np.all(side[walls == -1] < 0.0)
    collar = np.isin(mesh.cell_attributes, card.faults[0]["hanging_wall"])
    assert np.array_equal(collar, walls == 1)
    owners = {int(card.layer_of(a)["attribute"])
              for a in card.faults[0]["hanging_wall"]}
    assert owners == set(card.faults[0]["layers"])


# ------------------------------------------------------------ the manifest

def test_the_flat_manifest_with_faults_describes_them_and_round_trips(built, tmp_path):
    res = built("surface_side")
    data = json.loads(res.manifest_path.read_text())
    assert data["schema"] == sc.SCHEMA_FLAT == "planetmodel.mesh.manifest/7"
    assert [e["attributes"] for e in data["layers"]] == [[1, 3], [2]]
    record = data["faults"][0]
    assert list(record)[-1] == "face_edge"
    assert (record["attribute"], record["tip"], record["free_edge"],
            record["junction"], record["face_edge"]) == (8, 1, 2, None, 4)
    card = sc.read(res.manifest_path)
    assert card.is_flat and card.has_faults
    assert card.fault_curves == sc.FLAT_FAULT_CURVES
    assert card.face_attributes == (3, 4, 5, 6, 7)
    assert card.element_attributes() == (1, 3, 2)
    text = card.describe()
    assert "face edge 4" in text and "free edge 2" in text and "collars [3]" in text
    again = sc.write(tmp_path / "again", card)
    assert again.read_text() == res.manifest_path.read_text()
    assert sc.read(again) == card
    sc.validate_against(card, layer_count=2, interface_count=2, fault_count=1,
                        groups={"faces": [3, 4, 5, 6, 7], "faults": [8],
                                "collars": [3]})


def test_a_flat_manifest_without_faults_has_none_of_their_keys(tmp_path):
    """The faults' keys are not written for a box without faults: its
    manifest is schema 7 with no `faults` and no layer `attributes`, and
    reads back as written."""
    g = box()
    b = g.skeleton.boundaries
    layers = [sc.FlatLayerEntry.from_layer(lay, attribute=i + 1, depth_top=b[i],
                                           depth_bottom=b[i + 1])
              for i, lay in enumerate(g.layers)]
    interfaces = [sc.FlatInterfaceEntry.from_interface(f, attribute=k + 1,
                                                       depth=f.radius)
                  for k, f in enumerate(g.interfaces)]
    faces = [sc.FaceEntry(attribute=3 + k, name=name, axis=axis, value=value,
                          outward=outward)
             for k, (name, axis, outward, value) in enumerate(
                 [("base", "z", -1, -1.0), ("x_min", "x", -1, 0.0),
                  ("x_max", "x", 1, 1.0), ("y_min", "y", -1, -0.5),
                  ("y_max", "y", 1, 0.5)])]
    card = sc.MeshManifest.from_build(
        mesh=sc.mesh_block("m.msh", format="msh", nodes="reference"),
        layers=layers, interfaces=interfaces, faces=faces,
        coordinates=("x", "y", "z"))
    assert card.schema == sc.SCHEMA_FLAT and not card.has_faults
    assert all("attributes" not in e for e in card.layers)
    path = sc.write(tmp_path / "m", card)
    data = json.loads(path.read_text())
    assert data["schema"] == sc.SCHEMA_FLAT and "faults" not in data
    assert sc.read(path) == card and sc.write(tmp_path / "again", card).read_text() \
        == path.read_text()
    empty = tmp_path / "empty.json"
    empty.write_text(json.dumps(data | {"faults": []}))
    assert sc.read(empty) == card


def broken(card, change):
    bad = copy.deepcopy(card)
    change(bad)
    return bad


FAULTED_BREAKAGES = [
    (lambda c: c.faults[0].__setitem__("face_edge", 3), r"face_edge is 3, expected 4"),
    (lambda c: c.faults[0].__setitem__("free_edge", 5), r"free_edge is 5, expected 2"),
    (lambda c: c.faults[0].__setitem__("attribute", 3),
     r"attribute is 3, expected 8, after the 2 interfaces and the 5 faces"),
    (lambda c: c.faults[0].__setitem__("name", "x_max"), "share a name with a face"),
    (lambda c: c.faults[0].pop("face_edge"), r"faults\[0\].face_edge"),
    (lambda c: c.layers[0].__setitem__("attributes", [1]),
     r"hanging_wall is \[3\], not collar"),
    (lambda c: c.faults[0].__setitem__("hanging_wall", []), "which is in no fault"),
    (lambda c: c.faults.clear(),
     r"faults lists none, and layers\[0\].attributes is listed"),
    (lambda c: setattr(c, "schema", sc.SCHEMA_WITH_FAULTS), r"layers\[0\].r_inner"),
]


@pytest.mark.parametrize("change,match", FAULTED_BREAKAGES)
def test_validate_structure_holds_a_faulted_flat_manifest_to_its_rules(built, change,
                                                                        match):
    card = sc.read(built("surface_side").manifest_path)
    sc.validate_structure(card)
    with pytest.raises(ValueError, match=f"malformed manifest: .*{match}"):
        sc.validate_structure(broken(card, change))


# ------------------------------------------------------------ the rules

def test_the_curves_rule_takes_a_box_s_faces():
    """A square of the plane z = 1/2 in a unit cube of 4^3 cubes, x and y
    in [1/4, 1]: its edges on x = 1 and y = 1 are its face edge, on the
    cube's sides; the others its tip, the only edges pinned; the cells
    touching its face edge are in its walls."""
    points, cells = kuhn(4)
    pick = [tuple(sorted(c[f].tolist())) for c in cells for f in sc._TET_FACES
            if np.allclose(points[c[f], 2], 0.5)
            and points[c[f], :2].min() >= 0.25 - 1e-12]
    faces = np.array(sorted(set(pick)))
    n = np.cross(points[faces[:, 1]] - points[faces[:, 0]],
                 points[faces[:, 2]] - points[faces[:, 0]])
    faces[n[:, 2] < 0] = faces[n[:, 2] < 0][:, [0, 2, 1]]
    outer = boundary_faces(cells)
    top = outer[np.all(np.isclose(points[outer, 2], 1.0), axis=1)]
    sides = outer[~np.all(np.isclose(points[outer, 2], 1.0), axis=1)]
    (curves,) = sc.fault_curves([faces], top, faces=sides)
    x = points[curves.face_edge]
    assert len(curves.face_edge) == 6
    assert np.all(np.isclose(x[..., 0], 1.0).all(axis=1)
                  | np.isclose(x[..., 1], 1.0).all(axis=1))
    assert len(curves.tip) == 6 and curves.free_edge.shape == (0, 2)
    assert curves.pinned.tolist() == curves.tip.tolist()
    walls = sc.fault_walls(points, cells, faces, curves.pinned, name="f")
    z = points[cells].mean(axis=1)[:, 2]
    assert np.all(z[walls == 1] > 0.5) and np.all(z[walls == -1] < 0.5)
    free = np.setdiff1d(np.unique(curves.face_edge), np.unique(curves.pinned))
    on_edge = np.isin(cells, free).any(axis=1)
    assert free.size == 5 and np.all(walls[on_edge] != 0)
    # pinned there, as without the box's faces, fewer cells are in a wall
    blind = sc.fault_walls(points, cells, faces,
                           sc.fault_curves([faces], top)[0].pinned, name="f")
    assert np.all(blind[blind != 0] == walls[blind != 0])
    assert np.count_nonzero(blind) < np.count_nonzero(walls)
    assert np.array_equal(sc.walls_of_faults(points, cells, [faces], top, faces=sides,
                                             names=["f"])[0], walls)
    assert sc.fault_curve_groups(2) == (4, 5, 6)
    assert sc.fault_curve_groups(2, curves=sc.FLAT_FAULT_CURVES) == (5, 6, 7, 8)


# ------------------------------------------------------------ refusals

class Stretch(MappingBase):
    """A vertical stretch, z -> 1.2 z."""

    def __call__(self, X):
        return np.asarray(X, dtype=float) * [1.0, 1.0, 1.2]

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(np.diag([1.0, 1.0, 1.2]), X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        return np.full(np.shape(X)[:-1], 1.2)


def test_what_a_box_does_not_take_is_refused_by_name():
    fault = Fault([plane(0.5, 0.0, 0.7)])
    with pytest.raises(ValueError, match=r"faults in a section \(a flat geometry over "
                                         r"a Rectangle\) are not built: a fault there "
                                         "is a segment"):
        MeshSpec(cartesian.Geometry(Skeleton(DEPTHS), cartesian.Rectangle((0.0, 1.0))),
                 SIZING, faults=[EmbeddedFault(fault)])
    # under a mapping the fault is pulled back at the build, as on the sphere
    assert MeshSpec(box(mapping=Stretch()), SIZING, faults=[EmbeddedFault(fault)])
    sphere = Fault([Plane.at_depth(GeographicFrame(1.0), 0.0, 0.0, 0.2, strike=0.0,
                                   dip=45.0, length=0.2, width=0.1, slip=ZERO)])
    with pytest.raises(ValueError, match="is in a GeographicFrame: a flat geometry "
                                         "over a Box takes faults in a CartesianFrame"):
        MeshSpec(box(), SIZING, faults=[EmbeddedFault(sphere)])
    with pytest.raises(ValueError, match=r"faults named \['base'\] would share a name "
                                         "with a face"):
        MeshSpec(box(), SIZING, faults=[EmbeddedFault(fault, name="base")])
    assert MeshSpec(box(), SIZING, faults=[EmbeddedFault(fault)]).fault_names == [
        "fault_1"]


@pytest.mark.parametrize("make,match", [
    (lambda: plane(2.0, 0.0, 0.5), "wholly outside the geometry"),
    # its edge 0.005 short of the side x = 1, running along it
    (lambda: plane(0.795, 0.0, 0.6, strike=90.0),
     r"comes within 0.005 of face 'x_max' \(x = 1\).*does not end on, touch or run "
     "along a face: stop it short of the face"),
    (lambda: plane(0.5, 0.0, 0.4 - top_at_surface() - 0.005),
     "comes within .* of interface 'moho'"),
    (lambda: plane(0.5, 0.0, top_at_surface() + 0.005),
     "comes within .* of interface 'surface'"),
    # through the side x = 1, entering the box by 0.005 alone
    (lambda: plane(1.0975, 0.0, 0.6, strike=90.0, length=0.205),
     r"reaches no deeper than 0.005 inside the face 'x_max' \(x = 1\).*touches the "
     "face rather than entering"),
    # a sill lying on the base, whose edges cannot clear it
    (lambda: plane(0.5, 0.0, 1.0, dip=0.0),
     r"meets the face 'base' \(depth = 1\) at too shallow an angle"),
])
def test_what_the_builder_refuses_before_meshing(make, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        build(tmp_path, {"f": make()})
    assert not list(tmp_path.iterdir())


def test_a_fault_cutting_the_box_through_is_refused(tmp_path):
    """A plane from side to side and from above the top to below the base
    cuts the box in two; one from side to side through the top cuts the
    free surface in two: neither is built."""
    through = Plane(LOCAL, [0.5, 0.0, -0.5], strike=0.0, dip=80.0, length=1.4,
                    width=1.4, slip=ZERO)
    with pytest.raises(ValueError, match="layers should stay whole: a fault must not "
                                         "cut a layer right through"):
        build(tmp_path, {"f": through})
    rupture = plane(0.5, 0.0, 0.12, length=1.4, width=0.4)
    with pytest.raises(ValueError, match=r"cut the interface at depth 0 into 2 "
                                         "pieces: a fault cutting an interface"):
        build(tmp_path, {"f": rupture})
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------ AdGIA's box

def adgia(dip=30.0, strike=0.0, scale=1.0 / 8.0):
    """AdGIA's faulted box (`meshes/faulted_box.py`, flat topography) at unit
    scale: its box 8 wide and 4 deep, its fault centred at x = 0 and 1 deep,
    dipping 30 degrees towards +x, 1.5 along dip by 1.5 along strike, all
    divided by 8.  Its `length` runs along dip and `fault_width` along
    strike, a plane's `width` and `length` here."""
    W, D = 8.0 * scale, 4.0 * scale
    geometry = cartesian.Geometry(Skeleton([0.0, D]),
                                  cartesian.Box([(-W / 2, W / 2), (-W / 2, W / 2)]),
                                  layer_names=["body"], interface_names=["surface"])
    fault = Plane(LOCAL, [0.0, 0.0, -1.0 * scale], strike=strike, dip=dip,
                  length=1.5 * scale, width=1.5 * scale, slip=Slip.from_rake(90.0, 1.0))
    return geometry, fault


def adgia_frame(dip, strike):
    """AdGIA's fault frame (`fault_frame` of faulted_box.py): down dip,
    strike, and the normal into the hanging wall, down dip x strike."""
    dip, strike = np.deg2rad(dip), np.deg2rad(strike)
    s = np.array([np.sin(strike), np.cos(strike), 0.0])
    d = np.array([np.cos(dip) * np.cos(strike), -np.cos(dip) * np.sin(strike),
                  -np.sin(dip)])
    return d, s, np.cross(d, s)


def test_adgia_s_faulted_box_at_unit_scale(tmp_path):
    """The fault is AdGIA's: its frame (down dip, strike, normal) and its
    corners; built in, it is buried, pinned all round (four tips), its
    collar on AdGIA's hanging-wall side above the plane, the footwall's
    cells below, and its faces cover its area."""
    geometry, fault = adgia()
    down, strike, normal = adgia_frame(30.0, 0.0)
    basis = fault.fault_basis(0.0, 0.0)
    assert np.allclose(basis[:, 0], strike, rtol=0.0, atol=1e-15)
    assert np.allclose(-basis[:, 1], down, rtol=0.0, atol=1e-15)
    assert np.allclose(basis[:, 2], normal, rtol=0.0, atol=1e-15)
    centre = np.array([0.0, 0.0, -0.125])
    for su, sv in ((-1, -1), (1, -1), (1, 1), (-1, 1)):
        corner = centre + sv * 0.5 * 0.1875 * down + su * 0.5 * 0.1875 * strike
        assert np.allclose(fault.cartesian(0.5 * su, 0.5 * sv), corner, rtol=0.0,
                           atol=1e-15)
    res = build_layered_mesh(
        MeshSpec(geometry, UniformInterfaces(0.08, 0.15, 0.2), order=2,
                 faults=[EmbeddedFault(Fault([fault]), name="fault",
                                       sizing=InterfaceSizing(0.04, 0.15, 0.1))]),
        tmp_path / "adgia")
    assert res.validation.ok, res.validation.failures
    testing.check_fault_mesh(res.msh_path, {"fault": Fault([fault])}, distance=1e-12)
    card = sc.read(res.manifest_path)
    record = card.fault_record("fault")
    assert record["tip"] == 1 and record["free_edge"] is record["face_edge"] is None
    assert record["attribute"] == 1 + 5 + 1 and record["hanging_wall"] == [2]
    mesh = read_mesh(res.msh_path)
    tri = mesh.faces_with(record["attribute"])[:, :3]
    x = mesh.points[tri]
    assert 0.5 * np.linalg.norm(np.cross(x[:, 1] - x[:, 0], x[:, 2] - x[:, 0]),
                                axis=1).sum() == pytest.approx(0.1875 ** 2, rel=1e-12)
    side = (mesh.points[mesh.cells[:, :4]].mean(axis=1) - centre) @ normal
    walls = mesh.walls(card)[0]
    assert np.all(side[walls == 1] > 0.0) and np.all(side[walls == -1] < 0.0)
    assert np.array_equal(mesh.cell_attributes == 2, walls == 1)
    assert res.counts["dim3"] < 5000


# ------------------------------------------------------------ pictures

def test_a_section_through_a_faulted_box_is_drawn(built):
    """A vertical section across the strike of the surface-rupturing fault:
    the cells' pieces tile it, each in its layer or collar's colour, and
    the fault's faces cut there leave its trace, a line at its dip from
    the surface down."""
    matplotlib = pytest.importorskip("matplotlib",
                                     reason="needs the planetmodel[plot] extra")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from planetmodel.plotting import mesh_section, plane_cut

    res = built("surface")
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    origin, axes = [0.0, 0.05, 0.0], [[1, 0, 0], [0, 0, 1]]
    cut = plane_cut(mesh.points, mesh.cells[:, :4], origin, axes)
    s, t = cut.pieces[..., 0], cut.pieces[..., 1]
    areas = 0.5 * np.abs(np.sum(s * np.roll(t, -1, axis=1) - np.roll(s, -1, axis=1) * t,
                                axis=1))
    assert np.isclose(areas.sum(), 1.0, rtol=1e-12)
    trace = plane_cut(mesh.points, mesh.faces_with(card.faults[0]["attribute"])[:, :3],
                      origin, axes)
    p = CASES["surface"][0]["f"]
    ends = trace.pieces.reshape(-1, 2)
    # every point of the trace on the plane's line in the section
    slope = -np.tan(np.deg2rad(50.0))
    x0 = p.cartesian(0.0, -0.5)[0]
    assert np.allclose(ends[:, 1], slope * (ends[:, 0] - x0), rtol=0.0, atol=1e-12)
    assert ends[:, 1].max() == pytest.approx(0.0, abs=1e-12)
    colours = np.array(["#e1e0d9", "#c3c2b7", "#eb6834"])[mesh.cell_attributes - 1]
    fig, ax = plt.subplots()
    polygons = mesh_section(ax, cut, colours=colours, linewidth=0.3)
    lines = mesh_section(ax, trace, colours="#2a78d6", linewidth=1.5)
    assert len(polygons.get_paths()) == len(cut.pieces)
    assert len(lines.get_segments()) == len(trace.pieces)
    plt.close(fig)
