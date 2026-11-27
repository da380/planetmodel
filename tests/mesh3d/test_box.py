"""Flat-layered meshes: a box cut by the planes of its interfaces, and its
cross-section in the (x, z) plane, each a few elements across.

Every cell lies in the layer its attribute says and every boundary
element on the plane its group says, facing out of the layer below it or
out of the body, at every element order; the manifest (schema 7)
validates, round-trips and refuses what breaks its rules; the sizing
measures distance from the planes; and what a box cannot take is refused
by name."""
import copy
import dataclasses
import json

import numpy as np
import pytest

from planetmodel import Skeleton, cartesian
from planetmodel.fault import Fault, GeographicFrame, Plane, Slip
from planetmodel.mapping import MappingBase
from planetmodel.mesh3d import (AngularResolution, CappedInterfaces, EmbeddedFault,
                                InterfaceSizing, MeshSpec, PerInterface, Shell,
                                UniformInterfaces, build_layered_mesh, manifest as sc,
                                read_mesh)
from planetmodel.mesh3d._session import session
from planetmodel.mesh3d._writer import read_groups

from conftest import COARSE, full_geometry

pytestmark = pytest.mark.gmsh

DEPTHS = [0.0, 0.3, 0.6, 1.0]
LAYERS = ["upper", "middle", "lower"]
INTERFACES = ["surface", "top_middle", "top_lower"]
#: Coarse sizing: a few elements across a unit box.
SIZING = UniformInterfaces(0.25, 0.4, 0.3)


def box(**kw):
    """Three layers of a unit box, x in [0, 1] and y in [-0.5, 0.5]."""
    return cartesian.Geometry(Skeleton(DEPTHS),
                              cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                              layer_names=LAYERS, interface_names=INTERFACES, **kw)


def rectangle(**kw):
    """The same three layers over x in [-1, 1]: a rectangle, invariant along y."""
    return cartesian.Geometry(Skeleton(DEPTHS), cartesian.Rectangle((-1.0, 1.0)),
                              layer_names=LAYERS, interface_names=INTERFACES, **kw)


GEOMETRIES = {"box": box, "rectangle": rectangle}
ORDERS = (1, 2, 3)


class Stretch(MappingBase):
    """A vertical stretch, z -> 1.2 z: no identity, and the same at every y."""

    def __call__(self, X):
        return np.asarray(X, dtype=float) * [1.0, 1.0, 1.2]

    def deformation_gradient(self, X):
        X = np.asarray(X, dtype=float)
        return np.broadcast_to(np.diag([1.0, 1.0, 1.2]), X.shape[:-1] + (3, 3)).copy()

    def jacobian(self, X):
        return np.full(np.shape(X)[:-1], 1.2)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each geometry built once per order."""
    root = tmp_path_factory.mktemp("flat")
    cache = {}

    def _built(kind, order=2):
        if (kind, order) not in cache:
            spec = MeshSpec(GEOMETRIES[kind](), SIZING, order=order)
            cache[(kind, order)] = build_layered_mesh(spec, root / f"{kind}_{order}")
        return cache[(kind, order)]

    return _built


def physical(mesh, card):
    """The physical points (x, y, z), (n, 3), the mesh's nodes stand for."""
    X = np.zeros((len(mesh.points), 3))
    X[:, list(card.mesh_axes(mesh.dimension))] = mesh.points[:, :mesh.dimension]
    return X


def planes(card):
    """Every boundary group's plane: attribute -> (physical axis, value,
    outward sign), the interfaces facing up."""
    out = {e["attribute"]: (2, -e["depth"], 1) for e in card.interfaces}
    out |= {e["attribute"]: (sc.AXES.index(e["axis"]), e["value"], e["outward"])
            for e in card.faces}
    return out


def normals(mesh, faces):
    """The normal of each boundary element in the mesh's coordinates, (m, d):
    (v1 - v0) x (v2 - v0) for a triangle, the direction turned clockwise
    for a line element."""
    p = mesh.points
    if mesh.dimension == 3:
        return np.cross(p[faces[:, 1]] - p[faces[:, 0]],
                        p[faces[:, 2]] - p[faces[:, 0]])
    t = p[faces[:, 1]] - p[faces[:, 0]]
    return np.stack([t[:, 1], -t[:, 0]], axis=1)


# ---------------------------------------------------------------- building

@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_a_box_and_a_rectangle_build_checked_at_every_order(built, kind, order):
    res = built(kind, order)
    v = res.validation
    assert v.ok, v.failures
    assert v.negative_jacobians == 0 and v.negative_cells == 0 and v.min_sicn > 0.0
    assert v.inward_faces == 0
    assert v.max_interface_radius_error < 1e-12
    d = {"box": 3, "rectangle": 2}[kind]
    assert res.spec.dimension == d and res.spec.order == order
    assert res.counts["layers"] == 3 and res.counts["interfaces"] == 3
    assert res.counts["faces"] == {"box": 5, "rectangle": 3}[kind]
    assert v.group_counts == {"layers": 3, "interfaces": 3,
                              "faces": res.counts["faces"]}
    assert res.mapping.is_identity and res.geometry is res.spec.geometry
    # a few elements across, as every meshing test is
    assert 10 < res.counts["elements"] < 3000


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_every_cell_lies_in_the_layer_its_attribute_says(built, kind):
    res = built(kind)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    assert sorted(set(mesh.cell_attributes.tolist())) == [1, 2, 3]
    depth = -physical(mesh, card)[:, 2]
    for a in (1, 2, 3):
        lo, hi = card.layer_interval(card.layers[a - 1])
        nodes = depth[mesh.cells[mesh.cell_attributes == a]]
        assert nodes.min() >= lo - 1e-12 and nodes.max() <= hi + 1e-12
    # the arrays' own rule agrees, by position alone
    assert np.array_equal(mesh.layers_by_position(card), mesh.cell_attributes)
    # a rectangle's mesh lies in the (x, z) plane, as its coordinates say
    if kind == "rectangle":
        assert card.coordinates == ["x", "z"] and np.all(mesh.points[:, 2] == 0.0)
        assert mesh.points[:, 1].min() == pytest.approx(-1.0, abs=1e-12)
        assert mesh.points[:, 1].max() == pytest.approx(0.0, abs=1e-12)


@pytest.mark.parametrize("order", ORDERS)
@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_every_boundary_element_lies_on_the_plane_its_group_says_facing_out(
        built, kind, order):
    res = built(kind, order)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    d = mesh.dimension
    X = physical(mesh, card)
    where = planes(card)
    assert sorted(set(mesh.face_attributes.tolist())) == sorted(where)
    assert len(where) == len(card.interfaces) + len(card.faces)
    corners = mesh.faces[:, :d]
    # by position: each element's corners lie on exactly one of the planes,
    # the one of its own group; its every node too
    on = np.stack([np.all(np.abs(X[corners, axis] - value) <= 1e-12, axis=1)
                   for axis, value, _ in where.values()], axis=1)
    assert np.all(on.sum(axis=1) == 1)
    assert np.array_equal(np.array(list(where))[np.argmax(on, axis=1)],
                          mesh.face_attributes)
    for a, (axis, value, outward) in where.items():
        nodes = mesh.faces_with(a)
        assert np.abs(X[nodes, axis] - value).max() <= 1e-12
        # facing up out of the layer below, or out of the body
        n = normals(mesh, nodes)
        column = card.mesh_axes(d).index(axis)
        assert np.all(outward * n[:, column] > 0.0), a
    # an interface has its two layers either side, a face of the body one cell
    for e in card.interfaces:
        cells = mesh.face_cells(mesh.faces_with(e["attribute"]))
        below, above = e["between_layers"]
        if above < 0:
            assert np.all(cells[:, 1] == -1)
            assert np.all(mesh.cell_attributes[cells[:, 0]] == below + 1)
        else:
            got = np.sort(mesh.cell_attributes[cells], axis=1)
            assert np.all(got == [above + 1, below + 1])
    for e in card.faces:
        cells = mesh.face_cells(mesh.faces_with(e["attribute"]))
        assert np.all(cells[:, 0] >= 0) and np.all(cells[:, 1] == -1)
        layers = set(mesh.cell_attributes[cells[:, 0]].tolist())
        assert layers == ({3} if e["name"] == "base" else {1, 2, 3})


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_mesh_file_names_its_groups(built, kind):
    res = built(kind)
    header = res.mesh_path.read_text(errors="ignore").splitlines()[:2]
    assert header[1].split()[0] == "2.2"
    with session(name="reread"):
        groups = read_groups(res.mesh_path)
    d = res.spec.dimension
    assert groups[d] == {1: "upper", 2: "middle", 3: "lower"}
    faces = {"box": ["base", "x_min", "x_max", "y_min", "y_max"],
             "rectangle": ["base", "x_min", "x_max"]}[kind]
    assert groups[d - 1] == dict(enumerate(INTERFACES + faces, start=1))


# ---------------------------------------------------------------- the manifest

@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_flat_manifest_describes_the_box_and_round_trips(built, kind, tmp_path):
    res = built(kind)
    data = json.loads(res.manifest_path.read_text())
    assert data["schema"] == sc.SCHEMA_FLAT == "planetmodel.mesh.manifest/7"
    assert "faults" not in data
    card = sc.read(res.manifest_path)
    assert card.is_flat and card.coordinate == "depth"
    assert card.coordinates == {"box": ["x", "y", "z"], "rectangle": ["x", "z"]}[kind]
    assert [(e["name"], e["depth_top"], e["depth_bottom"], e["in_geometry"])
            for e in card.layers] == [("upper", 0.0, 0.3, True),
                                      ("middle", 0.3, 0.6, True),
                                      ("lower", 0.6, 1.0, True)]
    assert [(e["attribute"], e["name"], e["depth"], e["between_layers"])
            for e in card.interfaces] == [(1, "surface", 0.0, [0, -1]),
                                          (2, "top_middle", 0.3, [1, 0]),
                                          (3, "top_lower", 0.6, [2, 1])]
    want = {"box": [(4, "base", "z", -1.0, -1), (5, "x_min", "x", 0.0, -1),
                    (6, "x_max", "x", 1.0, 1), (7, "y_min", "y", -0.5, -1),
                    (8, "y_max", "y", 0.5, 1)],
            "rectangle": [(4, "base", "z", -1.0, -1), (5, "x_min", "x", -1.0, -1),
                        (6, "x_max", "x", 1.0, 1)]}[kind]
    assert [(e["attribute"], e["name"], e["axis"], e["value"], e["outward"])
            for e in card.faces] == want
    assert np.array_equal(card.boundaries, DEPTHS)
    assert card.surface_attribute == 1 and card.face_attribute("base") == 4
    assert card.interface_attribute("top_lower") == 3
    assert card.scale == 1.0 and card.element_attributes() == (1, 2, 3)
    with pytest.raises(ValueError, match="no outer radius"):
        card.outer_radius
    with pytest.raises(KeyError, match="no face named 'top'"):
        card.face_attribute("top")
    text = card.describe()
    assert "coordinates" in text and "depth 0.3" in text and "faces" in text
    again = sc.write(tmp_path / "again", card)
    assert again.read_text() == res.manifest_path.read_text()
    assert sc.read(again) == card
    sc.validate_against(card, layer_count=3, interface_count=3,
                        groups={"layers": [1, 2, 3], "interfaces": [1, 2, 3],
                                "faces": list(range(4, 4 + len(want)))})
    with pytest.raises(ValueError, match="faces physical groups"):
        sc.validate_against(card, layer_count=3, interface_count=3,
                            groups={"faces": [5, 6]})


def broken(card, change):
    """A copy of `card` with `change` applied to it."""
    bad = copy.deepcopy(card)
    change(bad)
    return bad


FLAT_BREAKAGES = [
    (lambda c: c.faces[0].__setitem__("value", -0.9), "the base lies at"),
    (lambda c: c.faces.reverse(), "faces are"),
    (lambda c: c.faces[1].__setitem__("attribute", 9), r"faces\[1\].attribute"),
    (lambda c: c.faces[1].__setitem__("outward", 1), r"faces\[1\] \(x_min\)"),
    (lambda c: (c.faces[1].__setitem__("value", 2.0)), "x_min and x_max"),
    (lambda c: setattr(c, "coordinates", ["y", "z"]), "coordinates is"),
    (lambda c: c.interfaces[1].__setitem__("between_layers", [0, 1]),
     r"interfaces\[1\].between_layers"),
    (lambda c: c.interfaces[2].__setitem__("depth", 0.5), "not at the top"),
    (lambda c: c.layers[1].__setitem__("depth_top", 0.35), "layer above ends"),
    (lambda c: c.layers[0].__setitem__("in_geometry", False), "every layer"),
    (lambda c: c.interfaces.pop(), "one per layer"),
    (lambda c: c.interfaces[1].__setitem__("name", "base"), "share a name"),
    (lambda c: c.layers[0].__setitem__("r_inner", 0.0) or
     c.layers[0].pop("depth_top"), r"layers\[0\].depth_top"),
    (lambda c: setattr(c, "faults", [dataclasses.asdict(
        sc.FaultEntry(name="f", attribute=9, layers=[1]))]),
     r"layers\[0\].attributes is None"),
    (lambda c: c.layers[1].__setitem__("attributes", [2]),
     r"faults lists none, and layers\[1\].attributes is listed"),
    (lambda c: setattr(c, "schema", sc.SCHEMA), r"layers\[0\].r_inner"),
]


@pytest.mark.parametrize("change,match", FLAT_BREAKAGES)
def test_validate_structure_holds_a_flat_manifest_to_its_rules(built, change, match):
    card = sc.read(built("box").manifest_path)
    sc.validate_structure(card)
    with pytest.raises(ValueError, match=f"malformed manifest: .*{match}"):
        sc.validate_structure(broken(card, change))


def test_a_spherical_manifest_has_no_flat_keys(tmp_path):
    """A sphere's manifest is schema 5 as it was: no coordinates, no faces;
    and the flat keys are its no more when present."""
    res = build_layered_mesh(MeshSpec(full_geometry(), COARSE, dimension=2, order=1),
                             tmp_path / "disc")
    data = json.loads(res.manifest_path.read_text())
    assert data["schema"] == sc.SCHEMA
    assert "coordinates" not in data and "faces" not in data
    card = sc.read(res.manifest_path)
    assert not card.is_flat and card.coordinate == "radius" and card.faces == []
    assert card.mesh_axes(2) == (0, 1) and card.scale == card.outer_radius == 1.0
    stray = tmp_path / "stray.json"
    stray.write_text(json.dumps(data | {"faces": [{"name": "base"}],
                                        "coordinates": ["x", "z"]}))
    assert sc.read(stray) == card
    with pytest.raises(ValueError, match="coordinates and faces are a flat mesh's"):
        sc.validate_structure(broken(card, lambda c: setattr(c, "coordinates",
                                                              ["x", "y", "z"])))


# ---------------------------------------------------------------- sizing

@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_the_sizing_measures_distance_from_the_planes(kind, tmp_path):
    """A fine surface and a coarse rest: the elements on the surface are the
    surface's size, those on the base the far size, each as the ramp gives."""
    rule = PerInterface({"surface": InterfaceSizing(0.1, 0.35, 0.15)},
                        base=UniformInterfaces(0.35, 0.35, 0.3))
    res = build_layered_mesh(MeshSpec(GEOMETRIES[kind](), rule, order=1),
                             tmp_path / kind)
    assert res.validation.ok
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    d = mesh.dimension

    def edges(attribute):
        f = mesh.faces_with(attribute)[:, :d]
        p = mesh.points
        pairs = [(0, 1)] if d == 2 else [(0, 1), (1, 2), (2, 0)]
        return np.concatenate([np.linalg.norm(p[f[:, i]] - p[f[:, j]], axis=1)
                               for i, j in pairs])

    top, base = edges(1), edges(card.face_attribute("base"))
    assert 0.075 < np.median(top) < 0.14
    assert np.median(base) > 2.0 * np.median(top)


def test_the_thin_layer_check_judges_the_lowest_layer_by_its_top(tmp_path):
    """The base is no interface: the layer above it is judged by its top
    alone, as the centre's layer is on the sphere."""
    thin_base = cartesian.Geometry(Skeleton([0.0, 0.5, 0.52]),
                                   cartesian.Rectangle((0.0, 1.0)))
    with pytest.raises(ValueError, match=r"layer \[0.5, 0.52\] .* too coarse"):
        build_layered_mesh(MeshSpec(thin_base, SIZING), tmp_path / "thin")
    thin_top = cartesian.Geometry(Skeleton([0.0, 0.01, 1.0]),
                                  cartesian.Rectangle((0.0, 1.0)))
    with pytest.raises(ValueError, match=r"layer \[0, 0.01\] .* too coarse"):
        build_layered_mesh(MeshSpec(thin_top, SIZING), tmp_path / "thin_top")
    assert not list(tmp_path.iterdir())


# ---------------------------------------------------------------- refusals

def test_a_request_the_box_cannot_meet_is_refused_by_name():
    g = box()
    with pytest.raises(ValueError, match="takes no shells"):
        MeshSpec(g, SIZING, shells=[Shell(ratio=0.2)])
    with pytest.raises(ValueError, match="no spherical outer boundary"):
        MeshSpec(g, SIZING, outer_boundary="spherical")
    fault = Fault([Plane.at_depth(GeographicFrame(1.0), 0.0, 0.0, 0.2, strike=0.0,
                                  dip=45.0, length=0.2, width=0.1,
                                  slip=Slip(np.zeros((1, 1, 3))))])
    with pytest.raises(ValueError, match="a flat geometry over a Box takes faults "
                                         "in a CartesianFrame"):
        MeshSpec(g, SIZING, faults=[EmbeddedFault(fault)])
    with pytest.raises(ValueError, match="dimension must be 3 for a flat geometry "
                                         "over a Box"):
        MeshSpec(g, SIZING, dimension=2)
    with pytest.raises(ValueError, match="dimension must be 2 for a flat geometry "
                                         "over a Rectangle"):
        MeshSpec(rectangle(), SIZING, dimension=3)
    named = cartesian.Geometry(Skeleton(DEPTHS), cartesian.Rectangle((0.0, 1.0)),
                               interface_names=["top", "base", None])
    with pytest.raises(ValueError, match=r"named \['base'\] would share a name"):
        MeshSpec(named, SIZING)
    for rule in (AngularResolution(0.1, 0.3), CappedInterfaces(0.1, 0.3, 0.3)):
        with pytest.raises(ValueError, match="lies at 0, where .* scales with radius"):
            rule(g.interfaces, 1.0)
    with pytest.raises(TypeError, match="must be a Geometry"):
        MeshSpec(Skeleton(DEPTHS), SIZING)
    # the spec's defaults: the geometry's own dimension, nothing else asked
    assert MeshSpec(g, SIZING).dimension == 3
    assert MeshSpec(rectangle(), SIZING).dimension == 2
    assert MeshSpec(full_geometry(), COARSE).dimension == 3


# ---------------------------------------------------------------- geometries

def test_a_hollow_box_and_a_padded_domain_build(tmp_path):
    hollow = rectangle().hollowed(0.1)
    res = build_layered_mesh(MeshSpec(hollow, SIZING, order=1), tmp_path / "hollow")
    card = sc.read(res.manifest_path)
    assert [e["depth"] for e in card.interfaces] == [0.1, 0.3, 0.6]
    assert card.layers[0]["depth_top"] == 0.1 and card.surface_attribute == 1
    assert read_mesh(res.mesh_path).points[:, 1].max() == pytest.approx(-0.1, abs=1e-12)
    padded = cartesian.Geometry(Skeleton(DEPTHS),
                                cartesian.Rectangle((-1.0, 1.0), padding=0.25))
    res = build_layered_mesh(MeshSpec(padded, SIZING, order=1), tmp_path / "padded")
    card = sc.read(res.manifest_path)
    assert [e["value"] for e in card.faces] == [-1.0, -1.25, 1.25]
    x = read_mesh(res.mesh_path).points[:, 0]
    assert (x.min(), x.max()) == pytest.approx((-1.25, 1.25), abs=1e-12)


@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_a_mapping_is_carried_and_the_nodes_stay_on_the_reference_planes(kind,
                                                                         tmp_path):
    g = GEOMETRIES[kind](mapping=Stretch())
    res = build_layered_mesh(MeshSpec(g, SIZING, order=2), tmp_path / kind)
    assert res.validation.ok and res.mapping is g.mapping
    card = sc.read(res.manifest_path)
    assert card.mesh["nodes"] == "reference"
    depth = -physical(read_mesh(res.mesh_path), card)[:, 2]
    assert (depth.min(), depth.max()) == pytest.approx((0.0, 1.0), abs=1e-12)


# ---------------------------------------------------------------- pictures

@pytest.mark.parametrize("kind", list(GEOMETRIES))
def test_a_box_and_a_rectangle_are_drawn_by_layer(built, kind):
    """A vertical section of the box, and the rectangle's own mesh, tile the
    body's section exactly, each piece in its layer's colour."""
    matplotlib = pytest.importorskip("matplotlib",
                                     reason="needs the planetmodel[plot] extra")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from planetmodel.plotting import in_plane, mesh_section, plane_cut

    res = built(kind)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    if kind == "box":
        cut = plane_cut(mesh.points, mesh.cells[:, :4], [0.0, 0.1, 0.0],
                        [[1, 0, 0], [0, 0, 1]])
        rim = plane_cut(mesh.points, mesh.faces[:, :3], [0.0, 0.1, 0.0],
                        [[1, 0, 0], [0, 0, 1]])
        area, width = 1.0, 1.0
    else:
        cut = in_plane(mesh.points, mesh.cells[:, :3], [0.0, 0.0, 0.0],
                       [[1, 0, 0], [0, 1, 0]])
        rim = in_plane(mesh.points, mesh.faces[:, :2], [0.0, 0.0, 0.0],
                       [[1, 0, 0], [0, 1, 0]])
        area, width = 2.0, 2.0
    s, t = cut.pieces[..., 0], cut.pieces[..., 1]
    areas = 0.5 * np.abs(np.sum(s * np.roll(t, -1, axis=1) - np.roll(s, -1, axis=1) * t,
                                axis=1))
    assert np.isclose(areas.sum(), area, rtol=1e-12)
    # each piece within its cell's layer
    depth = -t.reshape(len(t), -1)
    for a in (1, 2, 3):
        lo, hi = card.layer_interval(card.layers[a - 1])
        inside = depth[mesh.cell_attributes[cut.index] == a]
        assert inside.min() >= lo - 1e-12 and inside.max() <= hi + 1e-12
    # the boundary curves: the three interfaces and the base across, the
    # sides down, around the body
    lengths = np.linalg.norm(rim.pieces[:, 1] - rim.pieces[:, 0], axis=1)
    assert np.isclose(lengths.sum(), 4 * width + 2 * 1.0, rtol=1e-12)
    colours = np.array(["#2a78d6", "#eb6834", "#1baf7a"])[mesh.cell_attributes - 1]
    fig, ax = plt.subplots()
    polygons = mesh_section(ax, cut, colours=colours, linewidth=0.3)
    lines = mesh_section(ax, rim, colours="#0b0b0b", linewidth=1.0)
    assert len(polygons.get_paths()) == len(cut.pieces)
    assert len(lines.get_segments()) == len(rim.pieces)
    plt.close(fig)
