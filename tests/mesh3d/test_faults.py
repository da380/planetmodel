"""Faults built into the layered mesh: pulled back, fragmented in, labelled.

Unit geometries a few elements across.  Every built mesh is held to
`testing.check_fault_mesh`, the contract a consumer relies on: the
faces conform and are oriented footwall to hanging wall, the nodes lie
on the fault once mapped, and the curve groups are the edges of the
faces by the format page's rule.  The gate of the pull-back: under a
radial bump the fault's nodes, mapped, lie on the true fault to the
pull-back's offset, which falls as the control grid is refined, and
with the pull-back off (the fault built in as given) they lie a
visible fraction of an element off it.
"""
import dataclasses
import json

import numpy as np
import pytest

from planetmodel import CallableDisplacement, Geometry, RadialStretch, Skeleton, testing
from planetmodel.displacement import layer_linear
from planetmodel.fault import (CartesianFrame, Fault, GeographicFrame, GreatCircle,
                               Plane, Slip, StructuredPatch)
from planetmodel.mapping import MappingBase
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec, Shell,
                                UniformInterfaces, build_layered_mesh,
                                manifest as sc, read_mesh)
from planetmodel.mesh3d.spec import (GRADIENT, OPTIMISE, OPTIMISE_WITH_FAULTS,
                                     largest_size, optimise_methods)

pytestmark = pytest.mark.gmsh

UNIT = GeographicFrame(1.0)
ZERO = Slip(np.zeros((1, 1, 3)))
#: Element size on the interfaces: the scale every offset is judged against.
H = 0.15
SIZING = UniformInterfaces(H, 0.3, 0.3)


def two_layers(mapping=None):
    """A unit ball of two layers."""
    g = Geometry(Skeleton([0.0, 0.6, 1.0]), layer_names=["core", "mantle"],
                 interface_names=["cmb", "surface"])
    return g if mapping is None else g.with_mapping(mapping)


def bump(amplitude=0.05):
    """A radial relief decaying with depth, a smooth bump over (30N, 40E)
    of 0.05 at the surface: topography on a unit ball."""
    def h(r, theta, phi):
        c = (np.cos(theta) * np.cos(np.deg2rad(60.0)) + np.sin(theta)
             * np.sin(np.deg2rad(60.0)) * np.cos(phi - np.deg2rad(40.0)))
        return amplitude * r ** 2 * np.exp(-(1.0 - c) / 0.05)
    return RadialStretch(CallableDisplacement(h, name="bump"), rmax=1.0)


def plane(lat=30.0, lon=40.0, depth=0.2, **kw):
    kw = {"strike": 20.0, "dip": 50.0, "length": 0.3, "width": 0.2} | kw
    return Plane.at_depth(UNIT, lat, lon, depth, slip=ZERO, **kw)


def reaching(dip=60.0, width=0.3, **kw):
    """A plane whose up-dip edge is at the reference surface."""
    return plane(depth=0.5 * width * np.sin(np.deg2rad(dip)), dip=dip, width=width,
                 **kw)


def curved():
    """A swept patch steepening with depth, wholly inside the mantle."""
    return StructuredPatch.swept(UNIT, GreatCircle([25.0, 30.0], [35.0, 50.0]),
                                 dip=lambda z: 25.0 + 100.0 * z, width=0.3, top=0.05,
                                 n_u=7, n_v=6, slip=ZERO)


def splayed():
    """A main fault and a splay rising from its centre line at 30 degrees."""
    main = plane(strike=0.0, dip=60.0, length=0.4, width=0.3)
    foot = main.cartesian(0.0, -0.1)
    sp = Plane(UNIT, UNIT.coordinates(foot), strike=0.0, dip=30.0, length=0.3,
               width=0.15, slip=ZERO)
    shift = foot - sp.cartesian(0.0, 0.5)
    sp = Plane(UNIT, UNIT.coordinates(sp.cartesian(0.0, 0.0) + shift), strike=0.0,
               dip=30.0, length=0.3, width=0.15, slip=ZERO)
    return main, sp


def build(tmp, faults, *, geometry=None, order=2, **spec):
    """Build a mesh with the faults {name: Fault or EmbeddedFault}."""
    items = [f if isinstance(f, EmbeddedFault) else EmbeddedFault(f, name=n)
             for n, f in faults.items()]
    return build_layered_mesh(MeshSpec(geometry or two_layers(), SIZING,
                                       faults=items, order=order, **spec), tmp)


def held(res, faults, distance=1e-12):
    """The mesh held to the contract, the fault given by name."""
    plain = {n: getattr(f, "fault", f) for n, f in faults.items()}
    off = max((p.offset for v in res.preimages.values() for p in v), default=0.0)
    testing.check_fault_mesh(res.msh_path, plain, mapping=res.mapping,
                             distance=max(distance, 2.0 * off))
    return sc.read(res.manifest_path)


def faces(res, attribute):
    """(node positions, the fault's elements), read as a consumer reads them."""
    mesh = read_mesh(res.msh_path)
    return mesh.points, mesh.faces_with(attribute)


def curve_nodes(res, group):
    """The positions of the nodes of a dimension-1 group."""
    mesh = read_mesh(res.msh_path)
    return mesh.points[np.unique(mesh.curves[group])]


def area(x, tri):
    """The area of linear (3) or quadratic (6-node) triangles of points x,
    by a degree-4 rule."""
    if tri.shape[1] == 3:
        p = x[tri]
        return 0.5 * np.linalg.norm(np.cross(p[:, 1] - p[:, 0], p[:, 2] - p[:, 0]),
                                    axis=1).sum()
    a, b = 0.445948490915965, 0.091576213509771
    pts = [(a, a), (1 - 2 * a, a), (a, 1 - 2 * a), (b, b), (1 - 2 * b, b),
           (b, 1 - 2 * b)]
    w = [0.223381589678011] * 3 + [0.109951743655322] * 3
    p = x[tri]
    total = 0.0
    for (s, t), wq in zip(pts, w):
        ds = np.array([4 * s + 4 * t - 3, 4 * s - 1, 0, 4 * (1 - 2 * s - t),
                       4 * t, -4 * t])
        dt = np.array([4 * s + 4 * t - 3, 0, 4 * t - 1, -4 * s, 4 * s,
                       4 * (1 - s - 2 * t)])
        xs, xt = np.einsum("k,pkj->pj", ds, p), np.einsum("k,pkj->pj", dt, p)
        total += 0.5 * wq * np.linalg.norm(np.cross(xs, xt), axis=1).sum()
    return total


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    """Each case built once: (result, faults by name)."""
    root = tmp_path_factory.mktemp("faults")
    cases = {
        "inside": {"f": Fault([plane()])},
        "crossing": {"f": Fault([plane(depth=0.4, dip=60.0, width=0.35)])},
        "surface": {"f": Fault([reaching()])},
        "two": {"a": Fault([plane()]),
                "b": Fault([plane(-30.0, 200.0, strike=100.0, dip=70.0)])},
        "splay": dict(zip(("main", "splay"), (Fault([p]) for p in splayed()))),
        "curved": {"slab": Fault([curved()])},
    }
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = (build(root / name, cases[name]), cases[name])
        return cache[name]
    return get


# ------------------------------------------------------------ the cases

def test_a_fault_inside_one_layer(built):
    res, faults = built("inside")
    v = res.validation
    assert v.ok and v.misoriented_fault_faces == 0 and v.nonconforming_fault_faces == 0
    assert res.counts["faults"] == 1
    card = held(res, faults)
    (record,) = card.faults
    assert record == {"name": "f", "attribute": 3, "layers": [2],
                      "hanging_wall": [3], "tip": 1, "free_edge": None,
                      "junction": None, "meets": [], "ends_on": [],
                      "controls": None, "offset": 0.0, "sizing": None}
    assert card.schema == sc.SCHEMA_WITH_FAULTS
    assert [e["attributes"] for e in card.layers] == [[1], [2, 3]]
    # the plane's area exactly, by straight triangles: measured 1e-16
    pos, tri = faces(res, 3)
    assert area(pos, tri[:, :3]) == pytest.approx(0.3 * 0.2, rel=1e-12)
    # its tip is its edge: every node on the boundary of the square
    found = faults["f"].locate(curve_nodes(res, 1))
    on_edge = np.maximum(np.abs(found.u), np.abs(found.v))
    assert np.allclose(on_edge, 0.5, atol=1e-12)
    assert np.abs(found.distance).max() < 1e-12


def test_a_fault_crossing_a_layer_boundary_is_one_fault_of_several_faces(built):
    res, faults = built("crossing")
    card = held(res, faults)
    (record,) = card.faults
    assert record["layers"] == [1, 2] and record["tip"] == 1
    mesh = read_mesh(res.msh_path)
    cells = mesh.face_cells(mesh.faces_with(record["attribute"])[:, :3])
    assert set(mesh.layers(card)[cells].ravel().tolist()) == {1, 2}
    # a collar in each layer it crosses
    assert [e["attributes"] for e in card.layers] == [[1, 3], [2, 4]]
    assert record["hanging_wall"] == [3, 4]
    # the seam where it crosses the sphere is no tip: the tip is its edge
    found = faults["f"].locate(curve_nodes(res, record["tip"]))
    assert np.allclose(np.maximum(np.abs(found.u), np.abs(found.v)), 0.5,
                       atol=1e-12)


def test_a_fault_reaching_the_surface_meets_it_along_a_free_edge(built):
    res, faults = built("surface")
    card = held(res, faults)
    (record,) = card.faults
    assert record["free_edge"] == 2 and record["tip"] == 1
    trace = curve_nodes(res, 2)
    assert np.allclose(np.linalg.norm(trace, axis=1), 1.0, rtol=0, atol=1e-12)
    assert np.abs(faults["f"].locate(trace, strict=False).distance).max() < 1e-12
    pos, tri = faces(res, record["attribute"])
    assert np.linalg.norm(pos[np.unique(tri)], axis=1).max() <= 1.0 + 1e-12


def test_two_separate_faults_have_their_own_attributes(built):
    res, faults = built("two")
    card = held(res, faults)
    a, b = card.faults
    assert (a["name"], a["attribute"], a["tip"]) == ("a", 3, 1)
    assert (b["name"], b["attribute"], b["tip"]) == ("b", 4, 4)
    assert a["junction"] is None and b["junction"] is None
    assert a["meets"] == b["meets"] == []
    names = {t: n for (d, t), n in read_mesh(res.msh_path).names.items()
             if d in (1, 2)}
    assert names[3] == "a" and names[4] == "b"


def test_a_splay_meets_its_main_fault_along_a_common_curve(built):
    res, faults = built("splay")
    card = held(res, faults)
    main, splay = card.faults
    assert main["meets"] == ["splay"] and splay["meets"] == ["main"]
    assert (main["junction"], splay["junction"]) == (3, 6)
    # the junction is one curve, on both faults
    for record in (main, splay):
        x = curve_nodes(res, record["junction"])
        for fault in faults.values():
            assert np.abs(fault.locate(x, strict=False).distance).max() < 1e-12
    assert np.allclose(np.sort(curve_nodes(res, 3), axis=0),
                       np.sort(curve_nodes(res, 6), axis=0))


def test_two_faults_crossing_each_other(tmp_path):
    a = plane(strike=0.0, dip=60.0, length=0.3, width=0.3)
    b = Plane(UNIT, a.centroid, strike=90.0, dip=60.0, length=0.3, width=0.3,
              slip=ZERO)
    faults = {"a": Fault([a]), "b": Fault([b])}
    res = build(tmp_path / "x", faults)
    card = held(res, faults)
    assert [f["meets"] for f in card.faults] == [["b"], ["a"]]
    # the curve where they cross is in the junction group of each
    first, second = (curve_nodes(res, f["junction"]) for f in card.faults)
    assert first.size and np.allclose(np.sort(first, axis=0), np.sort(second, axis=0))


def test_a_fault_of_two_patches_is_one_surface(tmp_path):
    a = plane(strike=0.0, dip=60.0, length=0.2)
    shift = a.cartesian(0.5, 0.0) - a.cartesian(-0.5, 0.0)
    b = Plane(UNIT, UNIT.coordinates(a.cartesian(0.0, 0.0) + shift), strike=0.0,
              dip=60.0, length=0.2, width=0.2, slip=ZERO)
    faults = {"seg": Fault([a, b])}
    card = held(build(tmp_path / "seg", faults), faults)
    (record,) = card.faults
    assert record["tip"] == 1 and record["junction"] is None


def test_a_curved_fault_is_built_in_exactly(built):
    res, faults = built("curved")
    card = held(res, faults)
    pos, tri = faces(res, card.faults[0]["attribute"])
    true = faults["slab"].quadrature(16, 16, points=4).weights.sum()
    # quadratic triangles on the B-spline: measured 3.6e-6 against the
    # straight triangles' 5.4e-3
    assert area(pos, tri) == pytest.approx(true, rel=5e-5)
    assert abs(area(pos, tri[:, :3]) / true - 1.0) > 1e-3


def test_a_bilinear_patch_is_built_in_exactly(tmp_path):
    """A degree-one B-spline, creased between its cells."""
    patch = StructuredPatch.swept(UNIT, GreatCircle([25.0, 30.0], [35.0, 50.0]),
                                  dip=lambda z: 25.0 + 100.0 * z, width=0.3,
                                  top=0.05, n_u=5, n_v=4, slip=ZERO,
                                  interpolation="linear")
    assert patch.spline().degree == (1, 1)
    faults = {"f": Fault([patch])}
    card = held(build(tmp_path / "bilinear", faults), faults)
    assert card.faults[0]["tip"] == 1


def test_a_fault_through_a_buffer_shell_is_trimmed_at_the_surface(tmp_path):
    faults = {"trace": Fault([reaching()])}
    res = build(tmp_path / "shell", faults, shells=[Shell(ratio=0.3)])
    card = held(res, faults)
    (record,) = card.faults
    assert record["attribute"] == 4 and record["layers"] == [2]
    assert record["free_edge"] == 2
    # trimmed at the model's surface, not the shell's: nothing beyond it
    pos, tri = faces(res, 4)
    assert np.linalg.norm(pos[np.unique(tri)], axis=1).max() <= 1.0 + 1e-12
    assert res.validation.min_sicn > 0.1


# ------------------------------------------------------------ the gate

def gate(res, fault):
    """The largest distance of the fault's nodes, mapped, from the true
    fault, over the element size; and the area's relative error.  Every
    node is located inside the fault but for those of its tip, which
    are located on its edge."""
    record = sc.read(res.manifest_path).faults[0]
    pos, tri = faces(res, record["attribute"])
    x = res.mapping(pos)
    nodes = np.unique(tri)
    found = fault.locate(x[nodes], strict=False)
    tip = curve_nodes(res, record["tip"])
    on_tip = np.isin(np.round(pos[nodes], 12), np.round(tip, 12)).all(axis=1)
    edge = np.maximum(np.abs(found.u), np.abs(found.v))[on_tip]
    distance = float(np.abs(found.distance).max())
    assert np.all(found.inside[~on_tip])
    assert np.allclose(edge, 0.5, rtol=0, atol=max(1e-12, 10 * distance))
    true = fault.quadrature(16, 16, points=4).weights.sum()
    return distance / H, area(x, tri) / true - 1.0


def test_the_gate_the_pull_back_puts_the_mapped_fault_on_the_true_one(tmp_path):
    """Under a bump of 0.05 the nodes, mapped, lie on the plane to the
    pull-back's offset, converging with the control grid; with the
    pull-back off (built under the identity, then mapped) they lie a
    quarter of an element off it."""
    m = bump()
    fault = Fault([plane()])
    errors = []
    for controls in (5, 9, 17):
        res = build(tmp_path / f"c{controls}", {"f": EmbeddedFault(
            fault, name="f", controls=controls)}, geometry=two_layers(m))
        held(res, {"f": fault})
        record = sc.read(res.manifest_path).faults[0]
        assert record["controls"] == [[controls, controls]]
        distance, area_error = gate(res, fault)
        assert distance * H <= record["offset"] * (1 + 1e-9)
        assert abs(area_error) < 100 * record["offset"]
        errors.append(distance)
    # measured: 1.4e-4, 2.0e-6, 1.1e-7 of an element
    assert errors[0] < 1e-3 and errors[2] < 1e-6
    assert errors[0] / errors[1] > 8 and errors[1] / errors[2] > 8
    off = build(tmp_path / "off", {"f": fault})
    distance, _ = gate(dataclasses.replace(off, mapping=m), fault)
    # measured: 0.24 of an element
    assert distance > 0.1


def test_the_gate_on_a_curved_fault_with_the_default_controls(tmp_path):
    m = bump()
    fault = Fault([curved()])
    res = build(tmp_path / "curved", {"f": fault}, geometry=two_layers(m))
    card = held(res, {"f": fault})
    (record,) = card.faults
    assert record["controls"] == [[17, 17]] and 0.0 < record["offset"] < 1e-6
    distance, area_error = gate(res, fault)
    # measured: 3.3e-7 of an element; area 2.7e-5 by quadratic triangles
    assert distance < 1e-5 and abs(area_error) < 1e-4


def test_under_topography_a_surface_fault_reaches_the_moved_surface(tmp_path):
    """The bump lifts the surface over the fault by up to 0.04: its edge,
    given at the reference radius, is carried across and trimmed, so the
    mesh's fault meets the physical surface exactly."""
    m = bump()
    fault = Fault([reaching()])
    res = build(tmp_path / "bumped", {"f": fault}, geometry=two_layers(m))
    card = held(res, {"f": fault})
    (record,) = card.faults
    assert record["free_edge"] == 2
    # on the reference surface, to the accuracy of OCC's crossing of a
    # B-spline with a sphere (measured 1e-10; a plane's is exact)
    trace = curve_nodes(res, 2)
    r = np.linalg.norm(trace, axis=1)
    assert np.abs(r - 1.0).max() < 1e-7
    # so mapped onto the physical surface, lifted above the fault's own
    # edge, which the bump leaves short of it
    theta, phi = _angles(trace).T
    lifted = np.linalg.norm(m(trace / r[:, None]), axis=1)
    assert np.allclose(lifted, 1.0 + m.h(1.0, theta, phi), rtol=0, atol=1e-14)
    assert np.abs(np.linalg.norm(m(trace), axis=1) - lifted).max() < 1e-7
    assert lifted.max() > 1.02
    # the fault's own up-dip edge stops short of that surface over the
    # bump (measured by 0.044 at its middle): the carrying closed the gap
    edge = fault.patches[0].cartesian(np.linspace(-0.5, 0.5, 9), -0.5)
    theta, phi = _angles(edge).T
    short = 1.0 + m.h(1.0, theta, phi) - np.linalg.norm(edge, axis=1)
    assert short.max() > 0.03


def _angles(x):
    r = np.linalg.norm(x, axis=1)
    return np.stack([np.arccos(x[:, 2] / r), np.arctan2(x[:, 1], x[:, 0])], axis=1)


def _over(theta, phi, width):
    """A bump over (30N, 40E) on the sphere, one at its top."""
    c = (np.cos(theta) * np.cos(np.deg2rad(60.0)) + np.sin(theta)
         * np.sin(np.deg2rad(60.0)) * np.cos(phi - np.deg2rad(40.0)))
    return np.exp(-(1.0 - c) / width)


def stacked():
    """Topography as a stack of reliefs, each layer stretched linearly
    between its own two: a bump of 0.05 on the surface over (30N, 40E)
    and a root of 0.04 on the CMB under it, the centre held, so that the
    stretch's radial derivative jumps at the CMB (a knot)."""
    return RadialStretch(layer_linear(
        Skeleton([0.0, 0.6, 1.0]),
        [None, lambda t, p: -0.04 * _over(t, p, 0.08),
         lambda t, p: 0.05 * _over(t, p, 0.05)]), rmax=1.0)


def test_a_fault_across_a_knot_of_the_stretch_is_built_in_pieces(tmp_path):
    """Across the CMB, where the stack's slope jumps, the preimage of a
    plane is creased along the crossing: it is built in two pieces, one in
    each layer, at the default rtol, joined along the crossing.  One fault
    in the mesh, a collar in each layer, its tip the plane's edge alone,
    the join inside it; mapped, its nodes lie on the plane on both sides,
    the join's on the CMB's relief, and the slip read at them is the
    plane's at the node's own place."""
    m = stacked()
    values = np.zeros((2, 2, 3))
    values[..., 0] = [[0.0, 0.0], [1.0, 1.0]]
    values[..., 1] = [[0.0, 2.0], [0.0, 2.0]]
    patch = Plane.at_depth(UNIT, 30.0, 40.0, 0.4, strike=20.0, dip=60.0, length=0.3,
                           width=0.35, slip=Slip(values, rule="linear"))
    fault = Fault([patch])
    res = build(tmp_path / "pieces", {"f": fault}, geometry=two_layers(m))
    assert len(res.preimages["f"]) == 2
    card = held(res, {"f": fault})
    (record,) = card.faults
    # built whole, it was refused at this rtol: 4.7e-5 off with 129 by 129
    # control nodes; in pieces, 4.9e-9 and 2.2e-9 off, through 33 nodes
    # down dip and 65 along strike, which lay their join on the CMB to
    # 6e-10 (with 33, 9e-9)
    assert record["controls"] == [[65, 33]]
    assert record["layers"] == [1, 2] and record["hanging_wall"] == [3, 4]
    assert [e["attributes"] for e in card.layers] == [[1, 3], [2, 4]]
    assert record["junction"] is None and record["free_edge"] is None
    mesh = read_mesh(res.msh_path)
    tri = mesh.faces_with(record["attribute"])
    layer = mesh.layers(card)[mesh.face_cells(tri[:, :3])[:, 0]]
    sides = [np.unique(tri[layer == k]) for k in (1, 2)]
    join = np.intersect1d(*sides)
    tip = np.unique(mesh.curves[record["tip"]])
    for nodes in sides:
        found = fault.locate(res.mapping(mesh.points[nodes]))
        # measured: 1.5e-10 above the CMB, 3.1e-10 below it
        assert np.abs(found.distance).max() <= 2.0 * record["offset"]
    # the join meets the tip at its two ends alone: no edge of the tip
    # runs along it
    assert join.size > 5 and np.count_nonzero(np.isin(join, tip)) == 2
    assert not np.any(np.all(np.isin(mesh.curves[record["tip"]][:, :2], join),
                             axis=1))
    x = res.mapping(mesh.points[join])
    theta, phi = _angles(x).T
    cmb = 0.6 + m.h(0.6, theta, phi)
    # measured: 3.2e-11 off the CMB's relief
    assert np.abs(np.linalg.norm(x, axis=1) - cmb).max() < 1e-8
    found = fault.locate(x)
    assert np.all(found.inside) and np.abs(found.v).max() < 0.4
    on = fault.locate(res.mapping(mesh.points[tip]))
    assert np.allclose(np.maximum(np.abs(on.u), np.abs(on.v)), 0.5, atol=1e-6)
    origin, (xu, xv) = patch.cartesian(0.0, 0.0), patch.tangents(0.0, 0.0)
    for nodes in (*sides, join):
        x = res.mapping(mesh.points[nodes])
        u, v = np.clip([(x - origin) @ xu / (xu @ xu),
                        (x - origin) @ xv / (xv @ xv)], -0.5, 0.5)
        want = np.stack([0.5 + u, 1.0 + 2.0 * v, 0.0 * u], axis=-1)
        got = fault.slip_at(x, max_distance=1e-6, basis="fault")
        assert np.allclose(got, want, rtol=0, atol=1e-9)


def test_pieces_whose_join_would_stray_off_the_knot_are_refused(tmp_path):
    """Through 17 control nodes along u, given, the edge where the pieces
    join lies 1.1e-7 off the CMB between them: beyond the CAD kernel's
    tolerance, which would cut the pieces where they pass through the
    CMB beside it."""
    fault = Fault([plane(depth=0.4, dip=60.0, width=0.35)])
    with pytest.raises(ValueError, match="their join lies 1.05e-07 off it, beyond "
                                         "1e-09.*give more control nodes along u, "
                                         "or none"):
        build(tmp_path / "coarse", {"f": EmbeddedFault(fault, name="f", controls=17)},
              geometry=two_layers(stacked()))


def test_a_curved_fault_s_edge_is_carried_as_the_patch_continued(tmp_path):
    """A swept patch steepening with depth whose up-dip edge is at the
    reference surface, under a bump of 0.08 there: its edge is carried
    across the surface by more than a third of its width, the part
    carried the preimage of the patch's own continuation past its edge
    (its bicubic's end pieces carried on), so that the mesh holds to the
    contract there, which measures the carried nodes against that
    continuation, as everywhere."""
    patch = StructuredPatch.swept(UNIT, GreatCircle([25.0, 35.0], [35.0, 45.0]),
                                  dip=lambda z: 45.0 + 100.0 * z, width=0.3, top=0.0,
                                  n_u=7, n_v=6, slip=ZERO)
    fault = Fault([patch])
    res = build(tmp_path / "carried", {"f": fault}, geometry=two_layers(bump(0.08)))
    # measured: the carried nodes 3.6e-9 at most off the patch continued
    # (with the preimage itself extrapolated past the edge, refused, its
    # edge not clear of the surface within the 0.34 of its width tried;
    # tried to half its width, 3.9e-4 off against the contract's 2.3e-4)
    card = held(res, {"f": fault})
    (record,) = card.faults
    assert record["free_edge"] == 2 and record["controls"] == [[33, 33]]
    pos, tri = faces(res, record["attribute"])
    x = res.mapping(pos[np.unique(tri)])
    found = fault.locate(x, strict=False)
    # carried 0.113 beyond the edge, more than 0.34 of the width of 0.3
    assert np.abs(found.distance[~found.inside]).max() > 0.34 * 0.3
    trace = curve_nodes(res, 2)
    assert np.abs(np.linalg.norm(trace, axis=1) - 1.0).max() < 1e-7


# ------------------------------------------------------------ orders and sizes

@pytest.mark.parametrize("order", [1, 3])
def test_other_orders_put_every_node_on_the_fault(order, tmp_path):
    m = bump()
    fault = Fault([curved()])
    res = build(tmp_path / f"o{order}", {"f": fault}, geometry=two_layers(m),
                order=order)
    held(res, {"f": fault})
    pos, tri = faces(res, 3)
    assert tri.shape[1] == {1: 3, 3: 10}[order]


def test_a_finer_mesh_is_asked_for_near_the_fault(tmp_path):
    fault = Fault([plane()])

    def mean_edge(res):
        pos, tri = faces(res, 3)
        p = pos[tri[:, :3]]
        return np.mean(np.linalg.norm(p - np.roll(p, 1, axis=1), axis=2))

    plain = build(tmp_path / "plain", {"f": fault}, order=1)
    sizing = InterfaceSizing(0.04, 0.3, 0.2)
    fine = build(tmp_path / "fine", {"f": EmbeddedFault(fault, name="f",
                                                        sizing=sizing)}, order=1)
    held(fine, {"f": fault})
    assert mean_edge(fine) < 0.5 * mean_edge(plain)
    assert mean_edge(fine) < 1.5 * 0.04
    assert sc.read(fine.manifest_path).faults[0]["sizing"] == {
        "size": 0.04, "far_size": 0.3, "decay_width": 0.2}


def test_a_fault_sized_by_a_number_is_graded_and_relocated(tmp_path):
    """The rule a fault is sized by (`EmbeddedFault`): a number is the size
    on it, graded at `GRADIENT` to the largest size the interfaces' rule
    gives (SIZING's far size, 0.3), and the linear mesh is relocated by
    Relocate3D; the mesh is the one that sizing and that optimisation
    spelled out give, and the manifest records the sizing resolved."""
    fault = Fault([plane()])

    def spec(sizing, **options):
        return MeshSpec(two_layers(), SIZING, order=1, **options,
                        faults=[EmbeddedFault(fault, name="f", sizing=sizing)])

    by_rule = spec(0.06)
    assert by_rule.optimise == OPTIMISE_WITH_FAULTS == ("Relocate3D",)
    res = build_layered_mesh(by_rule, tmp_path / "rule")
    held(res, {"f": fault})
    assert res.validation.ok and res.validation.warnings == []
    graded = InterfaceSizing.graded(0.06, 0.3)
    assert sc.read(res.manifest_path).faults[0]["sizing"] == {
        "size": 0.06, "far_size": 0.3, "decay_width": graded.decay_width}
    spelt = build_layered_mesh(spec(graded, optimise="Relocate3D"), tmp_path / "spelt")
    a, b = read_mesh(res.msh_path), read_mesh(spelt.msh_path)
    assert np.array_equal(a.cells, b.cells) and np.array_equal(a.faces, b.faces)
    assert np.allclose(a.points, b.points, rtol=0.0, atol=1e-12)


def test_the_relocation_moves_no_node_of_a_fault_or_an_interface(tmp_path):
    """The steep ramp without the relocation is the caller's to ask for
    (an InterfaceSizing as it is, `optimise=None`); a Netgen pass named
    for a mesh with a fault is skipped with a warning, leaving that same
    mesh; and the relocation the default runs moves nodes inside the
    layers only, the elements and every node of the faces, the fault's
    and the interfaces', as they were, the worst element no worse."""
    fault = Fault([plane()])

    def built(label, **options):
        return build_layered_mesh(
            MeshSpec(two_layers(), SIZING, order=1, **options,
                     faults=[EmbeddedFault(fault, name="f",
                                           sizing=InterfaceSizing(0.06, 0.3, 0.1))]),
            tmp_path / label)

    steep, netgen, moved = built("steep", optimise=None), built(
        "netgen", optimise="Netgen"), built("moved")
    assert steep.validation.warnings == [] and moved.validation.warnings == []
    assert netgen.validation.warnings == [
        "the Netgen optimisation was skipped: it crashes on a solid with a fault "
        "embedded in it"]
    held(moved, {"f": fault})
    a, b, c = (read_mesh(r.msh_path) for r in (steep, netgen, moved))
    assert np.array_equal(a.points, b.points) and np.array_equal(a.cells, b.cells)
    assert np.array_equal(a.cells, c.cells) and np.array_equal(a.faces, c.faces)
    on_faces = np.unique(a.faces[:, :3])
    assert np.array_equal(a.points[on_faces], c.points[on_faces])
    assert np.any(a.points != c.points)
    assert moved.validation.min_sicn >= steep.validation.min_sicn


# ------------------------------------------------------------ the record

def test_a_mesh_without_faults_has_the_manifest_it_always_had(tmp_path):
    res = build_layered_mesh(MeshSpec(two_layers(), SIZING), tmp_path / "plain")
    data = json.loads(res.manifest_path.read_text())
    assert "faults" not in data and data["schema"] == sc.SCHEMA
    assert all("attributes" not in e for e in data["layers"])
    card = sc.read(res.manifest_path)
    assert card.faults == [] and card.element_attributes() == (1, 2)
    assert "faults" not in res.counts and "collars" not in res.counts
    assert res.preimages == {}
    assert set(read_mesh(res.msh_path).cell_attributes.tolist()) == {1, 2}


def test_the_manifest_describes_the_faults(built):
    res, _ = built("splay")
    card = sc.read(res.manifest_path)
    text = card.describe()
    assert "faults" in text and "meets splay" in text and "junction 3" in text
    assert "collars [3, 4, 5]" in text and "hanging wall [3, 4]" in text
    assert card.fault_record("splay")["attribute"] == 4
    with pytest.raises(KeyError, match="no fault named"):
        card.fault_record("nope")
    assert "2 faults" in repr(card)


# ------------------------------------------------------------ refusals

def test_what_the_spec_refuses():
    fault = Fault([plane()])
    local = Fault([Plane(CartesianFrame(), [0.0, 0.0, -1.0], strike=0.0, dip=45.0,
                         length=1.0, width=1.0, slip=ZERO)])
    with pytest.raises(ValueError, match="Cartesian frame belongs in a regional"):
        MeshSpec(two_layers(), SIZING, faults=[EmbeddedFault(local)])
    with pytest.raises(TypeError, match="Fault"):
        EmbeddedFault(plane())
    with pytest.raises(ValueError, match="2D mesh takes none"):
        MeshSpec(two_layers(), SIZING, dimension=2, faults=[EmbeddedFault(fault)])
    with pytest.raises(ValueError, match="unique"):
        MeshSpec(two_layers(), SIZING, faults=[EmbeddedFault(fault, name="x"),
                                               EmbeddedFault(fault, name="x")])
    with pytest.raises(ValueError, match="share a name with an interface"):
        MeshSpec(two_layers(), SIZING, faults=[EmbeddedFault(fault, name="cmb")])
    with pytest.raises(TypeError, match="EmbeddedFault"):
        MeshSpec(two_layers(), SIZING, faults=[fault])
    with pytest.raises(ValueError, match="at least 4"):
        EmbeddedFault(fault, controls=3)
    with pytest.raises(ValueError, match="reach"):
        EmbeddedFault(fault, reach=-1.0)
    for bad in (True, "0.1", (0.1, 0.3)):
        with pytest.raises(TypeError, match="sizing must be a size"):
            EmbeddedFault(fault, sizing=bad)
    for bad in (0.0, -0.1, np.inf, np.nan):
        with pytest.raises(ValueError, match="positive and finite"):
            EmbeddedFault(fault, sizing=bad)
    assert MeshSpec(two_layers(), SIZING, faults=[EmbeddedFault(fault)]
                    ).fault_names == ["fault_1"]


def test_a_fault_s_sizing_resolves_by_the_rule():
    """A number is graded to the mesh's largest size (`largest_size`, the
    interfaces' largest far size), held flat where it is not below it;
    an InterfaceSizing is taken as it is; None is none.  A spec with a
    fault takes Relocate3D for "auto", one without Netgen, and either
    takes what it names."""
    fault = Fault([plane()])
    sizes = SIZING(two_layers().interfaces, 1.0)
    assert largest_size(sizes) == 0.3
    assert largest_size({0: InterfaceSizing(0.1, 0.2, 0.1),
                         1: InterfaceSizing(0.05, 0.4, 0.1)}) == 0.4
    with pytest.raises(ValueError, match="no sizing given"):
        largest_size({})
    by_rule = EmbeddedFault(fault, sizing=np.float64(0.06))
    assert type(by_rule.sizing) is float
    assert by_rule.resolved_sizing(0.3) == InterfaceSizing.graded(0.06, 0.3)
    assert by_rule.resolved_sizing(0.3).decay_width == pytest.approx(0.24 / GRADIENT)
    assert EmbeddedFault(fault, sizing=0.5).resolved_sizing(0.3) == \
        InterfaceSizing(0.5, 0.5, 0.5)
    steep = InterfaceSizing(0.06, 0.3, 0.1)
    assert EmbeddedFault(fault, sizing=steep).resolved_sizing(0.3) is steep
    assert EmbeddedFault(fault).resolved_sizing(0.3) is None
    with_fault = MeshSpec(two_layers(), SIZING, faults=[EmbeddedFault(fault)])
    assert with_fault.optimise == OPTIMISE_WITH_FAULTS
    assert MeshSpec(two_layers(), SIZING).optimise == OPTIMISE == ("Netgen",)
    assert MeshSpec(two_layers(), SIZING, optimise="auto").optimise == OPTIMISE
    for named, held_as in ((None, ()), ("Netgen", ("Netgen",)),
                           (["Relocate3D", "Netgen"], ("Relocate3D", "Netgen"))):
        assert MeshSpec(two_layers(), SIZING, optimise=named,
                        faults=[EmbeddedFault(fault)]).optimise == held_as
    assert optimise_methods("auto", faults=True) == OPTIMISE_WITH_FAULTS
    assert optimise_methods("auto") == OPTIMISE


#: The depth of the centroid of `plane()` whose up-dip edge is `d` below
#: the reference surface at its middle (its ends, a straight line's, are
#: shallower by about 0.011).
def _top_at(d):
    return 0.1 * np.sin(np.deg2rad(50.0)) + d


@pytest.mark.parametrize("make,match", [
    (lambda: plane(depth=-0.6), "wholly outside the geometry"),
    (lambda: plane(depth=1.0, length=0.4, width=0.4), "passes through the centre"),
    (lambda: plane(depth=0.4 - 0.1 * np.sin(np.deg2rad(50.0)) - 0.005),
     "comes within .* of interface 'cmb'"),
    (lambda: plane(depth=_top_at(0.02)), "comes within .* of interface 'surface'"),
    # its ends poke out of the surface, its middle runs 0.005 under it
    (lambda: plane(depth=_top_at(0.005), length=0.4),
     "comes within .* of interface 'surface'"),
    # it crosses the core-mantle boundary, the middle of its up-dip edge
    # runs 0.005 above it
    (lambda: plane(depth=0.4 + 0.1 * np.sin(np.deg2rad(50.0)) - 0.005, strike=0.0,
                   dip=50.0, length=0.6),
     "comes within .* of interface 'cmb'"),
    (lambda: reaching(dip=0.0, width=0.2), "touches the surface"),
])
def test_what_the_builder_refuses_before_meshing(make, match, tmp_path):
    with pytest.raises(ValueError, match=match):
        build(tmp_path / "no", {"f": Fault([make()])})
    assert not list(tmp_path.iterdir())


def test_a_fault_ending_on_an_interface_is_refused_with_the_way_round(tmp_path):
    """A plane whose down-dip edge bottoms out at the core-mantle boundary
    (at its middle; a straight edge's ends lie above it) is refused, the
    refusal saying to stop it short of the interface or carry it past."""
    from scipy.optimize import brentq

    depth = brentq(lambda d: np.linalg.norm(plane(depth=d).cartesian(0.0, 0.5)) - 0.6,
                   0.2, 0.4, xtol=1e-14)
    on = plane(depth=depth)
    with pytest.raises(ValueError, match=(
            r"interface 'cmb'.*does not end on, touch or run along an interface: "
            r"stop it short of the interface by more than \S+, or continue it "
            r"past the interface by as much")):
        build(tmp_path / "no", {"f": Fault([on])})
    assert not list(tmp_path.iterdir())


def test_an_edge_just_below_the_surface_is_carried_across_with_a_reach(tmp_path):
    shallow = Fault([plane(depth=_top_at(0.02))])
    res = build(tmp_path / "reach", {"f": EmbeddedFault(shallow, name="f",
                                                        reach=0.03)})
    card = held(res, {"f": shallow})
    assert card.faults[0]["free_edge"] == 2


def test_a_fault_into_the_hole_of_a_hollow_domain_is_refused(tmp_path):
    hollow = Geometry(Skeleton([0.5, 0.8, 1.0]))
    with pytest.raises(ValueError, match="below the inner boundary"):
        build(tmp_path / "hole", {"f": Fault([plane(depth=0.4, dip=80.0,
                                                    width=0.3)])},
              geometry=hollow)


def test_overlapping_faults_are_refused(tmp_path):
    fault = Fault([plane()])
    with pytest.raises(ValueError, match="overlap on a surface"):
        build(tmp_path / "twice", {"a": fault, "b": fault})


def test_a_fault_cutting_a_layer_through_is_refused(tmp_path):
    """A wide plane 0.3 from the centre cuts the core in two."""
    wide = Plane.at_depth(UNIT, 0.0, 0.0, 0.7, strike=0.0, dip=0.0, length=1.2,
                          width=1.2, slip=ZERO)
    with pytest.raises(ValueError, match="right through"):
        build(tmp_path / "cut", {"f": Fault([wide])})


def test_a_mapping_without_an_inverse_is_refused(tmp_path):
    class Grow(MappingBase):
        def __call__(self, X):
            return 1.01 * np.asarray(X, dtype=float)

        def deformation_gradient(self, X):
            X = np.asarray(X, dtype=float)
            return np.broadcast_to(1.01 * np.eye(3), X.shape[:-1] + (3, 3)).copy()

        def jacobian(self, X):
            return np.full(np.shape(X)[:-1], 1.01 ** 3)

    with pytest.raises(ValueError, match="does not invert itself"):
        build(tmp_path / "grow", {"f": Fault([plane()])},
              geometry=two_layers(Grow()))


# ------------------------------------------------------------ the format

def test_the_manifest_s_fault_records_are_checked():
    good = {"name": "f", "attribute": 3, "layers": [2], "hanging_wall": [3],
            "tip": 1, "free_edge": None, "junction": None, "meets": [],
            "ends_on": [], "controls": None, "offset": 0.0, "sizing": None}

    def card(*records, attributes=([1], [2, 3])):
        g = two_layers()
        b = g.skeleton.boundaries
        layers = [sc.LayerEntry.from_layer(lay, attribute=i + 1, r_inner=b[i],
                                           r_outer=b[i + 1], in_geometry=True)
                  for i, lay in enumerate(g.layers)]
        faces_ = [sc.InterfaceEntry.from_interface(f, attribute=k + 1,
                                                   radius=f.radius)
                  for k, f in enumerate(g.interfaces)]
        out = sc.MeshManifest.from_build(
            mesh=sc.mesh_block("m.msh", format="msh", nodes="reference"),
            layers=layers, interfaces=faces_)
        out.faults = [dict(r) for r in records]
        out.schema = sc.SCHEMA_WITH_FAULTS
        for e, a in zip(out.layers, attributes):
            e["attributes"] = list(a)
        return out

    sc.validate_structure(card(good))
    junction = [good | {"name": "a", "junction": 3, "meets": ["b"]},
                good | {"name": "b", "attribute": 4, "tip": 4, "junction": 6,
                        "meets": ["a"], "ends_on": ["a"], "hanging_wall": [3, 4]}]
    both = {"attributes": ([1], [2, 3, 4])}
    sc.validate_structure(card(*junction, **both))
    for bad, match in [
            (good | {"attribute": 2}, "attribute is 2, expected 3"),
            (good | {"layers": [3]}, "not attributes of layers"),
            (good | {"tip": 2}, "tip is 2, expected 1"),
            (good | {"name": "cmb"}, "share a name with an interface"),
            (good | {"meets": ["g"]}, "not the names of other faults"),
            (good | {"junction": 3}, "a junction is where it meets"),
            (good | {"ends_on": ["g"]}, "ends_on is \\['g'\\], not faults it meets"),
            (good | {"hanging_wall": [2]}, "hanging_wall is \\[2\\], not collar"),
            (good | {"hanging_wall": []}, "collar 3, which is in no fault's"),
            (good | {"controls": [[3, 9]]}, "controls"),
            (good | {"offset": -1.0}, "offset"),
            (good | {"sizing": {"size": 2.0, "far_size": 1.0,
                                "decay_width": 1.0}}, "sizing"),
            (good | {"layers": "2"}, "faults\\[0\\].layers")]:
        with pytest.raises(ValueError, match=match):
            sc.validate_structure(card(bad))
    for attributes, match in [
            (([1], [3, 2]), "not its own attribute 2"),
            (([1, 3], [2, 3]), "which layers\\[0\\] lists too"),
            (([1], [2, 4]), "not 3..3"),
            (([1], [2]), "hanging_wall is \\[3\\], not collar")]:
        with pytest.raises(ValueError, match=match):
            sc.validate_structure(card(good, attributes=attributes))
    # the collars numbered out of the order of layer and faults
    swapped = [junction[0] | {"hanging_wall": [4]},
               junction[1] | {"hanging_wall": [3, 4]}]
    with pytest.raises(ValueError, match="out of the order of layer and faults"):
        sc.validate_structure(card(*swapped, **both))
    with pytest.raises(ValueError, match="not every one of them meets"):
        sc.validate_structure(card(junction[0], junction[1] | {
            "meets": [], "junction": None, "ends_on": []}, **both))
    with pytest.raises(ValueError, match="manifest lists 1 faults, the mesh has 0"):
        sc.validate_against(card(good), layer_count=2, interface_count=2)
    # the schema says whether there are faults
    five = card(good)
    five.schema = sc.SCHEMA
    with pytest.raises(ValueError, match="faults are listed under schema"):
        sc.validate_structure(five)
    six = card()
    with pytest.raises(ValueError, match="has faults"):
        sc.validate_structure(six)


# ------------------------------------------------------------ the rules

#: A face in the plane z = 0, its normal along +z, and a corner above and
#: below it.
CORNERS = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
                    [0.2, 0.2, 1.0], [0.2, 0.2, -1.0], [0.5, 0.5, 0.0]])


def test_the_plus_cell_is_on_the_side_the_face_s_vertex_order_points_to():
    up, down = [0, 1, 2, 3], [1, 4, 2, 0]
    assert sc.fault_side(CORNERS, [0, 1, 2], [up, down]).tolist() == [1, -1]
    assert sc.fault_side(CORNERS, [0, 2, 1], [up, down]).tolist() == [-1, 1]
    faces = np.array([[0, 1, 2], [0, 2, 1]])
    cells = np.array([[down, up], [down, up]])
    assert sc.plus_cell(CORNERS, faces, cells).tolist() == [1, 0]
    with pytest.raises(ValueError, match="do not hold their face's three corners"):
        sc.fault_side(CORNERS, [0, 1, 2], [0, 1, 3, 4])
    with pytest.raises(ValueError, match="in their face's plane"):
        sc.fault_side(CORNERS, [0, 1, 2], [0, 1, 2, 5])
    with pytest.raises(ValueError, match="both their cells on one side"):
        sc.plus_cell(CORNERS, faces[:1], np.array([[up, up]]))
    with pytest.raises(ValueError, match="outside"):
        sc.fault_side(CORNERS, [0, 1, 2], [0, 1, 2, 6])


def test_the_curves_are_read_from_the_faces():
    """A square of two faces whose edge (0, 1) is on the surface, a second
    fault of one face meeting it along (2, 3), both ending there, and a
    third of two faces with (2, 3) inside it, which the square ends on and
    which ends on neither."""
    a, b, c = sc.fault_curves([[[0, 1, 2], [1, 3, 2]], [[3, 2, 4]],
                               [[3, 2, 5], [2, 3, 6]]], [[0, 1, 9]])
    assert a.free_edge.tolist() == [[0, 1]]
    assert a.tip.tolist() == [[0, 2], [1, 3]]
    assert a.junction.tolist() == a.junction_end.tolist() == [[2, 3]]
    assert a.ends_on == (1, 2)
    assert a.pinned.tolist() == [[0, 2], [1, 3], [2, 3]]
    assert b.free_edge.shape == (0, 2) and b.junction.tolist() == [[2, 3]]
    assert b.tip.tolist() == [[2, 4], [3, 4]] and b.ends_on == (0, 2)
    assert c.junction.tolist() == [[2, 3]] and c.junction_end.shape == (0, 2)
    assert c.ends_on == () and c.pinned.tolist() == c.tip.tolist()
    (alone,) = sc.fault_curves([np.array([[5, 4, 6]])], np.empty((0, 3), int))
    assert alone.tip.tolist() == [[4, 5], [4, 6], [5, 6]]
    with pytest.raises(ValueError, match="corner indices"):
        sc.fault_curves([[[0.0, 1.0, 2.0]]], [[0, 1, 9]])


def test_the_manifest_names_the_geometry_s_surface(built):
    res, _ = built("splay")
    assert sc.read(res.manifest_path).surface_attribute == 2


def test_the_check_catches_a_fault_other_than_the_one_built(built):
    """A fault longer than the one built leaves the mesh's tip inside it;
    one shorter leaves nodes beyond its edges, which a fault without a
    free edge cannot account for, and a fault with one only beyond that
    edge: the check refuses each."""
    res, _ = built("inside")
    for length, match in ((0.301, "a node of its tip lies"),
                          (0.299, "no free edge")):
        with pytest.raises(AssertionError, match=match):
            testing.check_fault_mesh(res.msh_path, {"f": Fault([plane(length=length)])},
                                     distance=1e-9)
    res, _ = built("surface")
    with pytest.raises(AssertionError, match="not its free edge"):
        testing.check_fault_mesh(res.msh_path, {"f": Fault([reaching(length=0.29)])},
                                 distance=1e-9)
    # the same plane shorter up dip alone (a bilinear patch through its
    # corners), where the fault was carried across the surface: the nodes
    # there lie on it continued, and pass
    U, V = np.meshgrid([-0.5, 0.5], [-0.5 + 0.02 / 0.3, 0.5], indexing="ij")
    shorter = StructuredPatch(UNIT, UNIT.coordinates(reaching().cartesian(U, V)),
                              slip=ZERO, interpolation="linear")
    testing.check_fault_mesh(res.msh_path, {"f": Fault([shorter])}, distance=1e-9)


def test_a_preimage_near_an_interface_is_refused_saying_so(tmp_path):
    """Under a uniform swell of 0.02 r^2 a plane whose up-dip edge stands
    0.0123 above the reference core-mantle boundary pulls back to 0.005
    above it: the refusal names the preimage, the reference radius and
    the clearance once mapped (0.0051, the swell moving the boundary
    too)."""
    from scipy.optimize import brentq

    swell = RadialStretch(CallableDisplacement(lambda r, t, p: 0.02 * r ** 2,
                                               name="swell"), rmax=1.0)
    target = 0.605 + 0.02 * 0.605 ** 2

    def top(depth):
        return np.linalg.norm(plane(depth=depth).cartesian(0.0, -0.5)) - target

    depth = brentq(top, 0.3, 0.5, xtol=1e-14)
    with pytest.raises(ValueError, match=(
            r"has a preimage under the geometry's mapping.* comes within 0.005 of "
            r"interface 'cmb' at the reference radius 0.6 \(once mapped, 0.00512 ")):
        build(tmp_path / "no", {"f": Fault([plane(depth=depth)])},
              geometry=two_layers(swell))
