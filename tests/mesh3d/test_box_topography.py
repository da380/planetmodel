"""A box under topography: the reference box meshed, its faults pulled back.

Unit boxes a few elements across, under a vertical stretch: a range
raised on the free surface, a root on the Moho, the base held.  The mesh
is the reference box, its nodes on the reference planes, and the mapping
moves them at the export: mapped, the surface's nodes lie on the
topography, the Moho's on its relief, the base's where they were, the
sides' on their planes.  A fault is given where it physically is and
built in through its preimage under the stretch (`fault.pull_back`,
generic: the stretch's inverse is all it asks): the gate of the
pull-back, the fault's nodes mapped forward on the true fault, measured
and converging with the control grid, and visibly off with the
pull-back left out; a fault whose edge is at the reference surface under
a mountain is carried across it, its trace on the topography; one cut
by the topography likewise, its nodes all on the fault; one leaving
through a side keeps its face edge there, the side being moved by
nothing along x; one stopping short of a side is not carried across it,
however high the topography above.  AdGIA's faulted box, its range
topography and its fault, is reproduced at unit scale in planetmodel's
reference-then-deform form.
"""
import dataclasses

import numpy as np
import pytest

from planetmodel import Skeleton, cartesian, testing
from planetmodel.fault import CartesianFrame, Fault, Plane, Slip
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec,
                                UniformInterfaces, build_layered_mesh, manifest as sc,
                                read_mesh)

pytestmark = pytest.mark.gmsh

LOCAL = CartesianFrame()
ZERO = Slip(np.zeros((1, 1, 3)))
DEPTHS = [0.0, 0.4, 1.0]
#: Element size on the interfaces: a few elements across the unit box.
H = 0.15
SIZING = UniformInterfaces(H, 0.3, 0.3)


def ridge(x, y):
    """A range along y over x = 0.6, higher towards y = 0, with a shallow
    basin in front of it: heights, positive up, of 0.1 at most."""
    return (0.08 * np.exp(-((x - 0.6) / 0.15) ** 2) * (1.0 + 0.25 * np.cos(np.pi * y))
            - 0.02 * np.exp(-((x - 0.3) / 0.1) ** 2))


def root(x, y):
    """The Moho's relief under the range: deeper there."""
    return -0.04 * np.exp(-((x - 0.6) / 0.2) ** 2)


def box(**kw):
    """Two layers of a unit box, x in [0, 1] and y in [-0.5, 0.5]."""
    return cartesian.Geometry(Skeleton(DEPTHS),
                              cartesian.Box([(0.0, 1.0), (-0.5, 0.5)]),
                              layer_names=["crust", "mantle"],
                              interface_names=["surface", "moho"], **kw)


def topography(surface=ridge, moho=root):
    """The box under the stretch of the surface's and the Moho's reliefs,
    the base held."""
    g = box()
    return g.stretched(cartesian.layer_linear(g.skeleton, [surface, moho, None]))


def plane(x, y, depth, **kw):
    """A plane of centroid (x, y) at the reference depth `depth`, striking
    along y and dipping 50 degrees towards +x by default."""
    kw = {"strike": 0.0, "dip": 50.0, "length": 0.4, "width": 0.25} | kw
    return Plane.at_depth(LOCAL, x, y, depth, slip=ZERO, **kw)


def build(tmp, faults, *, geometry=None, order=2, **spec):
    """Build a mesh with the faults {name: Plane or EmbeddedFault}."""
    items = [f if isinstance(f, EmbeddedFault) else EmbeddedFault(Fault([f]), name=n)
             for n, f in faults.items()]
    return build_layered_mesh(MeshSpec(geometry or topography(), SIZING, faults=items,
                                       order=order, **spec), tmp)


def held(res, faults):
    """The mesh held to the fault contract, its nodes mapped, to twice the
    pull-back's offset; the manifest."""
    plain = {n: f.fault if isinstance(f, EmbeddedFault) else Fault([f])
             for n, f in faults.items()}
    off = max((p.offset for v in res.preimages.values() for p in v), default=0.0)
    testing.check_fault_mesh(res.mesh_path, plain, mapping=res.mapping,
                             distance=max(1e-12, 2.0 * off))
    return sc.read(res.manifest_path)


def gate(res, fault):
    """The largest distance of the fault's nodes, mapped, from the true
    fault, over the element size; every node but those of its tip
    located inside it."""
    record = sc.read(res.manifest_path).faults[0]
    mesh = read_mesh(res.mesh_path)
    nodes = np.unique(mesh.faces_with(record["attribute"]))
    x = res.mapping(mesh.points[nodes])
    found = fault.locate(x, strict=False)
    tip = np.unique(mesh.curves[record["tip"]])
    assert np.all(found.inside[~np.isin(nodes, tip)])
    return float(np.abs(found.distance).max()) / H


def mapped_nodes(res, attribute):
    """The physical positions of the nodes of a boundary group."""
    mesh = read_mesh(res.mesh_path)
    return res.mapping(mesh.points[np.unique(mesh.faces_with(attribute))])


# ------------------------------------------------------------ the deformed box

def test_the_reference_box_is_meshed_and_the_mapping_moves_it_onto_the_relief(
        tmp_path):
    """The mesh is the reference box, its manifest the reference planes and
    nodes "reference"; mapped, the surface's nodes lie on the topography
    and the Moho's on its relief to rounding, the base's stay, the sides'
    keep their planes."""
    g = topography()
    res = build_layered_mesh(MeshSpec(g, SIZING, order=2), tmp_path / "box")
    assert res.validation.ok and res.mapping is g.mapping
    card = sc.read(res.manifest_path)
    assert card.mesh["nodes"] == "reference" and not card.faults
    assert [e["depth"] for e in card.interfaces] == [0.0, 0.4]
    assert [e["value"] for e in card.faces] == [-1.0, 0.0, 1.0, -0.5, 0.5]
    mesh = read_mesh(res.mesh_path)
    for k, relief in enumerate((ridge, root)):
        ref = mesh.points[np.unique(mesh.faces_with(card.interfaces[k]["attribute"]))]
        assert np.abs(ref[:, 2] + DEPTHS[k]).max() < 1e-12
        x = res.mapping(ref)
        assert np.abs(x[:, 2] - (relief(x[:, 0], x[:, 1]) - DEPTHS[k])).max() < 1e-12
    surface = mapped_nodes(res, card.interfaces[0]["attribute"])
    assert surface[:, 2].max() > 0.09 and surface[:, 2].min() < -0.015
    for face in card.faces:
        x = mapped_nodes(res, face["attribute"])
        axis = sc.AXES.index(face["axis"])
        assert np.abs(x[:, axis] - face["value"]).max() < 1e-12


# ------------------------------------------------------------ the gate

def test_the_gate_the_pull_back_puts_the_mapped_fault_on_the_true_one(tmp_path):
    """A fault buried in the crust under the range: its nodes, mapped, lie
    on the plane to the pull-back's offset, converging with the control
    grid as a cubic's, the fourth power of the spacing; built in as given
    and then mapped, a sizeable fraction of an element off it."""
    fault = plane(0.6, 0.1, 0.25, dip=40.0, length=0.5, width=0.25)
    errors = []
    for controls in (5, 9, 17):
        res = build(tmp_path / f"c{controls}", {"f": EmbeddedFault(
            Fault([fault]), name="f", controls=controls)})
        card = held(res, {"f": fault})
        (record,) = card.faults
        assert record["controls"] == [[controls, controls]]
        assert record["tip"] is not None and record["free_edge"] is None
        assert record["layers"] == [1]
        distance = gate(res, Fault([fault]))
        assert distance * H <= 1.5 * record["offset"]
        errors.append(distance)
    # measured: 6.9e-5, 2.8e-6, 1.3e-7 of an element
    assert errors[0] < 5e-4 and errors[2] < 1e-6
    assert errors[0] / errors[1] > 10 and errors[1] / errors[2] > 10
    off = build(tmp_path / "off", {"f": fault}, geometry=box())
    # measured: 0.16 of an element
    assert gate(dataclasses.replace(off, mapping=topography().mapping),
                Fault([fault])) > 0.1


def test_with_the_default_controls_the_offset_meets_rtol(tmp_path):
    fault = plane(0.6, 0.1, 0.25, dip=40.0, length=0.5, width=0.25)
    res = build(tmp_path / "default", {"f": fault})
    (record,) = held(res, {"f": fault}).faults
    assert record["controls"] == [[17, 17]]
    assert 0.0 < record["offset"] <= 1e-6 * np.hypot(0.5, 0.25)
    # measured: 1.3e-7 of an element
    assert gate(res, Fault([fault])) < 1e-6


def across_the_moho():
    """A plane across the Moho under the range, its slip varying over it:
    strike-slip 0.5 + u and dip-slip 1 + 2 v, bilinear."""
    values = np.zeros((2, 2, 3))
    values[..., 0] = [[0.0, 0.0], [1.0, 1.0]]
    values[..., 1] = [[0.0, 2.0], [0.0, 2.0]]
    return Plane.at_depth(LOCAL, 0.6, 0.1, 0.4, strike=0.0, dip=40.0, length=0.5,
                          width=0.4, slip=Slip(values, rule="linear"))


def by_side(res, record):
    """The fault's nodes on the faces in each layer, and the nodes the
    two layers' faces share: where its pieces join."""
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    tri = mesh.faces_with(record["attribute"])
    layer = mesh.layers(card)[mesh.face_cells(tri[:, :3])[:, 0]]
    sides = [np.unique(tri[layer == k]) for k in record["layers"]]
    return mesh, sides, np.intersect1d(*sides)


def test_a_fault_across_a_knot_of_the_stretch_is_built_in_pieces(tmp_path):
    """Across the Moho, where the stack's slope jumps (a knot), the
    preimage of a plane is creased along the crossing: it is built in two
    pieces, one in each layer, cut along the crossing and joined there, at
    the default rtol, through one grid.  The mesh's fault is one fault:
    one attribute, a collar in each layer as any fault crossing an
    interface has, its tip the plane's edge alone, the join inside it and
    in no curve group.  Mapped, its nodes lie on the plane on both sides,
    the join's on the Moho's relief; the slip a consumer reads at them is
    the plane's at the node's own place."""
    patch = across_the_moho()
    fault = Fault([patch])
    res = build(tmp_path / "pieces", {"f": EmbeddedFault(fault, name="f")})
    assert len(res.preimages["f"]) == 2
    card = held(res, {"f": patch})
    (record,) = card.faults
    # built whole, it was refused at this rtol: 3.5e-5 off with 129 by 129
    # control nodes; in pieces: 1.7e-7 and 9.8e-9 off with 17 by 17
    assert record["controls"] == [[17, 17]]
    assert record["offset"] <= 1e-6 * np.hypot(0.5, 0.4)
    assert record["layers"] == [1, 2] and record["hanging_wall"] == [3, 4]
    assert [e["attributes"] for e in card.layers] == [[1, 3], [2, 4]]
    assert record["junction"] is None and record["free_edge"] is None
    mesh, sides, join = by_side(res, record)
    tip = np.unique(mesh.curves[record["tip"]])
    for nodes in sides:
        found = fault.locate(res.mapping(mesh.points[nodes]))
        # measured: 2.5e-8 above the Moho, 5.4e-10 below it
        assert np.abs(found.distance).max() <= 2.0 * record["offset"]
    # the join meets the tip at its two ends alone: no edge of the tip
    # runs along it
    assert join.size > 5 and np.count_nonzero(np.isin(join, tip)) == 2
    assert not np.any(np.all(np.isin(mesh.curves[record["tip"]][:, :2], join),
                             axis=1))
    x = res.mapping(mesh.points[join])
    # measured: 4.8e-14 off the Moho's relief
    assert np.abs(x[:, 2] - (root(x[:, 0], x[:, 1]) - 0.4)).max() < 1e-9
    found = fault.locate(x)
    assert np.all(found.inside) and np.abs(found.v).max() < 0.4
    # the tip is the plane's edge: every node on the boundary of the square
    on = fault.locate(res.mapping(mesh.points[tip]))
    assert np.allclose(np.maximum(np.abs(on.u), np.abs(on.v)), 0.5, atol=1e-6)
    # the slip at each node, either side and on the join, is the plane's
    # there, its (u, v) by projection onto the plane, a node of the tip a
    # hair beyond the edge reading the edge
    origin, (xu, xv) = patch.cartesian(0.0, 0.0), patch.tangents(0.0, 0.0)
    for nodes in (*sides, join):
        x = res.mapping(mesh.points[nodes])
        u, v = np.clip([(x - origin) @ xu / (xu @ xu),
                        (x - origin) @ xv / (xv @ xv)], -0.5, 0.5)
        want = np.stack([0.5 + u, 1.0 + 2.0 * v, 0.0 * u], axis=-1)
        got = fault.slip_at(x, max_distance=1e-6, basis="fault")
        assert np.allclose(got, want, rtol=0, atol=1e-9)


def test_a_crossing_of_a_knot_that_does_not_run_along_the_strike_is_refused(
        tmp_path):
    """A plane striking along x whose down-dip edge runs across the Moho's
    root, above it over the root and below it at either end: its crossing
    of the Moho leaves through that edge, and is not split."""
    fault = plane(0.6, 0.0, 0.42 - 0.1 * np.sin(np.deg2rad(40.0)), strike=90.0,
                  dip=40.0, length=0.6, width=0.2)
    bottom = fault.cartesian(np.linspace(-0.5, 0.5, 5), 0.5)
    moho = root(bottom[:, 0], bottom[:, 1]) - 0.4
    assert bottom[2, 2] > moho[2] and bottom[0, 2] < moho[0]
    with pytest.raises(ValueError, match="does not run along its strike from one "
                                         "end of the patch to the other"):
        build(tmp_path / "corner", {"f": fault})
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------ the free surface

def test_an_edge_at_the_reference_surface_under_a_mountain_is_carried_onto_it(
        tmp_path):
    """The fault's up-dip edge at z = 0 lies 0.05 below the range: its
    preimage's edge is carried across the reference top and trimmed there,
    so its trace, mapped, lies on the topography, above the fault's own
    edge, the faces continuing the fault up to it: the part carried is
    the preimage of the plane continued past its edge."""
    width, dip = 0.25, 50.0
    fault = plane(0.56, 0.0, 0.5 * width * np.sin(np.deg2rad(dip)))
    res = build(tmp_path / "trace", {"f": fault})
    (record,) = held(res, {"f": fault}).faults
    assert record["free_edge"] is not None and record["face_edge"] is None
    mesh = read_mesh(res.mesh_path)
    trace = mesh.points[np.unique(mesh.curves[record["free_edge"]])]
    assert np.abs(trace[:, 2]).max() < 1e-9           # on the reference top
    x = res.mapping(trace)
    assert np.abs(x[:, 2] - ridge(x[:, 0], x[:, 1])).max() < 1e-9
    assert x[:, 2].min() > 0.03                       # lifted over the fault's edge
    edge = fault.cartesian(np.linspace(-0.5, 0.5, 9), -0.5)
    assert np.abs(edge[:, 2]).max() < 1e-15
    assert ridge(edge[:, 0], edge[:, 1]).min() > 0.045


def test_a_fault_cut_by_the_topography_has_its_trace_on_it(tmp_path):
    """A fault given rising above the range, cut there by the topography:
    its trace lies on the topography, and every node, the trace's
    included, on the fault itself to the pull-back's offset."""
    centroid = np.array([0.55, 0.0, -0.1])
    fault = Plane(LOCAL, centroid, strike=0.0, dip=60.0, length=0.4, width=0.5,
                  slip=ZERO)
    assert fault.cartesian(0.0, -0.5)[2] > 0.11       # its edge above the range
    res = build(tmp_path / "cut", {"f": fault})
    (record,) = held(res, {"f": fault}).faults
    assert record["free_edge"] is not None
    mesh = read_mesh(res.mesh_path)
    x = res.mapping(mesh.points[np.unique(mesh.curves[record["free_edge"]])])
    assert np.abs(x[:, 2] - ridge(x[:, 0], x[:, 1])).max() < 1e-9
    found = Fault([fault]).locate(x, strict=False)
    assert np.all(found.inside) and np.abs(found.distance).max() < 1e-6
    # measured: 7.4e-7 of an element over all its nodes
    assert gate(res, Fault([fault])) < 1e-5


def test_an_edge_under_the_crest_is_carried_as_the_plane_continued(tmp_path):
    """The fault's up-dip edge at z = 0 under the range's crest, which
    stands 0.083 above it: the edge is carried across the reference top,
    the part carried the preimage of the plane continued past its edge,
    so that, mapped, it lies on that plane as the rest of the fault lies
    on the fault, its trace on the topography."""
    width, dip = 0.25, 50.0
    fault = plane(0.615, 0.0, 0.5 * width * np.sin(np.deg2rad(dip)))
    edge = fault.cartesian(np.linspace(-0.5, 0.5, 9), -0.5)
    assert ridge(edge[:, 0], edge[:, 1]).max() > 0.08
    res = build(tmp_path / "crest", {"f": fault})
    (record,) = held(res, {"f": fault}).faults
    assert record["free_edge"] is not None and record["face_edge"] is None
    mesh = read_mesh(res.mesh_path)
    x = res.mapping(mesh.points[np.unique(mesh.faces_with(record["attribute"]))])
    found = Fault([fault]).locate(x, strict=False)
    beyond = ~found.inside
    # carried up to 0.074 beyond the edge, measured along the plane
    assert np.abs(found.distance[beyond]).max() > 0.07
    # every node on the plane continued, the carried ones too: measured
    # 2.3e-7 (with the preimage itself extrapolated past the edge they lay
    # up to 3.8e-4 off it, beyond the contract's 1.5e-4)
    normal = fault.normal(0.0, 0.0)
    assert np.abs((x - fault.cartesian(0.0, 0.0)) @ normal).max() \
        <= 2.0 * record["offset"]
    trace = res.mapping(mesh.points[np.unique(mesh.curves[record["free_edge"]])])
    assert np.abs(trace[:, 2] - ridge(trace[:, 0], trace[:, 1])).max() < 1e-9


# ------------------------------------------------------------ the sides

def ramp(x, y):
    """A surface rising towards the side x = 1, 0.1 there."""
    return 0.1 * x


def test_a_fault_leaves_through_a_side_under_topography(tmp_path):
    """A fault through the side x = 1 under the ramp: its face edge on the
    side, the side's nodes moved along z alone; another stopping 0.05
    short of the side, where the stretch lifts the side by more than
    that, is not carried across it: the stretch moves the side along z,
    not towards the fault, and its edge there is a tip."""
    g = topography(surface=ramp, moho=None)
    through = plane(0.9, 0.0, 0.2, strike=90.0, length=0.4, width=0.2)
    short = plane(0.75, 0.0, 0.2, strike=90.0, length=0.4, width=0.2)
    assert np.isclose(short.cartesian(0.5, 0.0)[0], 0.95)
    res = build(tmp_path / "through", {"f": through}, geometry=g)
    (record,) = held(res, {"f": through}).faults
    assert record["face_edge"] is not None and record["free_edge"] is None
    mesh = read_mesh(res.mesh_path)
    side = mesh.points[np.unique(mesh.curves[record["face_edge"]])]
    assert np.abs(side[:, 0] - 1.0).max() < 1e-9
    assert np.abs(res.mapping(side)[:, 0] - 1.0).max() < 1e-9
    res = build(tmp_path / "short", {"f": short}, geometry=g)
    (record,) = held(res, {"f": short}).faults
    assert record["face_edge"] is None and record["tip"] is not None
    mesh = read_mesh(res.mesh_path)
    x = res.mapping(mesh.points[np.unique(mesh.faces_with(record["attribute"]))])
    assert x[:, 0].max() == pytest.approx(0.95, abs=1e-9)
    # the stretch moves the side's points by more than 0.05 over the
    # preimage of the fault's end, along z: the distance to the side is
    # none of it
    X = g.mapping.inverse(short.cartesian(0.5, np.linspace(-0.5, 0.5, 9)))
    X[:, 0] = 1.0
    moved = g.mapping(X) - X
    assert np.linalg.norm(moved, axis=1).max() > 0.05 and np.all(moved[:, :2] == 0.0)


def test_a_mapping_without_an_inverse_is_refused_at_the_build(tmp_path):
    class Lift:
        """z -> 1.1 z: a mapping that does not invert itself."""

        def __call__(self, X):
            return np.asarray(X, dtype=float) * [1.0, 1.0, 1.1]

        def deformation_gradient(self, X):
            X = np.asarray(X, dtype=float)
            return np.broadcast_to(np.diag([1.0, 1.0, 1.1]), X.shape[:-1] + (3, 3))

        def jacobian(self, X):
            return np.full(np.shape(X)[:-1], 1.1)

    with pytest.raises(ValueError, match="does not invert itself"):
        build(tmp_path / "lift", {"f": plane(0.5, 0.0, 0.6)},
              geometry=box(mapping=Lift()))
    assert not list(tmp_path.iterdir())


# ------------------------------------------------------------ AdGIA's box

def range_front(x, y, relief):
    """AdGIA's range topography (`meshes/faulted_box.py`, `range_front`),
    in its own lengths: a steep front winding along strike, a crest and a
    long back slope, a foreland basin and rolling hills."""
    front = -1.6 + 0.3 * np.sin(np.pi * y / 5.0)
    rise = 0.5 * (1.0 + np.tanh((x - front) / 0.35))
    back = np.exp(-(np.maximum(x + 0.7, 0.0) / 2.6) ** 2)
    along = 0.65 + 0.35 * np.cos(np.pi * y / 5.0)
    ridge_ = 0.32 * rise * back * along
    foreland = -0.06 * np.exp(-((x - front + 0.9) / 0.8) ** 2) * along
    hills = (0.045 * np.sin(2 * np.pi * x / 3.3 + 0.6)
             * np.cos(2 * np.pi * y / 4.1 + 1.1)
             + 0.03 * np.sin(2 * np.pi * (x + 0.6 * y) / 2.7 + 2.0)
             + 0.02 * np.cos(2 * np.pi * (0.4 * x - y) / 3.1 + 0.3))
    return relief * (ridge_ + foreland + hills)


def adgia(scale=1.0 / 8.0):
    """AdGIA's faulted box in 3-D at unit scale, all lengths divided by 8:
    the box 8 wide and 4 deep, the range of relief 2 (its 3-D default) on
    its free surface, the base held; the fault centred at x = 0, 1 below
    z = 0, dipping 30 degrees towards +x, 1.5 along dip by 1.5 along
    strike, a plane where it physically is."""
    W, D = 8.0 * scale, 4.0 * scale

    def height(x, y):
        return scale * range_front(x / scale, y / scale, 2.0)

    g = cartesian.Geometry(Skeleton([0.0, D]),
                           cartesian.Box([(-W / 2, W / 2), (-W / 2, W / 2)]),
                           layer_names=["body"], interface_names=["surface"])
    g = g.stretched(cartesian.layer_linear(g.skeleton, [height, None]),
                    name="range")
    fault = Plane(LOCAL, [0.0, 0.0, -1.0 * scale], strike=0.0, dip=30.0,
                  length=1.5 * scale, width=1.5 * scale, slip=Slip.from_rake(90.0, 1.0))
    return g, fault, height


def test_adgia_s_range_over_its_fault_at_unit_scale(tmp_path):
    """AdGIA's range topography over its buried thrust: built on the
    reference box, the fault pulled back, four tips, the hanging wall
    above it; mapped, the surface is the range to rounding at its nodes,
    reaching 0.08 above the fault, and the fault's nodes lie on AdGIA's
    plane to the pull-back's offset."""
    g, fault, height = adgia()
    res = build_layered_mesh(
        MeshSpec(g, UniformInterfaces(0.08, 0.15, 0.2), order=2,
                 faults=[EmbeddedFault(Fault([fault]), name="fault",
                                       sizing=InterfaceSizing(0.04, 0.15, 0.1))]),
        tmp_path / "adgia")
    assert res.validation.ok, res.validation.failures
    card = held(res, {"fault": fault})
    record = card.fault_record("fault")
    assert record["tip"] == 1 and record["free_edge"] is record["face_edge"] is None
    assert record["hanging_wall"] == [2] and record["controls"] == [[17, 17]]
    surface = mapped_nodes(res, card.interfaces[0]["attribute"])
    assert np.abs(surface[:, 2] - height(surface[:, 0], surface[:, 1])).max() < 1e-12
    # the range over the hanging wall, 0.09 at its highest node at unit scale
    assert 0.08 < surface[:, 2].max() < 0.1
    mesh = read_mesh(res.mesh_path)
    centres = res.mapping(mesh.points[mesh.cells[:, :4]].mean(axis=1))
    normal = fault.normal(0.0, 0.0)
    side = (centres - fault.cartesian(0.0, 0.0)) @ normal
    walls = mesh.walls(card)[0]
    assert np.all(side[walls == 1] > 0.0) and np.all(side[walls == -1] < 0.0)
    # measured: 2.1e-6 of the element size on the fault, 0.04
    assert gate(res, Fault([fault])) * H / 0.04 < 1e-5
    assert res.counts["dim3"] < 5000


# ------------------------------------------------------------ pictures

def test_a_section_under_topography_is_drawn_in_both_shapes(tmp_path):
    """A vertical section across the range through a fault cut by it: in
    reference shape the cells tile the box's section exactly; mapped, they
    tile the section of the physical body, whose area the topography and
    the Moho's root change by the integral of the surface's relief, and
    the fault's faces cut there lie on its line."""
    matplotlib = pytest.importorskip("matplotlib",
                                     reason="needs the planetmodel[plot] extra")
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from planetmodel.plotting import mesh_section, plane_cut

    fault = Plane(LOCAL, [0.55, 0.0, -0.1], strike=0.0, dip=60.0, length=0.4,
                  width=0.5, slip=ZERO)
    res = build(tmp_path / "drawn", {"f": fault}, order=1)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.mesh_path)
    origin, axes = [0.0, 0.05, 0.0], [[1, 0, 0], [0, 0, 1]]
    x = res.mapping(mesh.points)
    for points, area in ((mesh.points, 1.0), (x, None)):
        cut = plane_cut(points, mesh.cells[:, :4], origin, axes)
        s, t = cut.pieces[..., 0], cut.pieces[..., 1]
        areas = 0.5 * np.abs(np.sum(s * np.roll(t, -1, axis=1)
                                    - np.roll(s, -1, axis=1) * t, axis=1))
        if area is not None:
            assert np.isclose(areas.sum(), area, rtol=1e-12)
        else:
            # the base held: the area grows by the surface's relief, to the
            # linear elements' chords
            xs = np.linspace(0.0, 1.0, 2001)
            relief = np.trapezoid(ridge(xs, 0.05), xs)
            assert areas.sum() == pytest.approx(1.0 + relief, abs=2e-3)
        fig, ax = plt.subplots()
        colours = np.array(["#e1e0d9", "#c3c2b7", "#eb6834", "#2a78d6"])[
            mesh.cell_attributes - 1]
        polygons = mesh_section(ax, cut, colours=colours, linewidth=0.3)
        assert len(polygons.get_paths()) == len(cut.pieces)
        plt.close(fig)
    trace = plane_cut(x, mesh.faces_with(card.faults[0]["attribute"])[:, :3], origin,
                      axes)
    ends = trace.pieces.reshape(-1, 2)
    slope = -np.tan(np.deg2rad(60.0))
    x0 = fault.cartesian(0.0, -0.5)[0] - fault.cartesian(0.0, -0.5)[2] / slope
    assert np.allclose(ends[:, 1], slope * (ends[:, 0] - x0), rtol=0.0, atol=1e-6)
