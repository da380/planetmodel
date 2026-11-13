"""The walls of a fault, and the collar attributes the mesher writes.

The classification (`manifest.fault_walls`): the cells holding a
slip-carrying vertex or edge of a fault, joined across faces that are
not the fault's and hold one too, fall into groups on its two sides,
each labelled by the faces of the fault it holds.  Held here to a hand
built mesh of a few hundred tetrahedra, where every answer is known, and
to the meshes the builder makes of the fault configurations it accepts:
every cell holding a slip-carrying vertex or edge is in a wall, the
hanging wall lies on the fault's normal side by an independent check
(the fault's own `locate`, signed), the cells touching a fault along its
pinned edges alone are in neither, and a face turned against the
others is caught.  The collar attributes are the classification's, and
survive refinement: every cell of a mesh refined once (each tetrahedron
cut in eight) that is in a wall of a fault inherits from its parent the
attribute that says which.
"""
import itertools

import numpy as np
import pytest

from planetmodel import testing
from planetmodel.fault import Fault, Plane, StructuredPatch
from planetmodel.mesh3d import (EmbeddedFault, InterfaceSizing, MeshSpec, Shell,
                                build_layered_mesh, manifest as sc, read_mesh)

from test_faults import (SIZING, UNIT, ZERO, curved, plane, reaching, two_layers)

pytestmark = pytest.mark.gmsh


# ------------------------------------------------------------ a hand-built mesh

def kuhn(n):
    """The unit cube in n^3 cubes, each cut into the six tetrahedra about
    its main diagonal: (points, cells), conforming."""
    g = np.linspace(0.0, 1.0, n + 1)
    points = np.array(list(itertools.product(g, g, g)))

    def at(i, j, k):
        return (i * (n + 1) + j) * (n + 1) + k

    cells = []
    for i, j, k in itertools.product(range(n), repeat=3):
        for order in itertools.permutations(range(3)):
            corner = [0, 0, 0]
            path = [at(i, j, k)]
            for axis in order:
                corner[axis] = 1
                path.append(at(i + corner[0], j + corner[1], k + corner[2]))
            cells.append(path)
    return points, np.array(cells)


def faces_in(points, cells, axis, value, box):
    """The faces of the cells lying in the plane x[axis] = value within
    box = (lo, hi) of the other two coordinates, each once, oriented
    along +axis."""
    other = [a for a in range(3) if a != axis]
    out = set()
    for cell in cells:
        for face in sc._TET_FACES:
            tri = cell[face]
            x = points[tri]
            if (np.allclose(x[:, axis], value) and x[:, other].min() >= box[0] - 1e-12
                    and x[:, other].max() <= box[1] + 1e-12):
                out.add(tuple(sorted(tri.tolist())))
    f = np.array(sorted(out))
    n = np.cross(points[f[:, 1]] - points[f[:, 0]], points[f[:, 2]] - points[f[:, 0]])
    f[n[:, axis] < 0] = f[n[:, axis] < 0][:, [0, 2, 1]]
    return f


def boundary_faces(cells):
    """The faces of a mesh held by one cell: its outer surface."""
    sides = np.sort(cells[:, sc._TET_FACES].reshape(-1, 3), axis=1)
    unique, counts = np.unique(sides, axis=0, return_counts=True)
    return unique[counts == 1]


@pytest.fixture(scope="module")
def cube():
    return kuhn(4)


def test_a_buried_square_is_classified_by_side(cube):
    """A square of the plane z = 1/2, x and y in [1/4, 3/4], inside a cube
    of 4^3 cubes: its one inner vertex and the edges of its faces that
    are not on its edge carry slip; the cells holding them are in its
    hanging wall above the plane and its footwall below, and the four
    cells holding only a corner's diagonal (an edge of the fault joining
    two of its tip's vertices) are among them."""
    points, cells = cube
    faces = faces_in(points, cells, 2, 0.5, (0.25, 0.75))
    (curves,) = sc.fault_curves([faces], boundary_faces(cells))
    assert len(curves.tip) == 8 and curves.pinned.tolist() == curves.tip.tolist()
    walls = sc.fault_walls(points, cells, faces, curves.pinned, name="f")
    z = points[cells].mean(axis=1)[:, 2]
    assert np.all(z[walls == 1] > 0.5) and np.all(z[walls == -1] < 0.5)
    assert np.count_nonzero(walls == 1) == np.count_nonzero(walls == -1) == 14
    # by its slip-carrying vertex alone, four cells fewer
    inner = np.flatnonzero(np.all(np.isclose(points, 0.5), axis=1))
    by_vertex = np.isin(cells, inner).any(axis=1)
    assert np.all(walls[by_vertex] != 0) and np.count_nonzero(by_vertex) == 24
    # every other cell touching the fault touches it on its tip alone
    on_fault = np.isin(cells, faces).any(axis=1)
    alone = on_fault & (walls == 0)
    assert np.any(alone)
    assert np.all(np.isin(cells[alone], np.concatenate([np.unique(curves.tip),
                                                        np.setdiff1d(np.arange(len(points)),
                                                                     faces)])))


def test_a_face_turned_against_the_others_is_caught(cube):
    points, cells = cube
    faces = faces_in(points, cells, 2, 0.5, (0.25, 0.75))
    (curves,) = sc.fault_curves([faces], boundary_faces(cells))
    turned = faces.copy()
    turned[0] = turned[0][[0, 2, 1]]
    with pytest.raises(ValueError, match="fault 'f': .* on both sides of its faces"):
        sc.fault_walls(points, cells, turned, curves.pinned, name="f")
    # a tip left out joins the two sides around it
    with pytest.raises(ValueError, match="joining its hanging wall to its footwall"):
        sc.fault_walls(points, cells, faces, curves.pinned[1:], name="f")
    holder = np.flatnonzero(np.isin(cells, faces[0]).sum(axis=1) == 3)[0]
    with pytest.raises(ValueError, match="1 of its 8 faces are not a face of two"):
        sc.fault_walls(points, np.delete(cells, holder, axis=0), faces,
                       curves.pinned, name="f")
    with pytest.raises(ValueError, match="corner indices"):
        sc.fault_walls(points, cells[:, :3], faces, curves.pinned)


def test_a_fault_meeting_the_surface_has_its_trace_free(cube):
    """The square run out to the face x = 0 of the cube: its edge there is
    its free edge, its vertices carry slip, and both walls reach it."""
    points, cells = cube
    faces = faces_in(points, cells, 2, 0.5, (0.0, 0.75))
    faces = faces[points[faces][:, :, 1].min(axis=1) >= 0.25 - 1e-12]
    (curves,) = sc.fault_curves([faces], boundary_faces(cells))
    assert len(curves.free_edge) == 2 and len(curves.junction_end) == 0
    walls = sc.fault_walls(points, cells, faces, curves.pinned)
    x = points[cells].min(axis=1)[:, 0]
    assert np.any((walls == 1) & (x == 0.0)) and np.any((walls == -1) & (x == 0.0))


def test_a_fault_ending_on_another_is_pinned_there_and_the_other_is_not(cube):
    """A square in z = 1/2 and a wall in x = 1/2 rising from it, z in
    [1/2, 3/4]: the wall's lower edge is on its own boundary and inside
    the square, so the wall ends on the square, its slip pinned there,
    and the square's is not; and the wall carried down to z = 1/4 crosses
    the square, neither pinned."""
    points, cells = cube
    square = faces_in(points, cells, 2, 0.5, (0.25, 0.75))
    surface = boundary_faces(cells)
    for low, ends in ((0.5, True), (0.25, False)):
        wall = faces_in(points, cells, 0, 0.5, (0.25, 0.75))
        z = points[wall][:, :, 2]
        wall = wall[(z.min(axis=1) >= low - 1e-12)]
        s, w = sc.fault_curves([square, wall], surface)
        assert len(s.junction) == len(w.junction) == 2
        assert len(s.junction_end) == 0 and s.ends_on == ()
        assert (len(w.junction_end), w.ends_on) == ((2, (0,)) if ends else (0, ()))
        walls = sc.walls_of_faults(points, cells, [square, wall], surface,
                                   names=["square", "wall"])
        centres = points[cells].mean(axis=1)
        assert np.all(centres[walls[0] == 1, 2] > 0.5)
        assert np.all(centres[walls[1] == 1, 0] > 0.5)
        assert np.all(centres[walls[1] == -1, 0] < 0.5)
        # the square's walls reach the junction from both sides of the wall
        near = np.abs(centres[:, 0] - 0.5) < 0.2
        assert {1, -1} <= set(walls[0][near].tolist())
        # where they cross, cells are in both hanging walls
        numbered = sc.collars(np.ones(len(cells), dtype=int), walls, layer_count=1)
        assert (1, (0, 1)) in numbered.combinations


def test_the_collars_are_numbered_by_layer_then_faults():
    walls = np.array([[1, 1, 0, 1, -1, 0],
                      [0, 1, 1, 1, 0, -1]])
    layers = np.array([2, 2, 2, 1, 1, 1])
    numbered = sc.collars(layers, walls, layer_count=3)
    assert numbered.combinations == ((1, (0, 1)), (2, (0,)), (2, (0, 1)), (2, (1,)))
    assert numbered.attributes.tolist() == [5, 6, 7, 4, 1, 1]
    assert numbered.layer_attributes(2) == [2, 5, 6, 7]
    assert numbered.layer_attributes(3) == [3]
    assert numbered.hanging_wall(0) == [4, 5, 6]
    assert numbered.hanging_wall(1) == [4, 6, 7]
    plain = sc.collars(layers, np.zeros((0, 6)), layer_count=3)
    assert plain.attributes.tolist() == layers.tolist() and plain.combinations == ()


# ------------------------------------------------------------ built meshes

def ending():
    """A main fault and a splay whose down-dip edge lies in it exactly: a
    bilinear patch through two points of the main plane, and the same two
    moved up dip and into its hanging wall."""
    main = plane(strike=0.0, dip=60.0, length=0.4, width=0.3)
    foot = main.cartesian(np.array([-0.3, 0.3]), np.array([0.1, 0.1]))
    up = main.cartesian(0.0, -0.5) - main.cartesian(0.0, 0.5)
    rise = 0.08 * main.normal(0.0, 0.0) + 0.2 * up
    corners = np.stack([foot + rise, foot], axis=1)
    splay = StructuredPatch(UNIT, UNIT.coordinates(corners), slip=ZERO,
                            interpolation="linear")
    return Fault([main]), Fault([splay])


def crossing():
    a = plane(strike=0.0, dip=60.0, length=0.3, width=0.3)
    b = Plane(UNIT, a.centroid, strike=90.0, dip=60.0, length=0.3, width=0.3,
              slip=ZERO)
    return Fault([a]), Fault([b])


def patches():
    a = plane(strike=0.0, dip=60.0, length=0.2)
    shift = a.cartesian(0.5, 0.0) - a.cartesian(-0.5, 0.0)
    b = Plane(UNIT, UNIT.coordinates(a.cartesian(0.0, 0.0) + shift), strike=0.0,
              dip=60.0, length=0.2, width=0.2, slip=ZERO)
    return Fault([a, b])


#: The configurations the builder accepts, by name: the faults by name,
#: and the spec's extra arguments.
CASES = {
    "inside": ({"f": Fault([plane()])}, {}),
    "across": ({"f": Fault([plane(depth=0.4, dip=60.0, width=0.35)])}, {}),
    "surface": ({"f": Fault([reaching()])}, {}),
    "shell": ({"f": Fault([reaching()])}, {"shells": [Shell(ratio=0.3)]}),
    "two": ({"a": Fault([plane()]),
             "b": Fault([plane(-30.0, 200.0, strike=100.0, dip=70.0)])}, {}),
    "splay": (dict(zip(("main", "splay"), ending())), {}),
    "crossing": (dict(zip(("a", "b"), crossing())), {}),
    "patches": ({"seg": patches()}, {}),
    "curved": ({"slab": Fault([curved()])}, {}),
}

NEAR = InterfaceSizing(0.06, 0.3, 0.1)


def build(tmp, name, *, order=1):
    faults, extra = CASES[name]
    items = [EmbeddedFault(f, name=n, sizing=NEAR) for n, f in faults.items()]
    return build_layered_mesh(MeshSpec(two_layers(), SIZING, faults=items, order=order,
                                       **extra), tmp / name)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    root = tmp_path_factory.mktemp("collars")
    cache = {}

    def get(name):
        if name not in cache:
            cache[name] = build(root, name)
        return cache[name]
    return get


@pytest.mark.parametrize("name", list(CASES))
def test_every_configuration_is_classified_and_its_collars_written(built, name):
    """The contract (`testing.check_fault_mesh`: the collar attributes the
    classification's, every cell's attribute one of the layer it lies
    in, the hanging wall on the fault's normal side by `locate`), and
    the classification's own promises on the mesh as read: every cell
    holding a slip-carrying vertex is in a wall, a cell touching the
    fault on its pinned edges alone in neither, a shell's cells in none,
    and a face turned against the others refused, naming the fault."""
    res = built(name)
    faults = CASES[name][0]
    testing.check_fault_mesh(res.msh_path, faults, mapping=res.mapping,
                             distance=1e-9)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    walls = mesh.walls(card)
    cells = mesh.cells[:, :4]
    surface = mesh.faces_with(card.surface_attribute)[:, :3]
    faces = [mesh.faces_with(e["attribute"])[:, :3] for e in card.faults]
    shell = ~np.isin(mesh.layers(card), [e["attribute"] for e in card.layers
                                         if e["in_geometry"]])
    for k, (record, found) in enumerate(zip(card.faults,
                                            sc.fault_curves(faces, surface))):
        carries = np.setdiff1d(np.unique(faces[k]), found.pinned)
        holds = np.isin(cells, carries).any(axis=1) & ~shell
        assert np.all(walls[k][holds] != 0), record["name"]
        touching = np.isin(cells, faces[k]).any(axis=1)
        pinned_only = touching & ~holds & ~shell & (walls[k] == 0)
        assert np.all(np.isin(cells[pinned_only],
                              np.concatenate([found.pinned.ravel(),
                                              np.setdiff1d(np.unique(cells),
                                                           faces[k])])))
        assert np.all(walls[k][shell] == 0)
        hanging = np.isin(mesh.cell_attributes, record["hanging_wall"])
        assert np.array_equal(hanging, walls[k] == 1)
        turned = faces[k].copy()
        middle = np.flatnonzero(np.isin(turned, carries).any(axis=1))[0]
        turned[middle] = turned[middle][[0, 2, 1]]
        with pytest.raises(ValueError, match=f"fault '{record['name']}'"):
            sc.fault_walls(mesh.points, cells[~shell], turned, found.pinned,
                           name=record["name"])


def test_a_splay_ends_on_its_main_fault_and_crossing_faults_share_a_collar(built):
    card = sc.read(built("splay").manifest_path)
    main, splay = card.faults
    assert splay["ends_on"] == ["main"] and main["ends_on"] == []
    assert set(main["hanging_wall"]) & set(splay["hanging_wall"])
    card = sc.read(built("crossing").manifest_path)
    a, b = card.faults
    assert a["ends_on"] == b["ends_on"] == []
    assert a["meets"] == ["b"]
    assert [e["attributes"] for e in card.layers] == [[1], [2, 3, 4, 5]]
    assert (a["hanging_wall"], b["hanging_wall"]) == ([3, 4], [4, 5])
    names = read_mesh(built("crossing").msh_path).names
    assert names[(3, 4)] == "mantle:a+b"


def test_a_shell_holds_no_collar(built):
    card = sc.read(built("shell").manifest_path)
    assert [e["attributes"] for e in card.layers] == [[1], [2, 4], [3]]
    assert card.shell_attributes == (3,)
    assert card.element_attributes(layers=[2, 3]) == (2, 4, 3)
    assert card.layer_of(4)["name"] == "mantle"
    with pytest.raises(KeyError, match="carry element attribute 5"):
        card.layer_of(5)


def test_the_builder_s_check_refuses_cells_that_disagree(built, tmp_path):
    """`confirm_collars` reads a written mesh back: a collar cell given
    its layer's attribute in the file fails it, and both files go."""
    import shutil

    from planetmodel.mesh3d._validate import confirm_collars

    res = built("inside")
    msh = tmp_path / "copy.msh"
    js = tmp_path / "copy.json"
    shutil.copyfile(res.msh_path, msh)
    shutil.copyfile(res.manifest_path, js)
    confirm_collars(msh, js)
    text = msh.read_text().splitlines()
    rows = range(text.index("$Elements") + 2, text.index("$EndElements"))
    # a tetrahedron of the collar (physical group 3) moved to the layer's
    # own group and entity
    tets = [(i, text[i].split()) for i in rows if text[i].split()[1] == "4"]
    entity = next(w[4] for _, w in tets if w[3] == "2")
    i, words = next((i, w) for i, w in tets if w[3] == "3")
    words[3:5] = ["2", entity]
    text[i] = " ".join(words)
    msh.write_text("\n".join(text) + "\n")
    with pytest.raises(RuntimeError, match="other element attribute than the "
                                           "classification"):
        confirm_collars(msh, js)
    assert not msh.exists() and not js.exists()


# ------------------------------------------------------------ refinement

def refine(points, cells, faces):
    """Every tetrahedron cut in eight at its edges' midpoints (Bey's rule),
    and every face in four, each child keeping its parent's vertex order:
    (points, cells, the parent of each cell, faces as given)."""
    edges = {}
    new = [p for p in points]

    def mid(a, b):
        key = (min(a, b), max(a, b))
        if key not in edges:
            edges[key] = len(new)
            new.append(0.5 * (points[a] + points[b]))
        return edges[key]

    children, parents = [], []
    for c, (v0, v1, v2, v3) in enumerate(cells.tolist()):
        m01, m02, m03 = mid(v0, v1), mid(v0, v2), mid(v0, v3)
        m12, m13, m23 = mid(v1, v2), mid(v1, v3), mid(v2, v3)
        children += [[v0, m01, m02, m03], [m01, v1, m12, m13], [m02, m12, v2, m23],
                     [m03, m13, m23, v3], [m01, m02, m03, m13], [m01, m02, m12, m13],
                     [m02, m03, m13, m23], [m02, m12, m13, m23]]
        parents += [c] * 8

    def split(tri):
        out = []
        for a, b, c in tri.tolist():
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            out += [[a, ab, ca], [ab, b, bc], [ca, bc, c], [ab, bc, ca]]
        return np.array(out, dtype=np.int64).reshape(-1, 3)

    faces = [split(f) for f in faces]
    return np.array(new), np.array(children), np.array(parents), faces


@pytest.mark.parametrize("name", ["inside", "surface", "splay", "crossing"])
def test_the_collar_attributes_survive_refinement(built, name):
    """Refined once, each child inheriting its parent's attribute: every
    child in a wall of a fault by the classification of the refined mesh
    carries an attribute of the fault's hanging wall where it is in that
    wall, and none where it is in the footwall.  The cells of a corner's
    diagonal matter here: its midpoint carries slip once refined."""
    res = built(name)
    card = sc.read(res.manifest_path)
    mesh = read_mesh(res.msh_path)
    faces = [mesh.faces_with(e["attribute"])[:, :3] for e in card.faults]
    surface = mesh.faces_with(card.surface_attribute)[:, :3]
    points, cells, parents, refined = refine(mesh.points, mesh.cells[:, :4],
                                             faces + [surface])
    walls = sc.walls_of_faults(points, cells, refined[:-1], refined[-1])
    inherited = mesh.cell_attributes[parents]
    for k, record in enumerate(card.faults):
        hanging = np.isin(inherited, record["hanging_wall"])
        assert np.all(hanging[walls[k] == 1]), record["name"]
        assert not np.any(hanging[walls[k] == -1]), record["name"]
        assert np.count_nonzero(walls[k]) > np.count_nonzero(mesh.walls(card)[k])


def test_a_reader_of_either_schema_reads_its_own_and_refuses_the_rest(built, tmp_path):
    """A manifest with faults is read back as written under schema 6; the
    same file declaring schema 5, whose cells are all their layers', is
    refused, as is one declaring a schema this library does not read."""
    import json

    res = built("inside")
    data = json.loads(res.manifest_path.read_text())
    assert data["schema"] == sc.SCHEMA_WITH_FAULTS == "planetmodel.mesh.manifest/6"
    card = sc.read(res.manifest_path)
    again = sc.write(tmp_path / "again", card)
    assert again.read_text() == res.manifest_path.read_text()
    for schema, match in ((sc.SCHEMA, "faults are listed under schema"),
                          ("planetmodel.mesh.manifest/7", "this planetmodel reads")):
        path = tmp_path / f"{schema.rsplit('/')[-1]}.json"
        path.write_text(json.dumps(data | {"schema": schema}))
        with pytest.raises(ValueError, match=match):
            sc.read(path)
