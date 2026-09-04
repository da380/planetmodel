"""Plan §4.2's MFEM delivery, measured on the acceptance body.

Everything here is asked of MFEM rather than of planetmodel: the mesh is
loaded the way a consumer loads it, the GridFunctions are read back off
disk, and the mass is a `LinearForm` against the constant one -- which
for a positive field is the integral of the interpolant over the curved
mesh, and is what a solver's assembly would see.

Three numbers say whether the delivery is right.  The mass of `rho` on
the MFEM side is the model's own `4 pi \\int rho r^2 dr`, to the accuracy
with which a mesh eight elements across represents a ball.  The physical
mesh the exporter makes agrees node for node with the one gmsh makes by
displacing the same reference mesh, which is the plan's promise that
there is one mapping and two ways of applying it.  And the pushed-forward
density integrates over the physical mesh to the mass the referential one
integrates to over the reference mesh, which is the conservation
law, `rho_phys = rho_ref / J`, read as an integral.

The body is `test_acceptance`'s three shells, at COARSE sizing, built
once for the module: five shells and a vacuum buffer, an interface with
relief, `layer_linear`, order 2, under two seconds.  One field is added
to it that the acceptance body has not -- a density on the core alone --
because a field whose domain is one layer is the case the manifest's
`layers` exists for.

Units.  The mesher divides geometry by `rref` and leaves fields in the
body's own scale system (`_units.py`), so the mesh is in units of
`rref = 1e6 m` and the density is in kg/m^3: a mass computed on the mesh
is the model's mass divided by `rref^3`, and the tests say so rather
than hiding it in a tolerance.
"""
import dataclasses

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.spatial import cKDTree

from planetmodel import DENSITY, RadialField, Skeleton
from planetmodel.io import manifest as sc
from planetmodel.mesh3d import build_layered_mesh, export_mfem
from planetmodel.model.materials import ElasticField, Symmetry
from planetmodel.model.units import Dimensions

from .test_acceptance import acceptance_spec, deck

mfem = pytest.importorskip("mfem.ser", reason="needs the planetmodel[mfem] extra")

pytestmark = [pytest.mark.gmsh, pytest.mark.mfem]

#: The quadrature the mass integrals are taken with.  The integrand is a
#: degree-2 interpolant against a degree-3 Jacobian on a curved
#: tetrahedron, so anything from 5 upwards is exact on the mesh as it
#: stands and the remaining error is the mesh's own geometry.
QUADRATURE_ORDER = 8


def body_with_a_core_field():
    """The acceptance deck, plus two fields that exist only in the core.

    `with_field` takes a single-layer field on that layer's interval,
    which is what a layer stores; the domain is `(0,)` and stays so
    through the surgery, since every cut and extension happens above it.
    The elasticity is there to carry a rank-4 Voigt value -- 36
    components in one GridFunction, pushed forward slot by slot -- on the
    56 elements of the core rather than on the whole mesh, which is all
    it takes to exercise the shape.
    """
    inner = Skeleton([0.0, 0.2e6])
    core = RadialField(inner, [lambda r: 8.0e3 - 1.0e-3 * r], name="core_rho",
                       character=DENSITY, dimensions=Dimensions.DENSITY)
    elastic = ElasticField(
        Symmetry.ISOTROPIC,
        {"kappa": RadialField(inner, [lambda r: 1.3e11 + 0.0 * r]),
         "mu": RadialField(inner, [lambda r: 6.7e10 + 0.0 * r])},
        name="elastic_moduli")
    return (deck().with_field(0, "core_rho", core)
            .with_field(0, "elastic_moduli", elastic))


def referential_spec():
    """The acceptance spec, delivered referentially.

    The exporter builds both deliveries from the reference mesh, so the
    mesh it is given must be the undisplaced one -- which is what
    `delivery="referential"` writes.
    """
    return dataclasses.replace(acceptance_spec(), delivery="referential",
                               body=body_with_a_core_field())


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    return build_layered_mesh(referential_spec(),
                              tmp_path_factory.mktemp("export") / "shells")


@pytest.fixture(scope="module")
def exported(built, tmp_path_factory):
    """Both deliveries of the one build, written beside each other."""
    root = tmp_path_factory.mktemp("delivery")
    return {"referential": export_mfem(built, root / "ref",
                                       delivery="referential"),
            "physical": export_mfem(built, root / "phys", delivery="physical")}


# ------------------------------------------------------------- the machinery

def load(export, name=None):
    """The mesh a consumer loads, and one of its GridFunctions.

    Constructed the way the manifest says and MFEM's own DataCollection
    does -- edges generated, no refinement marking, no orientation fix --
    because re-marking tetrahedra permutes the dofs the file is indexed
    by.
    """
    options = export.files["mesh_read_options"]
    mesh = mfem.Mesh(str(export.mesh_path), options["generate_edges"],
                     options["refine"], options["fix_orientation"])
    if name is None:
        return mesh
    return mesh, mfem.GridFunction(mesh, str(export.field_paths[name]))


def integrate(mesh, gf):
    """`\\int f dV` over the whole mesh, by MFEM's own quadrature."""
    integrator = mfem.DomainLFIntegrator(mfem.ConstantCoefficient(1.0))
    integrator.SetIntRule(mfem.IntRules.Get(mesh.GetElementGeometry(0),
                                            QUADRATURE_ORDER))
    form = mfem.LinearForm(gf.FESpace())
    form.AddDomainIntegrator(integrator)
    form.Assemble()
    return mfem.InnerProduct(form, gf)


def model_mass(body, name, *, divisor):
    """`4 pi \\int rho r^2 dr` over the layers that hold the field.

    In mesh units: the geometry was divided by `rref` and the density
    was not, so the mass a mesh integral produces is smaller than the
    model's by `rref^3`.
    """
    field = body[name]
    total = sum(quad(lambda r, i=i: field.evaluate(r, layer=i) * r * r,
                     *body.skeleton.interval(i), limit=200)[0]
                for i in field.domain)
    return 4.0 * np.pi * total / divisor ** 3


def nodes_of(mesh):
    """The nodal coordinates as (ndof, 3), whatever the ordering."""
    from planetmodel.mesh3d.export import _node_array
    return np.array(_node_array(mesh.GetNodes()), dtype=float)


def gmsh_perturbation(built):
    """The mesher's path: merge the reference mesh, displace it, read the nodes.

    The comparison the plan asks for is between two ways of applying one
    mapping, so the *same file* is displaced here -- not a second build,
    which would only prove the mesher deterministic.
    """
    import gmsh

    from planetmodel.mesh3d._displace import apply_mapping
    from planetmodel.mesh3d._session import session
    from planetmodel.mesh3d._units import GeometryScaledMapping

    scaled = GeometryScaledMapping(built.mapping, built.units)
    with session(name="perturbation"):
        gmsh.merge(str(built.msh_path))
        apply_mapping(scaled)
        _, coordinates, _ = gmsh.model.mesh.getNodes()
    return np.asarray(coordinates, dtype=float).reshape(-1, 3)


# ------------------------------------------------------------- the three numbers

def test_the_mass_on_the_mfem_side_is_the_model_s_mass(built, exported):
    """-1.383e-5 relative, and it is the ball, not the field.

    The density is exact on the mesh -- constant per layer, interpolated
    by an L2 space of the mesh order -- and the quadrature is exact too:
    a degree-2 interpolant against the cubic Jacobian of an order-2
    tetrahedron, unchanged from rule order 3 to 20.  So every part of the
    difference is the volume the curved tetrahedra enclose against the
    ball they approximate, which at COARSE sizing and order 2 is a part
    in 1e5 (the four layers holding rho enclose 3.5913143 against
    4 pi 0.95^3 / 3 = 3.5913640).  The tolerance is an order looser than
    the number observed, since it is a statement about the mesh rather
    than about this build.
    """
    mesh, gf = load(exported["referential"], "rho")
    wanted = model_mass(built.body, "rho", divisor=built.units.divisor)
    assert integrate(mesh, gf) == pytest.approx(wanted, rel=1e-4)


def test_the_exported_physical_mesh_is_the_gmsh_perturbation(built, exported):
    """Node for node, to 1e-12 in mesh units -- observed 7e-16.

    Both deliveries are checked against it: the physical mesh the
    exporter writes, and the reference mesh plus the displacement
    GridFunction, which is the one MFEM call the plan says the
    referential delivery costs a consumer.
    """
    reference = gmsh_perturbation(built)
    tree = cKDTree(reference)

    physical = nodes_of(load(exported["physical"]))
    assert physical.shape == reference.shape
    distance, index = tree.query(physical)
    assert distance.max() < 1e-12
    assert len(np.unique(index)) == len(reference)

    referential = exported["referential"]
    mesh = load(referential)
    displacement = mfem.GridFunction(mesh, str(referential.displacement_path))
    from planetmodel.mesh3d.export import _node_array
    assembled = (nodes_of(mesh)
                 + np.array(_node_array(displacement), dtype=float))
    assert tree.query(assembled)[0].max() < 1e-12


def test_the_pushed_forward_density_carries_the_same_mass(built, exported):
    """Plan §3.9 as an integral: rho_phys = rho_ref / J conserves mass.

    Observed 1.4e-8 relative.  It is not exact because the physical mesh
    interpolates `m` at its nodes rather than following it exactly, so
    the volume it encloses differs from the image of the reference mesh
    at the order of the geometry; the tolerance is two orders looser than
    the number, and far tighter than the mesh's own 1.4e-5 against the
    ball, which is the point -- the *same* mass travels.
    """
    reference = integrate(*load(exported["referential"], "rho"))
    physical = integrate(*load(exported["physical"], "rho"))
    assert physical == pytest.approx(reference, rel=1e-6)


# -------------------------------------------------------- what MFEM makes of it

def test_mfem_finds_nothing_to_fix_in_the_written_mesh(exported):
    """Both counts zero, on the file the consumer opens.

    The exporter checks the MSH it loads; this checks what it wrote,
    which is a different file in a different format and the one that is
    actually delivered.
    """
    for export in exported.values():
        mesh = load(export)
        assert mesh.CheckElementOrientation(True) == 0
        assert mesh.CheckBdrElementOrientation(True) == 0


def test_the_manifest_lists_every_file_it_wrote(built, exported):
    """A fresh read gives back the block, and the block is the truth."""
    export = exported["referential"]
    card = sc.read(export.manifest_path)

    assert card.delivery == "referential"
    assert card.files["mesh"] == export.mesh_path.name
    assert card.files["mesh_read_options"]["refine"] == 0

    entries = {e["name"]: e for e in card.files["grid_functions"]}
    assert set(entries) == {"rho", "core_rho", "elastic_moduli", "displacement"}
    for name, entry in entries.items():
        assert (export.mesh_path.parent / entry["file"]).is_file()
        assert entry["frame"] == "cartesian"

    rho = entries["rho"]
    assert rho["fe_space"] == "L2_3D_P2"
    assert rho["vdim"] == 1
    assert (rho["character_rank"], rho["character_weight"]) == (0, 1)
    assert rho["physical_dimensions"] == [1, -3, 0]      # mass, length, time
    assert rho["units"] == "kg m-3"
    assert rho["layers"] == [0, 1, 2, 3]                  # not the crust, not
    assert rho["attributes"] == [1, 2, 3, 4]              # the buffer
    assert rho["representation"] == "referential"

    moved = entries["displacement"]
    assert moved["kind"] == "displacement"
    assert moved["vdim"] == 3
    assert moved["fe_space"] == "H1_3D_P2"
    assert moved["physical_dimensions"] == [0, 1, 0]

    # The physical delivery says the other three things, on the same body.
    other = sc.read(exported["physical"].manifest_path)
    assert other.delivery == "physical"
    assert other.mapping["applied_to_nodes"] is True
    assert [e["name"] for e in other.files["grid_functions"]] == \
        [e["name"] for e in card.files["grid_functions"] if e["name"]
         != "displacement"]


def test_a_field_on_one_layer_is_written_there_and_zero_elsewhere(built,
                                                                  exported):
    """The domain reaches the file: values in the core, zeros above it.

    A GridFunction has no room for "not defined here" -- a NaN would
    spread through the first integrator that touched it -- so the value
    outside the domain is zero and the manifest's `layers` is what says
    the zero means nothing.
    """
    export = exported["referential"]
    mesh, gf = load(export, "core_rho")
    entry = next(e for e in export.files["grid_functions"]
                 if e["name"] == "core_rho")
    assert entry["layers"] == [0]
    assert entry["attributes"] == [1]

    values = gf.GetDataArray()
    fes = gf.FESpace()
    inside = np.concatenate([np.asarray(fes.GetElementDofs(e), dtype=int)
                             for e in range(mesh.GetNE())
                             if mesh.GetAttribute(e) == 1])
    outside = np.setdiff1d(np.arange(fes.GetNDofs()), inside)

    # 8000 - 1e-3 r over [0, 2e5]: strictly between 7800 and 8000.
    assert values[inside].min() > 7.8e3
    assert values[inside].max() <= 8.0e3
    assert np.all(values[outside] == 0.0)
    # The core is 56 elements across a ball of radius 0.2 and encloses
    # 0.64% less volume than the sphere it approximates, so its mass is
    # 5.8e-3 low: a statement about 56 tetrahedra, not about the export.
    assert integrate(mesh, gf) == pytest.approx(
        model_mass(built.body, "core_rho", divisor=built.units.divisor),
        rel=1e-2)


def test_a_rank_four_field_is_written_in_voigt_at_its_own_dofs(built,
                                                               exported):
    """36 components in one GridFunction, and the values are the field's.

    The check is a round trip through the file: the dof coordinates are
    recovered from the written mesh exactly as the exporter recovered
    them from the reference mesh, the field is evaluated there again,
    and the two agree to the precision the file was written with.  That
    pins the byNODES layout of
    a vdim-36 GridFunction, which no scalar field can.
    """
    from planetmodel.mesh3d.export import _dof_coordinates

    export = exported["referential"]
    entry = next(e for e in export.files["grid_functions"]
                 if e["name"] == "elastic_moduli")
    assert (entry["vdim"], entry["components"], entry["voigt"]) == (36, [6, 6], 1)
    assert (entry["character_rank"], entry["character_weight"]) == (4, 1)
    assert entry["units"] == "kg m-1 s-2"
    assert entry["layers"] == [0]

    mesh, gf = load(export, "elastic_moduli")
    fes = mfem.FiniteElementSpace(mesh, gf.FESpace().FEColl(), 1)
    X = _dof_coordinates(mesh, fes)
    core = np.unique(np.concatenate(
        [np.asarray(fes.GetElementDofs(e), dtype=int)
         for e in range(mesh.GetNE()) if mesh.GetAttribute(e) == 1]))

    field = built.body["elastic_moduli"]
    wanted = field.evaluate_at(X[core] * built.units.divisor, layer=0,
                               frame="cartesian")
    got = gf.GetDataArray().reshape(36, -1).T[core].reshape(-1, 6, 6)
    # Sixteen significant digits of text, so the round trip is exact to
    # the last bit of a modulus of 2.2e11 and not to the last bit of one.
    assert np.abs(got - wanted).max() <= 1e-15 * np.abs(wanted).max()

    outside = np.setdiff1d(np.arange(fes.GetNDofs()), core)
    assert np.all(gf.GetDataArray().reshape(36, -1).T[outside] == 0.0)


def test_a_named_subset_goes_into_h1_at_the_order_asked_for(built, tmp_path):
    """`fields=`, `order=` and `continuous=`, which are the three knobs.

    An H1 space shares its dof at an interface, so a discontinuous
    quantity put there loses one side of every jump -- which is why the
    flag is the caller's promise and not a default.  What is checked is
    that the promise is kept where it holds: strictly inside a layer the
    values are that layer's, and outside the domain they are still zero.
    """
    from planetmodel.mesh3d.export import _dof_coordinates

    export = export_mfem(built, tmp_path / "h1", delivery="referential",
                         fields=["rho"], order=1, continuous=("rho",))
    entry, = [e for e in export.files["grid_functions"]
              if e["kind"] == "field"]
    assert (entry["fe_space"], entry["continuous"]) == ("H1_3D_P1", True)
    assert set(export.field_paths) == {"rho"}

    mesh, gf = load(export, "rho")
    values = gf.GetDataArray()
    r = np.linalg.norm(_dof_coordinates(mesh, gf.FESpace()), axis=1)
    inside = (r > 0.05) & (r < 0.94)            # below the Moho at 0.95
    above = r > 1.01                            # above the surface at 1.0
    assert np.all(values[inside] == 5.0e3)
    assert np.all(values[above] == 0.0)


@pytest.fixture(scope="module")
def built_at_order_one(tmp_path_factory):
    """A straight-sided mesh: MFEM gives it no nodal space of its own."""
    spec = dataclasses.replace(referential_spec(), order=1)
    return build_layered_mesh(spec, tmp_path_factory.mktemp("linear") / "flat")


def test_a_straight_sided_mesh_exports_too(built_at_order_one, tmp_path):
    """The order-1 case, where `GetNodes()` starts out null.

    A mesh with no curved nodes has no nodal GridFunction until it is
    asked for one, and the displacement lives in exactly that space, so
    the exporter gives it the identity one and everything downstream --
    dof coordinates, the displacement, the physical mesh -- is unchanged.
    The straight-sided ball is about a percent short of the sphere, which
    is a statement about polyhedra and is the tolerance here.
    """
    export = export_mfem(built_at_order_one, tmp_path / "flat",
                         delivery="physical", fields=["rho"])
    mesh, gf = load(export, "rho")
    assert gf.FESpace().FEColl().Name() == "L2_3D_P1"
    wanted = model_mass(built_at_order_one.body, "rho",
                        divisor=built_at_order_one.units.divisor)
    assert integrate(mesh, gf) == pytest.approx(wanted, rel=2e-2)


def test_a_mesh_already_displaced_is_refused(built, tmp_path):
    """The exporter starts from the reference mesh, and says so.

    `delivery="physical"` in the *mesher* displaces the nodes before writing,
    so that file's coordinates are physical and the fields could only be
    evaluated on it by inverting the mapping.  Plan §4.2 builds both
    deliveries from the reference mesh instead, and a build that already
    moved is turned away rather than silently displaced twice.
    """
    card = sc.read(built.manifest_path)
    card.mapping["applied_to_nodes"] = True
    moved = dataclasses.replace(
        built, manifest_path=sc.write(tmp_path / "moved", card))

    with pytest.raises(ValueError, match="physical mesh"):
        export_mfem(moved, tmp_path / "nope", delivery="physical")
