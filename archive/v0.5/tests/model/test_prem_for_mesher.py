"""PREM and CRUST-1.0 provide what the mesher needs -- checked with no mesh.

The mesher consumes a short list from a body: boundary radii; per-layer
name, state, material and buffer flag; per-interface name and role; the
attached surfaces; and a mapping it can evaluate and check.  Whether the
real data produce that list is a model-layer fact, and this is the
reader-and-surgery run that establishes it -- in milliseconds, because
nothing is meshed.  The mesher's own handling of the same list is the
acceptance test, and it runs on a unit three-shell body in
tests/mesh3d/test_acceptance.py.

Two things the data force, both recorded here.  CRUST-1.0's mean Moho
lies *above* prem.nocrust's outer radius, so the deck grows to meet it.
And the depths are depths: attached raw at the mean Moho radius they put
the physical Moho twice as deep as the data says, which is why relief is
centred on attachment and why the check that the physical
Moho lands at the data's own depth is here.

The mapping's margin on this geometry is not a free parameter either.
A radial map between a uniform reference crust and a physical crust of
varying thickness stretches by the ratio of the two, so the margin at
true relief *is* the thinnest crustal column over the mean thickness --
0.33 on CRUST-1.0, six kilometres of oceanic crust in nineteen -- and
the map folds at one and a half times the relief.  The layer sets the
limit, not the rule.
"""
import warnings

import numpy as np
import pytest

from planetmodel import layer_linear
from planetmodel.io import read_isotropic_deck
from planetmodel.model.mapping import validity_lattice
from planetmodel.model.surface import Surface
from planetmodel.model.topography import GriddedTopography

pytestmark = pytest.mark.data

RREF = 6371.0e3
DECK = "tests/data/prem.nocrust"
MOHO = "tests/data/crust-1.0/depthtomoho.xyz"
THICKNESS = "tests/data/crust-1.0/crsthk.xyz"

LATTICE = dict(n_r=8, n_theta=45, n_phi=90)


@pytest.fixture(scope="module")
def relief():
    moho = GriddedTopography.from_xyz(MOHO, scale=1e3)
    surface = GriddedTopography.from_xyz(THICKNESS, scale=1e3) + moho
    return moho, surface


def centred(grid):
    """A grid as a departure from its own area-weighted mean.

    Still a grid -- the values move, the sampling does not -- so the
    relief attached to a boundary keeps its bounds and its file name.
    """
    return GriddedTopography(grid.lons, grid.lats, grid.values - grid.mean(),
                             interpolation=grid.interpolation, name=grid.name)


def acceptance_body(relief, exaggeration=1.0):
    """The deck, the surgery, and the two boundaries the crust needs.

    The deck is truncated at the Moho; on this data the Moho is 3 km
    above where the deck stops, so it is extended to meet it and the old
    boundary -- extrapolated mantle on both sides, no discontinuity --
    merged away.  The relief is centred before attachment, so each
    interface radius is the mean radius of the boundary it names.
    """
    moho, surface = relief
    r_moho, r_surf = RREF + moho.mean(), RREF + surface.mean()

    body, _ = read_isotropic_deck(DECK).coarsened(drop=range(-6, 0))
    assert r_moho > body.skeleton.boundaries[-1]           # the data's doing
    body = body.extended([r_moho], fields="extrapolate", interface_names=["moho"])
    body, _ = body.coarsened(drop=[len(body.interfaces) - 2])

    return (body.refined([r_moho - 300.0e3], names=["floor"], role="control")
                .extended([r_surf], fields=None, interface_names=["surface"])
                .with_buffer(ratio=0.2)
                .with_surface("moho", centred(moho) * exaggeration)
                .with_surface("surface", centred(surface) * exaggeration))


@pytest.fixture(scope="module")
def body(relief):
    return acceptance_body(relief)


def test_the_boundaries_are_where_the_data_put_them(body, relief):
    moho, surface = relief
    radii = {f.name: f.radius for f in body.interfaces}
    assert radii["moho"] == pytest.approx(RREF + moho.mean())
    assert radii["surface"] == pytest.approx(RREF + surface.mean())
    assert radii["floor"] == pytest.approx(radii["moho"] - 300.0e3)
    assert radii["buffer"] == pytest.approx(1.2 * radii["surface"])
    assert 6346.6e3 not in list(body.skeleton.boundaries)     # merged away
    assert np.all(np.diff(body.skeleton.boundaries) > 0.0)


def test_the_layers_and_interfaces_carry_what_the_manifest_records(body):
    names = [f.name for f in body.interfaces]
    assert names[-4:] == ["floor", "moho", "surface", "buffer"]
    assert [f.name for f in body.interfaces if f.role == "control"] == ["floor"]
    assert [lay.is_vacuum for lay in body.layers].count(True) == 1
    assert body.layers[-1].is_vacuum and body.layers[-1].name == "buffer"
    assert body.layers[-2].fields == {}                       # the crust
    assert "rho" in body.layers[-3].fields                    # the mantle
    assert {lay.state for lay in body.layers[:-1]} <= {"solid", "fluid"}


def test_the_surfaces_attach_to_the_boundaries_they_name(body):
    attached = {body.interfaces[i].name for i in body.surfaces}
    assert attached == {"moho", "surface"}
    for i, surface in body.surfaces.items():
        assert surface.reference_radius == pytest.approx(body.interfaces[i].radius)
        assert surface.is_centred(atol=1e-3)


def test_raw_depths_are_centred_on_attachment_and_the_shift_is_reported(relief):
    """The two routes to a centred boundary, and the warning on one.

    Handing `with_surface` the raw depth grid is the natural thing to
    write and was the silent mistake: the grid's mean is -21 km, and
    left in the relief it moves the physical Moho by that much again.
    Attachment removes it and says so.  Centring the surface first --
    `Surface(RREF, moho).centred()`, at the data's own mean radius --
    reaches the same boundary with nothing to report.
    """
    moho, _ = relief
    body = acceptance_body(relief)
    plain = body.without_surface("moho")

    with pytest.warns(UserWarning, match="area-weighted mean of -21421"):
        loud = plain.with_surface("moho", moho)

    quiet = plain.with_surface("moho", Surface(RREF, topography=moho).centred())

    t, p = np.array([0.4, 1.3, 2.7]), np.array([-1.0, 0.2, 2.5])
    expected = moho(t, p) - moho.mean()
    for b in (loud, quiet, body):
        assert b.surface("moho").height(t, p) == pytest.approx(expected, abs=1e-6)
        assert b.surface("moho").reference_radius == pytest.approx(
            RREF + moho.mean())


def test_an_uncentred_surface_is_refused_by_name(relief):
    """The mistake this contract exists to catch, refused where it is made."""
    moho, _ = relief
    plain = acceptance_body(relief).without_surface("moho")
    with pytest.raises(ValueError, match="relief of mean"):
        plain.with_surface("moho", Surface(RREF + moho.mean(), topography=moho))


def test_the_mapping_is_valid_and_its_knots_are_the_interfaces(body):
    mapping = body.mapping(rule=layer_linear())
    verdict = mapping.is_valid(sample=validity_lattice(body.skeleton, n_r=10,
                                                       n_theta=61, n_phi=120))
    assert verdict
    assert np.allclose(sorted(mapping.h.knots), body.skeleton.boundaries)


def test_the_physical_moho_lands_at_the_depth_the_data_gives(body, relief):
    """The check the first-stage listing did not make, and needed to.

    A point on the reference Moho maps to radius RREF + depth(theta,
    phi): the data's own number, to the metre, with no mesh anywhere.
    With uncentred relief this was out by the mean depth, 21 km, and
    every other assertion in this file still passed.
    """
    moho, _ = relief
    m = body.mapping(rule=layer_linear())
    r_moho = body.interface("moho").radius
    for t, p in ((0.6, 1.2), (2.0, -0.5), (np.pi / 2, 0.0)):
        n = np.array([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)])
        got = float(np.linalg.norm(m(r_moho * n)))
        assert got == pytest.approx(RREF + float(moho(t, p)), abs=1.0)


def test_the_thinnest_crustal_column_is_the_margin(relief, body):
    """Plan §1.3, on the data: the layer sets the limit, not the rule.

    `layer_linear` stretches the reference crust of uniform thickness
    onto the physical crust, so 1 + dh/dr there is the ratio of the two
    thicknesses and the margin is its minimum -- oceanic crust, six
    kilometres in a mean of nineteen.  Exaggerating the relief scales
    the departure from that mean, so the map folds a little past 1.4,
    and no larger factor belongs in an example on real crust.
    """
    moho, surface = relief
    r_moho, r_surf = RREF + moho.mean(), RREF + surface.mean()

    verdict = body.mapping(rule=layer_linear()).is_valid(
        sample=validity_lattice(body.skeleton, **LATTICE))
    assert verdict and 0.25 < verdict.margin < 0.4

    _, theta, phi = validity_lattice(body.skeleton, **LATTICE)
    thickness = ((r_surf + body.surface("surface").height(theta, phi))
                 - (r_moho + body.surface("moho").height(theta, phi)))
    assert verdict.margin == pytest.approx(
        float(thickness.min()) / (r_surf - r_moho), rel=0.05)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        folded = acceptance_body(relief, 1.5).mapping(rule=layer_linear())
    assert not [w for w in caught if "area-weighted" in str(w.message)]
    assert not folded.is_valid(sample=validity_lattice(body.skeleton, **LATTICE))
