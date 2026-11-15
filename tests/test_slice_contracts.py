"""The slice contracts hold the sphere's slices: points, layers and values
against the model, the labels and the normalisation; and its spectra."""
import numpy as np
import pytest

from planetmodel import (VECTOR, AnalyticField, LayeredIsotropicElastic,
                         flattening, gauss_legendre)
from planetmodel import slices
from planetmodel.frames import cartesian_points
from planetmodel.testing import check_slice, check_spectrum


def lateral(r, t, p):
    return 4.0 + 0.3 * r * np.cos(t) + 0.2 * r * np.sin(t) * np.cos(p)


def flow(r, t, p):
    X = cartesian_points(r, t, p)
    return np.stack([X[..., 1], -X[..., 0], 0.5 * X[..., 2]], axis=-1)


@pytest.fixture
def two():
    m = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                vs=[4.0, 3.0])
    m = m.with_field(1, "vs", AnalyticField((0.5, 1.0), lateral, name="vs"),
                     replace=True)
    return m.with_field(1, "flow", AnalyticField((0.5, 1.0), flow, character=VECTOR,
                                                 frame="cartesian", name="flow"))


@pytest.mark.parametrize("stretch", [None, 0.05], ids=["identity", "flattened"])
def test_every_spherical_slice_meets_the_slice_contract(two, stretch):
    m = two if stretch is None else two.stretched(flattening(stretch))
    grid = gauss_legendre(6)
    for at in ("reference", "physical"):
        check_slice(slices.sphere(m, "vs", radius=0.8, grid=grid, at=at), m)
        check_slice(slices.section(m, "vs", (0.4, 0.1), end=(1.9, 2.0), n=4, m=7,
                                   at=at), m)
        check_slice(slices.profile(m, "rho", 0.3, 1.0, n=4, at=at), m)
    check_slice(slices.sphere(m, "rho", radius=0.5, grid=grid, side="below"), m)
    check_slice(slices.sphere(m, "flow", radius=0.8, grid=grid, component=1,
                              frame="spherical"), m)
    check_slice(slices.sphere(m, "vs", radius=0.8, grid=grid, relative_to="mean"), m)
    check_slice(slices.mean_profile(m, "vs", mean_grid=grid, n=4), m)
    check_slice(slices.integrated(m, "vs", grid=grid), m)
    check_slice(slices.cutaway(m, "vs", n=24, inner=0.3), m)
    check_spectrum(slices.spectrum(m, "vs", radius=0.8, lmax=4))


def test_the_slice_contract_holds_a_physical_section_to_the_mapped_body(two):
    m = two.stretched(flattening(0.1))
    # cut at the reference outer radius, the equator's raised surface left
    # out: refused as the whole body, held to the bounds it was given
    for cut in (slices.section(m, "vs", (0.4, 0.1), end=(1.9, 2.0), n=4, m=7,
                               at="physical", radii=(0.0, 1.0)),
                slices.profile(m, "vs", 1.5, 0.3, n=4, at="physical",
                               radii=(0.0, 1.0))):
        with pytest.raises(AssertionError, match="where the mapping puts"):
            check_slice(cut, m)
        check_slice(cut, m, bounds=(0.0, 1.0))
    # a layer within the bounds may not be left out
    outer = slices.section(m, "vs", (0.4, 0.1), end=(1.9, 2.0), n=4, m=7,
                           at="physical", radii=(0.6, 2.0))
    with pytest.raises(AssertionError, match="leaves it out"):
        check_slice(outer, m)
    check_slice(outer, m, bounds=(0.6, 2.0))
