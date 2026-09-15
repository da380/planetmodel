"""The recipe's pure helpers, exercised without gmsh."""
import pytest

from planetmodel.io import recipe as rp
from planetmodel.model.surface import Surface
from planetmodel.model.topography import AnalyticTopography


def test_unit_suffixes_convert_and_booleans_are_left_alone():
    got = rp._lengths({"h_ref_km": 20, "fraction": 0.2, "flag_m": True,
                       "radii_km": [1, 2]})
    assert got == {"h_ref": 20000.0, "fraction": 0.2, "flag_m": True,
                   "radii": (1000.0, 2000.0)}


def test_a_key_given_twice_is_refused():
    with pytest.raises(ValueError, match="given twice"):
        rp._lengths({"h_ref_km": 20, "h_ref": 5.0})


def test_a_radius_is_a_number_and_nothing_else():
    """The dataset-specific ways of saying a radius are gone."""
    assert rp._metres(6.0e6, "[x]") == 6.0e6
    for said in ("mean_moho", {"surface": "moho", "km": 300}, True):
        with pytest.raises(ValueError, match="cannot read"):
            rp._metres(said, "[x]")


def test_a_surface_must_sit_where_the_recipe_puts_its_boundary():
    """Both numbers are named, since either could be the wrong one."""
    surface = Surface(6.0e6, topography=AnalyticTopography(lambda t, p: 0 * t))
    assert rp._placed("moho", surface, 6.0e6 + 1.0, 6.371e6).reference_radius \
        == 6.0e6 + 1.0                        # inside 1e-6 rref: recipe wins
    with pytest.raises(ValueError, match="6000000 m .* at 5990000 m"):
        rp._placed("moho", surface, 5.99e6, 6.371e6)
