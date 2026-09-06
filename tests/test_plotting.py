"""The radial profile helpers: radius upward, a joiner at every jump."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from planetmodel import PREM, LayeredIsotropicElastic, constant_field  # noqa: E402
from planetmodel.plotting import profile, radial_profile  # noqa: E402


def test_radius_is_vertical_and_every_jump_gets_a_joiner():
    m = LayeredIsotropicElastic([0.0, 0.5, 0.8, 1.0], rho=[3.0, 2.0, 2.0],
                                vp=[2.0, 2.0, 2.0], vs=[1.0, 0.0, 1.0])
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "rho", scale=2.0, value_scale=10.0, label="d")
    # three segments and two joiners (the second jump is zero-height, still drawn)
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert len(segments) == 3 and len(joiners) == 2
    x, y = segments[0].get_data()
    assert np.allclose(x, 30.0) and y[0] == 0.0 and np.isclose(y[-1], 1.0)
    jx, jy = joiners[0].get_data()
    assert np.allclose(jy, 1.0) and list(jx) == [30.0, 20.0]
    assert segments[0].get_label() == "d" and segments[1].get_label().startswith("_")
    assert len({ln.get_color() for ln in lines}) == 1
    assert all(ln.get_linestyle() == "-" for ln in lines)
    # the guides at the four boundaries sit beneath every profile line
    guides = [ln for ln in ax.get_lines() if ln not in lines]
    assert sorted(ln.get_ydata()[0] for ln in guides) == [0.0, 1.0, 1.6, 2.0]
    assert all(g.get_zorder() < ln.get_zorder() for g in guides for ln in lines)
    plt.close(fig)
    fig, ax = plt.subplots()
    radial_profile(ax, m, "rho", boundaries=False)
    assert len(ax.get_lines()) == 5
    plt.close(fig)


def test_a_missing_layer_is_a_gap_without_a_joiner():
    m = PREM(ocean=False)
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "qmu", joiners=True)
    held = m.layers_with("qmu")
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    assert len(segments) == len(held)
    # no joiner across the outer core, which lacks qmu
    cmb = m.geometry.interface("cmb").radius
    icb = m.geometry.interface("icb").radius
    assert all(ln.get_data()[1][0] not in (icb, cmb) for ln in joiners)
    plt.close(fig)


def test_refusals():
    m = LayeredIsotropicElastic([0.0, 1.0], rho=[1.0], vp=[1.0], vs=[1.0])
    fig, ax = plt.subplots()
    with pytest.raises(TypeError, match="complex"):
        profile(ax, [constant_field(1.0 + 1j, (0.0, 1.0))])
    with pytest.raises(ValueError, match="radial field of rank 0"):
        profile(ax, [m.elastic_moduli(0)])
    plt.close(fig)
