"""A flat model's profile is drawn depth downward, a step at every
interface, the padding held below the base."""
import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib",
                                 reason="needs the planetmodel[plot] extra")
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from planetmodel import DENSITY, LayeredIsotropicElastic, Skeleton  # noqa: E402
from planetmodel import cartesian as flat  # noqa: E402
from planetmodel.layerfunction import polynomial_layer  # noqa: E402
from planetmodel.plotting import profile, radial_profile  # noqa: E402


def padded_model():
    """Three layers in kilometres and g/cm^3, held below the base to 150."""
    g = flat.Geometry(Skeleton([0.0, 20.0, 60.0, 100.0]),
                      flat.Box([(0, 400), (0, 400)]))
    rho = [flat.DepthField((0.0, 20.0), polynomial_layer((0.0, 20.0), [2.6, 0.01]),
                           character=DENSITY),
           flat.DepthField((20.0, 60.0), polynomial_layer((20.0, 60.0), [3.2, 0.002]),
                           character=DENSITY),
           flat.DepthField((60.0, 100.0), polynomial_layer((60.0, 100.0), [3.3, 0.001]),
                           character=DENSITY)]
    model = flat.Model(g, [{"rho": r} for r in rho])
    return model.extended([150.0], fields="hold")


def test_a_flat_profile_runs_depth_downward(tmp_path):
    m = padded_model()
    fig, ax = plt.subplots()
    lines = radial_profile(ax, m, "rho")
    assert ax.yaxis_inverted()
    segments = [ln for ln in lines if ln.get_label() != "_joiner"]
    joiners = [ln for ln in lines if ln.get_label() == "_joiner"]
    assert len(segments) == 4 and len(joiners) == 3
    x, y = segments[0].get_data()
    assert y[0] == 0.0 and y[-1] == 20.0 and np.isclose(x[0], 2.6)
    # the held padding is the base's value, joined to it without a step
    pad_x, pad_y = segments[-1].get_data()
    assert pad_y[0] == 100.0 and pad_y[-1] == 150.0 and np.allclose(pad_x, 3.4)
    jx, jy = joiners[-1].get_data()
    assert list(jy) == [100.0, 100.0] and np.isclose(jx[0], jx[1])
    path = tmp_path / "flat-profile.png"
    fig.savefig(path)
    assert path.stat().st_size > 0
    plt.close(fig)


def test_profile_inverts_the_axis_once_and_leaves_the_sphere_upward():
    m = padded_model()
    fig, ax = plt.subplots()
    profile(ax, [layer["rho"] for layer in m.layers])
    profile(ax, [layer["rho"] for layer in m.layers])
    assert ax.yaxis_inverted()
    plt.close(fig)
    sphere = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[3.0, 2.0], vp=[2.0, 1.5],
                                     vs=[1.0, 0.8])
    fig, ax = plt.subplots()
    radial_profile(ax, sphere, "rho")
    assert not ax.yaxis_inverted()
    plt.close(fig)
