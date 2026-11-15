"""A flat model sampled on depth nodes times a horizontal grid's nodes.

The sample holds to `check_sample` (the contract the spherical sample is
held to, its lateral half asked of the grid and the model's coordinates)
over a box and a section, identity and stretched; its layout is (node,
x, y) or (node, x), a field of depth alone (node,); both one-sided values
survive at an interface; a missing field is refused or NaN; the depth
mesh is sized by the grid's spacing unless told; what is not a flat
model or grid is refused by name.
"""
import numpy as np
import pytest
from scipy.interpolate import PPoly

from planetmodel import DENSITY, SCALAR, VECTOR, RadialMesh, Skeleton
from planetmodel import cartesian as flat
from planetmodel.catalogue import LayeredIsotropicElastic
from planetmodel.layerfunction import polynomial_layer
from planetmodel.sampling import SampleBase, gauss_legendre
from planetmodel.testing import check_sample

DEPTHS = [0.0, 0.3, 1.0]
BOX = flat.Box([(0.0, 2.0), (-1.0, 1.0)])
RECT = flat.Rectangle((0.0, 2.0))


def fourier(interval, grid, char=SCALAR, seed=0):
    t = grid.transform
    k = np.sqrt((t.wavevectors.astype(float) ** 2).sum(axis=0))
    shape = char.voigt_shape or char.component_shape
    c = np.random.default_rng(seed).standard_normal((2, 1, t.size) + shape)
    c /= (1.0 + k).reshape((1, 1, -1) + (1,) * len(shape))
    c[1, 0, 0] += 10.0 * np.sqrt(grid.area)            # a positive mean
    return flat.FourierField(interval, grid, PPoly(c, np.array(interval)),
                             character=char)


def model_over(domain):
    """Two layers: a density of depth alone on both, a Fourier field on the
    top layer and a gridded one below under the same name, a vector given
    by a formula on both, and a scalar on the top layer alone."""
    sk = Skeleton(DEPTHS)
    g = flat.Geometry(sk, domain, layer_names=["upper", "lower"],
                      interface_names=["surface", "boundary"])
    grid = flat.Grid(domain, (8, 6) if len(domain.axes) == 2 else 8)
    top, low = sk.interval(0), sk.interval(1)

    def v(d, x, y):
        return np.stack(np.broadcast_arrays(1.0 + d, np.sin(np.pi * x),
                                            0.5 * np.cos(np.pi * y)), axis=-1)

    lower_q = flat.GriddedField.sampled(
        flat.AnalyticField(low, lambda d, x, y: 2.0 + d * np.cos(np.pi * x)), grid,
        ngll=3, lateral="cubic")
    layers = [
        {"rho": flat.DepthField(top, polynomial_layer(top, [2.6, 0.5]),
                                character=DENSITY),
         "q": fourier(top, grid),
         "v": flat.AnalyticField(top, v, character=VECTOR),
         "only_top": flat.DepthField(top, 1.0)},
        {"rho": flat.DepthField(low, polynomial_layer(low, [3.2, 0.1]),
                                character=DENSITY),
         "q": lower_q,
         "v": flat.AnalyticField(low, v, character=VECTOR)},
    ]
    return flat.Model(g, layers), grid


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "section"])
def test_a_flat_sample_holds_to_the_sample_contract(domain):
    model, grid = model_over(domain)
    s = flat.sample(model, grid, fields=["rho", "q", "v"], drmax=0.25)
    check_sample(s, model)
    assert isinstance(s, SampleBase) and s.grid is grid
    n = s.nnode
    assert s.fields["rho"].shape == (n,)
    assert s.fields["q"].shape == (n,) + grid.shape
    assert s.fields["v"].shape == (n,) + grid.shape + (3,)
    assert s.displacement is None and np.array_equal(s.depth, s.radius)
    assert s.layer_names == ("upper", "lower")
    assert repr(s).startswith(f"cartesian.Sample({n} nodes x ")


def test_both_one_sided_values_survive_at_the_interface():
    model, grid = model_over(BOX)
    s = flat.sample(model, grid, fields=["rho", "q"], drmax=0.5)
    at = np.flatnonzero(s.radius == 0.3)
    assert at.size == 2 and np.array_equal(s.node_layer[at], [0, 1])
    assert s.fields["rho"][at[0]] == pytest.approx(2.6 + 0.5 * 0.3)
    assert s.fields["rho"][at[1]] == pytest.approx(3.2 + 0.1 * 0.3)
    x, y = grid.nodes
    upper = model.layer(0)["q"].evaluate(0.3, x, y)
    lower = model.layer(1)["q"].evaluate(0.3, x, y)
    assert np.allclose(s.fields["q"][at[0]], upper, rtol=0, atol=1e-12)
    assert np.allclose(s.fields["q"][at[1]], lower, rtol=0, atol=1e-12)


def test_a_missing_field_is_refused_or_nan_and_the_default_names():
    model, grid = model_over(BOX)
    with pytest.raises(KeyError, match="only_top"):
        flat.sample(model, grid, fields=["only_top"])
    s = flat.sample(model, grid, fields=["only_top"], missing="nan")
    check_sample(s, model)
    assert np.all(np.isnan(s.fields["only_top"][s.node_layer == 1]))
    assert set(flat.sample(model, grid).fields) == {"rho", "q", "v"}


def test_the_depth_mesh_is_sized_by_the_grid_unless_told():
    model, grid = model_over(BOX)
    s = flat.sample(model, grid, fields=["rho"])
    assert s.radial.jac.max() * 2 <= min(grid.spacing) * (1 + 1e-12)
    mesh = RadialMesh(model.geometry, ngll=4, drmax=0.5)
    assert flat.sample(model, grid, fields=["rho"], radial=mesh).radial is mesh
    s = flat.sample(model, flat.Grid(BOX, 4), fields=["rho"], ngll=3)
    assert s.radial.ngll == 3


@pytest.mark.parametrize("domain", [BOX, RECT], ids=["box", "section"])
def test_a_stretched_model_samples_its_displacement(domain):
    """Under a vertical stretch the displacement is (w, 0, 0) in the local
    frame, w the upward displacement at the node."""
    model, grid = model_over(domain)
    surface = flat.FourierFunction(grid, 0.02 * np.eye(grid.transform.size)[1])
    w = flat.layer_linear(model.skeleton, [surface, None, None])
    stretched = model.stretched(w)
    s = flat.sample(stretched, grid, fields=["q"], drmax=0.25)
    check_sample(s, stretched)
    x, y = grid.nodes
    d = s.radius.reshape((-1,) + (1,) * grid.dimension)
    assert np.allclose(s.displacement[..., 0], w(d, x[None], y[None]), rtol=0,
                       atol=1e-15)
    assert np.all(s.displacement[..., 1:] == 0.0)


def test_check_sample_holds_the_spherical_sample_as_before():
    sphere = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                     vs=[4.0, 3.0])
    from planetmodel import sample
    s = sample(sphere, gauss_legendre(4), drmax=0.25)
    check_sample(s, sphere)
    assert s.grid is s.angular and s.angular.shape == (5, 9)
    model, grid = model_over(BOX)
    with pytest.raises(AssertionError, match="not an AngularGrid"):
        check_sample(flat.sample(model, grid, fields=["rho"]), _spherical_like(model))


def _spherical_like(model):
    """A spherical model on the flat model's skeleton, for the contract to
    see a flat sample offered as a spherical model's."""
    return LayeredIsotropicElastic(list(model.skeleton.boundaries),
                                   rho=[1.0, 1.0], vp=[2.0, 2.0], vs=[1.0, 1.0])


def test_refusals_name_what_is_wrong():
    model, grid = model_over(BOX)
    sphere = _spherical_like(model)
    with pytest.raises(TypeError, match="expected a cartesian.Model"):
        flat.sample(sphere, grid)
    with pytest.raises(TypeError, match="cartesian.Grid"):
        flat.sample(model, gauss_legendre(4))
    with pytest.raises(ValueError, match="missing must be one of"):
        flat.sample(model, grid, missing="zero")
    other = RadialMesh(Skeleton([0.0, 0.5, 1.0]), ngll=3, drmax=0.5)
    with pytest.raises(ValueError, match="another skeleton"):
        flat.sample(model, grid, radial=other)
    mesh = RadialMesh(model.geometry, ngll=3, drmax=0.5)
    with pytest.raises(ValueError, match="radial= or drmax="):
        flat.sample(model, grid, radial=mesh, drmax=0.1)
    with pytest.raises(TypeError, match="RadialMesh"):
        flat.sample(model, grid, radial=np.array([0.0, 1.0]))
