"""A field that jumps at a breakpoint inside its layer: the value an
element takes at its end is the one from inside it."""
import numpy as np
from scipy.interpolate import PPoly

from planetmodel import Geometry, Model, RadialMesh, Skeleton
from planetmodel.catalogue import PREM
from planetmodel.fields import JUMP_RTOL, RadialField, from_inside
from planetmodel.layerfunction import PolynomialLayer


def stepped():
    """1 + r on [0, 1), 5 - r on [1, 2]: a jump from 2 to 4 at r = 1."""
    return PolynomialLayer((0.0, 2.0), PPoly(np.array([[1.0, -1.0], [1.0, 4.0]]),
                                             [0.0, 1.0, 2.0]))


def test_the_limit_from_below_differs_only_on_a_jump():
    f = stepped()
    assert f(1.0) == 4.0 and f.below(1.0) == 2.0
    r = np.array([0.0, 0.3, 0.999, 1.001, 1.7, 2.0])
    assert np.allclose(f.below(r), f(r), rtol=1e-15)
    assert f.below(r.reshape(2, 3)).shape == (2, 3)
    # beyond the breakpoints the end pieces continue, as they do in a call
    assert np.isclose(f.below(2.5), f(2.5)) and np.isclose(f.below(-0.5), f(-0.5))
    field = RadialField((0.0, 2.0), f)
    assert field.below(1.0) == 2.0
    # a field that is not a polynomial states no limit
    assert RadialField((0.0, 2.0), lambda x: np.sin(x)).below(1.0) is None


def test_sampled_values_are_corrected_at_the_outer_end_of_an_element():
    field = RadialField((0.0, 2.0), stepped())
    r = np.array([0.0, 0.5, 1.0, 1.0, 1.5, 2.0])       # two elements of three nodes
    outer = np.array([False, False, True, False, False, True])
    values = field(r)
    assert values.tolist() == [1.0, 1.5, 4.0, 4.0, 3.5, 3.0]
    assert from_inside(field, r, values, outer).tolist() == [1.0, 1.5, 2.0, 4.0, 3.5,
                                                             3.0]
    assert values[2] == 4.0                             # the input is not changed
    # pieces that meet to rounding are not a jump: the values come back as given
    c = np.array([[1.0, 1.0], [1.0, 2.0 + 1e-13]])
    smooth = RadialField((0.0, 2.0), PolynomialLayer((0.0, 2.0),
                                                     PPoly(c, [0.0, 1.0, 2.0])))
    values = smooth(r)
    assert 1e-13 < JUMP_RTOL
    assert np.array_equal(from_inside(smooth, r, values, outer), values)


def test_nodal_values_take_each_side_of_a_jump():
    sk = Skeleton([0.0, 2.0])
    model = Model(Geometry(sk), [{"f": RadialField((0.0, 2.0), stepped(), name="f")}])
    mesh = RadialMesh(sk, ngll=3, edges=[0.0, 1.0, 2.0])
    nodal = mesh.nodal(model, "f")
    assert nodal.tolist() == [[1.0, 1.5, 2.0], [4.0, 3.5, 3.0]]
    # and the polynomial through them is the field on each element
    back = mesh.to_ppoly(nodal)
    assert np.allclose(back(np.array([0.2, 0.9, 1.1, 1.9])), [1.2, 1.9, 3.9, 3.1])


def test_a_field_that_does_not_jump_is_sampled_as_before():
    # PREM's fields are one polynomial a layer; a mesh that cuts its layers
    # samples them at the cuts from either side alike
    prem = PREM()
    mesh = RadialMesh(prem.skeleton, ngll=5, drmax=200e3)
    for name in ("rho", "vpv"):
        nodal = mesh.nodal(prem, name)
        plain = np.empty_like(nodal)
        for i in np.unique(mesh.layer):
            on = mesh.layer == i
            plain[on] = prem.layer(int(i))[name].evaluate(mesh.r[on], 0.0, 0.0)
        assert np.array_equal(nodal, plain)
