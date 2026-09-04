"""Skeleton: boundaries, membership and surgery."""
import numpy as np
import pytest

from planetmodel import CoarseningMap, Location, Skeleton


@pytest.fixture
def sk():
    return Skeleton([0.0, 1.0, 2.0, 3.0])


def test_construction_refusals():
    with pytest.raises(ValueError, match="at least two"):
        Skeleton([1.0])
    with pytest.raises(ValueError, match="1-d"):
        Skeleton([[0.0, 1.0]])
    with pytest.raises(ValueError, match="strictly increasing"):
        Skeleton([0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="non-negative"):
        Skeleton([-1.0, 1.0])


def test_basic_queries(sk):
    assert sk.nlayers == 3
    assert sk.span == 3.0
    assert not sk.is_hollow
    assert np.array_equal(sk.inner_boundaries, [1.0, 2.0])
    assert sk.interval(1) == (1.0, 2.0)
    assert sk.interval(-1) == (2.0, 3.0)
    assert sk.layer_index(-3) == 0
    with pytest.raises(IndexError):
        sk.interval(3)
    assert not sk.boundaries.flags.writeable


def test_hollow_skeleton():
    hollow = Skeleton([0.5, 1.0, 2.0])
    assert hollow.is_hollow
    assert hollow.span == 1.5
    assert hollow.locate(0.5).boundary == 0


def test_membership(sk):
    assert sk.spans(0.0, 3.0)
    assert sk.spans(1.0, 2.0, layer=1)
    assert sk.spans(1.0 + 1e-12, 2.0, layer=1)
    assert not sk.spans(1.0 + 1e-6, 2.0, layer=1)
    assert sk.spans(1.0 + 1e-6, 2.0, layer=1, rtol=1e-5)
    assert sk.contains(0.5, 2.5)
    assert not sk.contains(0.5, 3.5)


def test_locate(sk):
    assert sk.locate(0.5) == Location((0,))
    assert sk.locate(1.0) == Location((0, 1), boundary=1)
    assert sk.locate(0.0) == Location((0,), boundary=0)
    assert sk.locate(3.0) == Location((2,), boundary=3)
    assert sk.locate(1.0 + 1e-12).boundary == 1
    assert sk.locate(1.0 + 1e-6).boundary is None
    assert sk.locate(1.0 + 1e-6, rtol=1e-5).boundary == 1
    with pytest.raises(ValueError, match="outside"):
        sk.locate(3.1)
    with pytest.raises(ValueError, match="choose a side"):
        sk.locate(2.0).layer
    assert sk.locate(2.5).layer == 2


def test_refined_extended_truncated(sk):
    fine = sk.refined([0.5, 2.5])
    assert fine.nlayers == 5
    with pytest.raises(ValueError, match="already a boundary"):
        sk.refined([1.0])
    with pytest.raises(ValueError, match="cannot insert"):
        sk.refined([3.5])
    with pytest.raises(ValueError, match="duplicate"):
        sk.refined([0.5, 0.5])
    grown = sk.extended([4.0, 5.0])
    assert grown.nlayers == 5 and grown.boundaries[-1] == 5.0
    with pytest.raises(ValueError, match="cannot append"):
        sk.extended([2.5])
    cut = sk.truncated(2.5)
    assert np.array_equal(cut.boundaries, [0.0, 1.0, 2.0, 2.5])
    assert np.array_equal(sk.truncated(2.0).boundaries, [0.0, 1.0, 2.0])
    assert np.array_equal(sk.truncated(3.0).boundaries, sk.boundaries)
    with pytest.raises(ValueError, match="at or below"):
        sk.truncated(0.0)
    with pytest.raises(ValueError, match="beyond"):
        sk.truncated(4.0)


def test_hollowed(sk):
    inner = sk.hollowed(0.5)
    assert inner.is_hollow
    assert np.array_equal(inner.boundaries, [0.5, 1.0, 2.0, 3.0])
    assert np.array_equal(sk.hollowed(1.0).boundaries, [1.0, 2.0, 3.0])
    assert np.array_equal(sk.hollowed(0.0).boundaries, sk.boundaries)
    with pytest.raises(ValueError, match="at or above"):
        sk.hollowed(3.0)
    with pytest.raises(ValueError, match="below the innermost"):
        Skeleton([0.5, 1.0]).hollowed(0.2)


def test_coarsen(sk):
    coarse, cmap = sk.coarsen(drop=[0])
    assert np.array_equal(coarse.boundaries, [0.0, 2.0, 3.0])
    assert isinstance(cmap, CoarseningMap)
    assert cmap.layers == ((0, 1), (2,))
    assert cmap.kept_interfaces == (1,)
    assert cmap.dropped_interfaces == (0,)
    assert cmap.fine_layer(0.5) == 0 and cmap.fine_layer(1.0) == 1
    same, _ = sk.coarsen(keep=[1])
    assert same == coarse
    _, both = sk.coarsen(keep=[-1, 0])
    assert both.coarse == sk
    with pytest.raises(ValueError, match="exactly one"):
        sk.coarsen()
    with pytest.raises(IndexError):
        sk.coarsen(drop=[2])


def test_equality_and_repr(sk):
    assert sk == Skeleton([0, 1, 2, 3])
    assert sk != Skeleton([0, 1, 3])
    assert "3 layers" in repr(sk)
