"""The gmsh session manager, and the specification objects."""
import numpy as np
import pytest

import gmsh

from planetmodel import PREM, ReferenceBody, Skeleton
from planetmodel.mesh3d import (AngularResolution, BufferSpec, InterfaceSizing,
                           MeshSpec, PerInterface, UniformInterfaces)
from planetmodel.mesh3d._session import is_active, session, set_options
from planetmodel.model.topography import AnalyticTopography
from planetmodel.registry import lookup

pytestmark = pytest.mark.gmsh

COARSE = UniformInterfaces(0.12, 0.25, 0.3)


@pytest.fixture(scope="module")
def body():
    return (PREM(ocean=False)
            .name_interface(1, "cmb")
            .name_interface(-1, "surface"))


# ------------------------------------------------------------- session

def test_session_initialises_and_finalizes():
    assert not is_active()
    with session(name="test") as model:
        assert is_active()
        assert model.getCurrent() == "test"
    assert not is_active()


def test_session_finalizes_even_when_the_body_raises():
    """The whole reason this is a context manager."""
    with pytest.raises(RuntimeError, match="deliberate"):
        with session(name="test"):
            raise RuntimeError("deliberate")
    assert not is_active()


def test_sessions_do_not_nest():
    """One global model per process, so two callers sharing it is a bug."""
    with session(name="outer"):
        with pytest.raises(RuntimeError, match="does not nest"):
            with session(name="inner"):
                pass
    assert not is_active()


def test_set_options_dispatches_on_value_type():
    with session(name="test"):
        set_options({"Mesh.ElementOrder": 2, "Mesh.MshFileVersion": 2.2,
                     "General.ErrorFileName": "err.log"})
        assert gmsh.option.getNumber("Mesh.ElementOrder") == 2
        assert gmsh.option.getNumber("Mesh.MshFileVersion") == pytest.approx(2.2)
        assert gmsh.option.getString("General.ErrorFileName") == "err.log"


# --------------------------------------------------------- sizing rules

def test_the_sizing_rules_are_registered():
    assert lookup("sizing", "angular_resolution") is AngularResolution
    assert lookup("sizing", "uniform_interfaces") is UniformInterfaces
    assert lookup("sizing", "per_interface") is PerInterface


def test_angular_resolution_scales_with_radius(body):
    """Equal angular resolution: a deeper interface gets a smaller size."""
    rule = AngularResolution(h_ref=20e3, r_ref=6.371e6, h_far=200e3)
    sizes = rule(body.interfaces, 6.371e6)
    icb, surface = body.interfaces[0], body.interfaces[-1]
    assert sizes[icb.index].size < sizes[surface.index].size
    ratio = sizes[icb.index].size / sizes[surface.index].size
    assert ratio == pytest.approx(icb.radius / surface.radius, rel=1e-12)


def test_angular_resolution_never_exceeds_the_far_size(body):
    rule = AngularResolution(h_ref=1e9, r_ref=6.371e6, h_far=200e3)
    for sizing in rule(body.interfaces, 6.371e6).values():
        assert sizing.size <= sizing.far_size


def test_uniform_interfaces_gives_one_size(body):
    rule = UniformInterfaces(h_min=20e3, h_max=200e3, decay_width=200e3)
    sizes = rule(body.interfaces, 6.371e6)
    assert len({s.size for s in sizes.values()}) == 1
    assert all(s.size == 20e3 for s in sizes.values())


def test_per_interface_overrides_by_name_and_index(body):
    base = UniformInterfaces(20e3, 200e3, 200e3)
    fine = InterfaceSizing(1e3, 50e3, 100e3)
    rule = PerInterface({"cmb": fine, 0: fine}, base=base)
    sizes = rule(body.interfaces, 6.371e6)
    assert sizes[body.interface("cmb").index] is fine
    assert sizes[0] is fine
    assert sizes[body.interfaces[-1].index].size == 20e3


def test_per_interface_rejects_an_unknown_name(body):
    with pytest.raises(KeyError, match="no interface named"):
        PerInterface({"moho": InterfaceSizing(1e3, 2e3, 3e3)})(
            body.interfaces, 6.371e6)


def test_per_interface_without_a_base_must_cover_everything(body):
    with pytest.raises(ValueError, match="no sizing for interfaces"):
        PerInterface({0: InterfaceSizing(1e3, 2e3, 3e3)})(
            body.interfaces, 6.371e6)


def test_sizing_rules_are_frozen_and_comparable():
    a = AngularResolution(20e3, 6.371e6, 200e3)
    b = AngularResolution(20e3, 6.371e6, 200e3)
    assert a == b
    with pytest.raises(Exception):
        a.h_ref = 1.0


# ------------------------------------------------------ value validation

def test_interface_sizing_validates():
    with pytest.raises(ValueError, match="size must be positive"):
        InterfaceSizing(-1.0, 2.0, 3.0)
    with pytest.raises(ValueError, match="far_size .* is smaller"):
        InterfaceSizing(10.0, 1.0, 3.0)


def test_interface_sizing_scales():
    s = InterfaceSizing(1e3, 1e4, 1e5).scaled(1e-3)
    assert (s.size, s.far_size, s.decay_width) == (1.0, 10.0, 100.0)


def test_buffer_spec_is_exclusive():
    assert BufferSpec(ratio=0.2).ratio == 0.2
    with pytest.raises(ValueError, match="exactly one"):
        BufferSpec()
    with pytest.raises(ValueError, match="exactly one"):
        BufferSpec(ratio=0.2, radius=8e6)


def test_mesh_spec_validates(body):
    ok = dict(body=body, rref=6.371e6,
              sizing=UniformInterfaces(20e3, 200e3, 200e3))
    MeshSpec(**ok)                                   # the happy path
    with pytest.raises(ValueError, match="dimension must be"):
        MeshSpec(**ok, dimension=1)
    with pytest.raises(ValueError, match="element order"):
        MeshSpec(**ok, order=4)
    with pytest.raises(ValueError, match="delivery must be"):
        MeshSpec(**ok, delivery="halfway")
    with pytest.raises(ValueError, match="not both"):
        MeshSpec(**ok, keep_interfaces=[0], drop_interfaces=[1])
    with pytest.raises(ValueError, match="rref must be positive"):
        MeshSpec(body=body, rref=-1.0,
                 sizing=UniformInterfaces(20e3, 200e3, 200e3))
    with pytest.raises(ValueError, match="insert_names has"):
        MeshSpec(**ok, insert_radii=[1e6, 2e6], insert_names=["only-one"])


def test_mesh_spec_is_frozen(body):
    spec = MeshSpec(body=body, rref=6.371e6,
                    sizing=UniformInterfaces(20e3, 200e3, 200e3))
    with pytest.raises(Exception):
        spec.order = 3


def test_a_plain_function_is_a_valid_sizing_rule(body):
    """The rule is a callable; the shipped ones are a convenience."""
    def mine(interfaces, rref):
        return {f.index: InterfaceSizing(1e4, 1e5, 1e5) for f in interfaces}

    spec = MeshSpec(body=body, rref=6.371e6, sizing=mine)
    sizes = spec.sizing(body.interfaces, spec.rref)
    assert all(s.size == 1e4 for s in sizes.values())
    assert len(sizes) == len(body.interfaces)


def test_importing_mesh3d_does_not_disturb_the_process():
    """Importing the package must not start a session of its own."""
    assert not is_active()
    assert np.isfinite(6.371e6)


def test_surfaces_without_a_rule_are_refused():
    body = (ReferenceBody.from_fields(Skeleton([0.0, 1.0]), {})
            .name_interface(0, "surface"))
    relief = AnalyticTopography(lambda t, p: 0.01 * np.cos(t))
    with pytest.raises(ValueError, match="no mapping_rule"):
        MeshSpec(body=body, sizing=COARSE, surfaces={"surface": relief})
    with pytest.raises(ValueError, match="referential delivery"):
        MeshSpec(body=body, sizing=COARSE, delivery="referential")

