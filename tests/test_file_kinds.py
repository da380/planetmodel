"""Spherical and flat files side by side: the readers that read either kind
give each its own objects by the schema the file states, and the readers
of one kind's numbers refuse the other kind by its schema, naming the
reader that reads it."""
import numpy as np
import pytest

from planetmodel import LayeredIsotropicElastic, Model, gauss_legendre
from planetmodel import cartesian as flat
from planetmodel.surfacefile import SurfaceVariable

pytestmark = pytest.mark.netcdf
netCDF4 = pytest.importorskip("netCDF4", reason="needs the planetmodel[netcdf] extra")

from planetmodel import modelfile, surfacefile  # noqa: E402
from planetmodel.cartesian import modelfile as flat_modelfile  # noqa: E402
from planetmodel.cartesian import surfacefile as flat_surfacefile  # noqa: E402

BOX = flat.Box([(0.0, 2.0), (0.0, 1.0)])


@pytest.fixture
def model_files(tmp_path):
    ball = LayeredIsotropicElastic([0.0, 0.5, 1.0], rho=[10.0, 4.0], vp=[9.0, 7.0],
                                   vs=[4.0, 3.0])
    box = flat.LayeredIsotropicElastic([0.0, 0.5, 1.0], BOX, rho=[3.0, 3.3],
                                       vp=[6.0, 8.0], vs=[3.5, 4.5])
    return (modelfile.write_model(tmp_path / "ball.nc", ball, grid=gauss_legendre(2)),
            modelfile.write_model(tmp_path / "box.nc", box, grid=flat.Grid(BOX, 2)))


def test_the_model_readers_give_each_kind_its_own_model(model_files):
    ball, box = model_files
    assert modelfile.schema_of(ball) == modelfile.SCHEMA
    assert modelfile.schema_of(box) == modelfile.SCHEMA_FLAT
    assert type(modelfile.read_model(ball)) is Model
    assert type(modelfile.read_model(box)) is flat.Model
    modelfile.validate_structure(ball)
    modelfile.validate_structure(box)
    assert modelfile.read_surfaces(box) == {}
    assert flat_modelfile.read_model_file(box).domain == BOX
    assert modelfile.read_model_file(ball).domain is None


def test_the_numbers_readers_refuse_the_other_kind_by_name(model_files):
    ball, box = model_files
    with pytest.raises(ValueError, match=(
            r"schema is 'planetmodel.model/3', this reader reads "
            r"'planetmodel.model/1' and 'planetmodel.model/2'; a file of "
            r"'planetmodel.model/3' is read by "
            r"planetmodel.cartesian.modelfile.read_model_file")):
        modelfile.read_model_file(box)
    with pytest.raises(ValueError, match=(
            r"schema is 'planetmodel.model/1', this reader reads "
            r"'planetmodel.model/3'; a file of 'planetmodel.model/1' is read by "
            r"planetmodel.modelfile.read_model_file")):
        flat_modelfile.read_model_file(ball)


def test_the_surface_readers_give_and_refuse_by_kind(tmp_path):
    sphere = gauss_legendre(2)
    plane = flat.Grid(BOX, 4)
    on_sphere = surfacefile.write_surface(
        tmp_path / "sphere.nc", sphere, source="tests",
        fields={"k": SurfaceVariable(np.ones((sphere.ntheta, sphere.nphi)))})
    on_plane = flat_surfacefile.write_surface(
        tmp_path / "plane.nc", plane, source="tests",
        fields={"k": SurfaceVariable(np.ones(plane.shape))})
    assert surfacefile.schema_of(on_plane) == surfacefile.SCHEMA_FLAT
    assert type(surfacefile.read_surface(on_sphere)) is surfacefile.SurfaceHistory
    assert type(surfacefile.read_surface(on_plane)) is flat_surfacefile.SurfaceHistory
    surfacefile.validate_structure(on_plane)
    with pytest.raises(ValueError, match=(
            r"this reader reads 'planetmodel.surface/1'; a file of "
            r"'planetmodel.surface/2' is read by "
            r"planetmodel.cartesian.surfacefile.read_surface_file")):
        surfacefile.read_surface_file(on_plane)
    with pytest.raises(ValueError, match=(
            r"this reader reads 'planetmodel.surface/2'; a file of "
            r"'planetmodel.surface/1' is read by "
            r"planetmodel.surfacefile.read_surface_file")):
        flat_surfacefile.read_surface_file(on_sphere)
    with pytest.raises(TypeError, match="expected an AngularGrid"):
        surfacefile.write_surface(tmp_path / "x.nc", plane, source="tests",
                                  fields={"k": SurfaceVariable(np.ones(plane.shape))})
