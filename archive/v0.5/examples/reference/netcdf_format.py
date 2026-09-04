"""The netCDF model file, `planetmodel.model/1`.

`write_model(body, sample, path)` writes a body's sample -- the fields
on the radial mesh times the angular grid -- with enough metadata for a
consumer that has never heard of this library: the skeleton and layer
states, every field's character, dimensions and the layers it is
defined on (fill values elsewhere), the scales, the model class by
its registered name, and for every layer holding moduli a `/rheology`
record saying which law built its frequency-dependent moduli from
which fields and constants.  `read_model` restores the sample in full
and the body as the class the file names, rebuilding each law by
calling it again; a frequency-dependent field is never stored as
such, only a sample of it at a chosen frequency, as (real, imaginary).

This script writes PREM, walks the file with netCDF4, and reads it
back.  It needs the `netcdf` extra.
"""
import tempfile
import warnings
from pathlib import Path

import numpy as np

try:
    from netCDF4 import Dataset
except ImportError:
    print("netCDF4 is not installed (pip install 'planetmodel[netcdf]'); skipping")
    raise SystemExit(0)

from planetmodel import PREM, AngularGrid, read_model, write_model
from planetmodel.io.netcdf import SCHEMA

prem = PREM(ocean=False)
grid = AngularGrid.gauss_legendre(lmax=4)          # small: the layout is the point
sample = prem.sample(grid)
complex_sample = prem.sample(grid, omega=2.0 * np.pi / 100.0)   # the moduli at 100 s

with tempfile.TemporaryDirectory() as tmp:
    path = Path(tmp) / "prem.nc"
    write_model(prem, complex_sample, path)

    # -- the layout, as a consumer sees it -------------------------------------
    with Dataset(path) as ds:
        assert ds.schema == SCHEMA
        assert ds.model_class == "ViscoelasticModel"
        assert np.isclose(float(ds.reference_period), 1.0)     # the laws' period
        print("root:", ds.schema, ds.model_class, "period", ds.reference_period)
        print("groups:", list(ds.groups))
        sk = ds["skeleton"]
        assert sk["layer_state"][1] == "fluid"                  # the outer core
        print("layers:", list(sk["layer_state"][:]))
        fields = ds["fields"]
        rho = fields["rho"]
        assert rho.dimensions == ("node",)                       # radial: no angles
        assert tuple(rho.physical_dimensions) == (1, -3, 0)      # kg m^-3
        every_layer = "[" + ", ".join(map(str, range(prem.skeleton.nlayers))) + "]"
        assert rho.layers == every_layer
        print("rho:", rho.dimensions, "units", rho.units, "on layers", rho.layers)
        # The complex sample carries a trailing axis of length 2 and its omega.
        visco = fields["viscoelastic_moduli"]
        assert visco.dimensions[-1] == "part" and visco.shape[-1] == 2
        assert np.isclose(float(visco.omega), 2.0 * np.pi / 100.0)
        print("viscoelastic_moduli:", visco.dimensions, "at omega", visco.omega)
        # The rheology group: one row per layer, the law and its record.
        rheo = ds["rheology"]["viscoelastic_moduli"]
        laws = list(rheo["law"][:])
        assert set(laws) == {"constant_q"}
        print("laws:", laws[:3], "...")
        print("constants:", rheo["constants"][3],
              "dims", rheo["constant_dimensions"][3])
        assert "reference_period" in rheo["constants"][3]

    # -- reading back ------------------------------------------------------------
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")            # nothing to warn about here
        body, back = read_model(path)
    assert type(body).__name__ == "ViscoelasticModel"
    assert [lay.state for lay in body.layers] == [lay.state for lay in prem.layers]
    r = np.linspace(2.0e6, 6.0e6, 7)
    assert np.allclose(body["rho"].evaluate(r), prem["rho"].evaluate(r), rtol=1e-12)
    # The laws were rebuilt by name from the fields the file carries.
    assert body.layers[3]["viscoelastic_moduli"].law.law == "constant_q"
    omega = 2.0 * np.pi / 10.0
    assert np.allclose(body.viscoelastic_moduli.evaluate(r, omega=omega),
                       prem.viscoelastic_moduli.evaluate(r, omega=omega), rtol=1e-10)
    # The sample is restored in full, complex part and all.
    assert set(back.fields) == set(complex_sample.fields)
    assert back.metadata.omegas == complex_sample.metadata.omegas
    assert np.allclose(back.fields["viscoelastic_moduli"],
                       complex_sample.fields["viscoelastic_moduli"], equal_nan=True)

print("ok: the file is self-describing and the typed model comes back")
