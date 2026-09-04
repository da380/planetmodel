# Examples

Two tracks.

**`tutorials/`** are meant to be read. Each is a Python script in cell
format (`# %%` markers, prose in markdown cells), so it opens as a
notebook in VS Code or Jupyter and runs as a plain script from any
directory:

    python examples/tutorials/01_prem.py

| tutorial | what you do |
|---|---|
| `01_prem.py` | read PREM as the viscoelastic model it is; layers, states, views and domains; profiles; the moduli at a period and their dispersion; a netCDF file written and read back as the same class |
| `02_a_body_of_your_own.py` | a density from a table; a body from `Layer` values; a vacuum shell; surgery; a field type of your own checked by `check_field`; an `ElasticModel` and what it refuses |
| `03_topography_and_mapping.py` | the Moho of CRUST-1.0 as centred relief; an ellipsoidal boundary; the mapping and its validity; a density pushed forward; a physical elastic medium pulled back |
| `04_layered_rheology.py` | an elastic lithosphere over a Maxwell mantle over a fluid core; the moduli on a Laplace contour and at real frequencies; a Prony series; the law record in the file; a law of your own |
| `05_a_mesh_for_mfem.py` | a deck coarsened and given the Moho relief; a small 3D mesh and its manifest; the same build as a recipe; MFEM export (needs `planetmodel[meshing]`; `planetmodel[mfem]` for the export) |
| `06_love_numbers.py` | load Love numbers of PREM by radial spectral elements (`planetmodel.loading`, meant for pyslfp) |
| `07_random_fields.py` | Matern random fields of radius, of a shell, and on the layers of PREM (`planetmodel.randomfield`, meant for pygeoinf) |

**`reference/`** holds short scripts, one concept each, in the
numbers-and-checks style: the place to look up how one thing is done.

Every script runs headless (`MPLBACKEND=Agg`), writes figures to
`examples/figures/` and files to a temporary directory, and says so and
moves on when an optional extra (`netCDF4`, `gmsh`, PyMFEM) is missing.
CI runs them all.

Data: `prem.200` here is the PREM deck (mineos layout, SI); `tests/data/`
holds `prem.nocrust`, an isotropic deck ending at the Moho, and the
CRUST-1.0 Moho depth and crustal thickness grids (`lon lat value`, km).
