# Examples

Two kinds of material, after MFEM's: **examples**, small scripts that
each show one idea in under a hundred lines or so, grouped by area
(`conversion/`, more areas to come as the tutorials are split), and
**mini-apps** (`miniapps/`), medium-sized programs that use several
parts of the library together on a real problem, each in a directory
with its README. Each starts with a header naming what it needs:

```
# Needs: plot, maps
# Network: first run (S20RTS, 0.1 MB)
# Run: poetry run python examples/conversion/03_sl3d_chain.py
```

`Needs` lists the extras, `Network` whether it fetches a data set, `Run`
how to run it; `tests/test_examples.py` reads these lines to choose
each one's markers, so the test suite runs every example it can.

| example | what it shows |
|---|---|
| `conversion/01_velocity_to_temperature.py` | a shear-speed anomaly on PREM as a temperature anomaly, by a sensitivity and through density as sl3d does, with sl3d's parameter profiles (`plot`) |
| `conversion/02_temperature_to_viscosity.py` | the exponential law's viscosity, sl3d's eps against Milne et al.'s, and Latychev et al.'s renormalisation (`plot`) |
| `conversion/03_sl3d_chain.py` | sl3d's chain on S20RTS, maps at three depths, and what the model cites (`maps`, network) |
| `conversion/04_prony_branches.py` | Maxwell, Burgers and a fitted Andrade body as Prony branches, their moduli and losses, and a Burgers mantle frozen at a 100-year period (`plot`) |
| `conversion/05_temperature_from_a_table.py` | a table of Vs against temperature and pressure, written and read, inverted for temperature, and a temperature anomaly from it on PREM (`plot`, `netcdf`) |
| `conversion/06_lithosphere_base.py` | the base of the lithosphere at a viscosity and at a Maxwell time, as geometry (an interface moved onto it) and as a lid, mapped (`maps`) |
| `conversion/07_published_viscosity.py` | Lloyd et al.'s viscosity from GLAD-M25, fetched, put on PREM's mantle and mapped at three depths (`netcdf`, `maps`, network) |
| `conversion/08_power_law.py` | power-law creep's stress exponent and transition stress on a model, and AdGIA's gamma from them |

| mini-app | what it does |
|---|---|
| `miniapps/viscosity3d/` | S40RTS (or a published viscosity model) to a 3D viscosity model file for a GIA code: a radial viscosity with an elastic lid, sl3d's chain or Milne's eps, optionally renormalised, a thicker lid where the viscosity stays high, power-law creep or a Burgers body, written, read back, read by the C++ example reader, drawn (`netcdf`, `maps`, network) |

`tutorials/` walks through the ideas of the library in order, one script
per idea. Each is a plain Python file with `# %%` cell markers, so it runs
as a script from any directory and cell by cell in an editor. Every
tutorial runs headless in a few seconds; figures and the files a
tutorial keeps go under `figures/`, which is gitignored.

Each script also has a notebook generated from it under
`tutorials/notebooks/`, which opens in Jupyter or, with nothing
installed, in Google Colab through the badges below. The scripts are the
source: edit one and run `tutorials/make_notebooks.py`, which rewrites
its notebook, and a test checks that the two agree.

| tutorial | what it shows | Colab |
|---|---|---|
| `01_skeleton_and_geometry.py` | a skeleton, changing it, a geometry with names, hollow geometries, scaling | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/01_skeleton_and_geometry.ipynb) |
| `02_an_analytic_mapping.py` | a radial stretch from an analytic displacement, `F` and `J`, validity, kinks, a non-radial mapping | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/02_an_analytic_mapping.ipynb) |
| `03_a_radial_mesh.py` | the GLL reference element, a radial mesh over a skeleton, per-element nodes, the exact polynomial view, truncation by degree, a figure (uses the `plot` extra when present) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/03_a_radial_mesh.ipynb) |
| `04_a_mesh_for_mfem.py` | a mesh of a skeleton and its manifest, then topography through a mapping as a physical mesh, then a buffer shell with the topography tapered across it, each delivered to MFEM; the files are kept under `figures/tutorial_04_meshes/` for gmsh and glvis, or discarded with `--temp` (needs the `meshing` and `mfem` extras) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/04_a_mesh_for_mfem.ipynb) |
| `05_fields.py` | fields as functions of position on one layer, radial fields and the layer functions they store, exact arithmetic, real and complex dtypes, tensors and frames, push-forward, a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/05_fields.ipynb) |
| `06_a_model_of_your_own.py` | a model from scratch: geometry, fields, specs, validation, conversion of units, changing the layering, a model type of your own, a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/06_a_model_of_your_own.ipynb) |
| `07_prem.py` | PREM from its polynomials: layers, moduli, gravity, nodal values on a radial mesh, a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/07_prem.ipynb) |
| `08_simple_models.py` | homogeneous, layered and ellipsoidal models with analytic boundary shapes, a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/08_simple_models.ipynb) |
| `09_fields_for_mfem.py` | a model's fields exported beside its mesh and read back by MFEM (needs the `meshing` and `mfem` extras) | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/09_fields_for_mfem.ipynb) |
| `10_random_fields.py` | Matern random fields of radius, on the layers of PREM as a density perturbation, and on a shell as a field with a map and a slice, a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/10_random_fields.ipynb) |
| `11_models_from_decks.py` | PREM's mineos deck as a model against the polynomial PREM, the shipped GIA deck format and a model type for it with a viscosity column, a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/11_models_from_decks.ipynb) |
| `12_a_laterally_varying_model.py` | PREM with a harmonic perturbation of the shear velocity and a log perturbation of a viscosity, written to a model file, read back and compared, and drawn as maps at one depth (needs the `netcdf` and `maps` extras), a figure | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/12_a_laterally_varying_model.ipynb) |
| `13_pictures.py` | pictures of a perturbed PREM: slices and maps at one depth, a side taken at an interface, cross-sections as wedges and a disc, the body drawn flattened, profiles, spectra, a comparison of two models by degree and depth, a cutaway with an octant removed, and files for ParaView with first steps in it (needs the `maps` extra), figures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/13_pictures.ipynb) |
| `14_published_models.py` | published models from the registry: S40RTS read as the model it is and drawn, as a map, a section and cutaways of the Earth (an octant removed, with the surface dressed as the field, the relief, land and sea or a plain colour, with coastlines; a wedge cut to the core; on PREM, cuts to the core-mantle boundary on either side, the inner-core boundary and the centre; a sphere within the body as the surface, the sections above it kept; discs of the equator and a meridian through a sphere at a depth), put onto PREM and filtered, S362ANI+M given as velocities and put onto PREM through its own reference, the two compared, a regional model of the North Atlantic merged in, ANT-20 (Antarctica, on a polar stereographic map) merged in and drawn at 100 km with West Antarctica slow and East Antarctica fast, CRUST1.0 put on PREM as geometry (the Moho's shift, a section across the Andes as the body is), LITHO1.0's lithosphere base, what the model cites, the result written to a model file, a model of the Moon (needs the network the first time, and the `netcdf` and `maps` extras), figures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/14_published_models.ipynb) |
| `15_kernels.py` | a sensitivity kernel as a code writes it (the file `cpp/examples/write_kernel.cpp` writes, here written by the Python, or read from the C++ example's output given its path): maps at depths on a scale centred on zero and clipped at a percentile, a section, profiles and the one-dimensional kernel, the column integral, spectra, a kernel on an interface, and files for ParaView (needs the `netcdf` and `maps` extras), figures | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/da380/planetmodel/blob/main/examples/tutorials/notebooks/15_kernels.ipynb) |

```
poetry run python examples/tutorials/01_skeleton_and_geometry.py
```

To run a tutorial cell by cell as a notebook, install the `notebook` extra
(`poetry install --extras notebook`), which provides the IPython kernel
an editor's interactive window uses.
