# Examples

`tutorials/` walks through the ideas of the library in order, one script
per idea. Each is a plain Python file with `# %%` cell markers, so it runs
as a script from any directory and cell by cell in an editor. Every
tutorial runs headless in a few seconds; figures and the files a
tutorial keeps go under `figures/`, which is gitignored.

| tutorial | what it shows |
|---|---|
| `01_skeleton_and_geometry.py` | a skeleton, its surgery, a geometry with names, hollow geometries, scaling |
| `02_an_analytic_mapping.py` | a radial stretch from an analytic displacement, `F` and `J`, validity, kinks, a non-radial mapping |
| `03_a_radial_mesh.py` | the GLL reference element, a radial mesh over a skeleton, per-element nodes, the exact polynomial view, truncation by degree, a figure (uses the `plot` extra when present) |
| `04_a_mesh_for_mfem.py` | a 2D and a 3D mesh of a geometry with a buffer shell, the manifest, export to MFEM; the files are kept under `figures/tutorial_04_meshes/` for gmsh and glvis, or discarded with `--temp` (needs the `meshing` and `mfem` extras) |
| `05_fields.py` | fields as functions of position on one layer, radial fields as the exact special case, the algebra, real and complex dtypes, tensors and frames, push-forward, a figure |
| `06_a_model_of_your_own.py` | a model from scratch: geometry, fields, specs and constants, validation, conversion of units, surgery, a model type of your own, a figure |
| `07_prem.py` | PREM from its polynomials: layers, moduli, gravity, nodal values on a radial mesh, a figure |
| `08_simple_models.py` | homogeneous, layered and ellipsoidal models with analytic boundary shapes, a figure |
| `09_fields_for_mfem.py` | a model's fields exported beside its mesh and read back by MFEM (needs the `meshing` and `mfem` extras) |
| `10_love_numbers.py` | PREM's load and tidal Love numbers, convergence, the degree-2 radial solutions, the pyslfp file, a Maxwell mantle at one frequency, a figure |
| `11_random_fields.py` | Matern random fields of radius, on the layers of PREM as a density perturbation, and on a shell as a field with a map and a slice, a figure |

```
poetry run python examples/tutorials/01_skeleton_and_geometry.py
```

To run a tutorial cell by cell as a notebook, install the `notebook` extra
(`poetry install --extras notebook`), which provides the IPython kernel
an editor's interactive window uses.
