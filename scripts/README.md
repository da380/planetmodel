Manual runs, not tests. Scripts here build meshes at Earth scale, which
takes minutes and memory, and are never collected by `pytest`.

- `mfem_cross_check.py`: a PREM-shaped geometry with surface relief and a
  buffer shell, built at a chosen resolution and read back by PyMFEM.
- `cubed_sphere_prem.py`: PREM with all its layers as a cubed sphere of
  hexahedra, with a buffer shell and surface relief, built at a chosen
  number of cells an edge, exported with its density and read back by
  PyMFEM.
