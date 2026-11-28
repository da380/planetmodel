Scripts run by hand.  The two that build meshes do so at Earth scale,
which takes minutes and memory; they are not tests and are never
collected by `pytest`.

- `mfem_cross_check.py`: a PREM-shaped geometry with surface relief and a
  buffer shell, built at a chosen resolution and read back by PyMFEM.
- `cubed_sphere_prem.py`: PREM with all its layers as a cubed sphere of
  hexahedra, with a buffer shell and surface relief, built at a chosen
  number of cells an edge, exported with its density and read back by
  PyMFEM.
- `migrate_deck.py`: a Maxwell, prony or burgers deck as planetmodel 2
  wrote it (or a bare table, given its column names) written as a model
  file, with the names of its layers and interfaces and the behaviours
  of its model: `python scripts/migrate_deck.py earth.deck earth.nc`,
  and `--domain=x0,x1,y0,y1` for a deck of depths.
  `tests/test_migrate_deck.py` holds it to decks that version wrote.
