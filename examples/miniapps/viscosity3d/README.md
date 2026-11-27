# viscosity3d

A 3D viscosity model for a GIA code, from a published tomographic model
to a model file the C++ codes read.

```
poetry run python examples/miniapps/viscosity3d/viscosity3d.py
poetry run python examples/miniapps/viscosity3d/viscosity3d.py --law milne --renormalise
poetry run python examples/miniapps/viscosity3d/viscosity3d.py --burgers --out burgers.nc
poetry run python examples/miniapps/viscosity3d/viscosity3d.py --lid 1e21 --power-law 3.5 1e6
poetry run python examples/miniapps/viscosity3d/viscosity3d.py --published GLAD-M25-viscosity
```

Needs the `netcdf` and `maps` extras, and the network the first time
(S40RTS, 3 MB, fetched into the data cache).

## What it does

1. **A radial model.** PREM without its ocean, given a viscosity of
   2e21 Pa s below 670 km and 5e20 Pa s above, up to the base of PREM's
   lid, 80 km deep. The lid and the crust hold no viscosity, so a GIA
   code treats them as elastic.
2. **S40RTS on the mantle.** `datasets.transplanted` puts the model's
   shear-speed anomaly on vsv and vsh of the viscous layers, without its
   degree 0, so the radial model keeps its own mean (`--lmax` filters it).
3. **A viscosity.** `conversion.building` reads the anomaly back, turns
   it into a temperature anomaly through sl3d's density scaling and
   expansivity, and writes the viscosity as a log perturbation of the
   radial one: eta = eta_0 exp(-eps dT), eps sl3d's 0.01 per kelvin or,
   with `--law milne`, the 0.04 of Milne et al. (2018);
   `--renormalise` keeps each depth's mean viscosity the radial one, as
   Latychev et al. (2005) do.
4. **Or a published viscosity model** (`--published`): a registry entry
   such as `GLAD-M25-viscosity` (Lloyd et al. 2024, 142 MB) or
   `SMEAN2-VM5a-b1` (Steffen & Steffen 2024, 92 MB), put on the mantle as
   a log perturbation of the radial viscosity (`transplanted(...,
   mode="absolute", kind="log")`), in place of steps 2 and 3.
5. **A thicker lid** (`--lid`). Going down from the lid's base, the depth
   within 400 km where the viscosity first falls to the value given
   (`conversion.lithosphere.viscosity_threshold`); above it the
   viscosity is raised to 1e30 Pa s (`with_lid`), so the old continents'
   roots join the lithosphere. Below the lid the converted viscosity
   runs from about 1e20 to 3e21 Pa s, so `--lid 1e21` puts the base
   between 80 and 400 km.
6. **Power-law creep** (`--power-law N SIGMA`): the stress exponent and
   the transition stress (Pa) on every viscous layer, for a code that
   steps in time (`conversion.building.with_power_law`).
7. **Prony branches** (`--burgers`). The viscosity replaced by a Burgers
   body (Kelvin modulus fraction 0.2 and viscosity ratio 5, the
   relaxation strength 5 and ratio 5 of Caron et al. 2017)
   as its two branches, `relaxation_strength_k` and `relaxation_time_k`,
   the times written as log perturbations.
8. **The model file.** Written on a Gauss-Legendre grid of band 96
   (`--band`), at the nodes of 50 km elements, read bicubically, with
   the works the model rests on in its `citation`; read back, and
   compared with the model at 500 points of the lower mantle.
9. **The C++ library** (`--cpp`). Given the path of the example reader
   `planetmodel_read_model`, the file is read by it too:

   ```
   cmake -S cpp -B build/cpp && cmake --build build/cpp -j 8
   poetry run python examples/miniapps/viscosity3d/viscosity3d.py \
       --cpp build/cpp/examples/planetmodel_read_model
   ```

10. **Maps** of log10 of the viscosity over the radial one at 250, 1000
   and 2500 km, written to `examples/figures/viscosity3d.png`.

## How finely to write it

S40RTS goes to degree 40, and the exponential of its anomaly is sharper
still. With the viscosity read back at 3000 random points of the mantle:

| grid band | lateral rule | elements | file | log10 eta, largest error |
|---|---|---|---|---|
| 64 | bilinear | 50 km | 42 MB | 0.15 |
| 64 | bicubic | 50 km | 42 MB | 0.014 |
| 96 | bicubic | 50 km | 92 MB | 0.0021 |
| 128 | bicubic | 50 km | 161 MB | 0.0007 |

Halving the elements to 25 km changes nothing, S40RTS being smooth in
radius. The default, band 96 bicubic, holds the viscosity to half a
percent. A published model given on nodes a degree apart reads back
within about 0.1 in log10 between them, at band 96 or 180 alike: its
neighbouring nodes differ by up to a factor of ten, and its own reading
(bilinear in the viscosity) and the file's (bicubic in its logarithm)
differ there; that is two interpolations disagreeing, not information
lost.
