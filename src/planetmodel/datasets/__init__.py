"""Published models, fetched on demand and read into the library's own objects.

A data set is a model published elsewhere: a tomographic model, a crust
or lithosphere model, a radial reference, a surface of some body.  The
library keeps an entry for it, never its data: a `Dataset` names its
files (each a `Source`, with its URL, sha256 and mirrors, and for an
archive the members wanted), the reader that turns them into the object
the data set is, the model its perturbations are relative to, and its
title, citation and licence as the source states them.  `REGISTRY`
holds the shipped entries by name; a caller's own `Dataset` is used
exactly as a shipped one, passed where a name would go.

    fetch("S40RTS")              # paths of its files, cached and checked
    load("S40RTS")               # the data set as the object it is: a model
    describe("S40RTS"), cite("S40RTS")

A data set of perturbations reads as the model it is (`over_reference`):
the reference's geometry from the inner end of the perturbations'
support to the surface, each perturbed name a `PerturbedField` over the
reference's field on the layers the support covers, nothing beyond, so
that slices, pictures and the model file take it as it stands.  The
reference is the entry's (`REFERENCES` from the catalogue, or another
data set) or the caller's.  The Woodhouse-Ritsema `.sph` files (S20RTS,
S40RTS, S12RTS, SP12RTS) are read exactly, as harmonic fields on their
radial splines (`read_sph`, `sph_field`, `SphReader`); radial references
published as decks (STW105, ak135-F, iasp91, the Moon's models) read
through `planetmodel.deck` by `DeckReader`; netCDF grids of latitude and
longitude, EMC's among them, normalised by `read_grid` into a model of a
global grid (`grid_model`, `GridReader`) or a `RegionalGrid`, which also
holds a grid on a polar stereographic plane (`ProjectedData`,
`PolarStereographic`).  A
surface (a planet's shape, its topography, the radius of an interface
within it, an ice sheet's thickness) is a `Surface`, a function of the
angles with what it measures and from what datum, read from a table of
harmonic coefficients by `read_coefficients` (`CoefficientReader`) or
from a grid (`GridSurfaceReader`); a crustal or lithospheric model is
`Strata`, surfaces from the top down with the properties between them
(`read_crust1`, `litho1_strata`, a table's columns by `read_table`).  The
processing that fits them to a model of one's own (`rereferenced`,
`spread`, `filtered`, `merged`, one taper `raised_cosine`) is in
`planetmodel.datasets.processing`, and `transplanted` takes them in their
order to put a data set's perturbations on a model of one's own::

    transplanted(PREM(), "S40RTS", lmax=20)    # vsv, vsh perturbed; L, N with them

`with_strata` puts strata on a model, as geometry (chosen interfaces
moved to chosen surfaces, the layers between them filled) or as fields
(sampled through the chosen layers), and `relief` and `reliefs` give the
shifts of the interfaces alone; `moved` moves interfaces onto any
surfaces (a lithosphere's base found in a model's fields, say).

`fetch` downloads into `cache_directory()` (`PLANETMODEL_DATA`, or the
platform's per-user cache), checks each file's sha256 every time it is
used, refuses one that differs unless asked to keep it, never turns off
certificate checks, and leaves nothing behind when interrupted; a file
placed in the cache by hand is used as fetched.  `load` hands the paths
to the entry's reader.  Everything here is the standard library, numpy
and scipy; a reader needing an extra imports it where it is used.  The
sub-package is reached by module, off the top-level surface.
"""
from .building import (PAIRS, ROUTES, SurfaceOf, moved, relief, reliefs,
                       transplanted, with_strata)
from .decks import DeckReader, Names
from .entries import REGISTRY, DatasetLike, as_dataset, cite, describe
from .grids import (EARTH_RADIUS, SAME_RTOL, VERTICAL_RTOL, GridData, Gridded,
                    GridReader, ProjectedData, RegionalGrid, Variables, grid_model,
                    read_grid, read_table, ring_poles)
from .projections import ELLIPSOIDS, PolarStereographic
from .fetch import (CACHE_VARIABLE, OFFLINE_VARIABLE, TIMEOUT, cache_directory, fetch,
                    load, sha256_of)
from .models import (REFERENCES, Segments, over_reference, reference_field,
                     reference_of, standalone)
from .processing import (MODES, Perturbation, demeaned, filtered, merged,
                         raised_cosine, rereferenced, resampled, spread)
from .registry import ARCHIVES, Dataset, Reader, Source, archive_stem
from .sph import (MANTLE_SLOTS, SLOTS, SPH_RADII, SPLINE_KNOTS, SphFile, SphReader,
                  read_sph, sph_field, sph_splines)
from .coefficients import FORMATS, read_coefficients
from .strata import (CRUST1_STRATA, LITHO1_PROPERTIES, LITHO1_STRATA, Crust1Reader,
                     Litho1Reader, Property, Strata, Stratum, TableStrataReader,
                     litho1_strata, read_crust1)
from .surfaces import MEASURES, CoefficientReader, Datum, GridSurfaceReader, Surface

__all__ = ["Source", "Dataset", "Reader", "DatasetLike", "REGISTRY", "ARCHIVES",
           "as_dataset", "archive_stem", "describe", "cite", "cache_directory",
           "fetch", "load", "sha256_of", "CACHE_VARIABLE", "OFFLINE_VARIABLE",
           "TIMEOUT", "REFERENCES", "reference_of", "reference_field",
           "over_reference", "SphFile", "SphReader", "read_sph", "sph_field",
           "sph_splines", "SPLINE_KNOTS", "SPH_RADII", "SLOTS", "MANTLE_SLOTS",
           "DeckReader", "Names", "Segments", "standalone", "GridData", "GridReader",
           "RegionalGrid", "Variables", "grid_model", "read_grid", "EARTH_RADIUS",
           "SAME_RTOL", "VERTICAL_RTOL", "MODES", "Perturbation", "filtered",
           "merged", "raised_cosine", "rereferenced", "spread", "resampled",
           "demeaned", "transplanted", "PAIRS", "FORMATS", "read_coefficients",
           "MEASURES", "Datum", "Surface", "CoefficientReader", "GridSurfaceReader",
           "read_table", "ring_poles", "CRUST1_STRATA", "LITHO1_STRATA",
           "LITHO1_PROPERTIES", "Property", "Stratum", "Strata", "read_crust1",
           "litho1_strata", "Crust1Reader", "Litho1Reader", "TableStrataReader",
           "ProjectedData", "Gridded", "PolarStereographic", "ELLIPSOIDS", "ROUTES",
           "SurfaceOf", "relief", "reliefs", "with_strata", "moved"]
