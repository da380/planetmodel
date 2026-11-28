"""The shipped data sets, each entry written once, and the lookup by name.

`REGISTRY` maps each shipped entry's name to it, read-only; a caller's
own `Dataset` is used exactly as a shipped one, passed where a name
would go (`DatasetLike`).  Each entry states what its source states: the
original host first (mirrors after it), the sha256 of the file as
published, the licence in the source's own words, and every work the
source asks to be cited.  `describe` and `cite` give that text.
"""
from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from ..behaviours import BEHAVIOURS
from ..deck import MINEOS, ND, TVEL, mineos_constants, mineos_names, nd_names
from .decks import DeckReader
from .grids import EARTH_RADIUS, GridReader
from .registry import Dataset, Source
from .sph import SphReader
from .strata import Crust1Reader, Litho1Reader, TableStrataReader
from .surfaces import CoefficientReader, GridSurfaceReader

__all__ = ["DatasetLike", "REGISTRY", "as_dataset", "describe", "cite"]

#: A data set, or the name of a shipped one.
type DatasetLike = Dataset | str

#: Where the Woodhouse-Ritsema files are published, as one record.
_RTS = "https://zenodo.org/records/11212270/files"

#: SPECFEM3D_GLOBE's copies of S20RTS and S40RTS, at a release tag.
_SPECFEM = "https://raw.githubusercontent.com/SPECFEM/specfem3d_globe/v8.1.0/DATA"

#: The record of the Woodhouse-Ritsema files, to be cited beside each model.
_RTS_RECORD = ("Ritsema, J. and P. Koelemeijer (2026).  Tomographic filters for "
               "models S20RTS, S40RTS and SP12RTS.  Zenodo, "
               "doi:10.5281/zenodo.11212270")

#: What every entry of the Woodhouse-Ritsema files says of them.
_RTS_NOTES = ("Relative perturbations of the isotropic velocity, as fractions, "
              "between the CMB (3480 km) and the Moho (6346.6 km; 6346.619 km in "
              "Ritsema's code), relative to PREM's Voigt average; degree 0 is kept "
              "as the file holds it (Ritsema's maps leave it out).")

#: TauP's standard models, at a release tag.
_TAUP = ("https://raw.githubusercontent.com/crotwell/TauP/v3.2.1/src/main/resources/"
         "edu/sc/seis/TauP/StdModels")

#: TauP, cited where its files are used.
_TAUP_CITE = ("Crotwell, H. P., T. J. Owens and J. Ritsema (1999).  The TauP "
              "Toolkit: flexible seismic travel-time and ray-path utilities.  "
              "Seismol. Res. Lett. 70, 154-160, doi:10.1785/gssrl.70.2.154")

#: The licence of a model in TauP's repository, which states none of its own.
_TAUP_LICENCE = ("none stated for the model; the file is from TauP's repository, "
                 "LGPL-3.0")

#: The compilation of the Moon's models, and its archive's members.
_MOON = ("https://zenodo.org/records/3372489/files/"
         "SSR_2019_Garcia_Moon_Seismological_structure_supplementary_zenodo.zip")
_MOON_SHA = "dfdb5cb68db402d989e953f48f1a82cb8c4c7104d15897861f25e304ad4f3136"
_MOON_DIR = ("SSR_2019_Garcia_Moon_Seismological_structure_supplementary_zenodo/"
             "Sup_Material_SSR_Internal_Moon_Seismological_models/")
_MOON_CITE = ("Garcia, R. F., A. Khan, M. Drilleau, L. Margerin, T. Kawamura, D. Sun, "
              "M. A. Wieczorek, A. Rivoldini, C. Nunn, R. C. Weber, A. G. Marusiak, "
              "P. Lognonné, Y. Nakamura and P. Zhu (2019).  Lunar seismology: an "
              "update on interior structure models.  Space Sci. Rev. 215, 50, "
              "doi:10.1007/s11214-019-0613-y; the files at Zenodo, "
              "doi:10.5281/zenodo.3372489")
_MOON_NOTES = ("TauP's named discontinuities, joined linearly as TauP joins them; the "
               "body's radius is the deepest depth.")


#: Where the Earth Model Collaboration serves its netCDF files.
_EMC = "https://data.earthscope.org/archive/seismology/products/emc/netcdf"

#: The licence of an EMC file, stated in its DataCite record.
_EMC_LICENCE = "CC BY 4.0 (stated in the EMC's DataCite records)"


def _emc_cites(doi: str) -> tuple[str, str]:
    """What the EMC asks cited beside a model's paper: its page and itself."""
    return (f"EarthScope Data Services, Earth Model Collaboration: the model's page, "
            f"doi:{doi}; and the EMC, doi:10.17611/DP/EMC.1",
            "Trabant, C., A. R. Hutko, M. Bahavar, R. Karstens, T. Ahern and R. Aster "
            "(2012).  Data products at the IRIS DMC: stepping stones for research "
            "and other applications.  Seismol. Res. Lett. 83, 846-854, "
            "doi:10.1785/0220120032")


#: Earth2014's tables to degree 2160, at their host.
_EARTH2014 = "https://ddfe.curtin.edu.au/models/Earth2014/data_5min/shcs_to2160"

#: Earth2014, as its readme asks it cited.
_EARTH2014_CITE = ("Hirt, C. and M. Rexer (2015).  Earth2014: 1 arc-min shape, "
                   "topography, bedrock and ice-sheet models - available as gridded "
                   "data and degree-10,800 spherical harmonics.  Int. J. Appl. Earth "
                   "Obs. Geoinf. 39, 103-112, doi:10.1016/j.jag.2015.03.001")

#: What every Earth2014 entry says of its table.
_EARTH2014_NOTES = ("4 pi normalised, in metres, the degree-10,800 analysis cut at "
                    "degree 2160; its grids are in geodetic latitude, which the "
                    "table's colatitude is, so that read as the library's geocentric "
                    "colatitude a feature lies up to 0.19 degrees (21 km) from where "
                    "it is.  Mean sea level is taken as the sphere of the Earth's "
                    "mean radius, 6371 km.")

#: The licence of Earth2014, whose readme states none.
_EARTH2014_LICENCE = "none stated; the readme asks that the paper be cited"

#: The Mars and Moon shapes and their convention.
_SHAPES = "https://zenodo.org/records"
_SHAPE_NOTES = ("4 pi normalised without the Condon-Shortley phase, in metres, of "
                "the radius; to degree 719 (the record holds 1439, 2879 and 5759 "
                "too, for a caller's own entry).")

#: GRAIL's crustal thickness archive, and the directory of its members.
_GRAIL = "https://zenodo.org/records/997347/files/GRAILCrustalThicknessArchive.zip"
_GRAIL_DIR = "GRAILCrustalThicknessArchive/Data/"


#: CRUST1.0's own files, in its tarball.
_CRUST1 = "https://igppweb.ucsd.edu/~gabi/crust1/crust1.0.tar.gz"

#: What the CAM entries say of their thickness.
_CAM_NOTES = ("The thickness of the thermal lithosphere in km, taken here as "
              "below the sphere of the Earth's mean radius, 6371 km: with_datum "
              "measures it from another surface (Earth2014-BED, say).")


def _earth2014(layer: str, measures: str, title: str, *, datum: float | None,
               sha256: str, notes: str) -> Dataset:
    """An entry of one of Earth2014's tables to degree 2160."""
    return Dataset(f"Earth2014-{layer}",
                   (Source(f"{_EARTH2014}/Earth2014.{layer}2014.degree2160.bshc",
                           sha256),),
                   CoefficientReader(format="bshc", measures=measures, datum=datum,
                                     name=f"Earth2014 {layer}"),
                   title=title, citation=(_EARTH2014_CITE,),
                   licence=_EARTH2014_LICENCE,
                   notes=" ".join((notes, _EARTH2014_NOTES)))


def _moon(name: str, member: str, title: str, cite: str, *,
          notes: str = "") -> Dataset:
    """An entry of one model of the Moon from the compilation's archive."""
    return Dataset(name, (Source(_MOON, _MOON_SHA, members=(_MOON_DIR + member,)),),
                   DeckReader(ND, names=nd_names, kind="linear"), title=title,
                   citation=(cite, _MOON_CITE), licence="CC BY 4.0", body="Moon",
                   notes=" ".join(filter(None, (_MOON_NOTES, notes))))


#: GLAD-M25, as its paper is cited.
_GLAD = ("Lei, W., Y. Ruan, E. Bozdağ, D. Peter, M. Lefebvre, D. Komatitsch, J. Tromp, "
         "J. Hill, N. Podhorszki and D. Pugmire (2020).  Global adjoint tomography: "
         "model GLAD-M25.  Geophys. J. Int. 223, 1-21, doi:10.1093/gji/ggaa253")

#: Steffen & Steffen's ten models: (background profile, beta profile) and sha256.
_SMEAN2 = "https://zenodo.org/records/13907989/files"
_SMEAN2_SHA = {
    ("K05", 1): "06d4405ee2b8aa7be7407412c53988f5c6905f028ecc64ae6aa776829387c49e",
    ("K05", 4): "186bae58d544f37029f243ac0cdc8442f2e1b5cf641c2cfa041cf9577f654b96",
    ("OM13", 1): "47cd7127c021f57d9d1b2ef512985610c53e8193a423385363626d5673fb0569",
    ("OM13", 4): "c9020a7f27878ad9555b353a70b76c37655be70faef49a95693c5721fb164fc9",
    ("W12back", 1): "abdcfb81b03c451ff2791673fc66bca575cf1e702602d093d3ead157c8dd0bf2",
    ("W12back", 4): "add8316a377c0805b45cd52d3d56cbbd1a9323482e27bd2e1c16608992f761b4",
    ("VM2", 1): "3efcaf27cb3fc2fdfba174c4cb5189f615789b8979650eb71faa045e2be876a5",
    ("VM2", 4): "c946c8e00092b11dfe33f2510c99e1cab0fb1ad1284783b43c6ebd0d278f9dd8",
    ("VM5a", 1): "7d8a8c80d6a09d65b15fe7e560e3c832b41fa21ff2a7f92a019f0b8402cf9c66",
    ("VM5a", 4): "6b6b583227866e47f336212b0fe53c78eb900ce5c2e47520ad554dc495be8fa6",
}

#: Steffen & Steffen's beta profiles, as their readme states them.
_SMEAN2_BETA = {1: "0.65 to 665 km, 0.75 to 1200 km, 1 below",
                4: "0.8 to 1200 km, 1 below"}

_ENTRIES: tuple[Dataset, ...] = (
    Dataset(
        "S20RTS",
        (Source(f"{_RTS}/S20RTS.sph",
                "642d5c826950a61d7ae324fbc9b58844d2e96f7d5277b232f5acbd1d428dcbe2",
                mirrors=(f"{_SPECFEM}/s20rts/S20RTS.sph",)),),
        SphReader(),
        title="A degree-20 shear-velocity model of the mantle",
        citation=("Ritsema, J., H. J. van Heijst and J. H. Woodhouse (1999).  "
                  "Complex shear wave velocity structure imaged beneath Africa and "
                  "Iceland.  Science 286, 1925-1928, "
                  "doi:10.1126/science.286.5446.1925", _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM", notes=_RTS_NOTES),
    Dataset(
        "S40RTS",
        (Source(f"{_RTS}/S40RTS.sph",
                "95cd2ee3fb523189f52d2e8ae8ef822628d5e69b283cb1ffe8b29a6577750cf9",
                mirrors=(f"{_SPECFEM}/s40rts/s40rts.sph",)),),
        SphReader(),
        title="A degree-40 shear-velocity model of the mantle",
        citation=("Ritsema, J., A. Deuss, H. J. van Heijst and J. H. Woodhouse "
                  "(2011).  S40RTS: a degree-40 shear-velocity model for the mantle "
                  "from new Rayleigh wave dispersion, teleseismic traveltime and "
                  "normal-mode splitting function measurements.  Geophys. J. Int. "
                  "184, 1223-1236, doi:10.1111/j.1365-246X.2010.04884.x",
                  _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM", notes=_RTS_NOTES),
    Dataset(
        "S12RTS",
        (Source(f"{_RTS}/S12RTS.sph",
                "2cbd52fe7eb15fcbaf5cfdbb3c13291b23c4e1252ba664916c59b980d938dc48"),),
        SphReader(),
        title="A degree-12 shear-velocity model of the mantle",
        citation=("Koelemeijer, P., J. Ritsema, A. Deuss and H. J. van Heijst "
                  "(2016).  SP12RTS: a degree-12 model of shear- and "
                  "compressional-wave velocity for Earth's mantle.  Geophys. J. "
                  "Int. 204, 1024-1039, doi:10.1093/gji/ggv481", _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM", notes=_RTS_NOTES),
    Dataset(
        "SP12RTS",
        (Source(f"{_RTS}/SP12RTS..ES.sph",
                "ceb039fef9e19c45c4ebb98b62abb95b1cd9f51e0a56984304acfac693667d6c"),
         Source(f"{_RTS}/SP12RTS..EP.sph",
                "4575805bff1714568b7240715dc7168b8aefd305da5a9cee537b57e8fbfa1baf")),
        SphReader(parameters=("vs", "vp")),
        title="A degree-12 model of shear and compressional velocity of "
              "the mantle",
        citation=("Koelemeijer, P., J. Ritsema, A. Deuss and H. J. van Heijst "
                  "(2016).  SP12RTS: a degree-12 model of shear- and "
                  "compressional-wave velocity for Earth's mantle.  Geophys. J. "
                  "Int. 204, 1024-1039, doi:10.1093/gji/ggv481", _RTS_RECORD),
        licence="CC BY 4.0", reference="PREM",
        notes=_RTS_NOTES + "  The S part (..ES) is vs, the P part (..EP) vp."),
    Dataset(
        "STW105",
        (Source("https://zenodo.org/records/8357380/files/STW105",
                "ccc2060674502b8032ef7bb11d2ea08fd4c8532f19455d62a784907bb5588352"),),
        DeckReader(MINEOS, behaviours=BEHAVIOURS, names=mineos_names,
                   constants=mineos_constants),
        title="The radial reference of S362ANI, S362ANI+M and GLAD-M25 (REF)",
        citation=("Kustowski, B., G. Ekström and A. M. Dziewoński (2008).  "
                  "Anisotropic shear-wave velocity structure of the Earth's mantle: "
                  "a global model.  J. Geophys. Res. 113, B06306, "
                  "doi:10.1029/2007JB005169",
                  "Moulik, P. and G. Ekström (2014).  An anisotropic shear velocity "
                  "model of the Earth's mantle using normal modes, body waves, "
                  "surface waves and long-period waveforms.  Geophys. J. Int. 199, "
                  "1713-1738, doi:10.1093/gji/ggu356; the files at Zenodo, "
                  "doi:10.5281/zenodo.8357380"),
        licence="none stated",
        notes="A mineos deck in SI, its discontinuities repeated radii; its header "
              "names the core and gives the reference period.  EMC's copy moves the "
              "upper side of each discontinuity up by 0.1 m and is not this file."),
    Dataset(
        "ak135-F",
        (Source(f"{_TAUP}/ak135favg.nd",
                "d08d19dbaad8d90a51c663011239cc45ad48ea2ed9b4ec9e7647b1a196f1eb19"),),
        DeckReader(ND, names=nd_names, kind="linear"),
        title="ak135 with density and attenuation, the average Earth",
        citation=("Kennett, B. L. N., E. R. Engdahl and R. Buland (1995).  "
                  "Constraints on seismic velocities in the Earth from traveltimes.  "
                  "Geophys. J. Int. 122, 108-124, "
                  "doi:10.1111/j.1365-246X.1995.tb03540.x",
                  "Montagner, J.-P. and B. L. N. Kennett (1996).  How to reconcile "
                  "body-wave and normal-mode reference Earth models.  Geophys. J. "
                  "Int. 125, 229-248, doi:10.1111/j.1365-246X.1996.tb06548.x",
                  _TAUP_CITE),
        licence=_TAUP_LICENCE,
        notes="TauP's ak135favg.nd: the average of the oceanic and continental "
              "forms, with its 3 km ocean, joined linearly as TauP joins it; its "
              "Q columns are TauP's Qp (the published table's 'Q kappa') and Qs "
              "(qmu)."),
    Dataset(
        "iasp91",
        (Source(f"{_TAUP}/iasp91.tvel",
                "c16b31d1eeae292e857a4224ac98719abcad7aff5c0068204fcadeb2868dd6f2"),),
        DeckReader(TVEL, kind="linear"),
        title="The IASPEI 1991 seismological tables' velocity model",
        citation=("Kennett, B. L. N. and E. R. Engdahl (1991).  Traveltimes for "
                  "global earthquake location and phase identification.  Geophys. "
                  "J. Int. 105, 429-465, doi:10.1111/j.1365-246X.1991.tb06724.x",
                  _TAUP_CITE),
        licence=_TAUP_LICENCE,
        notes="TauP's iasp91.tvel, joined linearly; its density column is not "
              "part of the published model and its source is not stated."),
    Dataset(
        "S362ANI+M",
        (Source(f"{_EMC}/S362ANI%2BM_kmps.r0.0-n4c.nc",
                "7bf5076e70ab9cc59b6d0f5a20d513c34d9309eecd33873b9d30e2b47188b425"),),
        GridReader({"vs": ("vs", None), "vsv": ("vsv", None), "vsh": ("vsh", None)}),
        title="A radially anisotropic shear-velocity model of the mantle, on EMC's "
              "2-degree grid",
        citation=("Moulik, P. and G. Ekström (2014).  An anisotropic shear velocity "
                  "model of the Earth's mantle using normal modes, body waves, "
                  "surface waves and long-period waveforms.  Geophys. J. Int. 199, "
                  "1713-1738, doi:10.1093/gji/ggu356",
                  *_emc_cites("10.17611/DP/9991784")),
        licence=_EMC_LICENCE, reference="STW105",
        notes="Absolute velocities in km/s at 25 depths from 25 to 2890 km on a "
              "2-degree grid with both poles, from the native model on spherical "
              "splines; relative to STW105 as published."),
    Dataset(
        "SEMUCB-WM1",
        (Source(f"{_EMC}/semucb-2014-ucb-vs-viz-only.r0.1-n4c.nc",
                "92b191f909c05e32151044695ff2be956ca25dfa0f62470f38c0ea976857cd43"),),
        GridReader({"vs": ("vs", None)}),
        title="A whole-mantle shear-velocity model from spectral-element waveform "
              "tomography, on EMC's grid",
        citation=("French, S. W. and B. A. Romanowicz (2014).  Whole-mantle radially "
                  "anisotropic shear velocity structure from spectral-element "
                  "waveform tomography.  Geophys. J. Int. 199, 1303-1327, "
                  "doi:10.1093/gji/ggu334",
                  *_emc_cites("10.17611/dp/emc.2023.semucbwm1.1")),
        licence=_EMC_LICENCE,
        notes="Voigt-average vs in km/s at 74 depths from 30 to 2891 km, on a grid of "
              "about 0.99 degrees with a short last interval (a custom grid with "
              "both poles); EMC calls this grid one for viewing.  Its own 1D "
              "reference is not shipped, so it is read as values."),
    Dataset(
        "GLAD-M25",
        (Source(f"{_EMC}/glad-m25-vs.r0.0-n4c.nc",
                "8916f703b07b2c23587dc8f6a7eddd89505d5a5491bc6712cd9ca70fe1e23d93"),),
        GridReader({"vsv": ("vsv", None), "vsh": ("vsh", None)}, stride=2,
                   vertical=(50.0, None), seam="mean"),
        title="A global model from adjoint tomography, its shear velocities on EMC's "
              "half-degree grid",
        citation=(_GLAD, *_emc_cites("10.17611/dp/emc.2020.gladm2500.1")),
        licence=_EMC_LICENCE, reference="STW105",
        notes="vsv and vsh in km/s at 342 depths from -5 to 2891 km on a half-degree "
              "grid with both poles (505 MB), discontinuities at 410 and 650 km.  Read "
              "by default from 50 km down, every second node (one degree, about 1.3 "
              "GB as a model): its crust, with zeros in water and air, differs by up "
              "to 3.2 km/s between the columns at -180 and 180 degrees, which differ "
              "by at most 0.04 % from 50 km down, where they are averaged.  "
              "load(..., stride=1) reads the half-degree grid.  Its starting model's "
              "reference is STW105."),
    Dataset(
        "SGLOBE-rani",
        (Source(f"{_EMC}/SGLOBE-rani-voigt_percent.r0.0-n4c.nc",
                "78a90fa796659ae99539612126c8e4ec72ebf9c3d3dad1f0c45a754319b32113"),),
        GridReader({"dvs": ("vs", "relative")}),
        title="A joint model of isotropic and radially anisotropic mantle structure, "
              "its Voigt-average shear velocity on EMC's 2-degree grid",
        citation=("Chang, S.-J., A. M. G. Ferreira, J. Ritsema, H. J. van Heijst and "
                  "J. H. Woodhouse (2015).  Joint inversion for global isotropic and "
                  "radially anisotropic mantle structure including crustal thickness "
                  "perturbations.  J. Geophys. Res. Solid Earth 120, 4278-4300, "
                  "doi:10.1002/2014JB011824",
                  *_emc_cites("10.17611/dp/emc2015sgloberani")),
        licence=_EMC_LICENCE, reference="PREM",
        notes="The relative perturbation of the Voigt-average vs, in percent, at 37 "
              "depths from 25 to 2800 km, latitudes -88 to 88 every 2 degrees (the "
              "poles by the nearest ring); relative to PREM."),
    Dataset(
        "GyPSuM",
        (Source(f"{_EMC}/GYPSUMS_kmps.r0.0-n4c.nc",
                "acb1e8d9563dfdeffb016ededdbba9890ceab0efefb0139554856d4c97799cc5"),),
        GridReader({"vs": ("vs", None)}, seam="mean"),
        title="A joint tomographic model of mantle density and wave speeds: its "
              "shear velocity on EMC's 1-degree grid",
        citation=("Simmons, N. A., A. M. Forte, L. Boschi and S. P. Grand (2010).  "
                  "GyPSuM: a joint tomographic model of mantle density and seismic "
                  "wave speeds.  J. Geophys. Res. 115, B12310, "
                  "doi:10.1029/2010JB007631", *_emc_cites("10.17611/DP/9991624")),
        licence=_EMC_LICENCE,
        notes="Absolute vs in km/s, a layered model sampled at the top and bottom of "
              "each block 1 km apart (read linearly, a step a 1 km ramp).  Its "
              "columns at -180 and 180 degrees differ by up to 1.1 %, the source's "
              "gridding not being periodic; they are averaged.  The file's id says "
              "percent; it holds km/s."),
    Dataset(
        "CSEM-North-Atlantic",
        (Source(f"{_EMC}/csem-north-atlantic-2019.12.01.r0.0-n4c.nc",
                "d4151db9dbfdacbb90086140aea86f528225730eab7051f2ac3c731dbd8f1f04"),),
        GridReader({"vsv": ("vsv", None), "vsh": ("vsh", None)}),
        title="The North Atlantic of the Collaborative Seismic Earth Model, a regional "
              "grid",
        citation=("Fichtner, A., D.-P. van Herwaarden, M. Afanasiev, S. Simutė, L. "
                  "Krischer, Y. Çubuk-Sabuncu, T. Taymaz, L. Colli, E. Saygin, A. "
                  "Villaseñor, J. Trampert, P. Cupillard, H.-P. Bunge and H. Igel "
                  "(2018).  The Collaborative Seismic Earth Model: generation 1.  "
                  "Geophys. Res. Lett. 45, 4007-4016, doi:10.1029/2018GL077338",
                  "Krischer, L., A. Fichtner, C. Boehm and H. Igel (2018).  Automated "
                  "large-scale full seismic waveform inversion for North America and "
                  "the North Atlantic.  J. Geophys. Res. Solid Earth 123, 5902-5928, "
                  "doi:10.1029/2017JB015289",
                  "Rickers, F., A. Fichtner and J. Trampert (2013).  The Iceland-Jan "
                  "Mayen plume system and its impact on mantle dynamics in the North "
                  "Atlantic region: evidence from full-waveform inversion.  Earth "
                  "Planet. Sci. Lett. 367, 39-51, doi:10.1016/j.epsl.2013.02.022",
                  *_emc_cites("10.17611/dp/emccsemnatl20191201")),
        licence=_EMC_LICENCE,
        notes="vsv and vsh in km/s from the surface to 500 km, latitudes 20 to 75 and "
              "longitudes -55 to 10 every half degree: a regional grid, for merging "
              "into a global model."),
    _moon("VPREMOON", "MoonGarcia_PEPI_2011.nd",
          "VPREMOON, a very preliminary reference Moon model",
          "Garcia, R. F., J. Gagnepain-Beyneix, S. Chevrot and P. Lognonné (2011).  "
          "Very preliminary reference Moon model.  Phys. Earth Planet. Inter. 188, "
          "96-113, doi:10.1016/j.pepi.2011.06.015"),
    _moon("MoonWeber2011", "MoonWeber_Science_2011.nd",
          "A seismic model of the Moon's core",
          "Weber, R. C., P.-Y. Lin, E. J. Garnero, Q. Williams and P. Lognonné "
          "(2011).  Seismic detection of the lunar core.  Science 331, 309-312, "
          "doi:10.1126/science.1199375"),
    _moon("ISSI-MOON-M1", "ISSI_MOON_M1.nd",
          "The ISSI lunar team's model M1, the median of an ensemble",
          "Garcia et al. (2019), as below: the ISSI lunar team's models",
          notes="Its crust-mantle boundary is marked for the codes that need one; "
                "the ensemble does not define it."),
    _moon("ISSI-MOON-M2", "ISSI_MOON_M2.nd",
          "The ISSI lunar team's model M2, the median of an ensemble",
          "Garcia et al. (2019), as below: the ISSI lunar team's models"),
    _moon("ISSI-MOON-M3", "ISSI_MOON_M3.nd",
          "The ISSI lunar team's model M3, the median of an ensemble",
          "Garcia et al. (2019), as below: the ISSI lunar team's models",
          notes="Its crust-mantle boundary is marked for the codes that need one; "
                "it varies across the ensemble."),
    _earth2014("SUR", "elevation", "The Earth's surface: land, lakes, ice and the "
               "sea surface", datum=6371e3,
               sha256="5694260d135c17427270ed18d48af23f4788e5fbc1dfb9dcb19f1cf9b401c9ce",
               notes="The physical surface: the topography of land, the surface of "
                     "major lakes and ice sheets, and zero over the oceans."),
    _earth2014("BED", "elevation", "The Earth's bedrock, without water and ice",
               datum=6371e3,
               sha256="146dcc80f17d201352d391aa90f487f5ed16006a6a3966add2d023b998727af7",
               notes="The bedrock: the topography of land and the floors of oceans, "
                     "lakes and ice sheets."),
    _earth2014("ICE", "thickness", "The thickness of the Earth's major ice sheets",
               datum=None,
               sha256="04cd185cc668eba6f9bd1db527c4985703cce8d1a9fb993509e625e2bbecc78e",
               notes="The thickness of the major ice sheets, zero elsewhere, below "
                     "the ice's top, which is Earth2014-SUR there: give it that "
                     "datum (with_datum) to place its base."),
    Dataset(
        "Mars-MOLA-shape",
        (Source(f"{_SHAPES}/10820719/files/Mars_MOLA_shape_719.bshc.gz",
                "d24497a57476bb24c9905886637a9ab53518c4b970c5325858007d72d3e2e79e"),),
        CoefficientReader(format="bshc", measures="radius", name="Mars MOLA shape"),
        title="The shape of Mars from the Mars Orbiter Laser Altimeter",
        citation=("Wieczorek, M. A. (2024).  Spherical harmonic models of the shape "
                  "of Mars [MOLA], version 1.0.1.  Zenodo, "
                  "doi:10.5281/zenodo.10820719",
                  "Smith, D. E., M. T. Zuber, H. V. Frey, J. B. Garvin and others "
                  "(2001).  Mars Orbiter Laser Altimeter: experiment summary after "
                  "the first year of global mapping of Mars.  J. Geophys. Res. 106, "
                  "23689-23722, doi:10.1029/2000JE001364"),
        licence="CC BY 4.0", body="Mars", notes=_SHAPE_NOTES),
    Dataset(
        "Moon-LOLA-shape",
        (Source(f"{_SHAPES}/10820774/files/Moon_LOLA_shape_719.bshc.gz",
                "20cafa143fda3e5ba5af6868c9e95054fabf68d0050c4be64a83d44df1333b4c"),),
        CoefficientReader(format="bshc", measures="radius", name="Moon LOLA shape"),
        title="The shape of the Moon from the Lunar Orbiter Laser Altimeter",
        citation=("Wieczorek, M. A. (2024).  Spherical harmonic models of the shape "
                  "of the Moon [LOLA], version 1.0.1.  Zenodo, "
                  "doi:10.5281/zenodo.10820774",
                  "Smith, D. E., M. T. Zuber, G. A. Neumann, F. G. Lemoine and others "
                  "(2010).  Initial observations from the Lunar Orbiter Laser "
                  "Altimeter (LOLA).  Geophys. Res. Lett. 37, L18204, "
                  "doi:10.1029/2010GL043751"),
        licence="CC BY 4.0", body="Moon",
        notes=_SHAPE_NOTES + "  In the DE421 mean Earth/polar axis frame."),
    Dataset(
        "GRAIL-crust",
        (Source(_GRAIL,
                "4ea89cac2564f666c0eed5300c6d4852b43c284d46f80631d0bdd5710353ad2c",
                members=tuple(f"{_GRAIL_DIR}Model{k}_cmi.sh" for k in range(1, 5))),),
        CoefficientReader(format="shtools", measures="radius",
                          name="GRAIL crust-mantle interface"),
        title="The radius of the Moon's crust-mantle interface from GRAIL",
        citation=("Wieczorek, M. A., G. A. Neumann, F. Nimmo, W. S. Kiefer and "
                  "others (2013).  The crust of the Moon as seen by GRAIL.  Science "
                  "339, 671-675, doi:10.1126/science.1231530",
                  "Wieczorek, M. A. (2012).  GRAIL crustal thickness archive.  "
                  "Zenodo, doi:10.5281/zenodo.997347"),
        licence="BSD-3-Clause", body="Moon",
        notes="The radius of the crust-mantle interface of models 1 to 4 of the "
              "archive (which=0 to 3, or the file's name, Model1_cmi.sh ...; "
              "average crustal thicknesses 34, 35, 43 and 43 km), to degree 310, "
              "SHTOOLS text in metres, 4 pi normalised without the Condon-Shortley "
              "phase, in the principal-axis frame.  The crustal thickness is the "
              "archive's LOLA surface (in the same frame) less this radius; the "
              "archive's thickness files are not read."),
    Dataset(
        "CRUST1.0",
        (Source(_CRUST1,
                "0b41b46fc3e1a76debbbcb66ab407febaeee4dc3033db7a0a24d2bb7c7adfe3e",
                members=("crust1.bnds", "crust1.vp", "crust1.vs", "crust1.rho")),),
        Crust1Reader(),
        title="A 1-degree global model of the Earth's crust",
        citation=("Laske, G., G. Masters, Z. Ma and M. Pasyanos (2013).  Update on "
                  "CRUST1.0 - a 1-degree global model of Earth's crust.  Geophys. "
                  "Res. Abstracts 15, EGU2013-2658",),
        licence="none stated",
        notes="Eight strata (water, ice, three sediments, three crystalline crust) "
              "on 1-degree cells, their boundaries as elevations from sea level "
              "(the sphere of 6371 km), vp, vs and rho constant in each, and the "
              "mantle's beneath the Moho; an absent stratum has zero thickness.  "
              "The readme asks that its prototype status be borne in mind."),
    Dataset(
        "LITHO1.0",
        (Source(f"{_EMC}/LITHO1.0.r0.0-n4c.nc",
                "ba13dfe1b7685d7ab07aa201590568f48b79fd3109e7ba17c01dded2892c375f"),),
        Litho1Reader(),
        title="An updated crust and lithospheric model of the Earth",
        citation=("Pasyanos, M. E., T. G. Masters, G. Laske and Z. Ma (2014).  "
                  "LITHO1.0: an updated crust and lithospheric model of the Earth.  "
                  "J. Geophys. Res. Solid Earth 119, 2153-2173, "
                  "doi:10.1002/2013JB010626",
                  *_emc_cites("10.17611/dp/emc.2020.litho102nc.1")),
        licence=_EMC_LICENCE,
        notes="The Earth Model Collaboration's 1-degree grid of the model (natively "
              "on 40962 icosahedral nodes): CRUST1.0's strata and the lid, whose "
              "bottom is the lithosphere-asthenosphere boundary, as depths below "
              "sea level, each property (rho, vpv, vsv, vph, vsh, qkappa, qmu, eta) "
              "at each stratum's top and bottom, and the asthenosphere's beneath; "
              "the grid's glitched latitudes rebuilt.  In 273 of its 64800 cells "
              "a stratum's bottom lies below the next one's top (median 34 m, at "
              "most 2.6 km): the next top is taken."),
    Dataset(
        "CAM2016",
        (Source(f"{_EMC}/CAM2016Litho.r0.0-n4c.nc",
                "52b52d36f8b47078a097c2d1a32943b6c0d46c45903fb9f9be778490f649fdad"),),
        GridSurfaceReader("thickness", measures="thickness", datum=EARTH_RADIUS),
        title="The thickness of the lithosphere from multimode surface waves",
        citation=("Priestley, K., D. McKenzie and T. Ho (2018).  A lithosphere-"
                  "asthenosphere boundary - a global model derived from multimode "
                  "surface-wave tomography and petrology.  In Lithospheric "
                  "Discontinuities, Geophys. Monogr. 239, 111-123, "
                  "doi:10.1002/9781119249740.ch6",
                  "Ho, T., K. Priestley and E. Debayle (2016).  A global horizontal "
                  "shear velocity model of the upper mantle from multimode Love wave "
                  "measurements.  Geophys. J. Int. 207, 542-561, "
                  "doi:10.1093/gji/ggw292",
                  "Priestley, K. and D. McKenzie (2013).  The relationship between "
                  "shear wave velocity, temperature, attenuation and viscosity in "
                  "the shallow part of the mantle.  Earth Planet. Sci. Lett. 381, "
                  "78-91, doi:10.1016/j.epsl.2013.08.022",
                  *_emc_cites("10.17611/DP/EMCCAM2016")),
        licence=_EMC_LICENCE,
        notes="2-degree cells.  " + _CAM_NOTES),
    Dataset(
        "CAM2022",
        (Source(f"{_EMC}/CAM2022-lithosphere.r0.1-n4c.nc",
                "1d43f8a7b58482da78a976fa998cd9050db6878c7ddfbf082b102fdaab027de9"),),
        GridSurfaceReader("thickness", measures="thickness", datum=EARTH_RADIUS),
        title="The thickness of the lithosphere from the thermal and anisotropic "
              "structure of the upper mantle",
        citation=("Priestley, K., T. Ho, Y. Takei and D. McKenzie (2024).  The "
                  "thermal and anisotropic structure of the top 300 km of the "
                  "mantle.  Earth Planet. Sci. Lett. 626, 118525, "
                  "doi:10.1016/j.epsl.2023.118525",
                  *_emc_cites("10.17611/dp/emc.2024.cam2022.1")),
        licence=_EMC_LICENCE,
        notes="2-degree cells.  " + _CAM_NOTES),
    Dataset(
        "LithoRef18",
        (Source("https://www.earthbyte.org/webdav/ftp/earthbyte/gplates_portal/"
                "Afonso_etal_lithospheric_thickness_GJI2019/"
                "Afonso_etal_GJI_2019_LithoRef18.xyz",
                "645dbe3154866d4db3dd465b439a540a8dcde24489e6ac8849980109b99a927e"),),
        TableStrataReader(surfaces=("ELEVATION", "MOHO", "LAB"),
                          names=("crust", "lithospheric_mantle"), measures="elevation",
                          properties={"crust": {"rho": "RHO_C"},
                                      "lithospheric_mantle": {"rho": "RHO_L"}},
                          below={"rho": "RHO_SL"},
                          units={"ELEVATION": "m", "MOHO": "m", "LAB": "m",
                                 "RHO_C": "kg m-3", "RHO_L": "kg m-3",
                                 "RHO_SL": "kg m-3"}),
        title="A global reference model of the lithosphere and upper mantle",
        citation=("Afonso, J. C., F. Salajegheh, W. Szwillus, J. Ebbing and C. Gaina "
                  "(2019).  A global reference model of the lithosphere and upper "
                  "mantle from joint inversion and analysis of multiple data sets.  "
                  "Geophys. J. Int. 217, 1602-1628, doi:10.1093/gji/ggz094",),
        licence="none stated",
        notes="The crust from the solid surface (the sea floor in the oceans) to "
              "the Moho, and the lithospheric mantle from the Moho to the "
              "lithosphere-asthenosphere boundary, as elevations from sea level "
              "(the sphere of 6371 km), on 2-degree cells, each with its density, "
              "and the sublithospheric mantle's beneath; the table's geoid and "
              "gravity columns are not read."),
    Dataset(
        "ANT-20",
        (Source(f"{_EMC}/ANT-20-voigtVs.r0.0-n4c.nc",
                "ea0e3ed6aa1c7ec9eb7ba656e087e1bc5c470dbe00055aec5034be74b194bceb"),),
        GridReader({"voigtVs": ("vs", None)}),
        title="The Voigt-average shear velocity of Antarctica and the surrounding "
              "oceans",
        citation=("Lloyd, A. J., D. A. Wiens, H. Zhu, J. Tromp, A. A. Nyblade, R. C. "
                  "Aster, S. E. Hansen, I. W. D. Dalziel, T. J. Wilson, E. R. Ivins "
                  "and J. P. O'Donnell (2020).  Seismic structure of the Antarctic "
                  "upper mantle imaged with adjoint tomography.  J. Geophys. Res. "
                  "Solid Earth 125, e2019JB017823, doi:10.1029/2019JB017823",
                  *_emc_cites("10.17611/dp/emc.2020.ant20.1")),
        licence=_EMC_LICENCE,
        notes="Voigt-average vs in km/s on EPSG:3031's polar stereographic plane "
              "(true scale at 71 S), every 40 km to 6440 km from the pole, from 800 "
              "km deep to 4 km above sea level, missing outside the model (42 % of "
              "the square): a regional grid, for merging into a global model.  "
              "Its axes are those its comment and proj4text state (x along 90 E, "
              "y along 0 E); its variables' long names swap them, which the data "
              "refute.  Most reliable south of 60 S between 60 and about 650 km."),
    Dataset(
        "GLAD-M25-viscosity",
        (Source("https://webservices.bgs.ac.uk/accessions/download/184161"
                "?fileName=3DViscosity.nc.tar.gz",
                "98f2a8ce993f4e74fae99aafda5184cab52efc68ea38758e4c16d8f645c20a74",
                name="3DViscosity.nc.tar.gz", members=("3D_visc_infer_GLAD-M25.nc",)),),
        GridReader({"log10eta": ("viscosity", None)}, units={"log10eta": "Pa s"},
                   log10=("log10eta",), seam="mean", pole_rows="mean", stride=2),
        title="A 3D viscosity of the mantle inferred from GLAD-M25",
        citation=("Lloyd, A. J., O. Crawford, D. Al-Attar, J. Austermann, M. J. "
                  "Hoggard, F. D. Richards and F. Syvret (2024).  GIA imaging of 3-D "
                  "mantle viscosity based on palaeo sea level observations - Part I: "
                  "Sensitivity kernels for an Earth with laterally varying viscosity.  "
                  "Geophys. J. Int. 236, 1139-1171, doi:10.1093/gji/ggad455", _GLAD,
                  "British Geological Survey, National Geoscience Data Centre: 3D "
                  "viscosity model inferred from GLAD-M25, deposited data item "
                  "184161"),
        licence="Open Government Licence v3.0, with the acknowledgement 'Contains "
                "data supplied by UKRI.'",
        notes="log10 of the viscosity in Pa s from the surface to the CMB (201 depths, "
              "1 km apart to 80 km, 5 to 400, 25 to 700, 50 below) on a half-degree "
              "grid; read as the viscosity itself, every second node (one degree), "
              "bilinear between nodes.  Upper 400 km from GLAD-M25's shear speed "
              "through a calibrated anelastic parameterisation, lower mantle through "
              "mineral physics; the crust set to the lithosphere's mean, about 1.5e26 "
              "Pa s, its two seam columns differing by up to 6 in log10 in the top 80 "
              "km (averaged as it is read, as are the pole rows).  Unclipped, from "
              "10^18.3 to 10^39.1 Pa s; the paper floors it at 2e19 Pa s after "
              "filtering to degree 64."),
    *[Dataset(
        f"SMEAN2-{profile}-b{beta}",
        (Source(f"{_SMEAN2}/SMEAN2_{profile}_b{beta}.nc", sha),),
        GridReader({"viscosity": ("viscosity", None)}, vertical_units="km",
                   units={"viscosity": "Pa s"}, clip={"viscosity": (1e18, 1e30)}),
        title=f"A 3D viscosity of the mantle from SMEAN2 by Wu's beta scaling, "
              f"background {profile}, beta profile {beta}",
        citation=("Steffen, R. and H. Steffen (2024).  A set of global "
                  "three-dimensional viscosity models based on the SMEAN2 shear-wave "
                  "velocity tomography model.  Zenodo, doi:10.5281/zenodo.13907989",),
        licence="CC BY 4.0",
        notes=f"Viscosity in Pa s at 58 depths from 25 to 2875 km (layer midpoints) on "
              f"a half-degree grid; SMEAN2 to temperature by Karato's scaling, to "
              f"viscosity by Wu, Wang & Steffen (2013) with Ivins & Sammis's (1995) "
              f"activation, about the {profile} background profile, beta "
              f"{_SMEAN2_BETA[beta]}.  "
              f"Unclipped as published (10^2.65 Pa s at 75 km; the 25 km slice 1e40 Pa "
              f"s everywhere, the lithosphere's marker); read clipped to 1e18 to 1e30 "
              f"Pa s, so the lithosphere stays effectively elastic.")
      for (profile, beta), sha in _SMEAN2_SHA.items()],
)


def _registry(entries: tuple[Dataset, ...]) -> Mapping[str, Dataset]:
    """The entries by name, read-only, a name used twice refused."""
    by_name: dict[str, Dataset] = {}
    for entry in entries:
        if entry.name in by_name:
            raise ValueError(f"two entries are named {entry.name!r}")
        by_name[entry.name] = entry
    return MappingProxyType(by_name)


#: The shipped data sets by name.
REGISTRY: Mapping[str, Dataset] = _registry(_ENTRIES)


def as_dataset(dataset: DatasetLike) -> Dataset:
    """The entry itself, or the shipped entry of that name."""
    if isinstance(dataset, Dataset):
        return dataset
    try:
        return REGISTRY[dataset]
    except KeyError:
        raise KeyError(f"no shipped data set is named {dataset!r}; the shipped "
                       f"ones are {sorted(REGISTRY)}, and a Dataset of one's own "
                       "is passed as it is") from None


def cite(dataset: DatasetLike) -> str:
    """The works a data set asks to be cited, one to a line."""
    return "\n".join(as_dataset(dataset).citation)


def describe(dataset: DatasetLike) -> str:
    """What the entry says of a data set, as lines of text: its title and
    body, its reference, its files, its licence, its citation and notes."""
    d = as_dataset(dataset)
    lines = [f"{d.name}: {d.title}", f"  body: {d.body}"]
    if d.reference is not None:
        lines.append(f"  relative to: {d.reference}")
    for s in d.sources:
        lines.append(f"  file: {s.name} from {s.url} (sha256 {s.sha256[:12]}...)")
        lines += [f"        mirror {m}" for m in s.mirrors]
        if s.members is not None:
            lines.append(f"        members {', '.join(s.members)}")
    lines.append(f"  licence: {d.licence}")
    lines += [f"  cite: {c}" for c in d.citation]
    if d.notes:
        lines.append(f"  notes: {d.notes}")
    return "\n".join(lines)
