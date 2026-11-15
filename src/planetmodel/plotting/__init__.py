"""Pictures of models and meshes: profiles, sections, maps, spectra, VTK
files, and a mesh cut by a plane.

Making a picture is kept apart from drawing it: the numbers are
`planetmodel.slices`' (a sphere, a section, a profile, a spectrum), or
for a flat model `planetmodel.cartesian.slices`' (a plane, a section, a
profile, a spectrum, a block), sampled in the core for any model, or for
a mesh `plane_cut`'s, and the functions here draw them on
matplotlib axes they are given, returning what they drew and leaving
the figure to the caller.  Nothing here is particular to the Earth.

`profile` and `radial_profile` draw a field against radius, radius
upward, or against depth, depth downward, for a flat model of
`planetmodel.cartesian`, a step at every interface, a mean's spread
shaded where a slice carries one (`profiles`); `section` draws a
great-circle section as a wedge of the annulus or a disc, or in the
body's deformed shape (`sections`); `sphere_map` draws a sphere slice on
cartopy's projections (`maps`, the `maps` extra); for a flat model,
`plane_map` draws a plane slice as a map in (x, y), `vertical_section` a
section, depth downward, layer by layer, flat or in the body's shape,
in the coordinates `plane_cut` cuts a mesh in so that the two compose,
and `block` the box's top and two sides on three-dimensional axes, the
counterpart of the cutaway (`flat`); `cutaway` draws the
body with a wedge cut from it, the field on the faces of the cut and
the surface as the field, a plain colour, Natural Earth's land and sea
or its shaded relief, with coastlines (`cutaways`, cartopy where the
Earth's surface is drawn); `spectrum`,
`correlation` and `correlation_total` draw spectra and correlations by
degree and radius, or a flat model's by horizontal wavenumber and
depth, the wavelength along the top (`spectra`); `write_vtk` writes a
model as VTK files for ParaView, and `write_flat_vtk` a flat one (`vtk`,
no extra needed); `plane_cut` cuts the cells or
faces of a simplex mesh by a plane, or with `kind="crinkle"` takes the
cells it passes through whole, seen from one side, the elements' shape
kept where a cut shreds them, `in_plane` takes the elements of a
mesh lying in a plane whole (a two-dimensional mesh, or a box's top),
and `mesh_section` draws either, the elements' sections coloured as
asked and the faces' as curves, and
`clip_polygons` keeps the part of polygons on one side of a plane, for
a cutaway in three dimensions, `mesh_block` draws a block cut from a
volume mesh on three-dimensional axes (`block_planes` its six planes,
to clip what is drawn inside it), and `fault_curves` a fault's curves
by kind, each in its own line style (`FAULT_CURVE_STYLES`), in space or
projected onto a plane (`meshes`); `style` holds the look they
share, a signed field such as a sensitivity kernel drawn symmetric about
zero with `diverging=True`, its limits at a percentile with `clip=`, and
divided by its largest value with `normalised=True`, on maps and
sections alike.  matplotlib is not imported here: the axes are the caller's,
made with the `plot` extra, and cartopy only where a map is drawn.
"""
from .cutaways import COASTLINE_STYLE, LAND, LIGHT, SEA, cutaway
from .flat import BLOCK_VIEWS, block, grid_cell_edges, plane_map, vertical_section
from .maps import cell_edges, sphere_map
from .meshes import (BLOCK_VIEW, FAULT_CURVE_STYLES, PlaneCut, block_planes,
                     clip_polygons, fault_curves, in_plane, mesh_block, mesh_section,
                     plane_cut)
from .profiles import profile, radial_profile
from .sections import section
from .spectra import correlation, correlation_total, spectrum
from .style import BOUNDARY_STYLE, DIVERGING, SEQUENTIAL, colour_extend, colour_scale
from .vtk import write_flat_vtk, write_vtk

__all__ = ["profile", "radial_profile", "section", "sphere_map", "cell_edges",
           "plane_map", "vertical_section", "block", "grid_cell_edges",
           "BLOCK_VIEWS",
           "cutaway", "spectrum", "correlation", "correlation_total", "write_vtk",
           "write_flat_vtk",
           "PlaneCut", "plane_cut", "in_plane", "mesh_section", "clip_polygons",
           "block_planes", "mesh_block", "fault_curves", "FAULT_CURVE_STYLES",
           "BLOCK_VIEW",
           "BOUNDARY_STYLE", "DIVERGING", "SEQUENTIAL", "LIGHT", "LAND", "SEA",
           "COASTLINE_STYLE", "colour_scale", "colour_extend"]
