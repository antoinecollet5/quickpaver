# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Tools for polygonal paving, transfer matrices, and rectilinear grids.

``quickpaver`` provides utilities to generate polygonal tilings, manipulate
rectilinear grids, compute transfer matrices between geometries, and load simple
geographical test contours. The package is designed around cell-based geometry:
tiles, polygons, and grid cells are represented explicitly and can be combined
with NumPy arrays, Shapely geometries, and sparse SciPy matrices.

The public API is grouped into seven main areas:

- polygon and tiling generation;
- cell quality metrics;
- point sampling;
- VTK meshes (surface meshes and layered extrusion);
- transfer-matrix construction;
- geographical test contours;
- rectilinear-grid utilities.

Polygon helpers
^^^^^^^^^^^^^^^

Utilities for creating base polygons and describing supported tile types.

.. autosummary::
   :toctree: _autosummary

   PolygonType
   gen_polygon
   Disk

Tiling functions
^^^^^^^^^^^^^^^^

Functions to generate polygonal tilings over a target surface. Supported tile
families include rectangles, triangles, and hexagons. Tiling generators return
both the generated polygons and adjacency information between kept tiles.

.. autosummary::
   :toctree: _autosummary

   gen_hexagonal_tiling
   gen_rectangular_tiling
   gen_triangular_tiling
   gen_polygonal_tiling
   gen_polygonal_tiling_arrays
   extract_tiling_centers
   extract_tiling_vertices
   extract_tiling_edge_centers
   adjacency_by_shared_vertices
   adjacency_to_edges
   draw_adjacency

Cell quality
^^^^^^^^^^^^

Roundness quality of the cells of a tessellation (triangulations, tilings,
Voronoi diagrams): ``1`` for a regular polygon, decreasing towards ``0`` as the
cell is distorted, plus closed forms for stretched regular cells.

.. autosummary::
   :toctree: _autosummary

   multipolygon_quality
   triangle_anisotropy_quality
   square_anisotropy_quality
   hexagon_anisotropy_quality

Point sampling
^^^^^^^^^^^^^^

Blue-noise point sets whose local density follows a radius map (e.g. seeds of
a graded Voronoi mesh).

.. autosummary::
   :toctree: _autosummary

   poisson_disk_sampling

VTK meshes
^^^^^^^^^^

PyVista / VTK meshes from polygons: surface meshes whose cells share their
vertices, and layered (terrain-following) extrusion to 3D cells. Requires
the optional PyVista dependency (``pip install quickpaver[pyvista]``).

.. autosummary::
   :toctree: _autosummary

   shapely_polygons_to_vtu
   create_unstructured_mesh
   pv_extrude
   get_ncell4poly

Transfer matrices
^^^^^^^^^^^^^^^^^

Utilities to compute sparse transfer matrices between source and target
geometries.

.. autosummary::
   :toctree: _autosummary

   compute_transfer_matrix
   compute_transfer_matrix_with_intersections
   compute_transfer_matrix_with_intersection_vertices
   IntersectionVertices

Test data
^^^^^^^^^

Convenience functions providing geographical contours useful for examples,
tests, and demonstrations.

.. autosummary::
   :toctree: _autosummary

   load_france_contour
   load_corsica_contour
   load_france_and_corsica_contour

Regular grids
^^^^^^^^^^^^^

Classes and utilities for working with cell-based rectilinear grids. These
functions cover grid indexing, sparse neighbour operators, polygon selections,
binary dilation, resampling, and conservative or duplicative array upsampling.

.. autosummary::
   :toctree: _autosummary

   Grid
   RectilinearGrid
   TriMesh
   rlg_idx_to_nn
   rlg_nn_to_idx
   span_to_node_numbers_2d
   span_to_node_numbers_3d
   get_array_borders_selection
   get_owner_neigh_indices
   get_rlg_spatial_grad_mat
   make_rlg_spatial_gradient_matrices
   get_rlg_perm_mat
   make_rlg_spatial_permutation_matrices
   create_selections_array_2d
   get_polygon_selection_with_dilation_2d
   get_polygon_selection_with_dilation_3d
   binary_dilation
   resample_grid
   duplicative_upsample
   conservative_upsample
   intersects_mask

"""

from quickpaver.__about__ import __author__, __email__, __version__
from quickpaver._grid import (
    Grid,
    RectilinearGrid,
    TriMesh,
    binary_dilation,
    conservative_upsample,
    create_selections_array_2d,
    duplicative_upsample,
    get_array_borders_selection,
    get_owner_neigh_indices,
    get_polygon_selection_with_dilation_2d,
    get_polygon_selection_with_dilation_3d,
    get_rlg_perm_mat,
    get_rlg_spatial_grad_mat,
    make_rlg_spatial_gradient_matrices,
    make_rlg_spatial_permutation_matrices,
    resample_grid,
    rlg_idx_to_nn,
    rlg_nn_to_idx,
    span_to_node_numbers_2d,
    span_to_node_numbers_3d,
)
from quickpaver._mesh import (
    create_unstructured_mesh,
    get_ncell4poly,
    pv_extrude,
    shapely_polygons_to_vtu,
)
from quickpaver._plot import adjacency_to_edges, draw_adjacency
from quickpaver._quality import (
    hexagon_anisotropy_quality,
    multipolygon_quality,
    square_anisotropy_quality,
    triangle_anisotropy_quality,
)
from quickpaver._sampling import poisson_disk_sampling
from quickpaver._tiling import (
    Disk,
    PolygonType,
    adjacency_by_shared_vertices,
    extract_tiling_centers,
    extract_tiling_edge_centers,
    extract_tiling_vertices,
    gen_hexagonal_tiling,
    gen_polygon,
    gen_polygonal_tiling,
    gen_polygonal_tiling_arrays,
    gen_rectangular_tiling,
    gen_triangular_tiling,
    intersects_mask,
)
from quickpaver._transfer_matrix import (
    IntersectionVertices,
    compute_transfer_matrix,
    compute_transfer_matrix_with_intersection_vertices,
    compute_transfer_matrix_with_intersections,
)
from quickpaver.data import (
    load_corsica_contour,
    load_france_and_corsica_contour,
    load_france_contour,
)

__all__ = [
    "__author__",
    "__email__",
    "__version__",
    "PolygonType",
    "Disk",
    "gen_hexagonal_tiling",
    "gen_polygon",
    "gen_rectangular_tiling",
    "gen_triangular_tiling",
    "gen_polygonal_tiling",
    "gen_polygonal_tiling_arrays",
    "Grid",
    "RectilinearGrid",
    "TriMesh",
    "create_selections_array_2d",
    "rlg_idx_to_nn",
    "rlg_nn_to_idx",
    "span_to_node_numbers_2d",
    "span_to_node_numbers_3d",
    "binary_dilation",
    "get_owner_neigh_indices",
    "get_polygon_selection_with_dilation_2d",
    "get_polygon_selection_with_dilation_3d",
    "get_rlg_spatial_grad_mat",
    "make_rlg_spatial_gradient_matrices",
    "get_rlg_perm_mat",
    "make_rlg_spatial_permutation_matrices",
    "resample_grid",
    "duplicative_upsample",
    "conservative_upsample",
    "load_france_contour",
    "load_corsica_contour",
    "load_france_and_corsica_contour",
    "extract_tiling_centers",
    "extract_tiling_vertices",
    "extract_tiling_edge_centers",
    "IntersectionVertices",
    "compute_transfer_matrix",
    "compute_transfer_matrix_with_intersection_vertices",
    "compute_transfer_matrix_with_intersections",
    "get_array_borders_selection",
    "intersects_mask",
    "adjacency_by_shared_vertices",
    "adjacency_to_edges",
    "draw_adjacency",
    "multipolygon_quality",
    "triangle_anisotropy_quality",
    "square_anisotropy_quality",
    "hexagon_anisotropy_quality",
    "poisson_disk_sampling",
    "shapely_polygons_to_vtu",
    "create_unstructured_mesh",
    "pv_extrude",
    "get_ncell4poly",
]
