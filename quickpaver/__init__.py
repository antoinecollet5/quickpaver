# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Tools for polygonal paving, transfer matrices, and rectilinear grids.

``quickpaver`` provides utilities to generate polygonal tilings, manipulate
rectilinear grids, compute transfer matrices between geometries, and load simple
geographical test contours. The package is designed around cell-based geometry:
tiles, polygons, and grid cells are represented explicitly and can be combined
with NumPy arrays, Shapely geometries, and sparse SciPy matrices.

The public API is grouped into four main areas:

- polygon and tiling generation;
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
   extract_tiling_centers
   extract_tiling_vertices

Transfer matrices
^^^^^^^^^^^^^^^^^

Utilities to compute sparse transfer matrices between source and target
geometries.

.. autosummary::
   :toctree: _autosummary

   compute_transfer_matrix
   compute_transfer_matrix_rectilinear

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
from quickpaver._tiling import (
    PolygonType,
    extract_tiling_centers,
    extract_tiling_vertices,
    gen_hexagonal_tiling,
    gen_polygon,
    gen_polygonal_tiling,
    gen_rectangular_tiling,
    gen_triangular_tiling,
    intersects_mask,
)
from quickpaver._transfer_matrix import (
    compute_transfer_matrix,
    compute_transfer_matrix_rectilinear,
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
    "gen_hexagonal_tiling",
    "gen_polygon",
    "gen_rectangular_tiling",
    "gen_triangular_tiling",
    "gen_polygonal_tiling",
    "Grid",
    "RectilinearGrid",
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
    "compute_transfer_matrix",
    "compute_transfer_matrix_rectilinear",
    "get_array_borders_selection",
    "intersects_mask",
]
