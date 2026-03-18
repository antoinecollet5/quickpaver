# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""TODO

Helpers
^^^^^^^

Helpers to work with paving.

.. autosummary::
   :toctree: _autosummary

    PolygonType
    gen_polygon

Paving functions
^^^^^^^^^^^^^^^^

This is the core ! description TODO.

.. autosummary::
   :toctree: _autosummary

    gen_hexagonal_tiling
    gen_rectangular_tiling
    gen_triangular_tiling
    gen_polygonal_tiling
    extract_tiling_centers
    extract_tiling_vertices


Test data
^^^^^^^^^
Functions providing test data.

.. autosummary::
   :toctree: _autosummary

    load_france_contour
    load_corsica_contour
    load_france_and_corsica_contour


Regular grids
^^^^^^^^^^^^^

Provide utilities to work with rectilinear grids.

.. autosummary::
   :toctree: _autosummary

    RectilinearGrid
    binary_dilation
    conservative_upsample
    create_selections_array_2d
    duplicative_upsample
    get_owner_neigh_indices
    get_polygon_selection_with_dilation_2d
    get_polygon_selection_with_dilation_3d
    get_rlg_perm_mat
    get_rlg_spatial_grad_mat
    make_rlg_spatial_gradient_matrices
    make_rlg_spatial_permutation_matrices
    resample_grid
    rlg_idx_to_nn
    rlg_nn_to_idx
    span_to_node_numbers_2d
    span_to_node_numbers_3d

"""

from quickpaver.__about__ import __author__, __email__, __version__
from quickpaver._grid import (
    RectilinearGrid,
    binary_dilation,
    conservative_upsample,
    create_selections_array_2d,
    duplicative_upsample,
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
]
