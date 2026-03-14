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
    gen_circular_bounding_tiling
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


.. currentmodule:: pyrtid.utils.grid

Regular grids
^^^^^^^^^^^^^

Provide utilities to work with rectilinear grids.

.. autosummary::
   :toctree: _autosummary

    indices_to_node_number
    node_number_to_indices
    span_to_node_numbers_2d
    span_to_node_numbers_3d
    create_selections_array_2d
    RectilinearGrid
    get_polygon_selection_with_dilation_2d
    get_extended_grid_shape

"""

from quickpaver.__about__ import __author__, __email__, __version__
from quickpaver._grid import (
    RectilinearGrid,
    create_selections_array_2d,
    get_polygon_selection_with_dilation_2d,
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
    "gen_circular_bounding_tiling",
    "gen_polygonal_tiling",
    "RectilinearGrid",
    "create_selections_array_2d",
    "get_a_not_in_b_1d",
    "get_array_borders_selection_2d",
    "get_array_borders_selection_3d",
    "get_extended_grid_shape",
    "get_polygon_selection_with_dilation_2d",
    "indices_to_node_number",
    "node_number_to_indices",
    "span_to_node_numbers_2d",
    "span_to_node_numbers_3d",
    "rlg_idx_to_nn",
    "rlg_nn_to_idx",
    "load_france_contour",
    "load_corsica_contour",
    "load_france_and_corsica_contour",
    "extract_tiling_centers",
    "extract_tiling_vertices",
]
