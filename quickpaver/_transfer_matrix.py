# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""
Code for efficient transfer matrix construction based on grid cells intersections.
"""

import numpy as np
import shapely
from scipy.sparse import coo_array, csc_array
from shapely.strtree import STRtree


def compute_transfer_matrix(
    source_grid: shapely.MultiPolygon,
    target_grid: shapely.MultiPolygon,
    is_sanity_check: bool = False,
) -> csc_array:
    """
    Build a conservative transfer matrix between two polygonal grids.

    The resulting sparse matrix distributes values defined on the
    source grid onto the target grid using surface intersection
    prorata weights.

    The matrix coefficients are defined as:

    :contentReference[oaicite:0]{index=0}

    where:

    - :math:`S_i` is the i-th source polygon
    - :math:`T_j` is the j-th target polygon

    Therefore:

    .. math::

        \\sum_j W_{ij} = 1

    for every source polygon fully covered by the target grid.

    Parameters
    ----------
    source_grid : shapely.MultiPolygon
        Source polygon grid.
    target_grid : shapely.MultiPolygon
        Target polygon grid.
    is_sanity_check: bool
        Whether to perform a sanity check at the end of the transfer.
        The default is False.

    Returns
    -------
    scipy.sparse.csc_array
        Sparse conservative transfer matrix of shape:

        .. math::

            (n_{source}, n_{target})

        such that:

        .. math::

            v_{target} = W^T v_{source}

    Notes
    -----
    The implementation uses:

    - STRtree spatial indexing
    - vectorized Shapely intersection operations
    - sparse COO matrix assembly
    """

    # -----------------------------------------------------------------
    # Convert geometries to NumPy object arrays
    # -----------------------------------------------------------------

    source_polygons: np.ndarray = np.asarray(
        source_grid.geoms,
        dtype=object,
    )

    target_polygons: np.ndarray = np.asarray(
        target_grid.geoms,
        dtype=object,
    )

    n_source: int = len(source_polygons)
    n_target: int = len(target_polygons)

    # -----------------------------------------------------------------
    # Build spatial index on source polygons
    # -----------------------------------------------------------------

    tree: STRtree = STRtree(source_polygons)

    # -----------------------------------------------------------------
    # Query all intersecting polygon pairs
    #
    # Returned shape:
    #
    # pairs[0] -> indices in target_polygons
    # pairs[1] -> indices in source_polygons
    # -----------------------------------------------------------------

    pairs: np.ndarray = tree.query(
        target_polygons,
        predicate="intersects",
    )

    target_indices: np.ndarray = pairs[0]
    source_indices: np.ndarray = pairs[1]

    # -----------------------------------------------------------------
    # Compute vectorized polygon intersections
    # -----------------------------------------------------------------

    intersections: np.ndarray = shapely.intersection(
        source_polygons[source_indices],
        target_polygons[target_indices],
    )

    # -----------------------------------------------------------------
    # Compute intersection areas
    # -----------------------------------------------------------------

    intersection_areas: np.ndarray = shapely.area(intersections)

    # Remove empty / numerical-noise intersections
    valid_mask: np.ndarray = intersection_areas > 1e-15

    source_indices = source_indices[valid_mask]
    target_indices = target_indices[valid_mask]
    intersection_areas = intersection_areas[valid_mask]

    # -----------------------------------------------------------------
    # Conservative normalization
    #
    # Each source polygon distributes 100% of its quantity
    # over intersecting target polygons.
    # -----------------------------------------------------------------

    source_areas: np.ndarray = shapely.area(source_polygons)

    weights: np.ndarray = intersection_areas / source_areas[source_indices]

    # -----------------------------------------------------------------
    # Assemble sparse transfer matrix
    # -----------------------------------------------------------------

    transfer_matrix: csc_array = coo_array(
        (
            weights,
            (
                source_indices,
                target_indices,
            ),
        ),
        shape=(n_source, n_target),
    ).tocsc()

    # -----------------------------------------------------------------
    # Sanity check:
    # each source polygon must conserve its full quantity
    # -----------------------------------------------------------------

    # sanity check
    if is_sanity_check:
        np.testing.assert_allclose(
            (transfer_matrix.T * source_areas).sum(axis=1)
            / shapely.area(target_polygons),
            np.ones(transfer_matrix.shape[1]),
            atol=1e-10,
        )

    return transfer_matrix
