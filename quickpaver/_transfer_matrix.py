# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""
Conservative transfer-matrix construction between polygonal grids.

Given two grids of non-overlapping polygons (a "source" grid and a
"target" grid), this module builds a sparse matrix whose coefficients
are the fraction of each source-cell surface covered by each
target-cell, i.e. area(S_i intersection T_j) / area(S_i). Multiplying
a field defined on the source grid by this matrix's transpose yields a
surface-weighted (conservative) projection onto the target grid.

Two public entry points are exposed:

- :func:`compute_transfer_matrix_with_intersections` -- returns the
  sparse matrix together with the (source index, target index,
  intersection geometry) triplet for every nonzero entry.
- :func:`compute_transfer_matrix` -- convenience wrapper returning only
  the sparse matrix.

Both accept an optional boolean mask for either grid
(``source_grid_mask`` / ``target_grid_mask``): cells where the mask is
``False`` are excluded from the computation and contribute no nonzero
entries to the matrix (their row/column is entirely zero), while the
matrix shape itself is left unchanged so masking is transparent to
callers.

Internally, the entry points dispatch to one of three implementations
based on two booleans -- ``is_source_grid_rectilinear`` and
``is_target_grid_rectilinear`` -- describing which of the two grids is
a regular (possibly rotated) rectilinear grid:

- **Both arbitrary** (``False, False``): handled by an STRtree-based
  implementation using vectorized Shapely intersection operations.
- **Both rectilinear** (``True, True``): specializes to two rotated
  rectilinear (regular) grids and avoids Shapely entirely, offering
  two fast paths:

  1. **Separable** (relative angle = k x 90 degrees): the 2-D overlap
     factorises into two independent 1-D interval-overlap problems.
  2. **Non-separable** (arbitrary angle): grid-based candidate
     enumeration followed by a numba-parallel Sutherland-Hodgman clip
     and shoelace area computation per pair. Falls back to a
     vectorised numpy pipeline when numba is not installed.
- **Exactly one side rectilinear** (``True, False`` or ``False, True``):
  exploits the regularity of the rectilinear side to replace the
  generic STRtree candidate search with O(1) analytic index-range
  arithmetic per polygon on the arbitrary side, while still using
  Shapely for the exact clip (so arbitrary/non-convex/holed target or
  source polygons remain fully supported). This removes the
  tree-build/query cost that dominates the fully-arbitrary
  implementation when one side is regular.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import shapely
from scipy.sparse import coo_array, csc_array
from shapely.strtree import STRtree

from quickpaver._grid import RectilinearGrid
from quickpaver._types import ArrayLike, NDArrayBool, NDArrayInt

_HAS_NUMBA = False

try:
    from numba import njit, prange

    _HAS_NUMBA = True
except ModuleNotFoundError:
    # Fallback: no-op decorators
    def njit(*args, **kwargs):
        """Dummy decorator that does nothing"""

        def decorator(func):
            return func

        return decorator

    # Fallback: prange is just range
    prange = range  # ty:ignore[invalid-assignment]


# ===================================================================
# Public API
# ===================================================================


def compute_transfer_matrix_with_intersections(
    source_grid: Union[RectilinearGrid, shapely.MultiPolygon],
    target_grid: Union[RectilinearGrid, shapely.MultiPolygon],
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, shapely.MultiPolygon]:
    """
    Build a conservative transfer matrix between two polygonal grids,
    along with the per-pair intersection data.

    Dispatches to the fastest available implementation based on which
    of the two grids are regular (rectilinear) grids.

    Parameters
    ----------
    source_grid : RectilinearGrid or shapely.MultiPolygon
        Source grid. Either a ``RectilinearGrid`` (exposing ``cx, cy,
        dx, dy, nx, ny, theta``) or a ``shapely.MultiPolygon``.
    target_grid : RectilinearGrid or shapely.MultiPolygon
        Target grid, with the same convention as ``source_grid``.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over the source grid cells (length must equal the
        number of source cells: ``nx * ny`` for a ``RectilinearGrid``,
        ``len(source_grid.geoms)`` for a ``MultiPolygon``). Cells where
        the mask is ``False`` are excluded from the computation: they
        contribute no nonzero row in the returned matrix. If ``None``,
        all source cells are considered.
    target_grid_mask : 1-D array of bool, optional
        Boolean mask over the target grid cells, with the same
        convention as ``source_grid_mask`` but excluding columns
        instead of rows. If ``None``, all target cells are considered.
    is_sanity_check : bool, optional
        If True, verify that every fully-covered (and unmasked) source
        cell conserves its quantity exactly (up to 1e-10).

    Returns
    -------
    W : csc_array, shape (n_source, n_target)
        Sparse conservative transfer matrix. ``W[i, j] =
        |S_i ∩ T_j| / |S_i|``, so columns sum to 1 for fully covered,
        unmasked source polygons. The shape always reflects the full
        (unmasked) grid sizes; masked-out cells simply have no nonzero
        entries.
    source_indices : NDArrayInt, shape (nnz,)
        Source polygon id of each nonzero entry (row index in ``W``).
    target_indices : NDArrayInt, shape (nnz,)
        Target polygon id of each nonzero entry (column index in ``W``).
    intersections : shapely.MultiPolygon
        Intersection geometries, parallel to ``source_indices`` /
        ``target_indices``. Entries produced by the fast convex-clip
        path (used internally when exactly one grid is rectilinear)
        are returned as ``None`` placeholders, since that path computes
        areas without materializing exact intersection geometry.
    """
    is_source_grid_rectilinear = isinstance(source_grid, RectilinearGrid)
    is_target_grid_rectilinear = isinstance(target_grid, RectilinearGrid)

    if not is_source_grid_rectilinear and not is_target_grid_rectilinear:
        return _compute_transfer_matrix(
            source_grid,
            target_grid,
            source_grid_mask=source_grid_mask,
            target_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
        )

    if is_source_grid_rectilinear and is_target_grid_rectilinear:
        return _compute_transfer_matrix_rectilinear(
            source_grid,
            target_grid,
            source_grid_mask=source_grid_mask,
            target_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
        )

    if is_source_grid_rectilinear and not is_target_grid_rectilinear:
        return _compute_transfer_matrix_mixed(
            source_grid,
            target_grid,
            rectilinear_is_source=True,
            rectilinear_grid_mask=source_grid_mask,
            polygon_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
        )

    # not is_source_grid_rectilinear and is_target_grid_rectilinear
    return _compute_transfer_matrix_mixed(
        target_grid,
        source_grid,  # ty:ignore[invalid-argument-type]
        rectilinear_is_source=False,
        rectilinear_grid_mask=target_grid_mask,
        polygon_grid_mask=source_grid_mask,
        is_sanity_check=is_sanity_check,
    )


def compute_transfer_matrix(
    source_grid: Union[RectilinearGrid, shapely.MultiPolygon],
    target_grid: Union[RectilinearGrid, shapely.MultiPolygon],
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> csc_array:
    """
    Build a conservative transfer matrix between two polygonal grids.

    Dispatches to the fastest available implementation based on which
    of the two grids are regular (rectilinear) grids. This is a thin
    wrapper around :func:`compute_transfer_matrix_with_intersections`
    that discards the per-pair index/intersection data and returns
    only the sparse matrix.

    Parameters
    ----------
    source_grid : RectilinearGrid or shapely.MultiPolygon
        Source grid. Either a ``RectilinearGrid`` (exposing ``cx, cy,
        dx, dy, nx, ny, theta``) or a ``shapely.MultiPolygon``.
    target_grid : RectilinearGrid or shapely.MultiPolygon
        Target grid, with the same convention as ``source_grid``.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over the source grid cells (length must equal the
        number of source cells: ``nx * ny`` for a ``RectilinearGrid``,
        ``len(source_grid.geoms)`` for a ``MultiPolygon``). Cells where
        the mask is ``False`` are excluded from the computation: they
        contribute no nonzero row in the returned matrix. If ``None``,
        all source cells are considered.
    target_grid_mask : 1-D array of bool, optional
        Boolean mask over the target grid cells, with the same
        convention as ``source_grid_mask`` but excluding columns
        instead of rows. If ``None``, all target cells are considered.
    is_sanity_check : bool, optional
        If True, verify that every fully-covered (and unmasked) source
        cell conserves its quantity exactly (up to 1e-10).

    Returns
    -------
    scipy.sparse.csc_array
        Sparse conservative transfer matrix of shape
        ``(n_source, n_target)`` such that ``v_target = W.T @ v_source``.
        The shape always reflects the full (unmasked) grid sizes;
        masked-out cells simply have no nonzero entries.
    """
    return compute_transfer_matrix_with_intersections(
        source_grid,
        target_grid,
        source_grid_mask=source_grid_mask,
        target_grid_mask=target_grid_mask,
        is_sanity_check=is_sanity_check,
    )[0]


def _validate_mask(
    mask: Optional[NDArrayBool], n_cells: int, name: str
) -> Optional[NDArrayBool]:
    """Validate and normalize an optional boolean mask to shape (n_cells,)."""
    if mask is None:
        return None
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != n_cells:
        raise ValueError(
            f"{name} must be a 1-D boolean array of length {n_cells}, "
            f"got shape {mask.shape}."
        )
    return mask


# ===================================================================
# Transfer matrix for general grid
# ===================================================================


def _compute_transfer_matrix(
    source_grid: shapely.MultiPolygon,
    target_grid: shapely.MultiPolygon,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, shapely.MultiPolygon]:
    """
    Build a conservative transfer matrix between two arbitrary
    polygonal grids.

    The resulting sparse matrix distributes values defined on the
    source grid onto the target grid using surface intersection
    prorata weights.

    The matrix coefficients are defined as:

    .. math::

        W_{ij} = \\frac{|S_i \\cap T_j|}{|S_i|}

    where:

    - :math:`S_i` is the i-th source polygon
    - :math:`T_j` is the j-th target polygon

    Therefore:

    .. math::

        \\sum_j W_{ij} = 1

    for every unmasked source polygon fully covered by the (unmasked)
    target grid.

    Parameters
    ----------
    source_grid : shapely.MultiPolygon
        Source polygon grid.
    target_grid : shapely.MultiPolygon
        Target polygon grid.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over source polygons (length ``len(source_grid.geoms)``).
        Masked-out (``False``) polygons are excluded from the spatial
        query entirely and contribute no nonzero row. If ``None``, all
        source polygons are considered.
    target_grid_mask : 1-D array of bool, optional
        Boolean mask over target polygons (length ``len(target_grid.geoms)``),
        with the same convention as ``source_grid_mask`` but excluding
        columns instead of rows. If ``None``, all target polygons are
        considered.
    is_sanity_check : bool
        Whether to perform a sanity check at the end of the transfer.
        The default is False.

    Returns
    -------
    W : scipy.sparse.csc_array
        Sparse conservative transfer matrix of shape:

        .. math::

            (n_{source}, n_{target})

        such that:

        .. math::

            v_{target} = W^T v_{source}

    source_indices : NDArrayInt, shape (nnz,)
        Source polygon id of each nonzero entry.
    target_indices : NDArrayInt, shape (nnz,)
        Target polygon id of each nonzero entry.
    intersections : shapely.MultiPolygon
        Intersection geometries for each nonzero entry.

    Notes
    -----
    The implementation uses:

    - STRtree spatial indexing
    - vectorized Shapely intersection operations
    - sparse _coo matrix assembly

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

    source_grid_mask = _validate_mask(source_grid_mask, n_source, "source_grid_mask")
    target_grid_mask = _validate_mask(target_grid_mask, n_target, "target_grid_mask")

    # Restrict to unmasked polygons up front: this both saves work and
    # guarantees masked-out cells produce no nonzero entries.
    if source_grid_mask is not None:
        source_polygons = source_polygons[source_grid_mask]
    if target_grid_mask is not None:
        target_polygons = target_polygons[target_grid_mask]

    # Prepared geometries accelerate the exact "intersects" test
    # performed by the STRtree query below.
    shapely.prepare(source_polygons)

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
    # Compute intersection areas
    # -----------------------------------------------------------------

    # Cheap bounding-box pre-filter to discard pairs that can only
    # produce a point/line/sliver intersection before paying for
    # the exact GEOS intersection below.
    src_bounds = shapely.bounds(source_polygons[source_indices])
    tgt_bounds = shapely.bounds(target_polygons[target_indices])
    bbox_dx = np.minimum(src_bounds[:, 2], tgt_bounds[:, 2]) - np.maximum(
        src_bounds[:, 0], tgt_bounds[:, 0]
    )
    bbox_dy = np.minimum(src_bounds[:, 3], tgt_bounds[:, 3]) - np.maximum(
        src_bounds[:, 1], tgt_bounds[:, 1]
    )
    nontrivial = (bbox_dx > 1e-15) & (bbox_dy > 1e-15)

    source_indices = source_indices[nontrivial]
    target_indices = target_indices[nontrivial]

    # -------------------------------------------------------------
    # Compute vectorized polygon intersections
    # -------------------------------------------------------------

    intersections: np.ndarray = shapely.intersection(
        source_polygons[source_indices],
        target_polygons[target_indices],
    )

    intersection_areas = shapely.area(intersections)

    # Remove empty / numerical-noise intersections
    valid_mask: np.ndarray = intersection_areas > 1e-15

    source_indices = source_indices[valid_mask]
    target_indices = target_indices[valid_mask]
    intersection_areas = intersection_areas[valid_mask]
    intersections = intersections[valid_mask]

    # -----------------------------------------------------------------
    # Conservative normalization
    #
    # Each source polygon distributes 100% of its quantity
    # over intersecting target polygons.
    # -----------------------------------------------------------------

    source_areas: np.ndarray = shapely.area(source_polygons)
    weights: np.ndarray = intersection_areas / source_areas[source_indices]

    # -----------------------------------------------------------------
    # Map filtered-array indices back to original (unmasked) ids
    # -----------------------------------------------------------------

    if source_grid_mask is not None:
        source_orig_ids = np.flatnonzero(source_grid_mask)
        source_indices = source_orig_ids[source_indices]
    if target_grid_mask is not None:
        target_orig_ids = np.flatnonzero(target_grid_mask)
        target_indices = target_orig_ids[target_indices]

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
    # each fully-covered, unmasked source polygon must conserve its
    # full quantity
    # -----------------------------------------------------------------

    if is_sanity_check:
        _check_conservation(transfer_matrix)

    return (
        transfer_matrix,
        source_indices,
        target_indices,
        shapely.MultiPolygon(list(intersections)),
    )


# ===================================================================
# Transfer matrix for two rectilinear grids
# ===================================================================


def _compute_transfer_matrix_rectilinear(
    source_grid: RectilinearGrid,
    target_grid: RectilinearGrid,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, shapely.MultiPolygon]:
    """
    Build a conservative transfer matrix between two rotated
    rectilinear grids.

    Parameters
    ----------
    source_grid, target_grid : RectilinearGrid
        Grid objects exposing ``cx, cy, dx, dy, nx, ny, theta``, where
        ``theta`` is the grid rotation in degrees (counter-clockwise
        from the world x-axis) and ``nx``/``ny`` are the cell counts
        along the grid's local x / y axes.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over source cells, flattened in the same
        Fortran-order convention as the matrix rows (linear index
        ``j * nx + i``, ``x`` fastest), length ``source_grid.nx *
        source_grid.ny``. Masked-out (``False``) cells contribute no
        nonzero row. If ``None``, all source cells are considered.
    target_grid_mask : 1-D array of bool, optional
        Boolean mask over target cells, with the same flattening
        convention as ``source_grid_mask`` but excluding columns
        instead of rows. If ``None``, all target cells are considered.
    is_sanity_check : bool, optional
        If *True*, verify that every fully-covered, unmasked source
        cell conserves its quantity exactly (up to 1e-10).

    Returns
    -------
    W : scipy.sparse.csc_array
        Shape ``(n_source, n_target)`` with ``n = nx * ny``.
        Cell ``(i, j)`` maps to linear index ``j * nx + i`` (Fortran /
        column-major order, i.e. ``x`` varies fastest -- equivalent to
        ``np.ravel_multi_index((i, j), (nx, ny), order="F")``).
    source_indices : NDArrayInt, shape (nnz,)
        Source cell linear index of each nonzero entry.
    target_indices : NDArrayInt, shape (nnz,)
        Target cell linear index of each nonzero entry.
    intersections : shapely.MultiPolygon
        Empty. Exact intersection geometry is not computed by this
        analytic (Shapely-free) implementation; only areas are used to
        build the matrix.

    Notes
    -----
    When ``|target_angle - source_angle|`` is a multiple of 90 degrees
    (within 1e-9 rad) the overlap factorises into two 1-D problems.
    Otherwise a numba-parallel Sutherland-Hodgman clipper computes
    intersection areas (with a numpy-only fallback). No Shapely
    dependency in either case.
    """
    n_source = source_grid.nx * source_grid.ny
    n_target = target_grid.nx * target_grid.ny

    source_grid_mask = _validate_mask(source_grid_mask, n_source, "source_grid_mask")
    target_grid_mask = _validate_mask(target_grid_mask, n_target, "target_grid_mask")

    mat = _compute_transfer_matrix_rectilinear_impl(
        (source_grid.cx, source_grid.cy),
        source_grid.dx,
        source_grid.dy,
        source_grid.nx,
        source_grid.ny,
        source_grid.theta,
        (target_grid.cx, target_grid.cy),
        target_grid.dx,
        target_grid.dy,
        target_grid.nx,
        target_grid.ny,
        target_grid.theta,
        is_sanity_check=False,  # sanity check applied after masking below
    )

    mat = _apply_masks_to_matrix(mat, source_grid_mask, target_grid_mask)

    if is_sanity_check:
        _check_conservation(mat)

    _coo = mat.tocoo()
    return mat, _coo.row, _coo.col, shapely.MultiPolygon([])


def _apply_masks_to_matrix(
    mat: csc_array,
    source_grid_mask: Optional[NDArrayBool],
    target_grid_mask: Optional[NDArrayBool],
) -> csc_array:
    """Zero out rows/columns corresponding to masked-out cells, keeping shape."""
    if source_grid_mask is None and target_grid_mask is None:
        return mat

    _coo = mat.tocoo()
    keep = np.ones(len(_coo.data), dtype=bool)
    if source_grid_mask is not None:
        keep &= source_grid_mask[_coo.row]
    if target_grid_mask is not None:
        keep &= target_grid_mask[_coo.col]

    return coo_array(
        (_coo.data[keep], (_coo.row[keep], _coo.col[keep])),
        shape=mat.shape,
    ).tocsc()


# -------------------------------------------------------------------
# Actual computation (unmasked, raw parameters)
# -------------------------------------------------------------------
def _compute_transfer_matrix_rectilinear_impl(
    source_center: ArrayLike,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle_deg: float,
    target_center: ArrayLike,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    target_angle_deg: float,
    is_sanity_check: bool = False,
) -> csc_array:
    source_center = np.asarray(source_center, dtype=float)
    target_center = np.asarray(target_center, dtype=float)

    # ---- check whether the fast separable path applies ----
    source_angle_rad = np.deg2rad(source_angle_deg)
    target_angle_rad = np.deg2rad(target_angle_deg)
    rel_angle_rad = target_angle_rad - source_angle_rad
    k_exact = np.deg2rad(rel_angle_rad) / (np.pi / 2)
    k_round = round(k_exact)
    is_separable = abs(k_exact - k_round) < 1e-9

    if is_separable:
        return _separable_transfer(
            source_center,
            source_dx,
            source_dy,
            source_nx,
            source_ny,
            source_angle_rad,
            target_center,
            target_dx,
            target_dy,
            target_nx,
            target_ny,
            k_round % 4,
            is_sanity_check,
        )
    else:
        return _nonseparable_transfer(
            source_center,
            source_dx,
            source_dy,
            source_nx,
            source_ny,
            source_angle_rad,
            target_center,
            target_dx,
            target_dy,
            target_nx,
            target_ny,
            target_angle_rad,
            is_sanity_check,
        )


# ===================================================================
# Fast separable path  (relative angle = k x 90 degrees)
# ===================================================================


def _separable_transfer(
    source_center: np.ndarray,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle: float,
    target_center: np.ndarray,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    k90: int,
    is_sanity_check: bool,
) -> csc_array:
    """Separable transfer for relative rotation = *k90* x 90 degrees."""

    n_source = source_nx * source_ny
    n_target = target_nx * target_ny

    # -- source edges in source-local frame (always ascending) --
    src_x_edges = (np.arange(source_nx + 1) - source_nx / 2) * source_dx
    src_y_edges = (np.arange(source_ny + 1) - source_ny / 2) * source_dy

    # -- target origin in source-local frame --
    ca, sa = np.cos(source_angle), np.sin(source_angle)
    dx_w, dy_w = target_center - source_center
    tgt_origin_x = dx_w * ca + dy_w * sa
    tgt_origin_y = -dx_w * sa + dy_w * ca

    # -- which target dimension aligns with each source axis --
    if k90 == 0:
        sx_d, sx_n, sx_is_tgt_y = target_dx, target_nx, False
        sy_d, sy_n = target_dy, target_ny
        sx_sign, sy_sign = 1.0, 1.0
    elif k90 == 1:
        sx_d, sx_n, sx_is_tgt_y = target_dy, target_ny, True
        sy_d, sy_n = target_dx, target_nx
        sx_sign, sy_sign = -1.0, 1.0
    elif k90 == 2:
        sx_d, sx_n, sx_is_tgt_y = target_dx, target_nx, False
        sy_d, sy_n = target_dy, target_ny
        sx_sign, sy_sign = -1.0, -1.0
    else:
        sx_d, sx_n, sx_is_tgt_y = target_dy, target_ny, True
        sy_d, sy_n = target_dx, target_nx
        sx_sign, sy_sign = 1.0, -1.0

    raw_x = tgt_origin_x + sx_sign * (np.arange(sx_n + 1) - sx_n / 2) * sx_d
    raw_y = tgt_origin_y + sy_sign * (np.arange(sy_n + 1) - sy_n / 2) * sy_d

    tgt_x_edges, tx_perm = _sort_edges(raw_x, sx_n)
    tgt_y_edges, ty_perm = _sort_edges(raw_y, sy_n)

    is_x, it_x_sorted, ox = _compute_1d_overlaps(src_x_edges, tgt_x_edges)
    is_y, it_y_sorted, oy = _compute_1d_overlaps(src_y_edges, tgt_y_edges)

    if len(is_x) == 0 or len(is_y) == 0:
        return csc_array((n_source, n_target))

    it_x_orig = tx_perm[it_x_sorted]
    it_y_orig = ty_perm[it_y_sorted]

    nx_pairs = len(is_x)
    ny_pairs = len(is_y)

    src_ix = np.repeat(is_x, ny_pairs)
    src_jy = np.tile(is_y, nx_pairs)
    src_lin = src_jy * source_nx + src_ix

    if sx_is_tgt_y:
        tgt_jy = np.repeat(it_x_orig, ny_pairs)
        tgt_ix = np.tile(it_y_orig, nx_pairs)
    else:
        tgt_ix = np.repeat(it_x_orig, ny_pairs)
        tgt_jy = np.tile(it_y_orig, nx_pairs)

    tgt_lin = tgt_jy * target_nx + tgt_ix
    weights = np.repeat(ox, ny_pairs) * np.tile(oy, nx_pairs) / (source_dx * source_dy)

    mat = coo_array(
        (weights, (src_lin, tgt_lin)),
        shape=(n_source, n_target),
    ).tocsc()

    if is_sanity_check:
        _check_conservation(mat)
    return mat


def _sort_edges(edges: np.ndarray, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return ascending edges and a mapping from sorted cell index to original."""
    if edges[-1] >= edges[0]:
        return edges, np.arange(n)
    return edges[::-1].copy(), np.arange(n - 1, -1, -1)


def _compute_1d_overlaps(
    edges_a: np.ndarray, edges_b: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Find all overlapping interval pairs between two sorted edge arrays.

    Returns ``(idx_a, idx_b, overlaps)`` with only positive overlaps.
    """
    na = len(edges_a) - 1
    nb = len(edges_b) - 1
    if na == 0 or nb == 0:
        _e = np.empty(0, dtype=np.intp)
        return _e, _e, np.empty(0)

    right_b, left_b = edges_b[1:], edges_b[:-1]
    left_a, right_a = edges_a[:-1], edges_a[1:]

    j_starts = np.searchsorted(right_b, left_a, side="right")
    j_ends = np.searchsorted(left_b, right_a, side="left")

    counts = np.maximum(j_ends - j_starts, 0)
    total = counts.sum()
    if total == 0:
        _e = np.empty(0, dtype=np.intp)
        return _e, _e, np.empty(0)

    idx_a = np.repeat(np.arange(na, dtype=np.intp), counts)
    cum = np.empty(na + 1, dtype=np.intp)
    cum[0] = 0
    np.cumsum(counts, out=cum[1:])
    group_offset = np.arange(total, dtype=np.intp) - np.repeat(cum[:-1], counts)
    idx_b = group_offset + np.repeat(j_starts, counts)

    overlaps = np.minimum(right_a[idx_a], right_b[idx_b]) - np.maximum(
        left_a[idx_a], left_b[idx_b]
    )
    valid = overlaps > 1e-15
    return idx_a[valid], idx_b[valid], overlaps[valid]


# ===================================================================
# Non-separable path  (arbitrary relative angle)
#
# 1.  Source-local frame -> source cells axis-aligned.
# 2.  Rotated-rectangle template for target cells (all identical).
# 3.  Grid arithmetic -> candidate pairs (replaces STRtree).
# 4.  Sutherland-Hodgman clip + shoelace per pair.
#     Primary:  numba.prange  (~60x faster than Shapely)
#     Fallback: vectorised numpy vertex-collection pipeline
# ===================================================================


def _nonseparable_transfer(
    source_center: np.ndarray,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle: float,
    target_center: np.ndarray,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    target_angle: float,
    is_sanity_check: bool,
) -> csc_array:
    """Transfer matrix for two rectilinear grids at an arbitrary angle."""

    n_source = source_nx * source_ny
    n_target = target_nx * target_ny
    src_cell_area = source_dx * source_dy

    # -- source edges in source-local frame --
    src_x_edges = (np.arange(source_nx + 1) - source_nx / 2) * source_dx
    src_y_edges = (np.arange(source_ny + 1) - source_ny / 2) * source_dy

    # -- rotated target cell template --
    rel_angle = target_angle - source_angle
    cr, sr = np.cos(rel_angle), np.sin(rel_angle)
    hdx, hdy = target_dx / 2, target_dy / 2
    template = np.array(
        [[-hdx, -hdy], [hdx, -hdy], [hdx, hdy], [-hdx, hdy]],
        dtype=float,
    ) @ np.array([[cr, sr], [-sr, cr]])

    half_bx = abs(cr) * hdx + abs(sr) * hdy
    half_by = abs(sr) * hdx + abs(cr) * hdy

    # -- target cell centres in source-local frame --
    ca, sa = np.cos(source_angle), np.sin(source_angle)
    dw = target_center - source_center
    origin_x = dw[0] * ca + dw[1] * sa
    origin_y = -dw[0] * sa + dw[1] * ca

    kx = (np.arange(target_nx) - (target_nx - 1) / 2) * target_dx
    ly = (np.arange(target_ny) - (target_ny - 1) / 2) * target_dy
    tcx_grid = origin_x + cr * kx[:, None] - sr * ly[None, :]
    tcy_grid = origin_y + sr * kx[:, None] + cr * ly[None, :]
    tcx_flat = tcx_grid.ravel(order="F")
    tcy_flat = tcy_grid.ravel(order="F")

    # -- vectorised candidate-pair enumeration --
    src_x0, src_y0 = src_x_edges[0], src_y_edges[0]
    inv_sdx, inv_sdy = 1.0 / source_dx, 1.0 / source_dy

    i_lo = np.clip(
        np.floor((tcx_flat - half_bx - src_x0) * inv_sdx).astype(np.intp),
        0,
        source_nx - 1,
    )
    i_hi = np.clip(
        np.floor((tcx_flat + half_bx - src_x0) * inv_sdx).astype(np.intp),
        0,
        source_nx - 1,
    )
    j_lo = np.clip(
        np.floor((tcy_flat - half_by - src_y0) * inv_sdy).astype(np.intp),
        0,
        source_ny - 1,
    )
    j_hi = np.clip(
        np.floor((tcy_flat + half_by - src_y0) * inv_sdy).astype(np.intp),
        0,
        source_ny - 1,
    )

    ni = i_hi - i_lo + 1
    nj = j_hi - j_lo + 1
    counts_per_tgt = ni * nj
    total_pairs = int(counts_per_tgt.sum())

    if total_pairs == 0:
        return csc_array((n_source, n_target))

    # Expand into flat pair arrays
    tgt_flat_idx = np.repeat(np.arange(n_target, dtype=np.intp), counts_per_tgt)
    cum = np.empty(n_target + 1, dtype=np.intp)
    cum[0] = 0
    np.cumsum(counts_per_tgt, out=cum[1:])
    local_pos = np.arange(total_pairs, dtype=np.intp) - np.repeat(
        cum[:-1], counts_per_tgt
    )

    rep_nj = np.repeat(nj, counts_per_tgt)
    src_i = np.repeat(i_lo, counts_per_tgt) + local_pos // rep_nj
    src_j = np.repeat(j_lo, counts_per_tgt) + local_pos % rep_nj

    src_lin = src_j * source_nx + src_i
    tgt_lin = tgt_flat_idx

    pair_xmin = src_x_edges[src_i]
    pair_xmax = src_x_edges[src_i + 1]
    pair_ymin = src_y_edges[src_j]
    pair_ymax = src_y_edges[src_j + 1]

    pair_tvx = template[None, :, 0] + tcx_flat[tgt_flat_idx, None]  # (N, 4)
    pair_tvy = template[None, :, 1] + tcy_flat[tgt_flat_idx, None]

    # -- compute intersection areas --
    areas = _batch_clip_areas(
        np.ascontiguousarray(pair_tvx),
        np.ascontiguousarray(pair_tvy),
        pair_xmin,
        pair_ymin,
        pair_xmax,
        pair_ymax,
    )

    # -- filter and assemble sparse matrix --
    valid = areas > 1e-15
    weights = areas[valid] / src_cell_area

    mat = coo_array(
        (weights, (src_lin[valid], tgt_lin[valid])),
        shape=(n_source, n_target),
    ).tocsc()

    if is_sanity_check:
        _check_conservation(mat)
    return mat


# ===================================================================
# Intersection-area back-ends (numba primary, numpy fallback)
# ===================================================================


def _batch_clip_areas(
    tvx: np.ndarray,
    tvy: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    """Dispatch to the fastest available back-end."""
    if _HAS_NUMBA:
        return _batch_clip_numba(tvx, tvy, xmin, ymin, xmax, ymax)
    return _batch_clip_numpy(tvx, tvy, xmin, ymin, xmax, ymax)


# ---- numba back-end ------------------------------------------------


@njit(cache=True)
def _clip_area_single(
    vx: np.ndarray,
    vy: np.ndarray,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> float:
    """SH clip of a 4-vertex polygon against an AA rect -> area."""
    MAX_V = 8
    ax = np.empty(MAX_V)
    ay = np.empty(MAX_V)
    bx = np.empty(MAX_V)
    by = np.empty(MAX_V)
    n_in = 4
    for p in range(4):
        ax[p] = vx[p]
        ay[p] = vy[p]

    edges = (xmin, xmax, ymin, ymax)
    for e in range(4):
        ev = edges[e]
        n_out = 0
        if n_in < 3:
            return 0.0
        for i in range(n_in):
            pi = n_in - 1 if i == 0 else i - 1
            if e < 2:  # clip on x
                cc = ax[i]
                pc = ax[pi]
            else:  # clip on y
                cc = ay[i]
                pc = ay[pi]
            if e == 0 or e == 2:  # keep >=
                c_in = cc >= ev
                p_in = pc >= ev
            else:  # keep <=
                c_in = cc <= ev
                p_in = pc <= ev
            if p_in != c_in:
                d = cc - pc
                t = (ev - pc) / d if d != 0.0 else 0.0
                bx[n_out] = ax[pi] + t * (ax[i] - ax[pi])
                by[n_out] = ay[pi] + t * (ay[i] - ay[pi])
                n_out += 1
            if c_in:
                bx[n_out] = ax[i]
                by[n_out] = ay[i]
                n_out += 1
        n_in = n_out
        for p in range(n_in):
            ax[p] = bx[p]
            ay[p] = by[p]

    if n_in < 3:
        return 0.0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
    return abs(area) * 0.5


@njit(parallel=True, cache=True)
def _batch_clip_numba(
    tvx: np.ndarray,
    tvy: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    N = len(xmin)
    areas = np.empty(N)
    for idx in prange(N):  # ty:ignore[not-iterable]
        areas[idx] = _clip_area_single(
            tvx[idx],
            tvy[idx],
            xmin[idx],
            ymin[idx],
            xmax[idx],
            ymax[idx],
        )
    return areas


# ---- numpy fallback ------------------------------------------------


def _batch_clip_numpy(
    tvx: np.ndarray,
    tvy: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    """
    Vectorised intersection areas via vertex collection + shoelace.

    Collects vertices from three sources (target corners in source,
    source corners in target, edge-edge intersections), sorts by angle
    from centroid, and applies the shoelace formula - all in numpy.
    """
    N = len(xmin)

    all_x = np.full((N, 24), np.nan)
    all_y = np.full((N, 24), np.nan)

    # (a) target corners inside source rect
    t_in_s = (
        (tvx >= xmin[:, None])
        & (tvx <= xmax[:, None])
        & (tvy >= ymin[:, None])
        & (tvy <= ymax[:, None])
    )
    for c in range(4):
        m = t_in_s[:, c]
        all_x[m, c] = tvx[m, c]
        all_y[m, c] = tvy[m, c]

    # (b) source corners inside target rotated rect
    scx = np.column_stack([xmin, xmax, xmax, xmin])
    scy = np.column_stack([ymin, ymin, ymax, ymax])
    s_in_t = np.ones((N, 4), dtype=bool)
    for e in range(4):
        e1 = (e + 1) % 4
        ex = tvx[:, e1] - tvx[:, e]
        ey = tvy[:, e1] - tvy[:, e]
        px = scx - tvx[:, e : e + 1]
        py = scy - tvy[:, e : e + 1]
        s_in_t &= (ex[:, None] * py - ey[:, None] * px) >= -1e-15
    for c in range(4):
        m = s_in_t[:, c]
        all_x[m, 4 + c] = scx[m, c]
        all_y[m, 4 + c] = scy[m, c]

    # (c) edge-edge intersections (4 target x 4 source = 16)
    slot = 8
    for _te in range(4):
        te1 = (_te + 1) % 4
        p0x, p0y = tvx[:, _te], tvy[:, _te]
        p1x, p1y = tvx[:, te1], tvy[:, te1]
        dtx, dty = p1x - p0x, p1y - p0y
        for se in range(4):
            if se % 2 == 0:  # horizontal
                ey = ymin if se == 0 else ymax
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(np.abs(dty) > 1e-15, (ey - p0y) / dty, np.nan)
                ix = p0x + t * dtx
                ok = (
                    np.isfinite(t)
                    & (t >= -1e-12)
                    & (t <= 1 + 1e-12)
                    & (ix >= xmin - 1e-12)
                    & (ix <= xmax + 1e-12)
                )
                all_x[ok, slot] = ix[ok]
                all_y[ok, slot] = ey[ok] if np.ndim(ey) else ey
            else:  # vertical
                ex = xmax if se == 1 else xmin
                with np.errstate(divide="ignore", invalid="ignore"):
                    t = np.where(np.abs(dtx) > 1e-15, (ex - p0x) / dtx, np.nan)
                iy = p0y + t * dty
                ok = (
                    np.isfinite(t)
                    & (t >= -1e-12)
                    & (t <= 1 + 1e-12)
                    & (iy >= ymin - 1e-12)
                    & (iy <= ymax + 1e-12)
                )
                all_x[ok, slot] = ex[ok] if np.ndim(ex) else ex
                all_y[ok, slot] = iy[ok]
            slot += 1

    # angle-sort + shoelace
    fin = np.isfinite(all_x)
    vc = fin.sum(axis=1)
    sx = np.where(fin, all_x, 0.0)
    sy = np.where(fin, all_y, 0.0)
    denom = np.maximum(vc, 1).astype(float)
    cx = sx.sum(1) / denom
    cy = sy.sum(1) / denom
    ang = np.where(fin, np.arctan2(all_y - cy[:, None], all_x - cx[:, None]), np.inf)
    order = np.argsort(ang, axis=1)
    ri = np.arange(N)[:, None]
    sx = sx[ri, order]
    sy = sy[ri, order]
    cross = sx[:, :-1] * sy[:, 1:] - sx[:, 1:] * sy[:, :-1]
    emask = np.arange(23)[None, :] < (vc[:, None] - 1)
    svc = np.maximum(vc, 1).astype(np.intp)
    lx = sx[np.arange(N), svc - 1]
    ly = sy[np.arange(N), svc - 1]
    closing = lx * sy[:, 0] - sx[:, 0] * ly
    areas = 0.5 * np.abs((cross * emask).sum(1) + closing)
    areas[vc < 3] = 0.0
    return areas


# ===================================================================
# Generalized N-vertex Sutherland-Hodgman clip (numba primary, numpy
# fallback), used by the mixed path to bypass Shapely entirely when
# the arbitrary-side polygons are convex.
# ===================================================================

_MAX_CLIP_VERTS = 32  # generous headroom for polygon-vs-rect clip output


def _polygons_are_convex(poly_polygons: np.ndarray) -> NDArrayBool:
    """
    Vectorised convexity test for an array of shapely Polygon objects.

    A simple polygon (no self-intersections, no holes) is convex iff
    its signed cross product at every vertex has a constant sign. Uses
    a single batched ``shapely.get_coordinates`` call plus
    ``np.minimum.reduceat`` / ``np.maximum.reduceat`` (mirroring the
    bbox computation in `_polygon_candidate_ranges_in_rectilinear_grid`)
    to test the per-polygon sign-consistency without any Python loop
    over polygons or their vertices.

    Polygons with interior rings (holes) are conservatively treated as
    non-convex, since a polygon with a hole can never be convex.
    """
    n_poly = len(poly_polygons)
    if n_poly == 0:
        return np.empty(0, dtype=bool)

    has_holes = shapely.get_num_interior_rings(poly_polygons) > 0

    exteriors = shapely.get_exterior_ring(poly_polygons)
    coords, coord_poly_idx = shapely.get_coordinates(exteriors, return_index=True)
    # coord_poly_idx is sorted/non-decreasing; every polygon contributes
    # at least 4 points (3 distinct + closing repeat) since shapely
    # rejects degenerate rings, so `counts >= 4` always holds here.
    boundaries = np.searchsorted(coord_poly_idx, np.arange(n_poly + 1), side="left")
    starts = boundaries[:-1]
    stops = boundaries[1:]
    counts = stops - starts  # includes the repeated closing vertex

    x = coords[:, 0]
    y = coords[:, 1]

    # "previous" vertex per point: for each point at global index k
    # belonging to polygon p with local position L = k - starts[p],
    # prev(k) = starts[p] + (L - 1) mod (counts[p] - 1) using the
    # *true* vertex count (counts[p] - 1, since the last coordinate
    # duplicates the first). Build this via a per-point offset array.
    true_n = counts - 1  # true vertex count per polygon
    point_poly_idx = np.repeat(np.arange(n_poly), counts)
    local_pos = np.arange(len(x)) - np.repeat(starts, counts)
    rep_true_n = np.repeat(true_n, counts)

    # drop the duplicated closing vertex (local_pos == true_n) from
    # the active vertex set used for the cross-product test.
    is_closing = local_pos == rep_true_n
    active = ~is_closing

    active_poly_idx = point_poly_idx[active]
    active_local_pos = local_pos[active]
    active_true_n = rep_true_n[active]
    active_x = x[active]
    active_y = y[active]
    active_starts = starts[active_poly_idx]

    prev_local = (active_local_pos - 1) % active_true_n
    next_local = (active_local_pos + 1) % active_true_n
    prev_global = active_starts + prev_local
    next_global = active_starts + next_local

    px, py = x[prev_global], y[prev_global]
    nx_, ny_ = x[next_global], y[next_global]

    e1x, e1y = active_x - px, active_y - py
    e2x, e2y = nx_ - active_x, ny_ - active_y
    cross = e1x * e2y - e1y * e2x

    nonzero_mask = np.abs(cross) > 1e-15
    pos_mask = nonzero_mask & (cross > 0)
    neg_mask = nonzero_mask & (cross < 0)

    has_pos = np.zeros(n_poly, dtype=bool)
    has_neg = np.zeros(n_poly, dtype=bool)
    np.logical_or.at(has_pos, active_poly_idx[pos_mask], True)
    np.logical_or.at(has_neg, active_poly_idx[neg_mask], True)

    too_few_verts = true_n < 3
    convex = ~(has_pos & has_neg) & ~too_few_verts & ~has_holes
    return convex


@njit(cache=True)
def _clip_area_single_nverts(
    vx: np.ndarray,
    vy: np.ndarray,
    n_verts: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
) -> float:
    """
    Sutherland-Hodgman clip of a convex ``n_verts``-vertex polygon
    (vertices given CCW or CW, either works for an area-only result)
    against an axis-aligned rectangle -> clipped area.

    ``vx``/``vy`` are fixed-size buffers of length ``_MAX_CLIP_VERTS``;
    only the first ``n_verts`` entries are meaningful. The output
    polygon can gain at most one extra vertex per clip edge (4 edges),
    so ``_MAX_CLIP_VERTS`` must be >= n_verts + 4 for the largest input
    the caller will pass in.
    """
    MAX_V = _MAX_CLIP_VERTS
    ax = np.empty(MAX_V)
    ay = np.empty(MAX_V)
    bx = np.empty(MAX_V)
    by = np.empty(MAX_V)
    n_in = n_verts
    for p in range(n_verts):
        ax[p] = vx[p]
        ay[p] = vy[p]

    edges = (xmin, xmax, ymin, ymax)
    for e in range(4):
        ev = edges[e]
        n_out = 0
        if n_in < 3:
            return 0.0
        for i in range(n_in):
            pi = n_in - 1 if i == 0 else i - 1
            if e < 2:  # clip on x
                cc = ax[i]
                pc = ax[pi]
            else:  # clip on y
                cc = ay[i]
                pc = ay[pi]
            if e == 0 or e == 2:  # keep >=
                c_in = cc >= ev
                p_in = pc >= ev
            else:  # keep <=
                c_in = cc <= ev
                p_in = pc <= ev
            if p_in != c_in:
                d = cc - pc
                t = (ev - pc) / d if d != 0.0 else 0.0
                bx[n_out] = ax[pi] + t * (ax[i] - ax[pi])
                by[n_out] = ay[pi] + t * (ay[i] - ay[pi])
                n_out += 1
            if c_in:
                bx[n_out] = ax[i]
                by[n_out] = ay[i]
                n_out += 1
        n_in = n_out
        for p in range(n_in):
            ax[p] = bx[p]
            ay[p] = by[p]

    if n_in < 3:
        return 0.0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
    return abs(area) * 0.5


@njit(parallel=True, cache=True)
def _batch_clip_numba_nverts(
    vx: np.ndarray,
    vy: np.ndarray,
    n_verts: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    """
    Parallel batch clip of ``N`` convex polygons (each padded to
    ``vx.shape[1]`` columns, with only the first ``n_verts[k]``
    meaningful) against ``N`` axis-aligned rectangles.
    """
    N = len(xmin)
    areas = np.empty(N)
    for idx in prange(N):  # ty:ignore[not-iterable]
        areas[idx] = _clip_area_single_nverts(
            vx[idx],
            vy[idx],
            n_verts[idx],
            xmin[idx],
            ymin[idx],
            xmax[idx],
            ymax[idx],
        )
    return areas


def _batch_clip_numpy_nverts(
    vx: np.ndarray,
    vy: np.ndarray,
    n_verts: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    """
    Numpy fallback for :func:`_batch_clip_numba_nverts`: a simple
    per-pair Python loop calling a pure-Python Sutherland-Hodgman clip.
    Only used when numba is unavailable, so raw Python performance here
    is acceptable -- it is still far cheaper than a Shapely
    intersection call per pair since it avoids GEOS object overhead.
    """
    N = len(xmin)
    areas = np.empty(N)
    for k in range(N):
        m = int(n_verts[k])
        poly = list(zip(vx[k, :m].tolist(), vy[k, :m].tolist()))
        areas[k] = _sh_clip_area_python(poly, xmin[k], ymin[k], xmax[k], ymax[k])
    return areas


def _sh_clip_area_python(
    poly: list, xmin: float, ymin: float, xmax: float, ymax: float
) -> float:
    """Pure-Python Sutherland-Hodgman clip + shoelace, N-vertex input."""
    edges = (
        ("x", ">=", xmin),
        ("x", "<=", xmax),
        ("y", ">=", ymin),
        ("y", "<=", ymax),
    )
    output = poly
    for axis, op, ev in edges:
        if len(output) < 3:
            return 0.0
        inp = output
        output = []
        for i in range(len(inp)):
            cur = inp[i]
            prev = inp[i - 1]
            cc = cur[0] if axis == "x" else cur[1]
            pc = prev[0] if axis == "x" else prev[1]
            c_in = cc >= ev if op == ">=" else cc <= ev
            p_in = pc >= ev if op == ">=" else pc <= ev
            if p_in != c_in:
                d = cc - pc
                t = (ev - pc) / d if d != 0.0 else 0.0
                ix = prev[0] + t * (cur[0] - prev[0])
                iy = prev[1] + t * (cur[1] - prev[1])
                output.append((ix, iy))
            if c_in:
                output.append(cur)
    if len(output) < 3:
        return 0.0
    area = 0.0
    n = len(output)
    for i in range(n):
        j = (i + 1) % n
        area += output[i][0] * output[j][1] - output[j][0] * output[i][1]
    return abs(area) * 0.5


def _batch_clip_areas_nverts(
    vx: np.ndarray,
    vy: np.ndarray,
    n_verts: np.ndarray,
    xmin: np.ndarray,
    ymin: np.ndarray,
    xmax: np.ndarray,
    ymax: np.ndarray,
) -> np.ndarray:
    """Dispatch to the fastest available N-vertex clip back-end."""
    if _HAS_NUMBA:
        return _batch_clip_numba_nverts(vx, vy, n_verts, xmin, ymin, xmax, ymax)
    return _batch_clip_numpy_nverts(vx, vy, n_verts, xmin, ymin, xmax, ymax)


# ===================================================================
# Mixed path: exactly one grid is rectilinear, the other arbitrary
# polygons.
#
# Idea: build the source-local candidate ranges analytically (as in
# `_nonseparable_transfer`), instead of building an STRtree. Two
# back-ends for the actual clip:
#
# - Convex polygons (checked via `_polygons_are_convex`): bypass
#   Shapely entirely and use the numba-parallel (or numpy-fallback)
#   generalized Sutherland-Hodgman clipper against the axis-aligned
#   rectilinear cell -- same trick as `_nonseparable_transfer`, just
#   with an arbitrary convex polygon instead of a fixed 4-vertex
#   rotated-rectangle template.
# - Non-convex / holed polygons: fall back to Shapely's exact clip,
#   which handles arbitrary topology.
#
# Either way, candidate-pair generation never uses an STRtree, which
# removes the tree-build/query overhead that dominates the
# fully-arbitrary implementation when one side is a large regular
# grid.
# ===================================================================


def _rectilinear_grid_polygons_local_frame(
    center: np.ndarray,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    angle_deg: float,
) -> Tuple[np.ndarray, np.ndarray, float, float, Tuple[np.ndarray, np.ndarray]]:
    """
    Return per-cell world-frame polygon corner coordinates for a
    rectilinear grid, plus the local-frame edge vectors needed for the
    world -> local transform.

    Returns
    -------
    corners_x, corners_y : ndarray, shape (n_cells, 4)
        World-frame coordinates of the 4 corners of each cell, ordered
        counter-clockwise, cell linear index = j * nx + i (x fastest).
    angle_rad : float
        Grid rotation in radians.
    cell_area : float
        Area of a single cell (``dx * dy``).
    edges_local : tuple(ndarray, ndarray)
        ``(x_edges, y_edges)`` 1-D arrays of length ``nx + 1`` / ``ny + 1``
        giving cell boundaries in the grid's own local (unrotated) frame.
    """
    angle_rad = np.deg2rad(angle_deg)
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)

    x_edges = (np.arange(nx + 1) - nx / 2) * dx
    y_edges = (np.arange(ny + 1) - ny / 2) * dy

    x0 = x_edges[:-1][:, None]
    x1 = x_edges[1:][:, None]
    y0 = y_edges[:-1][None, :]
    y1 = y_edges[1:][None, :]

    # local-frame corners per cell, shape (nx, ny, 4)
    local_cx = np.stack(
        [
            np.broadcast_to(x0, (nx, ny)),
            np.broadcast_to(x1, (nx, ny)),
            np.broadcast_to(x1, (nx, ny)),
            np.broadcast_to(x0, (nx, ny)),
        ],
        axis=-1,
    )
    local_cy = np.stack(
        [
            np.broadcast_to(y0, (nx, ny)),
            np.broadcast_to(y0, (nx, ny)),
            np.broadcast_to(y1, (nx, ny)),
            np.broadcast_to(y1, (nx, ny)),
        ],
        axis=-1,
    )

    world_cx = center[0] + local_cx * ca - local_cy * sa
    world_cy = center[1] + local_cx * sa + local_cy * ca

    # flatten with x fastest: linear index = j * nx + i.
    # world_cx/world_cy have shape (nx, ny, 4); a plain C-order reshape
    # would group by i first (index = i * ny + j), so transpose to
    # (ny, nx, 4) before flattening to get j * nx + i as required.
    corners_x = world_cx.transpose(1, 0, 2).reshape(nx * ny, 4)
    corners_y = world_cy.transpose(1, 0, 2).reshape(nx * ny, 4)

    return corners_x, corners_y, angle_rad, dx * dy, (x_edges, y_edges)


def _polygon_candidate_ranges_in_rectilinear_grid(
    poly_x: np.ndarray,
    poly_y: np.ndarray,
    poly_slices: list,
    grid_center: np.ndarray,
    grid_angle_rad: float,
    x_edges: np.ndarray,
    y_edges: np.ndarray,
    nx: int,
    ny: int,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    For each polygon (given by concatenated world-frame vertex arrays
    ``poly_x``/``poly_y`` and per-polygon ``poly_slices`` = list of
    ``(start, stop)`` index pairs), compute the axis-aligned bounding
    box of its vertices in the rectilinear grid's local frame, and
    convert that bbox directly into a rectangular range of candidate
    cell indices ``[i_lo, i_hi] x [j_lo, j_hi]`` via O(1) arithmetic
    (no spatial index).

    Returns
    -------
    poly_idx, cell_i, cell_j : NDArrayInt
        Flattened arrays such that for entry ``k``, polygon
        ``poly_idx[k]`` has candidate source cell ``(cell_i[k],
        cell_j[k])``.
    """
    ca, sa = np.cos(grid_angle_rad), np.sin(grid_angle_rad)

    n_poly = len(poly_slices)

    dxw = poly_x - grid_center[0]
    dyw = poly_y - grid_center[1]
    local_x = dxw * ca + dyw * sa
    local_y = -dxw * sa + dyw * ca

    # Vectorised per-polygon bbox via a segment-id array and
    # np.minimum.reduceat / np.maximum.reduceat (avoids a Python loop
    # over polygons). ``starts`` gives the reduceat boundaries; all
    # slices are assumed contiguous and non-empty (guaranteed by the
    # caller, which builds them from polygon exterior coordinate
    # counts).
    starts = np.fromiter((s for s, _ in poly_slices), dtype=np.intp, count=n_poly)
    bbox_xmin = np.minimum.reduceat(local_x, starts)
    bbox_xmax = np.maximum.reduceat(local_x, starts)
    bbox_ymin = np.minimum.reduceat(local_y, starts)
    bbox_ymax = np.maximum.reduceat(local_y, starts)

    x0, y0 = x_edges[0], y_edges[0]
    dx = x_edges[1] - x_edges[0] if nx > 0 else 1.0
    dy = y_edges[1] - y_edges[0] if ny > 0 else 1.0
    inv_dx, inv_dy = 1.0 / dx, 1.0 / dy

    i_lo = np.clip(np.floor((bbox_xmin - x0) * inv_dx).astype(np.intp), 0, nx - 1)
    i_hi = np.clip(
        np.floor((bbox_xmax - x0) * inv_dx - 1e-12).astype(np.intp), 0, nx - 1
    )
    j_lo = np.clip(np.floor((bbox_ymin - y0) * inv_dy).astype(np.intp), 0, ny - 1)
    j_hi = np.clip(
        np.floor((bbox_ymax - y0) * inv_dy - 1e-12).astype(np.intp), 0, ny - 1
    )
    i_hi = np.maximum(i_hi, i_lo)
    j_hi = np.maximum(j_hi, j_lo)

    ni = i_hi - i_lo + 1
    nj = j_hi - j_lo + 1
    counts = ni * nj
    total = int(counts.sum())

    if total == 0:
        _e = np.empty(0, dtype=np.intp)
        return _e, _e, _e

    poly_idx = np.repeat(np.arange(n_poly, dtype=np.intp), counts)
    cum = np.empty(n_poly + 1, dtype=np.intp)
    cum[0] = 0
    np.cumsum(counts, out=cum[1:])
    local_pos = np.arange(total, dtype=np.intp) - np.repeat(cum[:-1], counts)

    rep_nj = np.repeat(nj, counts)
    cell_i = np.repeat(i_lo, counts) + local_pos // rep_nj
    cell_j = np.repeat(j_lo, counts) + local_pos % rep_nj

    return poly_idx, cell_i, cell_j


def _compute_transfer_matrix_mixed(
    rectilinear_grid: RectilinearGrid,
    polygon_grid: shapely.MultiPolygon,
    rectilinear_is_source: bool,
    rectilinear_grid_mask: Optional[NDArrayBool] = None,
    polygon_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, shapely.MultiPolygon]:
    """
    Conservative transfer matrix between one rectilinear grid and one
    arbitrary polygon grid.

    Exploits the rectilinear side's regularity for O(1) analytic
    candidate-pair generation while using Shapely for the exact
    (possibly non-convex) clip.

    Parameters
    ----------
    rectilinear_grid : RectilinearGrid
        Grid object exposing ``cx, cy, dx, dy, nx, ny, theta``.
    polygon_grid : shapely.MultiPolygon
        The arbitrary polygon grid (the other side of the transfer).
    rectilinear_is_source : bool
        If True, the rectilinear grid is the source and
        ``polygon_grid`` is the target. If False, the roles are
        reversed (rectilinear grid is the target).
    rectilinear_grid_mask : 1-D array of bool, optional
        Boolean mask over the rectilinear grid's cells, flattened in
        the same Fortran-order convention as the matrix axis it
        corresponds to (linear index ``j * nx + i``, ``x`` fastest),
        length ``rectilinear_grid.nx * rectilinear_grid.ny``.
        Masked-out (``False``) cells contribute no nonzero entry. If
        ``None``, all rectilinear cells are considered.
    polygon_grid_mask : 1-D array of bool, optional
        Boolean mask over ``polygon_grid`` polygons (length
        ``len(polygon_grid.geoms)``), with the same convention as
        ``rectilinear_grid_mask`` but for the polygon side. If
        ``None``, all polygons are considered.
    is_sanity_check : bool
        If True, verify conservation for fully-covered, unmasked
        source cells.

    Returns
    -------
    W : csc_array, shape (n_source, n_target)
        Sparse conservative transfer matrix ``W[i, j] = |S_i ∩ T_j| / |S_i|``.
    source_indices, target_indices : NDArrayInt, shape (nnz,)
        Row / column index of each nonzero entry.
    intersections : shapely.MultiPolygon
        Intersection geometries for each nonzero entry (parallel to
        ``source_indices`` / ``target_indices``). Entries produced by
        the fast convex-clip path are returned as ``None``
        placeholders, since that path computes areas without
        materializing exact intersection geometry.
    """
    rectilinear_center = np.asarray(
        (rectilinear_grid.cx, rectilinear_grid.cy), dtype=float
    )
    rectilinear_dx = rectilinear_grid.dx
    rectilinear_dy = rectilinear_grid.dy
    rectilinear_nx = rectilinear_grid.nx
    rectilinear_ny = rectilinear_grid.ny
    rectilinear_angle_deg = rectilinear_grid.theta

    n_rect = rectilinear_nx * rectilinear_ny
    n_poly = len(polygon_grid.geoms)

    rectilinear_grid_mask = _validate_mask(
        rectilinear_grid_mask, n_rect, "rectilinear grid mask"
    )
    polygon_grid_mask = _validate_mask(polygon_grid_mask, n_poly, "polygon grid mask")

    (
        rect_corners_x,
        rect_corners_y,
        rect_angle_rad,
        rect_cell_area,
        (x_edges, y_edges),
    ) = _rectilinear_grid_polygons_local_frame(
        rectilinear_center,
        rectilinear_dx,
        rectilinear_dy,
        rectilinear_nx,
        rectilinear_ny,
        rectilinear_angle_deg,
    )

    poly_polygons: np.ndarray = np.asarray(polygon_grid.geoms, dtype=object)

    def _empty_result() -> Tuple[
        csc_array, NDArrayInt, NDArrayInt, shapely.MultiPolygon
    ]:
        n_source = n_rect if rectilinear_is_source else n_poly
        n_target = n_poly if rectilinear_is_source else n_rect
        empty = csc_array((n_source, n_target))
        _e = np.empty(0, dtype=np.intp)
        return empty, _e, _e, shapely.MultiPolygon([])

    # Flatten polygon exterior coordinates once for the vectorised
    # local-frame bbox computation (works for simple, possibly
    # non-convex polygons; holes are ignored for the bbox candidate
    # search, which is conservative -- it can only *over*-include
    # candidates, never miss one, since Shapely still performs the
    # exact clip afterwards). Uses shapely's own coordinate extraction
    # (no per-polygon Python loop).
    if n_poly > 0:
        exteriors = shapely.get_exterior_ring(poly_polygons)
        coords, coord_poly_idx = shapely.get_coordinates(exteriors, return_index=True)
        poly_x = coords[:, 0]
        poly_y = coords[:, 1]
        # coord_poly_idx is sorted, non-decreasing: build (start, stop)
        # slice boundaries per polygon via searchsorted.
        boundaries = np.searchsorted(coord_poly_idx, np.arange(n_poly + 1), side="left")
        poly_slices = list(zip(boundaries[:-1], boundaries[1:]))
    else:
        poly_x = np.empty(0)
        poly_y = np.empty(0)
        poly_slices = []

    poly_idx, cell_i, cell_j = _polygon_candidate_ranges_in_rectilinear_grid(
        poly_x,
        poly_y,
        poly_slices,
        rectilinear_center,
        rect_angle_rad,
        x_edges,
        y_edges,
        rectilinear_nx,
        rectilinear_ny,
    )

    if len(poly_idx) == 0:
        return _empty_result()

    rect_lin = cell_j * rectilinear_nx + cell_i

    # -- apply masks early to avoid wasted clip work on excluded cells --
    if rectilinear_grid_mask is not None or polygon_grid_mask is not None:
        keep_masked = np.ones(len(poly_idx), dtype=bool)
        if rectilinear_grid_mask is not None:
            keep_masked &= rectilinear_grid_mask[rect_lin]
        if polygon_grid_mask is not None:
            keep_masked &= polygon_grid_mask[poly_idx]
        poly_idx = poly_idx[keep_masked]
        rect_lin = rect_lin[keep_masked]
        if len(poly_idx) == 0:
            return _empty_result()

    # -- bbox pre-filter (cheap, avoids paying for exact clip on
    #    sliver/edge-touch candidates) --
    rect_xmin = rect_corners_x[rect_lin].min(axis=1)
    rect_xmax = rect_corners_x[rect_lin].max(axis=1)
    rect_ymin = rect_corners_y[rect_lin].min(axis=1)
    rect_ymax = rect_corners_y[rect_lin].max(axis=1)

    poly_bounds = shapely.bounds(poly_polygons[poly_idx])
    bbox_dx = np.minimum(rect_xmax, poly_bounds[:, 2]) - np.maximum(
        rect_xmin, poly_bounds[:, 0]
    )
    bbox_dy = np.minimum(rect_ymax, poly_bounds[:, 3]) - np.maximum(
        rect_ymin, poly_bounds[:, 1]
    )
    nontrivial = (bbox_dx > 1e-15) & (bbox_dy > 1e-15)

    poly_idx = poly_idx[nontrivial]
    rect_lin = rect_lin[nontrivial]

    if len(poly_idx) == 0:
        return _empty_result()

    # -- split surviving candidates by convexity of their polygon side:
    #    convex polygons bypass Shapely via the numba/numpy N-vertex
    #    Sutherland-Hodgman clipper; non-convex / holed polygons fall
    #    back to the exact Shapely clip. --
    is_convex_per_poly = _polygons_are_convex(poly_polygons)
    candidate_is_convex = is_convex_per_poly[poly_idx]

    rect_lin_cvx = rect_lin[candidate_is_convex]
    poly_idx_cvx = poly_idx[candidate_is_convex]
    rect_lin_gen = rect_lin[~candidate_is_convex]
    poly_idx_gen = poly_idx[~candidate_is_convex]

    result_rect_lin_parts = []
    result_poly_idx_parts = []
    result_areas_parts = []
    result_intersections_parts = []

    # -- fast path: convex polygons, numba/numpy N-vertex SH clip --
    #
    # IMPORTANT: `_clip_area_single_nverts` assumes an axis-aligned
    # clip rectangle. The rectilinear grid's cells are only
    # axis-aligned in the grid's own *local* frame (they are rotated
    # in world coordinates whenever `rectilinear_angle_deg != 0`), so
    # both the polygon vertices and the clip rectangle bounds must be
    # expressed in that local frame here -- we cannot reuse the
    # world-frame `rect_corners_x/y` used by the Shapely branch.
    if len(poly_idx_cvx) > 0:
        exteriors_cvx = shapely.get_exterior_ring(poly_polygons[poly_idx_cvx])
        coords_cvx, coord_local_idx = shapely.get_coordinates(
            exteriors_cvx, return_index=True
        )
        # world -> rectilinear-grid-local transform (inverse rotation)
        rc_ca, rc_sa = np.cos(rect_angle_rad), np.sin(rect_angle_rad)
        dxw = coords_cvx[:, 0] - rectilinear_center[0]
        dyw = coords_cvx[:, 1] - rectilinear_center[1]
        coords_cvx_local = np.column_stack(
            [dxw * rc_ca + dyw * rc_sa, -dxw * rc_sa + dyw * rc_ca]
        )

        n_cand = len(poly_idx_cvx)
        counts_cvx = np.bincount(coord_local_idx, minlength=n_cand)
        # shapely rings repeat the first vertex at the end; drop it
        # per-candidate so `n_verts` reflects the true vertex count.
        counts_cvx = counts_cvx - 1
        max_verts = int(counts_cvx.max()) if n_cand > 0 else 0

        if max_verts + 4 > _MAX_CLIP_VERTS:
            raise ValueError(
                f"Polygon with {max_verts} vertices exceeds the "
                f"_MAX_CLIP_VERTS={_MAX_CLIP_VERTS} buffer capacity of "
                "the fast convex-clip path (needs max_verts + 4 slots "
                "for Sutherland-Hodgman output growth). Increase "
                "_MAX_CLIP_VERTS or exclude this polygon from the "
                "convex fast path."
            )

        boundaries_cvx = np.searchsorted(
            coord_local_idx, np.arange(n_cand + 1), side="left"
        )
        starts_cvx = boundaries_cvx[:-1]

        # Vectorised scatter of each candidate's (non-closing) vertices
        # into fixed-size padded buffers, replacing a Python loop over
        # candidates. For point at global coordinate index k belonging
        # to candidate c (via coord_local_idx), its local column in
        # the padded buffer is (k - starts_cvx[c]); only points with
        # local column < counts_cvx[c] are kept (drops the duplicated
        # closing vertex, which always lands at local column ==
        # counts_cvx[c]).
        point_cand_idx = coord_local_idx
        point_local_col = np.arange(len(coords_cvx_local)) - starts_cvx[point_cand_idx]
        keep = point_local_col < counts_cvx[point_cand_idx]

        vx_padded = np.zeros((n_cand, _MAX_CLIP_VERTS))
        vy_padded = np.zeros((n_cand, _MAX_CLIP_VERTS))
        vx_padded[point_cand_idx[keep], point_local_col[keep]] = coords_cvx_local[
            keep, 0
        ]
        vy_padded[point_cand_idx[keep], point_local_col[keep]] = coords_cvx_local[
            keep, 1
        ]

        # cell (i, j) from rect_lin_cvx = j * nx + i (Fortran order,
        # matching `_rectilinear_grid_polygons_local_frame`).
        cell_j_cvx, cell_i_cvx = np.divmod(rect_lin_cvx, rectilinear_nx)
        cell_xmin = x_edges[cell_i_cvx]
        cell_xmax = x_edges[cell_i_cvx + 1]
        cell_ymin = y_edges[cell_j_cvx]
        cell_ymax = y_edges[cell_j_cvx + 1]

        areas_cvx = _batch_clip_areas_nverts(
            np.ascontiguousarray(vx_padded),
            np.ascontiguousarray(vy_padded),
            counts_cvx.astype(np.intp),
            cell_xmin,
            cell_ymin,
            cell_xmax,
            cell_ymax,
        )

        valid_cvx = areas_cvx > 1e-15
        result_rect_lin_parts.append(rect_lin_cvx[valid_cvx])
        result_poly_idx_parts.append(poly_idx_cvx[valid_cvx])
        result_areas_parts.append(areas_cvx[valid_cvx])
        # Intersection geometries are not computed on the fast path
        # (area-only clipper); returned as None placeholders so the
        # combined `intersections` output stays index-aligned. Callers
        # needing exact intersection geometry for the convex fast path
        # can still get it from Shapely on request, but computing it
        # by default here would defeat the purpose of bypassing GEOS.
        result_intersections_parts.append(
            np.full(int(valid_cvx.sum()), None, dtype=object)
        )

    # -- general path: non-convex / holed polygons, exact Shapely clip --
    if len(poly_idx_gen) > 0:
        rect_polys_gen = shapely.polygons(
            np.stack(
                [rect_corners_x[rect_lin_gen], rect_corners_y[rect_lin_gen]],
                axis=-1,
            )
        )
        intersections_gen = shapely.intersection(
            rect_polys_gen, poly_polygons[poly_idx_gen]
        )
        areas_gen = shapely.area(intersections_gen)
        valid_gen = areas_gen > 1e-15
        result_rect_lin_parts.append(rect_lin_gen[valid_gen])
        result_poly_idx_parts.append(poly_idx_gen[valid_gen])
        result_areas_parts.append(areas_gen[valid_gen])
        result_intersections_parts.append(intersections_gen[valid_gen])

    rect_lin = np.concatenate(result_rect_lin_parts)
    poly_idx = np.concatenate(result_poly_idx_parts)
    intersection_areas = np.concatenate(result_areas_parts)
    intersections = np.concatenate(result_intersections_parts)

    # Redundant safety net: both branches above already filter on
    # area > 1e-15 individually, but re-applying here guards against
    # numerical edge cases at the boundary between the two paths.
    valid = intersection_areas > 1e-15
    rect_lin = rect_lin[valid]
    poly_idx = poly_idx[valid]
    intersection_areas = intersection_areas[valid]
    intersections = intersections[valid]

    rect_cell_areas = np.full(n_rect, rect_cell_area)

    if rectilinear_is_source:
        source_indices = rect_lin
        target_indices = poly_idx
        source_areas = rect_cell_areas
        n_source, n_target = n_rect, n_poly
    else:
        source_indices = poly_idx
        target_indices = rect_lin
        source_areas = shapely.area(poly_polygons)
        n_source, n_target = n_poly, n_rect

    weights = intersection_areas / source_areas[source_indices]

    mat = coo_array(
        (weights, (source_indices, target_indices)),
        shape=(n_source, n_target),
    ).tocsc()

    if is_sanity_check:
        _check_conservation(mat)

    return (
        mat,
        source_indices,
        target_indices,
        shapely.MultiPolygon(list(intersections)),
    )


# ===================================================================
# Sanity check
# ===================================================================


def _check_conservation(mat: csc_array) -> None:
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    covered = row_sums > 1 - 1e-6
    if covered.any():
        np.testing.assert_allclose(
            row_sums[covered],
            np.ones(covered.sum()),
            atol=1e-10,
            err_msg="Conservation violated: some fully-covered source cells "
            "do not have row sums equal to 1.",
        )
