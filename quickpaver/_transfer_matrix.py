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

Two implementations are provided:

- :func:`compute_transfer_matrix` handles arbitrary ``shapely``
  polygon grids using STRtree spatial indexing and vectorized
  Shapely intersection operations.
- :func:`compute_transfer_matrix_rectilinear` specializes to two
  rotated rectilinear (regular) grids and avoids Shapely entirely,
  offering two fast paths:

  1. **Separable** (relative angle = k x 90 degrees): the 2-D overlap
     factorises into two independent 1-D interval-overlap problems.
  2. **Non-separable** (arbitrary angle): grid-based candidate
     enumeration followed by a numba-parallel Sutherland-Hodgman clip
     and shoelace area computation per pair. Falls back to a
     vectorised numpy pipeline when numba is not installed.
"""

from __future__ import annotations

import numpy as np
import shapely
from scipy.sparse import coo_array, csc_array
from shapely.strtree import STRtree

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
# Transfer matrix for general grid
# ===================================================================


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


# ===================================================================
# Transfer matrix for rectilinear grids
# ===================================================================


def compute_transfer_matrix_rectilinear(
    source_center: np.ndarray,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle_deg: float,
    target_center: np.ndarray,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    target_angle_deg: float,
    is_sanity_check: bool = False,
) -> csc_array:
    """
    Build a conservative transfer matrix between two rotated rectilinear grids.

    Parameters
    ----------
    source_center, target_center : array-like, shape (2,)
        Grid centres in world coordinates.
    source_dx, source_dy, target_dx, target_dy : float
        Cell widths along each grid's local x / y axes.
    source_nx, source_ny, target_nx, target_ny : int
        Number of cells along each grid's local x / y axes.
    source_angle_deg, target_angle_deg : float
        Grid rotation angles in **degrees** (counter-clockwise from the
        world x-axis).
    is_sanity_check : bool, optional
        If *True*, verify that every fully-covered source cell conserves
        its quantity exactly (up to 1e-10).

    Returns
    -------
    scipy.sparse.csc_array
        Shape ``(n_source, n_target)`` with ``n = nx * ny``.
        Cell ``(i, j)`` maps to linear index ``j * nx + i`` (Fortran / column-major
        order, i.e. ``x`` varies fastest — equivalent to
        ``np.ravel_multi_index((i, j), (nx, ny), order="F")``).

    Notes
    -----
    When ``|target_angle - source_angle|`` is a multiple of 90 degrees (within
    1e-9 rad) the overlap factorises into two 1-D problems.  Otherwise
    a numba-parallel Sutherland-Hodgman clipper computes intersection
    areas (with a numpy-only fallback).  No Shapely dependency in either
    case.
    """
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
