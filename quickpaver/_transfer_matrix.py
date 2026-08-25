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

Each side of the transfer may be a :class:`~quickpaver._grid.RectilinearGrid`,
a :class:`~quickpaver._grid.TriMesh`, or an arbitrary
:class:`shapely.MultiPolygon`. Internally, the entry points dispatch to
one of several implementations based on which *kind* each side is:

- **Both arbitrary polygons** (``MultiPolygon, MultiPolygon``):
  handled by an STRtree-based implementation using vectorized Shapely
  intersection operations.
- **Both rectilinear** (``RectilinearGrid, RectilinearGrid``):
  specializes to two rotated rectilinear (regular) grids and avoids
  Shapely entirely, offering two fast paths:

  1. **Separable** (relative angle = k x 90 degrees): the 2-D overlap
     factorises into two independent 1-D interval-overlap problems.
  2. **Non-separable** (arbitrary angle): grid-based candidate
     enumeration followed by a numba-parallel Sutherland-Hodgman clip
     and shoelace area computation per pair. Falls back to a
     vectorised numpy pipeline when numba is not installed.
- **Exactly one side rectilinear, other side arbitrary polygons**
  (``RectilinearGrid, MultiPolygon``): exploits the regularity of the
  rectilinear side to replace the generic STRtree candidate search
  with O(1) analytic index-range arithmetic per polygon on the
  arbitrary side. Convex polygons bypass Shapely entirely via a
  numba/numpy Sutherland-Hodgman clip against the axis-aligned grid
  cell; non-convex/holed polygons fall back to Shapely's exact clip.
- **Rectilinear vs TriMesh** (either order): a *dedicated* path that
  never materializes Shapely polygons for the mesh side at all (no
  ``TriMesh.to_shapely()`` call). Triangle vertices are read directly
  from ``mesh.verts_xy[mesh.tri_verts]`` (a pure numpy gather),
  bounding boxes are computed with vectorized ``min``/``max`` reductions
  over the fixed ``(n_tri, 3)`` vertex array (no ``shapely.bounds``),
  and since every triangle has exactly 3 vertices, the padded clip
  buffers required by the generic N-vertex clip are built without any
  ragged-array bookkeeping (no ``get_coordinates``/``searchsorted``
  dance -- the vertex count is a compile-time constant). Every
  candidate is convex by construction, so the per-polygon convexity
  test is skipped entirely, and the fast numba/numpy Sutherland-Hodgman
  clip against the axis-aligned rectilinear cell is used for every
  candidate pair; there is no Shapely fallback branch on this path
  because a triangle can never be non-convex or holed.
- **Both TriMesh** (``TriMesh, TriMesh``): every triangle is convex and
  has exactly 3 vertices, so an STRtree is used only for candidate-pair
  search (no analytic index is available for an irregular mesh), and
  the exact overlap area is computed with a dedicated numba/numpy
  triangle-triangle Sutherland-Hodgman clip -- no GEOS intersection
  call in the hot loop.
- **One side TriMesh, the other an arbitrary polygon grid**
  (``TriMesh, MultiPolygon``): candidate pairs are still found via
  STRtree (no analytic index available on either side), but since the
  TriMesh side is always convex, the convexity test is skipped for it
  and only the arbitrary-polygon side is checked. Convex candidates on
  that side bypass Shapely via a generic convex-vs-convex
  Sutherland-Hodgman clip (subject = arbitrary polygon, clip window =
  triangle); non-convex/holed candidates fall back to Shapely. Note
  this path still needs ``TriMesh.to_shapely()`` for the STRtree, since
  there is no analytic index available on either side here (unlike the
  rectilinear/TriMesh case above).

Every code path returns its per-pair ``intersections`` output as a
plain ``numpy`` object array (see :data:`IntersectionsArray`) rather
than a ``shapely.MultiPolygon``: an object array can represent every
case a clip can legitimately produce (a single ``Polygon``, a ``None``
placeholder for a fast analytic path that never materializes exact
geometry, or even a ``MultiPolygon`` when an exact Shapely clip against
a concave polygon splits into disjoint pieces), whereas
``shapely.MultiPolygon(...)`` cannot hold ``None`` entries or nested
multi-part geometries. This keeps the return type uniform across every
dispatch branch instead of varying between ``MultiPolygon`` and
``ndarray`` depending on what happened to come out of the clip.
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np
import shapely
from scipy.sparse import coo_array, csc_array
from shapely.strtree import STRtree

from quickpaver._grid import RectilinearGrid, TriMesh
from quickpaver._types import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt

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
    prange = range


#: Any of the three grid representations accepted on either side of a
#: transfer. ``RectilinearGrid`` and ``TriMesh`` are handled by
#: specialized, Shapely-light code paths; plain ``shapely.MultiPolygon``
#: is the fully generic fallback.
GridLike = Union[RectilinearGrid, TriMesh, shapely.MultiPolygon]

#: Return type used for the per-nonzero-entry ``intersections`` output
#: of every function in this module: a 1-D numpy object array, one
#: entry per matrix nonzero, holding either a ``shapely.Polygon``, a
#: ``shapely.MultiPolygon`` (possible on the exact Shapely-clip
#: fallback branches when a concave polygon's intersection splits into
#: disjoint pieces), or ``None`` (a placeholder used by the purely
#: analytic/Shapely-free paths, which never materialize exact
#: geometry). See :func:`_pack_intersections`.
IntersectionsArray = NDArrayFloat

#: Hard cap on Sutherland-Hodgman clip buffer width, and the size of
#: the per-call scratch arrays inside the numba clip kernels (where it
#: must be a compile-time constant). Actual batch buffers are sized to
#: fit the polygons at hand via :func:`_clip_buffer_width` rather than
#: always allocating this width.
_MAX_CLIP_VERTS = 32


def _clip_buffer_width(max_verts: int, n_clip_edges: int) -> int:
    """
    Width required for the padded Sutherland-Hodgman vertex buffers.

    Clipping a convex subject polygon against a convex window adds at
    most one vertex per clip edge, so a buffer of ``max_verts +
    n_clip_edges`` columns holds every possible clip result (4 edges
    for an axis-aligned rectangle, 3 for a triangle window).

    Sizing buffers this way rather than always allocating
    ``_MAX_CLIP_VERTS`` columns matters because these are ``(n_candidate
    _pairs, width)`` arrays: for the common case of 4-vertex cells the
    width drops from 32 to 8, cutting that allocation and the
    associated memory traffic four-fold.

    Raises
    ------
    ValueError
        If the required width exceeds :data:`_MAX_CLIP_VERTS`, which is
        also the size of the numba kernels' internal scratch arrays.
    """
    width = max_verts + n_clip_edges
    if width > _MAX_CLIP_VERTS:
        raise ValueError(
            f"Polygon with {max_verts} vertices needs a clip buffer of "
            f"{width} columns (max_verts + {n_clip_edges} slots for "
            "Sutherland-Hodgman output growth), exceeding "
            f"_MAX_CLIP_VERTS={_MAX_CLIP_VERTS}. Increase "
            "_MAX_CLIP_VERTS or exclude this polygon from the convex "
            "fast path."
        )
    return width


# ===================================================================
# Public API
# ===================================================================


def compute_transfer_matrix_with_intersections(
    source_grid: GridLike,
    target_grid: GridLike,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
    """
    Build a conservative transfer matrix between two polygonal grids,
    along with the per-pair intersection data.

    Dispatches to the fastest available implementation based on which
    kind each of the two grids is (see the module docstring for the
    full decision table).

    Parameters
    ----------
    source_grid : RectilinearGrid, TriMesh or shapely.MultiPolygon
        Source grid.
    target_grid : RectilinearGrid, TriMesh or shapely.MultiPolygon
        Target grid, with the same convention as ``source_grid``.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over the source grid cells (length must equal the
        number of source cells: ``nx * ny`` for a ``RectilinearGrid``,
        ``n_tri`` for a ``TriMesh``, ``len(source_grid.geoms)`` for a
        ``MultiPolygon``). Cells where the mask is ``False`` are
        excluded from the computation: they contribute no nonzero row
        in the returned matrix. If ``None``, all source cells are
        considered.
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
    intersections : IntersectionsArray, shape (nnz,)
        Intersection geometries, parallel to ``source_indices`` /
        ``target_indices``. Entries produced by a fast analytic-area
        path (used internally whenever at least one side is a
        ``RectilinearGrid`` or a ``TriMesh``) are returned as ``None``
        placeholders, since those paths compute areas without
        materializing exact intersection geometry.
    """
    return _dispatch_transfer(
        source_grid,
        target_grid,
        source_grid_mask=source_grid_mask,
        target_grid_mask=target_grid_mask,
        is_sanity_check=is_sanity_check,
        with_intersections=True,
    )


def _dispatch_transfer(
    source_grid: GridLike,
    target_grid: GridLike,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
    with_intersections: bool = True,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
    """
    Shared dispatch body for both public entry points.

    ``with_intersections`` controls whether per-pair intersection
    *geometry* is materialized. When ``False`` (used by
    :func:`compute_transfer_matrix`, which discards it anyway), every
    path skips its :func:`_polygons_from_padded_verts` /
    ``shapely.intersection`` geometry-construction step and returns an
    empty ``intersections`` array. The clip kernels still compute
    intersection *areas* -- those are what the matrix coefficients are
    made of -- so the saving is the GEOS ``Polygon`` object
    construction, which dominates the cost of the geometry output.
    """
    is_source_rect = isinstance(source_grid, RectilinearGrid)
    is_target_rect = isinstance(target_grid, RectilinearGrid)
    is_source_tri = isinstance(source_grid, TriMesh)
    is_target_tri = isinstance(target_grid, TriMesh)

    # ---- TriMesh vs TriMesh: dedicated triangle-triangle clip.
    #      Analytic (Shapely-free) area-only path -- always returns an
    #      empty `intersections`, so `_check_intersections_alignment`
    #      does not apply here (see its docstring). ----
    if is_source_tri and is_target_tri:
        return _compute_transfer_matrix_trimesh(
            source_grid,
            target_grid,
            source_grid_mask=source_grid_mask,
            target_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )

    # ---- TriMesh vs RectilinearGrid: dedicated Shapely-free path.
    #      Neither `to_shapely()` nor any Shapely geometry object is
    #      built for the mesh side -- triangle vertices are consumed
    #      directly as numpy arrays. Analytic area-only path -- always
    #      returns an empty `intersections`, so
    #      `_check_intersections_alignment` does not apply here (see
    #      its docstring). ----
    if is_source_tri and is_target_rect:
        return _compute_transfer_matrix_rect_trimesh(
            target_grid,
            source_grid,
            rectilinear_is_source=False,
            rectilinear_grid_mask=target_grid_mask,
            trimesh_mask=source_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )

    if is_target_tri and is_source_rect:
        return _compute_transfer_matrix_rect_trimesh(
            source_grid,
            target_grid,
            rectilinear_is_source=True,
            rectilinear_grid_mask=source_grid_mask,
            trimesh_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )

    # ---- TriMesh vs arbitrary MultiPolygon: STRtree candidates (no
    #      analytic index available on either side), triangle side
    #      known-convex so only the polygon side needs a convexity
    #      test. ----
    if is_source_tri and not is_target_rect:
        result = _compute_transfer_matrix_trimesh_polygon(
            source_grid,
            target_grid,  # ty:ignore[invalid-argument-type]
            trimesh_is_source=True,
            trimesh_mask=source_grid_mask,
            polygon_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )
        if with_intersections:
            _check_intersections_alignment(result[0], result[3])
        return result

    if is_target_tri and not is_source_rect:
        result = _compute_transfer_matrix_trimesh_polygon(
            target_grid,
            source_grid,  # ty:ignore[invalid-argument-type]
            trimesh_is_source=False,
            trimesh_mask=target_grid_mask,
            polygon_mask=source_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )
        if with_intersections:
            _check_intersections_alignment(result[0], result[3])
        return result

    # ---- Neither side is a TriMesh: original RectilinearGrid /
    #      MultiPolygon dispatch. ----
    if not is_source_rect and not is_target_rect:
        result = _compute_transfer_matrix(
            source_grid,  # ty:ignore[invalid-argument-type]
            target_grid,  # ty:ignore[invalid-argument-type]
            source_grid_mask=source_grid_mask,
            target_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )
        if with_intersections:
            _check_intersections_alignment(result[0], result[3])
        return result

    # Analytic (Shapely-free) area-only path -- always returns an
    # empty `intersections`, so `_check_intersections_alignment` does
    # not apply here (see its docstring).
    if is_source_rect and is_target_rect:
        return _compute_transfer_matrix_rectilinear(
            source_grid,
            target_grid,
            source_grid_mask=source_grid_mask,
            target_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )

    if is_source_rect and not is_target_rect:
        result = _compute_transfer_matrix_mixed(
            source_grid,
            target_grid,  # ty:ignore[invalid-argument-type]
            rectilinear_is_source=True,
            rectilinear_grid_mask=source_grid_mask,
            polygon_grid_mask=target_grid_mask,
            is_sanity_check=is_sanity_check,
            with_intersections=with_intersections,
        )
        if with_intersections:
            _check_intersections_alignment(result[0], result[3])
        return result

    # not is_source_rect and is_target_rect
    result = _compute_transfer_matrix_mixed(
        target_grid,  # ty:ignore[invalid-argument-type]
        source_grid,  # ty:ignore[invalid-argument-type]
        rectilinear_is_source=False,
        rectilinear_grid_mask=target_grid_mask,
        polygon_grid_mask=source_grid_mask,
        is_sanity_check=is_sanity_check,
        with_intersections=with_intersections,
    )
    if with_intersections:
        _check_intersections_alignment(result[0], result[3])
    return result


def compute_transfer_matrix(
    source_grid: GridLike,
    target_grid: GridLike,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> csc_array:
    """
    Build a conservative transfer matrix between two polygonal grids.

    Dispatches to the fastest available implementation based on which
    kind each of the two grids is (see the module docstring). This is
    a thin wrapper around
    :func:`compute_transfer_matrix_with_intersections` that discards
    the per-pair index/intersection data and returns only the sparse
    matrix.

    Parameters
    ----------
    source_grid : RectilinearGrid, TriMesh or shapely.MultiPolygon
        Source grid.
    target_grid : RectilinearGrid, TriMesh or shapely.MultiPolygon
        Target grid, with the same convention as ``source_grid``.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over the source grid cells. See
        :func:`compute_transfer_matrix_with_intersections`.
    target_grid_mask : 1-D array of bool, optional
        Boolean mask over the target grid cells. See
        :func:`compute_transfer_matrix_with_intersections`.
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

    Notes
    -----
    Because the per-pair intersection geometry is discarded anyway,
    this skips building it in the first place (see
    :func:`_dispatch_transfer`), avoiding one ``shapely.Polygon``
    construction per nonzero matrix entry.
    """
    return _dispatch_transfer(
        source_grid,
        target_grid,
        source_grid_mask=source_grid_mask,
        target_grid_mask=target_grid_mask,
        is_sanity_check=is_sanity_check,
        with_intersections=False,
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
    with_intersections: bool = True,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
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
    intersections : IntersectionsArray, shape (nnz,)
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

    source_polygons: np.typing.NDArray[shapely.Geometry] = np.asarray(
        source_grid.geoms,
        dtype=object,
    )

    target_polygons: np.typing.NDArray[shapely.Geometry] = np.asarray(
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

    pairs: NDArrayInt = tree.query(
        target_polygons,
        predicate="intersects",
    )

    target_indices: NDArrayInt = pairs[0]
    source_indices: NDArrayInt = pairs[1]

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

    intersections: NDArrayFloat = shapely.intersection(
        source_polygons[source_indices],
        target_polygons[target_indices],
    )

    intersection_areas = shapely.area(intersections)

    # Remove empty / numerical-noise intersections
    valid_mask: NDArrayBool = intersection_areas > 1e-15

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

    source_areas: NDArrayFloat = shapely.area(source_polygons)
    weights: NDArrayFloat = intersection_areas / source_areas[source_indices]

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
        _pack_intersections(intersections)
        if with_intersections
        else _no_intersections(),
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
    with_intersections: bool = True,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
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
    intersections : IntersectionsArray, shape (nnz,)
        One intersection ``Polygon`` per nonzero matrix entry, built
        from the same clipped vertices already produced by the
        separable/non-separable analytic kernels via
        :func:`_polygons_from_padded_verts` -- no GEOS clipping call,
        only vectorized ``shapely.polygons`` construction from
        already-known vertex coordinates.

    Notes
    -----
    When ``|target_angle - source_angle|`` is a multiple of 90 degrees
    (within 1e-9 rad) the overlap factorises into two 1-D problems.
    Otherwise a numba-parallel Sutherland-Hodgman clipper computes
    intersection areas (with a numpy-only fallback). No Shapely
    dependency for the area computation itself in either case.
    """
    n_source = source_grid.nx * source_grid.ny
    n_target = target_grid.nx * target_grid.ny

    source_grid_mask = _validate_mask(source_grid_mask, n_source, "source_grid_mask")
    target_grid_mask = _validate_mask(target_grid_mask, n_target, "target_grid_mask")

    src_lin, tgt_lin, weights, intersections = (
        _compute_transfer_matrix_rectilinear_impl(
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
            with_intersections=with_intersections,
        )
    )

    # -- apply masks directly on the raw (src_lin, tgt_lin, ...) arrays
    #    -- these are the single source of truth for both the matrix
    #    and `intersections`, so masking them in lockstep here (rather
    #    than reconstructing a mask via `mat.tocoo()` afterward) keeps
    #    `intersections` guaranteed aligned with the final nonzero
    #    entries regardless of how `scipy.sparse` orders them
    #    internally. --
    if source_grid_mask is not None or target_grid_mask is not None:
        keep = np.ones(len(src_lin), dtype=bool)
        if source_grid_mask is not None:
            keep &= source_grid_mask[src_lin]
        if target_grid_mask is not None:
            keep &= target_grid_mask[tgt_lin]
        src_lin = src_lin[keep]
        tgt_lin = tgt_lin[keep]
        weights = weights[keep]
        # `intersections` is deliberately empty when geometry was not
        # requested, in which case there is nothing to keep in lockstep.
        if with_intersections:
            intersections = intersections[keep]

    mat = coo_array(
        (weights, (src_lin, tgt_lin)),
        shape=(n_source, n_target),
    ).tocsc()

    if is_sanity_check:
        _check_conservation(mat)

    return (
        mat,
        src_lin,
        tgt_lin,
        _pack_intersections(intersections)
        if with_intersections
        else _no_intersections(),
    )


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
    with_intersections: bool = True,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayFloat, NDArrayFloat]:
    """
    Returns the raw ``(src_lin, tgt_lin, weights, intersections)``
    quadruple -- see :func:`_separable_transfer` for why the matrix is
    not assembled here (COO/CSC reordering hazard for a separately
    tracked ``intersections`` array).
    """
    source_center = np.asarray(source_center, dtype=float)
    target_center = np.asarray(target_center, dtype=float)

    # ---- check whether the fast separable path applies ----
    source_angle_rad = np.deg2rad(source_angle_deg)
    target_angle_rad = np.deg2rad(target_angle_deg)
    rel_angle_rad = target_angle_rad - source_angle_rad
    # NOTE: `rel_angle_rad` is already in radians -- applying `np.deg2rad`
    # here (as an earlier revision did) scales by pi/180 a second time and
    # makes `k_exact` ~0 for every input, so only an exactly-zero relative
    # angle was ever detected as separable and the 90/180/270-degree cases
    # silently fell through to the much heavier non-separable clip.
    k_exact = rel_angle_rad / (np.pi / 2)
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
            with_intersections=with_intersections,
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
            with_intersections=with_intersections,
        )


# ===================================================================
# Fast separable path  (relative angle = k x 90 degrees)
# ===================================================================


def _separable_transfer(
    source_center: NDArrayFloat,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle: float,
    target_center: NDArrayFloat,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    k90: int,
    with_intersections: bool = True,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayFloat, NDArrayFloat]:
    """
    Separable transfer for relative rotation = *k90* x 90 degrees.

    Returns the raw ``(src_lin, tgt_lin, weights, intersections)``
    quadruple rather than an assembled matrix, so that
    :func:`_compute_transfer_matrix_rectilinear_impl` /
    :func:`_compute_transfer_matrix_rectilinear` can build the
    ``source_indices``/``target_indices`` outputs directly from these
    same arrays (never from ``mat.tocoo()``, whose row/col ordering
    ``scipy.sparse`` is free to permute during ``coo_array(...).tocsc()``
    construction -- which would silently desynchronize a separately
    computed ``intersections`` array from the matrix's own nonzero
    iteration order).

    Returns
    -------
    src_lin, tgt_lin : NDArrayInt, shape (n_pairs,)
        Source / target cell linear index of each candidate pair with
        positive overlap (``is_sanity_check`` is applied by the
        caller once the final matrix is assembled).
    weights : ndarray, shape (n_pairs,)
        ``area(S_i ∩ T_j) / area(S_i)`` for each pair.
    intersections : ndarray of object, shape (n_pairs,)
        One intersection ``Polygon`` (in world coordinates) per pair,
        aligned with ``src_lin``/``tgt_lin``/``weights``. Since both
        grids are axis-aligned in the source's local frame on this
        path, each intersection is exactly the axis-aligned overlap
        rectangle from the outer product of the two 1-D overlap
        intervals -- built directly from ``lo``/``hi`` returned by
        :func:`_compute_1d_overlaps`, with no clipping algorithm
        needed at all (not even the numba/numpy kernels used
        elsewhere -- the 1-D interval overlap already *is* the exact
        rectangle bound).
    """

    # -- source edges in source-local frame (always ascending) --
    src_x_edges = (
        np.arange(source_nx + 1, dtype=np.float64) - source_nx / 2
    ) * source_dx
    src_y_edges = (
        np.arange(source_ny + 1, dtype=np.float64) - source_ny / 2
    ) * source_dy

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

    is_x, it_x_sorted, ox, xlo, xhi = _compute_1d_overlaps(src_x_edges, tgt_x_edges)
    is_y, it_y_sorted, oy, ylo, yhi = _compute_1d_overlaps(src_y_edges, tgt_y_edges)

    _e_int = np.empty(0, dtype=np.intp)
    if len(is_x) == 0 or len(is_y) == 0:
        return _e_int, _e_int, np.empty(0), np.empty(0, dtype=object)

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

    # -- exact intersection rectangles: outer product of the x- and
    #    y-overlap intervals, in source-local frame, then rotated to
    #    world coordinates. Every entry here is a genuine 4-vertex
    #    axis-aligned (in local frame) rectangle -- no padding/vertex
    #    count bookkeeping needed since the vertex count is always
    #    exactly 4. --
    if with_intersections:
        rect_xlo = np.repeat(xlo, ny_pairs)
        rect_xhi = np.repeat(xhi, ny_pairs)
        rect_ylo = np.tile(ylo, nx_pairs)
        rect_yhi = np.tile(yhi, nx_pairs)
        local_vx = np.stack(
            [rect_xlo, rect_xhi, rect_xhi, rect_xlo], axis=-1
        )  # (n_pairs, 4)
        local_vy = np.stack([rect_ylo, rect_ylo, rect_yhi, rect_yhi], axis=-1)
        n_verts = np.full(len(rect_xlo), 4, dtype=np.intp)

        intersections = _polygons_from_padded_verts(
            local_vx,
            local_vy,
            n_verts,
            local_to_world=(source_center, ca, sa),
        )
    else:
        intersections = _no_intersections()

    return src_lin, tgt_lin, weights, intersections


def _sort_edges(edges: NDArrayFloat, n: int) -> Tuple[NDArrayFloat, NDArrayInt]:
    """Return ascending edges and a mapping from sorted cell index to original."""
    if edges[-1] >= edges[0]:
        return edges, np.arange(n)
    return edges[::-1].copy(), np.arange(n - 1, -1, -1)


def _compute_1d_overlaps(
    edges_a: NDArrayFloat, edges_b: NDArrayFloat
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
    """
    Find all overlapping interval pairs between two sorted edge arrays.

    Returns ``(idx_a, idx_b, overlaps, lo, hi)`` with only positive
    overlaps, where ``lo``/``hi`` are the overlap interval's own
    bounds (``lo = max(left_a, left_b)``, ``hi = min(right_a,
    right_b)``) -- needed by callers that build exact intersection
    rectangles (not just areas) from the same 1-D overlap computation.
    """
    na = len(edges_a) - 1
    nb = len(edges_b) - 1
    if na == 0 or nb == 0:
        _e = np.empty(0, dtype=np.intp)
        _f = np.empty(0)
        return _e, _e, _f, _f, _f

    right_b, left_b = edges_b[1:], edges_b[:-1]
    left_a, right_a = edges_a[:-1], edges_a[1:]

    j_starts = np.searchsorted(right_b, left_a, side="right")
    j_ends = np.searchsorted(left_b, right_a, side="left")

    counts = np.maximum(j_ends - j_starts, 0)
    total = counts.sum()
    if total == 0:
        _e = np.empty(0, dtype=np.intp)
        _f = np.empty(0)
        return _e, _e, _f, _f, _f

    idx_a = np.repeat(np.arange(na, dtype=np.intp), counts)
    cum = np.empty(na + 1, dtype=np.intp)
    cum[0] = 0
    np.cumsum(counts, out=cum[1:])
    group_offset = np.arange(total, dtype=np.intp) - np.repeat(cum[:-1], counts)
    idx_b = group_offset + np.repeat(j_starts, counts)

    lo = np.maximum(left_a[idx_a], left_b[idx_b])
    hi = np.minimum(right_a[idx_a], right_b[idx_b])
    overlaps = hi - lo
    valid = overlaps > 1e-15
    return idx_a[valid], idx_b[valid], overlaps[valid], lo[valid], hi[valid]


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
    source_center: NDArrayFloat,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle: float,
    target_center: NDArrayFloat,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    target_angle: float,
    with_intersections: bool = True,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayFloat, NDArrayFloat]:
    """
    Transfer matrix for two rectilinear grids at an arbitrary angle.

    Returns the raw ``(src_lin, tgt_lin, weights, intersections)``
    quadruple rather than an assembled matrix -- see
    :func:`_separable_transfer` for why (avoiding the COO/CSC
    reordering hazard that would otherwise desynchronize
    ``intersections`` from the matrix's own nonzero iteration order).

    Returns
    -------
    src_lin, tgt_lin : NDArrayInt, shape (n_pairs,)
    weights : ndarray, shape (n_pairs,)
    intersections : ndarray of object, shape (n_pairs,)
        One intersection ``Polygon`` (in world coordinates) per pair,
        built via :func:`_polygons_from_padded_verts` from the same
        clipped vertices the area computation already produced -- no
        GEOS clipping call, only vectorized ``shapely.polygons``
        construction.
    """

    src_cell_area = source_dx * source_dy
    n_target = target_nx * target_ny

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
        _e = np.empty(0, dtype=np.intp)
        return _e, _e, np.empty(0), np.empty(0, dtype=object)

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

    # -- compute intersection areas AND clipped vertices --
    areas, out_x, out_y, out_n = _batch_clip_areas_and_verts(
        np.ascontiguousarray(pair_tvx),
        np.ascontiguousarray(pair_tvy),
        pair_xmin,
        pair_ymin,
        pair_xmax,
        pair_ymax,
    )

    # -- filter --
    valid = areas > 1e-15
    weights = areas[valid] / src_cell_area

    # -- build intersection geometry from the same clip vertices, in
    #    the source's world frame (out_x/out_y are in source-local
    #    frame, matching the frame `pair_xmin` etc. were computed in) --
    if with_intersections:
        intersections = _polygons_from_padded_verts(
            out_x[valid],
            out_y[valid],
            out_n[valid],
            local_to_world=(source_center, ca, sa),
        )
    else:
        intersections = _no_intersections()

    return src_lin[valid], tgt_lin[valid], weights, intersections


# ===================================================================
# Intersection-area back-ends (numba primary, numpy fallback)
# ===================================================================


def _batch_clip_areas_and_verts(
    tvx: NDArrayFloat,
    tvy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """
    Dispatch to the fastest available back-end, also returning the
    clipped polygon vertices (padded to a fixed width, with a
    per-pair vertex count) so callers can build exact ``shapely``
    geometry from them via a single vectorized ``shapely.polygons``
    call -- without ever invoking GEOS's own clipping algorithm.

    Returns
    -------
    areas : ndarray, shape (N,)
    out_x, out_y : ndarray, shape (N, 8)
        Padded clipped-polygon vertex coordinates; only the first
        ``out_n[k]`` columns of row ``k`` are meaningful.
    out_n : ndarray, shape (N,)
        Number of meaningful vertices per row (0 for an empty clip).
    """
    if _HAS_NUMBA:
        return _batch_clip_numba_and_verts(tvx, tvy, xmin, ymin, xmax, ymax)  # ty: ignore[invalid-argument-type]
    return _batch_clip_numpy_and_verts(tvx, tvy, xmin, ymin, xmax, ymax)


# ---- numba back-end ------------------------------------------------


@njit(cache=True)
def _clip_single_with_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    out_x: NDArrayFloat,
    out_y: NDArrayFloat,
) -> Tuple[float, int]:
    """
    Sutherland-Hodgman clip of a 4-vertex polygon against an
    axis-aligned rect, also writing the clipped polygon's vertices
    into ``out_x``/``out_y`` (each length >= 8) and returning
    ``(area, n_verts)``.
    """
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
            return 0.0, 0
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
        return 0.0, 0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
        out_x[i] = ax[i]
        out_y[i] = ay[i]
    return abs(area) * 0.5, n_in


@njit(parallel=True, cache=True)
def _batch_clip_numba_and_verts(
    tvx: NDArrayFloat,
    tvy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    N = len(xmin)
    areas = np.empty(N)
    out_x = np.zeros((N, 8))
    out_y = np.zeros((N, 8))
    out_n = np.zeros(N, dtype=np.intp)
    for idx in prange(N):  # ty:ignore[not-iterable]
        area, n_verts = _clip_single_with_verts(
            tvx[idx],
            tvy[idx],
            xmin[idx],
            ymin[idx],
            xmax[idx],
            ymax[idx],
            out_x[idx],
            out_y[idx],
        )
        areas[idx] = area
        out_n[idx] = n_verts
    return areas, out_x, out_y, out_n


# ---- numpy fallback ------------------------------------------------


def _batch_clip_numpy_core(
    tvx: NDArrayFloat,
    tvy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt, NDArrayInt]:
    """
    Shared core for the numpy fallback: collects vertices from three
    sources (target corners in source, source corners in target,
    edge-edge intersections), sorts by angle from centroid, and
    returns both the shoelace areas and the angle-sorted vertex
    buffer -- so the vertex-returning entry point
    (:func:`_batch_clip_numpy_and_verts`) does not have to duplicate
    the vertex-collection logic.
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
    vc = np.where(areas > 0.0, vc, 0)
    return areas, sx, sy, vc, order


def _batch_clip_numpy_and_verts(
    tvx: NDArrayFloat,
    tvy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """
    Numpy-only fallback for :func:`_batch_clip_numba_and_verts`: vertex
    collection + angle-sort + shoelace, returning the angle-sorted
    clipped-polygon vertices (padded to 24 columns, matching
    ``_batch_clip_numpy_core``'s internal buffer width) plus a
    per-row vertex count, so callers can build exact ``shapely``
    geometry from them.
    """
    areas, sx, sy, vc, _order = _batch_clip_numpy_core(tvx, tvy, xmin, ymin, xmax, ymax)
    return areas, sx, sy, vc


# ===================================================================
# Generalized N-vertex Sutherland-Hodgman clip against an
# AXIS-ALIGNED RECTANGLE (numba primary, numpy fallback), used by the
# rectilinear/polygon mixed path AND the rectilinear/TriMesh dedicated
# path to bypass Shapely entirely when the arbitrary-side polygons are
# convex (this always includes TriMesh triangles).
# ===================================================================


class _PolygonVertexData:
    """
    Exterior-ring vertex data for a batch of shapely polygons,
    extracted from GEOS exactly **once**.

    Three separate consumers in this module need the same exterior
    coordinates of the same polygon batch: the local-frame bounding-box
    candidate search, the convexity test, and the padded vertex buffers
    fed to the Sutherland-Hodgman clip kernels. Each of those used to
    run its own ``get_exterior_ring`` + ``get_coordinates`` pair (and
    the clip-buffer one ran it on an index array containing *duplicate*
    polygons -- once per candidate pair rather than once per polygon),
    so a polygon overlapping nine grid cells had its coordinates pulled
    out of GEOS eleven times. This class holds the single extraction all
    three now share.

    Attributes
    ----------
    x, y : ndarray, shape (n_coords,)
        Flat world-frame exterior coordinates of every polygon,
        concatenated in polygon order.
    starts : NDArrayInt, shape (n_poly,)
        Index into ``x``/``y`` where each polygon's ring begins.
    true_n : NDArrayInt, shape (n_poly,)
        Vertex count per polygon, *excluding* the duplicated closing
        vertex that shapely stores at the end of every ring.
    has_holes : NDArrayBool, shape (n_poly,)
        Whether each polygon has at least one interior ring.
    """

    __slots__ = ("x", "y", "starts", "true_n", "has_holes")

    def __init__(self, poly_polygons: NDArrayFloat) -> None:
        n_poly = len(poly_polygons)
        if n_poly == 0:
            _e = np.empty(0, dtype=np.intp)
            self.x = np.empty(0)
            self.y = np.empty(0)
            self.starts = _e
            self.true_n = _e
            self.has_holes = np.empty(0, dtype=bool)
            return

        self.has_holes = shapely.get_num_interior_rings(poly_polygons) > 0
        exteriors = shapely.get_exterior_ring(poly_polygons)
        coords, coord_poly_idx = shapely.get_coordinates(exteriors, return_index=True)
        self.x = coords[:, 0]
        self.y = coords[:, 1]
        # coord_poly_idx is sorted/non-decreasing, so searchsorted gives
        # the per-polygon ring boundaries directly. Counts follow from
        # the boundaries themselves -- a separate `np.bincount` pass
        # (as an earlier revision used) recomputes the same segmentation.
        boundaries = np.searchsorted(coord_poly_idx, np.arange(n_poly + 1), side="left")
        self.starts = boundaries[:-1]
        # shapely repeats the first vertex at the end of every ring;
        # drop it so `true_n` is the real vertex count.
        self.true_n = np.diff(boundaries) - 1

    @property
    def max_verts(self) -> int:
        """Largest true vertex count in the batch (0 if empty)."""
        return int(np.max(self.true_n)) if len(self.true_n) else 0

    def local_frame_bboxes(
        self, center: NDArrayFloat, angle_rad: float
    ) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayFloat]:
        """
        Per-polygon axis-aligned bounding box in a rotated grid's local
        frame, as ``(xmin, xmax, ymin, ymax)``.
        """
        if len(self.starts) == 0:
            _f = np.empty(0)
            return _f, _f, _f, _f
        ca, sa = np.cos(angle_rad), np.sin(angle_rad)
        dxw = self.x - center[0]
        dyw = self.y - center[1]
        local_x = dxw * ca + dyw * sa
        local_y = -dxw * sa + dyw * ca
        return (
            np.minimum.reduceat(local_x, self.starts),
            np.maximum.reduceat(local_x, self.starts),
            np.minimum.reduceat(local_y, self.starts),
            np.maximum.reduceat(local_y, self.starts),
        )

    def padded_vertex_buffers(
        self,
        width: int,
        local_to_grid: Optional[Tuple[NDArrayFloat, float]] = None,
    ) -> Tuple[NDArrayFloat, NDArrayFloat]:
        """
        Scatter every polygon's (non-closing) vertices into fixed-width
        ``(n_poly, width)`` buffers suitable for the clip kernels.

        Built **per polygon**, not per candidate pair: callers gather
        the rows they need with a fancy index, which is a plain memcpy
        rather than a repeated GEOS coordinate extraction.

        Parameters
        ----------
        width : int
            Buffer width; must be at least :attr:`max_verts`.
        local_to_grid : (center, angle_rad), optional
            If given, coordinates are rotated from world into that
            grid's local (unrotated) frame first -- required by the
            kernels that clip against an axis-aligned rectangle, since
            a rotated grid's cells are only axis-aligned in their own
            local frame.
        """
        n_poly = len(self.starts)
        vx = np.zeros((n_poly, width))
        vy = np.zeros((n_poly, width))
        if n_poly == 0:
            return vx, vy

        x, y = self.x, self.y
        if local_to_grid is not None:
            center, angle_rad = local_to_grid
            ca, sa = np.cos(angle_rad), np.sin(angle_rad)
            dxw = x - center[0]
            dyw = y - center[1]
            x = dxw * ca + dyw * sa
            y = -dxw * sa + dyw * ca

        counts = self.true_n + 1  # includes the closing vertex
        point_poly_idx = np.repeat(np.arange(n_poly), counts)
        local_col = np.arange(len(x)) - self.starts[point_poly_idx]
        # drop the closing vertex, which always lands at local_col == true_n
        keep = local_col < self.true_n[point_poly_idx]
        vx[point_poly_idx[keep], local_col[keep]] = x[keep]
        vy[point_poly_idx[keep], local_col[keep]] = y[keep]
        return vx, vy

    def convexity(self) -> NDArrayBool:
        """
        Vectorised per-polygon convexity test.

        A simple polygon (no self-intersections, no holes) is convex iff
        the signed cross product at every vertex has a constant sign.
        Polygons with interior rings are conservatively reported
        non-convex, since a polygon with a hole can never be convex.

        Notes
        -----
        Callers that already know every polygon is convex (e.g. a batch
        of :class:`~quickpaver._grid.TriMesh` triangles, convex by
        construction) should skip this rather than call it -- see
        ``polygon_grid_is_convex`` on
        :func:`_compute_transfer_matrix_mixed`. Better still, the
        dedicated :func:`_compute_transfer_matrix_rect_trimesh` path
        never builds Shapely polygons for the mesh side at all.
        """
        n_poly = len(self.starts)
        if n_poly == 0:
            return np.empty(0, dtype=bool)

        x, y = self.x, self.y
        counts = self.true_n + 1
        point_poly_idx = np.repeat(np.arange(n_poly), counts)
        local_pos = np.arange(len(x)) - np.repeat(self.starts, counts)
        rep_true_n = np.repeat(self.true_n, counts)

        active = local_pos < rep_true_n  # drop the duplicated closing vertex
        active_poly_idx = point_poly_idx[active]
        active_local_pos = local_pos[active]
        active_true_n = rep_true_n[active]
        active_x = x[active]
        active_y = y[active]
        active_starts = self.starts[active_poly_idx]

        prev_global = active_starts + (active_local_pos - 1) % active_true_n
        next_global = active_starts + (active_local_pos + 1) % active_true_n

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

        return ~(has_pos & has_neg) & ~(self.true_n < 3) & ~self.has_holes


@njit(cache=True)
def _clip_nverts_with_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    n_verts: int,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    out_x: NDArrayFloat,
    out_y: NDArrayFloat,
) -> Tuple[float, int]:
    """
    Sutherland-Hodgman clip of a convex ``n_verts``-vertex polygon
    against an axis-aligned rect, also writing the clipped polygon's
    vertices into ``out_x``/``out_y`` (each length >=
    ``_MAX_CLIP_VERTS``) and returning ``(area, n_out_verts)``.
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
            return 0.0, 0
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
        return 0.0, 0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
        out_x[i] = ax[i]
        out_y[i] = ay[i]
    return abs(area) * 0.5, n_in


@njit(parallel=True, cache=True)
def _batch_clip_numba_nverts_and_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    n_verts: NDArrayInt,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    N = len(xmin)
    width = vx.shape[1]
    areas = np.empty(N)
    out_x = np.zeros((N, width))
    out_y = np.zeros((N, width))
    out_n = np.zeros(N, dtype=np.intp)
    for idx in prange(N):  # ty:ignore[not-iterable]
        area, n_out_verts = _clip_nverts_with_verts(
            vx[idx],
            vy[idx],
            n_verts[idx],
            xmin[idx],
            ymin[idx],
            xmax[idx],
            ymax[idx],
            out_x[idx],
            out_y[idx],
        )
        areas[idx] = area
        out_n[idx] = n_out_verts
    return areas, out_x, out_y, out_n


def _sh_clip_python_with_verts(
    poly: list, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[float, list]:
    """Pure-Python Sutherland-Hodgman clip of an N-vertex polygon
    against an axis-aligned rect, returning ``(area, clipped_verts)``.
    """
    edges = (
        ("x", ">=", xmin),
        ("x", "<=", xmax),
        ("y", ">=", ymin),
        ("y", "<=", ymax),
    )
    output = poly
    for axis, op, ev in edges:
        if len(output) < 3:
            return 0.0, []
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
        return 0.0, []
    area = 0.0
    n = len(output)
    for i in range(n):
        j = (i + 1) % n
        area += output[i][0] * output[j][1] - output[j][0] * output[i][1]
    return abs(area) * 0.5, output


def _batch_clip_numpy_nverts_and_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    n_verts: NDArrayInt,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Numpy/pure-Python fallback for :func:`_batch_clip_numba_nverts_and_verts`."""
    N = len(xmin)
    width = vx.shape[1]
    areas = np.empty(N)
    out_x = np.zeros((N, width))
    out_y = np.zeros((N, width))
    out_n = np.zeros(N, dtype=np.intp)
    for k in range(N):
        m = int(n_verts[k])
        poly = list(zip(vx[k, :m].tolist(), vy[k, :m].tolist()))
        area, out_poly = _sh_clip_python_with_verts(
            poly, xmin[k], ymin[k], xmax[k], ymax[k]
        )
        areas[k] = area
        mo = len(out_poly)
        out_n[k] = mo
        for p in range(mo):
            out_x[k, p] = out_poly[p][0]
            out_y[k, p] = out_poly[p][1]
    return areas, out_x, out_y, out_n


def _batch_clip_areas_and_verts_nverts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    n_verts: NDArrayInt,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Dispatch to the fastest available N-vertex clip back-end, also
    returning clipped vertices."""
    if _HAS_NUMBA:
        return _batch_clip_numba_nverts_and_verts(
            vx,  # ty: ignore[invalid-argument-type]
            vy,  # ty: ignore[invalid-argument-type]
            n_verts,  # ty: ignore[invalid-argument-type]
            xmin,  # ty: ignore[invalid-argument-type]
            ymin,  # ty: ignore[invalid-argument-type]
            xmax,  # ty: ignore[invalid-argument-type]
            ymax,  # ty: ignore[invalid-argument-type]
        )
    return _batch_clip_numpy_nverts_and_verts(vx, vy, n_verts, xmin, ymin, xmax, ymax)


# ===================================================================
# Fixed 3-vertex Sutherland-Hodgman clip against an axis-aligned
# rectangle, specialized for triangle subjects (used by the dedicated
# rectilinear/TriMesh path). This is functionally a special case of
# `_clip_nverts_with_verts` with `n_verts == 3` hard-coded, but
# skipping the ragged bookkeeping (no `n_verts` array, no padding
# beyond the fixed 3 input columns) shaves a bit more overhead off the
# already-cheap triangle case, and lets the caller skip building an
# `n_verts` array full of 3s.
# ===================================================================


@njit(cache=True)
def _clip_tri_rect_with_verts(
    vx0: float,
    vx1: float,
    vx2: float,
    vy0: float,
    vy1: float,
    vy2: float,
    xmin: float,
    ymin: float,
    xmax: float,
    ymax: float,
    out_x: NDArrayFloat,
    out_y: NDArrayFloat,
) -> Tuple[float, int]:
    """
    Sutherland-Hodgman clip of a 3-vertex triangle against an
    axis-aligned rect, also writing the clipped polygon's vertices
    into ``out_x``/``out_y`` (each length >= 8) and returning
    ``(area, n_verts)``.
    """
    MAX_V = 8
    ax = np.empty(MAX_V)
    ay = np.empty(MAX_V)
    bx = np.empty(MAX_V)
    by = np.empty(MAX_V)
    ax[0], ax[1], ax[2] = vx0, vx1, vx2
    ay[0], ay[1], ay[2] = vy0, vy1, vy2
    n_in = 3

    edges = (xmin, xmax, ymin, ymax)
    for e in range(4):
        ev = edges[e]
        n_out = 0
        if n_in < 3:
            return 0.0, 0
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
        return 0.0, 0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
        out_x[i] = ax[i]
        out_y[i] = ay[i]
    return abs(area) * 0.5, n_in


@njit(parallel=True, cache=True)
def _batch_clip_tri_rect_and_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    N = len(xmin)
    areas = np.empty(N)
    out_x = np.zeros((N, 8))
    out_y = np.zeros((N, 8))
    out_n = np.zeros(N, dtype=np.intp)
    for idx in prange(N):  # ty:ignore[not-iterable]
        area, n_verts = _clip_tri_rect_with_verts(
            vx[idx, 0],
            vx[idx, 1],
            vx[idx, 2],
            vy[idx, 0],
            vy[idx, 1],
            vy[idx, 2],
            xmin[idx],
            ymin[idx],
            xmax[idx],
            ymax[idx],
            out_x[idx],
            out_y[idx],
        )
        areas[idx] = area
        out_n[idx] = n_verts
    return areas, out_x, out_y, out_n


def _batch_clip_tri_rect_numpy_with_verts_python(
    poly: list, xmin: float, ymin: float, xmax: float, ymax: float
) -> Tuple[float, list]:
    """Pure-Python Sutherland-Hodgman clip of a triangle against an
    axis-aligned rect, returning ``(area, clipped_verts)``."""
    edges = (
        ("x", ">=", xmin),
        ("x", "<=", xmax),
        ("y", ">=", ymin),
        ("y", "<=", ymax),
    )
    output = poly
    for axis, op, ev in edges:
        if len(output) < 3:
            return 0.0, []
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
        return 0.0, []
    area = 0.0
    n = len(output)
    for i in range(n):
        j = (i + 1) % n
        area += output[i][0] * output[j][1] - output[j][0] * output[i][1]
    return abs(area) * 0.5, output


def _batch_clip_tri_rect_numpy_and_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Numpy/pure-Python fallback for :func:`_batch_clip_tri_rect_and_verts`."""
    N = len(xmin)
    areas = np.empty(N)
    out_x = np.zeros((N, 8))
    out_y = np.zeros((N, 8))
    out_n = np.zeros(N, dtype=np.intp)
    for k in range(N):
        poly = [(vx[k, 0], vy[k, 0]), (vx[k, 1], vy[k, 1]), (vx[k, 2], vy[k, 2])]
        area, out_poly = _batch_clip_tri_rect_numpy_with_verts_python(
            poly, xmin[k], ymin[k], xmax[k], ymax[k]
        )
        areas[k] = area
        m = len(out_poly)
        out_n[k] = m
        for p in range(m):
            out_x[k, p] = out_poly[p][0]
            out_y[k, p] = out_poly[p][1]
    return areas, out_x, out_y, out_n


def _batch_clip_areas_and_verts_tri_rect(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    xmin: NDArrayFloat,
    ymin: NDArrayFloat,
    xmax: NDArrayFloat,
    ymax: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Dispatch to the fastest available triangle-vs-rect clip back-end,
    also returning clipped vertices (see
    :func:`_batch_clip_areas_and_verts_nverts` for the analogous
    N-vertex-input version)."""
    if _HAS_NUMBA:
        return _batch_clip_tri_rect_and_verts(
            np.ascontiguousarray(vx),  # ty: ignore[invalid-argument-type]
            np.ascontiguousarray(vy),  # ty: ignore[invalid-argument-type]
            xmin,  # ty: ignore[invalid-argument-type]
            ymin,  # ty: ignore[invalid-argument-type]
            xmax,  # ty: ignore[invalid-argument-type]
            ymax,  # ty: ignore[invalid-argument-type]
        )
    return _batch_clip_tri_rect_numpy_and_verts(vx, vy, xmin, ymin, xmax, ymax)


# ===================================================================
# Mixed path: exactly one grid is rectilinear, the other arbitrary
# polygons.
#
# Idea: build the source-local candidate ranges analytically (as in
# `_nonseparable_transfer`), instead of building an STRtree. Two
# back-ends for the actual clip:
#
# - Convex polygons (checked via `_PolygonVertexData.convexity`): bypass
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
#
# NOTE: the TriMesh <-> RectilinearGrid combination no longer goes
# through this function -- see `_compute_transfer_matrix_rect_trimesh`
# below, which reuses the candidate-search helpers here but never
# builds Shapely polygons for the mesh side.
# ===================================================================


def _rectilinear_cell_corners_world(
    center: NDArrayFloat,
    angle_rad: float,
    x_edges: NDArrayFloat,
    y_edges: NDArrayFloat,
    cell_lin: NDArrayInt,
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    World-frame corner coordinates of the requested rectilinear cells.

    Parameters
    ----------
    center : ndarray, shape (2,)
        Grid centre in world coordinates.
    angle_rad : float
        Grid rotation in radians.
    x_edges, y_edges : ndarray
        Cell boundaries in the grid's own local (unrotated) frame.
    cell_lin : NDArrayInt, shape (n,)
        Linear cell indices (``j * nx + i``, ``x`` fastest) to build
        corners for. May contain duplicates.

    Returns
    -------
    corners_x, corners_y : ndarray, shape (n, 4)
        Counter-clockwise world-frame corners, one row per requested
        cell.

    Notes
    -----
    Built **on demand for the requested cells only**. An earlier
    revision materialized the full ``(nx * ny, 4)`` corner arrays for
    the entire grid up front, even though the convex fast path never
    reads them (it clips in the grid's local frame, where a cell's
    extent is just two edge lookups) -- so for an all-convex polygon
    grid the whole allocation was pure waste, and for a mixed one only
    the non-convex candidates need it.
    """
    ca, sa = np.cos(angle_rad), np.sin(angle_rad)
    nx = len(x_edges) - 1
    cell_j, cell_i = np.divmod(cell_lin, nx)

    x0 = x_edges[cell_i]
    x1 = x_edges[cell_i + 1]
    y0 = y_edges[cell_j]
    y1 = y_edges[cell_j + 1]

    local_cx = np.stack([x0, x1, x1, x0], axis=-1)
    local_cy = np.stack([y0, y0, y1, y1], axis=-1)

    corners_x = center[0] + local_cx * ca - local_cy * sa
    corners_y = center[1] + local_cx * sa + local_cy * ca
    return corners_x, corners_y


def _rectilinear_local_frame_params(
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    angle_deg: float,
) -> Tuple[float, float, Tuple[NDArrayFloat, NDArrayFloat]]:
    """
    Cheap scalar/edge description of a rectilinear grid.

    Returns
    -------
    angle_rad : float
        Grid rotation in radians.
    cell_area : float
        Area of a single cell (``dx * dy``).
    edges_local : tuple(ndarray, ndarray)
        ``(x_edges, y_edges)`` 1-D arrays of length ``nx + 1`` /
        ``ny + 1`` giving cell boundaries in the grid's own local
        (unrotated) frame.
    """
    return (
        np.deg2rad(angle_deg),
        dx * dy,
        (
            (np.arange(nx + 1, dtype=np.float64) - nx / 2) * dx,
            (np.arange(ny + 1, dtype=np.float64) - ny / 2) * dy,
        ),
    )


def _bbox_candidate_ranges_in_rectilinear_grid(
    bbox_xmin: NDArrayFloat,
    bbox_xmax: NDArrayFloat,
    bbox_ymin: NDArrayFloat,
    bbox_ymax: NDArrayFloat,
    x_edges: NDArrayFloat,
    y_edges: NDArrayFloat,
    nx: int,
    ny: int,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Core O(1)-per-item candidate-range arithmetic shared by every
    "rectilinear vs many small shapes" path in this module: given, for
    each of ``n_items`` shapes, its axis-aligned bounding box already
    expressed in the rectilinear grid's own *local* (unrotated) frame,
    return the flattened list of ``(item_idx, cell_i, cell_j)``
    candidate triples covering every grid cell whose axis-aligned
    extent overlaps that bbox.

    This is a pure index-arithmetic routine -- no geometry, no Shapely,
    no per-item Python loop -- and is reused both by the generic
    rectilinear/polygon path (:func:`_compute_transfer_matrix_mixed`,
    which gets its bboxes from
    :meth:`_PolygonVertexData.local_frame_bboxes`) and
    by the dedicated rectilinear/TriMesh path
    (:func:`_compute_transfer_matrix_rect_trimesh`, which computes the
    bboxes directly from the ``(n_tri, 3)`` triangle vertex arrays via
    plain numpy ``min``/``max`` reductions).

    Returns
    -------
    item_idx, cell_i, cell_j : NDArrayInt
        Flattened arrays such that for entry ``k``, item ``item_idx[k]``
        has candidate grid cell ``(cell_i[k], cell_j[k])``.
    """
    n_items = len(bbox_xmin)
    if n_items == 0:
        _e = np.empty(0, dtype=np.intp)
        return _e, _e, _e

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

    item_idx = np.repeat(np.arange(n_items, dtype=np.intp), counts)
    cum = np.empty(n_items + 1, dtype=np.intp)
    cum[0] = 0
    np.cumsum(counts, out=cum[1:])
    local_pos = np.arange(total, dtype=np.intp) - np.repeat(cum[:-1], counts)

    rep_nj = np.repeat(nj, counts)
    cell_i = np.repeat(i_lo, counts) + local_pos // rep_nj
    cell_j = np.repeat(j_lo, counts) + local_pos % rep_nj

    return item_idx, cell_i, cell_j


def _compute_transfer_matrix_mixed(
    rectilinear_grid: RectilinearGrid,
    polygon_grid: shapely.MultiPolygon,
    rectilinear_is_source: bool,
    rectilinear_grid_mask: Optional[NDArrayBool] = None,
    polygon_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
    with_intersections: bool = True,
    polygon_grid_is_convex: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
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
    polygon_grid_is_convex : bool, optional
        If ``True``, skip the ``_PolygonVertexData.convexity`` test entirely
        and treat every polygon in ``polygon_grid`` as convex. By
        default ``False``, meaning convexity is detected automatically.
        (The TriMesh case that used to set this flag now goes through
        the dedicated :func:`_compute_transfer_matrix_rect_trimesh`
        path instead, which needs no Shapely polygons at all for the
        mesh side; this flag remains for any other caller that already
        knows its polygons are convex.)

    Returns
    -------
    W : csc_array, shape (n_source, n_target)
        Sparse conservative transfer matrix ``W[i, j] = |S_i ∩ T_j| / |S_i|``.
    source_indices, target_indices : NDArrayInt, shape (nnz,)
        Row / column index of each nonzero entry.
    intersections : IntersectionsArray, shape (nnz,)
        Intersection geometries for each nonzero entry (parallel to
        ``source_indices`` / ``target_indices``). Entries produced by
        the fast convex-clip path are returned as ``None``
        placeholders, since that path computes areas without
        materializing exact intersection geometry.
    """
    rectilinear_center = np.asarray(
        (rectilinear_grid.cx, rectilinear_grid.cy), dtype=float
    )
    rectilinear_nx = rectilinear_grid.nx
    rectilinear_ny = rectilinear_grid.ny

    n_rect = rectilinear_nx * rectilinear_ny
    n_poly = len(polygon_grid.geoms)

    rectilinear_grid_mask = _validate_mask(
        rectilinear_grid_mask, n_rect, "rectilinear grid mask"
    )
    polygon_grid_mask = _validate_mask(polygon_grid_mask, n_poly, "polygon grid mask")

    rect_angle_rad, rect_cell_area, (x_edges, y_edges) = (
        _rectilinear_local_frame_params(
            rectilinear_grid.dx,
            rectilinear_grid.dy,
            rectilinear_nx,
            rectilinear_ny,
            rectilinear_grid.theta,
        )
    )

    poly_polygons: NDArrayFloat = np.asarray(polygon_grid.geoms, dtype=object)

    n_source = n_rect if rectilinear_is_source else n_poly
    n_target = n_poly if rectilinear_is_source else n_rect

    def _empty_result() -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
        empty = csc_array((n_source, n_target))
        _e = np.empty(0, dtype=np.intp)
        return empty, _e, _e, _no_intersections()

    if n_rect == 0 or n_poly == 0:
        return _empty_result()

    # Single GEOS extraction of every polygon's exterior ring, reused
    # below for the candidate bbox search, the convexity test and the
    # padded clip buffers (see `_PolygonVertexData`).
    poly_data = _PolygonVertexData(poly_polygons)

    # Per-polygon bboxes in the rectilinear grid's local frame. Holes
    # are ignored here, which is conservative for a candidate search --
    # it can only *over*-include candidates, never miss one.
    poly_bbox_xmin, poly_bbox_xmax, poly_bbox_ymin, poly_bbox_ymax = (
        poly_data.local_frame_bboxes(rectilinear_center, rect_angle_rad)
    )

    poly_idx, cell_i, cell_j = _bbox_candidate_ranges_in_rectilinear_grid(
        poly_bbox_xmin,
        poly_bbox_xmax,
        poly_bbox_ymin,
        poly_bbox_ymax,
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

    # -- bbox pre-filter (cheap, avoids paying for the exact clip on
    #    sliver/edge-touch candidates). Done entirely in the grid's
    #    local frame: a cell's extent is then just two edge lookups,
    #    and the polygon bboxes were already computed above -- so this
    #    needs neither a `shapely.bounds` call on the (duplicate-laden)
    #    candidate array nor a gather-and-reduce over world-frame
    #    corners, both of which an earlier revision performed here. --
    cell_j_all, cell_i_all = np.divmod(rect_lin, rectilinear_nx)
    cell_xmin = x_edges[cell_i_all]
    cell_xmax = x_edges[cell_i_all + 1]
    cell_ymin = y_edges[cell_j_all]
    cell_ymax = y_edges[cell_j_all + 1]

    bbox_dx = np.minimum(cell_xmax, poly_bbox_xmax[poly_idx]) - np.maximum(
        cell_xmin, poly_bbox_xmin[poly_idx]
    )
    bbox_dy = np.minimum(cell_ymax, poly_bbox_ymax[poly_idx]) - np.maximum(
        cell_ymin, poly_bbox_ymin[poly_idx]
    )
    nontrivial = (bbox_dx > 1e-15) & (bbox_dy > 1e-15)

    poly_idx = poly_idx[nontrivial]
    rect_lin = rect_lin[nontrivial]
    cell_xmin = cell_xmin[nontrivial]
    cell_xmax = cell_xmax[nontrivial]
    cell_ymin = cell_ymin[nontrivial]
    cell_ymax = cell_ymax[nontrivial]

    if len(poly_idx) == 0:
        return _empty_result()

    # -- split surviving candidates by convexity of their polygon side:
    #    convex polygons bypass Shapely via the numba/numpy N-vertex
    #    Sutherland-Hodgman clipper; non-convex / holed polygons fall
    #    back to the exact Shapely clip. --
    if polygon_grid_is_convex:
        candidate_is_convex = np.ones(len(poly_idx), dtype=bool)
    else:
        candidate_is_convex = poly_data.convexity()[poly_idx]

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
    # IMPORTANT: `_clip_nverts_with_verts` assumes an axis-aligned clip
    # rectangle. The rectilinear grid's cells are only axis-aligned in
    # the grid's own *local* frame (they are rotated in world
    # coordinates whenever the grid angle is nonzero), so both the
    # polygon vertices and the clip bounds are expressed in that local
    # frame here.
    if len(poly_idx_cvx) > 0:
        max_verts = poly_data.max_verts
        clip_width = _clip_buffer_width(max_verts, n_clip_edges=4)

        # Padded buffers are built once **per polygon** and then gathered
        # by candidate. An earlier revision instead ran
        # `get_exterior_ring` + `get_coordinates` on `poly_polygons[
        # poly_idx_cvx]` -- an array holding one entry per *candidate
        # pair*, so a polygon overlapping nine cells was re-extracted
        # from GEOS nine times.
        vx_all, vy_all = poly_data.padded_vertex_buffers(
            clip_width, local_to_grid=(rectilinear_center, rect_angle_rad)
        )

        areas_cvx, out_x_cvx, out_y_cvx, out_n_cvx = _batch_clip_areas_and_verts_nverts(
            np.ascontiguousarray(vx_all[poly_idx_cvx]),
            np.ascontiguousarray(vy_all[poly_idx_cvx]),
            poly_data.true_n[poly_idx_cvx].astype(np.intp),
            cell_xmin[candidate_is_convex],
            cell_ymin[candidate_is_convex],
            cell_xmax[candidate_is_convex],
            cell_ymax[candidate_is_convex],
        )

        valid_cvx = areas_cvx > 1e-15
        result_rect_lin_parts.append(rect_lin_cvx[valid_cvx])
        result_poly_idx_parts.append(poly_idx_cvx[valid_cvx])
        result_areas_parts.append(areas_cvx[valid_cvx])
        if with_intersections:
            # Geometry is built from the same clipped vertices the area
            # computation already produced (vectorized `shapely.polygons`
            # construction -- no GEOS clipping call). Those vertices are
            # in the grid's local frame, so they are rotated back to
            # world coordinates to match the Shapely-exact branch below.
            result_intersections_parts.append(
                _polygons_from_padded_verts(
                    out_x_cvx[valid_cvx],
                    out_y_cvx[valid_cvx],
                    out_n_cvx[valid_cvx],
                    local_to_world=(
                        rectilinear_center,
                        np.cos(rect_angle_rad),
                        np.sin(rect_angle_rad),
                    ),
                )
            )

    # -- general path: non-convex / holed polygons, exact Shapely clip --
    if len(poly_idx_gen) > 0:
        # World-frame cell corners are built here only, for the cells
        # that actually reach the Shapely branch.
        gen_corners_x, gen_corners_y = _rectilinear_cell_corners_world(
            rectilinear_center, rect_angle_rad, x_edges, y_edges, rect_lin_gen
        )
        rect_polys_gen = shapely.polygons(
            np.stack([gen_corners_x, gen_corners_y], axis=-1)
        )
        intersections_gen = shapely.intersection(
            rect_polys_gen, poly_polygons[poly_idx_gen]
        )
        areas_gen = shapely.area(intersections_gen)
        valid_gen = areas_gen > 1e-15
        result_rect_lin_parts.append(rect_lin_gen[valid_gen])
        result_poly_idx_parts.append(poly_idx_gen[valid_gen])
        result_areas_parts.append(areas_gen[valid_gen])
        if with_intersections:
            result_intersections_parts.append(intersections_gen[valid_gen])

    rect_lin = np.concatenate(result_rect_lin_parts)
    poly_idx = np.concatenate(result_poly_idx_parts)
    intersection_areas = np.concatenate(result_areas_parts)

    # Redundant safety net: both branches above already filter on
    # area > 1e-15 individually, but re-applying here guards against
    # numerical edge cases at the boundary between the two paths.
    valid = intersection_areas > 1e-15
    rect_lin = rect_lin[valid]
    poly_idx = poly_idx[valid]
    intersection_areas = intersection_areas[valid]

    if with_intersections:
        intersections = np.concatenate(result_intersections_parts)[valid]
    else:
        intersections = _no_intersections()

    if rectilinear_is_source:
        source_indices = rect_lin
        target_indices = poly_idx
        # every rectilinear cell has the same area, so divide by the
        # scalar rather than materializing and gathering an n_rect array
        weights = intersection_areas / rect_cell_area
    else:
        source_indices = poly_idx
        target_indices = rect_lin
        weights = intersection_areas / shapely.area(poly_polygons)[source_indices]

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
        _pack_intersections(intersections) if with_intersections else intersections,
    )


# ===================================================================
# TriMesh helpers
# ===================================================================


def _tri_world_coords(mesh: TriMesh) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Per-triangle vertex world coordinates, each of shape ``(n_tri, 3)``.

    This reads directly off ``mesh.verts_xy`` / ``mesh.tri_verts`` --
    it never materializes Shapely ``Polygon`` objects, which is the
    whole point of the dedicated TriMesh fast paths in this module
    (building ``n_tri`` Shapely polygons and re-extracting their
    coordinates later would be strictly more expensive than a single
    fancy-index gather here).
    """
    tv = mesh.tri_verts
    vx = mesh.verts_xy[:, 0]
    vy = mesh.verts_xy[:, 1]
    return vx[tv], vy[tv]  # each (n_tri, 3)


def _ensure_ccw_triangles(
    tx: NDArrayFloat, ty: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """
    Return a copy of ``(tx, ty)`` (each shape ``(n_tri, 3)``) with every
    triangle's vertices re-ordered to counter-clockwise winding.

    The triangle-triangle and convex-vs-convex clip kernels below
    assume CCW input for both the subject and the clip polygon (the
    inside/outside half-plane test is ``cross(edge, point - edge0) >=
    0``, which flips sign under CW winding). A ``TriMesh`` does not
    itself guarantee a winding convention, so this normalization is
    applied once per mesh (vectorized, O(n_tri)) rather than re-checked
    per candidate pair inside the numba kernels.

    Note the rectilinear/TriMesh dedicated path below does NOT need
    CCW-normalized triangles: `_clip_tri_rect_with_verts` (like every
    kernel that clips against an axis-aligned rectangle) only tests
    each vertex against fixed axis thresholds and applies
    ``abs(...)`` to the final shoelace sum, so it is winding-agnostic
    and works correctly for either CW or CCW input. This normalization
    is therefore only invoked on the TriMesh-vs-TriMesh and
    TriMesh-vs-arbitrary-polygon paths, where the clip *window* itself
    is a triangle and winding direction matters for the half-plane
    inside/outside test.
    """
    signed_area2 = (tx[:, 1] - tx[:, 0]) * (ty[:, 2] - ty[:, 0]) - (
        tx[:, 2] - tx[:, 0]
    ) * (ty[:, 1] - ty[:, 0])
    is_cw = signed_area2 < 0.0
    if not is_cw.any():
        return tx, ty
    tx = tx.copy()
    ty = ty.copy()
    tx[is_cw, 1], tx[is_cw, 2] = tx[is_cw, 2], tx[is_cw, 1].copy()
    ty[is_cw, 1], ty[is_cw, 2] = ty[is_cw, 2], ty[is_cw, 1].copy()
    return tx, ty


# ===================================================================
# Dedicated RectilinearGrid vs TriMesh path (Shapely-free)
#
# This is the key optimization over routing TriMesh through the
# generic rectilinear/polygon mixed path: it never calls
# `TriMesh.to_shapely()`, never builds `n_tri` Shapely `Polygon`
# objects, and never re-extracts their coordinates via
# `get_exterior_ring` / `get_coordinates`. Every step operates directly
# on the `(n_tri, 3)` vertex arrays already available from
# `mesh.verts_xy[mesh.tri_verts]`:
#
#   1. Candidate-pair search: per-triangle bounding boxes come from a
#      vectorised `.min(axis=1)` / `.max(axis=1)` over the local-frame
#      vertex arrays (no `shapely.bounds`), then reuse the same O(1)
#      index-range arithmetic (`_bbox_candidate_ranges_in_rectilinear_grid`)
#      as the generic mixed path.
#   2. Per-candidate padded vertex buffers: since every triangle has
#      exactly 3 vertices, there is no ragged bookkeeping at all (no
#      `get_coordinates`/`searchsorted`/`bincount` dance) -- candidates
#      are just a `(n_cand, 3)` fancy-index gather from the mesh's own
#      local-frame vertex arrays.
#   3. Clip: a specialized 3-vertex-input Sutherland-Hodgman kernel
#      (`_batch_clip_areas_and_verts_tri_rect`) avoids even the generic
#      N-vertex padding used by `_batch_clip_areas_and_verts_nverts`.
#   4. No convexity test: a triangle is always convex, so there is no
#      Shapely fallback branch on this path at all.
# ===================================================================


def _compute_transfer_matrix_rect_trimesh(
    rectilinear_grid: RectilinearGrid,
    trimesh: TriMesh,
    rectilinear_is_source: bool,
    rectilinear_grid_mask: Optional[NDArrayBool] = None,
    trimesh_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
    with_intersections: bool = True,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
    """
    Conservative transfer matrix between a rectilinear grid and a
    triangular mesh, without ever constructing a Shapely geometry for
    the mesh side.

    Parameters
    ----------
    rectilinear_grid : RectilinearGrid
        Grid object exposing ``cx, cy, dx, dy, nx, ny, theta``.
    trimesh : TriMesh
        Triangular mesh, the other side of the transfer.
    rectilinear_is_source : bool
        If True, the rectilinear grid is the source and ``trimesh`` is
        the target. If False, the roles are reversed.
    rectilinear_grid_mask : 1-D array of bool, optional
        Boolean mask over the rectilinear grid's cells (Fortran-order
        linear index ``j * nx + i``), length ``nx * ny``. If ``None``,
        all cells are considered.
    trimesh_mask : 1-D array of bool, optional
        Boolean mask over ``trimesh`` triangles (length
        ``trimesh.n_tri``). If ``None``, all triangles are considered.
    is_sanity_check : bool, optional
        If True, verify conservation for fully-covered, unmasked
        source cells.

    Returns
    -------
    W : csc_array, shape (n_source, n_target)
        Sparse conservative transfer matrix.
    source_indices, target_indices : NDArrayInt, shape (nnz,)
        Row / column index of each nonzero entry.
    intersections : IntersectionsArray, shape (nnz,)
        One intersection ``Polygon`` per nonzero matrix entry, built
        via :func:`_polygons_from_padded_verts` from the same clipped
        vertices the area computation already produced -- no GEOS
        clipping call, only vectorized ``shapely.polygons``
        construction. This is still a fully Shapely-clip-free path for
        the mesh side; only the final geometry *construction* touches
        ``shapely`` (via its vectorized ``polygons`` constructor).
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
    n_tri = trimesh.n_tri

    rectilinear_grid_mask = _validate_mask(
        rectilinear_grid_mask, n_rect, "rectilinear grid mask"
    )
    trimesh_mask = _validate_mask(trimesh_mask, n_tri, "trimesh mask")

    n_source = n_rect if rectilinear_is_source else n_tri
    n_target = n_tri if rectilinear_is_source else n_rect

    def _empty_result() -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
        empty = csc_array((n_source, n_target))
        _e = np.empty(0, dtype=np.intp)
        return empty, _e, _e, _no_intersections()

    if n_rect == 0 or n_tri == 0:
        return _empty_result()

    rect_angle_rad = np.deg2rad(rectilinear_angle_deg)
    rc_ca, rc_sa = np.cos(rect_angle_rad), np.sin(rect_angle_rad)

    x_edges = (
        np.arange(rectilinear_nx + 1, dtype=np.float64) - rectilinear_nx / 2
    ) * rectilinear_dx
    y_edges = (
        np.arange(rectilinear_ny + 1, dtype=np.float64) - rectilinear_ny / 2
    ) * rectilinear_dy
    rect_cell_area = rectilinear_dx * rectilinear_dy

    # -- triangle vertices, world frame -> rectilinear-grid-local frame,
    #    entirely in numpy (no Shapely object ever built) --
    tri_wx, tri_wy = _tri_world_coords(trimesh)  # each (n_tri, 3)
    dxw = tri_wx - rectilinear_center[0]
    dyw = tri_wy - rectilinear_center[1]
    tri_lx = dxw * rc_ca + dyw * rc_sa
    tri_ly = -dxw * rc_sa + dyw * rc_ca

    # -- per-triangle bbox via plain numpy reductions (no shapely.bounds) --
    tri_bbox_xmin = tri_lx.min(axis=1)
    tri_bbox_xmax = tri_lx.max(axis=1)
    tri_bbox_ymin = tri_ly.min(axis=1)
    tri_bbox_ymax = tri_ly.max(axis=1)

    # -- candidate cell ranges via shared O(1) arithmetic --
    tri_idx, cell_i, cell_j = _bbox_candidate_ranges_in_rectilinear_grid(
        tri_bbox_xmin,
        tri_bbox_xmax,
        tri_bbox_ymin,
        tri_bbox_ymax,
        x_edges,
        y_edges,
        rectilinear_nx,
        rectilinear_ny,
    )

    if len(tri_idx) == 0:
        return _empty_result()

    rect_lin = cell_j * rectilinear_nx + cell_i

    # -- apply masks early to avoid wasted clip work on excluded cells --
    if rectilinear_grid_mask is not None or trimesh_mask is not None:
        keep_masked = np.ones(len(tri_idx), dtype=bool)
        if rectilinear_grid_mask is not None:
            keep_masked &= rectilinear_grid_mask[rect_lin]
        if trimesh_mask is not None:
            keep_masked &= trimesh_mask[tri_idx]
        tri_idx = tri_idx[keep_masked]
        rect_lin = rect_lin[keep_masked]
        if len(tri_idx) == 0:
            return _empty_result()

    # -- bbox pre-filter (cheap, avoids paying for the exact clip on
    #    sliver/edge-touch candidates); cell bbox is just its own
    #    local-frame edges, no need to go through world-frame corners --
    cell_j_all, cell_i_all = np.divmod(rect_lin, rectilinear_nx)
    cell_xmin = x_edges[cell_i_all]
    cell_xmax = x_edges[cell_i_all + 1]
    cell_ymin = y_edges[cell_j_all]
    cell_ymax = y_edges[cell_j_all + 1]

    cand_bbox_xmin = tri_bbox_xmin[tri_idx]
    cand_bbox_xmax = tri_bbox_xmax[tri_idx]
    cand_bbox_ymin = tri_bbox_ymin[tri_idx]
    cand_bbox_ymax = tri_bbox_ymax[tri_idx]

    bbox_dx = np.minimum(cell_xmax, cand_bbox_xmax) - np.maximum(
        cell_xmin, cand_bbox_xmin
    )
    bbox_dy = np.minimum(cell_ymax, cand_bbox_ymax) - np.maximum(
        cell_ymin, cand_bbox_ymin
    )
    nontrivial = (bbox_dx > 1e-15) & (bbox_dy > 1e-15)

    tri_idx = tri_idx[nontrivial]
    rect_lin = rect_lin[nontrivial]

    if len(tri_idx) == 0:
        return _empty_result()

    # -- exact clip: every candidate is convex (triangle), so there is
    #    no Shapely fallback branch on this path at all. Winding does
    #    not matter here (`_clip_tri_rect_with_verts` clips against
    #    fixed axis thresholds and takes abs() of the shoelace sum), so
    #    no CCW normalization is needed either. The clip bounds are the
    #    surviving subset of the cell extents already computed for the
    #    pre-filter, so there is no second `divmod` + edge gather. --
    areas, out_x, out_y, out_n = _batch_clip_areas_and_verts_tri_rect(
        tri_lx[tri_idx],
        tri_ly[tri_idx],
        cell_xmin[nontrivial],
        cell_ymin[nontrivial],
        cell_xmax[nontrivial],
        cell_ymax[nontrivial],
    )

    valid = areas > 1e-15
    tri_idx = tri_idx[valid]
    rect_lin = rect_lin[valid]
    areas = areas[valid]
    out_x = out_x[valid]
    out_y = out_y[valid]
    out_n = out_n[valid]

    if len(tri_idx) == 0:
        return _empty_result()

    if rectilinear_is_source:
        source_indices = rect_lin
        target_indices = tri_idx
        # uniform cell area -- divide by the scalar rather than
        # materializing and gathering an n_rect array of copies
        weights = areas / rect_cell_area
    else:
        source_indices = tri_idx
        target_indices = rect_lin
        weights = areas / trimesh.tri_area_m2[source_indices]

    mat = coo_array(
        (weights, (source_indices, target_indices)),
        shape=(n_source, n_target),
    ).tocsc()

    # -- build intersection geometry from the same clip vertices
    #    (out_x/out_y are in the rectilinear grid's local frame, i.e.
    #    the same frame `tri_lx`/`tri_ly` and the clip bounds were
    #    computed in) --
    if with_intersections:
        intersections = _pack_intersections(
            _polygons_from_padded_verts(
                out_x,
                out_y,
                out_n,
                local_to_world=(rectilinear_center, rc_ca, rc_sa),
            )
        )
    else:
        intersections = _no_intersections()

    if is_sanity_check:
        _check_conservation(mat)

    return mat, source_indices, target_indices, intersections


# ===================================================================
# TriMesh vs TriMesh: dedicated triangle-triangle clip.
#
# Every triangle is convex with exactly 3 vertices, so unlike the
# generic arbitrary-polygon path, the exact overlap area never needs
# GEOS: candidate pairs are still found via STRtree (a TriMesh has no
# analytic structure to exploit for O(1) index arithmetic, unlike a
# RectilinearGrid), but the area of each candidate pair is computed
# with a fixed-size (3-vertex-in, <=6-vertex-out) numba/numpy
# Sutherland-Hodgman clip instead of `shapely.intersection`.
# ===================================================================


@njit(cache=True)
def _clip_triangle_pair_with_verts(
    sx: NDArrayFloat,
    sy: NDArrayFloat,
    cx: NDArrayFloat,
    cy: NDArrayFloat,
    out_x: NDArrayFloat,
    out_y: NDArrayFloat,
) -> Tuple[float, int]:
    """
    Sutherland-Hodgman clip of a CCW subject triangle ``(sx, sy)``
    against a CCW clip triangle ``(cx, cy)``, also writing the clipped
    polygon's vertices into ``out_x``/``out_y`` (each length >= 9) and
    returning ``(area, n_verts)``.

    The clip polygon (a triangle) is convex, which is all
    Sutherland-Hodgman requires of the *clip* side; the subject
    triangle is convex too, so the intersection is guaranteed to be a
    single connected convex polygon (at most 6 vertices, since each of
    the 3 clip edges can add at most one vertex to a convex subject).
    """
    ax = np.empty(9)
    ay = np.empty(9)
    bx = np.empty(9)
    by = np.empty(9)
    n_in = 3
    for p in range(3):
        ax[p] = sx[p]
        ay[p] = sy[p]

    for e in range(3):
        e1 = (e + 1) % 3
        ex0, ey0 = cx[e], cy[e]
        edx, edy = cx[e1] - ex0, cy[e1] - ey0
        n_out = 0
        if n_in < 3:
            return 0.0, 0
        for i in range(n_in):
            pi = n_in - 1 if i == 0 else i - 1
            c_in = edx * (ay[i] - ey0) - edy * (ax[i] - ex0) >= 0.0
            p_in = edx * (ay[pi] - ey0) - edy * (ax[pi] - ex0) >= 0.0
            if p_in != c_in:
                dxp, dyp = ax[i] - ax[pi], ay[i] - ay[pi]
                denom = edx * dyp - edy * dxp
                t = (
                    0.0
                    if denom == 0.0
                    else (edx * (ay[pi] - ey0) - edy * (ax[pi] - ex0)) / -denom
                )
                bx[n_out] = ax[pi] + t * dxp
                by[n_out] = ay[pi] + t * dyp
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
        return 0.0, 0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
        out_x[i] = ax[i]
        out_y[i] = ay[i]
    return abs(area) * 0.5, n_in


@njit(parallel=True, cache=True)
def _batch_clip_triangles_and_verts(
    sx: NDArrayFloat, sy: NDArrayFloat, cx: NDArrayFloat, cy: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    n = sx.shape[0]
    areas = np.empty(n)
    out_x: NDArrayFloat = np.zeros((n, 9))
    out_y = np.zeros((n, 9))
    out_n = np.zeros(n, dtype=np.intp)
    for k in prange(n):  # ty:ignore[not-iterable]
        area, n_verts = _clip_triangle_pair_with_verts(
            sx[k], sy[k], cx[k], cy[k], out_x[k], out_y[k]
        )
        areas[k] = area
        out_n[k] = n_verts
    return areas, out_x, out_y, out_n


def _sh_clip_python_generic_with_verts(subject: list, clip: list) -> Tuple[float, list]:
    """
    Pure-Python Sutherland-Hodgman clip + shoelace, for an arbitrary
    convex ``clip`` polygon (not restricted to an axis-aligned
    rectangle). Both ``subject`` and ``clip`` are lists of ``(x, y)``
    tuples, CCW. Returns ``(area, clipped_verts)``.
    """
    output = subject
    n_clip = len(clip)
    for e in range(n_clip):
        if len(output) < 3:
            return 0.0, []
        ex0, ey0 = clip[e]
        ex1, ey1 = clip[(e + 1) % n_clip]
        edx, edy = ex1 - ex0, ey1 - ey0
        inp = output
        output = []
        for i in range(len(inp)):
            cur = inp[i]
            prev = inp[i - 1]
            c_in = edx * (cur[1] - ey0) - edy * (cur[0] - ex0) >= 0.0
            p_in = edx * (prev[1] - ey0) - edy * (prev[0] - ex0) >= 0.0
            if p_in != c_in:
                dxp, dyp = cur[0] - prev[0], cur[1] - prev[1]
                denom = edx * dyp - edy * dxp
                t = (
                    0.0
                    if denom == 0.0
                    else (edx * (prev[1] - ey0) - edy * (prev[0] - ex0)) / -denom
                )
                output.append((prev[0] + t * dxp, prev[1] + t * dyp))
            if c_in:
                output.append(cur)
    if len(output) < 3:
        return 0.0, []
    area = 0.0
    n = len(output)
    for i in range(n):
        j = (i + 1) % n
        area += output[i][0] * output[j][1] - output[j][0] * output[i][1]
    return abs(area) * 0.5, output


def _batch_clip_triangles_numpy_and_verts(
    sx: NDArrayFloat, sy: NDArrayFloat, cx: NDArrayFloat, cy: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Pure-Python fallback for :func:`_batch_clip_triangles_and_verts`."""
    n = sx.shape[0]
    areas = np.empty(n)
    out_x = np.zeros((n, 9))
    out_y = np.zeros((n, 9))
    out_n = np.zeros(n, dtype=np.intp)
    for k in range(n):
        subj = list(zip(sx[k].tolist(), sy[k].tolist()))
        clip = list(zip(cx[k].tolist(), cy[k].tolist()))
        area, out_poly = _sh_clip_python_generic_with_verts(subj, clip)
        areas[k] = area
        m = len(out_poly)
        out_n[k] = m
        for p in range(m):
            out_x[k, p] = out_poly[p][0]
            out_y[k, p] = out_poly[p][1]
    return areas, out_x, out_y, out_n


def _batch_clip_triangles_and_verts_dispatch(
    sx: NDArrayFloat, sy: NDArrayFloat, cx: NDArrayFloat, cy: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Dispatch to the fastest available triangle-triangle clip
    back-end, also returning clipped vertices."""
    if _HAS_NUMBA:
        return _batch_clip_triangles_and_verts(
            np.ascontiguousarray(sx),  # ty: ignore[invalid-argument-type]
            np.ascontiguousarray(sy),  # ty: ignore[invalid-argument-type]
            np.ascontiguousarray(cx),  # ty: ignore[invalid-argument-type]
            np.ascontiguousarray(cy),  # ty: ignore[invalid-argument-type]
        )
    return _batch_clip_triangles_numpy_and_verts(sx, sy, cx, cy)


def _compute_transfer_matrix_trimesh(
    source_mesh: TriMesh,
    target_mesh: TriMesh,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
    with_intersections: bool = True,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
    """
    Build a conservative transfer matrix between two triangular
    meshes.

    Candidate triangle pairs are found via an STRtree built on
    ``source_mesh`` (no analytic index is available for an irregular
    mesh, unlike a :class:`RectilinearGrid`), but the overlap area of
    each candidate pair is computed with a dedicated numba/numpy
    triangle-triangle Sutherland-Hodgman clip rather than
    ``shapely.intersection`` -- avoiding GEOS entirely in the hot loop.

    Parameters
    ----------
    source_mesh, target_mesh : TriMesh
        Source and target triangular meshes.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over source triangles (length
        ``source_mesh.n_tri``). Masked-out triangles are excluded from
        the STRtree and contribute no nonzero row. If ``None``, all
        source triangles are considered.
    target_grid_mask : 1-D array of bool, optional
        Boolean mask over target triangles (length
        ``target_mesh.n_tri``), with the same convention as
        ``source_grid_mask`` but excluding columns instead of rows. If
        ``None``, all target triangles are considered.
    is_sanity_check : bool, optional
        If True, verify that every fully-covered, unmasked source
        triangle conserves its quantity exactly (up to 1e-10).

    Returns
    -------
    W : csc_array, shape (source_mesh.n_tri, target_mesh.n_tri)
        Sparse conservative transfer matrix.
    source_indices, target_indices : NDArrayInt, shape (nnz,)
        Source / target triangle id of each nonzero entry.
    intersections : IntersectionsArray, shape (nnz,)
        One intersection ``Polygon`` (in world coordinates -- the
        triangle-triangle clip already works directly in world frame,
        unlike the rectilinear paths) per nonzero matrix entry, built
        via :func:`_polygons_from_padded_verts` from the same clipped
        vertices the area computation already produced -- no GEOS
        clipping call, only vectorized ``shapely.polygons``
        construction.
    """
    n_source = source_mesh.n_tri
    n_target = target_mesh.n_tri

    source_grid_mask = _validate_mask(source_grid_mask, n_source, "source_grid_mask")
    target_grid_mask = _validate_mask(target_grid_mask, n_target, "target_grid_mask")

    empty = (
        csc_array((n_source, n_target)),
        np.empty(0, dtype=np.intp),
        np.empty(0, dtype=np.intp),
        _no_intersections(),
    )
    if n_source == 0 or n_target == 0:
        return empty

    # -- triangle vertices straight from numpy; used for the bbox
    #    pre-filter below (no `shapely.bounds` call) and for the clip --
    src_tx, src_ty = _tri_world_coords(source_mesh)
    tgt_tx, tgt_ty = _tri_world_coords(target_mesh)

    # -- STRtree candidate search (Shapely used only for the spatial
    #    index, never for the area computation) --
    src_polys = np.asarray(source_mesh.to_shapely().geoms, dtype=object)
    tgt_polys = np.asarray(target_mesh.to_shapely().geoms, dtype=object)

    keep_src = (
        np.flatnonzero(source_grid_mask)
        if source_grid_mask is not None
        else np.arange(n_source)
    )
    keep_tgt = (
        np.flatnonzero(target_grid_mask)
        if target_grid_mask is not None
        else np.arange(n_target)
    )
    if len(keep_src) == 0 or len(keep_tgt) == 0:
        return empty

    # No `predicate="intersects"`: that makes GEOS run an exact
    # intersection test on every bbox candidate, but the exact
    # Sutherland-Hodgman clip below is already the source of truth and
    # returns zero area for non-overlapping pairs (with a cheap bbox
    # pre-filter in front of it). A plain bbox query is markedly
    # cheaper and also removes the need to `shapely.prepare` the tree
    # geometries; the few extra candidates it admits are rejected
    # downstream at negligible cost.
    tree = STRtree(src_polys[keep_src])
    pairs = tree.query(tgt_polys[keep_tgt])
    tgt_local, src_local = pairs[0], pairs[1]

    if len(src_local) == 0:
        return empty

    source_indices = keep_src[src_local]
    target_indices = keep_tgt[tgt_local]

    # -- cheap bbox pre-filter before the exact clip. Per-triangle
    #    bounds come from vectorized min/max over the fixed (n_tri, 3)
    #    vertex arrays, computed once per mesh and then gathered --
    #    rather than `shapely.bounds` on a candidate array that repeats
    #    each triangle once per pair. --
    src_bx0, src_bx1 = src_tx.min(axis=1), src_tx.max(axis=1)
    src_by0, src_by1 = src_ty.min(axis=1), src_ty.max(axis=1)
    tgt_bx0, tgt_bx1 = tgt_tx.min(axis=1), tgt_tx.max(axis=1)
    tgt_by0, tgt_by1 = tgt_ty.min(axis=1), tgt_ty.max(axis=1)

    bbox_dx = np.minimum(src_bx1[source_indices], tgt_bx1[target_indices]) - np.maximum(
        src_bx0[source_indices], tgt_bx0[target_indices]
    )
    bbox_dy = np.minimum(src_by1[source_indices], tgt_by1[target_indices]) - np.maximum(
        src_by0[source_indices], tgt_by0[target_indices]
    )
    nontrivial = (bbox_dx > 1e-15) & (bbox_dy > 1e-15)
    source_indices = source_indices[nontrivial]
    target_indices = target_indices[nontrivial]

    if len(source_indices) == 0:
        return empty

    # -- exact overlap area AND clipped vertices: triangle-triangle
    #    numba/numpy clip --
    src_tx, src_ty = _ensure_ccw_triangles(src_tx, src_ty)
    tgt_tx, tgt_ty = _ensure_ccw_triangles(tgt_tx, tgt_ty)

    areas, out_x, out_y, out_n = _batch_clip_triangles_and_verts_dispatch(
        src_tx[source_indices],
        src_ty[source_indices],
        tgt_tx[target_indices],
        tgt_ty[target_indices],
    )

    valid = areas > 1e-15
    source_indices = source_indices[valid]
    target_indices = target_indices[valid]
    areas = areas[valid]
    out_x = out_x[valid]
    out_y = out_y[valid]
    out_n = out_n[valid]

    weights = areas / source_mesh.tri_area_m2[source_indices]

    mat = coo_array(
        (weights, (source_indices, target_indices)),
        shape=(n_source, n_target),
    ).tocsc()

    # -- build intersection geometry from the same clip vertices. These
    #    are already in world coordinates (the triangle-triangle clip
    #    never transforms to a local frame), so no local_to_world
    #    rotation/translation is applied here. --
    if with_intersections:
        intersections = _pack_intersections(
            _polygons_from_padded_verts(out_x, out_y, out_n)
        )
    else:
        intersections = _no_intersections()

    if is_sanity_check:
        _check_conservation(mat)

    return mat, source_indices, target_indices, intersections


# ===================================================================
# TriMesh vs arbitrary MultiPolygon.
#
# Neither side has a regular structure to exploit for an analytic
# index, so candidate pairs are found via STRtree, exactly like the
# fully-generic path. The difference is in how the area of each
# candidate pair is computed: the TriMesh side is always convex (3
# vertices), so it is used as the *clip* polygon in a generic
# convex-vs-convex Sutherland-Hodgman clip; the arbitrary-polygon side
# is the *subject* and only needs to be checked for convexity (not the
# triangle side, which is skipped -- same idea as
# ``polygon_grid_is_convex`` in the rectilinear/polygon mixed path).
# Convex subjects bypass Shapely entirely; non-convex/holed subjects
# fall back to `shapely.intersection`.
#
# Unlike the rectilinear/TriMesh case, this path still needs
# `TriMesh.to_shapely()` for the STRtree, since there is no analytic
# index available on either side here.
# ===================================================================


def _compute_transfer_matrix_trimesh_polygon(
    trimesh: TriMesh,
    polygon_grid: shapely.MultiPolygon,
    trimesh_is_source: bool,
    trimesh_mask: Optional[NDArrayBool] = None,
    polygon_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
    with_intersections: bool = True,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
    """
    Conservative transfer matrix between a :class:`TriMesh` and an
    arbitrary polygon grid.

    Parameters
    ----------
    trimesh : TriMesh
        Triangular mesh, either the source or the target (see
        ``trimesh_is_source``).
    polygon_grid : shapely.MultiPolygon
        The other side of the transfer.
    trimesh_is_source : bool
        If True, ``trimesh`` is the source and ``polygon_grid`` is the
        target. If False, the roles are reversed.
    trimesh_mask : 1-D array of bool, optional
        Boolean mask over ``trimesh`` triangles (length
        ``trimesh.n_tri``). Masked-out triangles contribute no nonzero
        entry. If ``None``, all triangles are considered.
    polygon_mask : 1-D array of bool, optional
        Boolean mask over ``polygon_grid`` polygons (length
        ``len(polygon_grid.geoms)``). If ``None``, all polygons are
        considered.
    is_sanity_check : bool, optional
        If True, verify conservation for fully-covered, unmasked
        source cells.

    Returns
    -------
    W : csc_array, shape (n_source, n_target)
        Sparse conservative transfer matrix.
    source_indices, target_indices : NDArrayInt, shape (nnz,)
        Row / column index of each nonzero entry.
    intersections : IntersectionsArray, shape (nnz,)
        Intersection geometries for each nonzero entry. Entries
        produced by the fast convex-clip path are ``None``
        placeholders (area-only clipper); entries produced by the
        Shapely fallback carry the exact geometry.
    """
    n_tri = trimesh.n_tri
    n_poly = len(polygon_grid.geoms)

    trimesh_mask = _validate_mask(trimesh_mask, n_tri, "trimesh mask")
    polygon_mask = _validate_mask(polygon_mask, n_poly, "polygon mask")

    n_source = n_tri if trimesh_is_source else n_poly
    n_target = n_poly if trimesh_is_source else n_tri
    empty = (
        csc_array((n_source, n_target)),
        np.empty(0, dtype=np.intp),
        np.empty(0, dtype=np.intp),
        _no_intersections(),
    )
    if n_tri == 0 or n_poly == 0:
        return empty

    tri_polys = np.asarray(trimesh.to_shapely().geoms, dtype=object)
    poly_polys = np.asarray(polygon_grid.geoms, dtype=object)

    # Single GEOS extraction of the polygon side, shared by the
    # convexity test and the padded clip buffers below.
    poly_data = _PolygonVertexData(poly_polys)
    tri_tx, tri_ty = _tri_world_coords(trimesh)

    keep_tri = (
        np.flatnonzero(trimesh_mask) if trimesh_mask is not None else np.arange(n_tri)
    )
    keep_poly = (
        np.flatnonzero(polygon_mask) if polygon_mask is not None else np.arange(n_poly)
    )
    if len(keep_tri) == 0 or len(keep_poly) == 0:
        return empty

    # -- STRtree candidate search: tree built over the (always convex,
    #    cheap to prepare) triangle side, queried with the polygon
    #    side. --
    # Bbox-only query (no exact `intersects` predicate): the bbox
    # pre-filter plus the exact clip below already reject non-overlaps,
    # so paying GEOS for an exact test per candidate is redundant.
    tree = STRtree(tri_polys[keep_tri])
    pairs = tree.query(poly_polys[keep_poly])
    poly_local, tri_local = pairs[0], pairs[1]

    if len(tri_local) == 0:
        return empty

    tri_indices = keep_tri[tri_local]
    poly_indices = keep_poly[poly_local]

    # -- bbox pre-filter, from per-object bounds computed once and
    #    then gathered (the candidate index arrays repeat each triangle
    #    and each polygon once per pair, so calling `shapely.bounds`
    #    on them recomputed the same bounds many times over) --
    tri_bx0, tri_bx1 = tri_tx.min(axis=1), tri_tx.max(axis=1)
    tri_by0, tri_by1 = tri_ty.min(axis=1), tri_ty.max(axis=1)
    poly_bounds_all = shapely.bounds(poly_polys)

    bbox_dx = np.minimum(
        tri_bx1[tri_indices], poly_bounds_all[poly_indices, 2]
    ) - np.maximum(tri_bx0[tri_indices], poly_bounds_all[poly_indices, 0])
    bbox_dy = np.minimum(
        tri_by1[tri_indices], poly_bounds_all[poly_indices, 3]
    ) - np.maximum(tri_by0[tri_indices], poly_bounds_all[poly_indices, 1])
    nontrivial = (bbox_dx > 1e-15) & (bbox_dy > 1e-15)
    tri_indices = tri_indices[nontrivial]
    poly_indices = poly_indices[nontrivial]

    if len(tri_indices) == 0:
        return empty

    # -- split by convexity of the polygon side only; the triangle
    #    side is always convex, so it is never tested. --
    candidate_is_convex = poly_data.convexity()[poly_indices]

    tri_idx_cvx = tri_indices[candidate_is_convex]
    poly_idx_cvx = poly_indices[candidate_is_convex]
    tri_idx_gen = tri_indices[~candidate_is_convex]
    poly_idx_gen = poly_indices[~candidate_is_convex]

    result_tri_parts = []
    result_poly_parts = []
    result_area_parts = []
    result_ix_parts = []

    # -- fast path: convex polygon subject, triangle clip window --
    if len(poly_idx_cvx) > 0:
        ccw_tx, ccw_ty = _ensure_ccw_triangles(tri_tx, tri_ty)

        # Padded buffers built once per polygon (world frame -- the
        # triangle clip window is arbitrary, so no local-frame
        # transform is involved on this path) and gathered per
        # candidate, instead of re-extracting coordinates from GEOS
        # once per candidate pair.
        clip_width = _clip_buffer_width(poly_data.max_verts, n_clip_edges=3)
        vx_all, vy_all = poly_data.padded_vertex_buffers(clip_width)

        areas_cvx, out_x_cvx, out_y_cvx, out_n_cvx = (
            _batch_clip_areas_and_verts_convexclip(
                np.ascontiguousarray(vx_all[poly_idx_cvx]),
                np.ascontiguousarray(vy_all[poly_idx_cvx]),
                poly_data.true_n[poly_idx_cvx].astype(np.intp),
                np.ascontiguousarray(ccw_tx[tri_idx_cvx]),
                np.ascontiguousarray(ccw_ty[tri_idx_cvx]),
            )
        )

        valid_cvx = areas_cvx > 1e-15
        result_tri_parts.append(tri_idx_cvx[valid_cvx])
        result_poly_parts.append(poly_idx_cvx[valid_cvx])
        result_area_parts.append(areas_cvx[valid_cvx])
        if with_intersections:
            # Geometry from the same clipped vertices used for the area
            # (already world-frame, so no rotation needed).
            result_ix_parts.append(
                _polygons_from_padded_verts(
                    out_x_cvx[valid_cvx], out_y_cvx[valid_cvx], out_n_cvx[valid_cvx]
                )
            )

    # -- general path: non-convex / holed polygon subject, Shapely clip --
    if len(poly_idx_gen) > 0:
        intersections_gen = shapely.intersection(
            tri_polys[tri_idx_gen], poly_polys[poly_idx_gen]
        )
        areas_gen = shapely.area(intersections_gen)
        valid_gen = areas_gen > 1e-15
        result_tri_parts.append(tri_idx_gen[valid_gen])
        result_poly_parts.append(poly_idx_gen[valid_gen])
        result_area_parts.append(areas_gen[valid_gen])
        if with_intersections:
            result_ix_parts.append(intersections_gen[valid_gen])

    tri_indices = np.concatenate(result_tri_parts)
    poly_indices = np.concatenate(result_poly_parts)
    intersection_areas = np.concatenate(result_area_parts)

    valid = intersection_areas > 1e-15
    tri_indices = tri_indices[valid]
    poly_indices = poly_indices[valid]
    intersection_areas = intersection_areas[valid]

    if with_intersections:
        intersections = np.concatenate(result_ix_parts)[valid]
    else:
        intersections = _no_intersections()

    if trimesh_is_source:
        source_indices = tri_indices
        target_indices = poly_indices
        source_areas = trimesh.tri_area_m2
    else:
        source_indices = poly_indices
        target_indices = tri_indices
        source_areas = shapely.area(poly_polys)

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
        _pack_intersections(intersections) if with_intersections else intersections,
    )


# -------------------------------------------------------------------
# Generic convex-subject-vs-convex-clip-window clip (used by the
# TriMesh-vs-polygon path above): the clip window is an arbitrary
# triangle rather than an axis-aligned rectangle, so this generalizes
# `_clip_nverts_with_verts` (rect clip window) the same way
# `_clip_triangle_pair_with_verts` generalizes it to a triangle
# *subject*.
# -------------------------------------------------------------------


@njit(cache=True)
def _clip_convexclip_with_verts(
    subj_x: NDArrayFloat,
    subj_y: NDArrayFloat,
    n_subj: int,
    clip_x: NDArrayFloat,
    clip_y: NDArrayFloat,
    out_x: NDArrayFloat,
    out_y: NDArrayFloat,
) -> Tuple[float, int]:
    """
    Sutherland-Hodgman clip of a convex ``n_subj``-vertex subject
    polygon (CCW) against a CCW clip triangle, also writing the
    clipped polygon's vertices into ``out_x``/``out_y`` (each length
    >= ``_MAX_CLIP_VERTS``) and returning ``(area, n_out_verts)``.
    """
    MAX_V = _MAX_CLIP_VERTS
    ax = np.empty(MAX_V)
    ay = np.empty(MAX_V)
    bx = np.empty(MAX_V)
    by = np.empty(MAX_V)
    n_in = n_subj
    for p in range(n_subj):
        ax[p] = subj_x[p]
        ay[p] = subj_y[p]

    for e in range(3):
        e1 = (e + 1) % 3
        ex0, ey0 = clip_x[e], clip_y[e]
        edx, edy = clip_x[e1] - ex0, clip_y[e1] - ey0
        n_out = 0
        if n_in < 3:
            return 0.0, 0
        for i in range(n_in):
            pi = n_in - 1 if i == 0 else i - 1
            c_in = edx * (ay[i] - ey0) - edy * (ax[i] - ex0) >= 0.0
            p_in = edx * (ay[pi] - ey0) - edy * (ax[pi] - ex0) >= 0.0
            if p_in != c_in:
                dxp, dyp = ax[i] - ax[pi], ay[i] - ay[pi]
                denom = edx * dyp - edy * dxp
                t = (
                    0.0
                    if denom == 0.0
                    else (edx * (ay[pi] - ey0) - edy * (ax[pi] - ex0)) / -denom
                )
                bx[n_out] = ax[pi] + t * dxp
                by[n_out] = ay[pi] + t * dyp
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
        return 0.0, 0
    area = 0.0
    for i in range(n_in):
        j = (i + 1) % n_in
        area += ax[i] * ay[j] - ax[j] * ay[i]
        out_x[i] = ax[i]
        out_y[i] = ay[i]
    return abs(area) * 0.5, n_in


@njit(parallel=True, cache=True)
def _batch_clip_numba_convexclip_and_verts(
    subj_vx: NDArrayFloat,
    subj_vy: NDArrayFloat,
    n_subj: NDArrayInt,
    clip_vx: NDArrayFloat,
    clip_vy: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    N = len(n_subj)
    width = subj_vx.shape[1]
    areas = np.empty(N)
    out_x = np.zeros((N, width))
    out_y = np.zeros((N, width))
    out_n = np.zeros(N, dtype=np.intp)
    for idx in prange(N):  # ty:ignore[not-iterable]
        area, n_out_verts = _clip_convexclip_with_verts(
            subj_vx[idx],
            subj_vy[idx],
            n_subj[idx],
            clip_vx[idx],
            clip_vy[idx],
            out_x[idx],
            out_y[idx],
        )
        areas[idx] = area
        out_n[idx] = n_out_verts
    return areas, out_x, out_y, out_n


def _batch_clip_numpy_convexclip_and_verts(
    subj_vx: NDArrayFloat,
    subj_vy: NDArrayFloat,
    n_subj: NDArrayInt,
    clip_vx: NDArrayFloat,
    clip_vy: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Numpy/pure-Python fallback for :func:`_batch_clip_numba_convexclip_and_verts`."""
    N = len(n_subj)
    width = subj_vx.shape[1]
    areas = np.empty(N)
    out_x = np.zeros((N, width))
    out_y = np.zeros((N, width))
    out_n = np.zeros(N, dtype=np.intp)
    for k in range(N):
        m = int(n_subj[k])
        subj = list(zip(subj_vx[k, :m].tolist(), subj_vy[k, :m].tolist()))
        clip = list(zip(clip_vx[k].tolist(), clip_vy[k].tolist()))
        area, out_poly = _sh_clip_python_generic_with_verts(subj, clip)
        areas[k] = area
        mo = len(out_poly)
        out_n[k] = mo
        for p in range(mo):
            out_x[k, p] = out_poly[p][0]
            out_y[k, p] = out_poly[p][1]
    return areas, out_x, out_y, out_n


def _batch_clip_areas_and_verts_convexclip(
    subj_vx: NDArrayFloat,
    subj_vy: NDArrayFloat,
    n_subj: NDArrayInt,
    clip_vx: NDArrayFloat,
    clip_vy: NDArrayFloat,
) -> Tuple[NDArrayFloat, NDArrayFloat, NDArrayFloat, NDArrayInt]:
    """Dispatch to the fastest available convex-vs-triangle clip
    back-end, also returning clipped vertices."""
    if _HAS_NUMBA:
        return _batch_clip_numba_convexclip_and_verts(
            subj_vx,  # ty: ignore[invalid-argument-type]
            subj_vy,  # ty: ignore[invalid-argument-type]
            n_subj,  # ty: ignore[invalid-argument-type]
            clip_vx,  # ty: ignore[invalid-argument-type]
            clip_vy,  # ty: ignore[invalid-argument-type]
        )
    return _batch_clip_numpy_convexclip_and_verts(
        subj_vx, subj_vy, n_subj, clip_vx, clip_vy
    )


# ===================================================================
# Sanity check
# ===================================================================


def _polygons_from_padded_verts(
    vx: NDArrayFloat,
    vy: NDArrayFloat,
    n_verts: NDArrayInt,
    local_to_world: Optional[Tuple[NDArrayFloat, float, float]] = None,
) -> NDArrayFloat:
    """
    Build an array of ``shapely.Polygon`` objects from a padded,
    per-row clipped-vertex buffer, using the vectorized
    ``shapely.polygons`` constructor (object construction only -- this
    never invokes GEOS's own clipping/intersection algorithm, since
    the vertices were already computed by one of this module's own
    numba/numpy Sutherland-Hodgman kernels).

    Parameters
    ----------
    vx, vy : ndarray, shape (N, max_verts)
        Padded clipped-polygon vertex coordinates, one row per pair.
        Only the first ``n_verts[k]`` columns of row ``k`` are
        meaningful; the rest may hold arbitrary padding.
    n_verts : ndarray, shape (N,)
        Number of meaningful vertices per row. Rows with
        ``n_verts[k] < 3`` produce ``None`` (no valid polygon).
    local_to_world : (center, cos_angle, sin_angle), optional
        If given, vertices are first rotated by ``(cos_angle,
        sin_angle)`` and translated by ``center`` -- i.e. mapped from
        a grid's local (unrotated) frame into world coordinates -- to
        match the convention used elsewhere in this module (e.g. the
        fully generic path, which always works and reports geometry
        in world coordinates). If ``None``, ``vx``/``vy`` are assumed
        to already be in world coordinates.

    Returns
    -------
    polys : ndarray of object, shape (N,)
        One ``shapely.Polygon`` (or ``None`` where ``n_verts[k] < 3``)
        per row, in the same order as the input.
    """
    N = len(n_verts)
    polys = np.full(N, None, dtype=object)
    valid = n_verts >= 3
    if not valid.any():
        return polys

    idx = np.flatnonzero(valid)
    counts = n_verts[idx]

    # Group by distinct vertex count and build each group with one
    # vectorized `shapely.polygons` call on a regular (n_rows, k, 2)
    # array. There are typically only a handful of distinct counts
    # (e.g. {4}, or {3, 4, 5, 6} for a rotated-cell clip).
    #
    # A ragged one-shot alternative exists -- `shapely.linearrings`
    # accepts a flat coordinate array plus an `indices` argument, which
    # would build every polygon in a single call regardless of vertex
    # count -- but it measured consistently *slower* than this grouped
    # form (~0.8x on a representative rotated rectilinear workload with
    # shapely 2.1.2): the per-group calls operate on contiguous regular
    # arrays, which GEOS ingests faster than an indices-driven ragged
    # batch, and that outweighs the cost of the few extra calls.
    #
    # The world transform is applied per group, so it touches only the
    # `counts.sum()` coordinates that are actually used rather than the
    # full padded block (most of whose columns are padding).
    for k in np.unique(counts):
        rows = idx[counts == k]
        sub_vx = vx[rows, :k]
        sub_vy = vy[rows, :k]
        if local_to_world is not None:
            center, ca, sa = local_to_world
            wx = center[0] + sub_vx * ca - sub_vy * sa
            wy = center[1] + sub_vx * sa + sub_vy * ca
        else:
            wx, wy = sub_vx, sub_vy
        polys[rows] = shapely.polygons(np.stack([wx, wy], axis=-1))

    return polys


def _no_intersections() -> IntersectionsArray:
    """
    The empty :data:`IntersectionsArray` returned whenever per-pair
    geometry was not requested (``with_intersections=False``) or a path
    produced no candidate pairs at all.
    """
    return np.empty(0, dtype=object)


def _pack_intersections(intersections: NDArrayFloat) -> IntersectionsArray:
    """
    Package a per-nonzero-entry array of intersection geometries for
    return, preserving alignment with the matrix's nonzero entries.

    This is a thin, defensive pass-through: every code path in this
    module builds an ``intersections`` entry (a ``shapely.Polygon``,
    occasionally a ``shapely.MultiPolygon`` on the exact Shapely-clip
    fallback branches -- e.g. when a concave polygon's intersection
    with an axis-aligned cell splits into disjoint pieces -- or a
    ``None`` placeholder on the fast analytic paths) for every nonzero
    matrix coefficient, so the input is already length-aligned with
    the matrix's ``nnz``. This function simply normalizes it to a
    plain ``numpy`` object array (see :data:`IntersectionsArray`) so
    every dispatch branch in this module returns the exact same type,
    regardless of whether the entries happen to be uniform single
    ``Polygon`` objects or a mix that also includes ``None`` /
    ``MultiPolygon`` entries -- unlike ``shapely.MultiPolygon(...)``,
    which cannot represent either of those cases (it silently drops
    ``None`` entries and raises on non-``Polygon`` geometry).
    """
    return np.asarray(intersections, dtype=object)


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


def _check_intersections_alignment(
    mat: csc_array, intersections: IntersectionsArray
) -> None:
    """
    Verify that ``mat.nnz == len(intersections)``.

    This holds for every code path that returns one ``intersections``
    entry (real geometry or a ``None`` placeholder) per nonzero matrix
    coefficient -- i.e. the fully generic path
    (:func:`_compute_transfer_matrix`), the rectilinear/polygon mixed
    path (:func:`_compute_transfer_matrix_mixed`), and the
    TriMesh/polygon path (:func:`_compute_transfer_matrix_trimesh_polygon`).

    It does NOT hold, by design, for the purely analytic (Shapely-free)
    paths that never materialize per-pair geometry at all --
    :func:`_compute_transfer_matrix_rectilinear`,
    :func:`_compute_transfer_matrix_rect_trimesh`, and
    :func:`_compute_transfer_matrix_trimesh` -- which always return an
    *empty* ``intersections`` regardless of ``mat.nnz``. This check is
    therefore only wired into
    :func:`compute_transfer_matrix_with_intersections` for the three
    paths listed above where the invariant is actually meant to hold.
    """
    n_intersections = len(intersections)
    if mat.nnz != n_intersections:
        raise AssertionError(
            "Intersection/matrix misalignment: W.nnz "
            f"({mat.nnz}) != len(intersections) ({n_intersections}). "
            "Every nonzero matrix entry must have exactly one "
            "corresponding (possibly None-placeholder) intersection "
            "geometry."
        )
