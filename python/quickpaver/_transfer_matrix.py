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

Public entry points:

- :func:`compute_transfer_matrix` -- the sparse matrix only.
- :func:`compute_transfer_matrix_with_intersections` -- the matrix plus the
  (source index, target index, intersection geometry) triplet of every
  nonzero entry, as Shapely geometries.
- :func:`compute_transfer_matrix_with_intersection_vertices` -- the same
  triplets, with the intersection geometry returned as plain ragged
  coordinate arrays (:class:`IntersectionVertices`), without creating any
  Shapely object. The Shapely output of
  :func:`compute_transfer_matrix_with_intersections` is built from the
  same Rust output.

All accept an optional boolean mask for either grid
(``source_grid_mask`` / ``target_grid_mask``): cells where the mask is
``False`` are excluded from the computation and contribute no nonzero
entries to the matrix (their row/column is entirely zero), while the
matrix shape itself is left unchanged so masking is transparent to
callers.

Each side of the transfer may be a :class:`~quickpaver._grid.RectilinearGrid`,
a :class:`~quickpaver._grid.TriMesh`, a :class:`shapely.MultiPolygon`, or a
1-D sequence / array of :class:`shapely.Polygon`. All the geometry runs in
the compiled Rust extension (:mod:`quickpaver._quickpaver`), in parallel and
without the GIL:

- **Both rectilinear**: two 1-D interval-overlap problems when the relative
  angle is a multiple of 90 degrees, otherwise analytic candidate
  enumeration plus a Sutherland-Hodgman clip of every rotated target cell
  against the source cells.
- **Rectilinear vs anything else**: analytic O(1) candidate ranges in the
  rectilinear grid, then a clip of each polygon against the cells, in the
  grid's local frame. Cells lying strictly inside a convex polygon get their
  area analytically (no clip). Concave and holed polygons are clipped too
  (exterior minus holes: the area is exact since the cell is convex), so
  GEOS is used only to build their intersection *geometry* when requested.
- **Irregular vs irregular** (TriMesh / polygons on both sides): grid-hash
  bounding-box join, then a clip using a convex side of each pair as the
  window. Only pairs of two non-convex polygons fall back to GEOS.

Every code path returns its per-pair ``intersections`` output as a plain
``numpy`` object array (see :data:`IntersectionsArray`) holding a
``shapely.Polygon`` per entry, or the geometry returned by GEOS (possibly
a ``MultiPolygon``) for non-convex pairs.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import NamedTuple, Optional, Tuple, Union, cast

import numpy as np
import shapely
from scipy.sparse import coo_array, csc_array

from quickpaver._grid import RectilinearGrid, TriMesh
from quickpaver._types import ArrayLike, NDArrayBool, NDArrayFloat, NDArrayInt

try:
    from quickpaver import _quickpaver as _rs
except ImportError as exc:  # pragma: no cover - depends on the installation
    import sys

    raise ImportError(
        "The compiled extension of quickpaver (quickpaver._quickpaver) is missing "
        "or was built for another Python version (this is Python "
        f"{sys.version_info.major}.{sys.version_info.minor}). In a development "
        "checkout, rebuild it with `uv sync --reinstall-package quickpaver` (or "
        "`maturin develop --release`); otherwise reinstall quickpaver with pip."
    ) from exc

#: Any of the grid representations accepted on either side of a transfer.
#: ``RectilinearGrid`` and ``TriMesh`` are handled by dedicated paths;
#: polygons may be given as a ``shapely.MultiPolygon`` or as a 1-D
#: sequence / array of ``shapely.Polygon``.
GridLike = Union[
    RectilinearGrid,
    TriMesh,
    shapely.MultiPolygon,
    Sequence[shapely.Polygon],
    NDArrayFloat,
]

#: Return type used for the per-nonzero-entry ``intersections`` output: a
#: 1-D numpy object array, one entry per matrix nonzero, holding a
#: ``shapely.Polygon`` (or the GEOS geometry, possibly a ``MultiPolygon``,
#: for pairs involving non-convex polygons).
IntersectionsArray = NDArrayFloat

# Kind of an entry returned by the Rust kernels.
_KIND_GEOM = 0  # area and geometry computed in Rust
_KIND_AREA = 1  # area computed in Rust, geometry from GEOS (non-convex)
_KIND_GEOS = 2  # area and geometry from GEOS (two non-convex polygons)

# Geometry requested from a path: none, Shapely objects, or ragged arrays.
_NO_GEOM, _SHAPELY, _VERTICES = 0, 1, 2


class IntersectionVertices(NamedTuple):
    """
    Intersection geometries as ragged coordinate arrays (no Shapely objects).

    The layout is the one of :func:`shapely.to_ragged_array` for
    multipolygons: entry ``k`` (the ``k``-th nonzero of the matrix) is made
    of the polygons ``geometry_offsets[k]:geometry_offsets[k + 1]``; polygon
    ``p`` of the rings ``polygon_offsets[p]:polygon_offsets[p + 1]``
    (exterior ring first, then holes); ring ``r`` of the closed-ring
    coordinates ``coords[ring_offsets[r]:ring_offsets[r + 1]]``.

    Pairs clipped in Rust (every pair with a convex side) always give one
    polygon with a single ring. Only pairs involving non-convex polygons can
    give several polygons or holes.

    Attributes
    ----------
    coords : ndarray, shape (n_coords, 2)
        World-frame coordinates of every ring, rings closed.
    ring_offsets : NDArrayInt, shape (n_rings + 1,)
    polygon_offsets : NDArrayInt, shape (n_polygons + 1,)
    geometry_offsets : NDArrayInt, shape (n_entries + 1,)
    """

    coords: NDArrayFloat
    ring_offsets: NDArrayInt
    polygon_offsets: NDArrayInt
    geometry_offsets: NDArrayInt

    @property
    def n_entries(self) -> int:
        """Number of entries (matrix nonzeros)."""
        return len(self.geometry_offsets) - 1

    def to_shapely(self) -> IntersectionsArray:
        """
        One Shapely geometry per entry: a ``Polygon`` for single-part
        entries, a ``MultiPolygon`` otherwise (``None`` for an entry with no
        polygon).
        """
        n = self.n_entries
        out = np.full(n, None, dtype=object)
        if n == 0:
            return out
        polys = shapely.from_ragged_array(
            shapely.GeometryType.POLYGON,
            self.coords,
            (self.ring_offsets, self.polygon_offsets),
        )
        n_parts = np.diff(self.geometry_offsets)
        single = n_parts == 1
        out[single] = polys[self.geometry_offsets[:-1][single]]
        multi = np.flatnonzero(n_parts > 1)
        if len(multi):
            owner = np.repeat(np.arange(n), n_parts)
            sel = n_parts[owner] > 1
            out[multi] = shapely.multipolygons(
                polys[sel], indices=np.searchsorted(multi, owner[sel])
            )
        return out


# ===================================================================
# Public API
# ===================================================================


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
    kind each of the two grids is (see the module docstring).

    Parameters
    ----------
    source_grid : RectilinearGrid, TriMesh, shapely.MultiPolygon or sequence of Polygon
        Source grid.
    target_grid : RectilinearGrid, TriMesh, shapely.MultiPolygon or sequence of Polygon
        Target grid.
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
    """
    return _dispatch_transfer(
        source_grid,
        target_grid,
        source_grid_mask,
        target_grid_mask,
        is_sanity_check,
        _NO_GEOM,
    )[0]


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
    source_grid : RectilinearGrid, TriMesh, shapely.MultiPolygon or sequence of Polygon
        Source grid.
    target_grid : RectilinearGrid, TriMesh, shapely.MultiPolygon or sequence of Polygon
        Target grid, with the same convention as ``source_grid``.
    source_grid_mask : 1-D array of bool, optional
        Boolean mask over the source grid cells (length must equal the
        number of source cells: ``nx * ny`` for a ``RectilinearGrid``,
        ``n_tri`` for a ``TriMesh``, the number of polygons otherwise).
        Cells where the mask is ``False`` are excluded from the
        computation: they contribute no nonzero row in the returned
        matrix. If ``None``, all source cells are considered.
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
        |S_i ∩ T_j| / |S_i|``, so rows sum to 1 for fully covered,
        unmasked source polygons. The shape always reflects the full
        (unmasked) grid sizes; masked-out cells simply have no nonzero
        entries.
    source_indices : NDArrayInt, shape (nnz,)
        Source polygon id of each nonzero entry (row index in ``W``).
    target_indices : NDArrayInt, shape (nnz,)
        Target polygon id of each nonzero entry (column index in ``W``).
    intersections : IntersectionsArray, shape (nnz,)
        Intersection geometries (world coordinates), parallel to
        ``source_indices`` / ``target_indices``.
    """
    mat, src, tgt, ix = _dispatch_transfer(
        source_grid,
        target_grid,
        source_grid_mask,
        target_grid_mask,
        is_sanity_check,
        _SHAPELY,
    )
    return mat, src, tgt, cast(IntersectionsArray, ix)


def compute_transfer_matrix_with_intersection_vertices(
    source_grid: GridLike,
    target_grid: GridLike,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionVertices]:
    """
    Same as :func:`compute_transfer_matrix_with_intersections`, with the
    intersection geometries returned as ragged coordinate arrays instead of
    Shapely objects.

    This is the fast path when the geometry is consumed numerically
    (plotting, export, centroids, ...): building millions of Shapely objects
    costs far more than the transfer itself. Call
    :meth:`IntersectionVertices.to_shapely` to get the Shapely geometries.

    Parameters
    ----------
    source_grid, target_grid, source_grid_mask, target_grid_mask, is_sanity_check
        See :func:`compute_transfer_matrix_with_intersections`.

    Returns
    -------
    W : csc_array, shape (n_source, n_target)
        Sparse conservative transfer matrix.
    source_indices, target_indices : NDArrayInt, shape (nnz,)
        Row / column index of each nonzero entry.
    intersections : IntersectionVertices
        Intersection geometry of each entry (``n_entries == nnz``).
    """
    mat, src, tgt, iv = _dispatch_transfer(
        source_grid,
        target_grid,
        source_grid_mask,
        target_grid_mask,
        is_sanity_check,
        _VERTICES,
    )
    return mat, src, tgt, cast(IntersectionVertices, iv)


# ===================================================================
# Shared plumbing
# ===================================================================


class _Entries(NamedTuple):
    """Raw output of a transfer path (before matrix assembly)."""

    src: NDArrayInt
    tgt: NDArrayInt
    weights: NDArrayFloat
    #: Rust geometry: closed rings of the entries with ``ring_len > 0``.
    coords: NDArrayFloat
    ring_len: NDArrayInt
    #: GEOS geometry of the entries with ``ring_len == 0`` (``None`` elsewhere).
    geos: Optional[NDArrayFloat]


def _empty_entries() -> _Entries:
    e = np.empty(0, dtype=np.intp)
    return _Entries(e, e, np.empty(0), np.empty((0, 2)), e, np.empty(0, dtype=object))


def _validate_mask(
    mask: Optional[NDArrayBool], n_cells: int, name: str
) -> Optional[NDArrayBool]:
    """Validate and normalize an optional boolean mask to shape (n_cells,)."""
    if mask is None:
        return None
    mask = np.ascontiguousarray(mask, dtype=bool)
    if mask.ndim != 1 or mask.shape[0] != n_cells:
        raise ValueError(
            f"{name} must be a 1-D boolean array of length {n_cells}, "
            f"got shape {mask.shape}."
        )
    return mask


def _as_polygon_array(grid: GridLike) -> NDArrayFloat:
    """Polygons of a ``MultiPolygon`` / sequence as a 1-D object array."""
    if isinstance(grid, shapely.MultiPolygon):
        return shapely.get_parts(grid)
    arr = np.asarray(grid, dtype=object)
    if arr.ndim != 1:
        raise TypeError(
            "A polygon grid must be a shapely.MultiPolygon or a 1-D sequence of "
            f"shapely.Polygon, got an array of shape {arr.shape}."
        )
    is_polygon = shapely.get_type_id(arr) == shapely.GeometryType.POLYGON
    if len(arr) and not np.all(is_polygon):
        raise TypeError("Every cell of a polygon grid must be a shapely.Polygon.")
    return arr


class _Ragged(NamedTuple):
    """Polygons as ragged arrays (layout expected by the Rust kernels)."""

    x: NDArrayFloat
    y: NDArrayFloat
    ring_off: NDArrayInt
    poly_ring: NDArrayInt


def _ragged_polygons(polygons: NDArrayFloat) -> _Ragged:
    """Rings (exterior first, closed) of every polygon, one GEOS pass."""
    n = len(polygons)
    rings, ring_poly = shapely.get_rings(polygons, return_index=True)
    coords, coord_ring = shapely.get_coordinates(rings, return_index=True)
    ring_off = np.searchsorted(coord_ring, np.arange(len(rings) + 1)).astype(np.intp)
    poly_ring = np.searchsorted(ring_poly, np.arange(n + 1)).astype(np.intp)
    return _Ragged(
        np.ascontiguousarray(coords[:, 0]),
        np.ascontiguousarray(coords[:, 1]),
        ring_off,
        poly_ring,
    )


def _ragged_triangles(tx: NDArrayFloat, ty: NDArrayFloat) -> _Ragged:
    """``(n, 3)`` triangle vertex arrays as closed-ring ragged arrays."""
    n = len(tx)
    return _Ragged(
        np.ascontiguousarray(np.concatenate([tx, tx[:, :1]], axis=1).ravel()),
        np.ascontiguousarray(np.concatenate([ty, ty[:, :1]], axis=1).ravel()),
        np.arange(n + 1, dtype=np.intp) * 4,
        np.arange(n + 1, dtype=np.intp),
    )


def _assemble_csc(
    weights: NDArrayFloat,
    rows: NDArrayInt,
    cols: NDArrayInt,
    shape: Tuple[int, int],
) -> csc_array:
    """
    CSC matrix from *unique* (row, col) entries: every path produces each
    (source, target) pair at most once, so no duplicate summation is needed
    and the arrays are built directly (Rust counting sort).
    """
    if len(weights) < 2**31 - 1 and max(shape) < 2**31 - 1:
        data, indices, indptr = _rs.coo_to_csc(
            np.ascontiguousarray(rows, dtype=np.intp),
            np.ascontiguousarray(cols, dtype=np.intp),
            np.ascontiguousarray(weights, dtype=np.float64),
            int(shape[0]),
            int(shape[1]),
        )
        mat = csc_array((data, indices, indptr), shape=shape)
        mat.has_sorted_indices = True
        return mat
    return coo_array((weights, (rows, cols)), shape=shape).tocsc()  # pragma: no cover


def _rust_polygons(coords: NDArrayFloat, ring_len: NDArrayInt) -> IntersectionsArray:
    """Shapely polygons of the entries with Rust geometry (``None`` elsewhere)."""
    out = np.full(len(ring_len), None, dtype=object)
    has = np.flatnonzero(ring_len > 0)
    if len(has):
        ring_offsets = np.zeros(len(has) + 1, dtype=np.intp)
        np.cumsum(ring_len[has], out=ring_offsets[1:])
        out[has] = shapely.from_ragged_array(
            shapely.GeometryType.POLYGON,
            coords,
            (ring_offsets, np.arange(len(has) + 1)),
        )
    return out


def _entries_to_shapely(e: _Entries) -> IntersectionsArray:
    out = _rust_polygons(e.coords, e.ring_len)
    if e.geos is not None:
        sel = e.ring_len == 0
        out[sel] = e.geos[sel]
    return out


def _entries_to_vertices(e: _Entries) -> IntersectionVertices:
    """Merge the Rust rings and the (rare) GEOS geometries into one ragged set."""
    n = len(e.ring_len)
    slow = np.flatnonzero(e.ring_len == 0)
    if len(slow) == 0:
        # every entry is one Rust ring: the arrays are used as they are
        ring_offsets = np.zeros(n + 1, dtype=np.intp)
        np.cumsum(e.ring_len, out=ring_offsets[1:])
        one = np.arange(n + 1, dtype=np.intp)
        return IntersectionVertices(e.coords, ring_offsets, one, one.copy())
    fast = np.flatnonzero(e.ring_len > 0)

    # polygonal parts of the GEOS geometries, then their rings and coordinates
    if len(slow) and e.geos is not None:
        parts, part_owner = shapely.get_parts(e.geos[slow], return_index=True)
        keep = shapely.get_type_id(parts) == shapely.GeometryType.POLYGON
        keep &= ~shapely.is_empty(parts)
        parts, part_owner = parts[keep], slow[part_owner[keep]]
        rings, ring_part = shapely.get_rings(parts, return_index=True)
        g_coords, coord_ring = shapely.get_coordinates(rings, return_index=True)
        g_ring_len = np.bincount(coord_ring, minlength=len(rings))
        g_part_rings = np.bincount(ring_part, minlength=len(parts))
    else:
        part_owner = np.empty(0, dtype=np.intp)
        g_coords = np.empty((0, 2))
        g_ring_len = g_part_rings = ring_part = np.empty(0, dtype=np.intp)

    # entries -> polygons
    n_poly = np.zeros(n, dtype=np.intp)
    n_poly[fast] = 1
    np.add.at(n_poly, part_owner, 1)
    geometry_offsets = np.zeros(n + 1, dtype=np.intp)
    np.cumsum(n_poly, out=geometry_offsets[1:])
    poly_fast = geometry_offsets[fast]
    # rank of each GEOS part inside its entry (parts are grouped by owner)
    first = np.searchsorted(part_owner, part_owner, side="left")
    poly_slow = geometry_offsets[part_owner] + (np.arange(len(part_owner)) - first)

    # polygons -> rings
    n_polys = int(geometry_offsets[-1])
    rings_per_poly = np.zeros(n_polys, dtype=np.intp)
    rings_per_poly[poly_fast] = 1
    rings_per_poly[poly_slow] = g_part_rings
    polygon_offsets = np.zeros(n_polys + 1, dtype=np.intp)
    np.cumsum(rings_per_poly, out=polygon_offsets[1:])
    ring_fast = polygon_offsets[poly_fast]
    first_r = np.searchsorted(ring_part, ring_part, side="left")
    ring_slow = polygon_offsets[poly_slow[ring_part]] + (
        np.arange(len(ring_part)) - first_r
    )

    # rings -> coordinates
    n_rings = int(polygon_offsets[-1])
    coords_per_ring = np.zeros(n_rings, dtype=np.intp)
    coords_per_ring[ring_fast] = e.ring_len[fast]
    coords_per_ring[ring_slow] = g_ring_len
    ring_offsets = np.zeros(n_rings + 1, dtype=np.intp)
    np.cumsum(coords_per_ring, out=ring_offsets[1:])
    coords = np.empty((int(ring_offsets[-1]), 2))
    lens = e.ring_len[fast]
    start = np.repeat(ring_offsets[ring_fast], lens)
    local = np.arange(int(lens.sum())) - np.repeat(np.cumsum(lens) - lens, lens)
    coords[start + local] = e.coords
    g_start = np.repeat(ring_offsets[ring_slow], g_ring_len)
    g_local = np.arange(len(g_coords)) - np.repeat(
        np.cumsum(g_ring_len) - g_ring_len, g_ring_len
    )
    coords[g_start + g_local] = g_coords
    return IntersectionVertices(coords, ring_offsets, polygon_offsets, geometry_offsets)


def _no_intersections() -> IntersectionsArray:
    """The empty :data:`IntersectionsArray`."""
    return np.empty(0, dtype=object)


def _pack_intersections(intersections: NDArrayFloat) -> IntersectionsArray:
    """Normalize per-entry geometries to a plain numpy object array."""
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
    """Verify that ``mat.nnz == len(intersections)``."""
    if mat.nnz != len(intersections):
        raise AssertionError(
            "Intersection/matrix misalignment: W.nnz "
            f"({mat.nnz}) != len(intersections) ({len(intersections)}). "
            "Every nonzero matrix entry must have exactly one "
            "corresponding intersection geometry."
        )


def _n_cells(grid: GridLike) -> int:
    if isinstance(grid, RectilinearGrid):
        return grid.nx * grid.ny
    if isinstance(grid, TriMesh):
        return grid.n_tri
    return len(_as_polygon_array(grid))


# ===================================================================
# Dispatch
# ===================================================================


def _dispatch_transfer(
    source_grid: GridLike,
    target_grid: GridLike,
    source_grid_mask: Optional[NDArrayBool],
    target_grid_mask: Optional[NDArrayBool],
    is_sanity_check: bool,
    geom: int,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, object]:
    """Run the path matching the two grid kinds, assemble and format."""
    with_geom = geom != _NO_GEOM
    sg, tg = source_grid, target_grid
    structured = (RectilinearGrid, TriMesh)

    # polygon sides are converted once (also gives the cell counts)
    s_polys = None if isinstance(sg, structured) else _as_polygon_array(sg)
    t_polys = None if isinstance(tg, structured) else _as_polygon_array(tg)
    n_source = len(s_polys) if s_polys is not None else _n_cells(sg)
    n_target = len(t_polys) if t_polys is not None else _n_cells(tg)
    smask = _validate_mask(source_grid_mask, n_source, "source_grid_mask")
    tmask = _validate_mask(target_grid_mask, n_target, "target_grid_mask")

    if isinstance(sg, RectilinearGrid) and isinstance(tg, RectilinearGrid):
        e = _rect_rect_entries(sg, tg, smask, tmask, with_geom)
    elif isinstance(sg, RectilinearGrid) and isinstance(tg, TriMesh):
        e = _rect_trimesh_entries(sg, tg, True, smask, tmask, with_geom)
    elif isinstance(tg, RectilinearGrid) and isinstance(sg, TriMesh):
        e = _rect_trimesh_entries(tg, sg, False, tmask, smask, with_geom)
    elif isinstance(sg, RectilinearGrid) and t_polys is not None:
        e = _rect_polygon_entries(sg, t_polys, True, smask, tmask, with_geom)
    elif isinstance(tg, RectilinearGrid) and s_polys is not None:
        e = _rect_polygon_entries(tg, s_polys, False, tmask, smask, with_geom)
    elif isinstance(sg, TriMesh) and isinstance(tg, TriMesh):
        e = _trimesh_trimesh_entries(sg, tg, smask, tmask, with_geom)
    elif isinstance(sg, TriMesh) and t_polys is not None:
        e = _trimesh_polygon_entries(sg, t_polys, True, smask, tmask, with_geom)
    elif isinstance(tg, TriMesh) and s_polys is not None:
        e = _trimesh_polygon_entries(tg, s_polys, False, tmask, smask, with_geom)
    elif s_polys is not None and t_polys is not None:
        e = _polygon_polygon_entries(s_polys, t_polys, smask, tmask, with_geom)
    else:  # pragma: no cover - every combination is handled above
        raise TypeError("Unsupported grid types.")

    mat = _assemble_csc(e.weights, e.src, e.tgt, (n_source, n_target))
    if is_sanity_check:
        _check_conservation(mat)
    if geom == _NO_GEOM:
        return mat, e.src, e.tgt, _no_intersections()
    if geom == _SHAPELY:
        ix = _entries_to_shapely(e)
        _check_intersections_alignment(mat, ix)
        return mat, e.src, e.tgt, ix
    return mat, e.src, e.tgt, _entries_to_vertices(e)


# ===================================================================
# Rectilinear vs rectilinear
# ===================================================================


def _rect_rect_entries(
    source_grid: RectilinearGrid,
    target_grid: RectilinearGrid,
    smask: Optional[NDArrayBool],
    tmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    source_center = np.asarray((source_grid.cx, source_grid.cy), dtype=float)
    target_center = np.asarray((target_grid.cx, target_grid.cy), dtype=float)
    source_angle_rad = np.deg2rad(source_grid.theta)
    target_angle_rad = np.deg2rad(target_grid.theta)
    k_exact = (target_angle_rad - source_angle_rad) / (np.pi / 2)
    k_round = round(k_exact)
    args = (
        source_center,
        source_grid.dx,
        source_grid.dy,
        source_grid.nx,
        source_grid.ny,
        source_angle_rad,
        target_center,
        target_grid.dx,
        target_grid.dy,
        target_grid.nx,
        target_grid.ny,
    )
    if abs(k_exact - k_round) < 1e-9:
        return _separable_entries(*args, k_round % 4, smask, tmask, with_geom)
    return _nonseparable_entries(*args, target_angle_rad, smask, tmask, with_geom)


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
    overlaps, where ``lo``/``hi`` are the overlap interval's own bounds.
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


def _separable_entries(
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
    smask: Optional[NDArrayBool],
    tmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    """Relative rotation = k90 x 90 degrees: product of two 1-D overlaps."""
    src_x_edges = (
        np.arange(source_nx + 1, dtype=np.float64) - source_nx / 2
    ) * source_dx
    src_y_edges = (
        np.arange(source_ny + 1, dtype=np.float64) - source_ny / 2
    ) * source_dy

    ca, sa = np.cos(source_angle), np.sin(source_angle)
    dx_w, dy_w = target_center - source_center
    tgt_origin_x = dx_w * ca + dy_w * sa
    tgt_origin_y = -dx_w * sa + dy_w * ca

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

    is_x, it_x, ox, xlo, xhi = _compute_1d_overlaps(src_x_edges, tgt_x_edges)
    is_y, it_y, oy, ylo, yhi = _compute_1d_overlaps(src_y_edges, tgt_y_edges)
    if len(is_x) == 0 or len(is_y) == 0:
        return _empty_entries()

    c = np.ascontiguousarray
    src, tgt, weights, _kind, coords, ring_len = _rs.separable_transfer(
        c(is_x, dtype=np.intp),
        c(tx_perm[it_x], dtype=np.intp),
        c(ox),
        c(xlo),
        c(xhi),
        c(is_y, dtype=np.intp),
        c(ty_perm[it_y], dtype=np.intp),
        c(oy),
        c(ylo),
        c(yhi),
        bool(sx_is_tgt_y),
        int(source_nx),
        int(source_ny),
        int(target_nx),
        int(target_ny),
        float(source_dx * source_dy),
        smask,
        tmask,
        with_geom,
        (float(source_center[0]), float(source_center[1]), float(ca), float(sa)),
    )
    return _Entries(src, tgt, weights, coords, ring_len, None)


def _nonseparable_entries(
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
    smask: Optional[NDArrayBool],
    tmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    """Arbitrary relative angle: rotated target cells clipped in Rust."""
    rel_angle = target_angle - source_angle
    cr, sr = np.cos(rel_angle), np.sin(rel_angle)
    hdx, hdy = target_dx / 2, target_dy / 2
    template = np.array(
        [[-hdx, -hdy], [hdx, -hdy], [hdx, hdy], [-hdx, hdy]], dtype=float
    ) @ np.array([[cr, sr], [-sr, cr]])
    half_bx = abs(cr) * hdx + abs(sr) * hdy
    half_by = abs(sr) * hdx + abs(cr) * hdy

    ca, sa = np.cos(source_angle), np.sin(source_angle)
    dw = target_center - source_center
    origin_x = dw[0] * ca + dw[1] * sa
    origin_y = -dw[0] * sa + dw[1] * ca

    src, tgt, weights, _kind, coords, ring_len = _rs.nonseparable_transfer(
        int(source_nx),
        int(source_ny),
        float(source_dx),
        float(source_dy),
        int(target_nx),
        int(target_ny),
        float(target_dx),
        float(target_dy),
        float(origin_x),
        float(origin_y),
        float(cr),
        float(sr),
        [float(v) for v in template[:, 0]],
        [float(v) for v in template[:, 1]],
        float(half_bx),
        float(half_by),
        smask,
        tmask,
        with_geom,
        (float(source_center[0]), float(source_center[1]), float(ca), float(sa)),
    )
    return _Entries(src, tgt, weights, coords, ring_len, None)


def _nonseparable_transfer(
    source_center: ArrayLike,
    source_dx: float,
    source_dy: float,
    source_nx: int,
    source_ny: int,
    source_angle: float,
    target_center: ArrayLike,
    target_dx: float,
    target_dy: float,
    target_nx: int,
    target_ny: int,
    target_angle: float,
    with_intersections: bool = True,
    source_grid_mask: Optional[NDArrayBool] = None,
    target_grid_mask: Optional[NDArrayBool] = None,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayFloat, IntersectionsArray]:
    """
    Transfer between two rectilinear grids at an arbitrary angle (angles in
    radians), as raw ``(src_lin, tgt_lin, weights, intersections)``.
    """
    e = _nonseparable_entries(
        np.asarray(source_center, dtype=float),
        source_dx,
        source_dy,
        source_nx,
        source_ny,
        source_angle,
        np.asarray(target_center, dtype=float),
        target_dx,
        target_dy,
        target_nx,
        target_ny,
        target_angle,
        source_grid_mask,
        target_grid_mask,
        with_intersections,
    )
    ix = _entries_to_shapely(e) if with_intersections else _no_intersections()
    return e.src, e.tgt, e.weights, ix


# ===================================================================
# Rectilinear vs TriMesh / polygons
# ===================================================================


def _rect_frames(rect: RectilinearGrid):
    """Local-frame edges and world<->local frame parameters of a grid."""
    center = np.asarray((rect.cx, rect.cy), dtype=float)
    angle = np.deg2rad(rect.theta)
    ca, sa = float(np.cos(angle)), float(np.sin(angle))
    x_edges = (np.arange(rect.nx + 1, dtype=np.float64) - rect.nx / 2) * rect.dx
    y_edges = (np.arange(rect.ny + 1, dtype=np.float64) - rect.ny / 2) * rect.dy
    frame = (float(center[0]), float(center[1]), ca, sa)
    return center, angle, x_edges, y_edges, frame


def _rect_cell_polygons_world(
    rect: RectilinearGrid,
    x_edges: NDArrayFloat,
    y_edges: NDArrayFloat,
    cells: NDArrayInt,
) -> NDArrayFloat:
    """World-frame Shapely polygons of the given (linear-index) cells."""
    center, angle, *_ = _rect_frames(rect)
    ca, sa = np.cos(angle), np.sin(angle)
    cell_j, cell_i = np.divmod(cells, rect.nx)
    x0, x1 = x_edges[cell_i], x_edges[cell_i + 1]
    y0, y1 = y_edges[cell_j], y_edges[cell_j + 1]
    lx = np.stack([x0, x1, x1, x0], axis=-1)
    ly = np.stack([y0, y0, y1, y1], axis=-1)
    wx = center[0] + lx * ca - ly * sa
    wy = center[1] + lx * sa + ly * ca
    return shapely.polygons(np.stack([wx, wy], axis=-1))


def _rect_items(
    rect: RectilinearGrid,
    ragged: _Ragged,
    convex: Optional[NDArrayBool],
    rmask: Optional[NDArrayBool],
    imask: Optional[NDArrayBool],
    with_geom: bool,
):
    _, _, x_edges, y_edges, frame = _rect_frames(rect)
    return (
        _rs.rect_items_transfer(
            ragged.x,
            ragged.y,
            ragged.ring_off,
            ragged.poly_ring,
            convex,
            frame,
            x_edges,
            y_edges,
            rmask,
            imask,
            with_geom,
            frame,
        ),
        x_edges,
        y_edges,
    )


def _orient(rect_is_source, item, cell):
    return (cell, item) if rect_is_source else (item, cell)


def _rect_trimesh_entries(
    rect: RectilinearGrid,
    mesh: TriMesh,
    rect_is_source: bool,
    rmask: Optional[NDArrayBool],
    mmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    if mesh.n_tri == 0:
        return _empty_entries()
    tx, ty = _tri_world_coords(mesh)
    (item, cell, areas, _kind, coords, ring_len), _, _ = _rect_items(
        rect,
        _ragged_triangles(tx, ty),
        np.ones(mesh.n_tri, dtype=bool),
        rmask,
        mmask,
        with_geom,
    )
    src, tgt = _orient(rect_is_source, item, cell)
    weights = (
        areas / (rect.dx * rect.dy) if rect_is_source else areas / mesh.tri_area_m2[src]
    )
    return _Entries(src, tgt, weights, coords, ring_len, None)


def _rect_polygon_entries(
    rect: RectilinearGrid,
    polygons: NDArrayFloat,
    rect_is_source: bool,
    rmask: Optional[NDArrayBool],
    pmask: Optional[NDArrayBool],
    with_geom: bool,
    polygon_grid_is_convex: bool = False,
) -> _Entries:
    if len(polygons) == 0:
        return _empty_entries()
    convex = np.ones(len(polygons), dtype=bool) if polygon_grid_is_convex else None
    (item, cell, areas, kind, coords, ring_len), x_edges, y_edges = _rect_items(
        rect, _ragged_polygons(polygons), convex, rmask, pmask, with_geom
    )
    geos = None
    if with_geom:
        # exact geometry of the non-convex (or holed) polygons from GEOS
        need = np.flatnonzero(kind == _KIND_AREA)
        geos = np.full(len(item), None, dtype=object)
        if len(need):
            geos[need] = shapely.intersection(
                _rect_cell_polygons_world(rect, x_edges, y_edges, cell[need]),
                polygons[item[need]],
            )
    src, tgt = _orient(rect_is_source, item, cell)
    if rect_is_source:
        weights = areas / (rect.dx * rect.dy)
    else:
        weights = areas / shapely.area(polygons)[src]
    return _Entries(src, tgt, weights, coords, ring_len, geos)


def _compute_transfer_matrix_mixed(
    rectilinear_grid: RectilinearGrid,
    polygon_grid: GridLike,
    rectilinear_is_source: bool,
    rectilinear_grid_mask: Optional[NDArrayBool] = None,
    polygon_grid_mask: Optional[NDArrayBool] = None,
    is_sanity_check: bool = False,
    with_intersections: bool = True,
    polygon_grid_is_convex: bool = False,
) -> Tuple[csc_array, NDArrayInt, NDArrayInt, IntersectionsArray]:
    """
    Rectilinear grid vs polygon grid, with the option to declare every
    polygon convex (``polygon_grid_is_convex``, skipping the convexity test).
    """
    polygons = _as_polygon_array(polygon_grid)
    n_rect, n_poly = rectilinear_grid.nx * rectilinear_grid.ny, len(polygons)
    rmask = _validate_mask(rectilinear_grid_mask, n_rect, "rectilinear grid mask")
    pmask = _validate_mask(polygon_grid_mask, n_poly, "polygon grid mask")
    e = _rect_polygon_entries(
        rectilinear_grid,
        polygons,
        rectilinear_is_source,
        rmask,
        pmask,
        with_intersections,
        polygon_grid_is_convex,
    )
    shape = (n_rect, n_poly) if rectilinear_is_source else (n_poly, n_rect)
    mat = _assemble_csc(e.weights, e.src, e.tgt, shape)
    if is_sanity_check:
        _check_conservation(mat)
    ix = _entries_to_shapely(e) if with_intersections else _no_intersections()
    return mat, e.src, e.tgt, ix


# ===================================================================
# TriMesh / polygons on both sides
# ===================================================================


def _tri_world_coords(mesh: TriMesh) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Per-triangle vertex world coordinates, each of shape ``(n_tri, 3)``."""
    tv = mesh.tri_verts
    return mesh.verts_xy[:, 0][tv], mesh.verts_xy[:, 1][tv]


def _ensure_ccw_triangles(
    tx: NDArrayFloat, ty: NDArrayFloat
) -> Tuple[NDArrayFloat, NDArrayFloat]:
    """Copy of ``(tx, ty)`` with every triangle wound counter-clockwise."""
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


def _join(
    a: _Ragged,
    a_convex: Optional[NDArrayBool],
    b: _Ragged,
    b_convex: Optional[NDArrayBool],
    amask: Optional[NDArrayBool],
    bmask: Optional[NDArrayBool],
    prefer_a_window: bool,
    with_geom: bool,
):
    return _rs.join_clip_transfer(
        a.x,
        a.y,
        a.ring_off,
        a.poly_ring,
        a_convex,
        b.x,
        b.y,
        b.ring_off,
        b.poly_ring,
        b_convex,
        amask,
        bmask,
        prefer_a_window,
        with_geom,
    )


def _trimesh_trimesh_entries(
    source_mesh: TriMesh,
    target_mesh: TriMesh,
    smask: Optional[NDArrayBool],
    tmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    if source_mesh.n_tri == 0 or target_mesh.n_tri == 0:
        return _empty_entries()
    src_t = _ragged_triangles(*_ensure_ccw_triangles(*_tri_world_coords(source_mesh)))
    tgt_t = _ragged_triangles(*_ensure_ccw_triangles(*_tri_world_coords(target_mesh)))
    # subject = source triangle, window = target triangle
    src, tgt, areas, _kind, coords, ring_len = _join(
        src_t,
        np.ones(source_mesh.n_tri, dtype=bool),
        tgt_t,
        np.ones(target_mesh.n_tri, dtype=bool),
        smask,
        tmask,
        False,
        with_geom,
    )
    weights = areas / source_mesh.tri_area_m2[src]
    return _Entries(src, tgt, weights, coords, ring_len, None)


def _trimesh_polygon_entries(
    mesh: TriMesh,
    polygons: NDArrayFloat,
    trimesh_is_source: bool,
    mmask: Optional[NDArrayBool],
    pmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    if mesh.n_tri == 0 or len(polygons) == 0:
        return _empty_entries()
    tx, ty = _ensure_ccw_triangles(*_tri_world_coords(mesh))
    # window = triangle (always convex), subject = polygon
    tri, poly, areas, kind, coords, ring_len = _join(
        _ragged_triangles(tx, ty),
        np.ones(mesh.n_tri, dtype=bool),
        _ragged_polygons(polygons),
        None,
        mmask,
        pmask,
        True,
        with_geom,
    )
    geos = None
    if with_geom:
        need = np.flatnonzero(kind == _KIND_AREA)
        geos = np.full(len(tri), None, dtype=object)
        if len(need):
            wx, wy = _tri_world_coords(mesh)
            tri_polys = shapely.polygons(
                np.stack([wx[tri[need]], wy[tri[need]]], axis=-1)
            )
            geos[need] = shapely.intersection(tri_polys, polygons[poly[need]])
    if trimesh_is_source:
        src, tgt = tri, poly
        weights = areas / mesh.tri_area_m2[src]
    else:
        src, tgt = poly, tri
        weights = areas / shapely.area(polygons)[src]
    return _Entries(src, tgt, weights, coords, ring_len, geos)


def _polygon_polygon_entries(
    source_polygons: NDArrayFloat,
    target_polygons: NDArrayFloat,
    smask: Optional[NDArrayBool],
    tmask: Optional[NDArrayBool],
    with_geom: bool,
) -> _Entries:
    if len(source_polygons) == 0 or len(target_polygons) == 0:
        return _empty_entries()
    # window = target polygon when convex, else source polygon when convex;
    # pairs of two non-convex polygons come back as kind 2 (GEOS)
    src, tgt, areas, kind, coords, ring_len = _join(
        _ragged_polygons(source_polygons),
        None,
        _ragged_polygons(target_polygons),
        None,
        smask,
        tmask,
        False,
        with_geom,
    )
    geos = None
    need = (
        np.flatnonzero(kind != 0) if with_geom else np.flatnonzero(kind == _KIND_GEOS)
    )
    if len(need):
        ix = shapely.intersection(
            source_polygons[src[need]], target_polygons[tgt[need]]
        )
        both = kind[need] == _KIND_GEOS
        areas[need[both]] = shapely.area(ix[both])
        if with_geom:
            geos = np.full(len(src), None, dtype=object)
            geos[need] = ix
    elif with_geom:
        geos = np.full(len(src), None, dtype=object)
    # GEOS areas can be ~0 (touching / sliver pairs): same filter as Rust
    # (only kind-2 entries can be dropped, Rust entries are already filtered)
    keep = areas > 1e-15
    if not keep.all():
        src, tgt, areas = src[keep], tgt[keep], areas[keep]
        if geos is not None:
            ring_len, geos = ring_len[keep], geos[keep]
    weights = areas / shapely.area(source_polygons)[src]
    return _Entries(src, tgt, weights, coords, ring_len, geos)
