# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Tiling with triangles, squares and hexagons with support for anisotropy and
grid alignment with a given point."""

import math
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import shapely
import shapely.affinity
from numpy.typing import ArrayLike

from quickpaver._types import NDArrayBool, NDArrayFloat, NDArrayInt, StrEnum

SQRT3 = math.sqrt(3)


@dataclass(frozen=True)
class Disk:
    """A disk described analytically by its centre and radius.

    A :class:`Disk` is accepted anywhere a ``surface_to_cover`` is expected.
    Passing one instead of ``shapely.Point(...).buffer(radius)`` is both
    faster and more accurate: the tiling functions then decide tile
    membership with the exact circle rather than with the inscribed polygon
    that :meth:`shapely.Geometry.buffer` produces, and they never call a
    geometric predicate.

    Parameters
    ----------
    center : tuple of (float, float)
        ``(x, y)`` world-space coordinates of the disk centre.
    radius : float
        Radius of the disk. Must be strictly positive.

    Raises
    ------
    ValueError
        If ``radius`` is not strictly positive.

    Examples
    --------
    >>> tiling, adjacency = gen_hexagonal_tiling(Disk((0.0, 0.0), 5.0), 1.0)
    """

    center: Tuple[float, float]
    radius: float

    def __post_init__(self) -> None:
        if self.radius <= 0.0:
            raise ValueError("radius must be strictly positive.")

    @property
    def bounds(self) -> Tuple[float, float, float, float]:
        """Return ``(x_min, y_min, x_max, y_max)`` of the disk."""
        x, y = self.center
        r = self.radius
        return (x - r, y - r, x + r, y + r)

    @property
    def centroid(self) -> shapely.Point:
        """Return the disk centre as a :class:`shapely.Point`."""
        return shapely.Point(self.center)

    def as_array(self) -> NDArrayFloat:
        """Return the disk centre as an array of shape ``(2,)``."""
        return np.asarray(self.center, dtype=float)

    def to_polygon(self, quad_segs: int = 64) -> shapely.Polygon:
        """Return a polygonal approximation of the disk.

        Parameters
        ----------
        quad_segs : int, optional
            Number of segments used to approximate a quarter circle, by
            default ``64``.

        Returns
        -------
        shapely.Polygon
            Polygon inscribed in the disk.
        """
        return self.centroid.buffer(self.radius, quad_segs=quad_segs)


Surface = Union[shapely.Polygon, shapely.MultiPolygon, Disk]


def _distance_to_convex_rings(point: NDArrayFloat, rings: NDArrayFloat) -> NDArrayFloat:
    """Distance from a single point to convex polygons, zero when inside.

    Parameters
    ----------
    point : NDArrayFloat, shape (2,)
        Query point.
    rings : NDArrayFloat, shape (n_polygons, n_vertices, 2)
        Open rings of convex polygons with counter-clockwise vertices.

    Returns
    -------
    NDArrayFloat, shape (n_polygons,)
        Distance from ``point`` to each polygon, ``0.0`` when the point
        lies inside or on the boundary.
    """
    start = rings
    edge = np.roll(rings, -1, axis=1) - rings
    to_point = point[None, None, :] - start

    # Parameter of the orthogonal projection on each edge, clamped to the
    # segment so that the nearest point of the edge is obtained.
    ratio = np.clip((to_point * edge).sum(-1) / (edge * edge).sum(-1), 0.0, 1.0)
    nearest = start + ratio[..., None] * edge
    distance = np.hypot(point[0] - nearest[..., 0], point[1] - nearest[..., 1]).min(
        axis=1
    )

    # A counter-clockwise ring contains the point when it lies left of
    # every edge, in which case the distance is zero.
    side = edge[..., 0] * to_point[..., 1] - edge[..., 1] * to_point[..., 0]
    return np.where((side >= 0.0).all(axis=1), 0.0, distance)


def _disk_mask_from_rings(rings: NDArrayFloat, disk: Disk) -> NDArrayInt:
    """Exact keep mask of convex tiles against a disk.

    A tile is kept when its distance to the disk centre does not exceed the
    radius.  Two circle tests settle every tile except those straddling the
    rim: a tile whose centroid lies within ``radius - r_out`` is wholly
    inside, and one beyond ``radius + r_out`` cannot touch, where ``r_out``
    is the largest circumradius of the tiles.  Only the annulus in between,
    which holds ``O(perimeter / edge_length)`` tiles, needs the exact
    polygon distance.

    Parameters
    ----------
    rings : NDArrayFloat, shape (n_tiles, n_vertices, 2)
        Open rings of the candidate tiles.
    disk : Disk
        Disk to cover.

    Returns
    -------
    NDArrayInt of bool, shape (n_tiles,)
        Mask of the tiles intersecting the disk.
    """
    mask = np.zeros(len(rings), dtype=bool)
    if len(rings) == 0:
        return mask

    centroids = rings.mean(axis=1)
    relative = rings - centroids[:, None, :]
    r_out = float(np.hypot(relative[..., 0], relative[..., 1]).max())

    centre = disk.as_array()
    d2 = ((centroids - centre) ** 2).sum(axis=1)
    r_in = max(disk.radius - r_out, 0.0)

    inside = d2 <= r_in * r_in
    band = ~inside & (d2 <= (disk.radius + r_out) ** 2)
    mask |= inside

    band_idx = np.flatnonzero(band)
    if band_idx.size:
        distance = _distance_to_convex_rings(centre, rings[band_idx])
        mask[band_idx[distance <= disk.radius]] = True
    return mask


def _ragged_arange(counts: NDArrayInt) -> NDArrayInt:
    """Concatenate ``arange(c)`` for every count ``c`` in *counts*.

    Parameters
    ----------
    counts : NDArrayInt, shape (n,)
        Length of each run.

    Returns
    -------
    NDArrayInt, shape (counts.sum(),)
        Within-run positions, i.e. ``[0..c0-1, 0..c1-1, ...]``.
    """
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.cumsum(counts) - counts
    return np.arange(total, dtype=np.int64) - np.repeat(offsets, counts)


def _pairs_to_adj(
    src: NDArrayInt, dst: NDArrayInt, n_nodes: int
) -> Dict[int, List[int]]:
    """Group ``(src, dst)`` index pairs into an adjacency dictionary.

    The pairs are encoded as single integers so that a single
    :func:`numpy.unique` call sorts them lexicographically and removes
    duplicates at once.  The resulting neighbour lists are therefore
    sorted in ascending order and duplicate-free.

    All destinations are converted to Python objects with one bulk
    ``tolist`` call and the per-node lists are obtained by slicing that
    flat list, which avoids both a per-edge ``append`` and the per-node
    array views that :func:`numpy.split` would create.

    Parameters
    ----------
    src, dst : NDArrayInt, shape (n_edges,)
        Compact source and destination indices of every edge.  Both must
        lie in ``[0, n_nodes)``.
    n_nodes : int
        Total number of nodes.  Nodes without any edge are present in the
        output with an empty neighbour list.

    Returns
    -------
    Dict[int, List[int]]
        Node index -> sorted list of neighbour indices.
    """
    if n_nodes == 0:
        return {}
    if len(src) == 0:
        return {i: [] for i in range(n_nodes)}

    # Encode each (src, dst) pair as a single integer and sort, which orders
    # the edges by source and, within a source, by destination.  A plain sort
    # is used rather than np.unique, whose hash-based path is an order of
    # magnitude slower on this many keys; duplicates are then dropped in a
    # single linear scan of the sorted array.
    code = src.astype(np.int64, copy=False) * n_nodes + dst
    code.sort()
    code = code[np.concatenate(([True], np.diff(code) != 0))]
    src_sorted, dst_sorted = np.divmod(code, n_nodes)

    # Slice the sorted destinations into one contiguous run per source.
    counts = np.bincount(src_sorted, minlength=n_nodes)
    flat = dst_sorted.tolist()
    ends = np.cumsum(counts).tolist()
    adj = {}
    start = 0
    for node, end in enumerate(ends):
        adj[node] = flat[start:end]
        start = end
    return adj


def _lattice_centres(
    bounds: Tuple[float, float, float, float],
    b1: NDArrayFloat,
    b2: NDArrayFloat,
    anchor: NDArrayFloat,
    margin: int = 2,
) -> NDArrayFloat:
    """Generate centre coordinates of a Bravais lattice covering *bounds*.

    The lattice is ``anchor + i*b1 + j*b2`` for integers *i*, *j*.  The
    integer ranges are chosen so the generated nodes cover the bounding
    box expanded by *margin* cells on every side.  Because *anchor* is a
    node by construction (``i = j = 0``), passing the alignment point as
    *anchor* guarantees a centre lands exactly on it — inside or outside
    the surface — with no post-hoc shift and therefore no coverage gaps.

    Parameters
    ----------
    bounds : tuple of (float, float, float, float)
        ``(x_min, y_min, x_max, y_max)`` of the surface to cover.
    b1, b2 : NDArrayFloat, shape (2,)
        Primitive lattice basis vectors (column and row directions,
        including any anisotropy).
    anchor : NDArrayFloat, shape (2,)
        A point that must be a lattice node (the alignment point, or a
        default such as the bbox corner).
    margin : int
        Extra rings of cells added beyond the bbox to guarantee full
        coverage.  Default 2.

    Returns
    -------
    NDArrayFloat, shape (2, nj, ni)
        Lattice centre coordinate meshes.
    """
    x_min, y_min, x_max, y_max = bounds
    basis = np.column_stack([b1, b2])
    basis_inv = np.linalg.inv(basis)

    # bbox corners expressed in fractional lattice coordinates
    corners = np.array(
        [[x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]],
        dtype=float,
    )
    frac = basis_inv @ (corners - anchor).T  # (2, 4)

    i_min = int(np.floor(frac[0].min())) - margin
    i_max = int(np.ceil(frac[0].max())) + margin
    j_min = int(np.floor(frac[1].min())) - margin
    j_max = int(np.ceil(frac[1].max())) + margin

    ii, jj = np.meshgrid(
        np.arange(i_min, i_max + 1),
        np.arange(j_min, j_max + 1),
    )
    # centres[:, j, i] = anchor + i*b1 + j*b2
    centres = (
        anchor[:, None, None]
        + b1[:, None, None] * ii[None, :, :]
        + b2[:, None, None] * jj[None, :, :]
    )
    return centres


DEFAULT_ALIGNMENT_POINT: Tuple[float, float] = (0.0, 0.0)


def _resolve_anchor(alignment_point: Optional[ArrayLike]) -> NDArrayFloat:
    """Return the lattice anchor for an optional alignment point.

    When no alignment point is given the world origin is used, which makes
    the lattice phase absolute: it does not depend on the surface, so two
    surfaces tiled with the same parameters produce mutually conforming
    tilings, and editing, clipping or extending a surface shifts no tile.

    Parameters
    ----------
    alignment_point : array-like of shape (2,), optional
        ``(x, y)`` world-space coordinate that must be a lattice node.

    Returns
    -------
    NDArrayFloat, shape (2,)
        Anchor passed to :func:`_lattice_centres`.
    """
    if alignment_point is None:
        return np.asarray(DEFAULT_ALIGNMENT_POINT, dtype=float)
    return np.asarray(alignment_point, dtype=float).ravel()[:2]


def _validate_inputs(edge_length: float, anisotropy_ratio: float) -> None:
    # Validate the requested triangle edge length.
    if edge_length <= 0.0:
        raise ValueError("edge_length must be strictly positive.")

    # Validate the requested anisotropy ratio.
    if anisotropy_ratio <= 0.0:
        raise ValueError("anisotropy_ratio must be strictly positive.")


class PolygonType(StrEnum):
    """Polygon types."""

    TRIANGLE = "triangle"
    RECTANGLE = "rectangle"
    HEXAGON = "hexagon"


def gen_polygon(
    poly_type: Union[PolygonType, str],
    edge_length: float = 1.0,
    anisotropy_ratio: float = 1.0,
) -> shapely.Polygon:
    """
    Generate a reference polygon centred at the origin.

    The generated polygon is centred on ``(0, 0)`` and aligned with the
    coordinate axes. The polygon can be used as a base tile geometry before
    translation, rotation, or placement on a tiling lattice.

    ``anisotropy_ratio`` scales the polygon along the y-axis only. A value of
    ``1.0`` gives an isotropic polygon. Values different from ``1.0`` stretch or
    compress the polygon vertically while preserving the x-coordinates.

    Parameters
    ----------
    poly_type : Union[PolygonType, str]
        Type of polygon to generate. Supported values are:

        - :attr:`PolygonType.TRIANGLE`
        - :attr:`PolygonType.RECTANGLE`
        - :attr:`PolygonType.HEXAGON`

        Equivalent string values such as ``"triangle"``, ``"rectangle"``, and
        ``"hexagon"`` are also accepted if compatible with :class:`PolygonType`.
    edge_length : float, optional
        Base edge length of the generated polygon, by default 1.0.

        For rectangles, this corresponds to the rectangle width. The rectangle
        height is ``edge_length * anisotropy_ratio``.

        For triangles, this controls the side length of the isotropic reference
        triangle before vertical anisotropic scaling.

        For hexagons, this controls the circumradius of the isotropic reference
        hexagon, which is equal to the side length for a regular hexagon.
    anisotropy_ratio : float, optional
        Scaling factor applied along the y-axis, by default 1.0. For example,
        ``anisotropy_ratio=2.0`` produces polygons twice as tall as their
        isotropic counterpart.

    Returns
    -------
    shapely.Polygon
        Polygon centred at ``(0, 0)`` with vertices ordered counter-clockwise.

    Raises
    ------
    ValueError
        If ``poly_type`` is not a recognised :class:`PolygonType` member or
        valid string value.

    Notes
    -----
    The triangle and hexagon are generated from angular coordinates around the
    origin. The rectangle is generated directly from its half-width and
    half-height.

    This function does not validate that ``edge_length`` or
    ``anisotropy_ratio`` are strictly positive. Callers that require positive
    dimensions should validate inputs before calling this function.
    """
    if poly_type == PolygonType.TRIANGLE:
        return shapely.Polygon(
            [
                [
                    math.cos(math.radians(angle)) * edge_length / SQRT3,
                    math.sin(math.radians(angle))
                    * edge_length
                    * anisotropy_ratio
                    / SQRT3,
                ]
                for angle in range(90, 450, 120)
            ]
        )
    if poly_type == PolygonType.RECTANGLE:
        x = edge_length / 2.0
        y = x * anisotropy_ratio
        return shapely.Polygon([[-x, -y], [x, -y], [x, y], [-x, y]])
    if poly_type == PolygonType.HEXAGON:
        return shapely.Polygon(
            [
                [
                    math.cos(math.radians(angle)) * edge_length,
                    math.sin(math.radians(angle)) * edge_length * anisotropy_ratio,
                ]
                for angle in range(0, 360, 60)
            ]
        )
    raise ValueError(PolygonType(poly_type))


def _reference_tile_vertices(
    poly_type: PolygonType, edge_length: float, anisotropy_ratio: float
) -> NDArrayFloat:
    """Return the open ring of the reference tile centred on the origin.

    The closing repeat of the exterior ring is dropped: ring closure is
    handled by :func:`shapely.polygons`, so carrying the duplicate vertex
    through the broadcasting step would only inflate the coordinate array.

    Parameters
    ----------
    poly_type : PolygonType
        Type of tile geometry.
    edge_length : float
        Base edge length of the tile.
    anisotropy_ratio : float
        Vertical anisotropy ratio of the tile.

    Returns
    -------
    NDArrayFloat, shape (n_vertices, 2)
        Vertex coordinates relative to the tile centre.
    """
    ring = np.asarray(
        gen_polygon(poly_type, edge_length, anisotropy_ratio).exterior.coords,
        dtype=float,
    )
    return ring[:-1]


def _tiles_from_centres(
    centres: NDArrayFloat,
    verts: NDArrayFloat,
    surface_to_cover: Surface,
) -> Tuple[NDArrayFloat, NDArrayBool]:
    """Instantiate the tiles of a centre lattice that touch *surface_to_cover*.

    The lattice is generated with a few extra rings of cells beyond the
    surface bounding box, so a sizeable share of the centres cannot yield
    an intersecting tile.  Those are discarded with a pure NumPy bounding
    box test on the tile envelopes before any geometry is built, since
    creating Shapely polygons and running the intersection predicate are
    by far the most expensive steps.

    When *surface_to_cover* is a :class:`Disk`, membership is decided
    analytically by :func:`_disk_mask_from_rings` and no polygon is built
    for a rejected tile.

    Parameters
    ----------
    centres : NDArrayFloat, shape (2, nj, ni)
        Tile centre coordinate meshes.
    verts : NDArrayFloat, shape (n_vertices, 2)
        Open ring of the reference tile, relative to its centre.
    surface_to_cover : Surface
        Surface the tiles must intersect to be kept.

    Returns
    -------
    kept_polygons : NDArrayFloat of shapely.Polygon, shape (n_kept,)
        Tiles intersecting the surface, in lattice order.
    mask : NDArray of bool, shape (nj * ni,)
        Keep mask over the full lattice, suitable for the structured
        adjacency builders.
    """
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    # Half-extent of the tile envelope, identical for every tile.
    half_w = float(np.abs(verts[:, 0]).max())
    half_h = float(np.abs(verts[:, 1]).max())

    # Keep only the centres whose tile envelope overlaps the surface bbox.
    cx, cy = centres
    candidates = (
        (cx + half_w >= x_min)
        & (cx - half_w <= x_max)
        & (cy + half_h >= y_min)
        & (cy - half_h <= y_max)
    ).ravel()

    # Broadcast the reference vertices onto the candidate centres only.
    centres_flat = np.moveaxis(centres, 0, -1).reshape(-1, 2)[candidates]
    rings = centres_flat[:, None, :] + verts[None, :, :]

    mask = np.zeros(candidates.size, dtype=bool)
    if isinstance(surface_to_cover, Disk):
        # Analytic membership: build only the tiles that are kept.
        kept = _disk_mask_from_rings(rings, surface_to_cover)
        mask[candidates] = kept
        return shapely.polygons(rings[kept]), mask

    polygons = shapely.polygons(rings)

    # Scatter the predicate result back onto the full lattice.
    mask[candidates] = intersects_mask(polygons, surface_to_cover)
    return polygons[mask[candidates]], mask


def _vectorized_grid_adjacency(
    n_rows: int, n_cols: int, mask: NDArrayBool, offsets: NDArrayFloat
) -> Dict[int, List[int]]:
    """Vectorized adjacency builder for a uniform offset list.

    Every offset contributes a batch of ``(source, destination)`` pairs.
    The batches are concatenated and grouped once, so no Python-level work
    is done per edge.

    Parameters
    ----------
    n_rows, n_cols : int
        Grid dimensions.
    mask : NDArrayBool of shape (n_rows * n_cols,)
        Boolean array indicating which polygons exist.
    offsets : NDArrayFloat of shape (n_offsets, 2)
        ``(dr, dc)`` neighbour offsets applied uniformly to every cell.

    Returns
    -------
    Dict[int, List[int]]
    """
    mask_flat = mask.ravel()
    mask2d = mask_flat.reshape(n_rows, n_cols)
    valid = np.flatnonzero(mask_flat)
    grid_to_compact = -np.ones(n_rows * n_cols, dtype=np.intp)
    grid_to_compact[valid] = np.arange(len(valid))

    r_valid, c_valid = np.where(mask2d)

    src_parts: List[NDArrayInt] = []
    dst_parts: List[NDArrayInt] = []

    for dr, dc in offsets:
        nr = r_valid + dr
        nc = c_valid + dc
        in_bounds = (nr >= 0) & (nr < n_rows) & (nc >= 0) & (nc < n_cols)
        src = r_valid[in_bounds] * n_cols + c_valid[in_bounds]
        dst = nr[in_bounds] * n_cols + nc[in_bounds]
        valid_dst = mask_flat[dst]
        src_parts.append(grid_to_compact[src[valid_dst]])
        dst_parts.append(grid_to_compact[dst[valid_dst]])

    return _pairs_to_adj(
        np.concatenate(src_parts), np.concatenate(dst_parts), len(valid)
    )


def rectangular_grid_adjacency_masked(
    n_rows: int, n_cols: int, mask: NDArrayBool
) -> Dict[int, List[int]]:
    """
    Build adjacency dictionary for a rectangular grid of polygons with a mask.

    Note
    ----
    Only polygons where mask == True are included. Indices are compact:
    0 ... n_valid-1 for valid polygons.

    Neighbors are 4-connected (vertical and horizontal only): two rectangles
    are adjacent only when they share a full edge. Diagonal cells, which
    touch at a single corner, are not included since a shared vertex alone
    does not make two tiles adjacent.

    Neighbour lists are sorted in ascending order and duplicate-free.

    Parameters
    ----------
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    mask : NDArrayBool of shape (rows*cols,)
        Boolean array indicating which polygons exist (True).

    Returns
    -------
    adj : Dict[int, list[int]]
        Dictionary mapping each valid polygon's compact index to a list of neighbor
        indices.
    """
    offsets = np.array(
        [(-1, 0), (1, 0), (0, -1), (0, 1)],
        dtype=int,
    )
    return _vectorized_grid_adjacency(n_rows, n_cols, mask, offsets)


def gen_rectangular_tiling(
    surface_to_cover: Surface,
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    alignment_point: Optional[ArrayLike] = None,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Create a grid of hexagons within the given bounding box.

    Parameters
    ----------
    surface_to_cover : Surface
        Surface to cover with the tiling. Only the polygon intersecting this surface
        are kept.
    edge_length: float = 1.0
        Edge length for the base polygon.
    edge_length: float
        Edge length for the base polygon.
        E.g., choosing :py:attr:`PolygonType.RECTANGLE` with `anisotropy_ratio` = 2
        results in rectangles with scale (1.0, 2.0).
    alignment_point : array-like of shape (2,), optional
        ``(x, y)`` world-space coordinate used to shift the tiling
        so that one tile centre coincides with this point.
        When ``None`` (default) no alignment shift is applied.

    Returns
    -------
    Tuple[shapely.MultiPolygon, DefaultDict[int, Set[int]]]
        A tuple containing:
            - A shapely collection of polygons.
            - A dictionary where keys are hexagon indices and values are sets of
              adjacent hexagon indices.

    """
    _validate_inputs(edge_length, anisotropy_ratio)

    v_step = edge_length * anisotropy_ratio  #  Vertical step (height of a hexagon)
    h_step = edge_length  # Horizontal step (width of a hexagon)

    b1 = np.array([h_step, 0.0])
    b2 = np.array([0.0, v_step])
    centers = _lattice_centres(
        surface_to_cover.bounds, b1, b2, _resolve_anchor(alignment_point)
    )

    verts = _reference_tile_vertices(
        PolygonType.RECTANGLE, edge_length, anisotropy_ratio
    )

    # Build the tiles intersecting the surface, with the keep mask over the
    # full lattice.
    kept_polygons, mask = _tiles_from_centres(centers, verts, surface_to_cover)

    # Adjacency of kept polygons
    adjacency_dict = rectangular_grid_adjacency_masked(
        centers.shape[1], centers.shape[2], mask
    )
    # shapely.multipolygons builds the collection in C; the MultiPolygon
    # constructor walks the sequence and runs a Python-level emptiness check
    # on every member.
    return shapely.multipolygons(kept_polygons), adjacency_dict


def hexagonal_grid_adjacency_masked(
    nv: int,
    nh: int,
    mask: NDArrayBool,
) -> Dict[int, List[int]]:
    """
    Build adjacency dictionary for a hexagonal grid of polygons with a mask.

    Two column layouts produce a hexagonal tiling and they do not share the
    same index topology:

    Note
    ----
    Only polygons where mask == True are included. Indices are compact:
    0 ... n_valid-1 for valid polygons.

    Neighbour lists are sorted in ascending order and duplicate-free.

    Parameters
    ----------
    nv : int
        Number of rows in the grid.
    nh : int
        Number of columns in the grid.
    mask : NDArrayBool of shape (nv*nh,)
        Boolean array indicating which polygons exist (True).

    Returns
    -------
    adj : Dict[int, list[int]]
        Dictionary mapping each valid polygon's compact index to a list of neighbor
        indices.
    """
    # Sheared lattice: the six neighbour offsets are the same everywhere.
    offsets = np.array([(0, -1), (0, 1), (-1, 0), (1, 0), (1, 1), (-1, -1)], dtype=int)
    return _vectorized_grid_adjacency(nv, nh, mask, offsets)


def gen_hexagonal_tiling(
    surface_to_cover: Surface,
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    alignment_point: Optional[ArrayLike] = None,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Create a grid of hexagons within the given bounding box.

    Parameters
    ----------
    surface_to_cover : Surface
        Surface to cover with the tiling. Only the polygon intersecting this surface
        are kept.
    edge_length: float = 1.0
        Edge length for the base polygon.
    edge_length: float
        Edge length for the base polygon.
        E.g., choosing :py:attr:`PolygonType.RECTANGLE` with `anisotropy_ratio` = 2
        results in rectangles with scale (1.0, 2.0).
    alignment_point : array-like of shape (2,), optional
        ``(x, y)`` world-space coordinate used to shift the tiling
        so that one tile centre coincides with this point.
        When ``None`` (default) no alignment shift is applied.

    Returns
    -------
    Tuple[shapely.MultiPolygon, DefaultDict[int, Set[int]]]
        A tuple containing:
            - A shapely collection of polygons.
            - A dictionary where keys are hexagon indices and values are sets
              of adjacent hexagon indices.

    """
    _validate_inputs(edge_length, anisotropy_ratio)

    # Calculate the vertical and horizontal step distances between centers of hexagons
    v_step = (
        math.sqrt(3) * edge_length * anisotropy_ratio
    )  #  Vertical step (height of a hexagon)
    h_step = 1.5 * edge_length  # Horizontal step (width of a hexagon)

    # The column vector carries the half-step stagger, so the lattice is
    # sheared rather than parity-staggered and every cell has the same six
    # neighbour offsets.
    b1 = np.array([h_step, -v_step / 2.0])
    b2 = np.array([0.0, v_step])
    centers = _lattice_centres(
        surface_to_cover.bounds, b1, b2, _resolve_anchor(alignment_point)
    )

    # vertices for one polygon
    verts = _reference_tile_vertices(PolygonType.HEXAGON, edge_length, anisotropy_ratio)

    # Build the tiles intersecting the surface, with the keep mask over the
    # full lattice.
    kept_polygons, mask = _tiles_from_centres(centers, verts, surface_to_cover)

    # Adjacency of kept polygons
    adjacency_dict = hexagonal_grid_adjacency_masked(
        centers.shape[1],
        centers.shape[2],
        mask,
    )

    # Built in C rather than through the per-member MultiPolygon constructor.
    return shapely.multipolygons(kept_polygons), adjacency_dict


def extract_tiling_centers(
    polygons: Union[shapely.MultiPolygon, Iterable[shapely.Polygon]],
) -> NDArrayFloat:
    """
    Extract the centers of each tile (polygon).

    Parameters
    ----------
    polygons : Union[shapely.MultiPolygon, Iterable[shapely.Polygon]]
        Polygons for which centers are extracted.

    Returns
    -------
    NDArrayFloat
        - 2D Array of vertices coordinates with shape (n, 2), n being the number of
          polygons.
    """
    if isinstance(polygons, shapely.MultiPolygon):
        geom_array = np.array(polygons.geoms)
    else:
        geom_array = np.asarray(list(polygons))
    centroids = shapely.centroid(geom_array)
    return shapely.get_coordinates(centroids)


def extract_tiling_vertices(
    polygons: Union[shapely.MultiPolygon, Iterable[shapely.Polygon]],
    n_decimals: int = 2,
) -> Tuple[NDArrayFloat, Dict[int, List[int]], NDArrayInt, Dict[int, List[int]]]:
    """
    Extract the vertices of all polygons, deduplicating shared vertices.

    Vertices shared between adjacent polygons (identical coordinates up to
    ``n_decimals`` decimal places) are merged into a single entry.  The
    closing repeat of each exterior ring is dropped before deduplication.
    Both adjacency mappings are built from the same
    ``(cluster_indices, poly_indices)`` arrays so ``shapely.get_rings`` is
    called only once.

    Deduplication is performed on a single integer key per vertex, obtained
    by scaling the rounded coordinates and packing ``(x, y)`` into one
    ``int64``.  Sorting integers is markedly cheaper than sorting the void
    dtype of a structured array, and the resulting order is the same
    lexicographic ``(x, y)`` order.

    Parameters
    ----------
    polygons : Union[shapely.MultiPolygon, Iterable[shapely.Polygon]]
        Polygons whose vertices are to be extracted.
    n_decimals : int, optional
        Number of decimal places used when rounding coordinates before
        hashing for duplicate removal.  By default ``2``.

    Returns
    -------
    unique_coords : NDArrayFloat, shape (n_verts, 2)
        Coordinates of the ``n_verts`` deduplicated vertices, in
        lexicographic ``(x, y)`` order.
    vert_to_polys : Dict[int, List[int]]
        Deduplicated vertex id → sorted list of polygon ids that share it.
        Interior vertices shared by several polygons have lists with more
        than one entry; boundary vertices have exactly one.
    cluster_indices : NDArrayInt, shape (n_input_verts,)
        For every input vertex (closing repeats removed, in input order),
        its deduplicated vertex id.  Satisfies
        ``unique_coords[cluster_indices] ≈ original_coords``.
    poly_to_verts : Dict[int, List[int]]
        Polygon id → deduplicated vertex ids in ring order.
    """
    if isinstance(polygons, shapely.MultiPolygon):
        geom_array = np.array(polygons.geoms)
    else:
        geom_array = np.asarray(list(polygons))

    rings = shapely.get_rings(geom_array)  # single call
    n_per_ring = shapely.get_num_coordinates(rings)
    all_coords, poly_indices = shapely.get_coordinates(geom_array, return_index=True)

    # drop closing repeat (last vertex of every ring)
    drop = np.zeros(len(all_coords), dtype=bool)
    drop[np.cumsum(n_per_ring) - 1] = True
    coords = all_coords[~drop]
    poly_indices = poly_indices[~drop]

    # quantise the coordinates on the rounding grid, shifted to non-negative
    # values so the packing below is monotonic in x then y
    key = np.rint(coords * 10.0**n_decimals).astype(np.int64)
    key -= key.min(axis=0)

    # pack (x, y) into a single integer and deduplicate
    stride = int(key[:, 1].max()) + 1
    code = key[:, 0] * stride + key[:, 1]
    _, first, inverse = np.unique(code, return_index=True, return_inverse=True)
    cluster_indices = inverse.ravel().astype(np.int64)
    unique_coords = np.round(coords[first], decimals=n_decimals)

    # vertex → polygons, grouped in a single pass
    vert_to_polys = _pairs_to_adj(cluster_indices, poly_indices, len(unique_coords))

    # polygon → vertices, kept in ring order by slicing the input-ordered
    # cluster indices at the polygon boundaries
    n_per_poly = np.bincount(poly_indices, minlength=len(geom_array))
    poly_to_verts = {
        i: group.tolist()
        for i, group in enumerate(np.split(cluster_indices, np.cumsum(n_per_poly)[:-1]))
    }

    return (
        unique_coords,
        vert_to_polys,
        cluster_indices,
        poly_to_verts,
    )


def adjacency_by_shared_vertices(
    polygons: List[shapely.Polygon],
) -> Dict[int, List[int]]:
    """
    Compute adjacency dictionary based on polygons sharing vertices.

    All ``(polygon, polygon)`` pairs sharing a vertex are expanded at once
    with a ragged-index gather over the vertex groups, so the cost per
    shared vertex stays in NumPy rather than in a Python set update.

    Note
    ----
    This is vertex-based adjacency: two polygons that touch only at a
    single corner (no shared edge) are still considered neighbours here.
    For strict edge-sharing adjacency use the structured
    ``*_grid_adjacency_masked`` builders instead.

    Parameters
    ----------
    polygons : list of shapely.Polygon
        List of polygons (triangles, hexagons, etc.)

    Returns
    -------
    adj : dict[int, list[int]]
        Dictionary mapping polygon index to a sorted list of neighboring polygon
        indices (sharing at least one vertex).  Polygons without any neighbour
        map to an empty list.
    """
    _, _, cluster_indices, poly_to_verts = extract_tiling_vertices(polygons)
    n_polys = len(poly_to_verts)

    # polygon id of every input vertex, aligned with cluster_indices
    n_per_poly = np.fromiter(
        (len(v) for v in poly_to_verts.values()), dtype=np.int64, count=n_polys
    )
    poly_ids = np.repeat(
        np.fromiter(poly_to_verts.keys(), dtype=np.int64, count=n_polys), n_per_poly
    )

    # sort the (vertex, polygon) pairs so that polygons sharing a vertex form
    # contiguous runs
    order = np.argsort(cluster_indices, kind="stable")
    sorted_clusters = cluster_indices[order]
    sorted_polys = poly_ids[order]

    # start and size of the run each element belongs to
    counts = np.bincount(sorted_clusters)
    starts = np.cumsum(counts) - counts
    run_size = counts[sorted_clusters]
    run_start = starts[sorted_clusters]

    # cartesian product within every run: each element is paired with the
    # whole run it belongs to
    src = np.repeat(sorted_polys, run_size)
    dst = sorted_polys[np.repeat(run_start, run_size) + _ragged_arange(run_size)]

    # drop the self-pairs introduced by the cartesian product
    keep = src != dst
    return _pairs_to_adj(src[keep], dst[keep], n_polys)


def triangular_grid_adjacency_masked(
    nj: int, ni: int, mask: NDArrayBool
) -> Dict[int, List[int]]:
    """
    Build adjacency dictionary for a triangular grid produced by
    :func:`gen_triangular_tiling`.

    Each lattice cell ``(j, i)`` produces two triangles stored at flat
    indices ``(j * ni + i) * 2 + k`` where *k* is 0 (lower-right) or
    1 (upper-left).  Two triangles are neighbours only when they share a
    full edge (i.e. two vertices), never when they merely touch at a
    single lattice-node vertex.  Every triangle has exactly 3 edges, so
    it has at most 3 edge-sharing neighbours.

    Neighbour lists are sorted in ascending order and duplicate-free.

    Parameters
    ----------
    nj : int
        Number of lattice rows.
    ni : int
        Number of lattice columns.
    mask : NDArrayBool of shape (nj * ni * 2,)
        Boolean array indicating which triangles exist.

    Returns
    -------
    Dict[int, List[int]]
        Adjacency dictionary with compact (0-based) indices.
    """
    mask_flat = mask.ravel()
    n_total = nj * ni * 2
    valid = np.flatnonzero(mask_flat)
    grid_to_compact = -np.ones(n_total, dtype=np.intp)
    grid_to_compact[valid] = np.arange(len(valid))

    # Flat indices and decomposition into (j, i, k)
    flat_valid = valid
    k_valid = flat_valid % 2
    cell_valid = flat_valid // 2
    j_valid = cell_valid // ni
    i_valid = cell_valid % ni

    # Neighbour offsets as (dj, di, dk) relative to triangle (j, i, k).
    # Cell (j, i) has vertices p = origin + i*a + j*b, p+a, p+b and p+a+b.
    # The k=0 triangle is (p, p+a, p+b) and the k=1 triangle is
    # (p+a, p+a+b, p+b), so the two share the diagonal edge (p+a, p+b).
    #
    # Each triangle has exactly 3 edges, hence at most 3 edge-sharing
    # neighbours (vertex-only touches are intentionally excluded):
    #
    #   k=0 triangle (p, p+a, p+b):
    #     edge (p+a, p+b) -> diagonal, shared with (j,   i,   k=1)
    #     edge (p,   p+a) -> shared with              (j-1, i,   k=1)
    #     edge (p,   p+b) -> shared with              (j,   i-1, k=1)
    #
    #   k=1 triangle (p+a, p+a+b, p+b):
    #     edge (p+a, p+b)   -> diagonal, shared with (j,   i,   k=0)
    #     edge (p+a, p+a+b) -> shared with            (j,   i+1, k=0)
    #     edge (p+a+b, p+b) -> shared with            (j+1, i,   k=0)
    offsets_k0 = [
        (0, 0, 1),
        (-1, 0, 1),
        (0, -1, 1),
    ]
    offsets_k1 = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 0),
    ]

    src_parts: List[NDArrayInt] = []
    dst_parts: List[NDArrayInt] = []

    for k_src, offsets in [(0, offsets_k0), (1, offsets_k1)]:
        sel = k_valid == k_src
        j_sel = j_valid[sel]
        i_sel = i_valid[sel]
        flat_sel = flat_valid[sel]

        for dj, di, dk in offsets:
            nj_idx = j_sel + dj
            ni_idx = i_sel + di
            in_bounds = (nj_idx >= 0) & (nj_idx < nj) & (ni_idx >= 0) & (ni_idx < ni)
            dst_flat = (nj_idx[in_bounds] * ni + ni_idx[in_bounds]) * 2 + dk
            src_flat = flat_sel[in_bounds]
            valid_dst = mask_flat[dst_flat]
            src_parts.append(grid_to_compact[src_flat[valid_dst]])
            dst_parts.append(grid_to_compact[dst_flat[valid_dst]])

    # Group every (source, destination) pair in a single pass; the encoding
    # used there already yields sorted, duplicate-free neighbour lists.
    return _pairs_to_adj(
        np.concatenate(src_parts), np.concatenate(dst_parts), len(valid)
    )


def intersects_mask(
    polygons: NDArrayFloat,
    surface: Surface,
) -> NDArrayInt:
    """Return a boolean mask of polygons intersecting *surface*.

    When *surface* is a :class:`Disk`, membership is decided analytically
    against the exact circle and no geometric predicate is evaluated.

    Otherwise the surface is prepared first: this builds an index over its
    edges once and reuses it for every tile, instead of re-walking the
    boundary at each predicate evaluation.

    For simple surfaces the vectorized ``shapely.intersects`` is then used
    directly.  For complex multi-part surfaces an STRtree spatial index
    is built to avoid O(n) pairwise checks, and the query is issued with
    the individual parts rather than the multipolygon, whose overall
    envelope carries little information when the parts are far apart.

    Parameters
    ----------
    polygons : NDArrayFloat of shapely.Polygon, shape (n_polygons,)
        Candidate tiles. When *surface* is a :class:`Disk` they must be
        convex, which every tile produced by this module is.
    surface : Surface
        Surface the polygons must intersect.

    Returns
    -------
    NDArrayFloat of bool, shape (n_polygons,)
        Mask of the polygons intersecting *surface*.
    """
    if len(polygons) == 0:
        return np.zeros(0, dtype=bool)

    if isinstance(surface, Disk):
        coords, index = shapely.get_coordinates(polygons, return_index=True)
        n_vertices = np.bincount(index)[0] - 1
        rings = coords.reshape(len(polygons), n_vertices + 1, 2)[:, :-1, :]
        return _disk_mask_from_rings(rings, surface)

    if isinstance(surface, shapely.MultiPolygon) and len(surface.geoms) > 8:
        parts = np.asarray(surface.geoms)
        shapely.prepare(parts)
        tree = shapely.STRtree(polygons)
        hit = tree.query(parts, predicate="intersects")[1]
        mask = np.zeros(len(polygons), dtype=bool)
        mask[hit] = True
        return mask

    shapely.prepare(surface)
    return shapely.intersects(polygons, surface)


def gen_triangular_tiling(
    surface_to_cover: Surface,
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    alignment_point: Optional[ArrayLike] = None,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Generate a triangular tiling covering a given surface.

    The tiling is generated from a triangular vertex lattice.  Each elementary
    parallelogram of the lattice is split into two triangles, which guarantees
    exact edge sharing and avoids the centre/parity shift issues that can occur
    when triangles are generated independently from alternating centres.

    The triangular lattice is defined by two primitive vectors:

    ``a = (edge_length, 0)``

    ``b = (edge_length / 2, sqrt(3) / 2 * edge_length * anisotropy_ratio)``

    For each lattice node ``p``, two triangles are created:

    - lower/right triangle: ``(p, p + a, p + b)``
    - upper/left triangle: ``(p + a, p + a + b, p + b)``

    If ``alignment_point`` is provided, the lattice is shifted so that the
    centroid of one triangle lies exactly on this point.  More precisely, the
    centroid of the triangle ``(p, p + a, p + b)`` is aligned with
    ``alignment_point``.

    Parameters
    ----------
    surface_to_cover : Surface
        Surface to cover with triangular tiles. Only triangles intersecting this
        surface are kept in the returned tiling.
    edge_length : float
        Edge length of the triangles before anisotropic vertical scaling.
    anisotropy_ratio : float, optional
        Vertical anisotropy ratio applied to the triangle height, by default 1.0.
        A value larger than 1 stretches the triangular lattice vertically.
    alignment_point : Optional[ArrayLike], optional
        World-space coordinate ``(x, y)`` used to align the tiling. If provided,
        one triangle centroid is guaranteed to coincide exactly with this point.
        If ``None``, the lower-left corner of the surface bounding box is used
        as the default aligned triangle centroid.

    Returns
    -------
    Tuple[shapely.MultiPolygon, Dict[int, List[int]]]
        A tuple containing:

        - A :class:`shapely.MultiPolygon` with all triangles intersecting
          ``surface_to_cover``.
        - An adjacency dictionary mapping each kept triangle index to the list
          of neighbouring kept triangle indices sharing a full edge.

    Raises
    ------
    ValueError
        If ``edge_length`` is not strictly positive.
    ValueError
        If ``anisotropy_ratio`` is not strictly positive.

    Notes
    -----
    This implementation intentionally uses a vertex lattice instead of a centre
    lattice.  This is usually the most stable construction for triangular
    tilings because all shared edges and vertices are generated from the same
    coordinates.
    """
    _validate_inputs(edge_length, anisotropy_ratio)

    # Store the horizontal edge length using a short geometric name.
    h = float(edge_length)

    # Compute the anisotropic height of one triangle.
    v = h * SQRT3 / 2.0 * float(anisotropy_ratio)

    # Define the first primitive vector of the triangular vertex lattice.
    a = np.array([h, 0.0], dtype=float)

    # Define the second primitive vector of the triangular vertex lattice.
    b = np.array([h / 2.0, v], dtype=float)

    # Select the triangle centroid that should be aligned with the lattice.
    aligned_centroid = _resolve_anchor(alignment_point)

    # The centroid of triangle (p, p + a, p + b) is p + (a + b) / 3.
    # Therefore, choose the lattice vertex origin p so that this centroid
    # coincides exactly with aligned_centroid.
    vertex_anchor = aligned_centroid - (a + b) / 3.0

    # Generate lattice nodes covering the surface bounding box.
    # Here these are not triangle centres; they are vertex lattice origins.
    lattice_nodes = _lattice_centres(
        bounds=surface_to_cover.bounds,
        b1=a,
        b2=b,
        anchor=vertex_anchor,
        margin=3,
    )

    # Convert lattice node array from shape (2, nj, ni) to shape (nj, ni, 2).
    nodes = np.moveaxis(lattice_nodes, 0, -1)

    # Get the number of lattice rows.
    nj = nodes.shape[0]

    # Get the number of lattice columns.
    ni = nodes.shape[1]

    # Compute the first vertex p of each elementary lattice parallelogram.
    p00 = nodes

    # Compute the second vertex p + a of each elementary lattice parallelogram.
    p10 = nodes + a

    # Compute the third vertex p + b of each elementary lattice parallelogram.
    p01 = nodes + b

    # Compute the fourth vertex p + a + b of each elementary lattice parallelogram.
    p11 = nodes + a + b

    # Build the first triangle of each lattice parallelogram.
    # This triangle has centroid p + (a + b) / 3.
    tri_a_coords = np.stack([p00, p10, p01], axis=2)

    # Build the second triangle of each lattice parallelogram.
    # This triangle exactly shares the diagonal edge (p + a, p + b).
    tri_b_coords = np.stack([p10, p11, p01], axis=2)

    # Interleave both triangles per cell and flatten to a coordinate array.
    # Layout: flat index = (j * ni + i) * 2 + k, with k=0 for tri_a, k=1 for tri_b.
    tri_coords = np.stack([tri_a_coords, tri_b_coords], axis=2).reshape(-1, 3, 2)

    # Discard the triangles whose bounding box misses the surface bounding box
    # before building any geometry, since polygon creation and the intersection
    # predicate dominate the cost.
    x_min, y_min, x_max, y_max = surface_to_cover.bounds
    lo = tri_coords.min(axis=1)
    hi = tri_coords.max(axis=1)
    candidates = (
        (hi[:, 0] >= x_min)
        & (lo[:, 0] <= x_max)
        & (hi[:, 1] >= y_min)
        & (lo[:, 1] <= y_max)
    )

    mask = np.zeros(candidates.size, dtype=bool)
    if isinstance(surface_to_cover, Disk):
        # Analytic membership: build only the triangles that are kept.
        kept = _disk_mask_from_rings(tri_coords[candidates], surface_to_cover)
        mask[candidates] = kept
        kept_polygons = shapely.polygons(tri_coords[candidates][kept])
    else:
        # Convert the candidate triangle coordinate arrays into Shapely
        # polygons, then keep those intersecting the requested surface and
        # scatter the result back onto the full lattice for the adjacency
        # builder.
        polygons = shapely.polygons(tri_coords[candidates])
        mask[candidates] = intersects_mask(polygons, surface_to_cover)
        kept_polygons = polygons[mask[candidates]]

    # Compute adjacency from the structured grid topology.
    adjacency_dict = triangular_grid_adjacency_masked(nj, ni, mask)

    # Return the final tiling and its adjacency dictionary.
    # Built in C rather than through the per-member MultiPolygon constructor.
    return shapely.multipolygons(kept_polygons), adjacency_dict


def gen_polygonal_tiling(
    surface_to_cover: Surface,
    poly_type: PolygonType,
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    rot_deg: float = 0.0,
    alignment_point: Optional[ArrayLike] = None,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Cover the given surface with tiles (polygons) of the desired type.

    The tiling is generated in an axis-aligned (non-rotated) frame, then
    rotated by *rot_deg* degrees around the centroid of *surface_to_cover*
    so that the final tile orientations match the requested rotation.

    Parameters
    ----------
    surface_to_cover : Surface
        Surface to cover with the tiling.  Only tiles whose footprint
        intersects this surface are kept in the output.
    poly_type : PolygonType
        Type of tile geometry.  See :class:`PolygonType` for the
        available options (``HEXAGON``, ``TRIANGLE``, ``RECTANGLE``).
    edge_length : float
        Primary edge length for the base polygon in metres.
        For anisotropic tiles the secondary dimension is derived as
        ``edge_length * anisotropy_ratio``.
    anisotropy_ratio : float, optional
        Ratio of the secondary to the primary tile dimension.
        Must be >= 1.  For example, choosing :attr:`PolygonType.RECTANGLE`
        with ``anisotropy_ratio = 2`` produces rectangles with aspect
        ratio 1 : 2.  By default ``1.0`` (isotropic).
    rot_deg : float, optional
        Counter-clockwise rotation angle in degrees applied to the
        entire tiling around the centroid of *surface_to_cover*.
        By default ``0.0`` (no rotation).
    alignment_point : array-like of shape (2,), optional
        ``(x, y)`` world-space coordinate used to shift the tiling
        so that one tile centre coincides with this point.
        When ``None`` (default) no alignment shift is applied.

    Returns
    -------
    tiling : shapely.MultiPolygon
        Collection of tile polygons covering *surface_to_cover*.
    adjacency : dict of {int: list of int}
        Adjacency map where keys are tile indices (0-based, matching the
        order of geometries in *tiling*) and values are lists of
        neighbouring tile indices.

    Raises
    ------
    ValueError
        If *poly_type* is not a recognised :class:`PolygonType` member.

    Notes
    -----
    Internally the function un-rotates *surface_to_cover* by ``-rot_deg``,
    generates the axis-aligned tiling, then re-rotates the result by
    ``+rot_deg`` so that the output tiles are in world-space coordinates.
    """
    origin = surface_to_cover.centroid
    if isinstance(surface_to_cover, Disk):
        # A disk is invariant under rotation about its own centre, which is
        # exactly the origin used here, so the round-trip is skipped.
        rot_surface_to_cover: Surface = surface_to_cover
    else:
        rot_surface_to_cover = shapely.affinity.rotate(
            surface_to_cover,
            angle=-rot_deg,
            use_radians=False,
            origin=origin,
        )

    rot_alignment_point = alignment_point
    if alignment_point is not None and rot_deg != 0.0:
        rot_alignment_point = np.array(
            shapely.affinity.rotate(
                shapely.Point(alignment_point),
                angle=-rot_deg,
                use_radians=False,
                origin=origin,
            ).xy
        ).ravel()

    if poly_type == PolygonType.HEXAGON:
        _grid, _adj = gen_hexagonal_tiling(
            rot_surface_to_cover,
            edge_length=edge_length,
            anisotropy_ratio=anisotropy_ratio,
            alignment_point=rot_alignment_point,
        )
    elif poly_type == PolygonType.TRIANGLE:
        _grid, _adj = gen_triangular_tiling(
            rot_surface_to_cover,
            edge_length=edge_length,
            anisotropy_ratio=anisotropy_ratio,
            alignment_point=rot_alignment_point,
        )
    elif poly_type == PolygonType.RECTANGLE:
        _grid, _adj = gen_rectangular_tiling(
            rot_surface_to_cover,
            edge_length=edge_length,
            anisotropy_ratio=anisotropy_ratio,
            alignment_point=rot_alignment_point,
        )
    else:
        raise ValueError(PolygonType(poly_type))

    if rot_deg == 0.0:
        return _grid, _adj

    # Batch-rotate all polygon coordinates via shapely.get_coordinates /
    # shapely.set_coordinates instead of per-polygon Python loops.
    origin_xy = np.array(origin.coords[0])
    theta = np.radians(rot_deg)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array([[c, -s], [s, c]])

    geom_array = np.array(list(_grid.geoms))
    coords = shapely.get_coordinates(geom_array)
    rotated_coords = (coords - origin_xy) @ R.T + origin_xy
    rotated_geoms = shapely.set_coordinates(geom_array.copy(), rotated_coords)
    return shapely.multipolygons(rotated_geoms), _adj
