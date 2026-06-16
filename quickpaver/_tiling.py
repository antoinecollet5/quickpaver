# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Tiling with triangles, squares and hexagons with support for anisotropy and
grid alignment with a given point."""

import math
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np
import shapely
import shapely.affinity
from numpy.typing import ArrayLike

from quickpaver._types import NDArrayFloat, NDArrayInt, StrEnum

SQRT3 = math.sqrt(3)


def _lattice_centres(
    bounds: Tuple[float, float, float, float],
    b1: np.ndarray,
    b2: np.ndarray,
    anchor: np.ndarray,
    margin: int = 2,
) -> np.ndarray:
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
    b1, b2 : np.ndarray, shape (2,)
        Primitive lattice basis vectors (column and row directions,
        including any stagger and anisotropy).
    anchor : np.ndarray, shape (2,)
        A point that must be a lattice node (the alignment point, or a
        default such as the bbox corner).
    margin : int
        Extra rings of cells added beyond the bbox to guarantee full
        coverage.  Default 2.

    Returns
    -------
    np.ndarray, shape (2, nj, ni)
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


def rectangular_grid_adjacency_masked(
    n_rows: int, n_cols: int, mask: np.ndarray
) -> Dict[int, List[int]]:
    """
    Build adjacency dictionary for a rectangular grid of polygons with a mask, including
    diagonal neighbors.

    Note
    ----
    Only polygons where mask == True are included. Indices are compact:
    0 ... n_valid-1 for valid polygons.

    Neighbors are 8-connected (vertical, horizontal, and diagonal).

    Parameters
    ----------
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    mask : np.ndarray of shape (rows*cols,)
        Boolean array indicating which polygons exist (True).

    Returns
    -------
    adj : Dict[int, list[int]]
        Dictionary mapping each valid polygon's compact index to a list of neighbor
        indices.
    """

    # reshape mask to 2D
    mask2d = mask.reshape(n_rows, n_cols)

    # mapping from full grid index to compact index
    valid = np.flatnonzero(mask)
    grid_to_compact = -np.ones(n_rows * n_cols, dtype=int)
    grid_to_compact[valid] = np.arange(len(valid))

    # directions for 8-connectivity
    directions = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

    adj = defaultdict(list)

    # iterate only over valid cells
    r_valid, c_valid = np.where(mask2d)

    for r, c in zip(r_valid, c_valid):
        i_compact = grid_to_compact[r * n_cols + c]

        for dr, dc in directions:
            nr, nc = r + dr, c + dc

            if 0 <= nr < n_rows and 0 <= nc < n_cols and mask2d[nr, nc]:
                j_compact = grid_to_compact[nr * n_cols + nc]
                adj[i_compact.item()].append(j_compact.item())

    return dict(adj)


def _get_non_aligned_rect_centers(
    x_min: float, x_max: float, y_min: float, y_max: float, h_step: float, v_step: float
) -> NDArrayFloat:

    # number of columns of polygons
    nh: int = math.ceil((x_max - x_min) / h_step)
    # number of rows of polygons
    nv = math.ceil((y_max - y_min) / v_step)

    # compute the difference between the x coverage of the polygons and the required
    #  x, y range
    x_delta = nh * h_step - (x_max - x_min)
    y_delta = nv * v_step - (y_max - y_min)

    # compute the grid start
    h_start = x_min - x_delta / 2.0
    v_start = y_min - y_delta / 2.0

    # compute the polygon centers
    return np.array(
        np.meshgrid(
            np.linspace(h_start + h_step / 2.0, h_start + (nh - 0.5) * h_step, nh),
            np.linspace(v_start + v_step / 2.0, v_start + (nv - 0.5) * v_step, nv),
        )
    )


def gen_rectangular_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    alignment_point: Optional[ArrayLike] = None,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Create a grid of hexagons within the given bounding box.

    Parameters
    ----------
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon]
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

    # extract the coordinates of the bounding box
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    v_step = edge_length * anisotropy_ratio  #  Vertical step (height of a hexagon)
    h_step = edge_length  # Horizontal step (width of a hexagon)

    if alignment_point is None:
        centers = _get_non_aligned_rect_centers(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            h_step=h_step,
            v_step=v_step,
        )
    else:
        b1 = np.array([h_step, 0.0])
        b2 = np.array([0.0, v_step])
        anchor = (
            np.asarray(alignment_point, dtype=float).ravel()[:2]
            if alignment_point is not None
            else np.array([x_min, y_min])
        )
        centers = _lattice_centres(surface_to_cover.bounds, b1, b2, anchor)

    verts = np.array(
        gen_polygon(
            PolygonType.RECTANGLE, edge_length, anisotropy_ratio
        ).exterior.coords
    )
    # Broadcast vertices and create polygons (flatten)
    polygons = shapely.polygons(
        np.moveaxis(centers, 0, -1)[..., None, :] + verts[None, None, :, :]
    ).ravel()

    # Mask intersecting polygons (to keep)
    mask = shapely.intersects(polygons, surface_to_cover)
    # Adjacency of kept polygons
    adjacency_dict = rectangular_grid_adjacency_masked(
        centers.shape[1], centers.shape[2], mask
    )
    return shapely.MultiPolygon(polygons[mask]), adjacency_dict


def hexagonal_grid_adjacency_masked(
    nv: int, nh: int, mask: np.ndarray
) -> Dict[int, List[int]]:
    """
    Build adjacency dictionary for a rectangular grid of polygons with a mask,
    including diagonal neighbors.

    Note
    ----
    Only polygons where mask == True are included. Indices are compact:
    0 ... n_valid-1 for valid polygons.

    Neighbors are 8-connected (vertical, horizontal, and diagonal).

    Parameters
    ----------
    n_rows : int
        Number of rows in the grid.
    n_cols : int
        Number of columns in the grid.
    mask : np.ndarray of shape (rows*cols,)
        Boolean array indicating which polygons exist (True).

    Returns
    -------
    adj : Dict[int, list[int]]
        Dictionary mapping each valid polygon's compact index to a list of neighbor
        indices.
    """

    # reshape mask to 2D
    mask2d = mask.reshape(nv, nh)

    # mapping from full grid index to compact index
    valid = np.flatnonzero(mask)
    grid_to_compact = -np.ones(nv * nh, dtype=int)
    grid_to_compact[valid] = np.arange(len(valid))

    adj = defaultdict(list)

    # shifted columns are 0,2,4,...  (c % 2 == 0)
    # neighbor offsets for "down-shifted columns"
    # remove (-1, -1) and (-1, 1)
    shifted_offsets = [(-1, 0), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    # remove (1, 1) and (1, -1)
    normal_offsets = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, 0)]

    r_valid, c_valid = np.where(mask2d)

    for r, c in zip(r_valid, c_valid):
        i_compact = grid_to_compact[r * nh + c]
        offsets = shifted_offsets if c % 2 == 0 else normal_offsets

        for dr, dc in offsets:
            nr, nc = r + dr, c + dc
            if 0 <= nr < nv and 0 <= nc < nh and mask2d[nr, nc]:
                j_compact = grid_to_compact[nr * nh + nc]
                adj[i_compact.item()].append(j_compact.item())

    return dict(adj)


def _get_non_aligned_hex_centers(
    x_min: float, x_max: float, y_min: float, y_max: float, h_step: float, v_step: float
) -> NDArrayFloat:

    # number of columns of polygons
    nh: int = math.ceil((x_max - x_min) / h_step)
    # number of rows of polygons
    nv = math.ceil((y_max - y_min) / v_step) + 1

    # compute the difference between the x coverage of the polygons and the required
    # x, y range
    x_delta = nh * h_step - (x_max - x_min)
    y_delta = nv * v_step - (y_max - y_min)

    # compute the grid start
    h_start = x_min - x_delta / 2.0
    v_start = y_min - y_delta / 2.0

    # compute the polygon centers
    centers = np.array(
        np.meshgrid(
            np.linspace(h_start + h_step / 2.0, h_start + (nh - 0.5) * h_step, nh),
            np.linspace(v_start + v_step / 2.0, v_start + (nv - 0.5) * v_step, nv),
        )
    )

    # shift half of the columns down
    centers[1, :, ::2] += v_step / 2.0

    return centers


def gen_hexagonal_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    alignment_point: Optional[ArrayLike] = None,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Create a grid of hexagons within the given bounding box.

    Parameters
    ----------
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon]
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

    # extract the coordinates of the bounding box
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    # Calculate the vertical and horizontal step distances between centers of hexagons
    v_step = (
        math.sqrt(3) * edge_length * anisotropy_ratio
    )  #  Vertical step (height of a hexagon)
    h_step = 1.5 * edge_length  # Horizontal step (width of a hexagon)

    if alignment_point is None:
        centers = _get_non_aligned_hex_centers(
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            h_step=h_step,
            v_step=v_step,
        )
    else:
        # even columns shifted up by v_step/2 → b1 carries the -v_step/2 stagger
        b1 = np.array([h_step, -v_step / 2.0])
        b2 = np.array([0.0, v_step])

        anchor = (
            np.asarray(alignment_point, dtype=float).ravel()[:2]
            if alignment_point is not None
            else np.array([x_min, y_min])
        )
        centers = _lattice_centres(surface_to_cover.bounds, b1, b2, anchor)

    # vertices for one polygon
    verts = np.array(
        gen_polygon(PolygonType.HEXAGON, edge_length, anisotropy_ratio).exterior.coords
    )

    # Broadcast vertices and create polygons (flatten)
    polygons = shapely.polygons(
        np.moveaxis(centers, 0, -1)[..., None, :] + verts[None, None, :, :]
    ).ravel()

    # Mask intersecting polygons (to keep)
    mask = shapely.intersects(polygons, surface_to_cover)

    # Adjacency of kept polygons
    adjacency_dict = hexagonal_grid_adjacency_masked(
        centers.shape[1], centers.shape[2], mask
    )

    return shapely.MultiPolygon(polygons[mask]), adjacency_dict


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
        _polygons = polygons.geoms
    else:
        _polygons = polygons
    return np.array([geom.centroid.xy for geom in _polygons])[:, :, 0]


def extract_tiling_vertices(
    polygons: Union[shapely.MultiPolygon, Iterable[shapely.Polygon]],
    n_decimals: int = 2,
) -> Tuple[NDArrayFloat, Dict[int, List[int]], NDArrayInt]:
    """
    Extract the vertices of all polygons (without duplicates).

    Parameters
    ----------
    polygons : Union[shapely.MultiPolygon, Iterable[shapely.Polygon]]
        Polygons for which vertices are extracted.
    n_decimals : int, optional
        Number of decimals to use for the duplicate removal (it relies on hashing),
        by default 2.

    Returns
    -------
    Tuple[NDArrayFloat, Dict[int, List[int]], NDArrayInt]
        - 2D Array of vertices coordinates with shape (n, 2), n being the number of
          vertices extracted.
        - Dict with vertice id ad key and list of associated polygon id as values.
          This is because duplicated vertices are merged.
    """
    if isinstance(polygons, shapely.MultiPolygon):
        _polygons = polygons.geoms
    else:
        _polygons = polygons

    # Convert polygons to arrays of vertices (rounded to avoid floating point issues)
    verts = [
        np.round(np.array(p.exterior.coords[:-1]), decimals=n_decimals)
        for p in _polygons
    ]  # drop repeated last point

    # Build a dict: vertex tuple -> list of polygon indices
    vert_to_polys = defaultdict(list)

    # number of vertices
    nv = 0
    for i, v in enumerate(verts):
        for x, y in v:
            vert_to_polys[(x, y)].append(i)
            # update the number of vertices
            nv += 1

    # Cluster the vertices
    cluster_indices = np.zeros(nv, dtype=np.int64)
    _ids = {k: i for i, k in enumerate(vert_to_polys.keys())}
    # Iterate the points
    nv = 0
    for i, v in enumerate(verts):
        for x, y in v:
            cluster_indices[nv] = _ids[(x, y)]
            nv += 1

    return (
        np.array(list(vert_to_polys.keys())),
        {i: polys for i, polys in enumerate(vert_to_polys.values())},
        cluster_indices,
    )


def adjacency_by_shared_vertices(
    polygons: List[shapely.Polygon],
) -> Dict[int, List[int]]:
    """
    Compute adjacency dictionary based on polygons sharing vertices.

    Parameters
    ----------
    polygons : list of shapely.Polygon
        List of polygons (triangles, hexagons, etc.)

    Returns
    -------
    adj : dict[int, list[int]]
        Dictionary mapping polygon index to a list of neighboring polygon indices
        (sharing at least one vertex).
    """
    verts, vert_to_polys, _ = extract_tiling_vertices(polygons)

    # Build adjacency dict
    adj = defaultdict(set)
    for shared_polys in vert_to_polys.values():
        for i in shared_polys:
            for j in shared_polys:
                if i != j:
                    adj[i].add(j)

    # Convert sets to sorted lists
    return {i: sorted(list(neigh)) for i, neigh in adj.items()}


def gen_triangular_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
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
    surface_to_cover : Union[shapely.Polygon, shapely.MultiPolygon]
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
          of neighbouring kept triangle indices sharing at least one rounded
          vertex.

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

    # Extract the bounding box of the surface to cover.
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    # Store the horizontal edge length using a short geometric name.
    h = float(edge_length)

    # Compute the anisotropic height of one triangle.
    v = h * SQRT3 / 2.0 * float(anisotropy_ratio)

    # Define the first primitive vector of the triangular vertex lattice.
    a = np.array([h, 0.0], dtype=float)

    # Define the second primitive vector of the triangular vertex lattice.
    b = np.array([h / 2.0, v], dtype=float)

    # Select the triangle centroid that should be aligned with the lattice.
    aligned_centroid = (
        np.asarray(alignment_point, dtype=float).ravel()[:2]
        if alignment_point is not None
        else np.array([x_min, y_min], dtype=float)
    )

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

    # Convert the first triangle coordinate arrays into Shapely polygons.
    tri_a = shapely.polygons(tri_a_coords)

    # Convert the second triangle coordinate arrays into Shapely polygons.
    tri_b = shapely.polygons(tri_b_coords)

    # Allocate an object array containing both triangles for each lattice cell.
    polygons_2d = np.empty((nj, ni, 2), dtype=object)

    # Store the first triangle of each lattice cell.
    polygons_2d[:, :, 0] = tri_a

    # Store the second triangle of each lattice cell.
    polygons_2d[:, :, 1] = tri_b

    # Flatten all generated triangles to a one-dimensional array.
    polygons = polygons_2d.ravel()

    # Keep only triangles that intersect the requested surface.
    mask = shapely.intersects(polygons, surface_to_cover)

    # Extract the triangles that actually cover or touch the surface.
    kept_polygons = polygons[mask]

    # Compute adjacency from shared vertices on the kept triangles.
    adjacency_dict = adjacency_by_shared_vertices(kept_polygons)

    # Return the final tiling and its adjacency dictionary.
    return shapely.MultiPolygon(kept_polygons), adjacency_dict


def gen_polygonal_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
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
    surface_to_cover : shapely.Polygon or shapely.MultiPolygon
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
        Must be ≥ 1.  For example, choosing :attr:`PolygonType.RECTANGLE`
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
    rot_surface_to_cover = shapely.affinity.rotate(
        surface_to_cover,
        angle=-rot_deg,
        use_radians=False,
        origin=surface_to_cover.centroid,
    )

    rot_alignment_point = alignment_point
    if alignment_point is not None and rot_deg != 0.0:
        rot_alignment_point = np.array(
            shapely.affinity.rotate(
                shapely.Point(alignment_point),
                angle=-rot_deg,
                use_radians=False,
                origin=surface_to_cover.centroid,
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
    return shapely.affinity.rotate(
        _grid, angle=rot_deg, use_radians=False, origin=surface_to_cover.centroid
    ), _adj
