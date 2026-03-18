# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Tiling with triangles, squares and hexagons with support for anisotropy."""

import math
from collections import defaultdict
from typing import Dict, List, Tuple, Union

import numpy as np
import shapely
import shapely.affinity

from quickpaver._types import NDArrayFloat, NDArrayInt, StrEnum

SQRT3 = math.sqrt(3)


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
    _summary_

    Parameters
    ----------
    poly_type : PolygonType
        _description_
    edge_length : float, optional
        Edge length for the base polygon. The default is 1.0.
    edge_length: float
        Edge length for the base polygon.
        E.g., choosing :py:attr:`PolygonType.RECTANGLE` with `anisotropy_ratio` = 2
        results in rectangles with scale (1.0, 2.0).

    Returns
    -------
    shapely.Polygon
        _description_

    Raises
    ------
    ValueError
        _description_
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


def gen_rectangular_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
    edge_length: float,
    anisotropy_ratio: float = 1.0,
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

    Returns
    -------
    Tuple[shapely.MultiPolygon, DefaultDict[int, Set[int]]]
        A tuple containing:
            - A shapely collection of polygons.
            - A dictionary where keys are hexagon indices and values are sets of
              adjacent hexagon indices.

    """
    # extract the coordinates of the bounding box
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    v_step = edge_length * anisotropy_ratio  #  Vertical step (height of a hexagon)
    h_step = edge_length  # Horizontal step (width of a hexagon)

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
    centers = np.array(
        np.meshgrid(
            np.linspace(h_start + h_step / 2.0, h_start + (nh - 0.5) * h_step, nh),
            np.linspace(v_start + v_step / 2.0, v_start + (nv - 0.5) * v_step, nv),
        )
    )

    # vertices for one polygon
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
    adjacency_dict = rectangular_grid_adjacency_masked(nv, nh, mask)

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


def gen_hexagonal_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
    edge_length: float,
    anisotropy_ratio: float = 1.0,
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

    Returns
    -------
    Tuple[shapely.MultiPolygon, DefaultDict[int, Set[int]]]
        A tuple containing:
            - A shapely collection of polygons.
            - A dictionary where keys are hexagon indices and values are sets
              of adjacent hexagon indices.

    """
    # extract the coordinates of the bounding box
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    # Calculate the vertical and horizontal step distances between centers of hexagons
    v_step = (
        math.sqrt(3) * edge_length * anisotropy_ratio
    )  #  Vertical step (height of a hexagon)
    h_step = 1.5 * edge_length  # Horizontal step (width of a hexagon)

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
    adjacency_dict = hexagonal_grid_adjacency_masked(nv, nh, mask)

    return shapely.MultiPolygon(polygons[mask]), adjacency_dict


def extract_tiling_centers(polygons: List[shapely.Polygon]) -> NDArrayFloat:
    """
    _summary_

    Parameters
    ----------
    polygons : List[shapely.Polygon]
        _description_

    Returns
    -------
    NDArrayFloat
        _description_
    """
    return np.array([geom.centroid.xy for geom in polygons])[:, :, 0]


def extract_tiling_vertices(
    polygons: List[shapely.Polygon], n_decimals: int = 2
) -> Tuple[NDArrayFloat, Dict[int, List[int]], NDArrayInt]:
    """
    _summary_

    Parameters
    ----------
    polygons : List[shapely.Polygon]
        _description_
    n_decimals : int, optional
        _description_, by default 2

    Returns
    -------
    Tuple[NDArrayFloat, Dict[int, List[int]]]
        _description_
    """

    # Convert polygons to arrays of vertices (rounded to avoid floating point issues)
    verts = [
        np.round(np.array(p.exterior.coords[:-1]), decimals=n_decimals)
        for p in polygons
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

    Returns
    -------
    Tuple[shapely.MultiPolygon, DefaultDict[int, Set[int]]]
        A tuple containing:
            - A shapely collection of polygons.
            - A dictionary where keys are hexagon indices and values are sets of
              adjacent hexagon indices.

    """
    # extract the coordinates of the bounding box
    x_min, y_min, x_max, y_max = surface_to_cover.bounds

    # Calculate the vertical and horizontal step distances between centers of hexagons
    v_step = (
        math.sqrt((edge_length**2) * 3 / 4) * anisotropy_ratio
    )  #  Vertical step (height of a hexagon)
    h_step = edge_length / 2  # Horizontal step (width of a hexagon)

    # number of columns of polygons
    nh: int = math.ceil((x_max - x_min) / h_step) + 3
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

    # shift rows
    centers[0, ::2, :] += h_step

    # vertices for one polygon
    verts_up = np.array(
        gen_polygon(PolygonType.TRIANGLE, edge_length, anisotropy_ratio).exterior.coords
    )
    verts_down = np.array(
        shapely.affinity.rotate(
            gen_polygon(PolygonType.TRIANGLE, edge_length, anisotropy_ratio),
            180,
            use_radians=False,
        ).exterior.coords
    )

    _tmp = np.moveaxis(centers, 0, -1)
    # Broadcast vertices and create polygons (flatten)
    polygons = shapely.polygons(_tmp[:, :, None, :] + verts_up[None, None, :, :])

    polygons[:, ::2] = shapely.polygons(
        _tmp[:, ::2, None, :] + verts_down[None, None, :, :]
    )

    polygons = polygons.ravel()

    # Mask intersecting polygons (to keep)
    mask = shapely.intersects(polygons, surface_to_cover)

    # Adjacency of kept polygons
    adjacency_dict = adjacency_by_shared_vertices(polygons[mask])

    return shapely.MultiPolygon(polygons[mask]), adjacency_dict


def gen_polygonal_tiling(
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon],
    poly_type: PolygonType,
    edge_length: float,
    anisotropy_ratio: float = 1.0,
    rot_deg=0.0,
) -> Tuple[shapely.MultiPolygon, Dict[int, List[int]]]:
    """
    Cover the given surface with tiles (polygons) of the desired type.

    Parameters
    ----------
    surface_to_cover: Union[shapely.Polygon, shapely.MultiPolygon]
        Surface to cover with the tiling. Only the polygon intersecting this surface
        are kept.
    poly_type : PolygonType
        Type of tile (polygon) See :py:class:`PolygonType` for the available
        geometries.
    edge_length: float
        Edge length for the base polygon.
        E.g., choosing :py:attr:`PolygonType.RECTANGLE` with `anisotropy_ratio` = 2
        results in rectangles with scale (1.0, 2.0).
    edge_length : float
        _description_
    anisotropy_ratio : float, optional
        _description_, by default 1.0
    rot_deg : float, optional
        _description_, by default 0.

    Returns
    -------
    Tuple[shapely.MultiPolygon, Dict[int, List[int]]]
        A tuple containing:
            - A shapely collection of polygons.
            - A dictionary where keys are hexagon indices and values are sets of
              adjacent hexagon indices.

    """
    rot_surface_to_cover = shapely.affinity.rotate(
        surface_to_cover,
        angle=-rot_deg,
        use_radians=False,
        origin=surface_to_cover.centroid,
    )

    if poly_type == PolygonType.HEXAGON:
        _grid, _adj = gen_hexagonal_tiling(
            rot_surface_to_cover,
            edge_length=edge_length,
            anisotropy_ratio=anisotropy_ratio,
        )
    elif poly_type == PolygonType.TRIANGLE:
        _grid, _adj = gen_triangular_tiling(
            rot_surface_to_cover,
            edge_length=edge_length,
            anisotropy_ratio=anisotropy_ratio,
        )
    elif poly_type == PolygonType.RECTANGLE:
        _grid, _adj = gen_rectangular_tiling(
            rot_surface_to_cover,
            edge_length=edge_length,
            anisotropy_ratio=anisotropy_ratio,
        )
    else:
        raise ValueError(PolygonType(poly_type))
    return shapely.affinity.rotate(
        _grid, angle=rot_deg, use_radians=False, origin=surface_to_cover.centroid
    ), _adj
