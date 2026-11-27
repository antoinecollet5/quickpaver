# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Roundness quality of triangular, polygonal and Voronoi cells.

The score of a cell is ``1`` for a regular polygon and decreases towards ``0``
as the cell is distorted:

- triangles use the exact ratio ``2 r / R`` (inradius over circumradius);
- other polygons use ``(r_in / r_circ) / cos(pi / n)``, where ``r_circ`` is
  the largest vertex distance and ``r_in`` the smallest edge distance from a
  reference point (the area-weighted centroid unless one is given), and
  ``cos(pi / n)`` is that ratio for the regular ``n``-gon.

The evaluation runs in a single parallel Rust pass (GIL released) directly on
the vertex buffer returned by Shapely, or on ``(n, k, 2)`` vertex arrays such
as the ones of :func:`~quickpaver.gen_polygonal_tiling_arrays` without any
Shapely geometry at all.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import shapely

from quickpaver import _quickpaver as _rs
from quickpaver._types import NDArrayFloat

PolygonsLike = Union[
    shapely.MultiPolygon,
    shapely.Polygon,
    Sequence[Optional[shapely.Polygon]],
    NDArrayFloat,
]


def _vertex_buffer(
    polygons: PolygonsLike,
) -> tuple[NDArrayFloat, np.ndarray, bool]:
    """Interleaved vertices, CSR offsets (in vertices) and the closed flag."""
    if isinstance(polygons, np.ndarray) and polygons.dtype != object:
        arr = np.asarray(polygons, dtype=np.float64)
        if arr.ndim != 3 or arr.shape[2] != 2:
            raise ValueError(
                "vertex arrays must have shape (n_polygons, n_vertices, 2), "
                f"got {arr.shape}."
            )
        n, k = arr.shape[:2]
        offsets = np.arange(n + 1, dtype=np.intp) * k
        return np.ascontiguousarray(arr).reshape(-1), offsets, False

    if isinstance(polygons, (shapely.MultiPolygon, shapely.Polygon)):
        geoms = shapely.get_parts(polygons)
    else:
        geoms = np.asarray(list(polygons), dtype=object)
        if geoms.ndim != 1:
            geoms = geoms.ravel()
    if len(geoms):
        tid = shapely.get_type_id(geoms)
        if np.any((tid != shapely.GeometryType.POLYGON) & (tid != -1)):
            raise TypeError("Every geometry must be a shapely.Polygon (or None).")
    # Read the polygon coordinates directly (exterior ring first, then the
    # holes): ~2x cheaper than extracting the exterior rings, which makes GEOS
    # clone every ring.
    count = shapely.get_num_coordinates(geoms)
    xy = shapely.get_coordinates(geoms)
    holed = np.flatnonzero(shapely.get_num_interior_rings(geoms) > 0)
    if len(holed):
        # drop the hole coordinates: keep the first `ext` of every polygon
        ext = count.copy()
        ext[holed] = shapely.get_num_coordinates(
            shapely.get_exterior_ring(geoms[holed])
        )
        start = np.repeat(np.cumsum(count) - count, count)
        keep = np.arange(len(xy)) - start < np.repeat(ext, count)
        xy, count = xy[keep], ext
    offsets = np.zeros(len(geoms) + 1, dtype=np.intp)
    np.cumsum(count, out=offsets[1:])
    return np.ascontiguousarray(xy, dtype=np.float64).reshape(-1), offsets, True


def multipolygon_quality(
    multi_polygon: PolygonsLike,
    centers: Optional[NDArrayFloat] = None,
    parallel: bool = True,
) -> NDArrayFloat:
    """Roundness quality of every polygon of a tessellation.

    3-vertex polygons get the exact triangle ratio ``2 r / R``; polygons with
    more vertices get ``(r_in / r_circ) / cos(pi / n)`` around their centroid
    (or around ``centers``), so mixed collections (a triangulation, a Voronoi
    diagram with a few triangular boundary cells, a tiling) are handled in one
    call. Only exterior rings are evaluated: holes do not change the
    roundness of the cell outline.

    Parameters
    ----------
    multi_polygon : shapely.MultiPolygon, Sequence[shapely.Polygon] or NDArrayFloat
        The ``n_poly`` cells: a MultiPolygon (or a single Polygon), any
        sequence of Polygons or ``None`` (e.g. Voronoi cells,
        ``multi_polygon.geoms``), or
        a ``(n_poly, n_vertices, 2)`` array of open rings (e.g. the vertices
        returned by :func:`~quickpaver.gen_polygonal_tiling_arrays`), which
        skips Shapely entirely and is the fastest input.
    centers : NDArrayFloat, optional
        Array of shape ``(n_poly, 2)`` with one reference point per polygon
        (e.g. the Voronoi seed points), in the order of the polygons. By
        default, the area-weighted centroid of each polygon is used. Ignored
        for 3-vertex polygons, which always use ``2 r / R``.
    parallel : bool, default True
        Evaluate the polygons on all CPU cores. Small batches (a few thousand
        polygons) always run on one thread.

    Returns
    -------
    NDArrayFloat
        Array of shape ``(n_poly,)`` of scores in ``[0, 1]``, in the order of
        the polygons: ``1`` for a regular polygon, ``0`` for a degenerate
        (zero-area) one, ``NaN`` for an empty polygon or ``None``.

    Raises
    ------
    ValueError
        If ``centers`` does not have shape ``(n_poly, 2)``, or if a vertex
        array does not have shape ``(n_poly, n_vertices, 2)``.
    TypeError
        If a geometry is not a Polygon.

    Examples
    --------
    >>> import numpy as np
    >>> from shapely import MultiPolygon, Polygon
    >>> tri = Polygon([(0.0, 0.0), (1.0, 0.0), (0.5, np.sqrt(3) / 2)])
    >>> hexagon = Polygon(
    ...     [(np.cos(t), np.sin(t)) for t in np.linspace(0, 2 * np.pi, 7)[:-1]]
    ... )
    >>> scores = multipolygon_quality(MultiPolygon([tri, hexagon]))
    >>> bool(np.allclose(scores, [1.0, 1.0], atol=1e-6))
    True
    """
    xy, offsets, closed = _vertex_buffer(multi_polygon)
    n_poly = len(offsets) - 1
    c = None
    if centers is not None:
        c = np.asarray(centers, dtype=np.float64)
        if c.shape != (n_poly, 2):
            raise ValueError(f"centers must have shape ({n_poly}, 2), got {c.shape}.")
        c = np.ascontiguousarray(c).reshape(-1)
    return _rs.cell_quality(xy, offsets, closed, c, bool(parallel))


def _check_anisotropy(anisotropy: float) -> float:
    a = float(anisotropy)
    if not a > 0:
        raise ValueError(f"anisotropy must be > 0, got {anisotropy}.")
    return a


def triangle_anisotropy_quality(anisotropy: float) -> float:
    """Closed-form quality of an equilateral triangle under anisotropic stretch.

    The triangle is a unit-circumradius equilateral triangle (one vertex on
    the positive x-axis) stretched by ``x *= sqrt(a)``, ``y /= sqrt(a)``, an
    area-preserving map. It stays isosceles and its exact ``2 r / R`` ratio is

    .. math:: q(a) = \\frac{4 (\\sqrt{3 a^2 + 1} - 1)}{3 a^2 + 1},

    the value :func:`multipolygon_quality` returns for the stretched triangle.
    The triangle has no vertex on the y-axis, so ``q(a) != q(1 / a)``.

    Parameters
    ----------
    anisotropy : float
        Stretch factor ``a > 0``; ``a = 1`` is the undistorted triangle.

    Returns
    -------
    float
        Quality score in ``(0, 1]``.

    Examples
    --------
    >>> round(triangle_anisotropy_quality(1.0), 6)
    1.0
    >>> round(triangle_anisotropy_quality(2.0), 4)
    0.8017
    >>> round(triangle_anisotropy_quality(0.5), 4)
    0.738
    """
    a = _check_anisotropy(anisotropy)
    d = 3.0 * a * a + 1.0
    return float(4.0 * (np.sqrt(d) - 1.0) / d)


def square_anisotropy_quality(anisotropy: float) -> float:
    """Closed-form quality of a square under anisotropic stretch.

    The square is a unit-circumradius square in "diamond" orientation
    (vertices on the axes) stretched by ``x *= sqrt(a)``, ``y /= sqrt(a)``,
    which gives a rhombus of quality

    .. math:: q(a) = \\sqrt{\\frac{2}{1 + \\max(a, 1/a)^2}}.

    The vertex set is symmetric under swapping x and y, so ``q(a) = q(1/a)``.

    Parameters
    ----------
    anisotropy : float
        Stretch factor ``a > 0``; ``a = 1`` is the undistorted square.

    Returns
    -------
    float
        Quality score in ``(0, 1]``.

    Examples
    --------
    >>> round(square_anisotropy_quality(1.0), 6)
    1.0
    >>> round(square_anisotropy_quality(2.0), 4)
    0.6325
    >>> import math
    >>> math.isclose(square_anisotropy_quality(3.0), square_anisotropy_quality(1 / 3))
    True
    """
    a = _check_anisotropy(anisotropy)
    a = max(a, 1.0 / a)
    return float(np.sqrt(2.0 / (1.0 + a * a)))


def hexagon_anisotropy_quality(anisotropy: float) -> float:
    """Closed-form quality of a regular hexagon under anisotropic stretch.

    The hexagon is a unit-circumradius regular hexagon (one vertex on the
    positive x-axis) stretched by ``x *= sqrt(a)``, ``y /= sqrt(a)``. The
    limiting vertex / edge pair switches at ``a = 1``:

    - ``a >= 1``: the vertex on the x-axis is the farthest and the horizontal
      edges the nearest, so ``q(a) = 1 / a``;
    - ``a <= 1``: the off-axis vertices are the farthest and the slanted edges
      the nearest, so ``q(a) = 4 a / (a^2 + 3)``.

    The hexagon has no vertex on the y-axis, so ``q(a) != q(1 / a)``.

    Parameters
    ----------
    anisotropy : float
        Stretch factor ``a > 0``; ``a = 1`` is the undistorted hexagon.

    Returns
    -------
    float
        Quality score in ``(0, 1]``.

    Examples
    --------
    >>> round(hexagon_anisotropy_quality(1.0), 6)
    1.0
    >>> round(hexagon_anisotropy_quality(2.0), 4)
    0.5
    >>> round(hexagon_anisotropy_quality(0.5), 4)
    0.6154
    """
    a = _check_anisotropy(anisotropy)
    if a >= 1.0:
        return 1.0 / a
    return float(4.0 * a / (a * a + 3.0))
