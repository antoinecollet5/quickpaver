# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""VTK / PyVista meshes from polygons: surface meshes with shared vertices and
layered extrusion to 3D (terrain-following) meshes.

The geometry (vertex merging, cell orientation, 3D cell definitions) runs in
Rust with the GIL released; PyVista is only used to wrap the resulting arrays
and is an optional dependency (``pip install quickpaver[pyvista]``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterable, Optional, Sequence, Union

import numpy as np

from quickpaver import _quickpaver as _rs
from quickpaver._quality import PolygonsLike, _vertex_buffer
from quickpaver._types import ArrayLike, NDArrayFloat

if TYPE_CHECKING:  # pragma: no cover
    import pyvista as pv

#: VTK cell type of polygons.
_VTK_POLYGON = 7


def _pyvista() -> Any:
    try:
        import pyvista as pv
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise ImportError(
            "PyVista is required to build VTK meshes. "
            "Install it with `pip install quickpaver[pyvista]`."
        ) from exc
    return pv


def _points3d(vertices: ArrayLike, z: float = 0.0) -> NDArrayFloat:
    v = np.asarray(vertices, dtype=np.float64)
    if v.ndim != 2 or v.shape[1] not in (2, 3):
        raise ValueError(f"vertices must have shape (n, 2) or (n, 3), got {v.shape}.")
    if v.shape[1] == 2:
        v = np.column_stack([v, np.full(len(v), float(z))])
    return np.ascontiguousarray(v)


def get_ncell4poly(npoints: int) -> int:
    """Number of legacy VTK cell-array values of a 3D cell extruded from a polygon.

    An ``n``-vertex polygon becomes a wedge (``n = 3``), hexahedron (4),
    pentagonal prism (5) or hexagonal prism (6), each written as
    ``[2n, ids...]``, or a polyhedron (``n >= 7``) written as
    ``[n_items, n_faces, face_0..., face_1..., ...]``.

    Parameters
    ----------
    npoints : int
        Number of vertices of the polygon exterior (closing vertex excluded).

    Returns
    -------
    int
        Number of values of the 3D cell in a legacy VTK cell array.

    Raises
    ------
    ValueError
        If ``npoints < 3`` (not a polygon).

    Examples
    --------
    >>> [get_ncell4poly(n) for n in (3, 4, 5, 6, 7)]
    [7, 9, 11, 13, 53]
    """
    if npoints < 3:
        raise ValueError(
            "The number of exterior points for the polygon should be at least 3."
        )
    if npoints <= 6:
        return 1 + 2 * npoints
    return 2 + (npoints + 1) * 2 + npoints * 5


def create_unstructured_mesh(
    vertices: ArrayLike,
    connectivities: Union[Sequence[Union[Sequence[int], np.ndarray]], np.ndarray],
) -> "pv.UnstructuredGrid":
    """Polygon mesh from vertices and cell connectivities.

    Parameters
    ----------
    vertices : ArrayLike
        Vertex coordinates, shape ``(n_vertices, 2)`` (placed at ``z = 0``) or
        ``(n_vertices, 3)``.
    connectivities : Sequence[Sequence[int]] or ArrayLike
        Vertex ids of every cell, in ring order: a list of lists (any sizes)
        or an ``(n_cells, k)`` integer array. Vertices can be shared by
        several cells.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh of ``VTK_POLYGON`` cells (a :class:`vtk.vtkUnstructuredGrid`
        subclass).

    Raises
    ------
    ValueError
        If a cell has fewer than 3 vertices or references a missing vertex.

    Examples
    --------
    >>> mesh = create_unstructured_mesh(
    ...     [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0]], [[0, 1, 2, 3], [1, 4, 2]]
    ... )
    >>> mesh.n_points, mesh.n_cells
    (5, 2)
    """
    pv = _pyvista()
    pts = _points3d(vertices)
    flat: np.ndarray
    if isinstance(connectivities, np.ndarray) and connectivities.ndim == 2:
        conn = connectivities.astype(np.int64, copy=False)
        counts = np.full(len(conn), conn.shape[1], dtype=np.int64)
        flat = conn.ravel()
    else:
        rows: Iterable[Union[Sequence[int], np.ndarray]] = connectivities
        cells_list = [np.asarray(c, dtype=np.int64).ravel() for c in rows]
        counts = np.array([len(c) for c in cells_list], dtype=np.int64)
        flat = np.concatenate(cells_list) if cells_list else np.empty(0, dtype=np.int64)
    if np.any(counts < 3):
        raise ValueError("Every cell needs at least 3 vertices.")
    if len(flat) and (int(np.min(flat)) < 0 or int(np.max(flat)) >= len(pts)):
        raise ValueError("A cell references a vertex that does not exist.")
    cells = np.empty(len(flat) + len(counts), dtype=np.int64)
    head = np.cumsum(counts + 1) - (counts + 1)
    is_head = np.zeros(len(cells), dtype=bool)
    is_head[head] = True
    cells[head] = counts
    cells[~is_head] = flat
    types = np.full(len(counts), _VTK_POLYGON, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, types, pts)


def shapely_polygons_to_vtu(
    geoms: PolygonsLike, z: float = 0.0, merge_tol: Optional[float] = None
) -> "pv.UnstructuredGrid":
    """Polygon mesh with shared vertices from a set of polygons.

    Vertices closer than ``merge_tol`` are merged (single-linkage clustering:
    two vertices are merged when a chain of vertices at most ``merge_tol``
    apart joins them) into their mean position, so adjacent cells share their
    points. Every ring is oriented counter-clockwise (cell normals towards
    ``+z``) and vertices collapsed by the merge are removed from the ring.
    Holes are ignored.

    Parameters
    ----------
    geoms : shapely.MultiPolygon, Sequence[shapely.Polygon] or NDArrayFloat
        The cells: any input of :func:`~quickpaver.multipolygon_quality`, e.g.
        Voronoi cells, a MultiPolygon, or the ``(n, k, 2)`` vertex array of
        :func:`~quickpaver.gen_polygonal_tiling_arrays`.
    z : float, default 0.0
        Altitude of every point.
    merge_tol : float, optional
        Merge distance, in the units of the coordinates. By default
        ``1e-9`` times the largest extent of the polygons, which merges the
        copies of a shared vertex (identical up to rounding) and nothing
        else. Use a larger value for polygons whose shared vertices only
        nearly coincide.

    Returns
    -------
    pyvista.UnstructuredGrid
        Mesh of ``VTK_POLYGON`` cells, in the order of the polygons.

    Raises
    ------
    ValueError
        If a polygon is empty or keeps fewer than 3 distinct vertices after
        merging, or if ``merge_tol`` is negative.

    Examples
    --------
    >>> import shapely
    >>> a, b = shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)
    >>> mesh = shapely_polygons_to_vtu([a, b], z=5.0)
    >>> mesh.n_points, mesh.n_cells
    (6, 2)
    """
    pv = _pyvista()
    xy, offsets, closed = _vertex_buffer(geoms)
    if merge_tol is None:
        v = xy.reshape(-1, 2)
        extent = float(np.max(np.ptp(v, axis=0))) if len(v) else 0.0
        merge_tol = 1e-9 * extent
    merge_tol = float(merge_tol)
    if not merge_tol >= 0:
        raise ValueError(f"merge_tol must be >= 0, got {merge_tol}.")
    points, cells = _rs.polygon_mesh(xy, offsets, closed, merge_tol)
    n_cells = len(offsets) - 1
    types = np.full(n_cells, _VTK_POLYGON, dtype=np.uint8)
    return pv.UnstructuredGrid(cells, types, _points3d(points, z))


def pv_extrude(surfmesh: Any, thickness: ArrayLike) -> "pv.UnstructuredGrid":
    """Extrude a surface mesh into layers of 3D cells (terrain-following mesh).

    Every point of the surface is repeated on ``len(thickness) + 1`` levels,
    level ``k + 1`` being level ``k`` shifted vertically by ``thickness[k]``,
    so the layers follow the surface topography. Every 2D cell gives one 3D
    cell per layer: a wedge (triangle), hexahedron (quadrilateral), pentagonal
    or hexagonal prism, or a polyhedron for 7 or more vertices. Cells are
    oriented so that VTK sees positive volumes and outward polyhedron faces,
    whatever the orientation of the 2D cells.

    Parameters
    ----------
    surfmesh : pyvista.DataSet
        Surface mesh of 2D cells (triangles, quads, pixels or polygons), e.g.
        from :func:`shapely_polygons_to_vtu`. Other datasets (e.g. a
        ``PolyData``) are cast to an unstructured grid.
    thickness : ArrayLike
        Thickness of every layer (non-zero, finite, all of the same sign:
        positive to stack the layers upwards, negative downwards).

    Returns
    -------
    pyvista.UnstructuredGrid
        The 3D mesh; cell ``i * n_layers + j`` is surface cell ``i`` in layer
        ``j``. Cell data:

        - ``"Volume"``: horizontal area of the surface cell times the layer
          thickness (its exact volume);
        - ``"Layer"``: layer index (uint16);
        - ``"Cell centers"``: if the surface has a ``"Cell centers"`` cell
          array ``(n, 3)``, its points moved to the middle of every layer
          (``z + sum(thickness[:j]) + thickness[j] / 2``).

    Raises
    ------
    ValueError
        If the thicknesses are invalid, or a cell is not a 2D cell with at
        least 3 points.

    Examples
    --------
    >>> import shapely
    >>> surf = shapely_polygons_to_vtu([shapely.box(0, 0, 2, 1)])
    >>> vol = pv_extrude(surf, [1.0, 0.5])
    >>> vol.n_cells, [float(v) for v in vol["Volume"]]
    (2, [2.0, 1.0])
    """
    pv = _pyvista()
    th = np.atleast_1d(np.asarray(thickness, dtype=np.float64))
    if th.ndim != 1 or len(th) == 0:
        raise ValueError("thickness must be a non-empty 1D sequence.")
    if not np.all(np.isfinite(th)) or np.any(th == 0):
        raise ValueError("Every thickness must be finite and non-zero.")
    if np.any(th > 0) and np.any(th < 0):
        raise ValueError("Every thickness must have the same sign.")
    nl = len(th)
    if nl > np.iinfo(np.uint16).max + 1:
        raise ValueError(f"At most {np.iinfo(np.uint16).max + 1} layers.")
    if not isinstance(surfmesh, pv.UnstructuredGrid):
        surfmesh = surfmesh.cast_to_unstructured_grid()

    pts = np.ascontiguousarray(surfmesh.points, dtype=np.float64)
    cells, types, area = _rs.extrude_cells(
        np.ascontiguousarray(surfmesh.cells, dtype=np.int64),
        np.ascontiguousarray(surfmesh.celltypes, dtype=np.uint8),
        pts,
        nl,
        bool(th[0] < 0),
    )
    nc = len(pts)
    points = np.tile(pts, (nl + 1, 1))
    for i in range(nl):
        points[(i + 1) * nc : (i + 2) * nc, 2] = (
            points[i * nc : (i + 1) * nc, 2] + th[i]
        )

    mesh = pv.UnstructuredGrid(cells, types, points)
    n2d = len(area)
    mesh.cell_data["Volume"] = np.outer(area, np.abs(th)).ravel()
    mesh.cell_data["Layer"] = np.tile(np.arange(nl, dtype=np.uint16), n2d)

    if "Cell centers" in surfmesh.cell_data.keys():
        cc = np.asarray(surfmesh.cell_data["Cell centers"], dtype=np.float64)
        cc = cc.reshape(n2d, -1)
        mid = np.concatenate([[0.0], np.cumsum(th)[:-1]]) + th / 2.0
        out = np.repeat(cc[:, None, :], nl, axis=1)
        out[:, :, 2] += mid[None, :]
        mesh.cell_data["Cell centers"] = out.reshape(n2d * nl, -1)
    return mesh
