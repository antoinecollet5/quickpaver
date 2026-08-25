# SPDX-License-Identifier: BSD-3-Clause
"""Shared helpers for the transfer-matrix test suite."""

from __future__ import annotations

import math
from typing import List, Optional, Tuple

import numpy as np
import shapely

from quickpaver import TriMesh


def make_regular_grid(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> shapely.MultiPolygon:
    """Axis-aligned rectangular grid as a ``MultiPolygon`` (same convention
    as the helper used in ``test_transfer_matrix.py``)."""
    boxes = shapely.box(
        np.repeat(np.arange(nx) * dx + x0, ny),
        np.tile(np.arange(ny) * dy + y0, nx),
        np.repeat(np.arange(nx) * dx + x0 + dx, ny),
        np.tile(np.arange(ny) * dy + y0 + dy, nx),
    )
    return shapely.MultiPolygon(boxes)


def make_grid_trimesh(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
    ccw: bool = True,
) -> TriMesh:
    """
    A ``TriMesh`` covering ``[x0, x0 + nx*dx] x [y0, y0 + ny*dy]``, with each
    unit cell split into two triangles.

    Parameters
    ----------
    ccw : bool
        If True (default), triangles are wound counter-clockwise. If False,
        every triangle is wound clockwise instead (used to exercise
        :func:`quickpaver._transfer_matrix._ensure_ccw_triangles`).
    """
    xs = x0 + np.arange(nx + 1) * dx
    ys = y0 + np.arange(ny + 1) * dy
    verts = np.array([[x, y] for y in ys for x in xs], dtype=float)

    def vid(i: int, j: int) -> int:
        return j * (nx + 1) + i

    tri_verts: List[List[int]] = []
    for j in range(ny):
        for i in range(nx):
            v00, v10, v11, v01 = (
                vid(i, j),
                vid(i + 1, j),
                vid(i + 1, j + 1),
                vid(i, j + 1),
            )
            if ccw:
                tri_verts.append([v00, v10, v11])
                tri_verts.append([v00, v11, v01])
            else:
                tri_verts.append([v00, v11, v10])
                tri_verts.append([v00, v01, v11])
    tri_verts_arr = np.asarray(tri_verts, dtype=int)

    p0 = verts[tri_verts_arr[:, 0]]
    p1 = verts[tri_verts_arr[:, 1]]
    p2 = verts[tri_verts_arr[:, 2]]
    area = 0.5 * np.abs(
        (p1[:, 0] - p0[:, 0]) * (p2[:, 1] - p0[:, 1])
        - (p2[:, 0] - p0[:, 0]) * (p1[:, 1] - p0[:, 1])
    )
    e0 = np.linalg.norm(p1 - p0, axis=1)
    e1 = np.linalg.norm(p2 - p1, axis=1)
    e2 = np.linalg.norm(p0 - p2, axis=1)
    edge_lengths_m = np.stack([e0, e1, e2], axis=1)

    return TriMesh(
        verts_xy=verts,
        tri_verts=tri_verts_arr,
        edge_lengths_m=edge_lengths_m,
        tri_area_m2=area,
    )


def make_single_triangle_mesh(
    p0: Tuple[float, float],
    p1: Tuple[float, float],
    p2: Tuple[float, float],
) -> TriMesh:
    """A one-triangle ``TriMesh`` from three explicit (possibly CW) points."""
    verts = np.array([p0, p1, p2], dtype=float)
    tri_verts = np.array([[0, 1, 2]], dtype=int)
    area = 0.5 * abs(
        (p1[0] - p0[0]) * (p2[1] - p0[1]) - (p2[0] - p0[0]) * (p1[1] - p0[1])
    )
    e0 = math.dist(p0, p1)
    e1 = math.dist(p1, p2)
    e2 = math.dist(p2, p0)
    return TriMesh(
        verts_xy=verts,
        tri_verts=tri_verts,
        edge_lengths_m=np.array([[e0, e1, e2]]),
        tri_area_m2=np.array([area]),
    )


def rotate_trimesh(
    mesh: TriMesh, angle_deg: float, origin: Optional[Tuple[float, float]] = None
) -> TriMesh:
    """Rotate a mesh's vertices (CCW) about ``origin`` (default: mesh centroid)."""
    return mesh.transform(rot_deg=angle_deg, origin=origin)


def regular_ngon(
    cx: float, cy: float, radius: float, n: int, rotation_deg: float = 0.0
) -> shapely.Polygon:
    """A convex, holeless, CCW regular n-gon centered at ``(cx, cy)``."""
    angles = np.deg2rad(rotation_deg) + np.linspace(0.0, 2 * np.pi, n, endpoint=False)
    xs = cx + radius * np.cos(angles)
    ys = cy + radius * np.sin(angles)
    return shapely.Polygon(np.stack([xs, ys], axis=-1))


def l_shape_polygon(x0: float, y0: float, size: float) -> shapely.Polygon:
    """A simple non-convex 'L'-shaped polygon (6 vertices), CCW, holeless."""
    s = size
    pts = [
        (x0, y0),
        (x0 + s, y0),
        (x0 + s, y0 + s / 2),
        (x0 + s / 2, y0 + s / 2),
        (x0 + s / 2, y0 + s),
        (x0, y0 + s),
    ]
    return shapely.Polygon(pts)


def square_with_hole(
    x0: float, y0: float, size: float, hole_ratio: float = 0.3
) -> shapely.Polygon:
    """An axis-aligned square with a smaller square hole in the middle."""
    outer = [
        (x0, y0),
        (x0 + size, y0),
        (x0 + size, y0 + size),
        (x0, y0 + size),
    ]
    h = size * hole_ratio
    m = x0 + size / 2
    n = y0 + size / 2
    inner = [
        (m - h, n - h),
        (m - h, n + h),
        (m + h, n + h),
        (m + h, n - h),
    ]
    return shapely.Polygon(outer, [inner])
