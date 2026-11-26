# SPDX-License-Identifier: BSD-3-Clause
"""
Tests of the 0.7.0 tiling fast paths: the Rust inside / outside / ambiguous
classifier behind ``intersects_mask`` and the tile selection, and the
Shapely-free ``gen_polygonal_tiling_arrays``.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely

import quickpaver
from quickpaver._tiling import _intersects_geos


def _surfaces():
    star = shapely.Polygon(
        [
            (10 * np.cos(t) * (1 + 0.5 * (k % 2)), 10 * np.sin(t) * (1 + 0.5 * (k % 2)))
            for k, t in enumerate(np.linspace(0, 2 * np.pi, 23)[:-1])
        ]
    )
    holed = shapely.Polygon(
        [(0, 0), (20, 0), (20, 14), (0, 14)], [[(5, 4), (5, 10), (12, 10), (12, 4)]]
    )
    many = shapely.MultiPolygon(
        [
            shapely.box(3 * i, 3 * j, 3 * i + 2, 3 * j + 2)
            for i in range(4)
            for j in range(4)
        ]
    )
    return {
        "star": star,
        "holed": holed,
        "multi": many,
        "france": quickpaver.load_france_and_corsica_contour(),
    }


@pytest.mark.parametrize("name", ["star", "holed", "multi", "france"])
def test_intersects_mask_equals_geos(name: str) -> None:
    """Random polygons (including boundary-touching ones) get exactly the
    GEOS predicate, whatever the classifier decides."""
    surf = _surfaces()[name]
    x0, y0, x1, y1 = surf.bounds
    w = max(x1 - x0, y1 - y0)
    rng = np.random.default_rng(3)
    n = 4000
    cx = rng.uniform(x0 - 0.1 * w, x1 + 0.1 * w, n)
    cy = rng.uniform(y0 - 0.1 * w, y1 + 0.1 * w, n)
    r = rng.uniform(0.001, 0.05, n) * w
    polys = shapely.buffer(shapely.points(cx, cy), r, quad_segs=2)
    # polygons exactly touching the surface boundary from the outside
    ext = shapely.get_exterior_ring(shapely.get_parts(surf)[0])
    pts = shapely.line_interpolate_point(ext, np.linspace(0, 1, 50), normalized=True)
    touch = shapely.buffer(pts, 0.01 * w, cap_style="square")
    polys = np.concatenate([polys, touch, [shapely.Polygon()]])
    got = quickpaver.intersects_mask(polys, surf)
    shapely.prepare(surf)
    np.testing.assert_array_equal(got, shapely.intersects(polys, surf))
    np.testing.assert_array_equal(got, _intersects_geos(polys, surf))


@pytest.mark.parametrize("pt", ["rectangle", "hexagon", "triangle"])
@pytest.mark.parametrize("rot", [0.0, 23.0])
def test_arrays_api_matches_shapely_api(pt: str, rot: float) -> None:
    surf = _surfaces()["holed"]
    args = (surf, quickpaver.PolygonType(pt), 0.9, 1.2)
    tiling, adj = quickpaver.gen_polygonal_tiling(
        *args, rot_deg=rot, alignment_point=(1, 2)
    )
    verts, indptr, indices = quickpaver.gen_polygonal_tiling_arrays(
        *args, rot_deg=rot, alignment_point=(1, 2)
    )
    polys = shapely.get_parts(tiling)
    assert verts.shape == (
        len(polys),
        {"rectangle": 4, "hexagon": 6, "triangle": 3}[pt],
        2,
    )
    assert shapely.equals_exact(shapely.polygons(verts), polys, tolerance=0).all()
    assert len(indptr) == len(adj) + 1
    for i, neigh in adj.items():
        assert indices[indptr[i] : indptr[i + 1]].tolist() == neigh


def test_arrays_api_empty_and_disk() -> None:
    far = shapely.box(0, 0, 1e-9, 1e-9)
    v, ip, ix = quickpaver.gen_polygonal_tiling_arrays(
        far, quickpaver.PolygonType.HEXAGON, 1.0
    )
    assert len(v) == len(ip) - 1
    disk = quickpaver.Disk((0.0, 0.0), 5.0)
    v, ip, ix = quickpaver.gen_polygonal_tiling_arrays(
        disk, quickpaver.PolygonType.TRIANGLE, 0.5, rot_deg=10.0
    )
    g, adj = quickpaver.gen_polygonal_tiling(
        disk, quickpaver.PolygonType.TRIANGLE, 0.5, rot_deg=10.0
    )
    assert len(v) == len(adj) and ip[-1] == len(ix)
