# SPDX-License-Identifier: BSD-3-Clause
"""
Tests of the Rust-backed transfer machinery added in 0.7.0: the Shapely-free
vertices output, polygon arrays as grids, the interior-cell shortcut, exact
areas of concave / holed polygons without GEOS, input validation of the
extension and thread safety (free-threaded builds).
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
import shapely

import quickpaver
from quickpaver import IntersectionVertices, RectilinearGrid, TriMesh
from quickpaver import _quickpaver as rs
from quickpaver._tm_test_utils import (
    l_shape_polygon,
    make_grid_trimesh,
    make_regular_grid,
    regular_ngon,
    square_with_hole,
)


def _reference(source, target) -> np.ndarray:
    """Dense transfer matrix computed with GEOS only (pure Shapely)."""

    def polys(g):
        if isinstance(g, RectilinearGrid):
            return np.asarray(g.to_shapely().geoms, dtype=object)
        if isinstance(g, TriMesh):
            return shapely.polygons(g.verts_xy[g.tri_verts])
        if isinstance(g, shapely.MultiPolygon):
            return shapely.get_parts(g)
        return np.asarray(g, dtype=object)

    s, t = polys(source), polys(target)
    inter = shapely.area(shapely.intersection(s[:, None], t[None, :]))
    return inter / shapely.area(s)[:, None]


def _grids():
    rect = RectilinearGrid(cx=0.3, cy=-0.2, dx=0.5, dy=0.4, nx=12, ny=10, theta=0.0)
    rot = RectilinearGrid(cx=0.1, cy=0.2, dx=0.7, dy=0.6, nx=9, ny=8, theta=23.0)
    mesh = make_grid_trimesh(-2.0, -2.0, 0.55, 0.6, 7, 6)
    concave = [
        l_shape_polygon(-1.5, -1.0, 1.2),
        square_with_hole(0.4, -1.6, 2.0, 0.3),
        regular_ngon(-0.5, 1.2, 0.9, 7),
        shapely.Polygon([(1, 1), (2.5, 1), (2.5, 2.5), (2.0, 1.6), (1.4, 2.6)]),
    ]
    convex = [regular_ngon(x, y, 0.45, 6) for x in (-1.2, 0.0, 1.2) for y in (-1, 0, 1)]
    return {
        "rect": rect,
        "rot": rot,
        "mesh": mesh,
        "concave": shapely.MultiPolygon(concave),
        "convex": shapely.MultiPolygon(convex),
    }


PAIRS = [
    ("rect", "rot"),
    ("rot", "rect"),
    ("rect", "mesh"),
    ("mesh", "rect"),
    ("rect", "concave"),
    ("concave", "rect"),
    ("mesh", "concave"),
    ("concave", "mesh"),
    ("mesh", "convex"),
    ("convex", "concave"),
    ("concave", "convex"),
    ("concave", "concave"),
    ("mesh", "mesh"),
]


@pytest.mark.parametrize(("s", "t"), PAIRS)
def test_matrix_matches_geos_reference(s: str, t: str) -> None:
    g = _grids()
    W = quickpaver.compute_transfer_matrix(g[s], g[t])
    np.testing.assert_allclose(W.toarray(), _reference(g[s], g[t]), atol=1e-12)


@pytest.mark.parametrize(("s", "t"), PAIRS)
def test_vertices_match_shapely_output(s: str, t: str) -> None:
    g = _grids()
    W1, s1, t1, ix = quickpaver.compute_transfer_matrix_with_intersections(g[s], g[t])
    W2, s2, t2, iv = quickpaver.compute_transfer_matrix_with_intersection_vertices(
        g[s], g[t]
    )
    assert isinstance(iv, IntersectionVertices)
    assert iv.n_entries == W2.nnz == len(ix)
    assert (W1 != W2).nnz == 0
    np.testing.assert_array_equal(s1, s2)
    np.testing.assert_array_equal(t1, t2)
    rebuilt = iv.to_shapely()
    # same polygonal geometry (GEOS may add non-polygonal pieces, e.g. lines)
    np.testing.assert_allclose(shapely.area(rebuilt), shapely.area(ix), atol=1e-14)
    np.testing.assert_allclose(
        shapely.area(shapely.symmetric_difference(rebuilt, ix)), 0.0, atol=1e-12
    )
    # every entry's geometry area equals its matrix weight times the source area
    np.testing.assert_allclose(
        shapely.area(rebuilt), W1.toarray()[s1, t1] * _source_areas(g[s])[s1], rtol=1e-9
    )


def _source_areas(g) -> np.ndarray:
    if isinstance(g, RectilinearGrid):
        return np.full(g.nx * g.ny, g.dx * g.dy)
    if isinstance(g, TriMesh):
        return g.tri_area_m2
    if isinstance(g, shapely.MultiPolygon):
        return shapely.area(shapely.get_parts(g))
    return shapely.area(np.asarray(g, dtype=object))


def test_vertices_layout_of_holed_and_multipart_geometry() -> None:
    """Non-convex pairs keep the GEOS topology: several parts, holes."""
    u_shape = shapely.Polygon(
        [(0, 0), (3, 0), (3, 2), (2, 2), (2, 1), (1, 1), (1, 2), (0, 2)]
    )
    holed = square_with_hole(5.0, 0.0, 3.0, 1.0 / 6.0)  # 1 x 1 hole
    # a strip cutting the two arms of the U -> one entry with two polygons
    strip = RectilinearGrid(cx=1.5, cy=1.75, dx=4.0, dy=0.5, nx=1, ny=1)
    _, _, _, iv = quickpaver.compute_transfer_matrix_with_intersection_vertices(
        strip, [u_shape]
    )
    assert iv.n_entries == 1
    assert np.diff(iv.geometry_offsets).tolist() == [2]
    (geom,) = iv.to_shapely()
    assert geom.geom_type == "MultiPolygon"
    np.testing.assert_allclose(geom.area, 1.0, atol=1e-12)
    # a cell containing the holed square -> one polygon with two rings
    cell = RectilinearGrid(cx=6.5, cy=1.5, dx=4.0, dy=4.0, nx=1, ny=1)
    _, _, _, iv = quickpaver.compute_transfer_matrix_with_intersection_vertices(
        cell, [holed]
    )
    assert np.diff(iv.geometry_offsets).tolist() == [1]
    assert np.diff(iv.polygon_offsets).tolist() == [2]
    (geom,) = iv.to_shapely()
    assert geom.geom_type == "Polygon" and len(geom.interiors) == 1
    np.testing.assert_allclose(geom.area, 9.0 - 1.0, atol=1e-12)
    # rings are closed
    for r0, r1 in zip(iv.ring_offsets[:-1], iv.ring_offsets[1:]):
        np.testing.assert_array_equal(iv.coords[r0], iv.coords[r1 - 1])


def test_empty_vertices_output() -> None:
    rect = RectilinearGrid(nx=2, ny=2)
    far = shapely.MultiPolygon([shapely.box(100, 100, 101, 101)])
    W, s, t, iv = quickpaver.compute_transfer_matrix_with_intersection_vertices(
        rect, far
    )
    assert W.nnz == 0 and iv.n_entries == 0 and len(iv.to_shapely()) == 0


@pytest.mark.parametrize("kind", ["list", "array", "tuple"])
def test_polygon_sequences_are_accepted(kind: str) -> None:
    rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.25, dy=0.25, nx=8, ny=8)
    mp = make_regular_grid(0.0, 0.0, 0.6, 0.6, 4, 4)
    parts = list(mp.geoms)
    seq = {
        "list": parts,
        "array": np.asarray(parts, dtype=object),
        "tuple": tuple(parts),
    }
    for a, b in ((rect, mp), (mp, rect), (mp, mp)):
        ref = quickpaver.compute_transfer_matrix(a, b).toarray()
        a2 = seq[kind] if a is mp else a
        b2 = seq[kind] if b is mp else b
        np.testing.assert_array_equal(
            quickpaver.compute_transfer_matrix(a2, b2).toarray(), ref
        )


def test_bad_polygon_grids_raise() -> None:
    rect = RectilinearGrid(nx=2, ny=2)
    with pytest.raises(TypeError, match="shapely.Polygon"):
        quickpaver.compute_transfer_matrix(
            rect, [shapely.Point(0, 0).buffer(1), shapely.Point(1, 1)]
        )
    with pytest.raises(TypeError, match="1-D"):
        quickpaver.compute_transfer_matrix(rect, np.empty((2, 2), dtype=object))


def test_interior_shortcut_is_exact() -> None:
    """Large convex polygons over a fine grid: most cells are interior and
    get their area analytically; the result must match the clipped one."""
    fine = RectilinearGrid(cx=0.0, cy=0.0, dx=0.05, dy=0.04, nx=200, ny=220, theta=8.0)
    big = shapely.MultiPolygon(
        [regular_ngon(-2.0, -1.5, 2.5, 9), regular_ngon(2.5, 2.0, 2.0, 5)]
    )
    W = quickpaver.compute_transfer_matrix(big, fine)
    ref = quickpaver.compute_transfer_matrix(big, fine.to_shapely())
    np.testing.assert_allclose(W.toarray(), ref.toarray(), atol=1e-12)
    # interior cells exist and carry exactly one cell area
    W2 = quickpaver.compute_transfer_matrix(fine, big)
    assert np.sum(np.isclose(W2.data, 1.0, rtol=0, atol=1e-12)) > 1000
    # rotated rect <-> rect: coarse rotated target over a fine source
    coarse = RectilinearGrid(cx=0.2, cy=-0.1, dx=1.3, dy=0.9, nx=6, ny=7, theta=31.0)
    W3 = quickpaver.compute_transfer_matrix(fine, coarse)
    ref3 = quickpaver.compute_transfer_matrix(fine, coarse.to_shapely())
    np.testing.assert_allclose(W3.toarray(), ref3.toarray(), atol=1e-12)


def test_concave_and_holed_cells_without_geos_area() -> None:
    """Concave / holed polygons clipped by convex windows (rect cells or
    triangles) are handled in Rust; their weights match GEOS."""
    poly = [l_shape_polygon(0.1, 0.2, 2.0), square_with_hole(2.6, 0.1, 2.0, 0.3)]
    rect = RectilinearGrid(cx=2.5, cy=1.2, dx=0.3, dy=0.35, nx=18, ny=8, theta=11.0)
    mesh = make_grid_trimesh(0.0, 0.0, 0.4, 0.45, 13, 6)
    for a, b in ((rect, poly), (poly, rect), (mesh, poly), (poly, mesh)):
        np.testing.assert_allclose(
            quickpaver.compute_transfer_matrix(a, b).toarray(),
            _reference(a, b),
            atol=1e-12,
        )


def test_extension_validates_its_inputs() -> None:
    i = np.array([0, 1], dtype=np.intp)
    with pytest.raises(ValueError, match="out of bounds"):
        rs.coo_to_csc(i, np.array([0, 5], dtype=np.intp), np.ones(2), 2, 2)
    with pytest.raises(ValueError, match="length"):
        rs.coo_to_csc(i, i[:1], np.ones(2), 2, 2)
    x = np.zeros(4)
    with pytest.raises(ValueError, match="ragged"):
        rs.polygon_convexity(
            x, x, np.array([0, 9], dtype=np.intp), np.array([0, 1], dtype=np.intp)
        )
    ok = rs.polygon_convexity(
        np.array([0.0, 1.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0, 0.0]),
        np.array([0, 4], dtype=np.intp),
        np.array([0, 1], dtype=np.intp),
    )
    assert ok.tolist() == [True]


def test_concurrent_calls_are_deterministic() -> None:
    """The extension releases the GIL and is free-threading safe."""
    g = _grids()
    pairs = [("rect", "rot"), ("mesh", "concave"), ("concave", "convex")] * 4
    ref = {
        p: quickpaver.compute_transfer_matrix(g[p[0]], g[p[1]]).toarray() for p in pairs
    }
    with ThreadPoolExecutor(max_workers=6) as ex:
        res = list(
            ex.map(
                lambda p: quickpaver.compute_transfer_matrix(g[p[0]], g[p[1]]), pairs
            )
        )
    for p, W in zip(pairs, res):
        np.testing.assert_array_equal(W.toarray(), ref[p])
