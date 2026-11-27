# SPDX-License-Identifier: BSD-3-Clause
"""Tests of the cell roundness quality (``multipolygon_quality``) and of its
closed forms for stretched regular cells."""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely

import quickpaver as qp
from quickpaver._quality import _vertex_buffer


def _reference_quality(ring: np.ndarray, center=None) -> float:
    """Straightforward (slow) implementation of the metric, on an open
    ``(n, 2)`` ring."""
    n = len(ring)
    if n == 3:
        a = np.linalg.norm(ring[1] - ring[2])
        b = np.linalg.norm(ring[2] - ring[0])
        c = np.linalg.norm(ring[0] - ring[1])
        s = (a + b + c) / 2
        area = math.sqrt(max(s * (s - a) * (s - b) * (s - c), 0.0))
        if area < 1e-15:
            return 0.0
        return 2.0 * (area / s) / (a * b * c / (4 * area))
    if center is None:
        # centroid relative to the first vertex (exact up to rounding)
        q = ring - ring[0]
        x, y = q[:, 0], q[:, 1]
        x1, y1 = np.roll(x, -1), np.roll(y, -1)
        cr = x * y1 - x1 * y
        a2 = cr.sum()
        center = ring[0] + [
            ((x + x1) * cr).sum() / (3 * a2),
            ((y + y1) * cr).sum() / (3 * a2),
        ]
    p = ring - center
    r_circ = np.linalg.norm(p, axis=1).max()
    e = np.roll(p, -1, axis=0) - p
    t = np.clip(-(p * e).sum(1) / (e * e).sum(1), 0, 1)
    r_in = np.linalg.norm(p + t[:, None] * e, axis=1).min()
    return (r_in / r_circ) / math.cos(math.pi / n)


def _regular(n: int, r: float = 1.0, x0: float = 0.0, y0: float = 0.0, phase=0.0):
    t = 2 * np.pi * np.arange(n) / n + phase
    return np.c_[x0 + r * np.cos(t), y0 + r * np.sin(t)]


def _stretch(v: np.ndarray, a: float) -> np.ndarray:
    return v * [np.sqrt(a), 1.0 / np.sqrt(a)]


def _voronoi(n: int = 3000, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    pts = rng.uniform(0, 100, (n, 2))
    box = shapely.box(0, 0, 100, 100)
    cells = shapely.get_parts(
        shapely.voronoi_polygons(shapely.MultiPoint(pts), extend_to=box)
    )
    cells = shapely.intersection(cells, box)
    return cells[shapely.get_type_id(cells) == shapely.GeometryType.POLYGON]


@pytest.mark.parametrize("n", range(3, 13))
@pytest.mark.parametrize("offset", [0.0, 1e6])
def test_regular_polygons_score_one(n: int, offset: float) -> None:
    ring = _regular(n, 2.5, offset, -3 * offset, phase=0.3)
    assert qp.multipolygon_quality(ring[None])[0] == pytest.approx(1.0, abs=1e-9)
    assert qp.multipolygon_quality([shapely.Polygon(ring)])[0] == pytest.approx(
        1.0, abs=1e-9
    )


@pytest.mark.parametrize("a", [0.2, 0.5, 0.9, 1.0, 1.3, 2.0, 5.0])
def test_closed_forms_match_numeric(a: float) -> None:
    cases = [
        (3, 0.0, qp.triangle_anisotropy_quality),
        (4, 0.0, qp.square_anisotropy_quality),
        (6, 0.0, qp.hexagon_anisotropy_quality),
    ]
    for n, phase, f in cases:
        ring = _stretch(_regular(n, phase=phase), a)
        assert qp.multipolygon_quality(ring[None])[0] == pytest.approx(f(a), abs=1e-12)


def test_hexagon_closed_form_branches() -> None:
    # a >= 1: 1 / a (horizontal edges nearest), a <= 1: 4a / (a^2 + 3)
    assert qp.hexagon_anisotropy_quality(2.0) == pytest.approx(0.5)
    assert qp.hexagon_anisotropy_quality(4.0) == pytest.approx(0.25)
    assert qp.hexagon_anisotropy_quality(0.5) == pytest.approx(8 / 13)
    assert qp.square_anisotropy_quality(3.0) == pytest.approx(
        qp.square_anisotropy_quality(1 / 3)
    )
    assert qp.triangle_anisotropy_quality(2.0) != pytest.approx(
        qp.triangle_anisotropy_quality(0.5)
    )


@pytest.mark.parametrize(
    "f",
    [
        qp.triangle_anisotropy_quality,
        qp.square_anisotropy_quality,
        qp.hexagon_anisotropy_quality,
    ],
)
@pytest.mark.parametrize("a", [0.0, -1.0, float("nan")])
def test_closed_forms_reject_bad_anisotropy(f, a: float) -> None:
    with pytest.raises(ValueError, match="anisotropy"):
        f(a)


@pytest.mark.parametrize(
    "poly_type, closed_form, use_inverse",
    [
        (qp.PolygonType.TRIANGLE, qp.triangle_anisotropy_quality, False),
        (qp.PolygonType.RECTANGLE, qp.square_anisotropy_quality, False),
        # the tiling hexagons are rotated by 90 deg w.r.t. the reference one
        (qp.PolygonType.HEXAGON, qp.hexagon_anisotropy_quality, True),
    ],
)
@pytest.mark.parametrize("a", [0.5, 1.0, 2.0])
def test_tiling_quality_follows_closed_forms(
    poly_type, closed_form, use_inverse, a
) -> None:
    verts, _, _ = qp.gen_polygonal_tiling_arrays(
        shapely.box(0, 0, 40, 40), poly_type, 2.0, anisotropy_ratio=a
    )
    q = qp.multipolygon_quality(verts)
    expected = closed_form(1 / a if use_inverse else a)
    np.testing.assert_allclose(q, expected, atol=1e-12, rtol=0)


def test_voronoi_matches_reference() -> None:
    cells = _voronoi()
    q = qp.multipolygon_quality(cells)
    rings = [shapely.get_coordinates(c.exterior)[:-1] for c in cells]
    ref = np.array([_reference_quality(r) for r in rings])
    np.testing.assert_allclose(q, ref, atol=1e-12, rtol=0)
    assert np.all((q > 0) & (q <= 1 + 1e-12))
    # explicit centres (e.g. Voronoi seeds)
    centers = shapely.get_coordinates(shapely.point_on_surface(cells))
    qc = qp.multipolygon_quality(cells, centers=centers)
    ref_c = np.array([_reference_quality(r, c) for r, c in zip(rings, centers)])
    np.testing.assert_allclose(qc, ref_c, atol=1e-12, rtol=0)


def test_triangles_match_reference() -> None:
    rng = np.random.default_rng(1)
    tri = rng.uniform(-5, 5, (2000, 3, 2))
    q = qp.multipolygon_quality(tri)
    ref = np.array([_reference_quality(t) for t in tri])
    np.testing.assert_allclose(q, ref, atol=1e-10, rtol=0)
    # centres are ignored for triangles
    np.testing.assert_array_equal(
        qp.multipolygon_quality(tri, centers=rng.normal(size=(2000, 2))), q
    )


def test_far_from_origin_keeps_precision() -> None:
    cells = _voronoi(1500, seed=4)
    q0 = qp.multipolygon_quality(cells)
    moved = shapely.transform(cells, lambda c: c + [6.5e5, 4.8e6])
    np.testing.assert_allclose(qp.multipolygon_quality(moved), q0, atol=1e-8, rtol=0)


def test_input_types_agree() -> None:
    cells = _voronoi(500, seed=2)
    mp = shapely.MultiPolygon(list(cells))
    ref = qp.multipolygon_quality(cells)
    for x in (mp, mp.geoms, list(cells), tuple(cells), cells.reshape(-1, 1)):
        np.testing.assert_array_equal(qp.multipolygon_quality(x), ref)
    one = qp.multipolygon_quality(cells[0])
    assert one.shape == (1,) and one[0] == ref[0]
    tri = np.random.default_rng(3).uniform(0, 1, (50, 3, 2))
    as_polys = shapely.polygons(np.concatenate([tri, tri[:, :1]], axis=1))
    np.testing.assert_array_equal(
        qp.multipolygon_quality(tri), qp.multipolygon_quality(as_polys)
    )
    np.testing.assert_array_equal(
        qp.multipolygon_quality(tri.astype(np.float32)),
        qp.multipolygon_quality(tri.astype(np.float32).astype(float)),
    )


def test_holes_are_ignored() -> None:
    shell = [(0, 0), (4, 0), (4, 3), (0, 3)]
    plain = shapely.Polygon(shell)
    holed = shapely.Polygon(
        shell, [[(1, 1), (2, 1), (2, 2)], [(3, 1), (3.5, 1), (3, 2)]]
    )
    tri = shapely.Polygon([(0, 0), (1, 0), (0, 1)])
    q = qp.multipolygon_quality([plain, holed, tri, holed, plain])
    np.testing.assert_array_equal(q[[1, 3, 4]], q[0])
    assert q[2] == pytest.approx(_reference_quality(np.array([(0, 0), (1, 0), (0, 1)])))
    buf, off, closed = _vertex_buffer([holed, tri])
    assert closed and list(np.diff(off)) == [5, 4] and len(buf) == 2 * 9


def test_degenerate_and_empty_cells() -> None:
    flat = shapely.Polygon([(0, 0), (1, 0), (2, 0)])
    point_like = shapely.Polygon([(1, 1), (1, 1), (1, 1), (1, 1)])
    q = qp.multipolygon_quality([flat, shapely.Polygon(), None, point_like])
    assert q[0] == 0.0 and np.isnan(q[1]) and np.isnan(q[2]) and q[3] == 0.0
    assert qp.multipolygon_quality([]).shape == (0,)
    assert qp.multipolygon_quality(np.empty((0, 4, 2))).shape == (0,)
    assert qp.multipolygon_quality(shapely.MultiPolygon()).shape == (0,)


def test_parallel_and_serial_agree() -> None:
    cells = _voronoi(6000, seed=5)
    np.testing.assert_array_equal(
        qp.multipolygon_quality(cells, parallel=True),
        qp.multipolygon_quality(cells, parallel=False),
    )


def test_errors() -> None:
    cells = _voronoi(20)
    with pytest.raises(ValueError, match="centers must have shape"):
        qp.multipolygon_quality(cells, centers=np.zeros((len(cells) + 1, 2)))
    with pytest.raises(ValueError, match="centers must have shape"):
        qp.multipolygon_quality(cells, centers=np.zeros(len(cells)))
    with pytest.raises(TypeError, match="Polygon"):
        qp.multipolygon_quality([cells[0], shapely.Point(0, 0)])
    with pytest.raises(TypeError, match="Polygon"):
        qp.multipolygon_quality([shapely.MultiPolygon([cells[0]])])
    with pytest.raises(ValueError, match="shape"):
        qp.multipolygon_quality(np.zeros((3, 4)))
    with pytest.raises(ValueError, match="shape"):
        qp.multipolygon_quality(np.zeros((3, 4, 3)))


def test_rust_kernel_validates_offsets() -> None:
    from quickpaver import _quickpaver as rs

    xy = np.zeros(8)
    with pytest.raises(ValueError, match="offsets"):
        rs.cell_quality(xy, np.array([0, 5], dtype=np.intp), False)
    with pytest.raises(ValueError, match="offsets"):
        rs.cell_quality(xy, np.array([2, 1], dtype=np.intp), False)
    with pytest.raises(ValueError, match="pairs"):
        rs.cell_quality(np.zeros(7), np.array([0, 3], dtype=np.intp), False)
    with pytest.raises(ValueError, match="centers"):
        rs.cell_quality(xy, np.array([0, 4], dtype=np.intp), False, np.zeros(4))
