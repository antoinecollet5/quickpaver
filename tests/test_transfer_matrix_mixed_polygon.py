# SPDX-License-Identifier: BSD-3-Clause
"""
Tests for :func:`quickpaver._transfer_matrix._compute_transfer_matrix_mixed`
(RectilinearGrid <-> arbitrary MultiPolygon), reached through the public
:func:`quickpaver.compute_transfer_matrix` /
:func:`quickpaver.compute_transfer_matrix_with_intersections` entry points,
plus a direct white-box call for the ``polygon_grid_is_convex`` shortcut,
which the public API cannot reach.
"""

from __future__ import annotations

import numpy as np
import pytest
import shapely

import quickpaver
from quickpaver import RectilinearGrid
from quickpaver._tm_test_utils import (
    l_shape_polygon,
    make_regular_grid,
    regular_ngon,
    square_with_hole,
)
from quickpaver._transfer_matrix import _compute_transfer_matrix_mixed


class TestRectSourcePolygonTarget:
    def test_convex_polygons_conserve_mass(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.25, dy=0.25, nx=8, ny=8, theta=0.0)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(rect, target, is_sanity_check=True)
        assert W.shape == (64, 4)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_rotated_rect_vs_convex_polygons(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.3, dy=0.3, nx=7, ny=7, theta=13.0)
        target = shapely.MultiPolygon(
            [regular_ngon(1.0, 1.0, 0.9, 5), regular_ngon(2.5, 2.5, 0.5, 7)]
        )
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            rect, target
        )
        assert W.nnz > 0
        assert len(ix) == W.nnz
        # Convex, holeless polygons go through the fast analytic clip, which
        # still materializes real geometry from the already-clipped
        # vertices (a vectorized `shapely.polygons` construction, not a
        # GEOS clip call).
        assert all(isinstance(g, shapely.Polygon) for g in ix)

    def test_non_convex_polygon_uses_shapely_fallback(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.2, dy=0.2, nx=10, ny=10, theta=0.0)
        target = shapely.MultiPolygon([l_shape_polygon(0.0, 0.0, 2.0)])
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            rect, target
        )
        assert W.nnz > 0
        assert len(ix) == W.nnz
        assert any(g is not None for g in ix)

    def test_holed_polygon_uses_shapely_fallback(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.2, dy=0.2, nx=10, ny=10, theta=0.0)
        target = shapely.MultiPolygon([square_with_hole(0.0, 0.0, 2.0)])
        W = quickpaver.compute_transfer_matrix(rect, target)
        assert W.nnz > 0
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        assert np.all(row_sums <= 1.0 + 1e-9)

    def test_mixed_convex_and_non_convex_in_one_call(self) -> None:
        rect = RectilinearGrid(cx=1.5, cy=1.5, dx=0.2, dy=0.2, nx=15, ny=15, theta=0.0)
        target = shapely.MultiPolygon(
            [
                regular_ngon(0.5, 0.5, 0.4, 6),
                l_shape_polygon(1.5, 1.5, 1.2),
                square_with_hole(0.2, 2.2, 1.0),
            ]
        )
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            rect, target
        )
        assert W.shape[1] == 3
        assert W.nnz > 0
        assert len(ix) == W.nnz
        # Both the fast convex-clip branch and the Shapely fallback branch
        # ran within this single call; every entry carries real geometry.
        assert all(isinstance(g, (shapely.Polygon, shapely.MultiPolygon)) for g in ix)

    def test_disjoint(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = shapely.MultiPolygon([shapely.box(1000, 1000, 1001, 1001)])
        W = quickpaver.compute_transfer_matrix(rect, target)
        assert W.nnz == 0

    def test_masks(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        rect_mask = np.array([True, False, True, False])
        poly_mask = np.array([True, True, False, False])
        W = quickpaver.compute_transfer_matrix(
            rect, target, source_grid_mask=rect_mask, target_grid_mask=poly_mask
        )
        dense = W.toarray()
        assert np.all(dense[~rect_mask, :] == 0.0)
        assert np.all(dense[:, ~poly_mask] == 0.0)

    def test_fully_masked_rect_short_circuits(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(
            rect, target, source_grid_mask=np.zeros(4, dtype=bool)
        )
        assert W.nnz == 0

    def test_bad_mask_shape_raises(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        with pytest.raises(ValueError):
            quickpaver.compute_transfer_matrix(
                rect, target, target_grid_mask=np.zeros(99, dtype=bool)
            )

    def test_empty_polygon_grid(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        empty = shapely.MultiPolygon([])
        W = quickpaver.compute_transfer_matrix(rect, empty)
        assert W.shape == (4, 0)
        assert W.nnz == 0

    def test_with_intersections_false(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(rect, target)
        assert W.nnz > 0


class TestPolygonSourceRectTarget:
    def test_convex_polygons_conserve_mass(self) -> None:
        source = shapely.MultiPolygon(
            [regular_ngon(0.5, 0.5, 0.4, 6), regular_ngon(1.5, 1.5, 0.4, 6)]
        )
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.1, dy=0.1, nx=20, ny=20, theta=0.0)
        W = quickpaver.compute_transfer_matrix(source, rect, is_sanity_check=True)
        assert W.shape == (2, 400)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_non_convex_polygon_source(self) -> None:
        source = shapely.MultiPolygon([l_shape_polygon(0.0, 0.0, 2.0)])
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.2, dy=0.2, nx=10, ny=10, theta=0.0)
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            source, rect
        )
        assert W.nnz > 0
        assert len(ix) == W.nnz

    def test_masks(self) -> None:
        source = shapely.MultiPolygon(
            [regular_ngon(0.5, 0.5, 0.4, 6), regular_ngon(1.5, 1.5, 0.4, 6)]
        )
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        poly_mask = np.array([True, False])
        rect_mask = np.array([True, True, False, False])
        W = quickpaver.compute_transfer_matrix(
            source, rect, source_grid_mask=poly_mask, target_grid_mask=rect_mask
        )
        dense = W.toarray()
        assert np.all(dense[~poly_mask, :] == 0.0)
        assert np.all(dense[:, ~rect_mask] == 0.0)

    def test_disjoint(self) -> None:
        source = shapely.MultiPolygon([shapely.box(1000, 1000, 1001, 1001)])
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        W = quickpaver.compute_transfer_matrix(source, rect)
        assert W.nnz == 0


class TestManySidedPolygons:
    """Polygons of any vertex count are clipped exactly (no fixed-size
    buffer limit, which made v0.6.0 raise for more than 28 vertices)."""

    def test_many_sided_convex_polygon(self) -> None:
        """A 40-gon (e.g. a disk-clipped tile) next to small polygons."""
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=3, ny=3, theta=0.0)
        huge = regular_ngon(0.2, -0.1, 1.2, 40)  # overlaps several cells
        far = regular_ngon(1000.0, 1000.0, 5.0, 40)  # far away, disjoint
        small = regular_ngon(0.0, 0.0, 0.4, 6)  # overlaps the grid
        target = shapely.MultiPolygon([huge, far, small])
        mat = quickpaver.compute_transfer_matrix(rect, target)
        ref = quickpaver.compute_transfer_matrix(rect.to_shapely(), target)
        np.testing.assert_allclose(mat.toarray(), ref.toarray(), atol=1e-12)


class TestPolygonGridIsConvexShortcut:
    """``polygon_grid_is_convex=True`` skips the convexity test entirely.
    It is never set by the public dispatch (every caller lets convexity be
    auto-detected), so it is only reachable via a direct call to the
    private implementation -- exercised here as a white-box test."""

    def test_skips_convexity_test_and_still_clips_correctly(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=0.25, dy=0.25, nx=8, ny=8, theta=0.0)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        mat, src_idx, tgt_idx, ix = _compute_transfer_matrix_mixed(
            rect,
            target,
            rectilinear_is_source=True,
            is_sanity_check=True,
            with_intersections=True,
            polygon_grid_is_convex=True,
        )
        assert mat.shape == (64, 4)
        row_sums = np.asarray(mat.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)
        assert len(ix) == mat.nnz
