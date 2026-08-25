# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
import quickpaver
import shapely
from quickpaver import RectilinearGrid, TriMesh
from quickpaver._transfer_matrix import (
    _bbox_candidate_ranges_in_rectilinear_grid,
    _check_intersections_alignment,
    _compute_1d_overlaps,
    _no_intersections,
    _nonseparable_transfer,
    _pack_intersections,
    _polygons_from_padded_verts,
    _PolygonVertexData,
    _validate_mask,
)
from scipy.sparse import csc_array

from _tm_test_utils import (
    make_grid_trimesh,
    make_regular_grid,
    regular_ngon,
    square_with_hole,
)

# ---------------------------------------------------------------------------
# Masks on paths the original test files never exercised
# ---------------------------------------------------------------------------


class TestGenericPolygonMasks:
    def test_masks_restrict_rows_and_columns(self) -> None:
        source = make_regular_grid(0.0, 0.0, 1.0, 1.0, 3, 1)  # 3 cells
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 3, 1)  # 3 cells
        source_mask = np.array([True, False, True])
        target_mask = np.array([False, True, True])
        W = quickpaver.compute_transfer_matrix(
            source, target, source_grid_mask=source_mask, target_grid_mask=target_mask
        )
        dense = W.toarray()
        assert np.all(dense[~source_mask, :] == 0.0)
        assert np.all(dense[:, ~target_mask] == 0.0)
        # cell 0 (kept) maps only to itself, and only column 0 is masked
        # out there, so row 0 must be all zero.
        assert dense[0, :].sum() == 0.0
        # cell 2 (kept) maps to itself and column 2 is kept -> weight 1.
        assert dense[2, 2] == pytest.approx(1.0)

    def test_with_intersections_true(self) -> None:
        source = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = make_regular_grid(0.0, 0.0, 0.5, 0.5, 4, 4)
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            source, target
        )
        assert W.nnz > 0
        assert len(src_idx) == len(tgt_idx) == len(ix) == W.nnz
        assert all(isinstance(g, shapely.Polygon) for g in ix)

    def test_bad_mask_length_raises(self) -> None:
        source = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        with pytest.raises(ValueError):
            quickpaver.compute_transfer_matrix(
                source, target, target_grid_mask=np.zeros(1, dtype=bool)
            )


class TestRectilinearMasks:
    def test_masks_with_intersections_true_separable(self) -> None:
        """Exercises masking applied in lockstep to ``intersections`` on
        the fast separable rect<->rect path (``with_intersections=True``
        AND masks together in the same call)."""
        source = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = RectilinearGrid(cx=1.0, cy=1.0, dx=0.5, dy=0.5, nx=4, ny=4, theta=0.0)
        source_mask = np.array([True, False, True, False])
        target_mask = np.ones(16, dtype=bool)
        target_mask[0] = False
        (
            W,
            src_idx,
            tgt_idx,
            ix,
        ) = quickpaver.compute_transfer_matrix_with_intersections(
            source,
            target,
            source_grid_mask=source_mask,
            target_grid_mask=target_mask,
        )
        assert len(src_idx) == len(tgt_idx) == len(ix) == W.nnz
        dense = W.toarray()
        assert np.all(dense[~source_mask, :] == 0.0)
        assert np.all(dense[:, ~target_mask] == 0.0)
        assert all(isinstance(g, shapely.Polygon) for g in ix)

    def test_masks_with_intersections_true_nonseparable(self) -> None:
        source = RectilinearGrid(cx=1.0, cy=1.0, dx=0.5, dy=0.5, nx=4, ny=4, theta=0.0)
        target = RectilinearGrid(cx=1.0, cy=1.0, dx=0.5, dy=0.5, nx=4, ny=4, theta=17.0)
        source_mask = np.ones(16, dtype=bool)
        source_mask[::2] = False
        (
            W,
            src_idx,
            tgt_idx,
            ix,
        ) = quickpaver.compute_transfer_matrix_with_intersections(
            source, target, source_grid_mask=source_mask
        )
        assert len(src_idx) == len(tgt_idx) == len(ix) == W.nnz
        dense = W.toarray()
        assert np.all(dense[~source_mask, :] == 0.0)

    def test_with_intersections_true_no_masks_separable(self) -> None:
        source = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        target = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        W, si, ti, ix = quickpaver.compute_transfer_matrix_with_intersections(
            source, target
        )
        assert W.nnz == 4
        assert len(ix) == 4
        assert all(isinstance(g, shapely.Polygon) for g in ix)

    def test_with_intersections_true_nonseparable_no_masks(self) -> None:
        source = RectilinearGrid(cx=0.0, cy=0.0, dx=0.5, dy=0.5, nx=6, ny=6, theta=0.0)
        target = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=6, ny=6, theta=17.0)
        W, si, ti, ix = quickpaver.compute_transfer_matrix_with_intersections(
            source, target
        )
        assert W.nnz > 0
        assert len(ix) == W.nnz
        assert all(isinstance(g, shapely.Polygon) for g in ix)


class TestPolygonSourceTriMeshTargetWithIntersections:
    """Dispatch branch: source=arbitrary polygon, target=TriMesh, with
    per-pair geometry requested (a combination the other test files did
    not exercise together)."""

    def test_with_intersections_true(self) -> None:
        source = shapely.MultiPolygon(
            [regular_ngon(0.5, 0.5, 0.4, 6), regular_ngon(1.5, 1.5, 0.4, 6)]
        )
        target = make_grid_trimesh(0.0, 0.0, 0.5, 0.5, 4, 4)
        (
            W,
            src_idx,
            tgt_idx,
            ix,
        ) = quickpaver.compute_transfer_matrix_with_intersections(source, target)
        assert W.nnz > 0
        assert len(src_idx) == len(tgt_idx) == len(ix) == W.nnz


# ---------------------------------------------------------------------------
# Touching-only geometry: passes the cheap bbox candidate search, must be
# rejected by the exact-overlap filter that follows it.
# ---------------------------------------------------------------------------


class TestTouchingOnlyGeometry:
    def test_rect_vs_polygon_edge_touch_only(self) -> None:
        """Rect<->polygon mixed path: candidate found via O(1) index
        arithmetic, then rejected by the nontrivial-bbox-overlap filter."""
        rect = RectilinearGrid(cx=0.5, cy=0.5, dx=1.0, dy=1.0, nx=1, ny=1, theta=0.0)
        target = shapely.MultiPolygon([shapely.box(1.0, 0.0, 2.0, 1.0)])
        W = quickpaver.compute_transfer_matrix(rect, target)
        assert W.nnz == 0

    def test_rect_vs_trimesh_edge_touch_only(self) -> None:
        """Rect<->TriMesh dedicated path: same idea, triangle sharing only
        an edge with the cell."""
        rect = RectilinearGrid(cx=0.5, cy=0.5, dx=1.0, dy=1.0, nx=1, ny=1, theta=0.0)
        mesh = make_grid_trimesh(1.0, 0.0, 1.0, 1.0, 1, 1)  # triangles at x in [1,2]
        W = quickpaver.compute_transfer_matrix(rect, mesh)
        assert W.nnz == 0

    def test_rect_vs_trimesh_bbox_overlap_but_zero_exact_area(self) -> None:
        """A triangle whose bounding box overlaps the cell (passing the
        nontrivial-bbox filter) but whose actual shape does not reach the
        cell at all -- the exact clip area is zero, so this must be
        rejected by the *second* (post-clip) empty-result guard rather
        than the bbox one."""
        rect = RectilinearGrid(cx=0.5, cy=0.5, dx=1.0, dy=1.0, nx=1, ny=1, theta=0.0)
        mesh = TriMesh(
            verts_xy=np.array([[0.99, 2.0], [2.0, 2.0], [2.0, 0.99]]),
            tri_verts=np.array([[0, 1, 2]]),
            edge_lengths_m=np.zeros((1, 3)),
            tri_area_m2=np.array([0.5 * 1.01 * 1.01]),
        )
        # Sanity: the triangle's bbox does overlap the unit cell [0,1]^2.
        assert np.min(mesh.verts_xy[:, 0]) < 1.0
        assert np.min(mesh.verts_xy[:, 1]) < 1.0
        W = quickpaver.compute_transfer_matrix(rect, mesh)
        assert W.nnz == 0

    def test_trimesh_vs_polygon_edge_touch_only(self) -> None:
        """TriMesh<->polygon path: STRtree finds the bbox-touching pair,
        then the nontrivial-bbox filter must reject it."""
        mesh = TriMesh(
            verts_xy=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            tri_verts=np.array([[0, 1, 2]]),
            edge_lengths_m=np.zeros((1, 3)),
            tri_area_m2=np.array([0.5]),
        )
        target = shapely.MultiPolygon([shapely.box(1.0, 0.0, 2.0, 1.0)])
        W = quickpaver.compute_transfer_matrix(mesh, target)
        assert W.nnz == 0

    def test_trimesh_vs_trimesh_edge_touch_only(self) -> None:
        source = TriMesh(
            verts_xy=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]]),
            tri_verts=np.array([[0, 1, 2]]),
            edge_lengths_m=np.zeros((1, 3)),
            tri_area_m2=np.array([0.5]),
        )
        target = TriMesh(
            verts_xy=np.array([[1.0, 0.0], [2.0, 0.0], [1.0, 1.0]]),
            tri_verts=np.array([[0, 1, 2]]),
            edge_lengths_m=np.zeros((1, 3)),
            tri_area_m2=np.array([0.5]),
        )
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.nnz == 0

    def test_generic_polygon_vs_polygon_corner_touch_only(self) -> None:
        source = shapely.MultiPolygon([shapely.box(0, 0, 1, 1)])
        target = shapely.MultiPolygon([shapely.box(1, 1, 2, 2)])
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.nnz == 0


# ---------------------------------------------------------------------------
# Direct / white-box tests of private helpers whose defensive branches are
# not reachable through the public dispatch with valid grid objects.
# ---------------------------------------------------------------------------


class TestValidateMaskDirect:
    def test_none_passthrough(self) -> None:
        assert _validate_mask(None, 5, "m") is None

    def test_valid_mask_returned(self) -> None:
        m = np.array([True, False, True])
        out = _validate_mask(m, 3, "m")
        np.testing.assert_array_equal(out, m)

    def test_wrong_length_raises(self) -> None:
        with pytest.raises(ValueError, match="m must be a 1-D boolean array"):
            _validate_mask(np.zeros(2, dtype=bool), 5, "m")

    def test_wrong_ndim_raises(self) -> None:
        with pytest.raises(ValueError):
            _validate_mask(np.zeros((2, 2), dtype=bool), 4, "m")


class TestComputeOneDOverlapsDirect:
    def test_empty_a_axis_returns_empty(self) -> None:
        # A single edge value -> zero cells along this axis (na = 0).
        edges_a = np.array([0.0])
        edges_b = np.array([0.0, 1.0, 2.0])
        idx_a, idx_b, overlaps, lo, hi = _compute_1d_overlaps(edges_a, edges_b)
        assert len(idx_a) == len(idx_b) == len(overlaps) == len(lo) == len(hi) == 0

    def test_empty_b_axis_returns_empty(self) -> None:
        edges_a = np.array([0.0, 1.0])
        edges_b = np.array([0.0])
        idx_a, idx_b, overlaps, lo, hi = _compute_1d_overlaps(edges_a, edges_b)
        assert len(idx_a) == 0

    def test_normal_case_still_works(self) -> None:
        edges_a = np.array([0.0, 1.0, 2.0])
        edges_b = np.array([0.5, 1.5])
        idx_a, idx_b, overlaps, lo, hi = _compute_1d_overlaps(edges_a, edges_b)
        assert len(idx_a) > 0


class TestNonseparableTransferDirect:
    def test_empty_target_grid_returns_empty(self) -> None:
        """``target_nx=0`` forces the flattened target-cell arrays to be
        empty, so the total-candidate-pairs count is zero -- a state that
        cannot occur through ``RectilinearGrid`` (which enforces
        ``nx, ny >= 1``), so it is exercised directly here."""
        src_lin, tgt_lin, weights, intersections = _nonseparable_transfer(
            source_center=np.array([0.0, 0.0]),
            source_dx=1.0,
            source_dy=1.0,
            source_nx=3,
            source_ny=3,
            source_angle=0.0,
            target_center=np.array([0.0, 0.0]),
            target_dx=1.0,
            target_dy=1.0,
            target_nx=0,
            target_ny=3,
            target_angle=np.deg2rad(20.0),
            with_intersections=True,
        )
        assert len(src_lin) == 0
        assert len(tgt_lin) == 0
        assert len(weights) == 0
        assert len(intersections) == 0


class TestPolygonVertexDataDirect:
    def test_empty_batch(self) -> None:
        pvd = _PolygonVertexData(np.asarray([], dtype=object))
        assert len(pvd.x) == 0
        assert len(pvd.starts) == 0
        assert pvd.max_verts == 0

        bx0, bx1, by0, by1 = pvd.local_frame_bboxes(np.array([0.0, 0.0]), 0.0)
        assert len(bx0) == len(bx1) == len(by0) == len(by1) == 0

        vx, vy = pvd.padded_vertex_buffers(8)
        assert vx.shape == (0, 8)
        assert vy.shape == (0, 8)

        conv = pvd.convexity()
        assert len(conv) == 0

    def test_nonempty_batch_basic_fields(self) -> None:
        polys = np.asarray(
            [shapely.box(0, 0, 1, 1), shapely.Polygon([(0, 0), (2, 0), (1, 1)])],
            dtype=object,
        )
        pvd = _PolygonVertexData(polys)
        assert pvd.max_verts == 4
        np.testing.assert_array_equal(pvd.true_n, [4, 3])
        assert not pvd.has_holes.any()
        conv = pvd.convexity()
        assert conv.all()

    def test_holed_polygon_is_not_convex(self) -> None:
        polys = np.asarray([square_with_hole(0.0, 0.0, 2.0)], dtype=object)
        pvd = _PolygonVertexData(polys)
        assert pvd.has_holes[0]
        assert not pvd.convexity()[0]


class TestBboxCandidateRangesDirect:
    def test_empty_items_returns_empty(self) -> None:
        item_idx, cell_i, cell_j = _bbox_candidate_ranges_in_rectilinear_grid(
            bbox_xmin=np.array([]),
            bbox_xmax=np.array([]),
            bbox_ymin=np.array([]),
            bbox_ymax=np.array([]),
            x_edges=np.array([0.0, 1.0, 2.0]),
            y_edges=np.array([0.0, 1.0, 2.0]),
            nx=2,
            ny=2,
        )
        assert len(item_idx) == len(cell_i) == len(cell_j) == 0

    def test_normal_case(self) -> None:
        item_idx, cell_i, cell_j = _bbox_candidate_ranges_in_rectilinear_grid(
            bbox_xmin=np.array([0.2]),
            bbox_xmax=np.array([0.8]),
            bbox_ymin=np.array([0.2]),
            bbox_ymax=np.array([0.8]),
            x_edges=np.array([0.0, 1.0, 2.0]),
            y_edges=np.array([0.0, 1.0, 2.0]),
            nx=2,
            ny=2,
        )
        assert len(item_idx) == 1
        assert cell_i[0] == 0
        assert cell_j[0] == 0


class TestPolygonsFromPaddedVertsDirect:
    def test_all_invalid_returns_all_none(self) -> None:
        n_verts = np.array([0, 1, 2])
        vx = np.zeros((3, 4))
        vy = np.zeros((3, 4))
        polys = _polygons_from_padded_verts(vx, vy, n_verts)
        assert len(polys) == 3
        assert all(p is None for p in polys)

    def test_mixed_valid_and_invalid(self) -> None:
        n_verts = np.array([0, 4])
        vx = np.zeros((2, 4))
        vy = np.zeros((2, 4))
        vx[1] = [0.0, 1.0, 1.0, 0.0]
        vy[1] = [0.0, 0.0, 1.0, 1.0]
        polys = _polygons_from_padded_verts(vx, vy, n_verts)
        assert polys[0] is None
        assert isinstance(polys[1], shapely.Polygon)
        assert polys[1].area == pytest.approx(1.0)

    def test_world_transform_applied(self) -> None:
        n_verts = np.array([4])
        vx = np.array([[0.0, 1.0, 1.0, 0.0]])
        vy = np.array([[0.0, 0.0, 1.0, 1.0]])
        polys = _polygons_from_padded_verts(
            vx, vy, n_verts, local_to_world=(np.array([10.0, 20.0]), 1.0, 0.0)
        )
        cx, cy = polys[0].centroid.x, polys[0].centroid.y
        assert cx == pytest.approx(10.5)
        assert cy == pytest.approx(20.5)


class TestNoIntersectionsAndPack:
    def test_no_intersections_is_empty_object_array(self) -> None:
        out = _no_intersections()
        assert len(out) == 0
        assert out.dtype == object

    def test_pack_intersections_normalizes_to_object_array(self) -> None:
        polys = [shapely.box(0, 0, 1, 1), None]
        out = _pack_intersections(np.array(polys))
        assert out.dtype == object
        assert len(out) == 2


class TestCheckIntersectionsAlignmentDirect:
    def test_matching_lengths_does_not_raise(self) -> None:
        mat = csc_array(([1.0], ([0], [0])), shape=(1, 1))
        _check_intersections_alignment(mat, np.array([None], dtype=object))

    def test_mismatched_lengths_raises(self) -> None:
        mat = csc_array(([1.0], ([0], [0])), shape=(1, 1))
        with pytest.raises(AssertionError, match="misalignment"):
            _check_intersections_alignment(mat, np.empty(0, dtype=object))
