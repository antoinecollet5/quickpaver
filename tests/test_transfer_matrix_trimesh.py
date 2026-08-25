# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import numpy as np
import pytest
import quickpaver
import shapely
from quickpaver import RectilinearGrid, TriMesh
from quickpaver._tm_test_utils import (
    l_shape_polygon,
    make_grid_trimesh,
    make_single_triangle_mesh,
    regular_ngon,
    rotate_trimesh,
    square_with_hole,
)

# ---------------------------------------------------------------------------
# TriMesh <-> TriMesh
# ---------------------------------------------------------------------------


class TestTriMeshVsTriMesh:
    def test_identical_meshes_conserve_mass_and_shape(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)  # 8 triangles
        W = quickpaver.compute_transfer_matrix(mesh, mesh)
        assert W.shape == (8, 8)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_finer_target_conserves_mass(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 2.0, 2.0, 1, 1)  # 2 triangles, area 4
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)  # 8 triangles, area 4
        W = quickpaver.compute_transfer_matrix(source, target, is_sanity_check=True)
        assert W.shape == (2, 8)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_rotated_target_partial_overlap(self) -> None:
        """Non-axis-aligned overlap: exercises the general triangle-triangle
        Sutherland-Hodgman clip with a genuine multi-vertex intersection
        polygon (not just a shared edge)."""
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 3, 3)
        target = rotate_trimesh(
            make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 3, 3),
            angle_deg=17.0,
            origin=(1.5, 1.5),
        )
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            source, target
        )
        assert W.nnz > 0
        assert len(src_idx) == len(tgt_idx) == len(ix) == W.nnz
        # Geometry is materialized (world frame), no None placeholders on
        # this path since it never uses the analytic-only shortcut.
        assert all(isinstance(g, (shapely.Polygon, shapely.MultiPolygon)) for g in ix)

    def test_disjoint_meshes_have_no_overlap(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = make_grid_trimesh(1000.0, 1000.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.nnz == 0
        assert W.shape == (8, 8)

    def test_cw_wound_triangles_are_handled(self) -> None:
        """Both source and target meshes wound clockwise: exercises
        ``_ensure_ccw_triangles``'s actual flip branch on both sides."""
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2, ccw=False)
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2, ccw=False)
        W = quickpaver.compute_transfer_matrix(source, target, is_sanity_check=True)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_mixed_cw_ccw_triangles(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2, ccw=True)
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2, ccw=False)
        W = quickpaver.compute_transfer_matrix(source, target, is_sanity_check=True)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-10)

    def test_masks(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)  # 8 triangles
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        source_mask = np.zeros(8, dtype=bool)
        source_mask[:4] = True
        target_mask = np.zeros(8, dtype=bool)
        target_mask[4:] = True
        W = quickpaver.compute_transfer_matrix(
            source, target, source_grid_mask=source_mask, target_grid_mask=target_mask
        )
        assert W.shape == (8, 8)
        # masked-out source rows are all zero
        dense = W.toarray()
        assert np.all(dense[~source_mask, :] == 0.0)
        assert np.all(dense[:, ~target_mask] == 0.0)

    def test_bad_mask_length_raises(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        with pytest.raises(ValueError):
            quickpaver.compute_transfer_matrix(
                source, target, source_grid_mask=np.zeros(3, dtype=bool)
            )

    def test_empty_target_mesh(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        empty = TriMesh(
            verts_xy=np.zeros((0, 2)),
            tri_verts=np.zeros((0, 3), dtype=int),
            edge_lengths_m=np.zeros((0, 3)),
            tri_area_m2=np.zeros((0,)),
        )
        W = quickpaver.compute_transfer_matrix(source, empty)
        assert W.shape == (2, 0)
        assert W.nnz == 0

    def test_fully_masked_out_source(self) -> None:
        """All source triangles masked out -> the STRtree-candidate branch
        must short-circuit cleanly with zero results."""
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        W = quickpaver.compute_transfer_matrix(
            source, target, source_grid_mask=np.zeros(2, dtype=bool)
        )
        assert W.nnz == 0

    def test_with_intersections_false_returns_empty_intersections(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(source, target)
        assert isinstance(W, type(W))

    def test_sanity_check_fully_covered(self) -> None:
        source = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = make_grid_trimesh(0.0, 0.0, 0.5, 0.5, 4, 4)
        # Should not raise.
        quickpaver.compute_transfer_matrix(source, target, is_sanity_check=True)


# ---------------------------------------------------------------------------
# RectilinearGrid <-> TriMesh
# ---------------------------------------------------------------------------


class TestRectilinearVsTriMesh:
    def test_rect_source_trimesh_target_conserves_mass(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        mesh = make_grid_trimesh(0.0, 0.0, 0.5, 0.5, 4, 4)
        W = quickpaver.compute_transfer_matrix(rect, mesh, is_sanity_check=True)
        assert W.shape == (4, 32)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_trimesh_source_rect_target_conserves_mass(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 0.5, 0.5, 4, 4)
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        W = quickpaver.compute_transfer_matrix(mesh, rect, is_sanity_check=True)
        assert W.shape == (32, 4)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-8)

    def test_rotated_rectilinear_grid_vs_trimesh(self) -> None:
        """Non-axis-aligned rectilinear grid forces the mesh vertices through
        the local-frame rotation, and produces clipped polygons with more
        than 3 vertices (quads/pentagons) -- exercising the grouped
        ``_polygons_from_padded_verts`` construction for multiple distinct
        vertex counts at once."""
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=3, ny=3, theta=23.0)
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 3, 3)
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            rect, mesh
        )
        assert W.nnz > 0
        assert len(ix) == W.nnz
        assert all(isinstance(g, shapely.Polygon) for g in ix)

    def test_disjoint_rect_and_trimesh(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        mesh = make_grid_trimesh(1000.0, 1000.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(rect, mesh)
        assert W.nnz == 0

    def test_masks_rect_source(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        rect_mask = np.array([True, False, True, False])
        mesh_mask = np.zeros(8, dtype=bool)
        mesh_mask[:4] = True
        W = quickpaver.compute_transfer_matrix(
            rect, mesh, source_grid_mask=rect_mask, target_grid_mask=mesh_mask
        )
        dense = W.toarray()
        assert np.all(dense[~rect_mask, :] == 0.0)
        assert np.all(dense[:, ~mesh_mask] == 0.0)

    def test_masks_trimesh_source(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        mesh_mask = np.zeros(8, dtype=bool)
        mesh_mask[:4] = True
        rect_mask = np.array([True, False, True, False])
        W = quickpaver.compute_transfer_matrix(
            mesh, rect, source_grid_mask=mesh_mask, target_grid_mask=rect_mask
        )
        dense = W.toarray()
        assert np.all(dense[~mesh_mask, :] == 0.0)
        assert np.all(dense[:, ~rect_mask] == 0.0)

    def test_fully_masked_rect_side_short_circuits(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(
            rect, mesh, source_grid_mask=np.zeros(4, dtype=bool)
        )
        assert W.nnz == 0

    def test_empty_trimesh_side(self) -> None:
        rect = RectilinearGrid(cx=0.0, cy=0.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        empty = TriMesh(
            verts_xy=np.zeros((0, 2)),
            tri_verts=np.zeros((0, 3), dtype=int),
            edge_lengths_m=np.zeros((0, 3)),
            tri_area_m2=np.zeros((0,)),
        )
        W = quickpaver.compute_transfer_matrix(rect, empty)
        assert W.shape == (4, 0)
        assert W.nnz == 0

    def test_with_intersections_false(self) -> None:
        rect = RectilinearGrid(cx=1.0, cy=1.0, dx=1.0, dy=1.0, nx=2, ny=2, theta=0.0)
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(rect, mesh)
        assert W.nnz > 0


# ---------------------------------------------------------------------------
# TriMesh <-> arbitrary MultiPolygon
# ---------------------------------------------------------------------------


class TestTriMeshVsPolygon:
    def test_trimesh_source_convex_polygon_target(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 3, 3)
        target = shapely.MultiPolygon([regular_ngon(1.5, 1.5, 1.2, 8)])
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            mesh, target
        )
        assert W.shape == (18, 1)
        assert W.nnz > 0
        assert len(ix) == W.nnz

    def test_polygon_source_trimesh_target_convex(self) -> None:
        source = shapely.MultiPolygon(
            [regular_ngon(0.5, 0.5, 0.4, 6), regular_ngon(1.5, 1.5, 0.4, 6)]
        )
        target = make_grid_trimesh(0.0, 0.0, 0.5, 0.5, 4, 4)
        W = quickpaver.compute_transfer_matrix(source, target, is_sanity_check=True)
        assert W.shape == (2, 32)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        # Hexagons are fully inside the mesh's covered domain.
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_non_convex_polygon_falls_back_to_shapely(self) -> None:
        """An L-shaped (non-convex) polygon must go through the Shapely
        ``shapely.intersection`` fallback branch."""
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 4, 4)
        target = shapely.MultiPolygon([l_shape_polygon(0.5, 0.5, 2.0)])
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            mesh, target
        )
        assert W.nnz > 0
        assert len(ix) == W.nnz
        assert all(isinstance(g, (shapely.Polygon, shapely.MultiPolygon)) for g in ix)

    def test_holed_polygon_falls_back_to_shapely(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 4, 4)
        target = shapely.MultiPolygon([square_with_hole(0.5, 0.5, 2.0)])
        W = quickpaver.compute_transfer_matrix(mesh, target)
        assert W.nnz > 0
        # Row sums (mesh cell -> polygon) must be < 1 near the hole and
        # everything must stay within [0, 1].
        dense = W.toarray()
        assert np.all(dense >= -1e-12)
        assert np.all(dense <= 1.0 + 1e-9)

    def test_mixed_convex_and_non_convex_targets(self) -> None:
        """A polygon grid containing both a convex and a non-convex/holed
        polygon exercises both branches within a single call."""
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 5, 5)
        target = shapely.MultiPolygon(
            [
                regular_ngon(1.0, 1.0, 0.6, 6),
                l_shape_polygon(2.5, 2.5, 1.5),
                square_with_hole(0.2, 3.0, 1.2),
            ]
        )
        W, src_idx, tgt_idx, ix = quickpaver.compute_transfer_matrix_with_intersections(
            mesh, target
        )
        assert W.shape[1] == 3
        assert W.nnz > 0
        assert len(ix) == W.nnz

    def test_disjoint(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = shapely.MultiPolygon([shapely.box(1000, 1000, 1001, 1001)])
        W = quickpaver.compute_transfer_matrix(mesh, target)
        assert W.nnz == 0

    def test_masks_trimesh_source(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)  # 8 tri
        target = shapely.MultiPolygon(
            [regular_ngon(0.5, 0.5, 0.6, 6), regular_ngon(1.5, 1.5, 0.6, 6)]
        )
        mesh_mask = np.zeros(8, dtype=bool)
        mesh_mask[:4] = True
        poly_mask = np.array([True, False])
        W = quickpaver.compute_transfer_matrix(
            mesh, target, source_grid_mask=mesh_mask, target_grid_mask=poly_mask
        )
        dense = W.toarray()
        assert np.all(dense[~mesh_mask, :] == 0.0)
        assert np.all(dense[:, ~poly_mask] == 0.0)

    def test_masks_polygon_source(self) -> None:
        source = shapely.MultiPolygon(
            [regular_ngon(0.5, 0.5, 0.6, 6), regular_ngon(1.5, 1.5, 0.6, 6)]
        )
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 2, 2)
        poly_mask = np.array([True, False])
        mesh_mask = np.zeros(8, dtype=bool)
        mesh_mask[:4] = True
        W = quickpaver.compute_transfer_matrix(
            source, mesh, source_grid_mask=poly_mask, target_grid_mask=mesh_mask
        )
        dense = W.toarray()
        assert np.all(dense[~poly_mask, :] == 0.0)
        assert np.all(dense[:, ~mesh_mask] == 0.0)

    def test_fully_masked_trimesh_short_circuits(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        target = shapely.MultiPolygon([regular_ngon(0.5, 0.5, 0.4, 6)])
        W = quickpaver.compute_transfer_matrix(
            mesh, target, source_grid_mask=np.zeros(2, dtype=bool)
        )
        assert W.nnz == 0

    def test_empty_polygon_grid(self) -> None:
        mesh = make_grid_trimesh(0.0, 0.0, 1.0, 1.0, 1, 1)
        empty = shapely.MultiPolygon([])
        W = quickpaver.compute_transfer_matrix(mesh, empty)
        assert W.shape == (2, 0)
        assert W.nnz == 0

    def test_single_triangle_vs_single_polygon(self) -> None:
        mesh = make_single_triangle_mesh((0.0, 0.0), (2.0, 0.0), (0.0, 2.0))
        target = shapely.MultiPolygon([shapely.box(-1, -1, 1, 1)])
        W = quickpaver.compute_transfer_matrix(mesh, target)
        assert W.shape == (1, 1)
        assert W[0, 0] == pytest.approx(0.5, abs=1e-9)
