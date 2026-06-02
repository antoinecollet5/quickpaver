# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET
"""
Tests for :func:`quickpaver.compute_transfer_matrix`.

Coverage targets
----------------
- Normal overlapping grids (identical, shifted, partial overlap).
- Conservation property (row sums ≈ 1 for fully covered sources).
- Empty intersection (disjoint grids).
- Numerical-noise filtering (``intersection_areas > 1e-15``).
- Sanity check pass and fail paths.
- Non-square grid shapes (n_source ≠ n_target).
- Single-polygon edge case.

"""

from __future__ import annotations

import numpy as np
import pytest
import quickpaver
import shapely
from scipy.sparse import csc_array

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_regular_grid(
    x0: float,
    y0: float,
    dx: float,
    dy: float,
    nx: int,
    ny: int,
) -> shapely.MultiPolygon:
    """Build an axis-aligned rectangular grid as a ``MultiPolygon``.

    Parameters
    ----------
    x0, y0 : float
        Lower-left corner of the grid.
    dx, dy : float
        Cell size in x and y.
    nx, ny : int
        Number of cells in x and y.

    Returns
    -------
    shapely.MultiPolygon
        Grid cells as a collection of rectangular polygons.
    """
    boxes = shapely.box(
        np.repeat(np.arange(nx) * dx + x0, ny),
        np.tile(np.arange(ny) * dy + y0, nx),
        np.repeat(np.arange(nx) * dx + x0 + dx, ny),
        np.tile(np.arange(ny) * dy + y0 + dy, nx),
    )
    return shapely.MultiPolygon(boxes)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def grid_2x2() -> shapely.MultiPolygon:
    """A 2×2 unit grid at the origin."""
    return _make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)


@pytest.fixture()
def grid_4x4_fine() -> shapely.MultiPolygon:
    """A 4×4 grid with cells half the size, covering the same area as grid_2x2."""
    return _make_regular_grid(0.0, 0.0, 0.5, 0.5, 4, 4)


@pytest.fixture()
def grid_2x2_shifted() -> shapely.MultiPolygon:
    """A 2×2 unit grid shifted by (0.5, 0.5) — partially overlaps grid_2x2."""
    return _make_regular_grid(0.5, 0.5, 1.0, 1.0, 2, 2)


@pytest.fixture()
def grid_2x2_disjoint() -> shapely.MultiPolygon:
    """A 2×2 unit grid far away — no overlap with grid_2x2."""
    return _make_regular_grid(100.0, 100.0, 1.0, 1.0, 2, 2)


# ---------------------------------------------------------------------------
# Tests — basic properties
# ---------------------------------------------------------------------------


class TestIdenticalGrids:
    """Source and target are the same grid → transfer matrix is identity."""

    def test_shape(self, grid_2x2: shapely.MultiPolygon) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2)
        assert W.shape == (4, 4)

    def test_is_identity(self, grid_2x2: shapely.MultiPolygon) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2)
        np.testing.assert_allclose(W.toarray(), np.eye(4), atol=1e-12)

    def test_row_sums_are_one(self, grid_2x2: shapely.MultiPolygon) -> None:
        """Conservation: each source cell maps 100 % of its area."""
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_returns_csc(self, grid_2x2: shapely.MultiPolygon) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2)
        assert isinstance(W, csc_array)


class TestCoarseToFine:
    """Source is coarse (2×2), target is fine (4×4)."""

    def test_shape(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_4x4_fine)
        assert W.shape == (4, 16)

    def test_row_sums(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """Each coarse cell is fully covered by fine cells → row sum = 1."""
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_4x4_fine)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_each_coarse_maps_to_four_fine(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """Each coarse cell covers exactly 4 fine cells with weight 0.25."""
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_4x4_fine)
        for row_idx in range(4):
            row = W[row_idx].toarray().ravel()
            nonzero = row[row > 1e-15]
            assert len(nonzero) == 4
            np.testing.assert_allclose(nonzero, 0.25, atol=1e-12)


class TestFineToCoarse:
    """Source is fine (4×4), target is coarse (2×2)."""

    def test_shape(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        W = quickpaver.compute_transfer_matrix(grid_4x4_fine, grid_2x2)
        assert W.shape == (16, 4)

    def test_row_sums(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """Each fine cell is fully inside one coarse cell → row sum = 1."""
        W = quickpaver.compute_transfer_matrix(grid_4x4_fine, grid_2x2)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)

    def test_each_fine_maps_to_one_coarse(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """Each fine cell sits entirely inside one coarse cell → one weight = 1."""
        W = quickpaver.compute_transfer_matrix(grid_4x4_fine, grid_2x2)
        for row_idx in range(16):
            row = W[row_idx].toarray().ravel()
            nonzero = row[row > 1e-15]
            assert len(nonzero) == 1
            np.testing.assert_allclose(nonzero, 1.0, atol=1e-12)


# ---------------------------------------------------------------------------
# Tests — shifted / partial overlap
# ---------------------------------------------------------------------------


class TestShiftedGrids:
    """Grids overlap but are offset by half a cell."""

    def test_shape(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_2x2_shifted: shapely.MultiPolygon,
    ) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2_shifted)
        assert W.shape == (4, 4)

    def test_partial_row_sums(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_2x2_shifted: shapely.MultiPolygon,
    ) -> None:
        """Some source cells are only partially covered → row sum < 1."""
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2_shifted)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        # At least one row should have sum < 1 (partial coverage)
        assert np.any(row_sums < 1.0 - 1e-12)
        # All row sums must be in [0, 1]
        assert np.all(row_sums >= -1e-12)
        assert np.all(row_sums <= 1.0 + 1e-12)

    def test_weights_positive(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_2x2_shifted: shapely.MultiPolygon,
    ) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2_shifted)
        assert np.all(W.toarray() >= 0.0)


# ---------------------------------------------------------------------------
# Tests — disjoint grids (no overlap)
# ---------------------------------------------------------------------------


class TestDisjointGrids:
    """Source and target do not overlap at all."""

    def test_all_zeros(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_2x2_disjoint: shapely.MultiPolygon,
    ) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2_disjoint)
        assert W.shape == (4, 4)
        assert W.nnz == 0

    def test_row_sums_are_zero(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_2x2_disjoint: shapely.MultiPolygon,
    ) -> None:
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2_disjoint)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 0.0, atol=1e-15)


# ---------------------------------------------------------------------------
# Tests — single polygon edge case
# ---------------------------------------------------------------------------


class TestSinglePolygon:
    """Grids with only one cell each."""

    def test_identical_single(self) -> None:
        """One source cell, one identical target cell → 1×1 identity."""
        poly = shapely.MultiPolygon([shapely.box(0, 0, 1, 1)])
        W = quickpaver.compute_transfer_matrix(poly, poly)
        assert W.shape == (1, 1)
        np.testing.assert_allclose(W.toarray(), [[1.0]], atol=1e-12)

    def test_single_source_many_target(self) -> None:
        """One source cell split across a finer target grid."""
        source = shapely.MultiPolygon([shapely.box(0, 0, 2, 2)])
        target = _make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.shape == (1, 4)
        np.testing.assert_allclose(W.toarray(), [[0.25, 0.25, 0.25, 0.25]], atol=1e-12)

    def test_many_source_single_target(self) -> None:
        """Four source cells fully inside one large target cell."""
        source = _make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)
        target = shapely.MultiPolygon([shapely.box(0, 0, 2, 2)])
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.shape == (4, 1)
        np.testing.assert_allclose(
            W.toarray(), [[1.0], [1.0], [1.0], [1.0]], atol=1e-12
        )


# ---------------------------------------------------------------------------
# Tests — numerical noise filtering
# ---------------------------------------------------------------------------


class TestNumericalNoiseFiltering:
    """Grids that share only an edge or corner produce near-zero areas."""

    def test_edge_touching_only(self) -> None:
        """Two grids share an edge but no area → empty transfer matrix."""
        grid_a = shapely.MultiPolygon([shapely.box(0, 0, 1, 1)])
        grid_b = shapely.MultiPolygon([shapely.box(1, 0, 2, 1)])  # touching edge
        W = quickpaver.compute_transfer_matrix(grid_a, grid_b)
        assert W.nnz == 0

    def test_corner_touching_only(self) -> None:
        """Two grids share only a corner point → empty transfer matrix."""
        grid_a = shapely.MultiPolygon([shapely.box(0, 0, 1, 1)])
        grid_b = shapely.MultiPolygon([shapely.box(1, 1, 2, 2)])  # touching corner
        W = quickpaver.compute_transfer_matrix(grid_a, grid_b)
        assert W.nnz == 0


# ---------------------------------------------------------------------------
# Tests — sanity check flag
# ---------------------------------------------------------------------------


class TestSanityCheck:
    """Exercise the ``is_sanity_check=True`` code path."""

    def test_sanity_check_passes_identical(
        self,
        grid_2x2: shapely.MultiPolygon,
    ) -> None:
        """Identical grids must pass the conservation sanity check."""
        # Should not raise
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_2x2, is_sanity_check=True)
        assert W.shape == (4, 4)

    def test_sanity_check_passes_fine_to_coarse(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """Fine→coarse with full coverage must pass the sanity check."""
        W = quickpaver.compute_transfer_matrix(
            grid_4x4_fine, grid_2x2, is_sanity_check=True
        )
        assert W.shape == (16, 4)

    def test_sanity_check_passes_coarse_to_fine(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """Coarse→fine with full coverage must pass the sanity check."""
        W = quickpaver.compute_transfer_matrix(
            grid_2x2, grid_4x4_fine, is_sanity_check=True
        )
        assert W.shape == (4, 16)

    def test_sanity_check_fails_partial_coverage(self) -> None:
        """Target extends beyond source → conservation violated → raises."""
        source = _make_regular_grid(0.0, 0.0, 1.0, 1.0, 2, 2)  # covers [0,2]×[0,2]
        target = _make_regular_grid(0.0, 0.0, 1.0, 1.0, 3, 3)  # covers [0,3]×[0,3]
        # Some target cells are NOT fully covered by source → sanity check fails
        with pytest.raises(AssertionError):
            quickpaver.compute_transfer_matrix(source, target, is_sanity_check=True)


# ---------------------------------------------------------------------------
# Tests — conservation / transfer correctness
# ---------------------------------------------------------------------------


class TestConservation:
    """Verify that transferring a uniform field conserves total mass."""

    def test_uniform_field_coarse_to_fine(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """A uniform value on the coarse grid is preserved in total mass."""
        W = quickpaver.compute_transfer_matrix(grid_2x2, grid_4x4_fine)
        source_areas = np.array([g.area for g in grid_2x2.geoms])

        v_source = np.full(4, 5.0)
        mass_source = v_source * source_areas  # mass per source cell

        mass_target = W.T @ mass_source  # mass per target cell
        np.testing.assert_allclose(mass_target.sum(), mass_source.sum(), rtol=1e-10)

    def test_varying_field_fine_to_coarse(
        self,
        grid_2x2: shapely.MultiPolygon,
        grid_4x4_fine: shapely.MultiPolygon,
    ) -> None:
        """A linearly varying field is conserved in total mass."""
        W = quickpaver.compute_transfer_matrix(grid_4x4_fine, grid_2x2)
        source_areas = np.array([g.area for g in grid_4x4_fine.geoms])

        v_source = np.arange(16, dtype=float)
        mass_source = v_source * source_areas

        mass_target = W.T @ mass_source
        np.testing.assert_allclose(mass_target.sum(), mass_source.sum(), rtol=1e-10)


# ---------------------------------------------------------------------------
# Tests — non-square grids
# ---------------------------------------------------------------------------


class TestNonSquareGrids:
    """Source and target have different numbers of cells."""

    def test_3x2_to_2x3(self) -> None:
        source = _make_regular_grid(0.0, 0.0, 1.0, 1.0, 3, 2)  # 6 cells
        target = _make_regular_grid(0.0, 0.0, 1.5, 1.0, 2, 2)  # 4 cells
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.shape == (6, 4)
        # All weights non-negative
        assert np.all(W.toarray() >= 0.0)
        # Row sums ≤ 1
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        assert np.all(row_sums <= 1.0 + 1e-12)

    def test_1x5_to_5x1(self) -> None:
        """Horizontal strip vs vertical strip — partial overlap."""
        source = _make_regular_grid(0.0, 0.0, 1.0, 5.0, 5, 1)  # 5 cells
        target = _make_regular_grid(0.0, 0.0, 5.0, 1.0, 1, 5)  # 5 cells
        W = quickpaver.compute_transfer_matrix(source, target)
        assert W.shape == (5, 5)
        row_sums = np.asarray(W.sum(axis=1)).ravel()
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-12)
