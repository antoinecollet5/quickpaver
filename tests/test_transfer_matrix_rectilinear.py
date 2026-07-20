# SPDX-License-Identifier: BSD-3-Clause
"""Pytest tests for `compute_transfer_matrix_rectilinear`."""

from __future__ import annotations

import numpy as np
import pytest
import quickpaver


@pytest.mark.parametrize(
    "cx1, cy1, dx1, dy1, nx1, ny1, angle_deg_1, cx2, cy2, dx2, dy2,"
    " nx2, ny2, angle_deg_2",
    [
        # Identical grids (identity)
        (0.0, 0.0, 1.0, 1.0, 3, 3, 0.0, 0.0, 0.0, 1.0, 1.0, 3, 3, 0.0),
        # Rectangular grids k90_0
        (0.0, 0.0, 1.0, 1.0, 3, 5, 0.0, 0.0, 0.0, 1.0, 1.0, 3, 5, 0.0),
        # Asymmetric cell dimensions k90_0
        (0.0, 0.0, 2.0, 1.0, 2, 2, 0.0, 0.0, 0.0, 2.0, 1.0, 2, 2, 0.0),
        # Very small cells k90_0
        (0.0, 0.0, 0.01, 0.01, 2, 2, 0.0, 0.0, 0.0, 0.01, 0.01, 2, 2, 0.0),
        # Separable coarser target conserves mass
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, 0.0, 0.0, 2.0, 2.0, 2, 2, 0.0),
        # Separable k90_1 (90 degree)
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, np.pi / 2),
        # Separable k90_2 (180 degree)
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, np.pi),
        # Separable k90_3 (270 degree)
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, 3 * np.pi / 2),
        # Rectangular grids k90_1
        (0.0, 0.0, 1.0, 1.0, 3, 5, 0.0, 0.0, 0.0, 1.0, 1.0, 5, 3, np.pi / 2),
        # Separable offset grids k90_0
        (0.0, 0.0, 1.0, 1.0, 5, 5, 0.0, 1.5, 0.5, 1.0, 1.0, 5, 5, 0.0),
        # Offset translation only conserves mass
        (0.0, 0.0, 1.0, 1.0, 5, 5, 0.0, 0.5, 0.0, 1.0, 1.0, 5, 5, 0.0),
        # Separable offset with k90_1
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, 0.5, 0.5, 1.0, 1.0, 4, 4, np.pi / 2),
        # Separable offset with k90_2
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, 0.3, 0.7, 1.0, 1.0, 4, 4, np.pi),
        # Separable offset with k90_3
        (0.0, 0.0, 1.0, 1.0, 4, 4, 0.0, -0.5, 0.5, 1.0, 1.0, 4, 4, 3 * np.pi / 2),
        # Separable different scales k90_0
        (0.0, 0.0, 0.5, 0.5, 8, 8, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, 0.0),
        # Separable rotated source grid k90_0
        (0.0, 0.0, 1.0, 1.0, 4, 4, np.pi / 4, 0.0, 0.0, 1.0, 1.0, 4, 4, np.pi / 4),
        # Nonseparable small angle deviation
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, 1e-3),
        # Nonseparable arbitrary angle conserves mass (17 degrees)
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, np.deg2rad(17.0)),
        # Nonseparable 15 degree angle
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, np.deg2rad(15.0)),
        # Nonseparable 25 degree angle
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, np.deg2rad(25.0)),
        # Nonseparable 45 degree angle
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, np.deg2rad(45.0)),
        # Nonseparable 75 degree angle
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, np.deg2rad(75.0)),
        # Nonseparable negative angle (-30 degrees)
        (0.0, 0.0, 0.5, 0.5, 6, 6, 0.0, 0.0, 0.0, 1.0, 1.0, 6, 6, np.deg2rad(-30.0)),
        # Nonseparable offset centers and angle
        (1.0, 2.0, 1.0, 1.0, 5, 5, 0.0, 1.5, 2.5, 1.0, 1.0, 5, 5, np.deg2rad(33.0)),
        # Nonseparable both grids rotated
        (
            0.0,
            0.0,
            0.5,
            0.5,
            6,
            6,
            np.deg2rad(20.0),
            0.0,
            0.0,
            1.0,
            1.0,
            6,
            6,
            np.deg2rad(35.0),
        ),
        # Nonseparable fine source coarse target
        (0.0, 0.0, 0.5, 0.5, 8, 8, 0.0, 0.0, 0.0, 2.0, 2.0, 2, 2, np.deg2rad(22.0)),
        # Nonseparable coarse source fine target
        (0.0, 0.0, 2.0, 2.0, 2, 2, 0.0, 0.0, 0.0, 0.5, 0.5, 8, 8, np.deg2rad(18.0)),
        # Nonseparable rectangular grids
        (0.0, 0.0, 1.0, 1.0, 3, 5, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 6, np.deg2rad(37.0)),
        # Nonseparable small grids
        (0.0, 0.0, 1.0, 1.0, 2, 2, 0.0, 0.0, 0.0, 1.0, 1.0, 2, 2, np.deg2rad(23.0)),
        # Nonseparable large grids
        (0.0, 0.0, 1.0, 1.0, 10, 10, 0.0, 0.0, 0.0, 1.0, 1.0, 10, 10, np.deg2rad(11.0)),
        # Nonseparable dense query 5 degrees
        (0.0, 0.0, 0.5, 0.5, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, np.deg2rad(5.0)),
        # Nonseparable dense query 35 degrees
        (0.0, 0.0, 0.5, 0.5, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, np.deg2rad(35.0)),
        # Nonseparable dense query 65 degrees
        (0.0, 0.0, 0.5, 0.5, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, np.deg2rad(65.0)),
        # Nonseparable dense query 85 degrees
        (0.0, 0.0, 0.5, 0.5, 4, 4, 0.0, 0.0, 0.0, 1.0, 1.0, 4, 4, np.deg2rad(85.0)),
        # Single cell grids k90_0
        (0.0, 0.0, 1.0, 1.0, 1, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1, 1, 0.0),
        # Single cell grid nonseparable
        (0.0, 0.0, 1.0, 1.0, 1, 1, 0.0, 0.0, 0.0, 1.0, 1.0, 1, 1, np.deg2rad(33.0)),
        # Asymmetric cell dimensions nonseparable
        (0.0, 0.0, 2.0, 1.0, 3, 3, 0.0, 0.0, 0.0, 2.0, 1.0, 3, 3, np.deg2rad(28.0)),
        # Disjoint grids have no overlap
        (0.0, 0.0, 1.0, 1.0, 3, 3, 0.0, 1000.0, 1000.0, 1.0, 1.0, 3, 3, 0.4),
        # Large offset no overlap separable
        (0.0, 0.0, 1.0, 1.0, 3, 3, 0.0, 1000.0, 1000.0, 1.0, 1.0, 3, 3, 0.0),
        # Large offset no overlap nonseparable
        (
            0.0,
            0.0,
            1.0,
            1.0,
            3,
            3,
            0.0,
            1000.0,
            1000.0,
            1.0,
            1.0,
            3,
            3,
            np.deg2rad(45.0),
        ),
        # Nonseparable matches separable at zero angle
        (0.0, 0.0, 1.0, 1.0, 3, 3, 0.0, 0.0, 0.0, 1.0, 1.0, 3, 3, 1e-3),
        # Very small cells nonseparable
        (0.0, 0.0, 0.01, 0.01, 2, 2, 0.0, 0.0, 0.0, 0.01, 0.01, 2, 2, np.deg2rad(42.0)),
        # Nonseparable partial overlap
        (0.0, 0.0, 1.0, 1.0, 5, 5, 0.0, 3.0, 0.0, 1.0, 1.0, 5, 5, np.deg2rad(19.0)),
    ],
)
def test_compute_transfer_matrix_rectilinear_all_cases(
    cx1, cy1, dx1, dy1, nx1, ny1, angle_deg_1, cx2, cy2, dx2, dy2, nx2, ny2, angle_deg_2
) -> None:
    """Merged test: All test cases for compute_transfer_matrix_rectilinear."""

    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=np.array([cx1, cy1]),
        source_dx=dx1,
        source_dy=dy1,
        source_nx=nx1,
        source_ny=ny1,
        source_angle_deg=angle_deg_1,
        target_center=np.array([cx2, cy2]),
        target_dx=dx2,
        target_dy=dy2,
        target_nx=nx2,
        target_ny=ny2,
        target_angle_deg=angle_deg_2,
        is_sanity_check=True,
    )
    mat2 = quickpaver.compute_transfer_matrix(
        quickpaver.RectilinearGrid(
            cx=cx1, cy=cy1, dx=dx1, dy=dy1, nx=nx1, ny=ny1, theta=angle_deg_1
        ).to_shapely(),
        quickpaver.RectilinearGrid(
            cx=cx2, cy=cy2, dx=dx2, dy=dy2, nx=nx2, ny=ny2, theta=angle_deg_2
        ).to_shapely(),
    )

    np.testing.assert_allclose(mat.todense(), mat2.todense())


def test_identical_grids_is_identity() -> None:
    """A grid transferred onto itself must be the identity matrix."""
    center = np.array([0.0, 0.0])
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=center,
        source_dx=1.0,
        source_dy=1.0,
        source_nx=3,
        source_ny=3,
        source_angle_deg=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=3,
        target_ny=3,
        target_angle_deg=0.0,
        is_sanity_check=True,
    )
    dense = mat.toarray()
    np.testing.assert_allclose(dense, np.eye(9), atol=1e-10)


def test_separable_coarser_target_conserves_mass() -> None:
    """A finer source grid coarsened by 2x2 blocks must conserve total mass."""
    center = np.array([0.0, 0.0])
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=center,
        source_dx=1.0,
        source_dy=1.0,
        source_nx=4,
        source_ny=4,
        source_angle_deg=0.0,
        target_center=center,
        target_dx=2.0,
        target_dy=2.0,
        target_nx=2,
        target_ny=2,
        target_angle_deg=0.0,
        is_sanity_check=True,
    )
    assert mat.shape == (16, 4)
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(16), atol=1e-10)
    assert mat.sum() == pytest.approx(16.0)


@pytest.mark.parametrize("k90", [1, 2, 3])
def test_separable_90_degree_rotations_conserve_mass(k90: int) -> None:
    """Relative rotations that are exact multiples of 90 deg use the separable path."""
    center = np.array([0.0, 0.0])
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=center,
        source_dx=1.0,
        source_dy=1.0,
        source_nx=4,
        source_ny=4,
        source_angle_deg=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=4,
        target_ny=4,
        target_angle_deg=np.rad2deg(k90 * np.pi / 2),
        is_sanity_check=True,
    )
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    np.testing.assert_allclose(row_sums, np.ones(16), atol=1e-10)


def test_nonseparable_arbitrary_angle_conserves_mass() -> None:
    """Fully-covered source cells at an arbitrary relative angle still sum to 1."""
    center = np.array([0.0, 0.0])
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=center,
        source_dx=0.5,
        source_dy=0.5,
        source_nx=6,
        source_ny=6,
        source_angle_deg=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=6,
        target_ny=6,
        target_angle_deg=np.deg2rad(17.0),
        is_sanity_check=True,
    )
    # Fully interior source cells (well away from the rotated target's
    # boundary) must be fully covered and conserve their mass exactly.
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    fully_covered = row_sums > 1 - 1e-6
    assert fully_covered.any()
    np.testing.assert_allclose(
        row_sums[fully_covered], np.ones(fully_covered.sum()), atol=1e-10
    )


def test_nonseparable_matches_separable_at_zero_angle() -> None:
    """A "nonseparable"-looking call at a near-0 deg angle should match the
    separable path's result once snapped to an exact 0 deg angle."""
    center = np.array([0.0, 0.0])
    kwargs = dict(
        source_center=center,
        source_dx=1.0,
        source_dy=1.0,
        source_nx=3,
        source_ny=3,
        source_angle_deg=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=3,
        target_ny=3,
        is_sanity_check=True,
    )
    mat_separable = quickpaver.compute_transfer_matrix_rectilinear(
        target_angle_deg=0.0,
        **kwargs,  # ty:ignore[invalid-argument-type]  # noqa: E501
    )
    mat_nonseparable = quickpaver.compute_transfer_matrix_rectilinear(
        target_angle_deg=1e-3,
        **kwargs,  # ty:ignore[invalid-argument-type]  # noqa: E501
    )
    np.testing.assert_allclose(
        mat_separable.toarray(), mat_nonseparable.toarray(), atol=1e-2
    )


def test_disjoint_grids_have_no_overlap() -> None:
    """Grids that are far apart in space must not exchange any mass."""
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=np.array([0.0, 0.0]),
        source_dx=1.0,
        source_dy=1.0,
        source_nx=3,
        source_ny=3,
        source_angle_deg=0.0,
        target_center=np.array([1000.0, 1000.0]),
        target_dx=1.0,
        target_dy=1.0,
        target_nx=3,
        target_ny=3,
        target_angle_deg=0.4,
        is_sanity_check=False,
    )
    assert mat.nnz == 0


def test_offset_translation_only_conserves_mass() -> None:
    """A pure translation (no rotation) between grids of equal resolution
    still conserves each fully-covered source cell's mass."""
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=np.array([0.0, 0.0]),
        source_dx=1.0,
        source_dy=1.0,
        source_nx=5,
        source_ny=5,
        source_angle_deg=0.0,
        target_center=np.array([0.5, 0.0]),
        target_dx=1.0,
        target_dy=1.0,
        target_nx=5,
        target_ny=5,
        target_angle_deg=0.0,
        is_sanity_check=True,
    )
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    fully_covered = row_sums > 1 - 1e-6
    assert fully_covered.any()
    np.testing.assert_allclose(
        row_sums[fully_covered], np.ones(fully_covered.sum()), atol=1e-10
    )
