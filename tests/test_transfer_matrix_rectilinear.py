# SPDX-License-Identifier: BSD-3-Clause
"""Pytest tests for `compute_transfer_matrix_rectilinear`."""

from __future__ import annotations

import numpy as np
import pytest
import quickpaver


def test_identical_grids_is_identity() -> None:
    """A grid transferred onto itself must be the identity matrix."""
    center = np.array([0.0, 0.0])
    mat = quickpaver.compute_transfer_matrix_rectilinear(
        source_center=center,
        source_dx=1.0,
        source_dy=1.0,
        source_nx=3,
        source_ny=3,
        source_angle=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=3,
        target_ny=3,
        target_angle=0.0,
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
        source_angle=0.0,
        target_center=center,
        target_dx=2.0,
        target_dy=2.0,
        target_nx=2,
        target_ny=2,
        target_angle=0.0,
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
        source_angle=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=4,
        target_ny=4,
        target_angle=k90 * np.pi / 2,
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
        source_angle=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=6,
        target_ny=6,
        target_angle=np.deg2rad(17.0),
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
        source_angle=0.0,
        target_center=center,
        target_dx=1.0,
        target_dy=1.0,
        target_nx=3,
        target_ny=3,
        is_sanity_check=True,
    )
    mat_separable = quickpaver.compute_transfer_matrix_rectilinear(
        target_angle=0.0,
        **kwargs,  # ty:ignore[invalid-argument-type]  # noqa: E501
    )
    mat_nonseparable = quickpaver.compute_transfer_matrix_rectilinear(
        target_angle=1e-3,
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
        source_angle=0.0,
        target_center=np.array([1000.0, 1000.0]),
        target_dx=1.0,
        target_dy=1.0,
        target_nx=3,
        target_ny=3,
        target_angle=0.4,
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
        source_angle=0.0,
        target_center=np.array([0.5, 0.0]),
        target_dx=1.0,
        target_dy=1.0,
        target_nx=5,
        target_ny=5,
        target_angle=0.0,
        is_sanity_check=True,
    )
    row_sums = np.asarray(mat.sum(axis=1)).ravel()
    fully_covered = row_sums > 1 - 1e-6
    assert fully_covered.any()
    np.testing.assert_allclose(
        row_sums[fully_covered], np.ones(fully_covered.sum()), atol=1e-10
    )
