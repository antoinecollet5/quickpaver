# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

import numpy as np
from quickpaver import (
    RectilinearGrid,
    rlg_idx_to_nn,
    rlg_nn_to_idx,
    span_to_node_numbers_2d,
)


def test_span_to_node_numbers_1d() -> None:
    np.testing.assert_array_equal(
        span_to_node_numbers_2d((slice(0, 3), slice(None)), nx=21, ny=1),
        np.array([0, 1, 2]),
    )


def test_span_to_node_numbers_2d() -> None:
    np.testing.assert_equal(
        span_to_node_numbers_2d((slice(0, 4), slice(0, 3)), nx=21, ny=5),
        np.array([0, 21, 42, 1, 22, 43, 2, 23, 44, 3, 24, 45]),
    )


def non_rotated_rlgrid_10x9x8() -> RectilinearGrid:
    return RectilinearGrid(
        nx=10,
        ny=9,
        nz=8,
    )


def test_rlg_idx_to_nn():
    assert rlg_idx_to_nn(ix=0) == 0
    assert rlg_idx_to_nn(ix=1, indices_start_at_one=True) == 0
    # 11123	875	465	1.5	4	88	47	2	facies_empty
    assert (
        rlg_idx_to_nn(88, nx=89, iy=47, ny=78, iz=2, indices_start_at_one=True) == 11123
    )
    assert rlg_idx_to_nn(1, nx=89, iy=1, ny=78, iz=2, indices_start_at_one=True) == 6942
    assert (
        rlg_idx_to_nn(69, nx=89, iy=0, ny=78, iz=25, indices_start_at_one=False)
        == 173619
    )
    assert (
        rlg_idx_to_nn(89, nx=89, iy=78, ny=78, iz=47, indices_start_at_one=True)
        == 326273
    )


def test_hytec_node_number_to_test_indices():
    assert rlg_nn_to_idx(0, nx=1) == (0, 0, 0)
    assert rlg_nn_to_idx(0, nx=1, ny=1) == (0, 0, 0)
    assert rlg_nn_to_idx(0, nx=89, ny=78) == (0, 0, 0)
    assert rlg_nn_to_idx(11123, nx=89, ny=78) == (87, 46, 1)
    assert rlg_nn_to_idx(11123, nx=89, ny=78, indices_start_at_one=True) == (88, 47, 2)
    assert rlg_nn_to_idx(173619, nx=89, ny=78, indices_start_at_one=False) == (
        69,
        0,
        25,
    )
    assert rlg_nn_to_idx(326273, nx=89, ny=78, indices_start_at_one=True) == (
        89,
        78,
        47,
    )


def test_rlgrid_init_no_args() -> None:
    RectilinearGrid()


def test_rlgrid_nvoxels() -> None:
    assert (
        RectilinearGrid(
            nx=10,
            ny=10,
            nz=5,
        ).n_grid_cells
        == 500
    )


def test_rlgrid_volume() -> None:
    rlgrid = RectilinearGrid(nx=11, ny=67, nz=5, dx=8.8, dy=9.8, dz=6.5)
    # Test all_close
    assert abs(rlgrid.grid_cell_volume_m3 - 560.56) < 1e-5
    assert abs(rlgrid.total_volume_m3 - 2065663.6) < 1e-5


def test_non_rotated_rlgrid_bounds() -> None:
    # with origin at (0, 0, 0)
    np.testing.assert_equal(
        RectilinearGrid(nx=10, ny=10, nz=5, dx=2.1, dy=5.6, dz=9.0).bounds,
        np.array([[-10.5, 10.5], [-28.0, 28.0], [-22.5, 22.5]], dtype=np.float32),
    )

    # with non zero origin
    rlgrid = RectilinearGrid(
        cx=-89.5, cy=53.0, cz=-37.5, nx=10, ny=10, nz=5, dx=2.1, dy=5.6, dz=9.0
    )
    np.testing.assert_equal(
        rlgrid.bounds, np.array([[-100.0, -79.0], [25.0, 81.0], [-60.0, -15.0]])
    )

    # Test methods
    assert rlgrid.xmin == -100.0
    assert rlgrid.xmax == -79.0
    assert rlgrid.ymin == 25.0
    assert rlgrid.ymax == 81.0
    assert rlgrid.zmin == -60.0
    assert rlgrid.zmax == -15.0


def test_MCI01_rlgrid_bounds() -> None:
    # with non zero origin
    mci01_grid = RectilinearGrid(
        cx=1.36801258e05,
        cy=1.38457458e05,
        cz=-9.45000000e01,
        nx=45,
        ny=47,
        nz=27,
        dx=5.0,
        dy=5.0,
        dz=1.0,
        theta=30.0,
    )
    assert mci01_grid.n_grid_cells == 57105

    np.testing.assert_allclose(
        mci01_grid.bounds,
        np.array(
            [
                [1.367293e05, 1.370416e05],
                [1.383013e05, 1.386173e05],
                [-1.080000e02, -8.100000e01],
            ]
        ),
        rtol=1e-3,
    )

    assert abs(mci01_grid.x_extent - 312.35599) < 1e-3
    assert abs(mci01_grid.y_extent - 316.0159) < 1e-4
    assert abs(mci01_grid.z_extent - 27) < 1e-5


def test_rlgrid_get_indices() -> None:
    rlgrid = non_rotated_rlgrid_10x9x8()
    assert rlgrid.indices.shape == (3, 10, 9, 8)
    assert rlgrid.x_indices.shape == (720,)
    assert rlgrid.y_indices.shape == (720,)
    assert rlgrid.z_indices.shape == (720,)


def test_rlgrid_get_coords() -> None:
    rlgrid = non_rotated_rlgrid_10x9x8()
    assert rlgrid.origin_coords.shape == (3, 10, 9, 8)
    assert rlgrid.center_coords.shape == (3, 10, 9, 8)
    assert rlgrid.center_coords_2d.shape == (2, 10, 9)
    assert rlgrid._opposite_vertice_coords.shape == (3, 10, 9, 8)
