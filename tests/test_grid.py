"""Tests for the rectilinear grid module."""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import quickpaver
import shapely
from quickpaver._grid import (
    _get_centroid_voxel_coords,
    _get_free_grid_cells,
    _get_mask,
    _get_vertical_limits_indices,
    _get_vertices_centroid,
    _keep_a_b_if_c_in_a,
    _rotation_x,
    _rotation_y,
    _rotation_z,
)
from scipy.sparse import csc_array

# ---------------------------------------------------------------------------
# Fake PyVista objects and fixtures
# ---------------------------------------------------------------------------


class FakeImageData:
    """Minimal fake pyvista.ImageData object."""

    fail_on_direction_matrix = False

    def __init__(
        self,
        *,
        dimensions: tuple[int, int, int],
        spacing: tuple[float, float, float],
        origin: tuple[float, float, float],
        direction_matrix: np.ndarray | None = None,
    ) -> None:
        """Store ImageData construction arguments."""
        if direction_matrix is not None and self.fail_on_direction_matrix:
            raise TypeError("direction_matrix is unsupported")

        self.dimensions = dimensions
        self.spacing = spacing
        self.origin = origin
        self.direction_matrix = direction_matrix
        self.cell_data: dict[str, np.ndarray] = {}

    def cast_to_rectilinear_grid(self) -> FakeRectilinearGrid:
        """Return a fake rectilinear grid."""
        grid = FakeRectilinearGrid(
            dimensions=self.dimensions,
            spacing=self.spacing,
            origin=self.origin,
            direction_matrix=self.direction_matrix,
        )
        grid.cell_data = self.cell_data
        return grid

    def cast_to_structured_grid(self) -> FakeStructuredGrid:
        """Return a fake structured grid."""
        grid = FakeStructuredGrid(
            dimensions=self.dimensions,
            spacing=self.spacing,
            origin=self.origin,
            direction_matrix=self.direction_matrix,
        )
        grid.cell_data = self.cell_data
        return grid


class FakeRectilinearGrid(FakeImageData):
    """Minimal fake pyvista.RectilinearGrid object."""


class FakeStructuredGrid(FakeImageData):
    """Minimal fake pyvista.StructuredGrid object."""

    def __init__(self, **kwargs: Any) -> None:
        """Store StructuredGrid construction arguments."""
        super().__init__(**kwargs)
        self.rotations: list[tuple[str, float, tuple[float, float, float]]] = []

    def rotate_z(
        self,
        angle: float,
        *,
        point: tuple[float, float, float],
        inplace: bool,
    ) -> FakeStructuredGrid:
        """Record a z-axis rotation."""
        self.rotations.append(("z", angle, point))
        assert inplace is False
        return self

    def rotate_y(
        self,
        angle: float,
        *,
        point: tuple[float, float, float],
        inplace: bool,
    ) -> FakeStructuredGrid:
        """Record a y-axis rotation."""
        self.rotations.append(("y", angle, point))
        assert inplace is False
        return self

    def rotate_x(
        self,
        angle: float,
        *,
        point: tuple[float, float, float],
        inplace: bool,
    ) -> FakeStructuredGrid:
        """Record an x-axis rotation."""
        self.rotations.append(("x", angle, point))
        assert inplace is False
        return self


@pytest.fixture
def fake_pyvista(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Install a fake pyvista module."""
    FakeImageData.fail_on_direction_matrix = False

    fake_module = SimpleNamespace(
        ImageData=FakeImageData,
        RectilinearGrid=FakeRectilinearGrid,
        StructuredGrid=FakeStructuredGrid,
    )
    monkeypatch.setitem(sys.modules, "pyvista", fake_module)
    return fake_module


# ---------------------------------------------------------------------------
# Index conversion helpers
# ---------------------------------------------------------------------------


def test_rlg_idx_to_nn_scalar_and_arrays() -> None:
    """Check flattened numbering from scalar and array indices."""
    assert quickpaver.rlg_idx_to_nn(2, nx=10) == 2
    assert quickpaver.rlg_idx_to_nn(1, nx=3, iy=2, ny=4, iz=1) == 19

    result = quickpaver.rlg_idx_to_nn(
        np.array([0, 1]),
        nx=3,
        iy=np.array([0, 1]),
        ny=2,
        iz=np.array([0, 1]),
    )

    np.testing.assert_array_equal(result, np.array([0, 10]))


def test_rlg_idx_to_nn_one_based_indices_are_clipped() -> None:
    """Check one-based index conversion and clipping at zero."""
    result = quickpaver.rlg_idx_to_nn(
        ix=np.array([0, 1, 2]),
        nx=10,
        iy=np.array([0, 1, 2]),
        ny=10,
        iz=np.array([0, 1, 2]),
        indices_start_at_one=True,
    )

    np.testing.assert_array_equal(result, np.array([0, 0, 111]))


def test_rlg_nn_to_idx_scalar_and_array() -> None:
    """Check inverse flattened-number conversion."""
    ix, iy, iz = quickpaver.rlg_nn_to_idx(19, nx=3, ny=4)

    assert int(ix) == 1
    assert int(iy) == 2
    assert int(iz) == 1

    ix_arr, iy_arr, iz_arr = quickpaver.rlg_nn_to_idx(np.array([0, 10]), nx=3, ny=2)

    np.testing.assert_array_equal(ix_arr, np.array([0, 1]))
    np.testing.assert_array_equal(iy_arr, np.array([0, 1]))
    np.testing.assert_array_equal(iz_arr, np.array([0, 1]))


def test_rlg_nn_to_idx_one_based() -> None:
    """Check one-based output indices."""
    ix, iy, iz = quickpaver.rlg_nn_to_idx(
        19,
        nx=3,
        ny=4,
        indices_start_at_one=True,
    )

    assert int(ix) == 2
    assert int(iy) == 3
    assert int(iz) == 2


def test_span_to_node_numbers_2d() -> None:
    """Check 2D span-to-node conversion."""
    result = quickpaver.span_to_node_numbers_2d(
        (slice(1, 3), slice(0, 1)),
        nx=3,
        ny=2,
    )

    np.testing.assert_array_equal(result, np.array([1, 2], dtype=np.int32))


def test_span_to_node_numbers_3d() -> None:
    """Check 3D span-to-node conversion."""
    result = quickpaver.span_to_node_numbers_3d(
        (slice(0, 1), slice(1, 2), slice(None)),
        nx=2,
        ny=2,
        nz=2,
    )

    np.testing.assert_array_equal(result, np.array([2, 6], dtype=np.int32))


# ---------------------------------------------------------------------------
# Generic utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("nx", "ny", "expected"),
    [
        (0, 3, np.zeros((0, 3), dtype=bool)),
        (3, 0, np.zeros((3, 0), dtype=bool)),
        (1, 1, np.array([[True]])),
        (
            3,
            4,
            np.array(
                [
                    [True, True, True, True],
                    [True, False, False, True],
                    [True, True, True, True],
                ]
            ),
        ),
    ],
)
def test_get_array_borders_selection(
    nx: int,
    ny: int,
    expected: np.ndarray,
) -> None:
    """Check border mask generation, including empty and one-cell cases."""
    np.testing.assert_array_equal(
        quickpaver.get_array_borders_selection(nx, ny),
        expected,
    )


def test_rotation_matrices() -> None:
    """Check elementary 90-degree rotation matrices."""
    angle = np.pi / 2.0

    np.testing.assert_allclose(
        _rotation_x(angle) @ np.array([0.0, 1.0, 0.0]),
        np.array([0.0, 0.0, 1.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _rotation_y(angle) @ np.array([0.0, 0.0, 1.0]),
        np.array([1.0, 0.0, 0.0]),
        atol=1e-12,
    )
    np.testing.assert_allclose(
        _rotation_z(angle) @ np.array([1.0, 0.0, 0.0]),
        np.array([0.0, 1.0, 0.0]),
        atol=1e-12,
    )


def test_abstract_grid_placeholder_methods_are_executable() -> None:
    """Execute abstract placeholder bodies for line coverage."""
    assert quickpaver.Grid.n_grid_cells.fget(object()) is None  # ty:ignore[invalid-argument-type]
    assert quickpaver.Grid.make_spatial_gradient_matrices(object()) is None
    assert quickpaver.Grid.make_spatial_permutation_matrices(object()) is None


# ---------------------------------------------------------------------------
# RectilinearGrid construction and geometry
# ---------------------------------------------------------------------------


def test_rectilinear_grid_init_validation() -> None:
    """Check constructor validation."""
    with pytest.raises(ValueError, match="dx, dy and dz"):
        quickpaver.RectilinearGrid(dx=0.0)

    with pytest.raises(ValueError, match="nx, ny and nz"):
        quickpaver.RectilinearGrid(nx=0)


def test_rectilinear_grid_basic_properties() -> None:
    """Check basic geometry, indexing, volumes, and string representations."""
    grid = quickpaver.RectilinearGrid(
        cx=10.0,
        cy=20.0,
        cz=30.0,
        dx=2.0,
        dy=4.0,
        dz=6.0,
        nx=2,
        ny=3,
        nz=4,
    )

    assert grid.rot_center == (10.0, 20.0, 30.0)
    assert grid.shape == (2, 3, 4)
    assert grid.dims == (2.0, 4.0, 6.0)
    assert grid.n_grid_cells == 24

    assert grid.grid_cell_volume_m3 == 48.0
    assert grid.total_volume_m3 == 1152.0
    assert grid.gamma_ij_x_m2 == 24.0
    assert grid.gamma_ij_y_m2 == 12.0
    assert grid.gamma_ij_z_m2 == 8.0

    assert grid.origin == (8.0, 14.0, 18.0)
    assert grid.x0 == 8.0
    assert grid.y0 == 14.0
    assert grid.z0 == 18.0

    np.testing.assert_allclose(
        grid.bounds,
        np.array([[8.0, 12.0], [14.0, 26.0], [18.0, 42.0]]),
    )

    assert grid.x_extent == 4.0
    assert grid.y_extent == 12.0
    assert grid.z_extent == 24.0

    assert grid.indices.shape == (3, 2, 3, 4)
    assert grid.origin_coords.shape == (3, 2, 3, 4)
    assert grid.center_coords.shape == (3, 2, 3, 4)
    assert grid.non_rot_center_coords.shape == (3, 2, 3, 4)
    assert grid.center_coords_2d.shape == (2, 2, 3)
    assert grid.non_rot_center_coords_2d.shape == (2, 2, 3)
    assert grid._opposite_vertice_coords.shape == (3, 2, 3, 4)

    np.testing.assert_allclose(grid.origin_coords[:, 0, 0, 0], [8.0, 14.0, 18.0])
    np.testing.assert_allclose(grid.center_coords[:, 0, 0, 0], [9.0, 16.0, 21.0])

    assert grid.x_indices.size == grid.n_grid_cells
    assert grid.y_indices.size == grid.n_grid_cells
    assert grid.z_indices.size == grid.n_grid_cells

    assert "RectilinearGrid" in str(grid)
    assert "RectilinearGrid" in repr(grid)

    copied = grid.copy()

    assert copied is not grid
    assert copied.shape == grid.shape
    assert copied.dims == grid.dims
    assert copied.rot_center == grid.rot_center


def test_rectilinear_grid_rotation() -> None:
    """Check coordinate rotation around the grid centre."""
    grid = quickpaver.RectilinearGrid(theta=90.0)
    coords = np.array([[1.0], [0.0], [0.0]])

    rotated = grid._rotate_coords(coords)

    np.testing.assert_allclose(rotated[:, 0], [0.0, 1.0, 0.0], atol=1e-12)
    assert grid.bounding_box_vertices_coordinates.shape == (3, 8)


# ---------------------------------------------------------------------------
# Shapely export
# ---------------------------------------------------------------------------


def test_to_shapely_all_cells_masked_cells_and_empty() -> None:
    """Check conversion of 2D cells to Shapely polygons."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1, theta=30.0)

    all_cells = grid.to_shapely()
    assert isinstance(all_cells, shapely.MultiPolygon)
    assert len(all_cells.geoms) == 4

    mask = np.array([[True, False], [False, True]])
    selected = grid.to_shapely(mask=mask)
    assert isinstance(selected, shapely.MultiPolygon)
    assert len(selected.geoms) == 2

    empty = grid.to_shapely(mask=np.zeros((2, 2), dtype=bool))
    assert isinstance(empty, shapely.MultiPolygon)
    assert len(empty.geoms) == 0


# ---------------------------------------------------------------------------
# PyVista export
# ---------------------------------------------------------------------------


def test_to_pyvista_import_error(monkeypatch: pytest.MonkeyPatch) -> None:
    """Check optional PyVista dependency handling."""
    grid = quickpaver.RectilinearGrid()

    original_import = builtins.__import__

    def fake_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name == "pyvista":
            raise ImportError("missing pyvista")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    with pytest.raises(ImportError, match="PyVista is required"):
        grid.to_pyvista()


def test_to_pyvista_invalid_representation(fake_pyvista: SimpleNamespace) -> None:
    """Check invalid representation validation."""
    grid = quickpaver.RectilinearGrid()

    with pytest.raises(ValueError, match="representation must be"):
        grid.to_pyvista(representation="bad")  # ty:ignore[invalid-argument-type]


def test_to_pyvista_image_without_rotation(fake_pyvista: SimpleNamespace) -> None:
    """Check compact ImageData export without rotation."""
    grid = quickpaver.RectilinearGrid(
        cx=1.0,
        cy=2.0,
        cz=3.0,
        dx=2.0,
        dy=3.0,
        dz=4.0,
        nx=2,
        ny=3,
        nz=4,
    )

    pv_grid = grid.to_pyvista(
        cell_data={"ids": np.arange(grid.n_grid_cells).reshape(grid.shape, order="F")}
    )

    assert isinstance(pv_grid, FakeImageData)
    assert pv_grid.dimensions == (3, 4, 5)
    assert pv_grid.spacing == (2.0, 3.0, 4.0)
    assert pv_grid.origin == tuple(np.array([1.0, 2.0, 3.0]) + grid._local_origin)
    assert pv_grid.direction_matrix is None

    np.testing.assert_array_equal(
        pv_grid.cell_data["ids"],
        np.arange(grid.n_grid_cells),
    )


def test_to_pyvista_image_with_rotation(fake_pyvista: SimpleNamespace) -> None:
    """Check compact rotated ImageData export."""
    grid = quickpaver.RectilinearGrid(theta=10.0, phi=20.0, psi=30.0)

    pv_grid = grid.to_pyvista(representation="image")

    assert isinstance(pv_grid, FakeImageData)
    assert pv_grid.direction_matrix is not None
    np.testing.assert_allclose(pv_grid.origin, grid.origin)


def test_to_pyvista_image_direction_matrix_unsupported(
    fake_pyvista: SimpleNamespace,
) -> None:
    """Check RuntimeError for older PyVista without direction_matrix support."""
    FakeImageData.fail_on_direction_matrix = True
    grid = quickpaver.RectilinearGrid(theta=10.0)

    with pytest.raises(RuntimeError, match="direction_matrix"):
        grid.to_pyvista(representation="image")


def test_to_pyvista_rectilinear_non_rotated(fake_pyvista: SimpleNamespace) -> None:
    """Check RectilinearGrid export for non-rotated grids."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=2)

    pv_grid = grid.to_pyvista(representation="rectilinear")

    assert isinstance(pv_grid, FakeRectilinearGrid)


def test_to_pyvista_rectilinear_rotated_raises(fake_pyvista: SimpleNamespace) -> None:
    """Check that rotated grids are rejected as true RectilinearGrid."""
    grid = quickpaver.RectilinearGrid(theta=45.0)

    with pytest.raises(ValueError, match="rotated grid cannot"):
        grid.to_pyvista(representation="rectilinear")


def test_to_pyvista_structured_with_rotation(fake_pyvista: SimpleNamespace) -> None:
    """Check StructuredGrid export and PyVista rotation order."""
    grid = quickpaver.RectilinearGrid(theta=10.0, phi=20.0, psi=30.0)

    pv_grid = grid.to_pyvista(representation="structured")

    assert isinstance(pv_grid, FakeStructuredGrid)
    assert pv_grid.rotations == [
        ("z", 10.0, grid.rot_center),
        ("y", 20.0, grid.rot_center),
        ("x", 30.0, grid.rot_center),
    ]


def test_to_pyvista_apply_rotation_false(fake_pyvista: SimpleNamespace) -> None:
    """Check that rotations can be disabled."""
    grid = quickpaver.RectilinearGrid(theta=45.0)

    pv_grid = grid.to_pyvista(representation="image", apply_rotation=False)

    assert isinstance(pv_grid, FakeImageData)
    assert pv_grid.direction_matrix is None
    assert pv_grid.origin == tuple(np.array(grid.rot_center) + grid._local_origin)


def test_to_pyvista_invalid_cell_data_size(fake_pyvista: SimpleNamespace) -> None:
    """Check cell-data size validation."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1)

    with pytest.raises(ValueError, match="must contain exactly"):
        grid.to_pyvista(cell_data={"bad": np.arange(3)})


# ---------------------------------------------------------------------------
# Polygon selection helpers
# ---------------------------------------------------------------------------


def test_get_vertices_centroid() -> None:
    """Check centroid of polygon vertices."""
    vertices = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)]

    assert _get_vertices_centroid(vertices) == (1.0, 1.0)


def test_get_centroid_voxel_coords() -> None:
    """Check nearest voxel to polygon centroid."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=1)
    vertices = [(-0.25, -0.25), (0.25, -0.25), (0.25, 0.25), (-0.25, 0.25)]

    assert _get_centroid_voxel_coords(vertices, grid) == (1, 1)


def test_create_selections_array_2d_validation() -> None:
    """Check selection-array validation."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    with pytest.raises(ValueError, match="same length"):
        quickpaver.create_selections_array_2d([polygon], [1, 2], grid)

    with pytest.raises(ValueError, match="0 cannot"):
        quickpaver.create_selections_array_2d([polygon], [0], grid)


def test_create_selections_array_2d_overlap_priority() -> None:
    """Check that earlier polygons keep priority in overlapping areas."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1)
    polygon_all = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    polygon_same = polygon_all

    selection = quickpaver.create_selections_array_2d(
        [polygon_all, polygon_same],
        [5, 9],
        grid,
    )

    assert selection.dtype == np.int32
    assert np.all(selection == 5)


def test_get_free_grid_cells_and_mask() -> None:
    """Check free-cell detection and polygon-domain mask."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1)
    selection = np.array([[0, 1], [0, 0]], dtype=np.int32)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    coords = grid.center_coords_2d.reshape(2, -1, order="F").T

    np.testing.assert_array_equal(
        _get_free_grid_cells(selection),
        np.array([[True, False], [True, True]]),
    )

    mask = _get_mask(polygon, selection, coords, grid.nx, grid.ny)

    np.testing.assert_array_equal(mask, np.array([[True, False], [True, True]]))


def test_get_vertical_limits_indices() -> None:
    """Check vertical-limit conversion, including clamping."""
    assert _get_vertical_limits_indices(np.array([-1.0, 1.0]), -1.0, 1.0, 2) == (
        0,
        1,
    )
    assert _get_vertical_limits_indices(np.array([-99.0, 99.0]), 0.0, 1.0, 3) == (
        0,
        2,
    )


# ---------------------------------------------------------------------------
# Binary dilation
# ---------------------------------------------------------------------------


def test_binary_dilation_validation_and_zero_iterations() -> None:
    """Check binary dilation validation and zero-iteration behaviour."""
    seed = np.array([[False, True], [False, False]])
    domain = np.ones((2, 2), dtype=bool)

    with pytest.raises(ValueError, match="same shape"):
        quickpaver.binary_dilation(seed, np.ones((3, 3), dtype=bool))

    with pytest.raises(ValueError, match="non-negative"):
        quickpaver.binary_dilation(seed, domain, iterations=-1)

    np.testing.assert_array_equal(
        quickpaver.binary_dilation(seed, domain, iterations=0),
        seed,
    )


def test_binary_dilation_domain_mask() -> None:
    """Check 4-connected dilation constrained by the domain mask."""
    seed = np.zeros((3, 3), dtype=bool)
    seed[1, 1] = True
    domain = np.ones((3, 3), dtype=bool)
    domain[1, 2] = False

    result = quickpaver.binary_dilation(seed, domain, iterations=1)

    expected = np.array(
        [
            [False, True, False],
            [True, True, False],
            [False, True, False],
        ]
    )

    np.testing.assert_array_equal(result, expected)


# ---------------------------------------------------------------------------
# 2D polygon selection with dilation
# ---------------------------------------------------------------------------


def test_get_polygon_selection_with_dilation_2d() -> None:
    """Check 2D polygon selection grown by dilation."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=1)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    selection = quickpaver.get_polygon_selection_with_dilation_2d([polygon], grid)

    assert selection.shape == (3, 3)
    assert np.all(selection == 1)


def test_get_polygon_selection_with_dilation_2d_with_existing_selection() -> None:
    """Check that an existing 2D selection constrains dilation and is preserved."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=1)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    initial_selection = np.zeros((grid.nx, grid.ny), dtype=np.int32)
    initial_selection[0, 1] = 7
    initial_selection_before = initial_selection.copy()

    selection = quickpaver.get_polygon_selection_with_dilation_2d(
        [polygon],
        grid,
        selection=initial_selection,
    )

    expected = np.ones((grid.nx, grid.ny), dtype=np.int32)
    expected[0, 1] = 7

    np.testing.assert_array_equal(selection, expected)
    np.testing.assert_array_equal(initial_selection, initial_selection_before)


def test_get_polygon_selection_with_dilation_2d_existing_selection_blocks_cell() -> (
    None
):
    """Check that non-zero cells in an existing 2D selection are not overwritten."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=1)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    initial_selection = np.zeros((grid.nx, grid.ny), dtype=np.int32)
    initial_selection[2, 1] = 9

    selection = quickpaver.get_polygon_selection_with_dilation_2d(
        [polygon],
        grid,
        selection=initial_selection,
    )

    assert selection[2, 1] == 9
    assert np.count_nonzero(selection == 1) == grid.nx * grid.ny - 1


def test_get_poly_sel_with_dilation_2d_existing_selection_at_seed_is_overwritten() -> (
    None
):
    """Document current behaviour: the centroid seed overwrites existing selection."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=1)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    initial_selection = np.zeros((grid.nx, grid.ny), dtype=np.int32)
    initial_selection[1, 1] = 9

    selection = quickpaver.get_polygon_selection_with_dilation_2d(
        [polygon],
        grid,
        selection=initial_selection,
    )

    assert selection[1, 1] == 1


# ---------------------------------------------------------------------------
# 3D polygon selection with dilation
# ---------------------------------------------------------------------------


def test_get_polygon_selection_with_dilation_3d_validation() -> None:
    """Check 3D polygon selection validation."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=2)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    with pytest.raises(ValueError, match="vertical_limits"):
        quickpaver.get_polygon_selection_with_dilation_3d(
            [polygon, polygon],
            np.array([[-1.0, 1.0]]),
            grid,
        )


def test_get_polygon_selection_with_dilation_3d() -> None:
    """Check 3D polygon selection grown slice by slice."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=2)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    selection = quickpaver.get_polygon_selection_with_dilation_3d(
        [polygon],
        np.array([[-1.0, 1.0]]),
        grid,
    )

    assert selection.shape == (3, 3, 2)
    assert np.all(selection == 1)


def test_get_polygon_selection_with_dilation_3d_existing_selection_blocks_cells() -> (
    None
):
    """Check that non-zero cells in an existing 3D selection are not overwritten."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=2)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]
    vertical_limits = np.array([[-1.0, 1.0]])

    initial_selection = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int32)
    initial_selection[0, 1, 0] = 7
    initial_selection[2, 1, 1] = 8

    selection = quickpaver.get_polygon_selection_with_dilation_3d(
        [polygon],
        vertical_limits,
        grid,
        selection=initial_selection,
    )

    assert selection[0, 1, 0] == 7
    assert selection[2, 1, 1] == 8
    assert np.count_nonzero(selection == 1) == grid.nx * grid.ny * grid.nz - 2


def test_get_poly_sel_with_dilation_3d_skips_slices_outside_vertical_limits() -> None:
    """Check that 3D dilation skips z-slices outside polygon vertical limits."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=3, nz=3, dz=1.0)
    polygon = [(-2.0, -2.0), (2.0, -2.0), (2.0, 2.0), (-2.0, 2.0)]

    vertical_limits = np.array([[-0.1, 0.1]])

    selection = quickpaver.get_polygon_selection_with_dilation_3d(
        [polygon],
        vertical_limits,
        grid,
    )

    expected = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int32)
    expected[:, :, 1] = 1

    np.testing.assert_array_equal(selection, expected)


# ---------------------------------------------------------------------------
# Sparse matrix pair helpers
# ---------------------------------------------------------------------------


def test_keep_a_b_if_c_in_a() -> None:
    """Check pair filtering based on values in first array."""
    a = np.array([0, 1, 2, 3])
    b = np.array([10, 11, 12, 13])

    kept_a, kept_b = _keep_a_b_if_c_in_a(a, b, np.array([1, 3]))

    np.testing.assert_array_equal(kept_a, np.array([1, 3]))
    np.testing.assert_array_equal(kept_b, np.array([11, 13]))


def test_get_owner_neigh_indices() -> None:
    """Check paired owner/neighbour index generation and filtering."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    span_owner = (slice(0, 2), slice(None), slice(None))
    span_neigh = (slice(1, 3), slice(None), slice(None))

    owner, neigh = quickpaver.get_owner_neigh_indices(grid, span_owner, span_neigh)

    np.testing.assert_array_equal(owner, np.array([0, 1], dtype=np.int32))
    np.testing.assert_array_equal(neigh, np.array([1, 2], dtype=np.int32))

    owner, neigh = quickpaver.get_owner_neigh_indices(
        grid,
        span_owner,
        span_neigh,
        owner_indices_to_keep=np.array([1]),
    )

    np.testing.assert_array_equal(owner, np.array([1], dtype=np.int32))
    np.testing.assert_array_equal(neigh, np.array([2], dtype=np.int32))

    owner, neigh = quickpaver.get_owner_neigh_indices(
        grid,
        span_owner,
        span_neigh,
        neigh_indices_to_keep=np.array([1]),
    )

    np.testing.assert_array_equal(owner, np.array([0], dtype=np.int32))
    np.testing.assert_array_equal(neigh, np.array([1], dtype=np.int32))


# ---------------------------------------------------------------------------
# Sparse gradient matrices
# ---------------------------------------------------------------------------


def test_get_rlg_spatial_grad_mat_validation_and_empty() -> None:
    """Check gradient matrix validation and empty-axis case."""
    grid = quickpaver.RectilinearGrid(nx=1, ny=1, nz=1)

    with pytest.raises(ValueError, match="axis"):
        quickpaver.get_rlg_spatial_grad_mat(
            grid,
            n=1,
            axis=3,
            sub_selection=np.arange(grid.n_grid_cells),
        )

    with pytest.raises(ValueError, match="which"):
        quickpaver.get_rlg_spatial_grad_mat(
            grid,
            n=1,
            axis=0,
            sub_selection=np.arange(grid.n_grid_cells),
            which="invalid",  # ty:ignore[invalid-argument-type]
        )

    mat = quickpaver.get_rlg_spatial_grad_mat(
        grid,
        n=1,
        axis=0,
        sub_selection=np.arange(grid.n_grid_cells),
    )

    assert isinstance(mat, csc_array)
    np.testing.assert_array_equal(mat.toarray(), np.zeros((1, 1)))


@pytest.mark.parametrize(
    ("which", "expected"),
    [
        ("forward", np.array([[1.0, -1.0], [0.0, 0.0]])),
        ("backward", np.array([[0.0, 0.0], [-1.0, 1.0]])),
        ("both", np.array([[1.0, -1.0], [-1.0, 1.0]])),
    ],
)
def test_get_rlg_spatial_grad_mat_values(which: str, expected: np.ndarray) -> None:
    """Check x-axis gradient matrix values for simple two-cell grid."""
    grid = quickpaver.RectilinearGrid(dx=1.0, dy=2.0, dz=3.0, nx=2, ny=1, nz=1)

    mat = quickpaver.get_rlg_spatial_grad_mat(
        grid,
        n=grid.nx,
        axis=0,
        sub_selection=np.arange(grid.n_grid_cells),
        which=which,  # ty:ignore[invalid-argument-type]
    )

    np.testing.assert_allclose(mat.toarray(), expected)


def test_get_rlg_spatial_grad_mat_forward_with_sub_selection() -> None:
    """Check that forward gradient pairs are filtered by sub_selection."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    sub_selection = np.array([0, 1], dtype=np.int32)

    mat = quickpaver.get_rlg_spatial_grad_mat(
        grid,
        n=grid.nx,
        axis=0,
        sub_selection=sub_selection,
        which="forward",
    )

    expected = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(mat.toarray(), expected)


def test_get_rlg_spatial_grad_mat_backward_with_sub_selection() -> None:
    """Check that backward gradient pairs are filtered by sub_selection."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    sub_selection = np.array([1, 2], dtype=np.int32)

    mat = quickpaver.get_rlg_spatial_grad_mat(
        grid,
        n=grid.nx,
        axis=0,
        sub_selection=sub_selection,
        which="backward",
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0],
        ]
    )

    np.testing.assert_allclose(mat.toarray(), expected)


def test_get_rlg_spatial_grad_mat_both_with_isolated_sub_selection() -> None:
    """Check that isolated selected cells produce no gradient pairs."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    sub_selection = np.array([1], dtype=np.int32)

    mat = quickpaver.get_rlg_spatial_grad_mat(
        grid,
        n=grid.nx,
        axis=0,
        sub_selection=sub_selection,
        which="both",
    )

    np.testing.assert_allclose(mat.toarray(), np.zeros((3, 3)))


def test_make_rlg_spatial_gradient_matrices_and_method() -> None:
    """Check creation of x/y/z gradient matrices and instance wrapper."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=2)

    matrices = quickpaver.make_rlg_spatial_gradient_matrices(grid)
    method_matrices = grid.make_spatial_gradient_matrices(which="forward")

    assert len(matrices) == 3
    assert len(method_matrices) == 3
    assert all(isinstance(matrix, csc_array) for matrix in matrices)
    assert all(
        matrix.shape == (grid.n_grid_cells, grid.n_grid_cells) for matrix in matrices
    )


def test_make_rlg_spatial_gradient_matrices_with_sub_selection() -> None:
    """Check gradient matrix factory with sub_selection on a 2D grid."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1)
    sub_selection = np.array([0, 2], dtype=np.int32)

    gx, gy, gz = quickpaver.make_rlg_spatial_gradient_matrices(
        grid,
        sub_selection=sub_selection,
        which="forward",
    )

    expected_gx = np.zeros((4, 4))
    expected_gy = np.array(
        [
            [1.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_gz = np.zeros((4, 4))

    np.testing.assert_allclose(gx.toarray(), expected_gx)
    np.testing.assert_allclose(gy.toarray(), expected_gy)
    np.testing.assert_allclose(gz.toarray(), expected_gz)


def test_rectilinear_grid_make_spatial_gradient_matrices_with_sub_selection() -> None:
    """Check RectilinearGrid gradient wrapper with sub_selection."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    sub_selection = np.array([0, 1], dtype=np.int32)

    gx, gy, gz = grid.make_spatial_gradient_matrices(
        sub_selection=sub_selection,
        which="forward",
    )

    expected_gx = np.array(
        [
            [1.0, -1.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )

    np.testing.assert_allclose(gx.toarray(), expected_gx)
    np.testing.assert_allclose(gy.toarray(), np.zeros((3, 3)))
    np.testing.assert_allclose(gz.toarray(), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Sparse permutation matrices
# ---------------------------------------------------------------------------


def test_get_rlg_perm_mat_validation_empty_and_values() -> None:
    """Check permutation matrix validation, empty case, and simple values."""
    grid_empty_axis = quickpaver.RectilinearGrid(nx=1, ny=1, nz=1)

    with pytest.raises(ValueError, match="axis"):
        quickpaver.get_rlg_perm_mat(
            grid_empty_axis,
            n=1,
            axis=3,
            sub_selection=np.arange(grid_empty_axis.n_grid_cells),
        )

    empty = quickpaver.get_rlg_perm_mat(
        grid_empty_axis,
        n=1,
        axis=0,
        sub_selection=np.arange(grid_empty_axis.n_grid_cells),
    )

    np.testing.assert_array_equal(empty.toarray(), np.zeros((1, 1)))

    grid = quickpaver.RectilinearGrid(nx=2, ny=1, nz=1)

    mat = quickpaver.get_rlg_perm_mat(
        grid,
        n=grid.nx,
        axis=0,
        sub_selection=np.arange(grid.n_grid_cells),
    )

    np.testing.assert_array_equal(mat.toarray(), np.array([[0.0, 0.0], [1.0, 0.0]]))


def test_get_rlg_perm_mat_with_sub_selection() -> None:
    """Check that forward permutation pairs are filtered by sub_selection."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    sub_selection = np.array([1, 2], dtype=np.int32)

    mat = quickpaver.get_rlg_perm_mat(
        grid,
        n=grid.nx,
        axis=0,
        sub_selection=sub_selection,
    )

    expected = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(mat.toarray(), expected)


def test_make_rlg_spatial_permutation_matrices_and_method() -> None:
    """Check creation of x/y/z permutation matrices and instance wrapper."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=2)

    matrices = quickpaver.make_rlg_spatial_permutation_matrices(grid)
    method_matrices = grid.make_spatial_permutation_matrices()

    assert len(matrices) == 3
    assert len(method_matrices) == 3
    assert all(isinstance(matrix, csc_array) for matrix in matrices)
    assert all(
        matrix.shape == (grid.n_grid_cells, grid.n_grid_cells) for matrix in matrices
    )


def test_make_rlg_spatial_permutation_matrices_with_sub_selection() -> None:
    """Check permutation matrix factory with sub_selection on a 2D grid."""
    grid = quickpaver.RectilinearGrid(nx=2, ny=2, nz=1)
    sub_selection = np.array([0, 2], dtype=np.int32)

    px, py, pz = quickpaver.make_rlg_spatial_permutation_matrices(
        grid,
        sub_selection=sub_selection,
    )

    expected_px = np.zeros((4, 4))
    expected_py = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
        ]
    )
    expected_pz = np.zeros((4, 4))

    np.testing.assert_array_equal(px.toarray(), expected_px)
    np.testing.assert_array_equal(py.toarray(), expected_py)
    np.testing.assert_array_equal(pz.toarray(), expected_pz)


def test_rectilinear_grid_make_spatial_permutation_matrices_with_sub_selection() -> (
    None
):
    """Check RectilinearGrid permutation wrapper with sub_selection."""
    grid = quickpaver.RectilinearGrid(nx=3, ny=1, nz=1)
    sub_selection = np.array([1, 2], dtype=np.int32)

    px, py, pz = grid.make_spatial_permutation_matrices(
        sub_selection=sub_selection,
    )

    expected_px = np.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
    )

    np.testing.assert_array_equal(px.toarray(), expected_px)
    np.testing.assert_array_equal(py.toarray(), np.zeros((3, 3)))
    np.testing.assert_array_equal(pz.toarray(), np.zeros((3, 3)))


# ---------------------------------------------------------------------------
# Resampling and upsampling
# ---------------------------------------------------------------------------


def test_resample_grid_preserves_extent_and_rotation() -> None:
    """Check that resampling changes discretisation but preserves extent."""
    grid = quickpaver.RectilinearGrid(
        cx=1.0,
        cy=2.0,
        cz=3.0,
        dx=2.0,
        dy=3.0,
        dz=4.0,
        nx=4,
        ny=5,
        nz=6,
        theta=10.0,
        phi=20.0,
        psi=30.0,
    )

    resampled = quickpaver.resample_grid(grid, factor_x=0.5, factor_y=2.0, factor_z=0.0)

    assert resampled.shape == (2, 10, 1)
    assert resampled.rot_center == grid.rot_center
    assert resampled.theta == grid.theta
    assert resampled.phi == grid.phi
    assert resampled.psi == grid.psi

    assert resampled.nx * resampled.dx == pytest.approx(grid.nx * grid.dx)
    assert resampled.ny * resampled.dy == pytest.approx(grid.ny * grid.dy)
    assert resampled.nz * resampled.dz == pytest.approx(grid.nz * grid.dz)


def test_duplicative_upsample_validation_and_values() -> None:
    """Check duplicative upsampling."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="positive integer"):
        quickpaver.duplicative_upsample(array, factor=0)

    result = quickpaver.duplicative_upsample(array, factor=2)

    expected = np.array(
        [
            [1.0, 1.0, 2.0, 2.0],
            [1.0, 1.0, 2.0, 2.0],
            [3.0, 3.0, 4.0, 4.0],
            [3.0, 3.0, 4.0, 4.0],
        ]
    )

    np.testing.assert_array_equal(result, expected)


def test_conservative_upsample_validation_values_and_sum() -> None:
    """Check conservative upsampling preserves the total sum."""
    array = np.array([[1.0, 2.0], [3.0, 4.0]])

    with pytest.raises(ValueError, match="positive integer"):
        quickpaver.conservative_upsample(array, factor=0)

    result = quickpaver.conservative_upsample(array, factor=2)

    assert result.shape == (4, 4)
    assert result.sum() == pytest.approx(array.sum())

    np.testing.assert_array_equal(
        result[:2, :2],
        np.full((2, 2), 0.25),
    )
