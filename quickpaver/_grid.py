# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Rectilinear grid."""

import abc
import math
from dataclasses import dataclass
from typing import List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.path
import numpy as np
from scipy.sparse import csc_array, lil_array

from quickpaver._types import Int, NDArrayBool, NDArrayFloat, NDArrayInt


def rlg_idx_to_nn(
    ix: Int,
    nx: int = 1,
    iy: Int = 0,
    ny: int = 1,
    iz: Int = 0,
    indices_start_at_one: bool = False,
) -> NDArrayInt:
    """
    Convert indices (ix, iy, iz) to a node-number.

    For 1D and 2D, simply leave iy, ny, iz and nz to their default values.

    Note
    ----
    Node numbering start at zero.

    Warning
    -------
    This applies only for regular grids. It is not suited for vertex.

    Parameters
    ----------
    ix : int
        Index on the x-axis.
    nx : int, optional
        Number of grid cells on the x-axis. The default is 1.
    iy : int, optional
        Index on the y-axis. The default is 0.
    ny : int, optional
        Number of grid cells on the y-axis. The default is 1.
    iz : int, optional
        Index on the z-axis. The default is 0.
    indices_start_at_one: bool, optional
        Whether the indices start at 1. Otherwise, start at 0. The default is False.

    Returns
    -------
    int
        The node number.

    """
    _ix = np.asarray(ix, dtype=np.int64)
    _iy = np.asarray(iy, dtype=np.int64)
    _iz = np.asarray(iz, dtype=np.int64)

    if indices_start_at_one:
        ix = np.clip(_ix - 1, a_min=0, a_max=np.inf)
        iy = np.clip(_iy - 1, a_min=0, a_max=np.inf)
        iz = np.clip(_iz - 1, a_min=0, a_max=np.inf)
        # TODO check warnings
    return np.array(ix) + (np.array(iy) * nx) + (np.array(iz) * ny * nx)


def rlg_nn_to_idx(
    node_number: Int,
    nx: int = 1,
    ny: int = 1,
    indices_start_at_one: bool = False,
) -> Tuple[NDArrayInt, NDArrayInt, NDArrayInt]:
    """
    Convert a node-number to indices (ix, iy, iz) for a regular grid.

    For 1D and 2D, simply leave ny, and nz to their default values.

    Note
    ----
    Node numbering start at zero.

    Warning
    -------
    This applies only for regular grids. It is not suited for vertex.

    Parameters
    ----------
    nx : int
        Number of grid cells on the x-axis. The default is 1.
    ny : int, optional
        Number of grid cells on the y-axis. The default is 1.
    indices_start_at_one: bool, optional
        Whether the indices start at 1. Otherwise, start at 0. The default is False.

    Returns
    -------
    int
        The node number.

    """
    _node_number = np.array(node_number, dtype=np.int64)
    ix = (_node_number) % nx
    iz = (_node_number - ix) // (nx * ny)
    iy = (_node_number - ix - (nx * ny) * iz) // nx

    if indices_start_at_one:
        ix += 1
        iy += 1
        iz += 1

    return ix, iy, iz


class Grid(abc.ABC):
    """Define a grid."""

    @property
    @abc.abstractmethod
    def n_grid_cells(self) -> int:
        """Return the number of grid cells."""
        ...

    @abc.abstractmethod
    def make_spatial_gradient_matrices(
        self,
        sub_selection: Optional[NDArrayInt] = None,
        which: Literal["forward", "backward", "both"] = "both",
    ) -> Tuple[csc_array, csc_array, csc_array]:
        """
        Make the matrix which gives the spatial gradient of a field.

        Parameters
        ----------
        sub_selection : Optional[NDArrayInt], optional
            _description_, by default None

        Returns
        -------
        Tuple[csc_array, csc_array, csc_array]
            _description_
        """

        ...

    @abc.abstractmethod
    def make_spatial_permutation_matrices(
        self,
        sub_selection: Optional[NDArrayInt] = None,
    ) -> Tuple[csc_array, csc_array, csc_array]:
        """
        Make the matrix which gives the spatial permutation of a field.

        Parameters
        ----------
        sub_selection : Optional[NDArrayInt], optional
            _description_, by default None
        which : optional
            _description_, by default "both"

        Returns
        -------
        Tuple[csc_array, csc_array, csc_array]
            _description_
        """
        ...


def span_to_node_numbers_2d(
    span: Union[NDArrayInt, Tuple[slice, slice], slice], nx: int, ny: int
) -> NDArrayInt:
    """Convert the given span to an array of node indices."""
    _a = np.zeros((nx, ny))
    _a[span] = 1.0
    row, col = np.nonzero(_a)
    return np.array(rlg_idx_to_nn(row, nx=nx, iy=col, ny=ny), dtype=np.int32)


def span_to_node_numbers_3d(
    span: Union[NDArrayInt, Tuple[slice, slice, slice], slice],
    nx: int,
    ny: int,
    nz: int,
) -> NDArrayInt:
    """Convert the given span to an array of node indices."""
    _a = np.zeros((nx, ny, nz))
    _a[span] = 1.0
    ix, iy, iz = np.nonzero(_a)
    return np.array(rlg_idx_to_nn(ix, nx=nx, iy=iy, ny=ny, iz=iz), dtype=np.int32)


def get_array_borders_selection(nx: int, ny: int) -> NDArrayBool:
    """
    Get a selection of the array border as a bool array.

    Note
    ----
    There is no border for an awis of dim 1.

    Parameters
    ----------
    nx: int
        Number of grid cells along the x axis.
    ny: int
        Number of grid cells along the y axis.
    """
    _nx = nx - 2
    if _nx < 0:
        _nx = nx
    _ny = ny - 2
    if _ny < 0:
        _ny = ny
    return np.pad(
        np.zeros((_nx, _ny), dtype=np.bool_),
        ((min(max(nx - 1, 0), 1),), ((min(max(ny - 1, 0), 1),))),
        "constant",
        constant_values=1,
    )


def _rotation_x(theta) -> NDArrayFloat:
    """Matrix for a rotation around the x axis with theta radians."""
    return np.array(
        [
            [1, 0, 0],
            [0, math.cos(theta), -math.sin(theta)],
            [0, math.sin(theta), math.cos(theta)],
        ]
    )


def _rotation_y(theta) -> NDArrayFloat:
    """Matrix for a rotation around the y axis with theta radians."""
    return np.array(
        [
            [math.cos(theta), 0, math.sin(theta)],
            [0, 1, 0],
            [-math.sin(theta), 0, math.cos(theta)],
        ]
    )


def _rotation_z(theta) -> NDArrayFloat:
    """Matrix for a rotation around the z axis with theta radians."""
    return np.array(
        [
            [math.cos(theta), -math.sin(theta), 0],
            [math.sin(theta), math.cos(theta), 0],
            [0, 0, 1],
        ]
    )


@dataclass
class RectilinearGrid(Grid):
    """
    Represent a rectilinear 3D grid.

    Note
    ----
    For Euler angles:
    https://www.meccanismocomplesso.org/en/3d-rotations-and-euler-angles-in-python/

    """

    def __init__(
        self,
        x0: float = 0.0,
        y0: float = 0.0,
        z0: float = 0.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        rot_center: Optional[Tuple[float, float, float]] = None,
        theta: float = 0.0,
        phi: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        x0 : float
            Grid origin x coordinate (smalest value, not centroid) in meters.
        y0 : float
            Grid origin y coordinate (smalest value, not centroid) in meters.
        z0 : float
            Grid origin z coordinate (smalest value, not centroid) in meters.
        dx : float
            Mesh size along the x axis in meters.
        dy : float
            Mesh size along the y axis in meters.
        dz : float
            Mesh size along the z axis in meters.
        nx : int
            Number of meshes along the x axis.
        ny : int
            Number of meshes along the y axis.
        nz : int
            Number of meshes along the v axis.
        rot_center:
            Coordinates (x, y, z) used as a reference point for the grid rotation.
            If None, (x0, y0, z0) is used. The default is None.
        theta : float
            z-axis rotation angle in degrees with (x0, y0, z0) as origin.
        phi : float
            y-axis-rotation angle in degrees with (x0, y0, z0) as origin.
        psi : float
            x-axis-rotation angle in degrees with (x0, y0, z0) as origin.
        """
        self.x0: float = x0
        self.y0: float = y0
        self.z0: float = z0
        self.dx: float = dx
        self.dy: float = dy
        self.dz: float = dz
        self.nx: int = nx
        self.ny: int = ny
        self.nz: int = nz
        if rot_center is not None:
            self.rot_center: Tuple[float, float, float] = rot_center
        else:
            self.rot_center = (x0, y0, z0)
        self.theta = theta
        self.phi = phi
        self.psi = psi

    @property
    def origin(self) -> Tuple[float, float, float]:
        """Non rotated origin coords."""
        return (self.x0, self.y0, self.z0)

    @property
    def shape(self) -> Tuple[int, int, int]:
        return (self.nx, self.ny, self.nz)

    @property
    def dims(self) -> Tuple[float, float, float]:
        return (self.dx, self.dy, self.dz)

    @property
    def n_grid_cells(self) -> int:
        """Return the number of grid cells."""
        return self.nx * self.ny * self.nz

    @property
    def grid_cell_volume_m3(self) -> float:
        """Return the volume of one grid cell in m3."""
        return self.dx * self.dy * self.dz

    @property
    def total_volume_m3(self) -> float:
        """Return the total grid volume in m3."""
        return self.grid_cell_volume_m3 * self.n_grid_cells

    @property
    def gamma_ij_x_m2(self) -> float:
        """Return the surface of the frontiers along the x axis in m2"""
        return self.dy * self.dz

    @property
    def gamma_ij_y_m2(self) -> float:
        """Return the surface of the frontiers along the y axis in m2"""
        return self.dx * self.dz

    @property
    def gamma_ij_z_m2(self) -> float:
        """Return the surface of the frontiers along the z axis in m2"""
        return self.dx * self.dy

    @property
    def indices(self) -> NDArrayInt:
        """Return the grid indices with shape (3, nx, ny, nz)."""
        return np.asarray(
            np.meshgrid(range(self.nx), range(self.ny), range(self.nz), indexing="ij"),
            dtype=np.int32,
        )

    @property
    def _non_rotated_origin_coords(self) -> NDArrayFloat:
        """
        Return the grid meshes origin coordinates with shape (3, nx, ny, nz).

        Note
        ----
        Rotation is not applied.
        """
        return (
            self.indices.reshape(3, -1, order="F")
            * np.array([[self.dx, self.dy, self.dz]], dtype=np.float64).T
            + np.array([[self.x0, self.y0, self.z0]]).T
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    def _rotate_coords(self, non_rotated_coords: NDArrayFloat) -> NDArrayFloat:
        """
        Rotate the coordinates.

        Parameters
        ----------
        non_rotated_coords: NDArrayFloat
            Expected shape (3, nx, ny, nz)

        Note
        ----
        The rotation with the matrices multiplication is done relatively to point
        (0.0, 0.0, 0.0), so we should remove the origin point (x0, y0, z0) before the
        rotation and add it afterward.

        Return
        ------
        NDArrayFloat
            The rotated coordinates with shape (3, nx, ny, nz).
        """
        return (
            np.dot(
                _rotation_x(np.deg2rad(self.psi)),
                np.dot(
                    _rotation_y(np.deg2rad(self.phi)),
                    np.dot(
                        _rotation_z(np.deg2rad(self.theta)),
                        non_rotated_coords.reshape(3, -1, order="F")
                        - np.array([self.rot_center]).T,
                    ),
                ),
            )
            + np.array([self.rot_center]).T
        )

    @property
    def origin_coords(self) -> NDArrayFloat:
        """Return the grid meshes origin coordinates with shape (3, nx, ny, nz)."""
        return self._rotate_coords(self._non_rotated_origin_coords).reshape(
            3, self.nx, self.ny, self.nz, order="F"
        )

    @property
    def x_indices(self) -> NDArrayInt:
        """Return the grid meshes x-indices as 1D array."""
        return self.indices[0].ravel()

    @property
    def y_indices(self) -> NDArrayInt:
        """Return the grid meshes y-indices as 1D array."""
        return self.indices[1].ravel()

    @property
    def z_indices(self) -> NDArrayInt:
        """Return the grid meshes z-indices as 1D array."""
        return self.indices[2].ravel()

    @property
    def center_coords(self) -> NDArrayFloat:
        """Return the grid meshes center coordinates with shape (3, nx, ny, nz)."""
        return self._rotate_coords(
            (
                self._non_rotated_origin_coords.reshape(3, -1, order="F")
                + np.array([[self.dx / 2, self.dy / 2, self.dz / 2]]).T
            )
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    @property
    def non_rot_center_coords(self) -> NDArrayFloat:
        """Return the non rotated grid cell center coords with shape (3, nx, ny, nz)."""
        return (
            self._non_rotated_origin_coords.reshape(3, -1, order="F")
            + np.array([[self.dx / 2, self.dy / 2, self.dz / 2]]).T
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    @property
    def center_coords_2d(self) -> NDArrayFloat:
        """Return the coordinates of the voxel centers for an xy slice."""
        return self.center_coords[:2, :, :, 0]

    @property
    def non_rot_center_coords_2d(self) -> NDArrayFloat:
        """Return the non rotated coordinates of the voxel centers for an xy slice."""
        return self.non_rot_center_coords[:2, :, :, 0]

    @property
    def _opposite_vertice_coords(self) -> NDArrayFloat:
        """
        Return the grid meshes opposite coordinates with shape (3, nx, ny, nz).

        Note
        ----
        The opposite vertice is the origin symmetric with respect to the cell center.
        """
        return self._rotate_coords(
            (
                self._non_rotated_origin_coords.reshape(3, -1, order="F")
                + np.array([[self.dx, self.dy, self.dz]]).T
            )
        ).reshape(3, self.nx, self.ny, self.nz, order="F")

    @property
    def bounding_box_vertices_coordinates(self) -> NDArrayFloat:
        """Return the coordinates of the 8 bounding box vertices."""
        tmp = np.array(
            [
                [self.x0, self.y0, self.z0],
                [self.x0 + self.nx * self.dx, self.y0, self.z0],
                [self.x0 + self.nx * self.dx, self.y0 + self.ny * self.dy, self.z0],
                [self.x0, self.y0 + self.ny * self.dy, self.z0],
                [self.x0, self.y0, self.z0 + self.nz * self.dz],
                [self.x0 + self.nx * self.dx, self.y0, self.z0 + self.nz * self.dz],
                [
                    self.x0 + self.nx * self.dx,
                    self.y0 + self.ny * self.dy,
                    self.z0 + self.nz * self.dz,
                ],
                [self.x0, self.y0 + self.ny * self.dy, self.z0 + self.nz * self.dz],
            ]
        ).T
        return self._rotate_coords(tmp)

    @property
    def bounds(self) -> NDArrayFloat:
        """Return the bounds [[xmin, xmax], [ymin, ymax], [zmin, zmax]]."""
        # Create an array with the coordinates of the 8 non rotated grid summits
        # Apply rotation
        _ = self.bounding_box_vertices_coordinates
        return np.array(
            [
                _.min(axis=1),
                _.max(axis=1),
            ]
        ).T

    @property
    def xmin(self) -> float:
        """Return the minimum x of the grid."""
        return self.bounds[0, 0]

    @property
    def xmax(self) -> float:
        """Return the maximum x of the grid."""
        return self.bounds[0, 1]

    @property
    def ymin(self) -> float:
        """Return the minimum y of the grid."""
        return self.bounds[1, 0]

    @property
    def ymax(self) -> float:
        """Return the maximum y of the grid."""
        return self.bounds[1, 1]

    @property
    def zmin(self) -> float:
        """Return the minimum z of the grid."""
        return self.bounds[2, 0]

    @property
    def zmax(self) -> float:
        """Return the maximum z of the grid."""
        return self.bounds[2, 1]

    @property
    def x_extent(self) -> float:
        """Return the x extent in meters."""
        return self.xmax - self.xmin

    @property
    def y_extent(self) -> float:
        """Return the y extent in meters."""
        return self.ymax - self.ymin

    @property
    def z_extent(self) -> float:
        """Return the z extent in meters."""
        return self.zmax - self.zmin

    def make_spatial_gradient_matrices(
        self,
        sub_selection: Optional[NDArrayInt] = None,
        which: Literal["forward", "backward", "both"] = "both",
    ) -> Tuple[csc_array, csc_array, csc_array]:
        """
        Make the matrix which gives the spatial gradient of a field.

        Parameters
        ----------
        sub_selection : Optional[NDArrayInt], optional
            _description_, by default None
        which: Literal["forward", "backward", "both"] = "both"
            _description_, by default "both"

        Returns
        -------
        Tuple[csc_array, csc_array]
            _description_
        """
        return make_rlg_spatial_gradient_matrices(
            self, sub_selection=sub_selection, which=which
        )

    def make_spatial_permutation_matrices(
        self,
        sub_selection: Optional[NDArrayInt] = None,
    ) -> Tuple[csc_array, csc_array, csc_array]:
        """
        Make the matrix which gives the spatial permutation of a field.

        Parameters
        ----------
        sub_selection : Optional[NDArrayInt], optional
            _description_, by default None
        which: Literal["forward", "backward", "both"] = "both"
            _description_, by default "both"

        Returns
        -------
        Tuple[csc_array, csc_array]
            _description_
        """
        return make_rlg_spatial_permutation_matrices(self, sub_selection=sub_selection)


def get_vertices_centroid(
    vertices: Union[NDArrayFloat, List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """Get the vertices centroid."""
    _x_list = [vertex[0] for vertex in vertices]
    _y_list = [vertex[1] for vertex in vertices]
    _len = len(vertices)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (_x, _y)


def get_centroid_voxel_coords(
    vertices: Union[NDArrayFloat, List[Tuple[float, float]]],
    grid: RectilinearGrid,
) -> Tuple[Int, Int]:
    """
    For a given convex polygon an a 2D grid, give the centroid voxel.

    Parameters
    ----------
    vertices : Union[NDArrayFloat, List[Tuple[float, float]]]
        Coords of the convex polygon exterior ring with shape (M, 2)
    grid: RectilinearGrid,
        The grid definition (dimensions, position, etc.).

    Returns
    -------
    Tuple[int, int]
        x, y coordinates of the centroid voxel.
    """
    # This works for convex polygons only
    _x, _y = get_vertices_centroid(vertices)
    # get the closer integer
    distances = np.square(grid.center_coords_2d[0].ravel("F") - _x) + np.square(
        grid.center_coords_2d[1].ravel("F") - _y
    )
    ix, iy, _ = rlg_nn_to_idx(int(np.argmin(distances)), nx=grid.nx, ny=grid.ny)
    return (ix, iy)


def create_selections_array_2d(
    polygons: Sequence[Sequence[Tuple[float, float]]],
    sel_ids: Union[Sequence[int], NDArrayInt],
    grid: RectilinearGrid,
) -> NDArrayInt:
    """
    Return a grid array containing the sel_ids as values.

    The grid array has the dimension of the grid. It ensure that one grid block
    corresponds to a unique selection id.

    Parameters
    ----------
    polygons : Sequence[Sequence[Tuple[float, float]]]
        Sequence of polygons for which to perform the selection in the grid.
        The order matters as the first polygon will be prioritize if overlapping
        between polygons occurs.
    sel_ids : Union[Sequence[int], NDArrayInt]
        Sequence integers selection ids. An id cannot be zero
        (reserved for no selection).
    grid : RectilinearGrid
        The grid object for which to performm the selection.

    Returns
    -------
    NDArrayInt
        Grid selections array.
    """
    if 0 in sel_ids:
        raise ValueError(
            "0 cannot be part has sel_ids. It is reserved for empty selection."
        )

    # flatten points coordinates
    _sel_array = np.zeros((grid.nx, grid.ny), dtype=np.int8)

    # The mask sum ensure that a voxel is not selected twice
    mask_sum: Optional[NDArrayInt] = None
    for _polygon, cell_id in zip(polygons, sel_ids):
        # Select the mesh that belongs to the polygon
        path = mpl.path.Path(_polygon)
        mask = path.contains_points(
            grid.center_coords[:2, :, :, 0].reshape(2, -1, order="F").T
        )
        if mask_sum is not None:
            mask = np.logical_and(mask, ~mask_sum)
            mask_sum = np.logical_or(mask, mask_sum)
        else:
            mask_sum = mask
        _sel_array[mask.reshape(grid.nx, grid.ny, order="F")] = cell_id
    return _sel_array


def get_free_grid_cells(selection) -> NDArrayBool:
    """Return the free grid cells (no selected) as a boolean array."""
    return selection == 0


def _get_mask(
    polygon, selection: NDArrayInt, center_coords_2d: NDArrayFloat, nx: int, ny: int
) -> NDArrayBool:
    # Select the mesh that belongs to the polygon
    path = mpl.path.Path(polygon)
    mask = np.reshape(
        path.contains_points(center_coords_2d),
        (nx, ny),
        "F",
    )
    # Make sure that the voxels are not already part of a selection
    mask = np.logical_and(mask, get_free_grid_cells(selection))
    return mask


def binary_dilation(
    input: NDArrayBool, mask: NDArrayBool, iterations: int = 1
) -> NDArrayBool:
    _arr = input.copy()
    _arr[1:, :] = np.where(input[:-1, :], True, _arr[1:, :])
    _arr[:-1, :] = np.where(input[1:, :], True, _arr[:-1, :])
    _arr[:, 1:] = np.where(input[:, :-1], True, _arr[:, 1:])
    _arr[:, :-1] = np.where(input[:, 1:], True, _arr[:, :-1])
    # apply the masking
    _arr[~mask] = input[~mask]
    return _arr


def get_polygon_selection_with_dilation_2d(
    polygons: Union[List[NDArrayFloat], List[List[Tuple[float, float]]]],
    grid: RectilinearGrid,
    selection: Optional[NDArrayInt] = None,
) -> NDArrayInt:
    """Extend the selections using binary dilation.

    Parameters
    ----------
    polygon : Union[NDArrayFloat, List[Tuple[float, float]]]
        Coords of the exterior ring with shape (M, 2)
    grid: RectilinearGrid,
        The grid definition (dimensions, position, etc.).
    selection: Optional[NDArrayInt]
        An already existing selection as starting point. The default is None.
    """
    # Start by creating an empty grid with int type
    if selection is None:
        _selection = np.zeros((grid.nx, grid.ny), dtype=np.int8)
    else:
        _selection = selection.copy()

    # initiate _oldselection variable
    _old_selection = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int8)

    # Grid coordinates -> Flat array
    _grid_coords_2d = grid.center_coords_2d.reshape(2, -1, order="F").T

    # Create an initial selection for each cell (only one voxel selected)
    sel_ids = np.arange(len(polygons)) + 1
    for sel_id, vertices in zip(sel_ids, polygons):
        _selection[get_centroid_voxel_coords(vertices, grid)] = sel_id

    # Perform the dilation iteration by iteration to ensure a better split between
    # the selections.
    while np.not_equal(_selection, _old_selection).any():
        # Update the old_grid with the new one for the while
        _old_selection = _selection.copy()
        # Perform the dilation for each selection
        for sel_id, vertices in zip(sel_ids, polygons):
            # The mask is free cells + the contained
            mask = _get_mask(vertices, _selection, _grid_coords_2d, grid.nx, grid.ny)
            _selection[
                binary_dilation(_selection == sel_id, mask=mask, iterations=1)
            ] = sel_id
    return _selection


def _get_vertical_limits_indices(
    limits_in_m: NDArrayFloat, z0: float, dz: float, nz: int
) -> Tuple[int, int]:
    """
    Convert the vertical limits to grid indices.

    Note
    ----
    We have to take the max with 0.0 because the selections could go beyond the grid.
    """
    bot_index = int(max(np.round((limits_in_m[0] - z0 - dz / 2) / dz, 0), 0))
    top_index = int(min(np.round((limits_in_m[1] - z0 - dz / 2) / dz, 0), nz))
    return (bot_index, top_index)


def get_polygon_selection_with_dilation_3d(
    polygons: Union[List[NDArrayFloat], List[List[Tuple[float, float]]]],
    vertical_limits: Union[NDArrayFloat, List[List[Tuple[float, float]]]],
    grid: RectilinearGrid,
    selection: Optional[NDArrayInt] = None,
) -> NDArrayInt:
    """
    Extend the selections using binary dilation.

    Note
    ----
    Although there are for loops, this should be quite fast (>30s) even for grids with
    millions of meshes.

    Parameters
    ----------
    polygon : Union[NDArrayFloat, List[Tuple[float, float]]]
        Coords of the exterior ring with shape (M, 2)
    vertical_limits: Union[NDArrayFloat, List[Tuple[float, float]]]
        Top and bottom limits of the selections with shape (M, 2).
    grid: RectilinearGrid
        The grid definition.
    selection: Optional[NDArrayInt]
        An already existing selection as starting point. The default is None.
    """
    if selection is None:
        _selection = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int8)
    else:
        _selection = selection.copy()

    _vertical_limits: NDArrayFloat = np.asarray(vertical_limits).reshape(-1, 2)

    # Grid coordinates -> Flat array (2d, horizontal slice)
    _grid_coords_2d = grid.center_coords_2d.reshape(2, -1, order="F").T

    # Create an initial selection for each cell (only one voxel selected)
    sel_ids = np.arange(len(polygons)) + 1
    for i, (sel_id, vertices) in enumerate(zip(sel_ids, polygons)):
        # Get the vertical limits in indices
        _limits = _get_vertical_limits_indices(
            _vertical_limits[i, :], grid.z0, grid.dz, grid.nz
        )
        print(_limits)
        # Initial selection
        _selection[get_centroid_voxel_coords(vertices, grid)][
            _limits[0] : _limits[1] + 1
        ] = sel_id

    # initiate _oldselection variable
    _old_selection = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int8)

    # Copy polygons
    _polygons = polygons.copy()

    # Perform the dilation iteration by iteration to ensure a better split between
    # the selections.
    while np.not_equal(_selection, _old_selection).any():
        # Update the oldselection with the new one for the while
        _old_selection = _selection.copy()

        for iz in range(grid.nz):
            # Perform the dilation for each selection
            for i, (sel_id, vertices) in enumerate(zip(sel_ids, _polygons)):
                _limits = _get_vertical_limits_indices(
                    _vertical_limits[i, :], grid.z0, grid.dz, grid.nz
                )
                if iz < _limits[0] or iz > _limits[1]:
                    continue
                # The mask is free cells + the contained
                mask = _get_mask(
                    vertices, _selection[:, :, iz], _grid_coords_2d, grid.nx, grid.ny
                )
                _selection[:, :, iz][
                    binary_dilation(
                        _selection[:, :, iz] == sel_id, mask=mask, iterations=1
                    )
                ] = sel_id
    return _selection


def keep_a_b_if_c_in_a(
    a: NDArrayInt, b: NDArrayInt, c: NDArrayInt
) -> Tuple[NDArrayInt, NDArrayInt]:
    """Keep values in a and b if c in a."""
    is_kept = np.isin(a, c)
    return a[is_kept], b[is_kept]


def get_owner_neigh_indices(
    grid: RectilinearGrid,
    span_owner: Tuple[slice, slice, slice],
    span_neigh: Tuple[slice, slice, slice],
    owner_indices_to_keep: Optional[NDArrayInt] = None,
    neigh_indices_to_keep: Optional[NDArrayInt] = None,
) -> Tuple[NDArrayInt, NDArrayInt]:
    """_summary_

    Parameters
    ----------
    grid: RectilinearGrid
        _description_
    span_owner : Tuple[slice, slice]
        _description_
    span_neigh : Tuple[slice, slice]
        _description_
    indices_to_remove : NDArrayInt
        _description_

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        _description_
    """
    # Get indices
    indices_owner: NDArrayInt = span_to_node_numbers_3d(
        span_owner, nx=grid.nx, ny=grid.ny, nz=grid.nz
    )
    indices_neigh: NDArrayInt = span_to_node_numbers_3d(
        span_neigh, nx=grid.nx, ny=grid.ny, nz=grid.nz
    )

    if owner_indices_to_keep is not None:
        indices_owner, indices_neigh = keep_a_b_if_c_in_a(
            indices_owner, indices_neigh, owner_indices_to_keep
        )
    if neigh_indices_to_keep is not None:
        indices_neigh, indices_owner = keep_a_b_if_c_in_a(
            indices_neigh, indices_owner, neigh_indices_to_keep
        )
    return indices_owner, indices_neigh


def get_rlg_spatial_grad_mat(
    grid: RectilinearGrid,
    n: int,
    axis: int,
    sub_selection: NDArrayInt,
    which: Literal["forward", "backward", "both"] = "both",
) -> csc_array:
    # matrix for the spatial gradient along the axis
    mat = lil_array((grid.n_grid_cells, grid.n_grid_cells), dtype=np.float64)

    # no need to fill the mat with a single element
    if n < 2:
        return mat.tocsc()

    tmp = {
        0: grid.gamma_ij_x_m2,
        1: grid.gamma_ij_y_m2,
        2: grid.gamma_ij_z_m2,
    }[axis] / grid.grid_cell_volume_m3

    _slices1: List[slice] = [slice(0, n - 1), slice(None), slice(None)]
    _slices2: List[slice] = [slice(1, n), slice(None), slice(None)]
    slices1: Tuple[slice, slice, slice] = (
        _slices1[0 - axis],
        _slices1[1 - axis],
        _slices1[2 - axis],
    )
    slices2: Tuple[slice, slice, slice] = (
        _slices2[0 - axis],
        _slices2[1 - axis],
        _slices2[2 - axis],
    )

    if which in ["forward", "both"]:
        # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            slices1,
            slices2,
            owner_indices_to_keep=sub_selection,
            neigh_indices_to_keep=sub_selection,
        )

        mat[idc_owner, idc_neigh] -= tmp * np.ones(idc_owner.size)
        mat[idc_owner, idc_owner] += tmp * np.ones(idc_owner.size)

    if which in ["backward", "both"]:
        # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            slices2,
            slices1,
            owner_indices_to_keep=sub_selection,
            neigh_indices_to_keep=sub_selection,
        )

        mat[idc_owner, idc_neigh] -= tmp * np.ones(idc_owner.size)
        mat[idc_owner, idc_owner] += tmp * np.ones(idc_owner.size)

    return mat.tocsc()


def make_rlg_spatial_gradient_matrices(
    grid: RectilinearGrid,
    sub_selection: Optional[NDArrayInt] = None,
    which: Literal["forward", "backward", "both"] = "both",
) -> Tuple[csc_array, csc_array, csc_array]:
    """
    Make matrices to compute the spatial gradient along x and y axes of a field.

    The gradient is obtained by the dot product between the field and the matrix.

    Parameters
    ----------
    grid : RectilinearGrid
        Grid of the field
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.

    Returns
    -------
    Tuple[csc_array, csc_array, csc_array]
        Spatial gradient matrices for x and y axes.
    """
    if sub_selection is None:
        _sub_selection: NDArrayInt = np.arange(grid.n_grid_cells)
    else:
        _sub_selection = sub_selection

    return (
        get_rlg_spatial_grad_mat(
            grid, grid.nx, axis=0, sub_selection=_sub_selection, which=which
        ),
        get_rlg_spatial_grad_mat(
            grid, grid.ny, axis=1, sub_selection=_sub_selection, which=which
        ),
        get_rlg_spatial_grad_mat(
            grid, grid.nz, axis=2, sub_selection=_sub_selection, which=which
        ),
    )


def get_rlg_perm_mat(
    grid: RectilinearGrid,
    n: int,
    axis: int,
    sub_selection: NDArrayInt,
) -> csc_array:
    # matrix for the spatial gradient along the axis
    mat = lil_array((grid.n_grid_cells, grid.n_grid_cells), dtype=np.float64)

    # no need to fill the mat with a single element
    if n < 2:
        return mat.tocsc()

    _slices1: List[slice] = [slice(0, n - 1), slice(None), slice(None)]
    _slices2: List[slice] = [slice(1, n), slice(None), slice(None)]
    slices1: Tuple[slice, slice, slice] = (
        _slices1[0 - axis],
        _slices1[1 - axis],
        _slices1[2 - axis],
    )
    slices2: Tuple[slice, slice, slice] = (
        _slices2[0 - axis],
        _slices2[1 - axis],
        _slices2[2 - axis],
    )

    # Forward scheme:
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        slices1,
        slices2,
        owner_indices_to_keep=sub_selection,
        neigh_indices_to_keep=sub_selection,
    )

    mat[idc_neigh, idc_owner] = np.ones(idc_owner.size)
    return mat.tocsc()


def make_rlg_spatial_permutation_matrices(
    grid: RectilinearGrid, sub_selection: Optional[NDArrayInt] = None
) -> Tuple[csc_array, csc_array, csc_array]:
    """
    Make matrices to compute the spatial permutations along x and y axes of a field.

    Parameters
    ----------
    grid : RectilinearGrid
        Grid of the field
    sub_selection : Optional[NDArrayInt], optional
        Optional sub selection of the field. Non selected elements will be
        ignored in the gradient computation (as if non existing). If None, all
        elements are used. By default None.

    Returns
    -------
    Tuple[csc_array, csc_array]
        Spatial permutation matrices for x and y axes.
    """
    if sub_selection is None:
        _sub_selection: NDArrayInt = np.arange(grid.n_grid_cells)
    else:
        _sub_selection = sub_selection

    return (
        get_rlg_perm_mat(grid, grid.nx, 0, _sub_selection),
        get_rlg_perm_mat(grid, grid.ny, 1, _sub_selection),
        get_rlg_perm_mat(grid, grid.nz, 2, _sub_selection),
    )


def resample_grid(
    original_grid: RectilinearGrid, factor_x: float, factor_y: float, factor_z: float
) -> RectilinearGrid:
    # use the max to avoid ending up with zero.
    _nx = int(max(np.ceil(original_grid.nx * factor_x).item(), 1))
    _ny = int(max(np.ceil(original_grid.ny * factor_y).item(), 1))
    _nz = int(max(np.ceil(original_grid.nz * factor_z).item(), 1))
    return RectilinearGrid(
        x0=original_grid.x0,
        y0=original_grid.y0,
        z0=original_grid.z0,
        dx=original_grid.nx * original_grid.dx / _nx,
        dy=original_grid.ny * original_grid.dy / _ny,
        dz=original_grid.nz * original_grid.dz / _nz,
        nx=_nx,
        ny=_ny,
        nz=_nz,
        rot_center=original_grid.rot_center,
        theta=original_grid.theta,
        phi=original_grid.phi,
        psi=original_grid.psi,
    )


def duplicative_upsample(array: NDArrayFloat, factor: int) -> NDArrayFloat:
    ny, nx = array.shape
    # Each value is divided among (factor x factor) refined cells
    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def conservative_upsample(array: NDArrayFloat, factor: int) -> NDArrayFloat:
    return duplicative_upsample(array, factor) / (factor**2)
