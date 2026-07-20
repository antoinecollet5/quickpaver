# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Rectilinear grid.

The flattened node-numbering convention (``node_number = ix + iy*nx +
iz*ny*nx``) and every public array shape are unchanged from the original
implementation.
"""

from __future__ import annotations

import abc
import math
from typing import List, Literal, Optional, Sequence, Tuple, Union

import matplotlib as mpl
import matplotlib.path
import numpy as np
import shapely
from scipy.sparse import coo_array, csc_array

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
    ix_arr = np.asarray(ix, dtype=np.int64)
    iy_arr = np.asarray(iy, dtype=np.int64)
    iz_arr = np.asarray(iz, dtype=np.int64)

    if indices_start_at_one:
        ix_arr = np.maximum(ix_arr - 1, 0)
        iy_arr = np.maximum(iy_arr - 1, 0)
        iz_arr = np.maximum(iz_arr - 1, 0)

    return ix_arr + iy_arr * nx + iz_arr * ny * nx


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

    __slots__ = ()

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


def _as_full_slice_tuple(
    span: Union[NDArrayInt, Tuple[slice, ...], slice], ndims: int
) -> Optional[Tuple[slice, ...]]:
    """Return ``span`` normalized to a tuple of ``ndims`` slices, or ``None``.

    ``None`` is returned whenever ``span`` isn't expressible as plain slices
    (e.g. it's a boolean mask or an integer index array), signalling that the
    caller must fall back to a general (dense-array) selection strategy.
    """
    if isinstance(span, slice):
        span = (span,) + (slice(None),) * (ndims - 1)
    if not isinstance(span, tuple) or len(span) != ndims:
        return None
    if not all(isinstance(s, slice) for s in span):
        return None
    return span  # ty:ignore[invalid-return-type]


def span_to_node_numbers_2d(
    span: Union[NDArrayInt, Tuple[slice, slice], slice],
    nx: int,
    ny: int,
) -> NDArrayInt:
    """
    Convert a 2D grid span to flattened node numbers.

    The input ``span`` is applied to a temporary boolean-like array with shape
    ``(nx, ny)``. All selected cells are converted from 2D grid indices
    ``(ix, iy)`` to flattened regular-grid node numbers using
    :func:`rlg_idx_to_nn`.

    Parameters
    ----------
    span : Union[NDArrayInt, Tuple[slice, slice], slice]
        NumPy-style selection applied to a 2D array of shape ``(nx, ny)``.
        Examples include ``slice(None)``, ``(slice(0, 2), slice(None))``,
        or an integer index array compatible with NumPy indexing.
    nx : int
        Number of grid cells along the x-axis.
    ny : int
        Number of grid cells along the y-axis.

    Returns
    -------
    NDArrayInt
        One-dimensional array of flattened node numbers corresponding to the
        selected cells. The returned array has dtype ``np.int32``.

    Notes
    -----
    The flattened numbering convention is:

    ``node_number = ix + iy * nx``

    This function operates on grid cells, not on grid vertices.

    When ``span`` is made only of ``slice`` objects (the common case for
    finite-volume neighbour spans), indices are derived directly from the
    slice bounds with :func:`numpy.meshgrid` instead of materialising a full
    ``(nx, ny)`` dense array and scanning it with :func:`numpy.nonzero`. This
    avoids an ``O(nx*ny)`` allocation and scan whenever the selection can be
    described analytically. Non-slice selections (boolean masks, integer
    index arrays) fall back to the dense-array approach.
    """
    slices = _as_full_slice_tuple(span, ndims=2)
    if slices is not None:
        sx, sy = slices
        rx = range(*sx.indices(nx))
        ry = range(*sy.indices(ny))
        ix_grid, iy_grid = np.meshgrid(rx, ry, indexing="ij")
        return np.asarray(
            rlg_idx_to_nn(ix_grid.reshape(-1), nx=nx, iy=iy_grid.reshape(-1), ny=ny),
            dtype=np.int32,
        )

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
    """
    Convert a 3D grid span to flattened node numbers.

    The input ``span`` is applied to a temporary boolean-like array with shape
    ``(nx, ny, nz)``. All selected cells are converted from 3D grid indices
    ``(ix, iy, iz)`` to flattened regular-grid node numbers using
    :func:`rlg_idx_to_nn`.

    Parameters
    ----------
    span : Union[NDArrayInt, Tuple[slice, slice, slice], slice]
        NumPy-style selection applied to a 3D array of shape ``(nx, ny, nz)``.
        Examples include ``slice(None)``,
        ``(slice(0, 2), slice(None), slice(None))``, or an integer index array
        compatible with NumPy indexing.
    nx : int
        Number of grid cells along the x-axis.
    ny : int
        Number of grid cells along the y-axis.
    nz : int
        Number of grid cells along the z-axis.

    Returns
    -------
    NDArrayInt
        One-dimensional array of flattened node numbers corresponding to the
        selected cells. The returned array has dtype ``np.int32``.

    Notes
    -----
    The flattened numbering convention is:

    ``node_number = ix + iy * nx + iz * ny * nx``

    This function operates on grid cells, not on grid vertices.

    When ``span`` is made only of ``slice`` objects (the common case for
    finite-volume neighbour spans, e.g. ``(slice(0, n-1), slice(None),
    slice(None))``), indices are derived directly from the slice bounds with
    :func:`numpy.meshgrid` instead of materialising a full ``(nx, ny, nz)``
    dense array and scanning it with :func:`numpy.nonzero`. This avoids an
    ``O(nx*ny*nz)`` allocation and scan whenever the selection can be
    described analytically, which is significantly faster for large grids.
    Non-slice selections (boolean masks, integer index arrays) fall back to
    the dense-array approach.
    """
    slices = _as_full_slice_tuple(span, ndims=3)
    if slices is not None:
        sx, sy, sz = slices
        rx = range(*sx.indices(nx))
        ry = range(*sy.indices(ny))
        rz = range(*sz.indices(nz))
        ix_grid, iy_grid, iz_grid = np.meshgrid(rx, ry, rz, indexing="ij")
        return np.asarray(
            rlg_idx_to_nn(
                ix_grid.reshape(-1),
                nx=nx,
                iy=iy_grid.reshape(-1),
                ny=ny,
                iz=iz_grid.reshape(-1),
            ),
            dtype=np.int32,
        )

    _a = np.zeros((nx, ny, nz))
    _a[span] = 1.0
    ix, iy, iz = np.nonzero(_a)
    return np.array(rlg_idx_to_nn(ix, nx=nx, iy=iy, ny=ny, iz=iz), dtype=np.int32)


def get_array_borders_selection(nx: int, ny: int) -> NDArrayBool:
    """
    Get a selection of the array border as a bool array.

    Note
    ----
    There is no border for an axis of dim 1.

    Parameters
    ----------
    nx: int
        Number of grid cells along the x axis.
    ny: int
        Number of grid cells along the y axis.

    Returns
    -------
    NDArrayBool
        Boolean array with shape ``(nx, ny)``.
    """
    border = np.zeros((nx, ny), dtype=np.bool_)

    if nx == 0 or ny == 0:
        return border

    border[0, :] = True
    border[-1, :] = True
    border[:, 0] = True
    border[:, -1] = True

    return border


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


class RectilinearGrid(Grid):
    """
    Represent a rectilinear 3D grid.

    A rectilinear grid is defined by its centre coordinates, cell dimensions,
    number of cells along each axis, and optional Euler-angle rotations. The grid
    is initially constructed in a local, axis-aligned coordinate system and can be
    rotated around its centre to obtain world-space coordinates.

    The grid stores cell-based information, not vertex-based information. Grid
    indices therefore refer to cells/voxels, and flattened node numbers follow
    the regular-grid indexing convention used by :func:`rlg_idx_to_nn`.

    Parameters
    ----------
    cx : float, optional
        X coordinate of the grid centre in world space, by default 0.0.
    cy : float, optional
        Y coordinate of the grid centre in world space, by default 0.0.
    cz : float, optional
        Z coordinate of the grid centre in world space, by default 0.0.
    dx : float, optional
        Cell size along the local x-axis, in metres, by default 1.0.
    dy : float, optional
        Cell size along the local y-axis, in metres, by default 1.0.
    dz : float, optional
        Cell size along the local z-axis, in metres, by default 1.0.
    nx : int, optional
        Number of cells along the local x-axis, by default 1.
    ny : int, optional
        Number of cells along the local y-axis, by default 1.
    nz : int, optional
        Number of cells along the local z-axis, by default 1.
    theta : float, optional
        Rotation angle around the z-axis, in degrees, by default 0.0.
    phi : float, optional
        Rotation angle around the y-axis, in degrees, by default 0.0.
    psi : float, optional
        Rotation angle around the x-axis, in degrees, by default 0.0.

    Attributes
    ----------
    rot_center : tuple of float
        Rotation centre of the grid. This is always equal to
        ``(cx, cy, cz)``.
    shape : tuple of int
        Grid shape as ``(nx, ny, nz)``.
    dims : tuple of float
        Cell dimensions as ``(dx, dy, dz)``.
    n_grid_cells : int
        Total number of grid cells, equal to ``nx * ny * nz``.
    origin : tuple of float
        World-space coordinates of the rotated lower corner of the grid.
    origin_coords : NDArrayFloat
        World-space coordinates of all cell lower corners, with shape
        ``(3, nx, ny, nz)``.
    center_coords : NDArrayFloat
        World-space coordinates of all cell centres, with shape
        ``(3, nx, ny, nz)``.
    bounds : NDArrayFloat
        Axis-aligned world-space bounds of the rotated grid, with shape
        ``(3, 2)`` and rows ``[[xmin, xmax], [ymin, ymax], [zmin, zmax]]``.

    Notes
    -----
    The rotation order applied by :meth:`_rotate_coords` is:

    ``R_x(psi) @ R_y(phi) @ R_z(theta)``

    where angles are provided in degrees and internally converted to radians.
    Because matrix multiplication is applied from right to left, coordinates are
    first rotated around the z-axis, then around the y-axis, and finally around
    the x-axis.

    The rotation pivot is always the grid centre ``(cx, cy, cz)``.

    This class uses ``__slots__`` and therefore does not expose an instance
    ``__dict__``. Only the attributes listed in ``__slots__`` can be assigned.

    Per-cell coordinate/index stacks (``indices``, ``origin_coords``,
    ``center_coords``, ...) keep the natural ``(3, nx, ny, nz)`` public
    shape. Internally they are built as ``(3, nz, ny, nx)`` C-contiguous
    arrays and exposed via a zero-copy transpose view (see the module
    docstring and :meth:`_grid_shaped_view`), so flattening them with
    ``.reshape(3, -1, order="F")`` to get node-number order
    (``node_number = ix + iy*nx + iz*ny*nx``) is free instead of requiring a
    copy.
    """

    __slots__ = (
        "cx",
        "cy",
        "cz",
        "dx",
        "dy",
        "dz",
        "nx",
        "ny",
        "nz",
        "theta",
        "phi",
        "psi",
    )

    def __init__(
        self,
        cx: float = 0.0,
        cy: float = 0.0,
        cz: float = 0.0,
        dx: float = 1.0,
        dy: float = 1.0,
        dz: float = 1.0,
        nx: int = 1,
        ny: int = 1,
        nz: int = 1,
        theta: float = 0.0,
        phi: float = 0.0,
        psi: float = 0.0,
    ) -> None:
        """
        Initialize the instance.

        Parameters
        ----------
        cx : float
            Grid centre X (world space).
        cy : float
            Grid centre Y (world space).
        cz : float
            Grid centre Z (world space).
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
        theta : float
            z-axis rotation angle in degrees with (cx, cy, cz) as origin.
        phi : float
            y-axis-rotation angle in degrees with (cx, cy, cz) as origin.
        psi : float
            x-axis-rotation angle in degrees with (cx, cy, cz) as origin.
        """
        if dx <= 0.0 or dy <= 0.0 or dz <= 0.0:
            raise ValueError("dx, dy and dz must be strictly positive.")

        if nx < 1 or ny < 1 or nz < 1:
            raise ValueError("nx, ny and nz must be positive integers.")

        self.cx: float = cx
        self.cy: float = cy
        self.cz: float = cz
        self.dx: float = dx
        self.dy: float = dy
        self.dz: float = dz
        self.nx: int = int(nx)
        self.ny: int = int(ny)
        self.nz: int = int(nz)
        self.theta = theta
        self.phi = phi
        self.psi = psi

    def __str__(self) -> str:
        """Return a string representation of the instance."""
        return (
            "RectilinearGrid(\n"
            f"\tcx = {self.cx}\n"
            f"\tcy = {self.cy}\n"
            f"\tcz = {self.cz}\n"
            f"\tdx = {self.dx}\n"
            f"\tdy = {self.dy}\n"
            f"\tdz = {self.dz}\n"
            f"\tnx = {self.nx}\n"
            f"\tny = {self.ny}\n"
            f"\tnz = {self.nz}\n"
            f"\ttheta = {self.theta}\n"
            f"\tphi = {self.phi}\n"
            f"\tpsi = {self.psi}\n"
            ")"
        )

    def __repr__(self) -> str:
        return (
            "RectilinearGrid("
            f"center = {self.rot_center}, "
            f"origin = {self.origin}, "
            f"dx = {self.dx}, "
            f"dy = {self.dy}, "
            f"dz = {self.dz}, "
            f"nx = {self.nx}, "
            f"ny = {self.ny}, "
            f"nz = {self.nz}, "
            f"x0 = {self.x0}, "
            f"y0 = {self.y0}, "
            f"z0 = {self.z0}, "
            f"theta = {self.theta}, "
            f"phi = {self.phi}, "
            f"psi = {self.psi})"
        )

    @property
    def rot_center(self) -> Tuple[float, float, float]:
        """Rotation pivot — always the grid centre."""
        return (self.cx, self.cy, self.cz)

    @property
    def _local_origin(self) -> np.ndarray:
        """Corner origin in the LOCAL (unrotated) frame, relative to centre.

        Returns
        -------
        np.ndarray, shape (3,)
            Offset from centre to the smallest corner.
        """
        return np.array(
            [
                -self.nx * self.dx / 2.0,
                -self.ny * self.dy / 2.0,
                -self.nz * self.dz / 2.0,
            ]
        )

    @property
    def origin(self) -> Tuple[float, float, float]:
        """Corner origin in WORLD space (smallest corner after rotation).

        Returns
        -------
        tuple of (float, float, float)
            ``(x0, y0, z0)`` in world coordinates.
        """
        world = self._rotate_coords(
            self._local_origin.reshape(3, 1) + np.array([[self.cx, self.cy, self.cz]]).T
        )
        return (float(world[0, 0]), float(world[1, 0]), float(world[2, 0]))

    @property
    def x0(self) -> float:
        return self.origin[0]

    @property
    def y0(self) -> float:
        return self.origin[1]

    @property
    def z0(self) -> float:
        return self.origin[2]

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
        """Return the grid indices with shape (3, nx, ny, nz).

        Note
        ----
        Internally this is built as a C-contiguous ``(3, nz, ny, nx)`` stack
        and exposed as a ``(3, nx, ny, nz)`` *view* via :func:`numpy.transpose`
        (metadata-only, no data copy). This keeps the natural, intuitive
        public shape — matching every other x/y/z-ordered convention in this
        module — while still allowing ``.reshape(3, -1, order="F")`` to
        flatten it into node-number order (x fastest) as a zero-copy view
        instead of the copy a plain ``(3, nx, ny, nz)`` C-contiguous array
        would require. See :func:`_grid_shaped_view` for the general-purpose
        version of this trick used throughout the class.
        """
        iz, iy, ix = np.meshgrid(
            range(self.nz), range(self.ny), range(self.nx), indexing="ij"
        )
        stacked = np.asarray([ix, iy, iz], dtype=np.int32)  # shape (3, nz, ny, nx)
        return stacked.transpose(0, 3, 2, 1)  # shape (3, nx, ny, nz), zero-copy view

    def _grid_shaped_view(self, flat: NDArrayFloat) -> NDArrayFloat:
        """Reshape a flat ``(3, n_grid_cells)`` array to ``(3, nx, ny, nz)``.

        This is the zero-copy counterpart to
        ``flat.reshape(3, self.nx, self.ny, self.nz, order="F")``: reshaping
        a freshly-computed, C-contiguous ``(3, N)`` array directly into
        ``(3, nx, ny, nz)`` with ``order="F"`` forces a real copy, because
        Fortran order reads the *first* axis fastest while the source memory
        has the *last* axis fastest. Reshaping into the reversed
        ``(3, nz, ny, nx)`` shape first is a free view (default C order
        matches the source memory exactly), and the subsequent
        ``transpose`` back to the natural ``(3, nx, ny, nz)`` shape is
        metadata-only. The result is indistinguishable from a "real"
        ``(3, nx, ny, nz)`` array to calling code — including a later
        ``.reshape(3, -1, order="F")`` to flatten it back, which stays a
        zero-copy view too.
        """
        return flat.reshape(3, self.nz, self.ny, self.nx).transpose(0, 3, 2, 1)

    @property
    def _non_rotated_origin_coords(self) -> NDArrayFloat:
        """Grid cell corner coordinates in the unrotated frame.

        Shape ``(3, nx, ny, nz)``.
        """
        local = (
            self.indices.reshape(3, -1, order="F")  # zero-copy, see `indices`
            * np.array([[self.dx, self.dy, self.dz]]).T
            + self._local_origin.reshape(3, 1)  # relative to centre
            + np.array([[self.cx, self.cy, self.cz]]).T  # shift to world
        )
        return self._grid_shaped_view(local)

    def _rotate_coords(self, coords: NDArrayFloat) -> NDArrayFloat:
        """
        Rotate the coordinates.

        Parameters
        ----------
        coords: NDArrayFloat
            Expected shape (3, nx, ny, nz), or any shape whose first axis
            has length 3 (e.g. (3, 1) or (3, 8)).

        Return
        ------
        NDArrayFloat
            The rotated coordinates, with the same shape as ``coords``.
        """
        c = np.array([[self.cx, self.cy, self.cz]]).T
        flat = coords.reshape(3, -1) - c
        rotated = (
            _rotation_x(np.deg2rad(self.psi))
            @ _rotation_y(np.deg2rad(self.phi))
            @ _rotation_z(np.deg2rad(self.theta))
            @ flat
        ) + c
        return rotated.reshape(coords.shape)

    def copy(self) -> RectilinearGrid:
        """Return a deepcopy of the instance."""
        return RectilinearGrid(
            cx=self.cx,
            cy=self.cy,
            cz=self.cz,
            dx=self.dx,
            dy=self.dy,
            dz=self.dz,
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            theta=self.theta,
            phi=self.phi,
            psi=self.psi,
        )

    @property
    def origin_coords(self) -> NDArrayFloat:
        """Return the grid meshes origin coordinates with shape (3, nx, ny, nz)."""
        flat = self._non_rotated_origin_coords.reshape(3, -1, order="F")
        rotated_flat = self._rotate_coords(flat)
        return self._grid_shaped_view(rotated_flat)

    @property
    def x_indices(self) -> NDArrayInt:
        """Return the grid meshes x-indices as 1D array, in node-number order."""
        return self.indices[0].reshape(-1, order="F")

    @property
    def y_indices(self) -> NDArrayInt:
        """Return the grid meshes y-indices as 1D array, in node-number order."""
        return self.indices[1].reshape(-1, order="F")

    @property
    def z_indices(self) -> NDArrayInt:
        """Return the grid meshes z-indices as 1D array, in node-number order."""
        return self.indices[2].reshape(-1, order="F")

    @property
    def center_coords(self) -> NDArrayFloat:
        """Return the grid meshes center coordinates with shape (3, nx, ny, nz)."""
        flat = (
            self._non_rotated_origin_coords.reshape(3, -1, order="F")
            + np.array([[self.dx / 2, self.dy / 2, self.dz / 2]]).T
        )
        rotated_flat = self._rotate_coords(flat)
        return self._grid_shaped_view(rotated_flat)

    @property
    def non_rot_center_coords(self) -> NDArrayFloat:
        """Return the non rotated grid cell center coords with shape (3, nx, ny, nz)."""
        flat = (
            self._non_rotated_origin_coords.reshape(3, -1, order="F")
            + np.array([[self.dx / 2, self.dy / 2, self.dz / 2]]).T
        )
        return self._grid_shaped_view(flat)

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
        flat = (
            self._non_rotated_origin_coords.reshape(3, -1, order="F")
            + np.array([[self.dx, self.dy, self.dz]]).T
        )
        rotated_flat = self._rotate_coords(flat)
        return self._grid_shaped_view(rotated_flat)

    @property
    def bounding_box_vertices_coordinates(self) -> NDArrayFloat:
        """Return the coordinates of the 8 rotated bounding-box vertices."""
        lx0, ly0, lz0 = self._local_origin
        lx1 = lx0 + self.nx * self.dx
        ly1 = ly0 + self.ny * self.dy
        lz1 = lz0 + self.nz * self.dz

        local_vertices = np.array(
            [
                [lx0, ly0, lz0],
                [lx1, ly0, lz0],
                [lx1, ly1, lz0],
                [lx0, ly1, lz0],
                [lx0, ly0, lz1],
                [lx1, ly0, lz1],
                [lx1, ly1, lz1],
                [lx0, ly1, lz1],
            ],
            dtype=float,
        ).T

        world_vertices = local_vertices + np.array([[self.cx, self.cy, self.cz]]).T

        return self._rotate_coords(world_vertices)

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

    def to_shapely(self, mask: Optional[NDArrayBool] = None) -> shapely.MultiPolygon:
        """
        Return a 2d regular grid to 2d shapely polygons.

        Parameters
        ----------
        grid : quickpaver.RectilinearGrid
            The input 2d regular grid.
        mask : Optional[NDArrayBool], optional
            Boolean mask with shape ``(nx, ny)`` selecting cells to include.
            Cells equal to ``True`` are converted to polygons. If ``None``,
            all cells are converted.

        Returns
        -------
        shapely.MultiPolygon
            - Square panels as a MultiPolygon
        """
        half_dx = self.dx / 2.0
        half_dy = self.dy / 2.0
        # the grid x-y coordinates, flattened in node-number order (x fastest)
        grid_2d_cc = self.non_rot_center_coords_2d.reshape(2, -1, order="F").T

        if mask is None:
            inside_points = grid_2d_cc
        else:
            inside_points = grid_2d_cc[np.asarray(mask, dtype=bool).ravel(order="F")]

        if inside_points.size == 0:
            return shapely.MultiPolygon([])

        return shapely.affinity.rotate(  # ty:ignore[possibly-missing-submodule]
            shapely.MultiPolygon(
                shapely.box(
                    inside_points[:, 0] - half_dx,
                    inside_points[:, 1] - half_dy,
                    inside_points[:, 0] + half_dx,
                    inside_points[:, 1] + half_dy,
                )
            ),
            angle=self.theta,
            origin=list(
                self.rot_center[:2]
            ),  # must convert to a list to avoid == issues with numpy
        )

    def to_pyvista(
        self,
        cell_data: Optional[dict[str, Union[NDArrayFloat, NDArrayInt]]] = None,
        representation: Literal["image", "rectilinear", "structured"] = "image",
        apply_rotation: bool = True,
    ) -> object:
        """
        Convert the grid to a PyVista dataset.

        The preferred representation is ``"image"``, which returns a compact
        :class:`pyvista.ImageData` object. Since this grid has uniform cell
        spacing along each local axis, ``ImageData`` is the most memory-efficient
        PyVista representation.

        If rotations are enabled and ``representation="image"``, the grid
        orientation is stored using the PyVista ``direction_matrix`` argument.
        This preserves a compact image-data representation while representing
        the rotated coordinate frame.

        Parameters
        ----------
        cell_data : Optional[dict[str, Union[NDArrayFloat, NDArrayInt]]], optional
            Optional mapping of cell-data names to arrays. Each array must contain
            exactly ``n_grid_cells`` values, either as a flattened one-dimensional
            array (already in node-number order) or as an array with shape
            ``(nx, ny, nz)`` (the grid's natural axis convention). If ``None``,
            no cell data are attached. By default ``None``.
        representation : Literal["image", "rectilinear", "structured"], optional
            PyVista representation to return:

            - ``"image"`` returns a compact :class:`pyvista.ImageData`.
            - ``"rectilinear"`` returns a :class:`pyvista.RectilinearGrid`.
            This is only valid for non-rotated grids.
            - ``"structured"`` returns a :class:`pyvista.StructuredGrid`.
            This representation explicitly stores all grid points and supports
            rotations through PyVista geometry transforms.

            By default ``"image"``.
        apply_rotation : bool, optional
            Whether to apply the grid Euler rotations. If ``False``, the returned
            PyVista grid is axis-aligned. By default ``True``.

        Returns
        -------
        object
            PyVista dataset. Depending on ``representation``, this is usually one
            of:

            - :class:`pyvista.ImageData`
            - :class:`pyvista.RectilinearGrid`
            - :class:`pyvista.StructuredGrid`

        Raises
        ------
        ImportError
            If PyVista is not installed.
        ValueError
            If ``representation`` is invalid.
        ValueError
            If ``representation="rectilinear"`` is requested for a rotated grid.
        ValueError
            If a cell-data array does not contain exactly ``n_grid_cells`` values.
        RuntimeError
            If rotated ``ImageData`` is requested with a PyVista version that does
            not support ``direction_matrix``.

        Notes
        -----
        PyVista is imported locally so that it remains an optional dependency.

        The PyVista grid is built from vertices, not cell centres. Therefore, the
        PyVista point dimensions are ``(nx + 1, ny + 1, nz + 1)`` and the number
        of cells is ``nx * ny * nz``.

        The rotation order is consistent with :meth:`_rotate_coords`:

        ``R_x(psi) @ R_y(phi) @ R_z(theta)``

        For ``representation="structured"``, rotations are applied sequentially
        using PyVista as z, then y, then x rotations around ``self.rot_center``.

        VTK/PyVista's cell ordering requires x fastest, then y, then z — the
        same ``node_number = ix + iy*nx + iz*ny*nx`` convention used
        throughout this module. A cell-data array passed in with shape
        ``(nx, ny, nz)`` is flattened with ``order="F"`` to match that order.
        An array passed in pre-flattened is assumed to already be in
        node-number order.
        """
        try:
            import pyvista as pv
        except ImportError as exc:
            raise ImportError(
                "PyVista is required to export RectilinearGrid to PyVista. "
                "Install it with `pip install pyvista`."
            ) from exc

        if representation not in {"image", "rectilinear", "structured"}:
            raise ValueError(
                "representation must be one of {'image', 'rectilinear', 'structured'}."
            )

        has_rotation = (
            not np.isclose(self.theta, 0.0)
            or not np.isclose(self.phi, 0.0)
            or not np.isclose(self.psi, 0.0)
        )

        dimensions = (self.nx + 1, self.ny + 1, self.nz + 1)
        spacing = (self.dx, self.dy, self.dz)

        center = np.array([self.cx, self.cy, self.cz], dtype=float)

        unrotated_origin = tuple(center + self._local_origin)

        rotation_matrix = (
            _rotation_x(np.deg2rad(self.psi))
            @ _rotation_y(np.deg2rad(self.phi))
            @ _rotation_z(np.deg2rad(self.theta))
        )

        if representation == "image":
            if apply_rotation and has_rotation:
                rotated_origin = tuple(center + rotation_matrix @ self._local_origin)

                try:
                    pv_grid = pv.ImageData(
                        dimensions=dimensions,
                        spacing=spacing,
                        origin=rotated_origin,
                        direction_matrix=rotation_matrix,
                    )
                except TypeError as exc:
                    raise RuntimeError(
                        "Rotated PyVista ImageData requires a PyVista version "
                        "supporting the `direction_matrix` argument."
                    ) from exc
            else:
                pv_grid = pv.ImageData(
                    dimensions=dimensions,
                    spacing=spacing,
                    origin=unrotated_origin,
                )

        elif representation == "rectilinear":
            if apply_rotation and has_rotation:
                raise ValueError(
                    "A rotated grid cannot be represented as a true "
                    "pyvista.RectilinearGrid. Use representation='image' for "
                    "compact oriented ImageData, or representation='structured' "
                    "for explicitly rotated points."
                )

            image_grid = pv.ImageData(
                dimensions=dimensions,
                spacing=spacing,
                origin=unrotated_origin,
            )
            pv_grid = image_grid.cast_to_rectilinear_grid()

        else:
            image_grid = pv.ImageData(
                dimensions=dimensions,
                spacing=spacing,
                origin=unrotated_origin,
            )
            pv_grid = image_grid.cast_to_structured_grid()

            if apply_rotation and has_rotation:
                pivot = self.rot_center

                if not np.isclose(self.theta, 0.0):
                    pv_grid = pv_grid.rotate_z(
                        self.theta,
                        point=pivot,
                        inplace=False,
                    )

                if not np.isclose(self.phi, 0.0):
                    pv_grid = pv_grid.rotate_y(
                        self.phi,
                        point=pivot,
                        inplace=False,
                    )

                if not np.isclose(self.psi, 0.0):
                    pv_grid = pv_grid.rotate_x(
                        self.psi,
                        point=pivot,
                        inplace=False,
                    )

        if cell_data is not None:
            for name, values in cell_data.items():
                values_array = np.asarray(values)

                if values_array.size != self.n_grid_cells:
                    raise ValueError(
                        f"Cell-data array {name!r} must contain exactly "
                        f"{self.n_grid_cells} values, got {values_array.size}."
                    )

                pv_grid.cell_data[name] = values_array.reshape(-1, order="F")

        return pv_grid


def _get_vertices_centroid(
    vertices: Union[NDArrayFloat, List[Tuple[float, float]]],
) -> Tuple[float, float]:
    """Get the vertices centroid."""
    _x_list = [vertex[0] for vertex in vertices]
    _y_list = [vertex[1] for vertex in vertices]
    _len = len(vertices)
    _x = sum(_x_list) / _len
    _y = sum(_y_list) / _len
    return (_x, _y)


def _get_centroid_voxel_coords(
    vertices: Union[NDArrayFloat, List[Tuple[float, float]]],
    grid: RectilinearGrid,
    center_coords_2d: Optional[NDArrayFloat] = None,
) -> Tuple[Int, Int]:
    """
    For a given convex polygon an a 2D grid, give the centroid voxel.

    Parameters
    ----------
    vertices : Union[NDArrayFloat, List[Tuple[float, float]]]
        Coords of the convex polygon exterior ring with shape (M, 2)
    grid: RectilinearGrid,
        The grid definition (dimensions, position, etc.).
    center_coords_2d : Optional[NDArrayFloat], optional
        Precomputed ``grid.center_coords_2d`` (shape ``(2, nx, ny)``), to
        avoid recomputing the rotated coordinate grid (which involves a
        matrix multiply over every cell) when this function is called once
        per polygon in a loop. If ``None``, it is computed from ``grid``.
        By default ``None``.

    Returns
    -------
    Tuple[int, int]
        x, y coordinates of the centroid voxel.
    """
    if center_coords_2d is None:
        center_coords_2d = grid.center_coords_2d

    # This works for convex polygons only
    _x, _y = _get_vertices_centroid(vertices)
    # get the closer integer, in node-number (x-fastest) order.
    cx_flat = center_coords_2d[0].ravel("F")
    cy_flat = center_coords_2d[1].ravel("F")
    distances = np.square(cx_flat - _x) + np.square(cy_flat - _y)
    ix, iy, _ = rlg_nn_to_idx(int(np.argmin(distances)), nx=grid.nx, ny=grid.ny)
    return (int(ix), int(iy))


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
        Grid selections array with shape ``(nx, ny)``.
    """
    if len(polygons) != len(sel_ids):
        raise ValueError("polygons and sel_ids must have the same length.")

    _sel_ids = np.asarray(sel_ids)
    if np.any(_sel_ids == 0):
        raise ValueError(
            "0 cannot be part of sel_ids. It is reserved for empty selection."
        )

    # flatten points coordinates
    _sel_array = np.zeros((grid.nx, grid.ny), dtype=np.int32)

    # The mask sum ensure that a voxel is not selected twice
    mask_sum: Optional[NDArrayBool] = None
    for _polygon, cell_id in zip(polygons, _sel_ids):
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


def _get_free_grid_cells(selection) -> NDArrayBool:
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
    mask = np.logical_and(mask, _get_free_grid_cells(selection))
    return mask


def binary_dilation(
    seed_mask: NDArrayBool,
    domain_mask: NDArrayBool,
    iterations: int = 1,
) -> NDArrayBool:
    """
    Dilate a 2D boolean array within a constrained domain.

    The dilation uses 4-connectivity, meaning that each ``True`` cell expands
    to its direct horizontal and vertical neighbours only. Diagonal neighbours
    are not included.

    After each dilation step, cells outside ``domain_mask`` are forced to
    ``False``. This ensures that the dilated region never grows outside the
    allowed domain.

    Parameters
    ----------
    seed_mask : NDArrayBool
        Initial 2D boolean array to dilate. Cells equal to ``True`` are used as
        dilation seeds.
    domain_mask : NDArrayBool
        Boolean mask defining where dilation is allowed. Cells equal to ``True``
        belong to the allowed domain. Cells equal to ``False`` are excluded and
        are forced to remain ``False`` in the output.
    iterations : int, optional
        Number of dilation iterations to apply, by default 1. Each iteration
        expands the current ``True`` region by one cell in the four cardinal
        directions within ``domain_mask``.

    Returns
    -------
    NDArrayBool
        Dilated 2D boolean array with the same shape as ``seed_mask``. The result
        is always ``False`` outside ``domain_mask``.

    Raises
    ------
    ValueError
        If ``seed_mask`` and ``domain_mask`` do not have the same shape.
    ValueError
        If ``iterations`` is negative.
    """
    if seed_mask.shape != domain_mask.shape:
        raise ValueError("Seed_mask and domain_mask must have the same shape.")

    if iterations < 0:
        raise ValueError("Iterations must be non-negative.")

    arr = seed_mask.copy()

    for _ in range(iterations):
        old = arr.copy()

        arr[1:, :] |= old[:-1, :]
        arr[:-1, :] |= old[1:, :]
        arr[:, 1:] |= old[:, :-1]
        arr[:, :-1] |= old[:, 1:]

        arr[~domain_mask] = False

    return arr


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
        An already existing selection as starting point, with shape
        ``(nx, ny)``. The default is None.

    Returns
    -------
    NDArrayInt
        Selection array with shape ``(nx, ny)``.

    Notes
    -----
    Each polygon's point-in-polygon containment test against the grid is
    geometry-only and does not depend on the current selection state, so it
    is computed once here rather than being recomputed on every dilation
    iteration (as a naive per-iteration ``matplotlib.path.Path.contains_points``
    call would do). Only the free-cells part of the domain mask changes
    between iterations.
    """
    # Start by creating an empty grid with int type
    if selection is None:
        _selection = np.zeros((grid.nx, grid.ny), dtype=np.int32)
    else:
        _selection = selection.copy()

    # initiate _oldselection variable
    _old_selection = np.zeros_like(_selection)

    # Grid coordinates -> Flat array (computed once and reused for every
    # polygon, instead of re-triggering the grid's rotation math each time).
    _center_coords_2d = grid.center_coords_2d
    _grid_coords_2d = _center_coords_2d.reshape(2, -1, order="F").T

    # Create an initial selection for each cell (only one voxel selected)
    sel_ids = np.arange(len(polygons)) + 1
    for sel_id, vertices in zip(sel_ids, polygons):
        ix, iy = _get_centroid_voxel_coords(vertices, grid, _center_coords_2d)
        _selection[ix, iy] = sel_id

    # Precompute each polygon's containment mask once (geometry-only, does
    # not depend on the selection state, so it is invariant across dilation
    # iterations).
    containment_masks = [
        mpl.path.Path(vertices)
        .contains_points(_grid_coords_2d)
        .reshape(grid.nx, grid.ny, order="F")
        for vertices in polygons
    ]

    # Perform the dilation iteration by iteration to ensure a better split between
    # the selections.
    while np.not_equal(_selection, _old_selection).any():
        # Update the old_grid with the new one for the while
        _old_selection = _selection.copy()
        # Perform the dilation for each selection
        free_cells = _get_free_grid_cells(_selection)
        for sel_id, inside_polygon in zip(sel_ids, containment_masks):
            # The mask is free cells + the contained
            mask = np.logical_and(inside_polygon, free_cells)
            _selection[
                binary_dilation(_selection == sel_id, domain_mask=mask, iterations=1)
            ] = sel_id
            # A cell claimed by sel_id this sweep must stop being "free" for
            # the remaining polygons processed in this same sweep.
            free_cells = _get_free_grid_cells(_selection)
    return _selection


def _get_vertical_limits_indices(
    limits_in_m: NDArrayFloat,
    z0: float,
    dz: float,
    nz: int,
) -> Tuple[int, int]:
    """
    Convert vertical metric limits to inclusive z-index limits.
    Note
    ----
    We have to take the max with 0.0 because the selections could go beyond the grid.
    """
    bot_index = int(
        min(max(np.round((limits_in_m[0] - z0 - dz / 2.0) / dz, 0), 0), nz - 1)
    )
    top_index = int(
        min(max(np.round((limits_in_m[1] - z0 - dz / 2.0) / dz, 0), 0), nz - 1)
    )
    return bot_index, top_index


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
        An already existing selection as starting point, with shape
        ``(nx, ny, nz)``. The default is None.

    Returns
    -------
    NDArrayInt
        Selection array with shape ``(nx, ny, nz)``.

    Notes
    -----
    As in :func:`get_polygon_selection_with_dilation_2d`, each polygon's
    point-in-polygon containment mask and vertical index limits are
    geometry-only and independent of the current selection state, so both
    are computed once up front instead of being recomputed on every
    ``(z-slice, polygon)`` pair of every dilation iteration.
    """
    if selection is None:
        _selection = np.zeros((grid.nx, grid.ny, grid.nz), dtype=np.int32)
    else:
        _selection = selection.copy()

    _vertical_limits: NDArrayFloat = np.asarray(vertical_limits).reshape(-1, 2)

    if _vertical_limits.shape[0] != len(polygons):
        raise ValueError(
            "vertical_limits must contain one [bottom, top] pair per polygon."
        )

    # Grid coordinates -> Flat array (2d, horizontal slice), computed once.
    _center_coords_2d = grid.center_coords_2d
    _grid_coords_2d = _center_coords_2d.reshape(2, -1, order="F").T

    sel_ids = np.arange(len(polygons)) + 1

    # Precompute per-polygon vertical index limits and 2D containment masks
    # once; both are invariant across z-slices and dilation iterations.
    limits_per_polygon = [
        _get_vertical_limits_indices(_vertical_limits[i, :], grid.z0, grid.dz, grid.nz)
        for i in range(len(polygons))
    ]
    containment_masks = [
        mpl.path.Path(vertices)
        .contains_points(_grid_coords_2d)
        .reshape(grid.nx, grid.ny, order="F")
        for vertices in polygons
    ]

    # Create an initial selection for each cell (only one voxel selected)
    for i, (sel_id, vertices) in enumerate(zip(sel_ids, polygons)):
        _limits = limits_per_polygon[i]
        # Initial selection
        ix, iy = _get_centroid_voxel_coords(vertices, grid, _center_coords_2d)
        _selection[ix, iy, _limits[0] : _limits[1] + 1] = sel_id

    # initiate _oldselection variable
    _old_selection = np.zeros_like(_selection)

    # Perform the dilation iteration by iteration to ensure a better split between
    # the selections.
    while np.not_equal(_selection, _old_selection).any():
        # Update the oldselection with the new one for the while
        _old_selection = _selection.copy()

        for iz in range(grid.nz):
            # Perform the dilation for each selection
            for i, (sel_id, inside_polygon) in enumerate(
                zip(sel_ids, containment_masks)
            ):
                _limits = limits_per_polygon[i]
                if iz < _limits[0] or iz > _limits[1]:
                    continue
                # The mask is free cells + the contained. Free cells are
                # re-derived from the current (possibly just-updated) slice
                # so that sequential claims within the same sweep are
                # respected, matching the original behaviour.
                mask = np.logical_and(
                    inside_polygon, _get_free_grid_cells(_selection[:, :, iz])
                )
                _selection[:, :, iz][
                    binary_dilation(
                        _selection[:, :, iz] == sel_id, domain_mask=mask, iterations=1
                    )
                ] = sel_id
    return _selection


def _keep_a_b_if_c_in_a(
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
    """
    Return paired owner and neighbour flattened grid-cell indices.

    The function converts two matching 3D spans into flattened node-number arrays:
    one span for owner cells and one span for neighbouring cells. The returned
    arrays are paired element-wise, meaning that ``indices_owner[k]`` is connected
    to ``indices_neigh[k]``.

    Optional keep-lists can be used to remove pairs where either the owner or the
    neighbour does not belong to a selected subset of grid cells.

    Parameters
    ----------
    grid : RectilinearGrid
        Rectilinear grid defining the shape and flattened indexing convention.
    span_owner : Tuple[slice, slice, slice]
        Three-dimensional NumPy-style span selecting owner cells in an array of
        shape ``(grid.nz, grid.ny, grid.nx)``.
    span_neigh : Tuple[slice, slice, slice]
        Three-dimensional NumPy-style span selecting neighbour cells in an array
        of shape ``(grid.nz, grid.ny, grid.nx)``. This span must select the same
        number of cells as ``span_owner`` so that owner and neighbour indices can
        be paired element-wise.
    owner_indices_to_keep : Optional[NDArrayInt], optional
        Optional flattened owner-cell indices to keep. If provided, only pairs
        whose owner index belongs to this array are retained. By default ``None``.
    neigh_indices_to_keep : Optional[NDArrayInt], optional
        Optional flattened neighbour-cell indices to keep. If provided, only
        pairs whose neighbour index belongs to this array are retained. By
        default ``None``.

    Returns
    -------
    Tuple[NDArrayInt, NDArrayInt]
        Two one-dimensional arrays ``(indices_owner, indices_neigh)`` containing
        paired flattened grid-cell indices.

    Notes
    -----
    The flattened numbering convention is:

    ``node_number = ix + iy * grid.nx + iz * grid.ny * grid.nx``

    This helper is mainly used to build sparse finite-difference, finite-volume,
    or permutation matrices where each row/column contribution is based on
    owner-neighbour cell pairs.
    """
    # Get owner and neighbour flattened indices.
    indices_owner: NDArrayInt = span_to_node_numbers_3d(
        span_owner, nx=grid.nx, ny=grid.ny, nz=grid.nz
    )
    indices_neigh: NDArrayInt = span_to_node_numbers_3d(
        span_neigh, nx=grid.nx, ny=grid.ny, nz=grid.nz
    )

    if owner_indices_to_keep is not None:
        indices_owner, indices_neigh = _keep_a_b_if_c_in_a(
            indices_owner, indices_neigh, owner_indices_to_keep
        )

    if neigh_indices_to_keep is not None:
        indices_neigh, indices_owner = _keep_a_b_if_c_in_a(
            indices_neigh, indices_owner, neigh_indices_to_keep
        )

    return indices_owner, indices_neigh


def get_rlg_spatial_grad_mat(
    grid: RectilinearGrid,
    n: int,
    axis: int,
    sub_selection: Optional[NDArrayInt],
    which: Literal["forward", "backward", "both"] = "both",
) -> csc_array:
    """
    Build a sparse spatial-gradient matrix along one grid axis.

    The matrix approximates a finite-volume gradient-like operator along the
    selected axis. For each valid owner-neighbour cell pair, the matrix adds a
    positive contribution on the owner cell and a negative contribution on the
    neighbouring cell:

    ``mat[owner_index, owner_index] += interface_area / cell_volume``

    ``mat[owner_index, neighbour_index] -= interface_area / cell_volume``

    If ``field`` is a flattened grid-cell array, then ``mat @ field`` returns the
    directional difference between owner and neighbour values, scaled by the
    corresponding interface-area-to-volume ratio.

    Parameters
    ----------
    grid : RectilinearGrid
        Rectilinear grid defining the cell dimensions, shape, indexing
        convention, and total number of grid cells.
    n : int
        Number of cells along the selected axis. This is usually one of
        ``grid.nx``, ``grid.ny``, or ``grid.nz`` depending on ``axis``.
    axis : int
        Axis along which the gradient matrix is assembled:

        - ``0`` for the x-axis,
        - ``1`` for the y-axis,
        - ``2`` for the z-axis.

    sub_selection : NDArrayInt
        Flattened grid-cell indices to include in the gradient computation.
        Owner and neighbour cells must both belong to this selection to be
        connected in the matrix.
    which : Literal["forward", "backward", "both"], optional
        Difference scheme to assemble:

        - ``"forward"`` uses owner cells and their forward neighbours.
        - ``"backward"`` uses owner cells and their backward neighbours.
        - ``"both"`` combines forward and backward contributions.

        By default ``"both"``.

    Returns
    -------
    csc_array
        Sparse gradient matrix with shape
        ``(grid.n_grid_cells, grid.n_grid_cells)`` in CSC format.

    Raises
    ------
    ValueError
        If ``which`` is not one of ``"forward"``, ``"backward"``, or ``"both"``.

    Notes
    -----
    If ``n < 2``, no neighbour exists along the selected axis and an empty sparse
    matrix is returned.

    The coefficient magnitude is computed as:

    ``interface_area / cell_volume``

    with the interface area selected according to ``axis``:

    - x-axis: ``grid.gamma_ij_x_m2``
    - y-axis: ``grid.gamma_ij_y_m2``
    - z-axis: ``grid.gamma_ij_z_m2``

    The flattened grid-cell numbering convention is:

    ``node_number = ix + iy * grid.nx + iz * grid.ny * grid.nx``
    """
    if axis not in {0, 1, 2}:
        raise ValueError("axis must be one of {0, 1, 2}.")

    if which not in {"forward", "backward", "both"}:
        raise ValueError("which must be one of {'forward', 'backward', 'both'}.")

    shape = (grid.n_grid_cells, grid.n_grid_cells)

    # No neighbour exists if there is fewer than two cells along this axis.
    if n < 2:
        return coo_array(shape, dtype=np.float64).tocsc()

    tmp = {
        0: grid.gamma_ij_x_m2,
        1: grid.gamma_ij_y_m2,
        2: grid.gamma_ij_z_m2,
    }[axis] / grid.grid_cell_volume_m3

    # Spans are expressed against a (nx, ny, nz)-shaped array, so the array
    # axis matches the spatial axis directly.
    _slices1: List[slice] = [slice(None), slice(None), slice(None)]
    _slices2: List[slice] = [slice(None), slice(None), slice(None)]
    _slices1[axis] = slice(0, n - 1)
    _slices2[axis] = slice(1, n)
    slices1: Tuple[slice, slice, slice] = tuple(_slices1)  # ty:ignore[invalid-assignment]
    slices2: Tuple[slice, slice, slice] = tuple(_slices2)  # ty:ignore[invalid-assignment]

    # Accumulate (row, col, value) triplets and build the sparse matrix once
    # via COO -> CSC. This avoids scipy's LIL fancy-indexing path
    # (`__getitem__`/`__setitem__` on array-of-indices), which dominates
    # runtime for large grids because each `mat[idx, idx] += ...` call reads
    # back the current values before writing. COO->CSC conversion sums
    # duplicate (row, col) entries automatically, which reproduces the same
    # `+=`/`-=` accumulation semantics as the previous LIL-based code.
    rows: List[NDArrayInt] = []
    cols: List[NDArrayInt] = []
    data: List[NDArrayFloat] = []

    def _add_scheme(
        span_owner: Tuple[slice, slice, slice], span_neigh: Tuple[slice, slice, slice]
    ) -> None:
        idc_owner, idc_neigh = get_owner_neigh_indices(
            grid,
            span_owner,
            span_neigh,
            owner_indices_to_keep=sub_selection,
            neigh_indices_to_keep=sub_selection,
        )
        ones = np.ones(idc_owner.size)
        rows.append(idc_owner)
        cols.append(idc_neigh)
        data.append(-tmp * ones)
        rows.append(idc_owner)
        cols.append(idc_owner)
        data.append(tmp * ones)

    if which in ["forward", "both"]:
        # Forward scheme only: see PhD manuscript, chapter 7 for the explanaition.
        _add_scheme(slices1, slices2)

    if which in ["backward", "both"]:
        # Backward scheme: owner cells are connected to backward neighbours.
        _add_scheme(slices2, slices1)

    if not rows:
        return coo_array(shape, dtype=np.float64).tocsc()

    row = np.concatenate(rows)
    col = np.concatenate(cols)
    val = np.concatenate(data)
    return coo_array((val, (row, col)), shape=shape, dtype=np.float64).tocsc()


def make_rlg_spatial_gradient_matrices(
    grid: RectilinearGrid,
    sub_selection: Optional[NDArrayInt] = None,
    which: Literal["forward", "backward", "both"] = "both",
) -> Tuple[csc_array, csc_array, csc_array]:
    """
    Build sparse spatial-gradient matrices for a rectilinear grid.

    The returned matrices approximate spatial gradients along the x-, y-, and
    z-axes of a flattened grid-cell field. Each matrix is built from neighbouring
    cell pairs along one axis and uses the finite-volume ratio between interface
    area and cell volume.

    If ``field`` is a one-dimensional array of shape ``(grid.n_grid_cells,)``,
    then the directional gradient-like quantities can be obtained with:

    ``grad_x = gx @ field``

    ``grad_y = gy @ field``

    ``grad_z = gz @ field``

    where ``gx``, ``gy``, and ``gz`` are the matrices returned by this function.

    Parameters
    ----------
    grid : RectilinearGrid
        Rectilinear grid defining the cell dimensions, shape, indexing
        convention, and total number of grid cells.
    sub_selection : Optional[NDArrayInt], optional
        Optional flattened grid-cell indices to include in the gradient
        computation. Neighbour pairs are kept only when both the owner cell and
        the neighbouring cell belong to this selection. If ``None``, all grid
        cells are used. By default ``None``.
    which : Literal["forward", "backward", "both"], optional
        Difference scheme to assemble along each axis:

        - ``"forward"`` builds contributions from each cell to its forward
          neighbour.
        - ``"backward"`` builds contributions from each cell to its backward
          neighbour.
        - ``"both"`` combines forward and backward contributions.

        By default ``"both"``.

    Returns
    -------
    Tuple[csc_array, csc_array, csc_array]
        Sparse gradient matrices ``(gx, gy, gz)`` in CSC format, where:

        - ``gx`` acts along the x-axis,
        - ``gy`` acts along the y-axis,
        - ``gz`` acts along the z-axis.

        Each matrix has shape ``(grid.n_grid_cells, grid.n_grid_cells)``.

    Notes
    -----
    The flattened grid-cell numbering convention is:

    ``node_number = ix + iy * grid.nx + iz * grid.ny * grid.nx``

    The coefficient magnitude along each axis is computed as:

    ``interface_area / cell_volume``

    which gives:

    - x-axis: ``grid.gamma_ij_x_m2 / grid.grid_cell_volume_m3``
    - y-axis: ``grid.gamma_ij_y_m2 / grid.grid_cell_volume_m3``
    - z-axis: ``grid.gamma_ij_z_m2 / grid.grid_cell_volume_m3``

    Boundary cells without a valid neighbour along a given direction do not
    receive a contribution for that direction.
    """
    # NOTE: sub_selection is intentionally left as None (rather than eagerly
    # materialized as np.arange(grid.n_grid_cells)) when the caller doesn't
    # restrict cells. get_owner_neigh_indices treats None as "keep
    # everything" and skips the (otherwise no-op) np.isin filtering pass,
    # which is a measurable cost on large grids.
    return (
        get_rlg_spatial_grad_mat(
            grid, grid.nx, axis=0, sub_selection=sub_selection, which=which
        ),
        get_rlg_spatial_grad_mat(
            grid, grid.ny, axis=1, sub_selection=sub_selection, which=which
        ),
        get_rlg_spatial_grad_mat(
            grid, grid.nz, axis=2, sub_selection=sub_selection, which=which
        ),
    )


def get_rlg_perm_mat(
    grid: RectilinearGrid,
    n: int,
    axis: int,
    sub_selection: Optional[NDArrayInt],
) -> csc_array:
    """
    Build a sparse forward-neighbour permutation matrix along one grid axis.

    The returned matrix maps values from owner cells to their forward neighbours
    along the selected axis. For each valid owner/neighbour pair, the matrix
    contains one non-zero coefficient:

    ``mat[neighbour_index, owner_index] = 1``

    Therefore, if ``field`` is a flattened grid-cell array, then
    ``mat @ field`` gives an array where each forward neighbour receives the
    value of its owner cell. Cells without a valid backward owner along the
    selected axis remain zero.

    Parameters
    ----------
    grid : RectilinearGrid
        Rectilinear grid defining the shape, indexing convention, and total
        number of grid cells.
    n : int
        Number of cells along the selected axis. This is usually one of
        ``grid.nx``, ``grid.ny``, or ``grid.nz`` depending on ``axis``.
    axis : int
        Axis along which the forward-neighbour permutation is built:

        - ``0`` for the x-axis,
        - ``1`` for the y-axis,
        - ``2`` for the z-axis.

    sub_selection : NDArrayInt
        Flattened grid-cell indices to keep. Owner and neighbour cells must both
        belong to this selection to be connected in the permutation matrix.

    Returns
    -------
    csc_array
        Sparse permutation matrix with shape
        ``(grid.n_grid_cells, grid.n_grid_cells)`` in CSC format.

    Notes
    -----
    If ``n < 2``, no forward neighbour exists along the selected axis and an
    empty sparse matrix is returned.

    The grid uses the flattened numbering convention:

    ``node_number = ix + iy * grid.nx + iz * grid.ny * grid.nx``

    This matrix is not a full permutation matrix in the strict algebraic sense
    when boundary cells or cells outside ``sub_selection`` are present, because
    some rows and columns may contain only zeros.
    """

    if axis not in {0, 1, 2}:
        raise ValueError("axis must be one of {0, 1, 2}.")

    shape = (grid.n_grid_cells, grid.n_grid_cells)

    # No forward neighbour exists if there is fewer than two cells along this axis.
    if n < 2:
        return coo_array(shape, dtype=np.float64).tocsc()

    # Spans are expressed against a (nx, ny, nz)-shaped array, so the array
    # axis matches the spatial axis directly.
    _slices1: List[slice] = [slice(None), slice(None), slice(None)]
    _slices2: List[slice] = [slice(None), slice(None), slice(None)]
    _slices1[axis] = slice(0, n - 1)
    _slices2[axis] = slice(1, n)
    slices1: Tuple[slice, slice, slice] = tuple(_slices1)  # ty:ignore[invalid-assignment]
    slices2: Tuple[slice, slice, slice] = tuple(_slices2)  # ty:ignore[invalid-assignment]

    # Forward-neighbour pairs along the selected axis.
    idc_owner, idc_neigh = get_owner_neigh_indices(
        grid,
        slices1,
        slices2,
        owner_indices_to_keep=sub_selection,
        neigh_indices_to_keep=sub_selection,
    )

    return coo_array(
        (np.ones(idc_owner.size), (idc_neigh, idc_owner)),
        shape=shape,
        dtype=np.float64,
    ).tocsc()


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
    # See the corresponding note in make_rlg_spatial_gradient_matrices: None
    # is left as-is so get_owner_neigh_indices can skip the no-op isin
    # filtering pass when no real sub-selection is requested.
    return (
        get_rlg_perm_mat(grid, grid.nx, 0, sub_selection),
        get_rlg_perm_mat(grid, grid.ny, 1, sub_selection),
        get_rlg_perm_mat(grid, grid.nz, 2, sub_selection),
    )


def resample_grid(
    original_grid: RectilinearGrid,
    factor_x: float,
    factor_y: float,
    factor_z: float,
) -> RectilinearGrid:
    """
    Resample a rectilinear grid by changing the number of cells along each axis.

    The physical extent, centre coordinates, and rotation angles of the original
    grid are preserved. Only the number of cells and corresponding cell sizes are
    updated. The new number of cells along each axis is computed as the ceiling
    of the original number of cells multiplied by the corresponding resampling
    factor.

    Parameters
    ----------
    original_grid : RectilinearGrid
        Grid to resample.
    factor_x : float
        Resampling factor along the x-axis. Values greater than 1 refine the
        grid along x, while values between 0 and 1 coarsen it.
    factor_y : float
        Resampling factor along the y-axis. Values greater than 1 refine the
        grid along y, while values between 0 and 1 coarsen it.
    factor_z : float
        Resampling factor along the z-axis. Values greater than 1 refine the
        grid along z, while values between 0 and 1 coarsen it.

    Returns
    -------
    RectilinearGrid
        New rectilinear grid with updated shape and cell dimensions. The grid
        centre, total physical dimensions, and rotation angles are preserved.

    Notes
    -----
    The new number of cells is computed as:

    ``new_nx = max(ceil(original_grid.nx * factor_x), 1)``

    ``new_ny = max(ceil(original_grid.ny * factor_y), 1)``

    ``new_nz = max(ceil(original_grid.nz * factor_z), 1)``

    The new cell dimensions are then adjusted so that the physical extent along
    each axis remains unchanged:

    ``new_dx = original_grid.nx * original_grid.dx / new_nx``

    ``new_dy = original_grid.ny * original_grid.dy / new_ny``

    ``new_dz = original_grid.nz * original_grid.dz / new_nz``

    Therefore, this function changes the discretisation of the grid, but not its
    physical size or position.

    A minimum of one cell is enforced along each axis to avoid returning an empty
    grid.
    """
    # Use max(..., 1) to avoid ending up with zero cells along any axis.
    _nx = int(max(np.ceil(original_grid.nx * factor_x).item(), 1))
    _ny = int(max(np.ceil(original_grid.ny * factor_y).item(), 1))
    _nz = int(max(np.ceil(original_grid.nz * factor_z).item(), 1))

    return RectilinearGrid(
        cx=original_grid.cx,
        cy=original_grid.cy,
        cz=original_grid.cz,
        dx=original_grid.nx * original_grid.dx / _nx,
        dy=original_grid.ny * original_grid.dy / _ny,
        dz=original_grid.nz * original_grid.dz / _nz,
        nx=_nx,
        ny=_ny,
        nz=_nz,
        theta=original_grid.theta,
        phi=original_grid.phi,
        psi=original_grid.psi,
    )


def duplicative_upsample(array: NDArrayFloat, factor: int) -> NDArrayFloat:
    """
    Upsample a 2D array by duplicating each cell value.

    Each input cell is expanded into a ``factor x factor`` block containing the
    same value as the original cell. This operation preserves the original cell
    values locally, but it does not preserve the total sum of the array.

    Parameters
    ----------
    array : NDArrayFloat
        Two-dimensional array to upsample.
    factor : int
        Integer upsampling factor applied along both array axes. Each input cell
        becomes a square block of shape ``(factor, factor)`` in the output.
        Must be greater than or equal to 1.

    Returns
    -------
    NDArrayFloat
        Upsampled array with shape
        ``(array.shape[0] * factor, array.shape[1] * factor)``.

    Raises
    ------
    ValueError
        If ``factor`` is smaller than 1.

    Notes
    -----
    This function is suitable for intensive quantities, such as temperature or
    concentration, where duplicating the value over refined cells is meaningful.

    For extensive quantities, such as volume, mass, area, or energy, use
    :func:`conservative_upsample` instead to preserve the total sum.
    """
    if factor < 1:
        raise ValueError("factor must be a positive integer.")

    return np.repeat(np.repeat(array, factor, axis=0), factor, axis=1)


def conservative_upsample(array: NDArrayFloat, factor: int) -> NDArrayFloat:
    """
    Upsample a 2D extensive array while preserving its total sum.

    Each input cell is expanded into a ``factor x factor`` block. The original
    cell value is evenly distributed over the refined cells by dividing each
    duplicated value by ``factor ** 2``. As a result, the sum of the output array
    is equal to the sum of the input array, up to floating-point precision.

    Parameters
    ----------
    array : NDArrayFloat
        Two-dimensional array to upsample. Values are interpreted as extensive
        quantities attached to grid cells.
    factor : int
        Integer upsampling factor applied along both array axes. Each input cell
        becomes a square block of shape ``(factor, factor)`` in the output.
        Must be greater than or equal to 1.

    Returns
    -------
    NDArrayFloat
        Conservatively upsampled array with shape
        ``(array.shape[0] * factor, array.shape[1] * factor)``.

    Raises
    ------
    ValueError
        If ``factor`` is smaller than 1.

    Notes
    -----
    This function is appropriate for extensive quantities whose total value must
    be conserved during refinement, such as volume, mass, surface area, or
    energy.

    For intensive quantities, where the original value should simply be repeated
    over refined cells, use :func:`duplicative_upsample`.
    """
    if factor < 1:
        raise ValueError("factor must be a positive integer.")

    return duplicative_upsample(array, factor) / float(factor**2)
