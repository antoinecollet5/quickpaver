"""Type stubs for the compiled Rust extension ``quickpaver._quickpaver``."""

from typing import Any, Optional

import numpy as np
import numpy.typing as npt

# Inputs are converted by the extension (contiguous float64 / intp / bool
# arrays are expected); outputs have precise dtypes.
_F1 = npt.NDArray[Any]
_F2 = npt.NDArray[np.float64]
_I1 = npt.NDArray[Any]
_B1 = npt.NDArray[Any]
_OI1 = npt.NDArray[np.intp]
_Frame = tuple[float, float, float, float]
_Out = tuple[_OI1, _OI1, _F2, npt.NDArray[np.uint8], _F2, _OI1]

def rect_items_transfer(
    x: _F1,
    y: _F1,
    ring_off: _I1,
    poly_ring: _I1,
    convex: Optional[_B1],
    to_local: _Frame,
    x_edges: _F1,
    y_edges: _F1,
    cell_mask: Optional[_B1] = None,
    item_mask: Optional[_B1] = None,
    with_geom: bool = False,
    to_world: Optional[_Frame] = None,
    interior: bool = True,
) -> _Out: ...
def nonseparable_transfer(
    snx: int,
    sny: int,
    sdx: float,
    sdy: float,
    tnx: int,
    tny: int,
    tdx: float,
    tdy: float,
    origin_x: float,
    origin_y: float,
    cr: float,
    sr: float,
    template_x: list[float],
    template_y: list[float],
    half_bx: float,
    half_by: float,
    src_mask: Optional[_B1] = None,
    tgt_mask: Optional[_B1] = None,
    with_geom: bool = False,
    to_world: Optional[_Frame] = None,
    interior: bool = True,
) -> _Out: ...
def separable_transfer(
    is_x: _I1,
    it_x: _I1,
    ox: _F1,
    xlo: _F1,
    xhi: _F1,
    is_y: _I1,
    it_y: _I1,
    oy: _F1,
    ylo: _F1,
    yhi: _F1,
    sx_is_tgt_y: bool,
    snx: int,
    sny: int,
    tnx: int,
    tny: int,
    src_area: float,
    src_mask: Optional[_B1] = None,
    tgt_mask: Optional[_B1] = None,
    with_geom: bool = False,
    to_world: Optional[_Frame] = None,
) -> _Out: ...
def join_clip_transfer(
    ax: _F1,
    ay: _F1,
    a_ring_off: _I1,
    a_poly_ring: _I1,
    a_convex: Optional[_B1],
    bx: _F1,
    by: _F1,
    b_ring_off: _I1,
    b_poly_ring: _I1,
    b_convex: Optional[_B1],
    a_mask: Optional[_B1] = None,
    b_mask: Optional[_B1] = None,
    prefer_a_window: bool = True,
    with_geom: bool = False,
) -> _Out: ...
def polygon_convexity(
    x: _F1, y: _F1, ring_off: _I1, poly_ring: _I1
) -> npt.NDArray[np.bool_]: ...
def classify_boxes(
    xmin: _F1,
    ymin: _F1,
    xmax: _F1,
    ymax: _F1,
    x: _F1,
    y: _F1,
    ring_off: _I1,
    poly_ring: _I1,
    cell: float,
) -> npt.NDArray[np.int8]: ...
def coo_to_csc(
    rows: _I1, cols: _I1, data: _F1, n_rows: int, n_cols: int
) -> tuple[_F2, npt.NDArray[np.int32], npt.NDArray[np.int32]]: ...
