"""Benchmark scenarios covering every dispatch path of the transfer matrix."""

import numpy as np
import shapely

import quickpaver as qp
from quickpaver._tm_test_utils import make_grid_trimesh, rotate_trimesh


def _hexagons(cx, cy, radius, phase=0.0):
    ang = np.linspace(0, 2 * np.pi, 7)[:-1] + phase
    x = cx.ravel()[:, None] + radius * np.cos(ang)[None, :]
    y = cy.ravel()[:, None] + radius * np.sin(ang)[None, :]
    return shapely.MultiPolygon(list(shapely.polygons(np.stack([x, y], -1))))


def scenarios(scale: float = 1.0) -> dict:
    """``name -> (source_grid, target_grid)``; ``scale`` shrinks every size."""

    def s(n: int) -> int:
        return max(2, int(n * scale))

    rg = qp.RectilinearGrid
    src = rg(cx=0, cy=0, dx=1.0, dy=1.0, nx=s(1000), ny=s(1000))
    tgt_rot = rg(cx=3.3, cy=-1.7, dx=1.37, dy=1.21, nx=s(700), ny=s(700), theta=27.0)
    tgt_sep = rg(cx=3.3, cy=-1.7, dx=1.37, dy=1.21, nx=s(700), ny=s(700), theta=90.0)

    n_tm = s(300)
    tm = rotate_trimesh(
        make_grid_trimesh(-n_tm * 0.75, -n_tm * 0.75, 1.5, 1.5, n_tm, n_tm), 17.0
    )
    tm2 = make_grid_trimesh(-n_tm * 0.6, -n_tm * 0.6, 1.2, 1.2, n_tm, n_tm)

    n_poly = s(250)
    g = np.arange(n_poly) * 2.0 - n_poly
    polys = _hexagons(*np.meshgrid(g, g), 1.15, phase=0.2)

    fine = rg(cx=0, cy=0, dx=0.05, dy=0.05, nx=s(2000), ny=s(2000), theta=5.0)
    coarse = rg(cx=1.1, cy=-0.7, dx=2.3, dy=2.1, nx=s(40), ny=s(40), theta=33.0)
    n_big = s(14)
    gb = np.arange(n_big) * 7.0 - n_big * 3.5
    big = _hexagons(*np.meshgrid(gb, gb), 4.0)

    small_src = rg(cx=0, cy=0, dx=1.0, dy=1.0, nx=s(300), ny=s(300))
    return {
        "fine_vs_coarse_rotated": (fine, coarse),
        "fine_vs_big_hexagons": (fine, big),
        "rect_rect_rotated": (src, tgt_rot),
        "rect_rect_separable": (src, tgt_sep),
        "rect_trimesh": (src, tm),
        "trimesh_trimesh": (tm, tm2),
        "rect_polygons": (src, polys),
        "trimesh_polygons": (tm, polys),
        "polygons_polygons": (small_src.to_shapely(), polys),
    }
