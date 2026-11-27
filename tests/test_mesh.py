# SPDX-License-Identifier: BSD-3-Clause
"""Tests of the VTK mesh helpers (surface meshes with shared vertices and
layered extrusion)."""

from __future__ import annotations

import numpy as np
import pytest
import shapely

import quickpaver as qp
from quickpaver import _quickpaver as rs

pv = pytest.importorskip("pyvista")
vtk = pytest.importorskip("vtk")


def _voronoi(n: int = 400, size: float = 100.0, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    box = shapely.box(0, 0, size, size)
    cells = shapely.get_parts(
        shapely.voronoi_polygons(
            shapely.MultiPoint(rng.uniform(0, size, (n, 2))), extend_to=box
        )
    )
    cells = shapely.intersection(cells, box)
    return cells[shapely.get_type_id(cells) == shapely.GeometryType.POLYGON]


def _vtk_volumes(mesh) -> np.ndarray:
    return np.asarray(
        mesh.compute_cell_sizes(length=False, area=False, volume=True)["Volume"]
    )


def _validity(mesh) -> set:
    val = vtk.vtkCellValidator()
    val.SetInputData(mesh)
    val.Update()
    return set(np.asarray(pv.wrap(val.GetOutput()).cell_data["ValidityState"]))


def _ngon(n: int, cx: float, cw: bool) -> np.ndarray:
    t = 2 * np.pi * np.arange(n) / n
    ring = np.c_[cx + np.cos(t), np.sin(t)]
    return ring[::-1] if cw else ring


# --------------------------------------------------------------------------- #
# surface meshes
# --------------------------------------------------------------------------- #


def test_voronoi_surface_shares_vertices() -> None:
    cells = _voronoi()
    mesh = qp.shapely_polygons_to_vtu(cells, z=2.5)
    rings = shapely.get_exterior_ring(cells)
    distinct = np.unique(shapely.get_coordinates(rings).round(9), axis=0)
    assert mesh.n_cells == len(cells)
    assert mesh.n_points == len(distinct)
    assert np.all(np.asarray(mesh.points)[:, 2] == 2.5)
    assert np.asarray(mesh.points).dtype == np.float64
    # every cell has the polygon's area and is counter-clockwise (normal +z)
    areas = mesh.compute_cell_sizes(length=False, area=True, volume=False)["Area"]
    np.testing.assert_allclose(areas, shapely.area(cells), rtol=1e-12)
    P = np.asarray(mesh.points)
    for k, poly in enumerate(cells[:50]):
        cell = mesh.get_cell(k)
        ring = shapely.Polygon(P[cell.point_ids, :2])
        assert ring.exterior.is_ccw
        assert shapely.equals_exact(ring.normalize(), poly.normalize(), tolerance=1e-9)


def test_surface_input_types_agree() -> None:
    verts, _, _ = qp.gen_polygonal_tiling_arrays(
        shapely.box(0, 0, 30, 20), qp.PolygonType.HEXAGON, 2.0
    )
    polys = shapely.polygons(verts)
    ref = qp.shapely_polygons_to_vtu(verts)
    for x in (polys, list(polys), shapely.MultiPolygon(list(polys))):
        m = qp.shapely_polygons_to_vtu(x)
        np.testing.assert_array_equal(m.points, ref.points)
        np.testing.assert_array_equal(m.cells, ref.cells)


def test_merge_tolerance() -> None:
    a = shapely.box(0, 0, 1, 1)
    b = shapely.box(1.001, 0, 2, 1)  # 1e-3 gap
    assert qp.shapely_polygons_to_vtu([a, b]).n_points == 8
    m = qp.shapely_polygons_to_vtu([a, b], merge_tol=0.01)
    assert m.n_points == 6
    # merged points are the mean of their cluster
    assert np.isclose(np.asarray(m.points)[:, 0], 1.0005).sum() == 2
    with pytest.raises(ValueError, match="merge_tol"):
        qp.shapely_polygons_to_vtu([a], merge_tol=-1.0)


def test_merge_is_single_linkage() -> None:
    """Same clusters as scipy's single linkage cut at the tolerance."""
    from scipy.cluster.hierarchy import fcluster, linkage

    rng = np.random.default_rng(1)
    xy = rng.uniform(0, 10, (600, 2))
    tri = shapely.polygons(xy.reshape(-1, 3, 2))
    for tol in (0.05, 0.2, 0.4):
        ref = fcluster(linkage(xy, method="single"), tol, criterion="distance")
        try:
            pts, cells = rs.polygon_mesh(
                np.ascontiguousarray(xy).ravel(),
                np.arange(0, 601, 3, dtype=np.intp),
                False,
                tol,
            )
        except ValueError:
            continue  # some triangle collapsed: covered elsewhere
        assert len(pts) == len(np.unique(ref))
    assert len(tri) == 200


def test_collapsed_vertices_are_removed() -> None:
    # two vertices of the square closer than the tolerance
    sq = shapely.Polygon([(0, 0), (1, 0), (1, 1), (0.001, 1), (0, 1)])
    m = qp.shapely_polygons_to_vtu([sq], merge_tol=0.01)
    assert m.n_points == 4 and m.cells[0] == 4
    with pytest.raises(ValueError, match="fewer than 3"):
        qp.shapely_polygons_to_vtu([shapely.box(0, 0, 1, 1)], merge_tol=5.0)
    with pytest.raises(ValueError, match="fewer than 3"):
        qp.shapely_polygons_to_vtu([shapely.Polygon()])


def test_far_from_origin() -> None:
    cells = shapely.transform(_voronoi(200), lambda c: c + [6.5e5, 4.8e6])
    mesh = qp.shapely_polygons_to_vtu(cells)
    distinct = np.unique(
        shapely.get_coordinates(shapely.get_exterior_ring(cells)).round(6), axis=0
    )
    assert mesh.n_points == len(distinct)


def test_create_unstructured_mesh() -> None:
    m = qp.create_unstructured_mesh(
        [[0, 0], [1, 0], [1, 1], [0, 1], [2, 0]], [[0, 1, 2, 3], [1, 4, 2]]
    )
    assert isinstance(m, vtk.vtkUnstructuredGrid)
    assert list(m.cells) == [4, 0, 1, 2, 3, 3, 1, 4, 2]
    assert set(m.celltypes) == {7}
    assert np.all(np.asarray(m.points)[:, 2] == 0.0)
    arr = qp.create_unstructured_mesh(
        np.c_[np.random.rand(4, 2), np.ones(4)], np.array([[0, 1, 2], [1, 2, 3]])
    )
    assert list(arr.cells) == [3, 0, 1, 2, 3, 1, 2, 3]
    assert np.all(np.asarray(arr.points)[:, 2] == 1.0)
    with pytest.raises(ValueError, match="at least 3"):
        qp.create_unstructured_mesh([[0, 0], [1, 0]], [[0, 1]])
    with pytest.raises(ValueError, match="does not exist"):
        qp.create_unstructured_mesh([[0, 0], [1, 0], [1, 1]], [[0, 1, 3]])
    with pytest.raises(ValueError, match="shape"):
        qp.create_unstructured_mesh(np.zeros((3, 4)), [[0, 1, 2]])
    assert qp.create_unstructured_mesh(np.zeros((0, 3)), []).n_cells == 0


# --------------------------------------------------------------------------- #
# extrusion
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("cw", [False, True])
@pytest.mark.parametrize("thickness", [[1.0, 0.5], [-1.0, -0.5]])
def test_extruded_cells_are_valid(cw: bool, thickness) -> None:
    """Whatever the 2D orientation and the extrusion direction, every cell
    type (wedge ... polyhedron) is valid for VTK with a positive volume."""
    polys = [_ngon(n, 3 * n, cw) for n in range(3, 11)]
    verts = np.concatenate(polys)
    sizes = [len(p) for p in polys]
    conn = np.split(np.arange(len(verts)), np.cumsum(sizes)[:-1])
    surf = qp.create_unstructured_mesh(verts, conn)
    vol = qp.pv_extrude(surf, thickness)
    assert set(vol.celltypes) == {12, 13, 15, 16, 42}
    assert _validity(vol) == {0}
    v = _vtk_volumes(vol)
    assert np.all(v > 0)
    np.testing.assert_allclose(v, vol["Volume"], rtol=1e-12)


def test_extrusion_layout_and_data() -> None:
    cells = _voronoi(150)
    surf = qp.shapely_polygons_to_vtu(cells, z=10.0)
    centres = np.c_[
        shapely.get_coordinates(shapely.centroid(cells)), np.full(len(cells), 10.0)
    ]
    surf.cell_data["Cell centers"] = centres
    th = [1.0, 2.0, 0.5]
    vol = qp.pv_extrude(surf, th)
    n, nl = surf.n_cells, len(th)
    assert vol.n_cells == n * nl and vol.n_points == surf.n_points * (nl + 1)
    np.testing.assert_array_equal(vol["Layer"], np.tile(np.arange(nl), n))
    assert vol["Layer"].dtype == np.uint16
    np.testing.assert_allclose(
        vol["Volume"], np.outer(shapely.area(cells), th).ravel(), rtol=1e-12
    )
    np.testing.assert_allclose(_vtk_volumes(vol), vol["Volume"], rtol=1e-9)
    z = np.asarray(vol.points)[:, 2].reshape(nl + 1, -1)
    np.testing.assert_allclose(z[:, 0], [10.0, 11.0, 13.0, 13.5])
    cc = vol["Cell centers"].reshape(n, nl, 3)
    np.testing.assert_allclose(cc[:, :, 2], np.tile([10.5, 12.0, 13.25], (n, 1)))
    np.testing.assert_array_equal(cc[:, 1, :2], centres[:, :2])
    # every 3D cell sits inside its layer
    b = np.array([vol.get_cell(k).bounds for k in range(0, vol.n_cells, 7)])
    lay = np.asarray(vol["Layer"])[::7]
    np.testing.assert_allclose(b[:, 4], z[lay, 0])
    np.testing.assert_allclose(b[:, 5], z[lay + 1, 0])


def test_extrude_follows_topography_and_casts_datasets() -> None:
    plane = pv.Plane(i_resolution=6, j_resolution=4)  # PolyData of quads
    plane.points[:, 2] = 0.1 * plane.points[:, 0]  # tilted surface
    vol = qp.pv_extrude(plane, [0.2, 0.3])
    assert set(vol.celltypes) == {12} and _validity(vol) == {0}
    np.testing.assert_allclose(_vtk_volumes(vol), vol["Volume"], rtol=1e-9)
    # pixels (ImageData surface) are reordered into rings
    img = pv.ImageData(dimensions=(5, 4, 1)).cast_to_unstructured_grid()
    assert set(img.celltypes) == {8}
    vol = qp.pv_extrude(img, 1.0)
    assert set(vol.celltypes) == {12} and _validity(vol) == {0}
    np.testing.assert_allclose(vol["Volume"], 1.0)


def test_extrude_errors() -> None:
    surf = qp.shapely_polygons_to_vtu([shapely.box(0, 0, 1, 1)])
    for th, match in [
        ([], "non-empty"),
        ([[1.0]], "non-empty"),
        ([1.0, 0.0], "non-zero"),
        ([1.0, np.nan], "non-zero"),
        ([1.0, -1.0], "same sign"),
    ]:
        with pytest.raises(ValueError, match=match):
            qp.pv_extrude(surf, th)
    line = pv.Line().cast_to_unstructured_grid()
    with pytest.raises(ValueError, match="2D cells"):
        qp.pv_extrude(line, [1.0])
    pts = np.zeros((3, 3))
    with pytest.raises(ValueError, match="missing point"):
        rs.extrude_cells(np.array([3, 0, 1, 5]), np.array([7], np.uint8), pts, 1, False)
    with pytest.raises(ValueError, match="at least 3"):
        rs.extrude_cells(np.array([2, 0, 1]), np.array([7], np.uint8), pts, 1, False)
    with pytest.raises(ValueError, match="invalid cell array"):
        rs.extrude_cells(np.array([5, 0, 1]), np.array([7], np.uint8), pts, 1, False)
    with pytest.raises(ValueError, match="longer"):
        rs.extrude_cells(
            np.array([3, 0, 1, 2, 9]), np.array([7], np.uint8), pts, 1, False
        )
    with pytest.raises(ValueError, match="shorter"):
        rs.extrude_cells(
            np.array([3, 0, 1, 2]), np.array([7, 7], np.uint8), pts, 1, False
        )
    with pytest.raises(ValueError, match="shape"):
        rs.extrude_cells(
            np.array([3, 0, 1, 2]), np.array([7], np.uint8), np.zeros((3, 2)), 1, False
        )


def test_get_ncell4poly() -> None:
    assert [qp.get_ncell4poly(n) for n in (3, 4, 5, 6, 7, 10)] == [
        7,
        9,
        11,
        13,
        53,
        74,
    ]
    with pytest.raises(ValueError):
        qp.get_ncell4poly(2)
    # consistent with the cell arrays actually written
    pts = np.c_[_ngon(10, 0.0, False), np.zeros(10)]
    for n in (3, 6, 10):
        cells = np.r_[n, np.arange(n)].astype(np.int64)
        out, _, _ = rs.extrude_cells(cells, np.array([7], np.uint8), pts, 2, False)
        assert len(out) == 2 * qp.get_ncell4poly(n)
