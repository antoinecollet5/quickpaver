"""
Full-coverage pytest suite for :class:`trimesh_flat.TriMesh`.

Covers:
- n_verts / n_tri / area_m2 properties
- to_shapely() geometry conversion
- transform() with default origin (centroid), explicit origin,
  pure translation, pure rotation, and identity
- immutability (frozen dataclass) guarantees
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import shapely
from quickpaver import TriMesh

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def unit_square_mesh() -> TriMesh:
    """
    A unit square [0,1]x[0,1] split into two triangles:

    Triangle 0: (0,1,2)  -> right triangle, area 0.5
    Triangle 1: (0,2,3)  -> right triangle, area 0.5
    """
    verts_xy = np.array(
        [
            [0.0, 0.0],  # 0
            [1.0, 0.0],  # 1
            [1.0, 1.0],  # 2
            [0.0, 1.0],  # 3
        ]
    )
    tri_verts = np.array([[0, 1, 2], [0, 2, 3]], dtype=int)

    edge_lengths_m = np.array(
        [
            [1.0, 1.0, math.sqrt(2)],
            [math.sqrt(2), 1.0, 1.0],
        ]
    )
    tri_area_m2 = np.array([0.5, 0.5])

    return TriMesh(
        verts_xy=verts_xy,
        tri_verts=tri_verts,
        edge_lengths_m=edge_lengths_m,
        tri_area_m2=tri_area_m2,
    )


@pytest.fixture
def single_triangle_mesh() -> TriMesh:
    """A single 3-4-5 right triangle at the origin."""
    verts_xy = np.array(
        [
            [0.0, 0.0],
            [3.0, 0.0],
            [0.0, 4.0],
        ]
    )
    tri_verts = np.array([[0, 1, 2]], dtype=int)
    edge_lengths_m = np.array([[3.0, 5.0, 4.0]])
    tri_area_m2 = np.array([6.0])

    return TriMesh(
        verts_xy=verts_xy,
        tri_verts=tri_verts,
        edge_lengths_m=edge_lengths_m,
        tri_area_m2=tri_area_m2,
    )


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------


class TestProperties:
    def test_n_verts(self, unit_square_mesh):
        assert unit_square_mesh.n_verts == 4

    def test_n_tri(self, unit_square_mesh):
        assert unit_square_mesh.n_tri == 2

    def test_area_m2(self, unit_square_mesh):
        assert unit_square_mesh.area_m2 == pytest.approx(1.0)

    def test_n_verts_single_triangle(self, single_triangle_mesh):
        assert single_triangle_mesh.n_verts == 3

    def test_n_tri_single_triangle(self, single_triangle_mesh):
        assert single_triangle_mesh.n_tri == 1

    def test_area_m2_single_triangle(self, single_triangle_mesh):
        assert single_triangle_mesh.area_m2 == pytest.approx(6.0)

    def test_area_m2_returns_python_or_numpy_scalar(self, unit_square_mesh):
        # sanity: .sum() on ndarray behaves as expected and is comparable
        assert float(unit_square_mesh.area_m2) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# to_shapely
# ---------------------------------------------------------------------------


class TestToShapely:
    def test_returns_multipolygon(self, unit_square_mesh):
        mp = unit_square_mesh.to_shapely()
        assert isinstance(mp, shapely.MultiPolygon)

    def test_polygon_count_matches_n_tri(self, unit_square_mesh):
        mp = unit_square_mesh.to_shapely()
        assert len(mp.geoms) == unit_square_mesh.n_tri

    def test_polygon_order_matches_tri_verts(self, unit_square_mesh):
        mp = unit_square_mesh.to_shapely()
        # triangle 0 = verts 0,1,2 = (0,0),(1,0),(1,1)
        expected0 = shapely.Polygon([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        # triangle 1 = verts 0,2,3 = (0,0),(1,1),(0,1)
        expected1 = shapely.Polygon([[0.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        assert mp.geoms[0].equals(expected0)
        assert mp.geoms[1].equals(expected1)

    def test_total_shapely_area_matches_area_m2(self, unit_square_mesh):
        mp = unit_square_mesh.to_shapely()
        assert mp.area == pytest.approx(unit_square_mesh.area_m2)

    def test_single_triangle_shapely_area(self, single_triangle_mesh):
        mp = single_triangle_mesh.to_shapely()
        assert len(mp.geoms) == 1
        assert mp.geoms[0].area == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# transform
# ---------------------------------------------------------------------------


class TestTransform:
    def test_identity_transform_preserves_coords(self, unit_square_mesh):
        result = unit_square_mesh.transform()
        np.testing.assert_allclose(result.verts_xy, unit_square_mesh.verts_xy)

    def test_returns_new_instance(self, unit_square_mesh):
        result = unit_square_mesh.transform(x=1.0)
        assert result is not unit_square_mesh
        assert isinstance(result, TriMesh)

    def test_original_mesh_unmodified(self, unit_square_mesh):
        original_xy = unit_square_mesh.verts_xy.copy()
        unit_square_mesh.transform(rot_deg=45, x=2.0, y=-3.0)
        np.testing.assert_allclose(unit_square_mesh.verts_xy, original_xy)

    def test_pure_translation(self, unit_square_mesh):
        result = unit_square_mesh.transform(x=2.0, y=3.0)
        expected = unit_square_mesh.verts_xy + np.array([2.0, 3.0])
        np.testing.assert_allclose(result.verts_xy, expected)

    def test_non_geometry_fields_are_carried_over(self, unit_square_mesh):
        result = unit_square_mesh.transform(x=2.0, y=3.0)
        np.testing.assert_allclose(result.tri_verts, unit_square_mesh.tri_verts)
        np.testing.assert_allclose(
            result.edge_lengths_m, unit_square_mesh.edge_lengths_m
        )
        np.testing.assert_allclose(result.tri_area_m2, unit_square_mesh.tri_area_m2)

    def test_rotation_about_explicit_origin(self, single_triangle_mesh):
        # Rotate 90 deg CCW about the origin (0,0), which is vertex 0.
        result = single_triangle_mesh.transform(rot_deg=90.0, origin=(0.0, 0.0))
        # (3,0) -> (0,3); (0,4) -> (-4,0); (0,0) -> (0,0)
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 3.0],
                [-4.0, 0.0],
            ]
        )
        np.testing.assert_allclose(result.verts_xy, expected, atol=1e-10)

    def test_rotation_about_default_centroid(self, unit_square_mesh):
        # Rotating 180 degrees about the centroid should map the square
        # back onto itself (as a set of points), since it's symmetric.
        centroid = unit_square_mesh.verts_xy.mean(axis=0)
        result = unit_square_mesh.transform(rot_deg=180.0)
        expected = 2 * centroid - unit_square_mesh.verts_xy
        np.testing.assert_allclose(result.verts_xy, expected, atol=1e-10)

    def test_rotation_preserves_area(self, unit_square_mesh):
        result = unit_square_mesh.transform(rot_deg=37.0, x=5.0, y=-2.0)
        # area_m2 field is carried over unchanged, and since rotation +
        # translation are rigid, the true geometric area is unaffected too.
        mp_before = unit_square_mesh.to_shapely()
        mp_after = result.to_shapely()
        assert mp_after.area == pytest.approx(mp_before.area)

    def test_rotation_and_translation_combined(self, single_triangle_mesh):
        result = single_triangle_mesh.transform(
            rot_deg=90.0, x=10.0, y=-5.0, origin=(0.0, 0.0)
        )
        expected = np.array(
            [
                [0.0, 0.0],
                [0.0, 3.0],
                [-4.0, 0.0],
            ]
        ) + np.array([10.0, -5.0])
        np.testing.assert_allclose(result.verts_xy, expected, atol=1e-10)

    def test_default_origin_matches_explicit_centroid(self, unit_square_mesh):
        centroid = tuple(unit_square_mesh.verts_xy.mean(axis=0))
        result_default = unit_square_mesh.transform(rot_deg=25.0)
        result_explicit = unit_square_mesh.transform(rot_deg=25.0, origin=centroid)
        np.testing.assert_allclose(result_default.verts_xy, result_explicit.verts_xy)


# ---------------------------------------------------------------------------
# Immutability
# ---------------------------------------------------------------------------


class TestImmutability:
    def test_frozen_dataclass_raises_on_attribute_set(self, unit_square_mesh):
        with pytest.raises(AttributeError):
            unit_square_mesh.verts_xy = np.zeros((4, 2))

    def test_slots_prevents_new_attributes(self, unit_square_mesh):
        with pytest.raises((AttributeError, TypeError)):
            unit_square_mesh.extra_attr = 123
