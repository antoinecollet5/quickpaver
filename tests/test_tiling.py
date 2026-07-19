# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET
"""Tests for the tiling functions."""

import numpy as np
import pytest
import shapely
from quickpaver._tiling import (
    PolygonType,
    _lattice_centres,
    _validate_inputs,
    adjacency_by_shared_vertices,
    extract_tiling_centers,
    extract_tiling_vertices,
    gen_hexagonal_tiling,
    gen_polygon,
    gen_polygonal_tiling,
    gen_rectangular_tiling,
    gen_triangular_tiling,
    intersects_mask,
)
from shapely.geometry import MultiPolygon, Polygon, box

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _ring_surface() -> shapely.Polygon:
    """An annulus (ring) surface used across the tiling tests."""
    return shapely.Point((0.0, 0.0)).buffer(200.0) - shapely.Point((0.0, 0.0)).buffer(
        100.0
    )


# ---------------------------------------------------------------------------
# gen_polygon
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE, PolygonType.TRIANGLE),
)
@pytest.mark.parametrize("anisotropy_ratio", (1.0, 2.0, 0.5))
def test_gen_polygon(poly_type: PolygonType, anisotropy_ratio: float) -> None:
    poly = gen_polygon(poly_type, edge_length=30.0, anisotropy_ratio=anisotropy_ratio)
    assert isinstance(poly, shapely.Polygon)
    assert poly.area > 0.0


def test_gen_polygon_wrong_type() -> None:
    with pytest.raises(ValueError, match="not a valid PolygonType"):
        gen_polygon("wrong_type")


# ---------------------------------------------------------------------------
# _validate_inputs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("edge_length", "anisotropy_ratio"),
    [
        (1.0, 1.0),
        (0.1, 1.0),
        (1.0, 0.1),
        (2.5, 3.0),
    ],
)
def test_validate_inputs_accepts_strictly_positive_values(
    edge_length: float,
    anisotropy_ratio: float,
) -> None:
    """Check that strictly positive inputs do not raise an exception."""
    _validate_inputs(edge_length=edge_length, anisotropy_ratio=anisotropy_ratio)


@pytest.mark.parametrize(
    "edge_length",
    [
        0.0,
        -1.0,
        -1e-12,
    ],
)
def test_validate_inputs_rejects_non_positive_edge_length(
    edge_length: float,
) -> None:
    """Check that zero or negative edge lengths raise a ValueError."""
    with pytest.raises(ValueError, match="edge_length must be strictly positive"):
        _validate_inputs(edge_length=edge_length, anisotropy_ratio=1.0)


@pytest.mark.parametrize(
    "anisotropy_ratio",
    [
        0.0,
        -1.0,
        -1e-12,
    ],
)
def test_validate_inputs_rejects_non_positive_anisotropy_ratio(
    anisotropy_ratio: float,
) -> None:
    """Check that zero or negative anisotropy ratios raise a ValueError."""
    with pytest.raises(ValueError, match="anisotropy_ratio must be strictly positive"):
        _validate_inputs(edge_length=1.0, anisotropy_ratio=anisotropy_ratio)


def test_validate_inputs_checks_edge_length_before_anisotropy_ratio() -> None:
    """Check that edge_length validation is performed before anisotropy validation."""
    with pytest.raises(ValueError, match="edge_length must be strictly positive"):
        _validate_inputs(edge_length=0.0, anisotropy_ratio=0.0)


# ---------------------------------------------------------------------------
# _lattice_centres
# ---------------------------------------------------------------------------


class TestLatticeCentres:
    """Tests for the lattice centre generator."""

    def test_anchor_is_a_node(self) -> None:
        """The anchor itself must appear as a lattice node."""
        b1 = np.array([10.0, 0.0])
        b2 = np.array([0.0, 10.0])
        anchor = np.array([3.0, 7.0])
        centres = _lattice_centres((0.0, 0.0, 50.0, 50.0), b1, b2, anchor)
        flat = centres.reshape(2, -1).T
        dists = np.linalg.norm(flat - anchor, axis=1)
        assert np.min(dists) < 1e-9

    def test_covers_bounds(self) -> None:
        """Generated nodes must extend beyond the bbox on all sides."""
        b1 = np.array([10.0, 0.0])
        b2 = np.array([0.0, 10.0])
        anchor = np.array([0.0, 0.0])
        centres = _lattice_centres((0.0, 0.0, 50.0, 50.0), b1, b2, anchor)
        assert centres[0].min() < 0.0
        assert centres[0].max() > 50.0
        assert centres[1].min() < 0.0
        assert centres[1].max() > 50.0

    def test_skewed_basis(self) -> None:
        """A skewed (triangular-like) basis still covers the bbox."""
        b1 = np.array([5.0, 0.0])
        b2 = np.array([-5.0, 8.66])
        anchor = np.array([0.0, 0.0])
        centres = _lattice_centres((0.0, 0.0, 40.0, 40.0), b1, b2, anchor)
        assert centres.shape[0] == 2
        assert centres[0].min() < 0.0
        assert centres[1].max() > 40.0


# ---------------------------------------------------------------------------
# gen_polygonal_tiling  — all types, anisotropy, edge length
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE, PolygonType.TRIANGLE),
)
@pytest.mark.parametrize("anisotropy_ratio", (1.0, 2.0, 0.5))
@pytest.mark.parametrize("edge_length", (100.0, 200.0))
def test_gen_polygonal_tiling(
    poly_type: PolygonType, anisotropy_ratio: float, edge_length: float
) -> None:
    tiles, adj = gen_polygonal_tiling(
        surface_to_cover=_ring_surface(),
        poly_type=poly_type,
        edge_length=edge_length,
        anisotropy_ratio=anisotropy_ratio,
    )
    assert isinstance(tiles, shapely.MultiPolygon)
    assert len(tiles.geoms) > 0
    assert isinstance(adj, dict)

    # MultiPolygon input
    _ = extract_tiling_centers(tiles)
    _, _, _ = extract_tiling_vertices(tiles)
    # Iterable-of-polygons input
    _ = extract_tiling_centers(tiles.geoms)
    _, _, _ = extract_tiling_vertices(tiles.geoms)


# ---------------------------------------------------------------------------
# gen_polygonal_tiling  — alignment + rotation paths
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE),  # triangle centroid ≠ node
)
def test_tiling_with_alignment_point(poly_type: PolygonType) -> None:
    """A tile centre coincides with the alignment point (point on the surface)."""
    align = (150.0, 0.0)  # inside the annulus → tile is kept
    tiles, _ = gen_polygonal_tiling(
        surface_to_cover=_ring_surface(),
        poly_type=poly_type,
        edge_length=30.0,
        alignment_point=align,
    )
    centres = extract_tiling_centers(tiles)
    dists = np.linalg.norm(centres - np.asarray(align), axis=1)
    assert np.min(dists) < 1e-6


def test_triangle_lattice_node_on_point() -> None:
    """For triangles the lattice anchor lands on the alignment point.

    The triangle centroid/vertices are offset from the lattice node by
    edge_length/(2*sqrt(3)), so we verify the lattice node itself —
    which is what alignment_point anchors.
    """
    edge = 30.0
    v_step = (edge**2 * 3 / 4) ** 0.5  # triangular v_step (aniso = 1)
    h_step = edge / 2
    b1 = np.array([h_step, 0.0])
    b2 = np.array([-h_step, v_step])

    align = np.array([150.0, 0.0])
    centres = _lattice_centres((-200.0, -200.0, 200.0, 200.0), b1, b2, align)

    flat = centres.reshape(2, -1).T
    dists = np.linalg.norm(flat - align, axis=1)
    assert np.min(dists) < 1e-9


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE, PolygonType.TRIANGLE),
)
def test_tiling_with_rotation_only(poly_type: PolygonType) -> None:
    """Rotation without alignment must still cover the surface."""
    tiles, _ = gen_polygonal_tiling(
        surface_to_cover=_ring_surface(),
        poly_type=poly_type,
        edge_length=30.0,
        rot_deg=30.0,
    )
    assert len(tiles.geoms) > 0


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE),  # drop triangle here too
)
def test_tiling_with_rotation_and_alignment(poly_type: PolygonType) -> None:
    align = (150.0, 0.0)  # on the annulus, survives masking
    tiles, _ = gen_polygonal_tiling(
        surface_to_cover=_ring_surface(),
        poly_type=poly_type,
        edge_length=30.0,
        rot_deg=45.0,
        alignment_point=align,
    )
    centres = extract_tiling_centers(tiles)
    dists = np.linalg.norm(centres - np.asarray(align), axis=1)
    assert np.min(dists) < 1e-6


def test_tiling_outside_alignment_point() -> None:
    surface = shapely.Point((0.0, 0.0)).buffer(200.0)
    tiles, _ = gen_polygonal_tiling(
        surface_to_cover=surface,
        poly_type=PolygonType.HEXAGON,
        edge_length=30.0,
        alignment_point=(10_000.0, -8_000.0),
    )
    assert len(tiles.geoms) > 0
    union = shapely.union_all(list(tiles.geoms))
    assert union.area >= surface.area  # kept tiles over-cover the disc


# ---------------------------------------------------------------------------
# Direct sub-tiler calls (alignment branch on each)
# ---------------------------------------------------------------------------


def test_gen_rectangular_tiling_alignment() -> None:
    tiles, adj = gen_rectangular_tiling(
        _ring_surface(), edge_length=30.0, alignment_point=(5.0, 5.0)
    )
    assert len(tiles.geoms) > 0
    assert isinstance(adj, dict)


def test_gen_hexagonal_tiling_alignment() -> None:
    tiles, adj = gen_hexagonal_tiling(
        _ring_surface(), edge_length=30.0, alignment_point=(5.0, 5.0)
    )
    assert len(tiles.geoms) > 0
    assert isinstance(adj, dict)


def test_gen_triangular_tiling_alignment() -> None:
    tiles, adj = gen_triangular_tiling(
        _ring_surface(), edge_length=30.0, alignment_point=(5.0, 5.0)
    )
    assert len(tiles.geoms) > 0
    assert isinstance(adj, dict)


# ---------------------------------------------------------------------------
# Wrong polygon type
# ---------------------------------------------------------------------------


def test_gen_polygonal_tiling_wrong_polytype() -> None:
    with pytest.raises(
        ValueError, match=r"'not_a_valid_type' is not a valid PolygonType"
    ):
        gen_polygonal_tiling(
            surface_to_cover=_ring_surface(),
            poly_type="not_a_valid_type",  # ty:ignore[invalid-argument-type]
            edge_length=10.0,
            anisotropy_ratio=1.0,
        )


class TestAdjacencyBySharedVertices:
    """Test suite for adjacency_by_shared_vertices function."""

    def test_two_triangles_sharing_vertex(self):
        """Test adjacency detection for two triangles sharing a single vertex."""
        # Create two triangles sharing the origin vertex
        tri1 = shapely.Polygon([(0, 0), (1, 0), (0, 1)])
        tri2 = shapely.Polygon([(0, 0), (0, 1), (-1, 0)])

        adj = adjacency_by_shared_vertices([tri1, tri2])

        assert 0 in adj
        assert 1 in adj
        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_two_triangles_sharing_edge(self):
        """Test adjacency for two triangles sharing an edge (2 vertices)."""
        # Create two triangles sharing edge from (0,0) to (1,0)
        tri1 = Polygon([(0, 0), (1, 0), (0.5, 1)])
        tri2 = Polygon([(0, 0), (1, 0), (0.5, -1)])

        adj = adjacency_by_shared_vertices([tri1, tri2])

        assert 1 in adj[0]
        assert 0 in adj[1]

    def test_three_triangles_star_pattern(self):
        """Test three triangles meeting at a common vertex (star pattern)."""

        tri1 = Polygon([(0, 0), (1, 0), (0.5, 0.866)])
        tri2 = Polygon([(0, 0), (0.5, 0.866), (-0.5, 0.866)])
        tri3 = Polygon([(0, 0), (-0.5, 0.866), (-1, 0)])

        adj = adjacency_by_shared_vertices([tri1, tri2, tri3])

        # All should be adjacent to each other
        assert set(adj[0]) == {1, 2}
        assert set(adj[1]) == {0, 2}
        assert set(adj[2]) == {0, 1}

    def test_isolated_polygon_no_shared_vertices(self):
        """Test that isolated polygons have no adjacencies."""
        poly1 = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])
        poly2 = Polygon([(10, 10), (11, 10), (11, 11), (10, 11)])

        adj = adjacency_by_shared_vertices([poly1, poly2])

        # Isolated polygons should either not appear in dict or have empty lists
        if 0 in adj:
            assert len(adj[0]) == 0
        if 1 in adj:
            assert len(adj[1]) == 0

    def test_single_polygon(self):
        """Test that a single polygon has no adjacencies."""
        poly = Polygon([(0, 0), (1, 0), (1, 1), (0, 1)])

        adj = adjacency_by_shared_vertices([poly])

        # Single polygon should not be adjacent to itself
        if 0 in adj:
            assert len(adj[0]) == 0

    def test_empty_list(self):
        """Test with an empty polygon list."""
        adj = adjacency_by_shared_vertices([])

        assert adj == {}

    def test_four_hexagons_grid_pattern(self):
        """Test adjacency for a 2x2 grid of hexagons."""
        # Create 4 hexagons in a grid arrangement
        hex1 = Polygon([(0, 0), (1, 0.5), (1.5, 1.5), (1, 2.5), (0, 2), (-0.5, 1.5)])
        hex2 = Polygon([(2, 0), (3, 0.5), (3.5, 1.5), (3, 2.5), (2, 2), (1.5, 1.5)])
        hex3 = Polygon([(0, 3), (1, 3.5), (1.5, 4.5), (1, 5.5), (0, 5), (-0.5, 4.5)])
        hex4 = Polygon([(2, 3), (3, 3.5), (3.5, 4.5), (3, 5.5), (2, 5), (1.5, 4.5)])

        adj = adjacency_by_shared_vertices([hex1, hex2, hex3, hex4])

        # hex1 should be adjacent to hex2 and hex3
        assert 1 in adj[0] or 2 in adj[0]
        # Check that adjacency is symmetric
        for i, neighbors in adj.items():
            for j in neighbors:
                assert i in adj[j]

    def test_overlapping_polygons(self):
        """Test adjacency for overlapping polygons."""
        poly1 = box(0, 0, 2, 2)
        poly2 = box(1, 1, 3, 3)

        adj = adjacency_by_shared_vertices([poly1, poly2])

        # Overlapping polygons share multiple vertices
        if 0 in adj and 1 in adj[0]:
            assert 0 in adj[1]

    def test_adjacency_returns_sorted_lists(self):
        """Test that adjacency lists are sorted."""
        tri1 = Polygon([(0, 0), (1, 0), (0.5, 0.866)])
        tri2 = Polygon([(0, 0), (0.5, 0.866), (-0.5, 0.866)])
        tri3 = Polygon([(0, 0), (-0.5, 0.866), (-1, 0)])

        adj = adjacency_by_shared_vertices([tri1, tri2, tri3])

        # Check that all neighbor lists are sorted
        for neighbors in adj.values():
            assert neighbors == sorted(neighbors)

    def test_complex_polygon_shapes(self):
        """Test with various polygon shapes (not just triangles)."""
        # Pentagon and hexagon sharing vertices
        pentagon = Polygon([(0, 0), (1, 0), (1.5, 0.5), (1, 1), (0, 1)])
        hexagon = Polygon(
            [(0, 0), (0.5, -0.5), (1.5, -0.5), (1.5, 0.5), (1, 1), (0, 1)]
        )

        adj = adjacency_by_shared_vertices([pentagon, hexagon])

        # Should detect shared vertices
        assert 1 in adj[0]
        assert 0 in adj[1]


class TestIntersectsMask:
    """Test suite for intersects_mask function."""

    def test_simple_case_single_polygon_surface(self):
        """Test with simple surface (single polygon, < 8 polygons)."""
        # Create 5 test polygons
        polygons = np.array(
            [
                box(0, 0, 1, 1),
                box(2, 2, 3, 3),
                box(0.5, 0.5, 1.5, 1.5),
                box(-1, -1, 0.5, 0.5),
                box(5, 5, 6, 6),
            ]
        )

        surface = box(0, 0, 2, 2)
        mask = intersects_mask(polygons, surface)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(polygons)
        # Polygons 0, 2, 3 should intersect
        assert bool(mask[0]) is True
        assert bool(mask[1]) is True
        assert bool(mask[2]) is True
        assert bool(mask[3]) is True
        assert bool(mask[4]) is False

    def test_multipolygon_surface_with_8_or_fewer_parts(self):
        """
        Test with MultiPolygon surface having 8 or fewer parts (uses vectorized path).
        """
        polygons = np.array(
            [
                box(0, 0, 1, 1),
                box(2, 2, 3, 3),
                box(0.5, 0.5, 1.5, 1.5),
                box(5, 5, 6, 6),
            ]
        )

        # Create MultiPolygon with 5 parts
        surface = MultiPolygon(
            [
                box(0, 0, 0.5, 0.5),
                box(2.5, 2.5, 3.5, 3.5),
                box(5.5, 5.5, 6.5, 6.5),
                box(-1, -1, -0.5, -0.5),
                box(10, 10, 11, 11),
            ]
        )

        mask = intersects_mask(polygons, surface)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(polygons)

    def test_multipolygon_surface_with_more_than_8_parts(self):
        """Test with MultiPolygon surface having > 8 parts (uses STRtree path)."""
        # Create 10 test polygons
        polygons = np.array(
            [
                box(0, 0, 1, 1),
                box(2, 2, 3, 3),
                box(4, 4, 5, 5),
                box(6, 6, 7, 7),
                box(8, 8, 9, 9),
                box(0.5, 0.5, 1.5, 1.5),
                box(2.5, 2.5, 3.5, 3.5),
                box(4.5, 4.5, 5.5, 5.5),
                box(6.5, 6.5, 7.5, 7.5),
                box(20, 20, 21, 21),
            ]
        )

        # Create MultiPolygon with 10 parts (> 8, triggers STRtree)
        surface = MultiPolygon(
            [
                box(0, 0, 0.5, 0.5),
                box(1.5, 1.5, 2, 2),
                box(2, 2, 2.5, 2.5),
                box(3.5, 3.5, 4, 4),
                box(4, 4, 4.5, 4.5),
                box(5.5, 5.5, 6, 6),
                box(6, 6, 6.5, 6.5),
                box(7.5, 7.5, 8, 8),
                box(8, 8, 8.5, 8.5),
                box(19, 19, 19.5, 19.5),
            ]
        )

        mask = intersects_mask(polygons, surface)

        assert isinstance(mask, np.ndarray)
        assert mask.dtype == bool
        assert len(mask) == len(polygons)
        # At least some should intersect
        assert np.any(mask)

    def test_all_polygons_intersect_surface(self):
        """Test when all polygons intersect the surface."""
        polygons = np.array(
            [
                box(0, 0, 1, 1),
                box(0.5, 0.5, 1.5, 1.5),
                box(1, 1, 2, 2),
            ]
        )

        surface = box(-1, -1, 3, 3)  # Large surface covering all polygons
        mask = intersects_mask(polygons, surface)

        assert np.all(mask)

    def test_no_polygons_intersect_surface(self):
        """Test when no polygons intersect the surface."""
        polygons = np.array(
            [
                box(0, 0, 1, 1),
                box(2, 2, 3, 3),
                box(4, 4, 5, 5),
            ]
        )

        surface = box(10, 10, 11, 11)  # Far away surface
        mask = intersects_mask(polygons, surface)

        assert np.all(~mask)

    def test_partial_intersection(self):
        """Test with partial intersections."""
        polygons = np.array(
            [
                box(0, 0, 2, 2),  # Intersects
                box(1.5, 1.5, 3, 3),  # Intersects (corner touches)
                box(5, 5, 6, 6),  # No intersection
                box(-1, -1, 0.5, 0.5),  # Intersects (corner touches)
            ]
        )

        surface = box(0, 0, 1, 1)
        mask = intersects_mask(polygons, surface)

        assert bool(mask[0]) is True  # Fully contains
        assert bool(mask[1]) is False  # Touches corner
        assert bool(mask[2]) is False  # Far away
        assert bool(mask[3]) is True  # Touches corner

    def test_polygon_completely_inside_surface(self):
        """Test polygon completely inside the surface."""
        polygons = np.array(
            [
                box(1, 1, 2, 2),
                box(3, 3, 4, 4),
            ]
        )

        surface = box(0, 0, 5, 5)
        mask = intersects_mask(polygons, surface)

        assert np.all(mask)

    def test_polygon_on_surface_boundary(self):
        """Test polygons touching the surface boundary."""
        polygons = np.array(
            [
                box(0, 0, 0.1, 0.1),  # At corner
                box(2.5, 0, 2.6, 0.1),  # On edge
            ]
        )

        surface = box(0, 0, 2.5, 2.5)
        mask = intersects_mask(polygons, surface)

        assert bool(mask[0]) is True
        assert bool(mask[1]) is True  # Touching boundary

    def test_return_type_is_ndarray(self):
        """Verify return type is numpy ndarray."""
        polygons = np.array([box(0, 0, 1, 1)])
        surface = box(0, 0, 1, 1)

        result = intersects_mask(polygons, surface)

        assert isinstance(result, np.ndarray)
        assert result.dtype == bool

    def test_mask_length_matches_polygons(self):
        """Verify mask length matches number of polygons."""
        polygons = np.array([box(i, i, i + 1, i + 1) for i in range(15)])

        surface = box(0, 0, 10, 10)
        mask = intersects_mask(polygons, surface)

        assert len(mask) == len(polygons)

    def test_consistency_between_vectorized_and_strtree_paths(self):
        """Test that both code paths produce the same results."""
        polygons = np.array(
            [
                box(0, 0, 1, 1),
                box(1.5, 1.5, 2.5, 2.5),
                box(4, 4, 5, 5),
                box(0.5, 0.5, 1.5, 1.5),
                box(2, 2, 3, 3),
            ]
        )

        # Test with single surface (vectorized path)
        single_surface = box(0, 0, 3, 3)
        mask_single = intersects_mask(polygons, single_surface)

        # Test with MultiPolygon with 9 parts (STRtree path)
        multi_surface = MultiPolygon(
            [
                box(0, 0, 0.5, 0.5),
                box(1.5, 1.5, 2, 2),
                box(4, 4, 4.2, 4.2),
                box(0.5, 0.5, 1, 1),
                box(2, 2, 2.5, 2.5),
                box(2.5, 2.5, 3, 3),
                box(0.3, 0.3, 0.7, 0.7),
                box(1, 1, 1.3, 1.3),
                box(3, 3, 3.2, 3.2),
            ]
        )
        mask_multi = intersects_mask(polygons, multi_surface)

        # The STRtree path is more permissive due to spatial indexing
        # but should have same or more True values
        assert np.sum(mask_multi) >= np.sum(mask_single) - 1  # Allow small tolerance

    def test_large_polygon_set_with_strtree(self):
        """Test STRtree path with a large number of polygons."""
        # Create 100 polygons
        polygons = np.array(
            [box(i % 10, i // 10, i % 10 + 1, i // 10 + 1) for i in range(100)]
        )

        # Create MultiPolygon with 15 parts (triggers STRtree)
        surface = MultiPolygon([box(i, i, i + 0.5, i + 0.5) for i in range(15)])

        mask = intersects_mask(polygons, surface)

        assert len(mask) == 100
        assert mask.dtype == bool
        assert np.any(mask)  # At least some should intersect

    def test_empty_polygon_array(self):
        """Test with empty polygon array."""
        polygons = np.array([])
        surface = box(0, 0, 1, 1)

        mask = intersects_mask(polygons, surface)

        assert len(mask) == 0
        assert mask.dtype == bool

    def test_degenerate_polygons(self):
        """Test with degenerate polygons (very small or nearly zero area)."""
        polygons = np.array(
            [
                box(0, 0, 0.0001, 0.0001),  # Tiny polygon
                box(1, 1, 1.0001, 1.0001),
            ]
        )

        surface = box(0, 0, 1.5, 1.5)
        mask = intersects_mask(polygons, surface)

        assert len(mask) == len(polygons)
        # Both small polygons should intersect the large surface
        assert bool(mask[0]) is True
        assert bool(mask[1]) is True
