# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

"""Tests for :mod:`quickpaver.tiling`."""

import math
from typing import Dict, List, Optional, Sequence, Tuple, cast

import numpy as np
import pytest
import shapely
from quickpaver._tiling import (
    DEFAULT_ALIGNMENT_POINT,
    SQRT3,
    Disk,
    PolygonType,
    _disk_mask_from_rings,
    _distance_to_convex_rings,
    _lattice_centres,
    _pairs_to_adj,
    _ragged_arange,
    _reference_tile_vertices,
    _resolve_anchor,
    _tiles_from_centres,
    adjacency_by_shared_vertices,
    extract_tiling_centers,
    extract_tiling_vertices,
    gen_hexagonal_tiling,
    gen_polygon,
    gen_polygonal_tiling,
    gen_rectangular_tiling,
    gen_triangular_tiling,
    hexagonal_grid_adjacency_masked,
    intersects_mask,
    rectangular_grid_adjacency_masked,
    triangular_grid_adjacency_masked,
)

EDGE = 0.9

GENERATORS = {
    PolygonType.RECTANGLE: gen_rectangular_tiling,
    PolygonType.HEXAGON: gen_hexagonal_tiling,
    PolygonType.TRIANGLE: gen_triangular_tiling,
}


# --------------------------------------------------------------------------
# Fixtures and reference implementations
# --------------------------------------------------------------------------


@pytest.fixture
def square() -> shapely.Polygon:
    """Return a simple convex surface."""
    return shapely.box(0.0, 0.0, 6.0, 6.0)


@pytest.fixture
def lshape() -> shapely.Polygon:
    """Return a concave surface."""
    return shapely.Polygon([(0, 0), (12, 0), (12, 4), (5, 4), (5, 13), (0, 13)])


@pytest.fixture
def multi() -> shapely.MultiPolygon:
    """Return a multi-part surface with more than eight parts."""
    return shapely.MultiPolygon(
        [
            shapely.box(i * 6.0, j * 6.0, i * 6.0 + 2.0, j * 6.0 + 2.0)
            for i in range(4)
            for j in range(4)
        ]
    )


@pytest.fixture
def disk() -> Disk:
    """Return a disk surface."""
    return Disk((3.0, -2.0), 6.0)


def reference_adjacency(
    polygons: Sequence[shapely.Polygon], tol: float = 1e-9
) -> Dict[int, List[int]]:
    """Adjacency from a geometric edge-sharing test.

    Two polygons are neighbours only when their intersection has positive
    length, i.e. they share a boundary segment. Polygons that merely touch
    at a single point (a shared vertex, as happens at the corner of a fan
    of triangles or the diagonal corner of a rectangular grid) are not
    neighbours: their intersection is a `Point` (or `MultiPoint`), whose
    `.length` is zero, so it is excluded.

    The intersection is computed on a grid-snapped copy of the geometries
    (same ``UNION_GRID_SIZE`` used by :func:`robust_union`). Two tiles that
    share a true edge were generated from the same lattice but not
    necessarily from bit-identical floating-point coordinates, so an exact
    intersection can spuriously come back empty; snapping first makes the
    length test robust to that without blurring genuine vertex-only
    touches, which still snap to a zero-length point.
    """
    array = np.array(polygons)
    tree = shapely.STRtree(array)
    left, right = tree.query(array, predicate="dwithin", distance=tol)
    adj: Dict[int, set] = {i: set() for i in range(len(polygons))}
    for source, target in zip(left, right):
        if source == target:
            continue
        inter = shapely.intersection(
            array[source], array[target], grid_size=UNION_GRID_SIZE
        )
        if inter.length > tol:
            adj[int(source)].add(int(target))
    return {i: sorted(v) for i, v in adj.items()}


# Tiles meet along long, exactly shared edges. Overlaying them without a
# grid size makes the GEOS overlay drop slivers and report spurious holes,
# so every union below snaps to a grid far finer than any tile.
UNION_GRID_SIZE = 1e-9


def robust_union(polygons: Sequence[shapely.Polygon]) -> shapely.Geometry:
    """Return the union of *polygons*, snapped to a fine grid."""
    return shapely.union_all(np.array(list(polygons)), grid_size=UNION_GRID_SIZE)


def covered_area_error(
    surface: shapely.Polygon, polygons: Sequence[shapely.Polygon]
) -> float:
    """Return the area of *surface* left uncovered by *polygons*."""
    return surface.difference(robust_union(polygons)).area


def overlap_error(polygons: Sequence[shapely.Polygon]) -> float:
    """Return the total overlapping area between *polygons*."""
    return abs(sum(p.area for p in polygons) - robust_union(polygons).area)


# --------------------------------------------------------------------------
# Array helpers
# --------------------------------------------------------------------------


def test_ragged_arange() -> None:
    """Runs of increasing integers are concatenated."""
    result = _ragged_arange(np.array([3, 0, 2, 1]))
    assert np.array_equal(result, [0, 1, 2, 0, 1, 0])


def test_ragged_arange_empty() -> None:
    """A total count of zero yields an empty array."""
    result = _ragged_arange(np.zeros(4, dtype=np.int64))
    assert result.size == 0
    assert result.dtype == np.int64


def test_pairs_to_adj_matches_naive() -> None:
    """Grouping reproduces a dictionary built edge by edge."""
    rng = np.random.default_rng(0)
    n_nodes = 40
    src = rng.integers(0, n_nodes, 500)
    dst = rng.integers(0, n_nodes, 500)

    naive: Dict[int, set] = {i: set() for i in range(n_nodes)}
    for s, d in zip(src, dst):
        naive[int(s)].add(int(d))

    assert _pairs_to_adj(src, dst, n_nodes) == {i: sorted(v) for i, v in naive.items()}


def test_pairs_to_adj_no_nodes() -> None:
    """A graph without nodes yields an empty dictionary."""
    assert _pairs_to_adj(np.array([]), np.array([]), 0) == {}


def test_pairs_to_adj_no_edges() -> None:
    """Nodes without edges are present with empty neighbour lists."""
    assert _pairs_to_adj(np.array([]), np.array([]), 3) == {
        0: [],
        1: [],
        2: [],
    }


def test_pairs_to_adj_deduplicates() -> None:
    """Repeated pairs appear once in the neighbour list."""
    src = np.array([0, 0, 0, 1])
    dst = np.array([1, 1, 2, 0])
    assert _pairs_to_adj(src, dst, 3) == {0: [1, 2], 1: [0], 2: []}


# --------------------------------------------------------------------------
# Disk
# --------------------------------------------------------------------------


def test_disk_bounds_and_centroid() -> None:
    """Bounds and centroid follow from the centre and radius."""
    d = Disk((2.0, -1.0), 3.0)
    assert d.bounds == (-1.0, -4.0, 5.0, 2.0)
    assert d.centroid.equals(shapely.Point(2.0, -1.0))
    assert np.array_equal(d.as_array(), [2.0, -1.0])


def test_disk_to_polygon_is_inscribed() -> None:
    """The polygonal approximation is inscribed in the disk."""
    d = Disk((0.0, 0.0), 5.0)
    poly = d.to_polygon(quad_segs=16)
    assert poly.area < math.pi * 25.0
    assert poly.area > 0.99 * math.pi * 25.0
    assert d.to_polygon().area > poly.area


@pytest.mark.parametrize("radius", [0.0, -1.0])
def test_disk_rejects_non_positive_radius(radius: float) -> None:
    """A non-positive radius is refused."""
    with pytest.raises(ValueError, match="radius must be strictly positive"):
        Disk((0.0, 0.0), radius)


def test_distance_to_convex_rings_inside_and_outside() -> None:
    """The distance vanishes inside the polygon and is exact outside."""
    ring = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]])
    rings = np.stack([ring, ring + np.array([5.0, 0.0])])
    distance = _distance_to_convex_rings(np.array([1.0, 1.0]), rings)
    assert distance[0] == pytest.approx(0.0)
    assert distance[1] == pytest.approx(4.0)


def test_disk_mask_from_rings_empty() -> None:
    """An empty candidate set yields an empty mask."""
    rings = np.zeros((0, 4, 2))
    mask = _disk_mask_from_rings(rings, Disk((0.0, 0.0), 1.0))
    assert mask.shape == (0,)


def test_disk_mask_from_rings_without_band() -> None:
    """A tile far inside the disk is kept without an exact test."""
    ring = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    mask = _disk_mask_from_rings(ring[None, ...], Disk((0.0, 0.0), 100.0))
    assert mask.tolist() == [True]


def test_disk_mask_from_rings_rejects_far_tile() -> None:
    """A tile beyond the outer circle is discarded."""
    ring = np.array([[9.0, 9.0], [10.0, 9.0], [10.0, 10.0], [9.0, 10.0]])
    mask = _disk_mask_from_rings(ring[None, ...], Disk((0.0, 0.0), 1.0))
    assert mask.tolist() == [False]


# --------------------------------------------------------------------------
# Reference polygons
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("poly_type", "n_vertices"),
    [
        (PolygonType.TRIANGLE, 3),
        (PolygonType.RECTANGLE, 4),
        (PolygonType.HEXAGON, 6),
    ],
)
def test_gen_polygon_shape(poly_type: PolygonType, n_vertices: int) -> None:
    """Each polygon type has the expected vertex count and is centred."""
    poly = gen_polygon(poly_type, 2.0)
    assert len(poly.exterior.coords) == n_vertices + 1
    assert poly.centroid.x == pytest.approx(0.0)
    assert poly.centroid.y == pytest.approx(0.0)


def test_gen_polygon_anisotropy_scales_height() -> None:
    """The anisotropy ratio stretches the polygon along y only."""
    iso = gen_polygon(PolygonType.RECTANGLE, 2.0)
    aniso = gen_polygon(PolygonType.RECTANGLE, 2.0, 3.0)
    assert aniso.bounds[2] == pytest.approx(iso.bounds[2])
    assert aniso.bounds[3] == pytest.approx(3.0 * iso.bounds[3])


def test_gen_polygon_rejects_unknown_type() -> None:
    """An unsupported polygon type is refused."""
    with pytest.raises(ValueError):
        gen_polygon("pentagon")


def test_reference_tile_vertices_drops_closing_repeat() -> None:
    """The open ring omits the repeated closing vertex."""
    verts = _reference_tile_vertices(PolygonType.HEXAGON, 1.0, 1.0)
    assert verts.shape == (6, 2)
    assert not np.allclose(verts[0], verts[-1])


# --------------------------------------------------------------------------
# Lattice
# --------------------------------------------------------------------------


def test_lattice_centres_contains_anchor() -> None:
    """The anchor is a lattice node by construction."""
    anchor = np.array([1.234, -0.77])
    centres = _lattice_centres(
        (0.0, 0.0, 4.0, 4.0),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        anchor,
    )
    flat = np.moveaxis(centres, 0, -1).reshape(-1, 2)
    assert np.abs(flat - anchor).sum(axis=1).min() == pytest.approx(0.0)


def test_lattice_centres_covers_bounds() -> None:
    """The generated nodes extend beyond the requested bounds."""
    centres = _lattice_centres(
        (0.0, 0.0, 4.0, 4.0),
        np.array([1.0, 0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0]),
    )
    assert centres[0].min() < 0.0
    assert centres[0].max() > 4.0


# --------------------------------------------------------------------------
# Validation
# --------------------------------------------------------------------------


@pytest.mark.parametrize("poly_type", list(PolygonType))
@pytest.mark.parametrize("edge_length", [0.0, -1.0])
def test_generators_reject_non_positive_edge_length(
    square: shapely.Polygon, poly_type: PolygonType, edge_length: float
) -> None:
    """A non-positive edge length is refused."""
    with pytest.raises(ValueError, match="edge_length"):
        GENERATORS[poly_type](square, edge_length)


@pytest.mark.parametrize("poly_type", list(PolygonType))
@pytest.mark.parametrize("anisotropy_ratio", [0.0, -2.0])
def test_generators_reject_non_positive_anisotropy(
    square: shapely.Polygon, poly_type: PolygonType, anisotropy_ratio: float
) -> None:
    """A non-positive anisotropy ratio is refused."""
    with pytest.raises(ValueError, match="anisotropy_ratio"):
        GENERATORS[poly_type](square, 1.0, anisotropy_ratio)


# --------------------------------------------------------------------------
# Tilings on polygonal surfaces
# --------------------------------------------------------------------------


@pytest.mark.parametrize("poly_type", list(PolygonType))
@pytest.mark.parametrize("anisotropy_ratio", [1.0, 2.3])
@pytest.mark.parametrize("alignment_point", [None, (1.234, 3.77)])
def test_tiling_covers_square_without_overlap(
    square: shapely.Polygon,
    poly_type: PolygonType,
    anisotropy_ratio: float,
    alignment_point: Optional[Tuple[float, float]],
) -> None:
    """The tiling covers the surface exactly once."""
    tiling, adjacency = GENERATORS[poly_type](
        square, EDGE, anisotropy_ratio, alignment_point
    )
    polygons = list(tiling.geoms)
    assert covered_area_error(square, polygons) < 1e-9
    assert overlap_error(polygons) < 1e-6
    assert len(adjacency) == len(polygons)


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_tiling_keeps_only_intersecting_tiles(
    lshape: shapely.Polygon, poly_type: PolygonType
) -> None:
    """Every returned tile touches the surface."""
    tiling, _ = GENERATORS[poly_type](lshape, EDGE)
    assert all(p.intersects(lshape) for p in tiling.geoms)


@pytest.mark.parametrize("poly_type", list(PolygonType))
@pytest.mark.parametrize("alignment_point", [None, (1.234, 3.77)])
def test_structured_adjacency_matches_geometry(
    lshape: shapely.Polygon,
    poly_type: PolygonType,
    alignment_point: Optional[Tuple[float, float]],
) -> None:
    """Index-based adjacency agrees with a geometric edge-sharing test."""
    tiling, adjacency = GENERATORS[poly_type](lshape, EDGE, 1.0, alignment_point)
    assert adjacency == reference_adjacency(list(tiling.geoms))


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_tiling_on_multipart_surface(
    multi: shapely.MultiPolygon, poly_type: PolygonType
) -> None:
    """A multi-part surface is covered through the spatial index path."""
    tiling, adjacency = GENERATORS[poly_type](multi, EDGE)
    polygons = list(tiling.geoms)
    assert covered_area_error(multi, polygons) < 1e-9
    assert adjacency == reference_adjacency(polygons)


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_alignment_point_inside_surface_hits_a_tile_centre(
    square: shapely.Polygon, poly_type: PolygonType
) -> None:
    """A tile centroid lands exactly on an interior alignment point."""
    point = np.array([1.234, 3.77])
    tiling, _ = GENERATORS[poly_type](square, EDGE, 1.7, point)
    centroids = np.array([next(iter(p.centroid.coords)) for p in tiling.geoms])
    assert np.abs(centroids - point).sum(axis=1).min() < 1e-9


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_alignment_point_outside_surface(
    square: shapely.Polygon, poly_type: PolygonType
) -> None:
    """An exterior alignment point still yields a valid tiling."""
    tiling, adjacency = GENERATORS[poly_type](square, EDGE, 1.0, (-25.0, -30.0))
    polygons = list(tiling.geoms)
    assert covered_area_error(square, polygons) < 1e-9
    assert adjacency == reference_adjacency(polygons)


# --------------------------------------------------------------------------
# Tilings on a disk
# --------------------------------------------------------------------------


@pytest.mark.parametrize("poly_type", list(PolygonType))
@pytest.mark.parametrize("anisotropy_ratio", [1.0, 2.3])
def test_disk_tiling_matches_fine_polygon(
    disk: Disk, poly_type: PolygonType, anisotropy_ratio: float
) -> None:
    """The analytic disk path reproduces a finely resolved polygon."""
    analytic, adj_analytic = GENERATORS[poly_type](disk, EDGE, anisotropy_ratio)
    reference, adj_reference = GENERATORS[poly_type](
        disk.to_polygon(quad_segs=512), EDGE, anisotropy_ratio
    )
    assert {g.wkt for g in analytic.geoms} == {g.wkt for g in reference.geoms}
    assert adj_analytic == adj_reference


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_disk_tiling_covers_the_disk(disk: Disk, poly_type: PolygonType) -> None:
    """The tiling covers the exact disk."""
    tiling, _ = GENERATORS[poly_type](disk, EDGE)
    fine = disk.to_polygon(quad_segs=512)
    assert covered_area_error(fine, list(tiling.geoms)) < 1e-6


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_disk_tiling_keeps_only_intersecting_tiles(
    disk: Disk, poly_type: PolygonType
) -> None:
    """No returned tile lies entirely outside the disk."""
    tiling, _ = GENERATORS[poly_type](disk, EDGE)
    centre = shapely.Point(disk.center)
    assert all(shapely.distance(p, centre) <= disk.radius + 1e-12 for p in tiling.geoms)


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_disk_tiling_is_at_least_as_complete_as_buffer(
    disk: Disk, poly_type: PolygonType
) -> None:
    """The exact circle never drops a tile that the buffer keeps."""
    analytic, _ = GENERATORS[poly_type](disk, EDGE)
    buffered, _ = GENERATORS[poly_type](disk.to_polygon(quad_segs=8), EDGE)
    assert {g.wkt for g in buffered.geoms} <= {g.wkt for g in analytic.geoms}


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_disk_tiling_with_alignment_point(disk: Disk, poly_type: PolygonType) -> None:
    """Alignment works on a disk, on the lattice construction path."""
    point = np.array([3.0, -2.0])
    tiling, adjacency = GENERATORS[poly_type](disk, EDGE, 1.0, point)
    centroids = np.array([next(iter(p.centroid.coords)) for p in tiling.geoms])
    assert np.abs(centroids - point).sum(axis=1).min() < 1e-9
    assert adjacency == reference_adjacency(list(tiling.geoms))


def test_disk_tiling_with_exterior_alignment_point(disk: Disk) -> None:
    """An alignment point outside the disk is accepted."""
    tiling, _ = gen_rectangular_tiling(disk, EDGE, 1.0, (50.0, 50.0))
    fine = disk.to_polygon(quad_segs=512)
    assert covered_area_error(fine, list(tiling.geoms)) < 1e-6


# --------------------------------------------------------------------------
# intersects_mask
# --------------------------------------------------------------------------


def test_intersects_mask_empty_input(square: shapely.Polygon) -> None:
    """An empty polygon array yields an empty mask."""
    mask = intersects_mask(np.array([], dtype=object), square)
    assert mask.shape == (0,)


def test_intersects_mask_simple_surface(square: shapely.Polygon) -> None:
    """A simple surface uses the vectorized predicate."""
    polygons = np.array([shapely.box(1, 1, 2, 2), shapely.box(20, 20, 21, 21)])
    assert intersects_mask(polygons, square).tolist() == [True, False]


def test_intersects_mask_multipart_surface(
    multi: shapely.MultiPolygon,
) -> None:
    """A surface with many parts uses the spatial index path."""
    polygons = np.array([shapely.box(0.5, 0.5, 1.0, 1.0), shapely.box(-9, -9, -8, -8)])
    assert intersects_mask(polygons, multi).tolist() == [True, False]


def test_intersects_mask_small_multipart_surface() -> None:
    """A surface with few parts stays on the direct predicate path."""
    surface = shapely.MultiPolygon([shapely.box(0, 0, 1, 1), shapely.box(5, 5, 6, 6)])
    polygons = np.array([shapely.box(0.2, 0.2, 0.4, 0.4)])
    assert intersects_mask(polygons, surface).tolist() == [True]


def test_intersects_mask_disk(disk: Disk) -> None:
    """A disk surface is handled analytically."""
    polygons = np.array([shapely.box(3, -2, 4, -1), shapely.box(100, 100, 101, 101)])
    assert intersects_mask(polygons, disk).tolist() == [True, False]


def test_tiles_from_centres_disk_branch(disk: Disk) -> None:
    """The centre-lattice helper accepts a disk directly."""
    centres = np.array(
        np.meshgrid(np.linspace(-2.0, 8.0, 12), np.linspace(-7.0, 3.0, 12))
    )
    verts = _reference_tile_vertices(PolygonType.RECTANGLE, EDGE, 1.0)
    polygons, mask = _tiles_from_centres(centres, verts, disk)
    assert len(polygons) == int(mask.sum())
    assert mask.size == centres.shape[1] * centres.shape[2]


# --------------------------------------------------------------------------
# Adjacency builders
# --------------------------------------------------------------------------


def test_rectangular_adjacency_is_four_connected() -> None:
    """An interior cell of a full grid has four edge-sharing neighbours."""
    mask = np.ones(9, dtype=bool)
    adjacency = rectangular_grid_adjacency_masked(3, 3, mask)
    assert len(adjacency[4]) == 4
    assert len(adjacency[0]) == 2


def test_rectangular_adjacency_excludes_diagonal_neighbours() -> None:
    """Cells touching only at a corner are not adjacent."""
    mask = np.ones(4, dtype=bool)
    adjacency = rectangular_grid_adjacency_masked(2, 2, mask)
    # Flat index 0 is grid cell (0, 0); flat index 3 is grid cell (1, 1),
    # which is diagonal to (0, 0) and shares only a corner vertex.
    assert 3 not in adjacency[0]
    assert 0 not in adjacency[3]
    assert len(adjacency[0]) == 2


def test_rectangular_adjacency_honours_mask() -> None:
    """Masked-out cells are absent and indices stay compact."""
    mask = np.array([True, False, True, True], dtype=bool)
    adjacency = rectangular_grid_adjacency_masked(2, 2, mask)
    assert set(adjacency) == {0, 1, 2}
    assert all(n in adjacency for neigh in adjacency.values() for n in neigh)


def test_hexagonal_adjacency_sheared() -> None:
    """Both column layouts give six neighbours to an interior cell."""
    mask = np.ones(25, dtype=bool)
    sheared = hexagonal_grid_adjacency_masked(5, 5, mask)
    assert len(sheared[12]) == 6


def test_triangular_adjacency_is_symmetric() -> None:
    """Neighbour relations of the triangle lattice are symmetric."""
    mask = np.ones(4 * 4 * 2, dtype=bool)
    adjacency = triangular_grid_adjacency_masked(4, 4, mask)
    for node, neighbours in adjacency.items():
        for neighbour in neighbours:
            assert node in adjacency[neighbour]


def test_triangular_adjacency_has_at_most_three_edge_neighbours() -> None:
    """Every triangle has exactly 3 edges, so at most 3 neighbours."""
    mask = np.ones(4 * 4 * 2, dtype=bool)
    adjacency = triangular_grid_adjacency_masked(4, 4, mask)
    assert all(len(neighbours) <= 3 for neighbours in adjacency.values())
    # Cell (1, 1), k=0 -> flat index (1 * 4 + 1) * 2 + 0 = 10, an interior
    # triangle away from the lattice boundary, so it has all 3 neighbours.
    assert len(adjacency[10]) == 3


def test_triangular_adjacency_excludes_vertex_only_neighbours() -> None:
    """Triangles sharing only a corner vertex are not adjacent."""
    mask = np.ones(3 * 3 * 2, dtype=bool)
    adjacency = triangular_grid_adjacency_masked(3, 3, mask)
    # Cell (1, 1), k=0 -> flat index (1 * 3 + 1) * 2 + 0 = 8, an interior
    # triangle with exactly 3 edge-sharing neighbours (no vertex-only fan
    # neighbours counted in).
    assert len(adjacency[8]) == 3


def test_adjacency_builders_accept_empty_mask() -> None:
    """A fully masked-out grid yields an empty dictionary."""
    empty = np.zeros(9, dtype=bool)
    assert rectangular_grid_adjacency_masked(3, 3, empty) == {}
    assert hexagonal_grid_adjacency_masked(3, 3, empty) == {}
    assert triangular_grid_adjacency_masked(3, 3, np.zeros(18, dtype=bool)) == {}


# --------------------------------------------------------------------------
# Vertex extraction
# --------------------------------------------------------------------------


def test_extract_tiling_centers_from_multipolygon(
    square: shapely.Polygon,
) -> None:
    """Centres are extracted from a multipolygon."""
    tiling, _ = gen_rectangular_tiling(square, EDGE)
    centers = extract_tiling_centers(tiling)
    assert centers.shape == (len(tiling.geoms), 2)


def test_extract_tiling_centers_from_iterable(
    square: shapely.Polygon,
) -> None:
    """Centres are extracted from a plain iterable of polygons."""
    tiling, _ = gen_rectangular_tiling(square, EDGE)
    centers = extract_tiling_centers(list(tiling.geoms))
    assert centers.shape == (len(tiling.geoms), 2)


def test_extract_tiling_vertices_round_trip(square: shapely.Polygon) -> None:
    """Cluster indices map deduplicated vertices back to the input."""
    tiling, _ = gen_hexagonal_tiling(square, 1.0)
    polygons = list(tiling.geoms)
    coords, vert_to_polys, clusters, poly_to_verts = extract_tiling_vertices(polygons)

    assert np.array_equal(coords, coords[np.lexsort((coords[:, 1], coords[:, 0]))])
    for poly_id, vertex_ids in poly_to_verts.items():
        ring = np.round(np.array(polygons[poly_id].exterior.coords)[:-1], 2)
        assert np.allclose(coords[vertex_ids], ring)
        for vertex_id in vertex_ids:
            assert poly_id in vert_to_polys[vertex_id]
    assert clusters.shape == (6 * len(polygons),)


def test_extract_tiling_vertices_from_multipolygon(
    square: shapely.Polygon,
) -> None:
    """A multipolygon input is accepted directly."""
    tiling, _ = gen_rectangular_tiling(square, EDGE)
    coords, _, _, poly_to_verts = extract_tiling_vertices(tiling)
    assert len(poly_to_verts) == len(tiling.geoms)
    assert coords.shape[1] == 2


def test_extract_tiling_vertices_deduplicates_shared_corners() -> None:
    """Two touching squares share exactly two vertices."""
    polygons = [shapely.box(0, 0, 1, 1), shapely.box(1, 0, 2, 1)]
    coords, vert_to_polys, _, _ = extract_tiling_vertices(polygons)
    assert len(coords) == 6
    shared = [v for v, polys in vert_to_polys.items() if len(polys) == 2]
    assert len(shared) == 2


def test_adjacency_by_shared_vertices_matches_structured(
    square: shapely.Polygon,
) -> None:
    """Vertex-based adjacency agrees with the structured builder.

    This equivalence holds for the hexagonal tiling because, on this
    lattice, any two hexagons that share a vertex also share a full edge:
    there is no configuration where hexagons meet only at a single corner.
    That is not true in general (e.g. triangles and rectangles have
    vertex-only fan/diagonal neighbours), so this comparison is not
    extended to the other tile types.
    """
    tiling, adjacency = gen_hexagonal_tiling(square, 1.0)
    polygons = list(tiling.geoms)
    assert adjacency_by_shared_vertices(polygons) == adjacency


def test_adjacency_by_shared_vertices_isolated_polygon() -> None:
    """A polygon without shared vertices maps to an empty list."""
    polygons = [shapely.box(0, 0, 1, 1), shapely.box(10, 10, 11, 11)]
    assert adjacency_by_shared_vertices(polygons) == {0: [], 1: []}


# --------------------------------------------------------------------------
# gen_polygonal_tiling
# --------------------------------------------------------------------------


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_polygonal_tiling_without_rotation(
    square: shapely.Polygon, poly_type: PolygonType
) -> None:
    """Without rotation the dispatcher matches the dedicated generator."""
    dispatched, adj_dispatched = gen_polygonal_tiling(square, poly_type, EDGE, 1.4)
    direct, adj_direct = GENERATORS[poly_type](square, EDGE, 1.4)
    assert {g.wkt for g in dispatched.geoms} == {g.wkt for g in direct.geoms}
    assert adj_dispatched == adj_direct


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_polygonal_tiling_with_rotation(
    lshape: shapely.Polygon, poly_type: PolygonType
) -> None:
    """A rotated tiling still covers the surface."""
    tiling, adjacency = gen_polygonal_tiling(lshape, poly_type, 0.8, 1.4, rot_deg=27.0)
    polygons = list(tiling.geoms)
    assert covered_area_error(lshape, polygons) < 1e-9
    assert len(adjacency) == len(polygons)


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_polygonal_tiling_rotated_alignment_point(
    lshape: shapely.Polygon, poly_type: PolygonType
) -> None:
    """The alignment point is honoured in world space after rotation."""
    point = np.array([2.0, 3.0])
    tiling, _ = gen_polygonal_tiling(
        lshape, poly_type, 0.8, 1.4, rot_deg=27.0, alignment_point=point
    )
    centroids = np.array([next(iter(p.centroid.coords)) for p in tiling.geoms])
    assert np.abs(centroids - point).sum(axis=1).min() < 1e-8


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_polygonal_tiling_on_disk_with_rotation(
    disk: Disk, poly_type: PolygonType
) -> None:
    """A disk is rotation invariant, so the tile set is unchanged in size."""
    rotated, _ = gen_polygonal_tiling(disk, poly_type, EDGE, rot_deg=31.0)
    upright, _ = gen_polygonal_tiling(disk, poly_type, EDGE)
    assert len(rotated.geoms) == len(upright.geoms)
    fine = disk.to_polygon(quad_segs=512)
    assert covered_area_error(fine, list(rotated.geoms)) < 1e-6


def test_polygonal_tiling_rejects_unknown_type(
    square: shapely.Polygon,
) -> None:
    """An unsupported tile type is refused."""
    with pytest.raises(ValueError):
        gen_polygonal_tiling(square, cast(PolygonType, "pentagon"), EDGE)


# --------------------------------------------------------------------------
# Geometric consistency
# --------------------------------------------------------------------------


def test_hexagon_step_matches_reference_geometry() -> None:
    """Hexagon columns are spaced by one and a half edge lengths."""
    tiling, _ = gen_hexagonal_tiling(shapely.box(0, 0, 12, 12), 1.0)
    xs = np.unique(np.round(extract_tiling_centers(tiling)[:, 0], 6))
    assert np.allclose(np.diff(xs), 1.5)


def test_default_anchor_is_the_world_origin() -> None:
    """Without an alignment point a tile centre sits on the origin."""
    surface = shapely.box(-5.0, -5.0, 5.0, 5.0)
    for generator in GENERATORS.values():
        tiling, _ = generator(surface, EDGE)
        centroids = np.array([next(iter(p.centroid.coords)) for p in tiling.geoms])
        assert np.abs(centroids).sum(axis=1).min() < 1e-9


def test_resolve_anchor_accepts_a_point() -> None:
    """An explicit alignment point overrides the default anchor."""
    assert np.array_equal(_resolve_anchor(None), DEFAULT_ALIGNMENT_POINT)
    assert np.array_equal(_resolve_anchor((3.0, 4.0)), [3.0, 4.0])


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_tilings_of_nested_surfaces_conform(
    lshape: shapely.Polygon, poly_type: PolygonType
) -> None:
    """A sub-region reuses exactly the tiles of the enclosing surface."""
    sub_region = lshape.intersection(shapely.box(1.0, 1.0, 4.0, 9.0))
    whole, _ = GENERATORS[poly_type](lshape, EDGE)
    part, _ = GENERATORS[poly_type](sub_region, EDGE)
    assert {g.wkt for g in part.geoms} <= {g.wkt for g in whole.geoms}


@pytest.mark.parametrize("poly_type", list(PolygonType))
def test_clipping_the_surface_moves_no_tile(
    lshape: shapely.Polygon, poly_type: PolygonType
) -> None:
    """Editing the surface re-selects tiles but never re-phases them."""
    clipped = lshape.difference(shapely.box(11.4, 3.2, 12.0, 4.0))
    whole, _ = GENERATORS[poly_type](lshape, EDGE)
    edited, _ = GENERATORS[poly_type](clipped, EDGE)
    assert {g.wkt for g in edited.geoms} <= {g.wkt for g in whole.geoms}


def test_triangle_height_follows_anisotropy() -> None:
    """Triangle height scales with the anisotropy ratio."""
    tiling, _ = gen_triangular_tiling(shapely.box(0, 0, 6, 6), 1.0, 2.0)
    heights = [p.bounds[3] - p.bounds[1] for p in tiling.geoms]
    assert np.allclose(heights, SQRT3)
