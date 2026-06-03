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
    extract_tiling_centers,
    extract_tiling_vertices,
    gen_hexagonal_tiling,
    gen_polygon,
    gen_polygonal_tiling,
    gen_rectangular_tiling,
    gen_triangular_tiling,
)

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
