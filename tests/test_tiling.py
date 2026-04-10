import pytest
import shapely
from quickpaver import (
    PolygonType,
    extract_tiling_centers,
    extract_tiling_vertices,
    gen_polygon,
    gen_polygonal_tiling,
)


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE, PolygonType.TRIANGLE),
)
@pytest.mark.parametrize(
    "anisotropy_ratio",
    (1.0, 2.0, 0.5),
)
def test_gen_polygon(poly_type: PolygonType, anisotropy_ratio: float) -> None:
    gen_polygon(
        PolygonType.HEXAGON, edge_length=30.0, anisotropy_ratio=anisotropy_ratio
    )


def test_gen_polygon_wrong_type():
    with pytest.raises(ValueError, match=""):
        gen_polygon("wrong_type")


@pytest.mark.parametrize(
    "poly_type",
    (PolygonType.HEXAGON, PolygonType.RECTANGLE, PolygonType.TRIANGLE),
)
@pytest.mark.parametrize(
    "anisotropy_ratio",
    (1.0, 2.0, 0.5),
)
@pytest.mark.parametrize(
    "edge_length",
    (100.0, 200.0),
)
def test_gen_polygonal_tiling(
    poly_type: PolygonType, anisotropy_ratio: float, edge_length: float
) -> None:

    tiles, _ = gen_polygonal_tiling(
        surface_to_cover=shapely.Point((0.0, 0.0)).buffer(200.0)
        - shapely.Point((0.0, 0.0)).buffer(100.0),
        poly_type=poly_type,
        edge_length=10.0,
        anisotropy_ratio=anisotropy_ratio,
    )

    # MultiPolygon
    _ = extract_tiling_centers(tiles)
    _, _, _ = extract_tiling_vertices(tiles)

    # Iterable of polygons
    _ = extract_tiling_centers(tiles.geoms)
    _, _, _ = extract_tiling_vertices(tiles.geoms)


def test_gen_polygonal_tiling_wrong_polytype() -> None:
    with pytest.raises(
        ValueError, match=r"'not_a_valid_type' is not a valid PolygonType"
    ):
        gen_polygonal_tiling(
            surface_to_cover=shapely.Point((0.0, 0.0)).buffer(200.0)
            - shapely.Point((0.0, 0.0)).buffer(100.0),
            poly_type=PolygonType("not_a_valid_type"),
            edge_length=10.0,
            anisotropy_ratio=1.0,
        )
