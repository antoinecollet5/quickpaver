# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2026 Antoine COLLET

from importlib import resources

import shapely
import shapely.affinity


def load_france_contour() -> shapely.Polygon:
    with resources.files("quickpaver.data").joinpath("france.geojson").open("rb") as f:
        return shapely.affinity.scale(shapely.from_geojson(f.read()), 1.0, 1.2)


def load_corsica_contour() -> shapely.Polygon:
    with resources.files("quickpaver.data").joinpath("corsica.geojson").open("rb") as f:
        return shapely.from_geojson(f.read())


def load_france_and_corsica_contour() -> shapely.Polygon:
    france = load_france_contour()
    corsica = shapely.affinity.translate(load_corsica_contour(), 5000, -5000)

    return shapely.MultiPolygon([corsica, france])
