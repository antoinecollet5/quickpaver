"""Time the tiling generators on a box and on the France + Corsica contour."""

import time

import shapely

import quickpaver as qp

fr = qp.load_france_and_corsica_contour()
x0, y0, x1, y1 = fr.bounds
cases = [
    ("box 600x600", shapely.box(0, 0, 600, 600), 1.0),
    ("France+Corsica", fr, (x1 - x0) / 700),
]
for name, surf, el in cases:
    for pt in ["rectangle", "hexagon", "triangle"]:
        args = (surf, qp.PolygonType(pt), el)
        best = float("inf")
        for _ in range(2):
            t = time.perf_counter()
            _, adj = qp.gen_polygonal_tiling(*args, rot_deg=13)
            best = min(best, time.perf_counter() - t)
        t = time.perf_counter()
        qp.gen_polygonal_tiling_arrays(*args, rot_deg=13)
        arrays = time.perf_counter() - t
        print(
            f"{name:15s} {pt:9s} {len(adj):8d} tiles  shapely {best * 1e3:6.0f} ms"
            f"  arrays {arrays * 1e3:6.0f} ms"
        )
