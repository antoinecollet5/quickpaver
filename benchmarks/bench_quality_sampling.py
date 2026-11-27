"""Time ``multipolygon_quality``, ``poisson_disk_sampling`` and the VTK mesh helpers
(best of 3 runs).

usage: python benchmarks/bench_quality_sampling.py [--scale S]
"""

import argparse
import time

import numpy as np
import shapely

import quickpaver as qp

p = argparse.ArgumentParser()
p.add_argument("--scale", type=float, default=1.0)
a = p.parse_args()


def best(fn, repeat=3):
    fn()
    t = float("inf")
    for _ in range(repeat):
        t0 = time.perf_counter()
        fn()
        t = min(t, time.perf_counter() - t0)
    return t


print("quickpaver", qp.__version__)

# --- cell quality -----------------------------------------------------------
n = max(10, int(300_000 * a.scale))
ang = np.linspace(0, 2 * np.pi, 7)[:-1]
c = np.random.default_rng(0).uniform(0, 3000, (n, 2))
rings = np.stack([c[:, None, 0] + 1.1 * np.cos(ang), c[:, None, 1] + np.sin(ang)], -1)
polys = shapely.polygons(rings)
mp = shapely.MultiPolygon(list(polys))
for name, arg in [
    ("quality MultiPolygon", mp),
    ("quality Polygon array", polys),
    ("quality (n, 6, 2) array", rings),
]:
    t = best(lambda: qp.multipolygon_quality(arg))
    print(f"{name:28s} n={n:8d} {t * 1e3:8.1f} ms")

# --- Poisson disk sampling --------------------------------------------------
h, w = max(10, int(2000 * a.scale**0.5)), max(10, int(3000 * a.scale**0.5))
xx = np.arange(w)[None, :] * np.ones((h, 1))
fields = {
    "uniform r=1": np.full((h, w), 1.0),
    "uniform r=3": np.full((h, w), 3.0),
    "gradient 2..20": 2.0 + 18.0 * xx / (w - 1),
    "step 1.5 / 40": np.where(xx < w // 2, 1.5, 40.0),
}
for name, r in fields.items():
    r = r.astype(np.float32)
    t = best(lambda: qp.poisson_disk_sampling(r, seed=1))
    npts = qp.poisson_disk_sampling(r, seed=1)[0]
    print(
        f"pds {name:24s} n={npts:8d} {t * 1e3:8.1f} ms {t / npts * 1e9:7.0f} ns/point"
    )

# --- VTK meshes (needs pyvista) ---------------------------------------------
try:
    import pyvista  # noqa: F401
except ImportError:
    print("pyvista not installed: mesh benchmarks skipped")
else:
    n_seeds = max(10, int(100_000 * a.scale))
    seeds = np.random.default_rng(0).uniform(0, 1e5, (n_seeds, 2))
    box = shapely.box(0, 0, 1e5, 1e5)
    diagram = shapely.voronoi_polygons(shapely.MultiPoint(seeds), extend_to=box)
    cells = shapely.intersection(shapely.get_parts(diagram), box)
    t = best(lambda: qp.shapely_polygons_to_vtu(cells))
    print(f"{'mesh shapely_polygons_to_vtu':28s} n={len(cells):8d} {t * 1e3:8.1f} ms")
    surf = qp.shapely_polygons_to_vtu(cells)
    t = best(lambda: qp.pv_extrude(surf, [1.0] * 10))
    print(
        f"{'mesh pv_extrude (10 layers)':28s} n={10 * len(cells):8d} {t * 1e3:8.1f} ms"
    )
