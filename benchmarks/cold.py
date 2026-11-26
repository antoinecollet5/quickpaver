"""Import + first-call latency."""

import time

t0 = time.perf_counter()
import quickpaver as qp  # noqa: E402
from quickpaver._tm_test_utils import make_grid_trimesh  # noqa: E402

t1 = time.perf_counter()
a = qp.RectilinearGrid(nx=50, ny=50)
b = qp.RectilinearGrid(nx=40, ny=40, dx=1.3, dy=1.3, theta=20)
m = make_grid_trimesh(-20, -20, 1, 1, 40, 40)
m2 = make_grid_trimesh(-15, -15, 0.9, 0.9, 40, 40)
qp.compute_transfer_matrix(a, b)
qp.compute_transfer_matrix(a, m)
qp.compute_transfer_matrix(m, m2)
qp.compute_transfer_matrix(m, b.to_shapely())
t2 = time.perf_counter()
print(f"import {1e3 * (t1 - t0):5.0f} ms, first calls {1e3 * (t2 - t1):5.0f} ms")
