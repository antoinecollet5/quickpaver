"""Time every transfer-matrix path (best of ``--repeat`` runs).

usage: python benchmarks/run_bench.py [--scale S] [--only NAME]
                                       [--intersections | --vertices] [--profile]
"""

import argparse
import cProfile
import io
import json
import pstats
import resource
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from scenarios import scenarios  # noqa: E402

import quickpaver as qp  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--scale", type=float, default=1.0)
p.add_argument("--profile", action="store_true")
p.add_argument("--only", default=None)
mode = p.add_mutually_exclusive_group()
mode.add_argument("--intersections", action="store_true")
mode.add_argument("--vertices", action="store_true")
p.add_argument("--repeat", type=int, default=3)
p.add_argument("--json", default=None)
a = p.parse_args()

if a.vertices:
    fn = qp.compute_transfer_matrix_with_intersection_vertices
elif a.intersections:
    fn = qp.compute_transfer_matrix_with_intersections
else:
    fn = qp.compute_transfer_matrix

print("quickpaver", qp.__version__)
results = {}
for name, (s, t) in scenarios(a.scale).items():
    if a.only and a.only not in name:
        continue
    out = fn(s, t)  # warm-up
    best = float("inf")
    for _ in range(a.repeat):
        t0 = time.perf_counter()
        out = fn(s, t)
        best = min(best, time.perf_counter() - t0)
    W = out[0] if isinstance(out, tuple) else out
    results[name] = {"seconds": best, "nnz": int(W.nnz), "sum": float(W.sum())}
    print(f"{name:24s} {best * 1e3:9.1f} ms  nnz={W.nnz:>9d}  sum={W.sum():.10f}")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        fn(s, t)
        pr.disable()
        buf = io.StringIO()
        pstats.Stats(pr, stream=buf).sort_stats("tottime").print_stats(12)
        print("\n".join(buf.getvalue().splitlines()[6:22]))
if a.json:
    Path(a.json).write_text(json.dumps(results, indent=1))
peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
print(f"peak RSS: {peak:.0f} MB")
