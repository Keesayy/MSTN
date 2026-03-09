from heuristic import Heuristic_MSTN_alternating
from minlp_mstn import Solve_MINLP
from instance_io improt save_instance

import random_graph
import pathlib
import time
import matplotlib.pyplot as plt

OUTPUT_DIR = pathlib.Path("../Graphs_data")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

size = 50
n    = 12
smin = 2
smax = 8

for i in range(1):
    print(f"\n── Instance {i} ──────────────────────────────")

    G = random_graph.Create_random_graph(
        n=n, m=0,
        shape="square", integer_size=True,
        size_min=smin, size_max=smax,
        overlap_fraction=0.0, overlap_degree=0.0,
        grid_size=size,
    )
    G.Complete_graph()

    t0 = time.perf_counter()
    pos_h, mst_h, cost_h, original_cost, k = Heuristic_MSTN_alternating(G)
    time_heuristic = time.perf_counter() - t0

    t0 = time.perf_counter()
    print(f"Solving MINLP on {G.N} neighborhoods, {len(G.edges)} edges …")
    pos_g, mst_g, cost_g, _, gurobi_stats = Solve_MINLP(G, time_limit=1e9)
    time_gurobi = time.perf_counter() - t0

    optimal = gurobi_stats["status"] == 2
    gap_pct = 100.0 * (cost_h - cost_g) / cost_g if cost_g > 0 else float("nan")
    print(f"  {'✓ OPTIMAL' if optimal else '⚠ NOT PROVEN'}  "
          f"H={cost_h:.2f}  G={cost_g:.2f}  gap={gap_pct:+.2f}%  "
          f"time={time_gurobi:.1f}s")

    save_instance(
        OUTPUT_DIR / f"{i}.json", G,
        pos_h, mst_h, cost_h, original_cost, k,
        pos_g, mst_g, cost_g,
        gurobi_stats=gurobi_stats,
        extra_params=dict(
            n=n, m=0, shape="square", integer_size=True,
            size_min=smin, size_max=smax, grid_size=size,
            time_heuristic_s=round(time_heuristic, 4),
            time_gurobi_s=round(time_gurobi, 4),
        ),
    )
