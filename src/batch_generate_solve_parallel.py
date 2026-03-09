import csv
import pathlib
import time
import random
import numpy as np
from multiprocessing import Pool

import random_graph
from heuristic import Heuristic_MSTN_alternating
from minlp_mstn import Solve_MINLP
from instance_io import save_instance


OUTPUT_DIR = pathlib.Path("../Graphs_data_batch")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CSV_PATH = OUTPUT_DIR / "summary.csv"

# ── Overnight settings ────────────────────────────────────────────────────────
N_INSTANCES = 100

# Possible numbers of neighborhoods for each instance
N_CHOICES = [10, 12, 14, 15]

GRID_SIZE = 50
SIZE_MIN = 2
SIZE_MAX = 8

WORKERS = 4
THREADS_PER_JOB = 16
TIME_LIMIT = 1800         # 30 minutes per instance
MIP_FOCUS = 3

BASE_SEED = 1000


def fmt_seconds(s):
    s = int(round(s))
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    if h > 0:
        return "%dh%02dm%02ds" % (h, m, sec)
    if m > 0:
        return "%dm%02ds" % (m, sec)
    return "%ds" % sec


def safe_pct(x):
    if x is None:
        return float("nan")
    return 100.0 * x


def run_one(job):
    idx, seed = job

    # Choose n for this instance
    n = random.choice(N_CHOICES)

    random.seed(seed)
    np.random.seed(seed)

    G = random_graph.Create_random_graph(
        n=n,
        m=0,
        shape="square",
        integer_size=True,
        size_min=SIZE_MIN,
        size_max=SIZE_MAX,
        overlap_fraction=0.0,
        overlap_degree=0.0,
        grid_size=GRID_SIZE,
    )
    G.Complete_graph()

    t0 = time.perf_counter()
    pos_h, mst_h, cost_h, original_cost, hstats = Heuristic_MSTN_alternating(G)
    time_heuristic = time.perf_counter() - t0

    t0 = time.perfcounter()
    pos_g, mst_g, cost_g, _, gurobi_stats = Solve_MINLP(
        G,
        time_limit=TIME_LIMIT,
        threads=THREADS_PER_JOB,
        mip_focus=MIP_FOCUS,
        verbose=False,
    )
    time_gurobi = time.perf_counter() - t0

    out_path = OUTPUT_DIR / ("inst_%03d.json" % idx)

    save_instance(
        out_path,
        G,
        pos_h, mst_h, cost_h, original_cost, hstats,
        pos_g, mst_g, cost_g,
        gurobi_stats=gurobi_stats,
        extra_params=dict(
            n=n,
            m=0,
            shape="square",
            integer_size=True,
            size_min=SIZE_MIN,
            size_max=SIZE_MAX,
            overlap_fraction=0.0,
            overlap_degree=0.0,
            grid_size=GRID_SIZE,
            seed=seed,
            workers=WORKERS,
            threads=THREADS_PER_JOB,
            time_limit=TIME_LIMIT,
            mip_focus=MIP_FOCUS,
            time_heuristic_s=round(time_heuristic, 4),
            time_gurobi_s=round(time_gurobi, 4),
        ),
    )

    heur_gap_pct = (
        100.0 * (cost_h - cost_g) / cost_g
        if cost_g not in (None, 0.0)
        else float("nan")
    )

    return {
        "idx": idx,
        "seed": seed,
        "n": n,
        "file": out_path.name,
        "status": gurobi_stats["status"],
        "obj_val": gurobi_stats["obj_val"],
        "obj_bound": gurobi_stats["obj_bound"],
        "mip_gap": gurobi_stats["mip_gap"],
        "mip_gap_pct": safe_pct(gurobi_stats["mip_gap"]),
        "runtime_s": gurobi_stats["runtime_s"],
        "node_count": gurobi_stats["node_count"],
        "sol_count": gurobi_stats["sol_count"],
        "heur_cost": cost_h,
        "gurobi_cost": cost_g,
        "heur_gap_vs_gurobi_pct": heur_gap_pct,
        "optimal": int(gurobi_stats["status"] == 2),
    }


def main():
    jobs = [(i + 1, BASE_SEED + i) for i in range(N_INSTANCES)]
    rows = []
    start = time.perf_counter()

    with Pool(processes=WORKERS) as pool:
        for k, row in enumerate(pool.imap_unordered(run_one, jobs), start=1):
            rows.append(row)

            elapsed = time.perf_counter() - start
            avg_wall_per_finished = elapsed / float(k)
            remaining = N_INSTANCES - k
            eta = avg_wall_per_finished * remaining
            n_opt = sum(r["optimal"] for r in rows)

            print(
                "[%d/%d] %s | n=%d | %s | obj=%.3f | bound=%.3f | gap=%.2f%% | runtime=%s | elapsed=%s | ETA=%s | optimal_so_far=%d/%d"
                % (
                    k,
                    N_INSTANCES,
                    row["file"],
                    row["n"],
                    "OPTIMAL" if row["optimal"] else "NOT_PROVEN",
                    row["obj_val"],
                    row["obj_bound"],
                    row["mip_gap_pct"],
                    fmt_seconds(row["runtime_s"]),
                    fmt_seconds(elapsed),
                    fmt_seconds(eta),
                    n_opt,
                    k,
                )
            )

    rows.sort(key=lambda r: r["idx"])

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    total = time.perf_counter() - start
    n_opt = sum(r["optimal"] for r in rows)

    print("\n=== DONE ===")
    print("Total elapsed: %s" % fmt_seconds(total))
    print("Optimal instances: %d/%d" % (n_opt, N_INSTANCES))
    print("Summary CSV: %s" % CSV_PATH)


if __name__ == "__main__":
    main()
