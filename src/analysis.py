"""
AMD EPYC 7452 — 64 physical cores, 512 GB RAM
"""

import csv
import json
import math
import pathlib
import time
import random as _random
from dataclasses import dataclass, asdict
from multiprocessing import Pool
from typing import List

import numpy as np

import random_graph
from geometry import Point, SquareNeighborhood, CircleNeighborhood
from graph import GeoGraph
from heuristic import Heuristic_MSTN_alternating


N_WORKERS       = 64
WALL_TIME_LIMIT = 9.5 * 3600
OUTPUT_DIR      = pathlib.Path("batch_data")
BASE_SEED       = 42

SMIN_PERCENT    = 0.03
SMAX_PERCENT    = 0.10
ITER_SAVE_EVERY = 5

N_SCHEDULE = [
    (10,    1000, 10000),
    (20,    1000, 10000),
    (30,    1000, 10000),
    (50,    1000, 10000),
    (75,    1000, 10000),
    (100,   1000, 10000),
    (150,   1000, 10000),
    (200,   1000, 10000),
    (300,   1000, 10000),
    (500,   1000, 10000),
    (750,   1000, 10000),
    (1000,  1000, 10000),
]

HEURISTIC_CONFIGS = [
    dict(strategy="multistart", alpha=1.0,  n_random=5),
    dict(strategy="multistart", alpha=0.7,  n_random=5),
    dict(strategy="multistart", alpha=0.5,  n_random=5),
    dict(strategy="multistart", alpha=0.3,  n_random=5),
    dict(strategy="best",       alpha=1.0,  n_random=5),
    # dict(strategy="random",     alpha=1.0,  n_random=5),
    dict(strategy="multistart", alpha=1.0,  n_random=0),
    dict(strategy="multistart", alpha=1.0,  n_random=2),
    dict(strategy="multistart", alpha=1.0,  n_random=10),
]


def _schedule_params(n: int, d_target: float = 14.0):
    grid_size = max(50, round(d_target * math.sqrt(n)))
    d_avg     = grid_size / math.sqrt(n)
    smin      = max(1, round(SMIN_PERCENT * grid_size))
    smax      = max(smin + 1, min(round(SMAX_PERCENT * grid_size), int(0.45 * d_avg)))
    return grid_size, smin, smax


def _config_key(cfg: dict) -> str:
    return f"{cfg['strategy']}_a{cfg['alpha']}_r{cfg['n_random']}"


def _classify_node(nb, pos: Point, tol: float = 1e-6):
    d = float(np.linalg.norm(pos.To_numpy() - nb.center.To_numpy()))
    if isinstance(nb, SquareNeighborhood):
        half = nb.side / 2
        dx   = abs(pos.x - nb.center.x)
        dy   = abs(pos.y - nb.center.y)
        on_border = abs(dx - half) < tol or abs(dy - half) < tol
        r = half
    elif isinstance(nb, CircleNeighborhood):
        on_border = abs(d - nb.radius) < tol
        r = nb.radius
    else:
        return "center", 0.0
    norm_d = d / r if r > 0 else 0.0
    if d < tol:
        return "center", norm_d
    if on_border:
        return "boundary", norm_d
    return "interior", norm_d


@dataclass
class IterRecord:
    instance_id    : int
    config_key     : str
    strategy       : str
    alpha          : float
    n_random       : int
    n              : int
    iter           : int
    cost           : float
    n_center       : int
    n_interior     : int
    n_boundary     : int
    frac_boundary  : float
    frac_interior  : float
    frac_center    : float
    mean_norm_dist : float
    max_norm_dist  : float


@dataclass
class SummaryRecord:
    instance_id          : int
    seed                 : int
    n                    : int
    grid_size            : int
    smin                 : int
    smax                 : int
    config_key           : str
    strategy             : str
    alpha                : float
    n_random             : int
    original_cost        : float
    final_cost           : float
    reduction_pct        : float
    n_iters              : int
    time_s               : float
    best_start           : str
    final_frac_boundary  : float
    final_frac_interior  : float
    final_mean_norm_dist : float


def _make_iter_tracker(graph: GeoGraph, instance_id: int, cfg: dict, n: int):
    records = []
    key     = _config_key(cfg)
    state   = {"last_record": None}

    def on_iter(k: int, positions: List[Point], mst: list, cost: float):
        counts     = {"center": 0, "interior": 0, "boundary": 0}
        norm_dists = []
        for nb, pos in zip(graph.neighborhoods, positions):
            label, nd = _classify_node(nb, pos)
            counts[label] += 1
            norm_dists.append(nd)

        rec = IterRecord(
            instance_id    = instance_id,
            config_key     = key,
            strategy       = cfg["strategy"],
            alpha          = cfg["alpha"],
            n_random       = cfg["n_random"],
            n              = n,
            iter           = k,
            cost           = cost,
            n_center       = counts["center"],
            n_interior     = counts["interior"],
            n_boundary     = counts["boundary"],
            frac_boundary  = counts["boundary"] / n,
            frac_interior  = counts["interior"] / n,
            frac_center    = counts["center"]   / n,
            mean_norm_dist = float(np.mean(norm_dists)),
            max_norm_dist  = float(np.max(norm_dists)),
        )
        state["last_record"] = rec

        if k % ITER_SAVE_EVERY == 0:
            records.append(rec)

    def flush_last():
        last = state["last_record"]
        if last is not None and (not records or records[-1].iter != last.iter):
            records.append(last)

    return on_iter, flush_last, records


def _run_job(job: dict):
    n         = job["n"]
    iid       = job["instance_id"]
    seed      = job["seed"]
    max_iter  = job["max_iter"]
    grid_size = job["grid_size"]
    smin      = job["smin"]
    smax      = job["smax"]
    configs   = job["configs"]

    _random.seed(seed)
    np.random.seed(seed)

    G = random_graph.Create_random_graph(
        n                = n,
        m                = 0,
        shape            = "square",
        integer_size     = True,
        size_min         = smin,
        size_max         = smax,
        overlap_fraction = 0.0,
        overlap_degree   = 0.0,
        grid_size        = grid_size,
    )
    G.Complete_graph()

    iter_records    = []
    summary_records = []

    for cfg in configs:
        callback, flush_last, records = _make_iter_tracker(G, iid, cfg, n)

        t0 = time.perf_counter()
        result = Heuristic_MSTN_alternating(
            G,
            max_iter = max_iter,
            tol      = 1e-2,
            strategy = cfg["strategy"],
            alpha    = cfg["alpha"],
            n_random = cfg["n_random"],
            seed     = seed,
            on_iter  = callback,
        )
        elapsed = time.perf_counter() - t0

        flush_last()

        _, _, final_cost, original_cost, n_iters = result[:5]
        best_start    = result[5] if len(result) > 5 else ""
        last          = records[-1] if records else None
        reduction_pct = (
            100.0 * (original_cost - final_cost) / original_cost
            if original_cost > 0 else 0.0
        )

        iter_records.extend(records)
        summary_records.append(SummaryRecord(
            instance_id          = iid,
            seed                 = seed,
            n                    = n,
            grid_size            = grid_size,
            smin                 = smin,
            smax                 = smax,
            config_key           = _config_key(cfg),
            strategy             = cfg["strategy"],
            alpha                = cfg["alpha"],
            n_random             = cfg["n_random"],
            original_cost        = original_cost,
            final_cost           = final_cost,
            reduction_pct        = reduction_pct,
            n_iters              = n_iters + 1,
            time_s               = round(elapsed, 4),
            best_start           = best_start,
            final_frac_boundary  = last.frac_boundary    if last else 0.0,
            final_frac_interior  = last.frac_interior    if last else 0.0,
            final_mean_norm_dist = last.mean_norm_dist   if last else 0.0,
        ))

    return iter_records, summary_records


def _write_csv(path: pathlib.Path, records: list, mode: str = "w"):
    if not records:
        return
    fields = list(asdict(records[0]).keys())
    with open(path, mode, newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if mode == "w":
            w.writeheader()
        for r in records:
            w.writerow(asdict(r))


def _fmt(s: float) -> str:
    s = int(round(s))
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h}h{m:02d}m{sec:02d}s" if h else (f"{m}m{sec:02d}s" if m else f"{sec}s")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    summary_path = OUTPUT_DIR / "summary.csv"
    iters_path   = OUTPUT_DIR / "iterations.csv"
    meta_path    = OUTPUT_DIR / "instances.json"

    done = set()
    if summary_path.exists():
        with open(summary_path) as f:
            for row in csv.DictReader(f):
                done.add((int(row["instance_id"]), row["config_key"]))
        print(f"Checkpoint: {len(done)} (instance, config) pairs already done.")

    rng           = np.random.default_rng(BASE_SEED)
    all_jobs      = []
    instance_meta = []
    instance_id   = 1

    for n, n_instances, max_iter in N_SCHEDULE:
        grid_size, smin, smax = _schedule_params(n)

        for _ in range(n_instances):
            seed = int(rng.integers(0, 2**31))

            configs_to_run = [
                cfg for cfg in HEURISTIC_CONFIGS
                if (instance_id, _config_key(cfg)) not in done
            ]

            if configs_to_run:
                all_jobs.append(dict(
                    n           = n,
                    instance_id = instance_id,
                    seed        = seed,
                    max_iter    = max_iter,
                    grid_size   = grid_size,
                    smin        = smin,
                    smax        = smax,
                    configs     = configs_to_run,
                ))
                instance_meta.append(dict(
                    instance_id = instance_id,
                    seed        = seed,
                    n           = n,
                    grid_size   = grid_size,
                    smin        = smin,
                    smax        = smax,
                    max_iter    = max_iter,
                ))

            instance_id += 1

    meta_path.write_text(json.dumps(instance_meta, indent=2))
    print(f"Jobs      : {len(all_jobs)}")
    print(f"Workers   : {N_WORKERS}")
    print(f"Output    : {OUTPUT_DIR.resolve()}\n")

    headers_written = {"summary": summary_path.exists(), "iters": iters_path.exists()}
    start     = time.perf_counter()
    completed = 0
    total     = len(all_jobs)

    all_jobs.sort(key=lambda j: j["n"])

    with Pool(processes=N_WORKERS) as pool:
        for iter_recs, sum_recs in pool.imap_unordered(_run_job, all_jobs):
            elapsed = time.perf_counter() - start

            if elapsed >= WALL_TIME_LIMIT:
                print(f"\nWall time limit reached ({_fmt(WALL_TIME_LIMIT)}) — stopping.")
                pool.terminate()
                break

            _write_csv(summary_path, sum_recs, mode="a" if headers_written["summary"] else "w")
            _write_csv(iters_path,   iter_recs, mode="a" if headers_written["iters"]   else "w")
            headers_written["summary"] = True
            headers_written["iters"]   = True

            completed += 1
            eta = (elapsed / completed) * (total - completed)
            r   = sum_recs[0]
            print(
                f"[{completed:>5}/{total}]  n={r.n:<4}  "
                f"cost={r.final_cost:.2f}  -{r.reduction_pct:.1f}%  "
                f"elapsed={_fmt(elapsed)}  ETA={_fmt(eta)}"
            )

    print(f"\nDone. {completed}/{total} jobs in {_fmt(time.perf_counter() - start)}.")
    print(f"  {summary_path}")
    print(f"  {iters_path}")


if __name__ == "__main__":
    main()
