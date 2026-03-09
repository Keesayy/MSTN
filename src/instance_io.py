"""
instance_io.py
--------------
Save and reload a complete MSTN experiment to/from a JSON file.

Serialises
    - extra_params  : free-form dict of generation parameters (n, smin, …)
    - graph         : exact GeoGraph (neighborhoods + edges + grid_size)
    - heuristic     : pos_h, mst_h, cost_h, original_cost, k
    - gurobi        : pos_g, mst_g, cost_g, stats, full solver log

Usage
-----
    from instance_io import save_instance, load_instance

    # ── Save ────────────────────────────────────────────────────────────────
    save_instance(
        "my_instance",
        G,
        pos_h, mst_h, cost_h, original_cost, k,
        pos_g, mst_g, cost_g,
        gurobi_stats=gurobi_stats,
        extra_params=dict(n=n, m=0, shape="square", integer_size=True,
                          size_min=smin, size_max=smax, grid_size=size),
    )

    # ── Load ────────────────────────────────────────────────────────────────
    (G, pos_h, mst_h, cost_h, original_cost, k,
     pos_g, mst_g, cost_g, gurobi_stats, extra_params) = load_instance("my_instance")
"""

import json
import pathlib

from geometry import Point, SquareNeighborhood, CircleNeighborhood
from graph import GeoGraph


# ─────────────────────────────────────────────────────────────────────────────
# Internal serialisation helpers
# ─────────────────────────────────────────────────────────────────────────────

def _nb_to_dict(nb) -> dict:
    if isinstance(nb, SquareNeighborhood):
        return {"type": "square",
                "cx": nb.center.x, "cy": nb.center.y,
                "side": nb.side}
    elif isinstance(nb, CircleNeighborhood):
        return {"type": "circle",
                "cx": nb.center.x, "cy": nb.center.y,
                "radius": nb.radius}
    raise TypeError(f"Unknown neighborhood type: {type(nb)}")


def _dict_to_nb(d: dict):
    center = Point(d["cx"], d["cy"])
    if d["type"] == "square":
        return SquareNeighborhood(center, d["side"])
    elif d["type"] == "circle":
        return CircleNeighborhood(center, d["radius"])
    raise ValueError(f"Unknown neighborhood type in JSON: '{d['type']}'")


def _positions_to_list(positions) -> list:
    return [[p.x, p.y] for p in positions]


def _list_to_positions(lst: list) -> list:
    return [Point(float(xy[0]), float(xy[1])) for xy in lst]


def _mst_to_list(mst) -> list:
    return [[int(u), int(v), float(w)] for u, v, w in mst]


def _list_to_mst(lst: list) -> list:
    return [(int(row[0]), int(row[1]), float(row[2])) for row in lst]


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def save_instance(
    filepath: str,
    graph: GeoGraph,
    pos_h,
    mst_h,
    cost_h: float,
    original_cost: float,
    k: int,
    pos_g,
    mst_g,
    cost_g: float,
    gurobi_stats: dict = None,
    extra_params: dict = None,
    indent: int = 2,
) -> pathlib.Path:
    """
    Save a complete MSTN instance and both solver results to a JSON file.

    Parameters
    ----------
    filepath      : path (with or without .json extension)
    graph         : GeoGraph instance
    pos_h         : list[Point]    heuristic representatives
    mst_h         : list[(u,v,w)]  heuristic spanning tree
    cost_h        : float          heuristic MSTN cost
    original_cost : float          MST cost at neighborhood centers
    k             : int            heuristic iterations until convergence
    pos_g         : list[Point]    Gurobi representatives
    mst_g         : list[(u,v,w)]  Gurobi spanning tree
    cost_g        : float          Gurobi MSTN cost
    gurobi_stats  : dict | None    returned by Solve_MINLP —
                      keys: status, obj_val, obj_bound, mip_gap,
                            runtime_s, node_count, sol_count, log
    extra_params  : dict | None    any free-form metadata you want to keep
                      e.g. dict(n=20, smin=2, smax=8, grid_size=100, shape="square")

    Returns
    -------
    pathlib.Path — the file that was written
    """
    path = pathlib.Path(filepath).with_suffix(".json")

    data = {
        "extra_params": extra_params or {},
        "graph": {
            "grid_size":     graph.grid_size,
            "neighborhoods": [_nb_to_dict(nb) for nb in graph.neighborhoods],
            "edges":         [[int(u), int(v)] for u, v in graph.edges],
        },
        "heuristic": {
            "positions":     _positions_to_list(pos_h),
            "mst":           _mst_to_list(mst_h),
            "cost":          float(cost_h),
            "original_cost": float(original_cost),
            "k":             int(k),
        },
        "gurobi": {
            "positions": _positions_to_list(pos_g),
            "mst":       _mst_to_list(mst_g),
            "cost":      float(cost_g),
            "stats": {
                key: val
                for key, val in (gurobi_stats or {}).items()
                if key != "log"
            },
            "log": (gurobi_stats or {}).get("log", ""),
        },
    }

    path.write_text(json.dumps(data, indent=indent), encoding="utf-8")
    print(
        f"[instance_io] Saved → {path}  "
        f"({path.stat().st_size / 1024:.1f} kB, "
        f"n={graph.N}, |E|={len(graph.edges)})"
    )
    return path


def load_instance(filepath: str) -> tuple:
    """
    Reload a previously saved MSTN instance from a JSON file.

    Returns
    -------
    (G, pos_h, mst_h, cost_h, original_cost, k,
     pos_g, mst_g, cost_g, gurobi_stats, extra_params)

    G             : GeoGraph
    pos_h         : list[Point]
    mst_h         : list[(u, v, w)]
    cost_h        : float
    original_cost : float
    k             : int
    pos_g         : list[Point]
    mst_g         : list[(u, v, w)]
    cost_g        : float
    gurobi_stats  : dict — status, obj_val, obj_bound, mip_gap,
                           runtime_s, node_count, sol_count, log
    extra_params  : dict — whatever was passed at save time
    """
    path = pathlib.Path(filepath).with_suffix(".json")
    if not path.exists():
        raise FileNotFoundError(f"[instance_io] File not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    # ── GeoGraph ──────────────────────────────────────────────────────────────
    gd            = data["graph"]
    neighborhoods = [_dict_to_nb(d) for d in gd["neighborhoods"]]
    G             = GeoGraph(neighborhoods, grid_size=gd["grid_size"])
    for u, v in gd["edges"]:
        G.Add_edge(int(u), int(v))

    # ── Heuristic ─────────────────────────────────────────────────────────────
    hd            = data["heuristic"]
    pos_h         = _list_to_positions(hd["positions"])
    mst_h         = _list_to_mst(hd["mst"])
    cost_h        = float(hd["cost"])
    original_cost = float(hd["original_cost"])
    k             = int(hd["k"])

    # ── Gurobi ────────────────────────────────────────────────────────────────
    grd          = data["gurobi"]
    pos_g        = _list_to_positions(grd["positions"])
    mst_g        = _list_to_mst(grd["mst"])
    cost_g       = float(grd["cost"])
    gurobi_stats = {**grd.get("stats", {}), "log": grd.get("log", "")}

    # ── Extra params ──────────────────────────────────────────────────────────
    extra_params = data.get("extra_params", {})

    # ── Summary ───────────────────────────────────────────────────────────────
    gap_pct = 100.0 * (cost_h - cost_g) / cost_g if cost_g > 0 else float("nan")
    mip_gap = gurobi_stats.get("mip_gap")
    runtime = gurobi_stats.get("runtime_s")

    print(f"[instance_io] Loaded : {path}\n")
    print_dict(extra_params)
    print()
    print(
        f"n={G.N}\n|E|={len(G.edges)}\n"
        f"MST={original_cost:.2f}\n"
        f"Heuristic={cost_h:.2f}\n"
        f"Gurobi={cost_g:.2f}\n"
        f"gap={gap_pct:+.2f}%\n"
        f"MIP gap={f'{mip_gap:.4%}' if mip_gap is not None else 'n/a'}\n"
        f"runtime={f'{runtime:.1f}s' if runtime is not None else 'n/a'}\n"
    )

    return (
        G,
        pos_h, mst_h, cost_h, original_cost, k,
        pos_g, mst_g, cost_g,
        gurobi_stats,
        extra_params,
    )

def print_dict(extra_params: dict):
    if not extra_params:
        print("No extra_params")
        return
    
    for key, value in sorted(extra_params.items()):
        print(f"{key}: {value}")
