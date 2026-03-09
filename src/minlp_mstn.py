"""
-------------------------------------------------------
Variables
    yx[v], yy[v]  continuous — representative coordinates of vertex v inside N_v
    x[e] ∈ {0,1} binary     — 1 iff edge e is in the spanning tree
    u[e]  ≥ 0    continuous — Euclidean length  ‖y_i − y_j‖
    θ[e]  ≥ 0    continuous — McCormick product  u[e] · x[e]
    dx[e], dy[e] continuous — coordinate differences  y_ix−y_jx, y_iy−y_jy
    f[(i,j)] ≥ 0 continuous — directed flow (single-commodity spanning-tree)

Objective
    min  Σ_{e∈E} θ[e]

(A) Neighbourhood membership
    Square  →  box bounds on yx[v], yy[v]
    Circle  →  box bounds  +  SOC  (yx[v]−cx)²+(yy[v]−cy)² ≤ r²

(B) Coordinate-difference linking
    dx[e] = yx[i] − yx[j],   dy[e] = yy[i] − yy[j]   ∀ e = {i,j}

(C) Norm  (addGenConstrNorm, Gurobi ≥ 9.0)
    u[e] = ‖(dx[e], dy[e])‖₂                          ∀ e ∈ E

(D) McCormick linearisation of  θ[e] = u[e] · x[e]
    θ[e] ≤ Ũ_e · x[e]                                 (U1)
    θ[e] ≥ u[e] − Ũ_e · (1 − x[e])                    (U2)
    θ[e] ≤ u[e]                                       (U3)
    where  Ũ_e = dist(center_i, center_j) + r_i + r_j

(E) Spanning tree via single-commodity flow
    Σ_e x[e] = n − 1
    f[(i,j)] ≤ (n−1) · x[{i,j}]   ∀ directed arc
    f[(j,i)] ≤ (n−1) · x[{i,j}]   ∀ directed arc
    outflow(root) − inflow(root) = n − 1
    inflow(v) − outflow(v) = 1     ∀ v ≠ root
"""

import math
import gurobipy as gp
import tempfile
import pathlib
import os


from typing import Optional
from gurobipy import GRB

from geometry import Point, SquareNeighborhood, CircleNeighborhood
from graph import GeoGraph
from mst import Kruskal_MST
    

def _max_radius(nb) -> float:
    """
    Conservative bound on the furthest distance from center to any point inside nb.
    Half-diagonal for squares, radius for circles.
    """
    if isinstance(nb, SquareNeighborhood):
        return nb.side / 2.0 * math.sqrt(2.0)
    return nb.radius


# ─────────────────────────────────────────────────────────────────────────────
# Solver
# ─────────────────────────────────────────────────────────────────────────────

def _build_model(graph: GeoGraph, verbose: bool = True):
    """
    Build and return the Gurobi RL-MSTN model and all variable dicts.
    Used internally by both Solve_MINLP and LP_relaxation_bound.

    Returns
    -------
    m, x, yx, yy, u, theta, dx, dy, f, edges, U, original_cost
    """
    n     = graph.N
    edges = graph.edges

    if n < 2:
        raise ValueError("Graph must have at least 2 vertices.")
    if len(edges) < n - 1:
        raise ValueError(
            f"Graph has only {len(edges)} edges but needs at least {n-1}. "
            "Call G.Complete_graph() first."
        )

    centers = [nb.center for nb in graph.neighborhoods]
    _, original_cost = Kruskal_MST(centers, edges)

    def _upper_bound(i, j):
        nb_i, nb_j = graph.neighborhoods[i], graph.neighborhoods[j]
        d = math.hypot(nb_i.center.x - nb_j.center.x,
                       nb_i.center.y - nb_j.center.y)
        return d + _max_radius(nb_i) + _max_radius(nb_j)

    U = {e: _upper_bound(e[0], e[1]) for e in edges}

    m = gp.Model("RL_MSTN")
    m.setParam("OutputFlag", int(verbose))
    m.setParam("NonConvex", 2)

    # (1) Representatives
    yx, yy = {}, {}
    for v, nb in enumerate(graph.neighborhoods):
        cx, cy = nb.center.x, nb.center.y
        if isinstance(nb, SquareNeighborhood):
            h = nb.side / 2.0
            yx[v] = m.addVar(lb=cx - h, ub=cx + h, name=f"yx[{v}]")
            yy[v] = m.addVar(lb=cy - h, ub=cy + h, name=f"yy[{v}]")
        else:
            r = nb.radius
            yx[v] = m.addVar(lb=cx - r, ub=cx + r, name=f"yx[{v}]")
            yy[v] = m.addVar(lb=cy - r, ub=cy + r, name=f"yy[{v}]")

    # (2) Edge-selection  — type set later by the caller (BINARY or CONTINUOUS)
    x     = {e: m.addVar(lb=0.0, ub=1.0, name=f"x[{e}]")               for e in edges}
    u     = {e: m.addVar(lb=0.0, ub=U[e], name=f"u[{e}]")              for e in edges}
    theta = {e: m.addVar(lb=0.0, ub=U[e], name=f"theta[{e}]")          for e in edges}
    dx    = {e: m.addVar(lb=-U[e], ub=U[e], name=f"dx[{e}]")           for e in edges}
    dy    = {e: m.addVar(lb=-U[e], ub=U[e], name=f"dy[{e}]")           for e in edges}

    f = {}
    for (i, j) in edges:
        f[(i, j)] = m.addVar(lb=0.0, ub=n - 1, name=f"f[{i},{j}]")
        f[(j, i)] = m.addVar(lb=0.0, ub=n - 1, name=f"f[{j},{i}]")

    m.update()

    m.setObjective(gp.quicksum(theta[e] for e in edges), GRB.MINIMIZE)

    # (A) Circle membership
    for v, nb in enumerate(graph.neighborhoods):
        if isinstance(nb, CircleNeighborhood):
            cx, cy, r = nb.center.x, nb.center.y, nb.radius
            m.addQConstr(
                (yx[v] - cx) * (yx[v] - cx) + (yy[v] - cy) * (yy[v] - cy) <= r * r,
                name=f"circle[{v}]",
            )

    # (B) Coordinate differences
    for (i, j) in edges:
        e = (i, j)
        m.addConstr(dx[e] == yx[i] - yx[j], name=f"dx_link[{e}]")
        m.addConstr(dy[e] == yy[i] - yy[j], name=f"dy_link[{e}]")

    # (C) Euclidean norm
    for e in edges:
        m.addGenConstrNorm(u[e], [dx[e], dy[e]], 2.0, name=f"norm[{e}]")

    # (D) McCormick
    for e in edges:
        Ue = U[e]
        m.addConstr(theta[e] <= Ue * x[e],              name=f"U1[{e}]")
        m.addConstr(theta[e] >= u[e] - Ue * (1 - x[e]), name=f"U2[{e}]")
        m.addConstr(theta[e] <= u[e],                    name=f"U3[{e}]")

    # (E) Spanning tree SCF
    m.addConstr(gp.quicksum(x[e] for e in edges) == n - 1, name="tree_size")

    adj = {v: [] for v in range(n)}
    for (i, j) in edges:
        adj[i].append(j)
        adj[j].append(i)

    root = 0
    for (i, j) in edges:
        m.addConstr(f[(i, j)] <= (n - 1) * x[(i, j)], name=f"fcap[{i},{j}]")
        m.addConstr(f[(j, i)] <= (n - 1) * x[(i, j)], name=f"fcap[{j},{i}]")

    for v in range(n):
        out_flow = gp.quicksum(f[(v, w)] for w in adj[v])
        in_flow  = gp.quicksum(f[(w, v)] for w in adj[v])
        if v == root:
            m.addConstr(out_flow - in_flow == n - 1, name="flow[root]")
        else:
            m.addConstr(in_flow - out_flow == 1,     name=f"flow[{v}]")

    return m, x, yx, yy, u, theta, dx, dy, f, edges, U, original_cost


def Solve_MINLP(
    graph: GeoGraph,
    time_limit: Optional[float] = 300.0,
    threads: Optional[int] = None,
    mip_focus: Optional[int] = None,
    verbose: bool = True,
) -> tuple:
    """
    Solve the full RL-MSTN MINLP with binary x[e].

    Returns
    -------
    positions, mst, cost, original_cost, gurobi_stats
    """
    m, x, yx, yy, u, theta, dx, dy, f, edges, U, original_cost = \
        _build_model(graph, verbose)

    for e in edges:
        x[e].setAttr("VType", GRB.BINARY)
    m.update()

    if time_limit is not None:
        m.setParam("TimeLimit", float(time_limit))
    if threads is not None:
        m.setParam("Threads", int(threads))
    if mip_focus is not None:
        m.setParam("MIPFocus", int(mip_focus))

    log_fd, log_path = tempfile.mkstemp(suffix=".log", prefix="gurobi_")
    os.close(log_fd)

    try:
        m.setParam("LogFile", log_path)
        m.optimize()
        gurobi_log = pathlib.Path(log_path).read_text(encoding="utf-8", errors="replace")
    finally:
        try:
            os.unlink(log_path)
        except OSError:
            pass

    if m.SolCount == 0:
        raise RuntimeError(
            f"Gurobi found no feasible solution "
            f"(status={m.Status}, runtime={m.Runtime:.1f}s)."
        )

    gurobi_stats = {
        "status": m.Status,
        "obj_val": float(m.ObjVal),
        "obj_bound": float(m.ObjBound),
        "mip_gap": float(m.MIPGap) if m.IsMIP else None,
        "runtime_s": float(m.Runtime),
        "node_count": int(m.NodeCount) if m.IsMIP else None,
        "sol_count": int(m.SolCount),
        "log": gurobi_log,
        "n": graph.N,
    }

    n = graph.N
    positions = [Point(float(yx[v].X), float(yy[v].X)) for v in range(n)]
    selected = [(i, j) for (i, j) in edges if x[(i, j)].X > 0.5]
    mst = [
        (i, j, math.hypot(positions[i].x - positions[j].x,
                          positions[i].y - positions[j].y))
        for (i, j) in selected
    ]
    cost = sum(w for _, _, w in mst)

    return positions, mst, cost, original_cost, gurobi_stats



def LP_relaxation_bound(
    graph: GeoGraph,
    time_limit: float = 30.0,
    verbose: bool = False,
    verbose_bound: bool = True,   # prints each time obj_bound improves
) -> dict:

    m, x, yx, yy, u, theta, dx, dy, f, edges, U, original_cost = \
        _build_model(graph, verbose)

    for e in edges:
        x[e].setAttr("VType", GRB.BINARY)
    m.update()

    m.setParam("TimeLimit",   time_limit)
    m.setParam("MIPFocus",    3)
    m.setParam("Cuts",        2)
    m.setParam("Heuristics",  0)

    # Callback: print whenever the lower bound improves
    best_bound = [0.0]   # list so it is mutable inside the closure

    def _bound_callback(model, where):
        if where == GRB.Callback.MIP:
            current_bound = model.cbGet(GRB.Callback.MIP_OBJBND)
            if current_bound > best_bound[0] + 1e-6:
                best_bound[0] = current_bound
                runtime       = model.cbGet(GRB.Callback.RUNTIME)
                node_count    = int(model.cbGet(GRB.Callback.MIP_NODCNT))
                print(f"  [{runtime:6.1f}s]  nodes: {node_count:6d}  "
                      f"obj_bound: {current_bound:.6f}")

    if verbose_bound:
        print(f"Computing lower bound  (time_limit={time_limit}s) ...")
        print(f"  {'time':>8}   {'nodes':>8}   obj_bound")
        print(f"  {'-'*45}")
        m.optimize(_bound_callback)
    else:
        m.optimize()

    lp_bound = float(m.ObjBound)

    if verbose_bound:
        print(f"  {'-'*45}")
        print(f"  Final bound : {lp_bound:.6f}  "
              f"(runtime={m.Runtime:.1f}s, status={m.Status})")

    return {
        "lp_bound":      lp_bound,
        "original_cost": original_cost,
        "runtime_s":     float(m.Runtime),
        "status":        m.Status,
    }



if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from random_graph import Create_random_graph
    from heuristic import Heuristic_MSTN_alternating
    from visualize import Draw_solution_centered, Draw_graph_centered
    from instance_io import load_instance

    # G = Create_random_graph(
    #     n=30, m=0,
    #     shape="square",
    #     integer_size=True,
    #     size_min=1, size_max=8,
    #     grid_size=100,
    # )
    # G.Complete_graph()

    (G,
    pos_h, mst_h, cost_h, original_cost, k,
        pos_g, mst_g, cost_g,
        gurobi_stats,
        extra_params,) = load_instance("../Graphs_data/1.json")

    fig1, ax1 = Draw_graph_centered(G)



    # # Draw final solution
    # pos, mst, cost, original_cost, k = Heuristic_MSTN_alternating(G)
    # fig2, ax2 = Draw_solution_centered(G, pos, mst, original_cost)

    # print(f"Solving MINLP on {G.N} neighborhoods, {len(G.edges)} edges …")
    # positions, mst, cost, original_cost = Solve_MINLP(G, time_limit=120.0)

    # print(f"  MST at centers  : {original_cost:.4f}")
    # print(f"  MINLP optimum   : {cost:.4f}")
    # print(f"  Improvement     : {100*(original_cost-cost)/original_cost:.1f}%")

    # fig3, ax3 = Draw_solution_centered(G, positions, mst, original_cost)

    lb = LP_relaxation_bound(G, time_limit = 60, verbose_bound = False)
    print(f"LP lower bound   : {lb['lp_bound']:.4f}")

    # Then compare everything
    pos_h, mst_h, cost_h, orig, k = Heuristic_MSTN_alternating(G)
    print(cost_h)
    gap_vs_lp = 100 * (cost_h - lb["lp_bound"]) / lb["lp_bound"]
    print(f"Heuristic gap vs LP bound : {gap_vs_lp:+.2f}%")

    plt.show()
