from typing import List, Tuple, Optional, Callable
from geometry import Point, SquareNeighborhood, CircleNeighborhood
from mst import Kruskal_MST, Kruskal_MST_metric
from visualize import Draw_graph, Draw_solution, Draw_graph_centered, Draw_solution_centered, Draw_solution_centered_compare
from graph import GeoGraph


from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import os
import numpy as np

# ── distance metrics ──────────────────────────────────────────────────────────

def _euclidean(p: Point, q: Point) -> float:
    return float(np.linalg.norm(p.To_numpy() - q.To_numpy()))

def _manhattan(p: Point, q: Point) -> float:
    return abs(p.x - q.x) + abs(p.y - q.y)

def _chebyshev(p: Point, q: Point) -> float:
    return max(abs(p.x - q.x), abs(p.y - q.y))

DEFAULT_METRICS = [_euclidean, _manhattan, _chebyshev]

# ── shared building blocks ────────────────────────────────────────────────────

def _project_step(
    graph: GeoGraph,
    positions: List[Point],
    mst: List[Tuple[int, int, float]],
) -> Tuple[List[Point], float]:
    """Project each representative toward its MST neighbors' centroid."""
    neighbors = [[] for _ in range(graph.N)]
    for u, v, _ in mst:
        neighbors[u].append(v)
        neighbors[v].append(u)

    new_positions = positions.copy()
    max_move = 0.0
    for i in range(graph.N):
        if neighbors[i]:
            arr = np.vstack([positions[j].To_numpy() for j in neighbors[i]])
            cx, cy = arr.mean(axis=0)
            projected = graph.neighborhoods[i].Project(Point(cx, cy))
            move = np.linalg.norm(projected.To_numpy() - positions[i].To_numpy())
            max_move = max(max_move, move)
            new_positions[i] = projected
    return new_positions, max_move


def _random_positions(graph: GeoGraph, rng: np.random.Generator) -> List[Point]:
    """Uniformly random representative inside each neighborhood."""
    positions = []
    for nb in graph.neighborhoods:
        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            x = rng.uniform(nb.center.x - half, nb.center.x + half)
            y = rng.uniform(nb.center.y - half, nb.center.y + half)
            positions.append(Point(x, y))
        elif isinstance(nb, CircleNeighborhood):
            r = nb.radius * np.sqrt(rng.uniform(0, 1))
            theta = rng.uniform(0, 2 * np.pi)
            x = nb.center.x + r * np.cos(theta)
            y = nb.center.y + r * np.sin(theta)
            positions.append(Point(x, y))
        else:
            positions.append(nb.center)
    return positions


def _metric_initial_positions(
    graph: GeoGraph,
    positions: List[Point],
    metric: Callable,
) -> List[Point]:
    """One projection step from an MST built under a given metric."""
    mst, _ = Kruskal_MST_metric(positions, graph.edges, metric)
    projected, _ = _project_step(graph, positions, mst)
    return projected


# ── adaptive building blocks ──────────────────────────────────────────────────

def _project_step_adaptive(
    graph: GeoGraph,
    positions: List[Point],
    mst: List[Tuple[int, int, float]],
    alpha: float = 0.5,
) -> Tuple[List[Point], float]:
    """
    Partial-step projection: y_i^{k+1} = y_i^k + alpha * (z_i^k - y_i^k)
    where z_i^k is the projection of the MST-neighbors' centroid onto
    neighborhood i.  alpha=1.0 recovers the classic full step.
    Feasibility is guaranteed by convexity of each neighborhood.
    """
    neighbors = [[] for _ in range(graph.N)]
    for u, v, _ in mst:
        neighbors[u].append(v)
        neighbors[v].append(u)

    new_positions = positions.copy()
    max_move = 0.0
    for i in range(graph.N):
        if neighbors[i]:
            arr  = np.vstack([positions[j].To_numpy() for j in neighbors[i]])
            cx, cy = arr.mean(axis=0)
            zi = graph.neighborhoods[i].Project(Point(cx, cy))   # projected barycenter
            yi = positions[i].To_numpy()
            new_pos = yi + alpha * (zi.To_numpy() - yi)          # partial step
            new_point = Point(float(new_pos[0]), float(new_pos[1]))
            move = np.linalg.norm(new_pos - yi)
            max_move = max(max_move, move)
            new_positions[i] = new_point
    return new_positions, max_move


# ── _run_adaptive : ajout du paramètre on_iter ────────────────────────────────

def _run_adaptive(
    graph: GeoGraph,
    initial_positions: List[Point],
    max_iter: int,
    tol: float,
    alpha: float = 1.0,
    on_iter: Optional[Callable] = None,  # ← NOUVEAU
) -> Tuple[List[Point], list, float, int]:
    """MST → partial-step project loop. alpha=1.0 ≡ classic full step."""
    positions = initial_positions
    k = 0
    for k in range(max_iter):
        mst, _ = Kruskal_MST(positions, graph.edges)
        positions, max_move = _project_step_adaptive(graph, positions, mst, alpha=alpha)
        if on_iter is not None:
            mst_k, cost_k = Kruskal_MST(positions, graph.edges)
            on_iter(k, positions, mst_k, cost_k)
        if max_move < tol:
            break
    mst, cost = Kruskal_MST(positions, graph.edges)
    return positions, mst, cost, k


def Heuristic_MSTN_alternating(
    graph: GeoGraph,
    max_iter: int = 100,
    tol: float = 1e-2,
    initial_positions: Optional[List[Point]] = None,
    metrics: Optional[List[Callable]] = None,
    strategy: str = "best",      # "best" | "cycle" | "random" | "multistart"
    n_random: int = 5,
    seed: int = 67,
    alpha: float = 1.0,          # ← NOUVEAU  (1.0 = comportement classique)
    on_iter: Optional[Callable] = None,  # ← NOUVEAU  signature: (k, positions, mst, cost)
    verbose: bool = False,
):
    """
    Alternating MSTN heuristic.

    alpha : float in (0, 1]
        Step size for the projection update:
            y_i^{k+1} = y_i^k + alpha * (proj(centroid) - y_i^k)
        alpha=1.0  →  classic full-step (former Heuristic_MSTN_alternating)
        alpha<1.0  →  damped step      (former Heuristic_MSTN_adaptive)

    on_iter : callable | None
        Called at every iteration with signature:
            on_iter(k: int, positions: List[Point],
                    mst: list, cost: float)
        For strategy="multistart", called on the best start's trajectory
        (clean re-run after the selection pass — zero overhead on others).
    """
    rng = np.random.default_rng(seed)
    metrics = metrics or DEFAULT_METRICS
    centers = [nb.center for nb in graph.neighborhoods]
    positions = initial_positions if initial_positions is not None else centers

    _, original_cost = Kruskal_MST(centers, graph.edges)

    # ── multistart ────────────────────────────────────────────────────────────
    if strategy == "multistart":
        starts = [
            ("euclidean_centers", centers),
            *[(f"metric_{m.__name__}", _metric_initial_positions(graph, centers, m))
              for m in metrics if m is not _euclidean],
            *[(f"random_{i}", _random_positions(graph, rng))
              for i in range(n_random)],
        ]

        best_cost = float("inf")
        best_result = None
        best_init_pos = None

        # Pass 1 — find best start (no callback, fast)
        for label, init_pos in starts:
            pos, mst, cost, iters = _run_adaptive(
                graph, init_pos, max_iter, tol, alpha=alpha
            )
            if verbose:
                print(f"  [{label:<25}] cost={cost:.4f}  iters={iters+1}")
            if cost < best_cost:
                best_cost = cost
                best_result = (pos, mst, cost, original_cost, iters, label)
                best_init_pos = init_pos

        if verbose:
            print(f"\n  Best: {best_result[-1]} → cost={best_result[2]:.4f}")

        # Pass 2 — replay best start with callback (only if needed)
        if on_iter is not None:
            _run_adaptive(
                graph, best_init_pos, max_iter, tol,
                alpha=alpha, on_iter=on_iter,
            )

        return best_result  # (positions, mst, cost, original_cost, iters, label)

    # ── per-iteration strategies : best / cycle / random ─────────────────────
    for k in range(max_iter):
        if strategy == "best":
            active_metrics = metrics
        elif strategy == "cycle":
            active_metrics = [metrics[k % len(metrics)]]
        elif strategy == "random":
            active_metrics = [rng.choice(metrics)]
        else:
            raise ValueError(f"Unknown strategy '{strategy}'")

        best_candidate_cost = float("inf")
        best_candidate_pos = None
        best_max_move = 0.0

        for metric in active_metrics:
            mst_candidate, _ = Kruskal_MST_metric(positions, graph.edges, metric)
            candidate_pos, max_move = _project_step_adaptive(
                graph, positions, mst_candidate, alpha=alpha   # ← alpha ici aussi
            )
            _, candidate_cost = Kruskal_MST(candidate_pos, graph.edges)
            if candidate_cost < best_candidate_cost:
                best_candidate_cost = candidate_cost
                best_candidate_pos = candidate_pos
                best_max_move = max_move

        for _ in range(n_random):
            rand_pos = _random_positions(graph, rng)
            mst_rand, _ = Kruskal_MST(rand_pos, graph.edges)
            candidate_pos, max_move = _project_step_adaptive(
                graph, rand_pos, mst_rand, alpha=alpha         # ← alpha ici aussi
            )
            _, candidate_cost = Kruskal_MST(candidate_pos, graph.edges)
            if candidate_cost < best_candidate_cost:
                best_candidate_cost = candidate_cost
                best_candidate_pos = candidate_pos
                best_max_move = max_move

        positions = best_candidate_pos

        # callback
        if on_iter is not None:
            mst_k, cost_k = Kruskal_MST(positions, graph.edges)
            on_iter(k, positions, mst_k, cost_k)

        if best_max_move < tol:
            break

    mst, cost = Kruskal_MST(positions, graph.edges)
    return positions, mst, cost, original_cost, k


# Heuristic_MSTN_adaptive : deprecated wrapper

def Heuristic_MSTN_adaptive(graph, alpha=0.5, **kwargs):
    """Deprecated — use Heuristic_MSTN_alternating(strategy='multistart', alpha=alpha)."""
    import warnings
    warnings.warn(
        "Heuristic_MSTN_adaptive is deprecated. "
        "Use Heuristic_MSTN_alternating(strategy='multistart', alpha=alpha).",
        DeprecationWarning, stacklevel=2,
    )
    return Heuristic_MSTN_alternating(strategy="multistart", alpha=alpha, **kwargs)



def Heuristic_Grid(
    graph,
    max_iter:         int   = 16,
    strategy:         str   = 'best',
    metrics                 = None,
    n_random:         int   = 2,
    seed:             int   = 42,
    multistart_start: int   = 0,
    alpha:            float = 1.0,   # step size — only used by 'adaptive'
    save_path:        str   = None,
    log_path:         str   = None,
):
    rng     = np.random.default_rng(seed)
    metrics = metrics or DEFAULT_METRICS
    centers = [nb.center for nb in graph.neighborhoods]
    log_lines = []

    # ── initial positions ─────────────────────────────────────────────────────
    if strategy in ('multistart', 'adaptive'):
        starts = [
            ("Euclidean centers", centers),
            *[(f"Metric: {m.__name__}", _metric_initial_positions(graph, centers, m))
              for m in metrics if m is not _euclidean],
            *[(f"Random #{i}", _random_positions(graph, rng))
              for i in range(n_random)],
        ]
        start_label, positions = starts[multistart_start]
        alpha_tag = f" (α={alpha})" if strategy == 'adaptive' else ""
        log_lines.append(f"Strategy   : {strategy}{alpha_tag}")
        log_lines.append(f"Start index: {multistart_start} — {start_label}")
        log_lines.append(f"{'─'*45}")
        log_lines.append("Available starts (initial Euclidean MST cost):")
        for idx, (lbl, pts) in enumerate(starts):
            _, c  = Kruskal_MST(pts, graph.edges)
            marker = " ◄ selected" if idx == multistart_start else ""
            log_lines.append(f"  [{idx}] {lbl:<30} init_cost={c:.4f}{marker}")
        log_lines.append(f"{'─'*45}")
    else:
        positions   = centers
        start_label = None
        log_lines.append(f"Strategy: {strategy}")
        log_lines.append(f"{'─'*45}")

    # ── figure setup ──────────────────────────────────────────────────────────
    n_rows = int(np.ceil(np.sqrt(max_iter)))
    n_cols = int(np.ceil(max_iter / n_rows))
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(5.5 * n_cols, 6.0 * n_rows),
        gridspec_kw={'hspace': 0.55, 'wspace': 0.35},
    )
    axes = axes.flatten() if max_iter > 1 else [axes]

    mst0, cost0 = Kruskal_MST(positions, graph.edges)
    alpha_tag   = f" (α={alpha})" if strategy == 'adaptive' else ""
    fig.suptitle(
        f"MSTN Heuristic Iterations — strategy: {strategy}{alpha_tag}", fontsize=15
    )

    def _edge_set(mst):
        return {(min(u, v), max(u, v)) for u, v, _ in mst}

    prev_edge_set = _edge_set(mst0)

    for frame in range(max_iter):
        ax = axes[frame]
        ax.set_xlim(0, graph.grid_size)
        ax.set_ylim(0, graph.grid_size)
        ax.set_xticks(range(0, graph.grid_size + 1), minor=True)
        ax.set_yticks(range(0, graph.grid_size + 1), minor=True)
        ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
        ax.grid(which='major', linewidth=1,   linestyle='-', color='lightgray')
        ax.set_axisbelow(True)
        ax.set_aspect('equal')

        moved_nodes   = set()
        changed_edges = set()
        chosen_label  = None

        # ── frame 0: initial state ────────────────────────────────────────────
        if frame == 0:
            mst, cost = mst0, cost0
            chosen_label = "Initial (Euclidean centers)"
            log_lines.append(f"[frame 0] Initial cost = {cost:.4f}")

        # ── multistart / adaptive: partial-step projection ────────────────────
        elif strategy in ('multistart', 'adaptive'):
            mst_for_proj, _ = Kruskal_MST(positions, graph.edges)
            new_positions, _ = _project_step_adaptive(
                graph, positions, mst_for_proj, alpha=alpha
            )
            for i in range(graph.N):
                move = np.linalg.norm(
                    new_positions[i].To_numpy() - positions[i].To_numpy()
                )
                if move >= 1e-2:
                    moved_nodes.add(i)
            positions = new_positions
            mst, cost = Kruskal_MST(positions, graph.edges)
            log_lines.append(
                f"[frame {frame}] cost={cost:.4f}  moved_nodes={sorted(moved_nodes)}"
            )

        # ── per-iteration strategies ──────────────────────────────────────────
        else:
            if strategy == 'best':
                active_metrics = metrics
            elif strategy == 'cycle':
                active_metrics = [metrics[(frame - 1) % len(metrics)]]
            elif strategy == 'random':
                active_metrics = [rng.choice(metrics)]
            else:
                raise ValueError(f"Unknown strategy '{strategy}'")

            best_candidate_cost = float("inf")
            best_candidate_pos  = None

            for metric in active_metrics:
                mst_candidate, _ = Kruskal_MST_metric(positions, graph.edges, metric)
                candidate_pos, _ = _project_step(graph, positions, mst_candidate)
                _, candidate_cost = Kruskal_MST(candidate_pos, graph.edges)
                if candidate_cost < best_candidate_cost:
                    best_candidate_cost = candidate_cost
                    best_candidate_pos  = candidate_pos
                    chosen_label = f"Metric: {metric.__name__}"

            for r in range(n_random):
                rand_pos = _random_positions(graph, rng)
                mst_rand, _ = Kruskal_MST(rand_pos, graph.edges)
                candidate_pos, _ = _project_step(graph, rand_pos, mst_rand)
                _, candidate_cost = Kruskal_MST(candidate_pos, graph.edges)
                if candidate_cost < best_candidate_cost:
                    best_candidate_cost = candidate_cost
                    best_candidate_pos  = candidate_pos
                    chosen_label = f"Random #{r}"

            positions     = best_candidate_pos
            mst, cost     = Kruskal_MST(positions, graph.edges)
            changed_edges = _edge_set(mst).symmetric_difference(prev_edge_set)
            prev_edge_set = _edge_set(mst)
            log_lines.append(
                f"[frame {frame}] cost={cost:.4f}  "
                f"chosen={chosen_label}  changed_edges={len(changed_edges)}"
            )

        # ── draw neighborhoods ────────────────────────────────────────────────
        for nb in graph.neighborhoods:
            cx, cy = nb.center.x, nb.center.y
            if isinstance(nb, SquareNeighborhood):
                half = nb.side / 2
                ax.add_patch(patches.Rectangle(
                    (cx - half, cy - half), nb.side, nb.side,
                    fill=False, edgecolor='blue', linewidth=1.3))
            elif isinstance(nb, CircleNeighborhood):
                ax.add_patch(patches.Circle(
                    (cx, cy), nb.radius,
                    fill=False, edgecolor='blue', linewidth=1.3))

        # ── draw solution points ──────────────────────────────────────────────
        for i, p in enumerate(positions):
            ax.scatter(p.x, p.y, s=60, color='green')
            ax.text(p.x + 0.1, p.y + 0.1, str(i), fontsize=9, color='black')

        # ── draw MST edges ────────────────────────────────────────────────────
        for u, v, w in mst:
            p, q = positions[u], positions[v]
            if strategy in ('multistart', 'adaptive'):
                changed = u in moved_nodes or v in moved_nodes
            else:
                edge    = (min(u, v), max(u, v))
                changed = edge in changed_edges
            ax.plot([p.x, q.x], [p.y, q.y],
                    color='yellow' if changed else 'red',
                    lw=3 if changed else 2, alpha=0.8)

        # ── info box below x-axis ─────────────────────────────────────────────
        if strategy in ('multistart', 'adaptive'):
            alpha_str = f" | α={alpha}" if strategy == 'adaptive' else ""
            text_box  = f"Iter: {frame} | Weight: {cost:.2f}{alpha_str}"
        else:
            text_box  = f"Iter: {frame} | Weight: {cost:.2f}\nChosen: {chosen_label}"

        ax.text(
            0.5, -0.18, text_box,
            transform=ax.transAxes, fontsize=8, ha='center', va='top',
            bbox=dict(facecolor='white', edgecolor='black', boxstyle='round,pad=0.3'),
        )

    # ── summary log lines ─────────────────────────────────────────────────────
    log_lines.append(f"{'─'*45}")
    log_lines.append(f"Final cost : {cost:.4f}")
    log_lines.append(f"Iterations : {max_iter}")

    # ── legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        Line2D([0], [0], marker='o', color='green', markersize=8,
               linestyle='None', label='Solution points'),
    ]
    if strategy in ('multistart', 'adaptive'):
        legend_elements += [
            Line2D([0], [0], color='red', lw=2,
                   label=f'MST edge | Start: {start_label}'),
            Line2D([0], [0], color='yellow', lw=3,
                   label='Edge touching a moved node'),
        ]
    else:
        legend_elements += [
            Line2D([0], [0], color='red',    lw=2, label='Unchanged MST edge'),
            Line2D([0], [0], color='yellow', lw=3, label='Changed MST edge'),
        ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)

    for i in range(max_iter, len(axes)):
        axes[i].axis('off')
    plt.subplots_adjust(top=0.93)

    # ── save / show ───────────────────────────────────────────────────────────
    if save_path is not None:
        fig.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close(fig)
    else:
        plt.show()

    if log_path is not None:
        with open(log_path, 'w') as f:
            f.write('\n'.join(log_lines) + '\n')

    return log_lines


def Heuristic_Animate(
    graph,
    max_iter: int = 100,
    tol: float = 1e-4,
    interval: int = 200
):

    positions = [nb.center for nb in graph.neighborhoods]
    _, original_cost = Kruskal_MST(positions, graph.edges)

    fig, ax = plt.subplots(figsize=(10,10))
    g = graph.grid_size
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_xticks(range(0, g + 1), minor=True)
    ax.set_yticks(range(0, g + 1), minor=True)
    ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
    ax.grid(which='major', linewidth=1, linestyle='-', color='lightgray')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # Draw neighborhoods
    text_labels = []
    for i, nb in enumerate(graph.neighborhoods):
        cx, cy = nb.center.x, nb.center.y
        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            rect = patches.Rectangle((cx-half, cy-half), nb.side, nb.side,
                                     fill=False, edgecolor='blue', linewidth=1.5)
            ax.add_patch(rect)
        elif isinstance(nb, CircleNeighborhood):
            circ = patches.Circle((cx, cy), nb.radius,
                                  fill=False, edgecolor='blue', linewidth=1.5)
            ax.add_patch(circ)
        txt = ax.text(cx+0.1, cy+0.1, str(i), fontsize=12, color='black')
        text_labels.append(txt)

    # Scatter for solution points
    pts_scatter = ax.scatter([p.x for p in positions],
                             [p.y for p in positions],
                             color='green', s=70)

    # MST edges as line objects
    lines = [ax.plot([0,0], [0,0], color='red', lw=2)[0] for _ in range(graph.N-1)]

    prev_positions = positions.copy()  # track previous positions to detect movement

    def update(frame):
        nonlocal positions, prev_positions

        # Compute MST before moving (to know neighbors)
        mst, _ = Kruskal_MST(positions, graph.edges)
        neighbors = [[] for _ in range(graph.N)]
        for u, v, _ in mst:
            neighbors[u].append(v)
            neighbors[v].append(u)

        # Move positions toward neighbors
        new_positions = positions.copy()
        max_move = 0.0
        moved_nodes = set()
        for i in range(graph.N):
            if neighbors[i]:
                arr = np.vstack([positions[j].To_numpy() for j in neighbors[i]])
                cx, cy = arr.mean(axis=0)
                projected = graph.neighborhoods[i].Project(Point(cx, cy))
                move = np.linalg.norm(projected.To_numpy() - positions[i].To_numpy())
                if move > 1e-2:
                    moved_nodes.add(i)
                max_move = max(max_move, move)
                new_positions[i] = projected
        positions = new_positions

        # Recompute MST with updated positions
        mst, cost = Kruskal_MST(positions, graph.edges)

        # Update scatter
        pts_scatter.set_offsets(np.array([[p.x, p.y] for p in positions]))

        # Update text labels
        for i, txt in enumerate(text_labels):
            txt.set_position((positions[i].x + 0.1, positions[i].y + 0.1))

        # Update MST lines with moving edges in yellow
        for idx, (u, v, w) in enumerate(mst):
            p, q = positions[u], positions[v]
            lines[idx].set_data([p.x, q.x], [p.y, q.y])
            if u in moved_nodes or v in moved_nodes:
                lines[idx].set_color('yellow')
                lines[idx].set_linewidth(3)
            else:
                lines[idx].set_color('red')
                lines[idx].set_linewidth(2)

        ax.set_title(f"MSTN Heuristic Animation\nIteration {frame+1}, "
                     f"Max move: {max_move:.4f}, MSTN weight: {cost:.2f}, Total MST: {original_cost:.2f}")
        legend_elements = [
            Line2D([0], [0], color='red', lw=2, label='MST Edge'),
            Line2D([0], [0], marker='o', color='green', markersize=8,
                   linestyle='None', label='Solution Points'),
            Line2D([0], [0], color='yellow', lw=3, label='Moving Edge')
        ]
        ax.legend(handles=legend_elements, loc="upper left")
        ax.set_aspect('equal')

        prev_positions = positions.copy()
        return pts_scatter, *lines, *text_labels

    anim = FuncAnimation(fig, update, frames=max_iter, interval=interval, blit=False, repeat=False)
    plt.show()
    return anim


def Save_experiment(
    graph:              GeoGraph,
    output_dir:         str,
    strategies:         list  = None,
    max_iter:           int   = 16,
    metrics                   = None,
    n_random:           int   = 2,
    seed:               int   = 42,
    multistart_start          = 0,
    heuristic_max_iter: int   = 100,
    heuristic_tol:      float = 1e-2,
    alpha:              float = 0.5,   # step size used when 'adaptive' is in strategies
):
    os.makedirs(output_dir, exist_ok=True)
    strategies = strategies or ['best', 'cycle', 'random', 'multistart', 'adaptive']
    log_lines  = ["MSTN Experiment Summary", f"{'='*55}"]

    # ── 1. save graph ─────────────────────────────────────────────────────────
    fig_graph, _ = Draw_graph(graph)
    fig_graph.savefig(os.path.join(output_dir, 'graph.png'), bbox_inches='tight', dpi=150)
    plt.close(fig_graph)
    print("Saved: graph.png")

    # ── 2. run each strategy, collect cost + iters ────────────────────────────
    best_cost                             = float("inf")
    best_positions, best_mst, best_original_cost = None, None, None
    centers = [nb.center for nb in graph.neighborhoods]

    log_lines.append(f"\n{'Method':<28} {'Cost':>10} {'Iterations':>12}  Note")
    log_lines.append(f"{'─'*55}")

    # classic baseline
    c_pos, c_mst, c_cost, c_iters = _run_classic(
        graph, centers, heuristic_max_iter, heuristic_tol
    )
    log_lines.append(f"{'classic (euclidean)':<28} {c_cost:>10.4f} {c_iters+1:>12}")
    if c_cost < best_cost:
        best_cost, best_positions, best_mst = c_cost, c_pos, c_mst
        best_original_cost = Kruskal_MST(centers, graph.edges)[1]

    # diversified strategies
    for strat in strategies:
        if strat == 'adaptive':
            result = Heuristic_MSTN_adaptive(
                graph,
                max_iter=heuristic_max_iter,
                tol=heuristic_tol,
                alpha=alpha,
                metrics=metrics,
                n_random=n_random,
                seed=seed,
            )
            note = f"α={alpha}  best start: {result[5]}"
        else:
            result = Heuristic_MSTN_alternating(
                graph,
                max_iter=heuristic_max_iter,
                tol=heuristic_tol,
                strategy=strat,
                metrics=metrics,
                n_random=n_random,
                seed=seed,
            )
            note = f"best start: {result[5]}" if strat == 'multistart' else ""

        pos, mst, cost, orig_cost, iters = (
            result[0], result[1], result[2], result[3], result[4]
        )
        log_lines.append(f"{strat:<28} {cost:>10.4f} {iters+1:>12}  {note}")
        if cost < best_cost:
            best_cost          = cost
            best_positions     = pos
            best_mst           = mst
            best_original_cost = orig_cost

    log_lines.append(f"{'─'*55}")
    log_lines.append(f"{'Best overall':<28} {best_cost:>10.4f}")

    # ── 3. save best solution ─────────────────────────────────────────────────
    fig_sol, _ = Draw_solution(graph, best_positions, best_mst, best_original_cost)
    fig_sol.savefig(os.path.join(output_dir, 'best_solution.png'), bbox_inches='tight', dpi=150)
    plt.close(fig_sol)
    print(f"Saved: best_solution.png (cost={best_cost:.4f})")

    # ── 4. save Heuristic_Grid for each strategy ──────────────────────────────
    for strat in strategies:
        if strat in ('multistart', 'adaptive'):
            starts = multistart_start if isinstance(multistart_start, list) \
                     else [multistart_start]
            for idx in starts:
                fname_img = f'grid_{strat}_{idx}.png'
                Heuristic_Grid(
                    graph, max_iter=max_iter,
                    strategy=strat,
                    metrics=metrics, n_random=n_random, seed=seed,
                    multistart_start=idx,
                    alpha=alpha if strat == 'adaptive' else 1.0,
                    save_path=os.path.join(output_dir, fname_img),
                )
                print(f"Saved: {fname_img}")
        else:
            fname_img = f'grid_{strat}.png'
            Heuristic_Grid(
                graph, max_iter=max_iter, strategy=strat,
                metrics=metrics, n_random=n_random, seed=seed,
                save_path=os.path.join(output_dir, fname_img),
            )
            print(f"Saved: {fname_img}")

    # ── 5. write log ──────────────────────────────────────────────────────────
    log_path = os.path.join(output_dir, 'experiment.log')
    with open(log_path, 'w') as f:
        f.write('\n'.join(log_lines) + '\n')
    print(f"Saved: experiment.log")


def Run_best(
    graph: GeoGraph,
    strategies: list = None,
    metrics=None,
    n_random: int = 2,
    seed: int = 42,
    max_iter: int = 100,
    tol: float = 1e-2,
):
    strategies = strategies or ['classic', 'best', 'cycle', 'random', 'multistart']
    centers = [nb.center for nb in graph.neighborhoods]
    original_cost = Kruskal_MST(centers, graph.edges)[1]

    print(f"\n{'='*55}")
    print(f"{'Method':<28} {'Cost':>10} {'Iterations':>12}  Note")
    print(f"{'─'*55}")

    best_cost = float("inf")
    best_positions, best_mst, best_label = None, None, None

    for strat in strategies:
        if strat == 'classic':
            pos, mst, cost, iters = _run_classic(graph, centers, max_iter, tol)
            note = ""
        else:
            result = Heuristic_MSTN_alternating(
                graph,
                max_iter=max_iter,
                tol=tol,
                strategy=strat,
                metrics=metrics,
                n_random=n_random,
                seed=seed,
            )
            pos, mst, cost, iters = result[0], result[1], result[2], result[4]
            note = f"best start: {result[5]}" if strat == 'multistart' else ""

        print(f"{strat:<28} {cost:>10.4f} {iters+1:>12}  {note}")

        if cost < best_cost:
            best_cost      = cost
            best_positions = pos
            best_mst       = mst
            best_label     = strat if not note else f"multistart ({result[5]})"

    print(f"{'─'*55}")
    print(f"{'Best overall':<28} {best_cost:>10.4f}  ← {best_label}")
    print(f"{'='*55}\n")

    Draw_solution_centered(
        graph,
        best_positions,
        best_mst,
        original_cost,
        title=f"MSTN Best Solution — {best_label}  (cost={best_cost:.4f})",
    )

# Animate MSTN on your mixed graph G
# Heuristic_Animate(G, max_iter=25, tol=1e-2, interval=500)
# Heuristic_Grid(G, max_iter = 12)

