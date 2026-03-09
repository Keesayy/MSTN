import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from geometry import Point, SquareNeighborhood, CircleNeighborhood, Distance
from graph import GeoGraph

fig_size = (10, 6)

def Draw_graph(graph: GeoGraph):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_title("Initial Graph")

    g = graph.grid_size
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_xticks(range(0, g + 1), minor=True)
    ax.set_yticks(range(0, g + 1), minor=True)
    ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
    ax.grid(which='major', linewidth=1, linestyle='-', color='lightgray')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    centers = [nb.center for nb in graph.neighborhoods]

    # Draw neighborhoods
    for i, nb in enumerate(graph.neighborhoods):
        cx, cy = nb.center.x, nb.center.y

        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            rect = patches.Rectangle(
                (cx - half, cy - half),
                nb.side, nb.side,
                fill=False, edgecolor='blue', linewidth=1.5
            )
            ax.add_patch(rect)

        elif isinstance(nb, CircleNeighborhood):
            circ = patches.Circle(
                (cx, cy),
                nb.radius,
                fill=False, edgecolor='blue', linewidth=1.5
            )
            ax.add_patch(circ)

        # Vertex number
        ax.text(cx + 0.1, cy + 0.1, str(i), fontsize=12, color='black')

    # Draw centers
    ax.scatter([p.x for p in centers], [p.y for p in centers],
               color='black', s=50, label="Neighborhood Centers")

    # Draw edges
    for (u, v) in graph.edges:
        p, q = centers[u], centers[v]
        ax.plot([p.x, q.x], [p.y, q.y], color='gray', linewidth=1)

    # Distance list box (top-right)
    edge_report_list = graph.edges[:40]
    dist_report = "\n".join(f"{u} - {v} : {Distance(centers[u], centers[v]):.2f}" for (u, v) in edge_report_list)
    ax.text(
        1.02, 0.98,
        "Edge Distances:\n" + dist_report,
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    ax.legend(loc="upper left")
    ax.set_aspect('equal')

    return fig, ax

def Draw_solution(graph: GeoGraph, positions, mst, original_cost):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_title("MSTN Solution")

    g = graph.grid_size
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_xticks(range(0, g + 1), minor=True)
    ax.set_yticks(range(0, g + 1), minor=True)
    ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
    ax.grid(which='major', linewidth=0.8, linestyle='-', color='lightgray')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # Draw neighborhoods
    for i, nb in enumerate(graph.neighborhoods):
        cx, cy = nb.center.x, nb.center.y

        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            rect = patches.Rectangle(
                (cx - half, cy - half),
                nb.side, nb.side,
                fill=False, edgecolor='blue', linewidth=1.3
            )
            ax.add_patch(rect)

        elif isinstance(nb, CircleNeighborhood):
            circ = patches.Circle(
                (cx, cy),
                nb.radius,
                fill=False, edgecolor='blue', linewidth=1.3
            )
            ax.add_patch(circ)

    # Draw chosen points + labels
    for i, p in enumerate(positions):
        ax.scatter(p.x, p.y, s=70, color='green')
        ax.text(p.x + 0.1, p.y + 0.1, str(i), fontsize=12, color='black')

    # Draw MST edges
    total_cost = 0
    for (u, v, w) in mst:
        p, q = positions[u], positions[v]
        total_cost += w
        ax.plot([p.x, q.x], [p.y, q.y], color='red', linewidth=2)

    # Distance box
    mst_report = mst[:38]
    dist_report = "\n".join(f"{u} - {v} : {w:.2f}" for (u, v, w) in mst_report)
    ax.text(
        1.02, 0.98,
        f"Edge Distances:\n{dist_report}\n\nTotal = {total_cost:.2f}\nMST = {original_cost:.2f}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='MST Edge'),
        Line2D([0], [0], marker='o', color='green', markersize=8,
               linestyle='None', label='Solution Points')
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_aspect('equal')

    return fig, ax

def Draw_graph_centered(graph: GeoGraph):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_title("Initial Graph (Centered)")

    # Compute bounding box of neighborhoods
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    for nb in graph.neighborhoods:
        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            min_x = min(min_x, nb.center.x - half)
            max_x = max(max_x, nb.center.x + half)
            min_y = min(min_y, nb.center.y - half)
            max_y = max(max_y, nb.center.y + half)
        elif isinstance(nb, CircleNeighborhood):
            min_x = min(min_x, nb.center.x - nb.radius)
            max_x = max(max_x, nb.center.x + nb.radius)
            min_y = min(min_y, nb.center.y - nb.radius)
            max_y = max(max_y, nb.center.y + nb.radius)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    shift_x = int(graph.grid_size / 2 - cx)
    shift_y = int(graph.grid_size / 2 - cy)

    g = graph.grid_size
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_xticks(range(0, g + 1), minor=True)
    ax.set_yticks(range(0, g + 1), minor=True)
    ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
    ax.grid(which='major', linewidth=1, linestyle='-', color='lightgray')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    centers = [Point(nb.center.x + shift_x, nb.center.y + shift_y) for nb in graph.neighborhoods]

    # Draw neighborhoods
    for i, nb in enumerate(graph.neighborhoods):
        cx_nb, cy_nb = nb.center.x + shift_x, nb.center.y + shift_y
        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            rect = patches.Rectangle((cx_nb - half, cy_nb - half), nb.side, nb.side,
                                     fill=False, edgecolor='blue', linewidth=1.5)
            ax.add_patch(rect)
        elif isinstance(nb, CircleNeighborhood):
            circ = patches.Circle((cx_nb, cy_nb), nb.radius,
                                  fill=False, edgecolor='blue', linewidth=1.5)
            ax.add_patch(circ)
        ax.text(cx_nb + 0.1, cy_nb + 0.1, str(i), fontsize=12, color='black')

    # Draw centers
    ax.scatter([p.x for p in centers], [p.y for p in centers],
               color='black', s=50, label="Neighborhood Centers")

    # Draw edges
    for (u, v) in graph.edges:
        p, q = centers[u], centers[v]
        ax.plot([p.x, q.x], [p.y, q.y], color='gray', linewidth=0.2)

    # Distance list box
    edge_report_list = graph.edges[:40]
    dist_report = "\n".join(f"{u} - {v} : {Distance(centers[u], centers[v]):.2f}" for (u, v) in edge_report_list)
    ax.text(1.02, 0.98, "Edge Distances:\n" + dist_report,
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black'))

    ax.legend(loc="upper left")
    ax.set_aspect('equal')

    return fig, ax


def Draw_solution_centered(graph: GeoGraph, positions, mst, original_cost, title: str="MSTN Solution (Centered)"):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_title(title)

    # Compute bounding box of neighborhoods
    min_x, max_x, min_y, max_y = float('inf'), float('-inf'), float('inf'), float('-inf')
    for nb in graph.neighborhoods:
        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            min_x = min(min_x, nb.center.x - half)
            max_x = max(max_x, nb.center.x + half)
            min_y = min(min_y, nb.center.y - half)
            max_y = max(max_y, nb.center.y + half)
        elif isinstance(nb, CircleNeighborhood):
            min_x = min(min_x, nb.center.x - nb.radius)
            max_x = max(max_x, nb.center.x + nb.radius)
            min_y = min(min_y, nb.center.y - nb.radius)
            max_y = max(max_y, nb.center.y + nb.radius)

    cx = (min_x + max_x) / 2
    cy = (min_y + max_y) / 2
    shift_x = int(graph.grid_size / 2 - cx)
    shift_y = int(graph.grid_size / 2 - cy)

    g = graph.grid_size
    ax.set_xlim(0, g)
    ax.set_ylim(0, g)
    ax.set_xticks(range(0, g + 1), minor=True)
    ax.set_yticks(range(0, g + 1), minor=True)
    ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
    ax.grid(which='major', linewidth=0.8, linestyle='-', color='lightgray')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # Draw neighborhoods
    for nb in graph.neighborhoods:
        cx_nb, cy_nb = nb.center.x + shift_x, nb.center.y + shift_y
        if isinstance(nb, SquareNeighborhood):
            half = nb.side / 2
            rect = patches.Rectangle((cx_nb - half, cy_nb - half), nb.side, nb.side,
                                     fill=False, edgecolor='blue', linewidth=1.3)
            ax.add_patch(rect)
        elif isinstance(nb, CircleNeighborhood):
            circ = patches.Circle((cx_nb, cy_nb), nb.radius,
                                  fill=False, edgecolor='blue', linewidth=1.3)
            ax.add_patch(circ)

    # Draw solution points + labels
    shifted_positions = [Point(p.x + shift_x, p.y + shift_y) for p in positions]
    for i, p in enumerate(shifted_positions):
        ax.scatter(p.x, p.y, s=70, color='green')
        ax.text(p.x + 0.1, p.y + 0.1, str(i), fontsize=12, color='black')

    # Draw MST edges
    total_cost = 0
    for (u, v, w) in mst:
        p, q = shifted_positions[u], shifted_positions[v]
        total_cost += w
        ax.plot([p.x, q.x], [p.y, q.y], color='red', linewidth=2)

    # Distance box
    mst_report = mst[:37]
    dist_report = "\n".join(f"{u} - {v} : {w:.2f}" for (u, v, w) in mst_report)
    ax.text(1.02, 0.98,
            f"Edge Distances:\n{dist_report}\n\nTotal = {total_cost:.2f}\nMST = {original_cost:.2f}",
            transform=ax.transAxes, fontsize=10, verticalalignment='top',
            bbox=dict(facecolor='white', edgecolor='black'))

    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='MST Edge'),
        Line2D([0], [0], marker='o', color='green', markersize=8,
               linestyle='None', label='Solution Points')
    ]
    ax.legend(handles=legend_elements, loc="upper left")
    ax.set_aspect('equal')

    return fig, ax

def Draw_solution_centered_compare(
    graph: GeoGraph,
    heuristic_positions,   # list[Point] – one representative per vertex
    heuristic_mst,         # list[(u, v, w)]
    gurobi_positions,      # list[Point]
    gurobi_mst,            # list[(u, v, w)]
    original_cost: float,  # MST cost at neighborhood centers
) -> tuple:
    """
    Overlay the heuristic and Gurobi MSTN solutions on a single centred figure.
    """
    heuristic_cost = sum(w for _, _, w in heuristic_mst)
    gurobi_cost    = sum(w for _, _, w in gurobi_mst)
    gap_pct = (
        (heuristic_cost - gurobi_cost) / gurobi_cost * 100
        if gurobi_cost > 0 else 0.0
    )

    # Normalised edge sets 
    def _ne(u, v):
        return (min(u, v), max(u, v))

    h_edge_map = {_ne(u, v): w for u, v, w in heuristic_mst}
    g_edge_map = {_ne(u, v): w for u, v, w in gurobi_mst}
    h_only = set(h_edge_map) - set(g_edge_map)   # heuristic but NOT Gurobi
    g_only = set(g_edge_map) - set(h_edge_map)   # Gurobi but NOT heuristic
    shared = set(h_edge_map) & set(g_edge_map)   # in both trees

    # Centre shift 
    min_x = min_y =  float('inf')
    max_x = max_y = -float('inf')
    for nb in graph.neighborhoods:
        if isinstance(nb, SquareNeighborhood):
            h = nb.side / 2
            min_x = min(min_x, nb.center.x - h);  max_x = max(max_x, nb.center.x + h)
            min_y = min(min_y, nb.center.y - h);  max_y = max(max_y, nb.center.y + h)
        elif isinstance(nb, CircleNeighborhood):
            min_x = min(min_x, nb.center.x - nb.radius)
            max_x = max(max_x, nb.center.x + nb.radius)
            min_y = min(min_y, nb.center.y - nb.radius)
            max_y = max(max_y, nb.center.y + nb.radius)

    cx_bb   = (min_x + max_x) / 2
    cy_bb   = (min_y + max_y) / 2
    shift_x = int(graph.grid_size / 2 - cx_bb)
    shift_y = int(graph.grid_size / 2 - cy_bb)

    # Figure
    g   = graph.grid_size
    fig = plt.figure(figsize=fig_size)

    # Left axes: the actual plot
    ax = fig.add_axes([0.03, 0.08, 0.50, 0.88])
    ax.set_title("MSTN Comparison - Heuristic vs Gurobi (exact)", fontsize=12, pad=10)
    ax.set_xlim(0, g);  ax.set_ylim(0, g)
    ax.set_xticks(range(0, g + 1), minor=True)
    ax.set_yticks(range(0, g + 1), minor=True)
    ax.grid(which='minor', linewidth=0.4, linestyle='-', color='lightgray')
    ax.grid(which='major', linewidth=0.7, linestyle='-', color='lightgray')
    ax.set_axisbelow(True)
    ax.set_aspect('equal')

    # Neighbourhoods (blue outlines + vertex index)
    for i, nb in enumerate(graph.neighborhoods):
        cx_nb = nb.center.x + shift_x
        cy_nb = nb.center.y + shift_y
        if isinstance(nb, SquareNeighborhood):
            h = nb.side / 2
            ax.add_patch(patches.Rectangle(
                (cx_nb - h, cy_nb - h), nb.side, nb.side,
                fill=False, edgecolor='blue', linewidth=1.3
            ))
        elif isinstance(nb, CircleNeighborhood):
            ax.add_patch(patches.Circle(
                (cx_nb, cy_nb), nb.radius,
                fill=False, edgecolor='blue', linewidth=1.3
            ))

    # Shift representative positions 
    h_pts = [Point(p.x + shift_x, p.y + shift_y) for p in heuristic_positions]
    g_pts = [Point(p.x + shift_x, p.y + shift_y) for p in gurobi_positions]

    # Heuristic MST edges
    for (u, v, w) in heuristic_mst:
        e    = _ne(u, v)
        p, q = h_pts[u], h_pts[v]
        ls   = '--' if e in h_only else '-'
        ax.plot([p.x, q.x], [p.y, q.y],
                color='red', linewidth=2.5, linestyle=ls, alpha=0.85, zorder=2)

    # Gurobi MST edges
    for (u, v, w) in gurobi_mst:
        e    = _ne(u, v)
        p, q = g_pts[u], g_pts[v]
        ls   = '--' if e in g_only else '-'
        ax.plot([p.x, q.x], [p.y, q.y],
                color='limegreen', linewidth=1.5, linestyle=ls, alpha=0.95, zorder=3)

    # Representative points 
    for p in h_pts:
        ax.scatter(p.x, p.y, s=70, color='orange',
                   edgecolors='darkorange', linewidths=0.8, zorder=4)
    for p in g_pts:
        ax.scatter(p.x, p.y, s=35, color='limegreen',
                   edgecolors='darkgreen', linewidths=0.8, zorder=5)

    # Node index labels centred inside each dot
    for i, (ph, pg) in enumerate(zip(h_pts, g_pts)):
        ax.text(ph.x, ph.y, str(i),
                fontsize=7, fontweight='bold', color='white',
                ha='center', va='center', zorder=6)
        ax.text(pg.x, pg.y, str(i),
                fontsize=6, fontweight='bold', color='darkgreen',
                ha='center', va='center', zorder=7)

    # Right-side info box 
    box_lines = [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f" MST (centers)  : {original_cost:>8.2f}",
        f" MSTN Heuristic : {heuristic_cost:>8.2f}",
        f" MSTN Gurobi    : {gurobi_cost:>8.2f}",
        f" Gap to exact   : {gap_pct:>+8.2f}%",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        " Arc   |  H-cost  | G-cost  |",
        "───────┼──────────┼─────────┼",
    ]

    for e in sorted(set(h_edge_map) | set(g_edge_map)):
        u, v   = e
        in_h   = e in h_edge_map
        in_g   = e in g_edge_map
        wh_str = f"{h_edge_map[e]:7.2f}" if in_h else "   —   "
        wg_str = f"{g_edge_map[e]:7.2f}" if in_g else "   —   "
        box_lines.append(f" {u:2d}-{v:2d} | {wh_str}  | {wg_str} |")

    box_lines += [
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
        f" Shared arcs : {len(shared)}",
        f" H-only arcs : {len(h_only)}",
        f" G-only arcs : {len(g_only)}",
        "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━",
    ]

    ax.text(
        1.02, 0.80,
        "\n".join(box_lines),
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(facecolor='white', edgecolor='black', alpha=0.92),
    )

    # Legend outside the plot, to the right of the axes
    legend_elements = [
        Line2D([0], [0], color='red',       lw=2.5,
               label='Heuristic arc (shared with Gurobi)'),
        Line2D([0], [0], color='red',       lw=2.5, linestyle='--',
               label='Heuristic-only arc'),
        Line2D([0], [0], color='limegreen', lw=1.5,
               label='Gurobi arc (shared with heuristic)'),
        Line2D([0], [0], color='limegreen', lw=1.5, linestyle='--',
               label='Gurobi-only arc'),
        Line2D([0], [0], marker='o', color='red',       markersize=9,
               markeredgecolor='darkred',   linestyle='None',
               label='Heuristic representative'),
        Line2D([0], [0], marker='o', color='limegreen', markersize=7,
               markeredgecolor='darkgreen', linestyle='None',
               label='Gurobi representative'),
    ]

    ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0,
        ncol=1,
        fontsize=8,
        framealpha=0.95,
    )

    ax.set_aspect('equal')

    return fig, ax
