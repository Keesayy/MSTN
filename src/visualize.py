import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from geometry import Point, SquareNeighborhood, CircleNeighborhood, Distance
from graph import GeoGraph

fig_size = (10, 10)

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


def Draw_solution_centered(graph: GeoGraph, positions, mst, original_cost):
    fig = plt.figure(figsize=fig_size)
    ax = fig.add_subplot(111)
    ax.set_title("MSTN Solution (Centered)")

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
