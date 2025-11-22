from typing import List, Tuple
from geometry import Point, SquareNeighborhood, CircleNeighborhood
from mst import Kruskal_MST
from visualize import Draw_solution_centered
from graph import GeoGraph
from matplotlib.animation import FuncAnimation
import matplotlib.patches as patches
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

def Heuristic_MSTN_alternating(
    graph: GeoGraph,
    max_iter: int = 100,
    tol: float = 1e-4,
):
    positions = [nb.center for nb in graph.neighborhoods]
    _, original_cost = Kruskal_MST(positions, graph.edges)

    for k in range(max_iter):
        mst, cost = Kruskal_MST(positions, graph.edges)

        neighbors = [[] for _ in range(graph.N)]
        for u, v, _ in mst:
            neighbors[u].append(v)
            neighbors[v].append(u)

        new_positions = positions.copy()
        max_move = 0.0

        for i in range(graph.N):
            if neighbors[i]:  # only update if there are neighbors
                arr = np.vstack([positions[j].To_numpy() for j in neighbors[i]])
                cx, cy = arr.mean(axis=0)

                projected = graph.neighborhoods[i].Project(Point(cx, cy))

                move = np.linalg.norm(
                    projected.To_numpy() - positions[i].To_numpy()
                )
                max_move = max(max_move, move)

                new_positions[i] = projected

        positions = new_positions

        if max_move < tol: break

    mst, cost = Kruskal_MST(positions, graph.edges)
    return positions, mst, cost, original_cost, k

def Heuristic_Grid(graph, max_iter: int = 16, box_side='right'):

    positions = [nb.center for nb in graph.neighborhoods]

    n_rows = int(np.ceil(np.sqrt(max_iter)))
    n_cols = int(np.ceil(max_iter / n_rows))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = axes.flatten() if max_iter > 1 else [axes]

    # Compute MST for initial positions (iteration 0)
    mst0, cost0 = Kruskal_MST(positions, graph.edges)
    fig.suptitle(f"MSTN Heuristic Iterations", fontsize=14)

    prev_positions = positions.copy()

    for frame in range(max_iter):
        ax = axes[frame]
        ax.set_xlim(0, graph.grid_size)
        ax.set_ylim(0, graph.grid_size)
        ax.set_xticks(range(0, graph.grid_size+1), minor=True)
        ax.set_yticks(range(0, graph.grid_size+1), minor=True)
        ax.grid(which='minor', linewidth=0.5, linestyle='-', color='lightgray')
        ax.grid(which='major', linewidth=1, linestyle='-', color='lightgray')
        ax.set_axisbelow(True)
        ax.set_aspect('equal')

        # Use iteration 0 MST without moving nodes
        if frame == 0:
            mst, cost = mst0, cost0
            moved_nodes = set()
        else:
            # Compute MST on current positions
            mst, _ = Kruskal_MST(positions, graph.edges)

            # Build adjacency for moving points
            neighbors = [[] for _ in range(graph.N)]
            for u, v, _ in mst:
                neighbors[u].append(v)
                neighbors[v].append(u)

            # Move positions toward neighbors
            new_positions = positions.copy()
            moved_nodes = set()
            for i in range(graph.N):
                if neighbors[i]:
                    arr = np.vstack([positions[j].To_numpy() for j in neighbors[i]])
                    cx, cy = arr.mean(axis=0)
                    projected = graph.neighborhoods[i].Project(Point(cx, cy))
                    move = np.linalg.norm(projected.To_numpy() - positions[i].To_numpy())
                    if move > 1e-2:
                        moved_nodes.add(i)
                    new_positions[i] = projected
            positions = new_positions

            # Recompute MST on updated positions
            mst, cost = Kruskal_MST(positions, graph.edges)

        # Draw neighborhoods
        for nb in graph.neighborhoods:
            cx, cy = nb.center.x, nb.center.y
            if isinstance(nb, SquareNeighborhood):
                half = nb.side / 2
                rect = patches.Rectangle((cx-half, cy-half), nb.side, nb.side,
                                 fill=False, edgecolor='blue', linewidth=1.3)
                ax.add_patch(rect)
            elif isinstance(nb, CircleNeighborhood):
                circ = patches.Circle((cx, cy), nb.radius, fill=False,
                              edgecolor='blue', linewidth=1.3)
                ax.add_patch(circ)

        # Draw solution points + labels
        for i, p in enumerate(positions):
            ax.scatter(p.x, p.y, s=50, color='green')
            ax.text(p.x + 0.1, p.y + 0.1, str(i), fontsize=10, color='black')

        # Draw MST edges
        for u, v, w in mst:
            p, q = positions[u], positions[v]
            color = 'yellow' if u in moved_nodes or v in moved_nodes else 'red'
            linewidth = 2 if color == 'red' else 3
            ax.plot([p.x, q.x], [p.y, q.y], color=color, lw=linewidth, alpha=0.8)

        # Box with iteration and MST weight
        text_box = f"Iteration: {frame}\nMSTN Total Weight: {cost:.2f}"
        if box_side == 'right':
            ax.text(1.1, 0.5, text_box,
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black'))
        else:  # left
            ax.text(-0.18, 0.5, text_box,
                    transform=ax.transAxes, fontsize=10,
                    verticalalignment='center', bbox=dict(facecolor='white', edgecolor='black'))

    # Add single legend + info box
    legend_elements = [
        Line2D([0], [0], color='red', lw=2, label='MST Edge'),
        Line2D([0], [0], color='yellow', lw=3, label='Moving Edge'),
        Line2D([0], [0], marker='o', color='green', markersize=8,
               linestyle='None', label='Solution Points')
    ]

    fig.legend(handles=legend_elements, loc='upper right', fontsize=10)        

    # Hide unused subplots
    for i in range(max_iter, len(axes)):
        axes[i].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)
    plt.show()

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


