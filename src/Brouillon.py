import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def To_numpy(self) -> np.ndarray:
        return np.array([self.x, self.y], dtype=float)

    @staticmethod
    def From_numpy(a: np.ndarray) -> "Point":
        return Point(float(a[0]), float(a[1]))

@dataclass(frozen=True)
class SquareNeighborhood:
    center: Point
    side: float

    def Project(self, p: Point) -> Point:
        cx, cy = self.center.x, self.center.y
        half = self.side / 2
        px = min(max(p.x, cx - half), cx + half)
        py = min(max(p.y, cy - half), cy + half)
        return Point(px, py)

def visualize_projection_L_shape():
    # Node 0 square
    node0_square = SquareNeighborhood(center=Point(1, 1), side=1)

    # Three neighbors forming a triangle
    neighbors_positions = [
        Point(3, 3),
        Point(2, 0),
        Point(1, 3)
    ]

    # Compute barycenter
    arr = np.vstack([p.To_numpy() for p in neighbors_positions])
    barycenter_coords = arr.mean(axis=0)
    barycenter = Point.From_numpy(barycenter_coords)

    # Project barycenter onto the square
    projected = node0_square.Project(barycenter)

    # Plotting
    plt.figure(figsize=(7,7))

    # Plot neighbors
    for i, p in enumerate(neighbors_positions):
        plt.scatter(p.x, p.y, color='blue', s=100, label='Neighbor' if i==0 else "")
        plt.text(p.x+0.05, p.y+0.05, f'P{i}', fontsize=10)

    # Draw triangle connecting neighbors
    triangle_x = [p.x for p in neighbors_positions] + [neighbors_positions[0].x]
    triangle_y = [p.y for p in neighbors_positions] + [neighbors_positions[0].y]
    plt.plot(triangle_x, triangle_y, 'b-', linewidth=1, alpha=0.5, label='Triangle for barycenter')

    # Plot barycenter
    plt.scatter(barycenter.x, barycenter.y, color='red', s=100, label='Barycenter')
    plt.text(barycenter.x + 0.05, barycenter.y + 0.05, 'B', color='red', fontsize=10)

    # Draw lines from neighbors to barycenter
    for p in neighbors_positions:
        plt.plot([p.x, barycenter.x], [p.y, barycenter.y], 'r--', alpha=0.5)

    # Plot square for node0
    half = node0_square.side / 2
    square = plt.Rectangle(
        (node0_square.center.x - half, node0_square.center.y - half),
        node0_square.side, node0_square.side,
        edgecolor='green', facecolor='none', linewidth=2, label='Node 0 square'
    )
    plt.gca().add_patch(square)
    plt.scatter(node0_square.center.x, node0_square.center.y, color='green', s=100, label='Node 0 center')

    # Plot projected barycenter
    plt.scatter(projected.x, projected.y, color='orange', s=100, label='Projected barycenter')

    # Draw L-shaped orthogonal projection lines
    intermediate_x = projected.x
    intermediate_y = barycenter.y
    # Vertical line (adjust x)
    if barycenter.x != projected.x:
        plt.plot([barycenter.x, intermediate_x], [barycenter.y, intermediate_y], 'orange', linestyle=':', linewidth=2)
    # Horizontal line (adjust y)
    if barycenter.y != projected.y:
        plt.plot([intermediate_x, projected.x], [intermediate_y, projected.y], 'orange', linestyle=':', linewidth=2)

    # Draw edges from node0 center to neighbors
    for p in neighbors_positions:
        plt.plot([node0_square.center.x, p.x], [node0_square.center.y, p.y], 'k--', alpha=0.3)

    plt.xlim(-1, 3)
    plt.ylim(-1, 4)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.legend()
    plt.title("Barycenter Projection with L-Shaped Orthogonal Lines")
    plt.show()

visualize_projection_L_shape()
