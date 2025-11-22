from geometry import Point, SquareNeighborhood, CircleNeighborhood
from graph import GeoGraph
from heuristic import Heuristic_MSTN_alternating, Heuristic_Animate, Heuristic_Grid
from visualize import Draw_graph, Draw_solution, Draw_graph_centered, Draw_solution_centered
import matplotlib.pyplot as plt
import random_graph

def Build_demo_graph():
    neighborhoods = [
        SquareNeighborhood(Point(2, 3), 3),
        SquareNeighborhood(Point(7, 2), 2),
        SquareNeighborhood(Point(6, 6), 2),
        CircleNeighborhood(Point(3, 8), 1),
        SquareNeighborhood(Point(9, 7), 0.5),
    ]
    G = GeoGraph(neighborhoods)
    G.Complete_graph()
    return G


if __name__ == "__main__":
    # G = Build_demo_graph()

    G = random_graph.Create_random_graph(
        n = 30, m = 0, 
        shape = "square", integer_size = True, 
        size_min = 3.0, size_max = 7.0,
        overlap_fraction = 0.0, overlap_degree = 0.0, grid_size = 50)

    G.Complete_graph()

    # Draw initial graph
    fig1, ax1 = Draw_graph_centered(G)


    # Draw final solution
    pos, mst, cost, original_cost, k = Heuristic_MSTN_alternating(G, max_iter = 10)
    fig2, ax2 = Draw_solution_centered(G, pos, mst, original_cost)

    # Animate MSTN on your mixed graph G
    # Heuristic_Animate(G, max_iter=25, tol=1e-2, interval=500)
    Heuristic_Grid(G, max_iter = 12)

    # Run heuristic
    print("Heuristic iterations =", k)
    print("MSTN cost =", cost)
    print("MST cost =", original_cost)

    plt.show()