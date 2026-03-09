from geometry import Point, SquareNeighborhood, CircleNeighborhood
from graph import GeoGraph

from visualize import Draw_graph, Draw_solution, Draw_graph_centered, Draw_solution_centered, Draw_solution_centered_compare
from instance_io import save_instance, load_instance, print_dict
from experiment import Run_experiment_grid, Read_experiment_file

from heuristic import (
    Heuristic_MSTN_alternating,
    Heuristic_MSTN_adaptive,
    Heuristic_Animate, Heuristic_Grid, Save_experiment, Run_best
)

from minlp_mstn import Solve_MINLP

import random_graph
import matplotlib.pyplot as plt

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

# G = Build_demo_graph()

if __name__ == "__main__":
    size = 50
    n = 22
    smin = 2
    smax = 12

    G = random_graph.Create_random_graph(
        n = n, m = 0, 
        shape = "square", integer_size = True, 
        size_min = smin, size_max = smax,
        overlap_fraction = 0.0, overlap_degree = 0.0, grid_size = size)

    G.Complete_graph()

    # (G,
    # pos_h, mst_h, cost_h, original_cost, k,
    #     pos_g, mst_g, cost_g,
    #     gurobi_stats,
    #     extra_params,) = load_instance("../Graphs_data/1.json")

    # fig1, ax1 = Draw_solution_centered_compare(
    #     G, pos_h, mst_h, pos_g, mst_g, original_cost
    # )

    # # Draw initial graph
    # fig1, ax1 = Draw_graph_centered(G)
    # Run_best(G)


    pos, mst, cost, orig, iters, label = Heuristic_MSTN_adaptive(
        G,
        alpha=0.4,      # try 0.3, 0.5, 0.7, 1.0
        n_random=5,
        seed=42,
        verbose=True,
    )

    # fig2, ax2 = Draw_solution_centered(G, pos, mst, orig)

    pos, mst, cost, orig, iters, label = Heuristic_MSTN_alternating(
        G,
        strategy="multistart",
        n_random=5,       # number of random perturbation starts
        seed=42,
        verbose=True,     # prints each start's cost
    )

    # Save_experiment(G, output_dir='../Experiment/experiment_01')


    # # fig2, ax2 = Draw_solution_centered(G, pos_h, mst_h, original_cost)

    Heuristic_Grid(G, strategy='adaptive', alpha=0.4, max_iter=9, multistart_start=0)

    (G,
    pos_h, mst_h, cost_h, original_cost, k,
        pos_g, mst_g, cost_g,
        gurobi_stats,
        extra_params,) = load_instance("../Graphs_data/1.json")

    # --- Single comparison plot ---
    fig, ax = Draw_solution_centered_compare(
        G, pos_h, mst_h, pos_g, mst_g, original_cost
    )

    plt.show()


