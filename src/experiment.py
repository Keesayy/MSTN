import matplotlib.pyplot as plt
import random_graph
import time
import re
from heuristic import Heuristic_MSTN_alternating

def Run_experiment_grid(
        max_grid_size: int,
        n_percent: float,
        smin_percent: float,
        smax_percent: float,
        max_iter: int = 50
    ):
    n_values = []
    iteration_values = []
    reduction_values = []
    time_values = []

    for grid in range(1, max_grid_size + 1):

        n = max(1, int(grid * n_percent))
        smin = grid * smin_percent
        smax = grid * smax_percen

        G = random_graph.Create_random_graph(
            n = n,
            m = 0,
            shape = "square",
            integer_size = True,
            size_min = smin,
            size_max = smax,
            grid_size = grid
        )
        G.Complete_graph()

        start = time.perf_counter()
        pos, mst, cost_mstn, cost_centers, k = Heuristic_MSTN_alternating(G, max_iter = max_iter)
        elapsed = time.perf_counter() - start

        reduction = (cost_centers - cost_mstn) / cost_centers * 100 if cost_centers > 0 else 0

        # --- store ---
        n_values.append(n)
        iteration_values.append(k)
        reduction_values.append(reduction)
        time_values.append(elapsed)

        print(f"grid={grid}, n={n}, iter={k}, reduction={reduction:.2f}%, time={elapsed:.4f}s")

    fig1 = plt.figure(figsize=(7, 5))
    plt.title("Heuristic Iterations vs Number of Neighborhoods")

    plt.plot(n_values, iteration_values, color='black', label="Iteration Curve")
    plt.scatter(n_values, iteration_values, color='red', label="Measured Iterations")

    plt.xlabel("Number of Neighborhoods (n)")
    plt.ylabel("Heuristic Iterations")
    plt.grid(True)
    plt.legend(loc="lower right")

    plt.text(
        1.02, 0.98,
        f"Parameters:\n"
        f"max_grid_size = {max_grid_size}\n"
        f"n_percent = {n_percent}\n"
        f"smin% = {smin_percent}\n"
        f"smax% = {smax_percent}\n"
        f"max iter = {max_iter}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    fig2 = plt.figure(figsize=(7, 5))
    plt.title("MSTN Cost Reduction vs MST(Centers)")

    plt.plot(n_values, reduction_values, color='black', label="Reduction Curve")
    plt.scatter(n_values, reduction_values, color='blue', label="Measured Reduction")

    plt.xlabel("Number of Neighborhoods (n)")
    plt.ylabel("% Reduction (MST(center) → MSTN)")
    plt.grid(True)
    plt.legend(loc="lower right")

    plt.text(
        1.02, 0.98,
        f"Parameters:\n"
        f"max_grid_size = {max_grid_size}\n"
        f"n_percent = {n_percent}\n"
        f"smin% = {smin_percent}\n"
        f"smax% = {smax_percent}\n"
        f"max iter = {max_iter}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    fig3 = plt.figure(figsize=(7, 5))
    plt.title("Runtime of Heuristic vs Number of Neighborhoods")

    plt.plot(n_values, time_values, color='black', label="Runtime Curve")
    plt.scatter(n_values, time_values, color='purple', label="Measured Runtime")

    plt.xlabel("Number of Neighborhoods (n)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.legend(loc="upper left")

    plt.text(
        1.02, 0.98,
        f"Parameters:\n"
        f"max_grid_size = {max_grid_size}\n"
        f"n_percent = {n_percent}\n"
        f"smin% = {smin_percent}\n"
        f"smax% = {smax_percent}\n"
        f"max iter = {max_iter}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )
    
    return n_values, iteration_values, reduction_values, time_values


def Read_experiment_file(filename):

    grids = []
    n_values = []
    iteration_values = []
    reduction_values = []
    time_values = []

    max_grid_size = 1000 
    n_percent = 0.8 
    smin_percent = 0.03
    smax_percent = 0.1 
    max_iter = 1000

    pattern = re.compile(
        r"grid=(\d+), n=(\d+), iter=(\d+), reduction=([\d\.]+)%, time=([\d\.]+)s"
    )

    # Read file
    with open(filename, "r") as f:
        for line in f:
            match = pattern.search(line)
            if match:
                grid = int(match.group(1))
                n = int(match.group(2))
                it = int(match.group(3))
                red = float(match.group(4))
                t = float(match.group(5))

                grids.append(grid)
                n_values.append(n)
                iteration_values.append(it)
                reduction_values.append(red)
                time_values.append(t)

    fig1 = plt.figure(figsize=(7, 5))
    plt.title("Heuristic Iterations vs Number of Neighborhoods")

    plt.plot(n_values, iteration_values, color='black', label="Iteration Curve")
    plt.scatter(n_values, iteration_values, color='red', label="Measured Iterations")

    plt.xlabel("Number of Neighborhoods (n)")
    plt.ylabel("Heuristic Iterations")
    plt.grid(True)
    plt.legend(loc="lower right")

    plt.text(
        1.02, 0.98,
        f"Parameters:\n"
        f"max_grid_size = {max_grid_size}\n"
        f"n_percent = {n_percent}\n"
        f"smin% = {smin_percent}\n"
        f"smax% = {smax_percent}\n"
        f"max iter = {max_iter}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    fig2 = plt.figure(figsize=(7, 5))
    plt.title("MSTN Cost Reduction vs MST(Centers)")

    plt.plot(n_values, reduction_values, color='black', label="Reduction Curve")
    plt.scatter(n_values, reduction_values, color='blue', label="Measured Reduction")

    plt.xlabel("Number of Neighborhoods (n)")
    plt.ylabel("% Reduction (MST(center) → MSTN)")
    plt.grid(True)
    plt.legend(loc="lower right")

    plt.text(
        1.02, 0.98,
        f"Parameters:\n"
        f"max_grid_size = {max_grid_size}\n"
        f"n_percent = {n_percent}\n"
        f"smin% = {smin_percent}\n"
        f"smax% = {smax_percent}\n"
        f"max iter = {max_iter}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    fig3 = plt.figure(figsize=(7, 5))
    plt.title("Runtime of Heuristic vs Number of Neighborhoods")

    plt.plot(n_values, time_values, color='black', label="Runtime Curve")
    plt.scatter(n_values, time_values, color='purple', label="Measured Runtime")

    plt.xlabel("Number of Neighborhoods (n)")
    plt.ylabel("Execution Time (seconds)")
    plt.grid(True)
    plt.legend(loc="upper left")

    plt.text(
        1.02, 0.98,
        f"Parameters:\n"
        f"max_grid_size = {max_grid_size}\n"
        f"n_percent = {n_percent}\n"
        f"smin% = {smin_percent}\n"
        f"smax% = {smax_percent}\n"
        f"max iter = {max_iter}",
        transform=plt.gca().transAxes,
        fontsize=10,
        verticalalignment='top',
        bbox=dict(facecolor='white', edgecolor='black')
    )

    fig1.savefig("iterations_vs_neighborhoods.png", dpi=300, bbox_inches='tight')
    fig2.savefig("mstn_cost_reduction.png", dpi=300, bbox_inches='tight')
    fig3.savefig("heuristic_runtime.png", dpi=300, bbox_inches='tight')
        
    plt.show()


# Run_experiment_grid(max_grid_size = 1000, n_percent = 0.8, smin_percent = 0.03, smax_percent = 0.1, max_iter = 1000)
# Read_experiment_file("experiment")