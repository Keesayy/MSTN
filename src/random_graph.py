import random
from geometry import Point, SquareNeighborhood, CircleNeighborhood
from graph import GeoGraph

def Random_point(xmin=0, xmax=50, ymin=0, ymax=50):
    return Point(random.randint(xmin, xmax), random.randint(ymin, ymax))

def Random_square(integer_side=True, side_min=1, side_max=8,
                  xmin=0, xmax=50, ymin=0, ymax=50):
    if integer_side:
        side = random.randint(int(side_min), int(side_max))
        half_low = side // 2
        half_high = side - half_low
    else:
        side = random.uniform(side_min, side_max)
        half_low = side / 2
        half_high = side - half_low

    # Compute integer range for center (safe)
    cx_min = xmin + half_low
    cx_max = xmax - half_high
    cy_min = ymin + half_low
    cy_max = ymax - half_high

    center = Random_point(int(cx_min), int(cx_max), int(cy_min), int(cy_max))
    return SquareNeighborhood(center, side)

def Random_circle(integer_radius=False, radius_min=1, radius_max=4,
                  xmin=0, xmax=50, ymin=0, ymax=50):
    if integer_radius:
        radius = random.randint(int(radius_min), int(radius_max))
    else:
        radius = random.uniform(radius_min, radius_max)

    cx_min = xmin + radius
    cx_max = xmax - radius
    cy_min = ymin + radius
    cy_max = ymax - radius

    center = Random_point(int(cx_min), int(cx_max), int(cy_min), int(cy_max))
    return CircleNeighborhood(center, radius/2)

def Squares_overlap(sq1: SquareNeighborhood, sq2: SquareNeighborhood):
    half1 = sq1.side / 2
    half2 = sq2.side / 2
    return (abs(sq1.center.x - sq2.center.x) <= half1 + half2) and \
           (abs(sq1.center.y - sq2.center.y) <= half1 + half2)

def Circles_overlap(c1: CircleNeighborhood, c2: CircleNeighborhood):
    dx = c1.center.x - c2.center.x
    dy = c1.center.y - c2.center.y
    distance = (dx ** 2 + dy ** 2) ** 0.5
    return distance <= (c1.radius + c2.radius)

def Random_square_non_overlapping(integer_side=True, side_min=1, side_max=8,
                                  existing=[], xmin=0, xmax=50, ymin=0, ymax=50):
    for _ in range(100):
        sq = Random_square(integer_side, side_min, side_max, xmin, xmax, ymin, ymax)
        if all(not Squares_overlap(sq, other) for other in existing):
            return sq
    return sq

def Random_circle_non_overlapping(integer_radius=False, radius_min=1, radius_max=4,
                                  existing=[], xmin=0, xmax=50, ymin=0, ymax=50):
    for _ in range(100):
        c = Random_circle(integer_radius, radius_min, radius_max, xmin, xmax, ymin, ymax)
        if all(not Circles_overlap(c, other) for other in existing):
            return c
    return c

def Random_square_with_exact_overlap(integer_side=True, side_min=1, side_max=8,
                                     overlap_with=None, overlap_degree=0.5,
                                     xmin=0, xmax=50, ymin=0, ymax=50):
    sq = Random_square(integer_side, side_min, side_max, xmin, xmax, ymin, ymax)
    if overlap_with:
        target = random.choice(overlap_with)
        offset_x = random.uniform(-target.side * overlap_degree, target.side * overlap_degree)
        offset_y = random.uniform(-target.side * overlap_degree, target.side * overlap_degree)

        half_low = sq.side // 2
        half_high = sq.side - half_low

        new_cx = max(half_low, min(target.center.x + offset_x, xmax - half_high))
        new_cy = max(half_low, min(target.center.y + offset_y, ymax - half_high))

        center = Point(int(round(new_cx)), int(round(new_cy)))
        sq = SquareNeighborhood(center, sq.side)
    return sq

def Random_circle_with_exact_overlap(integer_radius=False, radius_min=1, radius_max=4,
                                     overlap_with=None, overlap_degree=0.5,
                                     xmin=0, xmax=50, ymin=0, ymax=50):
    c = Random_circle(integer_radius, radius_min, radius_max, xmin, xmax, ymin, ymax)
    if overlap_with:
        target = random.choice(overlap_with)
        offset_x = random.uniform(-target.radius * overlap_degree, target.radius * overlap_degree)
        offset_y = random.uniform(-target.radius * overlap_degree, target.radius * overlap_degree)

        new_cx = max(c.radius, min(target.center.x + offset_x, xmax - c.radius))
        new_cy = max(c.radius, min(target.center.y + offset_y, ymax - c.radius))
        center = Point(int(round(new_cx)), int(round(new_cy)))

        c = CircleNeighborhood(center, c.radius)
    return c

def Create_random_square_graph(n, m, integer_side=True, side_min=1, side_max=8,
                               overlap_fraction=0.0, overlap_degree=0.5,
                               grid_size=50):
    neighborhoods = []
    xmin = ymin = 1
    xmax = ymax = grid_size
    n_overlap = int(n * overlap_fraction)
    n_non_overlap = n - n_overlap

    for _ in range(n_non_overlap):
        neighborhoods.append(Random_square_non_overlapping(integer_side, side_min, side_max,
                                                           neighborhoods, xmin, xmax, ymin, ymax))
    for _ in range(n_overlap):
        neighborhoods.append(Random_square_with_exact_overlap(integer_side, side_min, side_max,
                                                              neighborhoods, overlap_degree,
                                                              xmin, xmax, ymin, ymax))
    G = GeoGraph(neighborhoods, grid_size=grid_size)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = min(m, len(all_edges))
    for u, v in random.sample(all_edges, m):
        G.Add_edge(u, v)
    return G

def Create_random_circle_graph(n, m, integer_radius=False, radius_min=1, radius_max=4,
                               overlap_fraction=0.0, overlap_degree=0.5,
                               grid_size=50):
    neighborhoods = []
    xmin = ymin = 0
    xmax = ymax = grid_size
    n_overlap = int(n * overlap_fraction)
    n_non_overlap = n - n_overlap

    for _ in range(n_non_overlap):
        neighborhoods.append(Random_circle_non_overlapping(integer_radius, radius_min, radius_max,
                                                           neighborhoods, xmin, xmax, ymin, ymax))
    for _ in range(n_overlap):
        neighborhoods.append(Random_circle_with_exact_overlap(integer_radius, radius_min, radius_max,
                                                              neighborhoods, overlap_degree,
                                                              xmin, xmax, ymin, ymax))
    G = GeoGraph(neighborhoods, grid_size=grid_size)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = min(m, len(all_edges))
    for u, v in random.sample(all_edges, m):
        G.Add_edge(u, v)
    return G

def Create_random_mixed_graph(n, m, integer_size=True,
                              size_min=1, size_max=8,
                              overlap_fraction=0.0, overlap_degree=0.5,
                              grid_size=50, square_fraction=0.5):
    neighborhoods = []
    xmin = ymin = 0
    xmax = ymax = grid_size

    n_squares = int(n * square_fraction)
    n_circles = n - n_squares

    n_overlap = int(n * overlap_fraction)
    n_non_overlap = n - n_overlap

    squares_non_overlap = min(n_squares, n_non_overlap)
    circles_non_overlap = n_non_overlap - squares_non_overlap

    for _ in range(squares_non_overlap):
        neighborhoods.append(Random_square_non_overlapping(integer_size, size_min, size_max,
                                                           neighborhoods, xmin, xmax, ymin, ymax))
    for _ in range(circles_non_overlap):
        neighborhoods.append(Random_circle_non_overlapping(integer_size, size_min, size_max,
                                                           neighborhoods, xmin, xmax, ymin, ymax))

    squares_overlap = min(n_squares - squares_non_overlap, n_overlap)
    circles_overlap = n_overlap - squares_overlap

    for _ in range(squares_overlap):
        neighborhoods.append(Random_square_with_exact_overlap(integer_size, size_min, size_max,
                                                              neighborhoods, overlap_degree,
                                                              xmin, xmax, ymin, ymax))
    for _ in range(circles_overlap):
        neighborhoods.append(Random_circle_with_exact_overlap(integer_size, size_min, size_max,
                                                              neighborhoods, overlap_degree,
                                                              xmin, xmax, ymin, ymax))
    G = GeoGraph(neighborhoods, grid_size=grid_size)
    all_edges = [(i, j) for i in range(n) for j in range(i + 1, n)]
    m = min(m, len(all_edges))
    for u, v in random.sample(all_edges, m):
        G.Add_edge(u, v)

    return G

def Create_random_graph(n, m, shape="square", integer_size=True,
                        size_min=1, size_max=8,
                        overlap_fraction=0.0, overlap_degree=0.5,
                        grid_size=50, square_fraction=0.5):
    shape = shape.lower()
    if shape == "square":
        return Create_random_square_graph(n, m, integer_size, size_min, size_max,
                                          overlap_fraction, overlap_degree, grid_size)
    elif shape == "circle":
        return Create_random_circle_graph(n, m, integer_size, size_min, size_max,
                                          overlap_fraction, overlap_degree, grid_size)
    elif shape == "mixed":
        return Create_random_mixed_graph(n, m, integer_size, size_min, size_max,
                                         overlap_fraction, overlap_degree, grid_size, square_fraction)
    else:
        raise ValueError("shape must be either 'square', 'circle', or 'mixed'")

