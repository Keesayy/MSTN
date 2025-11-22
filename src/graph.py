from dataclasses import dataclass
from typing import List, Tuple, Union
from geometry import Point, SquareNeighborhood, CircleNeighborhood

Neighborhood = Union[SquareNeighborhood, CircleNeighborhood]

@dataclass
class GeoGraph:
    neighborhoods: List[Neighborhood]
    edges: List[Tuple[int, int]]
    grid_size: int  # <--- NEW

    def __init__(self, neighborhoods: List[Neighborhood], grid_size: int = 50):
        self.neighborhoods = list(neighborhoods)
        self.edges = []
        self.grid_size = grid_size  # <--- store grid size

    @property
    def N(self) -> int:
        return len(self.neighborhoods)

    def Add_edge(self, u: int, v: int):
        if u > v:
            u, v = v, u
        if (u, v) not in self.edges:
            self.edges.append((u, v))

    def Complete_graph(self):
        self.edges = [(i, j) for i in range(self.N) for j in range(i+1, self.N)]
