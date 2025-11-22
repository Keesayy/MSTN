from geometry import Point
from typing import List, Tuple
import numpy as np

def Distance(p: Point, q: Point) -> float:
    return float(np.linalg.norm(p.To_numpy() - q.To_numpy()))

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0]*n

    def Find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def Union(self, a: int, b: int) -> bool:
        ra, rb = self.Find(a), self.Find(b)
        if ra == rb:
            return False
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1
        return True

def Kruskal_MST(points: List[Point], edges: List[Tuple[int,int]]):
    uf = UnionFind(len(points))
    weighted = [(Distance(points[u], points[v]), u, v) for u, v in edges]
    weighted.sort()
    mst = []
    total = 0

    for w, u, v in weighted:
        if uf.Union(u, v):
            mst.append((u, v, w))
            total += w

    return mst, total
