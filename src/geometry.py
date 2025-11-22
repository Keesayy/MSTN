from dataclasses import dataclass
import numpy as np

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


@dataclass(frozen=True)
class CircleNeighborhood:
    center: Point
    radius: float

    def Project(self, p: Point) -> Point:
        c = self.center.To_numpy()
        v = p.To_numpy() - c
        norm = np.linalg.norm(v)
        if norm <= self.radius or norm == 0:
            return p
        projected = c + (v / norm) * self.radius
        return Point.From_numpy(projected)


def Distance(p: Point, q: Point) -> float:
    return float(np.linalg.norm(p.To_numpy() - q.To_numpy()))
