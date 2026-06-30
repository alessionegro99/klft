#!/usr/bin/env python3
"""Generate validated SU(2) partition tables for KLFT.

Point sets follow Hartung et al., Eur. Phys. J. C 82, 237 (2022),
arXiv:2201.09625. Linear neighbors use its sign-aware unit-transfer rule.
Fibonacci neighbors are edges of the spherical Delaunay complex, obtained as
edges of the convex hull in R^4.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq
from scipy.spatial import ConvexHull

FloatArray = NDArray[np.float64]
IntArray = NDArray[np.int64]


def _linear_integer_points(m: int) -> IntArray:
    """Return unique signed integer vectors q with ||q||_1 = m."""
    if m < 1:
        raise ValueError("linear m must be >= 1")
    points: list[tuple[int, int, int, int]] = []
    for j0 in range(m + 1):
        for j1 in range(m - j0 + 1):
            for j2 in range(m - j0 - j1 + 1):
                j = (j0, j1, j2, m - j0 - j1 - j2)
                nonzero = [i for i, value in enumerate(j) if value]
                for signs in itertools.product((-1, 1), repeat=len(nonzero)):
                    q = list(j)
                    for i, sign in zip(nonzero, signs, strict=True):
                        q[i] *= sign
                    points.append(tuple(q))
    return np.asarray(sorted(points), dtype=np.int64)


def _linear_neighbors(integer_points: IntArray) -> list[list[int]]:
    point_to_index = {tuple(q): i for i, q in enumerate(integer_points)}
    adjacency: list[list[int]] = []
    for q in integer_points:
        neighbors: set[int] = set()
        for source in range(4):
            if q[source] == 0:
                continue
            reduced = q.copy()
            reduced[source] -= 1 if q[source] > 0 else -1
            for destination in range(4):
                if destination == source:
                    continue
                signs = (-1, 1) if q[destination] == 0 else (
                    1 if q[destination] > 0 else -1,
                )
                for sign in signs:
                    candidate = reduced.copy()
                    candidate[destination] += sign
                    neighbors.add(point_to_index[tuple(candidate)])
        adjacency.append(sorted(neighbors))
    return adjacency


def _to_klft_components(points: FloatArray) -> FloatArray:
    """Map paper (x0,x1,x2,x3) coordinates to KLFT SU2::comp order."""
    return points[:, [0, 3, 2, 1]].copy()


def linear_partition(m: int) -> tuple[FloatArray, FloatArray, list[list[int]]]:
    integer_points = _linear_integer_points(m)
    norms = np.linalg.norm(integer_points, axis=1)
    paper_points = integer_points.astype(np.float64) / norms[:, None]
    weights = (math.sqrt(2.0) / norms) ** 3
    return _to_klft_components(paper_points), weights, _linear_neighbors(integer_points)


def _inverse_psi_cdf(t: float) -> float:
    if t == 0.0:
        return 0.0
    return float(brentq(
        lambda psi: (psi - 0.5 * math.sin(2.0 * psi)) / math.pi - t,
        0.0, math.pi, xtol=5.0e-15, rtol=1.0e-14,
    ))


def _convex_hull_neighbors(points: FloatArray) -> list[list[int]]:
    if len(points) < 5:
        raise ValueError("Fibonacci tables require N >= 5")
    hull = ConvexHull(points)
    if len(np.unique(hull.simplices)) != len(points):
        raise ValueError("not every Fibonacci point is a convex-hull vertex")
    adjacency = [set() for _ in range(len(points))]
    for simplex in hull.simplices:
        for i, j in itertools.combinations(simplex, 2):
            adjacency[int(i)].add(int(j))
            adjacency[int(j)].add(int(i))
    return [sorted(row) for row in adjacency]


def fibonacci_partition(n: int) -> tuple[FloatArray, FloatArray, list[list[int]]]:
    if n < 5:
        raise ValueError("Fibonacci N must be >= 5")
    paper_points = np.empty((n, 4), dtype=np.float64)
    for m in range(n):
        psi = _inverse_psi_cdf(m / n)
        theta = math.acos(1.0 - 2.0 * ((m * math.sqrt(2.0)) % 1.0))
        phi = 2.0 * math.pi * ((m * math.sqrt(3.0)) % 1.0)
        sin_psi = math.sin(psi)
        paper_points[m] = (
            math.cos(psi), sin_psi * math.cos(theta),
            sin_psi * math.sin(theta) * math.cos(phi),
            sin_psi * math.sin(theta) * math.sin(phi),
        )
    points = _to_klft_components(paper_points)
    return points, np.ones(n, dtype=np.float64), _convex_hull_neighbors(points)


def _csr(adjacency: list[list[int]]) -> tuple[list[int], list[int]]:
    offsets = [0]
    indices: list[int] = []
    for row in adjacency:
        indices.extend(row)
        offsets.append(len(indices))
    return offsets, indices


def validate_table(table: dict[str, object]) -> None:
    points = np.asarray(table["points"], dtype=np.float64)
    weights = np.asarray(table["weights"], dtype=np.float64)
    graph = table["neighbors"]
    if not isinstance(graph, dict):
        raise ValueError("neighbors must be a mapping")
    offsets = np.asarray(graph["offsets"], dtype=np.int64)
    indices = np.asarray(graph["indices"], dtype=np.int64)
    n = len(points)
    if table.get("format_version") != 1:
        raise ValueError("unsupported format_version")
    if points.shape != (n, 4) or n == 0 or not np.all(np.isfinite(points)):
        raise ValueError("points must be a nonempty finite N x 4 array")
    if not np.allclose(np.linalg.norm(points, axis=1), 1.0, rtol=0.0, atol=2e-13):
        raise ValueError("partition points are not unit normalized")
    if len(np.unique(np.round(points, decimals=14), axis=0)) != n:
        raise ValueError("partition contains duplicate points")
    if weights.shape != (n,) or not np.all(np.isfinite(weights)) or np.any(weights <= 0.0):
        raise ValueError("weights must be finite and positive")
    if offsets.shape != (n + 1,) or offsets[0] != 0 or offsets[-1] != len(indices):
        raise ValueError("invalid CSR offsets")
    if np.any(np.diff(offsets) <= 0) or np.any(indices < 0) or np.any(indices >= n):
        raise ValueError("invalid CSR neighbor rows")
    adjacency = [set(map(int, indices[offsets[i]:offsets[i + 1]])) for i in range(n)]
    for i, row in enumerate(adjacency):
        if i in row or len(row) != offsets[i + 1] - offsets[i]:
            raise ValueError("neighbor rows contain self-edges or duplicates")
        if any(i not in adjacency[j] for j in row):
            raise ValueError("neighbor graph is not reciprocal")
    visited, frontier = {0}, [0]
    while frontier:
        i = frontier.pop()
        for j in adjacency[i] - visited:
            visited.add(j)
            frontier.append(j)
    if len(visited) != n:
        raise ValueError("neighbor graph is disconnected")
    if table.get("kind") == "linear":
        m = int(table["parameter"])
        if n != 8 * (m**3 + 2 * m) // 3:
            raise ValueError("incorrect linear point count")
        lookup = {tuple(np.round(point, 14)): i for i, point in enumerate(points)}
        for i, point in enumerate(points):
            partner = lookup.get(tuple(np.round(-point, 14)))
            if partner is None or not math.isclose(weights[i], weights[partner], rel_tol=2e-14):
                raise ValueError("linear table violates center symmetry")
            partner_neighbors = {
                lookup[tuple(np.round(-points[j], 14))] for j in adjacency[i]
            }
            if partner_neighbors != adjacency[partner]:
                raise ValueError("linear neighbor graph violates center symmetry")


def make_table(kind: str, parameter: int, imported_weights: Path | None) -> dict[str, object]:
    if kind == "linear":
        points, weights, adjacency = linear_partition(parameter)
    else:
        points, weights, adjacency = fibonacci_partition(parameter)
    if imported_weights is not None:
        weights = np.load(imported_weights) if imported_weights.suffix == ".npy" else np.loadtxt(imported_weights)
        weights = np.asarray(weights, dtype=np.float64).reshape(-1)
        if len(weights) != len(points):
            raise ValueError("imported weight count does not match point count")
    offsets, indices = _csr(adjacency)
    table: dict[str, object] = {
        "format_version": 1, "kind": kind, "parameter": parameter,
        "coordinate_convention": "klft_su2_comp_p0_p1_p2_p3",
        "weight_source": "imported" if imported_weights else ("analytic_linear" if kind == "linear" else "uniform"),
        "points": points.tolist(), "weights": weights.tolist(),
        "neighbors": {"offsets": offsets, "indices": indices},
    }
    validate_table(table)
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("kind", choices=("linear", "fibonacci"))
    parser.add_argument("parameter", type=int, help="m for linear or N for Fibonacci")
    parser.add_argument("output", type=Path)
    parser.add_argument("--weights", type=Path)
    args = parser.parse_args()
    table = make_table(args.kind, args.parameter, args.weights)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(table, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
