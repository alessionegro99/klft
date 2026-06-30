import collections
import math
import unittest

import numpy as np

try:
    from .generate_partitioning import fibonacci_partition, linear_partition
except ImportError:
    from generate_partitioning import fibonacci_partition, linear_partition


class LinearPartitionTests(unittest.TestCase):
    def test_counts_center_and_degrees(self) -> None:
        expected_degrees = {
            1: {6: 8},
            2: {6: 8, 10: 24},
            3: {6: 8, 10: 48, 12: 32},
        }
        for m, degree_counts in expected_degrees.items():
            with self.subTest(m=m):
                points, weights, adjacency = linear_partition(m)
                self.assertEqual(len(points), 8 * (m**3 + 2 * m) // 3)
                np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0,
                                           rtol=0.0, atol=2e-15)
                lookup = {tuple(point): i for i, point in enumerate(points)}
                for i, point in enumerate(points):
                    j = lookup[tuple(-point)]
                    self.assertEqual(weights[i], weights[j])
                    self.assertEqual({lookup[tuple(-points[k])] for k in adjacency[i]},
                                     set(adjacency[j]))
                self.assertEqual(dict(collections.Counter(map(len, adjacency))),
                                 degree_counts)


class FibonacciPartitionTests(unittest.TestCase):
    def test_volume_map_and_uniform_weights(self) -> None:
        n = 88
        points, weights, adjacency = fibonacci_partition(n)
        self.assertEqual(points.shape, (n, 4))
        np.testing.assert_allclose(np.linalg.norm(points, axis=1), 1.0,
                                   rtol=0.0, atol=2e-14)
        np.testing.assert_array_equal(weights, np.ones(n))
        for m, point in enumerate(points):
            psi = math.acos(np.clip(point[0], -1.0, 1.0))
            cdf = (psi - 0.5 * math.sin(2.0 * psi)) / math.pi
            self.assertAlmostEqual(cdf, m / n, places=13)
            self.assertGreater(len(adjacency[m]), 0)
            for j in adjacency[m]:
                self.assertIn(m, adjacency[j])


if __name__ == "__main__":
    unittest.main()
