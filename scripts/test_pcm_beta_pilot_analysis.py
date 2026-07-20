#!/usr/bin/env python3
"""Deterministic tests for chain-aware PCM-pilot resampling."""

from __future__ import annotations

import unittest

import numpy as np

from dirichlet_analysis import block_values
from pcm_beta_pilot_analysis import hierarchical_bootstrap, second_moment_length


class PcmBetaPilotAnalysisTest(unittest.TestCase):
    def test_blocks_never_cross_chain_boundaries(self) -> None:
        left = block_values(np.zeros((7, 1), dtype=np.float64), 3)
        right = block_values(np.full((7, 1), 10.0, dtype=np.float64), 3)
        np.testing.assert_array_equal(left[:, 0], [0.0, 0.0])
        np.testing.assert_array_equal(right[:, 0], [10.0, 10.0])

        # Pooling raw histories first would make the third block straddle the
        # artificial chain boundary and produce 20/3. This is forbidden.
        unsafe = block_values(
            np.concatenate((np.zeros((7, 1)), np.full((7, 1), 10.0))), 3
        )
        self.assertIn(20.0 / 3.0, unsafe[:, 0])

    def test_hierarchical_bootstrap_preserves_vector_covariance(self) -> None:
        first = np.asarray([[1.0, 2.0], [2.0, 4.0], [3.0, 6.0]])
        second = np.asarray([[4.0, 8.0], [5.0, 10.0], [6.0, 12.0]])
        estimate, samples, covariance = hierarchical_bootstrap(
            [first, second],
            lambda mean: mean,
            n_resamples=2000,
            rng=np.random.default_rng(12345),
        )
        np.testing.assert_allclose(estimate, [3.5, 7.0])
        np.testing.assert_allclose(samples[:, 1], 2.0 * samples[:, 0])
        self.assertAlmostEqual(covariance[1, 1], 4.0 * covariance[0, 0])

    def test_second_moment_length(self) -> None:
        ns = 96
        expected = 7.0
        ratio = (2.0 * np.sin(np.pi / ns) * expected) ** 2
        summary = np.asarray([0.0, 0.0, 1.0 + ratio, 1.0])
        self.assertAlmostEqual(second_moment_length(summary, ns), expected)


if __name__ == "__main__":
    unittest.main()
