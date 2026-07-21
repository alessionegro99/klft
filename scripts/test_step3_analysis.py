#!/usr/bin/env python3
"""Deterministic checks for the Step-3 statistical acceptance rules."""

from __future__ import annotations

import unittest

import numpy as np

from step3_analysis import (
    choose_block_length_sweeps,
    finite_volume_k0,
    scan_k0_fits,
    second_moment_samples,
)


class Step3AnalysisTest(unittest.TestCase):
    def test_block_length_is_aligned_and_strictly_longer_than_two_tau(self) -> None:
        for tau in (0.5, 2.5, 7.1, 20.0):
            length = choose_block_length_sweeps(tau)
            self.assertEqual(length % 5, 0)
            self.assertGreater(length, 2.0 * tau)

    def test_second_moment_length_uses_ensemble_structure_factors(self) -> None:
        ns = 96
        expected = np.asarray([5.0, 8.0], dtype=np.float64)
        ratio = (2.0 * np.sin(np.pi / ns) * expected) ** 2
        samples = np.zeros((2, 4), dtype=np.float64)
        samples[:, 2] = 1.0 + ratio
        samples[:, 3] = 1.0
        np.testing.assert_allclose(second_moment_samples(samples, ns), expected)

    def test_correlated_k0_scan_accepts_three_stable_adjacent_ranges(self) -> None:
        rng = np.random.default_rng(20260721)
        ns = 32
        separations = np.arange(ns // 2 + 1, dtype=np.int64)
        correlator = np.empty_like(separations, dtype=np.float64)
        correlator[0] = 1.0
        correlator[1:] = finite_volume_k0(
            separations[1:].astype(np.float64), 0.7, 0.2, ns
        )
        errors = np.full(len(separations), 2.0e-4, dtype=np.float64)
        errors[0] = 0.0
        covariance = np.diag(errors * errors)
        samples = np.broadcast_to(correlator, (300, len(correlator))).copy()
        samples[:, 1:] += rng.normal(
            scale=errors[1:], size=(len(samples), len(correlator) - 1)
        )
        xi, xi_error, scan = scan_k0_fits(
            separations, correlator, covariance, samples, ns
        )
        self.assertIsNotNone(xi)
        self.assertIsNotNone(xi_error)
        self.assertAlmostEqual(float(xi), 5.0, delta=0.05)
        self.assertTrue(any(entry.get("selected", False) for entry in scan))


if __name__ == "__main__":
    unittest.main()
