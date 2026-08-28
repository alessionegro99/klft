#!/usr/bin/env python3
"""Tests for fixed-beta/Nt first-stage diagnostic plots."""

from __future__ import annotations

import unittest

import numpy as np

from beta_nt_fixed.plot_first_stage import binned_means, fallback_blocking_curve


class FirstStagePlotTest(unittest.TestCase):
    def test_presentation_bins_are_nonoverlapping_means(self) -> None:
        steps = np.arange(1, 9, dtype=np.float64)
        values = np.arange(8, dtype=np.float64)
        centers, means = binned_means(steps, values, 4)
        np.testing.assert_allclose(centers, [2.5, 6.5])
        np.testing.assert_allclose(means, [1.5, 5.5])

    def test_pcm_fallback_keeps_chains_separate(self) -> None:
        rng = np.random.default_rng(20260728)
        chains = [rng.normal(size=3000) for _ in range(4)]
        lengths, errors, uncertainties = fallback_blocking_curve(chains)
        self.assertGreaterEqual(len(lengths), 3)
        self.assertEqual(lengths.shape, errors.shape)
        self.assertEqual(lengths.shape, uncertainties.shape)
        self.assertTrue(np.all(errors > 0.0))
        self.assertTrue(np.all(uncertainties > 0.0))


if __name__ == "__main__":
    unittest.main()
