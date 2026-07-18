#!/usr/bin/env python3
"""Deterministic tests for thermalization-cut diagnostics."""

from __future__ import annotations

import unittest

import numpy as np

from thermalization_diagnostics import mser5_cut


class ThermalizationDiagnosticsTest(unittest.TestCase):
    def test_constant_series_needs_no_deletion(self) -> None:
        self.assertEqual(mser5_cut(np.ones(100, dtype=np.float64)), 0)

    def test_initial_drift_is_deleted_in_batches(self) -> None:
        drift = np.linspace(8.0, 0.0, 100, dtype=np.float64)
        stationary = 0.02 * np.sin(np.arange(400, dtype=np.float64))
        cut = mser5_cut(np.concatenate((drift, stationary)))
        self.assertEqual(cut % 5, 0)
        self.assertGreaterEqual(cut, 75)
        self.assertLess(cut, 200)


if __name__ == "__main__":
    unittest.main()
