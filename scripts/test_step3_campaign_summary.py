#!/usr/bin/env python3
"""Deterministic tests for Step-3 campaign-level comparisons."""

from __future__ import annotations

import unittest

import numpy as np

from step3_campaign_summary import generalized_chi_square, weighted_scaling_fit


class Step3CampaignSummaryTest(unittest.TestCase):
    def test_generalized_chi_square_uses_resolved_covariance_modes(self) -> None:
        result = generalized_chi_square(
            np.zeros(3, dtype=np.float64),
            np.diag(np.asarray([1.0, 2.0, 0.0], dtype=np.float64)),
        )
        self.assertEqual(result["covariance_rank"], 2)
        self.assertEqual(result["chi2"], 0.0)
        self.assertEqual(result["p_value"], 1.0)

    def test_weighted_scaling_fit_recovers_linear_inverse_nt_form(self) -> None:
        nt = np.asarray([2.0, 3.0, 4.0], dtype=np.float64)
        expected_q0 = 1.25
        expected_c = -0.6
        quantity = expected_q0 + expected_c / nt
        result = weighted_scaling_fit(
            nt, quantity, np.full(3, 0.02, dtype=np.float64)
        )
        self.assertAlmostEqual(result["q0"], expected_q0)
        self.assertAlmostEqual(result["c"], expected_c)
        self.assertAlmostEqual(result["chi2"], 0.0, places=25)


if __name__ == "__main__":
    unittest.main()
