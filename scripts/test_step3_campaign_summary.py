#!/usr/bin/env python3
"""Deterministic tests for Step-3 campaign-level comparisons."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from step3_campaign_summary import (
    generalized_chi_square,
    weighted_scaling_fit,
    write_tables,
)


class Step3CampaignSummaryTest(unittest.TestCase):
    def test_tables_record_blocking_and_boundary_diagnostics(self) -> None:
        result = {
            "purpose": "test",
            "beta": 2.1,
            "physical_Nt": 3,
            "stored_L2": 4,
            "Ns": 48,
            "lambda_squared_tree": 40.0 / 7.0,
            "xi_2nd": 0.6,
            "xi_2nd_error": None,
            "Ns_over_xi_2nd": 80.0,
            "Ns_over_xi_2nd_error": None,
            "xi_2nd_bootstrap_valid_fraction": 0.9,
            "xi_exp": None,
            "xi_exp_error": None,
            "xi_exp_blocking_error_plateau": None,
            "thermalization_and_blocking": {
                "chains": [
                    {
                        "thermalization_cut_sweeps": 500,
                        "block_length_sweeps": 20,
                        "n_complete_blocks": 475,
                    }
                ],
                "blocking_error_plateau": {
                    "all_primary_and_secondary_plateaus_found": True,
                    "selected_common_block_length_sweeps": 20,
                    "minimum_tau_block_length_sweeps": 10,
                },
            },
        }
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            write_tables([(output / "analysis.json", result)], output)
            with (output / "campaign_summary.csv").open(
                encoding="utf-8", newline=""
            ) as stream:
                row = next(csv.DictReader(stream))
            self.assertEqual(row["blocking_plateau_all_found"], "True")
            self.assertEqual(row["blocking_common_length_sweeps"], "20")
            self.assertEqual(row["xi_exp_mass_blocking_plateau_pass"], "")
            table = np.load(output / "campaign_summary.npz")
            self.assertTrue(np.isnan(table["xi_2nd_error"][0]))
            self.assertEqual(table["blocking_common_length_sweeps"][0], 20)

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
