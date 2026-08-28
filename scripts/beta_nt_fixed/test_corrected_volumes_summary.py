#!/usr/bin/env python3
"""Deterministic tests for the corrected fixed-beta/Nt volume summary."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beta_nt_fixed.corrected_volumes_summary import summarize


class BetaNtFixedCorrectedVolumesSummaryTest(unittest.TestCase):
    @staticmethod
    def _result(nt: int, ns: int, xi: float, purpose: str) -> dict[str, object]:
        error = 0.2
        return {
            "purpose": purpose,
            "physical_Nt": nt,
            "stored_L2": nt + 1,
            "Ns": ns,
            "beta": 1.8 * nt,
            "lambda_squared_tree": 20.0 / 9.0,
            "xi_2nd": xi,
            "xi_2nd_error": error,
            "xi_2nd_68_percent_interval": [xi - error, xi + error],
            "Ns_over_xi_2nd": ns / xi,
            "Ns_over_xi_2nd_error": error,
            "Ns_over_xi_2nd_68_percent_interval": [
                ns / (xi + error),
                ns / (xi - error),
            ],
            "xi_exp": None,
            "xi_exp_error": None,
        }

    @staticmethod
    def _write_result(directory: Path, result: dict[str, object]) -> Path:
        directory.mkdir(parents=True)
        path = directory / "analysis.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        separation = np.arange(5, dtype=np.int64)
        correlator = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])
        covariance = np.diag([0.0, 0.01, 0.01, 0.01, 0.01])
        np.savez(
            path.with_name("covariance.npz"),
            R=separation,
            C_G=correlator,
            C_G_covariance=covariance,
        )
        return path

    def test_summary_uses_corrected_points_for_scaling(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = root / "pilot"
            finite_volume = root / "finite_volume"
            corrected = root / "corrected"
            self._write_result(
                pilot / "Nt_2_Ns_128_b_3p6",
                self._result(2, 128, 10.0, "anchor_finite_volume"),
            )
            self._write_result(
                pilot / "Nt_3_Ns_144_b_5p4",
                self._result(3, 144, 15.0, "constant_lambda_nt3"),
            )
            self._write_result(
                pilot / "Nt_4_Ns_192_b_7p2",
                self._result(4, 192, 20.0, "constant_lambda_nt4"),
            )
            self._write_result(
                corrected / "Nt_3_Ns_176_b_5p4",
                self._result(
                    3, 176, 18.0, "constant_lambda_nt3_volume_corrected"
                ),
            )
            self._write_result(
                corrected / "Nt_4_Ns_256_b_7p2",
                self._result(
                    4, 256, 24.0, "constant_lambda_nt4_volume_corrected"
                ),
            )
            finite_volume.mkdir()
            (finite_volume / "finite_volume_summary.json").write_text(
                json.dumps({"decision": {"finite_volume_gate_pass": True}}),
                encoding="utf-8",
            )

            result = summarize(pilot, finite_volume, corrected, corrected)

            fit = result["corrected_constant_lambda_scaling"][
                "xi_2nd_over_Nt"
            ]
            np.testing.assert_allclose(fit["q"], [5.0, 6.0, 6.0])
            self.assertTrue(
                result["decision"]["all_corrected_volume_ratio_gates_pass"]
            )
            self.assertTrue(
                result["pilot_vs_corrected_volume"][
                    "Nt3_Ns144_vs_Ns176"
                ]["full_correlator_pass"]
            )
            self.assertTrue(
                (corrected / "corrected_volume_summary.json").is_file()
            )
            self.assertTrue(
                (corrected / "corrected_volume_points.csv").is_file()
            )
            arrays = np.load(corrected / "correlator_comparison_Nt4.npz")
            np.testing.assert_array_equal(arrays["common_R"], [1, 2, 3, 4])


if __name__ == "__main__":
    unittest.main()
