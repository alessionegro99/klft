#!/usr/bin/env python3
"""Deterministic tests for the fixed-beta/Nt ``Ns=160`` decision."""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from beta_nt_fixed.finite_volume_nt2_summary import summarize


class BetaNtFixedFiniteVolumeNt2SummaryTest(unittest.TestCase):
    @staticmethod
    def _result(nt: int, ns: int, xi: float, purpose: str) -> dict[str, object]:
        return {
            "purpose": purpose,
            "physical_Nt": nt,
            "stored_L2": nt + 1,
            "Ns": ns,
            "beta": 3.6 if nt == 2 else 1.8 * nt,
            "lambda_squared_tree": 20.0 / 9.0,
            "xi_2nd": xi,
            "xi_2nd_error": 0.2,
            "xi_2nd_68_percent_interval": [xi - 0.2, xi + 0.2],
            "Ns_over_xi_2nd": ns / xi,
            "Ns_over_xi_2nd_error": 0.2,
            "Ns_over_xi_2nd_68_percent_interval": [
                ns / (xi + 0.2),
                ns / (xi - 0.2),
            ],
            "xi_exp": None,
            "xi_exp_error": None,
        }

    @staticmethod
    def _write_result(directory: Path, result: dict[str, object]) -> Path:
        directory.mkdir(parents=True)
        path = directory / "analysis.json"
        path.write_text(json.dumps(result), encoding="utf-8")
        return path

    def test_summary_accepts_stable_ns128_volume(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            pilot = root / "pilot"
            extension = root / "extension"
            output = root / "output"
            self._write_result(
                pilot / "Nt_2_Ns_96_b_3p6",
                self._result(2, 96, 10.0, "constant_lambda_anchor"),
            )
            ns128_path = self._write_result(
                pilot / "Nt_2_Ns_128_b_3p6",
                self._result(2, 128, 10.0, "anchor_finite_volume"),
            )
            self._write_result(
                pilot / "Nt_3_Ns_144_b_5p4",
                self._result(3, 144, 16.0, "constant_lambda_nt3"),
            )
            self._write_result(
                pilot / "Nt_4_Ns_192_b_7p2",
                self._result(4, 192, 24.0, "constant_lambda_nt4"),
            )
            ns160_path = self._write_result(
                extension / "Nt_2_Ns_160_b_3p6",
                self._result(2, 160, 10.1, "anchor_finite_volume_ns160"),
            )
            separation = np.arange(5, dtype=np.int64)
            correlator = np.asarray([1.0, 0.5, 0.25, 0.125, 0.0625])
            covariance = np.diag(np.asarray([0.0, 0.01, 0.01, 0.01, 0.01]))
            for path in (ns128_path, ns160_path):
                np.savez(
                    path.with_name("covariance.npz"),
                    R=separation,
                    C_G=correlator,
                    C_G_covariance=covariance,
                )

            result = summarize(pilot, extension, output)

            comparison = result[
                "finite_volume_comparison_Ns128_vs_Ns160"
            ]
            self.assertTrue(comparison["smaller_volume_accepted"])
            self.assertEqual(
                result["decision"]["selected_Nt2_spatial_volume"], 128
            )
            self.assertTrue((output / "finite_volume_points.csv").is_file())
            self.assertTrue(
                (output / "finite_volume_comparison.npz").is_file()
            )
            arrays = np.load(output / "finite_volume_comparison.npz")
            np.testing.assert_array_equal(arrays["common_R"], [1, 2, 3, 4])
            np.testing.assert_allclose(
                arrays["C_G_difference_Ns128_minus_Ns160"], 0.0
            )


if __name__ == "__main__":
    unittest.main()
