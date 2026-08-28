#!/usr/bin/env python3
"""Tests for fixed-beta/Nt continuum extrapolations."""

from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile

import numpy as np

from beta_nt_fixed.continuum_summary import (
    BOOTSTRAP_SAMPLES,
    summarize,
    weighted_continuum_fit,
)


class ContinuumSummaryTest(unittest.TestCase):
    def test_weighted_fit_recovers_known_p1_curve(self) -> None:
        nt = np.asarray([2.0, 3.0, 4.0, 5.0])
        errors = np.full(4, 0.02)
        values = 1.7 - 0.6 / nt
        parameters, chi2, dof, p_value = weighted_continuum_fit(
            nt, values, errors
        )
        np.testing.assert_allclose(parameters, [1.7, -0.6], atol=1.0e-12)
        self.assertAlmostEqual(chi2, 0.0, places=20)
        self.assertEqual(dof, 2)
        self.assertAlmostEqual(p_value, 1.0)

    def test_fit_requires_three_points(self) -> None:
        values = np.ones(2)
        with self.assertRaises(ValueError):
            weighted_continuum_fit(values, values, values)

    def test_summary_uses_only_selected_analyzed_points(self) -> None:
        rng = np.random.default_rng(1234)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection.tsv"
            rows = [
                "beta_over_nt\tg3_squared_L\tphysical_Nt\tbeta\tNs\trole\tanalysis_relative_path"
            ]
            for nt in (2, 3, 4):
                relative = f"line/nt{nt}"
                rows.append(
                    f"1.8\t2.222222222222\t{nt}\t{1.8 * nt}\t128\tcontinuum\t{relative}"
                )
                directory = root / "analysis" / relative
                directory.mkdir(parents=True)
                mass = (1.5 - 0.4 / nt) / nt
                xi_exp = 1.0 / mass
                result = {
                    "mass": mass,
                    "mass_error": 0.01 / nt,
                    "xi_exp": xi_exp,
                    "xi_2nd": 0.95 * xi_exp,
                }
                (directory / "analysis.json").write_text(
                    json.dumps(result), encoding="utf-8"
                )
                mass_samples = rng.normal(
                    mass, 0.01 / nt, size=BOOTSTRAP_SAMPLES
                )
                xi_exp_samples = 1.0 / mass_samples
                np.savez(
                    directory / "covariance.npz",
                    mass_bootstrap=mass_samples,
                    xi_exp_bootstrap=xi_exp_samples,
                    xi_2nd_bootstrap=0.95 * xi_exp_samples,
                )
            selection.write_text("\n".join(rows) + "\n", encoding="utf-8")
            summary = summarize(
                selection,
                root / "analysis",
                root / "continuum.npz",
                allow_incomplete=False,
            )
            fit = summary["lines"]["1.8"]["mass_Nt_p1_fit"]
            self.assertAlmostEqual(fit["continuum"], 1.5, places=12)
            self.assertEqual(fit["bootstrap_samples"], BOOTSTRAP_SAMPLES)
            self.assertNotIn("mass_Nt_fits", summary["lines"]["1.8"])
            self.assertTrue((root / "continuum.npz").is_file())
            with np.load(root / "continuum.npz") as archive:
                self.assertEqual(
                    archive["mass_line_1p8_p1"].shape,
                    (BOOTSTRAP_SAMPLES, 2),
                )

    def test_summary_rejects_fewer_than_1000_finite_replicas(self) -> None:
        rng = np.random.default_rng(5678)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            selection = root / "selection.tsv"
            rows = [
                "beta_over_nt\tg3_squared_L\tphysical_Nt\tbeta\tNs\trole\tanalysis_relative_path"
            ]
            for nt in (2, 3, 4):
                relative = f"line/nt{nt}"
                rows.append(
                    f"1.8\t2.222222222222\t{nt}\t{1.8 * nt}\t128\tcontinuum\t{relative}"
                )
                directory = root / "analysis" / relative
                directory.mkdir(parents=True)
                mass = (1.5 - 0.4 / nt) / nt
                xi_exp = 1.0 / mass
                (directory / "analysis.json").write_text(
                    json.dumps(
                        {
                            "mass": mass,
                            "mass_error": 0.01 / nt,
                            "xi_exp": xi_exp,
                            "xi_2nd": 0.95 * xi_exp,
                        }
                    ),
                    encoding="utf-8",
                )
                mass_samples = rng.normal(
                    mass, 0.01 / nt, size=BOOTSTRAP_SAMPLES - 1
                )
                xi_exp_samples = 1.0 / mass_samples
                np.savez(
                    directory / "covariance.npz",
                    mass_bootstrap=mass_samples,
                    xi_exp_bootstrap=xi_exp_samples,
                    xi_2nd_bootstrap=0.95 * xi_exp_samples,
                )
            selection.write_text("\n".join(rows) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "at least 1000 finite"):
                summarize(
                    selection,
                    root / "analysis",
                    root / "continuum.npz",
                    allow_incomplete=False,
                )


if __name__ == "__main__":
    unittest.main()
