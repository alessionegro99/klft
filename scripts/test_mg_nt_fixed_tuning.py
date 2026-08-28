#!/usr/bin/env python3
"""Deterministic checks for fixed-mass beta interpolation gates."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from mg_nt_fixed_tuning import Point, load_points, resolve_replacements, tune_group


class MgNtFixedTuningTest(unittest.TestCase):
    @staticmethod
    def _point(beta: float, mass_nt: float, target: float = 1.0) -> Point:
        nt = 4
        mass = mass_nt / nt
        error = 0.01 * mass
        rng = np.random.default_rng(round(1000 * beta))
        samples = rng.normal(mass, error, 1000)
        result: dict[str, object] = {
            "mass": mass,
            "mass_error": error,
            "thermalization_and_blocking": {
                "chains": [{"n_complete_blocks": 50}] * 4
            },
        }
        return Point(
            target,
            nt,
            48,
            beta,
            f"beta_{beta:g}",
            "mg_nt_fixed",
            Path(f"beta_{beta:g}/analysis.json"),
            result,
            samples,
        )

    def test_log_linear_bootstrap_recovers_known_target(self) -> None:
        expected_beta = 5.0
        points = [
            self._point(beta, float(np.exp(-0.8 * (beta - expected_beta))))
            for beta in (4.6, 5.0, 5.4)
        ]
        record, samples = tune_group(points, bootstrap_seed=20260724)
        self.assertTrue(record["accepted"])
        self.assertEqual(len(samples), 1000)
        self.assertAlmostEqual(record["beta_star"]["median"], expected_beta, delta=0.02)
        self.assertGreaterEqual(record["fit"]["p_value"], 0.05)

    def test_unbracketed_target_requests_another_point(self) -> None:
        points = [self._point(beta, mass_nt) for beta, mass_nt in (
            (4.6, 1.8), (5.0, 1.5), (5.4, 1.2)
        )]
        record, samples = tune_group(points, bootstrap_seed=20260724)
        self.assertFalse(record["accepted"])
        self.assertTrue(record["add_bracket_point"])
        self.assertIn("target is not bracketed", record["failure_reasons"])
        self.assertEqual(len(samples), 0)

    def test_missing_mass_is_a_hard_gate(self) -> None:
        points = [self._point(beta, mass_nt) for beta, mass_nt in (
            (4.6, 1.3), (5.0, 1.0), (5.4, 0.7)
        )]
        points[1].result["mass"] = None
        points[1].result["mass_error"] = None
        record, _ = tune_group(points, bootstrap_seed=20260724)
        self.assertFalse(record["accepted"])
        self.assertTrue(any("no accepted asymptotic mass" in reason for reason in record["failure_reasons"]))

    def test_four_points_select_closest_contiguous_bracketing_triple(self) -> None:
        expected_beta = 5.0
        points = [
            self._point(beta, float(np.exp(-0.8 * (beta - expected_beta))))
            for beta in (4.2, 4.6, 5.0, 5.4)
        ]
        record, samples = tune_group(points, bootstrap_seed=20260724)
        self.assertTrue(record["accepted"])
        self.assertEqual(
            record["local_point_selection"]["available_beta_values"],
            [4.2, 4.6, 5.0, 5.4],
        )
        self.assertEqual(
            record["local_point_selection"]["selected_beta_values"],
            [4.6, 5.0, 5.4],
        )
        self.assertEqual(len(samples), 1000)

    def test_extension_manifests_and_results_are_paired(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be supplied together"):
            load_points(
                Path("pilot.tsv"),
                Path("prior.tsv"),
                Path("results"),
                Path("audit"),
                [Path("round1.tsv")],
                [],
            )

    def test_explicit_replacement_supersedes_duplicate_coupling(self) -> None:
        original = self._point(5.0, 1.0)
        replacement = self._point(5.0, 0.98)
        replacement.result["purpose"] = "statistics_replacement"
        self.assertEqual(resolve_replacements([original, replacement]), [replacement])

    def test_unlabelled_duplicate_coupling_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "not an explicit replacement"):
            resolve_replacements([self._point(5.0, 1.0), self._point(5.0, 0.98)])


if __name__ == "__main__":
    unittest.main()
