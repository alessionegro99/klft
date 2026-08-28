#!/usr/bin/env python3
"""Checks for audit, qbig staging, and confirmation gates."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

from mg_nt_fixed_audit import audit
from mg_nt_fixed_confirmation import evaluate
from beta_nt_fixed.analysis import k0_r_max_values


ROOT = Path(__file__).resolve().parents[1]
CONFIRMATION_GENERATOR = (
    ROOT / "data" / "mg_nt_fixed" / "confirmation" / "generate_inputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "mg_nt_fixed_confirmation_inputs", CONFIRMATION_GENERATOR
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(CONFIRMATION_GENERATOR)
confirmation_inputs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = confirmation_inputs
SPEC.loader.exec_module(confirmation_inputs)


def tuning_summary() -> dict[str, object]:
    """Return the minimal accepted summary needed by confirmation code."""
    return {
        "all_nine_tunings_accepted": True,
        "confirmation_Nt4_c1_beta": 4.987654321,
        "groups": [
            {
                "target_c": 1.0,
                "physical_Nt": 4,
                "accepted": True,
                "fit": {"b": -0.8},
                "beta_star": {"statistical_16_84": [4.98, 5.00]},
            }
        ],
    }


class MgNtFixedWorkflowTest(unittest.TestCase):
    def test_fixed_mass_scan_reaches_ns_third_endpoint(self) -> None:
        self.assertEqual(k0_r_max_values(48, True), [24, 22, 20, 18, 16])
        self.assertEqual(k0_r_max_values(48, False), [24, 22, 20])

    def test_confirmation_generator_uses_tuned_beta_and_reserved_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "tuning_summary.json"
            summary.write_text(json.dumps(tuning_summary()) + "\n", encoding="utf-8")
            output = root / "campaign"
            confirmation_inputs.generate(summary, output)
            manifest = (output / "confirmation_manifest.tsv").read_text(
                encoding="utf-8"
            )
            self.assertEqual(manifest.count("\n"), 5)
            self.assertIn("4.98765432", manifest)
            for seed in confirmation_inputs.pilot.CONFIRMATION_SEEDS:
                self.assertIn(str(seed), manifest)
            self.assertEqual(
                len(list((output / "confirmation").rglob("input.yaml"))), 4
            )
            with self.assertRaises(FileExistsError):
                confirmation_inputs.generate(summary, output)

    def test_confirmation_uses_wider_one_sigma_or_three_percent_gate(self) -> None:
        accepted = evaluate(
            tuning_summary(), {"mass": 1.02 / 4.0, "mass_error": 0.005 / 4.0, "beta": 4.99}
        )
        rejected = evaluate(
            tuning_summary(), {"mass": 1.05 / 4.0, "mass_error": 0.005 / 4.0, "beta": 4.99}
        )
        self.assertTrue(accepted["accepted"])
        self.assertAlmostEqual(accepted["acceptance_threshold"], 0.03)
        self.assertFalse(rejected["accepted"])

    def test_prior_mass_audit_uses_combined_error(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            project = root / "project"
            current = root / "audit"
            ensemble = "Nt_2_Ns_64_b_3p0"
            prior_file = root / "prior.tsv"
            prior_file.write_text(
                "prior_index\ttarget_c\tphysical_Nt\tstored_L2\tNs\tbeta\tsweeps\t"
                "ensemble\tseeds\tsource_campaign\tsource_analysis\n"
                f"0\t0.5\t2\t3\t64\t3.0\t10000\t{ensemble}\t1,2,3,4\t"
                "beta_nt_fixed\tanalysis/old\n",
                encoding="utf-8",
            )
            old = project / "analysis" / "old"
            new = current / ensemble
            old.mkdir(parents=True)
            new.mkdir(parents=True)
            (old / "analysis.json").write_text(
                json.dumps({"mass": 0.25, "mass_error": 0.01}), encoding="utf-8"
            )
            (new / "analysis.json").write_text(
                json.dumps({"mass": 0.26, "mass_error": 0.02}), encoding="utf-8"
            )
            result = audit(prior_file, project, current)
            self.assertTrue(result["all_pass"])
            self.assertAlmostEqual(
                result["comparisons"][0]["difference_combined_sigma"],
                0.01 / (0.01**2 + 0.02**2) ** 0.5,
            )

    def test_qbig_arrays_are_uncapped_dependency_free_and_typed(self) -> None:
        qbig = ROOT / "data" / "mg_nt_fixed" / "pilot" / "qbig"
        production = (qbig / "submit_production.slurm").read_text(encoding="utf-8")
        timing = (qbig / "submit_timing.slurm").read_text(encoding="utf-8")
        diagnostics = (qbig / "submit_diagnostics.slurm").read_text(encoding="utf-8")
        analysis = (qbig / "submit_analysis.slurm").read_text(encoding="utf-8")
        confirmation = (qbig / "submit_confirmation_production.slurm").read_text(
            encoding="utf-8"
        )
        confirmation_gate = (qbig / "submit_confirmation_gate.slurm").read_text(
            encoding="utf-8"
        )
        queue = (qbig / "queue_stage.sh").read_text(encoding="utf-8")
        self.assertIn("#SBATCH --array=0-99\n", production)
        self.assertNotIn("#SBATCH --array=0-99%", production)
        self.assertNotIn("--nodelist", production)
        self.assertIn("#SBATCH --gres=gpu:pascal:1\n", production)
        self.assertIn("#SBATCH --gres=gpu:pascal:1\n", timing)
        self.assertIn("mass-analysis audit gate did not pass", production)
        self.assertIn("maximum <= 2.0 && total <= 12.0", timing)
        for script in (diagnostics, analysis):
            self.assertIn("#SBATCH --array=0-24\n", script)
            self.assertNotIn("#SBATCH --array=0-24%", script)
        self.assertNotIn("--dependency", queue)
        self.assertNotIn("afterok", queue)
        self.assertIn("tuning_final/tuning_summary.json", confirmation)
        self.assertIn("tuning_final/tuning_summary.json", confirmation_gate)


if __name__ == "__main__":
    unittest.main()
