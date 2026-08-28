#!/usr/bin/env python3
"""Deterministic tests for the immutable fixed-beta/Nt pilot generator."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = (
    ROOT
    / "data"
    / "beta_nt_fixed"
    / "comparison"
    / "provenance"
    / "legacy_pilot_inputs"
    / "generate_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("beta_nt_fixed_pilot_inputs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
pilot = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = pilot
SPEC.loader.exec_module(pilot)


class BetaNtFixedPilotGenerateInputsTest(unittest.TestCase):
    def test_qbig_stage_submission_has_no_dependencies(self) -> None:
        submission = (
            ROOT
            / "data"
            / "beta_nt_fixed"
            / "comparison"
            / "provenance"
            / "legacy_pilot_inputs"
            / "qbig"
            / "queue_all.sh"
        ).read_text(encoding="utf-8")
        self.assertNotIn("--dependency", submission)
        self.assertNotIn("afterok", submission)
        self.assertIn("analysis) script=submit_analysis.slurm", submission)
        self.assertIn('printf \'%s\\t%s\\tnone\\n\'', submission)

    def test_qbig_analysis_uses_only_manifested_replicas(self) -> None:
        wrapper = (
            ROOT
            / "data"
            / "beta_nt_fixed"
            / "comparison"
            / "provenance"
            / "legacy_pilot_inputs"
            / "qbig"
            / "analyze_ensemble.sh"
        ).read_text(encoding="utf-8")
        self.assertIn("mapfile -t manifest_runs", wrapper)
        self.assertIn("--chain-directory", wrapper)
        self.assertNotIn('for directory in "${data}"/seed_*', wrapper)

    def test_grid_geometry_cadence_and_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            pilot.generate(output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))

            self.assertEqual(len(rows), 32)
            self.assertEqual(
                len({row["seed"] for row in rows}), 32
            )
            self.assertEqual(
                len({row["run"].rsplit("/seed_", 1)[0] for row in rows}), 8
            )
            for row in rows:
                self.assertEqual(
                    int(row["stored_L2"]), int(row["physical_Nt"]) + 1
                )
                input_path = output / row["input"]
                text = input_path.read_text(encoding="utf-8")
                self.assertEqual(pilot.sha256_text(text), row["input_sha256"])
                self.assertIn("  measurement_interval: 1\n", text)
                self.assertIn(
                    "  dirichlet_suite_measurement_interval: 5\n", text
                )
                self.assertIn(
                    "  measure_dirichlet_plaquette_profiles: true\n", text
                )
                self.assertIn("  measure_boundary_wilson_loops: false\n", text)
                self.assertIn("  measure_wilson_loop_temporal: false\n", text)
                self.assertIn("  measure_wilson_loop_mu_nu: false\n", text)

            timing = (output / "timing" / "input.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("  L0: 192\n", timing)
            self.assertIn("  L2: 5\n", timing)
            self.assertIn("  nSweep: 100\n", timing)

    def test_generation_refuses_overwrites_without_partial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            pilot.generate(output)
            original = (output / "manifest.tsv").read_bytes()
            with self.assertRaises(FileExistsError):
                pilot.generate(output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
