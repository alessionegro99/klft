#!/usr/bin/env python3
"""Tests for the immutable fixed-beta/Nt ``Ns=160`` volume check."""

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
    / "beta_over_nt_1p8"
    / "validation"
    / "finite_volume_nt2"
    / "generate_inputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "beta_nt_fixed_finite_volume_nt2_inputs", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
volume_check = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = volume_check
SPEC.loader.exec_module(volume_check)


class BetaNtFixedFiniteVolumeNt2GenerateInputsTest(unittest.TestCase):
    def test_geometry_cadence_hashes_and_unique_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "finite_volume_nt2"
            volume_check.generate(output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            seeds = {int(row["seed"]) for row in rows}
            self.assertEqual(len(seeds), 4)
            self.assertFalse(seeds & volume_check.previous_pilot_seeds())
            self.assertNotIn(volume_check.SMOKE_SEED, seeds)
            for row in rows:
                self.assertEqual(row["physical_Nt"], "2")
                self.assertEqual(row["stored_L2"], "3")
                self.assertEqual(row["Ns"], "160")
                self.assertEqual(row["beta"], "3.6")
                text = (output / row["input"]).read_text(encoding="utf-8")
                self.assertEqual(
                    volume_check.pilot.sha256_text(text), row["input_sha256"]
                )
                self.assertIn("  nSweep: 10000\n", text)
                self.assertIn("  measurement_interval: 1\n", text)
                self.assertIn(
                    "  dirichlet_suite_measurement_interval: 5\n", text
                )
                self.assertIn("  measure_boundary_wilson_loops: false\n", text)

            smoke = (output / "smoke" / "input.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("  L0: 160\n", smoke)
            self.assertIn("  L2: 3\n", smoke)
            self.assertIn("  nSweep: 100\n", smoke)

    def test_refuses_overwrite_before_partial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "finite_volume_nt2"
            volume_check.generate(output)
            original = (output / "manifest.tsv").read_bytes()
            with self.assertRaises(FileExistsError):
                volume_check.generate(output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), original)

    def test_qbig_scripts_use_pascal_without_dependencies(self) -> None:
        qbig = (
            ROOT
            / "data"
            / "beta_nt_fixed"
            / "beta_over_nt_1p8"
            / "validation"
            / "finite_volume_nt2"
            / "qbig"
        )
        queue = (qbig / "queue_stage.sh").read_text(encoding="utf-8")
        production = (qbig / "submit_production.slurm").read_text(
            encoding="utf-8"
        )
        smoke = (qbig / "submit_smoke.slurm").read_text(encoding="utf-8")
        self.assertNotIn("--dependency", queue)
        self.assertNotIn("afterok", queue)
        self.assertIn("#SBATCH --array=0-3\n", production)
        self.assertNotIn("#SBATCH --array=0-3%", production)
        for script in (production, smoke):
            self.assertIn("#SBATCH --gres=gpu:pascal:1\n", script)
            self.assertIn('build="${remote_root}/build/klft"', script)
            self.assertIn("refusing to append", script)


if __name__ == "__main__":
    unittest.main()
