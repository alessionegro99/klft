#!/usr/bin/env python3
"""Tests for the immutable Step-3 ``Ns=160`` extension."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data" / "step3_fv160" / "generate_inputs.py"
SPEC = importlib.util.spec_from_file_location("step3_fv160_inputs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
fv160 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = fv160
SPEC.loader.exec_module(fv160)


class Step3Fv160GenerateInputsTest(unittest.TestCase):
    def test_geometry_cadence_hashes_and_unique_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "step3_fv160"
            fv160.generate(output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            seeds = {int(row["seed"]) for row in rows}
            self.assertEqual(len(seeds), 4)
            self.assertFalse(seeds & fv160.previous_step3_seeds())
            self.assertNotIn(fv160.SMOKE_SEED, seeds)
            for row in rows:
                self.assertEqual(row["physical_Nt"], "2")
                self.assertEqual(row["stored_L2"], "3")
                self.assertEqual(row["Ns"], "160")
                self.assertEqual(row["beta"], "3.6")
                text = (output / row["input"]).read_text(encoding="utf-8")
                self.assertEqual(fv160.step3.sha256_text(text), row["input_sha256"])
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
            output = Path(directory) / "step3_fv160"
            fv160.generate(output)
            original = (output / "manifest.tsv").read_bytes()
            with self.assertRaises(FileExistsError):
                fv160.generate(output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), original)

    def test_qbig_scripts_use_pascal_without_dependencies(self) -> None:
        qbig = ROOT / "data" / "step3_fv160" / "qbig"
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
