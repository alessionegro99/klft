#!/usr/bin/env python3
"""Tests for the corrected Step-3 constant-coupling volumes."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = (
    ROOT / "data" / "step3_corrected_volumes" / "generate_inputs.py"
)
SPEC = importlib.util.spec_from_file_location(
    "step3_corrected_volume_inputs", MODULE_PATH
)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
corrected = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = corrected
SPEC.loader.exec_module(corrected)


class Step3CorrectedVolumesGenerateInputsTest(unittest.TestCase):
    def test_grid_cadence_hashes_and_unique_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "step3_corrected_volumes"
            corrected.generate(output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 8)
            seeds = {int(row["seed"]) for row in rows}
            self.assertEqual(len(seeds), 8)
            self.assertFalse(seeds & corrected.previous_seeds())
            self.assertNotIn(corrected.TIMING_SEED, seeds)
            expected = [("3", "4", "176", "5.4")] * 4 + [
                ("4", "5", "256", "7.2")
            ] * 4
            self.assertEqual(
                [
                    (
                        row["physical_Nt"],
                        row["stored_L2"],
                        row["Ns"],
                        row["beta"],
                    )
                    for row in rows
                ],
                expected,
            )
            for row in rows:
                self.assertEqual(row["sweeps"], "20000")
                self.assertEqual(row["measurement_interval"], "1")
                self.assertEqual(
                    row["dirichlet_suite_measurement_interval"], "5"
                )
                text = (output / row["input"]).read_text(encoding="utf-8")
                self.assertEqual(
                    corrected.step3.sha256_text(text), row["input_sha256"]
                )
                self.assertIn("  nOverrelax: 5\n", text)
                self.assertIn("  measure_boundary_wilson_loops: false\n", text)

            timing = (output / "timing" / "input.yaml").read_text(
                encoding="utf-8"
            )
            self.assertIn("  L0: 256\n", timing)
            self.assertIn("  L2: 5\n", timing)
            self.assertIn("  nSweep: 100\n", timing)
            self.assertEqual(
                (output / "klft_commit.txt").read_text(encoding="utf-8"),
                f"{corrected.KLFT_COMMIT}\n",
            )

    def test_refuses_overwrite_before_partial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "step3_corrected_volumes"
            corrected.generate(output)
            original = (output / "manifest.tsv").read_bytes()
            with self.assertRaises(FileExistsError):
                corrected.generate(output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), original)

    def test_qbig_scripts_are_uncapped_and_dependency_free(self) -> None:
        qbig = ROOT / "data" / "step3_corrected_volumes" / "qbig"
        queue = (qbig / "queue_stage.sh").read_text(encoding="utf-8")
        timing = (qbig / "submit_timing.slurm").read_text(encoding="utf-8")
        production = (qbig / "submit_production.slurm").read_text(
            encoding="utf-8"
        )
        diagnostics = (qbig / "submit_diagnostics.slurm").read_text(
            encoding="utf-8"
        )
        analysis = (qbig / "submit_analysis.slurm").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("--dependency", queue)
        self.assertNotIn("afterok", queue)
        self.assertIn("#SBATCH --array=0-7\n", production)
        self.assertNotIn("#SBATCH --array=0-7%", production)
        self.assertNotIn("--nodelist", production)
        for script in (timing, production):
            self.assertIn("#SBATCH --gres=gpu:pascal:1\n", script)
            self.assertIn('build="${remote_root}/build/klft"', script)
        self.assertIn("maximum <= 2.0 && total <= 12.0", timing)
        self.assertIn("timing gate did not pass", production)
        for script in (diagnostics, analysis):
            self.assertNotIn("--dependency", script)
            self.assertIn("#SBATCH --array=0-1\n", script)
        self.assertIn("blocking_error_plateau", (
            ROOT / "scripts" / "step3_analysis.py"
        ).read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
