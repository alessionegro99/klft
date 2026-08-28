#!/usr/bin/env python3
"""Checks for the second fixed-mass bracket-extension generator."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parents[1]
MODULE = ROOT / "data/mg_nt_fixed/extensions/round2/generate_inputs.py"
SPEC = importlib.util.spec_from_file_location("mg_nt_fixed_round2", MODULE)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE)
round2 = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = round2
SPEC.loader.exec_module(round2)


class MgNtFixedRound2GenerateInputsTest(unittest.TestCase):
    def test_gate_grid_hashes_and_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            summary = root / "tuning.json"
            output = root / "round2"
            summary.write_text(
                json.dumps(
                    {
                        "groups": [
                            {
                                "target_c": 0.5,
                                "physical_Nt": 3,
                                "accepted": False,
                                "failure_reasons": [
                                    "log-linear interpolation has p < 0.05"
                                ],
                            },
                            {
                                "target_c": 1.0,
                                "physical_Nt": 4,
                                "accepted": True,
                                "failure_reasons": [],
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            with mock.patch.object(round2, "_existing_seeds", return_value=set()):
                round2.generate(summary, output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 4)
            self.assertEqual({row["beta"] for row in rows}, {"4.290"})
            self.assertEqual({row["Ns"] for row in rows}, {"80"})
            self.assertEqual(
                {int(row["seed"]) for row in rows}, set(round2.PRODUCTION_SEEDS)
            )
            with (output / "validation_manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                validation = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual([row["sweeps"] for row in validation], ["25", "100"])
            before = (output / "manifest.tsv").read_bytes()
            with mock.patch.object(round2, "_existing_seeds", return_value=set()):
                with self.assertRaises(FileExistsError):
                    round2.generate(summary, output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), before)

    def test_qbig_jobs_are_dependency_free_and_use_typed_gpus(self) -> None:
        qbig = MODULE.parent / "qbig"
        queue = (qbig / "queue_stage.sh").read_text(encoding="utf-8")
        production = (qbig / "submit_production.slurm").read_text(
            encoding="utf-8"
        )
        for name in ("submit_smoke.slurm", "submit_timing.slurm"):
            validation = (qbig / name).read_text(encoding="utf-8")
            self.assertIn("#SBATCH --gres=gpu:pascal:1\n", validation)
        self.assertIn("#SBATCH --array=0-3\n", production)
        self.assertNotIn("#SBATCH --array=0-3%", production)
        self.assertIn("#SBATCH --gres=gpu:pascal:1\n", production)
        self.assertNotIn("--nodelist", production)
        self.assertNotIn("--dependency", queue)
        self.assertNotIn("afterok", queue)


if __name__ == "__main__":
    unittest.main()
