#!/usr/bin/env python3
"""Checks for the fixed-mass final-check input generator."""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock


PATH = Path("data/mg_nt_fixed/final_checks/generate_inputs.py")
SPEC = importlib.util.spec_from_file_location("mg_nt_final_generate", PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


class FinalChecksGenerateInputsTest(unittest.TestCase):
    def test_generates_three_four_chain_ensembles_and_reanalysis_manifest(self) -> None:
        summary = {
            "all_nine_tunings_accepted": True,
            "groups": [
                {
                    "target_c": 0.5,
                    "physical_Nt": 3,
                    "local_point_selection": {"selected_beta_values": [4.27, 4.29, 4.4]},
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            trigger = root / "summary.json"
            trigger.write_text(json.dumps(summary))
            output = root / "final_checks"
            with mock.patch.object(MODULE, "_existing_seeds", return_value=set()):
                MODULE.generate(trigger, output)
            with (output / "manifest.tsv").open() as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            with (output / "reanalysis_manifest.tsv").open() as stream:
                reanalysis = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 12)
            self.assertEqual([int(row["sweeps"]) for row in rows[:4]], [80000] * 4)
            self.assertEqual({row["Ns"] for row in rows[4:]}, {"96"})
            self.assertEqual(len(reanalysis), 12)
            self.assertEqual({row["purpose"] for row in reanalysis}, {"final_bootstrap_replacement"})

    def test_qbig_jobs_are_dependency_free_and_use_typed_gpus(self) -> None:
        qbig = PATH.parent / "qbig"
        queue = (qbig / "queue_stage.sh").read_text()
        production = (qbig / "submit_production.slurm").read_text()
        self.assertIn("#SBATCH --array=0-11\n", production)
        self.assertIn("#SBATCH --gres=gpu:pascal:1\n", production)
        self.assertNotIn("--nodelist", production)
        self.assertNotIn("--dependency", queue)
        for name in ("submit_smoke.slurm", "submit_timing.slurm"):
            self.assertIn(
                "#SBATCH --gres=gpu:pascal:1\n", (qbig / name).read_text()
            )
        reanalysis = (qbig / "reanalyze_borderline.sh").read_text()
        self.assertIn('accepted="${root}/analysis/mg_nt_fixed/pilot/', reanalysis)
        self.assertNotIn('run_complete.txt', reanalysis)


if __name__ == "__main__":
    unittest.main()
