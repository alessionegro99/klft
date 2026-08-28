#!/usr/bin/env python3
"""Tests for the first fixed-mass bracket-extension generator."""

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
MODULE_PATH = (
    ROOT / "data" / "mg_nt_fixed" / "extensions" / "round1" / "generate_inputs.py"
)
SPEC = importlib.util.spec_from_file_location("mg_nt_fixed_extension", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
extension = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extension
SPEC.loader.exec_module(extension)


class MgNtFixedExtensionGenerateInputsTest(unittest.TestCase):
    def _summary(self, path: Path) -> None:
        groups = []
        for target in (0.5, 1.0, 2.0):
            for nt in (2, 3, 4):
                accepted = (target, nt) not in {(0.5, 3), (0.5, 4)}
                reasons: list[str] = []
                if (target, nt) == (0.5, 3):
                    reasons = ["log-linear interpolation has p < 0.05"]
                if (target, nt) == (0.5, 4):
                    reasons = ["target is not bracketed"]
                groups.append(
                    {
                        "target_c": target,
                        "physical_Nt": nt,
                        "accepted": accepted,
                        "failure_reasons": reasons,
                    }
                )
        path.write_text(json.dumps({"groups": groups}), encoding="utf-8")

    def test_grid_hashes_seed_provenance_and_overwrite_refusal(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            summary = temporary / "tuning.json"
            output = temporary / "round1"
            self._summary(summary)
            with mock.patch.object(extension, "_existing_seeds", return_value=set()):
                extension.generate(summary, output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 8)
            self.assertEqual(
                {(row["physical_Nt"], row["Ns"], row["beta"]) for row in rows},
                {("3", "80", "4.27"), ("4", "96", "5.50")},
            )
            self.assertEqual(
                {int(row["seed"]) for row in rows}, set(extension.EXTENSION_SEEDS)
            )
            for row in rows:
                text = (output / row["input"]).read_text(encoding="utf-8")
                self.assertEqual(extension.pilot.sha256_text(text), row["input_sha256"])
                self.assertEqual(row["campaign_type"], "nonperturbative_mg_nt")
                self.assertEqual(row["measurement_interval"], "1")
                self.assertEqual(row["dirichlet_suite_measurement_interval"], "5")
            provenance = (output / "seed_provenance.tsv").read_text(encoding="utf-8")
            self.assertIn("\t105\n", provenance)
            self.assertIn("\t112\n", provenance)
            manifest = (output / "manifest.tsv").read_bytes()
            with mock.patch.object(extension, "_existing_seeds", return_value=set()):
                with self.assertRaises(FileExistsError):
                    extension.generate(summary, output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), manifest)


if __name__ == "__main__":
    unittest.main()
