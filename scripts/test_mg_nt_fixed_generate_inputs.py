#!/usr/bin/env python3
"""Tests for the nonperturbative fixed-mass pilot generator."""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "data" / "mg_nt_fixed" / "pilot" / "generate_inputs.py"
SPEC = importlib.util.spec_from_file_location("mg_nt_fixed_inputs", MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(MODULE_PATH)
inputs = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = inputs
SPEC.loader.exec_module(inputs)


class MgNtFixedGenerateInputsTest(unittest.TestCase):
    def test_grid_provenance_hashes_and_unique_seeds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            inputs.generate(output)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 100)
            self.assertEqual({row["campaign_type"] for row in rows}, {
                "nonperturbative_mg_nt"
            })
            self.assertEqual({row["target_c"] for row in rows}, {"0.5", "1.0", "2.0"})
            seeds = {int(row["seed"]) for row in rows}
            self.assertEqual(len(seeds), 100)
            self.assertFalse(seeds & inputs.previous_seeds())
            self.assertFalse(seeds & set(inputs.CONFIRMATION_SEEDS))
            self.assertFalse(seeds & set(inputs.SPARE_BRACKET_SEEDS))
            self.assertNotIn(inputs.TIMING_SEED, seeds)
            self.assertEqual(sorted({int(row["physical_Nt"]) for row in rows}), [2, 3, 4])
            self.assertEqual(
                [(row["target_c"], row["physical_Nt"], row["beta"]) for row in rows[:8]],
                [("0.5", "2", "2.90")] * 4 + [("0.5", "2", "3.10")] * 4,
            )
            for row in rows:
                self.assertEqual(int(row["stored_L2"]), int(row["physical_Nt"]) + 1)
                self.assertEqual(row["sweeps"], "8000")
                self.assertEqual(row["measurement_interval"], "1")
                self.assertEqual(row["dirichlet_suite_measurement_interval"], "5")
                self.assertEqual(row["klft_commit"], inputs.KLFT_COMMIT)
                self.assertEqual(row["heatbath_sha256"], inputs.HEATBATH_SHA256)
                text = (output / row["input"]).read_text(encoding="utf-8")
                self.assertEqual(inputs.sha256_text(text), row["input_sha256"])
                self.assertIn(f"target m_G Nt={row['target_c']}", text)
                self.assertIn("  measure_boundary_wilson_loops: false\n", text)

            with (output / "prior_points.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                prior = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(prior), 2)
            self.assertEqual({row["source_campaign"] for row in prior}, {"beta_nt_fixed"})
            self.assertEqual({row["beta"] for row in prior}, {"3.0", "2.1"})
            reserved = (output / "reserved_seeds.tsv").read_text(encoding="utf-8")
            self.assertEqual(reserved.count("spare_bracket_chain_"), 4)

    def test_refuses_overwrite_before_partial_changes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pilot"
            inputs.generate(output)
            original = (output / "manifest.tsv").read_bytes()
            with self.assertRaises(FileExistsError):
                inputs.generate(output)
            self.assertEqual((output / "manifest.tsv").read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
