#!/usr/bin/env python3
"""Checks for fixed-beta/Nt continuum-production inputs."""

from __future__ import annotations

import csv
from decimal import Decimal
from pathlib import Path
import tempfile
import unittest

from data.beta_nt_fixed.comparison import generate_inputs as production


class ProductionGenerateInputsTest(unittest.TestCase):
    def test_grid_has_fixed_coupling_lines_and_safe_expected_volumes(self) -> None:
        self.assertEqual(
            sorted({ensemble.beta_over_nt for ensemble in production.GRID}),
            ["1.8", "2.0", "2.1"],
        )
        for ensemble in production.GRID:
            self.assertEqual(
                Decimal(ensemble.beta) / ensemble.nt,
                Decimal(ensemble.beta_over_nt),
            )
            self.assertEqual(
                ensemble.g3_squared_l,
                Decimal(4) / Decimal(ensemble.beta_over_nt),
            )
            expected_xi = Decimal(ensemble.expected_xi)
            self.assertGreaterEqual(Decimal(ensemble.ns) / expected_xi, 10)
            self.assertGreaterEqual(
                Decimal(ensemble.boundary_max_r) / expected_xi, 3
            )

    def test_input_enables_all_requested_observables_and_cadences(self) -> None:
        ensemble = production.GRID[-1]
        text = production.input_text(ensemble, production.SEEDS[0], 20)
        self.assertIn("  measurement_interval: 1\n", text)
        self.assertIn("  dirichlet_suite_measurement_interval: 5\n", text)
        self.assertIn("  boundary_wilson_measurement_interval: 20\n", text)
        self.assertIn("  measure_dirichlet_holonomy: true\n", text)
        self.assertIn("  measure_dirichlet_plaquette_profiles: true\n", text)
        self.assertIn("  measure_boundary_wilson_loops: true\n", text)
        self.assertIn("  measure_boundary_wilson_all_heights: true\n", text)
        self.assertIn(
            f"  boundary_wilson_max_r: {ensemble.boundary_max_r}\n", text
        )

    def test_generation_is_complete_unique_and_immutable(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            paths = production.generate(output)
            self.assertEqual(len(paths), 46)
            with (output / "manifest.tsv").open(
                encoding="utf-8", newline=""
            ) as stream:
                rows = list(csv.DictReader(stream, delimiter="\t"))
            self.assertEqual(len(rows), 40)
            self.assertEqual(
                [int(row["array_index"]) for row in rows], list(range(40))
            )
            self.assertEqual(len({int(row["seed"]) for row in rows}), 40)
            with self.assertRaises(FileExistsError):
                production.generate(output)

    def test_cadence25_revision_is_distinct_and_reproducible(self) -> None:
        self.assertFalse(set(production.SEEDS) & set(production.CADENCE25_SEEDS))
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            production.generate(
                output,
                seeds=production.CADENCE25_SEEDS,
                boundary_interval=production.CADENCE25_BOUNDARY_INTERVAL,
                smoke_sweeps=production.CADENCE25_SMOKE_SWEEPS,
                timing_sweeps=production.CADENCE25_TIMING_SWEEPS,
                excluded_manifest=(
                    Path(production.__file__).parent
                    / "provenance"
                    / "cadence25"
                    / "manifest.tsv"
                ),
            )
            smoke = (output / "smoke" / "input.yaml").read_text()
            timing = (output / "timing" / "input.yaml").read_text()
            self.assertIn("  nSweep: 25\n", smoke)
            self.assertIn("  nSweep: 25\n", timing)
            self.assertIn(
                "  boundary_wilson_measurement_interval: 25\n", timing
            )


if __name__ == "__main__":
    unittest.main()
