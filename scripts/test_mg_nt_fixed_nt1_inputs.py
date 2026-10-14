"""Checks for Nt=1 matched-beta inputs; no Monte Carlo or bootstrap is run."""

import csv
import hashlib
import json
from pathlib import Path
import re
import tempfile
import unittest

from mg_nt_fixed_nt1_inputs import generate, selected_tunings, unique_seeds


class MatchedBetaInputsTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.output = self.root / "output"
        self.tuning = self.root / "tuning.csv"
        self.build = self.root / "build.txt"
        self.reserved = self.root / "seeds.json"
        self.betas = ("2.989543168360997", "2.5044558261064154", "1.962386866460073")
        fields = ["target_c", "physical_Nt", "available", "selected", "include",
                  "beta_star", "bootstrap_samples", "flags"]
        with self.tuning.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            for c, beta in zip((0.5, 1, 2), self.betas):
                writer.writerow(dict(target_c=c, physical_Nt=2, available=True,
                                     selected=True, include=True, beta_star=beta,
                                     bootstrap_samples=500, flags="preliminary"))
        self.build.write_text(
            f"klft_commit {'a' * 40}\nheatbath_sha256 {'b' * 64}\n"
            "slab_boundary_flag 3\n"
            "build_path /qbigwork2/negro/dirichlet/build/klft\n"
            "cmake_options Release;KLFT_NDIM=3;KLFT_NC=2;CUDA=ON;PASCAL60=ON\n")
        self.reserved.write_text(json.dumps({
            "source_root": "/qbigwork2/negro/dirichlet/data", "seeds": [11, 22, 33]}))

    def prepare(self) -> dict:
        return generate(self.output, self.tuning, self.build, self.reserved, (64, 32, 16))

    def test_physics_cadences_and_full_precision(self) -> None:
        manifest = self.prepare()
        runs = manifest["runs"]
        self.assertEqual(len(runs), 16)
        self.assertEqual(sum(r["purpose"] == "production" for r in runs), 12)
        self.assertEqual(sum(r["purpose"] == "timing" for r in runs), 3)
        self.assertEqual(sum(r["purpose"] == "smoke" for r in runs), 1)
        self.assertEqual(len({r["seed"] for r in runs}), 16)
        self.assertFalse({r["seed"] for r in runs} & {11, 22, 33})
        for run in runs:
            raw = (self.output / run["input"]).read_bytes()
            text = raw.decode()
            self.assertEqual(hashlib.sha256(raw).hexdigest(), run["input_sha256"])
            values = dict(re.findall(r"^  (\w+): (.+)$", text, re.MULTILINE))
            self.assertEqual(values["L2"], "2")
            self.assertEqual(values["L0"], str(run["Ns"]))
            self.assertEqual(values["L1"], str(run["Ns"]))
            self.assertEqual(values["beta"], self.betas[(0.5, 1, 2).index(run["reference_c"])])
            self.assertEqual(values["nOverrelax"], "5")
            self.assertEqual(values["measurement_interval"], "1")
            self.assertEqual(values["dirichlet_suite_measurement_interval"], "5")
            self.assertEqual(values["measure_boundary_wilson_loops"], "false")
            self.assertEqual(values["temporal_dirichlet"], "true")
            self.assertEqual(values["nSweep"], str({"production": 8000, "timing": 100, "smoke": 8}[run["purpose"]]))
            self.assertIsNone(run["target_c"])
            self.assertFalse(run["fixed_c_continuum_input"])
            self.assertEqual(run["source_tuning_Nt"], 2)
            self.assertEqual(run["reference_tuning_flags"], ["preliminary"])
            self.assertRegex(run["input"], r"^c_[^/]+/Nt_1/Ns_\d+/beta_[^/]+/seed_\d+/input.yaml$")
        self.assertEqual(manifest["analysis_bootstrap_samples"], 500)

    def test_overwrite_refusal_preserves_every_input(self) -> None:
        self.prepare()
        before = {p: p.read_bytes() for p in self.output.rglob("*") if p.is_file()}
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(before, {p: p.read_bytes() for p in before})

    def test_existing_seed_directory_refuses_before_writes(self) -> None:
        first_seed = unique_seeds({11, 22, 33}, 16)[0]
        path = self.output / f"c_0p5/Nt_1/Ns_4/beta_2p989543168360997/seed_{first_seed}"
        path.mkdir(parents=True)
        sentinel = path / "existing.txt"
        sentinel.write_text("preserve")
        with self.assertRaises(FileExistsError):
            self.prepare()
        self.assertEqual(list(self.output.rglob("*.yaml")), [])
        self.assertEqual(sentinel.read_text(), "preserve")

    def test_seed_policy_is_deterministic_and_collision_free(self) -> None:
        first = unique_seeds(set(), 16)
        second = unique_seeds(set(first), 16)
        self.assertEqual(first, unique_seeds(set(), 16))
        self.assertTrue(all(0 < seed < 2**31 - 1 for seed in first + second))
        self.assertFalse(set(first) & set(second))

    def test_invalid_build_is_rejected_without_outputs(self) -> None:
        self.build.write_text("klft_commit invalid\n")
        with self.assertRaises(ValueError):
            self.prepare()
        self.assertFalse(self.output.exists())

    def test_undefined_or_duplicate_tuning_is_rejected(self) -> None:
        text = self.tuning.read_text()
        self.tuning.write_text(text + text.splitlines()[1] + "\n")
        with self.assertRaises(ValueError):
            selected_tunings(self.tuning)
        self.tuning.write_text(text.replace(self.betas[0], "nan"))
        with self.assertRaises(ValueError):
            selected_tunings(self.tuning)

    def test_unselected_and_historical_tunings_are_rejected(self) -> None:
        text = self.tuning.read_text()
        self.tuning.write_text(text.replace("500", "1000"))
        with self.assertRaises(ValueError):
            selected_tunings(self.tuning)
        self.tuning.write_text(text.replace("True,True,True", "True,False,True"))
        with self.assertRaises(ValueError):
            selected_tunings(self.tuning)


if __name__ == "__main__":
    unittest.main()
