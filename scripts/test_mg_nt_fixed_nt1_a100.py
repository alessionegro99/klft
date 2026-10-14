"""Test A100 production scope, immutable input reuse and execution safeguards."""

import copy
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import mg_nt_fixed_nt1_a100 as runner
from mg_nt_fixed_nt1_run import sha256


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mg_nt_fixed"


class A100ProductionTest(unittest.TestCase):
    def setUp(self) -> None:
        self.plan = json.loads((DATA / "nt1_a100_production.json").read_text())
        self.source = json.loads((DATA / "nt1_manifest.json").read_text())
        self.timings = json.loads((DATA / "open_slab_a100_runs.json").read_text())

    def test_all_production_inputs_remain_unchanged(self) -> None:
        runs, gate = runner.production_runs(self.plan, self.source, self.timings)
        self.assertEqual(len(runs), 12)
        self.assertAlmostEqual(gate["projected_total_gpu_hours"], 3.859448832770189)
        self.assertLess(gate["largest_chain_gpu_hours"], 1.)
        self.assertEqual(sha256(DATA / "nt1_manifest.json"), self.plan["source_manifest_sha256"])
        for name, digest in self.plan["evidence_sha256"].items():
            self.assertEqual(sha256(DATA / name), digest)
        for run in runs:
            path = DATA / run["input"]
            self.assertEqual(sha256(path), run["input_sha256"])
            original = {k: v for k, v in run.items() if k not in {"worker", "projected_gpu_hours"}}
            self.assertEqual(json.loads(path.with_name("run_metadata.json").read_text()), original)
            self.assertNotEqual(original["heatbath_sha256"], self.plan["heatbath_sha256"])
            self.assertIsNone(run["target_c"])
            self.assertFalse(run["fixed_c_continuum_input"])

    def test_authorization_geometry_and_bootstrap_policy_are_required(self) -> None:
        for key, value in (("production_authorized", False), ("node", "qbig"),
                           ("analysis_bootstrap_samples", 1000), ("target_c", 1.)):
            plan = copy.deepcopy(self.plan)
            plan[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.production_runs(plan, self.source, self.timings)
        for key, value in (("physical_Nt", 2), ("sweeps", 80000), ("measurement_interval", 5)):
            source = copy.deepcopy(self.source)
            next(r for r in source["runs"] if r["purpose"] == "production")[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.production_runs(self.plan, source, self.timings)

    def test_changed_assignment_or_duplicate_device_is_refused(self) -> None:
        plan = copy.deepcopy(self.plan)
        plan["runs"][0]["beta_text"] = "3.0"
        with self.assertRaisesRegex(ValueError, "Original input changed"):
            runner.production_runs(plan, self.source, self.timings)
        plan = copy.deepcopy(self.plan)
        plan["workers"][1]["gpu_uuid"] = plan["workers"][0]["gpu_uuid"]
        with self.assertRaisesRegex(ValueError, "distinct"):
            runner.production_runs(plan, self.source, self.timings)

    def test_stale_timing_and_over_budget_runs_are_refused(self) -> None:
        for key, value in (("heatbath_sha256", "stale"), ("elapsed_seconds", 100000.),
                           ("elapsed_seconds", float("nan")), ("n_overrelax", 2)):
            timings = copy.deepcopy(self.timings)
            next(r for r in timings["runs"] if r["purpose"] == "timing")[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                runner.production_runs(self.plan, self.source, timings)

    def test_gpu_occupancy_and_architecture_checks(self) -> None:
        with patch.object(subprocess, "check_output", side_effect=["NVIDIA A100", "1234"]):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                runner.check_gpu_idle("GPU-test")
        with patch.object(subprocess, "check_output", return_value="Tesla P100"):
            with self.assertRaisesRegex(RuntimeError, "not an A100"):
                runner.check_gpu_idle("GPU-test")
        with patch.object(subprocess, "check_output", side_effect=["NVIDIA A100", ""]):
            runner.check_gpu_idle("GPU-test")

    def test_existing_output_and_modified_metadata_are_refused(self) -> None:
        run = next(r.copy() for r in self.source["runs"] if r["purpose"] == "production")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / run["input"]
            path.parent.mkdir(parents=True)
            path.write_bytes((DATA / run["input"]).read_bytes())
            run["raw_path"] = str(path.parent)
            metadata = path.with_name("run_metadata.json")
            metadata.write_text(json.dumps(run))
            with patch.object(runner, "DATA", root):
                self.assertEqual(runner.check_unused_input(run), path.parent)
                marker = path.with_name("execution_start.json")
                marker.touch()
                with self.assertRaises(FileExistsError):
                    runner.check_unused_input(run)
                marker.unlink()
                metadata.write_text("{}")
                with self.assertRaisesRegex(ValueError, "metadata changed"):
                    runner.check_unused_input(run)

    def test_launcher_is_bounded_and_preserves_native_environment(self) -> None:
        path = DATA / "run_nt1_a100.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        self.assertIn("timeout --signal=TERM --kill-after=30s 2h", text)
        self.assertIn("flock -n 9", text)
        self.assertIn('taskset -c "${cpu}"', text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("scancel", text)
        self.assertLess(text.index("source /hadron/negro/setup_qbig3.sh"),
                        text.index("source /hadron/negro/load_modules_qbig3.sh"))
        for worker in self.plan["workers"]:
            self.assertIn(worker["gpu_uuid"], text)
            self.assertIn(f"cpu={worker['cpu']}", text)


if __name__ == "__main__":
    unittest.main()
