"""Check the bounded direct-A100 test setup without touching a GPU."""

import json
import os
from pathlib import Path
import re
import subprocess
import unittest
from unittest.mock import patch

import mg_nt_fixed_a100_test as runner
from mg_nt_fixed_nt1_inputs import render_input
from mg_nt_fixed_nt1_run import sha256


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/mg_nt_fixed"


class A100TestSetup(unittest.TestCase):
    def test_immutable_inputs_match_their_hashes_and_physics(self) -> None:
        manifest = json.loads((DATA / "a100_test_manifest.json").read_text())
        reference = json.loads((DATA / "nt1_manifest.json").read_text())
        self.assertFalse(manifest["production_authorized"])
        self.assertEqual(manifest["bootstrap_operations"], 0)
        self.assertEqual(len(manifest["runs"]), 4)
        self.assertEqual(sorted(r["sweeps"] for r in manifest["runs"]), [8, 100, 100, 100])
        for run in manifest["runs"]:
            source, = [r for r in reference["runs"] if r["purpose"] == run["purpose"]
                       and r["reference_c"] == run["reference_c"]]
            self.assertEqual(run["beta_text"], source["beta_text"])
            self.assertEqual(run["Ns"], source["Ns"])
            self.assertEqual(run["klft_commit"], runner.EXPECTED_COMMIT)
            self.assertFalse(run["production_authorized"])
            self.assertFalse(run["fixed_c_continuum_input"])
            self.assertIsNone(run["target_c"])
            path = DATA / run["input"]
            self.assertEqual(sha256(path), run["input_sha256"])
            # Preserve the source's numeric formatting in the descriptive comment.
            self.assertEqual(path.read_text(), render_input(
                dict(run, reference_c=source["reference_c"])))
            self.assertEqual(json.loads(path.with_name("run_metadata.json").read_text()), run)
            values = dict(re.findall(r"^  (\w+): (.+)$", path.read_text(), re.MULTILINE))
            self.assertEqual(values["L2"], "2")
            self.assertEqual(values["nOverrelax"], "5")
            self.assertEqual(values["measurement_interval"], "1")
            self.assertEqual(values["dirichlet_suite_measurement_interval"], "5")

    def test_all_new_seeds_are_independent_of_existing_inputs(self) -> None:
        reserved = set(json.loads((DATA / "nt1_seed_inventory.json").read_text())["seeds"])
        reserved.update(r["seed"] for r in json.loads((DATA / "nt1_manifest.json").read_text())["runs"])
        seeds = [r["seed"] for r in json.loads((DATA / "a100_test_manifest.json").read_text())["runs"]]
        self.assertEqual(len(set(seeds)), 4)
        self.assertFalse(set(seeds) & reserved)
        self.assertTrue(all(0 < seed < 2**31 - 1 for seed in seeds))

    def test_occupied_gpu_is_refused(self) -> None:
        with patch.object(subprocess, "check_output", return_value="12345\n"):
            with self.assertRaisesRegex(RuntimeError, "occupied"):
                runner.check_gpu_idle()
        with patch.object(subprocess, "check_output", return_value="\n"):
            runner.check_gpu_idle()

    def test_wrong_host_or_device_is_refused(self) -> None:
        with patch.object(runner.socket, "gethostname", return_value="qbig"):
            with self.assertRaises(RuntimeError):
                runner.main()
        with patch.object(runner.socket, "gethostname", return_value="lnode19"):
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(RuntimeError):
                    runner.main()

    def test_native_build_is_isolated_and_never_submits_production(self) -> None:
        path = DATA / "test_open_slab_a100.sh"
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        self.assertIn('source_dir="${remote_root}/src/klft"\n', text)
        self.assertIn('build_dir="${remote_root}/build/klft-a100"\n', text)
        self.assertIn("-DKokkos_ARCH_AMPERE80=ON", text)
        self.assertIn("compute-sanitizer --tool memcheck --error-exitcode 99", text)
        self.assertIn("flock -n 9", text)
        self.assertIn("set -o noclobber", text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("scancel", text)
        self.assertNotIn("PASCAL60", text)
        self.assertLess(text.index("source /hadron/negro/setup_qbig3.sh"),
                        text.index("source /hadron/negro/load_modules_qbig3.sh"))


if __name__ == "__main__":
    unittest.main()
