"""Execution checks using synthetic Nt=1 files; no simulation or bootstrap."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mg_nt_fixed_nt1_run as runner


class Nt1RunChecksTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.ns = 4
        self.sweeps = 10
        summary = np.column_stack((np.arange(1, 11), np.zeros((10, 2)),
                                   np.ones((10, 2)), np.full(10, np.nan)))
        np.savetxt(self.root / "holonomy_summary.out", summary)
        rows = [[step, r, 1. if r == 0 else .25, .5, np.nan]
                for step in range(1, 11) for r in range(3)]
        np.savetxt(self.root / "holonomy_correlator.out", rows,
                   fmt=["%d", "%d", "%.12g", "%.12g", "%.12g"])
        rows = [[step, kind, t, value] for step in (5, 10)
                for kind, t, value in ((0, 0, 1), (0, 1, 1), (1, .5, .6))]
        np.savetxt(self.root / "plaquette_profile.out", rows)
        links = np.zeros((16, 5, 4), dtype=np.float64)
        links[:, :, 0] = 1.
        self.checkpoint = self.root / "configuration_final.klft"
        with self.checkpoint.open("wb") as stream:
            stream.write(runner.HEADER.pack(b"KLFTCFG1", 1, 3, 2, 8, 3, 4, 4, 2, 80))
            stream.write(links.tobytes())

    def test_complete_native_output(self) -> None:
        result = runner.output_check(self.root, self.ns, self.sweeps)
        self.assertEqual(result["summary_measurements"], 10)
        self.assertEqual(result["correlator_rows"], 30)
        self.assertEqual(result["profile_measurements"], 2)
        self.assertEqual(result["checkpoint"]["physical_links"], 80)

    def test_missing_correlator_row(self) -> None:
        path = self.root / "holonomy_correlator.out"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runner.output_check(self.root, 4, 10)

    def test_zero_separation_normalization(self) -> None:
        path = self.root / "holonomy_correlator.out"
        path.write_text(path.read_text().replace("1 0 1 ", "1 0 0.99 ", 1))
        with self.assertRaisesRegex(ValueError, "correlator row"):
            runner.output_check(self.root, 4, 10)

    def test_wrong_profile_cadence(self) -> None:
        path = self.root / "plaquette_profile.out"
        values = np.loadtxt(path)
        values[0, 0] = 4.
        np.savetxt(path, values)
        with self.assertRaisesRegex(ValueError, "profile"):
            runner.output_check(self.root, 4, 10)

    def test_nonfinite_fundamental_observable(self) -> None:
        path = self.root / "holonomy_summary.out"
        values = np.loadtxt(path)
        values[0, 3] = np.nan
        np.savetxt(path, values)
        with self.assertRaisesRegex(ValueError, "summary"):
            runner.output_check(self.root, 4, 10)

    def test_checkpoint_has_no_closing_links(self) -> None:
        with self.checkpoint.open("ab") as stream:
            stream.write(bytes(32))
        with self.assertRaisesRegex(ValueError, "size mismatch"):
            runner.checkpoint_check(self.checkpoint, 4)

    def test_checkpoint_fixed_surface(self) -> None:
        with self.checkpoint.open("r+b") as stream:
            stream.seek(runner.HEADER.size)
            stream.write(np.array([0., 1., 0., 0.], dtype=np.float64).tobytes())
        with self.assertRaisesRegex(ValueError, "boundary link"):
            runner.checkpoint_check(self.checkpoint, 4)

    def test_checkpoint_temporal_norm(self) -> None:
        with self.checkpoint.open("r+b") as stream:
            stream.seek(runner.HEADER.size + 2 * 32)
            stream.write(np.array([1.01, 0., 0., 0.], dtype=np.float64).tobytes())
        with self.assertRaisesRegex(ValueError, "norm error"):
            runner.checkpoint_check(self.checkpoint, 4)

    def test_immutable_records(self) -> None:
        path = self.root / "execution_start.json"
        runner.write_json(path, {"preserve": True})
        with self.assertRaises(FileExistsError):
            runner.write_json(path, {})
        self.assertEqual(json.loads(path.read_text()), {"preserve": True})

    def test_timing_gates(self) -> None:
        runs = []
        for i, ns in enumerate((128, 512, 2048)):
            common = dict(Ns=ns, beta_text=str(i + 1), n_overrelax=5)
            runs.append(dict(common, purpose="timing", sweeps=100))
            runs.extend(dict(common, purpose="production", sweeps=8000, seed=10 * i + j)
                        for j in range(4))
        with patch.object(runner, "checked_result", return_value={"elapsed_seconds": 10.}):
            result = runner.timing_projection({"runs": runs})
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["projected_total_gpu_hours"], 12 * 800. / 3600.)
        with patch.object(runner, "checked_result", return_value={"elapsed_seconds": 60.}):
            self.assertFalse(runner.timing_projection({"runs": runs})["passed"])
        with patch.object(runner, "checked_result", return_value={"elapsed_seconds": 100.}):
            result = runner.timing_projection({"runs": runs})
        self.assertFalse(result["passed"])
        self.assertGreater(result["largest_chain_gpu_hours"], 2.)

    def test_no_login_or_non_p100_execution(self) -> None:
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaises(RuntimeError):
                runner.run_one("smoke", 0)
        with patch.dict(os.environ, {"SLURM_JOB_ID": "1", "SLURMD_NODENAME": "lnode03"}):
            with self.assertRaises(RuntimeError):
                runner.run_one("smoke", 0)

    def test_submission_is_manually_staged(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/mg_nt_fixed/nt1_run.slurm"
        subprocess.run(["bash", "-n", str(path)], check=True)
        text = path.read_text()
        self.assertIn("#SBATCH --gres=gpu:pascal:1", text)
        self.assertIn("#SBATCH --time=02:00:00", text)
        self.assertNotIn("--dependency", text)
        self.assertNotIn("--array=", text)
        self.assertNotIn("--nodelist", text)


if __name__ == "__main__":
    unittest.main()
