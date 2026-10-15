"""Synthetic checks for the matched-y inputs and arbitrary-thickness slab I/O."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import mg_nt_fixed_scaling_inputs as inputs
import mg_nt_fixed_scaling_run as runner
from mg_nt_fixed_nt1_run import checkpoint_check as nt1_checkpoint_check


class ScalingTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.build = self.root / "build.txt"
        self.validation = self.root / "validation.txt"
        self.reserved = self.root / "seeds.json"
        self.build.write_text(
            f"klft_commit {inputs.COMMIT}\nheatbath_sha256 {'b'*64}\n"
            "slab_boundary_flag 3\nbuild_path /qbigwork2/negro/dirichlet/build/klft-a100\n"
            "cmake_options KLFT_NDIM=3;KLFT_NC=2;CUDA=ON;AMPERE80=ON\n")
        self.validation.write_text(
            f"klft_commit {inputs.COMMIT}\nheatbath_sha256 {'b'*64}\nctest passed\ncuda_memcheck passed\n")
        self.reserved.write_text(json.dumps(dict(source_root=str(runner.ROOT / "data"), seeds=[12, 34])))

    def prepare(self) -> dict:
        return inputs.generate(self.root / "data/mg_nt_fixed", self.build, self.validation, self.reserved)

    def synthetic(self, nt: int) -> Path:
        directory = self.root / f"nt_{nt}"
        directory.mkdir()
        ns, sweeps = 4, 10
        np.savetxt(directory / "holonomy_summary.out",
                   np.column_stack((np.arange(1, 11), np.zeros((10, 2)),
                                    np.ones((10, 2)), np.full(10, np.nan))))
        rows = [[step, r, 1. if r == 0 else .25, .5, np.nan]
                for step in range(1, 11) for r in range(3)]
        np.savetxt(directory / "holonomy_correlator.out", rows,
                   fmt=["%d", "%d", "%.12g", "%.12g", "%.12g"])
        rows = [[step, kind, t, 1. if kind == 0 and t in (0, nt) else .7]
                for step in (5, 10) for kind, t in
                ([(0, t) for t in range(nt + 1)] + [(1, t + .5) for t in range(nt)])]
        np.savetxt(directory / "plaquette_profile.out", rows)
        links = np.zeros((ns * ns, 3 * nt + 2, 4), dtype=np.float64)
        links[:, :, 0] = 1.
        with (directory / "configuration_final.klft").open("wb") as stream:
            stream.write(runner.HEADER.pack(b"KLFTCFG1", 1, 3, 2, 8, 3, ns, ns, nt+1,
                                            ns * ns * (3 * nt + 2)))
            stream.write(links.tobytes())
        return directory

    def test_scope_and_cadences(self) -> None:
        manifest = self.prepare()
        runner.validate_manifest(manifest)
        self.assertEqual(len(manifest["runs"]), 21)
        self.assertEqual(sum(r["sweeps"] for r in manifest["runs"] if r["purpose"] == "production"), 128000)
        for run in manifest["runs"]:
            path = self.root / "data/mg_nt_fixed" / run["input"]
            self.assertEqual(runner.sha256(path), run["input_sha256"])
            self.assertIn(f"  L2: {run['physical_Nt']+1}\n", path.read_text())
            self.assertIn("  measure_boundary_wilson_loops: false\n", path.read_text())
            self.assertIsNone(run["target_c"])
            self.assertFalse(run["fixed_c_continuum_input"])

    def test_overwrite_refusal(self) -> None:
        self.prepare()
        with self.assertRaises(FileExistsError):
            self.prepare()

    def test_local_unsubmitted_seeds_are_reserved(self) -> None:
        reserved = inputs.unique_seeds({12, 34}, 1)[0]
        path = self.root / "data/other/input.yaml"
        path.parent.mkdir(parents=True)
        path.write_text(f"  seed: {reserved}\n")
        self.assertNotIn(reserved, {r["seed"] for r in self.prepare()["runs"]})

    def test_stale_validation(self) -> None:
        self.validation.write_text("ctest failed\n")
        with self.assertRaisesRegex(ValueError, "validation"):
            self.prepare()

    def test_each_approved_thickness_and_nt1_limit(self) -> None:
        for nt in (1, 3, 4, 6):
            with self.subTest(nt=nt):
                directory = self.synthetic(nt)
                result = runner.output_check(directory, 4, nt, 10)
                self.assertEqual(result["profile_rows"], 2 * (2 * nt + 1))
                self.assertEqual(result["checkpoint"]["physical_links"], 16 * (3 * nt + 2))
                if nt == 1:
                    baseline = nt1_checkpoint_check(directory / "configuration_final.klft", 4)
                    self.assertEqual(baseline["physical_links"], result["checkpoint"]["physical_links"])

    def test_missing_row(self) -> None:
        directory = self.synthetic(6)
        path = directory / "holonomy_correlator.out"
        path.write_text("\n".join(path.read_text().splitlines()[:-1]) + "\n")
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            runner.output_check(directory, 4, 6, 10)

    def test_profile_boundary_and_internal_slice(self) -> None:
        directory = self.synthetic(3)
        path = directory / "plaquette_profile.out"
        values = np.loadtxt(path)
        values[3, 3] = .9
        np.savetxt(path, values)
        with self.assertRaisesRegex(ValueError, "fixed-surface"):
            runner.output_check(directory, 4, 3, 10)

    def test_upper_boundary_checkpoint(self) -> None:
        directory = self.synthetic(4)
        path = directory / "configuration_final.klft"
        with path.open("r+b") as stream:
            stream.seek(runner.HEADER.size + 12 * 32)
            stream.write(np.array([0., 1., 0., 0.], dtype=np.float64).tobytes())
        with self.assertRaisesRegex(ValueError, "boundary link"):
            runner.checkpoint_check(path, 4, 4)

    def test_physical_link_norm(self) -> None:
        directory = self.synthetic(6)
        path = directory / "configuration_final.klft"
        with path.open("r+b") as stream:
            stream.seek(runner.HEADER.size + 3 * 32)
            stream.write(np.array([1.01, 0., 0., 0.], dtype=np.float64).tobytes())
        with self.assertRaisesRegex(ValueError, "norm error"):
            runner.checkpoint_check(path, 4, 6)

    def test_timing_budget(self) -> None:
        manifest = self.prepare()
        with patch.object(runner, "checked_result", return_value={"elapsed_seconds": 2.}):
            result = runner.timing_projection(manifest, "hash")
        self.assertTrue(result["passed"])
        self.assertAlmostEqual(result["projected_total_gpu_hours"], 16 * 160 / 3600)
        with patch.object(runner, "checked_result", return_value={"elapsed_seconds": 100.}):
            self.assertFalse(runner.timing_projection(manifest, "hash")["passed"])

    def test_changed_duration_or_target(self) -> None:
        manifest = self.prepare()
        manifest["runs"][-1]["target_c"] = .5
        with self.assertRaisesRegex(ValueError, "policy"):
            runner.validate_manifest(manifest)

    def test_launcher_is_bounded_and_manually_staged(self) -> None:
        path = Path(__file__).resolve().parents[1] / "data/mg_nt_fixed/run_scaling_a100.sh"
        text = path.read_text()
        for required in ("flock -n", "timeout --signal=TERM --kill-after=30s 2h", "taskset -c",
                         "preflight|production", "CUDA_VISIBLE_DEVICES", "lnode19"):
            self.assertIn(required, text)
        self.assertNotIn("sbatch", text)
        self.assertNotIn("GPU-c0d18349-2a36-8733-e53b-ae2ab1e14209", text)
        self.assertEqual(text.count("gpu=GPU-"), 3)


if __name__ == "__main__":
    unittest.main()
