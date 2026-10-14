"""Check the local build/validation staging safeguards without remote writes."""

from pathlib import Path
import os
import re
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = [ROOT / "data/mg_nt_fixed" / name for name in (
    "build_open_slab.slurm", "validate_open_slab.slurm",
)]


class OpenSlabSubmissionTest(unittest.TestCase):
    def test_shell_syntax(self) -> None:
        for path in SCRIPTS:
            subprocess.run(["bash", "-n", str(path)], check=True)

    def test_refuse_execution_outside_slurm(self) -> None:
        env = {key: value for key, value in os.environ.items()
               if not key.startswith("SLURM_")}
        for path in SCRIPTS:
            result = subprocess.run(["bash", str(path)], env=env,
                                    capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("SLURM_JOB_ID", result.stderr)

    def test_validation_resources_and_manual_staging(self) -> None:
        text = SCRIPTS[1].read_text()
        self.assertIn("#SBATCH --gres=gpu:pascal:1", text)
        self.assertIn("#SBATCH --ntasks=1", text)
        self.assertNotRegex(text, r"(?m)^#SBATCH.*--nodelist=.*,")
        for path in SCRIPTS:
            active = "\n".join(line for line in path.read_text().splitlines()
                               if not line.startswith("#"))
            self.assertNotRegex(active, r"\bsbatch\b|--dependency")
            self.assertIn('set -o noclobber', active)
            self.assertIn('${remote_root}/build/klft', active)

    def test_both_stages_bind_the_same_pushed_revision(self) -> None:
        commits = [re.search(r"^expected_commit=([0-9a-f]{40})$",
                             path.read_text(), re.MULTILINE).group(1)
                   for path in SCRIPTS]
        self.assertEqual(commits, ["c78efbd1ed6c60fc809db54ac9309fa532854982"] * 2)
        self.assertIn("production_timing pending", SCRIPTS[1].read_text())
        self.assertIn("artifact_sha256", SCRIPTS[0].read_text())
        self.assertIn("sha256sum --check -", SCRIPTS[1].read_text())


if __name__ == "__main__":
    unittest.main()
