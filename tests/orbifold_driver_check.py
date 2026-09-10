"""Check the 2+1D SU(3) CLI, restart, and dimension guard in a temporary folder.

Run with uv run --no-project tests/orbifold_driver_check.py /path/to/orbifold_hmc.
Uses only the Python standard library; no simulation output is retained.
"""

import math
from pathlib import Path
import subprocess
import sys
import tempfile


def run_check(executable: Path) -> None:
    """Verify finite complete histories, checkpoint restart, and input guards."""
    fixture = Path(__file__).resolve().parents[1] / "examples/orbifold_2p1_smoke.yaml"
    text = fixture.read_text()
    with tempfile.TemporaryDirectory(prefix="klft-orbifold-driver-") as temporary:
        directory = Path(temporary)

        def run_input(name: str, source: str, success: bool = True) -> str:
            path = directory / name
            path.write_text(source)
            result = subprocess.run(
                [str(executable), "-f", str(path)], cwd=directory,
                text=True, capture_output=True, timeout=60,
            )
            assert (result.returncode == 0) == success, result.stdout + result.stderr
            return result.stdout + result.stderr

        run_input("smoke.yaml", text)
        for name, count in (("hmc.out", 10), ("wilson.out", 20),
                            ("diagnostic.out", 10)):
            rows = [list(map(float, line.split()))
                    for line in (directory / name).read_text().splitlines()
                    if line and not line.startswith("#")]
            assert len(rows) == count, (name, len(rows))
            assert all(math.isfinite(value) for row in rows for value in row)
        assert "time_direction=2" in (directory / "hmc.out").read_text()
        assert (directory / "final.cfg").stat().st_size > 0
        assert "Refusing to overwrite" in run_input("again.yaml", text, False)
        assert "incompatible with this build" in run_input(
            "wrong-rank.yaml", text.replace("  L2: 4", "  L2: 4\n  L3: 4"), False)
        assert "must be positive" in run_input(
            "nonfinite.yaml", text.replace("g: 1.0", "g: .inf"), False)

        restart = text.replace("start: hot", "start: restart\n  configuration_input: final.cfg")
        restart = restart.replace("thermalization_trajectories: 5", "thermalization_trajectories: 0")
        restart = restart.replace("production_trajectories: 5", "production_trajectories: 1")
        restart = restart.replace("checkpoint_every: 5", "checkpoint_every: 1")
        for name in ("hmc.out", "wilson.out", "diagnostic.out"):
            restart = restart.replace(name, "restart_" + name)
        restart = restart.replace("configuration_output: final.cfg",
                                  "configuration_output: restart.cfg")
        run_input("restart.yaml", restart)
        assert (directory / "restart.cfg").stat().st_size == (directory / "final.cfg").stat().st_size
    print("2+1D orbifold driver smoke, restart, and input guards passed")


if __name__ == "__main__":
    run_check(Path(sys.argv[1]).resolve())
