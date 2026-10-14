#!/usr/bin/env python3
"""Check four short Nt=1 A100 runs after native GPU correctness validation.

The inputs are immutable, independent of the P100 runs, and contain at most
100 sweeps. This test produces timing and integrity records, without a mass
estimate, bootstrap, production submission or fixed-c continuum input.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import socket
import subprocess
import time

from mg_nt_fixed_nt1_run import ROOT, output_check, sha256, write_json


EXPECTED_COMMIT = "f47421d895469b72d4af8a9b8479807f2b09a692"
GPU = "GPU-439f10b2-3f2f-c6d2-12cc-deca128ca0f2"


def check_gpu_idle() -> None:
    """Refuse to overlap an existing compute process on the selected device."""
    pids = subprocess.check_output(
        ["nvidia-smi", "-i", GPU, "--query-compute-apps=pid",
         "--format=csv,noheader,nounits"], text=True).strip()
    if pids:
        raise RuntimeError(f"A100 already occupied by PID(s): {pids}")


def main() -> None:
    """Execute only the prepared smoke and 100-sweep timing inputs on lnode19."""
    if socket.gethostname() != "lnode19" or os.environ.get("CUDA_VISIBLE_DEVICES") != GPU:
        raise RuntimeError("Use the checked-in direct A100 test launcher")
    data = ROOT / "data/mg_nt_fixed"
    records = ROOT / "campaigns/mg_nt_fixed"
    binary = ROOT / "build/klft-a100/binaries/heatbath"
    manifest_path = data / "a100_test_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    binary_hash = sha256(binary)
    validation = (records / "open_slab_a100_validation.txt").read_text().splitlines()
    for required in (f"klft_commit {EXPECTED_COMMIT}", f"heatbath_sha256 {binary_hash}",
                     "ctest passed", "cuda_memcheck passed"):
        if required not in validation:
            raise ValueError("Missing or stale native A100 validation")
    runs = manifest["runs"]
    if (manifest["production_authorized"] or len(runs) != 4 or
            sorted(row["sweeps"] for row in runs) != [8, 100, 100, 100] or
            sorted(row["purpose"] for row in runs) != ["smoke", "timing", "timing", "timing"]):
        raise ValueError("This launcher is restricted to the four short test inputs")
    # Validate the whole input set before running its first member.
    for run in runs:
        relative = Path(run["input"])
        directory = data / relative.parent
        if (relative.is_absolute() or ".." in relative.parts or
                str(directory) != run["raw_path"] or run["klft_commit"] != EXPECTED_COMMIT or
                run["physical_Nt"] != 1 or run["stored_L2"] != 2 or
                run["fixed_c_continuum_input"] or run["target_c"] is not None or
                run["production_authorized"]):
            raise ValueError("Invalid A100 test manifest")
        if sha256(directory / "input.yaml") != run["input_sha256"]:
            raise ValueError("Test input checksum mismatch")
        if any(path.name not in {"input.yaml", "run_metadata.json"} for path in directory.iterdir()):
            raise FileExistsError(f"Refusing existing output: {directory}")
    results = []
    # Smoke first, then increasing volumes; never start a larger case after failure.
    for run in sorted(runs, key=lambda row: row["Ns"]):
        check_gpu_idle()
        directory = Path(run["raw_path"])
        start = dict(run, pid=os.getpid(), node=socket.gethostname(), gpu_uuid=GPU,
                     heatbath_sha256=binary_hash, manifest_sha256=sha256(manifest_path),
                     runner_sha256=sha256(Path(__file__)),
                     validation_sha256=sha256(records / "open_slab_a100_validation.txt"))
        write_json(directory / "execution_start.json", start)
        before = time.monotonic()
        with (directory / "run.log").open("x") as log:
            subprocess.run([str(binary), "-f", "input.yaml"], cwd=directory,
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=900)
        elapsed = time.monotonic() - before
        checks = output_check(directory, run["Ns"], run["sweeps"])
        outputs = {name: sha256(directory / name) for name in (
            "holonomy_correlator.out", "holonomy_summary.out", "plaquette_profile.out",
            "configuration_final.klft")}
        result = dict(start, status="complete", elapsed_seconds=elapsed,
                      checks=checks, output_sha256=outputs)
        write_json(directory / "execution_result.json", result)
        results.append(result)
        print(json.dumps({"Ns": run["Ns"], "beta": run["beta"], "sweeps": run["sweeps"],
                          "elapsed_seconds": elapsed, "checks": checks}), flush=True)
    timings = [row for row in results if row["purpose"] == "timing"]
    for row in timings:
        row["projected_8000_sweep_A100_hours"] = row["elapsed_seconds"] * 80. / 3600.
    report = {
        "status": "short_tests_passed", "backend": "CUDA/AMPERE80",
        "production_authorized": False, "p100_validation_unchanged": True,
        "volume_status": "unvalidated", "bootstrap_operations": 0,
        "timing_scope": "whole heatbath process, including startup, output and checkpoint",
        "projected_four_chains_per_beta_A100_hours":
            4. * sum(row["projected_8000_sweep_A100_hours"] for row in timings),
        "runs": results,
    }
    write_json(records / "open_slab_a100_runs.json", report)


if __name__ == "__main__":
    main()
