#!/usr/bin/env python3
"""Run and check one immutable Nt=1 input on a qbig P100 allocation.

This is an execution check, with no thermalization selection or bootstrap.
It never submits another job. Timing approval is a separate manual stage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import time
import shutil
from datetime import datetime, timezone

import numpy as np


ROOT = Path("/qbigwork2/negro/dirichlet")
DATA = ROOT / "data/mg_nt_fixed"
RECORDS = ROOT / "campaigns/mg_nt_fixed"
SOURCE = ROOT / "src/klft"
BINARY = ROOT / "build/klft/binaries/heatbath"
HEADER = struct.Struct("=8s5I3qQ")


def sha256(path: Path) -> str:
    """Hash a file in bounded memory; raw histories remain on qbig."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    """Write an exclusive compact record, refusing replacement."""
    with path.open("x") as stream:
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")


def checkpoint_check(path: Path, ns: int) -> dict:
    """Verify KLFT's native flag-3 Nt=1 checkpoint and every physical link.

    Layout follows KLFT include/io/gauge_configuration.hpp: x,y,t,mu,
    omitting the closing temporal link. Each spatial site stores five SU(2)
    quaternions: two lower boundary links, the temporal link, two upper links.
    """
    with path.open("rb") as stream:
        header = HEADER.unpack(stream.read(HEADER.size))
    expected = (b"KLFTCFG1", 1, 3, 2, 8, 3, ns, ns, 2, 5 * ns * ns)
    if header != expected or path.stat().st_size != HEADER.size + ns * ns * 5 * 32:
        raise ValueError("Checkpoint geometry, flag or size mismatch")
    links = np.memmap(path, mode="r", dtype=np.float64,
                      offset=HEADER.size, shape=(ns * ns, 5, 4))
    max_error = 0.0
    for start in range(0, len(links), 4096):
        chunk = links[start:start + 4096]
        if not np.isfinite(chunk).all():
            raise ValueError("Nonfinite checkpoint link")
        if not np.all(chunk[:, (0, 1, 3, 4), :] == np.array([1., 0., 0., 0.])):
            raise ValueError("Dirichlet boundary link changed")
        max_error = max(max_error, float(np.max(np.abs(np.sum(chunk * chunk, axis=2) - 1.))))
    if max_error > 1.e-10:
        raise ValueError(f"SU(2) norm error {max_error}")
    return {"boundary_flag": 3, "physical_links": 5 * ns * ns,
            "max_squared_norm_error": max_error}


def output_check(directory: Path, ns: int, sweeps: int) -> dict:
    """Check complete measurement order, cadence, finite observables and C_G(0).

    Per-configuration log slopes and xi_2nd can be undefined; the underlying
    correlators and structure factors must remain finite.
    """
    summary = np.loadtxt(directory / "holonomy_summary.out", ndmin=2)
    if (summary.shape != (sweeps, 6) or
            not np.array_equal(summary[:, 0], np.arange(1, sweeps + 1)) or
            not np.isfinite(summary[:, :5]).all() or
            not np.all(summary[:, 2] == 0.) or np.any(summary[:, 3:5] < 0.)):
        raise ValueError("Invalid holonomy summary or cadence")
    count = 0
    nr = ns // 2 + 1
    with (directory / "holonomy_correlator.out").open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.split()
            step, r = divmod(count, nr)
            if (len(values) != 5 or int(values[0]) != step + 1 or int(values[1]) != r or
                    not all(math.isfinite(float(v)) for v in values[2:4]) or
                    abs(float(values[2])) > 1. + 1.e-10 or
                    (r == 0 and float(values[2]) != 1.)):
                raise ValueError(f"Invalid correlator row {count}")
            count += 1
    if count != sweeps * nr:
        raise ValueError("Incomplete correlator history")
    profiles = np.loadtxt(directory / "plaquette_profile.out", ndmin=2)
    steps = np.arange(5, sweeps + 1, 5)
    keys = np.tile(np.array([[0, 0.], [0, 1.], [1, .5]]), (len(steps), 1))
    if (profiles.shape != (3 * len(steps), 4) or
            not np.isfinite(profiles).all() or
            not np.array_equal(profiles[:, 0], np.repeat(steps, 3)) or
            not np.array_equal(profiles[:, 1:3], keys) or
            not np.all(profiles[profiles[:, 1] == 0, 3] == 1.) or
            np.any(np.abs(profiles[:, 3]) > 1. + 1.e-10)):
        raise ValueError("Invalid plaquette profile, boundary or cadence")
    checkpoint = checkpoint_check(directory / "configuration_final.klft", ns)
    return {"summary_measurements": sweeps, "correlator_rows": count,
            "profile_measurements": len(steps), "checkpoint": checkpoint}


def checked_result(run: dict) -> dict:
    """Read a completed result and bind it to the exact input and executable."""
    record = json.loads((Path(run["raw_path"]) / "execution_result.json").read_text())
    for key in ("input_sha256", "heatbath_sha256"):
        if record[key] != run[key]:
            raise ValueError(f"Result {key} mismatch")
    if record["status"] != "complete":
        raise ValueError("Run did not complete")
    return record


def timing_projection(manifest: dict) -> dict:
    """Project the four 8,000-sweep chains at each beta from completed timings.

    Whole-process wall time includes initialization, file output and checkpoint
    saving, making its linear extrapolation conservative for fixed-size costs.
    """
    timings = [run for run in manifest["runs"] if run["purpose"] == "timing"]
    production = [run for run in manifest["runs"] if run["purpose"] == "production"]
    if len(timings) not in (3, 4) or len(production) != 4 * len(timings):
        raise ValueError("Incomplete Nt=1 manifest")
    projections = []
    for run in production:
        matching = [item for item in timings if
                    (item["Ns"], item["beta_text"], item["n_overrelax"]) ==
                    (run["Ns"], run["beta_text"], run["n_overrelax"])]
        if len(matching) != 1:
            raise ValueError("Missing unique production-cadence timing")
        timing = matching[0]
        result = checked_result(timing)
        seconds = float(result["elapsed_seconds"])
        if not math.isfinite(seconds) or seconds <= 0.:
            raise ValueError("Invalid timing")
        projections.append({"seed": run["seed"], "gpu_hours":
                            seconds / timing["sweeps"] * run["sweeps"] / 3600.})
    total = sum(row["gpu_hours"] for row in projections)
    largest = max(row["gpu_hours"] for row in projections)
    return {"passed": total <= 12. and largest <= 2.,
            "projected_total_gpu_hours": total, "largest_chain_gpu_hours": largest,
            "chains": projections}


def validate_scaling_manifest(manifest: dict) -> None:
    """Restrict the new batch to the four user-approved Ns=128 ensembles."""
    beta_grid = {1.45, 1.5, float('1.5333333333333333'), 1.55}
    runs = manifest['runs']
    production = [r for r in runs if r['purpose']=='production']
    timings = [r for r in runs if r['purpose']=='timing']
    smoke = [r for r in runs if r['purpose']=='smoke']
    if (manifest['backend'] != 'p100' or manifest['analysis_bootstrap_samples'] != 500
            or len(runs) != 21 or len(production) != 16 or len(timings) != 4 or len(smoke) != 1
            or len({r['seed'] for r in runs}) != 21
            or {r['seed'] for r in runs} & set(manifest['reserved_seeds'])):
        raise ValueError('Invalid Nt=1 matched-y scope or seeds')
    for beta in beta_grid:
        if sorted(r['chain_index'] for r in production if r['beta']==beta) != [0,1,2,3]:
            raise ValueError('Need four independent chains at each approved beta')
    if {r['beta'] for r in timings} != beta_grid or smoke[0]['beta'] != 1.55:
        raise ValueError('Invalid timing or smoke couplings')
    for run in runs:
        expected = dict(physical_Nt=1, stored_L2=2, Ns=4 if run['purpose']=='smoke' else 128,
                        sweeps={'smoke':10, 'timing':100, 'production':8000}[run['purpose']],
                        target_c=None, reference_c=.5, fixed_c_continuum_input=False,
                        campaign_type='matched_y_scaling', start='hot', n_overrelax=5,
                        measurement_interval=1, dirichlet_suite_measurement_interval=5,
                        boundary_wilson_enabled=False)
        if any(run[k] != value for k,value in expected.items()):
            raise ValueError('Approved geometry, cadence or campaign role changed')


def run_one(purpose: str, index: int, campaign: str = 'nt1') -> None:
    """Run one checked input in an existing typed-P100 Slurm allocation."""
    if not os.environ.get("SLURM_JOB_ID") or os.environ.get("SLURMD_NODENAME") not in {
            "lnode13", "lnode14", "lnode15", "lnode16", "lnode17"}:
        raise RuntimeError("Use the P100 Slurm submission script")
    if campaign not in ('nt1', 'nt1_scaling'):
        raise ValueError('Unknown Nt=1 campaign')
    manifest_path = DATA / f"{campaign}_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if campaign == 'nt1_scaling':
        validate_scaling_manifest(manifest)
        for name, digest in manifest['source_sha256'].items():
            if Path(name).name != name or sha256(RECORDS/name) != digest:
                raise ValueError(f'Execution source changed: {name}')
        if sha256(RECORDS/'nt1_scaling_run.slurm') != manifest['submission_sha256']:
            raise ValueError('Submission script changed')
        if sha256(ROOT/'project/uv.lock') != manifest['uv_lock_sha256']:
            raise ValueError('Locked environment changed')
        if sha256(RECORDS/'open_slab_validation.txt') != manifest['validation_sha256']:
            raise ValueError('GPU validation record changed')
        if shutil.disk_usage(ROOT).free < 12 * 1024**3:
            raise RuntimeError('Free disk space below the 12 GiB reserve')
    candidates = [row for row in manifest["runs"] if row["purpose"] == purpose]
    if index < 0 or index >= len(candidates):
        raise ValueError("Invalid task index")
    run = candidates[index]
    directory = Path(run["raw_path"])
    if directory != DATA / Path(run["input"]).parent or not directory.is_relative_to(DATA):
        raise ValueError("Run path mismatch")
    if any(path.name not in {"input.yaml", "run_metadata.json"} for path in directory.iterdir()):
        raise FileExistsError(f"Refusing existing output in {directory}")
    if sha256(directory / "input.yaml") != run["input_sha256"]:
        raise ValueError("Input checksum mismatch")
    if sha256(BINARY) != run["heatbath_sha256"]:
        raise ValueError("Executable checksum mismatch")
    if sha256(RECORDS / "open_slab_build.txt") != manifest["build_provenance_sha256"]:
        raise ValueError("Build provenance mismatch")
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain",
                                     "--ignore-submodules=none"], text=True).strip()
    if commit != run["klft_commit"] or dirty:
        raise ValueError("Source revision changed")
    validation = (RECORDS / "open_slab_validation.txt").read_text().splitlines()
    for line in (f"klft_commit {commit}", f"heatbath_sha256 {run['heatbath_sha256']}",
                 "ctest passed", "cuda_memcheck passed", "benchmark passed"):
        if line not in validation:
            raise ValueError("P100 validation is missing or stale")
    if purpose != "smoke":
        smoke, = [row for row in manifest["runs"] if row["purpose"] == "smoke"]
        checked_result(smoke)
    if purpose == "production":
        gate = json.loads((RECORDS / f"{campaign}_timing_gate.json").read_text())
        if gate["manifest_sha256"] != sha256(manifest_path) or not gate["passed"]:
            raise ValueError("Manual timing gate missing or stale")
        if not timing_projection(manifest)["passed"]:
            raise ValueError("Production would exceed the timing budget")
    start = {**run, "job_id": os.environ["SLURM_JOB_ID"],
             "array_index": os.environ.get("SLURM_ARRAY_TASK_ID"),
             "node": os.environ["SLURMD_NODENAME"],
             "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
             "input_sha256": run["input_sha256"], "heatbath_sha256": run["heatbath_sha256"],
             "manifest_sha256": sha256(manifest_path), "klft_commit": commit,
             "runner_sha256": sha256(Path(__file__)), "purpose": purpose,
             "started_at": datetime.now(timezone.utc).isoformat()}
    # Exclusive creation also prevents two duplicate jobs from appending.
    write_json(directory / "execution_start.json", start)
    before = time.monotonic()
    try:
        subprocess.run([str(BINARY), "-f", "input.yaml"], cwd=directory, check=True, timeout=7200)
        elapsed = time.monotonic() - before
        checks = output_check(directory, run["Ns"], run["sweeps"])
        outputs = {name: sha256(directory / name) for name in
                   ("holonomy_correlator.out", "holonomy_summary.out", "plaquette_profile.out",
                    "configuration_final.klft")}
        result = dict(start, status="complete", elapsed_seconds=elapsed,
                      completed_at=datetime.now(timezone.utc).isoformat(),
                      checks=checks, output_sha256=outputs)
        write_json(directory / "execution_result.json", result)
    except Exception as error:
        write_json(directory / "execution_result.json", dict(start, status='failed',
                   error=f'{type(error).__name__}: {error}', ended_at=datetime.now(timezone.utc).isoformat()))
        raise
    print(json.dumps(result, indent=2), flush=True)


def main() -> None:
    """Execute a manually selected stage or write its timing gate."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("purpose", choices=("smoke", "timing", "production", "timing_gate"))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument('--campaign', choices=('nt1', 'nt1_scaling'), default='nt1')
    args = parser.parse_args()
    if args.purpose == "timing_gate":
        path = DATA / f"{args.campaign}_manifest.json"
        manifest = json.loads(path.read_text())
        if args.campaign == 'nt1_scaling':
            validate_scaling_manifest(manifest)
            manifest_hash = sha256(path)
            for run in manifest['runs']:
                if run['purpose'] in ('smoke', 'timing'):
                    record = checked_result(run)
                    if record['manifest_sha256'] != manifest_hash:
                        raise ValueError('Preflight manifest changed')
                    for name,digest in record['output_sha256'].items():
                        if Path(name).name != name or sha256(Path(run['raw_path'])/name) != digest:
                            raise ValueError('Preflight output changed')
        result = timing_projection(manifest)
        result["manifest_sha256"] = sha256(path)
        write_json(RECORDS / f"{args.campaign}_timing_gate.json", result)
        print(json.dumps(result, indent=2))
        if not result["passed"]:
            raise SystemExit("Timing budget exceeded; production remains forbidden")
    else:
        run_one(args.purpose, args.index, args.campaign)


if __name__ == "__main__":
    main()
