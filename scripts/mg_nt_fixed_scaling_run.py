#!/usr/bin/env python3
"""Execute the authorized matched-y batch on three idle lnode19 A100 GPUs.

Preflight, timing approval and production are separate manual commands.
No command submits a subsequent stage or performs statistical analysis.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

import numpy as np

from mg_nt_fixed_nt1_a100 import check_gpu_idle, check_unused_input, utc_now
from mg_nt_fixed_nt1_run import HEADER, ROOT, sha256, write_json


DATA = ROOT / "data/mg_nt_fixed"
RECORDS = ROOT / "campaigns/mg_nt_fixed"
SOURCE = ROOT / "src/klft"
OUTPUTS = ("holonomy_correlator.out", "holonomy_summary.out",
           "plaquette_profile.out", "configuration_final.klft")
GRID = {(3, 96, 4.5), (4, 160, 6.0), (6, 320, 9.0), (4, 160, 6.2)}
GPUS = ("GPU-439f10b2-3f2f-c6d2-12cc-deca128ca0f2",
        "GPU-94eef473-cba0-0850-d8e5-dc5dafa2493a",
        "GPU-97c59efd-ea9e-94d9-7f18-1959e77798e4")


def checkpoint_check(path: Path, ns: int, nt: int) -> dict:
    """Check all physical links in KLFT's flag-3 open-slab checkpoint.

    Layout is defined by KLFT include/io/gauge_configuration.hpp. The final
    coordinate (time) varies fastest; every transverse site has 3*Nt+2 links.
    Boundary spatial links are at offsets 0,1 and 3*Nt,3*Nt+1.
    """
    stride = 3 * nt + 2
    nlinks = ns * ns * stride
    with path.open("rb") as stream:
        header = HEADER.unpack(stream.read(HEADER.size))
    if (header != (b"KLFTCFG1", 1, 3, 2, 8, 3, ns, ns, nt + 1, nlinks)
            or path.stat().st_size != HEADER.size + nlinks * 32):
        raise ValueError("Checkpoint geometry, flag or size mismatch")
    links = np.memmap(path, mode="r", dtype=np.float64,
                      offset=HEADER.size, shape=(ns * ns, stride, 4))
    max_error = 0.0
    for start in range(0, len(links), 4096):
        chunk = links[start:start + 4096]
        if not np.isfinite(chunk).all():
            raise ValueError("Nonfinite checkpoint link")
        if not np.all(chunk[:, (0, 1, 3 * nt, 3 * nt + 1)] == [1., 0., 0., 0.]):
            raise ValueError("Dirichlet boundary link changed")
        max_error = max(max_error, float(np.max(np.abs(np.sum(chunk * chunk, axis=2) - 1.))))
    if max_error > 1.e-10:
        raise ValueError(f"SU(2) norm error {max_error}")
    return dict(boundary_flag=3, physical_Nt=nt, stored_L2=nt + 1,
                physical_links=nlinks, max_squared_norm_error=max_error)


def output_check(directory: Path, ns: int, nt: int, sweeps: int) -> dict:
    """Check complete cadences, C_G(0), primitive observables and boundaries."""
    summary = np.loadtxt(directory / "holonomy_summary.out", ndmin=2)
    if (summary.shape != (sweeps, 6)
            or not np.array_equal(summary[:, 0], np.arange(1, sweeps + 1))
            or not np.isfinite(summary[:, :5]).all()
            or not np.all(summary[:, 2] == 0.) or np.any(summary[:, 3:5] < 0.)):
        raise ValueError("Invalid holonomy summary or cadence")
    count, nr = 0, ns // 2 + 1
    with (directory / "holonomy_correlator.out").open() as stream:
        for line in stream:
            if not line.strip() or line.startswith("#"):
                continue
            values = line.split()
            step, r = divmod(count, nr)
            if (len(values) != 5 or int(values[0]) != step + 1 or int(values[1]) != r
                    or not all(math.isfinite(float(v)) for v in values[2:4])
                    or abs(float(values[2])) > 1. + 1.e-10
                    or (r == 0 and float(values[2]) != 1.)):
                raise ValueError(f"Invalid correlator row {count}")
            count += 1
    if count != sweeps * nr:
        raise ValueError("Incomplete correlator history")
    profiles = np.loadtxt(directory / "plaquette_profile.out", ndmin=2)
    steps = np.arange(5, sweeps + 1, 5)
    keys = np.array([(0, t) for t in range(nt + 1)] + [(1, t + .5) for t in range(nt)])
    if (profiles.shape != (len(steps) * len(keys), 4)
            or not np.isfinite(profiles).all()
            or not np.array_equal(profiles[:, 0], np.repeat(steps, len(keys)))
            or not np.array_equal(profiles[:, 1:3], np.tile(keys, (len(steps), 1)))
            or np.any(np.abs(profiles[:, 3]) > 1. + 1.e-10)):
        raise ValueError("Invalid plaquette profile or cadence")
    boundary = (profiles[:, 1] == 0) & np.isin(profiles[:, 2], [0, nt])
    if not np.all(profiles[boundary, 3] == 1.):
        raise ValueError("Invalid fixed-surface plaquette")
    return dict(summary_measurements=sweeps, correlator_rows=count,
                profile_measurements=len(steps), profile_rows=len(profiles),
                checkpoint=checkpoint_check(directory / "configuration_final.klft", ns, nt))


def validate_manifest(manifest: dict) -> None:
    """Restrict execution to the four approved ensembles and independent chains."""
    runs = manifest["runs"]
    prod = [r for r in runs if r["purpose"] == "production"]
    timings = [r for r in runs if r["purpose"] == "timing"]
    smoke = [r for r in runs if r["purpose"] == "smoke"]
    if (manifest["backend"] != "a100" or manifest["analysis_bootstrap_samples"] != 500
            or len(runs) != 21 or len(prod) != 16 or len(timings) != 4 or len(smoke) != 1
            or len({r["seed"] for r in runs}) != 21
            or {r["seed"] for r in runs} & set(manifest["reserved_seeds"])):
        raise ValueError("Invalid approved scope or seed policy")
    for nt, ns, beta in GRID:
        matching = [r for r in prod if (r["physical_Nt"], r["Ns"], r["beta"]) == (nt, ns, beta)]
        if sorted(r["chain_index"] for r in matching) != [0, 1, 2, 3]:
            raise ValueError("Each ensemble needs four independent chains")
    if {(r["physical_Nt"], r["Ns"], r["beta"]) for r in timings} != GRID:
        raise ValueError("Timings do not cover every production geometry")
    if (smoke[0]["physical_Nt"], smoke[0]["Ns"], smoke[0]["beta"]) != (6, 4, 9.):
        raise ValueError("Invalid small-slab smoke test")
    for run in runs:
        expected = dict(sweeps={"smoke": 10, "timing": 100, "production": 8000}[run["purpose"]],
                        stored_L2=run["physical_Nt"] + 1, target_c=None,
                        fixed_c_continuum_input=False, n_overrelax=5,
                        measurement_interval=1, dirichlet_suite_measurement_interval=5,
                        boundary_wilson_enabled=False, start="hot")
        if any(run[k] != v for k, v in expected.items()):
            raise ValueError("Physics, duration or measurement policy changed")


def checked_result(run: dict, manifest_hash: str) -> dict:
    """Bind completion to the exact input, executable, manifest and saved outputs."""
    directory = Path(run["raw_path"])
    result = json.loads((directory / "execution_result.json").read_text())
    if (result["status"] != "complete" or result["manifest_sha256"] != manifest_hash
            or any(result[k] != run[k] for k in ("input_sha256", "heatbath_sha256"))):
        raise ValueError("Missing or stale completed preflight")
    if any(sha256(directory / name) != digest for name, digest in result["output_sha256"].items()):
        raise ValueError("Preflight output changed")
    return result


def timing_projection(manifest: dict, manifest_hash: str) -> dict:
    """Project all sixteen chains using the matching 100-sweep process timings."""
    timings = {}
    for run in manifest["runs"]:
        if run["purpose"] in {"smoke", "timing"}:
            result = checked_result(run, manifest_hash)
            if run["purpose"] == "timing":
                seconds = float(result["elapsed_seconds"])
                if not math.isfinite(seconds) or seconds <= 0.:
                    raise ValueError("Invalid timing")
                timings[(run["physical_Nt"], run["Ns"], run["beta"])] = seconds
    chains = [dict(seed=r["seed"], gpu_hours=timings[(r["physical_Nt"], r["Ns"], r["beta"])] * 80 / 3600)
              for r in manifest["runs"] if r["purpose"] == "production"]
    total = sum(r["gpu_hours"] for r in chains)
    largest = max(r["gpu_hours"] for r in chains)
    workers = [sum(r["gpu_hours"] for i, r in enumerate(chains) if i % 3 == w) for w in range(3)]
    return dict(passed=total <= 12. and largest <= 2. and max(workers) <= 2.,
                projected_worker_gpu_hours=workers, projected_total_gpu_hours=total,
                largest_chain_gpu_hours=largest, chains=chains, manifest_sha256=manifest_hash)


def run_stage(stage: str, worker_index: int) -> None:
    """Run preflight or one manually authorized, bounded production worker."""
    manifest_path, plan_path = DATA / "scaling_manifest.json", DATA / "scaling_execution.json"
    manifest, plan = (json.loads(p.read_text()) for p in (manifest_path, plan_path))
    validate_manifest(manifest)
    manifest_hash = sha256(manifest_path)
    if (plan["manifest_sha256"] != manifest_hash or plan["production_authorized"] is not True
            or plan["node"] != "lnode19" or plan["backend"] != "CUDA/AMPERE80"
            or plan["min_free_bytes"] < 12 * 1024**3):
        raise ValueError("Execution authorization changed")
    if socket.gethostname() != plan["node"] or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the authorized non-Slurm A100 node")
    workers = plan["workers"]
    expected_workers = [dict(index=i, gpu_uuid=gpu, cpu=32+i) for i, gpu in enumerate(GPUS)]
    if workers != expected_workers:
        raise ValueError("Use GPUs 3,4,5 on CPUs 32,33,34; leave GPU 6 free")
    worker, = [w for w in workers if w["index"] == worker_index]
    gpu = worker["gpu_uuid"]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or os.sched_getaffinity(0) != {worker["cpu"]}:
        raise RuntimeError("GPU or CPU affinity does not match the plan")
    for name, digest in plan["source_sha256"].items():
        if Path(name).name != name or sha256(RECORDS / name) != digest:
            raise ValueError(f"Execution source changed: {name}")
    if sha256(ROOT / "project/uv.lock") != plan["uv_lock_sha256"]:
        raise ValueError("Locked environment changed")
    for field in ("build_provenance", "validation"):
        if sha256(RECORDS / manifest[f"{field}_file"]) != manifest[f"{field}_sha256"]:
            raise ValueError(f"Native {field} changed")
    binary = Path(manifest["executable"])
    if binary != ROOT / "build/klft-a100/binaries/heatbath" or sha256(binary) != manifest["runs"][0]["heatbath_sha256"]:
        raise ValueError("Native executable changed")
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain",
                                     "--ignore-submodules=none"], text=True).strip()
    if dirty or commit != manifest["runs"][0]["klft_commit"]:
        raise ValueError("Source revision changed")
    if stage == "preflight":
        if worker_index != 0:
            raise ValueError("Preflight uses only worker zero")
        runs = [r for r in manifest["runs"] if r["purpose"] in {"smoke", "timing"}]
    else:
        gate = json.loads((RECORDS / "scaling_timing_gate.json").read_text())
        if gate["manifest_sha256"] != manifest_hash or not gate["passed"]:
            raise ValueError("Manual timing approval missing")
        if not timing_projection(manifest, manifest_hash)["passed"]:
            raise ValueError("Timing budget exceeded")
        production = [r for r in manifest["runs"] if r["purpose"] == "production"]
        runs = [r for i, r in enumerate(production) if i % 3 == worker_index]
    for run in runs:
        check_unused_input(run)
    start = dict(stage=stage, worker=worker_index, pid=os.getpid(), node=plan["node"],
                 gpu_uuid=gpu, cpu=worker["cpu"], manifest_sha256=manifest_hash,
                 execution_plan_sha256=sha256(plan_path), started_at=utc_now(),
                 seeds=[r["seed"] for r in runs], source_sha256=plan["source_sha256"])
    check_gpu_idle(gpu)
    write_json(RECORDS / f"scaling_{stage}_worker_{worker_index}_start.json", start)
    completed = []
    try:
        for run in runs:
            check_gpu_idle(gpu)
            if shutil.disk_usage(ROOT).free < plan["min_free_bytes"]:
                raise RuntimeError("Free storage below the production reserve")
            directory = check_unused_input(run)
            record = dict(run, **start)
            record["started_at"] = utc_now()
            write_json(directory / "execution_start.json", record)
            print(json.dumps(dict(status="started", seed=run["seed"], Nt=run["physical_Nt"],
                                  Ns=run["Ns"], beta=run["beta"], sweeps=run["sweeps"])), flush=True)
            before = time.monotonic()
            with (directory / "run.log").open("x") as log:
                subprocess.run([str(binary), "-f", "input.yaml"], cwd=directory, stdout=log,
                               stderr=subprocess.STDOUT, check=True,
                               timeout=7200 if stage == "production" else 600)
            elapsed = time.monotonic() - before
            checks = output_check(directory, run["Ns"], run["physical_Nt"], run["sweeps"])
            result = dict(record, status="complete", completed_at=utc_now(),
                          elapsed_seconds=elapsed, checks=checks,
                          output_sha256={name: sha256(directory / name) for name in OUTPUTS})
            write_json(directory / "execution_result.json", result)
            completed.append(run["seed"])
            print(json.dumps(dict(status="complete", seed=run["seed"],
                                  elapsed_seconds=elapsed, checks=checks)), flush=True)
        write_json(RECORDS / f"scaling_{stage}_worker_{worker_index}_result.json",
                   dict(start, status="complete", completed_at=utc_now(), completed_seeds=completed))
    except Exception as error:
        write_json(RECORDS / f"scaling_{stage}_worker_{worker_index}_result.json",
                   dict(start, status="failed", ended_at=utc_now(), completed_seeds=completed,
                        error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("preflight", "timing_gate", "production"))
    parser.add_argument("--worker", type=int, choices=(0, 1, 2), default=0)
    args = parser.parse_args()
    if args.stage == "timing_gate":
        path = DATA / "scaling_manifest.json"
        manifest = json.loads(path.read_text())
        validate_manifest(manifest)
        gate = timing_projection(manifest, sha256(path))
        write_json(RECORDS / "scaling_timing_gate.json", gate)
        print(json.dumps(gate, indent=2))
        if not gate["passed"]:
            raise SystemExit("Timing budget exceeded")
    else:
        run_stage(args.stage, args.worker)
