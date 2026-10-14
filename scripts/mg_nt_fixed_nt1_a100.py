#!/usr/bin/env python3
"""Run one three-ensemble Nt=1 worker on an approved, idle lnode19 A100.

The execution plan reuses immutable unused production inputs from the P100
plan. Original planning metadata stays unchanged; actual A100 provenance is
recorded in execution_start/result.json. No mass analysis or bootstrap runs.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import socket
import subprocess
import time

from mg_nt_fixed_nt1_run import ROOT, output_check, sha256, write_json


DATA = ROOT / "data/mg_nt_fixed"
RECORDS = ROOT / "campaigns/mg_nt_fixed"
SOURCE = ROOT / "src/klft"
PLAN = DATA / "nt1_a100_production.json"


def utc_now() -> str:
    """Return an explicit UTC timestamp for execution provenance."""
    return datetime.now(timezone.utc).isoformat()


def production_runs(plan: dict, source: dict, timings: dict) -> tuple[list[dict], dict]:
    """Bind all twelve inputs to successful native timings and enforce cost limits."""
    if (plan["production_authorized"] is not True or plan["node"] != "lnode19" or
            plan["backend"] != "CUDA/AMPERE80" or plan["target_c"] is not None or
            plan["fixed_c_continuum_input"] or plan["analysis_bootstrap_samples"] != 500 or
            plan["worker_time_limit_seconds"] != 7200 or plan["min_free_bytes"] < 12 * 1024**3):
        raise ValueError("Invalid production authorization or execution scope")
    workers = plan["workers"]
    if (len(workers) != 4 or sorted(w["index"] for w in workers) != list(range(4)) or
            len({w["gpu_uuid"] for w in workers}) != 4 or
            len({w["cpu"] for w in workers}) != 4):
        raise ValueError("Four distinct GPU/CPU workers are required")
    originals = [r for r in source["runs"] if r["purpose"] == "production"]
    if (len(originals) != 12 or len(plan["runs"]) != 12 or
            len({r["seed"] for r in originals}) != 12 or
            {r["seed"] for r in originals} != {r["seed"] for r in plan["runs"]}):
        raise ValueError("Production inputs or independent seeds are incomplete")
    if timings["status"] != "short_tests_passed":
        raise ValueError("Native smoke and timing tests did not pass")
    runs, projected = [], []
    for assignment in plan["runs"]:
        run, = [r for r in originals if r["seed"] == assignment["seed"]]
        for key in ("chain_index", "Ns", "beta_text", "reference_c", "input", "input_sha256"):
            if assignment[key] != run[key]:
                raise ValueError(f"Original input changed: {key}")
        if assignment["worker"] != run["chain_index"] or run["chain_index"] not in range(4):
            raise ValueError("Invalid independent-chain worker assignment")
        expected = {"physical_Nt": 1, "stored_L2": 2, "sweeps": 8000,
                    "n_overrelax": 5, "measurement_interval": 1,
                    "dirichlet_suite_measurement_interval": 5,
                    "boundary_wilson_enabled": False, "target_c": None,
                    "fixed_c_continuum_input": False, "klft_commit": plan["klft_commit"]}
        if any(run[key] != value for key, value in expected.items()):
            raise ValueError("Nt=1 physics or measurement policy changed")
        matching = [r for r in timings["runs"] if r["purpose"] == "timing" and
                    (r["Ns"], r["beta_text"]) == (run["Ns"], run["beta_text"])]
        if len(matching) != 1:
            raise ValueError("Missing unique matched native A100 timing")
        timing = matching[0]
        for key in ("physical_Nt", "stored_L2", "n_overrelax", "measurement_interval",
                    "dirichlet_suite_measurement_interval", "klft_commit"):
            if timing[key] != run[key]:
                raise ValueError(f"Timing physics mismatch: {key}")
        if (timing["status"] != "complete" or timing["sweeps"] != 100 or
                timing["heatbath_sha256"] != plan["heatbath_sha256"]):
            raise ValueError("Missing or stale completed A100 timing")
        seconds = float(timing["elapsed_seconds"])
        if not math.isfinite(seconds) or seconds <= 0.:
            raise ValueError("Invalid timing duration")
        # Whole-process times include startup, measurements and checkpoint I/O.
        hours = seconds * run["sweeps"] / timing["sweeps"] / 3600.
        projected.append(hours)
        runs.append(dict(run, worker=assignment["worker"], projected_gpu_hours=hours))
    if any(sorted(r["Ns"] for r in runs if r["worker"] == w) != [128, 512, 2048]
           for w in range(4)):
        raise ValueError("Each worker must contain one chain at each matched beta")
    gate = {"projected_total_gpu_hours": sum(projected),
            "largest_chain_gpu_hours": max(projected)}
    if gate["projected_total_gpu_hours"] > 12. or gate["largest_chain_gpu_hours"] > 2.:
        raise ValueError("A100 timing budget exceeded")
    return runs, gate


def check_gpu_idle(gpu: str) -> None:
    """Refuse another compute process on the assigned physical A100."""
    name = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-gpu=name",
                                    "--format=csv,noheader"], text=True).strip()
    if "A100" not in name:
        raise RuntimeError("Assigned device is not an A100")
    pids = subprocess.check_output(["nvidia-smi", "-i", gpu, "--query-compute-apps=pid",
                                    "--format=csv,noheader,nounits"], text=True).strip()
    if pids:
        raise RuntimeError(f"Assigned A100 is occupied by PID(s): {pids}")


def check_unused_input(run: dict) -> Path:
    """Refuse path escapes, altered metadata and all existing execution output."""
    relative = Path(run["input"])
    directory = DATA / relative.parent
    if (relative.is_absolute() or ".." in relative.parts or
            directory.resolve() != directory or str(directory) != run["raw_path"]):
        raise ValueError("Input path mismatch")
    if any(p.name not in {"input.yaml", "run_metadata.json"} for p in directory.iterdir()):
        raise FileExistsError(f"Refusing existing output in {directory}")
    original = {key: value for key, value in run.items()
                if key not in {"worker", "projected_gpu_hours"}}
    if (sha256(directory / "input.yaml") != run["input_sha256"] or
            json.loads((directory / "run_metadata.json").read_text()) != original):
        raise ValueError("Immutable input or original planning metadata changed")
    return directory


def run_worker(index: int) -> None:
    """Run three independent prepared chains with one GPU and a two-hour bound."""
    plan = json.loads(PLAN.read_text())
    if socket.gethostname() != plan["node"] or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Use the approved non-Slurm A100 node")
    worker, = [w for w in plan["workers"] if w["index"] == index]
    gpu = worker["gpu_uuid"]
    if (os.environ.get("CUDA_VISIBLE_DEVICES") != gpu or
            os.sched_getaffinity(0) != {worker["cpu"]}):
        raise RuntimeError("GPU selection or CPU affinity does not match the plan")
    source_path = DATA / "nt1_manifest.json"
    if sha256(source_path) != plan["source_manifest_sha256"]:
        raise ValueError("Original planning manifest changed")
    binary = ROOT / "build/klft-a100/binaries/heatbath"
    if str(binary) != plan["executable"] or sha256(binary) != plan["heatbath_sha256"]:
        raise ValueError("Native A100 executable changed")
    for name, digest in plan["evidence_sha256"].items():
        if Path(name).name != name or sha256(RECORDS / name) != digest:
            raise ValueError("Native build, validation or timing evidence changed")
    commit = subprocess.check_output(["git", "-C", str(SOURCE), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(SOURCE), "status", "--porcelain",
                                     "--ignore-submodules=none"], text=True).strip()
    if dirty or commit != plan["klft_commit"]:
        raise ValueError("Source revision changed")
    validation = (RECORDS / "open_slab_a100_validation.txt").read_text().splitlines()
    if not {f"klft_commit {commit}", f"heatbath_sha256 {plan['heatbath_sha256']}",
            "ctest passed", "cuda_memcheck passed"}.issubset(validation):
        raise ValueError("Native A100 correctness checks are incomplete")
    runs, gate = production_runs(plan, json.loads(source_path.read_text()),
                                 json.loads((RECORDS / "open_slab_a100_runs.json").read_text()))
    assigned = sorted((r for r in runs if r["worker"] == index), key=lambda r: r["Ns"])
    for run in assigned:
        check_unused_input(run)
    check_gpu_idle(gpu)
    if shutil.disk_usage(ROOT).free < plan["min_free_bytes"]:
        raise RuntimeError("Insufficient free storage for production")
    start = {"worker": index, "pid": os.getpid(), "parent_pid": os.getppid(),
             "node": socket.gethostname(), "gpu_uuid": gpu, "cpu": worker["cpu"],
             "started_at": utc_now(), "manifest_sha256": sha256(PLAN),
             "source_manifest_sha256": plan["source_manifest_sha256"],
             "klft_commit": commit, "heatbath_sha256": plan["heatbath_sha256"],
             "backend": plan["backend"], "evidence_sha256": plan["evidence_sha256"],
             "runner_sha256": sha256(Path(__file__)),
             "output_checker_sha256": sha256(Path(__file__).with_name("mg_nt_fixed_nt1_run.py")),
             "timing_gate": gate, "seeds": [r["seed"] for r in assigned]}
    write_json(RECORDS / f"nt1_a100_worker_{index}_start.json", start)
    completed = []
    try:
        for run in assigned:
            check_gpu_idle(gpu)
            directory = check_unused_input(run)
            if shutil.disk_usage(ROOT).free < plan["min_free_bytes"]:
                raise RuntimeError("Free storage dropped below the production reserve")
            record = dict(run, **{k: v for k, v in start.items() if k != "worker"})
            record.update(planned_heatbath_sha256=run["heatbath_sha256"],
                          started_at=utc_now(), production_authorized=True)
            write_json(directory / "execution_start.json", record)
            print(json.dumps({"status": "started", "seed": run["seed"],
                              "Ns": run["Ns"], "gpu_uuid": gpu}), flush=True)
            before = time.monotonic()
            with (directory / "run.log").open("x") as log:
                subprocess.run([str(binary), "-f", "input.yaml"], cwd=directory,
                               stdout=log, stderr=subprocess.STDOUT, check=True, timeout=7200)
            elapsed = time.monotonic() - before
            checks = output_check(directory, run["Ns"], run["sweeps"])
            outputs = {name: sha256(directory / name) for name in (
                "holonomy_correlator.out", "holonomy_summary.out", "plaquette_profile.out",
                "configuration_final.klft")}
            result = dict(record, status="complete", completed_at=utc_now(),
                          elapsed_seconds=elapsed, checks=checks, output_sha256=outputs)
            write_json(directory / "execution_result.json", result)
            completed.append(run["seed"])
            print(json.dumps({"status": "complete", "seed": run["seed"],
                              "elapsed_seconds": elapsed, "checks": checks}), flush=True)
        write_json(RECORDS / f"nt1_a100_worker_{index}_result.json",
                   dict(start, status="complete", completed_at=utc_now(), completed_seeds=completed))
    except Exception as error:
        write_json(RECORDS / f"nt1_a100_worker_{index}_result.json",
                   dict(start, status="failed", ended_at=utc_now(), completed_seeds=completed,
                        error=f"{type(error).__name__}: {error}"))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("worker", type=int, choices=range(4))
    run_worker(parser.parse_args().worker)
