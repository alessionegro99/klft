#!/usr/bin/env python3
"""Wait for the three authorized A100s and execute the existing guarded stages.

The user's automatic-dispatch request supersedes manual stage submission only.
Smoke, timing, output, hash, storage and overwrite checks remain mandatory.
This is a one-shot scheduler: failed or partial stages require manual review.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
import fcntl
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from mg_nt_fixed_nt1_run import sha256, write_json
from mg_nt_fixed_nt1_a100 import check_unused_input, utc_now
from mg_nt_fixed_scaling_run import DATA, GPUS, RECORDS, ROOT, timing_projection, validate_manifest


STATUS = RECORDS / "scaling_schedule_status.json"
LAUNCHER = RECORDS / "run_scaling_a100.sh"


def gpu_idle(gpu: str) -> bool:
    """Require a successful occupancy query; zero utilization alone is insufficient."""
    if gpu not in GPUS:
        raise ValueError("Only GPUs 3,4,5 are authorized")
    text = subprocess.check_output(
        ["nvidia-smi", "-i", gpu, "--query-compute-apps=pid", "--format=csv,noheader,nounits"],
        text=True, timeout=10).strip()
    if any(not line.strip().isdigit() for line in text.splitlines()):
        raise RuntimeError(f"Unrecognized GPU occupancy response: {text}")
    return not text


def validate_contract(config: dict) -> dict:
    """Verify the pinned scheduler, execution plan and unused immutable input set."""
    if (config["automatic_dispatch_authorized"] is not True or config["node"] != "lnode19"
            or config["poll_seconds"] != 30 or config["idle_confirmations"] != 2
            or config["max_wait_seconds"] != 48 * 3600 or config["leave_free_gpu_index"] != 6):
        raise ValueError("Automatic dispatch scope changed")
    if sha256(DATA / "scaling_execution.json") != config["execution_plan_sha256"]:
        raise ValueError("Execution plan changed")
    if sha256(DATA / "scaling_manifest.json") != config["manifest_sha256"]:
        raise ValueError("Input manifest changed")
    for name, digest in config["source_sha256"].items():
        if Path(name).name != name or sha256(RECORDS / name) != digest:
            raise ValueError(f"Scheduler source changed: {name}")
    plan = json.loads((DATA / "scaling_execution.json").read_text())
    for name, digest in plan["source_sha256"].items():
        if Path(name).name != name or sha256(RECORDS / name) != digest:
            raise ValueError(f"Execution source changed: {name}")
    if sha256(ROOT / "project/uv.lock") != plan["uv_lock_sha256"]:
        raise ValueError("Locked environment changed")
    manifest = json.loads((DATA / "scaling_manifest.json").read_text())
    validate_manifest(manifest)
    for run in manifest["runs"]:
        check_unused_input(run)
    for stage, worker in [("preflight", 0), *(('production', i) for i in range(3))]:
        for suffix in ("start", "result"):
            path = RECORDS / f"scaling_{stage}_worker_{worker}_{suffix}.json"
            if path.exists():
                raise FileExistsError(f"Review existing stage before rescheduling: {path}")
    if (RECORDS / "scaling_timing_gate.json").exists():
        raise FileExistsError("Review the existing timing gate before rescheduling")
    return manifest


def save_status(state: dict) -> None:
    """Replace only this scheduler's status atomically; emit state transitions."""
    state["updated_at"] = utc_now()
    temporary = STATUS.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2, allow_nan=False) + "\n")
    temporary.replace(STATUS)


def launch(stage: str, worker: int) -> subprocess.Popen:
    """Start the unchanged two-hour-bounded launcher with an exclusive new log."""
    if stage not in ("preflight", "production") or worker not in range(3):
        raise ValueError("Invalid execution stage")
    log = ROOT / "logs" / f"dir-scaling-scheduled-{stage}-{worker}.out"
    with log.open("x") as stream:
        return subprocess.Popen(["bash", str(LAUNCHER), stage, str(worker)],
                                stdin=subprocess.DEVNULL, stdout=stream,
                                stderr=subprocess.STDOUT, start_new_session=True)


def approve_timing(manifest: dict, manifest_hash: str) -> dict:
    """Apply the unchanged numerical gates under explicit automatic authorization."""
    gate = timing_projection(manifest, manifest_hash)
    gate["approval"] = "automatic dispatch explicitly requested by user; unchanged cost thresholds"
    write_json(RECORDS / "scaling_timing_gate.json", gate)
    if not gate["passed"]:
        raise RuntimeError("Smoke/timing gate failed; production remains forbidden")
    return gate


def worker_evidence(manifest: dict, worker: int) -> list[dict]:
    """Inspect compact execution records and first observable rows on the compute node."""
    production = [r for r in manifest["runs"] if r["purpose"] == "production"]
    evidence = []
    for i, run in enumerate(production):
        if i % 3 != worker:
            continue
        directory = Path(run["raw_path"])
        result_path = directory / "execution_result.json"
        item = dict(seed=run["seed"], started=(directory / "execution_start.json").exists())
        if result_path.exists():
            result = json.loads(result_path.read_text())
            if result["status"] != "complete" or result["input_sha256"] != run["input_sha256"]:
                raise ValueError("Production result identity failed")
            item.update(status="complete", checks=result["checks"])
        else:
            item["status"] = "started" if item["started"] else "waiting"
        correlator = directory / "holonomy_correlator.out"
        if correlator.exists():
            item["correlator_bytes"] = correlator.stat().st_size
            with correlator.open() as stream:
                first = next((line.split() for line in stream if line.strip() and not line.startswith("#")), None)
            if first is not None:
                if len(first) != 5 or int(first[0]) != 1 or int(first[1]) != 0 or float(first[2]) != 1.:
                    raise ValueError("First production correlator row is invalid")
                item["first_correlator_row"] = first
        evidence.append(item)
    return evidence


def check_stage_exit(stage: str, worker: int, code: int, manifest_hash: str) -> None:
    """Require both a successful process and its manifest-bound completion record."""
    if code != 0:
        raise RuntimeError(f"{stage} worker {worker} exited {code}; no automatic retry")
    result = json.loads((RECORDS / f"scaling_{stage}_worker_{worker}_result.json").read_text())
    if result["status"] != "complete" or result["manifest_sha256"] != manifest_hash:
        raise ValueError("Stage completion record is missing or stale")


def stop_owned(children: dict) -> None:
    """On scheduler failure/cancellation, signal only its own still-running groups."""
    for process in children.values():
        if process.poll() is None:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass


def schedule() -> None:
    """Poll for up to 48 hours, run gates, dispatch each free worker, then exit."""
    if socket.gethostname() != "lnode19" or os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("The scheduler must run directly on authorized lnode19")
    config_path = DATA / "scaling_schedule.json"
    config = json.loads(config_path.read_text())
    with (RECORDS / "scaling_schedule.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest = validate_contract(config)
        deadline = time.monotonic() + config["max_wait_seconds"]
        expires = datetime.now(timezone.utc) + timedelta(seconds=config["max_wait_seconds"])
        state = dict(status="waiting_for_preflight_gpu", pid=os.getpid(), node="lnode19",
                     started_at=utc_now(), expires_at=expires.isoformat(),
                     manifest_sha256=config["manifest_sha256"], config_sha256=sha256(config_path),
                     gpu_indices=[3, 4, 5], leave_free_gpu_index=6, workers={},
                     poll_seconds=30, idle_confirmations=2, bootstrap_operations=0)
        write_json(RECORDS / "scaling_schedule_start.json", state)
        save_status(state)
        print(json.dumps(state), flush=True)
        children, started, observed = {}, {}, set()
        idle_counts = [0, 0, 0]
        preflight_done = False
        try:
            while True:
                now = time.monotonic()
                for (stage, worker), process in list(children.items()):
                    code = process.poll()
                    if stage == "production" and worker not in observed and (now - started[worker] >= 60 or code is not None):
                        state["workers"][str(worker)]["inspection"] = dict(
                            elapsed_seconds=now-started[worker],
                            evidence=worker_evidence(manifest, worker))
                        observed.add(worker)
                    if code is None:
                        continue
                    check_stage_exit(stage, worker, code, config["manifest_sha256"])
                    if stage == "preflight":
                        state["timing_gate"] = approve_timing(manifest, config["manifest_sha256"])
                        preflight_done = True
                        idle_counts = [0, 0, 0]
                        state["status"] = "waiting_for_production_gpus"
                    else:
                        state["workers"][str(worker)].update(status="complete", completed_at=utc_now(),
                            evidence=worker_evidence(manifest, worker))
                    del children[(stage, worker)]
                if preflight_done and len(state["workers"]) == 3 and not children:
                    state["status"] = "complete"
                    break
                # Expiration forbids new starts; bounded in-flight jobs may finish.
                if now >= deadline:
                    state["status"] = "expired_waiting_for_resources"
                    if not children:
                        break
                else:
                    candidates = range(3) if preflight_done else (0,)
                    for worker in candidates:
                        if worker in started or ("preflight", 0) in children:
                            continue
                        idle_counts[worker] = idle_counts[worker] + 1 if gpu_idle(GPUS[worker]) else 0
                        if idle_counts[worker] < config["idle_confirmations"]:
                            continue
                        stage = "production" if preflight_done else "preflight"
                        process = launch(stage, worker)
                        children[(stage, worker)] = process
                        if stage == "production":
                            started[worker] = now
                            state["workers"][str(worker)] = dict(status="running", pid=process.pid,
                                                                   gpu_index=3+worker, started_at=utc_now())
                            state["status"] = "production_running"
                        else:
                            state.update(status="preflight_running", preflight_pid=process.pid)
                        print(json.dumps(dict(event="launched", stage=stage, worker=worker,
                                              pid=process.pid, at=utc_now())), flush=True)
                state["idle_confirmations_seen"] = idle_counts.copy()
                save_status(state)
                time.sleep(config["poll_seconds"])
            state["ended_at"] = utc_now()
            save_status(state)
            print(json.dumps(dict(status=state["status"], ended_at=state["ended_at"])), flush=True)
        except BaseException as error:
            stop_owned(children)
            state.update(status="stopped", ended_at=utc_now(), error=f"{type(error).__name__}: {error}")
            save_status(state)
            raise


if __name__ == "__main__":
    def cancelled(signum: int, frame: object) -> None:
        raise SystemExit(f"Scheduler cancelled by signal {signum}")

    signal.signal(signal.SIGTERM, cancelled)
    schedule()
