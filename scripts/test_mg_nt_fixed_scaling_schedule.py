"""Deterministic scheduler tests; no GPU, simulation or bootstrap is used."""
from contextlib import ExitStack
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch

import mg_nt_fixed_scaling_schedule as scheduler


class ScheduleTest(unittest.TestCase):
    def setUp(self) -> None:
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.data = self.root / "data"
        self.records = self.root / "records"
        self.data.mkdir()
        self.records.mkdir()
        (self.root / "project").mkdir()
        (self.root / "project/uv.lock").write_text("locked\n")
        (self.records / "runner.py").write_text("# pinned runner\n")
        (self.records / "scheduler.py").write_text("# pinned scheduler\n")
        self.manifest = dict(runs=[dict(seed=1)])
        (self.data / "scaling_manifest.json").write_text(json.dumps(self.manifest))
        plan = dict(source_sha256={"runner.py": scheduler.sha256(self.records / "runner.py")},
                    uv_lock_sha256=scheduler.sha256(self.root / "project/uv.lock"))
        (self.data / "scaling_execution.json").write_text(json.dumps(plan))
        self.config = dict(automatic_dispatch_authorized=True, node="lnode19", poll_seconds=30,
                           idle_confirmations=2, max_wait_seconds=48 * 3600, leave_free_gpu_index=6,
                           manifest_sha256=scheduler.sha256(self.data / "scaling_manifest.json"),
                           execution_plan_sha256=scheduler.sha256(self.data / "scaling_execution.json"),
                           source_sha256={"scheduler.py": scheduler.sha256(self.records / "scheduler.py")})
        (self.data / "scaling_schedule.json").write_text(json.dumps(self.config))
        stack = ExitStack()
        self.addCleanup(stack.close)
        for name, value in dict(ROOT=self.root, DATA=self.data, RECORDS=self.records,
                                STATUS=self.records / "scaling_schedule_status.json").items():
            stack.enter_context(patch.object(scheduler, name, value))
        stack.enter_context(patch.object(scheduler, "validate_manifest"))
        stack.enter_context(patch.object(scheduler, "check_unused_input"))

    def test_occupancy_and_reserved_gpu(self) -> None:
        with patch.object(scheduler.subprocess, "check_output", return_value=""):
            self.assertTrue(scheduler.gpu_idle(scheduler.GPUS[0]))
            with self.assertRaises(ValueError):
                scheduler.gpu_idle("GPU-c0d18349-2a36-8733-e53b-ae2ab1e14209")
        with patch.object(scheduler.subprocess, "check_output", return_value="1234\n5678\n"):
            self.assertFalse(scheduler.gpu_idle(scheduler.GPUS[0]))
        with patch.object(scheduler.subprocess, "check_output", return_value="Unknown Error"):
            with self.assertRaises(RuntimeError):
                scheduler.gpu_idle(scheduler.GPUS[0])
        with patch.object(scheduler.subprocess, "check_output", side_effect=subprocess.TimeoutExpired("query", 10)):
            with self.assertRaises(subprocess.TimeoutExpired):
                scheduler.gpu_idle(scheduler.GPUS[0])

    def test_contract_and_changed_hash(self) -> None:
        self.assertEqual(scheduler.validate_contract(self.config), self.manifest)
        (self.records / "runner.py").write_text("changed\n")
        with self.assertRaisesRegex(ValueError, "Execution source changed"):
            scheduler.validate_contract(self.config)

    def test_no_reuse_of_partial_stage(self) -> None:
        (self.records / "scaling_preflight_worker_0_start.json").write_text("{}")
        with self.assertRaises(FileExistsError):
            scheduler.validate_contract(self.config)

    def test_failed_timing_forbids_production(self) -> None:
        with patch.object(scheduler, "timing_projection", return_value={"passed": False}):
            with self.assertRaisesRegex(RuntimeError, "production remains forbidden"):
                scheduler.approve_timing(self.manifest, "hash")
        gate = json.loads((self.records / "scaling_timing_gate.json").read_text())
        self.assertFalse(gate["passed"])

    def exercise(self, *, busy: bool = False, gate_fails: bool = False,
                 preflight_code: int = 0) -> tuple[list, dict]:
        """Run the full state machine with a deterministic clock and fake children."""
        seconds = [0.]
        launches = []

        def launch(stage: str, worker: int) -> Mock:
            launches.append((stage, worker, seconds[0]))
            process = Mock(pid=100 + len(launches))
            if stage == "preflight":
                process.poll.return_value = preflight_code
            else:
                process.poll.side_effect = [None, None, 0]
            record = dict(status="complete", manifest_sha256=self.config["manifest_sha256"])
            (self.records / f"scaling_{stage}_worker_{worker}_result.json").write_text(json.dumps(record))
            return process

        def advance(interval: float) -> None:
            seconds[0] += self.config["max_wait_seconds"] if busy else interval

        with ExitStack() as stack:
            stack.enter_context(patch.object(scheduler.socket, "gethostname", return_value="lnode19"))
            stack.enter_context(patch.dict(scheduler.os.environ, {"SLURM_JOB_ID": ""}))
            stack.enter_context(patch.object(scheduler.time, "monotonic", side_effect=lambda: seconds[0]))
            stack.enter_context(patch.object(scheduler.time, "sleep", side_effect=advance))
            stack.enter_context(patch.object(scheduler, "gpu_idle", return_value=not busy))
            stack.enter_context(patch.object(scheduler, "launch", side_effect=launch))
            stack.enter_context(patch.object(scheduler, "worker_evidence", return_value=[{"checked": True}]))
            stack.enter_context(patch.object(scheduler, "timing_projection", return_value={"passed": not gate_fails}))
            stack.enter_context(patch.object(scheduler, "stop_owned"))
            if gate_fails or preflight_code:
                with self.assertRaises(RuntimeError):
                    scheduler.schedule()
            else:
                scheduler.schedule()
        return launches, json.loads(scheduler.STATUS.read_text())

    def test_preflight_then_three_workers_and_one_minute_inspection(self) -> None:
        launches, state = self.exercise()
        self.assertEqual([(s, w) for s, w, t in launches],
                         [("preflight", 0), ("production", 0), ("production", 1), ("production", 2)])
        self.assertEqual(launches[0][2], 30.)
        self.assertGreater(launches[1][2], launches[0][2])
        self.assertEqual(state["status"], "complete")
        self.assertEqual(state["leave_free_gpu_index"], 6)
        for worker in state["workers"].values():
            self.assertEqual(worker["status"], "complete")
            self.assertGreaterEqual(worker["inspection"]["elapsed_seconds"], 60.)
        self.assertEqual(state["bootstrap_operations"], 0)

    def test_busy_gpus_expire_without_launch(self) -> None:
        launches, state = self.exercise(busy=True)
        self.assertEqual(launches, [])
        self.assertEqual(state["status"], "expired_waiting_for_resources")

    def test_no_worker_after_failed_gate(self) -> None:
        launches, state = self.exercise(gate_fails=True)
        self.assertEqual(len(launches), 1)
        self.assertEqual(state["status"], "stopped")

    def test_no_retry_after_failed_preflight(self) -> None:
        launches, state = self.exercise(preflight_code=1)
        self.assertEqual(len(launches), 1)
        self.assertEqual(state["status"], "stopped")

    def test_cancellation_only_signals_owned_children(self) -> None:
        active, completed = Mock(pid=123), Mock(pid=456)
        active.poll.return_value, completed.poll.return_value = None, 0
        with patch.object(scheduler.os, "killpg") as kill:
            scheduler.stop_owned({("production", 0): active, ("production", 1): completed})
        kill.assert_called_once_with(123, scheduler.signal.SIGTERM)


if __name__ == "__main__":
    unittest.main()
