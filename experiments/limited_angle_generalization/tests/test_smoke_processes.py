"""Check that native worker failures cannot produce a success report."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import smoke_gpu


class SmokeProcessTest(unittest.TestCase):
    def test_workers_are_separate_and_results_combined(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"
            stages = []

            def worker(command):
                stage = command[command.index("--worker") + 1]
                stages.append(stage)
                Path(command[-1]).write_text(json.dumps({stage + "_passed": True}))
                return subprocess.CompletedProcess(command, 0)

            with patch.object(sys, "argv", ["smoke", "--execute", "--report", str(report)]), \
                    patch.object(smoke_gpu.subprocess, "run", side_effect=worker):
                smoke_gpu.main()
            self.assertEqual(stages, ["query", "projector"])
            result = json.loads(report.read_text())
            self.assertTrue(result["query_passed"] and result["projector_passed"])
            self.assertTrue(result["backend_process_isolation"])

    def test_crashed_worker_never_writes_success_even_with_partial_json(self):
        with tempfile.TemporaryDirectory() as directory:
            report = Path(directory) / "report.json"

            def crash(command):
                Path(command[-1]).write_text('{}')
                return subprocess.CompletedProcess(command, -11)

            with patch.object(sys, "argv", ["smoke", "--execute", "--report", str(report)]), \
                    patch.object(smoke_gpu.subprocess, "run", side_effect=crash) as run:
                with self.assertRaisesRegex(RuntimeError, "query worker failed.*-11"):
                    smoke_gpu.main()
            self.assertEqual(run.call_count, 1)
            self.assertFalse(report.exists())
