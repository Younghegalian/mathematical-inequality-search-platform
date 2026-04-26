from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from engine.recording import RunRecorder
from engine.runner import SearchRunner
from ns.targets import OMEGA_L3


class RecordingTest(unittest.TestCase):
    def test_search_writes_transition_trace_and_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            recorder = RunRecorder.create(OMEGA_L3.name, Path(temp_dir))
            result = SearchRunner(depth=2, width=20, recorder=recorder).run(OMEGA_L3)

            self.assertIsNotNone(result.run_path)
            self.assertIsNotNone(result.summary_path)
            self.assertTrue(result.run_path.exists())
            self.assertTrue(result.summary_path.exists())

            rows = [
                json.loads(line)
                for line in result.run_path.read_text(encoding="utf-8").splitlines()
            ]
            events = [row["event"] for row in rows]

            self.assertIn("run_start", events)
            self.assertIn("state_initial", events)
            self.assertIn("transition", events)
            self.assertIn("beam_selected", events)
            self.assertIn("run_complete", events)

            kept = [row for row in rows if row["event"] == "transition" and row["status"] == "kept"]
            self.assertEqual([row["action"] for row in kept], ["Gagliardo-Nirenberg", "Young"])

            summary = json.loads(result.summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["best_state"]["closure"]["status"], "BAD")
            self.assertEqual(summary["best_state"]["closure"]["ode_power"], "3")


if __name__ == "__main__":
    unittest.main()
