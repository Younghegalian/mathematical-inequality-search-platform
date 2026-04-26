from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.batch import build_jobs, run_batch


class BatchTest(unittest.TestCase):
    def test_build_jobs_respects_budget_and_cycles_combinations(self) -> None:
        jobs = build_jobs(
            budget=5,
            targets=["omega_L3", "omega_L6_squared"],
            depths=[2, 3],
            widths=[20],
        )

        self.assertEqual(len(jobs), 5)
        self.assertEqual([job.index for job in jobs], [1, 2, 3, 4, 5])
        self.assertEqual(jobs[0].target.name, "omega_L3")
        self.assertEqual(jobs[1].target.name, "omega_L3")
        self.assertEqual(jobs[2].target.name, "omega_L6_squared")
        self.assertEqual(jobs[3].target.name, "omega_L6_squared")
        self.assertEqual(jobs[4].target.name, "omega_L3")

    def test_run_batch_dry_run_does_not_write_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = run_batch(
                budget=3,
                targets=["omega_L3"],
                depths=[2],
                widths=[20],
                data_dir=Path(temp_dir),
                dry_run=True,
            )

            self.assertTrue(manifest["dry_run"])
            self.assertEqual(len(manifest["jobs"]), 3)
            self.assertEqual(manifest["runs"], [])
            self.assertFalse((Path(temp_dir) / "runs").exists())

    def test_run_batch_writes_manifest_and_runs(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            manifest = run_batch(
                budget=2,
                targets=["omega_L3"],
                depths=[2],
                widths=[20],
                data_dir=Path(temp_dir),
            )

            self.assertEqual(len(manifest["runs"]), 2)
            self.assertTrue(Path(manifest["manifest_path"]).exists())
            self.assertEqual(len(list((Path(temp_dir) / "runs").glob("*.jsonl"))), 2)
            self.assertEqual(len(list((Path(temp_dir) / "results").glob("*.json"))), 2)


if __name__ == "__main__":
    unittest.main()
