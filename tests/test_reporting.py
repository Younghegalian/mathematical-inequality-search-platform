from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from engine.batch import run_batch
from engine.reporting import generate_reports


class ReportingTest(unittest.TestCase):
    def test_generate_reports_writes_index_and_run_pages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            out_dir = Path(temp_dir) / "reports"
            run_batch(
                budget=2,
                targets=["omega_L3", "omega_L6_squared"],
                depths=[2],
                widths=[20],
                data_dir=data_dir,
            )

            result = generate_reports(data_dir, out_dir)

            index_path = Path(result["index_path"])
            run_pages = list((out_dir / "runs").glob("*.html"))
            self.assertTrue(index_path.exists())
            self.assertEqual(result["run_report_count"], 2)
            self.assertEqual(len(run_pages), 2)

            index_html = index_path.read_text(encoding="utf-8")
            self.assertIn("Candidate-Focused Search Report", index_html)
            self.assertIn("omega_L3", index_html)
            self.assertIn("omega_L6_squared", index_html)
            self.assertIn("Massive Search Funnel", index_html)
            self.assertIn("Survivor Queue Details", index_html)
            self.assertIn("Symbolic replay", index_html)
            self.assertIn("Target relevance", index_html)
            self.assertIn("Raw Diagnostics", index_html)
            self.assertIn("<svg", index_html)

            run_html = run_pages[0].read_text(encoding="utf-8")
            self.assertIn("Run Detail", run_html)
            self.assertIn("Best Proof Path", run_html)
            self.assertIn("Candidate Landscape", run_html)


if __name__ == "__main__":
    unittest.main()
