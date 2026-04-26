from __future__ import annotations

import sqlite3
import tempfile
import unittest
from pathlib import Path

from engine.batch import run_batch
from engine.indexing import build_sqlite_index, fetch_index_summaries
from engine.reporting import generate_reports


class IndexingTest(unittest.TestCase):
    def test_build_sqlite_index_and_report_from_it(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir) / "data"
            index_path = data_dir / "index" / "aisis.sqlite"
            out_dir = Path(temp_dir) / "reports"
            run_batch(
                budget=2,
                targets=["omega_L3", "omega_L6_squared"],
                depths=[2],
                widths=[20],
                data_dir=data_dir,
            )

            result = build_sqlite_index(data_dir, index_path)

            self.assertEqual(result.scanned, 2)
            self.assertEqual(result.inserted_or_updated, 2)
            self.assertTrue(index_path.exists())
            self.assertEqual(len(fetch_index_summaries(index_path)), 2)

            with sqlite3.connect(index_path) as connection:
                rows = connection.execute("select count(*) from runs").fetchone()[0]
            self.assertEqual(rows, 2)

            report = generate_reports(data_dir, out_dir, index_path=index_path)
            self.assertEqual(report["run_report_count"], 2)
            self.assertEqual(report["index_source"], str(index_path))
            self.assertTrue(Path(report["index_path"]).exists())


if __name__ == "__main__":
    unittest.main()
