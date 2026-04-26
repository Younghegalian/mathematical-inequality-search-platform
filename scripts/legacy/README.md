Legacy command-line helpers.

These scripts are kept for ad hoc debugging and old workflows, but they are not
part of the main large-scale AISIS process.

Current main flow:

```bash
python3 scripts/run_random.py --budget 1000
python3 scripts/build_index.py
python3 scripts/report.py
python3 scripts/run_verification.py --data-dir data
```

Legacy helpers:

```bash
python3 scripts/legacy/run_batch.py --budget 8 --dry-run
python3 scripts/legacy/analyze.py data/runs/<run_id>.jsonl
```
