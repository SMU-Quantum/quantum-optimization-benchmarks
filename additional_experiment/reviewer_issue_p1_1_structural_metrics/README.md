# P1.1 Structural Metrics

This experiment builds problem- and instance-level structural metrics used to
explain why QAP, MIS, MDKP, and MSP behave differently in the benchmark.

## What It Does

- Computes instance fingerprints such as QUBO size, density, degree statistics,
  constraint counts, and structural identifiers.
- Summarizes problem-level differences in `problem_structural_summary.csv`.
- Adds QAP-specific feasibility geometry in `qap_feasibility_geometry.csv`.
- Provides manuscript-ready interpretation in `structural_metrics_digest.md`.

## Main Files

- `build_structural_metrics.py`: regenerates the audit outputs.
- `instance_structural_fingerprints.csv`: per-instance structural metrics.
- `problem_structural_summary.csv`: problem-level summary table.
- `qap_feasibility_geometry.csv`: QAP permutation-feasibility geometry.
- `structural_metrics_digest.md`: concise reviewer-response digest.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p1_1_structural_metrics/build_structural_metrics.py
```
