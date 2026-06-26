# P1.3 QUBO Penalty Audit

This experiment documents how QUBO penalty terms are chosen and tests whether
reported conclusions are sensitive to those penalties.

## What It Does

- Reconstructs penalty provenance for MDKP, MIS, QAP, and MSP conversions.
- Records converter-derived penalty formulas and source locations.
- Runs reduced sensitivity checks over penalty multipliers.
- Produces a reviewer-facing explanation of why the chosen penalties are
  defensible.

## Main Files

- `run_penalty_audit.py`: regenerates the penalty audit.
- `penalty_provenance_ledger.csv`: penalty source and construction ledger.
- `penalty_sensitivity_replicates.csv`: replicate-level sensitivity data.
- `penalty_sensitivity_summary.csv`: compact sensitivity summary.
- `penalty_audit_digest.md`: manuscript-ready digest.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p1_3_penalty_audit/run_penalty_audit.py
```
