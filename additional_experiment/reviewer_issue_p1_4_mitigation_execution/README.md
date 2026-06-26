# P1.4 Mitigation Execution Audit

This audit checks which post-processing, mitigation, and repair steps are
actually present in the archived hardware artifacts.

## What It Does

- Reads hardware `result.json` artifacts and extracts execution/repair metadata.
- Separates executed procedures from proposed or descriptive procedures.
- Produces language for the manuscript that avoids overstating mitigation.

## Main Files

- `run_mitigation_execution_audit.py`: regenerates the audit table and digest.
- `mitigation_execution_provenance.csv`: artifact-level provenance table.
- `mitigation_execution_digest.md`: evidence summary.
- `manuscript_mitigation_execution_insert.md`: suggested manuscript insertion.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p1_4_mitigation_execution/run_mitigation_execution_audit.py
```
