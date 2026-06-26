# P2.2 Revised Figure 4

This folder contains the revised Figure 4 artifacts for circuit complexity,
hardware quality, and feasibility encoding.

## What It Does

- Rebuilds Figure 4 from archived hardware outputs.
- Separates feasible quality points from infeasible hardware outcomes.
- Adds explicit infeasibility encoding and fidelity reference thresholds.
- Produces the exact CSV behind the revised plot.

## Main Files

- `run_figure4_revision.py`: regenerates the figure and CSV outputs.
- `fig_circuit_complexity_vs_quality_revised.pdf`: revised figure.
- `fig_circuit_complexity_vs_quality_revised.png`: raster preview.
- `figure4_complexity_feasibility_data.csv`: point-level data.
- `figure4_complexity_feasibility_summary.csv`: method/problem summary.
- `figure4_complexity_feasibility_digest.md`: reviewer-response digest.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p2_2_figure4_feasibility/run_figure4_revision.py
```
