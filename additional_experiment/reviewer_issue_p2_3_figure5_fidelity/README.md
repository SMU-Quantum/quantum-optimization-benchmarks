# P2.3 Revised Figure 5

This folder contains the revised Figure 5 artifacts for hardware solution
quality versus the gate-count fidelity proxy.

## What It Does

- Regenerates Figure 5 using the same finite-gap MDKP/MIS points.
- Swaps the panels so fidelity increases left to right.
- Corrects the x-axis label to use `F_est`.
- Keeps MSP and QAP excluded because their hardware metrics are not comparable
  finite percentage-gap outcomes.

## Main Files

- `run_figure5_revision.py`: regenerates the figure and CSV outputs.
- `fig_fidelity_vs_quality_revised.pdf`: revised figure.
- `fig_fidelity_vs_quality_revised.png`: raster preview.
- `figure5_fidelity_quality_data.csv`: exact plotted data.
- `figure5_fidelity_quality_digest.md`: reviewer-response digest.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p2_3_figure5_fidelity/run_figure5_revision.py
```
