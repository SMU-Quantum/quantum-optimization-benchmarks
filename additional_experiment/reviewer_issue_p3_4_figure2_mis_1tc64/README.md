# P3.4 Revised Figure 2 MIS 1tc.64

This folder resolves the reviewer concern that MIS instance `1tc.64` was absent
from Figure 2 despite having at least one feasible hardware output.

## What It Does

- Rebuilds the MIS hardware feasibility plot from archived MIS hardware results.
- Includes `1tc.64` in panel (a).
- Shows WS-QAOA as the only feasible `1tc.64` method, with a 35% gap to BKS.
- Marks the other six `1tc.64` methods as infeasible so panel (a) and panel (b)
  are consistent.

## Main Files

- `run_figure2_mis_revision.py`: regenerates the figure and source CSV.
- `fig2_mis_hardware_feasibility_revised.pdf`: revised Figure 2.
- `fig2_mis_hardware_feasibility_revised.png`: raster preview.
- `figure2_mis_hardware_feasibility_data.csv`: exact source data.
- `figure2_mis_1tc64_digest.md`: reviewer-response digest and caption text.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p3_4_figure2_mis_1tc64/run_figure2_mis_revision.py
```
