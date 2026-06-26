# Figure 5 Fidelity/Quality Digest

Source data are the feasible MDKP and MIS `feasible_gap` rows from `reviewer_issue_p2_2_figure4_feasibility/figure4_complexity_feasibility_data.csv`; no hardware run is added, removed, or recalculated relative to that plottable finite-gap set.

## 1. Points Per Panel

- noise_dominated: total=37, MDKP=36, MIS=1.
- signal_preserving: total=65, MDKP=48, MIS=17.

## 2. F_est Range Per Panel

- noise_dominated: min=8.74532852302e-87, max=9.23876811634e-16.
- signal_preserving: min=0.0250587410575, max=0.988053892081.

## 3. Feasible Points Per Method And Panel

- noise_dominated: VQE=0, CVaR-VQE=0, QAOA=12, MA-QAOA=12, WS-QAOA=13, PCE=0, QRAO=0.
- signal_preserving: VQE=14, CVaR-VQE=15, QAOA=3, MA-QAOA=2, WS-QAOA=3, PCE=14, QRAO=14.

## 4. Exclusions

MSP is excluded because its reported hardware metric is TDev, not percentage optimality gap. QAP is excluded because no reported hardware QAP run is feasible, so it has no finite percentage-gap outcome for this plot.

## 5. Revision Scope

No new run was performed. The plotted finite-gap MDKP/MIS points are unchanged; the revision swaps panel order, fixes the x-axis label, uses logarithmic fidelity axes, moves the legend outside the data area, and updates annotations/caption language.
