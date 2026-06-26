# Additional Experiments

This folder collects reviewer-response experiments, audits, and regenerated
figures that were added after the main benchmark artifacts. Each subfolder is
scoped to one reviewer issue and contains its own README, source script where
applicable, and generated evidence files.

## Folder Index

- `reviewer_issue_p0_1_mps_validity`: MPS simulator validity checks and fixed-circuit MDKP validation evidence.
- `reviewer_issue_p0_2_optimizer_protocol`: Optimizer protocol audit, initialization-seed sensitivity runs, and PCE/SLSQP budget follow-up.
- `reviewer_issue_p0_4_random_baseline`: Matched uniform-random baseline for low-fidelity QAOA-family hardware artifacts.
- `reviewer_issue_p1_1_structural_metrics`: Problem-specific structural metrics for explaining QAP/MIS/MDKP/MSP difficulty.
- `reviewer_issue_p1_2_qaoa_counterfactuals`: QAOA compilation counterfactuals, including fractional gates and SWAP-aware compilation.
- `reviewer_issue_p1_3_penalty_audit`: QUBO penalty provenance and sensitivity checks.
- `reviewer_issue_p1_4_mitigation_execution`: Audit of whether mitigation and repair procedures were executed and how to describe them.
- `reviewer_issue_p2_2_figure4_feasibility`: Revised Figure 4 with explicit feasibility encoding and fidelity references.
- `reviewer_issue_p2_3_figure5_fidelity`: Revised Figure 5 with corrected fidelity-axis labeling and panel order.
- `reviewer_issue_p3_4_figure2_mis_1tc64`: Revised Figure 2 including the missing MIS `1tc.64` case.

Run scripts from the repository root unless a subfolder README says otherwise.
