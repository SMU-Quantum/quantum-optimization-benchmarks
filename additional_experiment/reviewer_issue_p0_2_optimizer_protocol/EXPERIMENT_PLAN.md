# P0.2 Experiment Plan

## Part A: Protocol Manifest

Status: complete.

Files:

- `optimizer_protocol_manifest.csv`
- `optimizer_budget_summary.csv`
- `protocol_audit_digest.md`

This answers the artifact-level questions about optimizers, budgets, shots,
initialization, final-parameter selection, post-processing, termination, and
replayability.

## Part B: Seed Sensitivity

Status: reduced run complete.

Completed reduced cases:

| ID | Problem | Instance | Method | Seeds |
|---|---|---|---|---|
| R1 | MDKP | hp1 | VQE | 1103, 4409, 7703 |
| R2 | MIS | 1tc.32 | WS-QAOA | 1103, 4409, 7703 |
| R3 | MIS | 1tc.16 | PCE | 1103, 4409, 7703 |

Output:

- `reduced_part_b_20260623T0416Z/`

The run records one final production-sampling result per initialization. It
does not perform independent final-sampling repetitions.

## PCE/SLSQP Follow-up

Status: reduced diagnostic complete.

This was added after identifying that the reviewer concern was specifically
about PCE/SLSQP budget comparability.

| ID | Problem | Instance | Method | Current feasibility |
|---|---|---|---|---|
| S1 | MIS | 1tc.8 | PCE/SLSQP | complete |
| S2 | MIS | 1tc.16 | PCE/SLSQP | complete |
| S3 | QAP | nug12 | PCE/SLSQP | complete |

Output:

- `pce_slsqp_followup_20260623T0515Z/`

The historical PCE artifact audit found 33 saved PCE artifacts. The saved
benchmark artifacts record max optimizer iterations of 200 and objective/circuit
evaluations of 124, 184, or 200, depending on whether the run stopped early or
hit the budget. The benchmark PCE source path uses SciPy COBYLA. SLSQP appears
in exploratory PCE notebooks and QRAO support, not in the reported PCE benchmark
runner.

## Part C: Budget Saturation

Status: not run, by decision.

For PCE/SLSQP, the reduced follow-up already includes maxiter 100 and 200. The
results show that increasing SLSQP maxiter doubles objective evaluations for
cap-limited cases, but did not materially improve decoded quality in the tested
current-PCE MIS cases and did not recover feasible QAP solutions.

## Expected Cost

Part A: seconds.

Reduced Part B: about 19 minutes.

PCE/SLSQP follow-up: about 13 minutes, dominated by the QAP SLSQP finite-
difference runs.
