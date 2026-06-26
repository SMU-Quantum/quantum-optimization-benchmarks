# Issue P0.2: Variational Optimization Protocol

This folder isolates reviewer concern P0.2: optimization protocol details,
budget fairness, convergence/termination evidence, and initialization
sensitivity.

## Immediate Evidence Generated Here

Part A is an artifact audit and does not rerun quantum experiments:

```bash
.venv/bin/python additional_experiment/reviewer_issue_p0_2_optimizer_protocol/build_optimizer_protocol_manifest.py
```

It generates:

- `optimizer_protocol_manifest.csv`
- `optimizer_budget_summary.csv`
- `protocol_audit_digest.md`

The manifest is intentionally conservative. Fields that are not provable from
the saved artifacts are set to `unknown_legacy_artifact` or `not_applicable`.

## Important Early Finding

The current artifacts indicate that PCE runs used `maxiter=200`, and most PCE
artifacts terminate with "Maximum number of function evaluations has been
exceeded." This conflicts with manuscript wording that describes a 100-iteration
SLSQP cap. The paper should not be edited until this audit is reviewed.

## Completed Follow-up Runs

Reduced Part B was run for three representative cases:

- MDKP hp1 VQE, three initialization seeds.
- MIS 1tc.32 WS-QAOA, three initialization seeds.
- MIS 1tc.16 current PCE, three initialization seeds.

The production output is in:

- `reduced_part_b_20260623T0416Z/`

A focused PCE/SLSQP follow-up was then run because the reviewer concern was
specifically about SLSQP and PCE budget comparability. This follow-up has two
parts:

- a historical PCE artifact audit over saved hardware PCE `result.json` files;
- a reduced current-PCE SLSQP diagnostic on MIS 1tc.8, MIS 1tc.16, and QAP
  nug12 using three initialization seeds and maxiter values 100 and 200.

The PCE/SLSQP output is in:

- `pce_slsqp_followup_20260623T0515Z/`

The key result is that SLSQP iterations are not comparable to COBYLA objective
evaluations. In the reduced diagnostic, 100 SLSQP iterations corresponded to
median objective-evaluation counts of 616, 2590, and 7690 for the three tested
cases. At maxiter 200, the corresponding medians were 616, 5105, and 15317.

## Remaining Limitation

The historical 182-parameter Brickwork PCE MDKP run cannot be exactly replayed
from the saved artifacts because no serialized circuit or parameter-to-gate
mapping is saved. The defensible manuscript response is therefore to report the
recovered fixed-budget metadata, state the replay limitation, and avoid claiming
SLSQP-converged historical PCE performance.
