# P0.2 Reviewer Response Notes

## Evidence Generated

Part A was completed as an audit of the existing saved benchmark artifacts. The audit produced:

- `optimizer_protocol_manifest.csv`: one row per recovered run artifact.
- `optimizer_budget_summary.csv`: grouped budget and stopping-summary table.
- `protocol_audit_digest.md`: short digest of the main protocol issues.

Reduced Part B was completed on three representative cases with three initialization seeds each:

- `R1_mdkp_hp1_vqe`: MDKP hp1 direct VQE, 60 logical qubits.
- `R2_mis_1tc32_ws_qaoa`: MIS 1tc.32 warm-start QAOA, 32 qubits.
- `R3_mis_1tc16_pce`: MIS 1tc.16 current PCE encoding, 16 logical variables encoded onto 4 qubits.

Part C was not run. The relevant Part C point should instead be addressed using the realized objective-evaluation counts and trace behavior from reduced Part B.

A focused PCE/SLSQP follow-up was also completed:

- `reported_pce_artifact_audit.csv`: historical saved PCE artifact audit.
- `pce_slsqp_diagnostic_runs.csv`: reduced current-PCE SLSQP run-level data.
- `pce_slsqp_diagnostic_summary.csv`: aggregate SLSQP budget/quality table.
- `pce_slsqp_followup_digest.md`: consolidated digest and LaTeX table.

## Main Findings

The artifact audit found 262 saved result artifacts. All audited runs used 1000 objective shots and 1000 final-sampling shots where the fields were recoverable. Planned optimizer budgets were consistently 200 objective evaluations/iterations in the saved artifacts, including PCE artifacts. This means the manuscript should not describe PCE as using a 100-iteration cap unless that claim is tied to a different, separately documented run set.

Termination behavior was mixed: 149 artifacts reached the budget, while 113 ended with optimizer success. No saved result artifacts contained optimizer tolerance values, so the revision should avoid claiming tolerance-based convergence unless new tolerance metadata is added.

Reduced Part B showed meaningful initialization sensitivity in decoded quality for the two larger cases:

| case | seeds | termination | decoded gap / quality values |
|---|---:|---|---|
| MDKP hp1 VQE | 3 | all budget reached at 200 evaluations | 24.40%, 20.07%, 24.63% final decoded gaps |
| MIS 1tc.32 WS-QAOA | 3 | optimizer success at 63-75 evaluations | 8.33%, 16.67%, 50.00% final decoded gaps |
| MIS 1tc.16 PCE | 3 | all budget reached at 200 evaluations | 62.50%, 62.50%, 62.50% final decoded gaps |

This supports a cautious manuscript change: initialization and optimizer stopping can materially affect measured performance, especially for nontrivial VQE/QAOA cases. Therefore the revised text should state the fixed-budget optimizer protocol, report realized function evaluations, and treat single-seed results as representative runs rather than as full optimizer-distribution estimates.

The PCE/SLSQP follow-up adds a second important point. The historical saved PCE
artifacts do not support the statement that the reported PCE benchmark results
were produced by SLSQP with a 100-iteration cap. Across 33 saved hardware PCE
artifacts, the recovered budget fields show max optimizer iterations of 200 and
objective/circuit evaluations of 124, 184, or 200. The benchmark PCE source path
uses SciPy COBYLA; SLSQP appears in exploratory PCE notebooks and QRAO support.

The reduced current-PCE SLSQP diagnostic shows why iteration-count comparisons
are unsafe. With three seeds:

| case | maxiter | median nfev | success rate | cap-hit rate | median decoded gap |
|---|---:|---:|---:|---:|---:|
| MIS 1tc.8 PCE/SLSQP | 100 | 616 | 1.00 | 0.00 | 25.0% |
| MIS 1tc.8 PCE/SLSQP | 200 | 616 | 1.00 | 0.00 | 25.0% |
| MIS 1tc.16 PCE/SLSQP | 100 | 2590 | 0.00 | 1.00 | 62.5% |
| MIS 1tc.16 PCE/SLSQP | 200 | 5105 | 0.00 | 1.00 | 62.5% |
| QAP nug12 PCE/SLSQP | 100 | 7690 | 0.00 | 1.00 | infeasible |
| QAP nug12 PCE/SLSQP | 200 | 15317 | 0.00 | 1.00 | infeasible |

Thus, the revision should report objective evaluations rather than treating
SLSQP iterations and COBYLA iterations as equivalent budget units.

## Suggested Manuscript Text

For the methods section:

> All reported benchmark artifacts were audited for optimizer budget and stopping metadata. For the recovered PCE hardware artifacts, the saved files record a fixed budget of 200 optimizer iterations/objective evaluations and 1000 shots per objective evaluation, followed by 1000 shots for final sampling. We report realized objective evaluations, optimizer status, decoded objective value, feasibility, and post-processing status for each run. Because the saved artifacts do not contain tolerance metadata or serialized historical PCE circuits, we do not claim tolerance-based convergence or exact replay of the historical Brickwork PCE runs.

For limitations:

> The reported variational results are based on fixed-budget optimizer runs and should not be interpreted as exhaustive convergence studies. A reduced seed-sensitivity check on representative MDKP, MIS warm-start QAOA, and PCE cases showed that initialization can change the decoded final quality, especially on larger VQE/QAOA instances. A focused PCE/SLSQP diagnostic further showed that optimizer iterations are not comparable across optimizers because SLSQP can require many objective evaluations per iteration. We therefore report realized objective-evaluation counts and avoid optimizer-independent performance claims.

For reviewer response:

> We added a protocol audit, a reduced initialization-sensitivity experiment, and a focused PCE/SLSQP follow-up. The audit recovered 262 saved artifacts and confirmed planned objective-evaluation budgets, objective-shot counts, final-sampling-shot counts, and realized termination status. For PCE specifically, 33 saved hardware PCE artifacts record 200 as the recovered optimizer budget, while the benchmark source path uses SciPy COBYLA; SLSQP appears in exploratory notebooks and QRAO support rather than in the reported PCE runner. Because the historical PCE artifacts do not serialize the original circuit and parameter mapping, we do not claim exact SLSQP replay of the historical Brickwork PCE runs. The reduced SLSQP diagnostic on current PCE cases is included to show that SLSQP iterations can correspond to hundreds or thousands of objective evaluations. We revised the manuscript to report realized objective-evaluation counts and to soften claims that depend on optimizer-budget equivalence.

## Caveats

- This is not a full revalidation over all benchmark instances.
- Final-sampling repetitions were not independently rerun; each reduced Part B row uses the production final sampled counts from the benchmark runner.
- Legacy Brickwork PCE artifacts remain non-replayable from the current source because the saved artifacts do not serialize the original circuit.
- The PCE/SLSQP diagnostic uses the current reproducible PCE implementation; it is not an exact replay of the historical Brickwork PCE hardware artifacts.
