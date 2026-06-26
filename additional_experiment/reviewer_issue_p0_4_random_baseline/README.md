# P0.4 Random Sampling Baseline

This folder contains the offline control experiment for the reviewer concern
about low-fidelity QAOA-family hardware results behaving like best-shot
selection from a noisy sampler.

## What was run

- Scope: low-fidelity QAOA, MA-QAOA, and WS-QAOA hardware artifacts only.
- Excluded by design: PCE and QRAO.
- Random seeds: `10001` through `10300`, giving 300 replicates per hardware run.
- Candidate pool: optimizer-trajectory counts only. The saved hardware path stores
  the best counts from the optimizer evaluation with the lowest expectation
  objective; it does not run an independent final sampling batch.
- Matched selection rule: for each random replicate, generate the same number of
  uniform bitstrings as the hardware run evaluated, grouped into the same
  optimizer-evaluation batches; select the batch with the lowest mean QUBO
  energy, then select the lowest-QUBO-energy bitstring from that batch.
- Shared post-processing: apply the same one-round local swap refinement used by
  the hardware artifact path.

The historical artifacts do not store per-gate calibration errors, so the
`fidelity_estimate` column is a reconstructed gate-count proxy used only to
select the low-fidelity subset. It should not be described as a measured process
fidelity.

QAP rows are recorded as feasibility-only appendix controls. All QAP hardware
rows in this subset are infeasible, and a valid uniform random permutation is
astronomically unlikely at the sampled candidate counts.

## Outputs

- `random_baseline_selection_ledger.csv`
- `random_baseline_replicates.csv`
- `random_baseline_summary.csv`
- `random_baseline_group_summary.csv`
- `plots/fig_low_fidelity_random_baseline.pdf`
- `random_baseline_digest.md`

The figure is intended for the appendix, not the main manuscript. The main
manuscript should use `random_baseline_group_summary.csv` as the compact
aggregate evidence table. In the figure, MDKP and MIS use optimality gap (%);
MSP uses TDev.

## Rerun

From the repository root:

```bash
PYTHONPATH=research_benchmark/src .venv/bin/python additional_experiment/reviewer_issue_p0_4_random_baseline/run_random_baseline.py --replicates 300 --output-dir additional_experiment/reviewer_issue_p0_4_random_baseline
```

The completed run took about 25.5 minutes on this machine.
