# P0.4 Random Sampling Baseline Digest

## Scope
- Low-fidelity QAOA-family hardware runs included: 106.
- Methods included: QAOA, MA-QAOA, WS-QAOA.
- Methods excluded: PCE and QRAO, per reviewer-response scope.
- Low-fidelity filter: reconstructed gate-count fidelity proxy `< 1e-3`; the saved artifacts do not retain per-gate calibration error rates.
- Figure `plots/fig_low_fidelity_random_baseline.pdf` is appendix-only; use the aggregate group summary table in the main manuscript.

## Six Reviewer Questions
1. Candidate pool eligible: optimizer-trajectory batches only. The code stores the best counts from the optimizer evaluation with the lowest expectation objective; there is no independent final sampling batch in these artifacts.
2. Number of low-fidelity QAOA-family hardware runs: 106.
3. Outperform matched uniform random: 1 runs.
4. Random-comparable: 51 runs.
5. Local refinement materially changes the random baseline: median finite per-run improvement is 34.1778 in each run's primary-metric units; inspect `median_random_local_refinement_improvement` for problem-specific units.
6. Problem dependence: MDKP and MSP are mostly random-comparable with some underperformance; MIS is mostly infeasible or sparse evidence; QAP is feasibility-only because all hardware QAP runs are infeasible and valid uniform random permutations are astronomically unlikely.

## Candidate Counts
- Unique eligible candidate counts: 45000, 46000, 47000, 48000, 49000, 50000, 51000, 52000, 53000, 54000, 55000, 56000, 57000, 58000, 59000, 60000, 61000, 62000, 63000, 64000, 65000, 66000, 67000, 68000, 72000, 73000, 200000

## Classification Counts
- outperforms_matched_uniform_random: 1
- random_comparable: 51
- underperforms_matched_uniform_random: 11
- hardware_infeasible_random_feasibility_comparison_only: 43

## Group Summary
- MDKP/ma_qaoa: n=12, feasible=12, random_median=22.3966987926 optimality gap (%), tail_median=0.588039867109, comparable_fraction=0.75
- MDKP/qaoa: n=12, feasible=12, random_median=22.9213978435 optimality gap (%), tail_median=0.760797342192, comparable_fraction=0.666666666667
- MDKP/ws_qaoa: n=12, feasible=12, random_median=22.7784239873 optimality gap (%), tail_median=0.755813953489, comparable_fraction=1
- MIS/ma_qaoa: n=5, feasible=0, random_median=33.3333333333 optimality gap (%), tail_median=nan, comparable_fraction=0
- MIS/qaoa: n=5, feasible=1, random_median=33.3333333333 optimality gap (%), tail_median=0.275747508306, comparable_fraction=0.2
- MIS/ws_qaoa: n=5, feasible=2, random_median=33.3333333333 optimality gap (%), tail_median=0.0581395348836, comparable_fraction=0.2
- MSP/ma_qaoa: n=8, feasible=8, random_median=117.5 TDev, tail_median=0.662790697674, comparable_fraction=0.875
- MSP/qaoa: n=8, feasible=8, random_median=115 TDev, tail_median=0.704318936878, comparable_fraction=0.875
- MSP/ws_qaoa: n=8, feasible=8, random_median=115.5 TDev, tail_median=0.747508305648, comparable_fraction=0.75
- QAP/ma_qaoa: n=11, feasible=0, random_median=nan optimality gap (%), tail_median=nan, comparable_fraction=0
- QAP/qaoa: n=11, feasible=0, random_median=nan optimality gap (%), tail_median=nan, comparable_fraction=0
- QAP/ws_qaoa: n=9, feasible=0, random_median=nan optimality gap (%), tail_median=nan, comparable_fraction=0

## Interpretation Note
The matched random control addresses the low-fidelity / best-shot-selection concern directly: it asks whether the reported hardware candidate is better than selecting from the same number of uniformly random bitstrings under the same downstream local refinement. Because calibration fields are absent from the saved artifacts, the fidelity column should be described as a reconstructed selection proxy rather than a measured process fidelity.
