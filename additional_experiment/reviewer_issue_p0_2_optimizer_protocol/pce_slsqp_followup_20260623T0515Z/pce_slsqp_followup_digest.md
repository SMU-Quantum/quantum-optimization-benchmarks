# PCE/SLSQP Follow-up Digest

## Scope

- Reduced executable SLSQP diagnostic: 3 current-PCE cases, 3 seeds, maxiter values [100, 200].
- Historical PCE artifact audit: saved hardware PCE `result.json` files under `research_benchmark/research_benchmark/results_hardware/*/*_pce`.
- This is not an exact replay of the historical 182-parameter Brickwork MDKP PCE run because the saved artifacts do not serialize the original circuit/parameter-to-gate mapping and the benchmark source path records COBYLA, not SLSQP.

## Historical Artifact Audit

- PCE artifacts audited: 33.
- Recorded ansatz families: Brickwork, EfficientSU2.
- Recorded trainable-parameter counts: 132, 18, 182, 24, 240, 306, 36, 380, 462, 48, 60, 66.
- Recorded max optimizer iterations: 200.
- Recorded objective/circuit evaluations: 124, 184, 200.
- Git/source audit found SLSQP in exploratory PCE notebooks and QRAO optimizer support, while the benchmark PCE runner calls SciPy COBYLA.

## Reduced SLSQP Diagnostic Summary

| case | maxiter | seeds | median nfev | success rate | cap-hit rate | feasible rate | best gap | median gap | worst gap |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1_mis_1tc8_pce | 100 | 3 | 616.0 | 1.00 | 0.00 | 1.00 | 25.0 | 25.0 | 25.0 |
| S1_mis_1tc8_pce | 200 | 3 | 616.0 | 1.00 | 0.00 | 1.00 | 25.0 | 25.0 | 25.0 |
| S2_mis_1tc16_pce | 100 | 3 | 2590.0 | 0.00 | 1.00 | 1.00 | 62.5 | 62.5 | 62.5 |
| S2_mis_1tc16_pce | 200 | 3 | 5105.0 | 0.00 | 1.00 | 1.00 | 62.5 | 62.5 | 62.5 |
| S3_qap_nug12_pce | 100 | 3 | 7690.0 | 0.00 | 1.00 | 0.00 | inf | inf | inf |
| S3_qap_nug12_pce | 200 | 3 | 15317.0 | 0.00 | 1.00 | 0.00 | inf | inf | inf |

## Interpretation

- SLSQP iterations are not comparable to COBYLA objective evaluations; the table reports `nfev` as the cross-method budget metric.
- The reduced current-PCE diagnostic is useful for answering the SLSQP budget conceptually, but it should not be used to validate the historical Brickwork PCE hardware results.
- For the historical PCE benchmark results, the defensible revision is to report the recovered fixed-budget metadata and state that exact SLSQP replay is not available from the saved artifacts.

## Recommended Rebuttal Text

> We revisited the PCE optimizer records in response to the SLSQP/budget concern. The saved benchmark artifacts and the benchmark runner record PCE as a fixed-budget run with 200 objective/circuit evaluations in the reported hardware artifacts; the current source path uses SciPy COBYLA for PCE, while SLSQP appears only in exploratory PCE notebooks and in the QRAO optimizer option. Because the historical PCE artifacts do not serialize the original circuit or parameter-to-gate mapping, we do not claim exact replay of the historical Brickwork PCE runs. We therefore revised the manuscript to report objective evaluations rather than treating optimizer iterations as equivalent across methods, and we added a reduced SLSQP diagnostic on current PCE cases to illustrate the difference between SLSQP iterations and realized objective evaluations.

## Appendix Table

```latex
\begin{table*}[t]
\centering
\caption{Reduced PCE SLSQP diagnostic. Objective evaluations rather than optimizer iterations are reported as the cross-method cost metric. This diagnostic uses the current reproducible PCE implementation and is not an exact replay of the historical Brickwork PCE hardware artifacts.}
\label{tab:pce_slsqp_diagnostic}
\footnotesize
\begin{tabular}{llrrrrrrrr}
\toprule
Problem & Instance & Max. iter. & Median $n_{\mathrm{fev}}$ & Success rate & Cap-hit rate & Best gap & Median gap & Worst gap & Feasible rate \\
\midrule
MIS & \texttt{1tc.8.txt} & 100 & 616.0 & 1.00 & 0.00 & 25.0 & 25.0 & 25.0 & 1.00 \\
MIS & \texttt{1tc.8.txt} & 200 & 616.0 & 1.00 & 0.00 & 25.0 & 25.0 & 25.0 & 1.00 \\
MIS & \texttt{1tc.16.txt} & 100 & 2590.0 & 0.00 & 1.00 & 62.5 & 62.5 & 62.5 & 1.00 \\
MIS & \texttt{1tc.16.txt} & 200 & 5105.0 & 0.00 & 1.00 & 62.5 & 62.5 & 62.5 & 1.00 \\
QAP & \texttt{nug12.dat} & 100 & 7690.0 & 0.00 & 1.00 & inf & inf & inf & 0.00 \\
QAP & \texttt{nug12.dat} & 200 & 15317.0 & 0.00 & 1.00 & inf & inf & inf & 0.00 \\
\bottomrule
\end{tabular}
\end{table*}
```
