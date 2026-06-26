# P1.2 QAOA Counterfactuals

This experiment addresses reviewer concerns about QAOA claims, SWAP-aware
compilation, fractional gates, and newer QAOA variants.

## What It Does

- Rebuilds representative QAOA-family circuits for MDKP, MIS, and QAP.
- Compares compilation choices and gate-count consequences.
- Produces counterfactual evidence for whether circuit depth and routing, rather
  than optimizer quality alone, explain degraded hardware behavior.

## Main Files

- `run_qaoa_compilation_audit.py`: regenerates the counterfactual audit.
- `qaoa_compilation_counterfactuals.csv`: per-case compilation data.
- `qaoa_compilation_counterfactual_summary.csv`: summarized comparisons.
- `fig_qaoa_compilation_counterfactuals.pdf`: reviewer-facing figure.
- `qaoa_counterfactual_digest.md`: interpretation and manuscript guidance.

## Rerun

```bash
.venv/bin/python additional_experiment/reviewer_issue_p1_2_qaoa_counterfactuals/run_qaoa_compilation_audit.py
```
