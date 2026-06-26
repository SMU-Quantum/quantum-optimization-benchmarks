# Reduced Part B Seed Sensitivity Digest

## Scope

- Reduced campaign: 3 cases, 3 initialization seeds each.
- No Part C budget extension was run.
- Final-sampling repetitions are not independently rerun here; each row uses the benchmark runner's production final sampled counts. This is a limitation relative to the full requested Part B design.

## Cases

- `R1_mdkp_hp1_vqe`: mkp / hp1.dat / vqe - MDKP direct variational baseline; 60 logical qubits.
- `R2_mis_1tc32_ws_qaoa`: mis / 1tc.32.txt / ws_qaoa - Structured QAOA initialization case on a nontrivial MIS instance.
- `R3_mis_1tc16_pce`: mis / 1tc.16.txt / pce - Current reproducible PCE/encoding case; avoids legacy MDKP Brickwork artifact.

## Summary

| case | n seeds | median best energy | IQR best energy | feasible fraction | median evals | termination |
|---|---:|---:|---:|---:|---:|---|
| R1_mdkp_hp1_vqe | 3 | 2688076.0 | 1842426.0 | 1.0 | 200.0 | budget_reached |
| R2_mis_1tc32_ws_qaoa | 3 | -9.0 | 2.5 | 1.0 | 70.0 | optimizer_success |
| R3_mis_1tc16_pce | 3 | -2.0 | 0.0 | 1.0 | 200.0 | budget_reached |
