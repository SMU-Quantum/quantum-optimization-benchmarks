# MDKP Layer A MPS Validation Digest

## Scope

- Fixed final-parameter validation for selected MDKP cases behind the negative simulator-hardware gap concern.
- Large 60/99-qubit circuits use sampled MPS counts rather than full statevector probability snapshots.
- The only changed variables across settings are MPS bond-dimension cap and truncation threshold.

## Environment

- Python executable: `/Users/monitsharma/SMU-Quantum-Repos/quantum-optimization-benchmarks/.venv/bin/python`
- Qiskit: `2.3.0`
- Qiskit Aer: `0.17.2`
- CPU: `Apple M3 Pro`
- Physical memory: `38654705664` bytes
- GPU used: `False`

## Setting Summary

| case | setting | chi max | TVD sampled vs production | mean gap % | sd gap % | best gap % | feasible run frac | raw feasible mass |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| vqe_hp1 | production_uncapped_1e-16 | 64 | 0 | 17.06 | 0 | 17.06 | 1 | 0.62 |
| vqe_hp1 | restricted_cap_32_1e-16 | 64 | 0 | 17.06 | 0 | 17.06 | 1 | 0.62 |
| vqe_hp1 | converged_cap_64_1e-16 | 64 | 0 | 17.06 | 0 | 17.06 | 1 | 0.62 |
| vqe_hp1 | conservative_cap_128_1e-16 | 64 | 0 | 17.06 | 0 | 17.06 | 1 | 0.62 |
| vqe_hp1 | threshold_uncapped_1e-12 | 64 | 0 | 17.06 | 0 | 17.06 | 1 | 0.62 |
| vqe_hp1 | threshold_uncapped_1e-8 | 67 | 0 | 17.06 | 0 | 17.06 | 1 | 0.62 |

## Files

- `mdkp_layer_a_summary.csv`: setting-level decoded-quality and distribution metrics.
- `mdkp_layer_a_replicates.csv`: per-repetition decoded-quality records.
- `skipped_cases.json`: cases that could not be exactly reconstructed from saved artifacts.
- `environment.json`: simulator and classical-resource metadata.
