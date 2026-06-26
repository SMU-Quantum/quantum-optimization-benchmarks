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
| cvar_pet2 | production_uncapped_1e-16 | 64 | 0 | 11.27 | 6.153 | 0.2136 | 1 | 0.6419 |
| cvar_pet2 | restricted_cap_32_1e-16 | 64 | 0.04635 | 11.27 | 6.153 | 0.2136 | 1 | 0.6418 |
| cvar_pet2 | converged_cap_64_1e-16 | 64 | 0 | 11.27 | 6.153 | 0.2136 | 1 | 0.6419 |
| cvar_pet2 | conservative_cap_128_1e-16 | 64 | 0 | 11.27 | 6.153 | 0.2136 | 1 | 0.6419 |
| cvar_pet2 | threshold_uncapped_1e-12 | 65 | 0 | 11.27 | 6.153 | 0.2136 | 1 | 0.6419 |
| cvar_pet2 | threshold_uncapped_1e-8 | 66 | 0.00135 | 11.27 | 6.153 | 0.2136 | 1 | 0.6421 |
| vqe_hp1 | production_uncapped_1e-16 | 66 | 0 | 18.69 | 4.927 | 9.977 | 1 | 0.6036 |
| vqe_hp1 | restricted_cap_33_1e-16 | 66 | 0.00585 | 18.69 | 4.927 | 9.977 | 1 | 0.6034 |
| vqe_hp1 | converged_cap_128_1e-16 | 66 | 0 | 18.69 | 4.927 | 9.977 | 1 | 0.6036 |
| vqe_hp1 | conservative_cap_256_1e-16 | 66 | 0 | 18.69 | 4.927 | 9.977 | 1 | 0.6036 |
| vqe_hp1 | threshold_uncapped_1e-12 | 66 | 0 | 18.69 | 4.927 | 9.977 | 1 | 0.6036 |
| vqe_hp1 | threshold_uncapped_1e-8 | 67 | 0.00035 | 18.69 | 4.927 | 9.977 | 1 | 0.6036 |

## Skipped Cases

- `pce_hp1`: Cannot reconstruct exact pce_hp1: stored theta has 182 parameters but current ansatz builder produces 42.

## Files

- `mdkp_layer_a_summary.csv`: setting-level decoded-quality and distribution metrics.
- `mdkp_layer_a_replicates.csv`: per-repetition decoded-quality records.
- `skipped_cases.json`: cases that could not be exactly reconstructed from saved artifacts.
- `environment.json`: simulator and classical-resource metadata.
