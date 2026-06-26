# Manuscript Table: MDKP Layer A MPS Stability

| MDKP case | MPS setting | max chi | sampled TVD vs production | raw feasible mass | mean decoded gap (%) | SD | best gap (%) | feasible run fraction |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| cvar_pet2 | production_uncapped_1e-16 | 64 | 0 | 0.6419 | 11.27 | 6.153 | 0.2136 | 1 |
| cvar_pet2 | restricted_cap_32_1e-16 | 64 | 0.04635 | 0.6418 | 11.27 | 6.153 | 0.2136 | 1 |
| cvar_pet2 | converged_cap_64_1e-16 | 64 | 0 | 0.6419 | 11.27 | 6.153 | 0.2136 | 1 |
| cvar_pet2 | conservative_cap_128_1e-16 | 64 | 0 | 0.6419 | 11.27 | 6.153 | 0.2136 | 1 |
| cvar_pet2 | threshold_uncapped_1e-12 | 65 | 0 | 0.6419 | 11.27 | 6.153 | 0.2136 | 1 |
| cvar_pet2 | threshold_uncapped_1e-8 | 66 | 0.00135 | 0.6421 | 11.27 | 6.153 | 0.2136 | 1 |
| vqe_hp1 | production_uncapped_1e-16 | 66 | 0 | 0.6036 | 18.69 | 4.927 | 9.977 | 1 |
| vqe_hp1 | restricted_cap_33_1e-16 | 66 | 0.00585 | 0.6034 | 18.69 | 4.927 | 9.977 | 1 |
| vqe_hp1 | converged_cap_128_1e-16 | 66 | 0 | 0.6036 | 18.69 | 4.927 | 9.977 | 1 |
| vqe_hp1 | conservative_cap_256_1e-16 | 66 | 0 | 0.6036 | 18.69 | 4.927 | 9.977 | 1 |
| vqe_hp1 | threshold_uncapped_1e-12 | 66 | 0 | 0.6036 | 18.69 | 4.927 | 9.977 | 1 |
| vqe_hp1 | threshold_uncapped_1e-8 | 67 | 0.00035 | 0.6036 | 18.69 | 4.927 | 9.977 | 1 |

Notes:

- Each setting uses 20 independent final-sampling repetitions with 1000 shots per repetition.
- VQE/CVaR circuits were reconstructed from the legacy RY plus circular-entanglement ansatz implied by the saved circuit metrics and final parameter vectors.
- `pce_hp1` was not included because the saved run used an older 182-parameter Brickwork PCE circuit, while the current source reconstructs a 42-parameter PCE ansatz and no serialized circuit artifact is present.
