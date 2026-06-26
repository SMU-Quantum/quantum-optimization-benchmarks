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

## Skipped Cases

- `pce_hp1`: Cannot reconstruct exact pce_hp1: stored theta has 182 parameters but current ansatz builder produces 42.

## Files

- `mdkp_layer_a_summary.csv`: setting-level decoded-quality and distribution metrics.
- `mdkp_layer_a_replicates.csv`: per-repetition decoded-quality records.
- `skipped_cases.json`: cases that could not be exactly reconstructed from saved artifacts.
- `environment.json`: simulator and classical-resource metadata.
