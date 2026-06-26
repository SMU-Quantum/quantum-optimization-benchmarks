# MPS Validity Audit Summary

- Python: `/Users/monitsharma/SMU-Quantum-Repos/quantum-optimization-benchmarks/.venv/bin/python`
- Qiskit: `2.3.0`
- Qiskit Aer: `0.17.2`
- CPU: `Apple M3 Pro`
- Platform: `macOS-26.5.1-arm64-arm-64bit`
- GPU used: `False`

| case | qubits | method | threshold | bond cap | max observed bond | TVD vs exact | TVD vs uncapped | expectation diff vs exact | runtime sec |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mis/pce/1dc.128.txt | 10 | statevector |  |  |  | 0 |  |  | 0.00278 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 |  | 4 | 4.54e-12 | 0 |  | 0.00252 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 16 | 4 | 4.54e-12 | 0 |  | 0.00225 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 32 | 4 | 4.54e-12 | 0 |  | 0.00218 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 64 | 4 | 4.54e-12 | 0 |  | 0.00216 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 128 | 4 | 4.54e-12 | 0 |  | 0.00212 |
| mis/pce/1dc.64.txt | 8 | statevector |  |  |  | 0 |  |  | 0.0014 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 |  | 4 | 1.18e-11 | 0 |  | 0.00128 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 16 | 4 | 1.18e-11 | 0 |  | 0.00118 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 32 | 4 | 1.18e-11 | 0 |  | 0.00134 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 64 | 4 | 1.18e-11 | 0 |  | 0.00115 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 128 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1et.64.txt | 8 | statevector |  |  |  | 0 |  |  | 0.0014 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 |  | 4 | 1.61e-11 | 0 |  | 0.0013 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 16 | 4 | 1.61e-11 | 0 |  | 0.00117 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 32 | 4 | 1.61e-11 | 0 |  | 0.00117 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 64 | 4 | 1.61e-11 | 0 |  | 0.00115 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 128 | 4 | 1.61e-11 | 0 |  | 0.00125 |
| mis/pce/1tc.16.txt | 4 | statevector |  |  |  | 0 |  |  | 0.000887 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 |  | 4 | 8.22e-16 | 0 |  | 0.000644 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 16 | 4 | 8.22e-16 | 0 |  | 0.000574 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 32 | 4 | 8.22e-16 | 0 |  | 0.00055 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 64 | 4 | 8.22e-16 | 0 |  | 0.000536 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 128 | 4 | 8.22e-16 | 0 |  | 0.000536 |
| mis/pce/1tc.32.txt | 6 | statevector |  |  |  | 0 |  |  | 0.00108 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 |  | 4 | 2.35e-11 | 0 |  | 0.00103 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 16 | 4 | 2.35e-11 | 0 |  | 0.00083 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 32 | 4 | 2.35e-11 | 0 |  | 0.000803 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 64 | 4 | 2.35e-11 | 0 |  | 0.000773 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 128 | 4 | 2.35e-11 | 0 |  | 0.000902 |
| mis/pce/1tc.64.txt | 8 | statevector |  |  |  | 0 |  |  | 0.00138 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 |  | 4 | 2.88e-11 | 0 |  | 0.00128 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 16 | 4 | 2.88e-11 | 0 |  | 0.00134 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 32 | 4 | 2.88e-11 | 0 |  | 0.00116 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 64 | 4 | 2.88e-11 | 0 |  | 0.00114 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 128 | 4 | 2.88e-11 | 0 |  | 0.00115 |
| mis/pce/1tc.8.txt | 3 | statevector |  |  |  | 0 |  |  | 0.00119 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 |  | 4 | 5.2e-16 | 0 |  | 0.000749 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 16 | 4 | 5.2e-16 | 0 |  | 0.00056 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 32 | 4 | 5.2e-16 | 0 |  | 0.000529 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 64 | 4 | 5.2e-16 | 0 |  | 0.000576 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 128 | 4 | 5.2e-16 | 0 |  | 0.000539 |

## Notes

- `bond cap` is empty for uncapped Aer MPS.
- `max observed bond` is extracted from `save_matrix_product_state` and Aer MPS log metadata.
- Aer records the truncation threshold and bond dimensions, but this API path does not expose a full discarded-weight table. Use the exact and bond-sweep deltas as the empirical truncation-error diagnostic.
