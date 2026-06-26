# MPS Validity Audit Summary

- Python: `/Users/monitsharma/SMU-Quantum-Repos/quantum-optimization-benchmarks/.venv/bin/python`
- Qiskit: `2.3.0`
- Qiskit Aer: `0.17.2`
- CPU: `Apple M3 Pro`
- Platform: `macOS-26.5.1-arm64-arm-64bit`
- GPU used: `False`

| case | qubits | method | threshold | bond cap | max observed bond | TVD vs exact | TVD vs uncapped | expectation diff vs exact | runtime sec |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| mis/pce/1dc.128.txt | 10 | statevector |  |  |  | 0 |  |  | 0.00472 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 |  | 4 | 4.54e-12 | 0 |  | 0.00288 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 2 | 4 | 0.241 | 0.241 |  | 0.00225 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 4 | 4 | 4.54e-12 | 0 |  | 0.00237 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 8 | 4 | 4.54e-12 | 0 |  | 0.00228 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 16 | 4 | 4.54e-12 | 0 |  | 0.00216 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 32 | 4 | 4.54e-12 | 0 |  | 0.00231 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 64 | 4 | 4.54e-12 | 0 |  | 0.00218 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 128 | 4 | 4.54e-12 | 0 |  | 0.00231 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-16 | 256 | 4 | 4.54e-12 | 0 |  | 0.00221 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 |  | 4 | 4.54e-12 | 0 |  | 0.00216 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 2 | 4 | 0.241 | 0.241 |  | 0.00228 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 4 | 4 | 4.54e-12 | 0 |  | 0.00224 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 8 | 4 | 4.54e-12 | 0 |  | 0.00231 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 16 | 4 | 4.54e-12 | 0 |  | 0.00227 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 32 | 4 | 4.54e-12 | 0 |  | 0.00216 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 64 | 4 | 4.54e-12 | 0 |  | 0.00213 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 128 | 4 | 4.54e-12 | 0 |  | 0.00213 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-14 | 256 | 4 | 4.54e-12 | 0 |  | 0.00224 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 |  | 4 | 4.54e-12 | 0 |  | 0.00218 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 2 | 4 | 0.241 | 0.241 |  | 0.00214 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 4 | 4 | 4.54e-12 | 0 |  | 0.00217 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 8 | 4 | 4.54e-12 | 0 |  | 0.00228 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 16 | 4 | 4.54e-12 | 0 |  | 0.00224 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 32 | 4 | 4.54e-12 | 0 |  | 0.00212 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 64 | 4 | 4.54e-12 | 0 |  | 0.00212 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 128 | 4 | 4.54e-12 | 0 |  | 0.00208 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-12 | 256 | 4 | 4.54e-12 | 0 |  | 0.00221 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 |  | 4 | 4.54e-12 | 0 |  | 0.00216 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 2 | 4 | 0.241 | 0.241 |  | 0.00209 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 4 | 4 | 4.54e-12 | 0 |  | 0.00209 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 8 | 4 | 4.54e-12 | 0 |  | 0.00212 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 16 | 4 | 4.54e-12 | 0 |  | 0.00225 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 32 | 4 | 4.54e-12 | 0 |  | 0.00218 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 64 | 4 | 4.54e-12 | 0 |  | 0.00223 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 128 | 4 | 4.54e-12 | 0 |  | 0.00222 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-10 | 256 | 4 | 4.54e-12 | 0 |  | 0.00226 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 |  | 4 | 4.54e-12 | 0 |  | 0.0027 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 2 | 4 | 0.241 | 0.241 |  | 0.00216 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 4 | 4 | 4.54e-12 | 0 |  | 0.00213 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 8 | 4 | 4.54e-12 | 0 |  | 0.00224 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 16 | 4 | 4.54e-12 | 0 |  | 0.00218 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 32 | 4 | 4.54e-12 | 0 |  | 0.0022 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 64 | 4 | 4.54e-12 | 0 |  | 0.00215 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 128 | 4 | 4.54e-12 | 0 |  | 0.00223 |
| mis/pce/1dc.128.txt | 10 | matrix_product_state | 1e-08 | 256 | 4 | 4.54e-12 | 0 |  | 0.00217 |
| mis/pce/1dc.64.txt | 8 | statevector |  |  |  | 0 |  |  | 0.00135 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 |  | 4 | 1.18e-11 | 0 |  | 0.00154 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 2 | 4 | 0.0678 | 0.0678 |  | 0.00135 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 4 | 4 | 1.18e-11 | 0 |  | 0.00119 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 8 | 4 | 1.18e-11 | 0 |  | 0.00118 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 16 | 4 | 1.18e-11 | 0 |  | 0.00121 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 32 | 4 | 1.18e-11 | 0 |  | 0.00116 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 64 | 4 | 1.18e-11 | 0 |  | 0.00155 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 128 | 4 | 1.18e-11 | 0 |  | 0.00139 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-16 | 256 | 4 | 1.18e-11 | 0 |  | 0.00124 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 |  | 4 | 1.18e-11 | 0 |  | 0.00118 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 2 | 4 | 0.0678 | 0.0678 |  | 0.00121 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 4 | 4 | 1.18e-11 | 0 |  | 0.00127 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 8 | 4 | 1.18e-11 | 0 |  | 0.00127 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 16 | 4 | 1.18e-11 | 0 |  | 0.00123 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 32 | 4 | 1.18e-11 | 0 |  | 0.0012 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 64 | 4 | 1.18e-11 | 0 |  | 0.00116 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 128 | 4 | 1.18e-11 | 0 |  | 0.00116 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-14 | 256 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 |  | 4 | 1.18e-11 | 0 |  | 0.00116 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 2 | 4 | 0.0678 | 0.0678 |  | 0.00124 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 4 | 4 | 1.18e-11 | 0 |  | 0.00115 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 8 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 16 | 4 | 1.18e-11 | 0 |  | 0.00127 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 32 | 4 | 1.18e-11 | 0 |  | 0.00121 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 64 | 4 | 1.18e-11 | 0 |  | 0.00117 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 128 | 4 | 1.18e-11 | 0 |  | 0.00122 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-12 | 256 | 4 | 1.18e-11 | 0 |  | 0.00115 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 |  | 4 | 1.18e-11 | 0 |  | 0.00117 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 2 | 4 | 0.0678 | 0.0678 |  | 0.00124 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 4 | 4 | 1.18e-11 | 0 |  | 0.0012 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 8 | 4 | 1.18e-11 | 0 |  | 0.00119 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 16 | 4 | 1.18e-11 | 0 |  | 0.00125 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 32 | 4 | 1.18e-11 | 0 |  | 0.00116 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 64 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 128 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-10 | 256 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 |  | 4 | 1.18e-11 | 0 |  | 0.00126 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 2 | 4 | 0.0678 | 0.0678 |  | 0.00124 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 4 | 4 | 1.18e-11 | 0 |  | 0.00119 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 8 | 4 | 1.18e-11 | 0 |  | 0.00119 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 16 | 4 | 1.18e-11 | 0 |  | 0.00117 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 32 | 4 | 1.18e-11 | 0 |  | 0.00116 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 64 | 4 | 1.18e-11 | 0 |  | 0.00121 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 128 | 4 | 1.18e-11 | 0 |  | 0.00117 |
| mis/pce/1dc.64.txt | 8 | matrix_product_state | 1e-08 | 256 | 4 | 1.18e-11 | 0 |  | 0.00114 |
| mis/pce/1et.64.txt | 8 | statevector |  |  |  | 0 |  |  | 0.00146 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 |  | 4 | 1.61e-11 | 0 |  | 0.00134 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 2 | 4 | 0.0616 | 0.0616 |  | 0.00122 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 4 | 4 | 1.61e-11 | 0 |  | 0.00117 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 8 | 4 | 1.61e-11 | 0 |  | 0.00122 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 16 | 4 | 1.61e-11 | 0 |  | 0.00122 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 32 | 4 | 1.61e-11 | 0 |  | 0.00135 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 64 | 4 | 1.61e-11 | 0 |  | 0.00133 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 128 | 4 | 1.61e-11 | 0 |  | 0.00119 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-16 | 256 | 4 | 1.61e-11 | 0 |  | 0.00118 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 |  | 4 | 1.61e-11 | 0 |  | 0.00116 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 2 | 4 | 0.0616 | 0.0616 |  | 0.00128 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 4 | 4 | 1.61e-11 | 0 |  | 0.00123 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 8 | 4 | 1.61e-11 | 0 |  | 0.00118 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 16 | 4 | 1.61e-11 | 0 |  | 0.00116 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 32 | 4 | 1.61e-11 | 0 |  | 0.00115 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 64 | 4 | 1.61e-11 | 0 |  | 0.00128 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 128 | 4 | 1.61e-11 | 0 |  | 0.00121 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-14 | 256 | 4 | 1.61e-11 | 0 |  | 0.00114 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 |  | 4 | 1.61e-11 | 0 |  | 0.00113 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 2 | 4 | 0.0616 | 0.0616 |  | 0.00122 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 4 | 4 | 1.61e-11 | 0 |  | 0.00121 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 8 | 4 | 1.61e-11 | 0 |  | 0.00123 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 16 | 4 | 1.61e-11 | 0 |  | 0.00116 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 32 | 4 | 1.61e-11 | 0 |  | 0.00133 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 64 | 4 | 1.61e-11 | 0 |  | 0.00138 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 128 | 4 | 1.61e-11 | 0 |  | 0.00138 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-12 | 256 | 4 | 1.61e-11 | 0 |  | 0.00142 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 |  | 4 | 1.61e-11 | 0 |  | 0.00131 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 2 | 4 | 0.0616 | 0.0616 |  | 0.00137 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 4 | 4 | 1.61e-11 | 0 |  | 0.00121 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 8 | 4 | 1.61e-11 | 0 |  | 0.00141 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 16 | 4 | 1.61e-11 | 0 |  | 0.00124 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 32 | 4 | 1.61e-11 | 0 |  | 0.00114 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 64 | 4 | 1.61e-11 | 0 |  | 0.00116 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 128 | 4 | 1.61e-11 | 0 |  | 0.00116 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-10 | 256 | 4 | 1.61e-11 | 0 |  | 0.00132 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 |  | 4 | 1.61e-11 | 0 |  | 0.0012 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 2 | 4 | 0.0616 | 0.0616 |  | 0.00126 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 4 | 4 | 1.61e-11 | 0 |  | 0.00125 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 8 | 4 | 1.61e-11 | 0 |  | 0.00132 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 16 | 4 | 1.61e-11 | 0 |  | 0.00171 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 32 | 4 | 1.61e-11 | 0 |  | 0.00119 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 64 | 4 | 1.61e-11 | 0 |  | 0.00117 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 128 | 4 | 1.61e-11 | 0 |  | 0.00117 |
| mis/pce/1et.64.txt | 8 | matrix_product_state | 1e-08 | 256 | 4 | 1.61e-11 | 0 |  | 0.00127 |
| mis/pce/1tc.16.txt | 4 | statevector |  |  |  | 0 |  |  | 0.000863 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 |  | 4 | 8.22e-16 | 0 |  | 0.000644 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 2 | 4 | 0.043 | 0.043 |  | 0.000587 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 4 | 4 | 8.22e-16 | 0 |  | 0.000715 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 8 | 4 | 8.22e-16 | 0 |  | 0.000603 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 16 | 4 | 8.22e-16 | 0 |  | 0.00058 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 32 | 4 | 8.22e-16 | 0 |  | 0.000594 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 64 | 4 | 8.22e-16 | 0 |  | 0.000547 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 128 | 4 | 8.22e-16 | 0 |  | 0.000554 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-16 | 256 | 4 | 8.22e-16 | 0 |  | 0.000625 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 |  | 4 | 8.22e-16 | 0 |  | 0.000588 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 2 | 4 | 0.043 | 0.043 |  | 0.000641 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 4 | 4 | 8.22e-16 | 0 |  | 0.000582 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 8 | 4 | 8.22e-16 | 0 |  | 0.000569 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 16 | 4 | 8.22e-16 | 0 |  | 0.000628 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 32 | 4 | 8.22e-16 | 0 |  | 0.000589 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 64 | 4 | 8.22e-16 | 0 |  | 0.000647 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 128 | 4 | 8.22e-16 | 0 |  | 0.000623 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-14 | 256 | 4 | 8.22e-16 | 0 |  | 0.000595 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 |  | 4 | 8.22e-16 | 0 |  | 0.000629 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 2 | 4 | 0.043 | 0.043 |  | 0.000571 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 4 | 4 | 8.22e-16 | 0 |  | 0.00115 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 8 | 4 | 8.22e-16 | 0 |  | 0.000603 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 16 | 4 | 8.22e-16 | 0 |  | 0.000585 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 32 | 4 | 8.22e-16 | 0 |  | 0.000575 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 64 | 4 | 8.22e-16 | 0 |  | 0.000547 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 128 | 4 | 8.22e-16 | 0 |  | 0.000678 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-12 | 256 | 4 | 8.22e-16 | 0 |  | 0.000612 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 |  | 4 | 8.22e-16 | 0 |  | 0.000569 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 2 | 4 | 0.043 | 0.043 |  | 0.000562 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 4 | 4 | 8.22e-16 | 0 |  | 0.000574 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 8 | 4 | 8.22e-16 | 0 |  | 0.000551 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 16 | 4 | 8.22e-16 | 0 |  | 0.000657 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 32 | 4 | 8.22e-16 | 0 |  | 0.000588 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 64 | 4 | 8.22e-16 | 0 |  | 0.000548 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 128 | 4 | 8.22e-16 | 0 |  | 0.000538 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-10 | 256 | 4 | 8.22e-16 | 0 |  | 0.000547 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 |  | 4 | 8.22e-16 | 0 |  | 0.000541 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 2 | 4 | 0.043 | 0.043 |  | 0.000585 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 4 | 4 | 8.22e-16 | 0 |  | 0.000562 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 8 | 4 | 8.22e-16 | 0 |  | 0.000531 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 16 | 4 | 8.22e-16 | 0 |  | 0.000579 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 32 | 4 | 8.22e-16 | 0 |  | 0.000571 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 64 | 4 | 8.22e-16 | 0 |  | 0.000563 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 128 | 4 | 8.22e-16 | 0 |  | 0.000583 |
| mis/pce/1tc.16.txt | 4 | matrix_product_state | 1e-08 | 256 | 4 | 8.22e-16 | 0 |  | 0.000589 |
| mis/pce/1tc.32.txt | 6 | statevector |  |  |  | 0 |  |  | 0.00102 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 |  | 4 | 2.35e-11 | 0 |  | 0.000936 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 2 | 4 | 0.0391 | 0.0391 |  | 0.000852 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 4 | 4 | 2.35e-11 | 0 |  | 0.000798 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 8 | 4 | 2.35e-11 | 0 |  | 0.000876 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 16 | 4 | 2.35e-11 | 0 |  | 0.000831 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 32 | 4 | 2.35e-11 | 0 |  | 0.000797 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 64 | 4 | 2.35e-11 | 0 |  | 0.000869 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 128 | 4 | 2.35e-11 | 0 |  | 0.000787 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-16 | 256 | 4 | 2.35e-11 | 0 |  | 0.000873 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 |  | 4 | 2.35e-11 | 0 |  | 0.000839 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 2 | 4 | 0.0391 | 0.0391 |  | 0.000805 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 4 | 4 | 2.35e-11 | 0 |  | 0.00131 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 8 | 4 | 2.35e-11 | 0 |  | 0.000925 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 16 | 4 | 2.35e-11 | 0 |  | 0.000914 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 32 | 4 | 2.35e-11 | 0 |  | 0.000828 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 64 | 4 | 2.35e-11 | 0 |  | 0.000795 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 128 | 4 | 2.35e-11 | 0 |  | 0.000896 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-14 | 256 | 4 | 2.35e-11 | 0 |  | 0.000812 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 |  | 4 | 2.35e-11 | 0 |  | 0.000917 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 2 | 4 | 0.0391 | 0.0391 |  | 0.000823 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 4 | 4 | 2.35e-11 | 0 |  | 0.000805 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 8 | 4 | 2.35e-11 | 0 |  | 0.000802 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 16 | 4 | 2.35e-11 | 0 |  | 0.000837 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 32 | 4 | 2.35e-11 | 0 |  | 0.000842 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 64 | 4 | 2.35e-11 | 0 |  | 0.000821 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 128 | 4 | 2.35e-11 | 0 |  | 0.000786 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-12 | 256 | 4 | 2.35e-11 | 0 |  | 0.000796 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 |  | 4 | 2.35e-11 | 0 |  | 0.000811 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 2 | 4 | 0.0391 | 0.0391 |  | 0.000867 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 4 | 4 | 2.35e-11 | 0 |  | 0.000828 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 8 | 4 | 2.35e-11 | 0 |  | 0.000794 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 16 | 4 | 2.35e-11 | 0 |  | 0.000872 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 32 | 4 | 2.35e-11 | 0 |  | 0.000818 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 64 | 4 | 2.35e-11 | 0 |  | 0.000855 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 128 | 4 | 2.35e-11 | 0 |  | 0.00082 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-10 | 256 | 4 | 2.35e-11 | 0 |  | 0.000821 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 |  | 4 | 2.35e-11 | 0 |  | 0.000792 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 2 | 4 | 0.0391 | 0.0391 |  | 0.000814 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 4 | 4 | 2.35e-11 | 0 |  | 0.000853 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 8 | 4 | 2.35e-11 | 0 |  | 0.000778 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 16 | 4 | 2.35e-11 | 0 |  | 0.000771 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 32 | 4 | 2.35e-11 | 0 |  | 0.000822 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 64 | 4 | 2.35e-11 | 0 |  | 0.000876 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 128 | 4 | 2.35e-11 | 0 |  | 0.000866 |
| mis/pce/1tc.32.txt | 6 | matrix_product_state | 1e-08 | 256 | 4 | 2.35e-11 | 0 |  | 0.0008 |
| mis/pce/1tc.64.txt | 8 | statevector |  |  |  | 0 |  |  | 0.00126 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 |  | 4 | 2.88e-11 | 0 |  | 0.00136 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 2 | 4 | 0.113 | 0.113 |  | 0.00133 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 4 | 4 | 2.88e-11 | 0 |  | 0.00135 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 8 | 4 | 2.88e-11 | 0 |  | 0.00119 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 16 | 4 | 2.88e-11 | 0 |  | 0.00165 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 32 | 4 | 2.88e-11 | 0 |  | 0.00159 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 64 | 4 | 2.88e-11 | 0 |  | 0.00138 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 128 | 4 | 2.88e-11 | 0 |  | 0.00125 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-16 | 256 | 4 | 2.88e-11 | 0 |  | 0.00123 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 |  | 4 | 2.88e-11 | 0 |  | 0.00128 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 2 | 4 | 0.113 | 0.113 |  | 0.00122 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 4 | 4 | 2.88e-11 | 0 |  | 0.00118 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 8 | 4 | 2.88e-11 | 0 |  | 0.00118 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 16 | 4 | 2.88e-11 | 0 |  | 0.00134 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 32 | 4 | 2.88e-11 | 0 |  | 0.00127 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 64 | 4 | 2.88e-11 | 0 |  | 0.00121 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 128 | 4 | 2.88e-11 | 0 |  | 0.0012 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-14 | 256 | 4 | 2.88e-11 | 0 |  | 0.00135 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 |  | 4 | 2.88e-11 | 0 |  | 0.0012 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 2 | 4 | 0.113 | 0.113 |  | 0.00116 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 4 | 4 | 2.88e-11 | 0 |  | 0.00118 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 8 | 4 | 2.88e-11 | 0 |  | 0.00125 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 16 | 4 | 2.88e-11 | 0 |  | 0.00117 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 32 | 4 | 2.88e-11 | 0 |  | 0.0013 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 64 | 4 | 2.88e-11 | 0 |  | 0.00131 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 128 | 4 | 2.88e-11 | 0 |  | 0.00122 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-12 | 256 | 4 | 2.88e-11 | 0 |  | 0.00115 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 |  | 4 | 2.88e-11 | 0 |  | 0.00117 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 2 | 4 | 0.113 | 0.113 |  | 0.00125 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 4 | 4 | 2.88e-11 | 0 |  | 0.00129 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 8 | 4 | 2.88e-11 | 0 |  | 0.00122 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 16 | 4 | 2.88e-11 | 0 |  | 0.00119 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 32 | 4 | 2.88e-11 | 0 |  | 0.00133 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 64 | 4 | 2.88e-11 | 0 |  | 0.00121 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 128 | 4 | 2.88e-11 | 0 |  | 0.00118 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-10 | 256 | 4 | 2.88e-11 | 0 |  | 0.00128 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 |  | 4 | 2.88e-11 | 0 |  | 0.00121 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 2 | 4 | 0.113 | 0.113 |  | 0.00113 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 4 | 4 | 2.88e-11 | 0 |  | 0.00116 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 8 | 4 | 2.88e-11 | 0 |  | 0.00128 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 16 | 4 | 2.88e-11 | 0 |  | 0.00117 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 32 | 4 | 2.88e-11 | 0 |  | 0.00117 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 64 | 4 | 2.88e-11 | 0 |  | 0.00123 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 128 | 4 | 2.88e-11 | 0 |  | 0.00131 |
| mis/pce/1tc.64.txt | 8 | matrix_product_state | 1e-08 | 256 | 4 | 2.88e-11 | 0 |  | 0.00122 |
| mis/pce/1tc.8.txt | 3 | statevector |  |  |  | 0 |  |  | 0.00069 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 |  | 4 | 5.2e-16 | 0 |  | 0.000556 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 2 | 4 | 5.2e-16 | 0 |  | 0.000578 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 4 | 4 | 5.2e-16 | 0 |  | 0.000457 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 8 | 4 | 5.2e-16 | 0 |  | 0.00051 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 16 | 4 | 5.2e-16 | 0 |  | 0.000467 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 32 | 4 | 5.2e-16 | 0 |  | 0.000505 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 64 | 4 | 5.2e-16 | 0 |  | 0.000483 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 128 | 4 | 5.2e-16 | 0 |  | 0.000556 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-16 | 256 | 4 | 5.2e-16 | 0 |  | 0.0005 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 |  | 4 | 5.2e-16 | 0 |  | 0.000584 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 2 | 4 | 5.2e-16 | 0 |  | 0.000502 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 4 | 4 | 5.2e-16 | 0 |  | 0.000502 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 8 | 4 | 5.2e-16 | 0 |  | 0.00051 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 16 | 4 | 5.2e-16 | 0 |  | 0.000537 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 32 | 4 | 5.2e-16 | 0 |  | 0.000769 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 64 | 4 | 5.2e-16 | 0 |  | 0.000705 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 128 | 4 | 5.2e-16 | 0 |  | 0.00149 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-14 | 256 | 4 | 5.2e-16 | 0 |  | 0.00151 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 |  | 4 | 5.2e-16 | 0 |  | 0.00137 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 2 | 4 | 5.2e-16 | 0 |  | 0.00123 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 4 | 4 | 5.2e-16 | 0 |  | 0.000767 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 8 | 4 | 5.2e-16 | 0 |  | 0.000817 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 16 | 4 | 5.2e-16 | 0 |  | 0.00124 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 32 | 4 | 5.2e-16 | 0 |  | 0.000575 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 64 | 4 | 5.2e-16 | 0 |  | 0.00057 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 128 | 4 | 5.2e-16 | 0 |  | 0.000504 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-12 | 256 | 4 | 5.2e-16 | 0 |  | 0.000556 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 |  | 4 | 5.2e-16 | 0 |  | 0.000503 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 2 | 4 | 5.2e-16 | 0 |  | 0.000472 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 4 | 4 | 5.2e-16 | 0 |  | 0.000478 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 8 | 4 | 5.2e-16 | 0 |  | 0.000581 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 16 | 4 | 5.2e-16 | 0 |  | 0.000525 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 32 | 4 | 5.2e-16 | 0 |  | 0.000465 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 64 | 4 | 5.2e-16 | 0 |  | 0.000518 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 128 | 4 | 5.2e-16 | 0 |  | 0.000478 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-10 | 256 | 4 | 5.2e-16 | 0 |  | 0.00058 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 |  | 4 | 5.2e-16 | 0 |  | 0.0005 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 2 | 4 | 5.2e-16 | 0 |  | 0.000471 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 4 | 4 | 5.2e-16 | 0 |  | 0.000582 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 8 | 4 | 5.2e-16 | 0 |  | 0.000752 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 16 | 4 | 5.2e-16 | 0 |  | 0.000608 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 32 | 4 | 5.2e-16 | 0 |  | 0.000519 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 64 | 4 | 5.2e-16 | 0 |  | 0.000652 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 128 | 4 | 5.2e-16 | 0 |  | 0.000565 |
| mis/pce/1tc.8.txt | 3 | matrix_product_state | 1e-08 | 256 | 4 | 5.2e-16 | 0 |  | 0.000563 |

## Notes

- `bond cap` is empty for uncapped Aer MPS.
- `max observed bond` is extracted from `save_matrix_product_state` and Aer MPS log metadata.
- Aer records the truncation threshold and bond dimensions, but this API path does not expose a full discarded-weight table. Use the exact and bond-sweep deltas as the empirical truncation-error diagnostic.
