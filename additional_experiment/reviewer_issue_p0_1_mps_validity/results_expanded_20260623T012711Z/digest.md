# Expanded MPS Validity Digest

## Run Scope

- Audited cases: 7 exact-reference cases and 315 MPS sweep rows.
- Problems/methods covered by currently available simulator artifacts: MIS/PCE only.
- Bond-dimension caps: uncapped, 2, 4, 8, 16, 32, 64, 128, 256.
- Truncation thresholds: 1e-16, 1e-14, 1e-12, 1e-10, 1e-8.

## Environment

- Python executable: `/Users/monitsharma/SMU-Quantum-Repos/quantum-optimization-benchmarks/.venv/bin/python`
- Qiskit: `2.3.0`
- Qiskit Aer: `0.17.2`
- CPU: `Apple M3 Pro`
- Physical memory: `38654705664` bytes
- GPU used: `False`

## Main Findings

- Maximum observed MPS bond dimension: `4`.
- Uncapped MPS maximum TVD vs exact: `2.878e-11`.
- Bond cap 2 maximum TVD vs uncapped MPS: `2.409e-01`. This cap is too small for these circuits.
- Bond caps >=4 maximum TVD vs uncapped MPS: `0.000e+00`.
- Changing truncation threshold from 1e-16 through 1e-8 did not change the convergence conclusion for these cases.

## Manuscript Use

For the currently available MIS/PCE simulator artifacts, exact statevector simulation validates the Aer MPS baseline to numerical precision when the MPS bond dimension is at least 4 or uncapped. Artificially capping the bond dimension at 2 produces substantial distribution error, so the manuscript should explicitly state that the production MPS runs were uncapped and should report the observed bond-dimension diagnostic.

This does not yet validate all manuscript-wide simulator claims, because the repository currently exposes only MIS/PCE simulator result files under the audited glob. The same audit should be rerun after generating simulator artifacts for MDKP, QAP, market-share, and the non-PCE methods used in the manuscript.

## Files

- `mps_validation.csv`: full row-level audit.
- `convergence_by_setting.csv`: compact threshold/bond-dimension summary.
- `summary.md`: full markdown table.
- `environment.json`: classical resource and simulator-version metadata.
