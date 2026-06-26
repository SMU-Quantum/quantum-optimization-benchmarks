# Mitigation and Execution Provenance Digest

## Corrected execution statement

Circuit compilation used `optimization_level=3` for backend-aware mapping, decomposition, routing, and optimization. This is transpilation, not error mitigation.

Variational objective evaluations used Runtime Estimator jobs with `resilience_level=2`; the historical protocol did not archive custom TREX, ZNE, twirling, extrapolator, or dynamical-decoupling settings beyond this managed preset.

Final candidate bitstrings came from a separate raw/default SamplerV2 path and then classical decoding, feasibility checks, and one-round local refinement. Therefore ZNE/TREX affected Estimator expectation-value evaluations used by the COBYLA objective, not a gradient and not the final sampled bitstrings used for decoding.

## Overhead answer

The archived campaign does not contain paired `resilience_level=0` jobs for the same circuits. The paper should not report a circuit-specific mitigation-overhead multiplier retrospectively. It can state that level-2 Runtime Estimator mitigation increases execution cost through managed ensembles of related circuits, but the exact historical multiplier was not isolated.

## Archived metadata coverage

- Provenance rows: 255
- Problems: MDKP=84, MIS=49, MSP=56, QAP=66
- Methods: cvar_vqe=38, ma_qaoa=35, pce=33, qaoa=41, qrao=34, vqe=38, ws_qaoa=36
- Backends observed in archived metadata: aer_simulator_mps, ibm_fez, ibm_kingston, ibm_marrakesh, ibm_torino, statevector_primitives
- Rows with mixed local fallback metadata alongside IBM metadata: 2

## Claims to remove

- Do not claim archived custom values for `resilience.measure_mitigation`, TREX randomizations/shots, ZNE noise factors, or ZNE extrapolator.
- Do not claim ZNE is unbiased in the limit of a perfect noise model.
- Do not describe final SamplerV2 bitstrings as TREX- or ZNE-mitigated.
