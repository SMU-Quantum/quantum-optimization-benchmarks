# Structural Metrics Digest

Generated from `build_structural_metrics.py` using the repository problem loaders and Qiskit's `QuadraticProgramToQubo` conversion path.

## Density by problem
- MDKP: median 0.67772 (range 0.260359 to 0.756219).
- MIS: median 0.180979 (range 0.0952381 to 0.269345).
- QAP: median 0.818182 (range 0.292735 to 1).
- MSP: median 0.589268 (range 0.54938 to 0.680851).

## Coefficient dynamic range by problem
- MDKP: range 8530.04 to 674760.
- MIS: range 9 to 129.
- QAP: range 107185 to 2.20593e+10.
- MSP: range 98560.1 to 1.70189e+06.

## QAP feasibility geometry
- tai10a: log10 feasible fraction -23.5432.
- tai10b: log10 feasible fraction -23.5432.
- n=12 QAP instances: log10 feasible fraction -34.668.
- QAP near-complete coupling density: had12, rou12, tai10a, tai12a (threshold rho_Q >= 0.9).
- Smallest standard QAPLIB instances tested here: tai10a and tai10b (n=10).
- Reduced QAP calibration ladder: not run; this package only reports the required non-execution structural analyses.
