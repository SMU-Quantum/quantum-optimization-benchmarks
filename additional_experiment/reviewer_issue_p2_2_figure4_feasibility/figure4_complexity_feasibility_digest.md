# Figure 4 Complexity/Feasibility Digest

Included rows are archived hardware outcomes with a recoverable backend-native transpiled two-qubit-gate count. Two MIS/QRAO artifacts are excluded from the figure data because one used `statevector_primitives` and one archived no backend/transpiled hardware metric.

## 1. Feasible and Infeasible Outcomes by Problem/Method

- MDKP/VQE: hardware_runs=12, feasible=12, infeasible=0.
- MDKP/CVaR-VQE: hardware_runs=12, feasible=12, infeasible=0.
- MDKP/QAOA: hardware_runs=12, feasible=12, infeasible=0.
- MDKP/MA-QAOA: hardware_runs=12, feasible=12, infeasible=0.
- MDKP/WS-QAOA: hardware_runs=12, feasible=12, infeasible=0.
- MDKP/PCE: hardware_runs=12, feasible=12, infeasible=0.
- MDKP/QRAO: hardware_runs=12, feasible=12, infeasible=0.
- MIS/VQE: hardware_runs=7, feasible=2, infeasible=5.
- MIS/CVaR-VQE: hardware_runs=7, feasible=3, infeasible=4.
- MIS/QAOA: hardware_runs=7, feasible=3, infeasible=4.
- MIS/MA-QAOA: hardware_runs=7, feasible=2, infeasible=5.
- MIS/WS-QAOA: hardware_runs=7, feasible=4, infeasible=3.
- MIS/PCE: hardware_runs=7, feasible=2, infeasible=5.
- MIS/QRAO: hardware_runs=5, feasible=2, infeasible=3.
- QAP/VQE: hardware_runs=11, feasible=0, infeasible=11.
- QAP/CVaR-VQE: hardware_runs=11, feasible=0, infeasible=11.
- QAP/QAOA: hardware_runs=14, feasible=0, infeasible=14.
- QAP/MA-QAOA: hardware_runs=8, feasible=0, infeasible=8.
- QAP/WS-QAOA: hardware_runs=9, feasible=0, infeasible=9.
- QAP/PCE: hardware_runs=6, feasible=0, infeasible=6.
- QAP/QRAO: hardware_runs=7, feasible=0, infeasible=7.
- MSP/VQE: hardware_runs=8, feasible=8, infeasible=0.
- MSP/CVaR-VQE: hardware_runs=8, feasible=8, infeasible=0.
- MSP/QAOA: hardware_runs=8, feasible=8, infeasible=0.
- MSP/MA-QAOA: hardware_runs=8, feasible=8, infeasible=0.
- MSP/WS-QAOA: hardware_runs=8, feasible=8, infeasible=0.
- MSP/PCE: hardware_runs=8, feasible=8, infeasible=0.
- MSP/QRAO: hardware_runs=8, feasible=8, infeasible=0.

## 2. Infeasible Panel Problem Families

MIS, QAP

## 3. Lowest-N2Q Infeasible Points

- MIS: 10 (PCE 1tc.32 on ibm_fez).
- QAP: 20 (PCE had12 on ibm_fez).
- MSP: no infeasible point.

## 4. Highest-N2Q Feasible Points

- MDKP: 65953 (WS-QAOA pet7 on ibm_torino).
- MIS: 14029 (WS-QAOA 1tc.64 on ibm_torino).

## 5. QAP Feasibility

All QAP results appear in the infeasible panel: true.

## 6. Reference Fidelity Thresholds

- Fref=0.1 corresponds to N2Q=766.4 (approximately 770).
- Fref=0.01 corresponds to N2Q=1532.8 (approximately 1540).

## 7. Fidelity Axis Interpretation

The top fidelity axis is a common reference mapping under epsilon_2Q_ref=3e-3, not each run's backend-specific F_est. Backend-specific fidelity estimates remain appendix/source-data quantities.
