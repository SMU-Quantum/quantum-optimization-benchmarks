# Penalty Audit Digest

1. Exact QUBO penalty terms:
- MDKP: `-sum_i p_i x_i + lambda_K sum_k (sum_i w_ki x_i + s_k - C_k)^2` after integer slack conversion.
- MIS: `-sum_i x_i + lambda_MIS sum_(i,j in E) x_i x_j`.
- QAP: dense flow-distance objective plus `lambda_Q` row/column one-hot penalties.
- MSP: absolute-deviation objective with target equalities enforced by `lambda_M` after binary expansion of deviation variables.

2. Exact code formula selecting each reported penalty value:
`lambda = 1.0 + (linear.upperbound - linear.lowerbound) + (quadratic.upperbound - quadratic.lowerbound)` from Qiskit's `LinearEqualityToPenalty` or `LinearInequalityToPenalty` converter stage.

3. Selection type: analytical Qiskit auto-bound rule, not hardware tuned and not grid tuned.
4. Inputs: pre-penalty objective linear/quadratic expression bounds at the constraint-elimination stage; per-instance values are in `selection_inputs`.
5. One penalty is shared across all constraints converted in each instance.
6. QUBO normalization before Hamiltonian construction: no.
7. Normalization factor/stage: factor 1; the converted QUBO is passed directly to `to_ising()` or to method-specific encodings.
8. All method families consume the same original penalized QUBO; PCE and QRAO encode/reduce only after that QUBO is built.

9. Four-instance penalty-sensitivity study:
- MDKP hp1: smallest certified feasible multiplier=not_applicable; reported feasible=unresolved_noncertified_incumbent_feasible=true; reported matches reference=unresolved_noncertified; higher penalties preserve reference=unresolved_noncertified_rows_present.
- MIS 1tc.32: smallest certified feasible multiplier=0.25; reported feasible=true; reported matches reference=true; higher penalties preserve reference=true.
- QAP tai10a: smallest certified feasible multiplier=not_applicable; reported feasible=unresolved_noncertified_incumbent_feasible=true; reported matches reference=unresolved_noncertified; higher penalties preserve reference=unresolved_noncertified_rows_present.
- MSP ms_seed0_prod3: smallest certified feasible multiplier=not_applicable; reported feasible=unresolved_noncertified_incumbent_feasible=true; reported matches reference=unresolved_noncertified; higher penalties preserve reference=unresolved_noncertified_rows_present.

10. Limitations:
- The sensitivity study is classical and formulation-level only; it does not rerun quantum hardware.
- Gurobi time limits are reported in replicate rows; any non-certified row must not be used as proof of penalty sufficiency.
- Raw lambda values are not cross-family hardness measures because objective scales and encodings differ.

Penalty scale ranges in the tested benchmark set:
- MDKP: 4022 to 182685.
- MIS: 9 to 129.
- QAP: 107185 to 1.102967e+10.
- MSP: 1963 to 20613.
