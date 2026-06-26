# Issue P0.1: MPS Simulation Validity

This folder isolates the reviewer concern about Matrix Product State (MPS)
simulation validity and the manuscript's former title wording.

## Recommended manuscript action

1. The title has been revised to:
   `From Circuits to Hardware: Benchmarking Standard and Qubit-Efficient Quantum Optimization on Real Hardware`.
2. Add a simulator-details paragraph to the Methods section:
   - Qiskit version and Qiskit Aer version.
   - Aer method: `matrix_product_state`.
   - Maximum bond dimension: uncapped unless a sweep explicitly sets
     `matrix_product_state_max_bond_dimension`.
   - Truncation threshold: `matrix_product_state_truncation_threshold=1e-16`
     for the current environment unless changed by the experiment.
   - CPU model, memory, operating system, thread-related environment variables,
     and GPU use.
3. Add a validation table:
   - Exact statevector vs MPS for small encoded instances.
   - MPS bond-dimension convergence for representative larger circuits.
   - Report total-variation distance between output distributions, expectation
     differences where the circuit qubits match the QUBO variables, maximum
     observed MPS bond dimension, runtime, and memory metadata.
4. Add a limitations paragraph acknowledging that other simulation methods
   exist, including stabilizer/Clifford-style methods where applicable,
   Pauli propagation, tensor-network simulators such as Google's qsim/g-sim
   family, and Gibbs-state/classical thermal approaches. State that this work
   uses MPS as one scalable classical reference, not as proof of classical
   intractability.

## Experiment to run

From the repository root:

```bash
.venv/bin/python additional_experiment/reviewer_issue_p0_1_mps_validity/mps_validity_audit.py \
  --result-glob 'research_benchmark/research_benchmark/results_simulator/**/result.json' \
  --max-cases 7 \
  --bond-dimensions uncapped,16,32,64,128 \
  --thresholds 1e-16 \
  --shots 0
```

Outputs are written under `additional_experiment/reviewer_issue_p0_1_mps_validity/results/`:

- `environment.json`: simulator version and classical-resource metadata.
- `mps_validation.csv`: per-case exact/MPS/sweep metrics.
- `summary.md`: manuscript-ready summary table and interpretation.

Use `--shots N` only if reviewer-facing sampled-count comparisons are needed.
With `--shots 0`, the script uses exact probability snapshots from the
simulators, which is cleaner for numerical validation.

## Interpretation rule

Treat the MPS baseline as validated for a circuit family only when:

- exact-vs-MPS total-variation distance is negligible for exact-feasible cases;
- increasing the bond-dimension cap does not materially change output
  probabilities or objective expectations;
- the maximum observed bond dimension is below the imposed cap for capped runs,
  or the capped runs converge to the uncapped result.

If these checks fail, the affected simulator-to-hardware claims should be
softened and described as observations relative to a possibly approximate MPS
reference.
