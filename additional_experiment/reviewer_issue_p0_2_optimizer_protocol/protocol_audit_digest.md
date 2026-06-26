# Optimizer Protocol Audit Digest

- Audited run artifacts: `262`.
- Non-replayable legacy artifacts: `23`.

## Termination Reasons

- `budget_reached`: 149
- `optimizer_success`: 113

## Key Cautions

- Fields not recoverable from artifacts are marked `unknown_legacy_artifact` or `not_applicable`.
- The current artifacts show `maxiter=200` for PCE runs; most PCE runs reached the function-evaluation budget.
- Legacy Brickwork PCE artifacts are marked `artifact_replayable=false` because no serialized circuit is saved and current source reconstructs a different PCE circuit.
- No tolerance values are present in the saved result artifacts; the manifest therefore does not claim tolerance convergence.
