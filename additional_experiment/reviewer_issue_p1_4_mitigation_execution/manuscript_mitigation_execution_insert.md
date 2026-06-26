# Manuscript Correction Insert

## Execution and Mitigation Pipeline

| Stage | Primitive / setting | Role in reported result |
| --- | --- | --- |
| Circuit compilation | `optimization_level=3` | Backend-aware mapping, gate decomposition, routing, and circuit optimization before execution. This is transpilation, not error mitigation. |
| Variational objective evaluation | Runtime EstimatorV2 with `resilience_level=2` | Runtime-managed mitigation affected expectation-value estimates used by COBYLA objective evaluations. |
| Final candidate sampling | Separate SamplerV2 with raw/default options | Unmitigated/default bitstring distribution used for decoding. |
| Decoding and local improvement | Classical code | Candidate selection, feasibility checks, and shared one-round local refinement. |

ZNE/TREX affected Estimator expectation-value evaluations during optimization, not the final sampled bitstrings used to decode solutions. Because the optimizer path used COBYLA, these mitigated Estimator values informed gradient-free objective evaluations, not gradient calculations.

We requested IBM Runtime's managed medium-resilience preset through `resilience_level=2`. We did not manually configure or archive custom TREX, ZNE noise factors, ZNE extrapolators, twirling factors, or dynamical-decoupling settings. Therefore the manuscript should not report those settings as historical protocol details.

## Mitigation Overhead

Estimator `resilience_level=2` can increase execution cost by using managed ensembles of related circuits for mitigation. However, the historical campaign did not include paired `resilience_level=0` executions of the same circuits. We therefore do not estimate a circuit-specific mitigation-overhead multiplier retrospectively. The provenance ledger reports this explicitly for each archived result.
