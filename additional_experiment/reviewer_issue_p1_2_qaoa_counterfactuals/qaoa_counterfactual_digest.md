# QAOA Compilation Counterfactual Digest

Compilation-only audit over 12 representative bound QAOA-family circuits: QAOA, MA-QAOA, and WS-QAOA for MDKP hp1, MIS 1tc.32, QAP tai10a, and MSP ms20 (mapped to generated ms_seed0_prod3).

1. SWAP-aware Heron mapping median two-qubit-gate reduction: -11.9791%.
2. Fractional-gate Heron median two-qubit-gate reduction: 4.06593%.
3. SWAP-aware Nighthawk median two-qubit-gate reduction: 19.7754%.
4. Circuits crossing F_est >= 0.01 under any counterfactual: none.
5. Circuits crossing F_est >= 1e-3 under any counterfactual: none.
6. The original broad QAOA conclusion should be softened: these results support a claim about the tested standard, MA-QAOA, and WS-QAOA implementations under the historical compilation/hardware settings, not QAOA in general.
7. Fractional-gate rows are compilation-resource counterfactuals only and are not directly comparable to the original resilience/ZNE configuration.

The saved historical artifacts omit calibrated per-gate errors, so all fidelity values use the same conservative gate-count proxy used in prior reviewer artifacts.
