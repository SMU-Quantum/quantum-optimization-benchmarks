# Figure 2 MIS 1tc.64 Digest

The plotting audit confirms the reviewer concern: `1tc.64` admits one feasible hardware solution and therefore should not be absent from panel (a) when the caption says panel (a) includes all MIS instances with at least one feasible hardware solution.

## Feasible Methods Per Instance

- 1tc.8: 7/7 feasible (VQE, CVaR-VQE, QAOA, MA-QAOA, WS-QAOA, QRAO, PCE).
- 1tc.16: 7/7 feasible (VQE, CVaR-VQE, QAOA, MA-QAOA, WS-QAOA, QRAO, PCE).
- 1tc.32: 4/7 feasible (CVaR-VQE, QAOA, WS-QAOA, QRAO).
- 1tc.64: 1/7 feasible (WS-QAOA).
- 1et.64: 0/7 feasible (none).
- 1dc.64: 0/7 feasible (none).
- 1dc.128: 0/7 feasible (none).

## 1tc.64 Resolution

`1tc.64` is now included in panel (a). Its only feasible method is WS-QAOA, with hardware objective 13 against BKS 20, giving a gap of 35%. The other six methods on `1tc.64` are marked as infeasible in panel (a), matching the `1/7` count in panel (b).

## Caption Replacement

Figure 2. Hardware performance and feasibility breakdown for MIS instances. Panel (a) reports the gap to BKS for every MIS instance with at least one feasible hardware solution; infeasible method-instance pairs within those groups are shown with hatched baseline markers. The 1tc.64 instance is included because WS-QAOA produced one feasible repaired hardware output, while the other six methods were infeasible. Panel (b) summarizes the number of feasible methods per instance as a function of problem size.
