# Quantum Optimization Benchmarks

This repository accompanies the paper draft:

**Beyond Simulation: Benchmarking Standard and Qubit-Efficient Quantum Optimization on Real Hardware**
Monit Sharma and Hoong Chuin Lau

The project is built around a hardware-aware benchmark of gate-based quantum optimization on four NP-hard 0-1 combinatorial problem classes:

- Multi-Dimensional Knapsack Problem (MDKP)
- Maximum Independent Set (MIS)
- Quadratic Assignment Problem (QAP)
- Market Share Problem (MSP)

Rather than reporting only simulator-side objective values, the benchmark compares methods under realistic execution constraints, including logical qubit requirements, transpiled circuit growth, two-qubit gate counts, backend eligibility, and execution robustness on real hardware.

## What The Paper Studies

The paper compares three broad families of quantum optimization methods:

- Variational energy-minimization methods: `VQE`, `CVaR-VQE`
- QAOA-style methods: `QAOA`, `MA-QAOA`, `WS-QAOA`
- Qubit-efficient methods: `PCE`, `QRAO`

All methods are evaluated through a shared problem-to-QUBO workflow and then studied in both simulator and hardware settings. The goal is not to claim quantum advantage, but to establish a reproducible empirical baseline for how these methods behave once compilation, routing, noise, and recovery are treated as first-class parts of the experiment.

## Main Benchmark Message

The paper’s central findings are reflected directly in this repository:

- No single method family dominates across all four benchmark problems.
- Lower logical qubit count improves executability, but does not by itself guarantee better recovered hardware solution quality.
- Simulator conclusions do not always survive hardware execution.
- Compilation overhead, transpiled depth, and two-qubit gate growth materially affect practical outcomes.

In other words, this repository is meant to support **full-stack benchmarking**, not just logical-circuit comparison.

## Repository Structure

The repo has two complementary layers.

### 1. Original benchmark datasets and notebook workflows

- [Market_Share](Market_Share)
- [Maximum_Independent_Set](Maximum_Independent_Set)
- [Multi_Dimension_Knapsack](Multi_Dimension_Knapsack)
- [Quadratic_Assignment_Problem](Quadratic_Assignment_Problem)

These folders preserve the benchmark sources, problem-specific notebooks, and historical artifacts used during development.

### 2. Reproducible research pipeline

The packaged workflow lives under [research_benchmark/README.md](research_benchmark/README.md). It provides:

- a common `qobench` CLI for listing instances and running model/QUBO experiments
- a hardware-facing runner for `VQE`, `CVaR-VQE`, `QAOA`, `MA-QAOA`, `WS-QAOA`, `PCE`, and `QRAO`
- a simulator-only runner
- a smoke-test runner across all supported methods
- structured output directories for results, logs, checkpoints, and paper artifacts

## Benchmark Scope In This Repo

The repository includes the benchmark ingredients described in the paper:

- problem instances across MDKP, MIS, QAP, and MSP
- standard and qubit-efficient quantum optimization implementations
- simulator and hardware execution workflows
- preserved logs, summaries, plots, and result tables

Important artifact locations:

- `research_benchmark/research_benchmark/results_hardware/`
- `research_benchmark/research_benchmark/results_simulator/`
- `research_benchmark/all_logs/`
- `research_benchmark/checkpoints/`

Those directories are intentionally versioned so readers can inspect the paper-facing outputs directly.

## Quickstart

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e research_benchmark
```

You can also use the repo helpers:

```bash
make venv
make install
make test
```

## Common Commands

List benchmark instances:

```bash
make list-instances PROBLEM=mis LIST_LIMIT=5
```

Run one classical/QUBO benchmark instance:

```bash
make run PROBLEM=mis INSTANCE=Maximum_Independent_Set/mis_benchmark_instances/1tc.8.txt
```

Run one simulator benchmark:

```bash
make simulate PROBLEM=mkp METHOD=qaoa INSTANCE=Multi_Dimension_Knapsack/MKP_Instances/sac94/hp/hp1.dat
```

Run the short all-method smoke suite:

```bash
make smoke PROBLEM=mis
```

Installed console scripts:

```bash
qobench --help
qobench-hardware --help
qobench-smoke --help
```

## Documentation Guide

Start with these files:

| Path | Purpose |
| --- | --- |
| [research_benchmark/README.md](research_benchmark/README.md) | Main runnable research pipeline. |
| [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md) | Repository map. |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Environment and reproduction notes. |
| [research_benchmark/research_benchmark/README.md](research_benchmark/research_benchmark/README.md) | Checked-in paper artifacts. |
| [research_benchmark/all_logs/README.md](research_benchmark/all_logs/README.md) | Archived raw logs. |
| [research_benchmark/checkpoints/README.md](research_benchmark/checkpoints/README.md) | Resume metadata for long runs. |

## Citation

If you use this repository, please cite the accompanying paper/manuscript:

```bibtex
@misc{sharma2025beyondsimulation,
  title={Beyond Simulation: Benchmarking Standard and Qubit-Efficient Quantum Optimization on Real Hardware},
  author={Monit Sharma and Hoong Chuin Lau},
  year={2025}
}
```

## License

MIT. See [LICENSE](LICENSE).
