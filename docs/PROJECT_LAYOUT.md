# Project Layout

This repository keeps the paper companion structure explicit instead of hiding it behind a single package directory.

## Top-Level Problem Folders

- `Market_Share/`: original market-share notebooks, inputs, and classical result files.
- `Maximum_Independent_Set/`: MIS instances and notebook workflow.
- `Multi_Dimension_Knapsack/`: MKP instances and notebook workflow.
- `Quadratic_Assignment_Problem/`: QAP instances and notebook workflow.

These folders are preserved because they match the paper-era notebook organization and benchmark sources.

## Runnable Research Project

- `research_benchmark/src/qobench/`: installable Python package.
- `research_benchmark/run_benchmark.py`: model/QUBO CLI wrapper.
- `research_benchmark/run_hardware_benchmark.py`: hardware/backend CLI wrapper.
- `research_benchmark/run_simulator_benchmark.py`: simulator-only wrapper.
- `research_benchmark/run_all_algorithms_smoke.py`: short all-method smoke runner.
- `research_benchmark/examples/`: sample JSON inputs and credential template.

## Versioned Artifacts

- `research_benchmark/research_benchmark/results_hardware/`: checked-in hardware summaries, traces, plots, and tables.
- `research_benchmark/research_benchmark/results_simulator/`: checked-in simulator results.
- `research_benchmark/research_benchmark/simulator_checkpoints/`: simulator resume files.
- `research_benchmark/all_logs/`: raw exploratory logs preserved for auditability.
- `research_benchmark/checkpoints/`: checkpoint JSONs for batch runs.

Those directories are documented locally:

- [../research_benchmark/research_benchmark/README.md](../research_benchmark/research_benchmark/README.md)
- [../research_benchmark/all_logs/README.md](../research_benchmark/all_logs/README.md)
- [../research_benchmark/checkpoints/README.md](../research_benchmark/checkpoints/README.md)

## Repo-Level Helpers

- `Makefile`: common setup and execution commands.
- `requirements.txt`: environment used for the paper companion repo.
- `.gitignore`: excludes local virtualenv and transient caches while keeping paper artifacts versioned.
