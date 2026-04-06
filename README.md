# Quantum Optimization Benchmarks

This repository is the code and artifact companion for the paper
_A Comparative Study of Quantum Optimization Techniques for Solving Combinatorial Optimization Benchmark Problems_.

It now has two clearly separated layers:

- Legacy benchmark notebooks and raw datasets in the top-level problem folders.
- A reproducible `qobench` research pipeline in [research_benchmark/README.md](research_benchmark/README.md) for scripted runs, smoke tests, hardware submissions, and structured result artifacts.

## Quickstart

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e research_benchmark
```

You can also use the repo targets:

```bash
make venv
make install
make test
```

## Common Commands

List a few benchmark instances:

```bash
make list-instances PROBLEM=mis LIST_LIMIT=5
```

Run one classical/QUBO benchmark instance:

```bash
make run PROBLEM=mis INSTANCE=Maximum_Independent_Set/mis_benchmark_instances/1tc.8.txt
```

Run one local simulator benchmark:

```bash
make simulate PROBLEM=mkp METHOD=qaoa INSTANCE=Multi_Dimension_Knapsack/MKP_Instances/sac94/hp/hp1.dat
```

Run the short multi-method smoke suite:

```bash
make smoke PROBLEM=mis
```

You can also call the installed console scripts directly:

```bash
qobench list-instances --problem mis --limit 5
qobench-hardware --help
qobench-smoke --problem mis
```

## Repository Guide

| Path | Purpose |
| --- | --- |
| [docs/PROJECT_LAYOUT.md](docs/PROJECT_LAYOUT.md) | High-level map of code, datasets, and artifacts. |
| [docs/REPRODUCIBILITY.md](docs/REPRODUCIBILITY.md) | Environment, commands, and artifact-writing conventions. |
| [research_benchmark/README.md](research_benchmark/README.md) | Main documentation for the packaged research pipeline. |
| [research_benchmark/research_benchmark/README.md](research_benchmark/research_benchmark/README.md) | Checked-in paper artifacts and figure/table outputs. |
| [research_benchmark/all_logs/README.md](research_benchmark/all_logs/README.md) | Preserved raw run logs from exploratory sweeps. |
| [research_benchmark/checkpoints/README.md](research_benchmark/checkpoints/README.md) | Resume metadata for long-running benchmark sweeps. |

## Benchmarks Included

- [Market_Share](Market_Share)
- [Maximum_Independent_Set](Maximum_Independent_Set)
- [Multi_Dimension_Knapsack](Multi_Dimension_Knapsack)
- [Quadratic_Assignment_Problem](Quadratic_Assignment_Problem)

The original notebooks remain in place for paper alignment and historical traceability.
The recommended entrypoint for new runs is the packaged workflow under `research_benchmark/`.

## Artifacts and Logs

The repository intentionally keeps the paper-facing artifacts in version control so visitors can inspect the reported outputs:

- `research_benchmark/research_benchmark/results_hardware/`
- `research_benchmark/research_benchmark/results_simulator/`
- `research_benchmark/all_logs/`
- `research_benchmark/checkpoints/`

Those directories are documented in-place instead of being hidden behind generated-only paths.

## Citation

If you use this repository, please cite:

```bibtex
@misc{sharma2025comparativestudyquantumoptimization,
  title={A Comparative Study of Quantum Optimization Techniques for Solving Combinatorial Optimization Benchmark Problems},
  author={Monit Sharma and Hoong Chuin Lau},
  year={2025},
  eprint={2503.12121},
  archivePrefix={arXiv},
  primaryClass={quant-ph},
  url={https://arxiv.org/abs/2503.12121}
}
```

## License

MIT. See [LICENSE](LICENSE).
