# Reproducibility

## Environment Setup

From the repository root:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip install -e research_benchmark
```

Sanity checks:

```bash
make test
qobench --help
qobench-hardware --help
qobench-smoke --help
```

## Recommended Run Order

1. List available instances with `qobench list-instances`.
2. Run one classical/model/QUBO job with `qobench run`.
3. Run one local simulator method with `research_benchmark/run_simulator_benchmark.py`.
4. Run `qobench-smoke` before attempting longer hardware sweeps.
5. Launch hardware runs only after credentials and backend access are confirmed.

## Where Outputs Go

The repository uses stable default locations anchored to the checked-out repo, not the current shell directory:

- `research_benchmark/runs/`
- `research_benchmark/research_benchmark/results_hardware/`
- `research_benchmark/research_benchmark/results_simulator/`
- `research_benchmark/research_benchmark/simulator_checkpoints/`
- `research_benchmark/checkpoints/`
- `research_benchmark/results_hardware_smoke/`

That path normalization matters because earlier scripts could place artifacts in different locations depending on whether they were launched from the repo root or from inside `research_benchmark/`.

## Preserved Paper Artifacts

This repository intentionally keeps benchmark outputs, plots, summaries, and raw logs under version control so a reader can inspect the reported details directly on the repo.

Use these directories as read-mostly historical records:

- `research_benchmark/research_benchmark/`
- `research_benchmark/all_logs/`
- `research_benchmark/checkpoints/`

## Credentials

- Do not commit IBM tokens or AWS credentials.
- Use `research_benchmark/examples/ibm_credentials.example.json` as the shape reference for IBM credential rotation.
- Keep local secret files outside version control.
