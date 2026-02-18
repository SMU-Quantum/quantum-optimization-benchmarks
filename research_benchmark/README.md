# Research Benchmark Pipeline

This folder contains a Python workflow for running benchmark optimization problems in a repeatable way.

It supports four problems:
- `mis` (Maximum Independent Set)
- `mkp` (Multi-Dimensional Knapsack)
- `qap` (Quadratic Assignment Problem)
- `market_share` (Market Share balancing)

You can run:
- the original notebook versions in the top-level problem folders
- the Python CLI for model/QUBO runs
- the hardware CLI for `vqe`, `cvar_vqe`, and `pce` on IBM, AWS, or local qiskit

## Setup

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 1) Run Original Notebook Versions

Use the notebook in each original folder:
- `Market_Share/market_share.ipynb`
- `Maximum_Independent_Set/mis.ipynb`
- `Multi_Dimension_Knapsack/mdkp.ipynb`
- `Quadratic_Assignment_Problem/qap.ipynb`

## 2) Run Python CLI (Model / QUBO Flow)

List instances:

```bash
.venv/bin/python research_benchmark/run_benchmark.py list-instances --problem mis
```

Run one problem instance:

```bash
.venv/bin/python research_benchmark/run_benchmark.py run --problem mkp --instance Multi_Dimension_Knapsack/MKP_Instances/sac94/hp/hp1.dat
```

Run with QUBO conversion:

```bash
.venv/bin/python research_benchmark/run_benchmark.py run --problem mis --to-qubo
```

## 3) Run Quantum Hardware Versions

Main entrypoint:

```bash
.venv/bin/python research_benchmark/run_hardware_benchmark.py --help
```

### Single QPU (example: MKP + CVaR-VQE on IBM)

```bash
.venv/bin/python research_benchmark/run_hardware_benchmark.py \
  --problem mkp \
  --instance Multi_Dimension_Knapsack/MKP_Instances/sac94/hp/hp1.dat \
  --method cvar_vqe \
  --execution-mode single \
  --qpu-id ibm_quantum \
  --only-qpu ibm_quantum \
  --shots 1000 \
  --maxiter 20
```

### Single QPU (example: MKP + PCE batching on IBM)

```bash
.venv/bin/python research_benchmark/run_hardware_benchmark.py \
  --problem mkp \
  --instance Multi_Dimension_Knapsack/MKP_Instances/sac94/hp/hp1.dat \
  --method pce \
  --execution-mode single \
  --qpu-id ibm_quantum \
  --only-qpu ibm_quantum \
  --pce-batch-size 4 \
  --shots 1000 \
  --maxiter 20
```

### Multi QPU (example: MIS + PCE)

```bash
.venv/bin/python research_benchmark/run_hardware_benchmark.py \
  --problem mis \
  --instance Maximum_Independent_Set/mis_benchmark_instances/1tc.8.txt \
  --method pce \
  --execution-mode multi \
  --qpus rigetti_ankaa3,iqm_emerald,ibm_quantum \
  --aws-profile quantum \
  --include-simulators
```

## Hardware Defaults and Behavior

- IBM credentials are loaded from saved system account by default.
- `--ibm-token` and `--ibm-instance` are optional overrides.
- AWS auth uses `--profile` or `--aws-profile`.
- IBM backend is selected from least-busy eligible devices.
- Queue status is printed at `--queue-status-seconds` interval (default: `120` seconds).
- Qiskit transpilation uses `--qiskit-optimization-level` (default: `3`).
- `--only-qpu` forces strict single-backend filtering.

## Outputs

Each hardware run writes to:

```text
research_benchmark/results_hardware/<problem>/<problem>_<method>_<timestamp>/
```

Artifacts:
- `result.json` (summary + metadata + final solution)
- `trace.jsonl` (evaluation trace)
- `best_counts.json` (best measurement counts)
- `qubo.lp` (QUBO LP dump)
- `run.log` (run logs)

Final objective values are reconstructed from the final bitstring at problem level.
A one-round local swap postprocess is applied at the end:
- maximization-aware for `MIS` and `MKP`
- minimization-aware for `QAP` and `market_share`

## Manual MDKP QPU Test Script

Quick manual test:

```bash
.venv/bin/python research_benchmark/tests/manual/test_mdkp_actual_qpu.py \
  --qpu-id ibm_quantum \
  --method cvar_vqe \
  --execution-mode single \
  --shots 1000 \
  --maxiter 20
```

Use `--dry-run` to print the exact command without submitting jobs.
