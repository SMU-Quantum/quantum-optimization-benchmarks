#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


LOGGER = logging.getLogger("qobench.manual_test_mdkp")


def _configure_logging(output_root: Path, level_name: str) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    log_path = output_root / f"manual_test_mdkp_{run_stamp}.log"
    level = getattr(logging, str(level_name).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level '{level_name}'.")
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_path, encoding="utf-8"),
        ],
        force=True,
    )
    return log_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Manual integration test: run one MDKP instance on a chosen actual QPU.",
    )
    parser.add_argument("--python-exe", default=sys.executable, help="Python executable to run the benchmark CLI.")
    parser.add_argument(
        "--instance",
        default=None,
        help="Path to MKP instance. Defaults to sac94/hp/hp1.dat in this repository.",
    )
    parser.add_argument("--method", choices=["vqe", "cvar_vqe", "pce"], default="cvar_vqe")
    parser.add_argument("--execution-mode", choices=["single", "multi"], default="single")
    parser.add_argument(
        "--qpu-id",
        default="ibm_quantum",
        help="Primary QPU id. Use ibm_quantum, rigetti_ankaa3, iqm_emerald, iqm_garnet, etc.",
    )
    parser.add_argument(
        "--qpus",
        default=None,
        help="Comma-separated QPU ids for multi mode. If omitted, uses --qpu-id.",
    )
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--entanglement", choices=["chain", "full"], default="chain")
    parser.add_argument(
        "--qiskit-optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Qiskit transpiler optimization level for IBM transpilation.",
    )
    parser.add_argument(
        "--pce-batch-size",
        type=int,
        default=1,
        help="PCE-only: number of candidates per hardware submission.",
    )
    parser.add_argument(
        "--queue-status-seconds",
        type=float,
        default=120.0,
        help="How often to print queue/status updates while waiting for job completion.",
    )
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=0,
        help="Safety cap for QUBO size. 0 disables cap.",
    )

    parser.add_argument("--profile", default=None, help="AWS profile name (same behavior as evaluate_multi_qpu_bandits.py).")
    parser.add_argument("--ibm-instance", default=None, help="Optional IBM instance CRN.")
    parser.add_argument("--ibm-token", default=None, help="Optional IBM token (not required if saved system-wide).")
    parser.add_argument("--no-aws", action="store_true")
    parser.add_argument("--no-ibm", action="store_true")
    parser.add_argument("--include-simulators", action="store_true")

    parser.add_argument("--output-root", default="research_benchmark/results_hardware_manual")
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
        help="Logging verbosity for this test wrapper.",
    )
    parser.add_argument(
        "--benchmark-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
        help="Logging verbosity passed to run_hardware_benchmark.py.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    output_root = Path(args.output_root).resolve()
    test_log_path = _configure_logging(output_root=output_root, level_name=args.log_level)
    LOGGER.info("Starting manual MDKP hardware test")
    LOGGER.info("Test wrapper log file: %s", test_log_path)
    LOGGER.info(
        "Test config | method=%s execution_mode=%s qpu_id=%s qpus=%s shots=%s maxiter=%s",
        args.method,
        args.execution_mode,
        args.qpu_id,
        args.qpus,
        args.shots,
        args.maxiter,
    )

    repo_root = Path(__file__).resolve().parents[3]
    runner = repo_root / "research_benchmark" / "run_hardware_benchmark.py"
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    if args.instance is not None:
        instance_path = Path(args.instance)
        if not instance_path.is_absolute():
            instance_path = (repo_root / instance_path).resolve()
    else:
        instance_path = (
            repo_root
            / "Multi_Dimension_Knapsack"
            / "MKP_Instances"
            / "sac94"
            / "hp"
            / "hp1.dat"
        )

    if not instance_path.exists():
        raise FileNotFoundError(f"Instance not found: {instance_path}")
    LOGGER.info("Using MDKP instance: %s", instance_path)

    qpu_list = args.qpus if args.qpus is not None else args.qpu_id
    command = [
        str(args.python_exe),
        str(runner),
        "--problem",
        "mkp",
        "--instance",
        str(instance_path),
        "--method",
        args.method,
        "--execution-mode",
        args.execution_mode,
        "--shots",
        str(int(args.shots)),
        "--maxiter",
        str(int(args.maxiter)),
        "--seed",
        str(int(args.seed)),
        "--layers",
        str(int(args.layers)),
        "--entanglement",
        args.entanglement,
        "--qiskit-optimization-level",
        str(int(args.qiskit_optimization_level)),
        "--output-root",
        str(output_root),
        "--only-qpu",
        args.qpu_id,
        "--log-level",
        args.benchmark_log_level,
        "--queue-status-seconds",
        str(float(args.queue_status_seconds)),
    ]
    if int(args.max_qubits) > 0:
        command.extend(["--max-qubits", str(int(args.max_qubits))])

    if args.execution_mode == "single":
        command.extend(["--qpu-id", args.qpu_id])
    else:
        command.extend(["--qpus", qpu_list])

    if args.timeout_sec is not None:
        command.extend(["--timeout-sec", str(float(args.timeout_sec))])
    if args.method == "pce":
        command.extend(["--pce-batch-size", str(int(args.pce_batch_size))])
    if args.profile:
        command.extend(["--profile", args.profile])
    if args.ibm_instance:
        command.extend(["--ibm-instance", args.ibm_instance])
    if args.ibm_token:
        command.extend(["--ibm-token", args.ibm_token])
    if args.no_aws:
        command.append("--no-aws")
    if args.no_ibm:
        command.append("--no-ibm")
    if args.include_simulators:
        command.append("--include-simulators")

    print("Running manual MDKP hardware test command:")
    print(" ".join(shlex.quote(part) for part in command))
    LOGGER.debug("Benchmark command: %s", " ".join(shlex.quote(part) for part in command))

    if args.dry_run:
        LOGGER.info("Dry run requested; no benchmark process started.")
        print("Dry run complete.")
        return 0

    LOGGER.info("Launching benchmark process...")
    bench_out_path = output_root / "manual_test_mdkp_benchmark_stdout.log"
    run_dir: Path | None = None
    process = subprocess.Popen(
        command,
        cwd=str(repo_root),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )

    assert process.stdout is not None
    with bench_out_path.open("w", encoding="utf-8") as bench_out:
        for line in process.stdout:
            print(line, end="")
            bench_out.write(line)
            stripped = line.strip()
            if stripped.startswith("Run directory: "):
                maybe_dir = Path(stripped.split("Run directory: ", maxsplit=1)[1]).expanduser()
                run_dir = maybe_dir.resolve()

    return_code = int(process.wait())
    LOGGER.info("Benchmark process finished | return_code=%s", return_code)
    LOGGER.info("Raw benchmark stdout saved to: %s", bench_out_path)

    if return_code == 0 and run_dir is not None:
        result_path = run_dir / "result.json"
        if result_path.exists():
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
                timing = payload.get("timing", {})
                metrics = payload.get("ansatz_metrics", {})
                print("\nInstance Summary")
                print(f"Total solve time (s): {float(timing.get('solve_runtime_sec', 0.0)):.3f}")
                print(f"Total runtime (s): {float(timing.get('total_runtime_sec', 0.0)):.3f}")
                print(f"Qubits: {metrics.get('num_qubits')}")
                print(f"Trainable parameters: {metrics.get('trainable_parameters')}")
                print(f"Depth: {metrics.get('depth')}")
                print(f"One-qubit gates: {metrics.get('one_qubit_gates')}")
                print(f"Two-qubit gates: {metrics.get('two_qubit_gates')}")
                LOGGER.info("Printed compact instance summary from: %s", result_path)
            except Exception as exc:
                LOGGER.warning("Could not parse compact summary from %s: %s", result_path, exc)
    return return_code


if __name__ == "__main__":
    raise SystemExit(main())
