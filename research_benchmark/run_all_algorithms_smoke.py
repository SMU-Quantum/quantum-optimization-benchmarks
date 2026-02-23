#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable


ALL_METHODS = [
    "vqe",
    "cvar_vqe",
    "qaoa",
    "cvar_qaoa",
    "ws_qaoa",
    "ma_qaoa",
    "pce",
    "qrao",
]


@dataclass
class MethodResult:
    method: str
    command: list[str]
    return_code: int
    elapsed_sec: float
    stdout_log: str
    result_dir: str | None
    status: str


def _parse_methods(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return list(ALL_METHODS)
    parsed = [part.strip().lower() for part in raw.split(",") if part.strip()]
    invalid = [m for m in parsed if m not in ALL_METHODS]
    if invalid:
        raise ValueError(f"Unknown method(s): {', '.join(invalid)}")
    return parsed


def _resolve_instance(
    *,
    repo_root: Path,
    problem: str,
    instance_arg: str | None,
) -> Path:
    if instance_arg:
        candidate = Path(instance_arg)
        if not candidate.is_absolute():
            cwd_candidate = (Path.cwd() / candidate).resolve()
            if cwd_candidate.exists():
                return cwd_candidate
            repo_candidate = (repo_root / candidate).resolve()
            if repo_candidate.exists():
                return repo_candidate
            raise FileNotFoundError(
                "Instance not found. Tried: "
                f"{cwd_candidate} and {repo_candidate}"
            )
        if not candidate.exists():
            raise FileNotFoundError(f"Instance not found: {candidate}")
        return candidate.resolve()

    src_path = (repo_root / "research_benchmark" / "src").resolve()
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))
    from qobench.problem_registry import get_problem

    benchmark_problem = get_problem(problem)
    discovered = benchmark_problem.list_instances(repo_root, limit=1)
    if discovered:
        return discovered[0].resolve()

    default = benchmark_problem.default_instance(repo_root)
    if default is None:
        raise FileNotFoundError(
            f"No default/discoverable instance found for problem '{problem}'. Pass --instance."
        )
    if default.is_absolute():
        return default
    return (repo_root / default).resolve()


def _extract_result_dir(output: str) -> str | None:
    for line in output.splitlines():
        stripped = line.strip()
        if stripped.startswith("Results: "):
            return stripped.split("Results: ", maxsplit=1)[1].strip()
        if stripped.startswith("Run directory: "):
            return stripped.split("Run directory: ", maxsplit=1)[1].strip()
    return None


def _iter_stdout(proc: subprocess.Popen[str]) -> Iterable[str]:
    assert proc.stdout is not None
    for line in proc.stdout:
        yield line


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Smoke-test helper: run all implemented quantum methods on one instance with a short budget "
            "(default: 10 optimizer iterations)."
        ),
    )
    parser.add_argument(
        "--python-exe",
        default=sys.executable,
        help="Python executable used to launch research_benchmark/run_hardware_benchmark.py",
    )
    parser.add_argument("--problem", required=True, choices=["mis", "mkp", "qap", "market_share"])
    parser.add_argument(
        "--instance",
        default=None,
        help="Optional instance path. If omitted, first discoverable instance for --problem is used.",
    )
    parser.add_argument(
        "--methods",
        default=None,
        help="Comma-separated method subset. Default: all methods.",
    )
    parser.add_argument("--qpu-id", default="local_qiskit")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--entanglement", choices=["chain", "circular", "full"], default="chain")
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--output-root",
        default=None,
        help=(
            "Folder where smoke test logs and summary are written. "
            "Default: <repo>/research_benchmark/results_hardware_smoke."
        ),
    )
    parser.add_argument(
        "--benchmark-log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
    )
    parser.add_argument("--aws-profile", default=None)
    parser.add_argument("--ibm-token", default=None)
    parser.add_argument("--ibm-instance", default=None)
    parser.add_argument("--ibm-credentials-json", default=None)
    parser.add_argument("--no-aws", action="store_true")
    parser.add_argument("--no-ibm", action="store_true")
    parser.add_argument(
        "--skip-preflight",
        action="store_true",
        help="Skip local preflight validation (compile + parser unit test).",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser


def _run_preflight(*, repo_root: Path, python_exe: str, run_root: Path) -> None:
    preflight_dir = run_root / "preflight"
    preflight_dir.mkdir(parents=True, exist_ok=True)
    checks: list[tuple[str, list[str]]] = [
        (
            "compileall",
            [
                python_exe,
                "-m",
                "compileall",
                "research_benchmark/src/qobench",
                "research_benchmark/run_all_algorithms_smoke.py",
            ],
        ),
        (
            "parsers_unittest",
            [
                python_exe,
                "-m",
                "unittest",
                "research_benchmark/tests/test_parsers.py",
            ],
        ),
    ]
    for name, command in checks:
        print(f"[preflight] {name}: {' '.join(shlex.quote(part) for part in command)}")
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            check=False,
        )
        log_path = preflight_dir / f"{name}.log"
        log_path.write_text(
            (completed.stdout or "") + ("\n" if completed.stdout else "") + (completed.stderr or ""),
            encoding="utf-8",
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"Preflight check '{name}' failed (rc={completed.returncode}). See {log_path}"
            )
        print(f"[preflight] {name}: OK (log: {log_path})")


def main() -> int:
    args = build_parser().parse_args()
    methods = _parse_methods(args.methods)

    repo_root = Path(__file__).resolve().parents[1]
    runner = repo_root / "research_benchmark" / "run_hardware_benchmark.py"
    if not runner.exists():
        raise FileNotFoundError(f"Runner not found: {runner}")

    instance_path = _resolve_instance(
        repo_root=repo_root,
        problem=args.problem,
        instance_arg=args.instance,
    )

    if args.output_root is None:
        output_root = (repo_root / "research_benchmark" / "results_hardware_smoke").resolve()
    else:
        output_root = Path(args.output_root)
        if not output_root.is_absolute():
            output_root = (Path.cwd() / output_root).resolve()
        else:
            output_root = output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    run_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_root = output_root / f"smoke_{args.problem}_{run_stamp}"
    run_root.mkdir(parents=True, exist_ok=True)

    print("Smoke Test Configuration")
    print(f"  Problem:   {args.problem}")
    print(f"  Instance:  {instance_path}")
    print(f"  Methods:   {', '.join(methods)}")
    print(f"  QPU:       {args.qpu_id}")
    print(f"  Max Iter:  {args.maxiter}")
    print(f"  Shots:     {args.shots}")
    print(f"  Output:    {run_root}")
    print()

    if not args.skip_preflight:
        _run_preflight(
            repo_root=repo_root,
            python_exe=str(args.python_exe),
            run_root=run_root,
        )
        print()

    results: list[MethodResult] = []

    for method in methods:
        method_dir = run_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        stdout_log_path = method_dir / "stdout.log"
        command = [
            str(args.python_exe),
            str(runner),
            "--problem",
            args.problem,
            "--instance",
            str(instance_path),
            "--method",
            method,
            "--execution-mode",
            "single",
            "--qpu-id",
            args.qpu_id,
            "--only-qpu",
            args.qpu_id,
            "--shots",
            str(int(args.shots)),
            "--maxiter",
            str(int(args.maxiter)),
            "--seed",
            str(int(args.seed)),
            "--layers",
            str(int(args.layers)),
            "--entanglement",
            str(args.entanglement),
            "--log-level",
            str(args.benchmark_log_level),
            "--output-root",
            str(run_root / "hardware_runs"),
            "--force-rerun",
            "--include-simulators",
        ]
        if args.timeout_sec is not None:
            command.extend(["--timeout-sec", str(float(args.timeout_sec))])
        if args.aws_profile:
            command.extend(["--aws-profile", args.aws_profile])
        if args.ibm_token:
            command.extend(["--ibm-token", args.ibm_token])
        if args.ibm_instance:
            command.extend(["--ibm-instance", args.ibm_instance])
        if args.ibm_credentials_json:
            command.extend(["--ibm-credentials-json", args.ibm_credentials_json])
        if args.no_aws:
            command.append("--no-aws")
        if args.no_ibm:
            command.append("--no-ibm")

        if method == "pce":
            command.extend(
                [
                    "--pce-compression-k",
                    "2",
                    "--pce-depth",
                    "0",
                    "--pce-population",
                    "6",
                    "--pce-elite-frac",
                    "0.34",
                    "--pce-parallel-workers",
                    "1",
                    "--pce-batch-size",
                    "1",
                ]
            )
        if method in {"cvar_vqe", "cvar_qaoa"}:
            command.extend(["--cvar-alpha", "0.25"])
        if method == "ws_qaoa":
            command.extend(["--ws-epsilon", "0.001"])
        if method == "qrao":
            command.extend(
                [
                    "--qrao-max-vars-per-qubit",
                    "3",
                    "--qrao-reps",
                    "2",
                    "--qrao-rounding",
                    "magic",
                    "--qrao-optimizer",
                    "cobyla",
                ]
            )

        print(f"[{method}]")
        print("  Command:", " ".join(shlex.quote(part) for part in command))
        if args.dry_run:
            results.append(
                MethodResult(
                    method=method,
                    command=command,
                    return_code=0,
                    elapsed_sec=0.0,
                    stdout_log=str(stdout_log_path),
                    result_dir=None,
                    status="DRY_RUN",
                )
            )
            continue

        start = time.perf_counter()
        process = subprocess.Popen(
            command,
            cwd=str(repo_root),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        lines: list[str] = []
        with stdout_log_path.open("w", encoding="utf-8") as out_file:
            for line in _iter_stdout(process):
                print(line, end="")
                out_file.write(line)
                lines.append(line)
        return_code = int(process.wait())
        elapsed = float(time.perf_counter() - start)
        combined_output = "".join(lines)
        result_dir = _extract_result_dir(combined_output)
        status = "OK" if return_code == 0 else "FAILED"
        print(f"  Status: {status} | elapsed={elapsed:.1f}s")
        print()

        results.append(
            MethodResult(
                method=method,
                command=command,
                return_code=return_code,
                elapsed_sec=elapsed,
                stdout_log=str(stdout_log_path),
                result_dir=result_dir,
                status=status,
            )
        )

    ok_count = sum(1 for r in results if r.status in {"OK", "DRY_RUN"})
    summary_payload = {
        "timestamp_utc": run_stamp,
        "problem": args.problem,
        "instance": str(instance_path),
        "qpu_id": args.qpu_id,
        "maxiter": int(args.maxiter),
        "shots": int(args.shots),
        "methods": methods,
        "ok_count": ok_count,
        "total": len(results),
        "results": [asdict(r) for r in results],
    }
    summary_path = run_root / "summary.json"
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print("Smoke Test Summary")
    print(f"  Passed: {ok_count}/{len(results)}")
    print(f"  Summary JSON: {summary_path}")

    failed = [r for r in results if r.status == "FAILED"]
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
