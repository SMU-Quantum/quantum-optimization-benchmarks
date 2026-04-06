from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

from .hardware_cli import main as hardware_main
from .paths import DEFAULT_PROJECT_ROOT, DEFAULT_SMOKE_RESULTS_ROOT, resolve_from_project_root
from .types import ProblemType


SUPPORTED_METHODS = (
    "vqe",
    "cvar_vqe",
    "qaoa",
    "cvar_qaoa",
    "ws_qaoa",
    "ma_qaoa",
    "pce",
    "qrao",
)


def _problem_choices() -> list[str]:
    return [problem.value for problem in ProblemType]


def _parse_methods(raw: str) -> list[str]:
    methods = [item.strip() for item in raw.split(",") if item.strip()]
    unknown = [method for method in methods if method not in SUPPORTED_METHODS]
    if unknown:
        raise ValueError(
            "Unsupported smoke-test methods: "
            + ", ".join(sorted(unknown))
            + ". Supported methods are: "
            + ", ".join(SUPPORTED_METHODS)
        )
    return methods


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qobench-smoke",
        description="Run a short local smoke test across multiple quantum methods.",
    )
    parser.add_argument("--project-root", default=str(DEFAULT_PROJECT_ROOT))
    parser.add_argument("--problem", required=True, choices=_problem_choices())
    parser.add_argument("--instance", default=None, help="Optional explicit instance path.")
    parser.add_argument("--methods", default=",".join(SUPPORTED_METHODS))
    parser.add_argument("--qpu-id", default="local_qiskit")
    parser.add_argument("--shots", type=int, default=128)
    parser.add_argument("--maxiter", type=int, default=10)
    parser.add_argument("--layers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-products", type=int, default=2)
    parser.add_argument("--output-root", default=str(DEFAULT_SMOKE_RESULTS_ROOT))
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue running later methods even if one smoke test fails.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    methods = _parse_methods(str(args.methods))

    smoke_stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    smoke_root = resolve_from_project_root(project_root, args.output_root) / f"smoke_{args.problem}_{smoke_stamp}"
    smoke_root.mkdir(parents=True, exist_ok=True)

    summary: dict[str, object] = {
        "problem": args.problem,
        "qpu_id": args.qpu_id,
        "shots": int(args.shots),
        "maxiter": int(args.maxiter),
        "layers": int(args.layers),
        "smoke_root": str(smoke_root),
        "results": [],
    }

    print(f"Smoke root: {smoke_root}")
    overall_success = True
    for method in methods:
        method_checkpoint_dir = smoke_root / "checkpoints" / method
        argv = [
            "--project-root",
            str(project_root),
            "--problem",
            args.problem,
            "--method",
            method,
            "--execution-mode",
            "single",
            "--qpu-id",
            args.qpu_id,
            "--only-qpu",
            args.qpu_id,
            "--shots",
            str(args.shots),
            "--maxiter",
            str(args.maxiter),
            "--layers",
            str(args.layers),
            "--seed",
            str(args.seed),
            "--num-products",
            str(args.num_products),
            "--output-root",
            str(smoke_root),
            "--checkpoint-dir",
            str(method_checkpoint_dir),
            "--force-rerun",
        ]
        if args.instance:
            argv.extend(["--instance", args.instance])
        if args.qpu_id == "local_qiskit":
            argv.extend(["--include-simulators", "--no-aws", "--no-ibm"])
        elif args.qpu_id == "ibm_quantum":
            argv.append("--no-aws")
        elif args.qpu_id in {"rigetti_ankaa3", "iqm_emerald", "iqm_garnet", "amazon_sv1"}:
            argv.append("--no-ibm")

        print(f"[{method}] starting")
        exit_code = hardware_main(argv)
        result = {
            "method": method,
            "exit_code": int(exit_code),
            "status": "ok" if exit_code == 0 else "failed",
            "checkpoint_dir": str(method_checkpoint_dir),
            "argv": argv,
        }
        print(f"[{method}] exit_code={exit_code}")
        summary["results"].append(result)

        if exit_code != 0:
            overall_success = False
            if not bool(args.continue_on_error):
                break

    (smoke_root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return 0 if overall_success else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
