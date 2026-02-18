from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

from .problem_registry import get_problem
from .runner import run_experiment
from .types import ProblemType, RunConfig


def _problem_choices() -> list[str]:
    return [problem.value for problem in ProblemType]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qobench",
        description="Research-friendly benchmark runner for combinatorial optimization problems.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser(
        "list-instances",
        help="List available dataset instances for a problem type.",
    )
    list_parser.add_argument("--problem", required=True, choices=_problem_choices())
    list_parser.add_argument("--limit", type=int, default=20)
    list_parser.add_argument("--project-root", default=".")

    run_parser = subparsers.add_parser(
        "run",
        help="Run a benchmark instance and write structured artifacts.",
    )
    run_parser.add_argument("--problem", required=True, choices=_problem_choices())
    run_parser.add_argument("--instance", default=None, help="Path to a dataset instance.")
    run_parser.add_argument("--seed", type=int, default=0)
    run_parser.add_argument("--time-limit", type=float, default=60.0)
    run_parser.add_argument("--to-qubo", action="store_true")
    run_parser.add_argument("--export-lp", action="store_true")
    run_parser.add_argument("--output-dir", default=None)
    run_parser.add_argument("--num-products", type=int, default=2)
    run_parser.add_argument("--project-root", default=".")

    return parser


def _run_command(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    config = RunConfig(
        problem=ProblemType(args.problem),
        project_root=project_root,
        instance_path=Path(args.instance) if args.instance else None,
        seed=int(args.seed),
        time_limit_sec=float(args.time_limit),
        to_qubo=bool(args.to_qubo),
        export_lp=bool(args.export_lp),
        output_dir=Path(args.output_dir) if args.output_dir else None,
        num_products=int(args.num_products),
    )

    try:
        artifacts = run_experiment(config)
    except ModuleNotFoundError as exc:
        print(f"Dependency error: {exc}")
        print("Install dependencies first, e.g. `pip install -r requirements.txt`.")
        return 1

    print(f"Run ID: {artifacts.run_id}")
    print(f"Status: {artifacts.status}")
    print(f"Output folder: {artifacts.output_dir}")
    print(f"Result JSON: {artifacts.result_json}")
    if artifacts.model_lp is not None:
        print(f"Model LP: {artifacts.model_lp}")
    if artifacts.qubo_lp is not None:
        print(f"QUBO LP: {artifacts.qubo_lp}")
    return 0


def _list_instances_command(args: argparse.Namespace) -> int:
    project_root = Path(args.project_root).resolve()
    problem = get_problem(args.problem)
    instances = problem.list_instances(project_root=project_root, limit=int(args.limit))

    if not instances:
        print("No instances found for this problem in the current project root.")
        return 0

    for path in instances:
        try:
            print(path.relative_to(project_root))
        except ValueError:
            print(path)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "list-instances":
        return _list_instances_command(args)
    if args.command == "run":
        return _run_command(args)

    parser.error(f"Unknown command: {args.command}")
    return 2
