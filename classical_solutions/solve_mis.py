#!/usr/bin/env python3
"""Classical exact baseline solver for Maximum Independent Set instances.

Default input:
  Maximum_Independent_Set/mis_benchmark_instances/**/*.txt

Default output:
  classical_solutions/results/mis/
"""

from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


PROBLEM_NAME = "MIS"
FORMULATION = "Exact binary integer programming formulation"
METHOD = "Gurobi branch-and-cut for maximum independent set"
DEFAULT_TIME_LIMIT_SEC = 3600.0


@dataclass(frozen=True)
class MISInstance:
    num_nodes: int
    edges: list[tuple[int, int]]
    source: Path


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def import_gurobi() -> tuple[Any, Any]:
    try:
        import gurobipy as gp  # type: ignore
        from gurobipy import GRB  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "gurobipy is not available in this interpreter. Install it in the repo venv with "
            "`uv pip install gurobipy --python ./.venv/bin/python`, then rerun this script."
        ) from exc
    return gp, GRB


def read_lines(path: Path) -> Iterable[str]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            yield from handle
    else:
        with path.open("r", encoding="utf-8") as handle:
            yield from handle


def parse_dimacs_graph(path: Path) -> MISInstance:
    if not path.exists():
        raise FileNotFoundError(f"MIS instance not found: {path}")

    num_nodes: int | None = None
    edges: list[tuple[int, int]] = []

    for raw_line in read_lines(path):
        line = raw_line.strip()
        if not line:
            continue
        parts = line.split()
        prefix = parts[0].lower()

        if prefix == "c":
            continue
        if prefix == "p":
            if len(parts) < 4:
                raise ValueError(f"Invalid MIS header in {path}: '{line}'")
            num_nodes = int(parts[2])
            continue
        if prefix == "e":
            if len(parts) < 3:
                raise ValueError(f"Invalid edge line in {path}: '{line}'")
            u = int(parts[1]) - 1
            v = int(parts[2]) - 1
            if u == v:
                continue
            a, b = (u, v) if u < v else (v, u)
            edges.append((a, b))

    if num_nodes is None:
        if not edges:
            raise ValueError(f"Could not infer node count from MIS instance: {path}")
        num_nodes = max(max(u, v) for u, v in edges) + 1

    unique_edges = sorted(set(edges))
    for u, v in unique_edges:
        if u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            raise ValueError(f"Edge ({u}, {v}) out of bounds for {num_nodes} nodes in {path}")

    return MISInstance(num_nodes=num_nodes, edges=unique_edges, source=path)


def list_instances(root: Path, limit: int | None) -> list[Path]:
    instances = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and (path.name.endswith(".txt") or path.name.endswith(".txt.gz"))
    )
    if limit is not None:
        instances = instances[:limit]
    return instances


def instance_id(instance_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(instance_root)
    except ValueError:
        rel = Path(path.name)
    if rel.name.endswith(".txt.gz"):
        parts = list(rel.parts)
        parts[-1] = parts[-1][:-7]
        return "__".join(parts)
    return "__".join(rel.with_suffix("").parts)


def finite_or_none(value: Any) -> float | int | str | None:
    if value is None:
        return None
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def safe_attr(obj: Any, name: str, default: Any = None) -> Any:
    try:
        return getattr(obj, name)
    except Exception:
        return default


def gurobi_status_name(GRB: Any, status: int | None) -> str | None:
    if status is None:
        return None
    mapping = {
        GRB.LOADED: "LOADED",
        GRB.OPTIMAL: "OPTIMAL",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
        GRB.CUTOFF: "CUTOFF",
        GRB.ITERATION_LIMIT: "ITERATION_LIMIT",
        GRB.NODE_LIMIT: "NODE_LIMIT",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
        GRB.INTERRUPTED: "INTERRUPTED",
        GRB.NUMERIC: "NUMERIC",
        GRB.SUBOPTIMAL: "SUBOPTIMAL",
        GRB.INPROGRESS: "INPROGRESS",
        GRB.USER_OBJ_LIMIT: "USER_OBJ_LIMIT",
        GRB.WORK_LIMIT: "WORK_LIMIT",
        GRB.MEM_LIMIT: "MEM_LIMIT",
    }
    return mapping.get(status, f"STATUS_{status}")


def configure_model(model: Any, args: argparse.Namespace) -> None:
    if args.time_limit and args.time_limit > 0:
        model.Params.TimeLimit = float(args.time_limit)
    if args.threads is not None and args.threads >= 0:
        model.Params.Threads = int(args.threads)
    if args.mip_gap is not None and args.mip_gap >= 0:
        model.Params.MIPGap = float(args.mip_gap)
    model.Params.MIPFocus = int(args.mip_focus)
    model.Params.DisplayInterval = int(args.display_interval)


def build_model(
    gp: Any,
    GRB: Any,
    instance: MISInstance,
    args: argparse.Namespace,
    log_path: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    if log_path.exists():
        log_path.unlink()
    env = gp.Env(empty=True)
    env.setParam("LogFile", str(log_path))
    env.setParam("LogToConsole", 1 if args.log_to_console else 0)
    env.start()

    model = gp.Model(f"MIS_{instance.source.stem}", env=env)
    configure_model(model, args)

    x = model.addVars(instance.num_nodes, vtype=GRB.BINARY, name="x")
    model.setObjective(gp.quicksum(x[i] for i in range(instance.num_nodes)), GRB.MAXIMIZE)
    model.addConstrs((x[u] + x[v] <= 1 for u, v in instance.edges), name="edge")
    model.update()

    context = {
        "x": x,
        "env": env,
    }
    return model, env, context


def solution_metrics(model: Any, instance: MISInstance, context: dict[str, Any]) -> dict[str, Any]:
    has_solution = safe_attr(model, "SolCount", 0) > 0
    selected_zero_based: list[int] = []
    selected_one_based: list[int] = []
    edge_violations: list[tuple[int, int]] = []
    binary_solution: list[int] = []

    if has_solution:
        x = context["x"]
        values = [float(x[i].X) for i in range(instance.num_nodes)]
        selected_zero_based = [idx for idx, value in enumerate(values) if value > 0.5]
        selected_one_based = [idx + 1 for idx in selected_zero_based]
        binary_solution = [int(round(value)) for value in values]
        selected = set(selected_zero_based)
        edge_violations = [(u + 1, v + 1) for u, v in instance.edges if u in selected and v in selected]

    return {
        "independent_set_zero_based": selected_zero_based,
        "independent_set_one_based": selected_one_based,
        "cardinality": len(selected_zero_based),
        "binary_solution": binary_solution,
        "edge_violation_count": len(edge_violations),
        "edge_violations_one_based": edge_violations,
    }


def solve_instance(
    gp: Any,
    GRB: Any,
    instance: MISInstance,
    instance_root: Path,
    output_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    inst_id = instance_id(instance_root, instance.source)
    logs_dir = output_dir / "logs"
    details_dir = output_dir / "details"
    solutions_dir = output_dir / "solutions"
    models_dir = output_dir / "models"
    for directory in (logs_dir, details_dir, solutions_dir):
        directory.mkdir(parents=True, exist_ok=True)
    if args.write_model:
        models_dir.mkdir(parents=True, exist_ok=True)

    log_path = logs_dir / f"{inst_id}_solver.log"
    details_path = details_dir / f"{inst_id}_details.txt"
    solution_path = solutions_dir / f"{inst_id}_solution.json"
    model_path = models_dir / f"{inst_id}.lp"

    started_at = datetime.now(timezone.utc).isoformat()
    build_start = time.perf_counter()
    model, env, context = build_model(gp, GRB, instance, args, log_path)
    build_wall_time = time.perf_counter() - build_start

    if args.write_model:
        model.write(str(model_path))

    solve_start = time.perf_counter()
    model.optimize()
    solve_wall_time = time.perf_counter() - solve_start

    has_solution = safe_attr(model, "SolCount", 0) > 0
    status_code = safe_attr(model, "Status", None)
    solution_data = solution_metrics(model, instance, context)
    version = ".".join(str(part) for part in gp.gurobi.version())

    record: dict[str, Any] = {
        "problem": PROBLEM_NAME,
        "instance": instance.source.name,
        "instance_id": inst_id,
        "instance_path": str(instance.source.resolve()),
        "formulation": FORMULATION,
        "classical_method": METHOD,
        "solver": "Gurobi",
        "solver_version": version,
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "started_at_utc": started_at,
        "time_limit_sec": args.time_limit if args.time_limit and args.time_limit > 0 else None,
        "threads": args.threads,
        "mip_gap_tolerance": args.mip_gap,
        "status_code": status_code,
        "status": gurobi_status_name(GRB, status_code),
        "primal_feasible": has_solution,
        "objective_value": safe_attr(model, "ObjVal", None) if has_solution else None,
        "best_bound": finite_or_none(safe_attr(model, "ObjBound", None)),
        "relative_gap": finite_or_none(safe_attr(model, "MIPGap", None)),
        "build_wall_time_sec": build_wall_time,
        "solve_wall_time_sec": solve_wall_time,
        "solver_runtime_sec": finite_or_none(safe_attr(model, "Runtime", None)),
        "solver_work": finite_or_none(safe_attr(model, "Work", None)),
        "nodes_processed": finite_or_none(safe_attr(model, "NodeCount", None)),
        "simplex_iterations": finite_or_none(safe_attr(model, "IterCount", None)),
        "barrier_iterations": finite_or_none(safe_attr(model, "BarIterCount", None)),
        "solution_count": safe_attr(model, "SolCount", None),
        "num_variables": safe_attr(model, "NumVars", None),
        "num_binary_variables": safe_attr(model, "NumBinVars", None),
        "num_constraints": safe_attr(model, "NumConstrs", None),
        "num_nonzeros": safe_attr(model, "NumNZs", None),
        "num_quadratic_nonzeros": safe_attr(model, "NumQNZs", None),
        "num_nodes": instance.num_nodes,
        "num_edges": len(instance.edges),
        "graph_density": (
            0.0
            if instance.num_nodes <= 1
            else (2.0 * len(instance.edges)) / (instance.num_nodes * (instance.num_nodes - 1))
        ),
        "log_file": str(log_path.resolve()),
        "details_file": str(details_path.resolve()),
        "solution_file": str(solution_path.resolve()),
    }
    if args.write_model:
        record["model_file"] = str(model_path.resolve())
    record.update(solution_data)

    solution_path.write_text(json.dumps(record, indent=2, sort_keys=True), encoding="utf-8")
    write_details(details_path, record)
    model.dispose()
    env.dispose()
    return record


def write_details(path: Path, record: dict[str, Any]) -> None:
    lines = [
        f"Problem: {record['problem']}",
        f"Instance: {record['instance']}",
        f"Instance path: {record['instance_path']}",
        f"Formulation: {record['formulation']}",
        f"Classical method: {record['classical_method']}",
        f"Solver: {record['solver']} {record['solver_version']}",
        f"Started at UTC: {record['started_at_utc']}",
        f"Status: {record['status']} (code {record['status_code']})",
        f"Primal feasible: {record['primal_feasible']}",
        f"Objective value: {record['objective_value']}",
        f"Best bound: {record['best_bound']}",
        f"Relative MIP gap: {record['relative_gap']}",
        f"Build wall time (sec): {record['build_wall_time_sec']}",
        f"Solve wall time (sec): {record['solve_wall_time_sec']}",
        f"Solver runtime (sec): {record['solver_runtime_sec']}",
        f"Solver work: {record['solver_work']}",
        f"Nodes processed: {record['nodes_processed']}",
        f"Simplex iterations: {record['simplex_iterations']}",
        f"Barrier iterations: {record['barrier_iterations']}",
        f"Solution count: {record['solution_count']}",
        f"Variables: {record['num_variables']}",
        f"Binary variables: {record['num_binary_variables']}",
        f"Constraints: {record['num_constraints']}",
        f"Nonzeros: {record['num_nonzeros']}",
        f"Quadratic nonzeros: {record['num_quadratic_nonzeros']}",
        f"Graph nodes: {record['num_nodes']}",
        f"Graph edges: {record['num_edges']}",
        f"Graph density: {record['graph_density']}",
        f"Independent set cardinality: {record['cardinality']}",
        f"Edge violation count: {record['edge_violation_count']}",
        f"Time limit (sec): {record['time_limit_sec']}",
        f"Threads: {record['threads']}",
        f"MIP gap tolerance: {record['mip_gap_tolerance']}",
        f"Raw solver log: {record['log_file']}",
        f"JSON solution: {record['solution_file']}",
        "",
        "Independent set (1-based):",
        json.dumps(record["independent_set_one_based"]),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_summaries(output_dir: Path, records: list[dict[str, Any]]) -> None:
    summary_json = output_dir / "summary.json"
    summary_csv = output_dir / "summary.csv"
    summary_json.write_text(json.dumps(records, indent=2, sort_keys=True), encoding="utf-8")

    columns = [
        "problem",
        "instance",
        "instance_id",
        "status",
        "objective_value",
        "cardinality",
        "best_bound",
        "relative_gap",
        "solve_wall_time_sec",
        "solver_runtime_sec",
        "solver_work",
        "nodes_processed",
        "simplex_iterations",
        "solution_count",
        "num_nodes",
        "num_edges",
        "graph_density",
        "num_variables",
        "num_binary_variables",
        "num_constraints",
        "num_nonzeros",
        "num_quadratic_nonzeros",
        "edge_violation_count",
        "time_limit_sec",
        "threads",
        "solver",
        "solver_version",
        "formulation",
        "classical_method",
        "log_file",
        "details_file",
        "solution_file",
    ]
    with summary_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Solve all MIS benchmark instances with Gurobi and write detailed logs."
    )
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=root / "Maximum_Independent_Set" / "mis_benchmark_instances",
        help="Root folder searched recursively for DIMACS .txt or .txt.gz MIS instances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "classical_solutions" / "results" / "mis",
        help="Directory for logs, details, solutions, and summaries.",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional limit for smoke runs.")
    parser.add_argument(
        "--time-limit",
        type=float,
        default=DEFAULT_TIME_LIMIT_SEC,
        help="Per-instance Gurobi time limit in seconds. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=0,
        help="Gurobi thread count. 0 lets Gurobi choose automatically.",
    )
    parser.add_argument(
        "--mip-gap",
        type=float,
        default=0.0,
        help="Relative MIP optimality gap target.",
    )
    parser.add_argument(
        "--mip-focus",
        type=int,
        default=0,
        choices=[0, 1, 2, 3],
        help="Gurobi MIPFocus parameter.",
    )
    parser.add_argument(
        "--display-interval",
        type=int,
        default=5,
        help="Gurobi log display interval in seconds.",
    )
    parser.add_argument(
        "--log-to-console",
        action="store_true",
        help="Also stream Gurobi logs to the console. Logs are always written to files.",
    )
    parser.add_argument(
        "--write-model",
        action="store_true",
        help="Also write each model as an LP file under output-dir/models.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    instances_root = args.instances_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    instances = list_instances(instances_root, args.limit)
    if not instances:
        print(f"No MIS instances found under {instances_root}.")
        return 1

    gp, GRB = import_gurobi()
    records: list[dict[str, Any]] = []
    for index, path in enumerate(instances, start=1):
        print(f"[{index}/{len(instances)}] Solving MIS {path}")
        instance = parse_dimacs_graph(path)
        record = solve_instance(gp, GRB, instance, instances_root, output_dir, args)
        records.append(record)
        print(
            f"  status={record['status']} objective={record['objective_value']} "
            f"gap={record['relative_gap']} time={record['solve_wall_time_sec']:.3f}s"
        )

    write_summaries(output_dir, records)
    print(f"Wrote summary CSV: {output_dir / 'summary.csv'}")
    print(f"Wrote summary JSON: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
