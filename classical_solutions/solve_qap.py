#!/usr/bin/env python3
"""Classical exact baseline solver for Quadratic Assignment instances.

Default input:
  Quadratic_Assignment_Problem/qapdata/**/*.dat

Default output:
  classical_solutions/results/qap/
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


PROBLEM_NAME = "QAP"
FORMULATION = "Exact binary quadratic assignment formulation"
METHOD = "Gurobi global branch-and-bound for binary quadratic assignment"
DEFAULT_TIME_LIMIT_SEC = 100.00


@dataclass(frozen=True)
class QAPInstance:
    n: int
    flow: list[list[int]]
    distance: list[list[int]]
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


def parse_qap_dat_file(path: Path) -> QAPInstance:
    if not path.exists():
        raise FileNotFoundError(f"QAP instance not found: {path}")

    try:
        tokens = [int(tok) for tok in path.read_text(encoding="utf-8").split()]
    except UnicodeDecodeError:
        tokens = [int(tok) for tok in path.read_text(encoding="latin-1").split()]

    if not tokens:
        raise ValueError(f"Empty QAP instance: {path}")

    n = tokens[0]
    needed = 1 + (2 * n * n)
    if len(tokens) < needed:
        raise ValueError(
            f"Invalid QAP file {path}: expected at least {needed} integers, got {len(tokens)}"
        )

    flow_flat = tokens[1 : 1 + n * n]
    distance_flat = tokens[1 + n * n : 1 + 2 * n * n]
    flow = [flow_flat[row * n : (row + 1) * n] for row in range(n)]
    distance = [distance_flat[row * n : (row + 1) * n] for row in range(n)]

    return QAPInstance(n=n, flow=flow, distance=distance, source=path)


def list_instances(root: Path, pattern: str, limit: int | None) -> list[Path]:
    instances = sorted(path for path in root.rglob(pattern) if path.is_file())
    if limit is not None:
        instances = instances[:limit]
    return instances


def instance_id(instance_root: Path, path: Path) -> str:
    try:
        rel = path.relative_to(instance_root)
    except ValueError:
        rel = Path(path.name)
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
    model.Params.NonConvex = 2


def qap_objective(instance: QAPInstance, permutation_zero_based: list[int]) -> int | None:
    if len(permutation_zero_based) != instance.n or any(loc < 0 for loc in permutation_zero_based):
        return None
    total = 0
    for i in range(instance.n):
        for k in range(instance.n):
            total += (
                instance.flow[i][k]
                * instance.distance[permutation_zero_based[i]][permutation_zero_based[k]]
            )
    return total


def build_model(
    gp: Any,
    GRB: Any,
    instance: QAPInstance,
    args: argparse.Namespace,
    log_path: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    if log_path.exists():
        log_path.unlink()
    env = gp.Env(empty=True)
    env.setParam("LogFile", str(log_path))
    env.setParam("LogToConsole", 1 if args.log_to_console else 0)
    env.start()

    n = instance.n
    model = gp.Model(f"QAP_{instance.source.stem}", env=env)
    configure_model(model, args)

    x = model.addVars(n, n, vtype=GRB.BINARY, name="x")
    model.addConstrs((gp.quicksum(x[i, j] for j in range(n)) == 1 for i in range(n)), name="facility")
    model.addConstrs((gp.quicksum(x[i, j] for i in range(n)) == 1 for j in range(n)), name="location")

    objective_terms = []
    nonzero_qap_terms = 0
    for i in range(n):
        for k in range(n):
            flow_value = instance.flow[i][k]
            if flow_value == 0:
                continue
            for j in range(n):
                for ell in range(n):
                    coefficient = flow_value * instance.distance[j][ell]
                    if coefficient == 0:
                        continue
                    nonzero_qap_terms += 1
                    objective_terms.append(coefficient * x[i, j] * x[k, ell])

    model.setObjective(gp.quicksum(objective_terms), GRB.MINIMIZE)
    model.update()

    context = {
        "x": x,
        "env": env,
        "num_assignment_variables": n * n,
        "num_assignment_constraints": 2 * n,
        "num_nonzero_qap_terms": nonzero_qap_terms,
    }
    return model, env, context


def solution_metrics(model: Any, instance: QAPInstance, context: dict[str, Any]) -> dict[str, Any]:
    has_solution = safe_attr(model, "SolCount", 0) > 0
    n = instance.n
    permutation_zero_based = [-1] * n
    assignment_pairs_one_based: list[dict[str, int]] = []
    assignment_matrix: list[list[int]] = [[0] * n for _ in range(n)]
    assignment_violation_count = 0

    if has_solution:
        x = context["x"]
        for i in range(n):
            selected = [j for j in range(n) if float(x[i, j].X) > 0.5]
            if len(selected) == 1:
                j = selected[0]
                permutation_zero_based[i] = j
                assignment_matrix[i][j] = 1
                assignment_pairs_one_based.append({"facility": i + 1, "location": j + 1})
            else:
                assignment_violation_count += 1
        used_locations = [loc for loc in permutation_zero_based if loc >= 0]
        assignment_violation_count += n - len(set(used_locations))

    permutation_one_based = [loc + 1 if loc >= 0 else -1 for loc in permutation_zero_based]
    objective_from_solution = qap_objective(instance, permutation_zero_based)

    return {
        "permutation_zero_based": permutation_zero_based,
        "permutation_one_based": permutation_one_based,
        "assignment_pairs_one_based": assignment_pairs_one_based,
        "assignment_matrix": assignment_matrix,
        "assignment_violation_count": assignment_violation_count,
        "objective_from_solution": objective_from_solution,
    }


def solve_instance(
    gp: Any,
    GRB: Any,
    instance: QAPInstance,
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
        "num_facilities": instance.n,
        "num_locations": instance.n,
        "num_assignment_variables": context["num_assignment_variables"],
        "num_assignment_constraints": context["num_assignment_constraints"],
        "num_nonzero_qap_terms": context["num_nonzero_qap_terms"],
        "flow_sum": sum(sum(row) for row in instance.flow),
        "distance_sum": sum(sum(row) for row in instance.distance),
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
        f"Objective recomputed from permutation: {record['objective_from_solution']}",
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
        f"Facilities/locations: {record['num_facilities']}",
        f"Assignment variables: {record['num_assignment_variables']}",
        f"Assignment constraints: {record['num_assignment_constraints']}",
        f"Nonzero QAP objective terms represented: {record['num_nonzero_qap_terms']}",
        f"Assignment violation count: {record['assignment_violation_count']}",
        f"Time limit (sec): {record['time_limit_sec']}",
        f"Threads: {record['threads']}",
        f"MIP gap tolerance: {record['mip_gap_tolerance']}",
        f"Raw solver log: {record['log_file']}",
        f"JSON solution: {record['solution_file']}",
        "",
        "Permutation, facility -> location (1-based):",
        json.dumps(record["permutation_one_based"]),
        "",
        "Assignment pairs (1-based):",
        json.dumps(record["assignment_pairs_one_based"], indent=2),
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
        "objective_from_solution",
        "best_bound",
        "relative_gap",
        "solve_wall_time_sec",
        "solver_runtime_sec",
        "solver_work",
        "nodes_processed",
        "simplex_iterations",
        "solution_count",
        "num_facilities",
        "num_locations",
        "num_assignment_variables",
        "num_assignment_constraints",
        "num_nonzero_qap_terms",
        "num_variables",
        "num_binary_variables",
        "num_constraints",
        "num_nonzeros",
        "num_quadratic_nonzeros",
        "assignment_violation_count",
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
        description="Solve all QAP benchmark instances with Gurobi and write detailed logs."
    )
    parser.add_argument(
        "--instances-root",
        type=Path,
        default=root / "Quadratic_Assignment_Problem" / "qapdata",
        help="Root folder searched recursively for QAP .dat instances.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "classical_solutions" / "results" / "qap",
        help="Directory for logs, details, solutions, and summaries.",
    )
    parser.add_argument("--pattern", default="*.dat", help="File glob used under --instances-root.")
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

    instances = list_instances(instances_root, args.pattern, args.limit)
    if not instances:
        print(f"No QAP instances found under {instances_root} matching {args.pattern}.")
        return 1

    gp, GRB = import_gurobi()
    records: list[dict[str, Any]] = []
    for index, path in enumerate(instances, start=1):
        print(f"[{index}/{len(instances)}] Solving QAP {path}")
        instance = parse_qap_dat_file(path)
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
