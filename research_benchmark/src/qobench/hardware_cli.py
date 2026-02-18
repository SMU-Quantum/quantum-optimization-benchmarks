from __future__ import annotations

import argparse
import json
import logging
import math
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .hardware_manager import QuantumHardwareManager
from .problem_registry import get_problem
from .quantum_methods import (
    QpuScheduler,
    QuboObjective,
    build_ansatz_bundle,
    run_pce_method,
    run_variational_method,
    serialize_trace,
)
from .serialization import to_jsonable
from .types import ProblemType


LOGGER = logging.getLogger("qobench.hardware_cli")


def _import_qubo_tools() -> tuple[Any, Any]:
    try:
        from qiskit_optimization.converters import QuadraticProgramToQubo
        from qiskit_optimization.translators import from_docplex_mp
    except Exception as exc:
        raise ModuleNotFoundError(
            "qiskit-optimization is required. Install dependencies in your project environment."
        ) from exc
    return from_docplex_mp, QuadraticProgramToQubo


def _resolve_instance_path(project_root: Path, instance: str | None) -> Path | None:
    if not instance:
        return None
    path = Path(instance)
    if path.is_absolute():
        return path
    return (project_root / path).resolve()


def _parse_qpu_list(raw: str | None) -> list[str]:
    if raw is None or raw.strip() == "":
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def _collect_ansatz_metrics(ansatz: Any) -> dict[str, int]:
    one_qubit_gates = 0
    two_qubit_gates = 0
    measurement_ops = 0
    for instruction, qargs, _ in ansatz.qiskit_template.data:
        if instruction.name == "measure":
            measurement_ops += 1
            continue
        qcount = len(qargs)
        if qcount == 1:
            one_qubit_gates += 1
        elif qcount == 2:
            two_qubit_gates += 1

    return {
        "num_qubits": int(ansatz.num_qubits),
        "trainable_parameters": int(ansatz.num_parameters),
        "depth": int(ansatz.qiskit_template.depth()),
        "one_qubit_gates": int(one_qubit_gates),
        "two_qubit_gates": int(two_qubit_gates),
        "measurement_ops": int(measurement_ops),
    }


def _resolve_log_level(level_name: str) -> int:
    level = getattr(logging, str(level_name).upper(), None)
    if not isinstance(level, int):
        raise ValueError(f"Unknown log level '{level_name}'.")
    return level


def _configure_logging(
    *,
    log_level_name: str,
    project_root: Path,
    run_dir: Path,
    explicit_log_file: str | None,
) -> Path:
    if explicit_log_file:
        log_path = Path(explicit_log_file)
        if not log_path.is_absolute():
            log_path = (project_root / log_path).resolve()
    else:
        log_path = run_dir / "run.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    level = _resolve_log_level(log_level_name)
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


def _decode_problem_solution(problem: ProblemType, assignment: dict[str, int]) -> dict[str, Any]:
    active_vars = sorted([name for name, value in assignment.items() if int(value) == 1])
    summary: dict[str, Any] = {"active_variable_count": len(active_vars), "active_variables": active_vars[:200]}

    if problem in (ProblemType.MIS, ProblemType.MKP):
        selected = []
        for name, value in assignment.items():
            if int(value) != 1 or not name.startswith("x_"):
                continue
            token = name.split("_", maxsplit=1)[1]
            if token.isdigit():
                selected.append(int(token))
        selected = sorted(set(selected))
        summary.update(
            {
                "selected_indices_zero_based": selected,
                "selected_indices_one_based": [idx + 1 for idx in selected],
            }
        )
        return summary

    if problem == ProblemType.QAP:
        pairs = []
        for name, value in assignment.items():
            if int(value) != 1 or not name.startswith("x_"):
                continue
            parts = name.split("_")
            if len(parts) == 3 and parts[1].isdigit() and parts[2].isdigit():
                i = int(parts[1])
                j = int(parts[2])
                pairs.append({"facility_zero_based": i, "location_zero_based": j})
        pairs.sort(key=lambda item: (item["facility_zero_based"], item["location_zero_based"]))
        summary["assignment_pairs_zero_based"] = pairs
        return summary

    if problem == ProblemType.MARKET_SHARE:
        retailers_d1 = []
        for name, value in assignment.items():
            if int(value) != 1 or not name.startswith("x_"):
                continue
            token = name.split("_", maxsplit=1)[1]
            if token.isdigit():
                retailers_d1.append(int(token))
        retailers_d1 = sorted(set(retailers_d1))
        summary["retailers_assigned_to_d1_zero_based"] = retailers_d1
        summary["retailers_assigned_to_d1_one_based"] = [idx + 1 for idx in retailers_d1]
        return summary

    return summary


def _bit(assignment: dict[str, int], name: str) -> int:
    try:
        return 1 if int(assignment.get(name, 0)) == 1 else 0
    except Exception:
        return 0


def _reconstruct_problem_objective(
    *,
    problem: ProblemType,
    instance: Any,
    assignment: dict[str, int],
) -> dict[str, Any]:
    if problem == ProblemType.MIS:
        num_nodes = int(getattr(instance, "num_nodes"))
        selected = [_bit(assignment, f"x_{i}") for i in range(num_nodes)]
        cardinality = int(sum(selected))
        violations = 0
        for u, v in getattr(instance, "edges"):
            if selected[int(u)] == 1 and selected[int(v)] == 1:
                violations += 1
        return {
            "objective_label": "maximum_independent_set_cardinality",
            "objective_sense": "maximize",
            "objective_value": float(cardinality),
            "feasible": bool(violations == 0),
            "constraint_violations": {"edge_conflicts": int(violations)},
        }

    if problem == ProblemType.MKP:
        n = int(getattr(instance, "n"))
        m = int(getattr(instance, "m"))
        profits = list(getattr(instance, "profits"))
        weights = list(getattr(instance, "weights"))
        capacities = list(getattr(instance, "capacities"))

        selected = [_bit(assignment, f"x_{i}") for i in range(n)]
        profit = float(sum(int(profits[i]) * selected[i] for i in range(n)))
        used = [
            float(sum(int(weights[j][i]) * selected[i] for i in range(n)))
            for j in range(m)
        ]
        over_capacity = [
            max(0.0, used[j] - float(capacities[j]))
            for j in range(m)
        ]
        return {
            "objective_label": "mkp_total_profit",
            "objective_sense": "maximize",
            "objective_value": profit,
            "feasible": bool(all(v <= 0.0 for v in over_capacity)),
            "constraint_violations": {
                "capacity_overflow_by_dimension": over_capacity,
            },
        }

    if problem == ProblemType.QAP:
        n = int(getattr(instance, "n"))
        flow = list(getattr(instance, "flow"))
        distance = list(getattr(instance, "distance"))

        x = [[_bit(assignment, f"x_{i}_{j}") for j in range(n)] for i in range(n)]
        objective_value = 0.0
        for i in range(n):
            for j in range(n):
                if x[i][j] == 0:
                    continue
                for k in range(n):
                    for l in range(n):
                        if x[k][l] == 0:
                            continue
                        objective_value += float(flow[i][k]) * float(distance[j][l])

        row_sums = [int(sum(x[i][j] for j in range(n))) for i in range(n)]
        col_sums = [int(sum(x[i][j] for i in range(n))) for j in range(n)]
        feasible = all(v == 1 for v in row_sums) and all(v == 1 for v in col_sums)
        return {
            "objective_label": "qap_assignment_cost",
            "objective_sense": "minimize",
            "objective_value": float(objective_value),
            "feasible": bool(feasible),
            "constraint_violations": {
                "row_sums": row_sums,
                "col_sums": col_sums,
            },
        }

    if problem == ProblemType.MARKET_SHARE:
        num_products = int(getattr(instance, "num_products"))
        num_retailers = int(getattr(instance, "num_retailers"))
        demands = list(getattr(instance, "demands"))
        targets = list(getattr(instance, "target_demands"))

        x = [_bit(assignment, f"x_{j}") for j in range(num_retailers)]
        realized = []
        abs_deviation = []
        for i in range(num_products):
            r = float(sum(int(demands[i][j]) * x[j] for j in range(num_retailers)))
            realized.append(r)
            abs_deviation.append(abs(r - float(targets[i])))
        objective_value = float(sum(abs_deviation))
        return {
            "objective_label": "market_share_total_absolute_deviation",
            "objective_sense": "minimize",
            "objective_value": objective_value,
            "feasible": True,
            "constraint_violations": {
                "absolute_deviation_per_product": abs_deviation,
                "realized_demands": realized,
                "target_demands": [float(v) for v in targets],
            },
        }

    return {
        "objective_label": "unknown",
        "objective_sense": "unknown",
        "objective_value": float("nan"),
        "feasible": False,
        "constraint_violations": {},
    }


def _assignment_to_bitstring(
    *,
    variable_names_in_order: list[str],
    assignment: dict[str, int],
) -> str:
    return "".join(
        "1" if _bit(assignment, name) == 1 else "0"
        for name in variable_names_in_order
    )


def _sum_numeric_values(value: Any) -> float:
    if isinstance(value, (int, float)):
        if math.isnan(float(value)):
            return 0.0
        return abs(float(value))
    if isinstance(value, dict):
        return sum(_sum_numeric_values(v) for v in value.values())
    if isinstance(value, list):
        return sum(_sum_numeric_values(v) for v in value)
    return 0.0


def _constraint_violation_score(problem: ProblemType, reconstructed: dict[str, Any]) -> float:
    if bool(reconstructed.get("feasible")):
        return 0.0
    violations = reconstructed.get("constraint_violations", {})
    if problem == ProblemType.MIS:
        return float(violations.get("edge_conflicts", 0.0))
    if problem == ProblemType.MKP:
        overflow = violations.get("capacity_overflow_by_dimension", [])
        return float(sum(max(0.0, float(v)) for v in overflow))
    if problem == ProblemType.QAP:
        row_sums = violations.get("row_sums", [])
        col_sums = violations.get("col_sums", [])
        row_gap = sum(abs(int(v) - 1) for v in row_sums)
        col_gap = sum(abs(int(v) - 1) for v in col_sums)
        return float(row_gap + col_gap)
    return _sum_numeric_values(violations)


def _is_better_reconstructed(
    *,
    problem: ProblemType,
    candidate: dict[str, Any],
    incumbent: dict[str, Any],
) -> bool:
    candidate_feasible = bool(candidate.get("feasible"))
    incumbent_feasible = bool(incumbent.get("feasible"))

    if candidate_feasible and not incumbent_feasible:
        return True
    if incumbent_feasible and not candidate_feasible:
        return False

    if not candidate_feasible and not incumbent_feasible:
        return _constraint_violation_score(problem, candidate) < _constraint_violation_score(problem, incumbent)

    sense = str(candidate.get("objective_sense") or incumbent.get("objective_sense") or "").lower()
    try:
        candidate_value = float(candidate.get("objective_value"))
        incumbent_value = float(incumbent.get("objective_value"))
    except Exception:
        return False

    if not math.isfinite(candidate_value):
        return False
    if not math.isfinite(incumbent_value):
        return True
    if sense == "maximize":
        return candidate_value > incumbent_value
    if sense == "minimize":
        return candidate_value < incumbent_value
    return False


def _qap_permutation_from_assignment(
    *,
    n: int,
    assignment: dict[str, int],
) -> list[int] | None:
    permutation: list[int] = []
    used_locations: set[int] = set()
    for i in range(n):
        active_locations = [j for j in range(n) if _bit(assignment, f"x_{i}_{j}") == 1]
        if len(active_locations) != 1:
            return None
        location = active_locations[0]
        if location in used_locations:
            return None
        used_locations.add(location)
        permutation.append(location)
    return permutation


def _run_one_round_local_swap(
    *,
    problem: ProblemType,
    instance: Any,
    assignment: dict[str, int],
    variable_names_in_order: list[str],
) -> dict[str, Any]:
    base_assignment = dict(assignment)
    best_assignment = dict(base_assignment)
    best_reconstructed = _reconstruct_problem_objective(
        problem=problem,
        instance=instance,
        assignment=best_assignment,
    )
    candidates_checked = 0

    def consider(candidate_assignment: dict[str, int]) -> None:
        nonlocal best_assignment, best_reconstructed, candidates_checked
        candidates_checked += 1
        candidate_reconstructed = _reconstruct_problem_objective(
            problem=problem,
            instance=instance,
            assignment=candidate_assignment,
        )
        if _is_better_reconstructed(
            problem=problem,
            candidate=candidate_reconstructed,
            incumbent=best_reconstructed,
        ):
            best_assignment = dict(candidate_assignment)
            best_reconstructed = candidate_reconstructed

    if problem == ProblemType.MIS:
        num_nodes = int(getattr(instance, "num_nodes"))
        selected = [i for i in range(num_nodes) if _bit(base_assignment, f"x_{i}") == 1]
        unselected = [i for i in range(num_nodes) if _bit(base_assignment, f"x_{i}") == 0]
        for i in range(num_nodes):
            candidate = dict(base_assignment)
            candidate[f"x_{i}"] = 1 - _bit(base_assignment, f"x_{i}")
            consider(candidate)
        for out_idx in selected:
            for in_idx in unselected:
                candidate = dict(base_assignment)
                candidate[f"x_{out_idx}"] = 0
                candidate[f"x_{in_idx}"] = 1
                consider(candidate)
    elif problem == ProblemType.MKP:
        n = int(getattr(instance, "n"))
        selected = [i for i in range(n) if _bit(base_assignment, f"x_{i}") == 1]
        unselected = [i for i in range(n) if _bit(base_assignment, f"x_{i}") == 0]
        for i in range(n):
            candidate = dict(base_assignment)
            candidate[f"x_{i}"] = 1 - _bit(base_assignment, f"x_{i}")
            consider(candidate)
        for out_idx in selected:
            for in_idx in unselected:
                candidate = dict(base_assignment)
                candidate[f"x_{out_idx}"] = 0
                candidate[f"x_{in_idx}"] = 1
                consider(candidate)
    elif problem == ProblemType.QAP:
        n = int(getattr(instance, "n"))
        permutation = _qap_permutation_from_assignment(n=n, assignment=base_assignment)
        if permutation is not None:
            for i in range(n):
                for k in range(i + 1, n):
                    li = permutation[i]
                    lk = permutation[k]
                    candidate = dict(base_assignment)
                    candidate[f"x_{i}_{li}"] = 0
                    candidate[f"x_{k}_{lk}"] = 0
                    candidate[f"x_{i}_{lk}"] = 1
                    candidate[f"x_{k}_{li}"] = 1
                    consider(candidate)
    elif problem == ProblemType.MARKET_SHARE:
        num_retailers = int(getattr(instance, "num_retailers"))
        selected = [j for j in range(num_retailers) if _bit(base_assignment, f"x_{j}") == 1]
        unselected = [j for j in range(num_retailers) if _bit(base_assignment, f"x_{j}") == 0]
        for j in range(num_retailers):
            candidate = dict(base_assignment)
            candidate[f"x_{j}"] = 1 - _bit(base_assignment, f"x_{j}")
            consider(candidate)
        for out_idx in selected:
            for in_idx in unselected:
                candidate = dict(base_assignment)
                candidate[f"x_{out_idx}"] = 0
                candidate[f"x_{in_idx}"] = 1
                consider(candidate)

    improved = best_assignment != base_assignment
    return {
        "improved": improved,
        "candidates_checked": int(candidates_checked),
        "assignment": best_assignment,
        "bitstring": _assignment_to_bitstring(
            variable_names_in_order=variable_names_in_order,
            assignment=best_assignment,
        ),
        "reconstructed": best_reconstructed,
        "base_reconstructed": _reconstruct_problem_objective(
            problem=problem,
            instance=instance,
            assignment=base_assignment,
        ),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="qobench-hardware",
        description=(
            "Run benchmark problems on quantum hardware/backends with VQE, CVaR-VQE, or PCE."
        ),
    )
    parser.add_argument("--problem", required=True, choices=[p.value for p in ProblemType])
    parser.add_argument("--instance", default=None, help="Path to problem instance. If omitted, uses problem default.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--method", required=True, choices=["vqe", "cvar_vqe", "pce"])
    parser.add_argument("--execution-mode", default="single", choices=["single", "multi"])
    parser.add_argument("--qpu-id", default=None, help="QPU id for single mode (e.g., ibm_quantum, rigetti_ankaa3, local_qiskit).")
    parser.add_argument("--qpus", default=None, help="Comma-separated QPU ids to restrict scheduling in multi mode.")
    parser.add_argument(
        "--only-qpu",
        default=None,
        help="Disable all QPUs except this one (strict single-backend filter).",
    )

    parser.add_argument("--layers", type=int, default=1, help="Ansatz entangling depth.")
    parser.add_argument("--entanglement", default="chain", choices=["chain", "full"])
    parser.add_argument("--shots", type=int, default=256)
    parser.add_argument("--maxiter", type=int, default=12)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=0,
        help="Safety cap for QUBO variable count. Set 0 to disable cap.",
    )
    parser.add_argument("--qubo-penalty", type=float, default=None)

    parser.add_argument("--cvar-alpha", type=float, default=0.2, help="Alpha for CVaR objective.")
    parser.add_argument("--pce-population", type=int, default=8)
    parser.add_argument("--pce-elite-frac", type=float, default=0.25)
    parser.add_argument("--pce-parallel-workers", type=int, default=2)
    parser.add_argument(
        "--pce-batch-size",
        type=int,
        default=1,
        help="Number of PCE candidates to submit in one hardware job (single-QPU mode).",
    )

    parser.add_argument("--aws-profile", "--profile", dest="aws_profile", default=None)
    parser.add_argument("--ibm-token", default=None)
    parser.add_argument("--ibm-instance", default=None)
    parser.add_argument(
        "--qiskit-optimization-level",
        type=int,
        default=3,
        choices=[0, 1, 2, 3],
        help="Qiskit transpiler optimization level for IBM backend transpilation.",
    )
    parser.add_argument("--no-aws", action="store_true")
    parser.add_argument("--no-ibm", action="store_true")
    parser.add_argument("--include-simulators", action="store_true", help="Allow simulator backends in scheduling.")
    parser.add_argument(
        "--min-window-minutes",
        type=float,
        default=10.0,
        help="Minimum remaining window required for any time-window-gated backend.",
    )
    parser.add_argument(
        "--min-aws-window-minutes",
        type=float,
        default=30.0,
        help="Minimum remaining window required for AWS hardware QPUs.",
    )
    parser.add_argument(
        "--ibm-min-runtime-seconds",
        type=float,
        default=50.0,
        help="Minimum IBM runtime budget required to schedule jobs.",
    )
    parser.add_argument(
        "--ibm-backend-refresh-seconds",
        type=float,
        default=300.0,
        help="Periodic IBM backend rediscovery interval.",
    )
    parser.add_argument(
        "--queue-status-seconds",
        type=float,
        default=120.0,
        help="How often to print queue/status updates while waiting for a submitted job.",
    )

    parser.add_argument("--model-time-limit", type=float, default=60.0, help="Time limit used only while building model metadata.")
    parser.add_argument("--num-products", type=int, default=2, help="Used only for generated market-share instances.")

    parser.add_argument(
        "--output-root",
        default="research_benchmark/results_hardware",
        help="Output root directory. Results are grouped by problem name.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        type=str.upper,
        help="Logging verbosity for console and run log file.",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Optional explicit log file path. Defaults to <run_dir>/run.log.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    run_start = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    problem_type = ProblemType(args.problem)
    now_utc = datetime.now(timezone.utc)
    run_stamp = now_utc.strftime("%Y%m%dT%H%M%SZ")
    output_root = Path(args.output_root).resolve()
    run_dir = output_root / problem_type.value / f"{problem_type.value}_{args.method}_{run_stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    log_path = _configure_logging(
        log_level_name=str(args.log_level),
        project_root=project_root,
        run_dir=run_dir,
        explicit_log_file=args.log_file,
    )
    LOGGER.info(
        "Run start | problem=%s method=%s mode=%s seed=%s",
        problem_type.value,
        args.method,
        args.execution_mode,
        args.seed,
    )

    instance_path = _resolve_instance_path(project_root, args.instance)
    LOGGER.debug(
        "Resolved paths | project_root=%s requested_instance=%s resolved_instance=%s output_root=%s log_file=%s",
        project_root,
        args.instance,
        instance_path,
        output_root,
        log_path,
    )

    problem = get_problem(problem_type)
    if instance_path is None:
        default_instance = problem.default_instance(project_root)
        if default_instance is not None:
            instance_path = default_instance.resolve()
    LOGGER.info("Instance | path=%s", instance_path)

    instance = problem.load_instance(
        instance_path,
        seed=int(args.seed),
        num_products=int(args.num_products),
    )
    LOGGER.debug("Problem instance loaded | problem=%s", problem_type.value)
    model, _ = problem.build_model(instance=instance, time_limit_sec=float(args.model_time_limit))
    LOGGER.debug("Problem model built | model_time_limit=%.2fs", float(args.model_time_limit))

    from_docplex_mp, QuadraticProgramToQubo = _import_qubo_tools()
    qp = from_docplex_mp(model)
    converter = (
        QuadraticProgramToQubo(penalty=float(args.qubo_penalty))
        if args.qubo_penalty is not None
        else QuadraticProgramToQubo()
    )
    qubo = converter.convert(qp)
    num_qubits = int(qubo.get_num_vars())
    LOGGER.info(
        "Problem size | problem=%s required_qubits=%s",
        problem_type.value,
        num_qubits,
    )
    LOGGER.debug(
        "QUBO details | qubo_penalty=%s max_qubits_cap=%s",
        args.qubo_penalty,
        args.max_qubits,
    )

    if int(args.max_qubits) > 0 and num_qubits > int(args.max_qubits):
        raise ValueError(
            f"QUBO has {num_qubits} variables, larger than --max-qubits={args.max_qubits}. "
            "Use a smaller instance or raise --max-qubits."
        )

    known_qpu_ids = {
        "rigetti_ankaa3",
        "iqm_emerald",
        "iqm_garnet",
        "amazon_sv1",
        "amazon_tn1",
        "ibm_quantum",
        "local_qiskit",
    }
    if args.only_qpu is not None and args.only_qpu not in known_qpu_ids:
        raise ValueError(f"Unknown --only-qpu '{args.only_qpu}'.")
    if args.only_qpu is not None and args.qpu_id is not None and args.qpu_id != args.only_qpu:
        raise ValueError("--qpu-id and --only-qpu conflict. They must match if both are set.")

    only_is_aws = args.only_qpu in {"rigetti_ankaa3", "iqm_emerald", "iqm_garnet", "amazon_sv1", "amazon_tn1"}
    if bool(args.no_aws) and only_is_aws:
        raise ValueError("--only-qpu requests an AWS backend but --no-aws is set.")
    if bool(args.no_ibm) and args.only_qpu == "ibm_quantum":
        raise ValueError("--only-qpu=ibm_quantum but --no-ibm is set.")

    LOGGER.debug(
        "Backend filters | only_qpu=%s no_aws=%s no_ibm=%s include_simulators=%s",
        args.only_qpu,
        bool(args.no_aws),
        bool(args.no_ibm),
        bool(args.include_simulators),
    )

    enabled_qpus = {args.only_qpu} if args.only_qpu is not None else None
    manager = QuantumHardwareManager(
        aws_profile=args.aws_profile,
        ibm_token=args.ibm_token,
        ibm_instance=args.ibm_instance,
        use_aws=not bool(args.no_aws),
        use_ibm=not bool(args.no_ibm),
        allow_simulators=True,
        enabled_qpu_ids=enabled_qpus,
        qiskit_optimization_level=int(args.qiskit_optimization_level),
    )
    manager.min_window_remaining_minutes = float(args.min_window_minutes)
    manager.min_aws_window_remaining_minutes = float(args.min_aws_window_minutes)
    manager.ibm_min_runtime_seconds = float(args.ibm_min_runtime_seconds)
    manager.ibm_backend_refresh_interval = float(args.ibm_backend_refresh_seconds)
    manager.job_status_log_interval = float(args.queue_status_seconds)
    init_status = manager.initialize()
    LOGGER.debug("Hardware initialization status: %s", init_status)
    LOGGER.debug("Hardware status snapshot after init: %s", manager.status_snapshot())

    include_simulators = bool(args.include_simulators)
    if args.only_qpu in {"local_qiskit", "amazon_sv1", "amazon_tn1"}:
        include_simulators = True
    available_qpus = manager.get_available_qpus_for_size(
        num_qubits=num_qubits,
        include_simulators=include_simulators,
    )
    LOGGER.info("Available QPUs | %s", available_qpus)
    snapshot = manager.status_snapshot()
    LOGGER.debug("QPU status snapshot: %s", snapshot)

    if args.execution_mode == "single":
        selected_qpu = args.qpu_id or args.only_qpu or (available_qpus[0] if available_qpus else None)
        if selected_qpu is None:
            raise RuntimeError(
                "No QPU available for this problem size. Try --include-simulators or different credentials."
            )
        if selected_qpu not in manager.qpus:
            raise ValueError(f"Unknown --qpu-id '{selected_qpu}'.")
        qpu_cfg = manager.qpus[selected_qpu]
        if not qpu_cfg.is_available:
            raise RuntimeError(f"QPU '{selected_qpu}' is unavailable: {qpu_cfg.last_error}")
        if qpu_cfg.max_qubits < num_qubits:
            raise RuntimeError(
                f"QPU '{selected_qpu}' supports {qpu_cfg.max_qubits} qubits, needs {num_qubits}."
            )
        scheduler_qpus = [selected_qpu]
    else:
        restricted_qpus = _parse_qpu_list(args.qpus)
        if args.only_qpu is not None:
            if restricted_qpus and restricted_qpus != [args.only_qpu]:
                raise ValueError("--qpus conflicts with --only-qpu. Keep only the same backend.")
            restricted_qpus = [args.only_qpu]

        if restricted_qpus:
            scheduler_qpus = [qpu_id for qpu_id in available_qpus if qpu_id in restricted_qpus]
        else:
            scheduler_qpus = list(available_qpus)
        if not scheduler_qpus:
            raise RuntimeError(
                "No QPUs available for multi mode with current filters. "
                "Try --include-simulators or relax --qpus."
            )

    LOGGER.info(
        "Execution targets | mode=%s qpus=%s",
        args.execution_mode,
        scheduler_qpus,
    )

    scheduler = QpuScheduler(mode=args.execution_mode, qpu_ids=scheduler_qpus)
    ansatz = build_ansatz_bundle(
        num_qubits=num_qubits,
        layers=int(args.layers),
        entanglement=str(args.entanglement),
    )
    ansatz_metrics = _collect_ansatz_metrics(ansatz)
    objective = QuboObjective(qubo=qubo)
    LOGGER.info(
        "Optimization start | method=%s objective=%s shots=%s maxiter=%s",
        args.method,
        "expectation" if args.method == "vqe" else ("cvar" if args.method == "cvar_vqe" else "expectation"),
        args.shots,
        args.maxiter,
    )

    solve_start = time.perf_counter()
    if args.method in ("vqe", "cvar_vqe"):
        if int(args.pce_batch_size) > 1:
            LOGGER.debug("Ignoring --pce-batch-size for method=%s (only used by pce)", args.method)
        optimization = run_variational_method(
            method=args.method,
            manager=manager,
            scheduler=scheduler,
            ansatz=ansatz,
            objective=objective,
            shots=int(args.shots),
            maxiter=int(args.maxiter),
            seed=int(args.seed),
            cvar_alpha=float(args.cvar_alpha),
            timeout_sec=float(args.timeout_sec) if args.timeout_sec is not None else None,
        )
    else:
        optimization = run_pce_method(
            manager=manager,
            scheduler=scheduler,
            ansatz=ansatz,
            objective=objective,
            shots=int(args.shots),
            maxiter=int(args.maxiter),
            seed=int(args.seed),
            population_size=int(args.pce_population),
            elite_frac=float(args.pce_elite_frac),
            timeout_sec=float(args.timeout_sec) if args.timeout_sec is not None else None,
            parallel_workers=int(args.pce_parallel_workers),
            batch_size=int(args.pce_batch_size),
        )
    solve_runtime_sec = float(time.perf_counter() - solve_start)
    LOGGER.info(
        "Optimization finished | evaluations=%s status=%s",
        optimization.total_evaluations,
        optimization.optimizer_status,
    )

    variable_names_in_order = [
        var_name
        for var_name, _ in sorted(
            objective.qubo.variables_index.items(),
            key=lambda item: int(item[1]),
        )
    ]

    raw_assignment = objective.assignment(optimization.best_bitstring)
    raw_decoded = _decode_problem_solution(problem_type, raw_assignment)
    raw_reconstructed = _reconstruct_problem_objective(
        problem=problem_type,
        instance=instance,
        assignment=raw_assignment,
    )
    local_swap = _run_one_round_local_swap(
        problem=problem_type,
        instance=instance,
        assignment=raw_assignment,
        variable_names_in_order=variable_names_in_order,
    )
    assignment = dict(local_swap["assignment"])
    decoded = _decode_problem_solution(problem_type, assignment)
    reconstructed = dict(local_swap["reconstructed"])
    final_bitstring = str(local_swap["bitstring"])
    postprocess_improved = bool(local_swap["improved"])

    LOGGER.info(
        "Local swap postprocess | improved=%s candidates_checked=%s",
        postprocess_improved,
        local_swap["candidates_checked"],
    )
    LOGGER.info(
        "Final solution | bitstring=%s objective_label=%s objective_value=%s feasible=%s",
        final_bitstring,
        reconstructed.get("objective_label"),
        reconstructed.get("objective_value"),
        reconstructed.get("feasible"),
    )

    (run_dir / "qubo.lp").write_text(qubo.export_as_lp_string(), encoding="utf-8")
    (run_dir / "trace.jsonl").write_text(serialize_trace(optimization.trace), encoding="utf-8")
    (run_dir / "best_counts.json").write_text(
        json.dumps(to_jsonable(optimization.best_counts), indent=2),
        encoding="utf-8",
    )

    result_payload = {
        "run_timestamp_utc": run_stamp,
        "run_directory": str(run_dir),
        "log_file": str(log_path),
        "problem": problem_type.value,
        "instance_path": str(instance_path) if instance_path is not None else None,
        "method": args.method,
        "execution_mode": args.execution_mode,
        "num_qubits": num_qubits,
        "selected_qpus": scheduler_qpus,
        "timing": {
            "solve_runtime_sec": solve_runtime_sec,
            "total_runtime_sec": float(time.perf_counter() - run_start),
        },
        "ansatz_metrics": ansatz_metrics,
        "init_status": init_status,
        "qpu_status_snapshot": manager.status_snapshot(),
        "optimizer": {
            "status": optimization.optimizer_status,
            "message": optimization.optimizer_message,
            "total_evaluations": optimization.total_evaluations,
            "qpu_usage": optimization.qpu_usage,
        },
        "best_result": {
            "optimization_objective_mode": optimization.objective_mode,
            "optimization_objective_value": optimization.best_value,
            "objective_value": reconstructed.get("objective_value"),
            "best_sample_energy": optimization.best_energy,
            "raw_best_bitstring": optimization.best_bitstring,
            "best_bitstring": final_bitstring,
            "best_theta": optimization.best_theta,
            "raw_decoded_solution": raw_decoded,
            "raw_reconstructed_problem_objective": raw_reconstructed,
            "postprocess": {
                "name": "one_round_local_swap",
                "improved": postprocess_improved,
                "candidates_checked": local_swap["candidates_checked"],
            },
            "decoded_solution": decoded,
            "reconstructed_problem_objective": reconstructed,
            "active_assignment": assignment,
        },
        "config": vars(args),
    }

    (run_dir / "result.json").write_text(
        json.dumps(to_jsonable(result_payload), indent=2),
        encoding="utf-8",
    )
    LOGGER.info(
        "Artifacts | result=%s trace=%s qubo=%s counts=%s",
        run_dir / "result.json",
        run_dir / "trace.jsonl",
        run_dir / "qubo.lp",
        run_dir / "best_counts.json",
    )
    total_runtime_sec = float(time.perf_counter() - run_start)

    print(f"Run directory: {run_dir}")
    print(f"Log file: {log_path}")
    print(f"Problem: {problem_type.value}")
    print(f"Method: {args.method}")
    print(f"QUBO variables: {num_qubits}")
    print(f"QPU mode: {args.execution_mode}")
    print(f"QPUs used: {scheduler_qpus}")
    print(f"Total solve time (s): {solve_runtime_sec:.3f}")
    print(f"Total runtime (s): {total_runtime_sec:.3f}")
    print(f"Circuit qubits: {ansatz_metrics['num_qubits']}")
    print(f"Trainable parameters: {ansatz_metrics['trainable_parameters']}")
    print(f"Circuit depth: {ansatz_metrics['depth']}")
    print(f"One-qubit gates: {ansatz_metrics['one_qubit_gates']}")
    print(f"Two-qubit gates: {ansatz_metrics['two_qubit_gates']}")
    reconstructed_value = reconstructed.get("objective_value")
    try:
        reconstructed_value_str = f"{float(reconstructed_value):.6f}"
    except Exception:
        reconstructed_value_str = str(reconstructed_value)
    print(
        f"Reconstructed objective ({reconstructed.get('objective_label')}): {reconstructed_value_str}"
    )
    print(f"Reconstructed feasible: {reconstructed.get('feasible')}")
    print(f"Final bitstring: {final_bitstring}")
    print(
        "Local swap postprocess: "
        f"improved={postprocess_improved} candidates_checked={local_swap['candidates_checked']}"
    )
    print(f"Artifacts: {run_dir / 'result.json'}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
