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
    PceEncoding,
    QpuScheduler,
    QuboObjective,
    build_algorithm_ansatz_bundle,
    estimate_pce_num_qubits,
    estimate_qrao_num_qubits,
    run_pce_method,
    run_qrao_method,
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


def _collect_ansatz_metrics(ansatz: Any) -> dict[str, Any]:
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
        "ansatz_family": str(getattr(ansatz, "ansatz_family", "custom")),
        "ansatz_reps": getattr(ansatz, "ansatz_reps", None),
        "ansatz_entanglement": getattr(ansatz, "ansatz_entanglement", None),
        "ansatz_metadata": getattr(ansatz, "metadata", None) or {},
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
    # Suppress noisy Qiskit transpiler pass-by-pass output
    for noisy_logger in (
        "qiskit.transpiler.passes",
        "qiskit.transpiler.runningpassmanager",
        "qiskit.transpiler.passmanager",
        "qiskit.compiler.transpiler",
        "qiskit.passmanager",
        "qiskit.passmanager.base_tasks",
        "qiskit.passmanager.flow_controllers",
        "qiskit.passmanager.passmanager",
        "qiskit_ibm_runtime.base_primitive",
        "management.get",
    ):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)
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
            "Run benchmark problems on quantum hardware/backends with VQE/QAOA-family methods, PCE, or QRAO."
        ),
    )
    parser.add_argument("--problem", required=True, choices=[p.value for p in ProblemType])
    parser.add_argument("--instance", default=None, help="Path to problem instance. If omitted, uses problem default.")
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--method",
        required=True,
        choices=[
            "vqe",
            "cvar_vqe",
            "qaoa",
            "cvar_qaoa",
            "ws_qaoa",
            "ma_qaoa",
            "pce",
            "qrao",
        ],
    )
    parser.add_argument("--execution-mode", default="single", choices=["single", "multi"])
    parser.add_argument("--qpu-id", default=None, help="QPU id for single mode (e.g., ibm_quantum, rigetti_ankaa3, local_qiskit).")
    parser.add_argument("--qpus", default=None, help="Comma-separated QPU ids to restrict scheduling in multi mode.")
    parser.add_argument(
        "--only-qpu",
        default=None,
        help="Disable all QPUs except this one (strict single-backend filter).",
    )

    parser.add_argument("--layers", type=int, default=3, help="Ansatz entangling depth (reps).")
    parser.add_argument("--entanglement", default="circular", choices=["chain", "circular", "full"])
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--timeout-sec", type=float, default=None)
    parser.add_argument(
        "--max-qubits",
        type=int,
        default=0,
        help="Safety cap for QUBO variable count. Set 0 to disable cap.",
    )
    parser.add_argument("--qubo-penalty", type=float, default=None)

    parser.add_argument("--cvar-alpha", type=float, default=0.25, help="Alpha for CVaR objective.")
    parser.add_argument(
        "--ws-epsilon",
        type=float,
        default=1e-3,
        help="Warm-start clipping epsilon for WS-QAOA continuous relaxation values.",
    )
    parser.add_argument("--pce-population", type=int, default=1)
    parser.add_argument("--pce-elite-frac", type=float, default=0.25)
    parser.add_argument("--pce-parallel-workers", type=int, default=2)
    parser.add_argument(
        "--pce-compression-k",
        type=int,
        default=2,
        help="PCE only: compression factor k used in Pauli-correlation encoding (default: 2).",
    )
    parser.add_argument(
        "--pce-depth",
        type=int,
        default=0,
        help="PCE only: Brickwork ansatz depth. Use 0 for auto depth=2*encoded_qubits.",
    )
    parser.add_argument(
        "--pce-batch-size",
        type=int,
        default=1,
        help="Number of PCE candidates to submit in one hardware job (single-QPU mode).",
    )
    parser.add_argument(
        "--qrao-max-vars-per-qubit",
        type=int,
        default=3,
        help="QRAO only: maximum binary variables encoded per qubit.",
    )
    parser.add_argument(
        "--qrao-reps",
        type=int,
        default=2,
        help="QRAO only: EfficientSU2 repetition depth for the min-eigen solver ansatz.",
    )
    parser.add_argument(
        "--qrao-rounding",
        type=str.lower,
        choices=["magic", "semideterministic"],
        default="magic",
        help="QRAO only: rounding scheme used to decode compressed solutions.",
    )
    parser.add_argument(
        "--qrao-optimizer",
        type=str.lower,
        choices=["cobyla", "powell", "slsqp", "spsa"],
        default="cobyla",
        help="QRAO only: classical optimizer for the VQE inner loop.",
    )

    parser.add_argument("--aws-profile", "--profile", dest="aws_profile", default=None)
    parser.add_argument("--ibm-token", default=None)
    parser.add_argument("--ibm-instance", default=None)
    parser.add_argument(
        "--ibm-credentials-json",
        default=None,
        help=(
            "Path to a JSON file containing rotating IBM credentials (token + instance/CRN). "
            "When runtime budget is low, the file is reloaded and the next credential is used. "
            "Defaults to research_benchmark/examples/ibm_credentials.example.json if not provided."
        ),
    )
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
        default=20.0,
        help="Minimum remaining window required for AWS hardware QPUs.",
    )
    parser.add_argument(
        "--ibm-min-runtime-seconds",
        type=float,
        default=15.0,
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
        default=180.0,
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
        "--checkpoint-dir",
        default="checkpoints",
        help="Directory for per-problem checkpoint JSONs that track completed instances.",
    )
    parser.add_argument(
        "--force-rerun",
        action="store_true",
        help="Ignore checkpoints and re-run all instances.",
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


def _load_checkpoint(checkpoint_path: Path) -> dict[str, Any]:
    """Load checkpoint JSON for a problem type.

    Returns a dict mapping instance_name -> result summary for completed instances.
    """
    if checkpoint_path.is_file():
        try:
            data = json.loads(checkpoint_path.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            LOGGER.warning("Failed to load checkpoint %s: %s", checkpoint_path, exc)
    return {}


def _save_checkpoint(
    checkpoint_path: Path,
    checkpoint: dict[str, Any],
) -> None:
    """Save checkpoint JSON atomically."""
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = checkpoint_path.with_suffix(".tmp")
    tmp_path.write_text(
        json.dumps(to_jsonable(checkpoint), indent=2),
        encoding="utf-8",
    )
    tmp_path.replace(checkpoint_path)


def _solve_single_instance(
    *,
    args: argparse.Namespace,
    problem: Any,
    problem_type: ProblemType,
    instance_path: Path,
    manager: Any,
    scheduler_qpus: list[str],
    run_dir: Path,
    run_stamp: str,
    log_path: Path,
    include_simulators: bool,
) -> dict[str, Any]:
    """Solve a single instance and return the result summary."""
    instance_name = instance_path.name if instance_path else "unknown"
    instance_start = time.perf_counter()

    LOGGER.info(
        "\n" + "=" * 70 + "\n  Instance: %s\n" + "=" * 70,
        instance_name,
    )

    # --- Detect generated instances (Market Share parameterised grid) ---
    from .problems.market_share import MarketShareProblem
    _actual_instance_path: Path | None = instance_path
    _load_seed = int(args.seed)
    _load_num_products = int(args.num_products)

    if isinstance(problem, MarketShareProblem) and instance_path.suffix == ".gen":
        parsed = MarketShareProblem.parse_generated_instance_name(instance_path.name)
        if parsed is not None:
            _load_seed, _load_num_products = parsed
        _actual_instance_path = None  # signal generation, not file loading

    instance = problem.load_instance(
        _actual_instance_path,
        seed=_load_seed,
        num_products=_load_num_products,
    )
    model, _ = problem.build_model(instance=instance, time_limit_sec=float(args.model_time_limit))

    from_docplex_mp, QuadraticProgramToQubo = _import_qubo_tools()
    qp = from_docplex_mp(model)
    converter = (
        QuadraticProgramToQubo(penalty=float(args.qubo_penalty))
        if args.qubo_penalty is not None
        else QuadraticProgramToQubo()
    )
    qubo = converter.convert(qp)
    logical_num_qubits = int(qubo.get_num_vars())
    execution_num_qubits = logical_num_qubits
    if args.method == "qrao":
        execution_num_qubits = int(
            estimate_qrao_num_qubits(
                qubo=qubo,
                max_vars_per_qubit=int(args.qrao_max_vars_per_qubit),
            )
        )
    elif args.method == "pce":
        execution_num_qubits = int(
            estimate_pce_num_qubits(
                num_variables=logical_num_qubits,
                compression_k=int(args.pce_compression_k),
            )
        )
    if args.method in {"qrao", "pce"}:
        LOGGER.info(
            "  Problem: %s | Instance: %s | Logical Qubits: %d | Execution Qubits: %d",
            problem_type.value,
            instance_name,
            logical_num_qubits,
            execution_num_qubits,
        )
    else:
        LOGGER.info(
            "  Problem: %s | Instance: %s | Qubits: %d",
            problem_type.value,
            instance_name,
            logical_num_qubits,
        )
    if args.method == "qrao":
        LOGGER.info(
            "  QRAO encoding | original_vars=%d encoded_qubits=%d max_vars_per_qubit=%d",
            logical_num_qubits,
            execution_num_qubits,
            int(args.qrao_max_vars_per_qubit),
        )
    elif args.method == "pce":
        LOGGER.info(
            "  PCE encoding | original_vars=%d encoded_qubits=%d compression_k=%d",
            logical_num_qubits,
            execution_num_qubits,
            int(args.pce_compression_k),
        )

    if int(args.max_qubits) > 0 and execution_num_qubits > int(args.max_qubits):
        LOGGER.warning(
            "  SKIPPED: %s requires %d qubits, exceeds --max-qubits=%s",
            instance_name,
            execution_num_qubits,
            args.max_qubits,
        )
        return {
            "instance": instance_name,
            "qubits": execution_num_qubits,
            "status": "SKIPPED",
            "reason": f"Too many qubits ({execution_num_qubits})",
        }

    # Re-check QPU availability for this problem size
    available_qpus = manager.get_available_qpus_for_size(
        num_qubits=execution_num_qubits,
        include_simulators=include_simulators,
    )
    # Use pre-selected QPUs but filter for this size
    usable_qpus = [q for q in scheduler_qpus if q in available_qpus]
    if not usable_qpus:
        # Try with simulators as fallback
        available_with_sims = manager.get_available_qpus_for_size(
            num_qubits=execution_num_qubits, include_simulators=True,
        )
        usable_qpus = [q for q in available_with_sims if q in scheduler_qpus] or list(available_with_sims)

    if not usable_qpus:
        LOGGER.warning(
            "  SKIPPED: No QPU available for %d-qubit instance %s",
            execution_num_qubits,
            instance_name,
        )
        return {
            "instance": instance_name,
            "qubits": execution_num_qubits,
            "status": "SKIPPED", "reason": "No QPU available",
        }

    scheduler = QpuScheduler(
        mode=args.execution_mode,
        qpu_ids=usable_qpus,
        manager=manager,
        num_qubits=execution_num_qubits,
    )
    objective = QuboObjective(qubo=qubo)
    ansatz = None
    ansatz_metrics: dict[str, Any] = {}
    if args.method != "qrao":
        ansatz = build_algorithm_ansatz_bundle(
            method=args.method,
            qubo=qubo,
            layers=int(args.layers),
            entanglement=str(args.entanglement),
            qp=qp,
            ws_epsilon=float(args.ws_epsilon),
            pce_compression_k=int(args.pce_compression_k),
            pce_depth=int(args.pce_depth),
        )
        ansatz_metrics = _collect_ansatz_metrics(ansatz)
        LOGGER.info(
            "  Ansatz | family=%s reps=%s entanglement=%s trainable_parameters=%s depth=%s 2q_gates=%s",
            ansatz_metrics.get("ansatz_family"),
            ansatz_metrics.get("ansatz_reps"),
            ansatz_metrics.get("ansatz_entanglement"),
            ansatz_metrics.get("trainable_parameters"),
            ansatz_metrics.get("depth"),
            ansatz_metrics.get("two_qubit_gates"),
        )
        if args.method == "ws_qaoa":
            warm_start = ansatz_metrics.get("ansatz_metadata", {}).get("warm_start", {})
            LOGGER.info(
                "  Warm-start | enabled=%s source=%s epsilon=%s initial_params=gamma=pi,beta=pi/2 angle_min=%.4f angle_max=%.4f angle_mean=%.4f",
                bool(warm_start.get("enabled", False)),
                warm_start.get("source", "unknown"),
                args.ws_epsilon,
                float(warm_start.get("angle_min", float("nan"))),
                float(warm_start.get("angle_max", float("nan"))),
                float(warm_start.get("angle_mean", float("nan"))),
            )
        if args.method == "pce":
            pce_meta = ansatz_metrics.get("ansatz_metadata", {}).get("pce_encoding", {})
            LOGGER.info(
                "  PCE setup | k=%s logical_vars=%s encoded_qubits=%s pauli_strings=%s brickwork_depth=%s",
                pce_meta.get("compression_k", args.pce_compression_k),
                pce_meta.get("logical_num_vars", logical_num_qubits),
                pce_meta.get("encoded_num_qubits", execution_num_qubits),
                len(pce_meta.get("pauli_strings", [])),
                ansatz_metrics.get("ansatz_reps"),
            )
    else:
        ansatz_metrics = {
            "ansatz_family": "EfficientSU2",
            "ansatz_reps": int(args.qrao_reps),
            "ansatz_entanglement": "linear",
            "ansatz_metadata": {},
            "num_qubits": int(execution_num_qubits),
            "trainable_parameters": None,
            "depth": None,
            "one_qubit_gates": None,
            "two_qubit_gates": None,
            "measurement_ops": None,
        }
        try:
            from qiskit.circuit.library import EfficientSU2

            qrao_preview = EfficientSU2(
                num_qubits=int(execution_num_qubits),
                entanglement="linear",
                reps=int(args.qrao_reps),
            ).decompose()
            one_qubit_gates = 0
            two_qubit_gates = 0
            for instruction, qargs, _ in qrao_preview.data:
                if instruction.name == "measure":
                    continue
                if len(qargs) == 1:
                    one_qubit_gates += 1
                elif len(qargs) == 2:
                    two_qubit_gates += 1
            ansatz_metrics.update(
                {
                    "trainable_parameters": int(qrao_preview.num_parameters),
                    "depth": int(qrao_preview.depth()),
                    "one_qubit_gates": int(one_qubit_gates),
                    "two_qubit_gates": int(two_qubit_gates),
                    "measurement_ops": 0,
                }
            )
        except Exception:
            pass
        LOGGER.info(
            "  Ansatz | family=%s reps=%s entanglement=%s trainable_parameters=%s depth=%s 2q_gates=%s",
            ansatz_metrics.get("ansatz_family"),
            ansatz_metrics.get("ansatz_reps"),
            ansatz_metrics.get("ansatz_entanglement"),
            ansatz_metrics.get("trainable_parameters"),
            ansatz_metrics.get("depth"),
            ansatz_metrics.get("two_qubit_gates"),
        )

    method_objective_label = "expectation"
    if args.method in {"cvar_vqe", "cvar_qaoa"}:
        method_objective_label = "cvar"
    elif args.method == "qrao":
        method_objective_label = "qrao"

    LOGGER.info(
        "  Optimization | method=%s objective=%s shots=%s maxiter=%s qpus=%s",
        args.method,
        method_objective_label,
        args.shots, args.maxiter, usable_qpus,
    )
    LOGGER.info(
        "  Protocol | shots=%s maxiter=%s timeout=%s repair=one_round_local_swap stopping=fixed_iterations",
        args.shots, args.maxiter,
        f"{args.timeout_sec}s" if args.timeout_sec is not None else "none",
    )

    # Create instance-specific output dir
    inst_dir = run_dir / instance_name.replace(".", "_")
    inst_dir.mkdir(parents=True, exist_ok=True)

    solve_start = time.perf_counter()
    try:
        if args.method in ("vqe", "cvar_vqe", "qaoa", "cvar_qaoa", "ws_qaoa", "ma_qaoa"):
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
        elif args.method == "pce":
            pce_encoding: PceEncoding | None = None
            pce_meta = ansatz_metrics.get("ansatz_metadata", {}).get("pce_encoding", {})
            if isinstance(pce_meta, dict):
                try:
                    pce_encoding = PceEncoding(
                        logical_num_vars=int(pce_meta.get("logical_num_vars", logical_num_qubits)),
                        encoded_num_qubits=int(pce_meta.get("encoded_num_qubits", execution_num_qubits)),
                        compression_k=int(pce_meta.get("compression_k", args.pce_compression_k)),
                        pauli_strings=[str(v) for v in list(pce_meta.get("pauli_strings", []))],
                    )
                except Exception:
                    pce_encoding = None
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
                pce_encoding=pce_encoding,
            )
        elif args.method == "qrao":
            optimization = run_qrao_method(
                manager=manager,
                scheduler=scheduler,
                qubo=qubo,
                shots=int(args.shots),
                maxiter=int(args.maxiter),
                seed=int(args.seed),
                max_vars_per_qubit=int(args.qrao_max_vars_per_qubit),
                reps=int(args.qrao_reps),
                rounding_scheme=str(args.qrao_rounding),
                optimizer_name=str(args.qrao_optimizer),
            )
        else:
            raise ValueError(f"Unsupported method '{args.method}'")
    except Exception as exc:
        solve_runtime_sec = float(time.perf_counter() - solve_start)
        LOGGER.warning("  FAILED: %s after %.1fs - %s", instance_name, solve_runtime_sec, exc)
        return {
            "instance": instance_name,
            "qubits": execution_num_qubits,
            "status": "FAILED", "reason": str(exc)[:200],
            "time_sec": solve_runtime_sec,
        }

    solve_runtime_sec = float(time.perf_counter() - solve_start)
    LOGGER.info(
        "  Optimization finished | evaluations=%s status=%s time=%.1fs",
        optimization.total_evaluations, optimization.optimizer_status, solve_runtime_sec,
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

    obj_value = reconstructed.get("objective_value")
    feasible = reconstructed.get("feasible")

    LOGGER.info(
        "  Result | objective=%s feasible=%s postprocess_improved=%s",
        obj_value, feasible, postprocess_improved,
    )

    # --- Collect all job metadata from trace ---
    all_job_metadata: list[dict[str, Any]] = []
    all_job_ids: list[str] = []
    for trace_entry in optimization.trace:
        meta = trace_entry.get("metadata")
        if meta:
            all_job_metadata.append(meta)
            jid = meta.get("job_id") or meta.get("task_id")
            if jid:
                all_job_ids.append(str(jid))

    # --- Calibration snapshot ---
    try:
        calibration = manager.get_calibration_snapshot()
    except Exception:
        calibration = {}

    # --- Benchmark protocol definition ---
    benchmark_protocol = {
        "budget": {
            "shots_per_circuit": int(args.shots),
            "max_optimizer_iterations": int(args.maxiter),
            "total_circuit_evaluations": optimization.total_evaluations,
            "wall_clock_cap_sec": float(args.timeout_sec) if args.timeout_sec is not None else None,
            "queue_time_policy": "included_in_wall_clock",
        },
        "objective_measured": {
            "optimizer_objective": optimization.objective_mode,
            "cvar_alpha": (
                float(args.cvar_alpha)
                if args.method in {"cvar_vqe", "cvar_qaoa"}
                else None
            ),
            "reported_best_sample_energy": optimization.best_energy,
            "reported_expected_value": optimization.best_value,
            "post_processed_objective": obj_value,
            "note": (
                "optimizer_objective is used to guide parameter updates; "
                "best_sample_energy is the lowest QUBO energy from any single bitstring; "
                "post_processed_objective is after local-swap repair"
            ),
        },
        "feasibility_policy": {
            "repair_allowed": True,
            "repair_method": "one_round_local_swap",
            "repair_description": (
                "After optimization, the best raw bitstring is taken. "
                "A single round of local bit-flip swap is applied: "
                "each variable is flipped and the resulting QUBO energy is checked. "
                "If any flip improves the objective, the best flip is kept."
            ),
            "repair_applied": postprocess_improved,
            "candidates_checked": local_swap["candidates_checked"],
            "raw_objective_before_repair": raw_reconstructed.get("objective_value"),
            "raw_feasible_before_repair": raw_reconstructed.get("feasible"),
        },
        "stopping_rule": {
            "type": "fixed_iterations",
            "max_iterations": int(args.maxiter),
            "actual_iterations": optimization.total_evaluations,
            "optimizer_status": optimization.optimizer_status,
            "optimizer_message": optimization.optimizer_message,
        },
    }

    # --- Save artifacts ---
    (inst_dir / "qubo.lp").write_text(qubo.export_as_lp_string(), encoding="utf-8")
    (inst_dir / "trace.jsonl").write_text(serialize_trace(optimization.trace), encoding="utf-8")
    (inst_dir / "best_counts.json").write_text(
        json.dumps(to_jsonable(optimization.best_counts), indent=2),
        encoding="utf-8",
    )

    # --- Build comprehensive result payload ---
    result_payload = {
        "schema_version": "2.0",
        "run_timestamp_utc": run_stamp,
        "run_directory": str(inst_dir),
        "log_file": str(log_path),
        "problem": problem_type.value,
        "instance_path": str(instance_path) if instance_path is not None else None,
        "instance_name": instance_name,

        # --- Benchmark Protocol ---
        "benchmark_protocol": benchmark_protocol,

        # --- Device & Execution Info ---
        "execution": {
            "method": args.method,
            "execution_mode": args.execution_mode,
            "selected_qpus": usable_qpus,
            "job_ids": all_job_ids,
            "total_jobs_dispatched": len(all_job_metadata),
            "seed": int(args.seed),
            "qrao": (
                {
                    "max_vars_per_qubit": int(args.qrao_max_vars_per_qubit),
                    "rounding": str(args.qrao_rounding),
                    "optimizer": str(args.qrao_optimizer),
                }
                if args.method == "qrao"
                else None
            ),
            "pce": (
                {
                    "compression_k": int(args.pce_compression_k),
                    "brickwork_depth": int(args.pce_depth) if int(args.pce_depth) > 0 else None,
                }
                if args.method == "pce"
                else None
            ),
        },

        # --- Circuit Metrics (pre-transpilation) ---
        "circuit_metrics": {
            "num_qubits": execution_num_qubits,
            "logical_qubits_before_encoding": logical_num_qubits,
            "ansatz_family": ansatz_metrics.get("ansatz_family"),
            "ansatz_reps": ansatz_metrics.get("ansatz_reps"),
            "ansatz_entanglement": ansatz_metrics.get("ansatz_entanglement"),
            "trainable_parameters": ansatz_metrics.get("trainable_parameters"),
            "depth_pretranspile": ansatz_metrics.get("depth"),
            "one_qubit_gates_pretranspile": ansatz_metrics.get("one_qubit_gates"),
            "two_qubit_gates_pretranspile": ansatz_metrics.get("two_qubit_gates"),
            "measurement_count": ansatz_metrics.get("measurement_ops"),
        },

        # --- Transpiler Settings ---
        "transpiler_settings": {
            "qiskit_optimization_level": int(args.qiskit_optimization_level),
            "routing_method": "preset_passmanager_default",
            "note": "transpiled circuit metrics available per-job in job_metadata",
        },

        # --- Device Calibration Snapshot ---
        "device_calibration": calibration,

        # --- Timing ---
        "timing": {
            "solve_runtime_sec": solve_runtime_sec,
            "total_instance_sec": float(time.perf_counter() - instance_start),
        },

        # --- QPU Status ---
        "qpu_status_snapshot": manager.status_snapshot(),

        # --- Optimizer Result ---
        "optimizer": {
            "status": optimization.optimizer_status,
            "message": optimization.optimizer_message,
            "total_evaluations": optimization.total_evaluations,
            "qpu_usage": optimization.qpu_usage,
        },

        # --- Per-job Metadata ---
        "job_metadata": all_job_metadata,

        # --- Best Result ---
        "best_result": {
            "optimization_objective_mode": optimization.objective_mode,
            "optimization_objective_value": optimization.best_value,
            "objective_value": obj_value,
            "feasible": feasible,
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

        # --- Raw Bitstring Counts ---
        "raw_bitstring_counts": optimization.best_counts,

        # --- Full Config ---
        "config": vars(args),
    }

    (inst_dir / "result.json").write_text(
        json.dumps(to_jsonable(result_payload), indent=2),
        encoding="utf-8",
    )

    return {
        "instance": instance_name,
        "qubits": execution_num_qubits,
        "status": "OK",
        "objective": obj_value,
        "feasible": feasible,
        "time_sec": solve_runtime_sec,
        "qpus": usable_qpus,
        "result_dir": str(inst_dir),
    }


def run(args: argparse.Namespace) -> int:
    run_start = time.perf_counter()
    project_root = Path(args.project_root).resolve()
    problem_type = ProblemType(args.problem)
    if args.method == "qrao":
        if args.execution_mode != "single":
            raise ValueError("--method=qrao currently requires --execution-mode single.")
        if int(args.qrao_max_vars_per_qubit) < 1:
            raise ValueError("--qrao-max-vars-per-qubit must be >= 1.")
    if args.method == "pce":
        if int(args.pce_compression_k) < 1:
            raise ValueError("--pce-compression-k must be >= 1.")
        if int(args.pce_depth) < 0:
            raise ValueError("--pce-depth must be >= 0.")
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

    problem = get_problem(problem_type)

    # --- Discover instances ---
    if args.instance is not None:
        # Single instance specified via CLI
        instance_path = _resolve_instance_path(project_root, args.instance)
        if instance_path is None:
            raise FileNotFoundError(f"Instance not found: {args.instance}")
        instance_paths = [instance_path]
    else:
        # Discover ALL instances from the problem's default folder
        instance_paths = problem.list_instances(project_root)
        if not instance_paths:
            # Fallback to default single instance
            default = problem.default_instance(project_root)
            if default is not None:
                instance_paths = [default.resolve()]
            else:
                raise FileNotFoundError(
                    f"No instances found for {problem_type.value}. "
                    "Specify --instance or check project_root."
                )

    LOGGER.info(
        "Instances discovered | problem=%s count=%d",
        problem_type.value, len(instance_paths),
    )
    for ip in instance_paths:
        LOGGER.info("  - %s", ip.name)

    # --- Initialize hardware (once for all instances) ---
    known_qpu_ids = {
        "rigetti_ankaa3",
        "iqm_emerald",
        "iqm_garnet",
        "amazon_sv1",
        "ibm_quantum",
        "local_qiskit",
    }
    if args.only_qpu is not None and args.only_qpu not in known_qpu_ids:
        raise ValueError(f"Unknown --only-qpu '{args.only_qpu}'.")
    if args.only_qpu is not None and args.qpu_id is not None and args.qpu_id != args.only_qpu:
        raise ValueError("--qpu-id and --only-qpu conflict. They must match if both are set.")

    only_is_aws = args.only_qpu in {"rigetti_ankaa3", "iqm_emerald", "iqm_garnet", "amazon_sv1"}
    if bool(args.no_aws) and only_is_aws:
        raise ValueError("--only-qpu requests an AWS backend but --no-aws is set.")
    if bool(args.no_ibm) and args.only_qpu == "ibm_quantum":
        raise ValueError("--only-qpu=ibm_quantum but --no-ibm is set.")

    include_simulators = bool(args.include_simulators)
    if args.only_qpu in {"local_qiskit", "amazon_sv1"}:
        include_simulators = True

    enabled_qpus = {args.only_qpu} if args.only_qpu is not None else None

    # Default IBM credentials JSON to the example file if not explicitly provided
    ibm_creds_json = args.ibm_credentials_json
    if ibm_creds_json is None:
        _default_creds = (
            Path(__file__).resolve().parent.parent.parent  # up from src/qobench/ to research_benchmark/
            / "examples" / "ibm_credentials.example.json"
        )
        if _default_creds.is_file():
            ibm_creds_json = str(_default_creds)
            LOGGER.info("Using default IBM credentials file: %s", ibm_creds_json)
        else:
            LOGGER.debug(
                "Default IBM credentials file not found at %s; proceeding without.",
                _default_creds,
            )

    manager = QuantumHardwareManager(
        aws_profile=args.aws_profile,
        ibm_token=args.ibm_token,
        ibm_instance=args.ibm_instance,
        ibm_credentials_json=ibm_creds_json,
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

    # -- QPU Status Report --
    from datetime import datetime as _dt, timezone as _tz, timedelta as _td
    _sgt = _tz(_td(hours=8))
    _now_sgt = _dt.now(_sgt)
    print(f"\n  Time (SGT): {_now_sgt.strftime('%Y-%m-%d %H:%M %a')}")
    ibm_service = manager.sessions.get("ibm_quantum")
    if ibm_service is not None:
        from qobench.hardware_manager import _get_ibm_usage_remaining_seconds
        ibm_remaining = _get_ibm_usage_remaining_seconds(ibm_service)
        if ibm_remaining is not None:
            m, s = divmod(int(ibm_remaining), 60)
            print(f"  IBM Runtime Budget: {m}m {s}s remaining")
    print()
    _hdr = f"  {'QPU':<20} {'Status':<12} {'Qubits':>6}  {'Detail'}"
    print(_hdr)
    print("  " + "-" * (len(_hdr) - 2))
    for qid, qpu in manager.qpus.items():
        if qpu.is_available:
            status_str = "READY"
        else:
            status_str = "OFFLINE"
        detail = qpu.last_error if qpu.last_error else (qpu.name if qpu.is_available else "")
        if qpu.is_available and qpu.provider == "ibm" and manager.ibm_backends_preferred:
            pref = ", ".join(b["name"] for b in manager.ibm_backends_preferred)
            detail = f"{len(manager.ibm_backends)} backends, preferred: {pref}"
        print(f"  {qid:<20} {status_str:<12} {qpu.max_qubits:>6}  {detail}")
    print()

    # Determine scheduler QPUs (use largest instance to seed the initial list)
    if args.execution_mode == "single":
        selected_qpu = args.qpu_id or args.only_qpu
        if selected_qpu is None:
            # Pick first available hardware QPU
            avail = manager.get_available_qpus_for_size(num_qubits=0, include_simulators=include_simulators)
            selected_qpu = avail[0] if avail else None
        if selected_qpu is None:
            raise RuntimeError("No QPU available. Try --include-simulators or different credentials.")
        scheduler_qpus = [selected_qpu]
    else:
        restricted_qpus = _parse_qpu_list(args.qpus)
        if args.only_qpu is not None:
            restricted_qpus = [args.only_qpu]
        avail = manager.get_available_qpus_for_size(num_qubits=0, include_simulators=include_simulators)
        if restricted_qpus:
            scheduler_qpus = [q for q in avail if q in restricted_qpus]
        else:
            scheduler_qpus = list(avail)
        if not scheduler_qpus:
            raise RuntimeError("No QPUs available. Try --include-simulators or relax --qpus.")

    if args.method == "qrao":
        qrao_supported_qpus = {"ibm_quantum", "local_qiskit"}
        scheduler_qpus = [q for q in scheduler_qpus if q in qrao_supported_qpus]
        if not scheduler_qpus:
            raise RuntimeError(
                "QRAO requires ibm_quantum or local_qiskit. "
                "Select one with --qpu-id/--only-qpu."
            )
        # Single-backend execution only.
        scheduler_qpus = [scheduler_qpus[0]]

    LOGGER.info(
        "Execution plan | mode=%s qpus=%s instances=%d",
        args.execution_mode, scheduler_qpus, len(instance_paths),
    )

    # --- Checkpoint setup ---
    checkpoint_dir = Path(args.checkpoint_dir).resolve()
    checkpoint_path = checkpoint_dir / f"{problem_type.value}.json"
    checkpoint: dict[str, Any] = {}
    if not bool(getattr(args, "force_rerun", False)):
        checkpoint = _load_checkpoint(checkpoint_path)
        if checkpoint:
            LOGGER.info(
                "Checkpoint loaded | %d completed instances for %s",
                len(checkpoint), problem_type.value,
            )

    # --- Solve each instance ---
    results: list[dict[str, Any]] = []
    for idx, inst_path in enumerate(instance_paths, 1):
        inst_name = inst_path.name

        # Check if instance is already completed in checkpoint
        if inst_name in checkpoint and not bool(getattr(args, "force_rerun", False)):
            prev = checkpoint[inst_name]
            LOGGER.info(
                "\n[%d/%d] SKIPPING (already completed): %s  [obj=%s, feasible=%s]",
                idx, len(instance_paths), inst_name,
                prev.get("objective", "?"), prev.get("feasible", "?"),
            )
            results.append(prev)
            continue

        LOGGER.info(
            "\n[%d/%d] Starting instance: %s",
            idx, len(instance_paths), inst_name,
        )
        result = _solve_single_instance(
            args=args,
            problem=problem,
            problem_type=problem_type,
            instance_path=inst_path,
            manager=manager,
            scheduler_qpus=scheduler_qpus,
            run_dir=run_dir,
            run_stamp=run_stamp,
            log_path=log_path,
            include_simulators=include_simulators,
        )
        results.append(result)

        # Save to checkpoint if instance completed successfully
        if result.get("status") == "OK":
            checkpoint[inst_name] = result
            _save_checkpoint(checkpoint_path, checkpoint)
            LOGGER.info(
                "  Checkpoint saved | %s -> %s",
                inst_name, checkpoint_path,
            )

    # --- Summary ---
    total_runtime_sec = float(time.perf_counter() - run_start)
    LOGGER.info("\n" + "=" * 70)
    LOGGER.info("  BENCHMARK SUMMARY | %s %s", problem_type.value.upper(), args.method)
    LOGGER.info("=" * 70)
    hdr = f"  {'Instance':<18} {'Qubits':>6} {'Status':<8} {'Objective':>12} {'Feasible':>8} {'Time(s)':>8}"
    LOGGER.info(hdr)
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    ok_count = 0
    for r in results:
        status = r.get("status", "?")
        obj = r.get("objective", "")
        feas = r.get("feasible", "")
        t = r.get("time_sec", 0)
        if status == "OK":
            ok_count += 1
            try:
                obj_str = f"{float(obj):.1f}"
            except Exception:
                obj_str = str(obj)
            LOGGER.info(
                "  %-18s %6d %-8s %12s %8s %8.1f",
                r["instance"], r["qubits"], status, obj_str, feas, t,
            )
        else:
            LOGGER.info(
                "  %-18s %6d %-8s %12s %8s %8.1f",
                r["instance"], r.get("qubits", 0), status,
                r.get("reason", "")[:12], "", t if t else 0,
            )
    LOGGER.info("  " + "-" * (len(hdr) - 2))
    LOGGER.info(
        "  Total: %d instances | %d OK | %.1fs total runtime",
        len(results), ok_count, total_runtime_sec,
    )
    LOGGER.info("  Results: %s", run_dir)

    # Save combined summary
    summary = {
        "problem": problem_type.value,
        "method": args.method,
        "total_instances": len(results),
        "ok_count": ok_count,
        "total_runtime_sec": total_runtime_sec,
        "results": results,
    }
    (run_dir / "summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2), encoding="utf-8",
    )

    return 0 if ok_count == len(results) else 1


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
