from __future__ import annotations

import csv
import json
import math
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "research_benchmark" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qobench.problem_registry import get_problem  # noqa: E402
from qobench.problems.market_share import MarketShareProblem  # noqa: E402
from qobench.types import ProblemType  # noqa: E402

EPS = 1e-9
SENSITIVITY_MULTIPLIERS = [0.25, 0.5, 0.75, 1.0, 1.25, 2.0, 4.0]
SENSITIVITY_TIME_LIMIT_SEC = 10.0

LEDGER_COLUMNS = [
    "problem",
    "instance",
    "constraint_family",
    "constraint_expression",
    "penalty_symbol",
    "raw_penalty_value",
    "selection_rule_type",
    "selection_formula",
    "selection_inputs",
    "objective_scale_reference",
    "analytical_sufficiency_condition",
    "sufficiency_condition_satisfied",
    "qubo_normalization_applied",
    "normalization_factor",
    "normalization_stage",
    "penalty_after_normalization",
    "same_original_qubo_used_by_vqe",
    "same_original_qubo_used_by_cvar_vqe",
    "same_original_qubo_used_by_qaoa",
    "same_original_qubo_used_by_maqaoa",
    "same_original_qubo_used_by_wsqaoa",
    "same_original_qubo_used_by_pce",
    "same_original_qubo_used_by_qrao",
    "constructor_source_file",
    "constructor_function",
    "code_commit_or_version",
    "notes",
]

REPLICATE_COLUMNS = [
    "problem",
    "instance",
    "penalty_multiplier",
    "raw_penalty_value",
    "normalization_factor",
    "effective_penalty_after_normalization",
    "solver_name",
    "solver_status",
    "optimality_certified",
    "mip_gap",
    "solve_time_seconds",
    "qubo_optimum_energy",
    "decoded_original_feasible",
    "decoded_objective_value",
    "matches_classical_reference",
    "constraint_violation_count",
    "constraint_violation_magnitude",
    "notes",
]

SUMMARY_COLUMNS = [
    "problem",
    "instance",
    "smallest_multiplier_with_certified_feasible_qubo_optimum",
    "reported_multiplier",
    "reported_penalty_is_feasible",
    "reported_penalty_matches_reference",
    "higher_penalties_preserve_reference_optimum",
    "observed_conditioning_tradeoff",
    "interpretation",
]


@dataclass(frozen=True)
class InstanceSpec:
    problem: str
    problem_type: ProblemType
    instance: str
    source_reference: str
    load_path: Path | None
    load_kwargs: dict[str, Any]
    source_library: str


@dataclass(frozen=True)
class PenaltyTrace:
    penalty: float
    stage_name: str
    constraints_before: int
    constraints_after: int
    vars_before: int
    vars_after: int
    linear_lower: float
    linear_upper: float
    quadratic_lower: float
    quadratic_upper: float
    objective_bound_range: float
    objective_max_abs_coeff: float


def fmt(value: Any, digits: int = 8) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "not_applicable"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def git_version() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    except Exception:
        return "unknown_git_commit"


def all_instance_specs() -> list[InstanceSpec]:
    specs: list[InstanceSpec] = []
    for path in get_problem(ProblemType.MKP).list_instances(ROOT):
        specs.append(
            InstanceSpec(
                "MDKP",
                ProblemType.MKP,
                path.stem,
                str(path.relative_to(ROOT)),
                path,
                {},
                "OR-Library MDKP hpp",
            )
        )
    for path in get_problem(ProblemType.MIS).list_instances(ROOT):
        specs.append(
            InstanceSpec(
                "MIS",
                ProblemType.MIS,
                path.stem,
                str(path.relative_to(ROOT)),
                path,
                {},
                "DIMACS-style MIS benchmark set",
            )
        )
    for path in get_problem(ProblemType.QAP).list_instances(ROOT):
        specs.append(
            InstanceSpec(
                "QAP",
                ProblemType.QAP,
                path.stem,
                str(path.relative_to(ROOT)),
                path,
                {},
                "QAPLIB",
            )
        )
    for virtual_path in get_problem(ProblemType.MARKET_SHARE).list_instances(ROOT):
        parsed = MarketShareProblem.parse_generated_instance_name(virtual_path.name)
        if parsed is None:
            continue
        seed, num_products = parsed
        specs.append(
            InstanceSpec(
                "MSP",
                ProblemType.MARKET_SHARE,
                virtual_path.stem,
                f"seed={seed}; num_products={num_products}; target_ratio=0.5",
                None,
                {"seed": seed, "num_products": num_products},
                "Generated market-share benchmark grid",
            )
        )
    return specs


def sensitivity_specs() -> list[InstanceSpec]:
    return [
        InstanceSpec(
            "MDKP",
            ProblemType.MKP,
            "hp1",
            "Multi_Dimension_Knapsack/MKP_Instances/hpp/hp1.dat",
            ROOT / "Multi_Dimension_Knapsack/MKP_Instances/hpp/hp1.dat",
            {},
            "OR-Library MDKP hpp",
        ),
        InstanceSpec(
            "MIS",
            ProblemType.MIS,
            "1tc.32",
            "Maximum_Independent_Set/mis_benchmark_instances/1tc.32.txt",
            ROOT / "Maximum_Independent_Set/mis_benchmark_instances/1tc.32.txt",
            {},
            "DIMACS-style MIS benchmark set",
        ),
        InstanceSpec(
            "QAP",
            ProblemType.QAP,
            "tai10a",
            "Quadratic_Assignment_Problem/qapdata/tai10a.dat",
            ROOT / "Quadratic_Assignment_Problem/qapdata/tai10a.dat",
            {},
            "QAPLIB",
        ),
        InstanceSpec(
            "MSP",
            ProblemType.MARKET_SHARE,
            "ms_seed0_prod3",
            "seed=0; num_products=3; target_ratio=0.5",
            None,
            {"seed": 0, "num_products": 3},
            "Generated market-share benchmark grid",
        ),
    ]


def load_problem_instance(spec: InstanceSpec) -> tuple[Any, Any, Any]:
    problem = get_problem(spec.problem_type)
    instance = problem.load_instance(spec.load_path, **spec.load_kwargs)
    model, _ = problem.build_model(instance=instance)
    return problem, instance, model


def objective_max_abs(problem: Any) -> float:
    values: list[float] = []
    values.extend(abs(float(v)) for v in problem.objective.linear.to_dict().values())
    values.extend(abs(float(v)) for v in problem.objective.quadratic.to_dict().values())
    return max([v for v in values if v > EPS], default=0.0)


def used_penalty_trace(qp: Any, supplied_penalty: float | None = None) -> tuple[Any, PenaltyTrace]:
    converter = QuadraticProgramToQubo(penalty=supplied_penalty)
    current = qp
    used: list[PenaltyTrace] = []
    for stage in converter._converters:
        lin_bounds = current.objective.linear.bounds
        quad_bounds = current.objective.quadratic.bounds
        constraints_before = len(current.linear_constraints) + len(current.quadratic_constraints)
        vars_before = current.get_num_vars()
        objective_bound_range = (
            float(lin_bounds.upperbound)
            - float(lin_bounds.lowerbound)
            + float(quad_bounds.upperbound)
            - float(quad_bounds.lowerbound)
        )
        max_abs_coeff = objective_max_abs(current)
        current = stage.convert(current)
        constraints_after = len(current.linear_constraints) + len(current.quadratic_constraints)
        penalty = getattr(stage, "penalty", None)
        if penalty is not None and constraints_after < constraints_before:
            used.append(
                PenaltyTrace(
                    penalty=float(penalty),
                    stage_name=type(stage).__name__,
                    constraints_before=constraints_before,
                    constraints_after=constraints_after,
                    vars_before=vars_before,
                    vars_after=current.get_num_vars(),
                    linear_lower=float(lin_bounds.lowerbound),
                    linear_upper=float(lin_bounds.upperbound),
                    quadratic_lower=float(quad_bounds.lowerbound),
                    quadratic_upper=float(quad_bounds.upperbound),
                    objective_bound_range=float(objective_bound_range),
                    objective_max_abs_coeff=float(max_abs_coeff),
                )
            )
    if len(used) != 1:
        raise RuntimeError(f"Expected exactly one used penalty stage, got {len(used)}")
    return current, used[0]


def constraint_info(problem: str) -> tuple[str, str, str, str, str]:
    if problem == "MDKP":
        return (
            "capacity inequalities with binary-expanded integer slack",
            "sum_i w_{ki} x_i + s_k - C_k = 0 after slack conversion",
            "lambda_K",
            "-sum_i p_i x_i + lambda_K sum_k (sum_i w_{ki}x_i + s_k - C_k)^2",
            "If lambda_K exceeds the pre-penalty objective bound range, any integer violation with squared magnitude at least 1 is dominated by the objective range.",
        )
    if problem == "MIS":
        return (
            "edge-conflict inequalities",
            "x_i + x_j <= 1 for each edge (i,j)",
            "lambda_MIS",
            "-sum_i x_i + lambda_MIS sum_(i,j in E) x_i x_j",
            "For unit-weight MIS, lambda_MIS > 1 is sufficient to prefer removing an endpoint of a violated edge; the Qiskit n+1 bound is conservative.",
        )
    if problem == "QAP":
        return (
            "row/column assignment equalities",
            "sum_j x_{ij} = 1 and sum_i x_{ij} = 1",
            "lambda_Q",
            "sum_{i,k,j,l} f_{ik}d_{jl}x_{ij}x_{kl} + lambda_Q[sum_i(1-sum_j x_{ij})^2 + sum_j(1-sum_i x_{ij})^2]",
            "If lambda_Q exceeds the pre-penalty objective bound range, any assignment equality violation with squared magnitude at least 1 is dominated by the objective range.",
        )
    return (
        "target-allocation equalities with binary-expanded deviation variables",
        "sum_j demand_{gj} x_j + s_plus_g - s_minus_g - target_g = 0",
        "lambda_M",
        "sum_g (s_plus_g+s_minus_g) + lambda_M sum_g(sum_j demand_{gj}x_j+s_plus_g-s_minus_g-target_g)^2",
        "If lambda_M exceeds the pre-penalty objective bound range, any integer equality violation with squared magnitude at least 1 is dominated by the objective range.",
    )


def build_ledger_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    version = git_version()
    for spec in all_instance_specs():
        _, _, model = load_problem_instance(spec)
        qp = from_docplex_mp(model)
        _, trace = used_penalty_trace(qp)
        family, expression, symbol, form, sufficiency = constraint_info(spec.problem)
        inputs = {
            "stage": trace.stage_name,
            "linear_lower": trace.linear_lower,
            "linear_upper": trace.linear_upper,
            "quadratic_lower": trace.quadratic_lower,
            "quadratic_upper": trace.quadratic_upper,
            "constraints_converted": trace.constraints_before - trace.constraints_after,
            "vars_before_stage": trace.vars_before,
            "vars_after_stage": trace.vars_after,
        }
        same = {
            "same_original_qubo_used_by_vqe": "true",
            "same_original_qubo_used_by_cvar_vqe": "true",
            "same_original_qubo_used_by_qaoa": "true",
            "same_original_qubo_used_by_maqaoa": "true",
            "same_original_qubo_used_by_wsqaoa": "true",
            "same_original_qubo_used_by_pce": "true",
            "same_original_qubo_used_by_qrao": "true",
        }
        rows.append(
            {
                "problem": spec.problem,
                "instance": spec.instance,
                "constraint_family": family,
                "constraint_expression": expression,
                "penalty_symbol": symbol,
                "raw_penalty_value": fmt(trace.penalty),
                "selection_rule_type": "analytical_auto_bound_from_qiskit_converter",
                "selection_formula": "lambda = 1.0 + (linear.upperbound - linear.lowerbound) + (quadratic.upperbound - quadratic.lowerbound)",
                "selection_inputs": json.dumps(inputs, sort_keys=True),
                "objective_scale_reference": (
                    f"pre-penalty objective_bound_range={fmt(trace.objective_bound_range)}; "
                    f"pre-penalty max_abs_objective_coefficient={fmt(trace.objective_max_abs_coeff)}"
                ),
                "analytical_sufficiency_condition": sufficiency,
                "sufficiency_condition_satisfied": fmt(trace.penalty > trace.objective_bound_range),
                "qubo_normalization_applied": "false",
                "normalization_factor": "1",
                "normalization_stage": "none; QUBO passed directly to to_ising()/method encodings",
                "penalty_after_normalization": fmt(trace.penalty),
                **same,
                "constructor_source_file": "research_benchmark/src/qobench/qubo.py; qiskit_optimization.converters.LinearEqualityToPenalty/LinearInequalityToPenalty",
                "constructor_function": "convert_docplex_to_qubo(); QuadraticProgramToQubo.convert(); _auto_define_penalty()",
                "code_commit_or_version": version,
                "notes": f"Penalty term form: {form}. All quantum methods construct or encode from this same converted QUBO; PCE/QRAO apply method-specific encodings after original QUBO construction.",
            }
        )
    return rows


def qubo_terms(qubo: Any) -> tuple[np.ndarray, dict[tuple[int, int], float], float]:
    n = qubo.get_num_vars()
    linear = np.array(qubo.objective.linear.to_array(), dtype=float)
    quadratic: dict[tuple[int, int], float] = defaultdict(float)
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        value = float(coeff)
        if abs(value) <= EPS:
            continue
        if int(i) == int(j):
            linear[int(i)] += value
        else:
            a, b = sorted((int(i), int(j)))
            quadratic[(a, b)] += value
    quadratic = {k: v for k, v in quadratic.items() if abs(v) > EPS}
    return linear, quadratic, float(qubo.objective.constant)


def solve_qubo_gurobi(qubo: Any, time_limit: float) -> dict[str, Any]:
    import gurobipy as gp
    from gurobipy import GRB

    linear, quadratic, constant = qubo_terms(qubo)
    n = len(linear)
    model = gp.Model("penalty_sensitivity_qubo")
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = float(time_limit)
    model.Params.NonConvex = 2
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    objective = gp.LinExpr(float(constant))
    for i, coeff in enumerate(linear):
        if abs(float(coeff)) > EPS:
            objective += float(coeff) * x[i]
    for (i, j), coeff in quadratic.items():
        objective += float(coeff) * x[i] * x[j]
    model.setObjective(objective, GRB.MINIMIZE)
    start = time.perf_counter()
    model.optimize()
    elapsed = time.perf_counter() - start
    has_solution = int(getattr(model, "SolCount", 0)) > 0
    vector = None
    if has_solution:
        vector = [1 if x[i].X >= 0.5 else 0 for i in range(n)]
    status_map = {
        GRB.OPTIMAL: "OPTIMAL",
        GRB.TIME_LIMIT: "TIME_LIMIT",
        GRB.INFEASIBLE: "INFEASIBLE",
        GRB.INF_OR_UNBD: "INF_OR_UNBD",
        GRB.UNBOUNDED: "UNBOUNDED",
    }
    return {
        "status": status_map.get(model.Status, str(model.Status)),
        "optimality_certified": bool(model.Status == GRB.OPTIMAL),
        "mip_gap": float(getattr(model, "MIPGap", math.nan)) if has_solution else math.nan,
        "solve_time_seconds": elapsed,
        "objective": float(model.ObjVal) if has_solution else math.nan,
        "vector": vector,
    }


def assignment_from_vector(qubo: Any, vector: list[int]) -> dict[str, int]:
    index_to_name = {idx: name for name, idx in qubo.variables_index.items()}
    return {index_to_name[i]: int(vector[i]) for i in range(len(vector))}


def reconstruct(problem: str, instance: Any, assignment: dict[str, int]) -> dict[str, Any]:
    def bit(name: str) -> int:
        return 1 if int(assignment.get(name, 0)) == 1 else 0

    if problem == "MIS":
        selected = [bit(f"x_{i}") for i in range(instance.num_nodes)]
        conflicts = sum(1 for u, v in instance.edges if selected[u] and selected[v])
        return {
            "feasible": conflicts == 0,
            "objective": float(sum(selected)),
            "violation_count": int(conflicts),
            "violation_magnitude": float(conflicts),
        }
    if problem == "MDKP":
        selected = [bit(f"x_{i}") for i in range(instance.n)]
        profit = sum(instance.profits[i] * selected[i] for i in range(instance.n))
        overflow = []
        for k in range(instance.m):
            used = sum(instance.weights[k][i] * selected[i] for i in range(instance.n))
            overflow.append(max(0.0, float(used - instance.capacities[k])))
        return {
            "feasible": all(v <= 0 for v in overflow),
            "objective": float(profit),
            "violation_count": int(sum(1 for v in overflow if v > 0)),
            "violation_magnitude": float(sum(overflow)),
        }
    if problem == "QAP":
        n = instance.n
        x = [[bit(f"x_{i}_{j}") for j in range(n)] for i in range(n)]
        objective = 0.0
        for i in range(n):
            for k in range(n):
                for j in range(n):
                    if not x[i][j]:
                        continue
                    for l in range(n):
                        if x[k][l]:
                            objective += instance.flow[i][k] * instance.distance[j][l]
        row_sums = [sum(row) for row in x]
        col_sums = [sum(x[i][j] for i in range(n)) for j in range(n)]
        deviations = [abs(v - 1) for v in row_sums + col_sums]
        return {
            "feasible": all(v == 0 for v in deviations),
            "objective": float(objective),
            "violation_count": int(sum(1 for v in deviations if v > 0)),
            "violation_magnitude": float(sum(deviations)),
        }
    num_products = instance.num_products
    num_retailers = instance.num_retailers
    x = [bit(f"x_{j}") for j in range(num_retailers)]
    deviations = []
    for g in range(num_products):
        realized = sum(instance.demands[g][j] * x[j] for j in range(num_retailers))
        deviations.append(abs(realized - instance.target_demands[g]))
    return {
        "feasible": True,
        "objective": float(sum(deviations)),
        "violation_count": 0,
        "violation_magnitude": 0.0,
    }


def read_reference_values() -> dict[tuple[str, str], float]:
    refs: dict[tuple[str, str], float] = {}
    paths = [
        ("MDKP", ROOT / "classical_solutions/results/mdkp/summary.csv", "instance", "objective_value"),
        ("MIS", ROOT / "classical_solutions/results/mis/summary.csv", "instance_id", "objective_value"),
        ("QAP", ROOT / "classical_solutions/results/qap/summary.csv", "instance_id", "objective_value"),
    ]
    for problem, path, id_col, value_col in paths:
        if not path.exists():
            continue
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                refs[(problem, str(row[id_col]).replace(".dat", "").replace(".txt", ""))] = float(row[value_col])
    return refs


def market_share_reference(instance: Any) -> float:
    best = math.inf
    n = instance.num_retailers
    for mask in range(1 << n):
        total = 0.0
        for g in range(instance.num_products):
            realized = 0
            for j in range(n):
                if (mask >> j) & 1:
                    realized += instance.demands[g][j]
            total += abs(realized - instance.target_demands[g])
            if total >= best:
                break
        if total < best:
            best = float(total)
    return best


def reference_value(spec: InstanceSpec, instance: Any, refs: dict[tuple[str, str], float]) -> float:
    if spec.problem == "MDKP":
        return refs[(spec.problem, f"{spec.instance}")]
    if spec.problem == "MIS":
        return refs[(spec.problem, spec.instance)]
    if spec.problem == "QAP":
        return refs[(spec.problem, spec.instance)]
    return market_share_reference(instance)


def matches_reference(problem: str, objective: float, reference: float, feasible: bool) -> bool:
    if not feasible or not math.isfinite(objective):
        return False
    return abs(float(objective) - float(reference)) <= 1e-6


def run_sensitivity() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    refs = read_reference_values()
    replicate_rows: list[dict[str, str]] = []
    by_case: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for spec in sensitivity_specs():
        _, instance, model = load_problem_instance(spec)
        qp = from_docplex_mp(model)
        _, trace = used_penalty_trace(qp)
        reference = reference_value(spec, instance, refs)
        for multiplier in SENSITIVITY_MULTIPLIERS:
            print(
                f"[sensitivity] {spec.problem} {spec.instance} multiplier={multiplier}",
                flush=True,
            )
            penalty = float(multiplier) * trace.penalty
            qubo, _ = used_penalty_trace(qp, supplied_penalty=penalty)
            result = solve_qubo_gurobi(qubo, SENSITIVITY_TIME_LIMIT_SEC)
            if result["vector"] is None:
                decoded = {
                    "feasible": False,
                    "objective": math.nan,
                    "violation_count": -1,
                    "violation_magnitude": math.nan,
                }
                assignment = {}
            else:
                assignment = assignment_from_vector(qubo, result["vector"])
                decoded = reconstruct(spec.problem, instance, assignment)
            match = matches_reference(
                spec.problem, decoded["objective"], reference, bool(decoded["feasible"])
            )
            notes = f"classical_reference={fmt(reference)}; time_limit_sec={fmt(SENSITIVITY_TIME_LIMIT_SEC)}"
            row = {
                "problem": spec.problem,
                "instance": spec.instance,
                "penalty_multiplier": fmt(multiplier),
                "raw_penalty_value": fmt(penalty),
                "normalization_factor": "1",
                "effective_penalty_after_normalization": fmt(penalty),
                "solver_name": "Gurobi 13.0.1",
                "solver_status": result["status"],
                "optimality_certified": fmt(result["optimality_certified"]),
                "mip_gap": fmt(result["mip_gap"]),
                "solve_time_seconds": fmt(result["solve_time_seconds"]),
                "qubo_optimum_energy": fmt(result["objective"]),
                "decoded_original_feasible": fmt(decoded["feasible"]),
                "decoded_objective_value": fmt(decoded["objective"]),
                "matches_classical_reference": fmt(match),
                "constraint_violation_count": fmt(decoded["violation_count"]),
                "constraint_violation_magnitude": fmt(decoded["violation_magnitude"]),
                "notes": notes,
            }
            replicate_rows.append(row)
            by_case[(spec.problem, spec.instance)].append(row)

    summary_rows: list[dict[str, str]] = []
    for (problem, instance), rows in by_case.items():
        rows_sorted = sorted(rows, key=lambda r: float(r["penalty_multiplier"]))
        certified_feasible = [
            float(r["penalty_multiplier"])
            for r in rows_sorted
            if r["optimality_certified"] == "true" and r["decoded_original_feasible"] == "true"
        ]
        smallest = min(certified_feasible) if certified_feasible else None
        reported = next(r for r in rows_sorted if abs(float(r["penalty_multiplier"]) - 1.0) < EPS)
        higher = [r for r in rows_sorted if float(r["penalty_multiplier"]) >= 1.0]
        higher_noncertified = any(r["optimality_certified"] != "true" for r in higher)
        higher_preserve = all(
            r["decoded_original_feasible"] == "true" and r["matches_classical_reference"] == "true"
            for r in higher
        )
        energies = [abs(float(r["raw_penalty_value"])) for r in rows_sorted]
        conditioning = (
            f"penalty scale increases {fmt(max(energies) / min(energies))}x across grid; "
            "larger multipliers preserve objective only if certified rows remain reference-matching."
        )
        reported_certified = reported["optimality_certified"] == "true"
        if not reported_certified:
            reported_feasibility = f"unresolved_noncertified_incumbent_feasible={reported['decoded_original_feasible']}"
            reported_match = "unresolved_noncertified"
            interpretation = (
                "Reported penalty row was not certified within the audit time limit; incumbent feasibility "
                "is recorded in the replicate CSV but is not proof of penalty sufficiency."
            )
        elif reported["decoded_original_feasible"] == "true" and reported["matches_classical_reference"] == "true":
            reported_feasibility = reported["decoded_original_feasible"]
            reported_match = reported["matches_classical_reference"]
            interpretation = "Reported penalty recovers a certified feasible QUBO optimum matching the classical reference."
        elif reported["decoded_original_feasible"] == "true":
            reported_feasibility = reported["decoded_original_feasible"]
            reported_match = reported["matches_classical_reference"]
            interpretation = "Reported penalty recovers a feasible certified QUBO optimum but not the recorded classical reference."
        else:
            reported_feasibility = reported["decoded_original_feasible"]
            reported_match = reported["matches_classical_reference"]
            interpretation = "Reported penalty gives an infeasible certified QUBO optimum; formulation requires further review."
        if higher_noncertified:
            higher_preserve_value = "unresolved_noncertified_rows_present"
        else:
            higher_preserve_value = fmt(higher_preserve)
        summary_rows.append(
            {
                "problem": problem,
                "instance": instance,
                "smallest_multiplier_with_certified_feasible_qubo_optimum": fmt(smallest),
                "reported_multiplier": "1",
                "reported_penalty_is_feasible": reported_feasibility,
                "reported_penalty_matches_reference": reported_match,
                "higher_penalties_preserve_reference_optimum": higher_preserve_value,
                "observed_conditioning_tradeoff": conditioning,
                "interpretation": interpretation,
            }
        )
    return replicate_rows, summary_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_digest(
    ledger_rows: list[dict[str, str]],
    replicate_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
) -> None:
    by_problem = defaultdict(list)
    for row in ledger_rows:
        by_problem[row["problem"]].append(float(row["raw_penalty_value"]))
    sensitivity = {
        (row["problem"], row["instance"]): row for row in summary_rows
    }
    lines = [
        "# Penalty Audit Digest",
        "",
        "1. Exact QUBO penalty terms:",
        "- MDKP: `-sum_i p_i x_i + lambda_K sum_k (sum_i w_ki x_i + s_k - C_k)^2` after integer slack conversion.",
        "- MIS: `-sum_i x_i + lambda_MIS sum_(i,j in E) x_i x_j`.",
        "- QAP: dense flow-distance objective plus `lambda_Q` row/column one-hot penalties.",
        "- MSP: absolute-deviation objective with target equalities enforced by `lambda_M` after binary expansion of deviation variables.",
        "",
        "2. Exact code formula selecting each reported penalty value:",
        "`lambda = 1.0 + (linear.upperbound - linear.lowerbound) + (quadratic.upperbound - quadratic.lowerbound)` from Qiskit's `LinearEqualityToPenalty` or `LinearInequalityToPenalty` converter stage.",
        "",
        "3. Selection type: analytical Qiskit auto-bound rule, not hardware tuned and not grid tuned.",
        "4. Inputs: pre-penalty objective linear/quadratic expression bounds at the constraint-elimination stage; per-instance values are in `selection_inputs`.",
        "5. One penalty is shared across all constraints converted in each instance.",
        "6. QUBO normalization before Hamiltonian construction: no.",
        "7. Normalization factor/stage: factor 1; the converted QUBO is passed directly to `to_ising()` or to method-specific encodings.",
        "8. All method families consume the same original penalized QUBO; PCE and QRAO encode/reduce only after that QUBO is built.",
        "",
        "9. Four-instance penalty-sensitivity study:",
    ]
    for key in [("MDKP", "hp1"), ("MIS", "1tc.32"), ("QAP", "tai10a"), ("MSP", "ms_seed0_prod3")]:
        row = sensitivity[key]
        lines.append(
            f"- {key[0]} {key[1]}: smallest certified feasible multiplier={row['smallest_multiplier_with_certified_feasible_qubo_optimum']}; "
            f"reported feasible={row['reported_penalty_is_feasible']}; "
            f"reported matches reference={row['reported_penalty_matches_reference']}; "
            f"higher penalties preserve reference={row['higher_penalties_preserve_reference_optimum']}."
        )
    lines.extend(
        [
            "",
            "10. Limitations:",
            "- The sensitivity study is classical and formulation-level only; it does not rerun quantum hardware.",
            "- Gurobi time limits are reported in replicate rows; any non-certified row must not be used as proof of penalty sufficiency.",
            "- Raw lambda values are not cross-family hardness measures because objective scales and encodings differ.",
            "",
            "Penalty scale ranges in the tested benchmark set:",
        ]
    )
    for problem in ["MDKP", "MIS", "QAP", "MSP"]:
        values = by_problem[problem]
        lines.append(f"- {problem}: {fmt(min(values))} to {fmt(max(values))}.")
    (OUT / "penalty_audit_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    ledger_rows = build_ledger_rows()
    replicate_rows, summary_rows = run_sensitivity()
    write_csv(OUT / "penalty_provenance_ledger.csv", LEDGER_COLUMNS, ledger_rows)
    write_csv(OUT / "penalty_sensitivity_replicates.csv", REPLICATE_COLUMNS, replicate_rows)
    write_csv(OUT / "penalty_sensitivity_summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_digest(ledger_rows, replicate_rows, summary_rows)


if __name__ == "__main__":
    main()
