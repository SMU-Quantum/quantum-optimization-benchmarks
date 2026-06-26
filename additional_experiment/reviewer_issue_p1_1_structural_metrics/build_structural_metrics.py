from __future__ import annotations

import csv
import json
import math
import statistics
import sys
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
from qobench.problems.mkp import MKPInstance  # noqa: E402
from qobench.problems.mis import MISInstance  # noqa: E402
from qobench.problems.qap import QAPInstance  # noqa: E402
from qobench.types import ProblemType  # noqa: E402

EPS = 1e-12

INSTANCE_COLUMNS = [
    "problem",
    "instance",
    "source_library",
    "source_reference",
    "original_problem_size",
    "qubo_variables",
    "original_constraint_count",
    "independent_constraint_count",
    "qubo_linear_terms",
    "qubo_quadratic_terms",
    "qubo_nonzero_terms",
    "qubo_density",
    "mean_qubo_degree",
    "max_qubo_degree",
    "min_abs_nonzero_coefficient",
    "median_abs_nonzero_coefficient",
    "max_abs_nonzero_coefficient",
    "coefficient_dynamic_range",
    "mean_abs_quadratic_coefficient",
    "max_abs_quadratic_coefficient",
    "penalty_scale_min",
    "penalty_scale_max",
    "penalty_to_objective_ratio",
    "feasibility_characterization",
    "log10_feasible_fraction",
    "notes",
]

SUMMARY_COLUMNS = [
    "problem",
    "n_instances",
    "source_library",
    "original_problem_size_range",
    "qubo_variable_range",
    "original_constraint_range",
    "qubo_density_median",
    "qubo_density_min",
    "qubo_density_max",
    "mean_qubo_degree_median",
    "max_qubo_degree_max",
    "coefficient_dynamic_range_median",
    "coefficient_dynamic_range_max",
    "penalty_to_objective_ratio_median",
    "qap_log10_feasible_fraction_range",
    "structural_interpretation",
]

QAP_GEOMETRY_COLUMNS = [
    "instance",
    "qap_dimension_n",
    "direct_qubo_variables",
    "assignment_constraints_total",
    "assignment_constraints_independent",
    "feasible_assignments",
    "binary_search_space_size",
    "log10_feasible_fraction",
    "qubo_density",
    "mean_qubo_degree",
    "max_qubo_degree",
    "min_abs_nonzero_coefficient",
    "max_abs_nonzero_coefficient",
    "coefficient_dynamic_range",
    "hardware_feasible_methods",
    "simulator_feasible_methods",
    "notes",
]

INTERPRETATIONS = {
    "MDKP": "Sparse-to-moderate packing couplings with capacity penalties.",
    "MIS": "Graph-structured sparsity determined by the input-edge set.",
    "QAP": (
        "Dense flow-distance couplings combined with row/column one-hot penalties; "
        "feasible assignments occupy an exponentially tiny fraction of binary strings."
    ),
    "MSP": "Target-allocation penalties with structured balancing constraints.",
}


@dataclass(frozen=True)
class BenchmarkInstance:
    problem: str
    instance_name: str
    source_library: str
    source_reference: str
    loaded: Any


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "not_applicable"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return str(float(value))
        return f"{float(value):.{digits}g}"
    return str(value)


def canonical_qubo_terms(qubo: Any) -> tuple[np.ndarray, dict[tuple[int, int], float]]:
    n = qubo.get_num_vars()
    diagonal = np.array(qubo.objective.linear.to_array(), dtype=float)
    off_diagonal: dict[tuple[int, int], float] = defaultdict(float)
    for (i, j), coeff in qubo.objective.quadratic.to_dict().items():
        value = float(coeff)
        if abs(value) <= EPS:
            continue
        if i == j:
            diagonal[int(i)] += value
        else:
            a, b = sorted((int(i), int(j)))
            off_diagonal[(a, b)] += value
    off_diagonal = {
        pair: value for pair, value in off_diagonal.items() if abs(value) > EPS
    }
    return diagonal, off_diagonal


def objective_max_abs(problem: Any) -> float:
    values: list[float] = []
    values.extend(abs(float(v)) for v in problem.objective.linear.to_dict().values())
    values.extend(abs(float(v)) for v in problem.objective.quadratic.to_dict().values())
    values = [value for value in values if value > EPS]
    return max(values) if values else 0.0


def convert_with_penalty_trace(model: Any) -> tuple[Any, Any, list[float], float]:
    qp = from_docplex_mp(model)
    converter = QuadraticProgramToQubo()
    current = qp
    used_penalties: list[float] = []
    objective_scales: list[float] = []

    for stage in converter._converters:
        before_constraints = len(current.linear_constraints) + len(current.quadratic_constraints)
        before_obj_max = objective_max_abs(current)
        current = stage.convert(current)
        after_constraints = len(current.linear_constraints) + len(current.quadratic_constraints)
        penalty = getattr(stage, "penalty", None)
        if penalty is not None and after_constraints < before_constraints:
            used_penalties.append(float(penalty))
            if before_obj_max > EPS:
                objective_scales.append(before_obj_max)

    objective_scale = max(objective_scales) if objective_scales else objective_max_abs(qp)
    return qp, current, used_penalties, objective_scale


def constraint_rank(qp: Any) -> int:
    rows: list[np.ndarray] = []
    for constraint in qp.linear_constraints:
        rows.append(np.array(constraint.linear.to_array(), dtype=float))
    if not rows:
        return 0
    return int(np.linalg.matrix_rank(np.vstack(rows), tol=1e-9))


def qap_log10_feasible_fraction(n: int) -> float:
    return (math.lgamma(n + 1) / math.log(10.0)) - (n * n * math.log10(2.0))


def instance_metadata(item: BenchmarkInstance) -> tuple[str, str, str]:
    inst = item.loaded
    if isinstance(inst, MKPInstance):
        return (
            f"{inst.n} items x {inst.m} dimensions",
            "capacity inequalities",
            "not_computed",
        )
    if isinstance(inst, MISInstance):
        return (
            f"{inst.num_nodes} nodes, {len(inst.edges)} edges",
            "edge-conflict inequalities",
            "not_computed",
        )
    if isinstance(inst, QAPInstance):
        return (
            f"n={inst.n}",
            f"permutation assignment; {inst.n}! feasible one-hot assignments",
            fmt(qap_log10_feasible_fraction(inst.n)),
        )
    return (
        f"{inst.num_products} products, {inst.num_retailers} retailers",
        "per-product target equalities with binary-expanded deviation variables",
        "not_computed",
    )


def qap_note(inst: QAPInstance) -> str:
    return (
        f"Direct one-hot QAP formulation has n^2={inst.n * inst.n} binaries, "
        f"2n={2 * inst.n} assignment equalities, and n! feasible assignments."
    )


def collect_instances() -> list[BenchmarkInstance]:
    instances: list[BenchmarkInstance] = []

    mkp_problem = get_problem(ProblemType.MKP)
    for path in mkp_problem.list_instances(ROOT):
        instances.append(
            BenchmarkInstance(
                problem="MDKP",
                instance_name=path.stem,
                source_library="OR-Library MDKP hpp",
                source_reference=str(path.relative_to(ROOT)),
                loaded=mkp_problem.load_instance(path),
            )
        )

    mis_problem = get_problem(ProblemType.MIS)
    for path in mis_problem.list_instances(ROOT):
        instances.append(
            BenchmarkInstance(
                problem="MIS",
                instance_name=path.stem,
                source_library="DIMACS-style MIS benchmark set",
                source_reference=str(path.relative_to(ROOT)),
                loaded=mis_problem.load_instance(path),
            )
        )

    qap_problem = get_problem(ProblemType.QAP)
    for path in qap_problem.list_instances(ROOT):
        instances.append(
            BenchmarkInstance(
                problem="QAP",
                instance_name=path.stem,
                source_library="QAPLIB",
                source_reference=str(path.relative_to(ROOT)),
                loaded=qap_problem.load_instance(path),
            )
        )

    msp_problem = get_problem(ProblemType.MARKET_SHARE)
    for virtual_path in msp_problem.list_instances(ROOT):
        parsed = MarketShareProblem.parse_generated_instance_name(virtual_path.name)
        if parsed is None:
            continue
        seed, num_products = parsed
        instances.append(
            BenchmarkInstance(
                problem="MSP",
                instance_name=virtual_path.stem,
                source_library="Generated market-share benchmark grid",
                source_reference=f"seed={seed}; num_products={num_products}; target_ratio=0.5",
                loaded=msp_problem.load_instance(None, seed=seed, num_products=num_products),
            )
        )

    return instances


def problem_to_type(problem: str) -> ProblemType:
    if problem == "MDKP":
        return ProblemType.MKP
    if problem == "MSP":
        return ProblemType.MARKET_SHARE
    return ProblemType(problem.lower())


def build_instance_row(item: BenchmarkInstance) -> tuple[dict[str, str], dict[str, Any]]:
    problem = get_problem(problem_to_type(item.problem))
    model, _ = problem.build_model(item.loaded)
    qp, qubo, penalties, objective_scale = convert_with_penalty_trace(model)

    diagonal, off_diagonal = canonical_qubo_terms(qubo)
    linear_terms = int(np.count_nonzero(np.abs(diagonal) > EPS))
    quadratic_terms = len(off_diagonal)
    nonzero_terms = linear_terms + quadratic_terms
    n_qubo = qubo.get_num_vars()
    density = 0.0 if n_qubo <= 1 else (2.0 * quadratic_terms) / (n_qubo * (n_qubo - 1))
    mean_degree = 0.0 if n_qubo == 0 else (2.0 * quadratic_terms) / n_qubo
    degrees = [0] * n_qubo
    for i, j in off_diagonal:
        degrees[i] += 1
        degrees[j] += 1
    max_degree = max(degrees) if degrees else 0

    abs_values = [abs(float(v)) for v in diagonal if abs(float(v)) > EPS]
    abs_values.extend(abs(value) for value in off_diagonal.values() if abs(value) > EPS)
    min_abs = min(abs_values) if abs_values else 0.0
    median_abs = statistics.median(abs_values) if abs_values else 0.0
    max_abs = max(abs_values) if abs_values else 0.0
    dynamic_range = (max_abs / min_abs) if min_abs > EPS else 0.0
    abs_quadratic = [abs(value) for value in off_diagonal.values() if abs(value) > EPS]
    mean_abs_quadratic = statistics.fmean(abs_quadratic) if abs_quadratic else 0.0
    max_abs_quadratic = max(abs_quadratic) if abs_quadratic else 0.0

    penalty_min = min(penalties) if penalties else None
    penalty_max = max(penalties) if penalties else None
    penalty_ratio = (
        penalty_max / objective_scale
        if penalty_max is not None and objective_scale > EPS
        else None
    )

    size, feasibility, log10_feasible = instance_metadata(item)
    note = qap_note(item.loaded) if isinstance(item.loaded, QAPInstance) else ""
    if item.problem == "MSP":
        note = "Generated instance; integer deviation variables are binary-expanded in QUBO conversion."
    if item.problem == "MDKP":
        note = "Inequality capacities are converted with integer slack variables before QUBO penalties."

    row = {
        "problem": item.problem,
        "instance": item.instance_name,
        "source_library": item.source_library,
        "source_reference": item.source_reference,
        "original_problem_size": size,
        "qubo_variables": fmt(n_qubo),
        "original_constraint_count": fmt(len(qp.linear_constraints) + len(qp.quadratic_constraints)),
        "independent_constraint_count": fmt(constraint_rank(qp)),
        "qubo_linear_terms": fmt(linear_terms),
        "qubo_quadratic_terms": fmt(quadratic_terms),
        "qubo_nonzero_terms": fmt(nonzero_terms),
        "qubo_density": fmt(density),
        "mean_qubo_degree": fmt(mean_degree),
        "max_qubo_degree": fmt(max_degree),
        "min_abs_nonzero_coefficient": fmt(min_abs),
        "median_abs_nonzero_coefficient": fmt(median_abs),
        "max_abs_nonzero_coefficient": fmt(max_abs),
        "coefficient_dynamic_range": fmt(dynamic_range),
        "mean_abs_quadratic_coefficient": fmt(mean_abs_quadratic),
        "max_abs_quadratic_coefficient": fmt(max_abs_quadratic),
        "penalty_scale_min": fmt(penalty_min),
        "penalty_scale_max": fmt(penalty_max),
        "penalty_to_objective_ratio": fmt(penalty_ratio),
        "feasibility_characterization": feasibility,
        "log10_feasible_fraction": log10_feasible,
        "notes": note,
    }
    numeric = {
        "qubo_variables": n_qubo,
        "original_constraint_count": len(qp.linear_constraints) + len(qp.quadratic_constraints),
        "qubo_density": density,
        "mean_qubo_degree": mean_degree,
        "max_qubo_degree": max_degree,
        "coefficient_dynamic_range": dynamic_range,
        "penalty_to_objective_ratio": penalty_ratio,
        "qap_log10_feasible_fraction": (
            qap_log10_feasible_fraction(item.loaded.n)
            if isinstance(item.loaded, QAPInstance)
            else None
        ),
        "qap_dimension_n": item.loaded.n if isinstance(item.loaded, QAPInstance) else None,
        "min_abs_nonzero_coefficient": min_abs,
        "max_abs_nonzero_coefficient": max_abs,
    }
    if isinstance(item.loaded, MKPInstance):
        numeric.update({"size_a": item.loaded.n, "size_b": item.loaded.m})
    elif isinstance(item.loaded, MISInstance):
        numeric.update({"size_a": item.loaded.num_nodes, "size_b": len(item.loaded.edges)})
    elif isinstance(item.loaded, QAPInstance):
        numeric.update({"size_a": item.loaded.n, "size_b": None})
    else:
        numeric.update({"size_a": item.loaded.num_products, "size_b": item.loaded.num_retailers})
    return row, numeric


def parse_yes(value: Any) -> bool:
    return str(value).strip().lower() in {"yes", "true", "1", "feasible"}


def qap_feasible_counts(root: Path) -> tuple[dict[str, int], dict[str, int]]:
    hardware: dict[str, set[str]] = defaultdict(set)
    simulator: dict[str, set[str]] = defaultdict(set)
    plot_path = root / "research_benchmark/research_benchmark/results_hardware/qap/plots/qap_plot_data_main.csv"
    if plot_path.exists():
        with plot_path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if parse_yes(row.get("Feas")):
                    hardware[row["Instance"]].add(row.get("Method", "unknown"))

    sim_root = root / "research_benchmark/research_benchmark/results_simulator/qap"
    if sim_root.exists():
        for result_path in sim_root.rglob("result.json"):
            data = json.loads(result_path.read_text(encoding="utf-8"))
            instance = result_path.parent.name.removesuffix("_dat")
            method = result_path.parents[1].name if len(result_path.parents) > 1 else "unknown"
            if parse_yes(data.get("feasible", data.get("is_feasible", False))):
                simulator[instance].add(method)

    return (
        {instance: len(methods) for instance, methods in hardware.items()},
        {instance: len(methods) for instance, methods in simulator.items()},
    )


def build_qap_geometry_rows(
    rows: list[dict[str, str]], numerics: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    hardware_counts, simulator_counts = qap_feasible_counts(ROOT)
    out_rows: list[dict[str, str]] = []
    for row in rows:
        if row["problem"] != "QAP":
            continue
        instance = row["instance"]
        n = int(numerics[instance]["qap_dimension_n"])
        n_qubo = int(row["qubo_variables"])
        note = (
            "No feasible hardware method in existing QAP artifacts; "
            "no QAP simulator feasible artifact found."
        )
        if simulator_counts.get(instance, 0) > 0:
            note = "Simulator feasible count read from existing QAP simulator artifacts."
        out_rows.append(
            {
                "instance": instance,
                "qap_dimension_n": fmt(n),
                "direct_qubo_variables": fmt(n_qubo),
                "assignment_constraints_total": fmt(2 * n),
                "assignment_constraints_independent": fmt(2 * n - 1),
                "feasible_assignments": fmt(math.factorial(n)),
                "binary_search_space_size": fmt(2**n_qubo),
                "log10_feasible_fraction": row["log10_feasible_fraction"],
                "qubo_density": row["qubo_density"],
                "mean_qubo_degree": row["mean_qubo_degree"],
                "max_qubo_degree": row["max_qubo_degree"],
                "min_abs_nonzero_coefficient": row["min_abs_nonzero_coefficient"],
                "max_abs_nonzero_coefficient": row["max_abs_nonzero_coefficient"],
                "coefficient_dynamic_range": row["coefficient_dynamic_range"],
                "hardware_feasible_methods": fmt(hardware_counts.get(instance, 0)),
                "simulator_feasible_methods": fmt(simulator_counts.get(instance, 0)),
                "notes": note,
            }
        )
    return out_rows


def range_text(values: list[Any], digits: int = 6) -> str:
    if not values:
        return "not_applicable"
    return f"{fmt(min(values), digits)}-{fmt(max(values), digits)}"


def original_size_range(problem: str, group: list[dict[str, str]], numeric_by_instance: dict[str, dict[str, Any]]) -> str:
    size_a = [numeric_by_instance[row["instance"]]["size_a"] for row in group]
    size_b = [numeric_by_instance[row["instance"]]["size_b"] for row in group]
    if problem == "MDKP":
        return f"{min(size_a)}-{max(size_a)} items; {min(size_b)}-{max(size_b)} dimensions"
    if problem == "MIS":
        return f"{min(size_a)}-{max(size_a)} nodes; {min(size_b)}-{max(size_b)} edges"
    if problem == "QAP":
        return f"n={min(size_a)}-{max(size_a)}"
    return f"{min(size_a)}-{max(size_a)} products; {min(size_b)}-{max(size_b)} retailers"


def build_summary_rows(
    rows: list[dict[str, str]], numeric_by_instance: dict[str, dict[str, Any]]
) -> list[dict[str, str]]:
    by_problem: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_problem[row["problem"]].append(row)

    out_rows: list[dict[str, str]] = []
    for problem in ["MDKP", "MIS", "QAP", "MSP"]:
        group = by_problem[problem]
        densities = [float(row["qubo_density"]) for row in group]
        degrees = [float(row["mean_qubo_degree"]) for row in group]
        max_degrees = [int(row["max_qubo_degree"]) for row in group]
        dynamic_ranges = [float(row["coefficient_dynamic_range"]) for row in group]
        penalty_ratios = [
            float(row["penalty_to_objective_ratio"])
            for row in group
            if row["penalty_to_objective_ratio"] != "not_applicable"
        ]
        qap_logs = [
            numeric_by_instance[row["instance"]]["qap_log10_feasible_fraction"]
            for row in group
            if numeric_by_instance[row["instance"]]["qap_log10_feasible_fraction"] is not None
        ]
        out_rows.append(
            {
                "problem": problem,
                "n_instances": fmt(len(group)),
                "source_library": "; ".join(sorted({row["source_library"] for row in group})),
                "original_problem_size_range": original_size_range(
                    problem, group, numeric_by_instance
                )
                if group
                else "not_applicable",
                "qubo_variable_range": range_text(
                    [int(row["qubo_variables"]) for row in group], digits=0
                ),
                "original_constraint_range": range_text(
                    [int(row["original_constraint_count"]) for row in group], digits=0
                ),
                "qubo_density_median": fmt(statistics.median(densities)),
                "qubo_density_min": fmt(min(densities)),
                "qubo_density_max": fmt(max(densities)),
                "mean_qubo_degree_median": fmt(statistics.median(degrees)),
                "max_qubo_degree_max": fmt(max(max_degrees)),
                "coefficient_dynamic_range_median": fmt(statistics.median(dynamic_ranges)),
                "coefficient_dynamic_range_max": fmt(max(dynamic_ranges)),
                "penalty_to_objective_ratio_median": fmt(
                    statistics.median(penalty_ratios) if penalty_ratios else None
                ),
                "qap_log10_feasible_fraction_range": range_text(qap_logs),
                "structural_interpretation": INTERPRETATIONS[problem],
            }
        )
    return out_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_digest(
    instance_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    qap_rows: list[dict[str, str]],
) -> None:
    summary_by_problem = {row["problem"]: row for row in summary_rows}
    dynamic_ranges = defaultdict(list)
    for row in instance_rows:
        dynamic_ranges[row["problem"]].append(float(row["coefficient_dynamic_range"]))

    qap_logs = {
        row["instance"]: row["log10_feasible_fraction"]
        for row in qap_rows
        if row["qap_dimension_n"] in {"10", "12"}
    }
    near_complete_instances = [
        row["instance"] for row in qap_rows if float(row["qubo_density"]) >= 0.9
    ]

    lines = [
        "# Structural Metrics Digest",
        "",
        "Generated from `build_structural_metrics.py` using the repository problem loaders "
        "and Qiskit's `QuadraticProgramToQubo` conversion path.",
        "",
        "## Density by problem",
    ]
    for problem in ["MDKP", "MIS", "QAP", "MSP"]:
        summary = summary_by_problem[problem]
        lines.append(
            f"- {problem}: median {summary['qubo_density_median']} "
            f"(range {summary['qubo_density_min']} to {summary['qubo_density_max']})."
        )

    lines.extend(["", "## Coefficient dynamic range by problem"])
    for problem in ["MDKP", "MIS", "QAP", "MSP"]:
        values = dynamic_ranges[problem]
        lines.append(f"- {problem}: range {fmt(min(values))} to {fmt(max(values))}.")

    lines.extend(["", "## QAP feasibility geometry"])
    for instance in ["tai10a", "tai10b"]:
        if instance in qap_logs:
            lines.append(f"- {instance}: log10 feasible fraction {qap_logs[instance]}.")
    n12_values = sorted(
        {
            row["log10_feasible_fraction"]
            for row in qap_rows
            if row["qap_dimension_n"] == "12"
        }
    )
    if n12_values:
        lines.append(f"- n=12 QAP instances: log10 feasible fraction {n12_values[0]}.")
    lines.append(
        "- QAP near-complete coupling density: "
        f"{', '.join(near_complete_instances) if near_complete_instances else 'none'} "
        "(threshold rho_Q >= 0.9)."
    )
    lines.append("- Smallest standard QAPLIB instances tested here: tai10a and tai10b (n=10).")
    lines.append(
        "- Reduced QAP calibration ladder: not run; this package only reports the required "
        "non-execution structural analyses."
    )

    (OUT / "structural_metrics_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    instance_rows: list[dict[str, str]] = []
    numeric_by_instance: dict[str, dict[str, Any]] = {}
    for item in collect_instances():
        row, numeric = build_instance_row(item)
        instance_rows.append(row)
        numeric_by_instance[item.instance_name] = numeric

    summary_rows = build_summary_rows(instance_rows, numeric_by_instance)
    qap_rows = build_qap_geometry_rows(instance_rows, numeric_by_instance)

    write_csv(OUT / "instance_structural_fingerprints.csv", INSTANCE_COLUMNS, instance_rows)
    write_csv(OUT / "problem_structural_summary.csv", SUMMARY_COLUMNS, summary_rows)
    write_csv(OUT / "qap_feasibility_geometry.csv", QAP_GEOMETRY_COLUMNS, qap_rows)
    write_digest(instance_rows, summary_rows, qap_rows)


if __name__ == "__main__":
    main()
