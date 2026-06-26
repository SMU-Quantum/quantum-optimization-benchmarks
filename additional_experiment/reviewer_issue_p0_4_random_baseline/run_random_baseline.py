#!/usr/bin/env python3
"""Matched uniform-random baseline for low-fidelity QAOA-family hardware runs.

This script is intentionally offline: it reads saved hardware artifacts, rebuilds
the corresponding QUBOs through the repository problem loaders, and compares the
reported hardware selected candidate with a matched uniform random candidate
selection control.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC = REPO_ROOT / "research_benchmark" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qobench.hardware_cli import (  # noqa: E402
    _import_qubo_tools,
    _reconstruct_problem_objective,
    _run_one_round_local_swap,
)
from qobench.problems import MISProblem, MKPProblem, QAPProblem, MarketShareProblem  # noqa: E402
from qobench.quantum_methods import QuboObjective  # noqa: E402
from qobench.types import ProblemType  # noqa: E402


METHODS = {"qaoa", "ma_qaoa", "ws_qaoa"}
REPLICA_SEEDS = list(range(10001, 10301))
PROBLEM_LABEL = {
    "mkp": "MDKP",
    "mis": "MIS",
    "qap": "QAP",
    "market_share": "MSP",
}
PROBLEM_TYPE = {
    "mkp": ProblemType.MKP,
    "mis": ProblemType.MIS,
    "qap": ProblemType.QAP,
    "market_share": ProblemType.MARKET_SHARE,
}


LEDGER_COLUMNS = [
    "run_id",
    "problem",
    "instance",
    "method",
    "execution_mode",
    "fidelity_estimate",
    "n_decision_variables",
    "candidate_selection_scope",
    "optimizer_objective_evaluations",
    "objective_shots_per_evaluation",
    "final_sampling_shots",
    "eligible_candidate_batches",
    "eligible_candidate_count",
    "native_decoder",
    "candidate_selection_rule",
    "tie_breaking_rule",
    "feasibility_handling_before_selection",
    "shared_local_refinement_applied",
    "shared_local_refinement_rule",
    "reported_primary_metric",
    "reported_primary_value",
    "reported_feasible",
    "artifact_source",
    "notes",
]

REPLICATE_COLUMNS = [
    "run_id",
    "problem",
    "instance",
    "method",
    "execution_mode",
    "fidelity_estimate",
    "random_seed",
    "n_decision_variables",
    "eligible_candidate_count",
    "raw_selected_feasible",
    "raw_selected_primary_metric",
    "post_refinement_feasible",
    "post_refinement_primary_metric",
    "local_refinement_neighbor_evaluations",
    "local_refinement_accepted_moves",
    "local_refinement_runtime_seconds",
]

SUMMARY_COLUMNS = [
    "run_id",
    "problem",
    "instance",
    "method",
    "execution_mode",
    "fidelity_estimate",
    "n_decision_variables",
    "eligible_candidate_count",
    "hardware_primary_metric",
    "hardware_feasible",
    "random_feasibility_rate",
    "random_metric_median",
    "random_metric_p025",
    "random_metric_p975",
    "hardware_random_tail_fraction",
    "hardware_vs_random_classification",
    "median_random_local_refinement_improvement",
    "notes",
]

GROUP_COLUMNS = [
    "problem",
    "method",
    "n_low_fidelity_hardware_runs",
    "n_hardware_feasible",
    "median_fidelity_estimate",
    "median_eligible_candidate_count",
    "median_hardware_primary_metric",
    "median_random_primary_metric",
    "median_hardware_random_tail_fraction",
    "fraction_outperforms_matched_uniform_random",
    "fraction_random_comparable",
    "fraction_underperforms_matched_uniform_random",
    "median_random_feasibility_rate",
]


@dataclass(slots=True)
class RunRecord:
    run_id: str
    problem_key: str
    problem: str
    instance: str
    method: str
    artifact: Path
    data: dict[str, Any]
    fidelity_estimate: float
    n_vars: int
    evals: int
    shots: int
    eligible_count: int
    hardware_feasible: bool
    hardware_metric: float
    optimum: float


@dataclass(slots=True)
class ProblemCache:
    instance: Any
    objective: QuboObjective
    variable_names_in_order: list[str]
    linear: np.ndarray
    quadratic: np.ndarray
    constant: float


def _safe_float(value: Any, default: float = math.nan) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isinf(value):
            return "inf"
        if math.isnan(value):
            return "nan"
        return f"{value:.12g}"
    if isinstance(value, bool):
        return "True" if value else "False"
    return str(value)


def _median(values: list[float]) -> float:
    clean = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(clean)) if clean else math.nan


def _percentile(values: list[float], q: float) -> float:
    clean = np.asarray([float(v) for v in values if math.isfinite(float(v))], dtype=float)
    if clean.size == 0:
        return math.inf
    return float(np.percentile(clean, q))


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _instance_stem(name: str) -> str:
    return Path(name).stem.replace(".txt", "").replace(".dat", "")


def _parse_optima(results_root: Path) -> dict[tuple[str, str], float]:
    optima: dict[tuple[str, str], float] = {}
    for table in results_root.glob("*/*/csv/main_table*.csv"):
        problem_key = table.parts[-4]
        with table.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames or "Instance" not in reader.fieldnames or "Optimal" not in reader.fieldnames:
                continue
            for row in reader:
                inst = str(row.get("Instance") or "").strip()
                if not inst:
                    continue
                opt = _safe_float(row.get("Optimal"))
                if math.isfinite(opt):
                    optima[(problem_key, inst)] = opt
    return optima


def _method_from_artifact(data: dict[str, Any]) -> str:
    execution = data.get("execution", {})
    config = data.get("config", {})
    return str(config.get("method") or execution.get("requested_method") or execution.get("method") or "")


def _problem_from_artifact(path: Path, data: dict[str, Any]) -> str:
    value = str(data.get("problem") or "")
    if value:
        return value
    parts = list(path.parts)
    for candidate in ("mkp", "mis", "qap", "market_share"):
        if candidate in parts:
            return candidate
    raise ValueError(f"Could not infer problem for {path}")


def _hardware_metric(problem_key: str, reconstructed: dict[str, Any], optimum: float) -> float:
    if not bool(reconstructed.get("feasible")):
        return math.inf
    value = _safe_float(reconstructed.get("objective_value"))
    if not math.isfinite(value):
        return math.inf
    if problem_key == "market_share":
        return value
    if not math.isfinite(optimum) or optimum == 0:
        return math.inf
    if problem_key in {"mis", "mkp"}:
        return 100.0 * (float(optimum) - value) / float(optimum)
    return 100.0 * (value - float(optimum)) / float(optimum)


def _fidelity_proxy(data: dict[str, Any]) -> float:
    """Reconstruct a low-fidelity selection estimate from saved gate counts.

    The historical artifacts do not retain per-gate calibration errors.  This
    conservative proxy uses typical order-of-magnitude error rates only for
    selecting the low-fidelity subset: 0.1% per 1q gate, 1% per 2q gate, and
    2% per measured qubit.
    """

    metadata = data.get("job_metadata") or []
    if isinstance(metadata, dict):
        metadata = [metadata]
    one_q: list[float] = []
    two_q: list[float] = []
    meas: list[float] = []
    for entry in metadata:
        if not isinstance(entry, dict):
            continue
        if entry.get("transpiled_1q_gates") is not None:
            one_q.append(float(entry["transpiled_1q_gates"]))
        if entry.get("transpiled_2q_gates") is not None:
            two_q.append(float(entry["transpiled_2q_gates"]))
        if entry.get("transpiled_measurements") is not None:
            meas.append(float(entry["transpiled_measurements"]))
    exposure = 0.001 * _median(one_q) + 0.01 * _median(two_q) + 0.02 * _median(meas)
    if not math.isfinite(exposure):
        return math.nan
    return float(math.exp(-exposure))


def _discover_runs(results_root: Path, optima: dict[tuple[str, str], float], threshold: float) -> list[RunRecord]:
    records: list[RunRecord] = []
    for artifact in sorted(results_root.rglob("result.json")):
        data = _read_json(artifact)
        method = _method_from_artifact(data)
        if method not in METHODS:
            continue
        problem_key = _problem_from_artifact(artifact, data)
        if problem_key not in PROBLEM_LABEL:
            continue
        fidelity = _fidelity_proxy(data)
        if not math.isfinite(fidelity) or fidelity >= threshold:
            continue
        instance_name = str(data.get("instance_name") or data.get("instance_path") or artifact.parent.name)
        instance = _instance_stem(instance_name)
        optimum = optima.get((problem_key, instance))
        if optimum is None:
            raise ValueError(f"No optimum found for {problem_key}/{instance} from {artifact}")
        best = data.get("best_result") or {}
        reconstructed = best.get("reconstructed_problem_objective") or {}
        n_vars = int(data.get("logical_num_qubits") or data.get("qubits") or best.get("num_qubits") or 0)
        if n_vars <= 0:
            bitstring = str(best.get("best_bitstring") or best.get("raw_best_bitstring") or "")
            n_vars = len(bitstring)
        budget = data.get("benchmark_protocol", {}).get("budget", {})
        evals = int(data.get("optimizer", {}).get("total_evaluations") or budget.get("total_circuit_evaluations") or 0)
        shots = int(budget.get("shots_per_circuit") or data.get("config", {}).get("shots") or 1000)
        run_id = f"{problem_key}_{method}_{instance}".replace(".", "_")
        hardware_feasible = bool(best.get("feasible") if "feasible" in best else reconstructed.get("feasible"))
        hardware_metric = _hardware_metric(problem_key, reconstructed, float(optimum))
        records.append(
            RunRecord(
                run_id=run_id,
                problem_key=problem_key,
                problem=PROBLEM_LABEL[problem_key],
                instance=instance,
                method=method,
                artifact=artifact,
                data=data,
                fidelity_estimate=fidelity,
                n_vars=n_vars,
                evals=evals,
                shots=shots,
                eligible_count=evals * shots,
                hardware_feasible=hardware_feasible,
                hardware_metric=hardware_metric,
                optimum=float(optimum),
            )
        )
    return records


def _load_problem(record: RunRecord) -> ProblemCache:
    from_docplex_mp, QuadraticProgramToQubo = _import_qubo_tools()
    if record.problem_key == "mis":
        problem = MISProblem()
        instance = problem.load_instance(REPO_ROOT / "Maximum_Independent_Set" / "mis_benchmark_instances" / f"{record.instance}.txt")
    elif record.problem_key == "mkp":
        problem = MKPProblem()
        instance = problem.load_instance(REPO_ROOT / "Multi_Dimension_Knapsack" / "MKP_Instances" / "hpp" / f"{record.instance}.dat")
    elif record.problem_key == "qap":
        problem = QAPProblem()
        instance = problem.load_instance(REPO_ROOT / "Quadratic_Assignment_Problem" / "qapdata" / f"{record.instance}.dat")
    elif record.problem_key == "market_share":
        problem = MarketShareProblem()
        parsed = MarketShareProblem.parse_generated_instance_name(f"{record.instance}.gen")
        if parsed is None:
            raise ValueError(f"Cannot parse generated market-share instance {record.instance}")
        seed, num_products = parsed
        instance = problem.load_instance(None, seed=seed, num_products=num_products)
    else:
        raise ValueError(record.problem_key)
    model, _ = problem.build_model(instance=instance, time_limit_sec=1.0)
    qp = from_docplex_mp(model)
    qubo = QuadraticProgramToQubo().convert(qp)
    objective = QuboObjective(qubo)
    variable_names_in_order = [
        name for name, _ in sorted(qubo.variables_index.items(), key=lambda item: int(item[1]))
    ]
    linear = np.asarray(qubo.objective.linear.to_array(), dtype=np.float64)
    quadratic = np.asarray(qubo.objective.quadratic.to_array(symmetric=False), dtype=np.float64)
    constant = float(qubo.objective.constant)
    return ProblemCache(
        instance=instance,
        objective=objective,
        variable_names_in_order=variable_names_in_order,
        linear=linear,
        quadratic=quadratic,
        constant=constant,
    )


def _energies(bits: np.ndarray, cache: ProblemCache) -> np.ndarray:
    x = bits.astype(np.float64, copy=False)
    values = cache.constant + x @ cache.linear
    values = values + np.sum((x @ cache.quadratic) * x, axis=1)
    return np.asarray(values, dtype=np.float64)


def _bitstring_from_row(row: np.ndarray) -> str:
    return "".join("1" if int(v) else "0" for v in row.tolist())


def _assignment_from_bits(row: np.ndarray, cache: ProblemCache) -> dict[str, int]:
    return {name: int(row[idx]) for idx, name in enumerate(cache.variable_names_in_order)}


def _select_random_candidate(record: RunRecord, cache: ProblemCache, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    best_batch_objective = math.inf
    best_batch_bits: np.ndarray | None = None
    best_batch_energies: np.ndarray | None = None
    chunk_evals = max(1, min(25, record.evals))
    remaining = int(record.evals)
    while remaining > 0:
        current = min(chunk_evals, remaining)
        bits = rng.integers(
            0,
            2,
            size=(current, record.shots, cache.objective.num_qubits),
            dtype=np.int8,
        )
        flat_bits = bits.reshape(current * record.shots, cache.objective.num_qubits)
        energies = _energies(flat_bits, cache).reshape(current, record.shots)
        batch_objectives = np.mean(energies, axis=1)
        chunk_best = int(np.argmin(batch_objectives))
        chunk_value = float(batch_objectives[chunk_best])
        if chunk_value < best_batch_objective:
            best_batch_objective = chunk_value
            best_batch_bits = bits[chunk_best].copy()
            best_batch_energies = energies[chunk_best].copy()
        remaining -= current
    if best_batch_bits is None or best_batch_energies is None:
        raise RuntimeError(f"No random candidates generated for {record.run_id}")
    min_energy = float(np.min(best_batch_energies))
    ties = np.where(np.isclose(best_batch_energies, min_energy))[0]
    if ties.size == 1:
        return best_batch_bits[int(ties[0])].copy()
    tied = best_batch_bits[ties]
    unique, counts = np.unique(tied, axis=0, return_counts=True)
    return unique[int(np.argmax(counts))].copy()


def _replicate_qap_feasibility_only(record: RunRecord, seed: int) -> dict[str, Any]:
    return {
        "run_id": record.run_id,
        "problem": record.problem,
        "instance": record.instance,
        "method": record.method,
        "execution_mode": "hardware",
        "fidelity_estimate": record.fidelity_estimate,
        "random_seed": seed,
        "n_decision_variables": record.n_vars,
        "eligible_candidate_count": record.eligible_count,
        "raw_selected_feasible": False,
        "raw_selected_primary_metric": math.inf,
        "post_refinement_feasible": False,
        "post_refinement_primary_metric": math.inf,
        "local_refinement_neighbor_evaluations": 0,
        "local_refinement_accepted_moves": 0,
        "local_refinement_runtime_seconds": 0.0,
    }


def _run_replicate(record: RunRecord, cache: ProblemCache, seed: int) -> dict[str, Any]:
    selected = _select_random_candidate(record, cache, seed)
    raw_assignment = _assignment_from_bits(selected, cache)
    raw_reconstructed = _reconstruct_problem_objective(
        problem=PROBLEM_TYPE[record.problem_key],
        instance=cache.instance,
        assignment=raw_assignment,
    )
    start = time.perf_counter()
    local = _run_one_round_local_swap(
        problem=PROBLEM_TYPE[record.problem_key],
        instance=cache.instance,
        assignment=raw_assignment,
        variable_names_in_order=cache.variable_names_in_order,
    )
    runtime = time.perf_counter() - start
    post_reconstructed = dict(local["reconstructed"])
    raw_metric = _hardware_metric(record.problem_key, raw_reconstructed, record.optimum)
    post_metric = _hardware_metric(record.problem_key, post_reconstructed, record.optimum)
    return {
        "run_id": record.run_id,
        "problem": record.problem,
        "instance": record.instance,
        "method": record.method,
        "execution_mode": "hardware",
        "fidelity_estimate": record.fidelity_estimate,
        "random_seed": seed,
        "n_decision_variables": record.n_vars,
        "eligible_candidate_count": record.eligible_count,
        "raw_selected_feasible": bool(raw_reconstructed.get("feasible")),
        "raw_selected_primary_metric": raw_metric,
        "post_refinement_feasible": bool(post_reconstructed.get("feasible")),
        "post_refinement_primary_metric": post_metric,
        "local_refinement_neighbor_evaluations": int(local.get("candidates_checked", 0)),
        "local_refinement_accepted_moves": 1 if bool(local.get("improved")) else 0,
        "local_refinement_runtime_seconds": float(runtime),
    }


def _ledger_row(record: RunRecord) -> dict[str, Any]:
    policy = record.data.get("benchmark_protocol", {}).get("feasibility_policy", {})
    execution_method = str(record.data.get("execution", {}).get("method") or "")
    fallback_note = ""
    if execution_method and execution_method != record.method:
        fallback_note = f"; effective execution method recorded in artifact was {execution_method}"
    primary_metric = "TDev" if record.problem_key == "market_share" else "optimality_gap_percent"
    return {
        "run_id": record.run_id,
        "problem": record.problem,
        "instance": record.instance,
        "method": record.method,
        "execution_mode": "hardware",
        "fidelity_estimate": record.fidelity_estimate,
        "n_decision_variables": record.n_vars,
        "candidate_selection_scope": "optimizer_trajectory_only",
        "optimizer_objective_evaluations": record.evals,
        "objective_shots_per_evaluation": record.shots,
        "final_sampling_shots": 0,
        "eligible_candidate_batches": record.evals,
        "eligible_candidate_count": record.eligible_count,
        "native_decoder": "qobench.QuboObjective.assignment",
        "candidate_selection_rule": "select optimizer-evaluation batch with minimum expectation objective, then lowest-QUBO-energy bitstring within that batch",
        "tie_breaking_rule": "earliest lower optimizer value because code updates only on '<'; within a counts batch, lower energy then larger count",
        "feasibility_handling_before_selection": "none; feasibility repair is not applied before candidate selection",
        "shared_local_refinement_applied": bool(policy.get("repair_allowed", True)),
        "shared_local_refinement_rule": str(policy.get("repair_method") or "one_round_local_swap"),
        "reported_primary_metric": primary_metric,
        "reported_primary_value": record.hardware_metric,
        "reported_feasible": record.hardware_feasible,
        "artifact_source": str(record.artifact.relative_to(REPO_ROOT)),
        "notes": "fidelity_estimate is a gate-count proxy reconstructed because saved artifacts omit calibration error rates" + fallback_note,
    }


def _write_csv(path: Path, columns: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: _fmt(row.get(column, "")) for column in columns})


def _summarize(record: RunRecord, rows: list[dict[str, Any]]) -> dict[str, Any]:
    post_metrics = [float(r["post_refinement_primary_metric"]) for r in rows]
    raw_metrics = [float(r["raw_selected_primary_metric"]) for r in rows]
    feasible = [bool(r["post_refinement_feasible"]) for r in rows]
    finite_post = [v for v in post_metrics if math.isfinite(v)]
    if record.hardware_feasible:
        count_leq = sum(1 for v in post_metrics if v <= record.hardware_metric)
        tail = (1.0 + float(count_leq)) / 301.0
        if tail <= 0.05:
            classification = "outperforms_matched_uniform_random"
        elif tail >= 0.95:
            classification = "underperforms_matched_uniform_random"
        else:
            classification = "random_comparable"
    else:
        tail = math.nan
        classification = "hardware_infeasible_random_feasibility_comparison_only"
    improvements = []
    for raw, post in zip(raw_metrics, post_metrics):
        if math.isfinite(raw) and math.isfinite(post):
            improvements.append(raw - post)
    return {
        "run_id": record.run_id,
        "problem": record.problem,
        "instance": record.instance,
        "method": record.method,
        "execution_mode": "hardware",
        "fidelity_estimate": record.fidelity_estimate,
        "n_decision_variables": record.n_vars,
        "eligible_candidate_count": record.eligible_count,
        "hardware_primary_metric": record.hardware_metric,
        "hardware_feasible": record.hardware_feasible,
        "random_feasibility_rate": sum(feasible) / float(len(feasible)) if feasible else math.nan,
        "random_metric_median": _median(finite_post),
        "random_metric_p025": _percentile(finite_post, 2.5),
        "random_metric_p975": _percentile(finite_post, 97.5),
        "hardware_random_tail_fraction": tail,
        "hardware_vs_random_classification": classification,
        "median_random_local_refinement_improvement": _median(improvements),
        "notes": "QAP rows are feasibility-only analytical controls; all hardware QAP runs are infeasible" if record.problem_key == "qap" else "",
    }


def _group_summary(summary_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in summary_rows:
        grouped.setdefault((str(row["problem"]), str(row["method"])), []).append(row)
    out: list[dict[str, Any]] = []
    for (problem, method), rows in sorted(grouped.items()):
        feasible_rows = [r for r in rows if bool(r["hardware_feasible"])]
        finite_hw = [float(r["hardware_primary_metric"]) for r in feasible_rows if math.isfinite(float(r["hardware_primary_metric"]))]
        finite_random = [float(r["random_metric_median"]) for r in rows if math.isfinite(float(r["random_metric_median"]))]
        finite_tail = [float(r["hardware_random_tail_fraction"]) for r in rows if math.isfinite(float(r["hardware_random_tail_fraction"]))]
        counts = {str(r["hardware_vs_random_classification"]): 0 for r in rows}
        for r in rows:
            counts[str(r["hardware_vs_random_classification"])] = counts.get(str(r["hardware_vs_random_classification"]), 0) + 1
        denom = float(len(rows)) if rows else 1.0
        out.append(
            {
                "problem": problem,
                "method": method,
                "n_low_fidelity_hardware_runs": len(rows),
                "n_hardware_feasible": len(feasible_rows),
                "median_fidelity_estimate": _median([float(r["fidelity_estimate"]) for r in rows]),
                "median_eligible_candidate_count": _median([float(r["eligible_candidate_count"]) for r in rows]),
                "median_hardware_primary_metric": _median(finite_hw),
                "median_random_primary_metric": _median(finite_random),
                "median_hardware_random_tail_fraction": _median(finite_tail),
                "fraction_outperforms_matched_uniform_random": counts.get("outperforms_matched_uniform_random", 0) / denom,
                "fraction_random_comparable": counts.get("random_comparable", 0) / denom,
                "fraction_underperforms_matched_uniform_random": counts.get("underperforms_matched_uniform_random", 0) / denom,
                "median_random_feasibility_rate": _median([float(r["random_feasibility_rate"]) for r in rows]),
            }
        )
    return out


def _plot(summary_rows: list[dict[str, Any]], out_path: Path) -> None:
    panels = [
        ("MDKP", "MDKP", "Optimality gap (%)"),
        ("MIS", "MIS", "Optimality gap (%)"),
        ("MSP", "MSP", "TDev"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.2), sharey=False)
    colors = {"qaoa": "#2f6f9f", "ma_qaoa": "#5f8f3a", "ws_qaoa": "#a34f4f"}
    for ax, (problem, title, ylabel) in zip(axes, panels):
        rows = [r for r in summary_rows if r["problem"] == problem]
        rows.sort(key=lambda r: (str(r["method"]), str(r["instance"])))
        labels = [_compact_plot_label(r) for r in rows]
        x = np.arange(len(rows), dtype=float)
        random_medians = np.asarray([float(r["random_metric_median"]) for r in rows], dtype=float)
        random_lo = np.asarray([float(r["random_metric_p025"]) for r in rows], dtype=float)
        random_hi = np.asarray([float(r["random_metric_p975"]) for r in rows], dtype=float)
        hardware = np.asarray([
            float(r["hardware_primary_metric"]) if math.isfinite(float(r["hardware_primary_metric"])) else np.nan
            for r in rows
        ], dtype=float)
        bar_colors = [colors.get(str(r["method"]), "#777777") for r in rows]
        finite = np.concatenate([
            random_medians[np.isfinite(random_medians)],
            random_lo[np.isfinite(random_lo)],
            random_hi[np.isfinite(random_hi)],
            hardware[np.isfinite(hardware)],
        ])
        y_top = float(np.max(finite) * 1.10) if finite.size else 1.0
        y_top = y_top if y_top > 0.0 else 1.0
        for idx in range(len(rows)):
            if math.isfinite(random_medians[idx]) and math.isfinite(random_lo[idx]) and math.isfinite(random_hi[idx]):
                lower = max(0.0, random_medians[idx] - random_lo[idx])
                upper = max(0.0, random_hi[idx] - random_medians[idx])
                ax.errorbar(
                    x[idx],
                    random_medians[idx],
                    yerr=np.array([[lower], [upper]]),
                    fmt="o",
                    mfc="white",
                    mec="#333333",
                    ecolor="#777777",
                    elinewidth=0.8,
                    capsize=2.5,
                    markersize=4.5,
                    zorder=2,
                )
            if math.isfinite(hardware[idx]) and math.isfinite(random_medians[idx]):
                ax.plot([x[idx], x[idx]], [random_medians[idx], hardware[idx]], color="#b0b0b0", linewidth=0.6, zorder=1)
            if math.isfinite(hardware[idx]):
                ax.scatter(x[idx], hardware[idx], s=22, color=bar_colors[idx], edgecolor="black", linewidth=0.35, zorder=3)
            else:
                ax.scatter(x[idx], y_top, marker="v", s=28, color=bar_colors[idx], edgecolor="black", linewidth=0.35, zorder=3)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.grid(axis="y", alpha=0.25, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.set_ylim(bottom=0.0, top=y_top * 1.05)
    handles = [
        plt.Line2D([0], [0], marker="o", color="none", markerfacecolor="#333333", markeredgecolor="black", markersize=5, label="hardware"),
        plt.Line2D([0], [0], marker="o", color="#777777", markerfacecolor="white", markeredgecolor="#333333", markersize=5, label="random median (2.5-97.5%)"),
        plt.Line2D([0], [0], marker="v", color="none", markerfacecolor="#777777", markeredgecolor="black", markersize=5, label="hardware infeasible"),
    ]
    axes[0].legend(handles=handles, loc="upper left", fontsize=7, frameon=False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def _compact_plot_label(row: dict[str, Any]) -> str:
    method = str(row["method"])
    prefix = {"ma_qaoa": "MA", "qaoa": "Q", "ws_qaoa": "WS"}.get(method, method)
    instance = str(row["instance"])
    if instance.startswith("ms_seed"):
        import re

        match = re.match(r"ms_seed(\d+)_prod(\d+)", instance)
        if match:
            seed, product = match.groups()
            return f"{prefix}-ms{product}{seed}"
    return f"{prefix}-{instance.replace('_', '.')}"


def _digest(records: list[RunRecord], summary_rows: list[dict[str, Any]], group_rows: list[dict[str, Any]], out_path: Path) -> None:
    classifications = {key: 0 for key in [
        "outperforms_matched_uniform_random",
        "random_comparable",
        "underperforms_matched_uniform_random",
        "hardware_infeasible_random_feasibility_comparison_only",
    ]}
    for row in summary_rows:
        classifications[str(row["hardware_vs_random_classification"])] = classifications.get(str(row["hardware_vs_random_classification"]), 0) + 1
    by_problem = {}
    for row in group_rows:
        by_problem.setdefault(row["problem"], []).append(row)
    median_improvement = _median([
        float(row["median_random_local_refinement_improvement"])
        for row in summary_rows
        if math.isfinite(float(row["median_random_local_refinement_improvement"]))
    ])
    candidate_counts = sorted({r.eligible_count for r in records})
    lines = [
        "# P0.4 Random Sampling Baseline Digest",
        "",
        "## Scope",
        f"- Low-fidelity QAOA-family hardware runs included: {len(records)}.",
        "- Methods included: QAOA, MA-QAOA, WS-QAOA.",
        "- Methods excluded: PCE and QRAO, per reviewer-response scope.",
        "- Low-fidelity filter: reconstructed gate-count fidelity proxy `< 1e-3`; the saved artifacts do not retain per-gate calibration error rates.",
        "",
        "## Six Reviewer Questions",
        "1. Candidate pool eligible: optimizer-trajectory batches only. The code stores the best counts from the optimizer evaluation with the lowest expectation objective; there is no independent final sampling batch in these artifacts.",
        f"2. Number of low-fidelity QAOA-family hardware runs: {len(records)}.",
        f"3. Outperform matched uniform random: {classifications.get('outperforms_matched_uniform_random', 0)} runs.",
        f"4. Random-comparable: {classifications.get('random_comparable', 0)} runs.",
        f"5. Local refinement median random-baseline improvement: {median_improvement:.6g} primary-metric units across finite rows.",
        "6. Problem dependence: see group summary below; QAP is feasibility-only because all hardware QAP runs are infeasible and valid uniform random permutations are astronomically unlikely.",
        "",
        "## Candidate Counts",
        "- Unique eligible candidate counts: " + ", ".join(str(v) for v in candidate_counts),
        "",
        "## Classification Counts",
    ]
    for key, value in classifications.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Group Summary"])
    for row in group_rows:
        lines.append(
            "- {problem}/{method}: n={n}, feasible={feas}, random_median={rand}, tail_median={tail}, comparable_fraction={comp}".format(
                problem=row["problem"],
                method=row["method"],
                n=row["n_low_fidelity_hardware_runs"],
                feas=row["n_hardware_feasible"],
                rand=_fmt(row["median_random_primary_metric"]),
                tail=_fmt(row["median_hardware_random_tail_fraction"]),
                comp=_fmt(row["fraction_random_comparable"]),
            )
        )
    lines.extend([
        "",
        "## Interpretation Note",
        "The matched random control addresses the low-fidelity / best-shot-selection concern directly: it asks whether the reported hardware candidate is better than selecting from the same number of uniformly random bitstrings under the same downstream local refinement. Because calibration fields are absent from the saved artifacts, the fidelity column should be described as a reconstructed selection proxy rather than a measured process fidelity.",
        "",
    ])
    out_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--threshold", type=float, default=1e-3)
    parser.add_argument("--replicates", type=int, default=300)
    parser.add_argument("--max-runs", type=int, default=None)
    args = parser.parse_args()

    out_dir = args.output_dir.resolve()
    results_root = REPO_ROOT / "research_benchmark" / "research_benchmark" / "results_hardware"
    optima = _parse_optima(results_root)
    records = _discover_runs(results_root, optima, threshold=float(args.threshold))
    if args.max_runs is not None:
        records = records[: int(args.max_runs)]
    records.sort(key=lambda r: (r.problem, r.method, r.instance))

    ledger_rows = [_ledger_row(record) for record in records]
    _write_csv(out_dir / "random_baseline_selection_ledger.csv", LEDGER_COLUMNS, ledger_rows)

    seeds = REPLICA_SEEDS[: int(args.replicates)]
    replicate_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    cache_by_instance: dict[tuple[str, str], ProblemCache] = {}
    for idx, record in enumerate(records, start=1):
        print(f"[{idx}/{len(records)}] {record.run_id} candidates={record.eligible_count}", flush=True)
        rows: list[dict[str, Any]] = []
        if record.problem_key == "qap":
            rows = [_replicate_qap_feasibility_only(record, seed) for seed in seeds]
        else:
            key = (record.problem_key, record.instance)
            if key not in cache_by_instance:
                cache_by_instance[key] = _load_problem(record)
            cache = cache_by_instance[key]
            for seed in seeds:
                rows.append(_run_replicate(record, cache, seed))
        replicate_rows.extend(rows)
        summary_rows.append(_summarize(record, rows))

    _write_csv(out_dir / "random_baseline_replicates.csv", REPLICATE_COLUMNS, replicate_rows)
    _write_csv(out_dir / "random_baseline_summary.csv", SUMMARY_COLUMNS, summary_rows)
    group_rows = _group_summary(summary_rows)
    _write_csv(out_dir / "random_baseline_group_summary.csv", GROUP_COLUMNS, group_rows)
    _plot(summary_rows, out_dir / "plots" / "fig_low_fidelity_random_baseline.pdf")
    _digest(records, summary_rows, group_rows, out_dir / "random_baseline_digest.md")
    print(f"Wrote outputs under {out_dir}")


if __name__ == "__main__":
    main()
