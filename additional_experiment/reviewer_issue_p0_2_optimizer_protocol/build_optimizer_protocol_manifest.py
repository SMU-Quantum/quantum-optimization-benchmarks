#!/usr/bin/env python3
from __future__ import annotations

import csv
import glob
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = EXPERIMENT_DIR

MANIFEST_COLUMNS = [
    "run_id",
    "problem",
    "instance",
    "execution_mode",
    "method",
    "ansatz_name",
    "ansatz_repetitions",
    "qaoa_depth",
    "num_trainable_parameters",
    "optimizer_name",
    "optimizer_library",
    "optimizer_version",
    "initialization_rule",
    "initialization_lower_bound",
    "initialization_upper_bound",
    "warm_start_source",
    "initialization_seed",
    "objective_shots_per_evaluation",
    "final_sampling_shots",
    "max_optimizer_iterations",
    "max_objective_evaluations",
    "function_tolerance",
    "parameter_tolerance",
    "gradient_tolerance",
    "finite_difference_step",
    "termination_reason",
    "realized_optimizer_iterations",
    "realized_objective_evaluations",
    "final_parameter_selection_rule",
    "postprocessing_rule",
    "wall_time_seconds",
    "artifact_replayable",
]

SUMMARY_COLUMNS = [
    "execution_mode",
    "method",
    "problem",
    "n_runs",
    "planned_max_iterations_min",
    "planned_max_iterations_max",
    "planned_max_evaluations_min",
    "planned_max_evaluations_max",
    "median_realized_iterations",
    "min_realized_iterations",
    "max_realized_iterations",
    "median_realized_evaluations",
    "min_realized_evaluations",
    "max_realized_evaluations",
    "fraction_budget_reached",
    "fraction_tolerance_reached",
    "fraction_optimizer_failure",
    "median_objective_shots",
    "median_final_sampling_shots",
]


def na() -> str:
    return "not_applicable"


def unknown() -> str:
    return "unknown_legacy_artifact"


def package_version(name: str) -> str:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return unknown()


def result_paths() -> list[Path]:
    patterns = [
        "research_benchmark/research_benchmark/results_hardware/**/result.json",
        "research_benchmark/research_benchmark/results_simulator/**/result.json",
    ]
    paths: list[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(str(REPO_ROOT / pattern), recursive=True))
    return sorted(paths)


def artifact_mode(path: Path, data: dict[str, Any]) -> str:
    if "results_simulator" in path.parts:
        prefix = "simulator"
    elif "results_hardware" in path.parts:
        prefix = "hardware"
    else:
        prefix = unknown()
    scheduler_mode = str(data.get("execution", {}).get("execution_mode") or data.get("config", {}).get("execution_mode") or unknown())
    return f"{prefix}_{scheduler_mode}"


def optimizer_info(method: str, data: dict[str, Any]) -> tuple[str, str, str]:
    if method == "qrao":
        name = str(data.get("config", {}).get("qrao_optimizer") or "cobyla")
        return name.upper(), "qiskit_algorithms", package_version("qiskit-algorithms")
    return "COBYLA", "scipy.optimize.minimize", package_version("scipy")


def init_info(method: str, data: dict[str, Any]) -> tuple[str, str, str, str]:
    config = data.get("config", {})
    seed = str(config.get("seed", unknown()))
    if method == "ws_qaoa":
        return "uniform_0_2pi_with_gamma_pi_beta_pi_over_2_override", "0", "2*pi", seed
    if method == "qrao":
        return "zero_initial_point", "0", "0", seed
    if method == "pce":
        return "uniform_0_1", "0", "1", seed
    return "uniform_0_2pi", "0", "2*pi", seed


def termination_reason(data: dict[str, Any]) -> str:
    opt = data.get("optimizer", {})
    msg = str(opt.get("message", "")).lower()
    status = str(opt.get("status", ""))
    if "maximum number of function evaluations" in msg or "max" in msg and "evalu" in msg:
        return "budget_reached"
    if "timeout" in msg:
        return "timeout"
    if "failed" in msg or "failure" in msg:
        return "optimizer_failure"
    if "terminated successfully" in msg:
        return "optimizer_success"
    if status in {"0", "1"} and msg and "finished" in msg:
        return "optimizer_success"
    if status not in {"0", "1", "2", "", "None"}:
        return "optimizer_failure"
    return unknown()


def qaoa_depth(method: str, config: dict[str, Any]) -> str:
    if method in {"qaoa", "cvar_qaoa", "ma_qaoa", "ws_qaoa"}:
        return str(config.get("layers", unknown()))
    return na()


def warm_start_source(method: str, data: dict[str, Any]) -> str:
    if method != "ws_qaoa":
        return na()
    meta = data.get("circuit_metrics", {}).get("ansatz_metadata", {})
    if isinstance(meta, dict):
        source = meta.get("warm_start", {}).get("source")
        if source:
            return str(source)
    return "continuous_relaxation_or_legacy_artifact"


def artifact_replayable(data: dict[str, Any]) -> str:
    metrics = data.get("circuit_metrics", {})
    method = str(data.get("execution", {}).get("method") or data.get("config", {}).get("method"))
    if method == "pce" and str(metrics.get("ansatz_family", "")).lower() == "brickwork":
        return "false"
    return "true"


def build_row(path: Path) -> dict[str, str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    execution = data.get("execution", {})
    metrics = data.get("circuit_metrics", {})
    protocol = data.get("benchmark_protocol", {})
    budget = protocol.get("budget", {})
    stopping = protocol.get("stopping_rule", {})
    opt = data.get("optimizer", {})
    method = str(execution.get("method") or config.get("method") or unknown())
    problem = str(data.get("problem") or config.get("problem") or unknown())
    instance = str(data.get("instance_name") or Path(str(config.get("instance", unknown()))).name)
    optimizer_name, optimizer_library, optimizer_version = optimizer_info(method, data)
    init_rule, init_lb, init_ub, init_seed = init_info(method, data)
    maxiter = str(config.get("maxiter", budget.get("max_optimizer_iterations", unknown())))
    realized = str(opt.get("total_evaluations", budget.get("total_circuit_evaluations", unknown())))
    wall_time = data.get("timing", {}).get("solve_runtime_sec", data.get("timing", {}).get("total_instance_sec", unknown()))
    ansatz_name = str(metrics.get("ansatz_family") or ("QRAO" if method == "qrao" else unknown()))
    ansatz_reps = str(metrics.get("ansatz_reps", config.get("qrao_reps", unknown() if method == "qrao" else na())))
    postprocess = protocol.get("feasibility_policy", {}).get("repair_method", "one_round_local_swap")
    return {
        "run_id": str(path.relative_to(REPO_ROOT)),
        "problem": problem,
        "instance": instance,
        "execution_mode": artifact_mode(path, data),
        "method": method,
        "ansatz_name": ansatz_name,
        "ansatz_repetitions": ansatz_reps,
        "qaoa_depth": qaoa_depth(method, config),
        "num_trainable_parameters": str(metrics.get("trainable_parameters", unknown())),
        "optimizer_name": optimizer_name,
        "optimizer_library": optimizer_library,
        "optimizer_version": optimizer_version,
        "initialization_rule": init_rule,
        "initialization_lower_bound": init_lb,
        "initialization_upper_bound": init_ub,
        "warm_start_source": warm_start_source(method, data),
        "initialization_seed": init_seed,
        "objective_shots_per_evaluation": str(budget.get("shots_per_circuit", config.get("shots", unknown()))),
        "final_sampling_shots": str(config.get("shots", budget.get("shots_per_circuit", unknown()))),
        "max_optimizer_iterations": maxiter,
        "max_objective_evaluations": maxiter,
        "function_tolerance": na(),
        "parameter_tolerance": na(),
        "gradient_tolerance": na(),
        "finite_difference_step": na(),
        "termination_reason": termination_reason(data),
        "realized_optimizer_iterations": str(stopping.get("actual_iterations", opt.get("total_evaluations", unknown()))),
        "realized_objective_evaluations": realized,
        "final_parameter_selection_rule": "best_objective_iterate",
        "postprocessing_rule": str(postprocess),
        "wall_time_seconds": str(wall_time),
        "artifact_replayable": artifact_replayable(data),
    }


def to_float(value: str) -> float | None:
    try:
        if value in {na(), unknown(), "", "None"}:
            return None
        return float(value)
    except Exception:
        return None


def median(values: list[float]) -> str:
    return str(statistics.median(values)) if values else unknown()


def min_s(values: list[float]) -> str:
    return str(min(values)) if values else unknown()


def max_s(values: list[float]) -> str:
    return str(max(values)) if values else unknown()


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    groups: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[(row["execution_mode"], row["method"], row["problem"])].append(row)
    out: list[dict[str, str]] = []
    for (execution_mode, method, problem), items in sorted(groups.items()):
        planned_iter = [v for v in (to_float(r["max_optimizer_iterations"]) for r in items) if v is not None]
        planned_eval = [v for v in (to_float(r["max_objective_evaluations"]) for r in items) if v is not None]
        realized_iter = [v for v in (to_float(r["realized_optimizer_iterations"]) for r in items) if v is not None]
        realized_eval = [v for v in (to_float(r["realized_objective_evaluations"]) for r in items) if v is not None]
        objective_shots = [v for v in (to_float(r["objective_shots_per_evaluation"]) for r in items) if v is not None]
        final_shots = [v for v in (to_float(r["final_sampling_shots"]) for r in items) if v is not None]
        n = len(items)
        out.append(
            {
                "execution_mode": execution_mode,
                "method": method,
                "problem": problem,
                "n_runs": str(n),
                "planned_max_iterations_min": min_s(planned_iter),
                "planned_max_iterations_max": max_s(planned_iter),
                "planned_max_evaluations_min": min_s(planned_eval),
                "planned_max_evaluations_max": max_s(planned_eval),
                "median_realized_iterations": median(realized_iter),
                "min_realized_iterations": min_s(realized_iter),
                "max_realized_iterations": max_s(realized_iter),
                "median_realized_evaluations": median(realized_eval),
                "min_realized_evaluations": min_s(realized_eval),
                "max_realized_evaluations": max_s(realized_eval),
                "fraction_budget_reached": str(sum(r["termination_reason"] == "budget_reached" for r in items) / n),
                "fraction_tolerance_reached": str(sum(r["termination_reason"] in {"function_tolerance", "parameter_tolerance", "gradient_tolerance"} for r in items) / n),
                "fraction_optimizer_failure": str(sum(r["termination_reason"] == "optimizer_failure" for r in items) / n),
                "median_objective_shots": median(objective_shots),
                "median_final_sampling_shots": median(final_shots),
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def write_digest(rows: list[dict[str, str]], summary: list[dict[str, str]]) -> None:
    total = len(rows)
    by_term: dict[str, int] = defaultdict(int)
    replay_false = 0
    for row in rows:
        by_term[row["termination_reason"]] += 1
        replay_false += int(row["artifact_replayable"] == "false")
    lines = [
        "# Optimizer Protocol Audit Digest",
        "",
        f"- Audited run artifacts: `{total}`.",
        f"- Non-replayable legacy artifacts: `{replay_false}`.",
        "",
        "## Termination Reasons",
        "",
    ]
    for key, value in sorted(by_term.items()):
        lines.append(f"- `{key}`: {value}")
    lines.extend(
        [
            "",
            "## Key Cautions",
            "",
            "- Fields not recoverable from artifacts are marked `unknown_legacy_artifact` or `not_applicable`.",
            "- The current artifacts show `maxiter=200` for PCE runs; most PCE runs reached the function-evaluation budget.",
            "- Legacy Brickwork PCE artifacts are marked `artifact_replayable=false` because no serialized circuit is saved and current source reconstructs a different PCE circuit.",
            "- No tolerance values are present in the saved result artifacts; the manifest therefore does not claim tolerance convergence.",
        ]
    )
    (OUT_DIR / "protocol_audit_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = [build_row(path) for path in result_paths()]
    summary = build_summary(rows)
    write_csv(OUT_DIR / "optimizer_protocol_manifest.csv", rows, MANIFEST_COLUMNS)
    write_csv(OUT_DIR / "optimizer_budget_summary.csv", summary, SUMMARY_COLUMNS)
    write_digest(rows, summary)
    print(f"Wrote {OUT_DIR / 'optimizer_protocol_manifest.csv'}")
    print(f"Wrote {OUT_DIR / 'optimizer_budget_summary.csv'}")
    print(f"Wrote {OUT_DIR / 'protocol_audit_digest.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
