#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
RUNNER = REPO_ROOT / "research_benchmark" / "run_simulator_benchmark.py"
PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

INIT_SEEDS = [1103, 4409, 7703]

CASES = [
    {
        "case_id": "R1_mdkp_hp1_vqe",
        "problem": "mkp",
        "instance": "Multi_Dimension_Knapsack/MKP_Instances/hpp/hp1.dat",
        "instance_label": "hp1.dat",
        "method": "vqe",
        "why": "MDKP direct variational baseline; 60 logical qubits.",
    },
    {
        "case_id": "R2_mis_1tc32_ws_qaoa",
        "problem": "mis",
        "instance": "Maximum_Independent_Set/mis_benchmark_instances/1tc.32.txt",
        "instance_label": "1tc.32.txt",
        "method": "ws_qaoa",
        "why": "Structured QAOA initialization case on a nontrivial MIS instance.",
    },
    {
        "case_id": "R3_mis_1tc16_pce",
        "problem": "mis",
        "instance": "Maximum_Independent_Set/mis_benchmark_instances/1tc.16.txt",
        "instance_label": "1tc.16.txt",
        "method": "pce",
        "why": "Current reproducible PCE/encoding case; avoids legacy MDKP Brickwork artifact.",
    },
]

REPLICATE_COLUMNS = [
    "case_id",
    "problem",
    "instance",
    "method",
    "initialization_seed",
    "final_sampling_seed",
    "optimizer_name",
    "max_optimizer_iterations",
    "max_objective_evaluations",
    "realized_optimizer_iterations",
    "realized_objective_evaluations",
    "termination_reason",
    "initial_energy",
    "best_energy",
    "final_energy",
    "final_decoded_gap_percent",
    "final_tdev",
    "final_mdev",
    "decoded_feasible",
    "runtime_seconds",
]

SUMMARY_COLUMNS = [
    "case_id",
    "problem",
    "instance",
    "method",
    "n_initializations",
    "n_final_sampling_repetitions_per_initialization",
    "median_best_energy",
    "iqr_best_energy",
    "mean_primary_quality",
    "sd_primary_quality",
    "median_primary_quality",
    "best_primary_quality",
    "worst_primary_quality",
    "feasible_initialization_fraction",
    "median_realized_objective_evaluations",
    "min_realized_objective_evaluations",
    "max_realized_objective_evaluations",
    "dominant_termination_reason",
]

TRACE_COLUMNS = [
    "case_id",
    "problem",
    "instance",
    "method",
    "initialization_seed",
    "budget_multiplier",
    "objective_evaluation_index",
    "optimizer_iteration_index",
    "objective_value",
    "best_objective_so_far",
    "cumulative_runtime_seconds",
]


def termination_reason(message: str) -> str:
    msg = str(message).lower()
    if "maximum number of function evaluations" in msg:
        return "budget_reached"
    if "terminated successfully" in msg:
        return "optimizer_success"
    if "fail" in msg:
        return "optimizer_failure"
    if "timeout" in msg:
        return "timeout"
    return "unknown_legacy_artifact"


def run_one(case: dict[str, str], seed: int, out_root: Path, maxiter: int, shots: int) -> Path:
    case_out = out_root / "raw_runs" / case["case_id"] / f"seed_{seed}"
    ckpt = out_root / "checkpoints" / case["case_id"] / f"seed_{seed}"
    cmd = [
        str(PYTHON),
        str(RUNNER),
        "--problem",
        case["problem"],
        "--method",
        case["method"],
        "--instance",
        case["instance"],
        "--seed",
        str(seed),
        "--shots",
        str(shots),
        "--maxiter",
        str(maxiter),
        "--output-root",
        str(case_out),
        "--checkpoint-dir",
        str(ckpt),
        "--force-rerun",
        "--log-level",
        "INFO",
    ]
    started = time.perf_counter()
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    elapsed = time.perf_counter() - started
    result_paths = sorted(case_out.glob("**/result.json"), key=lambda p: p.stat().st_mtime)
    if not result_paths:
        raise FileNotFoundError(f"No result.json produced for {case['case_id']} seed={seed}")
    result_path = result_paths[-1]
    marker = result_path.parent / "reduced_part_b_runtime.json"
    marker.write_text(json.dumps({"wrapper_runtime_seconds": elapsed}, indent=2), encoding="utf-8")
    return result_path


def bks_for_case(case: dict[str, str]) -> float | None:
    if case["problem"] == "mkp":
        path = REPO_ROOT / case["instance"]
        tokens = [int(tok) for tok in path.read_text(encoding="utf-8").split()]
        return float(tokens[2])
    if case["problem"] == "mis":
        summary_path = REPO_ROOT / "classical_solutions" / "results" / "mis" / "summary.json"
        if summary_path.exists():
            data = json.loads(summary_path.read_text(encoding="utf-8"))
            rows = data if isinstance(data, list) else data.get("instances", data.get("results", []))
            for row in rows:
                name = str(row.get("instance") or row.get("instance_name") or "")
                if name == case["instance_label"]:
                    for key in ("objective_value", "optimal_value", "best_objective", "cardinality"):
                        if key in row:
                            return float(row[key])
        # Fallback for the DIMACS toy family if summary shape changes.
        details = REPO_ROOT / "classical_solutions" / "results" / "mis" / "details" / f"{case['instance_label'].replace('.txt', '')}_details.txt"
        if details.exists():
            import re

            text = details.read_text(encoding="utf-8", errors="ignore")
            match = re.search(r"(?:objective|cardinality|size)\\D+(\\d+(?:\\.\\d+)?)", text, re.IGNORECASE)
            if match:
                return float(match.group(1))
    return None


def primary_quality(case: dict[str, str], result: dict[str, Any]) -> tuple[str, str, str, bool]:
    problem = case["problem"]
    best = result.get("best_result", {})
    reconstructed = best.get("reconstructed_problem_objective", {})
    feasible = bool(best.get("feasible", reconstructed.get("feasible", False)))
    objective = reconstructed.get("objective_value", best.get("objective_value"))
    if problem == "market_share":
        return "not_applicable", str(objective), str(objective), feasible
    gap: str | float = "unknown_legacy_artifact"
    try:
        bks = bks_for_case(case)
        if bks is not None and float(bks) != 0.0 and objective is not None:
            gap = max(0.0, (float(bks) - float(objective)) / abs(float(bks)) * 100.0)
    except Exception:
        gap = "unknown_legacy_artifact"
    return gap, "not_applicable", "not_applicable", feasible


def load_trace(result_path: Path) -> list[dict[str, Any]]:
    trace_path = result_path.parent / "trace.jsonl"
    rows = []
    if not trace_path.exists():
        return rows
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            data = json.loads(line)
        except Exception:
            continue
        if "evaluation" in data and "objective_value" in data:
            rows.append(data)
    return rows


def aggregate_result(case: dict[str, str], seed: int, result_path: Path) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    result = json.loads(result_path.read_text(encoding="utf-8"))
    config = result.get("config", {})
    optimizer = result.get("optimizer", {})
    best = result.get("best_result", {})
    timing = result.get("timing", {})
    term = termination_reason(optimizer.get("message", ""))
    gap, tdev, mdev, feasible = primary_quality(case, result)
    runtime = timing.get("solve_runtime_sec", timing.get("total_instance_sec", "unknown_legacy_artifact"))
    replicate_rows = []
    # The benchmark result has one final sampled count set. We label it as the
    # production final-sampling seed and keep the requested schema.
    replicate_rows.append(
        {
            "case_id": case["case_id"],
            "problem": case["problem"],
            "instance": case["instance_label"],
            "method": case["method"],
            "initialization_seed": seed,
            "final_sampling_seed": "production_sampler_seed",
            "optimizer_name": "COBYLA",
            "max_optimizer_iterations": config.get("maxiter", "unknown_legacy_artifact"),
            "max_objective_evaluations": config.get("maxiter", "unknown_legacy_artifact"),
            "realized_optimizer_iterations": optimizer.get("total_evaluations", "unknown_legacy_artifact"),
            "realized_objective_evaluations": optimizer.get("total_evaluations", "unknown_legacy_artifact"),
            "termination_reason": term,
            "initial_energy": "unknown_legacy_artifact",
            "best_energy": best.get("best_sample_energy", "unknown_legacy_artifact"),
            "final_energy": best.get("optimization_objective_value", "unknown_legacy_artifact"),
            "final_decoded_gap_percent": gap,
            "final_tdev": tdev,
            "final_mdev": mdev,
            "decoded_feasible": feasible,
            "runtime_seconds": runtime,
        }
    )
    trace_rows = []
    best_so_far = math.inf
    cumulative = 0.0
    for entry in load_trace(result_path):
        idx = int(entry.get("evaluation", len(trace_rows) + 1))
        value = float(entry.get("objective_value"))
        best_so_far = min(best_so_far, value)
        meta = entry.get("metadata") or {}
        try:
            cumulative += float(meta.get("elapsed_sec", 0.0))
        except Exception:
            pass
        trace_rows.append(
            {
                "case_id": case["case_id"],
                "problem": case["problem"],
                "instance": case["instance_label"],
                "method": case["method"],
                "initialization_seed": seed,
                "budget_multiplier": 1,
                "objective_evaluation_index": idx,
                "optimizer_iteration_index": idx,
                "objective_value": value,
                "best_objective_so_far": best_so_far,
                "cumulative_runtime_seconds": cumulative,
            }
        )
    return replicate_rows, trace_rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault(str(row["case_id"]), []).append(row)
    out = []
    for case_id, items in sorted(grouped.items()):
        best_energies = [float(r["best_energy"]) for r in items if str(r["best_energy"]) not in {"not_applicable", "unknown_legacy_artifact"}]
        evals = [float(r["realized_objective_evaluations"]) for r in items if str(r["realized_objective_evaluations"]).replace(".", "", 1).isdigit()]
        qualities = []
        for r in items:
            q = r["final_decoded_gap_percent"]
            if q in {"not_applicable", "unknown_legacy_artifact", ""}:
                continue
            qualities.append(float(q))
        if not qualities:
            qualities = best_energies
        feasible_fraction = sum(str(r["decoded_feasible"]).lower() == "true" for r in items) / float(len(items) or 1)
        reasons = {}
        for r in items:
            reasons[str(r["termination_reason"])] = reasons.get(str(r["termination_reason"]), 0) + 1
        dominant_reason = max(reasons, key=reasons.get) if reasons else "unknown_legacy_artifact"
        sorted_energy = sorted(best_energies)
        iqr = "unknown_legacy_artifact"
        if len(sorted_energy) >= 2:
            q1 = statistics.quantiles(sorted_energy, n=4, method="inclusive")[0]
            q3 = statistics.quantiles(sorted_energy, n=4, method="inclusive")[2]
            iqr = q3 - q1
        first = items[0]
        out.append(
            {
                "case_id": case_id,
                "problem": first["problem"],
                "instance": first["instance"],
                "method": first["method"],
                "n_initializations": len({r["initialization_seed"] for r in items}),
                "n_final_sampling_repetitions_per_initialization": 1,
                "median_best_energy": statistics.median(best_energies) if best_energies else "unknown_legacy_artifact",
                "iqr_best_energy": iqr,
                "mean_primary_quality": statistics.mean(qualities) if qualities else "unknown_legacy_artifact",
                "sd_primary_quality": statistics.stdev(qualities) if len(qualities) > 1 else 0.0 if qualities else "unknown_legacy_artifact",
                "median_primary_quality": statistics.median(qualities) if qualities else "unknown_legacy_artifact",
                "best_primary_quality": min(qualities) if qualities else "unknown_legacy_artifact",
                "worst_primary_quality": max(qualities) if qualities else "unknown_legacy_artifact",
                "feasible_initialization_fraction": feasible_fraction,
                "median_realized_objective_evaluations": statistics.median(evals) if evals else "unknown_legacy_artifact",
                "min_realized_objective_evaluations": min(evals) if evals else "unknown_legacy_artifact",
                "max_realized_objective_evaluations": max(evals) if evals else "unknown_legacy_artifact",
                "dominant_termination_reason": dominant_reason,
            }
        )
    return out


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_digest(out_dir: Path, replicate_rows: list[dict[str, Any]], summary_rows: list[dict[str, Any]]) -> None:
    lines = [
        "# Reduced Part B Seed Sensitivity Digest",
        "",
        "## Scope",
        "",
        "- Reduced campaign: 3 cases, 3 initialization seeds each.",
        "- No Part C budget extension was run.",
        "- Final-sampling repetitions are not independently rerun here; each row uses the benchmark runner's production final sampled counts. This is a limitation relative to the full requested Part B design.",
        "",
        "## Cases",
        "",
    ]
    for case in CASES:
        lines.append(f"- `{case['case_id']}`: {case['problem']} / {case['instance_label']} / {case['method']} - {case['why']}")
    lines.extend(["", "## Summary", ""])
    lines.append("| case | n seeds | median best energy | IQR best energy | feasible fraction | median evals | termination |")
    lines.append("|---|---:|---:|---:|---:|---:|---|")
    for row in summary_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["case_id"]),
                    str(row["n_initializations"]),
                    str(row["median_best_energy"]),
                    str(row["iqr_best_energy"]),
                    str(row["feasible_initialization_fraction"]),
                    str(row["median_realized_objective_evaluations"]),
                    str(row["dominant_termination_reason"]),
                ]
            )
            + " |"
        )
    (out_dir / "reduced_part_b_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "reduced_part_b"))
    parser.add_argument("--maxiter", type=int, default=200)
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--case-ids", nargs="*", default=[c["case_id"] for c in CASES])
    args = parser.parse_args()

    out_dir = (REPO_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    selected = [c for c in CASES if c["case_id"] in set(args.case_ids)]
    all_replicates: list[dict[str, Any]] = []
    all_traces: list[dict[str, Any]] = []
    run_index = []
    for case in selected:
        for seed in INIT_SEEDS:
            print(f"Running {case['case_id']} seed={seed}", flush=True)
            result_path = run_one(case, seed, out_dir, args.maxiter, args.shots)
            reps, traces = aggregate_result(case, seed, result_path)
            all_replicates.extend(reps)
            all_traces.extend(traces)
            run_index.append({"case_id": case["case_id"], "seed": seed, "result_json": str(result_path)})
    summary_rows = summarize(all_replicates)
    write_csv(out_dir / "optimizer_seed_sensitivity_replicates.csv", all_replicates, REPLICATE_COLUMNS)
    write_csv(out_dir / "optimizer_seed_sensitivity_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(out_dir / "optimizer_traces.csv", all_traces, TRACE_COLUMNS)
    (out_dir / "run_index.json").write_text(json.dumps(run_index, indent=2), encoding="utf-8")
    write_digest(out_dir, all_replicates, summary_rows)
    print(f"Wrote reduced Part B results to {out_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
