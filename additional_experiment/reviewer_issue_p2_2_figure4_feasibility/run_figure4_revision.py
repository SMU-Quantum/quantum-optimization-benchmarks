#!/usr/bin/env python3
"""Create the revised two-panel Figure 4 data, summary, digest, and plot."""

from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_benchmark" / "research_benchmark" / "results_hardware"
OUT = Path(__file__).resolve().parent

EPS_2Q_REF = 0.003
FREF_THRESHOLDS = (0.1, 0.01)

PROBLEM_LABEL = {
    "mkp": "MDKP",
    "mis": "MIS",
    "qap": "QAP",
    "market_share": "MSP",
}
PROBLEM_DIR = {
    "mkp": "mkp",
    "mis": "mis",
    "qap": "qap",
    "market_share": "market_share",
}
PROBLEM_COLORS = {
    "MDKP": "#2B8CBE",
    "MIS": "#1BAE91",
    "QAP": "#7B61B3",
    "MSP": "#E17C05",
}
LANES = ["MDKP", "MIS", "QAP", "MSP"]
LANE_Y = {problem: idx for idx, problem in enumerate(LANES)}

METHOD_LABEL = {
    "vqe": "VQE",
    "cvar_vqe": "CVaR-VQE",
    "qaoa": "QAOA",
    "ma_qaoa": "MA-QAOA",
    "ws_qaoa": "WS-QAOA",
    "pce": "PCE",
    "qrao": "QRAO",
}
METHOD_FAMILY = {
    "vqe": "VQE-family",
    "cvar_vqe": "VQE-family",
    "qaoa": "QAOA-family",
    "ma_qaoa": "QAOA-family",
    "ws_qaoa": "QAOA-family",
    "pce": "PCE",
    "qrao": "QRAO",
}
METHOD_MARKER = {
    "vqe": "o",
    "cvar_vqe": "s",
    "qaoa": "^",
    "ma_qaoa": "D",
    "ws_qaoa": "v",
    "pce": "p",
    "qrao": "X",
}
METHOD_MARKER_KEY = {
    "vqe": "circle",
    "cvar_vqe": "square",
    "qaoa": "upward_triangle",
    "ma_qaoa": "diamond",
    "ws_qaoa": "downward_triangle",
    "pce": "pentagon",
    "qrao": "X",
}
METHOD_ORDER = ["vqe", "cvar_vqe", "qaoa", "ma_qaoa", "ws_qaoa", "pce", "qrao"]

DATA_COLUMNS = [
    "problem",
    "instance",
    "method",
    "method_family",
    "backend",
    "transpiled_two_qubit_gates",
    "backend_two_qubit_error",
    "hardware_feasible",
    "hardware_optimality_gap_percent",
    "hardware_tdev",
    "quality_metric",
    "reference_fidelity_proxy",
    "point_panel",
    "infeasible_problem_lane",
    "marker_key",
    "problem_color_key",
    "source_artifact",
]

SUMMARY_COLUMNS = [
    "problem",
    "method",
    "hardware_runs",
    "feasible_runs",
    "infeasible_runs",
    "median_transpiled_two_qubit_gates_feasible",
    "median_transpiled_two_qubit_gates_infeasible",
    "min_transpiled_two_qubit_gates_infeasible",
    "max_transpiled_two_qubit_gates_infeasible",
    "notes",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]], columns: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def clean_instance(name: str) -> str:
    text = str(name).replace("\\", "/").split("/")[-1]
    for suffix in (".dat", ".txt", ".gen", "_dat", "_txt", "_gen"):
        if text.endswith(suffix):
            text = text[: -len(suffix)]
    return text.replace("_", ".") if text.startswith("1") else text


def safe_float(value: Any) -> float:
    if value is None:
        return math.nan
    text = str(value).strip()
    if text.lower() in {"", "nan", "not_available", "none"}:
        return math.nan
    if text.lower() == "inf":
        return math.inf
    try:
        return float(text)
    except ValueError:
        return math.nan


def fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isnan(value):
            return "not_available"
        if math.isinf(value):
            return "inf"
        return f"{value:.12g}"
    text = str(value).strip()
    return text if text else "not_available"


def load_trace(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def load_main_lookup(problem_key: str, method: str) -> dict[str, dict[str, str]]:
    prefix = PROBLEM_DIR[problem_key]
    method_dir = RESULTS / problem_key / f"{prefix}_{method}"
    candidates = [
        method_dir / "csv" / f"main_table_{method}.csv",
        method_dir / "csv" / "main_table.csv",
    ]
    for path in candidates:
        if path.exists():
            return {clean_instance(row["Instance"]): row for row in read_csv(path)}
    return {}


def result_paths() -> list[Path]:
    paths: list[Path] = []
    for problem_key in PROBLEM_LABEL:
        paths.extend(sorted((RESULTS / problem_key).glob("*/*/result.json")))
    return sorted(paths)


def choose_job_metadata(payload: dict[str, Any], trace_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], str]:
    target_energy = safe_float(
        payload.get("best_result", {}).get("best_sample_energy")
        or payload.get("benchmark_protocol", {}).get("objective_measured", {}).get("reported_best_sample_energy")
    )
    if math.isfinite(target_energy):
        for trace in trace_rows:
            if abs(safe_float(trace.get("best_sample_energy")) - target_energy) <= 1e-6:
                meta = trace.get("metadata")
                if isinstance(meta, dict) and meta.get("transpiled_2q_gates") is not None:
                    return meta, "trace_best_sample_energy_match"
    for trace in reversed(trace_rows):
        meta = trace.get("metadata")
        if isinstance(meta, dict) and meta.get("transpiled_2q_gates") is not None:
            return meta, "trace_last_metadata_fallback"
    job_metadata = payload.get("job_metadata")
    if isinstance(job_metadata, list):
        for meta in reversed(job_metadata):
            if isinstance(meta, dict) and meta.get("transpiled_2q_gates") is not None:
                return meta, "result_job_metadata_last_fallback"
    return {}, "metadata_not_available"


def backend_two_qubit_error(payload: dict[str, Any], backend: str) -> str:
    calibration = payload.get("device_calibration")
    if not isinstance(calibration, dict):
        return "not_available_in_source_artifact"
    candidates = [
        calibration.get(f"ibm_{backend}"),
        calibration.get(backend),
    ]
    for entry in candidates:
        if not isinstance(entry, dict):
            continue
        for key in ("two_qubit_gate_error", "two_qubit_error", "cz_error", "ecr_error"):
            value = entry.get(key)
            if isinstance(value, dict):
                for subkey in ("mean", "median", "avg"):
                    if value.get(subkey) is not None:
                        return fmt(value[subkey])
            elif value is not None:
                return fmt(value)
    return "not_available_in_source_artifact"


def infer_feasibility(payload: dict[str, Any], main_row: dict[str, str], problem: str) -> bool:
    best = payload.get("best_result")
    if isinstance(best, dict) and best.get("feasible") is not None:
        return bool(best.get("feasible"))
    text = str(main_row.get("Feas", "")).strip().lower()
    if text in {"yes", "true"}:
        return True
    if text in {"no", "false"}:
        return False
    if problem == "MSP":
        # MSP exact target mismatch is not infeasibility; the encoded assignment can be feasible with nonzero TDev.
        return True
    return False


def build_data_rows() -> list[dict[str, str]]:
    main_cache: dict[tuple[str, str], dict[str, dict[str, str]]] = {}
    rows: list[dict[str, str]] = []
    for path in result_paths():
        payload = json.loads(path.read_text(encoding="utf-8"))
        problem_key = str(payload.get("problem"))
        problem = PROBLEM_LABEL.get(problem_key)
        if problem is None:
            continue
        method = str(payload.get("execution", {}).get("method") or path.parents[1].name.split("_", 1)[-1])
        if method not in METHOD_LABEL:
            continue
        instance = clean_instance(payload.get("instance_name", path.parent.name))
        cache_key = (problem_key, method)
        if cache_key not in main_cache:
            main_cache[cache_key] = load_main_lookup(problem_key, method)
        main_row = main_cache[cache_key].get(instance, {})
        trace_rows = load_trace(path.with_name("trace.jsonl"))
        metadata, metadata_note = choose_job_metadata(payload, trace_rows)
        twoq = safe_float(metadata.get("transpiled_2q_gates"))
        if not math.isfinite(twoq) or twoq <= 0:
            continue
        backend = fmt(metadata.get("backend_name"))
        feasible = infer_feasibility(payload, main_row, problem)
        optimality_gap = safe_float(main_row.get("Gap%"))
        tdev = safe_float(
            main_row.get("TotalAbsoluteDeviation")
            or payload.get("best_result", {}).get("objective_value")
            or payload.get("benchmark_protocol", {}).get("objective_measured", {}).get("post_processed_objective")
        )
        if feasible and problem in {"MDKP", "MIS"}:
            point_panel = "feasible_gap"
            quality_metric = "optimality_gap_percent"
        elif not feasible:
            point_panel = "infeasible_lane"
            quality_metric = "not_applicable"
        elif problem == "MSP":
            point_panel = "excluded_from_gap_panel"
            quality_metric = "tdev"
        else:
            point_panel = "excluded_from_gap_panel"
            quality_metric = "not_applicable"
        rows.append(
            {
                "problem": problem,
                "instance": instance,
                "method": METHOD_LABEL[method],
                "method_family": METHOD_FAMILY[method],
                "backend": backend,
                "transpiled_two_qubit_gates": fmt(int(round(twoq))),
                "backend_two_qubit_error": backend_two_qubit_error(payload, backend),
                "hardware_feasible": fmt(feasible),
                "hardware_optimality_gap_percent": fmt(optimality_gap if feasible and problem in {"MDKP", "MIS"} else None),
                "hardware_tdev": fmt(tdev if problem == "MSP" and feasible else None),
                "quality_metric": quality_metric,
                "reference_fidelity_proxy": fmt((1.0 - EPS_2Q_REF) ** twoq),
                "point_panel": point_panel,
                "infeasible_problem_lane": problem if point_panel == "infeasible_lane" else "not_applicable",
                "marker_key": METHOD_MARKER_KEY[method],
                "problem_color_key": problem,
                "source_artifact": f"{path.relative_to(ROOT)}; metadata_selection={metadata_note}",
            }
        )
    return rows


def numeric(row: dict[str, str], key: str) -> float:
    return safe_float(row.get(key))


def build_summary_rows(data_rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in data_rows:
        grouped[(row["problem"], row["method"])].append(row)
    summary: list[dict[str, str]] = []
    for (problem, method), rows in sorted(grouped.items()):
        feasible = [numeric(r, "transpiled_two_qubit_gates") for r in rows if r["hardware_feasible"] == "true"]
        infeasible = [numeric(r, "transpiled_two_qubit_gates") for r in rows if r["hardware_feasible"] == "false"]
        feasible = [x for x in feasible if math.isfinite(x)]
        infeasible = [x for x in infeasible if math.isfinite(x)]
        notes = []
        if problem == "QAP" and rows and len(infeasible) == len(rows):
            notes.append("all_reported_qap_hardware_outcomes_infeasible")
        if problem == "MSP":
            notes.append("feasible_msp_uses_tdev_and_is_excluded_from_gap_panel")
        summary.append(
            {
                "problem": problem,
                "method": method,
                "hardware_runs": fmt(len(rows)),
                "feasible_runs": fmt(len(feasible)),
                "infeasible_runs": fmt(len(infeasible)),
                "median_transpiled_two_qubit_gates_feasible": fmt(statistics.median(feasible) if feasible else None),
                "median_transpiled_two_qubit_gates_infeasible": fmt(statistics.median(infeasible) if infeasible else None),
                "min_transpiled_two_qubit_gates_infeasible": fmt(min(infeasible) if infeasible else None),
                "max_transpiled_two_qubit_gates_infeasible": fmt(max(infeasible) if infeasible else None),
                "notes": ";".join(notes) if notes else "not_applicable",
            }
        )
    return summary


def fref_to_n2q(fref: float) -> float:
    return math.log(fref) / math.log(1.0 - EPS_2Q_REF)


def plot(data_rows: list[dict[str, str]]) -> None:
    feasible_rows = [
        row
        for row in data_rows
        if row["point_panel"] == "feasible_gap" and math.isfinite(numeric(row, "hardware_optimality_gap_percent"))
    ]
    infeasible_rows = [row for row in data_rows if row["point_panel"] == "infeasible_lane"]
    fig, (ax_gap, ax_inf) = plt.subplots(
        2,
        1,
        figsize=(8.7, 7.2),
        sharex=True,
        gridspec_kw={"height_ratios": [2.1, 1.25], "hspace": 0.08},
    )
    for ax in (ax_gap, ax_inf):
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        for fref in FREF_THRESHOLDS:
            ax.axvline(fref_to_n2q(fref), color="#777777", linestyle="--", linewidth=1.0, zorder=0.5)

    for row in feasible_rows:
        method_key = next(k for k, v in METHOD_LABEL.items() if v == row["method"])
        problem = row["problem"]
        ax_gap.scatter(
            numeric(row, "transpiled_two_qubit_gates"),
            numeric(row, "hardware_optimality_gap_percent"),
            marker=METHOD_MARKER[method_key],
            s=62,
            facecolor=PROBLEM_COLORS[problem],
            edgecolor="#1a1a1a",
            linewidth=0.55,
            alpha=0.9,
            zorder=2,
        )
    ax_gap.set_ylabel("Hardware optimality gap to BKS (%)")
    ax_gap.text(
        0.01,
        0.96,
        "(a) Feasible MDKP and MIS hardware outcomes",
        transform=ax_gap.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax_gap.set_ylim(bottom=-2)

    for idx, problem in enumerate(LANES):
        if idx % 2 == 0:
            ax_inf.axhspan(idx - 0.5, idx + 0.5, color="#f0f0f0", zorder=0)
    jitter_offsets = defaultdict(int)
    for row in infeasible_rows:
        method_key = next(k for k, v in METHOD_LABEL.items() if v == row["method"])
        problem = row["problem"]
        lane = LANE_Y[problem]
        count = jitter_offsets[(problem, row["method"])]
        jitter_offsets[(problem, row["method"])] += 1
        jitter = ((count % 7) - 3) * 0.045
        ax_inf.scatter(
            numeric(row, "transpiled_two_qubit_gates"),
            lane + jitter,
            marker=METHOD_MARKER[method_key],
            s=70,
            facecolor=PROBLEM_COLORS[problem],
            edgecolor="#1a1a1a",
            linewidth=0.55,
            alpha=0.9,
            zorder=2,
        )
    ax_inf.set_yticks([LANE_Y[p] for p in LANES])
    ax_inf.set_yticklabels(LANES)
    ax_inf.set_ylim(-0.55, len(LANES) - 0.45)
    ax_inf.text(
        0.01,
        0.92,
        "(b) Infeasible hardware outcomes",
        transform=ax_inf.transAxes,
        ha="left",
        va="top",
        fontsize=10.5,
        bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72, "pad": 1.5},
    )
    ax_inf.set_xlabel("Transpiled two-qubit gates, N2Q")
    ax_inf.set_ylabel("Problem family")

    x_values = [numeric(row, "transpiled_two_qubit_gates") for row in data_rows]
    x_values = [x for x in x_values if math.isfinite(x) and x > 0]
    ax_inf.set_xlim(max(1.0, min(x_values) * 0.75), max(x_values) * 1.35)

    top = ax_gap.twiny()
    top.set_xscale("log")
    top.set_xlim(ax_gap.get_xlim())
    ticks = [(100, "0.74"), (770, "1e-1"), (1540, "1e-2"), (2300, "1e-3"), (7670, "1e-10"), (23000, "1e-30")]
    xmin, xmax = ax_gap.get_xlim()
    ticks = [(x, label) for x, label in ticks if xmin <= x <= xmax]
    top.set_xticks([x for x, _ in ticks])
    top.set_xticklabels([label for _, label in ticks], fontsize=7.5, rotation=35, ha="left")
    top.set_xlabel("Reference fidelity proxy, Fref=(1-0.003)^N2Q", labelpad=7)

    method_handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKER[m],
            linestyle="",
            markerfacecolor="#9a9a9a",
            markeredgecolor="#1a1a1a",
            markersize=7,
            label=METHOD_LABEL[m],
        )
        for m in METHOD_ORDER
    ]
    problem_handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor=PROBLEM_COLORS[p],
            markeredgecolor="#1a1a1a",
            markersize=7,
            label=p,
        )
        for p in LANES
    ]
    handles = problem_handles + method_handles
    fig.legend(
        handles=handles,
        loc="center left",
        bbox_to_anchor=(0.83, 0.52),
        frameon=True,
        title="Problem / method",
        fontsize=8.5,
    )
    fig.suptitle("Backend-native circuit complexity, feasibility, and hardware solution quality", y=0.975, fontsize=13)
    fig.subplots_adjust(left=0.10, right=0.80, top=0.85, bottom=0.10, hspace=0.12)
    fig.savefig(OUT / "fig_circuit_complexity_vs_quality_revised.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_circuit_complexity_vs_quality_revised.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_digest(data_rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> None:
    by_problem_method = {(r["problem"], r["method"]): r for r in summary_rows}
    lane_problems = sorted({r["problem"] for r in data_rows if r["point_panel"] == "infeasible_lane"}, key=LANES.index)
    lines = [
        "# Figure 4 Complexity/Feasibility Digest",
        "",
        "Included rows are archived hardware outcomes with a recoverable backend-native transpiled two-qubit-gate count. Two MIS/QRAO artifacts are excluded from the figure data because one used `statevector_primitives` and one archived no backend/transpiled hardware metric.",
        "",
        "## 1. Feasible and Infeasible Outcomes by Problem/Method",
        "",
    ]
    for problem in LANES:
        for method in [METHOD_LABEL[m] for m in METHOD_ORDER]:
            row = by_problem_method.get((problem, method))
            if row is None:
                continue
            lines.append(
                f"- {problem}/{method}: hardware_runs={row['hardware_runs']}, feasible={row['feasible_runs']}, infeasible={row['infeasible_runs']}."
            )
    lines.extend(
        [
            "",
            "## 2. Infeasible Panel Problem Families",
            "",
            ", ".join(lane_problems) if lane_problems else "none",
            "",
            "## 3. Lowest-N2Q Infeasible Points",
            "",
        ]
    )
    for problem in ["MIS", "QAP", "MSP"]:
        vals = [
            (numeric(row, "transpiled_two_qubit_gates"), row)
            for row in data_rows
            if row["problem"] == problem and row["point_panel"] == "infeasible_lane"
        ]
        vals = [(x, row) for x, row in vals if math.isfinite(x)]
        if vals:
            x, row = min(vals, key=lambda item: item[0])
            lines.append(f"- {problem}: {fmt(int(x))} ({row['method']} {row['instance']} on {row['backend']}).")
        else:
            lines.append(f"- {problem}: no infeasible point.")
    lines.extend(["", "## 4. Highest-N2Q Feasible Points", ""])
    for problem in ["MDKP", "MIS"]:
        vals = [
            (numeric(row, "transpiled_two_qubit_gates"), row)
            for row in data_rows
            if row["problem"] == problem and row["hardware_feasible"] == "true"
        ]
        vals = [(x, row) for x, row in vals if math.isfinite(x)]
        if vals:
            x, row = max(vals, key=lambda item: item[0])
            lines.append(f"- {problem}: {fmt(int(x))} ({row['method']} {row['instance']} on {row['backend']}).")
    qap_rows = [r for r in data_rows if r["problem"] == "QAP"]
    all_qap_infeasible = bool(qap_rows) and all(r["point_panel"] == "infeasible_lane" for r in qap_rows)
    n01 = fref_to_n2q(0.1)
    n001 = fref_to_n2q(0.01)
    lines.extend(
        [
            "",
            "## 5. QAP Feasibility",
            "",
            f"All QAP results appear in the infeasible panel: {fmt(all_qap_infeasible)}.",
            "",
            "## 6. Reference Fidelity Thresholds",
            "",
            f"- Fref=0.1 corresponds to N2Q={n01:.1f} (approximately 770).",
            f"- Fref=0.01 corresponds to N2Q={n001:.1f} (approximately 1540).",
            "",
            "## 7. Fidelity Axis Interpretation",
            "",
            "The top fidelity axis is a common reference mapping under epsilon_2Q_ref=3e-3, not each run's backend-specific F_est. Backend-specific fidelity estimates remain appendix/source-data quantities.",
        ]
    )
    (OUT / "figure4_complexity_feasibility_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    data_rows = build_data_rows()
    summary_rows = build_summary_rows(data_rows)
    write_csv(OUT / "figure4_complexity_feasibility_data.csv", data_rows, DATA_COLUMNS)
    write_csv(OUT / "figure4_complexity_feasibility_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_digest(data_rows, summary_rows)
    plot(data_rows)
    print(f"wrote {len(data_rows)} data rows and {len(summary_rows)} summary rows")


if __name__ == "__main__":
    main()
