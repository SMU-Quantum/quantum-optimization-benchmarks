#!/usr/bin/env python3
"""Regenerate MIS Figure 2 with 1tc.64 explicitly included in panel (a)."""

from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "research_benchmark" / "research_benchmark" / "results_hardware" / "mis"
BKS_CSV = ROOT / "classical_solutions" / "results" / "mis" / "summary.csv"
OUT = Path(__file__).resolve().parent

INSTANCE_ORDER = ["1tc.8", "1tc.16", "1tc.32", "1tc.64", "1et.64", "1dc.64", "1dc.128"]
METHOD_ORDER = ["vqe", "cvar_vqe", "qaoa", "ma_qaoa", "ws_qaoa", "qrao", "pce"]
METHOD_LABEL = {
    "vqe": "VQE",
    "cvar_vqe": "CVaR-VQE",
    "qaoa": "QAOA",
    "ma_qaoa": "MA-QAOA",
    "ws_qaoa": "WS-QAOA",
    "qrao": "QRAO",
    "pce": "PCE",
}
METHOD_COLOR = {
    "vqe": "#1f77b4",
    "cvar_vqe": "#ff7f0e",
    "qaoa": "#2ca02c",
    "ma_qaoa": "#d62728",
    "ws_qaoa": "#9467bd",
    "qrao": "#e377c2",
    "pce": "#8c564b",
}

DATA_COLUMNS = [
    "problem",
    "instance",
    "method",
    "method_label",
    "bks_objective",
    "hardware_objective",
    "hardware_feasible",
    "gap_to_bks_percent",
    "selected_qpus",
    "panel_a_included",
    "source_artifact",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


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
        return f"{value:.12g}"
    text = str(value).strip()
    return text if text else "not_available"


def load_bks() -> dict[str, float]:
    bks: dict[str, float] = {}
    for row in read_csv(BKS_CSV):
        bks[row["instance_id"]] = float(row["objective_value"])
    return bks


def result_path(method: str, instance: str) -> Path:
    return RESULTS / f"mis_{method}" / f"{instance.replace('.', '_')}_txt" / "result.json"


def nested_get(payload: dict[str, Any], *keys: str) -> Any:
    value: Any = payload
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def objective_and_feasible(payload: dict[str, Any]) -> tuple[float, bool]:
    objective = nested_get(payload, "reconstructed_problem_objective", "objective_value")
    feasible = nested_get(payload, "reconstructed_problem_objective", "feasible")
    if objective is None:
        objective = nested_get(payload, "best_result", "objective_value")
    if feasible is None:
        feasible = nested_get(payload, "best_result", "feasible")
    if objective is None:
        objective = nested_get(payload, "benchmark_protocol", "objective_measured", "post_processed_objective")
    if feasible is None:
        feasible = nested_get(payload, "benchmark_protocol", "feasibility_policy", "raw_feasible_before_repair")
    return float(objective), bool(feasible)


def build_rows() -> list[dict[str, str]]:
    bks = load_bks()
    rows: list[dict[str, str]] = []
    for instance in INSTANCE_ORDER:
        for method in METHOD_ORDER:
            path = result_path(method, instance)
            if not path.exists():
                continue
            payload = json.loads(path.read_text(encoding="utf-8"))
            objective, feasible = objective_and_feasible(payload)
            gap = ((bks[instance] - objective) / bks[instance] * 100.0) if feasible else math.nan
            selected_qpus = payload.get("execution", {}).get("selected_qpus", [])
            rows.append(
                {
                    "problem": "MIS",
                    "instance": instance,
                    "method": method,
                    "method_label": METHOD_LABEL[method],
                    "bks_objective": fmt(bks[instance]),
                    "hardware_objective": fmt(objective),
                    "hardware_feasible": fmt(feasible),
                    "gap_to_bks_percent": fmt(gap if feasible else None),
                    "selected_qpus": "|".join(map(str, selected_qpus)) if selected_qpus else "not_available",
                    "panel_a_included": "pending",
                    "source_artifact": str(path.relative_to(ROOT)),
                }
            )
    feasible_instances = {row["instance"] for row in rows if row["hardware_feasible"] == "true"}
    for row in rows:
        row["panel_a_included"] = fmt(row["instance"] in feasible_instances)
    return rows


def safe_float(text: str) -> float:
    try:
        return float(text)
    except Exception:
        return math.nan


def plot(rows: list[dict[str, str]]) -> None:
    included_instances = [inst for inst in INSTANCE_ORDER if any(r["instance"] == inst and r["panel_a_included"] == "true" for r in rows)]
    feasible_counts = Counter(row["instance"] for row in rows if row["hardware_feasible"] == "true")
    rows_by_key = {(row["instance"], row["method"]): row for row in rows}

    fig, (ax_gap, ax_count) = plt.subplots(
        1,
        2,
        figsize=(12.4, 4.7),
        gridspec_kw={"width_ratios": [1.62, 1.0], "wspace": 0.12},
    )

    bar_width = 0.78 / len(METHOD_ORDER)
    infeasible_marker_height = 3.0
    for inst_idx, instance in enumerate(included_instances):
        for method_idx, method in enumerate(METHOD_ORDER):
            row = rows_by_key.get((instance, method))
            if row is None:
                continue
            x = inst_idx - 0.39 + (method_idx + 0.5) * bar_width
            feasible = row["hardware_feasible"] == "true"
            gap = safe_float(row["gap_to_bks_percent"])
            if feasible and math.isfinite(gap):
                ax_gap.bar(
                    x,
                    gap,
                    width=bar_width * 0.92,
                    color=METHOD_COLOR[method],
                    edgecolor="white",
                    linewidth=0.4,
                    zorder=2,
                )
                label = "0" if abs(gap) < 1e-9 else f"{gap:.1f}".rstrip("0").rstrip(".")
                ax_gap.text(x, gap + 1.0, label, ha="center", va="bottom", fontsize=6.5, rotation=90 if gap > 0 else 0)
            else:
                ax_gap.bar(
                    x,
                    infeasible_marker_height,
                    width=bar_width * 0.92,
                    color="#eeeeee",
                    edgecolor="#9a9a9a",
                    linewidth=0.45,
                    hatch="///",
                    zorder=1.5,
                )
        if instance == "1tc.64":
            ax_gap.annotate(
                "1 feasible method\n(WS-QAOA)",
                xy=(inst_idx, 35.0),
                xytext=(inst_idx - 0.35, 52.0),
                arrowprops={"arrowstyle": "->", "linewidth": 0.8, "color": "#333333"},
                ha="right",
                va="center",
                fontsize=8.0,
            )

    ax_gap.set_title("(a) MIS: Solution Quality on Feasible Instances", fontsize=10.5, weight="bold")
    ax_gap.set_ylabel("Gap to BKS (%)")
    ax_gap.set_xlabel("Instance")
    ax_gap.set_xticks(range(len(included_instances)))
    ax_gap.set_xticklabels(included_instances)
    ax_gap.set_ylim(0, 68)
    ax_gap.grid(axis="y", alpha=0.28)

    legend_handles = [
        Patch(facecolor=METHOD_COLOR[m], edgecolor="white", label=METHOD_LABEL[m]) for m in METHOD_ORDER
    ]
    legend_handles.append(Patch(facecolor="#eeeeee", edgecolor="#9a9a9a", hatch="///", label="Infeasible"))
    ax_gap.legend(handles=legend_handles, ncol=2, fontsize=7.0, frameon=False, loc="upper left")

    count_colors = ["#27ae60" if feasible_counts[inst] > 1 else "#f39c12" if feasible_counts[inst] == 1 else "#e74c3c" for inst in INSTANCE_ORDER]
    ax_count.bar(range(len(INSTANCE_ORDER)), [feasible_counts[inst] for inst in INSTANCE_ORDER], color=count_colors, width=0.78)
    for idx, instance in enumerate(INSTANCE_ORDER):
        count = feasible_counts[instance]
        color = "#111111" if count > 0 else "#c0392b"
        ax_count.text(idx, count + 0.12, f"{count}/7", ha="center", va="bottom", fontsize=8.0, weight="bold", color=color)
    ax_count.axhline(7, color="#999999", linestyle=":", linewidth=1.0)
    ax_count.set_title("(b) MIS: Feasibility Cliff", fontsize=10.5, weight="bold")
    ax_count.set_ylabel("Number of Feasible Methods (out of 7)")
    ax_count.set_xlabel("Instance")
    ax_count.set_xticks(range(len(INSTANCE_ORDER)))
    ax_count.set_xticklabels(INSTANCE_ORDER, rotation=45, ha="right")
    ax_count.set_ylim(0, 8.5)
    ax_count.grid(axis="y", alpha=0.28)
    top = ax_count.secondary_xaxis("top")
    top.set_xticks(range(len(INSTANCE_ORDER)))
    top.set_xticklabels(["8q", "16q", "32q", "64q", "64q", "64q", "128q"], fontsize=7.5, color="#555555")
    top.set_xlabel("Logical Qubits", fontsize=8.0, color="#555555")

    fig.suptitle("MIS: Hardware Performance and Feasibility Breakdown", fontsize=13.0, weight="bold", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.985, top=0.80, bottom=0.18)
    fig.savefig(OUT / "fig2_mis_hardware_feasibility_revised.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig2_mis_hardware_feasibility_revised.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_digest(rows: list[dict[str, str]]) -> None:
    feasible_counts = Counter(row["instance"] for row in rows if row["hardware_feasible"] == "true")
    by_instance = defaultdict(list)
    for row in rows:
        by_instance[row["instance"]].append(row)
    lines = [
        "# Figure 2 MIS 1tc.64 Digest",
        "",
        "The plotting audit confirms the reviewer concern: `1tc.64` admits one feasible hardware solution and therefore should not be absent from panel (a) when the caption says panel (a) includes all MIS instances with at least one feasible hardware solution.",
        "",
        "## Feasible Methods Per Instance",
        "",
    ]
    for instance in INSTANCE_ORDER:
        feasible = [row["method_label"] for row in by_instance[instance] if row["hardware_feasible"] == "true"]
        lines.append(f"- {instance}: {len(feasible)}/7 feasible ({', '.join(feasible) if feasible else 'none'}).")
    row_1tc64 = next(row for row in rows if row["instance"] == "1tc.64" and row["method"] == "ws_qaoa")
    lines.extend(
        [
            "",
            "## 1tc.64 Resolution",
            "",
            f"`1tc.64` is now included in panel (a). Its only feasible method is WS-QAOA, with hardware objective {row_1tc64['hardware_objective']} against BKS {row_1tc64['bks_objective']}, giving a gap of {row_1tc64['gap_to_bks_percent']}%. The other six methods on `1tc.64` are marked as infeasible in panel (a), matching the `1/7` count in panel (b).",
            "",
            "## Caption Replacement",
            "",
            "Figure 2. Hardware performance and feasibility breakdown for MIS instances. Panel (a) reports the gap to BKS for every MIS instance with at least one feasible hardware solution; infeasible method-instance pairs within those groups are shown with hatched baseline markers. The 1tc.64 instance is included because WS-QAOA produced one feasible repaired hardware output, while the other six methods were infeasible. Panel (b) summarizes the number of feasible methods per instance as a function of problem size.",
        ]
    )
    (OUT / "figure2_mis_1tc64_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(OUT / "figure2_mis_hardware_feasibility_data.csv", rows)
    write_digest(rows)
    plot(rows)
    print(f"wrote {len(rows)} MIS Figure 2 rows")


if __name__ == "__main__":
    main()
