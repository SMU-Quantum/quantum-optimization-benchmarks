#!/usr/bin/env python3
"""Regenerate Figure 5 with corrected panel order and x-axis labelling."""

from __future__ import annotations

import csv
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[1]
FIG4_DATA = ROOT / "reviewer_issue_p2_2_figure4_feasibility" / "figure4_complexity_feasibility_data.csv"
OUT = ROOT / "reviewer_issue_p2_3_figure5_fidelity"

BOUNDARY = 1e-3

PROBLEM_COLORS = {
    "MDKP": "#2B8CBE",
    "MIS": "#1BAE91",
}
METHOD_MARKERS = {
    "VQE": "o",
    "CVaR-VQE": "s",
    "QAOA": "^",
    "MA-QAOA": "D",
    "WS-QAOA": "v",
    "PCE": "p",
    "QRAO": "X",
}
METHOD_ORDER = ["VQE", "CVaR-VQE", "QAOA", "MA-QAOA", "WS-QAOA", "PCE", "QRAO"]
MARKER_KEY = {
    "VQE": "circle",
    "CVaR-VQE": "square",
    "QAOA": "upward_triangle",
    "MA-QAOA": "diamond",
    "WS-QAOA": "downward_triangle",
    "PCE": "pentagon",
    "QRAO": "X",
}

DATA_COLUMNS = [
    "problem",
    "instance",
    "method",
    "backend",
    "transpiled_two_qubit_gates",
    "fidelity_proxy",
    "hardware_feasible",
    "hardware_optimality_gap_percent",
    "marker_key",
    "problem_color_key",
    "panel",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=DATA_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        return float(str(value).strip())
    except Exception:
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


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for row in read_csv(FIG4_DATA):
        if row["point_panel"] != "feasible_gap":
            continue
        if row["problem"] not in {"MDKP", "MIS"}:
            continue
        fidelity = safe_float(row["reference_fidelity_proxy"])
        gap = safe_float(row["hardware_optimality_gap_percent"])
        twoq = safe_float(row["transpiled_two_qubit_gates"])
        if not (math.isfinite(fidelity) and fidelity > 0 and math.isfinite(gap) and math.isfinite(twoq)):
            continue
        method = row["method"]
        rows.append(
            {
                "problem": row["problem"],
                "instance": row["instance"],
                "method": method,
                "backend": row["backend"],
                "transpiled_two_qubit_gates": fmt(int(round(twoq))),
                "fidelity_proxy": fmt(fidelity),
                "hardware_feasible": "true",
                "hardware_optimality_gap_percent": fmt(gap),
                "marker_key": MARKER_KEY[method],
                "problem_color_key": row["problem"],
                "panel": "noise_dominated" if fidelity < BOUNDARY else "signal_preserving",
            }
        )
    return rows


def f(row: dict[str, str], key: str) -> float:
    return safe_float(row[key])


def plot(rows: list[dict[str, str]]) -> None:
    noise = [row for row in rows if row["panel"] == "noise_dominated"]
    signal = [row for row in rows if row["panel"] == "signal_preserving"]
    ymin = min(f(row, "hardware_optimality_gap_percent") for row in rows)
    ymax = max(f(row, "hardware_optimality_gap_percent") for row in rows)
    yrange = ymax - ymin

    fig, (ax_noise, ax_signal) = plt.subplots(
        1,
        2,
        figsize=(9.2, 4.6),
        sharey=True,
        gridspec_kw={"width_ratios": [1.05, 1.35], "wspace": 0.08},
    )
    panels = [
        (ax_noise, noise, "(a) Strongly noise-dominated\n" + r"$F_{\mathrm{est}}<10^{-3}$"),
        (ax_signal, signal, "(b) Signal-preserving\n" + r"$F_{\mathrm{est}}\geq10^{-3}$"),
    ]
    for ax, panel_rows, title in panels:
        ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
        ax.set_title(title, fontsize=10.0, pad=7)
        for row in panel_rows:
            method = row["method"]
            problem = row["problem"]
            ax.scatter(
                f(row, "fidelity_proxy"),
                f(row, "hardware_optimality_gap_percent"),
                marker=METHOD_MARKERS[method],
                s=62,
                facecolor=PROBLEM_COLORS[problem],
                edgecolor="#1a1a1a",
                linewidth=0.55,
                alpha=0.9,
                zorder=2,
            )
    ax_noise.set_ylabel("Hardware optimality gap to BKS (%)")
    for ax in (ax_noise, ax_signal):
        ax.set_xlabel(r"Gate-count fidelity proxy, $F_{\mathrm{est}}$")
        ax.set_ylim(ymin - 0.05 * yrange, ymax + 0.08 * yrange)

    min_noise = min(f(row, "fidelity_proxy") for row in noise) if noise else 1e-100
    ax_noise.set_xlim(min_noise * 0.55, BOUNDARY)
    ax_signal.set_xlim(BOUNDARY, 1.16)
    ax_noise.set_xticks([1e-80, 1e-60, 1e-40, 1e-20])
    ax_noise.set_xticklabels([r"$10^{-80}$", r"$10^{-60}$", r"$10^{-40}$", r"$10^{-20}$"])

    ax_noise.axvline(BOUNDARY, color="#777777", linestyle="--", linewidth=1.0)
    ax_signal.axvline(BOUNDARY, color="#777777", linestyle="--", linewidth=1.0)
    ax_noise.text(BOUNDARY / 1.28, ymin + 0.08 * yrange, r"$F_{\mathrm{est}}=10^{-3}$", ha="right", va="bottom", fontsize=8.2, color="#555555", rotation=90)
    ax_signal.text(BOUNDARY * 1.18, ymin + 0.08 * yrange, r"$F_{\mathrm{est}}=10^{-3}$", ha="left", va="bottom", fontsize=8.2, color="#555555", rotation=90)

    for value, label in [(0.01, r"$F_{\mathrm{est}}=0.01$"), (0.1, r"$F_{\mathrm{est}}=0.1$")]:
        ax_signal.axvline(value, color="#777777", linestyle="--", linewidth=1.0)
        ax_signal.text(value * 1.08, ymax - 0.05 * yrange, label, rotation=90, ha="left", va="top", fontsize=8.2, color="#555555")

    problem_handles = [
        Line2D([0], [0], marker="o", linestyle="", markerfacecolor=color, markeredgecolor="#1a1a1a", markersize=7, label=problem)
        for problem, color in PROBLEM_COLORS.items()
    ]
    method_handles = [
        Line2D([0], [0], marker=METHOD_MARKERS[m], linestyle="", markerfacecolor="#9a9a9a", markeredgecolor="#1a1a1a", markersize=7, label=m)
        for m in METHOD_ORDER
    ]
    fig.legend(
        handles=problem_handles + method_handles,
        title="Problem / method",
        loc="center left",
        bbox_to_anchor=(0.835, 0.52),
        frameon=True,
        fontsize=8.3,
    )
    fig.suptitle("Hardware solution quality vs. gate-count fidelity proxy", y=0.98, fontsize=13)
    fig.text(0.42, 0.025, "Increasing gate-count fidelity proxy ->", ha="center", va="center", fontsize=9.2)
    fig.subplots_adjust(left=0.10, right=0.80, top=0.82, bottom=0.16)
    fig.savefig(OUT / "fig_fidelity_vs_quality_revised.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig_fidelity_vs_quality_revised.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_digest(rows: list[dict[str, str]]) -> None:
    by_panel = defaultdict(list)
    for row in rows:
        by_panel[row["panel"]].append(row)
    lines = [
        "# Figure 5 Fidelity/Quality Digest",
        "",
        "Source data are the feasible MDKP and MIS `feasible_gap` rows from `reviewer_issue_p2_2_figure4_feasibility/figure4_complexity_feasibility_data.csv`; no hardware run is added, removed, or recalculated relative to that plottable finite-gap set.",
        "",
        "## 1. Points Per Panel",
        "",
    ]
    for panel in ["noise_dominated", "signal_preserving"]:
        panel_rows = by_panel[panel]
        counts = Counter(row["problem"] for row in panel_rows)
        lines.append(f"- {panel}: total={len(panel_rows)}, MDKP={counts.get('MDKP', 0)}, MIS={counts.get('MIS', 0)}.")
    lines.extend(["", "## 2. F_est Range Per Panel", ""])
    for panel in ["noise_dominated", "signal_preserving"]:
        vals = [f(row, "fidelity_proxy") for row in by_panel[panel]]
        if vals:
            lines.append(f"- {panel}: min={fmt(min(vals))}, max={fmt(max(vals))}.")
        else:
            lines.append(f"- {panel}: no points.")
    lines.extend(["", "## 3. Feasible Points Per Method And Panel", ""])
    for panel in ["noise_dominated", "signal_preserving"]:
        counts = Counter(row["method"] for row in by_panel[panel])
        method_text = ", ".join(f"{method}={counts.get(method, 0)}" for method in METHOD_ORDER)
        lines.append(f"- {panel}: {method_text}.")
    lines.extend(
        [
            "",
            "## 4. Exclusions",
            "",
            "MSP is excluded because its reported hardware metric is TDev, not percentage optimality gap. QAP is excluded because no reported hardware QAP run is feasible, so it has no finite percentage-gap outcome for this plot.",
            "",
            "## 5. Revision Scope",
            "",
            "No new run was performed. The plotted finite-gap MDKP/MIS points are unchanged; the revision swaps panel order, fixes the x-axis label, uses logarithmic fidelity axes, moves the legend outside the data area, and updates annotations/caption language.",
        ]
    )
    (OUT / "figure5_fidelity_quality_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(OUT / "figure5_fidelity_quality_data.csv", rows)
    write_digest(rows)
    plot(rows)
    print(f"wrote {len(rows)} Figure 5 rows")


if __name__ == "__main__":
    main()
