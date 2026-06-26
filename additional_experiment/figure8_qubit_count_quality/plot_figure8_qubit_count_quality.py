#!/usr/bin/env python3
"""Plot qubit count versus recovered hardware solution quality for MDKP and MIS."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


REPO_ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
RESULTS = REPO_ROOT / "research_benchmark" / "research_benchmark" / "results_hardware"

PROBLEMS = {
    "MDKP": "mkp",
    "MIS": "mis",
}
METHODS = ["pce", "qrao", "vqe", "cvar_vqe"]
METHOD_LABEL = {
    "pce": "PCE",
    "qrao": "QRAO",
    "vqe": "VQE",
    "cvar_vqe": "CVaR-VQE",
}
METHOD_COLOR = {
    "pce": "#8c564b",
    "qrao": "#e377c2",
    "vqe": "#1f77b4",
    "cvar_vqe": "#ff7f0e",
}
METHOD_MARKER = {
    "pce": "o",
    "qrao": "s",
    "vqe": "^",
    "cvar_vqe": "D",
}
CSV_COLUMNS = [
    "problem",
    "instance",
    "method",
    "method_label",
    "qubit_count",
    "gap_to_bks_percent",
    "hardware_feasible",
    "source_table",
]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def safe_float(value: Any) -> float:
    try:
        text = str(value).strip()
        if text.lower() in {"", "inf", "infinity", "nan"}:
            return math.nan
        return float(text)
    except Exception:
        return math.nan


def fmt(value: Any) -> str:
    if isinstance(value, float):
        if math.isnan(value):
            return "not_available"
        return f"{value:.12g}"
    if isinstance(value, bool):
        return str(value).lower()
    text = str(value).strip()
    return text if text else "not_available"


def main_table(problem_key: str, method: str) -> Path:
    method_dir = RESULTS / problem_key / f"{problem_key}_{method}" / "csv"
    candidates = sorted(method_dir.glob(f"main_table_{method}.csv")) + sorted(method_dir.glob("main_table.csv"))
    if not candidates:
        raise FileNotFoundError(f"No main_table CSV found for {problem_key}/{method}")
    return candidates[0]


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for problem_label, problem_key in PROBLEMS.items():
        for method in METHODS:
            source = main_table(problem_key, method)
            for row in read_csv(source):
                gap = safe_float(row.get("Gap%"))
                qubits = safe_float(row.get("Qubits"))
                feasible = str(row.get("Feas", "")).strip().lower() in {"yes", "true"}
                if not (feasible and math.isfinite(gap) and math.isfinite(qubits)):
                    continue
                rows.append(
                    {
                        "problem": problem_label,
                        "instance": row["Instance"],
                        "method": method,
                        "method_label": METHOD_LABEL[method],
                        "qubit_count": fmt(qubits),
                        "gap_to_bks_percent": fmt(gap),
                        "hardware_feasible": fmt(feasible),
                        "source_table": str(source.relative_to(REPO_ROOT)),
                    }
                )
    return rows


def plot(rows: list[dict[str, str]]) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 6.1))

    # Light background bands mirror the PCE, QRAO, and full-width qubit-count regimes.
    ax.axvspan(0, 15, color=METHOD_COLOR["pce"], alpha=0.045, linewidth=0)
    ax.axvspan(25, 70, color=METHOD_COLOR["qrao"], alpha=0.045, linewidth=0)
    ax.axvspan(75, 130, color=METHOD_COLOR["vqe"], alpha=0.035, linewidth=0)
    ax.text(2, 57.5, "PCE\nregion", color=METHOD_COLOR["pce"], fontsize=8, alpha=0.6, style="italic")
    ax.text(28, 57.5, "QRAO\nregion", color=METHOD_COLOR["qrao"], fontsize=8, alpha=0.6, style="italic")
    ax.text(75, 57.5, "Full-width\nregion", color=METHOD_COLOR["vqe"], fontsize=8, alpha=0.45, style="italic")

    for row in rows:
        method = row["method"]
        problem = row["problem"]
        x = safe_float(row["qubit_count"])
        y = safe_float(row["gap_to_bks_percent"])
        if problem == "MDKP":
            facecolor = METHOD_COLOR[method]
            alpha = 0.82
        else:
            facecolor = "none"
            alpha = 0.72
        ax.scatter(
            x,
            y,
            marker=METHOD_MARKER[method],
            s=75,
            facecolors=facecolor,
            edgecolors=METHOD_COLOR[method],
            linewidths=1.15,
            alpha=alpha,
            zorder=3,
        )

    annotations = [
        ("1tc.16\n(PCE: 62.5%)", "MIS", "pce", 4, 62.5, 8, 0),
        ("pet2\n(QRAO: 0.6%)", "MDKP", "qrao", 44, 0.64, 6, -16),
        ("pet2\n(CVaR: 1.5%)", "MDKP", "cvar_vqe", 99, 1.50, 10, 20),
        ("pet4\n(VQE: 44.7%)", "MDKP", "vqe", 107, 44.7, 8, 16),
    ]
    for text, _problem, method, x, y, dx, dy in annotations:
        ax.annotate(
            text,
            xy=(x, y),
            xytext=(dx, dy),
            textcoords="offset points",
            fontsize=7.0,
            color=METHOD_COLOR[method],
            arrowprops={"arrowstyle": "-", "color": METHOD_COLOR[method], "linewidth": 0.8},
        )

    handles = [
        Line2D(
            [0],
            [0],
            marker=METHOD_MARKER[method],
            linestyle="",
            markerfacecolor=METHOD_COLOR[method],
            markeredgecolor=METHOD_COLOR[method],
            markersize=7.5,
            label=f"{METHOD_LABEL[method]} (MDKP)",
        )
        for method in METHODS
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="",
            markerfacecolor="none",
            markeredgecolor="#777777",
            markersize=7.5,
            label="MIS (open markers)",
        )
    )

    ax.set_title(
        "Qubit Count vs. Recovered Solution Quality on Hardware\n"
        "(MDKP: filled markers, MIS: open markers)",
        fontsize=12.0,
        weight="bold",
    )
    ax.set_xlabel("Qubit Count")
    ax.set_ylabel("Gap to BKS (%)")
    ax.set_xlim(-2, 135)
    ax.set_ylim(-3, 68)
    ax.grid(alpha=0.28)
    ax.legend(handles=handles, loc="upper right", fontsize=8.0, frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / "fig8_qubit_count_vs_quality_revised.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig8_qubit_count_vs_quality_revised.png", dpi=240, bbox_inches="tight")
    plt.close(fig)


def write_readme() -> None:
    text = """# Figure 8 Qubit Count Revision

This folder reconstructs the MDKP/MIS qubit-count versus recovered hardware
solution-quality figure from the checked-in hardware `main_table` CSV files.

The only requested label change is applied in code:

```python
ax.set_xlabel("Qubit Count")
```

## Files

- `plot_figure8_qubit_count_quality.py`: plotting script.
- `figure8_qubit_count_quality_data.csv`: exact data used by the plot.
- `fig8_qubit_count_vs_quality_revised.pdf`: regenerated figure.
- `fig8_qubit_count_vs_quality_revised.png`: raster preview.

## Rerun

```bash
.venv/bin/python additional_experiment/figure8_qubit_count_quality/plot_figure8_qubit_count_quality.py
```
"""
    (OUT / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = build_rows()
    write_csv(OUT / "figure8_qubit_count_quality_data.csv", rows)
    plot(rows)
    write_readme()
    print(f"wrote {len(rows)} Figure 8 rows")


if __name__ == "__main__":
    main()
