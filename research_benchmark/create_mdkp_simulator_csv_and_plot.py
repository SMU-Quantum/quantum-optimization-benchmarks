#!/usr/bin/env python
"""Create MDKP simulator CSVs and an Instance vs Gap plot."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INSTANCE_ORDER = [
    "hp1",
    "hp2",
    "pb1",
    "pb2",
    "pb4",
    "pb5",
    "pet2",
    "pet3",
    "pet4",
    "pet5",
    "pet6",
    "pet7",
]

METHOD_ORDER = ["pce", "vqe", "cvar_vqe"]
METHOD_LABEL = {"pce": "PCE", "vqe": "VQE", "cvar_vqe": "CVaR-VQE"}
METHOD_COLORS = {
    "pce": "#0072B2",       # blue
    "vqe": "#009E73",       # bluish green
    "cvar_vqe": "#CC79A7",  # reddish purple
}


def _table_rows() -> list[dict[str, Any]]:
    return [
        {"Instance": "hp1", "OptimalBKS": 3418, "Variables": 60, "PCE_Qubits": 7, "PCE_Feas": "Yes", "PCE_GapPct": 20.88, "VQE_Feas": "Yes", "VQE_GapPct": 39.76, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 20.59},
        {"Instance": "hp2", "OptimalBKS": 3186, "Variables": 67, "PCE_Qubits": 8, "PCE_Feas": "Yes", "PCE_GapPct": 38.38, "VQE_Feas": "Yes", "VQE_GapPct": 12.34, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 21.78},
        {"Instance": "pb1", "OptimalBKS": 3090, "Variables": 59, "PCE_Qubits": 7, "PCE_Feas": "Yes", "PCE_GapPct": 14.04, "VQE_Feas": "Yes", "VQE_GapPct": 19.94, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 23.43},
        {"Instance": "pb2", "OptimalBKS": 3186, "Variables": 66, "PCE_Qubits": 8, "PCE_Feas": "Yes", "PCE_GapPct": 14.37, "VQE_Feas": "Yes", "VQE_GapPct": 19.49, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 22.78},
        {"Instance": "pb4", "OptimalBKS": 95168, "Variables": 45, "PCE_Qubits": 6, "PCE_Feas": "Yes", "PCE_GapPct": 32.91, "VQE_Feas": "No", "VQE_GapPct": math.inf, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 39.38},
        {"Instance": "pb5", "OptimalBKS": 2139, "Variables": 116, "PCE_Qubits": 10, "PCE_Feas": "Yes", "PCE_GapPct": 11.87, "VQE_Feas": "Yes", "VQE_GapPct": 4.25, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 33.34},
        {"Instance": "pet2", "OptimalBKS": 87061, "Variables": 99, "PCE_Qubits": 9, "PCE_Feas": "Yes", "PCE_GapPct": 28.19, "VQE_Feas": "Yes", "VQE_GapPct": 41.07, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 30.37},
        {"Instance": "pet3", "OptimalBKS": 4015, "Variables": 102, "PCE_Qubits": 9, "PCE_Feas": "Yes", "PCE_GapPct": 15.56, "VQE_Feas": "Yes", "VQE_GapPct": 4.98, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 34.74},
        {"Instance": "pet4", "OptimalBKS": 6120, "Variables": 107, "PCE_Qubits": 9, "PCE_Feas": "Yes", "PCE_GapPct": 50.98, "VQE_Feas": "Yes", "VQE_GapPct": 66.58, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 48.44},
        {"Instance": "pet5", "OptimalBKS": 12400, "Variables": 122, "PCE_Qubits": 10, "PCE_Feas": "Yes", "PCE_GapPct": 22.74, "VQE_Feas": "Yes", "VQE_GapPct": 33.23, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 53.15},
        {"Instance": "pet6", "OptimalBKS": 10618, "Variables": 86, "PCE_Qubits": 9, "PCE_Feas": "Yes", "PCE_GapPct": 33.84, "VQE_Feas": "Yes", "VQE_GapPct": 12.50, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 40.41},
        {"Instance": "pet7", "OptimalBKS": 16537, "Variables": 100, "PCE_Qubits": 9, "PCE_Feas": "Yes", "PCE_GapPct": 15.87, "VQE_Feas": "Yes", "VQE_GapPct": 43.46, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapPct": 46.47},
    ]


def _to_long(wide_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for _, row in wide_df.iterrows():
        for method in METHOD_ORDER:
            if method == "pce":
                gap = row["PCE_GapPct"]
                feas = row["PCE_Feas"]
                qubits = row["PCE_Qubits"]
            elif method == "vqe":
                gap = row["VQE_GapPct"]
                feas = row["VQE_Feas"]
                qubits = np.nan
            else:
                gap = row["CVAR_VQE_GapPct"]
                feas = row["CVAR_VQE_Feas"]
                qubits = np.nan

            rows.append(
                {
                    "Instance": row["Instance"],
                    "OptimalBKS": row["OptimalBKS"],
                    "Variables": row["Variables"],
                    "Method": method,
                    "MethodLabel": METHOD_LABEL[method],
                    "GapPct": float(gap),
                    "Feas": str(feas),
                    "Qubits": qubits,
                }
            )
    long_df = pd.DataFrame(rows)
    long_df["Instance"] = pd.Categorical(long_df["Instance"], categories=INSTANCE_ORDER, ordered=True)
    long_df = long_df.sort_values(["Instance", "Method"], key=lambda s: s.map({m: i for i, m in enumerate(METHOD_ORDER)}))
    return long_df


def _plot_gap_by_instance(long_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    finite_df = long_df[np.isfinite(long_df["GapPct"])].copy()
    inf_df = long_df[np.isinf(long_df["GapPct"])].copy()

    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    sns.barplot(
        data=finite_df,
        x="Instance",
        y="GapPct",
        hue="Method",
        hue_order=METHOD_ORDER,
        order=INSTANCE_ORDER,
        palette=METHOD_COLORS,
        edgecolor="black",
        linewidth=0.35,
        ax=ax,
    )
    hatches = ["", "//", "\\\\"]
    for idx, container in enumerate(ax.containers):
        for bar in container:
            bar.set_hatch(hatches[idx % len(hatches)])
        labels = [f"{float(bar.get_height()):.1f}" for bar in container]
        ax.bar_label(container, labels=labels, fontsize=6.5, rotation=90, padding=1)

    finite_max = float(finite_df["GapPct"].max()) if not finite_df.empty else 1.0
    y_top = finite_max * 1.18
    ax.set_ylim(0.0, y_top)

    if not inf_df.empty:
        bar_width = 0.8 / len(METHOD_ORDER)
        instance_to_x = {inst: idx for idx, inst in enumerate(INSTANCE_ORDER)}
        method_to_idx = {m: i for i, m in enumerate(METHOD_ORDER)}
        for _, row in inf_df.iterrows():
            inst = str(row["Instance"])
            method = str(row["Method"])
            if inst not in instance_to_x or method not in method_to_idx:
                continue
            x_idx = instance_to_x[inst]
            m_idx = method_to_idx[method]
            x = x_idx - 0.4 + (m_idx + 0.5) * bar_width
            y = y_top * 0.93
            ax.scatter([x], [y], marker="^", s=72, color=METHOD_COLORS[method], edgecolors="black", linewidths=0.4, zorder=5)
            ax.annotate("inf", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", va="bottom", fontsize=8)

    ax.set_title("MDKP Simulator: Gap to BKS Across Instances", fontsize=14, weight="bold")
    ax.set_xlabel("Instance")
    ax.set_ylabel("Gap to BKS (%)")
    ax.grid(alpha=0.28, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    labels = [METHOD_LABEL.get(lbl, lbl) for lbl in labels]
    ax.legend(handles, labels, title="Method", frameon=True, fontsize=9.5, title_fontsize=10.5)

    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "mdkp_simulator_gap_to_bks_by_instance.png", dpi=dpi, bbox_inches="tight")
    fig.savefig(out_dir / "mdkp_simulator_gap_to_bks_by_instance.pdf", bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create CSV and plot for MDKP simulator summary table.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/mkp/plots"),
        help="Output directory for CSV and plot files.",
    )
    parser.add_argument("--dpi", type=int, default=400, help="PNG DPI.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["legend.framealpha"] = 0.95

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    wide_df = pd.DataFrame(_table_rows())
    wide_df["Instance"] = pd.Categorical(wide_df["Instance"], categories=INSTANCE_ORDER, ordered=True)
    wide_df = wide_df.sort_values("Instance")
    long_df = _to_long(wide_df)

    wide_path = output_dir / "mdkp_simulator_results_wide.csv"
    long_path = output_dir / "mdkp_simulator_results_long.csv"
    wide_df.to_csv(wide_path, index=False)
    long_df.to_csv(long_path, index=False)

    _plot_gap_by_instance(long_df, output_dir, dpi=args.dpi)

    print(f"Wrote: {wide_path}")
    print(f"Wrote: {long_path}")
    print(f"Wrote: {output_dir / 'mdkp_simulator_gap_to_bks_by_instance.png'}")
    print(f"Wrote: {output_dir / 'mdkp_simulator_gap_to_bks_by_instance.pdf'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
