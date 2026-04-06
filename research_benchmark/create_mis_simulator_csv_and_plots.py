#!/usr/bin/env python
"""Create MIS simulator CSVs and family gap-to-BKS plots from provided summary tables."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

INSTANCE_ORDER = [
    "1tc.8",
    "1tc.16",
    "1tc.32",
    "1tc.64",
    "1et.64",
    "1dc.64",
    "1dc.128",
]

METHOD_LABELS = {
    "pce": "PCE",
    "qrao": "QRAO",
    "vqe": "VQE",
    "cvar_vqe": "CVaR-VQE",
    "qaoa": "QAOA",
    "ma_qaoa": "MA-QAOA",
}

METHOD_COLORS = {
    "pce": "#0072B2",
    "qrao": "#D55E00",
    "vqe": "#009E73",
    "cvar_vqe": "#CC79A7",
    "qaoa": "#E69F00",
    "ma_qaoa": "#56B4E9",
}


def _norm_instance(inst: str) -> str:
    return str(inst).strip()


def _gap_from_rsq(value: Any) -> tuple[str, float]:
    text = str(value).strip().lower()
    if text in {"inf", "infty", "infinity", "∞"}:
        return ("inf", float("inf"))
    if text == "me":
        return ("ME", float("nan"))
    return ("finite", 100.0 - float(value))


def _gap_from_gap_value(value: Any) -> tuple[str, float]:
    text = str(value).strip().lower()
    if text in {"inf", "infty", "infinity", "∞"}:
        return ("inf", float("inf"))
    if text == "me":
        return ("ME", float("nan"))
    return ("finite", float(value))


def _encoding_table() -> pd.DataFrame:
    rows = [
        {"Instance": "1tc.8", "OptimalBKS": 4, "Variables": 8, "PCE_Qubits": 3, "PCE_Feas": "Yes", "PCE_GapRaw": 0.0, "QRAO_Qubits": 4, "QRAO_Feas": "Yes", "QRAO_GapRaw": 0.0},
        {"Instance": "1tc.16", "OptimalBKS": 8, "Variables": 16, "PCE_Qubits": 4, "PCE_Feas": "Yes", "PCE_GapRaw": 0.0, "QRAO_Qubits": 6, "QRAO_Feas": "Yes", "QRAO_GapRaw": 12.5},
        {"Instance": "1tc.32", "OptimalBKS": 12, "Variables": 32, "PCE_Qubits": 6, "PCE_Feas": "No", "PCE_GapRaw": "inf", "QRAO_Qubits": 13, "QRAO_Feas": "Yes", "QRAO_GapRaw": 41.67},
        {"Instance": "1tc.64", "OptimalBKS": 20, "Variables": 64, "PCE_Qubits": 8, "PCE_Feas": "Yes", "PCE_GapRaw": 5.0, "QRAO_Qubits": 23, "QRAO_Feas": "Yes", "QRAO_GapRaw": 60.0},
        {"Instance": "1et.64", "OptimalBKS": 18, "Variables": 64, "PCE_Qubits": 8, "PCE_Feas": "Yes", "PCE_GapRaw": 11.1, "QRAO_Qubits": 24, "QRAO_Feas": "Yes", "QRAO_GapRaw": 27.78},
        {"Instance": "1dc.64", "OptimalBKS": 8, "Variables": 64, "PCE_Qubits": 8, "PCE_Feas": "Yes", "PCE_GapRaw": 12.5, "QRAO_Qubits": 18, "QRAO_Feas": "Yes", "QRAO_GapRaw": 25.0},
        {"Instance": "1dc.128", "OptimalBKS": 16, "Variables": 128, "PCE_Qubits": 10, "PCE_Feas": "No", "PCE_GapRaw": "inf", "QRAO_Qubits": 46, "QRAO_Feas": "No", "QRAO_GapRaw": "inf"},
    ]
    df = pd.DataFrame(rows)
    df["Instance"] = df["Instance"].map(_norm_instance)
    return df


def _vqe_table() -> pd.DataFrame:
    rows = [
        {"Instance": "1tc.8", "OptimalBKS": 4, "Variables": 8, "VQE_Feas": "Yes", "VQE_GapRaw": 0.0, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapRaw": 0.0},
        {"Instance": "1tc.16", "OptimalBKS": 8, "Variables": 16, "VQE_Feas": "Yes", "VQE_GapRaw": 25.0, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapRaw": 12.5},
        {"Instance": "1tc.32", "OptimalBKS": 12, "Variables": 32, "VQE_Feas": "Yes", "VQE_GapRaw": 25.0, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapRaw": 8.3},
        {"Instance": "1tc.64", "OptimalBKS": 20, "Variables": 64, "VQE_Feas": "Yes", "VQE_GapRaw": 60.0, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapRaw": 60.0},
        {"Instance": "1et.64", "OptimalBKS": 18, "Variables": 64, "VQE_Feas": "Yes", "VQE_GapRaw": 22.2, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapRaw": 22.2},
        {"Instance": "1dc.64", "OptimalBKS": 8, "Variables": 64, "VQE_Feas": "Yes", "VQE_GapRaw": 12.5, "CVAR_VQE_Feas": "Yes", "CVAR_VQE_GapRaw": 37.5},
    ]
    df = pd.DataFrame(rows)
    df["Instance"] = df["Instance"].map(_norm_instance)
    return df


def _qaoa_table() -> pd.DataFrame:
    rows = [
        {"Instance": "1tc.8", "OptimalBKS": 4, "Variables": 8, "QAOA_Feas": "Yes", "QAOA_RSQRaw": 100.0, "MA_QAOA_Feas": "Yes", "MA_QAOA_RSQRaw": 100.0},
        {"Instance": "1tc.16", "OptimalBKS": 8, "Variables": 16, "QAOA_Feas": "Yes", "QAOA_RSQRaw": 100.0, "MA_QAOA_Feas": "Yes", "MA_QAOA_RSQRaw": 50.0},
        {"Instance": "1tc.32", "OptimalBKS": 12, "Variables": 32, "QAOA_Feas": "Yes", "QAOA_RSQRaw": 83.3, "MA_QAOA_Feas": "Yes", "MA_QAOA_RSQRaw": 91.7},
        {"Instance": "1tc.64", "OptimalBKS": 20, "Variables": 64, "QAOA_Feas": "Yes", "QAOA_RSQRaw": 50.0, "MA_QAOA_Feas": "No", "MA_QAOA_RSQRaw": "inf"},
        {"Instance": "1et.64", "OptimalBKS": 18, "Variables": 64, "QAOA_Feas": "No", "QAOA_RSQRaw": "inf", "MA_QAOA_Feas": "No", "MA_QAOA_RSQRaw": "inf"},
        {"Instance": "1dc.64", "OptimalBKS": 8, "Variables": 64, "QAOA_Feas": "ME", "QAOA_RSQRaw": "ME", "MA_QAOA_Feas": "No", "MA_QAOA_RSQRaw": "inf"},
    ]
    df = pd.DataFrame(rows)
    df["Instance"] = df["Instance"].map(_norm_instance)
    return df


def _to_long(encoding_df: pd.DataFrame, vqe_df: pd.DataFrame, qaoa_df: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for _, row in encoding_df.iterrows():
        for method, qcol, fcol, gcol in (
            ("pce", "PCE_Qubits", "PCE_Feas", "PCE_GapRaw"),
            ("qrao", "QRAO_Qubits", "QRAO_Feas", "QRAO_GapRaw"),
        ):
            tag, gap = _gap_from_gap_value(row[gcol])
            rows.append(
                {
                    "Instance": row["Instance"],
                    "OptimalBKS": row["OptimalBKS"],
                    "Variables": row["Variables"],
                    "Method": method,
                    "MethodLabel": METHOD_LABELS[method],
                    "Qubits": row[qcol],
                    "Feas": str(row[fcol]),
                    "GapRaw": str(row[gcol]),
                    "GapTag": tag,
                    "GapPct": gap,
                    "Family": "encoding",
                }
            )

    for _, row in vqe_df.iterrows():
        for method, fcol, gcol in (
            ("vqe", "VQE_Feas", "VQE_GapRaw"),
            ("cvar_vqe", "CVAR_VQE_Feas", "CVAR_VQE_GapRaw"),
        ):
            tag, gap = _gap_from_gap_value(row[gcol])
            rows.append(
                {
                    "Instance": row["Instance"],
                    "OptimalBKS": row["OptimalBKS"],
                    "Variables": row["Variables"],
                    "Method": method,
                    "MethodLabel": METHOD_LABELS[method],
                    "Qubits": np.nan,
                    "Feas": str(row[fcol]),
                    "GapRaw": str(row[gcol]),
                    "GapTag": tag,
                    "GapPct": gap,
                    "Family": "vqe",
                }
            )

    for _, row in qaoa_df.iterrows():
        for method, fcol, rsq_col in (
            ("qaoa", "QAOA_Feas", "QAOA_RSQRaw"),
            ("ma_qaoa", "MA_QAOA_Feas", "MA_QAOA_RSQRaw"),
        ):
            tag, gap = _gap_from_rsq(row[rsq_col])
            rows.append(
                {
                    "Instance": row["Instance"],
                    "OptimalBKS": row["OptimalBKS"],
                    "Variables": row["Variables"],
                    "Method": method,
                    "MethodLabel": METHOD_LABELS[method],
                    "Qubits": np.nan,
                    "Feas": str(row[fcol]),
                    "GapRaw": str(row[rsq_col]),
                    "GapTag": tag,
                    "GapPct": gap,
                    "Family": "qaoa",
                }
            )

    df = pd.DataFrame(rows)
    df["Instance"] = pd.Categorical(df["Instance"], categories=INSTANCE_ORDER, ordered=True)
    return df.sort_values(["Instance", "Method"])


def _plot_family(
    *,
    long_df: pd.DataFrame,
    family_methods: list[str],
    title: str,
    out_stem: Path,
    dpi: int,
) -> None:
    family_df = long_df[long_df["Method"].isin(family_methods)].copy()
    if family_df.empty:
        return

    finite_df = family_df[family_df["GapTag"] == "finite"].copy()
    if finite_df.empty:
        return

    instances = [inst for inst in INSTANCE_ORDER if inst in set(family_df["Instance"].astype(str))]
    methods = [m for m in family_methods if m in set(family_df["Method"].astype(str))]
    palette = {m: METHOD_COLORS[m] for m in methods}

    fig, ax = plt.subplots(figsize=(12.6, 5.0))
    sns.barplot(
        data=finite_df,
        x="Instance",
        y="GapPct",
        hue="Method",
        hue_order=methods,
        order=instances,
        palette=palette,
        dodge=True,
        edgecolor="black",
        linewidth=0.35,
        ax=ax,
    )

    hatches = ["", "//", "\\\\", "xx"]
    for idx, container in enumerate(ax.containers):
        for bar in container:
            bar.set_hatch(hatches[idx % len(hatches)])
        labels = [
            f"{float(bar.get_height()):.1f}" if float(bar.get_height()) > 1e-9 else ""
            for bar in container
        ]
        ax.bar_label(container, labels=labels, fontsize=6.5, rotation=90, padding=1)

    finite_max = float(finite_df["GapPct"].max())
    y_top = max(1.0, finite_max * 1.22)
    ax.set_ylim(0.0, y_top)

    # Mark non-finite values (inf / ME) above bars.
    bar_width = 0.8 / len(methods)
    x_index = {inst: i for i, inst in enumerate(instances)}
    m_index = {m: i for i, m in enumerate(methods)}
    non_finite = family_df[family_df["GapTag"] != "finite"]
    for _, row in non_finite.iterrows():
        inst = str(row["Instance"])
        method = str(row["Method"])
        if inst not in x_index or method not in m_index:
            continue
        x = x_index[inst] - 0.4 + (m_index[method] + 0.5) * bar_width
        tag = str(row["GapTag"])
        if tag == "inf":
            y = y_top * 0.93
            ax.scatter([x], [y], marker="^", s=72, color=METHOD_COLORS[method], edgecolors="black", linewidths=0.4, zorder=5)
            ax.annotate("inf", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", va="bottom", fontsize=8)
        else:
            y = y_top * 0.82
            ax.scatter([x], [y], marker="X", s=68, color=METHOD_COLORS[method], edgecolors="black", linewidths=0.35, zorder=5)
            ax.annotate("ME", (x, y), textcoords="offset points", xytext=(0, 5), ha="center", va="bottom", fontsize=8)

    ax.text(
        0.995,
        0.98,
        "▲ = inf, X = ME",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=8,
    )
    ax.set_title(title, fontsize=14, weight="bold")
    ax.set_xlabel("Instance")
    ax.set_ylabel("Gap to BKS (%)")
    ax.grid(alpha=0.28, axis="y")
    handles, labels = ax.get_legend_handles_labels()
    pretty_labels = [METHOD_LABELS.get(lbl, lbl.upper()) for lbl in labels]
    ax.legend(handles, pretty_labels, title="Method", frameon=True, fontsize=9.0, title_fontsize=10.0)

    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create MIS simulator CSV tables and gap plots.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/mis/plots"),
        help="Output directory for CSVs and plot files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI for PNG export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["legend.framealpha"] = 0.95

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    encoding_df = _encoding_table()
    vqe_df = _vqe_table()
    qaoa_df = _qaoa_table()
    long_df = _to_long(encoding_df, vqe_df, qaoa_df)

    encoding_df.to_csv(output_dir / "mis_simulator_encoding_results_wide.csv", index=False)
    vqe_df.to_csv(output_dir / "mis_simulator_vqe_results_wide.csv", index=False)
    qaoa_df.to_csv(output_dir / "mis_simulator_qaoa_results_wide.csv", index=False)
    long_df.to_csv(output_dir / "mis_simulator_results_long.csv", index=False)

    _plot_family(
        long_df=long_df,
        family_methods=["pce", "qrao"],
        title="MIS Simulator: Gap to BKS Across Instances (Encoding Methods)",
        out_stem=output_dir / "mis_simulator_gap_to_bks_encoding_methods",
        dpi=int(args.dpi),
    )
    _plot_family(
        long_df=long_df,
        family_methods=["vqe", "cvar_vqe"],
        title="MIS Simulator: Gap to BKS Across Instances (VQE Variants)",
        out_stem=output_dir / "mis_simulator_gap_to_bks_vqe_variants",
        dpi=int(args.dpi),
    )
    _plot_family(
        long_df=long_df,
        family_methods=["qaoa", "ma_qaoa"],
        title="MIS Simulator: Gap to BKS Across Instances (QAOA Variants)",
        out_stem=output_dir / "mis_simulator_gap_to_bks_qaoa_variants",
        dpi=int(args.dpi),
    )

    print(f"Wrote: {output_dir / 'mis_simulator_encoding_results_wide.csv'}")
    print(f"Wrote: {output_dir / 'mis_simulator_vqe_results_wide.csv'}")
    print(f"Wrote: {output_dir / 'mis_simulator_qaoa_results_wide.csv'}")
    print(f"Wrote: {output_dir / 'mis_simulator_results_long.csv'}")
    print(f"Wrote: {output_dir / 'mis_simulator_gap_to_bks_encoding_methods.png'}")
    print(f"Wrote: {output_dir / 'mis_simulator_gap_to_bks_vqe_variants.png'}")
    print(f"Wrote: {output_dir / 'mis_simulator_gap_to_bks_qaoa_variants.png'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
