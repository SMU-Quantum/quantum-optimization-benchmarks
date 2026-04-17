#!/usr/bin/env python
"""Generate publication-style MKP hardware comparison plots across methods."""

from __future__ import annotations

import argparse
import json
import math
import re
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

METHOD_ORDER = ["pce", "qrao", "vqe", "cvar_vqe", "qaoa", "ma_qaoa", "ws_qaoa"]

METHOD_LABELS = {
    "pce": "PCE",
    "qrao": "QRAO",
    "vqe": "VQE",
    "cvar_vqe": "CVaR-VQE",
    "qaoa": "QAOA",
    "ma_qaoa": "MA-QAOA",
    "ws_qaoa": "WS-QAOA",
}

METHOD_FAMILY = {
    "pce": "Encoding",
    "qrao": "Encoding",
    "vqe": "VQE-family",
    "cvar_vqe": "VQE-family",
    "qaoa": "QAOA-family",
    "ma_qaoa": "QAOA-family",
    "ws_qaoa": "QAOA-family",
}

ENCODING_METHODS = ("pce", "qrao")

# Okabe-Ito style (paper-friendly, colorblind-safe) mapping
METHOD_COLORS = {
    "pce": "#0072B2",       # blue
    "qrao": "#D55E00",      # vermillion
    "vqe": "#009E73",       # bluish green
    "cvar_vqe": "#CC79A7",  # reddish purple
    "qaoa": "#E69F00",      # orange
    "ma_qaoa": "#56B4E9",   # sky blue
    "ws_qaoa": "#7A7A7A",   # neutral gray
}

METHOD_FAMILY_COLORS = {
    "Encoding": "#1B5E20",
    "VQE-family": "#0D47A1",
    "QAOA-family": "#8E24AA",
}


def _safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if "inf" in text:
            return math.inf
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _method_label(method: str) -> str:
    return METHOD_LABELS.get(method, method.upper())


def _method_family(method: str) -> str:
    return METHOD_FAMILY.get(method, "Other")


def _method_sort_key(method: str) -> int:
    if method in METHOD_ORDER:
        return METHOD_ORDER.index(method)
    return len(METHOD_ORDER) + 1


def _find_single_csv(csv_dir: Path, pattern: str) -> Path:
    matches = sorted(csv_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No file matching {pattern} under {csv_dir}")
    return matches[0]


def _discover_method_dirs(mkp_root: Path) -> list[Path]:
    out: list[Path] = []
    for path in sorted(mkp_root.glob("mkp_*")):
        if not path.is_dir():
            continue
        if (path / "csv").is_dir():
            out.append(path)
    return out


def _load_method_tables(
    mkp_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    main_rows: list[pd.DataFrame] = []
    appendix_a_rows: list[pd.DataFrame] = []
    appendix_b_rows: list[pd.DataFrame] = []
    job_latency_rows: list[dict[str, Any]] = []

    for method_dir in _discover_method_dirs(mkp_root):
        method = method_dir.name.replace("mkp_", "")
        csv_dir = method_dir / "csv"
        try:
            main_path = _find_single_csv(csv_dir, "main_table*.csv")
            a_path = _find_single_csv(csv_dir, "appendix_table_a*_circuit_compilation.csv")
            b_path = _find_single_csv(csv_dir, "appendix_table_b*_execution_robustness.csv")
        except FileNotFoundError:
            continue

        main_df = pd.read_csv(main_path)
        main_df["Method"] = method
        main_df["MethodLabel"] = _method_label(method)
        main_df["MethodFamily"] = _method_family(method)
        main_df["Instance"] = main_df["Instance"].astype(str)
        main_df["GapPct"] = main_df["Gap%"].apply(_safe_float)
        main_df["Qubits"] = pd.to_numeric(main_df["Qubits"], errors="coerce")
        main_df["Evals"] = pd.to_numeric(main_df["#Evals"], errors="coerce")
        main_rows.append(main_df)

        a_df = pd.read_csv(a_path)
        a_df["Method"] = method
        a_df["MethodLabel"] = _method_label(method)
        a_df["MethodFamily"] = _method_family(method)
        a_df["Instance"] = a_df["Instance"].astype(str)
        for col in (
            "LogicalQubits",
            "LogicalDepth",
            "LogicalGateCount",
            "Logical2QGates",
            "Parameters",
            "BackendQubits",
            "TranspiledDepth",
            "TranspiledTotalGates",
            "Transpiled2QGates",
        ):
            if col in a_df.columns:
                a_df[col] = pd.to_numeric(a_df[col], errors="coerce")
        appendix_a_rows.append(a_df)

        b_df = pd.read_csv(b_path)
        b_df["Method"] = method
        b_df["MethodLabel"] = _method_label(method)
        b_df["MethodFamily"] = _method_family(method)
        b_df["Instance"] = b_df["Instance"].astype(str)
        for col in (
            "ShotsPerEval",
            "#Evals",
            "MedianJobLatencySec",
            "P95JobLatencySec",
            "TotalWallClockMinutes",
        ):
            if col in b_df.columns:
                b_df[col] = pd.to_numeric(b_df[col], errors="coerce")
        if "#Jobs(ok/failed)" in b_df.columns:
            jobs = b_df["#Jobs(ok/failed)"].astype(str).str.extract(
                r"(?P<JobsOK>\d+)\s*/\s*(?P<JobsFailed>\d+)"
            )
            b_df["JobsOK"] = pd.to_numeric(jobs["JobsOK"], errors="coerce")
            b_df["JobsFailed"] = pd.to_numeric(jobs["JobsFailed"], errors="coerce")
        appendix_b_rows.append(b_df)

        for result_path in sorted(method_dir.glob("*/result.json")):
            instance = result_path.parent.name.replace("_dat", "")
            with result_path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            job_meta = payload.get("job_metadata", [])
            if not isinstance(job_meta, list):
                continue
            for meta in job_meta:
                if not isinstance(meta, dict):
                    continue
                latency = _safe_float(meta.get("elapsed_sec"))
                backend = str(meta.get("backend_name", "")).strip()
                if latency is None or not backend:
                    continue
                job_latency_rows.append(
                    {
                        "Instance": instance,
                        "Method": method,
                        "MethodLabel": _method_label(method),
                        "MethodFamily": _method_family(method),
                        "Backend": backend,
                        "LatencySec": float(latency),
                    }
                )

    if not main_rows or not appendix_a_rows or not appendix_b_rows:
        raise RuntimeError("Could not load MKP CSV tables. Check mkp root path and csv contents.")

    main_all = pd.concat(main_rows, ignore_index=True)
    a_all = pd.concat(appendix_a_rows, ignore_index=True)
    b_all = pd.concat(appendix_b_rows, ignore_index=True)
    job_latency_all = pd.DataFrame(job_latency_rows)
    return main_all, a_all, b_all, job_latency_all


def _save_figure(fig: plt.Figure, out_stem: Path, dpi: int) -> None:
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(out_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def _method_palette(methods: list[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    fallback = sns.color_palette("colorblind", n_colors=max(3, len(methods)))
    fidx = 0
    for method in methods:
        if method in METHOD_COLORS:
            out[method] = METHOD_COLORS[method]
        else:
            out[method] = fallback[fidx]
            fidx += 1
    return out


def _prep_main_df(main_df: pd.DataFrame) -> pd.DataFrame:
    out = main_df.copy()
    out = out[np.isfinite(out["GapPct"])]
    out["Instance"] = pd.Categorical(out["Instance"], categories=INSTANCE_ORDER, ordered=True)
    out = out.sort_values(["Instance", "Method"], key=lambda s: s.map(_method_sort_key))
    return out


def plot_gap_across_instances(main_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    data = _prep_main_df(main_df)
    data = data.copy()
    data["Instance"] = data["Instance"].astype(str)

    families: list[tuple[str, list[str], str]] = [
        ("vqe_variants", ["vqe", "cvar_vqe"], "Gap to BKS Across Instances (VQE Variants)"),
        ("qaoa_variants", ["qaoa", "ma_qaoa", "ws_qaoa"], "Gap to BKS Across Instances (QAOA Variants)"),
        ("encoding_methods", ["pce", "qrao"], "Gap to BKS Across Instances (Encoding Methods)"),
    ]

    for file_key, family_methods, title in families:
        fam_df = data[data["Method"].isin(family_methods)].copy()
        if fam_df.empty:
            continue
        methods = sorted(fam_df["Method"].unique(), key=_method_sort_key)
        palette = _method_palette(methods)

        fig, ax = plt.subplots(figsize=(12.6, 5.0))
        sns.barplot(
            data=fam_df,
            x="Instance",
            y="GapPct",
            hue="Method",
            hue_order=methods,
            order=[inst for inst in INSTANCE_ORDER if inst in set(fam_df["Instance"])],
            palette=palette,
            edgecolor="black",
            linewidth=0.35,
            ax=ax,
        )
        # Add bar-level numeric labels for paper readability.
        for container in ax.containers:
            labels = []
            for bar in container:
                height = float(bar.get_height())
                labels.append(f"{height:.1f}" if np.isfinite(height) else "")
            ax.bar_label(container, labels=labels, fontsize=6.5, rotation=90, padding=1)

        # Add hatch patterns to make plot robust in grayscale print.
        hatches = ["", "//", "\\\\", "xx", "..", "++"]
        for idx, container in enumerate(ax.containers):
            hatch = hatches[idx % len(hatches)]
            for bar in container:
                bar.set_hatch(hatch)

        ax.set_title(title, fontsize=14, weight="bold")
        ax.set_xlabel("Instance")
        ax.set_ylabel("Gap to BKS (%)")
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.28, axis="y")
        method_mean = fam_df.groupby("Method")["GapPct"].mean().to_dict()
        method_best = fam_df.groupby("Method")["GapPct"].min().to_dict()
        handles, labels = ax.get_legend_handles_labels()
        legend_labels: list[str] = []
        for label in labels:
            mean_val = method_mean.get(label)
            best_val = method_best.get(label)
            if mean_val is not None and best_val is not None:
                legend_labels.append(
                    f"{_method_label(label)} (mean={mean_val:.1f}, best={best_val:.1f})"
                )
            else:
                legend_labels.append(_method_label(label))
        ax.legend(handles, legend_labels, title="Method", frameon=True, fontsize=9.0, title_fontsize=10.0)

        _save_figure(fig, out_dir / f"gap_to_bks_{file_key}", dpi=dpi)


def _build_quality_cost_df(main_df: pd.DataFrame, appendix_a_df: pd.DataFrame) -> pd.DataFrame:
    merged = appendix_a_df.merge(
        main_df[["Instance", "Method", "GapPct", "MethodLabel", "MethodFamily"]],
        on=["Instance", "Method", "MethodLabel", "MethodFamily"],
        how="left",
    )
    merged = merged[np.isfinite(merged["GapPct"])]
    return merged


def plot_quality_cost_scatter(main_df: pd.DataFrame, appendix_a_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    data = _build_quality_cost_df(main_df, appendix_a_df)
    methods = sorted(data["Method"].unique(), key=_method_sort_key)
    palette = _method_palette(methods)

    metrics = [
        ("TranspiledDepth", "Transpiled depth"),
        ("Transpiled2QGates", "Transpiled 2Q gate count"),
    ]
    for metric, xlabel in metrics:
        metric_df = data[np.isfinite(data[metric])].copy()
        if metric_df.empty:
            continue
        metric_df = metric_df[metric_df[metric] > 0]
        if metric_df.empty:
            continue
        metric_df["Backend"] = metric_df["Backend"].astype(str)
        g = sns.relplot(
            data=metric_df,
            x=metric,
            y="GapPct",
            col="Backend",
            col_wrap=3,
            hue="Method",
            style="Method",
            kind="scatter",
            s=80,
            alpha=0.9,
            edgecolor="black",
            linewidth=0.25,
            height=4.0,
            aspect=1.08,
            palette=palette,
        )
        g.set_axis_labels(xlabel, "Gap to BKS (%)")
        g.set_titles("{col_name}")
        g.fig.suptitle(
            f"Quality-Cost Scatter: {xlabel} vs Gap",
            fontsize=14,
            weight="bold",
            y=1.02,
        )
        if g._legend is not None:
            for text in g._legend.texts:
                text.set_text(_method_label(text.get_text()))
            g._legend.set_title("Method")
        for axis, backend in zip(g.axes.flat, g.col_names):
            axis.grid(alpha=0.25)
            axis.set_ylim(bottom=0.0)
            backend_values = metric_df.loc[metric_df["Backend"] == backend, metric]
            if not backend_values.empty:
                xmin = float(backend_values.min())
                xmax = float(backend_values.max())
                left = max(xmin * 0.92, 1e-6)
                right = xmax * 1.06 if xmax > xmin else xmax * 1.15 + 1.0
                axis.set_xlim(left=left, right=right)
        _save_figure(g.figure, out_dir / f"quality_cost_scatter_{metric.lower()}", dpi=dpi)


def plot_backend_latency_distribution(job_latency_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    if job_latency_df.empty:
        return

    data = job_latency_df.copy()
    data = data[np.isfinite(data["LatencySec"])]
    if data.empty:
        return

    fig, ax = plt.subplots(figsize=(12.2, 5.8))
    sns.boxplot(
        data=data,
        x="Backend",
        y="LatencySec",
        hue="MethodFamily",
        palette=METHOD_FAMILY_COLORS,
        width=0.7,
        fliersize=0.0,
        linewidth=1.0,
        ax=ax,
    )
    sample_n = min(len(data), 2400)
    if sample_n > 0:
        sampled = data.sample(n=sample_n, random_state=7, replace=False)
        sns.stripplot(
            data=sampled,
            x="Backend",
            y="LatencySec",
            hue="MethodFamily",
            palette=METHOD_FAMILY_COLORS,
            dodge=True,
            alpha=0.25,
            size=2.4,
            linewidth=0,
            ax=ax,
        )
    ax.set_title("Backend Robustness: Job Latency Distribution by Backend", fontsize=14, weight="bold")
    ax.set_xlabel("Backend")
    ax.set_ylabel("Job latency (sec)")
    ax.set_yscale("log")
    ax.grid(alpha=0.28, which="both")
    handles, labels = ax.get_legend_handles_labels()
    dedup: dict[str, Any] = {}
    for handle, label in zip(handles, labels):
        if label not in dedup:
            dedup[label] = handle
    ax.legend(
        dedup.values(),
        dedup.keys(),
        title="Method family",
        ncol=3,
        frameon=True,
        fontsize=9,
        title_fontsize=10,
    )
    _save_figure(fig, out_dir / "backend_latency_distribution", dpi=dpi)


def plot_compilation_overhead(appendix_a_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    data = appendix_a_df.copy()
    data = data[np.isfinite(data["LogicalDepth"]) & np.isfinite(data["TranspiledDepth"])]
    data = data[(data["LogicalDepth"] > 0) & (data["TranspiledDepth"] > 0)]
    data["DepthOverhead"] = data["TranspiledDepth"] / data["LogicalDepth"]
    data = data[np.isfinite(data["DepthOverhead"])]
    data = data[data["DepthOverhead"] > 0]
    if data.empty:
        return

    fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(14.6, 5.9))
    sns.scatterplot(
        data=data,
        x="LogicalDepth",
        y="TranspiledDepth",
        hue="Backend",
        style="MethodFamily",
        s=70,
        alpha=0.88,
        edgecolor="black",
        linewidth=0.25,
        ax=ax_left,
    )
    min_depth = float(min(data["LogicalDepth"].min(), data["TranspiledDepth"].min()))
    max_depth = float(max(data["LogicalDepth"].max(), data["TranspiledDepth"].max()))
    ax_left.plot([min_depth, max_depth], [min_depth, max_depth], linestyle="--", linewidth=1.1, color="black")
    ax_left.set_xscale("log")
    ax_left.set_yscale("log")
    ax_left.set_title("Logical vs Transpiled Depth")
    ax_left.set_xlabel("Logical depth")
    ax_left.set_ylabel("Transpiled depth")
    ax_left.grid(alpha=0.25, which="both")

    sns.boxplot(
        data=data,
        x="Backend",
        y="DepthOverhead",
        color="#cfe6ff",
        fliersize=0.0,
        linewidth=1.0,
        ax=ax_right,
    )
    sns.stripplot(
        data=data,
        x="Backend",
        y="DepthOverhead",
        hue="MethodFamily",
        palette=METHOD_FAMILY_COLORS,
        dodge=False,
        alpha=0.45,
        size=3.4,
        linewidth=0,
        ax=ax_right,
    )
    medians = data.groupby("Backend", as_index=False)["DepthOverhead"].median()
    for idx, row in medians.iterrows():
        ax_right.text(
            idx,
            float(row["DepthOverhead"]) * 1.03,
            f"median={row['DepthOverhead']:.2f}x",
            ha="center",
            va="bottom",
            fontsize=8,
        )
    ax_right.axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.55)
    ax_right.set_title("Overhead Factor by Backend")
    ax_right.set_xlabel("Backend")
    ax_right.set_ylabel("Transpiled / logical depth")
    ax_right.grid(alpha=0.28, axis="y")

    handles_left, labels_left = ax_left.get_legend_handles_labels()
    ax_left.legend(
        handles_left,
        labels_left,
        title="Backend / family",
        fontsize=8.5,
        title_fontsize=9,
        frameon=True,
        loc="upper left",
    )
    handles_right, labels_right = ax_right.get_legend_handles_labels()
    if handles_right:
        dedup: dict[str, Any] = {}
        for handle, label in zip(handles_right, labels_right):
            if label not in dedup:
                dedup[label] = handle
        ax_right.legend(
            dedup.values(),
            dedup.keys(),
            title="Method family",
            fontsize=8.5,
            title_fontsize=9,
            frameon=True,
            loc="upper right",
        )
    fig.suptitle("Compilation Overhead (Depth): What Hardware Mapping Adds", fontsize=14, weight="bold")
    fig.subplots_adjust(top=0.86, wspace=0.25)
    _save_figure(fig, out_dir / "compilation_overhead_depth_ratio", dpi=dpi)


def plot_encoding_tradeoff(main_df: pd.DataFrame, out_dir: Path, dpi: int) -> None:
    data = main_df.copy()
    data = data[data["Method"].isin(ENCODING_METHODS)]
    data = data[np.isfinite(data["GapPct"]) & np.isfinite(data["Qubits"])]
    if data.empty:
        return
    data["Instance"] = pd.Categorical(data["Instance"], categories=INSTANCE_ORDER, ordered=True)
    data = data.sort_values(["Instance", "Method"], key=lambda s: s.map(_method_sort_key))

    method_colors = {"pce": METHOD_COLORS["pce"], "qrao": METHOD_COLORS["qrao"]}

    fig, ax = plt.subplots(figsize=(10.2, 6.0))
    for instance, group in data.groupby("Instance", observed=True):
        if len(group) == 2:
            ax.plot(
                group["Qubits"],
                group["GapPct"],
                color="0.75",
                linewidth=1.05,
                alpha=0.7,
                zorder=1,
            )
    sns.scatterplot(
        data=data,
        x="Qubits",
        y="GapPct",
        hue="Method",
        style="Method",
        palette=method_colors,
        s=130,
        edgecolor="black",
        linewidth=0.4,
        ax=ax,
        zorder=3,
    )
    for _, row in data.iterrows():
        ax.annotate(
            str(row["Instance"]),
            (float(row["Qubits"]), float(row["GapPct"])),
            textcoords="offset points",
            xytext=(4, 3),
            fontsize=8,
            alpha=0.9,
        )

    ax.set_title("Encoding Tradeoff: Qubits vs Gap (PCE vs QRAO)", fontsize=14, weight="bold")
    ax.set_xlabel("Qubits used")
    ax.set_ylabel("Gap to BKS (%)")
    ax.grid(alpha=0.28)
    handles, labels = ax.get_legend_handles_labels()
    labels = [_method_label(lbl) for lbl in labels]
    ax.legend(handles, labels, title="Method", frameon=True, fontsize=10, title_fontsize=11)
    _save_figure(fig, out_dir / "encoding_tradeoff_qubits_vs_gap", dpi=dpi)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate MKP cross-method comparison figures from per-method CSV tables."
    )
    parser.add_argument(
        "--mkp-root",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/mkp"),
        help="Root directory containing mkp_<method>/csv folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/mkp/plots"),
        help="Output directory for plot files.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI for raster (PNG) export.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["axes.titlepad"] = 11
    plt.rcParams["legend.framealpha"] = 0.95

    main_df, appendix_a_df, appendix_b_df, job_latency_df = _load_method_tables(args.mkp_root)

    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    # Save harmonized plot-data snapshots for reproducibility.
    main_df.to_csv(out_dir / "plot_data_main.csv", index=False)
    appendix_a_df.to_csv(out_dir / "plot_data_appendix_a.csv", index=False)
    appendix_b_df.to_csv(out_dir / "plot_data_appendix_b.csv", index=False)
    if not job_latency_df.empty:
        job_latency_df.to_csv(out_dir / "plot_data_job_latency.csv", index=False)

    plot_gap_across_instances(main_df, out_dir, dpi=args.dpi)
    plot_quality_cost_scatter(main_df, appendix_a_df, out_dir, dpi=args.dpi)
    plot_backend_latency_distribution(job_latency_df, out_dir, dpi=args.dpi)
    plot_compilation_overhead(appendix_a_df, out_dir, dpi=args.dpi)
    plot_encoding_tradeoff(main_df, out_dir, dpi=args.dpi)

    print(f"Saved plots to: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
