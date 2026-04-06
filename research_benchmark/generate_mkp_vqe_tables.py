#!/usr/bin/env python
"""Generate hardware benchmark CSV tables and optional gap plots."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any

OPTIMAL_BY_PROBLEM: dict[str, dict[str, int | float]] = {
    "mkp": {
        "hp1": 3418,
        "hp2": 3186,
        "pb1": 3090,
        "pb2": 3186,
        "pb4": 95168,
        "pb5": 2139,
        "pet2": 87061,
        "pet3": 4015,
        "pet4": 6120,
        "pet5": 12400,
        "pet6": 10618,
        "pet7": 16537,
    },
    # Exact MIS BKS values for provided benchmark set (1dc/1et/1tc)
    "mis": {
        "1dc.128": 16,
        "1dc.64": 10,
        "1et.64": 18,
        "1tc.8": 4,
        "1tc.16": 8,
        "1tc.32": 12,
        "1tc.64": 20,
    },
    # QAPLIB best known / optimal solutions (all proven optimal)
    "qap": {
        "chr12a": 9552,
        "chr12b": 9742,
        "chr12c": 11156,
        "nug12": 578,
        "had12": 1652,
        "rou12": 235528,
        "scr12": 31410,
        "tai10a": 135028,
        "tai10b": 1183760,
        "tai12a": 224416,
        "tai12b": 39464925,
    },
}

OBJECTIVE_SENSE_BY_PROBLEM: dict[str, str] = {
    "mkp": "max",
    "mis": "max",
    "market_share": "min",
    "qap": "min",
}

INSTANCE_ORDER_BY_PROBLEM: dict[str, list[str]] = {
    "mkp": [
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
    ],
    "mis": [
        "1tc.8",
        "1tc.16",
        "1tc.32",
        "1tc.64",
        "1et.64",
        "1dc.64",
        "1dc.128",
    ],
    "market_share": [
        "ms_seed0_prod2",
        "ms_seed0_prod3",
        "ms_seed0_prod4",
        "ms_seed0_prod5",
        "ms_seed1_prod2",
        "ms_seed1_prod3",
        "ms_seed1_prod4",
        "ms_seed1_prod5",
    ],
    "qap": [
        "chr12a",
        "chr12b",
        "chr12c",
        "nug12",
        "had12",
        "rou12",
        "scr12",
        "tai10a",
        "tai10b",
        "tai12a",
        "tai12b",
    ],
}

METHOD_LABELS = {
    "pce": "PCE",
    "qrao": "QRAO",
    "vqe": "VQE",
    "cvar_vqe": "CVaR-VQE",
    "qaoa": "QAOA",
    "ma_qaoa": "MA-QAOA",
    "ws_qaoa": "WS-QAOA",
}

METHOD_COLORS = {
    "pce": "#0072B2",
    "qrao": "#D55E00",
    "vqe": "#009E73",
    "cvar_vqe": "#CC79A7",
    "qaoa": "#E69F00",
    "ma_qaoa": "#56B4E9",
    "ws_qaoa": "#7A7A7A",
}

FAMILY_METHODS: list[tuple[str, list[str], str]] = [
    ("vqe_variants", ["vqe", "cvar_vqe"], "Gap to BKS Across Instances (VQE Variants)"),
    ("qaoa_variants", ["qaoa", "ma_qaoa", "ws_qaoa"], "Gap to BKS Across Instances (QAOA Variants)"),
    ("encoding_methods", ["pce", "qrao"], "Gap to BKS Across Instances (Encoding Methods)"),
]


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip().lower()
        if not text:
            return None
        if "inf" in text:
            return float("inf")
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _safe_int(value: Any) -> int | None:
    fval = _safe_float(value)
    if fval is None:
        return None
    if fval == float("inf"):
        return None
    return int(round(fval))


def _percentile(values: list[float], pct: float) -> float | None:
    if not values:
        return None
    sorted_values = sorted(values)
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * pct / 100.0
    low = int(rank)
    high = min(low + 1, len(sorted_values) - 1)
    frac = rank - low
    return sorted_values[low] * (1.0 - frac) + sorted_values[high] * frac


def _fmt_num(value: float | int | None, digits: int = 2) -> str:
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def _fmt_scalar(value: float | int | None) -> str | int | float:
    if value is None:
        return ""
    if isinstance(value, int):
        return value
    if float(value).is_integer():
        return int(round(float(value)))
    return round(float(value), 6)


def _fmt_yes_no(value: Any) -> str:
    if value is True:
        return "Yes"
    if value is False:
        return "No"
    return ""


def _instance_from_result(data: dict[str, Any], path: Path) -> str:
    instance_name = str(data.get("instance_name", "")).strip()
    if instance_name:
        return Path(instance_name).stem
    return path.parent.name.replace("_dat", "")


def _instance_sort_key(instance: str, instance_order: list[str]) -> tuple[int, int | str]:
    if instance in instance_order:
        return (0, instance_order.index(instance))
    return (1, instance)


def _market_share_label(instance: str) -> str:
    parts = instance.split("_")
    if len(parts) >= 3 and parts[0] == "ms":
        seed_part = parts[1]
        prod_part = parts[2]
        if seed_part.startswith("seed") and prod_part.startswith("prod"):
            seed = seed_part[4:]
            prod = prod_part[4:]
            if seed.isdigit() and prod.isdigit():
                return f"ms{prod}{seed}"
    return instance


def _market_share_plot_sort_key(instance: str) -> tuple[int, int, str]:
    parts = instance.split("_")
    if len(parts) >= 3 and parts[0] == "ms":
        seed_part = parts[1]
        prod_part = parts[2]
        if seed_part.startswith("seed") and prod_part.startswith("prod"):
            seed = seed_part[4:]
            prod = prod_part[4:]
            if seed.isdigit() and prod.isdigit():
                return (int(prod), int(seed), instance)
    return (10**9, 10**9, instance)


def _compute_gap_pct(
    *,
    objective: float | None,
    optimal: float | int | None,
    feasible: Any,
    objective_sense: str,
) -> str:
    if feasible is False:
        return "inf"
    if objective is None or optimal is None:
        return ""

    optimal_value = float(optimal)
    if optimal_value == 0.0:
        if objective_sense == "min":
            return "0.00" if float(objective) == 0.0 else "inf"
        return "0.00" if float(objective) == 0.0 else ""

    if objective_sense == "min":
        return _fmt_num((float(objective) - optimal_value) / optimal_value * 100.0, 2)
    return _fmt_num((optimal_value - float(objective)) / optimal_value * 100.0, 2)


@lru_cache(maxsize=1)
def _load_market_share_optimal_by_instance() -> dict[str, int | float]:
    optimal_by_instance: dict[str, int | float] = {}
    repo_root = Path(__file__).resolve().parent.parent
    candidate_dirs = [
        repo_root / "Market_Share",
        repo_root / "Market_Share" / "market_share_classical_results",
    ]

    for base_dir in candidate_dirs:
        if not base_dir.exists():
            continue
        for solution_path in sorted(base_dir.glob("*_market_sharing_results/market_sharing_solution*.json")):
            folder_name = solution_path.parent.name
            parts = folder_name.split("_", 2)
            if len(parts) < 2:
                continue
            try:
                num_products = int(parts[0])
                seed = int(parts[1])
            except ValueError:
                continue

            data = json.loads(solution_path.read_text(encoding="utf-8"))
            value = _safe_float(data.get("objective_value"))
            if value is None:
                continue

            key = f"ms_seed{seed}_prod{num_products}"
            if key not in optimal_by_instance:
                optimal_by_instance[key] = int(round(value)) if float(value).is_integer() else round(value, 6)

    missing = [
        f"ms_seed{seed}_prod{num_products}"
        for seed in (0, 1)
        for num_products in (2, 3, 4, 5)
        if f"ms_seed{seed}_prod{num_products}" not in optimal_by_instance
    ]
    if missing:
        raise RuntimeError(
            "Missing market_share reference solutions for: " + ", ".join(missing)
        )

    return optimal_by_instance


def _optimal_by_problem(problem: str) -> dict[str, int | float]:
    if problem == "market_share":
        return _load_market_share_optimal_by_instance()
    return OPTIMAL_BY_PROBLEM.get(problem, {})


def _write_csv(path: Path, headers: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=headers)
        writer.writeheader()
        writer.writerows(rows)


def build_rows(
    result_paths: list[Path],
    *,
    problem: str,
    optimal_by_instance: dict[str, int | float],
    instance_order: list[str],
    objective_sense: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    main_rows: list[dict[str, Any]] = []
    appendix_a_rows: list[dict[str, Any]] = []
    appendix_b_rows: list[dict[str, Any]] = []

    for result_path in result_paths:
        with result_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)

        instance = _instance_from_result(data, result_path)
        optimal = optimal_by_instance.get(instance)
        best_result = data.get("best_result", {})
        circuit_metrics = data.get("circuit_metrics", {})
        optimizer = data.get("optimizer", {})
        execution = data.get("execution", {})
        timing = data.get("timing", {})
        benchmark_protocol = data.get("benchmark_protocol", {})
        job_metadata = data.get("job_metadata", [])
        if not isinstance(job_metadata, list):
            job_metadata = []
        typed_job_metadata = [m for m in job_metadata if isinstance(m, dict)]

        objective = _safe_float(best_result.get("objective_value"))
        if objective is None:
            objective = _safe_float(
                benchmark_protocol.get("objective_measured", {}).get("post_processed_objective")
            )
        feasible = best_result.get("feasible")
        qubits = _safe_int(circuit_metrics.get("num_qubits"))
        evals = _safe_int(optimizer.get("total_evaluations"))
        if evals is None:
            evals = _safe_int(benchmark_protocol.get("stopping_rule", {}).get("actual_iterations"))

        gap_pct = _compute_gap_pct(
            objective=objective,
            optimal=optimal,
            feasible=feasible,
            objective_sense=objective_sense,
        )

        main_row: dict[str, Any] = {
            "Instance": instance,
            "Optimal": _fmt_scalar(optimal),
            "Qubits": qubits if qubits is not None else "",
            "Gap%": gap_pct,
            "#Evals": evals if evals is not None else "",
        }
        if problem == "market_share":
            reconstructed = best_result.get("reconstructed_problem_objective", {})
            if not isinstance(reconstructed, dict):
                reconstructed = {}
            violations = reconstructed.get("constraint_violations", {})
            if not isinstance(violations, dict):
                violations = {}

            abs_dev = [
                dev
                for dev in (
                    _safe_float(value)
                    for value in violations.get("absolute_deviation_per_product", [])
                )
                if dev is not None and dev != float("inf")
            ]
            total_abs_dev = _safe_float(reconstructed.get("objective_value"))
            if total_abs_dev is None and abs_dev:
                total_abs_dev = sum(abs_dev)
            max_abs_dev = max(abs_dev) if abs_dev else None
            products_with_deviation = sum(1 for dev in abs_dev if abs(dev) > 1e-9) if abs_dev else None
            exact_target_match = abs_dev and all(abs(dev) <= 1e-9 for dev in abs_dev)

            main_row.update(
                {
                    "ExactTargetMatch": _fmt_yes_no(exact_target_match) if abs_dev else "",
                    "ProductsWithDeviation": products_with_deviation if products_with_deviation is not None else "",
                    "TotalAbsoluteDeviation": _fmt_scalar(total_abs_dev),
                    "MaxAbsoluteDeviation": _fmt_scalar(max_abs_dev),
                }
            )
        else:
            main_row["Feas"] = _fmt_yes_no(feasible)

        main_rows.append(main_row)

        logical_1q = _safe_int(circuit_metrics.get("one_qubit_gates_pretranspile"))
        logical_2q = _safe_int(circuit_metrics.get("two_qubit_gates_pretranspile"))
        logical_meas = _safe_int(circuit_metrics.get("measurement_count"))
        logical_total = None
        if logical_1q is not None or logical_2q is not None or logical_meas is not None:
            logical_total = (logical_1q or 0) + (logical_2q or 0) + (logical_meas or 0)

        jobs_by_backend: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for meta in typed_job_metadata:
            backend_name = str(meta.get("backend_name", "")).strip()
            if backend_name:
                jobs_by_backend[backend_name].append(meta)
        if not jobs_by_backend:
            jobs_by_backend[""] = []

        for backend_name in sorted(jobs_by_backend):
            backend_jobs = jobs_by_backend[backend_name]
            transpiled_depths: list[float] = []
            transpiled_2q_gates: list[float] = []
            transpiled_total_gates: list[float] = []
            backend_qubits: list[float] = []

            for meta in backend_jobs:
                depth = _safe_float(meta.get("transpiled_depth"))
                gate_1q = _safe_float(meta.get("transpiled_1q_gates"))
                gate_2q = _safe_float(meta.get("transpiled_2q_gates"))
                gate_meas = _safe_float(meta.get("transpiled_measurements"))
                bq = _safe_float(meta.get("backend_qubits"))
                if depth is not None:
                    transpiled_depths.append(depth)
                if gate_2q is not None:
                    transpiled_2q_gates.append(gate_2q)
                if gate_1q is not None or gate_2q is not None or gate_meas is not None:
                    transpiled_total_gates.append((gate_1q or 0.0) + (gate_2q or 0.0) + (gate_meas or 0.0))
                if bq is not None:
                    backend_qubits.append(bq)

            backend_qubits_value = (
                _safe_int(statistics.median(backend_qubits)) if backend_qubits else None
            )
            appendix_a_rows.append(
                {
                    "Instance": instance,
                    "Backend": backend_name,
                    "BackendQubits": backend_qubits_value if backend_qubits_value is not None else "",
                    "LogicalQubits": qubits if qubits is not None else "",
                    "LogicalDepth": _safe_int(circuit_metrics.get("depth_pretranspile")) or "",
                    "LogicalGateCount": logical_total if logical_total is not None else "",
                    "Logical2QGates": logical_2q if logical_2q is not None else "",
                    "Parameters": _safe_int(circuit_metrics.get("trainable_parameters")) or "",
                    "TranspiledDepth": _fmt_num(statistics.median(transpiled_depths), 2) if transpiled_depths else "",
                    "TranspiledTotalGates": _fmt_num(statistics.median(transpiled_total_gates), 2)
                    if transpiled_total_gates
                    else "",
                    "Transpiled2QGates": _fmt_num(statistics.median(transpiled_2q_gates), 2)
                    if transpiled_2q_gates
                    else "",
                }
            )

        shots_per_eval = _safe_int(benchmark_protocol.get("budget", {}).get("shots_per_circuit"))
        if shots_per_eval is None and typed_job_metadata:
            shots_per_eval = _safe_int(typed_job_metadata[0].get("shots"))
        if shots_per_eval is None:
            shots_per_eval = _safe_int(data.get("config", {}).get("shots"))

        jobs_total = _safe_int(execution.get("total_jobs_dispatched"))
        if jobs_total is None:
            job_ids = execution.get("job_ids", [])
            if isinstance(job_ids, list):
                jobs_total = len(job_ids)
        if jobs_total is None:
            jobs_total = evals

        jobs_ok = len(typed_job_metadata)
        jobs_failed = max((jobs_total or 0) - jobs_ok, 0)

        latencies = [
            latency
            for latency in (_safe_float(meta.get("elapsed_sec")) for meta in typed_job_metadata)
            if latency is not None and latency != float("inf")
        ]
        latency_median = statistics.median(latencies) if latencies else None
        latency_p95 = _percentile(latencies, 95.0)

        total_wall_clock_sec = _safe_float(timing.get("total_instance_sec"))
        if total_wall_clock_sec is None:
            total_wall_clock_sec = _safe_float(timing.get("solve_runtime_sec"))
        total_wall_clock_minutes = total_wall_clock_sec / 60.0 if total_wall_clock_sec is not None else None

        appendix_b_rows.append(
            {
                "Instance": instance,
                "ShotsPerEval": shots_per_eval if shots_per_eval is not None else "",
                "#Evals": evals if evals is not None else "",
                "#Jobs(ok/failed)": f"{jobs_ok}/{jobs_failed}",
                "MedianJobLatencySec": _fmt_num(latency_median, 3),
                "P95JobLatencySec": _fmt_num(latency_p95, 3),
                "TotalWallClockMinutes": _fmt_num(total_wall_clock_minutes, 3),
            }
        )

    main_rows.sort(key=lambda row: _instance_sort_key(str(row["Instance"]), instance_order))
    appendix_a_rows.sort(
        key=lambda row: (_instance_sort_key(str(row["Instance"]), instance_order), str(row["Backend"]))
    )
    appendix_b_rows.sort(key=lambda row: _instance_sort_key(str(row["Instance"]), instance_order))
    return main_rows, appendix_a_rows, appendix_b_rows


def _infer_problem(input_dir: Path, requested_problem: str) -> str:
    if requested_problem in {"mkp", "mis", "market_share", "qap"}:
        return requested_problem
    name = input_dir.name.lower()
    if "market_share" in name:
        return "market_share"
    if "qap" in name:
        return "qap"
    if "mis" in name:
        return "mis"
    if "mkp" in name:
        return "mkp"
    for child in input_dir.iterdir() if input_dir.exists() else []:
        cname = child.name.lower()
        if cname.startswith("market_share_"):
            return "market_share"
        if cname.startswith("qap_"):
            return "qap"
        if cname.startswith("mis_"):
            return "mis"
        if cname.startswith("mkp_"):
            return "mkp"
    return "mkp"


def _method_from_dirname(dirname: str, problem: str) -> str:
    prefix = f"{problem}_"
    if dirname.startswith(prefix):
        return dirname[len(prefix) :]
    return dirname


def _write_method_tables(
    *,
    output_dir: Path,
    problem: str,
    method: str,
    main_rows: list[dict[str, Any]],
    appendix_a_rows: list[dict[str, Any]],
    appendix_b_rows: list[dict[str, Any]],
    use_suffix: bool,
) -> None:
    suffix = f"_{method}" if use_suffix and method else ""
    main_name = f"main_table{suffix}.csv"
    a_name = f"appendix_table_a{suffix}_circuit_compilation.csv"
    b_name = f"appendix_table_b{suffix}_execution_robustness.csv"
    if problem == "market_share":
        main_headers = [
            "Instance",
            "Optimal",
            "Qubits",
            "ExactTargetMatch",
            "ProductsWithDeviation",
            "TotalAbsoluteDeviation",
            "MaxAbsoluteDeviation",
            "Gap%",
            "#Evals",
        ]
    else:
        main_headers = ["Instance", "Optimal", "Qubits", "Feas", "Gap%", "#Evals"]

    _write_csv(
        output_dir / main_name,
        main_headers,
        main_rows,
    )
    _write_csv(
        output_dir / a_name,
        [
            "Instance",
            "Backend",
            "BackendQubits",
            "LogicalQubits",
            "LogicalDepth",
            "LogicalGateCount",
            "Logical2QGates",
            "Parameters",
            "TranspiledDepth",
            "TranspiledTotalGates",
            "Transpiled2QGates",
        ],
        appendix_a_rows,
    )
    _write_csv(
        output_dir / b_name,
        [
            "Instance",
            "ShotsPerEval",
            "#Evals",
            "#Jobs(ok/failed)",
            "MedianJobLatencySec",
            "P95JobLatencySec",
            "TotalWallClockMinutes",
        ],
        appendix_b_rows,
    )

    print(f"Wrote {len(main_rows)} rows to {output_dir / main_name}")
    print(f"Wrote {len(appendix_a_rows)} rows to {output_dir / a_name}")
    print(f"Wrote {len(appendix_b_rows)} rows to {output_dir / b_name}")


def _method_palette(methods: list[str]) -> dict[str, str]:
    return {method: METHOD_COLORS.get(method, "#4C4C4C") for method in methods}


def _write_gap_family_plots(
    *,
    records: list[dict[str, Any]],
    problem: str,
    plots_dir: Path,
    instance_order: list[str],
    dpi: int,
) -> None:
    if not records:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
    except Exception as exc:  # pragma: no cover
        print(f"Skipping plot generation: plotting dependencies unavailable ({exc})")
        return

    df = pd.DataFrame(records)
    if df.empty:
        return
    df["GapValue"] = df["Gap%"].apply(_safe_float)
    plots_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(plots_dir / f"{problem}_plot_data_main.csv", index=False)

    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams["legend.framealpha"] = 0.95

    if problem == "market_share":
        title_prefix = "Relative Objective Gap Across Instances"
        y_label = "Relative objective gap (%)"
        file_prefix = "relative_objective_gap"
    else:
        title_prefix = "Gap to BKS Across Instances"
        y_label = "Gap to BKS (%)"
        file_prefix = "gap_to_bks"

    for file_key, methods, title in FAMILY_METHODS:
        fam_df = df[df["Method"].isin(methods)].copy()
        fam_df = fam_df[np.isfinite(fam_df["GapValue"])]
        if fam_df.empty:
            continue
        methods_present = [m for m in methods if m in set(fam_df["Method"])]
        if not methods_present:
            continue
        palette = _method_palette(methods_present)
        raw_instances = sorted(set(fam_df["Instance"]))
        if problem == "market_share":
            inst_order_raw = sorted(raw_instances, key=_market_share_plot_sort_key)
            fam_df["PlotInstance"] = fam_df["Instance"].map(_market_share_label)
            x_col = "PlotInstance"
            inst_order = [_market_share_label(inst) for inst in inst_order_raw]
        else:
            inst_order = [inst for inst in instance_order if inst in set(fam_df["Instance"])]
            if not inst_order:
                inst_order = raw_instances
            x_col = "Instance"

        fig, ax = plt.subplots(figsize=(12.6, 5.0))
        sns.barplot(
            data=fam_df,
            x=x_col,
            y="GapValue",
            hue="Method",
            hue_order=methods_present,
            order=inst_order,
            palette=palette,
            dodge=True,
            edgecolor="black",
            linewidth=0.35,
            ax=ax,
        )
        hatches = ["", "//", "\\\\", "xx", "..", "++"]
        for idx, container in enumerate(ax.containers):
            hatch = hatches[idx % len(hatches)]
            for bar in container:
                bar.set_hatch(hatch)
            labels = [
                f"{float(bar.get_height()):.1f}" if float(bar.get_height()) > 1e-9 else ""
                for bar in container
            ]
            ax.bar_label(container, labels=labels, fontsize=6.5, rotation=90, padding=1)

        title_suffix = title.split("(", 1)[1] if "(" in title else title
        ax.set_title(f"{title_prefix} ({title_suffix}", fontsize=14, weight="bold")
        ax.set_xlabel("Instance")
        ax.set_ylabel(y_label)
        ax.set_ylim(bottom=0.0)
        ax.grid(alpha=0.28, axis="y")
        handles, labels = ax.get_legend_handles_labels()
        pretty_labels = [METHOD_LABELS.get(label, label.upper()) for label in labels]
        ax.legend(handles, pretty_labels, title="Method", frameon=True, fontsize=9.0, title_fontsize=10.0)

        stem = plots_dir / f"{file_prefix}_{file_key}"
        fig.savefig(stem.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
        fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
        plt.close(fig)
        print(f"Wrote plot: {stem.with_suffix('.png')}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate hardware benchmark main and appendix CSV tables from result.json files."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/mkp/mkp_vqe"),
        help="Method directory (<problem>_<method>) or problem root directory containing multiple method folders.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory. In multi-method mode, per-method CSVs go under <output-dir>/<method-dir>/csv.",
    )
    parser.add_argument(
        "--problem",
        choices=["mkp", "mis", "market_share", "qap", "auto"],
        default="auto",
        help="Problem type. Use auto to infer from input-dir name.",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate family-level gap bar plots when multiple methods are processed.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=None,
        help="Directory for generated plots. Default: <problem-root>/plots.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=400,
        help="DPI for PNG plot export.",
    )
    parser.add_argument(
        "--suffixed-names",
        action="store_true",
        help="Use suffixed CSV names in single-method mode (e.g., main_table_vqe.csv).",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Optional method filter (e.g., --methods pce qrao). Applies in multi-method mode.",
    )
    return parser.parse_args()


def _process_single_method(
    *,
    method_dir: Path,
    output_dir: Path,
    problem: str,
    instance_order: list[str],
    optimal_by_instance: dict[str, int | float],
    objective_sense: str,
    use_suffix: bool,
) -> tuple[str, list[dict[str, Any]]]:
    method = _method_from_dirname(method_dir.name, problem)
    result_paths = sorted(method_dir.glob("*/result.json"))
    if not result_paths:
        raise SystemExit(f"No result.json files found under: {method_dir}")
    main_rows, appendix_a_rows, appendix_b_rows = build_rows(
        result_paths,
        problem=problem,
        optimal_by_instance=optimal_by_instance,
        instance_order=instance_order,
        objective_sense=objective_sense,
    )
    _write_method_tables(
        output_dir=output_dir,
        problem=problem,
        method=method,
        main_rows=main_rows,
        appendix_a_rows=appendix_a_rows,
        appendix_b_rows=appendix_b_rows,
        use_suffix=use_suffix,
    )
    return method, main_rows


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir.resolve()
    if not input_dir.exists():
        raise SystemExit(f"Input directory not found: {input_dir}")

    problem = _infer_problem(input_dir, args.problem)
    optimal_by_instance = _optimal_by_problem(problem)
    instance_order = INSTANCE_ORDER_BY_PROBLEM.get(problem, [])
    objective_sense = OBJECTIVE_SENSE_BY_PROBLEM.get(problem, "max")

    requested_methods = (
        {m.strip().lower() for m in (args.methods or []) if m.strip()}
        if args.methods is not None
        else None
    )

    method_dirs = []
    for child in sorted(input_dir.iterdir()):
        if not child.is_dir():
            continue
        if not child.name.startswith(f"{problem}_"):
            continue
        method_name = _method_from_dirname(child.name, problem).lower()
        if requested_methods is not None and method_name not in requested_methods:
            continue
        if any(child.glob("*/result.json")):
            method_dirs.append(child)

    if method_dirs:
        all_main_records: list[dict[str, Any]] = []
        for method_dir in method_dirs:
            if args.output_dir is None:
                method_output_dir = method_dir / "csv"
            else:
                method_output_dir = args.output_dir / method_dir.name / "csv"
            method, main_rows = _process_single_method(
                method_dir=method_dir,
                output_dir=method_output_dir,
                problem=problem,
                instance_order=instance_order,
                optimal_by_instance=optimal_by_instance,
                objective_sense=objective_sense,
                use_suffix=True,
            )
            for row in main_rows:
                rec = dict(row)
                rec["Method"] = method
                all_main_records.append(rec)

        if args.generate_plots:
            plots_dir = args.plots_dir or (input_dir / "plots")
            _write_gap_family_plots(
                records=all_main_records,
                problem=problem,
                plots_dir=plots_dir,
                instance_order=instance_order,
                dpi=int(args.dpi),
            )
        return 0

    # Single method directory mode
    if args.output_dir is None:
        single_output_dir = input_dir / "csv"
    else:
        single_output_dir = args.output_dir
    _process_single_method(
        method_dir=input_dir,
        output_dir=single_output_dir,
        problem=problem,
        instance_order=instance_order,
        optimal_by_instance=optimal_by_instance,
        objective_sense=objective_sense,
        use_suffix=bool(args.suffixed_names),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
