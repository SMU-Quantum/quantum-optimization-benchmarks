#!/usr/bin/env python
"""Audit market-share hardware results for exact target satisfaction and deviation size."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def _safe_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        try:
            return float(text)
        except ValueError:
            return None
    return None


def _instance_from_result(data: dict[str, Any], result_path: Path) -> str:
    instance_name = str(data.get("instance_name", "")).strip()
    if instance_name:
        return Path(instance_name).stem
    return result_path.parent.name


def _method_from_dirname(dirname: str) -> str:
    prefix = "market_share_"
    return dirname[len(prefix):] if dirname.startswith(prefix) else dirname


def build_rows(results_root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for method_dir in sorted(results_root.iterdir()):
        if not method_dir.is_dir() or not method_dir.name.startswith("market_share_"):
            continue
        method = _method_from_dirname(method_dir.name)
        for result_path in sorted(method_dir.glob("*/result.json")):
            data = json.loads(result_path.read_text(encoding="utf-8"))
            best_result = data.get("best_result", {})
            reconstructed = best_result.get("reconstructed_problem_objective", {})
            violations = reconstructed.get("constraint_violations", {})

            abs_dev = [_safe_float(v) or 0.0 for v in violations.get("absolute_deviation_per_product", [])]
            realized = [_safe_float(v) or 0.0 for v in violations.get("realized_demands", [])]
            targets = [_safe_float(v) or 0.0 for v in violations.get("target_demands", [])]

            rows.append(
                {
                    "Method": method,
                    "Instance": _instance_from_result(data, result_path),
                    "StoredFeasible": best_result.get("feasible"),
                    "ReconstructedFeasible": reconstructed.get("feasible"),
                    "ExactTargetMatch": all(v == 0.0 for v in abs_dev),
                    "ProductsWithDeviation": sum(1 for v in abs_dev if v != 0.0),
                    "TotalAbsoluteDeviation": sum(abs_dev),
                    "MaxAbsoluteDeviation": max(abs_dev) if abs_dev else 0.0,
                    "DeviationVector": json.dumps(abs_dev),
                    "RealizedDemands": json.dumps(realized),
                    "TargetDemands": json.dumps(targets),
                    "ResultPath": str(result_path),
                }
            )

    rows.sort(key=lambda row: (str(row["Instance"]), str(row["Method"])))
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit market-share result feasibility/deviation.")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/market_share"),
        help="Root market_share results directory.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("research_benchmark/research_benchmark/results_hardware/market_share/market_share_solution_audit.csv"),
        help="Output CSV path.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    rows = build_rows(args.input_dir.resolve())
    if not rows:
        raise SystemExit(f"No market_share result.json files found under: {args.input_dir}")

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "Method",
                "Instance",
                "StoredFeasible",
                "ReconstructedFeasible",
                "ExactTargetMatch",
                "ProductsWithDeviation",
                "TotalAbsoluteDeviation",
                "MaxAbsoluteDeviation",
                "DeviationVector",
                "RealizedDemands",
                "TargetDemands",
                "ResultPath",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {len(rows)} rows to {args.output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
