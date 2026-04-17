#!/usr/bin/env python3
"""Generate LaTeX paper tables from classical solver result files.

The script reads each problem's final summary file when available. If a solver
is still running and summary.csv/summary.json has not been written yet, it falls
back to the per-instance solutions/*.json files that are written after each
completed instance.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Any


PROBLEMS = ("mdkp", "mis", "qap")

MIS_BKS: dict[str, int] = {
    "1dc.64": 10,
    "1dc.128": 16,
    "1dc.256": 30,
    "1dc.512": 52,
    "1dc.1024": 94,
    "1tc.8": 4,
    "1tc.16": 8,
    "1tc.32": 12,
    "1tc.64": 20,
    "1et.64": 18,
}

# QAPLIB best-known/optimal values for the QAP instances in this repository.
QAP_BKS: dict[str, int] = {
    "chr12a": 9552,
    "chr12b": 9742,
    "chr12c": 11156,
    "had12": 1652,
    "nug12": 578,
    "rou12": 235528,
    "scr12": 31410,
    "tai10a": 135028,
    "tai10b": 1183760,
    "tai12a": 224416,
    "tai12b": 39464925,
}

STATUS_SHORT: dict[str, str] = {
    "OPTIMAL": "OPT",
    "TIME_LIMIT": "TL",
    "ITERATION_LIMIT": "ITER",
    "NODE_LIMIT": "NODE",
    "SOLUTION_LIMIT": "SOL",
    "SUBOPTIMAL": "SUBOPT",
    "INFEASIBLE": "INFEAS",
    "INF_OR_UNBD": "INF/UNBD",
    "UNBOUNDED": "UNBD",
    "INTERRUPTED": "INT",
    "NUMERIC": "NUM",
    "WORK_LIMIT": "WORK",
    "MEM_LIMIT": "MEM",
}


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def load_records(problem_dir: Path) -> list[dict[str, Any]]:
    summary_json = problem_dir / "summary.json"
    summary_csv = problem_dir / "summary.csv"

    if summary_json.exists():
        data = read_json(summary_json)
        if isinstance(data, list):
            return [dict(row) for row in data]

    if summary_csv.exists():
        with summary_csv.open("r", newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]

    records: list[dict[str, Any]] = []
    for path in sorted((problem_dir / "solutions").glob("*_solution.json")):
        try:
            data = read_json(path)
        except json.JSONDecodeError as exc:
            print(f"Skipping incomplete JSON while solver is running: {path} ({exc})", file=sys.stderr)
            continue
        if isinstance(data, dict):
            records.append(data)
    return records


def to_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip()
        if text.lower() in {"none", "null", "nan", "--"}:
            return None
        try:
            number = float(text)
        except ValueError:
            return None
    if not math.isfinite(number):
        return None
    return number


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(round(number))


def strip_known_instance_suffix(name: str) -> str:
    for suffix in (".txt.gz", ".txt", ".dat"):
        if name.endswith(suffix):
            return name[: -len(suffix)]
    return name


def stem_name(record: dict[str, Any]) -> str:
    instance_id = str(record.get("instance_id") or "")
    if instance_id:
        return strip_known_instance_suffix(instance_id.split("__")[-1])
    return strip_known_instance_suffix(Path(str(record.get("instance") or "")).name)


def full_instance_name(record: dict[str, Any]) -> str:
    instance_id = str(record.get("instance_id") or "")
    if instance_id:
        return instance_id.replace("__", "/")
    return stem_name(record)


def display_names(records: list[dict[str, Any]]) -> dict[int, str]:
    stems = [stem_name(row) for row in records]
    duplicate_stems = {stem for stem in stems if stems.count(stem) > 1}
    names: dict[int, str] = {}
    for idx, row in enumerate(records):
        name = full_instance_name(row) if stem_name(row) in duplicate_stems else stem_name(row)
        names[idx] = latex_escape(name)
    return names


def natural_key(text: str) -> list[Any]:
    parts = re.split(r"(\d+)", text)
    return [int(part) if part.isdigit() else part.lower() for part in parts]


def sort_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(records, key=lambda row: natural_key(str(row.get("instance_id") or row.get("instance") or "")))


def latex_escape(text: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in text)


def format_int(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "--"
    if abs(number - round(number)) < 1e-6:
        return str(int(round(number)))
    return format_float(number, digits=2)


def format_float(value: Any, digits: int = 2) -> str:
    number = to_float(value)
    if number is None:
        return "--"
    text = f"{number:.{digits}f}"
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text if text else "0"


def format_gap_percent(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "--"
    percent = number * 100.0
    if abs(percent) < 0.05:
        return "0.0"
    if abs(percent) >= 10:
        return f"{percent:.1f}"
    return f"{percent:.2f}".rstrip("0").rstrip(".")


def format_time(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return "--"
    return f"{number:.2f}"


def short_status(value: Any, use_long_status: bool) -> str:
    text = str(value or "--")
    if use_long_status:
        return latex_escape(text)
    return latex_escape(STATUS_SHORT.get(text, text))


def bks_from_optimal(record: dict[str, Any]) -> int | None:
    status = str(record.get("status") or "")
    gap = to_float(record.get("relative_gap"))
    if status == "OPTIMAL" or gap == 0.0:
        return to_int(record.get("objective_value"))
    return None


def bks_mdkp(record: dict[str, Any]) -> str:
    value = to_int(record.get("known_optimal_value"))
    if value is None:
        value = bks_from_optimal(record)
    return format_int(value)


def bks_mis(record: dict[str, Any]) -> str:
    key = stem_name(record)
    value = MIS_BKS.get(key)
    if value is None:
        value = bks_from_optimal(record)
    return format_int(value)


def bks_qap(record: dict[str, Any]) -> str:
    key = stem_name(record)
    value = QAP_BKS.get(key)
    if value is None:
        value = bks_from_optimal(record)
    return format_int(value)


def mdkp_rows(records: list[dict[str, Any]], use_long_status: bool) -> list[str]:
    names = display_names(records)
    rows = []
    for idx, row in enumerate(records):
        size = rf"${format_int(row.get('num_items'))}\times {format_int(row.get('num_dimensions'))}$"
        rows.append(
            f"{names[idx]} & {size} & {bks_mdkp(row)} & "
            f"{format_int(row.get('objective_value'))} & {format_int(row.get('best_bound'))} & "
            f"{format_gap_percent(row.get('relative_gap'))} & {short_status(row.get('status'), use_long_status)} & "
            f"{format_time(row.get('solver_runtime_sec') or row.get('solve_wall_time_sec'))} \\\\"
        )
    return rows


def mis_rows(records: list[dict[str, Any]], use_long_status: bool) -> list[str]:
    names = display_names(records)
    rows = []
    for idx, row in enumerate(records):
        rows.append(
            f"{names[idx]} & {format_int(row.get('num_nodes'))} & {format_int(row.get('num_edges'))} & "
            f"{bks_mis(row)} & {format_int(row.get('objective_value'))} & {format_int(row.get('best_bound'))} & "
            f"{format_gap_percent(row.get('relative_gap'))} & {short_status(row.get('status'), use_long_status)} & "
            f"{format_time(row.get('solver_runtime_sec') or row.get('solve_wall_time_sec'))} \\\\"
        )
    return rows


def qap_rows(records: list[dict[str, Any]], use_long_status: bool) -> list[str]:
    names = display_names(records)
    rows = []
    for idx, row in enumerate(records):
        rows.append(
            f"{names[idx]} & {format_int(row.get('num_facilities'))} & {bks_qap(row)} & "
            f"{format_int(row.get('objective_value'))} & {format_int(row.get('best_bound'))} & "
            f"{format_gap_percent(row.get('relative_gap'))} & {short_status(row.get('status'), use_long_status)} & "
            f"{format_time(row.get('solver_runtime_sec') or row.get('solve_wall_time_sec'))} \\\\"
        )
    return rows


def table_block(problem: str, records: list[dict[str, Any]], use_long_status: bool) -> str:
    if problem == "mdkp":
        rows = mdkp_rows(records, use_long_status)
        lines = [
            r"\begin{table}[t]",
            r"\caption{\label{tab:mdkp_classical}",
            r"Classical Gurobi performance on MDKP instances.",
            r"Size is reported as items $\times$ dimensions.",
            r"BKS denotes the known optimal value from the benchmark file.",
            r"Gap is the final relative MIP gap at termination.}",
            r"\begin{indented}",
            r"\item[]\begin{tabular}{lccccccc}",
            r"\br",
            r"Instance & Size & BKS & Incumbent & Bound & Gap (\%) & Status & Time (s)\\",
            r"\mr",
            *rows,
            r"\br",
            r"\end{tabular}",
            r"\end{indented}",
            r"\end{table}",
        ]
        return "\n".join(lines) + "\n"

    if problem == "mis":
        rows = mis_rows(records, use_long_status)
        lines = [
            r"\begin{table}[t]",
            r"\caption{\label{tab:mis_classical}",
            r"Classical Gurobi performance on MIS instances.",
            r"$|V|$ and $|E|$ denote the number of vertices and edges.",
            r"BKS is the known optimal independent-set size.}",
            r"\begin{indented}",
            r"\item[]\begin{tabular}{lcccccccc}",
            r"\br",
            r"Instance & $|V|$ & $|E|$ & BKS & Incumbent & Bound & Gap (\%) & Status & Time (s)\\",
            r"\mr",
            *rows,
            r"\br",
            r"\end{tabular}",
            r"\end{indented}",
            r"\end{table}",
        ]
        return "\n".join(lines) + "\n"

    if problem == "qap":
        rows = qap_rows(records, use_long_status)
        lines = [
            r"\begin{table}[t]",
            r"\caption{\label{tab:qap_classical}",
            r"Classical Gurobi performance on QAP instances.",
            r"$n$ is the number of facilities/locations.",
            r"BKS denotes the known optimal objective value.}",
            r"\begin{indented}",
            r"\item[]\begin{tabular}{lccccccc}",
            r"\br",
            r"Instance & $n$ & BKS & Incumbent & Bound & Gap (\%) & Status & Time (s)\\",
            r"\mr",
            *rows,
            r"\br",
            r"\end{tabular}",
            r"\end{indented}",
            r"\end{table}",
        ]
        return "\n".join(lines) + "\n"

    raise ValueError(f"Unknown problem: {problem}")


def write_tables(
    results_root: Path,
    output_dir: Path,
    selected_problems: list[str],
    use_long_status: bool,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    combined_blocks: list[str] = []

    for problem in selected_problems:
        problem_dir = results_root / problem
        records = sort_records(load_records(problem_dir))
        if not records:
            print(f"No records found for {problem} under {problem_dir}; skipping.", file=sys.stderr)
            continue

        content = table_block(problem, records, use_long_status)
        path = output_dir / f"{problem}_classical_table.tex"
        path.write_text(content, encoding="utf-8")
        written.append(path)
        combined_blocks.append(content)

    if combined_blocks:
        combined_path = output_dir / "all_classical_tables.tex"
        combined_path.write_text("\n".join(combined_blocks), encoding="utf-8")
        written.append(combined_path)

    return written


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables from classical_solutions result files."
    )
    parser.add_argument(
        "--results-root",
        type=Path,
        default=root / "classical_solutions" / "results",
        help="Root containing mdkp/, mis/, and qap/ result directories.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root / "classical_solutions" / "tables",
        help="Directory where .tex table files are written.",
    )
    parser.add_argument(
        "--problem",
        choices=[*PROBLEMS, "all"],
        default="all",
        help="Generate one problem table or all problem tables.",
    )
    parser.add_argument(
        "--long-status",
        action="store_true",
        help="Use full Gurobi statuses instead of compact table labels such as OPT and TL.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    selected_problems = list(PROBLEMS) if args.problem == "all" else [args.problem]
    written = write_tables(
        results_root=args.results_root.resolve(),
        output_dir=args.output_dir.resolve(),
        selected_problems=selected_problems,
        use_long_status=args.long_status,
    )
    if not written:
        return 1
    for path in written:
        print(f"Wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
