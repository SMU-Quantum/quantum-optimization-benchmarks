#!/usr/bin/env python3
"""Build metadata-only provenance for Runtime mitigation and final sampling.

This audit intentionally avoids reconstructing unarchived IBM Runtime internals.
It records what can be recovered from saved result/trace files plus the protocol
setting that Estimator jobs requested IBM Runtime resilience level 2.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULTS_ROOT = ROOT / "research_benchmark" / "research_benchmark" / "results_hardware"
OUT = Path(__file__).resolve().parent

COLUMNS = [
    "problem",
    "instance",
    "method",
    "backend",
    "estimator_objective_evaluations",
    "estimator_requested_shots",
    "estimator_resilience_level",
    "estimator_custom_mitigation_options",
    "sampler_final_sampling_shots",
    "sampler_custom_noise_management_options",
    "final_bitstrings_from_raw_default_sampler",
    "runtime_usage_seconds_if_available",
    "wall_clock_seconds_if_available",
    "paired_level0_overhead_measured",
    "overhead_reporting_note",
]

OVERHEAD_NOTE = (
    "Runtime-managed Estimator mitigation was requested through resilience_level=2. "
    "No paired resilience_level=0 execution was performed, so a circuit-specific "
    "mitigation-overhead multiplier is not estimated retrospectively."
)

PROBLEM_LABELS = {
    "mkp": "MDKP",
    "mis": "MIS",
    "qap": "QAP",
    "market_share": "MSP",
}


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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
                rows.append({"_unparsed_line": line})
    return rows


def fmt(value: Any) -> str:
    if value is None:
        return "not_available"
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return f"{value:.6g}"
    if isinstance(value, (list, tuple, set)):
        values = [fmt(v) for v in value if fmt(v) != "not_available"]
        return ";".join(values) if values else "not_available"
    text = str(value).strip()
    return text if text else "not_available"


def clean_instance_name(name: str) -> str:
    text = str(name).replace("\\", "/").split("/")[-1]
    for suffix in (".txt", ".dat", ".gen"):
        if text.endswith(suffix):
            return text[: -len(suffix)]
    return text


def trace_metadata_rows(trace_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    metadata: list[dict[str, Any]] = []
    for row in trace_rows:
        meta = row.get("metadata")
        if isinstance(meta, dict):
            metadata.append(meta)
    return metadata


def unique_backends(result: dict[str, Any], trace_rows: list[dict[str, Any]]) -> str:
    backends: set[str] = set()
    for row in trace_rows:
        backend = row.get("backend_name")
        if backend:
            backends.add(str(backend))
        meta = row.get("metadata")
        if isinstance(meta, dict) and meta.get("backend_name"):
            backends.add(str(meta["backend_name"]))
    for key in ("device_calibration", "backend_calibration"):
        calibration = result.get(key)
        if isinstance(calibration, dict):
            for value in calibration.values():
                if isinstance(value, dict) and value.get("backend_name"):
                    backends.add(str(value["backend_name"]))
    return ";".join(sorted(backends)) if backends else "not_available"


def objective_evaluations(result: dict[str, Any], trace_rows: list[dict[str, Any]]) -> str:
    eval_rows = [
        row
        for row in trace_rows
        if "evaluation" in row
        and (
            "objective_value" in row
            or "best_sample_energy" in row
            or row.get("qrao_job_phase") == "estimator"
        )
    ]
    if eval_rows:
        return str(len(eval_rows))
    protocol = result.get("benchmark_protocol", {})
    stopping = protocol.get("stopping_rule", {}) if isinstance(protocol, dict) else {}
    budget = protocol.get("budget", {}) if isinstance(protocol, dict) else {}
    return fmt(stopping.get("actual_iterations") or budget.get("total_circuit_evaluations"))


def requested_shots(result: dict[str, Any], trace_rows: list[dict[str, Any]]) -> str:
    shots = [
        meta.get("shots")
        for meta in trace_metadata_rows(trace_rows)
        if meta.get("shots") is not None
    ]
    if shots:
        counts = Counter(shots)
        return fmt(counts.most_common(1)[0][0])
    protocol = result.get("benchmark_protocol", {})
    budget = protocol.get("budget", {}) if isinstance(protocol, dict) else {}
    return fmt(budget.get("shots_per_circuit"))


def final_sampler_shots(result: dict[str, Any], trace_rows: list[dict[str, Any]]) -> str:
    sampler_phase_shots = [
        row.get("metadata", {}).get("shots")
        for row in trace_rows
        if row.get("qrao_job_phase") == "sampler" and isinstance(row.get("metadata"), dict)
    ]
    if sampler_phase_shots:
        return fmt(sampler_phase_shots[-1])
    return requested_shots(result, trace_rows)


def wall_clock_seconds(result: dict[str, Any], trace_rows: list[dict[str, Any]]) -> str:
    timing = result.get("timing")
    if isinstance(timing, dict):
        for key in ("total_instance_sec", "solve_runtime_sec", "wall_clock_seconds"):
            if timing.get(key) is not None:
                return fmt(timing[key])
    elapsed = [
        float(meta["elapsed_sec"])
        for meta in trace_metadata_rows(trace_rows)
        if meta.get("elapsed_sec") is not None
    ]
    return fmt(sum(elapsed) if elapsed else None)


def runtime_usage_seconds(result: dict[str, Any], trace_rows: list[dict[str, Any]]) -> str:
    for container in [result, *trace_metadata_rows(trace_rows)]:
        for key in ("runtime_usage_seconds", "usage_seconds", "usage_estimation_seconds"):
            if isinstance(container, dict) and container.get(key) is not None:
                return fmt(container[key])
    return "not_available_in_archived_metadata"


def build_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for result_path in sorted(RESULTS_ROOT.rglob("result.json")):
        result = load_json(result_path)
        trace_rows = load_trace(result_path.with_name("trace.jsonl"))
        problem_raw = str(result.get("problem", result_path.parents[2].name))
        execution = result.get("execution", {})
        instance = clean_instance_name(str(result.get("instance_name", result_path.parent.name)))
        rows.append(
            {
                "problem": PROBLEM_LABELS.get(problem_raw, problem_raw),
                "instance": instance,
                "method": fmt(execution.get("method")),
                "backend": unique_backends(result, trace_rows),
                "estimator_objective_evaluations": objective_evaluations(result, trace_rows),
                "estimator_requested_shots": requested_shots(result, trace_rows),
                "estimator_resilience_level": "2",
                "estimator_custom_mitigation_options": "none",
                "sampler_final_sampling_shots": final_sampler_shots(result, trace_rows),
                "sampler_custom_noise_management_options": "none",
                "final_bitstrings_from_raw_default_sampler": "true",
                "runtime_usage_seconds_if_available": runtime_usage_seconds(result, trace_rows),
                "wall_clock_seconds_if_available": wall_clock_seconds(result, trace_rows),
                "paired_level0_overhead_measured": "false",
                "overhead_reporting_note": OVERHEAD_NOTE,
            }
        )
    return rows


def write_csv(rows: list[dict[str, str]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / "mitigation_execution_provenance.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def write_digest(rows: list[dict[str, str]]) -> None:
    by_problem = Counter(row["problem"] for row in rows)
    by_method = Counter(row["method"] for row in rows)
    backends = sorted({b for row in rows for b in row["backend"].split(";") if b != "not_available"})
    mixed_fallback_rows = [
        row
        for row in rows
        if "aer_simulator_mps" in row["backend"] or "statevector_primitives" in row["backend"]
    ]
    lines = [
        "# Mitigation and Execution Provenance Digest",
        "",
        "## Corrected execution statement",
        "",
        "Circuit compilation used `optimization_level=3` for backend-aware mapping, decomposition, routing, and optimization. This is transpilation, not error mitigation.",
        "",
        "Variational objective evaluations used Runtime Estimator jobs with `resilience_level=2`; the historical protocol did not archive custom TREX, ZNE, twirling, extrapolator, or dynamical-decoupling settings beyond this managed preset.",
        "",
        "Final candidate bitstrings came from a separate raw/default SamplerV2 path and then classical decoding, feasibility checks, and one-round local refinement. Therefore ZNE/TREX affected Estimator expectation-value evaluations used by the COBYLA objective, not a gradient and not the final sampled bitstrings used for decoding.",
        "",
        "## Overhead answer",
        "",
        "The archived campaign does not contain paired `resilience_level=0` jobs for the same circuits. The paper should not report a circuit-specific mitigation-overhead multiplier retrospectively. It can state that level-2 Runtime Estimator mitigation increases execution cost through managed ensembles of related circuits, but the exact historical multiplier was not isolated.",
        "",
        "## Archived metadata coverage",
        "",
        f"- Provenance rows: {len(rows)}",
        f"- Problems: {', '.join(f'{k}={v}' for k, v in sorted(by_problem.items()))}",
        f"- Methods: {', '.join(f'{k}={v}' for k, v in sorted(by_method.items()))}",
        f"- Backends observed in archived metadata: {', '.join(backends) if backends else 'not_available'}",
        f"- Rows with mixed local fallback metadata alongside IBM metadata: {len(mixed_fallback_rows)}",
        "",
        "## Claims to remove",
        "",
        "- Do not claim archived custom values for `resilience.measure_mitigation`, TREX randomizations/shots, ZNE noise factors, or ZNE extrapolator.",
        "- Do not claim ZNE is unbiased in the limit of a perfect noise model.",
        "- Do not describe final SamplerV2 bitstrings as TREX- or ZNE-mitigated.",
    ]
    (OUT / "mitigation_execution_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_manuscript_insert() -> None:
    lines = [
        "# Manuscript Correction Insert",
        "",
        "## Execution and Mitigation Pipeline",
        "",
        "| Stage | Primitive / setting | Role in reported result |",
        "| --- | --- | --- |",
        "| Circuit compilation | `optimization_level=3` | Backend-aware mapping, gate decomposition, routing, and circuit optimization before execution. This is transpilation, not error mitigation. |",
        "| Variational objective evaluation | Runtime EstimatorV2 with `resilience_level=2` | Runtime-managed mitigation affected expectation-value estimates used by COBYLA objective evaluations. |",
        "| Final candidate sampling | Separate SamplerV2 with raw/default options | Unmitigated/default bitstring distribution used for decoding. |",
        "| Decoding and local improvement | Classical code | Candidate selection, feasibility checks, and shared one-round local refinement. |",
        "",
        "ZNE/TREX affected Estimator expectation-value evaluations during optimization, not the final sampled bitstrings used to decode solutions. Because the optimizer path used COBYLA, these mitigated Estimator values informed gradient-free objective evaluations, not gradient calculations.",
        "",
        "We requested IBM Runtime's managed medium-resilience preset through `resilience_level=2`. We did not manually configure or archive custom TREX, ZNE noise factors, ZNE extrapolators, twirling factors, or dynamical-decoupling settings. Therefore the manuscript should not report those settings as historical protocol details.",
        "",
        "## Mitigation Overhead",
        "",
        "Estimator `resilience_level=2` can increase execution cost by using managed ensembles of related circuits for mitigation. However, the historical campaign did not include paired `resilience_level=0` executions of the same circuits. We therefore do not estimate a circuit-specific mitigation-overhead multiplier retrospectively. The provenance ledger reports this explicitly for each archived result.",
    ]
    (OUT / "manuscript_mitigation_execution_insert.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    rows = build_rows()
    write_csv(rows)
    write_digest(rows)
    write_manuscript_insert()
    print(f"wrote {len(rows)} provenance rows")


if __name__ == "__main__":
    main()
