#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
QOBENCH_SRC = REPO_ROOT / "research_benchmark" / "src"
if str(QOBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(QOBENCH_SRC))

from qobench.hardware_cli import (  # noqa: E402
    _import_qubo_tools,
    _reconstruct_problem_objective,
    _run_one_round_local_swap,
)
from qobench.problem_registry import get_problem  # noqa: E402
from qobench.quantum_methods import (  # noqa: E402
    _group_reversed_pce_pauli_operators,
    _maxcut_edges_from_weight_matrix,
    _maxcut_to_qubo_bitstring,
    _pce_weighted_nu,
    _qubo_to_maxcut_weight_matrix,
    build_algorithm_ansatz_bundle,
    estimate_pce_num_qubits,
    generate_pce_pauli_strings,
)
from qobench.types import ProblemType  # noqa: E402


SEEDS_3 = [1103, 4409, 7703]
SEEDS_8 = [1103, 2203, 3301, 4409, 5501, 6607, 7703, 8807]


@dataclass(frozen=True)
class CaseSpec:
    case_id: str
    problem: str
    instance: str
    note: str


DEFAULT_CASES = [
    CaseSpec(
        "S1_mis_1tc8_pce",
        "mis",
        "Maximum_Independent_Set/mis_benchmark_instances/1tc.8.txt",
        "Small current PCE case; exact MIS optimum is brute-force recoverable.",
    ),
    CaseSpec(
        "S2_mis_1tc16_pce",
        "mis",
        "Maximum_Independent_Set/mis_benchmark_instances/1tc.16.txt",
        "Current PCE case already present in the P0.2 reduced Part B run.",
    ),
    CaseSpec(
        "S3_qap_nug12_pce",
        "qap",
        "Quadratic_Assignment_Problem/qapdata/nug12.dat",
        "Smallest QAP PCE case with an appendix-table optimum.",
    ),
]


RUN_COLUMNS = [
    "case_id",
    "problem",
    "instance",
    "optimizer_name",
    "ansatz_family",
    "ansatz_reps",
    "ansatz_parameters",
    "logical_variables",
    "encoded_qubits",
    "compression_k",
    "initialization_seed",
    "maxiter",
    "ftol",
    "eps",
    "nit",
    "nfev",
    "njev",
    "status",
    "message",
    "success",
    "initial_objective",
    "best_objective",
    "final_objective",
    "final_decoded_objective",
    "final_decoded_gap_percent",
    "decoded_feasible",
    "raw_decoded_objective",
    "raw_decoded_feasible",
    "postprocess_improved",
    "candidates_checked",
    "shots_per_objective",
    "final_sampling_shots",
    "total_shots_estimate",
    "runtime_seconds",
]


SUMMARY_COLUMNS = [
    "case_id",
    "problem",
    "instance",
    "optimizer_name",
    "maxiter",
    "n_initializations",
    "median_nfev",
    "min_nfev",
    "max_nfev",
    "success_rate",
    "cap_hit_rate",
    "feasible_rate",
    "best_quality",
    "median_quality",
    "worst_quality",
    "median_runtime_seconds",
]


def write_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in columns})


def read_qap_optima() -> dict[str, float]:
    path = REPO_ROOT / "research_benchmark/research_benchmark/results_hardware/qap/qap_pce/csv/main_table_pce.csv"
    out: dict[str, float] = {}
    if not path.exists():
        return out
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            inst = str(row.get("Instance", "")).strip()
            try:
                out[inst] = float(row.get("Optimal", ""))
            except Exception:
                pass
    return out


def brute_force_mis_optimum(instance: Any) -> int | None:
    n = int(getattr(instance, "num_nodes"))
    if n > 24:
        return None
    edges = [(int(u), int(v)) for u, v in getattr(instance, "edges")]
    best = 0
    for mask in range(1 << n):
        bits = [(mask >> i) & 1 for i in range(n)]
        if any(bits[u] and bits[v] for u, v in edges):
            continue
        best = max(best, int(sum(bits)))
    return best


def known_optimum(problem: str, instance_name: str, instance: Any) -> float | None:
    if problem == "mis":
        opt = brute_force_mis_optimum(instance)
        return float(opt) if opt is not None else None
    if problem == "mkp":
        try:
            return float(getattr(instance, "optimal_value"))
        except Exception:
            return None
    if problem == "qap":
        stem = Path(instance_name).stem
        return read_qap_optima().get(stem)
    return None


def gap_percent(problem: str, value: Any, optimum: float | None, feasible: bool) -> float | str:
    if optimum is None or not feasible:
        return "inf" if not feasible else "unknown"
    try:
        v = float(value)
    except Exception:
        return "unknown"
    if not math.isfinite(v) or not math.isfinite(float(optimum)):
        return "unknown"
    if problem in {"mis", "mkp"}:
        if optimum == 0:
            return 0.0 if math.isclose(v, optimum) else "unknown"
        return float(max(0.0, (float(optimum) - v) / abs(float(optimum)) * 100.0))
    if problem == "qap":
        if optimum == 0:
            return 0.0 if math.isclose(v, optimum) else "unknown"
        return float(max(0.0, (v - float(optimum)) / abs(float(optimum)) * 100.0))
    return "unknown"


def build_qubo(problem_key: str, instance_path: Path) -> tuple[Any, Any, Any, Any]:
    problem_type = ProblemType(problem_key)
    problem = get_problem(problem_type)
    instance = problem.load_instance(instance_path)
    model, _ = problem.build_model(instance=instance)
    from_docplex_mp, QuadraticProgramToQubo = _import_qubo_tools()
    qp = from_docplex_mp(model)
    qubo = QuadraticProgramToQubo().convert(qp)
    return problem_type, instance, qp, qubo


def evaluate_current_pce_loss_factory(
    *,
    qubo: Any,
    pce_reps: int,
    compression_k: int,
) -> tuple[Any, list[Any], int, int]:
    from qiskit.quantum_info import Statevector

    weight_matrix = _qubo_to_maxcut_weight_matrix(qubo)
    edges = _maxcut_edges_from_weight_matrix(weight_matrix)
    num_nodes = int(weight_matrix.shape[0])
    encoded_qubits = estimate_pce_num_qubits(
        num_variables=num_nodes,
        compression_k=compression_k,
    )

    ansatz = build_algorithm_ansatz_bundle(
        method="pce",
        qubo=qubo,
        layers=3,
        entanglement="circular",
        qp=None,
        pce_compression_k=compression_k,
        pce_depth=pce_reps,
    )
    circuit = ansatz.qiskit_template
    params = list(ansatz.qiskit_parameters)
    pauli_strings = generate_pce_pauli_strings(
        num_qubits=encoded_qubits,
        num_variables=num_nodes,
        compression_k=compression_k,
    )
    blocks = _group_reversed_pce_pauli_operators(pauli_strings)
    observables = [op for block in blocks for op in block]
    alpha = float(encoded_qubits)
    beta = 0.5
    nu = _pce_weighted_nu(num_nodes, edges)

    def evaluate(theta: np.ndarray) -> dict[str, Any]:
        assignment = {param: float(theta[idx]) for idx, param in enumerate(params)}
        bound = circuit.assign_parameters(assignment, inplace=False)
        state = Statevector.from_instruction(bound)
        node_exp = []
        for op in observables:
            node_exp.append(float(np.real(state.expectation_value(op))))
        node_exp = np.asarray(node_exp[:num_nodes], dtype=float)

        edge_loss = 0.0
        for u, v, w in edges:
            edge_loss += float(w) * float(np.tanh(alpha * node_exp[u])) * float(np.tanh(alpha * node_exp[v]))
        reg_term = np.tanh(alpha * node_exp) ** 2
        reg_loss = float(beta * nu * (float(np.mean(reg_term)) ** 2))
        total_loss = float(edge_loss + reg_loss)

        maxcut_bits = [1 if float(node_exp[i]) >= 0.0 else 0 for i in range(num_nodes)]
        qubo_bitstring = _maxcut_to_qubo_bitstring(maxcut_bits)
        logical_vars = int(qubo.get_num_vars())
        if len(qubo_bitstring) < logical_vars:
            qubo_bitstring = qubo_bitstring.zfill(logical_vars)
        elif len(qubo_bitstring) > logical_vars:
            qubo_bitstring = qubo_bitstring[-logical_vars:]
        return {
            "loss": total_loss,
            "bitstring": qubo_bitstring,
            "node_expectations": [float(x) for x in node_exp],
        }

    return evaluate, params, encoded_qubits, int(ansatz.num_parameters)


def run_case(
    *,
    case: CaseSpec,
    seeds: list[int],
    maxiter_values: list[int],
    shots: int,
    final_shots: int,
    pce_reps: int,
    compression_k: int,
    ftol: float,
    eps: float,
) -> list[dict[str, Any]]:
    instance_path = REPO_ROOT / case.instance
    problem_type, instance, _qp, qubo = build_qubo(case.problem, instance_path)
    evaluator, _params, encoded_qubits, num_params = evaluate_current_pce_loss_factory(
        qubo=qubo,
        pce_reps=pce_reps,
        compression_k=compression_k,
    )
    variable_names = [
        var_name for var_name, _idx in sorted(qubo.variables_index.items(), key=lambda item: int(item[1]))
    ]
    optimum = known_optimum(case.problem, instance_path.name, instance)
    logical_variables = int(qubo.get_num_vars())
    rows: list[dict[str, Any]] = []

    for maxiter in maxiter_values:
        for seed in seeds:
            rng = np.random.default_rng(int(seed))
            x0 = rng.uniform(0.0, 1.0, size=(num_params,))
            best = {
                "objective": float("inf"),
                "bitstring": "0" * logical_variables,
            }
            initial_eval = evaluator(x0)
            initial_objective = float(initial_eval["loss"])

            def objective(theta: np.ndarray) -> float:
                evaluated = evaluator(theta)
                value = float(evaluated["loss"])
                if value < float(best["objective"]):
                    best["objective"] = value
                    best["bitstring"] = str(evaluated["bitstring"])
                return value

            start = time.perf_counter()
            result = minimize(
                objective,
                x0,
                method="SLSQP",
                options={"maxiter": int(maxiter), "ftol": float(ftol), "eps": float(eps), "disp": False},
            )
            runtime = float(time.perf_counter() - start)

            final_eval = evaluator(np.asarray(result.x, dtype=float))
            final_bitstring = str(final_eval["bitstring"])
            if float(final_eval["loss"]) < float(best["objective"]):
                best["objective"] = float(final_eval["loss"])
                best["bitstring"] = final_bitstring

            raw_assignment = {
                name: int(final_bitstring[idx]) if idx < len(final_bitstring) else 0
                for idx, name in enumerate(variable_names)
            }
            raw_reconstructed = _reconstruct_problem_objective(
                problem=problem_type,
                instance=instance,
                assignment=raw_assignment,
            )
            local_swap = _run_one_round_local_swap(
                problem=problem_type,
                instance=instance,
                assignment=raw_assignment,
                variable_names_in_order=variable_names,
            )
            reconstructed = dict(local_swap["reconstructed"])
            decoded_value = reconstructed.get("objective_value")
            feasible = bool(reconstructed.get("feasible"))
            nfev = int(getattr(result, "nfev", 0) or 0)
            rows.append(
                {
                    "case_id": case.case_id,
                    "problem": case.problem,
                    "instance": instance_path.name,
                    "optimizer_name": "SLSQP",
                    "ansatz_family": "current_PCE_EfficientSU2",
                    "ansatz_reps": int(pce_reps),
                    "ansatz_parameters": int(num_params),
                    "logical_variables": int(logical_variables),
                    "encoded_qubits": int(encoded_qubits),
                    "compression_k": int(compression_k),
                    "initialization_seed": int(seed),
                    "maxiter": int(maxiter),
                    "ftol": float(ftol),
                    "eps": float(eps),
                    "nit": int(getattr(result, "nit", 0) or 0),
                    "nfev": nfev,
                    "njev": int(getattr(result, "njev", 0) or 0),
                    "status": int(getattr(result, "status", -999)),
                    "message": str(getattr(result, "message", "")),
                    "success": bool(getattr(result, "success", False)),
                    "initial_objective": initial_objective,
                    "best_objective": float(best["objective"]),
                    "final_objective": float(final_eval["loss"]),
                    "final_decoded_objective": decoded_value,
                    "final_decoded_gap_percent": gap_percent(case.problem, decoded_value, optimum, feasible),
                    "decoded_feasible": feasible,
                    "raw_decoded_objective": raw_reconstructed.get("objective_value"),
                    "raw_decoded_feasible": bool(raw_reconstructed.get("feasible")),
                    "postprocess_improved": bool(local_swap["improved"]),
                    "candidates_checked": int(local_swap["candidates_checked"]),
                    "shots_per_objective": int(shots),
                    "final_sampling_shots": int(final_shots),
                    "total_shots_estimate": int(nfev * int(shots) + int(final_shots)),
                    "runtime_seconds": runtime,
                }
            )
            print(
                f"{case.case_id} seed={seed} maxiter={maxiter} "
                f"nit={rows[-1]['nit']} nfev={nfev} success={rows[-1]['success']} "
                f"decoded={decoded_value} feasible={feasible} time={runtime:.1f}s",
                flush=True,
            )
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    grouped: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((str(row["case_id"]), int(row["maxiter"])), []).append(row)

    for (case_id, maxiter), group in sorted(grouped.items()):
        qualities: list[float] = []
        for row in group:
            gap = row.get("final_decoded_gap_percent")
            try:
                value = float(gap)
            except Exception:
                value = float("inf")
            qualities.append(value)
        finite = [q for q in qualities if math.isfinite(q)]
        quality_for_sort = qualities if qualities else [float("inf")]
        out.append(
            {
                "case_id": case_id,
                "problem": group[0]["problem"],
                "instance": group[0]["instance"],
                "optimizer_name": "SLSQP",
                "maxiter": int(maxiter),
                "n_initializations": len(group),
                "median_nfev": float(np.median([float(r["nfev"]) for r in group])),
                "min_nfev": int(min(int(r["nfev"]) for r in group)),
                "max_nfev": int(max(int(r["nfev"]) for r in group)),
                "success_rate": float(np.mean([1.0 if bool(r["success"]) else 0.0 for r in group])),
                "cap_hit_rate": float(np.mean([1.0 if int(r["nit"]) >= int(maxiter) else 0.0 for r in group])),
                "feasible_rate": float(np.mean([1.0 if bool(r["decoded_feasible"]) else 0.0 for r in group])),
                "best_quality": min(quality_for_sort),
                "median_quality": float(np.median(finite)) if finite else "inf",
                "worst_quality": max(quality_for_sort),
                "median_runtime_seconds": float(np.median([float(r["runtime_seconds"]) for r in group])),
            }
        )
    return out


def audit_reported_pce_artifacts() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((REPO_ROOT / "research_benchmark/research_benchmark/results_hardware").glob("*/*_pce/*/result.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        protocol = data.get("benchmark_protocol", {})
        budget = protocol.get("budget", {})
        stopping = protocol.get("stopping_rule", {})
        circuit = data.get("circuit_metrics", {})
        optimizer = data.get("optimizer", {})
        rows.append(
            {
                "artifact": str(path.relative_to(REPO_ROOT)),
                "problem": data.get("problem"),
                "instance": data.get("instance_name"),
                "ansatz_family": circuit.get("ansatz_family"),
                "ansatz_reps": circuit.get("ansatz_reps"),
                "trainable_parameters": circuit.get("trainable_parameters"),
                "optimizer_recorded": "COBYLA_in_source_path",
                "max_optimizer_iterations": budget.get("max_optimizer_iterations"),
                "total_circuit_evaluations": budget.get("total_circuit_evaluations") or optimizer.get("total_evaluations"),
                "optimizer_status": stopping.get("optimizer_status") or optimizer.get("status"),
                "optimizer_message": stopping.get("optimizer_message") or optimizer.get("message"),
                "artifact_replayable_exact_slsqp": False,
            }
        )
    return rows


def write_markdown(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    summary_rows: list[dict[str, Any]],
    audit_rows: list[dict[str, Any]],
    seeds: list[int],
    maxiter_values: list[int],
) -> None:
    lines: list[str] = []
    lines.append("# PCE/SLSQP Follow-up Digest")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(f"- Reduced executable SLSQP diagnostic: {len(set(r['case_id'] for r in rows))} current-PCE cases, {len(seeds)} seeds, maxiter values {maxiter_values}.")
    lines.append("- Historical PCE artifact audit: saved hardware PCE `result.json` files under `research_benchmark/research_benchmark/results_hardware/*/*_pce`.")
    lines.append("- This is not an exact replay of the historical 182-parameter Brickwork MDKP PCE run because the saved artifacts do not serialize the original circuit/parameter-to-gate mapping and the benchmark source path records COBYLA, not SLSQP.")
    lines.append("")
    lines.append("## Historical Artifact Audit")
    lines.append("")
    lines.append(f"- PCE artifacts audited: {len(audit_rows)}.")
    if audit_rows:
        families = sorted(set(str(r.get("ansatz_family")) for r in audit_rows))
        params = sorted(set(str(r.get("trainable_parameters")) for r in audit_rows))
        maxiters = sorted(set(str(r.get("max_optimizer_iterations")) for r in audit_rows))
        evals = sorted(set(str(r.get("total_circuit_evaluations")) for r in audit_rows))
        lines.append(f"- Recorded ansatz families: {', '.join(families)}.")
        lines.append(f"- Recorded trainable-parameter counts: {', '.join(params)}.")
        lines.append(f"- Recorded max optimizer iterations: {', '.join(maxiters)}.")
        lines.append(f"- Recorded objective/circuit evaluations: {', '.join(evals)}.")
    lines.append("- Git/source audit found SLSQP in exploratory PCE notebooks and QRAO optimizer support, while the benchmark PCE runner calls SciPy COBYLA.")
    lines.append("")
    lines.append("## Reduced SLSQP Diagnostic Summary")
    lines.append("")
    lines.append("| case | maxiter | seeds | median nfev | success rate | cap-hit rate | feasible rate | best gap | median gap | worst gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in summary_rows:
        lines.append(
            "| {case_id} | {maxiter} | {n_initializations} | {median_nfev:.1f} | {success_rate:.2f} | {cap_hit_rate:.2f} | {feasible_rate:.2f} | {best_quality} | {median_quality} | {worst_quality} |".format(
                **row
            )
        )
    lines.append("")
    lines.append("## Interpretation")
    lines.append("")
    lines.append("- SLSQP iterations are not comparable to COBYLA objective evaluations; the table reports `nfev` as the cross-method budget metric.")
    lines.append("- The reduced current-PCE diagnostic is useful for answering the SLSQP budget conceptually, but it should not be used to validate the historical Brickwork PCE hardware results.")
    lines.append("- For the historical PCE benchmark results, the defensible revision is to report the recovered fixed-budget metadata and state that exact SLSQP replay is not available from the saved artifacts.")
    lines.append("")
    lines.append("## Recommended Rebuttal Text")
    lines.append("")
    lines.append("> We revisited the PCE optimizer records in response to the SLSQP/budget concern. The saved benchmark artifacts and the benchmark runner record PCE as a fixed-budget run with 200 objective/circuit evaluations in the reported hardware artifacts; the current source path uses SciPy COBYLA for PCE, while SLSQP appears only in exploratory PCE notebooks and in the QRAO optimizer option. Because the historical PCE artifacts do not serialize the original circuit or parameter-to-gate mapping, we do not claim exact replay of the historical Brickwork PCE runs. We therefore revised the manuscript to report objective evaluations rather than treating optimizer iterations as equivalent across methods, and we added a reduced SLSQP diagnostic on current PCE cases to illustrate the difference between SLSQP iterations and realized objective evaluations.")
    lines.append("")
    lines.append("## Appendix Table")
    lines.append("")
    lines.append("```latex")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\caption{Reduced PCE SLSQP diagnostic. Objective evaluations rather than optimizer iterations are reported as the cross-method cost metric. This diagnostic uses the current reproducible PCE implementation and is not an exact replay of the historical Brickwork PCE hardware artifacts.}")
    lines.append(r"\label{tab:pce_slsqp_diagnostic}")
    lines.append(r"\footnotesize")
    lines.append(r"\begin{tabular}{llrrrrrrrr}")
    lines.append(r"\toprule")
    lines.append(r"Problem & Instance & Max. iter. & Median $n_{\mathrm{fev}}$ & Success rate & Cap-hit rate & Best gap & Median gap & Worst gap & Feasible rate \\")
    lines.append(r"\midrule")
    for row in summary_rows:
        lines.append(
            f"{row['problem'].upper()} & \\texttt{{{row['instance']}}} & {row['maxiter']} & "
            f"{float(row['median_nfev']):.1f} & {float(row['success_rate']):.2f} & "
            f"{float(row['cap_hit_rate']):.2f} & {row['best_quality']} & {row['median_quality']} & "
            f"{row['worst_quality']} & {float(row['feasible_rate']):.2f} \\\\"
        )
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")
    lines.append("```")
    (out_dir / "pce_slsqp_followup_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seeds", default=",".join(str(s) for s in SEEDS_3))
    parser.add_argument("--use-eight-seeds", action="store_true")
    parser.add_argument("--maxiters", default="100,200")
    parser.add_argument("--shots", type=int, default=1000)
    parser.add_argument("--final-shots", type=int, default=1000)
    parser.add_argument("--pce-reps", type=int, default=2)
    parser.add_argument("--compression-k", type=int, default=2)
    parser.add_argument("--ftol", type=float, default=1e-6)
    parser.add_argument("--eps", type=float, default=1.4901161193847656e-8)
    parser.add_argument(
        "--case-ids",
        default=",".join(case.case_id for case in DEFAULT_CASES),
        help="Comma-separated subset of default case ids.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_dir = Path(args.output_dir) if args.output_dir else EXPERIMENT_DIR / f"pce_slsqp_followup_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    seeds = SEEDS_8 if bool(args.use_eight_seeds) else [int(x) for x in str(args.seeds).split(",") if x.strip()]
    maxiter_values = [int(x) for x in str(args.maxiters).split(",") if x.strip()]
    requested_cases = {x.strip() for x in str(args.case_ids).split(",") if x.strip()}
    cases = [case for case in DEFAULT_CASES if case.case_id in requested_cases]
    if not cases:
        raise ValueError("No cases selected.")

    all_rows: list[dict[str, Any]] = []
    for case in cases:
        all_rows.extend(
            run_case(
                case=case,
                seeds=seeds,
                maxiter_values=maxiter_values,
                shots=int(args.shots),
                final_shots=int(args.final_shots),
                pce_reps=int(args.pce_reps),
                compression_k=int(args.compression_k),
                ftol=float(args.ftol),
                eps=float(args.eps),
            )
        )

    summary_rows = summarize(all_rows)
    audit_rows = audit_reported_pce_artifacts()
    write_csv(out_dir / "pce_slsqp_diagnostic_runs.csv", all_rows, RUN_COLUMNS)
    write_csv(out_dir / "pce_slsqp_diagnostic_summary.csv", summary_rows, SUMMARY_COLUMNS)
    write_csv(
        out_dir / "reported_pce_artifact_audit.csv",
        audit_rows,
        [
            "artifact",
            "problem",
            "instance",
            "ansatz_family",
            "ansatz_reps",
            "trainable_parameters",
            "optimizer_recorded",
            "max_optimizer_iterations",
            "total_circuit_evaluations",
            "optimizer_status",
            "optimizer_message",
            "artifact_replayable_exact_slsqp",
        ],
    )
    (out_dir / "run_config.json").write_text(
        json.dumps(
            {
                "seeds": seeds,
                "maxiter_values": maxiter_values,
                "shots": int(args.shots),
                "final_shots": int(args.final_shots),
                "pce_reps": int(args.pce_reps),
                "compression_k": int(args.compression_k),
                "ftol": float(args.ftol),
                "eps": float(args.eps),
                "cases": [case.__dict__ for case in cases],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    write_markdown(
        out_dir=out_dir,
        rows=all_rows,
        summary_rows=summary_rows,
        audit_rows=audit_rows,
        seeds=seeds,
        maxiter_values=maxiter_values,
    )
    print(f"Wrote PCE/SLSQP follow-up to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
