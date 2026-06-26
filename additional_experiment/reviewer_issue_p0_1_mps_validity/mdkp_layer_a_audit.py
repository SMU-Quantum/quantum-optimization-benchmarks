#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
QOBENCH_SRC = REPO_ROOT / "research_benchmark" / "src"
if str(QOBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(QOBENCH_SRC))

from qobench.hardware_cli import (  # noqa: E402
    _decode_problem_solution,
    _reconstruct_problem_objective,
    _run_one_round_local_swap,
)
from qobench.problem_registry import get_problem  # noqa: E402
from qobench.quantum_methods import (  # noqa: E402
    QuboObjective,
    build_algorithm_ansatz_bundle,
)
from qobench.types import ProblemType  # noqa: E402


@dataclass(slots=True)
class AuditCase:
    case_id: str
    result_path: Path
    method: str
    instance_name: str
    instance_path: Path
    shots: int
    seed: int
    layers: int
    entanglement: str
    cvar_alpha: float
    pce_compression_k: int
    pce_depth: int
    ws_epsilon: float
    theta: list[float]


def _run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    data = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model_sysctl": _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_memory_bytes_sysctl": _run_text(["sysctl", "-n", "hw.memsize"]),
        "cpu_count": os.cpu_count(),
        "gpu_used": False,
    }
    try:
        import qiskit
        import qiskit_aer
        from qiskit_aer import AerSimulator

        sim = AerSimulator(method="matrix_product_state")
        data["qiskit_version"] = getattr(qiskit, "__version__", None)
        data["qiskit_aer_version"] = getattr(qiskit_aer, "__version__", None)
        data["aer_mps_default_options"] = {
            "matrix_product_state_max_bond_dimension": getattr(
                sim.options, "matrix_product_state_max_bond_dimension", None
            ),
            "matrix_product_state_truncation_threshold": getattr(
                sim.options, "matrix_product_state_truncation_threshold", None
            ),
            "mps_omp_threads": getattr(sim.options, "mps_omp_threads", None),
            "device": getattr(sim.options, "device", None),
            "precision": getattr(sim.options, "precision", None),
        }
    except Exception as exc:
        data["qiskit_import_error"] = repr(exc)
    return data


def find_instance(instance_name: str) -> Path:
    problem = get_problem(ProblemType.MKP)
    for path in problem.list_instances(REPO_ROOT, limit=None):
        if path.name == instance_name:
            return path.resolve()
    raise FileNotFoundError(f"Could not resolve MKP instance {instance_name}")


def load_case(result_path: Path, case_id: str) -> AuditCase:
    data = json.loads(result_path.read_text(encoding="utf-8"))
    config = data.get("config", {})
    best = data.get("best_result", {})
    method = str(data.get("execution", {}).get("method") or config.get("method"))
    instance_name = str(data.get("instance_name"))
    theta = best.get("best_theta")
    if not isinstance(theta, list) or not theta:
        raise ValueError(f"No best_theta found in {result_path}")
    return AuditCase(
        case_id=case_id,
        result_path=result_path.resolve(),
        method=method,
        instance_name=instance_name,
        instance_path=find_instance(instance_name),
        shots=int(config.get("shots", 1000)),
        seed=int(config.get("seed", 0)),
        layers=int(config.get("layers", 3)),
        entanglement=str(config.get("entanglement", "circular")),
        cvar_alpha=float(config.get("cvar_alpha", 0.25)),
        pce_compression_k=int(config.get("pce_compression_k", 2)),
        pce_depth=int(config.get("pce_depth", 0)),
        ws_epsilon=float(config.get("ws_epsilon", 1e-3)),
        theta=[float(x) for x in theta],
    )


def bind_case_circuit(case: AuditCase) -> tuple[Any, Any, Any, Any, dict[str, Any]]:
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import from_docplex_mp

    problem = get_problem(ProblemType.MKP)
    instance = problem.load_instance(case.instance_path, seed=case.seed)
    model, _context = problem.build_model(instance=instance, time_limit_sec=60.0)
    qp = from_docplex_mp(model)
    qubo = QuadraticProgramToQubo().convert(qp)
    objective = QuboObjective(qubo=qubo)
    ansatz = None
    if case.method in {"vqe", "cvar_vqe"}:
        expected_legacy_params = int(qubo.get_num_vars()) * (int(case.layers) + 1)
        if len(case.theta) == expected_legacy_params:
            circuit = build_legacy_ry_ansatz_circuit(
                num_qubits=int(qubo.get_num_vars()),
                layers=int(case.layers),
                entanglement=case.entanglement,
                theta=case.theta,
            )
        else:
            ansatz = build_algorithm_ansatz_bundle(
                method=case.method,
                qubo=qubo,
                layers=case.layers,
                entanglement=case.entanglement,
                qp=qp,
                ws_epsilon=case.ws_epsilon,
                pce_compression_k=case.pce_compression_k,
                pce_depth=case.pce_depth,
            )
            circuit = ansatz.qiskit_template
    else:
        ansatz = build_algorithm_ansatz_bundle(
            method=case.method,
            qubo=qubo,
            layers=case.layers,
            entanglement=case.entanglement,
            qp=qp,
            ws_epsilon=case.ws_epsilon,
            pce_compression_k=case.pce_compression_k,
            pce_depth=case.pce_depth,
        )
        circuit = ansatz.qiskit_template

    num_parameters = len(case.theta) if ansatz is None else int(ansatz.num_parameters)
    if int(num_parameters) != len(case.theta):
        raise ValueError(
            f"Cannot reconstruct exact {case.case_id}: stored theta has {len(case.theta)} "
            f"parameters but current ansatz builder produces {ansatz.num_parameters}."
        )
    if ansatz is not None:
        circuit = circuit.assign_parameters(
            {param: case.theta[idx] for idx, param in enumerate(ansatz.qiskit_parameters)}
        )
    if int(circuit.num_clbits) < int(circuit.num_qubits):
        from qiskit import ClassicalRegister

        circuit.add_register(ClassicalRegister(int(circuit.num_qubits), "c"))
    if not any(inst.operation.name == "measure" for inst in circuit.data):
        circuit.measure(range(int(circuit.num_qubits)), range(int(circuit.num_qubits)))
    metrics = {
        "num_qubits": int(circuit.num_qubits),
        "depth": int(circuit.depth() or 0),
        "size": int(circuit.size() or 0),
        "num_parameters": int(num_parameters),
        "logical_qubo_variables": int(qubo.get_num_vars()),
        "bks_optimal_value": int(instance.optimal_value),
    }
    return circuit, qubo, objective, instance, metrics


def build_legacy_ry_ansatz_circuit(
    *,
    num_qubits: int,
    layers: int,
    entanglement: str,
    theta: list[float],
) -> Any:
    from qiskit import QuantumCircuit

    n = int(num_qubits)
    qc = QuantumCircuit(n, n)
    cursor = 0

    def entangle() -> None:
        if n < 2:
            return
        if entanglement == "full":
            pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        elif entanglement == "chain":
            pairs = [(i, i + 1) for i in range(n - 1)]
        else:
            pairs = [(i, i + 1) for i in range(n - 1)] + [(n - 1, 0)]
        for u, v in pairs:
            qc.cx(u, v)

    for layer in range(int(layers) + 1):
        for q in range(n):
            qc.ry(float(theta[cursor]), q)
            cursor += 1
        if layer < int(layers):
            entangle()
    qc.measure(range(n), range(n))
    return qc


def parse_bond_dimensions_from_log(raw: str | None) -> list[int]:
    import re

    if not raw:
        return []
    out: list[int] = []
    for match in re.finditer(r"BD=\[([^\]]*)\]", raw):
        for token in match.group(1).split():
            try:
                out.append(int(token))
            except Exception:
                pass
    return out


def run_bond_diagnostic(
    circuit_with_measurements: Any,
    *,
    threshold: float,
    bond_dimension: int | None,
) -> dict[str, Any]:
    from qiskit_aer import AerSimulator

    qc = circuit_with_measurements.remove_final_measurements(inplace=False)
    qc.save_matrix_product_state(label="mps")
    options: dict[str, Any] = {
        "method": "matrix_product_state",
        "mps_log_data": True,
        "matrix_product_state_truncation_threshold": float(threshold),
    }
    if bond_dimension is not None:
        options["matrix_product_state_max_bond_dimension"] = int(bond_dimension)
    backend = AerSimulator(**options)
    t0 = time.perf_counter()
    result = backend.run(qc, shots=1).result()
    elapsed = time.perf_counter() - t0
    metadata = dict(result.results[0].metadata)
    saved_bonds: list[int] = []
    try:
        _gammas, lambdas = result.data(0)["mps"]
        saved_bonds = [int(len(values)) for values in lambdas]
    except Exception:
        saved_bonds = []
    log_bonds = parse_bond_dimensions_from_log(metadata.get("MPS_log_data"))
    all_bonds = saved_bonds + log_bonds
    return {
        "elapsed_sec": elapsed,
        "max_observed_bond_dimension": max(all_bonds) if all_bonds else None,
        "saved_bond_dimensions": saved_bonds,
        "log_bond_dimensions": log_bonds,
        "metadata": metadata,
    }


def normalize_counts(raw_counts: dict[Any, Any], num_qubits: int) -> dict[str, int]:
    counts: dict[str, int] = {}
    for key, value in raw_counts.items():
        if isinstance(key, int):
            bitstring = format(key, f"0{num_qubits}b")
        else:
            bitstring = str(key).replace(" ", "")
            if len(bitstring) < num_qubits:
                bitstring = bitstring.zfill(num_qubits)
            elif len(bitstring) > num_qubits:
                bitstring = bitstring[-num_qubits:]
        counts[bitstring] = counts.get(bitstring, 0) + int(value)
    return counts


def run_sampling(
    circuit: Any,
    *,
    shots: int,
    seed: int,
    threshold: float,
    bond_dimension: int | None,
) -> tuple[dict[str, int], dict[str, Any], float]:
    from qiskit_aer import AerSimulator

    options: dict[str, Any] = {
        "method": "matrix_product_state",
        "seed_simulator": int(seed),
        "matrix_product_state_truncation_threshold": float(threshold),
    }
    if bond_dimension is not None:
        options["matrix_product_state_max_bond_dimension"] = int(bond_dimension)
    backend = AerSimulator(**options)
    t0 = time.perf_counter()
    result = backend.run(circuit, shots=int(shots)).result()
    elapsed = time.perf_counter() - t0
    raw_counts = result.get_counts(0)
    metadata = dict(result.results[0].metadata)
    return normalize_counts(raw_counts, int(circuit.num_qubits)), metadata, elapsed


def total_variation_from_counts(a: dict[str, int], b: dict[str, int]) -> float:
    total_a = float(sum(a.values()) or 1)
    total_b = float(sum(b.values()) or 1)
    keys = set(a) | set(b)
    return 0.5 * sum(abs(a.get(k, 0) / total_a - b.get(k, 0) / total_b) for k in keys)


def aggregate_counts(counts_list: list[dict[str, int]]) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for counts in counts_list:
        counter.update(counts)
    return dict(counter)


def bitstring_reconstruction(
    *,
    bitstring: str,
    objective: Any,
    instance: Any,
    variable_names_in_order: list[str],
) -> dict[str, Any]:
    raw_assignment = objective.assignment(bitstring)
    local_swap = _run_one_round_local_swap(
        problem=ProblemType.MKP,
        instance=instance,
        assignment=raw_assignment,
        variable_names_in_order=variable_names_in_order,
    )
    assignment = dict(local_swap["assignment"])
    reconstructed = dict(local_swap["reconstructed"])
    decoded = _decode_problem_solution(ProblemType.MKP, assignment)
    return {
        "raw_assignment": raw_assignment,
        "assignment": assignment,
        "decoded": decoded,
        "reconstructed": reconstructed,
        "final_bitstring": str(local_swap["bitstring"]),
        "postprocess_improved": bool(local_swap["improved"]),
    }


def feasible_mass(counts: dict[str, int], objective: Any, instance: Any) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    mass = 0
    for bitstring, count in counts.items():
        reconstructed = _reconstruct_problem_objective(
            problem=ProblemType.MKP,
            instance=instance,
            assignment=objective.assignment(bitstring),
        )
        if bool(reconstructed.get("feasible")):
            mass += int(count)
    return float(mass) / float(total)


def summarize_setting(
    *,
    case: AuditCase,
    setting: dict[str, Any],
    counts_list: list[dict[str, int]],
    sampling_elapsed: list[float],
    objective: Any,
    instance: Any,
    variable_names_in_order: list[str],
    production_aggregate: dict[str, int] | None,
    production_top10: set[str] | None,
    bond_diag: dict[str, Any],
) -> dict[str, Any]:
    aggregate = aggregate_counts(counts_list)
    bks = float(instance.optimal_value)
    replicate_gaps: list[float] = []
    replicate_objectives: list[float] = []
    feasible_runs = 0
    best_gap = float("inf")
    best_objective = -float("inf")
    best_bitstring = ""
    for counts in counts_list:
        bitstring, _energy = objective.best_sample(counts)
        recon = bitstring_reconstruction(
            bitstring=bitstring,
            objective=objective,
            instance=instance,
            variable_names_in_order=variable_names_in_order,
        )
        reconstructed = recon["reconstructed"]
        feasible = bool(reconstructed.get("feasible"))
        objective_value = float(reconstructed.get("objective_value", float("nan")))
        if feasible:
            feasible_runs += 1
            gap = max(0.0, (bks - objective_value) / bks * 100.0) if bks else float("nan")
        else:
            gap = float("nan")
        if math.isfinite(gap):
            replicate_gaps.append(gap)
            if gap < best_gap:
                best_gap = gap
        if math.isfinite(objective_value):
            replicate_objectives.append(objective_value)
            if feasible and objective_value > best_objective:
                best_objective = objective_value
                best_bitstring = str(recon["final_bitstring"])
    expected_energy = objective.expectation(aggregate)
    cvar_energy = objective.cvar(aggregate, case.cvar_alpha)
    total_counts = sum(aggregate.values())
    top10_mass = ""
    if production_top10:
        top10_mass = sum(aggregate.get(k, 0) for k in production_top10) / float(total_counts or 1)
    return {
        "case_id": case.case_id,
        "method": case.method,
        "instance": case.instance_name,
        "setting_label": setting["label"],
        "threshold": setting["threshold"],
        "bond_dimension_cap": "" if setting["bond_dimension"] is None else setting["bond_dimension"],
        "shots_per_rep": case.shots,
        "repetitions": len(counts_list),
        "total_shots": total_counts,
        "max_observed_bond_dimension": bond_diag.get("max_observed_bond_dimension"),
        "bond_diagnostic_sec": bond_diag.get("elapsed_sec"),
        "sampling_runtime_sec_total": sum(sampling_elapsed),
        "sampling_runtime_sec_mean": statistics.mean(sampling_elapsed) if sampling_elapsed else "",
        "unique_bitstrings": len(aggregate),
        "tvd_vs_production_sampled": (
            total_variation_from_counts(aggregate, production_aggregate)
            if production_aggregate is not None
            else 0.0
        ),
        "expected_qubo_energy": expected_energy,
        "cvar_qubo_energy": cvar_energy,
        "feasible_sample_mass_raw": feasible_mass(aggregate, objective, instance),
        "production_top10_mass": top10_mass,
        "mean_gap_pct": statistics.mean(replicate_gaps) if replicate_gaps else "",
        "sd_gap_pct": statistics.stdev(replicate_gaps) if len(replicate_gaps) > 1 else 0.0 if replicate_gaps else "",
        "median_gap_pct": statistics.median(replicate_gaps) if replicate_gaps else "",
        "best_gap_pct": best_gap if math.isfinite(best_gap) else "",
        "feasible_run_fraction": feasible_runs / float(len(counts_list) or 1),
        "mean_best_recovered_objective": (
            statistics.mean(replicate_objectives) if replicate_objectives else ""
        ),
        "best_recovered_objective": best_objective if math.isfinite(best_objective) else "",
        "best_recovered_bitstring": best_bitstring,
    }


def next_power_of_two_at_least(value: int) -> int:
    v = max(1, int(value))
    return 1 << (v - 1).bit_length()


def build_settings(chi_obs: int) -> list[dict[str, Any]]:
    conv = next_power_of_two_at_least(max(1, chi_obs))
    restricted = max(2, int(math.floor(max(1, chi_obs) / 2)))
    settings = [
        {"label": "production_uncapped_1e-16", "bond_dimension": None, "threshold": 1e-16},
        {"label": f"restricted_cap_{restricted}_1e-16", "bond_dimension": restricted, "threshold": 1e-16},
        {"label": f"converged_cap_{conv}_1e-16", "bond_dimension": conv, "threshold": 1e-16},
        {"label": f"conservative_cap_{2 * conv}_1e-16", "bond_dimension": 2 * conv, "threshold": 1e-16},
        {"label": "threshold_uncapped_1e-12", "bond_dimension": None, "threshold": 1e-12},
        {"label": "threshold_uncapped_1e-8", "bond_dimension": None, "threshold": 1e-8},
    ]
    seen: set[tuple[int | None, float]] = set()
    deduped: list[dict[str, Any]] = []
    for setting in settings:
        key = (setting["bond_dimension"], float(setting["threshold"]))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(setting)
    return deduped


def write_digest(
    *,
    out_dir: Path,
    rows: list[dict[str, Any]],
    skipped: list[dict[str, Any]],
    environment: dict[str, Any],
) -> None:
    lines = [
        "# MDKP Layer A MPS Validation Digest",
        "",
        "## Scope",
        "",
        "- Fixed final-parameter validation for selected MDKP cases behind the negative simulator-hardware gap concern.",
        "- Large 60/99-qubit circuits use sampled MPS counts rather than full statevector probability snapshots.",
        "- The only changed variables across settings are MPS bond-dimension cap and truncation threshold.",
        "",
        "## Environment",
        "",
        f"- Python executable: `{environment.get('python_executable')}`",
        f"- Qiskit: `{environment.get('qiskit_version')}`",
        f"- Qiskit Aer: `{environment.get('qiskit_aer_version')}`",
        f"- CPU: `{environment.get('cpu_model_sysctl')}`",
        f"- Physical memory: `{environment.get('physical_memory_bytes_sysctl')}` bytes",
        f"- GPU used: `{environment.get('gpu_used')}`",
        "",
        "## Setting Summary",
        "",
        "| case | setting | chi max | TVD sampled vs production | mean gap % | sd gap % | best gap % | feasible run frac | raw feasible mass |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(v: Any) -> str:
            if v in ("", None):
                return ""
            if isinstance(v, float):
                return f"{v:.4g}"
            try:
                return f"{float(v):.4g}"
            except Exception:
                return str(v)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case_id", "")),
                    str(row.get("setting_label", "")),
                    fmt(row.get("max_observed_bond_dimension")),
                    fmt(row.get("tvd_vs_production_sampled")),
                    fmt(row.get("mean_gap_pct")),
                    fmt(row.get("sd_gap_pct")),
                    fmt(row.get("best_gap_pct")),
                    fmt(row.get("feasible_run_fraction")),
                    fmt(row.get("feasible_sample_mass_raw")),
                ]
            )
            + " |"
        )
    if skipped:
        lines.extend(["", "## Skipped Cases", ""])
        for item in skipped:
            lines.append(f"- `{item.get('case_id')}`: {item.get('reason')}")
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `mdkp_layer_a_summary.csv`: setting-level decoded-quality and distribution metrics.",
            "- `mdkp_layer_a_replicates.csv`: per-repetition decoded-quality records.",
            "- `skipped_cases.json`: cases that could not be exactly reconstructed from saved artifacts.",
            "- `environment.json`: simulator and classical-resource metadata.",
        ]
    )
    (out_dir / "digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="MDKP fixed-circuit MPS validation.")
    parser.add_argument(
        "--output-dir",
        default=str(EXPERIMENT_DIR / "mdkp_layer_a_results"),
    )
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--shots", type=int, default=0, help="Override shots per repetition; 0 uses source result shots.")
    parser.add_argument("--cases", nargs="*", default=["cvar_pet2", "vqe_hp1", "pce_hp1"])
    args = parser.parse_args()

    source_cases = {
        "cvar_pet2": REPO_ROOT / "research_benchmark/research_benchmark/results_hardware/mkp/mkp_cvar_vqe/pet2_dat/result.json",
        "vqe_hp1": REPO_ROOT / "research_benchmark/research_benchmark/results_hardware/mkp/mkp_vqe/hp1_dat/result.json",
        "pce_hp1": REPO_ROOT / "research_benchmark/research_benchmark/results_hardware/mkp/mkp_pce/hp1_dat/result.json",
    }
    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    summary_rows: list[dict[str, Any]] = []
    replicate_rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []

    for case_key in args.cases:
        try:
            case = load_case(source_cases[case_key], case_key)
            if int(args.shots) > 0:
                case.shots = int(args.shots)
            print(f"Preparing {case.case_id}: {case.method} {case.instance_name} shots={case.shots}")
            circuit, _qubo, objective, instance, metrics = bind_case_circuit(case)
        except Exception as exc:
            skipped.append({"case_id": case_key, "reason": str(exc)})
            print(f"Skipping {case_key}: {exc}")
            continue

        variable_names_in_order = [
            var_name
            for var_name, _ in sorted(
                objective.qubo.variables_index.items(),
                key=lambda item: int(item[1]),
            )
        ]

        production_diag = run_bond_diagnostic(
            circuit,
            threshold=1e-16,
            bond_dimension=None,
        )
        chi_obs = int(production_diag.get("max_observed_bond_dimension") or 1)
        settings = build_settings(chi_obs)
        diag_by_label = {"production_uncapped_1e-16": production_diag}
        production_aggregate: dict[str, int] | None = None
        production_top10: set[str] | None = None

        for setting in settings:
            print(f"  {case.case_id}: {setting['label']}")
            if setting["label"] in diag_by_label:
                bond_diag = diag_by_label[setting["label"]]
            else:
                bond_diag = run_bond_diagnostic(
                    circuit,
                    threshold=float(setting["threshold"]),
                    bond_dimension=setting["bond_dimension"],
                )
            counts_list: list[dict[str, int]] = []
            elapsed_list: list[float] = []
            for rep in range(int(args.repetitions)):
                seed = int(case.seed + 1009 * rep + 17)
                counts, metadata, elapsed = run_sampling(
                    circuit,
                    shots=int(case.shots),
                    seed=seed,
                    threshold=float(setting["threshold"]),
                    bond_dimension=setting["bond_dimension"],
                )
                counts_list.append(counts)
                elapsed_list.append(elapsed)
                bitstring, _energy = objective.best_sample(counts)
                recon = bitstring_reconstruction(
                    bitstring=bitstring,
                    objective=objective,
                    instance=instance,
                    variable_names_in_order=variable_names_in_order,
                )
                reconstructed = recon["reconstructed"]
                feasible = bool(reconstructed.get("feasible"))
                obj_val = float(reconstructed.get("objective_value", float("nan")))
                gap = (
                    max(0.0, (float(instance.optimal_value) - obj_val) / float(instance.optimal_value) * 100.0)
                    if feasible and float(instance.optimal_value)
                    else ""
                )
                replicate_rows.append(
                    {
                        "case_id": case.case_id,
                        "method": case.method,
                        "instance": case.instance_name,
                        "setting_label": setting["label"],
                        "replicate": rep,
                        "seed": seed,
                        "shots": case.shots,
                        "unique_bitstrings": len(counts),
                        "elapsed_sec": elapsed,
                        "best_objective": obj_val,
                        "best_gap_pct": gap,
                        "feasible": feasible,
                        "best_bitstring": recon["final_bitstring"],
                        "backend_method": metadata.get("method"),
                        "metadata_max_memory_mb": metadata.get("max_memory_mb"),
                    }
                )
            aggregate = aggregate_counts(counts_list)
            if setting["label"] == "production_uncapped_1e-16":
                production_aggregate = aggregate
                production_top10 = {
                    bitstring
                    for bitstring, _count in sorted(
                        aggregate.items(), key=lambda item: item[1], reverse=True
                    )[:10]
                }
            row = summarize_setting(
                case=case,
                setting=setting,
                counts_list=counts_list,
                sampling_elapsed=elapsed_list,
                objective=objective,
                instance=instance,
                variable_names_in_order=variable_names_in_order,
                production_aggregate=production_aggregate,
                production_top10=production_top10,
                bond_diag=bond_diag,
            )
            row.update(metrics)
            summary_rows.append(row)

    if summary_rows:
        summary_path = out_dir / "mdkp_layer_a_summary.csv"
        fieldnames = sorted({key for row in summary_rows for key in row})
        with summary_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(summary_rows)
    if replicate_rows:
        rep_path = out_dir / "mdkp_layer_a_replicates.csv"
        fieldnames = sorted({key for row in replicate_rows for key in row})
        with rep_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(replicate_rows)
    (out_dir / "skipped_cases.json").write_text(json.dumps(skipped, indent=2), encoding="utf-8")
    write_digest(out_dir=out_dir, rows=summary_rows, skipped=skipped, environment=environment)
    print(f"Wrote results to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
