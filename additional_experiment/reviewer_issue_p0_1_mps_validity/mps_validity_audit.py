#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import platform
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = Path(__file__).resolve().parents[2]
QOBENCH_SRC = REPO_ROOT / "research_benchmark" / "src"
if str(QOBENCH_SRC) not in sys.path:
    sys.path.insert(0, str(QOBENCH_SRC))


@dataclass(slots=True)
class Case:
    label: str
    result_json: Path | None
    problem: str
    method: str
    instance_name: str
    instance_path: Path | None
    layers: int
    entanglement: str
    seed: int
    best_theta: list[float] | None
    pce_compression_k: int
    pce_depth: int
    ws_epsilon: float


def _run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def collect_environment() -> dict[str, Any]:
    env_keys = [
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "VECLIB_MAXIMUM_THREADS",
        "NUMEXPR_NUM_THREADS",
        "QOBENCH_FORCE_MPS",
        "QOBENCH_LOCAL_SIMULATOR_METHOD",
        "QOBENCH_LOCAL_MAX_QUBITS",
    ]
    data: dict[str, Any] = {
        "python": sys.version,
        "python_executable": sys.executable,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_model_sysctl": _run_text(["sysctl", "-n", "machdep.cpu.brand_string"]),
        "physical_memory_bytes_sysctl": _run_text(["sysctl", "-n", "hw.memsize"]),
        "cpu_count": os.cpu_count(),
        "environment": {key: os.getenv(key) for key in env_keys if os.getenv(key) is not None},
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
            "method": "matrix_product_state",
            "device": getattr(sim.options, "device", None),
            "precision": getattr(sim.options, "precision", None),
            "enable_truncation": getattr(sim.options, "enable_truncation", None),
            "matrix_product_state_max_bond_dimension": getattr(
                sim.options, "matrix_product_state_max_bond_dimension", None
            ),
            "matrix_product_state_truncation_threshold": getattr(
                sim.options, "matrix_product_state_truncation_threshold", None
            ),
            "mps_log_data": getattr(sim.options, "mps_log_data", None),
            "mps_omp_threads": getattr(sim.options, "mps_omp_threads", None),
            "max_parallel_threads": getattr(sim.options, "max_parallel_threads", None),
            "max_memory_mb": getattr(sim.options, "max_memory_mb", None),
        }
    except Exception as exc:
        data["qiskit_import_error"] = repr(exc)
    return data


def _match_instance(problem: str, instance_name: str) -> Path | None:
    from qobench.problem_registry import get_problem
    from qobench.types import ProblemType

    problem_obj = get_problem(ProblemType(problem))
    for candidate in problem_obj.list_instances(REPO_ROOT, limit=None):
        if candidate.name == instance_name:
            return candidate.resolve()
        if candidate.name.replace(".", "_") == instance_name.replace(".", "_"):
            return candidate.resolve()
    default = problem_obj.default_instance(REPO_ROOT)
    return default.resolve() if default is not None and default.exists() else None


def load_cases(result_glob: str, max_cases: int) -> list[Case]:
    paths = sorted(Path(p) for p in glob.glob(result_glob, recursive=True))
    cases: list[Case] = []
    for path in paths:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        config = data.get("config", {})
        best = data.get("best_result", {})
        problem = str(config.get("problem") or data.get("problem") or "").strip()
        method = str(config.get("method") or data.get("execution", {}).get("method") or "").strip()
        if not problem or not method or method == "qrao":
            continue
        instance_name = str(data.get("instance_name") or Path(str(config.get("instance", ""))).name)
        if not instance_name:
            continue
        instance_path = _match_instance(problem, instance_name)
        if instance_path is None:
            continue
        best_theta = best.get("best_theta")
        if not isinstance(best_theta, list):
            best_theta = None
        cases.append(
            Case(
                label=f"{problem}/{method}/{instance_name}",
                result_json=path.resolve(),
                problem=problem,
                method=method,
                instance_name=instance_name,
                instance_path=instance_path,
                layers=int(config.get("layers", 3)),
                entanglement=str(config.get("entanglement", "circular")),
                seed=int(config.get("seed", 0)),
                best_theta=[float(x) for x in best_theta] if best_theta is not None else None,
                pce_compression_k=int(config.get("pce_compression_k", 2)),
                pce_depth=int(config.get("pce_depth", 0)),
                ws_epsilon=float(config.get("ws_epsilon", 1e-3)),
            )
        )
        if max_cases > 0 and len(cases) >= max_cases:
            break
    return cases


def build_case_circuit(case: Case) -> tuple[Any, Any | None, dict[str, Any]]:
    from qobench.problem_registry import get_problem
    from qobench.quantum_methods import build_algorithm_ansatz_bundle
    from qobench.types import ProblemType
    from qiskit_optimization.converters import QuadraticProgramToQubo
    from qiskit_optimization.translators import from_docplex_mp

    problem_obj = get_problem(ProblemType(case.problem))
    instance = problem_obj.load_instance(case.instance_path, seed=case.seed)
    model, _context = problem_obj.build_model(instance=instance, time_limit_sec=60.0)
    qp = from_docplex_mp(model)
    qubo = QuadraticProgramToQubo().convert(qp)
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
    if hasattr(circuit, "remove_final_measurements"):
        circuit = circuit.remove_final_measurements(inplace=False)
    params = sorted(list(circuit.parameters), key=lambda p: str(getattr(p, "name", p)))
    theta = case.best_theta
    if theta is None or len(theta) != len(params):
        theta = [0.123 + 0.017 * i for i in range(len(params))]
    name_to_value = {}
    original_params = sorted(ansatz.qiskit_parameters, key=lambda p: str(getattr(p, "name", p)))
    for idx, param in enumerate(original_params):
        if idx < len(theta):
            name_to_value[str(getattr(param, "name", param))] = float(theta[idx])
    bind_map = {
        param: name_to_value[str(getattr(param, "name", param))]
        for param in params
        if str(getattr(param, "name", param)) in name_to_value
    }
    bound = circuit.assign_parameters(bind_map)
    metrics = {
        "logical_qubo_variables": int(qubo.get_num_vars()),
        "execution_qubits": int(bound.num_qubits),
        "method": case.method,
        "depth": int(bound.depth() or 0),
        "size": int(bound.size() or 0),
        "num_parameters": len(params),
        "theta_source": "stored_best_theta" if case.best_theta is not None else "deterministic_placeholder",
    }
    return bound, qubo, metrics


def parse_bond_dimensions_from_log(raw: str | None) -> list[int]:
    if not raw:
        return []
    values: list[int] = []
    for match in re.finditer(r"BD=\[([^\]]*)\]", raw):
        for token in match.group(1).split():
            try:
                values.append(int(token))
            except Exception:
                pass
    return values


def probabilities_from_result_data(data: dict[str, Any]) -> dict[str, float]:
    raw = data.get("probs") or data.get("probabilities") or {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        if isinstance(key, int):
            bitstring = format(key, "b")
        else:
            bitstring = str(key).replace(" ", "")
        out[bitstring] = float(value)
    return out


def total_variation(p: dict[str, float], q: dict[str, float], num_qubits: int) -> float:
    keys = set(p) | set(q)
    total = 0.0
    for key in keys:
        k = key.zfill(num_qubits)
        total += abs(float(p.get(k, p.get(key, 0.0))) - float(q.get(k, q.get(key, 0.0))))
    return 0.5 * total


def distribution_expectation(probs: dict[str, float], qubo: Any, num_qubits: int) -> float | None:
    if int(qubo.get_num_vars()) != int(num_qubits):
        return None
    total = 0.0
    for bitstring, prob in probs.items():
        bits = bitstring.zfill(num_qubits)[-num_qubits:]
        vector = [1 if ch == "1" else 0 for ch in bits]
        total += float(prob) * float(qubo.objective.evaluate(vector))
    return total


def run_probability_snapshot(
    circuit: Any,
    *,
    method: str,
    threshold: float | None = None,
    bond_dimension: int | None = None,
) -> dict[str, Any]:
    from qiskit_aer import AerSimulator

    qc = circuit.copy()
    qc.save_probabilities_dict(label="probs")
    if method == "matrix_product_state":
        qc.save_matrix_product_state(label="mps")
    options: dict[str, Any] = {"method": method}
    if method == "matrix_product_state":
        options["mps_log_data"] = True
        if threshold is not None:
            options["matrix_product_state_truncation_threshold"] = float(threshold)
        if bond_dimension is not None:
            options["matrix_product_state_max_bond_dimension"] = int(bond_dimension)
    backend = AerSimulator(**options)
    start = time.perf_counter()
    result = backend.run(qc, shots=1).result()
    elapsed = time.perf_counter() - start
    metadata = dict(result.results[0].metadata)
    data = result.data(0)
    probs = probabilities_from_result_data(data)
    saved_bond_dims: list[int] = []
    if "mps" in data:
        try:
            _gammas, lambdas = data["mps"]
            saved_bond_dims = [int(len(values)) for values in lambdas]
        except Exception:
            saved_bond_dims = []
    log_bond_dims = parse_bond_dimensions_from_log(metadata.get("MPS_log_data"))
    return {
        "success": bool(result.success),
        "elapsed_sec": elapsed,
        "metadata": metadata,
        "probabilities": probs,
        "saved_bond_dimensions": saved_bond_dims,
        "log_bond_dimensions": log_bond_dims,
    }


def parse_bond_dimensions(raw: str) -> list[int | None]:
    out: list[int | None] = []
    for token in raw.split(","):
        token = token.strip().lower()
        if token in {"", "none", "uncapped", "unlimited"}:
            out.append(None)
        else:
            out.append(int(token))
    return out


def parse_thresholds(raw: str) -> list[float]:
    return [float(token.strip()) for token in raw.split(",") if token.strip()]


def write_summary(rows: list[dict[str, Any]], environment: dict[str, Any], out_path: Path) -> None:
    lines = [
        "# MPS Validity Audit Summary",
        "",
        f"- Python: `{environment.get('python_executable')}`",
        f"- Qiskit: `{environment.get('qiskit_version')}`",
        f"- Qiskit Aer: `{environment.get('qiskit_aer_version')}`",
        f"- CPU: `{environment.get('cpu_model_sysctl') or environment.get('processor')}`",
        f"- Platform: `{environment.get('platform')}`",
        f"- GPU used: `{environment.get('gpu_used')}`",
        "",
        "| case | qubits | method | threshold | bond cap | max observed bond | TVD vs exact | TVD vs uncapped | expectation diff vs exact | runtime sec |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        def fmt(value: Any) -> str:
            if value is None or value == "":
                return ""
            if isinstance(value, float):
                return f"{value:.3g}"
            return str(value)

        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("case", "")),
                    fmt(row.get("execution_qubits")),
                    str(row.get("simulator_method", "")),
                    fmt(row.get("threshold")),
                    fmt(row.get("bond_dimension_cap")),
                    fmt(row.get("max_observed_bond_dimension")),
                    fmt(row.get("tvd_vs_exact")),
                    fmt(row.get("tvd_vs_uncapped_mps")),
                    fmt(row.get("expectation_diff_vs_exact")),
                    fmt(row.get("elapsed_sec")),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Notes",
            "",
            "- `bond cap` is empty for uncapped Aer MPS.",
            "- `max observed bond` is extracted from `save_matrix_product_state` and Aer MPS log metadata.",
            "- Aer records the truncation threshold and bond dimensions, but this API path does not expose a full discarded-weight table. Use the exact and bond-sweep deltas as the empirical truncation-error diagnostic.",
        ]
    )
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Qiskit Aer MPS validity for benchmark circuits.")
    parser.add_argument(
        "--result-glob",
        default="research_benchmark/research_benchmark/results_simulator/**/result.json",
        help="Glob of existing result.json files whose stored best_theta values should be audited.",
    )
    parser.add_argument("--max-cases", type=int, default=7)
    parser.add_argument("--exact-max-qubits", type=int, default=24)
    parser.add_argument("--bond-dimensions", default="uncapped,16,32,64,128")
    parser.add_argument("--thresholds", default="1e-16")
    parser.add_argument("--output-dir", default=str(EXPERIMENT_DIR / "results"))
    parser.add_argument(
        "--shots",
        type=int,
        default=0,
        help="Reserved for future sampled-count checks. Probability snapshots are used when this is 0.",
    )
    args = parser.parse_args()

    if args.shots:
        print("This audit currently reports simulator probability snapshots; --shots is recorded but not sampled.")

    out_dir = Path(args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    environment = collect_environment()
    (out_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")

    cases = load_cases(args.result_glob, args.max_cases)
    if not cases:
        raise SystemExit("No auditable cases found. Check --result-glob or run simulator benchmarks first.")

    rows: list[dict[str, Any]] = []
    thresholds = parse_thresholds(args.thresholds)
    bond_dimensions = parse_bond_dimensions(args.bond_dimensions)
    for case in cases:
        print(f"Auditing {case.label}")
        circuit, qubo, metrics = build_case_circuit(case)
        exact = None
        exact_exp = None
        if int(circuit.num_qubits) <= int(args.exact_max_qubits):
            exact = run_probability_snapshot(circuit, method="statevector")
            exact_exp = distribution_expectation(exact["probabilities"], qubo, int(circuit.num_qubits))
            rows.append(
                {
                    "case": case.label,
                    "result_json": str(case.result_json) if case.result_json else "",
                    **metrics,
                    "simulator_method": "statevector",
                    "threshold": "",
                    "bond_dimension_cap": "",
                    "max_observed_bond_dimension": "",
                    "elapsed_sec": exact["elapsed_sec"],
                    "tvd_vs_exact": 0.0,
                    "tvd_vs_uncapped_mps": "",
                    "expectation": exact_exp,
                    "expectation_diff_vs_exact": 0.0 if exact_exp is not None else "",
                    "required_memory_mb": exact["metadata"].get("required_memory_mb"),
                    "max_memory_mb": exact["metadata"].get("max_memory_mb"),
                }
            )

        uncapped_by_threshold: dict[float, dict[str, Any]] = {}
        for threshold in thresholds:
            for bond_dimension in bond_dimensions:
                mps = run_probability_snapshot(
                    circuit,
                    method="matrix_product_state",
                    threshold=threshold,
                    bond_dimension=bond_dimension,
                )
                if bond_dimension is None:
                    uncapped_by_threshold[threshold] = mps
                ref_mps = uncapped_by_threshold.get(threshold)
                observed_bonds = list(mps["saved_bond_dimensions"]) + list(mps["log_bond_dimensions"])
                max_observed = max(observed_bonds) if observed_bonds else None
                mps_exp = distribution_expectation(mps["probabilities"], qubo, int(circuit.num_qubits))
                tvd_exact = (
                    total_variation(exact["probabilities"], mps["probabilities"], int(circuit.num_qubits))
                    if exact is not None
                    else None
                )
                tvd_uncapped = (
                    total_variation(ref_mps["probabilities"], mps["probabilities"], int(circuit.num_qubits))
                    if ref_mps is not None
                    else None
                )
                rows.append(
                    {
                        "case": case.label,
                        "result_json": str(case.result_json) if case.result_json else "",
                        **metrics,
                        "simulator_method": "matrix_product_state",
                        "threshold": threshold,
                        "bond_dimension_cap": "" if bond_dimension is None else bond_dimension,
                        "max_observed_bond_dimension": max_observed,
                        "elapsed_sec": mps["elapsed_sec"],
                        "tvd_vs_exact": tvd_exact,
                        "tvd_vs_uncapped_mps": tvd_uncapped,
                        "expectation": mps_exp,
                        "expectation_diff_vs_exact": (
                            abs(mps_exp - exact_exp)
                            if mps_exp is not None and exact_exp is not None
                            else ""
                        ),
                        "required_memory_mb": mps["metadata"].get("required_memory_mb"),
                        "max_memory_mb": mps["metadata"].get("max_memory_mb"),
                        "aer_mps_threshold_metadata": mps["metadata"].get(
                            "matrix_product_state_truncation_threshold"
                        ),
                        "aer_mps_max_bond_metadata": mps["metadata"].get(
                            "matrix_product_state_max_bond_dimension"
                        ),
                    }
                )

    csv_path = out_dir / "mps_validation.csv"
    fieldnames = sorted({key for row in rows for key in row.keys()})
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    write_summary(rows, environment, out_dir / "summary.md")
    print(f"Wrote {csv_path}")
    print(f"Wrote {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
