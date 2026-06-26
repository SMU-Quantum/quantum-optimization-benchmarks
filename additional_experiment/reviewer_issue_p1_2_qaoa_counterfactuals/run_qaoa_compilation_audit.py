from __future__ import annotations

import csv
import contextlib
import io
import json
import math
import statistics
import sys
import warnings
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from qiskit import transpile
from qiskit_ibm_runtime.fake_provider import FakeFez, FakeNighthawk

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
SRC = ROOT / "research_benchmark" / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from qobench.problem_registry import get_problem  # noqa: E402
from qobench.qubo import convert_docplex_to_qubo  # noqa: E402
from qobench.quantum_methods import build_algorithm_ansatz_bundle  # noqa: E402
from qobench.types import ProblemType  # noqa: E402

SEED_TRANSPILER = 123
COUNTERFACTUAL_OPT_LEVEL = 1

COUNTERFACTUAL_COLUMNS = [
    "problem",
    "instance",
    "method",
    "logical_qubits",
    "qaoa_depth",
    "parameter_count",
    "compilation_condition",
    "backend_topology",
    "backend_name_or_surrogate",
    "optimization_level",
    "layout_strategy",
    "routing_strategy",
    "use_fractional_gates",
    "mitigation_compatibility_note",
    "logical_two_qubit_gates",
    "transpiled_two_qubit_gates",
    "swap_count",
    "two_qubit_depth",
    "total_depth",
    "estimated_fidelity_proxy",
    "below_fidelity_0_1",
    "below_fidelity_0_01",
    "below_fidelity_1e_3",
    "notes",
]

SUMMARY_COLUMNS = [
    "problem",
    "method",
    "n_representative_circuits",
    "historical_median_two_qubit_gates",
    "swap_aware_heron_median_two_qubit_gates",
    "fractional_gate_heron_median_two_qubit_gates",
    "swap_aware_nighthawk_median_two_qubit_gates",
    "swap_aware_heron_median_reduction_percent",
    "fractional_gate_heron_median_reduction_percent",
    "swap_aware_nighthawk_median_reduction_percent",
    "historical_median_fidelity_proxy",
    "swap_aware_heron_median_fidelity_proxy",
    "fractional_gate_heron_median_gate_proxy",
    "swap_aware_nighthawk_median_fidelity_proxy",
    "n_circuits_crossing_0_1_under_any_counterfactual",
    "n_circuits_crossing_1e_3_under_any_counterfactual",
    "interpretation",
]


@dataclass(frozen=True)
class AuditCircuit:
    problem: str
    problem_type: ProblemType
    instance: str
    source_instance: str
    method: str
    result_path: Path
    load_path: Path | None
    load_kwargs: dict[str, Any]


def fmt(value: Any, digits: int = 6) -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return "not_available"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def median(values: list[float]) -> float | None:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return None
    return float(statistics.median(finite))


def fidelity_proxy(one_q: float, two_q: float, meas: float) -> float:
    exposure = 0.001 * float(one_q) + 0.01 * float(two_q) + 0.02 * float(meas)
    return float(math.exp(-exposure))


def method_label(method: str) -> str:
    return {"qaoa": "QAOA", "ma_qaoa": "MA-QAOA", "ws_qaoa": "WS-QAOA"}[method]


def short_label(problem: str, method: str) -> str:
    suffix = {"qaoa": "QAOA", "ma_qaoa": "MA", "ws_qaoa": "WS"}[method]
    return f"{problem}-{suffix}"


def selected_circuits() -> list[AuditCircuit]:
    selected: list[AuditCircuit] = []
    methods = ["qaoa", "ma_qaoa", "ws_qaoa"]
    specs = [
        (
            "MDKP",
            ProblemType.MKP,
            "hp1",
            "hp1.dat",
            ROOT / "Multi_Dimension_Knapsack/MKP_Instances/hpp/hp1.dat",
            {},
            "mkp",
            "hp1_dat",
        ),
        (
            "MIS",
            ProblemType.MIS,
            "1tc.32",
            "1tc.32.txt",
            ROOT / "Maximum_Independent_Set/mis_benchmark_instances/1tc.32.txt",
            {},
            "mis",
            "1tc_32_txt",
        ),
        (
            "QAP",
            ProblemType.QAP,
            "tai10a",
            "tai10a.dat",
            ROOT / "Quadratic_Assignment_Problem/qapdata/tai10a.dat",
            {},
            "qap",
            "tai10a_dat",
        ),
        (
            "MSP",
            ProblemType.MARKET_SHARE,
            "ms20",
            "ms_seed0_prod3.gen",
            None,
            {"seed": 0, "num_products": 3},
            "market_share",
            "ms_seed0_prod3_gen",
        )
    ]
    for problem, problem_type, instance, source, load_path, kwargs, folder, result_dir in specs:
        for method in methods:
            selected.append(
                AuditCircuit(
                    problem=problem,
                    problem_type=problem_type,
                    instance=instance,
                    source_instance=source,
                    method=method,
                    result_path=ROOT
                    / "research_benchmark/research_benchmark/results_hardware"
                    / folder
                    / f"{folder}_{method}"
                    / result_dir
                    / "result.json",
                    load_path=load_path,
                    load_kwargs=kwargs,
                )
            )
    return selected


def count_ops_by_arity(circuit: Any) -> tuple[int, int, int, int]:
    one_q = two_q = meas = swaps = 0
    for item in circuit.data:
        op = item.operation
        qargs = item.qubits
        name = str(op.name)
        if name == "measure":
            meas += 1
        elif name == "swap":
            swaps += 1
            two_q += 1
        elif len(qargs) == 1:
            one_q += 1
        elif len(qargs) == 2:
            two_q += 1
    return one_q, two_q, meas, swaps


def two_qubit_depth(circuit: Any) -> int:
    try:
        return int(circuit.depth(filter_function=lambda inst: len(inst.qubits) == 2))
    except Exception:
        layers = [0] * int(circuit.num_qubits)
        for item in circuit.data:
            qargs = list(item.qubits)
            if len(qargs) != 2:
                continue
            indices = [circuit.find_bit(q).index for q in qargs]
            layer = max(layers[idx] for idx in indices) + 1
            for idx in indices:
                layers[idx] = layer
        return max(layers) if layers else 0


def representative_backend_name(metadata: list[dict[str, Any]]) -> str:
    names = [str(entry.get("backend_name")) for entry in metadata if entry.get("backend_name")]
    if not names:
        return "unknown_historical_backend"
    return Counter(names).most_common(1)[0][0]


def historical_row(item: AuditCircuit, data: dict[str, Any]) -> dict[str, str]:
    metrics = data.get("circuit_metrics", {})
    metadata = data.get("job_metadata") or []
    if isinstance(metadata, dict):
        metadata = [metadata]
    one_q = median([float(entry.get("transpiled_1q_gates", math.nan)) for entry in metadata])
    two_q = median([float(entry.get("transpiled_2q_gates", math.nan)) for entry in metadata])
    meas = median([float(entry.get("transpiled_measurements", math.nan)) for entry in metadata])
    depth = median([float(entry.get("transpiled_depth", math.nan)) for entry in metadata])
    one_q = one_q if one_q is not None else 0.0
    two_q = two_q if two_q is not None else 0.0
    meas = meas if meas is not None else float(metrics.get("measurement_count", 0) or 0)
    proxy = fidelity_proxy(one_q, two_q, meas)
    logical_two_q = int(metrics.get("two_qubit_gates_pretranspile", 0) or 0)
    swap_estimate = max(0, round((two_q - 2.0 * logical_two_q) / 3.0))
    return {
        "problem": item.problem,
        "instance": item.instance,
        "method": method_label(item.method),
        "logical_qubits": fmt(metrics.get("logical_qubits_before_encoding") or metrics.get("num_qubits")),
        "qaoa_depth": fmt(metrics.get("ansatz_reps")),
        "parameter_count": fmt(metrics.get("trainable_parameters")),
        "compilation_condition": "historical_baseline",
        "backend_topology": "historical_heavy_hex",
        "backend_name_or_surrogate": representative_backend_name(metadata),
        "optimization_level": fmt(data.get("transpiler_settings", {}).get("qiskit_optimization_level")),
        "layout_strategy": "preset_passmanager_default",
        "routing_strategy": "preset_passmanager_default",
        "use_fractional_gates": "false",
        "mitigation_compatibility_note": "original hardware workflow; historical artifact row",
        "logical_two_qubit_gates": fmt(logical_two_q),
        "transpiled_two_qubit_gates": fmt(two_q),
        "swap_count": fmt(swap_estimate),
        "two_qubit_depth": "not_recorded_in_historical_artifact",
        "total_depth": fmt(depth),
        "estimated_fidelity_proxy": fmt(proxy),
        "below_fidelity_0_1": fmt(proxy < 0.1),
        "below_fidelity_0_01": fmt(proxy < 0.01),
        "below_fidelity_1e_3": fmt(proxy < 1e-3),
        "notes": (
            "Historical compilation metrics read from saved job metadata; "
            "historical_baseline_reconstructed because backend calibrations are not replayed."
        ),
    }


def build_bound_circuit(item: AuditCircuit, data: dict[str, Any]) -> tuple[Any, Any]:
    problem = get_problem(item.problem_type)
    instance = problem.load_instance(item.load_path, **item.load_kwargs)
    model, _ = problem.build_model(instance)
    qp, qubo = convert_docplex_to_qubo(model)
    layers = int(data.get("config", {}).get("layers", 3))
    # WS-QAOA may attempt a CPLEX relaxation and then fall back to the repo's
    # heuristic warm start for non-convex QPs. Keep that expected fallback from
    # polluting the audit logs.
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        ansatz = build_algorithm_ansatz_bundle(
            method=item.method,
            qubo=qubo,
            layers=layers,
            entanglement=str(data.get("config", {}).get("entanglement", "circular")),
            qp=qp,
            ws_epsilon=float(data.get("config", {}).get("ws_epsilon", 1e-3)),
        )
    theta = list(data.get("best_result", {}).get("best_theta", []))
    if len(theta) != len(ansatz.qiskit_parameters):
        raise ValueError(
            f"{item.problem}/{item.instance}/{item.method}: theta length {len(theta)} "
            f"does not match ansatz parameters {len(ansatz.qiskit_parameters)}"
        )
    bound = ansatz.qiskit_template.assign_parameters(
        {param: float(theta[idx]) for idx, param in enumerate(ansatz.qiskit_parameters)},
        inplace=False,
    )
    return bound, ansatz


def compile_condition(circuit: Any, condition: str) -> tuple[Any, dict[str, str]]:
    if condition == "swap_aware_heron":
        backend = FakeFez()
        compiled = transpile(
            circuit,
            backend=backend,
            optimization_level=COUNTERFACTUAL_OPT_LEVEL,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=SEED_TRANSPILER,
        )
        meta = {
            "backend_topology": "heron_heavy_hex_surrogate",
            "backend_name_or_surrogate": "fake_fez",
            "layout_strategy": "sabre_connectivity_aware_initial_layout",
            "routing_strategy": "sabre_swap_routing",
            "use_fractional_gates": "false",
            "mitigation_compatibility_note": "compatible as a compilation-only resource estimate",
            "notes": (
                "Qubit selection uses the fake_fez heavy-hex target; Qiskit SABRE layout/routing "
                "is a SWAP-aware heuristic, not an optimal or SAT-proved route; logical cost "
                "operator and final bound parameters are unchanged."
            ),
        }
        return compiled, meta

    if condition == "fractional_gate_heron":
        backend = FakeFez()
        compiled = transpile(
            circuit,
            basis_gates=["rz", "rx", "rzz", "x", "sx", "measure"],
            coupling_map=backend.coupling_map,
            optimization_level=COUNTERFACTUAL_OPT_LEVEL,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=SEED_TRANSPILER,
        )
        meta = {
            "backend_topology": "heron_heavy_hex_fractional_gate_surrogate",
            "backend_name_or_surrogate": "fake_fez_coupling_map_fractional_basis",
            "layout_strategy": "sabre_connectivity_aware_initial_layout",
            "routing_strategy": "sabre_swap_routing",
            "use_fractional_gates": "true",
            "mitigation_compatibility_note": (
                "Fractional-gate compilation is a circuit-resource counterfactual only; "
                "not executed under the original ZNE/probabilistic-error-amplification workflow."
            ),
            "notes": (
                "Bound circuit compiled with an RZZ-preserving basis on the Heron surrogate "
                "coupling map; fidelity column is a gate-count proxy \\u007eF_gate, not a "
                "calibrated fractional-gate fidelity estimate."
            ),
        }
        return compiled, meta

    if condition == "swap_aware_nighthawk":
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            backend = FakeNighthawk()
        compiled = transpile(
            circuit,
            backend=backend,
            optimization_level=COUNTERFACTUAL_OPT_LEVEL,
            layout_method="sabre",
            routing_method="sabre",
            seed_transpiler=SEED_TRANSPILER,
        )
        meta = {
            "backend_topology": "nighthawk_square_lattice_surrogate",
            "backend_name_or_surrogate": "fake_nighthawk",
            "layout_strategy": "sabre_connectivity_aware_initial_layout",
            "routing_strategy": "sabre_swap_routing",
            "use_fractional_gates": "false",
            "mitigation_compatibility_note": "topology-only compilation; no hardware execution used",
            "notes": (
                "Topology-only Nighthawk fake-backend compilation; no Nighthawk calibration data "
                "or hardware execution used; logical cost operator and final bound parameters "
                "are unchanged."
            ),
        }
        return compiled, meta

    raise ValueError(f"Unknown condition: {condition}")


def counterfactual_row(
    item: AuditCircuit,
    data: dict[str, Any],
    circuit: Any,
    ansatz: Any,
    condition: str,
) -> dict[str, str]:
    compiled, meta = compile_condition(circuit, condition)
    one_q, two_q, meas, explicit_swaps = count_ops_by_arity(compiled)
    logical_two_q = int(data.get("circuit_metrics", {}).get("two_qubit_gates_pretranspile", 0) or 0)
    fractional = condition == "fractional_gate_heron"
    expected_nonrouting_twoq = logical_two_q if fractional else 2 * logical_two_q
    swap_estimate = max(explicit_swaps, round(max(0, two_q - expected_nonrouting_twoq) / 3.0))
    proxy = fidelity_proxy(one_q, two_q, meas)
    return {
        "problem": item.problem,
        "instance": item.instance,
        "method": method_label(item.method),
        "logical_qubits": fmt(ansatz.num_qubits),
        "qaoa_depth": fmt(ansatz.ansatz_reps),
        "parameter_count": fmt(ansatz.num_parameters),
        "compilation_condition": condition,
        "backend_topology": meta["backend_topology"],
        "backend_name_or_surrogate": meta["backend_name_or_surrogate"],
        "optimization_level": fmt(COUNTERFACTUAL_OPT_LEVEL),
        "layout_strategy": meta["layout_strategy"],
        "routing_strategy": meta["routing_strategy"],
        "use_fractional_gates": meta["use_fractional_gates"],
        "mitigation_compatibility_note": meta["mitigation_compatibility_note"],
        "logical_two_qubit_gates": fmt(logical_two_q),
        "transpiled_two_qubit_gates": fmt(two_q),
        "swap_count": fmt(swap_estimate),
        "two_qubit_depth": fmt(two_qubit_depth(compiled)),
        "total_depth": fmt(compiled.depth()),
        "estimated_fidelity_proxy": fmt(proxy),
        "below_fidelity_0_1": fmt(proxy < 0.1),
        "below_fidelity_0_01": fmt(proxy < 0.01),
        "below_fidelity_1e_3": fmt(proxy < 1e-3),
        "notes": meta["notes"],
    }


def reduction_percent(historical: float, counterfactual: float) -> float | None:
    if historical <= 0 or not math.isfinite(historical) or not math.isfinite(counterfactual):
        return None
    return 100.0 * (historical - counterfactual) / historical


def interpretation_for(group: list[dict[str, str]]) -> str:
    by_condition = {row["compilation_condition"]: row for row in group}
    hist = float(by_condition["historical_baseline"]["transpiled_two_qubit_gates"])
    best_condition = min(
        [
            row
            for row in group
            if row["compilation_condition"] != "historical_baseline"
        ],
        key=lambda row: float(row["transpiled_two_qubit_gates"]),
    )
    best_gates = float(best_condition["transpiled_two_qubit_gates"])
    best_proxy = max(
        float(row["estimated_fidelity_proxy"])
        for row in group
        if row["compilation_condition"] != "historical_baseline"
    )
    reduction = reduction_percent(hist, best_gates)
    if best_proxy >= 1e-3:
        return (
            "Counterfactual compilation moves at least one representative circuit above "
            "the 1e-3 gate-count proxy threshold; broad QAOA claims should be softened."
        )
    if reduction is not None and reduction > 0:
        return (
            "Counterfactual compilation reduces two-qubit gates but remains below the "
            "1e-3 proxy threshold for this representative circuit."
        )
    return (
        "Counterfactual compilation does not materially improve this representative "
        "circuit under the gate-count proxy."
    )


def build_summary(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    by_circuit: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[(row["problem"], row["method"])].append(row)
        by_circuit[(row["problem"], row["instance"], row["method"])].append(row)

    summary_rows: list[dict[str, str]] = []
    for problem in ["MDKP", "MIS", "QAP", "MSP"]:
        for method in ["QAOA", "MA-QAOA", "WS-QAOA"]:
            group = grouped[(problem, method)]
            if not group:
                continue
            by_condition: dict[str, list[float]] = defaultdict(list)
            fidelity_by_condition: dict[str, list[float]] = defaultdict(list)
            for row in group:
                by_condition[row["compilation_condition"]].append(
                    float(row["transpiled_two_qubit_gates"])
                )
                fidelity_by_condition[row["compilation_condition"]].append(
                    float(row["estimated_fidelity_proxy"])
                )
            hist_med = statistics.median(by_condition["historical_baseline"])
            c1_med = statistics.median(by_condition["swap_aware_heron"])
            c2_med = statistics.median(by_condition["fractional_gate_heron"])
            c3_med = statistics.median(by_condition["swap_aware_nighthawk"])
            crossings_0_1 = 0
            crossings_1e_3 = 0
            interpretations: list[str] = []
            for key, circuit_rows in by_circuit.items():
                if key[0] != problem or key[2] != method:
                    continue
                counterfactuals = [
                    row for row in circuit_rows if row["compilation_condition"] != "historical_baseline"
                ]
                if any(float(row["estimated_fidelity_proxy"]) >= 0.1 for row in counterfactuals):
                    crossings_0_1 += 1
                if any(float(row["estimated_fidelity_proxy"]) >= 1e-3 for row in counterfactuals):
                    crossings_1e_3 += 1
                interpretations.append(interpretation_for(circuit_rows))
            summary_rows.append(
                {
                    "problem": problem,
                    "method": method,
                    "n_representative_circuits": "1",
                    "historical_median_two_qubit_gates": fmt(hist_med),
                    "swap_aware_heron_median_two_qubit_gates": fmt(c1_med),
                    "fractional_gate_heron_median_two_qubit_gates": fmt(c2_med),
                    "swap_aware_nighthawk_median_two_qubit_gates": fmt(c3_med),
                    "swap_aware_heron_median_reduction_percent": fmt(
                        reduction_percent(hist_med, c1_med)
                    ),
                    "fractional_gate_heron_median_reduction_percent": fmt(
                        reduction_percent(hist_med, c2_med)
                    ),
                    "swap_aware_nighthawk_median_reduction_percent": fmt(
                        reduction_percent(hist_med, c3_med)
                    ),
                    "historical_median_fidelity_proxy": fmt(
                        statistics.median(fidelity_by_condition["historical_baseline"])
                    ),
                    "swap_aware_heron_median_fidelity_proxy": fmt(
                        statistics.median(fidelity_by_condition["swap_aware_heron"])
                    ),
                    "fractional_gate_heron_median_gate_proxy": fmt(
                        statistics.median(fidelity_by_condition["fractional_gate_heron"])
                    ),
                    "swap_aware_nighthawk_median_fidelity_proxy": fmt(
                        statistics.median(fidelity_by_condition["swap_aware_nighthawk"])
                    ),
                    "n_circuits_crossing_0_1_under_any_counterfactual": fmt(crossings_0_1),
                    "n_circuits_crossing_1e_3_under_any_counterfactual": fmt(crossings_1e_3),
                    "interpretation": interpretations[0] if interpretations else "not_available",
                }
            )
    return summary_rows


def write_csv(path: Path, columns: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def make_figure(rows: list[dict[str, str]]) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    condition_order = [
        "historical_baseline",
        "swap_aware_heron",
        "fractional_gate_heron",
        "swap_aware_nighthawk",
    ]
    condition_labels = {
        "historical_baseline": "Historical",
        "swap_aware_heron": "SWAP-aware Heron",
        "fractional_gate_heron": "Fractional Heron",
        "swap_aware_nighthawk": "SWAP-aware Nighthawk",
    }
    colors = {
        "historical_baseline": "#4C78A8",
        "swap_aware_heron": "#F58518",
        "fractional_gate_heron": "#54A24B",
        "swap_aware_nighthawk": "#B279A2",
    }
    hatches = {
        "historical_baseline": "",
        "swap_aware_heron": "//",
        "fractional_gate_heron": "..",
        "swap_aware_nighthawk": "\\\\",
    }
    labels = []
    grouped: dict[str, dict[str, dict[str, str]]] = {}
    for item in selected_circuits():
        label = short_label(item.problem, item.method)
        labels.append(label)
        grouped[label] = {}
    for row in rows:
        label = short_label(row["problem"], row["method"].lower().replace("-", "_"))
        grouped[label][row["compilation_condition"]] = row

    x = np.arange(len(labels))
    width = 0.18
    fig, axes = plt.subplots(
        2,
        1,
        figsize=(11.2, 7.0),
        sharex=True,
        constrained_layout=True,
    )
    for idx, condition in enumerate(condition_order):
        offsets = x + (idx - 1.5) * width
        gate_values = [
            float(grouped[label][condition]["transpiled_two_qubit_gates"]) for label in labels
        ]
        proxy_values = [
            max(float(grouped[label][condition]["estimated_fidelity_proxy"]), 1e-300)
            for label in labels
        ]
        for ax, values in [(axes[0], gate_values), (axes[1], proxy_values)]:
            ax.bar(
                offsets,
                values,
                width=width,
                label=condition_labels[condition],
                color=colors[condition],
                hatch=hatches[condition],
                edgecolor="#333333",
                linewidth=0.35,
                alpha=0.92,
            )

    for ax in axes:
        ax.set_axisbelow(True)
        ax.grid(axis="y", which="major", color="#d0d0d0", linewidth=0.55, alpha=0.65)
        ax.grid(axis="y", which="minor", color="#e8e8e8", linewidth=0.35, alpha=0.45)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        for boundary in [2.5, 5.5, 8.5]:
            ax.axvline(boundary, color="#9a9a9a", linewidth=0.7, alpha=0.55)

    axes[0].set_ylabel("Transpiled two-qubit gates")
    axes[0].set_yscale("log")
    axes[0].set_title("(a) Post-transpilation two-qubit gate count", loc="left", fontweight="bold")
    axes[0].legend(
        ncols=4,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.22),
        frameon=False,
        columnspacing=1.2,
        handlelength=1.8,
    )

    axes[1].set_ylabel("Gate-count fidelity proxy")
    axes[1].set_yscale("log")
    axes[1].set_title("(b) Gate-count fidelity proxy", loc="left", fontweight="bold")
    axes[1].set_ylim(1e-305, 2.0)
    for threshold in [1.0, 0.01, 1e-3]:
        axes[1].axhline(threshold, color="#222222", linestyle="--", linewidth=0.75, alpha=0.65)
    axes[1].text(
        0.985,
        0.94,
        "dashed thresholds:\n1, 0.01, 1e-3",
        transform=axes[1].transAxes,
        ha="right",
        va="top",
        fontsize=7,
        color="#333333",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=28, ha="right", rotation_mode="anchor")
    axes[1].set_xlabel("Representative bound QAOA-family circuit")
    fig.savefig(OUT / "fig_qaoa_compilation_counterfactuals.pdf")
    plt.close(fig)


def write_digest(rows: list[dict[str, str]], summary_rows: list[dict[str, str]]) -> None:
    counterfactual_rows = [
        row for row in rows if row["compilation_condition"] != "historical_baseline"
    ]
    by_condition = defaultdict(list)
    for row in rows:
        by_condition[row["compilation_condition"]].append(row)

    def median_reduction(condition: str) -> float:
        values = []
        for hist in by_condition["historical_baseline"]:
            match = next(
                row
                for row in by_condition[condition]
                if row["problem"] == hist["problem"]
                and row["instance"] == hist["instance"]
                and row["method"] == hist["method"]
            )
            values.append(
                reduction_percent(
                    float(hist["transpiled_two_qubit_gates"]),
                    float(match["transpiled_two_qubit_gates"]),
                )
            )
        return float(statistics.median([v for v in values if v is not None]))

    crossings_0_01 = [
        f"{row['problem']}-{row['method']}"
        for row in counterfactual_rows
        if float(row["estimated_fidelity_proxy"]) >= 0.01
    ]
    crossings_1e_3 = [
        f"{row['problem']}-{row['method']}"
        for row in counterfactual_rows
        if float(row["estimated_fidelity_proxy"]) >= 1e-3
    ]

    lines = [
        "# QAOA Compilation Counterfactual Digest",
        "",
        "Compilation-only audit over 12 representative bound QAOA-family circuits: "
        "QAOA, MA-QAOA, and WS-QAOA for MDKP hp1, MIS 1tc.32, QAP tai10a, "
        "and MSP ms20 (mapped to generated ms_seed0_prod3).",
        "",
        f"1. SWAP-aware Heron mapping median two-qubit-gate reduction: {fmt(median_reduction('swap_aware_heron'))}%.",
        f"2. Fractional-gate Heron median two-qubit-gate reduction: {fmt(median_reduction('fractional_gate_heron'))}%.",
        f"3. SWAP-aware Nighthawk median two-qubit-gate reduction: {fmt(median_reduction('swap_aware_nighthawk'))}%.",
        "4. Circuits crossing F_est >= 0.01 under any counterfactual: "
        + (", ".join(sorted(set(crossings_0_01))) if crossings_0_01 else "none")
        + ".",
        "5. Circuits crossing F_est >= 1e-3 under any counterfactual: "
        + (", ".join(sorted(set(crossings_1e_3))) if crossings_1e_3 else "none")
        + ".",
        "6. The original broad QAOA conclusion should be softened: these results support "
        "a claim about the tested standard, MA-QAOA, and WS-QAOA implementations under the "
        "historical compilation/hardware settings, not QAOA in general.",
        "7. Fractional-gate rows are compilation-resource counterfactuals only and are not "
        "directly comparable to the original resilience/ZNE configuration.",
        "",
        "The saved historical artifacts omit calibrated per-gate errors, so all fidelity "
        "values use the same conservative gate-count proxy used in prior reviewer artifacts.",
    ]
    (OUT / "qaoa_counterfactual_digest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    rows: list[dict[str, str]] = []
    for item in selected_circuits():
        if not item.result_path.exists():
            raise FileNotFoundError(item.result_path)
        data = json.loads(item.result_path.read_text(encoding="utf-8"))
        rows.append(historical_row(item, data))
        circuit, ansatz = build_bound_circuit(item, data)
        for condition in [
            "swap_aware_heron",
            "fractional_gate_heron",
            "swap_aware_nighthawk",
        ]:
            rows.append(counterfactual_row(item, data, circuit, ansatz, condition))

    summary_rows = build_summary(rows)
    write_csv(OUT / "qaoa_compilation_counterfactuals.csv", COUNTERFACTUAL_COLUMNS, rows)
    write_csv(
        OUT / "qaoa_compilation_counterfactual_summary.csv",
        SUMMARY_COLUMNS,
        summary_rows,
    )
    make_figure(rows)
    write_digest(rows, summary_rows)


if __name__ == "__main__":
    main()
