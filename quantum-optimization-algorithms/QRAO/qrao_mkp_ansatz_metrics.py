from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from pathlib import Path

from qiskit.circuit.library import EfficientSU2
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit_ibm_runtime import (
    EstimatorV2 as Estimator,
    QiskitRuntimeService,
    SamplerV2 as Sampler,
)
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit_optimization.translators import from_docplex_mp


@dataclass(slots=True)
class MKPInstance:
    n: int
    m: int
    optimal_value: int
    profits: list[int]
    weights: list[list[int]]
    capacities: list[int]
    source: Path | None = None


def parse_mkp_dat_file(path: Path) -> MKPInstance:
    if not path.exists():
        raise FileNotFoundError(f"MKP instance not found: {path}")

    try:
        tokens = [int(tok) for tok in path.read_text(encoding="utf-8").split()]
    except UnicodeDecodeError:
        tokens = [int(tok) for tok in path.read_text(encoding="latin-1").split()]

    if len(tokens) < 3:
        raise ValueError(f"Invalid MKP file, expected at least 3 integers: {path}")

    n, m, optimal_value = tokens[0], tokens[1], tokens[2]
    cursor = 3
    required = n + (m * n) + m
    available = len(tokens) - cursor
    if available < required:
        raise ValueError(
            f"Invalid MKP file {path}: expected {required} values after header, got {available}"
        )

    profits = tokens[cursor : cursor + n]
    cursor += n

    weights: list[list[int]] = []
    for _ in range(m):
        row = tokens[cursor : cursor + n]
        cursor += n
        weights.append(row)

    capacities = tokens[cursor : cursor + m]

    return MKPInstance(
        n=n,
        m=m,
        optimal_value=optimal_value,
        profits=profits,
        weights=weights,
        capacities=capacities,
        source=path,
    )


def build_mkp_docplex_model(instance: MKPInstance):
    # Keep behavior aligned with benchmark MKP model setup.
    sys.path = [p for p in sys.path if "cplex" not in p.lower()]
    for key in list(os.environ.keys()):
        if "CPLEX" in key.upper():
            del os.environ[key]

    try:
        from docplex.mp.model import Model
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "docplex is required to build the MKP model. Install project dependencies first."
        ) from exc

    model = Model(name="MultiDimensionalKnapsack")
    x = model.binary_var_list(instance.n, name="x")
    model.maximize(model.sum(instance.profits[i] * x[i] for i in range(instance.n)))
    for j in range(instance.m):
        model.add_constraint(
            model.sum(instance.weights[j][i] * x[i] for i in range(instance.n))
            <= instance.capacities[j],
            ctname=f"capacity_{j}",
        )
    return model


def circuit_metrics(circuit) -> dict[str, int | None]:
    depth = circuit.depth()
    one_qubit_gates = 0
    two_qubit_gates = 0
    multi_qubit_gates = 0
    for inst, qargs, _ in circuit.data:
        if inst.name == "measure":
            continue
        qcount = len(qargs)
        if qcount == 1:
            one_qubit_gates += 1
        elif qcount == 2:
            two_qubit_gates += 1
        elif qcount > 2:
            multi_qubit_gates += 1

    total_gates = one_qubit_gates + two_qubit_gates + multi_qubit_gates
    return {
        "depth": int(depth) if depth is not None else None,
        "one_qubit_gates": int(one_qubit_gates),
        "total_gates": int(total_gates),
        "two_qubit_gates": int(two_qubit_gates),
        "multi_qubit_gates": int(multi_qubit_gates),
    }


def resolve_instance_paths(instance_arg: Path, limit: int | None = None) -> list[Path]:
    if instance_arg.is_file():
        return [instance_arg]
    if not instance_arg.exists():
        raise FileNotFoundError(f"Instance path does not exist: {instance_arg}")
    if not instance_arg.is_dir():
        raise ValueError(f"Expected a .dat file or directory, got: {instance_arg}")

    paths = sorted(instance_arg.glob("*.dat"))
    if limit is not None:
        paths = paths[:limit]
    if not paths:
        raise FileNotFoundError(f"No .dat files found in: {instance_arg}")
    return paths


def backend_display_name(backend) -> str:
    name = getattr(backend, "name", None)
    if callable(name):
        return str(name())
    return str(name)


def analyze_instance(
    instance_path: Path,
    *,
    service: QiskitRuntimeService,
    backend_name: str | None,
    qiskit_optimization_level: int,
    shots: int,
    qrao_max_vars_per_qubit: int,
    qrao_reps: int,
    qubo_penalty: float | None,
) -> None:
    instance = parse_mkp_dat_file(instance_path)
    model = build_mkp_docplex_model(instance)

    qp = from_docplex_mp(model)
    converter = (
        QuadraticProgramToQubo(penalty=float(qubo_penalty))
        if qubo_penalty is not None
        else QuadraticProgramToQubo()
    )
    qubo = converter.convert(qp)

    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=int(qrao_max_vars_per_qubit))
    encoding.encode(qubo)

    if backend_name:
        backend = service.backend(backend_name)
    else:
        backend = service.least_busy(
            operational=True,
            simulator=False,
            min_num_qubits=encoding.num_qubits,
        )

    pass_manager = generate_preset_pass_manager(
        backend=backend,
        optimization_level=int(qiskit_optimization_level),
    )

    # Primitive configuration is included for parity with the notebook setup.
    # This script still stops at ansatz metric reporting and does not submit jobs.
    estimator = Estimator(mode=backend)
    sampler = Sampler(mode=backend)
    sampler.options.default_shots = int(shots)
    # Compatibility shim for qiskit-optimization 0.8.0 MagicRounding.
    sampler.default_shots = int(shots)
    estimator.options.default_shots = int(shots)
    estimator.options.default_precision = 0.05
    # sampler.options.dynamical_decoupling.enable = True
    # sampler.options.dynamical_decoupling.sequence_type = "XY4"
    # sampler.options.twirling.enable_gates = True
    # sampler.options.twirling.num_randomizations = "auto"
    # estimator.options.dynamical_decoupling.enable = True
    # estimator.options.dynamical_decoupling.sequence_type = "XY4"
    # estimator.options.twirling.enable_gates = True
    # estimator.options.twirling.num_randomizations = "auto"

    # Match benchmark QRAO ansatz configuration.
    ansatz = EfficientSU2(
        num_qubits=encoding.num_qubits,
        entanglement="linear",
        reps=int(qrao_reps),
    ).decompose()

    pre = circuit_metrics(ansatz)
    post = circuit_metrics(pass_manager.run(ansatz))

    print(f"\n=== Instance: {instance_path.name} ===")
    print(f"Using backend: {backend_display_name(backend)}")
    print(f"Original binary variables: {encoding.num_vars}")
    print(f"Qubits after QRAO encoding: {encoding.num_qubits}")
    print(f"QRAO ansatz: EfficientSU2(reps={qrao_reps}, entanglement='linear')")
    print(f"Ansatz metrics before transpilation: {pre}")
    print(f"Ansatz metrics after transpilation: {post}")


def main() -> int:
    script_path = Path(__file__).resolve()
    project_root = script_path.parents[2]
    default_hpp_dir = project_root / "Multi_Dimension_Knapsack" / "MKP_Instances" / "hpp"

    parser = argparse.ArgumentParser(
        description=(
            "Build MKP->QUBO->QRAO encoding and print ansatz metrics before/after transpilation. "
            "This script stops before any Estimator/Sampler/VQE run."
        )
    )
    parser.add_argument(
        "--instance",
        type=Path,
        default=default_hpp_dir,
        help=(
            "Path to a MKP .dat file (for example hp1.dat) or to the hpp directory. "
            "Defaults to Multi_Dimension_Knapsack/MKP_Instances/hpp."
        ),
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If --instance is a directory, analyze only the first N .dat files.",
    )
    parser.add_argument(
        "--backend-name",
        type=str,
        default=None,
        help="Optional IBM backend name (for example ibm_brisbane). If omitted, uses least_busy.",
    )
    parser.add_argument(
        "--qiskit-optimization-level",
        type=int,
        default=3,
        help="Qiskit transpiler optimization level. Benchmark default for QRAO is 3.",
    )
    parser.add_argument(
        "--shots",
        type=int,
        default=2048,
        help="Default shots configured on sampler/estimator (no jobs are submitted in this script).",
    )
    parser.add_argument(
        "--qrao-max-vars-per-qubit",
        type=int,
        default=3,
        help="QRAO max variables per qubit. Benchmark default is 3.",
    )
    parser.add_argument(
        "--qrao-reps",
        type=int,
        default=3,
        help="EfficientSU2 reps for QRAO ansatz. Benchmark default is 3.",
    )
    parser.add_argument(
        "--qubo-penalty",
        type=float,
        default=None,
        help="Optional QUBO penalty passed to QuadraticProgramToQubo.",
    )
    args = parser.parse_args()

    instance_paths = resolve_instance_paths(args.instance.resolve(), limit=args.limit)
    service = QiskitRuntimeService()

    for instance_path in instance_paths:
        analyze_instance(
            instance_path=instance_path,
            service=service,
            backend_name=args.backend_name,
            qiskit_optimization_level=int(args.qiskit_optimization_level),
            shots=int(args.shots),
            qrao_max_vars_per_qubit=int(args.qrao_max_vars_per_qubit),
            qrao_reps=int(args.qrao_reps),
            qubo_penalty=args.qubo_penalty,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
