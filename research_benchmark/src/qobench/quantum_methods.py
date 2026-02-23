from __future__ import annotations

import json
import logging
import math
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Literal, Optional

import numpy as np
from scipy.optimize import minimize

from .hardware_manager import HAS_BRAKET, QPUWindowExpiredError, normalize_counts


LOGGER = logging.getLogger("qobench.quantum_methods")


try:
    from braket.circuits import Circuit, FreeParameter
except Exception:
    Circuit = None
    FreeParameter = None


@dataclass(slots=True)
class AnsatzBundle:
    ansatz_id: str
    num_qubits: int
    num_parameters: int
    qiskit_template: Any
    qiskit_parameters: list[Any]
    braket_template: Any
    braket_parameters: list[Any]
    ansatz_family: str = "custom"
    ansatz_reps: int | None = None
    ansatz_entanglement: str | None = None
    metadata: dict[str, Any] | None = None


@dataclass(slots=True)
class OptimizationResult:
    method: str
    objective_mode: str
    best_value: float
    best_theta: list[float]
    best_bitstring: str
    best_energy: float
    best_counts: dict[str, int]
    optimizer_status: str
    optimizer_message: str
    total_evaluations: int
    qpu_usage: dict[str, int]
    trace: list[dict[str, Any]]


@dataclass(slots=True)
class IsingTerms:
    num_qubits: int
    linear_terms: list[tuple[int, float]]
    quadratic_terms: list[tuple[int, int, float]]


@dataclass(slots=True)
class PceEncoding:
    logical_num_vars: int
    encoded_num_qubits: int
    compression_k: int
    pauli_strings: list[str]


class QuboObjective:
    def __init__(self, qubo: Any) -> None:
        self.qubo = qubo
        self.num_qubits = int(qubo.get_num_vars())
        self.variable_names = list(qubo.variables_index.keys())
        self._energy_cache: dict[str, float] = {}

    @staticmethod
    def _sanitize_bitstring(bitstring: str, num_qubits: int) -> str:
        bits = str(bitstring).replace(" ", "")
        if len(bits) < num_qubits:
            bits = bits.zfill(num_qubits)
        elif len(bits) > num_qubits:
            bits = bits[-num_qubits:]
        return bits

    def bitstring_to_vector(self, bitstring: str) -> list[int]:
        bits = self._sanitize_bitstring(bitstring, self.num_qubits)
        return [1 if ch == "1" else 0 for ch in bits]

    def energy(self, bitstring: str) -> float:
        bits = self._sanitize_bitstring(bitstring, self.num_qubits)
        if bits in self._energy_cache:
            return self._energy_cache[bits]
        vector = self.bitstring_to_vector(bits)
        value = float(self.qubo.objective.evaluate(vector))
        self._energy_cache[bits] = value
        return value

    def expectation(self, counts: dict[str, int]) -> float:
        total_shots = sum(int(v) for v in counts.values())
        if total_shots <= 0:
            raise ValueError("Counts must contain at least one shot.")
        total = 0.0
        for bitstring, count in counts.items():
            total += self.energy(bitstring) * int(count)
        return total / float(total_shots)

    def cvar(self, counts: dict[str, int], alpha: float) -> float:
        if not (0.0 < alpha <= 1.0):
            raise ValueError("alpha must be in (0, 1].")
        total_shots = sum(int(v) for v in counts.values())
        if total_shots <= 0:
            raise ValueError("Counts must contain at least one shot.")

        weighted: list[tuple[float, int]] = []
        for bitstring, count in counts.items():
            weighted.append((self.energy(bitstring), int(count)))
        weighted.sort(key=lambda item: item[0])  # lower energy is better (minimization)

        target_mass = alpha * float(total_shots)
        if target_mass <= 0:
            target_mass = 1.0

        consumed = 0.0
        value = 0.0
        for energy, count in weighted:
            if consumed >= target_mass:
                break
            take = min(float(count), target_mass - consumed)
            value += energy * take
            consumed += take
        if consumed <= 0.0:
            consumed = 1.0
        return value / consumed

    def best_sample(self, counts: dict[str, int]) -> tuple[str, float]:
        best_bitstring = ""
        best_energy = float("inf")
        best_count = -1
        for bitstring, count in counts.items():
            energy = self.energy(bitstring)
            if energy < best_energy:
                best_energy = energy
                best_bitstring = bitstring
                best_count = int(count)
            elif math.isclose(energy, best_energy) and int(count) > best_count:
                best_bitstring = bitstring
                best_count = int(count)
        return best_bitstring, best_energy

    def assignment(self, bitstring: str) -> dict[str, int]:
        vector = self.bitstring_to_vector(bitstring)
        return {
            var_name: int(vector[idx]) for var_name, idx in self.qubo.variables_index.items()
        }


def extract_ising_terms(qubo: Any) -> IsingTerms:
    hamiltonian, _ = qubo.to_ising()
    linear_terms: dict[int, float] = {}
    quadratic_terms: dict[tuple[int, int], float] = {}

    for pauli, coeff in zip(hamiltonian.paulis, hamiltonian.coeffs):
        value = float(np.real(coeff))
        if not math.isfinite(value) or abs(value) <= 1e-12:
            continue
        if np.any(pauli.x):
            # QUBO->Ising should only produce Z/I terms; ignore anything else.
            continue
        z_indices = [int(idx) for idx, bit in enumerate(pauli.z) if bool(bit)]
        if len(z_indices) == 1:
            q = z_indices[0]
            linear_terms[q] = linear_terms.get(q, 0.0) + value
        elif len(z_indices) == 2:
            u, v = sorted(z_indices)
            key = (u, v)
            quadratic_terms[key] = quadratic_terms.get(key, 0.0) + value

    linear_list = sorted(linear_terms.items(), key=lambda item: item[0])
    quadratic_list = sorted(
        [(u, v, c) for (u, v), c in quadratic_terms.items()],
        key=lambda item: (item[0], item[1]),
    )
    return IsingTerms(
        num_qubits=int(hamiltonian.num_qubits),
        linear_terms=[(int(q), float(c)) for q, c in linear_list],
        quadratic_terms=[(int(u), int(v), float(c)) for u, v, c in quadratic_list],
    )


def estimate_qrao_num_qubits(qubo: Any, max_vars_per_qubit: int) -> int:
    try:
        from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
    except Exception as exc:
        raise ModuleNotFoundError(
            "QRAO support requires qiskit-optimization[qrao] components."
        ) from exc

    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=int(max_vars_per_qubit))
    encoding.encode(qubo)
    return int(encoding.num_qubits)


def estimate_pce_num_qubits(*, num_variables: int, compression_k: int) -> int:
    m = int(num_variables)
    k = int(compression_k)
    if m <= 0:
        return 1
    if k < 1:
        raise ValueError("PCE compression k must be >= 1.")
    n = max(1, k)
    while 3 * math.comb(n, k) < m:
        n += 1
    return int(n)


def generate_pce_pauli_strings(*, num_qubits: int, num_variables: int, compression_k: int) -> list[str]:
    n = int(num_qubits)
    m = int(num_variables)
    k = int(compression_k)
    if n < 1:
        raise ValueError("num_qubits must be >= 1 for PCE.")
    if m < 1:
        return []
    if k < 1 or k > n:
        raise ValueError(f"Invalid PCE compression k={k} for n={n}.")

    from itertools import combinations

    supports = list(combinations(range(n), k))
    pauli_strings: list[str] = []
    for op in ("X", "Y", "Z"):
        for support in supports:
            chars = ["I"] * n
            for idx in support:
                chars[idx] = op
            pauli_strings.append("".join(chars))
            if len(pauli_strings) >= m:
                return pauli_strings
    if len(pauli_strings) < m:
        raise ValueError(
            f"PCE mapping insufficient: generated={len(pauli_strings)} required={m} "
            f"(n={n}, k={k})."
        )
    return pauli_strings[:m]


def decode_pce_counts(
    *,
    counts: dict[str, int],
    encoding: PceEncoding,
) -> dict[str, int]:
    supports = [
        tuple(idx for idx, ch in enumerate(pauli_str) if ch != "I")
        for pauli_str in encoding.pauli_strings
    ]
    decoded: dict[str, int] = {}
    for raw_bitstring, raw_count in counts.items():
        count = int(raw_count)
        if count <= 0:
            continue
        bits = str(raw_bitstring).replace(" ", "")
        if len(bits) < encoding.encoded_num_qubits:
            bits = bits.zfill(encoding.encoded_num_qubits)
        elif len(bits) > encoding.encoded_num_qubits:
            bits = bits[-encoding.encoded_num_qubits:]
        z_values = [1 if ch == "0" else -1 for ch in bits]
        logical_bits: list[str] = []
        for support in supports:
            parity = 1
            for idx in support:
                parity *= z_values[idx]
            logical_bits.append("0" if parity >= 0 else "1")
        logical_bitstring = "".join(logical_bits)
        decoded[logical_bitstring] = decoded.get(logical_bitstring, 0) + count
    if not decoded:
        decoded["0" * int(encoding.logical_num_vars)] = 1
    return decoded


def _entanglement_pairs(num_qubits: int, entanglement: str) -> list[tuple[int, int]]:
    if num_qubits < 2:
        return []
    if entanglement == "full":
        return [(i, j) for i in range(num_qubits) for j in range(i + 1, num_qubits)]
    if entanglement == "circular":
        pairs = [(i, i + 1) for i in range(num_qubits - 1)]
        pairs.append((num_qubits - 1, 0))  # wrap-around
        return pairs
    # Default: chain (linear)
    return [(i, i + 1) for i in range(num_qubits - 1)]


def _qiskit_entanglement_label(entanglement: str) -> str:
    if entanglement == "chain":
        return "linear"
    return entanglement


def build_vqe_ansatz_bundle(num_qubits: int, layers: int, entanglement: str) -> AnsatzBundle:
    from qiskit import ClassicalRegister
    from qiskit.circuit.library import EfficientSU2

    ent_pairs = _entanglement_pairs(int(num_qubits), entanglement)
    qiskit_qc = EfficientSU2(
        num_qubits=int(num_qubits),
        entanglement=_qiskit_entanglement_label(entanglement),
        reps=int(layers),
    ).decompose()
    if int(qiskit_qc.num_clbits) < int(num_qubits):
        qiskit_qc.add_register(ClassicalRegister(int(num_qubits), "c"))
    qiskit_qc.measure(range(int(num_qubits)), range(int(num_qubits)))
    qiskit_params = list(qiskit_qc.parameters)

    braket_qc = None
    braket_params: list[Any] = []
    if HAS_BRAKET and Circuit is not None and FreeParameter is not None:
        braket_qc = Circuit()
        param_index = 0
        for layer in range(int(layers) + 1):
            for q in range(int(num_qubits)):
                p_ry = FreeParameter(f"theta_{param_index}")
                braket_params.append(p_ry)
                braket_qc.ry(q, p_ry)
                param_index += 1
                p_rz = FreeParameter(f"theta_{param_index}")
                braket_params.append(p_rz)
                braket_qc.rz(q, p_rz)
                param_index += 1
            if layer < int(layers):
                for u, v in ent_pairs:
                    braket_qc.cnot(u, v)

    return AnsatzBundle(
        ansatz_id=f"vqe_efficientsu2_n{num_qubits}_l{layers}_e{entanglement}",
        num_qubits=int(num_qubits),
        num_parameters=len(qiskit_params),
        qiskit_template=qiskit_qc,
        qiskit_parameters=qiskit_params,
        braket_template=braket_qc,
        braket_parameters=braket_params,
        ansatz_family="EfficientSU2",
        ansatz_reps=int(layers),
        ansatz_entanglement=str(entanglement),
        metadata={},
    )


def _parameter_sort_key(param: Any) -> tuple[str, int]:
    name = str(getattr(param, "name", str(param)))
    digits = "".join(ch for ch in name if ch.isdigit())
    if digits:
        try:
            return ("", int(digits))
        except Exception:
            return (name, 0)
    return (name, 0)


def build_pce_ansatz_bundle(
    *,
    logical_num_vars: int,
    compression_k: int,
    depth: int,
) -> AnsatzBundle:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    encoded_num_qubits = estimate_pce_num_qubits(
        num_variables=int(logical_num_vars),
        compression_k=int(compression_k),
    )
    effective_depth = int(depth)
    if effective_depth <= 0:
        effective_depth = max(1, 2 * int(encoded_num_qubits))

    num_qubits = int(encoded_num_qubits)
    num_layers = int(effective_depth)
    qiskit_qc = QuantumCircuit(num_qubits, num_qubits)
    qiskit_param_by_name: dict[str, Any] = {}

    braket_qc = None
    braket_param_by_name: dict[str, Any] = {}
    if HAS_BRAKET and Circuit is not None and FreeParameter is not None:
        braket_qc = Circuit()

    def _new_param(name: str) -> tuple[Any, Any]:
        qparam = Parameter(name)
        qiskit_param_by_name[name] = qparam
        bparam = None
        if braket_qc is not None:
            bparam = FreeParameter(name)
            braket_param_by_name[name] = bparam
        return qparam, bparam

    # Keep naming close to the notebook implementation.
    first_block: list[tuple[Any, Any]] = []
    second_block: list[tuple[Any, Any]] = []
    for idx in range(1, (num_layers * num_qubits) + 1):
        first_block.append(_new_param(f"phi{idx}"))
    for idx in range((num_layers * num_qubits) + 1, (2 * num_layers * num_qubits) + 1):
        second_block.append(_new_param(f"phi{idx}"))

    for layer in range(num_layers):
        first_start = layer * num_qubits
        second_start = layer * num_qubits
        for q in range(num_qubits):
            qparam, bparam = first_block[first_start + q]
            if (layer % 3) == 1:
                qiskit_qc.ry(qparam, q)
                if braket_qc is not None:
                    braket_qc.ry(q, bparam)
            elif (layer % 3) == 2:
                qiskit_qc.rx(qparam, q)
                if braket_qc is not None:
                    braket_qc.rx(q, bparam)
            else:
                qiskit_qc.rz(qparam, q)
                if braket_qc is not None:
                    braket_qc.rz(q, bparam)

        for q in range(0, num_qubits, 2):
            if (q + 1) >= num_qubits:
                continue
            qparam, bparam = second_block[second_start + (q // 2)]
            qiskit_qc.rxx(qparam, q, q + 1)
            if braket_qc is not None:
                try:
                    braket_qc.xx(q, q + 1, bparam)
                except Exception:
                    braket_qc.cnot(q, q + 1)
                    braket_qc.rx(q + 1, bparam)
                    braket_qc.cnot(q, q + 1)

        qiskit_qc.barrier()

        for q in range(num_qubits):
            qparam, bparam = first_block[first_start + q]
            if (layer % 3) == 1:
                qiskit_qc.rz(qparam, q)
                if braket_qc is not None:
                    braket_qc.rz(q, bparam)
            elif (layer % 3) == 2:
                qiskit_qc.ry(qparam, q)
                if braket_qc is not None:
                    braket_qc.ry(q, bparam)
            else:
                qiskit_qc.rx(qparam, q)
                if braket_qc is not None:
                    braket_qc.rx(q, bparam)

        for q in range(1, num_qubits, 2):
            if (q + 1) >= num_qubits:
                continue
            idx = second_start + (num_qubits // 2) + (q // 2)
            if idx >= len(second_block):
                continue
            qparam, bparam = second_block[idx]
            qiskit_qc.rxx(qparam, q, q + 1)
            if braket_qc is not None:
                try:
                    braket_qc.xx(q, q + 1, bparam)
                except Exception:
                    braket_qc.cnot(q, q + 1)
                    braket_qc.rx(q + 1, bparam)
                    braket_qc.cnot(q, q + 1)
        qiskit_qc.barrier()

    qiskit_qc.measure(range(num_qubits), range(num_qubits))
    qiskit_params = sorted(list(qiskit_qc.parameters), key=_parameter_sort_key)
    braket_params = [
        braket_param_by_name[p.name]
        for p in qiskit_params
        if p.name in braket_param_by_name
    ]

    pauli_strings = generate_pce_pauli_strings(
        num_qubits=num_qubits,
        num_variables=int(logical_num_vars),
        compression_k=int(compression_k),
    )
    pce_encoding = PceEncoding(
        logical_num_vars=int(logical_num_vars),
        encoded_num_qubits=num_qubits,
        compression_k=int(compression_k),
        pauli_strings=pauli_strings,
    )
    return AnsatzBundle(
        ansatz_id=f"pce_brickwork_n{num_qubits}_d{num_layers}_k{compression_k}_m{logical_num_vars}",
        num_qubits=num_qubits,
        num_parameters=len(qiskit_params),
        qiskit_template=qiskit_qc,
        qiskit_parameters=qiskit_params,
        braket_template=braket_qc,
        braket_parameters=braket_params,
        ansatz_family="Brickwork",
        ansatz_reps=num_layers,
        ansatz_entanglement="brickwork",
        metadata={
            "pce_encoding": {
                "logical_num_vars": int(pce_encoding.logical_num_vars),
                "encoded_num_qubits": int(pce_encoding.encoded_num_qubits),
                "compression_k": int(pce_encoding.compression_k),
                "pauli_strings": list(pce_encoding.pauli_strings),
            }
        },
    )


def _build_warm_start_angles(
    *,
    qp: Any | None,
    num_qubits: int,
    epsilon: float,
) -> tuple[list[float], str]:
    safe_epsilon = min(0.49, max(0.0, float(epsilon)))
    if qp is None:
        return [float(math.pi / 2.0) for _ in range(num_qubits)], "uniform_no_qp"
    try:
        import copy

        from qiskit_optimization.algorithms import CplexOptimizer
        from qiskit_optimization.problems.variable import VarType

        relaxed_problem = copy.deepcopy(qp)
        for variable in relaxed_problem.variables:
            variable.vartype = VarType.CONTINUOUS

        solution = CplexOptimizer().solve(relaxed_problem)
        values = np.asarray(solution.x, dtype=float)
        if values.size != num_qubits:
            raise ValueError(
                f"Warm-start solution size mismatch: expected {num_qubits}, got {values.size}"
            )
        values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
        if safe_epsilon > 0.0:
            values = np.clip(values, safe_epsilon, 1.0 - safe_epsilon)
        else:
            values = np.clip(values, 0.0, 1.0)
        return [float(2.0 * np.arcsin(np.sqrt(v))) for v in values], "relaxed_qp"
    except Exception as exc:
        LOGGER.warning(
            "Warm-start relaxation failed (%s). Falling back to uniform warm-start state.",
            str(exc)[:200],
        )
        return [float(math.pi / 2.0) for _ in range(num_qubits)], "uniform_fallback"


def build_qaoa_ansatz_bundle(
    *,
    ising_terms: IsingTerms,
    layers: int,
    multi_angle: bool = False,
    warm_start_angles: list[float] | None = None,
    warm_start_source: str | None = None,
) -> AnsatzBundle:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    num_qubits = int(ising_terms.num_qubits)
    qiskit_qc = QuantumCircuit(num_qubits, num_qubits)
    qiskit_params: list[Any] = []

    braket_qc = None
    braket_params: list[Any] = []
    if HAS_BRAKET and Circuit is not None and FreeParameter is not None:
        braket_qc = Circuit()

    param_counter = 0

    def _new_parameter(prefix: str) -> tuple[Any, Any]:
        nonlocal param_counter
        name = f"{prefix}_{param_counter}"
        param_counter += 1
        qiskit_param = Parameter(name)
        qiskit_params.append(qiskit_param)
        braket_param = None
        if braket_qc is not None:
            braket_param = FreeParameter(name)
            braket_params.append(braket_param)
        return qiskit_param, braket_param

    if warm_start_angles is None:
        for q in range(num_qubits):
            qiskit_qc.h(q)
            if braket_qc is not None:
                braket_qc.h(q)
    else:
        if len(warm_start_angles) != num_qubits:
            raise ValueError(
                f"warm_start_angles length mismatch: expected {num_qubits}, got {len(warm_start_angles)}"
            )
        for q, angle in enumerate(warm_start_angles):
            qiskit_qc.ry(float(angle), q)
            if braket_qc is not None:
                braket_qc.ry(q, float(angle))

    for _layer in range(int(layers)):
        if multi_angle:
            linear_params: list[tuple[int, float, Any, Any]] = []
            quad_params: list[tuple[int, int, float, Any, Any]] = []
            for q, coeff in ising_terms.linear_terms:
                q_param, b_param = _new_parameter("gamma")
                linear_params.append((int(q), float(coeff), q_param, b_param))
            for u, v, coeff in ising_terms.quadratic_terms:
                q_param, b_param = _new_parameter("gamma")
                quad_params.append((int(u), int(v), float(coeff), q_param, b_param))
        else:
            shared_q_param, shared_b_param = _new_parameter("gamma")
            linear_params = [
                (int(q), float(coeff), shared_q_param, shared_b_param)
                for q, coeff in ising_terms.linear_terms
            ]
            quad_params = [
                (int(u), int(v), float(coeff), shared_q_param, shared_b_param)
                for u, v, coeff in ising_terms.quadratic_terms
            ]

        for q, coeff, q_param, b_param in linear_params:
            qiskit_qc.rz((2.0 * coeff) * q_param, q)
            if braket_qc is not None:
                braket_qc.rz(q, (2.0 * coeff) * b_param)

        for u, v, coeff, q_param, b_param in quad_params:
            angle_q = (2.0 * coeff) * q_param
            qiskit_qc.rzz(angle_q, u, v)
            if braket_qc is not None:
                angle_b = (2.0 * coeff) * b_param
                # Braket does not expose a direct RZZ gate in all SDK versions.
                braket_qc.cnot(u, v)
                braket_qc.rz(v, angle_b)
                braket_qc.cnot(u, v)

        if warm_start_angles is None:
            if multi_angle:
                for q in range(num_qubits):
                    q_param, b_param = _new_parameter("beta")
                    qiskit_qc.rx(2.0 * q_param, q)
                    if braket_qc is not None:
                        braket_qc.rx(q, 2.0 * b_param)
            else:
                q_param, b_param = _new_parameter("beta")
                for q in range(num_qubits):
                    qiskit_qc.rx(2.0 * q_param, q)
                    if braket_qc is not None:
                        braket_qc.rx(q, 2.0 * b_param)
        else:
            q_param, b_param = _new_parameter("beta")
            for q, angle in enumerate(warm_start_angles):
                theta = float(angle)
                qiskit_qc.ry(-theta, q)
                qiskit_qc.rz(-2.0 * q_param, q)
                qiskit_qc.ry(theta, q)
                if braket_qc is not None:
                    braket_qc.ry(q, -theta)
                    braket_qc.rz(q, -2.0 * b_param)
                    braket_qc.ry(q, theta)

    qiskit_qc.measure(range(num_qubits), range(num_qubits))
    if warm_start_angles is not None:
        ansatz_label = "ws_qaoa"
    elif multi_angle:
        ansatz_label = "ma_qaoa"
    else:
        ansatz_label = "qaoa"

    metadata: dict[str, Any] = {}
    if warm_start_angles is not None:
        min_angle = float(min(warm_start_angles)) if warm_start_angles else float("nan")
        max_angle = float(max(warm_start_angles)) if warm_start_angles else float("nan")
        mean_angle = float(np.mean(np.asarray(warm_start_angles, dtype=float))) if warm_start_angles else float("nan")
        metadata["warm_start"] = {
            "enabled": True,
            "source": str(warm_start_source or "unknown"),
            "angle_min": min_angle,
            "angle_max": max_angle,
            "angle_mean": mean_angle,
            "num_angles": int(len(warm_start_angles)),
        }

    return AnsatzBundle(
        ansatz_id=f"{ansatz_label}_n{num_qubits}_l{layers}",
        num_qubits=num_qubits,
        num_parameters=len(qiskit_params),
        qiskit_template=qiskit_qc,
        qiskit_parameters=qiskit_params,
        braket_template=braket_qc,
        braket_parameters=braket_params,
        ansatz_family="QAOAAnsatz" if not multi_angle else "QAOAAnsatz (multi-angle)",
        ansatz_reps=int(layers),
        ansatz_entanglement="problem_hamiltonian",
        metadata=metadata,
    )


def build_algorithm_ansatz_bundle(
    *,
    method: str,
    qubo: Any,
    layers: int,
    entanglement: str,
    qp: Any | None = None,
    ws_epsilon: float = 1e-3,
    pce_compression_k: int = 2,
    pce_depth: int = 0,
) -> AnsatzBundle:
    m = str(method).lower()
    if m in {"vqe", "cvar_vqe"}:
        return build_vqe_ansatz_bundle(
            num_qubits=int(qubo.get_num_vars()),
            layers=int(layers),
            entanglement=entanglement,
        )
    if m == "pce":
        return build_pce_ansatz_bundle(
            logical_num_vars=int(qubo.get_num_vars()),
            compression_k=int(pce_compression_k),
            depth=int(pce_depth),
        )

    terms = extract_ising_terms(qubo)
    if m in {"qaoa", "cvar_qaoa"}:
        return build_qaoa_ansatz_bundle(
            ising_terms=terms,
            layers=int(layers),
            multi_angle=False,
            warm_start_angles=None,
        )
    if m == "ws_qaoa":
        warm_start, warm_start_source = _build_warm_start_angles(
            qp=qp,
            num_qubits=int(terms.num_qubits),
            epsilon=float(ws_epsilon),
        )
        return build_qaoa_ansatz_bundle(
            ising_terms=terms,
            layers=int(layers),
            multi_angle=False,
            warm_start_angles=warm_start,
            warm_start_source=warm_start_source,
        )
    if m == "ma_qaoa":
        return build_qaoa_ansatz_bundle(
            ising_terms=terms,
            layers=int(layers),
            multi_angle=True,
            warm_start_angles=None,
        )
    raise ValueError(f"Unsupported ansatz method '{method}'.")


def build_ansatz_bundle(num_qubits: int, layers: int, entanglement: str) -> AnsatzBundle:
    # Backward-compatible wrapper for existing callers.
    return build_vqe_ansatz_bundle(num_qubits=num_qubits, layers=layers, entanglement=entanglement)


class QpuScheduler:
    """Round-robin QPU scheduler with availability-window awareness.

    Periodically calls ``manager.refresh_qpu_windows()`` to detect QPUs
    whose windows have opened or closed, and adjusts the active QPU list
    accordingly.
    """

    _WINDOW_CHECK_INTERVAL_SEC = 60.0  # re-check windows at most once per minute

    def __init__(
        self,
        mode: Literal["single", "multi"],
        qpu_ids: list[str],
        *,
        manager: Any = None,
        num_qubits: int = 0,
    ) -> None:
        if not qpu_ids:
            raise ValueError("qpu_ids must not be empty")
        self.mode = mode
        self.qpu_ids = list(qpu_ids)
        self._all_qpu_ids = list(qpu_ids)  # original full set
        self._rr_index = 0
        self._lock = threading.Lock()
        self._manager = manager
        self._num_qubits = num_qubits
        self._last_window_check = time.time()

    # ------------------------------------------------------------------
    # Window refresh helpers
    # ------------------------------------------------------------------
    def _maybe_refresh_windows(self) -> None:
        """Periodically re-check QPU availability windows."""
        if self._manager is None:
            return
        now = time.time()
        if now - self._last_window_check < self._WINDOW_CHECK_INTERVAL_SEC:
            return
        self._last_window_check = now
        available = self._manager.refresh_qpu_windows(
            num_qubits=self._num_qubits
        )
        # Re-add QPUs that have come back online
        for qpu_id in self._all_qpu_ids:
            if qpu_id in available and qpu_id not in self.qpu_ids:
                self.qpu_ids.append(qpu_id)
                LOGGER.info(
                    "Scheduler: QPU '%s' re-added to active pool", qpu_id,
                )

    def mark_offline(self, qpu_id: str) -> None:
        """Remove a QPU from the active rotation."""
        with self._lock:
            if qpu_id in self.qpu_ids:
                self.qpu_ids.remove(qpu_id)
                LOGGER.info(
                    "Scheduler: QPU '%s' removed from active pool  "
                    "(remaining: %s)",
                    qpu_id, self.qpu_ids or "(none)",
                )

    @property
    def has_available_qpus(self) -> bool:
        return len(self.qpu_ids) > 0

    def next_qpu(self) -> str:
        self._maybe_refresh_windows()
        with self._lock:
            if not self.qpu_ids:
                raise RuntimeError(
                    "No QPUs available in scheduler. "
                    "All QPU windows may have closed."
                )
            if self.mode == "single" or len(self.qpu_ids) == 1:
                return self.qpu_ids[0]
            qpu_id = self.qpu_ids[self._rr_index % len(self.qpu_ids)]
            self._rr_index += 1
            return qpu_id


def _top_counts(counts: dict[str, int], limit: int = 10) -> list[dict[str, Any]]:
    items = sorted(counts.items(), key=lambda item: int(item[1]), reverse=True)[:limit]
    return [{"bitstring": bitstring, "count": int(count)} for bitstring, count in items]


def _evaluate_theta(
    *,
    theta: np.ndarray,
    qpu_id: str,
    manager: Any,
    ansatz: AnsatzBundle,
    objective: QuboObjective,
    shots: int,
    timeout_sec: float | None,
    objective_mode: Literal["expectation", "cvar"],
    cvar_alpha: float,
    counts_transform: Callable[[dict[str, int]], dict[str, int]] | None = None,
) -> dict[str, Any]:
    theta_list = [float(value) for value in theta]
    LOGGER.debug(
        "Submitting evaluation | qpu_id=%s objective_mode=%s shots=%s timeout_sec=%s",
        qpu_id,
        objective_mode,
        shots,
        timeout_sec,
    )
    # QPUWindowExpiredError is intentionally NOT caught here;
    # callers (_objective_fn / PCE loop) handle failover.
    raw_counts, metadata = manager.run_counts(
        qpu_id=qpu_id,
        qiskit_template=ansatz.qiskit_template,
        qiskit_parameters=ansatz.qiskit_parameters,
        braket_template=ansatz.braket_template,
        braket_parameters=ansatz.braket_parameters,
        theta=theta_list,
        num_qubits=ansatz.num_qubits,
        shots=int(shots),
        timeout_sec=timeout_sec,
        ansatz_id=ansatz.ansatz_id,
    )
    counts = (
        counts_transform(dict(raw_counts))
        if counts_transform is not None
        else dict(raw_counts)
    )
    if objective_mode == "expectation":
        objective_value = objective.expectation(counts)
    else:
        objective_value = objective.cvar(counts, alpha=cvar_alpha)
    best_bitstring, best_energy = objective.best_sample(counts)
    LOGGER.debug(
        "Evaluation complete | qpu_id=%s objective_value=%.8f best_energy=%.8f best_bitstring=%s",
        qpu_id,
        float(objective_value),
        float(best_energy),
        best_bitstring,
    )
    return {
        "qpu_id": qpu_id,
        "theta": theta_list,
        "counts": counts,
        "raw_counts": dict(raw_counts),
        "metadata": metadata,
        "objective_value": float(objective_value),
        "best_bitstring": best_bitstring,
        "best_energy": float(best_energy),
    }


def _evaluate_thetas_batch(
    *,
    thetas: np.ndarray,
    qpu_id: str,
    manager: Any,
    ansatz: AnsatzBundle,
    objective: QuboObjective,
    shots: int,
    timeout_sec: float | None,
    counts_transform: Callable[[dict[str, int]], dict[str, int]] | None = None,
) -> list[dict[str, Any]]:
    if len(thetas) <= 0:
        return []
    theta_list_batch = [[float(value) for value in row] for row in thetas]
    LOGGER.debug(
        "Submitting batched evaluations | qpu_id=%s shots=%s timeout_sec=%s batch_size=%s",
        qpu_id,
        shots,
        timeout_sec,
        len(theta_list_batch),
    )
    raw_counts_list, metadata_list = manager.run_counts_batch(
        qpu_id=qpu_id,
        qiskit_template=ansatz.qiskit_template,
        qiskit_parameters=ansatz.qiskit_parameters,
        braket_template=ansatz.braket_template,
        braket_parameters=ansatz.braket_parameters,
        thetas=theta_list_batch,
        num_qubits=ansatz.num_qubits,
        shots=int(shots),
        timeout_sec=timeout_sec,
        ansatz_id=ansatz.ansatz_id,
    )
    results: list[dict[str, Any]] = []
    for idx, theta_list in enumerate(theta_list_batch):
        raw_counts = dict(raw_counts_list[idx])
        counts = counts_transform(raw_counts) if counts_transform is not None else raw_counts
        metadata = metadata_list[idx]
        objective_value = objective.expectation(counts)
        best_bitstring, best_energy = objective.best_sample(counts)
        results.append(
            {
                "qpu_id": qpu_id,
                "theta": theta_list,
                "counts": counts,
                "raw_counts": raw_counts,
                "metadata": metadata,
                "objective_value": float(objective_value),
                "best_bitstring": best_bitstring,
                "best_energy": float(best_energy),
            }
        )
    LOGGER.debug(
        "Batched evaluations complete | qpu_id=%s batch_size=%s",
        qpu_id,
        len(results),
    )
    return results


def run_variational_method(
    *,
    method: Literal["vqe", "cvar_vqe", "qaoa", "cvar_qaoa", "ws_qaoa", "ma_qaoa"],
    manager: Any,
    scheduler: QpuScheduler,
    ansatz: AnsatzBundle,
    objective: QuboObjective,
    shots: int,
    maxiter: int,
    seed: int,
    cvar_alpha: float,
    timeout_sec: float | None,
) -> OptimizationResult:
    objective_mode: Literal["expectation", "cvar"] = "expectation"
    if method in {"cvar_vqe", "cvar_qaoa"}:
        objective_mode = "cvar"

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=(ansatz.num_parameters,))
    if method == "ws_qaoa":
        warm_theta0 = np.array(theta0, dtype=float)
        for idx, param in enumerate(ansatz.qiskit_parameters):
            pname = str(getattr(param, "name", "")).lower()
            if "gamma" in pname:
                warm_theta0[idx] = float(np.pi)
            elif "beta" in pname:
                warm_theta0[idx] = float(np.pi / 2.0)
        theta0 = warm_theta0
    LOGGER.debug(
        "Variational optimizer start | method=%s objective_mode=%s qpu_mode=%s qpus=%s shots=%s maxiter=%s",
        method,
        objective_mode,
        scheduler.mode,
        scheduler.qpu_ids,
        shots,
        maxiter,
    )

    trace: list[dict[str, Any]] = []
    qpu_usage: dict[str, int] = {}
    best_value = float("inf")
    best_theta = [float(x) for x in theta0]
    best_bitstring = "0" * ansatz.num_qubits
    best_energy = float("inf")
    best_counts: dict[str, int] = {}
    eval_counter = 0

    def _objective_fn(theta: np.ndarray) -> float:
        nonlocal best_value, best_theta, best_counts, best_bitstring, best_energy, eval_counter
        eval_counter += 1

        # Retry loop: try each available QPU before giving up
        max_retries = max(len(scheduler._all_qpu_ids), 2)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            if not scheduler.has_available_qpus:
                LOGGER.warning(
                    "All QPUs offline at eval=%s. "
                    "Forcing window refresh...",
                    eval_counter,
                )
                scheduler._maybe_refresh_windows()
                if not scheduler.has_available_qpus:
                    raise RuntimeError(
                        "No QPUs available — all availability windows are closed."
                    )

            qpu_id = scheduler.next_qpu()
            try:
                result = _evaluate_theta(
                    theta=theta,
                    qpu_id=qpu_id,
                    manager=manager,
                    ansatz=ansatz,
                    objective=objective,
                    shots=shots,
                    timeout_sec=timeout_sec,
                    objective_mode=objective_mode,
                    cvar_alpha=cvar_alpha,
                )
                break  # success
            except QPUWindowExpiredError as exc:
                last_error = exc
                LOGGER.warning(
                    "QPU '%s' window expired at eval=%s (attempt %d/%d). "
                    "Trying next QPU...",
                    qpu_id, eval_counter, attempt + 1, max_retries,
                )
                scheduler.mark_offline(qpu_id)
                continue
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "QPU '%s' failed at eval=%s (attempt %d/%d): %s  "
                    "Trying next QPU...",
                    qpu_id, eval_counter, attempt + 1, max_retries,
                    str(exc)[:200],
                )
                # Don't permanently remove — device errors may be transient
                continue
        else:
            # All retries exhausted
            raise RuntimeError(
                f"All QPUs failed at eval={eval_counter}. "
                f"Last error: {last_error}"
            ) from last_error

        value = float(result["objective_value"])
        qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + 1

        LOGGER.debug(
            "Variational evaluation | eval=%s qpu_id=%s objective=%.8f best_sample_energy=%.8f",
            eval_counter,
            qpu_id,
            value,
            float(result["best_energy"]),
        )

        trace.append(
            {
                "evaluation": eval_counter,
                "qpu_id": qpu_id,
                "objective_value": value,
                "best_sample_energy": float(result["best_energy"]),
                "best_sample_bitstring": result["best_bitstring"],
                "metadata": result["metadata"],
                "top_counts": _top_counts(result["counts"]),
            }
        )

        if value < best_value:
            best_value = value
            best_theta = [float(x) for x in result["theta"]]
            best_counts = dict(result["counts"])
            best_bitstring = str(result["best_bitstring"])
            best_energy = float(result["best_energy"])
            LOGGER.debug(
                "New variational incumbent | eval=%s qpu_id=%s objective=%.8f bitstring=%s",
                eval_counter,
                qpu_id,
                best_value,
                best_bitstring,
            )
        return value

    minimize_result = minimize(
        fun=_objective_fn,
        x0=theta0,
        method="COBYLA",
        options={"maxiter": int(maxiter), "rhobeg": 0.4},
    )
    LOGGER.debug(
        "Variational optimizer complete | status=%s evaluations=%s best_objective=%.8f",
        minimize_result.status,
        eval_counter,
        best_value,
    )

    return OptimizationResult(
        method=method,
        objective_mode=objective_mode,
        best_value=float(best_value),
        best_theta=best_theta,
        best_bitstring=best_bitstring,
        best_energy=float(best_energy),
        best_counts=best_counts,
        optimizer_status=str(minimize_result.status),
        optimizer_message=str(minimize_result.message),
        total_evaluations=int(eval_counter),
        qpu_usage=qpu_usage,
        trace=trace,
    )


def run_pce_method(
    *,
    manager: Any,
    scheduler: QpuScheduler,
    ansatz: AnsatzBundle,
    objective: QuboObjective,
    shots: int,
    maxiter: int,
    seed: int,
    population_size: int,
    elite_frac: float,
    timeout_sec: float | None,
    parallel_workers: int,
    batch_size: int,
    pce_encoding: PceEncoding | None = None,
) -> OptimizationResult:
    if population_size < 2:
        raise ValueError("population_size must be >= 2")
    if not (0.0 < elite_frac <= 1.0):
        raise ValueError("elite_frac must be in (0, 1].")
    if batch_size < 1:
        raise ValueError("batch_size must be >= 1")

    rng = np.random.default_rng(seed)
    mean = rng.uniform(0.0, 2.0 * np.pi, size=(ansatz.num_parameters,))
    std = np.full(shape=(ansatz.num_parameters,), fill_value=np.pi / 2.0)
    elite_count = max(1, int(math.ceil(population_size * elite_frac)))
    LOGGER.debug(
        "PCE optimizer start | qpu_mode=%s qpus=%s shots=%s maxiter=%s population=%s elite_count=%s batch_size=%s pce_k=%s encoded_qubits=%s logical_vars=%s",
        scheduler.mode,
        scheduler.qpu_ids,
        shots,
        maxiter,
        population_size,
        elite_count,
        batch_size,
        None if pce_encoding is None else int(pce_encoding.compression_k),
        int(ansatz.num_qubits),
        int(objective.num_qubits),
    )

    counts_transform: Callable[[dict[str, int]], dict[str, int]] | None = None
    if pce_encoding is not None:
        counts_transform = lambda c: decode_pce_counts(counts=c, encoding=pce_encoding)

    trace: list[dict[str, Any]] = []
    qpu_usage: dict[str, int] = {}
    best_value = float("inf")
    best_theta = [float(x) for x in mean]
    best_bitstring = "0" * int(objective.num_qubits)
    best_energy = float("inf")
    best_counts: dict[str, int] = {}
    eval_counter = 0

    for iteration in range(1, int(maxiter) + 1):
        LOGGER.debug("PCE iteration start | iteration=%s/%s", iteration, maxiter)
        candidates = rng.normal(loc=mean, scale=np.maximum(std, 1e-3), size=(population_size, ansatz.num_parameters))
        candidates = np.mod(candidates, 2.0 * np.pi)

        eval_results: list[dict[str, Any]] = []
        use_batching = batch_size > 1 and (
            scheduler.mode == "single" or len(scheduler.qpu_ids) == 1
        )
        if batch_size > 1 and not use_batching and iteration == 1:
            LOGGER.debug(
                "PCE batching requested but disabled for multi-QPU scheduling; falling back to per-evaluation submissions."
            )

        if use_batching:
            for start_idx in range(0, population_size, batch_size):
                end_idx = min(population_size, start_idx + batch_size)
                qpu_id = scheduler.next_qpu()
                chunk = candidates[start_idx:end_idx]
                qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + len(chunk)
                eval_results.extend(
                    _evaluate_thetas_batch(
                        thetas=chunk,
                        qpu_id=qpu_id,
                        manager=manager,
                        ansatz=ansatz,
                        objective=objective,
                        shots=shots,
                        timeout_sec=timeout_sec,
                        counts_transform=counts_transform,
                    )
                )
        elif parallel_workers > 1 and scheduler.mode == "multi" and len(scheduler.qpu_ids) > 1:
            max_workers = min(int(parallel_workers), population_size)
            with ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = []
                for idx in range(population_size):
                    qpu_id = scheduler.next_qpu()
                    qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + 1
                    futures.append(
                        pool.submit(
                            _evaluate_theta,
                            theta=candidates[idx],
                            qpu_id=qpu_id,
                            manager=manager,
                            ansatz=ansatz,
                            objective=objective,
                            shots=shots,
                            timeout_sec=timeout_sec,
                            objective_mode="expectation",
                            cvar_alpha=1.0,
                            counts_transform=counts_transform,
                        )
                    )
                for future in as_completed(futures):
                    eval_results.append(future.result())
        else:
            for idx in range(population_size):
                # Retry with failover on window expiry
                max_retries = max(len(scheduler._all_qpu_ids), 2)
                for attempt in range(max_retries):
                    qpu_id = scheduler.next_qpu()
                    try:
                        result = _evaluate_theta(
                            theta=candidates[idx],
                            qpu_id=qpu_id,
                            manager=manager,
                            ansatz=ansatz,
                            objective=objective,
                            shots=shots,
                            timeout_sec=timeout_sec,
                            objective_mode="expectation",
                            cvar_alpha=1.0,
                            counts_transform=counts_transform,
                        )
                        qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + 1
                        eval_results.append(result)
                        break
                    except QPUWindowExpiredError:
                        LOGGER.warning(
                            "PCE: QPU '%s' window expired (attempt %d/%d). Trying next...",
                            qpu_id, attempt + 1, max_retries,
                        )
                        scheduler.mark_offline(qpu_id)
                    except Exception as exc:
                        LOGGER.warning(
                            "PCE: QPU '%s' failed (attempt %d/%d): %s  Trying next...",
                            qpu_id, attempt + 1, max_retries, str(exc)[:200],
                        )
                else:
                    raise RuntimeError(
                        "All QPUs failed during PCE evaluation"
                    )

        scored: list[tuple[float, dict[str, Any]]] = []
        for result in eval_results:
            value = float(result["objective_value"])
            qpu_id = str(result.get("qpu_id", result.get("metadata", {}).get("qpu_id", "unknown")))
            eval_counter += 1
            LOGGER.debug(
                "PCE evaluation | eval=%s iteration=%s qpu_id=%s objective=%.8f best_sample_energy=%.8f",
                eval_counter,
                iteration,
                qpu_id,
                value,
                float(result["best_energy"]),
            )
            trace.append(
                {
                    "evaluation": eval_counter,
                    "iteration": iteration,
                    "qpu_id": qpu_id,
                    "objective_value": value,
                    "best_sample_energy": float(result["best_energy"]),
                    "best_sample_bitstring": str(result["best_bitstring"]),
                    "metadata": result["metadata"],
                    "top_counts": _top_counts(result["counts"]),
                }
            )
            scored.append((value, result))

            if value < best_value:
                best_value = value
                best_theta = [float(x) for x in result["theta"]]
                best_counts = dict(result["counts"])
                best_bitstring = str(result["best_bitstring"])
                best_energy = float(result["best_energy"])
                LOGGER.debug(
                    "New PCE incumbent | eval=%s iteration=%s qpu_id=%s objective=%.8f bitstring=%s",
                    eval_counter,
                    iteration,
                    qpu_id,
                    best_value,
                    best_bitstring,
                )

        scored.sort(key=lambda item: item[0])
        elite = scored[:elite_count]
        elite_thetas = np.array([entry[1]["theta"] for entry in elite], dtype=float)
        mean = np.mean(elite_thetas, axis=0)
        std = np.std(elite_thetas, axis=0)

        trace.append(
            {
                "iteration_summary": iteration,
                "elite_count": elite_count,
                "best_value_so_far": float(best_value),
                "mean_theta_norm": float(np.linalg.norm(mean)),
                "std_theta_norm": float(np.linalg.norm(std)),
            }
        )
        LOGGER.debug(
            "PCE iteration summary | iteration=%s best_value_so_far=%.8f mean_norm=%.4f std_norm=%.4f",
            iteration,
            best_value,
            float(np.linalg.norm(mean)),
            float(np.linalg.norm(std)),
        )

    LOGGER.debug(
        "PCE optimizer complete | evaluations=%s best_objective=%.8f",
        eval_counter,
        best_value,
    )
    return OptimizationResult(
        method="pce",
        objective_mode="expectation",
        best_value=float(best_value),
        best_theta=best_theta,
        best_bitstring=best_bitstring,
        best_energy=float(best_energy),
        best_counts=best_counts,
        optimizer_status="0",
        optimizer_message="PCE finished",
        total_evaluations=int(eval_counter),
        qpu_usage=qpu_usage,
        trace=trace,
    )


def run_qrao_method(
    *,
    manager: Any,
    scheduler: QpuScheduler,
    qubo: Any,
    shots: int,
    maxiter: int,
    seed: int,
    max_vars_per_qubit: int,
    reps: int,
    rounding_scheme: Literal["magic", "semideterministic"],
    optimizer_name: str,
) -> OptimizationResult:
    try:
        from qiskit.circuit.library import EfficientSU2
        try:
            from qiskit.primitives import BackendEstimatorV2 as BackendEstimator
            from qiskit.primitives import BackendSamplerV2 as BackendSampler
        except Exception:
            from qiskit.primitives import BackendEstimator, BackendSampler
        from qiskit_algorithms import NumPyMinimumEigensolver, VQE
        from qiskit_algorithms.optimizers import COBYLA, POWELL, SPSA, SLSQP
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_optimization.algorithms.qrao import (
            MagicRounding,
            QuantumRandomAccessEncoding,
            QuantumRandomAccessOptimizer,
            SemideterministicRounding,
        )
    except Exception as exc:
        raise ModuleNotFoundError(
            "QRAO dependencies are missing. Install qiskit, qiskit-algorithms, and qiskit-optimization."
        ) from exc

    if scheduler.mode == "multi" and len(scheduler.qpu_ids) > 1:
        raise ValueError("QRAO currently supports a single selected QPU/backend.")

    encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=int(max_vars_per_qubit))
    encoding.encode(qubo)
    encoded_qubits = int(encoding.num_qubits)

    qpu_id = scheduler.next_qpu()
    qpu = manager.qpus.get(qpu_id)
    if qpu is None:
        raise RuntimeError(f"Unknown QPU id selected for QRAO: {qpu_id}")

    backend = None
    backend_name = qpu_id
    pending_jobs = None
    use_statevector_primitives = False
    statevector_estimator_cls: Any | None = None
    statevector_sampler_cls: Any | None = None

    if qpu.provider == "ibm":
        manager._refresh_ibm_usage_if_needed()
        if not qpu.is_available:
            raise RuntimeError(f"IBM unavailable: {qpu.last_error}")
        backend_info = manager._select_ibm_backend_info(encoded_qubits)
        if backend_info is None:
            raise RuntimeError(
                f"No IBM backend supports encoded QRAO size ({encoded_qubits} qubits)"
            )
        backend = backend_info["backend"]
        backend_name = str(backend_info.get("name", getattr(backend, "name", qpu_id)))
        pending_jobs = backend_info.get("pending_jobs")
    elif qpu.provider == "local_qiskit":
        try:
            from qiskit_aer import AerSimulator

            backend = AerSimulator(method="matrix_product_state")
            backend_name = "aer_simulator_mps"
        except Exception:
            try:
                from qiskit.primitives import StatevectorEstimator, StatevectorSampler

                use_statevector_primitives = True
                statevector_estimator_cls = StatevectorEstimator
                statevector_sampler_cls = StatevectorSampler
                backend_name = "statevector_primitives"
            except Exception:
                try:
                    from qiskit.providers.basic_provider import BasicSimulator

                    backend = BasicSimulator()
                    backend_name = "basic_simulator"
                except Exception as exc:
                    raise ModuleNotFoundError(
                        "QRAO on local_qiskit requires qiskit-aer, statevector primitives, or qiskit basic_provider."
                    ) from exc
    else:
        raise RuntimeError(
            f"QRAO currently supports only IBM or local_qiskit backends (got '{qpu.provider}')."
        )

    if use_statevector_primitives:
        estimator = statevector_estimator_cls()
        sampler = statevector_sampler_cls()
    else:
        def _build_primitive(cls: Any) -> Any:
            try:
                return cls(backend=backend)
            except Exception:
                return cls(backend)

        estimator = _build_primitive(BackendEstimator)
        try:
            sampler = BackendSampler(backend=backend, options={"default_shots": int(shots)})
        except Exception:
            try:
                sampler = BackendSampler(backend=backend)
            except Exception:
                sampler = BackendSampler(backend)

    effective_rounding_scheme = str(rounding_scheme).strip().lower()
    if effective_rounding_scheme == "magic":
        sampler_has_options = hasattr(sampler, "options")
        if use_statevector_primitives or not sampler_has_options:
            LOGGER.warning(
                "QRAO magic rounding not supported with sampler=%s; falling back to semideterministic rounding.",
                type(sampler).__name__,
            )
            effective_rounding_scheme = "semideterministic"

    optimizer_key = str(optimizer_name).strip().lower()
    if optimizer_key == "powell":
        optimizer = POWELL(maxiter=int(maxiter))
    elif optimizer_key == "slsqp":
        optimizer = SLSQP(maxiter=int(maxiter))
    elif optimizer_key == "spsa":
        optimizer = SPSA(maxiter=int(maxiter))
    else:
        optimizer_key = "cobyla"
        optimizer = COBYLA(maxiter=int(maxiter))

    algorithm_globals.random_seed = int(seed)
    ansatz = EfficientSU2(num_qubits=encoded_qubits, entanglement="linear", reps=int(reps)).decompose()
    qrao_one_q = 0
    qrao_two_q = 0
    for inst, qargs, _ in ansatz.data:
        if inst.name == "measure":
            continue
        if len(qargs) == 1:
            qrao_one_q += 1
        elif len(qargs) == 2:
            qrao_two_q += 1

    trace: list[dict[str, Any]] = [
        {
            "qrao_setup": {
                "qpu_id": qpu_id,
                "backend_name": backend_name,
                "backend_pending_jobs": pending_jobs,
                "encoded_qubits": encoded_qubits,
                "original_variables": int(qubo.get_num_vars()),
                "max_vars_per_qubit": int(max_vars_per_qubit),
                "compression_ratio": (
                    float(encoding.num_vars) / float(encoding.num_qubits)
                    if int(encoding.num_qubits) > 0
                    else None
                ),
                "ansatz_family": "EfficientSU2",
                "ansatz_reps": int(reps),
                "trainable_parameters": int(ansatz.num_parameters),
                "ansatz_depth": int(ansatz.depth()),
                "ansatz_one_qubit_gates": int(qrao_one_q),
                "ansatz_two_qubit_gates": int(qrao_two_q),
                "rounding_scheme": effective_rounding_scheme,
                "optimizer": optimizer_key,
            }
        }
    ]
    eval_counter = 0
    use_numpy_min_eigensolver = bool(
        qpu.provider == "local_qiskit"
        and (use_statevector_primitives or backend_name == "basic_simulator")
    )
    if use_numpy_min_eigensolver:
        trace.append(
            {
                "qrao_solver": {
                    "min_eigen_solver": "NumPyMinimumEigensolver",
                    "reason": "local_qiskit fallback without Aer backend support",
                }
            }
        )
        min_eigen_solver = NumPyMinimumEigensolver()
    else:
        def _callback(*args: Any) -> None:
            nonlocal eval_counter
            eval_counter += 1
            objective_value = float("nan")
            if len(args) >= 3:
                try:
                    objective_value = float(args[2])
                except Exception:
                    objective_value = float("nan")
            trace.append(
                {
                    "evaluation": int(eval_counter),
                    "qpu_id": qpu_id,
                    "backend_name": backend_name,
                    "objective_value": objective_value,
                }
            )

        min_eigen_solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=estimator,
            callback=_callback,
        )

    if effective_rounding_scheme == "magic":
        rounding = MagicRounding(sampler=sampler)
    else:
        rounding = SemideterministicRounding()

    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=min_eigen_solver,
        rounding_scheme=rounding,
    )
    result = qrao.solve(qubo)

    best_vector = [int(v) for v in list(result.x)]
    best_bitstring = "".join("1" if v == 1 else "0" for v in best_vector)
    best_energy = float(qubo.objective.evaluate(best_vector))
    best_value = float(getattr(result, "fval", best_energy))

    best_counts: dict[str, int] = {}
    samples = list(getattr(result, "samples", []) or [])
    for sample in samples[:64]:
        try:
            sample_vec = [int(v) for v in list(sample.x)]
            bitstring = "".join("1" if v == 1 else "0" for v in sample_vec)
            probability = float(getattr(sample, "probability", 0.0))
            count = int(round(probability * float(shots)))
            if count > 0:
                best_counts[bitstring] = best_counts.get(bitstring, 0) + count
        except Exception:
            continue
    if not best_counts:
        best_counts[best_bitstring] = int(max(1, shots))

    min_eigen_result = getattr(result, "min_eigen_solver_result", None)
    best_theta: list[float] = []
    if min_eigen_result is not None:
        optimal_point = getattr(min_eigen_result, "optimal_point", None)
        if optimal_point is not None:
            try:
                best_theta = [float(v) for v in list(optimal_point)]
            except Exception:
                best_theta = []
        if not best_theta:
            optimal_parameters = getattr(min_eigen_result, "optimal_parameters", None)
            if isinstance(optimal_parameters, dict):
                try:
                    items = sorted(
                        optimal_parameters.items(),
                        key=lambda item: str(item[0]),
                    )
                    best_theta = [float(v) for _, v in items]
                except Exception:
                    best_theta = []

    optimizer_status = "0"
    optimizer_message = "QRAO finished"
    if min_eigen_result is not None:
        optimizer_result = getattr(min_eigen_result, "optimizer_result", None)
        if optimizer_result is not None:
            optimizer_status = str(getattr(optimizer_result, "status", optimizer_status))
            optimizer_message = str(getattr(optimizer_result, "message", optimizer_message))

    return OptimizationResult(
        method="qrao",
        objective_mode="qrao",
        best_value=float(best_value),
        best_theta=best_theta,
        best_bitstring=best_bitstring,
        best_energy=float(best_energy),
        best_counts=best_counts,
        optimizer_status=optimizer_status,
        optimizer_message=optimizer_message,
        total_evaluations=int(max(eval_counter, 1)),
        qpu_usage={qpu_id: int(max(eval_counter, 1))},
        trace=trace,
    )


def serialize_trace(trace: list[dict[str, Any]]) -> str:
    lines = [json.dumps(entry, sort_keys=True) for entry in trace]
    return "\n".join(lines) + ("\n" if lines else "")
