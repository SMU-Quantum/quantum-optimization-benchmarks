from __future__ import annotations

import json
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Literal, Optional

import numpy as np
from scipy.optimize import minimize

from .hardware_manager import HAS_BRAKET, QPUWindowExpiredError, normalize_counts


LOGGER = logging.getLogger("qobench.quantum_methods")

COBYLA_INT32_MAX = (2 ** 31) - 1


def cobyla_workspace_length(num_parameters: int) -> int:
    n = max(0, int(num_parameters))
    return int(n * (3 * n + 11) + 6)


def cobyla_max_supported_parameters() -> int:
    # Solve n*(3n+11)+6 <= int32_max exactly using integer arithmetic.
    discriminant = int(12 * (COBYLA_INT32_MAX - 6) + 121)
    return int((math.isqrt(discriminant) - 11) // 6)


def cobyla_workspace_would_overflow(num_parameters: int) -> bool:
    return cobyla_workspace_length(num_parameters) > COBYLA_INT32_MAX


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
    # PCE notebook workflow encodes the weighted MaxCut graph (QUBO vars + anchor node 0).
    logical_qubo_vars = int(logical_num_vars)
    maxcut_num_nodes = int(logical_qubo_vars + 1)
    encoded_num_qubits = estimate_pce_num_qubits(
        num_variables=maxcut_num_nodes,
        compression_k=int(compression_k),
    )
    reps = int(depth) if int(depth) > 0 else 2

    try:
        from qiskit.circuit.library import efficient_su2

        qiskit_qc = efficient_su2(
            int(encoded_num_qubits),
            ["ry", "rz"],
            reps=int(reps),
        )
    except Exception:
        from qiskit.circuit.library import EfficientSU2

        qiskit_qc = EfficientSU2(
            num_qubits=int(encoded_num_qubits),
            su2_gates=["ry", "rz"],
            entanglement="linear",
            reps=int(reps),
        )
    qiskit_qc = qiskit_qc.decompose()
    qiskit_params = sorted(list(qiskit_qc.parameters), key=_parameter_sort_key)

    pauli_strings = generate_pce_pauli_strings(
        num_qubits=int(encoded_num_qubits),
        num_variables=int(logical_qubo_vars),
        compression_k=int(compression_k),
    )
    pce_encoding = PceEncoding(
        logical_num_vars=int(logical_qubo_vars),
        encoded_num_qubits=int(encoded_num_qubits),
        compression_k=int(compression_k),
        pauli_strings=pauli_strings,
    )

    # Qiskit-estimator PCE path is Qiskit-native only.
    return AnsatzBundle(
        ansatz_id=(
            f"pce_qiskit_efficientsu2_n{encoded_num_qubits}_r{reps}"
            f"_k{compression_k}_m{maxcut_num_nodes}"
        ),
        num_qubits=int(encoded_num_qubits),
        num_parameters=len(qiskit_params),
        qiskit_template=qiskit_qc,
        qiskit_parameters=qiskit_params,
        braket_template=None,
        braket_parameters=[],
        ansatz_family="EfficientSU2",
        ansatz_reps=int(reps),
        ansatz_entanglement="linear",
        metadata={
            "pce_encoding": {
                "logical_num_vars": int(pce_encoding.logical_num_vars),
                "encoded_num_qubits": int(pce_encoding.encoded_num_qubits),
                "compression_k": int(pce_encoding.compression_k),
                "pauli_strings": list(pce_encoding.pauli_strings),
            },
            "pce_maxcut_num_nodes": int(maxcut_num_nodes),
            "pce_qubo_num_vars": int(logical_qubo_vars),
            "pce_notebook_impl": "pce_generalized_merged",
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

    # Attempt 1: Use CplexOptimizer from qiskit_optimization (may fail due to
    # BaseSampler import issue in newer Qiskit versions).
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
    except Exception as exc_qiskit:
        LOGGER.debug(
            "Warm-start via CplexOptimizer failed (%s). Trying DOcplex direct LP relaxation.",
            str(exc_qiskit)[:200],
        )

    # Attempt 2: Solve the LP relaxation directly via DOcplex, bypassing the
    # qiskit_optimization import chain that triggers the BaseSampler issue.
    try:
        from docplex.mp.model import Model as DocplexModel

        # Extract the LP relaxation from the QuadraticProgram manually.
        lp_model = DocplexModel(name="warm_start_relaxation")
        lp_vars = []
        for i in range(num_qubits):
            var_name = f"x_{i}"
            # Try to get bounds from the original QP
            try:
                orig_var = qp.get_variable(var_name)
                lb = float(getattr(orig_var, "lowerbound", 0.0))
                ub = float(getattr(orig_var, "upperbound", 1.0))
            except Exception:
                lb, ub = 0.0, 1.0
            lp_vars.append(lp_model.continuous_var(lb=lb, ub=ub, name=var_name))

        # Build objective from the QP
        obj_expr = lp_model.linear_expr()
        try:
            obj_sense = qp.objective.sense.value  # 1=minimize, -1=maximize
        except Exception:
            obj_sense = 1  # default to minimize

        # Linear terms
        try:
            linear_dict = qp.objective.linear.to_dict()
            for idx, coeff in linear_dict.items():
                obj_expr += float(coeff) * lp_vars[int(idx)]
        except Exception:
            pass

        # Quadratic terms
        try:
            quad_dict = qp.objective.quadratic.to_dict()
            for (i, j), coeff in quad_dict.items():
                # For LP relaxation, approximate x_i * x_j as 0.5*(x_i + x_j) - 0.25
                obj_expr += float(coeff) * (0.5 * (lp_vars[int(i)] + lp_vars[int(j)]) - 0.25)
        except Exception:
            pass

        if obj_sense >= 0:
            lp_model.minimize(obj_expr)
        else:
            lp_model.maximize(obj_expr)

        # Add linear constraints from the QP
        try:
            for constraint in qp.linear_constraints:
                lhs = lp_model.linear_expr()
                for idx, coeff in constraint.linear.to_dict().items():
                    lhs += float(coeff) * lp_vars[int(idx)]
                sense = str(constraint.sense.name).lower()
                rhs = float(constraint.rhs)
                if sense in ("le", "<="):
                    lp_model.add_constraint(lhs <= rhs)
                elif sense in ("ge", ">="):
                    lp_model.add_constraint(lhs >= rhs)
                else:
                    lp_model.add_constraint(lhs == rhs)
        except Exception:
            pass

        sol = lp_model.solve()
        if sol is not None:
            values = np.array([sol.get_value(v) for v in lp_vars], dtype=float)
            values = np.nan_to_num(values, nan=0.5, posinf=1.0, neginf=0.0)
            if safe_epsilon > 0.0:
                values = np.clip(values, safe_epsilon, 1.0 - safe_epsilon)
            else:
                values = np.clip(values, 0.0, 1.0)
            LOGGER.info("Warm-start: DOcplex direct LP relaxation succeeded.")
            return [float(2.0 * np.arcsin(np.sqrt(v))) for v in values], "relaxed_docplex"
        else:
            LOGGER.debug("Warm-start: DOcplex LP relaxation returned no solution.")
    except Exception as exc_docplex:
        LOGGER.debug(
            "Warm-start via DOcplex failed (%s). Trying heuristic warm-start.",
            str(exc_docplex)[:200],
        )

    # Attempt 3: Heuristic warm-start based on QUBO linear coefficients.
    # Instead of uniform pi/2 (which encodes 0.5 for every variable),
    # use the sign of linear coefficients to bias towards 0 or 1.
    try:
        angles = []
        for i in range(num_qubits):
            try:
                linear_dict = qp.objective.linear.to_dict()
                coeff = float(linear_dict.get(i, 0.0))
            except Exception:
                coeff = 0.0

            # For minimisation: negative coeff => prefer x=1, positive => prefer x=0
            try:
                obj_sense = qp.objective.sense.value  # 1=minimize, -1=maximize
            except Exception:
                obj_sense = 1
            effective_coeff = coeff * obj_sense

            if effective_coeff < -1e-8:
                # Bias toward x=1
                v = max(safe_epsilon, 0.75) if safe_epsilon > 0 else 0.75
            elif effective_coeff > 1e-8:
                # Bias toward x=0
                v = min(1.0 - safe_epsilon, 0.25) if safe_epsilon > 0 else 0.25
            else:
                v = 0.5

            angles.append(float(2.0 * np.arcsin(np.sqrt(v))))

        LOGGER.info("Warm-start: Using heuristic warm-start from linear coefficients.")
        return angles, "heuristic_linear_coefficients"
    except Exception as exc_heuristic:
        LOGGER.warning(
            "Warm-start all methods failed. Final fallback: (%s). Using uniform warm-start state.",
            str(exc_heuristic)[:200],
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


def _qubo_to_maxcut_weight_matrix(qubo: Any) -> np.ndarray:
    quadratic = np.asarray(qubo.objective.quadratic.to_array(), dtype=float).copy()
    linear = np.asarray(qubo.objective.linear.to_array(), dtype=float).reshape(-1)
    if quadratic.shape[0] != quadratic.shape[1]:
        raise ValueError("QUBO quadratic matrix must be square.")
    if quadratic.shape[0] != linear.shape[0]:
        raise ValueError("QUBO linear and quadratic dimensions do not match.")

    # Same conversion used in notebook helpers: push linear terms to diagonal.
    diag = np.diag_indices_from(quadratic)
    quadratic[diag] = quadratic[diag] + linear

    n = int(linear.shape[0])
    graph = np.zeros((n + 1, n + 1), dtype=float)

    for i in range(1, n + 1):
        weight = float(np.sum(quadratic[i - 1, :]) + np.sum(quadratic[:, i - 1]))
        graph[0, i] = weight
        graph[i, 0] = weight

    for i in range(1, n + 1):
        for j in range(i + 1, n + 1):
            weight = float(quadratic[i - 1, j - 1] + quadratic[j - 1, i - 1])
            graph[i, j] = weight
            graph[j, i] = weight

    return graph


def _maxcut_edges_from_weight_matrix(weight_matrix: np.ndarray) -> list[tuple[int, int, float]]:
    n = int(weight_matrix.shape[0])
    edges: list[tuple[int, int, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            w = float(weight_matrix[i, j])
            if abs(w) > 1e-12:
                edges.append((i, j, w))
    return edges


def _mst_weight(num_nodes: int, edges: list[tuple[int, int, float]]) -> float:
    parent = list(range(num_nodes))
    rank = [0] * num_nodes

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> bool:
        ra = _find(a)
        rb = _find(b)
        if ra == rb:
            return False
        if rank[ra] < rank[rb]:
            parent[ra] = rb
        elif rank[ra] > rank[rb]:
            parent[rb] = ra
        else:
            parent[rb] = ra
            rank[ra] += 1
        return True

    total = 0.0
    for u, v, w in sorted(edges, key=lambda item: float(item[2])):
        if _union(int(u), int(v)):
            total += float(w)
    return float(total)


def _pce_weighted_nu(num_nodes: int, edges: list[tuple[int, int, float]]) -> float:
    if num_nodes <= 0:
        return 0.0
    w_graph = float(sum(float(w) for _, _, w in edges))
    w_tree_min = _mst_weight(int(num_nodes), edges) if edges else 0.0
    return float((w_graph / 2.0) + (w_tree_min / 4.0))


def _maxcut_to_qubo_bitstring(maxcut_bits: list[int]) -> str:
    if len(maxcut_bits) <= 1:
        return ""
    anchor = 1 if int(maxcut_bits[0]) == 1 else 0
    qubo_bits = []
    for bit in maxcut_bits[1:]:
        b = 1 if int(bit) == 1 else 0
        qubo_bits.append("1" if b == anchor else "0")
    return "".join(qubo_bits)


def _strip_final_measurements(circuit: Any) -> Any:
    try:
        return circuit.remove_final_measurements(inplace=False)
    except Exception:
        pass
    try:
        qc = circuit.copy()
        kept = []
        for item in getattr(qc, "data", []):
            try:
                op = item.operation
                name = str(getattr(op, "name", ""))
            except Exception:
                op = item[0] if isinstance(item, (list, tuple)) and item else None
                name = str(getattr(op, "name", ""))
            if name == "measure":
                continue
            kept.append(item)
        qc.data = kept
        return qc
    except Exception:
        return circuit


def _group_reversed_pce_pauli_operators(pauli_strings: list[str]) -> list[list[Any]]:
    from qiskit.quantum_info import SparsePauliOp

    x_labels: list[str] = []
    y_labels: list[str] = []
    z_labels: list[str] = []
    for label in pauli_strings:
        if "X" in label:
            x_labels.append(label[::-1])
        elif "Y" in label:
            y_labels.append(label[::-1])
        else:
            z_labels.append(label[::-1])

    return [
        [SparsePauliOp.from_list([(p, 1.0)]) for p in x_labels],
        [SparsePauliOp.from_list([(p, 1.0)]) for p in y_labels],
        [SparsePauliOp.from_list([(p, 1.0)]) for p in z_labels],
    ]


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
    if cobyla_workspace_would_overflow(ansatz.num_parameters):
        max_supported = cobyla_max_supported_parameters()
        workspace_len = cobyla_workspace_length(ansatz.num_parameters)
        raise RuntimeError(
            "COBYLA workspace overflow: "
            f"trainable_parameters={int(ansatz.num_parameters)} exceeds safe limit "
            f"{max_supported} (workspace={workspace_len} > int32_max={COBYLA_INT32_MAX})."
        )

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


def _run_pce_method_legacy_counts(
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
    counts_transform: Callable[[dict[str, int]], dict[str, int]] | None = None
    if pce_encoding is not None:
        counts_transform = lambda c: decode_pce_counts(counts=c, encoding=pce_encoding)

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=(ansatz.num_parameters,))

    LOGGER.debug(
        "PCE optimizer start | qpu_mode=%s qpus=%s shots=%s maxiter=%s pce_k=%s encoded_qubits=%s logical_vars=%s",
        scheduler.mode,
        scheduler.qpu_ids,
        shots,
        maxiter,
        None if pce_encoding is None else int(pce_encoding.compression_k),
        int(ansatz.num_qubits),
        int(objective.num_qubits),
    )

    trace: list[dict[str, Any]] = []
    qpu_usage: dict[str, int] = {}
    best_value = float("inf")
    best_theta = [float(x) for x in theta0]
    best_bitstring = "0" * int(objective.num_qubits)
    best_energy = float("inf")
    best_counts: dict[str, int] = {}
    eval_counter = 0

    def _objective_fn(theta: np.ndarray) -> float:
        nonlocal best_value, best_theta, best_counts, best_bitstring, best_energy, eval_counter
        eval_counter += 1

        max_retries = max(len(scheduler._all_qpu_ids), 2)
        last_error: Exception | None = None
        for attempt in range(max_retries):
            if not scheduler.has_available_qpus:
                LOGGER.warning(
                    "All QPUs offline at eval=%s. Forcing window refresh...",
                    eval_counter,
                )
                scheduler._maybe_refresh_windows()
                if not scheduler.has_available_qpus:
                    raise RuntimeError("No QPUs available — all availability windows are closed.")

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
                    objective_mode="expectation",
                    cvar_alpha=1.0,
                    counts_transform=counts_transform,
                )
                break
            except QPUWindowExpiredError as exc:
                last_error = exc
                LOGGER.warning(
                    "PCE: QPU '%s' window expired at eval=%s (attempt %d/%d). Trying next QPU...",
                    qpu_id, eval_counter, attempt + 1, max_retries,
                )
                scheduler.mark_offline(qpu_id)
                continue
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "PCE: QPU '%s' failed at eval=%s (attempt %d/%d): %s  Trying next QPU...",
                    qpu_id, eval_counter, attempt + 1, max_retries, str(exc)[:200],
                )
                continue
        else:
            raise RuntimeError(f"All QPUs failed at eval={eval_counter}. Last error: {last_error}") from last_error

        value = float(result["objective_value"])
        qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + 1

        LOGGER.debug(
            "PCE evaluation | eval=%s qpu_id=%s objective=%.8f best_sample_energy=%.8f",
            eval_counter,
            qpu_id,
            value,
            float(result["best_energy"]),
        )

        trace.append({
            "evaluation": eval_counter,
            "qpu_id": qpu_id,
            "objective_value": value,
            "best_sample_energy": float(result["best_energy"]),
            "best_sample_bitstring": result["best_bitstring"],
            "metadata": result["metadata"],
            "top_counts": _top_counts(result["counts"]),
        })

        if value < best_value:
            best_value = value
            best_theta = [float(x) for x in result["theta"]]
            best_counts = dict(result["counts"])
            best_bitstring = str(result["best_bitstring"])
            best_energy = float(result["best_energy"])
            LOGGER.debug(
                "New PCE incumbent | eval=%s qpu_id=%s objective=%.8f bitstring=%s",
                eval_counter, qpu_id, best_value, best_bitstring,
            )
        return value

    minimize_result = minimize(
        fun=_objective_fn,
        x0=theta0,
        method="COBYLA",
        options={"maxiter": int(maxiter), "rhobeg": 0.4},
    )

    LOGGER.debug(
        "PCE optimizer complete | status=%s evaluations=%s best_objective=%.8f",
        minimize_result.status,
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
    try:
        from qiskit.transpiler import generate_preset_pass_manager
    except Exception:
        LOGGER.warning(
            "PCE Qiskit-estimator implementation unavailable (missing transpiler). "
            "Falling back to legacy counts-based PCE."
        )
        return _run_pce_method_legacy_counts(
            manager=manager,
            scheduler=scheduler,
            ansatz=ansatz,
            objective=objective,
            shots=shots,
            maxiter=maxiter,
            seed=seed,
            population_size=population_size,
            elite_frac=elite_frac,
            timeout_sec=timeout_sec,
            parallel_workers=parallel_workers,
            batch_size=batch_size,
            pce_encoding=pce_encoding,
        )

    # If non-Qiskit providers are selected, preserve previous behaviour.
    selected_providers: set[str] = set()
    for qpu_id in scheduler.qpu_ids:
        qpu = manager.qpus.get(qpu_id)
        provider = str(getattr(qpu, "provider", "")).strip()
        if provider:
            selected_providers.add(provider)
    unsupported = [p for p in selected_providers if p not in {"local_qiskit", "ibm"}]
    if unsupported:
        LOGGER.warning(
            "PCE Qiskit-estimator path supports only local_qiskit/ibm providers. "
            "Selected providers=%s. Falling back to legacy counts-based PCE.",
            sorted(selected_providers),
        )
        return _run_pce_method_legacy_counts(
            manager=manager,
            scheduler=scheduler,
            ansatz=ansatz,
            objective=objective,
            shots=shots,
            maxiter=maxiter,
            seed=seed,
            population_size=population_size,
            elite_frac=elite_frac,
            timeout_sec=timeout_sec,
            parallel_workers=parallel_workers,
            batch_size=batch_size,
            pce_encoding=pce_encoding,
        )

    qubo = objective.qubo
    weight_matrix = _qubo_to_maxcut_weight_matrix(qubo)
    edges = _maxcut_edges_from_weight_matrix(weight_matrix)
    num_nodes = int(weight_matrix.shape[0])
    qubo_num_vars = int(objective.num_qubits)

    compression_k = int(
        (pce_encoding.compression_k if pce_encoding is not None else 2)
    )
    encoded_qubits = estimate_pce_num_qubits(
        num_variables=int(num_nodes),
        compression_k=int(compression_k),
    )
    if int(ansatz.num_qubits) != int(encoded_qubits):
        LOGGER.warning(
            "PCE ansatz qubit mismatch detected (bundle=%s, notebook_impl=%s). "
            "Using notebook_impl qubit count.",
            int(ansatz.num_qubits),
            int(encoded_qubits),
        )

    reps = int(ansatz.ansatz_reps) if ansatz.ansatz_reps is not None else 2
    if reps <= 0:
        reps = 2

    try:
        from qiskit.circuit.library import efficient_su2

        pce_qiskit_ansatz = efficient_su2(
            int(encoded_qubits),
            ["ry", "rz"],
            reps=int(reps),
        )
    except Exception:
        from qiskit.circuit.library import EfficientSU2

        pce_qiskit_ansatz = EfficientSU2(
            num_qubits=int(encoded_qubits),
            su2_gates=["ry", "rz"],
            entanglement="linear",
            reps=int(reps),
        )
    pce_qiskit_ansatz = _strip_final_measurements(pce_qiskit_ansatz.decompose())

    pauli_strings = generate_pce_pauli_strings(
        num_qubits=int(encoded_qubits),
        num_variables=int(num_nodes),
        compression_k=int(compression_k),
    )
    observable_blocks = _group_reversed_pce_pauli_operators(pauli_strings)

    alpha = float(encoded_qubits)
    beta = 0.5
    nu = _pce_weighted_nu(int(num_nodes), edges)

    LOGGER.debug(
        "PCE optimizer start | impl=qiskit_estimator qpu_mode=%s qpus=%s shots=%s maxiter=%s "
        "pce_k=%s encoded_qubits=%s maxcut_nodes=%s qubo_vars=%s reps=%s",
        scheduler.mode,
        scheduler.qpu_ids,
        shots,
        maxiter,
        compression_k,
        encoded_qubits,
        num_nodes,
        qubo_num_vars,
        reps,
    )

    def _set_nested_attr(target: Any, dotted_name: str, value: Any) -> bool:
        current = target
        parts = dotted_name.split(".")
        for name in parts[:-1]:
            if not hasattr(current, name):
                return False
            try:
                current = getattr(current, name)
            except Exception:
                return False
        leaf = parts[-1]
        if not hasattr(current, leaf):
            return False
        try:
            setattr(current, leaf, value)
        except Exception:
            return False
        return True

    def _configure_estimator_defaults(target: Any) -> None:
        if target is None:
            return
        _set_nested_attr(target, "options.default_shots", int(shots))
        _set_nested_attr(target, "options.default_precision", 0.05)
        try:
            setattr(target, "default_shots", int(shots))
        except Exception:
            pass

    def _configure_ibm_runtime_options(target: Any) -> None:
        pass

    estimator_cache: dict[str, dict[str, Any]] = {}

    def _build_context_for_qpu(qpu_id: str) -> dict[str, Any]:
        if qpu_id in estimator_cache:
            return estimator_cache[qpu_id]

        qpu = manager.qpus.get(qpu_id)
        if qpu is None:
            raise RuntimeError(f"Unknown QPU id '{qpu_id}'")
        provider = str(getattr(qpu, "provider", "")).strip()

        backend = None
        backend_name = qpu_id
        metadata_extra: dict[str, Any] = {}

        if provider == "local_qiskit":
            try:
                from qiskit_aer import AerSimulator
                from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
            except Exception as exc:
                raise ModuleNotFoundError(
                    "Local PCE estimator requires qiskit-aer primitives."
                ) from exc

            backend = AerSimulator(method="matrix_product_state")
            backend_name = "aer_simulator_mps"
            estimator = AerEstimatorV2()
        elif provider == "ibm":
            try:
                from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimatorV2
            except Exception as exc:
                raise ModuleNotFoundError(
                    "IBM PCE estimator requires qiskit-ibm-runtime."
                ) from exc

            manager._refresh_ibm_usage_if_needed(force=True)
            if not bool(getattr(qpu, "is_available", True)):
                raise RuntimeError(f"IBM unavailable: {getattr(qpu, 'last_error', 'unknown')}")
            backend_info = manager._select_ibm_backend_info(int(encoded_qubits))
            if backend_info is None:
                raise RuntimeError(
                    f"No IBM backend supports encoded PCE size ({encoded_qubits} qubits)."
                )
            backend = backend_info["backend"]
            backend_name = str(backend_info.get("name", getattr(backend, "name", qpu_id)))
            metadata_extra["pending_jobs"] = backend_info.get("pending_jobs")
            metadata_extra["backend_qubits"] = backend_info.get("num_qubits")
            LOGGER.info(
                "Selected IBM backend | selection=pce backend=%s qubits=%s pending_jobs=%s",
                backend_name,
                backend_info.get("num_qubits"),
                backend_info.get("pending_jobs"),
            )

            raw_estimator = RuntimeEstimatorV2(mode=backend)
            _configure_estimator_defaults(raw_estimator)
            _configure_ibm_runtime_options(raw_estimator)
            estimator = _QRAOPrimitiveWrapper(
                raw_estimator,
                label="estimator",
                manager=manager,
                qpu_id=qpu_id,
                backend_name=backend_name,
                qpu=qpu,
                num_qubits=int(encoded_qubits),
                primitive_cls=RuntimeEstimatorV2,
                configure_fns=[_configure_estimator_defaults, _configure_ibm_runtime_options],
                algo_name="PCE",
            )
            metadata_extra["runtime_guard"] = "qrao_primitive_wrapper"
        else:
            raise RuntimeError(
                f"PCE Qiskit-estimator implementation does not support provider '{provider}'."
            )

        pass_manager = generate_preset_pass_manager(
            optimization_level=3,
            backend=backend,
        )
        transpiled_ansatz = pass_manager.run(pce_qiskit_ansatz)

        layout = getattr(transpiled_ansatz, "layout", None)
        hamiltonian_blocks: list[list[Any]] = []
        for block in observable_blocks:
            prepared_block: list[Any] = []
            for op in block:
                if layout is None:
                    prepared_block.append(op)
                    continue
                try:
                    prepared_block.append(op.apply_layout(layout))
                except Exception:
                    prepared_block.append(op)
            hamiltonian_blocks.append(prepared_block)

        context = {
            "provider": provider,
            "qpu_id": qpu_id,
            "backend_name": backend_name,
            "metadata_extra": metadata_extra,
            "estimator": estimator,
            "ansatz": transpiled_ansatz,
            "hamiltonian_blocks": hamiltonian_blocks,
        }
        estimator_cache[qpu_id] = context
        return context

    def _evaluate_theta_qiskit_pce(theta: np.ndarray, qpu_id: str) -> dict[str, Any]:
        theta_list = [float(value) for value in theta]
        context = _build_context_for_qpu(qpu_id)

        pubs = []
        for block in context["hamiltonian_blocks"]:
            if block:
                pubs.append((context["ansatz"], block, theta_list))
        if not pubs:
            raise RuntimeError("PCE Hamiltonian is empty; cannot evaluate objective.")

        submit_t0 = time.time()
        job = context["estimator"].run(pubs)
        job_id = getattr(job, "job_id", None)
        if callable(job_id):
            try:
                job_id = job_id()
            except Exception:
                job_id = None
        result = job.result()
        elapsed_sec = float(time.time() - submit_t0)
        current_backend = str(context["backend_name"])
        estimator_meta = context["estimator"]
        if hasattr(estimator_meta, "backend_name"):
            try:
                current_backend = str(getattr(estimator_meta, "backend_name"))
            except Exception:
                current_backend = str(context["backend_name"])

        node_exp = np.zeros(int(num_nodes), dtype=float)
        exp_idx = 0
        for pub_result in result:
            evs = None
            try:
                evs = getattr(pub_result.data, "evs", None)
            except Exception:
                evs = None
            if evs is None:
                continue
            for ev in np.atleast_1d(evs).reshape(-1):
                if exp_idx < len(node_exp):
                    node_exp[exp_idx] = float(np.real(ev))
                exp_idx += 1

        edge_loss = 0.0
        for u, v, w in edges:
            edge_loss += float(w) * float(np.tanh(alpha * node_exp[u])) * float(np.tanh(alpha * node_exp[v]))

        reg_term = np.tanh(alpha * node_exp) ** 2
        reg_term = float(np.mean(reg_term) ** 2)
        reg_loss = float(beta * nu * reg_term)
        total_loss = float(edge_loss + reg_loss)

        maxcut_bits = [1 if float(node_exp[i]) >= 0.0 else 0 for i in range(int(num_nodes))]
        qubo_bitstring = _maxcut_to_qubo_bitstring(maxcut_bits)
        if len(qubo_bitstring) < qubo_num_vars:
            qubo_bitstring = qubo_bitstring.zfill(qubo_num_vars)
        elif len(qubo_bitstring) > qubo_num_vars:
            qubo_bitstring = qubo_bitstring[-qubo_num_vars:]

        best_energy = float(objective.energy(qubo_bitstring))
        pseudo_counts = {qubo_bitstring: int(shots)}
        node_exp_map = {int(i): float(node_exp[i]) for i in range(int(num_nodes))}
        eval_metadata: dict[str, Any] | None = None
        if hasattr(estimator_meta, "pop_oldest_completed_metadata"):
            try:
                popped = estimator_meta.pop_oldest_completed_metadata()
                if isinstance(popped, dict):
                    eval_metadata = dict(popped)
            except Exception:
                eval_metadata = None
        if eval_metadata is None:
            eval_metadata = {}

        metadata = dict(eval_metadata)
        metadata.setdefault("provider", context["provider"])
        metadata.setdefault("qpu_id", qpu_id)
        metadata.setdefault("backend_name", current_backend)
        metadata.setdefault("shots", int(shots))
        if job_id is not None and not str(metadata.get("job_id", "")).strip():
            metadata["job_id"] = str(job_id)
        if "pending_jobs" not in metadata:
            pending_jobs = context.get("metadata_extra", {}).get("pending_jobs")
            if pending_jobs is not None:
                metadata["pending_jobs"] = pending_jobs
        metadata["elapsed_sec"] = float(elapsed_sec)
        metadata["num_observables"] = int(num_nodes)
        if "runtime_guard" not in metadata:
            runtime_guard = context.get("metadata_extra", {}).get("runtime_guard")
            if runtime_guard is not None:
                metadata["runtime_guard"] = runtime_guard

        return {
            "qpu_id": qpu_id,
            "theta": theta_list,
            "counts": pseudo_counts,
            "raw_counts": pseudo_counts,
            "metadata": metadata,
            "objective_value": float(total_loss),
            "best_bitstring": str(qubo_bitstring),
            "best_energy": float(best_energy),
            "edge_loss": float(edge_loss),
            "regularization_loss": float(reg_loss),
            "node_exp_map": node_exp_map,
        }

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 1.0, size=(int(pce_qiskit_ansatz.num_parameters),))

    trace: list[dict[str, Any]] = []
    qpu_usage: dict[str, int] = {}
    best_value = float("inf")
    best_theta = [float(x) for x in theta0]
    best_bitstring = "0" * int(qubo_num_vars)
    best_energy = float("inf")
    best_counts: dict[str, int] = {}
    eval_counter = 0

    def _objective_fn(theta: np.ndarray) -> float:
        nonlocal best_value, best_theta, best_counts, best_bitstring, best_energy, eval_counter
        eval_counter += 1

        max_retries = max(len(scheduler._all_qpu_ids), 2)
        last_error: Exception | None = None
        result: dict[str, Any] | None = None
        qpu_id = ""

        for attempt in range(max_retries):
            if not scheduler.has_available_qpus:
                LOGGER.warning(
                    "All QPUs offline at eval=%s. Forcing window refresh...",
                    eval_counter,
                )
                scheduler._maybe_refresh_windows()
                if not scheduler.has_available_qpus:
                    raise RuntimeError("No QPUs available - all availability windows are closed.")

            qpu_id = scheduler.next_qpu()
            try:
                result = _evaluate_theta_qiskit_pce(theta, qpu_id)
                break
            except QPUWindowExpiredError as exc:
                last_error = exc
                LOGGER.warning(
                    "PCE: QPU '%s' window expired at eval=%s (attempt %d/%d). Trying next QPU...",
                    qpu_id,
                    eval_counter,
                    attempt + 1,
                    max_retries,
                )
                scheduler.mark_offline(qpu_id)
                continue
            except Exception as exc:
                last_error = exc
                LOGGER.warning(
                    "PCE: QPU '%s' failed at eval=%s (attempt %d/%d): %s  Trying next QPU...",
                    qpu_id,
                    eval_counter,
                    attempt + 1,
                    max_retries,
                    str(exc)[:200],
                )
                continue
        else:
            raise RuntimeError(
                f"All QPUs failed at eval={eval_counter}. Last error: {last_error}"
            ) from last_error

        if result is None:
            raise RuntimeError("PCE evaluation returned no result.")

        value = float(result["objective_value"])
        qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + 1

        LOGGER.debug(
            "PCE (Qiskit-estimator) evaluation | eval=%s qpu_id=%s objective=%.8f "
            "edge=%.8f reg=%.8f best_sample_energy=%.8f",
            eval_counter,
            qpu_id,
            value,
            float(result["edge_loss"]),
            float(result["regularization_loss"]),
            float(result["best_energy"]),
        )

        trace.append(
            {
                "evaluation": int(eval_counter),
                "qpu_id": qpu_id,
                "objective_value": value,
                "edge_loss": float(result["edge_loss"]),
                "regularization_loss": float(result["regularization_loss"]),
                "best_sample_energy": float(result["best_energy"]),
                "best_sample_bitstring": result["best_bitstring"],
                "metadata": result["metadata"],
                "top_counts": _top_counts(result["counts"]),
                "node_expectations": result["node_exp_map"],
            }
        )

        if value < best_value:
            best_value = value
            best_theta = [float(x) for x in result["theta"]]
            best_counts = dict(result["counts"])
            best_bitstring = str(result["best_bitstring"])
            best_energy = float(result["best_energy"])
            LOGGER.debug(
                "New PCE incumbent | eval=%s qpu_id=%s objective=%.8f bitstring=%s",
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
        "PCE optimizer complete | impl=qiskit_estimator status=%s evaluations=%s best_objective=%.8f",
        minimize_result.status,
        eval_counter,
        best_value,
    )

    # Mirror QRAO metadata draining so async completions are preserved.
    for cached_qpu_id, context in estimator_cache.items():
        primitive = context.get("estimator")
        if not hasattr(primitive, "drain_completed_metadata"):
            continue
        try:
            remaining_meta = primitive.drain_completed_metadata()
        except Exception:
            remaining_meta = []
        for meta in remaining_meta:
            if not isinstance(meta, dict):
                continue
            trace.append(
                {
                    "pce_job_phase": "estimator",
                    "qpu_id": str(meta.get("qpu_id", cached_qpu_id)),
                    "backend_name": str(
                        meta.get("backend_name", context.get("backend_name", cached_qpu_id))
                    ),
                    "metadata": meta,
                }
            )

    if not best_counts:
        best_counts = {str(best_bitstring): int(shots)}

    return OptimizationResult(
        method="pce",
        objective_mode="pce_qiskit_loss",
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


class _QRAOPrimitiveWrapper:
    """Transparent wrapper around an IBM Runtime primitive (Estimator/Sampler).

    Intercepts ``.run()`` calls to add per-job logging and periodic IBM
    budget checks.  When the current credential's budget drops below the
    threshold, the wrapper triggers credential rotation via the hardware
    manager, obtains a new backend, **rebuilds** the inner primitive on
    that backend (with the same configuration), and continues seamlessly
    — giving runtime-estimator methods the same credential-rotation
    behaviour that VQE/QAOA/PCE get automatically through ``manager.run_counts()``.
    """

    _BUDGET_CHECK_INTERVAL = 1  # check budget on EVERY primitive call

    def __init__(
        self,
        inner: Any,
        *,
        label: str,
        manager: Any,
        qpu_id: str,
        backend_name: str,
        qpu: Any,
        num_qubits: int,
        primitive_cls: type,
        configure_fns: list,
        algo_name: str = "QRAO",
    ) -> None:
        self._inner = inner
        self._label = label
        self._manager = manager
        self._qpu_id = qpu_id
        self._backend_name = backend_name
        self._qpu = qpu
        self._num_qubits = num_qubits
        self._primitive_cls = primitive_cls
        self._configure_fns = configure_fns
        self._algo_name = str(algo_name)
        self._job_count = 0
        self._completed_job_metadata: list[dict[str, Any]] = []
        self._metadata_lock = threading.Lock()

        # Store the qubit count of the *initial* backend so that dynamic
        # load-balancing only switches between same-topology machines.
        # e.g. ibm_fez (156q Heron) and ibm_marrakesh (156q Heron) share a
        # topology, but ibm_torino (133q Eagle) does NOT — switching to it
        # mid-optimisation causes ISA / coupling-map mismatches and crashes.
        self._backend_qubit_count: int | None = None
        for info in getattr(manager, "ibm_backends", []):
            if str(info.get("name", "")) == backend_name:
                self._backend_qubit_count = int(info.get("num_qubits", 0))
                break

    @property
    def job_count(self) -> int:
        return self._job_count

    @property
    def backend_name(self) -> str:
        return self._backend_name

    def _record_completed_metadata(self, metadata: dict[str, Any]) -> None:
        with self._metadata_lock:
            self._completed_job_metadata.append(dict(metadata))

    def pop_oldest_completed_metadata(self) -> dict[str, Any] | None:
        with self._metadata_lock:
            if not self._completed_job_metadata:
                return None
            return dict(self._completed_job_metadata.pop(0))

    def drain_completed_metadata(self) -> list[dict[str, Any]]:
        with self._metadata_lock:
            records = [dict(item) for item in self._completed_job_metadata]
            self._completed_job_metadata.clear()
            return records

    # Forward attribute access to the wrapped primitive so that
    # qiskit-optimization / qiskit-algorithms can inspect options,
    # set_options(), etc.
    def __getattr__(self, name: str) -> Any:
        return getattr(self._inner, name)

    def _lookup_backend_pending_jobs(self, backend_name: str) -> Any:
        for info in getattr(self._manager, "ibm_backends", []):
            try:
                if str(info.get("name", "")) == str(backend_name):
                    return info.get("pending_jobs")
            except Exception:
                continue
        return None

    @staticmethod
    def _extract_circuit_from_run_args(args: tuple[Any, ...]) -> Any | None:
        if not args:
            return None
        first = args[0]
        first_pub = first[0] if isinstance(first, (list, tuple)) and first else first
        if hasattr(first_pub, "circuit"):
            try:
                return getattr(first_pub, "circuit")
            except Exception:
                pass
        if isinstance(first_pub, (list, tuple)) and first_pub:
            circuit = first_pub[0]
            if hasattr(circuit, "data"):
                return circuit
        if hasattr(first_pub, "data"):
            return first_pub
        return None

    @staticmethod
    def _collect_circuit_metrics(circuit: Any) -> dict[str, Any]:
        if circuit is None:
            return {}
        metrics: dict[str, Any] = {}
        try:
            depth = circuit.depth()
            if depth is not None:
                metrics["transpiled_depth"] = int(depth)
        except Exception:
            pass
        one_q = 0
        two_q = 0
        meas = 0
        try:
            for item in getattr(circuit, "data", []):
                try:
                    inst, qargs, _ = item
                except Exception:
                    inst = getattr(item, "operation", None)
                    qargs = getattr(item, "qubits", ())
                name = str(getattr(inst, "name", ""))
                qlen = len(qargs) if qargs is not None else 0
                if name == "measure":
                    meas += 1
                elif qlen == 1:
                    one_q += 1
                elif qlen == 2:
                    two_q += 1
            metrics["transpiled_1q_gates"] = int(one_q)
            metrics["transpiled_2q_gates"] = int(two_q)
            metrics["transpiled_measurements"] = int(meas)
        except Exception:
            pass
        return metrics

    def _resolve_shots(self, kwargs: dict[str, Any]) -> int | None:
        shots = kwargs.get("shots")
        if shots is None:
            options = getattr(self._inner, "options", None)
            for attr in ("shots", "default_shots"):
                try:
                    candidate = getattr(options, attr)
                except Exception:
                    candidate = None
                if candidate is not None:
                    shots = candidate
                    break
        if shots is None:
            try:
                shots = getattr(self._inner, "default_shots", None)
            except Exception:
                shots = None
        try:
            return int(shots) if shots is not None else None
        except Exception:
            return None

    def _rebuild_primitive_on_new_backend(self) -> bool:
        """Rotate credentials and rebuild the inner primitive on a new backend.

        After credential rotation, we prefer the **same backend name** as
        the original (e.g. ``ibm_torino``) because the VQE's pass_manager
        transpiled the ansatz circuit for that specific backend's coupling
        map.  Switching to a different backend would cause transpilation
        mismatches and job failures.

        If all credentials are exhausted, the method will:
        1. Reload the credential JSON file (to discover newly added accounts)
        2. Wait and retry — mirroring VQE/QAOA's ``refresh_credentials_if_needed()``

        Returns True on success, False if rotation failed after all retries.
        """
        _MAX_RETRIES = 10
        _RETRY_WAIT_SECONDS = 30

        for attempt in range(1, _MAX_RETRIES + 1):
            # Step 1: Reload credential pool from disk (picks up new entries
            # added to the JSON file, just like VQE's refresh_credentials_if_needed).
            try:
                self._manager._load_ibm_credential_pool(force=False)
            except Exception:
                pass

            # Step 2: Refresh usage — this will rotate credentials if needed.
            try:
                self._manager._refresh_ibm_usage_if_needed(force=True)
            except Exception:
                pass

            # Step 3: Check if QPU is available after rotation.
            if self._qpu.is_available:
                break

            # QPU still unavailable — wait and retry (user may add new credentials).
            if attempt < _MAX_RETRIES:
                LOGGER.info(
                    "%s %s: All credentials exhausted (attempt %d/%d). "
                    "Waiting %ds for new credentials in %s ...",
                    self._algo_name,
                    self._label,
                    attempt,
                    _MAX_RETRIES,
                    _RETRY_WAIT_SECONDS,
                    self._manager.ibm_credentials_json,
                )
                time.sleep(_RETRY_WAIT_SECONDS)
            else:
                LOGGER.warning(
                    "%s %s: All credentials exhausted after %d retries.",
                    self._algo_name,
                    self._label,
                    _MAX_RETRIES,
                )
                return False

        # --- Find the SAME backend name from the (now rotated) credential ---
        # All IBM credentials give access to the same physical backends; only
        # the runtime budget differs per CRN instance.
        target_name = self._backend_name
        new_backend = None
        new_name = None

        # Search through available backends for a match by name
        for info in self._manager.ibm_backends:
            if str(info.get("name", "")) == target_name:
                if int(info.get("num_qubits", 0)) >= self._num_qubits:
                    new_backend = info["backend"]
                    new_name = target_name
                    LOGGER.info(
                        "Selected IBM backend | selection=runtime_same_after_rotation "
                        "backend=%s qubits=%s pending_jobs=%s",
                        target_name,
                        info.get("num_qubits"),
                        info.get("pending_jobs"),
                    )
                    break

        # Fallback: if the original backend isn't found, try any compatible one
        if new_backend is None:
            backend_info = self._manager._select_ibm_backend_info(self._num_qubits)
            if backend_info is None:
                return False
            new_backend = backend_info["backend"]
            new_name = str(backend_info.get("name", getattr(new_backend, "name", self._qpu_id)))
            LOGGER.warning(
                "%s %s: Original backend '%s' not found after rotation; "
                "using '%s' instead (may cause transpilation issues)",
                self._algo_name,
                self._label,
                target_name,
                new_name,
            )

        # Create a fresh primitive on the (same) backend
        new_inner = self._primitive_cls(mode=new_backend)
        for fn in self._configure_fns:
            fn(new_inner)

        old_name = self._backend_name
        self._inner = new_inner
        self._backend_name = new_name

        LOGGER.info(
            "%s %s: Credential rotated | old_backend=%s new_backend=%s",
            self._algo_name,
            self._label,
            old_name,
            new_name,
        )
        return True

    def run(self, *args: Any, **kwargs: Any) -> Any:
        self._job_count += 1

        # --- Pre-dispatch budget check with credential rotation ---
        runtime_budget_str = ""
        if self._qpu.provider == "ibm":
            remaining: float | None = None
            try:
                from qobench.hardware_manager import _get_ibm_usage_remaining_seconds
                # Always force a fresh check so we never rely on stale data.
                self._manager._refresh_ibm_usage_if_needed(force=True)
                service = self._manager.sessions.get("ibm_quantum")
                if service is not None:
                    remaining = _get_ibm_usage_remaining_seconds(service)
            except Exception:
                pass

            # Use the manager's configured threshold (set from CLI --ibm-min-runtime-seconds)
            # NOT the module-level constant IBM_MIN_RUNTIME_SECONDS which may differ.
            threshold = float(getattr(self._manager, "ibm_min_runtime_seconds", 15.0))

            if remaining is not None:
                runtime_budget_str = f" | IBM budget: {remaining:.0f}s left"
                if remaining < threshold:
                    LOGGER.warning(
                        "%s %s: IBM budget low (%.0fs < %.0fs) at job=%s. "
                        "Rotating credential...",
                        self._algo_name,
                        self._label,
                        remaining,
                        threshold,
                        self._job_count,
                    )
                    if self._rebuild_primitive_on_new_backend():
                        # Refresh budget display after rotation
                        try:
                            service = self._manager.sessions.get("ibm_quantum")
                            if service is not None:
                                new_remaining = _get_ibm_usage_remaining_seconds(service)
                                if new_remaining is not None:
                                    runtime_budget_str = f" | IBM budget: {new_remaining:.0f}s left"
                        except Exception:
                            pass
                    else:
                        LOGGER.warning(
                            "%s %s: All credentials exhausted at job=%s. "
                            "Stopping optimization.",
                            self._algo_name,
                            self._label,
                            self._job_count,
                        )
                        raise RuntimeError(
                            f"{self._algo_name} stopped: All IBM credentials exhausted "
                            f"at {self._label} job={self._job_count}."
                        )

        # Ensure we always select the least busy backend on a per-job basis.
        # IMPORTANT: only consider backends with the SAME qubit count as the
        # original to avoid topology / coupling-map mismatches (e.g. switching
        # from ibm_fez 156q Heron to ibm_torino 133q Eagle causes ISA failures).
        if self._qpu.provider == "ibm" and getattr(self._manager, "ibm_backends_preferred", None):
            best_backend = None
            best_name = self._backend_name
            best_pending = float('inf')
            required_backend_qubits = self._backend_qubit_count

            for info in self._manager.ibm_backends_preferred:
                bq = int(info.get("num_qubits", 0))
                # Must have enough qubits for the circuit AND must match the
                # original backend's qubit count to ensure same topology.
                if bq >= self._num_qubits and (required_backend_qubits is None or bq == required_backend_qubits):
                    backend = info["backend"]
                    try:
                        pending = getattr(backend.status(), "pending_jobs", float('inf'))
                        if isinstance(info, dict):
                            info["pending_jobs"] = pending
                        if pending < best_pending:
                            best_pending = pending
                            best_backend = backend
                            best_name = str(getattr(backend, "name", ""))
                    except Exception:
                        pass

            if best_backend is not None and best_name != self._backend_name and best_name != "":
                LOGGER.info(
                    "%s %s: Switching from %s to least busy backend %s (%d pending) at job=%d",
                    self._algo_name,
                    self._label,
                    self._backend_name,
                    best_name,
                    best_pending,
                    self._job_count,
                )
                new_inner = self._primitive_cls(mode=best_backend)
                for fn in self._configure_fns:
                    fn(new_inner)
                self._inner = new_inner
                self._backend_name = best_name

        LOGGER.info(
            "Dispatching %s %s job #%s -> %s | backend=%s%s",
            self._algo_name,
            self._label,
            self._job_count,
            self._qpu_id,
            self._backend_name,
            runtime_budget_str,
        )

        submit_start = time.time()
        submit_time_utc = datetime.now(timezone.utc).isoformat()
        dispatch_backend = self._backend_name
        job_metadata: dict[str, Any] = {
            "provider": str(getattr(self._qpu, "provider", "")),
            "qpu_id": self._qpu_id,
            "backend_name": dispatch_backend,
            "submit_time_utc": submit_time_utc,
        }
        if self._backend_qubit_count is not None:
            job_metadata["backend_qubits"] = int(self._backend_qubit_count)
        pending_jobs = self._lookup_backend_pending_jobs(dispatch_backend)
        if pending_jobs is not None:
            job_metadata["pending_jobs_at_submit"] = pending_jobs
        shots = self._resolve_shots(kwargs)
        if shots is not None:
            job_metadata["shots"] = int(shots)
        job_metadata.update(
            self._collect_circuit_metrics(
                self._extract_circuit_from_run_args(args)
            )
        )
        try:
            import warnings as _warnings
            from qobench.hardware_manager import _IBM_USAGE_LIMIT_WARNING_SUBSTR
            usage_limit_hit = False
            with _warnings.catch_warnings(record=True) as caught_warnings:
                _warnings.simplefilter("always")
                job = self._inner.run(*args, **kwargs)
            for w in caught_warnings:
                msg = str(w.message)
                if _IBM_USAGE_LIMIT_WARNING_SUBSTR in msg:
                    usage_limit_hit = True
                    LOGGER.warning(
                        "%s %s: Caught 'usage limit' warning during .run() at job=%s: %s",
                        self._algo_name,
                        self._label,
                        self._job_count,
                        msg[:200],
                    )
                    break
            if usage_limit_hit:
                LOGGER.info(
                    "%s %s: Usage limit hit — rotating credential and retrying job %s...",
                    self._algo_name,
                    self._label,
                    self._job_count,
                )
                if self._rebuild_primitive_on_new_backend():
                    job = self._inner.run(*args, **kwargs)
                else:
                    raise RuntimeError(
                        f"{self._algo_name} stopped: All IBM credentials exhausted "
                        f"at {self._label} job={self._job_count} (usage limit warning)."
                    )
            job_id = getattr(job, "job_id", None)
            if callable(job_id):
                job_id = job_id()
            if job_id is not None:
                job_metadata["job_id"] = str(job_id)
        except Exception as exc:
            job_metadata["fallback_path"] = "submit_exception"
            LOGGER.warning(
                "%s %s: QPU job #%s FAILED on %s (backend=%s): %s. "
                "Falling back to local MPS simulator...",
                self._algo_name,
                self._label,
                self._job_count,
                self._qpu_id,
                self._backend_name,
                exc,
            )
            try:
                job = self._run_mps_fallback(*args, **kwargs)
                job_metadata["provider"] = "local_qiskit"
                job_metadata["qpu_id"] = "local_qiskit"
                job_metadata["backend_name"] = "aer_simulator_mps"
            except Exception as mps_exc:
                job_metadata["fallback_path"] = "penalty_result"
                LOGGER.warning(
                    "%s %s: QPU job #%s MPS fallback also FAILED at submit: %s. "
                    "Will return penalty value.",
                    self._algo_name,
                    self._label,
                    self._job_count,
                    mps_exc,
                )
                # Wrap penalty result in a mock job so _JobWrapper can call .result()
                penalty_res = self._make_penalty_result(*args)
                class _PenaltyJob:
                    def result(self, *a, **kw):
                        return penalty_res
                    def __getattr__(self, name):
                        raise AttributeError(name)
                job = _PenaltyJob()

        # Wrap the returned job so we can measure total time (submit + QPU
        # execution + result retrieval) — matching VQE's "Job completed on
        # ibm_quantum in Xs" timing.
        wrapper_ref = self
        job_num = self._job_count
        backend_at_dispatch = str(job_metadata.get("backend_name", self._backend_name))
        run_args = args
        run_kwargs = kwargs

        class _JobWrapper:
            """Transparent job wrapper that logs QPU time on .result()."""

            def __init__(self, inner_job: Any, metadata: dict[str, Any]) -> None:
                self._inner_job = inner_job
                self._submit_time = submit_start
                self._result_logged = False
                self._metadata = dict(metadata)

            def result(self, *a: Any, **kw: Any) -> Any:
                try:
                    import warnings as _w
                    from qobench.hardware_manager import _IBM_USAGE_LIMIT_WARNING_SUBSTR
                    with _w.catch_warnings(record=True) as _cw:
                        _w.simplefilter("always")
                        res = self._inner_job.result(*a, **kw)
                    for _wn in _cw:
                        _msg = str(_wn.message)
                        if _IBM_USAGE_LIMIT_WARNING_SUBSTR in _msg:
                            LOGGER.warning(
                                "%s %s job #%s: 'usage limit' warning on result(). "
                                "Proactively rotating credential for next job.",
                                wrapper_ref._algo_name,
                                wrapper_ref._label,
                                job_num,
                            )
                            try:
                                wrapper_ref._manager._refresh_ibm_usage_if_needed(force=True)
                            except Exception:
                                pass
                            try:
                                wrapper_ref._rebuild_primitive_on_new_backend()
                            except Exception:
                                pass
                            break
                except Exception as exc:
                    self._metadata["fallback_path"] = "result_exception"
                    LOGGER.warning(
                        "%s %s job #%s result() FAILED on %s (backend=%s): %s. "
                        "Falling back to local MPS simulator...",
                        wrapper_ref._algo_name,
                        wrapper_ref._label,
                        job_num,
                        wrapper_ref._qpu_id,
                        backend_at_dispatch,
                        exc,
                    )
                    try:
                        fallback_job = wrapper_ref._run_mps_fallback(*run_args, **run_kwargs)
                        res = fallback_job.result()
                        self._metadata["provider"] = "local_qiskit"
                        self._metadata["qpu_id"] = "local_qiskit"
                        self._metadata["backend_name"] = "aer_simulator_mps"
                        fb_job_id = getattr(fallback_job, "job_id", None)
                        if callable(fb_job_id):
                            fb_job_id = fb_job_id()
                        if fb_job_id is not None:
                            self._metadata["job_id"] = str(fb_job_id)
                    except Exception as mps_exc:
                        self._metadata["fallback_path"] = "penalty_result"
                        LOGGER.warning(
                            "%s %s job #%s MPS fallback also FAILED: %s. "
                            "Returning penalty value so COBYLA can continue.",
                            wrapper_ref._algo_name,
                            wrapper_ref._label,
                            job_num,
                            mps_exc,
                        )
                        res = wrapper_ref._make_penalty_result(*run_args)
                if not self._result_logged:
                    self._result_logged = True
                    total = time.time() - self._submit_time
                    self._metadata["complete_time_utc"] = datetime.now(timezone.utc).isoformat()
                    self._metadata["elapsed_sec"] = float(total)
                    try:
                        status = self._inner_job.status()
                        self._metadata["job_status"] = str(status)
                    except Exception:
                        pass
                    wrapper_ref._record_completed_metadata(self._metadata)
                    LOGGER.info(
                        "%s %s job #%s completed on %s in %.1fs (backend=%s)",
                        wrapper_ref._algo_name,
                        wrapper_ref._label,
                        job_num,
                        wrapper_ref._qpu_id,
                        total,
                        backend_at_dispatch,
                    )
                return res

            def __getattr__(self, name: str) -> Any:
                return getattr(self._inner_job, name)

        return _JobWrapper(job, job_metadata)

    def _make_penalty_result(self, *args: Any) -> Any:
        """Construct a fake PrimitiveResult with a very large penalty value.

        This allows COBYLA to treat the failed evaluation as a terrible
        point and simply move to a different direction, rather than crashing
        with 'capi_return is NULL'.
        """
        try:
            from qiskit.primitives.containers import PrimitiveResult, PubResult
            from qiskit.primitives.containers.data_bin import DataBin

            # Count how many PUBs were in the original call
            num_pubs = 1
            if args:
                first_arg = args[0]
                if isinstance(first_arg, (list, tuple)):
                    num_pubs = len(first_arg) if len(first_arg) > 0 else 1

            pub_results = []
            for _ in range(num_pubs):
                # Return a very large energy so the optimizer rejects this point
                evs = np.array([1e15])
                stds = np.array([0.0])
                data = DataBin(evs=evs, stds=stds)
                pub_results.append(PubResult(data))

            penalty = PrimitiveResult(pub_results)
            LOGGER.info(
                "%s %s: Returning penalty result (1e15) for %d PUB(s) "
                "so optimizer can continue.",
                self._algo_name,
                self._label,
                num_pubs,
            )
            return penalty
        except Exception as pe:
            LOGGER.error(
                "%s %s: Could not construct penalty PrimitiveResult: %s. "
                "Attempting simple penalty fallback.",
                self._algo_name,
                self._label,
                pe,
            )
            # Absolute last resort: return a mock object
            class _MockResult:
                def __init__(self):
                    self.values = np.array([1e15])
                def __getitem__(self, idx):
                    class _PubResult:
                        def __init__(self):
                            self.data = type("DataBin", (), {"evs": np.array([1e15]), "stds": np.array([0.0])})()
                    return _PubResult()
                def __len__(self):
                    return 1
            return _MockResult()

    def _run_mps_fallback(self, *args: Any, **kwargs: Any) -> Any:
        """Re-run the primitive call on a local AerSimulator with MPS method.

        Matrix Product State can efficiently simulate 100+ qubit circuits
        as long as entanglement stays moderate (which is the case for the
        EfficientSU2 ansatz with linear entanglement used by QRAO).

        Since the circuits arriving here may be ISA-transpiled for a specific
        IBM backend (e.g. ibm_fez 156q), we strip layout metadata and
        re-transpile for the generic AerSimulator to avoid coupling-map
        mismatches.
        """
        try:
            from qiskit_aer import AerSimulator
            from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
        except ImportError:
            LOGGER.error(
                "%s %s: MPS fallback requires qiskit-aer. "
                "Install with: pip install qiskit-aer",
                self._algo_name,
                self._label,
            )
            raise

        mps_backend = AerSimulator(method="matrix_product_state")
        LOGGER.info(
            "%s %s: Running job #%s on local MPS simulator (%d encoded qubits)",
            self._algo_name,
            self._label,
            self._job_count,
            self._num_qubits,
        )

        # Choose the right local primitive to match the wrapped type
        if self._label == "estimator":
            local_primitive = BackendEstimatorV2(backend=mps_backend)
        else:
            local_primitive = BackendSamplerV2(backend=mps_backend)

        # Apply the same configuration as the wrapped primitive
        for fn in self._configure_fns:
            try:
                fn(local_primitive)
            except Exception:
                pass

        # The PUBs may contain ISA-transpiled circuits. Try to strip the
        # layout and re-transpile for the AerSimulator.
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit.circuit import QuantumCircuit
            mps_pm = generate_preset_pass_manager(
                backend=mps_backend,
                optimization_level=1,
            )
            new_args = list(args)
            if new_args and isinstance(new_args[0], (list, tuple)):
                new_pubs = []
                for pub in new_args[0]:
                    if isinstance(pub, (list, tuple)) and len(pub) >= 1:
                        circ = pub[0]
                        if isinstance(circ, QuantumCircuit):
                            # Strip layout and re-transpile
                            try:
                                new_circ = mps_pm.run(circ)
                                new_pub = (new_circ,) + tuple(pub[1:])
                                new_pubs.append(new_pub)
                            except Exception:
                                new_pubs.append(pub)
                        else:
                            new_pubs.append(pub)
                    else:
                        new_pubs.append(pub)
                new_args[0] = new_pubs
                args = tuple(new_args)
        except Exception as retranspile_err:
            LOGGER.debug(
                "%s %s: Could not re-transpile PUBs for MPS: %s (using originals)",
                self._algo_name,
                self._label,
                retranspile_err,
            )

        return local_primitive.run(*args, **kwargs)



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
        from qiskit_algorithms.optimizers import COBYLA, POWELL, SPSA, SLSQP
        from qiskit_algorithms.utils import algorithm_globals
        from qiskit_optimization.algorithms.qrao import (
            MagicRounding,
            QuantumRandomAccessEncoding,
            QuantumRandomAccessOptimizer,
            SemideterministicRounding,
        )
        from qiskit_optimization.minimum_eigensolvers import NumPyMinimumEigensolver, VQE
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

    # --- Pre-flight IBM budget check (matches VQE/QAOA/PCE behaviour) ---
    if qpu.provider == "ibm":
        manager._refresh_ibm_usage_if_needed()
        if not qpu.is_available:
            raise RuntimeError(
                f"IBM QPU '{qpu_id}' unavailable before QRAO start: {qpu.last_error}"
            )

    LOGGER.info(
        "QRAO optimizer start | qpu_id=%s qpu_mode=%s shots=%s maxiter=%s",
        qpu_id, scheduler.mode, shots, maxiter,
    )

    backend = None
    backend_name = qpu_id
    pending_jobs = None
    pass_manager = None
    primitive_kind = "unknown"
    estimator: Any | None = None
    sampler: Any | None = None
    use_numpy_min_eigensolver = False
    force_mps_env = str(os.getenv("QOBENCH_FORCE_MPS", "")).strip().lower()
    local_sim_method_env = str(os.getenv("QOBENCH_LOCAL_SIMULATOR_METHOD", "")).strip().lower()
    force_local_mps = (
        force_mps_env in {"1", "true", "yes", "on"}
        or local_sim_method_env in {"mps", "matrix_product_state"}
    )
    try:
        optimization_level = int(getattr(manager, "qiskit_optimization_level", 1))
    except Exception:
        optimization_level = 1
    optimization_level = max(0, min(3, optimization_level))
    effective_rounding_scheme = str(rounding_scheme).strip().lower()

    def _set_nested_attr(target: Any, dotted_name: str, value: Any) -> bool:
        current = target
        parts = dotted_name.split(".")
        for name in parts[:-1]:
            if not hasattr(current, name):
                return False
            try:
                current = getattr(current, name)
            except Exception:
                return False
        leaf = parts[-1]
        if not hasattr(current, leaf):
            return False
        try:
            setattr(current, leaf, value)
        except Exception:
            return False
        return True

    def _configure_sampler_shots(target: Any) -> None:
        if target is None:
            return
        if hasattr(target, "set_options"):
            try:
                target.set_options(shots=int(shots))
            except Exception:
                pass
        _set_nested_attr(target, "options.shots", int(shots))
        _set_nested_attr(target, "options.default_shots", int(shots))
        try:
            setattr(target, "default_shots", int(shots))
        except Exception:
            pass

    def _configure_estimator_defaults(target: Any) -> None:
        if target is None:
            return
        _set_nested_attr(target, "options.default_shots", int(shots))
        _set_nested_attr(target, "options.default_precision", 0.05)
        try:
            setattr(target, "default_shots", int(shots))
        except Exception:
            pass

    def _configure_ibm_runtime_options(target: Any) -> None:
        pass

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
        LOGGER.info(
            "Selected IBM backend | selection=qrao backend=%s qubits=%s pending_jobs=%s",
            backend_name,
            backend_info.get("num_qubits"),
            pending_jobs,
        )
        try:
            from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
            from qiskit_ibm_runtime import EstimatorV2 as RuntimeEstimator
            from qiskit_ibm_runtime import SamplerV2 as RuntimeSampler
        except Exception as exc:
            raise ModuleNotFoundError(
                "IBM QRAO execution requires qiskit-ibm-runtime with Runtime V2 primitives."
            ) from exc

        pass_manager = generate_preset_pass_manager(
            backend=backend,
            optimization_level=optimization_level,
        )
        raw_estimator = RuntimeEstimator(mode=backend)
        raw_sampler = RuntimeSampler(mode=backend)
        _configure_estimator_defaults(raw_estimator)
        _configure_sampler_shots(raw_sampler)
        _configure_ibm_runtime_options(raw_estimator)
        _configure_ibm_runtime_options(raw_sampler)
        # Wrap primitives for per-job logging, budget monitoring, and
        # seamless credential rotation (rebuild on new backend when
        # current credential budget runs low).
        estimator = _QRAOPrimitiveWrapper(
            raw_estimator,
            label="estimator",
            manager=manager,
            qpu_id=qpu_id,
            backend_name=backend_name,
            qpu=qpu,
            num_qubits=encoded_qubits,
            primitive_cls=RuntimeEstimator,
            configure_fns=[_configure_estimator_defaults, _configure_ibm_runtime_options],
            algo_name="QRAO",
        )
        sampler = _QRAOPrimitiveWrapper(
            raw_sampler,
            label="sampler",
            manager=manager,
            qpu_id=qpu_id,
            backend_name=backend_name,
            qpu=qpu,
            num_qubits=encoded_qubits,
            primitive_cls=RuntimeSampler,
            configure_fns=[_configure_sampler_shots, _configure_ibm_runtime_options],
            algo_name="QRAO",
        )
        primitive_kind = "runtime_v2"
    elif qpu.provider == "local_qiskit":
        if encoded_qubits <= 20 and not force_local_mps:
            try:
                from qiskit.primitives import StatevectorEstimator, StatevectorSampler

                estimator = StatevectorEstimator(seed=int(seed))
                sampler = StatevectorSampler(
                    default_shots=int(shots),
                    seed=int(seed),
                )
                backend_name = "statevector_primitives"
                primitive_kind = "statevector_v2"
            except Exception:
                estimator = None
                sampler = None

        if estimator is None or sampler is None:
            try:
                from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
                from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

                try:
                    from qiskit_aer import AerSimulator

                    backend = AerSimulator(method="matrix_product_state")
                    backend_name = "aer_simulator_mps"
                except Exception as exc:
                    if force_local_mps:
                        raise ModuleNotFoundError(
                            "Forced MPS simulation requested, but qiskit-aer is unavailable."
                        ) from exc
                    from qiskit.providers.basic_provider import BasicSimulator

                    backend = BasicSimulator()
                    backend_name = "basic_simulator"

                pass_manager = generate_preset_pass_manager(
                    backend=backend,
                    optimization_level=optimization_level,
                )
                estimator = BackendEstimatorV2(backend=backend)
                sampler = BackendSamplerV2(backend=backend)
                _configure_estimator_defaults(estimator)
                _configure_sampler_shots(sampler)
                primitive_kind = "backend_v2"
            except Exception:
                estimator = None
                sampler = None
                use_numpy_min_eigensolver = True
                backend_name = "numpy_minimum_eigensolver"
                primitive_kind = "numpy"
    else:
        raise RuntimeError(
            f"QRAO currently supports only IBM or local_qiskit backends (got '{qpu.provider}')."
        )

    if effective_rounding_scheme == "magic":
        if sampler is None:
            LOGGER.warning(
                "QRAO magic rounding is unavailable for backend=%s; falling back to semideterministic rounding.",
                backend_name,
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
    initial_point = np.zeros(ansatz.num_parameters, dtype=float)
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
                "primitive_kind": primitive_kind,
                "transpilation_optimization_level": (
                    int(optimization_level) if pass_manager is not None else None
                ),
            }
        }
    ]
    eval_counter = 0
    if use_numpy_min_eigensolver:
        trace.append(
            {
                "qrao_solver": {
                    "min_eigen_solver": "NumPyMinimumEigensolver",
                    "reason": "local_qiskit fallback without supported primitive backend",
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

            # Use current backend name from wrapper (may change after rotation)
            current_backend = (
                estimator.backend_name
                if hasattr(estimator, "backend_name")
                else backend_name
            )
            eval_metadata: dict[str, Any] | None = None
            if hasattr(estimator, "pop_oldest_completed_metadata"):
                try:
                    popped = estimator.pop_oldest_completed_metadata()
                    if isinstance(popped, dict):
                        eval_metadata = popped
                except Exception:
                    eval_metadata = None
            if eval_metadata is None:
                eval_metadata = {
                    "provider": str(getattr(qpu, "provider", "")),
                    "qpu_id": qpu_id,
                    "backend_name": current_backend,
                    "shots": int(shots),
                }
            if not str(eval_metadata.get("backend_name", "")).strip():
                eval_metadata["backend_name"] = current_backend
            if not str(eval_metadata.get("qpu_id", "")).strip():
                eval_metadata["qpu_id"] = qpu_id

            LOGGER.info(
                "QRAO evaluation | eval=%s qpu_id=%s backend=%s objective=%.8f",
                eval_counter, qpu_id, current_backend, objective_value,
            )

            trace.append(
                {
                    "evaluation": int(eval_counter),
                    "qpu_id": qpu_id,
                    "backend_name": current_backend,
                    "objective_value": objective_value,
                    "metadata": eval_metadata,
                }
            )


        min_eigen_solver = VQE(
            ansatz=ansatz,
            optimizer=optimizer,
            estimator=estimator,
            initial_point=initial_point,
            callback=_callback,
            pass_manager=pass_manager,
        )

    if effective_rounding_scheme == "magic":
        rounding = MagicRounding(
            sampler=sampler,
            basis_sampling="uniform",
            seed=int(seed),
            pass_manager=pass_manager,
        )
    else:
        rounding = SemideterministicRounding()

    qrao = QuantumRandomAccessOptimizer(
        min_eigen_solver=min_eigen_solver,
        rounding_scheme=rounding,
    )

    LOGGER.info(
        "Dispatching QRAO optimization -> %s | backend=%s encoded_qubits=%s "
        "rounding=%s optimizer=%s maxiter=%s shots=%s",
        qpu_id, backend_name, encoded_qubits,
        effective_rounding_scheme, optimizer_key, maxiter, shots,
    )

    # Suppress the verbose qiskit-optimization VQE logger that prints
    # "Optimization complete in X seconds. Found optimal point [huge array...]"
    # We provide our own structured logging instead.
    _qiskit_vqe_logger = logging.getLogger("qiskit_optimization.minimum_eigensolvers.vqe")
    _orig_level = _qiskit_vqe_logger.level
    _qiskit_vqe_logger.setLevel(logging.WARNING)
    try:
        result = qrao.solve(qubo)
    finally:
        _qiskit_vqe_logger.setLevel(_orig_level)

    for phase, primitive in (("estimator", estimator), ("sampler", sampler)):
        if hasattr(primitive, "drain_completed_metadata"):
            try:
                remaining_meta = primitive.drain_completed_metadata()
            except Exception:
                remaining_meta = []
            for meta in remaining_meta:
                if not isinstance(meta, dict):
                    continue
                trace.append(
                    {
                        "qrao_job_phase": phase,
                        "qpu_id": str(meta.get("qpu_id", qpu_id)),
                        "backend_name": str(meta.get("backend_name", backend_name)),
                        "metadata": meta,
                    }
                )

    LOGGER.info(
        "QRAO optimization complete | qpu_id=%s backend=%s evaluations=%s",
        qpu_id, backend_name, eval_counter,
    )

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
