from __future__ import annotations

import json
import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Literal, Optional

import numpy as np
from scipy.optimize import minimize

from .hardware_manager import HAS_BRAKET, normalize_counts


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


def build_ansatz_bundle(num_qubits: int, layers: int, entanglement: str) -> AnsatzBundle:
    from qiskit import QuantumCircuit
    from qiskit.circuit import Parameter

    ent_pairs = _entanglement_pairs(num_qubits, entanglement)
    qiskit_qc = QuantumCircuit(num_qubits, num_qubits)
    qiskit_params: list[Any] = []
    param_index = 0

    for layer in range(layers + 1):
        for q in range(num_qubits):
            p = Parameter(f"theta_{param_index}")
            qiskit_params.append(p)
            qiskit_qc.ry(p, q)
            param_index += 1
        if layer < layers:
            for u, v in ent_pairs:
                qiskit_qc.cx(u, v)
    qiskit_qc.measure(range(num_qubits), range(num_qubits))

    braket_qc = None
    braket_params = None
    if HAS_BRAKET and Circuit is not None and FreeParameter is not None:
        braket_qc = Circuit()
        braket_params = []
        pidx = 0
        for layer in range(layers + 1):
            for q in range(num_qubits):
                p = FreeParameter(f"theta_{pidx}")
                braket_params.append(p)
                braket_qc.ry(q, p)
                pidx += 1
            if layer < layers:
                for u, v in ent_pairs:
                    braket_qc.cz(u, v)

    return AnsatzBundle(
        ansatz_id=f"n{num_qubits}_l{layers}_e{entanglement}",
        num_qubits=num_qubits,
        num_parameters=len(qiskit_params),
        qiskit_template=qiskit_qc,
        qiskit_parameters=qiskit_params,
        braket_template=braket_qc,
        braket_parameters=braket_params if braket_params is not None else [],
    )


class QpuScheduler:
    def __init__(self, mode: Literal["single", "multi"], qpu_ids: list[str]) -> None:
        if not qpu_ids:
            raise ValueError("qpu_ids must not be empty")
        self.mode = mode
        self.qpu_ids = qpu_ids
        self._rr_index = 0
        self._lock = threading.Lock()

    def next_qpu(self) -> str:
        if self.mode == "single" or len(self.qpu_ids) == 1:
            return self.qpu_ids[0]
        with self._lock:
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
) -> dict[str, Any]:
    theta_list = [float(value) for value in theta]
    LOGGER.debug(
        "Submitting evaluation | qpu_id=%s objective_mode=%s shots=%s timeout_sec=%s",
        qpu_id,
        objective_mode,
        shots,
        timeout_sec,
    )
    counts, metadata = manager.run_counts(
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
    counts_list, metadata_list = manager.run_counts_batch(
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
        counts = counts_list[idx]
        metadata = metadata_list[idx]
        objective_value = objective.expectation(counts)
        best_bitstring, best_energy = objective.best_sample(counts)
        results.append(
            {
                "qpu_id": qpu_id,
                "theta": theta_list,
                "counts": counts,
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
    method: Literal["vqe", "cvar_vqe"],
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
    if method == "cvar_vqe":
        objective_mode = "cvar"

    rng = np.random.default_rng(seed)
    theta0 = rng.uniform(0.0, 2.0 * np.pi, size=(ansatz.num_parameters,))
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
        qpu_id = scheduler.next_qpu()
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
        "PCE optimizer start | qpu_mode=%s qpus=%s shots=%s maxiter=%s population=%s elite_count=%s batch_size=%s",
        scheduler.mode,
        scheduler.qpu_ids,
        shots,
        maxiter,
        population_size,
        elite_count,
        batch_size,
    )

    trace: list[dict[str, Any]] = []
    qpu_usage: dict[str, int] = {}
    best_value = float("inf")
    best_theta = [float(x) for x in mean]
    best_bitstring = "0" * ansatz.num_qubits
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
                        )
                    )
                for future in as_completed(futures):
                    eval_results.append(future.result())
        else:
            for idx in range(population_size):
                qpu_id = scheduler.next_qpu()
                qpu_usage[qpu_id] = qpu_usage.get(qpu_id, 0) + 1
                eval_results.append(
                    _evaluate_theta(
                        theta=candidates[idx],
                        qpu_id=qpu_id,
                        manager=manager,
                        ansatz=ansatz,
                        objective=objective,
                        shots=shots,
                        timeout_sec=timeout_sec,
                        objective_mode="expectation",
                        cvar_alpha=1.0,
                    )
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


def serialize_trace(trace: list[dict[str, Any]]) -> str:
    lines = [json.dumps(entry, sort_keys=True) for entry in trace]
    return "\n".join(lines) + ("\n" if lines else "")
