import time 
full_start_time = time.time()

import qiskit 
print("Qiskit Version: ",qiskit.__version__)
import os
from datetime import datetime

# Set matplotlib backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# quantum imports
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit,ParameterVector
from qiskit_algorithms.optimizers import COBYLA,POWELL,SLSQP,P_BFGS,ADAM,SPSA
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2,PauliEvolutionGate
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
backend = AerSimulator(method='matrix_product_state')


estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend, options={"default_shots": 8000})


"""
Create your problem instance here, till the qubo making part
"""


# Parameters for the knapsack problem
num_items = 5
max_weight = 10

# Generate random weights and values for the items

weights = np.random.randint(1, 10, size=num_items)
values = np.random.randint(10, 50, size=num_items)

# Capacity of the knapsack
capacity = int(0.6 * np.sum(weights))

# Create the Knapsack problem
knapsack = Knapsack(values.tolist(), weights.tolist(), capacity)

# Convert the problem to a QuadraticProgram
problem = knapsack.to_quadratic_program()

converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)

# classical solution
# Solve the problem using CplexOptimizer
optimizer = CplexOptimizer()
result = optimizer.solve(problem)

print("Exact Solution:")
print(result.fval)
print(result.x)

"""
Finish
"""

num_vars = qubo.get_num_vars()
print('Number of variables:', num_vars)
reps = 2
print('Number of repetitions:', reps)
# converting hamiltonian
qubitOp, offset = qubo.to_ising()


# Define CVaR function
def compute_cvar(probabilities, values, alpha):
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for p, v in zip(probs, vals):
        if total_prob + p > alpha:
            p = alpha - total_prob
        total_prob += p
        cvar += p * v
        if total_prob >= alpha:
            break
    return cvar / alpha


# Define the cost function
def eval_bitstring(H, x):
    spins = np.array([(-1) ** int(b) for b in x[::-1]])
    value = 0.0
    for pauli, coeff in zip(H.paulis, H.coeffs):
        z_indices = np.where(pauli.z)[0]
        contribution = coeff.real * np.prod(spins[z_indices])
        value += contribution
    return value



class CVaRObjective:
    def __init__(self, H, offset, alpha, sampler, ansatz):
        self.H = H
        self.offset = offset
        self.alpha = alpha
        self.sampler = sampler
        self.ansatz = ansatz
        self.history = []
        self.cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": []}

    def evaluate(self, params):
        assigned_circuit = self.ansatz.assign_parameters(params)
        assigned_circuit.measure_all()

        # Run circuit
        job = self.sampler.run([assigned_circuit])
        result = job.result()
        counts = result[0].data.meas.get_counts()
        total_shots = sum(counts.values())
        probabilities = [v / total_shots for v in counts.values()]
        bitstrings = list(counts.keys())
        values = [eval_bitstring(self.H, b) + self.offset for b in bitstrings]

        cvar = compute_cvar(probabilities, values, self.alpha)
        self.history.append(cvar)
        self.cost_history_dict["iters"] += 1
        self.cost_history_dict["cost_history"].append(cvar)
        print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {cvar}]")
        return cvar
    
# Build ansatz

# function to make qaoa circuit
def generate_sum_x_pauli_str(length):
    ret = []
    for i in range(length):
        paulis = ['I'] * length
        paulis[i] = 'X'
        ret.append(''.join(paulis))

    return ret

def qaoa_circuit(problem_ham: SparsePauliOp, depth: int = 1) -> QuantumCircuit:
    r"""
    Input:
    - problem_ham: Problem Hamiltonian to construct the QAOA circuit.
    Standard procedure would be:
    ```
        hamiltonian, offset = qubo.to_ising()
        qc = qaoa_circuit_from_qubo(hamiltonian)
    ```

    Returns:
    - qc: A QuantumCircuit object representing the QAOA circuit e^{-i\beta H_M} e^{-i\gamma H_C}.
    """
    num_qubits = problem_ham.num_qubits

    gamma = ParameterVector(name=r'$\gamma$', length=depth)
    beta = ParameterVector(name=r'$\beta$', length=depth)

    mixer_ham = SparsePauliOp(generate_sum_x_pauli_str(num_qubits))
    
    qc = QuantumCircuit(num_qubits)
    qc.h(range(num_qubits))

    for p in range(depth):
        exp_gamma = PauliEvolutionGate(problem_ham, time=gamma[p])
        exp_beta = PauliEvolutionGate(mixer_ham, time=beta[p])
        qc.append(exp_gamma, qargs=range(num_qubits))
        qc.append(exp_beta, qargs=range(num_qubits))

    return qc



num_qubits = qubitOp.num_qubits
print('Number of qubits:', num_qubits)
reps = 2
ansatz = qaoa_circuit(qubitOp, depth=reps)
ansatz = ansatz.decompose()
#ansatz = EfficientSU2(num_qubits, reps=reps).decompose()
num_params = ansatz.num_parameters

# Initial parameters
initial_params = 2 * np.pi * np.random.random(num_params)

# Optimization
alphas = [0.90, 0.75, 0.50, 0.25]
results_summary = {}
for alpha in alphas:
    print(f"\nStarting optimization for alpha = {alpha}")
    objective = CVaRObjective(qubitOp, offset, alpha, sampler, ansatz)

    start_time = time.time()
    res = minimize(
        objective.evaluate,
        initial_params,
        method="cobyla",
    )
    end_time = time.time()

    # Retrieve results
    final_circuit = ansatz.assign_parameters(res.x)
    final_circuit.measure_all()
    job = sampler.run([final_circuit], shots=int(1e4))
    result = job.result()
    counts = result[0].data.meas.get_counts()
    total_shots = sum(counts.values())

    # Normalize and retrieve distribution
    distribution = {k: v / total_shots for k, v in counts.items()}
    sorted_keys = sorted(distribution, key=distribution.get, reverse=True)

    # Get top 4 results
    top_4_results = sorted_keys[:4]
    top_4_probabilities = [distribution[k] for k in top_4_results]

    # Print top 4 results
    print("\nTop 4 Results:")
    for bitstring, probability in zip(top_4_results, top_4_probabilities):
        print(f"Bitstring: {bitstring}, Probability: {probability:.6f}")

    # Convert bitstrings to solutions
    print("\nConverted Solutions:")
    for bitstring in top_4_results:
        solution = converter.interpret([int(b) for b in bitstring])
        cost = problem.objective.evaluate(solution)
        feasible = problem.get_feasibility_info(solution)[0]
        print(f"Solution: {solution}, Cost: {cost}, Feasible: {feasible}")

    # Save cost history plot
    output_folder = "output_plots"
    os.makedirs(output_folder, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_name = f"cost_vs_iterations_{timestamp}.png"

    plt.plot(range(objective.cost_history_dict["iters"]), objective.cost_history_dict["cost_history"])
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.title("Cost vs. Iterations")
    output_path = os.path.join(output_folder, file_name)
    plt.savefig(output_path, format="png", dpi=300)
    plt.close()

    print(f"Plot saved to: {output_path}")
    print(f"Optimization Time: {end_time - start_time:.2f} seconds")

    # Store results summary
    results_summary[alpha] = {
        "top_4_results": top_4_results,
        "top_4_probabilities": top_4_probabilities,
        "optimization_time": end_time - start_time,
    }

# Summary of results for all alphas
print("\nResults Summary:")
for alpha, summary in results_summary.items():
    print(f"\nAlpha = {alpha}")
    print("Top 4 Results:", summary["top_4_results"])
    print("Top 4 Probabilities:", summary["top_4_probabilities"])
    print("Optimization Time: {:.2f} seconds".format(summary["optimization_time"]))

full_end_time = time.time()
print("\nTotal Runtime:", full_end_time - full_start_time, "seconds")

# python -u .\qaoa.py > qaoa_out.log 2>&1