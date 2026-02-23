import time
full_start_time = time.time()
import os
from datetime import datetime
import qiskit 
print("Qiskit Version",qiskit.__version__)

# basic imports

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import networkx as nx
import rustworkx as rx
from rustworkx.visualization import mpl_draw as draw_graph

# quantum imports
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer
from qiskit.circuit.library import QAOAAnsatz
# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimatorV2, BackendSamplerV2
from qiskit_aer import AerSimulator
backend = AerSimulator(method='matrix_product_state')


estimator = BackendEstimatorV2(backend=backend)
sampler = BackendSamplerV2(backend=backend, options={"default_shots": 8000})


"""
Generate the Problem 
"""
# Parameters for the knapsack problem
num_items = 5
max_weight = 10

# Generate random weights and values for the items

weights = np.random.randint(1, 10, size=num_items)
values = np.random.randint(10, 50, size=num_items)

# Capacity of the knapsack
capacity = int(0.6 * np.sum(weights))

print(f"Weights: {weights}")
print(f"Values: {values}")
print(f"Capacity: {capacity}")

# Create the Knapsack problem
knapsack = Knapsack(values.tolist(), weights.tolist(), capacity)

# Convert the problem to a QuadraticProgram
problem = knapsack.to_quadratic_program()

# Solve the problem using CplexOptimizer
optimizer = CplexOptimizer()
result = optimizer.solve(problem)

print("Exact Solution:")
print(result.fval)
print(result.x)


# convert the problem to a qubo

# problem to qubo
converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)


num_vars = qubo.get_num_vars()
print(f"Number of variables: {num_vars}")

qubitOp, offset = qubo.to_ising()


## QAOA

reps =3

print(f"Number of repetitions: {reps}")
circuit = QAOAAnsatz(cost_operator=qubitOp, reps=reps)
#circuit.measure_all()
circuit = circuit.decompose(reps=3)
circuit.draw('mpl',fold=-1)

initial_gamma = np.pi
initial_beta = np.pi/2

init_params = [initial_gamma, initial_beta]*reps

print("Number of parameters:", len(init_params))
# number of qubits, circuit depth, gate counts, 2 qubit gate count
print('Number of qubits:', circuit.num_qubits)
print('Circuit depth:', circuit.depth())
print('Gate counts:', dict(circuit.count_ops()))
# print new line
print()
print("-----------------------------------------------------")

# cost function

cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

def cost_func_estimator(params, ansatz, hamiltonian, estimator):

    # transform the observable defined on virtual qubits to
    # an observable defined on all physical qubits
    isa_hamiltonian = hamiltonian.apply_layout(ansatz.layout)

    pub = (ansatz, isa_hamiltonian, params)
    job = estimator.run([pub])

    results = job.result()[0]
    cost = results.data.evs


    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(cost)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {cost}]")



    objective_func_vals.append(cost)


    return cost



objective_func_vals = [] # Store the objective function values
optimization_time_start = time.time()
result = minimize(
    cost_func_estimator,
    init_params,
    args= (circuit, qubitOp, estimator),
    method="Powell",
    tol=1e-3
)
optimization_time_end = time.time()

print(result)

output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"cost_vs_iterations_{timestamp}.png"
output_path = os.path.join(output_folder, file_name)

# Plot "Cost vs. Iteration"
plt.figure(figsize=(12, 6))
plt.plot(objective_func_vals, label="Objective Function")
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Cost vs. Iteration")
plt.legend()

# Save the plot
plt.savefig(output_path, format="png", dpi=300)
plt.close()  # Close the plot to free up memory

print(f"Plot saved to: {output_path}")

# post processing
post_processing_time_start = time.time()
optimized_circuit = circuit.assign_parameters(result.x)
optimized_circuit.measure_all()
optimized_circuit.draw('mpl', idle_wires=False,fold=-1)

pub = (optimized_circuit,)
job = sampler.run([pub], shots=int(1e4))
counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

post_processing_time_end = time.time()

def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, num_vars)
most_likely_bitstring.reverse()

print("Result bitstring:", most_likely_bitstring)




# Find the indices of the top 4 values
top_4_indices = np.argsort(np.abs(values))[::-1][:4]
top_4_results = []
# Print the top 4 results with their probabilities
print("Top 4 Results:")
for idx in top_4_indices:
    bitstring = to_bitstring(keys[idx], num_vars)
    bitstring.reverse()
    top_4_results.append(bitstring)
    print(f"Bitstring: {bitstring}, Probability: {values[idx]:.6f}")



# Update matplotlib font size
matplotlib.rcParams.update({"font.size": 10})

# Assuming final_distribution_bin is defined elsewhere
final_bits = final_distribution_bin  

# Get the absolute values and sort to extract the top 16 and top 4 values
values = np.abs(list(final_bits.values()))
top_16_values = sorted(values, reverse=True)[:16]
top_4_values = sorted(values, reverse=True)[:4]

# Filter the top 16 bitstrings and their probabilities
top_16_bitstrings = []
top_16_probabilities = []

for bitstring, value in final_bits.items():
    if abs(value) in top_16_values:
        top_16_bitstrings.append(bitstring)
        top_16_probabilities.append(value)

# Sort the top 16 by probability for better visualization
sorted_indices = np.argsort(top_16_probabilities)[::-1]
top_16_bitstrings = [top_16_bitstrings[i] for i in sorted_indices]
top_16_probabilities = [top_16_probabilities[i] for i in sorted_indices]

file_name = f"result_distribution_{timestamp}.png"
output_path = os.path.join(output_folder, file_name)
# Plot the top 16 values
fig = plt.figure(figsize=(11, 6))
ax = fig.add_subplot(1, 1, 1)
plt.xticks(rotation=45)
plt.title("Result Distribution")
plt.xlabel("Bitstrings (reversed)")
plt.ylabel("Probability")

bars = ax.bar(top_16_bitstrings, top_16_probabilities, color="tab:grey")

# Highlight the top 4 bars in purple
for i, bar in enumerate(bars):
    if top_16_probabilities[i] in top_4_values:
        bar.set_color("tab:purple")

# Save the plot
plt.savefig(output_path, format="png", dpi=300)
plt.close(fig)  # Close the figure to free up memory

print(f"Plot saved to: {output_path}")


# convert 


result = converter.interpret(most_likely_bitstring)
cost = problem.objective.evaluate(result)
feasible =problem.get_feasibility_info(result)[0]


# print("Result knapsack:", result)
# print("Result value:", cost)
# print("Feasible:", feasible)

# Iterate through the list of bitstrings and evaluate for each
for bitstring in top_4_results:
    result = converter.interpret(bitstring)  # Interpret the bitstring
    cost = problem.objective.evaluate(result)  # Evaluate the cost for the bitstring
    feasible =problem.get_feasibility_info(result)[0]
    
    # Print the results
    print("Result knapsack:", result)
    print("Result value:", cost)
    print("Feasible solution:", feasible)
    print("--------------------")


full_end_time = time.time()

print()
print("-----------------------------------------------------")
print("Time taken for optimization:", optimization_time_end - optimization_time_start, "seconds")
print("Time taken for post-processing:", post_processing_time_end - post_processing_time_start, "seconds")
print("Total time taken:", full_end_time - full_start_time, "seconds")
# use this to run
# python -u .\qaoa.py > qaoa_out.log 2>&1

