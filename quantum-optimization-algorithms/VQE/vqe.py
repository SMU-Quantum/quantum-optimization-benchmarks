import time 
full_start_time = time.time()

import qiskit 
print("Qiskit Version: ",qiskit.__version__)
import os
from datetime import datetime

# basic imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# quantum imports
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
# Pre-defined ansatz circuit and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
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
num_items = 4
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

# make the ansatz circuit
ansatz = EfficientSU2(qubitOp.num_qubits,reps=reps)
ansatz = ansatz.decompose()
num_params = ansatz.num_parameters

print('Number of parameters:', num_params)

# number of qubits, circuit depth, gate counts, 2 qubit gate count
print('Number of qubits:', ansatz.num_qubits)
print('Circuit depth:', ansatz.depth())
print('Gate counts:', dict(ansatz.count_ops()))

# print new line
print()
print("-----------------------------------------------------")

# Define the cost function

def cost_func(params, ansatz, hamiltonian, estimator):
    """Return estimate of energy from estimator

    Parameters:
        params (ndarray): Array of ansatz parameters
        ansatz (QuantumCircuit): Parameterized ansatz circuit
        hamiltonian (SparsePauliOp): Operator representation of Hamiltonian
        estimator (EstimatorV2): Estimator primitive instance
        cost_history_dict: Dictionary for storing intermediate results

    Returns:
        float: Energy estimate
    """
    pub = (ansatz, [hamiltonian], [params])
    result = estimator.run(pubs=[pub]).result()
    energy = result[0].data.evs[0]

    cost_history_dict["iters"] += 1
    cost_history_dict["prev_vector"] = params
    cost_history_dict["cost_history"].append(energy)
    print(f"Iters. done: {cost_history_dict['iters']} [Current cost: {energy}]")

    return energy

cost_history_dict = {
    "prev_vector": None,
    "iters": 0,
    "cost_history": [],
}

# Initial parameters
x0 = 2 * np.pi * np.random.random(num_params)


# optimization loop
optimization_time_start = time.time()
res = minimize(
        cost_func,
        x0,
        args=(ansatz, qubitOp, estimator),
        method="cobyla",
    )
print()

optimization_time_end = time.time()
print(res)

# sanity check

all(cost_history_dict["prev_vector"] == res.x)
cost_history_dict["iters"] == res.nfev

# Define the folder where the plot will be saved
output_folder = "output_plots"
os.makedirs(output_folder, exist_ok=True)  # Create the folder if it doesn't exist
# Generate a dynamic file name with a timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
file_name = f"cost_vs_iterations_{timestamp}.png"
fig, ax = plt.subplots()
ax.plot(range(cost_history_dict["iters"]), cost_history_dict["cost_history"])
ax.set_xlabel("Iterations")
ax.set_ylabel("Cost")
ax.set_title("Cost vs. Iterations")
# Save the plot to the folder with the dynamic name
output_path = os.path.join(output_folder, file_name)
plt.savefig(output_path, format="png", dpi=300)
plt.close(fig)  # Close the figure to free up memory

print(f"Plot saved to: {output_path}")

post_processing_time_start = time.time()
# get the results
ansatz = ansatz.assign_parameters(res.x)
ansatz.measure_all()


pub = (ansatz,)
job = sampler.run([pub], shots=int(1e4))
counts_int = job.result()[0].data.meas.get_int_counts()
counts_bin = job.result()[0].data.meas.get_counts()
shots = sum(counts_int.values())
final_distribution_int = {key: val/shots for key, val in counts_int.items()}
final_distribution_bin = {key: val/shots for key, val in counts_bin.items()}

post_processing_time_end = time.time()
# auxiliary functions to sample most likely bitstring
def to_bitstring(integer, num_bits):
    result = np.binary_repr(integer, width=num_bits)
    return [int(digit) for digit in result]

keys = list(final_distribution_int.keys())
values = list(final_distribution_int.values())
most_likely = keys[np.argmax(np.abs(values))]
most_likely_bitstring = to_bitstring(most_likely, num_vars)
most_likely_bitstring.reverse()

# print("Result bitstring:", most_likely_bitstring)




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

file_name = f"result_distribution_{timestamp}.png"
# Save the plot
output_path = os.path.join(output_folder, file_name)
plt.savefig(output_path, format="png", dpi=300)
plt.close(fig)  # Close the figure to free up memory




# convert the bitstring to a solution

result = converter.interpret(most_likely_bitstring)
cost = problem.objective.evaluate(result)
feasible =problem.get_feasibility_info(result)[0]

print()
print("--------------------")

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

full_end_time = time.time()

print()
print("-----------------------------------------------------")
print("Time taken for optimization:", optimization_time_end - optimization_time_start, "seconds")
print("Time taken for post-processing:", post_processing_time_end - post_processing_time_start, "seconds")
print("Total time taken:", full_end_time - full_start_time, "seconds")
# use this to run
# python -u  vqe.py > vqe_out.log 2>&1

