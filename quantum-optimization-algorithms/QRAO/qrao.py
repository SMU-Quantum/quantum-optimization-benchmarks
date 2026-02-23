import time 
print("QRAO Start")
full_start_time = time.time()
import warnings
warnings.filterwarnings("ignore")
from docplex.mp.model import Model
from qiskit.circuit.library import QAOAAnsatz
import qiskit 
print("Qiskit Version: ",qiskit.__version__)
import os
from datetime import datetime
import glob
# basic imports
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

# quantum imports
from qiskit_optimization.applications import Maxcut, Knapsack
from qiskit.circuit import Parameter,QuantumCircuit
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_algorithms import VQE
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
from qiskit_algorithms.optimizers import COBYLA,POWELL,SLSQP,P_BFGS,ADAM,SPSA
# Pre-defined ansatz ansatz and operator class for Hamiltonian
from qiskit.circuit.library import EfficientSU2
from qiskit_algorithms import NumPyMinimumEigensolver
from qiskit_optimization.algorithms import MinimumEigenOptimizer

# SciPy minimizer routine
from scipy.optimize import minimize
from qiskit.primitives import BackendEstimator, BackendSampler
from qiskit_aer import AerSimulator
backend = AerSimulator(method='matrix_product_state')


estimator = BackendEstimator(backend=backend)
sampler = BackendSampler(backend=backend, options={"default_shots": 8000})

# qrao imports
from qiskit_optimization.algorithms.qrao import QuantumRandomAccessEncoding
from qiskit.circuit.library import RealAmplitudes, EfficientSU2
from qiskit_optimization.algorithms.qrao import (
    QuantumRandomAccessOptimizer,
    SemideterministicRounding,
)
from qiskit_optimization.algorithms.qrao import MagicRounding


"""
Create your problem instance here, till the qubo making part
"""

"""
Generate the Problem 
"""
# Parameters for the knapsack problem
num_items = 3
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




print()

             # made a quadratic program (problem)
converter = QuadraticProgramToQubo()        # converter for problem to qubo  
qubo = converter.convert(problem)                # the qubo



num_vars = qubo.get_num_vars()
print('Number of variables:', num_vars)
reps = 5
print('Number of repetitions:', reps)
# converting hamiltonian
encoding = QuantumRandomAccessEncoding(max_vars_per_qubit=3)

encoding.encode(qubo)

print(
    "We achieve a compression ratio of "
    f"({encoding.num_vars} binary variables : {encoding.num_qubits} qubits) "
    f"= {encoding.compression_ratio}.\n"
)


ansatz = EfficientSU2(num_qubits=encoding.num_qubits,entanglement='linear', reps=reps)
ansatz = ansatz.decompose(reps=2)
print('Number of qubits:', ansatz.num_qubits)
print('ansatz depth:', ansatz.depth())
print('Gate counts:', dict(ansatz.count_ops()))
vqe = VQE(
    ansatz=ansatz,
    optimizer=POWELL(maxfev=20),
    estimator=estimator,
)

# Use magic rounding
magic_rounding = MagicRounding(sampler=sampler)

# Construct the optimizer
qrao = QuantumRandomAccessOptimizer(min_eigen_solver=vqe, rounding_scheme=magic_rounding)

start_time = time.time()
results = qrao.solve(qubo)
end_time = time.time()
print(
    f"The objective function value: {results.fval}\n"
    f"x: {results.x}\n"
    f"relaxed function value: {-1 * results.relaxed_fval}\n"
)

# Extract the x values from the top 10 samples
top_x_values = [sample.x.tolist() for sample in results.samples[:10]]


for i, bitlist in enumerate(top_x_values):
    # Convert the QUBO bitstring to the original problem
    x = converter.interpret(bitlist)
    print(f"Top {i+1} Interpreted result: {x}")
    
    # Check if it's feasible
    is_feasible = problem.is_feasible(x)
    print(f"Is the result feasible? {is_feasible}")
    
    # Get the market share cost from this x
    cost = problem.objective.evaluate(x)
    print(f"Cost of the interpreted result: {cost}")
    print("-" * 50)  # Separator for readability



print("QRAO Finished")

print("----------------------------------------------")


full_end_time = time.time()
print(f"Optimization time: {end_time - start_time:.2f} seconds")
print(f"Total execution time: {full_end_time - full_start_time:.2f} seconds")

