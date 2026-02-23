import time 
full_start_time = time.time()

import qiskit 
print("Qiskit Version: ",qiskit.__version__)
import os
from datetime import datetime
# Set matplotlib backend to avoid Tkinter errors
import matplotlib
matplotlib.use('Agg')
# basic imports
import matplotlib
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


def compute_cvar(probabilities, values, alpha):
    """
    Computes the Conditional Value at Risk (CVaR) for given probabilities, values, and confidence level.
    CVaR is a risk assessment measure that quantifies the expected losses exceeding the Value at Risk (VaR) at a given confidence level.
    Args:
    probabilities (list or array): List or array of probabilities associated with each value.
    values (list or array): List or array of corresponding values.
    alpha (float): Confidence level (between 0 and 1).
    float: The computed CVaR value.
    Example:
    >>> probabilities = [0.1, 0.2, 0.3, 0.4]
    >>> values = [10, 20, 30, 40]
    >>> alpha = 0.95
    >>> compute_cvar(probabilities, values, alpha)
    35.0
    Notes:
    - The function first sorts the values and their corresponding probabilities.
    - It then accumulates the probabilities until the total probability reaches the confidence level alpha.
    - The CVaR is calculated as the weighted average of the values, considering only the top (1-alpha) portion of the distribution.
    
    Auxilliary method to computes CVaR for given probabilities, values, and confidence level.
    
    Attributes:
    - probabilities: list/array of probabilities
    - values: list/array of corresponding values
    - alpha: confidence level
    
    Returns:
    - CVaR
    """
    sorted_indices = np.argsort(values)
    probs = np.array(probabilities)[sorted_indices]
    vals = np.array(values)[sorted_indices]
    cvar = 0
    total_prob = 0
    for i, (p, v) in enumerate(zip(probs, vals)):
        done = False
        if p >= alpha - total_prob:
            p = alpha - total_prob
            done = True
        total_prob += p
        cvar += p * v
    cvar /= total_prob
    return cvar


# function to evaluate the bitstring

def eval_bitstring(H, x):
    """
    Evaluate the objective function for a given bitstring.
    
    Args:
        H (SparsePauliOp): Cost Hamiltonian.
        x (str): Bitstring (e.g., '101').
    
    Returns:
        float: Evaluated objective value.
    """
    # Translate the bitstring to spin representation (+1, -1)
    spins = np.array([(-1) ** int(b) for b in x[::-1]])
    value = 0.0

    # Loop over Pauli terms and compute the objective value
    for pauli, coeff in zip(H.paulis, H.coeffs):
        weight = coeff.real  # Get the real part of the coefficient
        z_indices = np.where(pauli.z)[0]  # Indices of Z operators in the Pauli term
        contribution = weight * np.prod(spins[z_indices])  # Compute contribution
        value += contribution

    return value


num_qubits = qubitOp.num_qubits
print('Number of qubits:', num_qubits)

reps = 2

class Objective:
    """
    Wrapper for objective function to track the history of evaluations.
    """
    def __init__(self, H, offset, alpha, num_qubits):
        self.history = []
        self.H = qubitOp
        self.offset = offset
        self.alpha = alpha
        self.num_qubits = num_qubits
        self.opt_history = []
        self.last_counts = {}  # Store the counts from the last circuit execution
        self.counts_history = []  # New attribute to store counts history


    def evaluate(self, thetas):
        """
        Evaluate the CVaR for a given set of parameters. 
        """
        # Create a new circuit
        qc = QuantumCircuit(num_qubits)

        # Create a single ParameterVector for all parameters
        theta = ParameterVector('theta', 2 * reps * num_qubits)

        # Build the circuit
        for r in range(reps):
            # Rotation layer of RY gates
            for i in range(num_qubits):
                qc.ry(theta[r * 2 * num_qubits + i], i)
            
            # Rotation layer of RZ gates
            for i in range(num_qubits):
                qc.rz(theta[r * 2 * num_qubits + num_qubits + i], i)
            
            # Entangling layer of CNOT gates
            if r < reps - 1:  # Add entanglement only between layers
                for i in range(num_qubits - 1):
                    qc.cx(i, i + 1)

        # Add measurement gates
        qc.measure_all()
        qc = qc.assign_parameters(thetas)

        # Execute the circuit

        job = sampler.run([qc])
        result = job.result()
        data_pub = result[0].data
        self.last_counts = data_pub.meas.get_counts()  # Store counts
        self.counts_history.append(self.last_counts)

        # Evaluate counts
        probabilities = np.array(list(self.last_counts.values()), dtype=float)  # Ensure float array
        values = np.zeros(len(self.last_counts), dtype=float)
        
        for i, x in enumerate(self.last_counts.keys()):
            values[i] = eval_bitstring(self.H, x) + self.offset
        
        # Normalize probabilities
        probabilities /= probabilities.sum()  # No more dtype issue

        # Compute CVaR
        cvar = compute_cvar(probabilities, values, self.alpha)
        self.history.append(cvar)
        return cvar

num_params = 2 * reps * num_qubits   
initial_params = 2 * np.pi * np.random.random(num_params)
cvar_histories = {}
optimal_bitstrings = {}  # To store the best bitstring for each alpha
optimal_values = {}  # To store the best objective value for each alpha


# Optimization loop
start_time = time.time()
alphas = [0.25]
objectives = []
maxiter=1000
optimizer = COBYLA(maxiter=maxiter)
for alpha in alphas:
    print(f"Running optimization for alpha = {alpha}")
    obj = Objective(qubitOp, offset, alpha, num_qubits)
    optimizer.minimize(fun=obj.evaluate, x0=initial_params)
    objectives.append(obj)
    
    # Store CVaR history for this alpha
    cvar_histories[alpha] = obj.history
    
    # Retrieve the optimal bitstring and objective value
    best_bitstring = None
    best_value = None
    
    
    # Iterate through the counts from the last iteration to find the best bitstring
    for bitstring, probability in obj.last_counts.items():
        value = eval_bitstring(qubitOp, bitstring) + offset
        if best_bitstring is None or value < best_value:
            best_bitstring = bitstring
            best_value = value
    bitstring_as_list = [int(bit) for bit in best_bitstring]
    
    optimal_bitstrings[alpha] = bitstring_as_list
    
    optimal_values[alpha] = best_value

end_time = time.time()

# Print the optimal bitstrings and values for each alpha, and also feasibility
print("\nOptimal Bitstrings and Objective Values:")
for alpha in alphas:
    # print(f"Alpha = {alpha}: Bitstring = {optimal_bitstrings[alpha]}, Objective Value = {optimal_values[alpha]}")
    
    # convert to market share solution 
    market_share_bitstring = converter.interpret(optimal_bitstrings[alpha])
    initial_feasible = problem.get_feasibility_info(market_share_bitstring)[0]
    market_share_cost = problem.objective.evaluate(market_share_bitstring)
    print(f"Feasible: {initial_feasible}")
    print(f"Multi Dimension Knapsack cost: {market_share_cost}")
    print(f"Execution time: {end_time - start_time:.2f} seconds")

