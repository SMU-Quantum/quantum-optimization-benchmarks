import warnings
warnings.filterwarnings("ignore")
##--CELL--##
import numpy as np
from qiskit_optimization.applications import Knapsack, Maxcut
from qiskit_optimization.algorithms import CplexOptimizer
from qiskit_optimization.translators import from_docplex_mp
from qiskit_optimization.converters import QuadraticProgramToQubo
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector

from qiskit.primitives import BackendEstimator
from qiskit_aer import AerSimulator
backend = AerSimulator(method='automatic')

estimator = BackendEstimator(backend=backend, options={'shots': 1000})
##--CELL--##
from pce import *
from qubo_to_maxcut import *
##--CELL--##
# the graph
import networkx as nx
import matplotlib.pyplot as plt
def create_graph_from_weight_matrix(w):
    G = nx.Graph()
    n = len(w)

    # Add nodes
    for i in range(n):
        G.add_node(i)

    # Add edges with weights, ignoring zero-weight edges
    for i in range(n):
        for j in range(i + 1, n):
            if w[i, j] != 0:
                G.add_edge(i, j, weight=w[i, j])

    return G

def draw_graph(G, colors, pos):
    default_axes = plt.axes(frameon=True)
    nx.draw_networkx(G, node_color=colors, node_size=600, alpha=0.8, ax=default_axes, pos=pos)
    edge_labels = nx.get_edge_attributes(G, "weight")
    nx.draw_networkx_edge_labels(G, pos=pos, edge_labels=edge_labels)
##--CELL--##
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
print(problem.prettyprint())
##--CELL--##
# Solve the problem using CplexOptimizer
optimizer = CplexOptimizer()
result = optimizer.solve(problem)

print("Solution:")
print(result.fval)
print(result.x)
##--CELL--##
# convert problem to a QUBO
converter = QuadraticProgramToQubo()
qubo = converter.convert(problem)
print(qubo.export_as_lp_string())
##--CELL--##
linear = qubo.objective.linear.to_array()
quadratic = qubo.objective.quadratic.to_array()
##--CELL--##
weight_max_cut_qubo = QUBO(quadratic, linear)
weight_max_cut_qubo.linear_to_square()
max_cut_graph = weight_max_cut_qubo.to_maxcut()
print(max_cut_graph)
##--CELL--##
max_cut = Maxcut(max_cut_graph)
problem_max_cut = max_cut.to_quadratic_program()
print(problem_max_cut.export_as_lp_string())
##--CELL--##
graph = create_graph_from_weight_matrix(max_cut_graph)
draw_graph(graph, "lightblue", nx.spring_layout(graph))
##--CELL--##
pauli_encoder = PauliCorrelationEncoding()

#edges,weights = pauli_encoder.get_edges_from_qp(problem)
k = 2    # type of compression (quadratic or cubic)
num_qubits = pauli_encoder.find_n(problem_max_cut.get_num_binary_vars(),k)


pauli_strings = SparsePauliOp(pauli_encoder.generate_pauli_strings(num_qubits,problem_max_cut.get_num_binary_vars(), k))


print(f"We can encode the problem with {num_qubits} qubits using {len(pauli_strings)} Pauli strings using k={k} compression,\n which are {pauli_strings}")


##--CELL--##
depth = 2 * num_qubits
# num_nodes = graph.number_of_nodes()
ansatz = pauli_encoder.BrickWork(depth= depth, num_qubits=num_qubits)
ansatz.draw('mpl',fold=-1)
##--CELL--##
pce = PauliCorrelationOptimizer(estimator=estimator, 
                                pauli_encoder=pauli_encoder,
                                depth=depth,
                                qp = problem_max_cut,
                                graph=graph,
                                num_qubits=num_qubits,
                                k=k,
                                max_cut_graph=max_cut_graph,
                                method='exact', 
                                loss_method='maxcut',
                                multi_op=False)  # method can be 'exact' or 'quantum' and loss can 
                                                # be qubo or maxcut

##--CELL--##
maxiter = 30
from qiskit_algorithms.optimizers import COBYLA,SLSQP,POWELL
optimizer = SLSQP(maxiter=maxiter)
params = np.random.rand(ansatz.num_parameters)
result=pce.optimize(optimizer,params)

##--CELL--##
final_ansatz = pauli_encoder.BrickWork(depth= depth, num_qubits=num_qubits).assign_parameters(result)
final_ansatz.draw('mpl',fold=-1)
##--CELL--##
psi_final = Statevector(final_ansatz)
##--CELL--##
# Example usage:
# Assuming you have instantiated `psi_final`, `pauli_strings`, and `weight_matrix`
max_cut_utility = MaxCutUtility(max_cut_graph)
initial_score, max_cut_solution_pce = max_cut_utility.evaluate_initial_score(psi_final, pauli_strings)
print(f"Initial score: {initial_score}")
print(f"Max cut solution: {max_cut_solution_pce}")
##--CELL--##
# convert to QUBO cost
qubo_solution_string = max_cut_utility.max_cut_to_qubo_solution(max_cut_solution_pce)
# get the qubo cost from the string
qubo_cost = qubo.objective.evaluate(qubo_solution_string)
print(f"QUBO cost: {qubo_cost}")
print(f"Qubo Solution Bitstring: {qubo_solution_string}")
##--CELL--##

result_initial = converter.interpret(qubo_solution_string)
initial_cost = problem.objective.evaluate(result_initial)
# check feasibility
inital_feasible = problem.get_feasibility_info(result_initial)[0]

print("Initial Knapsack score             :", initial_cost )
print("Initial Knapsack solution          :", result_initial)
print("Initial Knapsack solution feasible :", inital_feasible)
##--CELL--##
optimized_bitstring, final_cost = max_cut_utility.bit_swap_search(qubo, bitstring=qubo_solution_string)
print("Optimized bitstring:", optimized_bitstring)
print("Final cost:", final_cost)
##--CELL--##

result_initial = converter.interpret(optimized_bitstring)
initial_cost = problem.objective.evaluate(result_initial)
# check feasibility
inital_feasible = problem.get_feasibility_info(result_initial)[0]

print("Initial Knapsack score             :", initial_cost )
print("Initial Knapsack solution          :", result_initial)
print("Initial Knapsack solution feasible :", inital_feasible)
##--CELL--##
pauli_encoder = PauliCorrelationEncoding()

#edges,weights = pauli_encoder.get_edges_from_qp(problem)
k = 2    # type of compression (quadratic or cubic)
num_qubits = pauli_encoder.find_n(qubo.get_num_binary_vars(),k)


pauli_strings = SparsePauliOp(pauli_encoder.generate_pauli_strings(num_qubits,qubo.get_num_binary_vars(), k))


print(f"We can encode the problem with {num_qubits} qubits using {len(pauli_strings)} Pauli strings using k={k} compression,\n which are {pauli_strings}")


##--CELL--##
depth = 8
ansatz = pauli_encoder.BrickWork(depth= depth, num_qubits=num_qubits)
ansatz.draw('mpl',fold=-1)
##--CELL--##
pce = PauliCorrelationOptimizer(estimator=estimator, 
                                pauli_encoder=pauli_encoder,
                                depth=depth,
                                num_qubits=num_qubits,
                                k=k,
                                method='exact', 
                                loss_method='qubo',
                                qubo=qubo,
                                multi_op=True,
                                steps = 5)  # method can be 'exact' or 'quantum' and loss can 
                                                # be qubo or maxcut


##--CELL--##
maxiter = 30
from qiskit_algorithms.optimizers import SLSQP
optimizer =SLSQP(maxiter=maxiter)
params = np.random.rand(ansatz.num_parameters)
result=pce.optimize(optimizer,params)

##--CELL--##
final_ansatz = pauli_encoder.BrickWork(depth= depth, num_qubits=num_qubits).assign_parameters(result)
final_ansatz.draw('mpl',fold=-1)
##--CELL--##
psi_final = Statevector(final_ansatz)
##--CELL--##
# Example usage:
# Assuming you have instantiated `psi_final`, `pauli_strings`, and `weight_matrix`
qubo_utility = QUBOUtility()
qubo_bitstring = qubo_utility.evaluate_sign_function(psi_final, pauli_strings)
qubo_cost = qubo.objective.evaluate(qubo_bitstring)

print(qubo_bitstring)
print(qubo_cost)
##--CELL--##
result_initial = converter.interpret(qubo_bitstring)
initial_cost = problem.objective.evaluate(result_initial)
# check feasibility
inital_feasible = problem.get_feasibility_info(result_initial)[0]

print("Initial Knapsack score             :", initial_cost )
print("Initial Knapsack solution          :", result_initial)
print("Initial Knapsack solution feasible :", inital_feasible)
##--CELL--##
optimized_bitstring, final_cost = qubo_utility.bit_swap_search(qubo, bitstring=qubo_bitstring)
print("Optimized bitstring:", optimized_bitstring)
print("Final cost:", final_cost)
##--CELL--##

result_initial = converter.interpret(optimized_bitstring)
initial_cost = problem.objective.evaluate(result_initial)
# check feasibility
inital_feasible = problem.get_feasibility_info(result_initial)[0]

print("Initial Knapsack score             :", initial_cost )
print("Initial Knapsack solution          :", result_initial)
print("Initial Knapsack solution feasible :", inital_feasible)