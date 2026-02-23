import math
import matplotlib.pyplot as plt
from IPython.display import display, clear_output
from scipy.optimize import minimize
import numpy as np
from qiskit.quantum_info import Pauli, SparsePauliOp, Statevector
import pickle
from .qubo_utility import *
from .pauli_correlation_encoding import PauliCorrelationEncoding

def save_checkpoint(filename, params, cost, round_num):
    """
    Saves the current state of optimization to a file.

    Args:
    - filename (str): Path to save the checkpoint file.
    - params (np.ndarray): Optimized parameters at the checkpoint.
    - cost (float): The cost value associated with the parameters.
    - round_num (int): Current optimization round.
    """
    checkpoint = {'params': params, 'cost': cost, 'round_num': round_num}
    with open(filename, 'wb') as f:
        pickle.dump(checkpoint, f)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(filename):
    """
    Loads a saved optimization state from a file.

    Args:
    - filename (str): Path to the checkpoint file.

    Returns:
    - dict: Loaded checkpoint containing params, cost, and round_num.
    """
    with open(filename, 'rb') as f:
        checkpoint = pickle.load(f)
    print(f"Checkpoint loaded from {filename}")
    return checkpoint

def get_edges_from_qubo(Q):
        
        """
        Get the edges of the graph from the QUBO matrix.
        """
        E = []
        for i in range(len(Q)):
            for j in range(i, len(Q)):
                if Q[i][j] != 0:
                    E.append((i, j))


        return E

def adaptive_perturbation(params, perturbation_factor, failure_count, historical_trend, max_factor=1e-2):
    random_perturbation = np.random.normal(0, perturbation_factor * (1 + failure_count / 5), size=params.shape)
    directional_perturbation = perturbation_factor * np.sign(historical_trend)
    perturbed_params = params + random_perturbation + directional_perturbation
    # Ensure perturbed parameters remain in valid range
    return np.clip(perturbed_params, -np.pi, np.pi)


def weighted_blended_initialization(previous_params, performance_weights, blend_factor=0.7):
    """Blends previous parameters with random ones, weighted by performance."""
    random_params = np.random.rand(len(previous_params))
    weighted_params = performance_weights * previous_params
    return blend_factor * weighted_params + (1 - blend_factor) * random_params

def initialize_within_range(num_params, lower_bound=-np.pi, upper_bound=np.pi):
    """Initializes parameters uniformly within a specified range."""
    return np.random.uniform(lower_bound, upper_bound, size=num_params)

def history_based_cooling_schedule(temperature, alpha, improvement_history, window=5):
    """Dynamically adjusts cooling based on recent improvement trends."""
    if len(improvement_history) >= window:
        recent_trend = sum(improvement_history[-window:])
        if recent_trend > 0:
            alpha = min(alpha * 1.05, 0.99)
        else:
            alpha = max(alpha * 0.95, 0.8)
    return max(temperature * alpha, 1e-8)




class PauliCorrelationOptimizer:
    def __init__(self, estimator, pauli_encoder, depth, qp=None, graph=None, num_qubits=None, 
                 k=None, max_cut_graph=None, method="exact", 
                 loss_method="qubo", qubo=None, multi_op=False, steps = None):
        """
        Initialize the class with the required parameters.

        Parameters:
        estimator: The estimator object (only used if the method is not 'exact').
        pauli_encoder: The Pauli encoder for quantum optimization.
        depth: The depth of the quantum circuit.
        qp: Quadratic programming problem instance (required for non-'qubo' loss methods).
        graph: Graph instance for optimization (required for non-'qubo' loss methods).
        num_qubits: Number of qubits in the quantum system.
        k: Parameter for the problem instance.
        num_nodes: Number of nodes in the graph (required for non-'qubo' loss methods).
        max_cut_graph: Graph for Max-Cut optimization (required for non-'qubo' loss methods).
        method: String indicating the method ('exact' or 'quantum').
        loss_method: String indicating the loss method ('qubo' or others).
        qubo: Data structure for QUBO optimization (required for 'qubo' loss method).
        """
        self.method = method
        self.estimator = estimator if method != "exact" else None
        self.loss_method = loss_method
        self.pauli_encoder = pauli_encoder
        self.depth = depth
        self.num_qubits = num_qubits
        self.k = k
        self.multi_op = multi_op

        if self.loss_method == "qubo":
            # Parameters specific to QUBO
            self.qubo = qubo
            self.qp = None
            self.graph = None
            #self.num_nodes = None
            self.max_cut_graph = None
        else:
            # Parameters specific to non-QUBO methods
            self.qubo = None
            self.qp = qp
            self.graph = graph
            #self.num_nodes = num_nodes
            self.max_cut_graph = max_cut_graph
        if self.multi_op:
            self.steps = steps
        else:
            self.steps = None

        self.cost_history_dict = {"prev_vector": None, "iters": 0, "cost_history": []}    

        self.min_loss_observed = float('inf')
        self.max_loss_observed = float('-inf')

    

    def hyperparameters(self, qp, graph, k,qubo):
        """
        Find the optimal hyperparameters for the PauliCorrelationOptimizer.

        Args:
        - qp: QuadraticProgram
        - graph: Input graph for nu calculation
        - k: Some parameter for pauli_encoder

        Returns:
        - alpha, beta, m, nu: Hyperparameters
        """
        num_qubits = self.num_qubits
        if self.loss_method == "qubo":
            Q = qubo.objective.quadratic.to_array(symmetric=True)
            #num_qubits = self.num_qubits
            m = len(Q)
            c = 0.5
            nu = c * np.sqrt(np.sum(Q**2))
        else:
            #num_qubits = self.pauli_encoder.find_n(qp.get_num_vars(), k)
            #num_nodes = graph.number_of_nodes()
            m = qp.get_num_binary_vars()
            nu = self.pauli_encoder.calculate_nu(graph)
        
        # floor_k_over_2 = math.floor(k / 2)
        #alpha = num_qubits ** floor_k_over_2
        alpha = 1.5 * num_qubits
        beta = 0.5
        
        

        return alpha, beta, m, nu, k


    def extra_qubo_loss(self,psi, qubo):
        

        alpha, beta, m, nu, k = self.hyperparameters(None,None,self.k,qubo)
        sum_tanh_squared = 0.0
        operators = SparsePauliOp(self.pauli_encoder.generate_pauli_strings(self.num_qubits, m, k))
        linear_terms = qubo.objective.linear.to_array()

        for index,op in enumerate(operators):
            if self.method == "exact":
                pi_i = psi.expectation_value(op).real
                sum_tanh_squared += linear_terms[index] *  np.tanh(pi_i*alpha)  
            else:                                                 
                job_i = self.estimator.run(psi, op)
                pi_i = job_i.result().values[0]
                sum_tanh_squared += (linear_terms[index] *  np.tanh(pi_i*alpha))

               
        return sum_tanh_squared


    def regularization_loss(self, psi, qp, graph, k,qubo):
        """
        Calculate the regularization loss for the PauliCorrelationOptimizer.

        Args:
        - psi: QuantumCircuit or object representing the state.
        - qp: QuadraticProgram.
        - graph: Input graph for nu calculation.
        - k: Some parameter for pauli_encoder.

        Returns:
        - float: Regularization loss.
        """
        if self.loss_method != "qubo":
            alpha, beta, m, nu, k = self.hyperparameters(qp, graph, k,None)
        else:
            alpha, beta, m, nu, k = self.hyperparameters(None,None,self.k,qubo)
        sum_tanh_squared = 0.0
        operators = SparsePauliOp(self.pauli_encoder.generate_pauli_strings(self.num_qubits, m, k))

        for op in operators:
            if self.method == "exact":
                pi_i = psi.expectation_value(op).real
            else:  # "quantum"
                job_i = self.estimator.run(psi, op)
                pi_i = job_i.result().values[0]
            sum_tanh_squared += np.tanh(alpha * pi_i) ** 2

        mean_tanh_squared = sum_tanh_squared / m
        regularization_term = beta * nu * (mean_tanh_squared ** 2)

        return regularization_term
    

    def loss(self, psi, qp, graph, k,qubo):
        """
        Calculate the loss function for the PauliCorrelationOptimizer.

        Args:
        - psi: QuantumCircuit representing the ansatz.
        - qp: QuadraticProgram.
        - graph: Input graph for nu calculation.
        - k: Some parameter for pauli_encoder.

        Returns:
        - float: Loss function value.
        """
        if self.loss_method != "qubo":
            alpha, beta, m, nu, k = self.hyperparameters(qp, graph, k,None)
            E = self.pauli_encoder.get_edges_from_qp(qp)
            W = self.max_cut_graph
        else:
            alpha, beta, m, nu, k = self.hyperparameters(None,None,self.k,qubo)
            W = qubo.objective.quadratic.to_array(symmetric=True)
            E = get_edges_from_qubo(W)
        
        
        operators = SparsePauliOp(self.pauli_encoder.generate_pauli_strings(self.num_qubits, m, k))

        loss = 0.0

        for (i, j) in E:
            if self.method == "exact":
                psi_i = psi.expectation_value(operators[i]).real
                psi_j = psi.expectation_value(operators[j]).real
            else:  # "quantum"
                job_i = self.estimator.run(psi, operators[i])
                psi_i = job_i.result().values[0]
                job_j = self.estimator.run(psi, operators[j])
                psi_j = job_j.result().values[0]
            term = W[i, j] * np.tanh(alpha * psi_i) * np.tanh(alpha * psi_j)
            loss += term  

        return loss
    
    def total_loss(self,psi,qp,graph,k,qubo):
        """
        Calculate the total loss function for the PauliCorrelationOptimizer.

        Args:
        - ansatz: QuantumCircuit representing the ansatz
        - qp: QuadraticProgram
        - graph: Input graph for nu calculation
        - k: Some parameter for pauli_encoder

        Returns:
        - float: Total loss function value.
        """
        if self.loss_method != "qubo":
            loss = self.loss(psi,qp,graph,k,None)
            l_reg = self.regularization_loss(psi,qp,graph,k,None)
            total_loss = loss + l_reg  
        else:
            total_loss = self.loss(psi,None,None,self.k, qubo) + self.extra_qubo_loss(psi,qubo)+ self.regularization_loss(psi,None,None,self.k,qubo)
        
        return total_loss
    
    def solve(self, params):
        """
        Solve the optimization problem.

        Args:
        - params: Parameters for the quantum circuit.

        Returns:
        - float: Loss function value.
        """
        ansatz = self.pauli_encoder.BrickWork(depth=self.depth, num_qubits=self.num_qubits)
        ansatz = ansatz.assign_parameters(params)
        if self.method == "exact":
            psi = Statevector(ansatz)
        else:  # "quantum"
            psi = ansatz
        if self.loss_method != "qubo":
            loss = self.total_loss(psi, self.qp, self.graph, self.k,None)
        else:
            loss = self.total_loss(psi,None,None,self.k, self.qubo)

        #self.min_loss_observed = min(self.min_loss_observed, loss)
        #self.max_loss_observed = max(self.max_loss_observed, loss)
        self.cost_history_dict["iters"] += 1
        # Normalize using observed range
        #scaled_loss = (loss - self.min_loss_observed) / (self.max_loss_observed - self.min_loss_observed + 1e-8)
        
        self.cost_history_dict["cost_history"].append(loss)
        #print(f"Iters. done: {self.cost_history_dict['iters']} [Current cost: {loss}]")


        # Live plotting
        iters = list(range(1, self.cost_history_dict["iters"] + 1))
        costs = self.cost_history_dict["cost_history"]
        plt.figure(figsize=(10, 6))
        plt.plot(iters, costs, marker='o', linestyle='-', color='b')
        plt.xlabel("Iterations")
        plt.ylabel("Cost")
        plt.title("Optimization Progress: Iterations vs. Cost")
        plt.grid(True)
        plt.tight_layout()

        # Display plot and update live
        clear_output(wait=True)
        display(plt.gcf())
        plt.close()
        
        return loss
        
    def optimize(self,optimizer,params):
        """
        Optimize the ansatz using the PauliCorrelationOptimizer.

        Args:
        - initial_params: Initial parameters for the ansatz

        Returns:
        - float: Optimized loss function value.
        """

        if self.multi_op:
            print("Multi Reoptimization")

            # Initialization
            best_params = params.copy()
            best_qubo_cost = float('inf')
            no_improvement_count = 0
            max_no_improvement_rounds = self.steps  # Allow more rounds for exploration
            perturbation_factor = 1e-2  # Start with a higher initial perturbation
            decay_factor = 0.95  # Gradual decay for perturbation
            exploration_factor = 1.5  # Amplify exploration periodically
            restart_threshold = self.steps/2  # Restart after 5 consecutive failures
            improvement_history = []
            historical_trend = np.zeros_like(params)

            round_num = 0
            while no_improvement_count < max_no_improvement_rounds:
                round_num += 1
                print(f"\n--- Optimization Round {round_num} ---")

                # Dynamic perturbation scaling
                if round_num % 3 == 0:  # Periodic stronger exploration
                    # print("Performing stronger exploration.")
                    current_perturbation_factor = perturbation_factor * exploration_factor
                else:
                    current_perturbation_factor = perturbation_factor

                # Apply adaptive perturbation
                perturbed_params = adaptive_perturbation(
                    params, current_perturbation_factor, no_improvement_count, historical_trend
                )
                method = "Nelder-Mead"  # Change this to other methods like "COBYLA", "Powell", etc., if required.
                options = {"maxiter": 500, "disp": True}
                #result = optimizer.minimize(fun=self.solve, x0=perturbed_params)
                result = minimize(
                    fun=self.solve,  # Cost function to minimize
                    x0=perturbed_params,       # Initial parameters
                    method=method,  # Additional options
                    options=options
                )
                
                optimized_params = result.x
                pauli_encoder = PauliCorrelationEncoding()
                final_ansatz = pauli_encoder.BrickWork(depth= self.depth, num_qubits=self.num_qubits).assign_parameters(optimized_params)
                psi_final = Statevector(final_ansatz)
                qubo_utility = QUBOUtility()
                pauli_strings = SparsePauliOp(pauli_encoder.generate_pauli_strings(self.num_qubits,self.qubo.get_num_binary_vars(), self.k))
                qubo_bitstring = qubo_utility.evaluate_sign_function(psi_final, pauli_strings)
                # print(f"QUBO bitstring: {qubo_bitstring}")
                qubo_cost = self.qubo.objective.evaluate(qubo_bitstring)
                print(f"Best QUBO cost: {best_qubo_cost}")
                print(f"QUBO cost: {qubo_cost}")

                # Track improvements
                improvement = best_qubo_cost - qubo_cost
                if improvement > 0:
                    print(f"Improvement: {improvement}")
                else:
                    print(f"No improvement detected.")
                improvement_history.append(improvement)
                historical_trend = np.sign(optimized_params - params) * improvement

                # Update the best solution
                if qubo_cost < best_qubo_cost:
                    print(f"Improvement detected! Best QUBO cost updated: {qubo_cost}")
                    best_params = optimized_params.copy()
                    best_qubo_cost = qubo_cost
                    no_improvement_count = 0  # Reset failure count
                    perturbation_factor *= decay_factor  # Decay perturbation for precision
                else:
                    print(f"No improvement in round {round_num}.")
                    no_improvement_count += 1

                    # Restart or adapt exploration
                    if no_improvement_count >= restart_threshold:
                        print("Random restart triggered due to stagnation.")
                        params = np.random.uniform(-np.pi, np.pi, size=params.shape)  # Restart with random initialization
                        perturbation_factor = 1e-2  # Reset perturbation factor
                    elif no_improvement_count % 2 == 0:
                        # print("Exploring broader solution space using weighted blending.")
                        params = weighted_blended_initialization(best_params, historical_trend)
                    else:
                        # print("Focusing locally with stronger perturbation.")
                        params = adaptive_perturbation(
                            best_params, perturbation_factor * 2, no_improvement_count, historical_trend
                        )

                # Display remaining rounds before stopping
                rounds_remaining = max_no_improvement_rounds - no_improvement_count
                print(f"Consecutive no-improvement rounds: {no_improvement_count}. Rounds remaining before stopping: {rounds_remaining}.")

                # Save checkpoint periodically
                if round_num % 5 == 0:
                    print(f"Saving checkpoint at round {round_num}.")
                    save_checkpoint(f"checkpoint_round_{round_num}.pkl", best_params, best_qubo_cost, round_num)

            print("\nOptimization complete.")
            print(f"Best QUBO cost: {best_qubo_cost}")


            return best_params
        else:    
            print("Single Optimization")
            method = "Nelder-Mead"  # Change this to other methods like "COBYLA", "Powell", etc., if required.
            options = {"maxiter": 500, "disp": True}

            # Optimization loop
            result = minimize(
                fun=self.solve,  # Cost function to minimize
                x0=params,       # Initial parameters
                method=method,  # Additional options
                options=options
            )

            #result = optimizer.minimize(fun = self.solve, x0=params)
            return result.x
        


