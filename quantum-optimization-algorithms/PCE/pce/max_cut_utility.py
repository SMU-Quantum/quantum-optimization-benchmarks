import numpy as np
import random
import math
from collections import deque
from concurrent.futures import ThreadPoolExecutor
class MaxCutUtility:
    def __init__(self, weight_matrix):
        self.weight_matrix = weight_matrix

    @staticmethod
    def evaluate_sign_function(psi, operators):
        """
        Evaluate the sign function and binary values for each operator.

        Args:
            psi (Statevector): Quantum state vector.
            operators (list of SparsePauliOp): List of operators.

        Returns:
            list of int: List of binary values (0 or 1) for each operator.
        """
        binary_values = []

        for op in operators:
            pi_i = psi.expectation_value(op).real
            x_i = np.sign(pi_i)
            if x_i < 0:
                x_i = 0
            else:
                x_i = 1
            binary_values.append(x_i)

        return binary_values

    @staticmethod
    def evaluate_max_cut(solution, weight_matrix):
        """
        Evaluate the max cut value for a given solution.

        Args:
            solution (list of int): Binary list representing the solution.
            weight_matrix (numpy.ndarray): Weight matrix representing the graph.

        Returns:
            float: The value of the max cut.
        """
        cut_value = 0
        n = len(solution)
        for i in range(n):
            for j in range(i + 1, n):
                if solution[i] != solution[j]:
                    cut_value += weight_matrix[i, j]
        return cut_value
    
    @staticmethod
    def bit_swap_search(qubo, bitstring):
        """
        Perform a multi-bit flip and bit-swap search to optimize the QUBO objective value.

        Args:
            qubo: The QUBO object representing the optimization problem.
            bitstring: List[int], the initial bitstring to optimize.

        Returns:
            tuple: Optimized bitstring and its corresponding QUBO cost.
        """
        import numpy as np
        from itertools import combinations

        def evaluate_cost(bitstring):
            """Evaluate the cost of a bitstring using QUBO."""
            return qubo.objective.evaluate(np.array(bitstring, dtype=int))

        def compute_delta_cost(bitstring, indices):
            """
            Compute the change in cost caused by flipping multiple bits.

            Args:
                bitstring: The current bitstring.
                indices: A list of indices to flip.

            Returns:
                float: The change in cost.
            """
            # Flip the specified bits temporarily
            flipped_bitstring = bitstring[:]
            for i in indices:
                flipped_bitstring[i] = 1 - flipped_bitstring[i]

            # Compute the difference in cost
            original_cost = evaluate_cost(bitstring)
            flipped_cost = evaluate_cost(flipped_bitstring)
            return flipped_cost - original_cost

        # Initialize with the current solution
        best_bitstring = bitstring[:]
        best_cost = evaluate_cost(best_bitstring)

        print("Starting cost:", best_cost)

        # Perform single-bit flips
        for i in range(len(best_bitstring)):
            delta_cost = compute_delta_cost(best_bitstring, [i])
            if delta_cost < 0:  # Only flip if it improves the cost
                best_bitstring[i] = 1 - best_bitstring[i]
                best_cost += delta_cost
                print(f"Bit flip: Improved solution by flipping bit {i}: Cost = {best_cost}")

        # Perform multi-bit flips (pairs, triplets, etc.)
        for k in range(2, min(len(best_bitstring) + 1, 4)):  # Limit to flipping up to 3 bits
            for indices in combinations(range(len(best_bitstring)), k):
                delta_cost = compute_delta_cost(best_bitstring, indices)
                if delta_cost < 0:  # Only flip if it improves the cost
                    for i in indices:
                        best_bitstring[i] = 1 - best_bitstring[i]
                    best_cost += delta_cost
                    print(f"Multi-bit flip: Improved solution by flipping bits {indices}: Cost = {best_cost}")

        # Perform bit swaps
        for i in range(len(best_bitstring)):
            for j in range(i + 1, len(best_bitstring)):
                # Swap bits i and j
                swapped_bitstring = best_bitstring[:]
                swapped_bitstring[i] = 1 - swapped_bitstring[i]
                swapped_bitstring[j] = 1 - swapped_bitstring[j]

                # Compute the cost
                original_cost = evaluate_cost(best_bitstring)
                swapped_cost = evaluate_cost(swapped_bitstring)
                delta_cost = swapped_cost - original_cost

                if delta_cost < 0:  # Only swap if it improves the cost
                    best_bitstring[i] = 1 - best_bitstring[i]
                    best_bitstring[j] = 1 - best_bitstring[j]
                    best_cost += delta_cost
                    print(f"Bit swap: Improved solution by swapping bits {i} and {j}: Cost = {best_cost}")

        print("Final best cost:", best_cost)
        return best_bitstring, best_cost
    
    
    

    @staticmethod
    def max_cut_to_qubo_solution(cut_solution):
        """
        Convert the solution of the weighted max cut to the original QUBO solution.

        Parameters:
        cut_solution (list): A binary list representing the cut solution.

        Returns:
        list: The solution to the original QUBO problem.
        """
        n = len(cut_solution) - 1  # Subtract 1 because cut_solution includes the extra 0th node
        qubo_solution = []

        for i in range(1, n + 1):
            if cut_solution[i] != cut_solution[0]:
                qubo_solution.append(0)
            else:
                qubo_solution.append(1)

        return qubo_solution

    def evaluate_initial_score(self, psi, operators):
        """
        Evaluate the initial score for the max-cut solution.

        Args:
            psi (Statevector): Quantum state vector.
            operators (list of SparsePauliOp): List of operators.

        Returns:
            float: Initial max-cut score.
        """
        max_cut_solution_pce = self.evaluate_sign_function(psi, operators)
        initial_score = self.evaluate_max_cut(max_cut_solution_pce, self.weight_matrix)
        return initial_score, max_cut_solution_pce



