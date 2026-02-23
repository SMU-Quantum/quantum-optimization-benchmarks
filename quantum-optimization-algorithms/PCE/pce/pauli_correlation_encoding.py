# import necessary packages

import numpy as np
from math import comb
from typing import List
import networkx as nx

from itertools import combinations
from qiskit.circuit import QuantumCircuit, Parameter

class PauliCorrelationEncoding:

    def __init__(self):
        pass
    

    def get_edges_from_qp(self, qp):
        """
        Extract edges and weights from the quadratic program.

        Args:
            qp (QuadraticProgram): Quadratic program.

        Returns:
            E (list of tuples): List of edges.
            W (numpy.ndarray): Weight matrix.
        """
        quadratic_terms = qp.objective.quadratic.to_dict()
        E = []
        num_vars = qp.get_num_vars()
        
        for (i, j), weight in quadratic_terms.items():
            E.append((i, j))
            

        return E


    def find_n(self, m: int, k: int) -> int:
        """
        Find the smallest integer n such that m can be encoded with k-body Pauli relations,
        satisfying m <= 3 * comb(n, k).
        
        Args:
        - m: The number of binary variables.
        - k: The number of Pauli operators in each string.
        
        Returns:
        - n: The smallest integer such that m <= 3 * comb(n, k).
        """
        # Start with the smallest possible n
        n = 1
        while 3 * comb(n, k) < m:
            n += 1

        return n


    def generate_pauli_strings(self, n: int, m: int, k: int) -> List[str]:
        """
        Generate Pauli strings for n qubits with m strings and k-body Pauli correlation.

        Args:
        - n: The number of qubits
        - m: The number of Pauli strings to generate
        - k: The number of Pauli operators in each string

        Returns:
        - pauli_strings: A list of Pauli strings
        """
        # Ensure k is not greater than n
        if k > n:
            raise ValueError("k cannot be greater than n")
        
        # Calculate the maximum number of Pauli strings that can be generated
        max_pauli_strings = 3 * comb(n, k)
        
        # Check if the requested number of strings exceeds the maximum possible
        if m > max_pauli_strings:
            raise ValueError(f"The maximum number of Pauli strings that can be generated for n={n} and k={k} is {max_pauli_strings}. The requested m={m} exceeds this limit.")
        
        # Define Pauli operators
        pauli_ops = ['X', 'Y', 'Z']
        
        # Generate all possible positions for k Pauli operators in n qubits
        positions = list(combinations(range(n), k))

        # Initialize the list to hold the final Pauli strings
        pauli_strings = []

        # Generate the Pauli strings for each Pauli operator
        for op in pauli_ops:
            for pos in positions:
                pauli_string = ['I'] * n  # Start with all 'I'
                for i in pos:
                    pauli_string[i] = op  # Place the Pauli operator at the specified positions
                pauli_strings.append(''.join(pauli_string))  # Convert list to string and add to the result list
        
        # Ensure the output list length matches m
        return pauli_strings[:m]



    def calculate_nu(self, G):
        """
        Calculate the nu parameter based on the input graph G.

        Args:
            G: NetworkX graph or NumPy array representing a weight matrix.

        Returns:
            float: The calculated nu parameter.
        """
        if isinstance(G, nx.Graph):
            is_weighted = all('weight' in data for _, _, data in G.edges(data=True))
        elif isinstance(G, np.ndarray):
            is_weighted = True  # Assume it's a weighted matrix
        else:
            raise ValueError("Input must be a NetworkX graph or a NumPy array representing a weight matrix.")
        
        if is_weighted:
            # Weighted graph: calculate the Poljak-Turzik lower bound
            w_G = sum(data['weight'] for u, v, data in G.edges(data=True))

            # Calculate the minimum spanning tree of the graph
            mst = nx.minimum_spanning_tree(G, weight='weight')
            w_T_min = sum(data['weight'] for u, v, data in mst.edges(data=True))

            # Calculate nu
            nu = w_G / 2 + w_T_min / 4
        else:
            # Unweighted graph: calculate the Edwards-Erdos bound
            m = G.number_of_edges()
            n = G.number_of_nodes()
            
            # Edwards-Erdos bound calculation
            nu = m / 2 + (n - 1) / 4

        return nu


    def BrickWork(self,depth, num_qubits):
        """
        Generate a parameterized ansatz circuit with specified depth and number of qubits.
        
        Args:
        - depth: Depth of the circuit (should be an odd number)
        - num_qubits: Number of qubits in the circuit
        - params: List of parameters for the rotation gates
        
        Returns:
        - ansatz: A parameterized QuantumCircuit
        """
        
        ansatz = QuantumCircuit(num_qubits)
        
        # Extract parameters for the RY and RXX rotations
        phi = []
        for i in range(1, 2*depth*num_qubits + 1):
            phi.append(Parameter(f"phi{i}"))

        ry_params = np.asarray(phi[:depth*num_qubits])
        rxx_params = np.asarray(phi[depth*num_qubits:])
        
        

        splitted_ry = np.split(ry_params, depth)
        splitted_rxx = np.split(rxx_params, depth)
        
        for layer in range(depth):
            # Apply RY gates
            for i in range(num_qubits):
                if(layer % 3) == 1:
                    ansatz.ry(splitted_ry[layer][i], i)
                if(layer % 3) == 2:
                    ansatz.rx(splitted_ry[layer][i], i)
                if(layer % 3) == 0:
                    ansatz.rz(splitted_ry[layer][i], i)
                
                    
            
            # Apply RXX gates
            for i in range(0, num_qubits, 2):
                if i + 1 < num_qubits:
                    ansatz.rxx(splitted_rxx[layer][i//2], i, i + 1)

            ansatz.barrier()

            # Apply RY 2nd time
            for i in range(num_qubits):
                if(layer % 3) == 1:
                    ansatz.rz(splitted_ry[layer][i], i)
                if(layer % 3) == 2:
                    ansatz.ry(splitted_ry[layer][i], i)
                if(layer % 3) == 0:
                    ansatz.rx(splitted_ry[layer][i], i)

            # Apply RXX 2nd time
            for i in range(1, num_qubits, 2):
                if i + 1 < num_qubits:
                    ansatz.rxx(splitted_rxx[layer][(num_qubits//2) + i//2], i, i + 1)
            
            # Add a barrier for visual separation
            ansatz.barrier()
        
        return ansatz

