import numpy as np

class QUBO:

    def __init__(self, quad: np.array, linear: np.array) -> None:
        self.num_vars = self._check_shape(quad, linear)

        self.quad = quad.copy()
        self.linear = linear.copy()

    def to_maxcut(self) -> np.array:
        """
        Create a graph where its MAXCUT problem is equivalent to the original QUBO 
        Assume that the QUBO is in the following form, i.e. the linear terms are on the diagonal:
            Q = sum_{i=1}^{n} sum_{j=1}^{n} q_{i,j} * x_i * x_j
        """
        graph = np.zeros((self.num_vars + 1, self.num_vars + 1))

        # node 0 to all other nodes
        for i in range(1, self.num_vars + 1):
            graph[0, i] = np.sum(self.quad[i-1, :]) + np.sum(self.quad[:, i-1])
            graph[i, 0] = graph[0, i]
        
        # all other nodes to each other
        for i in range(1, self.num_vars + 1):
            for j in range(i + 1, self.num_vars + 1):
                graph[i, j] = self.quad[i-1, j-1] + self.quad[j-1, i-1]
                graph[j, i] = graph[i, j]

        return graph

    def linear_to_square(self) -> None:
        """
        Convert linear terms (c_i * x_i) to square terms (c_ii * x_i^2)
        """
        for i in range(self.num_vars):
            self.quad[i, i] += self.linear[i]
        self.linear = np.zeros(self.num_vars)
    
    def square_to_linear(self) -> None:
        """
        Convert square terms (c_ii * x_i^2) to linear terms (c_i * x_i)
        """
        for i in range(self.num_vars):
            self.linear[i] += self.quad[i, i]
        np.fill_diagonal(self.quad, 0)

    def _check_shape(self, quad: np.array, linear: np.array) -> None:
        """
        Check if the shape of the linear and quadratic terms match each other
        Return the number of variables in the QUBO
        """
        quad_shape, linear_shape = quad.shape, linear.shape
        
        #TODO: Add check for data type 
        quad_dtype, linear_dtype = quad.dtype, linear.dtype

        if len(quad_shape) != 2:
            raise ValueError("The quadratic terms are not 2D")
        if quad_shape[0] != quad_shape[1]:
            raise ValueError("The quadratic terms are not square")
        if len(linear_shape) != 1:
            raise ValueError("The linear terms are not 1D")
        if quad_shape[0] != linear_shape[0]:
            raise ValueError("The shape of the linear and quadratic terms do not match")
        
        return quad_shape[0]

    