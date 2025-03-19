# Multi-Dimensional Knapsack Problem (MDKP)

The **Multi-Dimensional Knapsack Problem (MDKP)** is an extension of the classical **Knapsack Problem**, where multiple constraints (dimensions) must be satisfied. It is widely used in resource allocation, logistics, and finance due to its real-world applicability.

---

## 1. Problem Statement

### **Input**
1. **Items**:
   - $n$ items, where each item $i$ has:
     - **Profit** $p_i$: The value or profit of the item.
     - **Weight** $w_{ij}$: The weight of the item in dimension $j$.

2. **Constraints**:
   - $m$ knapsack constraints, where each dimension $j$ has a capacity $c_j$.

---

### **Decision Variable**
- $x_i$: A binary variable indicating whether item $i$ is selected ($x_i = 1$) or not ($x_i = 0$).

---

### **Objective**
Maximize the total profit:
$$ \text{Maximize } Z = \sum_{i=1}^{n} p_i \cdot x_i $$

---

### **Constraints**
Ensure the total weight in each dimension does not exceed its capacity:
$$ \sum_{i=1}^{n} w_{ij} \cdot x_i \leq c_j \quad \forall j \in \{1, 2, \ldots, m\} $$

---

## 2. Key Characteristics

1. **Multiple Constraints**:
   - Unlike the classical knapsack problem, MDKP has $m$ constraints, making it multi-dimensional.

2. **NP-Hard Problem**:
   - Finding the optimal solution is computationally challenging due to its combinatorial nature.

3. **Applications**:
   - **Resource Allocation**: Assign resources to projects with budget and manpower constraints.
   - **Logistics**: Ship items while considering weight, volume, and other limits.
   - **Finance**: Optimize investments under risk and budget constraints.

---

## 3. Example

### **Problem Setup**
- **Items** ($n = 4$):
  - Profits: $p = [10, 20, 30, 40]$
  - Weights:
    - Dimension 1: $w_1 = [2, 3, 4, 5]$
    - Dimension 2: $w_2 = [1, 2, 3, 4]$
- **Capacities** ($m = 2$):
  - $c_1 = 8$
  - $c_2 = 5$

### **Formulation**
Maximize:
$$ Z = 10x_1 + 20x_2 + 30x_3 + 40x_4 $$

Subject to:
$$ 2x_1 + 3x_2 + 4x_3 + 5x_4 \leq 8 \quad (\text{Dimension 1}) $$
$$ 1x_1 + 2x_2 + 3x_3 + 4x_4 \leq 5 \quad (\text{Dimension 2}) $$
$$ x_i \in \{0, 1\} \quad \forall i \in \{1, 2, 3, 4\} $$

---

## 4. Mathematical Representation

1. **Variables**:
   - $x_i \in \{0, 1\}$: Binary variables indicating item inclusion.

2. **Objective Function**:
   - Maximize:
     $$ Z = \sum_{i=1}^{n} p_i \cdot x_i $$

3. **Constraints**:
   - Satisfy:
     $$ \sum_{i=1}^{n} w_{ij} \cdot x_i \leq c_j \quad \forall j \in \{1, 2, \ldots, m\} $$

---

## 5. Solution Methods

### **Exact Methods**
1. **Branch-and-Bound**:
   - Explore the solution space systematically by branching and pruning suboptimal solutions.
2. **Dynamic Programming**:
   - Solve using a state-based recursive approach (limited to small problem sizes).
3. **Mixed-Integer Linear Programming (MILP)**:
   - Use solvers like **CPLEX**, **Gurobi**, or **SCIP**.

---

### **Heuristic and Metaheuristic Methods**
1. **Greedy Algorithms**:
   - Select items based on profit-to-weight ratio.
2. **Genetic Algorithms (GA)**:
   - Evolve solutions through selection, crossover, and mutation.
3. **Simulated Annealing (SA)**:
   - Explore the solution space probabilistically to escape local optima.
4. **Ant Colony Optimization (ACO)** and **Particle Swarm Optimization (PSO)**:
   - Use nature-inspired approaches for approximate solutions.

---

## 6. Numerical Example

### **Problem**
- Items: $p = [10, 20, 30]$
- Weights: $w_1 = [2, 3, 4], w_2 = [3, 1, 5]$
- Capacities: $c_1 = 5, c_2 = 4$

### **Solution**
1. Compute profit-to-weight ratios:
   $$ \text{Ratios (Dimension 1): } \frac{p_1}{w_1} = 5, \frac{p_2}{w_1} = 6.67, \frac{p_3}{w_1} = 7.5 $$
   $$
   \text{Ratios (Dimension 2): } \frac{p_1}{w_2} = 3.33, \frac{p_2}{w_2} = 20, \frac{p_3}{w_2} = 6
   $$

2. Use a solver to maximize the total profit under constraints.

### **Result**
- Selected Items: $x = [1, 1, 0]$
- Total Profit: $Z = 10 + 20 = 30$

---

## 7. Applications in Real Life

### **Logistics**:
- Optimize cargo shipments considering weight, volume, and cost.

### **Cloud Resource Allocation**:
- Assign virtual machines to servers while meeting CPU, memory, and storage limits.

### **Finance**:
- Choose investments to maximize returns while adhering to budget and risk constraints.

---

## 8. Variants of MDKP

1. **Fractional MDKP**:
   - Items can be fractionally included ($x_i \in [0, 1]$).

2. **Multi-Objective MDKP**:
   - Optimize multiple objectives, such as profit and fairness.

3. **Dynamic MDKP**:
   - Problem parameters (e.g., capacities) change over time.

---

The **Multi-Dimensional Knapsack Problem** generalizes well to practical scenarios, offering a rich domain for research and real-world optimization. Let me know if you'd like help solving specific instances or exploring further!
