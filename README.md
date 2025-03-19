# Quantum Optimization Benchmarks

This repository, **Quantum Optimization Benchmarks**, provides a collection of benchmark datasets and Jupyter notebooks for solving combinatorial optimization problems. The repository includes benchmark instances for problems like **Market Share**, **Maximum Independent Set**, **Multi-Dimensional Knapsack**, and **Quadratic Assignment Problem**. It also provides Python code for formulating these problems and analyzing results.

See [Quantum Optimization Algorithms](https://github.com/MonitSharma/quantum_opt_algos) for implementation details

---

## Repository Structure

```plaintext
QUANTUM_OPTIMIZATION_BENCHMARKS/
- Market_Share/: Market share optimization problem
  - market_share_classical_results.ipynb: Classical results for the Market Share problem
  - market_share.ipynb: Code for solving the Market Share problem
  - readme.md: Documentation for the Market Share problem
- Maximum_Independent_Set/: Maximum Independent Set problem
  - mis_benchmark_instances/: Instances for the MIS problem
  - mis.ipynb: Code for solving the MIS problem
  - readme.md: Documentation for the MIS problem
- Multi_Dimension_Knapsack/: Multi-Dimensional Knapsack Problem
  - MKP_Instances/: Benchmark instances for MKP
  - mdkp.ipynb: Code for solving the MKP
  - readme.md: Documentation for the MKP problem
- Quadratic_Assignment_Problem/: Quadratic Assignment Problem
  - qapdata/: Benchmark instances for QAP
  - qap.ipynb: Code for solving the QAP
  - README.md: Documentation for the QAP problem
- requirements.txt: Python dependencies
```

---

## Getting Started


### Clone the Repository
```
git clone https://github.com/MonitSharma/quantum_optimization_benchmarks.git  
cd quantum-optimization-benchmarks  
```



### Set Up Virtual Environment
It is recommended to use a virtual environment to manage dependencies: 

```
python -m venv .venv  
source .venv/bin/activate  (Linux/macOS)  
.venv\Scripts\activate     (Windows)  
```

### Install Dependencies
Install the required Python libraries:  

```
pip install -r requirements.txt  
```

---

## Problem-Specific Details

### 1. Market Share Problem
- Directory: Market_Share/
- Description: A benchmark dataset and code to solve the Market Share problem using combinatorial optimization techniques.
- Notebook: market_share.ipynb contains the implementation.
- Benchmark Data: Details of classical results and test instances are stored in market_share_classical_results.ipynb.

### 2. Maximum Independent Set (MIS)
- Directory: Maximum_Independent_Set/
- Description: Instances for the Maximum Independent Set problem, which involves finding the largest subset of vertices such that no two are adjacent.
- Notebook: mis.ipynb provides the implementation.

### 3. Multi-Dimensional Knapsack Problem (MKP)
- Directory: Multi_Dimension_Knapsack/
- Description: Benchmark datasets and code for solving the Multi-Dimensional Knapsack Problem.
- Notebook: mdkp.ipynb contains the MKP implementation.
- Benchmark Data: Instances for testing are stored in MKP_Instances/.

### 4. Quadratic Assignment Problem (QAP)
- Directory: Quadratic_Assignment_Problem/
- Description: A benchmark dataset and implementation for the Quadratic Assignment Problem.
- Notebook: qap.ipynb contains the code for solving QAP.
- Benchmark Data: Stored in the qapdata/ directory.

---

## How to Use

1. Navigate to the problem-specific directory.
2. Open the Jupyter notebook (.ipynb) to explore the code.
3. Use the provided instances in the respective directories for testing.

Cite the paper, if you use this work

[A Comparative Study of Quantum Optimization Techniques for Solving Combinatorial Optimization Benchmark Problems](https://arxiv.org/abs/2503.12121)

```bash
@misc{sharma2025comparativestudyquantumoptimization,
      title={A Comparative Study of Quantum Optimization Techniques for Solving Combinatorial Optimization Benchmark Problems}, 
      author={Monit Sharma and Hoong Chuin Lau},
      year={2025},
      eprint={2503.12121},
      archivePrefix={arXiv},
      primaryClass={quant-ph},
      url={https://arxiv.org/abs/2503.12121}, 
}
```

---

## Contribution

We welcome contributions! If you have additional benchmark datasets, new formulations, or improvements, feel free to open an issue or submit a pull request.

---

## License

This repository is licensed under the MIT License.

---

## Contact

For questions or suggestions, please reach out to monitsharma@smu.edu.sg or open an issue in this repository.
