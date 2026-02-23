# Quantum Optimization Algorithms

Welcome to the **Quantum Optimization Algorithms** repository! This project contains implementations of various quantum optimization techniques designed for solving combinatorial optimization problems. By leveraging quantum computing, these algorithms aim to provide efficient solutions to problems that are computationally challenging for classical methods.

See [Quantum Optimization Benchmark](https://github.com/SMU-Quantum/quantum-optimization-benchmarks) for benchmarking instances.

## Table of Contents

- [Introduction](#introduction)
- [Implemented Algorithms](#implemented-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license)

## Introduction

Combinatorial optimization problems arise in many domains, including finance, logistics, and scheduling. Traditional optimization methods struggle to efficiently handle these problems as their size increases. Quantum optimization algorithms, such as QAOA and VQE, leverage quantum mechanics to find optimal or near-optimal solutions. This repository provides implementations of several quantum algorithms and their variants, along with benchmarking and performance evaluations.

## Implemented Algorithms

The repository includes implementations of the following quantum optimization algorithms:

- **Quantum Approximate Optimization Algorithm (QAOA)** – A variational algorithm for solving combinatorial problems by encoding them into quantum Hamiltonians.
- **Quantum Random Access Optimization (QRAO)** – A qubit-efficient approach designed to reduce hardware requirements while maintaining solution quality.
- **Warm Start QAOA** – An improved QAOA method that initializes parameters using classical solutions to enhance convergence.
- **Multi-Angle QAOA (MA-QAOA)** – A variant of QAOA that introduces multiple parameters per layer for improved performance.
- **Conditional Value-at-Risk QAOA (CVaR QAOA)** – A risk-aware modification of QAOA that focuses on optimizing the best subset of measurement outcomes.
- **Variational Quantum Eigensolver (VQE)** – A hybrid quantum-classical algorithm that minimizes the expectation value of a problem Hamiltonian.
- **Conditional Value-at-Risk VQE (CVaR VQE)** – A refinement of VQE that prioritizes high-quality solutions by filtering measurement results.
- **Pauli Correlation Encoding (PCE)** – A qubit-efficient encoding method that incorporates classical post-processing, such as bit swap search, to refine solutions.

## Installation

To set up the repository, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/SMU-Quantum/quantum-optimization-algorithms
   cd quantum_opt_algos
   ```

2. **Set up a virtual environment (recommended):**
    ```bash
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**
    Install the required dependencies using:
    ```bash
    pip install -r requirements.txt
    ```

    Ensure that your system has Python 3.9 or higher installed.


## Usage

This repository provides implementations of various quantum optimization algorithms. Each algorithm is structured as a standalone module and can be executed independently.

### Running an Algorithm

To run a specific quantum optimization algorithm, use:

```bash
python algorithms/<algorithm_name>.py
```

## Contribution

We welcome contributions to enhance and expand this repository. If you have improvements, bug fixes, or new quantum optimization algorithms to add, follow these steps:

### How to Contribute

1. **Fork the repository** by clicking the "Fork" button at the top right of this page.
2. **Clone your fork** to your local machine:
   ```bash
   git clone https://github.com/your-username/quantum_opt_algos.git
   cd quantum_opt_algos
    ```

3. **Create a new branch** for your feature or fix:

    ```bash
    git checkout -b feature/new_algorithm
    ```
4. **Implement your changes** and ensure they follow best coding practices.

5. **Commit your changes** with a clear and descriptive message:

    ```bash
    git commit -m "Added implementation for XYZ algorithm"
    ```
6. **Push the changes** to your forked repository:

    ```bash
    git push origin feature/new_algorithm
    ```

7. **Open a pull request (PR)** on the main repository and provide a detailed description of your changes.


### Contribution Guidelines

We encourage contributions that improve the performance, scalability, and usability of quantum optimization algorithms. To ensure a smooth collaboration, please follow these guidelines:

- **Code Quality**: Maintain clean, modular, and well-documented code. Use meaningful variable names and follow best practices.
- **Commit Messages**: Provide clear and descriptive commit messages that summarize the changes made.
- **Algorithm Implementation**: If adding a new quantum optimization algorithm, include:
  - A brief explanation of the method.
  - Example usage in a script or Jupyter Notebook.
  - Performance benchmarks (if applicable).
- **Testing**: Ensure that your code runs correctly. If possible, add tests to validate correctness.
- **Pull Requests (PRs)**: When submitting a PR:
  - Clearly describe the purpose and impact of your changes.
  - Reference any related issues if applicable.
  - Ensure that the branch is up to date with the latest `main` branch.

We appreciate all contributions, whether it's adding new algorithms, improving efficiency, or fixing bugs. Your contributions help advance the field of quantum optimization!

---

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

---

## Acknowledgments

This repository is developed to support research in quantum optimization. Contributions from the open-source community play a crucial role in improving implementations and expanding the benchmark datasets.

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


Feel free to explore, contribute, and collaborate!
