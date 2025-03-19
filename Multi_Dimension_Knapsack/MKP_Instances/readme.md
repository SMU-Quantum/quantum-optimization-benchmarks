The dataset can be found [here](https://www.researchgate.net/publication/271198281_Benchmark_instances_for_the_Multidimensional_Knapsack_Problem)


### Benchmark Instances for Multidimensional Knapsack Problem (MKP)

All benchmark instances from three well-known multidimensional knapsack problem (MKP) libraries are provided in a standard format here, as used in **Drake et al. (2016)**. This includes:

- **The SAC-94 dataset**, based on a variety of real-world problems.
- **The ORLib dataset**, proposed by Chu and Beasley (1998).
- **The GK dataset**, proposed by Glover and Kochenberger (n.d).

#### File Format
The format of these data files is as follows:
1. Number of variables (\( n \))
2. Number of constraints (\( m \))
3. Optimal value (0 if unavailable)
4. Profits (\( P(j) \)) for each \( n \)
5. An \( m \\times n \) matrix of constraints
6. Capacities (\( b(i) \)) for each \( m \)

#### Citation
Any use of these files is credited with a reference to **Drake et al. (2016)**. These instances were previously available at [http://www.cs.nott.ac.uk/~jqd/mkp/index.html](http://www.cs.nott.ac.uk/~jqd/mkp/index.html).

#### References
1. **Chu, P. C. and Beasley, J. E. (1998)**. A genetic algorithm for the multidimensional knapsack problem. *Journal of Heuristics, 4*(1):63–86.
2. **Drake, J. H., Ozcan, E. and Burke, E. K. (2016)**. A Case Study of Controlling Crossover in a Selection Hyper-heuristic Framework using the Multidimensional Knapsack Problem. *Evolutionary Computation, 24*(1):113–141.
3. **Glover, F. and Kochenberger, G. (n.d.)**. Benchmarks for “the multiple knapsack problem”. [http://hces.bus.olemiss.edu/tools.html](http://hces.bus.olemiss.edu/tools.html).
