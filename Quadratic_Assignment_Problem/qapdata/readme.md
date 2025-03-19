The data can be found [here](https://coral.ise.lehigh.edu/data-sets/qaplib/qaplib-problem-instances-and-solutions/)


## QAPLIB - Problem Instances and Solutions

### Problem Instances
The problem instances are provided in `.dat` files and are listed alphabetically by their authors. All instances are pure quadratic, and unless specified otherwise, they are symmetric. 

#### Format of Problem Data
The format of the `.dat` files is:
1. $ n $: Size of the instance (number of facilities/locations).
2. $ A $: Flow matrix.
3. $ B $: Distance matrix.

The objective is to solve the Quadratic Assignment Problem (QAP):
$$
\min_{p} \sum_{i=1}^n \sum_{j=1}^n a_{ij} b_{p(i)p(j)}
$$
where $ p $ is a permutation.

#### Example Instance Format
n
A
B


Each instance includes the problem size, best-known feasible solution, and the method used to derive it. If optimal, the permutation achieving the solution is provided. Non-optimal solutions include lower bounds and gaps.

---

### Heuristic Methods
The following heuristics are used to solve the QAP:
- **Ant Systems**: (ANT)
- **Genetic Hybrids**: (GEN), (GEN-2), (GEN-3), (GEN-4)
- **GRASP**: Greedy Randomized Adaptive Search Procedure
- **Scatter Search**: (ScS)
- **Simulated Annealing**: (SIM-1), (SIM-2), (SIM-3)
- **Simulated Jumping**: (SIMJ)
- **Tabu Search**: Parallel Adaptive Tabu Search (PA-TS), Reactive Tabu Search (Re-TS), Robust Tabu Search (Ro-TS), and Strict Tabu Search (S-TS)

---

### Data and Solutions
- **Compressed Data**: [qapdata.tar.gz (453,187 KB)](http://qaplib.zib.de)
- **Solutions**: [qapsoln.tar.gz (9,836 KB)](http://qaplib.zib.de)

### Lower Bounds
Lower bounds are computed using various methods, such as:
- **Gilmore-Lawler Bound (GLB)**
- **Elimination Bound (ELI)**
- **Interior Point LP Bound (IPLP)**
- **Semidefinite Programming Bound (SDP)**
- **Cutting Plane Bound (CUT)**

The gap between the best solution and the lower bound is expressed as:
$$
\text{gap} = \frac{\text{solution} - \text{bound}}{\text{solution}} \times 100\%
$$

---

### Notable Problem Instances
#### Burkard Instances
- **Bur26a**: $ n = 26 $, Feasible Solution = 5426670 (OPT)

#### Christofides Instances
- **Chr12a**: $ n = 12 $, Feasible Solution = 9552 (OPT)

#### Nugent Instances
- **Nug30**: $ n = 30 $, Feasible Solution = 6124 (OPT)

#### Taillard Instances
- **Tai256c**: $ n = 256 $, Feasible Solution = 44,759,294 (ANT)

---

### Contributors
The data and solutions were compiled by:
- **Peter Hahn** (hahn@seas.upenn.edu)
- **Miguel Anjos** (miguel-f.anjos@polymtl.ca)

For more details, visit the [QAPLIB Homepage](http://qaplib.zib.de).



Processing file: ../qapdata/nug16b.dat
Number of variables:  256
Processing file: ../qapdata/tai35b.dat
Number of variables:  1225
Processing file: ../qapdata/chr22a.dat
Number of variables:  484
Processing file: ../qapdata/esc16h.dat
Number of variables:  256
Processing file: ../qapdata/lipa40a.dat
Number of variables:  1600
Processing file: ../qapdata/chr18a.dat
Number of variables:  324
Processing file: ../qapdata/lipa60b.dat
Number of variables:  3600
Processing file: ../qapdata/esc16i.dat
Number of variables:  256
Processing file: ../qapdata/tai15a.dat
Number of variables:  225
Processing file: ../qapdata/nug16a.dat
Number of variables:  256
Processing file: ../qapdata/tai35a.dat
Number of variables:  1225
Processing file: ../qapdata/chr22b.dat
Number of variables:  484
Processing file: ../qapdata/lipa40b.dat
Number of variables:  1600
Processing file: ../qapdata/chr18b.dat
Number of variables:  324
Processing file: ../qapdata/tho30.dat
Number of variables:  900
Processing file: ../qapdata/lipa60a.dat
Number of variables:  3600
Processing file: ../qapdata/esc16j.dat
Number of variables:  256
Processing file: ../qapdata/tai15b.dat
Number of variables:  225
Processing file: ../qapdata/rou20.dat
Number of variables:  400
Processing file: ../qapdata/lipa20a.dat
Number of variables:  400
Processing file: ../qapdata/chr15a.dat
Number of variables:  225
Processing file: ../qapdata/tai80b.dat
Number of variables:  6400
Processing file: ../qapdata/lipa20b.dat
Number of variables:  400
Processing file: ../qapdata/chr15b.dat
Number of variables:  225
Processing file: ../qapdata/tai150b.dat
Number of variables:  22500
Processing file: ../qapdata/chr15c.dat
Number of variables:  225
Processing file: ../qapdata/tai80a.dat
Number of variables:  6400
Processing file: ../qapdata/lipa50b.dat
Number of variables:  2500
Processing file: ../qapdata/tai100a.dat
Number of variables:  10000
Processing file: ../qapdata/esc32a.dat
Number of variables:  1024
Processing file: ../qapdata/tai25a.dat
Number of variables:  625
Processing file: ../qapdata/chr12a.dat
Number of variables:  144
Processing file: ../qapdata/kra30a.dat
Number of variables:  900
Processing file: ../qapdata/nug12.dat
Number of variables:  144
Processing file: ../qapdata/had20.dat
Number of variables:  400
Processing file: ../qapdata/lipa70a.dat
Number of variables:  4900
Processing file: ../qapdata/lipa50a.dat
Number of variables:  2500
Processing file: ../qapdata/tai100b.dat
Number of variables:  10000
Processing file: ../qapdata/esc32b.dat
Number of variables:  1024
Processing file: ../qapdata/chr12c.dat
Number of variables:  144
Processing file: ../qapdata/tai25b.dat
Number of variables:  625
Processing file: ../qapdata/sko56.dat
Number of variables:  3136
Processing file: ../qapdata/sko42.dat
Number of variables:  1764
Processing file: ../qapdata/kra30b.dat
Number of variables:  900
Processing file: ../qapdata/chr12b.dat
Number of variables:  144
Processing file: ../qapdata/esc32c.dat
Number of variables:  1024
Processing file: ../qapdata/scr20.dat
Number of variables:  400
Processing file: ../qapdata/sko81.dat
Number of variables:  6561
Processing file: ../qapdata/lipa70b.dat
Number of variables:  4900
Processing file: ../qapdata/tai64c.dat
Number of variables:  4096
Processing file: ../qapdata/tai12a.dat
Number of variables:  144
Processing file: ../qapdata/nug15.dat
Number of variables:  225
Processing file: ../qapdata/esc32g.dat
Number of variables:  1024
Processing file: ../qapdata/lipa30b.dat
Number of variables:  900
Processing file: ../qapdata/sko90.dat
Number of variables:  8100
Processing file: ../qapdata/nug14.dat
Number of variables:  196
Processing file: ../qapdata/nug28.dat
Number of variables:  784
Processing file: ../qapdata/tho150.dat
Number of variables:  22500
Processing file: ../qapdata/chr25a.dat
Number of variables:  625
Processing file: ../qapdata/tai12b.dat
Number of variables:  144
Processing file: ../qapdata/had18.dat
Number of variables:  324
Processing file: ../qapdata/tho40.dat
Number of variables:  1600
Processing file: ../qapdata/lipa30a.dat
Number of variables:  900
Processing file: ../qapdata/esc32d.dat
Number of variables:  1024
Processing file: ../qapdata/esc32e.dat
Number of variables:  1024
Processing file: ../qapdata/nug17.dat
Number of variables:  289
Processing file: ../qapdata/esc64a.dat