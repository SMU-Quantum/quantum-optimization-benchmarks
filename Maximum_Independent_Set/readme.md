Dataset taken from [Challenge Problems: Independent Sets in Graphs](https://oeis.org/A265032/a265032.html)


# Graphs From Single-Deletion-Correcting Codes

## Files and Descriptions

- **1dc.64.txt.gz**: A 64-node graph in which the size of a maximal independent set is 10.  
  Example: The Varshamov-Tenegolts code VT0(6) of length 6 forms the independent set.

- **1dc.128.txt.gz**: A 128-node graph where the size of a maximal independent set is 16.  
  Example: The Varshamov-Tenegolts code VT0(7) of length 7 forms an independent set of size 16.  
  Magma confirms that no larger independent set exists.

- **1dc.256.txt.gz**: A 256-node graph with a maximal independent set of size 30.  
  Example: The Varshamov-Tenegolts code VT0(8) of length 8 forms an independent set.  
  David Applegate confirmed this size using Cliquer.

- **1dc.512.txt.gz**: A 512-node graph with a maximal independent set of size 52.  
  Example: The Varshamov-Tenegolts code VT0(9) forms an independent set.  
  Confirmed by various researchers including Ketan Narendra Patel.

- **1dc.1024.txt.gz**: A 1024-node graph where the maximal independent set size is 94.  
  Example: The Varshamov-Tenegolts code VT0(10) forms an independent set.  
  Brian Borchers proved this result using branch-and-bound techniques.

- **1dc.2048.txt.gz**: A 2048-node graph where the maximal independent set size is not known.  
  Example: The Varshamov-Tenegolts code VT0(11) forms an independent set of size 172.  
  The upper bound is 174, with the solution in the range 172–174.


## Information about the problem datasets

1. Graphs From Single-Deletion-Correcting Codes

    `1dc.64.txt.gz`, a 64-node graph in which the size of a maximal independent set is 10 (for example, the Varshamov-Tenegolts code VT0(6) of length 6). 

2. Graphs From Codes For Correcting a Single Transposition (Excluding the End-Around Transposition)

    These graphs all decompose into k+1 connected components, where k = log_2 of the number of nodes.
    That is, the subgraphs defided by the binary vectors of length n and weight w may be treated individually.

    `1tc.8.txt.gz`, an 8-node graph in which it is known that the size of a maximal independent set is 4. 

    `1tc.16.txt.gz`, a 16-node graph in which it is known that the size of a maximal independent set is 8.

    `1tc.32.txt.gz`, a 32-node graph in which it is known that the size of a maximal independent set is 12.

    `1tc.64.txt.gz`, a 64-node graph in which it is known that the size of a maximal independent set is 20.
    For example, nodes { 1, 5, 7, 8, 15, 21, 29, 31, 33, 34, 36, 40, 49, 50, 54, 56, 57, 61, 63, 64 }. 


3. Graphs From Codes For Correcting a Single Transposition (Including the End-Around Transposition)

    These graphs all decompose into k+1 connected components, where k = log_2 of the number of nodes.

    That is, the subgraphs defided by the binary vectors of length n and weight w may be treated individually.

    `1et.64.txt.gz`, a 64-node graph in which it is known that the size of a maximal independent 

## Notes
- The sizes of the maximal independent sets align with sequence A000016 in the OEIS for graphs with ≤1024 nodes.
- The agreement with the sequence for graphs ≥2048 nodes is an open problem.
