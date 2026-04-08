# Batched Floyd-Warshall All-Pairs Shortest Paths

## Problem background

Given a weighted directed graph with nonnegative edge weights, the all-pairs shortest path (APSP) problem asks for the shortest-path distance between every ordered pair of vertices. Floyd-Warshall is one of the most classical dynamic-programming algorithms for APSP: it progressively allows larger sets of intermediate vertices and updates a dense distance matrix.

This ORBench task uses a **batched** dense-graph setting. Each input contains `B` graphs with `N` vertices each, represented as flattened `N x N` adjacency matrices. The output is the exact shortest-path distance matrix for every graph.

## Algorithm source

- Floyd, R. W. (1962), *Algorithm 97: Shortest Path*
- Warshall, S. (1962), *A Theorem on Boolean Matrices*
- Standard textbook APSP / dynamic-programming algorithm

## Why it is suitable for GPU acceleration

Floyd-Warshall has a strict dependency across the outer `k` loop, but within each `k` iteration all `(i, j)` updates are independent:

`dist[i,j] = min(dist[i,j], dist[i,k] + dist[k,j])`

That gives a natural SIMT parallelization strategy per phase. Batched graphs add further coarse-grained parallelism across independent graph instances. Efficient GPU implementations can exploit tiling, shared memory, and phase-synchronous kernels.

## Input format

Stored in `input.bin` as:

- tensor `adj` (`int32`, length `B * N * N`): flattened row-major adjacency matrices, graph by graph
- param `B`: number of graphs
- param `N`: number of vertices per graph
- param `INF`: sentinel value for absent edges

Each graph uses row-major indexing. `adj[g*N*N + i*N + j]` is the edge weight from `i` to `j`. Diagonal entries are zero.

## Output format

- flattened APSP distance matrices in the same layout as the input adjacency matrices
- one integer per line in `output.txt` / `expected_output.txt`

## Notes

- All weights are nonnegative integers
- Generated graphs are strongly connected
- The task outputs **exact** integer shortest-path distances
