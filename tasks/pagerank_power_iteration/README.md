# PageRank Power Iteration

## Background

PageRank is a classic graph ranking algorithm that assigns an importance score to each vertex in a directed graph. It models a random surfer who follows outgoing links with probability `d` and teleports uniformly to any vertex with probability `1-d`. The algorithm is widely used in graph analytics, search, recommendation, and influence estimation.

This task computes PageRank by fixed-count power iteration on a directed graph stored in CSR-like **inbound** form. Each iteration updates every vertex from the ranks of its inbound neighbors.

## Source

Brin, S. and Page, L. (1998). *The anatomy of a large-scale hypertextual web search engine.* This is the standard textbook / systems PageRank formulation.

## Why it fits GPU acceleration

Each iteration updates all vertices independently once the previous rank vector is fixed. The main work is a sparse gather over inbound edges:

\[
\text{rank}_{new}[v] = \frac{1-d}{N} + d \sum_{u \in In(v)} \frac{\text{rank}[u]}{\text{outdeg}[u]}
\]

This exposes substantial parallelism across vertices and edges. The main bottleneck is repeated sparse memory access over CSR edge lists, which is exactly the kind of workload GPUs can accelerate with many concurrent threads.

## Input

Parameters:

- `N` : number of vertices
- `M` : number of directed edges
- `iters` : number of power iterations
- `damping_x1e6` : damping factor multiplied by `1e6`

Tensors:

- `row_ptr_in` (`int32`, length `N+1`) : inbound CSR row pointer
- `col_ind_in` (`int32`, length `M`) : source vertex for each inbound edge
- `inv_out_deg` (`float32`, length `N`) : `1 / out_degree[u]` for each source vertex

## Output

- `out` (`float32`, length `N`) : final PageRank score for each vertex after exactly `iters` iterations

The output is written as one float per line in vertex order.
