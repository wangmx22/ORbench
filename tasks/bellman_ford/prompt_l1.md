# Task: Bellman-Ford Single Source Shortest Path (GPU Acceleration)

## Objective

Given the CPU reference implementation below, write a CUDA program that computes the same result but runs faster on a GPU.

## Algorithm Description

Bellman-Ford computes the shortest distances from a single source node to all other nodes in a weighted directed graph. The algorithm iteratively relaxes all edges until convergence.

## CPU Reference Implementation

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define INF_VAL 1e30f

struct CSRGraph {
    int num_nodes, num_edges;
    int* row_offsets;
    int* col_indices;
    float* weights;
};

void bellman_ford_cpu(const CSRGraph* g, int source, float* dist) {
    for (int i = 0; i < g->num_nodes; i++) dist[i] = INF_VAL;
    dist[source] = 0.0f;

    for (int round = 0; round < g->num_nodes - 1; round++) {
        int updated = 0;
        for (int u = 0; u < g->num_nodes; u++) {
            if (dist[u] >= INF_VAL) continue;
            for (int idx = g->row_offsets[u]; idx < g->row_offsets[u + 1]; idx++) {
                int v = g->col_indices[idx];
                float nd = dist[u] + g->weights[idx];
                if (nd < dist[v]) {
                    dist[v] = nd;
                    updated = 1;
                }
            }
        }
        if (!updated) break;
    }
}
```

## GPU Implementation Hints

Consider the following optimization strategies:

1. **Parallelization**: Assign one thread per node (or per edge) to process relaxations in parallel. Use `atomicMin` with a CAS loop for float values to handle concurrent updates.

2. **Early termination**: Use a device-side `updated` flag (cleared via `cudaMemset` each round, set by any thread that performs a relaxation) to detect convergence and break early.

3. **Persistent kernel** (advanced): Instead of launching a new kernel each iteration, run a single long-running kernel with all iterations inside a loop. Use `cooperative_groups::grid_group::sync()` or atomic barriers for global synchronization between iterations. This eliminates the ~20μs CPU-GPU round-trip overhead per iteration.

4. **Edge-parallel vs Node-parallel**: Edge-parallel assigns one thread per edge (better load balance for skewed degree distributions), while node-parallel assigns one thread per node (less atomic contention).

## Input/Output Specification

- **Graph**: CSR format with `row_offsets[V+1]`, `col_indices[E]`, `weights[E]`
- **Source**: node 0
- **Output**: `dist[V]` array; unreachable nodes use `1e30f`
- **Sizes**: V=1000/E=5000, V=100000/E=500000, V=500000/E=2500000

## Requirements

1. Complete `.cu` file, compilable with: `nvcc -O2 -arch=sm_89 -o solution solution.cu`
2. Print `GPU_TIME_MS: <value>` and `RESULT: <dist values>`
3. Maximize speedup over the CPU reference.

## Output Format

```
GPU_TIME_MS: <time_in_milliseconds>
RESULT: <dist[0]> <dist[1]> ... <dist[V-1]>
```
