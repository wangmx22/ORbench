# Task: Bellman-Ford Single Source Shortest Path (GPU Acceleration)

## Objective

Given the CPU reference implementation below, write a CUDA program that computes the same result but runs faster on a GPU.

## Algorithm Description

Bellman-Ford computes the shortest distances from a single source node to all other nodes in a weighted directed graph. The graph may have negative edge weights (but no negative cycles). The algorithm iteratively relaxes all edges until convergence: for each edge (u, v, w), if `dist[u] + w < dist[v]`, update `dist[v]`. The algorithm converges in at most `V-1` iterations, but can terminate early if no distances change in an iteration.

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
    int* row_offsets;    // size: num_nodes + 1
    int* col_indices;    // size: num_edges
    float* weights;      // size: num_edges
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

## Input/Output Specification

- **Graph format**: CSR (Compressed Sparse Row)
  - `row_offsets[V+1]`: edge start indices for each node
  - `col_indices[E]`: destination node for each edge
  - `weights[E]`: edge weights (positive floats, range [1.0, 100.0])
- **Source node**: 0
- **Output**: `dist[V]` array, `dist[i]` = shortest distance from source to node i
  - Unreachable nodes: `dist[i] = 1e30f`
- **Input sizes**: The program will be tested with V=1000/E=5000, V=100000/E=500000, V=500000/E=2500000

## Requirements

Implement a `.cu` file with the following four functions (do NOT include `main()`):

```c
// Called once. Read graph from data_dir/*.bin, allocate GPU memory.
// Return the number of result floats (= number of nodes V).
int gpu_setup(const char* data_dir);

// Run the GPU Bellman-Ford computation. Must be re-entrant:
// reset distances each call (called multiple times for warmup + timing).
void gpu_run();

// Copy the distance results from GPU to the output buffer.
void gpu_get_results(float* output, int count);

// Free GPU memory.
void gpu_cleanup();
```

### Data files in `data_dir/`:
- `input.txt`: first line is `V E source seed`
- `row_offsets.bin`: `(V+1)` int32 values
- `col_indices.bin`: `E` int32 values
- `weights.bin`: `E` float32 values

### Compilation
The framework compiles your code with a timing harness:
```
nvcc -O2 -arch=sm_89 harness.cu solution.cu -o solution
```
The harness provides `main()`, handles warmup (3 runs), timing (10 CUDA Event trials), and output.

### Goal
Optimize `gpu_run()` for maximum speedup over the CPU reference.
