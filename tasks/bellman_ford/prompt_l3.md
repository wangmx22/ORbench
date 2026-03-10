# Task: Accelerate the following CPU code using CUDA

Write a GPU-accelerated version of this program. Output a complete `.cu` file compilable with `nvcc -O2 -arch=sm_89 -o solution solution.cu`. Print `GPU_TIME_MS: <value>` for timing and `RESULT: <values>` for output.

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

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

Test with V=1000/E=5000, V=100000/E=500000, V=500000/E=2500000. Source=0, seed=42.
