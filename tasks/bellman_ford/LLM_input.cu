// LLM_input.cu - GPU Bellman-Ford Template
//
// This file has IDENTICAL I/O and timing structure to cpu_reference.cu.
// The ONLY difference: the CPU algorithm is replaced with GPU code.
//
// Compile: nvcc -O2 -arch=sm_89 -o solution LLM_input.cu
// Run:     ./solution <data_dir>              (timing only, fast)
//          ./solution <data_dir> --validate   (timing + write output.bin)
//
// Example: ./solution tasks/bellman_ford/data/large --validate

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <cuda_runtime.h>

#define INF_VAL 1e30f

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Graph structure (same as cpu_reference.cu)
// ============================================================
struct CSRGraph {
    int num_nodes, num_edges;
    int* row_offsets;    // size: num_nodes + 1
    int* col_indices;    // size: num_edges
    float* weights;      // size: num_edges
};

// ============================================================
// Read binary files (same as cpu_reference.cu)
// ============================================================
int* read_int_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int* data = (int*)malloc(count * sizeof(int));
    size_t nread = fread(data, sizeof(int), count, f);
    if ((int)nread != count) {
        fprintf(stderr, "Expected %d ints from %s, got %zu\n", count, path, nread);
        exit(1);
    }
    fclose(f);
    return data;
}

float* read_float_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    size_t nread = fread(data, sizeof(float), count, f);
    if ((int)nread != count) {
        fprintf(stderr, "Expected %d floats from %s, got %zu\n", count, path, nread);
        exit(1);
    }
    fclose(f);
    return data;
}

// ============================================================
// CPU reference algorithm (for understanding, DO NOT modify)
// Your GPU implementation should produce the same results.
// ============================================================
//
// void bellman_ford_cpu(const CSRGraph* g, int source, float* dist) {
//     for (int i = 0; i < g->num_nodes; i++) dist[i] = INF_VAL;
//     dist[source] = 0.0f;
//
//     for (int round = 0; round < g->num_nodes - 1; round++) {
//         int updated = 0;
//         for (int u = 0; u < g->num_nodes; u++) {
//             if (dist[u] >= INF_VAL) continue;
//             for (int idx = g->row_offsets[u]; idx < g->row_offsets[u + 1]; idx++) {
//                 int v = g->col_indices[idx];
//                 float nd = dist[u] + g->weights[idx];
//                 if (nd < dist[v]) {
//                     dist[v] = nd;
//                     updated = 1;
//                 }
//             }
//         }
//         if (!updated) break;
//     }
// }

// ╔══════════════════════════════════════════════════════════════╗
// ║  LLM TODO: Implement your CUDA kernels and device functions ║
// ║  below. You may add as many kernels/helpers as needed.      ║
// ╚══════════════════════════════════════════════════════════════╝

// >>> LLM CODE START <<<




// >>> LLM CODE END <<<


// ╔══════════════════════════════════════════════════════════════╗
// ║  LLM TODO: Implement gpu_bellman_ford()                     ║
// ║                                                              ║
// ║  Input:                                                      ║
// ║    - g: graph in CSR format (already on HOST memory)         ║
// ║    - source: source node index                               ║
// ║    - dist: pre-allocated HOST array of size g->num_nodes     ║
// ║                                                              ║
// ║  Output:                                                     ║
// ║    - Write shortest distances into dist[]                    ║
// ║    - dist[i] = INF_VAL (1e30f) if node i is unreachable     ║
// ║                                                              ║
// ║  You must handle:                                            ║
// ║    1. cudaMalloc / cudaMemcpy (host → device)                ║
// ║    2. Kernel launch(es)                                      ║
// ║    3. cudaMemcpy (device → host) to fill dist[]              ║
// ║    4. cudaFree                                               ║
// ║                                                              ║
// ║  This function is called INSIDE the timing region.           ║
// ║  Minimize unnecessary work here for best performance.        ║
// ╚══════════════════════════════════════════════════════════════╝

void gpu_bellman_ford(const CSRGraph* g, int source, float* dist) {
    // >>> LLM CODE START <<<



    // >>> LLM CODE END <<<
}


// ============================================================
// Main (DO NOT MODIFY below this line)
// Identical structure to cpu_reference.cu:
//   1. Read input from .bin files
//   2. Time the algorithm (CUDA Events)
//   3. Optionally write output.bin for correctness checking
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [--validate]\n", argv[0]);
        fprintf(stderr, "  e.g.: %s tasks/bellman_ford/data/large\n", argv[0]);
        fprintf(stderr, "  --validate: write result to output.bin for correctness check\n");
        return 1;
    }

    const char* data_dir = argv[1];
    int do_validate = 0;
    if (argc >= 3 && strcmp(argv[2], "--validate") == 0) {
        do_validate = 1;
    }

    char path[512];

    // ---- Read input (same as cpu_reference.cu) ----
    int V, E, source, seed;
    snprintf(path, sizeof(path), "%s/input.txt", data_dir);
    FILE* meta = fopen(path, "r");
    if (!meta) { fprintf(stderr, "Cannot open %s\n", path); return 1; }
    fscanf(meta, "%d %d %d %d", &V, &E, &source, &seed);
    fclose(meta);

    fprintf(stderr, "Graph: V=%d, E=%d, source=%d\n", V, E, source);

    snprintf(path, sizeof(path), "%s/row_offsets.bin", data_dir);
    int* row_offsets = read_int_bin(path, V + 1);

    snprintf(path, sizeof(path), "%s/col_indices.bin", data_dir);
    int* col_indices = read_int_bin(path, E);

    snprintf(path, sizeof(path), "%s/weights.bin", data_dir);
    float* weights = read_float_bin(path, E);

    CSRGraph g;
    g.num_nodes = V;
    g.num_edges = E;
    g.row_offsets = row_offsets;
    g.col_indices = col_indices;
    g.weights = weights;

    float* dist = (float*)malloc(V * sizeof(float));

    // ---- Warmup (3 runs, not timed) ----
    for (int w = 0; w < 3; w++) {
        gpu_bellman_ford(&g, source, dist);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---- Timed runs (CUDA Events, 10 trials) ----
    int num_trials = 10;
    float total_ms = 0;
    float min_ms = 1e9, max_ms = 0;

    for (int t = 0; t < num_trials; t++) {
        cudaEvent_t start, stop;
        CHECK_CUDA(cudaEventCreate(&start));
        CHECK_CUDA(cudaEventCreate(&stop));

        CHECK_CUDA(cudaEventRecord(start));
        gpu_bellman_ford(&g, source, dist);
        CHECK_CUDA(cudaEventRecord(stop));
        CHECK_CUDA(cudaEventSynchronize(stop));

        float ms;
        CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
        total_ms += ms;
        if (ms < min_ms) min_ms = ms;
        if (ms > max_ms) max_ms = ms;

        CHECK_CUDA(cudaEventDestroy(start));
        CHECK_CUDA(cudaEventDestroy(stop));
    }

    float mean_ms = total_ms / num_trials;

    // ---- Print timing (framework parses this line) ----
    printf("GPU_TIME_MS: %.3f\n", mean_ms);
    fprintf(stderr, "Timing: mean=%.3f ms, min=%.3f ms, max=%.3f ms (%d trials)\n",
            mean_ms, min_ms, max_ms, num_trials);

    // ---- Validate: write output.bin (same as cpu_reference.cu) ----
    if (do_validate) {
        // Run one more time to get final results
        gpu_bellman_ford(&g, source, dist);
        CHECK_CUDA(cudaDeviceSynchronize());

        snprintf(path, sizeof(path), "%s/output.bin", data_dir);
        FILE* fout = fopen(path, "wb");
        fwrite(dist, sizeof(float), V, fout);
        fclose(fout);
        fprintf(stderr, "Results written to %s/output.bin\n", data_dir);
    }

    // ---- Cleanup ----
    free(dist);
    free(row_offsets);
    free(col_indices);
    free(weights);
    return 0;
}
