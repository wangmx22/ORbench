// code.cu - Bellman-Ford GPU solution (reads binary input from data/ directory)
//
// Compile: nvcc -O2 -arch=sm_89 -o code code.cu
// Run:     ./code data/large
//          ./code data/medium
//          ./code data/small

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <cuda_runtime.h>

#define INF_VAL 1e30f
#define BLOCK_SIZE 256

#define CHECK_CUDA(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", \
                __FILE__, __LINE__, cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================
// Read binary files
// ============================================================
int* read_int_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int* data = (int*)malloc(count * sizeof(int));
    size_t read = fread(data, sizeof(int), count, f);
    if ((int)read != count) {
        fprintf(stderr, "Expected %d ints from %s, got %zu\n", count, path, read);
        exit(1);
    }
    fclose(f);
    return data;
}

float* read_float_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    size_t read = fread(data, sizeof(float), count, f);
    if ((int)read != count) {
        fprintf(stderr, "Expected %d floats from %s, got %zu\n", count, path, read);
        exit(1);
    }
    fclose(f);
    return data;
}

void read_meta(const char* data_dir, int* V, int* E, int* source, int* seed) {
    char path[512];
    snprintf(path, sizeof(path), "%s/input.txt", data_dir);
    FILE* f = fopen(path, "r");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    fscanf(f, "%d %d %d %d", V, E, source, seed);
    fclose(f);
}

// ============================================================
// GPU atomicMin for float
// ============================================================
__device__ __forceinline__
float atomicMinFloat(float* addr, float value) {
    int* addr_as_int = (int*)addr;
    int old = *addr_as_int;
    int assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) return __int_as_float(old);
        old = atomicCAS(addr_as_int, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

// ============================================================
// GPU Kernel: per-node parallel Bellman-Ford
// ============================================================
__global__
void bf_kernel(
    int num_nodes,
    const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices,
    const float* __restrict__ weights,
    float* dist,
    int* updated)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;

    float dist_u = dist[u];
    if (dist_u >= INF_VAL) return;

    int start = row_offsets[u];
    int end   = row_offsets[u + 1];
    for (int idx = start; idx < end; idx++) {
        int v = col_indices[idx];
        float nd = dist_u + weights[idx];
        float old = atomicMinFloat(&dist[v], nd);
        if (nd < old) {
            *updated = 1;
        }
    }
}

// ============================================================
// Main
// ============================================================
int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <data_dir> [--validate]\n", argv[0]);
        fprintf(stderr, "  e.g.: %s data/large\n", argv[0]);
        fprintf(stderr, "  --validate: write result to output.bin for correctness check\n");
        return 1;
    }

    const char* data_dir = argv[1];
    int do_validate = 0;
    if (argc >= 3 && strcmp(argv[2], "--validate") == 0) {
        do_validate = 1;
    }

    char path[512];

    // ---- Read input ----
    int V, E, source, seed;
    read_meta(data_dir, &V, &E, &source, &seed);
    fprintf(stderr, "Graph: V=%d, E=%d, source=%d\n", V, E, source);

    snprintf(path, sizeof(path), "%s/row_offsets.bin", data_dir);
    int* row_offsets = read_int_bin(path, V + 1);

    snprintf(path, sizeof(path), "%s/col_indices.bin", data_dir);
    int* col_indices = read_int_bin(path, E);

    snprintf(path, sizeof(path), "%s/weights.bin", data_dir);
    float* weights = read_float_bin(path, E);

    // ---- Allocate GPU memory ----
    int *d_row_offsets, *d_col_indices, *d_updated;
    float *d_weights, *d_dist;

    CHECK_CUDA(cudaMalloc(&d_row_offsets, (V + 1) * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_col_indices, E * sizeof(int)));
    CHECK_CUDA(cudaMalloc(&d_weights, E * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_dist, V * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_updated, sizeof(int)));

    // ---- Upload graph to GPU ----
    CHECK_CUDA(cudaMemcpy(d_row_offsets, row_offsets, (V + 1) * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_col_indices, col_indices, E * sizeof(int), cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_weights, weights, E * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Init distance array ----
    float* dist_init = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) dist_init[i] = INF_VAL;
    dist_init[source] = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_dist, dist_init, V * sizeof(float), cudaMemcpyHostToDevice));
    free(dist_init);

    // ---- Warmup ----
    int grid = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    for (int w = 0; w < 3; w++) {
        CHECK_CUDA(cudaMemset(d_updated, 0, sizeof(int)));
        bf_kernel<<<grid, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices, d_weights, d_dist, d_updated);
        CHECK_CUDA(cudaDeviceSynchronize());
    }

    // ---- Re-init distance for actual run ----
    float* dist_reset = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) dist_reset[i] = INF_VAL;
    dist_reset[source] = 0.0f;
    CHECK_CUDA(cudaMemcpy(d_dist, dist_reset, V * sizeof(float), cudaMemcpyHostToDevice));
    free(dist_reset);

    // ---- Timed run ----
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));

    int h_updated;
    int rounds = 0;
    for (int r = 0; r < V - 1; r++) {
        CHECK_CUDA(cudaMemset(d_updated, 0, sizeof(int)));
        bf_kernel<<<grid, BLOCK_SIZE>>>(
            V, d_row_offsets, d_col_indices, d_weights, d_dist, d_updated);
        CHECK_CUDA(cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost));
        rounds++;
        if (!h_updated) break;
    }

    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));

    float ms = 0;
    CHECK_CUDA(cudaEventElapsedTime(&ms, start, stop));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));

    fprintf(stderr, "Converged in %d rounds, GPU time: %.3f ms\n", rounds, ms);

    // ---- Copy results back ----
    float* dist_gpu = (float*)malloc(V * sizeof(float));
    CHECK_CUDA(cudaMemcpy(dist_gpu, d_dist, V * sizeof(float), cudaMemcpyDeviceToHost));

    // ---- Output ----
    printf("GPU_TIME_MS: %.3f\n", ms);

    if (do_validate) {
        snprintf(path, sizeof(path), "%s/output.bin", data_dir);
        FILE* fout = fopen(path, "wb");
        fwrite(dist_gpu, sizeof(float), V, fout);
        fclose(fout);
        fprintf(stderr, "Results written to %s/output.bin\n", data_dir);
    }

    // ---- Cleanup ----
    CHECK_CUDA(cudaFree(d_row_offsets));
    CHECK_CUDA(cudaFree(d_col_indices));
    CHECK_CUDA(cudaFree(d_weights));
    CHECK_CUDA(cudaFree(d_dist));
    CHECK_CUDA(cudaFree(d_updated));
    free(row_offsets);
    free(col_indices);
    free(weights);
    free(dist_gpu);

    return 0;
}