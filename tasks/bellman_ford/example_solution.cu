// solution.cu - LLM 只需要实现这四个函数
// 框架编译: nvcc -O2 -arch=sm_89 harness.cu solution.cu -o solution

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cuda_runtime.h>

#define INF_VAL 1e30f
#define BLOCK_SIZE 256

// ============================================================
// Global state (persists across gpu_run calls)
// ============================================================
static int V, E, source;
static int *d_row_offsets, *d_col_indices, *d_updated;
static float *d_weights, *d_dist;
static float *dist_init;  // for resetting

// ============================================================
// Helper: read binary files
// ============================================================
static int* read_int_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    int* data = (int*)malloc(count * sizeof(int));
    fread(data, sizeof(int), count, f);
    fclose(f);
    return data;
}

static float* read_float_bin(const char* path, int count) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "Cannot open %s\n", path); exit(1); }
    float* data = (float*)malloc(count * sizeof(float));
    fread(data, sizeof(float), count, f);
    fclose(f);
    return data;
}

// ============================================================
// GPU kernel
// ============================================================
__device__ __forceinline__
float atomicMinFloat(float* addr, float value) {
    int* p = (int*)addr;
    int old = *p, assumed;
    do {
        assumed = old;
        if (__int_as_float(assumed) <= value) return __int_as_float(old);
        old = atomicCAS(p, assumed, __float_as_int(value));
    } while (assumed != old);
    return __int_as_float(old);
}

__global__ void bf_kernel(
    int num_nodes, const int* __restrict__ row_offsets,
    const int* __restrict__ col_indices, const float* __restrict__ weights,
    float* dist, int* updated)
{
    int u = blockIdx.x * blockDim.x + threadIdx.x;
    if (u >= num_nodes) return;
    float d = dist[u];
    if (d >= INF_VAL) return;
    for (int i = row_offsets[u]; i < row_offsets[u + 1]; i++) {
        float nd = d + weights[i];
        if (nd < atomicMinFloat(&dist[col_indices[i]], nd))
            *updated = 1;
    }
}

// ============================================================
// Interface implementation
// ============================================================

int gpu_setup(const char* data_dir) {
    char path[512];

    // Read meta
    int seed;
    snprintf(path, sizeof(path), "%s/input.txt", data_dir);
    FILE* f = fopen(path, "r");
    fscanf(f, "%d %d %d %d", &V, &E, &source, &seed);
    fclose(f);
    fprintf(stderr, "Graph: V=%d, E=%d, source=%d\n", V, E, source);

    // Read graph
    snprintf(path, sizeof(path), "%s/row_offsets.bin", data_dir);
    int* h_row = read_int_bin(path, V + 1);
    snprintf(path, sizeof(path), "%s/col_indices.bin", data_dir);
    int* h_col = read_int_bin(path, E);
    snprintf(path, sizeof(path), "%s/weights.bin", data_dir);
    float* h_w = read_float_bin(path, E);

    // Alloc GPU
    cudaMalloc(&d_row_offsets, (V + 1) * sizeof(int));
    cudaMalloc(&d_col_indices, E * sizeof(int));
    cudaMalloc(&d_weights, E * sizeof(float));
    cudaMalloc(&d_dist, V * sizeof(float));
    cudaMalloc(&d_updated, sizeof(int));

    // Upload graph (one time)
    cudaMemcpy(d_row_offsets, h_row, (V + 1) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_col_indices, h_col, E * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_weights, h_w, E * sizeof(float), cudaMemcpyHostToDevice);

    // Prepare initial distance array for resetting
    dist_init = (float*)malloc(V * sizeof(float));
    for (int i = 0; i < V; i++) dist_init[i] = INF_VAL;
    dist_init[source] = 0.0f;

    free(h_row); free(h_col); free(h_w);

    return V;  // number of result floats
}

void gpu_run() {
    // Reset distances
    cudaMemcpy(d_dist, dist_init, V * sizeof(float), cudaMemcpyHostToDevice);

    int grid = (V + BLOCK_SIZE - 1) / BLOCK_SIZE;
    int h_updated;

    for (int r = 0; r < V - 1; r++) {
        cudaMemset(d_updated, 0, sizeof(int));
        bf_kernel<<<grid, BLOCK_SIZE>>>(V, d_row_offsets, d_col_indices, d_weights, d_dist, d_updated);
        cudaMemcpy(&h_updated, d_updated, sizeof(int), cudaMemcpyDeviceToHost);
        if (!h_updated) break;
    }
}

void gpu_get_results(float* output, int count) {
    cudaMemcpy(output, d_dist, count * sizeof(float), cudaMemcpyDeviceToHost);
}

void gpu_cleanup() {
    cudaFree(d_row_offsets);
    cudaFree(d_col_indices);
    cudaFree(d_weights);
    cudaFree(d_dist);
    cudaFree(d_updated);
    free(dist_init);
}
