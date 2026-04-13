// gpu_baseline.cu -- SPH particle position update (GPU baseline)
//
// Faithfully ported from DualSPHysics JSphGpu_ker.cu
//   KerComputeStepPos / KerUpdatePos
//
// Simplified: no periodic boundaries, no floating bodies.
// One thread per particle, embarrassingly parallel.
// Persistent GPU memory (cudaMalloc in init, reuse in compute).

#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define BLOCK_SIZE 256

// ===== Device-side persistent arrays =====
static int    g_N;
static float  g_cell_size;
static float* d_pos_x;
static float* d_pos_y;
static float* d_pos_z;
static float* d_mov_x;
static float* d_mov_y;
static float* d_mov_z;
static double* d_out_x;
static double* d_out_y;
static double* d_out_z;
static int*    d_out_cell;

// ===== GPU kernel: KerComputeStepPos (simplified, non-periodic) =====
// Ported from DualSPHysics KerComputeStepPos / KerUpdatePos
// Each thread handles one particle: applies displacement and computes cell index.

__global__ void KerComputeStepPos(int N,
                                  const float* __restrict__ pos_x,
                                  const float* __restrict__ pos_y,
                                  const float* __restrict__ pos_z,
                                  const float* __restrict__ mov_x,
                                  const float* __restrict__ mov_y,
                                  const float* __restrict__ mov_z,
                                  float cell_size,
                                  double* __restrict__ out_x,
                                  double* __restrict__ out_y,
                                  double* __restrict__ out_z,
                                  int* __restrict__ out_cell)
{
    unsigned int pt = blockIdx.x * blockDim.x + threadIdx.x;
    if (pt < (unsigned int)N) {
        // Apply displacement (matches: rpos.x += movx; etc.)
        double rx = (double)pos_x[pt] + (double)mov_x[pt];
        double ry = (double)pos_y[pt] + (double)mov_y[pt];
        double rz = (double)pos_z[pt] + (double)mov_z[pt];

        out_x[pt] = rx;
        out_y[pt] = ry;
        out_z[pt] = rz;

        // Compute cell indices (matches dcell computation in DualSPHysics)
        int cx = (int)floor(rx / (double)cell_size);
        int cy = (int)floor(ry / (double)cell_size);
        int cz = (int)floor(rz / (double)cell_size);

        // Encode cell as linear index: cx + cy*1000 + cz*1000000
        out_cell[pt] = cx + cy * 1000 + cz * 1000000;
    }
}

// ===== Public interface (extern "C" for task_io linkage) =====

extern "C" {

void solution_init(int N,
                   const float* posxy_x, const float* posxy_y,
                   const float* posz,
                   const float* movxy_x, const float* movxy_y,
                   const float* movz,
                   float cell_size)
{
    g_N = N;
    g_cell_size = cell_size;

    size_t sz_f = (size_t)N * sizeof(float);
    size_t sz_d = (size_t)N * sizeof(double);
    size_t sz_i = (size_t)N * sizeof(int);

    // Allocate and copy input arrays to device
    cudaMalloc(&d_pos_x, sz_f);
    cudaMalloc(&d_pos_y, sz_f);
    cudaMalloc(&d_pos_z, sz_f);
    cudaMalloc(&d_mov_x, sz_f);
    cudaMalloc(&d_mov_y, sz_f);
    cudaMalloc(&d_mov_z, sz_f);

    cudaMemcpy(d_pos_x, posxy_x, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_y, posxy_y, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos_z, posz,    sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mov_x, movxy_x, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mov_y, movxy_y, sz_f, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mov_z, movz,    sz_f, cudaMemcpyHostToDevice);

    // Allocate output arrays on device
    cudaMalloc(&d_out_x, sz_d);
    cudaMalloc(&d_out_y, sz_d);
    cudaMalloc(&d_out_z, sz_d);
    cudaMalloc(&d_out_cell, sz_i);
}

void solution_compute(int N,
                      double* out_x, double* out_y, double* out_z,
                      int* out_cell)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    KerComputeStepPos<<<grid, block>>>(
        N,
        d_pos_x, d_pos_y, d_pos_z,
        d_mov_x, d_mov_y, d_mov_z,
        g_cell_size,
        d_out_x, d_out_y, d_out_z, d_out_cell);

    // Copy results back to host
    size_t sz_d = (size_t)N * sizeof(double);
    size_t sz_i = (size_t)N * sizeof(int);
    cudaMemcpy(out_x,    d_out_x,    sz_d, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_y,    d_out_y,    sz_d, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_z,    d_out_z,    sz_d, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_cell, d_out_cell, sz_i, cudaMemcpyDeviceToHost);
}

void solution_free(void)
{
    cudaFree(d_pos_x);
    cudaFree(d_pos_y);
    cudaFree(d_pos_z);
    cudaFree(d_mov_x);
    cudaFree(d_mov_y);
    cudaFree(d_mov_z);
    cudaFree(d_out_x);
    cudaFree(d_out_y);
    cudaFree(d_out_z);
    cudaFree(d_out_cell);
}

} // extern "C"
