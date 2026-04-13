// gpu_baseline.cu — Production-quality GROMACS-style CUDA non-bonded force kernel
//
// Key optimizations over the previous simplified version:
//   1. Warp-shuffle reductions for j-forces, i-forces, and energies (zero barrier)
//   2. float4 xq packing (x,y,z,q in one load)
//   3. float2 nbfp packing (c6,c12 in one load)
//   4. Proper exclusion bit handling without branching
//   5. __launch_bounds__ for optimal register allocation
//
// Uses the same flat ci/cj interface as the existing task_io and gen_data.
//
// Reference: gromacs/nbnxm/cuda/nbnxm_cuda_kernel.cuh
//            gromacs/nbnxm/cuda/nbnxm_cuda_kernel_utils.cuh

#include <cuda_runtime.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

#define CLUSTER_SIZE 4
#define THREADS_PER_BLOCK (CLUSTER_SIZE * CLUSTER_SIZE)  // 16
#define MIN_BLOCKS_PER_MP 16
#define ONE_4PI_EPS0 138.935456f
#define c_nbnxnMinDistanceSquared 3.82e-07f
#define c_oneSixth    (1.0f / 6.0f)
#define c_oneTwelfth  (1.0f / 12.0f)
#define c_fullWarpMask 0xFFFFFFFFu

// ─── Warp-shuffle j-force reduction ───
// Reduces (fx, fy, fz) across tidxi (4 threads) using interleaved shuffle.
// After reduction, tidxi=0 holds sum(fx), tidxi=1 holds sum(fy), tidxi=2 holds sum(fz).
static __forceinline__ __device__ void
reduce_force_j_warp_shfl(float fx, float fy, float fz,
                         float* __restrict__ fout, int tidxi, int aj)
{
    // Step 1: stride-1 exchange
    fx += __shfl_down_sync(c_fullWarpMask, fx, 1);
    fy += __shfl_up_sync(c_fullWarpMask, fy, 1);
    fz += __shfl_down_sync(c_fullWarpMask, fz, 1);
    if (tidxi & 1) fx = fy;

    // Step 2: stride-2 exchange
    fx += __shfl_down_sync(c_fullWarpMask, fx, 2);
    fz += __shfl_up_sync(c_fullWarpMask, fz, 2);
    if (tidxi & 2) fx = fz;

    // Step 3: stride-4 to combine across tidxj groups (within warp)
    fx += __shfl_down_sync(c_fullWarpMask, fx, 4);

    // tidxi 0,1,2 write x,y,z respectively
    if (tidxi < 3) {
        atomicAdd(&fout[aj * 3 + tidxi], fx);
    }
}

// ─── Warp-shuffle i-force reduction ───
// Reduces across tidxj (stride = CLUSTER_SIZE=4) within a warp
static __forceinline__ __device__ void
reduce_force_i_warp_shfl(float fx, float fy, float fz,
                         float* __restrict__ fout, int tidxj, int ai)
{
    fx += __shfl_down_sync(c_fullWarpMask, fx, CLUSTER_SIZE);
    fy += __shfl_up_sync(c_fullWarpMask, fy, CLUSTER_SIZE);
    fz += __shfl_down_sync(c_fullWarpMask, fz, CLUSTER_SIZE);
    if (tidxj & 1) fx = fy;

    fx += __shfl_down_sync(c_fullWarpMask, fx, 2 * CLUSTER_SIZE);
    fz += __shfl_up_sync(c_fullWarpMask, fz, 2 * CLUSTER_SIZE);
    if (tidxj & 2) fx = fz;

    if ((tidxj & 3) < 3) {
        atomicAdd(&fout[ai * 3 + (tidxj & 3)], fx);
    }
}

// ─── Warp-shuffle energy reduction ───
static __forceinline__ __device__ void
reduce_energy_warp_shfl(float E_lj, float E_el,
                        float* __restrict__ e_lj_global,
                        float* __restrict__ e_el_global, int tidx)
{
    int sh = 1;
    #pragma unroll 5
    for (int i = 0; i < 5; i++) {
        E_lj += __shfl_down_sync(c_fullWarpMask, E_lj, sh);
        E_el += __shfl_down_sync(c_fullWarpMask, E_el, sh);
        sh += sh;
    }
    if (tidx == 0) {
        atomicAdd(e_lj_global, E_lj);
        atomicAdd(e_el_global, E_el);
    }
}

// ═══════════════════════════════════════════════════════════
//  KERNEL: one block per ci entry, 4×4 threads, warp-shuffle reductions
// ═══════════════════════════════════════════════════════════
__launch_bounds__(THREADS_PER_BLOCK, MIN_BLOCKS_PER_MP)
__global__ void nbnxm_kernel_VF(
    int num_ci,
    const float4* __restrict__ xq,          // packed (x,y,z, q*epsfac) per atom
    const int*    __restrict__ atom_types,
    const float2* __restrict__ nbfp,        // packed (c6, c12) indexed by type_i*ntypes+type_j
    int num_types,
    const int*    __restrict__ ci_idx,
    const int*    __restrict__ ci_cj_start,
    const int*    __restrict__ ci_cj_end,
    const int*    __restrict__ cj_idx,
    const unsigned int* __restrict__ cj_excl,
    float* __restrict__ f,
    float* __restrict__ e_lj_global,
    float* __restrict__ e_coul_global,
    float rcut2
) {
    int bid = blockIdx.x;
    if (bid >= num_ci) return;

    unsigned int tidxi = threadIdx.x;  // i-atom index (0-3)
    unsigned int tidxj = threadIdx.y;  // j-atom index (0-3)
    unsigned int tidx  = threadIdx.y * CLUSTER_SIZE + threadIdx.x;

    int ci = ci_idx[bid];
    int cjind0 = ci_cj_start[bid];
    int cjind1 = ci_cj_end[bid];

    // Load i-cluster data into shared memory using float4
    __shared__ float4 s_xqi[CLUSTER_SIZE];
    __shared__ int    s_typei[CLUSTER_SIZE];

    if (tidx < CLUSTER_SIZE) {
        int ai = ci * CLUSTER_SIZE + tidx;
        s_xqi[tidx] = xq[ai];
        s_typei[tidx] = atom_types[ai];
    }
    __syncthreads();

    // i-atom data for this thread
    float4 xqi = s_xqi[tidxi];
    float  qi_val = xqi.w;
    int    type_i = s_typei[tidxi];

    // Local force accumulator for i-atom
    float fi_x = 0.0f, fi_y = 0.0f, fi_z = 0.0f;
    float E_lj = 0.0f, E_coul = 0.0f;

    // Loop over j-clusters
    for (int cjind = cjind0; cjind < cjind1; cjind++) {
        int cj = cj_idx[cjind];
        unsigned int excl = cj_excl[cjind];

        // Exclusion bit for this (tidxi, tidxj) pair
        float int_bit = (float)((excl >> (tidxi * CLUSTER_SIZE + tidxj)) & 1u);

        int aj = cj * CLUSTER_SIZE + tidxj;

        // Load j-atom data using float4
        float4 xqj = xq[aj];
        int    tj   = atom_types[aj];

        // Distance
        float dx = xqi.x - xqj.x;
        float dy = xqi.y - xqj.y;
        float dz = xqi.z - xqj.z;
        float rsq = dx * dx + dy * dy + dz * dz;

        // Combined cutoff + exclusion check (branchless multiply by int_bit)
        float within = (rsq < rcut2 && rsq >= c_nbnxnMinDistanceSquared) ? 1.0f : 0.0f;
        within *= int_bit;

        if (within > 0.0f) {
            float rinv = rsqrtf(rsq);
            float rinvsq = rinv * rinv;

            // LJ parameters via float2 load
            float2 c6c12 = nbfp[type_i * num_types + tj];
            float c6  = c6c12.x;
            float c12 = c6c12.y;

            float inv_r6 = rinvsq * rinvsq * rinvsq;
            inv_r6 *= int_bit;

            // LJ force
            float F_invr = inv_r6 * (c12 * inv_r6 - c6) * rinvsq;

            // Coulomb force (simple cutoff)
            float qq = qi_val * xqj.w;
            F_invr += qq * int_bit * rinvsq * rinv;

            float fx = F_invr * dx;
            float fy = F_invr * dy;
            float fz = F_invr * dz;

            // Accumulate i-force locally
            fi_x += fx;
            fi_y += fy;
            fi_z += fz;

            // j-force: reduce via warp shuffle instead of atomicAdd per pair
            reduce_force_j_warp_shfl(-fx, -fy, -fz, f, tidxi, aj);

            // Energies
            E_lj   += int_bit * (c12 * inv_r6 * inv_r6 * c_oneTwelfth
                                - c6 * inv_r6 * c_oneSixth);
            E_coul += qq * int_bit * rinv;
        }
    }

    // Reduce i-forces across tidxj using warp shuffle
    reduce_force_i_warp_shfl(fi_x, fi_y, fi_z, f, tidxj,
                             ci * CLUSTER_SIZE + tidxi);

    // Reduce energies using warp shuffle
    reduce_energy_warp_shfl(E_lj, E_coul, e_lj_global, e_coul_global, tidx);
}


// ═══════════════════════════════════════════════════════════
//  HOST INTERFACE — compatible with existing task_io.cu
// ═══════════════════════════════════════════════════════════

static float4*       d_xq = nullptr;
static int*          d_atom_types = nullptr;
static float2*       d_nbfp = nullptr;
static int*          d_ci_idx = nullptr;
static int*          d_ci_cj_start = nullptr;
static int*          d_ci_cj_end = nullptr;
static int*          d_cj_idx = nullptr;
static unsigned int* d_cj_excl = nullptr;
static float*        d_f = nullptr;
static float*        d_e_lj = nullptr;
static float*        d_e_coul = nullptr;

extern "C" void solution_compute(
    int N, int num_ci, int num_cj, int num_types, float rcut2,
    const float* x, const float* q, const int* type, const float* nbfp,
    const int* ci_idx, const int* ci_cj_start, const int* ci_cj_end,
    const int* cj_idx, const unsigned int* cj_excl,
    float* f, float* energy_out
) {
    int ntype2 = num_types * 2;

    // Pack x,y,z,q into float4 on host
    float4* h_xq = (float4*)malloc(N * sizeof(float4));
    for (int i = 0; i < N; i++) {
        h_xq[i].x = x[i * 3 + 0];
        h_xq[i].y = x[i * 3 + 1];
        h_xq[i].z = x[i * 3 + 2];
        h_xq[i].w = ONE_4PI_EPS0 * q[i];
    }

    // Pack nbfp (c6,c12 interleaved) into float2
    int nbfp_count = num_types * num_types;
    float2* h_nbfp = (float2*)malloc(nbfp_count * sizeof(float2));
    for (int i = 0; i < num_types; i++) {
        for (int j = 0; j < num_types; j++) {
            h_nbfp[i * num_types + j].x = nbfp[i * ntype2 + j * 2];      // c6
            h_nbfp[i * num_types + j].y = nbfp[i * ntype2 + j * 2 + 1];   // c12
        }
    }

    // Allocate device memory
    cudaMalloc(&d_xq, N * sizeof(float4));
    cudaMalloc(&d_atom_types, N * sizeof(int));
    cudaMalloc(&d_nbfp, nbfp_count * sizeof(float2));
    cudaMalloc(&d_ci_idx, num_ci * sizeof(int));
    cudaMalloc(&d_ci_cj_start, num_ci * sizeof(int));
    cudaMalloc(&d_ci_cj_end, num_ci * sizeof(int));
    cudaMalloc(&d_cj_idx, num_cj * sizeof(int));
    cudaMalloc(&d_cj_excl, num_cj * sizeof(unsigned int));
    cudaMalloc(&d_f, N * 3 * sizeof(float));
    cudaMalloc(&d_e_lj, sizeof(float));
    cudaMalloc(&d_e_coul, sizeof(float));

    // Upload
    cudaMemcpy(d_xq, h_xq, N * sizeof(float4), cudaMemcpyHostToDevice);
    cudaMemcpy(d_atom_types, type, N * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_nbfp, h_nbfp, nbfp_count * sizeof(float2), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci_idx, ci_idx, num_ci * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci_cj_start, ci_cj_start, num_ci * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_ci_cj_end, ci_cj_end, num_ci * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cj_idx, cj_idx, num_cj * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cj_excl, cj_excl, num_cj * sizeof(unsigned int), cudaMemcpyHostToDevice);

    // Zero outputs
    cudaMemset(d_f, 0, N * 3 * sizeof(float));
    cudaMemset(d_e_lj, 0, sizeof(float));
    cudaMemset(d_e_coul, 0, sizeof(float));

    // Launch
    dim3 block(CLUSTER_SIZE, CLUSTER_SIZE);
    dim3 grid(num_ci);

    nbnxm_kernel_VF<<<grid, block>>>(
        num_ci, d_xq, d_atom_types, d_nbfp, num_types,
        d_ci_idx, d_ci_cj_start, d_ci_cj_end,
        d_cj_idx, d_cj_excl,
        d_f, d_e_lj, d_e_coul, rcut2
    );

    // Download results
    cudaMemcpy(f, d_f, N * 3 * sizeof(float), cudaMemcpyDeviceToHost);
    float e_lj_h, e_coul_h;
    cudaMemcpy(&e_lj_h, d_e_lj, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&e_coul_h, d_e_coul, sizeof(float), cudaMemcpyDeviceToHost);
    energy_out[0] = e_lj_h;
    energy_out[1] = e_coul_h;

    free(h_xq);
    free(h_nbfp);
}

extern "C" void solution_free(void) {
    if (d_xq)           { cudaFree(d_xq);           d_xq = nullptr; }
    if (d_atom_types)    { cudaFree(d_atom_types);    d_atom_types = nullptr; }
    if (d_nbfp)          { cudaFree(d_nbfp);          d_nbfp = nullptr; }
    if (d_ci_idx)        { cudaFree(d_ci_idx);        d_ci_idx = nullptr; }
    if (d_ci_cj_start)   { cudaFree(d_ci_cj_start);   d_ci_cj_start = nullptr; }
    if (d_ci_cj_end)     { cudaFree(d_ci_cj_end);     d_ci_cj_end = nullptr; }
    if (d_cj_idx)        { cudaFree(d_cj_idx);        d_cj_idx = nullptr; }
    if (d_cj_excl)       { cudaFree(d_cj_excl);       d_cj_excl = nullptr; }
    if (d_f)             { cudaFree(d_f);             d_f = nullptr; }
    if (d_e_lj)          { cudaFree(d_e_lj);          d_e_lj = nullptr; }
    if (d_e_coul)        { cudaFree(d_e_coul);        d_e_coul = nullptr; }
}
