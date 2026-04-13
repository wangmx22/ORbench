// cpu_reference.c -- SPH particle position update (CPU baseline)
//
// Faithfully ported from DualSPHysics JSphGpu_ker.cu
//   KerUpdatePos / KerComputeStepPos
//
// Simplified: no periodic boundaries, no floating bodies.
// For each particle: new_pos = old_pos + displacement, recompute cell index.
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.
//
// Reference: Crespo et al. "DualSPHysics: Open-source parallel CFD solver
// based on SPH", Computer Physics Communications, 2015.

#include <math.h>
#include <stdlib.h>

// ===== Module-level static state =====
static int    g_N;
static const float* g_pos_x;
static const float* g_pos_y;
static const float* g_pos_z;
static const float* g_mov_x;
static const float* g_mov_y;
static const float* g_mov_z;
static float g_cell_size;

// ===== Ported from DualSPHysics KerUpdatePos (simplified, non-periodic) =====
// Original: template<bool periactive> __device__ void KerUpdatePos(...)
// Applies displacement to particle position and computes new cell index.

static void KerUpdatePos(double px, double py, double pz,
                         double movx, double movy, double movz,
                         float cell_size,
                         double* out_x, double* out_y, double* out_z,
                         int* out_cell_x, int* out_cell_y, int* out_cell_z)
{
    /* Apply displacement (matches: rpos.x+=movx; rpos.y+=movy; rpos.z+=movz;) */
    double rx = px + movx;
    double ry = py + movy;
    double rz = pz + movz;

    *out_x = rx;
    *out_y = ry;
    *out_z = rz;

    /* Compute cell indices (matches dcell computation in DualSPHysics) */
    *out_cell_x = (int)floor(rx / cell_size);
    *out_cell_y = (int)floor(ry / cell_size);
    *out_cell_z = (int)floor(rz / cell_size);
}

// ===== Public interface =====

void solution_init(int N,
                   const float* posxy_x, const float* posxy_y,
                   const float* posz,
                   const float* movxy_x, const float* movxy_y,
                   const float* movz,
                   float cell_size)
{
    g_N = N;
    g_pos_x = posxy_x;
    g_pos_y = posxy_y;
    g_pos_z = posz;
    g_mov_x = movxy_x;
    g_mov_y = movxy_y;
    g_mov_z = movz;
    g_cell_size = cell_size;
}

void solution_compute(int N,
                      double* out_x, double* out_y, double* out_z,
                      int* out_cell)
{
    int i;
    for (i = 0; i < N; i++) {
        int cx, cy, cz;
        KerUpdatePos(g_pos_x[i], g_pos_y[i], g_pos_z[i],
                     g_mov_x[i], g_mov_y[i], g_mov_z[i],
                     g_cell_size,
                     &out_x[i], &out_y[i], &out_z[i],
                     &cx, &cy, &cz);
        /* Encode cell as linear index: cx + cy*1000 + cz*1000000 */
        out_cell[i] = cx + cy * 1000 + cz * 1000000;
    }
}

void solution_free(void)
{
    /* All data owned by task_io; nothing to free here. */
}
