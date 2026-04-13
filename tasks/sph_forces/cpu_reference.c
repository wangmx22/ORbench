// cpu_reference.c -- SPH force computation (CPU baseline)
//
// Faithfully ported from DualSPHysics JSphGpu_ker.cu
//   KerInteractionForcesFluidBox (simplified: Wendland kernel only,
//   no floating bodies, no periodic boundaries, no shifting)
//
// Computes: pressure force (momentum equation), density derivative
// (continuity equation), and artificial viscosity for each particle
// by iterating over neighbors within the smoothing radius.
//
// Uses Wendland C2 kernel and Tait equation of state (Monaghan 1994).
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.
//
// Reference: Crespo et al. "DualSPHysics: Open-source parallel CFD solver
// based on SPH", Computer Physics Communications, 2015.

#include <math.h>
#include <stdlib.h>

// ===== SPH Constants =====
#define SPH_GAMMA 7.0f
#define ALMOSTZERO 1e-18f

// ===== Module-level static state =====
static int    g_N;
static const float* g_xs;
static const float* g_ys;
static const float* g_zs;
static const float* g_vxs;
static const float* g_vys;
static const float* g_vzs;
static const float* g_rhos;
static const float* g_masses;
static float  g_h;          /* smoothing length */
static float  g_cs0;        /* speed of sound */
static float  g_rhop0;      /* reference density */
static float  g_alpha_visc; /* artificial viscosity coefficient */
static const int* g_cell_begin;
static const int* g_cell_end;
static const int* g_sorted_idx;
static int    g_grid_nx;
static int    g_grid_ny;
static int    g_grid_nz;
static float  g_cell_size;

/* Precomputed kernel constants */
static float  g_awen;       /* Wendland kernel normalization */
static float  g_bwenh;      /* Wendland gradient normalization / h */
static float  g_kernelsize2; /* (2h)^2 = kernel support squared */
static float  g_cteb;       /* Tait EOS: B = cs0^2 * rhop0 / gamma */
static float  g_ovrhopzero; /* 1.0 / rhop0 */

// ===== Ported from DualSPHysics FunSphKernel_iker.h =====
// Wendland C2 kernel gradient factor.
// Original: __device__ float GetKernelWendland_Fac(float rr2, float h, float bwenh)

static float GetKernelWendland_Fac(float rr2, float h, float bwenh)
{
    float rad = sqrtf(rr2);
    float qq = rad / h;
    float wqq1 = 1.0f - 0.5f * qq;
    return bwenh * wqq1 * wqq1 * wqq1;
}

// ===== Ported from DualSPHysics FunSphEos_iker.h =====
// Tait equation of state (Monaghan 1994).
// Original: __device__ float ComputePressMonaghan(float rho, float rho0, float b, float gamma)

static float ComputePressMonaghan(float rho, float ovrhopzero, float cteb)
{
    return cteb * (powf(rho * ovrhopzero, SPH_GAMMA) - 1.0f);
}

// ===== Ported from DualSPHysics KerInteractionForcesFluidBox (simplified) =====
// Original: template<...> __device__ void KerInteractionForcesFluidBox(...)
// Iterates over neighbor particles in one cell, accumulating forces.

static void InteractionForcesFluidBox(
    int p1, int pini, int pfin,
    float p1x, float p1y, float p1z,
    float p1vx, float p1vy, float p1vz,
    float p1rho, float pressp1, float massp1,
    float h, float bwenh, float kernelsize2,
    float cs0, float alpha_visc, float ovrhopzero, float cteb,
    float* acex, float* acey, float* acez, float* arp1)
{
    int p2;
    for (p2 = pini; p2 < pfin; p2++) {
        int idx2 = g_sorted_idx[p2];
        if (idx2 == p1) continue;

        float p2x = g_xs[idx2];
        float p2y = g_ys[idx2];
        float p2z = g_zs[idx2];

        /* Distance computation (matches DualSPHysics drx/dry/drz) */
        float drx = p1x - p2x;
        float dry = p1y - p2y;
        float drz = p1z - p2z;
        float rr2 = drx * drx + dry * dry + drz * drz;

        if (rr2 <= kernelsize2 && rr2 >= ALMOSTZERO) {
            /* Compute kernel gradient factor */
            float fac = GetKernelWendland_Fac(rr2, h, bwenh);
            float frx = fac * drx;
            float fry = fac * dry;
            float frz = fac * drz;

            float p2rho = g_rhos[idx2];
            float massp2 = g_masses[idx2];

            /* Pressure force (momentum equation) */
            /* Original: const float pressp2=cufsph::ComputePressCte(velrhop2.w); */
            float pressp2 = ComputePressMonaghan(p2rho, ovrhopzero, cteb);
            float prs = (pressp1 + pressp2) / (p1rho * p2rho);
            float p_vpm = -prs * massp2;
            *acex += p_vpm * frx;
            *acey += p_vpm * fry;
            *acez += p_vpm * frz;

            /* Artificial viscosity (Monaghan 1992) */
            float dvx = p1vx - g_vxs[idx2];
            float dvy = p1vy - g_vys[idx2];
            float dvz = p1vz - g_vzs[idx2];
            float dot_rv = drx * dvx + dry * dvy + drz * dvz;

            if (dot_rv < 0.0f) {
                float rhobar = 0.5f * (p1rho + p2rho);
                float mu = h * dot_rv / (rr2 + 0.01f * h * h);
                float pi_visc = -alpha_visc * cs0 * mu / rhobar;
                float visc_force = -pi_visc * massp2;
                *acex += visc_force * frx;
                *acey += visc_force * fry;
                *acez += visc_force * frz;
            }

            /* Density derivative (continuity equation) */
            /* Original: arp1+= massp2*(dvx*frx+dvy*fry+dvz*frz)*(velrhop1.w/velrhop2.w); */
            *arp1 += massp2 * (dvx * frx + dvy * fry + dvz * frz) * (p1rho / p2rho);
        }
    }
}

// ===== Public interface =====

void solution_init(int N,
                   const float* xs, const float* ys, const float* zs,
                   const float* vxs, const float* vys, const float* vzs,
                   const float* rhos, const float* masses,
                   float h, float cs0, float rhop0, float alpha_visc,
                   const int* cell_begin, const int* cell_end,
                   const int* sorted_idx,
                   int grid_nx, int grid_ny, int grid_nz,
                   float cell_size)
{
    g_N = N;
    g_xs = xs;
    g_ys = ys;
    g_zs = zs;
    g_vxs = vxs;
    g_vys = vys;
    g_vzs = vzs;
    g_rhos = rhos;
    g_masses = masses;
    g_h = h;
    g_cs0 = cs0;
    g_rhop0 = rhop0;
    g_alpha_visc = alpha_visc;
    g_cell_begin = cell_begin;
    g_cell_end = cell_end;
    g_sorted_idx = sorted_idx;
    g_grid_nx = grid_nx;
    g_grid_ny = grid_ny;
    g_grid_nz = grid_nz;
    g_cell_size = cell_size;

    /* Precompute Wendland kernel constants (3D) */
    /* awen = 495/(32*pi*h^3) for 3D Wendland C2 */
    g_awen = 495.0f / (32.0f * 3.14159265358979f * h * h * h);
    /* bwenh = -10 * awen / h   (gradient prefactor) */
    /* Actually: fac = bwenh * (1 - q/2)^3, and grad W = fac * r_vec */
    g_bwenh = -10.0f * g_awen / h;
    /* Kernel support = 2h, so kernelsize2 = (2h)^2 */
    g_kernelsize2 = 4.0f * h * h;
    /* Tait EOS constant: B = cs0^2 * rhop0 / gamma */
    g_cteb = cs0 * cs0 * rhop0 / SPH_GAMMA;
    g_ovrhopzero = 1.0f / rhop0;
}

void solution_compute(int N,
                      float* ax, float* ay, float* az, float* drhodt)
{
    int i;
    int num_cells_xy = g_grid_nx * g_grid_ny;

    for (i = 0; i < N; i++) {
        float p1x = g_xs[i];
        float p1y = g_ys[i];
        float p1z = g_zs[i];
        float p1vx = g_vxs[i];
        float p1vy = g_vys[i];
        float p1vz = g_vzs[i];
        float p1rho = g_rhos[i];

        float pressp1 = ComputePressMonaghan(p1rho, g_ovrhopzero, g_cteb);

        float acex = 0.0f, acey = 0.0f, acez = 0.0f;
        float arp1 = 0.0f;

        /* Determine cell of particle i */
        int cx = (int)floorf(p1x / g_cell_size);
        int cy = (int)floorf(p1y / g_cell_size);
        int cz = (int)floorf(p1z / g_cell_size);

        /* Iterate over 3x3x3 neighbor cells */
        int dcx, dcy, dcz;
        for (dcz = -1; dcz <= 1; dcz++) {
            int ncz = cz + dcz;
            if (ncz < 0 || ncz >= g_grid_nz) continue;
            for (dcy = -1; dcy <= 1; dcy++) {
                int ncy = cy + dcy;
                if (ncy < 0 || ncy >= g_grid_ny) continue;
                for (dcx = -1; dcx <= 1; dcx++) {
                    int ncx = cx + dcx;
                    if (ncx < 0 || ncx >= g_grid_nx) continue;

                    int cell_id = ncx + ncy * g_grid_nx + ncz * num_cells_xy;
                    int pini = g_cell_begin[cell_id];
                    int pfin = g_cell_end[cell_id];
                    if (pini < 0 || pfin < 0) continue;

                    InteractionForcesFluidBox(
                        i, pini, pfin,
                        p1x, p1y, p1z,
                        p1vx, p1vy, p1vz,
                        p1rho, pressp1, g_masses[i],
                        g_h, g_bwenh, g_kernelsize2,
                        g_cs0, g_alpha_visc, g_ovrhopzero, g_cteb,
                        &acex, &acey, &acez, &arp1);
                }
            }
        }

        ax[i] = acex;
        ay[i] = acey;
        az[i] = acez;
        drhodt[i] = arp1;
    }
}

void solution_free(void)
{
    /* All data owned by task_io; nothing to free here. */
}
