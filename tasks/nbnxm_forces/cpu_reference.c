// cpu_reference.c — GROMACS-style cluster-based non-bonded force kernel (compute_only)
//
// Faithfully implements the GROMACS nbnxm reference kernel (4x4 cluster pairs).
// Reference: gromacs/nbnxm/kernels_reference/kernel_ref_outer.h + kernel_ref_inner.h
//
// Cluster pair format:
//   ci_entry[i] = (ci, cj_start, cj_end)  — i-cluster pointing to j-cluster range
//   cj_entry[j] = (cj, excl)              — j-cluster index + 16-bit exclusion mask
//
// For each ci_entry, iterate over cj_entry[cj_start..cj_end):
//   For each (i_atom, j_atom) pair in the 4x4 cluster pair:
//     if excl bit set AND within cutoff: compute LJ + Coulomb force
//
// Build: gcc -O2 -DORBENCH_COMPUTE_ONLY -I framework/
//        framework/harness_cpu.c tasks/nbnxm_forces/task_io_cpu.c
//        tasks/nbnxm_forces/cpu_reference.c -o solution_cpu -lm

#include <stdlib.h>
#include <string.h>
#include <math.h>

#define CLUSTER_SIZE 4
#define ONE_4PI_EPS0 138.935456f
#define MIN_RSQ 1e-12f

void solution_compute(
    int N,                        // number of atoms (padded to multiple of CLUSTER_SIZE)
    int num_ci,                   // number of i-cluster entries
    int num_cj,                   // total number of j-cluster entries
    int num_types,                // number of atom types
    float rcut2,                  // cutoff distance squared
    const float* x,               // [N*3] atom coordinates (cluster-contiguous)
    const float* q,               // [N] atom charges
    const int*   type,            // [N] atom type indices
    const float* nbfp,            // [num_types*num_types*2] LJ params (c6, c12 interleaved)
    const int*   ci_idx,          // [num_ci] i-cluster indices
    const int*   ci_cj_start,     // [num_ci] start index into cj arrays
    const int*   ci_cj_end,       // [num_ci] end index into cj arrays
    const int*   cj_idx,          // [num_cj] j-cluster indices
    const unsigned int* cj_excl,  // [num_cj] exclusion masks (16 bits for 4x4)
    float* f,                     // [N*3] output: accumulated forces
    float* energy_out             // [2] output: [E_lj, E_coul]
) {
    int ntype2 = num_types * 2;

    memset(f, 0, N * 3 * sizeof(float));
    double Vvdw = 0.0, Vc = 0.0;

    for (int ci_i = 0; ci_i < num_ci; ci_i++) {
        int ci = ci_idx[ci_i];
        int cjind0 = ci_cj_start[ci_i];
        int cjind1 = ci_cj_end[ci_i];

        // Load i-cluster coordinates and charges into local buffer
        float xi[CLUSTER_SIZE * 3];
        float qi[CLUSTER_SIZE];
        float fi_buf[CLUSTER_SIZE * 3];
        for (int i = 0; i < CLUSTER_SIZE; i++) {
            int ai = ci * CLUSTER_SIZE + i;
            xi[i*3+0] = x[ai*3+0];
            xi[i*3+1] = x[ai*3+1];
            xi[i*3+2] = x[ai*3+2];
            qi[i] = ONE_4PI_EPS0 * q[ai];
            fi_buf[i*3+0] = 0.0f;
            fi_buf[i*3+1] = 0.0f;
            fi_buf[i*3+2] = 0.0f;
        }

        // Loop over j-clusters
        for (int cjind = cjind0; cjind < cjind1; cjind++) {
            int cj = cj_idx[cjind];
            unsigned int excl = cj_excl[cjind];

            // 4x4 inner loop (matches GROMACS kernel_ref_inner.h)
            for (int i = 0; i < CLUSTER_SIZE; i++) {
                int ai = ci * CLUSTER_SIZE + i;
                int type_i_off = type[ai] * ntype2;

                for (int j = 0; j < CLUSTER_SIZE; j++) {
                    // Check exclusion mask
                    float interact = (float)((excl >> (i * CLUSTER_SIZE + j)) & 1);
                    float skipmask = interact;

                    int aj = cj * CLUSTER_SIZE + j;

                    // Distance calculation
                    float dx = xi[i*3+0] - x[aj*3+0];
                    float dy = xi[i*3+1] - x[aj*3+1];
                    float dz = xi[i*3+2] - x[aj*3+2];
                    float rsq = dx*dx + dy*dy + dz*dz;

                    // Cutoff check
                    skipmask = (rsq >= rcut2) ? 0.0f : skipmask;
                    rsq = (rsq < MIN_RSQ) ? MIN_RSQ : rsq;

                    float rinv = 1.0f / sqrtf(rsq);
                    rinv *= skipmask;
                    float rinvsq = rinv * rinv;

                    // LJ force
                    float c6  = nbfp[type_i_off + type[aj] * 2];
                    float c12 = nbfp[type_i_off + type[aj] * 2 + 1];

                    float rinvsix = interact * rinvsq * rinvsq * rinvsq;
                    float FrLJ6  = c6 * rinvsix;
                    float FrLJ12 = c12 * rinvsix * rinvsix;
                    float frLJ   = FrLJ12 - FrLJ6;

                    // Coulomb force
                    float qq = skipmask * qi[i] * q[aj];
                    float fcoul = qq * rinvsq * rinv;

                    // Total scalar force
                    float fscal = frLJ * rinvsq + fcoul;

                    float fx = fscal * dx;
                    float fy = fscal * dy;
                    float fz = fscal * dz;

                    // Newton's 3rd law
                    fi_buf[i*3+0] += fx;
                    fi_buf[i*3+1] += fy;
                    fi_buf[i*3+2] += fz;
                    f[aj*3+0] -= fx;
                    f[aj*3+1] -= fy;
                    f[aj*3+2] -= fz;

                    // Energies
                    float VLJ = FrLJ12 / 12.0f - FrLJ6 / 6.0f;
                    VLJ *= skipmask;
                    Vvdw += (double)VLJ;
                    Vc   += (double)(qq * rinv);
                }
            }
        }

        // Flush i-cluster force buffer to global force array
        for (int i = 0; i < CLUSTER_SIZE; i++) {
            int ai = ci * CLUSTER_SIZE + i;
            f[ai*3+0] += fi_buf[i*3+0];
            f[ai*3+1] += fi_buf[i*3+1];
            f[ai*3+2] += fi_buf[i*3+2];
        }
    }

    energy_out[0] = (float)Vvdw;
    energy_out[1] = (float)Vc;
}

void solution_free(void) {}
