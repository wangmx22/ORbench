// cpu_reference.c — Pairwise squared Euclidean distance matrix (CPU baseline)
//
// Faithfully ported from kNN-CUDA (vincentfpgarcia/kNN-CUDA, code/knncuda.cu)
// — specifically the `compute_distances` global-memory kernel that computes
// the SSD (sum of squared differences) matrix between two point sets.
//
// Layout matches the original kNN-CUDA code:
//   * `ref`   stored column-major: ref[k * ref_pitch + r]
//             where k is the dim-row index and r is the point column index
//   * `query` stored column-major: query[k * query_pitch + q]
//   * `dist`  stored row-by-r:    dist[r * query_pitch + q]
//             i.e. one row per ref point, one column per query point.
//   * pitches equal nb (no padding) since input sizes are powers of 2.
//
// Reference: Garcia, Debreuve, Barlaud, "Fast k Nearest Neighbor Search using
// GPU", CVPRW 2008; Garcia et al., "k-NN search: fast GPU-based
// implementations", ICIP 2010.
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <stdlib.h>

// ===== Module-level state =====
static int          g_ref_nb;
static int          g_query_nb;
static int          g_dim;
static const float* g_ref;     // column-major [dim, ref_nb]
static const float* g_query;   // column-major [dim, query_nb]

// ===== Function ported from knncuda.cu::compute_distances =====
// Each "thread" (r, q) computes ssd over the dim axis, exactly like the
// CUDA kernel which assigns one thread per output cell.
// Output: dist[r * query_pitch + q] (matches kernel's write at line 89).
static void compute_distancesCpu(int          ref_nb,
                                 int          ref_pitch,
                                 int          query_nb,
                                 int          query_pitch,
                                 int          height,            // == dim
                                 const float* ref,
                                 const float* query,
                                 float*       dist)
{
    int r, q, k;
    for (r = 0; r < ref_nb; r++) {
        for (q = 0; q < query_nb; q++) {
            float ssd = 0.0f;
            for (k = 0; k < height; k++) {
                float a = ref  [k * ref_pitch   + r];
                float b = query[k * query_pitch + q];
                float tmp = a - b;
                ssd += tmp * tmp;
            }
            dist[r * query_pitch + q] = ssd;
        }
    }
}

// ===== Public interface =====

void solution_init(int          ref_nb,
                   int          query_nb,
                   int          dim,
                   const float* ref,
                   const float* query)
{
    g_ref_nb   = ref_nb;
    g_query_nb = query_nb;
    g_dim      = dim;
    g_ref      = ref;
    g_query    = query;
}

void solution_compute(int    ref_nb,
                      int    query_nb,
                      int    dim,
                      float* dist)
{
    // pitches == nb (no padding for power-of-two sizes)
    compute_distancesCpu(ref_nb, ref_nb, query_nb, query_nb, dim,
                         g_ref, g_query, dist);
}

void solution_free(void)
{
    /* All input data owned by task_io; nothing to free here. */
}
