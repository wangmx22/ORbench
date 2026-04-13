// cpu_reference.c — All-pairs directed Hausdorff distance (CPU baseline)
//
// Faithfully ported from cuSpatial (rapidsai/cuspatial,
// cpp/include/cuspatial/detail/distance/hausdorff.cuh::kernel_hausdorff
// and cpp/src/distance/hausdorff.cu::directed_hausdorff_distance).
//
// Computes the directed Hausdorff distance for every ordered pair of point
// "spaces" (e.g. trajectories or polyline samples). For each LHS point in
// space i and each RHS space j, find the minimum Euclidean distance from the
// LHS point to any RHS point in space j; the directed Hausdorff distance
// h(i -> j) is the maximum of these minimums over all LHS points in space i.
//
// Output layout matches cuSpatial: results[j * num_spaces + i] holds
// h(i -> j) (note the transposed indexing — this is how cuSpatial atomicMax
// writes its results).
//
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>

// 2D point type — matches cuspatial::vec_2d<float>
typedef struct {
    float x;
    float y;
} vec_2d;

static inline float magnitude_squared(float a, float b)
{
    return a * a + b * b;
}

// ===== Module-level state =====
static int            g_num_points;
static int            g_num_spaces;
static const vec_2d*  g_points;
static const int*     g_space_offsets;

// ===== Function ported from cuspatial::detail::kernel_hausdorff =====
// Direct CPU port (no atomics needed since each (i, j) pair is touched
// independently). For each LHS point, loop over all RHS spaces; for each RHS
// space find the min squared distance, then take max over all LHS points in
// the same LHS space.
static void kernel_hausdorffCpu(int             num_points,
                                const vec_2d*   points,
                                int             num_spaces,
                                const int*      space_offsets,
                                float*          results)
{
    int i, j;

    // Initialize results to -1 (matches cuSpatial atomicMax sentinel).
    for (i = 0; i < num_spaces * num_spaces; i++) {
        results[i] = -1.0f;
    }

    // Walk every LHS point.
    for (int lhs_p_idx = 0; lhs_p_idx < num_points; lhs_p_idx++) {

        // Determine the LHS space this point belongs to via binary search
        // (mirrors cuspatial's `thrust::upper_bound(...) - 1`).
        int lo = 0;
        int hi = num_spaces;
        while (lo < hi) {
            int mid = (lo + hi) >> 1;
            if (space_offsets[mid] <= lhs_p_idx) lo = mid + 1;
            else                                  hi = mid;
        }
        int lhs_space_idx = lo - 1;

        vec_2d lhs_p = points[lhs_p_idx];

        // Loop over each RHS space.
        for (int rhs_space_idx = 0; rhs_space_idx < num_spaces; rhs_space_idx++) {
            int rhs_p_idx_begin = space_offsets[rhs_space_idx];
            int rhs_p_idx_end   = (rhs_space_idx + 1 == num_spaces)
                                    ? num_points
                                    : space_offsets[rhs_space_idx + 1];

            float min_distance_squared = INFINITY;

            for (j = rhs_p_idx_begin; j < rhs_p_idx_end; j++) {
                vec_2d rhs_p = points[j];
                float dsq = magnitude_squared(rhs_p.x - lhs_p.x,
                                              rhs_p.y - lhs_p.y);
                if (dsq < min_distance_squared) min_distance_squared = dsq;
            }

            int output_idx = rhs_space_idx * num_spaces + lhs_space_idx;
            float candidate = sqrtf(min_distance_squared);
            if (candidate > results[output_idx]) {
                results[output_idx] = candidate;
            }
        }
    }
}

// ===== Public interface =====

void solution_init(int           num_points,
                   int           num_spaces,
                   const float*  points_xy,      // [num_points * 2], x0,y0,x1,y1,...
                   const int*    space_offsets)
{
    g_num_points    = num_points;
    g_num_spaces    = num_spaces;
    g_points        = (const vec_2d*)points_xy;  // interleaved x,y matches vec_2d
    g_space_offsets = space_offsets;
}

void solution_compute(int    num_points,
                      int    num_spaces,
                      float* results)              // [num_spaces * num_spaces]
{
    kernel_hausdorffCpu(num_points, g_points, num_spaces,
                        g_space_offsets, results);
}

void solution_free(void)
{
    /* All input data owned by task_io; nothing to free here. */
}
