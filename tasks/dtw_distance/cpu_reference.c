// cpu_reference.c — Dynamic Time Warping (CPU baseline)
//
// Faithfully ported from cuDTW++ (asbschmidt/cuDTW), Schmidt & Hundt
// "cuDTW++: Ultra-Fast Dynamic Time Warping on CUDA-enabled GPUs"
// (Euro-Par 2020). This file is a sequential simulator of the
//   src/include/kernels/SHFL_FULLDTW_1023.cuh::shfl_FullDTW_1023
// kernel: it preserves every register, every save/restore pattern, every
// shuffle, and the epilogue r=16 quirk (which uses penalty_diag instead of
// penalty_temp1). The simulator emulates a 32-lane warp by storing each
// lane's state in array slots and replays warp-shuffles as explicit array
// shifts. Result bit-matches the kernel's output.
//
// Length-specialized to num_features = 1023 (lane = 1024 = 32 lanes x 32 cells).
// NO file I/O, NO main(). All I/O handled by task_io_cpu.c.

#include <math.h>
#include <stdlib.h>

#define LANES 32
#define REGS  32
#define WARP_SIZE 32

static int          g_num_entries;
static int          g_num_features;
static const float* g_subjects;
static const float* g_query;

// ===== Sequential simulator of shfl_FullDTW_1023 =====
//
// All variable names mirror the CUDA kernel (penalty_here<r>, penalty_left,
// penalty_diag, query_value, new_query_value, subject_value<r>) but each
// becomes an array indexed by lane.
static float shfl_FullDTW_1023_cpu(const float* query, const float* subject)
{
    const int num_features = 1023;
    const int lane         = num_features + 1;  // 1024

    // ----- Per-lane state -----
    float penalty_here[LANES][REGS];
    float penalty_left[LANES];
    float penalty_diag[LANES];
    float subject_value[LANES][REGS];
    float query_value[LANES];
    float new_query_value[LANES];

    // ----- Initialization (lines 24-57 of cuDTW kernel) -----
    for (int l = 0; l < LANES; l++) {
        penalty_left[l] = INFINITY;
        penalty_diag[l] = 0;
        for (int r = 0; r < REGS; r++) penalty_here[l][r] = INFINITY;
    }
    // thid==0 override (lines 71-107)
    {
        int l = 0;
        penalty_left[l] = INFINITY;
        penalty_diag[l] = INFINITY;
        for (int r = 0; r < REGS; r++) penalty_here[l][r] = INFINITY;
    }

    // ----- Subject loads (lines 110-142) -----
    for (int l = 0; l < LANES; l++) {
        subject_value[l][0] = (l == 0) ? 0.0f : subject[32 * l - 1];
        for (int r = 1; r < REGS; r++) {
            subject_value[l][r] = subject[32 * l + r - 1];
        }
    }

    // ----- Pre-step query loads (lines 145-151) -----
    for (int l = 0; l < LANES; l++) {
        query_value[l]     = INFINITY;
        new_query_value[l] = query[l];     // cQuery[thid]
    }
    query_value[0] = new_query_value[0];   // if (thid == 0) query_value = new_query_value
    penalty_here[0][1] = 0;                // if (thid == 0) penalty_here1 = 0
    // shfl_down(new_query_value, 1)  -- emulate (lane 31 keeps its value)
    {
        float tmp[LANES];
        for (int l = 0; l < LANES - 1; l++) tmp[l] = new_query_value[l + 1];
        tmp[LANES - 1] = new_query_value[LANES - 1];
        for (int l = 0; l < LANES; l++) new_query_value[l] = tmp[l];
    }

    // ----- First DP step (lines 158-222) -----
    {
        float old[LANES][REGS];
        for (int l = 0; l < LANES; l++)
            for (int r = 0; r < REGS; r++)
                old[l][r] = penalty_here[l][r];

        for (int l = 0; l < LANES; l++) {
            float qv  = query_value[l];
            float pl_ = penalty_left[l];
            float pd_ = penalty_diag[l];
            float* sv = subject_value[l];

            // r=0: 3rd arg = penalty_diag
            penalty_here[l][0] = (qv - sv[0]) * (qv - sv[0])
                + fminf(pl_, fminf(old[l][0], pd_));
            // r=1: 3rd arg = pt0 = old[0]
            penalty_here[l][1] = (qv - sv[1]) * (qv - sv[1])
                + fminf(penalty_here[l][0], fminf(old[l][1], old[l][0]));
            // r=2: 3rd arg = pt1 = INFINITY (first step special)
            penalty_here[l][2] = (qv - sv[2]) * (qv - sv[2])
                + fminf(penalty_here[l][1], fminf(old[l][2], INFINITY));
            // r >= 3: 3rd arg = old[r-1] (uniform pattern)
            for (int r = 3; r < REGS; r++) {
                penalty_here[l][r] = (qv - sv[r]) * (qv - sv[r])
                    + fminf(penalty_here[l][r - 1],
                            fminf(old[l][r], old[l][r - 1]));
            }
        }
    }

    // shfl_up(query_value, 1); lane 0 = new_query_value[0]; shfl_down(new_query_value, 1)
    // (lines 224-226)
    {
        float tmp[LANES];
        for (int l = 1; l < LANES; l++) tmp[l] = query_value[l - 1];
        tmp[0] = query_value[0];
        for (int l = 0; l < LANES; l++) query_value[l] = tmp[l];
        query_value[0] = new_query_value[0];

        for (int l = 0; l < LANES - 1; l++) tmp[l] = new_query_value[l + 1];
        tmp[LANES - 1] = new_query_value[LANES - 1];
        for (int l = 0; l < LANES; l++) new_query_value[l] = tmp[l];
    }
    int counter = 2;

    // penalty_diag = penalty_left; penalty_left = shfl_up(penalty_here31, 1); lane 0 set INF
    // (lines 229-233)
    for (int l = 0; l < LANES; l++) penalty_diag[l] = penalty_left[l];
    {
        float tmp[LANES];
        for (int l = 1; l < LANES; l++) tmp[l] = penalty_here[l - 1][31];
        tmp[0] = penalty_here[0][31];
        for (int l = 0; l < LANES; l++) penalty_left[l] = tmp[l];
        penalty_left[0] = INFINITY;
    }

    // ----- Main loop (lines 235-336) -----
    for (int k = 3; k < lane + WARP_SIZE - 1; k++) {
        // Save old penalty_here (each lane sees its own register snapshot)
        float old[LANES][REGS];
        for (int l = 0; l < LANES; l++)
            for (int r = 0; r < REGS; r++)
                old[l][r] = penalty_here[l][r];

        // DP step body (uniform pattern: r=0 uses penalty_diag, r>=1 uses old[r-1])
        for (int l = 0; l < LANES; l++) {
            float qv  = query_value[l];
            float pl_ = penalty_left[l];
            float pd_ = penalty_diag[l];
            float* sv = subject_value[l];

            penalty_here[l][0] = (qv - sv[0]) * (qv - sv[0])
                + fminf(pl_, fminf(old[l][0], pd_));
            for (int r = 1; r < REGS; r++) {
                penalty_here[l][r] = (qv - sv[r]) * (qv - sv[r])
                    + fminf(penalty_here[l][r - 1],
                            fminf(old[l][r], old[l][r - 1]));
            }
        }

        // counter % 32 == 0 -> new_query_value = cQuery[i + 2*thid - 1]
        // i = k - l (varies with lane in the kernel because l = thid)
        if (counter % 32 == 0) {
            for (int l = 0; l < LANES; l++) {
                int idx = (k - l) + 2 * l - 1;  // = k + l - 1
                if (idx >= 0 && idx < num_features) {
                    new_query_value[l] = query[idx];
                }
            }
        }

        // shfl_up(query_value, 1); lane 0 = new_query_value[0]; shfl_down(new_query_value, 1)
        {
            float tmp[LANES];
            for (int l = 1; l < LANES; l++) tmp[l] = query_value[l - 1];
            tmp[0] = query_value[0];
            for (int l = 0; l < LANES; l++) query_value[l] = tmp[l];
            query_value[0] = new_query_value[0];

            for (int l = 0; l < LANES - 1; l++) tmp[l] = new_query_value[l + 1];
            tmp[LANES - 1] = new_query_value[LANES - 1];
            for (int l = 0; l < LANES; l++) new_query_value[l] = tmp[l];
        }
        counter++;

        // penalty_diag = penalty_left; penalty_left = shfl_up(ph[31], 1); lane 0 = INF
        for (int l = 0; l < LANES; l++) penalty_diag[l] = penalty_left[l];
        {
            float tmp[LANES];
            for (int l = 1; l < LANES; l++) tmp[l] = penalty_here[l - 1][31];
            tmp[0] = penalty_here[0][31];
            for (int l = 0; l < LANES; l++) penalty_left[l] = tmp[l];
            penalty_left[0] = INFINITY;
        }
    }

    // ----- Epilogue (lines 337-400) — same body as main loop except r=16 -----
    {
        float old[LANES][REGS];
        for (int l = 0; l < LANES; l++)
            for (int r = 0; r < REGS; r++)
                old[l][r] = penalty_here[l][r];

        for (int l = 0; l < LANES; l++) {
            float qv  = query_value[l];
            float pl_ = penalty_left[l];
            float pd_ = penalty_diag[l];
            float* sv = subject_value[l];

            penalty_here[l][0] = (qv - sv[0]) * (qv - sv[0])
                + fminf(pl_, fminf(old[l][0], pd_));
            for (int r = 1; r < REGS; r++) {
                if (r == 16) {
                    // cuDTW epilogue quirk (line 371): penalty_diag instead of pt1 = old[15]
                    penalty_here[l][r] = (qv - sv[r]) * (qv - sv[r])
                        + fminf(penalty_here[l][r - 1],
                                fminf(old[l][r], pd_));
                } else {
                    penalty_here[l][r] = (qv - sv[r]) * (qv - sv[r])
                        + fminf(penalty_here[l][r - 1],
                                fminf(old[l][r], old[l][r - 1]));
                }
            }
        }
    }

    // Result: thid == blockDim.x - 1 (lane 31), penalty_here31 (line 405)
    return penalty_here[LANES - 1][REGS - 1];
}

// ===== Public interface =====

void solution_init(int          num_entries,
                   int          num_features,
                   const float* subjects,
                   const float* query)
{
    g_num_entries  = num_entries;
    g_num_features = num_features;
    g_subjects     = subjects;
    g_query        = query;
}

void solution_compute(int    num_entries,
                      int    num_features,
                      float* distances)
{
    (void)num_features;
    for (int e = 0; e < num_entries; e++) {
        const float* subj = g_subjects + (size_t)e * 1023;
        distances[e] = shfl_FullDTW_1023_cpu(g_query, subj);
    }
}

void solution_free(void)
{
    /* All input data owned by task_io; nothing to free here. */
}
