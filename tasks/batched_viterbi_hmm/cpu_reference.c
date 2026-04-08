#include <float.h>
#include <stdlib.h>

void solution_compute(
    int B,
    int T,
    int H,
    int V,
    const float* log_init,
    const float* log_trans,
    const float* log_emit,
    const int* observations,
    int* out_path
) {
    (void)V;
    if (B <= 0 || T <= 0 || H <= 0) return;

    float* dp_prev = (float*)malloc((size_t)B * (size_t)H * sizeof(float));
    float* dp_curr = (float*)malloc((size_t)B * (size_t)H * sizeof(float));
    int* backptr = (int*)malloc((size_t)B * (size_t)T * (size_t)H * sizeof(int));
    if (!dp_prev || !dp_curr || !backptr) {
        free(dp_prev); free(dp_curr); free(backptr);
        return;
    }

    for (int b = 0; b < B; ++b) {
        int obs0 = observations[(size_t)b * (size_t)T + 0];
        for (int j = 0; j < H; ++j) {
            dp_prev[(size_t)b * (size_t)H + j] = log_init[j] + log_emit[(size_t)j * (size_t)V + obs0];
            backptr[((size_t)b * (size_t)T + 0) * (size_t)H + j] = -1;
        }
    }

    for (int t = 1; t < T; ++t) {
        for (int b = 0; b < B; ++b) {
            int obs = observations[(size_t)b * (size_t)T + t];
            for (int j = 0; j < H; ++j) {
                float best = -FLT_MAX;
                int best_i = 0;
                for (int i = 0; i < H; ++i) {
                    float cand = dp_prev[(size_t)b * (size_t)H + i] + log_trans[(size_t)i * (size_t)H + j];
                    if (cand > best) {
                        best = cand;
                        best_i = i;
                    }
                }
                dp_curr[(size_t)b * (size_t)H + j] = best + log_emit[(size_t)j * (size_t)V + obs];
                backptr[((size_t)b * (size_t)T + t) * (size_t)H + j] = best_i;
            }
        }
        float* tmp = dp_prev;
        dp_prev = dp_curr;
        dp_curr = tmp;
    }

    for (int b = 0; b < B; ++b) {
        float best = -FLT_MAX;
        int best_state = 0;
        for (int j = 0; j < H; ++j) {
            float v = dp_prev[(size_t)b * (size_t)H + j];
            if (v > best) {
                best = v;
                best_state = j;
            }
        }
        out_path[(size_t)b * (size_t)T + (size_t)(T - 1)] = best_state;
        for (int t = T - 1; t >= 1; --t) {
            int prev_state = backptr[((size_t)b * (size_t)T + (size_t)t) * (size_t)H + best_state];
            out_path[(size_t)b * (size_t)T + (size_t)(t - 1)] = prev_state;
            best_state = prev_state;
        }
    }

    free(dp_prev);
    free(dp_curr);
    free(backptr);
}

void solution_free(void) {}
