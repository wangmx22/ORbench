#include <stdlib.h>

void solution_compute(
    int N,
    int M,
    int iters,
    int damping_x1e6,
    const int* row_ptr_in,
    const int* col_ind_in,
    const float* inv_out_deg,
    float* out
) {
    (void)M;
    const float d = (float)damping_x1e6 / 1000000.0f;
    const float base = (1.0f - d) / (float)N;

    float* cur = (float*)malloc((size_t)N * sizeof(float));
    float* nxt = (float*)malloc((size_t)N * sizeof(float));
    if (!cur || !nxt) {
        if (cur) free(cur);
        if (nxt) free(nxt);
        return;
    }

    const float init = 1.0f / (float)N;
    for (int i = 0; i < N; ++i) cur[i] = init;

    for (int it = 0; it < iters; ++it) {
        for (int v = 0; v < N; ++v) {
            float acc = 0.0f;
            for (int e = row_ptr_in[v]; e < row_ptr_in[v + 1]; ++e) {
                int u = col_ind_in[e];
                acc += cur[u] * inv_out_deg[u];
            }
            nxt[v] = base + d * acc;
        }
        float* tmp = cur;
        cur = nxt;
        nxt = tmp;
    }

    for (int i = 0; i < N; ++i) out[i] = cur[i];
    free(cur);
    free(nxt);
}

void solution_free(void) {
}
